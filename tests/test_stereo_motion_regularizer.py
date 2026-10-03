import numpy as np
import pytest

import mapping_geometry as geometry
from stereo_pose_arbitration import SupportedStereoFrame
from stereo_motion_regularizer import (
    _build_information,
    _projection_jacobian,
    _skew,
    build_stereo_motion_regularizer,
    se3_log,
    stereo_motion_residual,
)


def _skew_local(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _exp_se3(xi):
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    theta = np.linalg.norm(phi)
    w = _skew_local(phi)
    if theta < 1e-8:
        rotation = np.eye(3) + w + .5 * w @ w
        v = np.eye(3) + .5 * w + (1. / 6.) * w @ w
    else:
        rotation = np.eye(3) + np.sin(theta) / theta * w + (1. - np.cos(theta)) / theta**2 * (w @ w)
        v = np.eye(3) + (1. - np.cos(theta)) / theta**2 * w + (theta - np.sin(theta)) / theta**3 * (w @ w)
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = v @ rho
    return out


def _project(point, matrix, baseline, offset):
    x, y, z = np.asarray(point, dtype=float)
    hom = matrix @ np.array([x, y, z])
    u, v = hom[0] / z, hom[1] / z
    right_u = u - matrix[0, 0] * baseline / z - offset
    return np.array([u, v, right_u])


def _fixture(count=36, unit_scale=1., *, collinear=False, measurement=None):
    rng = np.random.default_rng(281)
    if collinear:
        source_points = np.c_[np.linspace(-1., 1., count), np.zeros(count), np.full(count, 8.)]
    else:
        source_points = np.c_[rng.uniform(-1.4, 1.4, count),
                              rng.uniform(-.8, .8, count),
                              rng.uniform(7., 14., count)]
    source_points *= unit_scale
    matrix = np.array([[870., 4.5, 960.], [0., 860., 540.], [0., 0., 1.]])
    baseline, offset = .54 * unit_scale, .65
    if measurement is None:
        measurement = _exp_se3(unit_scale * np.array([.11, -.04, .08, 0., 0., 0.])
                               + np.array([0., 0., 0., .045, -.07, .03]))
    rotation, translation = measurement[:3, :3], measurement[:3, 3]
    target_points = (source_points - translation) @ rotation
    source_observations = np.array([_project(p, matrix, baseline, offset) for p in source_points])
    target_observations = np.array([_project(p, matrix, baseline, offset) for p in target_points])
    source_pixels = source_observations[:, :2].astype(np.float32)
    target_pixels = target_observations[:, :2].astype(np.float32)
    source_right = source_observations[:, 2].astype(np.float32)
    target_right = target_observations[:, 2].astype(np.float32)
    descriptors = rng.integers(0, 255, size=(count, 32), dtype=np.uint8)
    source = SupportedStereoFrame(
        source_pixels, descriptors, source_points, source_right,
        np.full(count, -1, dtype=int), 17, (1920, 1080), "calib:test:v1")
    target = SupportedStereoFrame(
        target_pixels, descriptors, target_points, target_right,
        np.full(count, -1, dtype=int), 18, (1920, 1080), "calib:test:v1")
    pairs = np.c_[np.arange(count), np.arange(count)].astype(np.int64)
    return {
        "source": source, "target": target, "measurement": measurement,
        "matrix": matrix, "baseline": baseline, "offset": offset,
        "pairs": pairs, "source_obs": source_observations,
        "target_obs": target_observations,
    }


def _make_factor(data, **overrides):
    args = dict(
        training_pairs=data["pairs"],
        forward_inlier_pairs=data["pairs"],
        reverse_inlier_pairs=data["pairs"],
        training_source_pixels=data["source"].pixels[data["pairs"][:, 0]],
        training_target_pixels=data["target"].pixels[data["pairs"][:, 1]],
        training_source_points=data["source"].points[data["pairs"][:, 0]],
        training_target_points=data["target"].points[data["pairs"][:, 1]],
        matrix=data["matrix"], baseline=data["baseline"],
        disparity_offset=data["offset"], source_epoch=(9, 4),
        source_status="tracking", target_status="tracking",
        fit_source="raw_supported_reference_retry", fit_depth_policy="supported_raw",
        role="selected_reference",
    )
    args.update(overrides)
    return build_stereo_motion_regularizer(
        data["source"], data["target"], data["measurement"], **args)


def test_factor_is_immutable_honest_and_ranked_in_dimensionless_coordinates():
    data = _fixture()
    factor, reason = _make_factor(data)
    assert reason is None
    assert factor is not None
    assert factor.model == "correlated_pixel_motion_regularizer"
    assert factor.covariance_claim is False
    assert factor.intentionally_reuses_sensor_evidence is True
    assert factor.holdout_rows_used is False
    assert factor.ownership_certificate == "none"
    assert factor.source_frame == 17 and factor.target_frame == 18
    assert factor.source_epoch == (9, 4)
    assert factor.rank == 6 and np.isfinite(factor.condition_number)
    assert np.all(np.linalg.eigvalsh(factor.information) > 0.)
    assert np.allclose(factor.sqrt_information.T @ factor.sqrt_information,
                       factor.information, rtol=2e-9, atol=2e-9)
    for name in ("measurement", "sqrt_information", "information", "training_pairs",
                 "source_pixels", "source_points"):
        assert not getattr(factor, name).flags.writeable
        with pytest.raises(ValueError):
            getattr(factor, name).setflags(write=True)
    assert factor.training_pairs.dtype == np.int64
    assert factor.information_pairs.dtype == np.int64


def test_projection_jacobians_and_schur_match_finite_differences_and_full_normal():
    data = _fixture(count=24)
    factor, reason = _make_factor(data)
    assert reason is None
    matrix, baseline, offset = data["matrix"], data["baseline"], data["offset"]
    measurement = data["measurement"]
    source_index, target_index = factor.information_pairs[0]
    point = data["source"].points[source_index].copy()
    target_point = measurement[:3, :3].T @ (point - measurement[:3, 3])
    expected_pose_jacobian = _projection_jacobian(target_point, matrix, baseline) @ np.c_[
        -np.eye(3), _skew(target_point)]
    expected_point_jacobian = np.vstack((
        _projection_jacobian(point, matrix, baseline),
        _projection_jacobian(target_point, matrix, baseline) @ measurement[:3, :3].T,
    ))

    def residual(pose_delta, point_delta):
        perturbed_pose = measurement @ _exp_se3(pose_delta)
        perturbed_point = point + point_delta
        camera = perturbed_pose[:3, :3].T @ (perturbed_point - perturbed_pose[:3, 3])
        return np.r_[_project(perturbed_point, matrix, baseline, offset),
                     _project(camera, matrix, baseline, offset)]

    step = 1e-7
    numeric_pose = np.column_stack([
        (residual(np.eye(6)[i] * step, np.zeros(3))
         - residual(-np.eye(6)[i] * step, np.zeros(3))) / (2. * step)
        for i in range(6)
    ])[3:]
    numeric_point = np.column_stack([
        (residual(np.zeros(6), np.eye(3)[i] * step)
         - residual(np.zeros(6), -np.eye(3)[i] * step)) / (2. * step)
        for i in range(3)
    ])
    assert np.allclose(numeric_pose, expected_pose_jacobian, rtol=2e-5, atol=2e-5)
    assert np.allclose(numeric_point, expected_point_jacobian, rtol=2e-5, atol=2e-5)

    covariance = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 2.]])
    endpoint_whitener = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(3))
    block_whitener = np.zeros((6, 6))
    block_whitener[:3, :3] = endpoint_whitener
    block_whitener[3:, 3:] = endpoint_whitener
    information_from_schur = np.zeros((6, 6))
    local_jacobians = []
    for source_row, target_row in factor.information_pairs:
        source_point = data["source"].points[source_row]

        def pair_residual(pose_delta, point_delta):
            pose = measurement @ _exp_se3(pose_delta)
            value = source_point + point_delta
            camera = pose[:3, :3].T @ (value - pose[:3, 3])
            return np.r_[_project(value, matrix, baseline, offset),
                         _project(camera, matrix, baseline, offset)]

        jacobian = np.column_stack([
            (pair_residual(np.eye(6)[i] * step, np.zeros(3))
             - pair_residual(-np.eye(6)[i] * step, np.zeros(3))) / (2. * step)
            for i in range(6)
        ] + [
            (pair_residual(np.zeros(6), np.eye(3)[i] * step)
             - pair_residual(np.zeros(6), -np.eye(3)[i] * step)) / (2. * step)
            for i in range(3)
        ])
        jacobian = block_whitener @ jacobian
        local_jacobians.append(jacobian)
        pose_jacobian, point_jacobian = jacobian[:, :6], jacobian[:, 6:]
        information_from_schur += (
            pose_jacobian.T @ pose_jacobian
            - pose_jacobian.T @ point_jacobian
            @ np.linalg.solve(point_jacobian.T @ point_jacobian,
                              point_jacobian.T @ pose_jacobian)
        )
    full_jacobian = np.zeros((6 * len(local_jacobians), 6 + 3 * len(local_jacobians)))
    for index, jacobian in enumerate(local_jacobians):
        row_slice = slice(6 * index, 6 * (index + 1))
        point_slice = slice(6 + 3 * index, 9 + 3 * index)
        full_jacobian[row_slice, :6] = jacobian[:, :6]
        full_jacobian[row_slice, point_slice] = jacobian[:, 6:]
    full_normal = full_jacobian.T @ full_jacobian
    block_schur = (full_normal[:6, :6]
                   - full_normal[:6, 6:] @ np.linalg.solve(
                       full_normal[6:, 6:], full_normal[6:, :6]))
    assert np.allclose(information_from_schur, block_schur, rtol=2e-6, atol=2e-6)
    assert np.allclose(factor.information, block_schur, rtol=2e-5, atol=2e-5)


def test_correlated_right_coordinate_whitening_equals_independent_uv_disparity():
    rng = np.random.default_rng(28)
    covariance = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 2.]])
    whitener = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(3))
    residual = rng.normal(size=(40, 3))
    independent_uv_disparity = residual.copy()
    independent_uv_disparity[:, 2] = residual[:, 0] - residual[:, 2]
    assert np.allclose(np.sum((residual @ whitener.T) ** 2),
                       np.sum(independent_uv_disparity ** 2), rtol=1e-12, atol=1e-12)


def test_rank_deficient_motion_and_point_nuisance_fail_closed():
    line = _fixture(count=20, collinear=True)
    factor, reason = _make_factor(line)
    assert factor is None
    assert reason == "rank_deficient_motion_information"

    point = np.array([[0., 0., 10.]])
    with pytest.raises(ValueError, match="point_jacobian_rank_deficient"):
        _build_information(point, point, line["matrix"], 0., line["offset"], np.eye(4))


def test_physical_pixel_aliases_and_mismatched_fit_provenance_skip_factor():
    data = _fixture()
    duplicate_pixels = data["source"].pixels.copy()
    duplicate_pixels[1] = duplicate_pixels[0]
    data["source"] = SupportedStereoFrame(
        duplicate_pixels, data["source"].descriptors, data["source"].points,
        data["source"].right_u, data["source"].landmark_ids, 17,
        data["source"].image_size, data["source"].calibration_identity)
    factor, reason = _make_factor(data)
    assert factor is None and reason == "duplicate_physical_training_pixel"

    data = _fixture()
    altered_points = data["source"].points[data["pairs"][:, 0]].copy()
    altered_points[0, 0] += .01
    factor, reason = _make_factor(data, training_source_points=altered_points)
    assert factor is None and reason == "fit_geometry_differs_from_raw_supported_rows"
    altered_pixels = data["target"].pixels[data["pairs"][:, 1]].copy()
    altered_pixels[0, 0] += 1.
    factor, reason = _make_factor(data, training_target_pixels=altered_pixels)
    assert factor is None and reason == "fit_geometry_differs_from_raw_supported_rows"


@pytest.mark.parametrize(("endpoint", "frame"), [("source", 17.5), ("target", True)])
def test_builder_rejects_malformed_endpoint_frame_ids(endpoint, frame):
    from dataclasses import replace

    data = _fixture()
    data[endpoint] = replace(data[endpoint], frame=frame)
    factor, reason = _make_factor(data)
    assert factor is None
    assert f"{endpoint}_frame" in reason


def test_regularizer_is_finite_not_a_covariance_and_unit_scaling_preserves_objective():
    base = _fixture()
    scaled = _fixture(unit_scale=1000.)
    factor_a, reason_a = _make_factor(base)
    factor_b, reason_b = _make_factor(scaled)
    assert reason_a is None and reason_b is None
    assert factor_a.covariance_claim is False and factor_b.covariance_claim is False
    assert np.all(np.isfinite(factor_a.sqrt_information))
    assert np.all(np.isfinite(factor_b.sqrt_information))
    scale = np.diag([1000.] * 3 + [1.] * 3)
    assert np.allclose(scale.T @ factor_b.information @ scale,
                       factor_a.information, rtol=2e-8, atol=2e-8)
    delta = np.array([.002, -.001, .004, .003, -.002, .001])
    delta_scaled = scale @ delta
    pose_a = base["measurement"] @ _exp_se3(delta)
    pose_b = scaled["measurement"] @ _exp_se3(delta_scaled)
    residual_a = stereo_motion_residual(factor_a, np.eye(4), pose_a)
    residual_b = stereo_motion_residual(factor_b, np.eye(4), pose_b)
    assert np.allclose(residual_a, residual_b, rtol=2e-7, atol=2e-7)


def test_se3_log_uses_v_inverse_with_coupled_translation_and_stable_small_angles():
    tangent = np.array([.4, -.2, .13, .31, -.22, .17])
    transform = _exp_se3(tangent)
    logged = se3_log(transform)
    assert np.allclose(logged, tangent, rtol=1e-11, atol=1e-11)
    assert not np.allclose(logged[:3], transform[:3, 3])
    assert np.allclose(se3_log(np.linalg.inv(transform)), -tangent, rtol=1e-11, atol=1e-11)

    small = np.array([.4, -.2, .13, 1e-9, -2e-9, 3e-9])
    assert np.allclose(se3_log(_exp_se3(small)), small, rtol=2e-8, atol=2e-9)


def test_edge_residual_is_invariant_to_common_left_correction_but_not_one_sided():
    data = _fixture()
    factor, reason = _make_factor(data)
    assert reason is None
    source_pose = _exp_se3(np.array([.2, -.1, .03, .02, .01, -.04]))
    target_pose = source_pose @ data["measurement"] @ _exp_se3(
        np.array([.004, -.002, .001, .002, -.001, .003]))
    residual = stereo_motion_residual(factor, source_pose, target_pose)
    common = _exp_se3(np.array([-.4, .7, .1, -.08, .02, .11]))
    corrected = stereo_motion_residual(factor, common @ source_pose, common @ target_pose)
    assert np.allclose(residual, corrected, rtol=2e-9, atol=2e-9)
    one_sided = stereo_motion_residual(factor, source_pose, common @ target_pose)
    assert not np.allclose(residual, one_sided)


def test_estimator_training_capture_is_opt_in_and_returns_exact_directional_row_pairs(monkeypatch):
    count = 6
    data = _fixture(count=count)
    source = type("Frame", (), {
        "descriptors": data["source"].descriptors,
        "points": data["source"].points,
        "pixels": data["source"].pixels,
        "image_size": data["source"].image_size,
    })()
    target = type("Frame", (), {
        "descriptors": data["target"].descriptors,
        "points": data["target"].points,
        "pixels": data["target"].pixels,
        "image_size": data["target"].image_size,
    })()
    matches = data["pairs"]
    matcher_calls = []
    estimate_calls = []

    def estimated_pose(*args, **kwargs):
        estimate_calls.append(1)
        pose = (data["measurement"] if len(estimate_calls) % 2 == 1
                else np.linalg.inv(data["measurement"]))
        return pose.copy(), np.array([0, 2, 4]), .25

    monkeypatch.setattr(geometry, "estimate_pose", estimated_pose)
    monkeypatch.setattr(geometry, "refine_bidirectional_stereo", lambda pose, *a: (
        pose.copy(), {"refined": True}))

    def matcher(first, second):
        matcher_calls.append(1)
        return matches.copy()

    default = geometry.estimate_stereo_reference(
        source, target, data["matrix"], min_inliers=3, matcher=matcher)
    captured = geometry.estimate_stereo_reference(
        source, target, data["matrix"], min_inliers=3, matcher=matcher,
        capture_training=True)
    assert "training" not in default
    for key in default:
        if isinstance(default[key], np.ndarray):
            assert np.array_equal(default[key], captured[key])
        else:
            assert default[key] == captured[key]
    assert len(matcher_calls) == 2
    training = captured["training"]
    assert np.array_equal(training["training_pairs"], matches)
    assert np.array_equal(training["forward_inlier_pairs"], matches[[0, 2, 4]])
    assert np.array_equal(training["reverse_inlier_pairs"], matches[[0, 2, 4]])
    assert np.array_equal(training["training_source_pixels"], source.pixels)
    assert np.array_equal(training["training_target_pixels"], target.pixels)
    assert np.array_equal(training["training_source_points"], source.points)
    assert np.array_equal(training["training_target_points"], target.points)


def test_adjacent_factor_semantics_disclose_reused_evidence_and_not_independence():
    first = _fixture()
    factor, reason = _make_factor(first, role="parallel_map_agreeing")
    assert reason is None
    assert factor.intentionally_reuses_sensor_evidence
    assert factor.covariance_claim is False
    assert factor.ownership_certificate == "none"
    # The shared frame observations remain explicit endpoints in the next
    # edge; the representation makes reuse visible instead of claiming that
    # adjacent factors are independent likelihoods.
    assert factor.target_frame == 18
    assert factor.target_pixels.shape[0] == len(factor.information_pairs)


def test_direct_factor_construction_rejects_complex_and_inconsistent_records():
    from dataclasses import replace

    data = _fixture()
    factor, reason = _make_factor(data)
    assert reason is None
    with pytest.raises(ValueError, match="real numeric"):
        replace(factor, measurement=factor.measurement.astype(complex) + 1j)
    with pytest.raises(ValueError, match="inconsistent with information"):
        replace(factor, sqrt_information=np.eye(6))
    with pytest.raises(ValueError, match="cannot claim independence or covariance"):
        replace(factor, covariance_claim=True)

