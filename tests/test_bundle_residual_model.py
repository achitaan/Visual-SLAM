"""Opt-in left-u/v/disparity bundle residual model tests."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import MapState, MappingKeyframe, Observation


MATRIX = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE = 0.2
OFFSET = 6.5


def _pose(x):
    result = np.eye(4)
    result[0, 3] = x
    return result


def _state(*, metric=True, truth_last=1.0, initial_last=1.0, landmark_count=36):
    state = MapState(metric=metric)
    initial_poses = [_pose(0.0), _pose(0.5), _pose(initial_last)]
    truth_poses = [_pose(0.0), _pose(0.5), _pose(truth_last)]
    for ident, pose in enumerate(initial_poses):
        state.keyframes[ident] = MappingKeyframe(
            ident, ident, pose.copy(), np.empty((0, 2)),
            np.empty((0, 128)), np.empty(0, int))
    rng = np.random.default_rng(771)
    world = rng.uniform([-1.8, -1.2, 6.0], [1.8, 1.2, 13.0], (landmark_count, 3))
    for ident, point in enumerate(world):
        observations = {}
        for camera_id, pose in enumerate(truth_poses):
            pixel, depth = project(point[None], pose, MATRIX)
            right = None
            if metric and (ident + camera_id) % 2 == 0:
                right = float(right_pixel(pixel[0, 0], depth[0], MATRIX[0, 0],
                                          BASELINE, OFFSET))
            observations[camera_id] = Observation(pixel[0].copy(), right)
        state.add_landmark(point, np.ones(128), 0, observations)
    return state


def _residual_oracle(vector, state, model, *, selected_count=18, window=3):
    selected_ids = list(state.landmarks)[:selected_count]
    poses = {ident: frame.pose.copy() for ident, frame in state.keyframes.items()}
    free_ids = (1, 2)
    for index, ident in enumerate(free_ids):
        offset = 6 * index
        poses[ident][:3, :3] = Rotation.from_rotvec(vector[offset:offset + 3]).as_matrix()
        poses[ident][:3, 3] = vector[offset + 3:offset + 6]
    point_offset = 6 * len(free_ids)
    points = {
        ident: vector[point_offset + 3 * i:point_offset + 3 * i + 3]
        for i, ident in enumerate(selected_ids)
    }
    result = []

    def append(point, camera_id, observation):
        camera = poses[camera_id][:3, :3].T @ (point - poses[camera_id][:3, 3])
        z = float(camera[2])
        if z <= 0:
            result.extend([1e4] * (3 if observation.right_u is not None else 2))
            return
        projected = (MATRIX @ camera)[:2] / (MATRIX @ camera)[2]
        left = np.clip(projected - observation.pixel, -1e4, 1e4)
        result.extend(left.tolist())
        if observation.right_u is not None:
            if model == "left_right":
                pred_right = (projected[0] - MATRIX[0, 0] * BASELINE / z - OFFSET)
                result.append(float(pred_right - observation.right_u))
            else:
                pred_d = MATRIX[0, 0] * BASELINE / z + OFFSET
                measured_d = observation.pixel[0] - observation.right_u
                result.append(float(pred_d - measured_d))

    for ident in selected_ids:
        landmark = state.landmarks[ident]
        for camera_id, observation in landmark.observations.items():
            if camera_id in poses:
                append(points[ident], camera_id, observation)
    selected = set(selected_ids)
    for ident, landmark in state.landmarks.items():
        if ident in selected or len(landmark.observations) < 2:
            continue
        for camera_id, observation in landmark.observations.items():
            if camera_id in free_ids:
                append(landmark.position, camera_id, observation)
    return np.asarray(result, dtype=float)


def _huber_cost(residual):
    absolute = np.abs(residual)
    return float(np.sum(np.where(absolute <= 2.0, .5 * residual**2,
                                 2.0 * (absolute - 1.0))))


@pytest.mark.parametrize("mode", ["left_right", "left_disparity"])
def test_selected_and_held_rows_match_independent_oracle_in_both_pose_paths(
        monkeypatch, mode):
    captures = []

    def inspect(residual, initial, **kwargs):
        shifted = initial.copy()
        shifted[:6] += np.array([.003, -.002, .001, .012, -.008, .005])
        shifted[-3:] += np.array([.01, -.02, .03])
        captures.append((residual(initial).copy(), residual(shifted).copy(),
                         kwargs["jac_sparsity"].toarray().copy()))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    reports = []
    for optimized in (True, False):
        state = _state()
        report = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=18,
            disparity_offset=OFFSET, optimized=optimized,
            stereo_residuals=mode)
        reports.append(report)
    assert len(captures) == 2
    for path_index, (initial_rows, shifted_rows, pattern) in enumerate(captures):
        state = _state()
        initial = np.r_[
            *[np.r_[Rotation.from_matrix(state.keyframes[k].pose[:3, :3]).as_rotvec(),
                    state.keyframes[k].pose[:3, 3]] for k in (1, 2)],
            np.asarray([state.landmarks[i].position for i in range(18)]).ravel(),
        ]
        shifted = initial.copy()
        shifted[:6] += np.array([.003, -.002, .001, .012, -.008, .005])
        shifted[-3:] += np.array([.01, -.02, .03])
        expected_initial = _residual_oracle(initial, state, mode)
        expected_shifted = _residual_oracle(shifted, state, mode)
        np.testing.assert_allclose(initial_rows, expected_initial, atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(shifted_rows, expected_shifted, atol=1e-9, rtol=1e-12)
        if path_index == 0:
            reference_pattern = pattern
        else:
            np.testing.assert_array_equal(pattern, reference_pattern)
    np.testing.assert_allclose(captures[0][0], captures[1][0], atol=1e-12)
    np.testing.assert_allclose(captures[0][1], captures[1][1], atol=1e-12)
    for report in reports:
        assert report["affected_initial_cost"] == pytest.approx(
            _huber_cost(captures[0][0]), abs=1e-8)
    if mode == "left_disparity":
        model_report = reports[0]["stereo_residual_model"]
        assert model_report["model"] == "uniform_independent_u_v_disparity_1px_assumption"
        assert model_report["covariance_claim"] is False
        assert model_report["source_specific_noise_model"] is False
        assert model_report["source_provenance"] == "unavailable"
        assert model_report["active_stereo_observations"] > 0
        assert model_report["active_mono_observations"] > 0
        assert model_report["active_stereo_rows"] == 3 * model_report["active_stereo_observations"]
        assert model_report["active_mono_rows"] == 2 * model_report["active_mono_observations"]
    else:
        assert all("stereo_residual_model" not in report for report in reports)


def test_equal_horizontal_left_and_right_shift_cancels_only_in_disparity_row(monkeypatch):
    state = _state()
    observation = state.landmarks[0].observations[0]
    observation.pixel[0] += 4.0
    observation.right_u += 4.0
    captured = []

    def inspect(residual, initial, **_kwargs):
        captured.append(residual(initial).copy())
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    local_bundle_adjustment(state, MATRIX, BASELINE, window=3, max_landmarks=18,
                            disparity_offset=OFFSET, stereo_residuals="left_disparity")
    assert captured[0][0] == pytest.approx(-4.0)
    assert captured[0][2] == pytest.approx(0.0, abs=1e-10)


def test_disparity_depth_derivative_offset_and_gaussian_coordinate_identity(monkeypatch):
    state = _state()
    captured = []

    def inspect(residual, initial, **_kwargs):
        captured.append((residual, initial.copy()))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    local_bundle_adjustment(state, MATRIX, BASELINE, window=3, max_landmarks=18,
                            disparity_offset=OFFSET, stereo_residuals="left_disparity")
    residual, initial = captured[0]
    z = state.landmarks[0].position[2]
    point_z_index = 12 + 2
    epsilon = 1e-5
    lower, upper = initial.copy(), initial.copy()
    lower[point_z_index] -= epsilon
    upper[point_z_index] += epsilon
    numerical = (residual(upper)[2] - residual(lower)[2]) / (2 * epsilon)
    expected = -MATRIX[0, 0] * BASELINE / z**2
    assert numerical == pytest.approx(expected, rel=1e-7)

    # In Gaussian coordinates, e_d=e_u-e_right is only a linear transform of
    # independent left/right pixel errors. Componentwise Huber is intentionally
    # applied in the declared disparity basis rather than claimed equivalent.
    e_u, e_right, e_v = 1.25, -0.75, 0.5
    original = np.array([e_u, e_v, e_right])
    covariance = np.array([[1.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 0.0, 2.0]])
    transformed = np.array([e_u, e_v, e_u - e_right])
    whitened = np.array([e_u, e_v, e_right - e_u])
    assert transformed @ transformed == pytest.approx(
        original @ np.linalg.inv(covariance) @ original)
    assert whitened @ whitened == pytest.approx(transformed @ transformed)
    assert _huber_cost(np.array([e_u, e_v, e_u - e_right])) != pytest.approx(
        _huber_cost(np.array([e_u, e_v, e_right])))


@pytest.mark.parametrize("baseline", [0.0, -0.2, np.nan, np.inf, 0.2 + 1j])
def test_disparity_mode_rejects_invalid_baseline_before_solver(monkeypatch, baseline):
    state = _state()
    monkeypatch.setattr(bundle, "least_squares",
                        lambda *_args, **_kwargs: pytest.fail("solver must not run"))
    with pytest.raises(ValueError, match="positive stereo baseline"):
        local_bundle_adjustment(state, MATRIX, baseline, window=3,
                                disparity_offset=OFFSET,
                                stereo_residuals="left_disparity")


def test_unknown_or_nonmetric_disparity_mode_is_rejected_before_solver(monkeypatch):
    monkeypatch.setattr(bundle, "least_squares",
                        lambda *_args, **_kwargs: pytest.fail("solver must not run"))
    with pytest.raises(ValueError, match="stereo_residuals"):
        local_bundle_adjustment(_state(), MATRIX, BASELINE,
                                stereo_residuals="sensor_xyz")
    with pytest.raises(ValueError, match="metric stereo map"):
        local_bundle_adjustment(_state(metric=False), MATRIX, BASELINE,
                                stereo_residuals="left_disparity")


@pytest.mark.parametrize("bad_matrix", [
    MATRIX[:, :2],
    np.array([[250., 0., 320.], [0., np.nan, 240.], [0., 0., 1.]]),
    np.array([[250.+1j, 0., 320.], [0., 250., 240.], [0., 0., 1.]]),
    np.array([[250., 0., 320.], [0., 250., 240.], [0.1, 0., 1.]]),
])
def test_disparity_mode_rejects_invalid_calibration_before_solver(monkeypatch, bad_matrix):
    monkeypatch.setattr(bundle, "least_squares",
                        lambda *_args, **_kwargs: pytest.fail("solver must not run"))
    with pytest.raises(ValueError, match="calibration|fx/fy"):
        local_bundle_adjustment(_state(), bad_matrix, BASELINE, window=3,
                                disparity_offset=OFFSET,
                                stereo_residuals="left_disparity")


@pytest.mark.parametrize("bad_offset", [np.nan, np.inf, 1.0 + 1j])
def test_disparity_mode_rejects_invalid_offset_before_solver(monkeypatch, bad_offset):
    monkeypatch.setattr(bundle, "least_squares",
                        lambda *_args, **_kwargs: pytest.fail("solver must not run"))
    with pytest.raises(ValueError, match="disparity offset"):
        local_bundle_adjustment(_state(), MATRIX, BASELINE, window=3,
                                disparity_offset=bad_offset,
                                stereo_residuals="left_disparity")


def test_behind_camera_disparity_rows_cannot_cancel_large_penalties(monkeypatch):
    state = _state()
    captured = []

    def inspect(residual, initial, **_kwargs):
        invalid = initial.copy()
        invalid[12 + 2] = -1.0
        captured.append(residual(invalid).copy())
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    local_bundle_adjustment(state, MATRIX, BASELINE, window=3, max_landmarks=18,
                            disparity_offset=OFFSET, stereo_residuals="left_disparity")
    # Landmark 0 has stereo, left-only, and stereo observations in cameras 0–2.
    assert np.array_equal(captured[0][:8], np.full(8, 1e4))


def test_actual_scipy_disparity_solve_reduces_objective_and_preserves_gauge():
    state = _state(truth_last=1.06, initial_last=1.0, landmark_count=72)
    fixed_pose = state.keyframes[0].pose.copy()
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=48,
        disparity_offset=OFFSET, stereo_residuals="left_disparity")
    assert report["applied"] is True
    assert report["final_cost"] < report["initial_cost"]
    assert report["affected_final_cost"] < report["affected_initial_cost"]
    assert np.isfinite([report["initial_cost"], report["final_cost"]]).all()
    np.testing.assert_array_equal(state.keyframes[0].pose, fixed_pose)
    assert all(np.isfinite(frame.pose).all() for frame in state.keyframes.values())


def test_default_left_right_remains_the_default_mode():
    state = _state()
    report = local_bundle_adjustment(state, MATRIX, BASELINE, window=3,
                                     max_landmarks=18, disparity_offset=OFFSET)
    assert "stereo_residual_model" not in report
