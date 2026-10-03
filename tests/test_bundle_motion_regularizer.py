"""Opt-in correlated stereo-motion factors in local bundle adjustment."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import MapState, MappingKeyframe, Observation
from stereo_pose_arbitration import SupportedStereoFrame
from stereo_motion_regularizer import build_stereo_motion_regularizer, stereo_motion_residual


MATRIX = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
BASELINE = .2
OFFSET = 6.5


def _pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def _build_factor():
    rng = np.random.default_rng(148)
    source_points = np.c_[
        rng.uniform(-1.4, 1.4, 48),
        rng.uniform(-.8, .8, 48),
        rng.uniform(7., 14., 48),
    ]
    measurement = _pose(.5)
    target_points = source_points - measurement[:3, 3]
    source_pixels, source_depth = project(source_points, np.eye(4), MATRIX)
    target_pixels, target_depth = project(target_points, np.eye(4), MATRIX)
    source_right = right_pixel(source_pixels[:, 0], source_depth,
                               MATRIX[0, 0], BASELINE, OFFSET)
    target_right = right_pixel(target_pixels[:, 0], target_depth,
                               MATRIX[0, 0], BASELINE, OFFSET)
    descriptors = rng.integers(0, 255, (48, 32), dtype=np.uint8)
    source = SupportedStereoFrame(
        source_pixels.astype(np.float32), descriptors, source_points, source_right,
        np.full(48, -1, int), 2, (640, 480), "ba-calibration-v1")
    target = SupportedStereoFrame(
        target_pixels.astype(np.float32), descriptors, target_points, target_right,
        np.full(48, -1, int), 4, (640, 480), "ba-calibration-v1")
    pairs = np.c_[np.arange(48), np.arange(48)].astype(np.int64)
    factor, reason = build_stereo_motion_regularizer(
        source, target, measurement,
        training_pairs=pairs,
        forward_inlier_pairs=pairs,
        reverse_inlier_pairs=pairs,
        training_source_pixels=source.pixels[pairs[:, 0]],
        training_target_pixels=target.pixels[pairs[:, 1]],
        training_source_points=source.points[pairs[:, 0]],
        training_target_points=target.points[pairs[:, 1]],
        matrix=MATRIX, baseline=BASELINE, disparity_offset=OFFSET,
        source_epoch=(0, 0), source_status="tracking", target_status="tracking",
        fit_source="reserved_supported_training_rows",
        fit_depth_policy="supported_raw", role="selected_reference",
    )
    assert reason is None, reason
    return factor


def _state(*, truth_last=1.0, initial_last=1.1, landmark_count=48):
    state = MapState(metric=True)
    keyframe_poses = [_pose(0.), _pose(.5), _pose(initial_last)]
    for ident, value in enumerate(keyframe_poses):
        state.keyframes[ident] = MappingKeyframe(
            ident, 2 * ident, value.copy(), np.empty((0, 2)),
            np.empty((0, 128)), np.empty(0, int))
    actual_poses = [_pose(0.), _pose(.25), _pose(.5), _pose(.75), _pose(initial_last)]
    anchors = [0, 0, 1, 1, 2]
    for value, anchor in zip(actual_poses, anchors):
        state.record(value, "tracking", anchor)

    rng = np.random.default_rng(921)
    world = rng.uniform([-1.8, -1.2, 6.], [1.8, 1.2, 13.], (landmark_count, 3))
    observation_poses = [_pose(0.), _pose(.5), _pose(truth_last)]
    for point in world:
        observations = {}
        for camera_id, camera_pose in enumerate(observation_poses):
            pixel, depth = project(point[None], camera_pose, MATRIX)
            right = float(right_pixel(pixel[0, 0], depth[0], MATRIX[0, 0],
                                      BASELINE, OFFSET))
            observations[camera_id] = Observation(pixel[0], right)
        state.add_landmark(point, np.ones(128), 0, observations)

    factor = _build_factor()
    measurement = factor.measurement.copy()
    state.add_stereo_motion(2, 4, measurement)
    attached, reason = state.add_stereo_motion_regularizer(
        2, 4, factor, calibration_identity=factor.calibration_identity,
        source_pose=state.poses[2].copy(), source_geometry_revision=0)
    assert attached, reason
    return state, factor


def test_delayed_actual_frame_edge_adds_six_rows_and_declares_both_anchor_blocks(monkeypatch):
    state, factor = _state()
    original_poses = np.asarray(state.poses).copy()
    inspected = []

    def inspect(residual, initial, **kwargs):
        values = residual(initial)
        pattern = kwargs["jac_sparsity"].toarray().astype(bool)
        # Frames 2 and 4 are not the same camera sample as their keyframes;
        # the source and target frame poses propagate through free anchors 1,2.
        expected = stereo_motion_residual(factor, original_poses[2], original_poses[4])
        np.testing.assert_allclose(values[-6:], expected, rtol=1e-10, atol=1e-10)
        assert np.linalg.norm(values[-6:]) > 0.
        assert pattern[-6:, :12].all()
        assert not pattern[-6:, 12:].any()
        # Numeric checks cover all six variables in both endpoint anchors.
        for column in range(12):
            low, high = initial.copy(), initial.copy()
            low[column] -= 1e-6
            high[column] += 1e-6
            derivative = (residual(high) - residual(low)) / 2e-6
            assert np.max(np.abs(derivative[-6:][~pattern[-6:, column]]), initial=0.) < 1e-5
        inspected.append((values, pattern, kwargs))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)
    assert inspected
    regularizer = report["stereo_motion_regularizer"]
    assert regularizer["active_edges"] == [[2, 4]]
    assert regularizer["covariance_claim"] is False
    assert regularizer["intentionally_reuses_sensor_evidence"] is True
    assert regularizer["training_row_ids"]["2:4"]["information_rows"]
    assert regularizer["augmented_initial_huber_cost"] == pytest.approx(
        regularizer["initial_huber_cost"] + report["affected_initial_cost"])


def test_regularizer_off_preserves_solver_residual_pattern_and_report(monkeypatch):
    captured = []

    def inspect(residual, initial, **kwargs):
        captured.append((initial.copy(), residual(initial).copy(),
                         kwargs["jac_sparsity"].toarray().copy(), set(kwargs)))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    state_a, _ = _state()
    default_report = local_bundle_adjustment(
        state_a, MATRIX, BASELINE, window=3, disparity_offset=OFFSET)
    state_b, _ = _state()
    explicit_report = local_bundle_adjustment(
        state_b, MATRIX, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=False)
    assert len(captured) == 2
    for first, second in zip(captured[0][:3], captured[1][:3]):
        np.testing.assert_array_equal(first, second)
    assert captured[0][3] == captured[1][3]
    assert "stereo_motion_regularizer" not in default_report
    assert "stereo_motion_regularizer" not in explicit_report


def test_stale_or_mismatched_factor_is_skipped_without_changing_edge_ledger(monkeypatch):
    state, factor = _state()
    ledger_before = {edge: value.copy() for edge, value in state.stereo_motion.items()}
    wrong_matrix = MATRIX.copy()
    wrong_matrix[0, 0] += 1.
    captured = []

    def inspect(residual, initial, **kwargs):
        captured.append((len(residual(initial)), kwargs["jac_sparsity"].shape))
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    report = local_bundle_adjustment(
        state, wrong_matrix, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)
    regularizer = report["stereo_motion_regularizer"]
    assert regularizer["active_factors"] == 0
    assert regularizer["skipped_factors"][0]["reason"] == "calibration_mismatch"
    # The calibration mismatch excludes the factor, so no extra six rows enter
    # this solve's sparsity pattern.
    assert captured[0][0] == captured[0][1][0]
    for edge, value in ledger_before.items():
        np.testing.assert_array_equal(state.stereo_motion[edge], value)
    assert state.stereo_motion_regularizers[(2, 4)] is factor


def test_regularizer_rejects_invalid_or_mutated_calibration(monkeypatch):
    # Give the accepted optimizer a real reprojection improvement to commit;
    # the calibration mutation must still stop that otherwise valid apply.
    state, _ = _state(truth_last=1.1, initial_last=1.0, landmark_count=180)
    called = []

    def inspect(_residual, initial, **_kwargs):
        called.append(True)
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", inspect)
    complex_matrix = MATRIX.astype(np.complex128)
    complex_matrix[0, 0] += 1j
    rejected = local_bundle_adjustment(
        state, complex_matrix, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)
    assert rejected["reason"] == "invalid_regularizer_calibration"
    assert not called

    supplied = MATRIX.copy()
    real_solver = scipy_least_squares

    def mutate_during_solve(residual, initial, **kwargs):
        supplied[0, 0] += .25
        return real_solver(residual, initial, **kwargs)

    monkeypatch.setattr(bundle, "least_squares", mutate_during_solve)
    rejected = local_bundle_adjustment(
        state, supplied, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)
    assert rejected["reason"] == "stale_regularizer_calibration"
    assert not rejected["applied"]


def test_real_solver_reduces_objective_and_regularizer_limits_edge_drift():
    # Image observations prefer a displaced final camera, while the independent
    # raw stereo measurement remains the known 0.5 m motion. Both solutions must
    # improve their image objective; the correlated factor should reduce drift
    # against that accepted motion without altering the hard acceptance limit.
    state_off, _ = _state(truth_last=1.1, initial_last=1.0, landmark_count=180)
    state_on, _ = _state(truth_last=1.1, initial_last=1.0, landmark_count=180)
    report_off = local_bundle_adjustment(
        state_off, MATRIX.copy(), BASELINE, window=3, disparity_offset=OFFSET)
    report_on = local_bundle_adjustment(
        state_on, MATRIX.copy(), BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)

    assert report_off["applied"] and report_on["applied"]
    assert report_off["final_cost"] < report_off["initial_cost"]
    assert report_off["affected_final_cost"] < report_off["affected_initial_cost"]
    assert report_on["final_cost"] < report_on["initial_cost"]
    assert report_on["affected_final_cost"] < report_on["affected_initial_cost"]
    regularizer = report_on["stereo_motion_regularizer"]
    assert regularizer["augmented_final_huber_cost"] < regularizer["augmented_initial_huber_cost"]
    assert report_on["max_stereo_motion_translation_error_m"] < report_off[
        "max_stereo_motion_translation_error_m"]
    np.testing.assert_array_equal(state_on.keyframes[0].pose, _pose(0.))
    assert all(np.isfinite(state_on.keyframes[k].pose).all() for k in state_on.keyframes)


def test_existing_independent_motion_hard_gate_still_rejects_large_change(monkeypatch):
    state, _ = _state(truth_last=1.7, initial_last=1.0, landmark_count=180)

    def propose(_residual, initial, **_kwargs):
        value = initial.copy()
        # The second free keyframe is the second pose block; changing its x
        # translation improves its reprojection fit to the synthetic truth.
        value[9] += .7
        return SimpleNamespace(x=value, nfev=1, success=True)

    monkeypatch.setattr(bundle, "least_squares", propose)
    before = state.keyframes[2].pose.copy()
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, disparity_offset=OFFSET,
        stereo_motion_regularizer=True)
    assert report["affected_final_cost"] < report["affected_initial_cost"]
    assert report["reason"] == "independent_stereo_motion_inconsistency"
    assert state.revision == 0
    np.testing.assert_array_equal(state.keyframes[2].pose, before)


def test_mapstate_rejects_factor_from_changed_source_epoch():
    state, factor = _state()
    # A later geometry correction changes the source pose/revision. Immutable
    # camera-relative factors already installed remain valid, but stale newly
    # captured factors must not be attached.
    state.geometry_revision += 1
    # Remove the already-installed factor so the stale candidate reaches the
    # source-epoch validation rather than the duplicate-edge check.
    del state.stereo_motion_regularizers[(2, 4)]
    attached, reason = state.add_stereo_motion_regularizer(
        2, 4, factor, calibration_identity=factor.calibration_identity,
        source_pose=state.poses[2].copy(), source_geometry_revision=0)
    assert not attached
    assert reason == "source_geometry_epoch_changed"
    assert (2, 4) not in state.stereo_motion_regularizers
