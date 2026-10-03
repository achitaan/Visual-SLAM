"""Regression gates for independently optimized non-keyframe bundle cameras."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import Observation
from test_bundle_diagnostics import MATRIX, BASELINE, _scene
from test_owned_stereo_bundle import OFFSET, _factor


def _pose(x, rotvec=(0.0, 0.0, 0.0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    result[0, 3] = float(x)
    return result


def _free_source_case(*, source_x=1.65, source_bias=0.30, source_anchor=0, count=24):
    state, truth, existing_single_id = _scene()
    # This synthetic fixture uses frame IDs equal to the keyframe indices.
    for keyframe_id, keyframe in state.keyframes.items():
        keyframe.frame = keyframe_id
    state.record(state.poses[2].copy(), "tracking", source_anchor)

    true_source = _pose(source_x, (0.004, 0.012, -0.006))
    initial_source = true_source.copy()
    initial_source[0, 3] += source_bias
    state.poses[3] = initial_source.copy()
    state.relative_poses[3] = np.linalg.inv(state.keyframes[source_anchor].pose) @ initial_source

    rng = np.random.default_rng(7921)
    landmark_ids = [existing_single_id]
    for _ in range(count - 1):
        point = rng.uniform([-1.5, -1.1, 6.0], [1.5, 1.1, 11.0])
        target_pixel, depth = project(point[None], truth[2], MATRIX)
        target_right = right_pixel(
            target_pixel[0, 0], depth[0], MATRIX[0, 0], BASELINE, OFFSET
        )
        landmark_ids.append(state.add_landmark(
            point,
            rng.normal(size=128).astype(np.float32),
            2,
            {2: Observation(target_pixel[0].astype(np.float32), float(target_right))},
        ))
    return state, truth, true_source, landmark_ids


def _provider(state, truth, true_source, landmark_ids, calls=None):
    calls = [] if calls is None else calls

    def provide(payload):
        phase = payload.get("training_factor_phase", "prepare")
        calls.append(phase)
        if phase == "validate":
            return (), {"status": "validated", "reason": None}

        epoch = (payload["revision"], payload["geometry_revision"])
        source_anchor = int(state.pose_anchors[3])
        source_anchor_pose = state.keyframes[source_anchor].pose.copy()
        anchor_pose = state.keyframes[2].pose.copy()
        source_pose = state.poses[3].copy()
        target_pose = state.poses[2].copy()
        factors = []
        for index, landmark_id in enumerate(landmark_ids):
            landmark = state.landmarks[landmark_id]
            target_observation = landmark.observations[2]
            source_pixel, source_depth = project(
                landmark.position[None], true_source, MATRIX
            )
            source_right = right_pixel(
                source_pixel[0, 0], source_depth[0], MATRIX[0, 0], BASELINE, OFFSET
            )
            factor = _factor(
                source_frame=3,
                target_frame=2,
                source_anchor=source_anchor,
                target_anchor=2,
                source_pose=source_pose,
                target_pose=target_pose,
                source_anchor_pose=source_anchor_pose,
                target_anchor_pose=anchor_pose,
                source_pixel=source_pixel[0].astype(np.float32),
                target_pixel=target_observation.pixel.copy(),
                source_right=float(source_right),
                target_right=target_observation.right_u,
                landmark_id=landmark_id,
                source_landmark_id=-1,
                target_landmark_id=landmark_id,
                source_owned=False,
                target_owned=True,
                point=landmark.position.copy(),
                epoch=epoch,
            )
            factors.append(replace(factor, physical_identity=f"{index + 1:064x}"))
        return tuple(factors), {
            "status": "accepted",
            "selected_factors": len(factors),
            "model": "correlated_physical_stereo_image_rows_v1",
            "covariance_claim": False,
            "intentionally_reuses_sensor_evidence": True,
            "heldout_validation_claim": False,
        }

    return provide


def _snapshot(state):
    return {
        "revision": state.revision,
        "geometry_revision": state.geometry_revision,
        "poses": [pose.copy() for pose in state.poses],
        "relative_poses": [pose.copy() for pose in state.relative_poses],
        "keyframes": {key: frame.pose.copy() for key, frame in state.keyframes.items()},
        "points": {key: point.position.copy() for key, point in state.landmarks.items()},
    }


def _assert_snapshot_equal(state, saved):
    assert state.revision == saved["revision"]
    assert state.geometry_revision == saved["geometry_revision"]
    assert len(state.poses) == len(saved["poses"])
    assert len(state.relative_poses) == len(saved["relative_poses"])
    for actual, expected in zip(state.poses, saved["poses"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(state.relative_poses, saved["relative_poses"]):
        np.testing.assert_array_equal(actual, expected)
    assert state.keyframes.keys() == saved["keyframes"].keys()
    assert state.landmarks.keys() == saved["points"].keys()
    for key, expected in saved["keyframes"].items():
        np.testing.assert_array_equal(state.keyframes[key].pose, expected)
    for key, expected in saved["points"].items():
        np.testing.assert_array_equal(state.landmarks[key].position, expected)


def _params_to_pose(vector):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    pose[:3, 3] = vector[3:6]
    return pose


def _pose_to_params(pose):
    return np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]


def test_free_source_is_joint_gauge_invariant_has_correct_sparsity_and_commits_solver_pose(monkeypatch):
    state, truth, true_source, landmark_ids = _free_source_case(source_bias=0.1)
    original_source = state.poses[3].copy()
    original_target = state.keyframes[2].pose.copy()
    true_relative_source = np.linalg.inv(truth[2]) @ true_source
    calls = []
    provider = _provider(state, truth, true_source, landmark_ids, calls)
    real_solver = bundle.least_squares
    solver_capture = {}

    def inspect_solver(fun, initial, **kwargs):
        payload = prepared["payload"]
        selected = payload["selection"]["selected_landmarks"]
        assert len(selected) == 24
        assert state.pose_anchors[3] == 0  # The source uses the fixed origin anchor.
        assert {row["landmark_id"] for row in selected} == set(range(24))
        old_rows = sum(row["dimensions"] for row in payload["selected_observations"])
        old_rows += sum(row["dimensions"]
                        for row in payload["excluded_multiview_observations"])
        initial = np.asarray(initial, float).copy()
        baseline_residual = fun(initial)
        assert baseline_residual.shape == (old_rows + 6 * len(landmark_ids),)

        chart_start = len(payload["parameter_layout"]["initial_vector"])
        point_start = chart_start
        assert len(initial) == chart_start + 3 * len(landmark_ids) + 6
        target_offset = payload["parameter_layout"]["pose_offsets"]["2"]
        source_offset = chart_start + 3 * len(landmark_ids)
        initial_chart_source = _params_to_pose(
            initial[source_offset:source_offset + 6]
        )
        np.testing.assert_allclose(
            initial_chart_source, np.linalg.inv(original_target) @ original_source,
            rtol=0., atol=1e-11,
        )

        # T is the world realization of the chart reference. Keeping C and q
        # fixed while changing T applies a common world transform to the
        # reconstructed source and singleton points, so pair-only image rows
        # must remain exactly unchanged.
        common = _pose(-0.35, (0.11, -0.07, 0.045))
        moved = initial.copy()
        moved[target_offset:target_offset + 6] = _pose_to_params(
            common @ _params_to_pose(initial[target_offset:target_offset + 6])
        )
        moved_residual = fun(moved)
        np.testing.assert_array_equal(baseline_residual[old_rows:], moved_residual[old_rows:])

        # Compare central finite differences with the declared row sparsity.
        sparse = kwargs["jac_sparsity"].toarray().astype(bool)
        assert sparse.shape == (len(baseline_residual), len(initial))
        assert sparse[old_rows:old_rows + 3, source_offset:source_offset + 6].all()
        assert sparse[old_rows:old_rows + 3, point_start:point_start + 3].all()
        assert not sparse[old_rows:old_rows + 3, target_offset:target_offset + 6].any()
        assert sparse[old_rows + 3:old_rows + 6, point_start:point_start + 3].all()
        assert not sparse[old_rows + 3:old_rows + 6, source_offset:source_offset + 6].any()
        assert not sparse[old_rows + 3:old_rows + 6, target_offset:target_offset + 6].any()
        target_columns = list(range(target_offset, target_offset + 6))
        nuisance_columns = (list(range(source_offset, source_offset + 6))
                            + list(range(point_start, point_start + 3 * len(landmark_ids))))
        jacobian_columns = target_columns + nuisance_columns
        jacobian = np.empty((len(baseline_residual) - old_rows, len(jacobian_columns)))
        for output_column, column in enumerate(jacobian_columns):
            plus, minus = initial.copy(), initial.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            derivative = (fun(plus) - fun(minus))[old_rows:] / 2e-6
            jacobian[:, output_column] = derivative
            active = np.abs(derivative) > 1e-5
            assert not np.any(active & ~sparse[old_rows:, column]), (
                f"undeclared derivative at {column}"
            )

        # In the chart, pair-only rows have exactly zero target derivatives.
        target_jacobian = jacobian[:, :len(target_columns)]
        source_jacobian = jacobian[:, 6:12]
        point_jacobian = jacobian[:, 12:]
        np.testing.assert_array_equal(target_jacobian, np.zeros_like(target_jacobian))
        point_left, point_singular, _point_right = np.linalg.svd(
            point_jacobian, full_matrices=False
        )
        point_tolerance = (np.finfo(float).eps * max(point_jacobian.shape)
                           * point_singular[0])
        point_observable = point_singular > point_tolerance
        paired_pose_jacobian = np.column_stack((target_jacobian, source_jacobian))
        relative_projection = paired_pose_jacobian - point_left[:, point_observable] @ (
            point_left[:, point_observable].T @ paired_pose_jacobian
        )
        relative_information = relative_projection.T @ relative_projection
        assert np.linalg.matrix_rank(relative_information, tol=1e-8) > 0

        nuisance_jacobian = np.column_stack((source_jacobian, point_jacobian))
        left, singular_values, _right = np.linalg.svd(nuisance_jacobian, full_matrices=False)
        rank_tolerance = (np.finfo(float).eps * max(nuisance_jacobian.shape)
                          * singular_values[0])
        observable = singular_values > rank_tolerance
        projected = target_jacobian - left[:, observable] @ (
            left[:, observable].T @ target_jacobian
        )
        target_information = target_jacobian.T @ target_jacobian
        assert np.linalg.norm(target_information, ord="fro") == 0.0
        marginalized_information = projected.T @ projected
        assert np.linalg.norm(marginalized_information, ord="fro") == 0.0

        result = real_solver(fun, initial, **kwargs)
        solver_capture["x"] = result.x.copy()
        return result

    prepared = {}

    def provider_with_capture(payload):
        if payload.get("training_factor_phase") == "prepare":
            prepared["payload"] = payload
        return provider(payload)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    original_origin = state.keyframes[0].pose.copy()
    report = local_bundle_adjustment(
        state,
        MATRIX,
        BASELINE,
        window=3,
        max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=provider_with_capture,
    )

    assert calls == ["prepare", "validate"]
    assert report["applied"] is True, report
    summary = report["owned_stereo_image_bundle"]
    assert summary["status"] == "accepted"
    assert summary["optimized_intermediate_frames"] == [3]
    assert summary["optimized_single_view_points"] == len(landmark_ids)
    assert summary["unique_image_rows_added"] == 2 * len(landmark_ids)
    assert summary["reused_selected_points"] == 0
    assert not set(summary["factor_landmark_ids"]).intersection(range(24))
    chart = summary.get("optimizer_parameter_chart", summary)
    source_offset = chart["target_relative_camera_offsets"]["3"]
    target_offset = prepared["payload"]["parameter_layout"]["pose_offsets"]["2"]
    candidate_target = _params_to_pose(
        solver_capture["x"][target_offset:target_offset + 6]
    )
    candidate_chart_source = _params_to_pose(
        solver_capture["x"][source_offset:source_offset + 6]
    )
    candidate_source = candidate_target @ candidate_chart_source
    np.testing.assert_allclose(state.poses[3], candidate_source, rtol=0., atol=1e-12)
    np.testing.assert_allclose(
        state.relative_poses[3],
        np.linalg.inv(state.keyframes[state.pose_anchors[3]].pose) @ candidate_source,
        rtol=0., atol=1e-12,
    )
    assert state.pose_anchors[3] == 0
    assert chart["target_relative_camera_reference_keyframes"]["3"] == 2
    assert not np.allclose(state.relative_poses[3], candidate_chart_source)
    np.testing.assert_array_equal(state.keyframes[0].pose, original_origin)
    assert 0 in report["fixed_keyframes"]
    assert np.isfinite(state.poses[3]).all()
    assert not np.array_equal(state.poses[3], original_source)
    final_relative_source = np.linalg.inv(state.keyframes[2].pose) @ state.poses[3]
    relative_error = np.linalg.inv(true_relative_source) @ final_relative_source
    initial_relative_error = np.linalg.inv(true_relative_source) @ (
        np.linalg.inv(original_target) @ original_source
    )
    initial_world_error = np.linalg.norm(original_source[:3, 3] - true_source[:3, 3])
    final_world_error = np.linalg.norm(state.poses[3][:3, 3] - true_source[:3, 3])
    initial_chart_rotation_error_deg = np.degrees(
        Rotation.from_matrix(initial_relative_error[:3, :3]).magnitude()
    )
    final_chart_rotation_error_deg = np.degrees(
        Rotation.from_matrix(relative_error[:3, :3]).magnitude()
    )
    assert final_world_error < 0.5 * initial_world_error, (
        "synthetic source-pose recovery regressed under the chart: "
        f"world translation error {initial_world_error:.6f}m -> {final_world_error:.6f}m; "
        f"relative C translation error "
        f"{np.linalg.norm(initial_relative_error[:3, 3]):.6f}m -> "
        f"{np.linalg.norm(relative_error[:3, 3]):.6f}m; "
        f"relative C rotation error {initial_chart_rotation_error_deg:.6f}deg -> "
        f"{final_chart_rotation_error_deg:.6f}deg"
    )
    assert np.linalg.norm(relative_error[:3, 3]) < 1e-3
    assert final_chart_rotation_error_deg < 0.01
    assert summary["augmented_objective_reduced"] is True
    assert summary["affected_objective_reduced"] is True
    for landmark_id in landmark_ids:
        point_offset = chart["target_relative_point_offsets"][str(landmark_id)]
        chart_point = solver_capture["x"][point_offset:point_offset + 3]
        expected_world_point = (candidate_target @ np.r_[chart_point, 1.0])[:3]
        np.testing.assert_allclose(
            state.landmarks[landmark_id].position, expected_world_point,
            rtol=0., atol=1e-11,
        )


def point_start_from_payload(payload):
    return (payload["parameter_layout"]["point_offset"]
            + 3 * len(payload["selection"]["selected_landmarks"]))


def test_same_anchor_inbound_and_outbound_motion_edges_reject_candidate_atomically(monkeypatch):
    state, truth, true_source, landmark_ids = _free_source_case(
        source_x=2.0, source_bias=0.8, source_anchor=2
    )
    state.record(state.keyframes[2].pose.copy(), "tracking", 2)
    state.add_stereo_motion(2, 3, np.linalg.inv(state.poses[2]) @ state.poses[3])
    state.add_stereo_motion(3, 4, np.linalg.inv(state.poses[3]) @ state.poses[4])
    before = _snapshot(state)

    # Return a finite image-improving proposal that moves the source to the
    # known synthetic image geometry. Both motion records still match the
    # recorded pre-solve trajectory, so these same-anchor edges must veto it.
    real_solver = bundle.least_squares
    prepared = {}
    provider = _provider(state, truth, true_source, landmark_ids)

    def capture_provider(payload):
        if payload.get("training_factor_phase", "prepare") == "prepare":
            prepared["payload"] = payload
        return provider(payload)

    def candidate_crossing_motion_gate(fun, initial, **kwargs):
        result = real_solver(fun, initial, **kwargs)
        candidate = result.x.copy()
        layout = prepared["payload"]["parameter_layout"]
        target_offset = layout["pose_offsets"]["2"]
        chart_start = len(layout["initial_vector"])
        source_offset = chart_start + 3 * len(landmark_ids)
        target = _params_to_pose(candidate[target_offset:target_offset + 6])
        chart_source = np.linalg.inv(target) @ true_source
        candidate[source_offset:source_offset + 6] = _pose_to_params(chart_source)
        return SimpleNamespace(
            x=candidate,
            nfev=result.nfev,
            success=result.success,
            status=result.status,
            message=result.message,
            cost=result.cost,
            optimality=result.optimality,
        )

    monkeypatch.setattr(bundle, "least_squares", candidate_crossing_motion_gate)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40, disparity_offset=OFFSET,
        training_factor_provider=capture_provider,
    )

    assert report["applied"] is False
    assert report["reason"] == "independent_stereo_motion_inconsistency"
    assert report["owned_stereo_image_bundle"]["guarded_intermediate_motion_edges"] == [
        [2, 3], [3, 4]
    ]
    assert report["max_stereo_motion_translation_error_m"] > 0.5
    assert report["owned_stereo_image_bundle"]["affected_objective_reduced"] is True
    assert report["owned_stereo_image_bundle"]["augmented_objective_reduced"] is True
    _assert_snapshot_equal(state, before)


@pytest.mark.parametrize("mutation", ["status", "anchor"])
def test_stale_free_source_status_or_anchor_never_commits(monkeypatch, mutation):
    state, truth, true_source, landmark_ids = _free_source_case()
    before = _snapshot(state)
    real_solver = bundle.least_squares

    def mutate_after_solve(fun, initial, **kwargs):
        result = real_solver(fun, initial, **kwargs)
        if mutation == "status":
            state.statuses[3] = "lost"
        else:
            state.pose_anchors[3] = 2
        return result

    monkeypatch.setattr(bundle, "least_squares", mutate_after_solve)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40, disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, true_source, landmark_ids),
    )
    assert report["applied"] is False
    assert report["reason"] == "stale_intermediate_frame"
    assert state.revision == before["revision"]
    assert state.geometry_revision == before["geometry_revision"]
    for actual, expected in zip(state.poses, before["poses"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(state.relative_poses, before["relative_poses"]):
        np.testing.assert_array_equal(actual, expected)
    for key, expected in before["keyframes"].items():
        np.testing.assert_array_equal(state.keyframes[key].pose, expected)
    for key, expected in before["points"].items():
        np.testing.assert_array_equal(state.landmarks[key].position, expected)


@pytest.mark.parametrize("invalid", ["origin", "keyframe", "lost", "nonfinite", "mixed"])
def test_mapstate_frame_updates_validate_entire_batch_before_any_geometry_write(invalid):
    state, _, _ = _scene()
    for keyframe_id, keyframe in state.keyframes.items():
        keyframe.frame = keyframe_id
    state.record(state.poses[2].copy(), "tracking", 2)
    if invalid == "lost":
        state.statuses[3] = "lost"
    before = _snapshot(state)
    corrected = {key: frame.pose.copy() for key, frame in state.keyframes.items()}
    corrected[1][0, 3] += 0.015
    frame_pose = state.poses[3].copy()
    frame_pose[0, 3] += 0.12
    landmark_id = min(state.landmarks)
    changed_point = state.landmarks[landmark_id].position + np.array([0.1, 0.0, 0.0])
    if invalid == "origin":
        updates = {0: frame_pose}
    elif invalid == "keyframe":
        updates = {2: frame_pose}
    elif invalid == "lost":
        updates = {3: frame_pose}
    elif invalid == "nonfinite":
        frame_pose[1, 3] = np.nan
        updates = {3: frame_pose}
    else:
        updates = {3: frame_pose, 2: frame_pose}

    with pytest.raises((ValueError, TypeError)):
        state.apply_corrections(
            state.revision,
            corrected,
            propagate_landmarks=False,
            landmark_updates={landmark_id: changed_point},
            frame_updates=updates,
        )
    _assert_snapshot_equal(state, before)


def test_mapstate_frame_update_stale_revision_and_unequal_parallel_state_are_atomic():
    state, _, _ = _scene()
    for keyframe_id, keyframe in state.keyframes.items():
        keyframe.frame = keyframe_id
    state.record(state.poses[2].copy(), "tracking", 2)
    corrected = {key: frame.pose.copy() for key, frame in state.keyframes.items()}
    valid_update = state.poses[3].copy()
    valid_update[0, 3] += 0.1
    before = _snapshot(state)
    assert state.apply_corrections(
        state.revision + 1, corrected, frame_updates={3: valid_update}
    ) is False
    _assert_snapshot_equal(state, before)

    # A malformed parallel-state length is rejected before corrections or
    # landmark writes, preserving even the pre-existing malformed state.
    state.relative_poses.pop()
    malformed_relative = [pose.copy() for pose in state.relative_poses]
    malformed_snapshot = _snapshot(state)
    landmark_id = min(state.landmarks)
    with pytest.raises(ValueError, match="Inconsistent intermediate frame state"):
        state.apply_corrections(
            state.revision,
            corrected,
            propagate_landmarks=False,
            landmark_updates={landmark_id: state.landmarks[landmark_id].position + 1.0},
            frame_updates={3: valid_update},
        )
    _assert_snapshot_equal(state, malformed_snapshot)
    assert len(state.relative_poses) == len(malformed_relative)
    for actual, expected in zip(state.relative_poses, malformed_relative):
        np.testing.assert_array_equal(actual, expected)
