"""Regression tests for bounded, actual-frame source-image retention in BA."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from test_bundle_diagnostics import BASELINE, MATRIX, _scene
from test_free_source_stereo_bundle import (
    OFFSET,
    _free_source_case,
    _params_to_pose,
    _pose_to_params,
)
from test_owned_stereo_bundle import _factor
from slam_state import MapState


CALIBRATION_ID = "retained-source-test-calibration-v1"


def _state_with_source_frame():
    state, truth, single_id = _scene()
    # This accepted non-keyframe image is deliberately distinct from every
    # keyframe's identity and actual frame number.
    state.record(state.poses[2].copy(), "tracking", 2)
    return state, truth, single_id, 3


def _retained_record(state, frame, landmark_id, pixel, *, calibration=CALIBRATION_ID):
    anchor = int(state.pose_anchors[frame])
    return {
        "frame_id": int(frame),
        "calibration_identity": calibration,
        "measurement_role": "tracking_fit_consumed",
        "accepted_revision": int(state.revision),
        "accepted_geometry_revision": int(state.geometry_revision),
        "anchor_keyframe_id": anchor,
        "relative_pose": np.linalg.inv(state.keyframes[anchor].pose) @ state.poses[frame],
        "rows": [{"landmark_id": int(landmark_id),
                  "pixel_float32": np.asarray(pixel, np.float32).copy()}],
    }


def _map_snapshot(state):
    return {
        "revision": state.revision,
        "geometry_revision": state.geometry_revision,
        "poses": [x.copy() for x in state.poses],
        "relative_poses": [x.copy() for x in state.relative_poses],
        "pose_anchors": copy.deepcopy(state.pose_anchors),
        "statuses": list(state.statuses),
        "keyframes": {k: v.pose.copy() for k, v in state.keyframes.items()},
        "points": {k: v.position.copy() for k, v in state.landmarks.items()},
        "retained": copy.deepcopy(state.retained_source_observations),
    }


def _assert_map_snapshot(state, saved):
    assert state.revision == saved["revision"]
    assert state.geometry_revision == saved["geometry_revision"]
    assert len(state.poses) == len(saved["poses"])
    assert len(state.relative_poses) == len(saved["relative_poses"])
    assert state.pose_anchors == saved["pose_anchors"]
    assert state.statuses == saved["statuses"]
    for actual, expected in zip(state.poses, saved["poses"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(state.relative_poses, saved["relative_poses"]):
        np.testing.assert_array_equal(actual, expected)
    for key, expected in saved["keyframes"].items():
        np.testing.assert_array_equal(state.keyframes[key].pose, expected)
    for key, expected in saved["points"].items():
        np.testing.assert_array_equal(state.landmarks[key].position, expected)
    assert state.retained_source_observations.keys() == saved["retained"].keys()
    for frame, expected in saved["retained"].items():
        actual = state.retained_source_observations[frame]
        assert actual["frame_id"] == expected["frame_id"]
        assert actual["calibration_identity"] == expected["calibration_identity"]
        np.testing.assert_array_equal(actual["relative_pose"], expected["relative_pose"])
        assert len(actual["rows"]) == len(expected["rows"])
        for left, right in zip(actual["rows"], expected["rows"]):
            assert left["landmark_id"] == right["landmark_id"]
            np.testing.assert_array_equal(left["pixel_float32"], right["pixel_float32"])


def _pose(x, rotvec=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    pose[0, 3] = float(x)
    return pose


def _long_frame_scene(source_bias=0.11):
    """Use actual source/keyframe IDs 68/61/76 while keeping a small fixture."""
    state, _truth, _unused_source, _pair_ids = _free_source_case(
        source_bias=source_bias, count=24
    )
    anchor_id, target_id = 1, 2
    for keyframe_id, actual_frame in ((0, 0), (1, 61), (2, 76)):
        state.keyframes[keyframe_id].frame = actual_frame

    # A noncommuting stored source-relative pose catches accidental use of an
    # absolute world pose or keyframe-ID/frame-ID alias.
    relative68 = _pose(0.24, (0.09, -0.045, 0.035))
    true_source68 = state.keyframes[anchor_id].pose @ relative68
    current_source68 = true_source68.copy()
    current_source68[0, 3] += float(source_bias)
    poses, anchors, statuses, relatives = [], [], [], []
    for frame in range(77):
        if frame < 61:
            anchor = 0
        elif frame < 76:
            anchor = anchor_id
        else:
            anchor = target_id
        pose = state.keyframes[anchor].pose.copy()
        if frame == 68:
            pose = current_source68.copy()
            anchor = anchor_id
        elif frame == 61:
            pose = state.keyframes[anchor_id].pose.copy()
        elif frame == 76:
            pose = state.keyframes[target_id].pose.copy()
        poses.append(pose)
        anchors.append(anchor)
        statuses.append("tracking")
        relatives.append(np.linalg.inv(state.keyframes[anchor].pose) @ pose)
    state.poses = poses
    state.pose_anchors = anchors
    state.statuses = statuses
    state.relative_poses = relatives
    truth_points = {ident: state.landmarks[ident].position.copy() for ident in range(24)}
    return state, true_source68, truth_points, anchor_id, target_id


def _append_frame80(state, target_id=2):
    for frame in range(77, 81):
        pose = state.keyframes[target_id].pose.copy()
        if frame == 80:
            relative = _pose(0.18, (-0.055, 0.07, 0.025))
            pose = state.keyframes[target_id].pose @ relative
            pose[1, 3] += 0.08
        state.record(pose, "tracking", target_id)


def _clone_state(state):
    clone = MapState(metric=state.metric)
    clone.keyframes = copy.deepcopy(state.keyframes)
    clone.landmarks = copy.deepcopy(state.landmarks)
    clone.poses = [pose.copy() for pose in state.poses]
    clone.pose_anchors = copy.deepcopy(state.pose_anchors)
    clone.relative_poses = [pose.copy() for pose in state.relative_poses]
    clone.statuses = list(state.statuses)
    clone.stereo_motion = copy.deepcopy(state.stereo_motion)
    clone.retained_source_observations = copy.deepcopy(
        state.retained_source_observations
    )
    clone.revision = int(state.revision)
    clone.geometry_revision = int(state.geometry_revision)
    clone.next_landmark = int(state.next_landmark)
    return clone


def _owned_factor_provider(state, *, source_frame, target_frame, source_true,
                           truth_points, landmark_ids=(0,)):
    """Single current factor pool whose source is disjoint from retained rows."""
    calls = []

    def provide(payload):
        phase = payload.get("training_factor_phase", "prepare")
        calls.append(phase)
        if phase == "validate":
            return (), {"status": "validated", "reason": None}
        factors = []
        for index, ident in enumerate(landmark_ids):
            landmark = state.landmarks[int(ident)]
            point = np.asarray(landmark.position, float).copy()
            source_pixel, source_depth = project(
                truth_points[int(ident)][None], source_true, MATRIX
            )
            target_pixel = landmark.observations[2].pixel.copy()
            _target_pixel, target_depth = project(point[None], state.keyframes[2].pose, MATRIX)
            source_right = right_pixel(
                source_pixel[0, 0], source_depth[0], MATRIX[0, 0], BASELINE, OFFSET
            )
            target_right = right_pixel(
                target_pixel[0], target_depth[0], MATRIX[0, 0], BASELINE, OFFSET
            )
            source_anchor = int(state.pose_anchors[source_frame])
            factor = _factor(
                source_frame=source_frame,
                target_frame=target_frame,
                source_anchor=source_anchor,
                target_anchor=2,
                source_pose=state.poses[source_frame].copy(),
                target_pose=state.poses[target_frame].copy(),
                source_anchor_pose=state.keyframes[source_anchor].pose.copy(),
                target_anchor_pose=state.keyframes[2].pose.copy(),
                source_pixel=source_pixel[0].astype(np.float32),
                target_pixel=target_pixel,
                source_right=float(source_right),
                target_right=float(target_right),
                landmark_id=int(ident),
                source_landmark_id=-1,
                target_landmark_id=int(ident),
                source_owned=False,
                target_owned=True,
                point=point,
                epoch=(state.revision, state.geometry_revision),
            )
            factors.append(factor)
        factors = tuple(factors)
        return factors, {
            "status": "accepted",
            "selected_factors": len(factors),
            "model": "correlated_physical_stereo_image_rows_v1",
            "covariance_claim": False,
            "heldout_validation_claim": False,
            "intentionally_reuses_sensor_evidence": True,
        }

    provide.calls = calls
    return provide


def _history_provider(source_frame, source_true, truth_points, *, landmark_ids=None):
    calls = []

    def provide(payload):
        phase = payload["source_history_phase"]
        calls.append(phase)
        if phase == "validate":
            return [], {"status": "validated", "reason": None}
        selected = set(map(int, payload["source_history_selected_landmark_ids"]))
        excluded = set() if landmark_ids is None else set(map(int, landmark_ids))
        rows = []
        for ident in sorted(selected - excluded):
            pixel = project(truth_points[ident][None], source_true, MATRIX)[0][0]
            rows.append({
                "frame_id": int(source_frame),
                "landmark_id": ident,
                "pixel": pixel.astype(np.float32),
            })
        provide.prepare_rows = rows
        return rows, {"status": "eligible", "captured_rows": len(rows)}

    provide.calls = calls
    provide.prepare_rows = []
    return provide


def _run_first_bundle(state, source_true, truth_points, *, retain):
    source_frame, target_frame = 68, 76
    owned = _owned_factor_provider(
        state, source_frame=source_frame, target_frame=target_frame,
        source_true=source_true, truth_points=truth_points,
    )
    history = _history_provider(
        source_frame, source_true, truth_points, landmark_ids=(0,)
    )
    captured = {}

    def sink(phase, payload):
        if phase == "prepared":
            captured["prepared"] = copy.deepcopy(payload)

    real_solver = bundle.least_squares

    def solver_wrapper(fun, initial, **kwargs):
        initial = np.asarray(initial, float).copy()
        captured["initial"] = initial.copy()
        captured["residual_initial"] = np.asarray(fun(initial), float).copy()
        captured["sparsity"] = kwargs["jac_sparsity"].toarray().astype(bool)
        result = real_solver(fun, initial, **kwargs)
        captured["residual_final"] = np.asarray(fun(result.x), float).copy()
        return result

    bundle.least_squares = solver_wrapper
    try:
        result = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET, solver_accuracy="precise",
            training_factor_provider=owned,
            source_history_provider=history,
            retain_source_observations=retain,
            retained_source_calibration_identity=CALIBRATION_ID,
            retained_source_exclusions=(
                {"valid": True, "landmark_ids": [], "source_frame": source_frame,
                 "source_pixels": []}
                if retain else None
            ),
            diagnostic_sink=sink,
        )
    finally:
        bundle.least_squares = real_solver
    captured["owned_calls"] = list(owned.calls)
    captured["history_calls"] = list(history.calls)
    captured["history_input_rows"] = history.prepare_rows
    return result, captured


def _run_future_bundle(state, source_frame, source_true, truth_points, *, retain):
    target_frame = 76
    owned = _owned_factor_provider(
        state, source_frame=source_frame, target_frame=target_frame,
        source_true=source_true, truth_points=truth_points,
    )
    history = _history_provider(
        source_frame, source_true, truth_points,
        landmark_ids=tuple(range(len(truth_points))),
    )
    captured = {}

    def sink(phase, payload):
        if phase == "prepared":
            captured["prepared"] = copy.deepcopy(payload)

    real_solver = bundle.least_squares

    def solver_wrapper(fun, initial, **kwargs):
        initial = np.asarray(initial, float).copy()
        captured["fun"] = fun
        captured["initial"] = initial.copy()
        captured["residual_initial"] = np.asarray(fun(initial), float).copy()
        captured["sparsity"] = kwargs["jac_sparsity"].toarray().astype(bool)
        result = real_solver(fun, initial, **kwargs)
        captured["result_x"] = result.x.copy()
        captured["residual_final"] = np.asarray(fun(result.x), float).copy()
        return result

    source_pixel = project(truth_points[0][None], source_true, MATRIX)[0][0]
    bundle.least_squares = solver_wrapper
    try:
        result = local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET, solver_accuracy="precise",
            training_factor_provider=owned,
            source_history_provider=history,
            retain_source_observations=retain,
            retained_source_calibration_identity=CALIBRATION_ID,
            retained_source_exclusions=(
                {"valid": True, "landmark_ids": [0],
                 "source_frame": int(source_frame),
                 "source_pixels": [source_pixel.astype(np.float32)]}
                if retain else None
            ),
            diagnostic_sink=sink,
        )
    finally:
        bundle.least_squares = real_solver
    captured["owned_calls"] = list(owned.calls)
    captured["history_calls"] = list(history.calls)
    return result, captured


def test_accepted_current_history_rows_are_retained_for_future_actual_frame_use():
    state, source_true, truth_points, anchor_id, _target_id = _long_frame_scene()
    expected_pixels = {
        ident: project(truth_points[ident][None], source_true, MATRIX)[0][0].astype(np.float32)
        for ident in range(1, 24)
    }
    report, captured = _run_first_bundle(
        state, source_true, truth_points, retain=True
    )
    assert report["applied"] is True, report
    assert captured["owned_calls"] == ["prepare", "validate"]
    assert captured["history_calls"] == ["prepare", "validate"]
    history = report["owned_stereo_image_bundle"]["source_history_bundle"]
    assert history["source_frame"] == 68
    assert history["installed_rows"] == len(expected_pixels)
    # The target row is already present at its actual keyframe image; only the
    # non-keyframe source endpoint adds a new owned stereo row.
    assert report["owned_stereo_image_bundle"]["unique_image_rows_added"] == 1

    retained = report["retained_source_observations"]
    assert retained["requested"] is True
    assert retained["model"] == "anchored_actual_source_image_retention_v1"
    # There were no old retained rows before this first solve. The current
    # source-history measurements join this solve once, then become eligible
    # for a later solve only after atomic acceptance.
    assert retained["installed_rows"] == 0
    assert retained["components_added"] == 0
    assert retained["status"] == "registered_only"
    assert retained["fit_status"] == "skipped"
    assert retained["fit_reason"] == "no_eligible_retained_rows"
    assert retained["registry_status"] == "committed"
    assert retained["current_history_rows_registered"] == len(expected_pixels)
    assert retained["stored_current_rows"] == len(expected_pixels)
    assert retained["registered_rows_after_commit"] == len(expected_pixels)
    assert retained["registered_frames_after_commit"] == [68]
    registry = state.retained_source_observations
    assert set(registry) == {68}
    record = registry[68]
    assert record["frame_id"] == 68
    assert record["calibration_identity"] == CALIBRATION_ID
    assert record["measurement_role"] == "tracking_fit_consumed"
    assert record["anchor_keyframe_id"] == anchor_id
    assert 68 not in {keyframe.frame for keyframe in state.keyframes.values()}
    assert all(68 not in landmark.observations for landmark in state.landmarks.values())
    assert {row["landmark_id"] for row in record["rows"]} == set(expected_pixels)
    for row in record["rows"]:
        assert row["pixel_float32"].dtype == np.float32
        np.testing.assert_array_equal(
            row["pixel_float32"], expected_pixels[row["landmark_id"]]
        )
    reconstructed = state.keyframes[anchor_id].pose @ record["relative_pose"]
    np.testing.assert_allclose(reconstructed, state.poses[68], rtol=0., atol=1e-12)

    # The accepted solve used source-history rows exactly once. The registry is
    # written after acceptance for future solves; it cannot duplicate the same
    # rows in this just-completed objective.
    prepared = captured["prepared"]
    original_components = int(np.asarray(
        prepared["parameter_layout"]["dimension_mask"], dtype=bool
    ).sum()) + int(np.asarray(
        prepared["parameter_layout"]["held_out_dimension_mask"], dtype=bool
    ).sum())
    assert len(captured["residual_initial"]) == (
        original_components + 3 + 2 * len(expected_pixels)
    )
    assert report["affected_final_cost"] < report["affected_initial_cost"]
    assert report["augmented_final_cost"] < report["augmented_initial_cost"]

    # The provider-owned measurement buffers can be reused or mutated after
    # commit; the map registry keeps independent copies.
    expected_saved = copy.deepcopy(record)
    captured["history_input_rows"][0]["pixel"][:] = -100
    np.testing.assert_array_equal(
        registry[68]["rows"][0]["pixel_float32"],
        expected_saved["rows"][0]["pixel_float32"],
    )


def test_first_opt_in_registration_leaves_empty_registry_solve_equal_to_off():
    state_on, source_on, points_on, _anchor_on, _target_on = _long_frame_scene()
    state_off, source_off, points_off, _anchor_off, _target_off = _long_frame_scene()
    report_on, captured_on = _run_first_bundle(
        state_on, source_on, points_on, retain=True
    )
    report_off, captured_off = _run_first_bundle(
        state_off, source_off, points_off, retain=False
    )
    assert report_on["applied"] and report_off["applied"]
    np.testing.assert_array_equal(
        captured_on["residual_initial"], captured_off["residual_initial"]
    )
    np.testing.assert_array_equal(
        captured_on["residual_final"], captured_off["residual_final"]
    )
    for keyframe_id in state_on.keyframes:
        np.testing.assert_array_equal(
            state_on.keyframes[keyframe_id].pose,
            state_off.keyframes[keyframe_id].pose,
        )
    for landmark_id in state_on.landmarks:
        np.testing.assert_array_equal(
            state_on.landmarks[landmark_id].position,
            state_off.landmarks[landmark_id].position,
        )
    for pose_on, pose_off in zip(state_on.poses, state_off.poses):
        np.testing.assert_array_equal(pose_on, pose_off)
    assert state_off.retained_source_observations == {}
    assert set(state_on.retained_source_observations) == {68}
    assert report_on["initial_cost"] == report_off["initial_cost"]
    assert report_on["final_cost"] == report_off["final_cost"]


def _prepare_future_ablation_states():
    state_on, source68, truth_points, _anchor_id, _target_id = _long_frame_scene()
    first, _ = _run_first_bundle(state_on, source68, truth_points, retain=True)
    assert first["applied"] is True, first
    state_off = _clone_state(state_on)
    for state in (state_on, state_off):
        # A later small map correction perturbs a point whose old non-keyframe
        # image was retained. One keyframe measurement has realistic pixel noise
        # so the retained view can measurably influence its next estimate.
        state.landmarks[1].position += np.array([0.12, -0.07, 0.22])
        state.landmarks[1].observations[2].pixel += np.array([0.65, -0.4])
    relative80 = _pose(0.18, (-0.055, 0.07, 0.025))
    source80 = state_on.keyframes[2].pose @ relative80
    _append_frame80(state_on)
    _append_frame80(state_off)
    return state_on, state_off, source80, source68, truth_points


def _retained_pixel_error(state, frame, landmark_id, matrix=MATRIX):
    record = state.retained_source_observations[int(frame)]
    row = next(item for item in record["rows"]
               if item["landmark_id"] == int(landmark_id))
    camera = state.keyframes[record["anchor_keyframe_id"]].pose @ record["relative_pose"]
    predicted = project(
        state.landmarks[int(landmark_id)].position[None], camera, matrix
    )[0][0]
    return float(np.linalg.norm(predicted - row["pixel_float32"]))


def test_future_bundle_retained_rows_constrain_point_and_anchor_with_no_free_source_camera():
    state_on, state_off, source80, _source68, truth_points = (
        _prepare_future_ablation_states()
    )
    registry_off_before = copy.deepcopy(state_off.retained_source_observations)
    report_off, captured_off = _run_future_bundle(
        state_off, 80, source80, truth_points, retain=False
    )
    report_on, captured = _run_future_bundle(
        state_on, 80, source80, truth_points, retain=True
    )
    assert report_off["applied"] is True, report_off
    assert report_on["applied"] is True, report_on
    prepared_off = captured_off["prepared"]
    original_components_off = int(np.asarray(
        prepared_off["parameter_layout"]["dimension_mask"], dtype=bool
    ).sum()) + int(np.asarray(
        prepared_off["parameter_layout"]["held_out_dimension_mask"], dtype=bool
    ).sum())
    assert report_off["owned_stereo_image_bundle"]["unique_image_rows_added"] == 1
    assert len(captured_off["residual_initial"]) == original_components_off + 3
    assert state_off.retained_source_observations.keys() == registry_off_before.keys()
    for frame, old_record in registry_off_before.items():
        live_record = state_off.retained_source_observations[frame]
        assert live_record["accepted_revision"] == old_record["accepted_revision"]
        assert live_record["accepted_geometry_revision"] == old_record["accepted_geometry_revision"]
        np.testing.assert_array_equal(live_record["relative_pose"], old_record["relative_pose"])
        assert [row["landmark_id"] for row in live_record["rows"]] == [
            row["landmark_id"] for row in old_record["rows"]
        ]
        for live_row, old_row in zip(live_record["rows"], old_record["rows"]):
            np.testing.assert_array_equal(
                live_row["pixel_float32"], old_row["pixel_float32"]
            )
        transported = state_off.keyframes[live_record["anchor_keyframe_id"]].pose @ live_record["relative_pose"]
        np.testing.assert_allclose(transported, state_off.poses[frame], rtol=0., atol=1e-12)
    retained = report_on["retained_source_observations"]
    assert retained["fit_status"] == "active"
    assert retained["source_frames"] == [68]
    assert retained["installed_landmark_ids"] == list(range(1, 24))
    assert retained["installed_rows"] == 23
    assert retained["components_added"] == 46
    assert retained["cost_after"] < retained["cost_before"]
    assert report_on["augmented_final_cost"] < report_on["augmented_initial_cost"]
    assert report_on["affected_final_cost"] < report_on["affected_initial_cost"]

    # A second solve consumes old frame-68 rows once, while a current frame-80
    # owned pair contributes only its new source image (the target image is an
    # existing keyframe measurement). The OFF arm starts from the same accepted
    # first-stage geometry and does not add the 46 historical residual values.
    prepared = captured["prepared"]
    original_components = int(np.asarray(
        prepared["parameter_layout"]["dimension_mask"], dtype=bool
    ).sum()) + int(np.asarray(
        prepared["parameter_layout"]["held_out_dimension_mask"], dtype=bool
    ).sum())
    assert report_on["owned_stereo_image_bundle"]["unique_image_rows_added"] == 1
    assert len(captured["residual_initial"]) == original_components + 3 + 46
    chart = prepared["parameter_layout"]["target_relative_chart"]
    anchor_offset = chart["free_keyframe_pose_offsets"]["1"]
    source_camera_offset = chart["target_relative_camera_offsets"]["80"]
    point_index = next(
        index for index, row in enumerate(prepared["selection"]["selected_landmarks"])
        if row["landmark_id"] == 1
    )
    point_offset = chart["world_selected_point_offset"] + 3 * point_index
    row_start = len(captured["residual_initial"]) - 46
    rows = slice(row_start, row_start + 2)  # first installed retained row is LM 1
    sparsity = captured["sparsity"]
    assert sparsity[rows, anchor_offset:anchor_offset + 6].any()
    assert sparsity[rows, point_offset:point_offset + 3].any()
    assert not sparsity[rows, source_camera_offset:source_camera_offset + 6].any()

    # Check the declared dependencies against the actual residual function,
    # rather than relying only on the assembled sparsity matrix.
    initial = captured["initial"]
    fun = captured["fun"]
    base_residual = fun(initial)[rows]
    numeric_norms = {}
    for name, offset, width in (
        ("anchor", anchor_offset, 6),
        ("point", point_offset, 3),
        ("free_source_camera", source_camera_offset, 6),
    ):
        changes = []
        for column in range(offset, offset + width):
            stepped = initial.copy()
            stepped[column] += 1e-6
            changes.append((fun(stepped)[rows] - base_residual) / 1e-6)
        jacobian = np.column_stack(changes)
        numeric_norms[name] = float(np.linalg.norm(jacobian))
        declared = sparsity[rows, offset:offset + width]
        numerical_support = np.abs(jacobian) > 1e-6
        assert not np.any(numerical_support & ~declared)
    assert numeric_norms["anchor"] > 1e-3
    assert numeric_norms["point"] > 1e-3
    assert numeric_norms["free_source_camera"] < 1e-8

    # The retained source camera is transported with its accepted fixed
    # anchor-relative transform, and the resulting committed projection has
    # positive depth and agrees with the exact persisted image row.
    record = state_on.retained_source_observations[68]
    source68_after = state_on.keyframes[record["anchor_keyframe_id"]].pose @ record["relative_pose"]
    np.testing.assert_allclose(source68_after, state_on.poses[68], rtol=0., atol=1e-12)
    row1 = next(row for row in record["rows"] if row["landmark_id"] == 1)
    camera_point = (state_on.landmarks[1].position - source68_after[:3, 3]) @ source68_after[:3, :3]
    assert camera_point[2] > 0.
    error_on = _retained_pixel_error(state_on, 68, 1)
    error_off = _retained_pixel_error(state_off, 68, 1)
    assert error_on < error_off


def test_retained_rows_are_skipped_when_the_same_actual_frame_is_a_free_owned_camera():
    state, source68, truth_points, _anchor_id, _target_id = _long_frame_scene()
    first, _ = _run_first_bundle(state, source68, truth_points, retain=True)
    assert first["applied"] is True, first
    before_registry = copy.deepcopy(state.retained_source_observations)

    # The current owned factor references landmark 0 at frame 68. Cached rows
    # are landmarks 1..23, so exact (frame, landmark) deduplication cannot catch
    # this case; the camera endpoint is itself a free C variable in this solve.
    report, _captured = _run_future_bundle(
        state, 68, source68, truth_points, retain=True
    )
    assert report["applied"] is True, report
    assert report["owned_stereo_image_bundle"]["optimized_intermediate_frames"] == [68]
    retained = report["retained_source_observations"]
    assert retained["installed_rows"] == 0
    assert retained["components_added"] == 0
    assert state.retained_source_observations.keys() == before_registry.keys()
    for frame, old_record in before_registry.items():
        new_record = state.retained_source_observations[frame]
        assert new_record["anchor_keyframe_id"] == old_record["anchor_keyframe_id"]
        np.testing.assert_allclose(
            new_record["relative_pose"], state.relative_poses[frame],
            rtol=0., atol=1e-12,
        )
        np.testing.assert_allclose(
            state.keyframes[new_record["anchor_keyframe_id"]].pose
            @ new_record["relative_pose"],
            state.poses[frame], rtol=0., atol=1e-12,
        )
        assert len(new_record["rows"]) == len(old_record["rows"])
        for new_row, old_row in zip(new_record["rows"], old_record["rows"]):
            assert new_row["landmark_id"] == old_row["landmark_id"]
            np.testing.assert_array_equal(
                new_row["pixel_float32"], old_row["pixel_float32"]
            )


def test_map_commit_owns_actual_source_frame_registry_and_rejects_partial_or_stale_updates():
    state, _truth, _single_id, source_frame = _state_with_source_frame()
    landmark_id = 0
    source_pixel = np.array([211.25, 126.5], dtype=np.float32)
    record = _retained_record(state, source_frame, landmark_id, source_pixel)
    prospective = {source_frame: record}
    corrected = {key: value.pose.copy() for key, value in state.keyframes.items()}

    assert state.retained_source_observations == {}
    assert state.apply_corrections(
        state.revision, corrected, retained_source_updates=prospective,
        retained_source_limits={"max_frames": 1, "max_rows_per_frame": 1},
    )
    stored = state.retained_source_observations[source_frame]
    assert source_frame == 3
    assert source_frame not in {frame.frame for frame in state.keyframes.values()}
    assert stored["measurement_role"] == "tracking_fit_consumed"
    assert stored["anchor_keyframe_id"] == state.pose_anchors[source_frame]
    assert stored["anchor_keyframe_id"] in state.keyframes
    np.testing.assert_array_equal(stored["rows"][0]["pixel_float32"], source_pixel)

    # The registry is atomically committed and owns every mutable array.
    committed = _map_snapshot(state)
    source_pixel[:] = -1
    record["relative_pose"][:] = np.nan
    prospective[source_frame]["rows"][0]["pixel_float32"][:] = -2
    _assert_map_snapshot(state, committed)

    # A stale commit may not replace an already accepted registry.
    replacement = copy.deepcopy(state.retained_source_observations)
    replacement[source_frame]["rows"][0]["pixel_float32"][:] = [1, 2]
    assert not state.apply_corrections(
        state.revision - 1, corrected, retained_source_updates=replacement,
        retained_source_limits={"max_frames": 1, "max_rows_per_frame": 1},
    )
    _assert_map_snapshot(state, committed)


@pytest.mark.parametrize("invalid", [
    "duplicate_id", "duplicate_pixel", "unknown_landmark", "lost_frame",
    "bad_calibration", "bad_pose", "negative_frame_key", "bool_frame_key",
    "frame_mismatch", "negative_accepted_epoch", "bool_accepted_epoch",
    "future_accepted_epoch", "negative_geometry_epoch", "bool_geometry_epoch",
    "future_geometry_epoch", "too_many_rows", "too_many_frames", "missing_limits",
    "bool_frame_limit", "negative_frame_limit", "bool_row_limit", "zero_row_limit",
])
def test_invalid_retained_registry_batch_is_atomic(invalid):
    state, _truth, _single_id, frame = _state_with_source_frame()
    valid = _retained_record(state, frame, 0, np.array([12.5, 15.25], np.float32))
    records = {frame: valid}
    if invalid in ("duplicate_id", "duplicate_pixel"):
        second = _retained_record(
            state, frame,
            0 if invalid == "duplicate_id" else 1,
            np.array([42.0, 31.0], np.float32)
            if invalid == "duplicate_id" else np.array([12.5, 15.25], np.float32),
        )
        records[frame]["rows"].append(second["rows"][0])
    elif invalid == "unknown_landmark":
        records[frame]["rows"][0]["landmark_id"] = 99999
    elif invalid == "lost_frame":
        state.statuses[frame] = "lost"
    elif invalid == "bad_calibration":
        records[frame]["calibration_identity"] = ""
    elif invalid == "bad_pose":
        records[frame]["relative_pose"][0, 3] = np.nan
    elif invalid == "negative_frame_key":
        record = records.pop(frame)
        record["frame_id"] = -1
        records[-1] = record
    elif invalid == "bool_frame_key":
        record = records.pop(frame)
        record["frame_id"] = True
        records[True] = record
    elif invalid == "frame_mismatch":
        records[frame]["frame_id"] = frame + 1
    elif invalid == "negative_accepted_epoch":
        records[frame]["accepted_revision"] = -1
    elif invalid == "bool_accepted_epoch":
        records[frame]["accepted_revision"] = True
    elif invalid == "future_accepted_epoch":
        records[frame]["accepted_revision"] = state.revision + 2
    elif invalid == "negative_geometry_epoch":
        records[frame]["accepted_geometry_revision"] = -1
    elif invalid == "bool_geometry_epoch":
        records[frame]["accepted_geometry_revision"] = False
    elif invalid == "future_geometry_epoch":
        records[frame]["accepted_geometry_revision"] = state.geometry_revision + 2
    limits = {"max_frames": 1, "max_rows_per_frame": 1}
    if invalid == "too_many_rows":
        records[frame]["rows"].append({
            "landmark_id": 1,
            "pixel_float32": np.array([90.0, 72.0], dtype=np.float32),
        })
    elif invalid == "too_many_frames":
        second_pose = state.keyframes[1].pose @ _pose(0.05, (0.01, 0.02, -0.015))
        state.record(second_pose, "tracking", 1)
        second_frame = len(state.poses) - 1
        records[second_frame] = _retained_record(
            state, second_frame, 1, np.array([84.0, 56.0], dtype=np.float32)
        )
    elif invalid == "missing_limits":
        limits = None
    elif invalid == "bool_frame_limit":
        limits["max_frames"] = True
    elif invalid == "negative_frame_limit":
        limits["max_frames"] = -1
    elif invalid == "bool_row_limit":
        limits["max_rows_per_frame"] = False
    elif invalid == "zero_row_limit":
        limits["max_rows_per_frame"] = 0

    before = _map_snapshot(state)
    corrected = {key: value.pose.copy() for key, value in state.keyframes.items()}
    with pytest.raises((ValueError, TypeError)):
        state.apply_corrections(
            state.revision, corrected, retained_source_updates=records,
            retained_source_limits=limits,
        )
    _assert_map_snapshot(state, before)
