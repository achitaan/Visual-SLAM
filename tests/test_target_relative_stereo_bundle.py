"""Regression checks for the bounded target-relative stereo chart."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from test_bundle_diagnostics import MATRIX, BASELINE
from test_free_source_stereo_bundle import (
    OFFSET,
    _free_source_case,
    _params_to_pose,
    _pose_to_params,
    _provider,
)
from test_owned_stereo_bundle import _factor


def _pose(x, rotvec=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    pose[0, 3] = float(x)
    return pose


def _huber2(values):
    absolute = np.abs(np.asarray(values, float))
    return float(np.sum(np.where(absolute <= 2.0, 0.5 * absolute**2,
                                2.0 * (absolute - 1.0))))


def _old_row_count(payload):
    return (
        sum(int(row["dimensions"]) for row in payload["selected_observations"])
        + sum(int(row["dimensions"])
              for row in payload["excluded_multiview_observations"])
    )


def _selected_shared_factor(state, truth_source, payload):
    selected = payload["selection"]["selected_landmarks"]
    selected_id = next(
        int(item["landmark_id"])
        for item in selected
        if state.landmarks[int(item["landmark_id"])].observations[2].right_u is not None
    )
    landmark = state.landmarks[selected_id]
    point = landmark.position.copy()
    source_pixel, source_depth = project(point[None], truth_source, MATRIX)
    source_right = right_pixel(source_pixel[0, 0], source_depth[0],
                               MATRIX[0, 0], BASELINE, OFFSET)
    target_observation = landmark.observations[2]
    return _factor(
        source_frame=3,
        target_frame=2,
        source_anchor=int(state.pose_anchors[3]),
        target_anchor=2,
        source_pose=state.poses[3].copy(),
        target_pose=state.poses[2].copy(),
        source_anchor_pose=state.keyframes[int(state.pose_anchors[3])].pose.copy(),
        target_anchor_pose=state.keyframes[2].pose.copy(),
        source_pixel=source_pixel[0].astype(np.float32),
        target_pixel=target_observation.pixel.copy(),
        source_right=float(source_right),
        target_right=target_observation.right_u,
        landmark_id=selected_id,
        source_landmark_id=-1,
        target_landmark_id=selected_id,
        source_owned=False,
        target_owned=True,
        point=point,
        epoch=(state.revision, state.geometry_revision),
    )


def test_target_relative_singletons_match_world_projection_and_ignore_target_world_pose(monkeypatch):
    state, _truth, true_source, landmark_ids = _free_source_case(source_bias=0.23)
    # Exercise noncommuting rotations and a nonzero disparity offset.
    target_pose = _pose(1.08, (0.13, -0.075, 0.045))
    state.keyframes[2].pose = target_pose.copy()
    state.poses[2] = target_pose.copy()

    # The current map point is authoritative even when its retained image rows
    # would triangulate somewhere else.
    changed_id = landmark_ids[0]
    state.landmarks[changed_id].position += np.array([0.24, -0.16, 0.31])
    authoritative = {ident: state.landmarks[ident].position.copy() for ident in landmark_ids}
    target_initial = target_pose.copy()
    prepared = {}
    provided_factors = {}
    provider = _provider(state, _truth, true_source, landmark_ids)

    def capture_provider(payload):
        if payload.get("training_factor_phase", "prepare") == "prepare":
            prepared["payload"] = payload
        factors, report = provider(payload)
        if payload.get("training_factor_phase", "prepare") == "prepare":
            provided_factors["items"] = factors
        return factors, report

    saved = {}

    def inspect_solver(fun, initial, **kwargs):
        initial = np.asarray(initial, float).copy()
        payload = prepared["payload"]
        layout = payload["parameter_layout"]
        original_count = len(layout["initial_vector"])
        q_offsets = {ident: original_count + 3 * index
                     for index, ident in enumerate(sorted(landmark_ids))}
        camera_offset = original_count + 3 * len(landmark_ids)
        target_offset = layout["pose_offsets"]["2"]
        old_rows = _old_row_count(payload)

        # The chart adds one q per singleton and one C for the free source.
        assert len(initial) == original_count + 3 * len(landmark_ids) + 6
        chart_report = {}
        assert initial[camera_offset:camera_offset + 6].shape == (6,)
        initial_residual = fun(initial)
        pair_stop = old_rows + 6 * len(landmark_ids)
        pair_rows = initial_residual[old_rows:pair_stop]
        assert pair_rows.size == 6 * len(landmark_ids)

        target_transform = np.eye(4)
        target_transform[:3, :3] = Rotation.from_rotvec(
            initial[target_offset:target_offset + 3]
        ).as_matrix()
        target_transform[:3, 3] = initial[target_offset + 3:target_offset + 6]
        source_chart = _params_to_pose(initial[camera_offset:camera_offset + 6])
        for index, ident in enumerate(sorted(landmark_ids)):
            q = initial[q_offsets[ident]:q_offsets[ident] + 3]
            expected_q = (np.linalg.inv(target_initial)
                          @ np.r_[authoritative[ident], 1.0])[:3]
            np.testing.assert_allclose(q, expected_q, rtol=0.0, atol=1e-11)
            world_point = (target_transform @ np.r_[q, 1.0])[:3]
            world_source = target_transform @ source_chart
            factor = provided_factors["items"][index]

            predicted_source, depth_source = project(world_point[None], world_source, MATRIX)
            predicted_source_right = right_pixel(
                predicted_source[0, 0], depth_source[0], MATRIX[0, 0], BASELINE, OFFSET
            )
            expected_source = np.r_[
                predicted_source[0] - factor.source_pixel,
                predicted_source_right - factor.source_right_u,
            ]
            predicted_target, depth_target = project(world_point[None], target_transform, MATRIX)
            predicted_target_right = right_pixel(
                predicted_target[0, 0], depth_target[0], MATRIX[0, 0], BASELINE, OFFSET
            )
            expected_target = np.r_[
                predicted_target[0] - factor.target_pixel,
                predicted_target_right - factor.target_right_u,
            ]
            row = old_rows + 6 * index
            np.testing.assert_allclose(initial_residual[row:row + 3], expected_source,
                                       rtol=0.0, atol=2e-8)
            np.testing.assert_allclose(initial_residual[row + 3:row + 6], expected_target,
                                       rtol=0.0, atol=2e-8)

        # Holding C and q fixed while moving T changes the world realization of
        # the whole pair, but the direct chart residual rows remain bitwise equal.
        moved = initial.copy()
        moved[target_offset:target_offset + 6] = _pose_to_params(
            _pose(1.14, (0.18, -0.03, 0.09))
        )
        moved_residual = fun(moved)
        np.testing.assert_array_equal(initial_residual[old_rows:pair_stop],
                                      moved_residual[old_rows:pair_stop])
        assert _huber2(initial_residual[old_rows:pair_stop]) == _huber2(
            moved_residual[old_rows:pair_stop]
        )

        sparse = kwargs["jac_sparsity"].toarray().astype(bool)
        first_source = old_rows
        first_target = old_rows + 3
        q0 = q_offsets[sorted(landmark_ids)[0]]
        assert sparse[first_source:first_source + 3, camera_offset:camera_offset + 6].all()
        assert sparse[first_source:first_source + 3, q0:q0 + 3].all()
        assert not sparse[first_source:first_source + 3, target_offset:target_offset + 6].any()
        assert sparse[first_target:first_target + 3, q0:q0 + 3].all()
        assert not sparse[first_target:first_target + 3, camera_offset:camera_offset + 6].any()
        assert not sparse[first_target:first_target + 3, target_offset:target_offset + 6].any()

        saved.update(initial=initial, old_rows=old_rows, q_offsets=q_offsets,
                     camera_offset=camera_offset, target_offset=target_offset,
                     pair_stop=pair_stop, sparse=sparse)
        return SimpleNamespace(x=initial, nfev=1, success=True, status=1,
                               message="residual inspection", cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET, training_factor_provider=capture_provider,
    )
    summary = report["owned_stereo_image_bundle"]
    assert summary["coordinate_chart"] == "target_relative_intermediate_and_singleton_v1"
    assert summary["optimized_single_view_points"] == len(landmark_ids)
    assert summary["target_relative_camera_reference_keyframes"] == {"3": 2}
    for ident in landmark_ids:
        assert summary["target_relative_point_reference_keyframes"][str(ident)] == 2


def test_shared_source_row_retains_target_camera_coupling_and_declares_numeric_sparsity(monkeypatch):
    state, truth, true_source, singleton_ids = _free_source_case(source_bias=0.19)
    target_pose = _pose(1.08, (0.095, -0.06, 0.035))
    state.keyframes[2].pose = target_pose.copy()
    state.poses[2] = target_pose.copy()
    prepared = {}
    factors_seen = {}
    singleton_provider = _provider(state, truth, true_source, singleton_ids)

    def provider(payload):
        phase = payload.get("training_factor_phase", "prepare")
        if phase == "validate":
            return (), {"status": "validated", "reason": None}
        prepared["payload"] = payload
        factors, report = singleton_provider(payload)
        shared = _selected_shared_factor(state, true_source, payload)
        factors_seen["items"] = (*factors, shared)
        return factors_seen["items"], {**report, "selected_factors": len(factors) + 1}

    def inspect_solver(fun, initial, **kwargs):
        initial = np.asarray(initial, float).copy()
        layout = prepared["payload"]["parameter_layout"]
        old_rows = _old_row_count(prepared["payload"])
        original_count = len(layout["initial_vector"])
        q_count = len(singleton_ids)
        camera_offset = original_count + 3 * q_count
        target_offset = layout["pose_offsets"]["2"]
        shared = factors_seen["items"][-1]
        selected = prepared["payload"]["selection"]["selected_landmarks"]
        shared_rank = next(int(item["rank"]) for item in selected
                           if int(item["landmark_id"]) == shared.reused_landmark_id)
        point_offset = layout["point_offset"] + 3 * shared_rank
        shared_rows = slice(old_rows + 6 * q_count, old_rows + 6 * q_count + 3)
        initial_residual = fun(initial)
        assert initial_residual[shared_rows].size == 3

        moved_target = initial.copy()
        moved_target[target_offset:target_offset + 6] = _pose_to_params(
            _pose(1.11, (0.12, -0.035, 0.07))
        )
        target_moved_residual = fun(moved_target)
        np.testing.assert_array_equal(
            initial_residual[old_rows:old_rows + 6 * q_count],
            target_moved_residual[old_rows:old_rows + 6 * q_count],
        )
        assert not np.array_equal(initial_residual[shared_rows],
                                  target_moved_residual[shared_rows])

        sparse = kwargs["jac_sparsity"].toarray().astype(bool)
        # Independently finite-difference each physical block on the shared
        # source row. A missing T entry would omit real image information.
        for start, width in ((target_offset, 6), (camera_offset, 6), (point_offset, 3)):
            block_derivatives = []
            for column in range(start, start + width):
                plus, minus = initial.copy(), initial.copy()
                plus[column] += 1e-6
                minus[column] -= 1e-6
                derivative = (fun(plus)[shared_rows] - fun(minus)[shared_rows]) / 2e-6
                block_derivatives.append(derivative)
                assert np.any(np.abs(derivative) > 1e-5)
                assert sparse[shared_rows, column].all(), (
                    f"shared source derivative at column {column} is absent from jac_sparsity"
                )
            assert np.linalg.norm(np.column_stack(block_derivatives)) > 0.0

        # The declared sparse graph must contain q/C for pair-only rows and no
        # target block; central differences verify no hidden target derivative.
        q_offset = original_count
        source_rows = slice(old_rows, old_rows + 3)
        target_rows = slice(old_rows + 3, old_rows + 6)
        assert sparse[source_rows, q_offset:q_offset + 3].all()
        assert sparse[source_rows, camera_offset:camera_offset + 6].all()
        assert not sparse[source_rows, target_offset:target_offset + 6].any()
        assert sparse[target_rows, q_offset:q_offset + 3].all()
        assert not sparse[target_rows, camera_offset:camera_offset + 6].any()
        assert not sparse[target_rows, target_offset:target_offset + 6].any()

        # Applying one global world transform to T and the shared world point,
        # while keeping C and q fixed, preserves every appended image residual.
        global_transform = _pose(-0.37, (0.17, -0.11, 0.08))
        transformed = initial.copy()
        target = _params_to_pose(initial[target_offset:target_offset + 6])
        transformed[target_offset:target_offset + 6] = _pose_to_params(
            global_transform @ target
        )
        transformed[point_offset:point_offset + 3] = (
            global_transform[:3, :3] @ initial[point_offset:point_offset + 3]
            + global_transform[:3, 3]
        )
        np.testing.assert_allclose(fun(transformed)[old_rows:],
                                   initial_residual[old_rows:], rtol=1e-9, atol=2e-8)
        return SimpleNamespace(x=initial, nfev=1, success=True, status=1,
                               message="sparsity inspection", cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", inspect_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET, training_factor_provider=provider,
    )
    summary = report["owned_stereo_image_bundle"]
    assert summary["coordinate_chart"] == "target_relative_intermediate_and_singleton_v1"
    assert summary["reused_selected_points"] == 1


def test_target_relative_chart_rejects_a_non_keyframe_target_without_variable_duplication(monkeypatch):
    state, truth, true_source, singleton_ids = _free_source_case()
    base_provider = _provider(state, truth, true_source, singleton_ids)
    prepared = {}

    def non_keyframe_target_provider(payload):
        phase = payload.get("training_factor_phase", "prepare")
        if phase == "validate":
            return (), {"status": "validated", "reason": None}
        prepared["payload"] = payload
        factors, provider_report = base_provider(payload)
        source_factor = factors[0]
        point = state.landmarks[source_factor.reused_landmark_id].position
        target_pixel, target_depth = project(point[None], state.poses[3], MATRIX)
        target_right = right_pixel(target_pixel[0, 0], target_depth[0],
                                   MATRIX[0, 0], BASELINE, OFFSET)
        invalid = replace(
            source_factor,
            source_frame=2,
            target_frame=3,
            source_anchor=2,
            target_anchor=int(state.pose_anchors[3]),
            source_anchor_relative=np.eye(4),
            target_anchor_relative=(
                np.linalg.inv(state.keyframes[int(state.pose_anchors[3])].pose)
                @ state.poses[3]
            ),
            source_pose=state.poses[2].copy(),
            target_pose=state.poses[3].copy(),
            source_anchor_pose=state.keyframes[2].pose.copy(),
            target_anchor_pose=state.keyframes[int(state.pose_anchors[3])].pose.copy(),
            source_pixel=state.landmarks[source_factor.reused_landmark_id].observations[2].pixel.copy(),
            target_pixel=target_pixel[0].astype(np.float32),
            source_right_u=state.landmarks[source_factor.reused_landmark_id].observations[2].right_u,
            target_right_u=float(target_right),
            source_landmark_id=source_factor.reused_landmark_id,
            target_landmark_id=-1,
            source_has_existing_observation=True,
            target_has_existing_observation=False,
        )
        return (invalid,), provider_report

    seen = {}

    def capture_solver(fun, initial, **kwargs):
        seen["size"] = len(initial)
        seen["baseline_size"] = len(prepared["payload"]["parameter_layout"]["initial_vector"])
        fun(initial)
        return SimpleNamespace(x=np.asarray(initial).copy(), nfev=1, success=True, status=1,
                               message="fallback inspection", cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", capture_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET, training_factor_provider=non_keyframe_target_provider,
    )
    summary = report["owned_stereo_image_bundle"]
    assert summary["status"] == "rejected"
    assert summary["reason"] == "target_relative_reference_must_be_keyframe"
    assert seen["size"] == seen["baseline_size"]


def _source_frame_target_factor(state, truth_source, truth, point_id, target_frame,
                                payload, *, source_frame=3, source_anchor=None,
                                source_true_pose=None):
    point = state.landmarks[point_id].position.copy()
    if source_anchor is None:
        source_anchor = int(state.pose_anchors[source_frame])
    if source_true_pose is None:
        source_true_pose = truth_source
    source_pixel, source_depth = project(point[None], source_true_pose, MATRIX)
    target_pixel, target_depth = project(point[None], truth[target_frame], MATRIX)
    return replace(
        _factor(
            source_frame=source_frame,
            target_frame=target_frame,
            source_anchor=source_anchor,
            target_anchor=target_frame,
            source_pose=state.poses[source_frame].copy(),
            target_pose=state.poses[target_frame].copy(),
            source_anchor_pose=state.keyframes[source_anchor].pose.copy(),
            target_anchor_pose=state.keyframes[target_frame].pose.copy(),
            source_pixel=source_pixel[0].astype(np.float32),
            target_pixel=target_pixel[0].astype(np.float32),
            source_right=float(right_pixel(source_pixel[0, 0], source_depth[0],
                                           MATRIX[0, 0], BASELINE, OFFSET)),
            target_right=float(right_pixel(target_pixel[0, 0], target_depth[0],
                                           MATRIX[0, 0], BASELINE, OFFSET)),
            landmark_id=point_id,
            source_landmark_id=-1,
            target_landmark_id=-1,
            source_owned=False,
            target_owned=False,
            point=point,
            epoch=(state.revision, state.geometry_revision),
        ),
        physical_identity="ab" * 32,
    )


def _assert_chart_falls_back_without_chart_variables(monkeypatch, state, provider, expected_reason):
    before_poses = [pose.copy() for pose in state.poses]
    before_keyframes = {key: frame.pose.copy() for key, frame in state.keyframes.items()}
    before_points = {key: landmark.position.copy() for key, landmark in state.landmarks.items()}
    before_revision = state.revision
    prepared = {}
    base_provider = provider

    def capture_provider(payload):
        if payload.get("training_factor_phase", "prepare") == "prepare":
            prepared["payload"] = payload
        return base_provider(payload)

    seen = {}

    def original_solver(fun, initial, **kwargs):
        base_size = len(prepared["payload"]["parameter_layout"]["initial_vector"])
        seen["size"] = len(initial)
        seen["sparsity_shape"] = kwargs["jac_sparsity"].shape
        assert len(initial) == base_size
        assert kwargs["jac_sparsity"].shape[1] == base_size
        fun(initial)
        return SimpleNamespace(x=np.asarray(initial).copy(), nfev=1, success=True, status=1,
                               message="original BA fallback", cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", original_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET, training_factor_provider=capture_provider,
    )
    assert report["applied"] is False
    assert report["owned_stereo_image_bundle"]["status"] == "rejected"
    assert report["owned_stereo_image_bundle"]["reason"] == expected_reason
    assert seen["size"] == seen["sparsity_shape"][1]
    assert state.revision == before_revision
    for actual, expected in zip(state.poses, before_poses):
        np.testing.assert_array_equal(actual, expected)
    for key, expected in before_keyframes.items():
        np.testing.assert_array_equal(state.keyframes[key].pose, expected)
    for key, expected in before_points.items():
        np.testing.assert_array_equal(state.landmarks[key].position, expected)


def test_conflicting_target_keyframes_for_one_free_source_fall_back_to_original_ba(monkeypatch):
    state, truth, true_source, singleton_ids = _free_source_case()
    provider = _provider(state, truth, true_source, singleton_ids)

    def with_second_target(payload):
        phase = payload.get("training_factor_phase", "prepare")
        factors, report = provider(payload)
        if phase == "validate":
            return factors, report
        extra = _source_frame_target_factor(
            state, true_source, truth, 0, 1, payload
        )
        extra = replace(extra, target_landmark_id=0, target_has_existing_observation=True)
        return (*factors, extra), {**report, "selected_factors": len(factors) + 1}

    _assert_chart_falls_back_without_chart_variables(
        monkeypatch, state, with_second_target,
        "conflicting_target_relative_source_reference",
    )


def test_one_persistent_singleton_cannot_use_two_target_keyframes(monkeypatch):
    state, truth, true_source, singleton_ids = _free_source_case()
    # A second independently tracked source camera references the same
    # persistent singleton from KF 1 while the first factor references KF 2.
    true_source_4 = _pose(1.42, (0.03, -0.02, 0.015))
    state.record(_pose(1.50, (0.03, -0.02, 0.015)), "tracking", 1)
    provider = _provider(state, truth, true_source, singleton_ids)

    def with_second_singleton_reference(payload):
        phase = payload.get("training_factor_phase", "prepare")
        factors, report = provider(payload)
        if phase == "validate":
            return factors, report
        extra = _source_frame_target_factor(
            state, true_source_4, truth, singleton_ids[0], 1, payload,
            source_frame=4, source_anchor=1, source_true_pose=true_source_4,
        )
        return (*factors, extra), {**report, "selected_factors": len(factors) + 1}

    _assert_chart_falls_back_without_chart_variables(
        monkeypatch, state, with_second_singleton_reference,
        "singleton_has_non_target_observation",
    )


def test_changing_chart_reference_pose_without_revision_change_blocks_commit(monkeypatch):
    state, truth, true_source, singleton_ids = _free_source_case(source_bias=0.18)
    before_source = state.poses[3].copy()
    before_points = {ident: state.landmarks[ident].position.copy() for ident in singleton_ids}
    real_solver = bundle.least_squares

    def mutate_reference_after_solving(fun, initial, **kwargs):
        result = real_solver(fun, initial, **kwargs)
        state.keyframes[2].pose[0, 3] += 0.002
        return result

    monkeypatch.setattr(bundle, "least_squares", mutate_reference_after_solving)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=40,
        disparity_offset=OFFSET,
        training_factor_provider=_provider(state, truth, true_source, singleton_ids),
    )
    assert report["applied"] is False
    assert report["reason"] == "stale_target_relative_reference"
    np.testing.assert_array_equal(state.poses[3], before_source)
    for ident, expected in before_points.items():
        np.testing.assert_array_equal(state.landmarks[ident].position, expected)


def test_lsmr_tolerances_apply_only_to_active_chart_and_fallback_kwargs_stay_legacy(monkeypatch):
    captures = {}

    def run_case(mode):
        state, truth, true_source, singleton_ids = _free_source_case()
        captured = {}

        def inspect_solver(fun, initial, **kwargs):
            fun(initial)
            captured["initial_size"] = len(initial)
            captured["options"] = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in kwargs.items()
            }
            if "jac_sparsity" in captured["options"]:
                captured["options"]["jac_sparsity"] = kwargs["jac_sparsity"].toarray()
            return SimpleNamespace(
                x=np.asarray(initial).copy(), nfev=1, success=True, status=1,
                message="solver-options inspection", cost=0.0, optimality=0.0,
            )

        monkeypatch.setattr(bundle, "least_squares", inspect_solver)
        options = {}
        if mode == "active":
            options["training_factor_provider"] = _provider(
                state, truth, true_source, singleton_ids
            )
        elif mode == "empty":
            options["training_factor_provider"] = lambda _payload: (
                (), {"status": "rejected", "reason": "empty_test_pool"}
            )
        elif mode == "rejected":
            options["training_factor_provider"] = lambda _payload: (
                (object(),), {"status": "accepted", "reason": None}
            )
        local_bundle_adjustment(
            state, MATRIX, BASELINE, window=3, max_landmarks=40,
            disparity_offset=OFFSET, **options,
        )
        captures[mode] = captured

    for mode in ("off", "empty", "rejected", "active"):
        run_case(mode)

    active = captures["active"]
    active_options = active["options"]
    assert active_options["tr_options"] == {
        "atol": 1e-12,
        "btol": 1e-12,
        "maxiter": max(500, active["initial_size"]),
    }
    assert active_options["tr_options"]["maxiter"] >= 500

    legacy_modes = ("off", "empty", "rejected")
    for mode in legacy_modes:
        options = captures[mode]["options"]
        assert "tr_options" not in options
        assert options.keys() == captures["off"]["options"].keys()
        assert options["loss"] == "huber"
        assert options["f_scale"] == 2.0
        assert options["max_nfev"] == 30
        assert options["tr_solver"] == "lsmr"
        np.testing.assert_array_equal(options["x_scale"], captures["off"]["options"]["x_scale"])
        np.testing.assert_array_equal(
            options["jac_sparsity"], captures["off"]["options"]["jac_sparsity"]
        )

