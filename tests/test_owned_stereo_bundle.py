"""Integration gates for appending owned stereo image rows to local BA."""

import numpy as np
from dataclasses import replace
from types import SimpleNamespace

import local_bundle as bundle
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from stereo_training_factors import StereoTrainingFactor
from test_bundle_diagnostics import MATRIX, BASELINE, _scene


OFFSET = 3.25


def _factor(*, source_frame, target_frame, source_anchor, target_anchor,
            source_pose, target_pose, source_anchor_pose, target_anchor_pose,
            source_pixel, target_pixel, source_right, target_right,
            landmark_id, source_landmark_id, target_landmark_id,
            source_owned, target_owned, point, epoch):
    return StereoTrainingFactor(
        source_frame=source_frame, target_frame=target_frame,
        source_anchor=source_anchor, target_anchor=target_anchor,
        source_anchor_relative=np.linalg.inv(source_anchor_pose) @ source_pose,
        target_anchor_relative=np.linalg.inv(target_anchor_pose) @ target_pose,
        source_pose=source_pose, target_pose=target_pose,
        source_anchor_pose=source_anchor_pose, target_anchor_pose=target_anchor_pose,
        source_pixel=source_pixel, target_pixel=target_pixel,
        source_right_u=source_right, target_right_u=target_right,
        source_row=0, target_row=0, source_alias_rows=(0,), target_alias_rows=(0,),
        forward_role=True, reverse_role=True,
        source_landmark_id=source_landmark_id, target_landmark_id=target_landmark_id,
        reused_landmark_id=landmark_id,
        source_has_existing_observation=source_owned,
        target_has_existing_observation=target_owned,
        point_initial=point, matrix=MATRIX, baseline=BASELINE,
        disparity_offset=OFFSET, image_size=(640, 480),
        calibration_identity="synthetic-calibration-v1", source_epoch=epoch,
        fit_source="reserved_supported_training_rows", fit_depth_policy="supported_raw",
        fit_partition="reserved_training", heldout_status="excluded_external_arbitration_rows",
        physical_identity=("12" * 32),
    )


def _setup():
    state, truth, single_id = _scene()
    # In this small MapState fixture, incoming frame IDs equal pose-list indices.
    for keyframe_id, keyframe in state.keyframes.items():
        keyframe.frame = keyframe_id
    # Frame 3 is a non-keyframe observation transported from keyframe 2.
    state.record(state.poses[2].copy(), "tracking", 2)
    return state, truth, single_id


def _provider_for(state, truth, single_id, calls, *, duplicate=False, fail_validate=False,
                  mono_collision=False):
    def provider(payload):
        phase = payload.get("training_factor_phase", "prepare")
        calls.append(phase)
        if phase == "validate":
            return (), ({"status": "rejected", "reason": "test_stale_owner"}
                        if fail_validate else {"status": "validated", "reason": None})
        selected = payload["selection"]["selected_landmarks"]
        selected_item = next(item for item in selected if (
            (state.landmarks[int(item["landmark_id"])].observations[1].right_u is None)
            if mono_collision else
            (state.landmarks[int(item["landmark_id"])].observations[1].right_u is not None)
        ))
        selected_id = int(selected_item["landmark_id"])
        selected_point = np.asarray(selected_item["initial_world_position"], float)
        source_obs = state.landmarks[selected_id].observations[1]
        source_pose = state.poses[1].copy()
        target_pose = state.poses[3].copy()
        target_pixel, target_depth = project(selected_point[None], truth[2], MATRIX)
        target_right = right_pixel(target_pixel[0, 0], target_depth[0],
                                   MATRIX[0, 0], BASELINE, OFFSET)
        first = _factor(
            source_frame=1, target_frame=3, source_anchor=1, target_anchor=2,
            source_pose=source_pose, target_pose=target_pose,
            source_anchor_pose=state.keyframes[1].pose.copy(),
            target_anchor_pose=state.keyframes[2].pose.copy(),
            source_pixel=source_obs.pixel, target_pixel=target_pixel[0],
            source_right=(target_right if mono_collision else source_obs.right_u), target_right=target_right,
            landmark_id=selected_id, source_landmark_id=selected_id,
            target_landmark_id=-1, source_owned=True, target_owned=False,
            point=selected_point, epoch=(state.revision, state.geometry_revision),
        )

        singleton = state.landmarks[single_id]
        singleton_point = singleton.position.copy()
        src_pixel, src_depth = project(singleton_point[None], truth[1], MATRIX)
        src_right = right_pixel(src_pixel[0, 0], src_depth[0],
                                MATRIX[0, 0], BASELINE, OFFSET)
        target_obs = singleton.observations[2]
        second = _factor(
            source_frame=1, target_frame=2, source_anchor=1, target_anchor=2,
            source_pose=state.poses[1].copy(), target_pose=state.poses[2].copy(),
            source_anchor_pose=state.keyframes[1].pose.copy(),
            target_anchor_pose=state.keyframes[2].pose.copy(),
            source_pixel=src_pixel[0], target_pixel=target_obs.pixel,
            source_right=src_right, target_right=target_obs.right_u,
            landmark_id=single_id, source_landmark_id=-1,
            target_landmark_id=single_id, source_owned=False, target_owned=True,
            point=singleton_point, epoch=(state.revision, state.geometry_revision),
        )
        factors = (first, second)
        if mono_collision:
            factors = (first,)
        elif duplicate:
            factors = (*factors, first, second)
        return factors, {
            "status": "accepted", "selected_factors": 2,
            "model": "correlated_physical_stereo_image_rows_v1",
            "covariance_claim": False, "heldout_validation_claim": False,
            "intentionally_reuses_sensor_evidence": True,
        }
    return provider


def test_actual_scipy_joint_bundle_reuses_selected_point_and_optimizes_single_view_once():
    state, truth, single_id = _setup()
    calls = []
    before_single = state.landmarks[single_id].position.copy()
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(state, truth, single_id, calls),
    )
    assert calls == ["prepare", "validate"]
    assert report["applied"] is True
    bundle_report = report["owned_stereo_image_bundle"]
    assert bundle_report["status"] == "accepted"
    assert bundle_report["selected_factors"] == 2
    assert bundle_report["optimized_single_view_points"] == 1
    assert bundle_report["unique_image_rows_added"] == 3
    assert bundle_report["affected_objective_reduced"] is True
    assert bundle_report["augmented_objective_reduced"] is True
    assert report["independent_stereo_motion_checks"] == 1
    # The singleton is updated by the augmented solve, not by a second anchor propagation.
    assert not np.array_equal(state.landmarks[single_id].position, before_single)
    assert report["optimized_landmarks"] == 21
    assert report["anchor_propagated_single_view_landmarks"] == 0


def test_optimized_single_view_position_is_applied_once_not_propagated_again(monkeypatch):
    state, truth, single_id = _setup()
    calls = []
    solver = bundle.least_squares
    candidate = {}

    def capture_candidate(*args, **kwargs):
        result = solver(*args, **kwargs)
        candidate["x"] = result.x.copy()
        return result

    monkeypatch.setattr(bundle, "least_squares", capture_candidate)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(state, truth, single_id, calls),
    )
    assert report["applied"] is True
    np.testing.assert_allclose(state.landmarks[single_id].position, candidate["x"][-3:],
                               rtol=0., atol=1e-12)


def test_repeated_physical_factor_rows_are_collapsed_before_solver():
    canonical, truth, single_id = _setup()
    repeated, _, _ = _setup()
    canonical_report = local_bundle_adjustment(
        canonical, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(canonical, truth, single_id, []),
    )
    repeated_report = local_bundle_adjustment(
        repeated, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(repeated, truth, single_id, [], duplicate=True),
    )
    assert canonical_report["applied"] and repeated_report["applied"]
    assert canonical_report["owned_stereo_image_bundle"]["unique_image_rows_added"] == 3
    assert repeated_report["owned_stereo_image_bundle"]["unique_image_rows_added"] == 3
    assert canonical_report["affected_final_cost"] == repeated_report["affected_final_cost"]
    assert canonical_report["augmented_final_cost"] == repeated_report["augmented_final_cost"]
    for key in canonical.keyframes:
        np.testing.assert_array_equal(canonical.keyframes[key].pose, repeated.keyframes[key].pose)
    for key in canonical.landmarks:
        np.testing.assert_array_equal(canonical.landmarks[key].position,
                                      repeated.landmarks[key].position)


def test_final_owner_validation_failure_leaves_all_geometry_unchanged():
    state, truth, single_id = _setup()
    before_poses = [pose.copy() for pose in state.poses]
    before_keyframes = {key: value.pose.copy() for key, value in state.keyframes.items()}
    before_points = {key: value.position.copy() for key, value in state.landmarks.items()}
    revision = state.revision
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(
            state, truth, single_id, [], fail_validate=True
        ),
    )
    assert report["applied"] is False
    assert report["reason"] == "owned_stereo_final_validation_failed"
    assert report["owned_stereo_image_bundle"]["status"] == "rejected"
    assert state.revision == revision
    for actual, expected in zip(state.poses, before_poses):
        np.testing.assert_array_equal(actual, expected)
    for key in before_keyframes:
        np.testing.assert_array_equal(state.keyframes[key].pose, before_keyframes[key])
    for key in before_points:
        np.testing.assert_array_equal(state.landmarks[key].position, before_points[key])


def test_existing_left_only_pixel_cannot_be_counted_again_as_stereo_row():
    state, truth, single_id = _setup()
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=_provider_for(
            state, truth, single_id, [], mono_collision=True
        ),
    )
    assert report["owned_stereo_image_bundle"]["status"] == "rejected"
    assert report["owned_stereo_image_bundle"]["reason"] == (
        "training_factor_conflicts_with_original_observation"
    )


def test_augmented_improvement_cannot_override_worsened_original_objective(monkeypatch):
    state, truth, single_id = _setup()
    before_poses = [pose.copy() for pose in state.poses]
    before_points = {key: value.position.copy() for key, value in state.landmarks.items()}
    calls = []
    provider = _provider_for(state, truth, single_id, calls)
    old_row_count = {}

    def recording_provider(payload):
        phase = payload.get("training_factor_phase", "prepare")
        if phase == "validate":
            return provider(payload)
        old_row_count["rows"] = (
            sum(int(row["dimensions"]) for row in payload["selected_observations"])
            + sum(int(row["dimensions"])
                  for row in payload["excluded_multiview_observations"])
        )
        factors, report = provider(payload)
        biased = factors[1]
        source_pixel = biased.source_pixel.copy()
        source_pixel[0] -= 30.
        biased = replace(biased, source_pixel=source_pixel,
                         source_right_u=biased.source_right_u - 30.)
        return (factors[0], biased), report

    def huber_cost(values):
        absolute = np.abs(values)
        return float(np.sum(np.where(absolute <= 2.0, 0.5 * values * values,
                                    2.0 * (absolute - 1.0))))

    def adversarial_solver(fun, initial, **kwargs):
        old_count = old_row_count["rows"]
        at_start = fun(initial)
        old_start = huber_cost(at_start[:old_count])
        full_start = huber_cost(at_start)
        # Camera 1 is the first free pose; world-x translation is parameter +3.
        offset = 3
        candidate = None
        for distance in np.linspace(1e-5, .001, 200):
            trial = initial.copy()
            trial[offset] += distance
            residuals = fun(trial)
            if (huber_cost(residuals[:old_count]) > old_start + 1e-8
                    and huber_cost(residuals) < full_start - 1e-8):
                candidate = trial
                break
        assert candidate is not None, "fixture must expose the independent original-cost veto"
        return SimpleNamespace(x=candidate, nfev=1, success=True, status=1,
                               message="controlled admissible augmented decrease",
                               cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", adversarial_solver)
    report = local_bundle_adjustment(
        state, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET, training_factor_provider=recording_provider,
    )
    assert report["applied"] is False
    assert report["reason"] == "affected_observations_worsened"
    summary = report["owned_stereo_image_bundle"]
    assert summary["augmented_objective_reduced"] is True
    assert summary["affected_objective_reduced"] is False
    assert "validate" not in calls
    for actual, expected in zip(state.poses, before_poses):
        np.testing.assert_array_equal(actual, expected)
    for key in before_points:
        np.testing.assert_array_equal(state.landmarks[key].position, before_points[key])


def test_empty_provider_keeps_legacy_solver_state_and_acceptance(monkeypatch):
    expected, _, _ = _setup()
    actual, _, _ = _setup()
    expected_result = local_bundle_adjustment(
        expected, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
    )
    actual_result = local_bundle_adjustment(
        actual, MATRIX, BASELINE, window=3, max_landmarks=20,
        disparity_offset=OFFSET,
        training_factor_provider=lambda _payload: ((), {"status": "rejected", "reason": "none"}),
    )
    assert expected_result["applied"] == actual_result["applied"]
    for name in ("initial_cost", "final_cost", "affected_initial_cost", "affected_final_cost"):
        assert expected_result[name] == actual_result[name]
    for key in expected.keyframes:
        np.testing.assert_array_equal(expected.keyframes[key].pose, actual.keyframes[key].pose)
    for key in expected.landmarks:
        np.testing.assert_array_equal(expected.landmarks[key].position, actual.landmarks[key].position)

