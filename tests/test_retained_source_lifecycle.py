"""Lifecycle regressions for bounded retained source-image observations."""

import copy

import numpy as np
import pytest

import local_bundle as bundle
from mapping_geometry import project
from test_retained_source_observations import (
    CALIBRATION_ID,
    _assert_map_snapshot,
    _prepare_future_ablation_states,
    _retained_record,
    _run_future_bundle,
)


def test_expired_calibration_and_out_of_window_rows_are_reported_and_bounded():
    state, _off, source80, _source68, truth_points = _prepare_future_ablation_states()

    # Add four older, still eligible actual-frame records. Only the three
    # newest can enter this solve's retained-source window.
    for frame, landmark_id in ((60, 3), (62, 4), (63, 5), (67, 6)):
        pixel = project(
            state.landmarks[landmark_id].position[None], state.poses[frame],
            bundle_matrix(),
        )[0][0].astype(np.float32)
        state.retained_source_observations[frame] = _retained_record(
            state, frame, landmark_id, pixel
        )

    # A well-formed but different calibration is a legitimate expiry, not a
    # malformed registry. The row record remains structurally valid.
    state.retained_source_observations[68]["calibration_identity"] = (
        "retained-source-test-calibration-previous-v0"
    )

    report, _captured = _run_future_bundle(
        state, 80, source80, truth_points, retain=True
    )
    assert report["applied"] is True, report
    retention = report["retained_source_observations"]
    assert retention["expired_rows"] == 24  # 23 stale-calibration + 1 out-of-window
    assert retention["registered_frames_after_commit"] == [62, 63, 67]
    assert set(state.retained_source_observations) == {62, 63, 67}
    assert sum(len(record["rows"]) for record in state.retained_source_observations.values()) <= 3 * 40


@pytest.mark.parametrize("mutation", ["registry_pixel", "source_pose", "geometry_epoch"])
def test_late_retained_snapshot_mutation_rejects_without_partial_commit(monkeypatch, mutation):
    state, _off, source80, _source68, truth_points = _prepare_future_ablation_states()
    real_solver = bundle.least_squares
    injected_state = {}

    def solve_then_mutate(fun, initial, **kwargs):
        result = real_solver(fun, initial, **kwargs)
        if mutation == "registry_pixel":
            state.retained_source_observations[68]["rows"][0]["pixel_float32"][0] += 0.25
        elif mutation == "source_pose":
            state.poses[68][0, 3] += 0.001
        else:
            state.geometry_revision += 1
        injected_state["snapshot"] = {
            "revision": state.revision,
            "geometry_revision": state.geometry_revision,
            "poses": [pose.copy() for pose in state.poses],
            "relative_poses": [pose.copy() for pose in state.relative_poses],
            "pose_anchors": copy.deepcopy(state.pose_anchors),
            "statuses": list(state.statuses),
            "keyframes": {key: frame.pose.copy() for key, frame in state.keyframes.items()},
            "points": {key: value.position.copy() for key, value in state.landmarks.items()},
            "retained": copy.deepcopy(state.retained_source_observations),
        }
        return result

    monkeypatch.setattr(bundle, "least_squares", solve_then_mutate)
    report, _captured = _run_future_bundle(
        state, 80, source80, truth_points, retain=True
    )

    assert report["applied"] is False, report
    assert report["reason"] in {
        "retained_source_snapshot_changed",
        "retained_source_frame_changed",
    }
    _assert_map_snapshot(state, injected_state["snapshot"])


def bundle_matrix():
    # Keep the lifecycle fixture tied to the calibrated scene used by the
    # shared two-BA test helpers without copying numeric calibration values.
    from test_bundle_diagnostics import MATRIX

    return MATRIX
