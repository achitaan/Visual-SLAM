"""Owned, bounded bundle snapshots must remain outside BA decision-making."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import local_bundle as bundle
from bundle_diagnostics import BundleDiagnosticsWriter
from local_bundle import local_bundle_adjustment
from mapping_geometry import project, right_pixel
from slam_state import MapState, MappingKeyframe, Observation


MATRIX = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE = 0.2


def _scene():
    rng = np.random.default_rng(1703)
    truth = []
    estimate = []
    state = MapState(metric=True)
    points = rng.uniform([-1.6, -1.3, 5.0], [1.6, 1.3, 11.0], (24, 3))
    for keyframe_id, x in enumerate((0.0, 0.5, 1.0)):
        pose = np.eye(4)
        pose[0, 3] = x
        truth.append(pose.copy())
        candidate = pose.copy()
        if keyframe_id == 2:
            candidate[0, 3] += 0.08
        estimate.append(candidate)
        pixels, _ = project(points, pose, MATRIX)
        state.keyframes[keyframe_id] = MappingKeyframe(
            keyframe_id,
            100 + keyframe_id,
            candidate.copy(),
            pixels.astype(np.float32),
            rng.normal(size=(len(points), 128)).astype(np.float32),
            np.arange(len(points), dtype=np.int64),
        )
        state.record(candidate, "tracking", keyframe_id)

    for point_index, point in enumerate(points):
        observations = {}
        for keyframe_id, pose in enumerate(truth):
            pixel, depth = project(point[None, :], pose, MATRIX)
            has_right = (point_index + keyframe_id) % 4 != 0
            right_u = (
                float(right_pixel(pixel[0, 0], depth[0], MATRIX[0, 0], BASELINE, 3.25))
                if has_right else None
            )
            observations[keyframe_id] = Observation(pixel[0], right_u)
        state.add_landmark(point, rng.normal(size=128), 0, observations)

    # A single-view point exercises propagated-row capture independently of
    # the selected multiview and excluded held-out observations.
    singleton = points[0] + np.array([0.05, -0.03, 0.1])
    pixel, depth = project(singleton[None, :], truth[2], MATRIX)
    single_id = state.add_landmark(
        singleton,
        rng.normal(size=128),
        2,
        {2: Observation(pixel[0], float(right_pixel(
            pixel[0, 0], depth[0], MATRIX[0, 0], BASELINE, 3.25)))},
    )
    state.add_stereo_motion(0, 2, np.linalg.inv(truth[0]) @ truth[2])
    return state, truth, single_id


def _snapshot(state):
    return (
        state.revision,
        state.geometry_revision,
        {key: frame.pose.copy() for key, frame in state.keyframes.items()},
        {key: landmark.position.copy() for key, landmark in state.landmarks.items()},
    )


def _assert_same_snapshot(left, right):
    assert left[:2] == right[:2]
    assert left[2].keys() == right[2].keys()
    assert left[3].keys() == right[3].keys()
    for key in left[2]:
        np.testing.assert_array_equal(left[2][key], right[2][key])
    for key in left[3]:
        np.testing.assert_array_equal(left[3][key], right[3][key])


def _run(state, diagnostic_sink=None):
    return local_bundle_adjustment(
        state,
        MATRIX,
        BASELINE,
        window=3,
        max_landmarks=20,
        disparity_offset=3.25,
        diagnostic_sink=diagnostic_sink,
    )


def test_actual_solver_snapshots_preserve_order_gauges_and_owned_rows():
    state, _, single_id = _scene()
    phases = {}
    report = _run(state, lambda phase, payload: phases.__setitem__(phase, payload))

    assert report["applied"]
    assert set(phases) == {"prepared", "solved"}
    prepared = phases["prepared"]
    assert prepared["schema"] == "local_bundle_snapshot_v1"
    assert prepared["input_valid"] is True
    assert prepared["selection"]["eligible_landmarks_before_cap"] == 24
    assert prepared["selection"]["window_keyframe_ids"] == [0, 1, 2]
    assert prepared["selection"]["fixed_keyframe_ids"] == [0]
    assert prepared["selection"]["free_keyframe_ids"] == [1, 2]
    assert [row["frame_id"] for row in prepared["camera_poses"]] == [100, 101, 102]
    assert prepared["parameter_layout"]["residual_camera_keyframe_ids"] == [0, 1, 2]
    assert len(prepared["selected_observations"]) == 60
    assert {row["dimensions"] for row in prepared["selected_observations"]} == {2, 3}
    assert len(prepared["excluded_multiview_observations"]) == 8
    assert prepared["single_view_propagations"][0]["landmark_id"] == single_id
    assert prepared["motion_checks"][0]["frame_ids"] == [0, 2]
    assert prepared["parameter_layout"]["pose_offsets"] == {"1": 0, "2": 6}
    assert prepared["parameter_layout"]["initial_vector"]
    assert phases["solved"]["candidate"]["valid"] is True
    assert phases["solved"]["acceptance"] == "pending"
    assert "descriptors" not in json.dumps(prepared).lower()


def test_recording_mutating_and_absent_sinks_have_identical_actual_solver_results():
    baseline_state, _, _ = _scene()
    baseline_report = _run(baseline_state)
    baseline_snapshot = _snapshot(baseline_state)

    diagnostic_state, _, _ = _scene()

    def mutate_payload(_phase, payload):
        if "parameter_layout" in payload:
            payload["parameter_layout"]["initial_vector"][0] = 100000.0
            payload["selected_observations"][0]["pixel"][0] = -100000.0

    diagnostic_report = _run(diagnostic_state, mutate_payload)
    diagnostic_snapshot = _snapshot(diagnostic_state)
    _assert_same_snapshot(baseline_snapshot, diagnostic_snapshot)
    assert baseline_report["applied"] == diagnostic_report["applied"]
    for name in ("initial_cost", "final_cost", "affected_initial_cost", "affected_final_cost"):
        assert baseline_report[name] == diagnostic_report[name]


def test_callback_failures_are_reported_but_cannot_change_solver_acceptance(monkeypatch):
    def deterministic_solver(residual, initial, **_kwargs):
        residual(initial)
        return SimpleNamespace(x=initial.copy(), nfev=1, success=True, status=1,
                               message="unchanged", cost=0.0, optimality=0.0)

    monkeypatch.setattr(bundle, "least_squares", deterministic_solver)
    expected_state, _, _ = _scene()
    expected = _run(expected_state)
    expected_snapshot = _snapshot(expected_state)

    failed_state, _, _ = _scene()
    actual = _run(failed_state, lambda _phase, _payload: (_ for _ in ()).throw(OSError()))
    _assert_same_snapshot(expected_snapshot, _snapshot(failed_state))
    assert actual["applied"] == expected["applied"]
    assert actual.get("reason") == expected.get("reason")
    assert [item["phase"] for item in actual["diagnostic_errors"]] == ["prepared", "solved"]


def test_writer_hashes_relative_finite_phase_files_and_requires_all_phases(tmp_path):
    writer = BundleDiagnosticsWriter(tmp_path, frames=(127,))
    assert writer.should_capture(127)
    assert not writer.should_capture(128)
    for phase in ("prepared", "solved", "finished"):
        path = writer.emit(127, phase, {"value": phase}, image_size=(1242, 375))
        assert path == f"frame-00127/{phase}.json"
    manifest = writer.manifest()
    record = manifest["frames"][0]
    assert record["status"] == "complete"
    assert set(record["phases"]) == {"prepared", "solved", "finished"}
    for phase, entry in record["phases"].items():
        assert not Path(entry["path"]).is_absolute()
        saved = tmp_path / entry["path"]
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == entry["sha256"]
        assert json.loads(saved.read_text(encoding="utf-8"))["phase"] == phase

    # Manifest generation is pure and does not make export dependent on disk I/O.
    writer._atomic_write = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    assert writer.manifest()["frames"][0]["status"] == "complete"


def test_writer_preserves_errors_and_marks_finished_only_early_exit_skipped(tmp_path):
    writer = BundleDiagnosticsWriter(tmp_path / "early", frames=(7, 8))
    writer.emit(7, "finished", {"report": {"reason": "insufficient_keyframes"}})
    assert writer.mark_skipped(7, "insufficient_keyframes")
    assert writer.manifest()["frames"][0]["status"] == "skipped"
    assert "prepared" not in writer.manifest()["frames"][0]["phases"]

    with pytest.raises(ValueError):
        writer.emit(8, "prepared", {"input_valid": False, "invalid_fields": ["$.K"]})
    writer.emit(8, "solved", {"result": {"valid": True}})
    writer.emit(8, "finished", {"report": {"applied": False}})
    record = writer.manifest()["frames"][1]
    assert record["status"] == "error"
    assert record["errors"][0]["error_type"] == "ValueError"
    assert not writer.mark_skipped(8, "late skip")


def test_writer_rejects_nonfinite_payload_and_duplicate_or_unbounded_selection(tmp_path):
    writer = BundleDiagnosticsWriter(tmp_path, frames=(4,))
    with pytest.raises(ValueError):
        writer.emit(4, "prepared", {"value": float("nan")})
    assert writer.manifest()["frames"][0]["status"] == "error"
    with pytest.raises(ValueError, match="unique"):
        BundleDiagnosticsWriter(tmp_path / "dup", frames=(4, 4))
    with pytest.raises(ValueError, match="max_frames"):
        BundleDiagnosticsWriter(tmp_path / "large", frames=range(3), max_frames=2)
