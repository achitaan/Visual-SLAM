"""Capture-only tracking traces must preserve the actual evidence consumed."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import cv2 as cv
import shared_slam as shared_slam_module
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from slam_state import MappingKeyframe, Observation
from stereo_pose_arbitration import SupportedStereoFrame
from tracking_diagnostics import TrackingDiagnosticsWriter


def _evaluator_main():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shared_slam.py"
    spec = importlib.util.spec_from_file_location("future_trace_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _assert_evaluator_rejects(monkeypatch, capsys, args, expected_text):
    monkeypatch.setattr(sys, "argv", ["evaluate_shared_slam.py", *map(str, args)])
    with pytest.raises(SystemExit) as error:
        _evaluator_main()()
    assert error.value.code == 2
    assert expected_text in capsys.readouterr().err


def _complete_frame(writer, frame=14):
    for phase in (
        "frame_start_state",
        "selected_pre_keyframe",
        "accepted_pre_ba",
        "frame_end",
    ):
        assert writer.record_event(frame, phase, {})


def _tracking_scene(monkeypatch, writer=None, *, bundle_enabled=False, stereo=None):
    """Two keyframes plus disjoint detector/flow pools for solve fallback tests."""
    rng = np.random.default_rng(913)
    matrix = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])
    count = 20
    source_pixels = rng.uniform([60.0, 45.0], [300.0, 435.0], (count, 2)).astype(np.float32)
    depths = rng.uniform(6.0, 9.0, count)
    world = np.c_[source_pixels, np.ones(count)] @ np.linalg.inv(matrix).T * depths[:, None]
    descriptors = rng.normal(size=(count, 128)).astype(np.float32)
    detector_pixels = rng.uniform([420.0, 45.0], [600.0, 435.0], (15, 2)).astype(np.float32)
    detector_desc = descriptors[:15].copy()
    slam = SharedSlam(
        matrix,
        stereo=stereo,
        config=MappingConfig(
            min_inliers=12,
            keyframe_interval=1,
            bundle_enabled=bundle_enabled,
            loop_mode="off",
        ),
        tracking_diagnostics_writer=writer,
    )
    ids = np.arange(count, dtype=np.int64)
    slam.map.keyframes[0] = MappingKeyframe(
        0, 0, np.eye(4), source_pixels.copy(), descriptors.copy(), ids.copy(),
        depth_points=world.copy(),
    )
    slam.map.keyframes[1] = MappingKeyframe(
        1, 1, np.eye(4), source_pixels.copy(), descriptors.copy(), ids.copy(),
        depth_points=world.copy(),
    )
    for i in range(count):
        slam.map.add_landmark(
            world[i], descriptors[i], 0,
            {0: Observation(source_pixels[i]), 1: Observation(source_pixels[i])},
        )
    slam.map.record(np.eye(4), "tracking", 0)
    slam.map.record(np.eye(4), "tracking", 1)
    slam.last_keyframe = 1
    slam.previous_gray = np.zeros((480, 640), dtype=np.uint8)
    slam.previous_tracks = [(i, source_pixels[i].copy()) for i in range(count)]
    slam._extract = lambda _image, _right: (
        detector_pixels.copy(), detector_desc.copy(),
        np.full((15, 3), np.nan), np.full(15, np.nan),
    )
    slam._match = lambda _first, _second: np.c_[np.arange(15), np.arange(15)]
    flow_call = {"count": 0}

    def optical_flow(_first, _second, previous, _next=None, **_kwargs):
        flow_call["count"] += 1
        if flow_call["count"] % 2:
            points = np.asarray(previous).copy()
            points[..., 0] += 5.0
        else:
            points = np.asarray(previous).copy()
            points[..., 0] -= 5.0
        return points, np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr(cv, "calcOpticalFlowPyrLK", optical_flow)
    slam._future_test_pixels = detector_pixels
    slam._future_test_source_pixels = source_pixels
    return slam


def _scripted_estimator(monkeypatch, *, successful_inliers=12):
    calls = []

    def estimate(positions, observations, _matrix, _size, _minimum, **_kwargs):
        calls.append((np.asarray(positions).copy(), np.asarray(observations).copy()))
        if len(calls) == 1:
            return None
        pose = np.eye(4)
        valid = np.arange(successful_inliers, dtype=np.int64)
        return pose, valid, 0.0

    monkeypatch.setattr(shared_slam_module, "estimate_pose", estimate)
    return calls


def test_writer_deep_owns_rows_and_marks_capped_consumption_unknown(tmp_path):
    writer = TrackingDiagnosticsWriter(
        tmp_path, frames=(14,), max_rows_per_pool=2, max_state_rows=1
    )
    rows = [
        {
            "landmark_id": i,
            "world_position": np.array([float(i), 2.0, 8.0]),
            "pixel": np.array([100.25 + i, 200.5]),
            "detector_feature_index": -1 if i == 0 else i,
            "right_u": None,
            "right_u_valid": False,
            "right_role": "not_measured",
        }
        for i in range(3)
    ]
    probe = {"attempt_index": 0, "kind": "flow_assisted", "rows": rows}
    assert writer.record_event(14, "solve_probe", probe)

    # The collector must own a snapshot at the call boundary, not retain live
    # arrays or mutable structures from the pose-fitting path.
    rows[0]["pixel"][0] = -999
    probe["rows"].clear()
    assert writer.record_event(
        14,
        "frame_start_state",
        {"rows": [{"landmark_id": 8}, {"landmark_id": 9}], "input_row_count": 4},
    )
    _complete_frame(writer)
    record = writer.finish_frame(14, tracking_status="tracking")

    saved = tmp_path / record["path"]
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == record["sha256"]
    payload = json.loads(saved.read_text(encoding="utf-8"))
    events = payload["events"]
    captured_probe = next(item for item in events if item["phase"] == "solve_probe")
    assert captured_probe["rows"][0]["pixel"] == [100.25, 200.5]
    assert captured_probe["rows"][0]["right_u"] is None
    assert captured_probe["rows"][0]["right_u_valid"] is False
    assert captured_probe["rows"][0]["right_role"] == "not_measured"
    assert captured_probe["rows"][0]["detector_feature_index"] == -1
    assert captured_probe["input_row_count"] == 3
    assert captured_probe["stored_row_count"] == 2
    assert captured_probe["pool_complete"] is False
    assert captured_probe["consumption_unknown"] is True

    state = next(item for item in events if item["phase"] == "frame_start_state")
    assert state["input_row_count"] == 4
    assert state["stored_row_count"] == 1
    assert state["snapshot_complete"] is False
    assert state["snapshot_truncated"] is True
    assert payload["truncated"] is True


@pytest.mark.parametrize("frames", [(), (-1,), (14, 14), tuple(range(8))])
def test_writer_rejects_unbounded_or_out_of_window_frame_selection(tmp_path, frames):
    with pytest.raises(ValueError):
        TrackingDiagnosticsWriter(tmp_path / str(len(frames)), frames=frames)


def test_writer_returns_detached_payload_and_records_disk_failures(tmp_path, monkeypatch):
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(20,))
    assert writer.record_event(20, "solve_probe", {"rows": [{"pixel": [1.0, 2.0]}]})
    first = writer.frame_payload(20)
    first["events"][0]["rows"][0]["pixel"][0] = 999
    assert writer.frame_payload(20)["events"][0]["rows"][0]["pixel"] == [1.0, 2.0]

    monkeypatch.setattr(writer, "_atomic_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))
    finished = writer.finish_frame(20, tracking_status="lost")
    assert finished["status"] == "incomplete"
    assert finished["error_type"] == "OSError"
    assert writer.manifest()["frames"][0]["status"] == "incomplete"


def test_missing_reference_ledger_cannot_be_mislabeled_as_unused_evidence():
    slam = SharedSlam(
        np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]]),
        config=MappingConfig(loop_mode="off"),
    )
    try:
        context = slam._tracking_trace_reference_context(14, {}, {})
        assert context["reference_pool_status"] == "unknown_uninstrumented"
        assert context["fit_consumption_complete"] is False
        assert context["unused_claim_allowed"] is False
    finally:
        slam.close(finish=False)


def test_reference_trace_preserves_invalid_depth_holes_as_null_masks_without_invalidating_pose(
    tmp_path,
):
    matrix = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])
    slam = SharedSlam(matrix, config=MappingConfig(loop_mode="off"))
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(14,))
    pixels = np.array([[100.25, 80.5], [220.5, 180.25], [450.75, 300.5]], np.float32)
    points = np.array([[1.0, 2.0, 8.0], [np.nan, 3.0, 9.0], [np.nan, np.nan, np.nan]], np.float32)
    right_u = np.array([np.nan, 219.75, np.nan], np.float32)
    descriptors = np.zeros((3, 128), np.float32)
    landmark_ids = np.array([10, -1, 12], np.int64)

    def frame(frame_id):
        return SupportedStereoFrame(
            pixels, descriptors, points, right_u, landmark_ids,
            frame_id, (640, 480), "synthetic-calibration-id",
        )

    fit_pairs = np.array([[0, 1]], np.int64)
    slam._arbitration_context = {
        "previous": frame(14),
        "current": frame(15),
        "training_rows": {"fit_pairs": fit_pairs.copy()},
        "fit_pairs_snapshot": fit_pairs.copy(),
        "held_pairs": np.array([[2, 2]], np.int64),
    }
    try:
        context = slam._tracking_trace_reference_context(
            14, {"choice": "map", "score": 1.0}, {"pose_source": "map"}
        )
        accepted_pose = np.eye(4)
        assert writer.record_event(14, "accepted_pre_ba", {
            "accepted_pose": accepted_pose.copy(),
            "reference_context": context,
            "fit_consumption_complete": False,
            "unused_evidence_certified": False,
        })
        saved_event = writer.frame_payload(14)["events"][0]

        assert saved_event.get("snapshot_valid", True) is True
        np.testing.assert_array_equal(saved_event["accepted_pose"], accepted_pose)
        source = saved_event["reference_context"]["source"]
        target = saved_event["reference_context"]["target"]
        for endpoint in (source, target):
            np.testing.assert_array_equal(endpoint["pixels"], pixels)
            assert endpoint["right_u"] == [None, pytest.approx(219.75), None]
            assert endpoint["right_u_valid"] == [False, True, False]
            assert endpoint["points"][0] == [1.0, 2.0, 8.0]
            assert endpoint["points"][1] == [None, 3.0, 9.0]
            assert endpoint["points"][2] == [None, None, None]
            assert endpoint["points_valid"] == [True, False, False]
        assert context["reference_pool_status"] == "reserved_ledger_captured_other_reference_unknown"
        assert context["fit_consumption_complete"] is False
        assert context["unused_claim_allowed"] is False
        assert context["estimator_unused"] is False
        assert context["holdout_status"] == "fit_disjoint_selection_consumed"
        assert saved_event["unused_evidence_certified"] is False
    finally:
        slam.close(finish=False)


def test_throwing_diagnostics_writer_does_not_change_pose_map_or_fit_work(tmp_path, monkeypatch):
    def run(writer):
        slam = _tracking_scene(monkeypatch, writer)
        estimator_calls = _scripted_estimator(monkeypatch, successful_inliers=12)
        match_calls = {"count": 0}
        original_match = slam._match

        def counted_match(*args, **kwargs):
            match_calls["count"] += 1
            return original_match(*args, **kwargs)

        slam._match = counted_match
        measure_calls = {"count": 0}
        original_measure = slam._measure_stereo_pixels

        def counted_measure(*args, **kwargs):
            measure_calls["count"] += 1
            return original_measure(*args, **kwargs)

        slam._measure_stereo_pixels = counted_measure
        try:
            pose, info = slam.process(2, np.zeros((480, 640), dtype=np.uint8))
            snapshot = {
                "revision": slam.map.revision,
                "geometry_revision": slam.map.geometry_revision,
                "poses": [value.copy() for value in slam.map.poses],
                "statuses": list(slam.map.statuses),
                "keyframes": {
                    key: (frame.pose.copy(), frame.pixels.copy(), frame.landmark_ids.copy())
                    for key, frame in slam.map.keyframes.items()
                },
                "landmarks": {
                    key: (point.position.copy(), point.misses)
                    for key, point in slam.map.landmarks.items()
                },
                "accepted_tracks": [(i, p.copy()) for i, p in slam.accepted_tracks],
                "previous_tracks": [(i, p.copy()) for i, p in slam.previous_tracks],
            }
            return pose.copy(), {
                key: info.get(key)
                for key in ("tracking_ok", "num_matches", "num_inliers", "pose_source", "state")
            }, snapshot, (len(estimator_calls), match_calls["count"], measure_calls["count"])
        finally:
            slam.close(finish=False)

    expected = run(None)
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(2,))

    def fail_record(*_args, **_kwargs):
        raise OSError("diagnostic destination unavailable")

    monkeypatch.setattr(writer, "record_event", fail_record)
    actual = run(writer)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert actual[2]["revision"] == expected[2]["revision"]
    assert actual[2]["geometry_revision"] == expected[2]["geometry_revision"]
    assert actual[2]["statuses"] == expected[2]["statuses"]
    assert actual[2]["keyframes"].keys() == expected[2]["keyframes"].keys()
    for left, right in zip(actual[2]["poses"], expected[2]["poses"]):
        np.testing.assert_array_equal(left, right)
    for key in actual[2]["keyframes"]:
        for left, right in zip(actual[2]["keyframes"][key], expected[2]["keyframes"][key]):
            np.testing.assert_array_equal(left, right)
    for key in actual[2]["landmarks"]:
        np.testing.assert_array_equal(actual[2]["landmarks"][key][0], expected[2]["landmarks"][key][0])
        assert actual[2]["landmarks"][key][1] == expected[2]["landmarks"][key][1]
    for name in ("accepted_tracks", "previous_tracks"):
        assert [row[0] for row in actual[2][name]] == [row[0] for row in expected[2][name]]
        for (_, left), (_, right) in zip(actual[2][name], expected[2][name]):
            np.testing.assert_array_equal(left, right)
    assert expected[3] == actual[3]
    assert actual[3][0] == 2
    assert actual[3][2] == 0


def test_trace_marks_final_selection_then_keyframe_record_before_current_frame_ba(
    tmp_path, monkeypatch
):
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(2,))
    slam = _tracking_scene(monkeypatch, writer, bundle_enabled=True)
    _scripted_estimator(monkeypatch, successful_inliers=12)
    order = []
    original_record = writer.record_event

    def record(frame, phase, payload):
        order.append(("event", phase))
        return original_record(frame, phase, payload)

    monkeypatch.setattr(writer, "record_event", record)
    original_keyframe = slam._keyframe

    def keyframe(*args, **kwargs):
        order.append(("call", "keyframe"))
        return original_keyframe(*args, **kwargs)

    slam._keyframe = keyframe

    def bundle(_state, _matrix, _baseline, **_kwargs):
        order.append(("call", "local_bundle"))
        return {"applied": False, "reason": "test_noop"}

    monkeypatch.setattr(shared_slam_module, "local_bundle_adjustment", bundle)
    try:
        _pose, info = slam.process(2, np.zeros((480, 640), dtype=np.uint8))
        assert info["tracking_ok"]
        assert writer.frame_payload(2)["status"] == "complete"
        phases = [entry[1] for entry in order if entry[0] == "event"]
        assert "selected_pre_keyframe" in phases
        assert "accepted_pre_ba" in phases
        positions = {entry: order.index(entry) for entry in (
            ("event", "selected_pre_keyframe"),
            ("call", "keyframe"),
            ("event", "accepted_pre_ba"),
            ("call", "local_bundle"),
        )}
        assert positions[("event", "selected_pre_keyframe")] < positions[("call", "keyframe")]
        assert positions[("call", "keyframe")] < positions[("event", "accepted_pre_ba")]
        assert positions[("event", "accepted_pre_ba")] < positions[("call", "local_bundle")]
        accepted = next(
            event for event in writer.frame_payload(2)["events"]
            if event["phase"] == "accepted_pre_ba"
        )
        assert accepted["revision"] == slam.map.revision
        assert accepted["accepted_pose"] == slam.map.poses[-1].tolist()
    finally:
        slam.close(finish=False)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--tracking-diagnostics-dir", "{inside}"], "supplied together"),
        (["--tracking-diagnostics-dir", "{inside}", "--tracking-diagnostics-frames", "14", "14"], "duplicates"),
        (["--tracking-diagnostics-dir", "{inside}", "--tracking-diagnostics-frames", "-1"], "nonnegative"),
        (["--tracking-diagnostics-dir", "{inside}", "--tracking-diagnostics-frames", "0", "1", "2", "3", "4", "5", "6", "7"], "max 7"),
        (["--tracking-diagnostics-dir", "{outside}", "--tracking-diagnostics-frames", "14"], "inside the run output"),
    ],
)
def test_evaluator_cli_rejects_invalid_capture_selection_before_dataset_access(
    tmp_path, monkeypatch, capsys, extra, message
):
    output = tmp_path / "run-output"
    inside = output / "tracking"
    outside = tmp_path / "outside"
    rendered = [part.format(inside=inside, outside=outside) for part in extra]
    args = [
        "--data-root", tmp_path / "missing-dataset",
        "--output", output,
        *rendered,
    ]
    _assert_evaluator_rejects(monkeypatch, capsys, args, message)


def test_failed_flow_probe_and_descriptor_fallback_capture_full_pools_and_exact_lk_pixels(
    tmp_path, monkeypatch
):
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(2,))
    slam = _tracking_scene(monkeypatch, writer)
    estimator_calls = _scripted_estimator(monkeypatch, successful_inliers=12)
    try:
        pose, info = slam.process(2, np.zeros((480, 640), dtype=np.uint8))
        assert info["tracking_ok"]
        assert len(estimator_calls) == 2
        np.testing.assert_array_equal(pose, np.eye(4))

        trace = writer.frame_payload(2)
        assert trace["status"] == "complete"
        assert trace["errors"] == []
        probes = [event for event in trace["events"] if event["phase"] == "solve_probe"]
        assert len(probes) == 2
        failed, fallback = probes
        assert failed["attempt_index"] == 0 and fallback["attempt_index"] == 1
        assert failed["returned_status"] in ("failed", "rejected", None)
        assert failed["input_row_count"] == 20
        assert fallback["input_row_count"] == 15
        assert failed["pool_complete"] and fallback["pool_complete"]
        assert len(failed["rows"]) == 20
        assert len(fallback["rows"]) == 15
        assert len(fallback["inlier_landmark_ids"]) == 12
        # The final inlier list is not a substitute for the full pool consumed
        # by the fallback fit; rejected rows still belong to its dependency set.
        fallback_ids = {row["landmark_id"] for row in fallback["rows"]}
        assert len(fallback_ids - set(fallback["inlier_landmark_ids"])) == 3

        lk_rows = [row for row in failed["rows"] if row["landmark_id"] >= 15]
        assert len(lk_rows) == 5
        assert all(row["detector_feature_index"] == -1 for row in lk_rows)
        for row in lk_rows:
            landmark_id = row["landmark_id"]
            expected = slam._future_test_source_pixels[landmark_id] + np.array([5.0, 0.0])
            np.testing.assert_array_equal(np.asarray(row["pixel"], np.float32), expected)
            assert row["right_u"] is None and row["right_u_valid"] is False
            assert row["right_role"] == "not_measured"

        end = next(event for event in trace["events"] if event["phase"] == "frame_end")
        assert end["fit_consumption_complete"] is False
        assert end["reference_pool_status"] == "unknown_uninstrumented"
        assert end.get("unused_evidence_certified", False) is False
    finally:
        slam.close(finish=False)


def test_invalid_stereo_depth_is_explicit_null_with_false_valid_mask(tmp_path, monkeypatch):
    writer = TrackingDiagnosticsWriter(tmp_path, frames=(2,))
    q = np.eye(4)
    q[2, 3] = 100.0
    q[3, 2] = 1.0
    stereo = StereoCamera(object(), q, 0.2)
    slam = _tracking_scene(monkeypatch, writer, stereo=stereo)
    slam.current_disparity = np.zeros((480, 640), dtype=np.float32)
    _scripted_estimator(monkeypatch, successful_inliers=12)
    monkeypatch.setattr(
        shared_slam_module,
        "refine_stereo_map_pose",
        lambda solution, *_args, **_kwargs: (solution, {}),
    )
    try:
        _pose, info = slam.process(2, np.zeros((480, 640), dtype=np.uint8))
        assert info["tracking_ok"]
        probes = [event for event in writer.frame_payload(2)["events"]
                  if event["phase"] == "solve_probe"]
        fallback_rows = probes[1]["rows"]
        assert fallback_rows
        assert all(row["right_u"] is None for row in fallback_rows)
        assert all(row["right_u_valid"] is False for row in fallback_rows)
        assert all(row["right_role"] == "not_measured" for row in fallback_rows)
    finally:
        slam.close(finish=False)
