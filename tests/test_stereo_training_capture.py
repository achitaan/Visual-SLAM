from types import SimpleNamespace

import numpy as np

import mapping_geometry
from shared_slam import SharedSlam, _reserved_training_selected_as_edge
from slam_state import MapState, MappingKeyframe, Observation
from stereo_pose_arbitration import SupportedStereoFrame


def _stereo_frame(pixels, points, right_u, landmark_ids, frame, identity):
    pixels = np.asarray(pixels, np.float32)
    return SupportedStereoFrame(
        pixels, np.ones((len(pixels), 8), np.float32), np.asarray(points, float),
        np.asarray(right_u, float), np.asarray(landmark_ids, int), frame,
        (100, 80), identity,
    )


def test_estimator_row_ledger_records_original_forward_reverse_membership(monkeypatch):
    pairs = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    source = SimpleNamespace(
        descriptors=np.ones((3, 8), np.float32),
        points=np.array([[0., 0., 5.], [np.nan, np.nan, np.nan], [2., 0., 5.]]),
        pixels=np.array([[10., 10.], [20., 20.], [30., 30.]]),
        image_size=(100, 80),
    )
    target = SimpleNamespace(
        descriptors=np.ones((3, 8), np.float32),
        points=np.array([[0., 0., 5.], [1., 0., 5.], [2., 0., 5.]]),
        pixels=np.array([[11., 10.], [21., 20.], [31., 30.]]),
        image_size=(100, 80),
    )
    calls = []

    def estimate_pose(points, pixels, matrix, size, minimum, initial_pose=None):
        calls.append(len(points))
        return np.eye(4), np.array([0, 1] if len(calls) == 1 else [1, 2]), 0.25

    monkeypatch.setattr(mapping_geometry, "estimate_pose", estimate_pose)
    monkeypatch.setattr(mapping_geometry, "project", lambda points, pose, matrix:
                        (np.zeros((len(points), 2)), np.ones(len(points))))

    def refine(pose, *_args):
        return pose.copy(), {"applied": True}

    monkeypatch.setattr(mapping_geometry, "refine_bidirectional_stereo", refine)
    estimate = lambda first, second: pairs.copy()
    default = mapping_geometry.estimate_stereo_reference(
        source, target, np.eye(3), min_inliers=2, matcher=estimate
    )
    assert "training_rows" not in default
    calls.clear()
    captured = mapping_geometry.estimate_stereo_reference(
        source, target, np.eye(3), min_inliers=2, matcher=estimate,
        capture_rows=True,
    )
    ledger = captured["training_rows"]
    assert calls == [2, 3]
    assert np.array_equal(ledger["fit_pairs"], pairs)
    assert np.array_equal(ledger["forward_available_pairs"], [[0, 0], [2, 2]])
    assert np.array_equal(ledger["forward_inlier_pairs"], [[0, 0], [2, 2]])
    assert np.array_equal(ledger["reverse_inlier_pairs"], [[1, 1], [2, 2]])
    assert np.array_equal(ledger["forward_fit_row_indices"], [0, 2])
    assert np.array_equal(ledger["reverse_fit_row_indices"], [1, 2])
    assert ledger["refinement_applied"]
    assert not ledger["fit_pairs"].flags.writeable
    pairs[0] = [2, 1]
    assert np.array_equal(ledger["fit_pairs"][0], [0, 0])
    assert np.array_equal(default["measurement"], captured["measurement"])


def _diagnostic_context():
    state = MapState(metric=True)
    identity_pose = np.eye(4)
    source_pixels = np.array([[10., 10.], [11., 11.]], np.float32)
    target_pixels = np.array([[20., 20.], [21., 21.]], np.float32)
    source_kf = MappingKeyframe(0, 0, identity_pose.copy(), source_pixels.copy(),
                               np.ones((2, 8), np.float32), np.full(2, -1, int))
    target_kf = MappingKeyframe(1, 1, identity_pose.copy(), target_pixels.copy(),
                               np.ones((2, 8), np.float32), np.full(2, -1, int))
    state.keyframes.update({0: source_kf, 1: target_kf})
    state.record(identity_pose.copy(), "tracking", 0)
    state.record(identity_pose.copy(), "tracking", 1)
    source_id = state.add_landmark(
        np.array([0., 0., 5.]), np.ones(8), 0,
        {0: Observation(source_pixels[0].copy(), 15.)},
    )
    target_id = state.add_landmark(
        np.array([0.1, 0., 5.]), np.ones(8), 1,
        {1: Observation(target_pixels[0].copy(), 16.)},
    )
    source_kf.landmark_ids[0] = source_id
    target_kf.landmark_ids[0] = target_id
    matrix = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    stereo = SimpleNamespace(Q=np.eye(4), baseline=0.5, disparity_offset=0.)
    shared = SharedSlam.__new__(SharedSlam)
    shared.map = state
    shared.K = matrix
    shared.stereo = stereo
    shared.stereo_calibration_identity = shared._live_stereo_calibration_identity()
    shared.previous_tracks = [(source_id, source_pixels[0].copy())]
    identity = shared.stereo_calibration_identity
    previous = _stereo_frame(source_pixels, [[0., 0., 5.], [1., 0., 5.]],
                              [15., 14.], [source_id, -1], 0, identity)
    current = _stereo_frame(target_pixels, [[0.1, 0., 5.], [1.1, 0., 5.]],
                            [16., 15.], [-1, -1], 1, identity)
    source_anchor_pose = state.keyframes[0].pose.copy()
    context = {
        "previous": previous,
        "current": current,
        "fit": np.array([[0, 0]], dtype=np.int64),
        "fit_pairs_snapshot": np.array([[0, 0]], dtype=np.int64),
        "held_pairs": np.array([[1, 1]], dtype=np.int64),
        "excluded_targets": {1},
        "excluded_landmarks": {source_id},
        "training_rows": {
            "fit_pairs": np.array([[0, 0]], dtype=np.int64),
            "forward_inlier_pairs": np.array([[0, 0]], dtype=np.int64),
            "reverse_inlier_pairs": np.array([[0, 0]], dtype=np.int64),
            "forward_fit_row_indices": np.array([0], dtype=np.int64),
            "reverse_fit_row_indices": np.array([0], dtype=np.int64),
            "reverse_status": "verified",
            "refinement_attempted": True,
            "refinement_applied": True,
        },
        "fit_source_epoch": (state.revision, state.geometry_revision),
        "fit_source_state": {
            "source_pose": state.poses[0].copy(),
            "source_status": "tracking",
            "source_anchor_keyframe_id": 0,
            "source_anchor_pose": source_anchor_pose,
        },
        "verified": {"measurement": np.eye(4)},
    }
    return shared, context, target_id, target_pixels


def test_prepared_training_snapshot_keeps_exact_rows_owner_certificates_and_exclusions():
    shared, context, target_id, target_pixels = _diagnostic_context()
    before = target_pixels[0].copy()
    prepared = {
        "selection": {"selected_landmarks": [{"landmark_id": target_id}]},
        "single_view_propagations": [],
    }
    captured = shared._bundle_training_observations_context(
        1, (100, 80), context, {"reason": "reserved_supported_evidence"}, prepared
    )
    assert captured["status"] == "captured"
    assert captured["eligible"]
    assert captured["rows"][0]["roles"]["forward_inlier"]
    assert captured["rows"][0]["roles"]["reverse_inlier"]
    assert captured["rows"][0]["source_claimed_landmark_id"] == 0
    assert captured["rows"][0]["target_map_ownership"]["classification"] == (
        "reusable_existing_selected_point"
    )
    assert captured["excluded_heldout_pairs"][0]["roles"]["held_out"]
    assert captured["fit_holdout_physical_pixel_overlap"] == []
    assert {item["frame_id"] for item in captured["existing_observations"]} == {0, 1}
    assert captured["rows"][0]["target_pixel"]["values"] == before.tolist()
    target_pixels[0] = [77., 77.]
    assert captured["rows"][0]["target_pixel"]["values"] == before.tolist()


def test_stale_and_malformed_capture_ledgers_fail_closed():
    shared, context, target_id, _ = _diagnostic_context()
    prepared = {"selection": {"selected_landmarks": [{"landmark_id": target_id}]},
                "single_view_propagations": []}
    shared.map.poses[0][0, 3] += 0.1
    stale = shared._bundle_training_observations_context(
        1, (100, 80), context, {}, prepared
    )
    assert not stale["eligible"]
    assert stale["reason"] == "source_pose_or_anchor_changed_since_fit"

    shared, context, target_id, _ = _diagnostic_context()
    context["training_rows"]["forward_fit_row_indices"] = np.array([0.5])
    malformed = shared._bundle_training_observations_context(
        1, (100, 80), context, {}, prepared
    )
    assert not malformed["eligible"]
    assert malformed["reason"] == "malformed_training_row_ledger"


def test_reserved_rows_are_ineligible_when_map_wins_or_full_pool_consumes_them():
    measurement = np.eye(4)
    assert not _reserved_training_selected_as_edge(
        "map_pnp", "map", 4, 4, measurement, measurement, False
    )
    assert not _reserved_training_selected_as_edge(
        "stereo_tracking_reference", "existing_reference", 4, 4,
        measurement, measurement, True,
    )
    assert _reserved_training_selected_as_edge(
        "reserved_stereo_arbitration", "independent", 4, 4,
        measurement, measurement, False,
    )


def test_capture_keeps_full_360_row_partition_when_candidate_union_is_bounded():
    shared, context, _target_id, _ = _diagnostic_context()
    count, fit_count = 360, 193
    indices = np.arange(count)
    source_pixels = np.column_stack((10 + indices % 80, 5 + indices // 80)).astype(np.float32)
    target_pixels = np.column_stack((10 + indices % 80, 30 + indices // 80)).astype(np.float32)
    context["previous"] = _stereo_frame(
        source_pixels, np.tile([0., 0., 5.], (count, 1)), np.full(count, 15.),
        np.full(count, -1), 0, shared.stereo_calibration_identity)
    context["current"] = _stereo_frame(
        target_pixels, np.tile([0.1, 0., 5.], (count, 1)), np.full(count, 16.),
        np.full(count, -1), 1, shared.stereo_calibration_identity)
    fit = np.column_stack((np.arange(fit_count), np.arange(fit_count))).astype(np.int64)
    held = np.column_stack((np.arange(fit_count, count),
                            np.arange(fit_count, count))).astype(np.int64)
    context["fit"] = fit.copy()
    context["fit_pairs_snapshot"] = fit.copy()
    context["held_pairs"] = held.copy()
    context["excluded_targets"] = set(range(fit_count, count))
    context["excluded_landmarks"] = set()
    context["training_rows"].update({
        "fit_pairs": fit.copy(), "forward_inlier_pairs": fit.copy(),
        "reverse_inlier_pairs": fit.copy(),
        "forward_fit_row_indices": np.arange(fit_count, dtype=np.int64),
        "reverse_fit_row_indices": np.arange(fit_count, dtype=np.int64),
    })
    captured = shared._bundle_training_observations_context(
        1, (100, 80), context, {}, {"selection": {"selected_landmarks": []},
                                    "single_view_propagations": []}
    )
    assert captured["status"] == "captured"
    assert captured["eligible"]
    assert captured["fit_match_count"] == 193
    assert captured["heldout_match_count"] == 167
    assert len(captured["fit_training_matches"]) == 193
    assert len(captured["excluded_heldout_pairs"]) == 167
    assert captured["candidate_row_count"] == 193
