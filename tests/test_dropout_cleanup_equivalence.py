"""The optimized landmark cleanup must preserve the reference cleanup result."""

import numpy as np

from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam
from slam_state import MappingKeyframe, Observation


def _run_cleanup(cpu_optimizations):
    matrix = np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]])
    slam = SharedSlam(
        matrix,
        config=MappingConfig(loop_mode="off", bundle_enabled=False),
        performance=PerformanceConfig(cpu_optimizations=cpu_optimizations),
    )
    try:
        for ident in range(8):
            pose = np.eye(4)
            pose[0, 3] = ident * 0.1
            slam.map.keyframes[ident] = MappingKeyframe(
                ident,
                ident,
                pose,
                np.zeros((4, 2), np.float32),
                np.zeros((4, 128), np.float32),
                np.full(4, -1, int),
            )

        def add_landmark(anchor, observed_slots, misses):
            observations = {
                keyframe_id: Observation(np.array([float(slot), 2.0]))
                for keyframe_id, slot in observed_slots
            }
            ident = slam.map.add_landmark(
                np.array([float(anchor), 0.0, 5.0]),
                np.zeros(128, np.float32),
                anchor,
                observations,
            )
            slam.map.landmarks[ident].misses = misses
            for keyframe_id, slot in observed_slots:
                slam.map.keyframes[keyframe_id].landmark_ids[slot] = ident
            return ident

        dropped_a = add_landmark(0, [(0, 0), (2, 0)], misses=5)
        dropped_b = add_landmark(1, [(1, 1), (2, 1)], misses=7)
        live = add_landmark(7, [(7, 2)], misses=0)
        # A dropped landmark without observations must also be removed safely.
        unobserved = add_landmark(3, [], misses=5)
        slam.map.revision = 12
        slam.map.geometry_revision = 7
        slam.map.record(np.eye(4), "tracking", anchor=0)
        slam.last_keyframe = 0

        # Prime the cache so the cleanup is checked against an already-built
        # tracking snapshot, as it is during ordinary processing.
        slam._cached_landmarks()
        empty_pixels = np.empty((0, 2), np.float32)
        empty_descriptors = np.empty((0, 128), np.float32)
        empty_points = np.empty((0, 3), float)
        empty_right = np.empty(0, float)
        slam._extract = lambda image, right: (
            empty_pixels,
            empty_descriptors,
            empty_points,
            empty_right,
        )
        slam._track = lambda *args, **kwargs: (None, {})
        slam._relocalize = lambda *args, **kwargs: (None, {})

        _, info = slam.process(1, np.zeros((24, 32), np.uint8))
        _, _, cached_landmarks, positions = slam._cached_landmarks()
        keyframe_ids = {
            ident: keyframe.landmark_ids.copy()
            for ident, keyframe in slam.map.keyframes.items()
        }
        snapshot = {
            "landmark_ids": tuple(sorted(slam.map.landmarks)),
            "keyframe_ids": keyframe_ids,
            "positions": positions.copy(),
            "cached_landmark_ids": tuple(landmark.id for landmark in cached_landmarks),
            "revision": slam.map.revision,
            "geometry_revision": slam.map.geometry_revision,
            "cleanup_frame_state": info["state"],
            "reported_revision": info["map_revision"],
        }
        assert dropped_a not in slam.map.landmarks
        assert dropped_b not in slam.map.landmarks
        assert unobserved not in slam.map.landmarks
        assert live in slam.map.landmarks
        for keyframe_id, slot in ((0, 0), (2, 0), (1, 1), (2, 1)):
            assert slam.map.keyframes[keyframe_id].landmark_ids[slot] == -1
        assert slam.map.keyframes[7].landmark_ids[2] == live
        return snapshot
    finally:
        slam.close(finish=False)


def test_optimized_dropout_cleanup_matches_full_scan_via_process_path():
    optimized = _run_cleanup(cpu_optimizations=True)
    reference = _run_cleanup(cpu_optimizations=False)

    assert optimized["landmark_ids"] == reference["landmark_ids"]
    assert optimized["cached_landmark_ids"] == reference["cached_landmark_ids"]
    assert optimized["revision"] == reference["revision"] == 13
    assert optimized["geometry_revision"] == reference["geometry_revision"] == 8
    assert optimized["cleanup_frame_state"] == reference["cleanup_frame_state"] == "lost"
    assert optimized["reported_revision"] == reference["reported_revision"] == 13
    for ident in optimized["keyframe_ids"]:
        np.testing.assert_array_equal(
            optimized["keyframe_ids"][ident], reference["keyframe_ids"][ident]
        )
    np.testing.assert_array_equal(optimized["positions"], reference["positions"])
