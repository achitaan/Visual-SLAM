import numpy as np
import pytest

import shared_slam as shared_slam_module
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from slam_state import MappingKeyframe, Observation


K = np.array([[250.0, 0.0, 320.0], [0.0, 250.0, 240.0], [0.0, 0.0, 1.0]])


def stereo_camera():
    baseline = 0.54
    q = np.array(
        [[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 250], [0, 0, 1 / baseline, 0]],
        float,
    )
    return StereoCamera(None, q, baseline)


def test_stereo_keyframe_preserves_orientation_rows_but_creates_one_lm_per_pixel():
    slam = SharedSlam(K, stereo=stereo_camera())
    try:
        pixels = np.array(
            [[100.0, 100.0], [100.0, 100.0], [200.0, 100.0],
             [np.nextafter(np.float32(200.0), np.float32(np.inf)), 100.0]],
            np.float32,
        )
        descriptors = np.arange(4 * 128, dtype=np.float32).reshape(4, 128)
        points = np.array(
            [[-4.4, -2.8, 8.0], [-4.4, -2.8, 8.0], [-1.6, -2.8, 8.0],
             [-1.599, -2.8, 8.0]],
            float,
        )
        right_u = np.array([90.0, 90.0, 190.0, 190.0])

        keyframe_id = slam._keyframe(
            0, np.eye(4), pixels, descriptors, points, right_u, {}
        )

        keyframe = slam.map.keyframes[keyframe_id]
        assert len(slam.map.landmarks) == 3
        assert keyframe.landmark_ids[0] == keyframe.landmark_ids[1] >= 0
        assert keyframe.landmark_ids[2] >= 0
        assert keyframe.landmark_ids[3] >= 0
        assert keyframe.landmark_ids[2] != keyframe.landmark_ids[3]
        assert np.array_equal(keyframe.descriptors, descriptors)
        assert np.array_equal(keyframe.retrieval_descriptors, descriptors)
        for landmark in slam.map.landmarks.values():
            assert set(landmark.observations) == {keyframe_id}
        assert np.array_equal(
            slam.map.landmarks[int(keyframe.landmark_ids[0])].observations[keyframe_id].pixel,
            pixels[0],
        )
    finally:
        slam.close()


def test_stereo_bootstrap_threshold_and_landmark_count_use_physical_pixels():
    image = np.zeros((480, 640), np.uint8)
    for physical_count, should_initialize in ((29, False), (30, True)):
        slam = SharedSlam(K, stereo=stereo_camera(),
                          config=MappingConfig(loop_mode="off", bundle_enabled=False))
        try:
            physical_pixels = np.array(
                [[70.0 + (i % 6) * 90.0, 60.0 + (i // 6) * 75.0]
                 for i in range(physical_count)],
                np.float32,
            )
            pixels = np.vstack([physical_pixels, np.repeat(physical_pixels[:1], 3, axis=0)])
            descriptors = np.random.default_rng(81).normal(
                size=(len(pixels), 128)
            ).astype(np.float32)
            points = np.column_stack((
                (pixels[:, 0] - 320.0) / 50.0,
                (pixels[:, 1] - 240.0) / 50.0,
                np.full(len(pixels), 5.0),
            ))
            right_u = pixels[:, 0] - 20.0
            slam._extract = lambda *_: (
                pixels.copy(), descriptors.copy(), points.copy(), right_u.copy()
            )

            _, info = slam.process(0, image, image)

            assert info["tracking_ok"] is should_initialize
            assert len(slam.map.landmarks) == (physical_count if should_initialize else 0)
            if should_initialize:
                keyframe = slam.map.keyframes[0]
                assert len(keyframe.descriptors) == physical_count + 3
                assert keyframe.landmark_ids[-3:].tolist() == [
                    keyframe.landmark_ids[0]
                ] * 3
                assert len(slam.accepted_tracks) == physical_count
        finally:
            slam.close()


def test_tracked_identity_propagates_to_unassociated_pixel_aliases_once():
    slam = SharedSlam(K, stereo=stereo_camera())
    try:
        descriptor = np.ones(128, np.float32)
        landmark_id = slam.map.add_landmark([0.0, 0.0, 5.0], descriptor, 0, {})
        pixels = np.array([[120.0, 80.0], [120.0, 80.0]], np.float32)
        descriptors = np.stack([descriptor, descriptor * 2])
        points = np.array([[-4.0, -3.2, 8.0], [-4.0, -3.2, 8.0]])
        right_u = np.array([110.0, 110.0])
        associations = {0: landmark_id}

        keyframe_id = slam._keyframe(
            0, np.eye(4), pixels, descriptors, points, right_u, associations
        )

        keyframe = slam.map.keyframes[keyframe_id]
        assert len(slam.map.landmarks) == 1
        assert keyframe.landmark_ids.tolist() == [landmark_id, landmark_id]
        assert list(slam.map.landmarks[landmark_id].observations) == [keyframe_id]
        assert associations == {0: landmark_id, 1: landmark_id}
    finally:
        slam.close()


def test_conflicting_ids_or_same_pixel_metric_values_fail_closed():
    slam = SharedSlam(K, stereo=stereo_camera())
    try:
        first = slam.map.add_landmark([0.0, 0.0, 5.0], np.ones(128), 0, {})
        second = slam.map.add_landmark([0.0, 0.0, 5.0], np.zeros(128), 0, {})
        pixel = np.array([[120.0, 80.0], [120.0, 80.0]], np.float32)
        desc = np.stack([np.ones(128), np.zeros(128)]).astype(np.float32)
        points = np.array([[-4.0, -3.2, 8.0], [-4.0, -3.2, 8.0]])
        right = np.array([110.0, 110.0])
        associations = {0: first, 1: second}

        keyframe_id = slam._keyframe(0, np.eye(4), pixel, desc, points, right, associations)
        keyframe = slam.map.keyframes[keyframe_id]
        assert len(slam.map.landmarks) == 2
        assert keyframe.landmark_ids.tolist() == [-1, -1]
        assert not slam.map.landmarks[first].observations
        assert not slam.map.landmarks[second].observations

        inconsistent = np.array([[-4.0, -3.2, 8.0], [-4.2, -3.2, 8.0]])
        second_keyframe_id = slam._keyframe(
            1, np.eye(4), pixel, desc, inconsistent, right + [0.0, 1.0], {}
        )
        second_keyframe = slam.map.keyframes[second_keyframe_id]
        assert np.all(second_keyframe.landmark_ids == -1)
        assert len(slam.map.landmarks) == 2
    finally:
        slam.close()


def test_one_landmark_cannot_overwrite_two_distinct_pixel_observations():
    slam = SharedSlam(K, stereo=stereo_camera())
    try:
        descriptor = np.ones(128, np.float32)
        landmark_id = slam.map.add_landmark([0.0, 0.0, 5.0], descriptor, 0, {})
        pixels = np.array([[120.0, 80.0], [121.0, 80.0]], np.float32)
        associations = {0: landmark_id, 1: landmark_id}
        points = np.array([[-4.0, -3.2, 8.0], [-3.9, -3.2, 8.0]])

        keyframe_id = slam._keyframe(
            0, np.eye(4), pixels, np.stack([descriptor, descriptor]),
            points, np.array([110.0, 111.0]), associations,
        )

        assert len(slam.map.landmarks) == 1
        assert not slam.map.landmarks[landmark_id].observations
        assert np.all(slam.map.keyframes[keyframe_id].landmark_ids == -1)
    finally:
        slam.close()


def test_mutual_matches_collapse_alias_pairs_and_reject_cross_pixel_ambiguity():
    pixels = np.array([[10, 10], [10, 10], [20, 20]], np.float32)
    target = np.array([[30, 30], [30, 30], [40, 40]], np.float32)
    pairs = np.array([[0, 0], [1, 1], [2, 2]], int)

    collapsed = SharedSlam._collapse_physical_matches(pairs, pixels, target)
    assert len(collapsed) == 2
    assert len({tuple(target[j]) for _, j in collapsed}) == 2

    source_ambiguity = np.array([[0, 0], [1, 2]], int)
    assert not len(SharedSlam._collapse_physical_matches(
        source_ambiguity, pixels, target
    ))
    target_ambiguity = np.array([[0, 0], [2, 1]], int)
    assert not len(SharedSlam._collapse_physical_matches(
        target_ambiguity, pixels, target
    ))

    conflicting_ids = np.array([4, 5, -1])
    assert not len(SharedSlam._collapse_physical_matches(
        pairs[:2], pixels, target, first_landmark_ids=conflicting_ids,
    ))


def test_tracking_does_not_count_orientation_aliases_as_separate_pnp_rows(monkeypatch):
    slam = SharedSlam(K, config=MappingConfig(min_inliers=1))
    try:
        point = np.array([0.0, 0.0, 5.0])
        pixel = np.array([320.0, 240.0], np.float32)
        descriptor = np.zeros(128, np.float32)
        landmark_id = slam.map.add_landmark(
            point, descriptor, 0, {0: Observation(pixel.copy())}
        )
        keyframe = MappingKeyframe(
            0, 0, np.eye(4), np.stack([pixel, pixel]),
            np.stack([descriptor, descriptor + 1.0]),
            np.array([landmark_id, landmark_id]),
        )
        keyframe.image_size = (640, 480)
        slam.map.keyframes[0] = keyframe
        slam.map.record(np.eye(4), "tracking", 0)
        target_pixels = np.stack([pixel, pixel])
        target_descriptors = np.stack([descriptor, descriptor + 1.0])
        monkeypatch.setattr(
            slam, "_match", lambda first, second: np.array([[0, 0], [1, 1]], int)
        )
        seen = []

        def estimate(points, observed, *args, **kwargs):
            seen.append((points.copy(), observed.copy()))
            return np.eye(4), np.arange(len(points)), 0.0

        monkeypatch.setattr(shared_slam_module, "estimate_pose", estimate)
        result, info = slam._track(target_pixels, target_descriptors, (640, 480))

        assert result is not None
        assert info["num_matches"] == 1
        assert len(seen) == 1 and len(seen[0][0]) == 1
        assert len(slam.accepted_tracks) == 1
        assert result[1] == {0: landmark_id}
    finally:
        slam.close()


def test_tracking_drops_distinct_landmarks_competing_for_one_target_pixel(monkeypatch):
    slam = SharedSlam(K, config=MappingConfig(min_inliers=1))
    try:
        source_pixels = np.array([[300.0, 220.0], [340.0, 260.0]], np.float32)
        descriptors = np.stack([np.zeros(128), np.ones(128)]).astype(np.float32)
        ids = [
            slam.map.add_landmark(
                [float(i), 0.0, 5.0], descriptors[i], 0,
                {0: Observation(source_pixels[i].copy())},
            )
            for i in range(2)
        ]
        keyframe = MappingKeyframe(
            0, 0, np.eye(4), source_pixels, descriptors, np.asarray(ids)
        )
        keyframe.image_size = (640, 480)
        slam.map.keyframes[0] = keyframe
        slam.map.record(np.eye(4), "tracking", 0)
        target_pixels = np.array([[320.0, 240.0], [320.0, 240.0]], np.float32)
        monkeypatch.setattr(
            slam, "_match", lambda first, second: np.array([[0, 0], [1, 1]], int)
        )
        result, _ = slam._track(target_pixels, descriptors, (640, 480))

        assert result is None
        assert slam.accepted_tracks == []
    finally:
        slam.close()


def test_tracking_rejects_legacy_ids_claiming_one_source_pixel(monkeypatch):
    slam = SharedSlam(K, config=MappingConfig(min_inliers=1))
    try:
        pixel = np.array([320.0, 240.0], np.float32)
        descriptors = np.stack([np.zeros(128), np.ones(128)]).astype(np.float32)
        ids = [
            slam.map.add_landmark(
                [float(i), 0.0, 5.0], descriptors[i], 0,
                {0: Observation(pixel.copy())},
            )
            for i in range(2)
        ]
        keyframe = MappingKeyframe(
            0, 0, np.eye(4), np.stack([pixel, pixel]), descriptors, np.asarray(ids)
        )
        keyframe.image_size = (640, 480)
        slam.map.keyframes[0] = keyframe
        slam.map.record(np.eye(4), "tracking", 0)
        monkeypatch.setattr(
            slam, "_match", lambda *args, **kwargs: pytest.fail("ambiguous source IDs matched")
        )

        result, _ = slam._track(np.stack([pixel]), descriptors[:1], (640, 480))

        assert result is None
    finally:
        slam.close()


def test_stereo_reference_verification_receives_unique_geometry_rows(monkeypatch):
    slam = SharedSlam(K, stereo=stereo_camera())
    try:
        unique_pixels = np.array(
            [[70.0 + 20.0 * i, 100.0 + 3.0 * i] for i in range(20)], np.float32
        )
        pixels = np.vstack([unique_pixels, unique_pixels[:1]])
        points = np.column_stack((
            (pixels[:, 0] - 320.0) / 50.0,
            (pixels[:, 1] - 240.0) / 50.0,
            np.full(len(pixels), 5.0),
        ))
        descriptors = np.random.default_rng(43).normal(
            size=(len(pixels), 128)
        ).astype(np.float32)
        keyframe = MappingKeyframe(
            0, 0, np.eye(4), pixels.copy(), descriptors.copy(),
            np.full(len(pixels), -1, int), depth_points=points.copy(),
        )
        keyframe.image_size = (640, 480)
        slam.map.keyframes[0] = keyframe
        observed = []

        def verify(source, target, matrix, min_inliers):
            observed.append((source, target))
            return None

        monkeypatch.setattr(shared_slam_module, "verify_loop", verify)
        slam._keyframe_stereo_reference(
            pixels, descriptors, points, (640, 480), ranked=[(30, 0)]
        )

        assert len(observed) == 1
        source, target = observed[0]
        assert len(source.pixels) == len(target.pixels) == 20
        assert len({tuple(pixel) for pixel in source.pixels}) == 20
        assert len({tuple(pixel) for pixel in target.pixels}) == 20
        # The persistent/retrieval keyframe still keeps every SIFT row.
        assert len(keyframe.descriptors) == 21
    finally:
        slam.close()


def test_monocular_bootstrap_counts_physical_pairs_and_keeps_alias_descriptors(monkeypatch):
    slam = SharedSlam(K, config=MappingConfig(loop_mode="off"))
    try:
        physical_pixels = np.array(
            [[80.0 + (i % 10) * 45.0, 60.0 + (i // 10) * 45.0]
             for i in range(54)],
            np.float32,
        )
        pixels = np.vstack([physical_pixels[0], physical_pixels])
        descriptors = np.random.default_rng(100).normal(size=(len(pixels), 128)).astype(np.float32)
        blank_points = np.full((len(pixels), 3), np.nan)
        blank_right = np.full(len(pixels), np.nan)
        slam._extract = lambda *_: (pixels.copy(), descriptors.copy(),
                                    blank_points.copy(), blank_right.copy())
        monkeypatch.setattr(
            slam, "_match",
            lambda first, second: np.column_stack(
                (np.arange(len(first)), np.arange(len(first)))
            ),
        )
        observed_pair_count = []

        def initialize(first, second, matrix, size):
            observed_pair_count.append(len(first))
            assert len({tuple(p) for p in first}) == len(first)
            assert len({tuple(p) for p in second}) == len(second)
            positions = np.column_stack(
                (np.arange(len(first), dtype=float) * 0.01,
                 np.zeros(len(first)), np.full(len(first), 5.0))
            )
            return np.eye(4), positions, np.arange(len(first))

        monkeypatch.setattr(shared_slam_module, "initialize_monocular", initialize)
        image = np.zeros((480, 640), np.uint8)
        slam.process(0, image)
        slam.process(1, image)

        assert observed_pair_count == [54]
        assert len(slam.map.landmarks) == 54
        first, second = slam.map.keyframes[0], slam.map.keyframes[1]
        assert len(first.descriptors) == len(second.descriptors) == 55
        assert first.landmark_ids[0] == first.landmark_ids[1]
        assert second.landmark_ids[0] == second.landmark_ids[1]
        assert len(slam.accepted_tracks) == 54
    finally:
        slam.close()
