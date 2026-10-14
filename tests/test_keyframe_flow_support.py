"""Keyframe policy must count unique geometrically accepted map tracks."""
import cv2 as cv
import numpy as np
import pytest
from shared_slam import SharedSlam, MappingConfig
from slam_state import MappingKeyframe, Observation
from mapping_geometry import project


@pytest.mark.parametrize('landmarks,expected_keyframes', [(79, 2), (80, 1), (120, 1)])
def test_verified_flow_support_controls_early_keyframe_insertion(monkeypatch, landmarks, expected_keyframes):
    cv.setRNGSeed(0)
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    rng = np.random.default_rng(862)
    initial_pixels = rng.uniform([80., 50.], [560., 430.], (landmarks, 2))
    depths = rng.uniform(6., 10., landmarks)
    world = np.c_[initial_pixels, np.ones(landmarks)] @ np.linalg.inv(matrix).T * depths[:, None]
    descriptors = rng.normal(size=(landmarks, 128)).astype(np.float32)
    slam = SharedSlam(matrix, config=MappingConfig(bundle_enabled=False, loop_mode='off'))
    slam.map.keyframes[0] = MappingKeyframe(0, 0, np.eye(4), initial_pixels,
        descriptors, np.arange(landmarks), depth_points=world.copy())
    for point, descriptor, pixel in zip(world, descriptors, initial_pixels):
        slam.map.add_landmark(point, descriptor, 0, {0: Observation(pixel)})
    slam.last_keyframe = 0
    slam.map.record(np.eye(4), 'tracking', 0)
    slam.previous_gray = np.zeros((480, 640), np.uint8)
    slam.previous_tracks = list(enumerate(initial_pixels))
    current = {'frame': 1, 'flow_calls': 0}

    def truth():
        pose = np.eye(4); pose[0, 3] = .1 * current['frame']
        return pose

    def extract(*_):
        pixels, _ = project(world[:70], truth(), matrix)
        return pixels.astype(np.float32), descriptors[:70].copy(), np.full((70, 3), np.nan), np.full(70, np.nan)

    def flow(*_args, **_kwargs):
        current['flow_calls'] += 1
        if current['flow_calls'] % 2:
            ids = [ident for ident, _ in slam.previous_tracks]
            pixels, _ = project(world[ids], truth(), matrix)
        else:
            pixels = np.asarray([pixel for _, pixel in slam.previous_tracks])
        return pixels.astype(np.float32).reshape(-1, 1, 2), np.ones((len(pixels), 1), np.uint8), None

    slam._extract = extract
    monkeypatch.setattr(cv, 'calcOpticalFlowPyrLK', flow)
    try:
        for frame in [1, 2]:
            current['frame'] = frame
            pose, stats = slam.process(frame, np.zeros((480, 640), np.uint8))
            assert stats['tracking_ok']
            assert np.allclose(pose, truth(), atol=1e-4)
            assert stats['num_inliers'] == landmarks
            assert stats['tracked_landmarks'] == landmarks
        assert len(slam.map.keyframes) == expected_keyframes
    finally:
        slam.close()
