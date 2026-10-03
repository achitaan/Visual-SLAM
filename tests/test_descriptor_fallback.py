"""A rejected cluster of flow outliers must not discard independent map matches."""
import cv2 as cv
import numpy as np
from shared_slam import SharedSlam, MappingConfig
from mapping_geometry import project


def test_spatially_concentrated_flow_outliers_preserve_valid_descriptor_tracking(monkeypatch):
    cv.setRNGSeed(0)
    rng = np.random.default_rng(341)
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    good_pixels = rng.uniform([80., 40.], [560., 430.], (70, 2))
    bad_pixels = rng.uniform([140., 130.], [195., 285.], (300, 2))
    old_pixels = np.r_[good_pixels, bad_pixels]
    depths = rng.uniform(6., 9., len(old_pixels))
    world = np.c_[old_pixels, np.ones(len(old_pixels))] @ np.linalg.inv(matrix).T * depths[:, None]
    descriptors = rng.normal(size=(len(world), 128)).astype(np.float32)
    expected = np.eye(4);expected[0, 3] = .4
    observed, _ = project(world[:70], expected, matrix)
    slam = SharedSlam(matrix, config=MappingConfig(loop_mode='off'))
    for i, (position, descriptor) in enumerate(zip(world, descriptors)):
        lid = slam.map.add_landmark(position, descriptor, 0, {})
        assert lid == i
    slam.map.record(np.eye(4), 'tracking', None)
    slam.previous_gray = slam.current_gray = np.zeros((480, 640), np.uint8)
    slam.previous_tracks = list(enumerate(old_pixels.copy()))
    # Wrong repeated texture is forward/backward consistent, while genuine
    # descriptor matches retain their independently observed image positions.
    forward = np.r_[observed, bad_pixels].astype(np.float32).reshape(-1, 1, 2)
    backward = old_pixels.astype(np.float32).reshape(-1, 1, 2)
    results = iter((forward, backward))
    def flow(*args, **kwargs):
        return next(results).copy(), np.ones((len(old_pixels), 1), np.uint8), None
    monkeypatch.setattr(cv, 'calcOpticalFlowPyrLK', flow)
    result, stats = slam._track(observed, descriptors[:70], (640, 480))
    assert result is not None
    assert np.allclose(result[0], expected, atol=1e-5)
    assert stats['pose_source'] == 'descriptor_map_fallback'
    assert stats['flow_pose_rejection_reason'] == 'insufficient_spatial_support'
    assert stats['num_inliers'] == 70 and stats['feature_cells'] >= 3
    assert stats['reprojection_error'] < 1e-5
    slam.close()
