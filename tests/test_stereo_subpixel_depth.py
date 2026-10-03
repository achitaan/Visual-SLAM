"""Metric depth and right observations must share the actual feature coordinate."""
import numpy as np
import pytest
from shared_slam import SharedSlam, StereoCamera


def camera(offset=0.):
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    baseline = .2
    q = np.array([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 250],
                  [0, 0, 1/baseline, -offset/baseline]], float)
    return SharedSlam(matrix, stereo=StereoCamera(None, q, baseline))


@pytest.mark.parametrize('offset', [0., 2.])
def test_fractional_features_reconstruct_continuous_disparity_surface(offset):
    slam = camera(offset)
    y, x = np.mgrid[:480, :640]
    slam.current_disparity = 5. + .02*x + .01*y + offset
    pixels = np.array([[320.2, 241.25], [333.75, 245.125], [321.125, 230.7]])
    disparity = 5. + .02*pixels[:, 0] + .01*pixels[:, 1] + offset
    try:
        points, right = slam._measure_stereo_pixels(pixels)
        assert np.allclose(right, pixels[:, 0]-disparity, atol=1e-8, rtol=0)
        depth = 50./(disparity-offset)
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(slam.K).T
        assert np.allclose(points, rays*depth[:, None], atol=1e-8, rtol=0)
    finally:
        slam.close()


@pytest.mark.parametrize('bad', [-1., 30.])
def test_invalid_neighbors_and_depth_edges_do_not_supply_fractional_depth(bad):
    slam = camera()
    slam.current_disparity = np.full((480, 640), 10.)
    slam.current_disparity[241, 321] = bad
    try:
        points, right = slam._measure_stereo_pixels(np.array([[320.25, 240.25], [320., 240.]]))
        assert not np.isfinite(points[0]).any() and np.isnan(right[0])
        # Zero-weight neighbors cannot discard a valid observation at a pixel center.
        assert np.isfinite(points[1]).all() and right[1] == 310.
    finally:
        slam.close()
