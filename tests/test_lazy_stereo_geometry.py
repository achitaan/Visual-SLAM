from types import SimpleNamespace

import cv2 as cv
import numpy as np

from lazy_stereo_geometry import refine_with_lazy_stereo_geometry, seed_lazy_stereo_geometry
from mapping_geometry import project, right_pixel
from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam, StereoCamera


K = np.array([[700., 0., 620.], [0., 700., 188.], [0., 0., 1.]])
SIZE = (1241, 376)
BASELINE = .54


def test_seeded_geometry_follows_anchor_correction_and_expires():
    image = np.random.default_rng(319).integers(0, 256, SIZE[::-1], dtype=np.uint8)
    anchor_pose = np.eye(4)
    anchor_pose[:3, 3] = [1., -2., .25]
    anchor = SimpleNamespace(id=7, pose=anchor_pose)
    pose = np.eye(4)
    pose[:3, 3] = [3., .5, .75]
    inverse_K = np.linalg.inv(K)

    def stereo(pixels):
        depth = np.full(len(pixels), 12.)
        camera_points = (np.column_stack((pixels, np.ones(len(pixels)))) @ inverse_K.T) * depth[:, None]
        return camera_points, right_pixel(pixels[:, 0], depth, K[0, 0], BASELINE)

    geometry = seed_lazy_stereo_geometry(11, anchor, pose, image, np.empty((0, 2)), SIZE, stereo)
    assert geometry is not None
    assert geometry.active(12, 7) and not geometry.active(12, 8)
    assert not geometry.active(13, 7) and geometry.active(13, 7, require_previous=False)
    assert not geometry.active(15, 7, require_previous=False)
    # The manufactured points lie on the plane z = 12 in the original camera.
    pixels, depth = project(geometry.world_points(anchor_pose), pose, K)
    np.testing.assert_allclose(pixels, geometry.pixels, atol=1e-10, rtol=0)
    np.testing.assert_allclose(depth, 12., atol=1e-10, rtol=0)
    shift = np.array([4., -3., .375])
    corrected = anchor_pose.copy()
    corrected[:3, 3] += shift
    np.testing.assert_allclose(geometry.world_points(corrected), geometry.world_points(anchor_pose) + shift)
    advanced = geometry.advance(12, np.arange(20), geometry.pixels[:20])
    assert advanced.active(13, 7) and not np.shares_memory(advanced.pixels, geometry.pixels)
    assert advanced.advance(14, np.arange(20), advanced.pixels) is None


def test_refreshed_tracks_require_independent_right_reprojection():
    x, y = np.meshgrid(np.linspace(110., 1130., 10), np.linspace(30., 346., 9))
    pixels = np.column_stack((x.ravel(), y.ravel()))
    is_persistent = np.arange(len(pixels)) % 3 != 2
    pixels = np.concatenate((pixels[is_persistent], pixels[~is_persistent]))
    depth = np.linspace(8., 29., len(pixels))
    points = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T * depth[:, None]
    stereo = SimpleNamespace(baseline=BASELINE, disparity_offset=.25)
    right = right_pixel(pixels[:, 0], depth, K[0, 0], BASELINE, stereo.disparity_offset)
    primary_count = int(is_persistent.sum())
    solution = np.eye(4), np.arange(len(points)), 0.

    def refine(measured_right):
        return refine_with_lazy_stereo_geometry(
            solution, points, pixels, measured_right, primary_count, K, stereo, SIZE, 15, None,
        )

    # The manufactured left and right pixels are exact projections of the point cloud.
    pose, inliers, median = refine(right)
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(inliers, np.arange(len(points)))
    assert median < 1e-10
    corrupted = right.copy()
    corrupted[primary_count:] += 3.
    assert refine(corrupted) is None
    corrupted = right.copy()
    corrupted[:primary_count] = np.nan
    assert refine(corrupted) is None


def test_lazy_stereo_geometry_keeps_the_cpu_fallback_schedule(monkeypatch):
    import cuda_optical_flow

    monkeypatch.setattr(cuda_optical_flow, "create_cuda_optical_flow", lambda: (None, "test unavailable"))
    config = MappingConfig(use_lazy_stereo_geometry=True, stereo_depth_policy="verified_all",
                           stereo_map_depth_policy="verified")
    performance = PerformanceConfig(pipeline_extraction=True, optical_flow_backend="cuda")
    q = np.array([[1., 0., 0., -620.], [0., 1., 0., -188.], [0., 0., 0., 700.],
                  [0., 0., 1. / BASELINE, 0.]])
    stereo = StereoCamera(cv.StereoSGBM_create(numDisparities=16, blockSize=5), q, BASELINE)
    slam = SharedSlam(K, stereo=stereo, config=config, performance=performance)
    try:
        assert not slam._lazy_stereo_geometry_enabled and slam._lazy_stereo_geometry is None
        assert slam.loop_worker.min_schedule_frame_gap == 20
    finally:
        slam.close(finish=False)
