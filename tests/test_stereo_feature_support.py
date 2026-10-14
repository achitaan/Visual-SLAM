"""Weak texture must supply real stereo support without changing pose acceptance."""
import cv2 as cv
import numpy as np
from shared_slam import SharedSlam, MappingConfig, StereoCamera
from mapping_geometry import coverage


def texture_pair():
    rng = np.random.default_rng(62)
    texture = cv.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (0, 0), 1.)
    left = np.clip(128+(texture.astype(float)-128)/4, 0, 255).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, :-8] = left[:, 8:]
    return left, right


def camera():
    baseline = .54
    matrix = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])
    q = np.array([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 250], [0, 0, 1/baseline, 0.]])
    stereo = cv.StereoSGBM_create(numDisparities=96, blockSize=5,
        P1=8*3*25, P2=32*3*25, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32)
    return matrix, StereoCamera(stereo, q, baseline)


def test_low_contrast_stereo_texture_initializes_and_tracks_without_pose_relaxation():
    left, right = texture_pair()
    matrix, stereo = camera()
    assert not cv.SIFT_create(nfeatures=1500).detect(left)
    slam = SharedSlam(matrix, stereo, MappingConfig(loop_mode='off'))
    _, first = slam.process(0, left, right)
    assert first['tracking_ok'] and first['valid_stereo_depth'] >= 30
    observed = slam.map.keyframes[0].pixels
    assert coverage(observed, (640, 480)) >= 4
    _, second = slam.process(1, left, right)
    assert second['tracking_ok'] and second['num_inliers'] >= 15
    assert second['feature_cells'] >= 3 and second['reprojection_error'] <= 1.5
    assert second['stereo_verification'] == 'current_right_reprojection'
    assert np.linalg.norm(slam.map.poses[-1][:3, 3]) < .01
    slam.close()


def test_blank_stereo_images_still_cannot_initialize():
    matrix, stereo = camera()
    slam = SharedSlam(matrix, stereo, MappingConfig(loop_mode='off'))
    blank = np.full((480, 640), 128, np.uint8)
    _, first = slam.process(0, blank, blank)
    assert not first['tracking_ok'] and first['state'] == 'initializing'
    assert not slam.map.landmarks and not slam.map.keyframes
    slam.close()
