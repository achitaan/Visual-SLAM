from pathlib import Path
import cv2 as cv
import numpy as np
import pytest
from VisualOdometry import VisualOdometry
from StereoVisualOdometry import StereoVisualOdometry


def frontend():
    vo = VisualOdometry.__new__(VisualOdometry)
    vo.K = np.array([[700., 0, 600], [0, 700, 180], [0, 0, 1]])
    vo.P = np.c_[vo.K, [-380, 0, 0]]
    return vo


def project(points, matrix):
    pixels = (matrix @ points.T).T
    return (pixels[:, :2] / pixels[:, 2:]).astype(np.float32)


def test_monocular_pose_convention_and_outliers():
    cv.setRNGSeed(0)
    vo = frontend()
    points = np.random.default_rng(5).uniform([-3, -1, 5], [3, 1, 20], (200, 3))
    t = np.array([0.1, 0.0, -0.5])
    p1, p2 = project(points, vo.K), project(points + t, vo.K)
    p2[:30] = np.random.default_rng(4).uniform([0, 0], [1200, 360], (30, 2))
    transform, debug = vo.find_transf(p1, p2, return_debug=True)
    assert debug['tracking_ok']
    assert debug['num_inliers'] >= 150
    assert sum(debug['inlier_mask'][:30]) < 5
    assert np.allclose(transform[:3, :3], np.eye(3), atol=1e-3)
    assert np.dot(transform[:3, 3], -t / np.linalg.norm(t)) > 0.999
    assert debug['relative_scale'] == 1.0


@pytest.mark.parametrize('count', [0, 4, 7])
def test_insufficient_and_stationary_matches_hold_pose(count):
    vo = frontend()
    transform, debug = vo.find_transf(np.zeros((count, 2)), np.zeros((count, 2)), return_debug=True)
    assert np.allclose(transform, np.eye(4))
    assert not debug['tracking_ok']


def test_triangulated_descriptors_keep_indices():
    vo = frontend()
    points = np.array([[1., 0, 5], [0., 1, -5], [1., 1, 10], [0., 0, 8]])
    t = np.array([1., 0, 0])
    p1, p2 = project(points, vo.K), project(points + t, vo.K)
    result, indices = vo.triangulate_points(p1, p2, np.eye(3), t, [True, True, True, False], return_indices=True)
    assert indices == [0, 2]
    assert np.allclose(result, points[indices], atol=1e-4)


def make_sequence(tmp_path, cx_right=600, baseline=0.5):
    k_left = np.array([[700., 0, 600], [0, 700, 180], [0, 0, 1]])
    k_right = k_left.copy()
    k_right[0, 2] = cx_right
    p_left = np.c_[k_left, [0, 0, 0]]
    p_right = np.c_[k_right, [-700 * baseline, 0, 0]]
    calib = tmp_path / 'calib.txt'
    calib.write_text('P0: ' + ' '.join(map(str, p_left.ravel())) + '\nP1: ' + ' '.join(map(str, p_right.ravel())))
    for side in ('0', '1'):
        folder = tmp_path / ('image_' + side)
        folder.mkdir()
        for index in range(2):
            cv.imwrite(str(folder / f'{index:06}.png'), np.zeros((376, 1241), np.uint8))
    return str(tmp_path / 'image_'), str(calib)


def test_stereo_q_principal_point_offset_and_blank_frame(tmp_path):
    folder, calib = make_sequence(tmp_path, cx_right=610)
    vo = StereoVisualOdometry(folder, calib, use_brute_force=True, draw_matches=False)
    # Z=10, f=700, B=0.5: d = f*B/Z + cx_left-cx_right = 25.
    point = vo.Q @ np.array([600, 180, 25, 1])
    assert point[2] / point[3] == pytest.approx(10.0)
    assert vo.brute_force is not None
    transform, debug = vo.find_transf_pnp_debug(1)
    assert np.allclose(transform, np.eye(4))
    assert not debug['tracking_ok']


def test_missing_image_and_zero_baseline_fail(tmp_path):
    folder, calib = make_sequence(tmp_path, baseline=0)
    with pytest.raises(ValueError, match='positive'):
        StereoVisualOdometry(folder, calib, True, draw_matches=False)


def test_blank_monocular_features(tmp_path):
    folder, calib = make_sequence(tmp_path)
    vo = VisualOdometry(folder + '0', calib, False, camera_id=0, draw_matches=False, max_frames=2)
    p1, p2 = vo.flann_match_features(1)
    assert p1.shape == p2.shape == (0, 2)
    assert np.allclose(vo.poses[0], np.eye(4))
