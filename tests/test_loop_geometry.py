import cv2 as cv
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from loop_geometry import StereoLoopFrame, verify_loop, propagate_corrections
from SLAM import SLAM


def scene():
    rng = np.random.default_rng(12)
    matrix = np.array([[700., 0, 600], [0, 700, 180], [0, 0, 1]])
    points = rng.uniform([-5, -2, 8], [5, 2, 25], (200, 3)).astype(np.float32)
    descriptors = rng.normal(size=(200, 128)).astype(np.float32)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', 3, degrees=True).as_matrix()
    transform[:3, 3] = [.2, 0, -.5]
    current_points = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    def pixels(values):
        p = values @ matrix.T
        return (p[:, :2] / p[:, 2:]).astype(np.float32)
    first = StereoLoopFrame(pixels(points), points, descriptors, (1241, 376))
    second = StereoLoopFrame(pixels(current_points), current_points, descriptors.copy(), (1241, 376))
    return first, second, matrix, transform


def test_metric_loop_direction_with_outliers_and_bidirectional_geometry():
    cv.setRNGSeed(0)
    first, second, matrix, transform = scene()
    second.pixels[:35] = np.random.default_rng(4).uniform([0, 0], [1200, 360], (35, 2))
    loop = verify_loop(first, second, matrix)
    assert loop is not None
    assert loop['inliers'] >= 160
    assert loop['median_reprojection_px'] < .01
    assert np.allclose(loop['measurement'], np.linalg.inv(transform), atol=1e-4)


def test_appearance_match_without_geometry_is_rejected():
    cv.setRNGSeed(0)
    first, second, matrix, _ = scene()
    # Identical descriptors alone cannot authorize a loop edge.
    second.pixels[:] = np.random.default_rng(8).uniform([0, 0], [1200, 360], (200, 2))
    assert verify_loop(first, second, matrix) is None


def test_graph_corrections_reach_intermediate_frames_and_anchors():
    raw = [np.eye(4) for _ in range(11)]
    for i, pose in enumerate(raw):
        pose[0, 3] = i
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler('y', 20, degrees=True).as_matrix()
    correction[:3, 3] = [0, 1, 2]
    optimized = [raw[0], correction @ raw[10]]
    result = propagate_corrections(raw, [0, 10], optimized)
    assert np.allclose(result[0], optimized[0])
    assert np.allclose(result[10], optimized[1])
    assert result[5][1, 3] == pytest.approx(.5)
    assert np.linalg.det(result[5][:3, :3]) == pytest.approx(1.)
    assert np.allclose(result[5][:3, :3], Rotation.from_euler('y', 10, degrees=True).as_matrix())


def test_graph_propagation_refuses_incomplete_anchor_coverage():
    with pytest.raises(ValueError, match='span'):
        propagate_corrections([np.eye(4)] * 3, [0, 1], [np.eye(4)] * 2)


def test_image_measurement_corrects_graph_through_slam_wrapper():
    cv.setRNGSeed(0)
    first, second, matrix, _ = scene()
    measurement = verify_loop(first, second, matrix)['measurement']
    graph = SLAM()
    # A drifted prediction connects the same two camera frames through 10 odometry edges.
    for i in range(11):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(measurement[:3, :3]).as_rotvec() * i / 10).as_matrix()
        pose[:3, 3] = (measurement[:3, 3] + [1., 0, 0]) * i / 10
        graph.initial_poses.append(pose)
        if i:
            graph.add_odometry_edge(i - 1, i, np.linalg.inv(graph.initial_poses[i - 1]) @ pose)
    graph.add_loop_closure_edge(0, 10, measurement, np.eye(6) * 100)
    result = graph.optimize_pose_graph(100)
    assert np.allclose(result[0], graph.initial_poses[0])
    assert np.linalg.norm(result[-1][:3, 3] - measurement[:3, 3]) < .02
    assert np.allclose(result[-1][:3, :3], measurement[:3, :3], atol=1e-3)
