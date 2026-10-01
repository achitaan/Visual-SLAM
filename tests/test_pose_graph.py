import numpy as np
import json
from pathlib import Path
import pytest
from scipy.spatial.transform import Rotation
from pose_graph import optimize
from SLAM import SLAM
from slam_backend import SlamBackend
import slam_backend


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_independent_loop_constraint_reduces_drift_and_fixes_origin():
    initial = [pose(i * 1.1) for i in range(21)]
    edges = [(i, i + 1, pose(1.1), np.eye(6), False) for i in range(20)]
    edges.append((0, 20, pose(20), np.eye(6) * 100, True))
    result = optimize(initial, edges)
    assert np.allclose(result[0], initial[0])
    assert abs(result[-1][0, 3] - 20) < 0.02
    assert np.mean([(value[0, 3] - i) ** 2 for i, value in enumerate(result)]) < 0.01


def test_relative_edge_convention_with_rotated_origin():
    first = pose(5)
    first[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    expected = first @ pose(2)
    inaccurate = expected.copy()
    inaccurate[:3, 3] += [1, 0, 0]
    result = optimize([first, inaccurate], [(0, 1, pose(2), np.eye(6), False)])
    assert np.allclose(result[1], expected, atol=1e-5)


def test_invalid_graph_vertex_rejected():
    with pytest.raises(ValueError, match='invalid vertex'):
        optimize([pose(0), pose(1)], [(0, 2, pose(1), np.eye(6), False)])


def test_vertices_exist_before_vocabulary_and_missing_descriptors(monkeypatch):
    monkeypatch.setattr(slam_backend, 'VOCAB_BUILD_MIN_FRAMES', 100)
    backend = SlamBackend()
    for i in range(3):
        backend.maybe_add_keyframe(index=i * 15, pose_T_wc=pose(i), image=None,
                                  descriptors=None, timestamp=float(i), should_add=True)
    assert len(backend._slam.initial_poses) == 3
    assert len(backend._slam.odometry_edges) == 2
    backend._optimized_poses = [pose(0), pose(1), pose(2)]
    assert backend.pose_graph_state(30)['optimized_pose_T_wc'][0][3] == 2
    assert backend.pose_graph_state(31) is None


def test_appearance_candidates_do_not_create_unverified_loop_edges(monkeypatch):
    monkeypatch.setattr(slam_backend, 'VOCAB_BUILD_MIN_FRAMES', 2)
    monkeypatch.setattr(slam_backend, 'VOCAB_NUM_CLUSTERS', 2)
    backend = SlamBackend()
    descriptors = np.random.default_rng(3).normal(size=(30, 32)).astype(np.float32)
    for i in range(5):
        backend.maybe_add_keyframe(index=i * 15, pose_T_wc=pose(i), image=None,
                                  descriptors=descriptors, timestamp=float(i), should_add=True)
    assert len(backend._slam.initial_poses) == 5
    assert len(backend._slam.loop_edges) == 0


def test_large_loop_with_mixed_rotation_translation_scales_converges():
    truth, initial = [], []
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 40)):
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler('y', angle).as_matrix()
        expected[:3, 3] = [100 * np.sin(angle), 0, 100 * (np.cos(angle) - 1)]
        truth.append(expected)
        noisy = expected.copy()
        noisy[:3, :3] = Rotation.from_euler('y', angle + .001 * i).as_matrix()
        noisy[:3, 3] += [.15 * i, 0, .08 * i]
        initial.append(noisy)
    edges = [(i, i + 1, np.linalg.inv(initial[i]) @ initial[i + 1],
              np.diag([1000.] * 3 + [10.] * 3), False) for i in range(39)]
    for last in (37, 39):
        edges.append((0, last, np.linalg.inv(truth[0]) @ truth[last],
                      np.diag([2000.] * 3 + [20.] * 3), True))
    corrected = optimize(initial, edges, max_evaluations=500)
    assert np.allclose(corrected[0], initial[0])
    assert np.linalg.norm(corrected[-1][:3, 3] - truth[-1][:3, 3]) < .2


def test_real_seven_loop_graph_converges_without_ground_truth():
    fixture = json.loads((Path(__file__).parent / 'fixtures/kitti06_graph.json').read_text())
    initial = [np.array(pose) for pose in fixture['initial_poses']]
    indices = fixture['indices']
    odometry = np.diag([1000.] * 3 + [10.] * 3)
    loop_information = np.diag([2000.] * 3 + [20.] * 3)
    edges = [(i, i + 1, np.linalg.inv(initial[i]) @ initial[i + 1], odometry, False) for i in range(len(initial) - 1)]
    loop_edges = [(indices.index(loop['first_frame']), indices.index(loop['second_frame']),
                   np.array(loop['measurement']), loop_information, True) for loop in fixture['loops']]
    # The iterative linear solve stalled at 5,000 evaluations on this graph.
    diagnostics = {}
    corrected = optimize(initial, edges + loop_edges, max_evaluations=100, diagnostics=diagnostics)
    assert diagnostics['success']
    assert diagnostics['final_cost'] < diagnostics['initial_cost']
    assert diagnostics['solver'] == 'dense SVD with grouped forward differences'
    assert np.array_equal(corrected[0], initial[0])
    residual = sum(np.linalg.norm((np.linalg.inv(corrected[first]) @ corrected[second])[:3, 3] - measurement[:3, 3]) for first, second, measurement, *_ in loop_edges)
    assert residual < 1.0  # Sum across seven independently measured loop translations.


def test_grouped_graph_jacobian_matches_independent_forward_differences():
    from pose_graph import graph_vertex_groups, dense_graph_jacobian
    first = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3])
    second = np.array([1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 5])
    groups, rows = graph_vertex_groups(first, second, 9)
    parameters = np.random.default_rng(19).normal(size=48)
    parameters[3::6] *= 100
    def residual(values):
        state = np.r_[np.zeros(6), values].reshape(-1, 6)
        rotations = Rotation.from_rotvec(state[:, :3]).as_matrix()
        inverse = rotations[first].transpose(0, 2, 1)
        rotation_error = Rotation.from_matrix(inverse @ rotations[second]).as_rotvec()
        translation_error = np.einsum('nij,nj->ni', inverse, state[second, 3:] - state[first, 3:])
        return np.c_[rotation_error, translation_error].ravel()
    observed = dense_graph_jacobian(residual, parameters, groups, rows)
    expected = np.empty_like(observed)
    step = np.sqrt(np.finfo(float).eps) * np.where(parameters >= 0, 1., -1.) * np.maximum(1., np.abs(parameters))
    baseline = residual(parameters)
    for column in range(len(parameters)):
        perturbed = parameters.copy()
        perturbed[column] += step[column]
        expected[:, column] = (residual(perturbed) - baseline) / (perturbed[column] - parameters[column])
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-12)
    assert len(groups) * 6 < len(parameters)
