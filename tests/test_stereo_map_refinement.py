"""Current stereo must disambiguate map hypotheses that fit the left image alone."""
import numpy as np
from mapping_geometry import project, right_pixel, refine_stereo_map_pose

K = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])


def scene():
    return np.random.default_rng(931).uniform([-4., -3., 4.], [4., 3., 8.], (120, 3))


def test_left_perfect_hypothesis_with_wrong_depth_is_rejected():
    world = scene()
    pixels, depth = project(world, np.eye(4), K)
    right = right_pixel(pixels[:, 0], depth, K[0, 0], .54)
    false_world = world*.5  # Same left rays, incompatible metric stereo depths.
    assert np.allclose(project(false_world, np.eye(4), K)[0], pixels)
    result, stats = refine_stereo_map_pose((np.eye(4), np.arange(120), 0.),
        false_world, pixels, right, K, .54, (640, 480))
    assert result is None and stats['pose_rejection_reason'] == 'stereo_map_inconsistency'


def test_stereo_refinement_improves_pose_and_excludes_depth_outliers():
    world = scene()
    truth = np.eye(4);truth[0, 3] = .3
    pixels, depth = project(world, truth, K)
    right = right_pixel(pixels[:, 0], depth, K[0, 0], .54)
    right[:4] += 20.
    seed = truth.copy();seed[1, 3] += .01
    result, stats = refine_stereo_map_pose((seed, np.arange(120), .5),
        world, pixels, right, K, .54, (640, 480))
    assert result is not None and stats['stereo_refinement_applied']
    assert stats['stereo_final_cost'] < stats['stereo_initial_cost']
    assert np.linalg.norm(result[0][:3, 3]-truth[:3, 3]) < .01
    assert not set(range(4)) & set(result[1])
    assert stats['stereo_rejected_inliers'] == 4


def test_missing_depth_keeps_geometric_pose_without_claiming_stereo_verification():
    world = scene();pixels, _ = project(world, np.eye(4), K)
    initial = (np.eye(4), np.arange(120), 0.)
    result, stats = refine_stereo_map_pose(initial, world, pixels,
        np.full(120, np.nan), K, .54, (640, 480))
    assert result is initial
    assert stats['stereo_verification'] == 'insufficient_current_depth_support'


def test_rectified_principal_point_offset_is_respected():
    world = scene();pixels, depth = project(world, np.eye(4), K)
    right = right_pixel(pixels[:, 0], depth, K[0, 0], .54, 15.)
    result, stats = refine_stereo_map_pose((np.eye(4), np.arange(120), 0.),
        world, pixels, right, K, .54, (640, 480), disparity_offset=15.)
    assert result is not None and np.allclose(result[0], np.eye(4))
    assert stats['stereo_consistent_inliers'] == 120
