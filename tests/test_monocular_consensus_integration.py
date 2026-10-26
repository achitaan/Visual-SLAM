"""Independent enabled OpenCV integration and analytic final-consensus edges."""
import cv2 as cv
import numpy as np
import pytest
import mapping_geometry as geometry

K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
SIZE = (640, 480)


def scene(n=24):
    pixels = np.array([[80.+160.*(i % 4), 80.+160.*((i//4) % 3)] for i in range(n)])
    depths = np.full(n, 10.)
    points = np.c_[(pixels[:, 0]-320.)/250.*depths,
                   (pixels[:, 1]-240.)/250.*depths, depths]
    return points, pixels


def filter_pose(points, pixels, ids=None, rv=None, tv=None, minimum=10):
    diagnostics = {}
    result = geometry._validated_refined_consensus(
        points, pixels, K, SIZE, minimum,
        np.arange(len(points)) if ids is None else np.asarray(ids, int),
        np.zeros((3, 1)) if rv is None else rv,
        np.zeros((3, 1)) if tv is None else tv, diagnostics)
    return result, diagnostics


def test_real_opencv_enabled_pose_solver_returns_valid_final_rows():
    cv.setRNGSeed(0)
    rng = np.random.default_rng(91)
    points = rng.uniform([-5., -4., 5.], [5., 4., 15.], (120, 3))
    expected = np.eye(4)
    expected[:3, :3] = cv.Rodrigues(np.array([.03, -.05, .02]))[0]
    expected[:3, 3] = [.2, -.1, .3]
    pixels, _ = geometry.project(points, expected, K)
    report = {}
    result = geometry.estimate_pose(points, pixels, K, SIZE, min_inliers=20,
                                   revalidate_inliers=True, diagnostics=report)
    assert result is not None
    pose, ids, error = result
    predicted, depth = geometry.project(points[ids], pose, K)
    assert len(ids) >= 100 and error < 1e-4
    assert np.all(depth > 0) and np.max(np.linalg.norm(predicted-pixels[ids], axis=1)) < 1e-3
    assert np.allclose(pose, expected, atol=1e-4)
    assert report['refinement_consensus_revalidated'] is True
    assert report['consensus_refits'] <= 1


def test_filter_preserves_rows_after_nonidentity_rigid_frame_transform():
    points, pixels = scene()
    original, _ = filter_pose(points, pixels)
    rotation = cv.Rodrigues(np.array([.4, -.2, .3]))[0]
    translation = np.array([3., -5., 2.])
    transformed = points @ rotation.T + translation
    result, report = filter_pose(
        transformed, pixels, rv=cv.Rodrigues(rotation.T)[0],
        tv=(-rotation.T @ translation).reshape(3, 1))
    assert np.array_equal(result[1], original[1])
    assert result[2] < 1e-10 and report['consensus_refits'] == 0
    assert np.allclose(result[0][:3, :3], rotation)
    assert np.allclose(result[0][:3, 3], translation)


def test_final_refit_can_prune_below_count_gate(monkeypatch):
    points, pixels = scene()
    points[0] *= .001
    calls = []
    def extra_fit(*args):
        calls.append(1)
        return np.zeros((3, 1)), np.array([[.2], [0.], [0.]])
    monkeypatch.setattr(cv, 'solvePnPRefineLM', extra_fit)
    result, report = filter_pose(points, pixels, tv=np.array([[.001], [0.], [0.]]), minimum=20)
    assert result is None and len(calls) == 1
    assert report['consensus_refits'] == 1
    assert report['pose_rejection_reason'] == 'insufficient_refined_inlier_support'
    assert report['candidate_inliers'] == 0


def test_final_refit_can_destroy_spatial_coverage(monkeypatch):
    pixels = np.tile([80., 80.], (24, 1))
    pixels[1] = [320., 80.]; pixels[2] = [560., 80.]
    depths = np.full(24, 100.); depths[0] = .1; depths[1:3] = 10.
    points = np.c_[(pixels[:, 0]-320.)/250.*depths,
                   (pixels[:, 1]-240.)/250.*depths, depths]
    monkeypatch.setattr(cv, 'solvePnPRefineLM',
                        lambda *a: (np.zeros((3, 1)), np.array([[.1], [0.], [0.]])))
    result, report = filter_pose(points, pixels, tv=np.array([[.001], [0.], [0.]]))
    assert result is None and report['candidate_inliers'] == 21
    assert report['consensus_refits'] == 1 and report['candidate_feature_cells'] == 1
    assert report['pose_rejection_reason'] == 'insufficient_refined_spatial_support'


def test_above_two_pixel_gate_is_removed_not_rounded_into_consensus(monkeypatch):
    points, pixels = scene()
    pixels[0, 0] += 2.0001
    monkeypatch.setattr(cv, 'solvePnPRefineLM',
                        lambda *a: (np.zeros((3, 1)), np.zeros((3, 1))))
    result, report = filter_pose(points, pixels)
    assert result[1].tolist() == list(range(1, 24))
    assert report['consensus_refits'] == 1


def test_exact_quarter_of_original_inputs_passes_fraction_gate():
    points, pixels = scene(48)
    result, report = filter_pose(points, pixels, ids=range(12))
    assert result is not None and len(result[1]) == 12
    assert report['candidate_inlier_ratio'] == .25
