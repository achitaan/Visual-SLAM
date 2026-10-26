"""Manufactured baseline characterization; no missing overlay is imported."""
import cv2 as cv
import numpy as np
import pytest

from mapping_geometry import _estimate_pose_hypothesis, project


@pytest.mark.xfail(strict=True, reason="Baseline returns pre-LM mask without final per-row pixel gate")
def test_returned_pnp_rows_remain_inside_two_pixel_gate_after_lm(monkeypatch):
    matrix = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
    pixels = np.array([[u, v] for u in [80., 240., 400., 560.]
                       for v in [80., 240., 400.]], dtype=float)
    depth = np.full(len(pixels), 10.)
    depth[0] = 1.
    points = np.column_stack(((pixels[:, 0]-320.)/250.*depth,
                              (pixels[:, 1]-240.)/250.*depth, depth))
    # RANSAC's pose initially projects every row exactly. LM moves the camera
    # by .02 world units: far-row residuals .5px, one near-row residual 5px.
    monkeypatch.setattr(cv, "solvePnPRansac", lambda *a, **k:
                        (True, np.zeros((3, 1)), np.zeros((3, 1)),
                         np.arange(len(points)).reshape(-1, 1)))
    monkeypatch.setattr(cv, "solvePnPRefineLM", lambda *a, **k:
                        (np.zeros((3, 1)), np.array([[.02], [0.], [0.]])))
    result = _estimate_pose_hypothesis(points, pixels, matrix, (640, 480), min_inliers=10)
    assert result is not None
    pose, ids, _ = result
    predicted, depths = project(points[ids], pose, matrix)
    assert np.all(depths > 0)
    assert np.all(np.linalg.norm(predicted-pixels[ids], axis=1) <= 2.)
