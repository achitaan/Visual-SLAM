"""Unit tests for spatial grid-bucketed SIFT extraction and stereo classification (F9)."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend_features import (
    classify_stereo_features,
    extract_grid_bucketed_sift,
    select_grid_bucketed_indices,
)


def test_grid_bucketing_balances_clumped_vs_sparse_regions():
    rng = np.random.default_rng(12)
    # 900 high-response features clumped in left quarter + 200 medium-response spread across full image
    px_clumped = np.column_stack((rng.uniform(0, 200, 900), rng.uniform(0, 376, 900))).astype(np.float32)
    resp_clumped = rng.uniform(0.08, 0.15, 900).astype(np.float32)

    px_spread = np.column_stack((rng.uniform(250, 1240, 200), rng.uniform(0, 376, 200))).astype(np.float32)
    resp_spread = rng.uniform(0.03, 0.06, 200).astype(np.float32)

    pixels = np.vstack((px_clumped, px_spread))
    responses = np.concatenate((resp_clumped, resp_spread))

    kept = select_grid_bucketed_indices(pixels, responses, (1241, 376), target_features=300, grid_cols=16, grid_rows=6)
    assert len(kept) == 300
    # Spread region (indices >= 900) must be well-represented despite lower raw response
    spread_selected = int(np.sum(kept >= 900))
    assert spread_selected >= 100, f"Expected >= 100 spread features, got {spread_selected}"


def test_scaled_sift_maps_pixel_centers_back_to_the_full_image():
    shape = (80, 160)
    detected = np.array([[12.25, 10.0], [46.0, 30.75]], dtype=np.float32)
    desc = np.arange(256, dtype=np.float32).reshape(2, 128)

    def extract(image, **kwargs):
        assert image.shape == (40, 80)
        return detected, desc, np.zeros(2, np.int16), np.zeros(2, np.float32), np.ones(2, np.float32)

    actual = extract_grid_bucketed_sift(
        np.zeros(shape, dtype=np.uint8), target_features=2, detection_scale=0.5, native_sift=extract
    )

    # Each reduced pixel spans two original pixels, so its center shifts by
    # half a full-resolution pixel after doubling its reduced coordinate.
    expected = np.array([[25.0, 20.5], [92.5, 62.0]], dtype=np.float32)
    np.testing.assert_array_equal(actual.pixels, expected)
    np.testing.assert_array_equal([point.pt for point in actual.keypoints], expected)
    np.testing.assert_array_equal(actual.descriptors, desc)


def test_scaled_sift_retries_full_resolution_when_feature_support_is_low():
    gray = np.zeros((80, 160), dtype=np.uint8)
    pixels = np.array([[30, 20], [60, 40]], dtype=np.float32)
    calls = []

    def extract(image, **kwargs):
        calls.append(image.shape)
        count = 2 if image is gray else 1
        return (pixels[:count], np.zeros((count, 128), np.float32), np.zeros(count, np.int16),
                np.zeros(count, np.float32), np.ones(count, np.float32))

    actual = extract_grid_bucketed_sift(gray, target_features=2, detection_scale=0.5, native_sift=extract)

    assert calls == [(40, 80), gray.shape]
    np.testing.assert_array_equal(actual.pixels, pixels)


def test_extract_grid_bucketed_sift_and_classify_stereo():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(376, 1241), dtype=np.uint8)
    img = cv.GaussianBlur(img, (5, 5), 1.0)
    # Add structured corners across image
    for x in range(40, 1200, 60):
        for y in range(40, 340, 60):
            cv.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), 255, -1)
            cv.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), 0, -1)

    bucketed = extract_grid_bucketed_sift(img, target_features=400, grid_cols=16, grid_rows=6)
    assert 50 <= len(bucketed.pixels) <= 400
    assert bucketed.descriptors.shape == (len(bucketed.pixels), 128)
    assert bucketed.octaves.shape == (len(bucketed.pixels),)
    assert bucketed.angles.shape == (len(bucketed.pixels),)
    assert bucketed.responses.shape == (len(bucketed.pixels),)

    # Simulate half close (disp=35 px => Z ~ 11 m) and half far (disp=4 px => Z ~ 96 m)
    n = len(bucketed.pixels)
    right_u = np.full(n, np.nan, dtype=np.float32)
    right_u[: n // 2] = bucketed.pixels[: n // 2, 0] - 35.0
    right_u[n // 2 :] = bucketed.pixels[n // 2 :, 0] - 4.0

    feats = classify_stereo_features(bucketed, right_u, fx=718.856, baseline=0.537)
    assert int(feats.is_close_stereo.sum()) == n // 2
    assert int(feats.is_far_bearing.sum()) == n - n // 2
    assert np.all(feats.sigma_disparity_px > 0)
