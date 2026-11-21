"""Unit tests for frame-owned extraction and bounded FrontendExtractionPipeline."""

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

from frontend_pipeline import (
    ExtractedFrame,
    FrontendExtractionPipeline,
    _clone_sgbm,
    _clone_sift,
    extract_frame_owned,
)
from shared_slam import MappingConfig, SharedSlam, StereoCamera


def _synthetic_stereo_pair(seed: int = 42, shift: int = 14) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(200, 360), dtype=np.uint8)
    base = cv.GaussianBlur(base, (3, 3), 0.7)
    for y in range(16, 184, 28):
        for x in range(20, 330, 32):
            cv.rectangle(base, (x, y), (x + 12, y + 10), int((x + y) % 210 + 20), -1)
    left = base[:, shift : shift + 320].copy()
    right = base[:, 0:320].copy()
    return left, right


def _make_stereo_camera() -> tuple[np.ndarray, StereoCamera]:
    K = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = 0.54
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -160.0],
            [0.0, 1.0, 0.0, -100.0],
            [0.0, 0.0, 0.0, 500.0],
            [0.0, 0.0, 1.0 / baseline, 0.0],
        ],
        dtype=np.float32,
    )
    sgbm = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=600,
        P2=2400,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    return K, StereoCamera(sgbm, Q, baseline)


@pytest.mark.parametrize("preset", ["verified_baseline", "improved_stereo"])
def test_extract_frame_owned_matches_shared_slam_extract_exactly(preset):
    K, stereo = _make_stereo_camera()
    left, right = _synthetic_stereo_pair(seed=11)
    cfg = MappingConfig.from_preset(preset, stereo=True)
    slam = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        slam._prepare_frame_images(left, right)
        pixels, desc, points, right_u = slam._extract(left, right)
        expected_octaves = slam._current_octaves.copy()
        expected_disp = slam.current_disparity.copy()
        expected_supported = slam._supported_extraction
        expected_ver = dict(slam.stereo_depth_verification)

        # Worker-private detector & SGBM
        detector = _clone_sift(cfg, has_stereo=True)
        src_sgbm = getattr(slam, "_fast_stereo_sgbm", None) or slam.stereo.stereo
        sgbm = _clone_sgbm(src_sgbm)
        assert detector is not slam.detector
        assert detector is not slam._grid_detector
        assert sgbm is not src_sgbm

        owned = extract_frame_owned(
            0,
            left,
            right,
            config=cfg,
            inverse_K=slam.inverse_K,
            stereo_Q=slam.stereo.Q,
            stereo_search_config=slam.stereo_search_config,
            detector=detector,
            sgbm=sgbm,
        )
        assert isinstance(owned, ExtractedFrame)
        assert owned.index == 0
        assert owned.image_size == (left.shape[1], left.shape[0])
        np.testing.assert_array_equal(owned.pixels, pixels)
        np.testing.assert_array_equal(owned.descriptors, desc)
        np.testing.assert_array_equal(owned.octaves, expected_octaves)
        np.testing.assert_array_equal(owned.points, points)
        np.testing.assert_array_equal(owned.right_u, right_u)
        np.testing.assert_array_equal(owned.disparity, expected_disp)
        assert owned.stereo_depth_verification == expected_ver
        if expected_supported is not None:
            assert owned.supported_extraction is not None
            np.testing.assert_array_equal(owned.supported_extraction[0], expected_supported[0])
            np.testing.assert_array_equal(owned.supported_extraction[1], expected_supported[1])
    finally:
        slam.close(finish=False)


def test_pipelined_extraction_matches_synchronous_slam_and_preserves_order():
    K, stereo = _make_stereo_camera()
    cfg = MappingConfig.from_preset("improved_stereo", stereo=True, loop_mode="off")
    # Build 4 frames with subtle horizontal motion
    frames = [_synthetic_stereo_pair(seed=42, shift=12 + i) for i in range(4)]

    cv.setRNGSeed(0)
    np.random.seed(0)
    slam_sync = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        for i, (im_l, im_r) in enumerate(frames):
            slam_sync.process(i, im_l, im_r)
        sync_poses = [p.copy() for p in slam_sync.map.poses]
        sync_statuses = list(slam_sync.map.statuses)
    finally:
        slam_sync.close(finish=True)

    cv.setRNGSeed(0)
    np.random.seed(0)
    slam_pipe = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        with FrontendExtractionPipeline(slam_pipe, buffer_size=2) as pipe:
            pipe.submit(0, frames[0][0], frames[0][1])
            pipe.submit(1, frames[1][0], frames[1][1])
            next_sub = 2
            for i in range(len(frames)):
                ext = pipe.get(i)
                if next_sub < len(frames):
                    pipe.submit(next_sub, frames[next_sub][0], frames[next_sub][1])
                    next_sub += 1
                slam_pipe.process(i, extracted=ext)
        pipe_poses = [p.copy() for p in slam_pipe.map.poses]
        pipe_statuses = list(slam_pipe.map.statuses)
    finally:
        slam_pipe.close(finish=True)

    assert pipe_statuses == sync_statuses
    assert len(pipe_poses) == len(sync_poses)
    for p_s, p_p in zip(sync_poses, pipe_poses):
        np.testing.assert_allclose(p_p, p_s, atol=1e-9)


def test_pipeline_error_propagation_and_clean_shutdown():
    K, stereo = _make_stereo_camera()
    cfg = MappingConfig.from_preset("improved_stereo", stereo=True, loop_mode="off")
    slam = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        pipe = FrontendExtractionPipeline(slam, buffer_size=1)
        try:
            def failing_loader(idx: int):
                raise ValueError(f"synthetic loader failure on frame {idx}")

            pipe.submit_loader(0, failing_loader)
            with pytest.raises(ValueError, match="synthetic loader failure on frame 0"):
                pipe.get(0)
        finally:
            pipe.close()
            assert not pipe._thread.is_alive()
    finally:
        slam.close(finish=False)
