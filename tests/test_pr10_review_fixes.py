"""Regression tests for PR #10 code review findings."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest import mock

import cv2 as cv
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend_pipeline import FrontendExtractionPipeline
from local_map_tracker import fuse_neighbor_landmarks, triangulate_far_stereo_landmarks
import loop_harness
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from slam_state import MappingKeyframe, Observation


def _make_camera_and_stereo(w: int = 640, h: int = 240) -> tuple[np.ndarray, StereoCamera]:
    K = np.array([[500.0, 0.0, w / 2.0], [0.0, 500.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = 0.54
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -w / 2.0],
            [0.0, 1.0, 0.0, -h / 2.0],
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


def test_descriptor_bank_invalidates_after_in_place_fusion():
    """[P1] In-place landmark_ids update during fusion must invalidate descriptor row cache."""
    K, stereo = _make_camera_and_stereo()
    cfg = MappingConfig.from_preset("improved_stereo", stereo=True, loop_mode="off")
    slam = SharedSlam(K, stereo=stereo, config=cfg)
    try:
        desc_a = np.zeros(128, dtype=np.float32)
        desc_a[0] = 1.0
        desc_b = np.zeros(128, dtype=np.float32)
        desc_b[0] = 0.99
        desc_b[1] = float(np.sqrt(1.0 - 0.99 * 0.99))

        # World point at (0, 0, 10) projects to (320, 120) in kf0 (pose=I) and (320, 120) in kf1 (pose=I)
        pt = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        px = np.array([[320.0, 120.0]], dtype=np.float32)

        kf0 = MappingKeyframe(
            id=0,
            frame=0,
            pose=np.eye(4, dtype=np.float64),
            pixels=px.copy(),
            descriptors=desc_a.reshape(1, 128).copy(),
            landmark_ids=np.array([0], dtype=np.int64),
            image_size=(640, 240),
            octaves=np.array([0], dtype=np.int16),
        )
        kf1 = MappingKeyframe(
            id=1,
            frame=1,
            pose=np.eye(4, dtype=np.float64),
            pixels=px.copy(),
            descriptors=desc_b.reshape(1, 128).copy(),
            landmark_ids=np.array([-1], dtype=np.int64),
            image_size=(640, 240),
            octaves=np.array([0], dtype=np.int16),
        )
        slam.map.keyframes[0] = kf0
        slam.map.keyframes[1] = kf1
        lid = slam.map.add_landmark(
            pt, desc_a.copy(), 0, {0: Observation(px[0].copy(), None, octave=0)}
        )
        lm = slam.map.landmarks[lid]
        kf0.landmark_ids[0] = lm.id

        # Warm the descriptor bank cache for both keyframes (kf1 currently has no linked landmark)
        descs_before, ids_before = slam._landmark_descriptor_bank([lm])
        assert len(ids_before) == 1
        assert int(ids_before[0]) == lm.id

        # Fuse kf0's landmark into kf1 (mutates kf1.landmark_ids in place)
        fused = fuse_neighbor_landmarks(slam.map, kf1, [0], K, stereo.baseline, radius_px=4.0)
        assert fused == 1
        assert int(kf1.landmark_ids[0]) == lm.id

        # Without rebuilding the cache manually, descriptor bank must now return BOTH descriptors
        descs_after, ids_after = slam._landmark_descriptor_bank([lm])
        assert len(ids_after) == 2
        assert list(ids_after) == [lm.id, lm.id]
        np.testing.assert_allclose(descs_after[0], desc_a, atol=1e-6)
        np.testing.assert_allclose(descs_after[1], desc_b, atol=1e-6)
    finally:
        slam.close(finish=False)


def test_verified_baseline_uses_full_resolution_stereo():
    """[P2] verified_baseline must remain on full-resolution SGBM even for >=640x240 images."""
    w, h = 640, 240
    K, stereo = _make_camera_and_stereo(w=w, h=h)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(h, w + 24), dtype=np.uint8)
    base = cv.GaussianBlur(base, (3, 3), 0.7)
    left = base[:, 16 : 16 + w].copy()
    right = base[:, 0:w].copy()

    expected_disp = stereo.stereo.compute(left, right).astype(np.float32) / 16.0

    cfg_base = MappingConfig.from_preset("verified_baseline", stereo=True)
    assert getattr(cfg_base, "use_half_res_stereo", False) is False
    slam_base = SharedSlam(K, stereo=stereo, config=cfg_base)
    try:
        slam_base._prepare_frame_images(left, right)
        slam_base._extract(left, right)
        np.testing.assert_array_equal(slam_base.current_disparity, expected_disp)

        with FrontendExtractionPipeline(slam_base, buffer_size=1) as pipe:
            assert pipe._half_sgbm is None
            pipe.submit(0, left, right)
            ext = pipe.get(0)
            np.testing.assert_array_equal(ext.disparity, expected_disp)
    finally:
        slam_base.close(finish=False)

    cfg_imp = MappingConfig.from_preset("improved_stereo", stereo=True)
    assert getattr(cfg_imp, "use_half_res_stereo", False) is True


def test_loop_harness_forwards_disable_native_env(tmp_path, monkeypatch):
    """[P2] VISUAL_SLAM_DISABLE_NATIVE must be forwarded into the sanitized worker env."""
    import subprocess

    assert "VISUAL_SLAM_DISABLE_NATIVE" in loop_harness._SANITIZED_ENV_ALLOWLIST

    seq_dir = tmp_path / "data" / "sequences" / "00"
    (seq_dir / "image_0").mkdir(parents=True)
    (seq_dir / "image_1").mkdir(parents=True)
    (seq_dir / "calib.txt").write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    img = np.zeros((16, 16), dtype=np.uint8)
    for i in range(2):
        cv.imwrite(str(seq_dir / "image_0" / f"{i:06d}.png"), img)
        cv.imwrite(str(seq_dir / "image_1" / f"{i:06d}.png"), img)
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    monkeypatch.setenv("VISUAL_SLAM_DISABLE_NATIVE", "1")
    _cmd, worker_env, _seq_dir = loop_harness.build_airgapped_estimator_invocation(
        sequence="00",
        data_root=tmp_path / "data",
        poses_root=poses_dir,
        seq_out_dir=tmp_path / "out",
        preset="improved_stereo",
        effective_overrides={},
        max_frames=2,
        seed=0,
        opencv_threads=1,
        openblas_threads=1,
    )
    assert worker_env.get("VISUAL_SLAM_DISABLE_NATIVE") == "1"

    # Verify across a real subprocess boundary that native_accel honors the forwarded flag
    proc = subprocess.run(
        [sys.executable, "-c", "import os, native_accel; print(os.environ.get('VISUAL_SLAM_DISABLE_NATIVE'), native_accel.native_enabled())"],
        env=worker_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "1 False"


def test_far_triangulation_applies_geometric_window_before_descriptor_ranking():
    """[P2] Out-of-window lookalike descriptors must not suppress valid local far matches."""
    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    T2 = np.eye(4, dtype=np.float64)
    T2[0, 3] = 2.0  # 2m lateral motion -> clear parallax at 40m depth

    # 8 true far landmarks at depth 40m
    world_pts = np.array(
        [
            [-12.0 + 3.5 * i, -4.0 + 1.1 * (i % 4), 40.0 + 2.0 * i]
            for i in range(8)
        ],
        dtype=np.float64,
    )
    proj1 = (K @ world_pts.T).T
    px1 = (proj1[:, :2] / proj1[:, 2:3]).astype(np.float32)

    pts_c2 = (np.linalg.inv(T2)[:3, :3] @ world_pts.T + np.linalg.inv(T2)[:3, 3:4]).T
    proj2 = (K @ pts_c2.T).T
    px2_true = (proj2[:, :2] / proj2[:, 2:3]).astype(np.float32)

    rng = np.random.default_rng(123)
    base_descs = rng.uniform(0.0, 1.0, size=(8, 128)).astype(np.float32)
    base_descs /= np.linalg.norm(base_descs, axis=1, keepdims=True)
    # Add tiny noise to the true in-window matches
    true_descs2 = base_descs + rng.normal(0.0, 0.01, size=(8, 128)).astype(np.float32)
    true_descs2 /= np.linalg.norm(true_descs2, axis=1, keepdims=True)

    # Place exact-copy lookalikes 250 px away (outside the ~40 px geometric search window).
    # A global descriptor matcher would match these lookalikes (distance=0) and then reject
    # them at the window gate, triangulating 0 landmarks instead of all 8.
    px2_distractors = px2_true + np.array([250.0, 0.0], dtype=np.float32)
    px2_all = np.vstack([px2_true, px2_distractors])
    descs2_all = np.vstack([true_descs2, base_descs.copy()])

    kf1 = MappingKeyframe(
        id=0,
        frame=0,
        pose=T1,
        pixels=px1,
        descriptors=base_descs,
        landmark_ids=np.full(len(px1), -1, dtype=np.int64),
        image_size=(1240, 376),
    )
    kf2 = MappingKeyframe(
        id=1,
        frame=5,
        pose=T2,
        pixels=px2_all,
        descriptors=descs2_all,
        landmark_ids=np.full(len(px2_all), -1, dtype=np.int64),
        image_size=(1240, 376),
    )

    tri = triangulate_far_stereo_landmarks(kf1, kf2, K, baseline=0.54)
    assert len(tri) == 8
    matched_curr = {item.curr_feat_idx for item in tri}
    assert matched_curr == set(range(8))
