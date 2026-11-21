"""Parity and fallback tests for optional native C++/AVX-512 kernels."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analytic_ba
import mapping_geometry
import native_accel
import stereo_depth


def _make_textured_image(h: int = 240, w: int = 320, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 225, size=(h, w), dtype=np.uint8)
    img = cv.GaussianBlur(base, (5, 5), 1.2)
    for _ in range(45):
        cx = int(rng.integers(15, w - 15))
        cy = int(rng.integers(15, h - 15))
        r = int(rng.integers(4, 14))
        val = int(rng.integers(10, 245))
        cv.circle(img, (cx, cy), r, val, -1)
    for _ in range(30):
        x1, y1 = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        x2, y2 = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        cv.line(img, (x1, y1), (x2, y2), int(rng.integers(0, 255)), 2)
    return img


def test_native_matcher_exact_parity_vs_bfmatcher():
    if native_accel._get_matcher_lib() is None:
        pytest.skip("libfast_matcher.so not built on this host")
    rng = np.random.default_rng(42)
    # Create realistic integer [0..255] 128-D descriptors with matching pairs + noise
    base = rng.integers(0, 200, size=(300, 128), dtype=np.int32)
    a = base.astype(np.float32)
    b = np.vstack([
        np.clip(base[:180] + rng.integers(-6, 7, size=(180, 128)), 0, 255),
        rng.integers(0, 200, size=(150, 128), dtype=np.int32),
    ]).astype(np.float32)

    for ratio in (0.7, 0.75, 0.8):
        vnni = native_accel.match_sift_mutual_vnni(a, b, ratio)
        bf = mapping_geometry._match_descriptors_bf(a, b, ratio)
        assert vnni is not None
        assert np.array_equal(vnni, bf)

    # Non-integer float descriptors must cleanly fall back (return None)
    non_int = a + np.float32(0.25)
    assert native_accel.match_sift_mutual_vnni(non_int, b, 0.75) is None


def test_native_zncc_u8_and_f32_exact_parity(monkeypatch):
    if native_accel._get_zncc_lib() is None:
        pytest.skip("libfast_zncc.so not built on this host")
    left = _make_textured_image(200, 320, seed=11)
    # Synthetic stereo pair with known horizontal disparity = 14.3 px
    map_x = np.tile(np.arange(320, dtype=np.float32) + 14.3, (200, 1))
    map_y = np.tile(np.arange(200, dtype=np.float32)[:, None], (1, 320))
    right = cv.remap(left, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    rng = np.random.default_rng(123)
    pts = np.column_stack([
        rng.uniform(25.0, 295.0, size=120),
        rng.uniform(12.0, 188.0, size=120),
    ])
    cfg = stereo_depth.StereoSearchConfig()
    bounds = (1.0, 96.0)

    out_u8, cnt_u8 = native_accel.verify_stereo_zncc_avx512(left, right, pts, cfg, bounds)
    out_f32, cnt_f32 = native_accel.verify_stereo_zncc_avx512(
        left.astype(np.float32), right.astype(np.float32), pts, cfg, bounds
    )
    assert cnt_u8 == cnt_f32
    assert np.array_equal(np.isfinite(out_u8), np.isfinite(out_f32))
    valid = np.isfinite(out_u8)
    assert int(valid.sum()) > 50
    assert float(np.max(np.abs(out_u8[valid] - out_f32[valid]))) == 0.0
    # True disparity is ~14.3 px
    med_disp = float(np.median(pts[valid, 0] - out_u8[valid]))
    assert abs(med_disp - 14.3) < 0.25

    # Compare against Python/OpenCV fallback with VISUAL_SLAM_DISABLE_NATIVE=1
    monkeypatch.setenv("VISUAL_SLAM_DISABLE_NATIVE", "1")
    ref_u, ref_cnt = stereo_depth.verify_stereo_depth_candidates(
        left, right, pts, config=cfg, disparity_bounds=bounds
    )
    ref_valid = np.isfinite(ref_u)
    both = valid & ref_valid
    assert int(both.sum()) >= int(0.98 * valid.sum())
    assert float(np.max(np.abs(out_u8[both] - ref_u[both]))) < 1e-3


def test_native_sift_quality_and_format():
    if native_accel._get_sift_lib() is None:
        pytest.skip("libfast_sift.so not built on this host")
    img0 = _make_textured_image(240, 360, seed=19)
    M = np.array([[1.0, 0.0, 4.5], [0.0, 1.0, -2.0]], dtype=np.float32)
    img1 = cv.warpAffine(img0, M, (360, 240), borderMode=cv.BORDER_REPLICATE)

    res0 = native_accel.extract_sift_grid_avx512(img0, target_features=400)
    res1 = native_accel.extract_sift_grid_avx512(img1, target_features=400)
    assert res0 is not None and res1 is not None
    px0, d0, oct0, ang0, resp0 = res0
    px1, d1, _, _, _ = res1
    assert 100 <= len(px0) <= 400
    assert d0.shape == (len(px0), 128) and d0.dtype == np.float32
    assert np.all(np.isfinite(px0)) and np.all(np.isfinite(d0))
    assert np.all((d0 >= 0.0) & (d0 <= 255.0))
    assert np.all(d0 == np.floor(d0))

    pairs = mapping_geometry.match_descriptors(d0, d1, ratio=0.75)
    assert len(pairs) >= 60
    shift = px1[pairs[:, 1]] - px0[pairs[:, 0]]
    err = np.linalg.norm(shift - np.array([4.5, -2.0]), axis=1)
    assert float(np.median(err)) < 0.35


def test_native_ba_exact_parity_vs_python_schur(monkeypatch):
    if native_accel._get_ba_lib() is None:
        pytest.skip("libfast_ba.so not built on this host")
    rng = np.random.default_rng(99)
    K = np.array([[700.0, 0.0, 600.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = 0.54

    # 3 cameras (cam 0 fixed, cams 1 & 2 free) observing 40 3D points
    gt_cams = [np.eye(4, dtype=np.float64)]
    for k in (1, 2):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.array([0.05 * k, 0.0, 1.2 * k])
        gt_cams.append(T)
    gt_pts = np.column_stack([
        rng.uniform(-6.0, 6.0, size=40),
        rng.uniform(-2.0, 2.0, size=40),
        rng.uniform(8.0, 30.0, size=40),
    ])

    rec_cams, rec_pts, rec_px, rec_ru, rec_mask, rec_w = [], [], [], [], [], []
    for ci, T in enumerate(gt_cams):
        R, t = T[:3, :3], T[:3, 3]
        for pi in range(len(gt_pts)):
            pc = R.T @ (gt_pts[pi] - t)
            if pc[2] <= 1.0:
                continue
            u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
            v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
            ur = u - K[0, 0] * baseline / pc[2]
            rec_cams.append(ci)
            rec_pts.append(pi)
            rec_px.append([u + rng.normal(0, 0.25), v + rng.normal(0, 0.25)])
            rec_ru.append(ur + rng.normal(0, 0.25))
            rec_mask.append([True, True, True])
            rec_w.append([1.0, 1.0, 1.0])

    # Perturb initial poses & points
    base_cams = [T.copy() for T in gt_cams]
    base_cams[1][:3, 3] += np.array([0.04, -0.02, 0.05])
    base_cams[2][:3, 3] += np.array([-0.03, 0.03, -0.04])
    init_pts = gt_pts + rng.normal(0, 0.08, size=gt_pts.shape)

    x_free_init = np.zeros(12, dtype=np.float64)
    x_free_init[3:6] = base_cams[1][:3, 3]
    x_free_init[9:12] = base_cams[2][:3, 3]

    kwargs = dict(
        base_cams=np.asarray(base_cams, dtype=np.float64),
        free_cam_idx=np.array([1, 2], dtype=np.int32),
        x_free_poses=x_free_init.copy(),
        x_points=init_pts.ravel().copy(),
        rec_cams=np.asarray(rec_cams, dtype=np.int32),
        rec_pts=np.asarray(rec_pts, dtype=np.int32),
        rec_pixels=np.asarray(rec_px, dtype=np.float64),
        rec_right=np.asarray(rec_ru, dtype=np.float64),
        rec_mask=np.asarray(rec_mask, dtype=np.uint8),
        rec_weights=np.asarray(rec_w, dtype=np.float64),
        held_cams=np.empty(0, dtype=np.int32),
        held_pts=np.empty((0, 3), dtype=np.float64),
        held_pixels=np.empty((0, 2), dtype=np.float64),
        held_right=np.empty(0, dtype=np.float64),
        held_mask=np.empty((0, 3), dtype=np.uint8),
        K=K,
        baseline=baseline,
        disparity_offset=0.0,
        metric=True,
        f_scale=2.0,
        max_iter=25,
    )
    cpp_res = native_accel.solve_schur_local_bundle_cpp(**kwargs)
    assert cpp_res is not None
    fp_c, xp_c, st_c, _, cost_c, _ = cpp_res

    from scipy.spatial.transform import Rotation

    def camera_poses_for(vec):
        out = [T.copy() for T in base_cams]
        for slot, ci in enumerate((1, 2)):
            delta = vec[6 * slot : 6 * slot + 6]
            T = base_cams[ci].copy()
            T[:3, :3] = base_cams[ci][:3, :3] @ Rotation.from_rotvec(delta[:3]).as_matrix()
            T[:3, 3] = delta[3:6]
            out[ci] = T
        return np.asarray(out, dtype=np.float64)

    monkeypatch.setenv("VISUAL_SLAM_DISABLE_NATIVE", "1")
    x0 = np.concatenate([kwargs["x_free_poses"], kwargs["x_points"]])
    py_res = analytic_ba.solve_schur_local_bundle(
        initial=x0,
        camera_poses_for=camera_poses_for,
        free_camera_indices=kwargs["free_cam_idx"],
        free_offsets=np.arange(12, dtype=np.int32).reshape(2, 6),
        point_offset=12,
        point_limit=len(x0),
        record_cameras=kwargs["rec_cams"],
        record_points=kwargs["rec_pts"],
        measured_pixels=kwargs["rec_pixels"],
        measured_right=kwargs["rec_right"],
        dimension_mask=kwargs["rec_mask"].astype(bool),
        held_cameras=kwargs["held_cams"],
        held_points=kwargs["held_pts"],
        held_pixels=kwargs["held_pixels"],
        held_right=kwargs["held_right"],
        held_dimension_mask=kwargs["held_mask"].astype(bool),
        K=K,
        baseline=baseline,
        disparity_offset=0.0,
        metric=True,
        f_scale=2.0,
        max_iter=25,
        obs_weights=kwargs["rec_weights"],
    )
    x_cpp = np.concatenate([fp_c, xp_c])
    assert float(np.max(np.abs(x_cpp - py_res.x))) < 1e-9
    assert abs(cost_c - float(py_res.cost)) < 1e-9


def test_native_lk_parity_vs_opencv():
    if native_accel._get_lk_lib() is None:
        pytest.skip("libfast_lk.so not built on this host")
    img0 = _make_textured_image(240, 360, seed=31)
    M = np.array([[1.0, 0.0, 5.25], [0.0, 1.0, -3.5]], dtype=np.float32)
    img1 = cv.warpAffine(img0, M, (360, 240), borderMode=cv.BORDER_REPLICATE)

    corners = cv.goodFeaturesToTrack(img0, 150, 0.02, 8)
    assert corners is not None and len(corners) >= 50
    pts0 = corners.reshape(-1, 2).astype(np.float32)

    pyr0 = native_accel.build_lk_pyramid_avx512(img0)
    pyr1 = native_accel.build_lk_pyramid_avx512(img1)
    assert pyr0 is not None and pyr1 is not None
    nat_p1, nat_st = native_accel.track_lk_pyramids_avx512(
        pyr0, pyr1, pts0, None, use_initial_flow=False
    )
    cv_p1, cv_st, _ = cv.calcOpticalFlowPyrLK(
        img0, img1, pts0.reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=4,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    both = (nat_st.ravel() == 1) & (cv_st.ravel() == 1)
    assert int(both.sum()) >= int(0.95 * (cv_st.ravel() == 1).sum())
    diffs = np.linalg.norm(nat_p1.reshape(-1, 2)[both] - cv_p1.reshape(-1, 2)[both], axis=1)
    assert float(np.median(diffs)) < 0.05


def test_disable_native_env_clean_fallback(monkeypatch):
    monkeypatch.setenv("VISUAL_SLAM_DISABLE_NATIVE", "1")
    assert not native_accel.native_enabled()
    img = _make_textured_image(64, 64, seed=5)
    d = np.zeros((4, 128), dtype=np.float32)
    assert native_accel.match_sift_mutual_vnni(d, d, 0.75) is None
    assert native_accel.verify_stereo_zncc_avx512(
        img, img, np.array([[20.0, 20.0]]), stereo_depth.StereoSearchConfig(), (1.0, 32.0)
    ) is None
    assert native_accel.extract_sift_grid_avx512(img, target_features=50) is None
    assert native_accel.build_lk_pyramid_avx512(img) is None
