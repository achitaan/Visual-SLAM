"""Unit tests for vectorized analytic sparse BA Jacobians and stereo pose refinement (F7)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analytic_ba import (
    build_analytic_ba_jacobian,
    build_local_bundle_jacobian,
    refine_stereo_pose_analytic,
)
from local_bundle import local_bundle_adjustment
from mapping_geometry import right_pixel
from slam_state import MapState, MappingKeyframe, Observation


def _random_se3(rng: np.random.Generator, rot_std: float = 0.15, trans_std: float = 2.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, rot_std, 3)).as_matrix()
    T[:3, 3] = rng.normal(0.0, trans_std, 3)
    return T


def test_build_analytic_ba_jacobian_matches_central_finite_diff():
    rng = np.random.default_rng(42)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    baseline = 0.537
    disparity_offset = 0.25

    n_kf = 4
    n_pt = 12
    poses_cw = np.array([_random_se3(rng, 0.08, 1.5) for _ in range(n_kf)])
    points_w = np.column_stack((
        rng.uniform(-6.0, 6.0, n_pt),
        rng.uniform(-2.0, 2.0, n_pt),
        rng.uniform(10.0, 45.0, n_pt),
    ))
    obs_kf = []
    obs_pt = []
    mask = []
    for k in range(n_kf):
        for p in range(n_pt):
            if rng.random() < 0.7:
                obs_kf.append(k)
                obs_pt.append(p)
                # Mix full stereo [T, T, T] and bearing-only [T, T, F]
                mask.append([True, True, bool(rng.random() < 0.65)])
    obs_kf = np.asarray(obs_kf, dtype=np.intp)
    obs_pt = np.asarray(obs_pt, dtype=np.intp)
    mask = np.asarray(mask, dtype=bool)
    weights = rng.uniform(0.3, 1.5, size=int(mask.sum()))

    def eval_res(poses_arr, pts_arr):
        R = poses_arr[obs_kf, :3, :3]
        t = poses_arr[obs_kf, :3, 3]
        pc = np.einsum("nij,nj->ni", R, pts_arr[obs_pt]) + t
        h = pc @ K.T
        uv = h[:, :2] / h[:, 2:]
        uR = uv[:, 0] - K[0, 0] * baseline / pc[:, 2] - disparity_offset
        vals = np.column_stack((uv[:, 0], uv[:, 1], uR))
        return vals[mask] * weights

    J_ana = build_analytic_ba_jacobian(
        poses_cw, points_w, obs_kf, obs_pt, K, baseline, disparity_offset, mask, huber_weights=weights
    ).toarray()

    eps = 1e-6
    n_rows, n_cols = J_ana.shape
    J_num = np.zeros((n_rows, n_cols), dtype=np.float64)

    for k in range(n_kf):
        for d in range(6):
            delta = np.zeros(6)
            delta[d] = eps
            p_plus = poses_cw.copy()
            p_minus = poses_cw.copy()
            dR_p = Rotation.from_rotvec(delta[:3]).as_matrix()
            dR_m = Rotation.from_rotvec(-delta[:3]).as_matrix()
            p_plus[k, :3, :3] = dR_p @ poses_cw[k, :3, :3]
            p_plus[k, :3, 3] = dR_p @ poses_cw[k, :3, 3] + delta[3:]
            p_minus[k, :3, :3] = dR_m @ poses_cw[k, :3, :3]
            p_minus[k, :3, 3] = dR_m @ poses_cw[k, :3, 3] - delta[3:]
            J_num[:, 6 * k + d] = (eval_res(p_plus, points_w) - eval_res(p_minus, points_w)) / (2.0 * eps)

    for p in range(n_pt):
        for d in range(3):
            pts_p = points_w.copy()
            pts_m = points_w.copy()
            pts_p[p, d] += eps
            pts_m[p, d] -= eps
            J_num[:, 6 * n_kf + 3 * p + d] = (eval_res(poses_cw, pts_p) - eval_res(poses_cw, pts_m)) / (2.0 * eps)

    max_err = float(np.max(np.abs(J_ana - J_num)))
    assert max_err < 1e-6, f"Analytic vs finite-diff max_abs_err={max_err}"


def test_build_local_bundle_jacobian_matches_finite_diff():
    rng = np.random.default_rng(7)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    baseline = 0.54
    disparity_offset = 0.0

    # 3 cameras (camera 0 fixed, 1 & 2 free), 8 optimized points, 3 held-out fixed points
    base_cams = np.array([np.eye(4), _random_se3(rng, 0.1, 1.0), _random_se3(rng, 0.1, 2.0)])
    free_cam_idx = np.array([1, 2], dtype=np.intp)
    free_offsets = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.intp)
    point_offset = 12
    pts_init = np.column_stack((
        rng.uniform(-4.0, 4.0, 8),
        rng.uniform(-1.5, 1.5, 8),
        rng.uniform(12.0, 35.0, 8),
    ))
    x0 = np.concatenate((
        Rotation.from_matrix(base_cams[1, :3, :3]).as_rotvec(), base_cams[1, :3, 3],
        Rotation.from_matrix(base_cams[2, :3, :3]).as_rotvec(), base_cams[2, :3, 3],
        pts_init.ravel(),
    ))
    point_limit = len(x0)

    rec_cams = np.array([0, 1, 2] * 8, dtype=np.intp)
    rec_pts = np.repeat(np.arange(8, dtype=np.intp), 3)
    dim_mask = np.ones((len(rec_cams), 3), dtype=bool)
    dim_mask[::4, 2] = False  # mix bearing-only rows
    meas_pix = rng.uniform(100.0, 500.0, size=(len(rec_cams), 2))
    meas_right = meas_pix[:, 0] - 15.0

    held_cams = np.array([1, 2, 2], dtype=np.intp)
    held_pts = np.array([[1.0, 0.5, 18.0], [-2.0, -0.5, 22.0], [0.5, 0.2, 15.0]])
    held_pix = rng.uniform(100.0, 500.0, size=(3, 2))
    held_right = held_pix[:, 0] - 12.0
    held_mask = np.array([[True, True, True], [True, True, False], [True, True, True]])

    def cams_for(vec):
        cams = base_cams.copy()
        for slot, ci in enumerate(free_cam_idx):
            off = free_offsets[slot, 0]
            cams[ci, :3, :3] = Rotation.from_rotvec(vec[off:off + 3]).as_matrix()
            cams[ci, :3, 3] = vec[off + 3:off + 6]
        return cams

    def obs_res(pts, cams, cidx, pix, ru, msk):
        sel = cams[cidx]
        pc = np.einsum("ni,nij->nj", pts - sel[:, :3, 3], sel[:, :3, :3])
        h = pc @ K.T
        uv = h[:, :2] / np.maximum(h[:, 2:], 1e-9)
        vals = np.zeros((len(cidx), 3))
        vals[:, :2] = np.clip(uv - pix, -1e4, 1e4)
        vals[:, 2] = right_pixel(uv[:, 0], pc[:, 2], K[0, 0], baseline, disparity_offset) - ru
        vals[pc[:, 2] <= 0] = 1e4
        return vals[msk]

    def full_res(vec):
        cams = cams_for(vec)
        pts = vec[point_offset:point_limit].reshape(-1, 3)
        return np.r_[
            obs_res(pts[rec_pts], cams, rec_cams, meas_pix, meas_right, dim_mask),
            obs_res(held_pts, cams, held_cams, held_pix, held_right, held_mask),
        ]

    J_ana = build_local_bundle_jacobian(
        x0, cams_for(x0), free_cam_idx, free_offsets, point_offset, point_limit,
        rec_cams, rec_pts, meas_pix, dim_mask,
        held_cams, held_pts, held_pix, held_mask, K, baseline, metric=True,
    ).toarray()

    eps = 1e-6
    J_num = np.zeros_like(J_ana)
    for col in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[col] += eps
        xm[col] -= eps
        J_num[:, col] = (full_res(xp) - full_res(xm)) / (2.0 * eps)

    max_err = float(np.max(np.abs(J_ana - J_num)))
    assert max_err < 1e-6, f"local_bundle analytic Jacobian max_abs_err={max_err}"


def test_refine_stereo_pose_analytic_recovers_pose_with_outliers_and_bearings():
    rng = np.random.default_rng(123)
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    baseline = 0.54
    T_wc_true = np.eye(4)
    T_wc_true[:3, :3] = Rotation.from_rotvec([0.02, -0.05, 0.01]).as_matrix()
    T_wc_true[:3, 3] = [0.3, -0.1, 4.5]
    T_cw_true = np.linalg.inv(T_wc_true)

    # 30 close stereo points + 15 far bearing-only points
    n_close, n_far = 30, 15
    pts_close = np.column_stack((rng.uniform(-6, 6, n_close), rng.uniform(-2, 2, n_close), rng.uniform(10, 25, n_close)))
    pts_far = np.column_stack((rng.uniform(-30, 30, n_far), rng.uniform(-10, 10, n_far), rng.uniform(80, 160, n_far)))
    pts_w = np.vstack((pts_close, pts_far))
    pts_c = pts_w @ T_cw_true[:3, :3].T + T_cw_true[:3, 3]
    h = pts_c @ K.T
    uv = h[:, :2] / h[:, 2:]
    uR = uv[:, 0] - K[0, 0] * baseline / pts_c[:, 2]

    obs = np.column_stack((uv[:, 0], uv[:, 1], uR))
    obs += rng.normal(0.0, 0.15, size=obs.shape)
    # Inject 3 gross outliers
    obs[0, :2] += 40.0
    obs[5, 2] -= 30.0
    obs[10, :2] -= 25.0

    comp_mask = np.ones((n_close + n_far, 3), dtype=bool)
    comp_mask[n_close:, 2] = False  # far points are bearing-only!
    weights = np.ones_like(obs)
    weights[:n_close, 2] = 0.8

    # Perturb initial pose significantly
    T_wc_init = T_wc_true.copy()
    T_wc_init[:3, :3] = Rotation.from_rotvec([0.04, -0.03, 0.02]).as_matrix() @ T_wc_true[:3, :3]
    T_wc_init[:3, 3] = T_wc_true[:3, 3] + np.array([0.25, -0.15, 0.40])

    res = refine_stereo_pose_analytic(
        T_wc_init, pts_w, obs, weights, comp_mask, K, baseline, disparity_offset=0.0, f_scale=1.5, max_iter=20
    )
    assert res.converged
    assert res.final_cost < res.initial_cost * 0.10
    trans_err = float(np.linalg.norm(res.T_wc[:3, 3] - T_wc_true[:3, 3]))
    rot_err = float(np.linalg.norm(Rotation.from_matrix(res.T_wc[:3, :3].T @ T_wc_true[:3, :3]).as_rotvec()))
    assert trans_err < 0.05, f"Translation error too large: {trans_err}"
    assert rot_err < 0.005, f"Rotation error too large: {rot_err}"
    assert res.hessian_6x6.shape == (6, 6)
    assert not res.inlier_mask[0] and not res.inlier_mask[5] and not res.inlier_mask[10]


def test_local_bundle_analytic_parity_with_numerical():
    K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])
    baseline = 0.54

    def make_state():
        rng = np.random.default_rng(99)
        st = MapState(metric=True)
        true_poses = []
        for k in range(4):
            pose = np.eye(4)
            pose[:3, 3] = [0.1 * k, 0.0, 1.5 * k]
            true_poses.append(pose)
        # Perturb keyframes 1..3 slightly
        noisy_poses = [true_poses[0].copy()]
        for k in range(1, 4):
            p = true_poses[k].copy()
            p[:3, 3] += rng.normal(0.0, 0.04, 3)
            p[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.005, 3)).as_matrix() @ p[:3, :3]
            noisy_poses.append(p)
        desc = np.zeros(128, dtype=np.float32)
        for k in range(4):
            st.keyframes[k] = MappingKeyframe(
                k, k * 10, noisy_poses[k], np.empty((0, 2), np.float32),
                np.empty((0, 128), np.float32), np.empty(0, int),
            )
            st.record(noisy_poses[k], "tracking", anchor=k)
        for _ in range(30):
            pw = np.array([rng.uniform(-4, 4), rng.uniform(-1.5, 1.5), rng.uniform(10, 30)])
            obs = {}
            for k in range(4):
                T_cw = np.linalg.inv(true_poses[k])
                pc = T_cw[:3, :3] @ pw + T_cw[:3, 3]
                if pc[2] <= 1.0:
                    continue
                h = K @ pc
                uv = (h[:2] / h[2]).astype(np.float32)
                uR = float(uv[0] - K[0, 0] * baseline / pc[2])
                obs[k] = Observation(uv, uR)
            if len(obs) >= 2:
                st.add_landmark(pw + rng.normal(0.0, 0.05, 3), desc, anchor=0, observations=obs)
        return st

    st_num = make_state()
    st_ana = make_state()

    rep_num = local_bundle_adjustment(
        st_num, K, baseline, solver_accuracy="precise", use_analytic_jacobian=False, use_map_index=False
    )
    rep_ana = local_bundle_adjustment(
        st_ana, K, baseline, solver_accuracy="precise", use_analytic_jacobian=True, use_map_index=True
    )

    assert rep_num["applied"] is True
    assert rep_ana["applied"] is True
    for k in range(4):
        assert np.allclose(st_ana.keyframes[k].pose, st_num.keyframes[k].pose, atol=1e-4)
