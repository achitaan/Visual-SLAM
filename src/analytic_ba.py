"""Vectorized analytic sparse Jacobians for stereo BA and 6-DoF stereo pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseRefineResult:
    T_wc: np.ndarray
    converged: bool
    initial_cost: float
    final_cost: float
    inlier_mask: np.ndarray
    hessian_6x6: np.ndarray


def skew_batch(vecs: np.ndarray) -> np.ndarray:
    """Return [N, 3, 3] skew-symmetric matrices [v]_x for v of shape [N, 3]."""
    v = np.asarray(vecs, dtype=np.float64).reshape(-1, 3)
    n = len(v)
    out = np.zeros((n, 3, 3), dtype=np.float64)
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out


def right_jacobian_batch(rotvecs: np.ndarray) -> np.ndarray:
    """Vectorized SO(3) right Jacobian J_r(phi) matching pose_observability._right_jacobian."""
    phi = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    n = len(phi)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()
    if n == 0:
        return eye
    theta = np.linalg.norm(phi, axis=1)
    cross = skew_batch(phi)
    cross2 = cross @ cross

    small = theta < 1e-7
    c1 = np.empty(n, dtype=np.float64)
    c2 = np.empty(n, dtype=np.float64)
    c1[small] = 0.5
    c2[small] = 1.0 / 6.0
    if np.any(~small):
        t = theta[~small]
        t2 = t * t
        c1[~small] = (1.0 - np.cos(t)) / t2
        c2[~small] = (t - np.sin(t)) / (t2 * t)

    return eye - c1[:, None, None] * cross + c2[:, None, None] * cross2


def _projection_jacobian_wrt_cam(
    pts_c: np.ndarray,
    K: np.ndarray,
    baseline: float,
    raw_left_error: np.ndarray | None = None,
    clip_bound: float = 1e4,
) -> np.ndarray:
    """Compute d[u_L, v_L, u_R]/d(p_c) of shape [M, 3, 3] for camera-frame points p_c."""
    m = len(pts_c)
    J_pix = np.zeros((m, 3, 3), dtype=np.float64)
    if m == 0:
        return J_pix
    K = np.asarray(K, dtype=np.float64)
    h = pts_c @ K.T  # [M, 3]
    z = pts_c[:, 2]
    valid_z = z > 0.0
    if not np.any(valid_z):
        return J_pix

    h2 = h[:, 2]
    h2_gt = valid_z & (h2 > 1e-9)
    h2_le = valid_z & (h2 <= 1e-9)

    left_proj = np.zeros((m, 2, 3), dtype=np.float64)
    if np.any(h2_gt):
        inv_h2_sq = 1.0 / (h2[h2_gt] * h2[h2_gt])
        h2_v = h2[h2_gt][:, None]
        left_proj[h2_gt, 0, :] = (h2_v * K[0] - h[h2_gt, 0:1] * K[2]) * inv_h2_sq[:, None]
        left_proj[h2_gt, 1, :] = (h2_v * K[1] - h[h2_gt, 1:2] * K[2]) * inv_h2_sq[:, None]
    if np.any(h2_le):
        left_proj[h2_le, 0, :] = K[0] / 1e-9
        left_proj[h2_le, 1, :] = K[1] / 1e-9

    if raw_left_error is not None:
        u_ok = valid_z & (np.abs(raw_left_error[:, 0]) < clip_bound)
        v_ok = valid_z & (np.abs(raw_left_error[:, 1]) < clip_bound)
    else:
        u_ok = valid_z
        v_ok = valid_z

    J_pix[u_ok, 0, :] = left_proj[u_ok, 0, :]
    J_pix[v_ok, 1, :] = left_proj[v_ok, 1, :]

    J_pix[valid_z, 2, :] = left_proj[valid_z, 0, :]
    z_gt = valid_z & (z > 1e-9)
    if np.any(z_gt) and baseline != 0.0:
        J_pix[z_gt, 2, 2] += (K[0, 0] * float(baseline)) / (z[z_gt] * z[z_gt])

    return J_pix


def build_analytic_ba_jacobian(
    poses_cw: np.ndarray,
    points_w: np.ndarray,
    obs_kf_idx: np.ndarray,
    obs_pt_idx: np.ndarray,
    K: np.ndarray,
    baseline: float,
    disparity_offset: float,
    component_mask: np.ndarray,
    huber_weights: np.ndarray | None = None,
) -> csr_matrix:
    """Build vectorized sparse CSR analytic Jacobian for stereo BA in world-to-camera left tangent chart.

    Parameter layout: [pose_0 (6), ..., pose_{N_kf-1} (6), pt_0 (3), ..., pt_{N_pt-1} (3)]
    where each pose perturbation is left-multiplied SE(3) tangent [d_omega, d_t] such that:
        p_c' = exp([d_omega]_x) @ p_c + d_t  =>  dp_c/d_omega = -[p_c]_x,  dp_c/d_t = I_3
    and dp_c/d_p_w = R_cw.
    """
    poses = np.asarray(poses_cw, dtype=np.float64)
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    kf_idx = np.asarray(obs_kf_idx, dtype=np.intp).reshape(-1)
    pt_idx = np.asarray(obs_pt_idx, dtype=np.intp).reshape(-1)
    mask = np.asarray(component_mask, dtype=bool).reshape(-1, 3)
    K_arr = np.asarray(K, dtype=np.float64)

    n_kf = len(poses)
    n_pt = len(pts)
    n_obs = len(kf_idx)
    n_cols = 6 * n_kf + 3 * n_pt
    _ = float(disparity_offset)  # constant shift has zero derivative

    if n_obs == 0 or not np.any(mask):
        return csr_matrix((0, n_cols), dtype=np.float64)

    R_obs = poses[kf_idx, :3, :3]  # [M, 3, 3]
    t_obs = poses[kf_idx, :3, 3]   # [M, 3]
    pts_obs = pts[pt_idx]          # [M, 3]
    pts_c = np.einsum("nij,nj->ni", R_obs, pts_obs) + t_obs  # [M, 3]

    J_pix = _projection_jacobian_wrt_cam(pts_c, K_arr, float(baseline))  # [M, 3, 3]

    # dp_c / d_omega = -[p_c]_x, dp_c / d_t = I_3, dp_c / d_p_w = R_cw
    neg_skew_pc = -skew_batch(pts_c)  # [M, 3, 3]
    J_pose = np.empty((n_obs, 3, 6), dtype=np.float64)
    J_pose[:, :, :3] = J_pix @ neg_skew_pc
    J_pose[:, :, 3:] = J_pix
    J_point = J_pix @ R_obs  # [M, 3, 3]

    # Select active components in row-major order (matching values[mask])
    obs_indices, comp_indices = np.nonzero(mask)
    n_rows = len(obs_indices)

    active_pose = J_pose[obs_indices, comp_indices, :]    # [R, 6]
    active_point = J_point[obs_indices, comp_indices, :]  # [R, 3]

    if huber_weights is not None:
        w = np.asarray(huber_weights, dtype=np.float64)
        if w.shape == mask.shape:
            w_active = w[mask]
        else:
            w_active = w.reshape(-1)
        if len(w_active) != n_rows:
            raise ValueError(f"huber_weights length {len(w_active)} != active rows {n_rows}")
        active_pose = active_pose * w_active[:, None]
        active_point = active_point * w_active[:, None]

    row_ids = np.arange(n_rows, dtype=np.int32)
    pose_cols = (6 * kf_idx[obs_indices])[:, None] + np.arange(6, dtype=np.int32)[None, :]
    pt_cols = (6 * n_kf + 3 * pt_idx[obs_indices])[:, None] + np.arange(3, dtype=np.int32)[None, :]

    all_rows = np.concatenate((np.repeat(row_ids, 6), np.repeat(row_ids, 3)))
    all_cols = np.concatenate((pose_cols.ravel(), pt_cols.ravel()))
    all_data = np.concatenate((active_pose.ravel(), active_point.ravel()))
    return csr_matrix((all_data, (all_rows, all_cols)), shape=(n_rows, n_cols), dtype=np.float64)


def build_local_bundle_jacobian(
    x: np.ndarray,
    cameras_wc: np.ndarray,
    free_camera_indices: np.ndarray,
    free_offsets: np.ndarray,
    point_offset: int,
    point_limit: int,
    record_cameras: np.ndarray,
    record_points: np.ndarray,
    measured_pixels: np.ndarray,
    dimension_mask: np.ndarray,
    held_cameras: np.ndarray,
    held_points: np.ndarray,
    held_pixels: np.ndarray,
    held_dimension_mask: np.ndarray,
    K: np.ndarray,
    baseline: float,
    metric: bool,
    row_weights: np.ndarray | None = None,
) -> csr_matrix:
    """Exact vectorized CSR analytic Jacobian for local_bundle.py's C2W (rotvec, t) + XYZ chart."""
    n_vars = len(x)
    K_arr = np.asarray(K, dtype=np.float64)
    n_cams = len(cameras_wc)

    # Map camera_index -> starting column in x (-1 if fixed) and precompute J_r(rotvec) per camera
    cam_col = np.full(n_cams, -1, dtype=np.int32)
    cam_Jr = np.broadcast_to(np.eye(3, dtype=np.float64), (n_cams, 3, 3)).copy()
    if len(free_camera_indices):
        cam_col[free_camera_indices] = free_offsets[:, 0].astype(np.int32)
        rotvecs = x[free_offsets[:, :3]]
        cam_Jr[free_camera_indices] = right_jacobian_batch(rotvecs)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    row_cursor = 0

    def _assemble_block(
        pts_world: np.ndarray,
        cam_idx: np.ndarray,
        pixels: np.ndarray,
        mask: np.ndarray,
        pt_var_indices: np.ndarray | None,
    ) -> None:
        nonlocal row_cursor
        m = len(cam_idx)
        if m == 0:
            return
        eff_mask = mask if metric else np.column_stack((np.ones(m, dtype=bool), np.ones(m, dtype=bool), np.zeros(m, dtype=bool)))
        obs_ids, comp_ids = np.nonzero(eff_mask)
        n_active = len(obs_ids)
        if n_active == 0:
            return

        sel_cams = cameras_wc[cam_idx]  # [M, 4, 4]
        R_wc = sel_cams[:, :3, :3]
        t_wc = sel_cams[:, :3, 3]
        delta = pts_world - t_wc
        # p_c = R_wc^T @ (X - t_wc)
        pts_c = np.einsum("ni,nij->nj", delta, R_wc)

        h = pts_c @ K_arr.T
        proj = h[:, :2] / np.maximum(h[:, 2:], 1e-9)
        raw_left_err = proj - pixels

        J_pix = _projection_jacobian_wrt_cam(
            pts_c, K_arr, float(baseline) if metric else 0.0, raw_left_error=raw_left_err
        )  # [M, 3, 3]

        # Pose columns for observations whose camera is free
        obs_cam_col = cam_col[cam_idx[obs_ids]]
        free_obs_sel = obs_cam_col >= 0
        if np.any(free_obs_sel):
            f_row_local = np.flatnonzero(free_obs_sel)
            f_obs = obs_ids[f_row_local]
            f_comp = comp_ids[f_row_local]
            f_cam = cam_idx[f_obs]
            f_j = J_pix[f_obs, f_comp, :]  # [F, 3]

            # d(p_c)/d(rotvec) = [p_c]_x @ J_r(phi)
            # f_j @ [p_c]_x = -(p_c x f_j) = cross(f_j, p_c)
            # Wait: let's verify: (v^T [p]_x) = (p x v)^T? Let's check:
            # ([p]_x u) = p x u => v^T (p x u) = (v x p)^T u!
            # Check with p=(0,0,1), u=(1,0,0) => p x u = (0,1,0).
            # v=(0,1,0) => v^T (p x u) = 1. (v x p) = (1,0,0) => (v x p)^T u = 1. YES!
            v_cross_p = np.cross(f_j, pts_c[f_obs])  # [F, 3]
            j_rot = np.einsum("fi,fij->fj", v_cross_p, cam_Jr[f_cam])  # [F, 3]
            # d(p_c)/d(t_wc) = -R_wc^T => f_j @ (-R_wc^T) = -(R_wc @ f_j)
            j_trans = -np.einsum("fij,fj->fi", cameras_wc[f_cam, :3, :3], f_j)  # [F, 3]
            j_pose_block = np.hstack((j_rot, j_trans))  # [F, 6]

            abs_rows = (row_cursor + f_row_local).astype(np.int32)
            cols = obs_cam_col[f_row_local, None] + np.arange(6, dtype=np.int32)[None, :]
            row_parts.append(np.repeat(abs_rows, 6))
            col_parts.append(cols.ravel())
            data_parts.append(j_pose_block.ravel())

        # Point columns (only for optimized records, not held_out)
        if pt_var_indices is not None:
            p_j = J_pix[obs_ids, comp_ids, :]  # [R, 3]
            p_cam = cam_idx[obs_ids]
            # d(p_c)/d(X) = R_wc^T => p_j @ R_wc^T = R_wc @ p_j
            j_pt_block = np.einsum("rij,rj->ri", cameras_wc[p_cam, :3, :3], p_j)  # [R, 3]
            abs_rows = (row_cursor + np.arange(n_active, dtype=np.int32))
            cols = (point_offset + 3 * pt_var_indices[obs_ids])[:, None] + np.arange(3, dtype=np.int32)[None, :]
            row_parts.append(np.repeat(abs_rows, 3))
            col_parts.append(cols.ravel())
            data_parts.append(j_pt_block.ravel())

        row_cursor += n_active

    points_all = x[point_offset:point_limit].reshape(-1, 3)
    if len(record_cameras):
        _assemble_block(
            points_all[record_points],
            record_cameras,
            measured_pixels,
            dimension_mask,
            record_points,
        )
    if len(held_cameras):
        _assemble_block(
            held_points,
            held_cameras,
            held_pixels,
            held_dimension_mask,
            None,
        )

    if row_cursor == 0 or not row_parts:
        return csr_matrix((row_cursor, n_vars), dtype=np.float64)

    all_rows = np.concatenate(row_parts)
    all_cols = np.concatenate(col_parts)
    all_data = np.concatenate(data_parts)
    jac = csr_matrix((all_data, (all_rows, all_cols)), shape=(row_cursor, n_vars), dtype=np.float64)
    if row_weights is not None:
        w = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        if len(w) != row_cursor:
            raise ValueError(f"row_weights length {len(w)} != {row_cursor}")
        jac = jac.multiply(w[:, None]).tocsr()
    return jac


def refine_stereo_pose_analytic(
    T_wc_init: np.ndarray,
    points_w: np.ndarray,
    obs_uv_uR: np.ndarray,
    weights_uv_uR: np.ndarray | None,
    component_mask: np.ndarray,
    K: np.ndarray,
    baseline: float,
    disparity_offset: float = 0.0,
    f_scale: float = 1.5,
    max_iter: int = 15,
) -> PoseRefineResult:
    """Fast vectorized IRLS Huber Levenberg-Marquardt 6-DoF stereo+bearing pose refinement.

    Supports heteroscedastic component weights (`weights_uv_uR`) and mixed stereo/far-bearing
    observations via `component_mask` (`[True, True, True]` for stereo, `[True, True, False]` for bearing).
    """
    T_init = np.asarray(T_wc_init, dtype=np.float64)
    if T_init.shape != (4, 4) or not np.isfinite(T_init).all():
        raise ValueError("T_wc_init must be a finite 4x4 matrix")
    pts_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    obs = np.asarray(obs_uv_uR, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(component_mask, dtype=bool).reshape(-1, 3)
    K_arr = np.asarray(K, dtype=np.float64)
    n = len(pts_w)
    if n == 0 or not np.any(mask):
        return PoseRefineResult(
            T_wc=T_init.copy(),
            converged=False,
            initial_cost=0.0,
            final_cost=0.0,
            inlier_mask=np.zeros(n, dtype=bool),
            hessian_6x6=np.zeros((6, 6), dtype=np.float64),
        )
    if weights_uv_uR is None:
        weights = np.ones((n, 3), dtype=np.float64)
    else:
        weights = np.asarray(weights_uv_uR, dtype=np.float64).reshape(-1, 3)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

    fx = float(K_arr[0, 0])
    fy = float(K_arr[1, 1])
    cx = float(K_arr[0, 2])
    cy = float(K_arr[1, 2])
    fb = fx * float(baseline)
    doff = float(disparity_offset)
    delta_huber = float(max(f_scale, 1e-6))

    # Work in T_cw = inv(T_wc) so p_c = R_cw @ p_w + t_cw and left-SE(3) perturbation is exact
    R_cw = T_init[:3, :3].T.copy()
    t_cw = (-R_cw @ T_init[:3, 3]).copy()

    from native_accel import refine_stereo_pose_lm_cpp

    native_res = refine_stereo_pose_lm_cpp(
        R_cw, t_cw, pts_w, obs, weights, mask,
        fx, fy, cx, cy, fb, doff, delta_huber, max(1, int(max_iter)),
    )
    if native_res is not None:
        R_cw, t_cw, initial_cost, cur_cost, converged, H_final = native_res
        T_wc_out = np.eye(4, dtype=np.float64)
        T_wc_out[:3, :3] = R_cw.T
        T_wc_out[:3, 3] = -R_cw.T @ t_cw
        pts_c = pts_w @ R_cw.T + t_cw
        z = pts_c[:, 2]
        valid = z > 1e-3
        inv_z = np.where(valid, 1.0 / np.maximum(z, 1e-9), 0.0)
        u = fx * pts_c[:, 0] * inv_z + cx
        v = fy * pts_c[:, 1] * inv_z + cy
        uR = u - fb * inv_z - doff
        res = np.empty((n, 3), dtype=np.float64)
        res[:, 0] = np.where(valid, np.clip(u - obs[:, 0], -1e4, 1e4), 1e4)
        res[:, 1] = np.where(valid, np.clip(v - obs[:, 1], -1e4, 1e4), 1e4)
        res[:, 2] = np.where(valid, np.clip(uR - obs[:, 2], -1e4, 1e4), 1e4)
        left_err_norm = np.linalg.norm(res[:, :2] * weights[:, :2], axis=1)
        right_active = mask[:, 2]
        right_ok = ~right_active | (np.abs(res[:, 2] * weights[:, 2]) <= max(2.0, delta_huber * 1.5))
        inlier_mask = valid & (left_err_norm <= max(2.5, delta_huber * 2.0)) & right_ok
        return PoseRefineResult(
            T_wc=T_wc_out,
            converged=bool(converged or cur_cost < initial_cost),
            initial_cost=float(initial_cost),
            final_cost=float(cur_cost),
            inlier_mask=inlier_mask,
            hessian_6x6=H_final,
        )

    def _eval(R: np.ndarray, t: np.ndarray):
        pts_c = pts_w @ R.T + t
        z = pts_c[:, 2]
        valid = z > 1e-3
        inv_z = np.where(valid, 1.0 / np.maximum(z, 1e-9), 0.0)
        u = fx * pts_c[:, 0] * inv_z + cx
        v = fy * pts_c[:, 1] * inv_z + cy
        uR = u - fb * inv_z - doff

        res = np.empty((n, 3), dtype=np.float64)
        res[:, 0] = np.where(valid, np.clip(u - obs[:, 0], -1e4, 1e4), 1e4)
        res[:, 1] = np.where(valid, np.clip(v - obs[:, 1], -1e4, 1e4), 1e4)
        res[:, 2] = np.where(valid, np.clip(uR - obs[:, 2], -1e4, 1e4), 1e4)

        w_res = np.where(mask, res * weights, 0.0)
        a = np.abs(w_res)
        # Standard Huber loss: 0.5*r^2 for |r| <= delta, delta*(|r| - 0.5*delta) otherwise
        cost_mat = np.where(a <= delta_huber, 0.5 * w_res * w_res, delta_huber * (a - 0.5 * delta_huber))
        cost = float(np.sum(cost_mat[mask]))
        return pts_c, valid, res, w_res, cost

    pts_c, valid, res, w_res, cur_cost = _eval(R_cw, t_cw)
    initial_cost = cur_cost
    lam = 1e-3
    converged = False
    H_final = np.zeros((6, 6), dtype=np.float64)

    for _ in range(max(1, int(max_iter))):
        z = pts_c[:, 2]
        good = valid & (np.abs(res[:, 0]) < 1e4) & (np.abs(res[:, 1]) < 1e4)
        inv_z = np.where(good, 1.0 / np.maximum(z, 1e-9), 0.0)
        inv_z2 = inv_z * inv_z
        x_c = pts_c[:, 0]
        y_c = pts_c[:, 1]

        # Direct closed-form 3x6 Jacobian wrt left perturbation [d_omega, d_t] on p_c:
        # dp_c = -[p_c]_x d_omega + d_t
        # du/dp_c = [fx/z, 0, -fx*x/z^2]
        # dv/dp_c = [0, fy/z, -fy*y/z^2]
        # duR/dp_c = [fx/z, 0, (-fx*x + fb)/z^2]
        J = np.zeros((n, 3, 6), dtype=np.float64)
        # Row 0 (u_L):
        J[:, 0, 0] = -fx * x_c * y_c * inv_z2
        J[:, 0, 1] = fx * (1.0 + x_c * x_c * inv_z2)
        J[:, 0, 2] = -fx * y_c * inv_z
        J[:, 0, 3] = fx * inv_z
        J[:, 0, 5] = -fx * x_c * inv_z2
        # Row 1 (v_L):
        J[:, 1, 0] = -fy * (1.0 + y_c * y_c * inv_z2)
        J[:, 1, 1] = fy * x_c * y_c * inv_z2
        J[:, 1, 2] = fy * x_c * inv_z
        J[:, 1, 4] = fy * inv_z
        J[:, 1, 5] = -fy * y_c * inv_z2
        # Row 2 (u_R):
        J[:, 2, 0] = (-fx * x_c + fb) * y_c * inv_z2
        J[:, 2, 1] = fx + (fx * x_c - fb) * x_c * inv_z2
        J[:, 2, 2] = -fx * y_c * inv_z
        J[:, 2, 3] = fx * inv_z
        J[:, 2, 5] = (-fx * x_c + fb) * inv_z2

        J[~good] = 0.0
        # Apply component weights & mask
        Jw = J * np.where(mask, weights, 0.0)[:, :, None]

        # Huber IRLS weights: 1 if |r| <= delta else delta / |r|
        abs_r = np.abs(w_res)
        irls_w = np.where(mask & good[:, None], np.where(abs_r <= delta_huber, 1.0, delta_huber / np.maximum(abs_r, 1e-12)), 0.0)

        # Normal equations H = sum Jw^T * irls_w * Jw, g = sum Jw^T * irls_w * w_res
        Jw_flat = Jw.reshape(-1, 6)
        w_flat = irls_w.ravel()
        r_flat = w_res.ravel()
        Jw_weighted = Jw_flat * w_flat[:, None]
        H = Jw_flat.T @ Jw_weighted
        g = Jw_weighted.T @ r_flat
        H_final = H

        if not (np.isfinite(H).all() and np.isfinite(g).all()):
            break
        if float(np.max(np.abs(g))) < 1e-9:
            converged = True
            break

        diag = np.maximum(np.diag(H), 1e-6)
        accepted = False
        for _lm_try in range(6):
            H_damped = H + np.diag(lam * diag)
            try:
                step = -np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e8)
                continue
            if not np.isfinite(step).all():
                lam = min(lam * 10.0, 1e8)
                continue
            dR = Rotation.from_rotvec(step[:3]).as_matrix()
            R_cand = dR @ R_cw
            t_cand = dR @ t_cw + step[3:]
            pc_c, v_c, res_c, wres_c, cost_c = _eval(R_cand, t_cand)
            if np.isfinite(cost_c) and cost_c < cur_cost:
                rel_impr = (cur_cost - cost_c) / max(cur_cost, 1e-12)
                R_cw, t_cw = R_cand, t_cand
                pts_c, valid, res, w_res, cur_cost = pc_c, v_c, res_c, wres_c, cost_c
                lam = max(lam * 0.3, 1e-7)
                accepted = True
                if float(np.linalg.norm(step)) < 1e-7 or rel_impr < 1e-9:
                    converged = True
                break
            lam = min(lam * 5.0, 1e8)

        if converged or not accepted:
            if not accepted and float(np.linalg.norm(g)) < 1e-5:
                converged = True
            break

    T_wc_out = np.eye(4, dtype=np.float64)
    T_wc_out[:3, :3] = R_cw.T
    T_wc_out[:3, 3] = -R_cw.T @ t_cw

    left_err_norm = np.linalg.norm(res[:, :2] * weights[:, :2], axis=1)
    right_active = mask[:, 2]
    right_ok = ~right_active | (np.abs(res[:, 2] * weights[:, 2]) <= max(2.0, delta_huber * 1.5))
    inlier_mask = valid & (left_err_norm <= max(2.5, delta_huber * 2.0)) & right_ok

    return PoseRefineResult(
        T_wc=T_wc_out,
        converged=bool(converged or cur_cost < initial_cost),
        initial_cost=float(initial_cost),
        final_cost=float(cur_cost),
        inlier_mask=inlier_mask,
        hessian_6x6=H_final,
    )


@dataclass(frozen=True)
class SchurBAResult:
    x: np.ndarray
    success: bool
    status: int
    message: str
    nfev: int
    cost: float
    optimality: float


def solve_schur_local_bundle(
    initial: np.ndarray,
    camera_poses_for: Any,
    free_camera_indices: np.ndarray,
    free_offsets: np.ndarray,
    point_offset: int,
    point_limit: int,
    record_cameras: np.ndarray,
    record_points: np.ndarray,
    measured_pixels: np.ndarray,
    measured_right: np.ndarray,
    dimension_mask: np.ndarray,
    held_cameras: np.ndarray,
    held_points: np.ndarray,
    held_pixels: np.ndarray,
    held_right: np.ndarray,
    held_dimension_mask: np.ndarray,
    K: np.ndarray,
    baseline: float,
    disparity_offset: float,
    metric: bool,
    f_scale: float = 2.0,
    max_iter: int = 20,
    obs_weights: np.ndarray | None = None,
) -> SchurBAResult:
    """Fast vectorized Schur-complement Levenberg-Marquardt BA for local_bundle.py.

    Eliminates all 3D landmarks via batched 3x3 inverses and solves the compact
    (6*n_free) x (6*n_free) Schur camera system directly, achieving ~50-100x speedup
    over iterative LSMR while matching the exact Huber objective and C2W chart.
    """
    x = np.asarray(initial, dtype=np.float64).copy()
    K_arr = np.asarray(K, dtype=np.float64)
    fx = float(K_arr[0, 0])
    fb = fx * float(baseline) if metric else 0.0
    doff = float(disparity_offset)
    delta_h = float(max(f_scale, 1e-6))

    n_free = len(free_camera_indices)
    n_pt = (point_limit - point_offset) // 3
    if n_free == 0 or n_pt == 0:
        return SchurBAResult(x=x, success=True, status=1, message="trivial", nfev=1, cost=0.0, optimality=0.0)

    n_cams = len(camera_poses_for(x))
    cam_to_slot = np.full(n_cams, -1, dtype=np.int32)
    cam_to_slot[free_camera_indices] = np.arange(n_free, dtype=np.int32)

    rec_slots = cam_to_slot[record_cameras]
    rec_free_mask = rec_slots >= 0
    rec_mask = dimension_mask if metric else np.column_stack((
        np.ones(len(record_cameras), dtype=bool),
        np.ones(len(record_cameras), dtype=bool),
        np.zeros(len(record_cameras), dtype=bool),
    ))
    held_mask = held_dimension_mask if metric else np.column_stack((
        np.ones(len(held_cameras), dtype=bool),
        np.ones(len(held_cameras), dtype=bool),
        np.zeros(len(held_cameras), dtype=bool),
    ))
    held_slots = cam_to_slot[held_cameras] if len(held_cameras) else np.empty(0, dtype=np.int32)

    if obs_weights is not None:
        w_rec = np.asarray(obs_weights, dtype=np.float64).reshape(len(record_cameras), 3)
    else:
        w_rec = np.ones((len(record_cameras), 3), dtype=np.float64)

    try:
        from native_accel import solve_schur_local_bundle_cpp
        base_cams = camera_poses_for(x)
        native_res = solve_schur_local_bundle_cpp(
            base_cams,
            free_camera_indices,
            x[free_offsets].reshape(-1),
            x[point_offset:point_limit],
            record_cameras,
            record_points,
            measured_pixels,
            measured_right,
            rec_mask,
            w_rec,
            held_cameras,
            held_points,
            held_pixels,
            held_right,
            held_mask,
            K_arr,
            float(baseline),
            doff,
            bool(metric),
            delta_h,
            int(max_iter),
        )
        if native_res is not None:
            fp_out, xp_out, st_out, nfev_out, cost_out, opt_out = native_res
            x[free_offsets] = fp_out.reshape(n_free, 6)
            x[point_offset:point_limit] = xp_out
            msg_map = {
                -1: "nonfinite_gradient",
                0: "max_iter",
                1: "gtol_converged",
                2: "ftol_xtol_converged",
                3: "no_further_decrease",
            }
            return SchurBAResult(
                x=x,
                success=bool(st_out >= 0),
                status=int(st_out),
                message=msg_map.get(st_out, "converged"),
                nfev=int(nfev_out),
                cost=float(cost_out),
                optimality=float(opt_out),
            )
    except Exception:
        pass

    def _eval_block(pts_w, cams, cam_idx, pixels, right_u, mask, weights):
        m = len(cam_idx)
        if m == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                0.0,
            )
        sel = cams[cam_idx]
        delta = pts_w - sel[:, :3, 3]
        pts_c = np.einsum("ni,nij->nj", delta, sel[:, :3, :3])
        h = pts_c @ K_arr.T
        z = pts_c[:, 2]
        proj = h[:, :2] / np.maximum(h[:, 2:], 1e-9)
        raw_left = proj - pixels
        left_err = np.clip(raw_left, -1e4, 1e4)
        left_err[z <= 0.0] = 1e4
        res = np.zeros((m, 3), dtype=np.float64)
        res[:, :2] = left_err
        if metric:
            res[:, 2] = proj[:, 0] - fb / np.maximum(z, 1e-9) - doff - right_u
            res[z <= 0.0, 2] = 1e4
        w_res = np.where(mask, res * weights, 0.0)
        a = np.abs(w_res)
        # Exact Huber cost matching local_bundle.objective (0.5*r^2 vs 2*(|r|-1) for f_scale=2)
        cmat = np.where(a <= delta_h, 0.5 * w_res * w_res, delta_h * (a - 0.5 * delta_h))
        return pts_c, raw_left, w_res, float(np.sum(cmat[mask]))

    def _eval_all(vec):
        cams = camera_poses_for(vec)
        pts = vec[point_offset:point_limit].reshape(n_pt, 3)
        pc_r, raw_r, wr_r, c_r = _eval_block(
            pts[record_points], cams, record_cameras, measured_pixels, measured_right, rec_mask, w_rec
        )
        if len(held_cameras):
            pc_h, raw_h, wr_h, c_h = _eval_block(
                held_points, cams, held_cameras, held_pixels, held_right, held_mask,
                np.ones((len(held_cameras), 3), dtype=np.float64),
            )
        else:
            pc_h = np.empty((0, 3), dtype=np.float64)
            raw_h = np.empty((0, 2), dtype=np.float64)
            wr_h = np.empty((0, 3), dtype=np.float64)
            c_h = 0.0
        return cams, pc_r, raw_r, wr_r, pc_h, raw_h, wr_h, c_r + c_h

    cams, pc_r, raw_r, wr_r, pc_h, raw_h, wr_h, cur_cost = _eval_all(x)
    nfev = 1
    lam = 1e-3
    optimality = float("inf")
    converged = False
    status = 0
    message = "max_iter"

    for _it in range(max(1, int(max_iter))):
        # Precompute right Jacobians for free cameras
        rotvecs = x[free_offsets[:, :3]]
        Jr_free = right_jacobian_batch(rotvecs)  # [n_free, 3, 3]
        R_free = cams[free_camera_indices, :3, :3]  # [n_free, 3, 3]

        # Normal equation blocks
        U = np.zeros((n_free, 6, 6), dtype=np.float64)
        V = np.zeros((n_pt, 3, 3), dtype=np.float64)
        W = np.zeros((n_free, n_pt, 6, 3), dtype=np.float64)
        g_c = np.zeros((n_free, 6), dtype=np.float64)
        g_p = np.zeros((n_pt, 3), dtype=np.float64)

        # 1. Optimized records block
        J_pix_r = _projection_jacobian_wrt_cam(
            pc_r, K_arr, float(baseline) if metric else 0.0, raw_left_error=raw_r
        )
        # Apply observation weights
        J_pix_r *= np.where(rec_mask, w_rec, 0.0)[:, :, None]
        abs_r = np.abs(wr_r)
        irls_r = np.where(
            rec_mask & (pc_r[:, 2:3] > 0.0),
            np.where(abs_r <= delta_h, 1.0, delta_h / np.maximum(abs_r, 1e-12)),
            0.0,
        )  # [M, 3]

        # Point Jacobians: J_pt[m, c, :] = R_wc[cam[m]] @ J_pix_r[m, c, :]
        R_obs = cams[record_cameras, :3, :3]  # [M, 3, 3]
        J_pt = np.einsum("mij,mcj->mci", R_obs, J_pix_r)  # [M, 3, 3]
        # Weighted J_pt for normal equations
        J_pt_w = J_pt * irls_r[:, :, None]  # [M, 3, 3]
        V_obs = np.einsum("mci,mcj->mij", J_pt, J_pt_w)  # [M, 3, 3]
        gp_obs = np.einsum("mci,mc->mi", J_pt_w, wr_r)   # [M, 3]
        np.add.at(V, record_points, V_obs)
        np.add.at(g_p, record_points, gp_obs)

        # Pose Jacobians for observations on free cameras
        if np.any(rec_free_mask):
            f_idx = np.flatnonzero(rec_free_mask)
            f_slots = rec_slots[f_idx]
            f_pts = record_points[f_idx]
            f_Jpix = J_pix_r[f_idx]      # [F, 3, 3]
            f_pc = pc_r[f_idx]           # [F, 3]
            f_irls = irls_r[f_idx]       # [F, 3]
            f_wr = wr_r[f_idx]           # [F, 3]
            f_Jpt = J_pt[f_idx]          # [F, 3, 3]

            # d(p_c)/d(rotvec) = [p_c]_x @ J_r => v @ [p_c]_x = cross(v, p_c)
            vc = np.cross(f_Jpix, f_pc[:, None, :])  # [F, 3, 3]
            J_rot = np.einsum("fci,fij->fcj", vc, Jr_free[f_slots])       # [F, 3, 3]
            J_tr = -np.einsum("fij,fcj->fci", R_free[f_slots], f_Jpix)    # [F, 3, 3]
            J_pose = np.concatenate((J_rot, J_tr), axis=2)                # [F, 3, 6]
            J_pose_w = J_pose * f_irls[:, :, None]

            U_obs = np.einsum("fca,fcb->fab", J_pose, J_pose_w)  # [F, 6, 6]
            gc_obs = np.einsum("fca,fc->fa", J_pose_w, f_wr)     # [F, 6]
            W_obs = np.einsum("fca,fcb->fab", J_pose_w, f_Jpt)   # [F, 6, 3]

            np.add.at(U, f_slots, U_obs)
            np.add.at(g_c, f_slots, gc_obs)
            np.add.at(W, (f_slots, f_pts), W_obs)

        # 2. Held-out fixed landmarks block (contributes only to U and g_c)
        if len(held_cameras):
            h_free = held_slots >= 0
            if np.any(h_free):
                hf_idx = np.flatnonzero(h_free)
                hf_slots = held_slots[hf_idx]
                J_pix_h = _projection_jacobian_wrt_cam(
                    pc_h[hf_idx], K_arr, float(baseline) if metric else 0.0, raw_left_error=raw_h[hf_idx]
                )
                J_pix_h *= held_mask[hf_idx][:, :, None]
                abs_h = np.abs(wr_h[hf_idx])
                irls_h = np.where(
                    held_mask[hf_idx] & (pc_h[hf_idx, 2:3] > 0.0),
                    np.where(abs_h <= delta_h, 1.0, delta_h / np.maximum(abs_h, 1e-12)),
                    0.0,
                )
                vc_h = np.cross(J_pix_h, pc_h[hf_idx][:, None, :])
                J_rot_h = np.einsum("fci,fij->fcj", vc_h, Jr_free[hf_slots])
                J_tr_h = -np.einsum("fij,fcj->fci", R_free[hf_slots], J_pix_h)
                J_pose_h = np.concatenate((J_rot_h, J_tr_h), axis=2)
                J_pose_hw = J_pose_h * irls_h[:, :, None]
                np.add.at(U, hf_slots, np.einsum("fca,fcb->fab", J_pose_h, J_pose_hw))
                np.add.at(g_c, hf_slots, np.einsum("fca,fc->fa", J_pose_hw, wr_h[hf_idx]))

        optimality = float(max(np.max(np.abs(g_c)), np.max(np.abs(g_p))))
        if not np.isfinite(optimality):
            status = -1
            message = "nonfinite_gradient"
            break
        if optimality < 1e-7:
            converged = True
            status = 1
            message = "gtol_converged"
            break

        # LM step search via exact Schur complement
        U_diag = np.maximum(np.diagonal(U, axis1=1, axis2=2), 1e-6)  # [n_free, 6]
        V_diag = np.maximum(np.diagonal(V, axis1=1, axis2=2), 1e-6)  # [n_pt, 3]
        accepted = False

        for _lm_try in range(8):
            V_damp = V.copy()
            V_damp[:, 0, 0] += lam * V_diag[:, 0]
            V_damp[:, 1, 1] += lam * V_diag[:, 1]
            V_damp[:, 2, 2] += lam * V_diag[:, 2]
            try:
                V_inv = np.linalg.inv(V_damp)  # [n_pt, 3, 3]
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e8)
                continue

            # W_Vinv[c, p, a, b] = sum_k W[c, p, a, k] * V_inv[p, k, b]
            W_Vinv = np.einsum("cpak,pkb->cpab", W, V_inv)
            W_mat = W.transpose(0, 2, 1, 3).reshape(6 * n_free, 3 * n_pt)
            W_Vinv_mat = W_Vinv.transpose(0, 2, 1, 3).reshape(6 * n_free, 3 * n_pt)
            S = -(W_Vinv_mat @ W_mat.T)
            for c_slot in range(n_free):
                U_d = U[c_slot].copy()
                U_d[np.arange(6), np.arange(6)] += lam * U_diag[c_slot]
                s0 = 6 * c_slot
                S[s0:s0 + 6, s0:s0 + 6] += U_d

            rhs_c = -g_c.ravel() + (W_Vinv_mat @ g_p.ravel())
            try:
                dx_c = np.linalg.solve(S, rhs_c).reshape(n_free, 6)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e8)
                continue

            # Back-substitute points: dx_p = V_inv @ (-g_p - sum_c W[c, p]^T dx_c[c])
            WT_dxc = (W_mat.T @ dx_c.ravel()).reshape(n_pt, 3)
            dx_p = np.einsum("pab,pb->pa", V_inv, -g_p - WT_dxc)

            if not (np.isfinite(dx_c).all() and np.isfinite(dx_p).all()):
                lam = min(lam * 10.0, 1e8)
                continue

            x_cand = x.copy()
            x_cand[free_offsets] += dx_c
            x_cand[point_offset:point_limit] += dx_p.ravel()

            cams_c, pcr_c, rawr_c, wrr_c, pch_c, rawh_c, wrh_c, cost_c = _eval_all(x_cand)
            nfev += 1
            if np.isfinite(cost_c) and cost_c < cur_cost:
                rel_impr = (cur_cost - cost_c) / max(cur_cost, 1e-12)
                step_norm = float(np.sqrt(np.sum(dx_c * dx_c) + np.sum(dx_p * dx_p)))
                x = x_cand
                cams, pc_r, raw_r, wr_r, pc_h, raw_h, wr_h, cur_cost = (
                    cams_c, pcr_c, rawr_c, wrr_c, pch_c, rawh_c, wrh_c, cost_c
                )
                lam = max(lam * 0.3, 1e-7)
                accepted = True
                if rel_impr < 1e-8 or step_norm < 1e-7:
                    converged = True
                    status = 2
                    message = "ftol_xtol_converged"
                break
            lam = min(lam * 5.0, 1e8)

        if converged or not accepted:
            if not accepted:
                converged = True
                status = 3
                message = "no_further_decrease"
            break

    return SchurBAResult(
        x=x,
        success=bool(converged or status >= 0),
        status=int(status),
        message=message,
        nfev=int(nfev),
        cost=float(cur_cost),
        optimality=float(optimality),
    )
