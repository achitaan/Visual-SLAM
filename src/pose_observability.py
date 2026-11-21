"""Machine-precision pose observability for the original image-only BA graph.

The helper eliminates each selected world point with its own small Jacobian
block, then ranks the remaining camera columns. It is structural rank analysis,
not a covariance or conditioning estimate.
"""

import numpy as np
import time


def _skew(value):
    x, y, z = np.asarray(value, dtype=np.float64)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=np.float64)


def _right_jacobian(rotvec):
    phi = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    cross = _skew(phi)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * cross + (cross @ cross) / 6.0
    return (np.eye(3) - ((1.0 - np.cos(theta)) / theta**2) * cross
            + ((theta - np.sin(theta)) / theta**3) * (cross @ cross))


def _observation_jacobians(camera_to_world, camera_rotvec, world_point, pixel,
                           right_u, component_mask, matrix, baseline,
                           disparity_offset):
    """Return active raw-pixel residual/Jacobian rows in the BA chart.

    Pose columns use additive absolute Rodrigues coordinates followed by world
    translation. Point columns are world XYZ. A fixed camera simply ignores
    the returned pose block at the caller.
    """
    pose = np.asarray(camera_to_world, dtype=np.float64)
    point = np.asarray(world_point, dtype=np.float64)
    measured = np.asarray(pixel, dtype=np.float64)
    mask = np.asarray(component_mask, dtype=bool)
    K = np.asarray(matrix, dtype=np.float64)
    right_active = bool(mask[2])
    try:
        right_value = float(right_u) if right_active else 0.0
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid_active_right_measurement")
    if right_active and (np.iscomplexobj(right_u) or not np.isfinite(right_value)):
        raise ValueError("invalid_active_right_measurement")
    if (pose.shape != (4, 4) or point.shape != (3,) or measured.shape != (2,)
            or mask.shape != (3,) or K.shape != (3, 3)
            or not np.isfinite(pose).all() or not np.isfinite(point).all()
            or not np.isfinite(measured).all() or not np.isfinite(K).all()
            ):
        raise ValueError("invalid_observation_geometry")
    R = pose[:3, :3]
    p = R.T @ (point - pose[:3, 3])
    if not np.isfinite(p).all():
        raise ValueError("undefined_camera_depth")
    active = np.flatnonzero(mask)
    if p[2] <= 0.0:
        # The production residual replaces every behind-camera component with
        # the same constant penalty, so this observation contributes no local
        # differential information (including at exactly zero depth).
        return (np.full(len(active), 1e4, dtype=np.float64),
                np.zeros((len(active), 6), dtype=np.float64),
                np.zeros((len(active), 3), dtype=np.float64))
    if float(p[2]) < 1e-15:
        raise ValueError("undefined_camera_depth")

    # Match local_bundle.observation_residual: project through the complete K,
    # clamp the homogeneous denominator, then clip left-coordinate errors.
    h = K @ p
    denom = max(float(h[2]), 1e-9)
    uv = h[:2] / denom
    raw_error = uv - measured
    left_error = np.clip(raw_error, -1e4, 1e4)

    row_values = np.r_[left_error, 0.0]
    row_values[2] = uv[0] - K[0, 0] * float(baseline) / max(float(p[2]), 1e-9) - float(disparity_offset) - right_value

    if float(h[2]) > 1e-9:
        left_projection = np.vstack((
            (K[0] * h[2] - h[0] * K[2]) / (h[2] * h[2]),
            (K[1] * h[2] - h[1] * K[2]) / (h[2] * h[2]),
        ))
    else:
        left_projection = K[:2] / 1e-9

    Jpixel = np.zeros((3, 3), dtype=np.float64)
    for component in range(2):
        # Clipped left residual components are constant outside the open clip
        # interval. At the kink, fail closed rather than inventing a derivative.
        if abs(float(raw_error[component])) == 1e4:
            raise ValueError("nondifferentiable_left_clip_boundary")
        if abs(float(raw_error[component])) < 1e4:
            Jpixel[component] = left_projection[component]
    if p[2] > 0.0:
        if p[2] > 1e-9:
            Jpixel[2] = left_projection[0] + np.array(
                [0., 0., K[0, 0] * float(baseline) / (p[2] ** 2)])
        else:
            Jpixel[2] = left_projection[0]
    else:  # guarded above; retained for defensive clarity
        Jpixel[:] = 0.0

    Jpoint = Jpixel @ R.T
    Jpose = np.zeros((3, 6), dtype=np.float64)
    Jpose[:, :3] = Jpixel @ _skew(p) @ _right_jacobian(camera_rotvec)
    Jpose[:, 3:] = Jpixel @ (-R.T)
    return row_values[active], Jpose[active], Jpoint[active]


def original_image_pose_observability(
    camera_poses, free_camera_indices, free_camera_rotvecs, selected_points,
    selected_camera_indices, selected_point_indices, selected_pixels,
    selected_right_u, selected_component_mask, excluded_fixed_points,
    excluded_camera_indices, excluded_pixels, excluded_right_u,
    excluded_component_mask, matrix, baseline, disparity_offset,
    pose_scales, point_scales,
):
    """Rank the original image objective after eliminating selected points.

    Camera/point indices refer to the supplied camera and selected-point arrays.
    `pose_scales` and `point_scales` are the exact per-variable scales already
    used by the BA solver. Excluded points are fixed world coordinates and are
    appended directly as camera information rows.
    """
    scope = "original_image_world_chart_only"

    def invalid(reason):
        return {"status": "invalid_geometry", "reason": reason, "scope": scope,
                "pose_columns": int(6 * len(np.asarray(free_camera_indices))),
                "rank": 0, "nullity": int(6 * len(np.asarray(free_camera_indices))),
                "singular_values": [], "tolerance": 0.0,
                "per_point_rank_histogram": {}, "active_row_count": 0,
                "elapsed_s": float(time.perf_counter() - started)}

    started = time.perf_counter()
    try:
        cameras = np.asarray(camera_poses, dtype=np.float64)
        free_idx = np.asarray(free_camera_indices, dtype=np.intp).reshape(-1)
        rotvecs = np.asarray(free_camera_rotvecs, dtype=np.float64).reshape(-1, 3)
        points = np.asarray(selected_points, dtype=np.float64).reshape(-1, 3)
        sc = np.asarray(selected_camera_indices, dtype=np.intp).reshape(-1)
        sp = np.asarray(selected_point_indices, dtype=np.intp).reshape(-1)
        pix = np.asarray(selected_pixels, dtype=np.float64).reshape(-1, 2)
        sr = np.asarray(selected_right_u, dtype=np.float64).reshape(-1)
        sm = np.asarray(selected_component_mask, dtype=bool).reshape(-1, 3)
        fixed_points = np.asarray(excluded_fixed_points, dtype=np.float64).reshape(-1, 3)
        ec = np.asarray(excluded_camera_indices, dtype=np.intp).reshape(-1)
        epix = np.asarray(excluded_pixels, dtype=np.float64).reshape(-1, 2)
        er = np.asarray(excluded_right_u, dtype=np.float64).reshape(-1)
        em = np.asarray(excluded_component_mask, dtype=bool).reshape(-1, 3)
        K = np.asarray(matrix, dtype=np.float64)
        pose_scale = np.asarray(pose_scales, dtype=np.float64).reshape(-1, 6)
        point_scale = np.asarray(point_scales, dtype=np.float64).reshape(-1, 3)
        nfree, npose = len(free_idx), 6 * len(free_idx)
        if (cameras.ndim != 3 or cameras.shape[1:] != (4, 4)
                or rotvecs.shape != (nfree, 3) or pose_scale.shape != (nfree, 6)
                or point_scale.shape != (len(points), 3) or K.shape != (3, 3)
                or len(sc) != len(sp) or len(sc) != len(pix) or len(sc) != len(sr)
                or len(sc) != len(sm) or len(ec) != len(fixed_points)
                or len(ec) != len(epix) or len(ec) != len(er) or len(ec) != len(em)
                or np.any(sc < 0) or np.any(sc >= len(cameras))
                or np.any(ec < 0) or np.any(ec >= len(cameras))
                or np.any(sp < 0) or np.any(sp >= len(points))
                or np.any(free_idx < 0) or np.any(free_idx >= len(cameras))
                or not np.isfinite(cameras).all() or not np.isfinite(points).all()
                or not np.isfinite(pix).all()
                or not np.isfinite(sr[sm[:, 2]]).all()
                or not np.isfinite(fixed_points).all() or not np.isfinite(epix).all()
                or not np.isfinite(er[em[:, 2]]).all() or not np.isfinite(K).all()
                or not np.isfinite(pose_scale).all() or not np.isfinite(point_scale).all()
                or np.any(pose_scale <= 0.) or np.any(point_scale <= 0.)
                or not np.isfinite(float(baseline)) or float(baseline) < 0.
                or not np.isfinite(float(disparity_offset))):
            return invalid("malformed_or_nonfinite_observation_arrays")
        free_lookup = {int(camera): index for index, camera in enumerate(free_idx)}
        if len(free_lookup) != nfree:
            return invalid("duplicate_free_camera_index")

        from analytic_ba import right_jacobian_batch, _projection_jacobian_wrt_cam

        cam_to_slot = np.full(len(cameras), -1, dtype=np.int32)
        if nfree:
            cam_to_slot[free_idx] = np.arange(nfree, dtype=np.int32)
        cam_Jr = np.broadcast_to(np.eye(3, dtype=np.float64), (len(cameras), 3, 3)).copy()
        if nfree:
            cam_Jr[free_idx] = right_jacobian_batch(rotvecs)

        def _batch_jacobians(obs_cams, obs_pts, obs_pix, obs_mask):
            m = len(obs_cams)
            if m == 0:
                return np.empty((0, 3, 6), np.float64), np.empty((0, 3, 3), np.float64)
            sel_cams = cameras[obs_cams]
            R_wc = sel_cams[:, :3, :3]
            t_wc = sel_cams[:, :3, 3]
            pts_c = np.einsum("ni,nij->nj", obs_pts - t_wc, R_wc)
            if not np.isfinite(pts_c).all():
                raise ValueError("undefined_camera_depth")
            z = pts_c[:, 2]
            if np.any((z > 0.0) & (z < 1e-15)):
                raise ValueError("undefined_camera_depth")
            h = pts_c @ K.T
            uv = h[:, :2] / np.maximum(h[:, 2:], 1e-9)
            raw_err = uv - obs_pix
            pos_z = z > 0.0
            if np.any(pos_z[:, None] & (np.abs(raw_err) == 1e4)):
                raise ValueError("nondifferentiable_left_clip_boundary")
            J_pix = _projection_jacobian_wrt_cam(pts_c, K, float(baseline), raw_left_error=raw_err)
            J_point = np.einsum("nij,ncj->nci", R_wc, J_pix)
            J_pose = np.zeros((m, 3, 6), dtype=np.float64)
            slots = cam_to_slot[obs_cams]
            free_m = slots >= 0
            if np.any(free_m):
                fi = np.flatnonzero(free_m)
                vc = np.cross(J_pix[fi], pts_c[fi, None, :])
                J_pose[fi, :, :3] = np.einsum("fci,fij->fcj", vc, cam_Jr[obs_cams[fi]])
                J_pose[fi, :, 3:] = -np.einsum("fij,fcj->fci", R_wc[fi], J_pix[fi])
            return J_pose, J_point

        Jpose_s, Jpt_s = _batch_jacobians(sc, points[sp], pix, sm) if len(sc) else (
            np.empty((0, 3, 6)), np.empty((0, 3, 3))
        )
        active_count = int(sm.sum()) + int(em.sum())

        eps = np.finfo(np.float64).eps
        rank_histogram: dict[str, int] = {}
        reduced = []
        n_pts = len(points)

        if len(sc) and n_pts:
            # Flatten active rows across all selected observations once (zero per-obs np.zeros)
            obs_ids, comp_ids = np.nonzero(sm)
            n_act_s = len(obs_ids)
            if n_act_s:
                pt_of_row = sp[obs_ids]
                cam_of_row = sc[obs_ids]
                slots_of_row = cam_to_slot[cam_of_row]
                Jpt_rows = Jpt_s[obs_ids, comp_ids, :] * point_scale[pt_of_row]  # [R, 3]
                Jpose_rows = Jpose_s[obs_ids, comp_ids, :]  # [R, 6]
                free_r = slots_of_row >= 0
                if np.any(free_r):
                    Jpose_rows = Jpose_rows.copy()
                    Jpose_rows[free_r] *= pose_scale[slots_of_row[free_r]]

                # Group rows by point_index via stable argsort
                order = np.argsort(pt_of_row, kind="mergesort")
                sorted_pts = pt_of_row[order]
                counts = np.bincount(sorted_pts, minlength=n_pts)
                zero_cnt = int(np.count_nonzero(counts == 0))
                if zero_cnt:
                    rank_histogram["0"] = zero_cnt

                starts = np.zeros(n_pts, dtype=np.int64)
                if n_pts > 1:
                    np.cumsum(counts[:-1], out=starts[1:])

                # Group non-empty points by their exact row count m so we run ONE batched
                # np.linalg.svd per distinct m instead of hundreds of individual SVD calls.
                by_m: dict[int, list[int]] = {}
                for pid in np.flatnonzero(counts > 0).tolist():
                    by_m.setdefault(int(counts[pid]), []).append(pid)

                for m_rows, pids in by_m.items():
                    g = len(pids)
                    # Gather [g, m_rows] row indices in `order`
                    base = starts[pids]  # [g]
                    row_idx_2d = order[base[:, None] + np.arange(m_rows, dtype=np.int64)[None, :]]  # [g, m_rows]
                    B_batch = Jpt_rows[row_idx_2d]  # [g, m_rows, 3]
                    if m_rows <= 3:
                        # No nullspace rows possible when rank == m_rows; check rare rank deficiency
                        svals = np.linalg.svd(B_batch, compute_uv=False)  # [g, m_rows]
                        tols = eps * max(m_rows, 3) * svals[:, 0:1]
                        pranks = np.count_nonzero(svals > tols, axis=1)
                        u_r, c_r = np.unique(pranks, return_counts=True)
                        for rv, cv_cnt in zip(u_r.tolist(), c_r.tolist()):
                            rank_histogram[str(rv)] = rank_histogram.get(str(rv), 0) + cv_cnt
                        defic = np.flatnonzero(pranks < m_rows)
                        if len(defic):
                            U_d, s_d, _ = np.linalg.svd(B_batch[defic], full_matrices=True)
                            for di, bi in enumerate(defic.tolist()):
                                pr = int(pranks[bi])
                                r_ids = row_idx_2d[bi]
                                A_mat = np.zeros((m_rows, npose), dtype=np.float64)
                                sl = slots_of_row[r_ids]
                                for lr in range(m_rows):
                                    s_lr = int(sl[lr])
                                    if s_lr >= 0:
                                        A_mat[lr, 6 * s_lr:6 * s_lr + 6] = Jpose_rows[r_ids[lr]]
                                reduced.append(U_d[di, :, pr:].T @ A_mat)
                    else:
                        U_batch, svals, _ = np.linalg.svd(B_batch, full_matrices=True)  # U: [g, m, m]
                        tols = eps * max(m_rows, 3) * svals[:, 0:1]
                        pranks = np.count_nonzero(svals > tols, axis=1)
                        u_r, c_r = np.unique(pranks, return_counts=True)
                        for rv, cv_cnt in zip(u_r.tolist(), c_r.tolist()):
                            rank_histogram[str(rv)] = rank_histogram.get(str(rv), 0) + cv_cnt

                        # Build batched A_batch [g, m_rows, npose]
                        A_batch = np.zeros((g, m_rows, npose), dtype=np.float64)
                        sl_2d = slots_of_row[row_idx_2d]  # [g, m_rows]
                        fg, fm = np.nonzero(sl_2d >= 0)
                        if len(fg):
                            sl_vals = sl_2d[fg, fm]
                            jp_vals = Jpose_rows[row_idx_2d[fg, fm]]  # [K, 6]
                            cols_6 = (6 * sl_vals)[:, None] + np.arange(6, dtype=np.intp)[None, :]
                            A_batch[fg[:, None], fm[:, None], cols_6] = jp_vals

                        # Fast path: all points in this bucket have full rank 3 (99%+ of stereo points)
                        if np.all(pranks == 3):
                            # U[:, :, 3:]^T @ A_batch -> [g, m_rows - 3, npose]
                            proj = np.einsum("gmr,gmp->grp", U_batch[:, :, 3:], A_batch)
                            reduced.append(proj.reshape(-1, npose))
                        else:
                            for bi in range(g):
                                pr = int(pranks[bi])
                                if pr < m_rows:
                                    reduced.append(U_batch[bi, :, pr:].T @ A_batch[bi])
            else:
                if n_pts:
                    rank_histogram["0"] = n_pts
        elif n_pts:
            rank_histogram["0"] = n_pts

        if len(ec):
            Jpose_e, _Jpt_e = _batch_jacobians(ec, fixed_points, epix, em)
            e_obs, e_comp = np.nonzero(em)
            if len(e_obs):
                e_slots = cam_to_slot[ec[e_obs]]
                e_free = np.flatnonzero(e_slots >= 0)
                if len(e_free):
                    f_slots = e_slots[e_free]
                    e_block = np.zeros((len(e_free), npose), dtype=np.float64)
                    cols_6 = (6 * f_slots)[:, None] + np.arange(6, dtype=np.intp)[None, :]
                    e_rows = np.arange(len(e_free), dtype=np.intp)[:, None]
                    e_block[e_rows, cols_6] = (
                        Jpose_e[e_obs[e_free], e_comp[e_free], :] * pose_scale[f_slots]
                    )
                    reduced.append(e_block)

        C = (np.vstack(reduced) if reduced else np.empty((0, npose), dtype=np.float64))
        if not np.isfinite(C).all():
            return invalid("nonfinite_reduced_pose_jacobian")
        singular_values = np.linalg.svd(C, compute_uv=False) if C.size else np.empty(0)
        smax = float(singular_values[0]) if len(singular_values) else 0.0
        tolerance = eps * max(C.shape) * smax if C.size else 0.0
        rank = int(np.count_nonzero(singular_values > tolerance))
        nullity = int(npose - rank)
        return {
            "status": "observable" if nullity == 0 else "unobservable",
            "reason": None if nullity == 0 else "rank_deficient_image_pose_graph",
            "scope": scope,
            "pose_columns": int(npose),
            "rank": rank,
            "nullity": nullity,
            "singular_values": singular_values.tolist(),
            "tolerance": float(tolerance),
            "per_point_rank_histogram": rank_histogram,
            "active_row_count": int(active_count),
            "elapsed_s": float(time.perf_counter() - started),
        }
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, OverflowError) as error:
        return invalid(str(error) or type(error).__name__)
