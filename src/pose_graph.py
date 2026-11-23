"""Portable SE(3) pose graph with a fixed first camera-to-world pose."""
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from kitti import validate_pose


def graph_vertex_groups(first_indices, second_indices, count):
    """Color free vertices so no edge depends on two vertices in one group."""
    neighbours = {vertex: set() for vertex in range(1, count)}
    for first, second in zip(first_indices, second_indices):
        if first and second:
            neighbours[first].add(second)
            neighbours[second].add(first)
    colors = {}
    for vertex in sorted(neighbours, key=lambda v: (-len(neighbours[v]), v)):
        used = {colors[other] for other in neighbours[vertex] if other in colors}
        colors[vertex] = next(color for color in range(count) if color not in used)
    groups = [[vertex for vertex in sorted(colors) if colors[vertex] == color] for color in range(max(colors.values()) + 1)]
    rows = {vertex: (6 * np.flatnonzero((first_indices == vertex) | (second_indices == vertex))[:, None] + np.arange(6)).ravel() for vertex in neighbours}
    return groups, rows


def dense_graph_jacobian(residual, parameters, groups, rows):
    """Grouped forward differences, expanded to a dense Jacobian for SVD."""
    baseline = residual(parameters)
    jacobian = np.zeros((len(baseline), len(parameters)))
    step = np.sqrt(np.finfo(float).eps) * np.where(parameters >= 0, 1., -1.) * np.maximum(1., np.abs(parameters))
    for group in groups:
        for coordinate in range(6):
            columns = np.array([(vertex - 1) * 6 + coordinate for vertex in group])
            perturbed = parameters.copy()
            perturbed[columns] += step[columns]
            difference = residual(perturbed) - baseline
            for vertex, column in zip(group, columns):
                jacobian[rows[vertex], column] = difference[rows[vertex]] / (perturbed[column] - parameters[column])
    return jacobian


def optimize(poses, edges, max_evaluations=100, diagnostics=None):
    if not poses:
        return []
    for pose in poses:
        validate_pose(pose)
    if len(poses) == 1 or not edges:
        return [pose.copy() for pose in poses]
    weights = []
    for first, second, measurement, information, _ in edges:
        if not (0 <= first < len(poses) and 0 <= second < len(poses)) or first == second:
            raise ValueError("Pose graph edge references an invalid vertex")
        validate_pose(measurement)
        info = np.asarray(information)
        if info.shape != (6, 6) or not np.isfinite(info).all() or not np.allclose(info, info.T):
            raise ValueError("Information matrix must be finite symmetric 6x6")
        # W.T W = information; Cholesky returns the opposite orientation.
        weights.append(np.linalg.cholesky(info).T)
    initial = np.array([np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]] for pose in poses[1:]])
    first_indices = np.array([edge[0] for edge in edges])
    second_indices = np.array([edge[1] for edge in edges])
    measurements = np.array([edge[2] for edge in edges])
    measurement_inverse_rotation = measurements[:, :3, :3].transpose(0, 2, 1)
    weights = np.array(weights)

    def unpack(parameters):
        values = parameters.reshape(-1, 6)
        result = np.repeat(np.eye(4)[None], len(poses), axis=0)
        result[0] = poses[0]
        result[1:, :3, :3] = Rotation.from_rotvec(values[:, :3]).as_matrix()
        result[1:, :3, 3] = values[:, 3:]
        return result

    def residual(parameters):
        current = unpack(parameters)
        first, second = current[first_indices], current[second_indices]
        inverse_rotation = first[:, :3, :3].transpose(0, 2, 1)
        # Rigid inverses avoid general matrix inversion for every edge/evaluation.
        relative_rotation = inverse_rotation @ second[:, :3, :3]
        relative_translation = np.einsum('nij,nj->ni', inverse_rotation, second[:, :3, 3] - first[:, :3, 3])
        delta_rotation = measurement_inverse_rotation @ relative_rotation
        delta_translation = np.einsum('nij,nj->ni', measurement_inverse_rotation, relative_translation - measurements[:, :3, 3])
        errors = np.c_[Rotation.from_matrix(delta_rotation).as_rotvec(), delta_translation]
        return np.einsum('nij,nj->ni', weights, errors).ravel()

    sparsity = lil_matrix((6 * len(edges), 6 * (len(poses) - 1)), dtype=int)
    for index, (first, second, *_rest) in enumerate(edges):
        for vertex in (first, second):
            if vertex > 0:
                sparsity[index * 6:(index + 1) * 6, (vertex - 1) * 6:vertex * 6] = 1
    options = dict(max_nfev=max_evaluations, loss="huber", f_scale=1.0,
                   x_scale=np.tile([.1, .1, .1, 1., 1., 1.], len(poses) - 1))
    # Small anchored graphs are ill-conditioned: approximate LSMR steps can
    # stall for thousands of iterations. SVD solves the same objective reliably.
    # Bound the dense Jacobian; larger maps retain the sparse solve.
    dense_bytes = 6 * len(edges) * initial.size * np.dtype(float).itemsize
    started = time.perf_counter()
    def robust_cost(values):
        absolute = np.abs(values)
        return float(np.where(absolute <= 1., .5 * values**2, absolute - .5).sum())
    initial_cost = robust_cost(residual(initial.ravel()))
    if len(poses) <= 300 and dense_bytes <= 64 * 1024 * 1024:
        groups, rows = graph_vertex_groups(first_indices, second_indices, len(poses))
        jacobian_calls = 0
        def jacobian(parameters):
            nonlocal jacobian_calls
            jacobian_calls += 1
            if diagnostics is not None and jacobian_calls % 25 == 0:
                print(f'Graph solve: {jacobian_calls} linearizations, cost {robust_cost(residual(parameters)):.8g}, elapsed {time.perf_counter() - started:.1f}s', flush=True)
            return dense_graph_jacobian(residual, parameters, groups, rows)
        options['jac'] = jacobian
        options['tr_solver'] = 'exact'
        solver = 'dense SVD with grouped forward differences'
    else:
        return optimize_sparse_se3(
            poses, edges, max_evaluations=max_evaluations, huber_scale=1.0, diagnostics=diagnostics
        )
    result = least_squares(residual, initial.ravel(), **options)
    if diagnostics is not None:
        diagnostics.update(solver=solver, success=bool(result.success), function_evaluations=result.nfev,
                           jacobian_evaluations=result.njev, initial_cost=initial_cost, final_cost=float(result.cost),
                           optimality=float(result.optimality), message=result.message,
                           elapsed_s=time.perf_counter() - started, dense_jacobian_bytes=dense_bytes)
    if not np.isfinite(result.x).all() or not result.success:
        raise RuntimeError(f"Pose graph optimization did not converge: {result.message}; "
                           f"evaluations={result.nfev}, cost={result.cost:.8g}, optimality={result.optimality:.8g}")
    return list(unpack(result.x))


def _inv_right_jacobian_batch(rotvecs: np.ndarray) -> np.ndarray:
    """Vectorized inverse right Jacobian J_r^{-1}(phi) of SO(3) for [E, 3] rotvecs."""
    from analytic_ba import skew_batch

    phi = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    n = len(phi)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()
    if n == 0:
        return eye
    theta = np.linalg.norm(phi, axis=1)
    cross = skew_batch(phi)
    cross2 = cross @ cross
    small = theta < 1e-5
    c2 = np.empty(n, dtype=np.float64)
    c2[small] = 1.0 / 12.0
    if np.any(~small):
        t = theta[~small]
        half_t = 0.5 * t
        cot_half = np.cos(half_t) / np.maximum(np.sin(half_t), 1e-12)
        c2[~small] = (1.0 - half_t * cot_half) / (t * t)
    return eye + 0.5 * cross + c2[:, None, None] * cross2


def optimize_sparse_se3(
    poses: list[np.ndarray],
    edges: list[tuple],
    max_evaluations: int = 50,
    huber_scale: float = 1.0,
    diagnostics: dict | None = None,
) -> list[np.ndarray]:
    """Scalable uncapped (>300-KF) vectorized analytic-Jacobian sparse SE(3) pose graph solver."""
    from scipy.sparse import csc_matrix, csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    from analytic_ba import skew_batch

    if not poses:
        return []
    n_poses = len(poses)
    try:
        current = np.asarray(poses, dtype=np.float64)
        if current.shape != (n_poses, 4, 4) or not np.isfinite(current).all():
            raise ValueError("Invalid pose array")
        R_all = current[:, :3, :3]
        orth_err = np.max(np.abs(R_all.transpose(0, 2, 1) @ R_all - np.eye(3)))
        det_err = np.max(np.abs(np.linalg.det(R_all) - 1.0))
        bot_err = np.max(np.abs(current[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
        if orth_err > 1e-4 or det_err > 1e-4 or bot_err > 1e-6:
            raise ValueError("Non-SE(3) pose")
        current = current.copy()
    except Exception:
        for pose in poses:
            validate_pose(pose)
        current = np.asarray([p.copy() for p in poses], dtype=np.float64)
    if n_poses == 1 or not edges:
        return [current[k].copy() for k in range(n_poses)]

    started = time.perf_counter()
    n_edges = len(edges)
    fi = np.empty(n_edges, dtype=np.intp)
    si = np.empty(n_edges, dtype=np.intp)
    meas = np.empty((n_edges, 4, 4), dtype=np.float64)
    infos = np.empty((n_edges, 6, 6), dtype=np.float64)
    for idx, edge in enumerate(edges):
        first, second, measurement, information = edge[0], edge[1], edge[2], edge[3]
        if not (0 <= first < n_poses and 0 <= second < n_poses) or first == second:
            raise ValueError("Pose graph edge references an invalid vertex")
        fi[idx] = int(first)
        si[idx] = int(second)
        m_arr = np.asarray(measurement, dtype=np.float64)
        if m_arr.shape != (4, 4):
            validate_pose(measurement)
        meas[idx] = m_arr
        info_arr = np.asarray(information, dtype=np.float64)
        if info_arr.shape != (6, 6):
            raise ValueError("Information matrix must be finite symmetric 6x6")
        infos[idx] = info_arr

    # Vectorized SE(3) check on measurements and SPD check on information matrices.
    Rm_all = meas[:, :3, :3]
    if (
        not np.isfinite(meas).all()
        or np.max(np.abs(Rm_all.transpose(0, 2, 1) @ Rm_all - np.eye(3))) > 1e-4
        or np.max(np.abs(np.linalg.det(Rm_all) - 1.0)) > 1e-4
        or np.max(np.abs(meas[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6
    ):
        for edge in edges:
            validate_pose(edge[2])
    if not np.isfinite(infos).all() or not np.allclose(infos, infos.transpose(0, 2, 1)):
        raise ValueError("Information matrix must be finite symmetric 6x6")
    # Fast path for diagonal information matrices (standard in SLAM); general Cholesky fallback.
    diag_vals = np.diagonal(infos, axis1=1, axis2=2)
    off_mask = ~np.eye(6, dtype=bool)
    if np.all(infos[:, off_mask] == 0.0) and np.all(diag_vals > 0.0):
        W = np.zeros_like(infos)
        idx6 = np.arange(6)
        W[:, idx6, idx6] = np.sqrt(diag_vals)
    else:
        W = np.linalg.cholesky(infos).transpose(0, 2, 1)

    Rm_T = meas[:, :3, :3].transpose(0, 2, 1)       # [E, 3, 3]
    tm = meas[:, :3, 3]                              # [E, 3]
    delta_h = float(max(huber_scale, 1e-6))
    try:
        from native_accel import solve_sparse_se3_pgo_native
        native_res = solve_sparse_se3_pgo_native(
            current, fi, si, meas, W,
            huber_scale=delta_h,
            max_iter=max(1, min(int(max_evaluations), 60)),
        )
    except Exception:
        native_res = None
    if native_res is not None:
        solved_poses, st = native_res
        if np.isfinite(solved_poses).all() and st["final_cost"] <= st["initial_cost"] + 1e-9:
            if diagnostics is not None:
                diagnostics.update(
                    solver="native sparse block-6x6 SE(3) LM",
                    success=bool(st["converged"] or st["final_cost"] <= st["initial_cost"]),
                    function_evaluations=int(st["nfev"]),
                    jacobian_evaluations=int(st["njev"]),
                    initial_cost=float(st["initial_cost"]),
                    final_cost=float(st["final_cost"]),
                    optimality=float(st["optimality"]),
                    message="converged" if st["converged"] else "max_iterations",
                    elapsed_s=float(time.perf_counter() - started),
                )
            return [solved_poses[k].copy() for k in range(n_poses)]

    n_free = n_poses - 1
    n_vars = 6 * n_free
    n_rows = 6 * n_edges

    def _eval(cur_poses):
        R_i = cur_poses[fi, :3, :3]
        t_i = cur_poses[fi, :3, 3]
        R_j = cur_poses[si, :3, :3]
        t_j = cur_poses[si, :3, 3]
        R_iT = R_i.transpose(0, 2, 1)
        R_hat = R_iT @ R_j
        t_hat = np.einsum("nij,nj->ni", R_iT, t_j - t_i)
        dR = Rm_T @ R_hat
        dt = np.einsum("nij,nj->ni", Rm_T, t_hat - tm)
        phi = Rotation.from_matrix(dR).as_rotvec()
        err = np.hstack((phi, dt))  # [E, 6]
        res = np.einsum("nij,nj->ni", W, err)  # [E, 6]
        a = np.abs(res)
        cmat = np.where(a <= delta_h, 0.5 * res * res, delta_h * (a - 0.5 * delta_h))
        return R_hat, t_hat, dR, phi, res, float(np.sum(cmat))

    R_hat, t_hat, dR, phi, res, cur_cost = _eval(current)
    initial_cost = cur_cost
    lam = 1e-3
    nfev = 1
    njev = 0
    converged = False
    optimality = float("inf")

    fi_free = fi > 0
    si_free = si > 0
    fi_idx = np.flatnonzero(fi_free)
    si_idx = np.flatnonzero(si_free)
    base_6 = np.arange(6, dtype=np.int32)

    for _it in range(max(1, min(int(max_evaluations), 60))):
        njev += 1
        Jr_inv = _inv_right_jacobian_batch(phi)  # [E, 3, 3]

        row_parts = []
        col_parts = []
        data_parts = []

        if len(fi_idx):
            Ji = np.zeros((len(fi_idx), 6, 6), dtype=np.float64)
            Ji[:, :3, :3] = -Jr_inv[fi_idx] @ R_hat[fi_idx].transpose(0, 2, 1)
            Ji[:, 3:, :3] = Rm_T[fi_idx] @ skew_batch(t_hat[fi_idx])
            Ji[:, 3:, 3:] = -Rm_T[fi_idx]
            WJi = W[fi_idx] @ Ji  # [Fi, 6, 6]
            r_block = (6 * fi_idx)[:, None, None] + base_6[None, :, None]
            c_block = (6 * (fi[fi_idx] - 1))[:, None, None] + base_6[None, None, :]
            row_parts.append(np.broadcast_to(r_block, WJi.shape).ravel())
            col_parts.append(np.broadcast_to(c_block, WJi.shape).ravel())
            data_parts.append(WJi.ravel())

        if len(si_idx):
            Jj = np.zeros((len(si_idx), 6, 6), dtype=np.float64)
            Jj[:, :3, :3] = Jr_inv[si_idx]
            Jj[:, 3:, 3:] = dR[si_idx]
            WJj = W[si_idx] @ Jj  # [Si, 6, 6]
            r_block = (6 * si_idx)[:, None, None] + base_6[None, :, None]
            c_block = (6 * (si[si_idx] - 1))[:, None, None] + base_6[None, None, :]
            row_parts.append(np.broadcast_to(r_block, WJj.shape).ravel())
            col_parts.append(np.broadcast_to(c_block, WJj.shape).ravel())
            data_parts.append(WJj.ravel())

        J = csr_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(n_rows, n_vars),
            dtype=np.float64,
        )
        r_flat = res.ravel()
        abs_r = np.abs(r_flat)
        irls = np.where(abs_r <= delta_h, 1.0, delta_h / np.maximum(abs_r, 1e-12))
        JT = J.T.tocsr()
        J_w = J.multiply(irls[:, None]).tocsr()
        H = (JT @ J_w).tocsc()
        g = JT @ (irls * r_flat)
        optimality = float(np.max(np.abs(g))) if len(g) else 0.0
        if not np.isfinite(optimality):
            break
        if optimality < 1e-6:
            converged = True
            break

        h_diag = np.maximum(H.diagonal(), 1e-6)
        accepted = False
        for _lm_try in range(8):
            H_damp = (H + diags(lam * h_diag, format="csc")).tocsc()
            try:
                step = -spsolve(H_damp, g)
            except Exception:
                lam = min(lam * 10.0, 1e8)
                continue
            if not np.isfinite(step).all():
                lam = min(lam * 10.0, 1e8)
                continue

            step_mat = step.reshape(n_free, 6)
            cand = current.copy()
            dR_step = Rotation.from_rotvec(step_mat[:, :3]).as_matrix()
            cand[1:, :3, 3] = current[1:, :3, 3] + np.einsum("nij,nj->ni", current[1:, :3, :3], step_mat[:, 3:])
            cand[1:, :3, :3] = current[1:, :3, :3] @ dR_step

            rh_c, th_c, dR_c, phi_c, res_c, cost_c = _eval(cand)
            nfev += 1
            if np.isfinite(cost_c) and cost_c < cur_cost:
                rel_impr = (cur_cost - cost_c) / max(cur_cost, 1e-12)
                current = cand
                R_hat, t_hat, dR, phi, res, cur_cost = rh_c, th_c, dR_c, phi_c, res_c, cost_c
                lam = max(lam * 0.3, 1e-7)
                accepted = True
                if rel_impr < 1e-8 or float(np.linalg.norm(step)) < 1e-6:
                    converged = True
                break
            lam = min(lam * 5.0, 1e8)

        if converged or not accepted:
            if not accepted:
                converged = True
            break

    if diagnostics is not None:
        diagnostics.update(
            solver="sparse analytic SE(3) LM",
            success=bool(converged or cur_cost <= initial_cost),
            function_evaluations=int(nfev),
            jacobian_evaluations=int(njev),
            initial_cost=float(initial_cost),
            final_cost=float(cur_cost),
            optimality=float(optimality),
            message="converged" if converged else "max_iterations",
            elapsed_s=float(time.perf_counter() - started),
        )
    return [current[k].copy() for k in range(n_poses)]


def optimize_weighted_pose_graph(
    poses_wc: dict[int, np.ndarray] | list[np.ndarray],
    odometry_edges: list[tuple],
    loop_edges: list[tuple],
    huber_scale: float = 1.0,
    max_evaluations: int = 60,
    diagnostics: dict | None = None,
) -> dict[int, np.ndarray]:
    """Optimize an arbitrary-size (>300-KF) SE(3) pose graph with distance-scaled odometry & inlier-scaled loop weights."""
    if isinstance(poses_wc, dict):
        kf_ids = sorted(poses_wc.keys())
        id_to_idx = {kid: idx for idx, kid in enumerate(kf_ids)}
        poses_list = [np.asarray(poses_wc[kid], dtype=np.float64).copy() for kid in kf_ids]
    else:
        kf_ids = list(range(len(poses_wc)))
        id_to_idx = {kid: kid for kid in kf_ids}
        poses_list = [np.asarray(p, dtype=np.float64).copy() for p in poses_wc]

    edges = []
    for edge in odometry_edges:
        i_id, j_id, meas = edge[0], edge[1], np.asarray(edge[2], dtype=np.float64)
        if i_id not in id_to_idx or j_id not in id_to_idx:
            continue
        if len(edge) >= 4 and edge[3] is not None:
            info = np.asarray(edge[3], dtype=np.float64)
        else:
            dist_m = max(float(np.linalg.norm(meas[:3, 3])), 0.5)
            inv_d = 1.0 / dist_m
            info = np.diag([1000.0 * inv_d] * 3 + [10.0 * inv_d] * 3)
        edges.append((id_to_idx[i_id], id_to_idx[j_id], meas, info, "odometry"))

    for edge in loop_edges:
        i_id, j_id, meas = edge[0], edge[1], np.asarray(edge[2], dtype=np.float64)
        if i_id not in id_to_idx or j_id not in id_to_idx:
            continue
        if len(edge) >= 4 and isinstance(edge[3], np.ndarray) and edge[3].shape == (6, 6):
            info = np.asarray(edge[3], dtype=np.float64)
        else:
            meta = edge[3] if len(edge) >= 4 and isinstance(edge[3], dict) else {}
            inliers = float(meta.get("inliers", 50))
            med_err = max(0.5, float(meta.get("median_reprojection_px", 1.0)))
            scale = float(np.clip(inliers / 50.0, 0.5, 4.0) / (med_err * med_err))
            info = np.diag([2000.0 * scale] * 3 + [20.0 * scale] * 3)
        edges.append((id_to_idx[i_id], id_to_idx[j_id], meas, info, "loop"))

    optimized = optimize_sparse_se3(
        poses_list,
        edges,
        max_evaluations=max_evaluations,
        huber_scale=huber_scale,
        diagnostics=diagnostics,
    )
    return {kf_ids[idx]: optimized[idx] for idx in range(len(kf_ids))}
