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

        by_point = [[] for _ in range(len(points))]
        active_count = 0
        for row in range(len(sc)):
            camera_index, point_index = int(sc[row]), int(sp[row])
            phi = rotvecs[free_lookup[camera_index]] if camera_index in free_lookup else np.zeros(3)
            values, pose_jac, point_jac = _observation_jacobians(
                cameras[camera_index], phi, points[point_index], pix[row], sr[row],
                sm[row], K, baseline, disparity_offset)
            active_count += len(values)
            pose_block = np.zeros((len(values), npose), dtype=np.float64)
            if camera_index in free_lookup:
                slot = free_lookup[camera_index]
                pose_block[:, 6 * slot:6 * slot + 6] = pose_jac * pose_scale[slot]
            point_block = point_jac * point_scale[point_index]
            by_point[point_index].append((pose_block, point_block))

        reduced = []
        rank_histogram = {}
        eps = np.finfo(np.float64).eps
        for blocks in by_point:
            if not blocks:
                rank_histogram["0"] = rank_histogram.get("0", 0) + 1
                continue
            A = np.vstack([block[0] for block in blocks])
            B = np.vstack([block[1] for block in blocks])
            U, singular, _vh = np.linalg.svd(B, full_matrices=True)
            smax = float(singular[0]) if len(singular) else 0.0
            tol = eps * max(B.shape) * smax
            point_rank = int(np.count_nonzero(singular > tol))
            rank_histogram[str(point_rank)] = rank_histogram.get(str(point_rank), 0) + 1
            if point_rank < U.shape[1]:
                reduced.append(U[:, point_rank:].T @ A)

        for row in range(len(ec)):
            camera_index = int(ec[row])
            phi = rotvecs[free_lookup[camera_index]] if camera_index in free_lookup else np.zeros(3)
            _values, pose_jac, _point_jac = _observation_jacobians(
                cameras[camera_index], phi, fixed_points[row], epix[row], er[row],
                em[row], K, baseline, disparity_offset)
            active_count += len(_values)
            if camera_index in free_lookup:
                slot = free_lookup[camera_index]
                block = np.zeros((len(pose_jac), npose), dtype=np.float64)
                block[:, 6 * slot:6 * slot + 6] = pose_jac * pose_scale[slot]
                reduced.append(block)

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
