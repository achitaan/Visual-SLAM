"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from mapping_geometry import project, right_pixel


def local_bundle_adjustment(
    state,
    matrix,
    baseline=0.0,
    window=5,
    max_landmarks=200,
    disparity_offset=0.0,
    optimized=True,
    stereo_residuals="left_right",
):
    if stereo_residuals not in ("left_right", "left_disparity"):
        raise ValueError("stereo_residuals must be 'left_right' or 'left_disparity'")
    if stereo_residuals == "left_disparity":
        try:
            raw_matrix = np.asarray(matrix)
            if np.iscomplexobj(raw_matrix) or raw_matrix.shape != (3, 3):
                raise ValueError
            matrix = np.array(raw_matrix, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "left_disparity residuals require a finite normalized real 3x3 calibration"
            ) from None
        if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0
                or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)):
            raise ValueError(
                "left_disparity residuals require finite fx/fy>0 and a normalized camera row"
            )
        try:
            baseline_value = np.asarray(baseline)
            if (baseline_value.ndim != 0 or np.iscomplexobj(baseline_value)
                    or isinstance(baseline, (bool, np.bool_))):
                raise ValueError
            baseline = float(baseline_value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "left_disparity residuals require a finite positive stereo baseline"
            ) from None
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                "left_disparity residuals require a finite positive stereo baseline"
            )
        if not state.metric:
            raise ValueError("left_disparity residuals require a metric stereo map")
        try:
            offset_value = np.asarray(disparity_offset)
            if (offset_value.ndim != 0 or np.iscomplexobj(offset_value)
                    or isinstance(disparity_offset, (bool, np.bool_))):
                raise ValueError
            disparity_offset = float(offset_value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("left_disparity residuals require a finite real disparity offset") from None
        if not np.isfinite(disparity_offset):
            raise ValueError("left_disparity residuals require a finite real disparity offset")

    with state.lock:
        if len(state.keyframes) < 3:
            return {"applied": False, "reason": "insufficient_keyframes"}
        revision = state.revision
        selected = list(state.keyframes)[-window:]
        # Hold the two oldest monocular anchors to fix scale in the local window.
        fixed = set(selected[: 1 if state.metric else 2])
        free = [k for k in selected if k not in fixed]
        landmarks = [
            l
            for l in state.landmarks.values()
            if sum(k in selected for k in l.observations) >= 2
        ]
        landmarks.sort(
            key=lambda l: (
                sum(k in selected for k in l.observations),
                max(l.observations),
            ),
            reverse=True,
        )
        landmarks = landmarks[:max_landmarks]
        if len(landmarks) < 15:
            return {"applied": False, "reason": "insufficient_observations"}
        base = {k: state.keyframes[k].pose.copy() for k in state.keyframes}
        records = [
            (i, k, o)
            for i, l in enumerate(landmarks)
            for k, o in l.observations.items()
            if k in base
        ]
        # The bounded landmark selection can disconnect cameras from the oldest
        # fixed keyframe. Every observed component needs its own world gauge;
        # otherwise a lower reprojection objective can hide arbitrary map motion.
        neighbors = {}
        for landmark in landmarks:
            cameras = [k for k in landmark.observations if k in base]
            for k in cameras:
                neighbors.setdefault(k, set()).update(cameras)
        unsupported = set(free) - neighbors.keys()
        fixed.update(unsupported)
        remaining = set(neighbors)
        components = []
        while remaining:
            stack = [min(remaining)]
            component = set()
            while stack:
                k = stack.pop()
                if k in component:
                    continue
                component.add(k)
                stack.extend(neighbors[k] - component)
            remaining.difference_update(component)
            components.append(component)
            anchors = sorted(k for k in component if k not in free or k in fixed)
            if not anchors:
                anchors = [min(component)]
                fixed.add(anchors[0])
            metric_support = state.metric and any(
                o.right_u is not None for _, k, o in records if k in component
            )
            if not metric_support:
                # A monocular component needs distinct fixed camera centers to
                # constrain scale, including fixed boundary cameras outside the
                # local window. Coincident anchors cannot fix the scale gauge.
                origin = base[anchors[0]][:3, 3]
                separated = [
                    k for k in anchors if np.linalg.norm(base[k][:3, 3] - origin) > 1e-6
                ]
                if not separated:
                    candidates = [
                        k
                        for k in sorted(component)
                        if np.linalg.norm(base[k][:3, 3] - origin) > 1e-6
                    ]
                    if candidates:
                        fixed.add(candidates[0])
                    else:
                        fixed.update(component)
        free = [k for k in free if k not in fixed]
        if not free:
            return {"applied": False, "reason": "no_anchored_free_cameras"}
        optimized_ids = {landmark.id for landmark in landmarks}
        held_out = [
            (landmark.position.copy(), k, o)
            for landmark in state.landmarks.values()
            if landmark.id not in optimized_ids and len(landmark.observations) >= 2
            for k, o in landmark.observations.items()
            if k in free
        ]
        single_view = [
            (landmark.id, landmark.anchor, landmark.position.copy())
            for landmark in state.landmarks.values()
            if landmark.id not in optimized_ids and len(landmark.observations) == 1
            and landmark.anchor in free and landmark.anchor in landmark.observations
        ]
        pose_offset = {k: 6 * i for i, k in enumerate(free)}
        point_offset = 6 * len(free)
        initial = np.r_[
            np.concatenate(
                [
                    np.r_[
                        Rotation.from_matrix(base[k][:3, :3]).as_rotvec(),
                        base[k][:3, 3],
                    ]
                    for k in free
                ]
            ),
            np.array([l.position for l in landmarks]).ravel(),
        ]
        # Huber-weighted Jacobian columns can become almost zero for otherwise
        # valid large residuals. Using their norms for variable scaling can then
        # make the trust region stall at a poor pose. Use geometry units instead:
        # radians for rotation and one baseline for translations/world points.
        if state.metric and baseline > 0:
            length_scale = baseline
        else:
            separations = np.linalg.norm(np.diff(
                np.array([base[k][:3, 3] for k in selected]), axis=0), axis=1)
            positive = separations[separations > 1e-8]
            length_scale = float(np.median(positive)) if len(positive) else 1.
        variable_scale = np.r_[
            np.tile([1., 1., 1., length_scale, length_scale, length_scale], len(free)),
            np.full(3 * len(landmarks), length_scale),
        ]
        motion_checks = []
        for (first, second), measurement in state.stereo_motion.items():
            anchors = (state.pose_anchors[first], state.pose_anchors[second])
            # A shared rigid correction preserves relative motion within one anchor.
            if anchors[0] == anchors[1] or not any(a in free for a in anchors):
                continue
            motion_checks.append((measurement.copy(), anchors,
                                  (state.poses[first].copy(), state.poses[second].copy())))

    def unpack(x):
        poses = {k: p.copy() for k, p in base.items()}
        for k, offset in pose_offset.items():
            poses[k][:3, :3] = Rotation.from_rotvec(x[offset : offset + 3]).as_matrix()
            poses[k][:3, 3] = x[offset + 3 : offset + 6]
        return poses, x[point_offset:].reshape(-1, 3)

    dimensions = np.asarray(
        [3 if state.metric and o.right_u is not None else 2 for _, _, o in records],
        dtype=int,
    )
    held_dimensions = np.asarray(
        [3 if state.metric and o.right_u is not None else 2 for _, _, o in held_out],
        dtype=int,
    )
    pattern = lil_matrix((sum(dimensions)+sum(held_dimensions), len(initial)), dtype=int)
    row = 0
    for (i, k, _), dim in zip(records, dimensions):
        if k in pose_offset:
            pattern[row : row + dim, pose_offset[k] : pose_offset[k] + 6] = 1
        pattern[row : row + dim, point_offset + 3 * i : point_offset + 3 * i + 3] = 1
        row += dim
    for (_, k, _), dim in zip(held_out, held_dimensions):
        pattern[row : row+dim, pose_offset[k] : pose_offset[k]+6] = 1
        row += dim

    # Residual rows repeatedly use the same point indices, camera indices, and
    # measurements. Materialize those once so each solver evaluation only
    # projects arrays instead of rebuilding them from Python observation tuples.
    camera_ids = list(
        dict.fromkeys(
            [k for _, k, _ in records] + [k for _, k, _ in held_out]
        )
    )
    camera_index = {k: i for i, k in enumerate(camera_ids)}
    base_cameras = np.asarray([base[k] for k in camera_ids])
    record_points = np.asarray([i for i, _, _ in records], dtype=np.intp)
    record_cameras = np.asarray(
        [camera_index[k] for _, k, _ in records], dtype=np.intp
    )
    measured_pixels = np.asarray([o.pixel for _, _, o in records], dtype=float)
    measured_right = np.asarray(
        [o.right_u if o.right_u is not None else 0.0 for _, _, o in records],
        dtype=float,
    )
    dimension_mask = np.column_stack(
        (np.ones(len(records), dtype=bool), np.ones(len(records), dtype=bool), dimensions == 3)
    )
    held_points = np.asarray([p for p, _, _ in held_out], dtype=float).reshape(-1, 3)
    held_cameras = np.asarray(
        [camera_index[k] for _, k, _ in held_out], dtype=np.intp
    )
    held_pixels = np.asarray([o.pixel for _, _, o in held_out], dtype=float).reshape(-1, 2)
    held_right = np.asarray(
        [o.right_u if o.right_u is not None else 0.0 for _, _, o in held_out],
        dtype=float,
    )
    held_dimension_mask = np.column_stack(
        (
            np.ones(len(held_out), dtype=bool),
            np.ones(len(held_out), dtype=bool),
            held_dimensions == 3,
        )
    )
    free_camera_ids = [k for k in free if k in camera_index]
    free_camera_indices = np.asarray(
        [camera_index[k] for k in free_camera_ids], dtype=np.intp
    )
    free_offsets = np.asarray(
        [pose_offset[k] for k in free_camera_ids], dtype=np.intp
    ).reshape(-1, 1) + np.arange(6, dtype=np.intp)

    def camera_poses_for(x):
        if not optimized:
            poses, _ = unpack(x)
            return np.asarray([poses[k] for k in camera_ids])
        cameras = base_cameras.copy()
        if len(free_offsets):
            values = x[free_offsets]
            cameras[free_camera_indices, :3, :3] = Rotation.from_rotvec(
                values[:, :3]
            ).as_matrix()
            cameras[free_camera_indices, :3, 3] = values[:, 3:]
        return cameras

    def observation_residual(points, cameras, camera_indices, pixels, right, mask):
        if not len(camera_indices):
            return np.empty(0)
        selected_cameras = cameras[camera_indices]
        coordinates = points - selected_cameras[:, :3, 3]
        camera = np.einsum("ni,nij->nj", coordinates, selected_cameras[:, :3, :3])
        homogeneous = camera @ matrix.T
        z = camera[:, 2]
        projected = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        errors = np.clip(projected - pixels, -1e4, 1e4)
        errors[z <= 0] = 1e4
        if not state.metric:
            return errors.ravel()
        values = np.zeros((len(camera_indices), 3))
        values[:, :2] = errors
        if stereo_residuals == "left_right":
            # Preserve the original observation residual exactly by default.
            values[:, 2] = (
                right_pixel(projected[:, 0], z, matrix[0, 0], baseline, disparity_offset)
                - right
            )
        else:
            # Treat measured (u, v, disparity) as the declared uniform
            # approximate residual basis. Compute disparity directly rather
            # than combining clipped image residuals: for invalid depth that
            # combination could cancel the large penalties below.
            measured_disparity = pixels[:, 0] - right
            values[:, 2] = (
                matrix[0, 0] * baseline / np.maximum(z, 1e-9)
                + disparity_offset
                - measured_disparity
            )
        values[z <= 0] = 1e4
        return values[mask]

    def optimized_residual(x, cameras=None, points=None):
        if points is None:
            points = x[point_offset:].reshape(-1, 3)
        if cameras is None:
            cameras = camera_poses_for(x)
        return observation_residual(
            points[record_points],
            cameras,
            record_cameras,
            measured_pixels,
            measured_right,
            dimension_mask,
        )

    def objective(r):
        a = np.abs(r)
        return float(np.sum(np.where(a <= 2.0, 0.5 * r * r, 2.0 * (a - 1.0))))

    def held_out_residual(cameras):
        if not held_out:
            return np.empty(0)
        return observation_residual(
            held_points,
            cameras,
            held_cameras,
            held_pixels,
            held_right,
            held_dimension_mask,
        )

    def residual(x):
        cameras = camera_poses_for(x)
        points = x[point_offset:].reshape(-1, 3)
        return np.r_[
            optimized_residual(x, cameras, points),
            held_out_residual(cameras),
        ]

    initial_cameras = camera_poses_for(initial)
    before = objective(optimized_residual(initial, initial_cameras))
    held_before = objective(held_out_residual(initial_cameras))
    result = least_squares(
        residual,
        initial,
        jac_sparsity=pattern.tocsr(),
        loss="huber",
        f_scale=2.0,
        max_nfev=30,
        x_scale=variable_scale,
        tr_solver="lsmr",
    )
    result_cameras = camera_poses_for(result.x)
    result_points = result.x[point_offset:].reshape(-1, 3)
    after = objective(optimized_residual(result.x, result_cameras, result_points))
    report = {
        "applied": False,
        "initial_cost": before,
        "final_cost": after,
        "evaluations": int(result.nfev),
        "solver_success": bool(result.success),
        "fixed_keyframes": sorted(set(base) - set(free)),
        "observation_components": len(components),
        "unsupported_local_cameras": sorted(unsupported),
    }
    if stereo_residuals == "left_disparity":
        stereo_observations = int(np.count_nonzero(dimensions == 3)
                                  + np.count_nonzero(held_dimensions == 3))
        mono_observations = int(np.count_nonzero(dimensions == 2)
                                + np.count_nonzero(held_dimensions == 2))
        report["stereo_residual_model"] = {
            "mode": "left_disparity",
            "model": "uniform_independent_u_v_disparity_1px_assumption",
            "formula": (
                "e_d=(fx*baseline/z+disparity_offset)"
                "-(measured_left_u-measured_right_u)"
            ),
            "covariance_claim": False,
            "source_specific_noise_model": False,
            "source_provenance": "unavailable",
            "approximate_uniform_noise": True,
            "pose_priors_added": False,
            "objective_values_across_modes_comparable": False,
            "active_stereo_observations": stereo_observations,
            "active_mono_observations": mono_observations,
            "active_stereo_rows": 3 * stereo_observations,
            "active_mono_rows": 2 * mono_observations,
        }
    if not np.isfinite(result.x).all():
        return report
    poses, points = unpack(result.x)
    held_after = objective(held_out_residual(result_cameras))
    report.update(
        held_out_observations=len(held_out),
        held_out_initial_cost=held_before,
        held_out_final_cost=held_after,
        affected_initial_cost=before+held_before,
        affected_final_cost=after+held_after,
        optimized_landmarks=len(landmarks),
        anchor_propagated_single_view_landmarks=len(single_view),
        max_camera_translation_change=float(max(
            np.linalg.norm(poses[k][:3, 3]-base[k][:3, 3]) for k in free)),
    )
    if not np.isfinite(held_after) or not after+held_after < before+held_before:
        return {**report, "reason": "affected_observations_worsened"}
    if any(
        np.any(project(points[i : i + 1], poses[k], matrix)[1] <= 0)
        for i, k, _ in records
    ):
        return report
    motion_translation, motion_rotation = [], []
    for measurement, anchors, recorded in motion_checks:
        updated = [
            poses[anchor] @ np.linalg.inv(base[anchor]) @ pose
            if anchor is not None else pose
            for anchor, pose in zip(anchors, recorded)
        ]
        difference = np.linalg.inv(measurement) @ np.linalg.inv(updated[0]) @ updated[1]
        motion_translation.append(float(np.linalg.norm(difference[:3, 3])))
        motion_rotation.append(float(np.degrees(Rotation.from_matrix(difference[:3, :3]).magnitude())))
    report.update(
        independent_stereo_motion_checks=len(motion_checks),
        max_stereo_motion_translation_error_m=max(motion_translation, default=0.),
        max_stereo_motion_rotation_error_deg=max(motion_rotation, default=0.),
    )
    # Apply the same agreement limits used before tracking acceptance, including
    # earlier frame boundaries affected by this local correction.
    if any(v > .5 for v in motion_translation) or any(v > 1.5 for v in motion_rotation):
        return {**report, "reason": "independent_stereo_motion_inconsistency"}
    with state.lock:
        if state.revision != revision:
            return {**report, "reason": "stale_revision"}
        updates = {l.id: p.copy() for l, p in zip(landmarks, points)}
        for ident, anchor, position in single_view:
            camera = base[anchor][:3, :3].T @ (position-base[anchor][:3, 3])
            updates[ident] = poses[anchor][:3, :3] @ camera + poses[anchor][:3, 3]
        if not state.apply_corrections(revision, poses, propagate_landmarks=False, landmark_updates=updates):
            return report
        # Multiview world points are independent; single-view stereo points retain
        # their measured camera coordinates rather than imposing a pose prior.
        report["applied"] = True
    return report
