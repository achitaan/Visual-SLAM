"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from mapping_geometry import project, right_pixel
from stereo_motion_regularizer import stereo_motion_residual


def local_bundle_adjustment(
    state,
    matrix,
    baseline=0.0,
    window=5,
    max_landmarks=200,
    disparity_offset=0.0,
    optimized=True,
    stereo_motion_regularizer=False,
):
    input_matrix = matrix
    if stereo_motion_regularizer:
        # The enabled factor and its pixel Jacobians are tied to one immutable
        # calibration snapshot throughout this solve.
        try:
            raw_matrix = np.asarray(matrix)
            if np.iscomplexobj(raw_matrix):
                return {"applied": False, "reason": "invalid_regularizer_calibration"}
            matrix = np.array(raw_matrix, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError):
            return {"applied": False, "reason": "invalid_regularizer_calibration"}
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return {"applied": False, "reason": "invalid_regularizer_calibration"}
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
        regularizer_checks = []
        regularizer_skipped = []
        if stereo_motion_regularizer:
            matrix_snapshot = np.asarray(matrix, float)
            for edge, factor in state.stereo_motion_regularizers.items():
                first, second = edge
                reason = None
                if edge not in state.stereo_motion:
                    reason = "verified_motion_edge_missing"
                elif (not 0 <= first < second < len(state.poses)
                      or second >= len(state.pose_anchors)):
                    reason = "invalid_endpoint_frames"
                elif (state.statuses[first] not in ("tracking", "relocalized")
                      or state.statuses[second] not in ("tracking", "relocalized")
                      or factor.source_status != state.statuses[first]
                      or factor.target_status != state.statuses[second]):
                    reason = "endpoint_status_mismatch"
                elif (factor.source_frame != first or factor.target_frame != second
                      or not np.array_equal(factor.measurement,
                                            state.stereo_motion[edge])):
                    reason = "measurement_or_frame_mismatch"
                elif (factor.matrix.shape != (3, 3)
                      or not np.array_equal(factor.matrix, matrix_snapshot)
                      or factor.baseline != float(baseline)
                      or factor.disparity_offset != float(disparity_offset)):
                    reason = "calibration_mismatch"
                elif (factor.model != "correlated_pixel_motion_regularizer"
                      or factor.noise_model != "independent_1px_u_v_disparity_per_endpoint_v1"
                      or factor.tangent_order != (
                          "rho_x", "rho_y", "rho_z", "phi_x", "phi_y", "phi_z")
                      or factor.covariance_claim is not False
                      or factor.intentionally_reuses_sensor_evidence is not True
                      or factor.holdout_rows_used is not False):
                    reason = "invalid_factor_provenance_metadata"
                elif (factor.sqrt_information.shape != (6, 6)
                      or not np.isfinite(factor.sqrt_information).all()
                      or factor.information.shape != (6, 6)
                      or not np.isfinite(factor.information).all()
                      or not np.allclose(
                          factor.sqrt_information.T @ factor.sqrt_information,
                          factor.information, rtol=1e-9, atol=1e-10)
                      or factor.rank != 6
                      or not np.isfinite(factor.condition_number)
                      or factor.condition_number <= 0.):
                    reason = "invalid_information_factor"
                anchors = (state.pose_anchors[first], state.pose_anchors[second])
                if reason is None and (anchors[0] == anchors[1]
                                       or not any(anchor in pose_offset for anchor in anchors)):
                    reason = "constant_endpoint_motion"
                if reason is not None:
                    regularizer_skipped.append({
                        "source_frame": int(first), "target_frame": int(second),
                        "reason": reason,
                    })
                    continue
                regularizer_checks.append((
                    factor, anchors,
                    (state.poses[first].copy(), state.poses[second].copy()),
                ))

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
    motion_rows = 6 * len(regularizer_checks) if stereo_motion_regularizer else 0
    pattern = lil_matrix((sum(dimensions)+sum(held_dimensions)+motion_rows,
                          len(initial)), dtype=int)
    row = 0
    for (i, k, _), dim in zip(records, dimensions):
        if k in pose_offset:
            pattern[row : row + dim, pose_offset[k] : pose_offset[k] + 6] = 1
        pattern[row : row + dim, point_offset + 3 * i : point_offset + 3 * i + 3] = 1
        row += dim
    for (_, k, _), dim in zip(held_out, held_dimensions):
        pattern[row : row+dim, pose_offset[k] : pose_offset[k]+6] = 1
        row += dim
    if stereo_motion_regularizer:
        for _, anchors, _ in regularizer_checks:
            for anchor in set(anchors):
                if anchor in pose_offset:
                    pattern[row:row + 6, pose_offset[anchor]:pose_offset[anchor] + 6] = 1
            row += 6

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
        values[:, 2] = (
            right_pixel(projected[:, 0], z, matrix[0, 0], baseline, disparity_offset)
            - right
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

    def motion_regularizer_residual(x):
        if not stereo_motion_regularizer or not regularizer_checks:
            return np.empty(0)
        poses, _ = unpack(x)
        values = []
        for factor, anchors, recorded in regularizer_checks:
            updated = [
                poses[anchor] @ np.linalg.inv(base[anchor]) @ frame_pose
                if anchor is not None else frame_pose
                for anchor, frame_pose in zip(anchors, recorded)
            ]
            values.extend(stereo_motion_residual(factor, updated[0], updated[1]))
        return np.asarray(values, dtype=float)

    def residual(x):
        cameras = camera_poses_for(x)
        points = x[point_offset:].reshape(-1, 3)
        image_rows = np.r_[
            optimized_residual(x, cameras, points),
            held_out_residual(cameras),
        ]
        if not stereo_motion_regularizer:
            return image_rows
        return np.r_[image_rows, motion_regularizer_residual(x)]

    initial_cameras = camera_poses_for(initial)
    before = objective(optimized_residual(initial, initial_cameras))
    held_before = objective(held_out_residual(initial_cameras))
    if stereo_motion_regularizer:
        regularizer_before = objective(motion_regularizer_residual(initial))
        augmented_before = before + held_before + regularizer_before
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
    if stereo_motion_regularizer:
        regularizer_after = objective(motion_regularizer_residual(result.x))
        augmented_after = after + held_after + regularizer_after
        report.update(
            stereo_motion_regularizer={
                "model": "correlated_pixel_motion_regularizer",
                "covariance_claim": False,
                "intentionally_reuses_sensor_evidence": True,
                "holdout_rows_used": False,
                "active_factors": len(regularizer_checks),
                "skipped_factors": regularizer_skipped,
                "active_edges": [[int(f.source_frame), int(f.target_frame)]
                                 for f, _, _ in regularizer_checks],
                "initial_huber_cost": regularizer_before,
                "final_huber_cost": regularizer_after,
                "augmented_initial_huber_cost": augmented_before,
                "augmented_final_huber_cost": augmented_after,
                "fit_sources": sorted({factor.fit_source
                                        for factor, _, _ in regularizer_checks}),
                "fit_depth_policies": sorted({factor.fit_depth_policy
                                               for factor, _, _ in regularizer_checks}),
                "training_row_ids": {
                    f"{factor.source_frame}:{factor.target_frame}":
                    {
                        "all_training": factor.training_pairs.tolist(),
                        "forward_inliers": factor.forward_inlier_pairs.tolist(),
                        "reverse_inliers": factor.reverse_inlier_pairs.tolist(),
                        "information_rows": factor.information_pairs.tolist(),
                    }
                    for factor, _, _ in regularizer_checks
                },
                "overlap_with_reprojection_rows": "intentional_sensor_row_reuse",
            }
        )
        if not np.isfinite(held_after) or not after+held_after < before+held_before:
            return {**report, "reason": "affected_observations_worsened"}
        if (not np.isfinite(augmented_after)
                or not augmented_after < augmented_before):
            return {**report, "reason": "augmented_regularized_objective_worsened"}
    elif not np.isfinite(held_after) or not after+held_after < before+held_before:
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
        if stereo_motion_regularizer:
            try:
                live_matrix = np.asarray(input_matrix)
                calibration_unchanged = (
                    not np.iscomplexobj(live_matrix)
                    and live_matrix.shape == (3, 3)
                    and np.isfinite(live_matrix).all()
                    and np.array_equal(np.asarray(live_matrix, float), matrix)
                )
            except (TypeError, ValueError, OverflowError):
                calibration_unchanged = False
            if not calibration_unchanged:
                return {**report, "reason": "stale_regularizer_calibration"}
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
