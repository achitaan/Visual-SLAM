"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from mapping_geometry import project, right_pixel


def _diagnostic_array(value):
    """Return a deep-owned primitive snapshot without retaining NumPy views."""
    return np.asarray(value).copy().tolist()


def _diagnostic_vector(value):
    """Represent invalid numeric entries without emitting non-finite JSON."""
    array = np.asarray(value).reshape(-1)
    valid = np.isfinite(array)
    return {
        "values": [float(item) if is_valid else None
                   for item, is_valid in zip(array, valid)],
        "valid_mask": valid.astype(bool).tolist(),
        "invalid_indices": np.flatnonzero(~valid).astype(int).tolist(),
    }


def _diagnostic_value(value, invalid_fields=None, path="$"):
    """Copy report values into JSON-compatible primitives."""
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_value(item, invalid_fields, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _diagnostic_value(item, invalid_fields, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _diagnostic_value(value.tolist(), invalid_fields, path)
    if isinstance(value, np.generic):
        return _diagnostic_value(value.item(), invalid_fields, path)
    if isinstance(value, complex):
        if invalid_fields is not None:
            invalid_fields.append(path)
        return None
    if isinstance(value, float) and not np.isfinite(value):
        if invalid_fields is not None:
            invalid_fields.append(path)
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def local_bundle_adjustment(
    state,
    matrix,
    baseline=0.0,
    window=5,
    max_landmarks=200,
    disparity_offset=0.0,
    optimized=True,
    diagnostic_sink=None,
):
    diagnostics_enabled = diagnostic_sink is not None
    diagnostic_errors = [] if diagnostics_enabled else None

    def emit_diagnostic(phase, payload):
        if not diagnostics_enabled:
            return
        try:
            diagnostic_sink(phase, payload)
        except Exception as error:  # Diagnostics must never change BA acceptance.
            diagnostic_errors.append({
                "phase": str(phase),
                "error_type": type(error).__name__,
            })

    def attach_diagnostic_errors(report):
        if diagnostic_errors:
            report["diagnostic_errors"] = [dict(error) for error in diagnostic_errors]
        return report

    with state.lock:
        if len(state.keyframes) < 3:
            return {"applied": False, "reason": "insufficient_keyframes"}
        revision = state.revision
        geometry_revision = state.geometry_revision if diagnostics_enabled else None
        diagnostic_metric = bool(state.metric) if diagnostics_enabled else None
        diagnostic_frame_ids = (
            {int(key): int(value.frame) for key, value in state.keyframes.items()}
            if diagnostics_enabled else None
        )
        selected = list(state.keyframes)[-window:]
        diagnostic_keyframe_rows = {} if diagnostics_enabled else None
        if diagnostics_enabled:
            for keyframe_id in selected:
                keyframe = state.keyframes[keyframe_id]
                diagnostic_keyframe_rows[int(keyframe_id)] = {
                    "frame_id": int(keyframe.frame),
                    "pixels": _diagnostic_array(keyframe.pixels),
                    "landmark_ids": _diagnostic_array(keyframe.landmark_ids),
                }
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
        eligible_landmarks_pre_cap = len(landmarks) if diagnostics_enabled else None
        landmarks = landmarks[:max_landmarks]
        if len(landmarks) < 15:
            return {"applied": False, "reason": "insufficient_observations"}
        diagnostic_selected_landmarks = [] if diagnostics_enabled else None
        diagnostic_selected_ids = [] if diagnostics_enabled else None
        if diagnostics_enabled:
            for rank, landmark in enumerate(landmarks):
                diagnostic_selected_ids.append(int(landmark.id))
                observed_keyframes = [k for k in landmark.observations if k in selected]
                diagnostic_selected_landmarks.append({
                    "rank": int(rank),
                    "landmark_id": int(landmark.id),
                    "anchor_keyframe_id": (
                        None if landmark.anchor is None else int(landmark.anchor)
                    ),
                    "initial_world_position": _diagnostic_array(landmark.position),
                    "observations_in_window": int(len(observed_keyframes)),
                    "latest_observation_keyframe_id": (
                        None if not observed_keyframes else int(max(observed_keyframes))
                    ),
                })
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
        held_out_ids = [] if diagnostics_enabled else None
        if diagnostics_enabled:
            held_out_ids = [
                int(landmark.id)
                for landmark in state.landmarks.values()
                if landmark.id not in optimized_ids and len(landmark.observations) >= 2
                for k, _ in landmark.observations.items()
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
        diagnostic_motion_checks = [] if diagnostics_enabled else None
        for (first, second), measurement in state.stereo_motion.items():
            anchors = (state.pose_anchors[first], state.pose_anchors[second])
            # A shared rigid correction preserves relative motion within one anchor.
            if anchors[0] == anchors[1] or not any(a in free for a in anchors):
                continue
            motion_checks.append((measurement.copy(), anchors,
                                  (state.poses[first].copy(), state.poses[second].copy())))
            if diagnostics_enabled:
                diagnostic_motion_checks.append({
                    "frame_ids": [int(first), int(second)],
                    "measurement": _diagnostic_array(measurement),
                    "anchor_keyframe_ids": [
                        None if anchor is None else int(anchor) for anchor in anchors
                    ],
                    "recorded_camera_to_world": [
                        _diagnostic_array(pose) for pose in (state.poses[first], state.poses[second])
                    ],
                })

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

    def residual(x):
        cameras = camera_poses_for(x)
        points = x[point_offset:].reshape(-1, 3)
        return np.r_[
            optimized_residual(x, cameras, points),
            held_out_residual(cameras),
        ]

    if diagnostics_enabled:
        selected_rows = []
        for row_index, (_point_index, keyframe_id, _observation) in enumerate(records):
            point_index = int(record_points[row_index])
            selected_rows.append({
                "row_index": int(row_index),
                "point_index": point_index,
                "landmark_id": diagnostic_selected_ids[point_index],
                "keyframe_id": int(keyframe_id),
                "pixel": _diagnostic_array(measured_pixels[row_index]),
                "right_u": (
                    None if dimensions[row_index] != 3 else float(measured_right[row_index])
                ),
                "dimensions": int(dimensions[row_index]),
            })
        excluded_rows = []
        for row_index, (_point, _keyframe_index, _observation) in enumerate(held_out):
            keyframe_id = camera_ids[int(held_cameras[row_index])]
            excluded_rows.append({
                "row_index": int(row_index),
                "landmark_id": int(held_out_ids[row_index]),
                "keyframe_id": int(keyframe_id),
                "fixed_world_position": _diagnostic_array(held_points[row_index]),
                "pixel": _diagnostic_array(held_pixels[row_index]),
                "right_u": (
                    None if held_dimensions[row_index] != 3 else float(held_right[row_index])
                ),
                "dimensions": int(held_dimensions[row_index]),
            })
        point_initial = initial[point_offset:].reshape(-1, 3)
        camera_pose_snapshots = []
        for keyframe_id in sorted(base):
            camera_pose_snapshots.append({
                "keyframe_id": int(keyframe_id),
                "frame_id": diagnostic_frame_ids[int(keyframe_id)],
                "camera_to_world": _diagnostic_array(base[keyframe_id]),
            })
        prepared_payload = {
            "schema": "local_bundle_snapshot_v1",
            "revision": int(revision),
            "geometry_revision": int(geometry_revision),
            "metric": diagnostic_metric,
            "calibration": {
                "matrix": _diagnostic_array(matrix),
                "baseline": baseline,
                "disparity_offset": disparity_offset,
            },
            "selection": {
                "window": int(window),
                "max_landmarks": int(max_landmarks),
                "eligible_landmarks_before_cap": int(eligible_landmarks_pre_cap),
                "window_keyframe_ids": [int(key) for key in selected],
                "fixed_keyframe_ids": sorted(int(key) for key in fixed),
                "free_keyframe_ids": sorted(int(key) for key in free),
                "unsupported_keyframe_ids": sorted(int(key) for key in unsupported),
                "components": [sorted(int(key) for key in component)
                               for component in components],
                "selected_landmarks": diagnostic_selected_landmarks,
            },
            "camera_poses": camera_pose_snapshots,
            "keyframe_feature_ownership": [
                {"keyframe_id": int(keyframe_id), **diagnostic_keyframe_rows[int(keyframe_id)]}
                for keyframe_id in selected
            ],
            "parameter_layout": {
                "optimized_camera_mode": bool(optimized),
                "residual_camera_keyframe_ids": [int(key) for key in camera_ids],
                "pose_offsets": {str(key): int(value)
                                 for key, value in pose_offset.items()},
                "point_offset": int(point_offset),
                "initial_vector": _diagnostic_array(initial),
                "initial_selected_world_positions": _diagnostic_array(point_initial),
                "variable_scale": _diagnostic_array(variable_scale),
                "dimensions": _diagnostic_array(dimensions),
                "held_out_dimensions": _diagnostic_array(held_dimensions),
                "dimension_mask": _diagnostic_array(dimension_mask),
                "held_out_dimension_mask": _diagnostic_array(held_dimension_mask),
            },
            "selected_observations": selected_rows,
            "excluded_multiview_observations": excluded_rows,
            "single_view_propagations": [
                {"landmark_id": int(ident),
                 "anchor_keyframe_id": None if anchor is None else int(anchor),
                 "initial_world_position": _diagnostic_array(position)}
                for ident, anchor, position in single_view
            ],
            "motion_checks": diagnostic_motion_checks,
        }
        invalid_prepared_fields = []
        prepared_payload = _diagnostic_value(
            prepared_payload, invalid_prepared_fields, "$.prepared"
        )
        prepared_payload["input_valid"] = not invalid_prepared_fields
        prepared_payload["invalid_fields"] = invalid_prepared_fields
        try:
            emit_diagnostic("prepared", prepared_payload)
        except Exception as error:
            diagnostic_errors.append({
                "phase": "prepared",
                "error_type": type(error).__name__,
            })

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
    result_is_finite = bool(np.isfinite(result.x).all())
    poses = points = None
    if result_is_finite:
        poses, points = unpack(result.x)
    if diagnostics_enabled:
        invalid_report_fields = []
        report_snapshot = _diagnostic_value(report, invalid_report_fields)
        solved_payload = {
            "schema": "local_bundle_solved_v1",
            "revision": int(revision),
            "geometry_revision": int(geometry_revision),
            "result": {
                "x": _diagnostic_vector(result.x),
                "nfev": int(result.nfev),
                "success": bool(result.success),
                "status": int(getattr(result, "status", 0)),
                "message": str(getattr(result, "message", "")),
                "cost": _diagnostic_value(getattr(result, "cost", None)),
                "optimality": _diagnostic_value(getattr(result, "optimality", None)),
            },
            "candidate": {
                "valid": result_is_finite,
                "poses_camera_to_world": (
                    None if poses is None else {
                        str(key): _diagnostic_array(pose)
                        for key, pose in poses.items()
                    }
                ),
                "selected_landmark_world_positions": (
                    None if points is None else [
                        {"landmark_id": diagnostic_selected_ids[index],
                         "position": _diagnostic_array(point)}
                        for index, point in enumerate(points)
                    ]
                ),
            },
            "costs_before_acceptance": _diagnostic_value({
                "selected_initial": before,
                "selected_final": after,
                "held_out_initial": held_before,
            }, invalid_report_fields, "$.costs_before_acceptance"),
            "report_before_acceptance": report_snapshot,
            "invalid_report_fields": invalid_report_fields,
            "acceptance": "pending",
        }
        emit_diagnostic("solved", solved_payload)
    if not result_is_finite:
        return attach_diagnostic_errors(report)
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
        return attach_diagnostic_errors({**report, "reason": "affected_observations_worsened"})
    if any(
        np.any(project(points[i : i + 1], poses[k], matrix)[1] <= 0)
        for i, k, _ in records
    ):
        return attach_diagnostic_errors(report)
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
        return attach_diagnostic_errors({
            **report, "reason": "independent_stereo_motion_inconsistency"
        })
    with state.lock:
        if state.revision != revision:
            return attach_diagnostic_errors({**report, "reason": "stale_revision"})
        updates = {l.id: p.copy() for l, p in zip(landmarks, points)}
        for ident, anchor, position in single_view:
            camera = base[anchor][:3, :3].T @ (position-base[anchor][:3, 3])
            updates[ident] = poses[anchor][:3, :3] @ camera + poses[anchor][:3, 3]
        if not state.apply_corrections(revision, poses, propagate_landmarks=False, landmark_updates=updates):
            return attach_diagnostic_errors(report)
        # Multiview world points are independent; single-view stereo points retain
        # their measured camera coordinates rather than imposing a pose prior.
        report["applied"] = True
    return attach_diagnostic_errors(report)
