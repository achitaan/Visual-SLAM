"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import copy
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


def _training_pixel_key(frame_id, pixel):
    value = np.asarray(pixel, dtype=np.float32)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("invalid_training_pixel")
    key = tuple(0.0 if float(item) == 0.0 else float(item) for item in value)
    return int(frame_id), key


def _training_right_key(value):
    if value is None:
        return None
    result = float(np.float32(value))
    if not np.isfinite(result):
        raise ValueError("invalid_training_right_u")
    return result


def local_bundle_adjustment(
    state,
    matrix,
    baseline=0.0,
    window=5,
    max_landmarks=200,
    disparity_offset=0.0,
    optimized=True,
    diagnostic_sink=None,
    training_factor_provider=None,
    solver_accuracy="default",
    source_history_provider=None,
):
    if solver_accuracy not in ("default", "precise"):
        raise ValueError("solver_accuracy must be 'default' or 'precise'")

    def solver_metadata(effective, inner_options=None):
        return {
            "solver_accuracy_requested": solver_accuracy,
            "solver_accuracy_effective": effective,
            "solver_inner_options": (
                None if inner_options is None else dict(inner_options)
            ),
        }

    diagnostics_enabled = diagnostic_sink is not None
    provider_enabled = training_factor_provider is not None
    source_history_enabled = source_history_provider is not None
    snapshot_enabled = diagnostics_enabled or provider_enabled
    diagnostic_errors = [] if diagnostics_enabled else None
    training_bundle_summary = None
    source_history_rows = None
    source_history_source_frame = None
    source_history_summary = None
    source_history_validation_payload = None

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
        if training_bundle_summary is not None:
            if source_history_enabled and source_history_summary is not None:
                if (isinstance(report, dict) and report.get("reason") is not None
                        and source_history_summary.get("status") == "active"):
                    source_history_summary.update(
                        status="rejected", reason=str(report.get("reason"))
                    )
                training_bundle_summary["source_history_bundle"] = _diagnostic_value(
                    source_history_summary
                )
            if (isinstance(report, dict) and report.get("reason") is not None
                    and training_bundle_summary.get("status") == "active"):
                training_bundle_summary.update(
                    status="rejected", reason=str(report.get("reason"))
                )
            report["owned_stereo_image_bundle"] = _diagnostic_value(
                training_bundle_summary
            )
        if diagnostic_errors:
            report["diagnostic_errors"] = [dict(error) for error in diagnostic_errors]
        return report

    with state.lock:
        if len(state.keyframes) < 3:
            return {"applied": False, "reason": "insufficient_keyframes",
                    **solver_metadata("not_run")}
        revision = state.revision
        geometry_revision = state.geometry_revision if snapshot_enabled else None
        diagnostic_metric = bool(state.metric) if snapshot_enabled else None
        diagnostic_frame_ids = (
            {int(key): int(value.frame) for key, value in state.keyframes.items()}
            if snapshot_enabled else None
        )
        frame_status_snapshot = list(state.statuses) if snapshot_enabled else None
        frame_pose_snapshot = (
            [np.asarray(pose).copy() for pose in state.poses]
            if provider_enabled else None
        )
        frame_anchor_snapshot = list(state.pose_anchors) if provider_enabled else None
        stereo_motion_snapshot = (
            {edge: measurement.copy() for edge, measurement in state.stereo_motion.items()}
            if provider_enabled else None
        )
        keyframe_size_snapshot = (
            {int(key): (None if frame.image_size is None else tuple(frame.image_size))
             for key, frame in state.keyframes.items()}
            if provider_enabled else None
        )
        selected = list(state.keyframes)[-window:]
        diagnostic_keyframe_rows = {} if snapshot_enabled else None
        if snapshot_enabled:
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
        eligible_landmarks_pre_cap = len(landmarks) if snapshot_enabled else None
        landmarks = landmarks[:max_landmarks]
        if len(landmarks) < 15:
            return {"applied": False, "reason": "insufficient_observations",
                    **solver_metadata("not_run")}
        diagnostic_selected_landmarks = [] if snapshot_enabled else None
        diagnostic_selected_ids = [] if snapshot_enabled else None
        if snapshot_enabled:
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
            return {"applied": False, "reason": "no_anchored_free_cameras",
                    **solver_metadata("not_run")}
        optimized_ids = {landmark.id for landmark in landmarks}
        held_out = [
            (landmark.position.copy(), k, o)
            for landmark in state.landmarks.values()
            if landmark.id not in optimized_ids and len(landmark.observations) >= 2
            for k, o in landmark.observations.items()
            if k in free
        ]
        held_out_ids = [] if snapshot_enabled else None
        if snapshot_enabled:
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
        single_view_observation_rows = [] if provider_enabled else None
        if provider_enabled:
            for ident, anchor, _position in single_view:
                landmark = state.landmarks[ident]
                for keyframe_id, observation in landmark.observations.items():
                    if keyframe_id not in base:
                        continue
                    single_view_observation_rows.append({
                        "landmark_id": int(ident),
                        "keyframe_id": int(keyframe_id),
                        "frame_id": int(state.keyframes[keyframe_id].frame),
                        "anchor_keyframe_id": int(anchor),
                        "pixel": _diagnostic_array(observation.pixel),
                        "right_u": (None if observation.right_u is None
                                    else float(observation.right_u)),
                    })
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
        diagnostic_motion_checks = [] if snapshot_enabled else None
        for (first, second), measurement in state.stereo_motion.items():
            anchors = (state.pose_anchors[first], state.pose_anchors[second])
            # A shared rigid correction preserves relative motion within one anchor.
            if anchors[0] == anchors[1] or not any(a in free for a in anchors):
                continue
            motion_checks.append((measurement.copy(), anchors,
                                  (state.poses[first].copy(), state.poses[second].copy())))
            if snapshot_enabled:
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

    point_limit = len(initial)
    intermediate_pose_offsets = {}
    intermediate_reference_keyframes = {}
    target_relative_point_offsets = {}
    target_relative_point_reference_keyframes = {}
    extra_index = {}
    extra_ids = []
    frame_to_keyframe = (
        {frame: key for key, frame in diagnostic_frame_ids.items()}
        if provider_enabled else {}
    )

    def unpack(x):
        poses = {k: p.copy() for k, p in base.items()}
        for k, offset in pose_offset.items():
            poses[k][:3, :3] = Rotation.from_rotvec(x[offset : offset + 3]).as_matrix()
            poses[k][:3, 3] = x[offset + 3 : offset + 6]
        vector_points = x[point_offset:point_limit].reshape(-1, 3)
        if not target_relative_point_offsets:
            return poses, vector_points
        points = vector_points.copy()
        for landmark_id, offset in target_relative_point_offsets.items():
            reference_keyframe = target_relative_point_reference_keyframes[landmark_id]
            target_pose = poses[reference_keyframe]
            q = x[offset:offset + 3]
            points[extra_index[landmark_id]] = target_pose[:3, :3] @ q + target_pose[:3, 3]
        return poses, points

    def frame_pose_for(x, frame_id, poses):
        """One camera accessor for image residuals, guards and atomic commit."""
        if frame_id in frame_to_keyframe:
            return poses[frame_to_keyframe[frame_id]]
        offset = intermediate_pose_offsets.get(frame_id)
        if offset is not None:
            relative = np.eye(4)
            relative[:3, :3] = Rotation.from_rotvec(x[offset:offset + 3]).as_matrix()
            relative[:3, 3] = x[offset + 3:offset + 6]
            reference_keyframe = intermediate_reference_keyframes[frame_id]
            return poses[reference_keyframe] @ relative
        pose = frame_pose_snapshot[frame_id]
        anchor = frame_anchor_snapshot[frame_id]
        return (poses[anchor] @ np.linalg.inv(base[anchor]) @ pose
                if anchor is not None else pose.copy())

    def training_endpoint_poses(x, poses):
        frame_ids = owned_training_rows['frame_ids']
        cameras = {int(frame): frame_pose_for(x, int(frame), poses)
                   for frame in np.unique(frame_ids)}
        return np.asarray([cameras[int(frame)] for frame in frame_ids])

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

    owned_training_rows = None

    def owned_training_residual(x):
        rows = owned_training_rows
        if rows is None or not len(rows["point_indices"]):
            return np.empty(0)
        poses, all_points = unpack(x)
        endpoint_poses = training_endpoint_poses(x, poses)
        world = all_points[rows["point_indices"]]
        delta = world - endpoint_poses[:, :3, 3]
        camera_points = np.einsum("ni,nij->nj", delta, endpoint_poses[:, :3, :3])
        # Eligible singleton tracks use q=T^-1 P and C=T^-1 S directly. This
        # makes target-row residuals depend only on q and source-row residuals
        # only on (C,q), without numerically cancelling a free T perturbation.
        singleton_rows = rows["target_relative_point_mask"]
        if np.any(singleton_rows):
            row_ids = np.flatnonzero(singleton_rows)
            q_values = np.empty((len(row_ids), 3), dtype=float)
            point_indices = rows["point_indices"][row_ids]
            for index, point_index in enumerate(point_indices):
                q_offset = point_offset + 3 * int(point_index)
                q_values[index] = x[q_offset:q_offset + 3]
            target_rows = rows["row_is_target"][row_ids]
            camera_points[row_ids[target_rows]] = q_values[target_rows]
            source_row_ids = row_ids[~target_rows]
            source_q = q_values[~target_rows]
            for frame_id in np.unique(rows["frame_ids"][source_row_ids]):
                selected_rows = source_row_ids[rows["frame_ids"][source_row_ids] == frame_id]
                offset = intermediate_pose_offsets[int(frame_id)]
                relative_rotation = Rotation.from_rotvec(x[offset:offset + 3]).as_matrix()
                relative_translation = x[offset + 3:offset + 6]
                local = source_q[rows["frame_ids"][source_row_ids] == frame_id]
                camera_points[selected_rows] = (local - relative_translation) @ relative_rotation
        z = camera_points[:, 2]
        homogeneous = camera_points @ matrix.T
        projected = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        predicted_right = projected[:, 0] - matrix[0, 0] * baseline / np.maximum(z, 1e-9) - disparity_offset
        errors = np.column_stack((projected - rows["pixels"], predicted_right - rows["right_u"]))
        errors[:, :2] = np.clip(errors[:, :2], -1e4, 1e4)
        errors[z <= 0.] = 1e4
        return errors.ravel()

    def source_history_residual(x):
        rows = source_history_rows
        if not rows:
            return np.empty(0)
        poses, all_points = unpack(x)
        points = all_points[np.asarray([item["point_index"] for item in rows], dtype=np.intp)]
        cameras = np.asarray([
            frame_pose_for(x, int(item["frame_id"]), poses) for item in rows
        ], dtype=float)
        camera_points = np.einsum(
            "ni,nij->nj", points-cameras[:, :3, 3], cameras[:, :3, :3]
        )
        z = camera_points[:, 2]
        homogeneous = camera_points @ matrix.T
        projected = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        pixels = np.asarray([item["pixel"] for item in rows], dtype=float)
        errors = np.clip(projected-pixels, -1e4, 1e4)
        errors[z <= 0.] = 1e4
        return errors.ravel()

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
            points = x[point_offset:point_limit].reshape(-1, 3)
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
        points = x[point_offset:point_limit].reshape(-1, 3)
        original = np.r_[
            optimized_residual(x, cameras, points),
            held_out_residual(cameras),
        ]
        if owned_training_rows is None:
            return original
        values = np.r_[original, owned_training_residual(x)]
        if source_history_rows:
            values = np.r_[values, source_history_residual(x)]
        return values

    prepared_payload = None
    if snapshot_enabled:
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
        if provider_enabled:
            prepared_payload["frame_statuses"] = [str(value) for value in frame_status_snapshot]
            prepared_payload["single_view_observation_rows"] = single_view_observation_rows
        invalid_prepared_fields = []
        prepared_payload = _diagnostic_value(
            prepared_payload, invalid_prepared_fields, "$.prepared"
        )
        prepared_payload["input_valid"] = not invalid_prepared_fields
        prepared_payload["invalid_fields"] = invalid_prepared_fields
    original_variable_count = len(initial)
    original_pattern = pattern
    original_initial = initial.copy()
    original_scale = variable_scale.copy()
    if provider_enabled:
        training_bundle_summary = {
            "status": "rejected", "reason": "provider_not_evaluated",
            "model": "correlated_physical_stereo_image_rows_v1",
            "covariance_claim": False,
            "intentionally_reuses_sensor_evidence": True,
            "holdout_validation_claim": False,
            "selected_factors": 0,
        }
        if source_history_enabled:
            source_history_summary = {
                "requested": True,
                "status": "skipped",
                "reason": "owned_factor_pool_not_active",
                "captured_rows": 0,
                "candidate_rows": 0,
                "installed_rows": 0,
                "installed_landmark_ids": [],
                "exclusions": {},
                "duplicate_rows_collapsed": 0,
                "components_added": 0,
                "cost_before": 0.0,
                "cost_after": 0.0,
                "tracking_fit_evidence_reused": True,
                "covariance_claim": False,
                "heldout_validation_claim": False,
            }
        try:
            provider_payload = copy.deepcopy(prepared_payload)
            provider_payload["training_factor_phase"] = "prepare"
            factors, provider_report = training_factor_provider(provider_payload)
            provider_report = _diagnostic_value(provider_report if isinstance(provider_report, dict) else {})
            training_bundle_summary.update(provider_report)
            if not factors:
                training_bundle_summary.setdefault("reason", "no_eligible_factors")
            elif not diagnostic_metric or baseline <= 0.:
                training_bundle_summary.update(status="rejected", reason="metric_stereo_required")
            else:
                from stereo_training_factors import StereoTrainingFactor
                if len(frame_to_keyframe) != len(diagnostic_frame_ids):
                    raise ValueError('ambiguous_keyframe_image_binding')

                selected_by_id = {int(item["landmark_id"]): int(item["rank"])
                                  for item in diagnostic_selected_landmarks}
                single_by_id = {int(ident): (anchor, position.copy())
                                for ident, anchor, position in single_view}
                # An image row already present in the ordinary BA objective is reused
                # exactly once. Conflicting row ownership rejects the complete pool.
                original_physical = {}
                original_pixel_only = {}
                for row_index, (_point_index, keyframe_id, _observation) in enumerate(records):
                    fid = int(diagnostic_frame_ids[int(keyframe_id)])
                    key = _training_pixel_key(fid, measured_pixels[row_index])
                    right_value = (
                        _training_right_key(measured_right[row_index])
                        if dimensions[row_index] == 3 else None
                    )
                    value = (int(record_points[row_index]), right_value)
                    prior = original_pixel_only.get(key)
                    if prior is not None and prior != value:
                        raise ValueError("ambiguous_original_physical_observation")
                    original_pixel_only[key] = value
                    if value[1] is not None:
                        original_physical[(key, value[1])] = value[0]
                single_physical = {}
                single_pixel_only = {}
                for item in single_view_observation_rows:
                    fid = int(item["frame_id"])
                    key = _training_pixel_key(fid, item["pixel"])
                    point_id = int(item["landmark_id"])
                    right_key = _training_right_key(item["right_u"])
                    value = (point_id, right_key)
                    prior = single_pixel_only.get(key)
                    if prior is not None and prior != value:
                        raise ValueError("ambiguous_single_view_physical_observation")
                    single_pixel_only[key] = value
                    single_physical[(key, right_key)] = point_id

                used_single_ids = set()
                candidate_rows = []
                seen_rows = {}
                factor_ids = set()
                singleton_reference_keyframes = {}
                source_reference_keyframes = {}
                single_observations_by_id = {}
                for item in single_view_observation_rows:
                    single_observations_by_id.setdefault(int(item["landmark_id"]), []).append(item)
                for factor in factors:
                    if not isinstance(factor, StereoTrainingFactor):
                        raise ValueError("invalid_training_factor_type")
                    if (factor.fit_partition != "reserved_training"
                            or factor.heldout_status != "excluded_external_arbitration_rows"
                            or factor.heldout_validation_claim is not False
                            or factor.covariance_claim is not False
                            or factor.model != "correlated_physical_stereo_image_rows_v1"):
                        raise ValueError("invalid_training_factor_provenance")
                    if (not np.array_equal(factor.matrix, matrix)
                            or factor.baseline != float(baseline)
                            or factor.disparity_offset != float(disparity_offset)):
                        raise ValueError("training_factor_calibration_mismatch")
                    epoch = tuple(int(value) for value in factor.source_epoch)
                    if (len(epoch) != 2 or epoch[1] != int(geometry_revision)
                            or epoch[0] > int(revision)):
                        raise ValueError("training_factor_source_epoch_mismatch")
                    if any(frame >= min(len(frame_pose_snapshot), len(frame_status_snapshot),
                                        len(frame_anchor_snapshot))
                           for frame in (factor.source_frame, factor.target_frame)):
                        raise ValueError('training_factor_frame_snapshot_missing')
                    target_keyframe = frame_to_keyframe.get(int(factor.target_frame))
                    if target_keyframe is None:
                        raise ValueError("target_relative_reference_must_be_keyframe")
                    for frame_id, anchor_id, frozen_pose, frozen_anchor, status in (
                        (factor.source_frame, factor.source_anchor, factor.source_pose,
                         factor.source_anchor_pose, frame_status_snapshot[factor.source_frame]),
                        (factor.target_frame, factor.target_anchor, factor.target_pose,
                         factor.target_anchor_pose, frame_status_snapshot[factor.target_frame]),
                    ):
                        if (frame_id >= len(frame_pose_snapshot)
                                or status not in {"tracking", "relocalized", "accepted"}
                                or anchor_id not in base
                                or not np.array_equal(frame_pose_snapshot[frame_id], frozen_pose)
                                or not np.array_equal(base[anchor_id], frozen_anchor)):
                            raise ValueError("training_factor_endpoint_snapshot_mismatch")
                    if (not np.allclose(
                            np.linalg.inv(factor.source_anchor_pose) @ factor.source_pose,
                            factor.source_anchor_relative, rtol=0., atol=1e-9)
                            or not np.allclose(
                                np.linalg.inv(factor.target_anchor_pose) @ factor.target_pose,
                                factor.target_anchor_relative, rtol=0., atol=1e-9)):
                        raise ValueError("training_factor_anchor_transport_mismatch")
                    ident = int(factor.reused_landmark_id)
                    if ident < 0 or ident not in selected_by_id and ident not in single_by_id:
                        raise ValueError("training_factor_landmark_not_optimized")
                    if any(value >= 0 and int(value) != ident for value in
                           (factor.source_landmark_id, factor.target_landmark_id)):
                        raise ValueError("training_factor_endpoint_landmark_mismatch")
                    if ident in selected_by_id:
                        point_index = selected_by_id[ident]
                        point_initial = initial[point_offset + 3 * point_index:point_offset + 3 * point_index + 3]
                    else:
                        point_index = None
                        point_initial = single_by_id[ident][1]
                        used_single_ids.add(ident)
                        observations = single_observations_by_id.get(ident, [])
                        if (len(observations) != 1
                                or single_by_id[ident][0] != target_keyframe
                                or int(observations[0]["keyframe_id"]) != target_keyframe
                                or int(observations[0]["frame_id"]) != int(factor.target_frame)):
                            raise ValueError("singleton_has_non_target_observation")
                        if int(factor.source_frame) in frame_to_keyframe:
                            raise ValueError("unsupported_target_relative_singleton_source")
                        prior_reference = singleton_reference_keyframes.get(ident)
                        if prior_reference is not None and prior_reference != target_keyframe:
                            raise ValueError("ambiguous_target_relative_singleton_reference")
                        singleton_reference_keyframes[ident] = target_keyframe
                    if int(factor.source_frame) not in frame_to_keyframe:
                        prior_reference = source_reference_keyframes.get(int(factor.source_frame))
                        if prior_reference is not None and prior_reference != target_keyframe:
                            raise ValueError("conflicting_target_relative_source_reference")
                        source_reference_keyframes[int(factor.source_frame)] = target_keyframe
                    if not np.array_equal(point_initial, factor.point_initial):
                        raise ValueError("training_factor_point_snapshot_mismatch")
                    factor_ids.add(ident)
                    for role, frame_id, anchor_id, relative, pixel, right_value in (
                        ("source", factor.source_frame, factor.source_anchor,
                         factor.source_anchor_relative, factor.source_pixel, factor.source_right_u),
                        ("target", factor.target_frame, factor.target_anchor,
                         factor.target_anchor_relative, factor.target_pixel, factor.target_right_u),
                    ):
                        if frame_id in frame_to_keyframe:
                            actual_keyframe = frame_to_keyframe[frame_id]
                            if (actual_keyframe != anchor_id
                                    or not np.allclose(relative, np.eye(4), rtol=0., atol=1e-9)):
                                raise ValueError('training_factor_keyframe_camera_mismatch')
                        key = _training_pixel_key(frame_id, pixel)
                        right_key = _training_right_key(right_value)
                        physical = (key, right_key)
                        if key in original_pixel_only and original_pixel_only[key][1] != right_key:
                            raise ValueError("training_factor_conflicts_with_original_observation")
                        if physical in original_physical:
                            original_id = diagnostic_selected_ids[original_physical[physical]]
                            if original_id != ident:
                                raise ValueError("training_factor_original_owner_mismatch")
                            continue
                        if key in single_pixel_only and single_pixel_only[key][1] != right_key:
                            raise ValueError("training_factor_conflicts_with_single_view_observation")
                        if physical in single_physical and single_physical[physical] != ident:
                            raise ValueError("training_factor_single_view_owner_mismatch")
                        if key not in single_pixel_only and (frame_id == factor.source_frame and factor.source_has_existing_observation
                                                             or frame_id == factor.target_frame and factor.target_has_existing_observation):
                            # An existing owner must be one of the exact selected or singleton rows.
                            raise ValueError("training_factor_existing_observation_not_in_bundle")
                        row_key = (frame_id, tuple(key[1]), right_key)
                        value = (ident, int(anchor_id), tuple(np.asarray(relative).ravel().tolist()),
                                 int(target_keyframe))
                        if row_key in seen_rows:
                            if seen_rows[row_key] != value:
                                raise ValueError("ambiguous_training_physical_row")
                            continue
                        seen_rows[row_key] = value
                        candidate_rows.append({
                            "ident": ident,
                            "anchor_id": int(anchor_id),
                            "relative_pose": np.asarray(relative).copy(),
                            "pixel": np.asarray(pixel, dtype=float).copy(),
                            "right_u": float(right_value),
                            "frame_id": int(frame_id),
                            "role": role,
                            "reference_keyframe_id": int(target_keyframe),
                        })
                if not candidate_rows:
                    raise ValueError("no_new_owned_image_rows")
                # Optional left-only observations from the actual accepted
                # previous-frame track cache. This adds no pose or point
                # variables: each row reuses one selected multiview point and
                # the source camera already introduced by the owned factors.
                if source_history_enabled:
                    source_history_summary.update(
                        status="skipped", reason="source_history_not_eligible",
                        candidate_rows=0, installed_rows=0,
                        installed_landmark_ids=[], components_added=0,
                    )
                    source_frames = {int(factor.source_frame) for factor in factors}
                    if len(source_frames) == 1:
                        source_history_source_frame = next(iter(source_frames))
                        source_chart_available = (
                            source_history_source_frame in frame_to_keyframe
                            or source_history_source_frame in source_reference_keyframes
                        )
                        if source_chart_available:
                            original_frame_ids = sorted(
                                int(diagnostic_selected_ids[int(point_index)])
                                for point_index, keyframe_id, _observation in records
                                if int(diagnostic_frame_ids[int(keyframe_id)])
                                == source_history_source_frame
                            )
                            owned_frame_ids = sorted(
                                int(row["ident"]) for row in candidate_rows
                                if int(row["frame_id"]) == source_history_source_frame
                            )
                            history_payload = copy.deepcopy(prepared_payload)
                            history_payload.update({
                                "source_history_phase": "prepare",
                                "source_history_source_frame": int(source_history_source_frame),
                                "source_history_selected_landmark_ids": sorted(selected_by_id),
                                "source_history_original_frame_landmark_ids": original_frame_ids,
                                "source_history_owned_frame_landmark_ids": owned_frame_ids,
                            })
                            try:
                                proposed_rows, history_report = source_history_provider(history_payload)
                                if not isinstance(history_report, dict):
                                    history_report = {"status": "rejected",
                                                      "reason": "malformed_source_history_report"}
                                source_history_summary.update(_diagnostic_value(history_report))
                                source_history_summary["requested"] = True
                                source_history_summary["tracking_fit_evidence_reused"] = True
                                source_history_summary["covariance_claim"] = False
                                source_history_summary["heldout_validation_claim"] = False
                                if history_report.get("status") not in ("captured", "eligible", "prepared"):
                                    proposed_rows = ()
                                    source_history_summary.update(
                                        status="skipped",
                                        reason=history_report.get("reason", "source_history_not_eligible"),
                                    )
                                else:
                                    source_history_validation_payload = copy.deepcopy(history_payload)
                                    source_history_validation_payload["source_history_phase"] = "validate"
                            except Exception as error:
                                proposed_rows = ()
                                source_history_summary.update(
                                    status="skipped",
                                    reason=f"source_history_provider_{type(error).__name__}",
                                )
                            if isinstance(proposed_rows, (list, tuple)):
                                exclusions = dict(source_history_summary.get("exclusions", {}))
                                existing_source = {
                                    (source_history_source_frame, ident)
                                    for ident in original_frame_ids + owned_frame_ids
                                }
                                existing_source_pixels = {}
                                for row_index, (point_index, keyframe_id, observation) in enumerate(records):
                                    frame_id = int(diagnostic_frame_ids[int(keyframe_id)])
                                    if frame_id != source_history_source_frame:
                                        continue
                                    ident = int(diagnostic_selected_ids[int(record_points[row_index])])
                                    try:
                                        key = _training_pixel_key(frame_id, observation.pixel)[1]
                                    except Exception:
                                        continue
                                    existing_source_pixels.setdefault(key, set()).add(ident)
                                for owned_row in candidate_rows:
                                    if int(owned_row["frame_id"]) != source_history_source_frame:
                                        continue
                                    try:
                                        key = _training_pixel_key(
                                            source_history_source_frame, owned_row["pixel"]
                                        )[1]
                                    except Exception:
                                        continue
                                    existing_source_pixels.setdefault(key, set()).add(
                                        int(owned_row["ident"])
                                    )
                                grouped = []
                                source_conflict = False
                                for item in proposed_rows:
                                    try:
                                        if not isinstance(item, dict):
                                            raise ValueError("malformed_row")
                                        raw_frame_id = item["frame_id"]
                                        raw_ident = item["landmark_id"]
                                        if (not isinstance(raw_frame_id, (int, np.integer))
                                                or isinstance(raw_frame_id, (bool, np.bool_))
                                                or not isinstance(raw_ident, (int, np.integer))
                                                or isinstance(raw_ident, (bool, np.bool_))):
                                            raise ValueError("noninteger_source_identity")
                                        frame_id = int(raw_frame_id)
                                        ident = int(raw_ident)
                                        raw_pixel = np.asarray(item["pixel"])
                                        if (np.iscomplexobj(raw_pixel)
                                                or not np.issubdtype(raw_pixel.dtype, np.number)):
                                            raise ValueError("nonreal_source_pixel")
                                        pixel = np.asarray(raw_pixel, dtype=np.float32)
                                        key = _training_pixel_key(frame_id, pixel)
                                        if frame_id != source_history_source_frame:
                                            raise ValueError("source_frame_mismatch")
                                        if ident not in selected_by_id:
                                            exclusions["not_selected_multiview"] = exclusions.get(
                                                "not_selected_multiview", 0) + 1
                                            continue
                                        if (frame_id, ident) in existing_source:
                                            exclusions["same_frame_landmark_already_owned"] = exclusions.get(
                                                "same_frame_landmark_already_owned", 0) + 1
                                            continue
                                        physical_owners = existing_source_pixels.get(key[1], set())
                                        if physical_owners and ident not in physical_owners:
                                            source_conflict = True
                                            exclusions["physical_pixel_conflicts_with_existing_owner"] = exclusions.get(
                                                "physical_pixel_conflicts_with_existing_owner", 0) + 1
                                            continue
                                        grouped.append({"frame_id": frame_id,
                                                        "landmark_id": ident,
                                                        "point_index": selected_by_id[ident],
                                                        "pixel": pixel.copy(),
                                                        "pixel_key": key[1]})
                                    except Exception:
                                        exclusions["invalid_or_mismatched_row"] = exclusions.get(
                                            "invalid_or_mismatched_row", 0) + 1
                                id_pixels = {}
                                pixel_ids = {}
                                for row_item in grouped:
                                    id_pixels.setdefault(row_item["landmark_id"], set()).add(
                                        row_item["pixel_key"])
                                    pixel_ids.setdefault(row_item["pixel_key"], set()).add(
                                        row_item["landmark_id"])
                                conflicting_ids = {
                                    ident for ident, keys in id_pixels.items() if len(keys) > 1
                                }
                                conflicting_pixels = {
                                    pixel for pixel, ids in pixel_ids.items() if len(ids) > 1
                                }
                                if source_conflict:
                                    # Fail closed for the history sub-pool while
                                    # preserving independently valid owned factors.
                                    grouped = []
                                    id_pixels = {}
                                    pixel_ids = {}
                                    conflicting_ids = set()
                                    conflicting_pixels = set()
                                    exclusions["source_physical_ownership_conflict_fail_closed"] = 1
                                    source_history_summary["reason"] = (
                                        "source_physical_ownership_conflict"
                                    )
                                if conflicting_ids:
                                    exclusions["landmark_multiple_source_pixels"] = len(conflicting_ids)
                                if conflicting_pixels:
                                    exclusions["physical_pixel_multiple_landmarks"] = len(conflicting_pixels)
                                unique_history = []
                                seen_history = set()
                                for row_item in grouped:
                                    pair = (row_item["landmark_id"], row_item["pixel_key"])
                                    if (row_item["landmark_id"] in conflicting_ids
                                            or row_item["pixel_key"] in conflicting_pixels):
                                        continue
                                    if pair in seen_history:
                                        source_history_summary["duplicate_rows_collapsed"] = int(
                                            source_history_summary.get("duplicate_rows_collapsed", 0)) + 1
                                        continue
                                    seen_history.add(pair)
                                    unique_history.append(row_item)
                                source_history_rows = unique_history
                                source_history_summary.update({
                                    "source_frame": int(source_history_source_frame),
                                    "candidate_rows": int(len(grouped)),
                                    "installed_rows": int(len(unique_history)),
                                    "installed_landmark_ids": sorted({
                                        int(item["landmark_id"]) for item in unique_history
                                    }),
                                    "exclusions": exclusions,
                                    "components_added": int(2 * len(unique_history)),
                                    "status": "active" if unique_history else "skipped",
                                    "reason": None if unique_history else
                                        source_history_summary.get("reason", "no_eligible_history_rows"),
                                })
                    else:
                        source_history_summary.update(
                            reason="ambiguous_owned_source_frames",
                            source_frame_candidates=sorted(source_frames),
                        )
                extra_ids = sorted(used_single_ids)
                extra_index = {ident: len(landmarks) + i for i, ident in enumerate(extra_ids)}
                # Rebind singleton rows after deterministic point-index assignment.
                old_rows = int(np.sum(dimensions) + np.sum(held_dimensions))
                rows = {
                    "point_indices": np.asarray([
                        (selected_by_id[row["ident"]] if row["ident"] in selected_by_id
                         else extra_index[row["ident"]])
                        for row in candidate_rows], dtype=np.intp),
                    "anchor_ids": np.asarray([row["anchor_id"] for row in candidate_rows], dtype=np.intp),
                    "relative_poses": np.asarray([row["relative_pose"] for row in candidate_rows], dtype=float),
                    "pixels": np.asarray([row["pixel"] for row in candidate_rows], dtype=float),
                    "right_u": np.asarray([row["right_u"] for row in candidate_rows], dtype=float),
                    "frame_ids": np.asarray([row["frame_id"] for row in candidate_rows], dtype=np.intp),
                    "row_is_target": np.asarray([row["role"] == "target" for row in candidate_rows], dtype=bool),
                    "reference_keyframe_ids": np.asarray(
                        [row["reference_keyframe_id"] for row in candidate_rows], dtype=np.intp),
                    "target_relative_point_mask": np.asarray(
                        [row["ident"] in single_by_id for row in candidate_rows], dtype=bool),
                }
                target_relative_point_reference_keyframes = dict(singleton_reference_keyframes)
                extra_initial = np.asarray([
                    base[target_relative_point_reference_keyframes[ident]][:3, :3].T
                    @ (single_by_id[ident][1] - base[target_relative_point_reference_keyframes[ident]][:3, 3])
                    for ident in extra_ids
                ], dtype=float).reshape(-1)
                if len(extra_initial):
                    augmented_initial = np.r_[initial, extra_initial]
                    augmented_scale = np.r_[variable_scale, np.full(len(extra_initial), length_scale)]
                else:
                    augmented_initial = initial
                    augmented_scale = variable_scale
                # A non-keyframe image is a camera variable, not its anchor
                # multiplied by a supposedly exact accumulated motion. Reuse
                # true keyframe variables and optimize intermediate poses freely.
                intermediate_frames = sorted(source_reference_keyframes)
                intermediate_reference_keyframes = dict(source_reference_keyframes)
                # Added cameras/points must connect to the already anchored
                # visual graph. Never hide an orphan SE(3) gauge by attaching a
                # camera to its recorded anchor or adding an artificial prior.
                image_neighbors = {}
                for row in candidate_rows:
                    ident, frame_id = row["ident"], row["frame_id"]
                    camera_node, point_node = ('frame', frame_id), ('point', ident)
                    image_neighbors.setdefault(camera_node, set()).add(point_node)
                    image_neighbors.setdefault(point_node, set()).add(camera_node)
                reached = {node for node in image_neighbors
                           if node[0] == 'frame' and node[1] in frame_to_keyframe
                           or node[0] == 'point' and node[1] in selected_by_id}
                frontier = list(reached)
                while frontier:
                    for neighbor in image_neighbors[frontier.pop()] - reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
                if any(frame <= 0 or ('frame', frame) not in reached
                       for frame in intermediate_frames):
                    raise ValueError('unanchored_intermediate_camera_component')
                intermediate_offsets = {}
                augmented_point_limit = len(augmented_initial)
                for frame_id in intermediate_frames:
                    reference_keyframe = intermediate_reference_keyframes[frame_id]
                    relative_pose = np.linalg.inv(base[reference_keyframe]) @ frame_pose_snapshot[frame_id]
                    intermediate_offsets[frame_id] = len(augmented_initial)
                    augmented_initial = np.r_[augmented_initial,
                        Rotation.from_matrix(relative_pose[:3, :3]).as_rotvec(), relative_pose[:3, 3]]
                    augmented_scale = np.r_[augmented_scale,
                        1., 1., 1., length_scale, length_scale, length_scale]
                history_count = len(source_history_rows or [])
                augmented_pattern = lil_matrix((
                    old_rows + 3 * len(candidate_rows) + 2 * history_count,
                    len(augmented_initial),
                ), dtype=int)
                augmented_pattern[:old_rows, :original_variable_count] = original_pattern
                row = old_rows
                target_relative_point_offsets = {
                    ident: point_offset + 3 * extra_index[ident]
                    for ident in extra_ids
                }
                for candidate_index, (point_index, frame_id) in enumerate(
                        zip(rows["point_indices"], rows["frame_ids"])):
                    is_singleton = bool(rows["target_relative_point_mask"][candidate_index])
                    is_target = bool(rows["row_is_target"][candidate_index])
                    reference_keyframe = int(rows["reference_keyframe_ids"][candidate_index])
                    if is_singleton:
                        # A target singleton row is h(q); its free target pose
                        # has exactly no dependency. Its source row is h(C^-1q).
                        if not is_target:
                            camera_offset = intermediate_offsets.get(int(frame_id))
                            if camera_offset is None:
                                raise ValueError("unsupported_target_relative_singleton_source")
                            augmented_pattern[row:row + 3, camera_offset:camera_offset + 6] = 1
                    elif int(frame_id) in frame_to_keyframe:
                        camera_offset = pose_offset.get(frame_to_keyframe[int(frame_id)])
                        if camera_offset is not None:
                            augmented_pattern[row:row + 3, camera_offset:camera_offset + 6] = 1
                    else:
                        # Selected shared world points preserve all three
                        # dependencies: reference T, free relative camera C, P.
                        camera_offset = intermediate_offsets.get(int(frame_id))
                        if camera_offset is None:
                            raise ValueError("unsupported_target_relative_shared_source")
                        augmented_pattern[row:row + 3, camera_offset:camera_offset + 6] = 1
                        reference_offset = pose_offset.get(reference_keyframe)
                        if reference_offset is not None:
                            augmented_pattern[row:row + 3, reference_offset:reference_offset + 6] = 1
                    start = point_offset + 3 * int(point_index)
                    augmented_pattern[row:row + 3, start:start + 3] = 1
                    row += 3
                for history_row in source_history_rows or []:
                    frame_id = int(history_row["frame_id"])
                    if frame_id in frame_to_keyframe:
                        camera_offset = pose_offset.get(frame_to_keyframe[frame_id])
                        if camera_offset is not None:
                            augmented_pattern[row:row + 2, camera_offset:camera_offset + 6] = 1
                    else:
                        camera_offset = intermediate_offsets.get(frame_id)
                        if camera_offset is None:
                            raise ValueError("source_history_camera_not_in_owned_bundle_chart")
                        augmented_pattern[row:row + 2, camera_offset:camera_offset + 6] = 1
                        reference_keyframe = intermediate_reference_keyframes[frame_id]
                        reference_offset = pose_offset.get(reference_keyframe)
                        if reference_offset is not None:
                            augmented_pattern[row:row + 2, reference_offset:reference_offset + 6] = 1
                    point_start = point_offset + 3 * int(history_row["point_index"])
                    augmented_pattern[row:row + 2, point_start:point_start + 3] = 1
                    row += 2
                for ident, point_index in extra_index.items():
                    target_relative_point_offsets[ident] = point_offset + 3 * point_index
                initial = augmented_initial
                variable_scale = augmented_scale
                pattern = augmented_pattern
                owned_training_rows = rows
                point_limit = augmented_point_limit
                intermediate_pose_offsets = intermediate_offsets
                training_bundle_summary.update({
                    "status": "active", "reason": None,
                    "selected_factors": int(len(factors)),
                    "unique_image_rows_added": int(len(candidate_rows)),
                    "reused_selected_points": int(sum(i in selected_by_id for i in factor_ids)),
                    "optimized_single_view_points": int(len(extra_ids)),
                    "factor_landmark_ids": sorted(factor_ids),
                    "physical_row_identity": "exact_image_frame_float32_pixel_right_u",
                    "heldout_validation_claim": False,
                    "coordinate_chart": "target_relative_intermediate_and_singleton_v1",
                    "intermediate_camera_model": "target_relative_C_equals_T_inverse_S_no_anchor_prior",
                    "optimized_intermediate_frames": intermediate_frames,
                    "intermediate_pose_offsets": intermediate_offsets,
                    "target_relative_camera_offsets": intermediate_offsets,
                    "target_relative_camera_reference_keyframes": intermediate_reference_keyframes,
                    "target_relative_point_offsets": target_relative_point_offsets,
                    "target_relative_point_reference_keyframes": target_relative_point_reference_keyframes,
                    "target_relative_point_initial_values": {
                        str(ident): _diagnostic_array(augmented_initial[
                            target_relative_point_offsets[ident]:target_relative_point_offsets[ident] + 3
                        ]) for ident in extra_ids
                    },
                    "point_limit": point_limit,
                })
                if source_history_summary is not None and source_history_rows:
                    source_history_summary["source_reference_keyframe_id"] = (
                        int(intermediate_reference_keyframes[source_history_source_frame])
                        if source_history_source_frame in intermediate_reference_keyframes
                        else int(frame_to_keyframe[source_history_source_frame])
                    )
        except Exception as error:
            owned_training_rows = None
            source_history_rows = None
            initial = original_initial
            variable_scale = original_scale
            pattern = original_pattern
            point_limit = original_variable_count
            intermediate_pose_offsets = {}
            intermediate_reference_keyframes = {}
            target_relative_point_offsets = {}
            target_relative_point_reference_keyframes = {}
            extra_index = {}
            extra_ids = []
            training_bundle_summary.update({
                "status": "rejected", "reason": str(error) or type(error).__name__,
                "selected_factors": 0, "unique_image_rows_added": 0,
            })
            if source_history_summary is not None:
                source_history_summary.update(
                    status="skipped", reason="owned_factor_pool_rejected",
                    installed_rows=0, installed_landmark_ids=[], components_added=0,
                    cost_before=0.0, cost_after=0.0,
                )

    if diagnostics_enabled and prepared_payload is not None:
        layout = prepared_payload["parameter_layout"]
        if owned_training_rows is not None:
            chart_layout = {
                "coordinate_chart": "target_relative_intermediate_and_singleton_v1",
                "free_keyframe_pose_chart": "camera_to_world_rotvec_then_translation",
                "selected_point_chart": "world_xyz",
                "singleton_point_chart": "target_camera_xyz_q_equals_T_inverse_P",
                "intermediate_camera_chart": "source_camera_to_target_camera_C_equals_T_inverse_S",
                "original_variable_count": int(original_variable_count),
                "free_keyframe_pose_offsets": {
                    str(key): int(offset) for key, offset in pose_offset.items()
                },
                "world_selected_point_offset": int(point_offset),
                "world_selected_point_count": int(len(landmarks)),
                "point_limit": int(point_limit),
                "target_relative_point_offsets": {
                    str(ident): int(offset)
                    for ident, offset in target_relative_point_offsets.items()
                },
                "target_relative_point_reference_keyframes": {
                    str(ident): int(keyframe)
                    for ident, keyframe in target_relative_point_reference_keyframes.items()
                },
                "target_relative_point_initial_values": {
                    str(ident): _diagnostic_array(initial[offset:offset + 3])
                    for ident, offset in target_relative_point_offsets.items()
                },
                "target_relative_camera_offsets": {
                    str(frame): int(offset)
                    for frame, offset in intermediate_pose_offsets.items()
                },
                "target_relative_camera_reference_keyframes": {
                    str(frame): int(keyframe)
                    for frame, keyframe in intermediate_reference_keyframes.items()
                },
                "target_relative_camera_initial_values_rotvec_translation": {
                    str(frame): _diagnostic_array(initial[offset:offset + 6])
                    for frame, offset in intermediate_pose_offsets.items()
                },
            }
            training_bundle_summary["optimizer_parameter_chart"] = chart_layout
            layout["coordinate_chart"] = chart_layout["coordinate_chart"]
            layout["initial_vector"] = _diagnostic_array(initial)
            layout["variable_scale"] = _diagnostic_array(variable_scale)
            layout["point_limit"] = int(point_limit)
            layout["target_relative_chart"] = chart_layout
        elif provider_enabled:
            layout["coordinate_chart"] = "world_keyframe_poses_and_world_selected_points"
        if source_history_enabled and source_history_summary is not None:
            installed_rows = source_history_rows or []
            table_rows = [
                {
                    "frame_id": int(row["frame_id"]),
                    "landmark_id": int(row["landmark_id"]),
                    "selected_point_index": int(row["point_index"]),
                    "pixel_float32": np.asarray(row["pixel"], dtype=np.float32).tolist(),
                }
                for row in installed_rows
            ]
            active = bool(table_rows) and source_history_summary.get("status") == "active"
            prepared_payload["source_history_observations"] = {
                "schema": "source_history_image_rows_v1",
                "status": "active" if active else "skipped",
                "reason": None if active else source_history_summary.get("reason"),
                "source_frame": source_history_summary.get("source_frame"),
                "target_frame": source_history_summary.get("target_frame"),
                "source_calibration_identity": source_history_summary.get(
                    "source_calibration_identity"),
                "revision": source_history_summary.get("revision"),
                "geometry_revision": source_history_summary.get("geometry_revision"),
                "measurement_role": "tracking_fit_consumed",
                "selection_role": "consumed_or_unknown",
                "independent_unused_claim": False,
                "row_count": len(table_rows),
                "component_count": 2 * len(table_rows),
                "installed_landmark_ids": [row["landmark_id"] for row in table_rows],
                "exclusions": source_history_summary.get("exclusions", {}),
                "initial_source_camera_to_world": source_history_summary.get(
                    "initial_source_camera_to_world"),
                "source_status": source_history_summary.get("source_status"),
                "source_anchor_id": source_history_summary.get("source_anchor_id"),
                "initial_source_anchor_camera_to_world": source_history_summary.get(
                    "initial_source_anchor_camera_to_world"),
                "rows": table_rows,
            }
        emit_diagnostic("prepared", prepared_payload)

    initial_cameras = camera_poses_for(initial)
    before = objective(optimized_residual(initial, initial_cameras))
    held_before = objective(held_out_residual(initial_cameras))
    training_before = (
        objective(owned_training_residual(initial))
        if owned_training_rows is not None else 0.0
    )
    source_history_before = (
        objective(source_history_residual(initial)) if source_history_rows else 0.0
    )
    if training_bundle_summary is not None and owned_training_rows is not None:
        if source_history_summary is not None:
            source_history_summary.update({
                "cost_before": float(source_history_before),
                "components_added": int(2 * len(source_history_rows or [])),
            })
        training_bundle_summary.update({
            "initial_image_objective": float(training_before),
            "initial_affected_objective": float(before + held_before),
            "initial_augmented_objective": float(
                before + held_before + training_before + source_history_before
            ),
        })
    solver_options = {
        "jac_sparsity": pattern.tocsr(),
        "loss": "huber",
        "f_scale": 2.0,
        "max_nfev": 30,
        "x_scale": variable_scale,
        "tr_solver": "lsmr",
    }
    precise_inner_solve = solver_accuracy == "precise" or owned_training_rows is not None
    inner_options = None
    if precise_inner_solve:
        # The added target-relative q/C block can be exactly independent of a
        # free target pose while remaining coupled through shared world points.
        # Default LSMR tolerances produced inaccurate inner steps and a tiny
        # source-pose update despite a lower image cost. The explicit precise
        # policy applies the same inner accuracy to ordinary BA controls.
        # Keep the 30 outer evaluations and all acceptance rules unchanged.
        inner_options = {
            "atol": 1e-12,
            "btol": 1e-12,
            "maxiter": max(500, len(initial)),
        }
        solver_options["tr_options"] = dict(inner_options)
    result = least_squares(residual, initial, **solver_options)
    result_cameras = camera_poses_for(result.x)
    result_points = result.x[point_offset:point_limit].reshape(-1, 3)
    after = objective(optimized_residual(result.x, result_cameras, result_points))
    training_after = (
        objective(owned_training_residual(result.x))
        if owned_training_rows is not None and np.isfinite(result.x).all()
        else (float("inf") if owned_training_rows is not None else 0.0)
    )
    source_history_after = (
        objective(source_history_residual(result.x))
        if source_history_rows and np.isfinite(result.x).all()
        else (float("inf") if source_history_rows else 0.0)
    )
    report = {
        "applied": False,
        "initial_cost": before,
        "final_cost": after,
        "evaluations": int(result.nfev),
        "solver_success": bool(result.success),
        "fixed_keyframes": sorted(set(base) - set(free)),
        "observation_components": len(components),
        "unsupported_local_cameras": sorted(unsupported),
        **solver_metadata(
            "precise_lsmr" if precise_inner_solve else "legacy_defaults",
            inner_options,
        ),
    }
    result_is_finite = bool(np.isfinite(result.x).all())
    poses = points = None
    if result_is_finite:
        poses, points = unpack(result.x)
    if training_bundle_summary is not None:
        if source_history_summary is not None:
            source_history_summary["cost_after"] = float(source_history_after)
        training_bundle_summary.update({
            "final_image_objective": float(training_after),
            "solver_status": "finite_candidate" if result_is_finite else "nonfinite_candidate",
        })
        if result_is_finite and intermediate_pose_offsets:
            changes = []
            for frame in intermediate_pose_offsets:
                candidate = frame_pose_for(result.x, frame, poses)
                difference = np.linalg.inv(frame_pose_snapshot[frame]) @ candidate
                changes.append({
                    'frame_id': int(frame),
                    'translation_change_m': float(np.linalg.norm(difference[:3, 3])),
                    'rotation_change_deg': float(np.degrees(
                        Rotation.from_matrix(difference[:3, :3]).magnitude())),
                })
            training_bundle_summary['intermediate_camera_changes'] = changes
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
            "optimizer_parameter_chart": (
                _diagnostic_value(training_bundle_summary.get("optimizer_parameter_chart"))
                if training_bundle_summary is not None
                and training_bundle_summary.get("optimizer_parameter_chart") is not None
                else {
                    "coordinate_chart": "world_keyframe_poses_and_world_selected_points",
                    "point_limit": int(point_limit),
                }
            ),
            "candidate": {
                "valid": result_is_finite,
                "poses_camera_to_world": (
                    None if poses is None else {
                        str(key): _diagnostic_array(pose)
                        for key, pose in poses.items()
                    }
                ),
                "intermediate_camera_to_world": (
                    None if poses is None else {
                        str(frame): _diagnostic_array(frame_pose_for(result.x, frame, poses))
                        for frame in intermediate_pose_offsets
                    }
                ),
                "selected_landmark_world_positions": (
                    None if points is None else [
                        {"landmark_id": (
                            diagnostic_selected_ids[index]
                            if index < len(diagnostic_selected_ids)
                            else extra_ids[index - len(diagnostic_selected_ids)]
                        ),
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
        optimized_landmarks=len(landmarks) + (len(extra_ids) if owned_training_rows is not None else 0),
        anchor_propagated_single_view_landmarks=(
            len(single_view) - (len(extra_ids) if owned_training_rows is not None else 0)
        ),
        max_camera_translation_change=float(max(
            np.linalg.norm(poses[k][:3, 3]-base[k][:3, 3]) for k in free)),
    )
    if owned_training_rows is not None:
        augmented_initial = before + held_before + training_before + source_history_before
        augmented_final = after + held_after + training_after + source_history_after
        report.update(
            augmented_initial_cost=float(augmented_initial),
            augmented_final_cost=float(augmented_final),
        )
        training_bundle_summary.update({
            "final_affected_objective": float(after + held_after),
            "final_augmented_objective": float(augmented_final),
            "affected_objective_reduced": bool(after + held_after < before + held_before),
            "augmented_objective_reduced": bool(augmented_final < augmented_initial),
        })
        if (not np.isfinite(augmented_final)
                or not augmented_final < augmented_initial):
            return attach_diagnostic_errors({
                **report, "reason": "owned_stereo_augmented_objective_worsened"
            })
    if not np.isfinite(held_after) or not after+held_after < before+held_before:
        return attach_diagnostic_errors({**report, "reason": "affected_observations_worsened"})
    if any(
        np.any(project(points[i : i + 1], poses[k], matrix)[1] <= 0)
        for i, k, _ in records
    ):
        return attach_diagnostic_errors(report)
    if owned_training_rows is not None:
        endpoint_poses = training_endpoint_poses(result.x, poses)
        training_world = points[owned_training_rows["point_indices"]]
        training_camera_points = np.einsum(
            "ni,nij->nj", training_world - endpoint_poses[:, :3, 3],
            endpoint_poses[:, :3, :3],
        )
        if (not np.isfinite(training_camera_points).all()
                or np.any(training_camera_points[:, 2] <= 0.)):
            training_bundle_summary.update(status="rejected", reason="nonpositive_factor_depth")
            return attach_diagnostic_errors({**report, "reason": "owned_stereo_nonpositive_depth"})
    if source_history_rows:
        history_poses, history_points = unpack(result.x)
        for item in source_history_rows:
            camera_pose = frame_pose_for(result.x, int(item["frame_id"]), history_poses)
            camera_point = (history_points[int(item["point_index"])] - camera_pose[:3, 3]) @ camera_pose[:3, :3]
            if not np.isfinite(camera_point).all() or camera_point[2] <= 0.:
                if source_history_summary is not None:
                    source_history_summary.update(status="rejected", reason="nonpositive_history_depth")
                return attach_diagnostic_errors({
                    **report, "reason": "owned_stereo_source_history_nonpositive_depth"
                })
    motion_translation, motion_rotation = [], []
    guarded_motion = motion_checks
    if owned_training_rows is not None:
        guarded_motion = [
            (measurement, edge) for edge, measurement in stereo_motion_snapshot.items()
            if any(frame in intermediate_pose_offsets for frame in edge)
            or (frame_anchor_snapshot[edge[0]] != frame_anchor_snapshot[edge[1]]
                and any(frame_anchor_snapshot[frame] in free for frame in edge))
        ]
        training_bundle_summary['guarded_intermediate_motion_edges'] = [
            list(edge) for _measurement, edge in guarded_motion
            if any(frame in intermediate_pose_offsets for frame in edge)
        ]
    for item in guarded_motion:
        if owned_training_rows is not None:
            measurement, edge = item
            updated = [frame_pose_for(result.x, int(frame), poses) for frame in edge]
        else:
            measurement, anchors, recorded = item
            updated = [
                poses[anchor] @ np.linalg.inv(base[anchor]) @ pose
                if anchor is not None else pose
                for anchor, pose in zip(anchors, recorded)
            ]
        difference = np.linalg.inv(measurement) @ np.linalg.inv(updated[0]) @ updated[1]
        motion_translation.append(float(np.linalg.norm(difference[:3, 3])))
        motion_rotation.append(float(np.degrees(Rotation.from_matrix(difference[:3, :3]).magnitude())))
    report.update(
        independent_stereo_motion_checks=len(guarded_motion),
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
        if owned_training_rows is not None:
            for frame in intermediate_pose_offsets:
                if (frame >= min(len(state.poses), len(state.statuses), len(state.pose_anchors))
                        or not np.array_equal(state.poses[frame], frame_pose_snapshot[frame])
                        or state.statuses[frame] != frame_status_snapshot[frame]
                        or state.pose_anchors[frame] != frame_anchor_snapshot[frame]):
                    return attach_diagnostic_errors({**report, 'reason': 'stale_intermediate_frame'})
            for reference_keyframe in set(intermediate_reference_keyframes.values()) | set(
                    target_relative_point_reference_keyframes.values()):
                reference = int(reference_keyframe)
                if reference not in state.keyframes or reference not in base:
                    return attach_diagnostic_errors({**report, 'reason': 'stale_target_relative_reference'})
                reference_frame = int(diagnostic_frame_ids[reference])
                if (reference_frame >= min(len(state.poses), len(state.statuses), len(state.pose_anchors))
                        or int(state.keyframes[reference].frame) != reference_frame
                        or not np.array_equal(state.keyframes[reference].pose, base[reference])
                        or not np.array_equal(state.poses[reference_frame], frame_pose_snapshot[reference_frame])
                        or state.statuses[reference_frame] != frame_status_snapshot[reference_frame]
                        or state.pose_anchors[reference_frame] != frame_anchor_snapshot[reference_frame]
                        or frame_anchor_snapshot[reference_frame] != reference):
                    return attach_diagnostic_errors({**report, 'reason': 'stale_target_relative_reference'})
            validation_payload = copy.deepcopy(prepared_payload)
            validation_payload["training_factor_phase"] = "validate"
            try:
                _unused_factors, validation_report = training_factor_provider(validation_payload)
            except Exception as error:
                validation_report = {"status": "rejected", "reason": type(error).__name__}
            if (not isinstance(validation_report, dict)
                    or validation_report.get("status") != "validated"
                    or not isinstance(_unused_factors, tuple)
                    or len(_unused_factors) != 0):
                training_bundle_summary.update({
                    "status": "rejected", "reason": (
                        validation_report.get("reason", "final_factor_validation_failed")
                        if isinstance(validation_report, dict) else "malformed_final_factor_validation"
                    ),
                })
                return attach_diagnostic_errors({**report, "reason": "owned_stereo_final_validation_failed"})
            if source_history_rows:
                try:
                    if source_history_validation_payload is None:
                        raise ValueError("source_history_validation_snapshot_missing")
                    source_payload = copy.deepcopy(source_history_validation_payload)
                    source_payload["source_history_phase"] = "validate"
                    history_unused_rows, history_validation = source_history_provider(source_payload)
                except Exception as error:
                    history_unused_rows = None
                    history_validation = {"status": "rejected", "reason": type(error).__name__}
                if (not isinstance(history_validation, dict)
                        or history_validation.get("status") != "validated"
                        or not isinstance(history_unused_rows, (list, tuple))
                        or len(history_unused_rows) != 0):
                    if source_history_summary is not None:
                        source_history_summary.update({
                            "status": "rejected",
                            "reason": (history_validation.get("reason", "source_history_final_validation_failed")
                                       if isinstance(history_validation, dict)
                                       else "malformed_source_history_final_validation"),
                        })
                    return attach_diagnostic_errors({
                        **report, "reason": "owned_stereo_source_history_final_validation_failed"
                    })
                if source_history_summary is not None:
                    source_history_summary["late_validation"] = _diagnostic_value(history_validation)
            training_bundle_summary.update({
                "status": "candidate_validated", "final_validation": _diagnostic_value(validation_report),
            })
        updates = {l.id: p.copy() for l, p in zip(landmarks, points)}
        optimized_single_ids = set()
        if owned_training_rows is not None:
            optimized_single_ids = {
                ident for ident in extra_ids
            }
            for ident, point_index in extra_index.items():
                updates[ident] = points[point_index].copy()
        for ident, anchor, position in single_view:
            if ident in optimized_single_ids:
                continue
            camera = base[anchor][:3, :3].T @ (position-base[anchor][:3, 3])
            updates[ident] = poses[anchor][:3, :3] @ camera + poses[anchor][:3, 3]
        intermediate_updates = ({frame: frame_pose_for(result.x, frame, poses)
                                 for frame in intermediate_pose_offsets}
                                if owned_training_rows is not None else None)
        correction_arguments = {'propagate_landmarks': False, 'landmark_updates': updates}
        if intermediate_updates is not None:
            correction_arguments['frame_updates'] = intermediate_updates
        if not state.apply_corrections(revision, poses, **correction_arguments):
            if training_bundle_summary is not None and owned_training_rows is not None:
                training_bundle_summary.update(status="rejected", reason="atomic_apply_rejected")
            return attach_diagnostic_errors(report)
        # Multiview world points are independent; single-view stereo points retain
        # their measured camera coordinates rather than imposing a pose prior.
        report["applied"] = True
        if training_bundle_summary is not None and owned_training_rows is not None:
            training_bundle_summary.update(status="accepted", reason=None)
            training_bundle_summary['committed_map_revision'] = int(state.revision)
            training_bundle_summary['committed_geometry_revision'] = int(state.geometry_revision)
            if source_history_rows and source_history_summary is not None:
                source_history_summary.update(
                    status="accepted", reason=None,
                    committed_map_revision=int(state.revision),
                    committed_geometry_revision=int(state.geometry_revision),
                )
    return attach_diagnostic_errors(report)
