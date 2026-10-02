"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from mapping_geometry import project, right_pixel


def local_bundle_adjustment(state, matrix, baseline=0.0, window=5, max_landmarks=200, disparity_offset=0.0):
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

    dimensions = [
        3 if state.metric and o.right_u is not None else 2 for _, _, o in records
    ]
    held_dimensions = [3 if state.metric and o.right_u is not None else 2 for _, _, o in held_out]
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

    def optimized_residual(x):
        poses, points = unpack(x)
        cameras = np.array([poses[k] for _, k, _ in records])
        coordinates = points[np.array([i for i, _, _ in records])] - cameras[:, :3, 3]
        camera = np.einsum("ni,nij->nj", coordinates, cameras[:, :3, :3])
        homogeneous = camera @ matrix.T
        z = camera[:, 2]
        pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        errors = np.clip(pixels - np.array([o.pixel for _, _, o in records]), -1e4, 1e4)
        errors[z <= 0] = 1e4
        if not state.metric:
            return errors.ravel()
        values = np.zeros((len(records), 3))
        values[:, :2] = errors
        values[:, 2] = (
            right_pixel(pixels[:, 0], z, matrix[0, 0], baseline, disparity_offset)
            - np.array(
                [o.right_u if o.right_u is not None else 0.0 for _, _, o in records]
            )
        )
        values[z <= 0] = 1e4
        return values[np.array([[True, True, dim == 3] for dim in dimensions])]

    def objective(r):
        a = np.abs(r)
        return float(np.sum(np.where(a <= 2.0, 0.5 * r * r, 2.0 * (a - 1.0))))

    def held_out_residual(poses):
        if not held_out:
            return np.empty(0)
        cameras = np.array([poses[k] for _, k, _ in held_out])
        coordinates = np.array([p for p, _, _ in held_out]) - cameras[:, :3, 3]
        camera = np.einsum("ni,nij->nj", coordinates, cameras[:, :3, :3])
        homogeneous = camera @ matrix.T
        pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        errors = np.clip(pixels - np.array([o.pixel for _, _, o in held_out]), -1e4, 1e4)
        errors[camera[:, 2] <= 0] = 1e4
        if not state.metric:
            return errors.ravel()
        # Match the observation-by-observation row layout of the sparse
        # Jacobian, including mixed left-only and stereo observations.
        values = np.zeros((len(held_out), 3))
        values[:, :2] = errors
        values[:, 2] = (
            right_pixel(pixels[:, 0], camera[:, 2], matrix[0, 0], baseline, disparity_offset)
            - np.array([o.right_u if o.right_u is not None else 0. for _, _, o in held_out])
        )
        values[camera[:, 2] <= 0] = 1e4
        return values[np.array([[True, True, dim == 3] for dim in held_dimensions])]

    def residual(x):
        poses, _ = unpack(x)
        return np.r_[optimized_residual(x), held_out_residual(poses)]

    before = objective(optimized_residual(initial))
    held_before = objective(held_out_residual(base))
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
    after = objective(optimized_residual(result.x))
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
    held_after = objective(held_out_residual(poses))
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
