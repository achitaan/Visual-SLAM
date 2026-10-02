"""Bounded local bundle adjustment with explicit monocular gauge constraints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from mapping_geometry import project


def local_bundle_adjustment(state, matrix, baseline=0.0, window=5, max_landmarks=200, optimized=True):
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

    def unpack(x):
        poses = {k: p.copy() for k, p in base.items()}
        for k, offset in pose_offset.items():
            poses[k][:3, :3] = Rotation.from_rotvec(x[offset : offset + 3]).as_matrix()
            poses[k][:3, 3] = x[offset + 3 : offset + 6]
        return poses, x[point_offset:].reshape(-1, 3)

    dimensions = [
        3 if state.metric and o.right_u is not None else 2 for _, _, o in records
    ]
    pattern = lil_matrix((sum(dimensions), len(initial)), dtype=int)
    row = 0
    for (i, k, _), dim in zip(records, dimensions):
        if k in pose_offset:
            pattern[row : row + dim, pose_offset[k] : pose_offset[k] + 6] = 1
        pattern[row : row + dim, point_offset + 3 * i : point_offset + 3 * i + 3] = 1
        row += dim

    record_points = np.array([i for i, _, _ in records])
    camera_ids = list(dict.fromkeys(k for _, k, _ in records))
    camera_index = {k: i for i, k in enumerate(camera_ids)}
    record_cameras = np.array([camera_index[k] for _, k, _ in records])
    base_cameras = np.array([base[k] for k in camera_ids])
    free_camera_ids = [k for k in free if k in camera_index]
    free_camera_indices = np.array([camera_index[k] for k in free_camera_ids], int)
    free_offsets = np.array([pose_offset[k] for k in free_camera_ids], int)[:, None] + np.arange(6)
    measured_pixels = np.array([o.pixel for _, _, o in records])
    measured_right = np.array([o.right_u if o.right_u is not None else 0.0 for _, _, o in records])
    dimension_mask = np.array([[True, True, dim == 3] for dim in dimensions])

    def residual(x):
        if optimized:
            camera_poses = base_cameras.copy()
            values = x[free_offsets]
            if len(values):
                camera_poses[free_camera_indices, :3, :3] = Rotation.from_rotvec(values[:, :3]).as_matrix()
                camera_poses[free_camera_indices, :3, 3] = values[:, 3:]
            cameras = camera_poses[record_cameras]
            points = x[point_offset:].reshape(-1, 3)
        else:
            poses, points = unpack(x)
            cameras = np.array([poses[k] for _, k, _ in records])
        coordinates = points[record_points] - cameras[:, :3, 3]
        camera = np.einsum("ni,nij->nj", coordinates, cameras[:, :3, :3])
        homogeneous = camera @ matrix.T
        z = camera[:, 2]
        pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9)
        errors = np.clip(pixels - measured_pixels, -1e4, 1e4)
        errors[z <= 0] = 1e4
        if not state.metric:
            return errors.ravel()
        values = np.zeros((len(records), 3))
        values[:, :2] = errors
        values[:, 2] = (
            pixels[:, 0]
            - matrix[0, 0] * baseline / np.maximum(z, 1e-9)
            - measured_right
        )
        values[z <= 0] = 1e4
        return values[dimension_mask]

    def objective(r):
        a = np.abs(r)
        return float(np.sum(np.where(a <= 2.0, 0.5 * r * r, 2.0 * (a - 1.0))))

    before = objective(residual(initial))
    result = least_squares(
        residual,
        initial,
        jac_sparsity=pattern.tocsr(),
        loss="huber",
        f_scale=2.0,
        max_nfev=30,
        x_scale="jac",
        tr_solver="lsmr",
    )
    after = objective(residual(result.x))
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
    if not np.isfinite(result.x).all() or not after < before:
        return report
    poses, points = unpack(result.x)
    if any(
        np.any(project(points[i : i + 1], poses[k], matrix)[1] <= 0)
        for i, k, _ in records
    ):
        return report
    with state.lock:
        if state.revision != revision:
            return {**report, "reason": "stale_revision"}
        if not state.apply_corrections(revision, poses):
            return report
        for l, p in zip(landmarks, points):
            state.landmarks[l.id].position = p.copy()
        # Unoptimized landmarks follow their anchor correction; optimized ones use BA positions.
        report["applied"] = True
    return report
