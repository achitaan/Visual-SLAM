"""Certified finite two-bridge gauge canonicalization for original metric BA.

This module adds no residual, prior, measurement, or optimizer call. It proves
one finite symmetry of the existing world-point image objective, then chooses
the representative closest to the initial orientation of a stable camera.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from scipy.spatial.transform import Rotation


_SCOPE = "original_metric_world_chart_two_bridge_rotation"
_MAX_FREE_CAMERAS = 10
_INITIAL_PROBE_ANGLES = (0.371, -0.613)
_EPS = np.finfo(np.float64).eps


def _reject(reason, **extra):
    return None, {"status": "unsupported", "reason": reason, "scope": _SCOPE, **extra}


def _int_vector(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuO":
        raise ValueError(f"{name}_must_be_integer_vector")
    result = []
    for item in raw.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name}_must_be_integer_vector")
        item = int(item)
        if item < 0:
            raise ValueError(f"{name}_must_be_nonnegative")
        result.append(item)
    return np.asarray(result, dtype=np.int64)


def _real_array(value, name, shape=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name}_must_be_real_numeric")
    result = np.asarray(raw, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name}_shape_invalid")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}_nonfinite")
    return result.copy()


def _owned(value):
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _roundoff_bound(*arrays, operations=128):
    scale = 1.0
    for value in arrays:
        if value is None:
            continue
        array = np.asarray(value)
        if array.size and np.issubdtype(array.dtype, np.number):
            scale = max(scale, float(np.max(np.abs(array))))
    return float(_EPS * operations * scale)


def _axis(points, bridge_indices):
    pivot = np.asarray(points[int(bridge_indices[0])], dtype=np.float64).copy()
    delta = np.asarray(points[int(bridge_indices[1])], dtype=np.float64) - pivot
    length = float(np.linalg.norm(delta))
    tolerance = _roundoff_bound(points, operations=64)
    if not np.isfinite(length) or length <= tolerance:
        raise ValueError("coincident_bridge_points")
    return pivot, delta / length, tolerance


def _distance_to_axis(point, pivot, axis):
    return float(np.linalg.norm(np.cross(np.asarray(point, dtype=np.float64) - pivot, axis)))


def _single_null_observability(value, free_count, point_count):
    if not isinstance(value, dict):
        return "missing_initial_observability"
    if value.get("status") != "unobservable":
        return "initial_observability_not_rank_deficient"
    if value.get("nullity") != 1:
        return "initial_nullity_not_one"
    if value.get("pose_columns") != 6 * free_count:
        return "initial_pose_column_count_mismatch"
    histogram = value.get("per_point_rank_histogram")
    if not isinstance(histogram, dict):
        return "missing_point_block_rank_histogram"
    try:
        rank_three = int(histogram.get("3", 0))
        other = sum(int(count) for rank, count in histogram.items() if str(rank) != "3")
    except (TypeError, ValueError, OverflowError):
        return "invalid_point_block_rank_histogram"
    if rank_three != point_count or other != 0:
        return "selected_point_blocks_not_all_full_rank"
    return None


def certify_two_bridge_rotation(
    camera_ids,
    free_camera_indices,
    selected_record_cameras,
    selected_record_points,
    initial_points,
    excluded_points,
    excluded_cameras,
    initial_observability,
):
    """Certify a unique camera cluster bounded by exactly two selected points.

    Camera/point indices address the existing BA arrays. Stable camera IDs
    determine search and reference-camera order. Record arrays must include all
    selected observations, including rows on fixed cameras outside the window.
    Excluded arrays contain one fixed world point/camera pair per actual row.

    Returns an owned certificate and report, or None and a fail-closed report.
    """
    try:
        ids = _int_vector(camera_ids, "camera_ids")
        free = _int_vector(free_camera_indices, "free_camera_indices")
        record_cameras = _int_vector(selected_record_cameras, "record_cameras")
        record_points = _int_vector(selected_record_points, "record_points")
        held_cameras = _int_vector(excluded_cameras, "excluded_cameras")
        points = _real_array(initial_points, "initial_points")
        held_points = _real_array(excluded_points, "excluded_points")
    except (TypeError, ValueError, OverflowError) as error:
        return _reject(str(error))

    if len(set(ids.tolist())) != len(ids):
        return _reject("camera_ids_not_unique")
    if not len(free):
        return _reject("no_free_camera_subset")
    if len(free) > _MAX_FREE_CAMERAS:
        return _reject("unsupported_certificate_search_size",
                       free_camera_count=int(len(free)),
                       max_free_cameras=_MAX_FREE_CAMERAS)
    if len(set(free.tolist())) != len(free) or np.any(free >= len(ids)):
        return _reject("free_camera_indices_invalid")
    if (points.ndim != 2 or points.shape[1:] != (3,)
            or held_points.ndim != 2 or held_points.shape[1:] != (3,)
            or len(record_cameras) != len(record_points)
            or len(held_cameras) != len(held_points)
            or np.any(record_cameras >= len(ids))
            or np.any(held_cameras >= len(ids))
            or np.any(record_points >= len(points))):
        return _reject("observation_graph_shape_or_index_invalid")
    if not len(points) or not len(record_points):
        return _reject("empty_selected_point_graph")

    rank_reason = _single_null_observability(initial_observability, len(free), len(points))
    if rank_reason is not None:
        return _reject(rank_reason)

    observers = [set() for _ in range(len(points))]
    for camera, point in zip(record_cameras, record_points):
        observers[int(point)].add(int(camera))
    if any(not row for row in observers):
        return _reject("selected_point_without_observation")

    stable_free = sorted((int(index) for index in free), key=lambda index: int(ids[index]))
    candidates = []
    examined = 0
    for size in range(1, len(stable_free) + 1):
        for subset_tuple in combinations(stable_free, size):
            examined += 1
            subset = set(subset_tuple)
            internal, boundary = [], []
            observed_inside = set()
            for point_index, cameras in enumerate(observers):
                inside = cameras & subset
                if not inside:
                    continue
                observed_inside.update(inside)
                if cameras <= subset:
                    internal.append(point_index)
                else:
                    boundary.append(point_index)
            if observed_inside != subset or len(boundary) != 2:
                continue
            try:
                pivot, axis, tolerance = _axis(points, boundary)
            except ValueError:
                continue
            held_rows = [i for i, camera in enumerate(held_cameras) if int(camera) in subset]
            distances = [_distance_to_axis(held_points[i], pivot, axis) for i in held_rows]
            if any(distance > tolerance for distance in distances):
                continue
            candidates.append({
                "cluster": tuple(subset_tuple),
                "internal": tuple(internal),
                "bridges": tuple(boundary),
                "pivot": pivot,
                "axis": axis,
                "axis_tolerance": tolerance,
                "held_rows": tuple(held_rows),
                "held_axis_distances": tuple(distances),
            })

    if len(candidates) != 1:
        reason = "no_certified_two_bridge_cluster" if not candidates else "ambiguous_two_bridge_cluster"
        return _reject(reason, certified_subset_count=int(len(candidates)),
                       examined_subset_count=int(examined))

    selected = candidates[0]
    cluster = selected["cluster"]
    reference = min(cluster, key=lambda index: int(ids[index]))
    cert = {
        "status": "certified",
        "scope": _SCOPE,
        "camera_ids": _owned(ids),
        "free_camera_indices": _owned(free),
        "selected_record_cameras": _owned(record_cameras),
        "selected_record_points": _owned(record_points),
        "initial_points": _owned(points),
        "excluded_points": _owned(held_points),
        "excluded_cameras": _owned(held_cameras),
        "cluster_camera_indices": tuple(cluster),
        "cluster_camera_ids": tuple(int(ids[index]) for index in cluster),
        "bridge_point_indices": tuple(selected["bridges"]),
        "internal_point_indices": tuple(selected["internal"]),
        "reference_camera_index": int(reference),
        "reference_camera_id": int(ids[reference]),
        "initial_pivot": _owned(selected["pivot"]),
        "initial_axis": _owned(selected["axis"]),
        "initial_axis_roundoff_tolerance": float(selected["axis_tolerance"]),
        "held_rows_in_cluster": tuple(selected["held_rows"]),
        "examined_subset_count": int(examined),
        "initial_observability": {
            "status": str(initial_observability["status"]),
            "rank": int(initial_observability.get("rank", -1)),
            "nullity": int(initial_observability["nullity"]),
            "pose_columns": int(initial_observability["pose_columns"]),
            "per_point_rank_histogram": dict(initial_observability["per_point_rank_histogram"]),
        },
    }
    report = {
        "status": "certified",
        "reason": None,
        "scope": _SCOPE,
        "cluster_camera_indices": list(cluster),
        "cluster_camera_ids": [int(ids[index]) for index in cluster],
        "internal_point_indices": list(selected["internal"]),
        "bridge_point_indices": list(selected["bridges"]),
        "reference_camera_index": int(reference),
        "reference_camera_id": int(ids[reference]),
        "initial_axis": selected["axis"].tolist(),
        "initial_pivot": selected["pivot"].tolist(),
        "initial_axis_roundoff_tolerance": float(selected["axis_tolerance"]),
        "initial_held_axis_distances": list(selected["held_axis_distances"]),
        "examined_subset_count": int(examined),
        "supported_class": "exactly_two_selected_boundary_points_and_on_axis_fixed_rows",
    }
    return cert, report


def _matrix_quaternion_wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid_relative_rotation_quaternion")
    quaternion /= norm
    first = next((item for item in quaternion if abs(float(item)) > 8.0 * _EPS), 1.0)
    if first < 0.0:
        quaternion *= -1.0
    return quaternion


def _wrap_principal(theta):
    value = math.atan2(math.sin(float(theta)), math.cos(float(theta)))
    return math.pi if value <= -math.pi else value


def _axis_transform(cameras, points, cluster, internal, pivot, axis, theta):
    transformed_cameras = np.asarray(cameras, dtype=np.float64).copy()
    transformed_points = np.asarray(points, dtype=np.float64).copy()
    G = Rotation.from_rotvec(np.asarray(axis, dtype=np.float64) * float(theta)).as_matrix()
    translation = pivot - G @ pivot
    for index in cluster:
        transformed_cameras[index, :3, :3] = G @ transformed_cameras[index, :3, :3]
        transformed_cameras[index, :3, 3] = G @ transformed_cameras[index, :3, 3] + translation
    if internal:
        indices = np.asarray(internal, dtype=np.intp)
        transformed_points[indices] = (G @ transformed_points[indices].T).T + translation
    return transformed_cameras, transformed_points, G, translation


def _pack_action(vector, cameras, points, cluster, internal, free_indices,
                 free_offsets, point_offset):
    result = np.asarray(vector, dtype=np.float64).copy()
    camera_rows = {int(camera): row for row, camera in enumerate(free_indices)}
    for camera in cluster:
        row = camera_rows.get(int(camera))
        if row is None:
            raise ValueError("cluster_contains_nonfree_camera")
        offsets = free_offsets[row]
        result[offsets[:3]] = Rotation.from_matrix(cameras[camera, :3, :3]).as_rotvec()
        result[offsets[3:]] = cameras[camera, :3, 3]
    for point_index in internal:
        start = point_offset + 3 * int(point_index)
        result[start:start + 3] = points[int(point_index)]
    return result


def _compare_residuals(residual, original_vector, transformed_vector, geometry):
    try:
        original = np.asarray(residual(np.asarray(original_vector).copy()), dtype=np.float64).reshape(-1)
        changed = np.asarray(residual(np.asarray(transformed_vector).copy()), dtype=np.float64).reshape(-1)
    except Exception as error:  # fail closed if the production objective cannot be evaluated
        return False, {"status": "failed", "reason": f"residual_error:{type(error).__name__}"}
    if original.shape != changed.shape or not np.isfinite(original).all() or not np.isfinite(changed).all():
        return False, {"status": "failed", "reason": "residual_shape_or_finiteness_mismatch"}
    tolerance = _roundoff_bound(original, changed, *geometry, operations=512)
    maximum = float(np.max(np.abs(original - changed))) if original.size else 0.0
    return maximum <= tolerance, {
        "status": "passed" if maximum <= tolerance else "failed",
        "max_abs_difference": maximum,
        "roundoff_tolerance": tolerance,
        "residual_components": int(original.size),
    }


def canonicalize_two_bridge_candidate(
    initial_vector,
    raw_candidate_vector,
    camera_ids,
    free_camera_indices,
    free_offsets,
    point_offset,
    point_limit,
    initial_camera_poses,
    candidate_camera_poses,
    certificate,
    excluded_points,
    excluded_cameras,
    complete_residual,
):
    """Choose the closest-initial-orientation representative of a certified orbit.

    Returns (canonical vector, report), or (None, rejected report). Caller must
    run candidate observability and every normal cost/depth/motion/stale guard
    before committing the canonical vector.
    """
    try:
        if not isinstance(certificate, dict) or certificate.get("status") != "certified":
            raise ValueError("certificate_not_certified")
        ids = _int_vector(camera_ids, "camera_ids")
        free = _int_vector(free_camera_indices, "free_camera_indices")
        offsets = np.asarray(free_offsets)
        if (offsets.ndim != 2 or offsets.shape != (len(free), 6)
                or offsets.dtype.kind not in "iu" or np.iscomplexobj(offsets)):
            raise ValueError("free_offsets_shape_invalid")
        offsets = np.asarray(offsets, dtype=np.int64).copy()
        x0 = _real_array(initial_vector, "initial_vector")
        raw = _real_array(raw_candidate_vector, "raw_candidate_vector", shape=x0.shape)
        initial_cameras = _real_array(initial_camera_poses, "initial_camera_poses")
        candidate_cameras = _real_array(candidate_camera_poses, "candidate_camera_poses")
        held = _real_array(excluded_points, "excluded_points")
        held_cameras = _int_vector(excluded_cameras, "excluded_cameras")
        point_offset, point_limit = int(point_offset), int(point_limit)
    except (TypeError, ValueError, OverflowError) as error:
        return None, {"status": "rejected", "reason": str(error), "scope": _SCOPE}

    def reject(reason, **extra):
        return None, {"status": "rejected", "reason": reason, "scope": _SCOPE, **extra}

    try:
        cert_ids = np.asarray(certificate["camera_ids"], dtype=np.int64)
        cert_free = np.asarray(certificate["free_camera_indices"], dtype=np.int64)
        cert_record_cameras = np.asarray(certificate["selected_record_cameras"], dtype=np.int64)
        cert_record_points = np.asarray(certificate["selected_record_points"], dtype=np.int64)
        cert_initial_points = _real_array(certificate["initial_points"], "certificate_initial_points")
        cert_held = np.asarray(certificate["excluded_points"], dtype=np.float64)
        cert_held_cameras = np.asarray(certificate["excluded_cameras"], dtype=np.int64)
        cert_observability = certificate["initial_observability"]
        cluster = tuple(int(i) for i in certificate["cluster_camera_indices"])
        internal = tuple(int(i) for i in certificate["internal_point_indices"])
        bridge = tuple(int(i) for i in certificate["bridge_point_indices"])
        cert_reference = int(certificate["reference_camera_index"])
        cert_reference_id = int(certificate["reference_camera_id"])
        cert_initial_pivot = _real_array(certificate["initial_pivot"], "certificate_initial_pivot",
                                         shape=(3,))
        cert_initial_axis = _real_array(certificate["initial_axis"], "certificate_initial_axis",
                                        shape=(3,))
    except (KeyError, TypeError, ValueError, OverflowError):
        return reject("certificate_malformed")

    if (not callable(complete_residual) or not np.array_equal(ids, cert_ids)
            or not np.array_equal(free, cert_free)):
        return reject("certificate_binding_mismatch")
    if (initial_cameras.shape != candidate_cameras.shape
            or initial_cameras.shape != (len(ids), 4, 4)
            or len(held) != len(held_cameras)
            or not np.array_equal(held_cameras, cert_held_cameras)
            or not np.array_equal(held, cert_held)):
        return reject("candidate_geometry_binding_mismatch")
    if point_offset < 0 or point_limit < point_offset or point_limit > len(x0):
        return reject("point_block_bounds_invalid")
    if ((point_limit - point_offset) % 3
            or (point_limit - point_offset) // 3 != len(cert_initial_points)
            or point_limit != len(x0)
            or point_offset != 6 * len(free)):
        return reject("point_block_size_mismatch")
    if len(set(free.tolist())) != len(free) or np.any(free >= len(ids)):
        return reject("free_camera_indices_invalid")
    if np.any(offsets < 0) or np.any(offsets >= len(x0)):
        return reject("free_offsets_out_of_bounds")
    if len(set(offsets.ravel().tolist())) != offsets.size:
        return reject("free_offsets_duplicate_indices")
    expected_offsets = (
        np.arange(len(free), dtype=np.int64)[:, None] * 6
        + np.arange(6, dtype=np.int64)[None, :]
    )
    if not np.array_equal(offsets, expected_offsets):
        return reject("free_offsets_do_not_match_original_layout")
    if np.any((offsets >= point_offset) & (offsets < point_limit)):
        return reject("free_offsets_overlap_points")
    initial_points = x0[point_offset:point_limit].reshape(-1, 3)
    raw_points = raw[point_offset:point_limit].reshape(-1, 3)
    if not np.allclose(initial_points, cert_initial_points, rtol=0.0,
                       atol=_roundoff_bound(initial_points, cert_initial_points)):
        return reject("initial_point_block_binding_mismatch")

    if (cert_record_cameras.ndim != 1 or cert_record_points.ndim != 1
            or len(cert_record_cameras) != len(cert_record_points)
            or np.any(cert_record_cameras >= len(ids))
            or np.any(cert_record_points >= len(cert_initial_points))):
        return reject("certificate_record_graph_invalid")
    # Recompute the small bounded certificate from its owned graph data. This
    # prevents a hand-built or corrupted dictionary marked certified from
    # smuggling an arbitrary cluster or point classification into the action.
    rebuilt, rebuilt_report = certify_two_bridge_rotation(
        cert_ids, cert_free, cert_record_cameras, cert_record_points,
        cert_initial_points, cert_held, cert_held_cameras, cert_observability,
    )
    if rebuilt is None:
        return reject("certificate_revalidation_failed",
                      certificate_reason=rebuilt_report.get("reason"))
    expected_signature = (
        tuple(rebuilt["cluster_camera_indices"]),
        tuple(rebuilt["internal_point_indices"]),
        tuple(rebuilt["bridge_point_indices"]),
        int(rebuilt["reference_camera_index"]),
        int(rebuilt["reference_camera_id"]),
    )
    supplied_signature = (cluster, internal, bridge, cert_reference, cert_reference_id)
    if supplied_signature != expected_signature:
        return reject("certificate_cluster_or_point_binding_mismatch")
    cert_initial_pivot = np.asarray(rebuilt["initial_pivot"], dtype=np.float64)
    cert_initial_axis = np.asarray(rebuilt["initial_axis"], dtype=np.float64)
    cert = rebuilt
    cluster = tuple(int(i) for i in cert["cluster_camera_indices"])
    internal = tuple(int(i) for i in cert["internal_point_indices"])
    bridge = tuple(int(i) for i in cert["bridge_point_indices"])
    reference = int(cert["reference_camera_index"])

    for row, camera_index in enumerate(free):
        if int(camera_index) >= len(ids):
            return reject("free_camera_indices_invalid")
        start = offsets[row]
        for vector, cameras in ((x0, initial_cameras), (raw, candidate_cameras)):
            try:
                rotation = Rotation.from_rotvec(vector[start[:3]]).as_matrix()
            except ValueError:
                return reject("camera_rotvec_invalid")
            translation = vector[start[3:]]
            pose = cameras[int(camera_index)]
            tolerance = _roundoff_bound(pose, translation, operations=256)
            if (not np.allclose(rotation, pose[:3, :3], rtol=0.0, atol=tolerance)
                    or not np.allclose(translation, pose[:3, 3], rtol=0.0, atol=tolerance)):
                return reject("camera_pose_vector_binding_mismatch")

    if len(bridge) != 2 or any(index < 0 or index >= len(raw_points) for index in bridge):
        return reject("certificate_bridge_binding_invalid")
    if any(index < 0 or index >= len(raw_points) for index in internal):
        return reject("certificate_internal_point_index_invalid")
    try:
        candidate_pivot, candidate_axis, axis_tolerance = _axis(raw_points, bridge)
    except ValueError as error:
        return reject(str(error))
    cluster_set = set(cluster)
    held_rows = [i for i, camera in enumerate(held_cameras) if int(camera) in cluster_set]
    held_distances = [_distance_to_axis(held[i], candidate_pivot, candidate_axis)
                      for i in held_rows]
    if any(distance > axis_tolerance for distance in held_distances):
        return reject("candidate_held_point_off_bridge_axis",
                      candidate_axis=candidate_axis.tolist(),
                      candidate_pivot=candidate_pivot.tolist(),
                      candidate_axis_roundoff_tolerance=float(axis_tolerance),
                      candidate_held_axis_distances=held_distances)

    if (reference not in cluster_set or reference < 0 or reference >= len(ids)
            or int(ids[reference]) != int(cert["reference_camera_id"])):
        return reject("reference_camera_certificate_invalid")
    relative = candidate_cameras[reference, :3, :3] @ initial_cameras[reference, :3, :3].T
    try:
        w, qx, qy, qz = _matrix_quaternion_wxyz(relative)
    except (ValueError, np.linalg.LinAlgError):
        return reject("invalid_relative_rotation_quaternion")
    qvec = np.array([qx, qy, qz], dtype=np.float64)
    axial = float(np.dot(candidate_axis, qvec))
    uniqueness = float(math.hypot(w, axial))
    uniqueness_tolerance = 64.0 * _EPS
    trace_amplitude = 2.0 * uniqueness * uniqueness
    trace_tolerance = 64.0 * _EPS
    if trace_amplitude <= trace_tolerance:
        return reject("closest_orientation_nonunique",
                      orientation_uniqueness_measure=uniqueness,
                      trace_amplitude=trace_amplitude,
                      trace_roundoff_tolerance=trace_tolerance)
    theta = _wrap_principal(2.0 * math.atan2(-axial, w))

    initial_pivot = cert_initial_pivot
    initial_axis = cert_initial_axis
    initial_action_checks = []
    for angle in _INITIAL_PROBE_ANGLES:
        changed_cameras, changed_points, _G, _translation = _axis_transform(
            initial_cameras, initial_points, cluster, internal,
            initial_pivot, initial_axis, angle,
        )
        changed_vector = _pack_action(
            x0, changed_cameras, changed_points, cluster, internal, free,
            offsets, point_offset,
        )
        passed, check = _compare_residuals(
            complete_residual, x0, changed_vector,
            (initial_cameras, initial_points, held),
        )
        check["angle_rad"] = float(angle)
        initial_action_checks.append(check)
        if not passed:
            return reject("initial_finite_action_residual_mismatch",
                          initial_action_checks=initial_action_checks)

    transformed_cameras, transformed_points, G, translation = _axis_transform(
        candidate_cameras, raw_points, cluster, internal,
        candidate_pivot, candidate_axis, theta,
    )
    canonical = _pack_action(
        raw, transformed_cameras, transformed_points, cluster, internal,
        free, offsets, point_offset,
    )
    passed, candidate_check = _compare_residuals(
        complete_residual, raw, canonical,
        (candidate_cameras, raw_points, held),
    )
    if not passed:
        return reject("candidate_canonical_residual_mismatch",
                      initial_action_checks=initial_action_checks,
                      candidate_action_check=candidate_check)

    report = {
        "status": "canonicalized",
        "reason": None,
        "scope": _SCOPE,
        "cluster_camera_indices": list(cluster),
        "cluster_camera_ids": [int(ids[index]) for index in cluster],
        "internal_point_indices": list(internal),
        "bridge_point_indices": list(bridge),
        "reference_camera_index": reference,
        "reference_camera_id": int(ids[reference]),
        "initial_axis": initial_axis.tolist(),
        "initial_pivot": initial_pivot.tolist(),
        "candidate_axis": candidate_axis.tolist(),
        "candidate_pivot": candidate_pivot.tolist(),
        "candidate_axis_roundoff_tolerance": float(axis_tolerance),
        "candidate_held_axis_distances": held_distances,
        "theta_rad": float(theta),
        "orientation_uniqueness_measure": uniqueness,
        "trace_amplitude": trace_amplitude,
        "trace_roundoff_tolerance": trace_tolerance,
        "quaternion_wxyz": [float(w), float(qx), float(qy), float(qz)],
        "world_rotation": G.tolist(),
        "world_translation": translation.tolist(),
        "initial_action_probe_angles_rad": list(_INITIAL_PROBE_ANGLES),
        "initial_action_checks": initial_action_checks,
        "candidate_action_check": candidate_check,
        "bridge_points_unchanged": bool(np.array_equal(transformed_points[list(bridge)],
                                                       raw_points[list(bridge)])),
        "raw_optimizer_vector_retained_separately": True,
    }
    return canonical, report
