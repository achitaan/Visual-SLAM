"""Bounded, opt-in two-view stereo refinement over full XYZ training tracks.

This module only returns a relative pose candidate.  The caller remains
responsible for preserving the original verified pose on every rejection and
for applying the existing independent holdout arbitration unchanged.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.transform import Rotation

from mapping_geometry import coverage, project
from pose_observability import _observation_jacobians, original_image_pose_observability


_MASK = np.ones(3, dtype=bool)
_IDENTITY = np.eye(4, dtype=np.float64)
_DEPTH_MIN_M = 0.1
_DEPTH_MAX_M = 100.0
_MAX_TRANSLATION_M = 0.5
_MAX_ROTATION_DEG = 1.5
_MAX_NFEV = 15
_HUBER_F_SCALE = 1.5


def _finite_array(name, value, shape_tail):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name}_must_be_real")
    result = np.asarray(raw, dtype=np.float64)
    if (result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail
            or not np.isfinite(result).all()):
        raise ValueError(f"invalid_{name}")
    return np.array(result, dtype=np.float64, copy=True)


def _integer_pairs(name, value):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1:] != (2,) or np.iscomplexobj(raw)
            or raw.dtype.kind not in "iu"):
        raise ValueError(f"invalid_{name}")
    pairs = np.array(raw, dtype=np.int64, copy=True)
    if np.any(pairs < 0) or len({(int(a), int(b)) for a, b in pairs}) != len(pairs):
        raise ValueError(f"invalid_or_duplicate_{name}")
    return pairs


def _proper_pose(name, value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name}_must_be_real")
    pose = np.asarray(raw, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid_{name}")
    if (not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3),
                               rtol=0.0, atol=1e-6)
            or np.linalg.det(pose[:3, :3]) <= 0.0
            or abs(float(np.linalg.det(pose[:3, :3])) - 1.0) > 1e-6):
        raise ValueError(f"invalid_{name}")
    return np.array(pose, dtype=np.float64, copy=True)


def _pixel_key(pixel):
    # Match the physical endpoint identity used by the stereo pool partition.
    p = np.asarray(pixel, dtype=np.float32)
    return (float(p[0]), float(p[1]))


def _huber_cost(residual, f_scale=_HUBER_F_SCALE):
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    quadratic = absolute <= f_scale
    rho = np.empty_like(absolute)
    rho[quadratic] = absolute[quadratic] ** 2
    rho[~quadratic] = 2.0 * f_scale * absolute[~quadratic] - f_scale**2
    return float(0.5 * np.sum(rho))


def _motion_error(reference, candidate):
    delta = np.linalg.inv(reference) @ candidate
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
    return float(np.linalg.norm(delta[:3, 3])), angle


def _matrix_error(delta):
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
    return float(np.linalg.norm(delta[:3, 3])), angle


@dataclass
class TwoViewStereoProblem:
    """Owned immutable observations and the full-XYZ optimization chart."""

    x0: np.ndarray
    selected_pairs: np.ndarray
    selected_pair_indices: np.ndarray
    source_points: np.ndarray
    target_points: np.ndarray
    source_pixels: np.ndarray
    target_pixels: np.ndarray
    source_right_u: np.ndarray
    target_right_u: np.ndarray
    seed_pose: np.ndarray
    reverse_pose: np.ndarray
    matrix: np.ndarray
    baseline: float
    disparity_offset: float
    image_size: tuple
    min_inliers: int
    direction_forward_pairs: np.ndarray
    direction_reverse_pairs: np.ndarray
    direction_forward_source_points: np.ndarray
    direction_forward_target_pixels: np.ndarray
    direction_reverse_target_points: np.ndarray
    direction_reverse_source_pixels: np.ndarray
    selection_report: dict

    @property
    def point_count(self):
        return int(len(self.selected_pairs))

    @property
    def variable_count(self):
        return int(6 + 3 * self.point_count)

    @property
    def residual_rows(self):
        return int(6 * self.point_count)

    @property
    def scale(self):
        return np.r_[np.ones(3), np.full(3, self.baseline),
                     np.full(3 * self.point_count, self.baseline)]

    def decode(self, x):
        value = np.asarray(x, dtype=np.float64)
        if value.shape != self.x0.shape or not np.isfinite(value).all():
            raise ValueError("invalid_chart_vector")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix()
        pose[:3, 3] = value[3:6]
        return pose, np.array(value[6:].reshape(-1, 3), copy=True)

    def _depths(self, x):
        pose, points = self.decode(x)
        target_camera = (points - pose[:3, 3]) @ pose[:3, :3]
        return points[:, 2].copy(), target_camera[:, 2].copy()

    def _check_chart_domain(self, x):
        source_depth, target_depth = self._depths(x)
        if (not np.isfinite(source_depth).all() or not np.isfinite(target_depth).all()
                or np.any(source_depth <= _DEPTH_MIN_M)
                or np.any(source_depth >= _DEPTH_MAX_M)
                or np.any(target_depth <= _DEPTH_MIN_M)
                or np.any(target_depth >= _DEPTH_MAX_M)):
            raise ValueError("depth_outside_declared_0.1_100m_domain")
    def residual(self, x):
        self._check_chart_domain(x)
        pose, points = self.decode(x)
        phi = np.asarray(x[:3], dtype=np.float64)
        residual = np.empty(self.residual_rows, dtype=np.float64)
        for i, point in enumerate(points):
            source, _jp, _jx = _observation_jacobians(
                _IDENTITY, np.zeros(3), point, self.source_pixels[i],
                self.source_right_u[i], _MASK, self.matrix, self.baseline,
                self.disparity_offset)
            target, _jp, _jx = _observation_jacobians(
                pose, phi, point, self.target_pixels[i], self.target_right_u[i],
                _MASK, self.matrix, self.baseline, self.disparity_offset)
            residual[6 * i:6 * i + 3] = source
            residual[6 * i + 3:6 * i + 6] = target
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite_residual")
        return residual

    def jacobian(self, x):
        self._check_chart_domain(x)
        pose, points = self.decode(x)
        phi = np.asarray(x[:3], dtype=np.float64)
        rows, cols, values = [], [], []
        for i, point in enumerate(points):
            _rs, _pose_s, point_s = _observation_jacobians(
                _IDENTITY, np.zeros(3), point, self.source_pixels[i],
                self.source_right_u[i], _MASK, self.matrix, self.baseline,
                self.disparity_offset)
            _rt, pose_t, point_t = _observation_jacobians(
                pose, phi, point, self.target_pixels[i], self.target_right_u[i],
                _MASK, self.matrix, self.baseline, self.disparity_offset)
            source_rows = np.arange(6 * i, 6 * i + 3)
            target_rows = np.arange(6 * i + 3, 6 * i + 6)
            source_cols = np.arange(6 + 3 * i, 6 + 3 * i + 3)
            target_point_cols = source_cols
            target_pose_cols = np.arange(6)
            for rr, cc, block in (
                (source_rows, source_cols, point_s),
                (target_rows, target_pose_cols, pose_t),
                (target_rows, target_point_cols, point_t),
            ):
                row_grid, col_grid = np.meshgrid(rr, cc, indexing="ij")
                rows.extend(row_grid.ravel().tolist())
                cols.extend(col_grid.ravel().tolist())
                values.extend(np.asarray(block, dtype=np.float64).ravel().tolist())
        result = coo_matrix((values, (rows, cols)),
                            shape=(self.residual_rows, self.variable_count)).tocsr()
        if not np.isfinite(result.data).all():
            raise ValueError("nonfinite_jacobian")
        return result

    def sparsity(self):
        # A dedicated structural matrix keeps source point rows point-local,
        # while every target triplet couples the common six-pose block.
        rows, cols = [], []
        for i in range(self.point_count):
            for rr in range(6 * i, 6 * i + 3):
                for cc in range(6 + 3 * i, 6 + 3 * i + 3):
                    rows.append(rr); cols.append(cc)
            for rr in range(6 * i + 3, 6 * i + 6):
                for cc in range(6):
                    rows.append(rr); cols.append(cc)
                for cc in range(6 + 3 * i, 6 + 3 * i + 3):
                    rows.append(rr); cols.append(cc)
        return coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                          shape=(self.residual_rows, self.variable_count)).tocsr()

    def observability(self, x):
        pose, points = self.decode(x)
        camera_poses = np.stack((_IDENTITY, pose))
        selected_camera_indices = np.tile(np.array([0, 1], dtype=np.int64), self.point_count)
        selected_point_indices = np.repeat(np.arange(self.point_count, dtype=np.int64), 2)
        pixels = np.empty((2 * self.point_count, 2), dtype=np.float64)
        pixels[0::2] = self.source_pixels
        pixels[1::2] = self.target_pixels
        rights = np.empty(2 * self.point_count, dtype=np.float64)
        rights[0::2] = self.source_right_u
        rights[1::2] = self.target_right_u
        masks = np.ones((2 * self.point_count, 3), dtype=bool)
        pose_scales = np.r_[np.ones(3), np.full(3, self.baseline)][None, :]
        point_scales = np.full((self.point_count, 3), self.baseline, dtype=np.float64)
        observability = original_image_pose_observability(
            camera_poses, [1], [np.asarray(x[:3], dtype=np.float64)],
            points, selected_camera_indices, selected_point_indices, pixels, rights,
            masks, np.empty((0, 3)), np.empty(0, dtype=np.int64),
            np.empty((0, 2)), np.empty(0), np.empty((0, 3), dtype=bool),
            self.matrix, self.baseline, self.disparity_offset, pose_scales, point_scales)
        jacobian = self.jacobian(x)
        point_singular_values = []
        point_rank_failures = []
        eps = np.finfo(np.float64).eps
        for i in range(self.point_count):
            block = jacobian[6 * i:6 * i + 6, 6 + 3 * i:9 + 3 * i].toarray()
            singular = np.linalg.svd(block * self.baseline, compute_uv=False)
            tolerance = eps * max(block.shape) * (float(singular[0]) if len(singular) else 0.0)
            rank = int(np.count_nonzero(singular > tolerance))
            point_singular_values.append(singular.tolist())
            if rank != 3:
                point_rank_failures.append({"point_index": int(i), "rank": rank,
                                            "singular_values": singular.tolist(),
                                            "tolerance": float(tolerance)})
        observability["expected_rank"] = 6
        observability["pose_scales"] = [1.0, 1.0, 1.0,
                                        self.baseline, self.baseline, self.baseline]
        observability["point_scales"] = [self.baseline] * 3
        observability["point_singular_values"] = point_singular_values
        observability["point_rank_failures"] = point_rank_failures
        return observability

    def direction_checks(self, x):
        pose, _points = self.decode(x)
        inverse = np.linalg.inv(pose)
        forward_prediction, forward_depth = project(
            self.direction_forward_source_points, pose, self.matrix)
        reverse_prediction, reverse_depth = project(
            self.direction_reverse_target_points, inverse, self.matrix)
        forward_error = np.linalg.norm(
            forward_prediction - self.direction_forward_target_pixels, axis=1)
        reverse_error = np.linalg.norm(
            reverse_prediction - self.direction_reverse_source_pixels, axis=1)
        forward_median = float(np.median(forward_error)) if len(forward_error) else float("inf")
        reverse_median = float(np.median(reverse_error)) if len(reverse_error) else float("inf")
        result = {
            "forward_original_inlier_count": int(len(forward_error)),
            "reverse_original_inlier_count": int(len(reverse_error)),
            "minimum_original_directional_inliers": int(self.min_inliers),
            "forward_original_median_px": forward_median,
            "reverse_original_median_px": reverse_median,
            "forward_original_coverage_cells": int(coverage(
                self.direction_forward_target_pixels, self.image_size)),
            "reverse_original_coverage_cells": int(coverage(
                self.direction_reverse_source_pixels, self.image_size)),
            "forward_original_positive_depth": bool(np.all(forward_depth > 0.0)),
            "reverse_original_positive_depth": bool(np.all(reverse_depth > 0.0)),
        }
        result["passed"] = bool(
            np.isfinite(forward_median) and np.isfinite(reverse_median)
            and len(forward_error) >= self.min_inliers
            and len(reverse_error) >= self.min_inliers
            and forward_median <= 1.5 and reverse_median <= 1.5
            and result["forward_original_coverage_cells"] >= 3
            and result["reverse_original_coverage_cells"] >= 3
            and result["forward_original_positive_depth"]
            and result["reverse_original_positive_depth"])
        return result


def build_two_view_stereo_problem(
    seed_pose, reverse_pose, fit_pairs, forward_inlier_pairs, reverse_inlier_pairs,
    source_points, target_points, source_pixels, source_right_u,
    target_pixels, target_right_u, matrix, baseline, disparity_offset,
    image_size, *, min_inliers, held_pairs=None, source_landmark_ids=None,
    target_landmark_ids=None,
    excluded_landmark_ids=(),
):
    """Build a full-XYZ problem from the exact original training-row union.

    All feature arrays are in their original SupportedStereoFrame row order.
    Pair rows are selected by exact integer pair identity, never proximity.
    Heldout physical endpoints and held landmark identities are excluded.
    """
    seed = _proper_pose("seed_pose", seed_pose)
    fit = _integer_pairs("fit_pairs", fit_pairs)
    forward = _integer_pairs("forward_inlier_pairs", forward_inlier_pairs)
    reverse = _integer_pairs("reverse_inlier_pairs", reverse_inlier_pairs)
    source_xyz_raw = np.asarray(source_points)
    if np.iscomplexobj(source_xyz_raw):
        raise ValueError("source_points_must_be_real")
    source_xyz = np.asarray(source_xyz_raw, dtype=np.float64)
    if source_xyz.ndim != 2 or source_xyz.shape[1:] != (3,):
        raise ValueError("invalid_source_points")
    target_xyz_raw = np.asarray(target_points)
    if np.iscomplexobj(target_xyz_raw):
        raise ValueError("target_points_must_be_real")
    target_xyz = np.asarray(target_xyz_raw, dtype=np.float64)
    if target_xyz.ndim != 2 or target_xyz.shape[1:] != (3,):
        raise ValueError("invalid_target_points")
    source_xy = _finite_array("source_pixels", source_pixels, (2,))
    target_xy = _finite_array("target_pixels", target_pixels, (2,))
    source_right_raw = np.asarray(source_right_u)
    target_right_raw = np.asarray(target_right_u)
    if np.iscomplexobj(source_right_raw) or np.iscomplexobj(target_right_raw):
        raise ValueError("right_measurements_must_be_real")
    if source_right_raw.ndim != 1 or target_right_raw.ndim != 1:
        raise ValueError("right_measurements_must_be_vectors")
    source_right = np.asarray(source_right_raw, dtype=np.float64)
    target_right = np.asarray(target_right_raw, dtype=np.float64)
    if len(source_right) != len(source_xyz) or len(source_xy) != len(source_xyz):
        raise ValueError("source_observation_length_mismatch")
    if len(target_right) != len(target_xy) or len(target_xyz) != len(target_xy):
        raise ValueError("target_observation_length_mismatch")
    K_raw = np.asarray(matrix)
    if np.iscomplexobj(K_raw):
        raise ValueError("matrix_must_be_real")
    K = np.asarray(K_raw, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("invalid_calibration_matrix")
    if (not np.isfinite(float(baseline)) or float(baseline) <= 0.0
            or not np.isfinite(float(disparity_offset))):
        raise ValueError("invalid_stereo_calibration")
    baseline, disparity_offset = float(baseline), float(disparity_offset)
    if (not isinstance(image_size, (tuple, list, np.ndarray))
            or np.asarray(image_size).shape != (2,)
            or np.asarray(image_size).dtype.kind not in "iu"
            or np.any(np.asarray(image_size) <= 0)):
        raise ValueError("invalid_image_size")
    image_size = tuple(int(x) for x in image_size)
    if (not isinstance(min_inliers, (int, np.integer)) or isinstance(min_inliers, (bool, np.bool_))
            or int(min_inliers) < 1):
        raise ValueError("invalid_min_inliers")
    min_inliers = int(min_inliers)
    if np.any(fit[:, 0] >= len(source_xyz)) or np.any(fit[:, 1] >= len(target_xy)):
        raise ValueError("fit_pair_index_out_of_range")
    fit_set = {(int(a), int(b)) for a, b in fit}
    if any((int(a), int(b)) not in fit_set for pair in (forward, reverse) for a, b in pair):
        raise ValueError("inlier_pair_not_in_original_fit")
    forward_set = {(int(a), int(b)) for a, b in forward}
    reverse_set = {(int(a), int(b)) for a, b in reverse}
    union = forward_set | reverse_set

    # The fit-pool endpoint graph itself must be one-to-one.  Checking only
    # after eligibility filtering could hide an invalid alias competitor.
    fit_source_keys = [_pixel_key(source_xy[a]) for a, _b in fit]
    fit_target_keys = [_pixel_key(target_xy[b]) for _a, b in fit]
    if len(set(fit_source_keys)) != len(fit_source_keys) or len(set(fit_target_keys)) != len(fit_target_keys):
        raise ValueError("ambiguous_duplicate_original_fit_endpoint")

    held = _integer_pairs("held_pairs", held_pairs) if held_pairs is not None else np.empty((0, 2), np.int64)
    if len(held) and (np.any(held[:, 0] >= len(source_xyz)) or np.any(held[:, 1] >= len(target_xy))):
        raise ValueError("held_pair_index_out_of_range")
    held_source_keys = {_pixel_key(source_xy[a]) for a, _b in held}
    held_target_keys = {_pixel_key(target_xy[b]) for _a, b in held}
    if any((int(a), int(b)) in fit_set for a, b in held):
        raise ValueError("fit_held_pair_overlap")

    excluded_ids_raw = np.asarray(tuple(excluded_landmark_ids))
    if np.iscomplexobj(excluded_ids_raw) or excluded_ids_raw.dtype.kind not in "iu":
        if excluded_ids_raw.size:
            raise ValueError("invalid_excluded_landmark_ids")
        excluded_ids = set()
    else:
        excluded_ids = {int(value) for value in excluded_ids_raw.reshape(-1) if int(value) >= 0}
    source_lm = None
    if source_landmark_ids is not None:
        source_lm_raw = np.asarray(source_landmark_ids)
        if (np.iscomplexobj(source_lm_raw) or source_lm_raw.dtype.kind not in "iu"
                or source_lm_raw.ndim != 1 or len(source_lm_raw) != len(source_xyz)):
            raise ValueError("invalid_source_landmark_ids")
        source_lm = np.asarray(source_lm_raw, dtype=np.int64)
    elif excluded_ids:
        raise ValueError("source_landmark_ids_required_for_held_exclusion")
    target_lm = None
    if target_landmark_ids is not None:
        target_lm_raw = np.asarray(target_landmark_ids)
        if (np.iscomplexobj(target_lm_raw) or target_lm_raw.dtype.kind not in "iu"
                or target_lm_raw.ndim != 1 or len(target_lm_raw) != len(target_xy)):
            raise ValueError("invalid_target_landmark_ids")
        target_lm = np.asarray(target_lm_raw, dtype=np.int64)

    fit_index = {(int(a), int(b)): i for i, (a, b) in enumerate(fit)}
    selected_indices = []
    selected_pairs = []
    drops = {"nonfinite_source_geometry_or_measurement": 0,
             "nonfinite_target_right_or_pixel": 0,
             "held_physical_endpoint_alias": 0,
             "held_landmark_identity": 0,
             "held_target_landmark_identity": 0}
    for row, (a_raw, b_raw) in enumerate(fit):
        a, b = int(a_raw), int(b_raw)
        pair = (a, b)
        if pair not in union:
            continue
        if (_pixel_key(source_xy[a]) in held_source_keys
                or _pixel_key(target_xy[b]) in held_target_keys):
            drops["held_physical_endpoint_alias"] += 1
            continue
        if source_lm is not None and int(source_lm[a]) in excluded_ids and int(source_lm[a]) >= 0:
            drops["held_landmark_identity"] += 1
            continue
        if target_lm is not None and int(target_lm[b]) in excluded_ids and int(target_lm[b]) >= 0:
            drops["held_target_landmark_identity"] += 1
            continue
        if (not np.isfinite(source_xyz[a]).all() or not np.isfinite(source_right[a])
                or not np.isfinite(source_xy[a]).all()):
            drops["nonfinite_source_geometry_or_measurement"] += 1
            continue
        if (not np.isfinite(target_xy[b]).all() or not np.isfinite(target_right[b])):
            drops["nonfinite_target_right_or_pixel"] += 1
            continue
        selected_indices.append(row)
        selected_pairs.append(pair)
    selected_pairs = np.asarray(selected_pairs, dtype=np.int64).reshape(-1, 2)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if len(selected_pairs) < min_inliers:
        raise ValueError("insufficient_selected_training_union")

    # Ambiguous physical correspondence endpoints must not silently receive
    # repeated weight in the bundle, even when descriptor IDs differ.
    src_keys = [_pixel_key(source_xy[a]) for a, _b in selected_pairs]
    dst_keys = [_pixel_key(target_xy[b]) for _a, b in selected_pairs]
    if len(set(src_keys)) != len(src_keys) or len(set(dst_keys)) != len(dst_keys):
        raise ValueError("ambiguous_duplicate_training_endpoint")

    reverse_rows = np.asarray([fit_index[pair] for pair in fit_index if pair in reverse_set], dtype=np.int64)
    forward_rows = np.asarray([fit_index[pair] for pair in fit_index if pair in forward_set], dtype=np.int64)
    forward_pairs_ordered = fit[forward_rows]
    reverse_pairs_ordered = fit[reverse_rows]
    if (len(forward_rows) < min_inliers or len(reverse_rows) < min_inliers
            or not np.isfinite(target_xyz[reverse_pairs_ordered[:, 1]]).all()):
        raise ValueError("original_directional_inlier_geometry_unavailable")
    if not np.isfinite(source_xyz[forward_pairs_ordered[:, 0]]).all():
        raise ValueError("original_forward_xyz_unavailable")
    reverse_raw_depth = target_xyz[reverse_pairs_ordered[:, 1], 2]
    if np.any(reverse_raw_depth <= _DEPTH_MIN_M) or np.any(reverse_raw_depth >= _DEPTH_MAX_M):
        raise ValueError("original_reverse_xyz_outside_declared_depth_domain")
    if (coverage(source_xy[selected_pairs[:, 0]], image_size) < 3
            or coverage(target_xy[selected_pairs[:, 1]], image_size) < 3):
        raise ValueError("insufficient_two_view_training_coverage")
    for left_pixels, right_values in (
        (source_xy[selected_pairs[:, 0]], source_right[selected_pairs[:, 0]]),
        (target_xy[selected_pairs[:, 1]], target_right[selected_pairs[:, 1]]),
    ):
        if (np.any(left_pixels[:, 0] < 0.0) or np.any(left_pixels[:, 0] >= image_size[0])
                or np.any(left_pixels[:, 1] < 0.0) or np.any(left_pixels[:, 1] >= image_size[1])
                or np.any(right_values < 0.0) or np.any(right_values >= image_size[0])):
            raise ValueError("selected_measurement_outside_image")
        observed_disparity = left_pixels[:, 0] - right_values
        metric_disparity = observed_disparity - disparity_offset
        if (not np.isfinite(metric_disparity).all()
                or np.any(observed_disparity <= 0.0) or np.any(observed_disparity >= 96.0)
                or np.any(metric_disparity <= 0.0)):
            raise ValueError("selected_observed_disparity_outside_existing_domain")
    source_depth = source_xyz[selected_pairs[:, 0], 2]
    if np.any(source_depth <= _DEPTH_MIN_M) or np.any(source_depth >= _DEPTH_MAX_M):
        raise ValueError("original_source_xyz_outside_declared_depth_domain")

    reverse_pose = _proper_pose("reverse_pose", reverse_pose)
    seed_reverse_motion = _matrix_error(reverse_pose @ seed)
    if (seed_reverse_motion[0] > _MAX_TRANSLATION_M
            or seed_reverse_motion[1] > _MAX_ROTATION_DEG):
        raise ValueError("original_seed_reverse_consistency_failed")
    x0 = np.r_[Rotation.from_matrix(seed[:3, :3]).as_rotvec(), seed[:3, 3],
               source_xyz[selected_pairs[:, 0]].reshape(-1)]
    problem = TwoViewStereoProblem(
        x0=np.array(x0, dtype=np.float64, copy=True),
        selected_pairs=np.array(selected_pairs, copy=True),
        selected_pair_indices=np.array(selected_indices, copy=True),
        source_points=np.array(source_xyz[selected_pairs[:, 0]], copy=True),
        target_points=np.array(target_xyz[selected_pairs[:, 1]], copy=True),
        source_pixels=np.array(source_xy[selected_pairs[:, 0]], copy=True),
        target_pixels=np.array(target_xy[selected_pairs[:, 1]], copy=True),
        source_right_u=np.array(source_right[selected_pairs[:, 0]], copy=True),
        target_right_u=np.array(target_right[selected_pairs[:, 1]], copy=True),
        seed_pose=seed, reverse_pose=reverse_pose, matrix=np.array(K, copy=True),
        baseline=baseline, disparity_offset=disparity_offset,
        image_size=image_size, min_inliers=min_inliers,
        direction_forward_pairs=np.array(forward_pairs_ordered, copy=True),
        direction_reverse_pairs=np.array(reverse_pairs_ordered, copy=True),
        direction_forward_source_points=np.array(source_xyz[forward_pairs_ordered[:, 0]], copy=True),
        direction_forward_target_pixels=np.array(target_xy[forward_pairs_ordered[:, 1]], copy=True),
        direction_reverse_target_points=np.array(target_xyz[reverse_pairs_ordered[:, 1]], copy=True),
        direction_reverse_source_pixels=np.array(source_xy[reverse_pairs_ordered[:, 0]], copy=True),
        selection_report={
            "selected_count": int(len(selected_pairs)),
            "fit_count": int(len(fit)),
            "forward_inlier_count": int(len(forward)),
            "reverse_inlier_count": int(len(reverse)),
            "union_count": int(len(union)),
            "selected_pair_indices": selected_indices.tolist(),
            "selected_pairs": selected_pairs.tolist(),
            "dropped_rows": drops,
            "held_pair_count": int(len(held)),
            "held_source_endpoint_keys": int(len(held_source_keys)),
            "held_target_endpoint_keys": int(len(held_target_keys)),
            "excluded_landmark_count": int(len(excluded_ids)),
            "original_seed_reverse_translation_m": seed_reverse_motion[0],
            "original_seed_reverse_rotation_deg": seed_reverse_motion[1],
            "original_seed_reverse_consistency_passed": True,
            "source_coverage_cells": int(coverage(source_xy[selected_pairs[:, 0]], image_size)),
            "target_coverage_cells": int(coverage(target_xy[selected_pairs[:, 1]], image_size)),
        })
    problem.x0.setflags(write=False)
    return problem


def refine_two_view_stereo_training(problem, reverse_pose, *, max_nfev=_MAX_NFEV):
    """Run a capped analytic sparse solve and return only a guarded pose candidate."""
    report = {
        "status": "rejected",
        "accepted": False,
        "reason": "not_started",
        "selected_pair_indices": problem.selected_pair_indices.tolist(),
        "selected_pairs": problem.selected_pairs.tolist(),
        "selected_count": int(problem.point_count),
        "residual_rows": int(problem.residual_rows),
        "variable_count": int(problem.variable_count),
        "initial_observability": None,
        "final_observability": None,
        "solver": {"success": False, "status": None, "message": "not_started",
                   "nfev": 0, "njev": 0, "optimality": None, "cost": None,
                   "initial_cost": None, "max_nfev": int(max_nfev),
                   "loss": "huber", "f_scale": _HUBER_F_SCALE,
                   "tr_solver": "lsmr", "jacobian": "analytic_sparse",
                   "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8,
                   "tr_options": {"regularize": True, "atol": 1e-6,
                                  "btol": 1e-6, "conlim": 1e8,
                                  "maxiter": int(problem.variable_count)},
                   "capped_not_converged": None,
                   "raw_scaled_huber_gradient_l2": None,
                   "raw_scaled_huber_gradient_linf": None,
                   "raw_scaled_gradient_below_gtol": None,
                   "optimizer_reported_optimality_below_sqrt_epsilon": None},
        "guards": {},
        "selection": problem.selection_report,
    }
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or int(max_nfev) != _MAX_NFEV:
        report["reason"] = "max_nfev_must_equal_declared_cap_15"
        return None, report
    try:
        reverse = _proper_pose("reverse_pose", reverse_pose)
        if not np.array_equal(reverse, problem.reverse_pose):
            raise ValueError("reverse_pose_binding_mismatch")
        problem._check_chart_domain(problem.x0)
        initial_residual = problem.residual(problem.x0)
        initial_cost = _huber_cost(initial_residual)
        report["solver"]["initial_cost"] = initial_cost
        initial_obs = problem.observability(problem.x0)
        report["initial_observability"] = initial_obs
        init_rank_ok = (initial_obs.get("status") == "observable"
                        and int(initial_obs.get("rank", -1)) == 6
                        and int(initial_obs.get("nullity", 1)) == 0
                        and initial_obs.get("per_point_rank_histogram") == {"3": problem.point_count})
        if not init_rank_ok:
            raise ValueError("initial_image_graph_unobservable")

        result = least_squares(
            problem.residual, problem.x0, jac=problem.jacobian,
            method="trf", tr_solver="lsmr", loss="huber",
            f_scale=_HUBER_F_SCALE, x_scale=problem.scale,
            max_nfev=_MAX_NFEV, ftol=1e-8, xtol=1e-8, gtol=1e-8,
            tr_options={"regularize": True, "atol": 1e-6, "btol": 1e-6,
                        "conlim": 1e8, "maxiter": int(problem.variable_count)},
        )
        solver_info = {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev),
            "njev": int(result.njev or 0),
            "optimality": float(result.optimality),
            "cost": float(result.cost), "initial_cost": initial_cost,
            "max_nfev": _MAX_NFEV, "loss": "huber", "f_scale": _HUBER_F_SCALE,
            "tr_solver": "lsmr", "jacobian": "analytic_sparse",
            "x_scale": "rotation_1rad_translation_and_xyz_baseline",
            "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8,
            "tr_options": {"regularize": True, "atol": 1e-6,
                           "btol": 1e-6, "conlim": 1e8,
                           "maxiter": int(problem.variable_count)},
            "status_interpretation": (
                "converged" if result.success else
                ("capped_not_converged" if int(result.status) == 0 else "solver_failure")),
        }
        report["solver"] = solver_info
        x = np.asarray(result.x, dtype=np.float64)
        if (x.shape != problem.x0.shape or not np.isfinite(x).all()
                or not np.isfinite(result.cost)
                or not np.isfinite(result.optimality)):
            raise ValueError("nonfinite_solver_result")
        problem._check_chart_domain(x)
        candidate, points = problem.decode(x)
        final_cost = _huber_cost(problem.residual(x))
        raw_jacobian = problem.jacobian(x)
        final_residual = problem.residual(x)
        psi = np.clip(final_residual, -_HUBER_F_SCALE, _HUBER_F_SCALE)
        scaled_gradient = problem.scale * np.asarray(raw_jacobian.T @ psi).reshape(-1)
        if not np.isfinite(scaled_gradient).all():
            raise ValueError("nonfinite_scaled_huber_gradient")
        solver_info["raw_scaled_huber_gradient_l2"] = float(np.linalg.norm(scaled_gradient))
        solver_info["raw_scaled_huber_gradient_linf"] = float(np.max(np.abs(scaled_gradient)))
        solver_info["optimizer_reported_optimality_below_sqrt_epsilon"] = bool(
            float(result.optimality) <= np.sqrt(np.finfo(np.float64).eps))
        solver_info["raw_scaled_gradient_below_gtol"] = bool(
            float(np.max(np.abs(scaled_gradient))) <= 1e-8)
        solver_info["capped_not_converged"] = bool(
            not result.success and int(result.status) == 0)
        solver_info["robust_initial_cost"] = initial_cost
        solver_info["robust_final_cost"] = final_cost
        final_obs = problem.observability(x)
        report["final_observability"] = final_obs
        final_rank_ok = (final_obs.get("status") == "observable"
                         and int(final_obs.get("rank", -1)) == 6
                         and int(final_obs.get("nullity", 1)) == 0
                         and final_obs.get("per_point_rank_histogram") == {"3": problem.point_count})
        reverse_move = _matrix_error(reverse @ candidate)
        seed_move = _motion_error(problem.seed_pose, candidate)
        directions = problem.direction_checks(x)
        guards = {
            "initial_depth_domain": True,
            "candidate_depth_domain": True,
            "final_image_graph_rank": bool(final_rank_ok),
            "forward_original_direction": directions["passed"],
            "reverse_original_direction": directions["passed"],
            "reverse_pose_translation_m": reverse_move[0],
            "reverse_pose_rotation_deg": reverse_move[1],
            "reverse_pose_motion_passed": bool(
                reverse_move[0] <= _MAX_TRANSLATION_M and reverse_move[1] <= _MAX_ROTATION_DEG),
            "seed_translation_m": seed_move[0],
            "seed_rotation_deg": seed_move[1],
            "seed_motion_passed": bool(
                seed_move[0] <= _MAX_TRANSLATION_M and seed_move[1] <= _MAX_ROTATION_DEG),
            "robust_cost_decreased": bool(final_cost < initial_cost),
            "directions": directions,
            "optimized_xyz_written_to_map": False,
            "heldout_used_in_fit": False,
        }
        report["guards"] = guards
        solver_usable = bool(result.success or int(result.status) == 0)
        guards["solver_returned_success"] = bool(result.success)
        guards["solver_status_usable"] = solver_usable
        gates = (
            solver_usable, bool(final_rank_ok), directions["passed"],
            guards["reverse_pose_motion_passed"], guards["seed_motion_passed"],
            guards["robust_cost_decreased"],
        )
        failed_gates = []
        if not solver_usable:
            failed_gates.append("solver_status")
        if not final_rank_ok:
            failed_gates.append("final_image_graph_rank")
        if not directions["passed"]:
            failed_gates.append("original_directional_support_or_reprojection")
        if not guards["reverse_pose_motion_passed"]:
            failed_gates.append("reverse_pose_consistency")
        if not guards["seed_motion_passed"]:
            failed_gates.append("seed_movement")
        if not guards["robust_cost_decreased"]:
            failed_gates.append("robust_cost_not_decreased")
        if not all(gates):
            guards["failed_gates"] = failed_gates
            raise ValueError("candidate_failed:" + ",".join(failed_gates))
        report["status"] = "accepted"
        report["accepted"] = True
        report["reason"] = (
            "finite_capped_iterate_passed_all_geometry_gates"
            if solver_info["capped_not_converged"]
            else "all_declared_two_view_training_gates_passed")
        return candidate, report
    except Exception as error:
        report["status"] = "rejected"
        report["accepted"] = False
        report["reason"] = str(error) or type(error).__name__
        return None, report

