"""Pure image-factor construction for audited reserved stereo training rows.

This module captures no poses from a map and performs no fitting.  Callers must
pass immutable supported-stereo endpoints and the exact row-ID ledgers emitted
by the forward and reverse fits.  The returned records are an offline math input,
not a calibrated covariance or an independent validation set.
"""
from dataclasses import dataclass
from collections import Counter, defaultdict
import hashlib
import math
import numpy as np

MAX_TRAINING_FACTORS = 256
FACTOR_MODEL = "correlated_physical_stereo_image_rows_v1"
FIT_SOURCE = "reserved_supported_training_rows"
FIT_DEPTH_POLICY = "supported_raw"


def _real_array(value, shape=None, name="array", copy=True):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name}_must_be_real_numeric")
    array = np.array(raw, dtype=np.float64, copy=copy)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"invalid_{name}")
    return array


def _integer(value, name, minimum=None):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))):
        raise ValueError(f"{name}_must_be_integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}_out_of_range")
    return result


def _real_scalar(value, name):
    raw = np.asarray(value)
    if raw.shape != () or np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name}_must_be_real_scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"invalid_{name}")
    return result


def _integer_tuple(value, length, name, minimum=0):
    raw = tuple(value)
    if len(raw) != length:
        raise ValueError(f"invalid_{name}")
    return tuple(_integer(x, name, minimum) for x in raw)


def _readonly(array, dtype=np.float64):
    value = np.ascontiguousarray(array, dtype=dtype)
    # A bytes-backed array cannot be made writable again by a recipient.
    result = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    result.setflags(write=False)
    return result


def _proper_se3(value, name="pose"):
    pose = _real_array(value, (4, 4), name)
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7, rtol=0.)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-7, rtol=0.)):
        raise ValueError(f"invalid_{name}_se3")
    return pose


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _so3_exp(phi):
    phi = _real_array(phi, (3,), "rotation_vector")
    theta = float(np.linalg.norm(phi))
    W = _skew(phi)
    if theta < 1e-8:
        return np.eye(3) + W + .5 * (W @ W)
    return np.eye(3) + (math.sin(theta) / theta) * W + ((1. - math.cos(theta)) / theta**2) * (W @ W)


def se3_exp(xi):
    """SE(3) exponential for right tangent order ``[rho_x,y,z, phi_x,y,z]``."""
    xi = _real_array(xi, (6,), "tangent")
    rho, phi = xi[:3], xi[3:]
    theta = float(np.linalg.norm(phi))
    W = _skew(phi)
    if theta < 1e-8:
        V = np.eye(3) + .5 * W + (1. / 6.) * (W @ W)
    else:
        V = (np.eye(3) + ((1. - math.cos(theta)) / theta**2) * W
             + ((theta - math.sin(theta)) / theta**3) * (W @ W))
    result = np.eye(4)
    result[:3, :3] = _so3_exp(phi)
    result[:3, 3] = V @ rho
    return result


def se3_adjoint(transform):
    """Adjoint for ``[rho, phi]`` tangent coordinates."""
    T = _proper_se3(transform, "adjoint_transform")
    R, t = T[:3, :3], T[:3, 3]
    return np.block([[R, _skew(t) @ R], [np.zeros((3, 3)), R]])


@dataclass(frozen=True)
class EndpointPose:
    """Accepted image-frame pose and its live keyframe anchor at one map epoch."""
    frame_id: int
    pose: np.ndarray
    anchor_keyframe_id: int
    anchor_pose: np.ndarray
    status: str
    source_epoch: tuple

    def __post_init__(self):
        frame = _integer(self.frame_id, "frame_id", 0)
        anchor = _integer(self.anchor_keyframe_id, "anchor_keyframe_id", 0)
        epoch = _integer_tuple(self.source_epoch, 2, "source_epoch", 0)
        pose = _proper_se3(self.pose, "endpoint_pose")
        anchor_pose = _proper_se3(self.anchor_pose, "anchor_pose")
        if self.status not in {"tracking", "relocalized", "accepted"}:
            raise ValueError("endpoint_status_not_accepted")
        relative = np.linalg.inv(anchor_pose) @ pose
        _proper_se3(relative, "anchor_relative_pose")
        object.__setattr__(self, "frame_id", frame)
        object.__setattr__(self, "anchor_keyframe_id", anchor)
        object.__setattr__(self, "source_epoch", epoch)
        object.__setattr__(self, "pose", _readonly(pose))
        object.__setattr__(self, "anchor_pose", _readonly(anchor_pose))

    @property
    def anchor_relative_pose(self):
        return _readonly(np.linalg.inv(self.anchor_pose) @ self.pose)

    @property
    def anchor_frame_id(self):
        """Compatibility alias; this is a keyframe ID, not an image frame ID."""
        return self.anchor_keyframe_id


@dataclass(frozen=True)
class StereoTrainingFactor:
    """One unambiguous physical track and its two actual image measurements."""
    source_frame: int
    target_frame: int
    source_anchor: int
    target_anchor: int
    source_anchor_relative: np.ndarray
    target_anchor_relative: np.ndarray
    source_pose: np.ndarray
    target_pose: np.ndarray
    source_anchor_pose: np.ndarray
    target_anchor_pose: np.ndarray
    source_pixel: np.ndarray
    target_pixel: np.ndarray
    source_right_u: float
    target_right_u: float
    source_row: int
    target_row: int
    source_alias_rows: tuple
    target_alias_rows: tuple
    forward_role: bool
    reverse_role: bool
    source_landmark_id: int
    target_landmark_id: int
    reused_landmark_id: int
    source_has_existing_observation: bool
    target_has_existing_observation: bool
    point_initial: np.ndarray
    matrix: np.ndarray
    baseline: float
    disparity_offset: float
    image_size: tuple
    calibration_identity: str
    source_epoch: tuple
    fit_source: str
    fit_depth_policy: str
    fit_partition: str
    heldout_status: str
    physical_identity: str
    model: str = FACTOR_MODEL
    covariance_claim: bool = False
    heldout_validation_claim: bool = False

    def __post_init__(self):
        for name in ("source_frame", "target_frame", "source_anchor", "target_anchor", "source_row", "target_row"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 0))
        for name in ("source_landmark_id", "target_landmark_id", "reused_landmark_id"):
            value = _integer(getattr(self, name), name, -1)
            object.__setattr__(self, name, value)
        for name in ("source_has_existing_observation", "target_has_existing_observation"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"invalid_{name}")
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("source_anchor_relative", "target_anchor_relative", "source_pose", "target_pose",
                     "source_anchor_pose", "target_anchor_pose"):
            object.__setattr__(self, name, _readonly(_proper_se3(getattr(self, name), name)))
        for name in ("source_pixel", "target_pixel"):
            object.__setattr__(self, name, _readonly(_real_array(getattr(self, name), (2,), name), np.float64))
        object.__setattr__(self, "point_initial", _readonly(_real_array(self.point_initial, (3,), "point_initial")))
        object.__setattr__(self, "matrix", _readonly(_real_array(self.matrix, (3, 3), "matrix")))
        for name in ("source_right_u", "target_right_u", "baseline", "disparity_offset"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"invalid_{name}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_alias_rows", tuple(_integer(x, "source_alias_row", 0) for x in self.source_alias_rows))
        object.__setattr__(self, "target_alias_rows", tuple(_integer(x, "target_alias_row", 0) for x in self.target_alias_rows))
        object.__setattr__(self, "image_size", _integer_tuple(self.image_size, 2, "image_size", 1))
        object.__setattr__(self, "source_epoch", _integer_tuple(self.source_epoch, 2, "source_epoch", 0))
        if self.source_frame == self.target_frame or self.source_anchor_relative.shape != (4, 4):
            raise ValueError("invalid_factor_endpoints")
        if self.fit_source != FIT_SOURCE or self.fit_depth_policy != FIT_DEPTH_POLICY:
            raise ValueError("invalid_factor_provenance")
        if self.fit_partition != "reserved_training" or self.heldout_status != "excluded_external_arbitration_rows":
            raise ValueError("invalid_factor_partition")
        if self.model != FACTOR_MODEL or self.covariance_claim is not False or self.heldout_validation_claim is not False:
            raise ValueError("invalid_factor_certification_metadata")
        if not isinstance(self.calibration_identity, str) or not self.calibration_identity:
            raise ValueError("missing_calibration_identity")
        if not isinstance(self.physical_identity, str) or len(self.physical_identity) != 64:
            raise ValueError("invalid_physical_identity")

    def to_dict(self):
        """JSON-safe immutable provenance; it contains neither a map object nor GT."""
        arrays = ("source_anchor_relative", "target_anchor_relative", "source_pose", "target_pose",
                  "source_anchor_pose", "target_anchor_pose", "source_pixel", "target_pixel",
                  "point_initial", "matrix")
        result = {name: getattr(self, name).tolist() for name in arrays}
        for name in self.__dataclass_fields__:
            if name in arrays:
                continue
            value = getattr(self, name)
            result[name] = list(value) if isinstance(value, tuple) else value
        return result


def _frame_arrays(frame, name):
    frame_id = _integer(getattr(frame, "frame", None), f"{name}_frame", 0)
    size = _integer_tuple(getattr(frame, "image_size", None), 2, f"{name}_image_size", 1)
    pixels = np.asarray(getattr(frame, "pixels", None))
    points = np.asarray(getattr(frame, "points", None))
    right_u = np.asarray(getattr(frame, "right_u", None))
    landmark_ids = np.asarray(getattr(frame, "landmark_ids", None))
    if (np.iscomplexobj(pixels) or np.iscomplexobj(points) or np.iscomplexobj(right_u)
            or not np.issubdtype(pixels.dtype, np.number) or not np.issubdtype(points.dtype, np.number)
            or not np.issubdtype(right_u.dtype, np.number)):
        raise ValueError(f"{name}_arrays_must_be_real_numeric")
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or points.shape != (len(pixels), 3)
            or right_u.shape != (len(pixels),) or landmark_ids.shape != (len(pixels),)
            or not np.issubdtype(landmark_ids.dtype, np.integer) or np.issubdtype(landmark_ids.dtype, np.bool_)
            or np.any(landmark_ids < -1)):
        raise ValueError(f"invalid_{name}_frame_shapes")
    calibration_identity = getattr(frame, "calibration_identity", None)
    if not isinstance(calibration_identity, str) or not calibration_identity:
        raise ValueError(f"missing_{name}_calibration_identity")
    pixels32 = np.asarray(pixels, dtype=np.float32)
    if not np.isfinite(pixels32).all():
        raise ValueError(f"invalid_{name}_pixels")
    points64 = np.asarray(points, dtype=np.float64)
    right64 = np.asarray(right_u, dtype=np.float64)
    return {
        "frame": frame_id, "size": size, "pixels": pixels32,
        "points": points64, "right_u": right64, "landmark_ids": landmark_ids.astype(np.int64, copy=False),
        "calibration_identity": calibration_identity,
    }


def _pixel_key(pixel):
    p = np.asarray(pixel, dtype=np.float32)
    # Python float keys compare exact float32 values; canonicalize signed zero.
    return tuple(0.0 if float(x) == 0.0 else float(x) for x in p)


def _pair_array(value, name, rows_source, rows_target):
    raw = np.asarray(value)
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if (raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name}_must_be_integer_Nx2")
    pairs = raw.astype(np.int64, copy=True)
    if np.any(pairs < 0) or np.any(pairs[:, 0] >= rows_source) or np.any(pairs[:, 1] >= rows_target):
        raise ValueError(f"{name}_row_out_of_range")
    return pairs


def _validated_calibration(matrix, baseline, offset, image_size):
    K = _real_array(matrix, (3, 3), "calibration_matrix")
    if (K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0., 0., 1.], atol=1e-10, rtol=0.)
            or not np.isclose(K[1, 0], 0., atol=1e-10, rtol=0.)):
        raise ValueError("invalid_calibration_matrix")
    baseline = _real_scalar(baseline, "baseline")
    offset = _real_scalar(offset, "disparity_offset")
    if baseline <= 0:
        raise ValueError("invalid_stereo_calibration")
    return K, baseline, offset, _integer_tuple(image_size, 2, "image_size", 1)


def _obs_projection(p, K, baseline, offset):
    x, y, z = p
    if not np.isfinite(p).all() or z <= 0:
        raise ValueError("invalid_point_cheirality")
    fx, skew, cx = K[0]
    fy, cy = K[1, 1], K[1, 2]
    u = fx*x/z + skew*y/z + cx
    v = fy*y/z + K[1, 2]
    right_u = u - fx*baseline/z - offset
    Jproj = np.array([
        [fx/z, skew/z, -(fx*x + skew*y)/(z*z)],
        [0., fy/z, -fy*y/(z*z)],
        [fx/z, skew/z, -(fx*x + skew*y)/(z*z) + fx*baseline/(z*z)],
    ])
    return np.array([u, v, right_u]), Jproj


def stereo_projection(camera_point, matrix, baseline, disparity_offset=0.):
    """Return ``[u,v,right_u]`` and the 3x3 camera-point Jacobian."""
    K, b, off, _ = _validated_calibration(matrix, baseline, disparity_offset, (1, 1))
    return _obs_projection(_real_array(camera_point, (3,), "camera_point"), K, b, off)


def _transported_pose(anchor_pose, anchor_relative_pose):
    return _proper_se3(anchor_pose, "live_anchor_pose") @ _proper_se3(anchor_relative_pose, "anchor_relative_pose")


def _pose_jacobian(camera_point, Jproj, camera_to_world, anchor_relative):
    local = np.column_stack((-np.eye(3), _skew(camera_point)))
    return Jproj @ local @ se3_adjoint(np.linalg.inv(anchor_relative))


def build_stereo_training_factors(
    source, target, source_endpoint, target_endpoint, *, matrix, baseline,
    disparity_offset=0., calibration_identity, fit_pairs,
    forward_inlier_pairs, reverse_inlier_pairs, heldout_pairs=(),
    fit_source=FIT_SOURCE, fit_depth_policy=FIT_DEPTH_POLICY,
    fit_partition="reserved_training", holdout_status="excluded_external_arbitration_rows",
    consumed_full_pool=False, reservation_context=True,
    existing_observations=(), existing_landmark_points=None, maximum=MAX_TRAINING_FACTORS,
):
    """Create bounded factors from actual, unique forward/reverse training edges.

    Row pairs are exact ``(source feature row, target feature row)`` indices from
    the already-run mutual matcher.  No descriptors are rematched and no pose is
    fit here.  Held-out physical endpoints are removed before edge canonicalizing.
    Returns ``(tuple[factor], report)``; all metadata failures reject augmentation.
    """
    report = {
        "status": "rejected", "reason": None, "candidate_edges": 0,
        "selected_factors": 0, "cap": None, "truncated": 0,
        "rejections": {}, "model": FACTOR_MODEL, "covariance_claim": False,
        "intentionally_reuses_sensor_evidence": True,
        "heldout_validation_claim": False, "pose_fit_performed": False,
        "selection_policy": "lexicographic_exact_float32_physical_endpoints_cap256_v1",
        "fit_source": fit_source, "fit_depth_policy": fit_depth_policy,
        "fit_partition": fit_partition, "holdout_status": holdout_status,
    }
    def reject(reason):
        report["reason"] = reason
        report["status"] = "rejected"
        return tuple(), report
    try:
        if (fit_source != FIT_SOURCE or fit_depth_policy != FIT_DEPTH_POLICY
                or fit_partition != "reserved_training"
                or holdout_status != "excluded_external_arbitration_rows"
                or consumed_full_pool or reservation_context is not True):
            return reject("unusable_training_partition")
        if not isinstance(calibration_identity, str) or not calibration_identity:
            return reject("missing_calibration_identity")
        src, dst = _frame_arrays(source, "source"), _frame_arrays(target, "target")
        if src["frame"] == dst["frame"] or src["size"] != dst["size"]:
            return reject("invalid_endpoint_identity_or_image_size")
        if (src["calibration_identity"] != calibration_identity
                or dst["calibration_identity"] != calibration_identity):
            return reject("calibration_identity_mismatch")
        epoch = source_endpoint.source_epoch
        if epoch != target_endpoint.source_epoch:
            return reject("endpoint_epoch_mismatch")
        if source_endpoint.frame_id != src["frame"] or target_endpoint.frame_id != dst["frame"]:
            return reject("pose_frame_identity_mismatch")
        if source_endpoint.status not in {"tracking", "relocalized", "accepted"} or target_endpoint.status not in {"tracking", "relocalized", "accepted"}:
            return reject("endpoint_status_not_accepted")
        K, b, off, size = _validated_calibration(matrix, baseline, disparity_offset, src["size"])
        if size != src["size"]:
            return reject("calibration_image_size_mismatch")
        if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, (int, np.integer)) or not 1 <= int(maximum) <= MAX_TRAINING_FACTORS:
            return reject("invalid_factor_cap")
        maximum = int(maximum)
        src_pose = _proper_se3(source_endpoint.pose, "source_pose")
        dst_pose = _proper_se3(target_endpoint.pose, "target_pose")
        src_anchor_pose = _proper_se3(source_endpoint.anchor_pose, "source_anchor_pose")
        dst_anchor_pose = _proper_se3(target_endpoint.anchor_pose, "target_anchor_pose")
        src_rel = _proper_se3(np.linalg.inv(src_anchor_pose) @ src_pose, "source_anchor_relative")
        dst_rel = _proper_se3(np.linalg.inv(dst_anchor_pose) @ dst_pose, "target_anchor_relative")
        fit = _pair_array(fit_pairs, "fit_pairs", len(src["pixels"]), len(dst["pixels"]))
        fwd = _pair_array(forward_inlier_pairs, "forward_inlier_pairs", len(src["pixels"]), len(dst["pixels"]))
        rev = _pair_array(reverse_inlier_pairs, "reverse_inlier_pairs", len(src["pixels"]), len(dst["pixels"]))
        held = _pair_array(heldout_pairs, "heldout_pairs", len(src["pixels"]), len(dst["pixels"]))
        fit_set = {tuple(map(int, pair)) for pair in fit}
        if not len(fwd) or not len(rev) or any(tuple(map(int, pair)) not in fit_set for pair in np.concatenate((fwd, rev), axis=0)):
            return reject("missing_or_unbound_bidirectional_training_rows")
        roles = defaultdict(set)
        for pair in fwd:
            roles[tuple(map(int, pair))].add("forward")
        for pair in rev:
            roles[tuple(map(int, pair))].add("reverse")
        held_source, held_target = set(), set()
        for i, j in held:
            held_source.add(_pixel_key(src["pixels"][i]));held_target.add(_pixel_key(dst["pixels"][j]))
        # Group appearances by exact float32 pixel identity, preserving all row IDs.
        src_aliases, dst_aliases = defaultdict(list), defaultdict(list)
        for i, pixel in enumerate(src["pixels"]): src_aliases[_pixel_key(pixel)].append(i)
        for j, pixel in enumerate(dst["pixels"]): dst_aliases[_pixel_key(pixel)].append(j)
        observations = {}
        owner_pixels = defaultdict(set)
        owner_rows_by_measurement = defaultdict(set)
        for item in existing_observations:
            if not isinstance(item, dict):
                return reject("malformed_existing_observation")
            try:
                key = (_integer(item.get("frame_id"), "observation_frame", 0), _integer(item.get("landmark_id"), "observation_landmark", 0))
                pixel = np.asarray(item.get("pixel"), dtype=np.float32)
                right = np.asarray(item.get("right_u"))
                if (pixel.shape != (2,) or not np.isfinite(pixel).all() or np.iscomplexobj(right)
                        or not np.issubdtype(right.dtype, np.number) or np.issubdtype(right.dtype, np.bool_)
                        or right.size != 1 or not np.isfinite(right.astype(float)).all()):
                    return reject("malformed_existing_observation")
                entry = (_pixel_key(pixel), float(np.float32(right.reshape(-1)[0])))
                if key in observations and observations[key] != entry:
                    return reject("conflicting_existing_observation_identity")
                observations[key] = entry
                owner_pixels[(key[0], key[1])].add(entry[0])
                owner_rows_by_measurement[(key[0], entry[0], entry[1])].add(key[1])
            except (ValueError, TypeError, OverflowError):
                return reject("malformed_existing_observation")
        if existing_landmark_points is None:
            point_lookup = {}
        elif hasattr(existing_landmark_points, "items"):
            point_lookup = dict(existing_landmark_points)
        else:
            return reject("malformed_existing_landmark_points")
        # Actual observation owners, unlike detector/flow feature labels, are
        # certified by exact (frame, float32 pixel, right-u) identity.
        ambiguous_ids = set()
        for (frame_id, lm), pixels_for_lm in owner_pixels.items():
            if len(pixels_for_lm) > 1:
                ambiguous_ids.add(lm)
        candidates = defaultdict(lambda: {"pairs": [], "roles": set()})
        rejection_counts = Counter()
        all_rows = sorted(set(roles))
        report["candidate_edges"] = len(all_rows)
        for i, j in all_rows:
            sk, tk = _pixel_key(src["pixels"][i]), _pixel_key(dst["pixels"][j])
            if sk in held_source or tk in held_target:
                rejection_counts["heldout_physical_endpoint"] += 1
                continue
            candidates[(sk, tk)]["pairs"].append((i, j))
            candidates[(sk, tk)]["roles"].update(roles[(i, j)])
        source_degrees, target_degrees = defaultdict(set), defaultdict(set)
        for sk, tk in candidates:
            source_degrees[sk].add(tk);target_degrees[tk].add(sk)
        records = []
        for (sk, tk), edge in candidates.items():
            if len(source_degrees[sk]) > 1 or len(target_degrees[tk]) > 1:
                rejection_counts["ambiguous_physical_correspondence"] += 1
                continue
            pair_rows = sorted(edge["pairs"])
            i, j = pair_rows[0]
            sa, ta = tuple(src_aliases[sk]), tuple(dst_aliases[tk])
            source_claims = sorted({int(src["landmark_ids"][r]) for r in sa if int(src["landmark_ids"][r]) >= 0})
            target_claims = sorted({int(dst["landmark_ids"][r]) for r in ta if int(dst["landmark_ids"][r]) >= 0})
            if len(source_claims) > 1 or len(target_claims) > 1:
                rejection_counts["ambiguous_landmark_identity"] += 1
                continue
            # Exact pixel aliases must describe one sensor sample, including right-u and source initializer.
            s_rights = np.asarray([src["right_u"][r] for r in sa], dtype=np.float32)
            t_rights = np.asarray([dst["right_u"][r] for r in ta], dtype=np.float32)
            if (not np.isfinite(s_rights).all() or not np.isfinite(t_rights).all()
                    or len(np.unique(s_rights)) > 1 or len(np.unique(t_rights)) > 1):
                rejection_counts["conflicting_alias_measurement"] += 1
                continue
            source_pixel, target_pixel = src["pixels"][i], dst["pixels"][j]
            width, height = size
            valid_sensor_rows = (
                np.all((source_pixel >= [0., 0.]) & (source_pixel < [width, height]))
                and np.all((target_pixel >= [0., 0.]) & (target_pixel < [width, height]))
                and 0. <= float(s_rights[0]) < width and 0. <= float(t_rights[0]) < width
                and source_pixel[0] - float(s_rights[0]) - off > 0.
                and target_pixel[0] - float(t_rights[0]) - off > 0.
            )
            if not valid_sensor_rows:
                rejection_counts["invalid_sensor_geometry"] += 1
                continue
            source_owners = set()
            target_owners = set()
            for lm, frame_id, pixel_key, right_value, owner_set in (
                (source_claims[0] if source_claims else -1, src["frame"], sk, float(s_rights[0]), source_owners),
                (target_claims[0] if target_claims else -1, dst["frame"], tk, float(t_rights[0]), target_owners),
            ):
                owner_set.update(owner_rows_by_measurement.get((frame_id, pixel_key, float(np.float32(right_value))), set()))
                # A feature ID is only a claim. It is reusable only with the exact
                # actual observation certificate for this measured pixel/right-u.
                if lm >= 0:
                    expected = (pixel_key, float(np.float32(right_value)))
                    prior = observations.get((frame_id, lm))
                    if ((prior is not None and prior != expected)
                            or (owner_set and lm not in owner_set)):
                        owner_set.add(-2)  # conflicting claim, handled below
            if -2 in source_owners or -2 in target_owners or len(source_owners - {-2}) > 1 or len(target_owners - {-2}) > 1:
                rejection_counts["conflicting_endpoint_landmark_ids"] += 1
                continue
            source_lms = sorted(source_owners - {-2})
            target_lms = sorted(target_owners - {-2})
            if any(lm in ambiguous_ids for lm in source_lms + target_lms):
                rejection_counts["ambiguous_landmark_identity"] += 1
                continue
            source_lm = source_lms[0] if source_lms else -1
            target_lm = target_lms[0] if target_lms else -1
            if source_lm >= 0 and target_lm >= 0 and source_lm != target_lm:
                rejection_counts["conflicting_endpoint_landmark_ids"] += 1
                continue
            if (source_claims and source_lm < 0) or (target_claims and target_lm < 0):
                rejection_counts["unverified_landmark_claim_ignored"] += 1
            source_points = src["points"][list(sa)]
            finite_source = np.isfinite(source_points).all(axis=1)
            if not np.any(finite_source):
                rejection_counts["missing_source_initializer"] += 1
                continue
            source_point = source_points[np.flatnonzero(finite_source)[0]]
            if np.any(finite_source) and not np.array_equal(source_points[finite_source], np.broadcast_to(source_point, source_points[finite_source].shape)):
                rejection_counts["conflicting_source_geometry_alias"] += 1
                continue
            if source_point[2] <= 0:
                rejection_counts["invalid_source_initializer"] += 1
                continue
            # Any claimed existing point must bind to this exact endpoint observation.
            reused = source_lm if source_lm >= 0 else target_lm
            if reused >= 0:
                if reused not in point_lookup:
                    rejection_counts["missing_existing_landmark_point"] += 1
                    continue
                world_initial = _real_array(point_lookup[reused], (3,), "existing_landmark_point")
            else:
                world_initial = src_pose[:3, :3] @ source_point + src_pose[:3, 3]
            if not np.isfinite(world_initial).all():
                rejection_counts["invalid_world_point_initializer"] += 1
                continue
            h = hashlib.sha256()
            for endpoint_key in (sk, tk):
                h.update(np.asarray(endpoint_key, dtype="<f4").tobytes())
            identity = h.hexdigest()
            records.append(StereoTrainingFactor(
                source_frame=src["frame"], target_frame=dst["frame"],
                source_anchor=source_endpoint.anchor_frame_id, target_anchor=target_endpoint.anchor_frame_id,
                source_anchor_relative=src_rel, target_anchor_relative=dst_rel,
                source_pose=src_pose, target_pose=dst_pose,
                source_anchor_pose=src_anchor_pose, target_anchor_pose=dst_anchor_pose,
                source_pixel=src["pixels"][i], target_pixel=dst["pixels"][j],
                source_right_u=float(s_rights[0]), target_right_u=float(t_rights[0]),
                source_row=i, target_row=j, source_alias_rows=sa, target_alias_rows=ta,
                forward_role="forward" in edge["roles"], reverse_role="reverse" in edge["roles"],
                source_landmark_id=source_lm, target_landmark_id=target_lm,
                reused_landmark_id=reused,
                source_has_existing_observation=source_lm >= 0,
                target_has_existing_observation=target_lm >= 0,
                point_initial=world_initial,
                matrix=K, baseline=b, disparity_offset=off, image_size=size,
                calibration_identity=calibration_identity, source_epoch=epoch,
                fit_source=fit_source, fit_depth_policy=fit_depth_policy,
                fit_partition=fit_partition, heldout_status=holdout_status,
                physical_identity=identity,
            ))
        records.sort(key=lambda f: (tuple(f.source_pixel), tuple(f.target_pixel), f.source_row, f.target_row))
        report["cap"] = maximum
        report["eligible_before_cap"] = len(records)
        report["truncated"] = max(0, len(records) - maximum)
        records = records[:maximum]
        report["selected_factors"] = len(records)
        report["rejections"] = dict(sorted(rejection_counts.items()))
        report["physical_identity_sha256"] = hashlib.sha256("".join(f.physical_identity for f in records).encode()).hexdigest()
        report["status"] = "accepted" if records else "rejected"
        report["reason"] = None if records else "no_unambiguous_supported_training_rows"
        report["source_epoch"] = list(epoch)
        return tuple(records), report
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        return reject(str(exc) or "invalid_factor_metadata")


def _factor_observation_rows(factor, *, anchor_poses=None, point_world=None):
    anchor_poses = {} if anchor_poses is None else anchor_poses
    X = factor.point_initial if point_world is None else _real_array(point_world, (3,), "point_world")
    endpoint_items = (
        (factor.source_frame, factor.source_anchor, factor.source_anchor_relative,
         factor.source_pixel, factor.source_right_u, factor.source_landmark_id,
         factor.source_anchor_pose),
        (factor.target_frame, factor.target_anchor, factor.target_anchor_relative,
         factor.target_pixel, factor.target_right_u, factor.target_landmark_id,
         factor.target_anchor_pose),
    )
    output = []
    for frame_id, anchor_id, relative, pixel, right_u, landmark_id, frozen_anchor in endpoint_items:
        anchor = _proper_se3(anchor_poses.get(anchor_id, frozen_anchor), "live_anchor_pose")
        T = _transported_pose(anchor, relative)
        p = T[:3, :3].T @ (X - T[:3, 3])
        predicted, Jproj = _obs_projection(p, factor.matrix, factor.baseline, factor.disparity_offset)
        observed = np.array([pixel[0], pixel[1], right_u], dtype=np.float64)
        residual = predicted - observed
        Janchor = _pose_jacobian(p, Jproj, T, relative)
        Jpoint = Jproj @ T[:3, :3].T
        if not np.isfinite(residual).all() or not np.isfinite(Janchor).all() or not np.isfinite(Jpoint).all():
            raise ValueError("nonfinite_factor_linearization")
        output.append({
            "frame_id": frame_id, "anchor_id": anchor_id,
            "pixel_key": _pixel_key(pixel), "right_u": float(right_u),
            "landmark_id": landmark_id, "residual": residual,
            "anchor_jacobian": Janchor, "point_jacobian": Jpoint,
            "pose": T,
        })
    return output


def linearize_training_factor(factor, *, source_anchor_pose=None, target_anchor_pose=None,
                              point_world=None, huber_delta=2.):
    """Evaluate six actual pixel residuals and analytic anchor/point Jacobians."""
    if not isinstance(factor, StereoTrainingFactor):
        raise ValueError("invalid_training_factor")
    delta = _real_scalar(huber_delta, "huber_delta")
    if delta <= 0:
        raise ValueError("invalid_huber_delta")
    anchors = {factor.source_anchor: factor.source_anchor_pose,
               factor.target_anchor: factor.target_anchor_pose}
    if source_anchor_pose is not None:
        anchors[factor.source_anchor] = source_anchor_pose
    if target_anchor_pose is not None:
        anchors[factor.target_anchor] = target_anchor_pose
    rows = _factor_observation_rows(factor, anchor_poses=anchors, point_world=point_world)
    residual = np.concatenate([row["residual"] for row in rows])
    point_jac = np.vstack([row["point_jacobian"] for row in rows])
    anchor_jac = {}
    for row_index, row in enumerate(rows):
        aid = row["anchor_id"]
        if aid not in anchor_jac:
            anchor_jac[aid] = np.zeros((6, 6), dtype=np.float64)
        anchor_jac[aid][3*row_index:3*row_index+3] += row["anchor_jacobian"]
    absolute = np.abs(residual)
    cost = float(np.sum(np.where(absolute <= delta, .5*residual*residual, delta*(absolute-.5*delta))))
    weights = np.minimum(1., delta/np.maximum(absolute, 1e-15))
    return {
        "residual": residual, "anchor_jacobians": anchor_jac,
        "point_jacobian": point_jac, "huber_weights": weights,
        "objective": cost, "model": FACTOR_MODEL,
        "covariance_claim": False, "heldout_validation_claim": False,
    }


def factor_schur_information(factor, *, anchor_poses=None, free_anchor_ids=None,
                             point_world=None, huber_delta=2.):
    """Return one track's Huber-IRLS camera Schur block after eliminating X.

    Fixed endpoint cameras still constrain the nuisance point.  Translation and
    rotation columns use the declared ``[rho,phi]`` right-tangent convention;
    no eigenvalue floor or pose prior is added.
    """
    anchor_poses = {} if anchor_poses is None else dict(anchor_poses)
    source_anchor_pose = anchor_poses.get(factor.source_anchor, factor.source_anchor_pose)
    target_anchor_pose = anchor_poses.get(factor.target_anchor, factor.target_anchor_pose)
    lin = linearize_training_factor(
        factor, source_anchor_pose=source_anchor_pose,
        target_anchor_pose=target_anchor_pose, point_world=point_world,
        huber_delta=huber_delta,
    )
    order = tuple(dict.fromkeys((factor.source_anchor, factor.target_anchor)))
    free = order if free_anchor_ids is None else tuple(_integer(x, "free_anchor_id", 0) for x in free_anchor_ids)
    if len(set(free)) != len(free) or any(x not in order for x in free):
        raise ValueError("invalid_free_anchor_ids")
    weights = lin["huber_weights"]
    sqrt_w = np.sqrt(weights)
    Jx = lin["point_jacobian"] * sqrt_w[:, None]
    Jc_blocks = [lin["anchor_jacobians"][fid] * sqrt_w[:, None] for fid in free]
    Jc = np.concatenate(Jc_blocks, axis=1) if Jc_blocks else np.zeros((6, 0))
    schur = _project_point_nuisance(Jc, Jx)
    metric = np.tile([factor.baseline]*3 + [1.]*3, len(free))
    dimensionless = metric[:, None] * schur * metric[None, :] if schur.size else schur
    eigenvalues = np.linalg.eigvalsh(.5*(dimensionless+dimensionless.T)) if dimensionless.size else np.zeros(0)
    rank = _psd_rank(eigenvalues, dimensionless.shape[0])
    return {
        "anchor_order": list(free), "information": schur,
        "eigenvalues": eigenvalues, "rank": rank,
        "factor_objective": lin["objective"], "residual": lin["residual"],
        "information_kind": "gauss_newton_huber_irls_schur_after_point_nuisance",
        "rank_tolerance": "machine_epsilon_times_dimension_times_largest_dimensionless_eigenvalue",
        "translation_scale_m": factor.baseline,
        "model": FACTOR_MODEL, "covariance_claim": False,
        "heldout_validation_claim": False,
    }


def _schur_group(rows, point_initial, free, huber_delta):
    if not rows:
        return np.zeros((6*len(free), 6*len(free))), 0.
    # A physical point is one nuisance variable. Duplicate identity rows are
    # coalesced before this function; one map point can occur once per view.
    seen, frame_points, unique = {}, {}, []
    for row in rows:
        key = (row["frame_id"], row["pixel_key"])
        value = (row["right_u"], int(row.get("landmark_id", -1)))
        if key in seen:
            old = seen[key]
            if np.float32(old[0]) != np.float32(value[0]) or (old[1] >= 0 and value[1] >= 0 and old[1] != value[1]):
                raise ValueError("conflicting_duplicate_physical_observation")
            continue
        seen[key] = value
        if row["frame_id"] in frame_points and frame_points[row["frame_id"]] != row["pixel_key"]:
            raise ValueError("landmark_reused_at_multiple_pixels_in_one_view")
        frame_points[row["frame_id"]] = row["pixel_key"]
        unique.append(row)
    residual = np.concatenate([row["residual"] for row in unique])
    Jx = np.vstack([row["point_jacobian"] for row in unique])
    Jc = np.zeros((3*len(unique), 6*len(free)), dtype=np.float64)
    for r, row in enumerate(unique):
        if row["anchor_id"] in free:
            c = free.index(row["anchor_id"])
            Jc[3*r:3*r+3, 6*c:6*c+6] += row["anchor_jacobian"]
    abs_r = np.abs(residual)
    weights = np.minimum(1., huber_delta/np.maximum(abs_r, 1e-15))
    sw = np.sqrt(weights)
    Jx *= sw[:, None]
    Jc *= sw[:, None]
    Hcc = _project_point_nuisance(Jc, Jx)
    objective = float(np.sum(np.where(abs_r <= huber_delta, .5*residual*residual,
                                      huber_delta*(abs_r-.5*huber_delta))))
    return Hcc, objective


def _project_point_nuisance(Jcamera, Jpoint):
    """Schur-complement square-root projection with only roundoff rank handling."""
    if Jcamera.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if Jpoint.shape[1] == 0 or Jpoint.shape[0] == 0:
        return Jcamera.T @ Jcamera
    U, singular, _ = np.linalg.svd(Jpoint, full_matrices=False)
    largest = float(singular[0]) if len(singular) else 0.
    tol = np.finfo(np.float64).eps * max(Jpoint.shape) * largest
    rank = int(np.count_nonzero(singular > tol))
    if rank:
        basis = U[:, :rank]
        projected = Jcamera - basis @ (basis.T @ Jcamera)
    else:
        projected = Jcamera
    result = projected.T @ projected
    return .5*(result+result.T)


def _psd_rank(eigenvalues, dimension):
    values = np.asarray(eigenvalues, dtype=np.float64)
    largest = float(np.max(np.abs(values), initial=0.))
    tol = np.finfo(np.float64).eps * max(1, int(dimension)) * largest
    return int(np.count_nonzero(values > tol))


def _base_observation_row(item, factor, anchor_poses):
    if not isinstance(item, dict):
        raise ValueError("malformed_base_observation")
    frame = _integer(item.get("frame_id"), "base_frame_id", 0)
    landmark = _integer(item.get("landmark_id"), "base_landmark_id", 0)
    anchor = _integer(item.get("anchor_id", item.get("anchor_frame_id")), "base_anchor_id", 0)
    pixel = np.asarray(item.get("pixel"), dtype=np.float32)
    raw_right = np.asarray(item.get("right_u"))
    if (pixel.shape != (2,) or not np.isfinite(pixel).all() or np.iscomplexobj(raw_right)
            or not np.issubdtype(raw_right.dtype, np.number) or np.issubdtype(raw_right.dtype, np.bool_)
            or raw_right.size != 1):
        raise ValueError("malformed_base_observation")
    right = float(np.float32(raw_right.reshape(-1)[0]))
    if not np.isfinite(right):
        raise ValueError("malformed_base_observation")
    anchor_pose = _proper_se3(anchor_poses.get(anchor, item.get("anchor_pose")), "base_anchor_pose")
    if "anchor_relative_pose" in item:
        relative = _proper_se3(item["anchor_relative_pose"], "base_anchor_relative_pose")
    else:
        camera_pose = _proper_se3(item.get("camera_pose"), "base_camera_pose")
        relative = _proper_se3(np.linalg.inv(anchor_pose) @ camera_pose, "base_anchor_relative_pose")
    T = _transported_pose(anchor_pose, relative)
    X = _real_array(item.get("point_world"), (3,), "base_point_world")
    p = T[:3, :3].T @ (X-T[:3, 3])
    pred, Jproj = _obs_projection(p, factor.matrix, factor.baseline, factor.disparity_offset)
    observed = np.array([pixel[0], pixel[1], right])
    return {
        "frame_id": frame, "anchor_id": anchor, "pixel_key": _pixel_key(pixel),
        "right_u": right, "landmark_id": landmark, "residual": pred-observed,
        "anchor_jacobian": _pose_jacobian(p, Jproj, T, relative),
        "point_jacobian": Jproj @ T[:3, :3].T, "pose": T,
        "point_world": X,
    }


def accumulate_schur_information(factors, *, anchor_poses=None, free_anchor_ids=(),
                                  huber_delta=2., base_observations=()):
    """Accumulate factors, combining rows that share one persistent point ID.

    Optional ``base_observations`` use the schema documented by
    ``_base_observation_row``.  A target residual already present in the normal
    bundle window is deduplicated by exact ``(frame,float32 pixel,right-u)``;
    it is never counted twice before eliminating the shared 3D nuisance.
    """
    factors = tuple(factors)
    free = tuple(_integer(x, "free_anchor_id", 0) for x in free_anchor_ids)
    if len(set(free)) != len(free):
        raise ValueError("duplicate_free_anchor_ids")
    delta = _real_scalar(huber_delta, "huber_delta")
    if delta <= 0:
        raise ValueError("invalid_huber_delta")
    anchors = {} if anchor_poses is None else dict(anchor_poses)
    groups, point_by_group = defaultdict(list), {}
    factor_group = {}
    total_rows = 0
    for factor in factors:
        if not isinstance(factor, StereoTrainingFactor):
            raise ValueError("invalid_training_factor")
        key = ("landmark", factor.reused_landmark_id) if factor.reused_landmark_id >= 0 else ("physical", factor.physical_identity)
        factor_group[factor.physical_identity] = key
        point = factor.point_initial
        if key in point_by_group and not np.array_equal(point_by_group[key], point):
            raise ValueError("shared_landmark_point_snapshot_mismatch")
        point_by_group[key] = point
        for row in _factor_observation_rows(factor, anchor_poses=anchors, point_world=point):
            row["point_world"] = point
            groups[key].append(row)
            total_rows += 1
    base_rows_count = 0
    for item in base_observations:
        if not isinstance(item, dict):
            raise ValueError("malformed_base_observation")
        landmark = _integer(item.get("landmark_id"), "base_landmark_id", 0)
        key = ("landmark", landmark)
        if key not in point_by_group:
            continue
        factor = next(f for f in factors if f.reused_landmark_id == landmark)
        row = _base_observation_row(item, factor, anchors)
        if not np.array_equal(row["point_world"], point_by_group[key]):
            raise ValueError("base_and_raw_point_snapshot_mismatch")
        groups[key].append(row)
        base_rows_count += 1
    total = np.zeros((6*len(free), 6*len(free)), dtype=np.float64)
    objective = 0.
    used = 0
    for key, rows in groups.items():
        block, cost = _schur_group(rows, point_by_group[key], free, delta)
        total += block
        objective += cost
        used += 1
    total = .5*(total+total.T)
    translation_scale = factors[0].baseline if factors else 1.0
    if factors and any(f.calibration_identity != factors[0].calibration_identity
                       or f.baseline != translation_scale
                       or f.disparity_offset != factors[0].disparity_offset
                       or not np.array_equal(f.matrix, factors[0].matrix) for f in factors):
        raise ValueError("mixed_calibration_factor_pool")
    metric = np.tile([translation_scale]*3+[1.]*3, len(free))
    dimensionless = metric[:, None]*total*metric[None, :] if total.size else total
    values = np.linalg.eigvalsh(.5*(dimensionless+dimensionless.T)) if dimensionless.size else np.zeros(0)
    return {
        "anchor_order": list(free), "information": total,
        "eigenvalues": values, "rank": _psd_rank(values, dimensionless.shape[0]),
        "objective": float(objective), "factor_count": len(factors),
        "point_group_count": len(groups), "raw_observation_rows": total_rows,
        "base_observation_rows_supplied": base_rows_count,
        "translation_scale_m": translation_scale,
        "rank_tolerance": "machine_epsilon_times_dimension_times_largest_dimensionless_eigenvalue",
        "information_kind": "grouped_gauss_newton_huber_irls_schur_after_shared_point_nuisance",
        "model": FACTOR_MODEL, "covariance_claim": False,
        "heldout_validation_claim": False,
    }


def _marginal_block(information, keep, translation_scale_m=1.0):
    H = np.asarray(information, dtype=np.float64)
    keep = tuple(keep)
    nuisance = tuple(i for i in range(H.shape[0]) if i not in keep)
    A = H[np.ix_(keep, keep)]
    if nuisance:
        B = H[np.ix_(keep, nuisance)]
        C = H[np.ix_(nuisance, nuisance)]
        scale = float(translation_scale_m)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid_translation_scale")
        if H.shape[0] % 6:
            raise ValueError("pose_hessian_dimension_not_multiple_of_6")
        unit = np.tile([scale]*3+[1.]*3, H.shape[0]//6)
        D = unit[list(nuisance)]
        Cdim = D[:, None] * C * D[None, :]
        Bdim = B * D[None, :]
        values, vectors = np.linalg.eigh(.5*(Cdim+Cdim.T))
        largest = float(np.max(np.abs(values), initial=0.))
        tol = np.finfo(np.float64).eps * max(Cdim.shape) * largest
        keep_eigen = values > tol
        if np.any(keep_eigen):
            Cplus = (vectors[:, keep_eigen] / values[keep_eigen]) @ vectors[:, keep_eigen].T
            A = A - Bdim @ Cplus @ Bdim.T
    return .5*(A+A.T)

def motion_rotation_information(summary, *, source_anchor, target_anchor,
                                source_anchor_relative, target_anchor_relative,
                                relative_pose):
    """Separate newest-camera absolute rotation from source-to-target relative rotation.

    The input ``summary`` is the anchor-coordinate result of
    ``accumulate_schur_information``.  The relative spectrum reparameterizes
    target motion as ``xi_t = xi_rel + Ad_(Z^-1) Ad_(C_s^-1) xi_source_anchor``
    before eliminating source pose and relative translation nuisance.
    """
    order = tuple(summary["anchor_order"])
    H = _real_array(summary["information"], (6*len(order), 6*len(order)), "anchor_information")
    source_anchor = _integer(source_anchor, "source_anchor", 0)
    target_anchor = _integer(target_anchor, "target_anchor", 0)
    if target_anchor not in order:
        return {"absolute_newest_rotation": None, "relative_rotation": None,
                "reason": "target_anchor_not_free", "covariance_claim": False}
    src_rel = _proper_se3(source_anchor_relative, "source_anchor_relative")
    dst_rel = _proper_se3(target_anchor_relative, "target_anchor_relative")
    Z = _proper_se3(relative_pose, "relative_pose")
    target_block = [6*order.index(target_anchor)+i for i in range(6)]
    scale = float(summary.get("translation_scale_m", 1.0))
    abs_H = _marginal_block(H, target_block, scale)
    D_inv = np.linalg.inv(se3_adjoint(np.linalg.inv(dst_rel)))
    Mabs = D_inv
    Habs_camera = Mabs.T @ abs_H @ Mabs
    abs_rot = _marginal_block(Habs_camera, (3, 4, 5), scale)
    if source_anchor == target_anchor:
        return {"absolute_newest_rotation": {"information": abs_rot,
                    "eigenvalues": np.linalg.eigvalsh(abs_rot), "rank": _psd_rank(np.linalg.eigvalsh(abs_rot), 3),
                    "coordinate_order": ["phi_x", "phi_y", "phi_z"]},
                "relative_rotation": None, "reason": "shared_anchor_relative_pose_is_fixed",
                "covariance_claim": False}
    src_free = source_anchor in order
    if src_free:
        endpoints = (source_anchor, target_anchor)
        indices = [6*order.index(a)+i for a in endpoints for i in range(6)]
        Hpair = _marginal_block(H, indices, scale)
        S = se3_adjoint(np.linalg.inv(src_rel))
        A = se3_adjoint(np.linalg.inv(Z))
        D = se3_adjoint(np.linalg.inv(dst_rel))
        D_inv = np.linalg.inv(D)
        M = np.block([[np.eye(6), np.zeros((6, 6))], [D_inv @ A @ S, D_inv]])
        Hrelative_coordinates = M.T @ Hpair @ M
        relative_rot = _marginal_block(Hrelative_coordinates, tuple(range(9, 12)), scale)
    else:
        # A fixed source contributes residual information but no source columns;
        # target camera relative and absolute coordinates differ only by its fixed transport.
        relative_rot = abs_rot.copy()
    def spectrum(block):
        vals = np.linalg.eigvalsh(block)
        return {"information": block, "eigenvalues": vals,
                "rank": _psd_rank(vals, block.shape[0]),
                "coordinate_order": ["phi_x", "phi_y", "phi_z"]}
    return {"absolute_newest_rotation": spectrum(abs_rot),
            "relative_rotation": spectrum(relative_rot),
            "reason": None, "covariance_claim": False,
            "information_kind": "marginalized_huber_irls_schur_rotation_information"}
