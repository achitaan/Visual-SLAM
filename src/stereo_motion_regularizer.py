"""Local pixel-evidence curvature for an opt-in stereo-motion regularizer.

This factor intentionally reuses stereo observations already consumed by pose
verification and bundle reprojection. It is a correlated-evidence regularizer,
not an independent measurement or a calibrated pose covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MODEL = "correlated_pixel_motion_regularizer"
NOISE_MODEL = "independent_1px_u_v_disparity_per_endpoint_v1"
TANGENT_ORDER = ("rho_x", "rho_y", "rho_z", "phi_x", "phi_y", "phi_z")


def _real_array(value, *, name):
    """Convert real numeric input without silently discarding complex parts."""
    raw = np.asarray(value)
    if (not np.issubdtype(raw.dtype, np.number)
            or np.issubdtype(raw.dtype, np.complexfloating)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(raw, dtype=np.float64)


def _readonly_array(value, *, name):
    array = np.array(_real_array(value, name=name), dtype=np.float64,
                     order="C", copy=True)
    # An ndarray that merely has WRITEABLE cleared can have it re-enabled when
    # it owns its allocation. Backing this view with immutable bytes prevents
    # both item assignment and setflags(write=True).
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _readonly_pairs(value, *, name):
    raw = np.asarray(value)
    if (not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name} must contain integer row IDs")
    array = np.array(raw, dtype=np.int64, order="C", copy=True)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(array.shape)


def _integer_tuple(value, *, name, length, positive=False):
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a {length}-integer tuple") from error
    if (len(values) != length
            or any(not isinstance(item, (int, np.integer))
                   or isinstance(item, (bool, np.bool_)) for item in values)):
        raise ValueError(f"{name} must be a {length}-integer tuple")
    values = tuple(int(item) for item in values)
    if any(item < (1 if positive else 0) for item in values):
        raise ValueError(f"{name} contains an invalid value")
    return values


def _real_scalar(value, *, name):
    raw = np.asarray(value)
    if (raw.shape != () or not np.issubdtype(raw.dtype, np.number)
            or np.issubdtype(raw.dtype, np.complexfloating)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class StereoMotionRegularizer:
    """Immutable relative-pose curvature and its raw observation provenance.

    ``measurement`` maps target-camera coordinates into source-camera
    coordinates and therefore composes as ``T_source @ measurement``. The
    stored square root satisfies ``sqrt_information.T @ sqrt_information ==
    information`` in right-tangent ``[rho, phi]`` coordinates.
    """

    measurement: np.ndarray
    sqrt_information: np.ndarray
    information: np.ndarray
    matrix: np.ndarray
    baseline: float
    disparity_offset: float
    source_frame: int
    target_frame: int
    source_epoch: tuple[int, int]
    source_status: str
    target_status: str
    image_size: tuple[int, int]
    calibration_identity: str
    training_pairs: np.ndarray
    forward_inlier_pairs: np.ndarray
    reverse_inlier_pairs: np.ndarray
    information_pairs: np.ndarray
    training_source_pixels: np.ndarray
    training_target_pixels: np.ndarray
    training_source_points: np.ndarray
    training_target_points: np.ndarray
    source_pixels: np.ndarray
    target_pixels: np.ndarray
    source_right_u: np.ndarray
    target_right_u: np.ndarray
    source_points: np.ndarray
    target_points: np.ndarray
    fit_source: str
    fit_depth_policy: str
    role: str
    rank: int
    dimensionless_singular_values: tuple[float, ...]
    condition_number: float
    model: str = MODEL
    noise_model: str = NOISE_MODEL
    tangent_order: tuple[str, ...] = TANGENT_ORDER
    covariance_claim: bool = False
    intentionally_reuses_sensor_evidence: bool = True
    holdout_rows_used: bool = False
    ownership_certificate: str = "none"

    def __post_init__(self):
        pair_names = ("training_pairs", "forward_inlier_pairs", "reverse_inlier_pairs",
                      "information_pairs")
        array_names = (
            "measurement", "sqrt_information", "information", "matrix",
            "training_source_pixels", "training_target_pixels",
            "training_source_points", "training_target_points", "source_pixels",
            "target_pixels", "source_right_u", "target_right_u", "source_points",
            "target_points",
        )
        for name in pair_names:
            object.__setattr__(self, name, _readonly_pairs(getattr(self, name), name=name))
        for name in array_names:
            object.__setattr__(self, name, _readonly_array(getattr(self, name), name=name))

        for name in ("source_status", "target_status", "calibration_identity", "fit_source",
                     "fit_depth_policy", "role", "model", "noise_model",
                     "ownership_certificate"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        for name in ("calibration_identity", "fit_source", "role"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.source_status not in ("tracking", "relocalized") or self.target_status not in (
                "tracking", "relocalized"):
            raise ValueError("factor endpoints must have accepted tracking states")
        if self.fit_depth_policy != "supported_raw":
            raise ValueError("factor must identify raw supported depth")
        if (self.model != MODEL or self.noise_model != NOISE_MODEL
                or tuple(self.tangent_order) != TANGENT_ORDER):
            raise ValueError("factor model metadata does not match this implementation")
        if (self.covariance_claim is not False
                or self.intentionally_reuses_sensor_evidence is not True
                or self.holdout_rows_used is not False
                or self.ownership_certificate != "none"):
            raise ValueError("factor evidence semantics cannot claim independence or covariance")

        source_frame = _integer_tuple((self.source_frame,), name="source_frame", length=1)[0]
        target_frame = _integer_tuple((self.target_frame,), name="target_frame", length=1)[0]
        if source_frame < 0 or not 1 <= target_frame - source_frame <= 3:
            raise ValueError("factor endpoints must be ordered within three frames")
        epoch = _integer_tuple(self.source_epoch, name="source_epoch", length=2)
        image_size = _integer_tuple(self.image_size, name="image_size", length=2, positive=True)
        baseline = _real_scalar(self.baseline, name="baseline")
        offset = _real_scalar(self.disparity_offset, name="disparity_offset")
        condition = _real_scalar(self.condition_number, name="condition_number")
        if baseline <= 0. or condition < 1.:
            raise ValueError("baseline and information condition must be positive")
        if (not isinstance(self.rank, (int, np.integer))
                or isinstance(self.rank, (bool, np.bool_)) or int(self.rank) != 6):
            raise ValueError("regularizer information must have full rank six")
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "source_epoch", epoch)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "disparity_offset", offset)
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "rank", 6)

        singular_values = _real_array(self.dimensionless_singular_values,
                                      name="dimensionless_singular_values")
        if (singular_values.shape != (6,) or not np.isfinite(singular_values).all()
                or np.any(singular_values <= 0.)):
            raise ValueError("dimensionless singular values must be six finite positive values")
        singular_values = tuple(float(x) for x in singular_values)
        object.__setattr__(self, "dimensionless_singular_values", singular_values)

        if not _proper_se3(self.measurement):
            raise ValueError("measurement must be a proper finite SE(3) transform")
        if not _intrinsic_is_valid(self.matrix):
            raise ValueError("matrix must be a valid finite camera intrinsic matrix")
        if (self.sqrt_information.shape != (6, 6) or self.information.shape != (6, 6)
                or not np.isfinite(self.sqrt_information).all()
                or not np.isfinite(self.information).all()):
            raise ValueError("information and square-root information must be finite 6x6 matrices")
        if not np.allclose(self.information, self.information.T, rtol=1e-10, atol=1e-12):
            raise ValueError("information must be symmetric")
        scale = np.diag([baseline] * 3 + [1.] * 3)
        dimensionless = scale.T @ self.information @ scale
        dimensionless = 0.5 * (dimensionless + dimensionless.T)
        actual_singular_values = np.linalg.svd(dimensionless, compute_uv=False)
        eigenvalues = np.linalg.eigvalsh(dimensionless)
        if (not np.isfinite(actual_singular_values).all()
                or eigenvalues[0] <= 0.
                or not np.allclose(actual_singular_values, singular_values,
                                   rtol=1e-7, atol=1e-10)
                or not np.isclose(actual_singular_values[0] / actual_singular_values[-1],
                                  condition, rtol=1e-7, atol=1e-10)):
            raise ValueError("information rank/conditioning metadata is inconsistent")
        sqrt_dimensionless = self.sqrt_information @ scale
        if not np.allclose(sqrt_dimensionless.T @ sqrt_dimensionless,
                           dimensionless, rtol=2e-7, atol=1e-9):
            raise ValueError("square-root information is inconsistent with information")

        training = self.training_pairs
        forward = self.forward_inlier_pairs
        reverse = self.reverse_inlier_pairs
        information_pairs = self.information_pairs
        if (training.ndim != 2 or training.shape[1:] != (2,) or len(training) < 6
                or forward.ndim != 2 or forward.shape[1:] != (2,)
                or reverse.ndim != 2 or reverse.shape[1:] != (2,)
                or information_pairs.ndim != 2 or information_pairs.shape[1:] != (2,)
                or len(information_pairs) < 6
                or any(np.any(array < 0) for array in (training, forward, reverse,
                                                       information_pairs))):
            raise ValueError("factor row provenance has invalid pair tables")
        sets = [{tuple(row) for row in pairs.tolist()} for pairs in (training, forward, reverse)]
        if (len(sets[0]) != len(training) or len(sets[1]) != len(forward)
                or len(sets[2]) != len(reverse) or not sets[1] <= sets[0]
                or not sets[2] <= sets[0]):
            raise ValueError("factor directional pairs are inconsistent with training pairs")
        expected_info = np.asarray(sorted(sets[1] & sets[2]), dtype=np.int64).reshape(-1, 2)
        if not np.array_equal(information_pairs, expected_info):
            raise ValueError("information rows must be exactly the bidirectional inlier intersection")

        count = len(training)
        if (self.training_source_pixels.shape != (count, 2)
                or self.training_target_pixels.shape != (count, 2)
                or self.training_source_points.shape != (count, 3)
                or self.training_target_points.shape != (count, 3)):
            raise ValueError("raw training snapshots do not match pair table")
        if (not np.isfinite(self.training_source_pixels).all()
                or not np.isfinite(self.training_target_pixels).all()
                or np.isinf(self.training_source_points).any()
                or np.isinf(self.training_target_points).any()):
            raise ValueError("training snapshots contain invalid pixel/point values")
        if (np.any(self.training_source_pixels < 0.)
                or np.any(self.training_target_pixels < 0.)
                or np.any(self.training_source_pixels[:, 0] >= image_size[0])
                or np.any(self.training_target_pixels[:, 0] >= image_size[0])
                or np.any(self.training_source_pixels[:, 1] >= image_size[1])
                or np.any(self.training_target_pixels[:, 1] >= image_size[1])):
            raise ValueError("training pixels fall outside endpoint image bounds")
        training_index = {tuple(pair): index for index, pair in enumerate(training.tolist())}
        rows = np.asarray([training_index[tuple(pair)] for pair in information_pairs.tolist()],
                          dtype=np.int64)
        if (self.source_pixels.shape != (len(information_pairs), 2)
                or self.target_pixels.shape != (len(information_pairs), 2)
                or self.source_right_u.shape != (len(information_pairs),)
                or self.target_right_u.shape != (len(information_pairs),)
                or self.source_points.shape != (len(information_pairs), 3)
                or self.target_points.shape != (len(information_pairs), 3)):
            raise ValueError("information snapshots do not match selected pair rows")
        evidence_arrays = (self.source_pixels, self.target_pixels, self.source_right_u,
                           self.target_right_u, self.source_points, self.target_points)
        if any(not np.isfinite(array).all() for array in evidence_arrays):
            raise ValueError("information snapshots must be finite")
        if (not np.array_equal(self.source_pixels, self.training_source_pixels[rows])
                or not np.array_equal(self.target_pixels, self.training_target_pixels[rows])
                or not np.array_equal(self.source_points, self.training_source_points[rows])
                or not np.array_equal(self.target_points, self.training_target_points[rows])):
            raise ValueError("information evidence does not match captured training rows")
        if (np.any(self.source_pixels < 0.) or np.any(self.target_pixels < 0.)
                or np.any(self.source_pixels[:, 0] >= image_size[0])
                or np.any(self.target_pixels[:, 0] >= image_size[0])
                or np.any(self.source_pixels[:, 1] >= image_size[1])
                or np.any(self.target_pixels[:, 1] >= image_size[1])
                or np.any(self.source_right_u < 0.) or np.any(self.target_right_u < 0.)
                or np.any(self.source_right_u >= image_size[0])
                or np.any(self.target_right_u >= image_size[0])
                or np.any(self.source_points[:, 2] <= 0.)
                or np.any(self.target_points[:, 2] <= 0.)):
            raise ValueError("information observations violate image or depth bounds")


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rotation_log(rotation):
    rotation = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.) * 0.5, -1., 1.))
    theta = math.acos(cosine)
    skew_vector = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if theta < 1e-7:
        # theta/(2 sin(theta)) expanded around zero.
        theta2 = theta * theta
        return 0.5 * (1. + theta2 / 6. + 7. * theta2 * theta2 / 360.) * skew_vector
    if math.pi - theta > 1e-5:
        return (theta / (2. * math.sin(theta))) * skew_vector

    # The antisymmetric formula is ill-conditioned near pi. Recover an axis
    # from the symmetric part and choose its sign from the available skew.
    diagonal = np.maximum((np.diag(rotation) + 1.) * 0.5, 0.)
    pivot = int(np.argmax(diagonal))
    axis = np.zeros(3)
    axis[pivot] = math.sqrt(float(diagonal[pivot]))
    if axis[pivot] <= np.finfo(float).eps:
        raise ValueError("rotation logarithm is undefined at this matrix")
    for index in range(3):
        if index != pivot:
            axis[index] = (rotation[index, pivot] + rotation[pivot, index]) / (4. * axis[pivot])
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.:
        raise ValueError("rotation logarithm produced an invalid axis")
    axis /= norm
    if np.linalg.norm(skew_vector) > 1e-10 and np.dot(axis, skew_vector) < 0.:
        axis = -axis
    return theta * axis


def se3_log(transform):
    """Return right-tangent ``[rho, phi]`` with ``rho = V(phi)^-1 t``."""
    transform = _real_array(transform, name="transform")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.)):
        raise ValueError("SE(3) transform must be finite 4x4 homogeneous matrix")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0.)):
        raise ValueError("SE(3) rotation must be a proper rotation")
    phi = _rotation_log(rotation)
    theta = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    theta2 = theta * theta
    if theta < 1e-5:
        coefficient = 1. / 12. + theta2 / 720. + theta2 * theta2 / 30240.
    else:
        coefficient = (1. / theta2
                       - 1. / (2. * theta * math.tan(theta * 0.5)))
    v_inverse = np.eye(3) - 0.5 * phi_hat + coefficient * (phi_hat @ phi_hat)
    rho = v_inverse @ transform[:3, 3]
    result = np.r_[rho, phi]
    if not np.isfinite(result).all():
        raise ValueError("SE(3) logarithm is non-finite")
    return result


def stereo_motion_residual(factor: StereoMotionRegularizer,
                           source_pose: np.ndarray,
                           target_pose: np.ndarray) -> np.ndarray:
    """Evaluate the whitened edge residual on caller-propagated endpoint poses."""
    source_pose = np.asarray(source_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    if source_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError("Endpoint poses must be 4x4")
    relative_error = np.linalg.inv(factor.measurement) @ np.linalg.inv(source_pose) @ target_pose
    return factor.sqrt_information @ se3_log(relative_error)


def _projection_jacobian(point: np.ndarray, matrix: np.ndarray, baseline: float) -> np.ndarray:
    x, y, z = _real_array(point, name="point").reshape(3)
    matrix = _real_array(matrix, name="matrix")
    fx, skew_xy, cx = matrix[0]
    fy = matrix[1, 1]
    if z <= 0. or not np.isfinite([x, y, z]).all():
        raise ValueError("measurement point must have positive finite depth")
    common = fx * x + skew_xy * y
    return np.array([
        [fx / z, skew_xy / z, -common / (z * z)],
        [0., fy / z, -fy * y / (z * z)],
        [fx / z, skew_xy / z, -(common - fx * baseline) / (z * z)],
    ])


def _proper_se3(transform):
    try:
        value = _real_array(transform, name="measurement")
    except (TypeError, ValueError):
        return False
    return bool(value.shape == (4, 4) and np.isfinite(value).all()
                and np.allclose(value[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.)
                and np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3),
                                atol=1e-7, rtol=0.)
                and np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-7, rtol=0.))


def _pair_array(value, name, source_length, target_length, *, allow_empty=False):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1:] != (2,)
            or not np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.bool_)
            or (not allow_empty and len(array) == 0)):
        raise ValueError(f"invalid_{name}_shape_or_type")
    array = array.astype(np.int64, copy=True)
    if (np.any(array < 0) or np.any(array[:, 0] >= source_length)
            or np.any(array[:, 1] >= target_length)):
        raise ValueError(f"{name}_row_out_of_range")
    if len({tuple(row) for row in array.tolist()}) != len(array):
        raise ValueError(f"duplicate_{name}_rows")
    return array


def _exact_array_equal_allow_nan(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        return False
    return bool(np.all((first == second) | (np.isnan(first) & np.isnan(second))))


def _intrinsic_is_valid(matrix):
    try:
        matrix = _real_array(matrix, name="matrix")
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.shape == (3, 3) and np.isfinite(matrix).all()
        and matrix[0, 0] > 0. and matrix[1, 1] > 0.
        and np.isclose(matrix[1, 0], 0., atol=1e-12, rtol=0.)
        and np.allclose(matrix[2], [0., 0., 1.], atol=1e-10, rtol=0.)
    )


def _build_information(source_points, target_points, matrix, baseline, offset,
                       measurement):
    # The single-endpoint noise covariance in (u,v,right_u), induced by
    # independent one-pixel noise in (u,v,disparity), is correlated across u.
    covariance = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 2.]])
    endpoint_whitener = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(3))
    whitener = np.zeros((6, 6))
    whitener[:3, :3] = endpoint_whitener
    whitener[3:, 3:] = endpoint_whitener
    rotation, translation = measurement[:3, :3], measurement[:3, 3]
    information = np.zeros((6, 6), dtype=float)
    for source_point, target_point in zip(source_points, target_points):
        if (not np.isfinite(source_point).all() or not np.isfinite(target_point).all()
                or source_point[2] <= 0. or target_point[2] <= 0.):
            raise ValueError("invalid_positive_depth_training_measurement")
        predicted_target = rotation.T @ (source_point - translation)
        if not np.isfinite(predicted_target).all() or predicted_target[2] <= 0.:
            raise ValueError("measurement_pose_has_invalid_target_cheirality")

        source_jacobian = _projection_jacobian(source_point, matrix, baseline)
        target_jacobian = _projection_jacobian(predicted_target, matrix, baseline)
        pose_jacobian = target_jacobian @ np.c_[-np.eye(3), _skew(predicted_target)]
        point_jacobian = np.vstack((source_jacobian, target_jacobian @ rotation.T))
        pose_jacobian = whitener @ np.vstack((np.zeros((3, 6)), pose_jacobian))
        point_jacobian = whitener @ point_jacobian
        # QR projection is the stable Schur complement without forming or
        # inverting the nuisance-point normal matrix.
        q, r = np.linalg.qr(point_jacobian, mode="reduced")
        singular = np.linalg.svd(r, compute_uv=False)
        tolerance = max(point_jacobian.shape) * np.finfo(float).eps * singular[0]
        if int(np.sum(singular > tolerance)) != 3:
            raise ValueError("point_jacobian_rank_deficient")
        marginalized_pose = pose_jacobian - q @ (q.T @ pose_jacobian)
        information += marginalized_pose.T @ marginalized_pose
    information = 0.5 * (information + information.T)
    if not np.isfinite(information).all():
        raise ValueError("nonfinite_information")
    return information


def build_stereo_motion_regularizer(
    source,
    target,
    measurement,
    *,
    training_pairs,
    forward_inlier_pairs,
    reverse_inlier_pairs,
    training_source_pixels,
    training_target_pixels,
    training_source_points,
    training_target_points,
    matrix,
    baseline,
    disparity_offset,
    source_epoch,
    source_status,
    target_status,
    fit_source,
    fit_depth_policy,
    role,
):
    """Build a fail-closed factor from the exact raw rows used by pose fitting.

    Returns ``(factor, None)`` or ``(None, reason)``. Information uses only
    pairs that are inliers in both already-completed forward and reverse PnP
    checks. The factor remains explicitly correlated with the observations used
    by stereo verification and bundle reprojection.
    """
    try:
        baseline = _real_scalar(baseline, name="baseline")
        disparity_offset = _real_scalar(disparity_offset, name="disparity_offset")
        if source is None or target is None:
            return None, "invalid_pose_or_calibration"
        # Own the validated measurement/calibration snapshots before any
        # information calculation; caller mutation cannot change a factor
        # midway through construction.
        matrix = np.array(_real_array(matrix, name="matrix"), dtype=np.float64,
                          order="C", copy=True)
        measurement = np.array(_real_array(measurement, name="measurement"), dtype=np.float64,
                               order="C", copy=True)
        if (not _proper_se3(measurement) or not _intrinsic_is_valid(matrix)
                or baseline <= 0.):
            return None, "invalid_pose_or_calibration"
        source_frame = _integer_tuple((source.frame,), name="source_frame", length=1)[0]
        target_frame = _integer_tuple((target.frame,), name="target_frame", length=1)[0]
        if (source_frame < 0 or target_frame <= source_frame
                or target_frame - source_frame > 3):
            return None, "invalid_endpoint_frames"
        if (source_status not in ("tracking", "relocalized")
                or target_status not in ("tracking", "relocalized")):
            return None, "endpoint_not_accepted"
        epoch = tuple(source_epoch)
        if (len(epoch) != 2 or any(not isinstance(x, (int, np.integer))
                                  or isinstance(x, (bool, np.bool_)) or x < 0 for x in epoch)):
            return None, "invalid_source_epoch"
        if fit_depth_policy != "supported_raw":
            return None, "fit_depth_policy_not_raw_supported"
        if not isinstance(fit_source, str) or not fit_source or not isinstance(role, str) or not role:
            return None, "invalid_fit_provenance"
        calibration_identity = source.calibration_identity
        if (not isinstance(calibration_identity, str) or not calibration_identity
                or target.calibration_identity != calibration_identity):
            return None, "calibration_identity_mismatch"
        image_size = tuple(source.image_size)
        if (len(image_size) != 2 or tuple(target.image_size) != image_size
                or any(not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_))
                       or size <= 0 for size in image_size)):
            return None, "endpoint_image_size_mismatch"

        source_pixels = _real_array(source.pixels, name="source_pixels")
        target_pixels = _real_array(target.pixels, name="target_pixels")
        source_points = _real_array(source.points, name="source_points")
        target_points = _real_array(target.points, name="target_points")
        source_right = _real_array(source.right_u, name="source_right_u")
        target_right = _real_array(target.right_u, name="target_right_u")
        n_source, n_target = len(source_pixels), len(target_pixels)
        if (source_pixels.shape != (n_source, 2) or target_pixels.shape != (n_target, 2)
                or source_points.shape != (n_source, 3) or target_points.shape != (n_target, 3)
                or source_right.shape != (n_source,) or target_right.shape != (n_target,)):
            return None, "malformed_raw_endpoint_arrays"

        pairs = _pair_array(training_pairs, "training", n_source, n_target)
        forward = _pair_array(forward_inlier_pairs, "forward_inlier", n_source, n_target,
                              allow_empty=True)
        reverse = _pair_array(reverse_inlier_pairs, "reverse_inlier", n_source, n_target,
                              allow_empty=True)
        training_set = {tuple(row) for row in pairs.tolist()}
        forward_set = {tuple(row) for row in forward.tolist()}
        reverse_set = {tuple(row) for row in reverse.tolist()}
        if not forward_set <= training_set or not reverse_set <= training_set:
            return None, "inlier_pair_not_in_training_table"
        if len(forward) == 0 or len(reverse) == 0:
            return None, "missing_bidirectional_inlier_rows"

        fit_source_points = _real_array(training_source_points, name="training_source_points")
        fit_target_points = _real_array(training_target_points, name="training_target_points")
        fit_source_pixels = _real_array(training_source_pixels, name="training_source_pixels")
        fit_target_pixels = _real_array(training_target_pixels, name="training_target_pixels")
        if (fit_source_points.shape != (len(pairs), 3)
                or fit_target_points.shape != (len(pairs), 3)
                or fit_source_pixels.shape != (len(pairs), 2)
                or fit_target_pixels.shape != (len(pairs), 2)):
            return None, "fit_point_metadata_shape_mismatch"
        raw_source_training_pixels = source_pixels[pairs[:, 0]]
        raw_target_training_pixels = target_pixels[pairs[:, 1]]
        raw_source_training = source_points[pairs[:, 0]]
        raw_target_training = target_points[pairs[:, 1]]
        if (not np.isfinite(fit_source_pixels).all()
                or not np.isfinite(fit_target_pixels).all()
                or not np.array_equal(np.asarray(fit_source_pixels, np.float32),
                                      np.asarray(raw_source_training_pixels, np.float32))
                or not np.array_equal(np.asarray(fit_target_pixels, np.float32),
                                      np.asarray(raw_target_training_pixels, np.float32))
                or not _exact_array_equal_allow_nan(fit_source_points, raw_source_training)
                or not _exact_array_equal_allow_nan(fit_target_points, raw_target_training)):
            return None, "fit_geometry_differs_from_raw_supported_rows"

        # A physical measurement must appear once in each endpoint's training
        # table. Exact float32 keys match the repository's landmark identity.
        source_keys = [tuple(np.asarray(source_pixels[i], np.float32).tolist())
                       for i in pairs[:, 0]]
        target_keys = [tuple(np.asarray(target_pixels[i], np.float32).tolist())
                       for i in pairs[:, 1]]
        if (len(set(source_keys)) != len(source_keys)
                or len(set(target_keys)) != len(target_keys)):
            return None, "duplicate_physical_training_pixel"

        # Use the same rows certified by both PnP directions. This retains no
        # reserved arbitration holdout and makes row provenance explicit.
        information_pairs = np.asarray(sorted(forward_set & reverse_set), dtype=np.int64)
        if len(information_pairs) < 6:
            return None, "insufficient_bidirectional_training_rows"
        sids, tids = information_pairs.T
        sxy, txy = source_pixels[sids], target_pixels[tids]
        sright, tright = source_right[sids], target_right[tids]
        spoints, tpoints = source_points[sids], target_points[tids]
        width, height = image_size
        if (not np.isfinite(sxy).all() or not np.isfinite(txy).all()
                or not np.isfinite(sright).all() or not np.isfinite(tright).all()
                or not np.isfinite(spoints).all() or not np.isfinite(tpoints).all()
                or np.any(sxy < 0.) or np.any(txy < 0.)
                or np.any(sxy[:, 0] >= width) or np.any(txy[:, 0] >= width)
                or np.any(sxy[:, 1] >= height) or np.any(txy[:, 1] >= height)
                or np.any(sright < 0.) or np.any(tright < 0.)
                or np.any(sright >= width) or np.any(tright >= width)
                or np.any(spoints[:, 2] <= 0.) or np.any(tpoints[:, 2] <= 0.)):
            return None, "invalid_raw_stereo_training_measurement"

        information = _build_information(spoints, tpoints, matrix, baseline,
                                         disparity_offset, measurement)
        scale = np.diag([baseline] * 3 + [1.] * 3)
        dimensionless = scale.T @ information @ scale
        dimensionless = 0.5 * (dimensionless + dimensionless.T)
        singular_values = np.linalg.svd(dimensionless, compute_uv=False)
        if not np.isfinite(singular_values).all() or singular_values[0] <= 0.:
            return None, "invalid_dimensionless_information"
        rank_tolerance = max(dimensionless.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > rank_tolerance))
        if rank < 6:
            return None, "rank_deficient_motion_information"
        eigenvalues = np.linalg.eigvalsh(dimensionless)
        psd_tolerance = max(dimensionless.shape) * np.finfo(float).eps * singular_values[0]
        if float(eigenvalues[0]) < -psd_tolerance:
            return None, "non_psd_motion_information"
        if float(eigenvalues[0]) <= psd_tolerance:
            return None, "rank_deficient_motion_information"
        dimensionless_sqrt = np.linalg.cholesky(dimensionless).T
        sqrt_information = dimensionless_sqrt @ np.linalg.inv(scale)
        condition = float(singular_values[0] / singular_values[-1])
        if not np.isfinite(condition):
            return None, "invalid_information_condition"

        factor = StereoMotionRegularizer(
            measurement=measurement, sqrt_information=sqrt_information,
            information=information, matrix=matrix, baseline=baseline,
            disparity_offset=disparity_offset, source_frame=source_frame,
            target_frame=target_frame, source_epoch=epoch,
            source_status=source_status, target_status=target_status,
            image_size=image_size, calibration_identity=calibration_identity,
            training_pairs=pairs, forward_inlier_pairs=forward,
            reverse_inlier_pairs=reverse, information_pairs=information_pairs,
            training_source_pixels=fit_source_pixels,
            training_target_pixels=fit_target_pixels,
            training_source_points=fit_source_points,
            training_target_points=fit_target_points,
            source_pixels=sxy, target_pixels=txy, source_right_u=sright,
            target_right_u=tright, source_points=spoints, target_points=tpoints,
            fit_source=fit_source, fit_depth_policy=fit_depth_policy, role=role,
            rank=rank, dimensionless_singular_values=tuple(singular_values),
            condition_number=condition,
        )
        return factor, None
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError,
            np.linalg.LinAlgError, FloatingPointError) as error:
        reason = str(error) or "invalid_motion_regularizer_evidence"
        return None, reason


__all__ = [
    "StereoMotionRegularizer", "build_stereo_motion_regularizer", "se3_log",
    "stereo_motion_residual", "MODEL", "NOISE_MODEL", "TANGENT_ORDER",
]
