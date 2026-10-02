"""Unwired preparation: compare stereo pose hypotheses on reserved raw evidence.

The caller must reserve evidence before either solve and certify the independent
hypothesis passed existing training gates. This module never fits a pose, changes
tracking thresholds, or reads reference trajectories.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class SupportedStereoHoldout:
    points: np.ndarray
    left: np.ndarray
    right_u: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray
    landmark_ids: np.ndarray
    provenance: str
    source_frame: int
    calibration_identity: str

    def __post_init__(self):
        # Own snapshots, never views into mutable map/cache arrays.
        for name in ('points', 'left', 'right_u', 'source_ids', 'target_ids', 'landmark_ids'):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _cells(pixels, size):
    return len(np.unique(np.clip((pixels / np.asarray(size) * 3).astype(int), 0, 2), axis=0)) if len(pixels) else 0


def _score(pose, evidence, matrix, baseline, offset, size, minimum):
    n = len(evidence.points)
    report = {'cost': None, 'inliers': 0, 'cells': 0, 'eligible': False}
    pose = np.asarray(pose, float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-7)):
        report['reason'] = 'invalid_pose'
        return report
    camera = (evidence.points - pose[:3, 3]) @ pose[:3, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        report['reason'] = 'invalid_cheirality'
        return report
    homogeneous = camera @ matrix.T
    predicted = homogeneous[:, :2] / homogeneous[:, 2, None]
    right = predicted[:, 0] - matrix[0, 0] * baseline / camera[:, 2] - offset
    residuals = np.c_[predicted-evidence.left, right-evidence.right_u]
    if not np.isfinite(residuals).all():
        report['reason'] = 'nonfinite_residual'
        return report
    # Every hypothesis uses all three components of the exact same observations.
    # No fitting, RANSAC, trimming, or candidate-specific normalization here.
    absolute = np.abs(residuals)
    with np.errstate(over='ignore'):
        cost = float(np.mean(np.where(absolute <= 2., .5*residuals**2, 2.*(absolute-1.))))
    if not np.isfinite(cost):
        report['reason'] = 'nonfinite_objective'
        return report
    left_error = np.linalg.norm(residuals[:, :2], axis=1)
    right_error = absolute[:, 2]
    mask = (left_error <= 2.) & (right_error <= 2.)
    count = int(mask.sum())
    cells = _cells(evidence.left[mask], size)
    eligible = (count >= minimum and count/n >= .25 and cells >= 3
                and np.median(left_error[mask]) <= 1.5
                and np.median(right_error[mask]) <= 1.5)
    report.update(cost=cost, inliers=count, cells=cells, eligible=bool(eligible),
                  reason='accepted_support' if eligible else 'insufficient_support')
    return report


def arbitrate_stereo_pose(map_relative, independent_relative, evidence, matrix, baseline,
                          image_size, *, disparity_offset=0., minimum_inliers=15,
                          calibration_identity=None, independent_training_verified=False,
                          map_holdout_excluded=False, independent_fit_source_ids=None,
                          independent_fit_target_ids=None, map_fit_target_ids=None,
                          map_fit_landmark_ids=None):
    """Return ``map`` unless independent reserved evidence justifies replacement.

    Relative poses express the current camera in the previous camera coordinates.
    The previous world pose is composed by the caller after selection. Existing
    independent .5m/1.5deg checks remain the caller's responsibility.
    """
    report = {'choice': 'map', 'reason': 'missing_metadata', 'holdout_count': 0,
              'map': None, 'independent': None}
    if (evidence is None or not isinstance(evidence, SupportedStereoHoldout)
            or not independent_training_verified or not map_holdout_excluded
            or any(v is None for v in (independent_fit_source_ids, independent_fit_target_ids,
                                      map_fit_target_ids, map_fit_landmark_ids))):
        return report
    if (evidence.provenance != 'immutable_supported_extraction'
            or not evidence.calibration_identity
            or calibration_identity != evidence.calibration_identity
            or not isinstance(evidence.source_frame, (int, np.integer)) or evidence.source_frame < 0):
        report['reason'] = 'invalid_provenance'
        return report
    n = len(evidence.points)
    report['holdout_count'] = n
    if (evidence.points.shape != (n, 3) or evidence.left.shape != (n, 2)
            or any(getattr(evidence, k).shape != (n,) for k in ('right_u', 'source_ids', 'target_ids', 'landmark_ids'))
            or not n or not np.isfinite(evidence.points).all() or not np.isfinite(evidence.left).all()
            or not np.isfinite(evidence.right_u).all() or np.any(evidence.points[:, 2] <= 0)
            or len(np.unique(evidence.source_ids)) != n or len(np.unique(evidence.target_ids)) != n
            or any(not np.issubdtype(getattr(evidence, k).dtype, np.integer)
                   for k in ('source_ids', 'target_ids', 'landmark_ids'))
            or np.any(evidence.source_ids < 0) or np.any(evidence.target_ids < 0)):
        report['reason'] = 'invalid_evidence'
        return report
    if (np.intersect1d(evidence.source_ids, independent_fit_source_ids).size
            or np.intersect1d(evidence.target_ids, independent_fit_target_ids).size
            or np.intersect1d(evidence.target_ids, map_fit_target_ids).size
            or np.intersect1d(evidence.landmark_ids[evidence.landmark_ids >= 0], map_fit_landmark_ids).size):
        report['reason'] = 'fit_overlap'
        return report
    digest = hashlib.sha256()
    for name in ('source_ids', 'target_ids', 'landmark_ids'):
        values = np.ascontiguousarray(getattr(evidence, name), dtype='<i8')
        digest.update(name.encode()+b'\0'+len(values).to_bytes(8, 'little')+values.tobytes())
    report.update(partition_sha256=digest.hexdigest(), source_frame=int(evidence.source_frame),
                  calibration_identity=evidence.calibration_identity,
                  measurement_provenance=evidence.provenance)
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or min(matrix[0, 0], matrix[1, 1]) <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.isfinite([baseline, disparity_offset]).all() or baseline <= 0
            or np.asarray(image_size).shape != (2,) or not np.isfinite(image_size).all()
            or min(image_size) <= 0):
        report['reason'] = 'invalid_calibration'
        return report
    if (np.any(evidence.left < 0) or np.any(evidence.left >= np.asarray(image_size))
            or np.any(evidence.right_u < 0) or np.any(evidence.right_u >= image_size[0])):
        report['reason'] = 'invalid_image_domain'
        return report
    a = _score(map_relative, evidence, matrix, baseline, disparity_offset, image_size, minimum_inliers)
    b = _score(independent_relative, evidence, matrix, baseline, disparity_offset, image_size, minimum_inliers)
    report.update(map=a, independent=b)
    if not b['eligible']:
        report['reason'] = 'independent_support_failed'
        return report
    if b['inliers'] < a['inliers'] or b['cells'] < a['cells']:
        report['reason'] = 'independent_support_not_dominant'
        return report
    tolerance = 64*np.finfo(float).eps*max(1., a['cost'] or 0., b['cost'])
    if a['cost'] is not None and not b['cost'] < a['cost']-tolerance:
        report['reason'] = 'no_strict_cost_improvement'
        return report
    report.update(choice='independent', reason='reserved_stereo_evidence_improved')
    return report
