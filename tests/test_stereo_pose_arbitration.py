"""Synthetic counterexamples for the unwired reserved-evidence scorer."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stereo_pose_arbitration import SupportedStereoHoldout, arbitrate_stereo_pose


def fixture(offset=.35):
    matrix = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1.]])
    baseline = .537
    pixels = np.array([(x, y) for x in [160., 450., 740., 1080.] for y in [60., 180., 310.]]*2)
    z = np.linspace(12., 50., len(pixels))
    points = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(matrix).T*z[:, None]
    correct = np.eye(4)
    correct[2, 3] = 2.1
    camera = (points-correct[:3, 3]) @ correct[:3, :3]
    h = camera @ matrix.T
    left = h[:, :2]/h[:, 2, None]
    right = left[:, 0]-matrix[0, 0]*baseline/camera[:, 2]-offset
    evidence = SupportedStereoHoldout(points, left, right, np.arange(24), np.arange(24), np.arange(100, 124),
                                      'immutable_supported_extraction', 123, 'calibration-sha256')
    angle = np.radians(1.1)
    wrong = correct.copy()
    wrong[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    options = dict(calibration_identity='calibration-sha256', independent_training_verified=True,
                   map_holdout_excluded=True, independent_fit_source_ids=[50, 51],
                   independent_fit_target_ids=[50, 51], map_fit_target_ids=[50, 51], map_fit_landmark_ids=[150])
    return evidence, matrix, baseline, correct, wrong, options, offset


def call(f, map_pose=None, independent_pose=None, **changes):
    evidence, matrix, baseline, correct, wrong, options, offset = f
    options.update(changes)
    return arbitrate_stereo_pose(wrong if map_pose is None else map_pose,
                                 correct if independent_pose is None else independent_pose,
                                 evidence, matrix, baseline, (1241, 376), disparity_offset=offset, **options)


def test_coherent_map_can_fit_wrong_roll_while_raw_reserved_stereo_rejects_it():
    f = fixture()
    evidence, matrix, baseline, correct, wrong, _, _ = f
    # Corrupted map geometry is internally exact under the wrong pose. Map left
    # and right reprojection checks cannot detect this gauge-local corruption.
    camera = (evidence.points-correct[:3, 3]) @ correct[:3, :3]
    corrupted_world = camera @ wrong[:3, :3].T + wrong[:3, 3]
    assert np.allclose((corrupted_world-wrong[:3, 3]) @ wrong[:3, :3], camera)
    # Wrong 1.1deg roll remains inside the existing 1.5deg hard disagreement gate.
    report = call(f)
    assert report['choice'] == 'independent'
    assert report['independent']['inliers'] == 24
    assert report['independent']['cells'] >= 3
    assert report['independent']['cost'] < report['map']['cost']


def test_snapshot_owns_arrays_and_is_read_only():
    f = fixture()
    evidence = f[0]
    original = evidence.points.copy()
    rebuilt = SupportedStereoHoldout(original, evidence.left, evidence.right_u, evidence.source_ids,
        evidence.target_ids, evidence.landmark_ids, evidence.provenance, 123, evidence.calibration_identity)
    original[:] = 999
    assert np.array_equal(rebuilt.points, evidence.points)
    with pytest.raises(ValueError):
        rebuilt.points[0, 0] = 1


@pytest.mark.parametrize('key', ['independent_fit_source_ids', 'independent_fit_target_ids', 'map_fit_target_ids'])
def test_any_fit_overlap_abstains(key):
    report = call(fixture(), **{key: [0]})
    assert report['choice'] == 'map' and report['reason'] == 'fit_overlap'


def test_flow_landmark_overlap_abstains_even_without_detector_feature():
    assert call(fixture(), map_fit_landmark_ids=[100])['reason'] == 'fit_overlap'


@pytest.mark.parametrize('change', [dict(independent_training_verified=False), dict(map_holdout_excluded=False),
                                    dict(independent_fit_source_ids=None), dict(calibration_identity='different')])
def test_missing_verification_or_provenance_abstains(change):
    assert call(fixture(), **change)['choice'] == 'map'


def test_exact_and_numerical_ties_keep_map():
    f = fixture()
    assert call(f, map_pose=f[3])['reason'] == 'no_strict_cost_improvement'
    perturbed = f[3].copy()
    perturbed[0, 3] += 1e-12
    assert call(f, map_pose=perturbed)['choice'] == 'map'


@pytest.mark.parametrize('kind', ['nan', 'behind', 'nonrotation'])
def test_bad_independent_prediction_cannot_win(kind):
    f = fixture()
    bad = f[3].copy()
    if kind == 'nan': bad[0, 3] = np.nan
    if kind == 'behind': bad[2, 3] = 100
    if kind == 'nonrotation': bad[0, 0] = 2
    assert call(f, independent_pose=bad)['choice'] == 'map'


def test_too_few_holdout_observations_abstains():
    f = list(fixture())
    e = f[0]
    f[0] = SupportedStereoHoldout(*(getattr(e, n)[:10] for n in ('points', 'left', 'right_u', 'source_ids', 'target_ids', 'landmark_ids')),
                                 e.provenance, e.source_frame, e.calibration_identity)
    assert call(f)['reason'] == 'independent_support_failed'


def test_no_cross_candidate_trimming_normalization_or_offset_loss():
    f = fixture(offset=1.25)
    report = call(f)
    assert report['independent']['cost'] < 1e-20
    assert report['holdout_count'] == 24
    # A single enormous independent residual remains in the common objective.
    e = f[0]
    left = e.left.copy()
    left[0, 0] += 80
    changed = SupportedStereoHoldout(e.points, left, e.right_u, e.source_ids, e.target_ids, e.landmark_ids,
                                    e.provenance, e.source_frame, e.calibration_identity)
    f = list(f); f[0] = changed
    report = call(f)
    assert report['independent']['cost'] > 1
    assert report['independent']['inliers'] == 23


@pytest.mark.parametrize('points', [np.array(3.), np.zeros(24), np.zeros((24, 2))])
def test_invalid_source_dimensions_abstain(points):
    f = list(fixture()); e = f[0]
    f[0] = SupportedStereoHoldout(points, e.left, e.right_u, e.source_ids, e.target_ids, e.landmark_ids,
                                 e.provenance, e.source_frame, e.calibration_identity)
    assert call(f)['reason'] == 'invalid_evidence'


@pytest.mark.parametrize('field', ['points', 'left', 'right_u'])
def test_nonfinite_evidence_abstains(field):
    f = list(fixture()); e = f[0]
    values = {name: getattr(e, name).copy() for name in ('points', 'left', 'right_u', 'source_ids', 'target_ids', 'landmark_ids')}
    values[field].flat[0] = np.nan
    f[0] = SupportedStereoHoldout(**values, provenance=e.provenance, source_frame=e.source_frame,
                                 calibration_identity=e.calibration_identity)
    assert call(f)['reason'] == 'invalid_evidence'


@pytest.mark.parametrize('field', ['left', 'right_u'])
def test_image_domain_abstains(field):
    f = list(fixture()); e = f[0]
    values = {name: getattr(e, name).copy() for name in ('points', 'left', 'right_u', 'source_ids', 'target_ids', 'landmark_ids')}
    values[field].flat[0] = -1
    f[0] = SupportedStereoHoldout(**values, provenance=e.provenance, source_frame=e.source_frame,
                                 calibration_identity=e.calibration_identity)
    assert call(f)['reason'] == 'invalid_image_domain'


@pytest.mark.parametrize('kind', ['nonfinite_matrix', 'invalid_baseline'])
def test_invalid_calibration_abstains(kind):
    f = list(fixture())
    if kind == 'nonfinite_matrix':
        f[1] = f[1].copy(); f[1][0, 0] = np.nan
    else:
        f[2] = 0.
    assert call(f)['reason'] == 'invalid_calibration'
