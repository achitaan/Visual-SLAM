import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('rotation_audit', Path(__file__).parents[1] / 'scripts/audit_monocular_rotations.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    n = 4
    global_rotation = Rotation.from_euler('xyz', [25, -40, 70], degrees=True).as_matrix()
    gt_rotations = Rotation.from_euler('xyz', [[0, 0, 0], [10, 20, 30], [-15, 45, 25], [90, 10, 10]], degrees=True).as_matrix()
    truth = np.column_stack([np.arange(n), np.zeros((n, 3)), Rotation.from_matrix(gt_rotations).as_quat()])
    poses = np.zeros((n, 3, 4)); poses[:, :, :3] = global_rotation.T @ gt_rotations
    metric = {'associated_frames': list(range(n)), 'alignment': {'rotation': global_rotation.tolist()}}
    summaries = [{'folder': str(i), 'processed_frames': n, 'metrics': {label: copy.deepcopy(metric) for label in ('online', 'corrected')}} for i in range(2)]
    runs = [{'tracking': [{'state': state} for state in states]} for states in [['tracking', 'lost', 'relocalized', 'tracking'], ['lost', 'tracking', 'tracking', 'tracking']]]
    exports = [{label: poses.copy() for label in ('online', 'corrected')} for _ in runs]
    return {'runs': summaries}, runs, exports, np.arange(n), truth


def test_known_global_rotation_and_same_fit_for_subsets():
    report, runs, exports, stamps, truth = fixture()
    # Local camera offset survives the global alignment; no subset refit removes it.
    offset = Rotation.from_euler('x', 12, degrees=True).as_matrix()
    exports[1]['online'][:, :, :3] = exports[1]['online'][:, :, :3] @ offset
    result = audit.audit(report, runs, exports, stamps, truth)
    first, second = [r['metrics']['online'] for r in result['runs']]
    assert first['all']['maximum_deg'] < 1e-12
    assert second['angular_error_deg'] == pytest.approx([12]*4)
    assert second['all']['p95_deg'] == pytest.approx(12)
    assert second['accepted']['p95_deg'] == pytest.approx(12)
    assert second['common_accepted']['p95_deg'] == pytest.approx(12)
    assert first['common_accepted_associated_frames'] == second['common_accepted_associated_frames'] == [2, 3]
    assert first['recovery_events'][0]['frame'] == 2
    assert first['recovery_events'][0]['angular_error_deg'] < 1e-12


def test_association_reuses_historical_greedy_unique_nearest_policy():
    truth = np.column_stack([[0., .04, .1], np.zeros((3, 3)), np.tile([0, 0, 0, 1], (3, 1))])
    pairs, rotations = audit.associations([0., .01, .04, .13], truth, 4)
    assert pairs == [(0, 0), (2, 1)]  # no second-best match when nearest already used
    assert rotations.shape == (3, 3, 3)


def test_p95_convention():
    assert audit.distribution([0., 1., 2., 3.])['p95_deg'] == pytest.approx(2.85)
    assert audit.distribution([]) is None


@pytest.mark.parametrize('quaternion', [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, float('nan'), 1]])
def test_invalid_gt_quaternions_rejected(quaternion):
    report, runs, exports, stamps, truth = fixture()
    truth[0, 4:] = quaternion
    with pytest.raises(ValueError):
        audit.audit(report, runs, exports, stamps, truth)


@pytest.mark.parametrize('rotation', [np.diag([1., 1., -1.]), np.eye(3)*2, np.full((3, 3), np.nan), np.zeros((2, 2))])
def test_invalid_alignment_rotations_rejected(rotation):
    report, runs, exports, stamps, truth = fixture()
    report['runs'][0]['metrics']['online']['alignment']['rotation'] = rotation.tolist()
    with pytest.raises(ValueError):
        audit.audit(report, runs, exports, stamps, truth)


def test_invalid_pose_rotation_rejected_even_if_frame_not_accepted():
    report, runs, exports, stamps, truth = fixture()
    exports[0]['online'][1, :, :3] *= 2
    with pytest.raises(ValueError, match='SO'):
        audit.audit(report, runs, exports, stamps, truth)


def test_mismatched_saved_association_rejected():
    report, runs, exports, stamps, truth = fixture()
    report['runs'][0]['metrics']['online']['associated_frames'] = [0, 2, 3]
    with pytest.raises(ValueError, match='association'):
        audit.audit(report, runs, exports, stamps, truth)


def test_unavailable_fit_preserves_other_audits_without_common_claim():
    report, runs, exports, stamps, truth = fixture()
    report['runs'][1]['metrics']['online'] = {'evaluation_error': 'stationary fit'}
    result = audit.audit(report, runs, exports, stamps, truth)
    assert result['runs'][0]['metrics']['online']['all']['maximum_deg'] < 1e-12
    for row in result['runs']:
        assert row['metrics']['online']['common_accepted'] is None
        assert row['metrics']['online']['common_unassessable']
        assert row['metrics']['corrected']['common_accepted']['count'] == 2


def test_missing_pose_export_is_unassessable():
    report, runs, exports, stamps, truth = fixture()
    exports[0]['online'] = None
    assert audit.audit(report, runs, exports, stamps, truth)['runs'][0]['metrics']['online']['unassessable']


def test_cli_hashes_and_refuses_overwrite(tmp_path):
    report, runs, exports, stamps, truth = fixture()
    data = tmp_path / 'data'; data.mkdir()
    (data / 'rgb.txt').write_text(''.join(f'{t} rgb/{t}.png\n' for t in stamps))
    np.savetxt(data / 'groundtruth.txt', truth)
    folders = []
    for i, run in enumerate(runs):
        folder = tmp_path / str(i); folder.mkdir(); folders.append(folder)
        report['runs'][i]['folder'] = str(folder)
        (folder / 'run.json').write_text(json.dumps(run))
        (folder / 'evaluation.json').write_text(json.dumps({'dataset': 'tum', 'sequence': 'freiburg1_desk', 'frames': 4}))
        for label, filename in [('online', 'online-poses.txt'), ('corrected', 'poses.txt')]:
            np.savetxt(folder / filename, exports[i][label].reshape(-1, 12))
    scores = tmp_path / 'scores.json'; scores.write_text(json.dumps(report))
    output = tmp_path / 'audit.json'
    args = ['--scores', str(scores), '--dataset', str(data), '--sequence', 'freiburg1_desk', '--runs', *map(str, folders), '--output', str(output)]
    assert audit.main(args) == 0
    raw = output.read_bytes(); result = json.loads(raw)
    assert len(result['inputs']) == 11
    assert all(len(row['sha256']) == 64 for row in result['inputs'])
    with pytest.raises(FileExistsError):
        audit.main(args)
    assert output.read_bytes() == raw
    args[args.index('freiburg1_desk')] = 'freiburg1_room'
    with pytest.raises(SystemExit):
        audit.main(args)


def test_quaternion_sign_is_orientation_equivalent_and_pi_is_finite():
    report, runs, exports, stamps, truth = fixture()
    truth[:, 4:] *= -1
    exports[0]['corrected'][:, :, :3] = exports[0]['corrected'][:, :, :3] @ Rotation.from_euler('y', 180, degrees=True).as_matrix()
    metric = audit.audit(report, runs, exports, stamps, truth)['runs'][0]['metrics']['corrected']
    assert metric['angular_error_deg'] == pytest.approx([180]*4)


def test_empty_common_subset_is_unassessable_not_perfect():
    report, runs, exports, stamps, truth = fixture()
    runs[1]['tracking'] = [{'state': 'lost'} for _ in range(4)]
    result = audit.audit(report, runs, exports, stamps, truth)
    assert result['runs'][1]['metrics']['online']['accepted'] is None
    assert all(row['metrics']['online']['common_accepted'] is None for row in result['runs'])
    assert all(row['metrics']['online']['common_unassessable'] for row in result['runs'])


def test_unassociated_recovery_event_explicitly_unassessable():
    report, runs, exports, stamps, truth = fixture()
    truth = np.delete(truth, 2, axis=0)
    for row in report['runs']:
        for metric in row['metrics'].values():
            metric['associated_frames'] = [0, 1, 3]
    result = audit.audit(report, runs, exports, stamps, truth)
    event = result['runs'][0]['metrics']['online']['recovery_events'][0]
    assert event == {'frame': 2, 'angular_error_deg': None, 'unassessable': 'No associated ground truth'}


@pytest.mark.parametrize('target', ['rgb', 'gt'])
def test_duplicate_timestamps_rejected(target):
    report, runs, exports, stamps, truth = fixture()
    if target == 'rgb':
        stamps[1] = stamps[0]
    else:
        truth[1, 0] = truth[0, 0]
    with pytest.raises(ValueError, match='timestamps'):
        audit.audit(report, runs, exports, stamps, truth)
