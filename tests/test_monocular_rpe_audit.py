import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('rpe', Path(__file__).parents[1] / 'scripts/audit_monocular_rpe.py')
rpe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpe)


def poses(xs):
    result = np.tile(np.eye(4), (len(xs), 1, 1))
    result[:, 0, 3] = xs
    return result


def test_fixed_second_not_frame_offset_and_missing_association_no_substitution():
    times = [0., .4, 1., 1.4, 2.]
    assert rpe.fixed_pairs(times, [0, 1, 2, 3, 4]) == [(0, 2), (1, 3), (2, 4)]
    # Nearest target missing: do not silently use another associated timestamp.
    assert rpe.fixed_pairs([0., .99, 1.001], [0, 1]) == []


def test_endpoint_tie_earliest_and_no_interpolation():
    assert rpe.fixed_pairs([0., .984375, 1.015625], [0, 1, 2]) == [(0, 1)]
    assert rpe.fixed_pairs([0., .97, 1.03], [0, 1, 2]) == []


def test_saved_scale_and_global_coordinate_change_cancel():
    estimate = poses([0., 1., 2.])
    truth = poses([0., 2., 4.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    truth[:, :3, :3] = rotation
    truth[:, :3, 3] = truth[:, :3, 3] @ rotation.T + [8., 9., 10.]
    result = rpe.corrected_pair_errors(estimate, truth, [0., 1., 2.], [0, 1, 2], 2.)
    assert all(row['translation_error_m'] == 0. and row['rotation_error_deg'] == 0. for row in result)


def test_known_translation_and_rotation_errors():
    estimate, truth = poses([0., 1.]), poses([0., 3.])
    truth[1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    result = rpe.corrected_pair_errors(estimate, truth, [0., 1.], [0, 1], 1.)[0]
    assert result['translation_error_m'] == 2.
    assert result['rotation_error_deg'] == pytest.approx(90.)


@pytest.mark.parametrize('times', [[0., 0.], [1., 0.], [0., float('nan')]])
def test_invalid_times(times):
    with pytest.raises(ValueError):
        rpe.fixed_pairs(times, [0, 1])


@pytest.mark.parametrize('frames', [[0, 0], [True], [1.0], [-1], [2]])
def test_invalid_associated_frames(frames):
    with pytest.raises(ValueError):
        rpe.fixed_pairs([0., 1.], frames)


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf'), True])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        rpe.corrected_pair_errors(poses([0., 1.]), poses([0., 1.]), [0., 1.], [0, 1], scale)


def test_invalid_rotation_rejected_not_projected_to_so3():
    estimate = poses([0., 1.]); estimate[1, 0, 0] = -1.
    with pytest.raises(ValueError, match='SO3'):
        rpe.corrected_pair_errors(estimate, poses([0., 1.]), [0., 1.], [0, 1], 1.)


def test_empty_pair_set_has_no_invented_zero_error():
    assert rpe.corrected_pair_errors(poses([0.]), poses([0.]), [0.], [0], 1.) == []


def test_partial_range_does_not_change_full_input_endpoint_selection():
    assert rpe.corrected_pair_errors(poses([0., 1.]), poses([0., 1.]),
                                     [0., .99, 1.001], [0, 1], 1.) == []


def test_noncommuting_pose_errors_match_independent_homogeneous_calculation():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(817)
    for _ in range(12):
        estimated, truth = poses([0., 0., 0.]), poses([0., 0., 0.])
        estimated[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
        truth[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
        estimated[:, :3, 3], truth[:, :3, 3] = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        scale = 1.7
        alignment = Rotation.random(random_state=rng).as_matrix()
        aligned = estimated.copy()
        aligned[:, :3, :3] = alignment @ estimated[:, :3, :3]
        aligned[:, :3, 3] = scale * (estimated[:, :3, 3] @ alignment.T) + [3., -4., 8.]
        result = rpe.corrected_pair_errors(estimated, truth, [0., 1., 2.], [0, 1, 2], scale)
        for row in result:
            i, j = row['first_frame'], row['last_frame']
            error = np.linalg.inv(np.linalg.inv(aligned[i]) @ aligned[j]) @ (np.linalg.inv(truth[i]) @ truth[j])
            assert row['translation_error_m'] == pytest.approx(np.linalg.norm(error[:3, 3]))
            assert row['rotation_error_deg'] == pytest.approx(np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude()))


def test_oversized_scale_fails_cleanly():
    with pytest.raises(ValueError):
        rpe.corrected_pair_errors(poses([0., 1.]), poses([0., 1.]), [0., 1.], [0, 1], 10**400)


def integrated_fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    n = 4
    estimate = poses([0., 1., 2., 3.])[:, :3]
    truth = np.column_stack([np.arange(n), 2*np.arange(n), np.zeros((n, 2)), np.tile([0., 0., 0., 1.], (n, 1))])
    metric = {'associated_frames': list(range(n)), 'alignment': {'rotation': np.eye(3).tolist(), 'scale': 2.}}
    report = {'runs': [{'folder': str(i), 'processed_frames': n, 'metrics': {'online': metric, 'corrected': metric}} for i in range(2)]}
    runs = [{'tracking': [{'state': s} for s in states]} for states in
            [['tracking']*4, ['tracking', 'lost', 'tracking', 'tracking']]]
    exports = [{'online': estimate, 'corrected': estimate} for _ in runs]
    return report, runs, exports, np.arange(n), truth


def test_integrated_rpe_common_accepted_endpoints_and_online_unassessable(monkeypatch):
    result = rpe.audit(*integrated_fixture(monkeypatch))
    assert result['online']['unassessable']
    for row in result['runs']:
        assert row['corrected']['common_accepted']['pair_ids'] == [[2, 3]]
        assert row['corrected']['common_accepted']['translation']['rmse_m'] == 0.
        assert row['corrected']['common_accepted']['rotation']['p95_deg'] == 0.


def test_missing_corrected_export_cannot_claim_common_comparison(monkeypatch):
    args = integrated_fixture(monkeypatch)
    args[2][1]['corrected'] = None
    result = rpe.audit(*args)
    assert all(row['corrected']['common_accepted']['unassessable'] for row in result['runs'])


def test_rpe_cli_hashes_inputs_and_preserves_existing_output(monkeypatch, tmp_path):
    import json
    report, runs, exports, times, truth = integrated_fixture(monkeypatch)
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    (dataset / 'rgb.txt').write_text(''.join(f'{t} image.png\n' for t in times))
    np.savetxt(dataset / 'groundtruth.txt', truth)
    folders = []
    for i, (run, exported) in enumerate(zip(runs, exports)):
        folder = tmp_path / str(i); folder.mkdir(); folders.append(str(folder))
        report['runs'][i]['folder'] = str(folder)
        (folder / 'run.json').write_text(json.dumps(run))
        (folder / 'evaluation.json').write_text(json.dumps({'dataset': 'tum', 'sequence': 'fixture', 'frames': 4}))
        for label, filename in [('online', 'online-poses.txt'), ('corrected', 'poses.txt')]:
            np.savetxt(folder / filename, exported[label].reshape(-1, 12))
    source = tmp_path / 'scores.json'; source.write_text(json.dumps(report))
    out = tmp_path / 'rpe.json'
    args = ['--scores', str(source), '--dataset', str(dataset), '--sequence', 'fixture', '--runs', *folders, '--output', str(out)]
    assert rpe.main(args) == 0
    original = out.read_bytes(); result = json.loads(original)
    assert len(result['inputs']) == 11
    assert all(len(row['sha256']) == 64 for row in result['inputs'])
    with pytest.raises(FileExistsError):
        rpe.main(args)
    assert out.read_bytes() == original
