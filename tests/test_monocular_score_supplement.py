import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'supplement', Path(__file__).parents[1] / 'scripts/supplement_monocular_scores.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def fixture(states, residuals, frames=None):
    frames = list(range(len(states))) if frames is None else frames
    summary = {'folder': 'fixture', 'processed_frames': len(states), 'metrics': {
        name: {'associated_frames': frames, 'residuals_m': residuals}
        for name in ('online', 'corrected')}}
    return summary, {'tracking': [{'state': state} for state in states]}


def test_linear_percentile_and_rmse_independently_calculated():
    result = scorer.distribution([0., 1., 2., 3.])
    assert result == {'count': 4, 'rmse_m': pytest.approx(3.5**.5),
                      'median_m': 1.5, 'p95_m': pytest.approx(2.85), 'maximum_m': 3.}
    assert scorer.distribution([]) is None
    assert scorer.distribution([7.])['p95_m'] == 7.
    assert scorer.distribution([0., 0.])['rmse_m'] == 0.
    assert scorer.distribution([1e308, 1e308])['rmse_m'] == 1e308


def test_same_common_accepted_associated_frames_no_refit():
    a, ar = fixture(['tracking', 'lost', 'tracking', 'relocalized'], [0., 1., 2., 10.])
    b, br = fixture(['lost', 'tracking', 'tracking', 'tracking'], [20., 30., 40.], [1, 2, 3])
    result = scorer.supplement({'runs': [a, b]}, [ar, br])
    for export in ('online', 'corrected'):
        x, y = [r['metrics'][export] for r in result['runs']]
        assert x['common_accepted_associated_frames'] == y['common_accepted_associated_frames'] == [2, 3]
        assert x['common_accepted_translation']['p95_m'] == pytest.approx(9.6)
        assert y['common_accepted_translation']['p95_m'] == pytest.approx(39.5)
        assert x['common_accepted_translation']['rmse_m'] == pytest.approx(52**.5)


def test_association_and_acceptance_intersection_over_all_runs():
    a, ar = fixture(['tracking']*4, [1., 2., 3.], [0, 1, 3])
    b, br = fixture(['tracking']*4, [4., 5., 6.], [0, 2, 3])
    c, cr = fixture(['lost', 'tracking', 'tracking', 'tracking'], [7., 8., 9., 10.])
    result = scorer.supplement({'runs': [a, b, c]}, [ar, br, cr])
    assert all(r['metrics']['online']['common_accepted_associated_frames'] == [3] for r in result['runs'])


@pytest.mark.parametrize('accepted,unstable', [(26, True), (27, False), (30, False)])
def test_next_thirty_window_excludes_event_and_threshold(accepted, unstable):
    states = ['lost', 'relocalized'] + ['tracking']*accepted + ['lost']*(30-accepted)
    event = scorer.recovery_stability(states)[0]
    assert event == {'frame': 1, 'window_start_frame': 2, 'observed_window_frames': 30,
                     'accepted_frames': accepted, 'censored': False,
                     'unstable_continuation': unstable}


@pytest.mark.parametrize('remaining', [0, 1, 26, 27, 29])
def test_short_terminal_windows_always_censored(remaining):
    event = scorer.recovery_stability(['lost', 'tracking'] + ['tracking']*remaining)[0]
    assert event['censored'] is True and event['unstable_continuation'] is None
    assert event['observed_window_frames'] == remaining


def test_initialization_not_recovery_and_repeated_recovery():
    states = ['initializing', 'tracking', 'lost', 'relocalized', 'lost', 'tracking']
    assert [e['frame'] for e in scorer.recovery_stability(states)] == [3, 5]


def test_empty_intersection_is_unassessable_not_zero_error():
    a, ar = fixture(['lost', 'tracking'], [1., 2.])
    b, br = fixture(['tracking', 'lost'], [3., 4.])
    result = scorer.supplement({'runs': [a, b]}, [ar, br])
    metric = result['runs'][0]['metrics']['online']
    assert metric['common_accepted_translation'] is None
    assert metric['unassessable']


def test_failed_export_marks_all_comparisons_unassessable():
    a, ar = fixture(['tracking'], [1.])
    b, br = fixture(['tracking'], [2.])
    b['metrics']['online'] = {'evaluation_error': 'ill-conditioned full fit'}
    result = scorer.supplement({'runs': [a, b]}, [ar, br])
    assert all(r['metrics']['online'].get('unassessable') for r in result['runs'])
    assert result['runs'][0]['metrics']['corrected']['common_accepted_translation']


@pytest.mark.parametrize('values', [[float('nan')], [float('inf')], [-1.], [True], ['1'], [10**400]])
def test_invalid_residuals_not_hidden_by_lost_mask(values):
    a, ar = fixture(['lost'], values)
    b, br = fixture(['tracking'], [2.])
    with pytest.raises(ValueError):
        scorer.supplement({'runs': [a, b]}, [ar, br])


@pytest.mark.parametrize('frames', [[0, 0], [-1, 0], [True, 1], [0., 1], [0, 2]])
def test_bad_association_ids_rejected(frames):
    a, ar = fixture(['tracking']*2, [1., 2.], frames)
    with pytest.raises(ValueError):
        scorer.supplement({'runs': [a, a]}, [ar, ar])


def test_missing_or_unknown_tracking_evidence_rejected():
    a, ar = fixture(['tracking'], [1.])
    with pytest.raises(ValueError):
        scorer.supplement({'runs': [a, a]}, [ar, {'tracking': []}])
    with pytest.raises(ValueError):
        scorer.supplement({'runs': [a, a]}, [ar, {'tracking': [{'state': 'accepted'}]}])


def test_cli_binds_run_order_hashes_and_refuses_overwrite(tmp_path):
    folders, summaries = [], []
    for label in ('a', 'b'):
        folder = tmp_path / label
        folder.mkdir()
        summary, run = fixture(['tracking'], [1.])
        summary['folder'] = str(folder)
        (folder / 'run.json').write_text(json.dumps(run))
        folders.append(str(folder))
        summaries.append(summary)
    report = tmp_path / 'scores.json'
    report.write_text(json.dumps({'runs': summaries}))
    output = tmp_path / 'supplement.json'
    args = ['--scores', str(report), '--runs', *folders, '--output', str(output)]
    assert scorer.main(args) == 0
    result = json.loads(output.read_text())
    assert len(result['inputs']) == 3
    assert all(len(row['sha256']) == 64 for row in result['inputs'])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        scorer.main(args)
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        scorer.main(['--scores', str(report), '--runs', *reversed(folders), '--output', str(output)])
