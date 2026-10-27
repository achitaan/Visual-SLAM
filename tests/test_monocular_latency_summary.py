import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('latency', Path(__file__).parents[1] / 'scripts/summarize_monocular_latency.py')
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


def fixture():
    run = {'tracking': [{'frame': i, 'state': state} for i, state in enumerate(
        ['initializing', 'tracking', 'lost', 'relocalized', 'tracking'])]}
    timing = {'definition': 'fixture process-only timing', 'records': [
        {'frame': i, 'process_wall_seconds': float(i)} for i in range(5)]}
    return run, timing


def test_statistics_linear_p95_and_empty_stratum():
    assert latency.statistics([0., 1., 2., 3.]) == {'count': 4, 'median_seconds': 1.5,
                                                   'p95_seconds': pytest.approx(2.85), 'maximum_seconds': 3.}
    assert latency.statistics([]) == {'count': 0, 'median_seconds': None, 'p95_seconds': None, 'maximum_seconds': None}


def test_exact_id_join_not_position_and_recovery_classification():
    run, timing = fixture()
    run['tracking'].reverse(); timing['records'] = timing['records'][2:] + timing['records'][:2]
    result = latency.summarize(run, timing)
    assert result['exact_frame_join']
    assert result['by_state']['tracking']['count'] == 2
    assert result['by_state']['tracking']['p95_seconds'] == pytest.approx(3.85)
    assert result['by_recovery']['recovery']['count'] == 1
    assert result['by_recovery']['recovery']['maximum_seconds'] == 3.
    assert result['by_recovery']['nonrecovery']['count'] == 4
    assert result['by_recovery']['unassessable_missing_predecessor']['count'] == 0


def test_missing_timing_does_not_change_preceding_input_state():
    run, timing = fixture()
    del timing['records'][2]  # lost frame absent from timings, present in tracking
    result = latency.summarize(run, timing)
    assert result['by_recovery']['recovery']['count'] == 1
    assert result['tracking_without_timing'] == [2]
    assert result['exact_frame_join'] is False


def test_missing_preceding_tracking_frame_is_unassessable():
    run, timing = fixture(); del run['tracking'][2]
    result = latency.summarize(run, timing)
    assert result['by_recovery']['recovery']['count'] == 0
    assert result['by_recovery']['unassessable_missing_predecessor']['count'] == 1
    assert result['timing_without_tracking'] == [2]
    assert result['all_process']['count'] == 5
    assert result['matched_process']['count'] == 4
    assert result['exact_frame_join'] is False


def test_initialization_to_tracking_and_frame_zero_are_not_recoveries():
    run, timing = fixture()
    result = latency.summarize(run, timing)
    assert result['classified_rows'][0]['recovery_class'] == 'nonrecovery'
    assert result['classified_rows'][1]['recovery_class'] == 'nonrecovery'


@pytest.mark.parametrize('field', ['tracking', 'timing'])
@pytest.mark.parametrize('frame', [-1, True, 1.0, '1', None])
def test_invalid_frame_ids(field, frame):
    run, timing = fixture()
    (run['tracking'] if field == 'tracking' else timing['records'])[0]['frame'] = frame
    with pytest.raises(ValueError, match='frame IDs'):
        latency.summarize(run, timing)


@pytest.mark.parametrize('field', ['tracking', 'timing'])
def test_duplicate_frame_ids(field):
    run, timing = fixture(); rows = run['tracking'] if field == 'tracking' else timing['records']
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match='Duplicate'):
        latency.summarize(run, timing)


@pytest.mark.parametrize('value', [-1, True, '1', None, float('nan'), float('inf'), 10**400])
def test_invalid_latencies(value):
    run, timing = fixture(); timing['records'][0]['process_wall_seconds'] = value
    with pytest.raises(ValueError, match='latency'):
        latency.summarize(run, timing)


def test_unknown_state_and_malformed_containers():
    run, timing = fixture(); run['tracking'][0]['state'] = 'accepted'
    with pytest.raises(ValueError, match='state'):
        latency.summarize(run, timing)
    with pytest.raises(ValueError):
        latency.summarize([], timing)
    with pytest.raises(ValueError):
        latency.summarize({'tracking': {}}, timing)


def test_empty_records_no_completion_claim():
    result = latency.summarize({'tracking': []}, {'records': []})
    assert result['all_process']['count'] == 0 and result['all_process']['p95_seconds'] is None
    assert 'complete dataset' in result['limitations'][0]


def test_large_finite_latency_percentile():
    run, timing = fixture()
    for row in timing['records']:
        row['process_wall_seconds'] = 1e308
    assert latency.summarize(run, timing)['all_process']['p95_seconds'] == 1e308


def test_cli_hashes_and_preserves_existing_outputs(tmp_path):
    run, timing = fixture(); folder = tmp_path / 'run'; folder.mkdir()
    (folder / 'run.json').write_text(json.dumps(run))
    (folder / 'frame-latencies.json').write_text(json.dumps(timing))
    output = tmp_path / 'summary.json'; args = ['--runs', str(folder), '--output', str(output)]
    assert latency.main(args) == 0
    saved = output.read_bytes(); result = json.loads(saved)
    assert len(result['inputs']) == 2 and all(len(row['sha256']) == 64 for row in result['inputs'])
    with pytest.raises(FileExistsError):
        latency.main(args)
    assert output.read_bytes() == saved
    with pytest.raises(SystemExit):
        latency.main(['--runs', str(folder), str(folder), '--output', str(tmp_path/'duplicate.json')])
    assert not (tmp_path/'duplicate.json').exists()
