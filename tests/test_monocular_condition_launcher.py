"""Mocked evaluator launches only; the real CPU-ownership interlock stays on."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_monocular_condition.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('monocular_condition_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grant_files(tmp_path):
    receipt = tmp_path / 'fixture-only-not-real-release.txt'
    receipt.write_text('Synthetic test fixture: NOT authoritative process termination evidence.\n')
    reference = {'path': receipt.name, 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    value = {
        'schema_version': 1, 'condition': 'consensus', 'seed': 0, 'owner': 'test fixture only',
        'total_seconds': 900.,
        'deadline_utc': (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        'prior_job_terminal_receipt': reference, 'cpu_release_receipt': reference,
        'reserve_justification': 'Hypothetical test arithmetic, not an established bound.',
        'cancellation_justification': 'Mocked subprocess only; not an execution grant.',
        'shutdown_export_reserve_seconds': 30., 'terminal_receipt_reserve_seconds': 5.,
        'sequences': {
            'desk': {'start_frame': 0, 'frames': 613, 'expected_launch_seconds': 288.900008358,
                     'uncertainty_seconds': 50., 'estimate_evidence': 'Historical observed wall, fixture.'},
            'room': {'start_frame': 0, 'frames': 1362, 'expected_launch_seconds': 520.385951437,
                     'uncertainty_seconds': 50., 'estimate_evidence': 'Historical observed wall, fixture.'},
        },
    }
    path = tmp_path / 'fixture-admission.json'
    path.write_text(json.dumps(value))
    return path, value


def prepared(runner, tmp_path, monkeypatch):
    _, grant = grant_files(tmp_path)
    # Unit tests alone exercise post-release branches with a fake evaluator.
    monkeypatch.setattr(runner, 'UNRESOLVED_CPU_OWNER', None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=6*1024**3))
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    for folder in ('src', 'scripts'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'fixture.py').write_text('# no estimator code\n')
    return grant, clock


def run_mocked(runner, tmp_path, grant):
    return runner.run_condition(
        tmp_path, tmp_path / 'results', 'consensus', 0, grant,
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1))


def test_actual_entry_point_rejects_self_authored_release_without_child(runner, tmp_path, monkeypatch):
    path, _ = grant_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('No child permitted'))
    with pytest.raises(SystemExit) as blocked:
        runner.main(['--condition', 'consensus', '--admission', str(path)])
    assert blocked.value.code == 2
    assert not (tmp_path / 'results').exists()


def test_direct_run_api_cannot_bypass_original_job_interlock(runner, tmp_path, monkeypatch):
    _, grant = grant_files(tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('No child permitted'))
    with pytest.raises(ValueError, match='termination is unproven'):
        run_mocked(runner, tmp_path, grant)
    assert not (tmp_path / 'results').exists()


def test_missing_admission_cannot_start_a_child(runner, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('No child permitted'))
    with pytest.raises(SystemExit) as missing:
        runner.main(['--condition', 'consensus'])
    assert missing.value.code == 2


@pytest.mark.parametrize('change', ['wrong_condition', 'wrong_seed', 'wrong_range',
                                   'changed_receipt', 'expired', 'zero_uncertainty',
                                   'missing_reserve_justification'])
def test_declared_admission_is_scope_and_hash_bound(runner, tmp_path, change):
    path, grant = grant_files(tmp_path)
    if change == 'wrong_condition':
        grant['condition'] = 'recovery'
    elif change == 'wrong_seed':
        grant['seed'] = True
    elif change == 'wrong_range':
        grant['sequences']['room']['frames'] = 300
    elif change == 'changed_receipt':
        grant['cpu_release_receipt'] = dict(grant['cpu_release_receipt'], sha256='0'*64)
    elif change == 'expired':
        grant['deadline_utc'] = '2020-01-01T00:00:00Z'
    elif change == 'zero_uncertainty':
        grant['sequences']['room']['uncertainty_seconds'] = 0
    else:
        grant['reserve_justification'] = ''
    path.write_text(json.dumps(grant))
    with pytest.raises(ValueError):
        runner.load_admission(path, tmp_path, 'consensus', 0)


def test_underbudget_first_scene_produces_no_child_or_run_directory(runner, tmp_path, monkeypatch):
    grant, _ = prepared(runner, tmp_path, monkeypatch)
    grant['total_seconds'] = 60.
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('No child permitted'))
    records = run_mocked(runner, tmp_path, grant)
    assert len(records) == 1 and 'not_started' in records[0]
    assert not (tmp_path / 'results/consensus-desk-full-seed0').exists()


def test_room_is_not_started_after_desk_consumes_its_allowance(runner, tmp_path, monkeypatch):
    grant, clock = prepared(runner, tmp_path, monkeypatch)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--output')+1])
        (out / 'evaluation.json').write_text(json.dumps({
            'frames': 613, 'states': {'accepted': 613}, 'peak_memory_mb': 1.}))
        clock[0] = 840.
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    records = run_mocked(runner, tmp_path, grant)
    assert len(calls) == 1 and records[0]['full_range'] is True
    assert records[1]['scene'] == 'room' and 'not_started' in records[1]
    assert not (tmp_path / 'results/consensus-room-full-seed0').exists()


def test_disk_is_rechecked_before_second_scene(runner, tmp_path, monkeypatch):
    grant, _ = prepared(runner, tmp_path, monkeypatch)
    free = [6*1024**3]
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free[0]))
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--output')+1])
        (out / 'evaluation.json').write_text(json.dumps({
            'frames': 613, 'states': {'accepted': 613}, 'peak_memory_mb': 1.}))
        free[0] = 5*1024**3 - 1
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    records = run_mocked(runner, tmp_path, grant)
    assert len(calls) == 1 and len(records[1]['not_started']) == 1
    assert 'free disk floor' in records[1]['not_started'][0]


@pytest.mark.parametrize('free', [5*1024**3, (5120+256)*1024**2 - 1])
def test_evaluator_startup_export_reserve_is_checked_without_launch(runner, tmp_path, monkeypatch, free):
    grant, _ = prepared(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('No child permitted'))
    records = run_mocked(runner, tmp_path, grant)
    assert records[0]['planning_decision']['minimum_free_bytes'] == (5120+256)*1024**2
    assert records[0]['planning_decision']['planning_fits'] is False


@pytest.mark.parametrize('failure', ['timeout', 'crash', 'malformed_export', 'incomplete_export'])
def test_terminal_failure_stops_batch_and_preserves_receipt(runner, tmp_path, monkeypatch, failure):
    grant, _ = prepared(runner, tmp_path, monkeypatch)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--output')+1])
        if failure == 'timeout':
            raise runner.subprocess.TimeoutExpired(command, kwargs['timeout'])
        if failure == 'malformed_export':
            (out / 'evaluation.json').write_text('{bad json')
        elif failure == 'incomplete_export':
            (out / 'evaluation.json').write_text(json.dumps({
                'frames': 300, 'states': {'accepted': 300}, 'peak_memory_mb': 1.}))
        return SimpleNamespace(returncode=1 if failure == 'crash' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    records = run_mocked(runner, tmp_path, grant)
    assert len(calls) == len(records) == 1
    saved = json.loads((tmp_path / 'results/consensus-desk-full-seed0/launch-result.json').read_text())
    assert saved == records[0] and saved['owned_child_wait_observed'] is True
    assert not (tmp_path / 'results/consensus-room-full-seed0').exists()
    if failure == 'timeout':
        assert saved['exit'] == 124 and saved['timed_out'] is True
    elif failure == 'malformed_export':
        assert saved['full_range'] is False and saved['evaluation_error']


def test_failed_final_child_is_not_reported_as_success_by_cli(runner, tmp_path, monkeypatch):
    path, _ = grant_files(tmp_path)
    monkeypatch.setattr(runner, 'UNRESOLVED_CPU_OWNER', None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'run_condition', lambda *a: [
        {'exit': 0, 'full_range': True}, {'exit': 1, 'full_range': True}])
    assert runner.main(['--condition', 'consensus', '--admission', str(path)]) == 2


def test_historical_result_cannot_be_overwritten(runner, tmp_path, monkeypatch):
    grant, _ = prepared(runner, tmp_path, monkeypatch)
    root = tmp_path / 'results'
    root.mkdir()
    saved = root / 'consensus-batch-seed0-result.json'
    saved.write_text('historical result bytes')
    with pytest.raises(ValueError, match='Never overwrite'):
        run_mocked(runner, tmp_path, grant)
    assert saved.read_text() == 'historical result bytes'


def test_large_bookkeeping_delay_does_not_spend_shutdown_reserve(runner, tmp_path, monkeypatch):
    grant, clock = prepared(runner, tmp_path, monkeypatch)
    original_write = runner._write_json
    calls = []
    def delayed_write(path, value):
        original_write(path, value)
        if path.name == 'preparation.json':
            clock[0] += 70.
    monkeypatch.setattr(runner, '_write_json', delayed_write)
    def child(command, **kwargs):
        calls.append(command)
        processing = float(command[command.index('--max-wall-seconds') + 1])
        assert kwargs['timeout'] == pytest.approx(825.)
        assert processing == pytest.approx(795.)
        assert kwargs['timeout'] - processing == grant['shutdown_export_reserve_seconds']
        out = Path(command[command.index('--output') + 1])
        assert not (out / 'command.json').exists()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    records = run_mocked(runner, tmp_path, grant)
    assert len(calls) == len(records) == 1
    saved = json.loads((tmp_path / 'results/consensus-desk-full-seed0/command.json').read_text())
    assert saved['command'] == calls[0]
    assert saved['processing_allowance_seconds'] == 795.


def test_instrumentation_entry_point_is_portable_tracked_source(runner):
    repo = Path(runner.__file__).resolve().parents[1]
    harness = repo / 'scripts/run_monocular_instrumented.py'
    assert harness.is_file()
    assert not harness.is_relative_to(repo / 'results')


def test_final_command_metadata_io_cannot_extend_the_child_deadline(runner, tmp_path, monkeypatch):
    grant, clock = prepared(runner, tmp_path, monkeypatch)
    original_write = runner._write_json
    events = []
    def delayed_write(path, value):
        if path.name == 'command.json':
            events.append('final_metadata')
            clock[0] += 70.
        original_write(path, value)
    monkeypatch.setattr(runner, '_write_json', delayed_write)
    def child(command, **kwargs):
        events.append('child')
        assert clock[0] == 0.  # no final command I/O before budgeted launch
        assert kwargs['timeout'] == 895.
        assert float(command[command.index('--max-wall-seconds') + 1]) == 865.
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    records = run_mocked(runner, tmp_path, grant)
    assert len(records) == 1 and events == ['child', 'final_metadata']
    assert records[0]['launch_wall_seconds'] == 0.
