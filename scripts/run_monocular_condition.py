"""Run one full TUM monocular condition only after explicit owner admission.

This is not an admission authority: receipt contents, cancellation support and
reserve justification need owner review. A lost executor transport still leaves
the job uncertain. No cleanup of other processes or shared locks is attempted.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from monocular_run_guard import check_sequence_start


FRAMES = {'desk': 613, 'room': 1362}
EVALUATOR_MIN_FREE_MIB = 5120
# Existing evaluator startup includes a further 256 MiB export reserve.
STARTUP_MIN_FREE_BYTES = (EVALUATOR_MIN_FREE_MIB + 256) * 1024**2
FLAGS = {
    'alloff': [],
    'consensus': ['--monocular-refined-consensus'],
    'recovery': ['--monocular-recovery-fallback'],
    'numerical': ['--monocular-scale-safe-triangulation'],
    'covisible-numerical': ['--monocular-covisible-triangulation',
                           '--monocular-scale-safe-triangulation'],
}
# An input JSON file cannot resolve this external fact. Remove this interlock
# only in a reviewed follow-up after the parent records authoritative terminal
# evidence tied to the original job and explicitly releases shared CPU ownership.
UNRESOLVED_CPU_OWNER = (
    'Original shared-CPU job termination is unproven; '
    'a manifest, missing PID or new namespace is not a CPU release'
)


def require_cpu_release():
    if UNRESOLVED_CPU_OWNER is not None:
        raise ValueError(UNRESOLVED_CPU_OWNER)


def _positive(name, value):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must contain an explicit owner statement')


def _receipt(base, name, receipt):
    path = (base / receipt['path']).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'{name} must be a repository-local evidence file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError(f'{name} receipt hash mismatch')


def load_admission(path, base, condition, seed):
    """Check declared scope and immutable receipts, not their truth/authority."""
    grant = json.loads(path.read_text())
    if (grant.get('schema_version') != 1 or grant.get('condition') != condition
            or type(grant.get('seed')) is not int or grant['seed'] != seed):
        raise ValueError('admission must match schema, single condition and seed')
    for name in ('owner', 'reserve_justification', 'cancellation_justification'):
        _text(name, grant.get(name))
    for name in ('prior_job_terminal_receipt', 'cpu_release_receipt'):
        _receipt(base, name, grant[name])
    for name in ('total_seconds', 'shutdown_export_reserve_seconds',
                 'terminal_receipt_reserve_seconds'):
        _positive(name, grant[name])
    deadline = dt.datetime.fromisoformat(grant['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline.utcoffset() != dt.timedelta(0):
        raise ValueError('admission deadline_utc must have UTC timezone')
    if deadline <= dt.datetime.now(dt.timezone.utc):
        raise ValueError('admission deadline has expired')
    for scene, expected in FRAMES.items():
        estimate = grant['sequences'][scene]
        if (type(estimate.get('start_frame')) is not int or estimate['start_frame'] != 0
                or type(estimate.get('frames')) is not int or estimate['frames'] != expected):
            raise ValueError(f'{scene} admission must name the complete input range')
        _positive(f'{scene} expected_launch_seconds', estimate['expected_launch_seconds'])
        _positive(f'{scene} uncertainty_seconds', estimate['uncertainty_seconds'])
        _text(f'{scene} estimate_evidence', estimate.get('estimate_evidence'))
    return grant, deadline


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def completion_summary(evaluation, scene, expected):
    """Validate complete processing, not trajectory quality or scientific gates.

    Keep raw exports unchanged. Validate before copying fields into strict JSON
    terminal receipts so malformed/nonfinite data cannot erase terminal evidence.
    Initialization failure and tracking loss are legitimate full-range outcomes.
    """
    if not isinstance(evaluation, dict):
        raise ValueError('evaluation must be an object')
    if evaluation.get('dataset') != 'tum' or evaluation.get('sequence') != 'freiburg1_' + scene:
        raise ValueError('evaluation dataset/sequence does not match launched scene')
    frames = evaluation.get('frames')
    if type(frames) is not int or frames != expected:
        raise ValueError('evaluation must account for the complete integer frame range')
    if evaluation.get('coverage') != 'full':
        raise ValueError('evaluation coverage is not full')
    if evaluation.get('status') not in (
            'completed', 'completed_with_tracking_loss', 'initialization_failed'):
        raise ValueError('evaluation status is not a terminal full-range outcome')
    if 'interruption' not in evaluation or evaluation['interruption'] is not None:
        raise ValueError('evaluation has missing or nonempty interruption evidence')
    storage = evaluation.get('storage_guard')
    if not isinstance(storage, dict) or 'interruption' not in storage or storage['interruption'] is not None:
        raise ValueError('evaluation has missing or nonempty storage interruption evidence')
    states = evaluation.get('states')
    if (not isinstance(states, dict) or not states
            or not set(states) <= {'initializing', 'tracking', 'lost', 'relocalized'}
            or any(type(count) is not int or count < 0 for count in states.values())
            or sum(states.values()) != expected):
        raise ValueError('evaluation states must exhaustively account for every frame')
    memory = evaluation.get('peak_memory_mb')
    try:
        finite_memory = math.isfinite(memory)
    except (TypeError, OverflowError):
        finite_memory = False
    if (isinstance(memory, bool) or not isinstance(memory, (int, float))
            or not finite_memory or memory < 0):
        raise ValueError('evaluation peak_memory_mb must be finite and nonnegative')
    return {'frames': frames, 'states': states, 'full_range': True,
            'peak_memory_mb': memory, 'coverage': evaluation['coverage'],
            'evaluation_status': evaluation['status']}


def run_condition(base, root, condition, seed, grant, wall_deadline):
    require_cpu_release()
    started = time.monotonic()
    # Both the absolute owner deadline and this invocation's granted duration apply.
    available = min(grant['total_seconds'],
                    (wall_deadline - dt.datetime.now(dt.timezone.utc)).total_seconds())
    deadline = started + available
    result_path = root / f'{condition}-batch-seed{seed}-result.json'
    if result_path.exists():
        raise ValueError('Never overwrite an existing batch result')
    root.mkdir(parents=True, exist_ok=True)
    records = []
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'MPLCONFIGDIR': str(base / '.mpl-cache'),
           'XDG_CACHE_HOME': str(base / '.cache')}
    try:
        for scene, expected in FRAMES.items():
            out = root / f'{condition}-{scene}-full-seed{seed}'
            if out.exists():
                raise ValueError('Never overwrite an existing run')
            estimate = grant['sequences'][scene]
            decision = check_sequence_start(
                remaining_seconds=max(0., deadline - time.monotonic()),
                expected_launch_seconds=estimate['expected_launch_seconds'],
                uncertainty_seconds=estimate['uncertainty_seconds'],
                shutdown_export_reserve_seconds=grant['shutdown_export_reserve_seconds'],
                terminal_receipt_reserve_seconds=grant['terminal_receipt_reserve_seconds'],
                disk_free_bytes=shutil.disk_usage(base).free,
                minimum_free_bytes=STARTUP_MIN_FREE_BYTES,
            )
            if not decision['planning_fits']:
                records.append({'scene': scene, 'not_started': decision['blockers'],
                                'planning_decision': decision})
                break
            out.mkdir()
            child_timeout = (deadline - time.monotonic()
                             - grant['terminal_receipt_reserve_seconds'])
            processing_allowance = child_timeout - grant['shutdown_export_reserve_seconds']
            data = base / '.datasets' / 'tum' / f'rgbd_dataset_freiburg1_{scene}'
            command = [str(base / '.venv/bin/python'),
                       str(base / 'scripts/run_monocular_instrumented.py'), '--entry-point',
                       str(base / 'scripts/evaluate_shared_slam.py'), '--frame-timings',
                       str(out / 'frame-latencies.json'), '--', '--dataset', 'tum',
                       '--data-root', str(data), '--poses-root', str(data),
                       '--sequence', 'freiburg1_' + scene, '--output', str(out),
                       '--seed', str(seed), '--features', '1500', '--retrieval', 'current',
                       '--opencv-threads', '1', '--matching-backend', 'cpu',
                       '--loop-mode', 'live', '--min-free-mb', str(EVALUATOR_MIN_FREE_MIB),
                       '--max-wall-seconds', str(processing_allowance), *FLAGS[condition]]
            command_record = {
                'command': command, 'environment': {k: env[k] for k in (
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'MPLCONFIGDIR', 'XDG_CACHE_HOME')},
                'expected_frames': expected, 'condition': condition, 'seed': seed,
                'planning_decision': decision, 'admission': grant,
                'source_sha256': {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ('src', 'scripts')
                                  for p in (base / folder).glob('*.py')},
                'harness_version': 2,
            }
            _write_json(out / 'preparation.json', command_record)
            print('PREPARED', condition, scene, flush=True)
            with (out / 'stdout-stderr.log').open('w') as log:
                # Finish potentially blocking preparation, log opening, stdout
                # and free-disk I/O BEFORE allocating relative child budgets.
                free = shutil.disk_usage(base).free
                began = time.monotonic()
                decision = check_sequence_start(
                    remaining_seconds=max(0., deadline - began),
                    expected_launch_seconds=estimate['expected_launch_seconds'],
                    uncertainty_seconds=estimate['uncertainty_seconds'],
                    shutdown_export_reserve_seconds=grant['shutdown_export_reserve_seconds'],
                    terminal_receipt_reserve_seconds=grant['terminal_receipt_reserve_seconds'],
                    disk_free_bytes=free, minimum_free_bytes=STARTUP_MIN_FREE_BYTES,
                )
                if not decision['planning_fits']:
                    records.append({'scene': scene, 'not_started': decision['blockers'],
                                    'planning_decision': decision})
                    break
                child_timeout = deadline - began - grant['terminal_receipt_reserve_seconds']
                processing_allowance = child_timeout - grant['shutdown_export_reserve_seconds']
                command[command.index('--max-wall-seconds') + 1] = str(processing_allowance)
                command_record['planning_decision'] = decision
                command_record['child_timeout_seconds'] = child_timeout
                command_record['processing_allowance_seconds'] = processing_allowance
                # No metadata writes between this allocation and owned launch.
                # Persist the exact launched command after terminal acknowledgement.
                try:
                    child = subprocess.run(command, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=child_timeout)
                    status, timed_out = child.returncode, False
                except subprocess.TimeoutExpired:
                    # subprocess.run kills and waits for its owned child before raising.
                    # This is not evidence about a different/lost executor's process.
                    status, timed_out = 124, True
            record = {'condition': condition, 'scene': scene, 'exit': status,
                      'timed_out': timed_out, 'owned_child_wait_observed': True,
                      'launch_wall_seconds': time.monotonic() - began,
                      'utc_terminal': dt.datetime.now(dt.timezone.utc).isoformat()}
            # Persist terminal acknowledgement before interpreting optional exports.
            records.append(record)
            _write_json(out / 'launch-result.json', record)
            _write_json(out / 'command.json', command_record)
            if (out / 'evaluation.json').exists():
                try:
                    evaluation = json.loads((out / 'evaluation.json').read_text())
                    record.update(completion_summary(evaluation, scene, expected))
                except (ValueError, KeyError, TypeError) as error:
                    record.update(full_range=False, evaluation_error=str(error))
            _write_json(out / 'launch-result.json', record)
            print('TERMINAL', json.dumps(record), flush=True)
            if status != 0 or record.get('full_range') is not True:
                break
    finally:
        _write_json(result_path, {'harness_version': 2,
                                 'granted_duration_seconds': grant['total_seconds'],
                                 'elapsed_seconds': time.monotonic() - started, 'runs': records})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True, choices=FLAGS)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args(argv)
    base = Path.cwd()
    try:
        require_cpu_release()
        grant, deadline = load_admission(args.admission, base, args.condition, args.seed)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    records = run_condition(base, base / 'results/vm-recovery-20261026',
                            args.condition, args.seed, grant, deadline)
    return 0 if len(records) == 2 and all(
        r.get('exit') == 0 and not r.get('timed_out') and r.get('full_range') is True
        for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
