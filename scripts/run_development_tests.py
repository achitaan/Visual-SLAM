"""Sequential, budgeted diagnostic cycles; incomplete work is never reused as a pass."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmark_identity import (
    _finite_numeric_file,
    _finite_ply,
    _load_finite_json,
    source_contract,
)
from test_budget import Budget, write_json
from data_preflight import preflight

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {
    'baseline': [],
    'map-only': ['--disable-bundle', '--loop-mode', 'off'],
    'bundle': ['--loop-mode', 'off'],
    'live': ['--loop-mode', 'live'],
    'offline': ['--loop-mode', 'offline'],
}
CLEANUP_RESERVE_SECONDS = 10
TIMING_SAFETY_MARGIN = 1.25
CURATED_TESTS = (
    'tests/test_shared_slam.py',
    'tests/test_physical_landmark_identity.py',
    'tests/test_keyframe_retrieval.py',
    'tests/test_keyframe_flow_support.py',
    'tests/test_append_only_corrections.py',
    'tests/test_local_bundle_landmarks.py',
    'tests/test_bundle_stereo_motion.py',
    'tests/test_bundle_diagnostics.py',
    'tests/test_bundle_diagnostics_cli.py',
    'tests/test_stereo_subpixel_depth.py',
    'tests/test_verified_stereo_depth.py',
    'tests/test_verified_all_stereo_acquisition.py',
    'tests/test_bidirectional_refinement.py',
    'tests/test_stereo_motion_prior.py',
    'tests/test_stereo_map_refinement.py',
    'tests/test_stereo_feature_support.py',
    'tests/test_descriptor_fallback.py',
    'tests/test_feature_cache.py',
    'tests/test_integration_foundations.py',
    'tests/test_loop_performance_integration.py',
    'tests/test_tracking_performance_integration.py',
    'tests/test_dropout_cleanup_equivalence.py',
    'tests/test_bundle_performance_equivalence.py',
    'tests/test_descriptor_matching_cuda.py',
    'tests/test_stereo_regression_diagnostics.py',
    'tests/test_test_budget.py',
    'tests/test_development_runner.py',
    'tests/test_development_timing_history.py',
    'tests/test_stereo_pose_arbitration.py',
    'tests/test_stereo_arbitration_tracking.py',
    'tests/test_stereo_reference_retention.py',
    'tests/test_stereo_full_pool_fallback.py',
    'tests/test_stereo_hard_conflict_selection.py',
    'tests/test_stereo_pose_arbitration_cli.py',
    'tests/test_stereo_raw_reference_retry.py',
    'tests/test_owned_stereo_bundle_wiring.py',
    'tests/test_owned_stereo_bundle.py',
    'tests/test_free_source_stereo_bundle.py',
    'tests/test_target_relative_stereo_bundle.py',
    'tests/test_bundle_solver_accuracy_control.py',
    'tests/test_bundle_stereo_motion.py',
)


def terminate_owned_tree(process):
    if os.name != 'nt':
        import signal
        os.killpg(process.pid,signal.SIGKILL)
        return
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),('name',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    snapshot=kernel.CreateToolhelp32Snapshot(2,0);entry=Entry();entry.size=ctypes.sizeof(entry);rows=[]
    if kernel.Process32FirstW(snapshot,ctypes.byref(entry)):
        while True:
            rows.append((entry.pid,entry.parent))
            if not kernel.Process32NextW(snapshot,ctypes.byref(entry)):break
    kernel.CloseHandle(snapshot)
    owned={process.pid};order=[process.pid]
    while True:
        children=[pid for pid,parent in rows if parent in owned and pid not in owned]
        if not children:break
        owned.update(children);order.extend(children)
    # The root PID comes from our Popen handle; only its enumerated descendants qualify.
    for pid in reversed(order):
        handle=kernel.OpenProcess(1,False,pid)
        if handle:
            try:kernel.TerminateProcess(handle,1)
            finally:kernel.CloseHandle(handle)
    if process.poll() is None:process.kill()


def run_owned(command, log, seconds, env, *, on_start=None):
    """Terminate only this supervisor's process tree if a worker misses its deadline."""
    started = time.monotonic()
    with Path(log).open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        if on_start is not None:
            on_start(process.pid)
        try:
            code = process.wait(timeout=seconds)
            return {'exit_code': code, 'elapsed_s': time.monotonic()-started, 'timed_out': False}
        except subprocess.TimeoutExpired:
            terminate_owned_tree(process)
            process.wait(timeout=5)
            return {'exit_code': process.returncode, 'elapsed_s': time.monotonic()-started, 'timed_out': True}


def child_timeout(budget, maximum, reserve=CLEANUP_RESERVE_SECONDS):
    """Cap child runtime while leaving time for terminating it and saving results."""
    return max(0.0, min(float(maximum), budget.remaining - reserve))


def _finite_positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def dependency_runtime_identity(repo=None):
    """Expose the lockfiles and installed runtime versions used by this cycle."""
    repo = REPO if repo is None else repo
    contract = source_contract(repo)
    return {
        'version': 1,
        'dependencies': contract['dependencies'],
        'dependencies_sha256': contract['dependencies_sha256'],
    }


def current_mapping_configuration(variant, stereo_depth_policy, stereo_pose_arbitration=False,
                                  stereo_raw_reference_retry=False,
                                  stereo_owned_image_bundle=False,
                                  bundle_solver_accuracy='default',
                                  stereo_source_history_bundle=False):
    """Return the exact evaluator config represented by a development variant."""
    if bundle_solver_accuracy not in ('default', 'precise'):
        raise ValueError('bundle_solver_accuracy must be default or precise')
    if variant == 'baseline':
        if stereo_source_history_bundle:
            raise ValueError('source history is not available for baseline')
        if bundle_solver_accuracy != 'default':
            raise ValueError('precise bundle solver accuracy is not available for baseline')
        return {'feature_extractor': 'preserved_stereo_defaults', 'loop_mode': 'off'}
    source = str(REPO / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    from shared_slam import MappingConfig
    loop_mode = {'map-only': 'off', 'bundle': 'off', 'live': 'live', 'offline': 'offline'}[variant]
    return dict(MappingConfig(bundle_enabled=variant != 'map-only', loop_mode=loop_mode,
                              stereo_depth_policy=stereo_depth_policy,
                              stereo_pose_arbitration=stereo_pose_arbitration,
                              stereo_raw_reference_retry=stereo_raw_reference_retry,
                              stereo_owned_image_bundle=stereo_owned_image_bundle,
                              bundle_solver_accuracy=bundle_solver_accuracy,
                              stereo_source_history_bundle=stereo_source_history_bundle).__dict__)


def inspect_timing_history(paths, identity, coverage, configuration, current_revision):
    """Read explicit evaluation reports as runtime evidence only, never as reusable results."""
    accepted, rejected = [], []
    expected_performance = identity.get('performance')
    for path in paths:
        item = {'path': str(Path(path).expanduser().resolve()), 'use': 'cost_estimate_only'}
        try:
            evidence_path = Path(path).expanduser()
            if not evidence_path.is_file():
                raise ValueError('not a readable evaluation file')
            raw = evidence_path.read_bytes()
            item['evidence_sha256'] = hashlib.sha256(raw).hexdigest()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError('evaluation root is not an object')
            source_identity = report.get('development_identity')
            if not isinstance(source_identity, dict):
                raise ValueError('missing development identity')
            runtime_identity = identity.get('runtime_identity')
            if (not isinstance(runtime_identity, dict)
                    or source_identity.get('runtime_identity') != runtime_identity):
                raise ValueError('mismatched dependency/runtime identity')
            if report.get('status') not in ('completed', 'completed_with_tracking_loss'):
                raise ValueError('evaluation is incomplete')
            for field, expected in (('sequence', identity.get('sequence')),
                                    ('frames', identity.get('frames'))):
                if report.get(field) != expected or source_identity.get(field) != expected:
                    raise ValueError(f'mismatched {field}')
            if source_identity.get('variant') != identity.get('variant'):
                raise ValueError('mismatched variant')
            if (identity.get('variant') != 'baseline'
                    and source_identity.get('stereo_pose_arbitration')
                    is not identity.get('stereo_pose_arbitration')):
                raise ValueError('mismatched stereo-pose-arbitration mode')
            if (identity.get('variant') != 'baseline'
                    and source_identity.get('stereo_raw_reference_retry')
                    is not identity.get('stereo_raw_reference_retry')):
                raise ValueError('mismatched stereo-raw-reference-retry mode')
            if (identity.get('variant') != 'baseline'
                    and source_identity.get('stereo_owned_image_bundle', False)
                    is not identity.get('stereo_owned_image_bundle', False)):
                raise ValueError('mismatched owned-stereo-image-bundle mode')
            if (identity.get('variant') != 'baseline'
                    and source_identity.get('stereo_source_history_bundle', False)
                    is not identity.get('stereo_source_history_bundle', False)):
                raise ValueError('mismatched stereo-source-history-bundle mode')
            expected_accuracy = identity.get('bundle_solver_accuracy', 'default')
            historical_accuracy = source_identity.get('bundle_solver_accuracy')
            report_accuracy = report.get('bundle_solver_accuracy')
            if identity.get('variant') != 'baseline':
                if expected_accuracy not in ('default', 'precise'):
                    raise ValueError('invalid requested bundle solver accuracy mode')
                if not isinstance(configuration, dict):
                    raise ValueError('missing expected evaluator configuration')
                expected_configuration = dict(configuration)
                expected_configured_accuracy = expected_configuration.get('bundle_solver_accuracy')
                if expected_accuracy == 'precise':
                    if expected_configured_accuracy != 'precise':
                        raise ValueError('precise timing history requires an explicit precise expected configuration')
                else:
                    if expected_configured_accuracy not in (None, 'default'):
                        raise ValueError('mismatched expected bundle solver accuracy mode')
                    # Old explicit cost-history fixtures can predate this opt-in field.
                    # Normalize only default mode; precise history requires explicit provenance.
                    expected_configuration.setdefault('bundle_solver_accuracy', 'default')
                historical_configuration = report.get('configuration')
                if not isinstance(historical_configuration, dict):
                    raise ValueError('missing evaluator configuration')
                configured_accuracy = historical_configuration.get('bundle_solver_accuracy')
                observed_modes = (historical_accuracy, report_accuracy, configured_accuracy)
                if expected_accuracy == 'precise':
                    if any(value != 'precise' for value in observed_modes):
                        raise ValueError('missing or mismatched precise bundle solver accuracy mode')
                else:
                    if any(value not in (None, 'default') for value in observed_modes):
                        raise ValueError('mismatched bundle solver accuracy mode')
                    # Explicit historical evaluation files may predate this opt-in field.
                    # Treat absent mode as default only for cost estimation, never reuse.
                    historical_configuration = dict(historical_configuration)
                    historical_configuration.setdefault('bundle_solver_accuracy', 'default')
                    if historical_accuracy is None or report_accuracy is None or configured_accuracy is None:
                        item['bundle_solver_accuracy_legacy_default_normalized'] = True
            elif expected_accuracy != 'default':
                raise ValueError('precise bundle solver accuracy is not available for baseline')
            historical_diagnostics = source_identity.get('bundle_diagnostics_enabled', False)
            historical_frames = source_identity.get('bundle_diagnostics_frames', [])
            if (historical_diagnostics is not identity.get('bundle_diagnostics_enabled', False)
                    or historical_frames != identity.get('bundle_diagnostics_frames', [])):
                raise ValueError('mismatched bundle-diagnostic capture mode')
            if report.get('stereo') is not True:
                raise ValueError('sensor mode is not stereo')
            if report.get('coverage') != coverage:
                raise ValueError('mismatched frame coverage')
            if source_identity.get('cached') is not identity.get('cached'):
                raise ValueError('mismatched feature-cache timing category')
            cache_metadata = report.get('feature_cache')
            if identity.get('variant') != 'baseline':
                if not isinstance(cache_metadata, dict) or cache_metadata.get('enabled') is not identity.get('cached'):
                    raise ValueError('missing or mismatched evaluator feature-cache metadata')
            elif isinstance(cache_metadata, dict) and cache_metadata.get('enabled'):
                raise ValueError('mismatched baseline feature-cache metadata')
            if identity.get('variant') == 'baseline':
                historical_configuration = report.get('configuration')
                expected_configuration = configuration
            if historical_configuration != expected_configuration:
                raise ValueError('mismatched evaluator configuration')

            if identity.get('variant') == 'baseline':
                historical_performance = source_identity.get('performance')
                if historical_performance not in (None, {'preserved_defaults': True}):
                    raise ValueError('mismatched baseline performance mode')
            elif source_identity.get('performance') != expected_performance:
                raise ValueError('mismatched backend/performance mode')
            elif identity.get('variant') != 'baseline':
                performance_metadata = report.get('performance_configuration')
                backend_metadata = report.get('matching_backend')
                if not isinstance(performance_metadata, dict) or any(
                        performance_metadata.get(field) != expected_performance[field]
                        for field in ('matching_backend', 'retrieval', 'cpu_optimizations')):
                    raise ValueError('missing or mismatched evaluator performance configuration')
                if performance_metadata.get('profile') is not False:
                    raise ValueError('mismatched evaluator profiling mode')
                if (not isinstance(backend_metadata, dict)
                        or backend_metadata.get('requested') != expected_performance['matching_backend']):
                    raise ValueError('missing or mismatched evaluator backend metadata')
                if report.get('opencv_threads') != expected_performance['opencv_threads']:
                    raise ValueError('mismatched evaluator OpenCV thread count')

            expected_depth = identity.get('stereo_depth_policy')
            if identity.get('variant') != 'baseline':
                historical_depths = [value for value in (
                    source_identity.get('stereo_depth_policy'), report.get('stereo_depth_policy'),
                    report.get('configuration', {}).get('stereo_depth_policy')) if value is not None]
                if not historical_depths or any(value != expected_depth for value in historical_depths):
                    raise ValueError('missing or mismatched stereo-depth policy')

            for field in ('input', 'reference'):
                expected = identity.get(field)
                historical = source_identity.get(field)
                if expected is not None and historical is None:
                    raise ValueError(f'missing ordered {field} fingerprint')
                if historical is not None and expected is not None and historical != expected:
                    raise ValueError(f'mismatched ordered {field} fingerprint')

            revision = source_identity.get('revision')
            if not isinstance(revision, str) or not revision:
                raise ValueError('missing source revision')
            elapsed = report.get('total_wall_seconds')
            timing_field = 'total_wall_seconds'
            if not _finite_positive(elapsed):
                elapsed = report.get('elapsed_s')
                timing_field = 'elapsed_s'
            if not _finite_positive(elapsed):
                raise ValueError('missing finite completed-run timing')
            frames = identity['frames']
            item.update(source_revision=revision,
                        source_revision_matches=(revision == current_revision),
                        rate_s_per_frame=float(elapsed) / frames,
                        timing_field=timing_field, elapsed_seconds=float(elapsed),
                        frames=frames)
            accepted.append(item)
        except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            item['reason'] = str(error)
            rejected.append(item)
    return {'accepted': accepted, 'rejected': rejected, 'usage': 'cost_estimate_only'}


def estimate_case_runtime(frames, existing_samples, history_paths, identity, coverage,
                          configuration, current_revision, fallback_rate):
    """Use the slowest compatible observed rate with margin; never infer a pass from timing."""
    history = inspect_timing_history(history_paths, identity, coverage, configuration,
                                     current_revision)
    rates = [{'rate_s_per_frame': float(rate), 'source': 'current_cycle'}
             for rate in existing_samples if _finite_positive(rate)]
    rates.extend({'rate_s_per_frame': row['rate_s_per_frame'],
                  'source': 'historical_evaluation', 'path': row['path'],
                  'source_revision': row['source_revision'],
                  'source_revision_matches': row['source_revision_matches'],
                  'use': 'cost_estimate_only'} for row in history['accepted'])
    if not rates:
        rates = [{'rate_s_per_frame': float(fallback_rate), 'source': 'conservative_default'}]
    selected = max(rates, key=lambda row: row['rate_s_per_frame'])
    rate = selected['rate_s_per_frame']
    return {'rate_s_per_frame': rate, 'estimated_seconds': frames * rate * TIMING_SAFETY_MARGIN,
            'safety_margin_factor': TIMING_SAFETY_MARGIN, 'selected_rate': selected,
            'observed_rates': rates, 'timing_history': history}


def reusable(report, identity):
    return (report.get('development_identity') == identity
            and report.get('status') in ('completed','completed_with_tracking_loss')
            and report.get('frames') == identity['frames'])


def _source_archive_hashes(repo=None):
    repo = REPO if repo is None else repo
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((Path(repo) / 'src').glob('*.py'))
    }


def _poses_are_so3(path, frames, tolerance=1e-3):
    """Use the same SO(3) tolerance as kitti.validate_pose for saved rows."""
    if not _finite_numeric_file(path, pose_rows=frames):
        return False
    try:
        rows = [[float(token) for token in line.split()]
                for line in Path(path).read_text(encoding='ascii').splitlines()
                if line.strip()]
    except (OSError, UnicodeError, ValueError):
        return False
    for values in rows:
        rotation = (values[0:3], values[4:7], values[8:11])
        for first in range(3):
            for second in range(3):
                dot = sum(rotation[first][axis] * rotation[second][axis]
                          for axis in range(3))
                expected = 1.0 if first == second else 0.0
                if abs(dot - expected) > tolerance:
                    return False
        a, b, c = rotation
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if abs(determinant - 1.0) > tolerance:
            return False
    return True


def _finite_vector(values, size):
    return (isinstance(values, list) and len(values) == size
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in values))


def _ply_vertex_count(path):
    try:
        with Path(path).open(encoding='ascii') as stream:
            for line in stream:
                line = line.strip()
                if line == 'end_header':
                    break
                fields = line.split()
                if len(fields) == 3 and fields[:2] == ['element', 'vertex']:
                    return int(fields[2])
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _bundle_diagnostics_reusable(output, identity, report, run):
    """Require every requested diagnostic frame to be captured or explicitly skipped."""
    enabled = identity.get('bundle_diagnostics_enabled', False)
    frames = identity.get('bundle_diagnostics_frames', [])
    if not isinstance(enabled, bool) or not isinstance(frames, list):
        return False
    if (any(not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
            for frame in frames) or frames != sorted(set(frames))):
        return False
    report_manifest = report.get('bundle_diagnostics')
    run_manifest = run.get('bundle_diagnostics')
    if not enabled:
        if not frames:
            return (report_manifest is None or report_manifest == {
                'enabled': False, 'selected_frames': []
            }) and (run_manifest is None or run_manifest == {
                'enabled': False, 'selected_frames': []
            })
        return False
    if not frames or not isinstance(report_manifest, dict) or run_manifest != report_manifest:
        return False
    if report.get('bundle_diagnostic_errors') != [] or run.get('bundle_diagnostic_errors') != []:
        return False
    if (report_manifest.get('schema_version') != 1
            or report_manifest.get('enabled') is not True
            or report_manifest.get('selected_frames') != frames):
        return False
    max_frames = report_manifest.get('max_frames')
    max_snapshots = report_manifest.get('max_snapshots')
    if (not isinstance(max_frames, int) or isinstance(max_frames, bool)
            or not isinstance(max_snapshots, int) or isinstance(max_snapshots, bool)
            or max_frames < len(frames) or max_snapshots < 3 * len(frames)):
        return False
    records = report_manifest.get('frames')
    if not isinstance(records, list) or len(records) != len(frames):
        return False
    by_frame = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        frame = record.get('frame')
        if not isinstance(frame, int) or isinstance(frame, bool) or frame in by_frame:
            return False
        by_frame[frame] = record
    if set(by_frame) != set(frames):
        return False
    root = Path(output).resolve()
    diagnostic_root = (root / 'bundle-diagnostics').resolve()
    expected_phases = {'prepared', 'solved', 'finished'}
    for frame in frames:
        record = by_frame[frame]
        status = record.get('status')
        reason = record.get('reason')
        phases = record.get('phases')
        errors = record.get('errors', [])
        if not isinstance(errors, list) or errors:
            return False
        if status == 'skipped':
            if not isinstance(reason, str) or not reason.strip():
                return False
            if not isinstance(phases, dict):
                return False
            if 'prepared' in phases or 'solved' in phases:
                return False
        elif status == 'complete':
            if not isinstance(phases, dict) or set(phases) != expected_phases:
                return False
            image_size = record.get('image_size')
            if (not isinstance(image_size, list) or len(image_size) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                           for value in image_size)):
                return False
        else:
            return False
        if not isinstance(phases, dict):
            return False
        for phase, artifact in phases.items():
            if phase not in expected_phases or not isinstance(artifact, dict):
                return False
            relative = artifact.get('path')
            digest = artifact.get('sha256')
            if (artifact.get('status') != 'written'
                    or not isinstance(relative, str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)):
                return False
            relative_path = Path(relative)
            if relative_path.is_absolute() or '..' in relative_path.parts:
                return False
            path = (diagnostic_root / relative_path).resolve()
            if not path.is_relative_to(diagnostic_root) or not path.is_file():
                return False
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    return False
                snapshot = _load_finite_json(path)
            except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
                return False
            if not isinstance(snapshot, dict):
                return False
            snapshot_frame = snapshot.get('frame')
            snapshot_phase = snapshot.get('phase')
            if (snapshot_frame != frame or snapshot_phase != phase):
                return False
    return True


def reusable_export(report_path, identity):
    """Reuse only exact completed reports with finite poses/maps and intact source."""
    report_path = Path(report_path)
    output = report_path.parent
    try:
        report = _load_finite_json(report_path)
        if not isinstance(report, dict) or not reusable(report, identity):
            return False
        frames = identity['frames']
        if (report.get('coverage') != identity.get('coverage')
                or report.get('sequence') != identity.get('sequence')
                or report.get('stereo') is not True
                or report.get('ground_truth_used_for_estimation') is not False):
            return False
        if not _poses_are_so3(output / 'poses.txt', frames):
            return False
        run = {}
        if identity.get('variant') != 'baseline':
            expected_configuration = current_mapping_configuration(
                identity['variant'], identity['stereo_depth_policy'],
                identity['stereo_pose_arbitration'],
                identity.get('stereo_raw_reference_retry', False),
                identity.get('stereo_owned_image_bundle', False),
                identity.get('bundle_solver_accuracy', 'default'),
                identity.get('stereo_source_history_bundle', False))
            run = _load_finite_json(output / 'run.json')
            preview = _load_finite_json(output / 'preview.json')
            if not isinstance(run, dict) or not isinstance(preview, dict):
                return False
            if (report.get('configuration') != expected_configuration
                    or run.get('configuration') != expected_configuration):
                return False
            tracking = run.get('tracking')
            if not isinstance(tracking, list) or len(tracking) != frames:
                return False
            state_counts = {}
            for frame_index, item in enumerate(tracking):
                if not isinstance(item, dict):
                    return False
                frame = item.get('frame')
                if (not isinstance(frame, int) or isinstance(frame, bool)
                        or frame != frame_index):
                    return False
                state = item.get('state')
                if state not in {'initializing', 'tracking', 'lost', 'relocalized'}:
                    return False
                state_counts[state] = state_counts.get(state, 0) + 1
            lost_frames = state_counts.get('lost', 0)
            if (report.get('states') != state_counts
                    or report.get('lost_frames') != lost_frames):
                return False
            sparse_path = output / 'sparse.ply'
            if not _finite_ply(sparse_path):
                return False
            vertex_count = _ply_vertex_count(sparse_path)
            if (not isinstance(vertex_count, int) or vertex_count < 0
                    or run.get('sparse_points') != vertex_count
                    or report.get('landmarks') != vertex_count):
                return False
            trajectory = preview.get('trajectory')
            sparse_preview = preview.get('sparse')
            if (not isinstance(trajectory, list) or len(trajectory) != frames
                    or any(not _finite_vector(pose, 3) for pose in trajectory)
                    or not isinstance(sparse_preview, list)
                    or len(sparse_preview) > min(vertex_count, 20000)
                    or any(not _finite_vector(point, 3) for point in sparse_preview)):
                return False
        if not _bundle_diagnostics_reusable(output, identity, report, run):
            return False

        lost_frames = report.get('lost_frames')
        expected_status = ('completed_with_tracking_loss'
                           if isinstance(lost_frames, int) and lost_frames > 0
                           else 'completed')
        if (not isinstance(lost_frames, int) or isinstance(lost_frames, bool)
                or lost_frames < 0 or lost_frames > frames
                or report.get('status') != expected_status):
            return False

        source_hashes = report.get('source_sha256')
        current_hashes = _source_archive_hashes()
        if not isinstance(source_hashes, dict) or source_hashes != current_hashes:
            return False
        source_dir = output / 'source'
        archived = {path.name for path in source_dir.iterdir() if path.is_file()}
        if archived != set(source_hashes):
            return False
        for name, expected_hash in source_hashes.items():
            if (Path(name).name != name or not name.endswith('.py')
                    or hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                    != expected_hash):
                return False
        evaluator_name = ('evaluate_stereo_baseline.py'
                          if identity.get('variant') == 'baseline'
                          else 'evaluate_shared_slam.py')
        evaluator_hash = hashlib.sha256(
            (REPO / 'scripts' / evaluator_name).read_bytes()).hexdigest()
        if (report.get('evaluator_sha256') != evaluator_hash
                or hashlib.sha256((output / 'evaluator.py').read_bytes()).hexdigest()
                != evaluator_hash):
            return False
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def process_running(pid):
    """Read process liveness without signaling a potentially reused Windows PID."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    code = wintypes.DWORD()
    try:
        return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def resume_manifest(path, revision, profile, requested, probe=process_running):
    if not path.exists():
        return {'revision': revision, 'profile': profile, 'requested': requested, 'attempts': []}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if any(manifest.get(k) != v for k, v in
           [('revision', revision), ('profile', profile), ('requested', requested)]):
        raise ValueError('Mismatched cycle identity; preserve this manifest and use a separate output')
    if manifest.get('status') == 'running':
        pids = [manifest.get('supervisor_pid')]
        if manifest.get('active_case'):
            pids.append(manifest['active_case'].get('worker_pid'))
        if any(probe(pid) is not False for pid in pids):
            raise ValueError('Existing cycle may still be active; verify it before resuming')
        interrupted = manifest.pop('active_case', None)
        manifest.setdefault('interruptions', []).append({
            'status': 'interrupted_supervisor_exit', 'active_case': interrupted,
            'previous_supervisor_pid': manifest.get('supervisor_pid'), 'observed_unix': time.time()})
    return manifest


def validate_release_gate(profile, gate_path, fingerprint, runtime_identity=None):
    if profile != 'release':
        return
    if not gate_path:
        raise ValueError('Release runs require --release-ready')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('revision') != fingerprint or gate.get('passed') is not True:
        raise ValueError('Focused gate must pass for this exact revision')
    runtime_identity = runtime_identity or dependency_runtime_identity()
    if (gate.get('runtime_identity') != runtime_identity
            or gate.get('runtime_identity_sha256') != _canonical_sha256(runtime_identity)):
        raise ValueError('Focused gate must match this exact dependency/runtime identity')


def validation_provenance(runtime_identity=None):
    runtime_identity = runtime_identity or dependency_runtime_identity()
    return {
        'tests_sha256': hashlib.sha256(b''.join(
            p.name.encode()+b'\0'+p.read_bytes()
            for p in sorted((REPO/'tests').glob('test_*.py')))).hexdigest(),
        'runtime_identity': runtime_identity,
        'runtime_identity_sha256': _canonical_sha256(runtime_identity),
    }


def source_fingerprint(runtime_identity=None):
    runtime_identity = runtime_identity or dependency_runtime_identity()
    sources = sorted((REPO/'src').glob('*.py')) + [
        REPO/'scripts'/name for name in ('evaluate_shared_slam.py',
        'evaluate_stereo_baseline.py', 'run_development_tests.py',
        'benchmark_identity.py', 'test_budget.py', 'data_preflight.py',
        'benchmark_telemetry.py')]
    sources += sorted((REPO/'tests').glob('test_*.py'))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    digest.update(b'\0runtime-identity\0')
    digest.update(json.dumps(runtime_identity, sort_keys=True,
                             separators=(',', ':')).encode('utf-8'))
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['quick','focused','release'], default='quick')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--poses-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/stereo-rework'))
    parser.add_argument('--budget-seconds', type=float)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['bundle'])
    parser.add_argument('--release-ready', type=Path, help='Exact-revision focused gate assessment required for full runs')
    parser.add_argument('--feature-cache',type=Path,help='Bounded cached diagnostics; unavailable for release performance runs')
    parser.add_argument('--timing-history', type=Path, nargs='+', action='append', default=[],
                        help='Explicit completed evaluation JSON files used only to estimate runtime')
    parser.add_argument('--matching-backend',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--stereo-depth-policy',choices=['supported','verified_fallback','verified_all'],default='supported')
    parser.add_argument('--stereo-pose-arbitration', action='store_true',
                        help='Enable reserved-evidence stereo pose arbitration for SLAM variants')
    parser.add_argument('--stereo-raw-reference-retry', action='store_true',
                        help='Retry failed configured references with guarded raw-supported stereo geometry')
    parser.add_argument('--stereo-owned-image-bundle', action='store_true',
                        help='Use selected reserved raw stereo image rows in the local bundle')
    parser.add_argument('--stereo-source-history-bundle', action='store_true',
                        help='Add accepted source-frame map image observations to the experimental bundle')
    parser.add_argument('--bundle-solver-accuracy', choices=['default', 'precise'], default='default',
                        help='Default keeps current policy; precise applies tight LSMR tolerances to all bundle solves')
    parser.add_argument('--bundle-diagnostics-frames', type=int, nargs='+',
                        help='Capture immutable local bundle snapshots for selected frame IDs')
    parser.add_argument('--retrieval',choices=['current','indexed','exhaustive'],default='current')
    parser.add_argument('--no-cpu-optimizations',action='store_true')
    parser.add_argument('--opencv-threads',type=int,default=1)
    args=parser.parse_args()
    if args.stereo_owned_image_bundle and (
            not args.stereo_pose_arbitration or not any(v != 'baseline' for v in args.variants)):
        parser.error('--stereo-owned-image-bundle requires --stereo-pose-arbitration and a SharedSlam variant')
    if args.stereo_source_history_bundle and (not args.stereo_owned_image_bundle or 'baseline' in args.variants):
        parser.error('--stereo-source-history-bundle requires --stereo-owned-image-bundle and SharedSlam variants only')
    if args.bundle_solver_accuracy == 'precise' and 'baseline' in args.variants:
        parser.error('--bundle-solver-accuracy precise cannot be combined with the preserved baseline variant')
    bundle_diagnostics_frames = (sorted(args.bundle_diagnostics_frames)
                                 if args.bundle_diagnostics_frames is not None else [])
    if args.bundle_diagnostics_frames is not None:
        if any(frame < 0 for frame in bundle_diagnostics_frames):
            parser.error('--bundle-diagnostics-frames must contain nonnegative frame IDs')
        if len(set(bundle_diagnostics_frames)) != len(bundle_diagnostics_frames):
            parser.error('--bundle-diagnostics-frames must not contain duplicates')
        if 'baseline' in args.variants:
            parser.error('bundle diagnostics require SharedSlam variants; remove baseline')
    timing_history_paths = [path for group in args.timing_history for path in group]
    if args.profile=='release' and args.feature_cache:parser.error('Release performance must use uncached extraction')
    if args.opencv_threads < 1:parser.error('--opencv-threads must be positive')
    seconds=args.budget_seconds if args.budget_seconds is not None else (300 if args.profile=='quick' else 3600)
    if not 0 < seconds <= 3600:parser.error('Cycle budget must be within 1–3600 seconds')
    budget=Budget(seconds)
    runtime_identity = dependency_runtime_identity()
    fingerprint=source_fingerprint(runtime_identity)
    try:
        validate_release_gate(args.profile, args.release_ready, fingerprint,
                              runtime_identity)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    root=args.output/fingerprint[:12]/args.profile
    manifest_path=root/'cycle.json'
    requested={'variants': args.variants, 'data_root': str(args.data_root.resolve()),
               'poses_root': str(args.poses_root.resolve()),
               'feature_cache': str(args.feature_cache.resolve()) if args.feature_cache else None,
               'stereo_depth_policy': args.stereo_depth_policy,
               'stereo_pose_arbitration': args.stereo_pose_arbitration,
               'stereo_raw_reference_retry': args.stereo_raw_reference_retry,
               'stereo_owned_image_bundle': args.stereo_owned_image_bundle,
               'stereo_source_history_bundle': args.stereo_source_history_bundle,
               'bundle_solver_accuracy': args.bundle_solver_accuracy,
               'bundle_diagnostics_enabled': bool(bundle_diagnostics_frames),
               'bundle_diagnostics_frames': bundle_diagnostics_frames,
               'performance': {'matching_backend':args.matching_backend,'retrieval':args.retrieval,
                               'cpu_optimizations':not args.no_cpu_optimizations,'opencv_threads':args.opencv_threads}}
    try:
        manifest=resume_manifest(manifest_path, fingerprint, args.profile, requested)
    except ValueError as error:
        parser.error(str(error))
    root.mkdir(parents=True,exist_ok=True)
    manifest['validation_provenance']=validation_provenance(runtime_identity)
    manifest.update(profile=args.profile,budget_seconds=seconds,status='running',supervisor_pid=os.getpid())
    write_json(manifest_path,manifest)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8','MPLCONFIGDIR':str(REPO/'.mpl-cache')}
    checks_timeout=child_timeout(budget,120)
    if checks_timeout <= 0:
        manifest.update(status='deferred_budget',next_case={'phase':'checks','estimated_seconds':1})
        write_json(manifest_path,manifest);print(json.dumps(manifest['next_case']),flush=True);return 2
    checks=run_owned([sys.executable,'-m','pytest',*CURATED_TESTS,'-q'],root/'checks.log',checks_timeout,env)
    manifest['checks']=checks;write_json(manifest_path,manifest)
    if checks['exit_code'] != 0:
        manifest['status']='failed_checks';write_json(manifest_path,manifest);return 1
    cases=[('04',80)] if args.profile=='quick' else [('04',80),('01',350)]
    if args.profile=='release':
        cases=[('01',1101),('04',271),('00',4541),('07',1101)]
    for seq,frames in cases:
        preparation_started=time.monotonic()
        try:
            inputs=preflight(args.data_root,seq,frames,budget=budget)
        except (OSError,ValueError,TimeoutError) as error:
            manifest['attempts'].append({'sequence':seq,'status':'input_failure','error_type':type(error).__name__})
            manifest['status']='input_failure';write_json(manifest_path,manifest);return 1
        manifest.setdefault('preflight',[]).append({'sequence':seq,'frames':frames,
            'elapsed_s':time.monotonic()-preparation_started,'input':inputs})
        write_json(manifest_path,manifest)
        for variant in args.variants:
            if source_fingerprint() != fingerprint:
                manifest['status']='interrupted_source_change';write_json(manifest_path,manifest);return 1
            if (root/'stop.request').exists():
                manifest['status']='interrupted_requested_stop';write_json(manifest_path,manifest);return 1
            expected_coverage = 'full' if args.profile == 'release' else 'partial'
            identity={'revision':fingerprint,'runtime_identity':runtime_identity,
                      'sequence':seq,'frames':frames,'coverage':expected_coverage,
                      'input':inputs['sha256'],'variant':variant,
                      'cached':args.feature_cache is not None and variant!='baseline',
                      'stereo_depth_policy': args.stereo_depth_policy if variant!='baseline' else 'preserved_defaults',
                      'stereo_pose_arbitration': args.stereo_pose_arbitration if variant!='baseline' else False,
                      'stereo_raw_reference_retry': args.stereo_raw_reference_retry if variant!='baseline' else False,
                      'stereo_owned_image_bundle': args.stereo_owned_image_bundle if variant!='baseline' else False,
                      'stereo_source_history_bundle': args.stereo_source_history_bundle if variant!='baseline' else False,
                      'bundle_solver_accuracy': args.bundle_solver_accuracy if variant!='baseline' else 'default',
                      'bundle_diagnostics_enabled': bool(bundle_diagnostics_frames),
                      'bundle_diagnostics_frames': bundle_diagnostics_frames,
                      'performance': requested['performance'] if variant!='baseline' else {'preserved_defaults':True},
                      'reference':hashlib.sha256((args.poses_root/f'{seq}.txt').read_bytes()).hexdigest()}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
            folder=root/f'{seq}-{variant}-{key}'
            report_path=folder/'evaluation.json'
            if not report_path.exists():
                for candidate in (args.output/fingerprint[:12]).glob(f'*/{folder.name}/evaluation.json'):
                    if reusable_export(candidate, identity):
                        manifest['attempts'].append({'sequence':seq,'variant':variant,'output':str(candidate.parent.relative_to(args.output/fingerprint[:12])),'status':'reused'})
                        write_json(manifest_path,manifest)
                        break
                else:
                    candidate=None
                if candidate is not None:continue
            if report_path.exists() and reusable_export(report_path, identity):
                manifest['attempts'].append({'sequence':seq,'variant':variant,'output':folder.name,'status':'reused'})
                write_json(manifest_path,manifest);continue
            # Historical reports affect scheduling only. They never satisfy a case or supply metrics.
            samples=[r['elapsed_s']/r['frames'] for r in manifest['attempts']
                     if r.get('sequence') == seq and r.get('variant') == variant
                     and r.get('frames') == frames and r.get('status') in
                     ('completed', 'completed_with_tracking_loss')
                     and _finite_positive(r.get('elapsed_s'))]
            expected_configuration = current_mapping_configuration(
                variant, args.stereo_depth_policy,
                args.stereo_pose_arbitration and variant != 'baseline',
                args.stereo_raw_reference_retry and variant != 'baseline',
                args.stereo_owned_image_bundle and variant != 'baseline',
                args.bundle_solver_accuracy if variant != 'baseline' else 'default',
                args.stereo_source_history_bundle and variant != 'baseline')
            estimate_details = estimate_case_runtime(
                frames, samples, timing_history_paths, identity, expected_coverage,
                expected_configuration, fingerprint, fallback_rate=2.0 if seq == '04' else 4.0)
            estimate=estimate_details['estimated_seconds']
            manifest.setdefault('timing_estimates', []).append({
                'sequence': seq, 'variant': variant, 'coverage': expected_coverage,
                'estimate': estimate_details})
            write_json(manifest_path, manifest)
            if not budget.permits(estimate,reserve=60):
                manifest.update(status='deferred_budget',next_case={'sequence':seq,'variant':variant,'estimated_seconds':estimate})
                write_json(manifest_path,manifest);print(json.dumps(manifest['next_case']),flush=True);return 2
            folder.mkdir(parents=True,exist_ok=True)
            if report_path.exists():
                # Retain interrupted exports instead of overwriting the evidence.
                folder=folder.with_name(folder.name+'-retry-'+str(time.time_ns()));folder.mkdir()
                report_path=folder/'evaluation.json'
            command=[sys.executable,'-u',str(REPO/'scripts/evaluate_shared_slam.py'),'--stereo','--data-root',str(args.data_root),
                     '--poses-root',str(args.poses_root),'--sequence',seq,'--max-frames',str(frames),'--output',str(folder),
                     '--max-wall-seconds',str(max(1,budget.remaining-60)),*VARIANTS[variant]]
            if variant=='baseline':
                command[2]=str(REPO/'scripts/evaluate_stereo_baseline.py');command.remove('--stereo')
            else:
                command.extend(['--matching-backend',args.matching_backend,'--retrieval',args.retrieval,
                                '--opencv-threads',str(args.opencv_threads),
                                '--stereo-depth-policy',args.stereo_depth_policy])
                if args.stereo_pose_arbitration:command.append('--stereo-pose-arbitration')
                if args.stereo_raw_reference_retry:command.append('--stereo-raw-reference-retry')
                if args.stereo_owned_image_bundle:command.append('--stereo-owned-image-bundle')
                if args.stereo_source_history_bundle:command.append('--stereo-source-history-bundle')
                command.extend(['--bundle-solver-accuracy', args.bundle_solver_accuracy])
                if bundle_diagnostics_frames:
                    command.extend(['--bundle-diagnostics-dir',
                                    str((folder / 'bundle-diagnostics').resolve()),
                                    '--bundle-diagnostics-frames',
                                    *map(str, bundle_diagnostics_frames)])
                if args.no_cpu_optimizations:command.append('--no-cpu-optimizations')
                if args.feature_cache:command.extend(['--feature-cache',str(args.feature_cache)])
            command.extend(['--stop-file',str(root/'stop.request')])
            if args.profile=='release':
                index=command.index('--max-frames');del command[index:index+2]
            print(f'{seq} {variant}: {frames} frames; estimated {estimate:.0f}s; remaining {budget.remaining:.0f}s',flush=True)
            def started(pid):
                manifest['active_case']={'sequence':seq,'variant':variant,'output':folder.name,'worker_pid':pid,'started_unix':time.time()}
                write_json(manifest_path,manifest)
            result=run_owned(command,folder/'runner.log',max(1,budget.remaining-50),env,on_start=started)
            manifest.pop('active_case',None)
            row={'sequence':seq,'variant':variant,'output':folder.name,**result}
            source_unchanged = source_fingerprint() == fingerprint
            if report_path.exists():
                try:
                    report = _load_finite_json(report_path)
                    if not isinstance(report, dict):
                        raise ValueError('evaluation report root is not an object')
                    report['development_identity'] = identity
                    write_json(report_path, report)
                    row.update(frames=report['frames'], status=report['status'],
                               metrics=report.get('metrics'),
                               stage_timings=report.get('stage_timings'),
                               lost_frames=report.get('lost_frames'))
                except (OSError, ValueError, TypeError, KeyError) as error:
                    row.update(status='invalid_report',
                               error=f'{type(error).__name__}: {error}')
            else:row['status']='interrupted_time_budget' if result['timed_out'] else 'worker_failure'
            manifest['attempts'].append(row);write_json(manifest_path,manifest)
            if not source_unchanged:
                row.update(status='invalid_source',
                           error='Source, tests, dependencies, or runtime changed during evaluation')
                manifest['status'] = 'invalid_source'
                write_json(manifest_path, manifest)
                return 1
            if (row.get('status') in ('completed', 'completed_with_tracking_loss')
                    and row.get('frames') == frames
                    and not reusable_export(report_path, identity)):
                row.update(status='invalid_artifacts',
                           error='Completed evaluator exports failed exact finite/source validation')
                manifest['status'] = 'invalid_artifacts'
                write_json(manifest_path, manifest)
                return 1
            if row['status'] not in ('completed','completed_with_tracking_loss') or row.get('frames')!=frames:
                manifest['status']=row['status'];write_json(manifest_path,manifest);return 1
            plot_timeout=child_timeout(budget,15)
            if variant!='baseline' and budget.permits(15,reserve=CLEANUP_RESERVE_SECONDS) and plot_timeout > 0:
                plot=run_owned([sys.executable,str(REPO/'scripts/plot_shared_slam.py'),'--run',str(folder),'--reference',str(args.poses_root/f'{seq}.txt')],folder/'plot.log',plot_timeout,env)
                row['plot']=plot;write_json(manifest_path,manifest)
    manifest.update(status='completed_diagnostics',elapsed_s=seconds-budget.remaining)
    manifest.pop('next_case',None)
    write_json(manifest_path,manifest)
    print(manifest_path,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
