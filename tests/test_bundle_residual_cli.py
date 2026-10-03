import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]


def run_cli(script, *arguments):
    return subprocess.run(
        [sys.executable, str(REPO / script), *map(str, arguments)],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'scripts'))
    path = REPO / 'scripts' / 'run_development_tests.py'
    spec = importlib.util.spec_from_file_location('bundle_residual_dev_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arguments', [
    ('--stereo-bundle-residuals', 'left_disparity'),
    ('--slam', '--stereo-bundle-residuals', 'left_disparity'),
    ('--stereo', '--stereo-bundle-residuals', 'left_disparity'),
])
def test_main_left_disparity_requires_slam_and_stereo(arguments):
    result = run_cli('src/main.py', *arguments)

    assert result.returncode == 2
    assert '--stereo-bundle-residuals left_disparity requires --slam --stereo' in result.stderr


def test_main_accepts_left_disparity_with_slam_stereo(tmp_path):
    result = run_cli('src/main.py', '--slam', '--stereo', '--data-root', tmp_path,
                     '--stereo-bundle-residuals', 'left_disparity')

    assert '--stereo-bundle-residuals left_disparity requires --slam --stereo' not in result.stderr


def test_evaluator_left_disparity_requires_stereo(tmp_path):
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo-bundle-residuals',
                     'left_disparity', '--output', tmp_path / 'evaluation')

    assert result.returncode == 2
    assert '--stereo-bundle-residuals left_disparity requires --stereo' in result.stderr


def test_evaluator_accepts_left_disparity_with_stereo(tmp_path):
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo',
                     '--stereo-bundle-residuals', 'left_disparity',
                     '--output', tmp_path / 'evaluation')

    assert result.returncode == 2
    assert '--stereo-bundle-residuals left_disparity requires --stereo' not in result.stderr
    assert '--data-root is required for local input' in result.stderr


def test_mapping_configuration_and_reuse_identity_distinguish_residual_modes(monkeypatch):
    module = load_runner(monkeypatch)
    default = module.current_mapping_configuration(
        'bundle', 'supported', False, False)
    ablation = module.current_mapping_configuration(
        'bundle', 'supported', False, False, 'left_disparity')

    assert default['stereo_bundle_residuals'] == 'left_right'
    assert ablation['stereo_bundle_residuals'] == 'left_disparity'
    assert default != ablation
    # The baseline has no shared-map bundle residual setting and stays isolated
    # at the compatibility mode even when the bundle variant opts into the ablation.
    baseline = module.current_mapping_configuration(
        'baseline', 'supported', False, False, 'left_disparity')
    assert baseline == {'feature_extractor': 'preserved_stereo_defaults', 'loop_mode': 'off'}


def test_runner_keeps_baseline_left_right_and_forwards_bundle_ablation(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    monkeypatch.setattr(module, 'REPO', tmp_path)
    monkeypatch.setattr(module, 'CURATED_TESTS', ())
    monkeypatch.setattr(module, 'dependency_runtime_identity', lambda: {'version': 1})
    monkeypatch.setattr(module, 'source_fingerprint', lambda *_: 'frozen-fingerprint')
    monkeypatch.setattr(module, 'validate_release_gate', lambda *args: None)
    monkeypatch.setattr(module, 'preflight', lambda *_args, **_kwargs: {
        'sha256': 'ordered-input-hash', 'frames': 80, 'cameras': [0, 1],
        'shape': [370, 1226],
    })
    monkeypatch.setattr(module, 'current_mapping_configuration',
                        lambda variant, _depth, _arbitration, _retry, residuals: (
                            {'feature_extractor': 'preserved_stereo_defaults', 'loop_mode': 'off'}
                            if variant == 'baseline' else {
                                'bundle_enabled': variant != 'map-only', 'loop_mode': 'off',
                                'stereo_bundle_residuals': residuals,
                            }))
    monkeypatch.setattr(module, 'estimate_case_runtime', lambda *args, **kwargs: {
        'estimated_seconds': 1.0, 'rate_s_per_frame': 0.01,
        'timing_history': {'accepted': [], 'rejected': [], 'usage': 'cost_estimate_only'},
    })

    commands = []

    def fake_run_owned(command, log, seconds, env, *, on_start=None):
        commands.append(list(map(str, command)))
        if '-m' in command and 'pytest' in command:
            return {'exit_code': 0, 'elapsed_s': 0.01, 'timed_out': False}
        if on_start is not None:
            on_start(12345)
        # End before evaluating: this test checks the exact command and identity,
        # not evaluator outputs.
        return {'exit_code': 1, 'elapsed_s': 0.01, 'timed_out': False}

    monkeypatch.setattr(module, 'run_owned', fake_run_owned)
    data_root, poses_root = tmp_path / 'data', tmp_path / 'poses'
    data_root.mkdir()
    poses_root.mkdir()
    (poses_root / '04.txt').write_text('reference\n', encoding='ascii')

    def run_variant(variant, residual_mode, output_name):
        output = tmp_path / output_name
        monkeypatch.setattr(sys, 'argv', [
            'run_development_tests.py', '--profile', 'quick', '--data-root', str(data_root),
            '--poses-root', str(poses_root), '--output', str(output),
            '--budget-seconds', '300', '--variants', variant,
            '--stereo-bundle-residuals', residual_mode,
        ])
        assert module.main() == 1
        cycle_path = output / 'frozen-finge' / 'quick' / 'cycle.json'
        cycle = json.loads(cycle_path.read_text(encoding='utf-8'))
        return cycle, commands[-1]

    bundle_cycle, bundle_command = run_variant('bundle', 'left_disparity', 'bundle-output')
    assert bundle_command[ bundle_command.index('--stereo-bundle-residuals') + 1 ] == 'left_disparity'
    assert bundle_cycle['requested']['stereo_bundle_residuals'] == 'left_disparity'
    assert bundle_cycle['attempts'][0]['identity']['stereo_bundle_residuals'] == 'left_disparity'

    baseline_cycle, baseline_command = run_variant('baseline', 'left_disparity', 'baseline-output')
    assert '--stereo-bundle-residuals' not in baseline_command
    assert baseline_cycle['requested']['stereo_bundle_residuals'] == 'left_disparity'
    assert baseline_cycle['attempts'][0]['identity']['stereo_bundle_residuals'] == 'left_right'


def _write_bundle_export(module, tmp_path, monkeypatch, *, residual_mode):
    repo = tmp_path / 'fixture-repo'
    source = repo / 'src'
    scripts = repo / 'scripts'
    source.mkdir(parents=True)
    scripts.mkdir()
    (source / 'fixture.py').write_text('fixture source\n', encoding='utf-8')
    evaluator_source = b'fixture evaluator\n'
    (scripts / 'evaluate_shared_slam.py').write_bytes(evaluator_source)
    monkeypatch.setattr(module, 'REPO', repo)
    configuration = {
        'bundle_enabled': True, 'loop_mode': 'off',
        'stereo_depth_policy': 'supported',
        'stereo_bundle_residuals': residual_mode,
    }
    monkeypatch.setattr(module, 'current_mapping_configuration',
                        lambda *_args: configuration)

    output = tmp_path / 'saved-bundle'
    archive = output / 'source'
    archive.mkdir(parents=True)
    source_hashes = {
        path.name: module.hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.glob('*.py')
    }
    for name in source_hashes:
        (archive / name).write_bytes((source / name).read_bytes())
    (output / 'evaluator.py').write_bytes(evaluator_source)
    identity = {
        'revision': 'fixture-revision', 'runtime_identity': {'version': 1},
        'sequence': '04', 'frames': 2, 'coverage': 'partial',
        'variant': 'bundle', 'cached': False, 'input': 'input-hash',
        'reference': 'reference-hash', 'stereo_depth_policy': 'supported',
        'stereo_pose_arbitration': False, 'stereo_raw_reference_retry': False,
        'stereo_bundle_residuals': residual_mode,
    }
    report = {
        'development_identity': identity, 'status': 'completed', 'frames': 2,
        'coverage': 'partial', 'sequence': '04', 'stereo': True,
        'ground_truth_used_for_estimation': False, 'lost_frames': 0,
        'landmarks': 1, 'states': {'tracking': 2},
        'configuration': configuration,
        'source_sha256': source_hashes,
        'evaluator_sha256': module.hashlib.sha256(evaluator_source).hexdigest(),
    }
    (output / 'evaluation.json').write_text(json.dumps(report), encoding='utf-8')
    (output / 'run.json').write_text(json.dumps({
        'configuration': configuration, 'sparse_points': 1,
        'tracking': [{'frame': 0, 'state': 'tracking'},
                     {'frame': 1, 'state': 'tracking'}],
    }), encoding='utf-8')
    (output / 'preview.json').write_text(json.dumps({
        'trajectory': [[0, 0, 0], [1, 0, 0]], 'sparse': [[0, 0, 1]],
    }), encoding='utf-8')
    (output / 'sparse.ply').write_text(
        'ply\nformat ascii 1.0\nelement vertex 1\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n0 0 1 10 20 30\n', encoding='ascii')
    (output / 'poses.txt').write_text(
        '1 0 0 0 0 1 0 0 0 0 1 0\n'
        '1 0 0 1 0 1 0 0 0 0 1 0\n', encoding='ascii')
    return output / 'evaluation.json', identity


def test_completed_export_reuse_requires_exact_residual_mode(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    report_path, identity = _write_bundle_export(
        module, tmp_path, monkeypatch, residual_mode='left_disparity')

    assert module.reusable_export(report_path, identity)

    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['configuration']['stereo_bundle_residuals'] = 'left_right'
    run_path = report_path.parent / 'run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run['configuration']['stereo_bundle_residuals'] = 'left_right'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    run_path.write_text(json.dumps(run), encoding='utf-8')
    assert not module.reusable_export(report_path, identity)


def test_legacy_left_right_history_remains_cost_only_compatible(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    monkeypatch.setattr(module, 'current_mapping_configuration',
                        lambda _variant, _depth, _arbitration, _retry, residuals: {
                            'bundle_enabled': True, 'loop_mode': 'off',
                            'stereo_depth_policy': 'supported',
                            'stereo_bundle_residuals': residuals,
                        })
    identity = {
        'revision': 'new-source', 'sequence': '04', 'frames': 2,
        'input': 'ordered-input', 'reference': 'reference', 'variant': 'bundle',
        'cached': False, 'stereo_depth_policy': 'supported',
        'stereo_pose_arbitration': False, 'stereo_raw_reference_retry': False,
        'stereo_bundle_residuals': 'left_right',
        'performance': {'matching_backend': 'cpu', 'retrieval': 'current',
                        'cpu_optimizations': True, 'opencv_threads': 1},
        'runtime_identity': {'version': 1},
    }
    legacy_identity = dict(identity, revision='old-source')
    legacy_identity.pop('stereo_bundle_residuals')
    legacy_configuration = module.current_mapping_configuration(
        'bundle', 'supported', False, False, 'left_right')
    legacy_configuration.pop('stereo_bundle_residuals')
    path = tmp_path / 'legacy-history.json'
    path.write_text(json.dumps({
        'sequence': '04', 'frames': 2, 'stereo': True, 'coverage': 'partial',
        'status': 'completed', 'total_wall_seconds': 3.0,
        'configuration': legacy_configuration, 'feature_cache': {'enabled': False},
        'development_identity': legacy_identity,
        'performance_configuration': {
            'matching_backend': 'cpu', 'retrieval': 'current',
            'cpu_optimizations': True, 'profile': False,
        },
        'matching_backend': {'requested': 'cpu'}, 'opencv_threads': 1,
    }), encoding='utf-8')

    history = module.inspect_timing_history(
        [path], identity, 'partial',
        module.current_mapping_configuration('bundle', 'supported', False,
                                             False, 'left_right'),
        'new-source')

    assert len(history['accepted']) == 1
    assert history['accepted'][0]['use'] == 'cost_estimate_only'
    assert 'metrics' not in history['accepted'][0]


def test_timing_history_rejects_a_different_residual_mode(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    monkeypatch.setattr(module, 'current_mapping_configuration',
                        lambda _variant, _depth, _arbitration, _retry, residuals: {
                            'bundle_enabled': True, 'loop_mode': 'off',
                            'stereo_depth_policy': 'supported',
                            'stereo_bundle_residuals': residuals,
                        })
    expected_identity = {
        'revision': 'new-source', 'sequence': '04', 'frames': 2,
        'input': 'ordered-input', 'reference': 'reference', 'variant': 'bundle',
        'cached': False, 'stereo_depth_policy': 'supported',
        'stereo_pose_arbitration': False, 'stereo_raw_reference_retry': False,
        'stereo_bundle_residuals': 'left_disparity',
        'performance': {'matching_backend': 'cpu', 'retrieval': 'current',
                        'cpu_optimizations': True, 'opencv_threads': 1},
        'runtime_identity': {'version': 1},
    }
    old_config = module.current_mapping_configuration('bundle', 'supported', False,
                                                       False, 'left_right')
    old_identity = dict(expected_identity, revision='old-source',
                        stereo_bundle_residuals='left_right')
    path = tmp_path / 'history.json'
    path.write_text(json.dumps({
        'sequence': '04', 'frames': 2, 'stereo': True, 'coverage': 'partial',
        'status': 'completed', 'total_wall_seconds': 3.0,
        'configuration': old_config, 'feature_cache': {'enabled': False},
        'development_identity': old_identity,
        'performance_configuration': {
            'matching_backend': 'cpu', 'retrieval': 'current',
            'cpu_optimizations': True, 'profile': False,
        },
        'matching_backend': {'requested': 'cpu'}, 'opencv_threads': 1,
    }), encoding='utf-8')

    history = module.inspect_timing_history(
        [path], expected_identity, 'partial',
        module.current_mapping_configuration('bundle', 'supported', False,
                                             False, 'left_disparity'),
        'new-source')

    assert not history['accepted']
    assert 'mismatched stereo bundle residual mode' in history['rejected'][0]['reason']
