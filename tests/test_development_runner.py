import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'scripts'))
    path = REPO / 'scripts' / 'run_development_tests.py'
    spec = importlib.util.spec_from_file_location('run_development_tests_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_fingerprint_includes_test_provenance(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    repo = tmp_path
    for directory in ('src', 'scripts', 'tests'):
        (repo / directory).mkdir()
    (repo / 'src' / 'module.py').write_text('value = 1\n', encoding='utf-8')
    for name in ('evaluate_shared_slam.py', 'evaluate_stereo_baseline.py',
                 'run_development_tests.py', 'benchmark_identity.py',
                 'test_budget.py', 'data_preflight.py',
                 'benchmark_telemetry.py', 'stereo_pose_arbitration.py'):
        (repo / 'scripts' / name).write_text('# stable harness input\n', encoding='utf-8')
    test_file = repo / 'tests' / 'test_gate.py'
    test_file.write_text('def test_gate(): pass\n', encoding='utf-8')
    lock_file = repo / 'requirements-lock.txt'
    lock_file.write_text('numpy==1.0\n', encoding='utf-8')
    monkeypatch.setattr(module, 'REPO', repo)

    runtime_identity = module.dependency_runtime_identity()
    focused_revision = module.source_fingerprint(runtime_identity)
    gate = repo / 'gate.json'
    gate.write_text(json.dumps({
        'revision': focused_revision,
        'passed': True,
        'runtime_identity': runtime_identity,
        'runtime_identity_sha256': module._canonical_sha256(runtime_identity),
    }), encoding='utf-8')
    module.validate_release_gate('release', gate, focused_revision, runtime_identity)

    test_file.write_text('def test_gate(): assert True\n', encoding='utf-8')
    changed_revision = module.source_fingerprint(runtime_identity)
    assert changed_revision != focused_revision
    with pytest.raises(ValueError, match='exact revision'):
        module.validate_release_gate('release', gate, changed_revision, runtime_identity)

    test_file.write_text('def test_gate(): pass\n', encoding='utf-8')
    lock_file.write_text('numpy==1.1\n', encoding='utf-8')
    dependency_changed_runtime = module.dependency_runtime_identity()
    dependency_changed_revision = module.source_fingerprint(dependency_changed_runtime)
    assert dependency_changed_runtime['dependencies']['files']['requirements-lock.txt'] != (
        runtime_identity['dependencies']['files']['requirements-lock.txt'])
    assert dependency_changed_revision != focused_revision
    with pytest.raises(ValueError, match='exact revision'):
        module.validate_release_gate(
            'release', gate, dependency_changed_revision, dependency_changed_runtime)

    assert 'numpy' in runtime_identity['dependencies']['packages']
    package_changed_runtime = json.loads(json.dumps(runtime_identity))
    package_changed_runtime['dependencies']['packages']['numpy'] = 'different-version'
    package_changed_revision = module.source_fingerprint(package_changed_runtime)
    assert package_changed_revision != focused_revision
    with pytest.raises(ValueError, match='exact revision'):
        module.validate_release_gate(
            'release', gate, package_changed_revision, package_changed_runtime)
    with pytest.raises(ValueError, match='dependency/runtime identity'):
        module.validate_release_gate(
            'release', gate, focused_revision, package_changed_runtime)

    legacy_gate = repo / 'legacy-gate.json'
    legacy_gate.write_text(json.dumps({
        'revision': dependency_changed_revision, 'passed': True,
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='dependency/runtime identity'):
        module.validate_release_gate(
            'release', legacy_gate, dependency_changed_revision,
            dependency_changed_runtime)


def _write_completed_bundle(module, tmp_path, monkeypatch, *, variant='bundle'):
    repo = tmp_path / 'repo'
    source = repo / 'src'
    source.mkdir(parents=True)
    (source / 'fixture.py').write_text('source version one\n', encoding='utf-8')
    scripts = repo / 'scripts'
    scripts.mkdir()
    evaluator_name = ('evaluate_stereo_baseline.py'
                      if variant == 'baseline' else 'evaluate_shared_slam.py')
    evaluator_source = b'fixture evaluator source\n'
    (scripts / evaluator_name).write_bytes(evaluator_source)
    monkeypatch.setattr(module, 'REPO', repo)
    configuration = {
        'bundle_enabled': True,
        'loop_mode': 'off',
        'stereo_depth_policy': 'verified_fallback',
    }
    monkeypatch.setattr(module, 'current_mapping_configuration',
                        lambda *_args: configuration)
    output = tmp_path / 'case'
    (output / 'source').mkdir(parents=True)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob('*.py'))
    }
    for name in source_hashes:
        (output / 'source' / name).write_bytes((source / name).read_bytes())
    (output / 'evaluator.py').write_bytes(evaluator_source)
    identity = {
        'revision': 'source-fingerprint',
        'runtime_identity': {'version': 1, 'dependencies_sha256': 'runtime-hash'},
        'sequence': '04', 'frames': 2, 'coverage': 'partial', 'variant': variant,
        'cached': False, 'input': 'ordered-input-hash', 'reference': 'reference-hash',
        'stereo_depth_policy': 'verified_fallback',
        'stereo_pose_arbitration': False,
    }
    report = {
        'development_identity': identity,
        'status': 'completed', 'frames': 2, 'coverage': 'partial',
        'sequence': '04', 'stereo': True,
        'ground_truth_used_for_estimation': False,
        'lost_frames': 0,
        'configuration': configuration,
        'evaluator_sha256': hashlib.sha256(evaluator_source).hexdigest(),
        'source_sha256': source_hashes,
    }
    if variant != 'baseline':
        report.update(landmarks=1, states={'tracking': 2})
        (output / 'run.json').write_text(json.dumps({
            'sparse_points': 1,
            'configuration': configuration,
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
    report_path = output / 'evaluation.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    return output, report_path, identity


def test_completed_export_reuse_requires_finite_so3_pose_and_consistent_map(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    output, report_path, identity = _write_completed_bundle(module, tmp_path, monkeypatch)
    assert module.reusable_export(report_path, identity)

    poses = output / 'poses.txt'
    poses.write_text(
        '2 0 0 0 0 1 0 0 0 0 1 0\n'
        '1 0 0 1 0 1 0 0 0 0 1 0\n', encoding='ascii')
    assert not module.reusable_export(report_path, identity)

    poses.write_text(
        '1 0 0 0 0 1 0 0 0 0 1 0\n'
        '1 0 0 1 0 1 0 0 0 0 1 0\n', encoding='ascii')
    (output / 'sparse.ply').write_text(
        'ply\nformat ascii 1.0\nelement vertex 2\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n0 0 1 10 20 30\n', encoding='ascii')
    assert not module.reusable_export(report_path, identity)


def test_completed_export_reuse_rejects_nonfinite_and_tampered_source(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    output, report_path, identity = _write_completed_bundle(module, tmp_path, monkeypatch)
    (output / 'preview.json').write_text('{"trajectory":[[NaN,0,0]]}', encoding='utf-8')
    assert not module.reusable_export(report_path, identity)

    (output / 'preview.json').write_text(json.dumps({
        'trajectory': [[0, 0, 0], [1, 0, 0]], 'sparse': [[0, 0, 1]],
    }), encoding='utf-8')
    (output / 'source' / 'fixture.py').write_text('tampered source\n', encoding='utf-8')
    assert not module.reusable_export(report_path, identity)

    (output / 'source' / 'fixture.py').write_text(
        (module.REPO / 'src' / 'fixture.py').read_text(encoding='utf-8'),
        encoding='utf-8')
    (output / 'evaluator.py').write_text('tampered evaluator\n', encoding='utf-8')
    assert not module.reusable_export(report_path, identity)


def test_completed_export_reuse_rejects_non_numeric_preview_geometry(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    output, report_path, identity = _write_completed_bundle(module, tmp_path, monkeypatch)
    (output / 'preview.json').write_text(json.dumps({
        'trajectory': [[0, 0, 0], [1, 0, 0]], 'sparse': [['x', 0, 1]],
    }), encoding='utf-8')
    assert not module.reusable_export(report_path, identity)


def test_completed_export_reuse_rejects_configuration_and_ground_truth_mismatch(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    output, report_path, identity = _write_completed_bundle(module, tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['configuration']['bundle_enabled'] = False
    report_path.write_text(json.dumps(report), encoding='utf-8')
    assert not module.reusable_export(report_path, identity)

    report['configuration']['bundle_enabled'] = True
    report['ground_truth_used_for_estimation'] = True
    report_path.write_text(json.dumps(report), encoding='utf-8')
    assert not module.reusable_export(report_path, identity)


@pytest.mark.parametrize('bad_tracking', [
    [{'frame': 1, 'state': 'tracking'}, {'frame': 0, 'state': 'tracking'}],
    [{'frame': 0, 'state': 'tracking'}],
    [{'frame': 0, 'state': 'lost'}, {'frame': 1, 'state': 'tracking'}],
])
def test_completed_export_reuse_requires_tracking_rows_to_match_report(
        tmp_path, monkeypatch, bad_tracking):
    module = load_runner(monkeypatch)
    output, report_path, identity = _write_completed_bundle(module, tmp_path, monkeypatch)
    run_path = output / 'run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run['tracking'] = bad_tracking
    run_path.write_text(json.dumps(run), encoding='utf-8')
    assert not module.reusable_export(report_path, identity)


def test_baseline_reuse_validates_poses_and_source_without_map_exports(
        tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    _, report_path, identity = _write_completed_bundle(
        module, tmp_path, monkeypatch, variant='baseline')
    assert module.reusable_export(report_path, identity)


def test_curated_checks_cover_performance_regressions_and_exist(monkeypatch):
    module = load_runner(monkeypatch)
    required = {
        'tests/test_integration_foundations.py',
        'tests/test_loop_performance_integration.py',
        'tests/test_tracking_performance_integration.py',
        'tests/test_bundle_performance_equivalence.py',
        'tests/test_descriptor_matching_cuda.py',
        'tests/test_stereo_regression_diagnostics.py',
        'tests/test_development_runner.py',
    }
    assert required.issubset(module.CURATED_TESTS)
    assert all((REPO / test).is_file() for test in module.CURATED_TESTS)


def test_child_timeout_preserves_cleanup_reserve(monkeypatch):
    module = load_runner(monkeypatch)
    now = [0.0]
    budget = module.Budget(100, clock=lambda: now[0])

    assert module.child_timeout(budget, 120) == 100 - module.CLEANUP_RESERVE_SECONDS
    assert module.child_timeout(budget, 3) == 3
    now[0] = 95
    assert module.child_timeout(budget, 120) == 0
