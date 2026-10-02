import importlib.util
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
                 'run_development_tests.py', 'test_budget.py', 'data_preflight.py',
                 'benchmark_telemetry.py'):
        (repo / 'scripts' / name).write_text('# stable harness input\n', encoding='utf-8')
    test_file = repo / 'tests' / 'test_gate.py'
    test_file.write_text('def test_gate(): pass\n', encoding='utf-8')
    monkeypatch.setattr(module, 'REPO', repo)

    focused_revision = module.source_fingerprint()
    gate = repo / 'gate.json'
    gate.write_text('{"revision": "%s", "passed": true}' % focused_revision, encoding='utf-8')
    module.validate_release_gate('release', gate, focused_revision)

    test_file.write_text('def test_gate(): assert True\n', encoding='utf-8')
    changed_revision = module.source_fingerprint()
    assert changed_revision != focused_revision
    with pytest.raises(ValueError, match='exact revision'):
        module.validate_release_gate('release', gate, changed_revision)


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
