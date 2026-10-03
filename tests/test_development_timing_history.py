import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'scripts'))
    path = REPO / 'scripts' / 'run_development_tests.py'
    spec = importlib.util.spec_from_file_location('development_timing_history_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_case():
    return {
        'revision': 'current-source-fingerprint',
        'sequence': '01',
        'frames': 350,
        'input': 'ordered-image-fingerprint',
        'reference': 'reference-fingerprint',
        'variant': 'bundle',
        'cached': False,
        'stereo_depth_policy': 'supported',
        'performance': {
            'matching_backend': 'cpu', 'retrieval': 'current',
            'cpu_optimizations': True, 'opencv_threads': 1,
        },
        'runtime_identity': {
            'version': 1,
            'dependencies_sha256': 'same-runtime-dependencies',
            'dependencies': {'packages': {'numpy': '2.5.3'}},
        },
    }


def expected_configuration():
    return {
        'features': 1500, 'min_inliers': 15, 'keyframe_interval': 10,
        'max_landmarks': 2000, 'bundle_window': 5, 'bundle_enabled': True,
        'loop_mode': 'off', 'retrieval_candidates': 8,
        'stereo_feature_contrast_threshold': 0.02,
        'stereo_depth_policy': 'supported',
    }


def evaluation_report():
    identity = expected_case()
    identity['revision'] = 'older-source-revision'
    return {
        'sequence': '01', 'frames': 350, 'stereo': True, 'coverage': 'partial',
        'status': 'completed_with_tracking_loss', 'total_wall_seconds': 140.0,
        'elapsed_s': 136.0, 'configuration': expected_configuration(),
        'feature_cache': {'enabled': False}, 'development_identity': identity,
        'performance_configuration': {
            'matching_backend': 'cpu', 'retrieval': 'current',
            'cpu_optimizations': True, 'profile': False,
        },
        'matching_backend': {'requested': 'cpu'}, 'opencv_threads': 1,
        # This information must never flow into a new result or accuracy decision.
        'metrics': {'ate_rmse_m': 0.0},
    }


def test_history_is_cost_only_and_accepts_cross_revision_exact_mode(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(evaluation_report()), encoding='utf-8')

    estimate = module.estimate_case_runtime(
        350, [], [path], expected_case(), 'partial', expected_configuration(),
        'current-source-fingerprint', fallback_rate=4.0)

    assert estimate['estimated_seconds'] == pytest.approx(175.0)
    accepted = estimate['timing_history']['accepted']
    assert len(accepted) == 1
    assert accepted[0]['source_revision'] == 'older-source-revision'
    assert accepted[0]['source_revision_matches'] is False
    assert accepted[0]['use'] == 'cost_estimate_only'
    assert estimate['timing_history']['usage'] == 'cost_estimate_only'
    assert 'metrics' not in accepted[0]


@pytest.mark.parametrize('change', [
    lambda r: r.update(sequence='04'),
    lambda r: r.update(frames=349),
    lambda r: r.update(stereo=False),
    lambda r: r.update(coverage='full'),
    lambda r: r.update(status='interrupted_time_budget'),
    lambda r: r['development_identity'].update(variant='live'),
    lambda r: r['development_identity'].update(cached=True),
    lambda r: r['development_identity']['performance'].update(matching_backend='cuda'),
    lambda r: r['development_identity'].update(stereo_depth_policy='verified_fallback'),
    lambda r: r['development_identity'].update(input='different-image-order'),
    lambda r: r['development_identity'].update(reference='different-reference'),
    lambda r: r['development_identity'].pop('input'),
    lambda r: r['development_identity'].pop('reference'),
    lambda r: r['configuration'].update(max_landmarks=1999),
    lambda r: r['feature_cache'].update(enabled=True),
    lambda r: r['performance_configuration'].update(cpu_optimizations=False),
    lambda r: r['performance_configuration'].update(profile=True),
    lambda r: r['matching_backend'].update(requested='cuda'),
    lambda r: r['development_identity'].pop('runtime_identity'),
    lambda r: r['development_identity']['runtime_identity']['dependencies']['packages'].update(
        numpy='different-runtime-version'),
])
def test_incompatible_or_incomplete_history_uses_conservative_default(tmp_path, monkeypatch, change):
    module = load_runner(monkeypatch)
    report = evaluation_report()
    change(report)
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(report), encoding='utf-8')

    estimate = module.estimate_case_runtime(
        350, [], [path], expected_case(), 'partial', expected_configuration(),
        'current-source-fingerprint', fallback_rate=4.0)

    assert estimate['estimated_seconds'] == pytest.approx(1750.0)
    assert estimate['selected_rate']['source'] == 'conservative_default'
    assert not estimate['timing_history']['accepted']
    assert len(estimate['timing_history']['rejected']) == 1


def test_invalid_path_and_missing_depth_metadata_are_rejected(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    missing = tmp_path / 'absent.json'
    no_depth = evaluation_report()
    no_depth['configuration'].pop('stereo_depth_policy')
    path = tmp_path / 'no-depth.json'
    path.write_text(json.dumps(no_depth), encoding='utf-8')

    estimate = module.estimate_case_runtime(
        350, [], [missing, path], expected_case(), 'partial', expected_configuration(),
        'current-source-fingerprint', fallback_rate=4.0)

    assert estimate['estimated_seconds'] == pytest.approx(1750.0)
    assert [row['reason'] for row in estimate['timing_history']['rejected']] == [
        'not a readable evaluation file', 'mismatched evaluator configuration']


def test_slowest_compatible_current_or_historical_rate_sets_budget(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    path = tmp_path / 'evaluation.json'
    report = evaluation_report()
    report['total_wall_seconds'] = 210.0
    path.write_text(json.dumps(report), encoding='utf-8')

    estimate = module.estimate_case_runtime(
        350, [0.5], [path], expected_case(), 'partial', expected_configuration(),
        'current-source-fingerprint', fallback_rate=4.0)

    assert estimate['rate_s_per_frame'] == pytest.approx(0.6)
    assert estimate['estimated_seconds'] == pytest.approx(262.5)
    assert estimate['selected_rate']['source'] == 'historical_evaluation'
