"""Retained-image constraints are part of the declared experiment identity."""
import json
import pytest
from test_development_timing_history import (
    load_runner, evaluation_report, expected_case, expected_configuration,
)


@pytest.mark.parametrize('requested', [False, True])
def test_retention_modes_cannot_share_timing_history(tmp_path, monkeypatch, requested):
    module = load_runner(monkeypatch)
    identity = expected_case()
    identity['stereo_retained_source_observations'] = requested
    report = evaluation_report()
    report['development_identity']['stereo_retained_source_observations'] = not requested
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    estimate = module.estimate_case_runtime(
        350, [], [path], identity, 'partial', expected_configuration(),
        'current-source-fingerprint', fallback_rate=4.0)
    assert not estimate['timing_history']['accepted']
    assert estimate['timing_history']['rejected'][0]['reason'] == (
        'mismatched stereo-retained-source-observations mode')


def test_preserved_baseline_rejects_retention(monkeypatch):
    module = load_runner(monkeypatch)
    with pytest.raises(ValueError, match='not available for baseline'):
        module.current_mapping_configuration(
            'baseline', 'verified_fallback', stereo_retained_source_observations=True)
