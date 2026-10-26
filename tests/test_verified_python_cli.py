"""The original application must expose the already-verified Python profile."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from shared_slam import MappingConfig

ROOT = Path(__file__).resolve().parents[1]
MONOCULAR_RESEARCH_FIELDS = (
    'monocular_refined_consensus',
    'monocular_recovery_fallback',
    'monocular_covisible_triangulation',
    'monocular_scale_safe_triangulation',
)


def _assert_confirmed_profile_with_disabled_research(actual, recorded):
    # Keep the historical benchmark artifact unchanged. The sole schema
    # extension is these four explicitly disabled options, not arbitrary extras.
    assert not set(recorded).intersection(MONOCULAR_RESEARCH_FIELDS)
    assert actual == {**recorded, **{name: False for name in MONOCULAR_RESEARCH_FIELDS}}
    # Equality alone would accidentally accept numeric zero instead of False.
    assert all(actual[name] is False for name in MONOCULAR_RESEARCH_FIELDS)


def test_application_config_matches_confirmed_original_python_case():
    spec = importlib.util.spec_from_file_location('original_verified_main', ROOT / 'src/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((ROOT / 'docs/benchmark/original-python-verified.json').read_text())['configuration']
    actual = asdict(module._mapping_config_from_args(SimpleNamespace(**expected)))
    _assert_confirmed_profile_with_disabled_research(actual, expected)


@pytest.mark.parametrize('mutation', [
    'unknown_extra', 'missing_legacy', 'changed_legacy', 'missing_research',
])
def test_original_profile_compatibility_rejects_unregistered_schema_or_value_changes(mutation):
    recorded = {'features': 1500}
    actual = {**recorded, **{name: False for name in MONOCULAR_RESEARCH_FIELDS}}
    if mutation == 'unknown_extra':
        actual['unregistered_research_option'] = False
    elif mutation == 'missing_legacy':
        del actual['features']
    elif mutation == 'changed_legacy':
        actual['features'] = 1501
    else:
        del actual[MONOCULAR_RESEARCH_FIELDS[0]]
    with pytest.raises(AssertionError):
        _assert_confirmed_profile_with_disabled_research(actual, recorded)


@pytest.mark.parametrize('name', MONOCULAR_RESEARCH_FIELDS)
@pytest.mark.parametrize('value', [True, 0])
def test_original_profile_compatibility_requires_each_research_value_to_be_boolean_false(name, value):
    recorded = {'features': 1500}
    actual = {**recorded, **{field: False for field in MONOCULAR_RESEARCH_FIELDS}}
    actual[name] = value
    with pytest.raises(AssertionError):
        _assert_confirmed_profile_with_disabled_research(actual, recorded)


@pytest.mark.parametrize('flag,message', [
    ('--stereo-descriptor-association-retry', 'requires --slam --stereo --stereo-pose-arbitration'),
    ('--stereo-verified-reference-retry', 'requires --slam --stereo'),
])
def test_stereo_profile_options_reject_incompatible_modes(flag, message):
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'src/main.py'), flag],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
