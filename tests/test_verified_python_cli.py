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


def test_application_config_matches_confirmed_original_python_case():
    spec = importlib.util.spec_from_file_location('original_verified_main', ROOT / 'src/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((ROOT / 'docs/benchmark/original-python-verified.json').read_text())['configuration']
    actual = asdict(module._mapping_config_from_args(SimpleNamespace(**expected)))
    assert actual == expected


@pytest.mark.parametrize('flag,message', [
    ('--stereo-descriptor-association-retry', 'requires --slam --stereo --stereo-pose-arbitration'),
    ('--stereo-verified-reference-retry', 'requires --slam --stereo'),
])
def test_stereo_profile_options_reject_incompatible_modes(flag, message):
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'src/main.py'), flag],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
