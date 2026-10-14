import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def run_cli(script, *arguments):
    return subprocess.run(
        [sys.executable, str(REPO / script), *map(str, arguments)],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize('arguments', [
    ('--stereo-pose-arbitration',),
    ('--slam', '--stereo-pose-arbitration'),
    ('--stereo', '--stereo-pose-arbitration'),
])
def test_main_arbitration_requires_slam_and_stereo(arguments):
    result = run_cli('src/main.py', *arguments)

    assert result.returncode == 2
    assert '--stereo-pose-arbitration requires --slam --stereo' in result.stderr


def test_main_accepts_arbitration_when_slam_and_stereo_are_selected():
    result = run_cli('src/main.py', '--slam', '--stereo', '--stereo-pose-arbitration')

    assert result.returncode == 2
    assert '--stereo-pose-arbitration requires --slam --stereo' not in result.stderr
    assert '--stereo requires --data-root' in result.stderr


def test_evaluator_arbitration_requires_stereo(tmp_path):
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo-pose-arbitration',
                     '--output', tmp_path / 'evaluation')

    assert result.returncode == 2
    assert '--stereo-pose-arbitration requires --stereo' in result.stderr


def test_evaluator_accepts_arbitration_when_stereo_is_selected(tmp_path):
    result = run_cli('scripts/evaluate_shared_slam.py', '--stereo',
                     '--stereo-pose-arbitration', '--output', tmp_path / 'evaluation')

    assert result.returncode == 2
    assert '--stereo-pose-arbitration requires --stereo' not in result.stderr
    assert '--data-root is required for local input' in result.stderr


def test_development_runtime_configuration_records_the_opt_in(monkeypatch):
    scripts = str(REPO / 'scripts')
    monkeypatch.syspath_prepend(scripts)
    path = REPO / 'scripts' / 'run_development_tests.py'
    spec = importlib.util.spec_from_file_location('pose_arbitration_dev_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    disabled = module.current_mapping_configuration('bundle', 'supported', False)
    enabled = module.current_mapping_configuration('bundle', 'supported', True)

    assert disabled['stereo_pose_arbitration'] is False
    assert enabled['stereo_pose_arbitration'] is True
