"""Experimental refinement cannot silently run in an unsupported input mode."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', [[], ['--slam'], ['--stereo'],
                                  ['--slam', '--stereo']])
def test_main_rejects_refinement_before_opening_input(mode, monkeypatch, capsys):
    entry = load_entrypoint('two_view_cli_main', 'src/main.py')
    monkeypatch.setattr(sys, 'argv', ['main.py', *mode, '--stereo-two-view-refinement'])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert '--stereo-two-view-refinement' in error
    assert 'requires' in error
    assert '--stereo-pose-arbitration' in error


@pytest.mark.parametrize('mode', [[], ['--stereo']])
def test_evaluator_rejects_refinement_before_opening_input(mode, monkeypatch, capsys, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    entry = load_entrypoint('two_view_cli_eval', 'scripts/evaluate_shared_slam.py')
    monkeypatch.setattr(sys, 'argv', ['evaluate_shared_slam.py', *mode,
                      '--stereo-two-view-refinement', '--output', str(tmp_path / 'run')])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert '--stereo-two-view-refinement' in error
    assert 'requires' in error
    assert '--stereo-pose-arbitration' in error


def test_valid_refinement_mode_reaches_normal_input_validation(monkeypatch, capsys):
    entry = load_entrypoint('two_view_cli_valid_main', 'src/main.py')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--slam', '--stereo',
                      '--stereo-pose-arbitration', '--stereo-two-view-refinement'])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert '--stereo-two-view-refinement requires' not in error
    assert '--stereo requires --data-root' in error


@pytest.mark.parametrize('entrypoint', ['main', 'evaluator'])
def test_dense_solver_requires_refinement_before_input_open(entrypoint, monkeypatch, capsys, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    relative = 'src/main.py' if entrypoint == 'main' else 'scripts/evaluate_shared_slam.py'
    entry = load_entrypoint('two_view_dense_' + entrypoint, relative)
    args = ['entry', '--stereo', '--stereo-pose-arbitration',
            '--stereo-two-view-solver', 'dense_exact']
    if entrypoint == 'main':
        args += ['--slam']
    else:
        args += ['--output', str(tmp_path / 'run')]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert '--stereo-two-view-solver dense_exact requires --stereo-two-view-refinement' in error


def test_solver_mapping_config_default_and_scope():
    from shared_slam import MappingConfig
    assert MappingConfig().stereo_two_view_solver == 'sparse_lsmr'
    with pytest.raises(ValueError, match='requires'):
        MappingConfig(stereo_two_view_solver='dense_exact')
    with pytest.raises(ValueError):
        MappingConfig(stereo_two_view_solver='automatic')
    config = MappingConfig(stereo_pose_arbitration=True, stereo_two_view_refinement=True,
                           stereo_two_view_solver='dense_exact')
    assert config.stereo_two_view_solver == 'dense_exact'
