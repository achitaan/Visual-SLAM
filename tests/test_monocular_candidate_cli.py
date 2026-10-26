import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from shared_slam import MappingConfig, SharedSlam, StereoCamera

OPTIONS=['monocular_refined_consensus','monocular_recovery_fallback','monocular_covisible_triangulation','monocular_scale_safe_triangulation']


def evaluator(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('monocular_candidate_evaluator',root/'scripts/evaluate_shared_slam.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name',OPTIONS)
def test_cli_flags_reach_mapping_config(name,tmp_path,monkeypatch):
    module=evaluator(monkeypatch)
    (tmp_path/'rgb.txt').write_text('1.0 rgb/one.png\n')
    caught=[]
    def capture(*args,**kwargs):
        caught.append(kwargs['config']);raise RuntimeError('captured config before processing')
    monkeypatch.setattr(module,'SharedSlam',capture)
    monkeypatch.setattr(sys,'argv',['evaluate_shared_slam.py','--dataset','tum','--data-root',str(tmp_path),
                                  '--output',str(tmp_path/'out'),'--'+name.replace('_','-')])
    with pytest.raises(RuntimeError,match='captured config'):
        module.main()
    assert getattr(caught[0],name) is True
    assert sum(bool(getattr(caught[0],option)) for option in OPTIONS)==1
    assert json.loads(json.dumps(caught[0].__dict__))[name] is True


@pytest.mark.parametrize('name',OPTIONS)
def test_cli_stereo_refuses_monocular_flags(name,tmp_path,monkeypatch,capsys):
    module=evaluator(monkeypatch)
    monkeypatch.setattr(sys,'argv',['evaluate_shared_slam.py','--stereo','--output',str(tmp_path/'out'),
                                  '--'+name.replace('_','-')])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code==2
    assert 'Monocular research options require monocular input' in capsys.readouterr().err


@pytest.mark.parametrize('name',OPTIONS)
def test_library_stereo_refuses_monocular_flags(name):
    stereo=StereoCamera(None,np.eye(4),.54)
    with pytest.raises(ValueError,match='Monocular research options require monocular input'):
        SharedSlam(np.eye(3),stereo=stereo,config=MappingConfig(**{name:True}))


def test_defaults_off_and_types_are_strict():
    config=MappingConfig()
    assert all(getattr(config,name) is False for name in OPTIONS)
    for name in OPTIONS:
        with pytest.raises(ValueError,match='must be boolean'):
            MappingConfig(**{name:1})
