"""Budget exhaustion cannot be represented as successful benchmark coverage."""
import importlib.util
from pathlib import Path
import numpy as np
import cv2 as cv
import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_includes_reserve_and_elapsed_work():
    tick = [10.0]
    budget = load('test_budget').Budget(60, clock=lambda: tick[0])
    assert budget.permits(45, reserve=10)
    tick[0] += 6
    assert not budget.permits(45, reserve=10)
    tick[0] += 100
    assert budget.remaining == 0
    with pytest.raises(ValueError):
        load('test_budget').Budget(0)


def test_preflight_checks_both_streams_and_content_identity(tmp_path):
    module = load('data_preflight')
    root = tmp_path / 'sequences/04'
    root.mkdir(parents=True)
    (root/'calib.txt').write_text('test calibration')
    for cam in [0,1]:
        folder=root/f'image_{cam}';folder.mkdir()
        for i in range(2):cv.imwrite(str(folder/f'{i:06d}.png'),np.full((20,30),i,np.uint8))
    first=module.preflight(tmp_path,'04',2)
    cv.imwrite(str(root/'image_1/000001.png'),np.full((20,30),42,np.uint8))
    assert module.preflight(tmp_path,'04',2)['sha256'] != first['sha256']
    (root/'image_1/000001.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Unreadable'):
        module.preflight(tmp_path,'04',2)


def test_reuse_rejects_wrong_revision_input_and_interruption(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=load('run_development_tests')
    identity={'revision':'abc','input':'images','frames':80,'variant':'bundle'}
    report={'development_identity':identity.copy(),'frames':80,'status':'completed'}
    assert module.reusable(report,identity)
    assert not module.reusable({**report,'frames':79},identity)
    assert not module.reusable({**report,'status':'interrupted_time_budget'},identity)
    for field in ['revision','input','variant']:
        assert not module.reusable(report,{**identity,field:'different'})


def test_supervisor_enforces_deadline_on_owned_worker(tmp_path,monkeypatch):
    import sys,os
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    result=load('run_development_tests').run_owned([sys.executable,'-c','import time; time.sleep(20)'],tmp_path/'worker.log',.2,os.environ.copy())
    assert result['timed_out'] and result['exit_code']!=0
    assert result['elapsed_s']<5


def test_resume_keeps_history_and_rejects_incompatible_source(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=load('run_shared_benchmark')
    initial={'source_sha256':'abc','coverage':'full','input_source':'local_images','requested_sequences':['04'],'requested_modes':['stereo'],'runs':[]}
    p=tmp_path/'batch.json';saved={**initial,'runs':[{'sequence':'04','status':'complete'}]};p.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='preserved'):module.resume_manifest(p,initial,False)
    resumed=module.resume_manifest(p,initial,True)
    assert resumed['runs']==saved['runs']
    with pytest.raises(ValueError,match='mismatched'):module.resume_manifest(p,{**initial,'source_sha256':'new'},True)
    assert json.loads(p.read_text())==saved
