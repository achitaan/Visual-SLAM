import importlib.util
import subprocess
import sys
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[1]
@pytest.mark.parametrize("script,args,required", [
    ("src/main.py", (), "--slam --stereo"),
    ("src/main.py", ("--slam",), "--slam --stereo"),
    ("scripts/evaluate_shared_slam.py", ("--output", "unused"), "--stereo"),
])
def test_regularizer_rejects_nonstereo_modes(script,args,required):
    result=subprocess.run([sys.executable,str(REPO/script),*args,"--stereo-motion-regularizer"],cwd=REPO,capture_output=True,text=True,timeout=30)
    assert result.returncode == 2
    assert "--stereo-motion-regularizer requires "+required in result.stderr

def test_regularizer_configuration_is_explicit_and_baseline_isolated(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"scripts"))
    spec=importlib.util.spec_from_file_location("regularizer_dev_runner",REPO/"scripts/run_development_tests.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.current_mapping_configuration("bundle","supported")["stereo_motion_regularizer"] is False
    assert module.current_mapping_configuration("bundle","supported",stereo_motion_regularizer=True)["stereo_motion_regularizer"] is True
    assert "stereo_motion_regularizer" not in module.current_mapping_configuration("baseline","supported",stereo_motion_regularizer=True)
