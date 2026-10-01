import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from kitti import load_poses_txt


@pytest.mark.parametrize('telemetry', [False, True])
def test_sample_cli_exports_kitti_and_exits(tmp_path, telemetry):
    output = tmp_path / 'poses.txt'
    command = [sys.executable, 'src/main.py', '--max-frames', '4', '--frame-delay-ms', '0', '--output', str(output)]
    if not telemetry:
        command.append('--no-telemetry')
    else:
        command.extend(['--telemetry-port', '0'])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    poses = load_poses_txt(output)
    assert len(poses) == 4
    assert np.isfinite(poses).all()
