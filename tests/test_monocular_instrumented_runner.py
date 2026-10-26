"""Harmless real subprocess smoke: manufactured evaluator, no SLAM imports."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('failing', [False, True])
def test_portable_instrumentation_preserves_frame_boundary_and_failure_export(tmp_path, failing):
    evaluator = tmp_path / 'fake_evaluator.py'
    evaluator.write_text('''import sys
class SharedSlam:
    def process(self, index):
        if index == 1 and '--fail' in sys.argv:
            raise RuntimeError('manufactured frame failure')
        return index
def main():
    assert '--fixture-forwarded' in sys.argv
    slam = SharedSlam()
    for index in (0, 1):
        slam.process(index)
''')
    output = tmp_path / 'timings/frame-latencies.json'
    harness = Path(__file__).resolve().parents[1] / 'scripts/run_monocular_instrumented.py'
    result = subprocess.run([
        sys.executable, str(harness), '--entry-point', str(evaluator),
        '--frame-timings', str(output), '--', '--fixture-forwarded',
        *(['--fail'] if failing else []),
    ], capture_output=True, text=True, timeout=5)
    assert (result.returncode != 0) is failing
    if failing:
        assert 'manufactured frame failure' in result.stderr
    timings = json.loads(output.read_text())
    assert [row['frame'] for row in timings['records']] == [0, 1]
    assert all(row['process_wall_seconds'] >= 0 for row in timings['records'])
    assert 'excludes loading/export/shutdown' in timings['definition']
