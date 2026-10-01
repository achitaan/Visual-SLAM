"""Compare saved metric drift reports with a compiled official KITTI calcSequenceErrors wrapper."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import io
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=Path('results/benchmark-batch/devkit/parity.exe'))
    parser.add_argument('--poses-root', type=Path, default=Path('results/benchmark-batch/reference/poses'))
    parser.add_argument('--reports-root', type=Path, default=Path('results/benchmark-batch/stereo'))
    parser.add_argument('--output', type=Path, default=Path('results/benchmark-batch/devkit-parity.json'))
    args = parser.parse_args()
    results = []
    for report_path in sorted(args.reports_root.glob('[0-9][0-9].json')):
        report = json.loads(report_path.read_text())
        if report.get('status', 'complete') != 'complete':
            continue
        metrics = report.get('after', report)
        process = subprocess.run([str(args.executable.resolve()), str((args.poses_root / f'{report_path.stem}.txt').resolve()), str(report_path.with_suffix('.txt').resolve())], capture_output=True, text=True, check=True)
        reference = np.loadtxt(io.StringIO(process.stdout), ndmin=2)
        observed = np.array([[s['first_frame'], s['length_m'], s['translation_percent'], s['rotation_deg_per_m']] for s in metrics['segments']])
        same_keys = reference.shape == observed.shape and np.array_equal(reference[:, :2], observed[:, :2])
        differences = np.max(np.abs(reference[:, 2:] - observed[:, 2:]), axis=0) if same_keys else [None, None]
        results.append(dict(sequence=report_path.stem, matching_segment_keys=bool(same_keys),
                            reference_segments=len(reference), observed_segments=len(observed),
                            max_translation_difference_percent=differences[0], max_rotation_difference_deg_per_m=differences[1],
                            within_tolerance=bool(same_keys and differences[0] < 1e-4 and differences[1] < 1e-4)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(reference='Unchanged calcSequenceErrors from official KITTI devkit_odometry.zip',
                                        translation_tolerance_percent=1e-4, rotation_tolerance_deg_per_m=1e-4, results=results), indent=2, allow_nan=False))
    print(json.dumps(results, indent=2))
    return 0 if all(result['within_tolerance'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
