"""Refresh saved raw/graph metrics with the devkit-compatible evaluator; keep run metadata."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti import load_poses_txt
from metrics import evaluate_trajectory
from json_output import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    source_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/metrics.py').read_bytes()).hexdigest()
    updated = 0
    for kind in ('stereo', 'graph'):
        for path in sorted((args.output_root / kind).glob('[0-9][0-9].json')):
            report = json.loads(path.read_text())
            if report.get('status', 'complete') != 'complete':
                continue
            gt = load_poses_txt(args.output_root / 'reference/poses' / f'{path.stem}.txt')
            poses = load_poses_txt(path.with_suffix('.txt'))
            if kind == 'stereo':
                report.update(evaluate_trajectory(gt, poses, 'se3'))
            else:
                raw = load_poses_txt(args.output_root / 'stereo' / f'{path.stem}.txt')
                report['before'] = evaluate_trajectory(gt, raw, 'se3')
                report['after'] = evaluate_trajectory(gt, poses, 'se3')
            report['metrics_source_sha256'] = source_hash
            report['segment_distance_precision'] = 'float32, matching official KITTI devkit'
            write_json(path, report)
            updated += 1
    print(f'Refreshed {updated} complete raw/graph metric reports')


if __name__ == '__main__':
    main()
