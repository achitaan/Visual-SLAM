"""Recover metrics after tracking finished but auxiliary reporting failed.

Tracking counts must be backed by an exact completion line in a retained batch log.
"""
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
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--tracked-pairs', type=int, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    if args.sequence not in [f'{i:02d}' for i in range(11)]:
        parser.error('Sequence must be 00–10')
    root, sequence = args.output_root, args.sequence
    trajectory_path = root / 'stereo' / f'{sequence}.txt'
    raw = load_poses_txt(trajectory_path)
    gt = load_poses_txt(root / 'reference/poses' / f'{sequence}.txt')
    if len(raw) != len(gt) or not 0 <= args.tracked_pairs < len(raw):
        raise ValueError('Invalid/incomplete raw trajectory or tracking count')
    completion = f'{sequence}: stereo {len(raw)}/{len(raw)}, tracked {args.tracked_pairs}/{len(raw) - 1},'
    proof = [(path, line) for path in root.glob('*.log') for line in path.read_text().splitlines() if line.startswith(completion)]
    if not proof:
        raise ValueError('No retained log proves the supplied complete tracking count')
    report = evaluate_trajectory(gt, raw, 'se3')
    report.update(sequence=sequence, source='stereo_vo', total_frames=len(gt), status='complete',
                  tracked_pairs=args.tracked_pairs, lost_pairs=len(raw) - 1 - args.tracked_pairs,
                  translation_scale='metric', processing_fps=None, elapsed_s=None,
                  recovered_reporting_from_saved_raw=True,
                  raw_trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                  tracking_count_source=dict(log=str(proof[0][0].resolve()), completion_line=proof[0][1]),
                  timing_note='Auxiliary reporting failed after complete tracking; performance timing is unavailable.')
    write_json(root / 'stereo' / f'{sequence}.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'segments'}, indent=2))


if __name__ == '__main__':
    main()
