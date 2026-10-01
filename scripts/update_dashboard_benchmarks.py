"""Publish measured KITTI reports into the dashboard; no synthetic results."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti import load_poses_txt
from json_output import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
runs = []
reports = [repo / 'results/stereo-120/00.json', repo / 'results/stereo-full-04/04.json', repo / 'results/stereo-verified-04/04.json', repo / 'results/stereo-loop-07/07.json'] + sorted((repo / 'results/stereo-multi-300').glob('[0-9][0-9].json')) + sorted((repo / 'results/benchmark-batch/stereo').glob('[0-9][0-9].json'))
# Prefer coverage, then recency; a partial retry cannot replace a full benchmark.
reports_by_sequence = {}
for path in reports:
    if path.exists():
        old = reports_by_sequence.get(path.stem)
        rank = (json.loads(path.read_text())['frames'], path.stat().st_mtime)
        if old is None or rank > (json.loads(old.read_text())['frames'], old.stat().st_mtime):
            reports_by_sequence[path.stem] = path
for path in reports_by_sequence.values():
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    sequence = report['sequence']
    gt = load_poses_txt(args.data_root / 'poses' / f'{sequence}.txt')
    estimated = load_poses_txt(path.with_suffix('.txt'))
    count = report['frames']
    if len(estimated) != count or count > len(gt):
        raise ValueError(f'Pose/report coverage mismatch: {path}')
    step = max(1, count // 150)
    indices = sorted(set(range(0, count, step)) | {count - 1})
    runs.append({**{key: report.get(key) for key in ['sequence', 'frames', 'ate_rmse_m', 'raw_ate_rmse_m', 'translation_percent', 'rotation_deg_per_m', 'segment_count', 'lost_pairs', 'processing_fps']},
                 'total_frames': len(gt), 'complete': count == len(gt),
                 'updated_at': datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(),
                 'trajectory': {'estimated': [[float(estimated[i][0, 3]), float(estimated[i][2, 3])] for i in indices],
                                'ground_truth': [[float(gt[i][0, 3]), float(gt[i][2, 3])] for i in indices]}})
runs.sort(key=lambda run: run['sequence'])
experiments = []
graph_paths = sorted((repo / 'results/pose-graph').glob('[0-9][0-9].json')) + sorted((repo / 'results/benchmark-batch/graph').glob('[0-9][0-9].json'))
graph_by_sequence = {}
for path in graph_paths:
    if path.stem not in graph_by_sequence or path.stat().st_mtime > graph_by_sequence[path.stem].stat().st_mtime:
        graph_by_sequence[path.stem] = path
for path in graph_by_sequence.values():
    report = json.loads(path.read_text())
    sequence = report['sequence']
    raw = load_poses_txt(Path(report['raw_trajectory_path']) if 'raw_trajectory_path' in report else repo / f'results/stereo-loop-{sequence}/{sequence}.txt')
    corrected = load_poses_txt(path.with_suffix('.txt'))
    gt = load_poses_txt(args.data_root / 'poses' / f'{sequence}.txt')
    count = report['frames']
    if any(len(values) != count for values in (raw, corrected, gt)):
        raise ValueError(f'Graph experiment coverage mismatch: {path}')
    indices = sorted(set(range(0, count, max(1, count // 200))) | {count - 1})
    experiments.append({**{key: report[key] for key in ['sequence', 'frames', 'keyframe_count', 'verified_loop_count', 'candidate_pairs_tested', 'ground_truth_used_for_constraints', 'scope']},
                        'before': {k: v for k, v in report['before'].items() if k != 'segments'},
                        'after': {k: v for k, v in report['after'].items() if k != 'segments'},
                        'trajectory': {name: [[float(values[i][0, 3]), float(values[i][2, 3])] for i in indices] for name, values in [('raw', raw), ('corrected', corrected), ('ground_truth', gt)]}})
payload = {'dataset': 'KITTI odometry', 'estimator': 'Stereo VO · SIFT / SGBM / PnP', 'ate_alignment': 'SE(3)', 'drift_alignment': 'None', 'runs': runs, 'graph_experiments': experiments}
write_json(repo / 'dashboard/data/benchmarks.json', payload)
print(f'Published {len(runs)} measured benchmark reports')
