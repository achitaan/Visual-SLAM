"""Render a saved graph experiment without modifying or aligning its trajectories."""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti import load_poses_txt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sequence', default='07')
parser.add_argument('--poses-root', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
report = json.loads((repo / f'results/pose-graph/{args.sequence}.json').read_text())
raw, corrected, gt = [np.array([pose[:3, 3] for pose in load_poses_txt(path)]) for path in
                       (repo / f'results/stereo-loop-{args.sequence}/{args.sequence}.txt',
                        repo / f'results/pose-graph/{args.sequence}.txt', args.poses_root / f'{args.sequence}.txt')]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
fig.suptitle(f"KITTI {args.sequence} · {len(raw)} frames · {report['verified_loop_count']} verified image loops", fontsize=15)
for values, name, color in ((gt, 'Ground truth', '#11a58b'), (raw, 'Raw stereo VO', '#5375dd'), (corrected, 'Graph corrected', '#d39a4b')):
    axes[0].plot(values[:, 0], values[:, 2], color=color, label=name, lw=1.7)
axes[0].set(xlabel='X (m)', ylabel='Z (m)', title='Raw coordinates · equal axis scale')
axes[0].set_aspect('equal', adjustable='datalim')
axes[0].legend(frameon=False, fontsize=9)
for values, name, color in ((raw, 'Raw stereo VO', '#5375dd'), (corrected, 'Graph corrected', '#d39a4b')):
    axes[1].plot(np.linalg.norm(values - gt, axis=1), label=name, color=color, lw=1.7)
axes[1].set(xlabel='Frame', ylabel='Position error (m)', title='No trajectory alignment')
axes[1].legend(frameon=False, fontsize=9)
for axis in axes:
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
fig.savefig(repo / f'results/pose-graph/{args.sequence}-comparison.png', dpi=180)
print('Saved graph comparison figure')
