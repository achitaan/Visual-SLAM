"""Compare saved stereo diagnostic trajectories without rerunning estimation."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti import load_poses_txt
from metrics import umeyama_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=run-directory')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth = np.array([p[:3, 3] for p in load_poses_txt(args.reference)])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = ['#44536a', '#2a8b76', '#ba6744', '#6474ba']
    labels, errors = [], []
    identity = None
    coverages = set()
    for number, item in enumerate(args.run):
        label, directory = item.split('=', 1)
        directory = Path(directory)
        report = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
        current = (report['sequence'], report['frames'], report['stereo'])
        if not current[2] or (identity is not None and current != identity):
            parser.error('Compare stereo runs with matching sequence and frame coverage')
        if report['metrics']['ate_alignment'] != 'se3' or report['metrics']['alignment_scale'] != 1.:
            parser.error('Stereo comparison requires SE(3) evaluation without scale fitting')
        identity = current
        coverages.add(report.get('coverage', 'partial'))
        poses = load_poses_txt(directory / 'poses.txt')
        if len(poses) != report['frames'] or len(truth) < len(poses) or not np.isfinite(np.asarray(poses)).all():
            parser.error('Trajectory exports must be finite and match reported/reference coverage')
        estimated = np.array([p[:3, 3] for p in poses])
        reference = truth[:len(estimated)]
        rotation, _, translation = umeyama_alignment(estimated, reference, with_scale=False)
        aligned = estimated @ rotation.T + translation
        color = colors[number % len(colors)]
        axes[0, 0].plot(aligned[:, 0], aligned[:, 2], color=color, label=label)
        axes[0, 1].plot(np.linalg.norm(aligned - reference, axis=1), color=color, label=label)
        labels.append(label)
        errors.append(report['metrics']['ate_rmse_m'])
        preview = directory / 'preview.json'
        if preview.exists():
            sparse = np.asarray(json.loads(preview.read_text())['sparse']).reshape(-1, 3)
            axes[1, 1].clear()
            axes[1, 1].scatter(sparse[:, 0], sparse[:, 2], s=1, alpha=.4, color=color)
            axes[1, 1].plot(estimated[:, 0], estimated[:, 2], color='#22343d')
            axes[1, 1].set(title=f'{label}: sparse preview in map coordinates', xlabel='X (m)', ylabel='Z (m)')
    axes[0, 0].plot(reference[:, 0], reference[:, 2], '--', color='black', label='Reference')
    axes[0, 0].set(title=f'KITTI {report["sequence"]}: {len(estimated)} frames, SE(3) alignment', xlabel='X (m)', ylabel='Z (m)')
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Position error after alignment', xlabel='Frame', ylabel='Error (m)')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].barh(labels, errors, color=colors[:len(labels)])
    scope = 'full-sequence diagnostic' if coverages == {'full'} else 'prefix diagnostic'
    axes[1, 0].set(title=f'ATE RMSE: {scope}, not a release benchmark', xlabel='ATE (m)')
    for axis in axes.ravel():
        axis.grid(alpha=.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    main()
