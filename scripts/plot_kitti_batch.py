"""Export every measured KITTI trajectory/error graph and a complete benchmark report."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from kitti import load_poses_txt

COLORS = dict(gt='#22343d', raw='#4d73d7', graph='#d78b25')


def draw_pair(axes, sequence, raw, graph, truth, report):
    trajectory, errors = axes
    xyz = {key: np.array([pose[:3, 3] for pose in poses]) for key, poses in [('gt', truth), ('raw', raw)]}
    if graph is not None:
        xyz['graph'] = np.array([pose[:3, 3] for pose in graph])
    for key, label in [('gt', 'Ground truth'), ('raw', 'Raw stereo VO'), ('graph', 'Pose graph')]:
        if key in xyz:
            trajectory.plot(xyz[key][:, 0], xyz[key][:, 2], color=COLORS[key], label=label, linewidth=1.4, linestyle='--' if key == 'graph' else '-')
    trajectory.set(xlabel='X (m)', ylabel='Z (m)', title=f'KITTI {sequence} · {len(raw):,} frames · {report.get("verified_loop_count", 0)} verified loops')
    trajectory.set_aspect('equal', adjustable='datalim')
    trajectory.legend(fontsize=8, loc='best')
    for key, label in [('raw', 'Raw stereo VO'), ('graph', 'Pose graph')]:
        if key in xyz:
            errors.plot(np.linalg.norm(xyz[key] - xyz['gt'], axis=1), color=COLORS[key], linewidth=1.1, label=label, linestyle='--' if key == 'graph' else '-')
    errors.set(xlabel='Frame', ylabel='Position error (m)', title='Unaligned position error · original metric coordinates')
    errors.legend(fontsize=8, loc='best')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    root = args.output_root.resolve()
    plots = root / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    records, table, comparisons = [], [], []
    for sequence in [f'{i:02d}' for i in range(11)]:
        raw_report_path = root / 'stereo' / f'{sequence}.json'
        graph_report_path = root / 'graph' / f'{sequence}.json'
        if not raw_report_path.exists():
            table.append(f'| {sequence} | — | Pending/failed | — | — | — | — | — |')
            continue
        raw_report = json.loads(raw_report_path.read_text())
        graph_report = json.loads(graph_report_path.read_text()) if graph_report_path.exists() else {}
        raw = load_poses_txt(raw_report_path.with_suffix('.txt'))
        truth = load_poses_txt(root / 'reference/poses' / f'{sequence}.txt')[:len(raw)]
        graph = load_poses_txt(graph_report_path.with_suffix('.txt')) if graph_report else None
        if graph is not None and len(graph) != len(raw):
            raise ValueError('Corrected trajectory coverage mismatch')
        records.append((sequence, raw, graph, truth, graph_report))
        after = graph_report.get('after', {})
        comparisons.append((sequence, raw_report, after))
        table.append(f"| {sequence} | {len(raw):,} | {number(raw_report['ate_rmse_m'])} → {number(after.get('ate_rmse_m'))} | {number(raw_report['translation_percent'])} → {number(after.get('translation_percent'))} | {number(raw_report['rotation_deg_per_m'], 5)} → {number(after.get('rotation_deg_per_m'), 5)} | {graph_report.get('verified_loop_count', '—')} | {raw_report['lost_pairs']} | {raw_report.get('status', 'complete')} |")
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        draw_pair(axes, sequence, raw, graph, truth, graph_report)
        figure.savefig(plots / f'{sequence}.png', dpi=180)
        plt.close(figure)
    galleries = []
    for offset in range(0, len(records), 4):
        group = records[offset:offset + 4]
        figure, axes = plt.subplots(len(group), 2, figsize=(12, 4.6 * len(group)), layout='constrained', squeeze=False)
        for index, record in enumerate(group):
            draw_pair(axes[index], *record)
        target = plots / f'graphs-{group[0][0]}-{group[-1][0]}.png'
        figure.savefig(target, dpi=160)
        plt.close(figure)
        galleries.append(target)
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    overview = plots / 'benchmark-overview.png'
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    locations = np.arange(len(comparisons))
    for axis, (key, label) in zip(axes, [('ate_rmse_m', 'SE(3)-aligned ATE (m)'), ('translation_percent', 'Translation drift (%)'), ('rotation_deg_per_m', 'Rotation drift (degrees/m)')]):
        for offset, side, color in [(-.2, 1, COLORS['raw']), (.2, 2, COLORS['graph'])]:
            values = [row[side].get(key) for row in comparisons]
            axis.bar(locations + offset, [np.nan if value is None else value for value in values], width=.38, color=color, label='Raw stereo VO' if side == 1 else 'Pose graph')
        axis.set(xlabel='KITTI sequence', ylabel=label)
        axis.set_xticks(locations, [row[0] for row in comparisons])
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=8)
    figure.savefig(overview, dpi=180)
    plt.close(figure)
    lines = ['# KITTI 00–10 benchmark', '', 'Each arrow compares raw stereo VO with offline pose graph correction. ATE uses SE(3) alignment; drift and plots retain metric scale. Ground truth is used only for evaluation. Live map/tracking feedback remains unfinished.', '',
             f"Coverage: {summary.get('completed_stereo_sequences', len(records))}/11 full stereo runs, {summary.get('completed_graph_sequences', 'unknown')}/11 graph runs; {summary.get('full_stereo_frames', 'unknown'):,} frames; {summary.get('full_stereo_lost_pairs', 'unknown')} lost tracking pairs." if isinstance(summary.get('full_stereo_frames'), int) else 'Coverage is listed per sequence below.', '',
             f"Segment-weighted translation drift: {number(summary.get('raw_translation_percent'))}% → {number(summary.get('corrected_translation_percent'))}%. Segment-weighted rotation drift: {number(summary.get('raw_rotation_deg_per_m'), 5)} → {number(summary.get('corrected_rotation_deg_per_m'), 5)} degrees/m.", '',
             '| Sequence | Frames | ATE m, raw → graph | Translation %, raw → graph | Rotation °/m, raw → graph | Verified loops | Lost pairs | Coverage |',
             '|---|---:|---:|---:|---:|---:|---:|---:|', *table, '', f'![Benchmark overview]({overview.as_posix()})', '', 'All trajectory and position-error graphs:', '']
    for path in galleries:
        lines += [f'![{path.stem}]({path.as_posix()})', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Exported {len(records)} individual graph pairs, {len(galleries)} galleries, and {root / "REPORT.md"}')


if __name__ == '__main__':
    main()
