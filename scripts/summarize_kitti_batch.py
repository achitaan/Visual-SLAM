"""Summarize complete/failed/pending KITTI runs without dropping missing sequences."""
import argparse
import json
from pathlib import Path
from json_output import write_json


def weighted(reports, key):
    usable = [report for report in reports if report['segment_count'] and report[key] is not None]
    count = sum(report['segment_count'] for report in usable)
    return sum(report[key] * report['segment_count'] for report in usable) / count if count else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    rows, before, after = [], [], []
    for sequence in [f'{i:02d}' for i in range(11)]:
        row = dict(sequence=sequence, stereo_status='pending', graph_status='pending')
        status_path = args.output_root / f'{sequence}-status.json'
        if status_path.exists():
            row['run_status'] = json.loads(status_path.read_text())
        for kind in ('stereo', 'graph'):
            path = args.output_root / kind / f'{sequence}.json'
            if path.exists():
                report = json.loads(path.read_text())
                row[f'{kind}_status'] = report.get('status', 'complete')
                if kind == 'stereo':
                    row['frames'] = report['frames']
                    row['lost_pairs'] = report['lost_pairs']
                    row['raw'] = {key: value for key, value in report.items() if key != 'segments'}
                    if row['stereo_status'] == 'complete':
                        before.append(report)
                else:
                    row['verified_loops'] = report['verified_loop_count']
                    row['corrected'] = {key: value for key, value in report['after'].items() if key != 'segments'}
                    if row['graph_status'] == 'complete':
                        after.append(report['after'])
            elif row.get('run_status', {}).get('status') == 'failed' and row['run_status']['phase'] == kind:
                row[f'{kind}_status'] = 'failed'
        rows.append(row)
    summary = dict(expected_sequences=11, completed_stereo_sequences=len(before), completed_graph_sequences=len(after),
                   sequences=rows, full_stereo_frames=sum(report['frames'] for report in before),
                   full_stereo_lost_pairs=sum(report['lost_pairs'] for report in before),
                   aggregate_method='Segment-count weighted; complete sequences only. Raw/graph coverage may differ.',
                   raw_translation_percent=weighted(before, 'translation_percent'),
                   raw_rotation_deg_per_m=weighted(before, 'rotation_deg_per_m'),
                   corrected_translation_percent=weighted(after, 'translation_percent'),
                   corrected_rotation_deg_per_m=weighted(after, 'rotation_deg_per_m'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / 'summary.json'
    write_json(path, summary)
    print(f"Summary: {len(before)}/11 full stereo runs, {len(after)}/11 graph runs")


if __name__ == '__main__':
    main()
