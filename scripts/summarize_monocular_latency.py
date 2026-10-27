"""Offline state/recovery strata of recorded SharedSlam.process wall latency.

This is process-only wall time, not input-to-output latency or stage/lock cost.
No estimator is imported, run, or modified; no speed-benefit verdict is made.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ACCEPTED = {'tracking', 'relocalized'}
STATES = ACCEPTED | {'initializing', 'lost'}


def statistics(values):
    if not values:
        return {'count': 0, 'median_seconds': None, 'p95_seconds': None, 'maximum_seconds': None}
    return {'count': len(values), 'median_seconds': float(np.percentile(values, 50, method='linear')),
            'p95_seconds': float(np.percentile(values, 95, method='linear')),
            'maximum_seconds': max(values)}


def frame_records(rows, label):
    if not isinstance(rows, list):
        raise ValueError(f'{label} records must be a list')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get('frame')) is not int or row['frame'] < 0:
            raise ValueError(f'{label} frame IDs must be nonnegative integers')
        if row['frame'] in result:
            raise ValueError(f'Duplicate {label} frame ID')
        result[row['frame']] = row
    return result


def summarize(run, timings):
    if not isinstance(run, dict) or not isinstance(timings, dict):
        raise ValueError('Run and timing inputs must be objects')
    tracking = frame_records(run.get('tracking'), 'tracking')
    timing_rows = frame_records(timings.get('records'), 'timing')
    states, seconds = {}, {}
    for frame, row in tracking.items():
        state = row.get('state')
        if not isinstance(state, str) or state not in STATES:
            raise ValueError('Unknown tracking state')
        states[frame] = state
    for frame, row in timing_rows.items():
        value = row.get('process_wall_seconds')
        try:
            finite = math.isfinite(value)
        except (TypeError, OverflowError):
            finite = False
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not finite or value < 0:
            raise ValueError('Process latency must be finite, nonnegative, and numeric')
        seconds[frame] = value
    matched = sorted(states.keys() & seconds.keys())
    by_state = {state: [] for state in sorted(STATES)}
    by_recovery = {'recovery': [], 'nonrecovery': [], 'unassessable_missing_predecessor': []}
    classifications = []
    for frame in matched:
        by_state[states[frame]].append(seconds[frame])
        # The previous INPUT frame is required, not the previous recorded timing.
        if frame > 0 and frame - 1 not in states:
            group = 'unassessable_missing_predecessor'
        elif frame > 0 and states[frame - 1] == 'lost' and states[frame] in ACCEPTED:
            group = 'recovery'
        else:
            group = 'nonrecovery'
        by_recovery[group].append(seconds[frame])
        classifications.append({'frame': frame, 'state': states[frame], 'recovery_class': group,
                                'process_wall_seconds': seconds[frame]})
    return {'all_process': statistics(list(seconds.values())),
            'matched_process': statistics([seconds[i] for i in matched]),
            'by_state': {key: statistics(values) for key, values in by_state.items()},
            'by_recovery': {key: statistics(values) for key, values in by_recovery.items()},
            'matched_frames': matched, 'classified_rows': classifications,
            'tracking_without_timing': sorted(states.keys() - seconds.keys()),
            'timing_without_tracking': sorted(seconds.keys() - states.keys()),
            'exact_frame_join': states.keys() == seconds.keys(),
            'recorded_definition': timings.get('definition'),
            'limitations': ['Exact join is not evidence of complete dataset processing',
                           'State/recovery strata include matched records only; all_process includes unmatched timing rows',
                           'Recovery means lost at input frame i-1 followed by tracking/relocalized at i',
                           'Process wall time includes synchronous waits/worker contention; excludes loading/export/shutdown',
                           'Process-time map size unavailable; map_revision/tracked_landmarks are not proxies',
                           'Stage and lock timings unavailable; no speed-benefit verdict']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if len({folder.resolve() for folder in args.runs}) != len(args.runs):
        parser.error('Duplicate run directories are not permitted')
    results, evidence = [], []
    for folder in args.runs:
        paths = [folder / 'run.json', folder / 'frame-latencies.json']
        raw = [path.read_bytes() for path in paths]
        run, timing = [json.loads(value) for value in raw]
        result = summarize(run, timing)
        result['folder'] = str(folder)
        results.append(result)
        evidence.extend({'path': str(path), 'sha256': hashlib.sha256(value).hexdigest()}
                        for path, value in zip(paths, raw))
    payload = json.dumps({'schema': 'monocular_latency_summary_v1',
                          'percentile_method': 'numpy linear (index=(n-1)*q)',
                          'runs': results, 'inputs': evidence}, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
