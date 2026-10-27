"""Offline supplements to saved full-fit scores; never runs or imports SLAM.

Inputs are a historical score_runs.py report and its run.json files, in report
order. This preserves original alignment/residuals; it is not an independent
validation of their provenance or a promotion/gate verdict.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ACCEPTED = {'tracking', 'relocalized'}
STATES = ACCEPTED | {'initializing', 'lost'}


def distribution(values):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Residuals must be a finite nonnegative vector') from error
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Residuals must be a finite nonnegative vector')
    if not len(values):
        return None
    # Scale first to avoid overflow when squaring otherwise finite residuals.
    scale = float(values.max())
    return {'count': len(values),
            'rmse_m': float(scale * np.sqrt(np.mean((values / scale)**2))) if scale else 0.,
            'median_m': float(np.percentile(values, 50, method='linear')),
            'p95_m': float(np.percentile(values, 95, method='linear')),
            'maximum_m': scale}


def recovery_stability(states):
    """Event excluded: inspect exactly event+1 through event+30 input frames."""
    events = []
    for frame in range(1, len(states)):
        if states[frame - 1] != 'lost' or states[frame] not in ACCEPTED:
            continue
        window = states[frame + 1:frame + 31]
        count = sum(state in ACCEPTED for state in window)
        censored = len(window) < 30
        events.append({'frame': frame, 'window_start_frame': frame + 1,
                       'observed_window_frames': len(window), 'accepted_frames': count,
                       'censored': censored,
                       'unstable_continuation': None if censored else count < 27})
    return events


def supplement(report, runs):
    summaries = report.get('runs') if isinstance(report, dict) else None
    if not isinstance(summaries, list) or len(summaries) != len(runs) or len(runs) < 2:
        raise ValueError('Need at least two runs, in exactly the saved report order')
    rows = []
    for summary, run in zip(summaries, runs):
        tracking = run.get('tracking') if isinstance(run, dict) else None
        if not isinstance(summary, dict) or not isinstance(tracking, list):
            raise ValueError('Invalid score summary or tracking rows')
        states = [row.get('state') if isinstance(row, dict) else None for row in tracking]
        if any(not isinstance(state, str) or state not in STATES for state in states):
            raise ValueError('Unknown tracking state')
        if type(summary.get('processed_frames')) is not int or summary['processed_frames'] != len(states):
            raise ValueError('Saved scores and tracking frame counts differ')
        rows.append({'folder': summary.get('folder'), 'states': states,
                     'recovery_stability': recovery_stability(states), 'metrics': {}})
    common_accepted = set.intersection(*(
        {i for i, state in enumerate(row['states']) if state in ACCEPTED} for row in rows))
    for export in ('online', 'corrected'):
        residual_maps = []
        error = None
        for summary, row in zip(summaries, rows):
            metrics = summary.get('metrics')
            metric = metrics.get(export) if isinstance(metrics, dict) else None
            if not isinstance(metric, dict) or 'evaluation_error' in metric:
                error = 'At least one full-fit export is unavailable; comparison unassessable'
                break
            frames, values = metric.get('associated_frames'), metric.get('residuals_m')
            if (not isinstance(frames, list) or not isinstance(values, list)
                    or len(frames) != len(values)
                    or any(type(i) is not int or not 0 <= i < len(row['states']) for i in frames)
                    or len(set(frames)) != len(frames)):
                raise ValueError('Invalid or duplicate associated frame IDs')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                raise ValueError('Residuals must be numeric, not booleans')
            distribution(values)  # validate before filtering; never hide invalid rows
            residual_maps.append(dict(zip(frames, values)))
        if error:
            for row in rows:
                row['metrics'][export] = {'unassessable': error}
            continue
        # Same associated AND accepted input rows for every run; no subset refit.
        frames = sorted(common_accepted.intersection(*(set(m) for m in residual_maps)))
        for row, residuals in zip(rows, residual_maps):
            row['metrics'][export] = {
                'common_accepted_associated_frames': frames,
                'common_accepted_translation': distribution([residuals[i] for i in frames]),
                'unassessable': None if frames else 'No common accepted associated frames'}
    for row in rows:
        del row['states']
    return {'schema': 'monocular_score_supplement_v1',
            'percentile_method': 'numpy linear (index=(n-1)*q)',
            'alignment': 'Reuse supplied full-fit residuals; no refitting',
            'limitations': ['Caller must verify matching dataset, frame identity and full-fit score provenance',
                           'Not completion, geometric correctness, false-recovery or promotion evidence',
                           'Recovery <27/30 is a diagnostic; short terminal windows are censored'],
            'runs': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True,
                        help='Run directories in exactly saved report order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [args.scores] + [folder / 'run.json' for folder in args.runs]
    raw = [path.read_bytes() for path in paths]
    report, *runs = [json.loads(value) for value in raw]
    # Exact path binding prevents silently pairing a report with reordered runs.
    if [Path(row['folder']).resolve() for row in report['runs']] != [p.resolve() for p in args.runs]:
        parser.error('--runs must match saved report folder identities and order')
    result = supplement(report, runs)
    result['inputs'] = [{'path': str(path), 'sha256': hashlib.sha256(value).hexdigest()}
                        for path, value in zip(paths, raw)]
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    # Exclusive creation: historical output and input artifacts are never replaced.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
