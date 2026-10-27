"""Offline TUM orientation audit using saved full-trajectory alignment rotations.

Never refits an alignment or imports/runs the estimator. No RPE is computed:
online correction-epoch provenance must be resolved separately.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ACCEPTED = {'tracking', 'relocalized'}
STATES = ACCEPTED | {'initializing', 'lost'}


def so3(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(np.swapaxes(matrix, -1, -2) @ matrix, np.eye(3), rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.det(matrix), 1., rtol=0, atol=1e-6)):
        raise ValueError('Expected finite proper SO(3) rotation(s)')
    return matrix


def associations(timestamps, truth, frames):
    timestamps, truth = np.asarray(timestamps, float), np.asarray(truth, float)
    if (timestamps.ndim != 1 or len(timestamps) < frames or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or truth.ndim != 2 or truth.shape[1] != 8
            or not len(truth) or not np.isfinite(truth).all() or np.any(np.diff(truth[:, 0]) <= 0)):
        raise ValueError('Invalid TUM timestamps/ground truth')
    quaternions = truth[:, 4:8]  # TUM xyzw, camera-to-world
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1., rtol=0, atol=1e-3):
        raise ValueError('Ground-truth quaternions must be normalized')
    used, pairs = set(), []
    for frame, stamp in enumerate(timestamps[:frames]):
        index = int(np.argmin(abs(truth[:, 0] - stamp)))
        if abs(truth[index, 0] - stamp) <= .02 and index not in used:
            used.add(index)
            pairs.append((frame, index))
    return pairs, Rotation.from_quat(quaternions).as_matrix()


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {'count': len(values), 'rmse_deg': float(np.sqrt(np.mean(values**2))),
            'median_deg': float(np.percentile(values, 50, method='linear')),
            'p95_deg': float(np.percentile(values, 95, method='linear')),
            'maximum_deg': float(values.max())}


def audit(report, runs, exports, timestamps, truth):
    summaries = report.get('runs') if isinstance(report, dict) else None
    if not isinstance(summaries, list) or len(summaries) != len(runs) or len(exports) != len(runs) or len(runs) < 2:
        raise ValueError('Need at least two matching report/run/export entries')
    output, accepted_sets = [], []
    for summary, run, poses_by_export in zip(summaries, runs, exports):
        tracking = run.get('tracking') if isinstance(run, dict) else None
        if not isinstance(tracking, list) or not isinstance(summary, dict):
            raise ValueError('Invalid tracking or score summary')
        states = [row.get('state') if isinstance(row, dict) else None for row in tracking]
        if any(not isinstance(state, str) or state not in STATES for state in states):
            raise ValueError('Unknown tracking state')
        if type(summary.get('processed_frames')) is not int or summary['processed_frames'] != len(states):
            raise ValueError('Score/tracking frame count mismatch')
        pairs, gt_rotations = associations(timestamps, truth, len(states))
        frames = [frame for frame, _ in pairs]
        gt_ids = [index for _, index in pairs]
        accepted = {i for i, state in enumerate(states) if state in ACCEPTED}
        accepted_sets.append(accepted)
        recovery = [i for i in range(1, len(states)) if states[i-1] == 'lost' and states[i] in ACCEPTED]
        result = {'folder': summary.get('folder'), 'metrics': {}}
        metrics = summary.get('metrics')
        for label in ('online', 'corrected'):
            metric = metrics.get(label) if isinstance(metrics, dict) else None
            if not isinstance(metric, dict) or 'evaluation_error' in metric or 'alignment' not in metric:
                result['metrics'][label] = {'unassessable': 'Saved full-fit alignment unavailable'}
                continue
            saved_frames = metric.get('associated_frames')
            if (not isinstance(saved_frames, list) or any(type(i) is not int for i in saved_frames)
                    or saved_frames != frames):
                raise ValueError('Reconstructed association differs from saved full-fit frame IDs')
            alignment = metric['alignment']
            if not isinstance(alignment, dict):
                raise ValueError('Invalid saved alignment')
            rotation = so3(alignment.get('rotation'))
            if rotation.shape != (3, 3):
                raise ValueError('Expected one full-fit alignment rotation')
            poses = poses_by_export.get(label)
            if poses is None:
                result['metrics'][label] = {'unassessable': 'Pose export unavailable'}
                continue
            poses = np.asarray(poses, float)
            if poses.shape != (len(states), 3, 4) or not np.isfinite(poses).all():
                raise ValueError('Pose export must have one finite 3x4 pose per processed frame')
            so3(poses[:, :, :3])
            # Camera-to-world alignment is left multiplication; scale/translation
            # do not act on orientation. All subsets use these same fixed errors.
            aligned = rotation @ poses[frames, :, :3]
            relative = np.swapaxes(gt_rotations[gt_ids], -1, -2) @ aligned
            angles = np.rad2deg(Rotation.from_matrix(relative).magnitude()) if frames else np.array([])
            by_frame = dict(zip(frames, angles.tolist()))
            result['metrics'][label] = {
                'associated_frames': frames, 'ground_truth_row_ids': gt_ids,
                'angular_error_deg': angles.tolist(), 'all': distribution(angles),
                'accepted': distribution([by_frame[i] for i in frames if i in accepted]),
                'recovery_events': [{'frame': i, 'angular_error_deg': by_frame.get(i),
                                     'unassessable': None if i in by_frame else 'No associated ground truth'}
                                    for i in recovery]}
        output.append(result)
    common = set.intersection(*accepted_sets)
    for label in ('online', 'corrected'):
        metrics = [row['metrics'][label] for row in output]
        if any('unassessable' in metric for metric in metrics):
            for metric in metrics:
                metric['common_accepted'] = None
                metric['common_unassessable'] = 'At least one run lacks this export/full fit'
            continue
        frames = sorted(common.intersection(*(set(metric['associated_frames']) for metric in metrics)))
        for metric in metrics:
            errors = dict(zip(metric['associated_frames'], metric['angular_error_deg']))
            metric['common_accepted_associated_frames'] = frames
            metric['common_accepted'] = distribution([errors[i] for i in frames])
            metric['common_unassessable'] = None if frames else 'No common accepted associated frames'
    return {'schema': 'monocular_rotation_audit_v1', 'runs': output,
            'percentile_method': 'numpy linear (index=(n-1)*q)',
            'alignment': 'Saved full-fit rotation left-multiplied onto camera-to-world orientations; never refit',
            'limitations': ['Caller must verify original full-fit score provenance and dataset identity',
                           'No RPE: online correction-epoch provenance unresolved',
                           'No completion, false-recovery threshold, or promotion verdict']}


def load_inputs(args, parser):
    evidence = []
    def read(path):
        raw = path.read_bytes()
        evidence.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        return raw
    report = json.loads(read(args.scores))
    if [Path(row['folder']).resolve() for row in report['runs']] != [path.resolve() for path in args.runs]:
        parser.error('--runs must match saved report identities/order')
    rgb = read(args.dataset / 'rgb.txt').decode('utf-8')
    timestamps = [float(line.split()[0]) for line in rgb.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    truth = np.loadtxt(io.BytesIO(read(args.dataset / 'groundtruth.txt')), ndmin=2)
    runs, exports = [], []
    for folder in args.runs:
        evaluation = json.loads(read(folder / 'evaluation.json'))
        if evaluation.get('dataset') != 'tum' or evaluation.get('sequence') != args.sequence:
            parser.error('Evaluation dataset/sequence does not match requested TUM identity')
        run = json.loads(read(folder / 'run.json'))
        if type(evaluation.get('frames')) is not int or evaluation['frames'] != len(run.get('tracking', [])):
            parser.error('Evaluation/tracking frame count mismatch')
        runs.append(run)
        poses = {}
        for label, filename in [('online', 'online-poses.txt'), ('corrected', 'poses.txt')]:
            path = folder / filename
            if path.exists():
                raw = np.loadtxt(io.BytesIO(read(path)), ndmin=2)
                if raw.shape[1] != 12:
                    raise ValueError('Pose text rows must contain exactly 12 values')
                poses[label] = raw.reshape(-1, 3, 4)
            else:
                poses[label] = None
        exports.append(poses)
    return report, runs, exports, timestamps, truth, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sequence', required=True, help='Evaluation sequence identity, e.g. freiburg1_desk')
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report, runs, exports, timestamps, truth, evidence = load_inputs(args, parser)
    result = audit(report, runs, exports, timestamps, truth)
    result.update(inputs=evidence, dataset=str(args.dataset), sequence=args.sequence)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
