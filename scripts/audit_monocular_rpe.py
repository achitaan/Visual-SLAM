"""Fixed-time relative pose diagnostics for final corrected exports only.

This module never estimates a trajectory. Online pairs are not supported because
saved process-time poses lack common correction-epoch transforms. The declared
endpoint rule is nearest full-input timestamp to start+1 second, within 20ms;
equal-distance ties choose the earlier frame. No interpolation or pair refit.
"""
import math

import numpy as np


def fixed_pairs(timestamps, associated_frames):
    times = np.asarray(timestamps, dtype=float)
    if (times.ndim != 1 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Input timestamps must be finite and strictly increasing')
    frames = list(associated_frames)
    if (any(type(i) is not int or not 0 <= i < len(times) for i in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError('Associated frames must be unique in-range integer IDs')
    available = set(frames)
    pairs = []
    for first in sorted(available):
        target = times[first] + 1.
        upper = int(np.searchsorted(times, target))
        candidates = [i for i in (upper - 1, upper) if first < i < len(times)]
        if not candidates:
            continue
        last = min(candidates, key=lambda i: (abs(times[i] - target), i))
        if abs(times[last] - target) <= .02 and last in available:
            pairs.append((first, last))
    return pairs


def validate_poses(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError('Poses must be finite Nx4x4 matrices')
    if not np.allclose(poses[:, 3], [0., 0., 0., 1.], rtol=0., atol=1e-8):
        raise ValueError('Invalid homogeneous pose row')
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0., atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1., rtol=0., atol=1e-6)):
        raise ValueError('Pose rotations must be proper SO3')
    return poses


def corrected_pair_errors(poses, truth, timestamps, associated_frames, scale):
    """Use one saved full-fit scale; global rotation/translation cancel in RPE.

    poses covers all input rows, truth follows associated_frames exactly.
    Output units are metres and degrees (not distance-normalized KITTI drift).
    """
    poses, truth = validate_poses(poses), validate_poses(truth)
    frames = list(associated_frames)
    if (len(poses) > len(timestamps) or len(truth) != len(frames)
            or any(type(i) is not int or not 0 <= i < len(poses) for i in frames)):
        raise ValueError('Pose/timestamp/association lengths differ')
    try:
        finite_scale = math.isfinite(scale)
    except (TypeError, OverflowError):
        finite_scale = False
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not finite_scale or scale <= 0:
        raise ValueError('Saved full-fit scale must be finite and positive')
    pairs = fixed_pairs(timestamps, frames)
    reference = dict(zip(frames, truth))
    result = []
    for first, last in pairs:
        estimated_rotation = poses[first, :3, :3].T @ poses[last, :3, :3]
        estimated_translation = scale * (poses[first, :3, :3].T @ (poses[last, :3, 3] - poses[first, :3, 3]))
        gt_rotation = reference[first][:3, :3].T @ reference[last][:3, :3]
        gt_translation = reference[first][:3, :3].T @ (reference[last][:3, 3] - reference[first][:3, 3])
        # ||R_est^T (t_gt-t_est)|| = ||t_gt-t_est||.
        translation = float(np.linalg.norm(gt_translation - estimated_translation))
        rotation = float(np.degrees(np.arccos(np.clip(
            (np.trace(estimated_rotation.T @ gt_rotation) - 1.) / 2., -1., 1.))))
        if not np.isfinite(translation) or not np.isfinite(rotation):
            raise ValueError('Relative errors overflowed')
        result.append({'first_frame': first, 'last_frame': last,
                       'delta_seconds': float(timestamps[last] - timestamps[first]),
                       'translation_error_m': translation, 'rotation_error_deg': rotation})
    return result


def audit(report, runs, exports, timestamps, truth):
    # Reuse orientation audit's strict input/association/full-fit validation.
    from audit_monocular_rotations import audit as rotation_audit, associations
    from supplement_monocular_scores import distribution
    validated = rotation_audit(report, runs, exports, timestamps, truth)
    accepted = [{i for i, row in enumerate(run['tracking'])
                 if row['state'] in ('tracking', 'relocalized')} for run in runs]
    common = set.intersection(*accepted)
    output = []
    for summary, run, exported, checked, accepted_frames in zip(
            report['runs'], runs, exports, validated['runs'], accepted):
        result = {'folder': summary.get('folder'), 'corrected': {}}
        if 'unassessable' in checked['metrics']['corrected']:
            result['corrected'] = {'unassessable': 'Corrected export/full-fit unavailable'}
            output.append(result)
            continue
        pairs, rotations = associations(timestamps, truth, len(run['tracking']))
        frames, gt_ids = [p[0] for p in pairs], [p[1] for p in pairs]
        gt = np.tile(np.eye(4), (len(gt_ids), 1, 1))
        gt[:, :3, :3], gt[:, :3, 3] = rotations[gt_ids], truth[gt_ids, 1:4]
        poses = np.tile(np.eye(4), (len(run['tracking']), 1, 1))
        poses[:, :3] = exported['corrected']
        scale = summary['metrics']['corrected']['alignment'].get('scale')
        errors = corrected_pair_errors(poses, gt, timestamps, frames, scale)
        result['corrected']['pairs'] = errors
        for label, mask in [('all', None), ('accepted', accepted_frames), ('common_accepted', common)]:
            selected = [row for row in errors if mask is None or
                        (row['first_frame'] in mask and row['last_frame'] in mask)]
            translation = distribution([row['translation_error_m'] for row in selected])
            angle = distribution([row['rotation_error_deg'] for row in selected])
            if angle:
                angle = {key.replace('_m', '_deg'): value for key, value in angle.items()}
            result['corrected'][label] = {'pair_ids': [[r['first_frame'], r['last_frame']] for r in selected],
                                          'translation': translation, 'rotation': angle,
                                          'unassessable': None if selected else 'No eligible pairs'}
        output.append(result)
    # A missing export cannot silently turn the other run's common subset into a comparison.
    if any('unassessable' in row['corrected'] for row in output):
        for row in output:
            row['corrected']['common_accepted'] = {'unassessable': 'At least one corrected export/full fit unavailable'}
    else:
        # Different processed ranges can produce different available endpoint sets.
        common_pairs = set.intersection(*({tuple(p) for p in row['corrected']['common_accepted']['pair_ids']}
                                          for row in output))
        for row in output:
            selected = [p for p in row['corrected']['pairs'] if (p['first_frame'], p['last_frame']) in common_pairs]
            angle = distribution([p['rotation_error_deg'] for p in selected])
            row['corrected']['common_accepted'] = {
                'pair_ids': sorted([list(p) for p in common_pairs]),
                'translation': distribution([p['translation_error_m'] for p in selected]),
                'rotation': {k.replace('_m', '_deg'): v for k, v in angle.items()} if angle else None,
                'unassessable': None if selected else 'No common accepted associated pairs'}
    return {'schema': 'monocular_corrected_rpe_audit_v1', 'runs': output,
            'pair_policy': 'Nearest full-input endpoint to start+1s within20ms; tieearliest; no interpolation',
            'alignment': 'Saved full-fit Sim3 scale; global rotation/translation cancel; no pair/subset refit',
            'percentile_method': 'numpy linear (index=(n-1)*q)',
            'online': {'unassessable': 'No common correction-epoch transforms in saved online exports'},
            'limitations': ['Final corrected trajectory diagnostic, not online odometry latency/accuracy',
                           'Caller must verify dataset/frame/full-fit and completed-export provenance',
                           'No completion, false-recovery threshold or promotion verdict']}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from audit_monocular_rotations import load_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
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
