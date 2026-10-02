"""Check exported trajectories against estimator-only, verified stereo measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti import load_poses_txt, validate_pose


def inspect(folder):
    report = json.loads((folder / 'evaluation.json').read_text())
    run = json.loads((folder / 'run.json').read_text())
    if not report['stereo'] or report['ground_truth_used_for_estimation']:
        raise ValueError('Requires an image-only stereo estimate')
    poses = load_poses_txt(folder / 'poses.txt')
    if len(poses) != report['frames'] or len(run['tracking']) != len(poses):
        raise ValueError('Frame coverage mismatch')
    for pose in poses:
        validate_pose(pose)
    for name, digest in report['source_sha256'].items():
        if hashlib.sha256((folder / 'source' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source archive mismatch')
    measurements = run.get('independent_stereo_motion')
    if measurements is None:
        raise ValueError('This saved revision does not export independent stereo motion')
    errors = []
    for edge in measurements:
        first, second = edge['previous_frame'], edge['frame']
        if not 0 <= first < second < len(poses):
            raise ValueError('Invalid measurement endpoints')
        if any(run['tracking'][i]['state'] not in ('tracking', 'relocalized') for i in (first, second)):
            raise ValueError('Failed frame contributed a motion measurement')
        measured = np.asarray(edge['measurement'], float)
        validate_pose(measured)
        error = np.linalg.inv(measured) @ np.linalg.inv(poses[first]) @ poses[second]
        errors.append({'previous_frame': first, 'frame': second,
                       'translation_error_m': float(np.linalg.norm(error[:3, 3])),
                       'rotation_error_deg': float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))})
    violations = [e for e in errors if e['translation_error_m'] > .5 or e['rotation_error_deg'] > 1.5]
    return {'frames': len(poses), 'coverage': report['coverage'], 'measurements': len(errors),
            'max_translation_error_m': max((e['translation_error_m'] for e in errors), default=0.),
            'max_rotation_error_deg': max((e['rotation_error_deg'] for e in errors), default=0.),
            'violations': violations, 'passed': bool(errors) and not violations,
            'reference_poses_used': False, 'source_archives_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect(args.run)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
