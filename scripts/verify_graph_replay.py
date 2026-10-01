"""Compare retained graph references with constraint-only optimizer replays."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from kitti import load_poses_txt
from json_output import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    root = args.output_root
    tolerances = dict(position_m=.01, rotation_deg=.001, ate_m=.001, translation_percent=.0001)
    results = []
    for reference_path in sorted((root / 'solver-reference').glob('[0-9][0-9].json')):
        sequence = reference_path.stem
        old = json.loads(reference_path.read_text())
        new = json.loads((root / 'graph' / reference_path.name).read_text())
        previous = np.array(load_poses_txt(reference_path.with_suffix('.txt')))
        replay = np.array(load_poses_txt(root / 'graph' / f'{sequence}.txt'))
        if previous.shape != replay.shape:
            raise ValueError(f'{sequence}: replay coverage differs')
        distance = float(np.linalg.norm(previous[:, :3, 3] - replay[:, :3, 3], axis=1).max())
        rotation = float(np.rad2deg(Rotation.from_matrix(previous[:, :3, :3].transpose(0, 2, 1) @ replay[:, :3, :3]).magnitude()).max())
        ate_delta = abs(old['after']['ate_rmse_m'] - new['after']['ate_rmse_m'])
        translation_delta = abs(old['after']['translation_percent'] - new['after']['translation_percent'])
        same_loops = old['loops'] == new['loops']
        results.append(dict(sequence=sequence, frames=len(replay), identical_measured_loops=same_loops,
                            max_position_difference_m=distance, max_rotation_difference_deg=rotation,
                            ate_difference_m=ate_delta, translation_difference_percent=translation_delta,
                            within_tolerance=bool(same_loops and distance <= tolerances['position_m'] and rotation <= tolerances['rotation_deg']
                                                  and ate_delta <= tolerances['ate_m'] and translation_delta <= tolerances['translation_percent'])))
    report = dict(tolerances=tolerances, compared_sequences=len(results), passed=bool(results) and all(row['within_tolerance'] for row in results), results=results,
                  scope='Same measured image constraints and graph weights; replays read serialized raw KITTI poses and use the current grouped Jacobian.')
    write_json(root / 'graph-replay-verification.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
