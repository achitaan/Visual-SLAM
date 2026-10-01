"""Audit full public KITTI coverage, retained constraints and evaluator parity."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from kitti import load_poses_txt, validate_pose
from json_output import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('results/benchmark-batch'))
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows, errors = [], []
    for sequence in [f'{index:02d}' for index in range(11)]:
        try:
            raw_path = root / 'stereo' / f'{sequence}.txt'
            raw_report = json.loads(raw_path.with_suffix('.json').read_text())
            graph_path = root / 'graph' / f'{sequence}.txt'
            graph_report = json.loads(graph_path.with_suffix('.json').read_text())
            cache = json.loads((root / 'graph' / f'{sequence}-constraints.json').read_text())
            raw, graph = load_poses_txt(raw_path), load_poses_txt(graph_path)
            truth = load_poses_txt(root / 'reference/poses' / f'{sequence}.txt')
            if not (len(raw) == len(graph) == len(truth) == raw_report['frames'] == graph_report['frames']):
                raise ValueError('Full frame counts differ')
            if raw_report['status'] != 'complete' or graph_report['status'] != 'complete':
                raise ValueError('Incomplete run')
            if raw_report['tracked_pairs'] + raw_report['lost_pairs'] != len(raw) - 1:
                raise ValueError('Tracking pair counts differ')
            if cache['raw_trajectory_sha256'] != hashlib.sha256(raw_path.read_bytes()).hexdigest():
                raise ValueError('Constraint/raw checksum mismatch')
            if graph_report['ground_truth_used_for_constraints'] is not False:
                raise ValueError('Graph constraint provenance is not evaluation-only')
            if graph_report['loops'] != cache['loops'] or graph_report['verified_loop_count'] != len(cache['loops']):
                raise ValueError('Verified loop records differ')
            if graph_report['keyframe_indices'] != cache['keyframe_indices']:
                raise ValueError('Keyframe records differ')
            if not np.array_equal(raw[0], graph[0]):
                raise ValueError('First pose was moved')
            if not cache['loops'] and not np.array_equal(raw, graph):
                raise ValueError('A zero-loop trajectory was changed')
            for loop in cache['loops']:
                validate_pose(np.asarray(loop['measurement']))
                if loop['second_frame'] - loop['first_frame'] < cache['min_frame_gap']:
                    raise ValueError('Loop violates temporal separation')
            for metric in ('ate_rmse_m', 'translation_percent', 'rotation_deg_per_m'):
                if not np.isclose(raw_report[metric], graph_report['before'][metric], atol=1e-9, rtol=0):
                    raise ValueError(f'Raw/graph baseline differs for {metric}')
            row = dict(sequence=sequence, frames=len(raw), lost_pairs=raw_report['lost_pairs'], verified_loops=len(cache['loops']),
                       raw_sha256=cache['raw_trajectory_sha256'], corrected_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                       translation_regressed=graph_report['after']['translation_percent'] > raw_report['translation_percent'] + 1e-9,
                       ate_regressed=graph_report['after']['ate_rmse_m'] > raw_report['ate_rmse_m'] + 1e-9)
            rows.append(row)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{sequence}: {error}')
    for name in ('devkit-parity.json', 'devkit-parity-graph.json'):
        try:
            parity = json.loads((root / name).read_text())
            if len(parity['results']) != 11 or not all(row['within_tolerance'] for row in parity['results']):
                errors.append(f'{name}: incomplete or failed official evaluator comparison')
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{name}: {error}')
    try:
        provenance = json.loads((root / 'ground-truth-verification.json').read_text())
        if not provenance['passed'] or len(provenance['results']) != 11:
            errors.append('Official public ground-truth comparison failed or is incomplete')
        for reference in provenance['results']:
            current = root / 'reference/poses' / f"{reference['sequence']}.txt"
            if hashlib.sha256(current.read_bytes()).hexdigest() != reference['local_file_sha256']:
                errors.append(f"Ground-truth reference changed: {reference['sequence']}")
    except (OSError, ValueError, KeyError) as error:
        errors.append(f'ground-truth-verification.json: {error}')
    scratch = Path(__file__).resolve().parents[1] / '.datasets/batch-scratch'
    if scratch.exists() and any(scratch.iterdir()):
        errors.append('Owned temporary benchmark cache has not been cleaned')
    report = dict(passed=not errors, complete_sequences=len(rows), total_frames=sum(row['frames'] for row in rows),
                  total_lost_pairs=sum(row['lost_pairs'] for row in rows), sequences=rows, errors=errors,
                  checks='Full coverage, finite rigid poses, tracking counts, fixed origin, zero-loop identity, saved measured transforms, raw checksums and official drift parity.')
    write_json(root / 'verification.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
