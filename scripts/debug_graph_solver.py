"""Compare graph linear solvers on saved image constraints, without changing production settings."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pose_graph
from eval_pose_graph import optimize_image_graph
from kitti import load_poses_txt
from json_output import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='06')
    parser.add_argument('--solver', choices=['exact', 'lsmr'], default='exact')
    parser.add_argument('--max-evaluations', type=int, default=500)
    args = parser.parse_args()
    root = Path('results/benchmark-batch')
    cache = json.loads((root / 'graph' / f'{args.sequence}-constraints.json').read_text())
    raw = load_poses_txt(root / 'stereo' / f'{args.sequence}.txt')
    original = pose_graph.least_squares
    def measured(fun, x0, **kwargs):
        if args.solver == 'exact':
            kwargs.pop('jac_sparsity', None)
            kwargs['tr_solver'], kwargs['tr_options'] = 'exact', {}
        started = time.perf_counter()
        result = original(fun, x0, **kwargs)
        stats = dict(sequence=args.sequence, solver=args.solver, success=bool(result.success),
                     evaluations=int(result.nfev), cost=float(result.cost), optimality=float(result.optimality),
                     message=str(result.message), elapsed_s=time.perf_counter() - started)
        write_json(root / 'diagnostics' / f'{args.sequence}-{args.solver}.json', stats)
        print(json.dumps(stats, indent=2), flush=True)
        return result
    pose_graph.least_squares = measured
    optimize_image_graph(raw, cache['keyframe_indices'], cache['loops'], args.max_evaluations)


if __name__ == '__main__':
    main()
