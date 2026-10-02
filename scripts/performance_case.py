"""Run an unchanged baseline evaluator with lightweight frame-latency observation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    args, evaluation_args = parser.parse_known_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = args.revision_root.resolve()
    spec = importlib.util.spec_from_file_location("pilot_evaluator", root / "scripts/evaluate_shared_slam.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    frame_times = []
    original = evaluator.SharedSlam.process

    def observed(self, *args, **kwargs):
        started = perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            frame_times.append(perf_counter() - started)

    evaluator.SharedSlam.process = observed
    sys.argv = [str(root / "scripts/evaluate_shared_slam.py"), *evaluation_args]
    evaluator.main()
    output = Path(evaluation_args[evaluation_args.index("--output") + 1])
    path = output / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    import numpy as np
    report["frame_latency"] = {
        "calls": len(frame_times),
        "total_s": sum(frame_times),
        "median_ms": float(np.median(frame_times) * 1000),
        "p95_ms": float(np.percentile(frame_times, 95) * 1000),
    }
    report["baseline_observation"] = "Frame wall times only; estimator and evaluator sources unchanged"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
