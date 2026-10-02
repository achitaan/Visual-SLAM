"""Matched whole-process KITTI04 check; keeps frozen sources and old results intact."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from run_performance_pilot import quality_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opencv-threads", type=int, default=1, help="Candidate workers; frozen baseline remains one")
    parser.add_argument("--reuse-baseline", type=Path, help="Read-only previous comparison.json with identical input and frozen source hashes")
    args = parser.parse_args()
    if args.opencv_threads < 1:
        parser.error("Worker count must be positive")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing experiments are never overwritten")
    output.mkdir(parents=True)
    images = sorted((args.data_root / "sequences/04").glob("image_[01]/*.png"))
    if len(images) != 542:
        parser.error("Full KITTI04 must contain 271 paired images")
    input_files = images + [args.data_root / "sequences/04/calib.txt", args.data_root / "sequences/04/times.txt", args.poses_root / "04.txt"]
    input_hashes = {str(p.relative_to(args.data_root)) if p.is_relative_to(args.data_root) else "reference/04.txt": hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MPLCONFIGDIR": str(repo / ".mpl-cache")}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "timing_provisional": True,
              "candidate_opencv_threads": args.opencv_threads,
              "policy": "Parent wall time from child launch to successful exit, including interpreter/imports, estimator setup, all frames, input loading, background shutdown, exports and reference evaluation. Serial runs, frozen OpenCV workers=1, BLAS workers=1. Candidate worker count recorded. Shared-host background load uncontrolled; same full input files, no profiling.",
              "input_sha256": input_hashes, "runs": {}}
    path = output / "comparison.json"
    for name, root, interpreter in [("baseline", args.baseline_root.resolve(), Path(sys.executable)), ("candidate", repo, args.gpu_python.resolve())]:
        if name == "baseline" and args.reuse_baseline:
            previous = json.loads(args.reuse_baseline.read_text(encoding="utf-8"))
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "src").glob("*.py")}
            base = previous["runs"]["baseline"]
            if previous["input_sha256"] != input_hashes or base["exit_code"] != 0 or base["evaluation"]["source_sha256"] != hashes:
                raise RuntimeError("Saved baseline inputs/source do not match this experiment")
            report["runs"]["baseline"] = base
            report["baseline_reused_from_started_utc"] = previous["started_utc"]
            continue
        target = output / name
        command = [str(interpreter), str(repo / "scripts/performance_case.py"), "--revision-root", str(root), "--dataset", "kitti", "--data-root", str(args.data_root.resolve()), "--poses-root", str(args.poses_root.resolve()), "--sequence", "04", "--stereo", "--output", str(target)]
        if name == "candidate":
            command += ["--matching-backend", "cuda", "--opencv-threads", str(args.opencv_threads)]
        # Warm the same disk files immediately before each measured process.
        # This does not populate or modify any estimator/frozen benchmark cache.
        for p in input_files:
            if hashlib.sha256(p.read_bytes()).hexdigest() != input_hashes[str(p.relative_to(args.data_root)) if p.is_relative_to(args.data_root) else "reference/04.txt"]:
                raise RuntimeError("Input changed between runs")
        print(name, flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            started = perf_counter()
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = perf_counter() - started
        report["runs"][name] = {"end_to_end_elapsed_s": elapsed, "exit_code": result.returncode}
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect its separate log")
        report["runs"][name]["evaluation"] = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
    base, candidate = report["runs"]["baseline"], report["runs"]["candidate"]
    report.update(end_to_end_speedup=base["end_to_end_elapsed_s"] / candidate["end_to_end_elapsed_s"],
                  elapsed_reduction_fraction=1 - candidate["end_to_end_elapsed_s"] / base["end_to_end_elapsed_s"],
                  worthwhile_end_to_end=candidate["end_to_end_elapsed_s"] <= .8 * base["end_to_end_elapsed_s"],
                  quality_passed=quality_passed(base["evaluation"], candidate["evaluation"]),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "input_sha256")}, indent=2))
    if not (report["worthwhile_end_to_end"] and report["quality_passed"]):
        raise SystemExit("Whole-process speed/quality gate failed")


if __name__ == "__main__":
    main()
