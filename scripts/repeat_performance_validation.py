"""Revalidate a retrieval correction on the same small pilot and promoted case."""

import argparse
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
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=Path("results/performance"))
    parser.add_argument("--full-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error("Choose a new output directory; completed reports are never overwritten")
    args.output.mkdir(parents=True)
    manifest = {"timing_provisional": True, "runs": [], "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / "src").glob("*.py")}}
    manifest_path = args.output / "validation.json"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MPLCONFIGDIR": str(repo / ".mpl-cache")}

    def launch(name, command):
        print(name, flush=True)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
            started = perf_counter()
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry = {"name": name, "pid": process.pid, "status": "running"}
            manifest["runs"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            code = process.wait()
            entry.update(status="complete" if code == 0 else "failed", exit_code=code, end_to_end_elapsed_s=perf_counter() - started)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if code:
            raise RuntimeError(f"{name} failed; see its isolated log")

    launch("small-pilot", [sys.executable, str(repo / "scripts/run_performance_pilot.py"), "--development-root", str(args.development_root.resolve()), "--gpu-python", str(args.gpu_python.resolve()), "--output", str(args.output.resolve()), "--final-only"])
    comparisons = []
    for label in ["04-stereo", "04-mono", "01-stereo", "01-mono", "tum-desk"]:
        baseline = json.loads((args.baselines / ("baseline" + label) / "evaluation.json").read_text(encoding="utf-8"))
        candidate = json.loads((args.output / ("final" + label) / "evaluation.json").read_text(encoding="utf-8"))
        passed = quality_passed(baseline, candidate) and candidate["configuration"] == baseline["configuration"] and candidate["source_sha256"] == manifest["source_sha256"]
        comparisons.append({"case": label, "quality_passed": passed, "processing_speedup": baseline["elapsed_s"] / candidate["elapsed_s"]})
    manifest["small_pilot_comparisons"] = comparisons
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not all(c["quality_passed"] for c in comparisons):
        raise RuntimeError("Small pilot quality/provenance gate failed; full repeat not launched")
    dev = args.development_root.resolve()
    launch("promoted01-stereo", [str(args.gpu_python.resolve()), str(repo / "scripts/evaluate_shared_slam.py"), "--dataset", "kitti", "--data-root", str(dev / ".datasets/shared-kitti"), "--poses-root", str(dev / "results/benchmark-batch/reference/poses"), "--sequence", "01", "--stereo", "--matching-backend", "cuda", "--output", str((args.output / "promoted01-stereo").resolve())])
    baseline = json.loads(args.full_baseline.read_text(encoding="utf-8"))
    candidate = json.loads((args.output / "promoted01-stereo/evaluation.json").read_text(encoding="utf-8"))
    manifest["full_comparison"] = {"quality_passed": quality_passed(baseline, candidate), "source_matches": candidate["source_sha256"] == manifest["source_sha256"], "processing_speedup": baseline["elapsed_s"] / candidate["elapsed_s"], "target_processing_speedup": 2.5}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not (manifest["full_comparison"]["quality_passed"] and manifest["full_comparison"]["source_matches"]):
        raise RuntimeError("Full repeat quality/provenance gate failed")
    print(json.dumps(manifest["full_comparison"], indent=2))


if __name__ == "__main__":
    main()
