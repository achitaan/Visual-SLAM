"""Check current-source pilot evidence against explicit speed and quality gates."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from run_performance_pilot import quality_passed


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=Path("results/performance"))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--full-baseline", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--end-to-end", type=Path, required=True)
    parser.add_argument("--tests-log", type=Path, required=True)
    parser.add_argument("--target-speedup", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_speedup <= 1:
        parser.error("Target must exceed one")
    repo = Path(__file__).resolve().parents[1]
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / "src").glob("*.py")}
    rows, missing = [], []
    for label, frames in [("04-stereo", 271), ("04-mono", 271), ("01-stereo", 300), ("01-mono", 300), ("tum-desk", 300), ("01-full-stereo", 1101)]:
        baseline = args.baselines / ("baseline" + label) if label != "01-full-stereo" else args.full_baseline.parent
        candidate = args.candidates / ("final" + label) if label != "01-full-stereo" else args.candidates / "promoted01-stereo"
        if not (candidate / "evaluation.json").exists():
            missing.append(label)
            continue
        b, c = read(baseline / "evaluation.json"), read(candidate / "evaluation.json")
        poses_b, poses_c = np.loadtxt(baseline / "poses.txt"), np.loadtxt(candidate / "poses.txt")
        provenance = c["source_sha256"] == current
        quality = quality_passed(b, c) and b["configuration"] == c["configuration"] and b["frames"] == c["frames"] == frames
        rows.append({"case": label, "frames": frames, "current_source": provenance, "quality_passed": quality,
                     "frozen_elapsed_s": b["elapsed_s"], "candidate_elapsed_s": c["elapsed_s"],
                     "processing_speedup": b["elapsed_s"] / c["elapsed_s"],
                     "frozen_ate_m": b["metrics"]["ate_rmse_m"], "candidate_ate_m": c["metrics"]["ate_rmse_m"],
                     "lost_intervals_equal": b["lost_intervals"] == c["lost_intervals"],
                     "maximum_pose_element_difference": float(np.max(np.abs(poses_b - poses_c))) if poses_b.shape == poses_c.shape else None,
                     "ram_increase_percent": 100 * (c["peak_memory_mb"] / b["peak_memory_mb"] - 1),
                     "landmark_cache_array_payload_mb": c["landmarks"] * (128 * 4 + 3 * 8 + 8) / 1024**2,
                     "estimator_setup_ram_mb": c.get("estimator_setup_peak_memory_mb"),
                     "torch_peak_allocated_vram_mb": c.get("matching", {}).get("peak_cuda_allocated_mb"),
                     "torch_peak_reserved_vram_mb": c.get("matching", {}).get("peak_cuda_reserved_mb")})
    retrieval, end_to_end = read(args.retrieval), read(args.end_to_end)
    no_misses = len(retrieval["audit"]) == retrieval["known_loops"] == retrieval["retrieved_loops"] == 15 and all(a["retrieved"] and a["first_keyframe"] in a["shortlist"] and len(a["shortlist"]) <= 20 for a in retrieval["audit"])
    retrieval_current = retrieval.get("index_sha256") == current["keyframe_index.py"] and retrieval.get("loop_worker_sha256") == current["live_loops.py"] and retrieval.get("keyframe_writer_sha256") == current["shared_slam.py"]
    base, candidate = end_to_end["runs"]["baseline"], end_to_end["runs"]["candidate"]
    e2e_passed = base["exit_code"] == candidate["exit_code"] == 0 and quality_passed(base["evaluation"], candidate["evaluation"]) and candidate["evaluation"]["source_sha256"] == current and candidate["end_to_end_elapsed_s"] <= .8 * base["end_to_end_elapsed_s"]
    frozen = {}
    for p in (args.baselines / "frozen/src").glob("*.py"):
        data = subprocess.check_output(["git", "show", "63454a0:src/" + p.name], cwd=repo)
        frozen[p.name] = p.read_bytes().replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n")
    tests_text = args.tests_log.read_text(encoding="utf-8")
    tests_passed = "95 passed" in tests_text and "FAILED" not in tests_text and "ERROR" not in tests_text
    target_cases = [r["case"] for r in rows if r["current_source"] and r["quality_passed"] and r["processing_speedup"] >= args.target_speedup]
    aggregate_speedup = sum(r["frozen_elapsed_s"] for r in rows) / sum(r["candidate_elapsed_s"] for r in rows) if rows else None
    aggregate_passed = not missing and aggregate_speedup is not None and aggregate_speedup >= args.target_speedup
    passed = not missing and len(rows) == 6 and all(r["current_source"] and r["quality_passed"] for r in rows) and no_misses and retrieval_current and e2e_passed and tests_passed and len(frozen) == 24 and all(frozen.values()) and aggregate_passed
    report = {"target_processing_speedup": args.target_speedup, "scope": "Sum of processing elapsed times for the five requested pilot cases plus promoted full KITTI01 stereo, with identical inputs/modes/frame limits and quality gates on every case. Case-specific results are also reported.", "timing_provisional": True,
              "passed": passed, "missing_cases": missing, "cases": rows, "processing_target_cases": target_cases,
              "aggregate_processing_speedup": aggregate_speedup, "aggregate_processing_target_passed": aggregate_passed,
              "known_pair_retrieval_passed": no_misses, "retrieval_matches_current_source": retrieval_current,
              "whole_process_20_percent_gate_passed": e2e_passed, "whole_process_speedup": end_to_end["end_to_end_speedup"],
              "whole_process_candidate_workers": end_to_end["candidate_opencv_threads"],
              "frozen_snapshot_matches_63454a0": frozen, "tests_passed": tests_passed, "tests_log_sha256": hashlib.sha256(args.tests_log.read_bytes()).hexdigest(),
              "current_source_sha256": current,
              "memory_investigation": "CUDA runtime setup and persistent SIFT/position/ID cache payload explain the observed process RAM increase. The appearance view shares memory with keyframe descriptors. Container overhead is excluded from payload estimates; Torch VRAM excludes driver/context allocations.",
              "merge_ready": False, "limitations": "Shared-host timing is provisional; index was tuned on these saved known pairs. Review and held-out live validation remain prerequisites to merging."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("cases", "frozen_snapshot_matches_63454a0", "current_source_sha256")}, indent=2))
    if not passed:
        raise SystemExit("Acceptance incomplete or failed; inspect the evidence report")


if __name__ == "__main__":
    main()
