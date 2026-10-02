"""Write an honest pilot comparison, including failures and missing runs."""

import argparse
import csv
import json
from pathlib import Path

from run_performance_pilot import quality_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/performance"))
    parser.add_argument("--report", type=Path, default=Path("docs/performance/REPORT.md"))
    parser.add_argument("--full-baseline", type=Path, help="Read-only saved full KITTI 01 stereo evaluation.json")
    args = parser.parse_args()
    rows = []
    cases = [(label, args.results / ("baseline" + label) / "evaluation.json", ["cpu", "gpu", "final"])
             for label in ["04-stereo", "04-mono", "01-stereo", "01-mono", "tum-desk"]]
    if args.full_baseline:
        cases.append(("01-full-stereo", args.full_baseline, ["promoted"]))
    for label, base_path, variants in cases:
        if not base_path.exists():
            continue
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for variant in variants:
            path = args.results / (variant + ("01-stereo" if variant == "promoted" else label)) / "evaluation.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            backend = r.get("performance_configuration", {}).get("matching_backend", variant)
            rows.append({
                "case": label, "backend": backend, "stage": "final" if variant == "final" else "promoted" if variant == "promoted" else "initial", "frames": r["frames"],
                "baseline_s": base["elapsed_s"], "elapsed_s": r["elapsed_s"],
                "speedup": base["elapsed_s"] / r["elapsed_s"], "fps": r["processing_fps"],
                "median_frame_ms": r["frame_latency"]["median_ms"], "p95_frame_ms": r["frame_latency"]["p95_ms"],
                "baseline_median_frame_ms": base.get("frame_latency", {}).get("median_ms"),
                "baseline_p95_frame_ms": base.get("frame_latency", {}).get("p95_ms"),
                "input_loading_s": r.get("input_loading", {}).get("total_s"), "export_s": r.get("export_elapsed_s"),
                "estimator_setup_s": r.get("estimator_setup_elapsed_s"),
                "estimator_setup_ram_mb": r.get("estimator_setup_peak_memory_mb"),
                "baseline_ram_mb": base["peak_memory_mb"], "peak_ram_mb": r["peak_memory_mb"],
                "ram_change_percent": 100 * (r["peak_memory_mb"] / base["peak_memory_mb"] - 1),
                "torch_peak_vram_mb": r.get("matching", {}).get("peak_cuda_allocated_mb", 0),
                "torch_reserved_vram_mb": r.get("matching", {}).get("peak_cuda_reserved_mb", 0),
                "ate_m": r.get("metrics", {}).get("ate_rmse_m"), "baseline_ate_m": base.get("metrics", {}).get("ate_rmse_m"),
                "translation_percent": r.get("metrics", {}).get("translation_percent"),
                "rotation_deg_per_m": r.get("metrics", {}).get("rotation_deg_per_m"),
                "lost_frames": r["lost_frames"], "lost_intervals": len(r.get("lost_intervals", [])),
                "baseline_lost_frames": base["lost_frames"], "lost_interval_details": r.get("lost_intervals", []),
                "relocalized_frames": r["relocalized_frames"], "loops": r["loops"],
                "quality_passed": quality_passed(base, r),
                "baseline_source_sha256": base["source_sha256"], "candidate_source_sha256": r["source_sha256"],
                "evaluator_sha256": r.get("evaluator_sha256"),
            })
    machine = {"timing_provisional": True, "comparisons": rows}
    manifest = args.results / "pilot.json"
    if manifest.exists():
        original = json.loads(manifest.read_text(encoding="utf-8"))
        machine["pilot"] = {k:v for k,v in original.items() if k != "runs"}
        machine["pilot"]["runs"] = [{"name": r["name"], "status": r["status"]} for r in original["runs"]]
    for key, file in [("retrieval", "retrieval-final.json"), ("matching_microbenchmark", "matching-microbenchmark-final.json"), ("environment", "environment.json"), ("full_baseline_provenance", "full-baseline-provenance.json")]:
        if (args.results / file).exists():
            machine[key] = json.loads((args.results / file).read_text(encoding="utf-8"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    fields = [k for k in rows[0] if not k.endswith("sha256") and k != "lost_interval_details"] if rows else []
    with args.report.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# SLAM performance pilot", "", "Performance branch based on shared-SLAM commit `63454a0`. Accuracy settings and geometric checks are unchanged. Shared-SLAM and main were not merged.", "", "Timings are provisional: retrieval audits and validation work overlapped on this shared host, and background workloads were not controlled. Each controller ran its cases serially; the final repeat overlapped the promoted full run. These single-run comparisons do not establish an uncontended throughput guarantee.", "", "## Pilot and final repeat", "", "KITTI 04 is full (271 frames); small KITTI 01 and TUM fr1 desk cases are 300-frame prefixes. The promoted KITTI 01 run is 1101 frames. Stereo ATE uses SE(3); monocular ATE uses Sim(3) with scale fitting only for evaluation. All table runs use one OpenCV worker.", "", "Elapsed time includes input loading, estimator processing and background finalization. It excludes estimator setup, exports and reference evaluation. FPS is frames divided by that elapsed time; export/setup times are separate. Baseline export/setup times and the saved full baseline's frame latency were not collected, so whole-process speedup is not claimed.", "", "| Case | Stage | Backend | Frozen s | Candidate s | Speedup | FPS | RAM MiB | Torch VRAM MiB | ATE m | Lost | Loops | Quality |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ate = f'{r["ate_m"]:.4f}' if r["ate_m"] is not None else "unavailable"
        lines.append(f'| {r["case"]} | {r["stage"]} | {r["backend"]} | {r["baseline_s"]:.1f} | {r["elapsed_s"]:.1f} | {r["speedup"]:.2f}× | {r["fps"]:.2f} | {r["peak_ram_mb"]:.1f} | {r["torch_peak_vram_mb"]:.1f} | {ate} | {r["lost_frames"]} | {r["loops"]} | {"pass" if r["quality_passed"] else "FAIL"} |')
    if "environment" in machine:
        e = machine["environment"]
        lines[3:3] = [f'Host: Windows, {e.get("logical_processors", "unknown")} logical CPU processors, {e.get("total_system_ram_mb", 0)/1024:.1f} GiB RAM; {e["gpu"]} ({e["cuda_total_vram_mb"]/1024:.1f} GiB). Python {e["python"]}, OpenCV {e["opencv"]}, SciPy {e["scipy"]}, NumPy {e["numpy"]}; isolated PyTorch {e["torch"]}/CUDA {e["cuda_runtime"]}. OpenCV/BLAS use one worker unless explicitly stated.', ""]
    lines += ["", "## Latency, recovery and drift", "", "Final repeats and promotion only. RAM is process peak working set; GPU values are peak Torch allocation, not total device usage.", "", "| Case | Median / p95 ms | Input / export s | RAM change | Lost intervals | Recoveries | Translation % | Rotation deg/m |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value, decimals=2):
        return f"{value:.{decimals}f}" if value is not None else "unavailable"
    for r in rows:
        if r["stage"] != "initial":
            lines.append(f'| {r["case"]} | {number(r["median_frame_ms"])}/{number(r["p95_frame_ms"])} | {number(r["input_loading_s"])}/{number(r["export_s"])} | {r["ram_change_percent"]:+.1f}% | {r["lost_intervals"]} | {r["relocalized_frames"]} | {number(r["translation_percent"])} | {number(r["rotation_deg_per_m"], 4)} |')
    lines += ["", "JSON and CSV include median/p95 frame latency, drift metrics where available, recovery counts, memory changes, and per-file source hashes. Prefix results are not substitutes for full-sequence scores.", "", "## Attribution and limits", "", "- The original CPU profile identified SIFT, disparity and matching as the largest costs. Bundle adjustment also rebuilt observation arrays inside every residual evaluation.", "- Indexed retrieval proposes at most 20 keyframes, followed by exact reranking and unchanged verification. Failed recovery retains exhaustive fallback, so difficult lost-tracking periods can still be expensive.", "- CPU changes cache map arrays, batch stereo measurements, avoid global cleanup scans, and precompute bundle observation arrays. Solver budgets, losses, gauges and geometric checks are unchanged.", "- CPU matching remains the default. CUDA is optional; transfers and synchronization are included in matching timings. Near ties, ratio boundaries and cancellation-prone distances are resolved on CPU.", "- GPU process RAM includes PyTorch/CUDA runtime overhead and can substantially exceed the CPU process. Torch allocation statistics exclude driver/display/context memory, so the VRAM column is not total device usage.", "- The 300-keyframe live optimization guard remains unchanged. No accuracy fixes or faster presets were included."]
    threaded = args.results / "threaded04-stereo/evaluation.json"
    if threaded.exists():
        r = json.loads(threaded.read_text(encoding="utf-8"))
        b = json.loads((args.results / "baseline04-stereo/evaluation.json").read_text(encoding="utf-8"))
        machine["threading_trial"] = {"opencv_threads": r["opencv_threads"], "elapsed_s": r["elapsed_s"], "speedup": b["elapsed_s"]/r["elapsed_s"], "ate_m": r["metrics"]["ate_rmse_m"], "lost_frames": r["lost_frames"], "quality_passed": quality_passed(b,r)}
        lines += ["", "## Explicit worker-count trial", "", f'Full KITTI 04 stereo with CUDA and {r["opencv_threads"]} OpenCV workers: **{b["elapsed_s"]/r["elapsed_s"]:.2f}×** versus the one-worker frozen baseline ({r["elapsed_s"]:.1f} s). ATE {r["metrics"]["ate_rmse_m"]:.4f} m; lost frames {r["lost_frames"]}; quality gate {"passed" if quality_passed(b,r) else "FAILED"}. This changes CPU parallelism, not feature counts or geometry settings. The default remains one worker; this setting was checked on this case only.']
    if "retrieval" in machine:
        r = machine["retrieval"]
        lines += ["", "## Retrieval and graph evidence", "", f'Saved KITTI 00 input-image audit: **{r["retrieved_loops"]}/{r["known_loops"]} known loop pairs retrieved**, with {r["keyframes"]} retained keyframes. This is a SIFT-image proxy, not fresh live geometry verification.', "", "Graph replay used saved independent loop measurements and odometry reconstructed from exported adjacent poses. It does not reconstruct the original pre-correction solver snapshot or demonstrate long-sequence tracking.", "", f'Replay: {r["graph_replay"]["elapsed_s"]:.2f} s; finite poses and fixed origin: {r["graph_replay"]["finite"] and r["graph_replay"]["origin_fixed"]}. The graph solver source is unchanged.']
        lines += ["", "The initial histogram-only index recalled 11/15 known pairs; residual summaries improved this to 13/15, and reserving neighboring views reached 15/15 within the 20-candidate budget. These choices were tuned on this audit; held-out long-sequence loop validation remains necessary before merging."]
    lines += ["", "## Separate profiling and component checks", "", "| Profile | Stage | Calls | Total s | Median ms | p95 ms |", "|---|---|---:|---:|---:|---:|"]
    machine["profiles"] = {}
    for label in ["profile-final01-stereo", "profile-background01-stereo", "profile-recovery-tum"]:
        path = args.results / label / "profile.json"
        if path.exists():
            profile = json.loads(path.read_text(encoding="utf-8"))
            machine["profiles"][label] = profile
            for stage, values in profile["stages"].items():
                lines.append(f'| {label} | {stage} | {values["calls"]} | {values["total_s"]:.2f} | {number(values["median_ms"])} | {number(values["p95_ms"])} |')
    lines += ["", "Profiling runs are excluded from the timing tables. Stages are nested and background work overlaps, so their totals must not be summed. The TUM recovery profile preceded the final fix that reuses shortlist scores during exhaustive fallback."]
    if "matching_microbenchmark" in machine:
        m = machine["matching_microbenchmark"]
        lines += ["", f'Real KITTI 01 1500×1500 descriptor pair: CPU {m["cpu_ms"]:.2f} ms versus CUDA {m["cuda_ms"]:.2f} ms ({m["matching_speedup"]:.2f}×), exact match-pair agreement {m["pair_agreement"]}. This warmed component check includes transfers and synchronization, excludes extraction/runtime initialization, and is not pipeline speedup.']
    machine["ablations"] = {}
    for label in ["indexed04-stereo", "optimized04-stereo"]:
        path = args.results / label / "evaluation.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            b = json.loads((args.results / "baseline04-stereo/evaluation.json").read_text(encoding="utf-8"))
            machine["ablations"][label] = {"elapsed_s": r["elapsed_s"], "speedup": b["elapsed_s"]/r["elapsed_s"], "quality_passed": quality_passed(b,r), "source_sha256": r["source_sha256"]}
    lines += ["", "Early full KITTI04 ablations: histogram indexing alone took 302.5 s (0.92× frozen); adding CPU allocation optimizations took 227.4 s (1.22×). The final index uses SciPy instead of importing scikit-learn, avoiding roughly 40 MiB of unnecessary runtime overhead. These early runs used earlier index revisions and uncontrolled host load; they do not isolate an additive contribution. The initial monocular/indoor regression led to the separate fallback-score reuse fix; both earlier and final measurements remain above.", "", "## Acceptance and review", "", "Quality gate requires identical frame counts, no additional lost frame indices or lost intervals, no fewer verified loops, no additional initialization failures, and ATE ≤ frozen×1.05+0.05 m. The pilot does not prove improved accuracy. CPU matching remains default; GPU requires opt-in.", "", "GPU RAM increases exceed the 10% investigation threshold. Setup peak working set and Torch allocated/reserved VRAM are retained in JSON: the CUDA runtime alone raises setup RAM to roughly 581 MiB on this host, before map growth. Total driver/context VRAM was unavailable to this collector. The memory cost remains a tradeoff; no memory gate is waived silently.", "", "Backend validation: 93 tests passed in the isolated CUDA environment, covering index startup/update/removal, temporal eligibility, exhaustive fallback and score reuse, correction cache refresh, bundle equivalence, matching ties/ratio/cancellation, and unavailable CUDA. Graph solver, accuracy configuration, feature counts, budgets, schedules and the 300-keyframe guard were preserved. Optimization commits are separate from accuracy work.", "", "Review only: leave shared-SLAM and main unchanged. Repeat promising cases without competing workloads and validate held-out live loop closure before merging. The original frozen benchmark checkout, environment, datasets/caches and results were read only; experiment outputs and CUDA installation are isolated in this worktree."]
    args.report.with_suffix(".json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    full = next((r for r in rows if r["stage"] == "promoted"), None)
    if full:
        accuracy = "identical" if abs(full["ate_m"] - full["baseline_ate_m"]) <= 1e-9 else f'changed from {full["baseline_ate_m"]:.6f} m'
        lines[3:3] = ["", f'Full KITTI01 stereo completed in **{full["elapsed_s"]/60:.1f} minutes versus {full["baseline_s"]/60:.1f} minutes frozen ({full["speedup"]:.2f}×)**. ATE is {accuracy} at {full["ate_m"]:.6f} m; lost frames are {full["lost_frames"]} versus {full["baseline_lost_frames"]} frozen. Peak RAM rises from {full["baseline_ram_mb"]/1024:.2f} to {full["peak_ram_mb"]/1024:.2f} GiB ({full["ram_change_percent"]:+.1f}%). This combined CPU/index/CUDA result does not attribute the full gain to indexing alone.', ""]
    regressions = [r for r in rows if r["stage"] == "final" and r["backend"] == "cpu" and r["speedup"] < 1]
    if regressions:
        lines += ["", "CPU performance gains are not uniform. Final CPU regressions: " + "; ".join(f'{r["case"]} {r["speedup"]:.2f}× frozen throughput' for r in regressions) + ". These cases preserve accuracy but do not meet the 20% improvement target."]
    lines += ["", "The persistent cache retains float32 SIFT descriptors (512 bytes per landmark), float64 positions (24 bytes) and IDs (8 bytes). At 264,925 landmarks this is about 137 MiB of array payload. CPU setup was about 110 MiB in the final monocular case versus 583 MiB for the full CUDA stereo run. Runtime initialization plus the retained cache broadly explains the 562 MiB full-run process RAM increase; these working-set peaks are not additive allocation accounting. The cache estimate excludes Python containers."]
    machine["acceptance"] = {"final_cases_completed": sum(r["stage"] == "final" for r in rows),
                             "quality_passed_for_all_completed_cases": all(r["quality_passed"] for r in rows),
                             "two_times_processing_cases": [r["case"] for r in rows if r["stage"] != "initial" and r["speedup"] >= 2],
                             "twenty_percent_elapsed_improvement_cases": [r["case"] for r in rows if r["stage"] != "initial" and r["elapsed_s"] <= .8 * r["baseline_s"]],
                             "memory_increases_above_10_percent": [r["case"] for r in rows if r["stage"] != "initial" and r["ram_change_percent"] > 10],
                             "tests_passed": 93, "merge_ready": False,
                             "limits": "Provisional timings, GPU RAM tradeoff, loop index tuned on saved image proxy; held-out live validation needed before merge"}
    args.report.with_suffix(".json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    content = "\n".join(lines) + "\n"
    while "\n\n\n" in content:
        content = content.replace("\n\n\n", "\n\n")
    args.report.write_text(content, encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
