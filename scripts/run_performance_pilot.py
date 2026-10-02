"""Small serial frozen/candidate pilot; never modifies the benchmark checkout."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def quality_passed(baseline, candidate):
    a, b = baseline.get("metrics", {}).get("ate_rmse_m"), candidate.get("metrics", {}).get("ate_rmse_m")
    def lost_indices(report):
        return {i for interval in report.get("lost_intervals", [])
                for i in range(interval["first_frame"], interval["last_frame"] + 1)}
    return (
        candidate["frames"] == baseline["frames"]
        and candidate["lost_frames"] <= baseline["lost_frames"]
        and candidate["initializing_frames"] <= baseline["initializing_frames"]
        and len(candidate.get("lost_intervals", [])) <= len(baseline.get("lost_intervals", []))
        and lost_indices(candidate) <= lost_indices(baseline)
        and candidate["loops"] >= baseline["loops"]
        and a is not None and b is not None and b <= a * 1.05 + .05
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/performance"))
    parser.add_argument("--gpu-python", type=Path)
    parser.add_argument("--final-only", action="store_true", help="Repeat the five small cases against the current source; keeps earlier reports")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    dev = args.development_root.resolve()
    frozen, candidate = output / "frozen", output / "candidate"
    poses = dev / "results/benchmark-batch/reference/poses"
    kitti01 = dev / ".datasets/shared-kitti"
    kitti04 = repo / ".datasets/performance-kitti"
    desk = dev / ".datasets/tum/rgbd_dataset_freiburg1_desk"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MPLCONFIGDIR": str(repo / ".mpl-cache")}
    manifest_path = output / ("final-pilot.json" if args.final_only else "pilot.json")
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "timing_provisional": True,
                "timing_note": "Shared host; background workloads were not controlled and other validation overlapped. Runs in this controller are serial.", "runs": []}

    def save():
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    def run(name, root, dataset, data, sequence, stereo=False, limit=None, gpu=False, profile=False):
        target = output / name
        row = {"name": name, "status": "running", "revision_root": str(root)}
        manifest["runs"].append(row)
        save()
        if (target / "evaluation.json").exists():
            existing = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "src").glob("*.py")}
            if (existing.get("source_sha256") != hashes or existing.get("dataset") != dataset or
                    existing.get("sequence") != sequence or existing.get("stereo") != stereo or
                    (limit is not None and existing.get("frames") != limit) or
                    (root != frozen and existing.get("performance_configuration", {}).get("matching_backend") != ("cuda" if gpu else "cpu"))):
                raise RuntimeError(f"Existing {name} belongs to different inputs/source/options; use a fresh output directory")
            row["status"] = "existing_report"
            save()
            return existing
        interpreter = str(args.gpu_python.resolve()) if gpu else sys.executable
        command = [interpreter]
        if root == frozen:
            command += [str(repo / "scripts/performance_case.py"), "--revision-root", str(root)]
        else:
            command += [str(root / "scripts/evaluate_shared_slam.py")]
        command += ["--dataset", dataset, "--data-root", str(data), "--poses-root", str(poses if dataset == "kitti" else data), "--sequence", sequence, "--output", str(target)]
        if stereo:
            command += ["--stereo"]
        if limit:
            command += ["--max-frames", str(limit)]
        if gpu:
            command += ["--matching-backend", "cuda"]
        if profile:
            command += ["--profile", str(target / "profile.json")]
        row["command"] = command
        print(name, flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        row["exit_code"] = result.returncode
        row["status"] = "complete" if result.returncode == 0 else "failed"
        save()
        if result.returncode:
            raise RuntimeError(f"Pilot {name} failed; see its separate log")
        return json.loads((target / "evaluation.json").read_text(encoding="utf-8"))

    cases = [
        ("04-stereo", "kitti", kitti04, "04", True, None),
        ("04-mono", "kitti", kitti04, "04", False, None),
        ("01-stereo", "kitti", kitti01, "01", True, 300),
        ("01-mono", "kitti", kitti01, "01", False, 300),
        ("tum-desk", "tum", desk, "fr1-desk", False, 300),
    ]
    reports = {}
    if args.final_only:
        for label, dataset, data, sequence, stereo, limit in cases:
            run("final" + label, repo, dataset, data, sequence, stereo, limit,
                gpu=bool(args.gpu_python and stereo))
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return
    for label, dataset, data, sequence, stereo, limit in cases:
        baseline = run("baseline" + label, frozen, dataset, data, sequence, stereo, limit)
        optimized = run("cpu" + label, candidate, dataset, data, sequence, stereo, limit)
        reports[label] = (baseline, optimized)
        if args.gpu_python and stereo:
            gpu = run("gpu" + label, candidate, dataset, data, sequence, stereo, limit, gpu=True)
            reports[label + "-gpu"] = (baseline, gpu)
    run("profile-final01-stereo", candidate, "kitti", kitti01, "01", True, 100, profile=True)
    gpu_checks = [reports.get(label + "-gpu") for label in ("04-stereo", "01-stereo")]
    if args.gpu_python and all(pair and quality_passed(*pair) and pair[1]["elapsed_s"] <= .8 * pair[0]["elapsed_s"] for pair in gpu_checks):
        run("promoted01-stereo", candidate, "kitti", kitti01, "01", True, gpu=True)
        manifest["promotion"] = "Full KITTI 01 stereo after two passing GPU pilots with >=20% lower processing elapsed time"
    else:
        manifest["promotion"] = "Not promoted: two passing stereo pilots with >=20% lower processing elapsed time required"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
