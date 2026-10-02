"""Run fixed-configuration stereo/monocular KITTI tests with owned bounded caches."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from download_kitti_sequence import RangeFile, URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["00", "01", "04", "07", "02", "03", "05", "06", "08", "09", "10"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["stereo", "mono"], default=["stereo", "mono"]
    )
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Read existing KITTI sequences without downloading or deleting images",
    )
    parser.add_argument("--output", type=Path, default=Path("results/shared-benchmark"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path(".datasets/shared-benchmark-scratch")
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if any(s not in [f"{i:02d}" for i in range(11)] for s in args.sequences):
        parser.error("Evaluation sequences must be 00–10")
    repo = Path(__file__).resolve().parents[1]
    sources = [p.name for p in sorted((repo / "src").glob("*.py"))]
    fingerprint = hashlib.sha256(
        b"".join((repo / "src" / name).read_bytes() for name in sources)
    ).hexdigest()
    coverage = "full" if args.max_frames is None else f"prefix{args.max_frames}"
    root = args.output / fingerprint[:12] / coverage
    root.mkdir(parents=True, exist_ok=True)
    if args.data_root is None:
        args.cache_root.mkdir(parents=True, exist_ok=True)
    status = {
        "source_sha256": fingerprint,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "configuration_policy": "One fixed MappingConfig per sensor mode; no sequence overrides",
        "coverage": coverage,
        "input_source": "local_images" if args.data_root else "official_remote_archive",
    }

    def save():
        pending = root / "batch.json.part"
        pending.write_text(json.dumps(status, indent=2), encoding="utf-8")
        pending.replace(root / "batch.json")

    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    save()
    for sequence in args.sequences:
        inputs = (
            nullcontext(args.data_root)
            if args.data_root is not None
            else tempfile.TemporaryDirectory(prefix=f"{sequence}-", dir=args.cache_root)
        )
        with inputs as temporary:
            cache = Path(temporary).resolve()
            local = args.data_root is not None
            if not local:
                (cache / "owner.json").write_text(
                    json.dumps(
                        {
                            "purpose": "temporary shared SLAM benchmark images",
                            "sequence": sequence,
                            "source": URL,
                            "process": os.getpid(),
                        }
                    )
                )
                with RangeFile(URL) as remote, zipfile.ZipFile(remote) as archive:
                    prefix = f"dataset/sequences/{sequence}/"
                    entries = [
                        e
                        for e in archive.infolist()
                        if e.filename.startswith(prefix)
                        and (
                            e.filename.endswith(".png")
                            or Path(e.filename).name in ("calib.txt", "times.txt")
                        )
                    ]
                    if args.max_frames:
                        entries = [
                            e
                            for e in entries
                            if not e.filename.endswith(".png")
                            or int(Path(e.filename).stem) < args.max_frames
                        ]
                    required = sum(e.file_size for e in entries)
                    local = shutil.disk_usage(cache).free > required + 750 * 1024**2
                    if local:
                        for number, entry in enumerate(
                            sorted(entries, key=lambda e: e.header_offset)
                        ):
                            relative = Path(*Path(entry.filename).parts[1:])
                            target = (cache / relative).resolve()
                            if not target.is_relative_to(cache):
                                raise ValueError("Archive member escapes owned cache")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            remote.prefetch(
                                entry.header_offset,
                                entry.compress_size
                                + len(entry.filename.encode("utf-8"))
                                + 128,
                            )
                            target.write_bytes(archive.read(entry))
                            if number % 200 == 0:
                                print(
                                    f"{sequence}: cached {number+1}/{len(entries)} files",
                                    flush=True,
                                )
            for mode in args.modes:
                current = hashlib.sha256(
                    b"".join((repo / "src" / name).read_bytes() for name in sources)
                ).hexdigest()
                if current != fingerprint:
                    raise RuntimeError(
                        "Estimator source changed; start a new benchmark revision"
                    )
                output = root / f"kitti{sequence}-{mode}"
                if (output / "evaluation.json").exists():
                    existing = json.loads((output / "evaluation.json").read_text())
                    expected_coverage = "partial" if args.max_frames else "full"
                    if (
                        existing["coverage"] != expected_coverage
                        or existing["status"].startswith("interrupted")
                        or any(existing.get("diagnostic_overrides", {}).values())
                    ):
                        raise RuntimeError(
                            "Existing report is incomplete or diagnostic; use a fresh "
                            "--output directory to retain it and run a new evaluation"
                        )
                    status["runs"].append(
                        {
                            "sequence": sequence,
                            "mode": mode,
                            "status": "existing_report",
                            "output": output.name,
                        }
                    )
                    save()
                    continue
                command = [
                    sys.executable,
                    str(repo / "scripts/evaluate_shared_slam.py"),
                    "--sequence",
                    sequence,
                    "--poses-root",
                    str(args.poses_root),
                    "--output",
                    str(output),
                ]
                command += ["--data-root", str(cache)] if local else ["--remote"]
                if mode == "stereo":
                    command += ["--stereo"]
                if args.max_frames:
                    command += ["--max-frames", str(args.max_frames)]
                row = {
                    "sequence": sequence,
                    "mode": mode,
                    "status": "running",
                    "output": output.name,
                }
                status["runs"].append(row)
                save()
                with (root / f"{sequence}-{mode}.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    result = subprocess.run(
                        command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT
                    )
                row["exit_code"] = result.returncode
                row["status"] = "complete" if result.returncode == 0 else "failed"
                if result.returncode == 0:
                    evaluation = json.loads((output / "evaluation.json").read_text())
                    row["frames"] = evaluation["frames"]
                    row["coverage"] = evaluation["coverage"]
                    row["evaluation_status"] = evaluation["status"]
                    if evaluation["status"] == "interrupted_low_disk_space":
                        row["status"] = "interrupted_low_disk_space"
                save()
                print(f'{sequence} {mode}: {row["status"]}', flush=True)
                if row["status"] == "interrupted_low_disk_space":
                    raise SystemExit("Benchmark stopped: export reserve reached")
                if result.returncode == 0:
                    subprocess.run(
                        [
                            sys.executable,
                            str(repo / "scripts/plot_shared_slam.py"),
                            "--run",
                            str(output),
                            "--reference",
                            str(args.poses_root / f"{sequence}.txt"),
                        ],
                        env=env,
                        cwd=repo,
                        check=False,
                    )
        # Only owned TemporaryDirectory inputs are removed; local datasets remain intact.
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(root / "batch.json")


if __name__ == "__main__":
    main()
