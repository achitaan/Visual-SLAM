"""Refresh current paired coverage from completed reports of one frozen revision."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
COUNTS = dict(
    zip(
        [f"{i:02d}" for i in range(11)],
        [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201],
    )
)


def refresh(current_path, destination):
    current = json.loads(current_path.read_text(encoding="utf-8"))
    root = REPO / current["batch_root"]
    batch = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    source = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((REPO / "src").glob("*.py"))
    }
    fingerprint = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((REPO / "src").glob("*.py")))
    ).hexdigest()
    if fingerprint != current["source_fingerprint"]:
        raise ValueError("Current estimator differs from selected frozen batch")
    rows = []
    completed = []
    for sequence in sorted(COUNTS):
        for sensor in current["modes"]:
            saved = next(
                (
                    r
                    for r in reversed(batch["runs"])
                    if r["sequence"] == sequence and r["mode"] == sensor
                ),
                None,
            )
            row = dict(
                sequence=sequence,
                sensor=sensor,
                expected_frames=COUNTS[sequence],
                status=saved["status"] if saved else "queued",
                frames=None,
                ate_m=None,
                alignment="se3" if sensor == "stereo" else "sim3",
                alignment_scale=None,
                translation_percent=None,
                lost_frames=None,
                verified_loops=None,
                source_fingerprint=fingerprint,
            )
            if saved and saved["status"] in ("complete", "existing_report"):
                report = json.loads(
                    (root / saved["output"] / "evaluation.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    report["source_sha256"] != source
                    or report["coverage"] != "full"
                    or report["frames"] != COUNTS[sequence]
                    or report["stereo"] != (sensor == "stereo")
                ):
                    raise ValueError(
                        f"Invalid full coverage or provenance: {sequence} {sensor}"
                    )
                metrics = report.get("metrics", {})
                row.update(
                    status="complete",
                    frames=report["frames"],
                    ate_m=metrics.get("ate_rmse_m"),
                    alignment_scale=metrics.get("alignment_scale"),
                    translation_percent=metrics.get("translation_percent"),
                    lost_frames=report["lost_frames"],
                    verified_loops=report["loops"],
                )
                completed.append(
                    {
                        **row,
                        "evaluation_status": report["status"],
                        "initialization_frame": report["initialization_frame"],
                        "lost_intervals": report["lost_intervals"],
                        "elapsed_s": report["elapsed_s"],
                        "peak_memory_mb": report["peak_memory_mb"],
                        "median_tracking_reprojection_px": report.get(
                            "median_tracking_reprojection_px"
                        ),
                        "telemetry": report.get("telemetry", {"enabled": False}),
                    }
                )
            rows.append(row)
    snapshot = dict(
        scope="Current shared pipeline, full paired KITTI 00-10 runs",
        source_fingerprint=fingerprint,
        completed_sensor_runs=len(completed),
        requested_sensor_runs=len(rows),
        rows=rows,
        completed_details=completed,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kitti-shared-paired-status.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (destination / "kitti-shared-paired-status.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = destination / "SENSOR_COMPARISON.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        text = re.sub(
            r"coverage snapshot was written with \d+ of 22 sensor runs completed\.",
            f"coverage snapshot was written with {len(completed)} of 22 sensor runs completed.",
            text,
        )
        label = {
            "complete": "Complete",
            "running": "Running",
            "queued": "Queued",
            "failed": "Input/run failure",
        }
        for sequence, count in COUNTS.items():
            modes = [
                next(
                    r for r in rows if r["sequence"] == sequence and r["sensor"] == mode
                )
                for mode in ("stereo", "mono")
            ]
            line = f'| {sequence} | {count:,} | {label.get(modes[0]["status"], modes[0]["status"])} | {label.get(modes[1]["status"], modes[1]["status"])} |'
            text = re.sub(
                rf"^\| {sequence} \| [\d,]+ \| (?:Complete|Running|Queued|Input/run failure) \| (?:Complete|Running|Queued|Input/run failure) \|$",
                line,
                text,
                flags=re.M,
            )
        report_path.write_text(text, encoding="utf-8")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO / "results/shared-kitti-comparison/current.json",
    )
    parser.add_argument("--destination", type=Path, default=REPO / "docs/benchmark")
    args = parser.parse_args()
    snapshot = refresh(args.current, args.destination)
    print(
        f'{snapshot["completed_sensor_runs"]}/{snapshot["requested_sensor_runs"]} complete sensor runs'
    )


if __name__ == "__main__":
    main()
