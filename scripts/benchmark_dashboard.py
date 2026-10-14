"""Read-only dashboard bridge for a shared-SLAM benchmark batch."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from telemetry import TelemetryServer, TelemetryState


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return (
                bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                and code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def retain_active_evaluator_provenance(root):
    """Correct shutdown-time hashing for workers loaded before telemetry was added."""
    manifest_path = root / "evaluator-provenance.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    source = (root / manifest["snapshot"]).read_bytes()
    import hashlib

    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != manifest["actual_evaluator_sha256"]:
        raise ValueError("Preserved evaluator snapshot hash mismatch")
    for row in manifest["loaded_before_edit"]:
        folder = root / row["output"]
        report_path = folder / "evaluation.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        if report.get("evaluator_provenance_note"):
            continue
        report["evaluator_sha256_reported_at_shutdown"] = report.get("evaluator_sha256")
        report["evaluator_sha256"] = actual_hash
        report["evaluator_provenance_note"] = (
            "Evaluator bytes preserved before adding dashboard telemetry while this worker was running. Shutdown-time file hashing was replaced with the hash of the script actually loaded; estimator source and numerical results are unchanged."
        )
        report["telemetry"] = {"enabled": False}
        (folder / "evaluator.py").write_bytes(source)
        temporary = report_path.with_suffix(".part")
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(report_path)


def batch_snapshot(current_path, runner_alive):
    current = read_json(current_path)
    root = (REPO / current["batch_root"]).resolve()
    if not root.is_relative_to(REPO / "results"):
        raise ValueError("Batch must be inside workspace results")
    batch = read_json(root / "batch.json")
    retain_active_evaluator_provenance(root)
    runs = batch["runs"]
    active = next((r for r in reversed(runs) if r["status"] == "running"), None)
    progress = None
    if active:
        log = root / f'{active["sequence"]}-{active["mode"]}.log'
        if log.exists():
            checkpoints = re.findall(
                r"^\S+ (\d+)/(\d+) (\w+) landmarks=(\d+)$",
                log.read_text(encoding="utf-8", errors="replace"),
                re.M,
            )
            if checkpoints:
                frame, total, state, landmarks = checkpoints[-1]
                progress = dict(
                    frames=int(frame),
                    total_frames=int(total),
                    state=state,
                    landmarks=int(landmarks),
                    updated_at=log.stat().st_mtime,
                )
    rows = []
    for sequence in current["sequence_order"]:
        for mode in current["modes"]:
            row = next(
                (
                    r
                    for r in reversed(runs)
                    if r["sequence"] == sequence and r["mode"] == mode
                ),
                None,
            )
            item = dict(
                sequence=sequence,
                sensor=mode,
                status=row["status"] if row else "queued",
            )
            if row and row["status"] in ("complete", "existing_report"):
                report = read_json(root / row["output"] / "evaluation.json")
                metrics = report.get("metrics", {})
                item.update(
                    frames=report["frames"],
                    lost_frames=report["lost_frames"],
                    ate_rmse_m=metrics.get("ate_rmse_m"),
                    alignment="SE(3)" if mode == "stereo" else "Sim(3)",
                    translation_percent=metrics.get("translation_percent"),
                )
            rows.append(item)
    message = dict(
        schema_version=1,
        kind="benchmark_status",
        timestamp=time.time(),
        running=runner_alive and active is not None,
        paused=bool(batch.get("paused")),
        finished=bool(batch.get("finished_utc")),
        completed=sum(r["status"] in ("complete", "existing_report") for r in rows),
        total=len(rows),
        rows=rows,
        active=(
            dict(
                sequence=active["sequence"],
                sensor=active["mode"],
                run_id=active["output"],
                progress=progress,
            )
            if active
            else None
        ),
        stream_available=False,
    )
    feed = None
    feed_path = root / "dashboard-frame.json"
    displayed_run = active or (runs[-1] if (batch.get("finished_utc") or batch.get("paused")) and runs else None)
    if feed_path.exists() and displayed_run:
        candidate = read_json(feed_path)
        if (
            candidate.get("run_id") == displayed_run["output"]
            and candidate.get("schema_version") == 1
        ):
            feed = candidate
            message["stream_available"] = True
    return message, feed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO / "results/shared-kitti-comparison/current.json",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = TelemetryServer(
        "127.0.0.1", args.port, TelemetryState(mode="slam", mode_locked=True)
    )
    server.start()
    print(f"Benchmark dashboard listening on ws://localhost:{args.port}", flush=True)
    try:
        while True:
            try:
                pid_path = args.current.parent / "runner.pid"
                alive = (
                    process_alive(int(pid_path.read_text().strip()))
                    if pid_path.exists()
                    else False
                )
                status, frame = batch_snapshot(args.current, alive)
                server.publish(status)
                if frame:
                    server.publish(frame)
            except (OSError, ValueError, KeyError) as error:
                print(f"Waiting for readable batch state: {error}", flush=True)
            time.sleep(3)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
