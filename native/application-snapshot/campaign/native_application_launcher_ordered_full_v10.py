"""Bounded Windows launcher v4 for one isolated serialized-native Stop path.

The browser first receives a control-only hello.  Only then may the launcher
atomically create its WSL child.  Live poses are decoded from the actual Linux
stage over the WSL share; destination exports are never replayed as live data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(r"C:\Users\Achita\Documents\ChatGPT\VSLAM")
B = ROOT / "results/recovery-campaign/20261006T003837Z"
GPU_PYTHON = Path(r"C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe")
WSL_PYTHON = "/root/slam-native-20261009/env/bin/python"
WSL_PREFIX = ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", WSL_PYTHON, "-B"]
RESERVE_BYTES = 1610612736
RUN = re.compile(r"native_prototype_[a-z0-9_]+")
FILES = re.compile(r"[A-Za-z0-9_.-]+")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise ValueError(f"object required: {path}")
    return value


def save_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf8") as stream:
        stream.write(payload)


def save_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError("only the pinned C: workspace may enter WSL")
    return "/mnt/c/" + str(resolved)[3:].replace("\\", "/")


def prepared_protocol(run_id: str, sequence: str, frames: int, seed: int) -> tuple[Path, dict[str, Any]]:
    if not RUN.fullmatch(run_id):
        raise ValueError("scoped lowercase run ID required")
    development = load(B / "native-origin-serialized-development.json")
    review = load(B / "native-component-review18-B1-result.json")
    packet = load(B / "native-component-review18-B1-packet.json")
    if (review.get("verdict_B1") != "clear" or review.get("decision_boundary") != "corrected_binding_assembly_only"
            or development.get("status") != "complete_serialized_two_sequence_screen_pending_saved_verification"):
        raise ValueError("reviewed serialized application binding is unavailable")
    for name, expected in {**packet["sources"], **packet["evidence"]}.items():
        if sha(B / name) != expected:
            raise ValueError("review18 source/evidence identity changed: " + name)
    from native_application_presets_v2 import select_preset
    selected = select_preset(sequence, frames, seed)
    choices = [selected["spec"]]
    wrapper = B / "run_native_serialized_application_case_v5.py"
    candidate = dict(development["candidate_identity"])
    candidate["driver_sha256"] = sha(wrapper)
    protocol = {
        "status": "prepared_native_application_release_v10_not_accuracy_acceptance",
        "utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Isolated serialized-native live Stop qualification; no accuracy, performance, native-rate, retention, or main claim.",
        "active_case": run_id, "runs": {run_id: choices[0]},
        "candidate_identity": candidate,
        "source_manifest_sha256": development["source_manifest_sha256"],
        "launcher_sha256": sha(Path(__file__)),
        "binary_sha256": development["binary_sha256"],
        "libraries_sha256": development["libraries_sha256"],
        "driver_sha256": candidate["driver_sha256"],
        "observer_sha256": development["observer_sha256"],
        "observer_delivery_sha256": development["observer_delivery_sha256"],
        "watchdog_sha256": development["watchdog_sha256"],
        "process_owner_sha256": development["process_owner_sha256"],
        "output_transport_sha256": development["output_transport_sha256"],
        "compact_transport_sha256": development["compact_transport_sha256"],
        "bounded_finalizer_sha256": packet["sources"]["native_application_raw_finalizer.py"],
        "entire_arm_owner_sha256": development["entire_arm_owner_sha256"],
        "bounded_driver_core_sha256": packet["sources"]["native_application_live_driver_core_v2.py"],
        "reserve_bytes": RESERVE_BYTES, "ground_truth_read": False,
        "prepared_preset_provenance": selected,
        "presets_catalog_sha256": sha(B / "native-application-prepared-presets.json"),
        "presets_selector_sha256": sha(B / "native_application_presets_v2.py"),
        "main_changed": False, "retained": False,
        "review18_result_sha256": sha(B / "native-component-review18-B1-result.json"),
        "application_components_sha256": {
            name: sha(B / name) for name in (
            "native_application_admission_v4.py", "native_application_arm_owner_v6.py",
                "run_native_serialized_application_case_v5.py",
                "native_application_stop_state_v4.py", "native_application_control_v4.py", "native_application_control_ordered_v6.py",
                "native_application_follower_v3.py", "native_application_follower_v4.py")}}
    protocol_name = run_id.replace("_", "-") + "-application-release-v10.json"
    path = B / protocol_name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    return path, protocol


def partial_hashes(stage: Path) -> dict[str, Any]:
    if not stage.is_dir():
        return {}
    result = {}
    for path in sorted(stage.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink() or not FILES.fullmatch(path.name):
            raise ValueError("unowned file in native partial stage")
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("partial native source changed during hash")
        result[path.name] = {"bytes": before.st_size, "sha256": digest,
                             "source": str(path)}
    return result


def seal_cleanup(run_id: str, token: str, journal: Path, audit: Path,
                 log_path: Path, *, mode: str = "cancelled") -> dict[str, Any]:
    command = [*WSL_PREFIX, wsl_path(B / "native_application_admission_v4.py"),
               "--seal-terminal", run_id, token, wsl_path(journal), wsl_path(audit), "10", mode]
    with log_path.open("x", encoding="utf8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=15, check=False, text=True)
    if completed.returncode != 0 or not audit.is_file():
        raise RuntimeError("whole owned cleanup seal was not produced")
    value = load(audit)
    if value.get("all_owned_exited") is not True:
        raise RuntimeError("whole owned cleanup remains unconfirmed")
    return value


def run(args) -> dict[str, Any]:
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        raise ValueError("use the pinned GPU Python -B runtime")
    protocol_path, protocol = prepared_protocol(args.run_id, args.sequence,
                                                  args.frames, args.seed)
    spec = protocol["runs"][args.run_id]
    if args.serve_after < 0 or args.serve_after > 300 or args.start_timeout <= 0 or args.start_timeout > 180:
        raise ValueError("bounded readiness/serving interval required")
    forecast = protocol["prepared_preset_provenance"]["output_forecast_bytes"]
    if shutil.disk_usage(ROOT).free < RESERVE_BYTES + forecast:
        raise OSError("insufficient space above preserved 1.5GiB reserve")
    app = B / f"{args.run_id}_application_release_v10"
    journal = B / f"{args.run_id}-admissions-v3.jsonl"
    audit_path = B / f"{args.run_id}-whole-exit-audit-v4.json"
    result_path = app / "result.json"
    for path in (app, journal, audit_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    app.mkdir()
    save_exclusive(protocol_path, protocol)
    sys.path[:0] = [str(B / "native-app-adapter-origin"), str(B),
                    str(B / "retry-bundle-operator-source/src")]
    import numpy as np
    from telemetry import TelemetryState
    from native_application_control_ordered_v6 import NativeApplicationControlServer
    from native_application_control_v4 import NativeApplicationTelemetryServer
    from native_application_follower_v4 import NativeApplicationFollower
    from native_application_stop_state_v4 import ApplicationStopState
    from native_app_telemetry import NativeTelemetry
    from native_live_snapshots_delivery import NativeSnapshotReader
    from native_map_export import export_snapshot
    from native_online_validation import verify_online_records

    state = ApplicationStopState(args.run_id, args.frames)
    token = os.urandom(32).hex()
    control = NativeApplicationControlServer("127.0.0.1", args.control_port, state)
    telemetry_server = NativeApplicationTelemetryServer(
        "127.0.0.1", args.telemetry_port,
        TelemetryState(mode="slam", mode_locked=True))
    result: dict[str, Any] = {
        "schema": "native_application_result_release_v10", "status": "preparing_no_acceptance",
        "run_id": args.run_id, "sequence": args.sequence,
        "expected_original_ids": list(range(args.frames)), "expected_frames": args.frames,
        "seed": args.seed, "ground_truth_read": False, "retained": False,
        "main_changed": False, "accuracy_claim": False, "performance_claim": False,
        "native_rate_claim": False, "protocol": str(protocol_path),
        "protocol_sha256": sha(protocol_path), "admission_journal": str(journal),
        "whole_exit_audit": str(audit_path), "attempt_limit": 1,
        "launcher_sha256": sha(Path(__file__))}
    worker = None
    dispatch_log = None
    follower = None
    snapshots = []
    frames = []
    stop_error = None
    cleanup_attempted = False
    completed_artifacts = None
    completed_pose_sha256 = None
    started = time.monotonic()
    try:
        control.start()
        telemetry_server.start()
        link = (f"http://127.0.0.1:4301/native-stop-v2?run={args.run_id}"
                f"&sequence={args.sequence}&frames={args.frames}"
                f"&ws=ws%3A%2F%2F127.0.0.1%3A{telemetry_server.port}%2F"
                f"&control=ws%3A%2F%2F127.0.0.1%3A{control.port}%2F")
        protocol.update(status="awaiting_control_hello_and_telemetry_client",
                        telemetry_url=f"ws://127.0.0.1:{telemetry_server.port}/",
                        control_url=f"ws://127.0.0.1:{control.port}/", browser_url=link)
        save_replace(protocol_path, protocol)
        print(json.dumps({"status": protocol["status"], "browser_url": link,
                          "protocol": str(protocol_path)}), flush=True)
        ready_deadline = time.monotonic() + args.start_timeout
        while not state.cancel_requested.is_set():
            if control.hello_delivered.is_set() and telemetry_server.client_ready.is_set():
                break
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("control hello/telemetry client readiness deadline")
            time.sleep(.02)
        if state.phase == "cancelled":
            result.update(status="cancelled_before_spawn_incomplete", emitted_original_ids=[],
                          unprocessed_original_ids=list(range(args.frames)), process_created=False,
                          cleanup_confirmed=True, partial_artifacts={})
            control.publish_state()
            return result
        if args.start_delay:
            stop = min(ready_deadline, time.monotonic() + args.start_delay)
            while time.monotonic() < stop and not state.cancel_requested.is_set():
                time.sleep(.01)
        dispatch_log = (app / "dispatch.log").open("x", encoding="utf8")
        command = [*WSL_PREFIX, wsl_path(B / "run_native_serialized_application_case_v5.py"),
                   protocol_path.name, args.run_id, repr(spec["cap_s"]), state.digest, token]

        def create_worker():
            return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=dispatch_log,
                                    stderr=subprocess.STDOUT, bufsize=0)

        worker = state.spawn_owned(create_worker)
        if worker is None:
            result.update(status="cancelled_before_spawn_incomplete", emitted_original_ids=[],
                          unprocessed_original_ids=list(range(args.frames)), process_created=False,
                          cleanup_confirmed=True, partial_artifacts={})
            control.publish_state()
            return result
        # spawn_owned publishes the running transition under the same lock that
        # serializes Stop.  Snapshot immediately so a racing Stop is published
        # as stopping rather than overwriting it with a stale running status.
        published_state = state.snapshot()
        control.publish_state()
        protocol["status"] = (
            "running_native_application" if published_state["phase"] == "running"
            else "stopping_native_application"
        )
        protocol["owned_windows_pid"] = worker.pid
        save_replace(protocol_path, protocol)
        follower = NativeApplicationFollower(
            args.run_id, args.frames, journal=journal,
            cancelled=state.cancel_requested.is_set, timeout_s=spec["cap_s"] + 20)
        expected_snapshots = sorted(set(range(spec["snapshot_interval"], args.frames,
                                              spec["snapshot_interval"])) | {args.frames - 1})
        reader = NativeSnapshotReader(
            follower.directory, expected_snapshots, pending_s=15,
            absolute_deadline=time.monotonic() + spec["cap_s"] + 20,
            budget=follower._budget, process_poll=worker.poll)
        telemetry = NativeTelemetry(sequence=args.sequence, total_frames=args.frames,
                                    run_id=args.run_id)
        with (app / "poses.txt").open("xb") as pose_stream, \
                (app / "published.jsonl").open("x", encoding="utf8") as messages:
            def publish(message):
                messages.write(json.dumps(message, allow_nan=False) + "\n")
                messages.flush()
                telemetry_server.publish(message)

            for frame in follower.frames(worker.poll):
                frames.append(frame)
                pose_stream.write(frame.pose_line)
                pose_stream.flush()
                publish(telemetry.push(frame))
                for snapshot in reader.poll(follower.metadata_rows):
                    snapshots[:] = [snapshot]
                    publish(telemetry.accept_snapshot(snapshot))
            if not state.cancel_requested.is_set():
                for snapshot in reader.poll(follower.metadata_rows, finished=True):
                    snapshots[:] = [snapshot]
                    publish(telemetry.accept_snapshot(snapshot))
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if state.cancel_requested.is_set():
                worker.terminate()
                worker.wait(timeout=3)
                result["owned_windows_wrapper_terminated_after_linux_cleanup_wait"] = True
            else:
                raise
        if worker.stdin is not None:
            worker.stdin.close()
        if state.phase == "stopping":
            try:
                cleanup_attempted = True
                audit = seal_cleanup(args.run_id, token, journal, audit_path,
                                     app / "whole-exit-audit.log")
                state.acknowledge_owned_exit(audit)
                elapsed = time.monotonic() - state.stop_requested_monotonic_s
                result.update(status="user_cancelled_incomplete_not_acceptance",
                              cleanup_confirmed=True, request_to_all_exit_s=elapsed,
                              whole_exit_audit_sha256=sha(audit_path))
            except BaseException as exc:
                stop_error = exc
                result.update(status="stopping_cleanup_unconfirmed", cleanup_confirmed=False,
                              cleanup_error=repr(exc))
            control.publish_state()
        elif worker.returncode == 0 and follower.status == "complete_stream_lifecycle_not_accuracy_acceptance":
            case = load(B / args.run_id / "case.json")
            if case.get("status") != "completed_seeded_native_pending_evaluation":
                raise ValueError("native case did not complete")
            rows = []
            with (follower.directory / "online_frames.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            with (follower.directory / "optimizer_events.csv").open(newline="") as stream:
                events = list(csv.DictReader(stream))
            with (follower.directory / "lifecycle.csv").open(newline="") as stream:
                lifecycle = list(csv.DictReader(stream))
            if len(lifecycle) != 1:
                raise ValueError("one native lifecycle row required")
            proof = verify_online_records(
                rows, events, np.asarray([frame.pose_T_wc for frame in frames])[:, :3, :],
                lifecycle[0], expected_frames=args.frames)
            if snapshots:
                calibration = ROOT / ".datasets/shared-kitti/sequences" / args.sequence / "calib.txt"
                line = next(value for value in calibration.read_text().splitlines()
                            if value.startswith("P0:"))
                matrix = np.asarray([float(value) for value in line.split()[1:]]).reshape(3, 4)[:, :3]
                export_snapshot(app / "map", snapshots[-1], rows,
                                (app / "poses.txt").read_bytes(), matrix)
            # Completion and authenticated Stop race at this single locked
            # commitment, after normal admission closure and whole-tree proof.
            cleanup_attempted = True
            audit = seal_cleanup(args.run_id, token, journal, audit_path,
                                 app / "whole-exit-audit.log", mode="completed")
            # Cleanup proof is independent of processing/evidence acceptance.
            # Perform every required terminal file read before publishing completion.
            result["whole_owned_cleanup_confirmed"] = True
            result["whole_exit_audit_sha256"] = sha(audit_path)
            completed_artifacts = partial_hashes(follower.directory)
            completed_pose_sha256 = sha(app / "poses.txt")
            completed_case_sha256 = sha(B / args.run_id / "case.json")
            if state.mark_complete(audit):
                result.update(status="completed_application_pending_independent_acceptance",
                              cleanup_confirmed=True, completion_proof=proof,
                              native_case_sha256=completed_case_sha256)
            else:
                state.acknowledge_owned_exit(audit)
                result.update(status="user_cancelled_incomplete_not_acceptance",
                              cleanup_confirmed=True,
                              request_to_all_exit_s=time.monotonic() - state.stop_requested_monotonic_s,
                              terminal_race="authenticated_stop_won_after_native_validation")
            control.publish_state()
        else:
            state.mark_failed()
            result.update(status="incomplete_application_no_acceptance",
                          cleanup_confirmed=False, worker_returncode=worker.returncode)
        emitted = [frame.original_id for frame in frames]
        result.update(
            emitted_original_ids=emitted,
            unprocessed_original_ids=list(range(len(emitted), args.frames)),
            emitted_prefix_sha256=(completed_pose_sha256 if completed_pose_sha256 is not None
                                   else sha(app / "poses.txt")),
            follower_status=follower.status,
            causal_observation=follower.causal_observation,
            accepted_snapshot_original_ids=reader.accepted_frames,
            expected_snapshot_original_ids=expected_snapshots,
            stop_state=state.snapshot(), worker_returncode=worker.returncode,
            partial_artifacts=(completed_artifacts if completed_artifacts is not None
                               else partial_hashes(follower.directory)
                               if result.get("cleanup_confirmed") else None))
        if state.cancel_requested.is_set() and emitted != list(range(len(emitted))):
            raise ValueError("cancelled emitted prefix is not exact original-ID coverage")
        if state.cancel_requested.is_set() and result["status"] != "user_cancelled_incomplete_not_acceptance":
            result["incomplete_never_accepted"] = True
        if stop_error is not None:
            raise stop_error
        return result
    except BaseException as exc:
        result["error"] = repr(exc)
        result["error_traceback"] = traceback.format_exc()
        if state.phase == "stopping" and worker is not None and not cleanup_attempted:
            try:
                cleanup_attempted = True
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=3)
                    result["owned_windows_wrapper_terminated_after_linux_cleanup_wait"] = True
                if worker.stdin is not None and not worker.stdin.closed:
                    worker.stdin.close()
                audit = seal_cleanup(args.run_id, token, journal, audit_path,
                                     app / "whole-exit-audit.log")
                state.acknowledge_owned_exit(audit)
                result.update(
                    status="user_cancelled_incomplete_not_acceptance",
                    cleanup_confirmed=True,
                    request_to_all_exit_s=time.monotonic() - state.stop_requested_monotonic_s,
                    whole_exit_audit_sha256=sha(audit_path),
                    prefix_validation_error=repr(exc))
                control.publish_state()
                emitted = [frame.original_id for frame in frames]
                result.update(
                    emitted_original_ids=emitted,
                    unprocessed_original_ids=list(range(len(emitted), args.frames)),
                    stop_state=state.snapshot(), worker_returncode=worker.returncode,
                    partial_artifacts=(partial_hashes(follower.directory)
                                       if follower is not None else {}))
                if follower is not None:
                    result["causal_observation"] = follower.causal_observation
                    result["follower_status"] = follower.status
                poses = app / "poses.txt"
                if poses.exists():
                    result["emitted_prefix_sha256"] = sha(poses)
                return result
            except BaseException as cleanup_exc:
                result.update(status="stopping_cleanup_unconfirmed", cleanup_confirmed=False,
                              cleanup_error=repr(cleanup_exc))
        else:
            state.mark_failed()
            if state.phase == "failed":
                result.update(status="incomplete_application_no_acceptance",
                              cleanup_confirmed=False)
        result["stop_state"] = state.snapshot()
        try:
            control.publish_state()
        except Exception as publish_exc:
            result["failure_state_publication_error"] = repr(publish_exc)
        if follower is not None:
            result["causal_observation"] = follower.causal_observation
            result["emitted_original_ids"] = [frame.original_id for frame in frames]
            result["unprocessed_original_ids"] = list(range(len(frames), args.frames))
        raise
    finally:
        result["wall_s"] = time.monotonic() - started
        result["utc"] = datetime.now(timezone.utc).isoformat()
        if worker is not None and worker.stdin is not None and not worker.stdin.closed:
            worker.stdin.close()
        if dispatch_log is not None:
            dispatch_log.close()
        try:
            save_exclusive(result_path, result)
        except FileExistsError:
            pass
        protocol.update(status=result.get("status", "incomplete_application_no_acceptance"),
                        active_case=None if result.get("cleanup_confirmed") else args.run_id,
                        result=str(result_path), updated_utc=result["utc"])
        save_replace(protocol_path, protocol)
        if args.serve_after:
            end = time.monotonic() + args.serve_after
            while time.monotonic() < end and not state.service_stop.is_set():
                time.sleep(min(.05, end - time.monotonic()))
        try:
            telemetry_server.stop()
        finally:
            control.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=("native_prototype_stop_" +
                        datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()))
    parser.add_argument("--sequence", default="04", choices=("01", "04", "07", "09"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--telemetry-port", type=int, default=0)
    parser.add_argument("--control-port", type=int, default=0)
    parser.add_argument("--start-timeout", type=float, default=120)
    parser.add_argument("--start-delay", type=float, default=.25)
    parser.add_argument("--serve-after", type=float, default=5)
    value = run(parser.parse_args())
    print(json.dumps({key: value.get(key) for key in (
        "status", "run_id", "expected_frames", "emitted_original_ids",
        "cleanup_confirmed", "request_to_all_exit_s")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


