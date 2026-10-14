"""V2 arm driver: every caller operation inherits the outer original deadline."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

from native_compact_output import NativeCompactOutput
from native_finalizer_owned import FinalizationFailure, finalize_owned
from native_bounded_arm_owner_v2 import inherited_arm_context, publish_outer_terminal
from native_observation_delivery import CommittedPrefixObserver, ObservationDeferred
from native_process_owner_guarded import OwnedPausedProcess


B = Path(__file__).resolve().parent
ROOT = B.parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(identity_key: str, build_name: str, build_status: str,
        source_manifest_name: str, driver_path: Path) -> dict:
    outer_context = inherited_arm_context(float(os.environ["NATIVE_ARM_ORIGINAL_CAP_S"]))
    if len(sys.argv) != 3:
        raise ValueError("bounded native arm requires PROTOCOL NAME")
    protocol_name, name = sys.argv[1:]
    if not re.fullmatch(r"[a-z0-9-]+\.json", protocol_name) or not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("unscoped protocol or arm name")
    protocol = json.loads((B / protocol_name).read_text())
    if protocol["active_case"] != name:
        raise ValueError("protocol does not own requested active case")
    protocol.update(protocol[identity_key])
    spec = protocol["runs"][name]
    seq, count, seed, cap = spec["sequence"], spec["frames"], spec["seed"], spec["cap_s"]
    if not re.fullmatch(r"\d\d", seq) or not 0 <= seed <= 2147483647 or count <= 0 or not 0 < cap <= 1800:
        raise ValueError("invalid bounded native arm specification")
    if outer_context["cap_s"] != cap:
        raise ValueError("protocol cap differs from inherited original arm cap")
    for helper, key in (
            ("native_observation.py", "observer_sha256"),
            ("native_observation_delivery.py", "observer_delivery_sha256"),
            ("native_owned_deadline.py", "watchdog_sha256"),
            ("native_process_owner_guarded.py", "process_owner_sha256"),
            ("native_local_output.py", "output_transport_sha256"),
            ("native_compact_output.py", "compact_transport_sha256"),
            ("native_finalizer_owned.py", "bounded_finalizer_sha256"),
            ("native_bounded_arm_owner_v2.py", "entire_arm_owner_sha256"),
            ("native_application_live_driver_core.py", "bounded_driver_core_sha256")):
        if sha(B / helper) != protocol[key]:
            raise ValueError(f"helper identity changed: {helper}")
    if sha(driver_path) != protocol["driver_sha256"]:
        raise ValueError("bounded successor driver identity changed")
    build = json.loads((B / build_name).read_text())
    if build["status"] != build_status:
        raise ValueError("unexpected frozen native build status")
    source = Path(build["source"])
    build_root = source.parent
    environment = Path(build["environment_prefix"])
    build["shared_libraries_sha256"] = protocol["libraries_sha256"]
    if sha(B / source_manifest_name) != protocol["source_manifest_sha256"]:
        raise ValueError("frozen source manifest identity changed")
    manifest = json.loads((B / source_manifest_name).read_text())
    for relative, expected in manifest["files"].items():
        if sha(source / relative) != expected:
            raise ValueError(f"source identity changed: {relative}")
    for relative, expected in build["shared_libraries_sha256"].items():
        if sha(build_root / relative) != expected:
            raise ValueError(f"library identity changed: {relative}")
    binary = source / "bin/native_online_kitti"
    if sha(binary) != protocol["binary_sha256"] or sha(binary) != build["binary_sha256"]:
        raise ValueError("native binary identity changed")
    config = B / "native-online-build/configurations-real" / f"{seq}.yaml"
    if sha(config) != spec["configuration_sha256"]:
        raise ValueError("configuration identity changed")
    sequence = ROOT / ".datasets/shared-kitti/sequences" / seq
    fingerprint = hashlib.sha256()
    input_hashes = {}
    names = ["calib.txt", "times.txt"] + [
        f"image_{camera}/{frame:06d}.png"
        for camera in (0, 1) for frame in range(count)]
    for relative in names:
        data = (sequence / relative).read_bytes()
        input_hashes[relative] = hashlib.sha256(data).hexdigest()
        if relative.startswith("image_"):
            fingerprint.update(relative.removeprefix("image_").encode())
        fingerprint.update(data)
    if fingerprint.hexdigest() != spec["input_sha256"]:
        raise ValueError("input fingerprint changed")
    destination = B / name
    destination.mkdir(exist_ok=False)
    transport = NativeCompactOutput(
        destination, name, physical_root=ROOT,
        reserve_bytes=protocol["reserve_bytes"])
    output = transport.stage
    command = [str(binary), str(build_root / "orbslam3-online/Vocabulary/ORBvoc.txt"),
               str(config), str(sequence), str(output), str(count), str(seed),
               str(spec["snapshot_interval"])]
    environment_values = os.environ.copy()
    environment_values.update(
        LD_LIBRARY_PATH=":".join(map(str, [environment / "lib", build_root / "install/lib",
                                             source / "lib", source / "Thirdparty/DBoW2/lib",
                                             source / "Thirdparty/g2o/lib"])),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    record = {
        "status": "running_seeded_native_arm", "name": name, "command": command,
        "expected_frames": count, "cap_s": cap, "seed": seed,
        "ground_truth_read": False, "input_fingerprint": fingerprint.hexdigest(),
        "input_sha256": input_hashes, "source_manifest_sha256": protocol["source_manifest_sha256"],
        "binary_sha256": sha(binary), "shared_libraries_sha256": build["shared_libraries_sha256"],
        "configuration_sha256": sha(config), "driver_sha256": sha(driver_path),
        "bounded_driver_core_sha256": sha(Path(__file__)), "prefix_snapshots": [],
        "purpose": protocol["purpose"], "observer_sha256": protocol["observer_sha256"],
        "outer_arm": outer_context}

    def save_running() -> None:
        (destination / "case.json").write_text(json.dumps(record, indent=2) + "\n")

    observer = CommittedPrefixObserver()
    guard = None
    process = None
    prefix = b""
    started = outer_context["arm_started_monotonic_s"]
    deadline = outer_context["absolute_deadline_monotonic_s"]
    failure: BaseException | None = None
    save_running()

    def read_delivery(path: Path) -> bytes:
        while True:
            if guard is not None:
                guard.check()
            try:
                return observer.read_prefix(path, deadline=deadline, poll=process.poll)
            except ObservationDeferred:
                record["observation_deferrals"] = observer.budget.deferrals
                time.sleep(.02)

    try:
        with (output / "worker.log").open("w") as log:
            with OwnedPausedProcess(deadline=deadline,
                                    result_path=output / "owned-deadline.json") as owner:
                process, guard = owner.start(command, env=environment_values, stdout=log)
                record["pid"] = process.pid
                save_running()
                owner.release()
                while process.poll() is None:
                    guard.check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native arm original absolute cap")
                    pose_path = output / "online_poses.txt"
                    if pose_path.exists():
                        current = read_delivery(pose_path)
                        if not current.startswith(prefix):
                            raise ValueError("emitted history rewritten")
                        if current != prefix:
                            prefix = current
                            record["prefix_snapshots"].append({
                                "rows": prefix.count(b"\n"),
                                "sha256": hashlib.sha256(prefix).hexdigest(),
                                "observed_s": time.monotonic() - started})
                    time.sleep(.02)
        record["exit_code"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError("native runner failed")
        record["watchdog"] = json.loads((output / "owned-deadline.json").read_text())
        if record["watchdog"]["status"] != "writer_exited_before_deadline" or record["watchdog"]["signals"]:
            raise RuntimeError("native writer deadline proof failed")
        final = read_delivery(output / "online_poses.txt")
        if observer.raw_size != len(final) or not final.startswith(prefix) or final.count(b"\n") != count:
            raise ValueError("final committed pose prefix differs")
        with (output / "online_frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["original_id"]) for row in rows] != list(range(count)):
            raise ValueError("original frame sequence differs")
        with (output / "seed.csv").open() as stream:
            seeds = list(csv.DictReader(stream))
        expected_seed = seed if seed else 4294967295
        if (len(seeds) != 1 or int(seeds[0]["requested_seed"]) != seed
                or int(seeds[0]["native_seed_once"]) != 1
                or int(seeds[0]["opencv_state_before_system"]) != expected_seed):
            raise ValueError("native seed evidence differs")
        for relative, expected in manifest["files"].items():
            if sha(source / relative) != expected:
                raise ValueError(f"source changed during run: {relative}")
        for relative, expected in build["shared_libraries_sha256"].items():
            if sha(build_root / relative) != expected:
                raise ValueError(f"library changed during run: {relative}")
        for relative, expected in input_hashes.items():
            if sha(sequence / relative) != expected:
                raise ValueError(f"input changed during run: {relative}")
        expected_snapshots = set(range(spec["snapshot_interval"], count,
                                       spec["snapshot_interval"])) | {count - 1}
        actual_snapshots = {int(path.stem.split("-")[1])
                            for path in output.glob("snapshot-*.json")}
        if actual_snapshots != expected_snapshots:
            raise ValueError("snapshot set differs")
        record.update(
            status="completed_seeded_native_pending_evaluation",
            snapshots=sorted(expected_snapshots),
            output_hashes={path.name: sha(path) for path in output.iterdir()
                           if path.is_file()})
    except BaseException as exc:
        failure = exc
        record.update(status="incomplete_seeded_native_arm", error=repr(exc))
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    finally:
        if process is not None:
            record["exit_code"] = process.poll()
        record.update(
            observation=observer.statistics(),
            observer_delivery_sha256=protocol["observer_delivery_sha256"],
            watchdog_sha256=protocol["watchdog_sha256"],
            process_owner_sha256=protocol["process_owner_sha256"],
            bounded_finalizer_sha256=protocol["bounded_finalizer_sha256"],
            utc=datetime.now(timezone.utc).isoformat())
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(
            child_peak_rss_kib=usage.ru_maxrss, child_user_s=usage.ru_utime,
            child_system_s=usage.ru_stime,
            child_usage_scope="native_writer_and_pre_finalizer_children",
            unknown_child_usage=None)
        if process is None or process.poll() is None:
            save_running()
        else:
            try:
                completed = record["status"] == "completed_seeded_native_pending_evaluation"
                record = finalize_owned(
                    transport=transport, record=record,
                    arm_started_monotonic_s=started, cap_s=cap,
                    mode="local",
                    native_hashes=None,
                    watchdog_path=B / f"{name}-finalizer-owned-deadline.json",
                    request_path=B / f"{name}-finalizer-request.json",
                    log_path=B / f"{name}-finalizer.log")
            except FinalizationFailure as exc:
                if failure is None:
                    failure = exc
    if failure is not None:
        raise failure
    publish_outer_terminal(outer_context, destination / "case.json")
    print(json.dumps({key: record[key]
                      for key in ("status", "expected_frames", "seed", "wall_s", "exit_code")}),
          flush=True)
    return record


