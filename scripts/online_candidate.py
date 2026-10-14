"""Repository entry for the isolated, reviewed experimental online SLAM presets.

Use --check for a source-bound command plan without native/browser execution.
This frontend is a local integration component, not a replacement for main.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/recovery-campaign/20261006T003837Z"
LOCAL_ROOT = Path(r"C:\Users\Achita\Documents\ChatGPT\VSLAM")
PYTHON = Path(r"C:\Users\Achita\.codex\worktrees\e824\VSLAM\.venv-performance-gpu\Scripts\python.exe")
RETENTION = "native-full01-experimental-prototype-retention29.json"
RETENTION_SHA = "4787863254c99027b471c515f8525c926b9066d14677ea844508f210737c9c23"
PRESETS = {
    "quick04": ("04", 60, 2, "native_application_launcher_release_v7.py"),
    "short04": ("04", 80, 0, "native_application_launcher_release_v7.py"),
    "short01": ("01", 350, 0, "native_application_launcher_release_v7.py"),
    "full01": ("01", 1101, 0, "native_application_launcher_release_v10.py"),
}

def command_tail(command: str) -> str:
    command = command.strip()
    if command.startswith('"'):
        end = command.find('"', 1)
        if end < 0:
            raise ValueError("Malformed process command line; admission unavailable")
        return command[end + 1:].strip().casefold()
    return command.partition(" ")[2].strip().casefold()

def check_worker_observation(windows: list[dict], linux: str, *, own_pid: int, parent_pid: int) -> None:
    own = next((row for row in windows if row["ProcessId"] == own_pid), None)
    if own is None or not own.get("CommandLine"):
        raise ValueError("Current process identity missing; admission unavailable")
    allowed = {own_pid}
    parent = next((row for row in windows if row["ProcessId"] == parent_pid), None)
    if parent and parent.get("CommandLine") and command_tail(parent["CommandLine"]) == command_tail(own["CommandLine"]):
        allowed.add(parent_pid)  # The qualified venv may forward through one identical Python parent.
    campaign = str(BASE).replace("\\", "/").casefold()
    for row in windows:
        if row["ProcessId"] in allowed:
            continue
        if not isinstance(row.get("CommandLine"), str) or not row["CommandLine"].strip():
            raise ValueError(f"Python PID {row['ProcessId']} has no observable command identity; admission unavailable")
        command = row["CommandLine"].replace("\\", "/").casefold()
        if campaign in command or "scripts/online_candidate.py" in command:
            raise ValueError(f"Another campaign process is live (Windows PID {row['ProcessId']}); no child started")
    if not linux.strip():
        raise ValueError("Linux process observation empty; admission unavailable")
    for line in linux.splitlines():
        if ("/bin/native_online_kitti" in line or "/slam-native-20261009/env/bin/python" in line
                or "/results/recovery-campaign/20261006t003837z/" in line.casefold()):
            raise ValueError("Another native/campaign process is live in Ubuntu; no child started")

def assert_runtime_ready() -> dict:
    import os
    query = ("Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\\.exe$'} | "
             "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    raw = subprocess.check_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
                                  text=True, timeout=10).strip()
    windows = json.loads(raw) if raw else []
    if not isinstance(windows, list):
        windows = [windows]
    linux = subprocess.check_output(["wsl.exe", "-d", "Ubuntu", "-u", "root", "--",
                                     "ps", "-eo", "pid=,args="], text=True, timeout=10)
    check_worker_observation(windows, linux, own_pid=os.getpid(), parent_pid=os.getppid())
    for port in (8779, 8780):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    with urlopen("http://127.0.0.1:4301/native-stop-v2", timeout=10) as response:
        if response.status != 200:
            raise ValueError("Isolated dashboard4301 is unavailable; no child started")
    return dict(actual_process_observation_pass=True, backend_ports_available=[8779, 8780],
                isolated_dashboard_http_status=200, observation_not_atomic_scheduler=True)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name: str) -> dict:
    value = json.loads((BASE / name).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected bound record object: " + name)
    return value

def verify_terminal_result(value: dict) -> dict:
    version = "v10" if value["reviewed_inner_launcher"].endswith("v10.py") else "v7"
    path = BASE / (value["run_id"] + "_application_release_" + version) / "result.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Current application terminal result missing or outside bound; cleanup unconfirmed")
    result = json.loads(path.read_text(encoding="utf8"))
    if (result.get("run_id") != value["run_id"] or result.get("expected_frames") != value["frames"]
            or result.get("cleanup_confirmed") is not True):
        raise ValueError("Current application terminal identity/cleanup is unconfirmed")
    emitted = result.get("emitted_original_ids")
    unprocessed = result.get("unprocessed_original_ids")
    if (not isinstance(emitted, list) or not isinstance(unprocessed, list)
            or len(emitted) > value["frames"]
            or any(type(frame) is not int for frame in emitted + unprocessed)
            or emitted != list(range(len(emitted)))
            or unprocessed != list(range(len(emitted), value["frames"]))):
        raise ValueError("Terminal original-ID coverage differs; cancellation is not accepted")
    return result

def run_foreground(value: dict) -> int:
    if sys.platform != "win32":
        raise ValueError("This qualified local entry requires the Windows process-group boundary")
    process = subprocess.Popen(value["command"], cwd=ROOT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        # Windows' indefinite WaitForSingleObject defers Python's SIGINT
        # handling until child exit. Short bounded waits service CtrlC while
        # the owned launcher remains live; its original deadlines are unchanged.
        while True:
            try:
                code = process.wait(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            print("Requesting owned application Stop; waiting for terminal cleanup.", flush=True)
            request = [str(PYTHON), "-B", str(ROOT / "scripts/online_candidate_stop.py"), value["run_id"]]
            try:
                subprocess.run(request, cwd=ROOT, check=True, timeout=18)
            except (OSError, subprocess.SubprocessError) as error:
                print("Stop request unconfirmed: " + type(error).__name__ + "; waiting for the owned launcher.", file=sys.stderr)
            # Reviewed launchers may serve their terminal result for120seconds.
            # Never kill an arbitrary child or treat a wait timeout as owned cleanup.
            try:
                code = process.wait(timeout=150)
            except subprocess.TimeoutExpired as error:
                raise ValueError("Owned launcher remains live; cleanup unconfirmed, no automatic replay") from error
            verify_terminal_result(value)
            return 130
        finally:
            signal.signal(signal.SIGINT, previous)
    if code == 0:
        verify_terminal_result(value)
    return code

def plan(preset: str, run_id: str) -> dict:
    if ROOT.resolve() != LOCAL_ROOT.resolve():
        raise ValueError("This entry requires the existing qualified local checkout; portable setup is not established")
    if preset not in PRESETS:
        raise ValueError("Only the four independently reviewed presets are available")
    if not re.fullmatch(r"native_prototype_[a-z0-9_]{1,80}", run_id):
        raise ValueError("A scoped native_prototype run ID is required")
    if digest(BASE / RETENTION) != RETENTION_SHA:
        raise ValueError("Experimental retention record changed")
    retained = load(RETENTION)
    if retained["status"] != "retained_tested_full01_online_application_experimental_not_main":
        raise ValueError("Retained experimental application unavailable")
    review = load(retained["independent_review"])
    if digest(BASE / retained["independent_review"]) != retained["review_sha256"]:
        raise ValueError("Independent review record changed")
    if not review["independent_review"] or review["verdict"] != "clear_tested_prepared_full01_application_for_experimental_retention":
        raise ValueError("Independent experimental scope clearance missing")
    if review["blocking_findings"] or review["evidence_hash_mismatches"]:
        raise ValueError("Blocking review finding or identity mismatch")
    for name, expected in retained["source_sha256"].items():
        if digest(BASE / name) != expected:
            raise ValueError("Reviewed source identity changed: " + name)
    sequence, frames, seed, launcher = PRESETS[preset]
    entries = [entry for entry in retained["prepared_entrypoints"]
               if entry["launcher"] == launcher and [sequence, frames, seed] in entry["presets"]]
    if len(entries) != 1 or [sequence, frames, seed] not in review["closed_scope"]["prepared_presets"]:
        raise ValueError("Requested application preset lies outside reviewed retention")
    version = "v10" if launcher.endswith("v10.py") else "v7"
    if (BASE / (run_id + "_application_release_" + version)).exists():
        raise FileExistsError("Run output already exists; do not overwrite or replay it")
    if not PYTHON.is_file():
        raise FileNotFoundError("Qualified GPU Python runtime missing; no installation attempted")
    required = entries[0].get("required_free_bytes", 1610612736 + 70000000)
    free = shutil.disk_usage(ROOT).free
    if free < required:
        raise ValueError(f"Insufficient physical capacity: {free} bytes available, {required} required")
    command = [str(PYTHON), "-B", str(BASE / launcher), "--run-id", run_id,
        "--sequence", sequence, "--frames", str(frames), "--seed", str(seed),
        "--telemetry-port", "8779", "--control-port", "8780",
        "--start-delay", "1", "--serve-after", "120"]
    return dict(status="source_bound_experimental_command_plan_not_a_new_validation",
        preset=preset, sequence=sequence, frames=frames, seed=seed, run_id=run_id,
        command=command, cwd=str(ROOT), required_physical_free_bytes=required,
        current_physical_free_bytes=free, dashboard_origin="http://127.0.0.1:4301",
        reviewed_inner_launcher=launcher, repository_frontend_independent_review=False,
        runtime_admission_checked=False, repository_ctrl_c_qualified=False,
        main_replacement=False, native_rate_qualified=False,
        prerequisite="Reuse isolated dashboard4301, no active benchmark/native worker, backendports8779/8780 free; open printed URL before120second readiness deadline")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--run-id", help="Optional fresh scoped native_prototype ID for reproducible run records")
    parser.add_argument("--check", action="store_true", help="Validate bindings and print plan; start no native/browser process")
    parser.add_argument("--check-host", action="store_true", help="With --check, also inspect live workers, ports and dashboard; start no child")
    args = parser.parse_args(argv)
    if args.check_host and not args.check:
        parser.error("--check-host requires --check")
    run_id = args.run_id or ("native_prototype_repo_" + datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fz"))
    try:
        value = plan(args.preset, run_id)
        if args.check:
            if args.check_host:
                value["runtime_admission"] = assert_runtime_ready()
                value["runtime_admission_checked"] = True
            print(json.dumps(value, indent=2, allow_nan=False))
            return 0
        assert_runtime_ready()
        value = plan(args.preset, run_id)  # Recheck source/capacity after bounded host observations.
        print("Experimental prepared preset. Reuse dashboard4301 and open the launcher's printed URL before its deadline.", flush=True)
        # Preserve the documented argv; isolate console signals so the parent can
        # request the existing authenticated owned Stop instead of blind termination.
        return run_foreground(value)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("Online candidate entry refused: " + str(error), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
