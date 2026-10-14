"""Own an entire Linux arm driver under its one original absolute deadline."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from native_owned_deadline import identity
from native_application_stop_owner_v2 import OwnedPausedProcess


SCHEMA = "native_entire_arm_owner_v2"
ENV_START = "NATIVE_ARM_ORIGINAL_START_MONOTONIC_S"
ENV_DEADLINE = "NATIVE_ARM_ORIGINAL_DEADLINE_MONOTONIC_S"
ENV_CAP = "NATIVE_ARM_ORIGINAL_CAP_S"
ENV_WATCHDOG = "NATIVE_ARM_OUTER_WATCHDOG_PATH"
ENV_TERMINAL = "NATIVE_ARM_OUTER_TERMINAL_PATH"
ENV_INNER = "NATIVE_ARM_OWNED_INNER_V2"
TERMINAL_TEMP_SUFFIX = ".tmp"


class EntireArmFailure(RuntimeError):
    def __init__(self, message: str, proof: dict[str, Any]):
        super().__init__(message)
        self.proof = proof


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def inherited_arm_context(expected_cap_s: float) -> dict[str, Any]:
    """Validate inherited timing/identity before the inner driver reads inputs."""
    if sys.platform != "linux" or os.environ.get(ENV_INNER) != "1":
        raise RuntimeError("entire arm driver requires owned Linux bootstrap admission")
    try:
        started = float(os.environ[ENV_START])
        deadline = float(os.environ[ENV_DEADLINE])
        cap = float(os.environ[ENV_CAP])
    except (KeyError, ValueError) as exc:
        raise ValueError("missing inherited original arm timing") from exc
    if not all(math.isfinite(value) for value in (started, deadline, cap, expected_cap_s)):
        raise ValueError("finite inherited arm timing required")
    if cap <= 0 or cap != expected_cap_s or deadline != started + cap:
        raise ValueError("inner driver may not replace the inherited original deadline")
    if time.monotonic() >= deadline:
        raise TimeoutError("original arm deadline expired before inner driver admission")
    pid = os.getpid()
    ticks, process_group, session = identity(pid)
    if process_group != pid or session != pid:
        raise RuntimeError("entire arm driver is not its owned session leader")
    watchdog_path = Path(os.environ[ENV_WATCHDOG])
    terminal_path = Path(os.environ[ENV_TERMINAL])
    return {
        "schema": SCHEMA,
        "arm_started_monotonic_s": started,
        "absolute_deadline_monotonic_s": deadline,
        "cap_s": cap,
        "process_identity": {
            "pid": pid, "pid_start_ticks": ticks,
            "process_group": process_group, "session": session},
        "watchdog_path": str(watchdog_path),
        "terminal_path": str(terminal_path)
    }


def publish_outer_terminal(context: dict[str, Any], case_path: Path) -> dict[str, Any]:
    """Publish the last inner-driver success marker while the outer guard is live."""
    deadline = context["absolute_deadline_monotonic_s"]
    if time.monotonic() >= deadline:
        raise TimeoutError("original deadline before outer terminal publication")
    case = json.loads(case_path.read_text())
    if case.get("status") != "completed_seeded_native_pending_evaluation":
        raise ValueError("outer terminal publication requires completed bounded case")
    if case.get("wall_s", math.inf) > context["cap_s"]:
        raise ValueError("case wall exceeds inherited original cap")
    case_hash = file_sha256(case_path)
    terminal = {
        "schema": SCHEMA,
        "status": "entire_arm_driver_terminal_within_original_deadline",
        "arm_started_monotonic_s": context["arm_started_monotonic_s"],
        "absolute_deadline_monotonic_s": deadline,
        "cap_s": context["cap_s"],
        "terminal_monotonic_s": time.monotonic(),
        "process_identity": context["process_identity"],
        "watchdog_path": context["watchdog_path"],
        "case_path": str(case_path),
        "case_sha256": case_hash,
        "unknown_usage": None,
        "unknown_usage_zero_filled": False
    }
    if terminal["terminal_monotonic_s"] >= deadline:
        raise TimeoutError("outer terminal timestamp exceeded original deadline")
    target = Path(context["terminal_path"])
    temporary = Path(str(target) + TERMINAL_TEMP_SUFFIX)
    for path in (target, temporary):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    payload = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if time.monotonic() >= deadline:
        raise TimeoutError("original deadline before outer terminal replace")
    os.replace(temporary, target)
    if time.monotonic() >= deadline:
        raise TimeoutError("original deadline during outer terminal publication")
    return terminal


def verify_outer_saved(case: dict[str, Any], outer_terminal: dict[str, Any],
                       outer_watchdog: dict[str, Any], *, cap_s: float,
                       case_sha256: str) -> dict[str, Any]:
    if case.get("status") != "completed_seeded_native_pending_evaluation":
        raise ValueError("bounded case is not completed")
    if case.get("cap_s") != cap_s or case.get("wall_s", math.inf) > cap_s:
        raise ValueError("case wall exceeds explicit original cap")
    if outer_terminal.get("schema") != SCHEMA or outer_terminal.get("status") != "entire_arm_driver_terminal_within_original_deadline":
        raise ValueError("outer terminal proof absent")
    start = outer_terminal.get("arm_started_monotonic_s")
    deadline = outer_terminal.get("absolute_deadline_monotonic_s")
    if outer_terminal.get("cap_s") != cap_s or deadline != start + cap_s:
        raise ValueError("outer proof changed original start/cap/deadline")
    if outer_terminal.get("terminal_monotonic_s", math.inf) >= deadline:
        raise ValueError("entire arm driver terminal proof is late")
    outer_identity = outer_terminal.get("process_identity")
    if not isinstance(outer_identity, dict) or any(type(outer_identity.get(key)) is not int
                                                   for key in ("pid", "pid_start_ticks", "process_group", "session")):
        raise ValueError("outer process identity is absent")
    if (outer_identity["pid"] <= 0 or outer_identity["process_group"] != outer_identity["pid"]
            or outer_identity["session"] != outer_identity["pid"]):
        raise ValueError("outer driver was not an owned session leader")
    if (outer_watchdog.get("pid"), outer_watchdog.get("pid_start_ticks")) != (
            outer_identity["pid"], outer_identity["pid_start_ticks"]):
        raise ValueError("outer watchdog identity differs from driver terminal")
    if outer_watchdog.get("deadline_monotonic_s") != deadline:
        raise ValueError("outer watchdog deadline differs from original deadline")
    if outer_watchdog.get("status") != "writer_exited_before_deadline" or outer_watchdog.get("signals") != []:
        raise ValueError("outer watchdog does not prove unsignalled driver exit")
    if (not isinstance(case_sha256, str) or len(case_sha256) != 64
            or outer_terminal.get("case_sha256") != case_sha256):
        raise ValueError("outer terminal proof does not bind the saved case bytes")
    if outer_terminal.get("unknown_usage", "missing") is not None or outer_terminal.get("unknown_usage_zero_filled") is not False:
        raise ValueError("outer unknown usage must remain null and not zero-filled")
    return {"status": "verified_entire_arm_original_deadline_v2",
            "cap_s": cap_s, "outer_process_identity": outer_identity}


def run_owned_arm(*, command: list[str], name: str, cap_s: float,
                  control_root: Path, case_path: Path | None,
                  extra_env: dict[str, str] | None = None,
                  abort_before_release: bool = False,
                  cleanup_control_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Bootstrap an entire arm child before it can perform arm I/O."""
    started = time.monotonic()
    if (sys.platform != "linux" or not re.fullmatch(r"[a-z0-9_]+", name)
            or not math.isfinite(cap_s) or not 0 < cap_s <= 1800):
        raise ValueError("invalid entire-arm bootstrap request")
    deadline = started + cap_s
    control_root = Path(control_root)
    resolved_control_root = control_root.resolve()
    for cleanup in cleanup_control_paths:
        cleanup = Path(cleanup)
        if (not cleanup.resolve().is_relative_to(resolved_control_root)
                or not cleanup.name.endswith("finalizer-request.json")):
            raise ValueError("cleanup path is not a scoped finalizer request")
    watchdog_path = control_root / f"{name}-outer-owned-deadline-v2.json"
    terminal_path = control_root / f"{name}-outer-terminal-v2.json"
    terminal_temp = Path(str(terminal_path) + TERMINAL_TEMP_SUFFIX)
    log_path = control_root / f"{name}-outer-driver-v2.log"
    for path in (watchdog_path, terminal_path, terminal_temp, log_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    environment = os.environ.copy()
    environment.update({
        ENV_INNER: "1", ENV_START: repr(started), ENV_DEADLINE: repr(deadline),
        ENV_CAP: repr(cap_s), ENV_WATCHDOG: str(watchdog_path),
        ENV_TERMINAL: str(terminal_path)})
    if extra_env:
        environment.update(extra_env)
    proof: dict[str, Any] = {
        "schema": SCHEMA, "status": "entire_arm_not_accepted",
        "arm_started_monotonic_s": started,
        "absolute_deadline_monotonic_s": deadline, "cap_s": cap_s,
        "watchdog_path": str(watchdog_path), "terminal_path": str(terminal_path),
        "log_path": str(log_path), "unknown_usage": None}
    process = None
    try:
        with log_path.open("xb") as log:
            with OwnedPausedProcess(deadline=deadline, result_path=watchdog_path) as owner:
                process, guard = owner.start(command, env=environment, stdout=log)
                expected = guard.expected
                proof["expected_identity"] = {
                    "pid": process.pid, "pid_start_ticks": expected[0],
                    "process_group": expected[1], "session": expected[2]}
                if abort_before_release:
                    raise EntireArmFailure("qualification abort before bootstrap release", proof)
                owner.release()
                guard_error = None
                while process.poll() is None:
                    try:
                        guard.check()
                    except TimeoutError as exc:
                        guard_error = repr(exc)
                    time.sleep(.005)
                proof["returncode"] = process.returncode
                if guard_error:
                    proof["guard_error"] = guard_error
        proof["driver_exit_observed_monotonic_s"] = time.monotonic()
        proof["watchdog"] = json.loads(watchdog_path.read_text())
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        proof["terminal"] = terminal
        accepted = (
            proof.get("returncode") == 0
            and isinstance(terminal, dict)
            and terminal.get("process_identity") == proof.get("expected_identity")
            and terminal.get("arm_started_monotonic_s") == started
            and terminal.get("absolute_deadline_monotonic_s") == deadline
            and terminal.get("terminal_monotonic_s", math.inf) < deadline
            and proof["watchdog"].get("status") == "writer_exited_before_deadline"
            and proof["watchdog"].get("signals") == [])
        if case_path is not None and accepted:
            case_raw = case_path.read_bytes()
            verify_outer_saved(json.loads(case_raw), terminal, proof["watchdog"],
                               cap_s=cap_s,
                               case_sha256=hashlib.sha256(case_raw).hexdigest())
        proof["observed_monotonic_s"] = time.monotonic()
        accepted = accepted and proof["observed_monotonic_s"] < deadline
        if not accepted:
            raise EntireArmFailure("entire arm result rejected", proof)
        proof["status"] = "entire_arm_accepted_within_original_deadline"
        return proof
    except EntireArmFailure as exc:
        if process is not None:
            proof["returncode"] = process.poll()
        if watchdog_path.exists():
            try:
                proof["watchdog"] = json.loads(watchdog_path.read_text())
            except Exception as watch_exc:
                proof["watchdog_error"] = repr(watch_exc)
        raise
    except BaseException as exc:
        proof["error"] = repr(exc)
        if watchdog_path.exists():
            try:
                proof["watchdog"] = json.loads(watchdog_path.read_text())
            except Exception as watch_exc:
                proof["watchdog_error"] = repr(watch_exc)
        raise EntireArmFailure("entire arm bootstrap failed", proof) from exc
    finally:
        # Only a never-published temporary control marker is startup cleanup.
        if terminal_temp.exists() and not terminal_temp.is_symlink():
            terminal_temp.unlink()
        # These paths must be explicit caller-owned transient requests.  They
        # are cleaned only after the guarded driver is terminal; payload/stage
        # evidence and completed proof files are never included here.
        for cleanup in cleanup_control_paths:
            cleanup = Path(cleanup)
            if cleanup.exists() and not cleanup.is_symlink() and cleanup.is_file():
                cleanup.unlink()
