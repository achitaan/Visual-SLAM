"""Journaled entire-arm owner for the isolated native Stop application."""
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

from native_application_admission_v4 import (
    ApplicationOwnedPausedProcess, ENV_JOURNAL, ENV_TOKEN, initialize_journal)
from native_application_arm_owner_v4 import (
    ENV_CAP, ENV_DEADLINE, ENV_INNER, ENV_START, ENV_TERMINAL, ENV_WATCHDOG,
    SCHEMA, TERMINAL_TEMP_SUFFIX, EntireArmFailure, verify_outer_saved)
from native_application_stop_owner_v2 import STOP_DIGEST, STOP_RUN


def run_owned_arm(*, command: list[str], name: str, cap_s: float,
                  control_root: Path, case_path: Path | None,
                  control_digest: str, admission_token: str,
                  journal_path: Path,
                  extra_env: dict[str, str] | None = None,
                  cleanup_control_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    started = time.monotonic()
    if (sys.platform != "linux" or not re.fullmatch(r"native_prototype_[a-z0-9_]+", name)
            or not math.isfinite(cap_s) or not 0 < cap_s <= 1800
            or not re.fullmatch(r"[a-f0-9]{64}", control_digest)
            or not re.fullmatch(r"[a-f0-9]{64}", admission_token)):
        raise ValueError("invalid journaled entire-arm request")
    deadline = started + cap_s
    control_root = Path(control_root)
    resolved_control_root = control_root.resolve()
    for cleanup in cleanup_control_paths:
        cleanup = Path(cleanup)
        if (not cleanup.resolve().is_relative_to(resolved_control_root)
                or not cleanup.name.endswith("finalizer-request.json")):
            raise ValueError("cleanup path is not a scoped finalizer request")
    watchdog_path = control_root / f"{name}-outer-owned-deadline-v5.json"
    terminal_path = control_root / f"{name}-outer-terminal-v5.json"
    terminal_temp = Path(str(terminal_path) + TERMINAL_TEMP_SUFFIX)
    log_path = control_root / f"{name}-outer-driver-v5.log"
    for path in (watchdog_path, terminal_path, terminal_temp, log_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    os.environ[STOP_RUN] = name
    os.environ[STOP_DIGEST] = control_digest
    os.environ[ENV_JOURNAL] = str(journal_path)
    os.environ[ENV_TOKEN] = admission_token
    initialize_journal(run_id=name, journal_path=journal_path, token=admission_token,
                       deadline=deadline, cap_s=cap_s)
    environment = os.environ.copy()
    environment.update({
        ENV_INNER: "1", ENV_START: repr(started), ENV_DEADLINE: repr(deadline),
        ENV_CAP: repr(cap_s), ENV_WATCHDOG: str(watchdog_path),
        ENV_TERMINAL: str(terminal_path), STOP_RUN: name, STOP_DIGEST: control_digest,
        ENV_JOURNAL: str(journal_path), ENV_TOKEN: admission_token})
    if extra_env:
        environment.update(extra_env)
    proof: dict[str, Any] = {
        "schema": SCHEMA, "status": "entire_arm_not_accepted",
        "arm_started_monotonic_s": started,
        "absolute_deadline_monotonic_s": deadline, "cap_s": cap_s,
        "watchdog_path": str(watchdog_path), "terminal_path": str(terminal_path),
        "log_path": str(log_path), "admission_journal": str(journal_path),
        "unknown_usage": None}
    process = None
    try:
        with log_path.open("xb") as log:
            with ApplicationOwnedPausedProcess(deadline=deadline,
                                               result_path=watchdog_path) as owner:
                process, guard = owner.start(command, env=environment, stdout=log)
                expected = guard.expected
                proof["expected_identity"] = {
                    "pid": process.pid, "pid_start_ticks": expected[0],
                    "process_group": expected[1], "session": expected[2]}
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
            proof.get("returncode") == 0 and isinstance(terminal, dict)
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
        if not accepted or proof["observed_monotonic_s"] >= deadline:
            raise EntireArmFailure("journaled entire-arm result rejected", proof)
        proof["status"] = "entire_arm_accepted_within_original_deadline"
        return proof
    except EntireArmFailure:
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
        raise EntireArmFailure("journaled entire-arm bootstrap failed", proof) from exc
    finally:
        if terminal_temp.exists() and not terminal_temp.is_symlink():
            terminal_temp.unlink()
        for cleanup in cleanup_control_paths:
            cleanup = Path(cleanup)
            if cleanup.exists() and not cleanup.is_symlink() and cleanup.is_file():
                cleanup.unlink()
