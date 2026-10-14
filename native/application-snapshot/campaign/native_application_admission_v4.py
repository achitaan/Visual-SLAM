"""Same-boot admission journal and bounded cleanup for the isolated native app.

Every Linux process group created by the application owner is admitted while a
shared file lock excludes Stop.  Stop closes that gate before loading the
journal, so a racing spawn is either absent or fully recorded.  Client data can
never select a PID, path, signal, group, session, or deadline.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time
from typing import Any

from native_application_stop_owner_v2 import (
    MAX_CONTROL, STOP_DIGEST, STOP_RUN, validate_control)
from native_owned_deadline import OwnedDeadline, identity
from native_process_owner_guarded import OwnedPausedProcess as OriginalOwner


SCHEMA = "native_application_admission_v3"
ENV_JOURNAL = "NATIVE_APPLICATION_ADMISSION_JOURNAL"
ENV_TOKEN = "NATIVE_APPLICATION_ADMISSION_TOKEN"
BOOT_PATH = Path("/proc/sys/kernel/random/boot_id")
BASE = Path(__file__).resolve().parent
RUN = re.compile(r"native_prototype_[a-z0-9_]+")
HEX64 = re.compile(r"[a-f0-9]{64}")


def _boot_id() -> str:
    value = BOOT_PATH.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[a-f0-9-]{36}", value):
        raise ValueError("Linux boot identity is unavailable")
    return value


def _journal_path(run_id: str, value: str | Path | None = None) -> Path:
    if not RUN.fullmatch(run_id):
        raise ValueError("scoped native application run ID required")
    path = Path(value if value is not None else os.environ.get(ENV_JOURNAL, ""))
    if path.name != f"{run_id}-admissions-v3.jsonl":
        raise ValueError("run-bound admission journal name required")
    if path.parent.resolve() != BASE.resolve() or path.is_symlink():
        raise ValueError("admission journal must be an owned BASE file")
    return path


def _lock_path(journal: Path) -> Path:
    return journal.with_suffix(journal.suffix + ".lock")


class admission_lock:
    def __init__(self, journal: Path):
        self.path = _lock_path(journal)
        self.stream = None

    def __enter__(self):
        if self.path.is_symlink():
            raise ValueError("admission lock may not be a symlink")
        self.stream = self.path.open("a+b")
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, typ, value, traceback):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        return False


def _common() -> tuple[str, Path, str, str, float]:
    run_id = os.environ.get(STOP_RUN, "")
    digest = os.environ.get(STOP_DIGEST, "")
    token = os.environ.get(ENV_TOKEN, "")
    journal = _journal_path(run_id)
    if not HEX64.fullmatch(digest) or not HEX64.fullmatch(token):
        raise ValueError("complete run capability and admission token required")
    rows = read_journal(journal)
    header = rows[0]
    deadline = header.get("original_deadline_monotonic_s")
    if (header.get("schema") != SCHEMA or header.get("event") != "journal_opened"
            or header.get("run_id") != run_id or header.get("boot_id") != _boot_id()
            or header.get("token_sha256") != hashlib.sha256(token.encode("ascii")).hexdigest()
            or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("admission journal header differs from current run/boot")
    return run_id, journal, token, digest, float(deadline)


def _write_line(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf8")
    if len(payload) > 4096:
        raise ValueError("admission record exceeds atomic line bound")
    flags = os.O_WRONLY | os.O_NOFOLLOW
    flags |= os.O_CREAT | os.O_EXCL if exclusive else os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("partial admission record write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_journal(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("bounded regular admission journal required")
    rows = []
    with path.open("r", encoding="utf8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("admission record must be an object")
            rows.append(value)
            if len(rows) > 256:
                raise ValueError("admission journal record bound exceeded")
    if not rows:
        raise ValueError("empty admission journal")
    return rows


def initialize_journal(*, run_id: str, journal_path: Path, token: str,
                       deadline: float, cap_s: float) -> Path:
    if sys.platform != "linux" or not HEX64.fullmatch(token):
        raise ValueError("Linux admission token required")
    journal = _journal_path(run_id, journal_path)
    if (not math.isfinite(deadline) or not math.isfinite(cap_s) or cap_s <= 0
            or deadline <= time.monotonic()):
        raise ValueError("future original application deadline required")
    for path in (journal, _lock_path(journal)):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    header = {
        "schema": SCHEMA, "event": "journal_opened", "run_id": run_id,
        "boot_id": _boot_id(), "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "original_deadline_monotonic_s": deadline, "original_cap_s": cap_s,
        "monotonic_s": time.monotonic(), "sealed": False}
    _write_line(journal, header, exclusive=True)
    _lock_path(journal).open("xb").close()
    os.environ[ENV_JOURNAL] = str(journal)
    os.environ[ENV_TOKEN] = token
    caller = identity(os.getpid())
    append_event("linux_dispatch_caller", "caller_observer", os.getpid(), caller,
                 deadline, signal_scope=False)
    return journal


def append_event(role: str, kind: str, pid: int, expected: tuple[int, int, int],
                 deadline: float, *, signal_scope: bool, owner_id: str | None = None) -> str:
    run_id, journal, token, _digest, original_deadline = _common()
    if deadline != original_deadline or time.monotonic() >= original_deadline:
        raise TimeoutError("admission occurred outside the original deadline")
    if not re.fullmatch(r"[a-z0-9_.-]+", role) or kind not in {
            "caller_observer", "owned_group", "gate_helper", "deadline_guardian",
            "control_guardian"}:
        raise ValueError("invalid admission role or kind")
    ticks, group, session = expected
    if any(type(value) is not int or value <= 0 for value in (pid, ticks, group, session)):
        raise ValueError("exact positive Linux process identity required")
    admission_id = hashlib.sha256(
        f"{run_id}:{role}:{kind}:{pid}:{ticks}:{group}:{session}".encode("ascii")).hexdigest()
    row = {
        "schema": SCHEMA, "event": "process_admitted", "run_id": run_id,
        "boot_id": _boot_id(), "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "original_deadline_monotonic_s": original_deadline, "monotonic_s": time.monotonic(),
        "admission_id": admission_id, "role": role, "kind": kind, "pid": pid,
        "pid_start_ticks": ticks, "process_group": group, "session": session,
        "signal_scope": signal_scope, "owner_id": owner_id}
    _write_line(journal, row)
    return admission_id


def _gate_closed(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("event") in {
        "stop_gate_closed", "completion_gate_closed", "cancellation_gate_closed"
    } for row in rows)


def _role(result_path: Path) -> str:
    name = result_path.name
    if "outer-owned-deadline" in name:
        return "outer_driver"
    if "finalizer-owned-deadline" in name:
        return "raw_finalizer"
    if name == "owned-deadline.json":
        return "native_writer"
    return "owned_helper"


class JournaledOwnedDeadline(OwnedDeadline):
    def __init__(self, process, *, deadline, result_path, role: str, **kwargs):
        super().__init__(process, deadline=deadline, result_path=result_path, **kwargs)
        owner_id = append_event(role, "owned_group", process.pid, self.expected,
                                deadline, signal_scope=True)
        append_event(role + ".gate", "gate_helper", process.pid, self.expected,
                     deadline, signal_scope=False, owner_id=owner_id)
        guardian_expected = identity(self.guard.pid)
        append_event(role + ".deadline", "deadline_guardian", self.guard.pid,
                     guardian_expected, deadline, signal_scope=False, owner_id=owner_id)
        self.admission_id = owner_id


class JournaledOwnedPausedProcess(OriginalOwner):
    """Frozen owner behavior with journaled spawn admission."""
    def start(self, command, *, env, stdout, **kwargs):
        run_id, journal, _token, _digest, _deadline = _common()
        role = _role(self.result_path)
        with admission_lock(journal):
            if _gate_closed(read_journal(journal)):
                raise RuntimeError("application Stop gate already closed")
            return super().start(
                command, env=env, stdout=stdout,
                guard_factory=lambda process, deadline, result_path:
                    JournaledOwnedDeadline(process, deadline=deadline,
                                           result_path=result_path, role=role),
                **kwargs)


class ApplicationOwnedPausedProcess(JournaledOwnedPausedProcess):
    """Outer owner with an admitted run-bound Stop guardian."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_guard: subprocess.Popen | None = None
        self.stop_result = Path(str(self.result_path) + ".application-stop-v3.json")

    def start(self, command, *, env, stdout, **kwargs):
        run_id, journal, token, digest, deadline = _common()
        if self.stop_result.exists() or self.stop_result.is_symlink():
            raise FileExistsError(self.stop_result)
        process, guard = super().start(command, env=env, stdout=stdout, **kwargs)
        try:
            # The watcher acquires this same lock before arming or reading stdin.
            # Even a Stop already queued in the pipe cannot precede its admission.
            with admission_lock(journal):
                self.stop_guard = subprocess.Popen(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--watch-stop",
                     run_id, digest, token, str(journal), repr(deadline),
                     str(process.pid), str(guard.expected[0]), str(self.stop_result)],
                    stdin=sys.stdin.buffer, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True)
                expected = identity(self.stop_guard.pid)
                append_event("application_stop_guardian", "control_guardian", self.stop_guard.pid,
                             expected, deadline, signal_scope=False, owner_id=guard.admission_id)
            ready, _, _ = select.select([self.stop_guard.stdout], [], [], 3.0)
            if not ready or json.loads(self.stop_guard.stdout.readline()).get("status") != "armed":
                raise TimeoutError("application Stop guardian admission failed")
            return process, guard
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            watcher = self.stop_guard
            if watcher is not None:
                try:
                    watcher.wait(timeout=14)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait(timeout=4)
                    raise
                finally:
                    watcher.stdout.close()
                    watcher.stderr.close()
                    self.stop_guard = None
                if watcher.returncode != 0:
                    raise RuntimeError("application Stop guardian failed")


def _proc_rows(*, token: str | None = None) -> list[dict[str, Any]]:
    rows = []
    marker = None if token is None else f"{ENV_TOKEN}={token}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            has_token = False
            if marker is not None:
                environ = (entry / "environ").read_bytes().split(b"\0")
                has_token = marker in environ
            rows.append({"pid": int(entry.name), "state": fields[0],
                         "group": int(fields[2]), "session": int(fields[3]),
                         "start_ticks": int(fields[19]), "has_token": has_token})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


def _admitted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    admitted = [row for row in rows if row.get("event") == "process_admitted"]
    ids = [row.get("admission_id") for row in admitted]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate process admission")
    groups = [row for row in admitted if row.get("kind") == "owned_group"]
    deadline_helpers = [row for row in admitted if row.get("kind") == "deadline_guardian"]
    if not groups or not any(row.get("role") == "outer_driver" for row in groups):
        raise ValueError("outer owned group admission missing")
    for group in groups:
        if (group.get("process_group") != group.get("pid")
                or group.get("session") != group.get("pid")):
            raise ValueError("owned group is not an admitted session leader")
        if not any(helper.get("owner_id") == group.get("admission_id")
                   for helper in deadline_helpers):
            raise ValueError("owned group deadline guardian admission missing")
    if not any(row.get("kind") == "control_guardian" for row in admitted):
        raise ValueError("control guardian admission missing")
    return admitted


def _signal_groups(admitted: list[dict[str, Any]], token: str, sig: signal.Signals) -> list[int]:
    processes = _proc_rows(token=token)
    signalled = []
    for row in reversed(admitted):
        if row.get("signal_scope") is not True or row.get("kind") != "owned_group":
            continue
        group, session = row["process_group"], row["session"]
        live = [p for p in processes if p["group"] == group and p["session"] == session
                and p["state"] not in ("Z", "X")]
        if not live:
            continue
        # An owned session cannot gain an unrelated member.  This remains safe
        # after leader exit and reaches a TERM-ignoring surviving group child.
        try:
            os.killpg(group, sig)
            signalled.append(group)
        except ProcessLookupError:
            pass
    return signalled


def _same_generation(pid: int, ticks: int) -> bool:
    try:
        return identity(pid)[0] == ticks
    except (FileNotFoundError, ProcessLookupError):
        return False


def watch_stop(run_id: str, digest: str, token: str, journal_path: Path,
               deadline: float, outer_pid: int, outer_ticks: int, output: Path) -> None:
    if (sys.platform != "linux" or not RUN.fullmatch(run_id) or not HEX64.fullmatch(digest)
            or not HEX64.fullmatch(token) or not math.isfinite(deadline)
            or outer_pid <= 0 or outer_ticks <= 0):
        raise ValueError("invalid journaled Stop ownership")
    os.environ[STOP_RUN] = run_id
    os.environ[STOP_DIGEST] = digest
    os.environ[ENV_TOKEN] = token
    os.environ[ENV_JOURNAL] = str(_journal_path(run_id, journal_path))
    journal = Path(os.environ[ENV_JOURNAL])
    with admission_lock(journal):
        _run, journal, _token, _digest, original_deadline = _common()
        own_ticks, own_group, own_session = identity(os.getpid())
        own_records = [row for row in read_journal(journal)
                       if row.get("event") == "process_admitted"
                       and row.get("kind") == "control_guardian"
                       and row.get("role") == "application_stop_guardian"
                       and (row.get("pid"), row.get("pid_start_ticks"),
                            row.get("process_group"), row.get("session")) ==
                           (os.getpid(), own_ticks, own_group, own_session)]
        if len(own_records) != 1:
            raise ValueError("Stop guardian must be admitted before consuming commands")
    if deadline != original_deadline:
        raise ValueError("Stop guardian deadline differs from original")
    state = {"schema": SCHEMA, "status": "armed", "run_id": run_id,
             "boot_id": _boot_id(), "original_deadline_monotonic_s": deadline,
             "rejected_controls": 0, "signals": []}
    with output.open("x", encoding="utf8") as stream:
        json.dump(state, stream)
    print(json.dumps({"status": "armed"}), flush=True)
    buffer = bytearray()
    discarding = False
    try:
        while time.monotonic() < deadline:
            if not _same_generation(outer_pid, outer_ticks):
                state["status"] = "owned_outer_driver_exited_without_stop"
                return
            ready, _, _ = select.select([sys.stdin.buffer], [], [], min(.025, deadline-time.monotonic()))
            if not ready:
                continue
            part = os.read(sys.stdin.fileno(), 1024)
            if not part:
                state["status"] = "control_pipe_closed_without_stop"
                return
            accepted = False
            for byte in part:
                if byte == 10:
                    if not discarding and validate_control(bytes(buffer), run_id, digest):
                        accepted = True
                        break
                    state["rejected_controls"] += 1
                    buffer.clear()
                    discarding = False
                elif not discarding:
                    buffer.append(byte)
                    if len(buffer) > MAX_CONTROL:
                        buffer.clear()
                        discarding = True
            if not accepted:
                continue
            with admission_lock(journal):
                rows = read_journal(journal)
                admitted = _admitted(rows)
                gate = {"schema": SCHEMA, "event": "stop_gate_closed", "run_id": run_id,
                        "boot_id": _boot_id(), "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
                        "original_deadline_monotonic_s": deadline, "monotonic_s": time.monotonic()}
                _write_line(journal, gate)
            state["request_observed_monotonic_s"] = time.monotonic()
            for sig, grace in ((signal.SIGTERM, .5), (signal.SIGKILL, .5)):
                groups = _signal_groups(admitted, token, sig)
                if groups:
                    state["signals"].append({"signal": int(sig), "groups": groups,
                                             "monotonic_s": time.monotonic()})
                stop = min(deadline, time.monotonic() + grace)
                while time.monotonic() < stop:
                    live = _proc_rows(token=token)
                    if not any(p["has_token"] and p["pid"] != os.getpid()
                               and p["state"] not in ("Z", "X") for p in live):
                        break
                    time.sleep(.02)
            state["status"] = "stop_signals_finished_cleanup_requires_sealed_audit"
            return
        state["status"] = "original_deadline_left_to_unchanged_watchdogs"
    finally:
        state["completed_monotonic_s"] = time.monotonic()
        output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf8")


def seal_audit(run_id: str, token: str, journal_path: Path, audit_path: Path,
               cleanup_s: float, *, mode: str = "stop") -> dict[str, Any]:
    if (sys.platform != "linux" or not RUN.fullmatch(run_id) or not HEX64.fullmatch(token)
            or not math.isfinite(cleanup_s) or not .1 <= cleanup_s <= 15):
        raise ValueError("bounded Linux cleanup audit request required")
    if mode not in {"stop", "completed", "cancelled"}:
        raise ValueError("explicit terminal gate purpose required")
    os.environ[STOP_RUN] = run_id
    os.environ[ENV_TOKEN] = token
    os.environ[ENV_JOURNAL] = str(_journal_path(run_id, journal_path))
    rows = read_journal(Path(os.environ[ENV_JOURNAL]))
    header = rows[0]
    expected_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    if (header.get("boot_id") != _boot_id() or header.get("token_sha256") != expected_digest):
        raise ValueError("stale or foreign admission journal")
    deadline = header.get("original_deadline_monotonic_s")
    for row in rows:
        if (row.get("schema") != SCHEMA or row.get("run_id") != run_id
                or row.get("boot_id") != header["boot_id"]
                or row.get("token_sha256") != expected_digest
                or row.get("original_deadline_monotonic_s") != deadline):
            raise ValueError("partial or mixed admission journal")
    if mode in {"completed", "cancelled"}:
        # Closing the admission gate is a separate action from signalling Stop.
        # Completion must never fabricate a stop_gate_closed event.
        with admission_lock(Path(os.environ[ENV_JOURNAL])):
            current = read_journal(Path(os.environ[ENV_JOURNAL]))
            if current != rows or any(row.get("event") == "journal_sealed" for row in current):
                raise ValueError("terminal gate journal changed or already sealed")
            if time.monotonic() >= deadline:
                raise TimeoutError("terminal admission closure exceeds original arm deadline")
            if mode == "completed":
                case_path = BASE / run_id / "case.json"
                if case_path.is_symlink() or case_path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("bounded completed native case required")
                case = json.loads(case_path.read_text())
                if case.get("status") != "completed_seeded_native_pending_evaluation" or case.get("exit_code") != 0:
                    raise ValueError("normal closure requires completed native dispatch")
            if not _gate_closed(current):
                _write_line(Path(os.environ[ENV_JOURNAL]), {
                    "schema": SCHEMA,
                    "event": "completion_gate_closed" if mode == "completed" else "cancellation_gate_closed",
                    "run_id": run_id, "boot_id": header["boot_id"],
                    "token_sha256": expected_digest,
                    "original_deadline_monotonic_s": deadline,
                    "monotonic_s": time.monotonic(),
                    "terminal_purpose": mode})
                rows = read_journal(Path(os.environ[ENV_JOURNAL]))
    admitted = _admitted(rows)
    if not _gate_closed(rows):
        raise ValueError("cleanup seal requires a recorded Stop gate close")
    end = min(time.monotonic() + cleanup_s, deadline)
    proof = None
    while time.monotonic() < end:
        processes = _proc_rows(token=token)
        process_by_identity = {(p["pid"], p["start_ticks"]): p for p in processes
                               if p["state"] not in ("Z", "X")}
        owned = [row for row in admitted if row["kind"] == "owned_group"]
        helpers = [row for row in admitted
                   if row["kind"] not in ("owned_group", "caller_observer")]
        callers = [row for row in admitted if row["kind"] == "caller_observer"]
        groups = []
        for row in owned:
            live = [p["pid"] for p in processes
                    if p["group"] == row["process_group"] and p["session"] == row["session"]
                    and p["state"] not in ("Z", "X")]
            groups.append({"admission_id": row["admission_id"], "role": row["role"],
                           "kind": row["kind"], "live_members": live})
        live_helpers = [{"admission_id": row["admission_id"], "role": row["role"],
                         "pid": row["pid"], "pid_start_ticks": row["pid_start_ticks"]}
                        for row in helpers
                        if (row["pid"], row["pid_start_ticks"]) in process_by_identity]
        live_callers = [{"admission_id": row["admission_id"], "role": row["role"],
                         "pid": row["pid"], "pid_start_ticks": row["pid_start_ticks"]}
                        for row in callers
                        if (row["pid"], row["pid_start_ticks"]) in process_by_identity]
        admitted_exact = {(row["pid"], row["pid_start_ticks"])
                          for row in helpers + callers}
        escaped = [p["pid"] for p in processes if p["has_token"] and p["pid"] != os.getpid()
                   and p["state"] not in ("Z", "X")
                   and (p["pid"], p["start_ticks"]) not in admitted_exact
                   and not any(p["group"] == row["process_group"] and p["session"] == row["session"]
                               for row in owned)]
        proof = {"owned_groups": groups, "live_guardians_or_helpers": live_helpers,
                 "live_caller_observers": live_callers,
                 "token_processes_outside_admissions": escaped}
        if (all(not group["live_members"] for group in groups)
                and not live_helpers and not live_callers and not escaped):
            break
        time.sleep(.02)
    else:
        raise TimeoutError(("whole admitted process cleanup unconfirmed", proof))
    journal_bytes = Path(os.environ[ENV_JOURNAL]).read_bytes()
    seal = {"schema": SCHEMA, "event": "journal_sealed", "run_id": run_id,
            "boot_id": _boot_id(), "token_sha256": expected_digest,
            "original_deadline_monotonic_s": deadline, "monotonic_s": time.monotonic(),
            "preseal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
            "admission_count": len(admitted), "all_owned_exited": True,
            "group_session_escape_assumption": "token_scan_catches_inherited-environment_escape; deliberate environment clearing is outside the admitted native source contract"}
    with admission_lock(Path(os.environ[ENV_JOURNAL])):
        current = read_journal(Path(os.environ[ENV_JOURNAL]))
        if current != rows or any(row.get("event") == "journal_sealed" for row in current):
            raise ValueError("admission journal changed before one-time seal")
        _write_line(Path(os.environ[ENV_JOURNAL]), seal)
    result = {"schema": SCHEMA, "status": "sealed_all_admitted_groups_and_token_orphans_exited",
              "run_id": run_id, "boot_id": _boot_id(), "all_owned_exited": True,
              "admission_count": len(admitted), "proof": proof, "seal": seal,
              "signals_sent": 0}
    result["gate_purpose"] = mode
    if audit_path.parent.resolve() != BASE.resolve() or audit_path.is_symlink():
        raise ValueError("audit output must be a new BASE file")
    with audit_path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 10 and sys.argv[1] == "--watch-stop":
        watch_stop(sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]),
                   float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), Path(sys.argv[9]))
    elif len(sys.argv) == 8 and sys.argv[1] == "--seal-terminal":
        seal_audit(sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]),
                   float(sys.argv[6]), mode=sys.argv[7])
    elif len(sys.argv) == 7 and sys.argv[1] == "--seal-audit":
        seal_audit(sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]),
                   float(sys.argv[6]))
    else:
        raise ValueError("admission helper requires owned watch or seal arguments")
