"""Bound post-writer export and terminal publication to the arm's original cap.

Linux only.  The finalizer is a separately owned session-leader child.  It is
spawned paused, admitted only after its independent watchdog is armed, and is
never given a replacement deadline.  Existing local/compact exporters remain
the byte-preservation implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

from native_compact_output import NativeCompactOutput
from native_local_output import NativeLocalOutput, directory
from native_process_owner_guarded import OwnedPausedProcess


SCHEMA = "native_owned_finalizer_v1"
SUCCESS = "completed_seeded_native_pending_evaluation"
INCOMPLETE = "incomplete_seeded_native_arm"
CONTROL_NAME = ".native-owned-finalizer-request.json"
TEMP_NAME = ".case.json.native-owned-finalizer.tmp"


class FinalizationFailure(RuntimeError):
    """A terminal failure carrying process ownership evidence."""

    def __init__(self, message: str, proof: dict[str, Any]):
        super().__init__(message)
        self.proof = proof


def _sha256(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("original arm deadline during final destination hash")
            part = stream.read(1024 * 1024)
            if not part:
                return digest.hexdigest()
            digest.update(part)


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _load_request(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    request = json.loads(raw)
    if request.get("schema") != SCHEMA:
        raise ValueError("unknown finalizer request schema")
    return request, hashlib.sha256(raw).hexdigest()


def _validate_request(request: dict[str, Any]) -> tuple[float, float, float]:
    started = request.get("arm_started_monotonic_s")
    cap = request.get("cap_s")
    deadline = request.get("absolute_deadline_monotonic_s")
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (started, cap, deadline)):
        raise ValueError("finite original arm timing required")
    if cap <= 0 or deadline != started + cap:
        raise ValueError("finalizer must use exact original start plus cap")
    if request.get("writer_exited") is not True:
        raise ValueError("finalizer cannot run while native writer is live")
    if request.get("mode") not in ("compact", "local"):
        raise ValueError("unknown finalizer mode")
    record = request.get("record")
    if not isinstance(record, dict) or record.get("cap_s") != cap:
        raise ValueError("record cap differs from original finalizer cap")
    return float(started), float(cap), float(deadline)


def _rehydrate_transport(request: dict[str, Any]) -> NativeLocalOutput:
    kind = NativeCompactOutput if request["mode"] == "compact" else NativeLocalOutput
    transport = kind.__new__(kind)
    transport.destination = Path(request["destination"])
    transport.physical_root = Path(request["physical_root"])
    transport.reserve_bytes = request["reserve_bytes"]
    transport.clock = time.monotonic
    transport.stage = Path(request["stage"])
    transport.created_signature = tuple(request["stage_signature"])
    transport.exported = False
    directory(transport.destination)
    directory(transport.stage)
    if (transport.stage.stat().st_dev, transport.stage.stat().st_ino) != transport.created_signature:
        raise ValueError("stage identity changed before bounded finalizer")
    return transport


def _qualification_fault(request: dict[str, Any], deadline: float) -> None:
    fault = request.get("qualification_fault")
    if fault is None:
        return
    if request.get("qualification_fixture") is not True:
        raise ValueError("fault injection is restricted to qualification fixtures")
    if fault == "exception_before_export":
        raise RuntimeError("qualification exception before export")
    if fault == "block_export_io":
        fifo = Path(request["qualification_fifo"])
        if not fifo.is_fifo():
            raise ValueError("qualification FIFO required")
        # Install the fault at the unchanged exporter's real fsync call.  No
        # FIFO writer exists, so this child blocks in kernel I/O while the
        # independent watchdog remains the only deadline enforcer.
        import native_compact_output as compact_module

        def blocked_fsync(_fd: int) -> None:
            with fifo.open("rb"):
                raise AssertionError("unreachable blocked exporter fsync")

        compact_module.os.fsync = blocked_fsync
        return
    if fault != "delay_final_hash":
        raise ValueError("unknown qualification fault")
    delay = request.get("qualification_delay_s")
    if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay <= 0:
        raise ValueError("positive finite qualification delay required")
    time.sleep(delay)
    if time.monotonic() >= deadline:
        raise TimeoutError("qualification delayed final hash crossed original cap")


def _publish_case(case_path: Path, record: dict[str, Any], deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("original arm deadline before terminal publication")
    temporary = case_path.parent / TEMP_NAME
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if time.monotonic() >= deadline:
        raise TimeoutError("original arm deadline before terminal replace")
    os.replace(temporary, case_path)
    if time.monotonic() >= deadline:
        # The parent observes this non-zero exit/watchdog race and replaces the
        # visible record with an incomplete one.  Late success is never valid.
        raise TimeoutError("original arm deadline during terminal publication")


def child_finalize(request_path: Path) -> int:
    request, request_sha256 = _load_request(request_path)
    started, cap, deadline = _validate_request(request)
    # The request is the only startup control artifact eligible for cleanup.
    # Payloads, stage evidence, logs, and watchdog proof are never removed.
    request_path.unlink()
    if time.monotonic() >= deadline:
        raise TimeoutError("original arm deadline before finalizer work")
    transport = _rehydrate_transport(request)
    if request.get("qualification_fault") in ("exception_before_export", "block_export_io"):
        _qualification_fault(request, deadline)
    record = dict(request["record"])
    if request["mode"] == "compact":
        native_hashes = request.get("native_hashes")
        if not isinstance(native_hashes, dict):
            raise ValueError("compact finalization requires native hashes")
        output_transport = transport.export_compact(
            deadline=deadline, writer_exited=True, native_hashes=native_hashes)
        if {name: item["raw_sha256"] for name, item in output_transport["files"].items()} != native_hashes:
            raise ValueError("compact raw hashes differ from native stage")
        record["native_output_hashes"] = native_hashes
        expected = {item["output_name"]: item["sha256"]
                    for item in output_transport["files"].values()}
    else:
        output_transport = transport.export(deadline=deadline, writer_exited=True)
        expected = {name: item["sha256"] for name, item in output_transport["files"].items()}
    record["output_transport"] = output_transport
    if request.get("qualification_fault") == "delay_final_hash":
        _qualification_fault(request, deadline)
    destination = Path(request["destination"])
    actual = {name: _sha256(destination / name, deadline) for name in sorted(expected)}
    if actual != expected:
        raise ValueError("final destination hash set differs from exported metadata")
    if {path.name for path in destination.iterdir()} != set(expected) | {"case.json"}:
        raise ValueError("final destination file set contains unowned artifacts")
    terminal = time.monotonic()
    elapsed = terminal - started
    if terminal >= deadline or elapsed > cap:
        raise TimeoutError("terminal finalization exceeded original arm cap")
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    identity = {
        "pid": os.getpid(),
        "pid_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "process_group": os.getpgrp(),
        "session": os.getsid(0)
    }
    if identity["process_group"] != identity["pid"] or identity["session"] != identity["pid"]:
        raise RuntimeError("finalizer is not its owned session leader")
    record.update(
        status=(SUCCESS if request["mode"] == "compact" else record.get("status", INCOMPLETE)),
        output_hashes=actual,
        wall_s=elapsed,
        utc=datetime.now(timezone.utc).isoformat(),
        finalization={
            "schema": SCHEMA,
            "status": "terminal_case_published_within_original_deadline",
            "request_sha256": request_sha256,
            "arm_started_monotonic_s": started,
            "absolute_deadline_monotonic_s": deadline,
            "terminal_monotonic_s": terminal,
            "elapsed_s": elapsed,
            "cap_s": cap,
            "process_identity": identity,
            "watchdog_path": request["watchdog_path"],
            "finalizer_self_usage": {
                "max_rss_kib": self_usage.ru_maxrss,
                "user_s": self_usage.ru_utime,
                "system_s": self_usage.ru_stime
            },
            "unknown_usage": None,
            "unknown_usage_zero_filled": False
        })
    _publish_case(Path(request["case_path"]), record, deadline)
    return 0


def _failure_record(case_path: Path, base_record: dict[str, Any], proof: dict[str, Any], error: str) -> None:
    record = dict(base_record)
    record.update(status=INCOMPLETE, error=error, finalization_failure=proof)
    temporary = case_path.parent / TEMP_NAME
    if temporary.exists() and not temporary.is_symlink():
        temporary.unlink()
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, case_path)


def finalize_owned(*, transport: NativeLocalOutput, record: dict[str, Any],
                   arm_started_monotonic_s: float, cap_s: float,
                   mode: str, native_hashes: dict[str, str] | None,
                   watchdog_path: Path, request_path: Path, log_path: Path,
                   qualification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run export/hash/publication in a separately guarded Linux child."""
    if mode == "local" and record.get("status") not in (SUCCESS, INCOMPLETE):
        raise ValueError("application local export requires validated terminal input status")
    expected_status = SUCCESS if mode == "compact" else record["status"]
    deadline = arm_started_monotonic_s + cap_s
    case_path = transport.destination / "case.json"
    if not (math.isfinite(deadline) and deadline > time.monotonic()):
        proof = {
            "status": "prestart_deadline_expired", "deadline_monotonic_s": deadline,
            "observed_monotonic_s": time.monotonic(), "unknown_usage": None}
        _failure_record(case_path, record, proof,
                        "no original deadline remains before finalizer admission")
        raise FinalizationFailure("no original deadline remains before finalizer admission", proof)
    for path in (request_path, watchdog_path, log_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    request: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "record": record,
        "writer_exited": True,
        "arm_started_monotonic_s": arm_started_monotonic_s,
        "cap_s": cap_s,
        "absolute_deadline_monotonic_s": deadline,
        "stage": str(transport.stage),
        "stage_signature": list(transport.created_signature),
        "destination": str(transport.destination),
        "physical_root": str(transport.physical_root),
        "reserve_bytes": transport.reserve_bytes,
        "native_hashes": native_hashes,
        "case_path": str(case_path),
        "watchdog_path": str(watchdog_path)
    }
    if qualification:
        request.update(qualification_fixture=True, **qualification)
    _write_exclusive_json(request_path, request)
    proof: dict[str, Any] = {
        "status": "owned_finalizer_not_accepted",
        "deadline_monotonic_s": deadline,
        "request_path": str(request_path),
        "watchdog_path": str(watchdog_path),
        "log_path": str(log_path),
        "unknown_usage": None
    }
    try:
        with log_path.open("xb") as log:
            with OwnedPausedProcess(deadline=deadline, result_path=watchdog_path) as owner:
                command = [sys.executable, "-B", str(Path(__file__).resolve()),
                           "--finalize", str(request_path)]
                process, guard = owner.start(command, env=os.environ.copy(), stdout=log)
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
                if guard_error is not None:
                    proof["guard_error"] = guard_error
        observed = time.monotonic()
        proof["observed_monotonic_s"] = observed
        proof["watchdog"] = json.loads(watchdog_path.read_text())
        published = json.loads(case_path.read_text()) if case_path.exists() else None
        finalization = published.get("finalization") if isinstance(published, dict) else None
        accepted = (
            proof.get("returncode") == 0
            and proof["watchdog"].get("status") == "writer_exited_before_deadline"
            and proof["watchdog"].get("signals") == []
            and isinstance(finalization, dict)
            and published.get("status") == expected_status
            and finalization.get("status") == "terminal_case_published_within_original_deadline"
            and finalization.get("process_identity") == proof.get("expected_identity")
            and finalization.get("absolute_deadline_monotonic_s") == deadline
            and finalization.get("terminal_monotonic_s", math.inf) < deadline
            and finalization.get("elapsed_s", math.inf) <= cap_s
            and published.get("wall_s", math.inf) <= published.get("cap_s", -math.inf)
            and observed < deadline)
        if not accepted:
            proof["published_status"] = published.get("status") if isinstance(published, dict) else None
            _failure_record(case_path, record, proof, "owned finalizer did not complete within original arm deadline")
            raise FinalizationFailure("owned finalizer result rejected", proof)
        return published
    except FinalizationFailure:
        raise
    except BaseException as exc:
        proof["observed_monotonic_s"] = time.monotonic()
        proof["error"] = repr(exc)
        if watchdog_path.exists():
            try:
                proof["watchdog"] = json.loads(watchdog_path.read_text())
            except Exception as watch_exc:
                proof["watchdog_error"] = repr(watch_exc)
        _failure_record(case_path, record, proof, repr(exc))
        raise FinalizationFailure("owned finalizer failed", proof) from exc
    finally:
        # Cleanup is scoped to this new transient control request.  If the child
        # never execs, no request is left that could be mistaken for evidence.
        if request_path.exists() and not request_path.is_symlink():
            request_path.unlink()


def verify_saved_finalization(case: dict[str, Any], watchdog: dict[str, Any], *, cap_s: float) -> dict[str, Any]:
    """Pure saved-evidence gate shared by qualification and later supervisors."""
    finalization = case.get("finalization")
    if case.get("status") != SUCCESS or not isinstance(finalization, dict):
        raise ValueError("saved case is not a completed bounded finalization")
    if case.get("cap_s") != cap_s or case.get("wall_s", math.inf) > cap_s:
        raise ValueError("saved terminal wall exceeds explicit spec cap")
    if finalization.get("cap_s") != cap_s or finalization.get("elapsed_s", math.inf) > cap_s:
        raise ValueError("saved finalizer elapsed exceeds explicit spec cap")
    if finalization.get("terminal_monotonic_s", math.inf) >= finalization.get("absolute_deadline_monotonic_s", -math.inf):
        raise ValueError("saved finalizer terminal timestamp is late")
    identity = finalization.get("process_identity")
    if not isinstance(identity, dict) or any(type(identity.get(key)) is not int
                                             for key in ("pid", "pid_start_ticks", "process_group", "session")):
        raise ValueError("saved finalizer process identity is absent or unknown")
    if identity["pid"] <= 0 or identity["process_group"] != identity["pid"] or identity["session"] != identity["pid"]:
        raise ValueError("saved finalizer was not an owned session leader")
    if watchdog.get("pid") != identity["pid"] or watchdog.get("pid_start_ticks") != identity["pid_start_ticks"]:
        raise ValueError("saved watchdog identity differs from finalizer")
    if watchdog.get("status") != "writer_exited_before_deadline" or watchdog.get("signals") != []:
        raise ValueError("saved watchdog does not prove unsignalled bounded completion")
    usage = finalization.get("finalizer_self_usage")
    if not isinstance(usage, dict) or any(not isinstance(usage.get(key), (int, float)) or usage[key] < 0
                                          for key in ("max_rss_kib", "user_s", "system_s")):
        raise ValueError("saved finalizer usage is unknown or invalid")
    if finalization.get("unknown_usage", "missing") is not None or finalization.get("unknown_usage_zero_filled") is not False:
        raise ValueError("unknown usage must remain explicit null, never zero-filled")
    return {"status": "verified_saved_bounded_finalization", "wall_s": case["wall_s"],
            "cap_s": cap_s, "process_identity": identity}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--finalize":
        raise SystemExit("owned finalizer requires --finalize REQUEST")
    raise SystemExit(child_finalize(Path(sys.argv[2])))
