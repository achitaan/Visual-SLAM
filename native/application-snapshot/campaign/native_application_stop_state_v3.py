"""Atomic Windows spawn-versus-Stop state for one native application run."""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

from native_application_stop_owner_v2 import validate_control


class ApplicationStopState:
    def __init__(self, run_id: str, expected_frames: int):
        if not re.fullmatch(r"native_prototype_[a-z0-9_]+", run_id):
            raise ValueError("scoped native application run ID required")
        if type(expected_frames) is not int or expected_frames <= 0:
            raise ValueError("positive expected frame count required")
        self.run_id = run_id
        self.expected_frames = expected_frames
        self.capability = secrets.token_hex(32)
        self.digest = hashlib.sha256(self.capability.encode("ascii")).hexdigest()
        self._lock = threading.Lock()
        self.cancel_requested = threading.Event()
        self.service_stop = threading.Event()
        self.phase = "prepared"
        self.control_error: str | None = None
        self.stop_requested_monotonic_s: float | None = None
        self.last_status = "prepared_no_process"
        self._worker_control: Callable[[bytes], None] | None = None

    def spawn_owned(self, factory: Callable[[], Any]) -> Any | None:
        """Create and attach the direct child while Stop is excluded.

        If Stop obtains the lock first, phase is cancelled and factory is never
        called.  If spawn obtains it first, the pipe callback is retained before
        a racing Stop may observe the running phase.
        """
        if not callable(factory):
            raise TypeError("owned process factory required")
        with self._lock:
            if self.phase == "cancelled":
                return None
            if self.phase != "prepared":
                raise RuntimeError("one application spawn only")
            self.phase = "spawning"
            try:
                process = factory()
                if process.stdin is None:
                    raise RuntimeError("owned control pipe was not retained")

                def write_control(payload: bytes) -> None:
                    process.stdin.write(payload)
                    process.stdin.flush()

                self._worker_control = write_control
                self.phase = "running"
                self.last_status = "running_cleanup_not_applicable"
                return process
            except BaseException:
                self.phase = "failed"
                self.last_status = "spawn_failed_no_completion"
                self._worker_control = None
                raise

    def receive(self, payload: bytes) -> dict[str, Any]:
        if not validate_control(payload, self.run_id, self.digest):
            return self.snapshot(status="rejected_control")
        with self._lock:
            if self.phase == "completed":
                self.service_stop.set()
                self.last_status = "completed_service_stop_requested"
                return self.snapshot_locked()
            if self.phase == "cancelled":
                value = self.snapshot_locked()
                value["status"] = "already_cancelled_incomplete"
                return value
            if self.phase == "stopping":
                value = self.snapshot_locked()
                value["status"] = "already_stopping_cleanup_unconfirmed"
                return value
            if self.phase == "failed":
                value = self.snapshot_locked()
                value["status"] = "already_failed_incomplete"
                return value
            if self.phase == "prepared":
                self.phase = "cancelled"
                self.cancel_requested.set()
                self.stop_requested_monotonic_s = time.monotonic()
                self.last_status = "cancelled_before_spawn_incomplete"
                return self.snapshot_locked()
            if self.phase != "running":
                raise RuntimeError("spawn phase may not escape the atomic state lock")
            self.phase = "stopping"
            self.cancel_requested.set()
            self.stop_requested_monotonic_s = time.monotonic()
            self.last_status = "stop_requested_cleanup_unconfirmed"
            writer = self._worker_control
        try:
            writer(payload + b"\n")
        except Exception as exc:
            with self._lock:
                self.control_error = type(exc).__name__
                self.last_status = "stop_control_error_cleanup_unconfirmed"
                return self.snapshot_locked()
        return self.snapshot()

    def acknowledge_owned_exit(self, audit: dict[str, Any]) -> None:
        if (not isinstance(audit, dict)
                or audit.get("status") != "sealed_all_admitted_groups_and_token_orphans_exited"
                or audit.get("run_id") != self.run_id
                or audit.get("all_owned_exited") is not True
                or audit.get("seal", {}).get("all_owned_exited") is not True):
            raise ValueError("sealed current-run whole-exit audit required")
        with self._lock:
            if self.phase != "stopping":
                raise RuntimeError("cleanup acknowledgement requires a Stop request")
            self.phase = "cancelled"
            self.last_status = "cancelled_cleanup_confirmed_incomplete"
            self._worker_control = None

    def mark_complete(self) -> bool:
        with self._lock:
            if self.phase in ("stopping", "cancelled"):
                return False
            if self.phase != "running":
                raise RuntimeError("completion requires current running application")
            self.phase = "completed"
            self.last_status = "completed_run_evidence_unchanged"
            self._worker_control = None
            return True

    def mark_failed(self, status: str = "failed_incomplete_no_acceptance") -> None:
        with self._lock:
            if self.phase not in ("stopping", "cancelled"):
                self.phase = "failed"
                self.last_status = status
            self._worker_control = None

    def snapshot_locked(self) -> dict[str, Any]:
        return {"schema": "native_application_control_v1", "type": "control_state",
                "run_id": self.run_id, "expected_frames": self.expected_frames,
                "phase": self.phase, "status": self.last_status,
                "cleanup_confirmed": self.phase == "cancelled",
                "incomplete": self.phase in ("cancelled", "stopping", "failed"),
                "control_error": self.control_error, "restart_allowed": False,
                "stop_requested_monotonic_s": self.stop_requested_monotonic_s,
                "clear_is_display_only": True}

    def snapshot(self, *, status: str | None = None) -> dict[str, Any]:
        with self._lock:
            value = self.snapshot_locked()
        if status is not None:
            value["status"] = status
        return value

    def hello(self) -> dict[str, Any]:
        value = self.snapshot()
        value.update(type="control_hello", capability=self.capability,
                     hello_before_native_frames=True)
        return value
