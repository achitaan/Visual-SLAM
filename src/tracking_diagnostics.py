"""Bounded capture-only tracking trace writer.

This module stores owned JSON snapshots emitted by SharedSlam. It does not
invoke matching, stereo sampling, pose fitting, or mutate estimator state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock

import numpy as np


_MAX_FRAMES = 7
_PHASES = (
    "frame_start_state",
    "solve_probe",
    "solve_skipped",
    "selected_pre_keyframe",
    "accepted_pre_ba",
    "cohort_map_before_ba",
    "cohort_map_after_ba",
    "frame_end",
)


class TrackingDiagnosticsWriter:
    """Persist a small allowlisted set of immutable tracking evidence traces."""

    def __init__(self, output_dir, *, frames, max_rows_per_pool=4096,
                 max_state_rows=20000, max_events_per_frame=256):
        self.output_dir = Path(output_dir)
        values = tuple(frames)
        if (not values or len(values) > _MAX_FRAMES
                or any(not isinstance(value, (int, np.integer))
                       or isinstance(value, (bool, np.bool_))
                       or int(value) < 0
                       for value in values)
                or len({int(value) for value in values}) != len(values)):
            raise ValueError("tracking frames must be distinct nonnegative IDs (max 7)")
        if (not isinstance(max_rows_per_pool, int) or max_rows_per_pool < 1
                or not isinstance(max_state_rows, int) or max_state_rows < 1
                or not isinstance(max_events_per_frame, int) or max_events_per_frame < 1):
            raise ValueError("tracking diagnostic bounds must be positive integers")
        self.frames = tuple(sorted(int(value) for value in values))
        self.max_rows_per_pool = int(max_rows_per_pool)
        self.max_state_rows = int(max_state_rows)
        self.max_events_per_frame = int(max_events_per_frame)
        self._lock = RLock()
        self._records = {
            frame: {
                "frame": frame,
                "status": "pending",
                "tracking_status": None,
                "phases": [],
                "events": [],
                "errors": [],
                "truncated": False,
                "provenance_complete": True,
            }
            for frame in self.frames
        }
        self._manifest_errors = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_capture(self, frame):
        try:
            return int(frame) in self._records
        except (TypeError, ValueError, OverflowError):
            return False

    def _owned_value(self, value, invalid, path="$", depth=0):
        if depth > 32:
            invalid.append(path)
            return None
        if isinstance(value, np.ndarray):
            return self._owned_value(value.tolist(), invalid, path, depth + 1)
        if isinstance(value, np.generic):
            return self._owned_value(value.item(), invalid, path, depth + 1)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            invalid.append(path)
            return None
        if isinstance(value, dict):
            return {
                str(key): self._owned_value(item, invalid, f"{path}.{key}", depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._owned_value(item, invalid, f"{path}[{index}]", depth + 1)
                for index, item in enumerate(value)
            ]
        invalid.append(path)
        return None

    def _bound_event(self, phase, payload):
        payload = dict(payload)
        rows = payload.get("rows")
        if isinstance(rows, list):
            state_event = phase in {
                "cohort_map_before_ba", "cohort_map_after_ba", "frame_start_state",
                "selected_pre_keyframe", "accepted_pre_ba",
            }
            cap = self.max_state_rows if state_event else self.max_rows_per_pool
            count = int(payload.get("input_row_count", len(rows)))
            payload["input_row_count"] = count
            if len(rows) > cap:
                payload["rows"] = rows[:cap]
            stored = len(payload["rows"])
            payload["stored_row_count"] = stored
            complete_key = "snapshot_complete" if state_event else "pool_complete"
            was_complete = payload.get(complete_key, True)
            payload[complete_key] = bool(was_complete and count <= cap and stored == count)
            if state_event:
                payload["snapshot_truncated"] = not payload[complete_key]
            else:
                payload["consumption_unknown"] = not payload[complete_key]
        return payload

    def record_event(self, frame, phase, payload):
        """Append an owned event; all I/O/serialization failures stay diagnostic-only."""
        try:
            frame = int(frame)
            if frame not in self._records or phase not in _PHASES:
                return False
            if not isinstance(payload, dict):
                raise TypeError("tracking event payload must be a mapping")
            invalid = []
            owned = self._owned_value(payload, invalid)
            if invalid:
                owned["invalid_fields"] = sorted(set(invalid))
                owned["snapshot_valid"] = False
            owned = self._bound_event(phase, owned)
            with self._lock:
                record = self._records[frame]
                if len(record["events"]) >= self.max_events_per_frame:
                    record["truncated"] = True
                    record["errors"].append({
                        "phase": phase, "error_type": "EventLimitExceeded",
                    })
                    return False
                event = {"sequence": len(record["events"]), "phase": phase, **owned}
                record["events"].append(event)
                if phase not in record["phases"]:
                    record["phases"].append(phase)
                if owned.get("snapshot_truncated") or owned.get("pool_complete") is False:
                    record["truncated"] = True
                if (owned.get("consumption_unknown")
                        or owned.get("fit_consumption_complete") is False
                        or owned.get("reference_pool_status") == "unknown_uninstrumented"
                        or owned.get("snapshot_valid") is False):
                    record["provenance_complete"] = False
            return True
        except Exception as error:
            try:
                with self._lock:
                    record = self._records.get(int(frame))
                    if record is not None:
                        record["errors"].append({
                            "phase": str(phase), "error_type": type(error).__name__,
                        })
            except Exception:
                pass
            return False

    def _atomic_json(self, path, payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return hashlib.sha256(encoded).hexdigest()

    def finish_frame(self, frame, *, tracking_status=None):
        try:
            frame = int(frame)
            if frame not in self._records:
                return {"frame": frame, "status": "unselected"}
            with self._lock:
                record = self._records[frame]
                record["tracking_status"] = (
                    None if tracking_status is None else str(tracking_status)
                )
                required = {"frame_start_state", "selected_pre_keyframe",
                            "accepted_pre_ba", "frame_end"}
                complete = required.issubset(record["phases"])
                record["status"] = (
                    "complete" if complete and not record["errors"] and not record["truncated"]
                    else "incomplete"
                )
                file_payload = {
                    "schema": "future_tracking_trace_v1",
                    "frame": frame,
                    "status": record["status"],
                    "tracking_status": record["tracking_status"],
                    "phases": list(record["phases"]),
                    "events": list(record["events"]),
                    "errors": list(record["errors"]),
                    "truncated": bool(record["truncated"]),
                }
            path = self.output_dir / "frames" / f"frame_{frame:04d}.json"
            digest = self._atomic_json(path, file_payload)
            with self._lock:
                record = self._records[frame]
                record["path"] = path.relative_to(self.output_dir).as_posix()
                record["sha256"] = digest
                if record["errors"]:
                    record["status"] = "incomplete"
                return self._manifest_frame(record)
        except Exception as error:
            try:
                with self._lock:
                    record = self._records.get(int(frame))
                    if record is not None:
                        record["status"] = "incomplete"
                        record["errors"].append({
                            "phase": "finish", "error_type": type(error).__name__,
                        })
            except Exception:
                pass
            return {"frame": frame, "status": "incomplete",
                    "error_type": type(error).__name__}

    def mark_skipped(self, frame, reason):
        try:
            frame = int(frame)
            if frame not in self._records:
                return False
            with self._lock:
                record = self._records[frame]
                if record["status"] != "pending" or record["events"]:
                    return False
                record["status"] = "skipped"
                record["skip_reason"] = str(reason)
            return True
        except Exception:
            return False

    def record_error(self, frame, phase, error_type):
        try:
            frame = int(frame)
            with self._lock:
                record = self._records.get(frame)
                if record is None:
                    return False
                record["errors"].append({
                    "phase": str(phase), "error_type": str(error_type),
                })
                record["provenance_complete"] = False
            return True
        except Exception:
            return False

    @staticmethod
    def _manifest_frame(record):
        return {
            "frame": int(record["frame"]),
            "status": str(record["status"]),
            "tracking_status": record.get("tracking_status"),
            "phases": list(record.get("phases", [])),
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "event_count": len(record.get("events", [])),
            "errors": list(record.get("errors", [])),
            "truncated": bool(record.get("truncated", False)),
            "provenance_complete": bool(record.get("provenance_complete", True)),
            "skip_reason": record.get("skip_reason"),
        }

    def frame_payload(self, frame):
        """Return a detached in-memory frame payload for callers/tests."""
        try:
            frame = int(frame)
            with self._lock:
                record = self._records.get(frame)
                if record is None:
                    return None
                return {
                    "schema": "future_tracking_trace_v1",
                    "frame": frame,
                    "status": record["status"],
                    "tracking_status": record["tracking_status"],
                    "phases": list(record["phases"]),
                    "events": json.loads(json.dumps(record["events"])),
                    "errors": list(record["errors"]),
                    "truncated": bool(record["truncated"]),
                    "provenance_complete": bool(record["provenance_complete"]),
                }
        except Exception:
            return None

    def manifest(self):
        with self._lock:
            payload = {
                "schema": "future_tracking_diagnostics_manifest_v1",
                "enabled": True,
                "selected_frames": list(self.frames),
                "max_rows_per_pool": self.max_rows_per_pool,
                "max_state_rows": self.max_state_rows,
                "max_events_per_frame": self.max_events_per_frame,
                "frames": [self._manifest_frame(self._records[frame])
                           for frame in self.frames],
                "errors": list(self._manifest_errors),
            }
        try:
            digest = self._atomic_json(self.output_dir / "manifest.json", payload)
            payload["manifest_sha256"] = digest
        except Exception as error:
            payload["errors"].append({
                "phase": "manifest", "error_type": type(error).__name__,
            })
        return payload
