"""Bounded, JSON-only snapshots for local bundle-adjustment diagnostics.

The writer is deliberately separate from the estimator. It accepts only owned
JSON-compatible payloads, stores each phase atomically, and never puts absolute
paths in its manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


_PHASES = ("prepared", "solved", "finished")


class BundleDiagnosticsWriter:
    """Write a bounded set of per-frame BA diagnostic snapshots.

    ``frames`` is an explicit allowlist. An empty allowlist captures nothing;
    ``max_frames`` and ``max_snapshots`` keep accidental large captures bounded.
    """

    def __init__(self, output_dir, *, frames=(), max_frames=32, max_snapshots=96):
        self.output_dir = Path(output_dir)
        self.frames = tuple(int(frame) for frame in frames)
        if any(frame < 0 for frame in self.frames) or len(set(self.frames)) != len(self.frames):
            raise ValueError("diagnostic frames must be unique nonnegative integers")
        self.max_frames = int(max_frames)
        self.max_snapshots = int(max_snapshots)
        if self.max_frames <= 0 or self.max_snapshots <= 0:
            raise ValueError("diagnostic capture limits must be positive")
        if len(self.frames) > self.max_frames:
            raise ValueError("requested diagnostic frames exceed max_frames")
        if len(self.frames) * len(_PHASES) > self.max_snapshots:
            raise ValueError("requested diagnostic phases exceed max_snapshots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames = {
            frame: {
                "frame": frame,
                "status": "pending",
                "phases": {},
                "errors": [],
            }
            for frame in self.frames
        }

    def should_capture(self, frame):
        """Return whether ``frame`` is included in this explicit capture."""
        try:
            frame = int(frame)
        except (TypeError, ValueError, OverflowError):
            return False
        return frame in self._frames

    def _record_error(self, frame, phase, error):
        record = self._frames[frame]
        record["errors"].append({
            "phase": str(phase),
            "error_type": type(error).__name__,
        })
        record["status"] = "error"

    @staticmethod
    def _json_bytes(payload):
        return (json.dumps(payload, allow_nan=False, sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

    def _atomic_write(self, relative_path, payload):
        data = self._json_bytes(payload)
        destination = self.output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"diagnostic snapshot already exists: {relative_path}")
        fd, temporary = tempfile.mkstemp(prefix=".bundle-diagnostics-", suffix=".tmp",
                                         dir=str(self.output_dir))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # A snapshot path is unique to one requested frame/phase. Refuse
            # stale output from an earlier run rather than silently replace it.
            os.link(temporary, destination)
            os.unlink(temporary)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return hashlib.sha256(data).hexdigest()

    def emit(self, frame, phase, payload, image_size=None):
        """Atomically write one phase and return its relative path.

        Serialization and filesystem failures are recorded and re-raised so a
        caller can isolate them from estimator decisions while retaining an
        auditable error in :meth:`manifest`.
        """
        try:
            frame = int(frame)
        except (TypeError, ValueError, OverflowError):
            return None
        if frame not in self._frames:
            return None
        if phase not in _PHASES:
            error = ValueError("unknown diagnostic phase")
            self._record_error(frame, phase, error)
            raise error
        record = self._frames[frame]
        if phase in record["phases"]:
            error = ValueError("diagnostic phase already emitted for frame")
            self._record_error(frame, phase, error)
            raise error

        if isinstance(payload, dict) and payload.get("input_valid") is False:
            error = ValueError("prepared diagnostic input contains invalid fields")
            self._record_error(frame, phase, error)
            raise error
        relative_path = f"frame-{frame:05d}/{phase}.json"
        document = {
            "schema_version": 1,
            "frame": frame,
            "phase": phase,
            "image_size": None if image_size is None else [int(v) for v in image_size],
            "payload": payload,
        }
        try:
            digest = self._atomic_write(relative_path, document)
        except Exception as error:
            self._record_error(frame, phase, error)
            raise
        record["phases"][phase] = {
            "path": relative_path,
            "sha256": digest,
            "status": "written",
        }
        if record["errors"]:
            record["status"] = "error"
        elif set(record["phases"]) == set(_PHASES):
            record["status"] = "complete"
        elif record["status"] != "skipped":
            record["status"] = "partial"
        if image_size is not None:
            record["image_size"] = [int(v) for v in image_size]
        return relative_path

    def mark_skipped(self, frame, reason):
        """Mark a selected frame as not requiring a BA capture, if untouched."""
        try:
            frame = int(frame)
        except (TypeError, ValueError, OverflowError):
            return False
        record = self._frames.get(frame)
        if record is None or record["errors"] or record["status"] not in ("pending", "partial"):
            return False
        if "prepared" in record["phases"] or "solved" in record["phases"]:
            return False
        record["status"] = "skipped"
        record["reason"] = str(reason)
        return True

    def manifest(self):
        """Return detached manifest data with relative links only.

        The owning run/evaluation report persists this object. Keeping this
        method free of I/O prevents diagnostic manifest failures from breaking
        estimator exports.
        """
        records = []
        for frame in self.frames:
            record = self._frames[frame]
            detached = {
                "frame": int(frame),
                "status": str(record["status"]),
                "phases": {
                    phase: dict(details)
                    for phase, details in record["phases"].items()
                },
                "errors": [dict(error) for error in record["errors"]],
            }
            if "reason" in record:
                detached["reason"] = record["reason"]
            if "image_size" in record:
                detached["image_size"] = list(record["image_size"])
            records.append(detached)
        result = {
            "schema_version": 1,
            "enabled": True,
            "selected_frames": list(self.frames),
            "max_frames": self.max_frames,
            "max_snapshots": self.max_snapshots,
            "frames": records,
        }
        return result
