"""Cancellation-aware reader of the actual Linux native stage over WSL UNC.

The qualified decoder still owns pose/metadata validation.  Cancellation may
close a complete validated prefix, but never pads it to expected coverage and
never treats a post-writer destination copy as live evidence.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
import time

from native_app_follower import NativeAppFrame
from native_app_follower_delivery import NativeAppFollower as FrozenDeliveryFollower
from native_online_adapter import FIELDS, HEADER, OnlinePairDecoder


WSL_STAGE_ROOT = Path(r"\\wsl.localhost\Ubuntu\root\slam-native-20261009\output-local-qualified")
WSL_PROC_ROOT = Path(r"\\wsl.localhost\Ubuntu\proc")
SCHEMA = "native_application_admission_v3"


class LinuxWriterProbe:
    def __init__(self, run_id: str, journal: Path, *, proc_root: Path = WSL_PROC_ROOT):
        if not re.fullmatch(r"native_prototype_[a-z0-9_]+", run_id):
            raise ValueError("scoped native run required")
        self.run_id, self.journal, self.proc_root = run_id, Path(journal), Path(proc_root)
        self.admission = None

    def _load_admission(self):
        if self.admission is not None:
            return self.admission
        try:
            rows = [json.loads(line) for line in self.journal.read_text(encoding="utf8").splitlines()]
        except FileNotFoundError:
            return None
        candidates = [row for row in rows if row.get("schema") == SCHEMA
                      and row.get("event") == "process_admitted"
                      and row.get("run_id") == self.run_id
                      and row.get("role") == "native_writer"
                      and row.get("kind") == "owned_group"]
        if len(candidates) > 1:
            raise ValueError("one native writer admission required")
        if candidates:
            self.admission = candidates[0]
        return self.admission

    def alive(self) -> bool:
        admission = self._load_admission()
        if admission is None:
            return False
        try:
            fields = (self.proc_root / str(admission["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, OSError):
            return False
        return (fields[0] not in ("Z", "X")
                and int(fields[2]) == admission["process_group"]
                and int(fields[3]) == admission["session"]
                and int(fields[19]) == admission["pid_start_ticks"])


class NativeApplicationFollower(FrozenDeliveryFollower):
    def __init__(self, run_id: str, expected_frames: int, *, journal: Path,
                 cancelled, stage_root: Path = WSL_STAGE_ROOT, **kwargs):
        directory = Path(stage_root) / run_id
        super().__init__(directory, expected_frames, **kwargs)
        self.cancelled = cancelled
        self.writer_probe = LinuxWriterProbe(run_id, journal)
        self.live_observations = []
        self.partial_prefix_verified = False

    def frames(self, poll):
        if self.status != "new":
            raise ValueError("one owned stream per follower")
        self.status = "running"
        decoder = OnlinePairDecoder()
        offsets = {"poses": 0, "metadata": 0}
        pending_rows = deque()
        raw_pending = b""
        header = False
        deadline = self.clock() + self.timeout_s
        last = None
        try:
            while True:
                if self.clock() >= deadline:
                    raise TimeoutError("native application stream deadline exceeded")
                exit_code = poll()
                poses = self._prefix("online_poses.txt", deadline, poll)[offsets["poses"]:]
                metadata = self._prefix("online_frames.csv", deadline, poll)[offsets["metadata"]:]
                changed = bool(poses or metadata)
                while poses or metadata:
                    pose_chunk, metadata_chunk = poses[:65536], metadata[:65536]
                    decoder.append(poses=pose_chunk, metadata=metadata_chunk)
                    offsets["poses"] += len(pose_chunk)
                    offsets["metadata"] += len(metadata_chunk)
                    poses, metadata = poses[len(pose_chunk):], metadata[len(metadata_chunk):]
                    raw_pending += metadata_chunk
                    while b"\n" in raw_pending:
                        line, raw_pending = raw_pending.split(b"\n", 1)
                        if not header:
                            if line + b"\n" != HEADER:
                                raise ValueError("unexpected metadata header")
                            header = True
                        else:
                            parts = line.decode("ascii").split(",")
                            if len(parts) != len(FIELDS):
                                raise ValueError("wrong metadata width")
                            pending_rows.append(tuple(
                                (key, float(value) if key == "timestamp" else int(value))
                                for key, value in zip(FIELDS, parts)))
                    while (frame := decoder.pop_frame()) is not None:
                        actual = pending_rows.popleft()
                        if dict(actual)["original_id"] != frame.original_id:
                            raise ValueError("metadata and emitted pose association mismatch")
                        self._committed_metadata.append(actual)
                        last = NativeAppFrame(frame, actual)
                        self.live_observations.append({
                            "original_id": frame.original_id,
                            "windows_monotonic_s": time.monotonic(),
                            "linux_native_writer_alive": self.writer_probe.alive(),
                            "source": str(self.directory)})
                        yield last
                if exit_code is not None and not changed:
                    if self.cancelled():
                        actual_count = len(self._committed_metadata)
                        if actual_count:
                            decoder.finish(actual_count)
                        elif header or raw_pending or pending_rows or offsets["poses"]:
                            raise ValueError("cancelled native prefix has unpaired partial bytes")
                        self.partial_prefix_verified = True
                        self.status = "cancelled_validated_prefix_incomplete"
                        return
                    if exit_code != 0:
                        raise ValueError(f"native process exited with code {exit_code}")
                    if any(reader.raw_size != len(reader.prefix)
                           for reader in self._readers.values()):
                        raise ValueError("partial final native output")
                    decoder.finish(self.expected_frames)
                    if raw_pending or pending_rows:
                        raise ValueError("unconsumed final native metadata")
                    self._verify_lifecycle(last)
                    self.status = "complete_stream_lifecycle_not_accuracy_acceptance"
                    return
                if not changed:
                    self.sleep(self.poll_s)
        finally:
            if self.status not in ("cancelled_validated_prefix_incomplete",
                                   "complete_stream_lifecycle_not_accuracy_acceptance"):
                self.status = "incomplete"

    @property
    def causal_observation(self):
        return {"source": str(self.directory), "live_observations": list(self.live_observations),
                "positive_prefix_seen_while_linux_writer_alive": any(
                    row["linux_native_writer_alive"] for row in self.live_observations),
                "postwriter_destination_used_for_live_frames": False,
                "partial_prefix_verified": self.partial_prefix_verified,
                "delivery": self.observation}
