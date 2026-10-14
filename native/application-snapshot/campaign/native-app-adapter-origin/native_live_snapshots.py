"""Read complete online map copies without changing emitted pose history.

Input rows must come from committed native metadata, not inferred frame IDs.
The caller owns the worker and its total deadline. This reader never signals
or restarts it. Partial files have a separate finite readiness deadline.
Unchanged accepted files use stat checks while live; finalization rehashes all
bytes. Thus edits preserving stat metadata are detected by the final audit,
not necessarily at the instant of the edit.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import time

from native_map_export import validate_snapshot


def _signature(stat):
    # On the deployed Windows Python, fstat and path.stat can disagree on
    # ctime (write versus creation time) for the same stable handle. Identity,
    # length, mtime and the mandatory final content hash are portable here.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class NativeSnapshotReader:
    def __init__(self, directory, expected_frames, *, pending_s=10.,
                 max_bytes=64 * 1024 * 1024, clock=time.monotonic):
        frames = list(expected_frames)
        if (any(type(f) is not int or f < 0 for f in frames)
                or frames != sorted(set(frames))):
            raise ValueError('strictly increasing nonnegative snapshot IDs required')
        if not math.isfinite(pending_s) or pending_s <= 0:
            raise ValueError('finite positive snapshot readiness deadline required')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive integer snapshot byte limit required')
        self.directory = Path(directory)
        self.expected = tuple(frames)
        self.pending_s = pending_s
        self.max_bytes = max_bytes
        self.clock = clock
        self._rows = []
        self._accepted = {}
        self._pending = {}
        self._failed = False
        self._finished = False

    @property
    def accepted_frames(self):
        return list(self._accepted)

    def _read(self, path):
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if before.st_size > self.max_bytes:
                raise ValueError('snapshot byte limit exceeded')
            data = stream.read(self.max_bytes + 1)
            after = os.fstat(stream.fileno())
            visible = path.stat()
        if len(data) > self.max_bytes:
            raise ValueError('snapshot byte limit exceeded')
        if (before.st_dev, before.st_ino) != (visible.st_dev, visible.st_ino):
            raise ValueError('snapshot replaced during read')
        if after.st_size < before.st_size:
            raise ValueError('snapshot truncated during read')
        # A growing partial JSON is not an accepted map. Revisit it on a later
        # poll only within the original readiness deadline.
        if _signature(before) != _signature(after) or _signature(after) != _signature(visible):
            return None, None
        return data, _signature(after)

    def _pending_or_fail(self, frame, finished):
        if finished:
            raise ValueError(f'incomplete final snapshot {frame}')
        start = self._pending.setdefault(frame, self.clock())
        if self.clock() - start >= self.pending_s:
            raise TimeoutError(f'snapshot {frame} readiness deadline exceeded')

    def poll(self, rows, *, finished=False):
        if self._failed or self._finished:
            raise ValueError('snapshot reader is failed or finalized')
        try:
            return self._poll(rows, finished)
        except Exception:
            self._failed = True
            raise

    def _poll(self, rows, finished):
        if len(rows) < len(self._rows) or rows[:len(self._rows)] != self._rows:
            raise ValueError('committed native metadata changed or shrank')
        if [int(r['original_id']) for r in rows] != list(range(len(rows))):
            raise ValueError('original metadata row gap')
        self._rows = copy.deepcopy(rows)
        allowed = {f'snapshot-{f}.json' for f in self.expected}
        if any(p.name not in allowed for p in self.directory.glob('snapshot-*.json')):
            raise ValueError('unexpected snapshot ID or filename')

        for frame, accepted in self._accepted.items():
            path = self.directory / f'snapshot-{frame}.json'
            signature = _signature(path.stat())
            if signature[:2] != accepted['signature'][:2]:
                raise ValueError('accepted snapshot replaced')
            if finished or signature != accepted['signature']:
                data, stable = self._read(path)
                if (stable is None or hashlib.sha256(data).hexdigest() != accepted['sha256']):
                    raise ValueError('accepted snapshot bytes changed')
                accepted['signature'] = stable
            if frame + 1 < len(rows) and accepted['completion_ns'] > int(rows[frame + 1]['start_ns']):
                raise ValueError('snapshot includes future input interval')

        delivered = []
        for frame in self.expected:
            if frame in self._accepted:
                continue
            if frame >= len(rows):
                if finished:
                    raise ValueError('snapshot outside final committed coverage')
                break
            path = self.directory / f'snapshot-{frame}.json'
            self._pending_or_fail(frame, False)
            try:
                data, signature = self._read(path)
            except FileNotFoundError:
                self._pending_or_fail(frame, finished)
                break
            if data is None or not data.endswith(b'\n'):
                self._pending_or_fail(frame, finished)
                break
            snapshot = json.loads(data)
            if snapshot.get('original_id') != frame:
                raise ValueError('snapshot filename and original ID disagree')
            validate_snapshot(snapshot, rows)
            self._accepted[frame] = dict(signature=signature,
                                         sha256=hashlib.sha256(data).hexdigest(),
                                         completion_ns=snapshot['completion_ns'])
            del self._pending[frame]
            delivered.append(snapshot)
        if finished:
            if self.accepted_frames != list(self.expected):
                raise ValueError('missing final snapshot coverage')
            self._finished = True
        return delivered
