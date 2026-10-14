"""Snapshot I/O delivery successor; frozen causal/geometry checks preserved."""
import copy
import hashlib
import json
import math
import time

from native_live_snapshots import NativeSnapshotReader as FrozenSnapshotReader, _signature
from native_map_export import validate_snapshot
from native_observation_delivery import DeliveryBudget, ObservationDeferred


class NativeSnapshotReader(FrozenSnapshotReader):
    def __init__(self, *args, absolute_deadline, budget=None, process_poll=lambda: None,
                 read_snapshot=None, stat_file=None, scan_snapshots=None, **kwargs):
        if not math.isfinite(absolute_deadline):
            raise ValueError('finite absolute application deadline required')
        if budget is not None:
            if 'clock' in kwargs and kwargs['clock'] != budget.clock:
                raise ValueError('one clock required for shared delivery budget')
            kwargs['clock'] = budget.clock
        super().__init__(*args, **kwargs)
        if absolute_deadline <= self.clock():
            raise ValueError('future absolute application deadline required')
        self.absolute_deadline = absolute_deadline
        self.budget = budget or DeliveryBudget(clock=self.clock)
        self.process_poll = process_poll
        self._read_operation = read_snapshot or (lambda path: FrozenSnapshotReader._read(self, path))
        self._stat_operation = stat_file or (lambda path: path.stat())
        self._scan_operation = scan_snapshots or (lambda: list(self.directory.glob('snapshot-*.json')))

    def _io(self, operation, path=None):
        deadline = self.absolute_deadline
        if path is not None:
            # A pending file's original readiness clock never resets on retries.
            frame = int(path.stem.removeprefix('snapshot-'))
            if frame in self._pending:
                deadline = min(deadline, self._pending[frame] + self.pending_s)
        while True:
            try:
                return self.budget.call(operation, deadline=deadline, poll=self.process_poll)
            except ObservationDeferred:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError('snapshot absolute/readiness deadline exceeded')
                self.budget.sleep(min(self.budget.retry_s, remaining))

    def _read(self, path):
        return self._io(lambda: self._read_operation(path), path)

    def _stat(self, path):
        return self._io(lambda: self._stat_operation(path), path)

    def _scan(self):
        return self._io(self._scan_operation)

    def poll(self, rows, *, finished=False):
        if self.clock() >= self.absolute_deadline:
            self._failed = True
            raise TimeoutError('absolute snapshot application deadline exceeded')
        result = super().poll(rows, finished=finished)
        if self.clock() >= self.absolute_deadline:
            self._failed = True
            raise TimeoutError('absolute snapshot validation deadline exceeded')
        return result

    def _poll(self, rows, finished):
        if len(rows) < len(self._rows) or rows[:len(self._rows)] != self._rows:
            raise ValueError('committed native metadata changed or shrank')
        if [int(r['original_id']) for r in rows] != list(range(len(rows))):
            raise ValueError('original metadata row gap')
        self._rows = copy.deepcopy(rows)
        allowed = {f'snapshot-{f}.json' for f in self.expected}
        if any(p.name not in allowed for p in self._scan()):
            raise ValueError('unexpected snapshot ID or filename')

        for frame, accepted in self._accepted.items():
            path = self.directory / f'snapshot-{frame}.json'
            signature = _signature(self._stat(path))
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
