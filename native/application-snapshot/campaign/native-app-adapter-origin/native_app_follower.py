"""Application successor: owned actual metadata plus qualified bounded reads.

The original decoder and lifecycle verifier remain unchanged. A shared error
budget covers both streams. Caller owns the process; no restart or signal is
performed here. Successful ingestion alone is not optimizer acceptance.
"""
from collections import deque
from dataclasses import dataclass
from native_observation import CommittedPrefixObserver, ObservationBudget
from native_online_adapter import OnlinePairDecoder, HEADER, FIELDS, EmittedFrame
from native_run_follower import NativeRunFollower


@dataclass(frozen=True)
class NativeAppFrame:
    frame: EmittedFrame
    metadata: tuple

    def __getattr__(self, name):
        return getattr(self.frame, name)

    @property
    def metadata_row(self):
        return dict(self.metadata)


class NativeAppFollower(NativeRunFollower):
    def __init__(self, *args, read_snapshot=None, **kwargs):
        super().__init__(*args, **kwargs)
        options = {} if read_snapshot is None else dict(read_snapshot=read_snapshot)
        self._readers = {name:CommittedPrefixObserver(**options) for name in
                         ('online_poses.txt','online_frames.csv')}
        self._budget = ObservationBudget(clock=self.clock, sleep=self.sleep)
        for reader in self._readers.values():
            reader.budget = self._budget
        self._committed_metadata = []

    @property
    def metadata_rows(self):
        return [dict(row) for row in self._committed_metadata]

    @property
    def observation(self):
        return dict(transient_errors=self._budget.transient_errors,
                    events=list(self._budget.events),
                    streams={name:r.statistics() for name,r in self._readers.items()})

    def _prefix(self, name, deadline, poll):
        reader = self._readers[name]
        try:
            return reader.read_prefix(self.directory/name,deadline=deadline,poll=poll)
        except FileNotFoundError:
            if reader.identity is not None:
                raise ValueError('native output disappeared')
            return b''

    def frames(self, poll):
        if self.status != 'new':
            raise ValueError('one owned stream per follower')
        self.status = 'running'
        decoder = OnlinePairDecoder()
        offsets = dict(poses=0,metadata=0)
        pending_rows = deque()
        raw_pending = b''
        header = False
        deadline = self.clock()+self.timeout_s
        last = None
        try:
            while True:
                if self.clock() >= deadline:
                    raise TimeoutError('native app stream deadline exceeded')
                exit_code = poll()  # Sample before final read, then drain.
                poses = self._prefix('online_poses.txt',deadline,poll)[offsets['poses']:]
                metadata = self._prefix('online_frames.csv',deadline,poll)[offsets['metadata']:]
                changed = bool(poses or metadata)
                while poses or metadata:
                    pc,mc = poses[:65536],metadata[:65536]
                    decoder.append(poses=pc,metadata=mc)
                    offsets['poses'] += len(pc);offsets['metadata'] += len(mc)
                    poses,metadata = poses[len(pc):],metadata[len(mc):]
                    raw_pending += mc
                    while b'\n' in raw_pending:
                        line,raw_pending = raw_pending.split(b'\n',1)
                        if not header:
                            if line+b'\n' != HEADER:
                                raise ValueError('unexpected metadata header')
                            header = True
                        else:
                            parts = line.decode('ascii').split(',')
                            if len(parts) != len(FIELDS):
                                raise ValueError('wrong metadata width')
                            pending_rows.append(tuple((k,float(v) if k=='timestamp' else int(v))
                                                      for k,v in zip(FIELDS,parts)))
                    while (frame := decoder.pop_frame()) is not None:
                        actual = pending_rows.popleft()
                        if dict(actual)['original_id'] != frame.original_id:
                            raise ValueError('metadata and emitted pose association mismatch')
                        self._committed_metadata.append(actual)
                        last = NativeAppFrame(frame,actual)
                        yield last
                if exit_code is not None:
                    if exit_code != 0:
                        raise ValueError(f'native process exited with code {exit_code}')
                    if not changed:
                        if any(r.raw_size != len(r.prefix) for r in self._readers.values()):
                            raise ValueError('partial final native output')
                        decoder.finish(self.expected_frames)
                        if raw_pending or pending_rows:
                            raise ValueError('unconsumed final native metadata')
                        self._verify_lifecycle(last)
                        self.status = 'complete_stream_lifecycle_not_accuracy_acceptance'
                        return
                if not changed:
                    self.sleep(self.poll_s)
        finally:
            if self.status != 'complete_stream_lifecycle_not_accuracy_acceptance':
                self.status = 'incomplete'
