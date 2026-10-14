"""Require actual writer admission before the first WSL UNC stream operation.

No new errno is retried or ignored. All frozen decoder, prefix/identity,
partial cancellation, lifecycle and original absolute timeout checks remain.
The readiness gate prevents accessing the live stage during WSL startup;
it is not evidence that a writer is alive or that any frame was accepted.
"""
import math
from native_application_follower_v3 import NativeApplicationFollower as FrozenFollower


class NativeApplicationFollower(FrozenFollower):
    def _writer_admitted(self):
        admission=self.writer_probe._load_admission()
        if admission is None:return False
        if (any(type(admission[key]) is not int or admission[key]<=0
                for key in ('pid','pid_start_ticks','process_group','session'))
                or admission['process_group']!=admission['pid'] or admission['session']!=admission['pid']
                or not math.isfinite(admission['original_deadline_monotonic_s'])):
            raise ValueError('exact run-bound native writer admission required before stage read')
        return True

    def _prefix(self,name,deadline,poll):
        if not self._writer_admitted():
            if poll() is not None and not self.cancelled():
                raise ValueError('native writer exited without required stage admission')
            # Empty readiness has no decoded rows, source-file access or success
            # counters. The inherited loop keeps its original deadline/Stop.
            return b''
        return super()._prefix(name,deadline,poll)
