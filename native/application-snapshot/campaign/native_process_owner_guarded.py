"""Linux native launch gate: no estimator executes before its independent cap.

The gate and native executable share the same PID, process group and session.
The caller must retain this context across every post-spawn evidence operation.
"""
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from native_owned_deadline import OwnedDeadline


class OwnedPausedProcess:
    def __init__(self, *, deadline, result_path):
        if sys.platform != 'linux':
            raise RuntimeError('owned launch gate requires Linux')
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError('future finite absolute deadline required')
        self.deadline = deadline
        self.result_path = Path(result_path)
        self.process = None
        self.guard = None
        self._write_fd = None
        self._released = False

    def __enter__(self):
        return self

    def start(self, command, *, env, stdout, guard_factory=OwnedDeadline):
        if self.process is not None:
            raise RuntimeError('one owned spawn only')
        read_fd, self._write_fd = os.pipe()
        try:
            # If admission or the owner dies, this child exits without exec.
            self.process = subprocess.Popen(
                [sys.executable, '-B', str(Path(__file__).resolve()), '--gate',
                 str(read_fd), repr(self.deadline), '--', *command],
                env=env, stdout=stdout, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(read_fd,))
            os.close(read_fd)
            read_fd = None
            self.guard = guard_factory(self.process, deadline=self.deadline,
                                       result_path=self.result_path)
            return self.process, self.guard
        except BaseException:
            self.close()
            raise
        finally:
            if read_fd is not None:
                os.close(read_fd)

    def release(self):
        if self.guard is None or self.process is None or self._released:
            raise RuntimeError('one release after watchdog admission required')
        self.guard.check()
        if self.process.poll() is not None or time.monotonic() >= self.deadline:
            raise TimeoutError('native gate deadline or child exit before release')
        try:
            if os.write(self._write_fd, b'G') != 1:
                raise RuntimeError('incomplete gate release')
            self._released = True
        finally:
            self._close_pipe()

    def _close_pipe(self):
        if self._write_fd is not None:
            os.close(self._write_fd)
            self._write_fd = None

    def close(self):
        self._close_pipe()
        p = self.process
        if p is not None and p.poll() is None:
            # p is our unreaped direct child, admitted as its own session leader.
            # A zombie cannot have its PID reused before our wait/poll reaps it.
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=4)
        if self.guard is not None:
            self.guard.close()
            self.guard = None

    def __exit__(self, typ, value, traceback):
        self.close()
        return False


def gated_exec(fd, deadline, command):
    if sys.platform != 'linux' or not command or not math.isfinite(deadline):
        raise ValueError('invalid owned native gate')
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 124
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready or os.read(fd, 1) != b'G':
            return 124
        if time.monotonic() >= deadline:
            return 124
    finally:
        os.close(fd)
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    if len(sys.argv) < 6 or sys.argv[1] != '--gate' or sys.argv[4] != '--':
        raise ValueError('native gate requires its owned launcher')
    raise SystemExit(gated_exec(int(sys.argv[2]), float(sys.argv[3]), sys.argv[5:]))
