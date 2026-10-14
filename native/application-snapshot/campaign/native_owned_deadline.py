"""Linux-only independent watchdog for one Popen-owned session leader.

It runs in a separate interpreter so a blocked observation read cannot extend
the writer's execution deadline. No arbitrary PID admission; no writer restart.
"""
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def identity(pid):
    """Linux PID generation plus group/session; refuse other session ownership."""
    stat = Path(f'/proc/{pid}/stat').read_text()
    fields = stat.rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ProcessLookupError('writer is already a zombie')
    return (int(fields[19]), os.getpgid(pid), os.getsid(pid))


def still_owned(pid, expected):
    try:
        return identity(pid) == expected
    except (FileNotFoundError, ProcessLookupError):
        return False


class OwnedDeadline:
    def __init__(self, process, *, deadline, result_path, grace_s=10.):
        if sys.platform != 'linux':
            raise RuntimeError('native watchdog requires Linux process-session ownership')
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError('future finite absolute deadline required')
        if not math.isfinite(grace_s) or not 0 < grace_s <= 10:
            raise ValueError('finite cleanup grace in (0,10] required')
        if process.poll() is not None:
            raise ValueError('writer already exited before watchdog admission')
        expected = identity(process.pid)
        if expected[1:] != (process.pid, process.pid):
            raise ValueError('writer must be owned new-session leader')
        self.process, self.expected = process, expected
        self.result_path = Path(result_path)
        if self.result_path.exists():
            raise FileExistsError(self.result_path)
        self.guard = subprocess.Popen(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--watch', str(process.pid),
             str(expected[0]), repr(deadline), repr(grace_s), str(self.result_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        ready, _, _ = select.select([self.guard.stdout], [], [], 3.)
        if not ready:
            self.guard.stdin.close()
            self.guard.wait(timeout=grace_s + 4)
            raise TimeoutError('independent watchdog failed readiness')
        line = self.guard.stdout.readline()
        if json.loads(line).get('status') != 'armed':
            raise RuntimeError('independent watchdog refused ownership')

    def check(self):
        if self.guard.poll() is not None:
            state = json.loads(self.result_path.read_text())
            if state['status'] != 'writer_exited_before_deadline':
                raise TimeoutError(('owned watchdog terminated', state))

    def close(self):
        # Never cancel a running writer's only independent cap.
        if self.process.poll() is None:
            raise RuntimeError('cannot disarm watchdog while writer is live')
        if self.guard.poll() is None:
            try:
                self.guard.stdin.write(b'C'); self.guard.stdin.flush()
            except BrokenPipeError:
                pass
        self.guard.stdin.close()
        self.guard.wait(timeout=14)
        self.guard.stdout.close(); self.guard.stderr.close()
        state = json.loads(self.result_path.read_text())
        if self.guard.returncode != 0:
            raise RuntimeError(('watchdog failed', state))
        return state


def watch(pid, ticks, deadline, grace, output):
    if sys.platform != 'linux' or not math.isfinite(deadline) or not 0 < grace <= 10:
        raise ValueError('invalid watchdog platform/bounds')
    expected = (ticks, pid, pid)
    state = dict(status='armed', pid=pid, pid_start_ticks=ticks,
                 deadline_monotonic_s=deadline, grace_s=grace, signals=[])
    # Exclusive creation binds this guardian to a fresh, caller-owned result.
    with output.open('x') as stream: json.dump(state, stream)
    print(json.dumps({'status': 'armed'}), flush=True)
    try:
        reason = 'absolute_deadline'
        while still_owned(pid, expected):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin.buffer], [], [], min(remaining, .05))
            if ready:
                command = os.read(sys.stdin.fileno(), 1)
                if command == b'C':
                    # The owner may only cancel after the writer actually exits.
                    if still_owned(pid, expected):
                        reason = 'invalid_live_disarm'; break
                    state['status'] = 'writer_exited_before_deadline'; return
                reason = 'owner_pipe_closed'; break
        else:
            state['status'] = 'writer_exited_before_deadline'; return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not still_owned(pid, expected): break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            state['signals'].append(dict(signal=int(sig), monotonic_s=time.monotonic()))
            if sig == signal.SIGTERM:
                stop = time.monotonic() + grace
                while still_owned(pid, expected) and time.monotonic() < stop:
                    time.sleep(min(.02, max(0., stop - time.monotonic())))
        state['status'] = 'owned_writer_terminated_' + reason
    except Exception as exc:
        state.update(status='watchdog_failed', error=repr(exc))
        raise
    finally:
        state['completed_monotonic_s'] = time.monotonic()
        output.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != '--watch':
        raise ValueError('watchdog requires its owned launcher')
    watch(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
          float(sys.argv[5]), Path(sys.argv[6]))
