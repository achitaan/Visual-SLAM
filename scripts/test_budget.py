"""Wall-clock accounting shared by development supervisors and evaluators."""
import json
import math
import time
from pathlib import Path


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        if seconds is not None and seconds <= 0:
            raise ValueError("Wall-clock budget must be positive")
        self.clock, self.seconds = clock, seconds
        self.started = clock()

    @property
    def remaining(self):
        return float("inf") if self.seconds is None else max(0, self.seconds - (self.clock() - self.started))

    def permits(self, estimated_seconds, reserve=0):
        return estimated_seconds + reserve < self.remaining


def write_json(path, value, *, retry_timeout=0.5, retry_interval=0.05,
               clock=None, sleep=None):
    """Atomically write finite JSON, retrying only transient replace locks.

    The retry budget covers at most half a second in aggregate. Optional clock
    and sleep callables make the transient and persistent-lock paths testable
    without real waits.
    """
    if (not isinstance(retry_timeout, (int, float))
            or not math.isfinite(retry_timeout)
            or not 0 <= retry_timeout <= 0.5):
        raise ValueError("retry_timeout must be between zero and 0.5 seconds")
    if (not isinstance(retry_interval, (int, float))
            or not math.isfinite(retry_interval) or retry_interval <= 0):
        raise ValueError("retry_interval must be finite and positive")
    clock = time.monotonic if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    deadline = clock() + retry_timeout
    max_retries = math.ceil(retry_timeout / retry_interval) if retry_timeout else 0
    retries = 0
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            remaining = deadline - clock()
            if remaining <= 0 or retries >= max_retries:
                raise
            sleep(min(retry_interval, remaining))
            retries += 1
