"""Wall-clock accounting shared by development supervisors and evaluators."""
import json
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
