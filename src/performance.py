"""Performance controls independent of estimator accuracy settings."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class PerformanceConfig:
    retrieval: str = "indexed"
    matching_backend: str = "cpu"
    cpu_optimizations: bool = True
    profile: bool = False

    def __post_init__(self):
        if self.retrieval not in ("indexed", "exhaustive"):
            raise ValueError("Retrieval must be indexed or exhaustive")
        if self.matching_backend not in ("cpu", "cuda", "auto"):
            raise ValueError("Matching backend must be cpu, cuda or auto")


class StageProfiler:
    """Thread-safe wall times; nested/background stages must not be summed."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.samples = {}
        self.counters = {}
        self.lock = Lock()

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.samples.setdefault(stage, []).append(perf_counter() - started)

    def call(self, stage, function, *args, **kwargs):
        with self.measure(stage):
            return function(*args, **kwargs)

    def count(self, name, value=1):
        if self.enabled:
            with self.lock:
                self.counters[name] = self.counters.get(name, 0) + value

    def report(self):
        with self.lock:
            return {
                "enabled": self.enabled,
                "timing_policy": "Nested wall times; background work overlaps tracking",
                "stages": {name: latency_stats(values) for name, values in self.samples.items()},
                "counters": self.counters.copy(),
            }


def latency_stats(values):
    values = np.asarray(values, float)
    return {
        "calls": len(values),
        "total_s": float(values.sum()),
        "median_ms": float(np.median(values) * 1000) if len(values) else None,
        "p95_ms": float(np.percentile(values, 95) * 1000) if len(values) else None,
    }


def profiled(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            return self.profiler.call(stage, function, self, *args, **kwargs)
        return wrapped
    return decorate
