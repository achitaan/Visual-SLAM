"""Inclusive stage timings; nested stages must not be added together."""
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from threading import Lock

import numpy as np


class StageProfile:
    """Thread-safe stage totals with optional bounded latency samples.

    ``report`` retains the original compact API. ``detailed_report`` adds
    latency summaries and counters for opt-in performance runs.
    """

    def __init__(self, detailed=False, enabled=True, sample_latencies=None, max_samples=512):
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.enabled = bool(enabled)
        self.detailed = bool(detailed)
        self.sample_latencies = self.detailed if sample_latencies is None else bool(sample_latencies)
        self.max_samples = int(max_samples)
        self.values = defaultdict(lambda: [0, 0.0])
        self.samples = defaultdict(lambda: deque(maxlen=self.max_samples))
        self.counters = {}
        self.lock = Lock()

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            with self.lock:
                self.values[name][0] += 1
                self.values[name][1] += elapsed
                if self.sample_latencies:
                    self.samples[name].append(elapsed)

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def count(self, name, value=1):
        if not self.enabled:
            return
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def wrap(self, name, method):
        @wraps(method)
        def timed(*args, **kwargs):
            with self.measure(name):
                return method(*args, **kwargs)
        return timed

    def report(self):
        with self.lock:
            return {
                name: {"calls": n, "inclusive_seconds": seconds}
                for name, (n, seconds) in self.values.items()
            }

    def detailed_report(self):
        """Return performance summaries; sample storage is capped per stage."""
        with self.lock:
            stages = {}
            for name, (calls, seconds) in self.values.items():
                samples = np.asarray(self.samples.get(name, ()), dtype=float)
                stages[name] = {
                    "calls": calls,
                    "inclusive_seconds": seconds,
                    "total_s": seconds,
                    "sample_count": int(len(samples)),
                    "median_ms": float(np.median(samples) * 1000) if len(samples) else None,
                    "p95_ms": float(np.percentile(samples, 95) * 1000) if len(samples) else None,
                }
            return {
                "enabled": self.enabled,
                "timing_policy": "Nested wall times; background work overlaps tracking",
                "stages": stages,
                "counters": self.counters.copy(),
            }
