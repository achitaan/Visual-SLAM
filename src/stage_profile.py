"""Inclusive stage timings; nested stages must not be added together."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from time import perf_counter


class StageProfile:
    def __init__(self):
        self.values = defaultdict(lambda: [0, 0.0])

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.values[name][0] += 1
            self.values[name][1] += perf_counter() - start

    def wrap(self, name, method):
        @wraps(method)
        def timed(*args, **kwargs):
            with self.measure(name):
                return method(*args, **kwargs)
        return timed

    def report(self):
        return {name: {"calls": n, "inclusive_seconds": s} for name, (n, s) in self.values.items()}
