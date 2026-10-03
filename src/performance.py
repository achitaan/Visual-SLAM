"""Opt-in controls for retrieval, descriptor matching, and profiling."""

from dataclasses import dataclass
from functools import wraps

import numpy as np

from stage_profile import StageProfile


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance choices kept separate from estimator accuracy settings."""

    retrieval: str = "current"
    matching_backend: str = "cpu"
    cpu_optimizations: bool = True
    profile: bool = False

    def __post_init__(self):
        if self.retrieval not in ("current", "indexed", "exhaustive"):
            raise ValueError("Retrieval must be current, indexed or exhaustive")
        if self.matching_backend not in ("cpu", "cuda", "auto"):
            raise ValueError("Matching backend must be cpu, cuda or auto")


# Keep the old import available without maintaining a second profiler.
StageProfiler = StageProfile


def latency_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "calls": int(len(values)),
        "total_s": float(values.sum()),
        "median_ms": float(np.median(values) * 1000) if len(values) else None,
        "p95_ms": float(np.percentile(values, 95) * 1000) if len(values) else None,
    }


def profiled(stage):
    """Measure a method through its instance's StageProfile-compatible profiler."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            profiler = getattr(self, "profiler", getattr(self, "profile", None))
            if profiler is None:
                return function(self, *args, **kwargs)
            return profiler.call(stage, function, self, *args, **kwargs)

        return wrapped

    return decorate
