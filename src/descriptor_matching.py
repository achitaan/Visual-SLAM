"""Optional exact CUDA L2 search; CPU remains the default and tie arbiter."""

import platform
from threading import Lock
from time import perf_counter

import cv2 as cv
import numpy as np

from mapping_geometry import match_descriptors


class DescriptorMatcher:
    """Mutual ratio matcher with an optional CUDA path for float SIFT rows.

    CPU matching is always available. CUDA rows close to a nearest-neighbor
    tie, a ratio boundary, or a cancellation-sensitive distance are recomputed
    by OpenCV so the CPU matcher remains the decision authority.
    """

    def __init__(self, backend="cpu"):
        if backend not in ("cpu", "cuda", "auto"):
            raise ValueError("Matching backend must be cpu, cuda or auto")
        self.requested = backend
        self.torch = None
        self.reason = None
        self.python_version = platform.python_version()
        self.pytorch_version = None
        self.cuda_runtime_version = None
        self.cuda_device_index = None
        self.cuda_device_name = None
        self.cuda_device_capability = None
        self.auto_decisions = {}
        self.cuda_calls = 0
        self.cpu_calls = 0
        self.ambiguous_rows = 0
        self._lock = Lock()
        if backend != "cpu":
            try:
                import torch

                self.pytorch_version = str(torch.__version__)
                runtime_version = getattr(torch.version, "cuda", None)
                self.cuda_runtime_version = (
                    str(runtime_version) if runtime_version is not None else None
                )
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable in this PyTorch environment")
                # Reduced-precision matrix multiplication can change close L2
                # comparisons, so keep the distance kernel in full precision.
                torch.backends.cuda.matmul.allow_tf32 = False
                self.torch = torch
                torch.empty(1, device="cuda")
                torch.cuda.synchronize()
                try:
                    device = int(torch.cuda.current_device())
                    properties = torch.cuda.get_device_properties(device)
                    capability = torch.cuda.get_device_capability(device)
                    self.cuda_device_index = device
                    self.cuda_device_name = str(properties.name)
                    self.cuda_device_capability = [int(value) for value in capability]
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    # Provenance should not disable an otherwise usable CUDA path.
                    pass
            except (ImportError, OSError, RuntimeError) as error:
                self.reason = str(error)
                if backend == "cuda":
                    raise RuntimeError("CUDA matching requested but unavailable: " + self.reason) from error

    def _cpu(self, first, second, ratio):
        with self._lock:
            self.cpu_calls += 1
        return match_descriptors(first, second, ratio)

    def __call__(self, first, second, ratio=0.7):
        if first is None or second is None or len(first) < 2 or len(second) < 2:
            return np.empty((0, 2), int)
        # GPU distances are evaluated in float32. Keep other descriptor types
        # on the reference implementation to avoid a precision conversion.
        if (
            self.torch is None
            or not isinstance(first, np.ndarray)
            or not isinstance(second, np.ndarray)
            or first.dtype != np.float32
            or second.dtype != np.float32
        ):
            return self._cpu(first, second, ratio)

        bucket = (int(np.log2(len(first))), int(np.log2(len(second))))
        if self.requested == "auto":
            decision = self.auto_decisions.get(bucket)
            if len(first) * len(second) < 250_000 or decision is False:
                return self._cpu(first, second, ratio)
            if decision is None:
                started = perf_counter()
                cpu = self._cpu(first, second, ratio)
                cpu_time = perf_counter() - started
                started = perf_counter()
                try:
                    gpu = self._cuda(first, second, ratio)
                except (OSError, RuntimeError) as error:
                    self.reason = str(error)
                    self.torch = None
                    self.auto_decisions[bucket] = False
                    return cpu
                gpu_time = perf_counter() - started
                use_gpu = gpu_time < cpu_time * 0.8 and np.array_equal(cpu, gpu)
                self.auto_decisions[bucket] = use_gpu
                return gpu if use_gpu else cpu

        try:
            return self._cuda(first, second, ratio)
        except (OSError, RuntimeError) as error:
            if self.requested != "auto":
                raise
            self.reason = str(error)
            self.torch = None
            return self._cpu(first, second, ratio)

    def _direction(self, first, second, ratio):
        torch = self.torch
        mappings = np.full(len(first), -1, int)
        train = torch.as_tensor(np.ascontiguousarray(second), device="cuda")
        train_norm = float(np.max(np.einsum("ij,ij->i", second, second)))
        for start in range(0, len(first), 1024):
            source = np.ascontiguousarray(first[start:start + 1024])
            query = torch.as_tensor(source, device="cuda")
            distances = torch.cdist(query, train, compute_mode="use_mm_for_euclid_dist")
            values, indices = torch.topk(distances, 2, dim=1, largest=False, sorted=True)
            values, indices = values.cpu().numpy(), indices.cpu().numpy()
            squared = values.astype(np.float64) ** 2
            accepted = values[:, 0] < ratio * values[:, 1]

            # cdist's matrix-multiply identity subtracts large, similar terms.
            # Use a conservative float32 error bound to route uncertain rows
            # back through the same CPU L2 and ratio tests used by the default.
            source_norm = np.einsum("ij,ij->i", source, source)
            bound = 8 * np.finfo(np.float32).eps * (source.shape[1] + 2) * (source_norm + train_norm)
            near_tie = squared[:, 1] - squared[:, 0] <= 2 * bound
            near_ratio = np.abs(squared[:, 0] - ratio ** 2 * squared[:, 1]) <= 2 * bound
            cancellation_sensitive = squared[:, 0] <= bound
            # Cancellation can turn a real match into a rejected GPU ratio as
            # well as accept a false match. Recheck both kinds of uncertain row.
            ambiguous = near_tie | near_ratio | cancellation_sensitive
            rows = np.flatnonzero(ambiguous)
            with self._lock:
                self.ambiguous_rows += len(rows)
            if len(rows):
                cpu_matches = cv.BFMatcher(cv.NORM_L2).knnMatch(source[rows], second, k=2)
                for row, pair in zip(rows, cpu_matches):
                    accepted[row] = len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
                    if pair:
                        indices[row, 0] = pair[0].trainIdx
            selected = np.flatnonzero(accepted)
            mappings[start + selected] = indices[selected, 0]
        return mappings

    def _cuda(self, first, second, ratio):
        with self.torch.inference_mode():
            forward = self._direction(first, second, ratio)
            backward = self._direction(second, first, ratio)
            self.torch.cuda.synchronize()
        with self._lock:
            self.cuda_calls += 1
        rows = np.flatnonzero(forward >= 0)
        rows = rows[backward[forward[rows]] == rows]
        return np.column_stack([rows, forward[rows]]).astype(int)

    def metadata(self):
        with self._lock:
            result = {
                "requested": self.requested,
                "cuda_available": self.torch is not None,
                "fallback_reason": self.reason,
                "python_version": self.python_version,
                "pytorch_version": self.pytorch_version,
                "cuda_runtime_version": self.cuda_runtime_version,
                "cuda_device_index": self.cuda_device_index,
                "cuda_device_name": self.cuda_device_name,
                "cuda_device_capability": self.cuda_device_capability,
                "cuda_calls": self.cuda_calls,
                "cpu_calls": self.cpu_calls,
                "ambiguous_rows_on_cpu": self.ambiguous_rows,
                "auto_decisions": {str(key): value for key, value in self.auto_decisions.items()},
            }
        if self.torch is not None:
            try:
                result["peak_cuda_allocated_mb"] = self.torch.cuda.max_memory_allocated() / 1024**2
                result["peak_cuda_reserved_mb"] = self.torch.cuda.max_memory_reserved() / 1024**2
            except (OSError, RuntimeError):
                result["peak_cuda_allocated_mb"] = 0.0
                result["peak_cuda_reserved_mb"] = 0.0
        else:
            result["peak_cuda_allocated_mb"] = 0.0
            result["peak_cuda_reserved_mb"] = 0.0
        return result

    def report(self):
        """Alias used by callers that collect component reports uniformly."""
        return self.metadata()
