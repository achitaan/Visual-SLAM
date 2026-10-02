"""Optional exact CUDA L2 search; CPU remains the default and tie arbiter."""

from time import perf_counter
import cv2 as cv
import numpy as np
from mapping_geometry import match_descriptors


class DescriptorMatcher:
    def __init__(self, backend="cpu"):
        self.requested = backend
        self.torch = None
        self.reason = None
        self.auto_decisions = {}
        self.cuda_calls = 0
        self.cpu_calls = 0
        self.ambiguous_rows = 0
        if backend != "cpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable in this PyTorch environment")
                # No reduced-precision matrix multiplication in descriptor distances.
                torch.backends.cuda.matmul.allow_tf32 = False
                self.torch = torch
                torch.empty(1, device="cuda")
                torch.cuda.synchronize()
            except (ImportError, OSError, RuntimeError) as error:
                self.reason = str(error)
                if backend == "cuda":
                    raise RuntimeError("CUDA matching requested but unavailable: " + self.reason) from error

    def __call__(self, first, second, ratio=0.7):
        if first is None or second is None or len(first) < 2 or len(second) < 2:
            return np.empty((0, 2), int)
        if self.torch is None or first.dtype == np.uint8 or second.dtype == np.uint8:
            self.cpu_calls += 1
            return match_descriptors(first, second, ratio)
        bucket = (int(np.log2(len(first))), int(np.log2(len(second))))
        if self.requested == "auto":
            if len(first) * len(second) < 250_000 or self.auto_decisions.get(bucket) is False:
                self.cpu_calls += 1
                return match_descriptors(first, second, ratio)
            if bucket not in self.auto_decisions:
                started = perf_counter()
                cpu = match_descriptors(first, second, ratio)
                cpu_time = perf_counter() - started
                started = perf_counter()
                gpu = self._cuda(first, second, ratio)
                gpu_time = perf_counter() - started
                use_gpu = gpu_time < cpu_time * .8 and np.array_equal(cpu, gpu)
                self.auto_decisions[bucket] = use_gpu
                return gpu if use_gpu else cpu
        return self._cuda(first, second, ratio)

    def _direction(self, first, second, ratio):
        torch = self.torch
        mappings = np.full(len(first), -1, int)
        train = torch.as_tensor(np.ascontiguousarray(second, dtype=np.float32), device="cuda")
        train_norm = float(np.max(np.einsum("ij,ij->i", second, second)))
        for start in range(0, len(first), 1024):
            source = np.ascontiguousarray(first[start:start + 1024], dtype=np.float32)
            query = torch.as_tensor(source, device="cuda")
            distances = torch.cdist(query, train, compute_mode="use_mm_for_euclid_dist")
            values, indices = torch.topk(distances, 2, dim=1, largest=False, sorted=True)
            values, indices = values.cpu().numpy(), indices.cpu().numpy()
            accepted = values[:, 0] < ratio * values[:, 1]
            # Roundoff near equal nearest neighbors or the ratio boundary is
            # resolved by exactly the same OpenCV search as the CPU estimator.
            # Bound cancellation in ||a||^2 + ||b||^2 - 2<a,b>. A relative
            # distance tolerance alone misses almost-identical large vectors.
            squared = values.astype(np.float64) ** 2
            bound = 8 * np.finfo(np.float32).eps * (source.shape[1] + 2) * (np.einsum("ij,ij->i", source, source) + train_norm)
            ambiguous = (accepted & (squared[:, 1] - squared[:, 0] <= 2 * bound)) | (np.abs(squared[:, 0] - ratio ** 2 * squared[:, 1]) <= 2 * bound)
            rows = np.flatnonzero(ambiguous)
            self.ambiguous_rows += len(rows)
            if len(rows):
                matches = cv.BFMatcher(cv.NORM_L2).knnMatch(source[rows], second, k=2)
                for row, pair in zip(rows, matches):
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
        self.cuda_calls += 1
        rows = np.flatnonzero(forward >= 0)
        rows = rows[backward[forward[rows]] == rows]
        return np.column_stack([rows, forward[rows]]).astype(int)

    def metadata(self):
        return {
            "requested": self.requested, "cuda_available": self.torch is not None,
            "fallback_reason": self.reason, "cuda_calls": self.cuda_calls,
            "cpu_calls": self.cpu_calls, "ambiguous_rows_on_cpu": self.ambiguous_rows,
            "auto_decisions": {str(k): v for k, v in self.auto_decisions.items()},
            "peak_cuda_allocated_mb": self.torch.cuda.max_memory_allocated() / 1024**2 if self.torch else 0.0,
            "peak_cuda_reserved_mb": self.torch.cuda.max_memory_reserved() / 1024**2 if self.torch else 0.0,
        }
