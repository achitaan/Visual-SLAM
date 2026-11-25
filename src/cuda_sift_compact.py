"""Compact CUDA extrema and descriptor extraction for the native SIFT detector."""

from collections import OrderedDict
import ctypes
from pathlib import Path
from threading import local
from time import perf_counter

import numpy as np

from cuda_kernels import load_cuda_kernels
from cuda_sift_pyramid import CudaSiftPyramid, _RADII, _SiftPyramidWorkspace, _blocks
from frontend_features import SiftDescriptorReadback
from native_accel import compact_sift_functions, native_sift_first_octave, sift_pyramid_parameters


_SYMBOLS = ("sift_sparse_seeds", "sift_pack_gaussian", "sift_descriptors")
_EXTREMUM_BYTES = 60
_KEYPOINT_BYTES = 32


class _CompactSiftWorkspace(_SiftPyramidWorkspace):
    seed_capacity = 4096

    def __init__(self, backend, shape, first_octave, parameters, target, threshold):
        super().__init__(backend, shape, first_octave, parameters, readback=False)
        self._kernels = {**self._kernels, **backend._compact_kernels}
        self._threshold = threshold
        torch, device = self._torch, backend._device
        gaussian_size = sum(3 * h * w for h, w in self._shapes)
        self._gaussian = torch.empty(gaussian_size, dtype=torch.int16, device=device)
        self._seeds = torch.empty((self.seed_capacity, _EXTREMUM_BYTES), dtype=torch.uint8, device=device)
        self._count = torch.empty(1, dtype=torch.int32, device=device)
        self._host_gaussian = torch.empty_like(self._gaussian, device="cpu", pin_memory=True)
        self._host_seeds = torch.empty_like(self._seeds, device="cpu", pin_memory=True)
        self._host_count = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self._keypoints = torch.empty((target, _KEYPOINT_BYTES), dtype=torch.uint8, device=device)
        self._descriptors = torch.empty((target, 128), dtype=torch.float32, device=device)
        self._host_keypoints = torch.empty_like(self._keypoints, device="cpu", pin_memory=True)
        self._host_descriptors = torch.empty_like(self._descriptors, device="cpu", pin_memory=True)
        self._descriptor_weights = backend._descriptor_weights
        self._pending = None
        self._graph = self._capture_graph(self._launch_detection)

    def _launch_detection(self):
        self._image.copy_(self._host_input, non_blocking=True)
        self._launch_pyramid()
        self._count.zero_()
        pyramid_offset = gaussian_offset = 0
        for octave, (h, w) in enumerate(self._shapes):
            area = h * w
            gaussian_count = 3 * area
            blocks = (_blocks(gaussian_count, 256), 1, 1)
            self._call("sift_pack_gaussian", blocks, (256, 1, 1), self._pyramid[pyramid_offset:],
                       self._gaussian[gaussian_offset:], gaussian_count)
            if h > 10 and w >= 42:
                self._call("sift_sparse_seeds", blocks, (256, 1, 1),
                           self._pyramid[pyramid_offset + gaussian_count:], self._seeds, self._count,
                           self.seed_capacity, h, w, octave, gaussian_offset, self._threshold)
            pyramid_offset += 8 * area
            gaussian_offset += gaussian_count
        self._host_gaussian.copy_(self._gaussian, non_blocking=True)
        self._host_seeds.copy_(self._seeds, non_blocking=True)
        self._host_count.copy_(self._count, non_blocking=True)

    def prepare(self, gray):
        if self._pending is not None:
            self._pending.finish()
            self._pending = None
        np.copyto(self._input, gray)
        with self._torch.cuda.stream(self._stream):
            if self._graph is None:
                self._launch_detection()
            else:
                self._graph.replay()
        self._stream.synchronize()
        return int(self._host_count[0])

    def describe(self, count, finish):
        with self._torch.cuda.stream(self._stream):
            self._keypoints[:count].copy_(self._host_keypoints[:count], non_blocking=True)
            self._call("sift_descriptors", (count, 1, 1), (256, 1, 1), self._gaussian,
                       self._keypoints, self._descriptor_weights, self._descriptors)
            self._host_descriptors[:count].copy_(self._descriptors[:count], non_blocking=True)
        self._pending = SiftDescriptorReadback(finish)
        return self._pending


class CudaCompactSift(CudaSiftPyramid):
    """Retain DoG and descriptor sampling on CUDA and select keypoints on the CPU."""

    def __init__(self, torch, device):
        functions = compact_sift_functions()
        if functions is None:
            raise RuntimeError("Rebuild fast_sift with compact CUDA SIFT support")
        super().__init__(torch, device)
        self._detect, copy_weights = functions
        weights = np.empty((4, 256), np.float32)
        copy_weights(weights.ctypes.data)
        self._descriptor_weights = torch.from_numpy(weights).to(device)
        source = Path(__file__).with_name("native_kernels") / "cuda_sift_compact.cu"
        self._compact_kernels, _ = load_cuda_kernels(torch, device, source, _SYMBOLS, (b"--std=c++17",))
        self._compact_workspace = local()
        self._compact_frames = 0
        self._seed_overflows = 0
        self._native_detection_s = 0.0
        self._descriptor_wait_s = 0.0

    def extract(self, gray, **kwargs):
        """Return complete host arrays for callers outside the frontend pipeline."""
        result = self.extract_deferred(gray, **kwargs)
        if result is None or not isinstance(result[1], SiftDescriptorReadback):
            return result
        descriptors = result[1].finish()
        return None if descriptors is None else (result[0], descriptors, *result[2:])

    def extract_deferred(self, gray, *, first_octave=None, target_features=1500, grid_cols=16,
                         grid_rows=6, contrast_threshold=0.02):
        if self._failure is not None or gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16:
            return None
        if first_octave is None:
            first_octave = native_sift_first_octave()
        target = max(1, int(target_features))
        started = perf_counter()
        try:
            # The native prefilter uses half the contrast threshold in Q9 intensity units.
            threshold = int(np.floor(np.float32(contrast_threshold) * np.float32(85 * 256)))
            cache = getattr(self._compact_workspace, "cache", None)
            if cache is None:
                self._compact_workspace.cache = cache = OrderedDict()
            key = gray.shape, first_octave, target, threshold
            workspace = cache.get(key)
            if workspace is None:
                parameters = sift_pyramid_parameters(gray.shape, first_octave)
                if parameters is None or not all(radius in _RADII for radius in parameters[2]):
                    return None
                workspace = cache[key] = _CompactSiftWorkspace(
                    self, gray.shape, first_octave, parameters, target, threshold,
                )
                if len(cache) > 2:
                    _, evicted = cache.popitem(last=False)
                    if evicted._pending is not None:
                        evicted._pending.finish()
            cache.move_to_end(key)
            seed_count = workspace.prepare(gray)
            if seed_count > workspace.seed_capacity:
                with self._lock:
                    self._seed_overflows += 1
                return super().extract(gray, first_octave=first_octave, target_features=target,
                                       grid_cols=grid_cols, grid_rows=grid_rows,
                                       contrast_threshold=contrast_threshold)
            built = perf_counter()

            pixels = np.empty((target, 2), np.float32)
            octaves = np.empty(target, np.int16)
            angles = np.empty(target, np.float32)
            responses = np.empty(target, np.float32)
            count = ctypes.c_int32()
            packed = workspace._host_gaussian
            status = self._detect(
                packed.data_ptr(), packed.numel(), workspace._host_seeds.data_ptr(), seed_count,
                *gray.shape, target, grid_cols, grid_rows, contrast_threshold, first_octave,
                pixels.ctypes.data, workspace._host_keypoints.data_ptr(), octaves.ctypes.data,
                angles.ctypes.data, responses.ctypes.data, ctypes.byref(count),
            )
            if status != 1:
                raise RuntimeError("Native SIFT rejected the compact CUDA extrema")
            detected = perf_counter()
            n = count.value
            if n:
                stream, output = workspace._stream, workspace._host_descriptors
                descriptors = workspace.describe(n, lambda: self._read_descriptors(stream, output, n))
            else:
                descriptors = np.empty((0, 128), np.float32)
                with self._lock:
                    self._compact_frames += 1
        except (OSError, RuntimeError, ValueError, OverflowError) as error:
            with self._lock:
                self._failure = str(error)
            return None
        with self._lock:
            self._pyramid_s += built - started
            self._native_detection_s += detected - built
        return pixels[:n], descriptors, octaves[:n], angles[:n], responses[:n]

    def _read_descriptors(self, stream, output, count):
        started = perf_counter()
        try:
            stream.synchronize()
            descriptors = output[:count].numpy().copy()
        except (OSError, RuntimeError, ValueError) as error:
            with self._lock:
                self._failure = str(error)
            return None
        with self._lock:
            self._compact_frames += 1
            self._descriptor_wait_s += perf_counter() - started
        return descriptors

    def report(self):
        info = super().report()
        with self._lock:
            info.update(requested="cuda-compact", cuda_frames=self._frames + self._compact_frames,
                        compact_frames=self._compact_frames, compact_seed_overflows=self._seed_overflows,
                        native_detection_s=self._native_detection_s, descriptor_wait_s=self._descriptor_wait_s)
        return info
