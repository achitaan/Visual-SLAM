"""Optional CUDA pyramids and fused forward-backward Lucas-Kanade tracking."""

import ctypes
from pathlib import Path
import sys
from threading import Lock, local

import numpy as np

from cuda_kernels import _VOID, _check, load_cuda_kernels
from native_accel import build_lk_pyramid_avx512


class CudaLkPyramid:
    """A frame-owned device pyramid with a lazily constructed CPU counterpart."""

    def __init__(self, gray, max_level, device_pixels, ready):
        self.source_gray = gray
        self.shape = gray.shape
        self.max_level = max_level
        self.device_pixels = device_pixels
        self.ready = ready
        self._native = None

    def native(self):
        if self._native is None:
            self._native = build_lk_pyramid_avx512(self.source_gray, self.max_level)
        return self._native


class CudaOpticalFlow:
    def __init__(self, torch, device):
        self._torch = torch
        self._device = device
        source = Path(__file__).with_name("native_kernels") / "cuda_lk.cu"
        self._kernels, self._launch = load_cuda_kernels(
            torch, device, source, ("init_gray", "down_gray", "track_lk_pair"),
            (b"--std=c++17",),
        )
        self._local = local()
        self._lock = Lock()
        self._failure = None
        self._frames = self._pairs = self._points = 0

    def _upload(self, array):
        array = np.ascontiguousarray(array)
        torch = self._torch
        host = torch.from_numpy(array) if array.flags.writeable else torch.tensor(array, device="cpu")
        return host.to(self._device, non_blocking=True)

    def _call(self, kernel, blocks, threads, stream, *arguments):
        values = [ctypes.c_int(value) if isinstance(value, int) else _VOID(value.data_ptr())
                  for value in arguments]
        params = (_VOID * len(values))(*(ctypes.addressof(value) for value in values))
        _check(self._launch(self._kernels[kernel], blocks, 1, 1, threads, 1, 1, 0,
                            stream.cuda_stream, params, None), "cuLaunchKernel")

    def prepare(self, gray, max_level):
        if (self._failure is not None or gray is None or gray.ndim != 2
                or gray.dtype != np.uint8 or min(gray.shape) < 8 or not 0 <= max_level <= 6):
            return None
        torch = self._torch
        try:
            if not hasattr(self._local, "stream"):
                self._local.stream = torch.cuda.Stream(device=self._device)
            stream = self._local.stream
            shapes = [gray.shape]
            for _ in range(max_level):
                height, width = shapes[-1]
                shapes.append(((height + 1) // 2, (width + 1) // 2))
            with torch.cuda.stream(stream):
                pyramid = torch.empty(sum(h * w for h, w in shapes),
                                      dtype=torch.float32, device=self._device)
                source = self._upload(gray)
                size = gray.size
                self._call("init_gray", (size + 255) // 256, 256, stream, source, pyramid, size)
                previous, offset = 0, size
                for (height, width), (new_height, new_width) in zip(shapes, shapes[1:]):
                    size = new_height * new_width
                    self._call("down_gray", (size + 255) // 256, 256, stream,
                               pyramid, previous, offset, width, height, new_width, new_height)
                    previous, offset = offset, offset + size
                ready = torch.cuda.Event()
                ready.record(stream)
        except (OSError, RuntimeError) as error:
            with self._lock:
                self._failure = str(error)
            return None
        with self._lock:
            self._frames += 1
        return CudaLkPyramid(gray, max_level, pyramid, ready)

    def track_pair(self, previous, current, points, initial, max_level):
        if (self._failure is not None or not isinstance(previous, CudaLkPyramid)
                or not isinstance(current, CudaLkPyramid) or previous.shape != current.shape
                or not 0 <= max_level <= min(previous.max_level, current.max_level) or not len(points)):
            return None
        torch = self._torch
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 2)
        initial = np.ascontiguousarray(initial, dtype=np.float32).reshape(-1, 2)
        if points.shape != initial.shape or not (np.isfinite(points).all() and np.isfinite(initial).all()):
            return None
        try:
            stream = torch.cuda.current_stream(device=self._device)
            with torch.cuda.stream(stream):
                for pyramid in (previous, current):
                    stream.wait_event(pyramid.ready)
                    pyramid.device_pixels.record_stream(stream)
                source, seed = self._upload(points), self._upload(initial)
                count = len(points)
                xy = torch.empty((count, 4), dtype=torch.float32, device=self._device)
                valid = torch.empty((count, 2), dtype=torch.uint8, device=self._device)
                height, width = current.shape
                self._call("track_lk_pair", count, 128, stream, previous.device_pixels,
                           current.device_pixels, source, seed, xy, valid, width, height, max_level)
                cpu_xy = xy.to("cpu", non_blocking=True)
                cpu_valid = valid.to("cpu", non_blocking=True)
                stream.synchronize()
        except (OSError, RuntimeError) as error:
            with self._lock:
                self._failure = str(error)
            return None
        xy = cpu_xy.numpy()
        valid = cpu_valid.numpy()
        with self._lock:
            self._pairs += 1
            self._points += count
        return (np.ascontiguousarray(xy[:, None, :2]), valid[:, :1].copy(),
                np.ascontiguousarray(xy[:, None, 2:]), valid[:, 1:].copy())

    def report(self):
        with self._lock:
            return {"requested": "cuda", "fallback_reason": self._failure,
                    "prepared_frames": self._frames, "fused_pairs": self._pairs, "tracked_points": self._points}


def create_cuda_optical_flow():
    if sys.platform != "win32":
        return None, "CUDA optical flow currently loads the Windows PyTorch NVRTC library"
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            return None, "PyTorch cannot access an NVIDIA CUDA device"
        device = torch.device("cuda", torch.cuda.current_device())
        return CudaOpticalFlow(torch, device), None
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        return None, str(error)
