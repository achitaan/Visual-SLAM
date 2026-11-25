"""CUDA scale-space construction for the native grid-bucketed SIFT extractor."""

import ctypes
from pathlib import Path
import sys
from threading import Lock, local
from time import perf_counter

import numpy as np

from cuda_kernels import _VOID, _check, load_cuda_kernels
from native_accel import (extract_sift_grid_avx512, native_sift_first_octave,
                          sift_pyramid_available, sift_pyramid_parameters)


_RADII = (3, 4, 5, 6, 7, 9)
_SYMBOLS = ("sift_base", "sift_down", *(f"sift_blur_{radius}" for radius in _RADII))


def _blocks(length, size):
    return (length + size - 1) // size


class _SiftPyramidWorkspace:
    def __init__(self, backend, shape, first_octave, parameters, *, readback=True):
        shapes, weights, radii = parameters
        self.key = shape, first_octave
        self._torch = torch = backend._torch
        self._stream = torch.cuda.Stream(device=backend._device)
        self._kernels, self._launch = backend._kernels, backend._launch
        self._shape = shape
        self._doubled = int(first_octave == -1)
        self._shapes, self._radii = shapes, radii
        # The CPU consumes Gaussian layers 1..3 and DoG layers 0..4 in each octave.
        self._output_count = sum(8 * h * w for h, w in shapes)
        self._base_size = shapes[0][0] * shapes[0][1]
        size = self._output_count + 2 * self._base_size
        if size > np.iinfo(np.int32).max:
            raise ValueError("CUDA SIFT image exceeds the pyramid index range")
        self._host_input = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        self._input = self._host_input.numpy()
        self._image = torch.empty(shape, dtype=torch.uint8, device=backend._device)
        self._pyramid = torch.empty(size, dtype=torch.int16, device=backend._device)
        self._weights = torch.from_numpy(weights).to(backend._device)
        if readback:
            self._host_output = torch.empty(self._output_count, dtype=torch.int16, pin_memory=True)
            self._output = self._host_output.numpy()
            self._graph = self._capture_graph(self._launch_pyramid)

    def _capture_graph(self, launch):
        torch = self._torch
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._stream, capture_error_mode="thread_local"):
                launch()
            return graph
        except (AttributeError, RuntimeError, TypeError):
            # Direct launches also support drivers where graph capture is unavailable.
            return None

    def _call(self, name, grid, block, *args):
        values = [ctypes.c_int32(arg) if isinstance(arg, int) else _VOID(arg.data_ptr()) for arg in args]
        params = (_VOID * len(values))(*(ctypes.addressof(value) for value in values))
        _check(self._launch(self._kernels[name], *grid, *block, 0,
                            self._stream.cuda_stream, params, None), "cuLaunchKernel")

    def _launch_pyramid(self):
        height, width = self._shape
        # Reuse the scratch planes across octaves. Neither layer is read back.
        base, scratch = self._output_count, self._output_count + self._base_size
        self._call("sift_base", (_blocks(self._base_size, 256), 1, 1), (256, 1, 1),
                   self._image, self._pyramid, width, height, self._doubled, scratch)
        offset = 0
        for octave, (h, w) in enumerate(self._shapes):
            count = h * w
            bounds = (_blocks(w, 32), _blocks(h, 8), 1)
            if octave == 0:
                self._call(f"sift_blur_{self._radii[0]}", bounds, (32, 8, 1),
                           self._pyramid, self._weights, 0, scratch, base, -1, w, h)
            else:
                self._call("sift_down", (_blocks(count, 256), 1, 1), (256, 1, 1),
                           self._pyramid, previous_third, base, previous_width, w, h)
            source = base
            destinations = (offset, offset + count, offset + 2 * count, scratch, -1)
            for layer, destination in enumerate(destinations, start=1):
                dog = offset + (layer + 2) * count
                self._call(f"sift_blur_{self._radii[layer]}", bounds, (32, 8, 1),
                           self._pyramid, self._weights, 25 * layer, source, destination, dog, w, h)
                source = destination
            previous_third, previous_width = destinations[2], w
            offset += 8 * count

    def prepare(self, gray):
        np.copyto(self._input, gray)
        with self._torch.cuda.stream(self._stream):
            self._image.copy_(self._host_input, non_blocking=True)
            if self._graph is None:
                self._launch_pyramid()
            else:
                self._graph.replay()
            self._host_output.copy_(self._pyramid[:self._output_count], non_blocking=True)
        self._stream.synchronize()
        return self._output


class CudaSiftPyramid:
    def __init__(self, torch, device):
        self._torch, self._device = torch, device
        source = Path(__file__).with_name("native_kernels") / "cuda_sift_pyramid.cu"
        self._kernels, self._launch = load_cuda_kernels(torch, device, source, _SYMBOLS, (b"--std=c++17",))
        self._workspace = local()
        self._lock = Lock()
        self._failure = None
        self._frames = 0
        self._pyramid_s = 0.0
        self._native_s = 0.0

    def extract(self, gray, *, first_octave=None, **kwargs):
        if self._failure is not None or gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16:
            return None
        if first_octave is None:
            first_octave = native_sift_first_octave()
        started = perf_counter()
        try:
            workspace = getattr(self._workspace, "value", None)
            if workspace is None or workspace.key != (gray.shape, first_octave):
                parameters = sift_pyramid_parameters(gray.shape, first_octave)
                if parameters is None or not all(radius in _RADII for radius in parameters[2]):
                    return None
                workspace = _SiftPyramidWorkspace(self, gray.shape, first_octave, parameters)
                self._workspace.value = workspace
            packed = workspace.prepare(gray)
            built = perf_counter()
            result = extract_sift_grid_avx512(gray, first_octave=first_octave, pyramid=packed, **kwargs)
            if result is None:
                raise RuntimeError("Native SIFT rejected the CUDA pyramid")
        except (OSError, RuntimeError, ValueError) as error:
            with self._lock:
                self._failure = str(error)
            return None
        with self._lock:
            self._frames += 1
            self._pyramid_s += built - started
            self._native_s += perf_counter() - built
        return result

    def report(self):
        with self._lock:
            return {"requested": "cuda", "fallback_reason": self._failure, "cuda_frames": self._frames,
                    "pyramid_s": self._pyramid_s, "native_descriptors_s": self._native_s}


def create_cuda_sift_pyramid(*, compact=False):
    if sys.platform != "win32":
        return None, "CUDA SIFT currently loads the Windows PyTorch NVRTC library"
    if not sift_pyramid_available():
        return None, "Rebuild the AVX-512 fast_sift kernel with CUDA pyramid support"
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            return None, "PyTorch cannot access an NVIDIA CUDA device"
        device = torch.device("cuda", torch.cuda.current_device())
        if compact:
            from cuda_sift_compact import CudaCompactSift

            return CudaCompactSift(torch, device), None
        return CudaSiftPyramid(torch, device), None
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        return None, str(error)
