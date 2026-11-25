"""Optional Windows CUDA implementation of the sparse stereo verifier."""

import ctypes
from functools import cache
from pathlib import Path
import re
import sys
from threading import Lock, local

import numpy as np

from native_accel import _ZNCC_REASON_NAMES
from stereo_depth import StereoSearchConfig, verify_stereo_depth_candidates


_DEFAULT_SEARCH = StereoSearchConfig()
_CUDA_SEARCH = StereoSearchConfig(
    patch_size=11, max_disparity=96, minimum_correlation=0.85,
    uniqueness_ratio=0.7, round_trip_px=0.5, minimum_texture_std=2.0,
)
# Smaller batches spend more time uploading and launching than the CPU takes.
_MIN_CUDA_POINTS = 256
_VOID = ctypes.c_void_p


def _bind(library, name, *argtypes):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with code {code}")


def _compile_kernel(torch, device):
    string = ctypes.c_char_p
    size_type = ctypes.c_size_t
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    candidates = sorted(
        path for path in lib_dir.glob("nvrtc64_*.dll")
        if re.fullmatch(r"nvrtc64_\d+_0\.dll", path.name)
    )
    if len(candidates) != 1:
        raise RuntimeError("Expected one NVRTC library in the CUDA PyTorch installation")
    nvrtc = ctypes.CDLL(str(candidates[0]))
    create = _bind(nvrtc, "nvrtcCreateProgram", ctypes.POINTER(_VOID), string,
                   string, ctypes.c_int, _VOID, _VOID)
    compile_program = _bind(nvrtc, "nvrtcCompileProgram", _VOID,
                            ctypes.c_int, ctypes.POINTER(string))
    destroy = _bind(nvrtc, "nvrtcDestroyProgram", ctypes.POINTER(_VOID))
    source = Path(__file__).with_name("native_kernels") / "cuda_zncc.cu"
    program = _VOID()
    _check(create(ctypes.byref(program), source.read_bytes(), b"cuda_zncc.cu",
                  0, None, None), "nvrtcCreateProgram")
    try:
        major, minor = torch.cuda.get_device_capability(device)
        options = (string * 2)(f"--gpu-architecture=sm_{major}{minor}".encode(), b"--fmad=false")
        status = compile_program(program, len(options), options)
        size = size_type()
        if status:
            read_size = _bind(nvrtc, "nvrtcGetProgramLogSize", _VOID, ctypes.POINTER(size_type))
            _check(read_size(program, ctypes.byref(size)), "nvrtcGetProgramLogSize")
            log = ctypes.create_string_buffer(size.value)
            read_log = _bind(nvrtc, "nvrtcGetProgramLog", _VOID, _VOID)
            _check(read_log(program, log), "nvrtcGetProgramLog")
            raise RuntimeError(log.value.decode(errors="replace").strip())
        read_size = _bind(nvrtc, "nvrtcGetCUBINSize", _VOID, ctypes.POINTER(size_type))
        _check(read_size(program, ctypes.byref(size)), "nvrtcGetCUBINSize")
        compiled = ctypes.create_string_buffer(size.value)
        read_cubin = _bind(nvrtc, "nvrtcGetCUBIN", _VOID, _VOID)
        _check(read_cubin(program, compiled), "nvrtcGetCUBIN")
        return compiled
    finally:
        destroy(ctypes.byref(program))


@cache
def _load_kernel(torch, device):
    """Load one CUDA module into PyTorch's primary context for this device."""
    driver = ctypes.WinDLL("nvcuda.dll")
    compiled = _compile_kernel(torch, device)
    load_module = _bind(driver, "cuModuleLoadData", ctypes.POINTER(_VOID), _VOID)
    get_function = _bind(driver, "cuModuleGetFunction", ctypes.POINTER(_VOID), _VOID, ctypes.c_char_p)
    launch = _bind(driver, "cuLaunchKernel", _VOID, *([ctypes.c_uint] * 7),
                   _VOID, ctypes.POINTER(_VOID), _VOID)
    module, kernel = _VOID(), _VOID()
    with torch.cuda.device(device):
        torch.empty(1, device=device)
        _check(load_module(ctypes.byref(module), compiled), "cuModuleLoadData")
        _check(get_function(ctypes.byref(kernel), module, b"verify_zncc"), "cuModuleGetFunction")
    # The cached module stays loaded for the lifetime of the primary context.
    return kernel, launch


class CudaStereoVerifier:
    """Dispatch large standard uint8 batches to CUDA and use the CPU otherwise."""

    def __init__(self, torch, device):
        self._torch = torch
        self._device = device
        self._kernel, self._launch = _load_kernel(torch, device)
        self._thread = local()
        self._lock = Lock()
        self._cuda_batches = self._cuda_points = self._cpu_batches = 0
        self._failure = None

    def __call__(self, left, right, pixels, config=None, disparity_bounds=None):
        bounds = (0.0, 96.0) if disparity_bounds is None else disparity_bounds
        search = _DEFAULT_SEARCH if config is None else config
        if (
            self._failure is None
            and isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.ndim == 2 and left.shape == right.shape
            and left.dtype == right.dtype == np.uint8
            and len(pixels) >= _MIN_CUDA_POINTS
            and search == _CUDA_SEARCH
            and len(bounds) == 2
            and 0.0 <= bounds[0] < bounds[1] <= 96.0
        ):
            coordinates = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
            if np.isfinite(coordinates).all():
                try:
                    measured, codes = self._run(left, right, coordinates, bounds)
                except (OSError, RuntimeError) as exc:
                    with self._lock:
                        self._failure = str(exc)
                else:
                    counts = {"candidates": len(coordinates), "verified": int(np.count_nonzero(codes == 0))}
                    for code, occurrences in enumerate(np.bincount(codes)):
                        if code and occurrences:
                            key = _ZNCC_REASON_NAMES.get(code - 100 if 100 < code < 200 else code, "unknown")
                            counts[key] = counts.get(key, 0) + int(occurrences)
                    with self._lock:
                        self._cuda_batches += 1
                        self._cuda_points += len(coordinates)
                    return measured, counts
        with self._lock:
            self._cpu_batches += 1
        return verify_stereo_depth_candidates(left, right, pixels, config, disparity_bounds)

    def _run(self, left, right, coordinates, bounds):
        torch = self._torch
        if not hasattr(self._thread, "stream"):
            self._thread.stream = torch.cuda.Stream(device=self._device)
        stream = self._thread.stream

        def upload(array):
            array = np.ascontiguousarray(array)
            host = torch.from_numpy(array) if array.flags.writeable else torch.tensor(array, device="cpu")
            return host.to(self._device, non_blocking=True)

        with torch.cuda.stream(stream):
            left_gpu, right_gpu, pixels_gpu = (upload(array) for array in (left, right, coordinates))
            count = len(coordinates)
            measured = torch.empty(count, dtype=torch.float64, device=self._device)
            codes = torch.empty(count, dtype=torch.int32, device=self._device)
            tensors = (left_gpu, right_gpu, pixels_gpu, measured, codes)
            arguments = [
                *(_VOID(tensor.data_ptr()) for tensor in tensors),
                *(ctypes.c_int(size) for size in left.shape),
                *(ctypes.c_double(value) for value in bounds),
            ]
            params = (_VOID * len(arguments))(*(ctypes.addressof(arg) for arg in arguments))
            _check(self._launch(self._kernel, count, 1, 1, 128, 1, 1, 0,
                                stream.cuda_stream, params, None), "cuLaunchKernel")
            offsets_cpu = measured.to("cpu", non_blocking=True)
            codes_cpu = codes.to("cpu", non_blocking=True)
            stream.synchronize()
        return offsets_cpu.numpy(), codes_cpu.numpy()

    def report(self):
        with self._lock:
            return {
                "requested": "cuda",
                "fallback_reason": self._failure,
                "cuda_batches": self._cuda_batches,
                "cuda_points": self._cuda_points,
                "cpu_batches": self._cpu_batches,
            }


def create_cuda_stereo_verifier():
    if sys.platform != "win32":
        return None, "CUDA stereo verification currently loads the Windows PyTorch NVRTC library"
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            return None, "PyTorch cannot access an NVIDIA CUDA device"
        device = torch.device("cuda", torch.cuda.current_device())
        return CudaStereoVerifier(torch, device), None
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        return None, str(exc)
