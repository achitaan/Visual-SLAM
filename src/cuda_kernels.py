"""Compile and load repository CUDA kernels using Windows PyTorch NVRTC."""

import ctypes
from functools import cache
from pathlib import Path
import re


_VOID = ctypes.c_void_p


def _bind(library, name, *argtypes):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with code {code}")


def _compile_kernel(torch, device, source, extra_options):
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
    program = _VOID()
    _check(create(ctypes.byref(program), source.read_bytes(), source.name.encode(),
                  0, None, None), "nvrtcCreateProgram")
    try:
        major, minor = torch.cuda.get_device_capability(device)
        options = (string * (2 + len(extra_options)))(
            f"--gpu-architecture=sm_{major}{minor}".encode(), b"--fmad=false", *extra_options,
        )
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
def load_cuda_kernels(torch, device, source, symbols, extra_options=()):
    """Load one CUDA module into PyTorch's primary context for this device."""
    driver = ctypes.WinDLL("nvcuda.dll")
    compiled = _compile_kernel(torch, device, source, extra_options)
    load_module = _bind(driver, "cuModuleLoadData", ctypes.POINTER(_VOID), _VOID)
    get_function = _bind(driver, "cuModuleGetFunction", ctypes.POINTER(_VOID), _VOID, ctypes.c_char_p)
    launch = _bind(driver, "cuLaunchKernel", _VOID, *([ctypes.c_uint] * 7),
                   _VOID, ctypes.POINTER(_VOID), _VOID)
    module = _VOID()
    kernels = {}
    with torch.cuda.device(device):
        torch.empty(1, device=device)
        _check(load_module(ctypes.byref(module), compiled), "cuModuleLoadData")
        for symbol in symbols:
            kernel = _VOID()
            _check(get_function(ctypes.byref(kernel), module, symbol.encode()), "cuModuleGetFunction")
            kernels[symbol] = kernel
    # The cached module stays loaded for the lifetime of the primary context.
    return kernels, launch
