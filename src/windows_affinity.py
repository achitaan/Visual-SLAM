"""Optional core placement for the Windows estimator subprocess."""

import ctypes
from ctypes import wintypes
from struct import Struct
import sys


WINDOWS_CORE_PLACEMENTS = ("default", "split", "isolated")

# SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX followed by the fixed part of
# PROCESSOR_RELATIONSHIP, then one x64 GROUP_AFFINITY per reported group.
_CORE_HEADER = Struct("<IIBB20xH")
_GROUP_AFFINITY = Struct("<QH6x")


def _physical_core_masks(topology, allowed):
    """Return process-eligible masks for a homogeneous, single-group topology."""
    offset = 0
    cores = []
    efficiency = set()
    while offset < len(topology):
        if len(topology) - offset < _CORE_HEADER.size:
            raise ValueError("Truncated Windows processor topology")
        relation, size, _, core_type, groups = _CORE_HEADER.unpack_from(topology, offset)
        if (relation != 0 or groups != 1
                or size < _CORE_HEADER.size + _GROUP_AFFINITY.size
                or offset + size > len(topology)):
            raise ValueError("Unsupported Windows processor topology")
        mask, group = _GROUP_AFFINITY.unpack_from(topology, offset + _CORE_HEADER.size)
        if group != 0:
            raise ValueError("Core placement requires one Windows processor group")
        if mask & allowed:
            cores.append(mask & allowed)
            efficiency.add(core_type)
        offset += size
    if len(efficiency) > 1:
        raise ValueError("Core placement requires homogeneous physical cores")
    return sorted(cores)


def _windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    size_pointer = ctypes.POINTER(ctypes.c_size_t)
    api.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
    ]
    api.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetCurrentThread.restype = wintypes.HANDLE
    api.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, size_pointer, size_pointer]
    api.GetProcessAffinityMask.restype = wintypes.BOOL
    api.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    api.SetProcessAffinityMask.restype = wintypes.BOOL
    api.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    api.SetThreadAffinityMask.restype = ctypes.c_size_t
    return api


class WindowsCorePlacement:
    """Place the benchmark's three long-lived estimator threads."""

    def __init__(self, mode):
        if mode not in WINDOWS_CORE_PLACEMENTS:
            raise ValueError(f"Unknown Windows core placement: {mode}")
        self.report = {"requested": mode, "applied_threads": {}, "fallback_reason": None}
        self._thread_masks = {}
        if mode == "default":
            return
        if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.report["fallback_reason"] = "Core placement requires 64-bit Windows"
            return
        try:
            self._configure()
        except (OSError, ValueError) as error:
            self.report["fallback_reason"] = str(error)

    def _configure(self):
        api = self._api = _windows_api()
        process = api.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not api.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        size = wintypes.DWORD()
        api.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        topology = ctypes.create_string_buffer(size.value)
        if not api.GetLogicalProcessorInformationEx(0, topology, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        cores = _physical_core_masks(topology.raw[:size.value], allowed.value)
        if len(cores) < 3:
            raise ValueError("Split core placement requires three accessible physical cores")
        self.report["allowed_process_mask"] = allowed.value
        self.report["physical_core_masks"] = cores
        first_threads = [mask & -mask for mask in cores]
        self._thread_masks = dict(zip(("tracking", "frontend", "loop"), reversed(first_threads[-3:])))
        if self.report["requested"] == "isolated":
            mask = sum(first_threads)
            if api.SetProcessAffinityMask(process, mask):
                self.report["applied_process_mask"] = mask
            else:
                error = ctypes.WinError(ctypes.get_last_error())
                self.report["fallback_reason"] = (
                    f"Could not limit the estimator to one logical processor per core: {error}"
                )

    def bind(self, role):
        mask = self._thread_masks.get(role)
        if mask is None:
            return
        api = self._api
        if api.SetThreadAffinityMask(api.GetCurrentThread(), mask):
            self.report["applied_threads"][role] = mask
        else:
            error = ctypes.WinError(ctypes.get_last_error())
            self.report["fallback_reason"] = f"Could not place {role} thread: {error}"
