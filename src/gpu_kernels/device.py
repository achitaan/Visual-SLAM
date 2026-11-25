"""Hardware detection and backend selection (--device {auto,cuda,cpu} / VISUAL_SLAM_DEVICE)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
from typing import Any

import cv2 as cv

import native_accel

VALID_DEVICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute backend metadata for real-time telemetry and dispatch."""

    requested: str
    effective: str
    backend_label: str
    torch_available: bool
    cuda_available: bool
    cv2_cuda_available: bool
    native_cpu_available: bool
    native_cpu_kernels: tuple[str, ...]
    device_name: str
    compute_capability: tuple[int, int] | None = None
    vram_gb: float | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["native_cpu_kernels"] = list(self.native_cpu_kernels)
        if self.compute_capability is not None:
            d["compute_capability"] = list(self.compute_capability)
        return d


def _probe_native_kernels() -> tuple[str, ...]:
    if not native_accel.native_enabled():
        return ()
    names = ("fast_matcher", "fast_zncc", "fast_sift", "fast_ba", "fast_lk", "fast_pgo")
    available = []
    for name in names:
        if native_accel._load_library(name) is not None:
            available.append(name)
    return tuple(available)


def _probe_cv2_cuda() -> bool:
    try:
        cuda_mod = getattr(cv, "cuda", None)
        if cuda_mod is None or not hasattr(cuda_mod, "getCudaEnabledDeviceCount"):
            return False
        return int(cuda_mod.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


def _probe_torch_cuda() -> tuple[bool, bool, str | None, tuple[int, int] | None, float | None, str | None]:
    try:
        import torch
    except ImportError:
        return False, False, None, None, None, "PyTorch is not installed"
    try:
        if not torch.cuda.is_available():
            return True, False, None, None, None, "CUDA is unavailable in PyTorch"
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        vram_gb = round(float(props.total_memory) / (1024.0 ** 3), 2)
        return True, True, str(props.name), (int(cap[0]), int(cap[1])), vram_gb, None
    except Exception as exc:
        return True, False, None, None, None, f"CUDA probe failed: {exc}"


def detect_device() -> DeviceInfo:
    """Detect hardware capabilities using default VISUAL_SLAM_DEVICE or 'auto'."""
    return select_device(None)


def select_device(
    requested: str | None = None,
    *,
    strict: bool = False,
) -> DeviceInfo:
    """Resolve ``--device {auto,cuda,cpu}`` or ``VISUAL_SLAM_DEVICE`` into a DeviceInfo."""
    req = (
        requested
        if requested is not None
        else os.environ.get("VISUAL_SLAM_DEVICE", "auto")
    ).strip().lower()
    if req not in VALID_DEVICES:
        raise ValueError(f"Invalid device {req!r}; expected one of {VALID_DEVICES}")

    native_kernels = _probe_native_kernels()
    native_ok = len(native_kernels) > 0
    cv2_cuda = _probe_cv2_cuda()

    if req == "cpu":
        cpu_name = platform.processor() or platform.machine() or "CPU"
        label = "cpu:native" if native_ok else "cpu:python"
        return DeviceInfo(
            requested=req,
            effective="cpu",
            backend_label=label,
            torch_available=False,
            cuda_available=False,
            cv2_cuda_available=cv2_cuda,
            native_cpu_available=native_ok,
            native_cpu_kernels=native_kernels,
            device_name=cpu_name,
            fallback_reason=None,
        )

    torch_avail, cuda_avail, gpu_name, cap, vram_gb, reason = _probe_torch_cuda()
    if cuda_avail or cv2_cuda:
        effective_name = gpu_name or "NVIDIA CUDA GPU"
        return DeviceInfo(
            requested=req,
            effective="cuda",
            backend_label=f"cuda:{effective_name}",
            torch_available=torch_avail,
            cuda_available=cuda_avail,
            cv2_cuda_available=cv2_cuda,
            native_cpu_available=native_ok,
            native_cpu_kernels=native_kernels,
            device_name=effective_name,
            compute_capability=cap,
            vram_gb=vram_gb,
            fallback_reason=None,
        )

    if req == "cuda" and strict:
        raise RuntimeError(f"CUDA device requested but unavailable: {reason or 'no CUDA device'}")

    cpu_name = platform.processor() or platform.machine() or "CPU"
    label = "cpu:native" if native_ok else "cpu:python"
    return DeviceInfo(
        requested=req,
        effective="cpu",
        backend_label=label,
        torch_available=torch_avail,
        cuda_available=False,
        cv2_cuda_available=False,
        native_cpu_available=native_ok,
        native_cpu_kernels=native_kernels,
        device_name=cpu_name,
        fallback_reason=reason or "CUDA unavailable; using CPU fallback",
    )
