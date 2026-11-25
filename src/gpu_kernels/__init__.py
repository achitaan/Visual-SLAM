"""Optional NVIDIA GPU acceleration backend with automatic CPU native & Python fallback."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .device import DeviceInfo, detect_device, select_device
from .frontend_gpu import GpuFrontendAccelerator

__all__ = [
    "DeviceInfo",
    "GpuFrontendAccelerator",
    "detect_device",
    "select_device",
]
