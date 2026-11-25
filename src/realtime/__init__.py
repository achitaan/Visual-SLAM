"""Real-time Visual-SLAM streaming, async mapping, server, and CLI package."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .engine import RealtimeSlamEngine
from .frame_source import (
    DepthAIStereoSource,
    FrameSource,
    OpenCVStereoCameraSource,
    PacedKittiSource,
    RealSenseStereoSource,
    StereoCalibration,
    StereoFrame,
    StereoRectifier,
    StereoVideoFileSource,
)
from .local_mapping_worker import (
    LocalBACorrection,
    LocalBASnapshot,
    LocalMappingWorker,
    apply_local_ba_correction,
)
from .mailbox import FrameMailbox
from .server import build_source_from_spec, create_app

__all__ = [
    "DepthAIStereoSource",
    "FrameMailbox",
    "FrameSource",
    "LocalBACorrection",
    "LocalBASnapshot",
    "LocalMappingWorker",
    "OpenCVStereoCameraSource",
    "PacedKittiSource",
    "RealSenseStereoSource",
    "RealtimeSlamEngine",
    "StereoCalibration",
    "StereoFrame",
    "StereoRectifier",
    "StereoVideoFileSource",
    "apply_local_ba_correction",
    "build_source_from_spec",
    "create_app",
]
