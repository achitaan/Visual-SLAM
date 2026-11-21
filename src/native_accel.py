"""Optional native C++/AVX-512 kernel loader with seamless Python/OpenCV fallback.

Each kernel accelerates one self-contained numerical or vision operation (descriptor
matching, epipolar ZNCC verification, SIFT extraction, Schur-complement BA) while
preserving the Python SLAM engine architecture and falling back cleanly when native
binaries are unavailable or disabled via ``VISUAL_SLAM_DISABLE_NATIVE=1``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

_KERNEL_DIR = Path(__file__).resolve().parent / "native_kernels"
_LOADED: dict[str, Any] = {}


def native_enabled() -> bool:
    return os.environ.get("VISUAL_SLAM_DISABLE_NATIVE", "0").strip().lower() not in (
        "1", "true", "yes", "on"
    )


def _load_library(name: str) -> ctypes.CDLL | None:
    if not native_enabled():
        return None
    if name in _LOADED:
        return _LOADED[name]
    lib_path = _KERNEL_DIR / f"lib{name}.so"
    if not lib_path.is_file():
        _LOADED[name] = None
        return None
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError:
        lib = None
    _LOADED[name] = lib
    return lib


def _get_matcher_lib() -> ctypes.CDLL | None:
    lib = _load_library("fast_matcher")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.match_sift_mutual_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.match_sift_mutual_f32.restype = ctypes.c_int32
        lib._configured = True
    return lib


def match_sift_mutual_vnni(
    first: np.ndarray,
    second: np.ndarray,
    ratio: float,
) -> np.ndarray | None:
    """Run AVX-512 VNNI exact integer SIFT mutual L2 ratio matcher.

    Returns an ``[K, 2]`` ``int`` array identical to ``cv.BFMatcher(cv.NORM_L2)``,
    or ``None`` if the native kernel is unavailable, inputs are not 128-D exact
    integer ``[0..255]`` float32 descriptors, or any row sits near the ratio boundary.
    """
    lib = _get_matcher_lib()
    if lib is None:
        return None
    if first.shape[1] != 128 or second.shape[1] != 128:
        return None
    a = np.ascontiguousarray(first, dtype=np.float32)
    b = np.ascontiguousarray(second, dtype=np.float32)
    m, n = len(a), len(b)
    out_pairs = np.empty((min(m, n), 2), dtype=np.int32)
    out_cnt = ctypes.c_int32(0)
    amb_a = np.empty(m, dtype=np.int32)
    amb_a_cnt = ctypes.c_int32(0)
    amb_b = np.empty(n, dtype=np.int32)
    amb_b_cnt = ctypes.c_int32(0)
    status = lib.match_sift_mutual_f32(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        float(ratio),
        out_pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(out_cnt),
        amb_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(amb_a_cnt),
        amb_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(amb_b_cnt),
    )
    if status != 1:
        return None
    return out_pairs[: out_cnt.value].astype(int, copy=False)


_ZNCC_REASON_NAMES: dict[int, str] = {
    1: "patch_bounds",
    2: "low_texture",
    3: "search_bounds",
    4: "correlation",
    5: "insufficient_alternatives",
    6: "near_tie",
    7: "ambiguous",
    8: "unrefinable_peak",
    9: "flat_peak",
    10: "unstable_peak",
    11: "refined_bounds",
    12: "low_target_texture",
    13: "refined_correlation",
    200: "round_trip",
}


def _get_zncc_lib() -> ctypes.CDLL | None:
    lib = _load_library("fast_zncc")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.verify_stereo_zncc_batch_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.verify_stereo_zncc_batch_f32.restype = ctypes.c_int32
        lib._configured = True
    return lib


def verify_stereo_zncc_avx512(
    left: np.ndarray,
    right: np.ndarray,
    finite_pixels: np.ndarray,
    config: Any,
    disparity_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict[str, int]] | None:
    """Run AVX-512 C++ epipolar ZNCC + round-trip verification.

    Returns ``(right_u, reason_counts)`` or ``None`` if unavailable/unsupported.
    """
    lib = _get_zncc_lib()
    if lib is None:
        return None
    left_c = np.ascontiguousarray(left, dtype=np.float32)
    right_c = np.ascontiguousarray(right, dtype=np.float32)
    pts_c = np.ascontiguousarray(finite_pixels, dtype=np.float64)
    n = len(pts_c)
    out_u = np.empty(n, dtype=np.float64)
    out_r = np.empty(n, dtype=np.int32)
    ok = lib.verify_stereo_zncc_batch_f32(
        left_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        right_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        left_c.shape[0],
        left_c.shape[1],
        pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        int(config.patch_size),
        int(config.max_disparity),
        float(config.minimum_correlation),
        float(config.uniqueness_ratio),
        float(config.round_trip_px),
        float(config.minimum_texture_std),
        float(disparity_bounds[0]),
        float(disparity_bounds[1]),
        out_u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_r.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if ok != 1:
        return None
    counts: dict[str, int] = {"verified": 0}
    for code in out_r.tolist():
        if code == 0:
            counts["verified"] += 1
        else:
            base = code - 100 if (100 < code < 200) else code
            name = _ZNCC_REASON_NAMES.get(base, "unknown")
            counts[name] = counts.get(name, 0) + 1
    return out_u, counts


def _get_sift_lib() -> ctypes.CDLL | None:
    if os.environ.get("VISUAL_SLAM_DISABLE_FAST_SIFT", "0").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return None
    lib = _load_library("fast_sift")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.extract_sift_grid_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.extract_sift_grid_avx512.restype = ctypes.c_int32
        lib._configured = True
    return lib


def extract_sift_grid_avx512(
    image_gray: np.ndarray,
    *,
    target_features: int = 1500,
    grid_cols: int = 16,
    grid_rows: int = 6,
    contrast_threshold: float = 0.02,
    first_octave: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run AVX-512 C++ SIFT with early spatial grid bucketing before descriptor computation."""
    lib = _get_sift_lib()
    if lib is None or image_gray.dtype != np.uint8 or image_gray.ndim != 2:
        return None
    if first_octave is None:
        raw_fo = os.environ.get("VISUAL_SLAM_SIFT_FIRST_OCTAVE", "-1").strip()
        first_octave = int(raw_fo) if raw_fo in ("-1", "0") else -1
    img = np.ascontiguousarray(image_gray, dtype=np.uint8)
    target = max(1, int(target_features))
    px = np.empty((target, 2), dtype=np.float32)
    desc = np.empty((target, 128), dtype=np.float32)
    octs = np.empty(target, dtype=np.int16)
    angles = np.empty(target, dtype=np.float32)
    responses = np.empty(target, dtype=np.float32)
    cnt = ctypes.c_int32(0)
    ok = lib.extract_sift_grid_avx512(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        img.shape[0],
        img.shape[1],
        target,
        int(grid_cols),
        int(grid_rows),
        float(contrast_threshold),
        int(first_octave),
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        desc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        octs.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        angles.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        responses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(cnt),
    )
    if ok != 1:
        return None
    n = int(cnt.value)
    return px[:n], desc[:n], octs[:n], angles[:n], responses[:n]
