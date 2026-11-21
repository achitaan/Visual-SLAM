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
        _common = [
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
        lib.verify_stereo_zncc_batch_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            *_common,
        ]
        lib.verify_stereo_zncc_batch_f32.restype = ctypes.c_int32
        if hasattr(lib, "verify_stereo_zncc_batch_u8"):
            lib.verify_stereo_zncc_batch_u8.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                *_common,
            ]
            lib.verify_stereo_zncc_batch_u8.restype = ctypes.c_int32
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
    pts_c = np.ascontiguousarray(finite_pixels, dtype=np.float64)
    n = len(pts_c)
    out_u = np.empty(n, dtype=np.float64)
    out_r = np.empty(n, dtype=np.int32)
    if (
        left.dtype == np.uint8
        and right.dtype == np.uint8
        and hasattr(lib, "verify_stereo_zncc_batch_u8")
    ):
        left_c = np.ascontiguousarray(left, dtype=np.uint8)
        right_c = np.ascontiguousarray(right, dtype=np.uint8)
        ok = lib.verify_stereo_zncc_batch_u8(
            left_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            right_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
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
    else:
        left_c = np.ascontiguousarray(left, dtype=np.float32)
        right_c = np.ascontiguousarray(right, dtype=np.float32)
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


def _get_ba_lib() -> ctypes.CDLL | None:
    lib = _load_library("fast_ba")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.solve_schur_local_bundle_cpp.argtypes = [
            ctypes.c_int32,  # n_cams
            ctypes.c_int32,  # n_free
            ctypes.c_int32,  # n_pt
            ctypes.c_int32,  # n_rec
            ctypes.c_int32,  # n_held
            ctypes.POINTER(ctypes.c_double),  # base_cams_R
            ctypes.POINTER(ctypes.c_double),  # base_cams_t
            ctypes.POINTER(ctypes.c_int32),   # free_cam_idx
            ctypes.POINTER(ctypes.c_double),  # x_free_poses
            ctypes.POINTER(ctypes.c_double),  # x_points
            ctypes.POINTER(ctypes.c_int32),   # rec_cams
            ctypes.POINTER(ctypes.c_int32),   # rec_pts
            ctypes.POINTER(ctypes.c_double),  # rec_pixels
            ctypes.POINTER(ctypes.c_double),  # rec_right
            ctypes.POINTER(ctypes.c_uint8),   # rec_mask
            ctypes.POINTER(ctypes.c_double),  # rec_weights
            ctypes.POINTER(ctypes.c_int32),   # held_cams
            ctypes.POINTER(ctypes.c_double),  # held_pts
            ctypes.POINTER(ctypes.c_double),  # held_pixels
            ctypes.POINTER(ctypes.c_double),  # held_right
            ctypes.POINTER(ctypes.c_uint8),   # held_mask
            ctypes.POINTER(ctypes.c_double),  # K
            ctypes.c_double,  # baseline
            ctypes.c_double,  # disparity_offset
            ctypes.c_int32,   # metric
            ctypes.c_double,  # f_scale
            ctypes.c_int32,   # max_iter
            ctypes.POINTER(ctypes.c_int32),   # out_status
            ctypes.POINTER(ctypes.c_int32),   # out_nfev
            ctypes.POINTER(ctypes.c_double),  # out_cost
            ctypes.POINTER(ctypes.c_double),  # out_optimality
        ]
        lib.solve_schur_local_bundle_cpp.restype = ctypes.c_int32
        lib._configured = True
    return lib


def solve_schur_local_bundle_cpp(
    base_cams: np.ndarray,
    free_cam_idx: np.ndarray,
    x_free_poses: np.ndarray,
    x_points: np.ndarray,
    rec_cams: np.ndarray,
    rec_pts: np.ndarray,
    rec_pixels: np.ndarray,
    rec_right: np.ndarray,
    rec_mask: np.ndarray,
    rec_weights: np.ndarray,
    held_cams: np.ndarray,
    held_pts: np.ndarray,
    held_pixels: np.ndarray,
    held_right: np.ndarray,
    held_mask: np.ndarray,
    K: np.ndarray,
    baseline: float,
    disparity_offset: float,
    metric: bool,
    f_scale: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float] | None:
    """Run native C++ exact Schur-complement Huber-IRLS LM local BA solver."""
    lib = _get_ba_lib()
    if lib is None:
        return None
    n_cams = len(base_cams)
    n_free = len(free_cam_idx)
    n_pt = len(x_points) // 3
    n_rec = len(rec_cams)
    n_held = len(held_cams)

    base_R = np.ascontiguousarray(base_cams[:, :3, :3], dtype=np.float64)
    base_t = np.ascontiguousarray(base_cams[:, :3, 3], dtype=np.float64)
    fci = np.ascontiguousarray(free_cam_idx, dtype=np.int32)
    fp = np.ascontiguousarray(x_free_poses, dtype=np.float64).copy()
    xp = np.ascontiguousarray(x_points, dtype=np.float64).copy()

    rc = np.ascontiguousarray(rec_cams, dtype=np.int32)
    rp = np.ascontiguousarray(rec_pts, dtype=np.int32)
    rpx = np.ascontiguousarray(rec_pixels, dtype=np.float64)
    rru = np.ascontiguousarray(rec_right, dtype=np.float64)
    rm = np.ascontiguousarray(rec_mask, dtype=np.uint8)
    rw = np.ascontiguousarray(rec_weights, dtype=np.float64)

    hc = np.ascontiguousarray(held_cams, dtype=np.int32) if n_held else np.empty(1, dtype=np.int32)
    hpt = np.ascontiguousarray(held_pts, dtype=np.float64) if n_held else np.empty(3, dtype=np.float64)
    hpx = np.ascontiguousarray(held_pixels, dtype=np.float64) if n_held else np.empty(2, dtype=np.float64)
    hru = np.ascontiguousarray(held_right, dtype=np.float64) if n_held else np.empty(1, dtype=np.float64)
    hm = np.ascontiguousarray(held_mask, dtype=np.uint8) if n_held else np.empty(3, dtype=np.uint8)

    K_c = np.ascontiguousarray(K, dtype=np.float64)
    out_status = ctypes.c_int32(0)
    out_nfev = ctypes.c_int32(0)
    out_cost = ctypes.c_double(0.0)
    out_opt = ctypes.c_double(0.0)

    ok = lib.solve_schur_local_bundle_cpp(
        n_cams,
        n_free,
        n_pt,
        n_rec,
        n_held,
        base_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        base_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fci.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        fp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        xp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        rp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        rpx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rru.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        hpt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hpx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hru.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        K_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(baseline),
        float(disparity_offset),
        1 if metric else 0,
        float(f_scale),
        int(max_iter),
        ctypes.byref(out_status),
        ctypes.byref(out_nfev),
        ctypes.byref(out_cost),
        ctypes.byref(out_opt),
    )
    if ok != 1:
        return None
    return fp, xp, int(out_status.value), int(out_nfev.value), float(out_cost.value), float(out_opt.value)


def _get_lk_lib() -> ctypes.CDLL | None:
    lib = _load_library("fast_lk")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.lk_pyramid_create.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.lk_pyramid_create.restype = ctypes.c_void_p
        lib.lk_pyramid_destroy.argtypes = [ctypes.c_void_p]
        lib.lk_pyramid_destroy.restype = None
        lib.lk_track_pyramids_avx512.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.lk_track_pyramids_avx512.restype = ctypes.c_int32
        lib._configured = True
    return lib


class LkPyramid:
    """RAII wrapper for a cached C++ multi-level LK image pyramid."""

    __slots__ = ("_handle", "_lib", "max_level", "shape", "source_gray")

    def __init__(self, handle: int, lib: ctypes.CDLL, max_level: int, shape: tuple[int, int], source_gray: np.ndarray):
        self._handle = handle
        self._lib = lib
        self.max_level = int(max_level)
        self.shape = shape
        self.source_gray = source_gray

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if handle and lib is not None:
            self._handle = None
            try:
                lib.lk_pyramid_destroy(handle)
            except Exception:
                pass


def build_lk_pyramid_avx512(gray: np.ndarray, max_level: int = 4) -> LkPyramid | None:
    """Build a reusable 5-level AVX-512 optical flow pyramid for a uint8 grayscale frame."""
    lib = _get_lk_lib()
    if lib is None or gray is None or gray.dtype != np.uint8 or gray.ndim != 2:
        return None
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return None
    img = np.ascontiguousarray(gray, dtype=np.uint8)
    handle = lib.lk_pyramid_create(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        img.shape[0],
        img.shape[1],
        int(max_level),
    )
    if not handle:
        return None
    return LkPyramid(handle, lib, int(max_level), (int(img.shape[0]), int(img.shape[1])), gray)


def track_lk_pyramids_avx512(
    prev_pyr: LkPyramid,
    next_pyr: LkPyramid,
    prev_pts: np.ndarray,
    next_init: np.ndarray | None,
    *,
    max_level: int = 4,
    use_initial_flow: bool = True,
    max_iter: int = 30,
    eps: float = 0.01,
    min_eig_thresh: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run AVX-512 inverse-compositional pyramidal LK between two cached LkPyramid objects."""
    lib = _get_lk_lib()
    if lib is None or prev_pyr is None or next_pyr is None or not prev_pyr._handle or not next_pyr._handle:
        return None
    p0 = np.ascontiguousarray(prev_pts.reshape(-1, 2), dtype=np.float32)
    n = len(p0)
    if n == 0:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.uint8)
    if use_initial_flow and next_init is not None:
        p1 = np.ascontiguousarray(next_init.reshape(-1, 2), dtype=np.float32).copy()
        use_init_flag = 1
    else:
        p1 = p0.copy()
        use_init_flag = 0
    st = np.empty(n, dtype=np.uint8)
    ok = lib.lk_track_pyramids_avx512(
        prev_pyr._handle,
        next_pyr._handle,
        p0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        p1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        st.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        n,
        10,
        int(max_level),
        use_init_flag,
        int(max_iter),
        float(eps * eps),
        float(min_eig_thresh),
    )
    if ok != 1:
        return None
    return p1.reshape(n, 1, 2), st.reshape(n, 1)


