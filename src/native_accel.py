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

# Match the explicit ISA targets used by the Windows build script. NumPy's
# runtime feature table accounts for both CPU support and OS SIMD state.
_WINDOWS_SIMD_REQUIREMENTS = {
    "fast_matcher": ("AVX512F", "AVX512VNNI"),
    "fast_zncc": ("AVX512F",),
    "fast_sift": ("AVX512F", "AVX512BW", "AVX512VL", "AVX512DQ"),
    "fast_lk": ("AVX512F",),
}


def windows_missing_simd(name: str) -> tuple[str, ...]:
    required = _WINDOWS_SIMD_REQUIREMENTS.get(name, ())
    if not required:
        return ()
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        from numpy._core._multiarray_umath import __cpu_features__ as features
    else:
        from numpy.core._multiarray_umath import __cpu_features__ as features
    return tuple(flag for flag in required if not features.get(flag, False))


def native_enabled() -> bool:
    return os.environ.get("VISUAL_SLAM_DISABLE_NATIVE", "0").strip().lower() not in (
        "1", "true", "yes", "on"
    )


def _load_library(name: str) -> ctypes.CDLL | None:
    if not native_enabled():
        return None
    if name in _LOADED:
        return _LOADED[name]
    windows = os.name == "nt"
    if windows and windows_missing_simd(name):
        _LOADED[name] = None
        return None
    lib_path = _KERNEL_DIR / f"lib{name}{'.dll' if windows else '.so'}"
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
        if hasattr(lib, "match_projected_landmarks_f32"):
            lib.match_projected_landmarks_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_int32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            lib.match_projected_landmarks_f32.restype = ctypes.c_int32
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


def match_projected_landmarks_avx512(
    proj_uv: np.ndarray,
    lm_descs: np.ndarray,
    lm_oct: np.ndarray | None,
    f_pixels: np.ndarray,
    f_descs: np.ndarray,
    f_oct: np.ndarray | None,
    *,
    radius_px: float,
    ratio: float,
    max_octave_diff: int,
    max_l2_distance: float,
    max_single_l2_distance: float,
) -> np.ndarray | None:
    """AVX-512 spatial-grid projective landmark matcher (exact float32 L2^2 + ratio + 1-to-1)."""
    lib = _get_matcher_lib()
    if lib is None or not hasattr(lib, "match_projected_landmarks_f32"):
        return None
    if lm_descs.ndim != 2 or lm_descs.shape[1] != 128 or f_descs.ndim != 2 or f_descs.shape[1] != 128:
        return None
    p_uv = np.ascontiguousarray(proj_uv, dtype=np.float32)
    l_d = np.ascontiguousarray(lm_descs, dtype=np.float32)
    f_px = np.ascontiguousarray(f_pixels, dtype=np.float32)
    f_d = np.ascontiguousarray(f_descs, dtype=np.float32)
    n_lm = len(p_uv)
    n_feat = len(f_px)
    if n_lm == 0 or n_feat == 0:
        return np.empty((0, 2), dtype=np.int32)
    if lm_oct is not None and f_oct is not None and max_octave_diff >= 0:
        l_o = np.ascontiguousarray(lm_oct, dtype=np.int16)
        f_o = np.ascontiguousarray(f_oct, dtype=np.int16)
        l_o_ptr = l_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        f_o_ptr = f_o.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    else:
        l_o_ptr = ctypes.POINTER(ctypes.c_int16)()
        f_o_ptr = ctypes.POINTER(ctypes.c_int16)()
    out_pairs = np.empty((min(n_lm, n_feat), 2), dtype=np.int32)
    out_cnt = ctypes.c_int32(0)
    status = lib.match_projected_landmarks_f32(
        p_uv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        l_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        l_o_ptr,
        n_lm,
        f_px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        f_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        f_o_ptr,
        n_feat,
        float(radius_px),
        float(ratio),
        int(max_octave_diff),
        float(max_l2_distance),
        float(max_single_l2_distance),
        out_pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(out_cnt),
    )
    if status != 1:
        return None
    return out_pairs[: out_cnt.value]


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
        if hasattr(lib, "extract_sift_pyramid_grid_avx512") and hasattr(lib, "sift_pyramid_parameters"):
            lib.extract_sift_pyramid_grid_avx512.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, *lib.extract_sift_grid_avx512.argtypes[1:],
            ]
            lib.extract_sift_pyramid_grid_avx512.restype = ctypes.c_int32
            lib.sift_pyramid_parameters.argtypes = [ctypes.c_int32] * 3 + [ctypes.c_void_p] * 4
            lib.sift_pyramid_parameters.restype = ctypes.c_int32
        lib._configured = True
    return lib


def native_sift_first_octave() -> int:
    return 0 if os.environ.get("VISUAL_SLAM_SIFT_FIRST_OCTAVE", "").strip() == "0" else -1


def sift_pyramid_available() -> bool:
    lib = _get_sift_lib()
    return (lib is not None and hasattr(lib, "extract_sift_pyramid_grid_avx512")
            and hasattr(lib, "sift_pyramid_parameters"))


def compact_sift_functions():
    """Return the native compact detector and descriptor weights, if available."""
    lib = _get_sift_lib()
    if lib is None or not hasattr(lib, "detect_sift_sparse_seeds") or not hasattr(lib, "copy_sift_descriptor_weights"):
        return None
    detect = lib.detect_sift_sparse_seeds
    detect.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int32,
        *([ctypes.c_int32] * 5), ctypes.c_float, ctypes.c_int32, *([ctypes.c_void_p] * 6),
    ]
    detect.restype = ctypes.c_int32
    copy_weights = lib.copy_sift_descriptor_weights
    copy_weights.argtypes = [ctypes.c_void_p]
    copy_weights.restype = None
    return detect, copy_weights


def sift_pyramid_parameters(shape: tuple[int, int], first_octave: int):
    """Return the octave shapes and exact integer coefficients from the native SIFT kernel."""
    lib = _get_sift_lib()
    if lib is None or not hasattr(lib, "sift_pyramid_parameters"):
        return None
    heights = np.empty(6, dtype=np.int32)
    widths = np.empty(6, dtype=np.int32)
    weights = np.zeros((6, 25), dtype=np.int16)
    radii = np.empty(6, dtype=np.int32)
    count = lib.sift_pyramid_parameters(
        *shape, first_octave, heights.ctypes.data, widths.ctypes.data,
        weights.ctypes.data, radii.ctypes.data,
    )
    if not 1 <= count <= len(heights):
        return None
    return tuple(zip(heights[:count].tolist(), widths[:count].tolist())), weights, tuple(radii.tolist())


def extract_sift_grid_avx512(
    image_gray: np.ndarray,
    *,
    target_features: int = 1500,
    grid_cols: int = 16,
    grid_rows: int = 6,
    contrast_threshold: float = 0.02,
    first_octave: int | None = None,
    pyramid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run AVX-512 C++ SIFT with early spatial grid bucketing before descriptor computation."""
    lib = _get_sift_lib()
    if lib is None or image_gray.dtype != np.uint8 or image_gray.ndim != 2:
        return None
    if first_octave is None:
        first_octave = native_sift_first_octave()
    if pyramid is None:
        img = np.ascontiguousarray(image_gray, dtype=np.uint8)
        extract = lib.extract_sift_grid_avx512
        source = (img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),)
    elif (pyramid.dtype == np.int16 and pyramid.ndim == 1 and pyramid.flags.c_contiguous
            and hasattr(lib, "extract_sift_pyramid_grid_avx512")):
        extract = lib.extract_sift_pyramid_grid_avx512
        source = (pyramid.ctypes.data, pyramid.size)
    else:
        return None
    target = max(1, int(target_features))
    px = np.empty((target, 2), dtype=np.float32)
    desc = np.empty((target, 128), dtype=np.float32)
    octs = np.empty(target, dtype=np.int16)
    angles = np.empty(target, dtype=np.float32)
    responses = np.empty(target, dtype=np.float32)
    cnt = ctypes.c_int32(0)
    ok = extract(
        *source,
        image_gray.shape[0],
        image_gray.shape[1],
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
        if hasattr(lib, "project_rank3_camera_rows"):
            lib.project_rank3_camera_rows.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
            ]
            lib.project_rank3_camera_rows.restype = ctypes.c_int32
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
        lib.refine_stereo_pose_lm_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # R_cw
            ctypes.POINTER(ctypes.c_double),  # t_cw
            ctypes.POINTER(ctypes.c_double),  # pts_w
            ctypes.POINTER(ctypes.c_double),  # obs
            ctypes.POINTER(ctypes.c_double),  # weights
            ctypes.POINTER(ctypes.c_uint8),   # mask
            ctypes.c_int32,   # n
            ctypes.c_double,  # fx
            ctypes.c_double,  # fy
            ctypes.c_double,  # cx
            ctypes.c_double,  # cy
            ctypes.c_double,  # fb
            ctypes.c_double,  # doff
            ctypes.c_double,  # delta_huber
            ctypes.c_int32,   # max_iter
            ctypes.POINTER(ctypes.c_double),  # out_initial_cost
            ctypes.POINTER(ctypes.c_double),  # out_final_cost
            ctypes.POINTER(ctypes.c_int32),   # out_converged
            ctypes.POINTER(ctypes.c_double),  # out_H_final
        ]
        lib.refine_stereo_pose_lm_f64.restype = ctypes.c_int32
        lib.refine_bidirectional_stereo_lm_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # R_io
            ctypes.POINTER(ctypes.c_double),  # t_io
            ctypes.POINTER(ctypes.c_double),  # src_pts
            ctypes.POINTER(ctypes.c_double),  # tgt_pix
            ctypes.c_int32,   # n1
            ctypes.POINTER(ctypes.c_double),  # tgt_pts
            ctypes.POINTER(ctypes.c_double),  # src_pix
            ctypes.c_int32,   # n2
            ctypes.c_double,  # fx
            ctypes.c_double,  # fy
            ctypes.c_double,  # cx
            ctypes.c_double,  # cy
            ctypes.c_double,  # f_scale
            ctypes.c_int32,   # max_iter
            ctypes.POINTER(ctypes.c_double),  # out_initial_cost
            ctypes.POINTER(ctypes.c_double),  # out_final_cost
        ]
        lib.refine_bidirectional_stereo_lm_f64.restype = ctypes.c_int32
        lib._configured = True
    return lib


def project_rank3_camera_rows(
    point_rows: np.ndarray,
    camera_rows: np.ndarray,
    camera_slots: np.ndarray,
    pose_columns: int,
) -> np.ndarray | None:
    """Eliminate well-conditioned point blocks or defer to the reference SVD."""
    lib = _get_ba_lib()
    if lib is None or not hasattr(lib, "project_rank3_camera_rows") or camera_slots.ndim != 2:
        return None
    batch, rows = camera_slots.shape
    if (not batch or not 4 <= rows <= 64 or not 0 < pose_columns <= 60
            or pose_columns % 6 or point_rows.shape != (batch, rows, 3)
            or camera_rows.shape != (batch, rows, 6)):
        return None
    points = np.ascontiguousarray(point_rows, dtype=np.float64)
    cameras = np.ascontiguousarray(camera_rows, dtype=np.float64)
    slots = np.ascontiguousarray(camera_slots, dtype=np.int32)
    projected = np.empty((batch, rows - 3, pose_columns), dtype=np.float64)
    status = lib.project_rank3_camera_rows(points.ctypes.data, cameras.ctypes.data,
                                           slots.ctypes.data, batch, rows, pose_columns,
                                           projected.ctypes.data)
    return projected if status == 1 else None


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


def refine_stereo_pose_lm_cpp(
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    pts_w: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fb: float,
    doff: float,
    delta_huber: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, float, bool, np.ndarray] | None:
    """Native C++ 6-DoF Huber-IRLS LM stereo pose refiner."""
    lib = _get_ba_lib()
    if lib is None:
        return None
    n = len(pts_w)
    R_arr = np.ascontiguousarray(R_cw, dtype=np.float64).reshape(3, 3).copy()
    t_arr = np.ascontiguousarray(t_cw, dtype=np.float64).reshape(3).copy()
    pts_c = np.ascontiguousarray(pts_w, dtype=np.float64).reshape(n, 3)
    obs_c = np.ascontiguousarray(obs, dtype=np.float64).reshape(n, 3)
    w_c = np.ascontiguousarray(weights, dtype=np.float64).reshape(n, 3)
    m_c = np.ascontiguousarray(mask, dtype=np.uint8).reshape(n, 3)
    out_init = ctypes.c_double(0.0)
    out_final = ctypes.c_double(0.0)
    out_conv = ctypes.c_int32(0)
    H_out = np.zeros((6, 6), dtype=np.float64)
    ok = lib.refine_stereo_pose_lm_f64(
        R_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        obs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        m_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(n),
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        float(fb),
        float(doff),
        float(delta_huber),
        int(max_iter),
        ctypes.byref(out_init),
        ctypes.byref(out_final),
        ctypes.byref(out_conv),
        H_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if ok != 1:
        return None
    return R_arr, t_arr, float(out_init.value), float(out_final.value), bool(out_conv.value), H_out


def refine_bidirectional_stereo_lm_cpp(
    R_init: np.ndarray,
    t_init: np.ndarray,
    src_pts: np.ndarray,
    tgt_pix: np.ndarray,
    tgt_pts: np.ndarray,
    src_pix: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    f_scale: float = 1.5,
    max_iter: int = 15,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Native C++ 6-DoF Huber-IRLS LM bidirectional stereo pose refiner."""
    lib = _get_ba_lib()
    if lib is None:
        return None
    n1 = len(src_pts)
    n2 = len(tgt_pts)
    R_arr = np.ascontiguousarray(R_init, dtype=np.float64).reshape(3, 3).copy()
    t_arr = np.ascontiguousarray(t_init, dtype=np.float64).reshape(3).copy()
    sp = np.ascontiguousarray(src_pts, dtype=np.float64).reshape(n1, 3)
    tpix = np.ascontiguousarray(tgt_pix, dtype=np.float64).reshape(n1, 2)
    tp = np.ascontiguousarray(tgt_pts, dtype=np.float64).reshape(n2, 3)
    spix = np.ascontiguousarray(src_pix, dtype=np.float64).reshape(n2, 2)
    out_init = ctypes.c_double(0.0)
    out_final = ctypes.c_double(0.0)
    _dbl_p = ctypes.POINTER(ctypes.c_double)
    ok = lib.refine_bidirectional_stereo_lm_f64(
        R_arr.ctypes.data_as(_dbl_p),
        t_arr.ctypes.data_as(_dbl_p),
        sp.ctypes.data_as(_dbl_p) if n1 else None,
        tpix.ctypes.data_as(_dbl_p) if n1 else None,
        int(n1),
        tp.ctypes.data_as(_dbl_p) if n2 else None,
        spix.ctypes.data_as(_dbl_p) if n2 else None,
        int(n2),
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        float(f_scale),
        int(max_iter),
        ctypes.byref(out_init),
        ctypes.byref(out_final),
    )
    if ok != 1:
        return None
    return R_arr, t_arr, float(out_init.value), float(out_final.value)


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


def _get_pgo_lib() -> ctypes.CDLL | None:
    lib = _load_library("fast_pgo")
    if lib is not None and not getattr(lib, "_configured", False):
        lib.solve_sparse_se3_pgo.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.solve_sparse_se3_pgo.restype = ctypes.c_int32
        lib._configured = True
    return lib


def solve_sparse_se3_pgo_native(
    poses: np.ndarray,
    fi: np.ndarray,
    si: np.ndarray,
    meas: np.ndarray,
    W: np.ndarray,
    huber_scale: float = 1.0,
    max_iter: int = 50,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Solve sparse SE(3) pose graph via native block-6x6 MD-Cholesky LM solver."""
    lib = _get_pgo_lib()
    if lib is None:
        return None
    poses_buf = np.ascontiguousarray(poses, dtype=np.float64).copy()
    fi_buf = np.ascontiguousarray(fi, dtype=np.int32)
    si_buf = np.ascontiguousarray(si, dtype=np.int32)
    meas_buf = np.ascontiguousarray(meas, dtype=np.float64)
    W_buf = np.ascontiguousarray(W, dtype=np.float64)
    stats = np.zeros(5, dtype=np.float64)
    converged = int(
        lib.solve_sparse_se3_pgo(
            int(poses_buf.shape[0]),
            poses_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(fi_buf.shape[0]),
            fi_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            si_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            meas_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            W_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(huber_scale),
            int(max_iter),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    info = {
        "converged": bool(converged),
        "initial_cost": float(stats[0]),
        "final_cost": float(stats[1]),
        "optimality": float(stats[2]),
        "nfev": int(round(stats[3])),
        "njev": int(round(stats[4])),
    }
    return poses_buf, info
