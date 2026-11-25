"""Spatial grid-bucketed SIFT extraction with octave/angle/response metadata and stereo depth uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Callable

import cv2 as cv
import numpy as np


def unpack_sift_octave(octave_packed: int) -> int:
    """Decode OpenCV SIFT packed octave integer into signed octave index."""
    oct_val = int(octave_packed) & 255
    if oct_val >= 128:
        oct_val -= 256
    return oct_val


def validate_sift_detection_scale(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 < value <= 1:
        raise ValueError("sift_detection_scale must be a number in (0, 1]")
    return float(value)


class SiftDescriptorReadback:
    """Resolve one pending descriptor copy to host memory."""

    def __init__(self, wait: Callable[[], np.ndarray | None]):
        self._wait = wait
        self._descriptors = None

    def finish(self) -> np.ndarray | None:
        if self._wait is not None:
            wait, self._wait = self._wait, None
            self._descriptors = wait()
        return self._descriptors


@dataclass(frozen=True)
class BucketedKeypoints:
    keypoints: list[cv.KeyPoint]
    pixels: np.ndarray       # float32 [N, 2]
    descriptors: np.ndarray | SiftDescriptorReadback  # float32 [N, 128] after readback
    octaves: np.ndarray      # int16 [N]
    angles: np.ndarray       # float32 [N]
    responses: np.ndarray    # float32 [N]


@dataclass(frozen=True)
class StereoFrameFeatures:
    pixels: np.ndarray              # float32 [N, 2]
    descriptors: np.ndarray         # float32 [N, 128]
    octaves: np.ndarray             # int16 [N]
    angles: np.ndarray              # float32 [N]
    responses: np.ndarray           # float32 [N]
    disparities: np.ndarray         # float32 [N], -1.0 if unmatched
    right_u: np.ndarray             # float32 [N], NaN if unmatched
    depths_m: np.ndarray            # float32 [N], NaN if unmatched
    sigma_disparity_px: np.ndarray  # float32 [N]
    is_close_stereo: np.ndarray     # bool [N], 0.1 < Z < 40 * baseline
    is_far_bearing: np.ndarray      # bool [N]


def select_grid_bucketed_indices(
    pixels: np.ndarray,
    responses: np.ndarray,
    image_size: tuple[int, int],
    target_features: int = 1500,
    grid_cols: int = 16,
    grid_rows: int = 6,
) -> np.ndarray:
    """Select up to target_features indices distributed uniformly across a grid_cols x grid_rows grid.

    Uses per-cell quota first (sorted by response descending within each cell), then
    backfills remaining budget globally by highest response from unselected keypoints.
    """
    n = len(pixels)
    target = max(0, int(target_features))
    if n == 0 or target == 0:
        return np.empty(0, dtype=np.int64)
    if n <= target:
        return np.arange(n, dtype=np.int64)

    width, height = float(image_size[0]), float(image_size[1])
    cols = max(1, int(grid_cols))
    rows = max(1, int(grid_rows))
    n_cells = cols * rows
    cell_quota = max(1, int(np.ceil(target / n_cells)))

    # Sort all keypoints by descending response (stable tie-break by index)
    order = np.argsort(-responses, kind="mergesort")
    ordered_px = pixels[order]

    gx = np.clip((ordered_px[:, 0] * (cols / max(width, 1.0))).astype(np.int32), 0, cols - 1)
    gy = np.clip((ordered_px[:, 1] * (rows / max(height, 1.0))).astype(np.int32), 0, rows - 1)
    cell_ids = gy * cols + gx

    selected_mask = np.zeros(n, dtype=bool)
    cell_counts = [0] * n_cells
    cell_id_list = cell_ids.tolist()
    selected_positions: list[int] = []

    # Pass 1: fill each spatial cell up to cell_quota in descending response order
    for pos, cid in enumerate(cell_id_list):
        cnt = cell_counts[cid]
        if cnt < cell_quota:
            cell_counts[cid] = cnt + 1
            selected_positions.append(pos)
            if len(selected_positions) >= target:
                break

    if selected_positions:
        selected_mask[selected_positions] = True
    selected_count = len(selected_positions)

    # Pass 2: backfill any remaining budget globally by descending response
    if selected_count < target:
        remaining = np.flatnonzero(~selected_mask)
        need = target - selected_count
        selected_mask[remaining[:need]] = True

    chosen = order[selected_mask]
    # Keep deterministic ascending index order or descending response order
    return np.sort(chosen).astype(np.int64)


def extract_grid_bucketed_sift(
    image_gray: np.ndarray,
    target_features: int = 1500,
    grid_cols: int = 16,
    grid_rows: int = 6,
    contrast_threshold: float = 0.02,
    detector: cv.SIFT | None = None,
    *,
    build_cv_keypoints: bool = True,
    native_sift=None,
    detection_scale: float = 1.0,
) -> BucketedKeypoints:
    """Detect SIFT keypoints with spatial grid bucketing and compute descriptors only for kept keypoints."""
    if image_gray is None or np.ndim(image_gray) != 2:
        raise ValueError("image_gray must be a 2D grayscale image")
    height, width = image_gray.shape
    target = max(1, int(target_features))
    scale = validate_sift_detection_scale(detection_scale)
    if scale < 1.0:
        small_width, small_height = round(width * scale), round(height * scale)
        if min(small_width, small_height) >= 16 and (small_width, small_height) != (width, height):
            reduced = cv.resize(image_gray, (small_width, small_height), interpolation=cv.INTER_AREA)
            features = extract_grid_bucketed_sift(
                reduced, target, grid_cols, grid_rows, contrast_threshold, detector,
                build_cv_keypoints=build_cv_keypoints, native_sift=native_sift,
            )
            if len(features.pixels) >= min(target, 100):
                # Map detector pixel centers to the original coordinate grid.
                ratios = np.array([width / small_width, height / small_height], dtype=np.float32)
                pixels = (features.pixels + np.float32(0.5)) * ratios - np.float32(0.5)
                if features.keypoints:
                    size_ratio = float(np.sqrt(ratios.prod()))
                    for keypoint, pixel in zip(features.keypoints, pixels):
                        keypoint.pt = (float(pixel[0]), float(pixel[1]))
                        keypoint.size *= size_ratio
                return replace(features, pixels=pixels)
            if isinstance(features.descriptors, SiftDescriptorReadback):
                features.descriptors.finish()

    if (detector is None or isinstance(detector, cv.SIFT)) and image_gray.dtype == np.uint8:
        from native_accel import extract_sift_grid_avx512

        native_args = dict(
            target_features=target,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            contrast_threshold=contrast_threshold,
        )
        native_res = native_sift(image_gray, **native_args) if native_sift is not None else None
        if native_res is None:
            native_res = extract_sift_grid_avx512(image_gray, **native_args)
        if native_res is not None:
            pixels, kept_desc, octaves, angles, responses = native_res
            kept_kps = (
                [
                    cv.KeyPoint(
                        float(pixels[i, 0]),
                        float(pixels[i, 1]),
                        float(3.2 * (2.0 ** int(octaves[i]))),
                        float(angles[i]),
                        float(responses[i]),
                        int(octaves[i]) & 255,
                    )
                    for i in range(len(pixels))
                ]
                if build_cv_keypoints
                else []
            )
            return BucketedKeypoints(
                keypoints=kept_kps,
                pixels=pixels,
                descriptors=kept_desc,
                octaves=octaves,
                angles=angles,
                responses=responses,
            )
    if detector is None:
        # Single-pass detectAndCompute avoids rebuilding the SIFT DoG pyramid twice
        raw_cap = max(int(target * 1.4), 2100)
        detector = cv.SIFT_create(
            nfeatures=raw_cap,
            contrastThreshold=float(contrast_threshold),
        )
    keypoints, desc = detector.detectAndCompute(image_gray, None)
    if not keypoints or desc is None or len(desc) == 0:
        return BucketedKeypoints(
            keypoints=[],
            pixels=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            octaves=np.empty(0, dtype=np.int16),
            angles=np.empty(0, dtype=np.float32),
            responses=np.empty(0, dtype=np.float32),
        )

    all_pixels = cv.KeyPoint_convert(keypoints).reshape(-1, 2)
    all_responses = np.fromiter((kp.response for kp in keypoints), dtype=np.float32, count=len(keypoints))

    kept_idx = select_grid_bucketed_indices(
        all_pixels,
        all_responses,
        (width, height),
        target_features=target,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    kept_kps = [keypoints[int(i)] for i in kept_idx]
    pixels = all_pixels[kept_idx]
    kept_desc = np.asarray(desc[kept_idx], dtype=np.float32)
    raw_oct = np.fromiter((kp.octave for kp in kept_kps), dtype=np.int32, count=len(kept_kps))
    oct_u8 = raw_oct & 255
    octaves = np.where(oct_u8 >= 128, oct_u8 - 256, oct_u8).astype(np.int16)
    angles = np.fromiter((kp.angle for kp in kept_kps), dtype=np.float32, count=len(kept_kps))
    responses = all_responses[kept_idx]
    return BucketedKeypoints(
        keypoints=kept_kps,
        pixels=pixels,
        descriptors=kept_desc,
        octaves=octaves,
        angles=angles,
        responses=responses,
    )


def classify_stereo_features(
    bucketed: BucketedKeypoints,
    right_u: np.ndarray,
    fx: float,
    baseline: float,
    disparity_offset: float = 0.0,
    close_baseline_multiple: float = 40.0,
    min_depth_m: float = 0.1,
    base_sigma_d_px: float = 0.75,
) -> StereoFrameFeatures:
    """Attach stereo disparities, depths, disparity uncertainty, and close/far classification."""
    n = len(bucketed.pixels)
    right_arr = np.asarray(right_u, dtype=np.float32).reshape(-1)
    if len(right_arr) != n:
        raise ValueError(f"right_u length {len(right_arr)} != pixels length {n}")

    valid_r = np.isfinite(right_arr)
    disp = np.full(n, -1.0, dtype=np.float32)
    if np.any(valid_r):
        disp_vals = (bucketed.pixels[valid_r, 0] - right_arr[valid_r] - float(disparity_offset)).astype(np.float32)
        disp[valid_r] = np.where(disp_vals > 0.0, disp_vals, -1.0)

    valid_d = disp > 1e-3
    depths = np.full(n, np.nan, dtype=np.float32)
    fb = float(fx) * float(baseline)
    if np.any(valid_d) and fb > 0.0:
        depths[valid_d] = (fb / disp[valid_d]).astype(np.float32)

    # Scale-dependent disparity uncertainty: higher SIFT octaves have coarser localization
    oct_scale = np.power(1.2, np.clip(bucketed.octaves.astype(np.float32), 0.0, 6.0))
    sigma_d = (float(base_sigma_d_px) * oct_scale).astype(np.float32)

    close_max_z = float(close_baseline_multiple) * float(baseline)
    is_close = valid_d & np.isfinite(depths) & (depths > float(min_depth_m)) & (depths < close_max_z)
    # Far bearing candidates: either depth >= 40*B or unmatched/low-disparity high-confidence feature
    is_far = ~is_close

    return StereoFrameFeatures(
        pixels=bucketed.pixels,
        descriptors=bucketed.descriptors,
        octaves=bucketed.octaves,
        angles=bucketed.angles,
        responses=bucketed.responses,
        disparities=disp,
        right_u=right_arr,
        depths_m=depths,
        sigma_disparity_px=sigma_d,
        is_close_stereo=is_close,
        is_far_bearing=is_far,
    )
