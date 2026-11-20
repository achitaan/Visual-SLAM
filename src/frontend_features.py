"""Spatial grid-bucketed SIFT extraction with octave/angle/response metadata and stereo depth uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


def unpack_sift_octave(octave_packed: int) -> int:
    """Decode OpenCV SIFT packed octave integer into signed octave index."""
    oct_val = int(octave_packed) & 255
    if oct_val >= 128:
        oct_val -= 256
    return oct_val


@dataclass(frozen=True)
class BucketedKeypoints:
    keypoints: list[cv.KeyPoint]
    pixels: np.ndarray       # float32 [N, 2]
    descriptors: np.ndarray  # float32 [N, 128]
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
    cell_counts = np.zeros(n_cells, dtype=np.int32)
    selected_count = 0

    # Pass 1: fill each spatial cell up to cell_quota in descending response order
    for pos in range(n):
        cid = int(cell_ids[pos])
        if cell_counts[cid] < cell_quota:
            cell_counts[cid] += 1
            selected_mask[pos] = True
            selected_count += 1
            if selected_count >= target:
                break

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
) -> BucketedKeypoints:
    """Detect SIFT keypoints with spatial grid bucketing and compute descriptors only for kept keypoints."""
    if image_gray is None or np.ndim(image_gray) != 2:
        raise ValueError("image_gray must be a 2D grayscale image")
    height, width = image_gray.shape
    target = max(1, int(target_features))
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

    all_pixels = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    all_responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

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
    octaves = np.array([unpack_sift_octave(kp.octave) for kp in kept_kps], dtype=np.int16)
    angles = np.array([kp.angle for kp in kept_kps], dtype=np.float32)
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
