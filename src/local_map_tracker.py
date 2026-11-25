"""Projective local-map tracking, octave-gated matching, neighbor landmark fusion, and far-point triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np

from slam_state import Observation


@dataclass(frozen=True)
class TriangulatedLandmark:
    position_w: np.ndarray
    prev_feat_idx: int
    curr_feat_idx: int
    descriptor: np.ndarray
    octave: int
    parallax_deg: float
    depth_m: float
    is_far: bool = True


def _normalize_descriptors(descs: np.ndarray) -> np.ndarray:
    arr = np.asarray(descs, dtype=np.float32)
    if arr.ndim != 2 or len(arr) == 0:
        return np.empty((0, 128), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _build_spatial_grid(
    pixels: np.ndarray,
    cell_size: float,
) -> tuple[dict[tuple[int, int], list[int]], float]:
    inv_cell = 1.0 / max(float(cell_size), 1.0)
    grid: dict[tuple[int, int], list[int]] = {}
    if len(pixels) == 0:
        return grid, inv_cell
    gx = np.floor(pixels[:, 0] * inv_cell).astype(np.int32)
    gy = np.floor(pixels[:, 1] * inv_cell).astype(np.int32)
    for idx in range(len(pixels)):
        key = (int(gx[idx]), int(gy[idx]))
        bucket = grid.get(key)
        if bucket is None:
            grid[key] = [idx]
        else:
            bucket.append(idx)
    return grid, inv_cell


def match_landmarks_by_projection(
    frustum_candidates: Any,
    frame_features: Any,
    landmark_descriptors: np.ndarray,
    landmark_octaves: np.ndarray | None = None,
    radius_px: float = 36.0,
    ratio: float = 0.75,
    max_octave_diff: int = 1,
    max_l2_distance: float = 240.0,
    max_single_l2_distance: float = 180.0,
) -> dict[int, int]:
    """Match projected map landmarks to current-frame features inside a local pixel radius.

    Enforces spatial radius gating, SIFT octave consistency (|delta_octave| <= max_octave_diff),
    local Lowe's ratio test, and unique 1-to-1 feature ownership (closest descriptor wins).
    Returns mapping {landmark_id: frame_feature_index}.
    """
    lm_ids = np.asarray(frustum_candidates.landmark_ids, dtype=np.int64).reshape(-1)
    proj_uv = np.asarray(frustum_candidates.projected_uv, dtype=np.float32).reshape(-1, 2)
    if len(lm_ids) == 0:
        return {}

    if hasattr(frame_features, "pixels"):
        f_pixels = np.asarray(frame_features.pixels, dtype=np.float32).reshape(-1, 2)
        f_descs = np.asarray(frame_features.descriptors, dtype=np.float32)
        f_octaves = getattr(frame_features, "octaves", None)
    else:
        f_pixels = np.asarray(frame_features["pixels"], dtype=np.float32).reshape(-1, 2)
        f_descs = np.asarray(frame_features["descriptors"], dtype=np.float32)
        f_octaves = frame_features.get("octaves")

    if len(f_pixels) == 0 or len(f_descs) == 0:
        return {}

    lm_descs = np.asarray(landmark_descriptors, dtype=np.float32)
    if len(lm_descs) != len(lm_ids):
        raise ValueError(f"landmark_descriptors length {len(lm_descs)} != landmark_ids length {len(lm_ids)}")

    use_octaves = (
        landmark_octaves is not None
        and f_octaves is not None
        and len(landmark_octaves) == len(lm_ids)
        and len(f_octaves) == len(f_pixels)
        and max_octave_diff >= 0
    )
    if use_octaves:
        lm_oct = np.asarray(landmark_octaves, dtype=np.int16).reshape(-1)
        f_oct = np.asarray(f_octaves, dtype=np.int16).reshape(-1)
    else:
        lm_oct = None
        f_oct = None

    rad = max(1.0, float(radius_px))
    try:
        from native_accel import match_projected_landmarks_avx512
        native_pairs = match_projected_landmarks_avx512(
            proj_uv,
            lm_descs,
            lm_oct,
            f_pixels,
            f_descs,
            f_oct,
            radius_px=rad,
            ratio=float(ratio),
            max_octave_diff=int(max_octave_diff) if use_octaves else -1,
            max_l2_distance=float(max_l2_distance),
            max_single_l2_distance=float(max_single_l2_distance),
        )
        if native_pairs is not None:
            return {
                int(lm_ids[lm_i]): int(feat_i)
                for lm_i, feat_i in native_pairs.tolist()
            }
    except Exception:
        pass

    rad_sq = rad * rad
    max_dist_sq = float(max_l2_distance) * float(max_l2_distance)
    max_single_sq = float(max_single_l2_distance) * float(max_single_l2_distance)
    ratio_sq = float(ratio) * float(ratio)

    # Precompute squared norms for exact fast L2^2 = ||a||^2 + ||b||^2 - 2 a.b
    f_norm2 = np.sum(f_descs * f_descs, axis=1)
    lm_norm2 = np.sum(lm_descs * lm_descs, axis=1)

    grid, inv_cell = _build_spatial_grid(f_pixels, rad)
    # feature_idx -> (best_dist_sq, landmark_id)
    best_for_feat: dict[int, tuple[float, int]] = {}

    lm_gx = np.floor(proj_uv[:, 0] * inv_cell).astype(np.int32)
    lm_gy = np.floor(proj_uv[:, 1] * inv_cell).astype(np.int32)

    for i in range(len(lm_ids)):
        cx = int(lm_gx[i])
        cy = int(lm_gy[i])
        cand_list: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = grid.get((cx + dx, cy + dy))
                if bucket:
                    cand_list.extend(bucket)
        if not cand_list:
            continue

        cands = np.asarray(cand_list, dtype=np.intp)
        diff = f_pixels[cands] - proj_uv[i]
        d_px_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        in_rad = d_px_sq <= rad_sq
        if use_octaves:
            in_rad &= np.abs(f_oct[cands] - lm_oct[i]) <= max_octave_diff
        if not np.any(in_rad):
            continue

        cands = cands[in_rad]
        dists_sq = np.maximum(
            0.0,
            f_norm2[cands] + lm_norm2[i] - 2.0 * (f_descs[cands] @ lm_descs[i]),
        )
        if len(dists_sq) == 1:
            best_pos = 0
            best_d2 = float(dists_sq[0])
            if best_d2 > max_single_sq:
                continue
        else:
            part = np.argpartition(dists_sq, 1)
            best_pos = int(part[0])
            second_pos = int(part[1])
            best_d2 = float(dists_sq[best_pos])
            second_d2 = float(dists_sq[second_pos])
            if best_d2 > max_dist_sq:
                continue
            if second_d2 > 1e-12 and best_d2 >= ratio_sq * second_d2:
                continue

        feat_idx = int(cands[best_pos])
        lm_id = int(lm_ids[i])
        prev = best_for_feat.get(feat_idx)
        if prev is None or best_d2 < prev[0]:
            best_for_feat[feat_idx] = (best_d2, lm_id)

    return {lm_id: feat_idx for feat_idx, (_d2, lm_id) in best_for_feat.items()}


def fuse_neighbor_landmarks(
    map_state: Any,
    keyframe: Any,
    neighbor_kf_ids: list[int] | set[int],
    K: np.ndarray,
    baseline: float = 0.0,
    radius_px: float = 4.0,
    max_l2_distance: float = 240.0,
    right_u_values: np.ndarray | None = None,
) -> int:
    """Project landmarks from neighbor keyframes into keyframe and fuse/attach matching observations.

    Increases multi-view track density without creating duplicate landmarks.
    Returns count of newly attached observations / fused landmarks.
    """
    if keyframe is None or len(keyframe.pixels) == 0 or not neighbor_kf_ids:
        return 0
    kf_id = int(keyframe.id)
    K_arr = np.asarray(K, dtype=np.float64)
    T_cw = np.linalg.inv(np.asarray(keyframe.pose, dtype=np.float64))
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    width, height = keyframe.image_size if keyframe.image_size is not None else (
        int(K_arr[0, 2] * 2), int(K_arr[1, 2] * 2)
    )

    with map_state.lock:
        if hasattr(map_state, "index") and len(map_state.index) == len(map_state.landmarks):
            cand_lm_ids = map_state.index.landmarks_for_keyframes(neighbor_kf_ids)
        else:
            nb_set = set(int(k) for k in neighbor_kf_ids)
            cand_lm_ids = {
                lid for lid, lm in map_state.landmarks.items()
                if any(k in nb_set for k in lm.observations)
            }

        # Filter to landmarks not yet observed in this keyframe
        unseen_lms = [
            map_state.landmarks[lid]
            for lid in cand_lm_ids
            if lid in map_state.landmarks and kf_id not in map_state.landmarks[lid].observations
        ]
        if not unseen_lms:
            return 0

        pts_w = np.array([lm.position for lm in unseen_lms], dtype=np.float64)
        pts_c = pts_w @ R_cw.T + t_cw
        z = pts_c[:, 2]
        valid = z > 0.5
        if not np.any(valid):
            return 0

        idx_v = np.flatnonzero(valid)
        pc = pts_c[idx_v]
        uv = np.column_stack((
            K_arr[0, 0] * pc[:, 0] / pc[:, 2] + K_arr[0, 2],
            K_arr[1, 1] * pc[:, 1] / pc[:, 2] + K_arr[1, 2],
        ))
        in_img = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
        )
        if not np.any(in_img):
            return 0

        idx_v = idx_v[in_img]
        uv = uv[in_img]

        f_pixels = np.asarray(keyframe.pixels, dtype=np.float32)
        f_descs = np.asarray(keyframe.descriptors, dtype=np.float32)
        f_norm2 = np.sum(f_descs * f_descs, axis=1)
        f_octs = getattr(keyframe, "octaves", None)

        rad = max(1.0, float(radius_px))
        rad_sq = rad * rad
        max_d2 = float(max_l2_distance) * float(max_l2_distance)

        fused = 0
        native_fused_pairs = None
        try:
            from native_accel import match_projected_landmarks_avx512
            vis_lms = [unseen_lms[int(p)] for p in idx_v.tolist()]
            lm_descs_arr = np.asarray([lm.descriptor for lm in vis_lms], dtype=np.float32)
            lm_octs_arr = (
                np.asarray([int(getattr(lm, "octave", 0)) for lm in vis_lms], dtype=np.int16)
                if f_octs is not None else None
            )
            native_fused_pairs = match_projected_landmarks_avx512(
                uv.astype(np.float32, copy=False),
                lm_descs_arr,
                lm_octs_arr,
                f_pixels,
                f_descs,
                f_octs,
                radius_px=rad,
                ratio=1.01,
                max_octave_diff=1 if f_octs is not None else -1,
                max_l2_distance=float(max_l2_distance),
                max_single_l2_distance=float(max_l2_distance),
            )
        except Exception:
            native_fused_pairs = None

        if native_fused_pairs is not None:
            for row_i, feat_idx in native_fused_pairs.tolist():
                lm = unseen_lms[int(idx_v[row_i])]
                existing_lid = int(keyframe.landmark_ids[feat_idx])
                if existing_lid < 0:
                    ru = None
                    if right_u_values is not None and feat_idx < len(right_u_values):
                        rv = float(right_u_values[feat_idx])
                        if np.isfinite(rv):
                            ru = rv
                    obs = Observation(
                        f_pixels[feat_idx].copy(),
                        ru,
                        octave=int(f_octs[feat_idx]) if f_octs is not None else 0,
                    )
                    keyframe.landmark_ids[feat_idx] = int(lm.id)
                    map_state.add_observation(int(lm.id), kf_id, obs)
                    fused += 1
                elif existing_lid != int(lm.id) and existing_lid in map_state.landmarks:
                    other = map_state.landmarks[existing_lid]
                    if (len(other.observations) == 1
                            and len(lm.observations) >= 2
                            and float(np.linalg.norm(other.position - lm.position)) < max(0.5, 0.05 * float(pc[row_i, 2]))):
                        obs_other = other.observations.get(kf_id)
                        if obs_other is not None:
                            keyframe.landmark_ids[feat_idx] = int(lm.id)
                            map_state.remove_landmark(existing_lid)
                            map_state.add_observation(int(lm.id), kf_id, obs_other)
                            fused += 1
            if fused > 0:
                keyframe.landmark_ids_version = int(getattr(keyframe, "landmark_ids_version", 0)) + 1
                map_state.landmark_ids_version = int(getattr(map_state, "landmark_ids_version", 0)) + 1
            return fused

        grid, inv_cell = _build_spatial_grid(f_pixels, max(rad, 4.0))

        for row_i, lm_pos in enumerate(idx_v):
            lm = unseen_lms[int(lm_pos)]
            u_p, v_p = float(uv[row_i, 0]), float(uv[row_i, 1])
            cx, cy = int(np.floor(u_p * inv_cell)), int(np.floor(v_p * inv_cell))
            cand_list: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy))
                    if bucket:
                        cand_list.extend(bucket)
            if not cand_list:
                continue
            cands = np.asarray(cand_list, dtype=np.intp)
            diff = f_pixels[cands] - np.array([u_p, v_p], dtype=np.float32)
            in_r = (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) <= rad_sq
            if not np.any(in_r):
                continue
            cands = cands[in_r]
            if f_octs is not None:
                oct_ok = np.abs(f_octs[cands] - int(getattr(lm, "octave", 0))) <= 1
                if not np.any(oct_ok):
                    continue
                cands = cands[oct_ok]

            lm_desc = np.asarray(lm.descriptor, dtype=np.float32)
            lm_n2 = float(np.dot(lm_desc, lm_desc))
            d2 = np.maximum(0.0, f_norm2[cands] + lm_n2 - 2.0 * (f_descs[cands] @ lm_desc))
            best_i = int(np.argmin(d2))
            if float(d2[best_i]) > max_d2:
                continue
            feat_idx = int(cands[best_i])

            existing_lid = int(keyframe.landmark_ids[feat_idx])
            if existing_lid < 0:
                ru = None
                if right_u_values is not None and feat_idx < len(right_u_values):
                    rv = float(right_u_values[feat_idx])
                    if np.isfinite(rv):
                        ru = rv
                obs = Observation(
                    f_pixels[feat_idx].copy(),
                    ru,
                    octave=int(f_octs[feat_idx]) if f_octs is not None else 0,
                )
                keyframe.landmark_ids[feat_idx] = int(lm.id)
                map_state.add_observation(int(lm.id), kf_id, obs)
                fused += 1
            elif existing_lid != int(lm.id) and existing_lid in map_state.landmarks:
                other = map_state.landmarks[existing_lid]
                # Fuse only if the duplicate landmark is single-view and geometrically close
                if (len(other.observations) == 1
                        and len(lm.observations) >= 2
                        and float(np.linalg.norm(other.position - lm.position)) < max(0.5, 0.05 * float(pc[row_i, 2]))):
                    obs_other = other.observations.get(kf_id)
                    if obs_other is not None:
                        keyframe.landmark_ids[feat_idx] = int(lm.id)
                        map_state.remove_landmark(existing_lid)
                        map_state.add_observation(int(lm.id), kf_id, obs_other)
                        fused += 1
        if fused > 0:
            keyframe.landmark_ids_version = int(getattr(keyframe, "landmark_ids_version", 0)) + 1
            map_state.landmark_ids_version = int(getattr(map_state, "landmark_ids_version", 0)) + 1
        _ = baseline
        return fused


def triangulate_far_stereo_landmarks(
    prev_kf: Any,
    curr_kf: Any,
    K: np.ndarray,
    baseline: float = 0.0,
    min_parallax_deg: float = 0.35,
    min_depth_m: float = 20.0,
    max_depth_m: float = 400.0,
    max_reproj_px: float = 2.0,
    ratio: float = 0.72,
    max_landmarks: int = 120,
) -> list[TriangulatedLandmark]:
    """Temporally triangulate distant bearing-only landmarks across two keyframes.

    Gives highway/open-road scenes (like KITTI 01) strong rotational anchors at 20..400 m
    where stereo disparity is too small for single-frame triangulation.
    """
    _ = baseline
    if prev_kf is None or curr_kf is None:
        return []
    prev_unassigned = np.flatnonzero(np.asarray(prev_kf.landmark_ids) < 0)
    curr_unassigned = np.flatnonzero(np.asarray(curr_kf.landmark_ids) < 0)
    if len(prev_unassigned) < 8 or len(curr_unassigned) < 8:
        return []

    T_wc1 = np.asarray(prev_kf.pose, dtype=np.float64)
    T_wc2 = np.asarray(curr_kf.pose, dtype=np.float64)
    # Require sufficient camera baseline between keyframes
    cam_dist = float(np.linalg.norm(T_wc2[:3, 3] - T_wc1[:3, 3]))
    if cam_dist < 0.25:
        return []

    K_arr = np.asarray(K, dtype=np.float64)
    T_cw1 = np.linalg.inv(T_wc1)
    T_cw2 = np.linalg.inv(T_wc2)
    P1 = K_arr @ T_cw1[:3, :]
    P2 = K_arr @ T_cw2[:3, :]

    # Match unassigned features using normalized descriptor similarity + local search
    # Restrict matches to plausible far-point image motion (pure rotation prediction + small shift)
    d1 = _normalize_descriptors(prev_kf.descriptors[prev_unassigned])
    d2 = _normalize_descriptors(curr_kf.descriptors[curr_unassigned])
    if len(d1) == 0 or len(d2) == 0:
        return []

    # Predict far-point pixel positions in curr_kf via infinite-depth homography H_inf = K R_21 K^-1
    R_21 = T_cw2[:3, :3] @ T_wc1[:3, :3]
    H_inf = K_arr @ R_21 @ np.linalg.inv(K_arr)
    px1 = np.asarray(prev_kf.pixels[prev_unassigned], dtype=np.float64)
    px2 = np.asarray(curr_kf.pixels[curr_unassigned], dtype=np.float64)

    p1_h = np.column_stack((px1, np.ones(len(px1))))
    pred2_h = p1_h @ H_inf.T
    valid_h = pred2_h[:, 2] > 1e-6
    pred2 = np.full_like(px1, -1e6)
    pred2[valid_h] = pred2_h[valid_h, :2] / pred2_h[valid_h, 2:3]

    # Far points shift very little from H_inf prediction (e.g., < 45 px even at 20 m & 3 m motion)
    search_rad = max(25.0, min(80.0, 20.0 + 10.0 * cam_dist))
    rad_sq = search_rad * search_rad

    pairs: list[tuple[int, int]] = []
    used_curr: set[int] = set()
    cos_min = 0.72
    r2_thresh = float(ratio * ratio)
    max_unit_l2 = float(np.sqrt(max(0.0, 2.0 - 2.0 * cos_min)))

    # Fast path: AVX-512 spatial-grid matcher applies the geometric search window
    # BEFORE descriptor ranking (exact parity with the local window fallback).
    native_used = False
    valid_i1 = np.flatnonzero(valid_h)
    if len(valid_i1):
        try:
            from native_accel import match_projected_landmarks_f32
            native_pairs = match_projected_landmarks_f32(
                pred2[valid_i1],
                d1[valid_i1],
                px2,
                d2,
                radius_px=float(search_rad),
                ratio=float(ratio),
                max_l2_distance=max_unit_l2,
                max_single_l2_distance=max_unit_l2,
            )
            if native_pairs is not None:
                native_used = True
                for row_a, i2_k in native_pairs.tolist():
                    i1_k = int(valid_i1[row_a])
                    if i2_k not in used_curr:
                        used_curr.add(i2_k)
                        pairs.append((i1_k, i2_k))
        except Exception:
            native_used = False

    if not native_used:
        # Fallback: single BLAS SGEMM for all normalized descriptor similarities + float32 distance mask
        valid_i1 = np.flatnonzero(valid_h)
        if len(valid_i1):
            p_sub = pred2[valid_i1].astype(np.float32, copy=False)  # [M, 2]
            px2_f32 = px2.astype(np.float32, copy=False)
            dx = p_sub[:, 0:1] - px2_f32[None, :, 0]
            dy = p_sub[:, 1:2] - px2_f32[None, :, 1]
            dx *= dx
            dy *= dy
            dx += dy
            in_rad = dx <= np.float32(rad_sq)
            del dy, dx
            row_has = in_rad.any(axis=1)
            act = np.flatnonzero(row_has)
            if len(act):
                act_i1 = valid_i1[act]
                d1_sub = np.ascontiguousarray(d1[act_i1], np.float32)
                d2_c = d2 if (d2.dtype == np.float32 and d2.flags.c_contiguous) else np.ascontiguousarray(d2, np.float32)
                sims_all = d1_sub @ d2_c.T
                in_rad_act = in_rad[act]
                sims_all[~in_rad_act] = np.float32(-2.0)
                best_j = np.argmax(sims_all, axis=1)
                rows_a = np.arange(len(act))
                best_s = sims_all[rows_a, best_j].copy()
                sims_all[rows_a, best_j] = np.float32(-2.0)
                second_s = np.max(sims_all, axis=1)
                counts = in_rad_act.sum(axis=1)
                best_s_list = best_s.tolist()
                second_s_list = second_s.tolist()
                best_j_list = best_j.tolist()
                counts_list = counts.tolist()
                act_i1_list = act_i1.tolist()
                for idx_a in range(len(act)):
                    bs = best_s_list[idx_a]
                    if bs < cos_min:
                        continue
                    if counts_list[idx_a] > 1:
                        d1_sq = max(0.0, 2.0 - 2.0 * bs)
                        d2_sq = max(0.0, 2.0 - 2.0 * second_s_list[idx_a])
                        if d2_sq > 1e-12 and d1_sq >= r2_thresh * d2_sq:
                            continue
                    i2 = best_j_list[idx_a]
                    if i2 not in used_curr:
                        used_curr.add(i2)
                        pairs.append((act_i1_list[idx_a], i2))

    if not pairs:
        return []

    p1_arr = np.array([px1[i1] for i1, _ in pairs], dtype=np.float64).T  # [2, M]
    p2_arr = np.array([px2[i2] for _, i2 in pairs], dtype=np.float64).T  # [2, M]
    pts_4d = cv.triangulatePoints(P1, P2, p1_arr, p2_arr)
    w = pts_4d[3]
    good_w = np.isfinite(w) & (np.abs(w) > 1e-9)
    if not np.any(good_w):
        return []

    pts_w = np.full((len(pairs), 3), np.nan, dtype=np.float64)
    pts_w[good_w] = (pts_4d[:3, good_w] / w[good_w]).T

    c1 = T_wc1[:3, 3]
    c2 = T_wc2[:3, 3]
    min_cos_parallax = float(np.cos(np.deg2rad(float(min_parallax_deg))))
    prev_octs = getattr(prev_kf, "octaves", None)
    curr_octs = getattr(curr_kf, "octaves", None)

    # Screen clear depth and parallax failures in a batch. Keep the scalar
    # geometry checks for retained points, including near each cutoff.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        depth1 = pts_w @ T_cw1[2, :3] + T_cw1[2, 3]
        depth2 = pts_w @ T_cw2[2, :3] + T_cw2[2, 3]
        point_magnitude = np.max(np.abs(pts_w), axis=1)
        pad1 = 1e-12 * (1 + point_magnitude * np.abs(T_cw1[2, :3]).sum() + abs(T_cw1[2, 3]))
        pad2 = 1e-12 * (1 + point_magnitude * np.abs(T_cw2[2, :3]).sum() + abs(T_cw2[2, 3]))
        outside = ((depth1 < min_depth_m - pad1) | (depth1 > max_depth_m + pad1)
                   | (depth2 < min_depth_m - pad2) | (depth2 > max_depth_m + pad2))
        considered = np.flatnonzero(np.isfinite(pts_w).all(axis=1) & ~outside)
        rays1, rays2 = pts_w[considered] - c1, pts_w[considered] - c2
        scale = np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1)
        cosine = np.einsum("ij,ij->i", rays1, rays2) / scale
        clearly_low_parallax = np.isfinite(scale) & (scale > 0.0) & (cosine > min_cos_parallax + 1e-12)
        considered = considered[~clearly_low_parallax]

    results: list[TriangulatedLandmark] = []
    for k in considered.tolist():
        i1, i2 = pairs[k]
        X = pts_w[k]
        if not np.isfinite(X).all():
            continue
        pc1 = T_cw1[:3, :3] @ X + T_cw1[:3, 3]
        pc2 = T_cw2[:3, :3] @ X + T_cw2[:3, 3]
        if not (min_depth_m <= pc1[2] <= max_depth_m and min_depth_m <= pc2[2] <= max_depth_m):
            continue

        # Parallax angle between rays from c1 and c2 to X
        r1 = X - c1
        r2 = X - c2
        n1 = float(np.linalg.norm(r1))
        n2 = float(np.linalg.norm(r2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_ang = float(np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0))
        if cos_ang > min_cos_parallax:
            continue
        parallax_deg = float(np.rad2deg(np.arccos(cos_ang)))

        # Reprojection error check in both views
        h1 = K_arr @ pc1
        h2 = K_arr @ pc2
        err1 = float(np.linalg.norm(h1[:2] / h1[2] - px1[i1]))
        err2 = float(np.linalg.norm(h2[:2] / h2[2] - px2[i2]))
        if err1 > max_reproj_px or err2 > max_reproj_px:
            continue

        prev_f = int(prev_unassigned[i1])
        curr_f = int(curr_unassigned[i2])
        oct_val = int(curr_octs[curr_f]) if curr_octs is not None else (
            int(prev_octs[prev_f]) if prev_octs is not None else 0
        )
        results.append(
            TriangulatedLandmark(
                position_w=X.copy(),
                prev_feat_idx=prev_f,
                curr_feat_idx=curr_f,
                descriptor=np.asarray(curr_kf.descriptors[curr_f], dtype=np.float32).copy(),
                octave=oct_val,
                parallax_deg=parallax_deg,
                depth_m=float(pc2[2]),
                is_far=True,
            )
        )
        if len(results) >= max_landmarks:
            break

    return results
