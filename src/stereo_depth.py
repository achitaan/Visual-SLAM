"""Independent sparse correspondence checks for rectified stereo images."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class StereoSearchConfig:
    # Fixed sensor-mode verification gates; these do not change pose acceptance.
    patch_size: int = 11
    max_disparity: int = 96
    minimum_correlation: float = 0.85
    uniqueness_ratio: float = 0.7
    round_trip_px: float = 0.5
    minimum_texture_std: float = 2.0

    def __post_init__(self):
        if (not isinstance(self.patch_size, (int, np.integer)) or not isinstance(self.max_disparity, (int, np.integer))
                or self.patch_size < 3 or self.patch_size % 2 != 1 or self.max_disparity < 2):
            raise ValueError('Invalid stereo search dimensions')
        if not (0 < self.minimum_correlation < 1 and 0 < self.uniqueness_ratio < 1):
            raise ValueError('Invalid photometric acceptance gates')
        if not np.isfinite([self.round_trip_px, self.minimum_texture_std]).all() or min(self.round_trip_px, self.minimum_texture_std) <= 0:
            raise ValueError('Invalid stereo verification tolerances')


def _search(source, target, pixel, direction, config, disparity_bounds):
    radius = config.patch_size // 2
    height, width = source.shape
    x, y = map(float, pixel)
    if not (radius <= x <= width-1-radius and radius <= y <= height-1-radius):
        return None, 'patch_bounds'
    patch = cv.getRectSubPix(source, (config.patch_size, config.patch_size), (x, y))
    if float(np.std(patch)) < config.minimum_texture_std:
        return None, 'low_texture'
    strip_width = config.patch_size + config.max_disparity
    strip_center = x + direction * config.max_disparity / 2
    strip = cv.getRectSubPix(target, (strip_width, config.patch_size), (strip_center, y))
    scores = np.clip(cv.matchTemplate(strip, patch, cv.TM_CCOEFF_NORMED).ravel().astype(float), -1, 1)
    positions = strip_center - config.max_disparity/2 + np.arange(len(scores))
    disparities = direction * (positions-x)
    valid = ((positions >= radius) & (positions <= width-1-radius)
             & (disparities > disparity_bounds[0]) & (disparities < disparity_bounds[1])
             & np.isfinite(scores))
    if not valid.any():
        return None, 'search_bounds'
    masked = np.where(valid, scores, -np.inf)
    best = int(np.argmax(masked))
    if scores[best] < config.minimum_correlation:
        return None, 'correlation'
    # Adjacent samples represent the same peak; all other epipolar candidates
    # remain competitors. Equal perfect correlations must still be rejected.
    competitors = valid & (np.abs(np.arange(len(scores))-best) > 1)
    if not competitors.any():
        return None, 'insufficient_alternatives'
    second = float(np.max(scores[competitors]))
    tolerance = 8*np.finfo(np.float32).eps
    if scores[best]-second <= tolerance:
        return None, 'near_tie'
    if 1-scores[best] >= config.uniqueness_ratio * (1-second):
        return None, 'ambiguous'
    if not (0 < best < len(scores)-1 and valid[best-1] and valid[best+1]):
        return None, 'unrefinable_peak'
    curvature = scores[best-1]-2*scores[best]+scores[best+1]
    if curvature >= -tolerance:
        return None, 'flat_peak'
    offset = float(0.5*(scores[best-1]-scores[best+1])/curvature)
    if abs(offset) > 0.5:
        return None, 'unstable_peak'
    result = float(positions[best]+offset)
    if not (radius <= result <= width-1-radius and disparity_bounds[0] < direction*(result-x) < disparity_bounds[1]):
        return None, 'refined_bounds'
    actual = cv.getRectSubPix(target, (config.patch_size, config.patch_size), (result, y))
    if float(np.std(actual)) < config.minimum_texture_std:
        return None, 'low_target_texture'
    correlation = float(cv.matchTemplate(actual, patch, cv.TM_CCOEFF_NORMED)[0, 0])
    if not np.isfinite(correlation) or correlation < config.minimum_correlation:
        return None, 'refined_correlation'
    return result, None


_ORIGINAL_SEARCH = _search
_ORIGINAL_MATCH_TEMPLATE = cv.matchTemplate


def _sample_rect_batch(image, cx, cy, width_out, height_out):
    """Sample N subpixel rectangles using a single batched cv.remap call."""
    n = len(cx)
    x0 = (cx - (width_out - 1) * 0.5).astype(np.float32)
    y0 = (cy - (height_out - 1) * 0.5).astype(np.float32)
    dx = np.arange(width_out, dtype=np.float32)
    dy = np.arange(height_out, dtype=np.float32)
    mx = np.broadcast_to(x0[:, None, None] + dx[None, None, :], (n, height_out, width_out))
    my = np.broadcast_to(y0[:, None, None] + dy[None, :, None], (n, height_out, width_out))
    remapped = cv.remap(
        image,
        np.ascontiguousarray(mx.reshape(n * height_out, width_out)),
        np.ascontiguousarray(my.reshape(n * height_out, width_out)),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    return remapped.reshape(n, height_out, width_out)


def _batch_search(source, target, pts, direction, config, disparity_bounds, batch_size=128):
    """Batched equivalent of _search preserving all photometric/geometric gates."""
    n_total = len(pts)
    patch_size = int(config.patch_size)
    max_disp = int(config.max_disparity)
    radius = patch_size // 2
    height, width = source.shape
    strip_w = patch_size + max_disp
    n_scores = max_disp + 1
    patch_area = float(patch_size * patch_size)
    tolerance = 8.0 * float(np.finfo(np.float32).eps)
    flat_var_floor = 1e-6 * patch_area

    out_x = np.full(n_total, np.nan, dtype=np.float64)
    reasons = [None] * n_total
    offsets = np.arange(n_scores, dtype=np.float64)

    for start in range(0, n_total, batch_size):
        stop = min(n_total, start + batch_size)
        sub = pts[start:stop]
        x = sub[:, 0]
        y = sub[:, 1]
        in_bounds = (
            (x >= radius)
            & (x <= width - 1 - radius)
            & (y >= radius)
            & (y <= height - 1 - radius)
        )
        if not in_bounds.all():
            for li in np.flatnonzero(~in_bounds):
                reasons[start + int(li)] = 'patch_bounds'
        idx_b = np.flatnonzero(in_bounds)
        if not len(idx_b):
            continue

        xb = x[idx_b]
        yb = y[idx_b]
        patches = _sample_rect_batch(source, xb, yb, patch_size, patch_size)
        p64 = patches.astype(np.float64)
        p_mean = p64.mean(axis=(1, 2))
        p_zm = p64 - p_mean[:, None, None]
        p_ss = np.einsum('bij,bij->b', p_zm, p_zm)
        p_std = np.sqrt(p_ss / patch_area)
        ok_tex = p_std >= config.minimum_texture_std
        if not ok_tex.all():
            for li in np.flatnonzero(~ok_tex):
                reasons[start + int(idx_b[li])] = 'low_texture'
        keep = np.flatnonzero(ok_tex)
        if not len(keep):
            continue

        idx_b = idx_b[keep]
        xb = xb[keep]
        yb = yb[keep]
        p_zm = p_zm[keep]
        p_ss = p_ss[keep]
        batch_n = len(idx_b)

        strip_center = xb + direction * max_disp * 0.5
        strips = _sample_rect_batch(target, strip_center, yb, strip_w, patch_size)
        s64 = strips.astype(np.float64)

        col_s = s64.sum(axis=1)
        col_sq = np.einsum('bij,bij->bj', s64, s64)
        cs = np.empty((batch_n, strip_w + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(col_s, axis=1, out=cs[:, 1:])
        csq = np.empty((batch_n, strip_w + 1), dtype=np.float64)
        csq[:, 0] = 0.0
        np.cumsum(col_sq, axis=1, out=csq[:, 1:])
        win_s = cs[:, patch_size:] - cs[:, :-patch_size]
        win_sq = csq[:, patch_size:] - csq[:, :-patch_size]
        win_ss = np.maximum(win_sq - (win_s * win_s) / patch_area, 0.0)

        p_zm_t = np.ascontiguousarray(p_zm.transpose(2, 0, 1))  # (P, B, P)
        num = np.zeros((batch_n, n_scores), dtype=np.float64)
        for dx_idx in range(patch_size):
            num += (p_zm_t[dx_idx][:, None, :] @ s64[:, :, dx_idx:dx_idx + n_scores])[:, 0, :]

        denom = np.sqrt(p_ss[:, None] * win_ss)
        valid_den = (win_ss > flat_var_floor) & (denom > 0.0)
        scores = np.where(valid_den, num / np.where(valid_den, denom, 1.0), 0.0)
        scores = np.clip(scores, -1.0, 1.0)

        positions = (strip_center - max_disp * 0.5)[:, None] + offsets[None, :]
        disparities = direction * (positions - xb[:, None])
        valid = (
            (positions >= radius)
            & (positions <= width - 1 - radius)
            & (disparities > disparity_bounds[0])
            & (disparities < disparity_bounds[1])
            & np.isfinite(scores)
        )

        any_valid = valid.any(axis=1)
        masked = np.where(valid, scores, -np.inf)
        best_idx = np.argmax(masked, axis=1)
        rows_b = np.arange(batch_n)
        s_best = scores[rows_b, best_idx]
        comp = valid & (np.abs(offsets[None, :] - best_idx[:, None]) > 1.0)
        any_comp = comp.any(axis=1)
        second_scores = np.max(np.where(comp, scores, -np.inf), axis=1)

        cand_refined = []
        for bi in range(batch_n):
            gi = start + int(idx_b[bi])
            if not any_valid[bi]:
                reasons[gi] = 'search_bounds'
                continue
            sb = float(s_best[bi])
            if sb < config.minimum_correlation:
                reasons[gi] = 'correlation'
                continue
            if not any_comp[bi]:
                reasons[gi] = 'insufficient_alternatives'
                continue
            sec = float(second_scores[bi])
            if sb - sec <= tolerance:
                reasons[gi] = 'near_tie'
                continue
            if 1.0 - sb >= config.uniqueness_ratio * (1.0 - sec):
                reasons[gi] = 'ambiguous'
                continue
            best = int(best_idx[bi])
            if not (0 < best < n_scores - 1 and valid[bi, best - 1] and valid[bi, best + 1]):
                reasons[gi] = 'unrefinable_peak'
                continue
            s_prev = float(scores[bi, best - 1])
            s_next = float(scores[bi, best + 1])
            curvature = s_prev - 2.0 * sb + s_next
            if curvature >= -tolerance:
                reasons[gi] = 'flat_peak'
                continue
            offset = 0.5 * (s_prev - s_next) / curvature
            if abs(offset) > 0.5:
                reasons[gi] = 'unstable_peak'
                continue
            res_x = float(positions[bi, best] + offset)
            if not (
                radius <= res_x <= width - 1 - radius
                and disparity_bounds[0] < direction * (res_x - xb[bi]) < disparity_bounds[1]
            ):
                reasons[gi] = 'refined_bounds'
                continue
            cand_refined.append((bi, gi, res_x))

        if cand_refined:
            r_bi = np.array([c[0] for c in cand_refined], dtype=np.int32)
            r_rx = np.array([c[2] for c in cand_refined], dtype=np.float64)
            r_ry = yb[r_bi]
            actuals = _sample_rect_batch(target, r_rx, r_ry, patch_size, patch_size).astype(np.float64)
            a_zm = actuals - actuals.mean(axis=(1, 2), keepdims=True)
            a_ss = np.einsum('bij,bij->b', a_zm, a_zm)
            a_std = np.sqrt(a_ss / patch_area)
            ref_zm = p_zm[r_bi]
            ref_ss = p_ss[r_bi]
            corr_num = np.einsum('bij,bij->b', a_zm, ref_zm)
            corr_den = np.sqrt(a_ss * ref_ss)
            valid_c = (a_ss > flat_var_floor) & (corr_den > 0.0)
            corrs = np.where(valid_c, corr_num / np.where(valid_c, corr_den, 1.0), 0.0)
            for k_idx, (_, gi, res_x) in enumerate(cand_refined):
                if a_std[k_idx] < config.minimum_texture_std:
                    reasons[gi] = 'low_target_texture'
                elif not np.isfinite(corrs[k_idx]) or corrs[k_idx] < config.minimum_correlation:
                    reasons[gi] = 'refined_correlation'
                else:
                    out_x[gi] = res_x

    return out_x, reasons


def verify_stereo_depth_candidates(left, right, pixels, config=None, disparity_bounds=None):
    """Return independently verified right x positions and rejection counts.

    Receives images and image coordinates only. Disparity proposals, poses, map
    geometry and reference data cannot influence the full-range searches.
    Image intensity units are grayscale 0..255, as produced by the sequence reader.
    """
    config = config or StereoSearchConfig()
    if disparity_bounds is None:
        disparity_bounds = (0., float(config.max_disparity))
    if not (np.isfinite(disparity_bounds).all() and 0 <= disparity_bounds[0] < disparity_bounds[1] <= config.max_disparity):
        raise ValueError('Invalid calibrated disparity domain')
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    result = np.full(len(pixels), np.nan)
    counters = {'candidates': len(pixels), 'verified': 0}
    if left is None or right is None:
        counters['missing_images'] = len(pixels)
        return result, counters
    if np.ndim(left) != 2 or np.shape(left) != np.shape(right):
        raise ValueError('Stereo verification needs matching grayscale images')
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Stereo verification images must be finite')
    if not len(pixels):
        return result, counters

    if _search is _ORIGINAL_SEARCH and cv.matchTemplate is _ORIGINAL_MATCH_TEMPLATE:
        finite_mask = np.isfinite(pixels).all(axis=1)
        if not finite_mask.all():
            counters['nonfinite_pixel'] = int((~finite_mask).sum())
        finite_idx = np.flatnonzero(finite_mask)
        if not len(finite_idx):
            return result, counters
        xr, fwd_reasons = _batch_search(
            left, right, pixels[finite_idx], -1, config, disparity_bounds
        )
        ok_fwd = []
        for k, reason in enumerate(fwd_reasons):
            if reason is None:
                ok_fwd.append(k)
            else:
                counters[reason] = counters.get(reason, 0) + 1
        if ok_fwd:
            ok_arr = np.asarray(ok_fwd, dtype=np.int32)
            rev_pts = np.column_stack((xr[ok_arr], pixels[finite_idx[ok_arr], 1]))
            xl, rev_reasons = _batch_search(
                right, left, rev_pts, 1, config, disparity_bounds
            )
            for m, reason in enumerate(rev_reasons):
                gi = int(finite_idx[ok_arr[m]])
                if reason is not None:
                    counters[reason] = counters.get(reason, 0) + 1
                elif abs(xl[m] - pixels[gi, 0]) > config.round_trip_px:
                    counters['round_trip'] = counters.get('round_trip', 0) + 1
                else:
                    result[gi] = xr[ok_arr[m]]
                    counters['verified'] += 1
        return result, counters

    for index, pixel in enumerate(pixels):
        if not np.isfinite(pixel).all():
            reason = 'nonfinite_pixel'
        else:
            xr, reason = _search(left, right, pixel, -1, config, disparity_bounds)
            if reason is None:
                reverse, reason = _search(right, left, [xr, pixel[1]], 1, config, disparity_bounds)
                if reason is None and abs(reverse-pixel[0]) > config.round_trip_px:
                    reason = 'round_trip'
                if reason is None:
                    result[index] = xr
                    counters['verified'] += 1
                    continue
        counters[reason] = counters.get(reason, 0)+1
    return result, counters
