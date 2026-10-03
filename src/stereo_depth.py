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
