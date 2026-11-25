"""Short-lived, stereo-verified geometry for the optical-flow tracking path."""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from mapping_geometry import coverage, project, refine_stereo_map_pose, right_pixel


MIN_PERSISTENT_TRACKS = 50
MIN_REFRESH_SUPPORT = 55


@dataclass(frozen=True)
class LazyStereoGeometry:
    frame: int
    anchor_id: int
    expires: int
    anchor_points: np.ndarray
    pixels: np.ndarray

    def active(self, frame, anchor_id, *, require_previous=True):
        return (self.anchor_id == anchor_id and self.frame < frame <= self.expires
                and (not require_previous or self.frame == frame - 1)
                and len(self.pixels) >= 16)

    def world_points(self, anchor_pose):
        return self.anchor_points @ anchor_pose[:3, :3].T + anchor_pose[:3, 3]

    def advance(self, frame, rows, pixels):
        if frame >= self.expires:
            return None
        return LazyStereoGeometry(frame, self.anchor_id, self.expires,
                                  self.anchor_points[rows], pixels.copy())


def seed_lazy_stereo_geometry(frame, anchor, pose, gray, tracked_pixels, size, measure_stereo):
    """Seed corners away from persistent tracks and retain only verified metric depth."""
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 32:
        return None
    width, height = size
    reduced = cv.resize(gray, ((width + 1) // 2, (height + 1) // 2),
                        interpolation=cv.INTER_AREA)
    scale = np.array([width / reduced.shape[1], height / reduced.shape[0]], dtype=np.float32)
    mask = np.full(reduced.shape, 255, dtype=np.uint8)
    mask[:5] = mask[-5:] = 0
    mask[:, :9] = mask[:, -9:] = 0
    reduced_tracks = np.rint((tracked_pixels + .5) / scale - .5).astype(int)
    for x, y in reduced_tracks.tolist():
        cv.circle(mask, (x, y), 5, 0, -1)
    corners = cv.goodFeaturesToTrack(reduced, 144, .012, 6., mask=mask, blockSize=5)
    if corners is None or len(corners) < 42:
        return None
    pixels = (corners[:, 0] + .5) * scale - .5
    points, right = measure_stereo(pixels)
    rows = np.flatnonzero(np.isfinite(right) & np.isfinite(points).all(axis=1)
                          & (points[:, 2] >= 4.) & (points[:, 2] <= 42.))[:90]
    if len(rows) < 42 or coverage(pixels[rows], size) < 6:
        return None
    relative = np.linalg.inv(anchor.pose) @ pose
    anchor_points = points[rows] @ relative[:3, :3].T + relative[:3, 3]
    return LazyStereoGeometry(frame, anchor.id, frame + 3, anchor_points, pixels[rows])


def refine_with_lazy_stereo_geometry(
    solution, positions, pixels, measured_right, primary_count, matrix, stereo,
    size, min_inliers, weights,
):
    """Refine on persistent landmarks and use supplemental tracks to verify support."""
    pose, rows, median = solution
    persistent = rows[rows < primary_count]
    if len(persistent) < MIN_PERSISTENT_TRACKS or coverage(pixels[persistent], size) < 6:
        return None
    refined, _ = refine_stereo_map_pose(
        (pose, persistent, median), positions, pixels, measured_right, matrix,
        stereo.baseline, size, min_inliers, stereo.disparity_offset,
        use_analytic=True, weights_uv_uR=weights,
    )
    if refined is None:
        return None
    pose, persistent, median = refined
    if (len(persistent) < MIN_PERSISTENT_TRACKS or np.isfinite(measured_right[persistent]).sum() < 29
            or coverage(pixels[persistent], size) < 6 or median > .90):
        return None
    supplemental = rows[rows >= primary_count]
    uv, z = project(positions[supplemental], pose, matrix)
    expected_right = right_pixel(uv[:, 0], z, matrix[0, 0], stereo.baseline, stereo.disparity_offset)
    agree = ((z > 0) & (np.linalg.norm(pixels[supplemental] - uv, axis=1) <= 1.3)
             & (np.abs(measured_right[supplemental] - expected_right) <= 1.2))
    verified = supplemental[agree]
    if len(persistent) + len(verified) < 80 or len(verified) < 14:
        return None
    return pose, np.concatenate((persistent, verified)), median
