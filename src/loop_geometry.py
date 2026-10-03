"""Independent metric loop measurements from paired stereo feature geometry."""
from dataclasses import dataclass
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class StereoLoopFrame:
    pixels: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    image_size: tuple


def _collapse_physical_pairs(pairs, first_pixels, second_pixels):
    """Keep one unambiguous correspondence per exact float32 pixel edge."""
    first_pixels = np.asarray(first_pixels, np.float32)
    second_pixels = np.asarray(second_pixels, np.float32)
    if (first_pixels.ndim != 2 or first_pixels.shape[1:] != (2,)
            or second_pixels.ndim != 2 or second_pixels.shape[1:] != (2,)):
        return np.empty((0, 2), int)
    edge_rows, first_targets, second_sources = {}, {}, {}
    for first, second in pairs:
        first_pixel, second_pixel = first_pixels[first], second_pixels[second]
        if not np.isfinite(first_pixel).all() or not np.isfinite(second_pixel).all():
            continue
        first_key = tuple(first_pixel.tolist())
        second_key = tuple(second_pixel.tolist())
        edge = (first_key, second_key)
        edge_rows.setdefault(edge, []).append((int(first), int(second)))
        first_targets.setdefault(first_key, set()).add(second_key)
        second_sources.setdefault(second_key, set()).add(first_key)
    ambiguous_first = {key for key, targets in first_targets.items() if len(targets) > 1}
    ambiguous_second = {key for key, sources in second_sources.items() if len(sources) > 1}
    selected = [min(rows) for (first_key, second_key), rows in edge_rows.items()
                if first_key not in ambiguous_first and second_key not in ambiguous_second]
    return np.asarray(sorted(selected), int).reshape(-1, 2)


def extract_loop_frame(vo, index):
    left, right = vo.Images_1[index], vo.Images_2[index]
    # Bound retained descriptors for batch retrieval on memory-limited hosts.
    detector = cv.ORB_create(nfeatures=1500) if hasattr(vo, 'orb') else cv.SIFT_create(nfeatures=1500)
    keypoints, descriptors = detector.detectAndCompute(left, None)
    disparity = vo.stereo.compute(left, right).astype(np.float32) / 16
    dense = cv.reprojectImageTo3D(disparity, vo.Q)
    pixels, points, selected = [], [], []
    for i, keypoint in enumerate(keypoints):
        u, v = np.rint(keypoint.pt).astype(int)
        if not (0 <= u < left.shape[1] and 0 <= v < left.shape[0]):
            continue
        p = dense[v, u]
        if 0 < disparity[v, u] < 96 and np.isfinite(p).all() and .1 < p[2] < 100:
            pixels.append(keypoint.pt); points.append(p); selected.append(i)
    desc = descriptors[selected] if selected else np.empty((0, 128), np.float32)
    return StereoLoopFrame(np.asarray(pixels, np.float32).reshape(-1, 2), np.asarray(points, np.float32).reshape(-1, 3), desc, (left.shape[1], left.shape[0]))


def verify_loop(first, second, matrix, min_inliers=30, pairs=None):
    """Return Z_ij = inverse(T_wi) T_wj, without trajectory or GT evidence.

    Optional ``pairs`` are precomputed mutual descriptor rows from the complete
    appearance arrays. They still pass through the same forward and reverse
    geometric verification below.
    """
    if len(first.descriptors) < min_inliers or len(second.descriptors) < min_inliers:
        return None
    if pairs is None:
        matcher = cv.BFMatcher(cv.NORM_HAMMING) if first.descriptors.dtype == np.uint8 else cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        def matches(a, b):
            return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance}
        forward = matches(first.descriptors, second.descriptors)
        backward = matches(second.descriptors, first.descriptors)
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    else:
        pairs = np.asarray(pairs)
        if (pairs.ndim != 2 or pairs.shape[1:] != (2,)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.issubdtype(pairs.dtype, np.bool_)
                or np.any(pairs < 0)
                or np.any(pairs[:, 0] >= len(first.descriptors))
                or np.any(pairs[:, 1] >= len(second.descriptors))
                or len(np.unique(pairs[:, 0])) != len(pairs)
                or len(np.unique(pairs[:, 1])) != len(pairs)):
            return None
        pairs = np.unique(pairs.astype(int, copy=False), axis=0)
        pairs = _collapse_physical_pairs(pairs, first.pixels, second.pixels)
    if len(pairs) < min_inliers:
        return None
    a, b = np.array(pairs).T
    def solve(points, pixels, image_size):
        # PnP needs source 3D and target 2D, not target depth. Missing depth
        # in the other direction must not discard an otherwise valid image match.
        available = np.flatnonzero(np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1))
        if len(available) < min_inliers:
            return None
        points, pixels = points[available], pixels[available]
        ok, rv, tv, inliers = cv.solvePnPRansac(points, pixels, matrix, None, iterationsCount=500,
                                              reprojectionError=2., confidence=.999, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < min_inliers or len(inliers) / len(points) < .35:
            return None
        ids = inliers.ravel()
        rv, tv = cv.solvePnPRefineLM(points[ids], pixels[ids], matrix, None, rv, tv)
        transform = np.eye(4); transform[:3, :3] = cv.Rodrigues(rv)[0]; transform[:3, 3] = tv.ravel()
        depths = (transform[:3, :3] @ points[ids].T + tv).T[:, 2]
        predicted = cv.projectPoints(points[ids], rv, tv, matrix, None)[0].reshape(-1, 2)
        residual = np.linalg.norm(predicted - pixels[ids], axis=1)
        width, height = image_size
        cells = np.floor(pixels[ids] / [width / 3, height / 3]).clip(0, 2).astype(int)
        if np.any(depths <= 0) or np.median(residual) > 1.5 or len(np.unique(cells, axis=0)) < 3:
            return None
        return transform, available[ids], float(np.median(residual)), len(available)
    result = solve(first.points[a], second.pixels[b], second.image_size)
    reverse = solve(second.points[b], first.pixels[a], first.image_size)
    if result is None or reverse is None:
        return None
    transform, inliers, error, support = result
    consistency = reverse[0] @ transform
    rotation_error = np.degrees(Rotation.from_matrix(consistency[:3, :3]).magnitude())
    translation_error = np.linalg.norm(consistency[:3, 3])
    if translation_error > .5 or rotation_error > 1.5:
        return None
    return {'measurement': np.linalg.inv(transform), 'matches': support, 'inliers': len(inliers),
            'descriptor_matches': len(pairs), 'reverse_depth_support': reverse[3],
            'median_reprojection_px': error, 'reverse_translation_error_m': float(translation_error),
            'reverse_rotation_error_deg': float(rotation_error)}


def propagate_corrections(raw, frame_indices, optimized):
    """Interpolate rigid world-frame corrections, preserving exact graph anchors."""
    if len(frame_indices) != len(optimized) or frame_indices[0] != 0 or frame_indices[-1] != len(raw) - 1:
        raise ValueError('Graph anchors must span the entire trajectory')
    corrections = [pose @ np.linalg.inv(raw[index]) for index, pose in zip(frame_indices, optimized)]
    result = [None] * len(raw)
    for k in range(len(frame_indices) - 1):
        start, end = frame_indices[k:k + 2]
        if end <= start:
            raise ValueError('Graph frame indices must increase')
        rotation = Slerp([0, 1], Rotation.from_matrix(np.array([corrections[k][:3, :3], corrections[k + 1][:3, :3]])))
        for index in range(start, end + 1):
            fraction = (index - start) / (end - start)
            correction = np.eye(4)
            correction[:3, :3] = rotation(fraction).as_matrix()
            correction[:3, 3] = (1 - fraction) * corrections[k][:3, 3] + fraction * corrections[k + 1][:3, 3]
            result[index] = correction @ raw[index]
    return result
