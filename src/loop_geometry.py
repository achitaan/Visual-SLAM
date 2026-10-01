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


def verify_loop(first, second, matrix, min_inliers=30):
    """Return Z_ij = inverse(T_wi) T_wj, or None; never uses a trajectory or GT."""
    if len(first.descriptors) < min_inliers or len(second.descriptors) < min_inliers:
        return None
    matcher = cv.BFMatcher(cv.NORM_HAMMING) if first.descriptors.dtype == np.uint8 else cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    def matches(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance}
    forward = matches(first.descriptors, second.descriptors)
    backward = matches(second.descriptors, first.descriptors)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(pairs) < min_inliers:
        return None
    a, b = np.array(pairs).T
    def solve(points, pixels):
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
        width, height = second.image_size
        cells = np.floor(pixels[ids] / [width / 3, height / 3]).clip(0, 2).astype(int)
        if np.any(depths <= 0) or np.median(residual) > 1.5 or len(np.unique(cells, axis=0)) < 3:
            return None
        return transform, ids, float(np.median(residual))
    result = solve(first.points[a], second.pixels[b])
    reverse = solve(second.points[b], first.pixels[a])
    if result is None or reverse is None:
        return None
    transform, inliers, error = result
    consistency = reverse[0] @ transform
    rotation_error = np.degrees(Rotation.from_matrix(consistency[:3, :3]).magnitude())
    translation_error = np.linalg.norm(consistency[:3, 3])
    if translation_error > .5 or rotation_error > 1.5:
        return None
    return {'measurement': np.linalg.inv(transform), 'matches': len(pairs), 'inliers': len(inliers),
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
