"""Image-only initialization, projection, triangulation and geometric pose checks."""

import cv2 as cv
import numpy as np


def project(points, pose, matrix):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ matrix.T
    return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-9), camera[:, 2]


def match_descriptors(first, second, ratio=0.7):
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return np.empty((0, 2), int)
    matcher = cv.BFMatcher(cv.NORM_HAMMING if first.dtype == np.uint8 else cv.NORM_L2)

    def matches(a, b):
        return {
            p[0].queryIdx: p[0].trainIdx
            for p in matcher.knnMatch(a, b, k=2)
            if len(p) == 2 and p[0].distance < ratio * p[1].distance
        }

    forward, backward = matches(first, second), matches(second, first)
    return np.array(
        [(i, j) for i, j in forward.items() if backward.get(j) == i], int
    ).reshape(-1, 2)


def triangulate(
    first_pixels,
    second_pixels,
    first_pose,
    second_pose,
    matrix,
    min_angle=1.0,
    max_error=2.0,
):
    first_pixels = np.asarray(first_pixels, np.float64)
    second_pixels = np.asarray(second_pixels, np.float64)
    n = len(first_pixels)
    if not n:
        return np.empty((0, 3)), np.empty(0, int)
    a, b = np.linalg.inv(first_pose)[:3], np.linalg.inv(second_pose)[:3]
    homogeneous = cv.triangulatePoints(
        matrix @ a, matrix @ b, first_pixels.T, second_pixels.T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        points = (homogeneous[:3] / homogeneous[3]).T
    pred1, z1 = project(points, first_pose, matrix)
    pred2, z2 = project(points, second_pose, matrix)
    ray1, ray2 = points - first_pose[:3, 3], points - second_pose[:3, 3]
    angles = np.degrees(
        np.arccos(
            np.clip(
                np.sum(ray1 * ray2, axis=1)
                / np.maximum(
                    np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1), 1e-12
                ),
                -1,
                1,
            )
        )
    )
    valid = (
        np.isfinite(points).all(axis=1)
        & (z1 > 0)
        & (z2 > 0)
        & (angles >= min_angle)
        & (np.linalg.norm(pred1 - first_pixels, axis=1) <= max_error)
        & (np.linalg.norm(pred2 - second_pixels, axis=1) <= max_error)
    )
    ids = np.flatnonzero(valid)
    return points[ids], ids


def coverage(pixels, size):
    return (
        len(np.unique(np.clip((pixels / np.array(size) * 3).astype(int), 0, 2), axis=0))
        if len(pixels)
        else 0
    )


def initialize_monocular(first, second, matrix, size):
    if len(first) < 50:
        return None
    essential, mask = cv.findEssentialMat(
        first, second, matrix, method=cv.RANSAC, prob=0.999, threshold=1.0
    )
    _, hmask = cv.findHomography(first, second, cv.RANSAC, 2.0)
    if essential is None or essential.shape != (3, 3) or mask is None:
        return None
    essential_support = int(np.count_nonzero(mask))
    # A dominant planar/rotation explanation cannot establish reliable depth.
    if hmask is not None and np.count_nonzero(hmask) >= 0.9 * essential_support:
        return None
    count, r, t, mask = cv.recoverPose(essential, first, second, matrix, mask=mask)
    if count < 50:
        return None
    pose = np.eye(4)
    pose[:3, :3] = r.T
    pose[:3, 3] = (-r.T @ t).ravel()
    points, ids = triangulate(first, second, np.eye(4), pose, matrix, min_angle=1.5)
    ids = ids[mask.ravel()[ids] != 0]
    points, _ = triangulate(
        first[ids], second[ids], np.eye(4), pose, matrix, min_angle=1.5
    )
    # Enough well-conditioned depth must explain the original matched support,
    # rather than accepting a tiny near-point subset of predominantly weak geometry.
    if (
        len(points) != len(ids)
        or len(ids) < max(50, int(0.35 * len(first)))
        or coverage(second[ids], size) < 4
    ):
        return None
    return pose, points, ids


def estimate_pose(
    points, pixels, matrix, size, min_inliers=15, initial_pose=None, diagnostics=None
):
    """Keep a geometrically validated seeded pose; retry independently on failure.

    A motion prior cannot suppress otherwise valid correspondence geometry.
    Both hypotheses use the same support, cheirality and reprojection checks.
    Selecting independent PnP unconditionally introduced monocular scale drift
    in complete sequence tests, so valid seeded estimates are preserved.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    seeded_diagnostics = {}
    if initial_pose is not None:
        seeded = None
        if np.shape(initial_pose) != (4, 4) or not np.isfinite(initial_pose).all():
            seeded_diagnostics["pose_rejection_reason"] = "invalid_motion_prior"
        else:
            try:
                seeded = _estimate_pose_hypothesis(
                    points,
                    pixels,
                    matrix,
                    size,
                    min_inliers,
                    initial_pose,
                    seeded_diagnostics,
                )
            except (cv.error, np.linalg.LinAlgError):
                seeded_diagnostics["pose_rejection_reason"] = "seeded_solver_failed"
        if seeded is not None:
            diagnostics.update(seeded_diagnostics, pnp_method="iterative_ransac")
            diagnostics.pop("pose_rejection_reason", None)
            return seeded
    independent_diagnostics = {}
    try:
        independent = _estimate_pose_hypothesis(
            points,
            pixels,
            matrix,
            size,
            min_inliers,
            diagnostics=independent_diagnostics,
        )
    except cv.error:
        independent = None
        independent_diagnostics["pose_rejection_reason"] = "independent_solver_failed"
    diagnostics.update(independent_diagnostics, pnp_method="epnp_ransac")
    if initial_pose is not None:
        diagnostics["seeded_rejection_reason"] = seeded_diagnostics.get(
            "pose_rejection_reason"
        )
    if independent is not None:
        diagnostics.pop("pose_rejection_reason", None)
    return independent


def _estimate_pose_hypothesis(
    points, pixels, matrix, size, min_inliers=15, initial_pose=None, diagnostics=None
):
    diagnostics = diagnostics if diagnostics is not None else {}
    if len(points) < min_inliers:
        diagnostics["pose_rejection_reason"] = "insufficient_correspondences"
        return None
    arguments = {}
    if initial_pose is not None:
        inverse = np.linalg.inv(initial_pose)
        arguments = {
            "rvec": cv.Rodrigues(inverse[:3, :3])[0],
            "tvec": inverse[:3, 3].reshape(3, 1),
            "useExtrinsicGuess": True,
        }
    ok, rv, tv, inliers = cv.solvePnPRansac(
        np.asarray(points, np.float64),
        np.asarray(pixels, np.float64),
        matrix,
        None,
        iterationsCount=200,
        reprojectionError=2.0,
        confidence=0.995,
        flags=cv.SOLVEPNP_ITERATIVE if initial_pose is not None else cv.SOLVEPNP_EPNP,
        **arguments
    )
    if (
        not ok
        or inliers is None
        or len(inliers) < min_inliers
        or len(inliers) / len(points) < 0.25
    ):
        diagnostics.update(
            candidate_inliers=0 if inliers is None else len(inliers),
            pose_rejection_reason=(
                "ransac_failed" if not ok else "insufficient_inlier_support"
            ),
        )
        return None
    ids = inliers.ravel()
    rv, tv = cv.solvePnPRefineLM(
        points[ids].astype(float), pixels[ids].astype(float), matrix, None, rv, tv
    )
    pose = np.eye(4)
    pose[:3, :3] = cv.Rodrigues(rv)[0].T
    pose[:3, 3] = (-pose[:3, :3] @ tv).ravel()
    predicted, depth = project(points[ids], pose, matrix)
    error = np.linalg.norm(predicted - pixels[ids], axis=1)
    diagnostics.update(
        candidate_inliers=len(ids),
        candidate_inlier_ratio=len(ids) / len(points),
        candidate_median_reprojection_px=float(np.median(error)),
        candidate_feature_cells=coverage(pixels[ids], size),
    )
    if (
        not np.isfinite(pose).all()
        or np.any(depth <= 0)
        or np.median(error) > 1.5
        or coverage(pixels[ids], size) < 3
    ):
        diagnostics["pose_rejection_reason"] = (
            "invalid_pose"
            if not np.isfinite(pose).all()
            else (
                "negative_depth"
                if np.any(depth <= 0)
                else (
                    "reprojection_error"
                    if np.median(error) > 1.5
                    else "insufficient_spatial_support"
                )
            )
        )
        return None
    return pose, ids, float(np.median(error))


def estimate_stereo_reference(
    source, target, matrix, min_inliers=15, initial_pose=None
):
    """Frame tracking with the map tracker's PnP checks, never a loop constraint.

    Source stereo supplies metric 3D; target image observations suffice. When
    target stereo also yields an accepted reverse pose, contradictory estimates
    are rejected. Loop verification has its own stricter mandatory reverse checks.
    """
    pairs = match_descriptors(source.descriptors, target.descriptors)
    if not len(pairs):
        return None
    a, b = pairs.T
    available = np.isfinite(source.points[a]).all(axis=1)
    a_source, b_target = a[available], b[available]
    result = estimate_pose(
        source.points[a_source],
        target.pixels[b_target],
        matrix,
        target.image_size,
        min_inliers,
        initial_pose=initial_pose,
    )
    if result is None:
        return None
    measurement, valid, error = result
    reverse_available = np.isfinite(target.points[b]).all(axis=1)
    reverse = estimate_pose(
        target.points[b[reverse_available]],
        source.pixels[a[reverse_available]],
        matrix,
        source.image_size,
        min_inliers,
        initial_pose=np.linalg.inv(measurement),
    )
    reverse_translation = reverse_rotation = None
    refinement = None
    if reverse is not None:
        consistency = reverse[0] @ measurement
        reverse_translation = float(np.linalg.norm(consistency[:3, 3]))
        reverse_rotation = float(
            np.degrees(np.linalg.norm(cv.Rodrigues(consistency[:3, :3])[0]))
        )
        if reverse_translation > 0.5 or reverse_rotation > 1.5:
            return None
        reverse_a = a[reverse_available][reverse[1]]
        reverse_b = b[reverse_available][reverse[1]]
        measurement, refinement = refine_bidirectional_stereo(
            measurement, source.points[a_source[valid]], target.pixels[b_target[valid]],
            target.points[reverse_b], source.pixels[reverse_a], matrix,
        )
        predicted, _ = project(source.points[a_source[valid]], measurement, matrix)
        error = float(np.median(np.linalg.norm(predicted-target.pixels[b_target[valid]], axis=1)))
    return {
        "measurement": measurement,
        "matches": len(a_source),
        "inliers": len(valid),
        "median_reprojection_px": error,
        "reverse_checked": reverse is not None,
        "reverse_translation_error_m": reverse_translation,
        "reverse_rotation_error_deg": reverse_rotation,
        "target_features": b_target[valid].tolist(),
        "bidirectional_refinement": refinement,
    }


def refine_bidirectional_stereo(pose, source_points, target_pixels, target_points, source_pixels, matrix):
    """Refine one relative pose using independently verified observations in both views."""
    from scipy.optimize import least_squares

    initial = np.r_[cv.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]]

    def unpack(value):
        camera = np.eye(4)
        camera[:3, :3] = cv.Rodrigues(value[:3])[0]
        camera[:3, 3] = value[3:]
        return camera

    def errors(camera):
        first, z1 = project(source_points, camera, matrix)
        second, z2 = project(target_points, np.linalg.inv(camera), matrix)
        first -= target_pixels
        second -= source_pixels
        first[z1 <= 0] = 1e4
        second[z2 <= 0] = 1e4
        return first, second, z1, z2

    def residual(value):
        first, second, _, _ = errors(unpack(value))
        return np.clip(np.r_[first.ravel(), second.ravel()], -1e4, 1e4)

    def objective(value):
        r = residual(value);a = np.abs(r)
        return float(np.sum(np.where(a <= 1.5, .5*r*r, 1.5*(a-.75))))

    before = objective(initial)
    solved = least_squares(residual, initial, loss='huber', f_scale=1.5, max_nfev=20)
    report = {'applied': False, 'initial_cost': before, 'final_cost': objective(solved.x),
              'forward_observations': len(source_points), 'reverse_observations': len(target_points)}
    if not np.isfinite(solved.x).all() or not report['final_cost'] < before:
        return pose, report
    candidate = unpack(solved.x)
    first, second, z1, z2 = errors(candidate)
    if (np.any(z1 <= 0) or np.any(z2 <= 0) or
            np.median(np.linalg.norm(first, axis=1)) > 1.5 or
            np.median(np.linalg.norm(second, axis=1)) > 1.5):
        return pose, report
    report['applied'] = True
    return candidate, report
