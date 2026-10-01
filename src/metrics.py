"""Trajectory metrics. KITTI drift uses unscaled camera-to-world poses."""
import numpy as np
from kitti import validate_pose


def umeyama_alignment(src, dst, with_scale=True):
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 2:
        raise ValueError("Alignment requires matching Nx3 trajectories with at least two points")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Trajectories must be finite")
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    centered_src, centered_dst = src - src_mean, dst - dst_mean
    u, singular_values, vt = np.linalg.svd(centered_dst.T @ centered_src / len(src))
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    variance = np.mean(np.sum(centered_src ** 2, axis=1))
    if with_scale and variance < 1e-12:
        raise ValueError("Cannot estimate scale from a stationary trajectory")
    scale = float(np.dot(singular_values, signs) / variance) if with_scale else 1.0
    return rotation, scale, dst_mean - scale * (rotation @ src_mean)


def compute_ate_rmse(gt, est):
    gt, est = np.asarray(gt, dtype=float), np.asarray(est, dtype=float)
    if gt.shape != est.shape or gt.ndim != 2 or gt.shape[1] != 3 or len(gt) == 0:
        raise ValueError("ATE requires matching nonempty Nx3 trajectories")
    if not np.isfinite(gt).all() or not np.isfinite(est).all():
        raise ValueError("Trajectories must be finite")
    return float(np.sqrt(np.mean(np.sum((gt - est) ** 2, axis=1))))


def segment_errors(gt_poses, est_poses, lengths=range(100, 801, 100), step=10):
    """Devkit convention: start every 10 frames; end past requested distance."""
    if len(gt_poses) != len(est_poses) or len(gt_poses) < 2:
        raise ValueError("Segment metrics require equal trajectories with at least two poses")
    lengths = tuple(lengths)
    if step < 1 or not lengths or any(length <= 0 for length in lengths):
        raise ValueError("Segment lengths and frame step must be positive")
    for pose in list(gt_poses) + list(est_poses):
        validate_pose(pose)
    xyz = np.array([pose[:3, 3] for pose in gt_poses])
    # Match the official devkit's float32 coordinate differences and cumulative
    # distances, including rounding when choosing each segment endpoint.
    delta = np.diff(xyz, axis=0).astype(np.float32)
    squared = (delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]) + delta[:, 2] * delta[:, 2]
    distances = np.r_[np.float32(0), np.cumsum(np.sqrt(squared), dtype=np.float32)]
    errors = []
    for first in range(0, len(gt_poses), step):
        for length in lengths:
            threshold = np.float32(distances[first] + np.float32(length))
            last = int(np.searchsorted(distances, threshold, side="right"))
            if last >= len(gt_poses):
                continue
            gt_delta = np.linalg.inv(gt_poses[first]) @ gt_poses[last]
            est_delta = np.linalg.inv(est_poses[first]) @ est_poses[last]
            error = np.linalg.inv(est_delta) @ gt_delta
            angle = np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1.0, 1.0))
            errors.append({"first_frame": first, "last_frame": last, "length_m": float(length),
                           "translation_percent": float(100 * np.linalg.norm(error[:3, 3]) / length),
                           "rotation_deg_per_m": float(np.degrees(angle) / length)})
    return errors


def evaluate_trajectory(gt_poses, est_poses, alignment="se3"):
    if len(gt_poses) != len(est_poses) or len(gt_poses) < 2:
        raise ValueError("Evaluation requires equal trajectories with at least two poses")
    if alignment not in ("none", "se3", "sim3"):
        raise ValueError("alignment must be none, se3, or sim3")
    segments = segment_errors(gt_poses, est_poses)
    gt = np.array([pose[:3, 3] for pose in gt_poses])
    est = np.array([pose[:3, 3] for pose in est_poses])
    aligned, scale = est, 1.0
    if alignment != "none":
        rotation, scale, translation = umeyama_alignment(est, gt, with_scale=alignment == "sim3")
        aligned = (scale * (rotation @ est.T)).T + translation
    return {"frames": len(gt), "ate_alignment": alignment, "alignment_scale": scale,
            "ate_rmse_m": compute_ate_rmse(gt, aligned), "raw_ate_rmse_m": compute_ate_rmse(gt, est),
            "segment_count": len(segments),
            "translation_percent": float(np.mean([s["translation_percent"] for s in segments])) if segments else None,
            "rotation_deg_per_m": float(np.mean([s["rotation_deg_per_m"] for s in segments])) if segments else None,
            "segments": segments}
