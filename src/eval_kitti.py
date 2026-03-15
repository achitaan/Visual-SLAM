import argparse
import math
import os
from typing import List, Tuple

import numpy as np

from StereoVisualOdometry import StereoVisualOdometry
from VisualOdometry import VisualOdometry


def load_poses_txt(path: str) -> List[np.ndarray]:
    poses = []
    with open(path, "r") as f:
        for line in f:
            values = np.fromstring(line.strip(), dtype=np.float64, sep=" ")
            if values.size != 12:
                continue
            T = values.reshape(3, 4)
            T = np.vstack((T, [0.0, 0.0, 0.0, 1.0]))
            poses.append(T)
    return poses


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    # Align src to dst: dst ~= scale * R * src + t
    assert src.shape == dst.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = dst_centered.T @ src_centered / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = np.var(src_centered, axis=0).sum()
    scale = 1.0 if var_src < 1e-12 else (S.sum() / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return R, scale, t


def compute_ate_rmse(gt: np.ndarray, est: np.ndarray) -> float:
    errors = np.linalg.norm(gt - est, axis=1)
    return math.sqrt(np.mean(errors**2))


def run_vo_sequence(
    image_dir: str,
    calib_path: str,
    camera_id: int,
    max_frames: int | None,
    use_stereo: bool,
    stereo_root: str | None = None,
) -> List[np.ndarray]:
    if use_stereo and stereo_root:
        vo = StereoVisualOdometry(
            stereo_root,
            calib_path,
            use_brute_force=False,
            poses_path=None,
            draw_matches=False,
        )
        num_frames = len(vo.Images_1)
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
        print(f"Running stereo VO on {num_frames} frames...")
        for i in range(1, num_frames):
            T, _ = vo.find_transf_pnp_debug(i)
            vo.poses.append(vo.poses[-1] @ T)
        return vo.poses[:num_frames]

    vo = VisualOdometry(
        image_dir,
        calib_path,
        use_brute_force=False,
        camera_id=camera_id,
        draw_matches=False,
    )
    num_frames = len(vo.Images)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    print(f"Running mono VO on {num_frames} frames...")
    for i in range(1, num_frames):
        p1, p2 = vo.flann_match_features(i)
        T = vo.find_transf(p1, p2)
        vo.poses.append(vo.poses[-1] @ T)
    return vo.poses[:num_frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--data-root", default=r"C:\Users\SSGSS\Documents\Visual-SLAM\data_odometry_gray\dataset")
    parser.add_argument("--poses-root", default=r"C:\Users\SSGSS\Documents\Visual-SLAM\data_odometry_poses\dataset\poses")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stereo", action="store_true")
    args = parser.parse_args()

    seq = args.sequence
    image_dir = os.path.join(args.data_root, "sequences", seq, "image_0")
    calib_path = os.path.join(args.data_root, "sequences", seq, "calib.txt")
    gt_path = os.path.join(args.poses_root, f"{seq}.txt")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing image dir: {image_dir}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Missing calib file: {calib_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Missing gt poses: {gt_path}")

    gt_poses = load_poses_txt(gt_path)
    stereo_root = os.path.join(args.data_root, "sequences", seq, "image_")
    est_poses = run_vo_sequence(
        image_dir,
        calib_path,
        camera_id=0,
        max_frames=args.max_frames,
        use_stereo=args.stereo,
        stereo_root=stereo_root,
    )

    count = min(len(gt_poses), len(est_poses))
    gt_xyz = np.array([pose[:3, 3] for pose in gt_poses[:count]])
    est_xyz = np.array([pose[:3, 3] for pose in est_poses[:count]])

    R, scale, t = umeyama_alignment(est_xyz, gt_xyz)
    est_aligned = (scale * (R @ est_xyz.T)).T + t

    ate = compute_ate_rmse(gt_xyz, est_aligned)
    print(f"Sequence {seq}")
    print(f"Frames evaluated: {count}")
    print(f"Alignment scale: {scale:.4f}")
    print(f"ATE RMSE (m): {ate:.4f}")


if __name__ == "__main__":
    main()
