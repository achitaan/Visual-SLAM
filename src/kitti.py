"""KITTI data validation and trajectory I/O, independent of OpenCV."""

from pathlib import Path

import numpy as np


def load_poses_txt(path: str | Path) -> list[np.ndarray]:
    poses = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = np.array([float(value) for value in line.split()])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid pose values") from exc
        if values.size != 12 or not np.isfinite(values).all():
            raise ValueError(f"{path}:{line_number}: expected 12 finite pose values")
        pose = np.eye(4)
        pose[:3] = values.reshape(3, 4)
        validate_pose(pose)
        poses.append(pose)
    if not poses:
        raise ValueError(f"No poses in {path}")
    return poses


def validate_pose(pose: np.ndarray) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 pose")
    p30 = float(pose[3, 0])
    p31 = float(pose[3, 1])
    p32 = float(pose[3, 2])
    p33 = float(pose[3, 3])
    if (
        abs(p30) > 1e-5
        or abs(p31) > 1e-5
        or abs(p32) > 1e-5
        or abs(p33 - 1.0) > 1e-5 + 1e-5
    ):
        raise ValueError("Invalid homogeneous pose row")
    r00 = float(pose[0, 0]); r01 = float(pose[0, 1]); r02 = float(pose[0, 2])
    r10 = float(pose[1, 0]); r11 = float(pose[1, 1]); r12 = float(pose[1, 2])
    r20 = float(pose[2, 0]); r21 = float(pose[2, 1]); r22 = float(pose[2, 2])
    g00 = r00 * r00 + r10 * r10 + r20 * r20
    g11 = r01 * r01 + r11 * r11 + r21 * r21
    g22 = r02 * r02 + r12 * r12 + r22 * r22
    g01 = r00 * r01 + r10 * r11 + r20 * r21
    g02 = r00 * r02 + r10 * r12 + r20 * r22
    g12 = r01 * r02 + r11 * r12 + r21 * r22
    diag_tol = 1e-3 + 1e-5
    if (
        abs(g00 - 1.0) > diag_tol
        or abs(g11 - 1.0) > diag_tol
        or abs(g22 - 1.0) > diag_tol
        or abs(g01) > 1e-3
        or abs(g02) > 1e-3
        or abs(g12) > 1e-3
    ):
        raise ValueError("Pose rotation must belong to SO(3)")
    det = (
        r00 * (r11 * r22 - r12 * r21)
        - r01 * (r10 * r22 - r12 * r20)
        + r02 * (r10 * r21 - r11 * r20)
    )
    if abs(det - 1.0) > diag_tol:
        raise ValueError("Pose rotation must belong to SO(3)")


def save_poses_txt(path: str | Path, poses: list[np.ndarray]) -> None:
    if not poses:
        raise ValueError("Cannot save an empty trajectory")
    for pose in poses:
        validate_pose(pose)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, np.array([pose[:3].reshape(-1) for pose in poses]), fmt="%.10e")


def image_paths(folder: str | Path, max_frames: int | None = None) -> list[Path]:
    if max_frames is not None and max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image directory: {folder}")
    paths = sorted(path for path in folder.glob("*.png") if path.is_file())
    paths = paths[:max_frames] if max_frames is not None else paths
    if len(paths) < 2:
        raise ValueError(f"Need at least two PNG frames in {folder}")
    return paths


def validate_sequence(root: str | Path, sequence: str, stereo: bool = False,
                      max_frames: int | None = None) -> Path:
    if len(sequence) != 2 or not sequence.isdigit():
        raise ValueError("Sequence must be a two-digit KITTI identifier")
    sequence_dir = Path(root) / "sequences" / sequence
    if not (sequence_dir / "calib.txt").is_file():
        raise FileNotFoundError(f"Missing calibration: {sequence_dir / 'calib.txt'}")
    left = image_paths(sequence_dir / "image_0", max_frames)
    if stereo:
        right = image_paths(sequence_dir / "image_1", max_frames)
        if [path.name for path in left] != [path.name for path in right]:
            raise ValueError(f"Left/right frame names do not match in sequence {sequence}")
    return sequence_dir
