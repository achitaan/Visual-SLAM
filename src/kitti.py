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
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous pose row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
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
