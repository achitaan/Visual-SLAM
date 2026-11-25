"""Pluggable stereo/monocular frame sources, calibration loaders, and rectification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from shared_slam import StereoCamera


@dataclass(frozen=True)
class StereoFrame:
    """Single timestamped stereo (or monocular) frame emitted by a FrameSource."""

    frame_id: int
    timestamp_s: float
    left: np.ndarray
    right: np.ndarray | None = None
    source_fps: float | None = None
    dropped_before: int = 0
    captured_at_s: float = field(default_factory=time.perf_counter)


def _as_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"Calibration field '{name}' must have finite shape {shape}, got {arr.shape}")
    return arr.copy()


def _as_optional_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _as_matrix(value, shape, name)


def _as_dist(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(5, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Distortion coefficients must be finite")
    return arr.copy()


def _q_from_projections(P1: np.ndarray, P2: np.ndarray) -> tuple[np.ndarray, float]:
    f = float(P1[0, 0])
    f_right = float(P2[0, 0])
    fy_left = float(P1[1, 1])
    fy_right = float(P2[1, 1])
    if (
        not np.isfinite(f) or f <= 0.0
        or not np.isfinite(f_right) or f_right <= 0.0
        or not np.isfinite(fy_left) or fy_left <= 0.0
        or not np.isfinite(fy_right) or fy_right <= 0.0
    ):
        raise ValueError("Stereo calibration requires positive finite focal lengths")
    cx_left = float(P1[0, 2])
    cy_left = float(P1[1, 2])
    cx_right = float(P2[0, 2])
    tx = float(P2[0, 3]) / f_right
    baseline = float(P1[0, 3]) / f - tx
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("Stereo calibration requires a positive left-to-right baseline")
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -cx_left],
            [0.0, 1.0, 0.0, -cy_left],
            [0.0, 0.0, 0.0, f],
            [0.0, 0.0, 1.0 / baseline, (cx_right - cx_left) / baseline],
        ],
        dtype=np.float32,
    )
    return Q, float(baseline)


@dataclass
class StereoCalibration:
    """Stereo or monocular camera intrinsics, extrinsics, and rectification geometry."""

    K_left: np.ndarray
    K_right: np.ndarray | None = None
    dist_left: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    dist_right: np.ndarray | None = None
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    baseline: float = 0.0
    image_size: tuple[int, int] | None = None
    is_rectified: bool = True

    def __post_init__(self) -> None:
        self.K_left = _as_matrix(self.K_left, (3, 3), "K_left")
        self.dist_left = _as_dist(self.dist_left)
        self.K_right = _as_optional_matrix(self.K_right, (3, 3), "K_right")
        self.dist_right = _as_dist(self.dist_right) if self.dist_right is not None else None
        self.R = _as_optional_matrix(self.R, (3, 3), "R")
        if self.T is not None:
            self.T = _as_matrix(np.asarray(self.T, dtype=np.float64).reshape(3), (3,), "T")
        self.R1 = _as_optional_matrix(self.R1, (3, 3), "R1")
        self.R2 = _as_optional_matrix(self.R2, (3, 3), "R2")
        self.P1 = _as_optional_matrix(self.P1, (3, 4), "P1")
        self.P2 = _as_optional_matrix(self.P2, (3, 4), "P2")
        if self.Q is not None:
            self.Q = np.asarray(self.Q, dtype=np.float32)
            if self.Q.shape != (4, 4) or not np.isfinite(self.Q).all():
                raise ValueError("Calibration Q must be a finite 4x4 matrix")
        if self.image_size is not None:
            w, h = int(self.image_size[0]), int(self.image_size[1])
            if w <= 0 or h <= 0:
                raise ValueError("image_size dimensions must be positive")
            self.image_size = (w, h)
        if self.P1 is not None and self.P2 is not None:
            Q_calc, baseline = _q_from_projections(self.P1, self.P2)
            if self.baseline == 0.0:
                self.baseline = baseline
            if self.Q is None:
                self.Q = Q_calc
        elif self.baseline == 0.0 and self.T is not None:
            self.baseline = float(np.linalg.norm(self.T))

    @property
    def is_stereo(self) -> bool:
        return self.K_right is not None or (self.P2 is not None and self.baseline > 0.0)

    def rectified_intrinsics(self) -> np.ndarray:
        if self.P1 is not None:
            return self.P1[:3, :3].copy()
        return self.K_left.copy()

    def ensure_rectification(
        self,
        image_size: tuple[int, int] | None = None,
        *,
        alpha: float = 0.0,
    ) -> "StereoCalibration":
        """Compute R1, R2, P1, P2, Q via cv.stereoRectify if not already populated."""
        size = image_size or self.image_size
        if not self.is_stereo:
            if self.P1 is None:
                P1 = np.zeros((3, 4), dtype=np.float64)
                P1[:3, :3] = self.K_left
                self.P1 = P1
            if self.R1 is None:
                self.R1 = np.eye(3, dtype=np.float64)
            return self
        if self.P1 is not None and self.P2 is not None and self.Q is not None:
            if self.R1 is None:
                self.R1 = np.eye(3, dtype=np.float64)
            if self.R2 is None:
                self.R2 = np.eye(3, dtype=np.float64)
            if self.baseline <= 0.0:
                _, self.baseline = _q_from_projections(self.P1, self.P2)
            return self
        if size is None:
            raise ValueError("image_size is required to compute stereo rectification matrices")
        if self.K_right is None or self.R is None or self.T is None:
            raise ValueError("Unrectified stereo calibration requires K_right, R, and T")
        dist_r = self.dist_right if self.dist_right is not None else np.zeros(5, dtype=np.float64)
        R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
            self.K_left,
            self.dist_left,
            self.K_right,
            dist_r,
            (int(size[0]), int(size[1])),
            self.R,
            self.T.reshape(3, 1),
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
        )
        self.R1 = R1.astype(np.float64)
        self.R2 = R2.astype(np.float64)
        self.P1 = P1.astype(np.float64)
        self.P2 = P2.astype(np.float64)
        Q_calc, baseline = _q_from_projections(self.P1, self.P2)
        self.Q = Q_calc if not np.isfinite(Q).all() else Q.astype(np.float32)
        # Keep sign convention consistent with StereoVisualOdometry (Q[3,2] = +1/B)
        if float(self.Q[3, 2]) < 0.0:
            self.Q = Q_calc
        self.baseline = baseline
        self.image_size = (int(size[0]), int(size[1]))
        return self

    def make_stereo_camera(
        self,
        *,
        num_disparities: int = 96,
        block_size: int = 5,
        min_disparity: int = 0,
        image_size: tuple[int, int] | None = None,
    ) -> StereoCamera | None:
        """Build a SharedSlam StereoCamera matching the repository SGBM parameters."""
        if not self.is_stereo:
            return None
        self.ensure_rectification(image_size=image_size)
        if self.Q is None or self.baseline <= 0.0:
            raise ValueError("Stereo calibration lacks valid Q matrix or positive baseline")
        sgbm = cv.StereoSGBM_create(
            minDisparity=int(min_disparity),
            numDisparities=int(num_disparities),
            blockSize=int(block_size),
            P1=8 * 3 * int(block_size) ** 2,
            P2=32 * 3 * int(block_size) ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
        return StereoCamera(sgbm, self.Q.copy(), float(self.baseline))

    @classmethod
    def from_kitti_calib(
        cls,
        path: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        stereo: bool = True,
        camera_pair: tuple[str, str] = ("P0", "P1"),
    ) -> "StereoCalibration":
        """Load KITTI calib.txt (supports labeled 'P0:'/'P1:' and unlabeled 2-line formats)."""
        text = Path(path).read_text(encoding="utf-8")
        labeled: dict[str, np.ndarray] = {}
        unlabeled: list[np.ndarray] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, rhs = line.partition(":")
                vals = [float(v) for v in rhs.split()]
                if len(vals) == 12:
                    labeled[key.strip()] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
            else:
                vals = [float(v) for v in line.split()]
                if len(vals) == 12:
                    unlabeled.append(np.asarray(vals, dtype=np.float64).reshape(3, 4))

        left_key, right_key = camera_pair
        if left_key in labeled:
            P1 = labeled[left_key]
            P2 = labeled.get(right_key) if stereo else None
        elif unlabeled:
            P1 = unlabeled[0]
            P2 = unlabeled[1] if (stereo and len(unlabeled) > 1) else None
        else:
            raise ValueError(f"Could not parse KITTI projection matrices from {path}")

        K_left = P1[:3, :3].copy()
        if stereo and P2 is not None:
            K_right = P2[:3, :3].copy()
            Q, baseline = _q_from_projections(P1, P2)
            T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
            return cls(
                K_left=K_left,
                K_right=K_right,
                dist_left=np.zeros(5, dtype=np.float64),
                dist_right=np.zeros(5, dtype=np.float64),
                R=np.eye(3, dtype=np.float64),
                T=T,
                R1=np.eye(3, dtype=np.float64),
                R2=np.eye(3, dtype=np.float64),
                P1=P1,
                P2=P2,
                Q=Q,
                baseline=baseline,
                image_size=image_size,
                is_rectified=True,
            )
        return cls(
            K_left=K_left,
            dist_left=np.zeros(5, dtype=np.float64),
            R1=np.eye(3, dtype=np.float64),
            P1=P1,
            baseline=0.0,
            image_size=image_size,
            is_rectified=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StereoCalibration":
        if not isinstance(data, dict):
            raise ValueError("Calibration data must be a mapping")
        K_left = data.get("K_left", data.get("camera_matrix_left", data.get("K1")))
        if K_left is None:
            raise ValueError("Calibration missing K_left / camera_matrix_left / K1")
        K_right = data.get("K_right", data.get("camera_matrix_right", data.get("K2")))
        dist_left = data.get("dist_left", data.get("dist_coeffs_left", data.get("D1")))
        dist_right = data.get("dist_right", data.get("dist_coeffs_right", data.get("D2")))
        image_size_raw = data.get("image_size")
        image_size = None
        if image_size_raw is not None:
            if isinstance(image_size_raw, dict):
                image_size = (int(image_size_raw["width"]), int(image_size_raw["height"]))
            else:
                image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        elif "image_width" in data and "image_height" in data:
            image_size = (int(data["image_width"]), int(data["image_height"]))

        is_rectified = data.get("is_rectified")
        if is_rectified is None:
            dl = _as_dist(dist_left)
            dr = _as_dist(dist_right) if dist_right is not None else np.zeros(5)
            R_raw = data.get("R")
            r_is_eye = R_raw is None or np.allclose(np.asarray(R_raw, dtype=np.float64), np.eye(3), atol=1e-6)
            is_rectified = bool(
                np.allclose(dl, 0.0, atol=1e-9)
                and np.allclose(dr, 0.0, atol=1e-9)
                and r_is_eye
            )

        calib = cls(
            K_left=K_left,
            K_right=K_right,
            dist_left=dist_left,
            dist_right=dist_right,
            R=data.get("R"),
            T=data.get("T"),
            R1=data.get("R1"),
            R2=data.get("R2"),
            P1=data.get("P1"),
            P2=data.get("P2"),
            Q=data.get("Q"),
            baseline=float(data.get("baseline", 0.0)),
            image_size=image_size,
            is_rectified=bool(is_rectified),
        )
        if calib.is_stereo and calib.image_size is not None and (calib.P1 is None or calib.P2 is None or calib.Q is None):
            if calib.R is not None and calib.T is not None:
                calib.ensure_rectification()
        return calib

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StereoCalibration":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "StereoCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        stereo: bool = True,
    ) -> "StereoCalibration":
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(p)
        if suffix == ".json":
            return cls.from_json(p)
        return cls.from_kitti_calib(p, image_size=image_size, stereo=stereo)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "K_left": self.K_left.tolist(),
            "dist_left": self.dist_left.tolist(),
            "baseline": float(self.baseline),
            "is_rectified": bool(self.is_rectified),
        }
        if self.image_size is not None:
            out["image_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        for attr in ("K_right", "dist_right", "R", "T", "R1", "R2", "P1", "P2", "Q"):
            val = getattr(self, attr)
            if val is not None:
                out[attr] = np.asarray(val, dtype=np.float64).tolist()
        return out

    def save_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class StereoRectifier:
    """Precomputed LUT stereo/monocular undistortion and rectification."""

    def __init__(
        self,
        calibration: StereoCalibration,
        image_size: tuple[int, int] | None = None,
        *,
        alpha: float = 0.0,
        gpu_accelerator: Any = None,
    ) -> None:
        self.calibration = calibration
        self.alpha = float(alpha)
        self.gpu_accelerator = gpu_accelerator
        self._size: tuple[int, int] | None = None
        self._map1_l: np.ndarray | None = None
        self._map2_l: np.ndarray | None = None
        self._map1_r: np.ndarray | None = None
        self._map2_r: np.ndarray | None = None
        self._map_x_l_f32: np.ndarray | None = None
        self._map_y_l_f32: np.ndarray | None = None
        self._map_x_r_f32: np.ndarray | None = None
        self._map_y_r_f32: np.ndarray | None = None
        self._noop = False
        size = image_size or calibration.image_size
        if size is not None:
            self._init_maps(size)

    @property
    def is_noop(self) -> bool:
        return self._noop

    def _check_noop(self) -> bool:
        c = self.calibration
        if not c.is_rectified:
            return False
        if not np.allclose(c.dist_left, 0.0, atol=1e-9):
            return False
        if c.dist_right is not None and not np.allclose(c.dist_right, 0.0, atol=1e-9):
            return False
        if c.R1 is not None and not np.allclose(c.R1, np.eye(3), atol=1e-7):
            return False
        if c.R2 is not None and not np.allclose(c.R2, np.eye(3), atol=1e-7):
            return False
        if c.P1 is not None and not np.allclose(c.P1[:3, :3], c.K_left, atol=1e-6):
            return False
        if c.K_right is not None and c.P2 is not None and not np.allclose(c.P2[:3, :3], c.K_right, atol=1e-6):
            return False
        return True

    def _init_maps(self, size: tuple[int, int]) -> None:
        w, h = int(size[0]), int(size[1])
        self._size = (w, h)
        self._noop = self._check_noop()
        if self._noop:
            return
        self.calibration.ensure_rectification(image_size=(w, h), alpha=self.alpha)
        c = self.calibration
        R1 = c.R1 if c.R1 is not None else np.eye(3, dtype=np.float64)
        P1 = c.P1 if c.P1 is not None else c.K_left
        self._map_x_l_f32, self._map_y_l_f32 = cv.initUndistortRectifyMap(
            c.K_left, c.dist_left, R1, P1, (w, h), cv.CV_32FC1
        )
        self._map1_l, self._map2_l = cv.convertMaps(
            self._map_x_l_f32, self._map_y_l_f32, cv.CV_16SC2
        )
        if c.is_stereo and c.K_right is not None:
            R2 = c.R2 if c.R2 is not None else np.eye(3, dtype=np.float64)
            P2 = c.P2 if c.P2 is not None else c.K_right
            dist_r = c.dist_right if c.dist_right is not None else np.zeros(5, dtype=np.float64)
            self._map_x_r_f32, self._map_y_r_f32 = cv.initUndistortRectifyMap(
                c.K_right, dist_r, R2, P2, (w, h), cv.CV_32FC1
            )
            self._map1_r, self._map2_r = cv.convertMaps(
                self._map_x_r_f32, self._map_y_r_f32, cv.CV_16SC2
            )

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        h, w = left.shape[:2]
        if self._size != (w, h):
            self._init_maps((w, h))
        if self._noop:
            return left, right
        if (
            self.gpu_accelerator is not None
            and self._map_x_l_f32 is not None
            and self._map_y_l_f32 is not None
        ):
            try:
                rect_l = self.gpu_accelerator.rectify_image(left, self._map_x_l_f32, self._map_y_l_f32)
                rect_r = (
                    self.gpu_accelerator.rectify_image(right, self._map_x_r_f32, self._map_y_r_f32)
                    if right is not None and self._map_x_r_f32 is not None and self._map_y_r_f32 is not None
                    else right
                )
                return rect_l, rect_r
            except Exception:
                pass
        rect_l = cv.remap(left, self._map1_l, self._map2_l, cv.INTER_LINEAR)
        rect_r = (
            cv.remap(right, self._map1_r, self._map2_r, cv.INTER_LINEAR)
            if right is not None and self._map1_r is not None
            else right
        )
        return rect_l, rect_r

    def rectify_frame(self, frame: StereoFrame) -> StereoFrame:
        rect_l, rect_r = self.rectify(frame.left, frame.right)
        if rect_l is frame.left and rect_r is frame.right:
            return frame
        return StereoFrame(
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            left=rect_l,
            right=rect_r,
            source_fps=frame.source_fps,
            dropped_before=frame.dropped_before,
            captured_at_s=frame.captured_at_s,
        )


class FrameSource(ABC):
    """Abstract real-time or replay frame source."""

    @property
    @abstractmethod
    def calibration(self) -> StereoCalibration | None:
        """Camera calibration associated with this source."""

    @property
    def source_name(self) -> str:
        return self.__class__.__name__

    @property
    def total_frames(self) -> int | None:
        return None

    @property
    def fps(self) -> float | None:
        return None

    @abstractmethod
    def open(self) -> None:
        """Open underlying hardware or dataset stream."""

    @abstractmethod
    def read(self) -> StereoFrame | None:
        """Read next frame, or None on end-of-stream / closed."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""

    def set_playback_rate(self, rate: float) -> None:
        """Optionally adjust playback speed (no-op for live cameras)."""
        _ = rate

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _resolve_kitti_sequence_dir(root_or_dir: str | Path, sequence: str | None) -> Path:
    base = Path(root_or_dir)
    if sequence is not None and (base / "sequences" / sequence).is_dir():
        return base / "sequences" / sequence
    if (base / "calib.txt").is_file():
        return base
    if sequence is not None and (base / sequence).is_dir():
        return base / sequence
    raise FileNotFoundError(
        f"Could not locate KITTI sequence directory from root={root_or_dir!r}, sequence={sequence!r}"
    )


def _load_kitti_times(seq_dir: Path, count: int, default_fps: float = 10.0) -> list[float]:
    times_path = seq_dir / "times.txt"
    if times_path.is_file():
        values = [float(line.strip()) for line in times_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(values) >= count:
            return values[:count]
    dt = 1.0 / max(default_fps, 1e-6)
    return [i * dt for i in range(count)]


class PacedKittiSource(FrameSource):
    """KITTI odometry dataset reader with wall-clock timestamp pacing and optional frame-drop."""

    def __init__(
        self,
        data_root: str | Path,
        sequence: str | None = "00",
        *,
        stereo: bool = True,
        rate: float = 1.0,
        drop_stale: bool = True,
        max_frames: int | None = None,
        loop: bool = False,
        default_fps: float = 10.0,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.stereo = bool(stereo)
        self._rate = float(rate)
        self.drop_stale = bool(drop_stale)
        self.max_frames = max_frames
        self.loop = bool(loop)
        self.default_fps = float(default_fps)
        self._clock = clock
        self._sleeper = sleeper

        self._seq_dir: Path | None = None
        self._left_paths: list[Path] = []
        self._right_paths: list[Path] = []
        self._times: list[float] = []
        self._calib: StereoCalibration | None = None
        self._cursor: int = 0
        self._wall_origin_s: float | None = None
        self._time_origin_s: float = 0.0
        self._opened = False
        self._lock = threading.Lock()
        self._source_fps: float = default_fps

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        seq = self.sequence or (self._seq_dir.name if self._seq_dir else "custom")
        return f"kitti:{seq}"

    @property
    def total_frames(self) -> int | None:
        return len(self._left_paths) if self._opened else None

    @property
    def fps(self) -> float | None:
        return self._source_fps * self._rate if self._rate > 0 else self._source_fps

    def set_playback_rate(self, rate: float) -> None:
        with self._lock:
            now = self._clock()
            if self._wall_origin_s is not None and self._cursor > 0 and self._rate > 0:
                last_idx = min(self._cursor - 1, len(self._times) - 1)
                self._time_origin_s = self._times[last_idx]
                self._wall_origin_s = now
            self._rate = float(rate)

    def open(self) -> None:
        if self._opened:
            return
        seq_dir = _resolve_kitti_sequence_dir(self.data_root, self.sequence)
        self._seq_dir = seq_dir
        left_dir = seq_dir / "image_0"
        if not left_dir.is_dir():
            left_dir = seq_dir / "image_l"
        if not left_dir.is_dir():
            raise FileNotFoundError(f"Missing left image folder in {seq_dir}")
        left_paths = sorted(p for p in left_dir.glob("*.png") if p.is_file())
        if self.max_frames is not None:
            left_paths = left_paths[: int(self.max_frames)]
        if not left_paths:
            raise ValueError(f"No PNG frames found in {left_dir}")
        right_paths: list[Path] = []
        if self.stereo:
            right_dir = seq_dir / "image_1"
            if not right_dir.is_dir():
                right_dir = seq_dir / "image_r"
            if right_dir.is_dir():
                right_paths = sorted(p for p in right_dir.glob("*.png") if p.is_file())
                if self.max_frames is not None:
                    right_paths = right_paths[: int(self.max_frames)]
                if len(right_paths) != len(left_paths):
                    raise ValueError(
                        f"Left ({len(left_paths)}) and right ({len(right_paths)}) counts differ in {seq_dir}"
                    )
            else:
                self.stereo = False

        sample = cv.imread(str(left_paths[0]), cv.IMREAD_GRAYSCALE)
        if sample is None:
            raise OSError(f"Failed to read first KITTI frame: {left_paths[0]}")
        img_size = (int(sample.shape[1]), int(sample.shape[0]))
        calib_path = seq_dir / "calib.txt"
        self._calib = StereoCalibration.from_kitti_calib(
            calib_path, image_size=img_size, stereo=self.stereo
        )
        self._left_paths = left_paths
        self._right_paths = right_paths
        self._times = _load_kitti_times(seq_dir, len(left_paths), default_fps=self.default_fps)
        if len(self._times) >= 2:
            dts = np.diff(self._times)
            pos = dts[dts > 1e-6]
            if len(pos):
                self._source_fps = float(1.0 / np.median(pos))
        self._cursor = 0
        self._wall_origin_s = None
        self._time_origin_s = self._times[0] if self._times else 0.0
        self._opened = True

    def read(self) -> StereoFrame | None:
        if not self._opened:
            self.open()
        with self._lock:
            n = len(self._left_paths)
            if self._cursor >= n:
                if not self.loop:
                    return None
                self._cursor = 0
                self._wall_origin_s = None
                self._time_origin_s = self._times[0]

            rate = self._rate
            idx = self._cursor
            dropped = 0
            if rate > 0.0:
                now = self._clock()
                if self._wall_origin_s is None:
                    self._wall_origin_s = now
                    self._time_origin_s = self._times[idx]
                else:
                    elapsed_stream = (now - self._wall_origin_s) * rate
                    target_stream = self._times[idx] - self._time_origin_s
                    if self.drop_stale and elapsed_stream > target_stream:
                        # Skip frames whose scheduled timestamp is already behind wall clock.
                        while (
                            idx + 1 < n
                            and (self._times[idx + 1] - self._time_origin_s) <= elapsed_stream
                        ):
                            idx += 1
                            dropped += 1
                    elif target_stream > elapsed_stream:
                        sleep_s = (target_stream - elapsed_stream) / rate
                        if sleep_s > 0.0:
                            self._lock.release()
                            try:
                                self._sleeper(sleep_s)
                            finally:
                                self._lock.acquire()
            self._cursor = idx + 1
            ts = float(self._times[idx])
            left_path = self._left_paths[idx]
            right_path = self._right_paths[idx] if self.stereo and self._right_paths else None

        left = cv.imread(str(left_path), cv.IMREAD_GRAYSCALE)
        if left is None:
            raise OSError(f"Failed to read left frame: {left_path}")
        right = None
        if right_path is not None:
            right = cv.imread(str(right_path), cv.IMREAD_GRAYSCALE)
            if right is None:
                raise OSError(f"Failed to read right frame: {right_path}")

        return StereoFrame(
            frame_id=int(idx),
            timestamp_s=ts,
            left=left,
            right=right,
            source_fps=self.fps,
            dropped_before=int(dropped),
        )

    def close(self) -> None:
        with self._lock:
            self._opened = False


class StereoVideoFileSource(FrameSource):
    """Side-by-side stereo video (.mp4/.mkv) or dual left/right video file source."""

    def __init__(
        self,
        left_path: str | Path,
        right_path: str | Path | None = None,
        *,
        calibration: StereoCalibration | str | Path | None = None,
        layout: str = "auto",
        rate: float = 1.0,
        drop_stale: bool = False,
        grayscale: bool = False,
        max_frames: int | None = None,
        loop: bool = False,
        gpu_accelerator: Any = None,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        if layout not in ("auto", "sbs", "dual", "mono"):
            raise ValueError("layout must be 'auto', 'sbs', 'dual', or 'mono'")
        self.left_path = Path(left_path)
        self.right_path = Path(right_path) if right_path is not None else None
        if layout == "auto":
            self.layout = "dual" if self.right_path is not None else "sbs"
        else:
            self.layout = layout
        self._calib = (
            StereoCalibration.from_file(calibration)
            if isinstance(calibration, (str, Path))
            else calibration
        )
        self._rectifier = (
            StereoRectifier(self._calib, gpu_accelerator=gpu_accelerator)
            if self._calib is not None
            else None
        )
        self._rate = float(rate)
        self.drop_stale = bool(drop_stale)
        self.grayscale = bool(grayscale)
        self.max_frames = max_frames
        self.loop = bool(loop)
        self._clock = clock
        self._sleeper = sleeper

        self._cap_l: cv.VideoCapture | None = None
        self._cap_r: cv.VideoCapture | None = None
        self._fps: float = 30.0
        self._total: int | None = None
        self._frame_id: int = 0
        self._wall_origin_s: float | None = None
        self._opened = False

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        return f"video:{self.left_path.name}"

    @property
    def total_frames(self) -> int | None:
        return self._total

    @property
    def fps(self) -> float | None:
        return self._fps * self._rate if self._rate > 0 else self._fps

    def set_playback_rate(self, rate: float) -> None:
        self._rate = float(rate)

    def open(self) -> None:
        if self._opened:
            return
        if not self.left_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.left_path}")
        self._cap_l = cv.VideoCapture(str(self.left_path))
        if not self._cap_l.isOpened():
            raise OSError(f"Failed to open video file: {self.left_path}")
        if self.layout == "dual":
            if self.right_path is None or not self.right_path.is_file():
                raise FileNotFoundError(f"Right video file not found: {self.right_path}")
            self._cap_r = cv.VideoCapture(str(self.right_path))
            if not self._cap_r.isOpened():
                self._cap_l.release()
                raise OSError(f"Failed to open right video file: {self.right_path}")
        fps = float(self._cap_l.get(cv.CAP_PROP_FPS) or 0.0)
        self._fps = fps if np.isfinite(fps) and fps > 0.1 else 30.0
        total = int(self._cap_l.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        if self.max_frames is not None and total > 0:
            total = min(total, int(self.max_frames))
        self._total = total if total > 0 else self.max_frames
        self._frame_id = 0
        self._wall_origin_s = None
        self._opened = True

    def _split_and_format(
        self, raw_l: np.ndarray, raw_r: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.layout == "sbs":
            w = raw_l.shape[1]
            half = w // 2
            left = np.ascontiguousarray(raw_l[:, :half])
            right = np.ascontiguousarray(raw_l[:, half : 2 * half])
        elif self.layout == "dual":
            left, right = raw_l, raw_r
        else:
            left, right = raw_l, None
        if self.grayscale:
            if left.ndim == 3:
                left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
            if right is not None and right.ndim == 3:
                right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        if self._rectifier is not None:
            left, right = self._rectifier.rectify(left, right)
        return left, right

    def read(self) -> StereoFrame | None:
        if not self._opened:
            self.open()
        assert self._cap_l is not None
        if self.max_frames is not None and self._frame_id >= self.max_frames:
            return None
        dropped = 0
        if self._rate > 0.0:
            now = self._clock()
            if self._wall_origin_s is None:
                self._wall_origin_s = now
            else:
                target_s = self._frame_id / (self._fps * self._rate)
                elapsed_s = now - self._wall_origin_s
                if self.drop_stale and elapsed_s > target_s + (1.0 / (self._fps * self._rate)):
                    skip_n = int((elapsed_s - target_s) * self._fps * self._rate)
                    for _ in range(skip_n):
                        if self.max_frames is not None and self._frame_id + 1 >= self.max_frames:
                            break
                        if not self._cap_l.grab():
                            break
                        if self._cap_r is not None:
                            self._cap_r.grab()
                        self._frame_id += 1
                        dropped += 1
                elif target_s > elapsed_s:
                    self._sleeper(target_s - elapsed_s)

        ok_l, raw_l = self._cap_l.read()
        if not ok_l or raw_l is None:
            if self.loop and self._frame_id > 0:
                self._cap_l.set(cv.CAP_PROP_POS_FRAMES, 0)
                if self._cap_r is not None:
                    self._cap_r.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                self._wall_origin_s = self._clock()
                ok_l, raw_l = self._cap_l.read()
                if not ok_l or raw_l is None:
                    return None
            else:
                return None
        raw_r = None
        if self._cap_r is not None:
            ok_r, raw_r = self._cap_r.read()
            if not ok_r or raw_r is None:
                return None

        fid = self._frame_id
        self._frame_id += 1
        ts = fid / self._fps
        left, right = self._split_and_format(raw_l, raw_r)
        return StereoFrame(
            frame_id=fid,
            timestamp_s=float(ts),
            left=left,
            right=right,
            source_fps=self.fps,
            dropped_before=dropped,
        )

    def close(self) -> None:
        self._opened = False
        if self._cap_l is not None:
            self._cap_l.release()
            self._cap_l = None
        if self._cap_r is not None:
            self._cap_r.release()
            self._cap_r = None


class OpenCVStereoCameraSource(FrameSource):
    """Live UVC stereo camera (side-by-side or dual device indices) on a dedicated capture thread."""

    def __init__(
        self,
        device_id: int = 0,
        right_device_id: int | None = None,
        *,
        calibration: StereoCalibration | str | Path | None = None,
        layout: str = "auto",
        width: int | None = None,
        height: int | None = None,
        target_fps: float = 30.0,
        grayscale: bool = False,
        gpu_accelerator: Any = None,
        capture_factory=cv.VideoCapture,
    ) -> None:
        if layout not in ("auto", "sbs", "dual", "mono"):
            raise ValueError("layout must be 'auto', 'sbs', 'dual', or 'mono'")
        self.device_id = int(device_id)
        self.right_device_id = int(right_device_id) if right_device_id is not None else None
        self.layout = ("dual" if self.right_device_id is not None else "sbs") if layout == "auto" else layout
        self.width = width
        self.height = height
        self.target_fps = float(target_fps)
        self.grayscale = bool(grayscale)
        self._capture_factory = capture_factory
        self._calib = (
            StereoCalibration.from_file(calibration)
            if isinstance(calibration, (str, Path))
            else calibration
        )
        self._rectifier = (
            StereoRectifier(self._calib, gpu_accelerator=gpu_accelerator)
            if self._calib is not None
            else None
        )
        self._cap_l: Any = None
        self._cap_r: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: StereoFrame | None = None
        self._dropped_since_read: int = 0
        self._seq: int = 0
        self._opened = False

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        if self.right_device_id is not None:
            return f"uvc:{self.device_id}+{self.right_device_id}"
        return f"uvc:{self.device_id}"

    @property
    def fps(self) -> float | None:
        return self.target_fps

    def open(self) -> None:
        if self._opened:
            return
        self._cap_l = self._capture_factory(self.device_id)
        if not self._cap_l.isOpened():
            raise OSError(f"Failed to open camera device {self.device_id}")
        if self.width is not None:
            self._cap_l.set(cv.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            self._cap_l.set(cv.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.target_fps > 0:
            self._cap_l.set(cv.CAP_PROP_FPS, float(self.target_fps))
        if self.layout == "dual" and self.right_device_id is not None:
            self._cap_r = self._capture_factory(self.right_device_id)
            if not self._cap_r.isOpened():
                self._cap_l.release()
                raise OSError(f"Failed to open right camera device {self.right_device_id}")
            if self.width is not None:
                self._cap_r.set(cv.CAP_PROP_FRAME_WIDTH, int(self.width))
            if self.height is not None:
                self._cap_r.set(cv.CAP_PROP_FRAME_HEIGHT, int(self.height))
            if self.target_fps > 0:
                self._cap_r.set(cv.CAP_PROP_FPS, float(self.target_fps))

        self._stop.clear()
        self._opened = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="uvc-capture", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            if self._cap_l is None:
                break
            ok_l, raw_l = self._cap_l.read()
            if not ok_l or raw_l is None:
                time.sleep(0.005)
                continue
            raw_r = None
            if self._cap_r is not None:
                ok_r, raw_r = self._cap_r.read()
                if not ok_r or raw_r is None:
                    continue
            if self.layout == "sbs":
                half = raw_l.shape[1] // 2
                left = np.ascontiguousarray(raw_l[:, :half])
                right = np.ascontiguousarray(raw_l[:, half : 2 * half])
            elif self.layout == "dual":
                left, right = raw_l, raw_r
            else:
                left, right = raw_l, None
            if self.grayscale:
                if left.ndim == 3:
                    left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
                if right is not None and right.ndim == 3:
                    right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
            if self._rectifier is not None:
                left, right = self._rectifier.rectify(left, right)
            now_mono = time.monotonic()
            frame = StereoFrame(
                frame_id=self._seq,
                timestamp_s=float(now_mono - t0),
                left=left,
                right=right,
                source_fps=self.target_fps,
            )
            self._seq += 1
            with self._cond:
                if self._latest is not None:
                    self._dropped_since_read += 1
                self._latest = frame
                self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> StereoFrame | None:
        if not self._opened:
            self.open()
        with self._cond:
            if self._latest is None and not self._stop.is_set():
                self._cond.wait(timeout=timeout)
            if self._latest is None:
                return None
            frame = self._latest
            dropped = self._dropped_since_read
            self._latest = None
            self._dropped_since_read = 0
        if dropped:
            return StereoFrame(
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
                left=frame.left,
                right=frame.right,
                source_fps=frame.source_fps,
                dropped_before=frame.dropped_before + dropped,
                captured_at_s=frame.captured_at_s,
            )
        return frame

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap_l is not None:
            self._cap_l.release()
            self._cap_l = None
        if self._cap_r is not None:
            self._cap_r.release()
            self._cap_r = None
        self._opened = False


class RealSenseStereoSource(FrameSource):
    """Optional Intel RealSense D435/D455 IR stereo source (requires pyrealsense2)."""

    def __init__(
        self,
        *,
        width: int = 848,
        height: int = 480,
        target_fps: int = 30,
        disable_emitter: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(target_fps)
        self.disable_emitter = bool(disable_emitter)
        self._pipeline: Any = None
        self._calib: StereoCalibration | None = None
        self._frame_id = 0
        self._t0_ms: float | None = None

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        return "realsense:ir_stereo"

    @property
    def fps(self) -> float | None:
        return float(self.target_fps)

    def open(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "RealSenseStereoSource requires the optional 'pyrealsense2' package. "
                "Install it via `pip install pyrealsense2`."
            ) from exc
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.target_fps)
        cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.target_fps)
        profile = pipeline.start(cfg)
        if self.disable_emitter:
            try:
                sensor = profile.get_device().first_depth_sensor()
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, 0.0)
            except Exception:
                pass
        ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intr1 = ir1.get_intrinsics()
        intr2 = ir2.get_intrinsics()
        extr = ir1.get_extrinsics_to(ir2)
        K1 = np.array([[intr1.fx, 0.0, intr1.ppx], [0.0, intr1.fy, intr1.ppy], [0.0, 0.0, 1.0]])
        K2 = np.array([[intr2.fx, 0.0, intr2.ppx], [0.0, intr2.fy, intr2.ppy], [0.0, 0.0, 1.0]])
        baseline = float(np.linalg.norm(extr.translation))
        P1 = np.hstack([K1, np.zeros((3, 1))])
        P2 = np.hstack([K2, np.array([[-intr2.fx * baseline], [0.0], [0.0]])])
        self._calib = StereoCalibration(
            K_left=K1,
            K_right=K2,
            R=np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3),
            T=np.asarray(extr.translation, dtype=np.float64).reshape(3),
            P1=P1,
            P2=P2,
            baseline=baseline,
            image_size=(self.width, self.height),
            is_rectified=True,
        )
        self._pipeline = pipeline
        self._frame_id = 0
        self._t0_ms = None

    def read(self) -> StereoFrame | None:
        if self._pipeline is None:
            self.open()
        frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)
        if not ir1 or not ir2:
            return None
        ts_ms = float(ir1.get_timestamp())
        if self._t0_ms is None:
            self._t0_ms = ts_ms
        fid = self._frame_id
        self._frame_id += 1
        return StereoFrame(
            frame_id=fid,
            timestamp_s=(ts_ms - self._t0_ms) * 1e-3,
            left=np.asanyarray(ir1.get_data()).copy(),
            right=np.asanyarray(ir2.get_data()).copy(),
            source_fps=float(self.target_fps),
        )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class DepthAIStereoSource(FrameSource):
    """Optional Luxonis OAK-D DepthAI stereo source (requires depthai)."""

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 400,
        target_fps: float = 30.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target_fps = float(target_fps)
        self._device: Any = None
        self._q_left: Any = None
        self._q_right: Any = None
        self._calib: StereoCalibration | None = None
        self._frame_id = 0
        self._t0_s: float | None = None

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        return "depthai:oak_stereo"

    @property
    def fps(self) -> float | None:
        return self.target_fps

    def open(self) -> None:
        try:
            import depthai as dai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "DepthAIStereoSource requires the optional 'depthai' package. "
                "Install it via `pip install depthai`."
            ) from exc
        pipeline = dai.Pipeline()
        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_l.setFps(self.target_fps)
        mono_r.setFps(self.target_fps)
        xout_l = pipeline.create(dai.node.XLinkOut)
        xout_r = pipeline.create(dai.node.XLinkOut)
        xout_l.setStreamName("left")
        xout_r.setStreamName("right")
        mono_l.out.link(xout_l.input)
        mono_r.out.link(xout_r.input)
        device = dai.Device(pipeline)
        calib_data = device.readCalibration()
        K_l = np.asarray(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, self.width, self.height), dtype=np.float64)
        K_r = np.asarray(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, self.width, self.height), dtype=np.float64)
        d_l = np.asarray(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.LEFT), dtype=np.float64)
        d_r = np.asarray(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT), dtype=np.float64)
        extr = np.asarray(calib_data.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT), dtype=np.float64)
        R = extr[:3, :3]
        T = extr[:3, 3] * 1e-2  # DepthAI extrinsics translation is in cm
        self._calib = StereoCalibration(
            K_left=K_l,
            K_right=K_r,
            dist_left=d_l,
            dist_right=d_r,
            R=R,
            T=T,
            image_size=(self.width, self.height),
            is_rectified=False,
        )
        self._calib.ensure_rectification()
        self._device = device
        self._q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        self._q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)
        self._frame_id = 0
        self._t0_s = None

    def read(self) -> StereoFrame | None:
        if self._device is None:
            self.open()
        in_l = self._q_left.get()
        in_r = self._q_right.get()
        ts_s = float(in_l.getTimestamp().total_seconds())
        if self._t0_s is None:
            self._t0_s = ts_s
        fid = self._frame_id
        self._frame_id += 1
        return StereoFrame(
            frame_id=fid,
            timestamp_s=ts_s - self._t0_s,
            left=in_l.getCvFrame(),
            right=in_r.getCvFrame(),
            source_fps=self.target_fps,
        )

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
            self._q_left = None
            self._q_right = None
