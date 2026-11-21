"""Frame-owned frontend extraction and bounded producer-consumer pipelining.

Extraction produces self-contained ``ExtractedFrame`` objects using worker-private
OpenCV ``SIFT`` and ``StereoSGBM`` instances so background threads never mutate
``SharedSlam`` tracking state or share stateful OpenCV objects across threads.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable

import cv2 as cv
import numpy as np

from frontend_features import extract_grid_bucketed_sift, unpack_sift_octave
from stereo_depth import verify_stereo_depth_candidates


@dataclass(frozen=True)
class ExtractedFrame:
    """Immutable, frame-owned output of frontend feature and stereo extraction."""

    index: int
    image_size: tuple[int, int]
    gray: np.ndarray
    left_gray_f32: np.ndarray | None
    right_gray_f32: np.ndarray | None
    pixels: np.ndarray
    descriptors: np.ndarray
    octaves: np.ndarray
    points: np.ndarray
    right_u: np.ndarray
    disparity: np.ndarray | None
    supported_extraction: tuple[np.ndarray, np.ndarray] | None
    stereo_depth_verification: dict[str, int]
    io_elapsed_s: float = 0.0
    extract_elapsed_s: float = 0.0


def _clone_sgbm(src: Any) -> Any:
    """Create a thread-private StereoSGBM instance with identical parameters."""
    if src is None:
        return None
    if isinstance(src, cv.StereoSGBM):
        return cv.StereoSGBM_create(
            minDisparity=src.getMinDisparity(),
            numDisparities=src.getNumDisparities(),
            blockSize=src.getBlockSize(),
            P1=src.getP1(),
            P2=src.getP2(),
            disp12MaxDiff=src.getDisp12MaxDiff(),
            uniquenessRatio=src.getUniquenessRatio(),
            speckleWindowSize=src.getSpeckleWindowSize(),
            speckleRange=src.getSpeckleRange(),
            mode=src.getMode(),
        )
    raise TypeError(f"Unsupported stereo matcher type for worker cloning: {type(src)!r}")


def _clone_sift(config: Any, has_stereo: bool) -> cv.SIFT:
    """Create a thread-private SIFT detector matching SharedSlam's active detector."""
    contrast = config.stereo_feature_contrast_threshold if has_stereo else 0.04
    if getattr(config, "use_grid_bucketing", False):
        nfeatures = max(int(config.features * 1.4), 2100)
    else:
        nfeatures = int(config.features)
    return cv.SIFT_create(nfeatures=nfeatures, contrastThreshold=float(contrast))


def _measure_stereo_pixels_owned(
    pixels: np.ndarray,
    *,
    disparity_map: np.ndarray | None,
    left_gray_f32: np.ndarray | None,
    right_gray_f32: np.ndarray | None,
    Q: np.ndarray,
    inverse_K: np.ndarray,
    stereo_depth_policy: str,
    stereo_search_config: Any,
    capture_supported: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None, dict[str, int]]:
    """Pure equivalent of SharedSlam._measure_stereo_pixels without mutating self."""
    points = np.full((len(pixels), 3), np.nan)
    right_u = np.full(len(pixels), np.nan)
    supported_extraction = None
    verification_counts: dict[str, int] = {}

    if not len(pixels):
        return points, right_u, supported_extraction, verification_counts

    finite = np.isfinite(pixels).all(axis=1)
    if stereo_depth_policy == "verified_all":
        if left_gray_f32 is None or right_gray_f32 is None:
            verification_counts["missing_images"] = int(len(pixels))
            if capture_supported:
                supported_extraction = (points.copy(), right_u.copy())
            return points, right_u, supported_extraction, verification_counts
        if (
            np.ndim(left_gray_f32) != 2
            or np.ndim(right_gray_f32) != 2
            or np.shape(left_gray_f32) != np.shape(right_gray_f32)
        ):
            verification_counts["invalid_images"] = int(len(pixels))
            if capture_supported:
                supported_extraction = (points.copy(), right_u.copy())
            return points, right_u, supported_extraction, verification_counts
        height, width = left_gray_f32.shape
        inside = (
            finite
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= height - 1)
        )
        ids = np.flatnonzero(inside)
        raw_q = np.asarray(Q)
        if (
            np.iscomplexobj(raw_q)
            or raw_q.shape != (4, 4)
            or not np.issubdtype(raw_q.dtype, np.number)
            or not np.isfinite(raw_q).all()
        ):
            verification_counts["invalid_calibration"] = int(len(ids))
            if capture_supported:
                supported_extraction = (points.copy(), right_u.copy())
            return points, right_u, supported_extraction, verification_counts
        q = raw_q.astype(float, copy=True)
        if q[3, 2] <= 0 or q[2, 3] <= 0 or not np.isfinite(inverse_K).all():
            verification_counts["invalid_calibration"] = int(len(ids))
            if capture_supported:
                supported_extraction = (points.copy(), right_u.copy())
            return points, right_u, supported_extraction, verification_counts
        measured = np.full(len(ids), np.nan)
        counters: dict[str, int] = {"candidates": int(len(ids)), "verified": 0}
        if len(ids):
            q32, q23 = float(q[3, 2]), float(q[2, 3])
            if np.isfinite([q32, q23, q[3, 3]]).all() and q32 > 0 and q23 > 0:
                bf, offset = q23 / q32, -float(q[3, 3]) / q32
                bounds = (max(0.0, offset + bf / 100.0), min(96.0, offset + bf / 0.1))
                if bounds[0] < bounds[1]:
                    measured, counters = verify_stereo_depth_candidates(
                        left_gray_f32,
                        right_gray_f32,
                        np.asarray(pixels, float)[ids],
                        stereo_search_config,
                        bounds,
                    )
        for key, value in counters.items():
            verification_counts[key] = verification_counts.get(key, 0) + int(value)
        disparity = np.asarray(pixels, float)[ids, 0] - measured
        denominator = q[3, 2] * disparity + q[3, 3]
        depth = np.divide(
            q[2, 3], denominator, out=np.full(len(ids), np.nan), where=denominator > 0
        )
        valid = np.isfinite(measured) & (disparity > 0.0) & (disparity < 96.0)
        valid &= np.isfinite(depth) & (depth > 0.1) & (depth < 100.0)
        accepted = ids[valid]
        rays = np.c_[np.asarray(pixels, float)[accepted], np.ones(len(accepted))] @ inverse_K.T
        points[accepted] = rays * depth[valid, None]
        right_u[accepted] = measured[valid]
        if capture_supported:
            supported_extraction = (points.copy(), right_u.copy())
        return points, right_u, supported_extraction, verification_counts

    if disparity_map is None:
        return points, right_u, supported_extraction, verification_counts

    height, width = disparity_map.shape
    inside = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    ids = np.flatnonzero(inside)
    if not len(ids):
        return points, right_u, supported_extraction, verification_counts

    coordinates = np.floor(pixels[ids]).astype(int)
    x, y = coordinates.T
    next_x, next_y = np.minimum(x + 1, width - 1), np.minimum(y + 1, height - 1)
    fraction = pixels[ids] - coordinates
    dx, dy = fraction.T
    weights = np.c_[(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy]
    neighbors = np.c_[
        disparity_map[y, x],
        disparity_map[y, next_x],
        disparity_map[next_y, x],
        disparity_map[next_y, next_x],
    ]
    active = weights > 0
    valid_neighbors = np.isfinite(neighbors) & (neighbors > 0) & (neighbors < 96)
    spread = np.max(np.where(active, neighbors, -np.inf), axis=1) - np.min(
        np.where(active, neighbors, np.inf), axis=1
    )
    supported = np.all(~active | valid_neighbors, axis=1) & (spread <= 2.0)
    disparity = np.sum(np.where(active, neighbors, 0.0) * weights, axis=1)
    denominator = Q[3, 2] * disparity + Q[3, 3]
    depth = np.divide(
        Q[2, 3],
        denominator,
        out=np.full(len(ids), np.nan),
        where=denominator > 0,
    )
    valid = supported & (disparity > 0) & (disparity < 96) & np.isfinite(depth)
    valid &= (depth > 0.1) & (depth < 100)
    ids, disparity, depth = ids[valid], disparity[valid], depth[valid]
    rays = np.c_[pixels[ids], np.ones(len(ids))] @ inverse_K.T
    points[ids] = rays * depth[:, None]
    right_u[ids] = pixels[ids, 0] - disparity
    if capture_supported:
        supported_extraction = (points.copy(), right_u.copy())

    if stereo_depth_policy == "verified_fallback":
        candidates = np.flatnonzero(inside & ~np.isfinite(points).all(axis=1))
        if len(candidates):
            nearest = np.rint(pixels[candidates]).astype(int)
            raw = disparity_map[nearest[:, 1], nearest[:, 0]]
            den = Q[3, 2] * raw + Q[3, 3]
            z = np.divide(Q[2, 3], den, out=np.full(len(raw), np.nan), where=den > 0)
            eligible = np.isfinite(raw) & (raw > 0) & (raw < 96)
            eligible &= np.isfinite(z) & (z > 0.1) & (z < 100)
            candidates = candidates[eligible]
            if len(candidates) and Q[3, 2] > 0 and Q[2, 3] > 0:
                bf, offset = Q[2, 3] / Q[3, 2], -Q[3, 3] / Q[3, 2]
                bounds = (max(0.0, offset + bf / 100), min(96.0, offset + bf / 0.1))
                if bounds[0] < bounds[1]:
                    measured, counters = verify_stereo_depth_candidates(
                        left_gray_f32,
                        right_gray_f32,
                        pixels[candidates],
                        stereo_search_config,
                        bounds,
                    )
                    for key, value in counters.items():
                        verification_counts[key] = verification_counts.get(key, 0) + int(value)
                    d = pixels[candidates, 0] - measured
                    den = Q[3, 2] * d + Q[3, 3]
                    z = np.divide(Q[2, 3], den, out=np.full(len(d), np.nan), where=den > 0)
                    good = np.isfinite(z) & (d > 0) & (d < 96) & (z > 0.1) & (z < 100)
                    accepted = candidates[good]
                    rays = np.c_[pixels[accepted], np.ones(len(accepted))] @ inverse_K.T
                    points[accepted] = rays * z[good, None]
                    right_u[accepted] = measured[good]

    return points, right_u, supported_extraction, verification_counts


def extract_frame_owned(
    index: int,
    image: np.ndarray,
    right: np.ndarray | None,
    *,
    config: Any,
    inverse_K: np.ndarray,
    stereo_Q: np.ndarray | None,
    stereo_search_config: Any,
    detector: cv.SIFT,
    sgbm: Any = None,
    io_elapsed_s: float = 0.0,
) -> ExtractedFrame:
    """Run frontend extraction purely on frame inputs without mutating shared SLAM state."""
    if image is None:
        raise ValueError("Missing image")
    t0 = time.perf_counter()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_size = (int(image.shape[1]), int(image.shape[0]))
    has_stereo = stereo_Q is not None
    need_f32_pair = has_stereo and (
        config.stereo_depth_policy in ("verified_fallback", "verified_all")
        or getattr(config, "stereo_map_depth_policy", "inherit") == "verified"
    )
    right_gray = None
    left_gray_f32 = np.asarray(gray, np.float32) if need_f32_pair else None
    right_gray_f32 = None
    if has_stereo:
        if right is None or right.shape[:2] != image.shape[:2]:
            raise ValueError("Stereo requires matching right image")
        right_gray = cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right
        if need_f32_pair:
            right_gray_f32 = np.asarray(right_gray, np.float32)

    if getattr(config, "use_grid_bucketing", False):
        bucketed = extract_grid_bucketed_sift(
            gray,
            target_features=config.features,
            grid_cols=16,
            grid_rows=6,
            contrast_threshold=config.stereo_feature_contrast_threshold,
            detector=detector,
        )
        pixels = bucketed.pixels
        desc = bucketed.descriptors
        octaves = bucketed.octaves
    else:
        keypoints, desc = detector.detectAndCompute(gray, None)
        pixels = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        desc = desc if desc is not None else np.empty((0, 128), np.float32)
        octaves = np.array(
            [unpack_sift_octave(k.octave) for k in keypoints], np.int16
        )

    points = np.full((len(pixels), 3), np.nan)
    right_u = np.full(len(pixels), np.nan)
    disparity = None
    supported_extraction = None
    verification_counts: dict[str, int] = {}

    if has_stereo:
        if sgbm is None:
            raise ValueError("Stereo extraction requires a StereoSGBM matcher instance")
        disparity = sgbm.compute(gray, right_gray).astype(np.float32) / 16.0
        capture_supported = bool(
            config.stereo_pose_arbitration or config.stereo_raw_reference_retry
        )
        points, right_u, supported_extraction, verification_counts = _measure_stereo_pixels_owned(
            pixels,
            disparity_map=disparity,
            left_gray_f32=left_gray_f32,
            right_gray_f32=right_gray_f32,
            Q=stereo_Q,
            inverse_K=inverse_K,
            stereo_depth_policy=config.stereo_depth_policy,
            stereo_search_config=stereo_search_config,
            capture_supported=capture_supported,
        )

    extract_elapsed_s = max(time.perf_counter() - t0, 0.0)
    return ExtractedFrame(
        index=int(index),
        image_size=image_size,
        gray=gray,
        left_gray_f32=left_gray_f32,
        right_gray_f32=right_gray_f32,
        pixels=pixels,
        descriptors=desc,
        octaves=octaves,
        points=points,
        right_u=right_u,
        disparity=disparity,
        supported_extraction=supported_extraction,
        stereo_depth_verification=verification_counts,
        io_elapsed_s=float(io_elapsed_s),
        extract_elapsed_s=float(extract_elapsed_s),
    )


_STOP = object()


class FrontendExtractionPipeline:
    """Bounded single-worker producer-consumer frontend extraction pipeline.

    Guarantees:
    - Thread-private OpenCV SIFT and StereoSGBM instances (no cross-thread sharing).
    - Strictly bounded in-flight memory (``buffer_size`` frames).
    - Strict FIFO in-order output verified by frame index.
    - Clean worker exception propagation and deterministic shutdown.
    """

    def __init__(self, slam: Any, *, buffer_size: int = 2) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self._config = slam.config
        self._inverse_K = slam.inverse_K.copy()
        self._stereo_Q = slam.stereo.Q.copy() if slam.stereo is not None else None
        self._stereo_search_config = slam.stereo_search_config
        self._detector = _clone_sift(slam.config, has_stereo=slam.stereo is not None)
        src_sgbm = (
            (getattr(slam, "_fast_stereo_sgbm", None) or slam.stereo.stereo)
            if slam.stereo is not None
            else None
        )
        self._sgbm = _clone_sgbm(src_sgbm)
        self._in_q: queue.Queue[Any] = queue.Queue(maxsize=int(buffer_size))
        self._out_q: queue.Queue[Any] = queue.Queue(maxsize=int(buffer_size))
        self._stop_event = threading.Event()
        self._closed = False
        self._next_submit_index = 0
        self._next_get_index = 0
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="frontend-extraction",
            daemon=True,
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._in_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                index, loader, image, right = item
                try:
                    io_dt = 0.0
                    if loader is not None:
                        t_io = time.perf_counter()
                        image, right = loader(index)
                        io_dt = max(time.perf_counter() - t_io, 0.0)
                    extracted = extract_frame_owned(
                        index,
                        image,
                        right,
                        config=self._config,
                        inverse_K=self._inverse_K,
                        stereo_Q=self._stereo_Q,
                        stereo_search_config=self._stereo_search_config,
                        detector=self._detector,
                        sgbm=self._sgbm,
                        io_elapsed_s=io_dt,
                    )
                    self._put_output((index, extracted, None))
                except BaseException as exc:
                    self._put_output((index, None, exc))
                    self._stop_event.set()
                    break
        finally:
            self._stop_event.set()

    def _put_output(self, payload: Any) -> None:
        while not self._stop_event.is_set():
            try:
                self._out_q.put(payload, timeout=0.05)
                return
            except queue.Full:
                continue
        # Best-effort delivery on stop (e.g., error notification)
        try:
            self._out_q.put_nowait(payload)
        except queue.Full:
            pass

    def submit(
        self,
        index: int,
        image: np.ndarray,
        right: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("FrontendExtractionPipeline is closed")
        if int(index) != self._next_submit_index:
            raise ValueError(
                f"Expected consecutive submit index {self._next_submit_index}, got {index}"
            )
        self._next_submit_index += 1
        while not self._stop_event.is_set():
            try:
                self._in_q.put((int(index), None, image, right), timeout=0.05)
                return
            except queue.Full:
                continue

    def submit_loader(
        self,
        index: int,
        loader: Callable[[int], tuple[np.ndarray, np.ndarray | None]],
    ) -> None:
        if self._closed:
            raise RuntimeError("FrontendExtractionPipeline is closed")
        if int(index) != self._next_submit_index:
            raise ValueError(
                f"Expected consecutive submit index {self._next_submit_index}, got {index}"
            )
        self._next_submit_index += 1
        while not self._stop_event.is_set():
            try:
                self._in_q.put((int(index), loader, None, None), timeout=0.05)
                return
            except queue.Full:
                continue

    def get(self, expected_index: int) -> ExtractedFrame:
        if int(expected_index) != self._next_get_index:
            raise ValueError(
                f"Expected in-order get index {self._next_get_index}, got {expected_index}"
            )
        while True:
            try:
                idx, extracted, exc = self._out_q.get(timeout=0.1)
                break
            except queue.Empty:
                if not self._thread.is_alive():
                    raise RuntimeError(
                        f"FrontendExtractionPipeline worker exited before frame {expected_index}"
                    )
        if exc is not None:
            self.close()
            raise exc
        if idx != int(expected_index) or extracted is None:
            self.close()
            raise RuntimeError(
                f"FrontendExtractionPipeline out-of-order result: expected {expected_index}, got {idx}"
            )
        self._next_get_index += 1
        return extracted

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        # Drain input and output queues so worker cannot stay blocked on put()
        for q in (self._in_q, self._out_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        try:
            self._in_q.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "FrontendExtractionPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
