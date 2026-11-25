"""Real-time Visual-SLAM orchestration engine.

Wires:
    CaptureThread -> FrameMailbox -> TrackingThread (SharedSlam)
                                            <-> LocalMappingWorker (async BA)
                                            <-> LiveLoopWorker (async PGO)
"""

from __future__ import annotations

import base64
from collections import deque
from pathlib import Path
import threading
import time
from typing import Any

import cv2 as cv
import numpy as np

from gpu_kernels.frontend_gpu import GpuFrontendAccelerator
from kitti import save_poses_txt
from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam

from .frame_source import FrameSource, StereoCalibration, StereoFrame
from .mailbox import FrameMailbox


def _encode_f32_b64(points_nx3: np.ndarray) -> str:
    if len(points_nx3) == 0:
        return ""
    arr = np.ascontiguousarray(points_nx3, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


class RealtimeSlamEngine:
    """Thread-safe real-time Visual-SLAM engine with rolling telemetry and live map deltas."""

    def __init__(
        self,
        source: FrameSource | None = None,
        *,
        config: MappingConfig | None = None,
        performance: PerformanceConfig | None = None,
        drop_frames: bool = True,
        mailbox_maxsize: int = 2,
        device: str = "auto",
        async_local_ba: bool = True,
        jpeg_quality: int = 75,
        max_points_per_keyframe: int = 300,
    ) -> None:
        self._source: FrameSource | None = source
        self._base_config = config
        self._performance = performance or PerformanceConfig(cpu_optimizations=True)
        self._drop_frames = bool(drop_frames)
        self._mailbox_maxsize = int(mailbox_maxsize)
        self._async_local_ba = bool(async_local_ba)
        self._jpeg_quality = int(np.clip(jpeg_quality, 30, 95))
        self._max_points_per_kf = int(max_points_per_keyframe)

        self.gpu = GpuFrontendAccelerator(device=device)
        self._mailbox = FrameMailbox(
            mode="latest" if self._drop_frames else "fifo",
            maxsize=self._mailbox_maxsize,
        )

        self._ctrl_lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = not paused
        self._stop_event = threading.Event()
        self._running = False
        self._paused = False
        self._playback_rate = 1.0

        self._slam: SharedSlam | None = None
        self._capture_thread: threading.Thread | None = None
        self._tracking_thread: threading.Thread | None = None

        # Telemetry & live visualization state (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._frame_counter = 0
        self._last_source_frame_id = -1
        self._last_timestamp_s = 0.0
        self._last_pose = np.eye(4, dtype=np.float64)
        self._last_tracking_state = "INIT"
        self._last_status = "initializing"
        self._last_inliers = 0
        self._last_matches = 0
        self._last_reproj_err: float | None = None
        self._last_active_landmarks = 0
        self._last_keyframes = 0
        self._last_pose_revision = 0
        self._last_map_revision = 0
        self._last_verified_loops = 0
        self._source_dropped_frames = 0
        self._capture_timestamps: deque[float] = deque(maxlen=60)
        self._track_timestamps: deque[float] = deque(maxlen=60)
        self._track_latencies_ms: deque[float] = deque(maxlen=256)
        self._loop_v2a_ms: deque[float] = deque(maxlen=64)
        self._seen_loop_events = 0
        self._loop_edges: list[dict[str, Any]] = []
        self._trajectory_xyz: list[list[float]] = []

        # Video frame slot (lazy JPEG encoding: stores raw gray + feature overlay info)
        self._video_lock = threading.Lock()
        self._video_seq = 0
        self._video_raw: tuple[np.ndarray, list[list[float]], list[bool], dict[str, Any]] | None = None
        self._video_jpeg_cache: tuple[int, bytes] | None = None
        self._video_subscribers = 0

        self._last_error: str | None = None
        self._finished = False

    @property
    def slam(self) -> SharedSlam | None:
        return self._slam

    @property
    def running(self) -> bool:
        with self._ctrl_lock:
            return self._running

    @property
    def paused(self) -> bool:
        with self._ctrl_lock:
            return self._paused

    @property
    def finished(self) -> bool:
        with self._state_lock:
            return self._finished

    def _build_slam(self, calib: StereoCalibration) -> SharedSlam:
        stereo_cam = calib.make_stereo_camera() if calib.is_stereo else None
        K = calib.rectified_intrinsics()
        if self._base_config is not None:
            cfg = self._base_config
        else:
            cfg = MappingConfig.from_preset(
                "improved_stereo",
                stereo=stereo_cam is not None,
                async_local_ba=self._async_local_ba,
            )
        perf = self._performance
        if self.gpu.uses_gpu and perf.matching_backend == "cpu":
            perf = PerformanceConfig(
                matching_backend="auto",
                retrieval=perf.retrieval,
                profile=perf.profile,
                cpu_optimizations=perf.cpu_optimizations,
            )
        return SharedSlam(K, stereo=stereo_cam, config=cfg, performance=perf)

    def _reset_metrics_locked(self) -> None:
        self._frame_counter = 0
        self._last_source_frame_id = -1
        self._last_timestamp_s = 0.0
        self._last_pose = np.eye(4, dtype=np.float64)
        self._last_tracking_state = "INIT"
        self._last_status = "initializing"
        self._last_inliers = 0
        self._last_matches = 0
        self._last_reproj_err = None
        self._last_active_landmarks = 0
        self._last_keyframes = 0
        self._last_pose_revision = 0
        self._last_map_revision = 0
        self._last_verified_loops = 0
        self._source_dropped_frames = 0
        self._capture_timestamps.clear()
        self._track_timestamps.clear()
        self._track_latencies_ms.clear()
        self._loop_v2a_ms.clear()
        self._seen_loop_events = 0
        self._loop_edges.clear()
        self._trajectory_xyz.clear()
        self._last_error = None
        self._finished = False

    def start(self) -> None:
        with self._ctrl_lock:
            if self._running:
                if self._paused:
                    self.resume()
                return
            if self._source is None:
                raise RuntimeError("Cannot start RealtimeSlamEngine without a FrameSource")
            self._source.open()
            calib = self._source.calibration
            if calib is None:
                raise RuntimeError("FrameSource must provide a StereoCalibration after open()")
            if self._slam is None:
                self._slam = self._build_slam(calib)
            self._mailbox = FrameMailbox(
                mode="latest" if self._drop_frames else "fifo",
                maxsize=self._mailbox_maxsize,
            )
            self._stop_event.clear()
            self._pause_event.set()
            self._paused = False
            self._running = True
            with self._state_lock:
                self._finished = False
                self._last_error = None

            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="slam-capture", daemon=True
            )
            self._tracking_thread = threading.Thread(
                target=self._tracking_loop, name="slam-tracking", daemon=True
            )
            self._capture_thread.start()
            self._tracking_thread.start()

    def pause(self) -> None:
        with self._ctrl_lock:
            if not self._running:
                return
            self._paused = True
            self._pause_event.clear()

    def resume(self) -> None:
        with self._ctrl_lock:
            if not self._running:
                return
            self._paused = False
            self._pause_event.set()

    def stop(self, *, finish: bool = False, timeout: float = 5.0) -> None:
        with self._ctrl_lock:
            self._stop_event.set()
            self._pause_event.set()
            self._mailbox.close()
            cap_t = self._capture_thread
            trk_t = self._tracking_thread
            self._capture_thread = None
            self._tracking_thread = None
            self._running = False
            self._paused = False

        if cap_t is not None and cap_t.is_alive():
            cap_t.join(timeout=timeout)
        if trk_t is not None and trk_t.is_alive():
            trk_t.join(timeout=timeout)

        with self._ctrl_lock:
            if self._slam is not None:
                try:
                    self._slam.close(finish=finish)
                except Exception:
                    pass
            if self._source is not None:
                try:
                    self._source.close()
                except Exception:
                    pass

    def reset_map(self) -> None:
        with self._ctrl_lock:
            was_running = self._running
            was_paused = self._paused
            source = self._source
        if was_running:
            self.stop(finish=False)
        with self._ctrl_lock:
            self._slam = None
            with self._state_lock:
                self._reset_metrics_locked()
            with self._video_lock:
                self._video_raw = None
                self._video_jpeg_cache = None
            self._source = source
        if was_running and self._source is not None:
            self.start()
            if was_paused:
                self.pause()

    def switch_source(self, new_source: FrameSource, *, auto_start: bool = True) -> None:
        self.stop(finish=False)
        with self._ctrl_lock:
            self._slam = None
            self._source = new_source
            with self._state_lock:
                self._reset_metrics_locked()
            with self._video_lock:
                self._video_raw = None
                self._video_jpeg_cache = None
        if auto_start:
            self.start()

    def set_playback_rate(self, rate: float) -> None:
        r = max(0.0, float(rate))
        with self._ctrl_lock:
            self._playback_rate = r
            if self._source is not None:
                self._source.set_playback_rate(r)

    def set_drop_frames(self, drop_frames: bool) -> None:
        with self._ctrl_lock:
            self._drop_frames = bool(drop_frames)
            if self._source is not None and hasattr(self._source, "drop_stale"):
                setattr(self._source, "drop_stale", self._drop_frames)

    def save_trajectory(self, path: str | Path) -> Path:
        target = Path(path)
        with self._ctrl_lock:
            if self._slam is None:
                raise RuntimeError("No active SLAM session to save trajectory from")
            with self._slam.map.lock:
                poses = [p.copy() for p in self._slam.map.poses]
        if not poses:
            raise ValueError("Cannot save an empty trajectory")
        save_poses_txt(target, poses)
        return target

    def wait_finished(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else (time.perf_counter() + float(timeout))
        while True:
            if self.finished or (not self.running and self._tracking_thread is None):
                return True
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            time.sleep(0.02)

    def _capture_loop(self) -> None:
        source = self._source
        if source is None:
            self._mailbox.close()
            return
        try:
            while not self._stop_event.is_set():
                if not self._pause_event.is_set():
                    time.sleep(0.02)
                    continue
                frame = source.read()
                if frame is None:
                    break
                now = time.perf_counter()
                with self._state_lock:
                    self._capture_timestamps.append(now)
                    self._source_dropped_frames += int(frame.dropped_before)
                while not self._stop_event.is_set():
                    if self._mailbox.put(frame, timeout=0.25):
                        break
                    if self._mailbox.closed:
                        break
        except Exception as exc:
            with self._state_lock:
                self._last_error = f"capture: {type(exc).__name__}: {exc}"
        finally:
            self._mailbox.close()

    def _tracking_loop(self) -> None:
        slam = self._slam
        if slam is None:
            return
        from frontend_pipeline import (
            ExtractedFrame,
            _clone_half_sgbm,
            _clone_sgbm,
            _clone_sift,
            extract_frame_owned,
        )

        from concurrent.futures import ThreadPoolExecutor

        use_lazy = (
            slam.stereo is not None
            and bool(slam.performance.cpu_optimizations)
            and bool(getattr(slam.config, "use_grid_bucketing", False))
        )
        lazy_detector = _clone_sift(slam.config, True) if use_lazy else None
        lazy_sgbm = _clone_sgbm(slam.stereo.stereo) if use_lazy and slam.stereo is not None else None
        lazy_half_sgbm = (
            _clone_half_sgbm(lazy_sgbm)
            if (use_lazy and getattr(slam.config, "use_half_res_stereo", False))
            else None
        )
        sgbm_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt-sgbm")
            if use_lazy
            else None
        )
        lk_levels = 4 if getattr(slam.config, "use_projective_tracking", False) else 3

        def _make_lazy(idx: int, left: np.ndarray, right: np.ndarray | None) -> ExtractedFrame:
            gray = cv.cvtColor(left, cv.COLOR_BGR2GRAY) if left.ndim == 3 else left
            right_gray = (
                cv.cvtColor(right, cv.COLOR_BGR2GRAY)
                if (right is not None and right.ndim == 3)
                else right
            )
            image_size = (int(gray.shape[1]), int(gray.shape[0]))
            lk_pyr = None
            try:
                from native_accel import build_lk_pyramid_avx512

                lk_pyr = build_lk_pyramid_avx512(gray, max_level=lk_levels)
            except Exception:
                lk_pyr = None
            left_f32 = gray if gray.dtype == np.uint8 else np.asarray(gray, np.float32)
            right_f32 = (
                right_gray
                if (right_gray is None or right_gray.dtype == np.uint8)
                else np.asarray(right_gray, np.float32)
            )

            def _materialize() -> ExtractedFrame:
                return extract_frame_owned(
                    idx,
                    gray,
                    right_gray,
                    config=slam.config,
                    inverse_K=slam.inverse_K,
                    stereo_Q=slam.stereo.Q if slam.stereo is not None else None,
                    stereo_search_config=slam.stereo_search_config,
                    detector=lazy_detector,
                    sgbm=lazy_sgbm,
                    half_sgbm=lazy_half_sgbm,
                    sgbm_executor=sgbm_executor,
                    precomputed_lk_pyramid=lk_pyr,
                )

            return ExtractedFrame(
                index=int(idx),
                image_size=image_size,
                gray=gray,
                left_gray_f32=left_f32,
                right_gray_f32=right_f32,
                pixels=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
                octaves=np.empty(0, dtype=np.int16),
                points=np.empty((0, 3), dtype=np.float64),
                right_u=np.empty(0, dtype=np.float64),
                disparity=None,
                supported_extraction=None,
                stereo_depth_verification={},
                lk_pyramid=lk_pyr,
                lazy_extractor=_materialize,
            )

        try:
            while not self._stop_event.is_set():
                if not self._pause_event.is_set():
                    time.sleep(0.02)
                    continue
                frame = self._mailbox.get(timeout=0.1)
                if frame is None:
                    if self._mailbox.closed:
                        break
                    continue
                t0 = time.perf_counter()
                idx = self._frame_counter
                if use_lazy and frame.right is not None:
                    ext = _make_lazy(idx, frame.left, frame.right)
                    pose, info = slam.process_frame(
                        idx,
                        extracted=ext,
                        timestamp_s=frame.timestamp_s,
                    )
                else:
                    pose, info = slam.process_frame(
                        idx,
                        frame.left,
                        frame.right,
                        timestamp_s=frame.timestamp_s,
                    )
                track_ms = (time.perf_counter() - t0) * 1000.0
                now = time.perf_counter()
                self._record_tracking_step(slam, frame, pose, info, track_ms, now)
                self._frame_counter += 1
        except Exception as exc:
            with self._state_lock:
                self._last_error = f"tracking: {type(exc).__name__}: {exc}"
        finally:
            if sgbm_executor is not None:
                sgbm_executor.shutdown(wait=False)
            with self._state_lock:
                self._finished = True
            with self._ctrl_lock:
                self._running = False

    def _record_tracking_step(
        self,
        slam: SharedSlam,
        frame: StereoFrame,
        pose: np.ndarray,
        info: dict[str, Any],
        track_ms: float,
        now: float,
    ) -> None:
        # Drain newly applied loop closure events from LiveLoopWorker
        loop_events = slam.loop_worker.events
        new_loop_edges: list[dict[str, Any]] = []
        new_v2a: list[float] = []
        if len(loop_events) > self._seen_loop_events:
            for ev in loop_events[self._seen_loop_events :]:
                if ev.get("type") == "loop_applied":
                    for sec in ev.get("verification_to_apply_s") or ():
                        new_v2a.append(float(sec) * 1000.0)
            self._seen_loop_events = len(loop_events)
            verified_pairs = list(slam.loop_worker.verified.keys())
            existing_pairs = {(e["from_kf"], e["to_kf"]) for e in self._loop_edges}
            for i, j in verified_pairs:
                if (int(i), int(j)) not in existing_pairs:
                    new_loop_edges.append(
                        {"from_kf": int(i), "to_kf": int(j), "timestamp_s": float(frame.timestamp_s)}
                    )

        with slam.map.lock:
            n_kfs = len(slam.map.keyframes)
            n_lms = len(slam.map.landmarks)
            pose_rev = int(slam.map.pose_revision)
            map_rev = int(slam.map.revision)

        with self._state_lock:
            self._last_source_frame_id = int(frame.frame_id)
            self._last_timestamp_s = float(frame.timestamp_s)
            self._last_pose = pose.copy()
            self._last_tracking_state = str(info.get("tracking_state", slam.tracking_state))
            self._last_status = str(info.get("state", "tracking"))
            self._last_inliers = int(info.get("num_inliers", 0) or 0)
            self._last_matches = int(info.get("num_matches", 0) or 0)
            reproj = info.get("reprojection_error")
            self._last_reproj_err = float(reproj) if reproj is not None and np.isfinite(reproj) else None
            self._last_active_landmarks = n_lms
            self._last_keyframes = n_kfs
            self._last_pose_revision = pose_rev
            self._last_map_revision = map_rev
            self._last_verified_loops = len(slam.loop_worker.verified)
            self._track_timestamps.append(now)
            self._track_latencies_ms.append(track_ms)
            for ms in new_v2a:
                self._loop_v2a_ms.append(ms)
            if new_loop_edges:
                self._loop_edges.extend(new_loop_edges)
            self._trajectory_xyz.append([float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])])

        # Store lightweight raw frame + overlay data for lazy JPEG encoding
        feat_pts = info.get("feature_points") or []
        inlier_mask = info.get("inlier_mask") or []
        hud = {
            "frame_id": int(frame.frame_id),
            "state": self._last_tracking_state,
            "inliers": self._last_inliers,
            "track_ms": track_ms,
        }
        with self._video_lock:
            self._video_seq += 1
            self._video_raw = (frame.left, feat_pts, inlier_mask, hud)
            self._video_jpeg_cache = None

    @staticmethod
    def _rolling_fps(timestamps: deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        span = timestamps[-1] - timestamps[0]
        if span <= 1e-6:
            return 0.0
        return float((len(timestamps) - 1) / span)

    def get_telemetry(self) -> dict[str, Any]:
        mb_stats = self._mailbox.stats()
        ba_p50_ms = 0.0
        ba_stats: dict[str, Any] = {}
        slam = self._slam
        if slam is not None and getattr(slam, "_local_mapping_worker", None) is not None:
            ba_stats = slam._local_mapping_worker.stats()
            ba_p50_ms = float(ba_stats.get("ba_p50_ms", 0.0))

        with self._state_lock:
            track_arr = list(self._track_latencies_ms)
            v2a_arr = list(self._loop_v2a_ms)
            t_p50 = float(np.percentile(track_arr, 50)) if track_arr else 0.0
            t_p95 = float(np.percentile(track_arr, 95)) if track_arr else 0.0
            v2a_p95 = float(np.percentile(v2a_arr, 95)) if v2a_arr else 0.0
            dropped = int(mb_stats["dropped_frames"]) + int(self._source_dropped_frames)
            return {
                "type": "telemetry",
                "running": self._running,
                "paused": self._paused,
                "finished": self._finished,
                "source": self._source.source_name if self._source is not None else "none",
                "total_frames": self._source.total_frames if self._source is not None else None,
                "playback_rate": self._playback_rate,
                "drop_frames": self._drop_frames,
                "frame_id": self._last_source_frame_id,
                "processed_frames": self._frame_counter,
                "timestamp_s": round(self._last_timestamp_s, 4),
                "pose_T_wc": self._last_pose.tolist(),
                "capture_fps": round(self._rolling_fps(self._capture_timestamps), 2),
                "tracking_fps": round(self._rolling_fps(self._track_timestamps), 2),
                "track_p50_ms": round(t_p50, 2),
                "track_p95_ms": round(t_p95, 2),
                "ba_p50_ms": round(ba_p50_ms, 2),
                "loop_v2a_p95_ms": round(v2a_p95, 2),
                "dropped_frames": dropped,
                "queue_age_ms": round(float(mb_stats["queue_age_ms"]), 2),
                "tracking_state": self._last_tracking_state,
                "state": self._last_status,
                "inliers": self._last_inliers,
                "matches": self._last_matches,
                "reprojection_error": (
                    round(self._last_reproj_err, 3) if self._last_reproj_err is not None else None
                ),
                "active_landmarks": self._last_active_landmarks,
                "keyframes": self._last_keyframes,
                "verified_loops_applied": self._last_verified_loops,
                "pose_revision": self._last_pose_revision,
                "map_revision": self._last_map_revision,
                "backend_device": self.gpu.backend_label,
                "device_info": self.gpu.device_info.to_dict(),
                "ba_stats": ba_stats,
                "last_error": self._last_error,
            }

    def _serialize_keyframe_chunk(self, state: Any, kf_id: int) -> dict[str, Any]:
        kf = state.keyframes[kf_id]
        R_T = kf.pose[:3, :3].T
        t = kf.pose[:3, 3]
        local_pts = []
        if hasattr(state, "index") and len(state.index) == len(state.landmarks):
            lm_ids = state.index._anchor_landmarks.get(kf_id, ())
            if lm_ids:
                ids_list = list(lm_ids)
                if len(ids_list) > self._max_points_per_kf:
                    step = max(1, len(ids_list) // self._max_points_per_kf)
                    ids_list = ids_list[::step][: self._max_points_per_kf]
                slots = [state.index._id_to_slot[i] for i in ids_list if i in state.index._id_to_slot]
                if slots:
                    w_pts = state.index._positions[slots]
                    local_arr = (w_pts - t) @ R_T.T
                    return {
                        "kf_id": int(kf_id),
                        "frame_id": int(kf.frame),
                        "pose_T_wc": kf.pose.tolist(),
                        "point_count": int(len(local_arr)),
                        "points_f32_b64": _encode_f32_b64(local_arr),
                    }
        for lm in state.landmarks.values():
            if lm.anchor == kf_id:
                local_pts.append(R_T @ (lm.position - t))
                if len(local_pts) >= self._max_points_per_kf:
                    break
        arr = np.asarray(local_pts, dtype=np.float32).reshape(-1, 3) if local_pts else np.empty((0, 3), dtype=np.float32)
        return {
            "kf_id": int(kf_id),
            "frame_id": int(kf.frame),
            "pose_T_wc": kf.pose.tolist(),
            "point_count": int(len(arr)),
            "points_f32_b64": _encode_f32_b64(arr),
        }

    def get_map_snapshot(self) -> dict[str, Any]:
        """Return full map state (keyframes with anchor-relative Float32 point chunks, trajectory, loops)."""
        slam = self._slam
        if slam is None:
            return {
                "type": "map_snapshot",
                "pose_revision": 0,
                "map_revision": 0,
                "keyframes": [],
                "trajectory": [],
                "loop_edges": [],
                "telemetry": self.get_telemetry(),
            }
        with slam.map.lock:
            kf_ids = sorted(slam.map.keyframes)
            chunks = [self._serialize_keyframe_chunk(slam.map, k) for k in kf_ids]
            traj = [[float(p[0, 3]), float(p[1, 3]), float(p[2, 3])] for p in slam.map.poses]
            pose_rev = int(slam.map.pose_revision)
            map_rev = int(slam.map.revision)
        with self._state_lock:
            loops = list(self._loop_edges)
        return {
            "type": "map_snapshot",
            "pose_revision": pose_rev,
            "map_revision": map_rev,
            "max_kf_id": kf_ids[-1] if kf_ids else -1,
            "keyframes": chunks,
            "trajectory": traj,
            "loop_edges": loops,
            "telemetry": self.get_telemetry(),
        }

    def get_map_delta(
        self,
        *,
        since_kf_id: int,
        since_pose_revision: int,
        since_loop_count: int,
        since_traj_len: int,
    ) -> tuple[dict[str, Any] | None, int, int, int, int]:
        """Compute compact incremental map update (new KF chunks + O(K) pose batch on BA/loop)."""
        slam = self._slam
        if slam is None:
            return None, since_kf_id, since_pose_revision, since_loop_count, since_traj_len
        with slam.map.lock:
            kf_ids = sorted(slam.map.keyframes)
            max_kf = kf_ids[-1] if kf_ids else -1
            pose_rev = int(slam.map.pose_revision)
            total_poses = len(slam.map.poses)
            was_reset = (max_kf < since_kf_id) or (total_poses < since_traj_len)
        if was_reset:
            snap = self.get_map_snapshot()
            return (
                snap,
                int(snap.get("max_kf_id", -1)),
                int(snap.get("pose_revision", 0)),
                len(snap.get("loop_edges") or ()),
                len(snap.get("trajectory") or ()),
            )
        with slam.map.lock:
            kf_ids = sorted(slam.map.keyframes)
            max_kf = kf_ids[-1] if kf_ids else -1
            pose_rev = int(slam.map.pose_revision)
            new_kf_ids = [k for k in kf_ids if k > since_kf_id]
            new_keyframes = [self._serialize_keyframe_chunk(slam.map, k) for k in new_kf_ids]
            kf_pose_batch: dict[str, list[list[float]]] | None = None
            traj_replace: list[list[float]] | None = None
            traj_append: list[list[float]] | None = None
            total_poses = len(slam.map.poses)

            if pose_rev != since_pose_revision:
                kf_pose_batch = {
                    str(k): slam.map.keyframes[k].pose.tolist()
                    for k in kf_ids
                }
                traj_replace = [
                    [float(p[0, 3]), float(p[1, 3]), float(p[2, 3])]
                    for p in slam.map.poses
                ]
            elif total_poses > since_traj_len:
                traj_append = [
                    [float(p[0, 3]), float(p[1, 3]), float(p[2, 3])]
                    for p in slam.map.poses[since_traj_len:]
                ]
        with self._state_lock:
            loops = self._loop_edges[since_loop_count:]
            total_loops = len(self._loop_edges)

        if not new_keyframes and kf_pose_batch is None and not loops and not traj_append:
            return None, max_kf, pose_rev, total_loops, total_poses

        payload: dict[str, Any] = {
            "type": "map_delta",
            "pose_revision": pose_rev,
            "new_keyframes": new_keyframes,
            "loop_edges": loops,
        }
        if kf_pose_batch is not None:
            payload["kf_pose_batch"] = kf_pose_batch
        if traj_replace is not None:
            payload["trajectory"] = traj_replace
        elif traj_append is not None:
            payload["trajectory_append"] = traj_append
        return payload, max_kf, pose_rev, total_loops, total_poses

    def get_latest_video_jpeg(self, last_seq: int = -1) -> tuple[int, bytes | None]:
        """Lazily render & JPEG-encode the latest annotated camera frame if seq > last_seq."""
        with self._video_lock:
            seq = self._video_seq
            if seq == 0 or seq == last_seq or self._video_raw is None:
                return seq, None
            if self._video_jpeg_cache is not None and self._video_jpeg_cache[0] == seq:
                return seq, self._video_jpeg_cache[1]
            img, feat_pts, inlier_mask, hud = self._video_raw

        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        n_draw = min(len(feat_pts), 400)
        for i in range(n_draw):
            pt = feat_pts[i]
            x, y = int(round(pt[0])), int(round(pt[1]))
            is_inlier = i < len(inlier_mask) and bool(inlier_mask[i])
            color = (60, 220, 120) if is_inlier else (70, 90, 230)
            cv.circle(vis, (x, y), 2, color, -1, lineType=cv.LINE_AA)

        state_str = hud.get("state", "INIT")
        banner = f"{state_str} | F#{hud.get('frame_id', 0)} | INL {hud.get('inliers', 0)} | {hud.get('track_ms', 0.0):.1f} ms"
        cv.rectangle(vis, (8, 8), (380, 34), (16, 20, 28), -1)
        cv.putText(vis, banner, (14, 26), cv.FONT_HERSHEY_SIMPLEX, 0.5, (235, 242, 250), 1, cv.LINE_AA)

        ok, buf = cv.imencode(".jpg", vis, [int(cv.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return seq, None
        jpeg_bytes = buf.tobytes()
        with self._video_lock:
            if self._video_seq == seq:
                self._video_jpeg_cache = (seq, jpeg_bytes)
        return seq, jpeg_bytes
