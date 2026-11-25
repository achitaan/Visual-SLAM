"""Comprehensive tests for Phases 1, 2, 3, and 5 of Real-Time Visual-SLAM."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import threading
import time

import cv2 as cv
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from realtime.engine import RealtimeSlamEngine
from realtime.frame_source import (
    DepthAIStereoSource,
    FrameSource,
    OpenCVStereoCameraSource,
    PacedKittiSource,
    RealSenseStereoSource,
    StereoCalibration,
    StereoFrame,
    StereoRectifier,
    StereoVideoFileSource,
)
from realtime.local_mapping_worker import (
    LocalBACorrection,
    LocalBASnapshot,
    LocalMappingWorker,
    apply_local_ba_correction,
)
from realtime.mailbox import FrameMailbox
from realtime.server import create_app
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from slam_state import MapState, MappingKeyframe, Observation


def _make_textured_scene_image(w: int = 480, h: int = 320, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(35, 220, size=(h, w), dtype=np.uint8)
    img = cv.GaussianBlur(img, (3, 3), 0.9)
    for _ in range(70):
        cx = int(rng.integers(12, w - 12))
        cy = int(rng.integers(12, h - 12))
        r = int(rng.integers(3, 14))
        cv.circle(img, (cx, cy), r, int(rng.integers(10, 245)), -1)
    for _ in range(50):
        x1, y1 = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
        x2, y2 = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
        cv.line(img, (x1, y1), (x2, y2), int(rng.integers(0, 255)), 2)
    return img


def _make_synthetic_stereo_sequence(
    n_frames: int = 12,
    w: int = 360,
    h: int = 220,
    disparity_px: float = 16.0,
) -> tuple[StereoCalibration, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate a textured planar stereo sequence moving smoothly in x."""
    canvas_w = w + n_frames * 6 + int(disparity_px) + 40
    canvas = _make_textured_scene_image(canvas_w, h + 40, seed=42)
    K = np.array([[360.0, 0.0, w * 0.5], [0.0, 360.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = 0.5
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-K[0, 0] * baseline], [0.0], [0.0]])])
    calib = StereoCalibration(
        K_left=K,
        K_right=K.copy(),
        P1=P1,
        P2=P2,
        baseline=baseline,
        image_size=(w, h),
        is_rectified=True,
    )
    pairs = []
    d_int = int(round(disparity_px))
    for i in range(n_frames):
        ox = 10 + i * 3
        oy = 20
        left = np.ascontiguousarray(canvas[oy : oy + h, ox : ox + w])
        right = np.ascontiguousarray(canvas[oy : oy + h, ox + d_int : ox + d_int + w])
        pairs.append((left, right))
    return calib, pairs


class _SyntheticStereoSource(FrameSource):
    def __init__(self, calib: StereoCalibration, pairs: list[tuple[np.ndarray, np.ndarray]], fps: float = 20.0):
        self._calib = calib
        self._pairs = pairs
        self._fps = fps
        self._idx = 0
        self._open = False

    @property
    def calibration(self) -> StereoCalibration | None:
        return self._calib

    @property
    def source_name(self) -> str:
        return "synthetic:stereo"

    @property
    def total_frames(self) -> int | None:
        return len(self._pairs)

    @property
    def fps(self) -> float | None:
        return self._fps

    def open(self) -> None:
        self._idx = 0
        self._open = True

    def read(self) -> StereoFrame | None:
        if not self._open or self._idx >= len(self._pairs):
            return None
        left, right = self._pairs[self._idx]
        fid = self._idx
        self._idx += 1
        return StereoFrame(
            frame_id=fid,
            timestamp_s=fid / self._fps,
            left=left.copy(),
            right=right.copy(),
            source_fps=self._fps,
        )

    def close(self) -> None:
        self._open = False


# ---------------------------------------------------------------------------
# Phase 1: FrameMailbox tests (latest vs fifo)
# ---------------------------------------------------------------------------


def test_frame_mailbox_latest_drops_stale_without_blocking():
    now_t = [100.0]
    mb = FrameMailbox(mode="latest", maxsize=2, clock=lambda: now_t[0])
    img = np.zeros((16, 16), dtype=np.uint8)
    for i in range(5):
        now_t[0] = 100.0 + i * 0.01
        assert mb.put(StereoFrame(frame_id=i, timestamp_s=i * 0.1, left=img))

    assert mb.dropped_frames == 4
    assert mb.received_frames == 5
    assert mb.size == 1

    now_t[0] = 100.055  # 15 ms after frame 4 was enqueued at 100.04
    got = mb.get(timeout=0.5)
    assert got is not None
    assert got.frame_id == 4
    assert got.dropped_before == 4
    assert abs(mb.queue_age_ms - 15.0) < 1e-3
    assert mb.delivered_frames == 1


def test_frame_mailbox_fifo_lossless_backpressure():
    mb = FrameMailbox(mode="fifo", maxsize=2)
    img = np.zeros((16, 16), dtype=np.uint8)
    assert mb.put(StereoFrame(frame_id=0, timestamp_s=0.0, left=img))
    assert mb.put(StereoFrame(frame_id=1, timestamp_s=0.1, left=img))
    # Queue is full -> short timeout returns False without dropping anything
    assert not mb.put(StereoFrame(frame_id=2, timestamp_s=0.2, left=img), timeout=0.03)
    assert mb.dropped_frames == 0

    received_ids = []

    def producer():
        for i in range(2, 6):
            assert mb.put(StereoFrame(frame_id=i, timestamp_s=i * 0.1, left=img), timeout=1.0)
        mb.close()

    t = threading.Thread(target=producer)
    t.start()
    while True:
        f = mb.get(timeout=1.0)
        if f is None:
            break
        received_ids.append(f.frame_id)
    t.join(timeout=2.0)

    assert received_ids == [0, 1, 2, 3, 4, 5]
    assert mb.dropped_frames == 0


# ---------------------------------------------------------------------------
# Phase 1: Calibration, Rectification & FrameSource adapters
# ---------------------------------------------------------------------------


def test_stereo_calibration_kitti_yaml_json_and_rectifier(tmp_path: Path):
    # 1. KITTI unlabeled 2-line format (KITTI_sequence_1/calib.txt)
    kitti_calib = StereoCalibration.from_file(REPO_ROOT / "KITTI_sequence_1" / "calib.txt")
    assert kitti_calib.is_stereo
    assert kitti_calib.baseline == pytest.approx(379.8145 / 707.0912, rel=1e-4)
    assert StereoRectifier(kitti_calib, image_size=(1241, 376)).is_noop

    # 2. Unrectified distorted calibration -> YAML & JSON roundtrip -> StereoRectifier
    K_l = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    K_r = np.array([[502.0, 0.0, 318.0], [0.0, 502.0, 242.0], [0.0, 0.0, 1.0]])
    dist_l = np.array([-0.12, 0.03, 0.001, -0.001, 0.0])
    dist_r = np.array([-0.10, 0.02, -0.001, 0.001, 0.0])
    R = Rotation.from_euler("xyz", [0.4, -0.3, 0.2], degrees=True).as_matrix()
    T = np.array([-0.12, 0.002, -0.001])

    calib = StereoCalibration(
        K_left=K_l,
        K_right=K_r,
        dist_left=dist_l,
        dist_right=dist_r,
        R=R,
        T=T,
        image_size=(640, 480),
        is_rectified=False,
    )
    calib.ensure_rectification()
    assert calib.P1 is not None and calib.P2 is not None and calib.Q is not None
    assert abs(calib.baseline - 0.12) < 0.01

    yaml_path = tmp_path / "stereo.yaml"
    json_path = tmp_path / "stereo.json"
    calib.save_yaml(yaml_path)
    calib.save_json(json_path)

    from_y = StereoCalibration.from_file(yaml_path)
    from_j = StereoCalibration.from_file(json_path)
    assert np.allclose(from_y.K_left, K_l)
    assert np.allclose(from_j.P2, calib.P2)
    assert from_y.baseline == pytest.approx(calib.baseline, rel=1e-6)

    rectifier = StereoRectifier(from_y)
    assert not rectifier.is_noop
    img = _make_textured_scene_image(640, 480, seed=5)
    rl, rr = rectifier.rectify(img, img)
    assert rl.shape == img.shape and rr is not None and rr.shape == img.shape


def test_paced_kitti_source_drop_stale_and_lossless(tmp_path: Path):
    seq_dir = tmp_path / "sequences" / "00"
    (seq_dir / "image_0").mkdir(parents=True)
    (seq_dir / "image_1").mkdir(parents=True)
    img = _make_textured_scene_image(160, 96, seed=3)
    for i in range(6):
        cv.imwrite(str(seq_dir / "image_0" / f"{i:06d}.png"), img)
        cv.imwrite(str(seq_dir / "image_1" / f"{i:06d}.png"), img)
    (seq_dir / "calib.txt").write_text(
        "P0: 700 0 80 0 0 700 48 0 0 0 1 0\n"
        "P1: 700 0 80 -350 0 700 48 0 0 0 1 0\n",
        encoding="utf-8",
    )
    (seq_dir / "times.txt").write_text("\n".join(f"{i * 0.1:.3f}" for i in range(6)) + "\n", encoding="utf-8")

    # Simulate slow consumer with fake monotonic clock jumping 0.25s between reads
    fake_t = [0.0]
    sleeps = []

    def clock():
        return fake_t[0]

    def sleeper(dt):
        sleeps.append(dt)
        fake_t[0] += dt

    src_drop = PacedKittiSource(
        tmp_path, "00", rate=1.0, drop_stale=True, clock=clock, sleeper=sleeper
    )
    with src_drop:
        f0 = src_drop.read()
        assert f0 is not None and f0.frame_id == 0 and f0.dropped_before == 0
        fake_t[0] = 0.25  # 2.5 frames elapsed -> skips frame 1, delivers frame 2
        f_next = src_drop.read()
        assert f_next is not None and f_next.frame_id == 2 and f_next.dropped_before == 1

    # Lossless mode delivers all frames in order
    src_fifo = PacedKittiSource(tmp_path, "00", rate=0.0, drop_stale=False)
    ids = []
    with src_fifo:
        while (f := src_fifo.read()) is not None:
            ids.append(f.frame_id)
    assert ids == [0, 1, 2, 3, 4, 5]


def test_stereo_video_and_uvc_camera_and_optional_sources(tmp_path: Path):
    # Write a tiny side-by-side stereo AVI video and verify StereoVideoFileSource
    h, w = 80, 120
    sbs_path = tmp_path / "stereo_sbs.avi"
    writer = cv.VideoWriter(str(sbs_path), cv.VideoWriter_fourcc(*"MJPG"), 20.0, (w * 2, h))
    assert writer.isOpened()
    for i in range(4):
        frame = np.full((h, w * 2, 3), 40 + i * 20, dtype=np.uint8)
        frame[:, :w, 0] = 200
        frame[:, w:, 2] = 180
        writer.write(frame)
    writer.release()

    vsrc = StereoVideoFileSource(sbs_path, layout="sbs", rate=0.0, grayscale=True)
    frames = []
    with vsrc:
        while (f := vsrc.read()) is not None:
            frames.append(f)
    assert len(frames) == 4
    assert frames[0].left.shape == (h, w)
    assert frames[0].right is not None and frames[0].right.shape == (h, w)

    # Mock UVC camera capture factory
    class _FakeCap:
        def __init__(self, dev):
            self.dev = dev
            self.n = 0
            self.opened = True

        def isOpened(self):
            return self.opened

        def set(self, prop, val):
            return True

        def read(self):
            self.n += 1
            time.sleep(0.005)
            img = np.full((60, 160, 3), (self.n * 10) % 255, dtype=np.uint8)
            return True, img

        def release(self):
            self.opened = False

    cam = OpenCVStereoCameraSource(0, layout="sbs", grayscale=True, capture_factory=_FakeCap)
    with cam:
        f = cam.read(timeout=1.0)
        assert f is not None
        assert f.left.shape == (60, 80)
        assert f.right is not None and f.right.shape == (60, 80)

    # Soft dependencies raise clear ImportError when absent
    if "pyrealsense2" not in sys.modules:
        with pytest.raises(ImportError, match="pyrealsense2"):
            RealSenseStereoSource().open()
    if "depthai" not in sys.modules:
        with pytest.raises(ImportError, match="depthai"):
            DepthAIStereoSource().open()


# ---------------------------------------------------------------------------
# Phase 1: Timestamp-aware SE(3) prediction & physical speed guard
# ---------------------------------------------------------------------------


def test_dt_scaled_se3_prediction_and_speed_guard():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    slam = SharedSlam(K, config=MappingConfig(loop_mode="off", bundle_enabled=False))

    # Pure translation + yaw rotation increment of 1.0m forward, 5 deg yaw per 0.1s
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_euler("y", 5.0, degrees=True).as_matrix()
    T1[:3, 3] = np.array([0.0, 0.0, 1.0])

    slam.map.poses = [T0, T1]
    slam.map.statuses = ["tracking", "tracking"]
    slam.map.pose_anchors = [None, None]
    slam._frame_timestamps_s = [0.0, 0.1]

    # Case 1: nominal dt=0.1 -> dt_ratio = 1.0 -> exact matrix product T1 @ inv(T0) @ T1
    slam._update_frame_timing(0.2)
    assert slam._current_dt_ratio == pytest.approx(1.0)
    pred_1x = slam._motion_prediction()
    assert np.allclose(pred_1x, T1 @ np.linalg.inv(T0) @ T1, atol=1e-12)

    # Case 2: 1 dropped frame (dt=0.2 vs prev_dt=0.1 -> dt_ratio = 2.0) -> T1 @ delta^2
    slam._update_frame_timing(0.3)
    assert slam._current_dt_ratio == pytest.approx(2.0)
    pred_2x = slam._motion_prediction()
    delta = np.linalg.inv(T0) @ T1
    assert np.allclose(pred_2x, T1 @ delta @ delta, atol=1e-10)

    # Case 3: extreme pause (dt=5.0) clamped to dt_ratio = 3.0
    slam._update_frame_timing(5.1)
    assert slam._current_dt_ratio == pytest.approx(3.0)

    # Case 4: tiny dt (dt=0.005) clamped to dt_ratio = 0.25
    slam._update_frame_timing(0.105)
    assert slam._current_dt_ratio == pytest.approx(0.25)

    # Case 5: timestamp_s=None defaults to dt=0.1 and dt_ratio=1.0 (bit-exact offline)
    slam._update_frame_timing(None)
    assert slam._current_dt_s == 0.1
    assert slam._current_dt_ratio == 1.0
    assert np.array_equal(slam._motion_prediction(), T1 @ np.linalg.inv(T0) @ T1)

    # Physical speed guard: 1.5 m in 0.05 s = 30 m/s > 17.0 m/s -> triggers prefetch
    slam.stereo = StereoCamera(
        cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5),
        np.array([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 500], [0, 0, 2.0, 0]], np.float32),
        0.5,
    )
    kf0 = MappingKeyframe(0, 0, T0, np.zeros((1, 2)), np.zeros((1, 128), np.float32), np.array([-1]))
    kf1 = MappingKeyframe(1, 1, T1, np.zeros((1, 2)), np.zeros((1, 128), np.float32), np.array([-1]))
    slam.map.keyframes = {0: kf0, 1: kf1}
    slam.previous_tracks = [(i, np.array([10.0, 10.0])) for i in range(100)]
    T_fast = np.eye(4)
    T_fast[:3, 3] = [0.0, 0.0, 2.5]  # step = 1.5m
    slam.map.poses = [T0, T1, T_fast]
    slam._frame_timestamps_s = [0.0, 0.1, 0.15]  # dt=0.05s => 30 m/s > 17 m/s
    assert slam.should_prefetch_full_extraction(3) is True
    # Same 1.5m step over 0.2s = 7.5 m/s < 17 m/s => no high-speed trigger
    slam._frame_timestamps_s = [0.0, 0.1, 0.30]
    assert slam.should_prefetch_full_extraction(3) is False
    slam.close(finish=False)


# ---------------------------------------------------------------------------
# Phase 2: LocalMappingWorker thread safety & landmark_ids_version invalidation
# ---------------------------------------------------------------------------


def test_local_mapping_worker_thread_safety_coalescing_and_version_bump():
    state = MapState(metric=True)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    # Populate 4 keyframes + 12 landmarks
    for kf_id in range(4):
        p = np.eye(4)
        p[:3, 3] = [0.0, 0.0, float(kf_id)]
        kf = MappingKeyframe(
            id=kf_id,
            frame=kf_id * 5,
            pose=p,
            pixels=np.zeros((12, 2), dtype=np.float32),
            descriptors=np.zeros((12, 128), dtype=np.float32),
            landmark_ids=np.arange(12, dtype=int),
        )
        state.keyframes[kf_id] = kf
        state.record(p, "tracking", kf_id)

    for lid in range(12):
        obs = {
            kf_id: Observation(np.array([320.0 + lid * 5.0, 240.0], dtype=np.float32), right_u=300.0 + lid * 5.0)
            for kf_id in range(4)
        }
        state.add_landmark(
            np.array([lid * 0.2, 0.0, 8.0]),
            np.zeros(128, dtype=np.float32),
            anchor=0,
            observations=obs,
        )

    # Custom fake solver that sleeps briefly, moves KF 2 & 3 slightly, moves landmark 0, and culls (lid=1, kf=3)
    solver_threads = []

    def fake_solver(shadow_state, matrix, baseline, **kwargs):
        solver_threads.append(threading.current_thread().name)
        time.sleep(0.03)
        corrected = {k: kf.pose.copy() for k, kf in shadow_state.keyframes.items()}
        corrected[2][:3, 3] += np.array([0.02, 0.0, 0.0])
        corrected[3][:3, 3] += np.array([0.05, 0.0, 0.0])
        lm_upd = {0: np.array([0.15, 0.01, 8.05])}
        shadow_state.apply_corrections(
            shadow_state.revision, corrected, propagate_landmarks=False, landmark_updates=lm_upd
        )
        shadow_state.remove_observation(1, 3)
        return {"applied": True, "cost_before": 10.0, "cost_after": 4.0}

    worker = LocalMappingWorker(solver=fake_solver)
    try:
        # Submit 3 snapshots rapidly -> coalesces queued snapshots
        for sid in (1, 2, 3):
            snap = LocalBASnapshot.capture(
                state,
                snapshot_id=sid,
                frame_index=15,
                anchor_kf_id=3,
                K=K,
                baseline=0.5,
                bundle_kwargs={"window": 4},
            )
            assert snap is not None
            worker.submit(snap)

        # Meanwhile on tracking thread, add KF 4 (simulating live tracking progress during BA!)
        p4 = np.eye(4)
        p4[:3, 3] = [0.0, 0.0, 4.0]
        state.keyframes[4] = MappingKeyframe(
            id=4,
            frame=20,
            pose=p4,
            pixels=np.zeros((1, 2), dtype=np.float32),
            descriptors=np.zeros((1, 128), dtype=np.float32),
            landmark_ids=np.array([-1]),
        )
        state.record(p4, "tracking", 4)
        state.revision += 1

        assert worker.wait_idle(timeout=3.0)
        corr = worker.poll_correction()
        assert corr is not None
        assert all(name == "local-mapping-ba" for name in solver_threads)

        # Verify single-writer: live state has NOT been mutated before apply_local_ba_correction
        assert state.keyframes[3].pose[0, 3] == pytest.approx(0.0)
        assert state.landmark_ids_version == 0
        assert state.keyframes[3].landmark_ids_version == 0

        applied = apply_local_ba_correction(state, corr)
        assert applied is True

        # KF 2, 3 moved by BA; KF 4 rigidly propagated by KF 3's delta (+0.05 in x)!
        assert state.keyframes[2].pose[0, 3] == pytest.approx(0.02)
        assert state.keyframes[3].pose[0, 3] == pytest.approx(0.05)
        assert state.keyframes[4].pose[0, 3] == pytest.approx(0.05)
        assert np.allclose(state.landmarks[0].position, [0.15, 0.01, 8.05])
        # Outlier observation (lid=1, kf=3) culled AND landmark_ids_version incremented!
        assert 3 not in state.landmarks[1].observations
        assert state.keyframes[3].landmark_ids[1] == -1
        assert state.keyframes[3].landmark_ids_version == 1
        assert state.landmark_ids_version == 1

        # Re-applying the now-stale correction must be rejected cleanly
        assert apply_local_ba_correction(state, corr) is False
    finally:
        worker.close(wait=False)


# ---------------------------------------------------------------------------
# Phase 5: Live relocalization recovery (INIT -> TRACKING -> LOST -> RELOCALIZING -> TRACKING)
# ---------------------------------------------------------------------------


def test_live_relocalization_recovery_after_occlusion():
    calib, pairs = _make_synthetic_stereo_sequence(n_frames=8, w=360, h=220, disparity_px=16.0)
    stereo_cam = calib.make_stereo_camera(num_disparities=64, block_size=5)
    cfg = MappingConfig(
        features=800,
        keyframe_interval=2,
        bundle_enabled=False,
        loop_mode="off",
    )
    slam = SharedSlam(calib.rectified_intrinsics(), stereo=stereo_cam, config=cfg)
    assert slam.tracking_state == "INIT"

    states = []
    try:
        # Frames 0..3: healthy tracking
        for i in range(4):
            _, info = slam.process_frame(i, pairs[i][0], pairs[i][1], timestamp_s=i * 0.1)
            states.append(info["tracking_state"])
            assert info["tracking_ok"] is True
            assert info["tracking_state"] == "TRACKING"

        # Frames 4..6: complete lens occlusion (pure black frames) -> LOST -> RELOCALIZING
        black = np.zeros_like(pairs[0][0])
        for idx in (4, 5, 6):
            _, info = slam.process_frame(idx, black, black, timestamp_s=idx * 0.1)
            states.append(info["tracking_state"])
            assert info["tracking_ok"] is False
        assert states[4] == "LOST"
        assert states[5] == "RELOCALIZING"
        assert states[6] == "RELOCALIZING"

        # Frame 7: lens uncovered back at frame 2's view -> automatic place-recognition + PnP recovery!
        pose_rec, info_rec = slam.process_frame(7, pairs[2][0], pairs[2][1], timestamp_s=0.7)
        assert info_rec["tracking_ok"] is True
        assert info_rec["tracking_state"] == "TRACKING"
        assert info_rec["state"] == "relocalized"
        # Recovered pose should be close to frame 2's pose
        err_m = float(np.linalg.norm(pose_rec[:3, 3] - slam.map.poses[2][:3, 3]))
        assert err_m < 0.25
    finally:
        slam.close(finish=False)


# ---------------------------------------------------------------------------
# Phase 3: RealtimeSlamEngine + FastAPI REST & WebSocket streaming
# ---------------------------------------------------------------------------


def test_realtime_engine_and_fastapi_rest_and_websockets(tmp_path: Path):
    calib, pairs = _make_synthetic_stereo_sequence(n_frames=8, w=320, h=200, disparity_px=16.0)
    src = _SyntheticStereoSource(calib, pairs, fps=40.0)
    cfg = MappingConfig(
        features=600,
        keyframe_interval=2,
        bundle_enabled=True,
        loop_mode="off",
        async_local_ba=True,
        use_incremental_map_index=True,
        use_analytic_ba_jacobians=True,
    )
    engine = RealtimeSlamEngine(
        source=src,
        config=cfg,
        drop_frames=False,
        device="cpu",
        async_local_ba=True,
    )
    app = create_app(engine, telemetry_hz=40.0, map_hz=20.0, video_hz=30.0)

    try:
        with TestClient(app) as client:
            # Start engine via REST
            r = client.post("/api/control/start")
            assert r.status_code == 200 and r.json()["ok"] is True

            assert engine.wait_finished(timeout=15.0)

            # Check /api/status
            st = client.get("/api/status").json()
            assert st["processed_frames"] == 8
            assert st["keyframes"] >= 2
            assert st["active_landmarks"] > 20
            assert st["tracking_state"] == "TRACKING"
            assert "track_p50_ms" in st and "track_p95_ms" in st
            assert "backend_device" in st

            # Check /api/snapshot with base64 Float32 anchor-relative point chunks
            snap = client.get("/api/snapshot").json()
            assert snap["type"] == "map_snapshot"
            assert len(snap["keyframes"]) >= 2
            assert len(snap["trajectory"]) == 8
            kf0 = snap["keyframes"][0]
            raw_bytes = base64.b64decode(kf0["points_f32_b64"])
            pts = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 3)
            assert len(pts) == kf0["point_count"] and len(pts) > 0

            # Check control endpoints: rate, drop_frames, pause, resume, save_trajectory
            assert client.post("/api/control/rate", json={"rate": 2.0}).json()["playback_rate"] == 2.0
            assert client.post("/api/control/drop_frames", json={"drop_frames": True}).json()["drop_frames"] is True
            assert client.post("/api/control/pause").status_code == 200
            assert client.post("/api/control/resume").status_code == 200

            traj_out = tmp_path / "traj.txt"
            res_save = client.post("/api/control/save_trajectory", json={"path": str(traj_out)})
            assert res_save.status_code == 200 and traj_out.is_file()

            # WebSocket /ws/telemetry
            with client.websocket_connect("/ws/telemetry") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "telemetry"
                assert msg["processed_frames"] == 8

            # WebSocket /ws/map
            with client.websocket_connect("/ws/map") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "map_snapshot"
                assert len(msg["keyframes"]) >= 2

            # WebSocket /ws/video (binary JPEG)
            with client.websocket_connect("/ws/video") as ws:
                jpeg_bytes = ws.receive_bytes()
                assert jpeg_bytes[:3] == b"\xff\xd8\xff"

            # Reset clears map
            res_reset = client.post("/api/control/reset")
            assert res_reset.status_code == 200
            assert client.get("/api/status").json()["processed_frames"] == 0
    finally:
        engine.stop(finish=False)


# ---------------------------------------------------------------------------
# Phase 5: Chessboard stereo calibration utility
# ---------------------------------------------------------------------------


def test_calibrate_stereo_utility_synthetic_chessboard(tmp_path: Path):
    import calibrate_stereo

    cols, rows = 7, 5
    sq = 0.04
    w, h = 480, 360
    K = np.array([[420.0, 0.0, w * 0.5], [0.0, 420.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = 0.12

    # Build a crisp chessboard pattern image (8x6 squares -> 7x5 inner corners)
    cell = 40
    board_img = np.full(((rows + 3) * cell, (cols + 3) * cell), 230, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = (r + 1) * cell, (c + 1) * cell
                board_img[y0 : y0 + cell, x0 : x0 + cell] = 25

    # Board plane 3D corners corresponding to pattern image corners
    bw_m = (cols + 3) * sq
    bh_m = (rows + 3) * sq
    src_pts = np.array(
        [[0, 0], [board_img.shape[1] - 1, 0], [board_img.shape[1] - 1, board_img.shape[0] - 1], [0, board_img.shape[0] - 1]],
        dtype=np.float32,
    )
    plane_3d = np.array(
        [
            [-bw_m * 0.5, -bh_m * 0.5, 0.0],
            [bw_m * 0.5, -bh_m * 0.5, 0.0],
            [bw_m * 0.5, bh_m * 0.5, 0.0],
            [-bw_m * 0.5, bh_m * 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    poses = [
        ([0.0, 0.0, 0.0], [0.04, 0.0, 0.75]),
        ([12.0, -8.0, 5.0], [0.08, -0.03, 0.80]),
        ([-10.0, 12.0, -6.0], [0.02, 0.04, 0.85]),
        ([15.0, 10.0, 8.0], [0.06, 0.02, 0.78]),
        ([-14.0, -10.0, -4.0], [0.05, -0.02, 0.82]),
        ([6.0, -15.0, 10.0], [0.07, 0.03, 0.76]),
        ([-8.0, 6.0, -12.0], [0.03, -0.04, 0.84]),
    ]

    left_imgs, right_imgs = [], []
    for euler_deg, trans in poses:
        R_b = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        t_b = np.asarray(trans, dtype=np.float64)
        pts_l = (plane_3d @ R_b.T) + t_b
        pts_r = pts_l + np.array([-baseline, 0.0, 0.0])

        uv_l = (pts_l @ K.T)
        uv_l = (uv_l[:, :2] / uv_l[:, 2:3]).astype(np.float32)
        uv_r = (pts_r @ K.T)
        uv_r = (uv_r[:, :2] / uv_r[:, 2:3]).astype(np.float32)

        H_l = cv.getPerspectiveTransform(src_pts, uv_l)
        H_r = cv.getPerspectiveTransform(src_pts, uv_r)
        il = cv.warpPerspective(board_img, H_l, (w, h), borderValue=180)
        ir = cv.warpPerspective(board_img, H_r, (w, h), borderValue=180)
        left_imgs.append(il)
        right_imgs.append(ir)

    calib, report = calibrate_stereo.calibrate_stereo_pairs(
        left_imgs,
        right_imgs,
        board_size=(cols, rows),
        square_size_m=sq,
        min_valid_pairs=5,
    )
    assert report["pairs_used"] >= 5
    assert report["rms_stereo_px"] < 1.0
    assert report["mean_epipolar_y_error_px"] < 0.5
    assert abs(calib.baseline - baseline) < 0.005

    out_yaml = tmp_path / "calibrated.yaml"
    calib.save_yaml(out_yaml)
    loaded = StereoCalibration.from_file(out_yaml)
    assert loaded.baseline == pytest.approx(calib.baseline, rel=1e-5)
