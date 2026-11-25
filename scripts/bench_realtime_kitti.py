"""Benchmark paced real-time KITTI stereo SLAM: Synchronous vs Async Local BA.

Measures tracking latency (p50 / p95, split by keyframe vs non-keyframe),
local BA solve latency, loop closure count & verification-to-apply (v2a) p95,
tracking loss count, and SE(3) ATE RMSE against KITTI ground truth.

Usage::

    python3 scripts/bench_realtime_kitti.py --kitti-root .datasets/kitti --sequence 00 --max-frames 1800 --rate 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend_pipeline import FrontendExtractionPipeline
from metrics import evaluate_trajectory
from realtime.frame_source import PacedKittiSource, StereoCalibration
from shared_slam import MappingConfig, PerformanceConfig, SharedSlam


def load_kitti_poses(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray([float(v) for v in parts[:12]], dtype=np.float64).reshape(3, 4)
            poses.append(mat)
            if max_frames is not None and len(poses) >= max_frames:
                break
    return poses


def run_benchmark_mode(
    *,
    kitti_root: Path,
    sequence: str,
    max_frames: int,
    rate: float,
    async_local_ba: bool,
    preset: str = "improved_stereo",
    pipeline_buffer: int = 2,
    use_pipeline_prefetch: bool = True,
) -> dict[str, Any]:
    seq_dir = kitti_root / "sequences" / sequence
    calib = StereoCalibration.from_kitti_calib(seq_dir / "calib.txt")
    source = PacedKittiSource(
        seq_dir,
        rate=rate,
        drop_stale=False,
        max_frames=max_frames,
    )
    cfg = MappingConfig.from_preset(
        preset,
        stereo=True,
        loop_mode="live",
        async_local_ba=async_local_ba,
    )
    perf = PerformanceConfig(cpu_optimizations=True)
    slam = SharedSlam(
        calib.K_left,
        stereo=calib.make_stereo_camera(),
        config=cfg,
        performance=perf,
    )

    track_ms_all: list[float] = []
    keyframe_ms: list[float] = []
    non_keyframe_ms: list[float] = []
    lost_frames = 0
    wall_start = time.perf_counter()

    try:
        if use_pipeline_prefetch:
            # Real-time paced capture with bounded background frontend extraction
            # (matches loop_harness pipelined mode + wall-clock pacing via PacedKittiSource).
            pipeline = FrontendExtractionPipeline(
                slam,
                buffer_size=max(1, int(pipeline_buffer)),
                lazy_extraction=True,
            )
            try:
                timestamps: dict[int, float] = {}
                next_submit = 0
                submitted_eof = False

                def _read_and_submit(eager: bool) -> bool:
                    nonlocal next_submit, submitted_eof
                    if submitted_eof:
                        return False
                    sf = source.read()
                    if sf is None:
                        submitted_eof = True
                        return False
                    idx = next_submit
                    timestamps[idx] = sf.timestamp_s
                    pipeline.submit(idx, sf.left, sf.right, eager=eager)
                    next_submit += 1
                    return True

                # Prime bounded buffer with 1 frame (or pipeline_buffer)
                for _ in range(max(1, int(pipeline_buffer))):
                    if not _read_and_submit(eager=(next_submit == 0)):
                        break

                frame_idx = 0
                while frame_idx < next_submit:
                    extracted = pipeline.get(frame_idx)
                    eager_next = slam.should_prefetch_full_extraction(next_submit)
                    _read_and_submit(eager=eager_next)

                    prev_kfs = len(slam.map.keyframes)
                    t0 = time.perf_counter()
                    _pose, info = slam.process_frame(
                        frame_idx,
                        extracted=extracted,
                        timestamp_s=timestamps.pop(frame_idx, None),
                    )
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    track_ms_all.append(dt_ms)
                    if len(slam.map.keyframes) > prev_kfs:
                        keyframe_ms.append(dt_ms)
                    else:
                        non_keyframe_ms.append(dt_ms)
                    if not info.get("tracking_ok", False):
                        lost_frames += 1
                    frame_idx += 1
            finally:
                pipeline.close()
        else:
            frame_idx = 0
            while True:
                sf = source.read()
                if sf is None:
                    break
                prev_kfs = len(slam.map.keyframes)
                t0 = time.perf_counter()
                _pose, info = slam.process_frame(
                    frame_idx,
                    sf.left,
                    sf.right,
                    timestamp_s=sf.timestamp_s,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                track_ms_all.append(dt_ms)
                if len(slam.map.keyframes) > prev_kfs:
                    keyframe_ms.append(dt_ms)
                else:
                    non_keyframe_ms.append(dt_ms)
                if not info.get("tracking_ok", False):
                    lost_frames += 1
                frame_idx += 1
    finally:
        wall_elapsed_s = time.perf_counter() - wall_start
        worker_stats = (
            slam._local_mapping_worker.stats()
            if getattr(slam, "_local_mapping_worker", None) is not None
            else {}
        )
        slam.close(finish=True)
        source.close()

    # Collect loop closure metrics
    v2a_s: list[float] = []
    loop_discarded = 0
    for ev in slam.loop_worker.events:
        if ev.get("type") == "loop_applied":
            v2a_s.extend(float(v) for v in ev.get("verification_to_apply_s", ()))
        elif ev.get("type") == "loop_discarded":
            loop_discarded += 1

    # Evaluate against ground truth if available
    ate_rmse_m: float | None = None
    translation_percent: float | None = None
    poses_path = kitti_root / "poses" / f"{sequence}.txt"
    if poses_path.is_file() and slam.map.poses:
        gt = load_kitti_poses(poses_path, max_frames=len(slam.map.poses))
        n = min(len(gt), len(slam.map.poses))
        if n >= 5:
            metrics = evaluate_trajectory(gt[:n], slam.map.poses[:n], alignment="se3")
            ate_rmse_m = float(metrics["ate_rmse_m"])
            if metrics.get("translation_percent") is not None:
                translation_percent = float(metrics["translation_percent"])

    def _pct(arr: list[float], p: float) -> float:
        return round(float(np.percentile(arr, p)), 3) if arr else 0.0

    result = {
        "mode": "async_local_ba" if async_local_ba else "sync_local_ba",
        "async_local_ba": bool(async_local_ba),
        "sequence": sequence,
        "frames": len(track_ms_all),
        "rate": float(rate),
        "wall_elapsed_s": round(wall_elapsed_s, 3),
        "effective_fps": round(len(track_ms_all) / max(wall_elapsed_s, 1e-6), 2),
        "track_p50_ms": _pct(track_ms_all, 50),
        "track_p95_ms": _pct(track_ms_all, 95),
        "track_mean_ms": round(float(np.mean(track_ms_all)), 3) if track_ms_all else 0.0,
        "keyframe_p50_ms": _pct(keyframe_ms, 50),
        "keyframe_p95_ms": _pct(keyframe_ms, 95),
        "non_keyframe_p50_ms": _pct(non_keyframe_ms, 50),
        "non_keyframe_p95_ms": _pct(non_keyframe_ms, 95),
        "lost_frames": int(lost_frames),
        "keyframes": int(len(slam.map.keyframes)),
        "landmarks": int(len(slam.map.landmarks)),
        "verified_loops_applied": int(len(slam.loop_worker.verified)),
        "loop_discarded": int(loop_discarded),
        "v2a_p50_ms": _pct([v * 1000.0 for v in v2a_s], 50),
        "v2a_p95_ms": _pct([v * 1000.0 for v in v2a_s], 95),
        "ate_rmse_m": round(ate_rmse_m, 4) if ate_rmse_m is not None else None,
        "translation_percent": round(translation_percent, 4) if translation_percent is not None else None,
        "async_ba_worker": worker_stats,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Sync vs Async Local BA on paced KITTI.")
    parser.add_argument("--kitti-root", type=Path, default=Path(".datasets/kitti"))
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--preset", default="improved_stereo")
    parser.add_argument("--pipeline-buffer", type=int, default=2)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--async-only", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    modes: list[bool] = []
    if not args.async_only:
        modes.append(False)
    if not args.sync_only:
        modes.append(True)

    results: dict[str, Any] = {}
    for async_ba in modes:
        key = "async_local_ba" if async_ba else "sync_local_ba"
        print(f"Running {key} on KITTI {args.sequence} ({args.max_frames} frames, rate={args.rate})...", flush=True)
        res = run_benchmark_mode(
            kitti_root=args.kitti_root,
            sequence=args.sequence,
            max_frames=args.max_frames,
            rate=args.rate,
            async_local_ba=async_ba,
            preset=args.preset,
            pipeline_buffer=args.pipeline_buffer,
            use_pipeline_prefetch=not args.no_prefetch,
        )
        results[key] = res
        print(json.dumps(res, indent=2), flush=True)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
