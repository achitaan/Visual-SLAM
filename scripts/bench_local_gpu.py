#!/usr/bin/env python3
"""Benchmark local NVIDIA GPU vs. CPU native SIMD / OpenCV frontend kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpu_kernels.device import select_device
from gpu_kernels.frontend_gpu import GpuFrontendAccelerator
from mapping_geometry import match_descriptors


def _synthetic_rectify_maps(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[700.0, 0.0, w * 0.5], [0.0, 700.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.15, 0.04, 0.001, -0.001, 0.0], dtype=np.float64)
    return cv.initUndistortRectifyMap(K, dist, np.eye(3), K, (w, h), cv.CV_32FC1)


def _bench_device(device_req: str, iters: int, width: int, height: int) -> dict:
    info = select_device(device_req)
    accel = GpuFrontendAccelerator(device=device_req)
    rng = np.random.default_rng(7)
    left = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    right = np.roll(left, -12, axis=1)
    map_x, map_y = _synthetic_rectify_maps(width, height)

    # Warmup
    rl, rr = accel.rectify_stereo_pair(left, right, map_x, map_y, map_x, map_y)
    ref_l = cv.remap(left, map_x, map_y, cv.INTER_LINEAR)
    rect_mad = float(np.mean(np.abs(rl.astype(np.float32) - ref_l.astype(np.float32))))

    rect_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        accel.rectify_stereo_pair(left, right, map_x, map_y, map_x, map_y)
        rect_times.append((time.perf_counter() - t0) * 1000.0)

    match_results = {}
    for n_desc in (500, 1000, 1500):
        base = rng.integers(0, 220, size=(n_desc, 128), dtype=np.int32)
        da = base.astype(np.float32)
        db = np.clip(base + rng.integers(-5, 6, size=(n_desc, 128)), 0, 255).astype(np.float32)
        pairs = accel.match_descriptors_batched(da, db, ratio=0.75)
        ref_pairs = match_descriptors(da, db, ratio=0.75)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            accel.match_descriptors_batched(da, db, ratio=0.75)
            times.append((time.perf_counter() - t0) * 1000.0)
        match_results[str(n_desc)] = {
            "p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "pairs": int(len(pairs)),
            "exact_parity": bool(np.array_equal(pairs, ref_pairs)),
        }

    disp_times = []
    for _ in range(max(3, iters // 2)):
        t0 = time.perf_counter()
        accel.compute_disparity(left, right)
        disp_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "device": info.to_dict(),
        "resolution": [width, height],
        "rectify_stereo_pair": {
            "p50_ms": round(float(np.percentile(rect_times, 50)), 3),
            "p95_ms": round(float(np.percentile(rect_times, 95)), 3),
            "mean_abs_diff_vs_cv_remap": round(rect_mad, 4),
        },
        "match_descriptors_batched": match_results,
        "compute_disparity": {
            "p50_ms": round(float(np.percentile(disp_times, 50)), 3),
            "p95_ms": round(float(np.percentile(disp_times, 95)), 3),
        },
        "stats": accel.stats(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--width", type=int, default=1241)
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {
        "cpu": _bench_device("cpu", args.iters, args.width, args.height),
        "requested": _bench_device(args.device, args.iters, args.width, args.height),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
