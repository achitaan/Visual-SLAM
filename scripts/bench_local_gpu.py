"""Benchmark the stereo frontend on a local CUDA device or the OpenCV CPU path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

# Match the SLAM CLI's BLAS defaults; respect explicit user overrides.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from local_gpu_frontend import create_cuda_frontend, tensor_gradient_pyramid, tensor_match_sift, tensor_stereo_disparity
from mapping_geometry import match_descriptors

KITTI_SIZE = (1241, 376)


def load_frames(root: Path | None, count: int, seed: int):
    if root is None:
        width, height = KITTI_SIZE
        rng = np.random.default_rng(seed)
        texture = rng.integers(0, 256, (height, width + 8 + count), dtype=np.uint8)
        return [
            (texture[:, i:i + width].copy(), texture[:, i + 8:i + 8 + width].copy())
            for i in range(count)
        ]

    left_dir, right_dir = root / "image_0", root / "image_1"
    paths = sorted(left_dir.glob("*.png"))[:count]
    if len(paths) != count:
        raise ValueError(f"need {count} PNG frames in {left_dir}, found {len(paths)}")
    frames = []
    for path in paths:
        right_path = right_dir / path.name
        if not right_path.is_file():
            raise ValueError(f"missing matching right image: {right_path}")
        left, right = cv.imread(str(path), cv.IMREAD_GRAYSCALE), cv.imread(str(right_path), cv.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"unreadable or mismatched stereo pair: {path.name}")
        if frames and left.shape != frames[0][0].shape:
            raise ValueError(f"inconsistent sequence dimensions: {path.name}")
        frames.append((left, right))
    return frames


def cpu_gradient_pyramid(gray: np.ndarray, *, levels: int = 3):
    image = gray.astype(np.float32) / 255
    pyramid = []
    for level in range(levels):
        gradient = np.stack([
            cv.Sobel(image, cv.CV_32F, dx, dy, ksize=3, borderType=cv.BORDER_REPLICATE)
            for dx, dy in ((1, 0), (0, 1))
        ])
        pyramid.append((image, gradient))
        if level + 1 == levels or min(image.shape) <= 2:
            break
        image = cv.pyrDown(image)
    return pyramid


class CpuBenchmark:
    def __init__(self, features: int):
        self.sift = cv.SIFT_create(nfeatures=features, contrastThreshold=0.02)
        self.sgbm = cv.StereoSGBM_create(
            numDisparities=96, blockSize=5, P1=600, P2=2400,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        )

    def step(self, left, right, previous):
        cpu_gradient_pyramid(left)
        keypoints, desc = self.sift.detectAndCompute(left, None)
        if desc is None:
            desc = np.empty((0, 128), np.float32)
        if previous is not None:
            match_descriptors(previous, desc)
        self.sgbm.compute(left, right).astype(np.float32) / 16.0
        np.asarray([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
        return desc

    @staticmethod
    def synchronize():
        pass


class CudaBenchmark:
    def __init__(self, frontend):
        self.frontend = frontend

    def step(self, left, right, previous):
        frontend = self.frontend
        image = frontend.image(left)
        tensor_gradient_pyramid(image)
        features = frontend.extract(image)
        desc = features[1]
        if previous is not None:
            tensor_match_sift(previous, desc).cpu().numpy()
        tensor_stereo_disparity(image, frontend.image(right)).cpu().numpy()
        frontend.numpy_features(features, left.shape[::-1])
        return desc

    def synchronize(self):
        self.frontend.torch.cuda.synchronize(self.frontend.device)


def measure(backend, frames, warmup: int, repeats: int) -> dict:
    samples_ms = []
    for _ in range(repeats):
        # Seed temporal matching. The first measured frame always has a predecessor.
        previous = None
        for left, right in frames[:warmup + 1]:
            previous = backend.step(left, right, previous)
        backend.synchronize()
        for left, right in frames[warmup + 1:]:
            start = perf_counter()
            previous = backend.step(left, right, previous)
            backend.synchronize()
            samples_ms.append((perf_counter() - start) * 1000)
    return {
        "samples": len(samples_ms), "mean_ms": float(np.mean(samples_ms)),
        "median_ms": float(np.median(samples_ms)), "p95_ms": float(np.percentile(samples_ms, 95)),
        "effective_fps": 1000 * len(samples_ms) / sum(samples_ms),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "both"), default="auto")
    parser.add_argument("--kitti-root", type=Path, help="KITTI sequence directory with image_0/ and image_1/")
    parser.add_argument("--frames", type=int, default=20, help="timed frames per repeat")
    parser.add_argument("--warmup", type=int, default=3, help="untimed frames per repeat, in addition to seed frame")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--features", type=int, default=1500)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="synthetic input seed")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)
    if min(args.frames, args.repeats, args.opencv_threads) < 1 or min(args.warmup, args.seed) < 0:
        parser.error("frames, repeats and OpenCV threads must be positive; warmup and seed must be nonnegative")
    if not 1 <= args.features <= 10000:
        parser.error("features must be between 1 and 10000")
    cv.setNumThreads(args.opencv_threads)
    cuda, device, reason = None, None, None
    if args.backend != "cpu":
        cuda, device, reason = create_cuda_frontend(features=args.features)
        if cuda is None and args.backend in ("cuda", "both"):
            parser.error(f"CUDA frontend unavailable: {reason}")
    try:
        frames = load_frames(args.kitti_root, args.frames + args.warmup + 1, args.seed)
    except ValueError as error:
        parser.error(str(error))
    h, w = frames[0][0].shape
    backends = {}
    if args.backend in ("cpu", "both") or cuda is None:
        backends["cpu"] = CpuBenchmark(args.features)
    if cuda is not None:
        backends["cuda"] = CudaBenchmark(cuda)
    report = {
        "requested_backend": args.backend, "cuda_device": asdict(device) if device else None,
        "cuda_unavailable_reason": reason, "width": w, "height": h,
        "input": str(args.kitti_root) if args.kitti_root else "synthetic",
        "seed": args.seed if args.kitti_root is None else None,
        "python": platform.python_version(), "opencv": cv.__version__, "numpy": np.__version__,
        "features": args.features, "opencv_threads": args.opencv_threads,
        "openblas_threads": os.environ["OPENBLAS_NUM_THREADS"], "omp_threads": os.environ["OMP_NUM_THREADS"],
        "frames": args.frames, "warmup": args.warmup, "repeats": args.repeats,
        "results": {name: measure(backend, frames, args.warmup, args.repeats) for name, backend in backends.items()},
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['input']} | {w} x {h} | {args.frames} timed frames x {args.repeats} repeats")
        if device is not None:
            major, minor = device.compute_capability
            print(f"{device.name} | compute {major}.{minor} | {device.total_vram_bytes / 1024**3:.2f} GiB VRAM"
                  f" | PyTorch {device.torch_version} / CUDA {device.cuda_version}")
        if reason is not None:
            print(f"CUDA frontend unavailable; using CPU: {reason}")
        for name, stats in report["results"].items():
            print(f"{name}: {stats['mean_ms']:.2f} ms/frame mean, {stats['median_ms']:.2f} ms median,"
                  f" {stats['p95_ms']:.2f} ms p95, {stats['effective_fps']:.2f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
