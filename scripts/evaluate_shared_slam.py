"""Evaluate shared SLAM; reference files are opened only after estimator shutdown."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from pathlib import Path
import sys
import time
import json
from collections import Counter
from contextlib import ExitStack
import tempfile
import zipfile
import re
import atexit
import hashlib
import shutil
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_slam import SharedSlam, StereoCamera, MappingConfig
from StereoVisualOdometry import StereoVisualOdometry
from VisualOdometry import VisualOdometry
from kitti import load_poses_txt, validate_sequence
from reconstruction import export_run
from metrics import evaluate_trajectory
from benchmark_telemetry import SnapshotWriter, snapshot_target
from test_budget import Budget, write_json
from feature_cache import FeatureCache, extraction_signature


def peak_memory_mb():
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024 if sys.platform != "darwin" else 1024**2
        )
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (n, ctypes.c_size_t)
            for n in [
                "peak",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            ]
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(process), ctypes.byref(counters), counters.cb
    )
    return counters.peak / 1024**2


def export_reserve_bytes(state, minimum_mb):
    """Conservative room for observations, keyframe images and finite map exports."""
    observations = sum(len(point.observations) for point in state.landmarks.values())
    return (
        int(minimum_mb * 1024**2)
        + len(state.keyframes) * 1024**2
        + observations * 512
        + len(state.landmarks) * 256
    )


def main():
    invocation_started = time.perf_counter()
    evaluator_source = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["kitti", "tum"], default="kitti")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Stream KITTI images from its official archive through bounded caches",
    )
    parser.add_argument("--poses-root", type=Path)
    parser.add_argument("--sequence", default="04")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--disable-bundle", action="store_true")
    parser.add_argument("--loop-mode", choices=["off", "live", "offline"], default="live")
    parser.add_argument("--feature-cache", type=Path, help="Optional diagnostic cache; excludes timings from official performance claims")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        help="Optional display snapshot; no estimator or reference inputs",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=512,
        help="Keep this much free space in addition to estimated export size",
    )
    args = parser.parse_args()
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    budget = Budget(args.max_wall_seconds)
    budget.started -= time.perf_counter() - invocation_started
    if args.min_free_mb < 128:
        parser.error("--min-free-mb must be at least 128")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.remote and not re.fullmatch(r"(0[0-9]|1[0-9]|2[01])", args.sequence):
        parser.error("KITTI sequence must be 00–21")
    if args.remote and args.dataset != "kitti":
        parser.error("--remote supports KITTI only")
    if not args.remote and args.data_root is None:
        parser.error("--data-root is required for local input")
    resources = ExitStack()
    atexit.register(resources.close)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < (args.min_free_mb + 256) * 1024**2:
        raise OSError(
            "Insufficient space to start: free space must cover the export reserve"
        )
    cv.setNumThreads(1)
    cv.setRNGSeed(0)
    times = None
    if args.dataset == "kitti":
        sources = None
        if args.remote:
            from download_kitti_sequence import RangeFile, URL
            from run_kitti_stream import RemoteImages

            remote = resources.enter_context(RangeFile(URL))
            archive = resources.enter_context(zipfile.ZipFile(remote))
            prefix = f"dataset/sequences/{args.sequence}/"
            cameras = range(2 if args.stereo else 1)
            catalogs = [
                sorted(
                    (
                        e
                        for e in archive.infolist()
                        if re.fullmatch(
                            prefix + rf"image_{camera}/[0-9]{{6}}\.png", e.filename
                        )
                    ),
                    key=lambda e: e.filename,
                )
                for camera in cameras
            ]
            if not catalogs[0] or any(len(c) != len(catalogs[0]) for c in catalogs):
                raise ValueError("Invalid image catalog")
            count = min(args.max_frames or len(catalogs[0]), len(catalogs[0]))
            # KITTI ZIP members are shuffled physically. Fetch each selected
            # member, rather than downloading unrelated multi-megabyte blocks.
            sources = [RemoteImages(archive, c[:count]) for c in catalogs]
            seq = Path(
                resources.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix="shared-kitti-", dir=args.output.parent
                    )
                )
            )
            (seq / "calib.txt").write_bytes(archive.read(prefix + "calib.txt"))
            for camera, source in enumerate(sources):
                folder = seq / f"image_{camera}"
                folder.mkdir()
                for i in range(min(2, count)):
                    cv.imwrite(str(folder / f"{i:06d}.png"), source[i])
        else:
            seq = validate_sequence(
                args.data_root, args.sequence, args.stereo, args.max_frames
            )
        if args.stereo:
            vo = StereoVisualOdometry(
                str(seq / "image_"),
                str(seq / "calib.txt"),
                False,
                draw_matches=False,
                max_frames=args.max_frames,
            )
            images = vo.Images_1
            right = vo.Images_2
            matrix = vo.K1
        else:
            vo = VisualOdometry(
                str(seq / "image_0"),
                str(seq / "calib.txt"),
                False,
                camera_id=0,
                draw_matches=False,
                max_frames=args.max_frames,
            )
            images = vo.Images
            right = None
            matrix = vo.K
        if sources is not None:
            images = sources[0]
            right = sources[1] if args.stereo else None
            paths = [seq / "image_0" / f"{i:06d}.png" for i in range(len(images))]
        else:
            paths = images.paths
        loader = lambda i: images[i]
    else:
        if args.stereo:
            parser.error("TUM evaluation uses RGB-only monocular input")
        entries = [
            line.split()
            for line in (args.data_root / "rgb.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if args.max_frames:
            entries = entries[: args.max_frames]
        times = np.array([float(e[0]) for e in entries])
        paths = [args.data_root / e[1] for e in entries]
        matrix = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]])
        distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

        def loader(i):
            image = cv.imread(str(paths[i]), cv.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unreadable RGB input")
            return cv.undistort(image, matrix, distortion)

        vo = None
        right = None
    camera = StereoCamera(vo.stereo, vo.Q, vo.baseline) if args.stereo else None
    slam = SharedSlam(matrix, stereo=camera, config=MappingConfig(bundle_enabled=not args.disable_bundle, loop_mode=args.loop_mode))
    source_snapshot = {
        p.name: p.read_bytes()
        for p in (Path(__file__).resolve().parents[1] / "src").glob("*.py")
    }
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in source_snapshot.items()
    }
    cache = None
    if args.feature_cache:
        signature = extraction_signature(slam, cv)
        cache = FeatureCache(args.feature_cache, signature)
        extract = slam._extract
        def cached_extract(image, right_image):
            key = cache.key(image, right_image)
            values = cache.get(key)
            if values is not None:
                slam.current_disparity = values[4] if args.stereo else None
                return values[:4]
            result = extract(image, right_image)
            cache.put(key, (*result, slam.current_disparity if args.stereo else np.empty((0,0))))
            return result
        slam._extract = cached_extract
    target = snapshot_target(args.output, args.telemetry_file)
    observer = (
        SnapshotWriter(target, args.sequence, args.output.name, len(paths))
        if target
        else None
    )
    started = time.perf_counter()
    initialized_at = None
    storage_interruption = None
    interruption = None
    args.output.mkdir(parents=True, exist_ok=True)
    last_image = None
    try:
        for i in range(len(paths)):
            if budget.remaining <= max(2, min(30, (args.max_wall_seconds or 0) * .1)):
                interruption = {"reason": "interrupted_time_budget", "next_frame": i}
                break
            if args.stop_file and args.stop_file.exists():
                interruption = {"reason": "interrupted_user_stop", "next_frame": i}
                break
            if i % 50 == 0:
                free = shutil.disk_usage(args.output.parent).free
                required = export_reserve_bytes(slam.map, args.min_free_mb)
                if free < required:
                    storage_interruption = {
                        "next_frame": i,
                        "free_bytes": free,
                        "required_bytes": required,
                    }
                    break
            try:
                with slam.profile.measure("image_loading"):
                    left_image = loader(i)
                    right_image = right[i] if right is not None else None
            except (OSError, ValueError) as error:
                interruption = {"reason": "interrupted_input_error", "next_frame": i, "error_type": type(error).__name__, "input_name": Path(paths[i]).name}
                break
            _, info = slam.process(
                i, left_image, right_image
            )
            last_image = left_image
            if observer:
                observer.publish(slam, i, left_image, info)
            if initialized_at is None and info["tracking_ok"]:
                initialized_at = time.perf_counter() - started
            if i % 50 == 0:
                from kitti import save_poses_txt
                save_poses_txt(args.output / "checkpoint-poses.txt", slam.map.poses)
                write_json(args.output / "checkpoint.json", {"frames": i + 1, "coverage": "partial", "status": "running_checkpoint", "source_sha256": source_hashes, "configuration": slam.config.__dict__, "states": dict(Counter(slam.map.statuses))})
                print(
                    f'{args.sequence} {i+1}/{len(paths)} {info["state"]} landmarks={len(slam.map.landmarks)}',
                    flush=True,
                )
    finally:
        slam.close(finish=not (interruption or storage_interruption))
    if observer and slam.map.poses:
        observer.publish(slam, len(slam.map.poses) - 1, left_image, info, force=True)
    elapsed = time.perf_counter() - started
    processed_frames = len(slam.map.poses)
    paths = paths[:processed_frames]
    export_run(slam, args.output, paths, image_loader=loader, include_images=not bool(interruption))
    source_folder = args.output / "source"
    source_folder.mkdir(exist_ok=True)
    for name, data in source_snapshot.items():
        (source_folder / name).write_bytes(data)
    resources.close()
    states = Counter(slam.map.statuses)
    report = {
        "sequence": args.sequence,
        "dataset": args.dataset,
        "frames": processed_frames,
        "stereo": args.stereo,
        "translation_scale": "metric" if args.stereo else "arbitrary",
        "states": dict(states),
        "lost_frames": states["lost"],
        "initializing_frames": states["initializing"],
        "relocalized_frames": states["relocalized"],
        "initialization_frame": next(
            (i for i, s in enumerate(slam.map.statuses) if s == "tracking"), None
        ),
        "keyframes": len(slam.map.keyframes),
        "landmarks": len(slam.map.landmarks),
        "elapsed_s": elapsed,
        "processing_fps": len(paths) / elapsed,
        "peak_memory_mb": peak_memory_mb(),
        "loops": len(slam.loop_worker.verified),
        "loop_events": slam.loop_worker.events,
        "configuration": slam.config.__dict__,
        "coverage": "partial" if args.max_frames or storage_interruption or interruption else "full",
        "ground_truth_used_for_estimation": False,
    }
    report["source_sha256"] = source_hashes
    report["evaluator_sha256"] = hashlib.sha256(evaluator_source).hexdigest()
    report["stage_timings"] = slam.profile.report()
    report["stage_timing_semantics"] = "Inclusive timings; nested stages overlap"
    report["diagnostic_overrides"] = {"disable_bundle": args.disable_bundle, "loop_mode": args.loop_mode if args.loop_mode != "live" else None}
    report["wall_budget_seconds"] = args.max_wall_seconds
    report["feature_cache"] = cache.metadata() if cache else {"enabled": False}
    report["interruption"] = interruption
    (args.output / "evaluator.py").write_bytes(evaluator_source)
    report["telemetry"] = observer.metadata() if observer else {"enabled": False}
    if observer:
        (args.output / "benchmark_telemetry.py").write_bytes(observer.source)
    report["storage_guard"] = {
        "minimum_free_mb": args.min_free_mb,
        "interruption": storage_interruption,
    }
    report["input_source"] = (
        "official_remote_archive" if args.remote else "local_images"
    )
    report["initialization_elapsed_s"] = initialized_at
    report["status"] = (
        "initialization_failed"
        if initialized_at is None
        else "completed_with_tracking_loss" if states["lost"] else "completed"
    )
    if storage_interruption:
        report["status"] = "interrupted_low_disk_space"
    if interruption:
        report["status"] = interruption["reason"]
    report["lost_intervals"] = []
    start = None
    for i, status in enumerate([*slam.map.statuses, "end"]):
        if status == "lost" and start is None:
            start = i
        elif status != "lost" and start is not None:
            report["lost_intervals"].append(
                {
                    "first_frame": start,
                    "last_frame": i - 1,
                    "recovered": status in ("tracking", "relocalized"),
                }
            )
            start = None
    errors = [
        d["reprojection_error"] for d in slam.diagnostics if "reprojection_error" in d
    ]
    report["median_tracking_reprojection_px"] = (
        float(np.median(errors)) if errors else None
    )
    report["bundle_adjustments_applied"] = sum(
        r["applied"] for r in slam.bundle_reports
    )
    # Evaluation owns reference I/O. Estimator and reconstruction only received images/calibration.
    if args.poses_root is not None:
        if args.dataset == "kitti":
            truth = load_poses_txt(args.poses_root / f"{args.sequence}.txt")[
                : len(paths)
            ]
            estimates = slam.map.poses
        else:
            gt = np.loadtxt(args.poses_root / "groundtruth.txt")
            truth = []
            estimates = []
            used = set()
            for timestamp, p in zip(times, slam.map.poses):
                index = int(np.argmin(abs(gt[:, 0] - timestamp)))
                if abs(gt[index, 0] - timestamp) > 0.02 or index in used:
                    continue
                used.add(index)
                g = np.eye(4)
                g[:3, :3] = Rotation.from_quat(gt[index, 4:8]).as_matrix()
                g[:3, 3] = gt[index, 1:4]
                truth.append(g)
                estimates.append(p)
        if len(truth) >= 2:
            try:
                metrics = evaluate_trajectory(
                    truth, estimates, "se3" if args.stereo else "sim3"
                )
            except ValueError as error:
                report["evaluation_error"] = str(error)
                metrics = {}
            if not args.stereo:
                for key in [
                    "raw_ate_rmse_m",
                    "translation_percent",
                    "rotation_deg_per_m",
                ]:
                    metrics[key] = None
                metrics["segments"] = []
                metrics["segment_count"] = 0
            report["metrics"] = {k: v for k, v in metrics.items() if k != "segments"}
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    report["total_wall_seconds"] = time.perf_counter() - invocation_started
    write_json(args.output / "evaluation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("loop_events", "source_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
