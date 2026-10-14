"""Tracking ablation and independent stereo diagnostics; references are evaluator-only."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import shared_slam
from shared_slam import SharedSlam, StereoCamera
from StereoVisualOdometry import StereoVisualOdometry
from kitti import load_poses_txt, save_poses_txt
from metrics import evaluate_trajectory
from mapping_geometry import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--max-frames", type=int, default=350)
    parser.add_argument(
        "--disable-bundle", action="store_true", help="Diagnostic ablation only"
    )
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv.setNumThreads(1)
    cv.setRNGSeed(0)
    seq = args.data_root / "sequences" / args.sequence
    vo = StereoVisualOdometry(
        str(seq / "image_"),
        str(seq / "calib.txt"),
        False,
        draw_matches=False,
        max_frames=args.max_frames,
    )
    slam = SharedSlam(vo.K1, StereoCamera(vo.stereo, vo.Q, vo.baseline))
    source = {
        p.name: p.read_bytes()
        for p in (Path(__file__).resolve().parents[1] / "src").glob("*.py")
    }
    records = []
    original_bundle = shared_slam.local_bundle_adjustment

    def inspect_bundle(state, matrix, baseline, **options):
        selected = list(state.keyframes)[-options.get("window", 5) :]
        active = [
            l
            for l in state.landmarks.values()
            if sum(k in selected for k in l.observations) >= 2
        ]
        active.sort(
            key=lambda l: (
                sum(k in selected for k in l.observations),
                max(l.observations),
            ),
            reverse=True,
        )
        observations = [
            o for l in active[:200] for k, o in l.observations.items() if k in selected
        ]
        before = state.poses[-1].copy()
        report = (
            {"applied": False, "reason": "diagnostic_ablation"}
            if args.disable_bundle
            else original_bundle(state, matrix, baseline, **options)
        )
        records[-1].update(
            bundle_stereo_observations=sum(o.right_u is not None for o in observations),
            bundle_observations=len(observations),
            bundle_pose_change_m=float(
                np.linalg.norm(state.poses[-1][:3, 3] - before[:3, 3])
            ),
        )
        return report

    shared_slam.local_bundle_adjustment = inspect_bundle
    original_track = slam._track

    def inspect_track(*inputs, **options):
        result, stats = original_track(*inputs, **options)
        if result is not None:
            pose, _ = result
            pixels = np.array([p for _, p in slam.accepted_tracks])
            world = np.array(
                [slam.map.landmarks[lid].position for lid, _ in slam.accepted_tracks]
            )
            _, z = project(world, pose, slam.K)
            uv = np.rint(pixels).astype(int)
            inside = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < current_disparity.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < current_disparity.shape[0])
            )
            disparity = np.full(len(uv), -1.0)
            disparity[inside] = current_disparity[uv[inside, 1], uv[inside, 0]]
            valid = (disparity > 0) & (disparity < 96)
            if valid.any():
                measured_z = slam.K[0, 0] * vo.baseline / disparity[valid]
                stats.update(
                    independent_depth_ratio=float(np.median(z[valid] / measured_z)),
                    independent_disparity_error_px=float(
                        np.median(
                            np.abs(
                                slam.K[0, 0] * vo.baseline / z[valid] - disparity[valid]
                            )
                        )
                    ),
                    independent_depth_support=int(valid.sum()),
                )
        return result, stats

    slam._track = inspect_track
    started = time.perf_counter()
    try:
        for i in range(len(vo.Images_1)):
            left, right = vo.Images_1[i], vo.Images_2[i]
            current_disparity = vo.stereo.compute(left, right).astype(np.float32) / 16
            records.append({"frame": i})
            _, info = slam.process(i, left, right)
            records[-1].update(
                {
                    k: v
                    for k, v in info.items()
                    if k not in ("feature_points", "inlier_mask")
                }
            )
            if i % 50 == 0:
                print(
                    i,
                    info["state"],
                    records[-1].get("independent_depth_ratio"),
                    flush=True,
                )
    finally:
        slam.close()
        shared_slam.local_bundle_adjustment = original_bundle
    args.output.mkdir(parents=True, exist_ok=True)
    save_poses_txt(args.output / "poses.txt", slam.map.poses)
    # Reference I/O occurs after estimator shutdown, never inside a diagnostic callback.
    truth = load_poses_txt(args.poses_root / f"{args.sequence}.txt")[
        : len(slam.map.poses)
    ]
    report = {
        "disable_bundle": args.disable_bundle,
        "elapsed_s": time.perf_counter() - started,
        "metrics": evaluate_trajectory(truth, slam.map.poses),
        "tracking": records,
        "ground_truth_used_for_estimation": False,
        "configuration": slam.config.__dict__,
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in source.items()
        },
    }
    (args.output / "source").mkdir(exist_ok=True)
    for name, data in source.items():
        (args.output / "source" / name).write_bytes(data)
    (args.output / "diagnosis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(
        json.dumps(
            {k: v for k, v in report["metrics"].items() if k != "segments"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
