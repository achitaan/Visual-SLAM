"""Compare frame-tracking and loop acceptance; load reference poses only afterward."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from StereoVisualOdometry import StereoVisualOdometry
from shared_slam import SharedSlam, StereoCamera
from loop_geometry import StereoLoopFrame, verify_loop
from mapping_geometry import estimate_pose, estimate_stereo_reference, match_descriptors
from kitti import load_poses_txt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start", type=int, required=True, help="First target frame")
    parser.add_argument(
        "--stop", type=int, required=True, help="Exclusive target frame"
    )
    parser.add_argument("--loop-min-inliers", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 1 or args.stop <= args.start or args.loop_min_inliers < 15:
        parser.error("Require start >= 1, stop > start and loop minimum >= 15")
    cv.setNumThreads(1)
    cv.setRNGSeed(0)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve().parents[1] / "src").glob("*.py")
    }
    sequence = args.data_root / "sequences" / args.sequence
    vo = StereoVisualOdometry(
        str(sequence / "image_"),
        str(sequence / "calib.txt"),
        False,
        draw_matches=False,
        max_frames=args.stop,
    )
    if len(vo.Images_1) < args.stop:
        parser.error("Requested interval exceeds sequence")
    slam = SharedSlam(vo.K1, StereoCamera(vo.stereo, vo.Q, vo.baseline))
    records = []

    def geometry(index):
        left = vo.Images_1[index]
        pixels, descriptors, points, _ = slam._extract(left, vo.Images_2[index])
        return StereoLoopFrame(
            pixels, points, descriptors, (left.shape[1], left.shape[0])
        )

    try:
        first = geometry(args.start - 1)
        for frame in range(args.start, args.stop):
            second = geometry(frame)
            pairs = match_descriptors(first.descriptors, second.descriptors)
            a, b = pairs.T if len(pairs) else (np.empty(0, int), np.empty(0, int))
            available = np.isfinite(first.points[a]).all(axis=1)
            diagnostics = {}
            forward = estimate_pose(
                first.points[a[available]],
                second.pixels[b[available]],
                vo.K1,
                second.image_size,
                initial_pose=np.eye(4),
                diagnostics=diagnostics,
            )
            loop = verify_loop(first, second, vo.K1, min_inliers=args.loop_min_inliers)
            tracking = estimate_stereo_reference(
                first, second, vo.K1, initial_pose=np.eye(4)
            )
            records.append(
                {
                    "frame": frame,
                    "diagnostics": diagnostics,
                    "forward": None if forward is None else forward[0].tolist(),
                    "loop": None if loop is None else loop["measurement"].tolist(),
                    "tracking": (
                        None if tracking is None else tracking["measurement"].tolist()
                    ),
                }
            )
            first = second
    finally:
        slam.close()
    # All measurements are complete before reference I/O.
    truth = load_poses_txt(args.poses_root / f"{args.sequence}.txt")
    summary = {}
    for method in ("forward", "loop", "tracking"):
        errors = []
        for record in records:
            if record[method] is None:
                continue
            frame = record["frame"]
            reference = np.linalg.inv(truth[frame - 1]) @ truth[frame]
            error = float(
                np.linalg.norm(np.asarray(record[method])[:3, 3] - reference[:3, 3])
            )
            record[f"{method}_translation_error_m"] = error
            errors.append(error)
        summary[method] = {
            "accepted": len(errors),
            "median_translation_error_m": float(np.median(errors)) if errors else None,
            "mean_translation_error_m": float(np.mean(errors)) if errors else None,
        }
    report = {
        "sequence": args.sequence,
        "start": args.start,
        "stop": args.stop,
        "loop_min_inliers": args.loop_min_inliers,
        "ground_truth_used_for_estimation": False,
        "source_sha256": source_hashes,
        "summary": summary,
        "pairs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
