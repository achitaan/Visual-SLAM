"""Evaluate saved TUM depth predictions without changing reconstruction artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kitti import load_poses_txt
from metrics import umeyama_alignment


def entries(path):
    return [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_file = args.run / "run.json"
    run = json.loads(run_file.read_text())
    manifest = json.loads((args.dense / "manifest.json").read_text())
    if (
        hashlib.sha256(run_file.read_bytes()).hexdigest()
        != manifest["source_run_sha256"]
    ):
        raise ValueError("Dense reconstruction belongs to another run revision")
    rgb = entries(args.reference / "rgb.txt")
    depth_entries = entries(args.reference / "depth.txt")
    depth_times = np.array([float(e[0]) for e in depth_entries])
    poses = load_poses_txt(args.run / "poses.txt")
    truth = np.loadtxt(args.reference / "groundtruth.txt")
    reference = []
    estimated = []
    used = set()
    for e, p in zip(rgb, poses):
        i = int(np.argmin(abs(truth[:, 0] - float(e[0]))))
        if abs(truth[i, 0] - float(e[0])) > 0.02 or i in used:
            continue
        used.add(i)
        reference.append(truth[i, 1:4])
        estimated.append(p[:3, 3])
    scale = 1.0
    if run["translation_scale"] == "arbitrary":
        _, scale, _ = umeyama_alignment(
            np.array(estimated), np.array(reference), with_scale=True
        )
    K = np.array(run["camera_matrix"])
    distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
    accumulators = {
        name: {
            "pixels": 0,
            "absolute_relative_sum": 0.0,
            "squared_error_sum": 0.0,
            "delta1_count": 0,
        }
        for name in ["raw_model_meters", "trajectory_aligned_map_depth"]
    }
    frames = []
    for keyframe in run["keyframes"]:
        timestamp = float(rgb[keyframe["frame"]][0])
        i = int(np.argmin(abs(depth_times - timestamp)))
        if abs(depth_times[i] - timestamp) > 0.02:
            frames.append(
                {"keyframe": keyframe["id"], "status": "no_depth_association"}
            )
            continue
        prediction_path = args.dense / f'depth-{keyframe["id"]:06d}.npz'
        if not prediction_path.exists():
            frames.append({"keyframe": keyframe["id"], "status": "no_prediction"})
            continue
        reference_depth = cv.imread(
            str(args.reference / depth_entries[i][1]), cv.IMREAD_UNCHANGED
        )
        if reference_depth is None:
            raise ValueError("Unreadable reference depth")
        reference_depth = reference_depth.astype(np.float32) / 5000.0
        # Estimation used undistorted RGB; remap evaluator-only registered depth
        # through the same FR1 calibration, with nearest-neighbor interpolation.
        size = (reference_depth.shape[1], reference_depth.shape[0])
        map_x, map_y = cv.initUndistortRectifyMap(
            K, distortion, None, K, size, cv.CV_32FC1
        )
        reference_depth = cv.remap(reference_depth, map_x, map_y, cv.INTER_NEAREST)
        with np.load(prediction_path, allow_pickle=False) as prediction:
            raw = prediction["depth"]
            valid = prediction["valid"]
            to_map = float(prediction["scale_to_map"])
        if raw.shape != reference_depth.shape:
            raise ValueError("Reference/prediction coordinate mismatch")
        valid = (
            valid
            & np.isfinite(raw)
            & (raw > 0)
            & (reference_depth > 0)
            & np.isfinite(reference_depth)
        )
        count = int(valid.sum())
        frames.append(
            {"keyframe": keyframe["id"], "status": "evaluated", "valid_pixels": count}
        )
        for name, depth in [
            ("raw_model_meters", raw),
            ("trajectory_aligned_map_depth", raw * to_map * scale),
        ]:
            if not np.isfinite(depth).all() or count == 0:
                continue
            value = depth[valid]
            target = reference_depth[valid]
            a = accumulators[name]
            a["pixels"] += count
            a["absolute_relative_sum"] += float(
                np.sum(abs(value - target) / target, dtype=np.float64)
            )
            a["squared_error_sum"] += float(
                np.sum((value - target) ** 2, dtype=np.float64)
            )
            a["delta1_count"] += int(
                np.sum(np.maximum(value / target, target / value) < 1.25)
            )
    metrics = {
        name: (
            {
                "pixels": a["pixels"],
                "absolute_relative_error": a["absolute_relative_sum"] / a["pixels"],
                "rmse_m": float(np.sqrt(a["squared_error_sum"] / a["pixels"])),
                "delta1": a["delta1_count"] / a["pixels"],
            }
            if a["pixels"]
            else None
        )
        for name, a in accumulators.items()
    }
    report = {
        "dataset": "TUM FR1",
        "reference_used_only_for_evaluation": True,
        "trajectory_alignment_scale": scale,
        "metrics": metrics,
        "keyframes": frames,
        "source_run_sha256": manifest["source_run_sha256"],
        "provider": manifest["provider"],
    }
    output = args.output or args.dense / "depth-evaluation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
