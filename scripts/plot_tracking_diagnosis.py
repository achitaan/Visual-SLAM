"""Plot saved stereo tracking ablations against evaluator-only KITTI poses."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kitti import load_poses_txt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        parser.error("Supply one label for each run")
    truth = np.array(load_poses_txt(args.reference))
    estimates = [np.array(load_poses_txt(p / "poses.txt")) for p in args.runs]
    frames = min(len(truth), *(len(p) for p in estimates))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")

    def distance(poses):
        return np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(poses[:frames, :3, 3], axis=0), axis=1))
        ]

    axes[0, 0].plot(
        distance(truth), color="black", linewidth=2, label="KITTI ground truth"
    )
    truth_steps = np.diff(distance(truth))
    axes[0, 1].plot(
        np.arange(1, frames),
        truth_steps,
        color="black",
        linewidth=2,
        label="KITTI ground truth",
    )
    for folder, label, poses in zip(args.runs, args.labels, estimates):
        report_path = folder / "diagnosis.json"
        if not report_path.exists():
            report_path = folder / "run.json"
        report = json.loads(report_path.read_text())
        records = report["tracking"][:frames]
        (line,) = axes[0, 0].plot(distance(poses), label=label)
        color = line.get_color()
        axes[0, 1].plot(
            np.arange(1, frames), np.diff(distance(poses)), color=color, alpha=0.7
        )
        indices = [d["frame"] for d in records if "reprojection_error" in d]
        axes[1, 0].plot(
            indices,
            [d["reprojection_error"] for d in records if "reprojection_error" in d],
            color=color,
            alpha=0.7,
        )
        lost = [d["frame"] for d in records if d["state"] == "lost"]
        axes[1, 1].scatter(
            lost,
            np.full(len(lost), args.labels.index(label)),
            marker="|",
            s=100,
            color=color,
        )
    axes[0, 0].set(title="Accumulated path length", ylabel="Distance (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Motion per frame", ylabel="Translation (m)")
    axes[1, 0].set(
        title="Accepted left-image reprojection error", ylabel="Median error (pixels)"
    )
    axes[1, 1].set(
        title="Lost frames", yticks=range(len(args.labels)), yticklabels=args.labels
    )
    for ax in axes.ravel():
        ax.set_xlabel("Frame")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Stereo tracking diagnosis · first {frames} KITTI frames", fontsize=14
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
