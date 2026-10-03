"""Render saved SLAM trajectories, tracking diagnostics and sparse/dense maps."""

import argparse
import json
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import umeyama_alignment
from kitti import load_poses_txt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dense", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference", type=Path, help="Evaluator-only KITTI poses file or TUM folder"
    )
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    preview = json.loads(((args.dense or args.run) / "preview.json").read_text())
    diagnostics = run["tracking"]
    figure, axes = plt.subplots(
        3 if args.reference else 2,
        2,
        figsize=(12, 12 if args.reference else 8),
        constrained_layout=True,
    )
    trajectory = np.array(preview["trajectory"])
    sparse = np.array(preview["sparse"]).reshape(-1, 3)
    dense = np.array(preview["dense"]).reshape(-1, 3)
    units = (
        "m" if run["translation_scale"] == "metric" else "map units (arbitrary scale)"
    )
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 2], color="#4775d1", linewidth=1.5)
    axes[0, 0].set(
        title="Final exported camera trajectory",
        xlabel=f"X ({units})",
        ylabel=f"Z ({units})",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    if args.reference and any(d.get("tracking_ok") for d in diagnostics):
        indices = np.arange(len(trajectory))
        if args.reference.is_file():
            truth = np.array([p[:3, 3] for p in load_poses_txt(args.reference)])[
                : len(trajectory)
            ]
        else:
            entries = [
                l.split()
                for l in (args.reference / "rgb.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")
            ]
            gt = np.loadtxt(args.reference / "groundtruth.txt")
            pairs = []
            used = set()
            for i, e in enumerate(entries[: len(trajectory)]):
                j = int(np.argmin(abs(gt[:, 0] - float(e[0]))))
                if abs(gt[j, 0] - float(e[0])) <= 0.02 and j not in used:
                    pairs.append((i, j))
                    used.add(j)
            indices = np.array([i for i, _ in pairs])
            truth = gt[[j for _, j in pairs], 1:4]
        estimated = trajectory[indices]
        rotation, scale, translation = umeyama_alignment(
            estimated, truth, with_scale=run["translation_scale"] == "arbitrary"
        )
        aligned = scale * (estimated @ rotation.T) + translation
        axes[0, 0].clear()
        axes[0, 0].plot(truth[:, 0], truth[:, 2], label="Reference", color="#22343d")
        axes[0, 0].plot(
            aligned[:, 0], aligned[:, 2], label="SLAM", color="#4775d1", linestyle="--"
        )
        alignment = "Sim(3)" if run["translation_scale"] == "arbitrary" else "SE(3)"
        axes[0, 0].set(
            title=f"{alignment}-aligned evaluation · reference used after tracking",
            xlabel="X (m)",
            ylabel="Z (m)",
        )
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 0].legend()
        axes[2, 0].plot(
            indices, np.linalg.norm(aligned - truth, axis=1), color="#4775d1"
        )
        axes[2, 0].set(
            title=f"Position error after {alignment} alignment",
            xlabel="Frame",
            ylabel="Error (m)",
        )
        axes[2, 1].imshow(
            plt.imread(args.run / run["keyframes"][0]["image"]), cmap="gray"
        )
        axes[2, 1].set_title("Estimator input · first keyframe")
        axes[2, 1].axis("off")
    elif args.reference:
        # Held initialization poses cannot define a monocular alignment scale.
        axes[0, 0].set_title("Uninitialized trajectory · held poses only")
        axes[2, 0].text(.5, .5, "Accuracy unavailable: initialization failed",
                        ha="center", va="center", transform=axes[2, 0].transAxes)
        axes[2, 0].axis("off")
        axes[2, 1].text(.5, .5, "No initialized keyframe image",
                        ha="center", va="center", transform=axes[2, 1].transAxes)
        axes[2, 1].axis("off")
    frames = [d["frame"] for d in diagnostics]
    axes[0, 1].plot(
        frames, [d.get("num_matches", 0) for d in diagnostics], label="Correspondences"
    )
    axes[0, 1].plot(
        frames,
        [d.get("num_inliers", 0) if d.get("tracking_ok") else 0 for d in diagnostics],
        label="Accepted inliers",
    )
    for d in diagnostics:
        if d["state"] == "lost":
            axes[0, 1].axvspan(
                d["frame"] - 0.5, d["frame"] + 0.5, color="#d95d5d", alpha=0.12
            )
    axes[0, 1].set(
        title="Tracking support · red intervals are held poses",
        xlabel="Frame",
        ylabel="Features",
    )
    axes[0, 1].legend()
    if len(sparse):
        axes[1, 0].scatter(
            sparse[:, 0], sparse[:, 2], s=2, c=sparse[:, 1], cmap="viridis", alpha=0.6
        )
    axes[1, 0].set(
        title=f"Persistent sparse map · {len(sparse):,} preview points",
        xlabel=f"X ({units})",
        ylabel=f"Z ({units})",
    )
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    if len(dense):
        colors = np.array(preview.get("dense_colors", [])) / 255.0
        axes[1, 1].scatter(
            dense[:, 0],
            dense[:, 2],
            s=1,
            c=colors if len(colors) == len(dense) else "#8ba9b5",
        )
        axes[1, 1].set(
            title=f"Dense reconstruction · {len(dense):,} sampled points",
            xlabel=f"X ({units})",
            ylabel=f"Z ({units})",
        )
        axes[1, 1].set_aspect("equal", adjustable="datalim")
    else:
        errors = [
            d.get("reprojection_error", np.nan) if d.get("tracking_ok") else np.nan
            for d in diagnostics
        ]
        axes[1, 1].plot(frames, errors, color="#cd852b")
        axes[1, 1].set(
            title="Accepted tracking reprojection residual",
            xlabel="Frame",
            ylabel="Median error (pixels)",
        )
    for axis in axes.flat:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f'Shared SLAM · {len(trajectory)} frames · revision {run["revision"]} · {run["translation_scale"]} scale'
    )
    output = args.output or args.run / "overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
