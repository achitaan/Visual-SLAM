"""Compare completed stereo and monocular runs from one estimator revision."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kitti import load_poses_txt
from metrics import umeyama_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-run", type=Path, required=True)
    parser.add_argument("--mono-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders = [args.stereo_run, args.mono_run]
    reports = [json.loads((p / "evaluation.json").read_text()) for p in folders]
    if (
        not reports[0]["stereo"]
        or reports[1]["stereo"]
        or any(r["dataset"] != "kitti" or r["coverage"] != "full" for r in reports)
        or reports[0]["sequence"] != reports[1]["sequence"]
        or reports[0]["frames"] != reports[1]["frames"]
        or reports[0]["source_sha256"] != reports[1]["source_sha256"]
    ):
        parser.error(
            "Require complete paired KITTI runs with identical source fingerprints"
        )
    truth = np.array([p[:3, 3] for p in load_poses_txt(args.reference)])
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for column, (folder, report, name) in enumerate(
        zip(folders, reports, ["Stereo", "Monocular"])
    ):
        preview = json.loads((folder / "preview.json").read_text())
        trajectory = np.array(preview["trajectory"])
        sparse = np.array(preview["sparse"]).reshape(-1, 3)
        units = "meters" if report["stereo"] else "map units (arbitrary scale)"
        axes[0, column].plot(trajectory[:, 0], trajectory[:, 2], color="#4775d1")
        axes[0, column].set(
            title=f"{name} · original exported trajectory",
            xlabel=f"X ({units})",
            ylabel=f"Z ({units})",
        )
        axes[0, column].set_aspect("equal", adjustable="datalim")
        if len(sparse):
            axes[1, column].scatter(
                sparse[:, 0],
                sparse[:, 2],
                c=sparse[:, 1],
                s=2,
                cmap="viridis",
                alpha=0.6,
            )
        axes[1, column].set(
            title=f'{report["landmarks"]:,} sparse landmarks · {len(sparse):,} shown',
            xlabel=f"X ({units})",
            ylabel=f"Z ({units})",
        )
        axes[1, column].set_aspect("equal", adjustable="datalim")
        reference = truth[: len(trajectory)]
        rotation, scale, translation = umeyama_alignment(
            trajectory, reference, with_scale=not report["stereo"]
        )
        aligned = scale * (trajectory @ rotation.T) + translation
        axes[2, column].plot(
            np.linalg.norm(aligned - reference, axis=1), color="#4775d1"
        )
        alignment = (
            "SE(3) · scale fixed"
            if report["stereo"]
            else "Sim(3) · evaluation fits scale"
        )
        axes[2, column].set(
            title=f'{alignment} · ATE {report["metrics"]["ate_rmse_m"]:.3f} m',
            xlabel="Frame",
            ylabel="Aligned position error (m)",
        )
        axes[2, column].text(
            0.02,
            0.94,
            f'Initialization: frame {report["initialization_frame"]} · lost: {report["lost_frames"]}',
            transform=axes[2, column].transAxes,
            va="top",
            fontsize=9,
        )
    limit = max(axis.get_ylim()[1] for axis in axes[2])
    for axis in axes[2]:
        axis.set_ylim(0, limit)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f'KITTI {reports[0]["sequence"]} · {reports[0]["frames"]} frames · identical estimator revision\nMonocular fitted-scale accuracy does not establish metric accuracy'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
