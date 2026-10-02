"""Saved-run diagnostics must distinguish evidence, refuse invalid comparisons,
and preserve input artifacts while reporting retroactive pose changes.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from kitti import save_poses_txt

spec = importlib.util.spec_from_file_location("stereo_regression_diagnostics",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_stereo_regression.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def saved_run(folder, *, source="flow_assisted_map", shift=0., gt_used=False):
    folder.mkdir()
    (folder / "source").mkdir()
    archive = folder / "source" / "shared_slam.py"
    archive.write_text("# archived source\n")
    poses = [np.eye(4) for _ in range(3)]
    for i, pose in enumerate(poses):
        pose[2, 3] = i + (shift if i else 0.)
    save_poses_txt(folder / "poses.txt", poses)
    evaluation = {"stereo": True, "ground_truth_used_for_estimation": gt_used, "sequence": "04", "frames": 3,
                  "source_sha256": {"shared_slam.py": hashlib.sha256(archive.read_bytes()).hexdigest()}}
    measurement = np.eye(4)
    measurement[2, 3] = 1.
    run = {"tracking": [{"frame": i, "pose_source": source if i == 2 else "flow_assisted_map"} for i in range(3)],
           "keyframes": [{"frame": 0}, {"frame": 2}], "bundle_adjustment": [],
           "independent_stereo_motion": [{"previous_frame": 0, "frame": 1, "measurement": measurement.tolist()}]}
    (folder / "run.json").write_text(json.dumps(run))
    (folder / "evaluation.json").write_text(json.dumps(evaluation))
    return poses


def test_distinguishes_pose_motion_source_and_does_not_modify_inputs(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    saved_run(first)
    saved_run(second, shift=.7, source="stereo_tracking_reference")
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = diagnostics.compare(first, second)
    assert report["first_tracking_divergence"]["frame"] == 2
    assert [r["frame"] for r in report["pose_source_changes"]] == [2]
    assert report["frames"][1]["exported_increment_difference"]["translation_m"] == pytest.approx(.7)
    assert len(report["second_motion_agreement"]["violations"]) == 1
    assert "posthoc_reference_evaluation" not in report
    assert all(p.read_bytes() == value for p, value in originals.items())


@pytest.mark.parametrize("failure", ["source", "coverage", "ground_truth", "endpoint"])
def test_refuses_invalid_saved_evidence(tmp_path, failure):
    folder = tmp_path / "a"
    saved_run(folder, gt_used=failure == "ground_truth")
    if failure == "source":
        (folder / "source" / "shared_slam.py").write_text("modified")
    if failure == "coverage":
        evaluation = json.loads((folder / "evaluation.json").read_text())
        evaluation["frames"] = 5
        (folder / "evaluation.json").write_text(json.dumps(evaluation))
    if failure == "endpoint":
        run = json.loads((folder / "run.json").read_text())
        run["independent_stereo_motion"][0]["frame"] = 3
        (folder / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        diagnostics.load_run(folder)


def test_cache_comparison_pairs_image_identity_and_detects_depth_support_loss(tmp_path):
    for signature in ("old", "new"):
        (tmp_path / signature).mkdir()
    original = {"pixels": np.array([[1., 2.], [3., 4.]]), "descriptors": np.ones((2, 4)),
                "disparity": np.ones((3, 4)), "points": np.array([[0., 0., 10.], [0., 0., 20.]])}
    changed = {**original, "points": np.array([[0., 0., 11.], [np.nan, np.nan, np.nan]])}
    np.savez(tmp_path / "old" / "image-hash.npz", **original)
    np.savez(tmp_path / "new" / "image-hash.npz", **changed)
    np.savez(tmp_path / "old" / "unpaired-image.npz", **original)
    report = diagnostics.compare_caches(tmp_path, "old", "new")
    assert report["paired_blobs"] == 1 and report["unpaired_first"] == 1
    assert all(report["exact_extraction_arrays"].values())
    assert report["lost_depth"] == 1 and report["gained_depth"] == 0
    assert report["common_depth_change_m_percentiles"]["median"] == 1.
    np.savez(tmp_path / "new" / "image-hash.npz", **{**changed, "pixels": original["pixels"] + 1.})
    with pytest.raises(ValueError, match="feature indices"):
        diagnostics.compare_caches(tmp_path, "old", "new")


def test_reference_is_explicit_and_reports_offline_evaluation_only(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    poses = saved_run(first)
    saved_run(second)
    reference = tmp_path / "reference.txt"
    save_poses_txt(reference, poses)
    report = diagnostics.compare(first, second, reference_poses=reference)
    assert report["estimator_invoked"] is False
    assert report["posthoc_reference_evaluation"]["reference_used_for_estimation"] is False
    assert report["posthoc_reference_evaluation"]["second"]["raw_ate_rmse_m"] == 0.
