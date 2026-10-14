"""Compare saved stereo runs without constructing or invoking an estimator.

Reference poses are optional, explicit, and used only in a separate posthoc report.
Final exported poses may include retroactive BA corrections; their first divergence
is not necessarily the first online divergence. Cache files are opened read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kitti import load_poses_txt, validate_pose


def pose_difference(first, second):
    delta = np.linalg.inv(first) @ second
    return {"translation_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_deg": float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))}


def load_run(folder):
    folder = Path(folder)
    evaluation = json.loads((folder / "evaluation.json").read_text())
    run = json.loads((folder / "run.json").read_text())
    if not evaluation["stereo"] or evaluation["ground_truth_used_for_estimation"]:
        raise ValueError("Requires an image-only stereo estimate")
    poses = load_poses_txt(folder / "poses.txt")
    if len(poses) != evaluation["frames"] or len(run["tracking"]) != len(poses):
        raise ValueError("Saved frame coverage mismatch")
    for pose in poses:
        validate_pose(pose)
    for name, digest in evaluation["source_sha256"].items():
        if hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Saved source archive mismatch: " + name)
    for edge in run.get("independent_stereo_motion", []):
        first, second = edge["previous_frame"], edge["frame"]
        if not 0 <= first < second < len(poses):
            raise ValueError("Invalid stereo motion endpoints")
        validate_pose(np.asarray(edge["measurement"], float))
    return run, evaluation, poses


def motion_evidence(run, poses):
    result = []
    for edge in run.get("independent_stereo_motion", []):
        first, second = edge["previous_frame"], edge["frame"]
        result.append({"previous_frame": first, "frame": second,
                       **pose_difference(np.asarray(edge["measurement"]),
                                         np.linalg.inv(poses[first]) @ poses[second])})
    return {"available": "independent_stereo_motion" in run, "edges": result,
            "violations": [e for e in result if e["translation_m"] > .5 or e["rotation_deg"] > 1.5]}


def compare_caches(root, first_signature, second_signature):
    first, second = Path(root) / first_signature, Path(root) / second_signature
    names_a = {p.name for p in first.glob("*.npz")}
    names_b = {p.name for p in second.glob("*.npz")}
    exact = {k: True for k in ("pixels", "descriptors", "disparity")}
    totals = np.zeros(5, dtype=int)
    depth_change = []
    for name in sorted(names_a & names_b):
        with np.load(first / name, allow_pickle=False) as a, np.load(second / name, allow_pickle=False) as b:
            for key in exact:
                exact[key] &= np.array_equal(a[key], b[key], equal_nan=True)
            if a["points"].shape != b["points"].shape or not np.array_equal(a["pixels"], b["pixels"]):
                raise ValueError("Cache feature indices differ; point comparison is invalid")
            va = np.isfinite(a["points"]).all(axis=1)
            vb = np.isfinite(b["points"]).all(axis=1)
            totals += [len(va), va.sum(), vb.sum(), (va & ~vb).sum(), (vb & ~va).sum()]
            common = va & vb
            depth_change.extend((b["points"][common, 2] - a["points"][common, 2]).tolist())
    return {"paired_blobs": len(names_a & names_b), "unpaired_first": len(names_a - names_b),
            "unpaired_second": len(names_b - names_a), "exact_extraction_arrays": exact,
            "feature_count": int(totals[0]), "first_valid_depth": int(totals[1]),
            "second_valid_depth": int(totals[2]), "lost_depth": int(totals[3]),
            "gained_depth": int(totals[4]), "common_depth_change_m_percentiles":
            dict(zip(("min", "p05", "median", "p95", "max"),
                     np.percentile(depth_change, [0, 5, 50, 95, 100]).tolist())) if depth_change else None,
            "frame_mapping": "Blob names identify image content; this aggregate does not infer frame order."}


def compare(first, second, cache_root=None, reference_poses=None):
    a, ea, pa = load_run(first)
    b, eb, pb = load_run(second)
    if ea["sequence"] != eb["sequence"]:
        raise ValueError("Cannot compare different sequences")
    identities = [e.get("development_identity", {}) for e in (ea, eb)]
    if all(i.get("input") for i in identities) and identities[0]["input"] != identities[1]["input"]:
        raise ValueError("Saved image input identities differ")
    count = min(len(pa), len(pb))
    frames, source_changes = [], []
    for i in range(count):
        ta, tb = a["tracking"][i], b["tracking"][i]
        changed = [key for key in sorted(set(ta) | set(tb)) if ta.get(key) != tb.get(key)]
        row = {"frame": i, "changed_tracking_fields": changed,
               "exported_pose_difference": pose_difference(pa[i], pb[i]),
               "first_pose_source": ta.get("pose_source"), "second_pose_source": tb.get("pose_source"),
               "first_valid_depth": ta.get("valid_stereo_depth"), "second_valid_depth": tb.get("valid_stereo_depth"),
               "first_tracked_landmarks": ta.get("tracked_landmarks"), "second_tracked_landmarks": tb.get("tracked_landmarks")}
        if i:
            row["exported_increment_difference"] = pose_difference(np.linalg.inv(pa[i-1]) @ pa[i],
                                                                     np.linalg.inv(pb[i-1]) @ pb[i])
        frames.append(row)
        if ta.get("pose_source") != tb.get("pose_source"):
            source_changes.append({**row, "second_map_reference_translation_error_m": tb.get("map_reference_translation_error_m"),
                                   "second_map_reference_rotation_error_deg": tb.get("map_reference_rotation_error_deg")})
    measurements = [{(e["previous_frame"], e["frame"]): np.asarray(e["measurement"], float)
                     for e in run.get("independent_stereo_motion", [])} for run in (a, b)]
    measured_changes = [{"previous_frame": edge[0], "frame": edge[1],
                         **pose_difference(measurements[0][edge], measurements[1][edge])}
                        for edge in sorted(set(measurements[0]) & set(measurements[1]))]
    source_a, source_b = ea["source_sha256"], eb["source_sha256"]
    report = {"estimator_invoked": False, "source_archives_verified": True,
              "compared_frames": count, "first_saved_frames": len(pa), "second_saved_frames": len(pb),
              "changed_source_files": [k for k in sorted(set(source_a) | set(source_b)) if source_a.get(k) != source_b.get(k)],
              "first_tracking_divergence": next((f for f in frames if f["changed_tracking_fields"]), None),
              "pose_source_changes": source_changes, "frames": frames,
              "first_keyframe_frames": [k["frame"] for k in a["keyframes"]],
              "second_keyframe_frames": [k["frame"] for k in b["keyframes"]],
              "first_bundle_adjustment": a["bundle_adjustment"], "second_bundle_adjustment": b["bundle_adjustment"],
              "first_motion_agreement": motion_evidence(a, pa), "second_motion_agreement": motion_evidence(b, pb),
              "independent_measurement_differences": measured_changes,
              "first_saved_metrics": ea.get("metrics"), "second_saved_metrics": eb.get("metrics"),
              "first_feature_cache": ea.get("feature_cache"), "second_feature_cache": eb.get("feature_cache"),
              "interpretation_limit": "Exported poses include retroactive corrections; saved metrics alone do not isolate online tracking from BA."}
    if cache_root is not None:
        report["cache_comparison"] = compare_caches(cache_root, ea["feature_cache"]["signature"], eb["feature_cache"]["signature"])
    if reference_poses is not None:
        # Explicit offline evaluation boundary. No estimator is imported or run.
        from metrics import evaluate_trajectory
        gt = load_poses_txt(reference_poses)
        if len(gt) < count:
            raise ValueError("Reference does not cover saved runs")
        report["posthoc_reference_evaluation"] = {"reference_used_for_estimation": False,
            "first": evaluate_trajectory(gt[:count], pa[:count]),
            "second": evaluate_trajectory(gt[:count], pb[:count])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--reference-poses", type=Path, help="Optional offline evaluation only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.first, args.second, args.cache_root, args.reference_poses)
    value = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value, encoding="utf-8")
    else:
        print(value, end="")


if __name__ == "__main__":
    main()
