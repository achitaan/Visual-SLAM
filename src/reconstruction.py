"""Portable sparse-map exports and optional after-run learned-depth reconstruction."""

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2 as cv
from kitti import save_poses_txt
from mapping_geometry import project


def _motion_regularizer_manifest(state):
    """Export frozen raw-fit provenance; invalid unused fit geometry is explicit."""
    def portable(value, allow_invalid=False):
        if isinstance(value, np.ndarray):
            return portable(value.tolist(), allow_invalid)
        if isinstance(value, (tuple, list)):
            return [portable(item, allow_invalid) for item in value]
        if isinstance(value, np.generic):
            return portable(value.item(), allow_invalid)
        if isinstance(value, float) and not np.isfinite(value):
            if not allow_invalid:
                raise ValueError("Nonfinite motion regularizer artifact")
            return None
        return value

    result = []
    for edge, factor in state.stereo_motion_regularizers.items():
        record = {field.name: portable(getattr(factor, field.name), field.name in
                           ("training_source_points", "training_target_points"))
                  for field in fields(factor)}
        record.update(
            previous_frame=int(edge[0]), frame=int(edge[1]),
            invalid_unused_fit_geometry_encoded_as_null=True,
            training_source_depth_valid=np.isfinite(factor.training_source_points).all(axis=1).tolist(),
            training_target_depth_valid=np.isfinite(factor.training_target_points).all(axis=1).tolist(),
        )
        result.append(record)
    return result


def write_ply(path, points, colors=None):
    points = np.asarray(points, float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite point cloud")
    colors = (
        np.full((len(points), 3), 160, np.uint8)
        if colors is None
        else np.asarray(colors, np.uint8)
    )
    if colors.shape != (len(points), 3):
        raise ValueError("Color/point mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(points, colors):
            f.write(
                " ".join([*(f"{v:.8g}" for v in point), *(str(int(v)) for v in color)])
                + "\n"
            )


def export_run(slam, folder, image_paths, image_loader=None, include_images=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = slam.map
    if state.poses:
        save_poses_txt(folder / "poses.txt", state.poses)
    else:
        (folder / "poses.txt").write_text("", encoding="ascii")
    if slam.loop_worker.last_correction_before:
        save_poses_txt(
            folder / "before-final-loop.txt", slam.loop_worker.last_correction_before
        )
        save_poses_txt(
            folder / "after-final-loop.txt", slam.loop_worker.last_correction_after
        )
    landmarks = list(state.landmarks.values())
    write_ply(folder / "sparse.ply", [l.position for l in landmarks])
    # Store only inputs used by the estimator. Reference poses/depth are never included.
    keyframes = []
    images = folder / "images"
    images.mkdir(exist_ok=True)
    for ident, k in state.keyframes.items():
        source = Path(image_paths[k.frame])
        image = (
            image_loader(k.frame)
            if image_loader is not None
            else cv.imread(str(source), cv.IMREAD_UNCHANGED)
        ) if include_images else None
        if include_images and image is None:
            raise ValueError(f"Unreadable reconstruction input: {source.name}")
        name = f"images/{ident:06d}.png"
        if include_images and not cv.imwrite(str(folder / name), image):
            raise OSError("Unable to save reconstruction image")
        observations = [
            {
                "landmark": l.id,
                "pixel": l.observations[ident].pixel.tolist(),
                "position": l.position.tolist(),
            }
            for l in landmarks
            if ident in l.observations
        ]
        keyframes.append(
            {
                "id": ident,
                "frame": k.frame,
                "pose": k.pose.tolist(),
                "image": name if include_images else None,
                "observations": observations,
            }
        )
    payload = {
        "version": 1,
        "revision": state.revision,
        "translation_scale": "metric" if state.metric else "arbitrary",
        "camera_matrix": slam.K.tolist(),
        "keyframes": keyframes,
        "tracking": slam.diagnostics,
        "bundle_adjustment": slam.bundle_reports,
        "sparse_points": len(landmarks),
        "depth_source": "geometry",
        "configuration": slam.config.__dict__,
        "performance_configuration": slam.performance.__dict__,
        "matching_backend": slam.matcher.metadata(),
        "loop_events": slam.loop_worker.events,
        "independent_stereo_motion": [
            {"previous_frame": first, "frame": second, "measurement": measurement.tolist()}
            for (first, second), measurement in state.stereo_motion.items()
        ],
        "verified_loops": [
            {
                "first_keyframe": i,
                "second_keyframe": j,
                "geometry": "stereo" if state.metric else "triangulated_monocular",
                **{
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in measurement.items()
                },
            }
            for (i, j), measurement in slam.loop_worker.verified.items()
        ],
    }
    if getattr(slam.config, "stereo_motion_regularizer", False):
        payload["stereo_motion_regularizers"] = _motion_regularizer_manifest(state)
    (folder / "run.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    preview = {
        "revision": state.revision,
        "translation_scale": payload["translation_scale"],
        "depth_source": "geometry",
        "sparse": slam.map_points_sample(20000),
        "dense": [],
        "trajectory": [p[:3, 3].tolist() for p in state.poses],
    }
    (folder / "preview.json").write_text(
        json.dumps(preview, allow_nan=False), encoding="utf-8"
    )
    return payload


@dataclass
class DepthPrediction:
    depth: np.ndarray
    valid: np.ndarray
    units: str
    provenance: dict


class DepthAnythingProvider:
    def __init__(
        self, source, checkpoint, environment="outdoor", device="auto", input_size=518
    ):
        import sys
        import torch

        source = Path(source).resolve()
        if not (source / "depth_anything_v2/dpt.py").is_file():
            raise ValueError(
                "Model source must be the metric_depth directory from Depth Anything V2"
            )
        sys.path.insert(0, str(source))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.input_size = input_size
        if input_size < 140:
            raise ValueError("Depth input size must be at least 140")
        self.max_depth = 80 if environment == "outdoor" else 20
        self.model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=self.max_depth,
        )
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        try:
            self.model.to(self.device).eval()
        except RuntimeError:
            if device != "auto":
                raise
            self.device = "cpu"
            self.model.to(self.device).eval()
        self.provenance = {
            "model": "Depth Anything V2 Metric Small",
            "environment": environment,
            "training_domain": (
                "Virtual KITTI 2" if environment == "outdoor" else "Hypersim"
            ),
            "checkpoint_sha256": hashlib.sha256(
                Path(checkpoint).read_bytes()
            ).hexdigest(),
            "device": self.device,
            "units": "predicted_meters",
            "input_size": input_size,
        }

    def predict(self, image):
        import torch
        from depth_anything_v2.util.transform import (
            Resize,
            NormalizeImage,
            PrepareForNet,
        )
        from torchvision.transforms import Compose

        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image
        transform = Compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        tensor = (
            torch.from_numpy(
                transform({"image": cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255.0})[
                    "image"
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            result = self.model(tensor)
            depth = (
                torch.nn.functional.interpolate(
                    result[:, None],
                    image.shape[:2],
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]
                .cpu()
                .numpy()
            )
        depth = np.asarray(depth, np.float32)
        if depth.shape != image.shape[:2]:
            raise ValueError("Depth prediction must match image coordinates")
        return DepthPrediction(
            depth,
            np.isfinite(depth) & (depth > 0) & (depth < self.max_depth),
            "predicted_meters",
            dict(self.provenance),
        )


def fit_depth_scale(predicted, reference, min_points=15):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > 0)
        & (reference > 0)
    )
    if np.count_nonzero(valid) < min_points:
        return None
    ratios = reference[valid] / predicted[valid]
    scale = float(np.median(ratios))
    deviation = np.abs(np.log(ratios / scale))
    good = deviation < 0.25
    if good.sum() < min_points or good.mean() < 0.6:
        return None
    return float(np.median(ratios[good]))


def fuse_reconstruction(
    run_folder, provider, output, voxel=0.1, max_points=500000, keyframe_stride=1
):
    run_folder, output = Path(run_folder), Path(output)
    run = json.loads((run_folder / "run.json").read_text())
    if voxel <= 0 or max_points < 1 or keyframe_stride < 1:
        raise ValueError("Invalid fusion limits")
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.array(run["camera_matrix"])
    voxels = {}
    reports = []
    previous = None
    discarded_voxels = 0
    for k in run["keyframes"][::keyframe_stride]:
        image = cv.imread(str(run_folder / k["image"]), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable keyframe image")
        prediction = provider.predict(image)
        depth = prediction.depth
        if depth.shape != image.shape[:2] or prediction.valid.shape != depth.shape:
            raise ValueError("Depth coordinate mismatch")
        if prediction.valid.dtype != np.bool_:
            raise ValueError("Depth validity mask must be boolean")
        pose = np.array(k["pose"])
        observations = k["observations"]
        pixels = np.array([o["pixel"] for o in observations]).reshape(-1, 2)
        world = np.array([o["position"] for o in observations]).reshape(-1, 3)
        _, z = project(world, pose, matrix)
        uv = np.rint(pixels).astype(int)
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < depth.shape[1])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < depth.shape[0])
        )
        sampled = np.full(len(uv), np.nan)
        inside_ids = np.flatnonzero(inside)
        inside_ids = inside_ids[prediction.valid[uv[inside_ids, 1], uv[inside_ids, 0]]]
        sampled[inside_ids] = depth[uv[inside_ids, 1], uv[inside_ids, 0]]
        scale = fit_depth_scale(sampled, z)
        np.savez_compressed(
            output / f'depth-{k["id"]:06d}.npz',
            depth=depth,
            valid=prediction.valid,
            scale_to_map=scale if scale is not None else np.nan,
        )
        if scale is None:
            reports.append(
                {
                    "keyframe": k["id"],
                    "status": "omitted",
                    "reason": "insufficient_sparse_depth_agreement",
                }
            )
            continue
        scaled = depth * scale
        gy, gx = np.gradient(scaled)
        valid = (
            prediction.valid
            & np.isfinite(scaled)
            & (scaled > 0)
            & (np.hypot(gx, gy) < 0.1 * np.maximum(scaled, 1e-6))
        )
        ys, xs = np.mgrid[: depth.shape[0] : 2, : depth.shape[1] : 2]
        xs, ys = xs.ravel(), ys.ravel()
        keep = valid[ys, xs]
        xs, ys = xs[keep], ys[keep]
        rays = np.c_[xs, ys, np.ones(len(xs))] @ np.linalg.inv(matrix).T
        camera = rays * scaled[ys, xs, None]
        points = camera @ pose[:3, :3].T + pose[:3, 3]
        colors = image[ys, xs, ::-1]
        if previous is not None:
            old_pose, old_depth, old_valid = previous
            projected, old_z = project(points, old_pose, matrix)
            u = np.rint(projected).astype(int)
            overlap = (
                (old_z > 0)
                & (u[:, 0] >= 0)
                & (u[:, 0] < old_depth.shape[1])
                & (u[:, 1] >= 0)
                & (u[:, 1] < old_depth.shape[0])
            )
            ids = np.flatnonzero(overlap)
            observed = old_depth[u[ids, 1], u[ids, 0]]
            checked = old_valid[u[ids, 1], u[ids, 0]]
            consistent = np.ones(len(points), bool)
            consistent[ids] = checked & (
                np.abs(observed - old_z[ids]) < 0.1 * np.maximum(observed, 1e-6)
            )
            points, colors = points[consistent], colors[consistent]
        for point, color in zip(points, colors):
            key = tuple(np.floor(point / voxel).astype(int))
            if key in voxels:
                old, count, old_color = voxels[key]
                voxels[key] = (old + point, count + 1, old_color + color.astype(float))
            elif len(voxels) < max_points:
                voxels[key] = (point.copy(), 1, color.astype(float))
            else:
                discarded_voxels += 1
        reports.append(
            {
                "keyframe": k["id"],
                "status": "fused",
                "scale_to_map": scale,
                "accepted_points": len(points),
            }
        )
        previous = (pose, scaled, valid)
    points = np.array([p / n for p, n, _ in voxels.values()]).reshape(-1, 3)
    colors = np.array([c / n for _, n, c in voxels.values()], np.uint8).reshape(-1, 3)
    write_ply(output / "dense.ply", points, colors)
    manifest = {
        "map_revision": run["revision"],
        "translation_scale": run["translation_scale"],
        "depth_source": "learned_reconstruction_only",
        "provider": provider.provenance,
        "points": len(points),
        "point_limit": max_points,
        "keyframe_stride": keyframe_stride,
        "points_discarded_at_limit": discarded_voxels,
        "voxel_size_map_units": voxel,
        "keyframes": reports,
        "source_run_sha256": hashlib.sha256(
            (run_folder / "run.json").read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    preview = json.loads((run_folder / "preview.json").read_text())
    preview.update(
        dense=points[:: max(1, len(points) // 20000)][:20000].tolist(),
        dense_colors=colors[:: max(1, len(points) // 20000)][:20000].tolist(),
        depth_source="learned_reconstruction_only",
        depth_model=provider.provenance.get("model", "unspecified"),
        training_domain=provider.provenance.get("training_domain", "unspecified"),
        checkpoint_sha256=provider.provenance.get("checkpoint_sha256"),
    )
    (output / "preview.json").write_text(
        json.dumps(preview, allow_nan=False), encoding="utf-8"
    )
    return manifest
