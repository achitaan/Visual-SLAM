"""Authoritative persistent map. Estimator inputs deliberately exclude reference poses."""

from dataclasses import dataclass, field
from threading import RLock
import numpy as np
from kitti import validate_pose


@dataclass
class Observation:
    pixel: np.ndarray
    right_u: float | None = None


@dataclass
class Landmark:
    id: int
    position: np.ndarray
    descriptor: np.ndarray
    anchor: int
    observations: dict[int, Observation] = field(default_factory=dict)
    misses: int = 0


@dataclass
class MappingKeyframe:
    id: int
    frame: int
    pose: np.ndarray
    pixels: np.ndarray
    descriptors: np.ndarray
    landmark_ids: np.ndarray
    image: np.ndarray | None = None
    depth_points: np.ndarray | None = None
    image_size: tuple[int, int] | None = None


class MapState:
    def __init__(self, metric=False):
        self.metric = metric
        self.keyframes = {}
        self.landmarks = {}
        self.poses = []
        self.pose_anchors = []
        self.relative_poses = []
        self.statuses = []
        self.stereo_motion = {}
        self.revision = 0
        self.geometry_revision = 0
        self.next_landmark = 0
        self.lock = RLock()

    def add_stereo_motion(self, previous, current, measurement):
        """Retain independently verified metric motion between accepted frames."""
        validate_pose(measurement)
        with self.lock:
            if (not self.metric or not 0 <= previous < current < len(self.poses)
                    or self.statuses[previous] not in ('tracking', 'relocalized')
                    or self.statuses[current] not in ('tracking', 'relocalized')):
                raise ValueError('Stereo motion requires two accepted metric frames')
            key = (previous, current)
            if key in self.stereo_motion:
                raise ValueError('Stereo motion already recorded')
            # Append-only evidence accompanies recorded frames, without changing geometry.
            self.stereo_motion[key] = measurement.copy()

    def add_landmark(self, position, descriptor, anchor, observations):
        position = np.asarray(position, float).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("Nonfinite landmark")
        ident = self.next_landmark
        self.next_landmark += 1
        self.landmarks[ident] = Landmark(
            ident, position.copy(), descriptor.copy(), anchor, observations
        )
        return ident

    def record(self, pose, status, anchor=None):
        validate_pose(pose)
        with self.lock:
            self.poses.append(pose.copy())
            self.statuses.append(status)
            self.pose_anchors.append(anchor)
            self.relative_poses.append(
                np.linalg.inv(self.keyframes[anchor].pose) @ pose
                if anchor is not None
                else pose.copy()
            )

    def apply_corrections(self, expected_revision, corrected, scales=None, *, propagate_landmarks=True, landmark_updates=None, frame_updates=None):
        """Commit a complete correction atomically; reject stale snapshots and moved origin."""
        with self.lock:
            if self.revision != expected_revision or set(corrected) != set(
                self.keyframes
            ):
                return False
            scales = scales or {i: 1.0 for i in corrected}
            for ident, pose in corrected.items():
                validate_pose(pose)
                if (
                    ident not in scales
                    or not np.isfinite(scales[ident])
                    or scales[ident] <= 0
                ):
                    raise ValueError("Invalid correction scale")
            if (
                not np.allclose(corrected[0], self.keyframes[0].pose, atol=1e-8)
                or abs(scales[0] - 1) > 1e-8
            ):
                raise ValueError("Origin must remain fixed")
            # Joint local solves may optimize an accepted intermediate camera.
            # Validate all overrides before changing any landmark or pose. A
            # keyframe has one authoritative pose in `corrected`, never two.
            intermediate = {}
            keyframe_frames = {keyframe.frame for keyframe in self.keyframes.values()}
            if frame_updates is not None:
                if not (len(self.poses) == len(self.statuses) == len(self.pose_anchors)
                        == len(self.relative_poses)):
                    raise ValueError('Inconsistent intermediate frame state')
                for frame, pose in frame_updates.items():
                    if (not isinstance(frame, (int, np.integer))
                            or isinstance(frame, (bool, np.bool_))
                            or not 0 < frame < min(len(self.poses), len(self.statuses), len(self.pose_anchors))
                            or frame in keyframe_frames
                            or self.statuses[frame] not in ('tracking', 'relocalized', 'accepted')
                            or self.pose_anchors[frame] not in corrected):
                        raise ValueError("Invalid intermediate frame correction")
                    raw = np.asarray(pose)
                    if np.iscomplexobj(raw):
                        raise ValueError("Intermediate frame correction must be real")
                    validate_pose(raw)
                    intermediate[int(frame)] = np.asarray(raw, float).copy()
            corrections = {}
            for ident, keyframe in self.keyframes.items():
                old = keyframe.pose
                rotation = corrected[ident][:3, :3] @ old[:3, :3].T
                scale = scales[ident]
                translation = corrected[ident][:3, 3] - scale * rotation @ old[:3, 3]
                corrections[ident] = (rotation, scale, translation)
            positions = {}
            if landmark_updates is not None and not set(landmark_updates).issubset(self.landmarks):
                raise ValueError("Unknown corrected landmark")
            for ident, landmark in self.landmarks.items():
                if landmark_updates is not None and ident in landmark_updates:
                    positions[ident] = np.asarray(landmark_updates[ident], float).reshape(3).copy()
                elif propagate_landmarks:
                    rotation, scale, translation = corrections[landmark.anchor]
                    positions[ident] = scale * rotation @ landmark.position + translation
                else:
                    positions[ident] = landmark.position.copy()
                if not np.isfinite(positions[ident]).all():
                    raise ValueError("Nonfinite corrected landmark")
            poses = []
            for frame, (pose, anchor) in enumerate(zip(self.poses, self.pose_anchors)):
                updated = pose.copy()
                if anchor is not None:
                    rotation, scale, translation = corrections[anchor]
                    updated[:3, :3] = rotation @ pose[:3, :3]
                    updated[:3, 3] = scale * rotation @ pose[:3, 3] + translation
                if frame in intermediate:
                    updated = intermediate[frame].copy()
                validate_pose(updated)
                poses.append(updated)
            for ident, position in positions.items():
                self.landmarks[ident].position = position
            for ident, pose in corrected.items():
                self.keyframes[ident].pose = pose.copy()
                if not self.metric and self.keyframes[ident].depth_points is not None:
                    self.keyframes[ident].depth_points *= scales[ident]
            self.poses = poses
            self.relative_poses = [
                np.linalg.inv(self.keyframes[a].pose) @ p if a is not None else p.copy()
                for p, a in zip(poses, self.pose_anchors)
            ]
            self.revision += 1
            self.geometry_revision += 1
            return True

    def apply_snapshot_corrections(self, revision, geometry_revision, original, corrected, scales=None):
        """Allow append-only progress, but never rebase across changed snapshot geometry."""
        with self.lock:
            ids = sorted(original)
            current = sorted(self.keyframes)
            if (self.geometry_revision != geometry_revision or not ids
                    or current[:len(ids)] != ids or set(corrected) != set(original)):
                return False
            if any(not np.array_equal(self.keyframes[i].pose, original[i]) for i in ids):
                return False
            if self.revision != revision and len(current) == len(ids):
                return False
            if isinstance(scales,dict) and set(scales)!=set(ids):
                return False
            values = (np.ones(len(ids)) if scales is None else
                      np.array([scales[i] for i in ids],float) if isinstance(scales,dict) else np.asarray(scales,float))
            if values.shape != (len(ids),) or not np.isfinite(values).all() or np.any(values <= 0):
                return False
            expanded = {i:p.copy() for i,p in corrected.items()}
            last = ids[-1]
            rotation = corrected[last][:3,:3] @ original[last][:3,:3].T
            scale = values[-1]
            translation = corrected[last][:3,3] - scale * rotation @ original[last][:3,3]
            for ident in current[len(ids):]:
                pose = self.keyframes[ident].pose.copy()
                pose[:3,:3] = rotation @ pose[:3,:3]
                pose[:3,3] = scale * rotation @ pose[:3,3] + translation
                expanded[ident] = pose
            expanded_scales = dict(zip(current, np.r_[values, np.full(len(current)-len(ids), scale)]))
            return self.apply_corrections(self.revision, expanded, expanded_scales)
