"""Authoritative persistent map. Estimator inputs deliberately exclude reference poses."""

from dataclasses import dataclass, field
from math import isfinite as _isfinite
from threading import RLock
import numpy as np
from kitti import validate_pose
from map_index import IncrementalMapIndex


@dataclass
class Observation:
    pixel: np.ndarray
    right_u: float | None = None
    octave: int = 0
    sigma_disparity: float = 1.0


@dataclass
class Landmark:
    id: int
    position: np.ndarray
    descriptor: np.ndarray
    anchor: int
    observations: dict[int, Observation] = field(default_factory=dict)
    _misses: int = field(default=0, init=False, repr=False, compare=False)
    misses: int = 0
    octave: int = 0
    is_far: bool = False
    _high_miss_set: set[int] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        # Preserve dataclass field init semantics while syncing _high_miss_set if attached later.
        self._misses = int(self._misses)


def _lm_get_misses(self) -> int:
    return self._misses


def _lm_set_misses(self, value: int) -> None:
    self._misses = value
    hms = getattr(self, "_high_miss_set", None)
    if hms is not None:
        if value >= 5:
            hms.add(self.id)
        else:
            hms.discard(self.id)


Landmark.misses = property(_lm_get_misses, _lm_set_misses)  # type: ignore[assignment]


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
    octaves: np.ndarray | None = None
    landmark_ids_version: int = 0


class MapState:
    def __init__(self, metric=False):
        self.metric = metric
        self.keyframes = {}
        self.landmarks = {}
        self.landmark_ids_version = 0
        self.poses = []
        self.pose_anchors = []
        self.relative_poses = []
        self.statuses = []
        self.stereo_motion = {}
        # Actual non-keyframe image observations retained for a later local BA.
        # This registry is separate from Landmark.observations, whose keys are
        # exclusively persistent keyframe IDs.
        self.retained_source_observations = {}
        self.revision = 0
        self.geometry_revision = 0
        self.pose_revision = 0
        self.next_landmark = 0
        self.lock = RLock()
        self.index = IncrementalMapIndex()
        self._high_miss_landmarks: set[int] = set()
        self._high_miss_tracked_count: int = 0
        self._stereo_motion_by_anchor: dict[int, list[tuple[int, int]]] = {}

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
            a0 = self.pose_anchors[previous]
            a1 = self.pose_anchors[current]
            if a0 != a1:
                if a0 is not None:
                    self._stereo_motion_by_anchor.setdefault(a0, []).append(key)
                if a1 is not None:
                    self._stereo_motion_by_anchor.setdefault(a1, []).append(key)

    def add_landmark(self, position, descriptor, anchor, observations, *, octave=0, is_far=False):
        position = np.asarray(position, float).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("Nonfinite landmark")
        ident = self.next_landmark
        self.next_landmark += 1
        landmark = Landmark(
            ident, position.copy(), descriptor.copy(), anchor, observations,
            octave=int(octave), is_far=bool(is_far),
        )
        landmark._high_miss_set = self._high_miss_landmarks
        if landmark.misses >= 5:
            self._high_miss_landmarks.add(ident)
        self.landmarks[ident] = landmark
        self._high_miss_tracked_count += 1
        self.index.on_landmark_added(landmark)
        return ident

    def add_landmarks_batch(
        self,
        positions: np.ndarray,
        descriptors: np.ndarray,
        anchor: int,
        pixels: np.ndarray,
        right_us: np.ndarray,
        octaves: list[int] | np.ndarray,
        *,
        is_far: bool = False,
    ) -> list[int]:
        """Bulk-insert verified single-observation landmarks with bit-exact ID order."""
        n = len(positions)
        if n == 0:
            return []
        pos_arr = np.ascontiguousarray(positions, dtype=np.float64).reshape(n, 3)
        if not np.isfinite(pos_arr).all():
            raise ValueError("Nonfinite landmark")
        desc_arr = np.ascontiguousarray(descriptors)
        px_arr = np.ascontiguousarray(pixels, dtype=np.float32).reshape(n, 2)
        px_finite = np.isfinite(px_arr).all(axis=1).tolist()
        px_bytes = px_arr.tobytes()
        px_list = px_arr.tolist()
        ru_list = np.asarray(right_us, dtype=np.float64).tolist()
        oct_list = octaves.tolist() if isinstance(octaves, np.ndarray) else list(octaves)

        from owned_observation import OwnedObservation, _UNPACK_2I
        anchor_id = int(anchor)
        start_id = self.next_landmark
        self.next_landmark = start_id + n
        ids_list = list(range(start_id, start_id + n))
        pid_i32_list: list[tuple[int, int] | None] = [None] * n
        landmarks = self.landmarks
        hms = self._high_miss_landmarks
        is_far_b = bool(is_far)
        _from_f32 = OwnedObservation.from_f32_bytes

        for k in range(n):
            ident = start_id + k
            if px_finite[k]:
                buf = px_bytes[k * 8 : (k + 1) * 8]
                pid = _UNPACK_2I(buf)
                pid_i32_list[k] = pid
                x, y = px_list[k]
                obs = _from_f32(x, y, buf, pid, ru_list[k])
            else:
                obs = OwnedObservation(px_arr[k], ru_list[k])
            lm = Landmark(
                ident,
                pos_arr[k],
                desc_arr[k],
                anchor_id,
                {anchor_id: obs},
                octave=int(oct_list[k]),
                is_far=is_far_b,
            )
            lm._high_miss_set = hms
            landmarks[ident] = lm

        self._high_miss_tracked_count += n
        self.index.on_landmarks_added_batch(
            ids_list, pos_arr, anchor_id, pid_i32_list, is_far=is_far_b
        )
        return ids_list

    def high_miss_landmarks(self, min_misses: int = 5) -> dict[int, Landmark]:
        """Return landmarks with misses >= min_misses in O(|dropped|) when tracked."""
        if (
            min_misses == 5
            and self._high_miss_tracked_count == len(self.landmarks)
            and len(self.index) == len(self.landmarks)
        ):
            if not self._high_miss_landmarks:
                return {}
            out: dict[int, Landmark] = {}
            for ident in sorted(self._high_miss_landmarks):
                lm = self.landmarks.get(ident)
                if lm is not None and lm.misses >= 5:
                    out[ident] = lm
                else:
                    self._high_miss_landmarks.discard(ident)
            return out
        return {i: l for i, l in self.landmarks.items() if l.misses >= min_misses}

    def add_observation(self, landmark_id, keyframe_id, observation):
        with self.lock:
            landmark = self.landmarks.get(landmark_id)
            if landmark is None:
                return
            previous = landmark.observations.get(keyframe_id)
            if previous is not None:
                self.index.on_observation_removed(landmark_id, keyframe_id, previous)
            landmark.observations[keyframe_id] = observation
            self.index.on_observation_added(landmark_id, observation, keyframe_id=keyframe_id)

    def remove_observation(self, landmark_id, keyframe_id):
        with self.lock:
            landmark = self.landmarks.get(landmark_id)
            if landmark is None:
                return None
            removed = landmark.observations.pop(keyframe_id, None)
            if removed is not None:
                self.index.on_observation_removed(landmark_id, keyframe_id, removed)
            return removed

    def remove_landmark(self, landmark_id):
        with self.lock:
            removed = self.landmarks.pop(landmark_id, None)
            if removed is not None:
                if removed._high_miss_set is self._high_miss_landmarks:
                    self._high_miss_tracked_count = max(0, self._high_miss_tracked_count - 1)
                removed._high_miss_set = None
                self._high_miss_landmarks.discard(landmark_id)
                self.index.on_landmark_removed(landmark_id)
            return removed

    def record(self, pose, status, anchor=None):
        validate_pose(pose)
        with self.lock:
            frame_idx = len(self.poses)
            self.poses.append(pose.copy())
            self.statuses.append(status)
            self.pose_anchors.append(anchor)
            if anchor is not None:
                fba = getattr(self, "_frames_by_anchor", None)
                if fba is None:
                    fba = {}
                    self._frames_by_anchor = fba
                fba.setdefault(anchor, []).append(frame_idx)
            self.relative_poses.append(
                np.linalg.inv(self.keyframes[anchor].pose) @ pose
                if anchor is not None
                else pose.copy()
            )

    def apply_corrections(self, expected_revision, corrected, scales=None, *, propagate_landmarks=True, landmark_updates=None, frame_updates=None, retained_source_updates=None, retained_source_limits=None):
        """Commit a complete correction atomically; reject stale snapshots and moved origin."""
        with self.lock:
            if self.revision != expected_revision or set(corrected) != set(
                self.keyframes
            ):
                return False
            scales = scales or {i: 1.0 for i in corrected}
            for ident, pose in corrected.items():
                old_pose = self.keyframes[ident].pose
                if pose is not old_pose and not (
                    isinstance(pose, np.ndarray)
                    and pose.dtype == np.float64
                    and isinstance(old_pose, np.ndarray)
                    and old_pose.dtype == np.float64
                    and np.array_equal(pose, old_pose)
                ):
                    validate_pose(pose)
                if (
                    ident not in scales
                    or not np.isfinite(scales[ident])
                    or scales[ident] <= 0
                ):
                    raise ValueError("Invalid correction scale")
            if (
                corrected[0] is not self.keyframes[0].pose
                and not np.allclose(corrected[0], self.keyframes[0].pose, atol=1e-8)
            ) or abs(scales[0] - 1) > 1e-8:
                raise ValueError("Origin must remain fixed")
            # Joint local solves may optimize an accepted intermediate camera.
            # Validate all overrides before changing any landmark or pose. A
            # keyframe has one authoritative pose in `corrected`, never two.
            intermediate = {}
            if frame_updates is not None:
                keyframe_frames = {keyframe.frame for keyframe in self.keyframes.values()}
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
            moved_kfs = set()
            corrections = {}
            for ident, keyframe in self.keyframes.items():
                old = keyframe.pose
                new_pose = corrected[ident]
                scale = scales[ident]
                if scale == 1.0 and (new_pose is old or np.array_equal(new_pose, old)):
                    continue
                moved_kfs.add(ident)
                rotation = new_pose[:3, :3] @ old[:3, :3].T
                translation = new_pose[:3, 3] - scale * rotation @ old[:3, 3]
                corrections[ident] = (rotation, scale, translation)
            positions = {}
            bulk_anchor_updates = None
            if landmark_updates is not None and not all(i in self.landmarks for i in landmark_updates):
                raise ValueError("Unknown corrected landmark")
            if not propagate_landmarks:
                if landmark_updates is not None:
                    if len(landmark_updates) >= 32 and len(self.index) == len(self.landmarks):
                        id_list = list(landmark_updates.keys())
                        id_to_slot = self.index._id_to_slot
                        slots = [id_to_slot[i] for i in id_list]
                        new_arr = np.asarray(list(landmark_updates.values()), dtype=np.float64)
                        if new_arr.shape != (len(id_list), 3):
                            new_arr = np.vstack([np.asarray(v, dtype=np.float64).reshape(3) for v in landmark_updates.values()])
                        else:
                            new_arr = np.ascontiguousarray(new_arr, dtype=np.float64).copy()
                        if not np.isfinite(new_arr).all():
                            raise ValueError("Nonfinite corrected landmark")
                        bulk_anchor_updates = [(id_list, slots, new_arr)]
                    else:
                        for ident, raw_pos in landmark_updates.items():
                            if (type(raw_pos) is np.ndarray and raw_pos.dtype == np.float64
                                    and raw_pos.shape == (3,)):
                                pos = raw_pos.copy()
                            else:
                                pos = np.asarray(raw_pos, float).reshape(3).copy()
                            x, y, z = pos.tolist()
                            if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
                                raise ValueError("Nonfinite corrected landmark")
                            positions[ident] = pos
            elif (
                len(self.index) == len(self.landmarks)
                and (moved_kfs or landmark_updates)
            ):
                bulk_anchor_updates = []
                anc_map = self.index._anchor_landmarks
                id_to_slot = self.index._id_to_slot
                idx_pos = self.index._positions
                if landmark_updates:
                    upd_ids = []
                    upd_slots = []
                    upd_rows = []
                    for ident, raw_pos in landmark_updates.items():
                        if (type(raw_pos) is np.ndarray and raw_pos.dtype == np.float64
                                and raw_pos.shape == (3,)):
                            pos = raw_pos.copy()
                        else:
                            pos = np.asarray(raw_pos, float).reshape(3).copy()
                        x, y, z = pos.tolist()
                        if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
                            raise ValueError("Nonfinite corrected landmark")
                        upd_ids.append(ident)
                        upd_slots.append(id_to_slot[ident])
                        upd_rows.append(pos)
                    if upd_ids:
                        bulk_anchor_updates.append(
                            (upd_ids, upd_slots, np.asarray(upd_rows, dtype=np.float64))
                        )
                for anchor_id in moved_kfs:
                    lm_set = anc_map.get(anchor_id)
                    if not lm_set:
                        continue
                    if landmark_updates:
                        id_list = [i for i in lm_set if i not in landmark_updates]
                    else:
                        id_list = list(lm_set)
                    if not id_list:
                        continue
                    slots = [id_to_slot[i] for i in id_list]
                    rotation, scale, translation = corrections[anchor_id]
                    new_arr = (scale * (idx_pos[slots] @ rotation.T)) + translation
                    if not np.isfinite(new_arr).all():
                        raise ValueError("Nonfinite corrected landmark")
                    bulk_anchor_updates.append((id_list, slots, new_arr))
            else:
                if len(self.index) == len(self.landmarks) and len(moved_kfs) < len(self.keyframes):
                    affected_ids = self.index.landmarks_for_anchors(moved_kfs)
                    if landmark_updates is not None:
                        affected_ids = affected_ids | set(landmark_updates)
                    by_anchor: dict[int, list[int]] = {}
                    for ident in affected_ids:
                        landmark = self.landmarks.get(ident)
                        if landmark is None:
                            continue
                        if landmark_updates is not None and ident in landmark_updates:
                            raw_pos = landmark_updates[ident]
                            if (type(raw_pos) is np.ndarray and raw_pos.dtype == np.float64
                                    and raw_pos.shape == (3,)):
                                pos = raw_pos.copy()
                            else:
                                pos = np.asarray(raw_pos, float).reshape(3).copy()
                            x, y, z = pos.tolist()
                            if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
                                raise ValueError("Nonfinite corrected landmark")
                            positions[ident] = pos
                        else:
                            by_anchor.setdefault(landmark.anchor, []).append(ident)
                    for anchor_id, id_list in by_anchor.items():
                        rotation, scale, translation = corrections[anchor_id]
                        if len(id_list) >= 8:
                            pts_arr = np.array([self.landmarks[i].position for i in id_list], dtype=np.float64)
                            new_arr = (scale * (pts_arr @ rotation.T)) + translation
                            if not np.isfinite(new_arr).all():
                                raise ValueError("Nonfinite corrected landmark")
                            for i, row in zip(id_list, new_arr):
                                positions[i] = row
                        else:
                            for i in id_list:
                                pos = scale * rotation @ self.landmarks[i].position + translation
                                x, y, z = pos.tolist()
                                if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
                                    raise ValueError("Nonfinite corrected landmark")
                                positions[i] = pos
                else:
                    for ident, landmark in self.landmarks.items():
                        if landmark_updates is not None and ident in landmark_updates:
                            raw_pos = landmark_updates[ident]
                            pos = (raw_pos.copy() if (type(raw_pos) is np.ndarray and raw_pos.dtype == np.float64 and raw_pos.shape == (3,))
                                   else np.asarray(raw_pos, float).reshape(3).copy())
                        elif landmark.anchor in moved_kfs:
                            rotation, scale, translation = corrections[landmark.anchor]
                            pos = scale * rotation @ landmark.position + translation
                        else:
                            continue
                        x, y, z = pos.tolist()
                        if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
                            raise ValueError("Nonfinite corrected landmark")
                        positions[ident] = pos
            poses = list(self.poses)
            moved_frames = []
            fba = getattr(self, "_frames_by_anchor", None)
            if (fba is not None and not intermediate
                    and len(moved_kfs) < len(self.keyframes)
                    and len(self.poses) == len(self.pose_anchors)):
                for anchor in moved_kfs:
                    f_list = fba.get(anchor)
                    if not f_list:
                        continue
                    rotation, scale, translation = corrections[anchor]
                    for frame in f_list:
                        if frame >= len(poses) or self.pose_anchors[frame] != anchor:
                            continue
                        pose = poses[frame]
                        updated = pose.copy()
                        updated[:3, :3] = rotation @ pose[:3, :3]
                        updated[:3, 3] = scale * rotation @ pose[:3, 3] + translation
                        validate_pose(updated)
                        poses[frame] = updated
                        moved_frames.append(frame)
            else:
                for frame, (pose, anchor) in enumerate(zip(self.poses, self.pose_anchors)):
                    if frame in intermediate:
                        updated = intermediate[frame].copy()
                        validate_pose(updated)
                        poses[frame] = updated
                        moved_frames.append(frame)
                    elif anchor in moved_kfs:
                        rotation, scale, translation = corrections[anchor]
                        updated = pose.copy()
                        updated[:3, :3] = rotation @ pose[:3, :3]
                        updated[:3, 3] = scale * rotation @ pose[:3, 3] + translation
                        validate_pose(updated)
                        poses[frame] = updated
                        moved_frames.append(frame)
            retained_registry = None
            if retained_source_updates is not None:
                if not isinstance(retained_source_updates, dict):
                    raise ValueError("Retained source observations must be a complete mapping")
                if (not isinstance(retained_source_limits, dict)
                        or set(retained_source_limits) != {"max_frames", "max_rows_per_frame"}):
                    raise ValueError("Retained source observation limits are required")
                max_retained_frames = retained_source_limits.get("max_frames")
                max_retained_rows = retained_source_limits.get("max_rows_per_frame")
                if any(not isinstance(value, (int, np.integer))
                       or isinstance(value, (bool, np.bool_)) or int(value) <= 0
                       for value in (max_retained_frames, max_retained_rows)):
                    raise ValueError("Invalid retained source observation limits")
                max_retained_frames = int(max_retained_frames)
                max_retained_rows = int(max_retained_rows)
                if len(retained_source_updates) > max_retained_frames:
                    raise ValueError("Retained source frame limit exceeded")
                retained_registry = {}
                physical_owners = set()
                for raw_frame, raw_record in retained_source_updates.items():
                    if (not isinstance(raw_frame, (int, np.integer))
                            or isinstance(raw_frame, (bool, np.bool_))):
                        raise ValueError("Invalid retained source frame ID")
                    frame = int(raw_frame)
                    if (not isinstance(raw_record, dict)
                            or not isinstance(raw_record.get("frame_id"), (int, np.integer))
                            or isinstance(raw_record.get("frame_id"), (bool, np.bool_))
                            or int(raw_record["frame_id"]) != frame
                            or not isinstance(raw_record.get("calibration_identity"), str)
                            or not raw_record.get("calibration_identity")
                            or raw_record.get("measurement_role") != "tracking_fit_consumed"):
                        raise ValueError("Invalid retained source record metadata")
                    for revision_field, maximum in (
                        ("accepted_revision", self.revision + 1),
                        ("accepted_geometry_revision", self.geometry_revision + 1),
                    ):
                        value = raw_record.get(revision_field)
                        if (not isinstance(value, (int, np.integer))
                                or isinstance(value, (bool, np.bool_))
                                or value < 0 or value > maximum):
                            raise ValueError("Invalid retained source record epoch")
                    if (frame < 0 or frame >= len(poses) or frame >= len(self.statuses)
                            or frame >= len(self.pose_anchors)
                            or self.statuses[frame] not in ("tracking", "relocalized", "accepted")):
                        raise ValueError("Retained source frame is unavailable")
                    anchor = raw_record.get("anchor_keyframe_id")
                    if (not isinstance(anchor, (int, np.integer))
                            or isinstance(anchor, (bool, np.bool_))
                            or int(anchor) not in corrected
                            or self.pose_anchors[frame] != int(anchor)):
                        raise ValueError("Invalid retained source anchor")
                    raw_relative = np.asarray(raw_record.get("relative_pose"))
                    if (np.iscomplexobj(raw_relative)
                            or not np.issubdtype(raw_relative.dtype, np.number)):
                        raise ValueError("Retained source relative pose must be real numeric data")
                    try:
                        validate_pose(raw_relative)
                    except (TypeError, ValueError):
                        raise ValueError("Invalid retained source relative pose")
                    relative = np.asarray(raw_relative, dtype=float).copy()
                    candidate_source = corrected[int(anchor)] @ relative
                    if not np.allclose(candidate_source, poses[frame], rtol=0., atol=1e-7):
                        raise ValueError("Retained source pose disagrees with propagated frame")
                    rows = raw_record.get("rows")
                    if (not isinstance(rows, (list, tuple))
                            or len(rows) > max_retained_rows):
                        raise ValueError("Invalid retained source rows")
                    copied_rows = []
                    seen_landmarks = set()
                    for raw_row in rows:
                        if not isinstance(raw_row, dict):
                            raise ValueError("Invalid retained source row")
                        ident = raw_row.get("landmark_id")
                        if (not isinstance(ident, (int, np.integer))
                                or isinstance(ident, (bool, np.bool_))):
                            raise ValueError("Invalid retained source landmark ID")
                        ident = int(ident)
                        if ident not in self.landmarks or ident in seen_landmarks:
                            raise ValueError("Unknown or duplicate retained source landmark")
                        raw_pixel = np.asarray(raw_row.get("pixel_float32"))
                        if (np.iscomplexobj(raw_pixel)
                                or not np.issubdtype(raw_pixel.dtype, np.number)
                                or raw_pixel.shape != (2,)
                                or not np.isfinite(raw_pixel).all()):
                            raise ValueError("Invalid retained source pixel")
                        pixel = np.asarray(raw_pixel, dtype=np.float32)
                        if not np.isfinite(pixel).all():
                            raise ValueError("Invalid retained source pixel")
                        pixel_key = tuple(0.0 if float(x) == 0.0 else float(x)
                                          for x in pixel)
                        physical_key = (frame, pixel_key)
                        if physical_key in physical_owners:
                            raise ValueError("Duplicate retained source physical pixel")
                        physical_owners.add(physical_key)
                        seen_landmarks.add(ident)
                        copied_rows.append({"landmark_id": ident,
                                            "pixel_float32": pixel.copy()})
                    retained_registry[frame] = {
                        "frame_id": frame,
                        "calibration_identity": raw_record["calibration_identity"],
                        "measurement_role": "tracking_fit_consumed",
                        "accepted_revision": int(raw_record["accepted_revision"]),
                        "accepted_geometry_revision": int(
                            raw_record["accepted_geometry_revision"]
                        ),
                        "anchor_keyframe_id": int(anchor),
                        "relative_pose": relative,
                        "rows": copied_rows,
                    }
            if bulk_anchor_updates:
                landmarks_map = self.landmarks
                idx_pos = self.index._positions
                for id_list, slots, new_arr in bulk_anchor_updates:
                    idx_pos[slots] = new_arr
                    for i, row in zip(id_list, new_arr):
                        landmarks_map[i].position = row
            elif positions:
                for ident, position in positions.items():
                    self.landmarks[ident].position = position
                self.index.on_landmarks_bulk_moved(positions)
            for ident in moved_kfs:
                self.keyframes[ident].pose = corrected[ident].copy()
                if not self.metric and self.keyframes[ident].depth_points is not None:
                    self.keyframes[ident].depth_points *= scales[ident]
            self.poses = poses
            if len(self.relative_poses) == len(poses):
                kf_inv_cache = {}
                for frame in moved_frames:
                    a = self.pose_anchors[frame]
                    if a is None:
                        self.relative_poses[frame] = poses[frame].copy()
                    else:
                        inv_kf = kf_inv_cache.get(a)
                        if inv_kf is None:
                            inv_kf = np.linalg.inv(self.keyframes[a].pose)
                            kf_inv_cache[a] = inv_kf
                        self.relative_poses[frame] = inv_kf @ poses[frame]
            else:
                self.relative_poses = [
                    np.linalg.inv(self.keyframes[a].pose) @ p if a is not None else p.copy()
                    for p, a in zip(poses, self.pose_anchors)
                ]
            if retained_registry is not None:
                self.retained_source_observations = retained_registry
            self.revision += 1
            self.geometry_revision += 1
            self.pose_revision += 1
            return True

    def apply_snapshot_corrections(self, revision, geometry_revision, original, corrected, scales=None, *, pose_revision=None):
        """Allow append-only progress, but never rebase across changed snapshot geometry."""
        with self.lock:
            ids = sorted(original)
            current = sorted(self.keyframes)
            if pose_revision is not None:
                if (self.pose_revision != pose_revision or not ids
                        or current[:len(ids)] != ids or set(corrected) != set(original)):
                    return False
                if any(not np.array_equal(self.keyframes[i].pose, original[i]) for i in ids):
                    return False
            else:
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
