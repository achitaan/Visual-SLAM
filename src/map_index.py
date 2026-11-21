"""Incremental O(1) pixel-identity conflict, reverse keyframe, and spatial frustum map index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _pixel_identity_key(pixel: Any) -> tuple[int, int] | None:
    arr = np.asarray(pixel, dtype=np.float32).reshape(-1)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        return None
    return (int(np.float32(arr[0]).view(np.int32)), int(np.float32(arr[1]).view(np.int32)))


@dataclass(frozen=True)
class FrustumQueryResult:
    landmark_ids: np.ndarray
    projected_uv: np.ndarray
    depths: np.ndarray


class IncrementalMapIndex:
    """Maintains incremental pixel-conflict, reverse-keyframe, and frustum lookup state.

    All updates run in O(deg(landmark)) time instead of scanning O(|M| + |K|)
    on every tracking frame or local bundle adjustment.
    """

    def __init__(self, initial_capacity: int = 2048) -> None:
        cap = max(16, int(initial_capacity))
        self._capacity = cap
        self._size = 0
        self._ids = np.empty(cap, dtype=np.int64)
        self._positions = np.empty((cap, 3), dtype=np.float64)
        self._is_far = np.zeros(cap, dtype=bool)
        self._max_obs_kf = np.full(cap, -1, dtype=np.int32)
        self._num_obs = np.zeros(cap, dtype=np.int32)
        self._is_conflicted = np.zeros(cap, dtype=bool)
        self._id_to_slot: dict[int, int] = {}

        # (keyframe_id, pixel_key) -> set of landmark_ids claiming that exact pixel
        self._pixel_claims: dict[tuple[int, tuple[int, int]], set[int]] = {}
        # landmark_id -> set of (keyframe_id, pixel_key) keys registered for that landmark
        self._landmark_pixel_keys: dict[int, set[tuple[int, tuple[int, int]]]] = {}
        # landmark_id -> number of conflicted claim buckets currently containing this landmark
        self._conflict_counts: dict[int, int] = {}
        self._conflicted_cache: frozenset[int] = frozenset()
        self._conflicted_dirty = False

        # Reverse index: keyframe_id -> set of landmark_ids observed in that keyframe
        self._kf_landmarks: dict[int, set[int]] = {}
        self._landmark_kfs: dict[int, set[int]] = {}

        # Reverse anchor index: anchor_keyframe_id -> set of landmark_ids anchored there
        self._anchor_landmarks: dict[int, set[int]] = {}
        self._landmark_anchor: dict[int, int] = {}

        # Visibility/found counters for stale single-view pruning
        self._visible_counts: dict[int, int] = {}
        self._found_counts: dict[int, int] = {}

        # Structural version for caching recent-keyframe candidate slots in query_frustum
        self._struct_version: int = 0
        self._recent_kf_cache_version: int = -1
        self._recent_kf_cache_window: int = 0
        self._recent_kf_slots: np.ndarray | None = None

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_ids = np.empty(new_cap, dtype=np.int64)
        new_pos = np.empty((new_cap, 3), dtype=np.float64)
        new_far = np.zeros(new_cap, dtype=bool)
        new_max_obs = np.full(new_cap, -1, dtype=np.int32)
        new_num_obs = np.zeros(new_cap, dtype=np.int32)
        new_conflicted = np.zeros(new_cap, dtype=bool)
        new_ids[: self._size] = self._ids[: self._size]
        new_pos[: self._size] = self._positions[: self._size]
        new_far[: self._size] = self._is_far[: self._size]
        new_max_obs[: self._size] = self._max_obs_kf[: self._size]
        new_num_obs[: self._size] = self._num_obs[: self._size]
        new_conflicted[: self._size] = self._is_conflicted[: self._size]
        self._ids = new_ids
        self._positions = new_pos
        self._is_far = new_far
        self._max_obs_kf = new_max_obs
        self._num_obs = new_num_obs
        self._is_conflicted = new_conflicted
        self._capacity = new_cap

    def _add_claim(self, landmark_id: int, keyframe_id: int, pixel: Any) -> None:
        pkey = _pixel_identity_key(pixel)
        if pkey is None:
            return
        self._add_claim_key(landmark_id, keyframe_id, pkey)

    def _add_claim_key(self, landmark_id: int, keyframe_id: int, pkey: tuple[int, int]) -> None:
        bucket_key = (int(keyframe_id), pkey)
        lm_keys = self._landmark_pixel_keys.setdefault(landmark_id, set())
        if bucket_key in lm_keys:
            return
        lm_keys.add(bucket_key)
        bucket = self._pixel_claims.get(bucket_key)
        if bucket is None:
            self._pixel_claims[bucket_key] = {landmark_id}
            return
        prev_len = len(bucket)
        bucket.add(landmark_id)
        if prev_len == 1 and len(bucket) == 2:
            for existing_id in bucket:
                self._conflict_counts[existing_id] = self._conflict_counts.get(existing_id, 0) + 1
                slot = self._id_to_slot.get(existing_id)
                if slot is not None:
                    self._is_conflicted[slot] = True
            self._conflicted_dirty = True
        elif prev_len >= 2:
            self._conflict_counts[landmark_id] = self._conflict_counts.get(landmark_id, 0) + 1
            slot = self._id_to_slot.get(landmark_id)
            if slot is not None:
                self._is_conflicted[slot] = True
            self._conflicted_dirty = True

    def _remove_claim_bucket(
        self,
        landmark_id: int,
        bucket_key: tuple[int, tuple[int, int]],
    ) -> None:
        bucket = self._pixel_claims.get(bucket_key)
        if bucket is None or landmark_id not in bucket:
            return
        prev_len = len(bucket)
        bucket.remove(landmark_id)
        if prev_len >= 2:
            self._decrement_conflict(landmark_id)
            if len(bucket) == 1:
                survivor = next(iter(bucket))
                self._decrement_conflict(survivor)
        if not bucket:
            del self._pixel_claims[bucket_key]

    def _remove_claims_for_landmark(self, landmark_id: int) -> None:
        keys = self._landmark_pixel_keys.pop(landmark_id, None)
        if not keys:
            return
        for bucket_key in keys:
            self._remove_claim_bucket(landmark_id, bucket_key)

    def _decrement_conflict(self, landmark_id: int) -> None:
        cnt = self._conflict_counts.get(landmark_id, 0)
        if cnt <= 1:
            if landmark_id in self._conflict_counts:
                del self._conflict_counts[landmark_id]
                slot = self._id_to_slot.get(landmark_id)
                if slot is not None:
                    self._is_conflicted[slot] = False
                self._conflicted_dirty = True
        else:
            self._conflict_counts[landmark_id] = cnt - 1

    def on_landmark_added(self, landmark: Any) -> None:
        lm_id = int(landmark.id)
        pos = np.asarray(landmark.position, dtype=np.float64).reshape(3)
        is_far = bool(getattr(landmark, "is_far", False))
        slot = self._id_to_slot.get(lm_id)
        if slot is None:
            if self._size >= self._capacity:
                self._grow()
            slot = self._size
            self._size += 1
            self._ids[slot] = lm_id
            self._id_to_slot[lm_id] = slot
            self._max_obs_kf[slot] = -1
            self._num_obs[slot] = 0
        self._positions[slot] = pos
        self._is_far[slot] = is_far
        self._is_conflicted[slot] = lm_id in self._conflict_counts
        self._visible_counts.setdefault(lm_id, 1)
        self._found_counts.setdefault(lm_id, 1)

        anchor = getattr(landmark, "anchor", None)
        if anchor is not None:
            anchor_id = int(anchor)
            prev_anchor = self._landmark_anchor.get(lm_id)
            if prev_anchor is not None and prev_anchor != anchor_id:
                prev_set = self._anchor_landmarks.get(prev_anchor)
                if prev_set is not None:
                    prev_set.discard(lm_id)
                    if not prev_set:
                        del self._anchor_landmarks[prev_anchor]
            self._landmark_anchor[lm_id] = anchor_id
            self._anchor_landmarks.setdefault(anchor_id, set()).add(lm_id)

        observations = getattr(landmark, "observations", None) or {}
        for kf_id, obs in observations.items():
            self.on_observation_added(lm_id, obs, keyframe_id=int(kf_id))
        self._struct_version += 1

    def on_observation_added(
        self,
        landmark_id: int,
        obs: Any,
        keyframe_id: int | None = None,
    ) -> None:
        lm_id = int(landmark_id)
        # Support both (landmark_id, obs, keyframe_id=k) and (landmark_id, keyframe_id, obs)
        if keyframe_id is None:
            if isinstance(obs, (int, np.integer)):
                return
            kf_attr = getattr(obs, "keyframe_id", None)
            if kf_attr is None:
                return
            kf_id = int(kf_attr)
            actual_obs = obs
        elif isinstance(obs, (int, np.integer)) and not isinstance(keyframe_id, (int, np.integer)):
            kf_id = int(obs)
            actual_obs = keyframe_id
        else:
            kf_id = int(keyframe_id)
            actual_obs = obs

        self._kf_landmarks.setdefault(kf_id, set()).add(lm_id)
        lm_kfs = self._landmark_kfs.setdefault(lm_id, set())
        lm_kfs.add(kf_id)
        slot = self._id_to_slot.get(lm_id)
        if slot is not None:
            self._num_obs[slot] = len(lm_kfs)
            if kf_id > self._max_obs_kf[slot]:
                self._max_obs_kf[slot] = kf_id
        self._found_counts[lm_id] = self._found_counts.get(lm_id, 0) + 1
        pid_i32 = getattr(actual_obs, "_pixel_id_i32", None)
        if pid_i32 is not None and getattr(actual_obs, "_identity_owned", False):
            self._add_claim_key(lm_id, kf_id, pid_i32)
        else:
            pixel = getattr(actual_obs, "pixel", None)
            if pixel is not None:
                self._add_claim(lm_id, kf_id, pixel)
        self._struct_version += 1

    def on_observation_removed(
        self,
        landmark_id: int,
        keyframe_id: int,
        obs: Any = None,
    ) -> None:
        lm_id = int(landmark_id)
        kf_id = int(keyframe_id)

        kf_set = self._kf_landmarks.get(kf_id)
        if kf_set is not None:
            kf_set.discard(lm_id)
            if not kf_set:
                del self._kf_landmarks[kf_id]

        lm_kfs = self._landmark_kfs.get(lm_id)
        self._struct_version += 1
        if lm_kfs is not None:
            lm_kfs.discard(kf_id)
            if not lm_kfs:
                del self._landmark_kfs[lm_id]
                lm_kfs = None

        slot = self._id_to_slot.get(lm_id)
        if slot is not None:
            if not lm_kfs:
                self._num_obs[slot] = 0
                self._max_obs_kf[slot] = -1
            else:
                self._num_obs[slot] = len(lm_kfs)
                if kf_id == self._max_obs_kf[slot]:
                    self._max_obs_kf[slot] = max(lm_kfs)

        lm_keys = self._landmark_pixel_keys.get(lm_id)
        if lm_keys:
            matching_keys = [key for key in lm_keys if key[0] == kf_id]
            for bucket_key in matching_keys:
                lm_keys.discard(bucket_key)
                self._remove_claim_bucket(lm_id, bucket_key)
            if not lm_keys:
                self._landmark_pixel_keys.pop(lm_id, None)
        elif obs is not None:
            pixel = getattr(obs, "pixel", None)
            if pixel is not None:
                pkey = _pixel_identity_key(pixel)
                if pkey is not None:
                    self._remove_claim_bucket(lm_id, (kf_id, pkey))

        if lm_id in self._found_counts:
            self._found_counts[lm_id] = max(0, self._found_counts[lm_id] - 1)

    def on_landmark_moved(self, landmark_id: int, position: np.ndarray) -> None:
        slot = self._id_to_slot.get(int(landmark_id))
        if slot is not None:
            self._positions[slot] = np.asarray(position, dtype=np.float64).reshape(3)

    def on_landmarks_bulk_moved(self, positions_by_id: dict[int, np.ndarray]) -> None:
        id_to_slot = self._id_to_slot
        positions = self._positions
        for lm_id, pos in positions_by_id.items():
            slot = id_to_slot.get(int(lm_id))
            if slot is not None:
                positions[slot] = pos

    def on_landmark_removed(self, landmark_id: int) -> None:
        lm_id = int(landmark_id)
        slot = self._id_to_slot.pop(lm_id, None)
        if slot is not None:
            last = self._size - 1
            if slot != last:
                moved_id = int(self._ids[last])
                self._ids[slot] = moved_id
                self._positions[slot] = self._positions[last]
                self._is_far[slot] = self._is_far[last]
                self._max_obs_kf[slot] = self._max_obs_kf[last]
                self._num_obs[slot] = self._num_obs[last]
                self._is_conflicted[slot] = self._is_conflicted[last]
                self._id_to_slot[moved_id] = slot
            self._size = last

        self._remove_claims_for_landmark(lm_id)
        kfs = self._landmark_kfs.pop(lm_id, None)
        if kfs:
            for kf_id in kfs:
                kf_set = self._kf_landmarks.get(kf_id)
                if kf_set is not None:
                    kf_set.discard(lm_id)
                    if not kf_set:
                        del self._kf_landmarks[kf_id]
        anchor = self._landmark_anchor.pop(lm_id, None)
        if anchor is not None:
            anc_set = self._anchor_landmarks.get(anchor)
            if anc_set is not None:
                anc_set.discard(lm_id)
                if not anc_set:
                    del self._anchor_landmarks[anchor]
        self._visible_counts.pop(lm_id, None)
        self._found_counts.pop(lm_id, None)
        self._struct_version += 1

    def record_visibility(self, landmark_ids: np.ndarray | list[int]) -> None:
        for lm_id in landmark_ids:
            lid = int(lm_id)
            if lid in self._id_to_slot:
                self._visible_counts[lid] = self._visible_counts.get(lid, 0) + 1

    def record_found(self, landmark_ids: np.ndarray | list[int]) -> None:
        for lm_id in landmark_ids:
            lid = int(lm_id)
            if lid in self._id_to_slot:
                self._found_counts[lid] = self._found_counts.get(lid, 0) + 1

    def conflicted_landmark_ids(self) -> frozenset[int]:
        if self._conflicted_dirty:
            self._conflicted_cache = frozenset(self._conflict_counts.keys())
            self._conflicted_dirty = False
        return self._conflicted_cache

    def landmarks_for_keyframes(self, keyframe_ids: Any) -> set[int]:
        result: set[int] = set()
        for kf_id in keyframe_ids:
            lm_set = self._kf_landmarks.get(int(kf_id))
            if lm_set:
                result.update(lm_set)
        return result

    def landmarks_for_anchors(self, anchor_ids: Any) -> set[int]:
        result: set[int] = set()
        for anc_id in anchor_ids:
            lm_set = self._anchor_landmarks.get(int(anc_id))
            if lm_set:
                result.update(lm_set)
        return result

    def _recent_keyframe_slots(self, window: int = 12) -> np.ndarray | None:
        """Return ascending slots of all landmarks observed in the latest ``window`` keyframes.

        Every landmark with ``max_obs_kf >= min(recent_kfs)`` is guaranteed to appear
        in this subset, and every landmark outside it strictly has a smaller ``max_obs_kf``.
        """
        if len(self._kf_landmarks) <= window:
            return None
        if (
            self._recent_kf_cache_version == self._struct_version
            and self._recent_kf_cache_window == window
            and self._recent_kf_slots is not None
        ):
            return self._recent_kf_slots
        recent_kfs = sorted(self._kf_landmarks.keys(), reverse=True)[:window]
        id_to_slot = self._id_to_slot
        slot_set: set[int] = set()
        for kf_id in recent_kfs:
            for lm_id in self._kf_landmarks[kf_id]:
                slot = id_to_slot.get(lm_id)
                if slot is not None:
                    slot_set.add(slot)
        slots = np.fromiter(slot_set, dtype=np.int64, count=len(slot_set))
        slots.sort()
        self._recent_kf_cache_version = self._struct_version
        self._recent_kf_cache_window = window
        self._recent_kf_slots = slots
        return slots

    def _project_frustum_subset(
        self,
        slots: np.ndarray | None,
        R: np.ndarray,
        t: np.ndarray,
        K_arr: np.ndarray,
        width: float,
        height: float,
        max_depth: float,
        margin_px: float,
        include_far_bearings: bool,
        exclude_conflicted: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if slots is None:
            n = self._size
            pts_w = self._positions[:n]
            is_far = self._is_far[:n]
            is_conf = self._is_conflicted[:n]
        else:
            if not len(slots):
                empty = np.empty(0, dtype=np.int64)
                empty_f = np.empty(0, dtype=np.float64)
                return empty, empty_f, empty_f, empty_f
            pts_w = self._positions[slots]
            is_far = self._is_far[slots]
            is_conf = self._is_conflicted[slots]

        # Compute camera-frame Z first so points behind the camera skip X/Y matmul.
        z = pts_w @ R[2] + t[2]
        depth_ok = z > 1e-3
        if include_far_bearings:
            depth_ok &= (z <= max_depth) | is_far
        else:
            depth_ok &= z <= max_depth
        if exclude_conflicted:
            depth_ok &= ~is_conf

        cand_idx = np.flatnonzero(depth_ok)
        if not len(cand_idx):
            empty = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return empty, empty_f, empty_f, empty_f

        cand_w = pts_w[cand_idx]
        cand_xy = cand_w @ R[:2].T + t[:2]
        cand_z = z[cand_idx]
        fx, fy = K_arr[0, 0], K_arr[1, 1]
        cx, cy = K_arr[0, 2], K_arr[1, 2]
        inv_z = 1.0 / cand_z
        u = fx * cand_xy[:, 0] * inv_z + cx
        v = fy * cand_xy[:, 1] * inv_z + cy

        in_bounds = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u >= -margin_px)
            & (u < width + margin_px)
            & (v >= -margin_px)
            & (v < height + margin_px)
        )
        in_idx = np.flatnonzero(in_bounds)
        if not len(in_idx):
            empty = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float64)
            return empty, empty_f, empty_f, empty_f

        local_slots = cand_idx[in_idx]
        final_slots = local_slots if slots is None else slots[local_slots]
        return final_slots, u[in_idx], v[in_idx], cand_z[in_idx]

    def query_frustum(
        self,
        T_cw: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
        max_depth: float = 80.0,
        margin_px: float = 32.0,
        include_far_bearings: bool = False,
        *,
        exclude_conflicted: bool = False,
        max_landmarks: int | None = None,
    ) -> FrustumQueryResult:
        if self._size == 0:
            return FrustumQueryResult(
                landmark_ids=np.empty(0, dtype=np.int64),
                projected_uv=np.empty((0, 2), dtype=np.float64),
                depths=np.empty(0, dtype=np.float64),
            )
        T = np.asarray(T_cw, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("T_cw must be a finite 4x4 matrix")
        K_arr = np.asarray(K, dtype=np.float64)
        if K_arr.shape != (3, 3) or not np.isfinite(K_arr).all():
            raise ValueError("K must be a finite 3x3 matrix")
        width, height = float(image_size[0]), float(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image_size dimensions must be positive")
        if max_depth <= 0 or not np.isfinite(max_depth):
            raise ValueError("max_depth must be positive and finite")

        R = T[:3, :3]
        t = T[:3, 3]
        final_slots = None
        u_vis = v_vis = z_vis = None

        # Exact recent-keyframe top-K fast path: on large maps, every landmark observed in
        # the latest 12 keyframes strictly outranks all older landmarks in (max_obs_kf, num_obs).
        # If >= k of those recent landmarks are visible in the frustum, their top-k is
        # mathematically identical to scanning the entire map. Otherwise fall back to full scan.
        if max_landmarks is not None:
            k = int(max_landmarks)
            if k <= 0:
                return FrustumQueryResult(
                    landmark_ids=np.empty(0, dtype=np.int64),
                    projected_uv=np.empty((0, 2), dtype=np.float64),
                    depths=np.empty(0, dtype=np.float64),
                )
            if self._size > max(4 * k, 8000):
                recent_slots = self._recent_keyframe_slots(window=12)
                if recent_slots is not None and len(recent_slots) >= k and len(recent_slots) < self._size // 2:
                    r_slots, r_u, r_v, r_z = self._project_frustum_subset(
                        recent_slots, R, t, K_arr, width, height,
                        max_depth, margin_px, include_far_bearings, exclude_conflicted,
                    )
                    if len(r_slots) >= k:
                        final_slots, u_vis, v_vis, z_vis = r_slots, r_u, r_v, r_z

        if final_slots is None:
            final_slots, u_vis, v_vis, z_vis = self._project_frustum_subset(
                None, R, t, K_arr, width, height,
                max_depth, margin_px, include_far_bearings, exclude_conflicted,
            )

        if not len(final_slots):
            return FrustumQueryResult(
                landmark_ids=np.empty(0, dtype=np.int64),
                projected_uv=np.empty((0, 2), dtype=np.float64),
                depths=np.empty(0, dtype=np.float64),
            )

        if max_landmarks is not None and len(final_slots) > int(max_landmarks):
            k = int(max_landmarks)
            # Exact stable top-k descending by (max_obs_kf, num_obs), preserving
            # original ascending slot order on ties.
            keys = (
                ((self._max_obs_kf[final_slots].astype(np.int64) + 1) << 32)
                | self._num_obs[final_slots].astype(np.int64)
            )
            threshold = int(np.partition(keys, len(keys) - k)[len(keys) - k])
            above = np.flatnonzero(keys > threshold)
            need = k - len(above)
            if need > 0:
                equal = np.flatnonzero(keys == threshold)[:need]
                sel = np.concatenate((above, equal))
            else:
                sel = above
            order = sel[np.lexsort((sel, -keys[sel]))]
            final_slots = final_slots[order]
            u_vis = u_vis[order]
            v_vis = v_vis[order]
            z_vis = z_vis[order]

        uv = np.column_stack((u_vis, v_vis))
        return FrustumQueryResult(
            landmark_ids=self._ids[final_slots].copy(),
            projected_uv=uv,
            depths=z_vis.copy(),
        )

    def prune_stale_landmarks(
        self,
        map_state: Any,
        current_kf_id: int,
        min_age_kfs: int = 3,
        min_found_ratio: float = 0.25,
    ) -> int:
        """Remove old single-view landmarks with poor found/visible ratio."""
        if min_age_kfs < 0:
            raise ValueError("min_age_kfs must be non-negative")
        removed = 0
        with map_state.lock:
            to_delete: list[int] = []
            for lm_id, lm in list(map_state.landmarks.items()):
                if len(lm.observations) >= 2:
                    continue
                age = int(current_kf_id) - int(lm.anchor)
                if age < min_age_kfs:
                    continue
                vis = max(1, self._visible_counts.get(lm_id, 1))
                found = self._found_counts.get(lm_id, 1)
                ratio = float(found) / float(vis)
                if len(lm.observations) <= 1 and (
                    ratio < min_found_ratio or lm.misses >= 2 or age >= max(min_age_kfs * 2, 6)
                ):
                    to_delete.append(lm_id)
            for lm_id in to_delete:
                lm = map_state.landmarks.pop(lm_id, None)
                if lm is not None:
                    self.on_landmark_removed(lm_id)
                    removed += 1
        return removed

    def rebuild_from_map(self, map_state: Any) -> None:
        """Synchronize index completely from a MapState (e.g., after external edits)."""
        self._size = 0
        self._id_to_slot.clear()
        self._pixel_claims.clear()
        self._landmark_pixel_keys.clear()
        self._conflict_counts.clear()
        self._conflicted_cache = frozenset()
        self._conflicted_dirty = False
        self._kf_landmarks.clear()
        self._landmark_kfs.clear()
        self._anchor_landmarks.clear()
        self._landmark_anchor.clear()
        self._visible_counts.clear()
        self._found_counts.clear()
        self._struct_version += 1
        self._recent_kf_slots = None
        for lm in map_state.landmarks.values():
            self.on_landmark_added(lm)
