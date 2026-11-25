"""Cache the descriptor pool rows for landmarks used by live map tracking."""

from itertools import chain

import numpy as np


class TrackingDescriptorRows:
    def __init__(self, map_state, build, invalidate_keyframe):
        self.map = map_state
        self._build = build
        self._invalidate_keyframe = invalidate_keyframe
        # Rows are keyed by stable landmark IDs, independent of map-index slot swaps.
        self._rows = np.full((2048, 16), -1, dtype=np.intp)
        self._frame_rows = {}
        self._frame_version = map_state.landmark_ids_version

    def clear(self):
        self._rows.fill(-1)
        self._frame_rows.clear()
        self._invalidate_keyframe(self)
        self._lm_desc_fast_rev = None

    def invalidate_landmark(self, landmark_id):
        if 0 <= landmark_id < len(self._rows):
            self._rows[landmark_id, 0] = -1
        recent = getattr(self, "_lm_desc_fast_slots", None)
        if recent is not None:
            recent.pop(landmark_id, None)

    def discard_landmark(self, landmark_id):
        self.invalidate_landmark(landmark_id)
        if hasattr(self, "_lm_desc_cache"):
            removed = self._lm_desc_cache.pop(landmark_id, None)
            if removed is not None and removed[-1] is not None:
                self._lm_desc_free.extend(removed[-1])

    def keyframe_changed(self, keyframe_id):
        self._invalidate_keyframe(self, keyframe_id)
        frame = self.map.keyframes.get(keyframe_id)
        previous = self._frame_rows.get(keyframe_id)
        if (previous is not None and frame is previous[0]
                and frame.descriptors is previous[1] and not frame.descriptors.flags.writeable
                and frame.landmark_ids.shape == previous[2].shape):
            old_ids = previous[2]
            new_ids = frame.landmark_ids
            changed = np.flatnonzero(old_ids != new_ids)
            affected = np.concatenate((old_ids[changed], new_ids[changed]))
            old_ids[changed] = new_ids[changed]
            self._frame_rows[keyframe_id] = frame, frame.descriptors, old_ids, frame.landmark_ids_version
        else:
            index = self.map.index
            linked = index._kf_landmarks.get(keyframe_id, set())
            anchored = index._anchor_landmarks.get(keyframe_id, set())
            affected = np.fromiter(linked | anchored, dtype=np.intp)
            if previous is not None:
                affected = np.concatenate((affected, previous[2]))
            if frame is not None:
                self._remember_frame(keyframe_id, frame)
            else:
                self._frame_rows.pop(keyframe_id, None)
        live = affected[(affected >= 0) & (affected < len(self._rows))]
        self._rows[live, 0] = -1

    def _remember_frame(self, keyframe_id, frame):
        self._frame_rows[keyframe_id] = (
            frame, frame.descriptors, frame.landmark_ids.copy(), frame.landmark_ids_version,
        )

    def _refresh_frame_versions(self):
        if self._frame_version != self.map.landmark_ids_version:
            for key, (previous, _, _, version) in tuple(self._frame_rows.items()):
                frame = self.map.keyframes.get(key)
                if frame is not previous or frame.landmark_ids_version != version:
                    self.keyframe_changed(key)
            self._frame_version = self.map.landmark_ids_version

    def bank(self, landmarks):
        self._refresh_frame_versions()
        ids = np.fromiter((landmark.id for landmark in landmarks), dtype=np.intp, count=len(landmarks))
        if not len(ids) or ids.min() < 0:
            return self._build(self, landmarks)
        largest = int(ids.max())
        if largest >= len(self._rows):
            grown = np.full((max(largest + 1, 2 * len(self._rows)), self._rows.shape[1]), -1, dtype=np.intp)
            grown[:len(self._rows)] = self._rows
            self._rows = grown
        rows = self._rows[ids]
        missing = np.flatnonzero(rows[:, 0] < 0)
        if len(missing):
            result = self._build(self, [landmarks[row] for row in missing.tolist()])
            for key in self._kf_landmark_row_cache:
                if key not in self._frame_rows:
                    self._remember_frame(key, self.map.keyframes[key])
            slots = [self._lm_desc_cache[landmark_id][-1] for landmark_id in ids[missing].tolist()]
            if any(slot is None for slot in slots):
                return self._build(self, landmarks)
            counts = np.fromiter(map(len, slots), dtype=np.intp, count=len(slots))
            width = int(counts.max())
            if width > 64:
                return self._build(self, landmarks)
            if width > self._rows.shape[1]:
                grown = np.full((len(self._rows), min(64, 2 * width)), -1, dtype=np.intp)
                grown[:, :self._rows.shape[1]] = self._rows
                self._rows = grown
                rows = self._rows[ids]
            row, column = np.nonzero(np.arange(rows.shape[1])[None, :] < counts[:, None])
            flat_slots = np.fromiter(chain.from_iterable(slots), dtype=np.intp, count=len(row))
            rows[missing] = -1
            rows[missing[row], column] = flat_slots
            self._rows[ids[missing]] = rows[missing]
            if len(missing) == len(ids):
                return result
        location = np.nonzero(rows >= 0)
        return self._lm_desc_pool[rows[location]], ids[location[0]].astype(int, copy=False)
