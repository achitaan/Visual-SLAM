"""Frozen original identity oracle, not the optimized implementation.

Extracted from the verified original Python control source; provenance is in
owned_observation_control.json. Retained as a small portable regression oracle.
"""
import hashlib
import numpy as np

def _landmark_pixel_identity_conflicts(self, landmarks):
        """Find distinct LMs claiming one exact pixel in one keyframe."""
        claims = {}
        conflicts = set()
        cached_keyframes = set()
        # Inspect stored detector rows, not just the candidate subset. A
        # relocalization probe may contain only one of two legacy IDs that
        # claim the same source pixel. Cache the conflict set per keyframe;
        # exact content hashes detect old-keyframe ID edits without regrouping
        # every historical row on every frame.
        for keyframe_id, keyframe in self.map.keyframes.items():
            cached_keyframes.add(keyframe_id)
            pixels = np.asarray(getattr(keyframe, "pixels", np.empty((0, 2))), np.float32)
            identifiers = np.asarray(
                getattr(keyframe, "landmark_ids", np.empty(0, int))
            )
            if (pixels.ndim != 2 or pixels.shape[1:] != (2,)
                    or identifiers.shape != (len(pixels),)
                    or not np.issubdtype(identifiers.dtype, np.integer)
                    or np.issubdtype(identifiers.dtype, np.bool_)):
                self._identity_conflict_cache.pop(keyframe_id, None)
                continue
            pixel_bytes = np.ascontiguousarray(pixels)
            id_bytes = np.ascontiguousarray(identifiers)
            digest = hashlib.sha256()
            digest.update(str((pixel_bytes.dtype.str, pixel_bytes.shape)).encode())
            if pixel_bytes.size:
                digest.update(memoryview(pixel_bytes).cast("B"))
            digest.update(str((id_bytes.dtype.str, id_bytes.shape)).encode())
            if id_bytes.size:
                digest.update(memoryview(id_bytes).cast("B"))
            fingerprint = digest.digest()
            cached = self._identity_conflict_cache.get(keyframe_id)
            if cached is not None and cached[0] == fingerprint:
                conflicts.update(cached[1])
                continue
            frame_conflicts = set()
            groups, _ = self._physical_pixel_groups(pixels)
            for members in groups:
                ids = {int(identifiers[row]) for row in members
                       if int(identifiers[row]) >= 0}
                if len(ids) > 1:
                    frame_conflicts.update(ids)
            self._identity_conflict_cache[keyframe_id] = (
                fingerprint, frozenset(frame_conflicts)
            )
            conflicts.update(frame_conflicts)
        for stale_keyframe in set(self._identity_conflict_cache) - cached_keyframes:
            del self._identity_conflict_cache[stale_keyframe]
        for landmark in landmarks:
            for keyframe_id, observation in landmark.observations.items():
                # Synthetic map-only clients can provide observations without
                # keyframes; there is no detector-row identity to disambiguate.
                if keyframe_id not in self.map.keyframes:
                    continue
                pixel = np.asarray(observation.pixel, np.float32).reshape(-1)
                if pixel.shape == (2,) and np.isfinite(pixel).all():
                    claims.setdefault((int(keyframe_id), self._physical_pixel_key(pixel)), set()).add(
                        int(landmark.id)
                    )
        conflicts.update(landmark_id for identifiers in claims.values()
                         if len(identifiers) > 1 for landmark_id in identifiers)
        return conflicts
