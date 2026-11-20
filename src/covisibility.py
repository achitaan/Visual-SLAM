"""Keyframe covisibility graph for covisible local BA window selection and neighbor fusion."""

from __future__ import annotations

from collections import Counter
from typing import Any


class CovisibilityGraph:
    """Tracks shared-landmark counts between keyframes in O(|obs_in_kf|) per update."""

    def __init__(self) -> None:
        self._neighbors: dict[int, dict[int, int]] = {}

    def update_from_map(self, map_state: Any, keyframe_id: int | None = None) -> None:
        """Rebuild either a single keyframe's covisibility edges or the entire graph."""
        if keyframe_id is None:
            self._neighbors.clear()
            kf_counts: dict[int, Counter[int]] = {}
            for lm in map_state.landmarks.values():
                obs_kfs = [int(k) for k in lm.observations if k in map_state.keyframes]
                if len(obs_kfs) < 2:
                    continue
                for i, k1 in enumerate(obs_kfs):
                    c1 = kf_counts.setdefault(k1, Counter())
                    for k2 in obs_kfs[i + 1 :]:
                        c1[k2] += 1
                        kf_counts.setdefault(k2, Counter())[k1] += 1
            for kf, counter in kf_counts.items():
                self._neighbors[kf] = dict(counter)
            return

        kf_id = int(keyframe_id)
        if kf_id not in map_state.keyframes:
            return
        if hasattr(map_state, "index") and len(map_state.index) == len(map_state.landmarks):
            lm_ids = map_state.index.landmarks_for_keyframes([kf_id])
            lms = [map_state.landmarks[lid] for lid in lm_ids if lid in map_state.landmarks]
        else:
            lms = [lm for lm in map_state.landmarks.values() if kf_id in lm.observations]

        shared: Counter[int] = Counter()
        for lm in lms:
            for other_kf in lm.observations:
                ok = int(other_kf)
                if ok != kf_id and ok in map_state.keyframes:
                    shared[ok] += 1

        # Remove old reverse links
        old = self._neighbors.get(kf_id)
        if old:
            for other_kf in list(old.keys()):
                rev = self._neighbors.get(other_kf)
                if rev is not None:
                    rev.pop(kf_id, None)

        self._neighbors[kf_id] = dict(shared)
        for other_kf, count in shared.items():
            self._neighbors.setdefault(other_kf, {})[kf_id] = int(count)

    def shared_count(self, kf_a: int, kf_b: int) -> int:
        return int(self._neighbors.get(int(kf_a), {}).get(int(kf_b), 0))

    def top_covisible_keyframes(
        self,
        anchor_kf_id: int,
        max_keyframes: int = 8,
        min_shared: int = 8,
    ) -> list[int]:
        """Return up to max_keyframes neighbor keyframe IDs with >= min_shared landmarks, sorted by shared count desc."""
        anchor = int(anchor_kf_id)
        nb = self._neighbors.get(anchor, {})
        eligible = [
            (count, kf_id)
            for kf_id, count in nb.items()
            if count >= int(min_shared) and kf_id != anchor
        ]
        eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [kf_id for _cnt, kf_id in eligible[: max(0, int(max_keyframes))]]

    def select_ba_window(
        self,
        map_state: Any,
        anchor_kf_id: int,
        max_keyframes: int = 8,
        min_shared: int = 8,
    ) -> list[int]:
        """Select sorted local BA keyframe window combining anchor + strongest covisible keyframes.

        Always includes anchor_kf_id and preserves chronological ordering so the oldest
        keyframe in the window acts as the fixed metric gauge anchor.
        """
        anchor = int(anchor_kf_id)
        if anchor not in map_state.keyframes:
            return sorted(map_state.keyframes)[-max_keyframes:]

        self.update_from_map(map_state, anchor)
        top_nbs = self.top_covisible_keyframes(
            anchor, max_keyframes=max(1, int(max_keyframes) - 1), min_shared=min_shared
        )
        selected_set = {anchor}
        for kf_id in top_nbs:
            if kf_id in map_state.keyframes:
                selected_set.add(int(kf_id))

        # Backfill with most recent chronological keyframes if covisibility set is sparse
        if len(selected_set) < min(max_keyframes, len(map_state.keyframes)):
            for kf_id in reversed(sorted(map_state.keyframes)):
                selected_set.add(int(kf_id))
                if len(selected_set) >= max_keyframes:
                    break

        return sorted(selected_set)
