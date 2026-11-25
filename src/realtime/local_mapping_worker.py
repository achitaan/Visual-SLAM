"""Decoupled asynchronous local mapping / bundle adjustment worker.

Keeps ``MapState`` strictly single-writer on the tracking thread:
1. Tracking thread captures an immutable ``LocalBASnapshot`` of the local BA
   window (O(|local_window|) copy, typically < 0.5 ms).
2. ``LocalMappingWorker`` solves ``local_bundle_adjustment()`` on a dedicated
   background thread against an isolated shadow map state.
3. Tracking thread atomically applies the resulting ``LocalBACorrection`` patch
   at the start of ``process_frame()`` via ``MapState.apply_corrections()``,
   preserving descriptor/landmark cache versions (``landmark_ids_version``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Condition, Lock, RLock, Thread
import time
from typing import Any

import numpy as np

from local_bundle import local_bundle_adjustment
from slam_state import Observation


@dataclass
class _ShadowKeyframe:
    id: int
    frame: int
    pose: np.ndarray


@dataclass
class _ShadowLandmark:
    id: int
    position: np.ndarray
    anchor: int
    observations: dict[int, Observation]
    octave: int = 0
    is_far: bool = False


class _ShadowMapState:
    """Isolated MapState facade consumed exclusively by LocalMappingWorker."""

    def __init__(self, snapshot: "LocalBASnapshot") -> None:
        self.metric = snapshot.metric
        self.revision = snapshot.map_revision
        self.geometry_revision = snapshot.geometry_revision
        self.pose_revision = snapshot.pose_revision
        self.lock = RLock()
        self.keyframes: dict[int, _ShadowKeyframe] = {
            kf_id: _ShadowKeyframe(kf_id, snapshot.keyframe_frames[kf_id], pose.copy())
            for kf_id, pose in snapshot.keyframe_poses.items()
        }
        self.landmarks: dict[int, _ShadowLandmark] = {
            lid: _ShadowLandmark(
                id=lid,
                position=pos.copy(),
                anchor=anchor,
                observations=dict(obs),
                octave=octave,
                is_far=is_far,
            )
            for lid, (pos, anchor, obs, octave, is_far) in snapshot.landmarks.items()
        }
        self.poses: list[np.ndarray] = [
            p.copy() if p is not None else np.eye(4, dtype=np.float64)
            for p in snapshot.frame_poses
        ]
        self.pose_anchors: list[int | None] = list(snapshot.frame_anchors)
        self.statuses: list[str] = list(snapshot.frame_statuses)
        self.stereo_motion: dict[tuple[int, int], np.ndarray] = {
            edge: meas.copy() for edge, meas in snapshot.stereo_motion.items()
        }
        self._stereo_motion_by_anchor: dict[int, list[tuple[int, int]]] = {}
        for edge in self.stereo_motion:
            f0, f1 = edge
            if f0 < len(self.pose_anchors) and f1 < len(self.pose_anchors):
                a0 = self.pose_anchors[f0]
                a1 = self.pose_anchors[f1]
                if a0 != a1:
                    if a0 is not None:
                        self._stereo_motion_by_anchor.setdefault(a0, []).append(edge)
                    if a1 is not None:
                        self._stereo_motion_by_anchor.setdefault(a1, []).append(edge)
        self.retained_source_observations: dict[int, Any] = {}
        self.relative_poses: list[np.ndarray] = []

        # Captured outputs from local_bundle_adjustment()
        self.captured_poses: dict[int, np.ndarray] | None = None
        self.captured_landmark_updates: dict[int, np.ndarray] | None = None
        self.culled_observations: list[tuple[int, int]] = []

    def apply_corrections(
        self,
        expected_revision: int,
        corrected: dict[int, np.ndarray],
        scales=None,
        *,
        propagate_landmarks: bool = True,
        landmark_updates: dict[int, np.ndarray] | None = None,
        frame_updates=None,
        retained_source_updates=None,
        retained_source_limits=None,
    ) -> bool:
        _ = (scales, propagate_landmarks, frame_updates, retained_source_updates, retained_source_limits)
        if expected_revision != self.revision:
            return False
        self.captured_poses = {int(k): np.asarray(p, dtype=np.float64).copy() for k, p in corrected.items()}
        for k, p in self.captured_poses.items():
            if k in self.keyframes:
                self.keyframes[k].pose = p.copy()
        if landmark_updates is not None:
            self.captured_landmark_updates = {
                int(lid): np.asarray(pos, dtype=np.float64).reshape(3).copy()
                for lid, pos in landmark_updates.items()
            }
            for lid, pos in self.captured_landmark_updates.items():
                lm = self.landmarks.get(lid)
                if lm is not None:
                    lm.position = pos.copy()
        else:
            self.captured_landmark_updates = {}
        self.revision += 1
        self.geometry_revision += 1
        self.pose_revision += 1
        return True

    def remove_observation(self, landmark_id: int, keyframe_id: int):
        lm = self.landmarks.get(landmark_id)
        if lm is None:
            return None
        removed = lm.observations.pop(keyframe_id, None)
        if removed is not None:
            self.culled_observations.append((int(landmark_id), int(keyframe_id)))
        return removed


@dataclass(frozen=True)
class LocalBASnapshot:
    """Immutable self-contained snapshot of a local BA window."""

    snapshot_id: int
    frame_index: int
    anchor_kf_id: int
    map_revision: int
    geometry_revision: int
    pose_revision: int
    metric: bool
    selected_keyframes: tuple[int, ...]
    keyframe_poses: dict[int, np.ndarray]
    keyframe_frames: dict[int, int]
    landmarks: dict[int, tuple[np.ndarray, int, dict[int, Observation], int, bool]]
    stereo_motion: dict[tuple[int, int], np.ndarray]
    frame_poses: list[np.ndarray | None]
    frame_anchors: list[int | None]
    frame_statuses: list[str]
    K: np.ndarray
    baseline: float
    bundle_kwargs: dict[str, Any]
    created_at_s: float = field(default_factory=time.perf_counter)

    @classmethod
    def capture(
        cls,
        state: Any,
        *,
        snapshot_id: int,
        frame_index: int,
        anchor_kf_id: int,
        K: np.ndarray,
        baseline: float,
        bundle_kwargs: dict[str, Any],
    ) -> "LocalBASnapshot | None":
        """Capture a lightweight O(|window|) snapshot under ``state.lock``."""
        with state.lock:
            if len(state.keyframes) < 3:
                return None
            window = int(bundle_kwargs.get("window", 5))
            req_selected = bundle_kwargs.get("selected_keyframes")
            if req_selected is not None:
                selected = [int(k) for k in req_selected if int(k) in state.keyframes]
                if len(selected) < 3:
                    selected = [int(k) for k in list(state.keyframes)[-window:]]
            else:
                selected = [int(k) for k in list(state.keyframes)[-window:]]
            selected_set = set(selected)
            kf_ids_to_include = set(selected)
            if 0 in state.keyframes:
                kf_ids_to_include.add(0)

            keyframe_poses = {
                k: state.keyframes[k].pose.copy()
                for k in sorted(kf_ids_to_include)
            }
            keyframe_frames = {
                k: int(state.keyframes[k].frame)
                for k in sorted(kf_ids_to_include)
            }

            if (
                hasattr(state, "index")
                and len(state.index) == len(state.landmarks)
            ):
                lm_ids = state.index.landmarks_for_keyframes(selected)
                pool = [state.landmarks[lid] for lid in lm_ids if lid in state.landmarks]
            else:
                pool = [
                    lm
                    for lm in state.landmarks.values()
                    if any(k in selected_set for k in lm.observations)
                ]

            landmarks: dict[int, tuple[np.ndarray, int, dict[int, Observation], int, bool]] = {}
            for lm in pool:
                obs = lm.observations
                if len(obs) < 2:
                    continue
                obs_copy = {k: o for k, o in obs.items() if k in selected_set}
                if len(obs_copy) < 2:
                    continue
                landmarks[int(lm.id)] = (
                    lm.position.copy(),
                    int(lm.anchor),
                    obs_copy,
                    int(getattr(lm, "octave", 0)),
                    bool(getattr(lm, "is_far", False)),
                )

            stereo_motion: dict[tuple[int, int], np.ndarray] = {}
            needed_frames: set[int] = set()
            sm_by_anchor = getattr(state, "_stereo_motion_by_anchor", None)
            if state.metric and state.stereo_motion:
                if sm_by_anchor is not None:
                    seen_edges: set[tuple[int, int]] = set()
                    for kf_id in selected:
                        for edge in sm_by_anchor.get(kf_id, ()):
                            if edge in seen_edges:
                                continue
                            seen_edges.add(edge)
                            f0, f1 = edge
                            if f0 < len(state.pose_anchors) and f1 < len(state.pose_anchors):
                                a0, a1 = state.pose_anchors[f0], state.pose_anchors[f1]
                                if a0 in selected_set and a1 in selected_set and a0 != a1:
                                    stereo_motion[edge] = state.stereo_motion[edge].copy()
                                    needed_frames.add(f0)
                                    needed_frames.add(f1)
                else:
                    for edge, meas in state.stereo_motion.items():
                        f0, f1 = edge
                        if f0 < len(state.pose_anchors) and f1 < len(state.pose_anchors):
                            a0, a1 = state.pose_anchors[f0], state.pose_anchors[f1]
                            if a0 in selected_set and a1 in selected_set and a0 != a1:
                                stereo_motion[edge] = meas.copy()
                                needed_frames.add(f0)
                                needed_frames.add(f1)

            n_poses = (max(needed_frames) + 1) if needed_frames else 0
            frame_poses: list[np.ndarray | None] = [None] * n_poses
            frame_anchors: list[int | None] = [None] * n_poses
            frame_statuses: list[str] = [""] * n_poses
            for f in needed_frames:
                frame_poses[f] = state.poses[f].copy()
                frame_anchors[f] = state.pose_anchors[f]
                frame_statuses[f] = state.statuses[f]

            clean_kwargs = dict(bundle_kwargs)
            clean_kwargs["use_map_index"] = False
            clean_kwargs["selected_keyframes"] = list(selected)

            return cls(
                snapshot_id=int(snapshot_id),
                frame_index=int(frame_index),
                anchor_kf_id=int(anchor_kf_id),
                map_revision=int(state.revision),
                geometry_revision=int(state.geometry_revision),
                pose_revision=int(state.pose_revision),
                metric=bool(state.metric),
                selected_keyframes=tuple(selected),
                keyframe_poses=keyframe_poses,
                keyframe_frames=keyframe_frames,
                landmarks=landmarks,
                stereo_motion=stereo_motion,
                frame_poses=frame_poses,
                frame_anchors=frame_anchors,
                frame_statuses=frame_statuses,
                K=np.asarray(K, dtype=np.float64).copy(),
                baseline=float(baseline),
                bundle_kwargs=clean_kwargs,
            )


@dataclass(frozen=True)
class LocalBACorrection:
    """Compact pose + landmark patch computed off the tracking thread."""

    snapshot_id: int
    frame_index: int
    anchor_kf_id: int
    base_map_revision: int
    base_geometry_revision: int
    base_pose_revision: int
    base_poses: dict[int, np.ndarray]
    corrected_poses: dict[int, np.ndarray]
    landmark_updates: dict[int, np.ndarray]
    culled_observations: tuple[tuple[int, int], ...]
    report: dict[str, Any]
    elapsed_ms: float


class LocalMappingWorker:
    """Dedicated background worker running local BA with coalescing input and non-blocking inbox."""

    def __init__(self, *, solver=None) -> None:
        self._solver = solver if solver is not None else local_bundle_adjustment
        self._cond = Condition()
        self._pending_snapshot: LocalBASnapshot | None = None
        self._stop = False
        self._busy = False

        self._inbox_lock = Lock()
        self._pending_correction: LocalBACorrection | None = None

        self._stats_lock = Lock()
        self._submitted = 0
        self._coalesced = 0
        self._completed = 0
        self._applied_reports = 0
        self._failed = 0
        self._solve_times_ms: deque[float] = deque(maxlen=128)
        self._last_error: str | None = None

        self._thread = Thread(
            target=self._run_loop, name="local-mapping-ba", daemon=True
        )
        self._thread.start()

    def submit(self, snapshot: LocalBASnapshot) -> None:
        """Enqueue a snapshot; coalesces by replacing any unstarted snapshot."""
        with self._cond:
            if self._stop:
                return
            if self._pending_snapshot is not None:
                with self._stats_lock:
                    self._coalesced += 1
            self._pending_snapshot = snapshot
            with self._stats_lock:
                self._submitted += 1
            self._cond.notify_all()

    def poll_correction(self) -> LocalBACorrection | None:
        """Non-blocking fetch of the latest completed LocalBACorrection, if any."""
        with self._inbox_lock:
            corr = self._pending_correction
            self._pending_correction = None
            return corr

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until any queued and in-flight BA solve finishes."""
        deadline = None if timeout is None else (time.perf_counter() + max(0.0, float(timeout)))
        with self._cond:
            while (self._pending_snapshot is not None or self._busy) and not self._stop:
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        return False
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait(timeout=0.05)
            return True

    def close(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        if wait:
            self.wait_idle(timeout=timeout)
        with self._cond:
            self._stop = True
            self._pending_snapshot = None
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            times = list(self._solve_times_ms)
            p50 = float(np.percentile(times, 50)) if times else 0.0
            p95 = float(np.percentile(times, 95)) if times else 0.0
            return {
                "submitted": self._submitted,
                "coalesced": self._coalesced,
                "completed": self._completed,
                "failed": self._failed,
                "ba_p50_ms": round(p50, 3),
                "ba_p95_ms": round(p95, 3),
                "last_error": self._last_error,
            }

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending_snapshot is None and not self._stop:
                    self._cond.wait()
                if self._stop and self._pending_snapshot is None:
                    return
                snapshot = self._pending_snapshot
                self._pending_snapshot = None
                self._busy = True
            assert snapshot is not None
            try:
                correction = self._solve_snapshot(snapshot)
                if correction is not None:
                    with self._inbox_lock:
                        self._pending_correction = correction
            except Exception as exc:
                with self._stats_lock:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _solve_snapshot(self, snapshot: LocalBASnapshot) -> LocalBACorrection | None:
        t0 = time.perf_counter()
        shadow = _ShadowMapState(snapshot)
        report = self._solver(
            shadow,
            snapshot.K,
            snapshot.baseline,
            **snapshot.bundle_kwargs,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self._stats_lock:
            self._completed += 1
            self._solve_times_ms.append(elapsed_ms)

        if not isinstance(report, dict) or not report.get("applied", False):
            return None
        if shadow.captured_poses is None:
            return None

        moved_base: dict[int, np.ndarray] = {}
        moved_corrected: dict[int, np.ndarray] = {}
        for kf_id in snapshot.selected_keyframes:
            orig = snapshot.keyframe_poses[kf_id]
            new_pose = shadow.captured_poses.get(kf_id)
            if new_pose is not None and not np.array_equal(new_pose, orig):
                moved_base[kf_id] = orig.copy()
                moved_corrected[kf_id] = new_pose.copy()

        # Also keep anchor_kf_id in base/corrected if it moved
        if not moved_corrected and not shadow.captured_landmark_updates and not shadow.culled_observations:
            return None

        return LocalBACorrection(
            snapshot_id=snapshot.snapshot_id,
            frame_index=snapshot.frame_index,
            anchor_kf_id=snapshot.anchor_kf_id,
            base_map_revision=snapshot.map_revision,
            base_geometry_revision=snapshot.geometry_revision,
            base_pose_revision=snapshot.pose_revision,
            base_poses=moved_base,
            corrected_poses=moved_corrected,
            landmark_updates=shadow.captured_landmark_updates or {},
            culled_observations=tuple(shadow.culled_observations),
            report=dict(report),
            elapsed_ms=elapsed_ms,
        )


def apply_local_ba_correction(
    state: Any,
    correction: LocalBACorrection,
    *,
    kf_landmark_row_cache: dict[int, Any] | None = None,
) -> bool:
    """Atomically apply a ``LocalBACorrection`` on the tracking thread.

    Uses ``MapState.apply_corrections()`` (the vectorized bulk landmark path),
    rigidly propagates any keyframes/landmarks appended after the BA snapshot,
    and updates ``landmark_ids_version`` whenever outlier observations are culled.
    """
    with state.lock:
        if state.pose_revision != correction.base_pose_revision:
            return False
        for kf_id, expected_pose in correction.base_poses.items():
            kf = state.keyframes.get(kf_id)
            if kf is None or not np.array_equal(kf.pose, expected_pose):
                return False

        all_kf_ids = sorted(state.keyframes)
        if not all_kf_ids:
            return False

        # Compute rigid propagation delta for any keyframes added AFTER anchor_kf_id.
        anchor_id = correction.anchor_kf_id
        if anchor_id in correction.corrected_poses and anchor_id in correction.base_poses:
            old_anchor = correction.base_poses[anchor_id]
            new_anchor = correction.corrected_poses[anchor_id]
            tail_R = new_anchor[:3, :3] @ old_anchor[:3, :3].T
            tail_t = new_anchor[:3, 3] - tail_R @ old_anchor[:3, 3]
            has_tail_delta = True
        else:
            tail_R = np.eye(3, dtype=np.float64)
            tail_t = np.zeros(3, dtype=np.float64)
            has_tail_delta = False

        expanded_poses: dict[int, np.ndarray] = {}
        tail_kfs: set[int] = set()
        for kf_id in all_kf_ids:
            if kf_id in correction.corrected_poses:
                expanded_poses[kf_id] = correction.corrected_poses[kf_id]
            elif has_tail_delta and kf_id > anchor_id:
                pose = state.keyframes[kf_id].pose.copy()
                pose[:3, :3] = tail_R @ pose[:3, :3]
                pose[:3, 3] = tail_R @ pose[:3, 3] + tail_t
                expanded_poses[kf_id] = pose
                tail_kfs.add(kf_id)
            else:
                expanded_poses[kf_id] = state.keyframes[kf_id].pose

        # Build landmark_updates restricted to live landmarks, plus rigid propagation
        # for landmarks anchored to moved window keyframes or tail keyframes that were
        # created after the snapshot was taken.
        live_landmarks = state.landmarks
        merged_lm_updates: dict[int, np.ndarray] = {}
        for lid, pos in correction.landmark_updates.items():
            if lid in live_landmarks:
                merged_lm_updates[lid] = pos

        moved_anchors = set(correction.corrected_poses) | tail_kfs
        if moved_anchors and hasattr(state, "index") and len(state.index) == len(live_landmarks):
            anc_map = state.index._anchor_landmarks
            id_to_slot = state.index._id_to_slot
            idx_pos = state.index._positions
            for kf_id in moved_anchors:
                lm_set = anc_map.get(kf_id)
                if not lm_set:
                    continue
                missing_ids = [lid for lid in lm_set if lid not in merged_lm_updates and lid in live_landmarks]
                if not missing_ids:
                    continue
                old_p = state.keyframes[kf_id].pose
                new_p = expanded_poses[kf_id]
                rot = new_p[:3, :3] @ old_p[:3, :3].T
                trans = new_p[:3, 3] - rot @ old_p[:3, 3]
                slots = [id_to_slot[lid] for lid in missing_ids]
                moved_arr = (idx_pos[slots] @ rot.T) + trans
                for lid, row in zip(missing_ids, moved_arr):
                    merged_lm_updates[lid] = row
        elif moved_anchors:
            for lid, lm in live_landmarks.items():
                if lid not in merged_lm_updates and lm.anchor in moved_anchors:
                    old_p = state.keyframes[lm.anchor].pose
                    new_p = expanded_poses[lm.anchor]
                    rot = new_p[:3, :3] @ old_p[:3, :3].T
                    trans = new_p[:3, 3] - rot @ old_p[:3, 3]
                    merged_lm_updates[lid] = rot @ lm.position + trans

        applied = state.apply_corrections(
            state.revision,
            expanded_poses,
            propagate_landmarks=False,
            landmark_updates=merged_lm_updates if merged_lm_updates else None,
        )
        if not applied:
            return False

        # Apply chi2 outlier observation removals & bump landmark_ids_version safely
        culled_count = 0
        touched_kfs: set[int] = set()
        for lid, kf_id in correction.culled_observations:
            lm = state.landmarks.get(lid)
            if lm is None or kf_id not in lm.observations or len(lm.observations) <= 1:
                continue
            removed = state.remove_observation(lid, kf_id)
            if removed is not None:
                culled_count += 1
                kf = state.keyframes.get(kf_id)
                if kf is not None:
                    mask = kf.landmark_ids == lid
                    if mask.any():
                        kf.landmark_ids[mask] = -1
                        touched_kfs.add(kf_id)
                        if kf_landmark_row_cache is not None:
                            entry = kf_landmark_row_cache.get(kf_id)
                            if entry is not None:
                                entry[2].pop(lid, None)

        if touched_kfs:
            for kf_id in touched_kfs:
                kf = state.keyframes[kf_id]
                kf.landmark_ids_version = int(getattr(kf, "landmark_ids_version", 0)) + 1
            state.landmark_ids_version = int(getattr(state, "landmark_ids_version", 0)) + 1

        return True
