"""Unit tests for IncrementalMapIndex (F6)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from map_index import IncrementalMapIndex
from slam_state import Landmark, MapState, Observation


def test_incremental_conflict_tracking_matches_bruteforce():
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)

    lm0 = state.add_landmark(
        [1.0, 0.0, 10.0], desc, anchor=0,
        observations={0: Observation(np.array([100.0, 200.0], np.float32), 80.0)},
    )
    lm1 = state.add_landmark(
        [2.0, 0.0, 10.0], desc, anchor=0,
        observations={0: Observation(np.array([150.0, 200.0], np.float32), 130.0)},
    )
    assert state.index.conflicted_landmark_ids() == frozenset()

    # Duplicate pixel claim on keyframe 0 creates conflict for lm0 and lm2
    lm2 = state.add_landmark(
        [1.1, 0.0, 10.5], desc, anchor=0,
        observations={0: Observation(np.array([100.0, 200.0], np.float32), 81.0)},
    )
    assert state.index.conflicted_landmark_ids() == frozenset({lm0, lm2})

    # Removing lm2 resolves the conflict cleanly
    state.remove_landmark(lm2)
    assert state.index.conflicted_landmark_ids() == frozenset()
    assert lm1 not in state.index.conflicted_landmark_ids()


def test_frustum_query_and_far_bearings():
    idx = IncrementalMapIndex()
    desc = np.zeros(128, dtype=np.float32)
    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]])
    T_cw = np.eye(4)

    # Close visible point
    idx.on_landmark_added(Landmark(0, np.array([0.0, 0.0, 15.0]), desc, 0, {}, is_far=False))
    # Behind camera
    idx.on_landmark_added(Landmark(1, np.array([0.0, 0.0, -5.0]), desc, 0, {}, is_far=False))
    # Out of horizontal FOV
    idx.on_landmark_added(Landmark(2, np.array([50.0, 0.0, 10.0]), desc, 0, {}, is_far=False))
    # Far bearing point (Z = 200 m > max_depth=80 m)
    idx.on_landmark_added(Landmark(3, np.array([2.0, 0.0, 200.0]), desc, 0, {}, is_far=True))

    res_close = idx.query_frustum(T_cw, K, (1241, 376), max_depth=80.0, include_far_bearings=False)
    assert list(res_close.landmark_ids) == [0]
    assert res_close.projected_uv[0] == pytest.approx([620.0, 188.0])
    assert res_close.depths[0] == pytest.approx(15.0)

    res_with_far = idx.query_frustum(T_cw, K, (1241, 376), max_depth=80.0, include_far_bearings=True)
    assert set(res_with_far.landmark_ids.tolist()) == {0, 3}


def test_prune_stale_landmarks_preserves_multiview():
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)

    # Old single-view landmark -> should be pruned
    stale_id = state.add_landmark(
        [0.0, 0.0, 10.0], desc, anchor=0,
        observations={0: Observation(np.array([300.0, 150.0], np.float32), 280.0)},
    )
    state.landmarks[stale_id].misses = 3

    # Old multi-view landmark -> must never be pruned
    multi_id = state.add_landmark(
        [1.0, 0.0, 12.0], desc, anchor=0,
        observations={0: Observation(np.array([400.0, 150.0], np.float32), 380.0)},
    )
    state.add_observation(multi_id, 1, Observation(np.array([405.0, 150.0], np.float32), 385.0))

    # Young single-view landmark -> must not be pruned yet
    young_id = state.add_landmark(
        [2.0, 0.0, 12.0], desc, anchor=4,
        observations={4: Observation(np.array([500.0, 150.0], np.float32), 480.0)},
    )

    pruned = state.index.prune_stale_landmarks(state, current_kf_id=5, min_age_kfs=3)
    assert pruned == 1
    assert stale_id not in state.landmarks
    assert multi_id in state.landmarks
    assert young_id in state.landmarks
    assert len(state.index) == 2
