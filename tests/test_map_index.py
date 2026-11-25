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


def test_anchor_correction_slots_follow_insertions_and_swap_removal():
    state = MapState(metric=True, defer_bulk_landmark_positions=True)
    index = state.index
    descriptor = np.zeros(8, dtype=np.float32)

    def add(anchor, x):
        return state.add_landmark([x, 0., 10.], descriptor, anchor,
                                  {anchor: Observation(np.array([x, 20.], dtype=np.float32))})

    def assert_slots(anchor):
        ids, slots = index._correction_slots_for_anchor(anchor)
        assert ids == list(index._anchor_landmarks[anchor])
        np.testing.assert_array_equal(index._ids[slots], ids)
        return slots

    removed = add(0, 10.)
    add(0, 20.)
    add(2, 30.)
    moved = add(1, 40.)
    slots_before = assert_slots(1)
    assert_slots(0)
    unchanged = assert_slots(2)

    state.remove_landmark(removed)
    assert_slots(0)
    slots_after = assert_slots(1)
    # Dense index deletion fills the vacated slot with the last landmark.
    assert slots_before[0] == 3 and slots_after[0] == 0 and index._ids[0] == moved
    assert assert_slots(2) is unchanged

    state.add_landmarks_batch(np.array([[1., 0., 10.]]), descriptor[None], 0,
                              np.array([[50., 20.]]), np.array([40.]), [0])
    assert len(assert_slots(0)) == 2
    index.rebuild_from_map(state)
    assert assert_slots(2) is not unchanged
    assert_slots(0)
    assert_slots(1)


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


def test_remove_observation_clears_pixel_conflict_and_reverse_kf_index():
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)

    shared_px = np.array([250.5, 140.25], np.float32)
    lm0 = state.add_landmark(
        [0.5, 0.0, 10.0], desc, anchor=0,
        observations={
            0: Observation(np.array([240.0, 140.0], np.float32), 200.0),
            1: Observation(shared_px, 210.0),
        },
    )
    lm1 = state.add_landmark(
        [0.6, 0.0, 10.2], desc, anchor=0,
        observations={
            0: Observation(np.array([260.0, 140.0], np.float32), 220.0),
            1: Observation(shared_px, 211.0),
        },
    )
    assert state.index.conflicted_landmark_ids() == frozenset({lm0, lm1})
    assert state.index.landmarks_for_keyframes([1]) == {lm0, lm1}

    # Removing lm0's observation on kf 1 must clear conflict for BOTH landmarks and update kf 1 reverse index
    removed = state.remove_observation(lm0, 1)
    assert removed is not None
    assert 1 not in state.landmarks[lm0].observations
    assert state.index.conflicted_landmark_ids() == frozenset()
    assert state.index.landmarks_for_keyframes([1]) == {lm1}
    assert state.index.landmarks_for_keyframes([0]) == {lm0, lm1}


def test_local_bundle_outlier_culling_updates_map_index():
    from local_bundle import local_bundle_adjustment
    from slam_state import MappingKeyframe

    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]])
    baseline = 0.54
    fx_b = K[0, 0] * baseline
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)

    # Create 5 keyframes along +Z with exact ground-truth poses
    kf_poses = [np.eye(4) for _ in range(5)]
    for kf_id, pose in enumerate(kf_poses):
        pose[2, 3] = float(kf_id)
        state.keyframes[kf_id] = MappingKeyframe(
            id=kf_id,
            frame=kf_id * 5,
            pose=pose.copy(),
            pixels=np.empty((0, 2), np.float32),
            descriptors=np.empty((0, 128), np.float32),
            landmark_ids=np.empty(0, np.int64),
        )
        state.record(pose, "tracking", anchor=kf_id)

    def project_obs(pt_w, pose):
        pc = pose[:3, :3].T @ (pt_w - pose[:3, 3])
        h = K @ pc
        uv = h[:2] / h[2]
        right_u = float(uv[0] - fx_b / pc[2])
        return Observation(np.asarray(uv, dtype=np.float32), right_u=right_u)

    # Add clean inlier landmarks observed across all 5 keyframes so BA converges cleanly
    rng = np.random.default_rng(7)
    for _ in range(18):
        pt = np.array([rng.uniform(-4.0, 4.0), rng.uniform(-1.5, 1.5), rng.uniform(10.0, 24.0)])
        obs = {kf_id: project_obs(pt, kf_poses[kf_id]) for kf_id in range(5)}
        state.add_landmark(pt, desc, anchor=0, observations=obs)

    # Add one clean landmark lm_good and one landmark lm_bad whose kf=4 observation
    # falsely claims lm_good's exact pixel (both a gross outlier and a pixel conflict).
    pt_good = np.array([2.5, -0.5, 14.0])
    good_obs_kf4 = project_obs(pt_good, kf_poses[4])
    lm_good = state.add_landmark(
        pt_good, desc, anchor=0,
        observations={kf_id: project_obs(pt_good, kf_poses[kf_id]) for kf_id in range(5)},
    )

    pt_bad = np.array([-3.0, 0.8, 17.0])
    bad_obs = {kf_id: project_obs(pt_bad, kf_poses[kf_id]) for kf_id in range(4)}
    bad_obs[4] = Observation(good_obs_kf4.pixel.copy(), right_u=good_obs_kf4.right_u)
    lm_bad = state.add_landmark(pt_bad, desc, anchor=0, observations=bad_obs)

    assert {lm_good, lm_bad}.issubset(state.index.conflicted_landmark_ids())
    assert lm_bad in state.index.landmarks_for_keyframes([4])

    report = local_bundle_adjustment(
        state,
        K,
        baseline=baseline,
        window=5,
        max_landmarks=100,
        use_analytic_jacobian=True,
        use_map_index=True,
        cull_outliers_chi2=True,
    )
    assert report["applied"] is True
    assert report.get("culled_outlier_observations", 0) >= 1
    assert 4 not in state.landmarks[lm_bad].observations
    assert lm_bad not in state.index.landmarks_for_keyframes([4])
    assert lm_good not in state.index.conflicted_landmark_ids()
    assert lm_bad not in state.index.conflicted_landmark_ids()


def test_frustum_query_top_k_and_exclude_conflicted_matches_naive_sort():
    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]])
    T_cw = np.eye(4)
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)
    rng = np.random.default_rng(42)

    # Create 60 visible landmarks with deliberate ties in (max_obs, num_obs)
    # plus 2 conflicted landmarks and 5 removed landmarks to exercise swap-delete.
    for i in range(65):
        pos = np.array([rng.uniform(-4.0, 4.0), rng.uniform(-1.5, 1.5), rng.uniform(8.0, 40.0)])
        base_kf = i % 6
        obs = {
            base_kf: Observation(np.array([50.0 + i * 3.0, 100.0 + (i % 5)], np.float32), 30.0)
        }
        if i % 3 == 0:
            obs[base_kf + 2] = Observation(np.array([52.0 + i * 3.0, 101.0], np.float32), 31.0)
        if i % 5 == 0:
            obs[base_kf + 4] = Observation(np.array([54.0 + i * 3.0, 102.0], np.float32), 32.0)
        state.add_landmark(pos, desc, anchor=base_kf, observations=obs)

    # Create a pixel conflict between landmark 10 and landmark 11 on kf 9
    conflict_px = np.array([333.5, 177.25], np.float32)
    state.add_observation(10, 9, Observation(conflict_px, 310.0))
    state.add_observation(11, 9, Observation(conflict_px, 311.0))

    # Remove an observation (lowering max_obs_kf) and remove several landmarks (swap-delete)
    state.remove_observation(15, max(state.landmarks[15].observations))
    for dead_id in (2, 17, 31, 44):
        state.remove_landmark(dead_id)

    # Reference naive query + filter + stable Python sort
    full = state.index.query_frustum(T_cw, K, (1241, 376), max_depth=1e6, margin_px=40.0, include_far_bearings=True)
    amb = state.index.conflicted_landmark_ids()
    ref_ids = [int(lid) for lid in full.landmark_ids if int(lid) in state.landmarks and int(lid) not in amb]
    ref_order = sorted(
        range(len(ref_ids)),
        key=lambda idx: (
            max(state.landmarks[ref_ids[idx]].observations),
            len(state.landmarks[ref_ids[idx]].observations),
        ),
        reverse=True,
    )[:25]
    expected_ids = [ref_ids[idx] for idx in ref_order]

    fast = state.index.query_frustum(
        T_cw, K, (1241, 376),
        max_depth=1e6, margin_px=40.0, include_far_bearings=True,
        exclude_conflicted=True, max_landmarks=25,
    )
    assert fast.landmark_ids.tolist() == expected_ids
    assert 10 not in fast.landmark_ids.tolist()
    assert 11 not in fast.landmark_ids.tolist()


def test_sparse_apply_corrections_matches_dense_and_preserves_atomicity():
    from slam_state import MappingKeyframe

    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)
    for kf_id in range(6):
        pose = np.eye(4)
        pose[2, 3] = float(kf_id * 2.0)
        state.keyframes[kf_id] = MappingKeyframe(
            kf_id, kf_id * 4, pose.copy(),
            np.empty((0, 2), np.float32), np.empty((0, 128), np.float32), np.empty(0, int),
        )
        for offset in range(4):
            fp = pose.copy()
            fp[2, 3] += 0.5 * offset
            state.record(fp, "tracking", anchor=kf_id)
        for j in range(5):
            pt = np.array([float(j) - 2.0, 0.3, float(kf_id * 2.0) + 10.0])
            state.add_landmark(
                pt, desc, anchor=kf_id,
                observations={kf_id: Observation(np.array([100.0 + j * 10.0, 150.0], np.float32))},
            )

    # 1. Sparse propagate_landmarks=True moving only keyframes 4 and 5
    corrected = {k: kf.pose.copy() for k, kf in state.keyframes.items()}
    corrected[4][0, 3] += 0.25
    corrected[5][0, 3] += 0.50
    old_lm_kf0 = state.landmarks[0].position.copy()
    old_lm_kf4 = state.landmarks[20].position.copy()
    old_lm_kf5 = state.landmarks[25].position.copy()
    assert state.apply_corrections(state.revision, corrected, propagate_landmarks=True)
    np.testing.assert_array_equal(state.landmarks[0].position, old_lm_kf0)
    np.testing.assert_allclose(state.landmarks[20].position, old_lm_kf4 + [0.25, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(state.landmarks[25].position, old_lm_kf5 + [0.50, 0.0, 0.0], atol=1e-12)
    # Index positions must match MapState landmarks
    for lid, lm in state.landmarks.items():
        slot = state.index._id_to_slot[lid]
        np.testing.assert_allclose(state.index._positions[slot], lm.position, atol=1e-12)
    # Propagated frame poses and relative poses must stay consistent
    for frame, (p, a, rel) in enumerate(zip(state.poses, state.pose_anchors, state.relative_poses)):
        np.testing.assert_allclose(state.keyframes[a].pose @ rel, p, atol=1e-12)

    # 2. Atomic rollback on nonfinite landmark_update
    rev_before = state.revision
    pose5_before = state.keyframes[5].pose.copy()
    bad_corrected = {k: kf.pose.copy() for k, kf in state.keyframes.items()}
    bad_corrected[5][1, 3] += 1.0
    with pytest.raises(ValueError, match="Nonfinite corrected landmark"):
        state.apply_corrections(
            state.revision, bad_corrected, propagate_landmarks=False,
            landmark_updates={25: np.array([np.nan, 0.0, 1.0])},
        )
    assert state.revision == rev_before
    np.testing.assert_array_equal(state.keyframes[5].pose, pose5_before)


def test_high_miss_landmarks_incremental_and_fallback():
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)
    ids = [
        state.add_landmark([float(i), 0.0, 10.0], desc, anchor=0, observations={})
        for i in range(6)
    ]
    assert state.high_miss_landmarks(5) == {}

    # Set misses >= 5 out of order; result must preserve ascending ID order
    state.landmarks[ids[4]].misses = 5
    state.landmarks[ids[1]].misses = 7
    state.landmarks[ids[3]].misses = 5
    state.landmarks[ids[2]].misses = 4
    dropped = state.high_miss_landmarks(5)
    assert list(dropped.keys()) == [ids[1], ids[3], ids[4]]

    # Lowering misses removes landmark from high-miss set
    state.landmarks[ids[3]].misses = 0
    assert list(state.high_miss_landmarks(5).keys()) == [ids[1], ids[4]]

    # Removing landmark cleans tracking cleanly
    state.remove_landmark(ids[1])
    assert list(state.high_miss_landmarks(5).keys()) == [ids[4]]

    # Direct dict injection bypasses add_landmark -> automatic fallback to full scan
    injected = Landmark(999, np.array([1.0, 0.0, 10.0]), desc, 0, {}, misses=6)
    state.landmarks[999] = injected
    dropped_fb = state.high_miss_landmarks(5)
    assert set(dropped_fb.keys()) == {ids[4], 999}


def test_query_frustum_recent_keyframe_fast_path_and_fallback_exact_parity():
    K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 188.0], [0.0, 0.0, 1.0]])
    state = MapState(metric=True)
    desc = np.zeros(128, dtype=np.float32)
    rng = np.random.default_rng(19)

    for kf_id in range(30):
        n_pts = 300
        for j in range(n_pts):
            if kf_id < 18 and j < 150:
                pos = np.array([rng.uniform(-4.0, 4.0), rng.uniform(-1.5, 1.5), -rng.uniform(10.0, 40.0)])
            else:
                pos = np.array([rng.uniform(-4.0, 4.0), rng.uniform(-1.5, 1.5), rng.uniform(10.0, 40.0)])
            obs = {kf_id: Observation(np.array([10.0 + (j % 100) * 5.0, 20.0 + (j // 100) * 5.0], np.float32), 5.0)}
            if kf_id > 0 and j % 7 == 0:
                obs[kf_id - 1] = Observation(np.array([11.0 + (j % 100) * 5.0, 21.0 + (j // 100) * 5.0], np.float32), 6.0)
            state.add_landmark(pos, desc, anchor=kf_id, observations=obs)

    assert len(state.index) == 9000

    def full_scan_reference(T_cw, k):
        full = state.index.query_frustum(
            T_cw, K, (1241, 376), max_depth=80.0, margin_px=32.0,
            include_far_bearings=True, exclude_conflicted=True, max_landmarks=None,
        )
        slots = np.array([state.index._id_to_slot[int(lid)] for lid in full.landmark_ids], dtype=np.int64)
        keys = (
            ((state.index._max_obs_kf[slots].astype(np.int64) + 1) << 32)
            | state.index._num_obs[slots].astype(np.int64)
        )
        order = np.lexsort((np.arange(len(slots)), -keys))[:k]
        return full.landmark_ids[order], full.projected_uv[order], full.depths[order]

    # Case 1: Forward camera (+Z) -> hits recent-keyframe fast path
    T_fwd = np.eye(4)
    fast_fwd = state.index.query_frustum(
        T_fwd, K, (1241, 376), max_depth=80.0, margin_px=32.0,
        include_far_bearings=True, exclude_conflicted=True, max_landmarks=200,
    )
    ref_ids, ref_uv, ref_d = full_scan_reference(T_fwd, 200)
    np.testing.assert_array_equal(fast_fwd.landmark_ids, ref_ids)
    np.testing.assert_array_equal(fast_fwd.projected_uv, ref_uv)
    np.testing.assert_array_equal(fast_fwd.depths, ref_d)

    # Case 2: Backward camera (-Z) -> recent keyframes have 0 visible, must fall back to full scan
    T_back = np.eye(4)
    T_back[0, 0] = -1.0
    T_back[2, 2] = -1.0
    fast_back = state.index.query_frustum(
        T_back, K, (1241, 376), max_depth=80.0, margin_px=32.0,
        include_far_bearings=True, exclude_conflicted=True, max_landmarks=200,
    )
    ref_ids_b, ref_uv_b, ref_d_b = full_scan_reference(T_back, 200)
    assert len(fast_back.landmark_ids) == 200
    np.testing.assert_array_equal(fast_back.landmark_ids, ref_ids_b)
    np.testing.assert_array_equal(fast_back.projected_uv, ref_uv_b)
    np.testing.assert_array_equal(fast_back.depths, ref_d_b)



