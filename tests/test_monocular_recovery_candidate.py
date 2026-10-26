"""Manufactured contracts for the supplied October 23 recovery candidate.

PnP stubs isolate search/transaction correctness; these are not pose-accuracy tests.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from shared_slam import MappingConfig, SharedSlam
from slam_state import MappingKeyframe, Observation

K = np.array([[250., 0., 320.], [0., 250., 240.], [0., 0., 1.]])
SIZE = (640, 480)


@pytest.fixture
def slam():
    obj = SharedSlam(K, config=MappingConfig(loop_mode='off', bundle_enabled=False,
                                            monocular_recovery_fallback=True))
    yield obj
    obj.close(finish=False)


def pixels(n):
    return np.array([[40. + (i % 6)*95., 40. + ((i//6) % 5)*85.]
                     for i in range(n)], np.float32)


def add_pool(slam, ident, count=20, offset=0, ids=None, x=1.):
    p = pixels(count)
    descriptors = np.zeros((count, 128), np.float32)
    descriptors[:, 0] = np.arange(offset, offset + count)
    if ids is None:
        ids = [slam.map.add_landmark([float(i), 0., 5.], descriptors[i], ident,
                                    {ident: Observation(p[i])}) for i in range(count)]
    pose = np.eye(4); pose[0, 3] = x
    frame = MappingKeyframe(ident, ident, pose, p, descriptors, np.asarray(ids, int).copy())
    slam.map.keyframes[ident] = frame
    return frame


def marker_matches(first, second):
    """Exact descriptor IDs make input-pool behavior analytically controlled."""
    targets = {float(d[0]): i for i, d in enumerate(second)}
    return np.asarray([(i, targets[float(d[0])]) for i, d in enumerate(first)
                       if float(d[0]) in targets], int).reshape(-1, 2)


def query(count=20, offset=0):
    descriptors = np.zeros((count, 128), np.float32)
    descriptors[:, 0] = np.arange(offset, offset + count)
    return pixels(count), descriptors


def test_ranking_ignores_dead_3d_and_deduplicates_landmark_pool(slam, monkeypatch):
    monkeypatch.setattr(slam, '_match', marker_matches)
    p, d = query()
    live = add_pool(slam, 1)
    add_pool(slam, 2, ids=live.landmark_ids)
    add_pool(slam, 3, count=40, ids=np.full(40, 99999))
    with slam.map.lock:
        assert slam._rank_monocular_recovery(p, d, SIZE, [1, 2, 3, 999]) == [(20, 2)]
        assert slam._rank_monocular_recovery(p, d, SIZE, [1, 2], excluded_keyframes=[1]) == []


def test_ranking_physical_target_aliases_cannot_inflate_support(slam, monkeypatch):
    monkeypatch.setattr(slam, '_match', marker_matches)
    add_pool(slam, 0)
    p, d = query()
    p[1] = p[0]  # Two distinct landmarks claiming one target observation are ambiguous.
    with slam.map.lock:
        assert slam._rank_monocular_recovery(p, d, SIZE, [0]) == []


def test_ranking_rejects_cross_landmark_source_pixel_conflicts(slam, monkeypatch):
    monkeypatch.setattr(slam, '_match', marker_matches)
    frame = add_pool(slam, 0)
    frame.pixels[1] = frame.pixels[0]
    p, d = query()
    with slam.map.lock:
        assert slam._rank_monocular_recovery(p, d, SIZE, [0]) == []


def test_ranking_enforces_configured_input_floor_and_coverage(slam, monkeypatch):
    monkeypatch.setattr(slam, '_match', marker_matches)
    slam.config = MappingConfig(loop_mode='off', min_inliers=25,
                               monocular_recovery_fallback=True)
    add_pool(slam, 0, 24)
    p, d = query(24)
    with slam.map.lock:
        assert slam._rank_monocular_recovery(p, d, SIZE, [0]) == []
    add_pool(slam, 1, 25)
    p, d = query(25)
    with slam.map.lock:
        assert slam._rank_monocular_recovery(p, d, SIZE, [1]) == [(25, 1)]
        assert slam._rank_monocular_recovery(np.zeros_like(p), d, SIZE, [1]) == []


def test_empty_and_nonviable_outcomes(slam, monkeypatch):
    monkeypatch.setattr(slam, '_match', marker_matches)
    p, d = query()
    result, stats = slam._relocalize(p, d, SIZE)
    assert result is None and stats['monocular_recovery']['outcome'] == 'empty_map'
    add_pool(slam, 0, 19)
    result, stats = slam._relocalize(p, d, SIZE)
    assert result is None and stats['monocular_recovery']['outcome'] == 'no_viable_3d_pool'
    assert stats['monocular_recovery']['attempts'] == []


def setup_budget(slam, monkeypatch, pool_count, proposed):
    monkeypatch.setattr(slam, '_match', marker_matches)
    for ident in range(pool_count):
        add_pool(slam, ident)
    monkeypatch.setattr(slam.retrieval, 'update', lambda *_: None)
    monkeypatch.setattr(slam.retrieval, 'query', lambda *_: proposed)
    calls = []
    def fail(*args, candidate_ids, **kwargs):
        calls.append(frozenset(candidate_ids))
        return None, {'num_matches': 20, 'pose_rejection_reason': 'manufactured_reject'}
    monkeypatch.setattr(slam, '_track', fail)
    return calls


def test_two_pass_budget_is_five_plus_five_distinct_pools(slam, monkeypatch):
    calls = setup_budget(slam, monkeypatch, 12, list(range(8)))
    result, stats = slam._relocalize(*query(), SIZE)
    report = stats['monocular_recovery']
    assert result is None
    assert len(calls) == len(set(calls)) == 10
    assert [a['phase'] for a in report['attempts']] == ['primary']*5 + ['fallback']*5
    assert report['unprobed_fallback_pools'] == 2
    assert report['outcome'] == 'probe_budget_exhausted'
    assert all(not a['geometry_passed'] and not a['selected'] for a in report['attempts'])


def test_short_primary_does_not_transfer_unused_budget(slam, monkeypatch):
    calls = setup_budget(slam, monkeypatch, 9, [0])
    _, stats = slam._relocalize(*query(), SIZE)
    report = stats['monocular_recovery']
    assert len(calls) == 6
    assert report['unprobed_fallback_pools'] == 3
    assert report['outcome'] == 'probe_budget_exhausted'


def test_all_probed_rejected_outcome(slam, monkeypatch):
    calls = setup_budget(slam, monkeypatch, 3, [0])
    _, stats = slam._relocalize(*query(), SIZE)
    assert len(calls) == 3
    assert stats['monocular_recovery']['outcome'] == 'geometry_rejected_all_probed_pools'


def test_primary_success_skips_fallback_and_marks_only_winner(slam, monkeypatch):
    setup_budget(slam, monkeypatch, 8, [0, 1])
    calls = []
    def succeed(*args, candidate_ids, **kwargs):
        ident = next(k for k, f in slam.map.keyframes.items()
                     if set(f.landmark_ids) == candidate_ids)
        calls.append(ident)
        return (np.eye(4), {}), {'num_matches': 20, 'num_inliers': 20+ident}
    monkeypatch.setattr(slam, '_track', succeed)
    result, stats = slam._relocalize(*query(), SIZE)
    report = stats['monocular_recovery']
    assert result is not None and len(calls) == 2
    assert report['outcome'] == 'recovered' and report['fallback_viable_keyframes'] == 0
    assert sum(a['selected'] for a in report['attempts']) == 1
    assert next(a['keyframe'] for a in report['attempts'] if a['selected']) == 1


@pytest.mark.parametrize('mode', ['success', 'failure', 'exception'])
def test_recovery_probes_restore_epoch_and_commit_only_winner(slam, monkeypatch, mode):
    frames = [add_pool(slam, i, count=1) for i in range(3)]
    for lm in slam.map.landmarks.values():
        lm.misses = 7
    initial_tracks = [('initial', np.array([1., 2.]))]
    slam.accepted_tracks = initial_tracks
    calls = []
    def probe(*args, candidate_ids, **kwargs):
        assert all(lm.misses == 7 for lm in slam.map.landmarks.values())
        assert slam.accepted_tracks is initial_tracks
        ident = next(k for k, f in slam.map.keyframes.items()
                     if set(f.landmark_ids) == candidate_ids)
        calls.append(ident)
        for lm in slam.map.landmarks.values():
            lm.misses = ident + 10
        slam.accepted_tracks = [(ident, np.array([float(ident), 0.]))]
        if mode == 'exception' and ident == 1:
            raise RuntimeError('manufactured exception after mutation')
        if mode == 'failure':
            return None, {}
        return (np.eye(4), {}), {'num_inliers': [21, 30, 25][ident]}
    monkeypatch.setattr(slam, '_track', probe)
    ranked = [(30, 0), (29, 1), (28, 2)]
    if mode == 'exception':
        with pytest.raises(RuntimeError, match='manufactured exception'):
            slam._recover_ranked(*query(), SIZE, None, ranked)
    else:
        result, stats = slam._recover_ranked(*query(), SIZE, None, ranked)
        assert len(calls) == 3
        if mode == 'success':
            assert result is not None and stats['relocalization_keyframe'] == 1
            assert all(lm.misses == 11 for lm in slam.map.landmarks.values())
            assert slam.accepted_tracks[0][0] == 1
            return
        assert result is None
    assert all(lm.misses == 7 for lm in slam.map.landmarks.values())
    assert slam.accepted_tracks is initial_tracks


def test_older_covisibility_requires_reciprocal_live_observations(slam):
    slam.config = MappingConfig(loop_mode='off', bundle_window=2)
    current = add_pool(slam, 10, count=2, x=0.)
    lids = current.landmark_ids
    # Equal shared support: farther baseline wins; only two older additions allowed.
    for ident, x in [(1, 1.), (2, 3.), (3, 2.), (4, 100.), (5, 90.)]:
        add_pool(slam, ident, count=2, ids=lids, x=x)
    for lm in slam.map.landmarks.values():
        for ident in [1, 2, 3, 4]:
            lm.observations[ident] = Observation(np.array([0., 0.]))
    # 4 claims observations but its keyframe no longer claims landmarks.
    slam.map.keyframes[4].landmark_ids[:] = -1
    # 5 claims landmarks but has no reciprocal observations.
    assert slam._older_covisible_mapping_references(current, []) == [2, 3]
    assert slam._older_covisible_mapping_references(current, [2, 3]) == [1]
    # Current membership is also required: removing it removes all shared support.
    for lm in slam.map.landmarks.values():
        del lm.observations[current.id]
    assert slam._older_covisible_mapping_references(current, []) == []


def test_covisible_rejects_zero_baseline_future_and_stale_ids(slam):
    current = add_pool(slam, 10, count=1, x=0.)
    lid = int(current.landmark_ids[0]); lm = slam.map.landmarks[lid]
    for ident, x in [(1, 0.), (11, 3.)]:
        add_pool(slam, ident, count=1, ids=[lid], x=x)
        lm.observations[ident] = Observation(np.array([0., 0.]))
    lm.observations[999] = Observation(np.array([0., 0.]))
    current.landmark_ids = np.r_[current.landmark_ids, 8888, -1]
    assert slam._older_covisible_mapping_references(current, []) == []


def test_ranking_real_mutual_descriptor_matcher(slam):
    frame = add_pool(slam, 0)
    rng = np.random.default_rng(34)
    descriptors = rng.normal(size=(20, 128)).astype(np.float32)
    frame.descriptors = descriptors.copy()
    for row, lid in enumerate(frame.landmark_ids):
        slam.map.landmarks[int(lid)].descriptor = descriptors[row].copy()
    with slam.map.lock:
        ranked = slam._rank_monocular_recovery(frame.pixels, descriptors.copy(), SIZE, [0])
    assert ranked == [(20, 0)]


def test_fallback_success_skips_alias_of_already_probed_pool(slam, monkeypatch):
    setup_budget(slam, monkeypatch, 3, [0])
    # Newer keyframe is an alias of primary pool, and must not be retried in fallback.
    add_pool(slam, 9, ids=slam.map.keyframes[0].landmark_ids)
    calls = []
    primary_pool = frozenset(slam.map.keyframes[0].landmark_ids)
    def probe(*args, candidate_ids, **kwargs):
        pool = frozenset(candidate_ids); calls.append(pool)
        if pool == primary_pool:
            return None, {'num_matches': 20, 'pose_rejection_reason': 'manufactured_reject'}
        return (np.eye(4), {}), {'num_matches': 20, 'num_inliers': 20}
    monkeypatch.setattr(slam, '_track', probe)
    result, stats = slam._relocalize(*query(), SIZE)
    report = stats['monocular_recovery']
    assert result is not None and len(calls) == len(set(calls)) == 3
    assert calls.count(primary_pool) == 1
    assert [a['phase'] for a in report['attempts']] == ['primary', 'fallback', 'fallback']
    assert report['fallback_viable_keyframes'] == 2
    assert report['outcome'] == 'recovered'
    selected = [a for a in report['attempts'] if a['selected']]
    assert len(selected) == 1 and selected[0]['phase'] == 'fallback'
    assert all(a['geometry_passed'] for a in report['attempts'][1:])


def test_covisible_shared_count_precedes_baseline_tiebreak(slam):
    current = add_pool(slam, 10, count=2, x=0.)
    lids = current.landmark_ids
    add_pool(slam, 1, count=2, ids=lids, x=1.)
    add_pool(slam, 2, count=1, ids=lids[:1], x=100.)
    for lid in lids:
        slam.map.landmarks[int(lid)].observations[1] = Observation(np.array([0., 0.]))
    slam.map.landmarks[int(lids[0])].observations[2] = Observation(np.array([0., 0.]))
    assert slam._older_covisible_mapping_references(current, []) == [1, 2]
