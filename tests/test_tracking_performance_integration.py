"""Performance wiring retains map corrections, observations and geometry-only recovery."""
import numpy as np
from shared_slam import SharedSlam, MappingConfig, StereoCamera
from performance import PerformanceConfig

K = np.array([[250., 0, 320], [0, 250, 240], [0, 0, 1.]])


def test_landmark_cache_refreshes_positions_after_geometry_change_and_removal():
    slam = SharedSlam(K, config=MappingConfig(loop_mode='off'))
    try:
        first = slam.map.add_landmark([0, 0, 5], np.ones(128), 0, {})
        second = slam.map.add_landmark([1, 0, 5], np.zeros(128), 0, {})
        original = slam._cached_landmarks()[3].copy()
        slam.map.landmarks[first].position += [2, 0, 0]
        slam.map.revision += 1
        assert not np.array_equal(slam._cached_landmarks()[3], original)
        del slam.map.landmarks[second]
        assert [l.id for l in slam._cached_landmarks()[2]] == [first]
        third = slam.map.add_landmark([3, 0, 6], np.full(128, 3), 0, {})
        assert third in slam._cached_landmarks()[5]
    finally:
        slam.close()


def test_landmark_cache_materializes_arrays_under_map_lock(monkeypatch):
    slam = SharedSlam(K, config=MappingConfig(loop_mode='off'))
    try:
        slam.map.add_landmark([0, 0, 5], np.ones(128), 0, {})
        original = np.asarray
        ownership = []
        def inspect(*args, **kwargs):
            ownership.append(slam.map.lock._is_owned())
            return original(*args, **kwargs)
        monkeypatch.setattr(np, 'asarray', inspect)
        slam._cached_landmarks()
        assert ownership and all(ownership)
    finally:
        slam.close()


def test_appearance_view_excludes_appended_flow_descriptors_and_shares_buffer():
    slam = SharedSlam(K, config=MappingConfig(loop_mode='off'))
    try:
        ident = slam.map.add_landmark([0, 0, 5], np.ones(128), 0, {})
        slam.accepted_tracks = [(ident, np.array([320., 240.]))]
        key = slam._keyframe(0, np.eye(4), np.array([[100., 100.]]), np.zeros((1, 128)),
                             np.full((1, 3), np.nan), np.array([np.nan]), {})
        frame = slam.map.keyframes[key]
        assert len(frame.descriptors) == 2 and len(frame.retrieval_descriptors) == 1
        assert np.shares_memory(frame.retrieval_descriptors, frame.descriptors)
        assert frame.landmark_ids[1] == ident
    finally:
        slam.close()


def test_batched_flow_stereo_observations_match_reference_path():
    q = np.array([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 250], [0, 0, 5, 0]], float)
    outputs = []
    for optimized in (False, True):
        slam = SharedSlam(K, stereo=StereoCamera(None, q, .2),
                          config=MappingConfig(loop_mode='off'),
                          performance=PerformanceConfig(cpu_optimizations=optimized))
        try:
            slam.current_disparity = np.full((480, 640), 10.)
            lid = slam.map.add_landmark([0, 0, 5], np.ones(128), 0, {})
            slam.accepted_tracks = [(lid, np.array([320.25, 240.25]))]
            key = slam._keyframe(0, np.eye(4), np.array([[322., 240.]]), np.ones((1, 128)),
                                  np.full((1, 3), np.nan), np.array([np.nan]), {0: lid})
            observation = slam.map.landmarks[lid].observations[key]
            outputs.append((observation.pixel.copy(), observation.right_u))
        finally:
            slam.close()
    assert np.array_equal(outputs[0][0], outputs[1][0]) and outputs[0][1] == outputs[1][1]


def test_indexed_recovery_retains_exhaustive_fallback_and_reuses_scores(monkeypatch):
    from types import SimpleNamespace
    slam = SharedSlam(K, config=MappingConfig(loop_mode='off'),
                      performance=PerformanceConfig(retrieval='indexed'))
    try:
        for i in range(3):
            slam.map.keyframes[i] = SimpleNamespace(id=i, frame=i, descriptors=np.full((2, 128), i))
        monkeypatch.setattr(slam.retrieval_index, 'upsert', lambda *_: None)
        monkeypatch.setattr(slam.retrieval_index, 'query', lambda *_, **__: [0])
        matched = []
        def matching(first, _second):
            matched.append(int(first[0, 0]))
            return np.empty((0, 2), int)
        monkeypatch.setattr(slam, '_match', matching)
        seen = []
        def recover(*args):
            seen.append(args[-1])
            return (None, {}) if len(seen)==1 else ('verified', {'geometric': True})
        monkeypatch.setattr(slam, '_recover_ranked', recover)
        result, stats = slam._relocalize(np.empty((0, 2)), np.ones((2, 128)), (640, 480))
        assert result=='verified' and stats['geometric']
        assert matched==[0, 1, 2] and len(seen[1])==3
    finally:
        slam.close()
