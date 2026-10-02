"""Performance wiring retains map corrections, observations and geometry-only recovery."""
import numpy as np
import shared_slam as shared_slam_module
from shared_slam import SharedSlam, MappingConfig, StereoCamera
from performance import PerformanceConfig
from slam_state import Observation

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
        assert third in [landmark.id for landmark in slam._cached_landmarks()[2]]
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


def test_tracking_materializes_only_selected_descriptors_with_exact_equivalence(monkeypatch):
    rng = np.random.default_rng(2815)
    landmark_count = 2105
    all_descriptors = rng.normal(size=(landmark_count, 128)).astype(np.float32)
    positions = np.zeros((landmark_count, 3), dtype=float)
    positions[:, 0] = np.linspace(-0.2, 0.2, landmark_count)
    positions[:, 1] = np.linspace(-0.15, 0.15, landmark_count)
    positions[:, 2] = 5.0
    positions[:8, 0] = 100.0  # Filter these before applying max_landmarks.
    expected_descriptors = all_descriptors[8:2008]
    query_pixels = np.column_stack((np.linspace(100, 200, 24), np.linspace(120, 220, 24)))
    query_descriptors = expected_descriptors[:24].copy()
    query_descriptors += rng.normal(scale=0.001, size=query_descriptors.shape).astype(np.float32)

    proposed_pose = np.eye(4)
    proposed_pose[0, 3] = 0.125

    def fake_estimate_pose(points, pixels, matrix, size, min_inliers,
                           initial_pose=None, diagnostics=None):
        if diagnostics is not None:
            diagnostics["test_solver"] = "deterministic"
        return proposed_pose.copy(), np.arange(len(points)), 0.125

    monkeypatch.setattr(shared_slam_module, "estimate_pose", fake_estimate_pose)

    outputs = []
    for optimized in (True, False):
        slam = SharedSlam(
            K,
            config=MappingConfig(loop_mode="off", max_landmarks=2000),
            performance=PerformanceConfig(cpu_optimizations=optimized),
        )
        try:
            slam.map.record(np.eye(4), "tracking")
            for ident, (position, descriptor) in enumerate(zip(positions, all_descriptors)):
                slam.map.add_landmark(
                    position,
                    descriptor,
                    0,
                    {0: Observation(np.array([320.0, 240.0]))},
                )

            class CapturingMatcher:
                def __init__(self, matcher):
                    self.matcher = matcher
                    self.descriptor_rows = []
                    self.matches = []

                def __call__(self, first, second, ratio=0.7):
                    self.descriptor_rows.append(first.copy())
                    result = self.matcher(first, second, ratio)
                    self.matches.append(result.copy())
                    return result

            capture = CapturingMatcher(slam.matcher)
            slam.matcher = capture
            result, info = slam._track(query_pixels, query_descriptors, (640, 480))
            cache = slam._landmark_cache
            outputs.append((result, info, capture, cache))
        finally:
            slam.close()

    optimized, reference = outputs
    assert len(optimized[2].descriptor_rows) == len(reference[2].descriptor_rows) == 1
    selected = optimized[2].descriptor_rows[0]
    assert selected.shape == (2000, 128)
    assert selected.dtype == np.float32
    np.testing.assert_array_equal(selected, expected_descriptors)
    np.testing.assert_array_equal(selected, reference[2].descriptor_rows[0])
    expected_pairs = np.column_stack((np.arange(24), np.arange(24)))
    np.testing.assert_array_equal(optimized[2].matches[0], expected_pairs)
    np.testing.assert_array_equal(optimized[2].matches[0], reference[2].matches[0])
    np.testing.assert_array_equal(optimized[0][0], reference[0][0])
    assert optimized[0][1] == reference[0][1] == {i: i + 8 for i in range(24)}
    assert optimized[1] == reference[1]

    # The persistent cache holds only the map-order objects and positions;
    # descriptor materialization is bounded by max_landmarks at match time.
    assert len(optimized[3]) == 4
    assert len(optimized[3][2]) == landmark_count
    assert optimized[3][3].shape == (landmark_count, 3)
    assert optimized[3][3].nbytes == landmark_count * 3 * np.dtype(float).itemsize
    assert not any(
        isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 128
        for value in optimized[3]
    )
