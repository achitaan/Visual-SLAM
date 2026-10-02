import numpy as np
import pytest
from performance import PerformanceConfig, StageProfiler, latency_stats


def test_profile_disabled_and_worker_timings():
    from concurrent.futures import ThreadPoolExecutor
    disabled = StageProfiler()
    assert disabled.call("test", lambda: 2) == 2
    assert disabled.report()["stages"] == {}
    enabled = StageProfiler(True)
    with ThreadPoolExecutor(2) as executor:
        list(executor.map(lambda _: enabled.call("worker", lambda: 2), range(10)))
    assert enabled.report()["stages"]["worker"]["calls"] == 10
    assert latency_stats([])["p95_ms"] is None


def test_performance_configuration_does_not_change_accuracy_settings():
    from shared_slam import MappingConfig
    assert MappingConfig().features == 1500
    assert MappingConfig().bundle_window == 5
    with pytest.raises(ValueError):
        PerformanceConfig(retrieval="unknown")
    with pytest.raises(ValueError):
        PerformanceConfig(matching_backend="unknown")


def test_index_startup_updates_and_temporal_candidates():
    from keyframe_index import KeyframeIndex
    index = KeyframeIndex()
    rng = np.random.default_rng(4)
    descriptors = [rng.normal(i * 3, 0.1, (100, 128)).astype(np.float32) for i in range(6)]
    for i in range(4):
        index.upsert(i, i * 50, descriptors[i])
    assert index.query(descriptors[0], [0, 1]) is None
    index.upsert(4, 200, descriptors[4])
    assert index.query(descriptors[0], range(5))[0] == 0
    assert 0 not in index.query(descriptors[0], [1, 2, 3])
    index.upsert(5, 250, descriptors[5])
    assert index.query(descriptors[5], range(6))[0] == 5
    index.remove(5)
    assert 5 not in index.query(descriptors[5], range(6))
    index.upsert(1, 50, descriptors[0])
    assert index.query(descriptors[0], [1]) == [1]


def test_recovery_uses_exhaustive_fallback_and_does_not_accept_appearance(monkeypatch):
    from shared_slam import SharedSlam
    slam = SharedSlam(np.diag([250., 250., 1.]))
    slam.map.keyframes = {i: None for i in range(30)}
    calls = []
    monkeypatch.setattr(slam, "_rank_keyframes", lambda desc, exhaustive=False, cached=None: [(30, i) for i in range(30 if exhaustive else 20)])
    def recover(pixels, desc, size, points, ranked):
        calls.append(len(ranked))
        return (np.eye(4), {}) if len(ranked) == 30 else (None, {})
    monkeypatch.setattr(slam, "_recover_ranked", recover)
    assert slam._relocalize([], [], (640, 480))[0] is not None
    assert calls == [20, 30]
    slam.loop_worker.executor.shutdown()


def test_exhaustive_fallback_reuses_exact_shortlist_scores(monkeypatch):
    from types import SimpleNamespace
    from shared_slam import SharedSlam
    slam = SharedSlam(np.diag([250., 250., 1.]), performance=PerformanceConfig(retrieval="exhaustive"))
    slam.map.keyframes = {i: SimpleNamespace(id=i, frame=i*10, descriptors=np.full((2,128), i, np.float32)) for i in range(4)}
    calls = []
    def match(first, second):
        ident = int(first[0, 0]); calls.append(ident)
        return np.zeros((ident, 2), int)
    monkeypatch.setattr(slam, "_match", match)
    ranked = slam._rank_keyframes([], exhaustive=True, cached=[(42, 2)])
    assert calls == [0, 1, 3]
    assert ranked == [(42, 2), (3, 3), (1, 1), (0, 0)]
    slam.loop_worker.executor.shutdown()


def test_keyframe_appearance_view_preserves_flow_descriptors_without_copying():
    from shared_slam import SharedSlam
    slam = SharedSlam(np.diag([250., 250., 1.]))
    lid = slam.map.add_landmark([0, 0, 5], np.full(128, 7, np.float32), 0, {})
    slam.accepted_tracks = [(lid, np.array([20., 20.]))]
    slam.current_gray = np.zeros((100, 100), np.uint8)
    desc = np.ones((2, 128), np.float32)
    slam._keyframe(0, np.eye(4), np.array([[1., 1.], [2., 2.]], np.float32), desc,
                   np.full((2, 3), np.nan), np.full(2, np.nan), {})
    frame = slam.map.keyframes[0]
    assert len(frame.descriptors) == 3
    assert np.array_equal(frame.retrieval_descriptors, desc)
    assert np.shares_memory(frame.retrieval_descriptors, frame.descriptors)
    assert frame.landmark_ids[2] == lid
    assert np.array_equal(frame.descriptors[2], slam.map.landmarks[lid].descriptor)
    slam.close()


def test_loop_proposals_use_sift_appearance_and_exact_matching_uses_depth(monkeypatch):
    from live_loops import LiveLoopWorker
    from slam_state import MappingKeyframe
    frames = {}
    for i in range(2):
        desc = np.full((3, 128), i, np.float32)
        frames[i] = MappingKeyframe(i, i * 200, np.eye(4), np.zeros((3, 2)), desc, np.arange(3),
                                    depth_points=np.array([[0, 0, 5], [np.nan] * 3, [np.nan] * 3]))
        frames[i].retrieval_descriptors = desc[:2]
    calls = []
    class Index:
        def upsert(self, ident, frame, desc):
            assert desc is frames[ident].retrieval_descriptors
        def query(self, desc, eligible):
            assert desc is frames[1].retrieval_descriptors and eligible == [0]
            return [0]
    def match(first, second):
        calls.append((len(first), len(second)))
        return np.empty((0, 2), int)
    worker = LiveLoopWorker(np.diag([250., 250., 1.]), True, matcher=match)
    worker.retrieval_index = Index()
    monkeypatch.setattr("live_loops.verify_loop", lambda *args: None)
    result = worker._solve(frames, 0, {})
    assert calls == [(1, 1)] and result["correction"] is None
    worker.executor.shutdown()


def test_landmark_cache_refreshes_after_atomic_pose_corrections():
    from shared_slam import SharedSlam
    from slam_state import MappingKeyframe, Observation
    slam = SharedSlam(np.diag([250., 250., 1.]))
    slam.map.keyframes[0] = MappingKeyframe(0, 0, np.eye(4), np.zeros((1, 2)), np.ones((1, 128)), np.array([0]))
    shifted = np.eye(4)
    shifted[0, 3] = 1
    slam.map.keyframes[1] = MappingKeyframe(1, 10, shifted, np.zeros((1, 2)), np.ones((1, 128)), np.array([0]))
    slam.map.add_landmark([2, 0, 5], np.ones(128), 1, {1: Observation(np.zeros(2))})
    before = slam._cached_landmarks()[3].copy()
    corrected = shifted.copy()
    corrected[0, 3] = 2
    assert slam.map.apply_corrections(0, {0: np.eye(4), 1: corrected})
    assert np.allclose(slam._cached_landmarks()[3], before + [1, 0, 0])
    slam.loop_worker.executor.shutdown()


def test_bundle_reference_and_optimized_residuals_agree():
    import copy
    from slam_state import MapState, MappingKeyframe, Observation
    from local_bundle import local_bundle_adjustment
    from mapping_geometry import project
    K = np.array([[250., 0, 320], [0, 250., 240], [0, 0, 1.]])
    state = MapState(metric=True)
    rng = np.random.default_rng(3)
    points = rng.uniform([-2, -1, 4], [2, 1, 9], (40, 3))
    for i in range(3):
        pose = np.eye(4)
        pose[0, 3] = i * .3
        state.keyframes[i] = MappingKeyframe(i, i * 10, pose, np.zeros((40, 2)), np.ones((40, 128)), np.arange(40))
    for p in points:
        observations = {}
        for i, frame in state.keyframes.items():
            pixel, depth = project(p[None], frame.pose, K)
            observations[i] = Observation(pixel[0], pixel[0, 0] - 250 * .54 / depth[0])
        state.add_landmark(p + rng.normal(0, .02, 3), np.ones(128), 0, observations)
    other = MapState(metric=True)
    other.keyframes, other.landmarks = copy.deepcopy(state.keyframes), copy.deepcopy(state.landmarks)
    reference = local_bundle_adjustment(state, K, .54, optimized=False)
    fast = local_bundle_adjustment(other, K, .54)
    assert reference["applied"] == fast["applied"]
    assert np.isclose(reference["final_cost"], fast["final_cost"], atol=1e-9)
    for ident in state.landmarks:
        assert np.allclose(state.landmarks[ident].position, other.landmarks[ident].position, atol=1e-7)


def test_gpu_unavailability_and_cpu_matching(monkeypatch):
    import sys
    from types import SimpleNamespace
    from descriptor_matching import DescriptorMatcher
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    fallback = DescriptorMatcher("auto")
    assert fallback.torch is None and "unavailable" in fallback.reason
    with pytest.raises(RuntimeError, match="unavailable"):
        DescriptorMatcher("cuda")
    desc = np.eye(8, dtype=np.float32)
    expected = np.column_stack([np.arange(8), np.arange(8)])
    assert np.array_equal(fallback(desc, desc), expected)


def test_cuda_matches_cpu_including_ties_ratio_boundaries_and_binary():
    from descriptor_matching import DescriptorMatcher
    from mapping_geometry import match_descriptors
    try:
        gpu = DescriptorMatcher("cuda")
    except RuntimeError:
        pytest.skip("CUDA matching environment unavailable")
    rng = np.random.default_rng(44)
    for size in [30, 1100]:
        desc = rng.uniform(0, 255, (size, 128)).astype(np.float32)
        query = desc.copy() + rng.normal(0, .5, desc.shape).astype(np.float32)
        query[0] = query[1]
        query[2] = .7 * desc[2] + .3 * desc[3]
        assert np.array_equal(gpu(desc, query), match_descriptors(desc, query))
    binary = rng.integers(0, 255, (40, 32), dtype=np.uint8)
    assert np.array_equal(gpu(binary, binary), match_descriptors(binary, binary))
    near = np.full((12, 128), 255, np.float32)
    near[:, 0] += np.arange(12, dtype=np.float32) * .001
    assert np.array_equal(gpu(near, near), match_descriptors(near, near))
