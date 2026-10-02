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
    monkeypatch.setattr(slam, "_rank_keyframes", lambda desc, exhaustive=False: [(30, i) for i in range(30 if exhaustive else 20)])
    def recover(pixels, desc, size, points, ranked):
        calls.append(len(ranked))
        return (np.eye(4), {}) if len(ranked) == 30 else (None, {})
    monkeypatch.setattr(slam, "_recover_ranked", recover)
    assert slam._relocalize([], [], (640, 480))[0] is not None
    assert calls == [20, 30]
    slam.loop_worker.executor.shutdown()
