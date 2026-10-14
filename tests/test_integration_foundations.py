import builtins

import numpy as np
import pytest

from descriptor_matching import DescriptorMatcher
from keyframe_index import KeyframeIndex
from mapping_geometry import match_descriptors
from performance import PerformanceConfig
from stage_profile import StageProfile


def test_keyframe_index_limits_results_to_eligible_frames_and_removes_postings():
    rng = np.random.default_rng(7)
    index = KeyframeIndex()
    descriptors = {}
    for ident in range(5):
        descriptors[ident] = rng.normal(size=(40, 128)).astype(np.float32)
        index.upsert(ident, ident, descriptors[ident])

    assert index.vocabulary is not None
    candidates = index.query(descriptors[2], eligible={2, 4}, limit=8)
    assert candidates
    assert set(candidates) <= {2, 4}

    index.remove(2)
    assert 2 not in index.frames
    assert 2 not in index.histograms
    assert 2 not in index.residuals
    assert all(2 not in posting for posting in index.postings)
    assert 2 not in index.query(descriptors[2], eligible={2, 4}, limit=8)

    with pytest.raises(ValueError, match="128-dimensional float SIFT"):
        index.upsert(9, 9, np.zeros((3, 128), dtype=np.uint8))


def test_descriptor_matcher_cpu_preserves_tie_and_strict_ratio_behavior():
    first = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    tied_second = np.array([[1, 0], [-1, 0], [10, 0], [20, 0]], dtype=np.float32)
    matcher = DescriptorMatcher("cpu")
    tied = matcher(first, tied_second, ratio=0.7)
    np.testing.assert_array_equal(tied, match_descriptors(first, tied_second, ratio=0.7))
    assert [0, 0] not in tied.tolist()  # The first query has equal nearest neighbors.

    ratio_first = np.array([[0, 0], [8, 0]], dtype=np.float32)
    ratio_second = np.array([[1, 0], [2, 0], [8, 0]], dtype=np.float32)
    ratio_matches = matcher(ratio_first, ratio_second, ratio=0.5)
    # Distance 1 is exactly ratio * distance 2, so the strict ratio test rejects it.
    assert ratio_matches.tolist() == [[1, 2]]
    assert matcher.report()["cpu_calls"] == 2


def test_descriptor_matcher_auto_falls_back_when_torch_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("test environment has no optional torch dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    matcher = DescriptorMatcher("auto")
    first = np.array([[0, 0], [4, 0]], dtype=np.float32)
    second = np.array([[0, 0], [4, 0]], dtype=np.float32)
    np.testing.assert_array_equal(matcher(first, second), match_descriptors(first, second))
    report = matcher.report()
    assert report["cuda_available"] is False
    assert "optional torch dependency" in report["fallback_reason"]
    assert report["cpu_calls"] == 1

    with pytest.raises(RuntimeError, match="CUDA matching requested but unavailable"):
        DescriptorMatcher("cuda")


def test_stage_profile_keeps_legacy_report_and_adds_bounded_detail():
    profile = StageProfile(detailed=True, max_samples=2)
    assert profile.call("extract", lambda value: value + 1, 4) == 5
    profile.wrap("extract", lambda value: value * 2)(3)
    profile.call("extract", lambda: None)
    profile.count("matches", 7)

    legacy = profile.report()
    assert set(legacy["extract"]) == {"calls", "inclusive_seconds"}
    assert legacy["extract"]["calls"] == 3

    detail = profile.detailed_report()
    assert detail["stages"]["extract"]["calls"] == 3
    assert detail["stages"]["extract"]["sample_count"] == 2
    assert detail["counters"] == {"matches": 7}


def test_performance_config_preserves_current_retrieval_as_default():
    config = PerformanceConfig()
    assert config.retrieval == "current"
    assert config.matching_backend == "cpu"
    assert config.cpu_optimizations is True
    assert config.profile is False
    assert PerformanceConfig(retrieval="indexed").retrieval == "indexed"

    with pytest.raises(ValueError, match="current, indexed or exhaustive"):
        PerformanceConfig(retrieval="unknown")
