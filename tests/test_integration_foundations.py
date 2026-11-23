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


def test_gemm_l2_matcher_exact_parity_with_bfmatcher():
    from mapping_geometry import _match_descriptors_bf, _match_descriptors_gemm_l2

    rng = np.random.default_rng(123)
    # Build >128x128 matrices with realistic SIFT-like scale, exact ties, near-ratio boundaries, and zero-distance duplicates
    base = rng.uniform(0.0, 200.0, size=(200, 128)).astype(np.float32)
    second = base[:160].copy()
    # Perturb matches slightly, inject exact duplicate ties, and inject exact 0-distance copies
    second[:80] += rng.normal(0.0, 3.0, size=(80, 128)).astype(np.float32)
    second[80:90] = base[80:90]  # zero-distance cancellation sensitive
    second = np.vstack([second, second[0:5], rng.uniform(0.0, 200.0, size=(20, 128)).astype(np.float32)])

    # Test both rectangular orientations (len(a) > len(b) and len(a) < len(b))
    for a, b in ((base, second), (second, base)):
        expected = _match_descriptors_bf(a, b, ratio=0.7)
        actual_gemm = _match_descriptors_gemm_l2(a, b, ratio=0.7)
        actual_dispatch = match_descriptors(a, b, ratio=0.7)
        np.testing.assert_array_equal(actual_gemm, expected)
        np.testing.assert_array_equal(actual_dispatch, expected)


def test_keyframe_index_blas_matches_scipy_reference_and_caches_readonly_queries():
    from scipy.cluster.vq import kmeans2, vq

    rng = np.random.default_rng(31)
    index = KeyframeIndex()
    descs = {}
    for ident in range(12):
        arr = np.abs(rng.normal(size=(60, 128))).astype(np.float32)
        arr.setflags(write=False)
        descs[ident] = arr
        index.upsert(ident, ident * 10, arr)

    training = np.vstack([descs[i] for i in range(5)]).astype(np.float32)
    ref_vocab, _ = kmeans2(training, 128, iter=20, minit="++", rng=np.random.default_rng(0))
    np.testing.assert_allclose(index.vocabulary, ref_vocab, atol=1e-5, rtol=1e-5)

    # Verify encoding matches SciPy vq + unprojected difference
    words_ref, _ = vq(descs[11], index.vocabulary)
    counts_ref = np.bincount(words_ref, minlength=128).astype(np.float64)
    hist_ref = counts_ref / counts_ref.sum()
    np.testing.assert_allclose(index.histograms[11], hist_ref, atol=1e-12)

    # Verify query result matches postings-loop reference and read-only cache hits cleanly
    eligible = set(range(10))
    res_cached = index.query(descs[11], eligible=eligible, limit=8)
    writable_copy = descs[11].copy()
    res_uncached = index.query(writable_copy, eligible=eligible, limit=8)
    assert res_cached == res_uncached
    assert len(res_cached) == 8
    assert set(res_cached) <= eligible
