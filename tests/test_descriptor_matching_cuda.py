"""Optional CUDA descriptor matching stays identical to the CPU arbiter."""
import numpy as np
import pytest

from descriptor_matching import DescriptorMatcher
from mapping_geometry import match_descriptors


@pytest.fixture
def cuda_matcher():
    try:
        return DescriptorMatcher("cuda")
    except RuntimeError as error:
        pytest.skip(f"CUDA matching is unavailable: {error}")


def test_cuda_matches_cpu_on_unambiguous_float_descriptors(cuda_matcher):
    rng = np.random.default_rng(2701)
    first = rng.normal(size=(32, 128)).astype(np.float32)
    near = first + rng.normal(scale=0.02, size=first.shape).astype(np.float32)
    distractors = rng.normal(loc=8.0, size=(12, 128)).astype(np.float32)
    second = np.vstack((near, distractors))

    expected = DescriptorMatcher("cpu")(first, second)
    actual = cuda_matcher(first, second)

    np.testing.assert_array_equal(actual, expected)
    metadata = cuda_matcher.report()
    assert metadata["cuda_calls"] == 1
    assert metadata["cpu_calls"] == 0
    assert metadata["fallback_reason"] is None


def test_cuda_repeated_ties_match_cpu_and_report_cpu_rechecks(cuda_matcher):
    first = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    second = np.array([[0, 0], [0, 0], [10, 0], [20, 0]], dtype=np.float32)

    expected = match_descriptors(first, second, ratio=0.7)
    actual = cuda_matcher(first, second, ratio=0.7)

    np.testing.assert_array_equal(actual, expected)
    assert [0, 0] not in actual.tolist()  # Repeated nearest descriptors fail the strict ratio test.
    assert cuda_matcher.report()["ambiguous_rows_on_cpu"] > 0


def test_cuda_cancellation_and_near_ratio_rows_match_cpu(cuda_matcher):
    # The high-norm row has true distances 8 and 16, which pass a 0.7 ratio
    # test but are vulnerable to cancellation in the matrix-distance identity.
    # The low-norm row sits just inside the ratio boundary (0.699 / 1.0).
    high = np.full(32, 10_000.0, dtype=np.float32)
    first = np.vstack((high, np.zeros(32, dtype=np.float32)))
    high_nearest = high.copy()
    high_nearest[0] += 8.0
    high_second = high.copy()
    high_second[1] += 16.0
    low_nearest = np.zeros(32, dtype=np.float32)
    low_nearest[0] = 0.699
    low_second = np.zeros(32, dtype=np.float32)
    low_second[0] = -1.0
    second = np.vstack((high_nearest, high_second, low_nearest, low_second))

    cpu = DescriptorMatcher("cpu")
    expected = cpu(first, second, ratio=0.7)
    actual = cuda_matcher(first, second, ratio=0.7)

    np.testing.assert_array_equal(expected, np.array([[0, 0], [1, 2]]))
    np.testing.assert_array_equal(actual, expected)
    assert cuda_matcher.report()["ambiguous_rows_on_cpu"] > 0


def test_cpu_metadata_tracks_fallback_path_and_has_zero_cuda_usage():
    first = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    second = np.array([[0, 0], [0, 0], [10, 0], [20, 0]], dtype=np.float32)
    matcher = DescriptorMatcher("cpu")

    actual = matcher(first, second, ratio=0.7)
    expected = match_descriptors(first, second, ratio=0.7)

    np.testing.assert_array_equal(actual, expected)
    metadata = matcher.report()
    assert metadata["requested"] == "cpu"
    assert metadata["cuda_available"] is False
    assert metadata["fallback_reason"] is None
    assert metadata["cuda_calls"] == 0
    assert metadata["cpu_calls"] == 1
    assert metadata["ambiguous_rows_on_cpu"] == 0
    assert metadata["peak_cuda_allocated_mb"] == 0.0
    assert metadata["peak_cuda_reserved_mb"] == 0.0
