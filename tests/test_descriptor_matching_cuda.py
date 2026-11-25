"""Optional CUDA descriptor matching stays identical to the CPU arbiter."""
import builtins
import platform

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


def test_cuda_uses_one_matrix_for_quantized_sift(cuda_matcher):
    rng = np.random.default_rng(952)
    first = rng.integers(0, 256, size=(290, 128)).astype(np.float32)
    second = np.vstack((first[0], first[0], first[1:] + (first[1:] < 255)))
    expected = DescriptorMatcher("cpu")(first, second)
    actual = cuda_matcher(first, second)
    # The duplicate zero-distance nearest neighbors for row zero must reject.
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (289, 2)
    assert (actual[:, 0] > 0).all()
    assert cuda_matcher.metadata()["cuda_integer_calls"] == 1


def test_cuda_quantized_read_only_views_on_nondefault_stream(cuda_matcher):
    rng = np.random.default_rng(8146)
    descriptors = rng.integers(0, 256, size=(300, 256)).astype(np.float32)
    first = descriptors[:, ::2]
    query = np.arange(len(first))
    first[:, 0] = query // 256
    first[:, 1] = query % 256
    second = first[::-1].copy()
    first.setflags(write=False)
    second.setflags(write=False)

    torch = cuda_matcher.torch
    with torch.cuda.stream(torch.cuda.Stream()):
        actual = cuda_matcher(first, second)

    # Every descriptor occurs once in each input. Reversing the order gives
    # one exact zero-distance neighbor at train index len(first) - 1 - query.
    np.testing.assert_array_equal(actual, np.column_stack((query, query[::-1])))
    assert cuda_matcher.metadata()["cuda_integer_calls"] == 1


def test_fused_rank_matches_tensor_topk_on_partial_cuda_tiles(cuda_matcher, monkeypatch):
    if cuda_matcher._cuda_rank_kernel is None:
        pytest.skip("The fused CUDA rank kernel requires Windows NVRTC")
    rng = np.random.default_rng(901)
    first = rng.integers(0, 256, (129, 128), dtype=np.uint8).astype(np.float32)
    second = np.concatenate((first[::-1], first[[0]], first[:35] + (first[:35] < 255)))

    actual = cuda_matcher(first, second)
    assert cuda_matcher.report()["cuda_fused_rank_calls"] == 1

    monkeypatch.setattr(cuda_matcher, "_cuda_rank_kernel", None)
    np.testing.assert_array_equal(actual, cuda_matcher(first, second))
    assert cuda_matcher.report()["cuda_fused_rank_calls"] == 1
    # The reversed descriptors give each row one exact match except row zero,
    # whose two identical rows fail the strict ratio test.
    np.testing.assert_array_equal(actual, np.column_stack((np.arange(1, 129), np.arange(127, -1, -1))))


def test_small_loop_match_uses_native_cpu_without_changing_pairs(cuda_matcher):
    from native_accel import match_sift_mutual_vnni

    rng = np.random.default_rng(8120)
    first = rng.integers(0, 256, size=(128, 128)).astype(np.float32)
    second = first[::-1].copy()
    if match_sift_mutual_vnni(first, second, 0.7) is None:
        pytest.skip("The native VNNI matcher is unavailable")

    np.testing.assert_array_equal(cuda_matcher.match_loop(first, second), cuda_matcher(first, second))

    metadata = cuda_matcher.report()
    assert metadata["cpu_native_loop_calls"] == 1
    assert metadata["cpu_calls"] == metadata["cuda_calls"] == 1


@pytest.mark.parametrize("row_count,native_attempts", [(128, 1), (501, 0)])
def test_loop_match_uses_cuda_outside_the_native_path(cuda_matcher, monkeypatch, row_count, native_attempts):
    import native_accel

    rng = np.random.default_rng(8121)
    first = rng.integers(0, 256, size=(row_count, 128)).astype(np.float32)
    second = first[::-1].copy()
    attempts = []

    def unavailable(*args):
        attempts.append(args)
        return None

    monkeypatch.setattr(native_accel, "match_sift_mutual_vnni", unavailable)

    np.testing.assert_array_equal(cuda_matcher.match_loop(first, second), cuda_matcher(first, second))
    assert len(attempts) == native_attempts
    assert cuda_matcher.report()["cpu_native_loop_calls"] == 0
    assert cuda_matcher.report()["cuda_calls"] == 2


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
    assert metadata["python_version"] == platform.python_version()
    assert metadata["pytorch_version"] is None
    assert metadata["cuda_runtime_version"] is None
    assert metadata["cuda_device_index"] is None
    assert metadata["cuda_device_name"] is None
    assert metadata["cuda_device_capability"] is None


def test_cpu_metadata_does_not_import_optional_torch(monkeypatch):
    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("the CPU matcher must not import optional torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    metadata = DescriptorMatcher("cpu").metadata()

    assert metadata["python_version"] == platform.python_version()
    assert metadata["pytorch_version"] is None
    assert metadata["cuda_runtime_version"] is None
    assert metadata["cuda_device_index"] is None
    assert metadata["cuda_device_name"] is None
    assert metadata["cuda_device_capability"] is None


def test_cuda_metadata_records_runtime_and_device_provenance(cuda_matcher):
    metadata = cuda_matcher.metadata()
    torch = cuda_matcher.torch

    assert metadata["python_version"] == platform.python_version()
    assert metadata["pytorch_version"] == str(torch.__version__)
    assert metadata["cuda_runtime_version"] == str(torch.version.cuda)
    assert metadata["cuda_device_index"] == int(torch.cuda.current_device())
    assert metadata["cuda_device_name"] == torch.cuda.get_device_name()
    assert metadata["cuda_device_capability"] == list(torch.cuda.get_device_capability())
