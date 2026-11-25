import cv2 as cv
import numpy as np
import pytest

from cuda_sift_pyramid import create_cuda_sift_pyramid
from frontend_features import SiftDescriptorReadback, extract_grid_bucketed_sift
from native_accel import extract_sift_grid_avx512, sift_pyramid_available


def _texture():
    pixels = np.random.default_rng(481).integers(0, 256, (49, 78), dtype=np.uint8)
    return cv.resize(pixels, (317, 183), interpolation=cv.INTER_CUBIC)


@pytest.mark.parametrize("first_octave", [0, -1])
def test_cuda_sift_matches_the_native_extractor(first_octave):
    cuda, reason = create_cuda_sift_pyramid()
    if cuda is None:
        pytest.skip(reason)

    gray = _texture()[:, ::-1]
    options = dict(target_features=160, first_octave=first_octave)
    expected = extract_sift_grid_avx512(gray, **options)
    actual = cuda.extract(gray, **options)
    assert expected is not None and len(expected[0]) > 0
    assert actual is not None, cuda.report()
    # The existing AVX-512 SIFT implementation defines the reference rounding,
    # feature order and descriptor values for this exact backend.
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
    assert cuda.report()["cuda_frames"] == 1


@pytest.mark.parametrize("disable_native", [False, True])
def test_grid_sift_falls_back_to_the_same_cpu_features(monkeypatch, disable_native):
    if disable_native:
        monkeypatch.setenv("VISUAL_SLAM_DISABLE_FAST_SIFT", "1")
    gray = _texture()
    seen = []

    def unavailable(image, **options):
        seen.append((image, options))
        return None

    options = dict(target_features=160, build_cv_keypoints=False)
    expected = extract_grid_bucketed_sift(gray, **options)
    actual = extract_grid_bucketed_sift(gray, native_sift=unavailable, **options)
    assert len(seen) == 1 and seen[0][0] is gray
    # The same extractor without a CUDA request supplies the CPU reference.
    for field in ("pixels", "descriptors", "octaves", "angles", "responses"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_scaled_cuda_grid_sift_matches_the_scaled_cpu_extractor():
    cuda, reason = create_cuda_sift_pyramid()
    if cuda is None:
        pytest.skip(reason)
    gray = _texture()
    options = dict(target_features=64, detection_scale=0.78, build_cv_keypoints=False)
    expected = extract_grid_bucketed_sift(gray, **options)
    actual = extract_grid_bucketed_sift(gray, native_sift=cuda.extract, **options)
    assert len(expected.pixels) == options["target_features"]
    assert actual.keypoints == []
    for field in ("pixels", "descriptors", "octaves", "angles", "responses"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert cuda.report()["cuda_frames"] == 1


def test_native_sift_rejects_a_truncated_cuda_pyramid():
    if not sift_pyramid_available():
        pytest.skip("native SIFT pyramid consumer is unavailable")
    # No valid octave layout can fit in one int16 sample.
    assert extract_sift_grid_avx512(_texture(), pyramid=np.zeros(1, dtype=np.int16)) is None


@pytest.mark.parametrize("first_octave", [0, -1])
def test_compact_cuda_sift_matches_native_features(first_octave):
    cuda, reason = create_cuda_sift_pyramid(compact=True)
    if cuda is None:
        pytest.skip(reason)

    gray = _texture()[:, ::-1]
    options = dict(target_features=160, first_octave=first_octave)
    expected = extract_sift_grid_avx512(gray, **options)
    actual = cuda.extract_deferred(gray, **options)
    assert actual is not None and len(actual[0]) == 160, cuda.report()
    assert isinstance(actual[1], SiftDescriptorReadback)
    # Reuse the same workspace before consuming the first copy.
    next_result = cuda.extract(_texture(), **options)
    assert next_result is not None, cuda.report()
    pixels, pending, octaves, angles, responses = actual
    descriptors = pending.finish()
    assert isinstance(descriptors, np.ndarray) and descriptors.dtype == np.float32
    assert descriptors.shape == (160, 128)
    # The native SIFT implementation supplies the reference keypoint order and histogram.
    # CUDA and AVX-512 can round the final orientation and quantized bins differently.
    np.testing.assert_array_equal(pixels, expected[0])
    np.testing.assert_allclose(descriptors, expected[1], rtol=0, atol=1)
    np.testing.assert_array_equal(octaves, expected[2])
    np.testing.assert_allclose(angles, expected[3], rtol=0, atol=2e-4)
    np.testing.assert_array_equal(responses, expected[4])
    assert cuda.report()["compact_frames"] == 2


def test_compact_cuda_sift_uses_exact_legacy_path_on_seed_overflow(monkeypatch):
    from cuda_sift_compact import _CompactSiftWorkspace

    cuda, reason = create_cuda_sift_pyramid(compact=True)
    if cuda is None:
        pytest.skip(reason)
    monkeypatch.setattr(_CompactSiftWorkspace, "seed_capacity", 1)
    gray = _texture()
    options = dict(target_features=160, first_octave=0)
    expected = extract_sift_grid_avx512(gray, **options)
    actual = cuda.extract(gray, **options)
    assert actual is not None, cuda.report()
    for result, reference in zip(actual, expected):
        assert result.tobytes() == reference.tobytes()
    assert cuda.report()["cuda_frames"] == 1
    assert cuda.report()["compact_frames"] == 0
    assert cuda.report()["compact_seed_overflows"] == 1
