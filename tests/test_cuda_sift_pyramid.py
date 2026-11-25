import cv2 as cv
import numpy as np
import pytest

from cuda_sift_pyramid import create_cuda_sift_pyramid
from frontend_features import extract_grid_bucketed_sift
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


def test_native_sift_rejects_a_truncated_cuda_pyramid():
    if not sift_pyramid_available():
        pytest.skip("native SIFT pyramid consumer is unavailable")
    # No valid octave layout can fit in one int16 sample.
    assert extract_sift_grid_avx512(_texture(), pyramid=np.zeros(1, dtype=np.int16)) is None
