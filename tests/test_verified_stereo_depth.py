from dataclasses import replace
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from feature_cache import extraction_signature
from shared_slam import MappingConfig, SharedSlam, StereoCamera
import stereo_depth
from stereo_depth import StereoSearchConfig, verify_stereo_depth_candidates


def images(disparity=10.25):
    rng = np.random.default_rng(42)
    left = cv.GaussianBlur(rng.integers(0, 256, (96, 240), dtype=np.uint8), (3, 3), .6)
    right = cv.warpAffine(left, np.float32([[1, 0, -disparity], [0, 1, 0]]), (240, 96))
    return left, right


def camera(policy='verified_fallback', offset=2.):
    k = np.array([[100., 0, 120], [0, 100., 48], [0, 0, 1.]])
    q = np.array([[1, 0, 0, -120], [0, 1, 0, -48], [0, 0, 0, 100], [0, 0, 5., -5*offset]])
    matcher = cv.StereoSGBM_create(numDisparities=96, blockSize=5)
    return SharedSlam(k, stereo=StereoCamera(matcher, q, .2),
                      config=MappingConfig(loop_mode='off', bundle_enabled=False, stereo_depth_policy=policy))


def test_full_range_fractional_correspondence_without_a_disparity_proposal():
    left, right = images()
    pixels = np.array([[170.25, 50.25], [151.2, 38.3], [195.5, 65.5]])
    measured, counters = verify_stereo_depth_candidates(left, right, pixels)
    assert np.isfinite(measured).all(), counters
    assert np.allclose(pixels[:, 0]-measured, 10.25, atol=.3), measured
    assert counters['verified'] == len(pixels)


def test_verified_stereo_policy_cannot_be_used_for_monocular_estimation():
    with pytest.raises(ValueError, match='two calibrated cameras'):
        SharedSlam(np.eye(3), config=MappingConfig(stereo_depth_policy='verified_fallback'))


@pytest.mark.parametrize('case', ['constant', 'repeated', 'occluded'])
def test_texture_ambiguity_and_occlusion_fail_closed(case):
    left, right = images(10.)
    if case == 'constant':
        left[:] = right[:] = 80
    elif case == 'repeated':
        pattern = np.random.default_rng(8).integers(0, 256, (96, 12), dtype=np.uint8)
        left = np.tile(pattern, (1, 20))
        right = np.roll(left, -12, axis=1)
    else:
        right[:] = 80
    measured, _ = verify_stereo_depth_candidates(left, right, [[170., 50.]])
    assert np.isnan(measured).all()


def test_invalid_patch_and_calibrated_depth_domains_are_rejected():
    left, right = images(10.)
    result, _ = verify_stereo_depth_candidates(left, right, [[2., 50.], [170., 2.], [np.nan, 30.]])
    assert np.isnan(result).all()
    result, _ = verify_stereo_depth_candidates(left, right, [[170., 50.]], disparity_bounds=(20., 90.))
    assert np.isnan(result).all()


def test_reverse_search_must_return_to_the_original_observation(monkeypatch):
    directions = []
    def inconsistent(source, target, pixel, direction, config, bounds):
        directions.append(direction)
        return (160. if direction == -1 else 172.), None
    monkeypatch.setattr(stereo_depth, '_search', inconsistent)
    left, right = images(10.)
    result, counters = verify_stereo_depth_candidates(left, right, [[170., 50.]])
    assert directions == [-1, 1]
    assert np.isnan(result).all() and counters['round_trip'] == 1


def test_nearly_tied_float32_peaks_do_not_become_unique_matches(monkeypatch):
    left, right = images(10.)
    scores = np.full((1, 97), .1, np.float32)
    scores[0, 86], scores[0, 70] = 1., 1.-np.finfo(np.float32).eps
    monkeypatch.setattr(cv, 'matchTemplate', lambda *args: scores)
    result, counters = verify_stereo_depth_candidates(left, right, [[170., 50.]])
    assert np.isnan(result).all() and counters['near_tie'] == 1


def test_verified_depth_uses_measured_disparity_offset_and_preserves_supported_values():
    left, right = images()
    pixels = np.array([[170.25, 50.25], [120.5, 40.5], [2.3, 50.]])
    current, baseline = camera(), camera('supported')
    try:
        for slam in (current, baseline):
            slam.current_disparity = np.full(left.shape, 9., np.float32)
            slam.current_disparity[51, 171] = -1.
            slam.current_disparity[50, 3] = -1.
            slam._prepare_frame_images(left, right)
        old_points, old_right = baseline._measure_stereo_pixels(pixels)
        points, observed_right = current._measure_stereo_pixels(pixels)
        assert np.isnan(old_points[0]).all() and np.isfinite(points[0]).all()
        disparity = pixels[0, 0]-observed_right[0]
        assert abs(disparity-10.25) < .3 and abs(disparity-9.) > .9
        depth = 20./(disparity-2.)
        ray = np.r_[pixels[0], 1.] @ current.inverse_K.T
        np.testing.assert_allclose(points[0], ray*depth)
        np.testing.assert_array_equal(points[1], old_points[1])
        assert observed_right[1] == old_right[1]
        assert np.isnan(points[2]).all()
        current.current_right_gray = None
        missing, _ = current._measure_stereo_pixels(pixels)
        assert np.isnan(missing[0]).all()
    finally:
        current.close(); baseline.close()


def test_policy_settings_and_verifier_dependency_invalidate_extraction_cache(monkeypatch):
    current, baseline = camera(), camera('supported')
    try:
        signature = extraction_signature(current, cv)
        assert signature != extraction_signature(baseline, cv)
        current.stereo_search_config = replace(current.stereo_search_config, minimum_correlation=.9)
        assert signature != extraction_signature(current, cv)
        original = Path.read_bytes
        def changed(path):
            data = original(path)
            return data+b'\n# changed verification dependency' if path == Path(stereo_depth.__file__) else data
        signature = extraction_signature(current, cv)
        monkeypatch.setattr(Path, 'read_bytes', changed)
        assert signature != extraction_signature(current, cv)
    finally:
        current.close(); baseline.close()


def test_first_frame_and_cache_hit_prepare_the_current_pair_before_depth_admission():
    left, right = images(10.)
    pixels = np.array([[x+.25, y+.25] for y in [15, 30, 45, 60, 75] for x in [100, 120, 140, 160, 180, 200, 220]], np.float32)
    slam = camera()
    seen_pairs = []
    # This stand-in for cached extraction computes only depth; process must
    # prepare the current images even though the normal extractor is bypassed.
    def cached(image, right_image):
        seen_pairs.append((slam.current_left_gray.copy(), slam.current_right_gray.copy()))
        slam.current_disparity = np.full(image.shape, 9., np.float32)
        for x, y in pixels:
            slam.current_disparity[int(y)+1, int(x)+1] = -1.
        points, ru = slam._measure_stereo_pixels(pixels)
        return pixels, np.ones((len(pixels), 128), np.float32), points, ru
    slam._extract = cached
    try:
        _, info = slam.process(0, left, right)
        assert info['state'] == 'tracking' and info['valid_stereo_depth'] >= 30, info
        assert len(slam.map.landmarks) == info['valid_stereo_depth']
        next_left, next_right = 255-left, 255-right
        slam.process(1, next_left, next_right)
        np.testing.assert_array_equal(seen_pairs[1][0], next_left)
        np.testing.assert_array_equal(seen_pairs[1][1], next_right)
    finally:
        slam.close(finish=False)
