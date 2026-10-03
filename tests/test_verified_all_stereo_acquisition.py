import cv2 as cv
import numpy as np

from shared_slam import MappingConfig, SharedSlam, StereoCamera
from feature_cache import extraction_signature


def _images(disparity=10.25):
    rng = np.random.default_rng(42)
    left = cv.GaussianBlur(
        rng.integers(0, 256, (96, 240), dtype=np.uint8), (3, 3), .6)
    right = cv.warpAffine(
        left, np.float32([[1, 0, -disparity], [0, 1, 0]]), (240, 96))
    return left, right


def _slam(policy='verified_all', q_dtype=float):
    K = np.array([[100., 0, 120], [0, 100., 48], [0, 0, 1.]])
    offset = 2.
    Q = np.array([[1, 0, 0, -120], [0, 1, 0, -48],
                  [0, 0, 0, 100], [0, 0, 5., -5*offset]], dtype=q_dtype)
    stereo = StereoCamera(cv.StereoSGBM_create(numDisparities=96, blockSize=5), Q, .2)
    return SharedSlam(K, stereo=stereo, config=MappingConfig(
        loop_mode='off', bundle_enabled=False, stereo_depth_policy=policy))


def test_verified_all_ignores_dense_disparity_and_snapshots_actual_correspondences():
    left, right = _images()
    pixels = np.array([[170.25, 50.25], [151.2, 38.3], [195.5, 65.5]], np.float32)
    slam = _slam()
    try:
        outputs = []
        for dense in (np.full(left.shape, 10.875, np.float32),
                      np.full(left.shape, np.nan, np.float32)):
            slam._prepare_frame_images(left, right)
            slam.current_disparity = dense
            points, right_u = slam._measure_stereo_pixels(
                pixels, capture_supported=True)
            assert np.isfinite(points).all()
            disparity = pixels[:, 0] - right_u
            np.testing.assert_allclose(disparity, 10.25, atol=.3)
            expected_depth = 20. / (disparity - 2.)
            rays = np.c_[pixels, np.ones(len(pixels))] @ slam.inverse_K.T
            np.testing.assert_allclose(points, rays * expected_depth[:, None], rtol=1e-6)
            np.testing.assert_allclose(slam._supported_extraction[0], points)
            np.testing.assert_allclose(slam._supported_extraction[1], right_u)
            # Cache-hit remeasurement at the exact feature coordinates is the
            # same independent image measurement, regardless of dense input.
            cached_points, cached_right = slam._measure_supported_stereo_pixels(pixels)
            np.testing.assert_allclose(cached_points, points)
            np.testing.assert_allclose(cached_right, right_u)
            outputs.append((points.copy(), right_u.copy()))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        assert np.max(np.abs((pixels[:, 0] - outputs[0][1]) - 10.875)) > .3

        slam._supported_extraction = None  # Simulate a feature-cache hit.
        record = slam._capture_supported_stereo(
            0, pixels, np.zeros((len(pixels), 128), np.float32), (240, 96))
        np.testing.assert_array_equal(record.points, outputs[0][0])
        np.testing.assert_array_equal(record.right_u, outputs[0][1])

        fallback = _slam('verified_fallback')
        try:
            assert extraction_signature(slam, cv) != extraction_signature(fallback, cv)
        finally:
            fallback.close(finish=False)
    finally:
        slam.close(finish=False)


def test_verified_all_rejects_bad_photometry_and_never_falls_back_to_dense_depth():
    left, right = _images()
    pixels = np.array([[170.25, 50.25], [151.2, 38.3], [-1., 50.]], np.float32)
    slam = _slam()
    try:
        slam._prepare_frame_images(left, np.full_like(right, 80))
        slam.current_disparity = np.full(left.shape, 10.25, np.float32)
        points, right_u = slam._measure_stereo_pixels(pixels, capture_supported=True)
        assert np.isnan(points).all()
        assert np.isnan(right_u).all()
        assert np.isnan(slam._supported_extraction[0]).all()
        assert slam.stereo_depth_verification.get('verified', 0) == 0
        assert slam.stereo_depth_verification.get('candidates', 0) == 2

        slam.current_left_gray = slam.current_right_gray = None
        missing, missing_right = slam._measure_stereo_pixels(pixels, capture_supported=True)
        assert np.isnan(missing).all() and np.isnan(missing_right).all()
        assert slam.stereo_depth_verification['missing_images'] == len(pixels)
    finally:
        slam.close(finish=False)


def test_verified_all_fails_closed_on_complex_mutated_q_and_supported_default_remains_dense():
    left, right = _images()
    pixels = np.array([[170.25, 50.25]], np.float32)
    verified = _slam(q_dtype=complex)
    supported = _slam('supported')
    try:
        verified.stereo.Q[0, 0] = 1.+2.j
        verified._prepare_frame_images(left, right)
        verified.current_disparity = np.full(left.shape, 10.25, np.float32)
        invalid, invalid_right = verified._measure_stereo_pixels(pixels)
        assert np.isnan(invalid).all() and np.isnan(invalid_right).all()
        assert verified.stereo_depth_verification['invalid_calibration'] == 1

        supported._prepare_frame_images(left, right)
        supported.current_disparity = np.full(left.shape, 9., np.float32)
        legacy, legacy_right = supported._measure_stereo_pixels(pixels)
        assert np.isfinite(legacy).all()
        assert legacy_right[0] == pixels[0, 0] - 9.
    finally:
        verified.close(finish=False)
        supported.close(finish=False)
