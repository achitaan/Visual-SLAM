import cv2 as cv
import numpy as np
import pytest

import cuda_stereo_verification
from frontend_pipeline import FrontendExtractionPipeline
from native_accel import verify_stereo_zncc_avx512
from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam, StereoCamera
from stereo_depth import StereoSearchConfig, verify_stereo_depth_candidates


def shifted_stereo():
    image = np.random.default_rng(784).integers(0, 256, size=(128, 279), dtype=np.uint8)
    return image[:, :272].copy(), image[:, 7:].copy()


def test_cuda_verified_stereo_recovers_manufactured_shift():
    verifier, reason = cuda_stereo_verification.create_cuda_stereo_verifier()
    if verifier is None:
        pytest.skip(reason)
    left, right = shifted_stereo()
    xx, yy = np.meshgrid(np.arange(36, 250, 8), np.arange(12, 117, 8))
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    bounds = (1.0, 24.0)
    config = StereoSearchConfig()
    measured, counts = verifier(left, right, pixels, config, bounds)
    verified = np.isfinite(measured)
    assert counts["verified"] == verified.sum()
    assert verified.mean() > 0.98
    # The right image is the left image shifted by seven pixels. The estimator
    # interpolates the asymmetric neighboring correlations to a subpixel peak.
    np.testing.assert_allclose(pixels[verified, 0] - measured[verified], 7.0, atol=0.15, rtol=0)
    expected = verify_stereo_zncc_avx512(left, right, pixels, config, bounds)
    if expected is not None:
        # The AVX-512 implementation is the reference for coordinates and rejects.
        np.testing.assert_allclose(measured, expected[0], atol=1e-10, rtol=0, equal_nan=True)
        assert {key: value for key, value in counts.items() if key != "candidates"} == expected[1]
    small = pixels[:20]
    actual = verifier(left, right, small, config, bounds)
    expected_small = verify_stereo_depth_candidates(left, right, small, config, bounds)
    # A small batch is processed entirely by the existing CPU verifier.
    assert actual[0].tobytes() == expected_small[0].tobytes()
    assert actual[1] == expected_small[1]
    assert verifier.report()["cuda_batches"] == 1
    assert verifier.report()["cpu_batches"] == 1


@pytest.mark.parametrize("pipelined", [False, True])
def test_unavailable_cuda_stereo_falls_back_exactly(monkeypatch, pipelined):
    monkeypatch.setattr(cuda_stereo_verification, "create_cuda_stereo_verifier",
                        lambda: (None, "test: no CUDA verifier"))
    left, right = shifted_stereo()
    focal, cx, cy, baseline = 100.0, 136.0, 64.0, 0.54
    intrinsics = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal],
                  [0, 0, 1 / baseline, 0]])
    config = MappingConfig(stereo_depth_policy="verified_all", loop_mode="off", features=500)

    def measure(backend):
        stereo = StereoCamera(cv.StereoSGBM_create(numDisparities=32, blockSize=5), q, baseline)
        perf = PerformanceConfig(stereo_verification_backend=backend)
        slam = SharedSlam(intrinsics, stereo=stereo, config=config, performance=perf)
        try:
            assert slam.cuda_stereo_verifier is None
            if backend == "cuda":
                assert slam.cuda_stereo_verifier_reason == "test: no CUDA verifier"
            if pipelined:
                with FrontendExtractionPipeline(slam) as pipeline:
                    pipeline.submit(0, left, right)
                    frame = pipeline.get(0)
                return frame.pixels, frame.descriptors, frame.points, frame.right_u
            slam._prepare_frame_images(left, right)
            return slam._extract(left, right)
        finally:
            slam.close(finish=False)

    reference, fallback = measure("cpu"), measure("cuda")
    assert len(reference[0])
    # The CPU-only pipeline defines byte-for-byte fallback behavior.
    for expected, actual in zip(reference, fallback):
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype, expected.shape, expected.tobytes(),
        )
