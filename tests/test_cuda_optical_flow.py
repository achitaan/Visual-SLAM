import cv2 as cv
import numpy as np
import pytest

import cuda_optical_flow
from frontend_pipeline import FrontendExtractionPipeline
from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam, StereoCamera


def _translated_images():
    texture = np.random.default_rng(302).integers(0, 256, (56, 72), dtype=np.uint8)
    previous = cv.resize(texture, (720, 560), interpolation=cv.INTER_CUBIC)
    displacement = np.array([8, -8], dtype=np.float32)
    current = cv.warpAffine(previous, np.column_stack((np.eye(2), displacement)),
                            (previous.shape[1], previous.shape[0]), borderMode=cv.BORDER_REPLICATE)
    # Keep every tested patch inside the overlapping texture at the coarsest scale.
    x, y = np.meshgrid(np.arange(218, 511, 29), np.arange(213, 352, 23))
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32).reshape(-1, 1, 2)
    return previous, current, points, displacement


@pytest.mark.parametrize("levels", [3, 4])
def test_cuda_pyramid_tracking_recovers_known_displacement(levels):
    backend, reason = cuda_optical_flow.create_cuda_optical_flow()
    if backend is None:
        pytest.skip(reason)
    previous, current, points, displacement = _translated_images()
    previous_device = backend.prepare(previous, levels)
    current_device = backend.prepare(current, levels)
    assert previous_device is not None and current_device is not None
    flow = backend.track_pair(previous_device, current_device, points, points + displacement, levels)
    assert flow is not None
    forward, forward_ok, backward, backward_ok = flow
    assert forward.shape == backward.shape == points.shape
    assert forward.dtype == backward.dtype == np.float32
    assert forward_ok.dtype == backward_ok.dtype == np.uint8
    assert forward_ok.shape == backward_ok.shape == (len(points), 1)
    assert forward_ok.all() and backward_ok.all()
    # The images were created by translating the same texture by (8, -8) pixels.
    np.testing.assert_allclose(forward, points + displacement, rtol=0, atol=0.05)
    np.testing.assert_allclose(backward, points, rtol=0, atol=0.05)
    assert backend.report()["fused_pairs"] == 1


@pytest.mark.parametrize("failure_mode", ["startup", "tracking"])
def test_cuda_flow_failure_uses_identical_cpu_pyramid_path(monkeypatch, failure_mode):
    backend = None
    if failure_mode == "tracking":
        backend, reason = cuda_optical_flow.create_cuda_optical_flow()
        if backend is None:
            pytest.skip(reason)
        monkeypatch.setattr(backend, "track_pair", lambda *args: None)
    monkeypatch.setattr(cuda_optical_flow, "create_cuda_optical_flow",
                        lambda: (backend, "test unavailable" if backend is None else None))
    previous, current, points, offset = _translated_images()
    k = np.array([[250., 0., 360.], [0., 250., 280.], [0., 0., 1.]])
    q = np.array([[1., 0., 0., -360.], [0., 1., 0., -280.], [0., 0., 0., 250.], [0., 0., 2., 0.]])
    config = MappingConfig(loop_mode="off", features=400)

    def run(requested):
        camera = StereoCamera(cv.StereoSGBM_create(numDisparities=16, blockSize=5), q, 0.5)
        perf = PerformanceConfig(optical_flow_backend=requested, pipeline_extraction=True)
        slam = SharedSlam(k, stereo=camera, config=config, performance=perf)
        try:
            if requested == "cuda":
                assert slam.cuda_optical_flow is backend
                if backend is None:
                    assert slam.cuda_optical_flow_reason == "test unavailable"
            else:
                assert slam.cuda_optical_flow is None
            with FrontendExtractionPipeline(slam, lazy_extraction=True) as pipeline:
                pipeline.submit(0, previous, previous)
                first = pipeline.get(0)
                pipeline.submit(1, current, current)
                second = pipeline.get(1)
                if requested == "cuda" and backend is not None:
                    assert isinstance(first.lk_pyramid, cuda_optical_flow.CudaLkPyramid)
                    assert isinstance(second.lk_pyramid, cuda_optical_flow.CudaLkPyramid)
                slam.previous_gray, slam.current_gray = first.gray, second.gray
                slam.previous_lk_pyramid, slam.current_lk_pyramid = first.lk_pyramid, second.lk_pyramid
                return slam._track_pyramid_pair(points, points + offset, 3)
        finally:
            slam.close(finish=False)

    reference, fallback = run("cpu"), run("cuda")
    if reference is None:
        # Without the optional DLL, both requests use the existing OpenCV caller.
        assert fallback is None
    else:
        assert fallback is not None
        assert [array.tobytes() for array in fallback] == [array.tobytes() for array in reference]
