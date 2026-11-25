"""Manufactured tensor inputs and exact CPU fallback regression checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2 as cv
import numpy as np
import pytest

import local_gpu_frontend as gpu
from frontend_pipeline import FrontendExtractionPipeline
from performance import PerformanceConfig
from shared_slam import MappingConfig, SharedSlam, StereoCamera


@pytest.fixture(params=["cpu", "cuda"])
def tensor_device(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda":
        device, reason = gpu.detect_cuda()
        if device is None:
            pytest.skip(reason)
        yield torch, f"cuda:{device.index}"
        return
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch, "cpu"
    finally:
        torch.set_num_threads(previous_threads)


def shifted_pair(*, shift=7, height=88, width=151):
    base = np.random.default_rng(53).integers(0, 256, (height, width + shift), dtype=np.uint8)
    return base[:, :width].copy(), base[:, shift:].copy()


def test_gradient_and_pyramid_known_ramp(tensor_device):
    pytest.importorskip("kornia")
    torch, device = tensor_device
    x = torch.arange(67, device=device, dtype=torch.float32)
    y = torch.arange(41, device=device, dtype=torch.float32)
    ramp = (x + 2 * y[:, None])[None, None]
    result = gpu.tensor_gradient_pyramid(ramp, levels=3)
    # A normalized symmetric Gaussian preserves an affine ramp away from reflection padding.
    assert [im.shape[-2:] for im, _ in result] == [(41, 67), (21, 34), (11, 17)]
    torch.testing.assert_close(result[1][0][0, 0, 1:-1, 1:-1], ramp[0, 0, 2:-2:2, 2:-2:2])
    for level, (im, gradients) in enumerate(result):
        assert im.device == gradients.device == ramp.device
        # Unnormalized 3x3 Sobel multiplies the slope by eight.
        margin = 3
        gx, gy = gradients[0, 0, :, margin:-margin, margin:-margin]
        torch.testing.assert_close(gx, torch.full_like(gx, 8 * 2**level))
        torch.testing.assert_close(gy, torch.full_like(gy, 16 * 2**level))


def test_stereo_recovers_known_shift_and_rejects_ambiguous_regions(tensor_device):
    torch, device = tensor_device
    left, right = shifted_pair()
    a, b = [torch.as_tensor(im, device=device)[None, None].float() / 255 for im in (left, right)]
    disparity = gpu.tensor_stereo_disparity(a, b, num_disparities=17, chunk_size=4)
    assert disparity.device == a.device and disparity.dtype == torch.float32
    # The manufactured image has an exact left-minus-right disparity of 7 pixels.
    torch.testing.assert_close(disparity[2:-2, 9:-2], torch.full_like(disparity[2:-2, 9:-2], 7))
    assert (disparity[:2] == -1).all() and (disparity[:, -2:] == -1).all()
    # Every candidate in a constant image has the same SAD; none is unique.
    flat = torch.zeros_like(a)
    assert (gpu.tensor_stereo_disparity(flat, flat, num_disparities=17) == -1).all()


def test_mutual_ratio_sift_matching_known_permutation_and_ties(tensor_device):
    torch, device = tensor_device
    first = torch.eye(4, 128, device=device)
    second = first[[2, 0, 3, 1]]
    expected = torch.tensor([[0, 1], [1, 3], [2, 0], [3, 2]], device=device)
    # Each unit vector has a single zero-distance counterpart in the permutation.
    torch.testing.assert_close(gpu.tensor_match_sift(first, second), expected)
    # Duplicate zero-distance candidates violate the strict ratio test in each direction.
    assert gpu.tensor_match_sift(first, first.repeat(2, 1)).shape == (0, 2)
    assert gpu.tensor_match_sift(first[:1], first).shape == (0, 2)


def test_scale_space_sift_describes_translated_texture(tensor_device):
    pytest.importorskip("kornia")
    torch, device = tensor_device
    frontend = gpu.TensorFrontend(device, features=100)
    left, right = shifted_pair(height=128, width=192)
    a, b = frontend.image(left), frontend.image(right)
    pixels, desc, octaves, responses = frontend.extract(a)
    other, other_desc, _, _ = frontend.extract(b)
    assert 0 < len(pixels) <= 100 and desc.shape == (len(pixels), 128)
    assert octaves.shape == responses.shape == (len(pixels),)
    assert all(candidate.device == a.device for candidate in (pixels, desc, octaves, responses))
    assert desc.dtype == torch.float32 and octaves.dtype == torch.int16
    assert torch.isfinite(desc).all() and (responses > 0).all()
    torch.testing.assert_close(desc.norm(dim=1), torch.full_like(responses, 512), atol=0.05, rtol=0)
    matches = gpu.tensor_match_sift(desc, other_desc)
    shift = pixels[matches[:, 0]] - other[matches[:, 1]]
    # The exact shift is [7, 0]; resampling at deeper octaves is not pixel-exact.
    expected = torch.tensor([7.0, 0.0], device=device)
    assert ((shift - expected).abs().amax(dim=1) < 0.5).sum() >= 10
    assert frontend.extract(torch.zeros_like(a))[1].shape == (0, 128)


def make_slam(backend, *, grid_bucketing=False):
    focal, center_x, center_y, baseline = 100, 75, 44, 0.54
    intrinsics = np.array([[focal, 0, center_x], [0, focal, center_y], [0, 0, 1]])
    q = np.array([[1, 0, 0, -center_x], [0, 1, 0, -center_y], [0, 0, 0, focal], [0, 0, 1 / baseline, 0]])
    stereo = StereoCamera(cv.StereoSGBM_create(numDisparities=32, blockSize=5), q, baseline)
    config = MappingConfig(loop_mode="off", features=100, use_grid_bucketing=grid_bucketing)
    return SharedSlam(intrinsics, stereo=stereo, config=config,
                      performance=PerformanceConfig(frontend_backend=backend))


def snapshot(slam, left, right, pipelined):
    if pipelined:
        with FrontendExtractionPipeline(slam) as pipeline:
            pipeline.submit(0, left, right)
            result = pipeline.get(0)
        return result.pixels, result.descriptors, result.points, result.right_u, result.octaves, result.disparity
    slam._prepare_frame_images(left, right)
    return (*slam._extract(left, right), slam._current_octaves, slam.current_disparity)


@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("grid_bucketing", [False, True])
def test_unavailable_cuda_falls_back_bit_exact(monkeypatch, pipelined, grid_bucketing):
    monkeypatch.setattr(gpu, "detect_cuda", lambda: (None, "test: no CUDA"))
    left, right = shifted_pair()
    cpu, fallback = [make_slam(backend, grid_bucketing=grid_bucketing) for backend in ("cpu", "cuda")]
    try:
        assert fallback.cuda_frontend is None and fallback.cuda_frontend_reason == "test: no CUDA"
        baseline = snapshot(cpu, left, right, pipelined)
        result = snapshot(fallback, left, right, pipelined)
        # Compare bytes, including NaN encoding, against the unchanged CPU implementation.
        for old, new in zip(baseline, result):
            assert (old.shape, old.dtype, old.tobytes()) == (new.shape, new.dtype, new.tobytes())
    finally:
        cpu.close(finish=False)
        fallback.close(finish=False)


def test_tensor_frontend_integrates_with_sync_and_pipeline(tensor_device):
    pytest.importorskip("kornia")
    _, device = tensor_device
    slam = make_slam("cpu")
    try:
        slam.cuda_frontend = gpu.TensorFrontend(device, features=100)
        left, right = shifted_pair()
        sync, pipelined = [snapshot(slam, left, right, mode) for mode in (False, True)]
        # Both consumers use the same tensor frontend; independent pixel-shift oracle for disparity.
        for a, b in zip(sync, pipelined):
            np.testing.assert_allclose(a, b, atol=1e-5)
        np.testing.assert_array_equal(sync[-1][2:-2, 9:-2], 7)
    finally:
        slam.close(finish=False)


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_benchmark_cpu_cli_smoke(backend):
    script = Path(__file__).resolve().parents[1] / "scripts" / "bench_local_gpu.py"
    result = subprocess.run(
        [sys.executable, str(script), "--backend", backend, "--frames", "1", "--warmup", "0",
         "--repeats", "1", "--features", "30", "--json"],
        check=True, capture_output=True, text=True, timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    report = json.loads(result.stdout)
    assert (report["width"], report["height"]) == (1241, 376)
    assert report["requested_backend"] == backend and list(report["results"]) == ["cpu"]
    assert bool(report["cuda_unavailable_reason"]) == (backend == "auto")
    stats = report["results"]["cpu"]
    assert stats["samples"] == 1 and stats["effective_fps"] == pytest.approx(1000 / stats["mean_ms"])
