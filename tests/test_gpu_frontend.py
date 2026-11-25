"""Tests for Phase 4 GPU device selection, GPU/CPU frontend fallback parity, and native builds."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_kernels.device import select_device
from gpu_kernels.frontend_gpu import GpuFrontendAccelerator
from mapping_geometry import match_descriptors
import native_accel


def test_device_selection_cpu_auto_env_and_strict_cuda(monkeypatch):
    cpu_info = select_device("cpu")
    assert cpu_info.effective == "cpu"
    assert cpu_info.requested == "cpu"
    assert cpu_info.backend_label.startswith("cpu:")

    monkeypatch.setenv("VISUAL_SLAM_DEVICE", "cpu")
    env_info = select_device(None)
    assert env_info.effective == "cpu"

    auto_info = select_device("auto")
    assert auto_info.effective in ("cpu", "cuda")
    if not auto_info.cuda_available and not auto_info.cv2_cuda_available:
        # Non-strict cuda request falls back cleanly to cpu
        fb = select_device("cuda", strict=False)
        assert fb.effective == "cpu"
        assert fb.fallback_reason is not None
        with pytest.raises(RuntimeError, match="CUDA"):
            select_device("cuda", strict=True)

    with pytest.raises(ValueError):
        select_device("tpu")


def test_gpu_frontend_rectification_and_matching_parity(monkeypatch):
    rng = np.random.default_rng(123)
    h, w = 180, 240
    left = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    right = np.roll(left, -10, axis=1)

    K = np.array([[300.0, 0.0, w * 0.5], [0.0, 300.0, h * 0.5], [0.0, 0.0, 1.0]])
    dist = np.array([-0.15, 0.05, 0.001, -0.001, 0.0])
    map_x, map_y = cv.initUndistortRectifyMap(K, dist, np.eye(3), K, (w, h), cv.CV_32FC1)

    accel_cpu = GpuFrontendAccelerator(device="cpu")
    rl, rr = accel_cpu.rectify_stereo_pair(left, right, map_x, map_y, map_x, map_y)
    ref_l = cv.remap(left, map_x, map_y, cv.INTER_LINEAR)
    ref_r = cv.remap(right, map_x, map_y, cv.INTER_LINEAR)
    assert np.array_equal(rl, ref_l)
    assert np.array_equal(rr, ref_r)

    # Descriptor matching parity (native vs pure Python fallback)
    base = rng.integers(0, 210, size=(220, 128), dtype=np.int32)
    da = base.astype(np.float32)
    db = np.vstack(
        [
            np.clip(base[:140] + rng.integers(-5, 6, size=(140, 128)), 0, 255),
            rng.integers(0, 210, size=(100, 128), dtype=np.int32),
        ]
    ).astype(np.float32)

    pairs_native = accel_cpu.match_descriptors_batched(da, db, ratio=0.75)
    monkeypatch.setenv("VISUAL_SLAM_DISABLE_NATIVE", "1")
    pairs_py = match_descriptors(da, db, ratio=0.75)
    assert len(pairs_native) > 80
    assert np.array_equal(pairs_native, pairs_py)

    # If PyTorch is installed, also verify torch tensor rectification & batched matching parity
    accel_torch = GpuFrontendAccelerator(device="cpu", allow_torch_cpu_test=True)
    if accel_torch._torch is not None:
        tl, tr = accel_torch.rectify_stereo_pair(left, right, map_x, map_y, map_x, map_y)
        assert float(np.mean(np.abs(tl.astype(np.float32) - ref_l.astype(np.float32)))) < 1.5
        assert float(np.mean(np.abs(tr.astype(np.float32) - ref_r.astype(np.float32)))) < 1.5
        pairs_torch = accel_torch.match_descriptors_batched(da, db, ratio=0.75)
        assert np.array_equal(pairs_torch, pairs_py)

    # Disparity computation sanity
    disp = accel_cpu.compute_disparity(left, right, num_disparities=64, block_size=5)
    assert disp.shape == (h, w) and disp.dtype == np.float32


def test_build_native_windows_and_linux_command_generation():
    import build_native

    win_msvc = build_native._build_command(
        "cl.exe", Path("fast_sift.cpp"), Path("fast_sift.dll"), "fast_sift", "avx2", "Windows"
    )
    assert "/arch:AVX2" in win_msvc and "/LD" in win_msvc and "/fp:fast" in win_msvc

    win_ba = build_native._build_command(
        "cl.exe", Path("fast_ba.cpp"), Path("fast_ba.dll"), "fast_ba", "avx2", "Windows"
    )
    assert "/fp:precise" in win_ba

    linux_avx2 = build_native._build_command(
        "g++", Path("fast_matcher.cpp"), Path("libfast_matcher.so"), "fast_matcher", "avx2", "Linux"
    )
    assert "-mavx2" in linux_avx2 and "-mfma" in linux_avx2 and "-shared" in linux_avx2

    assert build_native._library_filename("fast_ba", "Windows") == "fast_ba.dll"
    assert build_native._library_filename("fast_ba", "Linux") == "libfast_ba.so"


def test_bench_local_gpu_smoke():
    import bench_local_gpu

    res = bench_local_gpu._bench_device("cpu", iters=2, width=160, height=120)
    assert res["device"]["effective"] == "cpu"
    assert res["rectify_stereo_pair"]["mean_abs_diff_vs_cv_remap"] == 0.0
    assert res["match_descriptors_batched"]["500"]["exact_parity"] is True
