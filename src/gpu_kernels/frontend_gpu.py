"""GPU-accelerated stereo rectification, batched descriptor matching, and stereo disparity.

Uses PyTorch CUDA (``grid_sample`` / ``cdist`` / ``topk``) and opportunistic ``cv2.cuda``
with zero Python-loop CUDA syncs, falling back seamlessly to native CPU SIMD kernels
and pure Python/OpenCV on CPU-only machines.
"""

from __future__ import annotations

from threading import Lock
import time
from typing import Any

import cv2 as cv
import numpy as np

from mapping_geometry import match_descriptors
from .device import DeviceInfo, select_device


class GpuFrontendAccelerator:
    """Frontend accelerator supporting local NVIDIA GPUs with transparent CPU fallback."""

    def __init__(
        self,
        device: str | None = None,
        *,
        strict: bool = False,
        allow_torch_cpu_test: bool = False,
    ) -> None:
        self.device_info: DeviceInfo = select_device(device, strict=strict)
        self._allow_torch_cpu_test = bool(allow_torch_cpu_test)
        self._torch: Any = None
        self._torch_device: str | None = None
        self._cv2_cuda: Any = None
        self._grid_cache: dict[tuple[int, int, tuple[int, int]], Any] = {}
        self._cv2_map_cache: dict[tuple[int, int, tuple[int, int]], tuple[Any, Any]] = {}
        self._lock = Lock()
        self._stats = {
            "rectify_gpu_calls": 0,
            "rectify_cpu_calls": 0,
            "match_gpu_calls": 0,
            "match_cpu_calls": 0,
            "stereo_gpu_calls": 0,
            "stereo_cpu_calls": 0,
            "rectify_elapsed_ms": 0.0,
            "match_elapsed_ms": 0.0,
        }

        if self.device_info.effective == "cuda":
            if self.device_info.cuda_available:
                import torch

                torch.backends.cuda.matmul.allow_tf32 = False
                self._torch = torch
                self._torch_device = "cuda"
            if self.device_info.cv2_cuda_available:
                self._cv2_cuda = cv.cuda
        elif self._allow_torch_cpu_test:
            try:
                import torch

                self._torch = torch
                self._torch_device = "cpu"
            except ImportError:
                pass

    @property
    def uses_gpu(self) -> bool:
        return self.device_info.effective == "cuda"

    @property
    def backend_label(self) -> str:
        return self.device_info.backend_label

    def _get_normalized_grid(
        self,
        map_x_f32: np.ndarray,
        map_y_f32: np.ndarray,
        src_hw: tuple[int, int],
    ) -> Any:
        assert self._torch is not None and self._torch_device is not None
        key = (id(map_x_f32), id(map_y_f32), src_hw)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        h, w = src_hw
        gx = (2.0 * np.asarray(map_x_f32, dtype=np.float32) / max(float(w - 1), 1.0)) - 1.0
        gy = (2.0 * np.asarray(map_y_f32, dtype=np.float32) / max(float(h - 1), 1.0)) - 1.0
        grid_np = np.stack([gx, gy], axis=-1)[None, ...]  # (1, H_out, W_out, 2)
        grid_t = self._torch.as_tensor(np.ascontiguousarray(grid_np), device=self._torch_device)
        if len(self._grid_cache) >= 16:
            self._grid_cache.clear()
        self._grid_cache[key] = grid_t
        return grid_t

    def rectify_image(
        self,
        image: np.ndarray,
        map_x_f32: np.ndarray,
        map_y_f32: np.ndarray,
    ) -> np.ndarray:
        """Rectify a single grayscale or color image on GPU (cv2.cuda / torch.grid_sample) or CPU."""
        t0 = time.perf_counter()
        if self._cv2_cuda is not None:
            key = (id(map_x_f32), id(map_y_f32), image.shape[:2])
            maps = self._cv2_map_cache.get(key)
            if maps is None:
                gm_x = self._cv2_cuda.GpuMat()
                gm_y = self._cv2_cuda.GpuMat()
                gm_x.upload(np.ascontiguousarray(map_x_f32, dtype=np.float32))
                gm_y.upload(np.ascontiguousarray(map_y_f32, dtype=np.float32))
                maps = (gm_x, gm_y)
                if len(self._cv2_map_cache) >= 16:
                    self._cv2_map_cache.clear()
                self._cv2_map_cache[key] = maps
            g_src = self._cv2_cuda.GpuMat()
            g_src.upload(np.ascontiguousarray(image))
            g_dst = self._cv2_cuda.remap(g_src, maps[0], maps[1], cv.INTER_LINEAR)
            out = g_dst.download()
            with self._lock:
                self._stats["rectify_gpu_calls"] += 1
                self._stats["rectify_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
            return out

        if self._torch is not None and self._torch_device is not None:
            torch = self._torch
            F = torch.nn.functional
            orig_dtype = image.dtype
            is_gray = image.ndim == 2
            h, w = image.shape[:2]
            grid = self._get_normalized_grid(map_x_f32, map_y_f32, (h, w))
            with torch.inference_mode():
                if is_gray:
                    arr = np.ascontiguousarray(image, dtype=np.float32)[None, None, ...]
                else:
                    arr = np.ascontiguousarray(np.transpose(image, (2, 0, 1)), dtype=np.float32)[None, ...]
                t_in = torch.as_tensor(arr, device=self._torch_device)
                t_out = F.grid_sample(
                    t_in, grid, mode="bilinear", padding_mode="border", align_corners=True
                )
                out_np = t_out[0].cpu().numpy()
            if is_gray:
                out_np = out_np[0]
            else:
                out_np = np.transpose(out_np, (1, 2, 0))
            if orig_dtype == np.uint8:
                out_np = np.clip(np.rint(out_np), 0, 255).astype(np.uint8)
            else:
                out_np = out_np.astype(orig_dtype, copy=False)
            with self._lock:
                self._stats["rectify_gpu_calls"] += 1
                self._stats["rectify_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
            return np.ascontiguousarray(out_np)

        out = cv.remap(image, map_x_f32, map_y_f32, cv.INTER_LINEAR)
        with self._lock:
            self._stats["rectify_cpu_calls"] += 1
            self._stats["rectify_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
        return out

    def rectify_stereo_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        map_x_l: np.ndarray,
        map_y_l: np.ndarray,
        map_x_r: np.ndarray,
        map_y_r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched stereo pair rectification (single GPU kernel launch when PyTorch is active)."""
        if self._torch is not None and self._torch_device is not None and left.shape == right.shape:
            t0 = time.perf_counter()
            torch = self._torch
            F = torch.nn.functional
            orig_dtype = left.dtype
            is_gray = left.ndim == 2
            h, w = left.shape[:2]
            grid_l = self._get_normalized_grid(map_x_l, map_y_l, (h, w))
            grid_r = self._get_normalized_grid(map_x_r, map_y_r, (h, w))
            with torch.inference_mode():
                grids = torch.cat([grid_l, grid_r], dim=0)  # (2, H, W, 2)
                if is_gray:
                    pair = np.stack([left, right], axis=0).astype(np.float32)[:, None, :, :]
                else:
                    pair = np.stack(
                        [np.transpose(left, (2, 0, 1)), np.transpose(right, (2, 0, 1))],
                        axis=0,
                    ).astype(np.float32)
                t_in = torch.as_tensor(np.ascontiguousarray(pair), device=self._torch_device)
                t_out = F.grid_sample(
                    t_in, grids, mode="bilinear", padding_mode="border", align_corners=True
                )
                out_np = t_out.cpu().numpy()
            if is_gray:
                rl, rr = out_np[0, 0], out_np[1, 0]
            else:
                rl = np.transpose(out_np[0], (1, 2, 0))
                rr = np.transpose(out_np[1], (1, 2, 0))
            if orig_dtype == np.uint8:
                rl = np.clip(np.rint(rl), 0, 255).astype(np.uint8)
                rr = np.clip(np.rint(rr), 0, 255).astype(np.uint8)
            else:
                rl = rl.astype(orig_dtype, copy=False)
                rr = rr.astype(orig_dtype, copy=False)
            with self._lock:
                self._stats["rectify_gpu_calls"] += 1
                self._stats["rectify_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
            return np.ascontiguousarray(rl), np.ascontiguousarray(rr)

        return (
            self.rectify_image(left, map_x_l, map_y_l),
            self.rectify_image(right, map_x_r, map_y_r),
        )

    def match_descriptors_batched(
        self,
        desc_a: np.ndarray,
        desc_b: np.ndarray,
        ratio: float = 0.75,
    ) -> np.ndarray:
        """Batched mutual L2 ratio descriptor matcher with zero Python-loop CUDA syncs."""
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return np.empty((0, 2), dtype=int)
        t0 = time.perf_counter()
        if (
            self._torch is not None
            and self._torch_device is not None
            and desc_a.dtype == np.float32
            and desc_b.dtype == np.float32
        ):
            torch = self._torch
            with torch.inference_mode():
                ta = torch.as_tensor(np.ascontiguousarray(desc_a), device=self._torch_device)
                tb = torch.as_tensor(np.ascontiguousarray(desc_b), device=self._torch_device)
                # Full pairwise distance matrix in a single batched op (no per-chunk Python loop syncs)
                dist = torch.cdist(ta, tb, compute_mode="donot_use_mm_for_euclid_dist_if_necessary")
                f_vals, f_idx = torch.topk(dist, 2, dim=1, largest=False, sorted=True)
                b_vals, b_idx = torch.topk(dist, 2, dim=0, largest=False, sorted=True)

                f_ok = f_vals[:, 0] < float(ratio) * f_vals[:, 1]
                b_ok = b_vals[0, :] < float(ratio) * b_vals[1, :]

                f_best = f_idx[:, 0]
                b_best = b_idx[0, :]
                ar = torch.arange(len(desc_a), device=self._torch_device)
                mutual = f_ok & b_ok[f_best] & (b_best[f_best] == ar)
                rows = torch.nonzero(mutual, as_tuple=False).squeeze(1)
                if rows.numel() == 0:
                    pairs = np.empty((0, 2), dtype=int)
                else:
                    cols = f_best[rows]
                    pairs = torch.stack([rows, cols], dim=1).cpu().numpy().astype(int, copy=False)
            with self._lock:
                self._stats["match_gpu_calls"] += 1
                self._stats["match_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
            return pairs

        pairs = match_descriptors(desc_a, desc_b, ratio=float(ratio))
        with self._lock:
            self._stats["match_cpu_calls"] += 1
            self._stats["match_elapsed_ms"] += (time.perf_counter() - t0) * 1000.0
        return pairs

    def compute_disparity(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        *,
        sgbm: Any = None,
        num_disparities: int = 96,
        block_size: int = 5,
    ) -> np.ndarray:
        """Compute float32 disparity via opportunistic cv2.cuda.StereoSGM or CPU StereoSGBM."""
        if self._cv2_cuda is not None and hasattr(self._cv2_cuda, "createStereoSGM"):
            try:
                gpu_sgm = self._cv2_cuda.createStereoSGM(0, int(num_disparities))
                gl = self._cv2_cuda.GpuMat()
                gr = self._cv2_cuda.GpuMat()
                gl.upload(np.ascontiguousarray(left_gray, dtype=np.uint8))
                gr.upload(np.ascontiguousarray(right_gray, dtype=np.uint8))
                gd = gpu_sgm.compute(gl, gr)
                disp = gd.download().astype(np.float32) / 16.0
                with self._lock:
                    self._stats["stereo_gpu_calls"] += 1
                return disp
            except Exception:
                pass
        if sgbm is None:
            sgbm = cv.StereoSGBM_create(
                minDisparity=0,
                numDisparities=int(num_disparities),
                blockSize=int(block_size),
                P1=8 * 3 * int(block_size) ** 2,
                P2=32 * 3 * int(block_size) ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        with self._lock:
            self._stats["stereo_cpu_calls"] += 1
        return disp

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": self.device_info.to_dict(),
                **self._stats,
            }
