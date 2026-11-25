"""Optional tensor frontend for local NVIDIA GPUs; no eager CUDA imports."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CudaDevice:
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_vram_bytes: int
    torch_version: str
    cuda_version: str


def detect_cuda() -> tuple[CudaDevice | None, str | None]:
    """Return the selected NVIDIA device, or the reason CUDA cannot run."""
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            return None, "PyTorch cannot access an NVIDIA CUDA device"
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        probe = torch.ones(1, device=f"cuda:{index}")
        probe.add_(1)
        torch.cuda.synchronize(index)
        return CudaDevice(
            index, props.name, (props.major, props.minor), props.total_memory,
            str(torch.__version__), torch.version.cuda,
        ), None
    except (ImportError, OSError, RuntimeError) as error:
        return None, str(error)


def tensor_gradient_pyramid(
    image: torch.Tensor, *, levels: int = 3,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return (image, Sobel [1,1,2,H,W]) at each scale for a float [1,1,H,W] image."""
    from kornia.filters import spatial_gradient
    from torch.nn import functional as F

    if levels < 1:
        raise ValueError("levels must be positive")
    row = image.new_tensor([1, 4, 6, 4, 1])
    kernel = (row[:, None] * row[None, :] / 256)[None, None]
    pyramid = []
    for level in range(levels):
        pyramid.append((image, spatial_gradient(image, normalized=False)))
        if level + 1 == levels or min(image.shape[-2:]) <= 2:
            break
        image = F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), kernel, stride=2)
    return tuple(pyramid)


def tensor_stereo_disparity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    num_disparities: int = 96,
    block_size: int = 5,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Rectified SAD block matching for float [1,1,H,W] inputs on one device.

    Returns float32 left-minus-right disparities in pixels. A value of -1 means
    unsupported (border, zero disparity, or ambiguous first and second costs).
    Only a chunk of the disparity cost volume is live at any one time.
    """
    import torch
    from torch.nn import functional as F

    if left.ndim != 4 or left.shape[:2] != (1, 1) or left.shape != right.shape:
        raise ValueError("stereo disparity requires matching [1,1,H,W] images")
    if num_disparities < 2 or block_size < 3 or block_size % 2 != 1 or chunk_size < 2:
        raise ValueError("need at least two disparities, an odd block >= 3, and chunk size >= 2")
    h, w = left.shape[-2:]
    disparities = min(num_disparities, w - block_size + 1)
    result = left.new_full((h, w), -1, dtype=torch.float32)
    if h < block_size or disparities < 2:
        return result

    # The selected disparity indices are discrete; there is no autograd path.
    left, right = left.detach(), right.detach()
    radius = block_size // 2
    right_pad = F.pad(right[0, 0], (disparities - 1, 0))
    # Adjacent disparities are overlapping views of the right image. Write
    # their differences straight into contiguous storage for the pooling kernel.
    windows = right_pad.unfold(-1, w, 1).transpose(0, 1)
    diff_buffer = torch.empty(
        (min(chunk_size, disparities), h, w),
        device=left.device, dtype=torch.promote_types(left.dtype, right.dtype),
    )
    best = left.new_full((h - 2 * radius, w - 2 * radius), float("inf"))
    second = best.clone()
    best_index = torch.zeros_like(best, dtype=torch.long)
    centers = torch.arange(radius, w - radius, device=left.device)
    for start in range(0, disparities, chunk_size):
        stop = min(start + chunk_size, disparities)
        ds = torch.arange(stop - 1, start - 1, -1, device=left.device)
        shifted = windows[disparities - stop:disparities - start]
        diff = diff_buffer[:stop - start]
        torch.sub(left[0], shifted, out=diff)
        costs = F.avg_pool2d(diff.abs_()[None], block_size, stride=1)[0]
        # The entire right block, not just its center, must be in bounds.
        costs.masked_fill_(centers < ds[:, None, None] + radius, float("inf"))
        values, local_ids = costs.topk(min(2, len(ds)), dim=0, largest=False, sorted=True)
        chunk_best = values[0]
        better = chunk_best < best
        best_index = torch.where(better, ds[local_ids[0]], best_index)
        second = torch.where(better, best, torch.minimum(second, chunk_best))
        if len(ds) > 1:
            second = torch.minimum(second, values[1])
        best = torch.minimum(best, chunk_best)

    # A single in-bounds candidate offers no evidence of a unique correspondence.
    valid = (best_index > 0) & torch.isfinite(second) & (best < 0.9 * second)
    result[radius:-radius, radius:-radius] = torch.where(valid, best_index, -1)
    return result


def tensor_match_sift(first: torch.Tensor, second: torch.Tensor, ratio: float = 0.7) -> torch.Tensor:
    """Mutual strict Lowe-ratio matches, ordered by the first descriptor index."""
    import torch

    if min(len(first), len(second)) < 2:
        return first.new_empty((0, 2), dtype=torch.long)

    def direction(query: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
        rows = []
        for block in query.split(512):
            distances = torch.cdist(block, train)
            values, indices = distances.topk(2, dim=1, largest=False, sorted=True)
            rows.append(torch.where(values[:, 0] < ratio * values[:, 1], indices[:, 0], -1))
        return torch.cat(rows)

    forward, backward = direction(first, second), direction(second, first)
    query = torch.arange(len(first), device=first.device)
    keep = (forward >= 0) & (backward[forward] == query)
    return torch.stack((query[keep], forward[keep]), dim=1)


class TensorFrontend:
    """Device-resident Kornia scale-space SIFT and PyTorch stereo."""

    def __init__(self, device: str, *, features: int, grid_bucketing: bool = False):
        import torch
        from kornia.feature import SIFTFeatureScaleSpace
        from kornia.geometry.transform import ScalePyramid

        self.torch = torch
        self.device = torch.device(device)
        self.features = features
        self.grid_bucketing = grid_bucketing
        self._frame_lock = Lock()
        cap = max(int(features * 1.4), 2100) if grid_bucketing else features
        self.sift = SIFTFeatureScaleSpace(num_features=cap, rootsift=False, device=self.device).eval()
        # Use the native resolution as octave zero: doubling both axes would
        # quadruple the working set for the largest 3-D DoG/NMS tensors.
        self.sift.detector.scale_pyr = ScalePyramid(3, 1.6, 32, double_image=False).to(self.device)
        if self.device.type == "cuda":
            from cuda_sift_localization import SparseQuadInterp3d

            self.sift.detector.nms = SparseQuadInterp3d()

    def image(self, gray: np.ndarray) -> torch.Tensor:
        if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
            raise ValueError("local CUDA frontend requires a nonempty uint8 grayscale image")
        return self.torch.as_tensor(np.ascontiguousarray(gray), device=self.device)[None, None].float() / 255

    def extract(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pixel coordinates, SIFT rows, octave estimates, and DoG responses."""
        from kornia.feature import get_laf_center, get_laf_scale

        torch = self.torch
        # Kornia's reflected Gaussian kernels require support even after downsampling.
        if min(image.shape[-2:]) < 32:
            return (
                image.new_empty((0, 2)), image.new_empty((0, 128)),
                image.new_empty(0, dtype=torch.int16), image.new_empty(0),
            )
        with torch.inference_mode():
            lafs, responses, desc = self.sift(image)
            pixels = get_laf_center(lafs)[0]
            # This detector uses a 6-sigma LAF and octave zero starts at sigma=1.6.
            sigma = get_laf_scale(lafs).reshape(-1) / 6
            responses = responses.reshape(-1)
            # SharedSlam also uses absolute L2 thresholds in OpenCV SIFT units.
            desc = desc[0] * 512
            h, w = image.shape[-2:]
            keep = (
                (responses > 0) & (sigma > 0) & torch.isfinite(desc).all(dim=1)
                & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            )
            octaves = (sigma[keep] / 1.6).log2().floor().to(torch.int16)
            return pixels[keep], desc[keep], octaves, responses[keep]

    def numpy_features(self, features, image_size: tuple[int, int]):
        """Convert at the SLAM boundary; apply the existing spatial grid policy."""
        pixels, descriptors, octaves, responses = (x.cpu().numpy() for x in features)
        if self.grid_bucketing:
            from frontend_features import select_grid_bucketed_indices

            keep = select_grid_bucketed_indices(pixels, responses, image_size, self.features)
            return pixels[keep], descriptors[keep], octaves[keep]
        return pixels, descriptors, octaves

    def frame(
        self, gray: np.ndarray, right_gray: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract one frame in the formats used by both SharedSlam frontends."""
        # Lazy materialization and a prefetched frame can share this module.
        with self._frame_lock:
            left = self.image(gray)
            pixels, desc, octaves = self.numpy_features(self.extract(left), gray.shape[::-1])
            disparity = None
            if right_gray is not None:
                disparity = tensor_stereo_disparity(left, self.image(right_gray)).cpu().numpy()
        return pixels, desc, octaves, disparity


def create_cuda_frontend(
    *, features: int, grid_bucketing: bool = False,
) -> tuple[TensorFrontend | None, CudaDevice | None, str | None]:
    device, reason = detect_cuda()
    if device is None:
        return None, None, reason
    try:
        return TensorFrontend(
            f"cuda:{device.index}", features=features, grid_bucketing=grid_bucketing,
        ), device, None
    except (ImportError, OSError, RuntimeError) as error:
        return None, device, str(error)
