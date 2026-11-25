"""Sparse evaluation of Kornia's scale-space quadratic interpolator."""

import torch
from kornia.geometry.subpix import NonMaximaSuppression3d
from kornia.utils import create_meshgrid3d
from kornia.utils.helpers import safe_solve_with_mask


class SparseQuadInterp3d(torch.nn.Module):
    """Match Kornia's dense score/coordinate interface, solving only at extrema."""

    def __init__(self):
        super().__init__()
        self._nms = NonMaximaSuppression3d((3, 3, 3))
        self._grid = None
        self._grid_key = None

    def forward(self, response):
        # Use the reference's strict interior NMS, including its tie and border
        # behavior. Non-maxima keep their original score and integer coordinates.
        maxima = self._nms(response, mask_only=True)
        indices = maxima.flatten().nonzero().flatten()
        batch, channels, depth, height, width = response.shape
        shape = (depth, height, width)
        key = (shape, response.device, response.dtype)
        if self._grid_key != key:
            self._grid = create_meshgrid3d(
                *shape, normalized_coordinates=False, device=response.device, dtype=response.dtype,
            ).permute(0, 4, 1, 2, 3).contiguous()
            self._grid_key = key
        coords = self._grid.unsqueeze(1).expand(batch, channels, -1, -1, -1, -1)
        if not indices.numel():
            return coords, response

        flat = response.flatten()
        strides = (1, width, height * width)  # x, y, scale
        center = flat[indices]
        delta = indices.new_tensor(strides)
        plus = flat[indices[:, None] + delta]
        minus = flat[indices[:, None] - delta]
        gradient = (plus - minus) * 0.5
        dxx, dyy, dss = (minus - (2 * center[:, None]) + plus).unbind(dim=1)

        def mixed(a, b):
            return 0.25 * (
                flat[indices - a - b] - flat[indices + a - b]
                - flat[indices - a + b] + flat[indices + a + b]
            )

        x, y, scale = strides
        dxy = mixed(x, y)
        # Kornia flips its second-derivative filters along the scale axis.
        dxs, dys = mixed(x, -scale), mixed(y, -scale)
        hessian = torch.stack((dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss), dim=1).reshape(-1, 3, 3)
        solution, _, solved = safe_solve_with_mask(gradient.unsqueeze(-1), hessian)
        offset = -solution.squeeze(-1)
        refine = solved & ~(offset.abs().amax(dim=1) > 0.7)
        offset = torch.where(refine[:, None], offset, 0)
        # A valid solve earns the same bonus even when its offset is discarded.
        # Preserve the addition order of Kornia's reference score.
        score = (center + 0.5 * (gradient * offset).sum(dim=1)) + 10.0 * solved

        scores = response.contiguous().clone()
        scores.view(-1)[indices] = score
        coordinates = coords.clone(memory_format=torch.contiguous_format)
        volume = depth * height * width
        coordinates.view(batch * channels, 3, volume)[indices // volume, :, indices % volume] += offset[:, (2, 0, 1)]
        return coordinates, scores
