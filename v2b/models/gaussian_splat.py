"""Orthographic BEV rasterisation of 3D Gaussians (the GaussianLSS splat).

GaussianLSS (Lu et al., CVPR 2025) replaces lift-splat's D-fold outer
product and cumsum pooling by one 3D Gaussian per image pixel, drawn
onto the BEV grid with the 3D Gaussian Splatting rasteriser compiled
for feature channels. ``third_party/GaussianLSS`` ships that renderer
hard-wired to a 200 x 200 grid over +-50 m with its own axis layout.
This module is the same renderer with the grid as a parameter and the
output in OpenCOOD's BEV layout: ``(B, C, ny, nx)`` with row = y and
column = x, both ascending from the grid's lower corner, so the tensor
drops into HEAL / CodeFilling where ``LiftSplatShoot.voxel_pooling``
returns.

The rasteriser (``diff_gaussian_rasterization`` from the GaussianLSS
checkout) is imported lazily: the channel count is a compile-time
constant of that extension (128 for the shipped build), and the module
must stay importable without it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# The rasteriser maps its first mean coordinate to the row and its
# second to the column, both *decreasing*: row = h/2 - X'/cell,
# col = w/2 - Y'/cell (measured, see tests). ``swap_axes`` turns an
# OpenCOOD-frame (x, y, z) into that (X', Y', z): X' = -y, Y' = -x.
SWAP = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def swap_axes_mean(
    mean: torch.Tensor, cx: float, cy: float, half_cell: float
) -> torch.Tensor:
    """OpenCOOD-frame means ``(..., 3)`` to the rasteriser's frame.

    ``cx, cy`` is the grid centre; ``half_cell`` shifts by half a cell
    so that a point at a cell's centre lands on that cell's pixel
    (the rasteriser's pixel centres sit on integer coordinates).
    """
    out = torch.empty_like(mean)
    out[..., 0] = cy - mean[..., 1] + half_cell
    out[..., 1] = cx - mean[..., 0] + half_cell
    out[..., 2] = mean[..., 2]
    return out


def swap_axes_cov(cov: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` covariances to the rasteriser's upper triangle.

    Applies ``P cov P^T`` for the axis swap and returns the six entries
    ``(xx, xy, xz, yy, yz, zz)`` the rasteriser reads.
    """
    xx, xy, xz = cov[..., 0, 0], cov[..., 0, 1], cov[..., 0, 2]
    yy, yz, zz = cov[..., 1, 1], cov[..., 1, 2], cov[..., 2, 2]
    return torch.stack((yy, xy, -yz, xx, -xz, zz), dim=-1)


class GaussianBevRenderer(nn.Module):  # type: ignore[misc]
    """Draw per-pixel Gaussians onto an OpenCOOD BEV grid.

    Parameters
    ----------
    channels:
        Feature channels per Gaussian; must equal the rasteriser's
        compiled ``NUM_CHANNELS``.
    xbound, ybound:
        ``(lower, upper, cell)`` as in HEAL's ``grid_conf``.
    opacity_min:
        Gaussians below this opacity are not drawn (GaussianLSS: 0.05).
    """

    def __init__(
        self,
        channels: int,
        xbound: tuple[float, float, float],
        ybound: tuple[float, float, float],
        opacity_min: float = 0.05,
    ) -> None:
        super().__init__()
        if not math.isclose(xbound[2], ybound[2]):
            raise ValueError("the rasteriser needs square cells")
        self.channels = channels
        self.cell = float(xbound[2])
        self.nx = int(round((xbound[1] - xbound[0]) / self.cell))
        self.ny = int(round((ybound[1] - ybound[0]) / self.cell))
        self.cx = float(xbound[0]) + self.cell * self.nx / 2
        self.cy = float(ybound[0]) + self.cell * self.ny / 2
        self.opacity_min = opacity_min
        self._settings: Any = None

    def settings(self, device: torch.device) -> Any:
        from diff_gaussian_rasterization import GaussianRasterizationSettings

        if self._settings is None or self._settings.bg.device != device:
            h, w = self.ny, self.nx
            sh, sw = 1.0 / self.cell, 1.0 / self.cell
            view = torch.tensor(
                [[0.0, sh, 0, 0], [sw, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            )
            proj = torch.tensor(
                [
                    [0.0, -sh, 0, h / 2],
                    [-sw, 0, 0, w / 2],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                ]
            )
            self._settings = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=math.tan(w * self.cell * 0.5),
                tanfovy=math.tan(h * self.cell * 0.5),
                bg=torch.zeros(self.channels, device=device),
                scale_modifier=1,
                viewmatrix=view.to(device),
                projmatrix=proj.to(device),
                sh_degree=0,
                campos=torch.zeros(3, device=device),
                prefiltered=False,
                debug=False,
            )
        return self._settings

    def forward(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        covs: torch.Tensor,
        opacities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one BEV per batch element.

        ``features (B, G, C)``, ``means (B, G, 3)`` in the OpenCOOD
        frame, ``covs (B, G, 3, 3)``, ``opacities (B, G, 1)``. Returns
        ``(B, C, ny, nx)`` and the mean number of Gaussians drawn.
        """
        from diff_gaussian_rasterization import GaussianRasterizer

        rasterizer = GaussianRasterizer()
        rasterizer.set_raster_settings(self.settings(means.device))
        keep = opacities[..., 0] > self.opacity_min
        means_r = swap_axes_mean(means, self.cx, self.cy, 0.5 * self.cell)
        covs_r = swap_axes_cov(covs)
        out = []
        for b in range(means.shape[0]):
            m = keep[b]
            bev, _ = rasterizer(
                means3D=means_r[b][m],
                means2D=None,
                shs=None,
                colors_precomp=features[b][m],
                opacities=opacities[b][m],
                scales=None,
                rotations=None,
                cov3D_precomp=covs_r[b][m],
            )
            out.append(bev)
        drawn = keep.float().sum(1).mean()
        return torch.stack(out, 0), drawn
