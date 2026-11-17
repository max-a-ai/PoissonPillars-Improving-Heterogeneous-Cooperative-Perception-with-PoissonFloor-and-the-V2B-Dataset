"""Self-supervised masked pretraining of the ego trunk, no labels.

CooPre-style: a share of the occupied BEV cells of the trunk's output
map (0.8 m for the m1 pillars and the m3 voxels) is masked (every
pillar or voxel inside them removed from the encoder's input) and the
trunk has to predict, from the surrounding ones, what the masked cells
held. The targets are the ones the capacity and codebook probes found
readable from BEV context: the cell's point count and its height
profile (``bins`` over the grid's z range), computed from the very
pillars the encoder would have seen. The decoder is one 1 x 1 conv on
the trunk's 256-channel map, so what is learned sits in the trunk.

The wrapper keeps the detector's attribute names (``encoder_<m>``,
``backbone_<m>``, ``layers_<m>``, ``shrink_conv_<m>`` for the ego
modality ``m``: m1 pillars, m3 SECOND voxels), so its checkpoint
warm-starts a detector run through ``init_from`` with the whole trunk
adopted and the heads fresh; ``decoder`` is dropped there by name. The
output map's size is read from the trunk on the first forward, and the
voxel grid (``voxel_size`` over ``lidar_range``) is folded onto it.

Spec key ``pretrain: {mask_ratio: 0.7, bins: 12}``; the trainer then
runs validation loss only (no AP) and keeps the lowest-loss epoch.

The S28 ablation knobs, all off by default (the S19 recipe):

* ``drop_below`` (m): points more than this far under the re-based
  floor (``floor_z``) are dropped before anything else, as sensor
  noise and mirror returns under the road;
* ``exclude_ground``: cells holding only road returns (nothing above
  ``floor_z`` + 0.15 m) are neither masked nor scored, so the trunk
  spends its capacity on structure, not on predicting flat asphalt;
* ``max_range`` (m): cells beyond it are neither masked nor scored;
* ``intensity_bins``: an extra target, the cell's intensity histogram
  (over 0 .. 0.4 of the squashed value, the last bin open), so the
  trunk has to encode reflectivity, which marks pedestrians;
* ``block``: mask ``block x block`` blocks of cells instead of single
  cells, so the context needed is longer-range;
* ``decoder_layers``: 2 puts a 3 x 3 conv and a ReLU before the 1 x 1;
* ``occupancy``: predict which height bins are occupied (binary) in
  place of the normalised profile (soft cross-entropy).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

GROUND_BAND = 0.15
INTENSITY_TOP = 0.4


class MaskedPretrainModel(nn.Module):  # type: ignore[misc]
    """The ego trunk with a reconstruction head; forward takes HEAL's
    ``batch["ego"]`` and returns predictions, targets and the mask."""

    def __init__(
        self,
        detector: Any,
        lidar_range: list[float],
        modality: str = "m1",
        voxel_size: list[float] | None = None,
        mask_ratio: float = 0.7,
        bins: int = 12,
        seed: int = 0,
        floor_z: float = -1.9,
        drop_below: float | None = None,
        exclude_ground: bool = False,
        max_range: float | None = None,
        intensity_bins: int = 0,
        block: int = 1,
        decoder_layers: int = 1,
        occupancy: bool = False,
    ) -> None:
        super().__init__()
        self.modality = str(modality)
        for part in ("encoder", "backbone", "layers", "shrink_conv"):
            name = f"{part}_{self.modality}"
            setattr(self, name, getattr(detector, name))
        self.layers_num = int(getattr(detector, f"layers_num_{self.modality}"))
        shrink = self._part("shrink_conv")
        in_ch = int(
            getattr(shrink, "out_channels", 0) or _out_channels(shrink)
        )
        out_ch = 1 + bins + intensity_bins
        if decoder_layers == 1:
            self.decoder: nn.Module = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        elif decoder_layers == 2:
            self.decoder = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
            )
        else:
            raise ValueError("decoder_layers is 1 or 2")
        self.mask_ratio = float(mask_ratio)
        self.bins = int(bins)
        self.intensity_bins = int(intensity_bins)
        self.z0, self.z1 = float(lidar_range[2]), float(lidar_range[5])
        self.x0, self.y0 = float(lidar_range[0]), float(lidar_range[1])
        self.x1, self.y1 = float(lidar_range[3]), float(lidar_range[4])
        if voxel_size is None:
            scatter = getattr(self._part("encoder"), "scatter", None)
            if scatter is None:
                raise ValueError("voxel_size is needed without a scatter")
            nx, ny = int(scatter.nx), int(scatter.ny)
        else:
            nx = round((float(lidar_range[3]) - self.x0) / voxel_size[0])
            ny = round((float(lidar_range[4]) - self.y0) / voxel_size[1])
        #: the encoder's input grid (pillars or voxels) in x and y
        self.grid = (int(nx), int(ny))
        #: the trunk's output map and the grid stride onto it, read on
        #: the first forward (m1: 2 pillars per cell; m3: 8 voxels)
        self.map_hw: tuple[int, int] | None = None
        self.stride = (1, 1)
        self.floor_z = float(floor_z)
        self.drop_below = drop_below
        self.exclude_ground = bool(exclude_ground)
        self.max_range = max_range
        self.block = int(block)
        self.occupancy = bool(occupancy)
        self.gen = torch.Generator().manual_seed(seed)

    def _part(self, part: str) -> Any:
        return getattr(self, f"{part}_{self.modality}")

    def trunk(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        m = self.modality
        scattered = self._part("encoder")({f"inputs_{m}": inputs}, m)
        feature = self._part("backbone")({"spatial_features": scattered})[
            "spatial_features_2d"
        ]
        layers = self._part("layers")
        feats = [feature]
        for i in range(1, self.layers_num):
            feature = layers.get_layer_i_feature(feature, layer_i=i)
            feats.append(feature)
        fused = layers.decode_multiscale_feature(feats)
        out: torch.Tensor = self._part("shrink_conv")(fused)
        return out

    def _map_size(self, inputs: dict[str, torch.Tensor]) -> tuple[int, int]:
        """The trunk's output map (h, w) and the grid stride onto it,
        measured once on an unmasked forward."""
        if self.map_hw is None:
            with torch.no_grad():
                out = self.trunk(inputs)
            h, w = int(out.shape[2]), int(out.shape[3])
            nx, ny = self.grid
            if ny % h or nx % w:
                raise RuntimeError(
                    f"trunk map {(h, w)} does not tile the input grid "
                    f"{(ny, nx)} of modality {self.modality}"
                )
            self.map_hw = (h, w)
            self.stride = (nx // w, ny // h)
        return self.map_hw

    def _drop_below_floor(
        self, feats: torch.Tensor, coords: torch.Tensor, nums: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repack the pillars without the points under the floor."""
        assert self.drop_below is not None
        n_max = feats.shape[1]
        valid = torch.arange(n_max, device=feats.device)[None, :] < nums.view(
            -1, 1
        )
        keep = valid & (feats[..., 2] >= self.floor_z - self.drop_below)
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
        feats = torch.gather(
            feats, 1, order.unsqueeze(-1).expand_as(feats)
        ).clone()
        nums = keep.sum(1)
        n_pts = torch.arange(n_max, device=feats.device)[None, :] < nums.view(
            -1, 1
        )
        feats[~n_pts] = 0.0
        pillars = nums > 0
        return feats[pillars], coords[pillars], nums[pillars]

    def _block_mask(
        self, occupied: torch.Tensor, b: int, h: int, w: int
    ) -> torch.Tensor:
        """A share ``mask_ratio`` of the scorable cells, as single cells
        or as ``block x block`` blocks grown from sampled seeds."""
        draw = torch.rand(occupied.shape, generator=self.gen).to(
            occupied.device
        )
        if self.block <= 1:
            return occupied & (draw < self.mask_ratio)
        seeds = (draw < self.mask_ratio / (self.block * self.block)).view(
            b, 1, h, w
        )
        grown = functional.max_pool2d(
            seeds.float(), self.block, stride=1, padding=self.block // 2
        )
        grown = grown[..., :h, :w].view(b * h * w) > 0
        return occupied & grown

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = batch[f"inputs_{self.modality}"]
        feats = inputs["voxel_features"]  # (P, N, F)
        coords = inputs["voxel_coords"]  # (P, 4): batch, z, y, x
        nums = inputs["voxel_num_points"]  # (P,)
        if self.drop_below is not None:
            feats, coords, nums = self._drop_below_floor(feats, coords, nums)
        device = feats.device
        b = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        # the map the trunk produces: the input grid over the stride
        h, w = self._map_size(inputs)
        sx, sy = self.stride
        cy = (coords[:, 2].long() // sy).clamp(0, h - 1)
        cx = (coords[:, 3].long() // sx).clamp(0, w - 1)
        cell = (coords[:, 0].long() * h + cy) * w + cx  # (P,)
        n_cells = b * h * w

        # targets from every pillar, before masking
        valid = torch.arange(feats.shape[1], device=device)[
            None, :
        ] < nums.view(-1, 1)
        z = feats[..., 2]
        frac = (z - self.z0) / max(self.z1 - self.z0, 1e-6)
        zbin = (frac * self.bins).long().clamp(0, self.bins - 1)
        count = torch.zeros(n_cells, device=device)
        count.index_add_(0, cell, nums.float())
        hist = torch.zeros(n_cells * self.bins, device=device)
        idx = (cell[:, None] * self.bins + zbin)[valid]
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        hist = hist.view(n_cells, self.bins)
        occupancy = (hist > 0).float()
        hist = hist / hist.sum(1, keepdim=True).clamp(min=1.0)
        ihist = None
        if self.intensity_bins:
            inten = feats[..., 3].clamp(min=0.0)
            ibin = (inten / INTENSITY_TOP * self.intensity_bins).long()
            ibin = ibin.clamp(0, self.intensity_bins - 1)
            ihist = torch.zeros(n_cells * self.intensity_bins, device=device)
            iidx = (cell[:, None] * self.intensity_bins + ibin)[valid]
            ihist.index_add_(
                0, iidx, torch.ones_like(iidx, dtype=torch.float32)
            )
            ihist = ihist.view(n_cells, self.intensity_bins)
            ihist = ihist / ihist.sum(1, keepdim=True).clamp(min=1.0)

        # which cells may be masked and scored
        scorable = count > 0
        if self.exclude_ground:
            zmax = torch.full((n_cells,), -1e9, device=device)
            zmax.scatter_reduce_(
                0,
                cell[:, None].expand_as(z)[valid],
                z[valid],
                reduce="amax",
            )
            scorable &= zmax > self.floor_z + GROUND_BAND
        if self.max_range is not None:
            gy = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
            gx = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
            cell_x = (float(self.x1) - self.x0) / w
            cell_y = (float(self.y1) - self.y0) / h
            px = self.x0 + (gx.float() + 0.5) * cell_x
            py = self.y0 + (gy.float() + 0.5) * cell_y
            scorable &= (px * px + py * py).sqrt().reshape(
                -1
            ) <= self.max_range

        masked = self._block_mask(scorable, b, h, w)
        keep = ~masked[cell]
        if int(keep.sum()) == 0:
            keep[:] = True
            masked[:] = False
        kept = {
            "voxel_features": feats[keep],
            "voxel_coords": coords[keep],
            "voxel_num_points": nums[keep],
        }
        fused = self.trunk(kept)  # (B, C, h, w)
        pred = self.decoder(fused)  # (B, 1 + bins [+ ibins], h, w)
        if pred.shape[0] != b or pred.shape[2] != h or pred.shape[3] != w:
            raise RuntimeError(
                f"trunk map {tuple(pred.shape)} does not match the "
                f"cell grid ({b}, {h}, {w}) at stride {self.stride}"
            )
        out = {
            "recon": pred,
            "target_count": count.view(b, h, w),
            "target_hist": hist.view(b, h, w, self.bins).permute(0, 3, 1, 2),
            "target_occ": occupancy.view(b, h, w, self.bins).permute(
                0, 3, 1, 2
            ),
            "masked": masked.view(b, h, w),
            "occupied": (count > 0).view(b, h, w),
            "bins": self.bins,
            "occupancy": self.occupancy,
        }
        if ihist is not None:
            out["target_ihist"] = ihist.view(
                b, h, w, self.intensity_bins
            ).permute(0, 3, 1, 2)
        return out


def _out_channels(module: Any) -> int:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("shrink head without a conv: cannot size the decoder")
    return int(convs[-1].out_channels)


class PretrainLoss:
    """Count (smooth L1 on log1p) and profile (soft cross-entropy, or
    binary occupancy per bin) on the masked cells, plus the intensity
    profile when predicted; the same call signature as HEAL's criterion."""

    def __init__(
        self,
        count_weight: float = 1.0,
        profile_weight: float = 1.0,
        intensity_weight: float = 1.0,
    ) -> None:
        self.count_weight = float(count_weight)
        self.profile_weight = float(profile_weight)
        self.intensity_weight = float(intensity_weight)
        self.loss_dict: dict[str, float] = {}

    def __call__(
        self, output: dict[str, Any], target: Any = None, suffix: str = ""
    ) -> torch.Tensor:
        pred = output["recon"]
        masked = output["masked"]
        count = output["target_count"]
        bins = int(output["bins"])
        if int(masked.sum()) == 0:
            zero = pred.sum() * 0.0
            self.loss_dict = {
                "recon_count": 0.0,
                "recon_profile": 0.0,
                "recon_intensity": 0.0,
                "masked_cells": 0.0,
                "total_loss": 0.0,
            }
            return zero
        log_count = torch.log1p(count[masked])
        pred_count = pred[:, 0][masked]
        count_loss = functional.smooth_l1_loss(pred_count, log_count)
        logits = pred[:, 1 : 1 + bins].permute(0, 2, 3, 1)[masked]  # (M, bins)
        if output.get("occupancy"):
            occ_m = output["target_occ"].permute(0, 2, 3, 1)[masked]
            profile_loss = functional.binary_cross_entropy_with_logits(
                logits, occ_m
            )
        else:
            hist_m = output["target_hist"].permute(0, 2, 3, 1)[masked]
            profile_loss = (
                -(hist_m * functional.log_softmax(logits, 1)).sum(1).mean()
            )
        total = (
            self.count_weight * count_loss + self.profile_weight * profile_loss
        )
        intensity_loss = pred.sum() * 0.0
        if "target_ihist" in output:
            ilogits = pred[:, 1 + bins :].permute(0, 2, 3, 1)[masked]
            ihist_m = output["target_ihist"].permute(0, 2, 3, 1)[masked]
            intensity_loss = (
                -(ihist_m * functional.log_softmax(ilogits, 1)).sum(1).mean()
            )
            total = total + self.intensity_weight * intensity_loss
        self.loss_dict = {
            "recon_count": float(count_loss.detach()),
            "recon_profile": float(profile_loss.detach()),
            "recon_intensity": float(intensity_loss.detach()),
            "masked_cells": float(masked.sum()) / max(pred.shape[0], 1),
            "total_loss": float(total.detach()),
        }
        return total

    def logging(self, *args: Any, **kwargs: Any) -> None:
        return None
