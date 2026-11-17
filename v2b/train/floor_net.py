"""A learned PoissonFloor: the m1 trunk regresses the Poisson floor height.

The Poisson fit
(718 ms per sweep on one CPU thread) stays the offline teacher; a small
network predicts its height map from the raw pillars in milliseconds.

The model is the masked pretraining's shape (``MaskedPretrainModel``:
the ego trunk plus a 1 x 1 decoder on the trunk's BEV map) with a
different target and loss and no masking: one output channel, the floor
height ``h`` per cell of the trunk's map (0.8 m for the m1 pillars), L1
against the teacher on the occupied cells. The teacher's height reaches
the model as the fifth point channel of a split derived with
``v2b-derive-ground.py --mode hag --surface poisson`` (``hag`` = height
above ground = ``z - h``, read with ``point_channels: 5``): per cell the
target is the mean of ``z - hag`` over the cell's points. The ``hag``
channel is zeroed before the encoder sees the pillars, so the network
cannot read the answer off its input.

Spec::

    pretrain: {target: floor}
    point_channels: 5
    sources: {train: [{dataset: waymo-opv2v-hag-poisson, split: train}], ...}

The trainer's pretraining path carries it (no labels, validation loss
picks the epoch). At test time :func:`height_field` turns one sweep into
a :class:`v2b.data.ground.GroundField`, which ``v2b-derive-ground.py
--surface learned`` plugs into the unchanged ``rebase`` machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from v2b.data.ground import GroundField
from v2b.train.pretrain import MaskedPretrainModel

HAG_CHANNEL = 4


class FloorModel(MaskedPretrainModel):
    """The ego trunk with a one-channel floor head; forward takes HEAL's
    ``batch["ego"]`` and returns the predicted and the target height map
    with the occupancy mask."""

    def __init__(
        self,
        detector: Any,
        lidar_range: list[float],
        modality: str = "m1",
        voxel_size: list[float] | None = None,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__(
            detector,
            lidar_range,
            modality=modality,
            voxel_size=voxel_size,
            mask_ratio=0.0,
            bins=0,
            intensity_bins=0,
            decoder_layers=decoder_layers,
        )

    def _cells(
        self, feats: torch.Tensor, coords: torch.Tensor, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, int, int, int]:
        b = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        h, w = self._map_size(inputs)
        sx, sy = self.stride
        cy = (coords[:, 2].long() // sy).clamp(0, h - 1)
        cx = (coords[:, 3].long() // sx).clamp(0, w - 1)
        cell = (coords[:, 0].long() * h + cy) * w + cx
        return cell, b, h, w

    def predict(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """The height map (B, h, w) of pillars whose ``hag`` channel (if
        any) is zeroed first."""
        feats = inputs["voxel_features"]
        if feats.shape[-1] > HAG_CHANNEL:
            feats = feats.clone()
            feats[..., HAG_CHANNEL] = 0.0
        kept = dict(inputs)
        kept["voxel_features"] = feats
        fused = self.trunk(kept)
        pred: torch.Tensor = self.decoder(fused)[:, 0]
        return pred

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = batch[f"inputs_{self.modality}"]
        feats = inputs["voxel_features"]  # (P, N, F), F = 5 with hag
        coords = inputs["voxel_coords"]  # (P, 4): batch, z, y, x
        nums = inputs["voxel_num_points"]  # (P,)
        if feats.shape[-1] <= HAG_CHANNEL:
            raise ValueError(
                "the floor target needs the hag point channel: derive the "
                "split with --mode hag and set point_channels: 5"
            )
        cell, b, h, w = self._cells(feats, coords, inputs)
        n_cells = b * h * w
        device = feats.device
        valid = torch.arange(feats.shape[1], device=device)[
            None, :
        ] < nums.view(-1, 1)
        # the teacher's floor height at every point: z - hag
        height = feats[..., 2] - feats[..., HAG_CHANNEL]
        total = torch.zeros(n_cells, device=device)
        total.index_add_(0, cell, (height * valid).sum(1))
        count = torch.zeros(n_cells, device=device)
        count.index_add_(0, cell, nums.float())
        occupied = count > 0
        target = total / count.clamp(min=1.0)
        pred = self.predict(inputs)
        if pred.shape[0] != b or pred.shape[1] != h or pred.shape[2] != w:
            raise RuntimeError(
                f"trunk map {tuple(pred.shape)} does not match the cell "
                f"grid ({b}, {h}, {w}) at stride {self.stride}"
            )
        return {
            "floor": pred,
            "target_floor": target.view(b, h, w),
            "occupied": occupied.view(b, h, w),
        }

    def height_field(self, inputs: dict[str, torch.Tensor]) -> GroundField:
        """One sweep's predicted floor as a bilinear height field in the
        sensor frame (cell = the trunk's stride on the pillar grid)."""
        with torch.no_grad():
            pred = self.predict(inputs)
        h, w = int(pred.shape[1]), int(pred.shape[2])
        cell_x = (self.x1 - self.x0) / w
        cell_y = (self.y1 - self.y0) / h
        if abs(cell_x - cell_y) > 1e-6:
            raise RuntimeError(f"non-square floor cells {cell_x} x {cell_y}")
        height = pred[0].detach().cpu().numpy().astype(np.float64)
        return GroundField(self.x0, self.y0, float(cell_x), height)


class FloorLoss:
    """L1 between the predicted and the teacher's floor height on the
    occupied cells; the same call signature as HEAL's criterion."""

    def __init__(self) -> None:
        self.loss_dict: dict[str, float] = {}

    def __call__(
        self, output: dict[str, Any], target: Any = None, suffix: str = ""
    ) -> torch.Tensor:
        pred = output["floor"]
        occupied = output["occupied"]
        if int(occupied.sum()) == 0:
            zero = pred.sum() * 0.0
            self.loss_dict = {"floor_l1": 0.0, "total_loss": 0.0}
            return zero
        err = (pred[occupied] - output["target_floor"][occupied]).abs()
        loss = err.mean()
        self.loss_dict = {
            "floor_l1": float(loss.detach()),
            "floor_p95": float(torch.quantile(err.detach(), 0.95)),
            "occupied_cells": float(occupied.sum()) / max(pred.shape[0], 1),
            "total_loss": float(loss.detach()),
        }
        return loss

    def logging(self, *args: Any, **kwargs: Any) -> None:
        return None
