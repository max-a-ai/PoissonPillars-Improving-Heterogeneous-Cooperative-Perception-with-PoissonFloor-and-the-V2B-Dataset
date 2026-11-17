"""LitePT as the m1 LiDAR encoder of the HEAL fork.

LitePT (Yue et al., CVPR 2026, arXiv 2512.13689) is a sparse 3-D point
backbone: convolution in the early, high-resolution stages, attention
with a rotary position encoding in the late, low-resolution ones.
This module puts it where HEAL's pillar net sits, so everything after
the encoder (the BEV backbone, neck, heads, the fusions, the masked
pretraining and the warm start) stays as it is:

* the points of HEAL's pillar tensor (``inputs_m1``: up to 32 points
  per 0.4 m pillar, the same cap the pillar net sees) are gathered and
  grid-sampled at ``grid`` m (one point per occupied cell);
* LitePT-S encodes them and returns one feature per kept cell at the
  finest level;
* the cell features are max-pooled into their 0.4 m pillar, projected
  to the pillar net's 64 channels, and scattered to the BEV map with
  HEAL's own scatter, so ``encoder_m1.scatter.nx / ny`` exist for the
  pretraining and the ``rope3d`` hook.

The LitePT grid must divide the pillar: both grids start at the lower
range corner, so every ``grid`` cell lies inside exactly one pillar.
Spec key (``configs/*.yaml``, the HEAL fork only)::

    litept:
      grid: 0.1                      # m, LitePT's cell (default)
      pooling: max                   # cells into the pillar: max | attn
      weights: outputs/checkpoints/litept-s-nuscenes-semseg.pth  # optional
      model: {drop_path: 0.1}        # LitePT(...) overrides, optional

``pooling: attn`` is the attention pooling on the cell level: the
pillar's max feature next to an attention-weighted sum of its cells
(softmax of a linear score per cell), mixed by a linear layer that
starts as the identity on the max half, so the model computes exactly
the max-pooled map at start and learns what to make of the extra.

``core_method: lite_pt_encoder`` is what the spec renders; the class is
registered on ``opencood.models.heter_encoders`` by :func:`register`.
Needs ``third_party/LitePT`` and, in the HEAL venv, ``spconv``,
``flash_attn`` and ``torch_scatter`` (prebuilt wheels, see the notes).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
LITEPT_DIR = REPO / "third_party" / "LitePT"

#: channels LitePT-S returns at its finest level (``dec_channels[0]``)
LITEPT_OUT = 72
#: HEAL's pillar-net width, what the BEV backbone expects
PILLAR_OUT = 64


def _litept_class() -> type[nn.Module]:
    """The standalone ``LitePT`` class from ``third_party/LitePT``."""
    if str(LITEPT_DIR) not in sys.path:
        sys.path.insert(0, str(LITEPT_DIR))
    module = importlib.import_module("litept.model")
    cls: type[nn.Module] = module.LitePT
    return cls


def grid_sample(
    coord: torch.Tensor,
    batch: torch.Tensor,
    origin: torch.Tensor,
    grid: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One point per occupied ``grid`` cell: ``(kept, grid_coord)``.

    ``kept`` indexes the first point of every cell in the input order
    (the inverse map is not needed, the pooling goes through the
    pillar index). ``grid_coord`` is the cell of each kept point,
    counted from ``origin``.
    """
    gc = torch.div(coord - origin, grid, rounding_mode="floor").long()
    span = gc.max(0).values + 1
    key = ((batch * span[0] + gc[:, 0]) * span[1] + gc[:, 1]) * span[2] + gc[
        :, 2
    ]
    _, inverse = torch.unique(key, return_inverse=True)
    n_cells = int(inverse.max().item()) + 1
    first = torch.full(
        (n_cells,), coord.shape[0], dtype=torch.long, device=coord.device
    )
    first.scatter_reduce_(
        0,
        inverse,
        torch.arange(coord.shape[0], device=coord.device),
        "amin",
        include_self=True,
    )
    return first, gc[first].int()


class LitePTEncoder(nn.Module):  # type: ignore[misc]
    """HEAL encoder: ``forward(data_dict, modality) -> (B, 64, ny, nx)``."""

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        enc = importlib.import_module("opencood.models.heter_encoders")
        rng = np.asarray(args["lidar_range"], dtype=np.float64)
        voxel = np.asarray(args["voxel_size"], dtype=np.float64)
        grid_size = np.round((rng[3:6] - rng[0:3]) / voxel).astype(np.int64)
        cfg = dict(args.get("litept") or {})
        self.grid = float(cfg.get("grid", 0.1))
        for ax in (0, 1):
            ratio = voxel[ax] / self.grid
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(
                    f"litept.grid {self.grid} must divide the pillar "
                    f"{voxel[ax]} so every cell lies in one pillar"
                )
        self.register_buffer(
            "origin", torch.tensor(rng[0:3], dtype=torch.float32)
        )
        self.pooling = str(cfg.get("pooling", "max"))
        if self.pooling not in ("max", "attn"):
            raise ValueError(
                f"litept.pooling {self.pooling!r}, expected max or attn"
            )
        if self.pooling == "attn":
            self.attn = nn.Linear(LITEPT_OUT, 1)
            self.mix = nn.Linear(2 * LITEPT_OUT, LITEPT_OUT, bias=True)
            with torch.no_grad():
                self.attn.weight.normal_(0.0, 0.01)
                self.attn.bias.zero_()
                self.mix.weight.zero_()
                self.mix.weight[:, :LITEPT_OUT] = torch.eye(LITEPT_OUT)
                self.mix.bias.zero_()
        self.backbone = _litept_class()(
            in_channels=4, **dict(cfg.get("model") or {})
        )
        self.proj = nn.Sequential(
            nn.Linear(LITEPT_OUT, PILLAR_OUT, bias=False),
            nn.BatchNorm1d(PILLAR_OUT, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.scatter = enc.PointPillarScatter(
            {"num_features": PILLAR_OUT, "grid_size": grid_size}
        )
        weights = cfg.get("weights")
        if weights:
            path = Path(weights)
            if not path.is_absolute():
                path = REPO / path
            state = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(
                state, strict=False
            )
            if missing or unexpected:
                raise ValueError(
                    f"litept.weights {path}: {len(missing)} missing, "
                    f"{len(unexpected)} unexpected tensors"
                )
            self.weights_from = str(path)

    def forward(
        self, data_dict: dict[str, Any], modality: str
    ) -> torch.Tensor:
        inputs = data_dict[f"inputs_{modality}"]
        vf = inputs["voxel_features"]  # (P, max_pts, 4) zero-padded
        vc = inputs["voxel_coords"]  # (P, 4) as batch, z, y, x
        vn = inputs["voxel_num_points"]  # (P,)
        n_pillars, max_pts = vf.shape[0], vf.shape[1]
        valid = torch.arange(max_pts, device=vf.device)[None, :] < vn[:, None]
        pillar_of_point = torch.arange(n_pillars, device=vf.device)[
            :, None
        ].expand(-1, max_pts)[valid]
        points = vf[valid].float()  # (N, 4)
        coord = points[:, :3]
        batch = vc[pillar_of_point, 0].long()
        with torch.autocast("cuda", enabled=False):
            kept, grid_coord = grid_sample(
                coord, batch, self.origin, self.grid
            )
            point = {
                "coord": coord[kept],
                "grid_coord": grid_coord,
                "feat": points[kept],
                "batch": batch[kept],
                "grid_size": self.grid,
            }
            out = self.backbone(point)
            feat: torch.Tensor = out.feat.float()  # (cells, 72)
        pillar_of_cell = pillar_of_point[kept]
        index = pillar_of_cell[:, None].expand(-1, feat.shape[1])
        pooled = torch.full(
            (n_pillars, feat.shape[1]),
            -torch.inf,
            dtype=feat.dtype,
            device=feat.device,
        )
        pooled.scatter_reduce_(0, index, feat, "amax", include_self=True)
        pooled = torch.where(torch.isinf(pooled), 0.0, pooled)
        if self.pooling == "attn":
            pooled = self.mix(
                torch.cat(
                    [pooled, self._attend(feat, pillar_of_cell, n_pillars)],
                    dim=1,
                )
            )
        pillar_features = self.proj(pooled)
        batch_dict = {"pillar_features": pillar_features, "voxel_coords": vc}
        spatial: torch.Tensor = self.scatter(batch_dict)["spatial_features"]
        return spatial

    def _attend(
        self, feat: torch.Tensor, pillar: torch.Tensor, n_pillars: int
    ) -> torch.Tensor:
        """Attention-weighted sum of the cells of every pillar: softmax
        over a pillar's cells of a linear score, ``(P, C)``; a pillar
        without cells gives zeros."""
        score = self.attn(feat).squeeze(-1)
        top = torch.full((n_pillars,), -torch.inf, device=feat.device)
        top.scatter_reduce_(0, pillar, score, "amax", include_self=True)
        w = torch.exp(score - top[pillar])
        norm = torch.zeros(n_pillars, device=feat.device)
        norm.scatter_add_(0, pillar, w)
        w = w / norm[pillar].clamp(min=1e-12)
        out = torch.zeros((n_pillars, feat.shape[1]), device=feat.device)
        index = pillar[:, None].expand(-1, feat.shape[1])
        out.scatter_add_(0, index, w[:, None] * feat)
        return out


def register() -> type[nn.Module]:
    """Make ``core_method: lite_pt_encoder`` resolvable in the fork.

    Idempotent; returns the registered class.
    """
    enc = importlib.import_module("opencood.models.heter_encoders")
    cls = getattr(enc, "LitePTEncoder", None)
    if cls is None:
        cls = LitePTEncoder
        setattr(enc, "LitePTEncoder", cls)  # noqa: B010
    assert cls is not None
    return cls
