"""A floor-consistency term on the detection heads.

A BEV detector that sees a person's head and shoulders can place the
box in the air; the ground truth never does, since labels are full
boxes, but a positive anchor with a label is the only place the
regression loss ever corrects that. The grounding loss acts on every
anchor the classifier is confident about, labelled or not: the
decoded box bottom (centre z minus half the height) must sit on the
floor. On a re-based split the floor is the constant
``-ORIGIN_ABOVE_GROUND`` everywhere, so the term needs no label and no
per-cell floor, which also makes it usable for self-supervised
fine-tuning on sweeps without boxes.

HEAL's regression target per anchor is ``(z - z_a) / h_a`` and
``log(h / h_a)`` (``VoxelPostprocessor.generate_label``, order hwl),
with ``z_a = -1.0`` for every anchor and ``h_a`` the class anchor
height; the anchors of a cell are class-major, yaw-minor, which is
how :func:`slot_table` lays out ``h_a`` and the class mask.
"""

from __future__ import annotations

from typing import Any

ANCHOR_CZ = -1.0


def slot_table(
    classes: list[str],
    anchors: dict[str, list[float]],
    n_yaws: int,
    pulled: list[str],
) -> tuple[list[float], list[bool]]:
    """Per anchor slot: the anchor height and whether the class is pulled."""
    heights: list[float] = []
    mask: list[bool] = []
    for c in classes:
        for _ in range(n_yaws):
            heights.append(float(anchors[c][2]))
            mask.append(c in pulled)
    return heights, mask


def box_bottoms(reg: Any, heights: Any) -> Any:
    """Decoded box bottom per anchor from ``(..., 7)`` deltas (hwl order).

    ``heights`` broadcasts over the anchor axis: ``(A,)`` against
    ``(B, H*W*A, 7)`` arranged with the anchor index last.
    """
    z = reg[..., 2] * heights + ANCHOR_CZ
    h = heights * reg[..., 3].clamp(max=4.0).exp()
    return z - h / 2


class GroundedLoss:
    """HEAL's criterion plus the floor term; same call signature."""

    def __init__(
        self,
        criterion: Any,
        heights: list[float],
        pulled: list[bool],
        floor_z: float,
        tau: float = 0.3,
        weight: float = 1.0,
    ) -> None:
        import torch

        self.criterion = criterion
        self.heights = torch.tensor(heights)
        self.pulled = torch.tensor(pulled)
        self.floor_z = float(floor_z)
        self.tau = float(tau)
        self.weight = float(weight)
        self.loss_dict: dict[str, float] = {}

    def __call__(
        self, output: dict[str, Any], target: dict[str, Any], suffix: str = ""
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        base = self.criterion(output, target, suffix)
        cls = output[f"cls_preds{suffix}"]
        reg = output[f"reg_preds{suffix}"]
        b = cls.shape[0]
        n_slots = len(self.heights)
        conf = torch.sigmoid(cls.permute(0, 2, 3, 1).reshape(b, -1))
        deltas = reg.permute(0, 2, 3, 1).reshape(b, -1, 7)
        heights = self.heights.to(reg.device)
        slot = torch.arange(deltas.shape[1], device=reg.device) % n_slots
        bottoms = box_bottoms(deltas, heights[slot])
        select = (conf.detach() >= self.tau) & self.pulled.to(reg.device)[slot]
        if select.any():
            err = bottoms[select] - self.floor_z
            ground = functional.smooth_l1_loss(err, torch.zeros_like(err))
        else:
            ground = bottoms.sum() * 0.0
        total = base + self.weight * ground
        self.loss_dict = dict(getattr(self.criterion, "loss_dict", {}))
        self.loss_dict["ground_loss"] = float(ground.detach())
        self.loss_dict["ground_anchors"] = float(select.sum())
        self.loss_dict["total_loss"] = float(total.detach())
        return total

    def logging(self, *args: Any, **kwargs: Any) -> Any:
        return self.criterion.logging(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.criterion, name)
