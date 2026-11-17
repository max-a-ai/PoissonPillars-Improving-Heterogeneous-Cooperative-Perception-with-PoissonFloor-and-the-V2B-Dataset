"""An evaluation-time filter on detected boxes from the returns inside.

The false pedestrians of the Waymo detectors are not in empty space:
on the frames of notebook 015 every one of them sits on real returns,
dark vertical things (poles, trunks, bushes, facade edges) whose
median intensity is half a person's (0.09 against 0.22, the clothing
the reflectivity analysis found bright). A box whose returns are too
dark is therefore dropped, per class, before the AP is scored:
``box_filter: {intensity_min: 0.10, classes: [pedestrian], z_min: -1.7,
margin: 0.2}`` in the spec. The returns come from the batch's own
pillars (at most 32 per pillar, the detector's view of the sweep),
those above ``z_min`` (the floor of a re-based sweep is -1.9 m) inside
the box footprint widened by ``margin``. The walls and the ground do
not enter: a visibility or shadow test cannot separate these boxes,
their intensity can.
"""

from __future__ import annotations

from typing import Any

import torch

from v2b.train.heal_data import box_classes


def valid_points(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """(M, F) points of a batch's pillars, padding removed."""
    feats = inputs["voxel_features"]
    nums = inputs["voxel_num_points"]
    valid = torch.arange(feats.shape[1], device=feats.device)[
        None, :
    ] < nums.view(-1, 1)
    return feats[valid]


def box_intensity(
    points: torch.Tensor,
    corners: torch.Tensor,
    margin: float = 0.2,
    z_min: float = -1.7,
) -> torch.Tensor:
    """Median intensity of the points in each box's BEV footprint
    (bottom four corners of ``corners`` (B, 8, 3)), above ``z_min``;
    0 for a box without points."""
    if len(corners) == 0:
        return torch.zeros(0, device=points.device)
    c = corners[:, :4, :2]  # (B, 4, 2)
    centre = c.mean(1)  # (B, 2)
    e = c[:, 1] - c[:, 0]
    yaw = torch.atan2(e[:, 1], e[:, 0])
    length = e.norm(dim=1)
    width = (c[:, 3] - c[:, 0]).norm(dim=1)
    above = points[points[:, 2] > z_min]
    out = torch.zeros(len(corners), device=points.device)
    if len(above) == 0:
        return out
    xy = above[:, :2]
    for b in range(len(corners)):
        d = xy - centre[b]
        cos, sin = torch.cos(yaw[b]), torch.sin(yaw[b])
        lx = cos * d[:, 0] + sin * d[:, 1]
        ly = -sin * d[:, 0] + cos * d[:, 1]
        inside = (lx.abs() <= length[b] / 2 + margin) & (
            ly.abs() <= width[b] / 2 + margin
        )
        if bool(inside.any()):
            out[b] = above[inside, 3].median()
    return out


def keep_boxes(
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    inputs: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    class_anchors: dict[str, list[float]] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """The boxes that pass the filter, and how many were dropped."""
    if boxes is None or scores is None or len(boxes) == 0:
        return boxes, scores, 0
    from v2b.train.opencood_train import footprint_lw

    classes = [str(c) for c in cfg.get("classes", ["pedestrian"])]
    if class_anchors:
        cls = box_classes(footprint_lw(boxes), class_anchors)
        tested = torch.tensor([c in classes for c in cls], device=boxes.device)
    else:
        tested = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
    inten = box_intensity(
        valid_points(inputs),
        boxes,
        margin=float(cfg.get("margin", 0.2)),
        z_min=float(cfg.get("z_min", -1.7)),
    )
    dark = inten < float(cfg.get("intensity_min", 0.10))
    keep = ~(tested & dark)
    return boxes[keep], scores[keep], int((~keep).sum())
