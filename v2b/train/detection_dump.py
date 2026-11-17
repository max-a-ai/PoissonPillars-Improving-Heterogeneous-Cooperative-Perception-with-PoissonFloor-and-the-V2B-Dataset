"""Per-frame detections and ground truth of an evaluation, as one npz.

The AP numbers of ``eval.json`` say how much a row gained; they do not
say where. ``--dump-detections`` on an eval-only run keeps every
decoded box and every labelled box of every scored frame, with the
frame's identity and a few attributes per labelled box, so a local
analysis (``v2b-attn-gain-strata.py``) can split a gain by range,
by points on the person, by the terrain under it, or by error type.

Arrays in the file (``F`` frames, ``M`` detections, ``G`` labels):

==================  =====================================================
``classes``         ``(C,)`` class names; ``*_class`` index into it
``frame_index``     ``(F,)`` index into the evaluation dataset
``frame_scenario``  ``(F,)`` scenario folder name (view-prefixed)
``frame_timestamp`` ``(F,)`` timestamp key, e.g. ``000004``
``frame_points``    ``(F,)`` voxelised points the model saw
``det_frame``       ``(M,)`` row into the frame arrays
``det_corners``     ``(M, 8, 3)`` box corners in the ego frame
``det_score``       ``(M,)``
``det_class``       ``(M,)`` nearest anchor size, -1 without anchors
``gt_frame``        ``(G,)``
``gt_corners``      ``(G, 8, 3)``
``gt_class``        ``(G,)``
``gt_points``       ``(G,)`` voxelised points inside the box
==================  =====================================================

Corners follow HEAL's order (``boxes_to_corners_3d``): 0 to 3 the
bottom face, 4 to 7 the top face above them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

F32 = npt.NDArray[np.float32]


def frame_identity(dataset: Any, index: int) -> tuple[str, str]:
    """``(scenario folder, timestamp key)`` of a HEAL dataset index,
    following ``retrieve_base_data``."""
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if index < ele:
            scenario_index = i
            break
    database = dataset.scenario_database[scenario_index]
    offset = (
        0 if scenario_index == 0 else dataset.len_record[scenario_index - 1]
    )
    key = str(dataset.return_timestamp_key(database, index - offset))
    folder = Path(dataset.scenario_folders[scenario_index]).name
    return folder, key


def valid_points(inputs: dict[str, Any]) -> F32:
    """``(N, 3)`` x y z of the voxelised points the model saw."""
    feats = inputs["voxel_features"]
    num = inputs["voxel_num_points"].long()
    n_max = feats.shape[1]
    valid = torch.arange(n_max, device=feats.device)[None, :] < num[:, None]
    return np.asarray(feats[..., :3][valid].cpu().numpy(), np.float32)


def points_in_boxes(points: F32, corners: F32) -> npt.NDArray[np.int32]:
    """Points inside each ``(G, 8, 3)`` corner box, a bounding-square
    prefilter before HEAL's exact rotated test."""
    from opencood.utils.box_utils import get_points_in_rotated_box_3d

    out = np.zeros(len(corners), np.int32)
    if len(points) == 0:
        return out
    for i, box in enumerate(corners):
        lo = box.min(0)
        hi = box.max(0)
        near = np.all((points >= lo) & (points <= hi), axis=1)
        if not near.any():
            continue
        out[i] = int(len(get_points_in_rotated_box_3d(points[near], box)))
    return out


class DetectionDump:
    """Collects frames during :func:`bev_ap`; :meth:`save` writes them."""

    def __init__(self, dataset: Any, classes: list[str]) -> None:
        self.dataset = dataset
        self.classes = list(classes)
        self.frames: list[tuple[int, str, str, int]] = []
        self.det: list[tuple[int, F32, F32, npt.NDArray[np.int8]]] = []
        self.gt: list[
            tuple[int, F32, npt.NDArray[np.int8], npt.NDArray[np.int32]]
        ] = []

    def class_index(self, names: list[str] | np.ndarray | None, n: int) -> Any:
        if names is None or n == 0:
            return np.full(n, -1, np.int8)
        return np.asarray(
            [
                self.classes.index(str(c)) if c in self.classes else -1
                for c in names
            ],
            np.int8,
        )

    def add(
        self,
        index: int,
        inputs: dict[str, Any],
        boxes: torch.Tensor | None,
        scores: torch.Tensor | None,
        gt: torch.Tensor | None,
        det_names: list[str] | np.ndarray | None,
        gt_names: list[str] | np.ndarray | None,
    ) -> None:
        row = len(self.frames)
        points = valid_points(inputs)
        folder, key = frame_identity(self.dataset, index)
        self.frames.append((index, folder, key, len(points)))
        if boxes is not None and len(boxes):
            self.det.append(
                (
                    row,
                    np.asarray(boxes.detach().cpu().numpy(), np.float32),
                    np.asarray(scores.detach().cpu().numpy(), np.float32)
                    if scores is not None
                    else np.zeros(len(boxes), np.float32),
                    self.class_index(det_names, len(boxes)),
                )
            )
        if gt is not None and len(gt):
            corners = np.asarray(gt.detach().cpu().numpy(), np.float32)
            self.gt.append(
                (
                    row,
                    corners,
                    self.class_index(gt_names, len(gt)),
                    points_in_boxes(points, corners),
                )
            )

    def save(self, path: Path) -> Path:
        def cat(parts: list[Any], empty: Any) -> Any:
            return np.concatenate(parts) if parts else empty

        det_rows = [np.full(len(c), r, np.int32) for r, c, _, _ in self.det]
        gt_rows = [np.full(len(c), r, np.int32) for r, c, _, _ in self.gt]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            classes=np.asarray(self.classes),
            frame_index=np.asarray([f[0] for f in self.frames], np.int32),
            frame_scenario=np.asarray([f[1] for f in self.frames]),
            frame_timestamp=np.asarray([f[2] for f in self.frames]),
            frame_points=np.asarray([f[3] for f in self.frames], np.int32),
            det_frame=cat(det_rows, np.zeros(0, np.int32)),
            det_corners=cat(
                [d[1] for d in self.det], np.zeros((0, 8, 3), np.float32)
            ),
            det_score=cat([d[2] for d in self.det], np.zeros(0, np.float32)),
            det_class=cat([d[3] for d in self.det], np.zeros(0, np.int8)),
            gt_frame=cat(gt_rows, np.zeros(0, np.int32)),
            gt_corners=cat(
                [g[1] for g in self.gt], np.zeros((0, 8, 3), np.float32)
            ),
            gt_class=cat([g[2] for g in self.gt], np.zeros(0, np.int8)),
            gt_points=cat([g[3] for g in self.gt], np.zeros(0, np.int32)),
        )
        return path
