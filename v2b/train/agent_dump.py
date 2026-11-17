"""Per-agent detections of a late-fusion evaluation, as one npz.

``--dump-detections`` keeps the ego's merged result; box-level
fusion experiments need what each agent would
*send*: its own decoded boxes in its own frame, before HEAL's late
``post_process`` projects them into the ego frame and merges them.
``--dump-agent-detections`` on a late-fusion eval-only run keeps, per
scored frame and per agent, the decoded boxes above the score
threshold (step 1 of ``VoxelPostprocessor.post_process``, before the
size, z and NMS steps, so an offline merge can reproduce the row
exactly), the agent's transformation matrix into the ego frame, and
the frame's (union) labels in the ego frame.

Arrays in the file (``F`` frames, ``A`` agents, ``M`` boxes, ``G``
labels):

==================  =====================================================
``classes``         ``(C,)`` class names; ``*_class`` index into it
``order``           HEAL's box order of ``det_box`` (``hwl`` or ``lwh``)
``score_threshold`` the decoder's score threshold
``nms_thresh``      the NMS threshold of the row
``gt_range``        ``(6,)`` the range mask of the row
``frame_index``     ``(F,)`` index into the evaluation dataset
``frame_scenario``  ``(F,)`` scenario folder name (view-prefixed)
``frame_timestamp`` ``(F,)`` timestamp key, e.g. ``000004``
``agent_frame``     ``(A,)`` row into the frame arrays
``agent_key``       ``(A,)`` the agent's key in the batch (``ego`` or id)
``agent_cav``       ``(A,)`` the agent's folder id in the scenario
``agent_matrix``    ``(A, 4, 4)`` agent frame to ego frame
``agent_points``    ``(A,)`` voxelised points the model saw
``det_agent``       ``(M,)`` row into the agent arrays
``det_box``         ``(M, 7)`` x y z + sizes in ``order`` + yaw, own frame
``det_corners``     ``(M, 8, 3)`` the same boxes as corners, own frame
``det_score``       ``(M,)``
``det_class``       ``(M,)`` nearest anchor size, -1 without anchors
``gt_frame``        ``(G,)``
``gt_corners``      ``(G, 8, 3)`` ego frame
``gt_class``        ``(G,)``
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

from v2b.train.detection_dump import frame_identity, valid_points

F32 = npt.NDArray[np.float32]


def decode_agent(
    post: Any, cav_content: dict[str, Any], output: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step 1 of ``VoxelPostprocessor.post_process`` for one agent.

    Returns the decoded boxes ``(N, 7)`` in ``post.params['order']``
    and their scores ``(N,)`` in the agent's own frame, above the
    score threshold, with the direction classifier applied: what
    ``post_process`` projects and merges.
    """
    from opencood.utils.common_utils import limit_period

    cls = output["psm"] if "psm" in output else output["cls_preds"]
    prob = torch.sigmoid(cls.permute(0, 2, 3, 1)).reshape(1, -1)
    reg = output["reg_preds"]
    if reg.dim() == 4:
        boxes = post.delta_to_boxes3d(reg, cav_content["anchor_box"])
    else:
        boxes = reg.view(1, -1, 7)
    mask = torch.gt(prob, post.params["target_args"]["score_threshold"])
    mask = mask.view(1, -1)
    boxes3d = torch.masked_select(
        boxes[0], mask.unsqueeze(2).repeat(1, 1, 7)[0]
    ).view(-1, 7)
    scores = torch.masked_select(prob[0], mask[0])
    if "dir_preds" in output and len(boxes3d):
        dir_offset = post.params["dir_args"]["dir_offset"]
        num_bins = post.params["dir_args"]["num_bins"]
        dm = output["dir_preds"]
        dir_cls = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
        dir_cls = dir_cls[mask]
        dir_labels = torch.max(dir_cls, dim=-1)[1]
        period = 2 * np.pi / num_bins
        dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
        boxes3d[..., 6] = (
            dir_rot + dir_offset + period * dir_labels.to(dir_cls.dtype)
        )
        boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)
    if "iou_preds" in output and len(boxes3d):
        iou = torch.sigmoid(
            output["iou_preds"].permute(0, 2, 3, 1).contiguous()
        ).reshape(1, -1)
        iou = (torch.clamp(iou, min=0.0, max=1.0) + 1) * 0.5
        scores = scores * torch.pow(iou.masked_select(mask), 4)
    return boxes3d, scores


def ego_cav_id(dataset: Any, index: int) -> str:
    """The folder id of the ego agent of a HEAL dataset index."""
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if index < ele:
            scenario_index = i
            break
    database = dataset.scenario_database[scenario_index]
    for cav_id, content in database.items():
        if content.get("ego"):
            return str(cav_id)
    return str(next(iter(database)))


class AgentDump:
    """Collects frames during the late branch of ``bev_ap``;
    :meth:`save` writes them."""

    def __init__(self, dataset: Any, classes: list[str], post: Any) -> None:
        self.dataset = dataset
        self.classes = list(classes)
        self.post = post
        self.frames: list[tuple[int, str, str]] = []
        self.agents: list[tuple[int, str, str, F32, int]] = []
        self.det: list[tuple[int, F32, F32, F32, npt.NDArray[np.int8]]] = []
        self.gt: list[tuple[int, F32, npt.NDArray[np.int8]]] = []

    def class_index(self, names: list[str] | None, n: int) -> Any:
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
        batch: dict[str, Any],
        outputs: dict[str, Any],
        gt: torch.Tensor | None,
        class_anchors: dict[str, list[float]] | None,
    ) -> None:
        from opencood.utils import box_utils

        from v2b.train.opencood_train import _names

        row = len(self.frames)
        folder, key = frame_identity(self.dataset, index)
        self.frames.append((index, folder, key))
        ego_id = ego_cav_id(self.dataset, index)
        for cav, content in batch.items():
            boxes, scores = decode_agent(self.post, content, outputs[cav])
            arow = len(self.agents)
            inputs = next(
                v for k, v in content.items() if k.startswith("inputs_")
            )
            matrix = np.asarray(
                content["transformation_matrix"].detach().cpu().numpy(),
                np.float32,
            ).reshape(4, 4)
            self.agents.append(
                (
                    row,
                    str(cav),
                    ego_id if cav == "ego" else str(cav),
                    matrix,
                    len(valid_points(inputs)),
                )
            )
            if len(boxes) == 0:
                continue
            corners = box_utils.boxes_to_corners_3d(
                boxes, order=self.post.params["order"]
            )
            self.det.append(
                (
                    arow,
                    np.asarray(boxes.detach().cpu().numpy(), np.float32),
                    np.asarray(corners.detach().cpu().numpy(), np.float32),
                    np.asarray(scores.detach().cpu().numpy(), np.float32),
                    self.class_index(
                        _names(corners, class_anchors), len(boxes)
                    ),
                )
            )
        if gt is not None and len(gt):
            self.gt.append(
                (
                    row,
                    np.asarray(gt.detach().cpu().numpy(), np.float32),
                    self.class_index(_names(gt, class_anchors), len(gt)),
                )
            )

    def save(self, path: Path) -> Path:
        def cat(parts: list[Any], empty: Any) -> Any:
            return np.concatenate(parts) if parts else empty

        det_rows = [np.full(len(d[1]), d[0], np.int32) for d in self.det]
        gt_rows = [np.full(len(g[1]), g[0], np.int32) for g in self.gt]
        params = self.post.params
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            classes=np.asarray(self.classes),
            order=np.asarray(str(params["order"])),
            score_threshold=np.asarray(
                float(params["target_args"]["score_threshold"])
            ),
            nms_thresh=np.asarray(float(params["nms_thresh"])),
            gt_range=np.asarray(params["gt_range"], np.float32),
            frame_index=np.asarray([f[0] for f in self.frames], np.int32),
            frame_scenario=np.asarray([f[1] for f in self.frames]),
            frame_timestamp=np.asarray([f[2] for f in self.frames]),
            agent_frame=np.asarray([a[0] for a in self.agents], np.int32),
            agent_key=np.asarray([a[1] for a in self.agents]),
            agent_cav=np.asarray([a[2] for a in self.agents]),
            agent_matrix=np.asarray(
                [a[3] for a in self.agents], np.float32
            ).reshape(-1, 4, 4),
            agent_points=np.asarray([a[4] for a in self.agents], np.int32),
            det_agent=cat(det_rows, np.zeros(0, np.int32)),
            det_box=cat(
                [d[1] for d in self.det], np.zeros((0, 7), np.float32)
            ),
            det_corners=cat(
                [d[2] for d in self.det], np.zeros((0, 8, 3), np.float32)
            ),
            det_score=cat([d[3] for d in self.det], np.zeros(0, np.float32)),
            det_class=cat([d[4] for d in self.det], np.zeros(0, np.int8)),
            gt_frame=cat(gt_rows, np.zeros(0, np.int32)),
            gt_corners=cat(
                [g[1] for g in self.gt], np.zeros((0, 8, 3), np.float32)
            ),
            gt_class=cat([g[2] for g in self.gt], np.zeros(0, np.int8)),
        )
        return path
