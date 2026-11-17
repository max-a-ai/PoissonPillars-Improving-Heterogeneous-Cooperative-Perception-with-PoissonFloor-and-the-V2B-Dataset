"""A HEAL checkpoint as a deterministic two-agent detector.

Runs inside ``third_party/HEAL/.venv`` (torch, spconv, the built ops);
the launcher sets ``V2B_FORK=heal`` before the trainer module is
imported. The detector is built the way the trainer builds it (the
same hooks: class-major anchors, pillar pooling, the single-agent
trunk inside the fusion model, the warm start), so a checkpoint
scores here exactly as it would under ``--eval-only``, minus the
sources of run-to-run drift of that path:

* HEAL's loaders call ``shuffle_points`` on every cloud, in loader
  workers with their own random state, and the voxel generator keeps
  the first ``max_points_per_voxel`` points of a pillar in the order it
  sees them: the pillar contents, and with them the scores, moved
  between runs. Here the clouds go in as stored.
* ``--ap-samples 300`` scores every ``len(val) // 300``-th frame, so
  the frame set changes with the size of the validation view.
* No cudnn flags were pinned and the score / NMS thresholds came from
  whichever base config a spec started from.

:func:`set_deterministic` pins the seeds and the cudnn / cuBLAS
algorithms; :class:`Detector` fixes the score and NMS thresholds per
call and never samples. Boxes come out as ``(M, 7)`` x y z l w h yaw
in the ego frame of the call.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from v2b.train.hypes import ExperimentSpec, render

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


def set_deterministic(seed: int = 0) -> dict[str, Any]:
    """Pin everything torch lets us pin; call before the first CUDA
    use (the cuBLAS workspace variable is read at context creation)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": "warn_only",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch": torch.__version__,
    }


def render_config(
    config: Path, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The spec's full hypes, rendered and written to ``out_dir`` then
    read back through the trainer's loader (which installs the data
    hooks the checkpoint needs); plus the raw spec document, for the
    ``coop`` block the spec class does not know."""
    from v2b.train.opencood_train import load_hypes

    doc: dict[str, Any] = yaml.safe_load(Path(config).read_text())
    spec = ExperimentSpec.from_yaml(config)
    hypes = render(spec, out_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    path.write_text(yaml.safe_dump(hypes, sort_keys=False))
    return load_hypes(str(path)), doc


def corners_to_boxes(corners: F64) -> F64:
    """``(M, 8, 3)`` HEAL corners to ``(M, 7)`` x y z l w h yaw."""
    if len(corners) == 0:
        return np.zeros((0, 7))
    centre = corners.mean(axis=1)
    length = np.linalg.norm(corners[:, 1, :2] - corners[:, 2, :2], axis=1)
    width = np.linalg.norm(corners[:, 1, :2] - corners[:, 0, :2], axis=1)
    height = np.abs(corners[:, 4, 2] - corners[:, 0, 2])
    heading = corners[:, 1, :2] - corners[:, 2, :2]
    yaw = np.arctan2(heading[:, 1], heading[:, 0])
    out: F64 = np.column_stack([centre, length, width, height, yaw])
    return out


class Detector:
    """One checkpoint, single or fused forward, fixed thresholds."""

    def __init__(
        self,
        hypes: dict[str, Any],
        init_from: Path,
        score_threshold: float,
        nms_threshold: float,
        max_num: int = 300,
    ) -> None:
        import torch
        from opencood.data_utils.post_processor import build_postprocessor
        from opencood.data_utils.pre_processor import build_preprocessor
        from opencood.tools import train_utils

        from v2b.train.opencood_train import class_anchors_of, warm_start

        self.torch = torch
        post_cfg = hypes["postprocess"]
        post_cfg["target_args"]["score_threshold"] = float(score_threshold)
        post_cfg["nms_thresh"] = float(nms_threshold)
        post_cfg["max_num"] = int(max_num)
        self.hypes = hypes
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = train_utils.create_model(hypes)
        self.notes: list[str] = []
        if hypes.get("v2b_bev_attention"):
            from v2b.train.bev_attention import install_bev_attention

            self.notes += install_bev_attention(
                model, hypes["v2b_bev_attention"]
            )
        if hypes.get("v2b_fusion_trunk") == "single":
            from v2b.train.fusion_trunk import (
                head_stride_factor,
                install_single_trunk,
            )

            self.notes.append(
                install_single_trunk(model, pool=head_stride_factor(hypes))
            )
        self.notes.append(warm_start(model, str(init_from)))
        self.model = model.to(self.device).eval()
        pre_cfg = hypes["heter"]["modality_setting"]["m1"]["preprocess"]
        self.pre = build_preprocessor(pre_cfg, train=False)
        self.post = build_postprocessor(post_cfg, train=False)
        self.anchors = torch.from_numpy(self.post.generate_anchor_box()).to(
            self.device
        )
        self.class_anchors = class_anchors_of(hypes) or {}
        self.lidar_range = [float(v) for v in pre_cfg["cav_lidar_range"]]
        self.max_cav = int(hypes["train_params"].get("max_cav", 5))
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

    def crop(self, pts: F32) -> F32:
        r = self.lidar_range
        keep = (
            (pts[:, 0] >= r[0])
            & (pts[:, 0] < r[3])
            & (pts[:, 1] >= r[1])
            & (pts[:, 1] < r[4])
            & (pts[:, 2] >= r[2])
            & (pts[:, 2] < r[5])
        )
        return np.ascontiguousarray(pts[keep])

    def _voxelise(self, clouds: list[F32]) -> dict[str, Any]:
        """Per-agent pillars stacked, column 0 of the coords = agent."""
        torch = self.torch
        feats, nums, coords = [], [], []
        for i, pts in enumerate(clouds):
            v = self.pre.preprocess(pts)
            feats.append(v["voxel_features"])
            nums.append(v["voxel_num_points"])
            coords.append(
                np.pad(v["voxel_coords"], ((0, 0), (1, 0)), constant_values=i)
            )
        return {
            "voxel_features": torch.from_numpy(np.concatenate(feats))
            .float()
            .to(self.device),
            "voxel_num_points": torch.from_numpy(np.concatenate(nums))
            .long()
            .to(self.device),
            "voxel_coords": torch.from_numpy(np.concatenate(coords))
            .long()
            .to(self.device),
        }

    def detect(
        self, clouds: list[F32], ego_from_agent: list[F64]
    ) -> tuple[F64, F64, list[str]]:
        """Boxes, scores and class names in the ego frame.

        ``clouds[i]`` is agent i's cloud in its own frame and
        ``ego_from_agent[i]`` its pose relative to agent 0, the ego.
        As in HEAL's loader each cloud loses the returns on the
        sensor's own vehicle, is projected into the ego frame
        (``proj_first``, the fusion rows' training setting: the fusion
        warps are then the identity) and cropped to the grid.
        """
        from opencood.utils.pcd_utils import mask_ego_points

        from v2b.train.heal_data import box_classes
        from v2b.train.opencood_train import footprint_lw

        torch = self.torch
        projected: list[F32] = []
        for pts, tf in zip(clouds, ego_from_agent, strict=True):
            own = mask_ego_points(np.asarray(pts, np.float32))
            xyz = own[:, :3] @ tf[:3, :3].T + tf[:3, 3]
            moved = np.concatenate(
                [xyz.astype(np.float32), own[:, 3:]], axis=1
            )
            projected.append(self.crop(moved))
        n = len(projected)
        eye = torch.eye(4, device=self.device)
        batch: dict[str, Any] = {
            "inputs_m1": self._voxelise(projected),
            "record_len": torch.tensor(
                [n], dtype=torch.long, device=self.device
            ),
            "pairwise_t_matrix": eye.view(1, 1, 1, 4, 4).repeat(
                1, self.max_cav, self.max_cav, 1, 1
            ),
            "agent_modality_list": ["m1"] * n,
            "anchor_box": self.anchors,
            "transformation_matrix": eye,
        }
        with torch.no_grad():
            output = self.model(batch)
            boxes, scores = self.post.post_process(
                {"ego": batch}, {"ego": output}
            )
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        corners = boxes.cpu().numpy().astype(np.float64)
        names = (
            box_classes(footprint_lw(boxes), self.class_anchors)
            if self.class_anchors
            else ["car"] * len(corners)
        )
        return (
            corners_to_boxes(corners),
            scores.cpu().numpy().astype(np.float64),
            [str(c) for c in names],
        )
