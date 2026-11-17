"""Loader hooks for the fusion rows.

The fusion investigation (notes, section 16) found two things in
HEAL's intermediate-fusion loader that a spec can now switch:

* ``fused_augment``: the loader never augments the LiDAR, while the
  single-agent loader flips, rotates and scales every sample. With
  6 282 training frames the fusion fine-tune overfits from epoch 0.
  :func:`install_fused_augment` draws one flip / rotation / scaling per
  sample (the ranges of the config's ``data_augment`` list) and applies
  it in the ego frame to every agent's cloud and to the ground truth,
  so the fused frame is augmented as one scene. Needs ``proj_first``:
  the clouds are then projected into the ego frame before pillars, and
  the pairwise transforms the fusion warps with stay the identity.
* ``union_labels``: each agent's yaml carries its own boxes, so the
  no-fusion rows are scored on the ego's labels and the fusion rows on
  the union (the fusion loader stacks every agent's boxes and keeps one
  per id). :func:`install_union_labels` gives every agent the union, so
  single and fused rows score against one ground truth. Eval only: a
  single agent trained on boxes it cannot see learns nothing from them.

Box rows follow HEAL's ``object_bbx_center``: x, y, z, h, w, l, yaw.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Augment = tuple[bool, float, float]  # flip over the x axis, yaw, scale


def augment_ranges(data_augment: list[dict[str, Any]]) -> dict[str, Any]:
    """The flip / rotation / scaling ranges of HEAL's ``data_augment``
    list (the single-agent recipe); a missing entry disables that
    part."""
    cfg: dict[str, Any] = {"flip": False, "rotate": None, "scale": None}
    for entry in data_augment:
        name = entry.get("NAME")
        if name == "random_world_flip":
            cfg["flip"] = "x" in entry.get("ALONG_AXIS_LIST", [])
        elif name == "random_world_rotation":
            lo, hi = entry["WORLD_ROT_ANGLE"]
            cfg["rotate"] = (float(lo), float(hi))
        elif name == "random_world_scaling":
            lo, hi = entry["WORLD_SCALE_RANGE"]
            cfg["scale"] = (float(lo), float(hi))
    return cfg


def draw_augment(cfg: dict[str, Any], u: Any) -> Augment:
    """One sample's transform from three uniform draws in [0, 1)
    (torch's per-worker generator, not numpy's shared state)."""
    flip = bool(cfg["flip"]) and float(u[0]) < 0.5
    angle = 0.0
    if cfg["rotate"]:
        lo, hi = cfg["rotate"]
        angle = lo + (hi - lo) * float(u[1])
    scale = 1.0
    if cfg["scale"]:
        lo, hi = cfg["scale"]
        scale = lo + (hi - lo) * float(u[2])
    return flip, angle, scale


def transform_points(points: Any, aug: Augment) -> Any:
    """``(N, 3 + C)`` points under flip (y -> -y), then yaw rotation
    about z, then scaling of x, y, z; the extra columns are kept."""
    flip, angle, scale = aug
    out = np.array(points, dtype=np.float64, copy=True)
    if flip:
        out[:, 1] = -out[:, 1]
    c, s = np.cos(angle), np.sin(angle)
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    out[:, :3] *= scale
    return out.astype(points.dtype, copy=False)


def transform_boxes(boxes: Any, aug: Augment) -> Any:
    """``(N, 7)`` HEAL boxes under the same transform as the points:
    a flip mirrors the centre and negates the yaw, the rotation turns
    centre and yaw, the scaling scales centre and extent."""
    flip, angle, scale = aug
    out = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 7)
    if flip:
        out[:, 1] = -out[:, 1]
        out[:, 6] = -out[:, 6]
    c, s = np.cos(angle), np.sin(angle)
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    out[:, 6] += angle
    out[:, :6] *= scale
    return out


def points_through_ego(points: Any, m: Any, aug: Augment) -> Any:
    """An agent's cloud, augmented in the ego frame and handed back in
    the agent's frame: ``m`` maps agent to ego, so the loader's own
    projection (``m`` again) lands on the augmented ego-frame cloud."""
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    hom = np.c_[xyz, np.ones(len(xyz))]
    ego = (np.asarray(m, dtype=np.float64) @ hom.T).T
    ego[:, :3] = transform_points(ego[:, :3], aug)
    back = (np.linalg.inv(np.asarray(m, dtype=np.float64)) @ ego.T).T
    out = np.array(points, copy=True)
    out[:, :3] = back[:, :3].astype(out.dtype)
    return out


def boxes_to_frame(boxes: Any, m: Any) -> Any:
    """``(N, 7)`` boxes carried by the rigid ``m`` (4 x 4): centres
    through the matrix, yaws by its heading."""
    out = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 7)
    if len(out) == 0:
        return out
    mm = np.asarray(m, dtype=np.float64)
    hom = np.c_[out[:, :3], np.ones(len(out))]
    out[:, :3] = (mm @ hom.T).T[:, :3]
    out[:, 6] += float(np.arctan2(mm[1, 0], mm[0, 0]))
    return out


def pad_boxes(boxes: Any, max_num: int) -> tuple[Any, Any]:
    """HEAL's padded ``(max_num, 7)`` centre array and its mask."""
    center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    n = min(len(boxes), max_num)
    center[:n] = np.asarray(boxes)[:n]
    mask[:n] = 1
    return center, mask


def merge_vehicles(data: Any) -> Any:
    """Every agent's ``params['vehicles']`` becomes the union over the
    agents of the sample; on a shared id the first agent (the ego, the
    loader lists it first) wins, as the fusion loader keeps the first
    box per id when it stacks the agents."""
    union: dict[Any, Any] = {}
    for cav in data.values():
        params = cav.get("params") if isinstance(cav, dict) else None
        if not isinstance(params, dict):
            continue
        for k, v in (params.get("vehicles") or {}).items():
            union.setdefault(k, v)
    for cav in data.values():
        params = cav.get("params") if isinstance(cav, dict) else None
        if isinstance(params, dict):
            params["vehicles"] = dict(union)
    return data


def install_union_labels() -> None:
    """Patch HEAL's OPV2V base loader (once) so every agent of a sample
    carries the union of the agents' labels."""
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset

    if getattr(opv2v_basedataset, "_v2b_union_labels", False):
        return
    base = opv2v_basedataset.OPV2VBaseDataset
    original = base.retrieve_base_data

    def retrieve_base_data(self: Any, idx: int) -> Any:
        return merge_vehicles(original(self, idx))

    base.retrieve_base_data = retrieve_base_data
    opv2v_basedataset._v2b_union_labels = True


def install_fused_augment() -> None:
    """Patch HEAL's intermediate heterogeneous dataset factory (once):
    in training, one transform per sample on every agent's cloud and
    on the ground truth; the ego-point mask moves into the hook so it
    runs on the unaugmented cloud of each agent."""
    import opencood.data_utils.datasets as datasets
    import torch
    from opencood.data_utils.datasets import (
        intermediate_heter_fusion_dataset as mod,
    )
    from opencood.utils import pcd_utils
    from opencood.utils.transformation_utils import x1_to_x2

    if getattr(mod, "_v2b_fused_augment", False):
        return
    original_factory = mod.getIntermediateheterFusionDataset
    real_mask = pcd_utils.mask_ego_points

    def factory(cls: Any) -> Any:
        made = original_factory(cls)
        original_getitem = made.__getitem__
        original_item = made.get_item_single_car

        class AugmentedIntermediateheterFusionDataset(made):  # type: ignore[misc,valid-type]
            def __init__(
                self: Any, params: Any, visualize: Any, train: bool = True
            ) -> None:
                super().__init__(params, visualize, train)
                if not self.proj_first:
                    raise ValueError(
                        "fused_augment needs fusion.args.proj_first: true"
                    )
                self.v2b_augment = augment_ranges(
                    list(params.get("data_augment") or [])
                )
                self._v2b_aug: Augment | None = None

            def __getitem__(self: Any, idx: int) -> Any:
                self._v2b_aug = (
                    draw_augment(self.v2b_augment, torch.rand(3).tolist())
                    if self.train
                    else None
                )
                try:
                    return original_getitem(self, idx)
                finally:
                    self._v2b_aug = None

            def get_item_single_car(
                self: Any, selected_cav_base: Any, ego_cav_base: Any
            ) -> Any:
                cav = dict(selected_cav_base)
                if "lidar_np" in cav:
                    cav["lidar_np"] = real_mask(cav["lidar_np"])
                aug = self._v2b_aug
                if aug is None or "lidar_np" not in cav:
                    return original_item(self, cav, ego_cav_base)
                m = x1_to_x2(
                    cav["params"]["lidar_pose"],
                    ego_cav_base["params"]["lidar_pose"],
                )
                cav["lidar_np"] = points_through_ego(cav["lidar_np"], m, aug)
                out = original_item(self, cav, ego_cav_base)
                out["object_bbx_center"] = transform_boxes(
                    out["object_bbx_center"], aug
                )
                if "single_object_bbx_center" in out:
                    # the agent's own supervision, in its own frame
                    own = boxes_to_frame(
                        out["object_bbx_center"], np.linalg.inv(m)
                    )
                    center, mask = pad_boxes(
                        own, int(self.params["postprocess"]["max_num"])
                    )
                    out["single_object_bbx_center"] = center
                    out["single_object_bbx_mask"] = mask
                    out["single_label_dict"] = (
                        self.post_processor.generate_label(
                            gt_box_center=center,
                            anchors=self.anchor_box,
                            mask=mask,
                        )
                    )
                return out

        AugmentedIntermediateheterFusionDataset.__name__ = made.__name__
        return AugmentedIntermediateheterFusionDataset

    # the hook masks the ego points itself, before the transform
    mod.mask_ego_points = lambda points: points
    mod.getIntermediateheterFusionDataset = factory
    # build_dataset resolves the factory by name in this namespace
    datasets.getIntermediateheterFusionDataset = factory
    mod._v2b_fused_augment = True
