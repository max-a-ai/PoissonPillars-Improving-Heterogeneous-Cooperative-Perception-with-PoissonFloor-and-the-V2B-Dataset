"""What HEAL's camera loader needs that the mixed views do not give it.

Two things in HEAL's ``OPV2VBaseDataset`` assume the upstream data
layout:

* **Depth maps.** HEAL looks for ``<ts>_depth<i>.png`` next to the
  camera PNGs and then rewrites ``OPV2V`` to ``OPV2V_Hetero`` in that
  path. On the NAS the depth maps live in
  ``publicdatasets/opv2v-h/OPV2V_Hetero/<split>/<scenario>/<agent>/``
  while a training view links scenarios out of
  ``publicdatasets/opv2v/<split>/`` under a prefixed name.
  :func:`depth_path` resolves the camera file through the view's
  symlink and swaps the dataset directory, leaving the literal
  ``OPV2V`` that HEAL's own replace turns into ``OPV2V_Hetero``.
  :func:`install_depth_lookup` wires it into the loader.
* **Modality assignment.** HEAL indexes a JSON of
  ``{scenario: {cav_id: modality}}`` by scenario name with no
  fallback; a view's prefixed names would raise. :func:`write_assignment`
  writes one for a view that gives every agent the same modality,
  which is what single-modality training needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

OPV2V_DIR = "opv2v"
OPV2VH_DIR = "opv2v-h"
#: HEAL replaces this with ``OPV2V_Hetero`` after the lookup.
HEAL_PLACEHOLDER = "OPV2V"


def depth_path(camera_png: str | Path) -> Path:
    """OPV2V-H location of a camera (or depth) PNG's depth map.

    Resolves symlinks first so a view's scenario link ends up at the
    NAS copy, then swaps the last ``opv2v`` directory for
    ``opv2v-h/OPV2V``.
    """
    real = Path(os.path.realpath(camera_png))
    parts = list(real.parts)
    hits = [i for i, p in enumerate(parts) if p == OPV2V_DIR]
    if not hits:
        raise ValueError(f"no {OPV2V_DIR!r} directory in {real}")
    if any(HEAL_PLACEHOLDER in p for p in parts):
        raise ValueError(
            f"{real} already contains {HEAL_PLACEHOLDER!r}; HEAL's "
            "OPV2V -> OPV2V_Hetero rewrite would hit it twice"
        )
    i = hits[-1]
    parts[i : i + 1] = [OPV2VH_DIR, HEAL_PLACEHOLDER]
    return Path(*parts)


def install_depth_lookup() -> None:
    """Route HEAL's depth file lookup through :func:`depth_path`.

    Idempotent. Must run after ``opencood`` is importable.
    """
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset

    cls = opv2v_basedataset.OPV2VBaseDataset
    if getattr(cls, "_v2b_depth_lookup", False):
        return
    original = cls.find_camera_files  # a staticmethod upstream

    def find_camera_files(
        cav_path: str, timestamp: str, sensor: str = "camera"
    ) -> list[str]:
        files: list[str] = original(cav_path, timestamp, sensor)
        if sensor != "depth" or all(Path(f).exists() for f in files):
            return files  # derived data keeps its depth maps next door
        try:
            return [str(depth_path(f)) for f in files]
        except ValueError:
            # not an OPV2V tree (LiDAR-only derived data): HEAL only opens
            # these names when the config asks for depth, so keep them
            return files

    cls.find_camera_files = staticmethod(find_camera_files)
    cls._v2b_depth_lookup = True


def write_assignment(view_roots: list[Path], out: Path, modality: str) -> Path:
    """``{scenario: {cav_id: modality}}`` for every agent of the views."""
    table: dict[str, dict[str, str]] = {}
    for root in view_roots:
        for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
            agents = sorted(
                p.name
                for p in scenario.iterdir()
                if p.is_dir() and p.name[0] != "."
            )
            table[scenario.name] = dict.fromkeys(agents, modality)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    return out


def install_opencv_cameras() -> None:
    """Let HEAL read cameras exported with ``frame: opencv``.

    ``OPV2VBaseDataset.get_ext_int`` assumes CARLA cameras and multiplies
    the camera pose by a handedness flip. Frames written by
    ``v2b.data.opv2v_layout`` store OpenCV cameras (z forward) with the
    ``frame`` key; for those the flip is skipped, so ``camera_to_lidar``
    is exactly ``inv(lidar_pose) @ cords``. OPV2V's own files have no
    ``frame`` key and keep the original path. Idempotent.
    """
    import numpy as np
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset
    from opencood.utils.transformation_utils import x1_to_x2

    cls = opv2v_basedataset.OPV2VBaseDataset
    if getattr(cls, "_v2b_opencv_cameras", False):
        return
    original = cls.get_ext_int

    def get_ext_int(self: Any, params: Any, camera_id: int) -> Any:
        cam = params[f"camera{camera_id}"]
        if cam.get("frame") != "opencv":
            return original(self, params, camera_id)
        cords = np.array(cam["cords"], dtype=np.float32)
        camera_to_lidar = x1_to_x2(cords, params["lidar_pose_clean"]).astype(
            np.float32
        )
        intrinsic = np.array(cam["intrinsic"], dtype=np.float32)
        return camera_to_lidar, intrinsic

    cls.get_ext_int = get_ext_int
    cls._v2b_opencv_cameras = True


def install_class_filter(classes: tuple[str, ...] | list[str]) -> None:
    """Keep only boxes whose ``obj_type`` is in ``classes`` when HEAL loads
    a label yaml (boxes without the key, OPV2V's own, are always kept).

    HEAL's heads are single-class; until the multi-class head lands this
    is how a run says which of car / bicycle / pedestrian / other it
    trains on. Patches the ``load_yaml`` the base dataset module calls.
    """
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset

    keep = set(classes)
    original = opv2v_basedataset.load_yaml

    def load_yaml(file: Any, opt: Any = None) -> Any:
        doc = original(file, opt) if opt is not None else original(file)
        return filter_vehicles(doc, keep)

    opv2v_basedataset.load_yaml = load_yaml
    opv2v_basedataset._v2b_class_filter = sorted(keep)


def filter_vehicles(doc: Any, keep: set[str]) -> Any:
    """Drop ``vehicles`` entries whose ``obj_type`` is not in ``keep``.

    Entries without the key (OPV2V's own labels) stay. Anything that is
    not a label dict is returned untouched.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("vehicles"), dict):
        return doc
    doc["vehicles"] = {
        k: v
        for k, v in doc["vehicles"].items()
        if not isinstance(v, dict)
        or v.get("obj_type", None) is None
        or v["obj_type"] in keep
    }
    return doc


def install_range_filter(max_range: float) -> None:
    """Keep only boxes within ``max_range`` metres of every agent when
    HEAL loads a label yaml: the labelling contract's dual radius (50 m
    of the car and of the bike), applied at load time so the exported
    files keep every box and a pseudo-labelled scene teaches the same
    range as a hand-labelled one. Reads the ``dist_car`` / ``dist_bike``
    the V2B export writes per box; boxes without them are kept. Chains
    with :func:`install_class_filter` (each wraps the current
    ``load_yaml``).
    """
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset

    original = opv2v_basedataset.load_yaml

    def load_yaml(file: Any, opt: Any = None) -> Any:
        doc = original(file, opt) if opt is not None else original(file)
        return filter_range(doc, max_range)

    opv2v_basedataset.load_yaml = load_yaml
    opv2v_basedataset._v2b_max_range = float(max_range)


def filter_range(doc: Any, max_range: float) -> Any:
    """Drop ``vehicles`` entries farther than ``max_range`` from either
    agent (``dist_car`` / ``dist_bike``); entries without the keys and
    anything that is not a label dict stay untouched."""
    if not isinstance(doc, dict) or not isinstance(doc.get("vehicles"), dict):
        return doc
    keep = {}
    for k, v in doc["vehicles"].items():
        far = (
            isinstance(v, dict)
            and "dist_car" in v
            and "dist_bike" in v
            and max(float(v["dist_car"]), float(v["dist_bike"])) > max_range
        )
        if not far:
            keep[k] = v
    doc["vehicles"] = keep
    return doc


def install_multiclass_anchors() -> None:
    """Let HEAL's voxel postprocessor build one anchor set per class.

    With ``anchor_args.l/w/h`` as lists (see
    ``v2b.train.hypes.apply_anchor_classes``) the anchors become
    class-major: for class c and yaw index j the anchor sits at index
    ``c * len(r) + j``; ``anchor_args.z``, when present, is each
    class's centre height (HEAL's -1.0 otherwise), which the z target
    and the decoded z are relative to. Label assignment, loss and
    decoding are untouched: HEAL matches boxes to anchors by IoU, and a
    box of one class overlaps the other classes' anchors too little to
    count, so the class of a prediction is its anchor's. Scalar
    ``l/w/h`` keep the stock path. Idempotent.
    """
    import copy

    from opencood.data_utils.post_processor import voxel_postprocessor

    cls = voxel_postprocessor.VoxelPostprocessor
    if getattr(cls, "_v2b_multiclass", False):
        return
    original = cls.generate_anchor_box

    def generate_anchor_box(self: Any) -> Any:
        args = self.params["anchor_args"]
        if not isinstance(args["l"], list):
            return original(self)
        heights = list(args.get("z") or [-1.0] * len(args["l"]))
        sizes = list(
            zip(args["l"], args["w"], args["h"], heights, strict=True)
        )
        yaws = list(args["r"])
        if self.anchor_num != len(sizes) * len(yaws):
            raise ValueError(
                f"anchor_args.num {self.anchor_num} != classes "
                f"{len(sizes)} x yaws {len(yaws)}"
            )
        per_class = []
        for length, width, height, centre_z in sizes:
            one = copy.copy(self)
            one.params = copy.deepcopy(self.params)
            one.params["anchor_args"].update(
                {"l": length, "w": width, "h": height, "num": len(yaws)}
            )
            one.anchor_num = len(yaws)
            boxes = original(one)
            boxes[..., 2] = centre_z
            per_class.append(boxes)
        return np.concatenate(per_class, axis=2)

    cls.generate_anchor_box = generate_anchor_box
    cls._v2b_multiclass = True


def box_classes(dims_lw: Any, anchors: dict[str, list[float]]) -> list[str]:
    """Class of each box from its (length, width): the nearest anchor size.

    HEAL's decoder returns boxes without their anchor index; sizes
    separate the three v2b classes well (car ~4 x 1.6, bicycle ~1.8 x
    0.6, pedestrian ~0.8 x 0.8), so the nearest anchor in (l, w) is a
    faithful stand-in until the decoder carries the index itself.
    """
    names = list(anchors)
    ref = np.array([[anchors[c][0], anchors[c][1]] for c in names])
    d = np.asarray(dims_lw, dtype=float).reshape(-1, 2)
    if len(d) == 0:
        return []
    dist = np.abs(np.log(d[:, None, :] / ref[None, :, :])).sum(-1)
    return [names[i] for i in dist.argmin(1)]


def install_class_thresholds(
    thresholds: dict[str, list[float]],
    classes: list[str],
    n_yaws: int,
    anchor_sizes: dict[str, list[float]],
) -> None:
    """Per-class positive / negative IoU for HEAL's anchor assignment.

    HEAL matches every anchor to boxes with one global pair (0.6 /
    0.45, sized for cars); a 0.75 m pedestrian box rarely reaches 0.6
    against its anchor, so most pedestrian anchors never become
    positives. With class-major anchors (``install_multiclass_anchors``)
    the anchor at flat index ``i`` has slot ``i % (classes * yaws)`` and
    class ``slot // yaws``, so the thresholds become one column per
    anchor that ``generate_label``'s ``iou > thr`` broadcasts against.
    Classes not in ``thresholds`` keep the global pair. Idempotent.
    """
    from opencood.data_utils.post_processor import voxel_postprocessor

    cls = voxel_postprocessor.VoxelPostprocessor
    if getattr(cls, "_v2b_class_thresholds", False):
        return
    original = cls.generate_label

    def generate_label(self: Any, **kwargs: Any) -> Any:
        anchors = kwargs.get("anchors")
        targs = self.params["target_args"]
        if anchors is None or np.ndim(anchors) != 4:
            return original(self, **kwargs)
        h, w, slots, _ = anchors.shape
        pos = np.full(slots, float(targs["pos_threshold"]))
        neg = np.full(slots, float(targs["neg_threshold"]))
        slot_class = np.array(
            [
                classes[slot // n_yaws]
                if slot // n_yaws < len(classes)
                else ""
                for slot in range(slots)
            ]
        )
        for slot in range(slots):
            if slot_class[slot] in thresholds:
                pos[slot], neg[slot] = thresholds[slot_class[slot]]
        # a lower threshold must not let a box of another class claim
        # these anchors: zero the overlaps across classes, so a box only
        # matches the anchors of its own (size-assigned) class
        gt = np.asarray(kwargs["gt_box_center"])
        mask = np.asarray(kwargs["mask"]).astype(bool)
        order = self.params.get("order", "hwl")
        lw = gt[mask][:, [5, 4]] if order == "hwl" else gt[mask][:, [3, 4]]
        gt_class = np.array(box_classes(lw, anchor_sizes))
        anchor_class = np.tile(slot_class, h * w)
        same = anchor_class[:, None] == gt_class[None, :]

        def masked_overlaps(a: Any, b: Any) -> Any:
            iou = np.asarray(real_overlaps(a, b))
            if iou.shape == same.shape:
                iou = np.where(same, iou, 0.0)
            return iou

        real_overlaps = voxel_postprocessor.bbox_overlaps
        saved = (targs["pos_threshold"], targs["neg_threshold"])
        targs["pos_threshold"] = np.tile(pos, h * w)[:, None]
        targs["neg_threshold"] = np.tile(neg, h * w)[:, None]
        voxel_postprocessor.bbox_overlaps = masked_overlaps
        try:
            return original(self, **kwargs)
        finally:
            targs["pos_threshold"], targs["neg_threshold"] = saved
            voxel_postprocessor.bbox_overlaps = real_overlaps

    cls.generate_label = generate_label
    cls._v2b_class_thresholds = True


#: HEAL's fixed per-point channels: x y z intensity.
HEAL_POINT_CHANNELS = 4


def widen_point_columns(
    weight: Any, own_in: int, raw_channels: int = HEAL_POINT_CHANNELS
) -> Any:
    """A first-PFN weight trained on four point channels, fit to more.

    The pillar feature net's first linear sees ``[raw channels, cluster
    offsets (3), centre offsets (3), (distance)]`` per point, so extra
    raw channels sit right after the first four. The columns of the
    old weight keep their place, the new columns start at zero: the
    warm-started model computes exactly what it did before and learns
    what to make of the new channel. Torch and numpy tensors alike.
    """
    old_in = weight.shape[1]
    extra = own_in - old_in
    if extra <= 0:
        raise ValueError(f"nothing to widen: {old_in} -> {own_in}")
    if hasattr(weight, "new_zeros"):
        out = weight.new_zeros((weight.shape[0], own_in))
    else:
        out = np.zeros((weight.shape[0], own_in), dtype=weight.dtype)
    out[:, :raw_channels] = weight[:, :raw_channels]
    out[:, raw_channels + extra :] = weight[:, raw_channels:]
    return out


def install_point_channels(n: int) -> None:
    """``n`` float channels per LiDAR point instead of HEAL's fixed four.

    Three places in HEAL pin the count: the open3d-based pcd reader
    (xyz plus one colour channel), the spconv voxel generator
    (``num_point_features=4``) and the PointPillar encoder's feature
    net (``PillarVFE(..., num_point_features=4)``). The reader becomes
    ours (``v2b.data.opv2v_layout.read_pcd``, which returns the
    ``hag`` field a derived split wrote), the other two get ``n``.
    Everything between them (range masks, flips, rotations, the
    projection into the ego frame) slices ``[:, :3]`` and carries the
    rest along. Idempotent for the same ``n``; a second, different
    ``n`` raises, the process has one point layout.
    """
    from opencood.data_utils.pre_processor import sp_voxel_preprocessor
    from opencood.models import heter_encoders
    from opencood.utils import pcd_utils

    from v2b.data.opv2v_layout import read_pcd

    installed = getattr(pcd_utils, "_v2b_point_channels", 0)
    if installed == n:
        return
    if installed:
        raise RuntimeError(
            f"point channels already set to {installed}, asked for {n}"
        )
    if n == HEAL_POINT_CHANNELS:
        return

    def pcd_to_np(pcd_file: str) -> Any:
        pts = read_pcd(Path(pcd_file))
        if pts.shape[1] != n:
            raise ValueError(
                f"{pcd_file}: {pts.shape[1]} point channels, the run "
                f"expects {n} (derive the split with the extra field)"
            )
        return pts

    pcd_utils.pcd_to_np = pcd_to_np

    base = heter_encoders.PillarVFE

    class PillarVFEWide(base):  # type: ignore[misc,valid-type]
        def __init__(
            self, model_cfg: Any, num_point_features: int, *a: Any, **k: Any
        ) -> None:
            super().__init__(model_cfg, n, *a, **k)

    heter_encoders.PillarVFE = PillarVFEWide

    cls = sp_voxel_preprocessor.SpVoxelPreprocessor
    original_init = cls.__init__

    def wide_init(self: Any, preprocess_params: Any, train: bool) -> None:
        original_init(self, preprocess_params, train)
        if self.spconv != 2:
            raise RuntimeError("point channels need spconv 2 (Point2Voxel)")
        from spconv.utils import Point2VoxelCPU3d

        self.voxel_generator = Point2VoxelCPU3d(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.lidar_range,
            max_num_points_per_voxel=self.max_points_per_voxel,
            num_point_features=n,
            max_num_voxels=self.max_voxels,
        )

    cls.__init__ = wide_init
    pcd_utils._v2b_point_channels = n


#: Height bins of the ``hist`` pooling: 12 bins over the grid's z range.
HIST_BINS = 12
POOLINGS = ("max", "meanmax", "attn", "hist")


def install_pillar_pooling(mode: str) -> None:
    """A second pooled vector next to PointPillars' per-pillar max.

    HEAL's pillar feature net lifts every point to 64 channels and
    keeps the per-channel maximum over the pillar's points (a code of
    effectively three directions, see the S12 ground-capacity study).
    This patch keeps that max and concatenates one more pooled vector,
    then mixes both back to 64 channels with a linear layer:

    * ``meanmax``: the mean over the pillar's valid points (64);
    * ``attn``: a softmax-weighted sum over the points with learned
      per-point weights (64);
    * ``hist``: the pillar's occupancy profile, the share of its points
      in 12 height bins over the grid's z range, plus the log point
      count (13).

    The mixing layer starts as identity on the max and zero on the new
    vector, so a warm-started model computes exactly what it did and
    learns what to make of the extra pooling. ``max`` installs nothing.
    Composes with ``install_point_channels`` (install that first).
    """
    if mode == "max":
        return
    if mode not in POOLINGS:
        raise ValueError(
            f"pillar_pooling {mode!r}, expected one of {POOLINGS}"
        )
    import torch
    from opencood.models import heter_encoders
    from torch import nn
    from torch.nn import functional as functional

    installed = getattr(heter_encoders, "_v2b_pillar_pooling", "")
    if installed == mode:
        return
    if installed:
        raise RuntimeError(f"pillar pooling already {installed}, asked {mode}")
    base = heter_encoders.PillarVFE

    def keep_forward(layer: Any, inputs: Any) -> Any:
        # PFNLayer.forward, with the per-point activations kept
        x = layer.linear(inputs)
        if layer.use_norm:
            x = layer.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = functional.relu(x)
        layer.v2b_points = x
        return torch.max(x, dim=1, keepdim=True)[0]

    class PillarVFEPooled(base):  # type: ignore[misc,valid-type]
        def __init__(
            self, model_cfg: Any, num_point_features: int, *a: Any, **k: Any
        ) -> None:
            super().__init__(model_cfg, num_point_features, *a, **k)
            rng = k.get("point_cloud_range", a[1] if len(a) > 1 else None)
            self.v2b_z0 = float(rng[2]) if rng is not None else -3.0
            self.v2b_z1 = float(rng[5]) if rng is not None else 3.0
            c = int(self.num_filters[-1])
            extra = {"meanmax": c, "attn": c, "hist": HIST_BINS + 1}[mode]
            self.v2b_mix = nn.Linear(c + extra, c, bias=True)
            with torch.no_grad():
                self.v2b_mix.weight.zero_()
                self.v2b_mix.weight[:, :c] = torch.eye(c)
                self.v2b_mix.bias.zero_()
            if mode == "attn":
                self.v2b_attn = nn.Linear(c, 1)
                with torch.no_grad():
                    self.v2b_attn.weight.normal_(0.0, 0.01)
                    self.v2b_attn.bias.zero_()
            last = self.pfn_layers[-1]
            last.__class__ = type(
                "PFNLayerKept", (last.__class__,), {"forward": keep_forward}
            )

        def forward(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
            out = super().forward(batch_dict)
            top = out["pillar_features"]
            if top.dim() == 1:
                top = top.unsqueeze(0)
            x = self.pfn_layers[-1].v2b_points  # (P, N, C)
            n_pts = batch_dict["voxel_num_points"]
            n_max = x.shape[1]
            valid = torch.arange(n_max, device=x.device)[None, :] < n_pts.view(
                -1, 1
            )
            w = valid.unsqueeze(-1).to(x.dtype)
            n = n_pts.clamp(min=1).view(-1, 1).to(x.dtype)
            if mode == "meanmax":
                extra = (x * w).sum(1) / n
            elif mode == "attn":
                logits = self.v2b_attn(x).squeeze(-1)
                logits = logits.masked_fill(~valid, -1e4)
                att = torch.softmax(logits, dim=1).unsqueeze(-1)
                extra = (x * att).sum(1)
            else:
                z = batch_dict["voxel_features"][..., 2]
                frac = (z - self.v2b_z0) / (self.v2b_z1 - self.v2b_z0)
                idx = (frac * HIST_BINS).long().clamp(0, HIST_BINS - 1)
                onehot = functional.one_hot(idx, HIST_BINS).to(x.dtype) * w
                hist = onehot.sum(1) / n
                extra = torch.cat([hist, torch.log1p(n) / 4.0], dim=1)
            out["pillar_features"] = self.v2b_mix(
                torch.cat([top, extra], dim=1)
            )
            return dict(out)

    heter_encoders.PillarVFE = PillarVFEPooled
    heter_encoders._v2b_pillar_pooling = mode


VOXEL_POOLINGS = ("mean", "attn")


def install_voxel_pooling(mode: str) -> None:
    """A second pooled vector next to SECOND's per-voxel mean.

    HEAL's SECOND (modality m3) feeds its sparse convolutions the mean
    of the up to five raw points of every 0.1 m voxel (``MeanVFE``).
    ``attn`` is the attention pooling at the voxel level: next to that
    mean, a softmax-weighted sum of the voxel's points with a learned
    per-point score read from the point and its offset to the voxel
    mean, mixed back to the voxel's input width by a linear layer that
    starts as the identity on the mean and zero on the attention half.
    At init the model computes exactly what the mean encoder computes;
    it then learns where inside the voxel to place its
    representative point. The sparse backbone is untouched, so mean-encoder
    weights load unchanged. ``mean`` installs nothing.
    """
    if mode == "mean":
        return
    if mode not in VOXEL_POOLINGS:
        raise ValueError(
            f"voxel_pooling {mode!r}, expected one of {VOXEL_POOLINGS}"
        )
    import torch
    from opencood.models import heter_encoders
    from torch import nn

    installed = getattr(heter_encoders, "_v2b_voxel_pooling", "")
    if installed == mode:
        return
    if installed:
        raise RuntimeError(f"voxel pooling already {installed}, asked {mode}")
    base = heter_encoders.MeanVFE

    class MeanVFEPooled(base):  # type: ignore[misc,valid-type]
        def __init__(
            self, model_cfg: Any, num_point_features: int, **kwargs: Any
        ) -> None:
            super().__init__(model_cfg, num_point_features, **kwargs)
            c = int(num_point_features)
            self.v2b_attn = nn.Linear(2 * c, 1)
            self.v2b_mix = nn.Linear(2 * c, c, bias=True)
            with torch.no_grad():
                self.v2b_attn.weight.normal_(0.0, 0.01)
                self.v2b_attn.bias.zero_()
                self.v2b_mix.weight.zero_()
                self.v2b_mix.weight[:, :c] = torch.eye(c)
                self.v2b_mix.bias.zero_()

        def forward(self, batch_dict: dict[str, Any], **kwargs: Any) -> Any:
            points = batch_dict["voxel_features"]  # (V, N, C), zero padded
            n_pts = batch_dict["voxel_num_points"]
            out = super().forward(batch_dict, **kwargs)
            mean = out["voxel_features"]  # (V, C)
            n_max = points.shape[1]
            valid = torch.arange(n_max, device=points.device)[
                None, :
            ] < n_pts.view(-1, 1)
            rel = points - mean.unsqueeze(1)
            logits = self.v2b_attn(torch.cat([points, rel], dim=-1))
            logits = logits.squeeze(-1).masked_fill(~valid, -1e4)
            att = torch.softmax(logits, dim=1).unsqueeze(-1)
            extra = (points * att).sum(1)
            out["voxel_features"] = self.v2b_mix(
                torch.cat([mean, extra], dim=1)
            ).contiguous()
            return out

    heter_encoders.MeanVFE = MeanVFEPooled
    heter_encoders._v2b_voxel_pooling = mode
