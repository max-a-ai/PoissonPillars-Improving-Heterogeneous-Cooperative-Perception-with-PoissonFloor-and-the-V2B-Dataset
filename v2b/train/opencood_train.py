#!/usr/bin/env python3
"""Train an OpenCOOD detector with wandb logging.

Runs **inside ``third_party/CodeFilling/.venv``**, which owns torch,
spconv and the built CUDA ops. It is never imported by the ``v2b``
package proper -- that environment has no torch on purpose, because the
detector stack pins its own CUDA build. ``v2b/scripts/v2b-main.py`` is
the launcher; run that, not this.

Differences from ``opencood/tools/train.py``, all deliberate:

* **wandb instead of tensorboard.** Every run in this project is
  traceable from a wandb curve back to a diff, so the run name is the
  same string in wandb, in ``outputs/<run-name>/`` and in
  ``.docs/progress.md``.
* **BEV average precision at each eval epoch**, not just a validation
  loss. OpenCOOD's AP matches boxes as rotated rectangles on their
  bottom face (``common_utils.convert_format`` keeps corners 0-3 and
  drops z), so it is an *AP@BEV* and says nothing about height
  accuracy -- the boxes themselves are full 7-DOF 3D.
* **Score-map statistics.** ``cls_preds_single`` is each agent's own
  classification map before fusion; ``sigmoid`` of its per-anchor max
  is Where2comm's spatial confidence (score) map and ``1 - C`` is the
  request map. Logging it from stage 1 onwards makes the stage-2
  communication gate's input observable before the gate exists.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.train.hypes import fork_dir  # noqa: E402
from v2b.utils.opencood_shims import install as install_shims  # noqa: E402

#: Which fork serves ``opencood``; the launcher sets it from the spec.
FORK = os.environ.get("V2B_FORK", "codefilling")
install_shims(fork_dir(FORK))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import wandb  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402

if FORK == "heal":  # HEAL keeps the AP helpers under utils
    from opencood.utils import eval_utils  # noqa: E402
else:
    from opencood.tools import eval_utils  # noqa: E402
from opencood.utils import box_utils  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hypes", required=True, help="rendered config.yaml")
    ap.add_argument("--run-dir", required=True, help="outputs/<run-name>")
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--wandb-name", required=True)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--init-from", default=None, help="warm-start ckpt dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-batches", type=int, default=150)
    ap.add_argument(
        "--ap-samples",
        type=int,
        default=300,
        help="frames scored for AP each eval epoch (0 disables AP, "
        "negative scores every validation frame)",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="score the warm-start or bestval checkpoint on the "
        "validation view and exit without training",
    )
    # the robustness harness (v2b/scripts/v2b-robustness.py): test-time
    # settings only, see v2b/train/robustness.py
    ap.add_argument("--eval-ckpt", default=None, help="eval only: this ckpt")
    ap.add_argument(
        "--eval-file",
        default=None,
        help="eval only: this weight file (e.g. a run's net_latest.pth) "
        "instead of the directory's bestval checkpoint",
    )
    ap.add_argument("--eval-out", default=None, help="eval only: json path")
    ap.add_argument(
        "--pose-noise",
        nargs=2,
        type=float,
        default=None,
        metavar=("SIGMA_T", "SIGMA_R"),
        help="eval only: noise on the other agents' pose (m, deg)",
    )
    ap.add_argument(
        "--latency",
        type=int,
        default=0,
        help="eval only: the other agents' frame lags by k frames",
    )
    ap.add_argument("--noise-seed", type=int, default=303)
    ap.add_argument(
        "--ego-only",
        action="store_true",
        help="eval only: a fusion model on the ego's map alone (the "
        "other agents are dropped before the fusion module)",
    )
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--dump-detections",
        action="store_true",
        help="eval only: keep every decoded and labelled box of the "
        "scored frames in <run-dir>/detections.npz "
        "(v2b.train.detection_dump)",
    )
    ap.add_argument(
        "--dump-agent-detections",
        action="store_true",
        help="eval only, late fusion: keep every agent's own decoded "
        "boxes (own frame, before the merge) in "
        "<run-dir>/agent_detections.npz (v2b.train.agent_dump)",
    )
    return ap.parse_args()


def ddp_env() -> tuple[int, int, int]:
    """``(rank, world_size, local_rank)`` under torchrun, else (0, 1, 0)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, 0
    return (
        int(os.environ.get("RANK", "0")),
        world,
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def per_rank_batch(global_batch: int, world: int) -> int:
    """The per-process batch that keeps the global batch of a 1-GPU row."""
    if global_batch % world:
        raise ValueError(f"batch {global_batch} does not split over {world}")
    return global_batch // world


def unwrap(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def make_loaders(
    hypes: dict[str, Any],
    workers: int,
    ap_samples: int,
    eval_only: bool = False,
    world: int = 1,
    rank: int = 0,
    late: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None, Any]:
    """Train, validation-loss and AP loaders over the mixed view.

    The AP loader is batch size 1 on purpose:
    ``VoxelPostprocessor.post_process`` reshapes the class map with a
    hard-coded leading ``1``, so decoding a larger batch would silently
    fold several frames' anchors into one frame's detections. With
    ``late`` (the late-fusion rows) it uses HEAL's test collate, which
    keeps every agent of the frame instead of the ego alone.
    """
    val_set = build_dataset(hypes, visualize=False, train=False)
    # an eval-only run needs no train view (it may not even exist yet)
    train_set = (
        val_set
        if eval_only
        else build_dataset(hypes, visualize=False, train=True)
    )
    collate = train_set.collate_batch_train
    share = float(hypes["train_params"].get("subset", 1.0) or 1.0)
    if not eval_only and 0 < share < 1:
        # a fixed random share of the train frames (the S28 short tests)
        n = len(train_set)
        pick = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        train_set = Subset(train_set, sorted(pick[: int(n * share)].tolist()))
    batch = per_rank_batch(int(hypes["train_params"]["batch_size"]), world)
    common: dict[str, Any] = {
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": True,
        "drop_last": True,
    }
    if workers:
        common["prefetch_factor"] = 4
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_set, num_replicas=world, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_set, batch_size=batch, sampler=sampler, **common
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch, shuffle=True, **common
        )
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, **common)
    ap_loader = None
    if ap_samples:
        stride = 1 if ap_samples < 0 else max(1, len(val_set) // ap_samples)
        subset = Subset(val_set, list(range(0, len(val_set), stride)))
        ap_loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=min(workers, 4),
            collate_fn=val_set.collate_batch_test if late else collate,
        )
    return train_loader, val_loader, ap_loader, val_set


def warm_start(model: Any, ckpt_dir: str) -> str:
    """Adopt whatever tensors of ``ckpt_dir`` fit ``model``.

    Shapes that disagree -- an anchor head sized for a different BEV
    grid, say -- are dropped rather than forced. The pillar encoder and
    the convolutional trunk transfer across grid sizes; grid-shaped
    tensors do not. One exception: the pillar net's first linear of a
    run with more point channels than the checkpoint (``point_channels``
    in the spec) is widened with zero columns for the new channels,
    so the warm start computes what the checkpoint did; the same for a
    shrink conv whose input grew by an extra (last) neck branch.
    """
    path = Path(ckpt_dir)
    weights = sorted(path.glob("net_epoch_bestval_at*.pth")) or sorted(
        path.glob("*.pth")
    )
    if not weights:
        raise FileNotFoundError(f"no .pth under {path}")
    state = torch.load(weights[0], map_location="cpu", weights_only=False)
    own = model.state_dict()
    keep = {
        k: v for k, v in state.items() if k in own and own[k].shape == v.shape
    }
    widened = 0
    for k, v in state.items():
        if (
            k.endswith("pfn_layers.0.linear.weight")
            and k in own
            and k not in keep
            and own[k].shape[0] == v.shape[0]
            and own[k].shape[1] > v.shape[1]
        ):
            from v2b.train.heal_data import widen_point_columns

            keep[k] = widen_point_columns(v, int(own[k].shape[1]))
            widened += 1
        elif (
            "shrink_conv" in k
            and k.endswith("weight")
            and k in own
            and k not in keep
            and v.dim() == 4
            and own[k].shape[0] == v.shape[0]
            and own[k].shape[2:] == v.shape[2:]
            and own[k].shape[1] > v.shape[1]
        ):
            # a neck with an extra (last) branch, e.g. S17's stride-16
            # stage: the old channels keep their weights, the new ones
            # start at zero, so the head sees what the checkpoint did
            wide = torch.zeros(own[k].shape, dtype=v.dtype)
            wide[:, : v.shape[1]] = v
            keep[k] = wide
            widened += 1
    model.load_state_dict(keep, strict=False)
    model.v2b_adopted = sorted(keep)  # for the frozen-trunk control
    return (
        f"warm start {weights[0].name}: {len(keep)}/{len(own)} tensors "
        f"adopted ({widened} widened for extra input channels), "
        f"{len(state) - len(keep)} skipped on shape"
    )


def score_map_stats(output: dict[str, Any]) -> dict[str, float]:
    """Spatial confidence (score) and request summary for one forward.

    Prefers the per-agent pre-fusion map, which is the one Where2comm
    actually gates on; falls back to the fused head for models that
    have no separate single branch.
    """
    psm = output.get("cls_preds_single", output.get("cls_preds"))
    if psm is None:
        return {}
    conf = torch.sigmoid(psm.detach()).amax(dim=1)
    stats = {
        "score_mean": float(conf.mean()),
        "score_max": float(conf.max()),
        "score_frac_above_0.2": float((conf > 0.2).float().mean()),
        "request_mean": float((1.0 - conf).mean()),
    }
    rate = output.get("comm_rate")
    if rate is not None:
        stats["comm_rate"] = float(torch.as_tensor(rate).float().mean())
    return stats


def _gt_corners(batch: dict[str, Any], order: str) -> torch.Tensor | None:
    """Ground-truth boxes of a batch-size-1 sample as ``(N, 8, 3)``."""
    ego = batch["ego"]
    mask = ego["object_bbx_mask"][0].bool()
    centers = ego["object_bbx_center"][0][mask]
    if centers.numel() == 0:
        return None
    return box_utils.boxes_to_corners_3d(centers, order)


def footprint_lw(corners: torch.Tensor) -> np.ndarray:
    """``(N, 2)`` length and width of ``(N, 8, 3)`` corner boxes."""
    if corners is None or len(corners) == 0:
        return np.zeros((0, 2))
    c = corners.detach().cpu().numpy()[:, :4, :2]
    e1 = np.linalg.norm(c[:, 1] - c[:, 0], axis=1)
    e2 = np.linalg.norm(c[:, 3] - c[:, 0], axis=1)
    return np.stack([np.maximum(e1, e2), np.minimum(e1, e2)], axis=1)


def _new_stat() -> dict[float, dict[str, Any]]:
    return {
        t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in IOU_THRESHOLDS
    }


def bev_ap(
    model: Any,
    loader: DataLoader,
    post: Any,
    anchors: torch.Tensor,
    order: str,
    device: torch.device,
    epoch: int,
    class_anchors: dict[str, list[float]] | None = None,
    box_filter: dict[str, Any] | None = None,
    late: bool = False,
    probe: Any = None,
    dump: Any = None,
    agent_dump: Any = None,
) -> dict[str, Any]:
    """AP@BEV over the AP subset, decoded frame by frame.

    ``anchor_box`` and an identity ``transformation_matrix`` are
    injected because the training collate omits both -- predictions are
    already in the ego frame, so identity is correct here. With
    ``class_anchors`` (the per-class sizes of a multi-class run) every
    box, predicted or true, is assigned the class of the nearest anchor
    size and AP is reported per class as well.

    ``late`` is the late-fusion path: the single-agent
    model runs on every agent of the frame, the postprocessor projects
    each agent's boxes into the ego frame with its
    ``transformation_matrix`` and NMS merges them (HEAL's own late
    inference), against the ego's (union) labels. The other agents'
    box counts give the bytes per message; ``probe`` (a
    ``FeatureProbe``) does the same for the feature-fusion rows.
    ``dump`` (a ``DetectionDump``) keeps every frame's boxes,
    ``agent_dump`` (an ``AgentDump``) every agent's own boxes before
    the late merge.
    """
    stat = _new_stat()
    per_class = {c: _new_stat() for c in (class_anchors or {})}
    eye = torch.eye(4, device=device)
    messages: list[tuple[int, int]] = []  # (boxes, VRU boxes) per message
    model.eval()
    indices = getattr(loader.dataset, "indices", None)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            gt = _gt_corners(batch, order)
            batch = train_utils.to_device(batch, device)
            if late:
                outputs = {cav: model(batch[cav]) for cav in batch}
                messages += message_boxes(batch, outputs, post, class_anchors)
                if agent_dump is not None:
                    agent_dump.add(
                        int(indices[i]) if indices is not None else i,
                        batch,
                        outputs,
                        gt,
                        class_anchors,
                    )
                boxes, scores = post.post_process(batch, outputs)
            else:
                batch["ego"]["epoch"] = epoch
                output = model(batch["ego"])
                batch["ego"]["anchor_box"] = anchors
                batch["ego"]["transformation_matrix"] = eye
                boxes, scores = post.post_process(
                    {"ego": batch["ego"]}, {"ego": output}
                )
            if box_filter and not late:
                from v2b.train.box_filter import keep_boxes

                boxes, scores, _ = keep_boxes(
                    boxes,
                    scores,
                    batch["ego"]["inputs_m1"],
                    box_filter,
                    class_anchors,
                )
            if dump is not None:
                inputs = next(
                    v
                    for k, v in batch["ego"].items()
                    if k.startswith("inputs_")
                )
                dump.add(
                    int(indices[i]) if indices is not None else i,
                    inputs,
                    boxes,
                    scores,
                    gt,
                    _names(boxes, class_anchors),
                    _names(gt, class_anchors),
                )
            if gt is None:
                continue
            score_frame(
                stat, per_class, boxes, scores, gt.to(device), class_anchors
            )
    out: dict[str, Any] = ap_summary(stat, per_class)
    out.update(message_summary(messages, probe))
    return out


def _names(
    corners: torch.Tensor | None, class_anchors: dict[str, list[float]] | None
) -> list[str] | None:
    """Class of each corner box by its footprint, ``None`` without
    per-class anchors or boxes."""
    if not class_anchors or corners is None or len(corners) == 0:
        return None
    from v2b.train.heal_data import box_classes

    return box_classes(footprint_lw(corners), class_anchors)


def message_boxes(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    post: Any,
    class_anchors: dict[str, list[float]] | None,
) -> list[tuple[int, int]]:
    """``(boxes, VRU boxes)`` each non-ego agent would send in a late
    fusion message: its own detections after its own NMS."""
    sent: list[tuple[int, int]] = []
    for cav in batch:
        if cav == "ego":
            continue
        boxes, _ = post.post_process({cav: batch[cav]}, {cav: outputs[cav]})
        n = 0 if boxes is None else int(len(boxes))
        vru = 0
        if n and class_anchors:
            from v2b.train.heal_data import box_classes

            classes = box_classes(footprint_lw(boxes), class_anchors)
            vru = sum(c != "car" for c in classes)
        sent.append((n, vru))
    return sent


def message_summary(
    messages: list[tuple[int, int]], probe: Any
) -> dict[str, Any]:
    """Bytes per message of the row under evaluation: the late box list
    (and the codebook message it implies) from the counted messages,
    the fp16 feature map from the probe's recorded shape."""
    from v2b.train.robustness import box_bytes, codebook_bytes

    out: dict[str, Any] = {}
    if messages:
        n = statistics.mean(m[0] for m in messages)
        vru = statistics.mean(m[1] for m in messages)
        out["msg/boxes_per_message"] = n
        out["msg/vru_per_message"] = vru
        out["msg/bytes_late_fp32"] = box_bytes(n)
        out["msg/bytes_codebook"] = codebook_bytes(n, vru)
        out["msg/messages"] = len(messages)
    if probe is not None and probe.shape is not None:
        out["msg/feature_shape"] = list(probe.shape)
        out["msg/bytes_feature_fp16"] = probe.message_bytes()
    return out


def score_frame(
    stat: dict[float, dict[str, Any]],
    per_class: dict[str, dict[float, dict[str, Any]]],
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    gt: torch.Tensor,
    class_anchors: dict[str, list[float]] | None,
) -> None:
    """Add one frame's matches to the running AP statistics.

    ``boxes`` and ``gt`` are ``(N, 8, 3)`` corners on one device. Shared
    by :func:`bev_ap` and ``v2b-score-bev.py``, which scores an external
    detector's boxes against the same ground truth with the same code.
    """
    device = gt.device
    for thr in IOU_THRESHOLDS:
        eval_utils.caluclate_tp_fp(boxes, scores, gt, stat, thr)
    if not class_anchors:
        return
    from v2b.train.heal_data import box_classes

    gt_cls = np.array(box_classes(footprint_lw(gt), class_anchors))
    det_cls = np.array(box_classes(footprint_lw(boxes), class_anchors))
    for c, cstat in per_class.items():
        g = gt[torch.from_numpy(gt_cls == c).to(device)]
        keep = torch.from_numpy(det_cls == c).to(device)
        d = boxes[keep] if boxes is not None and len(boxes) else boxes
        sc = scores[keep] if scores is not None and len(scores) else scores
        for thr in IOU_THRESHOLDS:
            eval_utils.caluclate_tp_fp(d, sc, g, cstat, thr)


def ap_summary(
    stat: dict[float, dict[str, Any]],
    per_class: dict[str, dict[float, dict[str, Any]]],
) -> dict[str, float]:
    """``val/ap_bev@<thr>[/<class>]`` from accumulated statistics."""
    out: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        if stat[thr]["gt"] == 0:
            continue
        ap, _, _ = eval_utils.calculate_ap(stat, thr)
        out[f"val/ap_bev@{thr}"] = float(ap)
        for c, cstat in per_class.items():
            if cstat[thr]["gt"]:
                ap, _, _ = eval_utils.calculate_ap(cstat, thr)
                out[f"val/ap_bev@{thr}/{c}"] = float(ap)
    return out


def validation_loss(
    model: Any,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
    epoch: int,
    max_batches: int,
    need_labels: bool = True,
) -> dict[str, float]:
    """Mean validation loss and score-map stats over a capped pass.

    Frames without boxes are skipped for a detector; a pretraining run
    (``need_labels=False``) scores them (it reads no labels, and an
    unlabelled validation split would otherwise average an empty list
    to NaN).
    """
    model.eval()
    losses: list[float] = []
    maps: list[dict[str, float]] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if batch is None:
                continue
            if need_labels and batch["ego"]["object_bbx_mask"].sum() == 0:
                continue
            batch = train_utils.to_device(batch, device)
            batch["ego"]["epoch"] = epoch
            output = model(batch["ego"])
            losses.append(float(criterion(output, batch["ego"]["label_dict"])))
            maps.append(score_map_stats(output))
    metrics = {"val/loss": statistics.mean(losses) if losses else float("nan")}
    for key in maps[0] if maps else {}:
        metrics[f"val/{key}"] = statistics.mean(m[key] for m in maps)
    return metrics


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    criterion: Any,
    optimizer: Any,
    device: torch.device,
    epoch: int,
    single_weight: float | None,
    step: int,
    need_labels: bool = True,
) -> int:
    """One pass over the training set; returns the new global step."""
    model.train()
    if hasattr(unwrap(model), "model_train_init"):
        unwrap(model).model_train_init()
    ddp = hasattr(model, "module")
    tic = time.time()
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        unlabelled = batch["ego"]["object_bbx_mask"].sum() == 0
        if need_labels and not ddp and unlabelled:
            # under DDP every rank must step together: a frame without
            # boxes still runs (HEAL's loss handles zero positives); a
            # pretraining run keeps them on one GPU too (no labels read)
            continue
        optimizer.zero_grad(set_to_none=True)
        batch = train_utils.to_device(batch, device)
        batch["ego"]["epoch"] = epoch
        output = model(batch["ego"])
        loss = criterion(output, batch["ego"]["label_dict"])
        if single_weight is not None:
            loss = loss + single_weight * criterion(
                output, batch["ego"]["label_dict_single"], suffix="_single"
            )
        loss.backward()
        optimizer.step()
        step += 1
        if i % 25 == 0:
            payload = {f"train/{k}": v for k, v in criterion.loss_dict.items()}
            payload["train/epoch"] = epoch
            payload["train/lr"] = optimizer.param_groups[0]["lr"]
            payload["train/it_per_s"] = (i + 1) / max(time.time() - tic, 1e-6)
            payload.update(
                {f"train/{k}": v for k, v in score_map_stats(output).items()}
            )
            wandb.log(payload, step=step)
    return step


def save_bestval(model: Any, run_dir: Path, epoch: int) -> Path:
    """Write the new best checkpoint and drop the one it replaces.

    Only the current best is ever of interest, and each is ~32 MB --
    run 1 finished with nine of them, 290 MB, eight of which no longer
    meant anything. Upstream's loop prunes; this one did not, which is
    the kind of thing that fills a disk quietly over a sweep.
    """
    path = run_dir / f"net_epoch_bestval_at{epoch + 1}.pth"
    torch.save(unwrap(model).state_dict(), path)
    for stale in run_dir.glob("net_epoch_bestval_at*.pth"):
        if stale != path:
            stale.unlink()
    return path


def load_hypes(path: str) -> dict[str, Any]:
    """The rendered config with the data hooks its run trained under.

    Through opencood's own loader, not plain yaml: it applies the
    ``yaml_parser`` named in the config, which is what recomputes
    anchor_args H/W/D from the retargeted range and voxel. The class
    filter, camera and anchor hooks must be installed before a dataset
    is built, or the ground truth differs from the run's.
    """
    hypes: dict[str, Any] = yaml_utils.load_yaml(path)
    if FORK == "heal":
        from v2b.train.heal_data import (
            install_class_filter,
            install_depth_lookup,
            install_multiclass_anchors,
            install_opencv_cameras,
        )

        install_depth_lookup()
        install_opencv_cameras()
        if hypes.get("v2b_anchor_classes"):
            install_multiclass_anchors()
        if hypes.get("v2b_class_thresholds"):
            from v2b.train.heal_data import install_class_thresholds

            install_class_thresholds(
                hypes["v2b_class_thresholds"],
                list(hypes["v2b_anchor_classes"]),
                len(hypes["postprocess"]["anchor_args"]["r"]),
                class_anchors_of(hypes) or {},
            )
        if hypes.get("v2b_classes"):
            install_class_filter(list(hypes["v2b_classes"]))
        if hypes.get("v2b_max_range"):
            from v2b.train.heal_data import install_range_filter

            install_range_filter(float(hypes["v2b_max_range"]))
        if hypes.get("v2b_point_channels"):
            from v2b.train.heal_data import install_point_channels

            install_point_channels(int(hypes["v2b_point_channels"]))
        if hypes.get("v2b_pillar_pooling"):
            from v2b.train.heal_data import install_pillar_pooling

            install_pillar_pooling(str(hypes["v2b_pillar_pooling"]))
        if hypes.get("v2b_voxel_pooling"):
            from v2b.train.heal_data import install_voxel_pooling

            install_voxel_pooling(str(hypes["v2b_voxel_pooling"]))
        if (hypes.get("v2b_codespace") or {}).get("twin"):
            from v2b.train.twin_data import install_twin_loader

            install_twin_loader()
        if hypes.get("v2b_fused_augment"):
            from v2b.train.fusion_data import install_fused_augment

            install_fused_augment()
        if hypes.get("v2b_union_labels"):
            from v2b.train.fusion_data import install_union_labels

            install_union_labels()
    elif FORK == "codefilling" and (
        hypes.get("v2b_classes") or hypes.get("v2b_max_range")
    ):
        # CodeFilling is a HEAL-derived fork with the same label loader,
        # so the class and range filters apply; its heads stay single-class
        from v2b.train.heal_data import install_class_filter

        if hypes.get("v2b_classes"):
            install_class_filter(list(hypes["v2b_classes"]))
        if hypes.get("v2b_max_range"):
            from v2b.train.heal_data import install_range_filter

            install_range_filter(float(hypes["v2b_max_range"]))
    if "gaussian_lift_splat" in Path(path).read_text():
        from v2b.models.gaussian_lift_splat import register

        register()
    if hypes.get("v2b_litept"):
        from v2b.models.litept_encoder import register as register_litept

        register_litept()
    return hypes


def robustness_setup(args: argparse.Namespace) -> list[str]:
    """Install the test-time hooks of the robustness harness before any
    dataset is built; returns one note per hook for the run's record."""
    notes: list[str] = []
    if not args.eval_only:
        if args.pose_noise is not None or args.latency:
            raise SystemExit("pose noise and latency are eval-only settings")
        return notes
    from v2b.train.robustness import install_latency, install_pose_noise

    if args.latency:
        install_latency(int(args.latency))
        notes.append(
            f"latency: the other agents' frame t-{int(args.latency)} "
            "(cloud and pose at capture time; labels of t)"
        )
    if args.pose_noise is not None:
        sigma_t, sigma_r = (float(s) for s in args.pose_noise)
        if sigma_t > 0 or sigma_r > 0:
            install_pose_noise(sigma_t, sigma_r, int(args.noise_seed))
        notes.append(
            f"pose noise on the other agents: sigma_t {sigma_t} m, "
            f"sigma_r {sigma_r} deg, seed {int(args.noise_seed)} per frame"
        )
    if args.deterministic:
        seed = int(args.noise_seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        notes.append(f"deterministic: torch and numpy seed {seed}, cuDNN")
    return notes


def class_anchors_of(hypes: dict[str, Any]) -> dict[str, list[float]] | None:
    """Per-class anchor sizes of a multi-class run, else ``None``."""
    if not hypes.get("v2b_anchor_classes"):
        return None
    aa = hypes["postprocess"]["anchor_args"]
    return {
        c: [aa["l"][i], aa["w"][i], aa["h"][i]]
        for i, c in enumerate(hypes["v2b_anchor_classes"])
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hypes = load_hypes(args.hypes)
    rank, world, local_rank = ddp_env()
    main_rank = rank == 0
    if world > 1:
        # rank 0 validates alone at every epoch (full V2X-Real test set
        # through a fusion model: well over NCCL's 10 min default) while
        # the others wait in the barrier; give the collectives hours
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain = hypes.get("v2b_pretrain")
    if pretrain:
        args.ap_samples = 0  # no detections to score
    late = bool(hypes.get("v2b_late_fusion"))
    if late and not args.eval_only:
        raise SystemExit("late_fusion rows are evaluation only (--eval-only)")
    robust = robustness_setup(args)
    train_loader, val_loader, ap_loader, val_set = make_loaders(
        hypes,
        args.num_workers,
        args.ap_samples,
        args.eval_only,
        world,
        rank,
        late,
    )
    model = train_utils.create_model(hypes)
    notes: list[str] = []
    if hypes.get("v2b_bev_attention"):
        # before the optimizer, so the new stages' parameters are trained
        from v2b.train.bev_attention import install_bev_attention

        notes += install_bev_attention(model, hypes["v2b_bev_attention"])
    if hypes.get("v2b_fusion_trunk") == "single":
        from v2b.train.fusion_trunk import (
            head_stride_factor,
            install_single_trunk,
        )

        notes.append(
            install_single_trunk(model, pool=head_stride_factor(hypes))
        )
    criterion = train_utils.create_loss(hypes)
    if pretrain and pretrain.get("target") == "floor":
        # the trunk regresses the Poisson floor height per cell
        # from the raw pillars (teacher = the hag point channel)
        from v2b.train.floor_net import FloorLoss, FloorModel

        ego = str(hypes["heter"]["ego_modality"]).split("&")[0]
        setting = hypes["heter"]["modality_setting"][ego]["preprocess"]
        model = FloorModel(
            model,
            list(setting["cav_lidar_range"]),
            modality=ego,
            voxel_size=[float(v) for v in setting["args"]["voxel_size"]],
            decoder_layers=int(pretrain.get("decoder_layers", 1)),
        )
        criterion = FloorLoss()
        notes.append(f"learned floor from the {ego} trunk: {pretrain}")
    elif pretrain:
        from v2b.train.pretrain import MaskedPretrainModel, PretrainLoss

        ego = str(hypes["heter"]["ego_modality"]).split("&")[0]
        setting = hypes["heter"]["modality_setting"][ego]["preprocess"]
        model = MaskedPretrainModel(
            model,
            list(setting["cav_lidar_range"]),
            modality=ego,
            voxel_size=[float(v) for v in setting["args"]["voxel_size"]],
            mask_ratio=float(pretrain.get("mask_ratio", 0.7)),
            bins=int(pretrain.get("bins", 12)),
            floor_z=float(pretrain.get("floor_z", -1.9)),
            drop_below=pretrain.get("drop_below"),
            exclude_ground=bool(pretrain.get("exclude_ground", False)),
            max_range=pretrain.get("max_range"),
            intensity_bins=int(pretrain.get("intensity_bins", 0)),
            block=int(pretrain.get("block", 1)),
            decoder_layers=int(pretrain.get("decoder_layers", 1)),
            occupancy=bool(pretrain.get("occupancy", False)),
        )
        criterion = PretrainLoss(
            count_weight=float(pretrain.get("count_weight", 1.0)),
            profile_weight=float(pretrain.get("profile_weight", 1.0)),
            intensity_weight=float(pretrain.get("intensity_weight", 1.0)),
        )
        notes.append(f"masked pretraining of the {ego} trunk: {pretrain}")
    codespace = hypes.get("v2b_codespace")
    if codespace:
        from v2b.train.codespace import CodeSpaceLoss, wrap_codespace

        model = wrap_codespace(model, codespace)
        criterion = CodeSpaceLoss(
            criterion,
            vq_weight=float(codespace.get("vq_weight", 1.0)),
            recon_weight=float(codespace.get("recon_weight", 1.0)),
            sim_weight=float(codespace.get("sim_weight", 1.0)),
        )
        notes.append(
            f"code space {codespace}: {model.bits_per_cell} bits per cell"
        )
    if hypes.get("v2b_grounding_loss"):
        from v2b.train.grounding import GroundedLoss, slot_table

        g = hypes["v2b_grounding_loss"]
        heights, pulled = slot_table(
            list(hypes["v2b_anchor_classes"]),
            class_anchors_of(hypes) or {},
            len(hypes["postprocess"]["anchor_args"]["r"]),
            [str(c) for c in g.get("classes", ["pedestrian", "bicycle"])],
        )
        criterion = GroundedLoss(
            criterion,
            heights,
            pulled,
            floor_z=float(g.get("floor_z", -1.9)),
            tau=float(g.get("tau", 0.3)),
            weight=float(g.get("weight", 1.0)),
        )
        print(f"grounding loss: {g}")

    init_epoch = 0
    if args.eval_only and args.eval_file:
        # one named weight file (the checkpoint re-selection, )
        state = torch.load(args.eval_file, map_location="cpu")
        train_utils.check_missing_key(model.state_dict(), state)
        model.load_state_dict(state, strict=False)
        init_epoch = int(hypes["train_params"]["epoches"])
        notes.append(f"eval of the weight file {args.eval_file}")
    elif args.eval_only and args.eval_ckpt:
        # the robustness harness names the checkpoint explicitly
        init_epoch, model = train_utils.load_saved_model(args.eval_ckpt, model)
        notes.append(f"eval of {args.eval_ckpt} at epoch {init_epoch}")
    elif args.eval_only and any(run_dir.glob("net_epoch*.pth")):
        # the run's own result wins over the spec's warm start
        init_epoch, model = train_utils.load_saved_model(str(run_dir), model)
        notes.append(f"eval of the run's checkpoint at epoch {init_epoch}")
    elif args.resume and any(run_dir.glob("net_epoch*.pth")):
        init_epoch, model = train_utils.load_saved_model(str(run_dir), model)
        notes.append(f"resumed at epoch {init_epoch}")
    elif args.init_from:
        notes.append(warm_start(model, args.init_from))
    model.to(device)
    if world > 1:
        # the warm start, the pooling and the attention stages are in
        # place on every rank; SyncBatchNorm keeps the batch statistics
        # of the 1-GPU rows (global batch 8, not 2 per GPU)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if hypes.get("v2b_freeze_trunk"):
        from v2b.train.fusion_trunk import freeze_adopted

        notes.append(
            freeze_adopted(model, list(getattr(model, "v2b_adopted", [])))
        )
    if world > 1:
        # HEAL's heter models carry heads and modules a forward may not
        # touch (the smoke on two GPUs stopped at iteration two without
        # this), so DDP has to tolerate parameters without gradients
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        per_gpu = per_rank_batch(
            int(hypes["train_params"]["batch_size"]), world
        )
        notes.append(
            f"DDP over {world} GPUs, {per_gpu} per GPU, SyncBatchNorm"
        )
    optimizer = train_utils.setup_optimizer(hypes, unwrap(model))
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, init_epoch=init_epoch
    )

    post = val_set.post_processor
    anchors = torch.from_numpy(post.generate_anchor_box()).to(device)
    order = hypes["postprocess"]["order"]
    class_anchors = class_anchors_of(hypes)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        dir=str(run_dir),
        mode=None if main_rank else "disabled",
        config={
            "hypes": hypes,
            "trainable_params": n_params,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_set),
            "bev_cells": list(anchors.shape[:2]),
            "notes": notes,
        },
    )
    print(f"model      : {n_params:,} trainable params")
    print(f"anchors    : {tuple(anchors.shape)} (H, W, n_anchor, 7)")
    print(
        f"samples    : {len(train_loader.dataset)} train / {len(val_set)} val"
    )
    for note in notes:
        print(f"note       : {note}")

    single_weight = (
        hypes["train_params"].get("single_weight", 1.0)
        if hypes["model"]["args"].get("supervise_single") and not pretrain
        else None
    )
    epochs = hypes["train_params"]["epoches"]
    eval_freq = hypes["train_params"].get("eval_freq", 1)
    best = float("inf")
    step = init_epoch * len(train_loader)

    if args.eval_only and not main_rank:
        return 0
    if args.eval_only:
        from v2b.train.robustness import FeatureProbe, delayed_frames

        notes += robust
        metrics: dict[str, Any] = validation_loss(
            model,
            val_loader,
            criterion,
            device,
            init_epoch,
            args.val_batches,
            need_labels=not pretrain,
        )
        if device.type == "cuda":
            # the loss pass ran a training-size batch; give its cache
            # back before the frame-by-frame AP pass (the GPU is shared)
            torch.cuda.empty_cache()
        probe = FeatureProbe()
        probe.attach(model)
        if args.ego_only:
            from v2b.train.robustness import install_ego_only

            if install_ego_only(model):
                robust.append(
                    "ego only: the fusion module sees the ego's map alone"
                )
            else:
                robust.append("ego only requested, but no fusion module")
        dump = None
        if args.dump_detections:
            from v2b.train.detection_dump import DetectionDump

            dump = DetectionDump(val_set, list((class_anchors or {}).keys()))
        agent_dump = None
        if args.dump_agent_detections:
            if not late:
                raise SystemExit("--dump-agent-detections needs late_fusion")
            from v2b.train.agent_dump import AgentDump

            agent_dump = AgentDump(
                val_set, list((class_anchors or {}).keys()), post
            )
        if ap_loader is not None:
            metrics.update(
                bev_ap(
                    model,
                    ap_loader,
                    post,
                    anchors,
                    order,
                    device,
                    init_epoch,
                    class_anchors,
                    hypes.get("v2b_box_filter"),
                    late,
                    probe,
                    dump,
                    agent_dump,
                )
            )
        probe.detach()
        if dump is not None:
            metrics["eval/detections"] = str(
                dump.save(run_dir / "detections.npz")
            )
        if agent_dump is not None:
            metrics["eval/agent_detections"] = str(
                agent_dump.save(run_dir / "agent_detections.npz")
            )
        metrics["eval/frames"] = len(ap_loader.dataset) if ap_loader else 0
        metrics["eval/checkpoint"] = (
            args.eval_file or args.eval_ckpt or str(run_dir)
        )
        metrics["eval/late_fusion"] = late
        if args.pose_noise is not None:
            metrics["eval/pose_noise_t"] = float(args.pose_noise[0])
            metrics["eval/pose_noise_r"] = float(args.pose_noise[1])
        metrics["eval/latency"] = int(args.latency)
        metrics["eval/ego_only"] = bool(args.ego_only)
        metrics["eval/delayed_frames"] = delayed_frames(
            list(val_set.len_record), int(args.latency)
        )
        metrics["eval/notes"] = robust
        wandb.log(metrics, step=step)
        out = Path(args.eval_out) if args.eval_out else run_dir / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=1))
        print(f"[eval] {json.dumps(metrics)}", flush=True)
        wandb.finish()
        return 0

    for epoch in range(init_epoch, epochs):
        if world > 1:
            train_loader.sampler.set_epoch(epoch)
        step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            single_weight,
            step,
            need_labels=not pretrain,
        )
        if main_rank:
            torch.save(unwrap(model).state_dict(), run_dir / "net_latest.pth")
            if hypes["train_params"].get("keep_epochs"):
                # every epoch's weights, for a later re-pick
                torch.save(
                    unwrap(model).state_dict(),
                    run_dir / f"net_epoch{epoch + 1}.pth",
                )
        if main_rank and (epoch % eval_freq == 0 or epoch == epochs - 1):
            # rank 0 validates alone: through the DDP wrapper every forward
            # would broadcast buffers to ranks that are already waiting in
            # the barrier below, which pairs mismatched collectives (a
            # run hangs there after epoch 0); the bare module issues none
            bare = unwrap(model)
            metrics = validation_loss(
                bare,
                val_loader,
                criterion,
                device,
                epoch,
                args.val_batches,
                need_labels=not pretrain,
            )
            if ap_loader is not None:
                metrics.update(
                    bev_ap(
                        bare,
                        ap_loader,
                        post,
                        anchors,
                        order,
                        device,
                        epoch,
                        class_anchors,
                        hypes.get("v2b_box_filter"),
                    )
                )
            wandb.log(metrics, step=step)
            print(f"[epoch {epoch}] {json.dumps(metrics)}", flush=True)
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                save_bestval(model, run_dir, epoch)
        scheduler.step(epoch)
        # the epoch-end reshuffle of HEAL's dataset, also behind a Subset
        ds = train_loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if hasattr(ds, "reinitialize"):
            ds.reinitialize()
        if world > 1:
            dist.barrier()

    wandb.finish()
    if world > 1:
        dist.destroy_process_group()
    print(f"done; checkpoints in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
