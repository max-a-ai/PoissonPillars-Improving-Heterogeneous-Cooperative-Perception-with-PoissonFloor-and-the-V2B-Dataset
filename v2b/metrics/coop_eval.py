"""Scoring of the cooperative V2B settings, numpy only and deterministic.

One frame is a set of labelled boxes in the car's LiDAR frame with the
car's and the bike's own point counts per box, and a set of detected
boxes in the same frame (the bike's and the fused detections are
brought into the car frame by their SLAM poses before they get here).
Every setting (car only, bike only, late fusion, intermediate fusion)
is scored against the same labels, in the same frame and the same
range, which is what makes the four rows of the title claim comparable.

**Targets.** A labelled box is a target when the repo's rule
(:func:`v2b.data.v2b_nusc.visible_to`: at least ``min_points`` own
points of an agent, or the single-agent floor when no other agent has
a point in it) makes it a target of the car or of the bike. Boxes in
range that are no agent's target, boxes flagged ``ignore`` (parked
two-wheelers) and boxes of class ``other`` are *ignore regions*: a
detection overlapping one by IoU >= 0.1 is dropped before matching,
so they are neither hits nor misses.

**Strata.** Besides ``all``, the targets are split by who sees them
(``vis:both``, ``vis:car_only``, ``vis:bike_only``) and by distance
from the car (``dist:0-20``, ``dist:20-40``, ``dist:40+``); in a
stratum the other targets become ignore regions. The time offset
between the two agents' sweeps is a per-frame property, so the ``dt``
strata are subsets of frames scored on the ``all`` targets.

**Determinism.** Matching is greedy in descending score with stable
sorts, AP is the all-point interpolated area under precision-recall
(as HEAL's ``calculate_ap``), and the bootstrap draws scenes with a
fixed seed; the same predictions file re-scores to the same digits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from v2b.data.v2b_nusc import bev_corners, visible_to, yaw_of
from v2b.metrics.bev_ap import iou_matrix, nms

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]

IOUS: tuple[float, ...] = (0.3, 0.5, 0.7)
CLASSES: tuple[str, ...] = ("car", "bicycle", "pedestrian")
AGENTS: tuple[str, ...] = ("car", "bike")
#: Distance bins from the car (m) and sweep offsets (ms): ``[lo, hi)``.
DIST_BINS: tuple[tuple[float, float], ...] = ((0, 20), (20, 40), (40, np.inf))
DT_BINS_MS: tuple[tuple[float, float], ...] = (
    (0, 25),
    (25, 50),
    (50, np.inf),
)
#: A detection on an ignore region is dropped from this IoU on.
IGNORE_IOU = 0.1


def bin_name(prefix: str, lo: float, hi: float) -> str:
    return (
        f"{prefix}:{lo:g}-{hi:g}" if np.isfinite(hi) else f"{prefix}:{lo:g}+"
    )


@dataclass
class FrameLabels:
    """The labelled boxes of one frame, car LiDAR frame."""

    scene: str
    frame: str
    boxes: F64  # (G, 7) x y z l w h yaw
    names: list[str]
    ignore: B  # (G,) parked two-wheelers
    counts: dict[str, I64]  # own points per agent, (G,) each
    dt_ms: float = 0.0  # |t_car - t_bike| of the two sweeps
    agent_gap_m: float = 0.0  # distance between the two sensors
    #: the bike's position in the car LiDAR frame (x, y); None when the
    #: predictions predate it (then the dual-radius mask is off)
    bike_xy: tuple[float, float] | None = None

    def within_radius(self, max_range: float | None) -> B:
        """Boxes within ``max_range`` of the car and of the bike in the
        ground plane: the labelling contract's dual radius. All True
        without a radius or without the bike's position."""
        n = len(self.boxes)
        if max_range is None or self.bike_xy is None:
            return np.ones(n, bool)
        bx, by = self.bike_xy
        if not np.isfinite(bx) or not np.isfinite(by):
            return np.ones(n, bool)
        d_car = np.hypot(self.boxes[:, 0], self.boxes[:, 1])
        d_bike = np.hypot(self.boxes[:, 0] - bx, self.boxes[:, 1] - by)
        out: B = np.maximum(d_car, d_bike) <= max_range
        return out

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, float).reshape(-1, 7)
        self.ignore = np.asarray(self.ignore, bool).reshape(-1)
        self.counts = {
            a: np.asarray(c, np.int64).reshape(-1)
            for a, c in self.counts.items()
        }


@dataclass
class FrameDetections:
    """The detections of one frame, car LiDAR frame, after NMS."""

    boxes: F64  # (D, 7)
    scores: F64  # (D,)
    names: list[str]
    agents: list[str]  # car / bike / fused

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, float).reshape(-1, 7)
        self.scores = np.asarray(self.scores, float).reshape(-1)


def transform_boxes(boxes: F64, tf: F64) -> F64:
    """``(N, 7)`` boxes through a 4x4: centres moved, yaw turned by the
    transform's heading (its x axis in the ground plane)."""
    out = np.asarray(boxes, float).reshape(-1, 7).copy()
    if len(out) == 0:
        return out
    xyz1 = np.concatenate([out[:, :3], np.ones((len(out), 1))], axis=1)
    out[:, :3] = (xyz1 @ tf.T)[:, :3]
    out[:, 6] = out[:, 6] + yaw_of(tf)
    return out


def corners_of(boxes: F64) -> F64:
    """``(N, 4, 2)`` BEV corners of ``(N, 7)`` boxes."""
    b = np.asarray(boxes, float).reshape(-1, 7)
    if len(b) == 0:
        return np.zeros((0, 4, 2))
    return np.stack(
        [bev_corners(r[:3], float(r[3]), float(r[4]), float(r[6])) for r in b]
    )


def in_range(boxes: F64, lidar_range: list[float] | F64) -> B:
    """Centres inside the grid's x / y extent."""
    b = np.asarray(boxes, float).reshape(-1, 7)
    r = [float(v) for v in lidar_range]
    out: B = (
        (b[:, 0] >= r[0])
        & (b[:, 0] < r[3])
        & (b[:, 1] >= r[1])
        & (b[:, 1] < r[4])
    )
    return out


def late_fuse(
    parts: list[FrameDetections], threshold: float
) -> FrameDetections:
    """Boxes of several agents in one frame, merged by greedy rotated
    NMS at ``threshold`` (class-agnostic, as HEAL's own NMS). Ties in
    score are broken by the input order, so the result is a function
    of the inputs alone."""
    boxes = np.concatenate([p.boxes for p in parts], axis=0)
    scores = np.concatenate([p.scores for p in parts], axis=0)
    names = [n for p in parts for n in p.names]
    agents = [a for p in parts for a in p.agents]
    if len(boxes) == 0:
        return FrameDetections(boxes, scores, [], [])
    order = np.lexsort((np.arange(len(scores)), -scores))
    boxes, scores = boxes[order], scores[order]
    names = [names[i] for i in order]
    agents = [agents[i] for i in order]
    keep = sorted(nms(corners_of(boxes), scores, threshold))
    return FrameDetections(
        boxes[keep],
        scores[keep],
        [names[i] for i in keep],
        [agents[i] for i in keep],
    )


def target_sets(
    labels: FrameLabels, min_points: int, single_floor: int
) -> dict[str, B]:
    """Which boxes are targets of each agent under the repo's rule."""
    counts = {a: [int(v) for v in labels.counts[a]] for a in AGENTS}
    return {
        a: np.asarray(visible_to(counts, a, min_points, single_floor), bool)
        for a in AGENTS
    }


def strata_of(
    labels: FrameLabels,
    targets: dict[str, B],
    lidar_range: list[float],
    max_range: float | None = None,
) -> dict[str, B]:
    """Target subsets per stratum name; the complement of a subset
    among the in-range boxes is that stratum's ignore set. With
    ``max_range`` a box beyond the dual radius is neither (the labels
    end there)."""
    car, bike = targets["car"], targets["bike"]
    usable = (
        in_range(labels.boxes, lidar_range)
        & labels.within_radius(max_range)
        & ~labels.ignore
        & np.asarray([n in CLASSES for n in labels.names], bool)
    )
    base = usable & (car | bike)
    dist = np.hypot(labels.boxes[:, 0], labels.boxes[:, 1])
    out: dict[str, B] = {
        "all": base,
        "vis:both": base & car & bike,
        "vis:car_only": base & car & ~bike,
        "vis:bike_only": base & bike & ~car,
    }
    for lo, hi in DIST_BINS:
        out[bin_name("dist", lo, hi)] = base & (dist >= lo) & (dist < hi)
    return out


def match_frame(
    det: F64, score: F64, gt: F64, ignore: F64, threshold: float
) -> tuple[F64, B]:
    """Greedy matching of one frame, one class, one IoU threshold.

    ``det``, ``gt`` and ``ignore`` are ``(N, 4, 2)`` corner sets; the
    detections are taken in descending score (stable), each claims the
    unmatched ground-truth box of highest IoU when that IoU reaches
    the threshold. Returns the kept detections' scores and hit flags.
    """
    if len(det) == 0:
        return np.zeros(0), np.zeros(0, bool)
    if len(ignore):
        over = iou_matrix(det, ignore).max(axis=1) >= IGNORE_IOU
        det, score = det[~over], score[~over]
        if len(det) == 0:
            return np.zeros(0), np.zeros(0, bool)
    order = np.argsort(-score, kind="stable")
    ious = iou_matrix(det, gt) if len(gt) else np.zeros((len(det), 0))
    taken = np.zeros(len(gt), bool)
    hits = np.zeros(len(det), bool)
    for i in order:
        if len(gt) == 0:
            break
        cand = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= threshold:
            taken[j] = hits[i] = True
    return score[order], hits[order]


def average_precision(scores: F64, hits: B, n_gt: int) -> float:
    """All-point interpolated AP (VOC 2010+, HEAL's ``calculate_ap``)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.asarray(hits, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = np.concatenate([[0.0], ctp / n_gt, [1.0]])
    prec = np.concatenate([[0.0], ctp / (ctp + cfp), [0.0]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.flatnonzero(rec[1:] != rec[:-1]) + 1
    return float(np.sum((rec[idx] - rec[idx - 1]) * prec[idx]))


@dataclass
class Cell:
    """One frame's matches for one (stratum, class, IoU)."""

    scores: F64
    hits: B
    n_gt: int


@dataclass
class FrameResult:
    scene: str
    frame: str
    dt_ms: float
    cells: dict[tuple[str, str, float], Cell] = field(default_factory=dict)
    n_det: dict[str, int] = field(default_factory=dict)


def score_frame(
    labels: FrameLabels,
    dets: FrameDetections,
    lidar_range: list[float],
    min_points: int,
    single_floor: int,
    classes: tuple[str, ...] = CLASSES,
    ious: tuple[float, ...] = IOUS,
    max_range: float | None = None,
) -> FrameResult:
    """Every stratum, class and IoU of one frame. ``max_range`` masks
    the labels' dual radius: a detection beyond it is dropped, a box
    beyond it is neither target nor ignore region."""
    targets = target_sets(labels, min_points, single_floor)
    strata = strata_of(labels, targets, lidar_range, max_range)
    usable_range = in_range(labels.boxes, lidar_range) & labels.within_radius(
        max_range
    )
    gt_c = corners_of(labels.boxes)
    gt_names = np.asarray(labels.names, str)
    keep_det = in_range(dets.boxes, lidar_range)
    if max_range is not None and labels.bike_xy is not None:
        keep_det &= FrameLabels(
            labels.scene,
            labels.frame,
            dets.boxes,
            list(dets.names),
            np.zeros(len(dets.boxes), bool),
            {},
            bike_xy=labels.bike_xy,
        ).within_radius(max_range)
    det_c = corners_of(dets.boxes[keep_det])
    det_s = dets.scores[keep_det]
    det_names = np.asarray(dets.names, str)[keep_det]
    out = FrameResult(labels.scene, labels.frame, labels.dt_ms)
    for c in classes:
        out.n_det[c] = int((det_names == c).sum())
    for name, subset in strata.items():
        # everything in range that is not a target of this stratum is an
        # ignore region for every class (an unlabelled-for-us object)
        ignore_all = usable_range & ~subset
        for c in classes:
            d = det_c[det_names == c]
            s = det_s[det_names == c]
            g = gt_c[subset & (gt_names == c)]
            ign = gt_c[ignore_all]
            for thr in ious:
                sc, hit = match_frame(d, s, g, ign, thr)
                out.cells[(name, c, thr)] = Cell(sc, hit, int(len(g)))
    return out


def _pool(frames: list[FrameResult], key: tuple[str, str, float]) -> Cell:
    parts = [f.cells[key] for f in frames if key in f.cells]
    if not parts:
        return Cell(np.zeros(0), np.zeros(0, bool), 0)
    return Cell(
        np.concatenate([p.scores for p in parts]),
        np.concatenate([p.hits for p in parts]),
        int(sum(p.n_gt for p in parts)),
    )


def _table(
    frames: list[FrameResult],
    strata: list[str],
    classes: tuple[str, ...],
    ious: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in strata:
        row: dict[str, Any] = {}
        for c in classes:
            cell: dict[str, Any] = {}
            for thr in ious:
                pooled = _pool(frames, (s, c, thr))
                cell[f"ap@{thr:g}"] = average_precision(
                    pooled.scores, pooled.hits, pooled.n_gt
                )
                cell["n_gt"] = pooled.n_gt
            row[c] = cell
        for thr in ious:
            aps = [row[c][f"ap@{thr:g}"] for c in classes if row[c]["n_gt"]]
            row[f"map@{thr:g}"] = float(np.mean(aps)) if aps else float("nan")
        out[s] = row
    return out


def _dt_strata(frames: list[FrameResult]) -> dict[str, list[FrameResult]]:
    out: dict[str, list[FrameResult]] = {}
    for lo, hi in DT_BINS_MS:
        out[bin_name("dt", lo, hi)] = [f for f in frames if lo <= f.dt_ms < hi]
    return out


def bootstrap(
    frames: list[FrameResult],
    classes: tuple[str, ...],
    ious: tuple[float, ...],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """95 % intervals of the ``all`` APs over scenes drawn with
    replacement; with one scene the frames are the unit instead, and
    the result says so."""
    scenes = sorted({f.scene for f in frames})
    unit = "scene" if len(scenes) > 1 else "frame"
    groups = (
        [[f for f in frames if f.scene == s] for s in scenes]
        if unit == "scene"
        else [[f] for f in frames]
    )
    rng = np.random.default_rng(seed)
    keys = [(c, thr) for c in classes for thr in ious]
    samples: dict[tuple[str, float], list[float]] = {k: [] for k in keys}
    for _ in range(draws):
        pick = rng.integers(0, len(groups), size=len(groups))
        drawn = [f for i in pick for f in groups[i]]
        for c, thr in keys:
            pooled = _pool(drawn, ("all", c, thr))
            samples[(c, thr)].append(
                average_precision(pooled.scores, pooled.hits, pooled.n_gt)
            )
    out: dict[str, Any] = {
        "unit": unit,
        "n_units": len(groups),
        "draws": draws,
        "seed": seed,
    }
    for c in classes:
        out[c] = {}
        for thr in ious:
            arr = np.asarray(samples[(c, thr)], float)
            arr = arr[np.isfinite(arr)]
            out[c][f"ap@{thr:g}"] = (
                [float(v) for v in np.percentile(arr, [2.5, 97.5])]
                if len(arr)
                else [float("nan"), float("nan")]
            )
    return out


def summarise(
    frames: list[FrameResult],
    classes: tuple[str, ...] = CLASSES,
    ious: tuple[float, ...] = IOUS,
    draws: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The result tables of one setting: ``ap[stratum][class]``, the
    ``dt`` strata, the bootstrap intervals and the frame counts."""
    strata = sorted(
        {k[0] for f in frames for k in f.cells}, key=_stratum_order
    )
    table = _table(frames, strata, classes, ious)
    for name, sub in _dt_strata(frames).items():
        row = _table(sub, ["all"], classes, ious)["all"] if sub else {}
        row["n_frames"] = len(sub)
        table[name] = row
    return {
        "frames": len(frames),
        "scenes": sorted({f.scene for f in frames}),
        "n_det": {
            c: int(sum(f.n_det.get(c, 0) for f in frames)) for c in classes
        },
        "ap": table,
        "ci95": bootstrap(frames, classes, ious, draws, seed)
        if frames
        else {},
    }


def _stratum_order(name: str) -> tuple[int, str]:
    prefix = name.split(":")[0]
    return ({"all": 0, "vis": 1, "dist": 2, "dt": 3}.get(prefix, 9), name)


# ------------------------------------------------------------ persistence
def save_predictions(
    path: Path,
    labels: list[FrameLabels],
    dets: list[FrameDetections],
    meta: dict[str, Any],
) -> None:
    """One compressed npz with every frame's labels and detections, so
    a re-score needs no GPU and no data."""
    n_gt = np.array([len(x.boxes) for x in labels], np.int64)
    n_det = np.array([len(x.boxes) for x in dets], np.int64)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        scene=np.array([x.scene for x in labels], str),
        frame=np.array([x.frame for x in labels], str),
        dt_ms=np.array([x.dt_ms for x in labels], float),
        agent_gap_m=np.array([x.agent_gap_m for x in labels], float),
        bike_xy=np.array(
            [
                x.bike_xy if x.bike_xy is not None else (np.nan, np.nan)
                for x in labels
            ],
            float,
        ).reshape(-1, 2),
        n_gt=n_gt,
        n_det=n_det,
        gt_boxes=np.concatenate([x.boxes for x in labels]).reshape(-1, 7),
        gt_names=np.array([n for x in labels for n in x.names], str),
        gt_ignore=np.concatenate([x.ignore for x in labels]).astype(bool),
        gt_count_car=np.concatenate([x.counts["car"] for x in labels]),
        gt_count_bike=np.concatenate([x.counts["bike"] for x in labels]),
        det_boxes=np.concatenate([x.boxes for x in dets]).reshape(-1, 7),
        det_scores=np.concatenate([x.scores for x in dets]),
        det_names=np.array([n for x in dets for n in x.names], str),
        det_agents=np.array([a for x in dets for a in x.agents], str),
    )


def load_predictions(
    path: Path,
) -> tuple[list[FrameLabels], list[FrameDetections], dict[str, Any]]:
    """The inverse of :func:`save_predictions`."""
    z = np.load(path, allow_pickle=False)
    meta: dict[str, Any] = json.loads(str(z["meta"]))
    g_off = np.concatenate([[0], np.cumsum(z["n_gt"])])
    d_off = np.concatenate([[0], np.cumsum(z["n_det"])])
    labels: list[FrameLabels] = []
    dets: list[FrameDetections] = []
    for i in range(len(z["scene"])):
        a, b = int(g_off[i]), int(g_off[i + 1])
        labels.append(
            FrameLabels(
                scene=str(z["scene"][i]),
                frame=str(z["frame"][i]),
                boxes=z["gt_boxes"][a:b],
                names=[str(n) for n in z["gt_names"][a:b]],
                ignore=z["gt_ignore"][a:b],
                counts={
                    "car": z["gt_count_car"][a:b],
                    "bike": z["gt_count_bike"][a:b],
                },
                dt_ms=float(z["dt_ms"][i]),
                agent_gap_m=float(z["agent_gap_m"][i]),
                bike_xy=(
                    (float(z["bike_xy"][i][0]), float(z["bike_xy"][i][1]))
                    if "bike_xy" in z.files
                    and np.all(np.isfinite(z["bike_xy"][i]))
                    else None
                ),
            )
        )
        a, b = int(d_off[i]), int(d_off[i + 1])
        dets.append(
            FrameDetections(
                boxes=z["det_boxes"][a:b],
                scores=z["det_scores"][a:b],
                names=[str(n) for n in z["det_names"][a:b]],
                agents=[str(n) for n in z["det_agents"][a:b]],
            )
        )
    return labels, dets, meta


def score_run(
    labels: list[FrameLabels],
    dets: list[FrameDetections],
    lidar_range: list[float],
    min_points: int = 10,
    single_floor: int = 1,
    draws: int = 1000,
    seed: int = 0,
    max_range: float | None = None,
) -> dict[str, Any]:
    """Score every frame and summarise; the one entry point of a run or
    of a re-score."""
    frames = [
        score_frame(
            lab,
            det,
            lidar_range,
            min_points,
            single_floor,
            max_range=max_range,
        )
        for lab, det in zip(labels, dets, strict=True)
    ]
    out = summarise(frames, draws=draws, seed=seed)
    masked = sum(1 for x in labels if x.bike_xy is not None)
    out["params"] = {
        "lidar_range": [float(v) for v in lidar_range],
        "min_points": min_points,
        "single_floor": single_floor,
        "max_range": max_range,
        "max_range_frames": masked if max_range is not None else 0,
        "ignore_iou": IGNORE_IOU,
        "ious": list(IOUS),
        "dist_bins_m": [list(b) for b in DIST_BINS],
        "dt_bins_ms": [list(b) for b in DT_BINS_MS],
    }
    return out
