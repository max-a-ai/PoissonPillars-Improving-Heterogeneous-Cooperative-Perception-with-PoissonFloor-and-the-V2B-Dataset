"""AP@BEV for rotated boxes, numpy only (no shapely in the HEAL venv).

Matching follows HEAL's ``eval_utils``: per frame, detections in score
order take the unmatched ground-truth box of highest BEV IoU, a hit
when that IoU reaches the threshold; AP is the all-point interpolated
area under precision-recall (VOC 2010+), as in ``calculate_ap``.

Ground-truth boxes flagged ``ignore`` (parked two-wheelers here) are
neither hits nor misses: a detection overlapping one by IoU >= 0.1 is
dropped before matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def _clip(subject: F64, a: F64, b: F64) -> F64:
    """Sutherland-Hodgman: the part of ``subject`` left of edge a->b."""
    if len(subject) == 0:
        return subject
    ab = b - a

    def side(p: F64) -> float:
        return float(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0]))

    out = []
    for i in range(len(subject)):
        p, q = subject[i], subject[(i + 1) % len(subject)]
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            t = sp / (sp - sq)
            out.append(p + t * (q - p))
    return np.array(out, float).reshape(-1, 2)


def _area(poly: F64) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(
        0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def _ccw(poly: F64) -> F64:
    x, y = poly[:, 0], poly[:, 1]
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if signed >= 0 else poly[::-1]


def iou(p: F64, q: F64) -> float:
    """IoU of two convex polygons ``(k, 2)``."""
    p, q = _ccw(np.asarray(p, float)), _ccw(np.asarray(q, float))
    # cheap reject on the circumscribed circles
    if np.linalg.norm(p.mean(0) - q.mean(0)) > (
        np.linalg.norm(p - p.mean(0), axis=1).max()
        + np.linalg.norm(q - q.mean(0), axis=1).max()
    ):
        return 0.0
    inter = p
    for i in range(len(q)):
        inter = _clip(inter, q[i], q[(i + 1) % len(q)])
        if len(inter) == 0:
            return 0.0
    a = _area(inter)
    union = _area(p) + _area(q) - a
    return a / union if union > 0 else 0.0


def iou_matrix(dets: F64, gts: F64) -> F64:
    """``(D, G)`` BEV IoU of ``(D, 4, 2)`` against ``(G, 4, 2)``."""
    out = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            out[i, j] = iou(d, g)
    return out


def nms(corners: F64, scores: F64, threshold: float) -> list[int]:
    """Indices kept by greedy rotated NMS, best score first."""
    keep: list[int] = []
    for i in np.argsort(-scores):
        if all(iou(corners[i], corners[k]) < threshold for k in keep):
            keep.append(int(i))
    return keep


@dataclass
class ApAccumulator:
    """Per-class hits over frames, for one detector and one threshold."""

    threshold: float
    scores: list[float] = field(default_factory=list)
    hits: list[bool] = field(default_factory=list)
    n_gt: int = 0

    def add(
        self,
        det: F64,
        score: F64,
        gt: F64,
        ignore: F64 | None = None,
    ) -> None:
        """One frame: ``det`` ``(D, 4, 2)``, ``gt`` ``(G, 4, 2)``."""
        det = np.asarray(det, float).reshape(-1, 4, 2)
        score = np.asarray(score, float).reshape(-1)
        gt = np.asarray(gt, float).reshape(-1, 4, 2)
        if ignore is not None and len(ignore) and len(det):
            over = iou_matrix(det, ignore).max(axis=1) >= 0.1
            det, score = det[~over], score[~over]
        self.n_gt += len(gt)
        if len(det) == 0:
            return
        ious = iou_matrix(det, gt) if len(gt) else np.zeros((len(det), 0))
        taken = np.zeros(len(gt), bool)
        for i in np.argsort(-score):
            hit = False
            if len(gt):
                cand = np.where(taken, -1.0, ious[i])
                j = int(np.argmax(cand))
                if cand[j] >= self.threshold:
                    taken[j] = hit = True
            self.scores.append(float(score[i]))
            self.hits.append(hit)

    def ap(self) -> float:
        if self.n_gt == 0:
            return float("nan")
        if not self.scores:
            return 0.0
        order = np.argsort(-np.asarray(self.scores))
        tp = np.asarray(self.hits, float)[order]
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = np.concatenate([[0.0], ctp / self.n_gt, [1.0]])
        prec = np.concatenate([[0.0], ctp / (ctp + cfp), [0.0]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        idx = np.flatnonzero(rec[1:] != rec[:-1]) + 1
        return float(np.sum((rec[idx] - rec[idx - 1]) * prec[idx]))

    def recall(self) -> float:
        return sum(self.hits) / self.n_gt if self.n_gt else float("nan")
