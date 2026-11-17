#!/usr/bin/env python3
"""Ground-transformed variants of a derived (OPV2V-layout) split.

Reads every ``<scenario>/<cav>/<index>.{pcd,yaml}`` of a derived split
and writes the same frames, LiDAR only, with the per-frame ground
estimate of ``v2b/data/ground.py`` applied:

* ``drop``: ground points (within 0.15 m of the fitted ground) removed;
* ``rebase``: z replaced by height above ground minus 1.9 m (and the
  sensor pose moved to that virtual origin, so cooperative splits keep
  their agents on one floor), so the
  ground sits at -1.9 m everywhere, the boxes' z moved the same way;
* ``both``: drop, then rebase;
* ``floor1``: ground points replaced by one point per 0.4 m cell at
  their median height (``floor_points``), the floor at its minimal
  footprint in the pillar grid;
* ``hag``: nothing removed or moved, a fifth point channel with the
  height above ground is appended (needs ``point_channels: 5`` in the
  spec), so the pillars see both the raw z and the normalised one.

The experiment behind it: does the pillar encoder lose pedestrians
because ground points crowd the pillars, or because sloping ground
moves the person out of the anchor's height band? The variants are
derived data like their source (``derived/<name>-<mode>``),
registered as ``<name>-<mode>`` datasets, and trained with the same
spec as the source.

    python3 v2b/scripts/v2b-derive-ground.py \\
        --src /path/to/derived/waymo/validate \\
        --out /path/to/derived/waymo-drop/validate \\
        --mode drop --shard 0/4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.data import ground  # noqa: E402
from v2b.data.ground import (  # noqa: E402
    SURFACES,
    fit_surface,
    floor_points,
    ground_mask,
    rebase,
    rebase_z,
    rebased_pose,
)
from v2b.data.opv2v_layout import (  # noqa: E402
    FrameRecord,
    matrix_from_pose,
    objects_from_yaml,
    read_pcd,
    write_frame,
)


class ErrorStats:
    """Height error of a surface against the ``hag`` channel of a
    ``--mode hag`` source: per point ``pred - (z - hag)``, binned by
    range (0 to 20, 20 to 40, beyond 40 m) and by whether the point is a
    ground return (``|hag| < tol``); mean absolute error and the 95th
    percentile from a 1 cm histogram of ``|err|`` up to 5 m."""

    RANGES = ((0.0, 20.0), (20.0, 40.0), (40.0, 1e9))
    BIN_M = 0.01
    TOP_M = 5.0

    def __init__(self, tol: float) -> None:
        self.tol = float(tol)
        n = int(self.TOP_M / self.BIN_M) + 1
        self.hist = np.zeros((len(self.RANGES), 2, n), np.int64)
        self.sum_abs = np.zeros((len(self.RANGES), 2), np.float64)
        self.sum_err = np.zeros((len(self.RANGES), 2), np.float64)
        self.count = np.zeros((len(self.RANGES), 2), np.int64)
        self.frames = 0

    def add(self, pts: Any, field: Any) -> None:
        target = pts[:, 2] - pts[:, 4]
        err = field.at(pts[:, :2].astype(np.float64)) - target
        rng = np.hypot(pts[:, 0], pts[:, 1])
        ground = np.abs(pts[:, 4]) < self.tol
        bins = np.minimum(
            (np.abs(err) / self.BIN_M).astype(int), self.hist.shape[2] - 1
        )
        for r, (lo, hi) in enumerate(self.RANGES):
            sel_r = (rng >= lo) & (rng < hi)
            for g in (0, 1):
                sel = sel_r & (ground if g else ~ground)
                if not sel.any():
                    continue
                self.hist[r, g] += np.bincount(
                    bins[sel], minlength=self.hist.shape[2]
                )
                self.sum_abs[r, g] += float(np.abs(err[sel]).sum())
                self.sum_err[r, g] += float(err[sel].sum())
                self.count[r, g] += int(sel.sum())
        self.frames += 1

    def summary(self) -> dict[str, Any]:
        def cell(r: int, g: int) -> dict[str, float | int]:
            n = int(self.count[r, g])
            if n == 0:
                return {"n": 0}
            cum = np.cumsum(self.hist[r, g])
            p95 = int(np.searchsorted(cum, 0.95 * n)) * self.BIN_M
            return {
                "n": n,
                "mae_m": round(float(self.sum_abs[r, g] / n), 4),
                "bias_m": round(float(self.sum_err[r, g] / n), 4),
                "p95_m": round(p95, 3),
            }

        out: dict[str, Any] = {"frames": self.frames, "tol_m": self.tol}
        total_n = int(self.count.sum())
        if total_n:
            cum = np.cumsum(self.hist.sum((0, 1)))
            out["all"] = {
                "n": total_n,
                "mae_m": round(float(self.sum_abs.sum() / total_n), 4),
                "bias_m": round(float(self.sum_err.sum() / total_n), 4),
                "p95_m": round(
                    int(np.searchsorted(cum, 0.95 * total_n)) * self.BIN_M, 3
                ),
            }
        for r, (lo, hi) in enumerate(self.RANGES):
            name = f"{lo:g}-{hi:g}m" if hi < 1e8 else f"{lo:g}m+"
            out[name] = {
                "ground": cell(r, 1),
                "non_ground": cell(r, 0),
            }
        return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="derived split directory")
    ap.add_argument("--out", required=True, help="output split directory")
    ap.add_argument(
        "--mode",
        choices=["drop", "rebase", "both", "hag", "floor1"],
        required=True,
    )
    ap.add_argument("--tol", type=float, default=0.15, help="ground band m")
    ap.add_argument(
        "--surface",
        choices=(*SURFACES, "map", "learned"),
        default="grid",
        help="floor reconstruction: 2 m local planes (grid), a Delaunay "
        "TIN, its alpha shape or a Poisson mesh of the ground returns, "
        "or the floor network of --model (learned)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="with --surface learned: the run directory of the floor "
        "network",
    )
    ap.add_argument(
        "--error-json",
        default=None,
        help="write the height error of the surface against the hag "
        "point channel of the source (a --mode hag split) here, by "
        "range and by ground return",
    )
    ap.add_argument(
        "--alpha", type=float, default=1.0, help="alpha-shape radius m"
    )
    ap.add_argument(
        "--map-floor",
        default=None,
        help="with --surface map: the floor directory of v2b-map-floor.py "
        "(<scenario>.npz, or <scenario>-w<k>.npz windows picked by frame "
        "index)",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=9,
        help="with --surface poisson: the octree depth (Tab. floor rows)",
    )
    ap.add_argument("--shard", default="0/1", help="i/n over scenarios")
    ap.add_argument(
        "--scenes",
        default=None,
        help="comma-separated scene names (without the v2b_ prefix) "
        "to derive; the others of the split are left alone",
    )
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def convert(
    yml: Path,
    src: Path,
    out: Path,
    mode: str,
    tol: float,
    surface: str = "grid",
    alpha: float = 1.0,
    map_floors: list[Any] | None = None,
    depth: int = 9,
    learned: Any | None = None,
    errors: ErrorStats | None = None,
) -> dict[str, int]:
    cav_dir = yml.parent
    scenario, cav = cav_dir.parent.name, cav_dir.name
    doc = yaml.safe_load(yml.read_text())
    pts = read_pcd(yml.with_suffix(".pcd"))
    world_from_lidar = matrix_from_pose(doc["lidar_pose"])
    np.linalg.inv(world_from_lidar)
    objects = objects_from_yaml(doc, world_from_lidar)
    field: Any
    if surface == "learned":
        if learned is None:
            raise SystemExit("--surface learned needs --model")
        field = learned.fit(pts[:, :4])
    elif surface == "map":
        from v2b.data.map_floor import MapSurface, floor_for

        if not map_floors:
            raise SystemExit("--surface map needs --map-floor")
        field = MapSurface(
            floor_for(
                map_floors, float(doc.get("timestamp", 0.0)), int(yml.stem)
            ),
            world_from_lidar,
        )
    else:
        field = fit_surface(pts, surface, alpha, tol, depth)
    if errors is not None and pts.shape[1] > 4:
        errors.add(pts, field)
    if pts.shape[1] > 4 and mode != "hag":
        pts = pts[:, :4]  # a hag source: the teacher channel stays behind
    gmask = ground_mask(pts, field, tol)
    stats = {"points": len(pts), "ground": int(gmask.sum())}
    if mode in ("drop", "both"):
        pts = pts[~gmask]
    if mode == "floor1":
        # the floor as one point per 0.4 m cell at its median height
        pts = floor_points(pts, gmask, cell=0.4)
    if mode == "hag":
        # keep the raw cloud, append height above ground as a fifth channel
        hag = (pts[:, 2] - field.at(pts[:, :2])).astype(np.float32)
        pts = np.concatenate([pts, hag[:, None]], axis=1)
    if mode in ("rebase", "both"):
        for o in objects:
            o.centre = o.centre.copy()
            o.centre[2] = rebase_z(o.centre[:2], float(o.centre[2]), field)
        pts = rebase(pts, field)
        # the pose follows the virtual sensor 1.9 m above its ground, so
        # agents of a cooperative split still agree on where the road is
        world_from_lidar = rebased_pose(world_from_lidar, field)
    frame = FrameRecord(
        scenario=scenario,
        cav=cav,
        index=int(yml.stem),
        timestamp_s=float(doc.get("timestamp", 0.0)),
        points=pts.astype(np.float32),
        world_from_lidar=world_from_lidar,
        objects=objects,
        speed_mps=float(doc.get("ego_speed", 0.0)),
    )
    write_frame(out, frame)
    return stats


def main() -> int:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    i, n = (int(v) for v in args.shard.split("/"))
    scenarios = sorted(p for p in src.iterdir() if p.is_dir())
    if args.scenes:
        wanted = {w for w in args.scenes.split(",") if w}
        scenarios = [
            p for p in scenarios if p.name in wanted or p.name[4:] in wanted
        ]
        missing = (
            wanted
            - {p.name for p in scenarios}
            - {p.name[4:] for p in scenarios}
        )
        if missing:
            raise SystemExit(f"scenes not under {src}: {sorted(missing)}")
    scenarios = scenarios[i::n]
    out.mkdir(parents=True, exist_ok=True)
    learned: Any | None = None
    if args.surface == "learned":
        if not args.model:
            raise SystemExit("--surface learned needs --model <run dir>")
        from v2b.train.floor_infer import LearnedFloor

        learned = LearnedFloor(Path(args.model))
    errors = ErrorStats(args.tol) if args.error_json else None
    t0 = time.time()
    done = 0
    total_pts = total_ground = 0
    for scenario in scenarios:
        floors = None
        if args.surface == "map":
            from v2b.data.map_floor import load_floor

            fdir = Path(args.map_floor)
            windows = sorted(fdir.glob(f"{scenario.name}-w*.npz"))
            whole = fdir / f"{scenario.name}.npz"
            floors = [load_floor(p) for p in (windows or [whole])]
            if not floors:
                raise SystemExit(f"no map floor for {scenario.name} in {fdir}")
        for yml in sorted(scenario.glob("*/*.yaml")):
            st = convert(
                yml,
                src,
                out,
                args.mode,
                args.tol,
                args.surface,
                args.alpha,
                floors,
                args.depth,
                learned,
                errors,
            )
            total_pts += st["points"]
            total_ground += st["ground"]
            done += 1
            if done % 100 == 0:
                print(
                    f"{done} frames, {done / (time.time() - t0):.1f}/s, "
                    f"ground share {total_ground / max(total_pts, 1):.2f}",
                    flush=True,
                )
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break
    summary = {
        "mode": args.mode,
        "surface": args.surface,
        "depth": args.depth,
        "poisson_fallbacks": ground.POISSON_FALLBACKS,
        "tol": args.tol,
        "frames": done,
        "ground_share": round(total_ground / max(total_pts, 1), 4),
        "seconds": round(time.time() - t0, 1),
    }
    tag = "" if args.shard == "0/1" else "-" + args.shard.replace("/", "of")
    (out / f"derive{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if errors is not None:
        report = {"surface": args.surface, "model": args.model}
        report.update(errors.summary())
        Path(args.error_json).write_text(json.dumps(report, indent=1))
        print(json.dumps(report.get("all", report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
