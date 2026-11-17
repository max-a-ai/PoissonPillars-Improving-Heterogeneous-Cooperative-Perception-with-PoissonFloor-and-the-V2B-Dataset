#!/usr/bin/env python3
"""A cooperative split re-based on one shared floor per frame.

The ``-rebase-poisson`` twins re-base each agent's sweep and labels on
that agent's own Poisson floor, so the two agents of a frame do not
share a floor and the fused frame mixes features with different floors
under them. This writes every frame of a cooperative OPV2V-layout split
with **one** Poisson floor fitted on the merged cloud of both agents:
the other agent's raw sweep is brought into the ego's raw frame with
the two raw poses, the floor is fitted on both sweeps together
(``fit_surface`` with the derive's defaults), both sweeps and the ego's
raw labels are re-based with it, and both agents get the ego's re-based
pose and one label file, so the fused loader's projection between the
agents is the identity. ``v2b-common-floor.py`` does
this for fixed agent ids; this one takes
the agents, the splits and the scenes as arguments and runs the scenes
in parallel::

    .venv/bin/python v2b/scripts/v2b-shared-floor.py \\
        --raw tumtraf-seq-opv2v --ego 1 --other 2 \\
        --out resources/data/generated/tumtraf-seq-opv2v-rebase-poisson-merged

The dataset is then registered (``MIXABLE``, ``config-global.json``,
the ``resources/data/<name>`` link) like any derived dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.data.ground import (  # noqa: E402
    fit_surface,
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

SPLITS = ("train", "validate", "test")
DATA = REPO / "resources" / "data"


def convert_frame(
    raw: Path,
    out: Path,
    scene: str,
    key: str,
    ego: str,
    other: str,
    depth: int,
    tol: float,
) -> dict[str, float]:
    """One frame, both agents on one floor fitted on both sweeps."""
    ego_raw = yaml.safe_load((raw / scene / ego / f"{key}.yaml").read_text())
    oth_raw = yaml.safe_load((raw / scene / other / f"{key}.yaml").read_text())
    w_ego = matrix_from_pose(ego_raw["lidar_pose"])
    w_oth = matrix_from_pose(oth_raw["lidar_pose"])
    pts = read_pcd(raw / scene / other / f"{key}.pcd")
    homo = np.c_[pts[:, :3], np.ones(len(pts))]
    in_ego = (np.linalg.inv(w_ego) @ w_oth @ homo.T).T[:, :3]
    oth_pts = np.c_[in_ego, pts[:, 3:4]].astype(np.float32)
    ego_pts = read_pcd(raw / scene / ego / f"{key}.pcd")
    both = np.concatenate([ego_pts[:, :4], oth_pts[:, :4]], axis=0)
    field = fit_surface(both, "poisson", tol=tol, depth=depth)
    pose = rebased_pose(w_ego, field)
    objects = objects_from_yaml(ego_raw, w_ego)
    for o in objects:
        o.centre = o.centre.copy()
        o.centre[2] = rebase_z(o.centre[:2], float(o.centre[2]), field)
    n = 0
    for cav, points, doc in (
        (ego, rebase(ego_pts, field), ego_raw),
        (other, rebase(oth_pts, field), oth_raw),
    ):
        write_frame(
            out,
            FrameRecord(
                scenario=scene,
                cav=cav,
                index=int(key),
                timestamp_s=float(doc.get("timestamp", 0.0)),
                points=points,
                world_from_lidar=pose,
                objects=objects,
                speed_mps=float(doc.get("ego_speed", 0.0)),
            ),
        )
        n += len(points)
    return {
        "points": float(n),
        "ego_shift": float(pose[2, 3] - w_ego[2, 3]),
    }


def convert_scene(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    raw, out, scene, ego, other, depth, tol, limit = args
    keys = sorted(p.stem for p in (raw / scene / ego).glob("*.yaml"))
    if limit:
        keys = keys[:limit]
    stats = []
    for key in keys:
        s = convert_frame(raw, out, scene, key, ego, other, depth, tol)
        s.update({"scene": scene, "key": key})
        stats.append(s)
    print(f"{scene}: {len(keys)} frames", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw", required=True, help="dataset under resources/data"
    )
    ap.add_argument("--ego", default="0", help="agent folder of the ego")
    ap.add_argument("--other", default="1", help="agent folder of the other")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument(
        "--scenes", nargs="+", default=None, help="default: every scene"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    jobs = []
    for split in args.splits:
        raw = DATA / args.raw / split
        if not raw.is_dir():
            continue
        scenes = args.scenes or sorted(
            p.name for p in raw.iterdir() if p.is_dir()
        )
        for scene in scenes:
            if not (raw / scene / args.ego).is_dir():
                continue
            jobs.append(
                (
                    raw,
                    out / split,
                    scene,
                    args.ego,
                    args.other,
                    args.depth,
                    args.tol,
                    args.limit,
                )
            )
    t0 = time.time()
    stats: list[dict[str, Any]] = []
    # not a Pool: the Poisson fit forks a child per sweep
    with ProcessPoolExecutor(
        args.workers, mp_context=get_context("fork")
    ) as ex:
        for part in ex.map(convert_scene, jobs):
            stats.extend(part)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shared-floor.json").write_text(json.dumps(stats, indent=1))
    shifts = [abs(s["ego_shift"]) for s in stats]
    dt = time.time() - t0
    print(
        f"{len(stats)} frames of {len(jobs)} scenes in {dt:.0f} s; ego "
        f"pose shift by the shared floor: median {np.median(shifts):.3f} m, "
        f"max {max(shifts):.3f} m"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
