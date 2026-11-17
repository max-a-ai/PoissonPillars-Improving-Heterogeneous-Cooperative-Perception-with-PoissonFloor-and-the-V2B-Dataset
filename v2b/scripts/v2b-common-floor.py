#!/usr/bin/env python3
"""Test frames re-based on one common floor for both agents.

The ``-rebase-poisson`` twin re-bases each agent's sweep and labels on
that agent's own Poisson floor, so the two agents of a frame do not
share a floor exactly. This writes the test scenes of a cooperative
split with the **car's** floor applied to both agents: the car's frames
are the production re-based files, copied; the bicycle's raw sweep is
brought into the car's raw frame with the two raw poses, re-based with
a Poisson floor fitted on the car's raw sweep (``v2b-derive-ground.py
--mode rebase --surface poisson``, the same fit), and written with the
car's re-based pose and the car's re-based label file, so the fused
loader's ``proj_first`` projection between the agents is the identity.
Evaluation data only (the test scenes), derived like its source::

    .venv/bin/python v2b/scripts/v2b-common-floor.py \\
        --out resources/data/generated/<name>-carfloor

The dataset is then registered by hand (``MIXABLE``,
``config-global.json``, the ``resources/data/<name>`` link) and scored
with an eval-only spec whose ``sources.test`` names it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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


def scene_split(dataset: str, scene: str) -> str:
    for split in SPLITS:
        if (DATA / dataset / split / scene).is_dir():
            return split
    raise FileNotFoundError(f"{scene} not under {DATA / dataset}")


def convert_frame(
    raw: Path,
    rebased: Path,
    out: Path,
    scene: str,
    key: str,
    depth: int,
    tol: float,
    fit: str = "merged",
) -> dict[str, float]:
    """One frame, both agents on one floor.

    ``fit`` ``merged``: the Poisson floor is fitted on the car's and the
    bicycle's raw sweeps together (the bicycle's brought into the car's
    raw frame), both sweeps and the car's raw labels are re-based with
    it and both agents get the car's re-based pose and one label file.
    ``car``: the car's production files are copied and the bicycle's
    sweep is re-based with a floor fitted on the car's sweep alone (it
    extrapolates far from the car; kept for the record).
    """
    car, bike = "0", "1"
    car_raw = yaml.safe_load((raw / scene / car / f"{key}.yaml").read_text())
    bike_raw = yaml.safe_load((raw / scene / bike / f"{key}.yaml").read_text())
    w_car = matrix_from_pose(car_raw["lidar_pose"])
    w_bike = matrix_from_pose(bike_raw["lidar_pose"])
    # the bicycle's raw sweep in the car's raw frame
    pts = read_pcd(raw / scene / bike / f"{key}.pcd")
    homo = np.c_[pts[:, :3], np.ones(len(pts))]
    in_car = (np.linalg.inv(w_car) @ w_bike @ homo.T).T[:, :3]
    bike_pts = np.c_[in_car, pts[:, 3:4]].astype(np.float32)
    car_pts = read_pcd(raw / scene / car / f"{key}.pcd")
    car_reb = yaml.safe_load(
        (rebased / scene / car / f"{key}.yaml").read_text()
    )
    w_car_reb = matrix_from_pose(car_reb["lidar_pose"])
    if fit == "car":
        dst = out / scene / car
        dst.mkdir(parents=True, exist_ok=True)
        for suffix in (".pcd", ".yaml"):
            shutil.copy2(rebased / scene / car / f"{key}{suffix}", dst)
        field = fit_surface(car_pts, "poisson", tol=tol, depth=depth)
        pose = w_car_reb
        objects = objects_from_yaml(car_reb, w_car_reb)
        frames = [(bike, rebase(bike_pts, field), bike_raw)]
    else:
        both = np.concatenate([car_pts[:, :4], bike_pts[:, :4]], axis=0)
        field = fit_surface(both, "poisson", tol=tol, depth=depth)
        pose = rebased_pose(w_car, field)
        objects = objects_from_yaml(car_raw, w_car)
        for o in objects:
            o.centre = o.centre.copy()
            o.centre[2] = rebase_z(o.centre[:2], float(o.centre[2]), field)
        frames = [
            (car, rebase(car_pts, field), car_raw),
            (bike, rebase(bike_pts, field), bike_raw),
        ]
    # the re-based car pose of this fit against the production one
    pose_diff = float(np.abs(pose[:3, 3] - w_car_reb[:3, 3]).max())
    n = 0
    for cav, points, doc in frames:
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
    return {"points": float(n), "pose_diff": pose_diff}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="v2b-coop-v2-hand-opv2v")
    ap.add_argument(
        "--rebased", default="v2b-coop-v2-hand-opv2v-rebase-poisson"
    )
    ap.add_argument(
        "--scenes", nargs="+", default=["v2b_scene_04", "v2b_scene_13"]
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fit", choices=("merged", "car"), default="merged")
    args = ap.parse_args()
    out = Path(args.out)
    stats: list[dict[str, Any]] = []
    t0 = time.time()
    for scene in args.scenes:
        split = scene_split(args.raw, scene)
        raw = DATA / args.raw / split
        rebased = DATA / args.rebased / split
        keys = sorted(p.stem for p in (raw / scene / "0").glob("*.yaml"))
        if args.limit:
            keys = keys[: args.limit]
        for i, key in enumerate(keys):
            s = convert_frame(
                raw,
                rebased,
                out / split,
                scene,
                key,
                args.depth,
                args.tol,
                args.fit,
            )
            s.update({"scene": scene, "key": key})
            stats.append(s)
            if i % 20 == 0:
                print(
                    f"{scene} {key}: {int(s['points'])} points, pose "
                    f"refit vs production {s['pose_diff']:.3f} m, "
                    f"{time.time() - t0:.0f} s",
                    flush=True,
                )
    (out / "common-floor.json").write_text(json.dumps(stats, indent=1))
    worst = max(s["pose_diff"] for s in stats) if stats else 0.0
    print(f"{len(stats)} frames, worst pose refit difference {worst:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
