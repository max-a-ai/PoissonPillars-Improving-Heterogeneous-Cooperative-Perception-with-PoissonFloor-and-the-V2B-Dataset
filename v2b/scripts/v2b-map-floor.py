#!/usr/bin/env python3
"""Build the floor once per scene from an OPV2V-layout split, and time
the per-point lookup that replaces the per-sweep fit.

``build`` reads every agent's sweeps of every scenario folder, takes
each sweep's ground returns (the robust 2 m grid fit and its 0.15 m
band, as the per-sweep path does), pools them in the map frame through
the yaml poses, thins them to the lowest per 0.25 m column and fits one
screened Poisson surface per scenario, written as
``<out>/<scenario>.npz``. With ``--leave-out M`` it also writes
``<out>/<scenario>-w<k>.npz`` per window of ``--window`` seconds, each
built without the frames within ``M`` seconds of that window, for the
"map built beforehand" question. ``v2b-derive-ground.py --mode rebase
--surface map --map-floor <out>`` then re-bases a split on these floors.

``time`` loads a floor and ``--frames`` sweeps of the scenario and
reports the median per-sweep cost of the lookup re-base on one CPU
thread (and on the GPU when torch sees one), next to the point count.

    .venv/bin/python v2b/scripts/v2b-map-floor.py build \\
        --src /path/to/derived/waymo/validate \\
        --out /path/to/derived/map-floors/waymo/validate \\
        [--shard i/n] [--leave-out 5 --window 2]
    .venv/bin/python v2b/scripts/v2b-map-floor.py time \\
        --src .../v2b-coop-v2-opv2v/train \\
        --floors .../map-floors/v2b-coop-v2/train \\
        --scenario v2b_scene_11 --frames 100
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.data.map_floor import (  # noqa: E402
    MapFloor,
    MapSurface,
    build_floor,
    load_floor,
    save_floor,
    seeds_of_sweep,
)
from v2b.data.opv2v_layout import matrix_from_pose, read_pcd  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True, help="split directory")
    b.add_argument("--out", required=True, help="floor directory")
    b.add_argument("--shard", default="0/1", help="i/n over scenarios")
    b.add_argument("--stride", type=int, default=1, help="every n-th frame")
    b.add_argument(
        "--leave-out",
        type=float,
        default=None,
        metavar="M",
        help="also build per-window floors without the frames within M s",
    )
    b.add_argument("--window", type=float, default=2.0, help="seconds")
    t = sub.add_parser("time")
    t.add_argument("--src", required=True)
    t.add_argument("--floors", required=True)
    t.add_argument("--scenario", required=True)
    t.add_argument("--agent", default=None)
    t.add_argument("--frames", type=int, default=100)
    t.add_argument("--out", default=None, help="json")
    return ap.parse_args()


def sweeps_of(
    scenario: Path, stride: int
) -> list[tuple[Path, float, int, Any]]:
    """(pcd path, timestamp, frame index, world_from_lidar) of every
    agent's frames."""
    out = []
    for agent in sorted(p for p in scenario.iterdir() if p.is_dir()):
        for yml in sorted(agent.glob("*.yaml"))[::stride]:
            doc = yaml.safe_load(yml.read_text())
            out.append(
                (
                    yml.with_suffix(".pcd"),
                    float(doc.get("timestamp", 0.0)),
                    int(yml.stem),
                    matrix_from_pose(doc["lidar_pose"]),
                )
            )
    return out


def cmd_build(args: argparse.Namespace) -> int:
    src, out = Path(args.src), Path(args.out)
    i, n = (int(v) for v in args.shard.split("/"))
    scenarios = sorted(p for p in src.iterdir() if p.is_dir())[i::n]
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    for scenario in scenarios:
        t0 = time.perf_counter()
        sweeps = sweeps_of(scenario, args.stride)
        seeds = []
        stamps = []
        indices = []
        for pcd, ts, idx, m in sweeps:
            seeds.append(seeds_of_sweep(read_pcd(pcd), m))
            stamps.append(ts)
            indices.append(idx)
        t_seeds = time.perf_counter() - t0
        floor = build_floor(seeds, len(sweeps))
        t_fit = time.perf_counter() - t0 - t_seeds
        if floor is None:
            print(f"{scenario.name}: too few seeds, no floor", flush=True)
            continue
        save_floor(floor, out / f"{scenario.name}.npz")
        entry: dict[str, Any] = {
            "scenario": scenario.name,
            "sweeps": len(sweeps),
            "seeds": floor.seeds,
            "grid": list(floor.field.height.shape),
            "cell_m": floor.field.cell,
            "seconds_seeds": round(t_seeds, 1),
            "seconds_fit": round(t_fit, 1),
            "windows": 0,
        }
        if args.leave_out is not None:
            all_ts = np.asarray(stamps, np.float64)
            lo, hi = float(all_ts.min()), float(all_ts.max())
            k = 0
            start = lo
            while start <= hi:
                end = start + args.window
                keep = (all_ts < start - args.leave_out) | (
                    ts > end + args.leave_out
                )
                sub = [s for s, ok in zip(seeds, keep, strict=True) if ok]
                f = build_floor(sub, int(keep.sum())) if sub else None
                if f is not None:
                    f.span = (start, end)
                    inside = (all_ts >= start) & (all_ts <= end)
                    if inside.any():
                        idx_in = np.asarray(indices)[inside]
                        f.frame_span = (int(idx_in.min()), int(idx_in.max()))
                    save_floor(f, out / f"{scenario.name}-w{k:02d}.npz")
                    k += 1
                start = end
            entry["windows"] = k
        report.append(entry)
        print(json.dumps(entry), flush=True)
    tag = "" if args.shard == "0/1" else "-" + args.shard.replace("/", "of")
    (out / f"build{tag}.json").write_text(json.dumps(report, indent=1))
    return 0


def cmd_time(args: argparse.Namespace) -> int:
    scenario = Path(args.src) / args.scenario
    floor: MapFloor = load_floor(Path(args.floors) / f"{args.scenario}.npz")
    agents = (
        [args.agent]
        if args.agent
        else sorted(p.name for p in scenario.iterdir() if p.is_dir())
    )
    sweeps = []
    for agent in agents:
        for yml in sorted((scenario / agent).glob("*.yaml")):
            doc = yaml.safe_load(yml.read_text())
            sweeps.append(
                (
                    read_pcd(yml.with_suffix(".pcd")),
                    matrix_from_pose(doc["lidar_pose"]),
                )
            )
    stride = max(1, len(sweeps) // args.frames)
    sweeps = sweeps[::stride][: args.frames]
    cpu_ms: list[float] = []
    for pts, m in sweeps:
        surf = MapSurface(floor, m)
        t0 = time.perf_counter()
        _ = pts[:, 2] - surf.at(pts[:, :2]) - 1.9
        cpu_ms.append((time.perf_counter() - t0) * 1e3)
    result: dict[str, Any] = {
        "scenario": args.scenario,
        "frames": len(sweeps),
        "points_median": int(statistics.median(len(p) for p, _ in sweeps)),
        "floor_grid": list(floor.field.height.shape),
        "floor_cell_m": floor.field.cell,
        "cpu_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "lookup_ms_cpu_median": round(statistics.median(cpu_ms), 3),
        "lookup_ms_cpu_p90": round(float(np.percentile(cpu_ms, 90)), 3),
    }
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            grid = torch.from_numpy(floor.field.height).float().to(dev)
            gpu_ms = []
            for pts, m in sweeps:
                p = torch.from_numpy(pts[:, :3]).float().to(dev)
                mm = torch.from_numpy(m).float().to(dev)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                w = p @ mm[:3, :3].T + mm[:3, 3]
                q = w[:, :2] - torch.from_numpy(floor.origin).float().to(dev)
                gx = (q[:, 0] - floor.field.x0) / floor.field.cell - 0.5
                gy = (q[:, 1] - floor.field.y0) / floor.field.cell - 0.5
                nh, nw = grid.shape
                sx = (gx / (nw - 1)) * 2 - 1
                sy = (gy / (nh - 1)) * 2 - 1
                sample = torch.nn.functional.grid_sample(
                    grid[None, None],
                    torch.stack([sx, sy], dim=1)[None, None],
                    align_corners=True,
                    padding_mode="border",
                )[0, 0, 0]
                _ = w[:, 2] - sample - 1.9
                torch.cuda.synchronize()
                gpu_ms.append((time.perf_counter() - t0) * 1e3)
            result["lookup_ms_gpu_median"] = round(
                statistics.median(gpu_ms), 3
            )
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["gpu"] = "torch not in this environment"
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


def main() -> int:
    args = parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_time(args)


if __name__ == "__main__":
    sys.exit(main())
