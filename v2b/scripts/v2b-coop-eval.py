#!/usr/bin/env python3
"""Score one cooperative setting of one checkpoint on one data root.

The paper's title claim, car only vs bike only vs car + bike, on V2B:
four settings per detector, each scored on the same labels in the
car's frame and range (``v2b.metrics.coop_eval``):

``car``           the car's sweep through the single-agent checkpoint
``bike``          the bike's sweep through the same checkpoint, its
                  boxes brought into the car frame by the SLAM poses
``late``          both of the above, merged by rotated NMS
``intermediate``  the two sweeps through the F-Cooper checkpoint, the
                  bike projected into the car frame (``proj_first``)

The spec (``configs/<eval>.yaml``) names the checkpoint, the grid and,
under ``coop:``, the setting and the floor; the floor is applied per
agent cloud before the detector sees it (Poisson for our rows). The
label source is a flag and lands in the output path and the json::

    outputs/coop-eval/<run>/<labels>/results.json       the tables
    outputs/coop-eval/<run>/<labels>/predictions.npz    per-frame boxes
    outputs/coop-eval/<run>/<labels>/run.json           what ran

``--rescore`` reads the npz back and recomputes results.json without
data or GPU (``v2b-eval.py`` does the same from the project venv).

Run inside the HEAL venv from the repo root, e.g. the dry run on the
annotated scene_11 scene::

    V2B_FORK=heal third_party/HEAL/.venv/bin/python \\
        v2b/scripts/v2b-coop-eval.py \\
        --config configs/<eval-car>.yaml \\
        --source raw --scene scene_11 \\
        --root <NAS>/v2b/2_processed_rosbags/nusc/scene_11/v2b \\
        --tables <NAS>/v2b/4_annotated/nusc/scene_11/v2b/v1.0-mini \\
        --labels annotated

and the test split of the two-agent export::

    ... --config configs/<eval-fused>.yaml \\
        --source export --labels pseudo \\
        --root /path/to/derived/v2b-coop-opv2v/test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("V2B_FORK", "heal")

from v2b.metrics.coop_eval import (  # noqa: E402
    FrameDetections,
    FrameLabels,
    late_fuse,
    load_predictions,
    save_predictions,
    score_run,
    transform_boxes,
)
from v2b.train.coop_data import prepare  # noqa: E402

SETTINGS = ("car", "bike", "late", "intermediate")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="configs/<eval>.yaml")
    ap.add_argument("--source", choices=["raw", "export"], default="export")
    ap.add_argument(
        "--root", help="export split dir, or the delivery's sweeps"
    )
    ap.add_argument("--tables", default=None, help="raw: annotated tables dir")
    ap.add_argument(
        "--label-tables",
        default=None,
        help="raw: a pseudo-label table set instead of the delivery's",
    )
    ap.add_argument("--scene", default=None, help="raw: the scene's name")
    ap.add_argument(
        "--nusc-root",
        default="/path/to/data/v2b/2_processed_rosbags/nusc",
        help="export: the raw deliveries, for each agent's own sweep stamp",
    )
    ap.add_argument(
        "--labels",
        required=True,
        choices=["annotated", "pseudo"],
        help="the label source, carried into the paths and the json",
    )
    ap.add_argument("--out", default=str(REPO / "outputs" / "coop-eval"))
    ap.add_argument(
        "--run", default=None, help="run name (default: the spec's)"
    )
    ap.add_argument(
        "--rescore", action="store_true", help="from predictions.npz"
    )
    ap.add_argument("--score-threshold", type=float, default=0.2)
    ap.add_argument("--nms", type=float, default=0.15)
    ap.add_argument("--min-points", type=int, default=10)
    ap.add_argument("--single-floor", type=int, default=1)
    ap.add_argument(
        "--max-range",
        type=float,
        default=50.0,
        help="the labels' dual radius in m: boxes and detections farther "
        "than this from the car or the bike are left out (0 = off)",
    )
    ap.add_argument("--draws", type=int, default=1000, help="bootstrap draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="floor pool size")
    ap.add_argument("--limit", type=int, default=0, help="first N frames")
    ap.add_argument(
        "--cache",
        default=str(REPO / "outputs" / "coop-eval" / "cache"),
        help="floored clouds, shared by the settings of one floor",
    )
    return ap.parse_args()


def source_spec(args: argparse.Namespace) -> dict[str, Any]:
    if not args.root:
        raise SystemExit("--root is required unless --rescore")
    if args.source == "raw":
        return {
            "kind": "raw",
            "root": str(Path(args.root).resolve()),
            "tables": str(Path(args.tables).resolve())
            if args.tables
            else None,
            "labels": (
                str(Path(args.label_tables).resolve())
                if args.label_tables
                else None
            ),
            "scene": args.scene or Path(args.root).resolve().parent.name,
        }
    return {
        "kind": "export",
        "root": str(Path(args.root).resolve()),
        "nusc_root": str(Path(args.nusc_root).resolve())
        if args.nusc_root
        else None,
    }


def fmt(v: float) -> str:
    return "  nan" if v != v else f"{v:.3f}"


def print_table(results: dict[str, Any], title: str) -> None:
    ap = results["ap"]
    print(
        f"\n{title}: {results['frames']} frames, "
        f"{len(results['scenes'])} scene(s)"
    )
    print(
        f"{'stratum':<16}"
        + "".join(f"{c:>26}" for c in ("car", "bicycle", "pedestrian"))
    )
    print(
        f"{'':<16}"
        + "".join(f"{'AP@.3  AP@.5  AP@.7  n_gt':>26}" for _ in range(3))
    )
    for name, row in ap.items():
        if "car" not in row:
            continue
        cells = "".join(
            f"{fmt(row[c]['ap@0.3']):>7}{fmt(row[c]['ap@0.5']):>7}"
            f"{fmt(row[c]['ap@0.7']):>7}{row[c]['n_gt']:>5}"
            for c in ("car", "bicycle", "pedestrian")
        )
        print(f"{name:<16}{cells}")
    ci = results.get("ci95") or {}
    if ci:
        print(
            f"95 % CI ({ci['unit']}s, n={ci['n_units']}, "
            f"{ci['draws']} draws) at AP@0.5: "
            + ", ".join(
                f"{c} [{ci[c]['ap@0.5'][0]:.3f}, {ci[c]['ap@0.5'][1]:.3f}]"
                for c in ("car", "bicycle", "pedestrian")
                if c in ci
            )
        )


def write_results(
    out_dir: Path,
    labels: list[FrameLabels],
    dets: list[FrameDetections],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    results = score_run(
        labels,
        dets,
        meta["lidar_range"],
        args.min_points,
        args.single_floor,
        args.draws,
        args.seed,
        max_range=args.max_range or None,
    )
    results = {
        "run": meta["run"],
        "labels": meta["labels"],
        "setting": meta["setting"],
        "detector": meta["detector"],
        "checkpoint": meta["init_from"],
        "floor": meta["floor"],
        "source": meta["source"],
        "score_threshold": meta["score_threshold"],
        "nms_threshold": meta["nms_threshold"],
        **results,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def run_setting(
    det: Any, setting: str, item: dict[str, Any], nms: float
) -> FrameDetections:
    eye = np.eye(4)
    car_from_bike = np.asarray(item["car_from_bike"], float)
    parts: list[FrameDetections] = []
    if setting in ("car", "late"):
        b, s, n = det.detect([item["car"]], [eye])
        parts.append(FrameDetections(b, s, n, ["car"] * len(n)))
    if setting in ("bike", "late"):
        b, s, n = det.detect([item["bike"]], [eye])
        parts.append(
            FrameDetections(
                transform_boxes(b, car_from_bike), s, n, ["bike"] * len(n)
            )
        )
    if setting == "intermediate":
        b, s, n = det.detect([item["car"], item["bike"]], [eye, car_from_bike])
        parts.append(FrameDetections(b, s, n, ["fused"] * len(n)))
    if setting == "late":
        return late_fuse(parts, nms)
    return parts[0]


def main() -> int:
    args = parse_args()
    doc = yaml.safe_load(Path(args.config).read_text())
    coop = doc.get("coop") or {}
    setting = str(coop.get("setting", ""))
    if setting not in SETTINGS:
        raise SystemExit(
            f"coop.setting must be one of {SETTINGS}, got {setting!r}"
        )
    run = args.run or str(doc["name"])
    out_dir = Path(args.out) / run / args.labels
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        labels, dets, meta = load_predictions(out_dir / "predictions.npz")
        results = write_results(out_dir, labels, dets, meta, args)
        print_table(results, f"{run} [{args.labels}] re-scored")
        return 0

    spec = source_spec(args)
    floor = str(coop.get("floor", "raw"))
    # the pool forks before CUDA is initialised (a fork afterwards hangs)
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"))
    from v2b.train.coop_data import open_source
    from v2b.train.coop_infer import Detector, render_config, set_deterministic

    flags = set_deterministic(args.seed)
    src = open_source(spec)
    n = len(src) if not args.limit else min(args.limit, len(src))
    hypes, _ = render_config(Path(args.config), out_dir)
    init_from = REPO / str(doc["init_from"])
    det = Detector(hypes, init_from, args.score_threshold, args.nms)
    for note in det.notes:
        print(f"note       : {note}")
    root = Path(spec["root"])
    cache = str(Path(args.cache) / f"{root.parent.name}-{root.name}" / floor)
    print(
        f"{run}: setting {setting}, floor {floor}, labels {args.labels}, "
        f"{n} frames from {spec['root']}",
        flush=True,
    )
    tasks = [(spec, i, floor, cache) for i in range(n)]
    labels_out: list[FrameLabels] = []
    dets_out: list[FrameDetections] = []
    t0 = time.perf_counter()
    for k, item in enumerate(pool.map(prepare, tasks, chunksize=1)):
        dets_out.append(run_setting(det, setting, item, args.nms))
        labels_out.append(
            FrameLabels(
                scene=str(item["scene"]),
                frame=str(item["frame"]),
                boxes=item["gt_boxes"],
                names=[str(x) for x in item["gt_names"]],
                ignore=item["gt_ignore"],
                counts={"car": item["count_car"], "bike": item["count_bike"]},
                dt_ms=float(item["dt_ms"]),
                agent_gap_m=float(item["agent_gap_m"]),
                bike_xy=(
                    float(item["car_from_bike"][0, 3]),
                    float(item["car_from_bike"][1, 3]),
                ),
            )
        )
        if k % 50 == 0:
            print(
                f"  {k}/{n} {item['scene']} {item['frame']}: "
                f"{len(dets_out[-1].boxes)} boxes, "
                f"{time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    pool.shutdown()
    meta = {
        "run": run,
        "config": str(args.config),
        "labels": args.labels,
        "setting": setting,
        "detector": str(coop.get("detector", "")),
        "init_from": str(doc["init_from"]),
        "floor": floor,
        "source": spec,
        "frames": n,
        "lidar_range": det.lidar_range,
        "score_threshold": det.score_threshold,
        "nms_threshold": det.nms_threshold,
        "min_points": args.min_points,
        "single_floor": args.single_floor,
        "max_range": args.max_range or None,
        "determinism": flags,
        "notes": det.notes,
        "gpu": det.torch.cuda.get_device_name(0)
        if det.device.type == "cuda"
        else "",
        "seconds": round(time.perf_counter() - t0, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_predictions(out_dir / "predictions.npz", labels_out, dets_out, meta)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    results = write_results(out_dir, labels_out, dets_out, meta, args)
    print_table(results, f"{run} [{args.labels}]")
    print(f"wrote      : {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
