#!/usr/bin/env python3
"""Export a dataset split into the OPV2V layout HEAL trains on.

Derived data only: the source stays untouched, the output goes to a
workspace (``/path/to/derived/<name>/<split>`` on a
cluster, ``outputs/derived/...`` here). One scenario folder per scene,
one agent folder per sensor rig, and per frame the point cloud, the
label yaml (with ``obj_type`` per box and ``frame: opencv`` per camera),
four canonical 800 x 600 camera PNGs and their LiDAR-splat depth maps.
See ``v2b/data/opv2v_layout.py`` for the format and
``v2b/train/heal_data.py`` for the loader patches that read it.

Examples::

    uv run python v2b/scripts/v2b-export-opv2v.py nuscenes \\
        --root resources/data/nuscenes --version v1.0-mini \\
        --split mini_train --out outputs/derived/nuscenes-mini/train
    uv run python v2b/scripts/v2b-export-opv2v.py argoverse2 \\
        --root resources/data/argoverse2/Sensor/sensor --split val \\
        --out /path/to/derived/argoverse2/val
    uv run python v2b/scripts/v2b-export-opv2v.py waymo \\
        --root resources/data/waymo --split validate --shard 0/4 \\
        --out /path/to/derived/waymo/validate
    .venv/bin/python v2b/scripts/v2b-export-opv2v.py v2b \\
        --split-file v2b/data/split-coop-v2.json --split test \\
        --labels pseudo --allow-unlabelled \\
        --out /path/to/derived/v2b-coop-opv2v/test

The ``v2b`` source (``v2b/data/export_v2b.py``) writes our car + bike
scenes as two agents per scenario; ``--labels`` picks the label
delivery (annotated: the company's boxes; pseudo: MS3D), a scene
without one is skipped unless ``--allow-unlabelled`` exports its clouds
with empty labels, and ``--labels-only`` rewrites the yamls of an
existing export once labels have landed. ``labels.json`` at the split
root records each scene's label source and box counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.data.export_v2b import (  # noqa: E402
    LABEL_SOURCES,
    ScenePlan,
    plan,
)
from v2b.data.opv2v_layout import FrameRecord, write_frame  # noqa: E402
from v2b.maps.scenes import default_v2b_root  # noqa: E402


def v2b_scenes(args: argparse.Namespace) -> list[str]:
    """The scenes of a v2b export: ``--scenes``, else the split's."""
    if args.scenes:
        return [s for s in args.scenes.split(",") if s]
    table = json.loads(Path(args.split_file).read_text())["splits"]
    if args.split not in table:
        raise KeyError(f"split {args.split!r} not in {sorted(table)}")
    scenes: list[str] = list(table[args.split])
    return scenes


def v2b_plan(args: argparse.Namespace) -> list[ScenePlan]:
    root = Path(args.root) if args.root else default_v2b_root()
    plans = plan(
        v2b_scenes(args),
        args.labels,
        root,
        args.allow_unlabelled,
        fallback=args.fallback,
    )
    for p in plans:
        print(f"{p.name}: {p.note}", flush=True)
    return plans


def write_labels_json(
    out: Path,
    source: str,
    plans: list[ScenePlan],
    boxes: dict[str, Counter[str]],
    samples: Counter[str],
    tag: str = "",
    fallback: str | None = None,
    max_range: float | None = None,
) -> None:
    """``labels.json`` at the split root: per scene the label delivery,
    its kind (``annotated`` / ``pseudo``, the fallback where the first
    choice had none) and the box counts. Each run (shard) writes its
    own ``labels<tag>.json`` first and ``labels.json`` is the merge of
    every shard file, so parallel shards never clobber each other."""
    doc: dict[str, Any] = {
        "labels": source,
        "fallback": fallback,
        "max_range": max_range,
        "scenes": {},
    }
    for p in plans:
        if p.name not in samples:
            continue  # another shard's scene, or skipped
        doc["scenes"][p.name] = {
            "source": None if p.labels is None else str(p.labels),
            "kind": p.source or None,
            "samples": samples[p.name],
            "boxes": dict(boxes.get(p.name, Counter())),
        }
    (out / f"labels{tag}.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True)
    )
    merged: dict[str, Any] = {
        "labels": source,
        "fallback": fallback,
        "max_range": max_range,
        "scenes": {},
    }
    for part in sorted(out.glob("labels-*.json")) + [out / "labels-.json"]:
        if part.exists():
            merged["scenes"].update(json.loads(part.read_text())["scenes"])
    merged["scenes"].update(doc["scenes"])
    merged["scenes"] = dict(sorted(merged["scenes"].items()))
    (out / "labels.json").write_text(
        json.dumps(merged, indent=1, sort_keys=True)
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dataset",
        choices=[
            "nuscenes",
            "argoverse2",
            "tumtraf",
            "waymo",
            "v2v4real",
            "v2xreal",
            "v2b",
        ],
    )
    ap.add_argument(
        "--root",
        default=None,
        help="dataset root directory (v2b: resources/data/v2b)",
    )
    ap.add_argument("--out", required=True, help="output split directory")
    ap.add_argument(
        "--split", default="train", help="devkit split, or all (nuscenes)"
    )
    ap.add_argument("--version", default="v1.0-trainval", help="nuscenes")
    ap.add_argument("--lidar", default=None, help="nuscenes LiDAR channel")
    ap.add_argument("--prefix", default=None, help="scenario name prefix")
    ap.add_argument("--limit", type=int, default=None, help="frames at most")
    ap.add_argument(
        "--shard",
        default="0/1",
        help="i/n: this process takes every n-th scene (log) from i",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=None,
        help="keep every n-th 10 Hz frame (default: waymo 5 = 2 Hz, "
        "v2v4real 1 = all frames)",
    )
    ap.add_argument(
        "--ego",
        choices=["vehicle", "infra"],
        default="vehicle",
        help="v2xreal: which kind of agent is cav 0 (VC or IC setting)",
    )
    ap.add_argument(
        "--agents",
        choices=["all", "vehicle", "infra"],
        default="all",
        help="v2xreal: agents kept (all; vehicle = V2V; infra = I2I)",
    )
    ap.add_argument(
        "--classes",
        choices=["v2b", "v2xreal"],
        default="v2b",
        help="v2xreal: our four classes, or the benchmark's three",
    )
    ap.add_argument(
        "--intensity",
        choices=["nuscenes", "ouster"],
        default="nuscenes",
        help="nuscenes: the point intensity column scaled from 0..255; "
        "ouster: our V2B recordings, tanh(i / 6000)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip frames whose yaml already exists (resume)",
    )
    ap.add_argument(
        "--labels",
        choices=list(LABEL_SOURCES),
        default="pseudo",
        help="v2b: the label delivery (annotated: 4_annotated; pseudo: "
        "the MS3D folder on the NAS, then v2b-derived/pseudo-labels)",
    )
    ap.add_argument(
        "--fallback",
        choices=list(LABEL_SOURCES),
        default=None,
        help="v2b: the label delivery a scene takes when it has none "
        "under --labels (hand labels first, pseudo where they are "
        "missing: --labels annotated --fallback pseudo)",
    )
    ap.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="v2b: keep only boxes within this radius of both agents "
        "(the contract's 50 m dual radius, so pseudo labels teach the "
        "same range as the hand labels)",
    )
    ap.add_argument(
        "--allow-unlabelled",
        action="store_true",
        help="v2b: export a scene without labels with empty yamls "
        "instead of skipping it",
    )
    ap.add_argument(
        "--labels-only",
        action="store_true",
        help="v2b: rewrite the yamls only, keep the point clouds",
    )
    ap.add_argument(
        "--scenes",
        default=None,
        help="v2b: comma-separated scene names (default: the split's "
        "scenes of --split-file)",
    )
    ap.add_argument(
        "--split-file",
        default=str(REPO / "v2b" / "data" / "split-coop-v2.json"),
        help="v2b: the scene split whose --split is exported",
    )
    args = ap.parse_args()
    if args.root is None and args.dataset != "v2b":
        ap.error("--root is required")
    return args


def frames_for(
    args: argparse.Namespace, plans: list[ScenePlan] | None = None
) -> Iterator[FrameRecord]:
    root = Path(args.root) if args.root else default_v2b_root()
    i, n = (int(v) for v in args.shard.split("/"))
    shard = (i, n)
    if args.dataset == "nuscenes":
        from v2b.data.export_nuscenes import iter_frames, split_scenes

        # "all": every scene of the tables, for nuScenes-format exports
        # (V2B's) whose scene names are not in the devkit splits
        scenes = None if args.split == "all" else split_scenes(args.split)
        kwargs = {}
        if args.lidar:
            kwargs["lidar"] = args.lidar
        if args.prefix:
            kwargs["prefix"] = args.prefix
        return iter_frames(
            root,
            args.version,
            scenes,
            limit=args.limit,
            shard=shard,
            intensity=args.intensity,
            **kwargs,
        )
    if args.dataset == "v2b":
        from v2b.data.export_v2b import iter_frames as v2b_frames

        return v2b_frames(
            plans if plans is not None else v2b_plan(args),
            Path(args.root) if args.root else None,
            limit=args.limit,
            shard=shard,
            max_range=args.max_range,
        )
    if args.dataset == "argoverse2":
        from v2b.data.export_argoverse2 import iter_frames as av2_frames

        return av2_frames(root / args.split, limit=args.limit, shard=shard)
    if args.dataset == "v2xreal":
        from v2b.data.export_v2xreal import iter_frames as v2xreal_frames

        split = {"validate": "val"}.get(args.split, args.split)
        return v2xreal_frames(
            root / split,
            limit=args.limit,
            shard=shard,
            every=args.every or 1,
            ego=args.ego,
            agents=args.agents,
            classes=args.classes,
        )
    if args.dataset == "v2v4real":
        from v2b.data.export_v2v4real import iter_frames as v2v4real_frames

        return v2v4real_frames(
            root / args.split,
            limit=args.limit,
            shard=shard,
            every=args.every or 1,
        )
    if args.dataset == "waymo":
        from v2b.data.export_waymo import SPLITS
        from v2b.data.export_waymo import iter_frames as waymo_frames

        split = SPLITS.get(args.split, args.split)
        return waymo_frames(
            root / split, limit=args.limit, shard=shard, every=args.every or 5
        )
    from v2b.data.export_tumtraf import iter_frames as tt_frames

    return tt_frames(root / args.split, limit=args.limit)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    per_scene: dict[str, Counter[str]] = {}
    samples: Counter[str] = Counter()
    n = 0
    t0 = time.time()
    plans = v2b_plan(args) if args.dataset == "v2b" else []
    for frame in frames_for(args, plans):
        stem = f"{frame.index:06d}"
        if (
            args.skip_existing
            and (out / frame.scenario / frame.cav / f"{stem}.yaml").exists()
        ):
            continue
        write_frame(out, frame, labels_only=args.labels_only)
        if frame.cav == "0" or args.dataset != "v2b":
            # one count per sample: the v2b agents share their boxes
            counts.update(o.cls for o in frame.objects)
            scene = frame.scenario.removeprefix("v2b_")
            per_scene.setdefault(scene, Counter()).update(
                o.cls for o in frame.objects
            )
            samples[scene] += 1
        n += 1
        if n % 50 == 0:
            rate = n / (time.time() - t0)
            print(f"{n} frames, {rate:.1f}/s, {dict(counts)}", flush=True)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "frames": n,
        "boxes": dict(counts),
        "seconds": round(time.time() - t0, 1),
    }
    tag = "" if args.shard == "0/1" else "-" + args.shard.replace("/", "of")
    if args.dataset == "v2b":
        summary["labels"] = args.labels
        summary["scenes"] = {
            k: {"samples": samples[k], "boxes": dict(v)}
            for k, v in sorted(per_scene.items())
        }
        write_labels_json(
            out,
            args.labels,
            plans,
            per_scene,
            samples,
            tag or "-",
            fallback=args.fallback,
            max_range=args.max_range,
        )
    (out / f"export{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
