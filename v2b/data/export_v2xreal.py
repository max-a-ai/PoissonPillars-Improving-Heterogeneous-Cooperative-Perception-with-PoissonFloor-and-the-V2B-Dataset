"""V2X-Real (Xiang et al. ECCV 2024) -> frames in the OPV2V layout.

V2X-Real is distributed in OPV2V's format: ``<split>/<scenario>/<agent>/
<index>.{bin,yaml}`` plus ``<index>_cam<k>.jpeg``, six-value
``lidar_pose`` in degrees, world-frame boxes with half-size ``extent``
and ``angle`` [roll, yaw, pitch], an ``obj_type`` from ten classes
(Pedestrian, Scooter, Motorcycle, BicycleRider, Car, Van, Truck,
BoxTruck, Bus, Barrier) and cameras with ``cords`` / ``extrinsic`` /
``intrinsic``. What differs from OPV2V, and what this adapter settles:

* point clouds are ``.bin`` (float32 x y z intensity, zero-padded to a
  fixed row count, intensity raw up to several thousand) instead of
  ``.pcd``; padding rows are dropped and the intensity is squashed with
  ``tanh(i / 255)``;
* agents are named ``1``, ``2`` (vehicles) and ``-1``, ``-2``
  (infrastructure, ``infra: true``). HEAL takes the first agent of a
  scenario as ego, so vehicles are written first: ``1 -> 0``,
  ``2 -> 1``, ``-1 -> 2``, ``-2 -> 3``;
* cameras are real OpenCV cameras; the yaml's ``extrinsic`` is the
  LiDAR-from-camera transform (checked by projection), so it is passed
  through unchanged as ``lidar_from_cam`` and the frames are marked
  ``frame: opencv``.

Agent roles make the four benchmark settings: HEAL takes ``cav 0`` as
the ego, so ``ego=vehicle`` (vehicles first) gives VC, ``ego=infra``
(roadside units first) IC; ``agents=vehicle`` / ``infra`` keep only one
kind for V2V / I2I. ``classes=v2xreal`` writes the benchmark's own three
classes (car / pedestrian / truck, ``V2XREAL_CLASSES``) instead of
ours; every box keeps its original ``category``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import (
    F32,
    CameraRecord,
    FrameRecord,
    ObjectRecord,
    matrix_from_pose,
)

AGENT_ORDER = {"1": "0", "2": "1", "-1": "2", "-2": "3"}
#: infrastructure first: the IC setting (ego = roadside unit -1)
AGENT_ORDER_INFRA = {"-1": "0", "-2": "1", "1": "2", "2": "3"}
#: the benchmark's grouping of the annotated categories into its three
#: scored classes (opencood/data_utils/__init__.py of the V2X-Real code,
#: SUPER_CLASS_MAP); categories not listed there are not scored
V2XREAL_CLASSES: dict[str, str] = {
    "longvehicle": "car",
    "car": "car",
    "policecar": "car",
    "child": "pedestrian",
    "roadworker": "pedestrian",
    "pedestrian": "pedestrian",
    "scooter": "pedestrian",
    "scooterrider": "pedestrian",
    "motorcycle": "pedestrian",
    "motorcylerider": "pedestrian",
    "motorcyclerider": "pedestrian",
    "bicyclerider": "pedestrian",
    "truck": "truck",
    "van": "truck",
    "trashcan": "truck",
    "concretetruck": "truck",
    "bus": "truck",
}


def v2xreal_class(label: str) -> str:
    """The benchmark's class of a category (``other`` when unscored)."""
    return V2XREAL_CLASSES.get(str(label).lower().strip(), "other")


def agent_order(ego: str) -> dict[str, str]:
    if ego == "vehicle":
        return AGENT_ORDER
    if ego == "infra":
        return AGENT_ORDER_INFRA
    raise ValueError(f"ego {ego!r}, expected vehicle or infra")


def read_bin(path: Path) -> F32:
    """``(N, 4)`` x y z intensity; zero padding rows removed."""
    raw = np.fromfile(path, np.float32)
    if raw.size % 4:
        raise ValueError(f"{path}: {raw.size} floats is not a multiple of 4")
    pts = raw.reshape(-1, 4)
    keep = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :3]).sum(axis=1) > 0)
    pts = pts[keep].copy()
    pts[:, 3] = np.tanh(np.maximum(pts[:, 3], 0.0) / 255.0)
    return pts


def _objects(
    doc: dict[str, Any], lidar_from_world: np.ndarray, classes: str = "v2b"
) -> list[ObjectRecord]:
    # the yaml's angle[1] is the box's world yaw; the LiDAR-frame yaw is
    # that minus the ego heading, the yaw of world_from_lidar. The yaw
    # of the inverse matrix is the negative of that, so adding it
    # doubled the ego heading on every box (every elongated class then
    # loses its AP at IoU 0.5 and 0.7, pedestrians do not)
    yaw_ego = -np.arctan2(lidar_from_world[1, 0], lidar_from_world[0, 0])
    to_class = canon if classes == "v2b" else v2xreal_class
    out = []
    for key, v in (doc.get("vehicles") or {}).items():
        loc = np.append(np.asarray(v["location"], float), 1.0)
        centre = (lidar_from_world @ loc)[:3]
        yaw = np.radians(float(v["angle"][1])) - yaw_ego
        category = str(v.get("obj_type", ""))
        out.append(
            ObjectRecord(
                obj_id=str(key),
                cls=to_class(category),
                centre=centre,
                lwh=2.0 * np.asarray(v["extent"], float),
                yaw=float(yaw),
                extra={"category": category},
            )
        )
    return out


def _cameras(
    doc: dict[str, Any], agent_dir: Path, stem: str
) -> list[CameraRecord]:
    cams = []
    for k in range(1, 5):
        cam = doc.get(f"cam{k}")
        img = agent_dir / f"{stem}_cam{k}.jpeg"
        if cam is None or not img.exists():
            continue
        try:
            image = Image.open(img)
            image.load()
        except (OSError, UnidentifiedImageError):
            print(f"unreadable image skipped: {img}", file=sys.stderr)
            continue
        cams.append(
            CameraRecord(
                name=f"cam{k}",
                image=image,
                intrinsic=np.asarray(cam["intrinsic"], float),
                lidar_from_cam=np.asarray(cam["extrinsic"], float),
            )
        )
    return cams


def iter_scenario(
    scenario_dir: Path,
    every: int = 1,
    ego: str = "vehicle",
    agents: str = "all",
    classes: str = "v2b",
) -> Iterator[FrameRecord]:
    """Every frame of every agent of one scenario, the ego's kind first
    (``ego`` vehicle or infra); ``agents`` keeps all, only the vehicles
    (V2V) or only the roadside units (I2I); ``classes`` v2b (car /
    bicycle / pedestrian / other) or v2xreal (the benchmark's car /
    pedestrian / truck)."""
    order = agent_order(ego)
    keep: Callable[[str], bool] = {
        "all": lambda name: True,
        "vehicle": lambda name: int(name) > 0,
        "infra": lambda name: int(name) < 0,
    }[agents]
    agent_dirs = sorted(
        (p for p in scenario_dir.iterdir() if p.is_dir() and keep(p.name)),
        key=lambda p: order.get(p.name, "9" + p.name),
    )
    for agent_dir in agent_dirs:
        cav = order.get(agent_dir.name, agent_dir.name)
        prev: tuple[np.ndarray, int] | None = None
        for index, yml in enumerate(sorted(agent_dir.glob("*.yaml"))[::every]):
            with yml.open() as fh:
                doc: dict[str, Any] = yaml.load(fh, Loader=yaml.Loader)
            world_from_lidar = matrix_from_pose(doc["lidar_pose"])
            lidar_from_world = np.linalg.inv(world_from_lidar)
            stamp = int(yml.stem)
            speed = float(doc.get("ego_speed") or 0.0)
            if speed == 0.0 and prev is not None and stamp > prev[1]:
                d = world_from_lidar[:3, 3] - prev[0][:3, 3]
                speed = float(np.linalg.norm(d[:2]) / ((stamp - prev[1]) / 10))
            prev = (world_from_lidar, stamp)
            try:
                points = read_bin(yml.with_suffix(".bin"))
            except ValueError as err:  # a truncated cloud (bad CRC in the zip)
                print(f"frame skipped: {err}", file=sys.stderr)
                continue
            yield FrameRecord(
                scenario=f"v2xreal_{scenario_dir.name}",
                cav=cav,
                index=index,
                timestamp_s=stamp / 10.0,
                points=points,
                world_from_lidar=world_from_lidar,
                objects=_objects(doc, lidar_from_world, classes),
                cameras=_cameras(doc, agent_dir, yml.stem),
                speed_mps=speed,
            )


def iter_frames(
    split_dir: Path,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    every: int = 1,
    ego: str = "vehicle",
    agents: str = "all",
    classes: str = "v2b",
) -> Iterator[FrameRecord]:
    """Every agent-frame of every scenario under ``<root>/<split>``."""
    n = 0
    scenarios = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for scenario in scenarios[shard[0] :: shard[1]]:
        for frame in iter_scenario(
            scenario, every=every, ego=ego, agents=agents, classes=classes
        ):
            yield frame
            n += 1
            if limit is not None and n >= limit:
                return
