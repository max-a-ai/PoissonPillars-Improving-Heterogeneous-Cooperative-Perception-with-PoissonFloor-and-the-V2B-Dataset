"""TUMTraf-V2X -> frames in the OPV2V layout, two LiDAR agents.

Each labelled frame of a split has a vehicle sweep (Robosense 32, in the
vehicle LiDAR frame), a roadside sweep (Ouster 64 at the s110 gantry)
and one OpenLABEL file with the boxes in the **registered** cloud's
frame, which is the roadside LiDAR's, plus the per-frame transform
vehicle LiDAR -> roadside LiDAR. The roadside LiDAR frame is the world here:
agent ``1`` is the vehicle with ``world_from_lidar`` = that transform,
agent ``2`` the infrastructure, whose own frame is the world shifted
down so that the road sits 1.9 m below its origin as in OPV2V (the
Ouster hangs 7.6 m up; HEAL's z window and anchor height assume a roof
LiDAR). Both agents share the
frame index, which is what HEAL's loader needs to pair them.

Cameras are not exported: the label files hold no camera calibration
(it lives in the dev kit's constants), so this is LiDAR-only data until
that is wired in.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import F64, FrameRecord, ObjectRecord
from v2b.data.v2b_cameras import quat_to_rot
from v2b.utils.boxes import load_pcd, load_pcd_intensity

REGISTERED = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
VEHICLE = "vehicle_lidar_robosense"
ROADSIDE = "s110_lidar_ouster_south"
TRANSFORM = "vehicle_lidar_robosense_to_s110_lidar_ouster_south"
#: OPV2V's LiDAR sits this far above the road; the roadside agent's
#: frame is shifted so its ground lands at the same height.
LIDAR_ABOVE_GROUND_M = 1.9


def ground_z(points: np.ndarray, radius_m: float = 30.0) -> float:
    """5th percentile of z within ``radius_m`` of the sensor: the road."""
    near = points[np.hypot(points[:, 0], points[:, 1]) < radius_m]
    return float(np.percentile(near[:, 2], 5)) if len(near) else 0.0


def _points(path: Path) -> np.ndarray:
    xyz = np.asarray(load_pcd(path), dtype=np.float32)
    inten = load_pcd_intensity(path)
    if inten is None:
        inten = np.zeros(len(xyz), np.float32)
    i = np.asarray(inten, dtype=np.float32)
    if i.max() > 1.0:
        i = i / 255.0
    pts = np.concatenate([xyz, np.clip(i, 0, 1)[:, None]], axis=1)
    return pts[np.isfinite(pts).all(axis=1)].astype(np.float32)


def _objects(frame: dict[str, Any]) -> list[ObjectRecord]:
    out = []
    for key, obj in frame["objects"].items():
        od = obj["object_data"]
        x, y, z, qx, qy, qz, qw, length, w, h = od["cuboid"]["val"][:10]
        rot = quat_to_rot([qw, qx, qy, qz])
        out.append(
            ObjectRecord(
                obj_id=str(od.get("name", key))[-8:],
                cls=canon(str(od["type"])),
                centre=np.array([x, y, z], float),
                lwh=np.array([length, w, h], float),
                yaw=float(np.arctan2(rot[1, 0], rot[0, 0])),
            )
        )
    return out


def _in_frame(
    objs: list[ObjectRecord], lidar_from_world: F64
) -> list[ObjectRecord]:
    yaw0 = float(np.arctan2(lidar_from_world[1, 0], lidar_from_world[0, 0]))
    out = []
    for o in objs:
        c = lidar_from_world @ np.append(o.centre, 1.0)
        out.append(ObjectRecord(o.obj_id, o.cls, c[:3], o.lwh, o.yaw + yaw0))
    return out


def iter_frames(
    split_dir: Path, limit: int | None = None, scenario: str | None = None
) -> Iterator[FrameRecord]:
    """Vehicle and roadside agent of every labelled frame of a split."""
    labels = sorted(
        (split_dir / "labels_point_clouds" / REGISTERED).glob("*.json")
    )
    name = scenario or f"tumtraf_{split_dir.name}"
    shift: F64 | None = None  # roadside LiDAR -> agent frame, fixed per split
    for index, label in enumerate(labels):
        doc = json.loads(label.read_text())
        frame = next(iter(doc["openlabel"]["frames"].values()))
        fp = frame["frame_properties"]
        clouds = {
            c.split("_", 2)[2].rsplit(".", 1)[0]: c
            for c in fp["point_cloud_file_names"]
        }
        world_from_vehicle = np.asarray(
            fp["transforms"][TRANSFORM]["transform_src_to_dst"]["matrix4x4"],
            float,
        )
        objs = _objects(frame)
        t_s = float(fp["timestamp"])
        roadside = _points(
            split_dir / "point_clouds" / ROADSIDE / clouds[ROADSIDE]
        )
        vehicle = _points(
            split_dir / "point_clouds" / VEHICLE / clouds[VEHICLE]
        )
        if shift is None:
            shift = np.eye(4)
            shift[2, 3] = ground_z(roadside) + LIDAR_ABOVE_GROUND_M
        roadside[:, :3] -= shift[:3, 3].astype(np.float32)
        yield FrameRecord(
            scenario=name,
            cav="1",
            index=index,
            timestamp_s=t_s,
            points=vehicle,
            world_from_lidar=world_from_vehicle,
            objects=_in_frame(objs, np.linalg.inv(world_from_vehicle)),
        )
        yield FrameRecord(
            scenario=name,
            cav="2",
            index=index,
            timestamp_s=t_s,
            points=roadside,
            world_from_lidar=shift,
            objects=_in_frame(objs, np.linalg.inv(shift)),
        )
        if limit is not None and index + 1 >= limit:
            return
