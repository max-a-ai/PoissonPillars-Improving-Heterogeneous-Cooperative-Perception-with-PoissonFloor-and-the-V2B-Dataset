"""nuScenes (and any nuScenes-format export, V2B's included) -> frames.

Reads the version tables directly (no devkit): per scene, every sample's
LiDAR keyframe becomes a :class:`FrameRecord` in the *level* LiDAR
frame (origin at the LiDAR, axes of the ego vehicle: x forward, z up;
see ``NuScenesTables.level_lidar``), with the annotations brought in
from the global frame, and up to four
cameras in OPV2V's order (front, right, left, back), each with the ego
motion between its timestamp and the LiDAR's folded into
``lidar_from_cam``.

Splits are the devkit's scene lists, vendored in
``nuscenes_splits.json`` (train 700, val 150, test 150, mini_train 8,
mini_val 2).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import (
    F64,
    CameraRecord,
    FrameRecord,
    ObjectRecord,
)
from v2b.data.v2b_cameras import quat_to_rot

SPLITS = Path(__file__).with_name("nuscenes_splits.json")
#: OPV2V's camera order front, right, left, back.
CAMERAS_NUSCENES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
)
CAMERAS_V2B_CAR = (
    "CAMERA_FRONT_LEFT",
    "CAMERA_RIGHT_BACK",
    "CAMERA_LEFT_BACK",
    "CAMERA_BACK_CENTER",
)


def _se3(rec: dict[str, Any]) -> F64:
    tf = np.eye(4)
    tf[:3, :3] = quat_to_rot(rec["rotation"])
    tf[:3, 3] = rec["translation"]
    return tf


class NuScenesTables:
    """The json tables of one version directory, indexed by token."""

    def __init__(self, dataroot: Path, version: str) -> None:
        self.root = dataroot
        meta = dataroot / version
        self.scene = self._load(meta / "scene.json")
        self.sample = self._load(meta / "sample.json")
        self.sample_data = self._load(meta / "sample_data.json")
        self.ego_pose = self._load(meta / "ego_pose.json")
        self.calibrated_sensor = self._load(meta / "calibrated_sensor.json")
        self.sensor = self._load(meta / "sensor.json")
        self.instance = self._load(meta / "instance.json")
        self.category = self._load(meta / "category.json")
        self.sample_annotation = self._load(meta / "sample_annotation.json")
        self.by_sample: dict[str, list[dict[str, Any]]] = {}
        for sd in self.sample_data.values():
            if sd["is_key_frame"]:
                self.by_sample.setdefault(sd["sample_token"], []).append(sd)
        self.anns_by_sample: dict[str, list[dict[str, Any]]] = {}
        for a in self.sample_annotation.values():
            self.anns_by_sample.setdefault(a["sample_token"], []).append(a)

    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        rows = json.loads(path.read_text())
        return {r["token"]: r for r in rows}

    def channel(self, sd: dict[str, Any]) -> str:
        cs = self.calibrated_sensor[sd["calibrated_sensor_token"]]
        return str(self.sensor[cs["sensor_token"]]["channel"])

    def world_from_sensor(self, sd: dict[str, Any]) -> F64:
        ego = _se3(self.ego_pose[sd["ego_pose_token"]])
        cal = _se3(self.calibrated_sensor[sd["calibrated_sensor_token"]])
        out: F64 = ego @ cal
        return out

    def level_lidar(self, sd: dict[str, Any]) -> tuple[F64, F64]:
        """``(world_from_lidar, lidar_from_sensor)`` of the level LiDAR frame.

        The LiDAR's own frame is rotated (nuScenes: 90 deg, y forward)
        and slightly tilted, so far boxes drift out of HEAL's z window.
        The exported frame keeps the LiDAR origin but takes the ego
        orientation: x forward, level. Points are rotated by the
        calibration rotation, nothing else.
        """
        ego = _se3(self.ego_pose[sd["ego_pose_token"]])
        cal = _se3(self.calibrated_sensor[sd["calibrated_sensor_token"]])
        shift = np.eye(4)
        shift[:3, 3] = cal[:3, 3]
        rot = np.eye(4)
        rot[:3, :3] = cal[:3, :3]
        return ego @ shift, rot

    def scene_samples(self, scene: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        tok = scene["first_sample_token"]
        while tok:
            s = self.sample[tok]
            out.append(s)
            tok = s["next"]
        return out


#: Ouster return values (our V2B recordings, raw up to ~5 700) are
#: squashed with tanh(i / 6000), which puts their median at Waymo's
#: 0.055 (v2b/data/v2b_nusc.py); nuScenes' HDL-32E intensity is 0..255
OUSTER_SCALE = 6000.0


def read_points(path: Path, intensity: str = "nuscenes") -> np.ndarray:
    """nuScenes ``.pcd.bin``: float32 x y z intensity ring; NaNs dropped.

    ``intensity`` nuscenes: 0..255 clipped to 0..1; ouster: the V2B
    convention tanh(i / 6000). The nuScenes scaling on Ouster values
    saturates nearly every point at 1.0.
    """
    raw = np.fromfile(path, dtype=np.float32)
    pts = raw.reshape(-1, 5)[:, :4]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if intensity == "ouster":
        pts[:, 3] = np.tanh(np.maximum(pts[:, 3], 0.0) / OUSTER_SCALE)
    elif intensity == "nuscenes":
        pts[:, 3] = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
    else:
        raise ValueError(
            f"intensity {intensity!r}, expected nuscenes or ouster"
        )
    return pts.astype(np.float32)


def split_scenes(split: str) -> set[str]:
    """Scene names of a devkit split (train, val, test, mini_train/val)."""
    table = json.loads(SPLITS.read_text())
    if split not in table:
        raise KeyError(f"split {split!r} not in {sorted(table)}")
    return set(table[split])


def iter_frames(
    dataroot: Path,
    version: str,
    scenes: set[str] | None,
    lidar: str = "LIDAR_TOP",
    cameras: tuple[str, ...] = CAMERAS_NUSCENES,
    prefix: str = "nusc",
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    intensity: str = "nuscenes",
) -> Iterator[FrameRecord]:
    """Every LiDAR keyframe of the selected scenes as a frame record.

    ``shard = (i, n)`` keeps every n-th selected scene starting at i, so
    n processes can split one split between them.
    """
    t = NuScenesTables(dataroot, version)
    n = 0
    chosen = [
        s
        for s in sorted(t.scene.values(), key=lambda s: str(s["name"]))
        if scenes is None or s["name"] in scenes
    ]
    for scene in chosen[shard[0] :: shard[1]]:
        samples = t.scene_samples(scene)
        prev_pose: F64 | None = None
        prev_t = 0.0
        for index, sample in enumerate(samples):
            sds = {t.channel(sd): sd for sd in t.by_sample[sample["token"]]}
            if lidar not in sds:
                continue
            sd = sds[lidar]
            world_from_lidar, lidar_from_sensor = t.level_lidar(sd)
            points = read_points(dataroot / sd["filename"], intensity)
            points[:, :3] = points[:, :3] @ lidar_from_sensor[:3, :3].T
            t_s = float(sd["timestamp"]) / 1e6
            speed = 0.0
            if prev_pose is not None and t_s > prev_t:
                d = world_from_lidar[:3, 3] - prev_pose[:3, 3]
                speed = float(np.linalg.norm(d[:2]) / (t_s - prev_t))
            prev_pose, prev_t = world_from_lidar, t_s
            lidar_from_world = np.linalg.inv(world_from_lidar)
            objects = []
            for a in t.anns_by_sample.get(sample["token"], []):
                inst = t.instance[a["instance_token"]]
                name = str(t.category[inst["category_token"]]["name"])
                box_in_world = _se3(a)
                box_in_lidar = lidar_from_world @ box_in_world
                w, length, h = (float(v) for v in a["size"])
                objects.append(
                    ObjectRecord(
                        obj_id=str(a["instance_token"])[:12],
                        cls=canon(name),
                        centre=box_in_lidar[:3, 3].copy(),
                        lwh=np.array([length, w, h]),
                        yaw=float(
                            np.arctan2(box_in_lidar[1, 0], box_in_lidar[0, 0])
                        ),
                    )
                )
            cams = []
            for ch in cameras:
                if ch not in sds:
                    continue
                csd = sds[ch]
                cal = t.calibrated_sensor[csd["calibrated_sensor_token"]]
                cams.append(
                    CameraRecord(
                        name=ch,
                        image=Image.open(dataroot / csd["filename"]),
                        intrinsic=np.asarray(cal["camera_intrinsic"], float),
                        lidar_from_cam=lidar_from_world
                        @ t.world_from_sensor(csd),
                    )
                )
            yield FrameRecord(
                scenario=f"{prefix}_{scene['name']}",
                cav="1",
                index=index,
                timestamp_s=t_s,
                points=points,
                world_from_lidar=world_from_lidar,
                objects=objects,
                cameras=cams,
                speed_mps=speed,
            )
            n += 1
            if limit is not None and n >= limit:
                return
