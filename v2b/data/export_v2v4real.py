"""V2V4Real (Xu et al. CVPR 2023) -> frames in the OPV2V layout.

V2V4Real already uses OPV2V's folder tree (``<scenario>/<cav>/<index>.
{pcd,yaml}``, two vehicles per scenario) but not its label semantics:
``lidar_pose`` is a 4x4 world-from-LiDAR matrix instead of the
six-value pose, the boxes are given in the ego LiDAR frame
(``location`` relative to the sensor, ``extent`` half sizes, ``angle``
[roll, yaw, pitch] in degrees) with an ``obj_type`` (``Car`` only), and
the point clouds are ASCII ``.pcd`` with the intensity packed into an
``rgb`` float the way OPV2V does. HEAL's loader expects OPV2V's own
semantics, so the frames are rewritten through ``write_frame`` instead
of patching the loader: that also gives every box the canonical class
and the level-frame conventions the other derived sets share. There
are no cameras. The two agents keep their folder names ``0`` and ``1``
as cav ids, so HEAL pairs them by index like OPV2V.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml
from pypcd4 import pypcd4

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import F32, F64, FrameRecord, ObjectRecord


def read_ascii_pcd(path: Path) -> F32:
    """``(N, 4)`` x y z intensity from an OPV2V-style ``.pcd``.

    The intensity sits in the red byte of the packed ``rgb`` float
    (open3d reads it as the colour's first channel, which is what
    HEAL's ``pcd_to_np`` uses).
    """
    pc = pypcd4.PointCloud.from_path(str(path))
    data = pc.pc_data
    names = data.dtype.names or ()
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(
        np.float32
    )
    if "intensity" in names:
        inten = np.asarray(data["intensity"], np.float32)
    else:
        packed = np.ascontiguousarray(data["rgb"], np.float32).view(np.uint32)
        inten = ((packed >> 16) & 255).astype(np.float32) / 255.0
    keep = np.isfinite(xyz).all(axis=1)
    return np.concatenate([xyz[keep], inten[keep][:, None]], axis=1)


def load_label(path: Path) -> dict[str, object]:
    """V2V4Real yaml (contains numpy objects, so the full loader)."""
    with path.open() as fh:
        doc: dict[str, object] = yaml.load(fh, Loader=yaml.Loader)
    return doc


def iter_scenario(scenario_dir: Path, every: int = 1) -> Iterator[FrameRecord]:
    """Every frame of every agent in one scenario."""
    for cav_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
        yamls = sorted(cav_dir.glob("*.yaml"))[::every]
        prev: tuple[F64, int] | None = None
        for index, yml in enumerate(yamls):
            doc = load_label(yml)
            world_from_lidar = np.asarray(doc["lidar_pose"], float).reshape(
                4, 4
            )
            stamp = int(yml.stem)
            speed = float(str(doc.get("ego_speed") or 0.0))
            if speed == 0.0 and prev is not None and stamp > prev[1]:
                d = world_from_lidar[:3, 3] - prev[0][:3, 3]
                speed = float(np.linalg.norm(d[:2]) / ((stamp - prev[1]) / 10))
            prev = (world_from_lidar, stamp)
            objects = []
            vehicles = doc.get("vehicles") or {}
            assert isinstance(vehicles, dict)
            for key, v in vehicles.items():
                ext = np.asarray(v["extent"], float)
                objects.append(
                    ObjectRecord(
                        obj_id=str(key),
                        cls=canon(str(v.get("obj_type", "Car"))),
                        centre=np.asarray(v["location"], float),
                        lwh=2.0 * ext,
                        yaw=float(np.radians(v["angle"][1])),
                    )
                )
            name = scenario_dir.name.replace("testoutput_CAV_data_", "")
            yield FrameRecord(
                scenario=f"v2v4real_{name}",
                cav=cav_dir.name,
                index=index,
                timestamp_s=stamp / 10.0,
                points=read_ascii_pcd(yml.with_suffix(".pcd")),
                world_from_lidar=world_from_lidar,
                objects=objects,
                speed_mps=speed,
            )


def iter_frames(
    split_dir: Path,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    every: int = 1,
) -> Iterator[FrameRecord]:
    """Every agent-frame of every scenario under ``<root>/<split>``."""
    n = 0
    scenarios = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for scenario in scenarios[shard[0] :: shard[1]]:
        for frame in iter_scenario(scenario, every=every):
            yield frame
            n += 1
            if limit is not None and n >= limit:
                return
