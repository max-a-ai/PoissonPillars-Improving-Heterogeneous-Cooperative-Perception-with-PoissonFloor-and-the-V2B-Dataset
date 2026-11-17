"""Argoverse 2 Sensor logs -> frames in the OPV2V layout.

One log is one scenario with one agent. Argoverse stores every LiDAR
sweep already in the ego-vehicle frame (``sensors/lidar/<ts>.feather``,
both LiDARs merged), the annotations of that sweep in the same frame
(``annotations.feather``), the ego pose per timestamp in the city frame
(``city_SE3_egovehicle.feather``) and the cameras as OpenCV frames in
the ego frame (``calibration/egovehicle_SE3_sensor.feather`` +
``intrinsics.feather``). The exported frame keeps the ego orientation but
puts the origin at the top LiDAR (``up_lidar``), ~1.9 m above the road
like OPV2V, so ``world_from_lidar`` is the city pose shifted by that
offset; a camera's image nearest to the sweep is used, with the ego
motion between the two timestamps folded
into ``lidar_from_cam``. Lens distortion (k1..k3) is ignored.

Cameras in OPV2V's order: ring_front_center (front), ring_side_right,
ring_side_left, ring_rear_left (back).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import (
    F64,
    CameraRecord,
    FrameRecord,
    ObjectRecord,
)
from v2b.data.v2b_cameras import quat_to_rot

CAMERAS_AV2 = (
    "ring_front_center",
    "ring_side_right",
    "ring_side_left",
    "ring_rear_left",
)


def _se3(row: Any) -> F64:
    tf = np.eye(4)
    tf[:3, :3] = quat_to_rot([row.qw, row.qx, row.qy, row.qz])
    tf[:3, 3] = (row.tx_m, row.ty_m, row.tz_m)
    return tf


def _nearest(times: np.ndarray, t: int) -> int:
    return int(np.argmin(np.abs(times - t)))


def iter_log(
    log_dir: Path, cameras: tuple[str, ...] = CAMERAS_AV2
) -> Iterator[FrameRecord]:
    """Every LiDAR sweep of one log as a frame record."""
    ann = pd.read_feather(log_dir / "annotations.feather")
    poses = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    pose_t = poses["timestamp_ns"].to_numpy(np.int64)
    ego_from_sensor = {
        r.sensor_name: _se3(r)
        for r in pd.read_feather(
            log_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        ).itertuples()
    }
    intr = {
        r.sensor_name: (
            np.array(
                [[r.fx_px, 0, r.cx_px], [0, r.fy_px, r.cy_px], [0, 0, 1.0]]
            ),
        )
        for r in pd.read_feather(
            log_dir / "calibration" / "intrinsics.feather"
        ).itertuples()
    }
    # OPV2V's frame has its origin at the LiDAR, ~1.9 m above ground;
    # Argoverse's ego origin is on the road, so shift to the top LiDAR
    ego_from_lidar = np.eye(4)
    ego_from_lidar[:3, 3] = ego_from_sensor["up_lidar"][:3, 3]
    lidar_shift = -ego_from_lidar[:3, 3]
    cam_files: dict[str, tuple[list[Path], np.ndarray]] = {}
    for cam in cameras:
        files = sorted((log_dir / "sensors" / "cameras" / cam).glob("*.jpg"))
        cam_files[cam] = (
            files,
            np.array([int(f.stem) for f in files], np.int64),
        )
    sweeps = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
    ann_by_t = {int(t): g for t, g in ann.groupby("timestamp_ns")}
    prev: tuple[F64, int] | None = None
    for index, sweep in enumerate(sweeps):
        t_ns = int(sweep.stem)
        world_from_ego = _se3(poses.iloc[_nearest(pose_t, t_ns)])
        pts_df = pd.read_feather(sweep)
        pts = np.stack(
            [
                pts_df["x"].to_numpy(np.float32),
                pts_df["y"].to_numpy(np.float32),
                pts_df["z"].to_numpy(np.float32),
                np.clip(
                    pts_df["intensity"].to_numpy(np.float32) / 255.0, 0, 1
                ),
            ],
            axis=1,
        )
        pts[:, :3] += lidar_shift.astype(np.float32)
        world_from_lidar = world_from_ego @ ego_from_lidar
        speed = 0.0
        if prev is not None and t_ns > prev[1]:
            d = world_from_ego[:3, 3] - prev[0][:3, 3]
            speed = float(np.linalg.norm(d[:2]) / ((t_ns - prev[1]) / 1e9))
        prev = (world_from_ego, t_ns)
        objects = []
        for r in ann_by_t.get(t_ns, pd.DataFrame()).itertuples():
            rot = quat_to_rot([r.qw, r.qx, r.qy, r.qz])
            objects.append(
                ObjectRecord(
                    obj_id=str(r.track_uuid)[:12],
                    cls=canon(str(r.category)),
                    centre=np.array([r.tx_m, r.ty_m, r.tz_m], float)
                    + lidar_shift,
                    lwh=np.array([r.length_m, r.width_m, r.height_m], float),
                    yaw=float(np.arctan2(rot[1, 0], rot[0, 0])),
                )
            )
        cams = []
        lidar_from_world = np.linalg.inv(world_from_lidar)
        for cam in cameras:
            files, times = cam_files[cam]
            if not files:
                continue
            j = _nearest(times, t_ns)
            world_from_ego_img = _se3(
                poses.iloc[_nearest(pose_t, int(times[j]))]
            )
            cams.append(
                CameraRecord(
                    name=cam,
                    image=Image.open(files[j]),
                    intrinsic=intr[cam][0],
                    lidar_from_cam=lidar_from_world
                    @ world_from_ego_img
                    @ ego_from_sensor[cam],
                )
            )
        yield FrameRecord(
            scenario=f"av2_{log_dir.name[:8]}",
            cav="1",
            index=index,
            timestamp_s=t_ns / 1e9,
            points=pts,
            world_from_lidar=world_from_lidar,
            objects=objects,
            cameras=cams,
            speed_mps=speed,
        )


def iter_frames(
    split_dir: Path,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
) -> Iterator[FrameRecord]:
    """Every sweep of every log under ``<root>/<split>``; ``shard=(i, n)``
    keeps every n-th log starting at i."""
    n = 0
    logs = [
        p
        for p in sorted(p for p in split_dir.iterdir() if p.is_dir())
        if (p / "annotations.feather").exists()
    ]
    for log_dir in logs[shard[0] :: shard[1]]:
        for frame in iter_log(log_dir):
            yield frame
            n += 1
            if limit is not None and n >= limit:
                return
