"""Waymo Open Dataset (perception, v2 parquet) -> frames in the OPV2V layout.

One segment is one scenario with one agent. Everything is read with
pyarrow, no TensorFlow: per frame the five range images (``lidar``),
the TOP laser's per-pixel poses (``lidar_pose``), the laser boxes
(``lidar_box``, vehicle frame), the vehicle pose (``vehicle_pose``) and
the camera JPEGs with their own poses (``camera_image``,
``camera_calibration``).

Waymo's vehicle frame (x forward, y left, z up) has its origin on the
road under the rear axle; the exported frame keeps the orientation but
puts the origin at the TOP laser (~2.2 m up) like OPV2V, so
``world_from_lidar`` is the vehicle pose shifted by that offset. Range
images are turned into points as ``waymo_open_dataset`` does: the TOP
laser's rolling shutter is undone with the per-pixel poses, the four
short-range lasers use the frame pose as is, only the first return is
kept, the intensity is squashed with ``tanh``. Waymo cameras look along
their own x axis; they are converted to OpenCV frames (z forward) and
the ego motion between the camera and the LiDAR timestamp is folded
into ``lidar_from_cam``. Lens distortion is ignored. Frames are taken at
2 Hz (every 5th of the 10 Hz frames) by default.

Cameras in OPV2V's order (front, right, left, back): FRONT,
FRONT_RIGHT, FRONT_LEFT and SIDE_RIGHT, since Waymo has no rear camera
and the right side is the curb side.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from v2b.data.class_map import canon
from v2b.data.opv2v_layout import (
    F32,
    F64,
    CameraRecord,
    FrameRecord,
    ObjectRecord,
)

SPLITS = {"train": "training", "validate": "validation"}
LASER_TOP = 1
CAMERAS_WAYMO: tuple[tuple[int, str], ...] = (
    (1, "FRONT"),
    (3, "FRONT_RIGHT"),
    (2, "FRONT_LEFT"),
    (5, "SIDE_RIGHT"),
)
BOX_TYPES = {
    1: "type_vehicle",
    2: "type_pedestrian",
    3: "other",
    4: "type_cyclist",
}
# OpenCV camera axes (x right, y down, z forward) as columns in Waymo's
# camera axes (x forward, y left, z up)
OPENCV_FROM_WAYMO_CAM: F64 = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)
TS = "key.frame_timestamp_micros"


def _list_rows(col: Any) -> list[np.ndarray]:
    """Each row of a ``list<float>`` column as its own array."""
    arr = col.combine_chunks()
    values = arr.values.to_numpy(zero_copy_only=False)
    offsets = arr.offsets.to_numpy()
    return [
        values[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]


def _mats(col: Any) -> list[F64]:
    """Each row of a ``fixed_size_list<double>[16]`` column as a 4x4."""
    return [np.asarray(v, float).reshape(4, 4) for v in col.to_pylist()]


def rotation_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> F64:
    """``R_z(yaw) R_y(pitch) R_x(roll)`` for arrays of angles, (..., 3, 3)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    out = np.empty(roll.shape + (3, 3), float)
    out[..., 0, 0] = cy * cp
    out[..., 0, 1] = cy * sp * sr - sy * cr
    out[..., 0, 2] = cy * sp * cr + sy * sr
    out[..., 1, 0] = sy * cp
    out[..., 1, 1] = sy * sp * sr + cy * cr
    out[..., 1, 2] = sy * sp * cr - cy * sr
    out[..., 2, 0] = -sp
    out[..., 2, 1] = cp * sr
    out[..., 2, 2] = cp * cr
    return out


def range_image_points(
    image: np.ndarray,
    extrinsic: F64,
    inclinations: np.ndarray,
    pixel_pose: np.ndarray | None = None,
    frame_pose: F64 | None = None,
) -> F32:
    """Points ``(N, 4)`` x y z intensity in the vehicle frame of one
    ``(H, W, 4)`` range image (range, intensity, elongation, nlz).

    Follows ``waymo_open_dataset.utils.range_image_utils``: beam
    inclinations are listed bottom-up and reversed per row, the azimuth
    runs from +pi to -pi across the columns minus the laser's yaw, and
    ``extrinsic`` (vehicle_from_laser) places the points. With
    ``pixel_pose`` (``(H, W, 6)`` roll, pitch, yaw, x, y, z: the
    world_from_vehicle pose at each pixel's time) the points are moved
    to the world and back with ``frame_pose``.
    """
    height, width = image.shape[:2]
    rng = image[..., 0].astype(np.float64)
    valid = rng > 0
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(width, 0, -1) - 0.5) / width
    azimuth = (ratios * 2 - 1) * np.pi - az_correction
    incl = np.asarray(inclinations, float)[::-1]
    if incl.shape[0] != height:
        raise ValueError(f"{incl.shape[0]} inclinations for {height} rows")
    cos_i = np.cos(incl)[:, None]
    pts = np.stack(
        [
            cos_i * np.cos(azimuth)[None, :] * rng,
            cos_i * np.sin(azimuth)[None, :] * rng,
            np.sin(incl)[:, None] * rng,
        ],
        axis=-1,
    )
    pts = pts @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if pixel_pose is not None:
        if frame_pose is None:
            raise ValueError("pixel_pose needs frame_pose")
        rot = rotation_rpy(
            pixel_pose[..., 0], pixel_pose[..., 1], pixel_pose[..., 2]
        )
        world = np.einsum("hwij,hwj->hwi", rot, pts) + pixel_pose[..., 3:6]
        vehicle_from_world = np.linalg.inv(frame_pose)
        pts = world @ vehicle_from_world[:3, :3].T + vehicle_from_world[:3, 3]
    intensity = np.tanh(np.maximum(image[..., 1], 0.0))
    out = np.concatenate([pts[valid], intensity[valid][:, None]], axis=1)
    return out.astype(np.float32)


def _table(path: Path, columns: list[str], ts: set[int] | None) -> Any:
    table = pq.read_table(path, columns=columns)
    if ts is not None:
        keep = pc.is_in(table[TS], value_set=pa.array(sorted(ts), pa.int64()))
        table = table.filter(keep)
    return table


def iter_segment(
    split_dir: Path,
    segment: str,
    every: int = 5,
    cameras: tuple[tuple[int, str], ...] = CAMERAS_WAYMO,
) -> Iterator[FrameRecord]:
    """Every ``every``-th frame of one segment as a frame record."""
    name = f"{segment}.parquet"
    calib = pq.read_table(split_dir / "lidar_calibration" / name)
    lasers: dict[int, tuple[F64, np.ndarray]] = {}
    for row in calib.to_pylist():
        ext = np.asarray(
            row["[LiDARCalibrationComponent].extrinsic.transform"], float
        ).reshape(4, 4)
        values = row["[LiDARCalibrationComponent].beam_inclination.values"]
        lasers[int(row["key.laser_name"])] = (
            ext,
            np.asarray(values, float) if values else np.empty(0),
        )
    cam_calib: dict[int, tuple[F64, F64]] = {}
    for row in pq.read_table(
        split_dir / "camera_calibration" / name
    ).to_pylist():
        p = "[CameraCalibrationComponent]."
        k = np.array(
            [
                [row[p + "intrinsic.f_u"], 0.0, row[p + "intrinsic.c_u"]],
                [0.0, row[p + "intrinsic.f_v"], row[p + "intrinsic.c_v"]],
                [0.0, 0.0, 1.0],
            ]
        )
        vehicle_from_cam = np.asarray(row[p + "extrinsic.transform"]).reshape(
            4, 4
        )
        vehicle_from_cam = vehicle_from_cam.copy()
        vehicle_from_cam[:3, :3] = (
            vehicle_from_cam[:3, :3] @ OPENCV_FROM_WAYMO_CAM
        )
        cam_calib[int(row["key.camera_name"])] = (k, vehicle_from_cam)
    poses_t = pq.read_table(split_dir / "vehicle_pose" / name)
    pose_rows = sorted(
        zip(
            poses_t[TS].to_pylist(),
            _mats(
                poses_t["[VehiclePoseComponent].world_from_vehicle.transform"]
            ),
            strict=True,
        ),
        key=lambda tm: tm[0],
    )
    ts_all = np.array([t for t, _ in pose_rows], np.int64)
    poses = [m for _, m in pose_rows]
    selected = [int(t) for t in ts_all[::every]]
    ts_set = set(selected)
    # OPV2V's frame has its origin at the LiDAR; Waymo's is on the road
    top_ext = lasers[LASER_TOP][0]
    vehicle_from_lidar = np.eye(4)
    vehicle_from_lidar[:3, 3] = top_ext[:3, 3]
    lidar_shift = -vehicle_from_lidar[:3, 3]
    li = "[LiDARComponent].range_image_return1."
    lidar_t = _table(
        split_dir / "lidar" / name,
        [TS, "key.laser_name", li + "values", li + "shape"],
        ts_set,
    )
    lidar_rows = _list_rows(lidar_t[li + "values"])
    lidar_shapes = lidar_t[li + "shape"].to_pylist()
    lidar_ts = lidar_t[TS].to_pylist()
    lidar_laser = lidar_t["key.laser_name"].to_pylist()
    lp = "[LiDARPoseComponent].range_image_return1."
    pose_t = _table(
        split_dir / "lidar_pose" / name,
        [TS, lp + "values", lp + "shape"],
        ts_set,
    )
    pixel_poses = {
        int(t): np.asarray(v, np.float64).reshape(s)
        for t, v, s in zip(
            pose_t[TS].to_pylist(),
            _list_rows(pose_t[lp + "values"]),
            pose_t[lp + "shape"].to_pylist(),
            strict=True,
        )
    }
    bp = "[LiDARBoxComponent]."
    box_t = _table(
        split_dir / "lidar_box" / name,
        [
            TS,
            "key.laser_object_id",
            bp + "box.center.x",
            bp + "box.center.y",
            bp + "box.center.z",
            bp + "box.size.x",
            bp + "box.size.y",
            bp + "box.size.z",
            bp + "box.heading",
            bp + "type",
        ],
        ts_set,
    )
    boxes_by_t: dict[int, list[dict[str, Any]]] = {}
    for row in box_t.to_pylist():
        boxes_by_t.setdefault(int(row[TS]), []).append(row)
    ci = "[CameraImageComponent]."
    cam_t = _table(
        split_dir / "camera_image" / name,
        [TS, "key.camera_name", ci + "image", ci + "pose.transform"],
        ts_set,
    )
    images: dict[tuple[int, int], tuple[bytes, F64]] = {}
    for t, c, im, pose in zip(
        cam_t[TS].to_pylist(),
        cam_t["key.camera_name"].to_pylist(),
        cam_t[ci + "image"].to_pylist(),
        _mats(cam_t[ci + "pose.transform"]),
        strict=True,
    ):
        images[(int(t), int(c))] = (im, pose)
    index_of = {int(t): i for i, t in enumerate(ts_all)}
    for index, t_us in enumerate(selected):
        i_all = index_of[t_us]
        world_from_vehicle = poses[i_all]
        world_from_lidar = world_from_vehicle @ vehicle_from_lidar
        parts: list[F32] = []
        for j, (t, laser) in enumerate(
            zip(lidar_ts, lidar_laser, strict=True)
        ):
            if int(t) != t_us:
                continue
            ext, incl = lasers[int(laser)]
            shape = lidar_shapes[j]
            image = lidar_rows[j].reshape(shape)
            if incl.size == 0:
                lo, hi = _incl_range(calib, int(laser))
                incl = (0.5 + np.arange(shape[0])) / shape[0] * (hi - lo) + lo
            pp = pixel_poses.get(t_us) if int(laser) == LASER_TOP else None
            parts.append(
                range_image_points(
                    image,
                    ext,
                    incl,
                    pp,
                    world_from_vehicle if pp is not None else None,
                )
            )
        pts = (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros((0, 4), np.float32)
        )
        pts[:, :3] += lidar_shift.astype(np.float32)
        speed = 0.0
        if i_all > 0:
            d = world_from_vehicle[:3, 3] - poses[i_all - 1][:3, 3]
            dt = (ts_all[i_all] - ts_all[i_all - 1]) / 1e6
            speed = float(np.linalg.norm(d[:2]) / dt) if dt > 0 else 0.0
        objects = [
            ObjectRecord(
                obj_id=str(r["key.laser_object_id"])[:12],
                cls=canon(BOX_TYPES.get(int(r[bp + "type"]), "other")),
                centre=np.array(
                    [
                        r[bp + "box.center.x"],
                        r[bp + "box.center.y"],
                        r[bp + "box.center.z"],
                    ],
                    float,
                )
                + lidar_shift,
                lwh=np.array(
                    [
                        r[bp + "box.size.x"],
                        r[bp + "box.size.y"],
                        r[bp + "box.size.z"],
                    ],
                    float,
                ),
                yaw=float(r[bp + "box.heading"]),
            )
            for r in boxes_by_t.get(t_us, [])
        ]
        lidar_from_world = np.linalg.inv(world_from_lidar)
        cams = []
        for cam_id, cam_name in cameras:
            hit = images.get((t_us, cam_id))
            if hit is None or cam_id not in cam_calib:
                continue
            jpeg, world_from_vehicle_img = hit
            k, vehicle_from_cam = cam_calib[cam_id]
            cams.append(
                CameraRecord(
                    name=cam_name,
                    image=Image.open(io.BytesIO(jpeg)),
                    intrinsic=k,
                    lidar_from_cam=lidar_from_world
                    @ world_from_vehicle_img
                    @ vehicle_from_cam,
                )
            )
        yield FrameRecord(
            scenario=f"waymo_{segment}",
            cav="1",
            index=index,
            timestamp_s=t_us / 1e6,
            points=pts,
            world_from_lidar=world_from_lidar,
            objects=objects,
            cameras=cams,
            speed_mps=speed,
        )


def _incl_range(calib: Any, laser: int) -> tuple[float, float]:
    for row in calib.to_pylist():
        if int(row["key.laser_name"]) == laser:
            return (
                float(row["[LiDARCalibrationComponent].beam_inclination.min"]),
                float(row["[LiDARCalibrationComponent].beam_inclination.max"]),
            )
    raise KeyError(laser)


def segments(split_dir: Path) -> list[str]:
    """Segment context names of a split (``training`` / ``validation``)."""
    return sorted(p.stem for p in (split_dir / "lidar_box").glob("*.parquet"))


def iter_frames(
    split_dir: Path,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    every: int = 5,
) -> Iterator[FrameRecord]:
    """Every ``every``-th frame of every segment under ``<root>/<split>``;
    ``shard=(i, n)`` keeps every n-th segment starting at i."""
    n = 0
    for segment in segments(split_dir)[shard[0] :: shard[1]]:
        for frame in iter_segment(split_dir, segment, every=every):
            yield frame
            n += 1
            if limit is not None and n >= limit:
                return
