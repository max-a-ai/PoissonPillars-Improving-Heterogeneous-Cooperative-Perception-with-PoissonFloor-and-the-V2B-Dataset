"""Write frames of any dataset in the OPV2V layout HEAL's loader reads.

One frame of one agent becomes, under ``<root>/<split>/<scenario>/<cav>/``:

``<index>.pcd``
    x, y, z and intensity in the agent's LiDAR frame, binary PCD with the
    intensity packed into an ``rgb`` field, which is where
    ``opencood.utils.pcd_utils.pcd_to_np`` reads it back (``colors[:, 0]``).
``<index>.yaml``
    ``lidar_pose`` (6-DoF, HEAL's ``x_to_world`` parametrisation),
    ``vehicles`` keyed by object id with ``location`` (world), ``extent``
    (half sizes), ``angle`` ``[roll, yaw, pitch]`` in degrees and our
    extra ``obj_type`` (car / bicycle / pedestrian / other), and
    ``camera0..3`` with ``cords`` (world pose of the camera), the
    ``intrinsic`` of the canonical 800 x 600 image, ``extrinsic``
    (LiDAR -> camera) and ``frame: opencv``. HEAL expects CARLA cameras
    and multiplies in a handedness flip; the ``frame`` key tells the
    v2b patch in ``v2b.train.heal_data`` to skip it.
``<index>_camera<i>.png``
    the camera image scaled to 800 px wide and cropped or padded to
    600 px high; the intrinsics are adjusted accordingly.
``<index>_depth<i>.png``
    8-bit metres like OPV2V-H's depth maps: LiDAR points splatted into
    the canonical image with a nearest-wins z-buffer, 255 where nothing
    was hit. HEAL's depth target clamps rather than masks, so a sparse
    map would teach "nearest bin" on every empty pixel.

Poses go through :func:`pose_from_matrix`, the exact inverse of HEAL's
``x_to_world`` (ported here as :func:`matrix_from_pose`), so the loader
reconstructs our 4x4s bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]

CANON_W, CANON_H = 800, 600
CLASSES = ("car", "bicycle", "pedestrian", "other")
FAR_DEPTH = 255


def matrix_from_pose(pose: list[float] | F64) -> F64:
    """HEAL's ``x_to_world``: ``[x, y, z, roll, yaw, pitch]`` deg -> 4x4."""
    x, y, z, roll, yaw, pitch = (float(v) for v in pose)
    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    m[0, 0] = c_p * c_y
    m[0, 1] = c_y * s_p * s_r - s_y * c_r
    m[0, 2] = -c_y * s_p * c_r - s_y * s_r
    m[1, 0] = s_y * c_p
    m[1, 1] = s_y * s_p * s_r + c_y * c_r
    m[1, 2] = -s_y * s_p * c_r + c_y * s_r
    m[2, 0] = s_p
    m[2, 1] = -c_p * s_r
    m[2, 2] = c_p * c_r
    return m


def pose_from_matrix(m: F64) -> list[float]:
    """Inverse of :func:`matrix_from_pose` for a proper rotation."""
    r = m[:3, :3]
    if np.linalg.det(r) < 0:
        raise ValueError("improper rotation (det < 0) cannot be a HEAL pose")
    pitch = np.degrees(np.arcsin(np.clip(r[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll = np.degrees(np.arctan2(-r[2, 1], r[2, 2]))
    return [
        float(m[0, 3]),
        float(m[1, 3]),
        float(m[2, 3]),
        float(roll),
        float(yaw),
        float(pitch),
    ]


@dataclass
class ObjectRecord:
    """One box in the agent's LiDAR frame: centre, l w h, yaw (rad)."""

    obj_id: str
    cls: str
    centre: F64
    lwh: F64
    yaw: float
    extra: dict[str, object] = field(default_factory=dict)  # e.g. smpl_id


@dataclass
class CameraRecord:
    """One camera: OpenCV frame (z forward), ``lidar_from_cam`` 4x4."""

    name: str
    image: Image.Image
    intrinsic: F64
    lidar_from_cam: F64


@dataclass
class FrameRecord:
    """One agent at one timestamp, everything in its LiDAR frame."""

    scenario: str
    cav: str
    index: int
    timestamp_s: float
    points: F32  # (N, 4) x y z intensity, intensity 0..1
    world_from_lidar: F64
    objects: list[ObjectRecord] = field(default_factory=list)
    cameras: list[CameraRecord] = field(default_factory=list)
    speed_mps: float = 0.0


def write_pcd(path: Path, points: F32) -> None:
    """Binary PCD with x y z and the intensity packed as grey ``rgb``.

    A fifth column, when given, is written as a float field ``hag``
    (height above ground): HEAL's open3d reader ignores it, our reader
    and the ``v2b_point_channels`` patch pick it up.
    """
    n = len(points)
    extra = points.shape[1] - 4
    grey = np.clip(points[:, 3] * 255.0, 0, 255).astype(np.uint32)
    rgb = (grey << 16) | (grey << 8) | grey
    rgb_f = rgb.view(np.float32)
    rows = np.empty((n, 4 + extra), dtype=np.float32)
    rows[:, :3] = points[:, :3]
    rows[:, 3] = rgb_f
    if extra:
        rows[:, 4:] = points[:, 4:]
    fields = "x y z rgb" + (" hag" if extra else "")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {fields}\nSIZE {'4 ' * (4 + extra)}".rstrip()
        + f"\nTYPE {'F ' * (4 + extra)}".rstrip()
        + f"\nCOUNT {'1 ' * (4 + extra)}".rstrip()
        + f"\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def read_pcd(path: Path) -> F32:
    """Read back what :func:`write_pcd` wrote: ``(N, 4)`` x y z intensity
    (0..1), or ``(N, 5)`` with the ``hag`` column when the file has it."""
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline().decode("ascii")
            header.append(line)
            if line.startswith("DATA"):
                break
        n = int(next(h for h in header if h.startswith("POINTS")).split()[1])
        fields = next(h for h in header if h.startswith("FIELDS")).split()[1:]
        k = len(fields)
        rows = np.frombuffer(f.read(n * 4 * k), dtype=np.float32).reshape(n, k)
    rgb = rows[:, 3].copy().view(np.uint32)
    out = np.empty((n, k), dtype=np.float32)
    out[:, :3] = rows[:, :3]
    out[:, 3] = (rgb & 0xFF).astype(np.float32) / 255.0
    if k > 4:
        out[:, 4:] = rows[:, 4:]
    return out


def canonical_camera(
    image: Image.Image, intrinsic: F64
) -> tuple[Image.Image, F64, tuple[float, int]]:
    """Scale to 800 px wide, then crop (tall) or pad (wide) to 600 px.

    Returns the image, the adjusted intrinsics and ``(scale, crop_y)``
    so the depth map can be drawn with the same mapping.
    """
    scale = CANON_W / image.width
    new_h = int(round(image.height * scale))
    im = image.convert("RGB").resize(
        (CANON_W, new_h), Image.Resampling.BILINEAR
    )
    k = intrinsic.copy()
    k[:2] *= scale
    if new_h >= CANON_H:
        crop_y = (new_h - CANON_H) // 2
        im = im.crop((0, crop_y, CANON_W, crop_y + CANON_H))
    else:
        crop_y = 0
        canvas = Image.new("RGB", (CANON_W, CANON_H))
        canvas.paste(im, (0, 0))
        im = canvas
    k[1, 2] -= crop_y
    return im, k, (scale, crop_y)


def depth_map(
    points: F32,
    cam_from_lidar: F64,
    intrinsic: F64,
    radius_px: int = 3,
    max_m: float = 254.0,
) -> npt.NDArray[np.uint8]:
    """8-bit metres on the canonical image from LiDAR points, nearest wins.

    Each point is splatted as a ``(2 r + 1)`` square; pixels no point
    reaches are ``FAR_DEPTH``.
    """
    p = points[:, :3] @ cam_from_lidar[:3, :3].T + cam_from_lidar[:3, 3]
    z = p[:, 2]
    keep = z > 0.5
    p, z = p[keep], z[keep]
    uv = (intrinsic @ p.T).T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    inside = (u >= 0) & (u < CANON_W) & (v >= 0) & (v < CANON_H)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.full((CANON_H, CANON_W), np.inf, dtype=np.float32)
    order = np.argsort(-z)  # far first, near overwrites
    u, v, z = u[order], v[order], z[order]
    for du in range(-radius_px, radius_px + 1):
        for dv in range(-radius_px, radius_px + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < CANON_W) & (vv >= 0) & (vv < CANON_H)
            depth[vv[ok], uu[ok]] = z[ok]
    out = np.full((CANON_H, CANON_W), FAR_DEPTH, dtype=np.uint8)
    hit = np.isfinite(depth)
    out[hit] = np.clip(np.round(depth[hit]), 1, max_m).astype(np.uint8)
    return out


def objects_from_yaml(
    doc: dict[str, Any], world_from_lidar: F64
) -> list[ObjectRecord]:
    """The inverse of :func:`frame_yaml`: a label dict's boxes back in
    the LiDAR frame, with the extra keys (``synthetic``, ``smpl_id``)
    kept in ``extra``.

    The yaml's ``angle[1]`` is the world yaw, which the writer makes as
    the ego yaw of ``world_from_lidar`` plus the box's LiDAR-frame yaw,
    so the LiDAR-frame yaw is the difference. (A reader that adds the
    yaw of the inverse matrix instead doubles the ego heading.)
    """
    lidar_from_world = np.linalg.inv(world_from_lidar)
    yaw_ego = np.arctan2(world_from_lidar[1, 0], world_from_lidar[0, 0])
    known = {"location", "center", "extent", "angle", "obj_type"}
    out = []
    for key, v in (doc.get("vehicles") or {}).items():
        loc = np.append(np.asarray(v["location"], float), 1.0)
        out.append(
            ObjectRecord(
                obj_id=str(key),
                cls=str(v.get("obj_type", "car")),
                centre=(lidar_from_world @ loc)[:3],
                lwh=2.0 * np.asarray(v["extent"], float),
                yaw=float(np.radians(v["angle"][1]) - yaw_ego),
                extra={k: x for k, x in v.items() if k not in known},
            )
        )
    return out


def frame_yaml(frame: FrameRecord, cameras_k: list[F64]) -> dict[str, object]:
    """The label dict HEAL's ``load_yaml`` will read for this frame."""
    pose = pose_from_matrix(frame.world_from_lidar)
    vehicles: dict[str, dict[str, object]] = {}
    for o in frame.objects:
        world = frame.world_from_lidar @ np.append(o.centre, 1.0)
        yaw_rad = (
            np.arctan2(
                frame.world_from_lidar[1, 0], frame.world_from_lidar[0, 0]
            )
            + o.yaw
        )
        vehicles[o.obj_id] = {
            "location": [float(v) for v in world[:3]],
            "center": [0.0, 0.0, 0.0],
            "extent": [float(v) / 2 for v in o.lwh],
            "angle": [0.0, float(np.degrees(yaw_rad)), 0.0],
            "obj_type": o.cls,
            **o.extra,
        }
    out: dict[str, object] = {
        "lidar_pose": pose,
        "true_ego_pos": pose,
        "ego_speed": float(frame.speed_mps),
        "timestamp": float(frame.timestamp_s),
        "vehicles": vehicles,
    }
    for i, (cam, k) in enumerate(zip(frame.cameras, cameras_k, strict=True)):
        world_from_cam = frame.world_from_lidar @ cam.lidar_from_cam
        out[f"camera{i}"] = {
            "cords": pose_from_matrix(world_from_cam),
            "intrinsic": k.tolist(),
            "extrinsic": np.linalg.inv(cam.lidar_from_cam).tolist(),
            "frame": "opencv",
            "source": cam.name,
        }
    return out


def write_frame(
    root: Path, frame: FrameRecord, labels_only: bool = False
) -> Path:
    """Write one frame; returns its directory ``<root>/<scenario>/<cav>``.

    ``labels_only`` rewrites the yaml and leaves the point cloud and
    the images in place (a label rerun over an existing export).
    """
    d = root / frame.scenario / frame.cav
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{frame.index:06d}"
    if not labels_only:
        write_pcd(d / f"{stem}.pcd", frame.points)
    ks: list[F64] = []
    for i, cam in enumerate(frame.cameras):
        im, k, _ = canonical_camera(cam.image, cam.intrinsic)
        if not labels_only:
            im.save(d / f"{stem}_camera{i}.png")
            dm = depth_map(frame.points, np.linalg.inv(cam.lidar_from_cam), k)
            Image.fromarray(dm, mode="L").save(d / f"{stem}_depth{i}.png")
        ks.append(k)
    (d / f"{stem}.yaml").write_text(
        yaml.safe_dump(frame_yaml(frame, ks), sort_keys=False)
    )
    return d
