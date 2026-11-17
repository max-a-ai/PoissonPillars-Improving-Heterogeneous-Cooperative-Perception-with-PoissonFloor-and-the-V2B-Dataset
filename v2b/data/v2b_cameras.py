"""Camera frames of a V2B agent in the form HEAL's lift-splat expects.

The nuScenes export of a scene (``nusc/<scene>/<agent>/``) stores every
sensor's pose in ``v1.0-mini/calibrated_sensor.json`` the nuScenes way:
``rotation`` is a ``[w, x, y, z]`` quaternion and ``translation`` the
sensor origin, both sensor -> ego; cameras are OpenCV frames (z
forward, x right, y down) with a 3x3 ``camera_intrinsic``. ``LIDAR_CONC``
carries a 180 deg yaw there (``w = 0, z = 1``), the same half turn
``SWEEP_TO_POSE`` measured against the SLAM poses, so

    camera -> sweep = inv(T_ego_lidarconc) @ T_ego_camera

and its inverse projects a sweep onto its images exactly. HEAL's
encoder wants, per camera, the
image resized and cropped to ``final_dim`` and normalised with a fourth
depth channel, ``rots``/``trans`` = camera -> LiDAR, the *original*
intrinsics, and ``post_rots``/``post_trans`` describing the resize and
crop in image space, computed as its evaluation-mode
``sample_augmentation`` does but from the real image size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]

#: The cameras each agent feeds HEAL's camera agent. The car's seven
#: cover the full ring; HEAL pads every agent to the largest set.
DEFAULT_CAMERAS: dict[str, tuple[str, ...]] = {
    "car": (
        "CAMERA_FRONT_LEFT",
        "CAMERA_FRONT_RIGHT",
        "CAMERA_RIGHT_FRONT",
        "CAMERA_RIGHT_BACK",
        "CAMERA_BACK_CENTER",
        "CAMERA_LEFT_BACK",
        "CAMERA_LEFT_FRONT",
    ),
    "bike": ("CAMERA_FRONT",),
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SensorCalib:
    """One ``calibrated_sensor`` entry: sensor -> ego and intrinsics."""

    channel: str
    ego_from_sensor: F64
    intrinsic: F64 | None


def quat_to_rot(q: tuple[float, float, float, float] | list[float]) -> F64:
    """Rotation matrix of a unit quaternion given as ``[w, x, y, z]``."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def load_calibration(nusc_dir: Path) -> dict[str, SensorCalib]:
    """Every sensor of the export, keyed by channel name."""
    meta = nusc_dir / "v1.0-mini"
    channels = {
        s["token"]: str(s["channel"])
        for s in json.loads((meta / "sensor.json").read_text())
    }
    out: dict[str, SensorCalib] = {}
    for c in json.loads((meta / "calibrated_sensor.json").read_text()):
        tf = np.eye(4)
        tf[:3, :3] = quat_to_rot(c["rotation"])
        tf[:3, 3] = c["translation"]
        k = c.get("camera_intrinsic")
        ch = channels[c["sensor_token"]]
        out[ch] = SensorCalib(
            ch, tf, np.asarray(k, dtype=np.float64) if k else None
        )
    return out


def camera_to_sweep(cal: dict[str, SensorCalib], camera: str) -> F64:
    """4x4 camera -> ``LIDAR_CONC`` sweep frame (HEAL's camera_to_lidar).

    ``p_sweep = tf @ p_camera``; its inverse projects sweeps into images.
    """
    tf: F64 = (
        np.linalg.inv(cal["LIDAR_CONC"].ego_from_sensor)
        @ cal[camera].ego_from_sensor
    )
    return tf


def nearest_image(
    nusc_dir: Path, camera: str, t_ns: int
) -> tuple[Path, float]:
    """The camera's frame closest to ``t_ns`` and its offset in ms."""
    files = sorted((nusc_dir / "sensor_data" / camera).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no {camera} images under {nusc_dir}")
    times = np.array([int(f.stem) for f in files], dtype=np.int64)
    i = int(np.argmin(np.abs(times - t_ns)))
    return files[i], float(times[i] - t_ns) / 1e6


def eval_crop(
    height: int,
    width: int,
    final_dim: tuple[int, int],
    bot_pct_lim: tuple[float, float],
) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """HEAL's evaluation-mode resize and crop for an image of this size.

    Returns ``(resize, (new_w, new_h), (x0, y0, x1, y1))``: scale so the
    short side reaches ``final_dim``, then a centre crop in x and a crop
    that keeps the bottom ``mean(bot_pct_lim)`` share off the bottom.
    """
    f_h, f_w = final_dim
    resize = max(f_h / height, f_w / width)
    new_w, new_h = int(width * resize), int(height * resize)
    crop_h = int((1 - float(np.mean(bot_pct_lim))) * new_h) - f_h
    crop_w = int(max(0, new_w - f_w) / 2)
    return resize, (new_w, new_h), (crop_w, crop_h, crop_w + f_w, crop_h + f_h)


def preprocess(
    img: Image.Image,
    final_dim: tuple[int, int],
    bot_pct_lim: tuple[float, float],
) -> tuple[F32, F64, F64]:
    """One image to HEAL's ``(4, fH, fW)`` tensor plus its post transform.

    Channels are the ImageNet-normalised RGB and a zero depth channel
    (the encoder reads it only for the depth loss). ``post_rot`` (3x3)
    and ``post_tran`` (3) map original pixel coordinates to the
    network input: ``u' = resize * u - crop_origin``.
    """
    resize, dims, crop = eval_crop(
        img.height, img.width, final_dim, bot_pct_lim
    )
    im = img.convert("RGB").resize(dims, Image.Resampling.BILINEAR).crop(crop)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    f_h, f_w = final_dim
    out = np.zeros((4, f_h, f_w), dtype=np.float32)
    out[:3] = rgb.transpose(2, 0, 1)
    post_rot = np.eye(3)
    post_rot[0, 0] = post_rot[1, 1] = resize
    post_tran = np.zeros(3)
    post_tran[:2] = -np.array(crop[:2], dtype=np.float64)
    return out, post_rot, post_tran


@dataclass
class CameraInputs:
    """HEAL's per-agent camera batch, arrays stacked over the cameras."""

    imgs: F32  # (N, 4, fH, fW)
    rots: F64  # (N, 3, 3) camera -> sweep
    trans: F64  # (N, 3)
    intrins: F64  # (N, 3, 3)
    post_rots: F64  # (N, 3, 3)
    post_trans: F64  # (N, 3)
    files: list[Path]
    dt_ms: list[float]

    def padded(self, n: int, far_m: float = 1.0e4) -> CameraInputs:
        """Copy with ``n`` cameras, the extra ones placed ``far_m`` away.

        HEAL stacks all camera agents into one ``(agents, N, ...)``
        tensor, so every agent needs the same N. A padding camera
        repeats the last image but sits ``far_m`` metres up, where
        every frustum point leaves the grid's z slab and the pooling
        drops it: an exact no-op, not an approximation.
        """
        k = len(self.files)
        if n < k:
            raise ValueError(f"cannot pad {k} cameras down to {n}")
        if n == k:
            return self
        idx = list(range(k)) + [k - 1] * (n - k)
        trans = self.trans[idx].copy()
        trans[k:, 2] += far_m
        return CameraInputs(
            self.imgs[idx],
            self.rots[idx],
            trans,
            self.intrins[idx],
            self.post_rots[idx],
            self.post_trans[idx],
            [self.files[i] for i in idx],
            [self.dt_ms[i] for i in idx],
        )


def camera_inputs(
    nusc_dir: Path,
    sweep_path: Path,
    cameras: tuple[str, ...],
    final_dim: tuple[int, int],
    bot_pct_lim: tuple[float, float] = (0.0, 0.05),
) -> CameraInputs:
    """The agent's cameras nearest to a sweep, in HEAL's input form."""
    cal = load_calibration(nusc_dir)
    t_ns = int(sweep_path.name.split(".")[0])
    imgs, rots, trans, intrins, prs, pts, files, dts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for cam in cameras:
        f, dt = nearest_image(nusc_dir, cam, t_ns)
        with Image.open(f) as im:
            x, pr, pt = preprocess(im, final_dim, bot_pct_lim)
        tf = camera_to_sweep(cal, cam)
        k = cal[cam].intrinsic
        if k is None:
            raise ValueError(f"{cam} has no intrinsics in {nusc_dir}")
        imgs.append(x)
        rots.append(tf[:3, :3])
        trans.append(tf[:3, 3])
        intrins.append(k)
        prs.append(pr)
        pts.append(pt)
        files.append(f)
        dts.append(dt)
    return CameraInputs(
        np.stack(imgs),
        np.stack(rots),
        np.stack(trans),
        np.stack(intrins),
        np.stack(prs),
        np.stack(pts),
        files,
        dts,
    )
