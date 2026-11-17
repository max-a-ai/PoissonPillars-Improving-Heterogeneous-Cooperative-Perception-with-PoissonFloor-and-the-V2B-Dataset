"""Rigid transforms, quaternions and TUM-format trajectories.

The V2B pipeline stores every localised ego pose in TUM format
(``timestamp tx ty tz qx qy qz qw``), one file per recording under
``2_processed_rosbags/bagfile_to_nusc/*.tum``. Poses live in the frame
of that scene's fused SLAM map, which is a *local* metric frame whose
origin is the SLAM start pose -- see :mod:`v2b.maps.georef` for turning
it into a global (UTM) frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def quat_to_rotation(quat: F64) -> F64:
    """Convert ``(qx, qy, qz, qw)`` quaternions to rotation matrices.

    Accepts a single quaternion of shape ``(4,)`` or a batch ``(N, 4)``
    and returns ``(3, 3)`` or ``(N, 3, 3)`` respectively. Input is
    normalised first, so slightly non-unit quaternions from a SLAM
    backend are handled without complaint.
    """
    single = quat.ndim == 1
    q = np.atleast_2d(np.asarray(quat, dtype=np.float64))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot[0] if single else rot


def yaw_from_quat(quat: F64) -> F64:
    """Extract the ground-plane heading (rad) from ``(qx,qy,qz,qw)``.

    Returns a scalar array for a single quaternion, otherwise ``(N,)``.
    """
    rot = quat_to_rotation(quat)
    if rot.ndim == 2:
        return np.asarray(np.arctan2(rot[1, 0], rot[0, 0]), dtype=np.float64)
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float64)


@dataclass(frozen=True)
class Trajectory:
    """A localised ego trajectory in a scene's SLAM-map frame.

    Attributes
    ----------
    timestamps:
        Unix seconds, shape ``(N,)``, strictly increasing.
    positions:
        ``(N, 3)`` metres in the map frame.
    quaternions:
        ``(N, 4)`` as ``(qx, qy, qz, qw)``.
    name:
        Free-form label, e.g. ``"bike"`` or ``"car"``.
    """

    timestamps: F64
    positions: F64
    quaternions: F64
    name: str = ""

    def __post_init__(self) -> None:
        n = self.timestamps.shape[0]
        if self.positions.shape != (n, 3):
            raise ValueError(
                f"positions must be (N, 3), got {self.positions.shape}"
            )
        if self.quaternions.shape != (n, 4):
            raise ValueError(
                f"quaternions must be (N, 4), got {self.quaternions.shape}"
            )

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def xy(self) -> F64:
        """Ground-plane track, shape ``(N, 2)``."""
        return self.positions[:, :2]

    @property
    def yaw(self) -> F64:
        """Heading per pose in radians, shape ``(N,)``."""
        return np.atleast_1d(yaw_from_quat(self.quaternions))

    @property
    def arclength(self) -> F64:
        """Cumulative distance travelled in the ground plane, ``(N,)``.

        Starts at 0.0 for the first pose.
        """
        steps = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)]).astype(np.float64)

    @property
    def speed(self) -> F64:
        """Ground-plane speed in m/s per pose, shape ``(N,)``.

        Uses central differences internally and one-sided differences at
        the ends, so the result is the same length as the trajectory.
        """
        if len(self) < 2:
            return np.zeros(len(self), dtype=np.float64)
        dist = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        dt = np.diff(self.timestamps)
        # Guard against duplicate timestamps from a stalled SLAM frame.
        dt = np.where(dt <= 0.0, np.nan, dt)
        seg = dist / dt
        mid = np.concatenate([[seg[0]], (seg[:-1] + seg[1:]) / 2.0, [seg[-1]]])
        filled: F64 = np.nan_to_num(mid, nan=0.0).astype(np.float64)
        return filled

    def resample_by_arclength(self, spacing: float) -> Trajectory:
        """Return a copy resampled to roughly constant ``spacing`` (m).

        Positions are linearly interpolated; orientation is taken from
        the nearest original pose, which is accurate enough for tube
        rendering and avoids a quaternion slerp dependency.
        """
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")
        s = self.arclength
        total = float(s[-1])
        if total <= 0.0:
            return self
        targets = np.arange(0.0, total, spacing, dtype=np.float64)
        pos = np.column_stack(
            [np.interp(targets, s, self.positions[:, i]) for i in range(3)]
        )
        times = np.interp(targets, s, self.timestamps)
        nearest = np.searchsorted(s, targets).clip(0, len(self) - 1)
        return Trajectory(
            timestamps=times.astype(np.float64),
            positions=pos.astype(np.float64),
            quaternions=self.quaternions[nearest],
            name=self.name,
        )


def load_tum(path: str | Path, name: str = "") -> Trajectory:
    """Read a TUM-format trajectory file.

    Each non-comment line is ``timestamp tx ty tz qx qy qz qw``. Blank
    lines and ``#`` comments are skipped.
    """
    raw = np.loadtxt(Path(path), comments="#", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 8:
        raise ValueError(
            f"{path}: expected 8 columns (TUM format), got {raw.shape[1]}"
        )
    return Trajectory(
        timestamps=raw[:, 0].copy(),
        positions=raw[:, 1:4].copy(),
        quaternions=raw[:, 4:8].copy(),
        name=name,
    )
