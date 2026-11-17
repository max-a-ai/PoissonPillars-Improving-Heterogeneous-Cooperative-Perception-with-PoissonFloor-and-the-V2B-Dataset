"""The floor once per scene: a Poisson surface of the ground returns of
every sweep pooled in the map frame, looked up at run time.

Per sweep the floor costs a fit (Tab. floor: 718 ms for the screened
Poisson surface). The ground is static, so a scene's floor can be
built once from all its sweeps, in the frame the poses live in, and
each later point re-based by a height lookup at its map x/y. This
module builds such floors (:func:`build_floor`), writes and reads them
(:func:`save_floor`, :func:`load_floor`), and wraps one as a
:class:`MapSurface` that answers ``at(xy)`` in a sensor frame, so the
per-sweep code path (``rebase``, ``rebased_pose``, ``rebase_z``) runs
unchanged on it.

``--leave-out`` floors exist for the question whether the result
depends on the scored frame being part of the map: the scene's frames
are cut into windows of ``window`` seconds and every window gets a
floor built from the frames outside ``[start - margin, end + margin]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from v2b.data.ground import (
    GroundField,
    Surface,
    fit_ground,
    ground_seeds,
    poisson_surface,
)

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

SEED_CELL = 0.25  # pooled seeds thinned to the lowest per cell [m]


@dataclass
class MapFloor:
    """A scene's floor as a height grid in a local map frame."""

    origin: F64  # (2,) map x/y subtracted before the grid
    field: GroundField  # heights relative to ``origin``, map z
    seeds: int
    frames: int
    span: tuple[float, float] | None = None  # the window it serves (s)
    frame_span: tuple[int, int] | None = None  # the same as frame indices

    def at_map(self, xy_map: F64) -> F64:
        """Map-frame ground height under map-frame x/y."""
        q = np.asarray(xy_map, np.float64)[:, :2] - self.origin
        return np.asarray(self.field.at(q), np.float64)


def seeds_of_sweep(points: F32, world_from_lidar: F64) -> F64:
    """The ground seeds of one sweep (its own robust fit) in the map."""
    field = fit_ground(points)
    seeds = ground_seeds(points, field)
    xyz = np.c_[seeds[:, :3], np.ones(len(seeds))] @ world_from_lidar.T
    return np.asarray(xyz[:, :3], np.float64)


def thin(seeds: F64, cell: float = SEED_CELL) -> F64:
    """One seed per ``cell`` x ``cell`` column, the lowest."""
    if len(seeds) == 0:
        return seeds
    key = np.floor(seeds[:, :2] / cell).astype(np.int64)
    order = np.lexsort((seeds[:, 2], key[:, 1], key[:, 0]))
    s = seeds[order]
    k = key[order]
    first = np.ones(len(s), bool)
    first[1:] = np.any(k[1:] != k[:-1], axis=1)
    return np.asarray(s[first], np.float64)


def build_floor(seed_sets: list[F64], frames: int) -> MapFloor | None:
    """One Poisson floor from the pooled seeds of ``seed_sets``."""
    pooled = thin(np.concatenate([s for s in seed_sets if len(s)]))
    if len(pooled) < 200:
        return None
    origin = np.floor(pooled[:, :2].mean(axis=0))
    local = pooled.copy()
    local[:, :2] -= origin
    extent = float(np.abs(local[:, :2]).max()) + 10.0
    fallback = fit_ground(local, extent=extent)
    surface = poisson_surface(local, fallback)
    field = (
        surface
        if isinstance(surface, GroundField)
        else _flatten(surface, local, fallback)
    )
    return MapFloor(
        origin=origin, field=field, seeds=len(pooled), frames=frames
    )


def _flatten(
    surface: Surface, local: F64, fallback: GroundField
) -> GroundField:
    """Read a Poisson surface back into one dense grid over the seeds'
    extent, so it can be saved as arrays."""
    cell = 0.2
    lo = np.floor(local[:, :2].min(axis=0) / cell) * cell - 2.0
    hi = np.ceil(local[:, :2].max(axis=0) / cell) * cell + 2.0
    n = int(np.ceil(max(hi[0] - lo[0], hi[1] - lo[1]) / cell))
    c = (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(lo[0] + c, lo[1] + c)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    height = surface.at(xy).reshape(n, n)
    del fallback
    return GroundField(float(lo[0]), float(lo[1]), cell, height)


def save_floor(floor: MapFloor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin=floor.origin,
        x0=floor.field.x0,
        y0=floor.field.y0,
        cell=floor.field.cell,
        height=floor.field.height.astype(np.float32),
        seeds=floor.seeds,
        frames=floor.frames,
        span=np.asarray(floor.span if floor.span else (np.nan, np.nan)),
        frame_span=np.asarray(
            floor.frame_span if floor.frame_span else (-1, -1), np.int64
        ),
    )
    return path


def load_floor(path: Path) -> MapFloor:
    z = np.load(path)
    span = tuple(float(v) for v in z["span"])
    fspan = (
        tuple(int(v) for v in z["frame_span"])
        if "frame_span" in z
        else (-1, -1)
    )
    return MapFloor(
        origin=np.asarray(z["origin"], np.float64),
        field=GroundField(
            float(z["x0"]),
            float(z["y0"]),
            float(z["cell"]),
            z["height"].astype(np.float64),
        ),
        seeds=int(z["seeds"]),
        frames=int(z["frames"]),
        span=None if np.isnan(span[0]) else (span[0], span[1]),
        frame_span=None if fspan[0] < 0 else (fspan[0], fspan[1]),
    )


def floor_for(
    floors: list[MapFloor], timestamp: float, frame: int | None = None
) -> MapFloor:
    """The floor whose window holds the frame: by ``frame`` index when
    given (exports of the same scene can carry timestamps in different
    units), else by ``timestamp``; a floor without a window serves every
    frame."""
    for f in floors:
        if f.span is None:
            return f
        if frame is not None and f.frame_span is not None:
            if f.frame_span[0] <= frame <= f.frame_span[1]:
                return f
        elif f.span[0] <= timestamp <= f.span[1]:
            return f
    raise KeyError(f"no map floor window holds frame {frame} / {timestamp}")


class MapSurface:
    """A scene floor seen from one sensor pose: ``at(xy)`` takes sensor
    frame x/y and returns the ground's z in the sensor frame, so the
    per-sweep re-base code runs on it unchanged.

    The map x/y under a sensor point are taken at sensor z = 0; for the
    level poses of the V2B export that is exact, for a tilted pose the
    error is the tilt times the ground's height offset, millimetres.
    """

    def __init__(self, floor: MapFloor, world_from_lidar: F64) -> None:
        self.floor = floor
        self.m = np.asarray(world_from_lidar, np.float64)
        if abs(self.m[2, 2]) < 0.5:
            raise ValueError("sensor z axis nearly horizontal; no lookup")

    def at(self, xy: npt.NDArray[np.floating]) -> F64:
        q = np.asarray(xy, np.float64)[:, :2]
        r, t = self.m[:3, :3], self.m[:3, 3]
        wx = r[0, 0] * q[:, 0] + r[0, 1] * q[:, 1] + t[0]
        wy = r[1, 0] * q[:, 0] + r[1, 1] * q[:, 1] + t[1]
        h = self.floor.at_map(np.column_stack([wx, wy]))
        # world z of the ground = t_z + r20 x + r21 y + r22 z_g
        return np.asarray(
            (h - t[2] - r[2, 0] * q[:, 0] - r[2, 1] * q[:, 1]) / r[2, 2],
            np.float64,
        )


def lookup_cost(
    floor: MapFloor, points: F32, world_from_lidar: F64
) -> dict[str, Any]:
    """Re-base one sweep by lookup and return what it took."""
    import time

    t0 = time.perf_counter()
    surf = MapSurface(floor, world_from_lidar)
    z = points[:, 2] - surf.at(points[:, :2]) - 1.9
    t1 = time.perf_counter()
    return {
        "points": int(len(points)),
        "ms": (t1 - t0) * 1e3,
        "z_mean": float(z.mean()),
    }
