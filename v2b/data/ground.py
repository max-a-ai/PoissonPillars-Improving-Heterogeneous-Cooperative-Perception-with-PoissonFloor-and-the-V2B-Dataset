"""Per-frame ground estimation for LiDAR sweeps, without a map or a model.

The sweep is binned into ``cell`` m cells; in every cell with enough
points the lowest ones seed a local plane (least squares on the points
within ``band`` m above the cell's low quantile), which gives the ground
height at the cell centre. Empty or thin cells are filled from their
neighbours (nearest valid cell), the field is smoothed once, and a
point is ground when it lies within ``tol`` m of the field at its x, y.
This is the Patchwork idea reduced to what BEV pillars need: a height
field that follows slopes and curbs well enough to separate a
pedestrian's 0 to 1.8 m from the road, in about 15 ms of numpy.

Two consumers:

* :func:`rebase` replaces z by height above ground minus
  ``ORIGIN_ABOVE_GROUND`` (1.9 m, the OPV2V / nuScenes LiDAR height
  the derived data and HEAL's fixed anchor height assume), so the
  ground sits at -1.9 m everywhere and a standing person's centre at
  -1.0 m, the anchor height, regardless of the road's slope;
* the ground flag drops the ground points or counts them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

ORIGIN_ABOVE_GROUND = 1.9


@dataclass
class GroundField:
    """Ground height on a regular x/y grid (``nan`` where unknown)."""

    x0: float
    y0: float
    cell: float
    height: F64  # (ny, nx)

    def at(self, xy: npt.NDArray[np.floating]) -> F64:
        """Ground height under each (x, y), bilinear in the field."""
        gx = (xy[:, 0] - self.x0) / self.cell - 0.5
        gy = (xy[:, 1] - self.y0) / self.cell - 0.5
        ny, nx = self.height.shape
        gx = np.clip(gx, 0, nx - 1.0001)
        gy = np.clip(gy, 0, ny - 1.0001)
        ix = np.floor(gx).astype(int)
        iy = np.floor(gy).astype(int)
        fx = gx - ix
        fy = gy - iy
        h = self.height
        out: F64 = (
            h[iy, ix] * (1 - fx) * (1 - fy)
            + h[iy, ix + 1] * fx * (1 - fy)
            + h[iy + 1, ix] * (1 - fx) * fy
            + h[iy + 1, ix + 1] * fx * fy
        )
        return out


def _fill_nearest(field: F64) -> F64:
    """Replace nan cells by the nearest valid cell (grid distance)."""
    valid = ~np.isnan(field)
    if valid.all():
        return field
    if not valid.any():
        return np.zeros_like(field)
    from scipy import ndimage

    idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled: F64 = field[idx[0], idx[1]]
    return filled


def fit_ground(
    points: npt.NDArray[np.floating],
    cell: float = 2.0,
    extent: float = 80.0,
    band: float = 0.3,
    min_points: int = 8,
    low_quantile: float = 0.1,
    max_slope: float = 0.35,
    reject_above: float = 0.25,
) -> GroundField:
    """Estimate the ground height field of one sweep (sensor frame).

    ``points`` is ``(N, >=3)``. Cells with fewer than ``min_points``
    points, or whose low points are not planar enough, stay unknown
    and are filled from their neighbours.

    Every cell is fitted at once: the points are sorted by cell and z
    with one composite key, the low quantile is read off the sorted z,
    and the per-cell planes come from the 3 x 3 normal equations of the
    low points (coordinates centred on the cell, so the plane's constant
    is the height at the cell centre) solved as one batch. About 15 ms
    for a 170 000-point sweep, 55 ms as a loop over the cells, the same
    field to floating-point rounding.
    """
    from scipy import ndimage

    xyz = np.asarray(points[:, :3], np.float64)
    n = int(np.ceil(2 * extent / cell))
    x0 = y0 = -extent
    ix = np.floor((xyz[:, 0] - x0) / cell).astype(int)
    iy = np.floor((xyz[:, 1] - y0) / cell).astype(int)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    key = (iy * n + ix)[inside]
    xyz = xyz[inside]
    # one sort by (cell, z) on a composite float64: cells 1024 m apart,
    # z clipped inside, resolved to 2 ** -29 m, finer than float32 z
    order = np.argsort(key * 1024.0 + np.clip(xyz[:, 2], -500.0, 500.0))
    key_s, xyz_s = key[order], xyz[order]
    z_s = xyz_s[:, 2]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], bounds])
    cnt = np.diff(np.concatenate([starts, [len(key_s)]]))
    seg = np.repeat(np.arange(len(starts)), cnt)  # the cell of each point
    enough = cnt >= min_points
    # np.quantile (linear interpolation) of each cell's z, from the sort
    pos = low_quantile * (cnt - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, cnt - 1)
    z_lo, z_hi = z_s[starts + lo], z_s[starts + hi]
    z_low = z_lo + (pos - lo) * (z_hi - z_lo)
    low = enough[seg] & (z_s <= (z_low + band)[seg])
    g = seg[low]
    cy, cx = np.divmod(key_s[starts], n)
    xc = x0 + (cx + 0.5) * cell
    yc = y0 + (cy + 0.5) * cell
    dx = xyz_s[low, 0] - xc[g]
    dy = xyz_s[low, 1] - yc[g]
    dz = z_s[low]
    m = len(starts)

    def total(w: F64 | None = None) -> F64:
        return np.asarray(np.bincount(g, weights=w, minlength=m), np.float64)

    n_low, sx, sy = total(), total(dx), total(dy)
    sxx, sxy, syy = total(dx * dx), total(dx * dy), total(dy * dy)
    sz, sxz, syz = total(dz), total(dx * dz), total(dy * dz)
    # plane z = a dx + b dy + c per cell, least squares on the low points
    normal = np.stack(
        [
            np.stack([sxx, sxy, sx], -1),
            np.stack([sxy, syy, sy], -1),
            np.stack([sx, sy, n_low], -1),
        ],
        -2,
    )
    rhs = np.stack([sxz, syz, sz], -1)
    fit = enough & (n_low >= 3)
    coef = np.einsum("kij,kj->ki", np.linalg.pinv(normal[fit]), rhs[fit])
    flat = np.hypot(coef[:, 0], coef[:, 1]) <= max_slope
    height = np.full((n, n), np.nan)
    height[cy[fit][flat], cx[fit][flat]] = coef[flat, 2]
    # a cell whose seed sits well above its neighbourhood was seeded by
    # an object's underside (a parked car fills a 2 m cell); unknown it
    # and refill from the neighbours. Curbs are 0.15 m, cars 0.3 m up.
    if reject_above > 0:
        filled = _fill_nearest(height)
        neighbourhood = ndimage.median_filter(filled, size=5, mode="nearest")
        high = ~np.isnan(height) & (height - neighbourhood > reject_above)
        height[high] = np.nan
    height = _fill_nearest(height)
    # one smoothing pass keeps curbs but removes single-cell outliers
    height = ndimage.median_filter(height, size=3, mode="nearest")
    return GroundField(x0, y0, cell, height)


class Surface(Protocol):
    """Anything that answers the ground height under (x, y)."""

    def at(self, xy: npt.NDArray[np.floating]) -> F64: ...


SURFACES = ("grid", "delaunay", "alpha", "poisson")


def ground_seeds(
    points: npt.NDArray[np.floating],
    field: GroundField,
    tol: float = 0.15,
    voxel: float = 0.25,
) -> F64:
    """The ground returns of a sweep, thinned to the lowest per ``voxel``.

    The robust grid fit picks the returns (within ``tol`` of its field,
    so a parked car's roof never seeds); keeping one, the lowest, per
    ``voxel`` x ``voxel`` column bounds the triangulation at ~20k
    vertices and drops returns from low vegetation over the road.
    """
    xyz = np.asarray(points[:, :3], np.float64)
    g = xyz[ground_mask(xyz, field, tol)]
    if len(g) == 0:
        return g
    ij = np.floor(g[:, :2] / voxel).astype(np.int64)
    key = (ij[:, 0] + (1 << 20)) * (1 << 21) + (ij[:, 1] + (1 << 20))
    order = np.lexsort((g[:, 2], key))
    first = np.concatenate([[True], np.diff(key[order]) != 0])
    seeds: F64 = g[order[first]]
    return seeds


@dataclass
class TriangulatedSurface:
    """A 2.5-D TIN over ground seeds: linear on each Delaunay triangle.

    With ``alpha`` set, only triangles whose circumradius is at most
    ``alpha`` m are kept (the alpha shape of the seeds), so the surface
    exists where ground was seen and long triangles bridging occluded
    gaps or crossing a car are dropped. Queries outside the kept
    triangles fall back to ``fallback``.
    """

    tri: Any  # scipy.spatial.Delaunay
    z: F64
    keep: npt.NDArray[np.bool_]
    fallback: GroundField

    def at(self, xy: npt.NDArray[np.floating]) -> F64:
        q = np.asarray(xy[:, :2], np.float64)
        out = self.fallback.at(q)
        simplex = self.tri.find_simplex(q)
        hit = simplex >= 0
        hit[hit] = self.keep[simplex[hit]]
        if hit.any():
            s = simplex[hit]
            t = self.tri.transform[s]
            b = np.einsum("nij,nj->ni", t[:, :2], q[hit] - t[:, 2])
            bary = np.column_stack([b, 1 - b.sum(axis=1)])
            out[hit] = (self.z[self.tri.simplices[s]] * bary).sum(axis=1)
        return out


def _circumradius(p: F64) -> F64:
    """Circumradius of each triangle ``p`` of shape (n, 3, 2)."""
    a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    b = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    area2 = np.abs(cross)
    with np.errstate(divide="ignore", invalid="ignore"):
        r: F64 = np.where(area2 > 0, a * b * c / (2 * area2), np.inf)
    return r


def triangulate(
    seeds: F64, fallback: GroundField, alpha: float | None = None
) -> Surface:
    """Delaunay TIN over ``seeds`` (alpha shape when ``alpha`` is set)."""
    if len(seeds) < 3:
        return fallback
    from scipy.spatial import Delaunay

    tri = Delaunay(seeds[:, :2])
    keep = np.ones(len(tri.simplices), bool)
    if alpha is not None:
        keep = _circumradius(seeds[tri.simplices][:, :, :2]) <= alpha
    return TriangulatedSurface(tri, seeds[:, 2].copy(), keep, fallback)


#: Frames whose Poisson child crashed or timed out and fell back to the
#: grid fit; Open3D 0.19 segfaults on a rare sweep.
POISSON_FALLBACKS = 0


def _poisson_heights(
    seeds: F64,
    x0: float,
    y0: float,
    cell: float,
    n: int,
    depth: int,
    trim_quantile: float,
) -> F64:
    """Poisson mesh of ``seeds`` ray-cast onto an ``n`` x ``n`` grid.

    Runs in a forked child (:func:`poisson_surface`): every Open3D call
    of a frame happens here, so a crash in it costs one frame, not the
    process. Cells the trimmed mesh does not cover come back ``nan``.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seeds))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    # one thread: the derive runs 32 of these side by side, and Open3D's
    # default (-1, every core) put a 128-core node at load ~500 and
    # made each frame 14x slower
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1
    )
    dens = np.asarray(density)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_quantile))
    height = np.full((n, n), np.nan)
    if len(mesh.triangles) == 0:
        return height
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    top = float(seeds[:, 2].max()) + 10.0
    c = (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(x0 + c, y0 + c)  # height[iy, ix]
    rays = np.zeros((n * n, 6), np.float32)
    rays[:, 0], rays[:, 1] = gx.ravel(), gy.ravel()
    rays[:, 2] = top
    rays[:, 5] = -1.0
    t = scene.cast_rays(o3d.core.Tensor.from_numpy(rays))["t_hit"].numpy()
    hit = np.isfinite(t)
    flat = height.ravel()
    flat[hit] = top - t[hit]
    return flat.reshape(n, n)


def poisson_height_grid(
    seeds: F64,
    x0: float,
    y0: float,
    cell: float,
    n: int,
    depth: int = 9,
    trim_quantile: float = 0.02,
) -> F64:
    """The Poisson floor of a whole map as an ``n`` x ``n`` height grid
    (``nan`` where the trimmed mesh has no surface): the per-sweep
    rasteriser run in this process, for the map-floor notebooks."""
    return _poisson_heights(seeds, x0, y0, cell, n, depth, trim_quantile)


def _poisson_child(conn: Any, *args: Any) -> None:
    conn.send(_poisson_heights(*args))
    conn.close()


def poisson_surface(
    seeds: F64,
    fallback: GroundField,
    depth: int = 9,
    trim_quantile: float = 0.05,
    cell: float = 0.2,
    timeout: float = 120.0,
) -> Surface:
    """Screened Poisson reconstruction of the ground seeds (Open3D).

    Normals point up (the ground is seen from above); vertices whose
    Poisson density is in the lowest ``trim_quantile`` are removed, so
    the watertight surface does not balloon over unseen areas. The mesh
    is read back as a ``cell`` m height grid by vertical ray casting,
    in a forked child process; cells it does not cover, and the whole
    frame when the child crashes or exceeds ``timeout`` s, fall back to
    the grid field.
    """
    global POISSON_FALLBACKS
    if len(seeds) < 50:
        return fallback
    import multiprocessing as mp

    lo = np.floor(seeds[:, :2].min(axis=0) / cell) * cell - cell
    span = float((seeds[:, :2].max(axis=0) - lo).max()) + 2 * cell
    n = int(np.ceil(span / cell))
    # a fork is cheapest, but a fork of a process whose thread pools are
    # already up (torch imported: the trainer, the inference scripts)
    # can deadlock in the child; spawn then, the seeds travel pickled
    start = os.environ.get("V2B_POISSON_CONTEXT") or (
        "spawn" if "torch" in sys.modules else "fork"
    )
    ctx: Any = mp.get_context(start)
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(
        target=_poisson_child,
        args=(
            send,
            seeds,
            float(lo[0]),
            float(lo[1]),
            cell,
            n,
            depth,
            trim_quantile,
        ),
    )
    child.start()
    send.close()
    height: F64 | None = None
    try:
        if recv.poll(timeout):
            height = recv.recv()
    except (EOFError, OSError):
        height = None
    child.join(timeout=5)
    if child.is_alive():
        child.kill()
        child.join()
    if height is None:
        POISSON_FALLBACKS += 1
        return fallback
    c = (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(float(lo[0]) + c, float(lo[1]) + c)
    holes = np.isnan(height)
    if holes.any():
        q = np.column_stack([gx[holes], gy[holes]])
        height[holes] = fallback.at(q)
    return _Clamped(
        GroundField(float(lo[0]), float(lo[1]), cell, height), fallback
    )


@dataclass
class _Clamped:
    """A fine field inside its own extent, the fallback outside it."""

    fine: GroundField
    fallback: GroundField

    def at(self, xy: npt.NDArray[np.floating]) -> F64:
        q = np.asarray(xy[:, :2], np.float64)
        n = self.fine.height.shape[0]
        hi = self.fine.cell * n
        inside = (
            (q[:, 0] >= self.fine.x0)
            & (q[:, 0] < self.fine.x0 + hi)
            & (q[:, 1] >= self.fine.y0)
            & (q[:, 1] < self.fine.y0 + hi)
        )
        out = self.fallback.at(q)
        if inside.any():
            out[inside] = self.fine.at(q[inside])
        return out


def fit_surface(
    points: npt.NDArray[np.floating],
    method: str = "grid",
    alpha: float = 1.0,
    tol: float = 0.15,
    depth: int = 9,
) -> Surface:
    """The ground of one sweep under one floor-reconstruction ``method``.

    ``grid`` is :func:`fit_ground` (2 m local planes); ``delaunay`` a
    TIN over its ground returns, ``alpha`` the same restricted to the
    alpha shape (circumradius <= ``alpha`` m), ``poisson`` a screened
    Poisson mesh of them at octree ``depth`` (the paper's floor is
    ``depth`` 9; Tab. floor's depth rows use 5 to 8). The three finer
    surfaces follow curbs and camber within a cell and fall back to the
    grid where they have no support.
    """
    if method not in SURFACES:
        raise ValueError(f"surface {method!r}, expected one of {SURFACES}")
    field = fit_ground(points)
    if method == "grid":
        return field
    seeds = ground_seeds(points, field, tol)
    if method == "delaunay":
        return triangulate(seeds, field)
    if method == "alpha":
        return triangulate(seeds, field, alpha)
    return poisson_surface(seeds, field, depth=depth)


def ground_mask(
    points: npt.NDArray[np.floating], field: Surface, tol: float = 0.15
) -> npt.NDArray[np.bool_]:
    """True for points within ``tol`` m of the ground field."""
    hag = points[:, 2] - field.at(points[:, :2])
    return np.asarray(np.abs(hag) <= tol)


def rebase(points: F32, field: Surface) -> F32:
    """Copy of ``points`` with z = height above ground - 1.9 m."""
    out = points.copy()
    out[:, 2] = (
        points[:, 2] - field.at(points[:, :2]) - ORIGIN_ABOVE_GROUND
    ).astype(np.float32)
    return out


def rebased_pose(world_from_lidar: F64, field: Surface) -> F64:
    """The sensor pose that keeps a re-based sweep in place in the world.

    :func:`rebase` moves every point by ``-(ground + 1.9)`` along the
    sensor's z, so the origin of the re-based frame sits 1.9 m above
    the ground under the sensor. A cooperative loader projects agents
    into each other's frames with their poses; with the original pose
    an infrastructure sweep (sensor 4.5 m up) would land 2.6 m above a
    vehicle's re-based floor. The returned pose is the original one
    with its origin moved to that virtual sensor: ``t' = t + R (0, 0,
    ground(0, 0) + 1.9)``. Agents then agree to within the ground's
    unevenness between them, which the per-agent re-base flattened.
    """
    g0 = float(field.at(np.zeros((1, 2)))[0])
    out = np.array(world_from_lidar, float, copy=True)
    out[:3, 3] += out[:3, :3] @ np.array([0.0, 0.0, g0 + ORIGIN_ABOVE_GROUND])
    return out


def rebase_z(xy: npt.NDArray[np.floating], z: float, field: Surface) -> float:
    """The re-based z of one object centre at ``xy``."""
    return float(
        z - field.at(np.asarray(xy, float)[None, :])[0] - ORIGIN_ABOVE_GROUND
    )


def floor_points(
    points: npt.NDArray[np.floating],
    ground: npt.NDArray[np.bool_],
    cell: float = 0.4,
) -> F32:
    """The floor as one point per ``cell``: the ground returns of a cell
    collapse to a single point at the cell centre, at their median
    height, with their median intensity (and the mean of any further
    columns). Non-ground points pass through unchanged. The minimal
    footprint of the floor in a pillar grid of the same cell size: one
    of the pillar's slots instead of most of them.
    """
    pts = np.asarray(points, np.float32)
    above = pts[~ground]
    g = pts[ground]
    if len(g) == 0:
        return above
    ij = np.floor(g[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + (1 << 20)) * (1 << 21) + (ij[:, 1] + (1 << 20))
    order = np.argsort(key, kind="stable")
    key_s, g_s = key[order], g[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key_s)]])
    out = np.empty((len(starts), pts.shape[1]), np.float32)
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        grp = g_s[a:b]
        i, j = divmod(int(key_s[a]), 1 << 21)
        out[k, 0] = (i - (1 << 20) + 0.5) * cell
        out[k, 1] = (j - (1 << 20) + 0.5) * cell
        out[k, 2] = np.median(grp[:, 2])
        out[k, 3] = np.median(grp[:, 3])
        if pts.shape[1] > 4:
            out[k, 4:] = grp[:, 4:].mean(axis=0)
    return np.concatenate([above, out], axis=0)
