"""Loading and plotting helpers for 3D boxes over LiDAR point clouds.

Handles the four label formats present on the NAS:

===================  ==========================================================
``opv2v_yaml``       OPV2V / V2XSet / V2V4Real. Boxes are in *world* frame, so
                     they are transformed into the agent's LiDAR frame using
                     the frame's ``lidar_pose``.
``dair_json``        DAIR-V2X. Boxes already sit in the sensor frame.
``openlabel_json``   TUMTraf-V2X. ASAM OpenLABEL; cuboid is
                     ``[x, y, z, qx, qy, qz, qw, l, w, h]``.
``kitti_txt``        V2V4Real KITTI export.
===================  ==========================================================

Plotting is Plotly, so it renders inline in Jupyter over SSH with no
GL context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import yaml
from pypcd4 import PointCloud  # type: ignore[attr-defined]

_REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = _REPO / "config-global.json"
COLOR_CODE_PATH = _REPO / "configs" / "color-code.json"

# 12 edges of a cuboid, as index pairs into the 8 corners produced by
# Box.corners()
_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # verticals
)


def _strip_jsonc(text: str) -> str:
    """Remove ``//`` line comments from JSONC, leaving anything inside strings
    alone.
    """
    out: list[str] = []
    in_string = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_color_code() -> dict[str, Any]:
    """Read ``color-code.json`` (JSONC - line comments allowed)."""
    cfg: dict[str, Any] = json.loads(_strip_jsonc(COLOR_CODE_PATH.read_text()))
    return cfg


COLOR_CODE = load_color_code()
CLASS_COLORS: dict[str, str] = COLOR_CODE["classes"]
DEFAULT_COLOR: str = COLOR_CODE["default_class_color"]
SETTINGS: dict[str, Any] = COLOR_CODE["settings"]
COLORMAPS: dict[str, str] = COLOR_CODE["colormaps"]


@dataclass
class Box:
    """One 3D box in the frame of the point cloud it belongs to.

    ``center`` is the geometric centre (x, y, z); ``size`` is (length, width,
    height) in metres; ``yaw`` is rotation about +z in radians.
    """

    center: np.ndarray
    size: np.ndarray
    yaw: float
    label: str = "unknown"

    def corners(self) -> np.ndarray:
        """Return the 8 corners as a (8, 3) array, bottom face first."""
        length, width, height = self.size
        x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2
        y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * width / 2
        z = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * height / 2
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        corners: np.ndarray = (rot @ np.vstack([x, y, z])).T + self.center
        return corners

    @property
    def color(self) -> str:
        return CLASS_COLORS.get(self.label.lower(), DEFAULT_COLOR)


def load_config() -> dict[str, Any]:
    """Read the project's dataset registry (JSONC - line comments allowed)."""
    cfg: dict[str, Any] = json.loads(_strip_jsonc(CONFIG_PATH.read_text()))
    return cfg


def dataset_path(name: str) -> Path:
    """Where one registered dataset is on this machine.

    ``resources/data/<name>`` is the link ``dm_link.py`` builds; when it
    is absent the NAS path from ``config-global.json`` is returned, so a
    caller can still report "not linked here" with a concrete path.
    """
    cfg = load_config()
    if name not in cfg["datasets"]:
        raise KeyError(
            f"{name!r} not in config-global.json "
            f"({', '.join(cfg['datasets'])})"
        )
    link = _REPO / "resources" / "data" / name
    if link.exists():
        return link
    return Path(str(cfg["nas_root"])) / str(cfg["datasets"][name]["nas_path"])


def dataset_available(name: str) -> bool:
    """True when the dataset's directory is reachable on this machine."""
    return dataset_path(name).is_dir()


def load_pcd(path: str | Path) -> np.ndarray:
    """Read a .pcd into an (N, 3) float array. Handles ascii and
    binary_compressed.
    """
    arr = PointCloud.from_path(str(path)).numpy(("x", "y", "z"))
    out: np.ndarray = np.asarray(arr, dtype=np.float32)
    return out


def load_pcd_intensity(path: str | Path) -> np.ndarray | None:
    """Per-point intensity, or ``None`` when the file has no such field.

    OPV2V, V2XSet and V2V4Real ship ``x y z rgb`` and carry no intensity;
    DAIR-V2X and TUMTraf do.
    """
    pc = PointCloud.from_path(str(path))
    if "intensity" not in pc.fields:
        return None
    vals: np.ndarray = np.asarray(
        pc.numpy(("intensity",)), dtype=np.float32
    ).ravel()
    return vals


def point_colors(
    points: np.ndarray,
    mode: str | None = None,
    intensity: np.ndarray | None = None,
    dataset: str | None = None,
) -> tuple[np.ndarray, str]:
    """Values to colour points by, plus the mode actually used.

    `mode` defaults to ``settings.point_cloud_coloring`` in
    ``color-code.json``. Asking for ``intensity`` on data that has none
    prints a notice and falls back
    to ``settings.fallback_coloring``.
    """
    mode = (mode or SETTINGS["point_cloud_coloring"]).lower()

    if mode == "intensity" and intensity is None:
        name = dataset or "unknown"
        print(f"intensity values are not available in current dataset: {name}")
        mode = str(SETTINGS["fallback_coloring"]).lower()

    if mode == "intensity" and intensity is not None:
        return intensity, "intensity"
    if mode == "distance":
        return np.linalg.norm(points[:, :3], axis=1), "distance"
    if mode != "height":
        raise ValueError(
            f"unknown coloring mode {mode!r}; use intensity, height or "
            f"distance"
        )
    return points[:, 2], "height"


# --------------------------------------------------------------------------- #
# label readers
# --------------------------------------------------------------------------- #
def _pose_to_matrix(pose: list[float]) -> np.ndarray:
    """CARLA ``[x, y, z, roll, yaw, pitch]`` (degrees) → 4x4
    world-from-sensor.
    """
    x, y, z, roll, yaw, pitch = pose
    r, y_, p = np.radians([roll, yaw, pitch])
    cr, sr, cy, sy, cp, sp = (
        np.cos(r),
        np.sin(r),
        np.cos(y_),
        np.sin(y_),
        np.cos(p),
        np.sin(p),
    )
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, [x, y, z]
    return m


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read an OPV2V-family yaml.

    Some V2XSet / V2V4Real files embed pickled numpy arrays
    (``!!python/object/apply:numpy...``), which ``safe_load`` refuses.
    Fall back to the permissive loader for those, as OpenCOOD itself
    does. Only ever point
    this at dataset files you trust.
    """
    text = path.read_text()
    try:
        loaded = yaml.safe_load(text)
    except yaml.constructor.ConstructorError:
        loaded = yaml.load(text, Loader=yaml.Loader)
    result: dict[str, Any] = dict(loaded)
    return result


def read_opv2v(label_path: str | Path) -> list[Box]:
    """OPV2V-family yaml → boxes in the LiDAR frame of that yaml's agent.

    The family is not uniform:

    * OPV2V / V2XSet store ``lidar_pose`` as
      ``[x, y, z, roll, yaw, pitch]`` and
      put the boxes in **world** coordinates, so they need transforming.
    * V2V4Real stores ``lidar_pose`` as a 4x4 matrix and already gives boxes in
      the **ego LiDAR** frame, so they are used as-is.

    The shape of ``lidar_pose`` is what distinguishes the two.
    """
    d = _load_yaml(Path(label_path))
    pose = np.asarray(d["lidar_pose"], dtype=float)
    world_frame = pose.size == 6  # 6-vector => OPV2V/V2XSet convention
    lidar_from_world = (
        np.linalg.inv(_pose_to_matrix(pose.ravel().tolist()))
        if world_frame
        else None
    )
    ego_yaw = float(pose.ravel()[4]) if world_frame else 0.0

    boxes: list[Box] = []
    for veh in (d.get("vehicles") or {}).values():
        loc = np.asarray(veh["location"], dtype=float).ravel()
        off = np.asarray(veh["center"], dtype=float).ravel()
        ext = np.asarray(veh["extent"], dtype=float).ravel()
        ang = np.asarray(veh["angle"], dtype=float).ravel()
        centre = loc + off
        if lidar_from_world is not None:
            centre = (lidar_from_world @ np.append(centre, 1.0))[:3]
        yaw = np.radians(ang[1] - ego_yaw)
        label = str(veh.get("obj_type", "car")).lower()
        # CARLA reports half-sizes in "extent"
        boxes.append(Box(centre, ext * 2.0, float(yaw), label))
    return boxes


def read_dair(label_path: str | Path) -> list[Box]:
    """DAIR-V2X json → boxes, already in the sensor frame."""
    boxes: list[Box] = []
    for o in json.loads(Path(label_path).read_text()):
        loc, dim = o["3d_location"], o["3d_dimensions"]
        boxes.append(
            Box(
                np.array([loc["x"], loc["y"], loc["z"]], dtype=float),
                np.array([dim["l"], dim["w"], dim["h"]], dtype=float),
                float(o.get("rotation", 0.0)),
                str(o.get("type", "unknown")),
            )
        )
    return boxes


def read_openlabel(
    label_path: str | Path, frame: str | None = None
) -> list[Box]:
    """TUMTraf OpenLABEL json → boxes for one frame (default: the first)."""
    frames = json.loads(Path(label_path).read_text())["openlabel"]["frames"]
    key = frame if frame is not None else next(iter(frames))
    boxes: list[Box] = []
    for obj in (frames[key].get("objects") or {}).values():
        cuboid = obj["object_data"]["cuboid"]
        val = (cuboid[0] if isinstance(cuboid, list) else cuboid)["val"]
        x, y, z, qx, qy, qz, qw, length, width, height = val
        # yaw about +z straight from the quaternion
        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )
        boxes.append(
            Box(
                np.array([x, y, z], dtype=float),
                np.array([length, width, height], dtype=float),
                float(yaw),
                str(obj["object_data"].get("type", "unknown")),
            )
        )
    return boxes


def read_kitti(label_path: str | Path) -> list[Box]:
    """KITTI-style txt → boxes. Columns: type … h w l x y z ry."""
    boxes: list[Box] = []
    for line in Path(label_path).read_text().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        h, w, length, x, y, z, ry = (float(v) for v in f[8:15])
        boxes.append(
            Box(np.array([x, y, z]), np.array([length, w, h]), ry, f[0])
        )
    return boxes


READERS = {
    "opv2v_yaml": read_opv2v,
    "dair_json": read_dair,
    "openlabel_json": read_openlabel,
    "kitti_txt": read_kitti,
}


def read_labels(label_path: str | Path, fmt: str) -> list[Box]:
    """Dispatch to the reader for `fmt` (see READERS)."""
    if fmt not in READERS:
        raise KeyError(f"no reader for {fmt!r}; have {', '.join(READERS)}")
    return READERS[fmt](label_path)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_scene(
    points: np.ndarray,
    boxes: list[Box] | None = None,
    title: str = "",
    color_by: str | None = None,
    intensity: np.ndarray | None = None,
    dataset: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Interactive 3D plot of a point cloud with wireframe boxes.

    Colouring and sizing come from ``color-code.json``; `color_by`
    overrides the
    configured mode for one call. Point clouds are randomly subsampled to
    ``settings.max_points`` so the browser stays responsive.
    """
    cap = max_points or int(SETTINGS["max_points"])
    idx = np.arange(len(points))
    if len(points) > cap:
        idx = np.random.default_rng(0).choice(len(points), cap, replace=False)
    pts = points[idx]
    inten = intensity[idx] if intensity is not None else None

    values, mode = point_colors(pts, color_by, inten, dataset)
    marker: dict[str, Any] = {
        "size": SETTINGS["point_size"],
        "opacity": SETTINGS["point_opacity"],
        "color": values,
        "colorscale": COLORMAPS.get(mode, "Viridis"),
        "showscale": False,
    }

    data: list[go.Scatter3d] = [
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=marker,
            name=f"points ({mode})",
            hoverinfo="skip",
        )
    ]

    for b in boxes or []:
        c = b.corners()
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for i, j in _EDGES:
            xs += [c[i, 0], c[j, 0], None]
            ys += [c[i, 1], c[j, 1], None]
            zs += [c[i, 2], c[j, 2], None]
        data.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": b.color, "width": SETTINGS["box_line_width"]},
                name=b.label,
                legendgroup=b.label,
                showlegend=False,
                hovertext=f"{b.label} {b.size.round(2)}",
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title
        or f"{len(points):,} points, {len(boxes or [])} boxes "
        f"- coloured by {mode}",
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (m)"},
            "yaxis": {"title": "y (m)"},
            "zaxis": {"title": "z (m)"},
        },
        margin={"l": 0, "r": 0, "t": 32, "b": 0},
        height=720,
        template=SETTINGS["template"],
    )
    return fig


def plot_bev(
    points: np.ndarray,
    boxes: list[Box] | None = None,
    title: str = "",
    color_by: str | None = None,
    intensity: np.ndarray | None = None,
    dataset: str | None = None,
) -> go.Figure:
    """Top-down bird's-eye view - faster to read than 3D for checking
    alignment.
    """
    values, mode = point_colors(points, color_by, intensity, dataset)
    fig = go.Figure()
    fig.add_scattergl(
        x=points[:, 0],
        y=points[:, 1],
        mode="markers",
        marker={
            "size": 1,
            "color": values,
            "colorscale": COLORMAPS.get(mode, "Viridis"),
            "opacity": 0.6,
        },
        name=f"points ({mode})",
        hoverinfo="skip",
    )
    for b in boxes or []:
        c = b.corners()[:4]  # bottom face only
        fig.add_scattergl(
            x=list(c[:, 0]) + [c[0, 0]],
            y=list(c[:, 1]) + [c[0, 1]],
            mode="lines",
            line={"color": b.color, "width": 2},
            name=b.label,
            showlegend=False,
            hovertext=b.label,
            hoverinfo="text",
        )
    fig.update_layout(
        title=title or f"BEV - coloured by {mode}",
        height=720,
        template=SETTINGS["template"],
        xaxis={"title": "x (m)", "scaleanchor": "y"},
        yaxis={"title": "y (m)"},
        margin={"l": 0, "r": 0, "t": 32, "b": 0},
    )
    return fig
