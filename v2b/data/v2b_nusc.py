"""A V2B scene in its nuScenes-format annotation delivery, read locally.

The v2b deliveries (``v2b/4_annotated/nusc/<scene>/v2b``) ship
one fused sweep per sample, ``LIDAR_FUSED_FUSED``: float32 ``(N, 5)``
x y z intensity agent, in the car's LiDAR frame, agent 1 = car, 2 =
bike. This module splits it back into the two agents' own clouds and
frames, so a single-agent detector sees each agent as it would on the
road:

* **car**: the car's LiDAR frame, as delivered (sensor ~2 m up);
* **bike**: the bike's ego frame, the ``BIKE_CAMERA_FRONT`` ego pose
  (origin at the bike's sensor head, ~1.2 m above the road).

Boxes come from ``sample_annotation`` in the global frame and are
returned in the car's LiDAR frame, the frame every agent's detections
are scored in. Raw Ouster intensities (median ~370, p99 ~3400) are
squashed with ``tanh(i / INTENSITY_SCALE)``: 6000 puts their median at
Waymo's 0.055 (one derived Waymo sweep: p10 / 50 / 90 = 0.016 / 0.055 /
0.20), the domain the Waymo-trained detectors saw. ``tanh(i / 255)``,
the V2X-Real export's choice, would put it at 0.93.

Pure numpy and json, no devkit: it runs in the HEAL venv as well.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from v2b.data.class_map import canon
from v2b.geometry.transforms import quat_to_rotation

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

LIDAR = "LIDAR_FUSED_FUSED"
BIKE_POSE = "BIKE_CAMERA_FRONT"
AGENT_IDS = {"car": 1, "bike": 2}
INTENSITY_SCALE = 6000.0
# a delivery whose last PARTIAL_TAIL or more frames carry no box is a
# partial delivery (labelling stopped), not a scene that ends empty
PARTIAL_TAIL = 3


def labelled_frames(n_frames: int, labelled: Iterable[int]) -> list[int]:
    """The frames of a delivery that count as labelled.

    A run of :data:`PARTIAL_TAIL` or more consecutive frames without a
    box is a stretch the annotators have not reached (a partial
    delivery), not a stretch where nothing was there, so it is left
    out; a shorter run inside labelled frames is kept as empty frames.
    A labelled run shorter than :data:`PARTIAL_TAIL` that touches an
    unlabelled stretch is left out too: a parked-car grid key sitting
    alone in an unlabelled stretch is not a fully labelled frame.
    """
    has = [False] * n_frames
    for i in labelled:
        if 0 <= i < n_frames:
            has[i] = True
    runs: list[tuple[bool, int, int]] = []  # (labelled, start, end)
    i = 0
    while i < n_frames:
        j = i
        while j < n_frames and has[j] == has[i]:
            j += 1
        runs.append((has[i], i, j))
        i = j
    # short empty runs inside labelled frames are empty frames, kept
    keep = [
        (lab or (end - start) < PARTIAL_TAIL, start, end)
        for lab, start, end in runs
    ]
    # a stranded short labelled run next to an unlabelled stretch is out
    out = []
    for k, (lab, start, end) in enumerate(keep):
        if lab and (end - start) < PARTIAL_TAIL:
            before = keep[k - 1][0] if k else True
            after = keep[k + 1][0] if k + 1 < len(keep) else True
            lab = before and after
        out.append((lab, start, end))
    return [i for lab, start, end in out if lab for i in range(start, end)]


def pose_matrix(translation: list[float], rotation_wxyz: list[float]) -> F64:
    """4x4 from a nuScenes translation and ``(w, x, y, z)`` quaternion."""
    w, x, y, z = rotation_wxyz
    m = np.eye(4)
    m[:3, :3] = quat_to_rotation(np.array([x, y, z, w], float))
    m[:3, 3] = translation
    return m


def transform(points: F32, tf: F64) -> F32:
    """Apply a 4x4 to the xyz columns of ``points``, keeping the rest."""
    xyz = points[:, :3] @ tf[:3, :3].T + tf[:3, 3]
    return np.concatenate([xyz.astype(np.float32), points[:, 3:]], axis=1)


def yaw_of(tf: F64) -> float:
    """Heading (rad) of a 4x4's x axis in its parent's ground plane."""
    return float(np.arctan2(tf[1, 0], tf[0, 0]))


def bev_corners(centre: F64, length: float, width: float, yaw: float) -> F64:
    """The four BEV corners ``(4, 2)`` of a box, counter-clockwise."""
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float) * np.array(
        [length / 2, width / 2]
    )
    rot = np.array([[c, -s], [s, c]])
    out: F64 = half @ rot.T + np.asarray(centre[:2], float)
    return out


@dataclass
class GtBox:
    """One annotated object in the car's LiDAR frame."""

    name: str  # v2b class: car / bicycle / pedestrian / other
    centre: F64  # (3,)
    lwh: F64  # (3,) length, width, height
    yaw: float
    ignore: bool  # parked two-wheeler: neither a hit nor a false alarm
    points: int  # fused LiDAR points inside, from the delivery
    instance: str = ""  # the delivery's instance token (one per object)
    category: str = ""  # the delivery's label before the class map
    interpolated: bool = False  # rebuilt by the densify tool, not labelled

    def corners(self) -> F64:
        return bev_corners(self.centre, self.lwh[0], self.lwh[1], self.yaw)


@dataclass
class LocalFrame:
    """One sample, split into the two agents."""

    index: int
    token: str
    timestamp_us: int
    car: F32  # (N, 4) car LiDAR frame
    bike: F32  # (M, 4) bike ego frame
    car_from_bike: F64  # 4x4: bike frame -> car LiDAR frame
    gt: list[GtBox]


class LocalScene:
    """The samples of one delivery under ``root`` (``v1.0-mini/``,
    ``sensor_data/``), in time order."""

    def __init__(
        self,
        root: Path,
        version: str = "v1.0-mini",
        intensity_scale: float = INTENSITY_SCALE,
        tables: Path | None = None,
        labels: Path | None = None,
    ) -> None:
        """``tables`` points the metadata elsewhere (an annotated copy
        of the delivery, ``4_annotated/.../v1.0-mini``) while the
        sweeps are read under ``root``. ``labels`` points only the
        annotation tables (category, attribute, instance,
        sample_annotation) elsewhere, e.g. a pseudo-label delivery
        whose sensor tables lack the fused sweep; its boxes must name
        the scene's own samples, anything else raises."""
        self.root = Path(root)
        self.intensity_scale = intensity_scale
        meta = Path(tables) if tables else self.root / version
        ann_dir = Path(labels) if labels else meta

        def table(name: str, where: Path = meta) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = json.loads(
                (where / f"{name}.json").read_text()
            )
            return rows

        sensor = {s["token"]: s["channel"] for s in table("sensor")}
        self._calib = {c["token"]: c for c in table("calibrated_sensor")}
        self._pose = {e["token"]: e for e in table("ego_pose")}
        self._channel = {
            t: sensor[c["sensor_token"]] for t, c in self._calib.items()
        }
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        for d in table("sample_data"):
            if not d["is_key_frame"]:
                continue
            ch = self._channel[d["calibrated_sensor_token"]]
            self._data.setdefault(d["sample_token"], {})[ch] = d
        self.samples = sorted(table("sample"), key=lambda s: s["timestamp"])
        category = {c["token"]: c["name"] for c in table("category", ann_dir)}
        self._attr = {
            a["token"]: a["name"] for a in table("attribute", ann_dir)
        }
        self._inst = {
            i["token"]: category[i["category_token"]]
            for i in table("instance", ann_dir)
        }
        self._anns: dict[str, list[dict[str, Any]]] = {}
        known = {s["token"] for s in self.samples}
        for a in table("sample_annotation", ann_dir):
            if a["sample_token"] not in known:
                raise ValueError(
                    f"{ann_dir}: annotation {a['token']} names sample "
                    f"{a['sample_token']}, which is not in {meta}"
                )
            self._anns.setdefault(a["sample_token"], []).append(a)

    def __len__(self) -> int:
        return len(self.samples)

    def labelled_frames(self) -> list[int]:
        """The frame indices that count as labelled
        (:func:`labelled_frames`); a partial delivery has fewer than
        ``len(self)``."""
        labelled = (
            i for i, s in enumerate(self.samples) if self._anns.get(s["token"])
        )
        return labelled_frames(len(self), labelled)

    def _ego(self, d: dict[str, Any]) -> F64:
        e = self._pose[d["ego_pose_token"]]
        return pose_matrix(e["translation"], e["rotation"])

    def _sensor(self, d: dict[str, Any]) -> F64:
        c = self._calib[d["calibrated_sensor_token"]]
        return pose_matrix(c["translation"], c["rotation"])

    def global_from_car_lidar(self, i: int) -> F64:
        d = self._data[self.samples[i]["token"]][LIDAR]
        return self._ego(d) @ self._sensor(d)

    def global_from_car_ego(self, i: int) -> F64:
        """The car's ego pose at the fused sweep: x forward. The fused
        LiDAR's own frame is a half turn off it (``calibrated_sensor``
        says so), which :meth:`global_from_car_lidar` includes."""
        return self._ego(self._data[self.samples[i]["token"]][LIDAR])

    def global_from_bike(self, i: int) -> F64:
        return self._ego(self._data[self.samples[i]["token"]][BIKE_POSE])

    def gt(self, i: int) -> list[GtBox]:
        """The sample's boxes in the car's LiDAR frame."""
        lidar_from_global = np.linalg.inv(self.global_from_car_lidar(i))
        out = []
        for a in self._anns.get(self.samples[i]["token"], []):
            box = lidar_from_global @ pose_matrix(
                a["translation"], a["rotation"]
            )
            label = self._inst[a["instance_token"]]
            attrs = {self._attr[t] for t in a["attribute_tokens"]}
            name = canon(label)
            # the detectors learnt Waymo's cyclist (bike + rider); a
            # parked two-wheeler is not a target, but not a miss either.
            # Only an attributed box can say so: pseudo labels carry no
            # attributes, and their bicycles are riders by construction
            parked = (
                name == "bicycle"
                and bool(attrs)
                and "cycle.with_rider" not in attrs
            )
            w, length, h = a["size"]  # nuScenes order: w, l, h
            out.append(
                GtBox(
                    name=name,
                    centre=box[:3, 3].copy(),
                    lwh=np.array([length, w, h], float),
                    yaw=yaw_of(box),
                    ignore=parked,
                    points=int(a.get("num_lidar_pts", 0)),
                    instance=str(a["instance_token"]),
                    category=str(label),
                    interpolated=bool(a.get("interpolated", False)),
                )
            )
        return out

    def frame(self, i: int) -> LocalFrame:
        s = self.samples[i]
        d = self._data[s["token"]][LIDAR]
        raw = np.fromfile(self.root / d["filename"], np.float32)
        pts = raw.reshape(-1, 5)
        pts = pts[np.isfinite(pts).all(axis=1)]
        cloud = pts[:, :4].copy()
        cloud[:, 3] = np.tanh(
            np.maximum(cloud[:, 3], 0.0) / self.intensity_scale
        )
        car_from_bike = np.linalg.inv(
            self.global_from_car_lidar(i)
        ) @ self.global_from_bike(i)
        bike_in_car = cloud[pts[:, 4] == AGENT_IDS["bike"]]
        return LocalFrame(
            index=i,
            token=s["token"],
            timestamp_us=int(s["timestamp"]),
            car=cloud[pts[:, 4] == AGENT_IDS["car"]],
            bike=transform(bike_in_car, np.linalg.inv(car_from_bike)),
            car_from_bike=car_from_bike,
            gt=self.gt(i),
        )


def box_point_counts(
    boxes: list[GtBox], clouds: dict[str, F32], margin: float = 0.2
) -> dict[str, list[int]]:
    """How many of each agent's points (car frame) fall inside each box:
    ``{agent: [n per box]}``; ``margin`` loosens the height test."""
    out: dict[str, list[int]] = {a: [] for a in clouds}
    for b in boxes:
        c, s = np.cos(-b.yaw), np.sin(-b.yaw)
        rot = np.array([[c, -s], [s, c]])
        for agent, pts in clouds.items():
            d = pts[:, :3] - b.centre
            xy = d[:, :2] @ rot.T
            inside = (
                (np.abs(xy[:, 0]) <= b.lwh[0] / 2)
                & (np.abs(xy[:, 1]) <= b.lwh[1] / 2)
                & (np.abs(d[:, 2]) <= b.lwh[2] / 2 + margin)
            )
            out[agent].append(int(inside.sum()))
    return out


def counts_for(frame: LocalFrame) -> dict[str, list[int]]:
    """Per-box points of the car and of the bike, both in the car frame."""
    return box_point_counts(
        frame.gt,
        {"car": frame.car, "bike": transform(frame.bike, frame.car_from_bike)},
    )


def visible_to(
    counts: dict[str, list[int]],
    agent: str,
    min_points: int,
    single_floor: int = 1,
) -> list[bool]:
    """Which boxes count as this agent's targets: at least ``min_points``
    of its own LiDAR points inside, or, when no other agent has a point
    in the box, at least ``single_floor``."""
    others = [a for a in counts if a != agent]
    own = counts[agent]
    out = []
    for i, n in enumerate(own):
        alone = all(counts[a][i] == 0 for a in others)
        out.append(n >= min_points or (alone and n >= single_floor))
    return out


@dataclass
class AgentSample:
    """One agent's view of one sample, in the agent's own frame."""

    agent: str  # car / bike
    index: int
    token: str
    timestamp_us: int
    points: F32  # (N, 4) the agent's own cloud, its frame
    car_from_agent: F64  # 4x4
    gt: list[GtBox]  # every box, car frame (as delivered)
    own_points: list[int]  # this agent's points per box
    visible: list[bool]  # the agent's targets under the rule


class AgentLoader:
    """The samples of a delivery as one agent sees them: the car with
    its five roof LiDARs in its LiDAR frame, the bike with its two in
    its ego frame. ``min_points`` and ``single_floor`` set the target
    rule of :func:`visible_to`; a delivery without annotations yields
    empty ``gt``."""

    def __init__(
        self,
        scene: LocalScene,
        agent: str,
        min_points: int = 10,
        single_floor: int = 1,
    ) -> None:
        if agent not in AGENT_IDS:
            raise ValueError(f"agent {agent!r}, expected car or bike")
        self.scene = scene
        self.agent = agent
        self.min_points = min_points
        self.single_floor = single_floor

    def __len__(self) -> int:
        return len(self.scene)

    def __getitem__(self, i: int) -> AgentSample:
        fr = self.scene.frame(i)
        counts = counts_for(fr) if fr.gt else {"car": [], "bike": []}
        return AgentSample(
            agent=self.agent,
            index=i,
            token=fr.token,
            timestamp_us=fr.timestamp_us,
            points=fr.car if self.agent == "car" else fr.bike,
            car_from_agent=np.eye(4)
            if self.agent == "car"
            else fr.car_from_bike,
            gt=fr.gt,
            own_points=counts[self.agent],
            visible=visible_to(
                counts, self.agent, self.min_points, self.single_floor
            ),
        )
