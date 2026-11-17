"""V2B (our car + bike recordings) -> two-agent frames in the OPV2V layout.

A V2B scene is a nuScenes-format export under
``v2b/2_processed_rosbags/nusc/<scene>/v2b`` whose one sweep
per sample, ``LIDAR_FUSED_FUSED``, holds both agents' points with an
agent column (``v2b.data.v2b_nusc``). This adapter splits every sample
into two frames of one scenario ``v2b_<scene>``, written the way the
V2X-Real export writes its two vehicles (``v2b.data.export_v2xreal``):

* cav ``0``, the **car**, ego: its five roof LiDARs; HEAL takes the
  first agent folder of a scenario as the ego;
* cav ``1``, the **bike**: its two LiDARs;

each in its **level frame**: origin at the agent's sensor (the fused
LiDAR, the bike's sensor head ``BIKE_CAMERA_FRONT``), x along the
agent's heading, z up along the map's z (:func:`level`), with
``lidar_pose`` in the scene's shared SLAM map frame, so the loader's
pairwise transforms put the bike's cloud on the car's. The car's ego
pose is within two degrees of level anyway (the single-agent scene_11
export kept it); the bike's sensor head is mounted pitched by two to
four degrees and the bike leans by several degrees in turns, which a
BEV grid's z window would feel at range, so the tilt goes into the
pose, not the cloud.

Labels come from a second delivery of the same scene, chosen by
``--labels``: ``annotated`` (the company's boxes, ``4_annotated``) or
``pseudo`` (MS3D pseudo labels, the NAS folder first, then the
v2b-derived one); :func:`find_labels` is the search order. Their
sample tokens must be the scene's own (``LocalScene`` checks). Every
box goes into both agents' yamls (the fusion loader keeps one per id,
``v2b.train.fusion_data``), in the agent's frame, with ``obj_type``
(car / bicycle / pedestrian / other), the original ``category``, the
delivery's fused point count ``points`` and the agent's own points in
the box ``own_points`` (counted in the car's LiDAR frame, the scoring
frame of ``v2b_nusc``; a recount in a tilted agent frame differs by a
few points on a dense box), which is what the target rule (ten own
points, single-agent floor one) is applied to downstream. ``ignore``
marks a parked two-wheeler of the annotated delivery.

No cameras: the fusion and single-agent configs this feeds are
LiDAR-only, as the existing scene_11 export is.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2b.data.opv2v_layout import F64, FrameRecord, ObjectRecord
from v2b.data.v2b_nusc import (
    GtBox,
    LocalFrame,
    LocalScene,
    counts_for,
    pose_matrix,
    transform,
    yaw_of,
)
from v2b.maps.scenes import default_v2b_root, load_scenes

VERSION = "v1.0-mini"
PROCESSED = Path("2_processed_rosbags/nusc")
PREFIX = "v2b"
#: agent -> OPV2V agent folder; the car first, HEAL's ego
CAVS: dict[str, str] = {"car": "0", "bike": "1"}
LABEL_SOURCES = ("annotated", "pseudo")
#: NAS paths relative to the v2b dataset root, in search order
LABEL_DIRS: dict[str, tuple[Path, ...]] = {
    "annotated": (Path("4_annotated/nusc"),),
    "pseudo": (Path("3_pseudo_labels/MS3D/nusc"),),
}
#: the second pseudo-label folder, relative to the NAS root
DERIVED_PSEUDO = Path("derived/pseudo-labels/MS3D/nusc")
#: the densified hand labels (``v2b-labels-densify.py``), relative to
#: the NAS root; searched before the raw delivery so a scene whose
#: parked cars were filled in is exported filled in
DERIVED_HAND = Path("derived/labels-hand/nusc")


def site_of(location_id: str) -> str:
    """The site of a recording: ``gs_a01`` and ``gs_a02`` are both
    ``gs``, the registry's location prefix."""
    return location_id.split("_")[0]


def nas_root(v2b_root: Path) -> Path:
    """The NAS root above ``v2b`` (the root is a symlink)."""
    return Path(os.path.realpath(v2b_root)).parents[1]


def label_dirs(source: str, v2b_root: Path) -> list[Path]:
    """The ``nusc`` parents searched for a scene's label delivery."""
    if source not in LABEL_SOURCES:
        raise ValueError(f"labels {source!r}, expected {LABEL_SOURCES}")
    real = Path(os.path.realpath(v2b_root))
    dirs = [real / rel for rel in LABEL_DIRS[source]]
    if source == "pseudo":
        dirs.append(nas_root(v2b_root) / DERIVED_PSEUDO)
    if source == "annotated":
        dirs.insert(0, nas_root(v2b_root) / DERIVED_HAND)
    return dirs


def find_labels(scene: str, source: str, v2b_root: Path) -> Path | None:
    """The first label delivery of the scene under ``source``: its
    ``v1.0-mini`` directory, or None when no folder has one."""
    for parent in label_dirs(source, v2b_root):
        meta = parent / scene / PREFIX / VERSION
        if all(
            (meta / f"{t}.json").is_file()
            for t in ("sample_annotation", "instance", "category")
        ):
            return meta
    return None


def processed_root(scene: str, v2b_root: Path) -> Path:
    return v2b_root / PROCESSED / scene / PREFIX


def open_scene(scene: str, v2b_root: Path, labels: Path | None) -> LocalScene:
    """The scene's sweeps and poses, with the boxes of ``labels``."""
    return LocalScene(processed_root(scene, v2b_root), labels=labels)


@dataclass(frozen=True)
class ScenePlan:
    """One scene of an export: where its labels come from, or why
    it is exported without boxes / skipped."""

    name: str
    labels: Path | None
    export: bool
    source: str = ""  # annotated / pseudo: which delivery ``labels`` is

    @property
    def note(self) -> str:
        if self.labels is not None:
            return f"{self.source or 'labels'}: {self.labels}"
        return "no labels" + ("" if self.export else ", skipped")


def plan(
    scenes: list[str],
    source: str,
    v2b_root: Path,
    allow_unlabelled: bool = False,
    fallback: str | None = None,
) -> list[ScenePlan]:
    """Which scenes get exported with which labels; a scene without a
    delivery under ``source`` takes the ``fallback`` source's delivery
    when one exists (hand labels first, pseudo labels where the hand
    labels have not landed), and is exported empty only with
    ``allow_unlabelled``."""
    known = load_scenes(v2b_root)
    out = []
    for name in scenes:
        if name not in known:
            raise KeyError(f"scene {name!r} not in the registry")
        kind = source
        labels = find_labels(name, source, v2b_root)
        if labels is None and fallback:
            labels = find_labels(name, fallback, v2b_root)
            kind = fallback
        out.append(
            ScenePlan(
                name,
                labels,
                labels is not None or allow_unlabelled,
                kind if labels is not None else "",
            )
        )
    return out


def level(world_from_x: F64) -> F64:
    """The level frame at a pose: its origin and heading, map z up."""
    c, s = np.cos(yaw_of(world_from_x)), np.sin(yaw_of(world_from_x))
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:3, 3] = world_from_x[:3, 3]
    return out


def agent_poses(scene: LocalScene, i: int) -> dict[str, F64]:
    """``world_from_agent`` of the car and the bike at sample ``i``:
    the car's heading at the fused LiDAR, the bike's at its head."""
    car = scene.global_from_car_ego(i).copy()
    car[:3, 3] = scene.global_from_car_lidar(i)[:3, 3]
    return {"car": level(car), "bike": level(scene.global_from_bike(i))}


def _object(
    box: GtBox,
    agent_from_sensor: F64,
    own: int,
    source: str = "",
    bike_xy: F64 | None = None,
) -> ObjectRecord:
    """The yaml record of one box; ``bike_xy`` (the bike in the car's
    LiDAR frame, where the box is) adds ``dist_car`` / ``dist_bike``,
    the ground-plane distances the loader's radius trim reads."""
    m = agent_from_sensor @ pose_matrix(
        list(box.centre), [np.cos(box.yaw / 2), 0.0, 0.0, np.sin(box.yaw / 2)]
    )
    extra: dict[str, object] = {
        "category": box.category,
        "points": int(box.points),
        "own_points": int(own),
    }
    if bike_xy is not None:
        extra["dist_car"] = round(float(np.hypot(*box.centre[:2])), 2)
        extra["dist_bike"] = round(
            float(np.hypot(*(box.centre[:2] - bike_xy[:2]))), 2
        )
    if box.ignore:
        extra["ignore"] = True
    if source:
        extra["source"] = source
    if box.interpolated:
        extra["interpolated"] = True
    return ObjectRecord(
        obj_id=box.instance,
        cls=box.name,
        centre=m[:3, 3].copy(),
        lwh=box.lwh.copy(),
        yaw=yaw_of(m),
        extra=extra,
    )


def within_radius(fr: LocalFrame, radius: float) -> list[GtBox]:
    """The boxes within ``radius`` of the car and of the bike in the
    ground plane: the contract's dual labelling radius, applied to a
    pseudo-label scene so it teaches the same range as a hand-labelled
    one (and nothing where the hand labels would have nothing)."""
    bike = fr.car_from_bike[:2, 3]
    out = []
    for b in fr.gt:
        d_car = float(np.hypot(b.centre[0], b.centre[1]))
        d_bike = float(np.hypot(b.centre[0] - bike[0], b.centre[1] - bike[1]))
        if max(d_car, d_bike) <= radius:
            out.append(b)
    return out


def agent_frames(
    scene: LocalScene,
    fr: LocalFrame,
    scenario: str,
    labelled: bool,
    speeds: dict[str, float],
    source: str = "",
) -> list[FrameRecord]:
    """The car's and the bike's frame of one sample."""
    i = fr.index
    world_from_sensor = scene.global_from_car_lidar(i)
    clouds = {
        "car": fr.car,
        "bike": transform(fr.bike, fr.car_from_bike),
    }  # both in the car's LiDAR frame, where the boxes are
    counts = counts_for(fr) if labelled and fr.gt else {"car": [], "bike": []}
    bike_xy = fr.car_from_bike[:2, 3]
    stamp = float(scene.samples[i]["timestamp"])
    frames = []
    for agent, world_from_agent in agent_poses(scene, i).items():
        agent_from_sensor = np.linalg.inv(world_from_agent) @ world_from_sensor
        objects = (
            [
                _object(b, agent_from_sensor, own, source, bike_xy)
                for b, own in zip(fr.gt, counts[agent], strict=True)
            ]
            if labelled
            else []
        )
        frames.append(
            FrameRecord(
                scenario=scenario,
                cav=CAVS[agent],
                index=i,
                timestamp_s=stamp,
                points=transform(clouds[agent], agent_from_sensor),
                world_from_lidar=world_from_agent,
                objects=objects,
                speed_mps=speeds.get(agent, 0.0),
            )
        )
    return frames


def iter_scene(
    name: str,
    v2b_root: Path,
    labels: Path | None,
    source: str = "",
    max_range: float | None = None,
) -> Iterator[FrameRecord]:
    """Every sample of one scene as two frames, the car's first;
    ``max_range`` keeps only the boxes within that radius of both
    agents (:func:`within_radius`). A partially labelled delivery
    yields only its labelled frames."""
    scene = open_scene(name, v2b_root, labels)
    scenario = f"{PREFIX}_{name}"
    prev: dict[str, tuple[F64, float]] = {}
    # a partial delivery: the stretches without a box are not empty,
    # they are not labelled yet, so they are left out
    frames = (
        scene.labelled_frames() if labels is not None else range(len(scene))
    )
    for i in frames:
        fr = scene.frame(i)
        if max_range is not None:
            fr.gt = within_radius(fr, max_range)
        stamp = float(scene.samples[i]["timestamp"])
        speeds = {}
        for agent, pose in agent_poses(scene, i).items():
            if agent in prev and stamp > prev[agent][1]:
                d = pose[:3, 3] - prev[agent][0][:3, 3]
                speeds[agent] = float(
                    np.linalg.norm(d[:2]) / (stamp - prev[agent][1])
                )
            prev[agent] = (pose, stamp)
        yield from agent_frames(
            scene, fr, scenario, labels is not None, speeds, source
        )


def iter_frames(
    plans: list[ScenePlan],
    v2b_root: Path | None = None,
    limit: int | None = None,
    shard: tuple[int, int] = (0, 1),
    max_range: float | None = None,
) -> Iterator[FrameRecord]:
    """Every agent-frame of the planned scenes (those marked for
    export), every n-th scene from the i-th for ``shard = (i, n)``."""
    root = v2b_root or default_v2b_root()
    n = 0
    chosen = [p for p in plans if p.export][shard[0] :: shard[1]]
    for p in chosen:
        for frame in iter_scene(
            p.name, root, p.labels, p.source, max_range=max_range
        ):
            yield frame
            n += 1
            if limit is not None and n >= limit:
                return
