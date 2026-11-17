"""Two-agent V2B frames for the cooperative evaluation, from either source.

A :class:`CoopSample` is one moment seen by both agents: the car's cloud
in the car's LiDAR frame, the bike's cloud in the bike's frame, both
world poses (the shared SLAM map), the two sweep times and the labels
in the car frame. Two readers produce it:

``raw``
    a delivery read through :class:`v2b.data.v2b_nusc.LocalScene`: the
    fused sweep split by its agent column, the annotated tables (or a
    pseudo-label table set) for the boxes, and the per-agent
    ``LIDAR_CONC`` sweep times next to the delivery for the time offset
    between the two agents (the fused sweep itself carries one stamp).
``export``
    the two-agent OPV2V export (``<split>/<scene>/<car>/<i>.pcd|yaml``
    and ``<bike>/...``, car first in sorted agent order), read with the
    repo's own PCD and label readers; the labels are the union of both
    agents' yamls keyed by object id, the car's entry winning.

The floor a checkpoint trained on is applied per agent cloud in the
agent's own frame (:func:`v2b.data.ground.fit_surface` + ``rebase``),
and the agent's world pose follows (``rebased_pose``), exactly as the
``-rebase-poisson`` exports were derived; the labels keep the car's
original frame, which differs from the re-based one by a shift along
the sensor's z only, invisible to BEV scoring. Own-point counts per
box are taken on the raw clouds before any floor. A Poisson fit costs
seconds per cloud, so :func:`prepare` caches the floored clouds under
``cache_dir`` and the four settings of one detector share them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import yaml

from v2b.data.opv2v_layout import matrix_from_pose, objects_from_yaml, read_pcd
from v2b.data.v2b_nusc import (
    GtBox,
    LocalScene,
    box_point_counts,
    transform,
    yaw_of,
)

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

AGENTS: tuple[str, str] = ("car", "bike")
FLOORS: tuple[str, ...] = ("raw", "grid", "delaunay", "alpha", "poisson")


@dataclass
class CoopSample:
    """One moment seen by the car and the bike."""

    scene: str
    frame: str
    car: F32  # (N, 4) car LiDAR frame
    bike: F32  # (M, 4) bike frame
    world_from_car: F64
    world_from_bike: F64
    t_car_s: float
    t_bike_s: float
    gt: list[GtBox]  # car LiDAR frame (original, not re-based)
    floor: str = "raw"

    @property
    def car_from_bike(self) -> F64:
        out: F64 = np.linalg.inv(self.world_from_car) @ self.world_from_bike
        return out

    @property
    def dt_ms(self) -> float:
        return abs(self.t_car_s - self.t_bike_s) * 1e3

    @property
    def agent_gap_m(self) -> float:
        return float(np.linalg.norm(self.car_from_bike[:3, 3]))


class Source(Protocol):
    """A sorted sequence of two-agent frames."""

    def __len__(self) -> int: ...

    def key(self, i: int) -> tuple[str, str]: ...

    def __getitem__(self, i: int) -> CoopSample: ...


def own_counts(sample: CoopSample) -> dict[str, I64]:
    """Each agent's own points inside each label, raw clouds, car frame."""
    if not sample.gt:
        return {a: np.zeros(0, np.int64) for a in AGENTS}
    counts = box_point_counts(
        sample.gt,
        {
            "car": sample.car,
            "bike": transform(sample.bike, sample.car_from_bike),
        },
    )
    return {a: np.asarray(counts[a], np.int64) for a in AGENTS}


def gt_arrays(gt: list[GtBox]) -> tuple[F64, list[str], npt.NDArray[np.bool_]]:
    """``(G, 7)`` boxes, names and ignore flags of a label list."""
    if not gt:
        return np.zeros((0, 7)), [], np.zeros(0, bool)
    boxes = np.array(
        [[*b.centre[:3], *b.lwh[:3], b.yaw] for b in gt], float
    ).reshape(-1, 7)
    return boxes, [b.name for b in gt], np.array([b.ignore for b in gt], bool)


# ------------------------------------------------------------ raw delivery
def _sweep_times(nusc_dir: Path) -> F64 | None:
    d = nusc_dir / "sensor_data" / "LIDAR_CONC"
    if not d.is_dir():
        return None
    stamps = sorted(int(p.name.split(".")[0]) for p in d.glob("*.pcd.bin"))
    return np.asarray(stamps, float) * 1e-9 if stamps else None


def _nearest(times: F64 | None, t: float) -> float:
    if times is None or len(times) == 0:
        return t
    return float(times[int(np.argmin(np.abs(times - t)))])


class RawSceneSource:
    """A delivery split into its two agents (see ``v2b_nusc``).

    ``root`` holds the sweeps (``sensor_data/LIDAR_FUSED_FUSED``),
    ``tables`` the annotated metadata, ``labels`` an alternative set
    of annotation tables (pseudo labels). The per-agent sweep times
    come from ``<root>/../car`` and ``<root>/../bike`` when those
    deliveries sit next to the fused one.
    """

    def __init__(
        self,
        root: Path,
        scene: str,
        tables: Path | None = None,
        labels: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.name = scene
        self.scene = LocalScene(self.root, tables=tables, labels=labels)
        self._t_car = _sweep_times(self.root.parent / "car")
        self._t_bike = _sweep_times(self.root.parent / "bike")

    def __len__(self) -> int:
        return len(self.scene)

    def key(self, i: int) -> tuple[str, str]:
        return self.name, f"{i:06d}"

    def __getitem__(self, i: int) -> CoopSample:
        fr = self.scene.frame(i)
        # the deliveries stamp seconds (float), not microseconds
        t = float(self.scene.samples[i]["timestamp"])
        return CoopSample(
            scene=self.name,
            frame=f"{i:06d}",
            car=np.ascontiguousarray(fr.car, np.float32),
            bike=np.ascontiguousarray(fr.bike, np.float32),
            world_from_car=self.scene.global_from_car_lidar(i),
            world_from_bike=self.scene.global_from_bike(i),
            t_car_s=_nearest(self._t_car, t),
            t_bike_s=_nearest(self._t_bike, t),
            gt=fr.gt,
        )


# ------------------------------------------------------------ OPV2V export
def _agent_dirs(scenario: Path) -> list[Path]:
    return sorted(
        p for p in scenario.iterdir() if p.is_dir() and p.name[0] != "."
    )


def _stems(agent: Path) -> set[str]:
    return {
        p.stem
        for p in agent.glob("*.yaml")
        if (agent / f"{p.stem}.pcd").exists()
    }


def _boxes_of(doc: dict[str, Any], world_from_lidar: F64) -> dict[str, GtBox]:
    out: dict[str, GtBox] = {}
    for o in objects_from_yaml(doc, world_from_lidar):
        flag = o.extra.get("ignore", o.extra.get("parked", False))
        out[o.obj_id] = GtBox(
            name=o.cls,
            centre=np.asarray(o.centre, float),
            lwh=np.asarray(o.lwh, float),
            yaw=float(o.yaw),
            ignore=bool(flag),
            points=int(
                str(o.extra.get("points", o.extra.get("num_points", 0)) or 0)
            ),
            instance=o.obj_id,
            category=str(o.extra.get("category", "")),
        )
    return out


class ExportSource:
    """The two-agent OPV2V export of a split: car = first agent in
    sorted order, bike = second, frames = the stems both agents have.

    The export stamps both agents with the frame's time; with
    ``nusc_root`` (``v2b/2_processed_rosbags/nusc``) each
    agent's own sweep time is the nearest ``LIDAR_CONC`` stamp of its
    delivery (``<nusc_root>/<scene>/{car,bike}``), which is what the
    sweep-offset strata need.
    """

    def __init__(self, split_dir: Path, nusc_root: Path | None = None) -> None:
        self.split_dir = Path(split_dir)
        self.nusc_root = Path(nusc_root) if nusc_root else None
        self._times: dict[tuple[str, str], F64 | None] = {}
        self.items: list[tuple[str, Path, Path, str]] = []
        for scenario in sorted(
            p for p in self.split_dir.iterdir() if p.is_dir()
        ):
            agents = _agent_dirs(scenario)
            if len(agents) < 2:
                continue
            car, bike = agents[0], agents[1]
            for stem in sorted(_stems(car) & _stems(bike)):
                self.items.append((scenario.name, car, bike, stem))

    def __len__(self) -> int:
        return len(self.items)

    def key(self, i: int) -> tuple[str, str]:
        scene, _, _, stem = self.items[i]
        return scene, stem

    def _stamp(self, scene: str, agent: str, t: float) -> float:
        if self.nusc_root is None:
            return t
        if (scene, agent) not in self._times:
            name = scene.removeprefix("v2b_")
            self._times[(scene, agent)] = _sweep_times(
                self.nusc_root / name / agent
            )
        return _nearest(self._times[(scene, agent)], t)

    def __getitem__(self, i: int) -> CoopSample:
        scene, car_dir, bike_dir, stem = self.items[i]
        car_doc = yaml.safe_load((car_dir / f"{stem}.yaml").read_text())
        bike_doc = yaml.safe_load((bike_dir / f"{stem}.yaml").read_text())
        world_from_car = matrix_from_pose(car_doc["lidar_pose"])
        world_from_bike = matrix_from_pose(bike_doc["lidar_pose"])
        boxes = _boxes_of(car_doc, world_from_car)
        car_from_bike = np.linalg.inv(world_from_car) @ world_from_bike
        for key, b in _boxes_of(bike_doc, world_from_bike).items():
            if key in boxes:
                continue
            c = car_from_bike @ np.append(b.centre, 1.0)
            b.centre = c[:3]
            b.yaw = b.yaw + yaw_of(car_from_bike)
            boxes[key] = b
        return CoopSample(
            scene=scene,
            frame=stem,
            car=read_pcd(car_dir / f"{stem}.pcd")[:, :4],
            bike=read_pcd(bike_dir / f"{stem}.pcd")[:, :4],
            world_from_car=world_from_car,
            world_from_bike=world_from_bike,
            t_car_s=self._stamp(
                scene, "car", float(car_doc.get("timestamp", 0.0))
            ),
            t_bike_s=self._stamp(
                scene, "bike", float(bike_doc.get("timestamp", 0.0))
            ),
            gt=[boxes[k] for k in sorted(boxes)],
        )


def open_source(spec: dict[str, Any]) -> Source:
    """A source from its picklable description (``kind``, paths)."""
    kind = spec["kind"]
    if kind == "raw":
        return RawSceneSource(
            Path(spec["root"]),
            str(spec.get("scene") or Path(spec["root"]).parent.name),
            Path(spec["tables"]) if spec.get("tables") else None,
            Path(spec["labels"]) if spec.get("labels") else None,
        )
    if kind == "export":
        return ExportSource(
            Path(spec["root"]),
            Path(spec["nusc_root"]) if spec.get("nusc_root") else None,
        )
    raise ValueError(f"source kind {kind!r}, expected raw or export")


# ------------------------------------------------------------ the floor
def floored(sample: CoopSample, method: str) -> CoopSample:
    """The sample with each agent's cloud re-based on its own floor
    and the world poses moved along; ``raw`` returns it unchanged."""
    if method not in FLOORS:
        raise ValueError(f"floor {method!r}, expected one of {FLOORS}")
    if method == "raw":
        return sample
    from v2b.data.ground import fit_surface, rebase, rebased_pose

    f_car = fit_surface(sample.car, method)
    f_bike = fit_surface(sample.bike, method)
    return CoopSample(
        scene=sample.scene,
        frame=sample.frame,
        car=rebase(sample.car, f_car),
        bike=rebase(sample.bike, f_bike),
        world_from_car=rebased_pose(sample.world_from_car, f_car),
        world_from_bike=rebased_pose(sample.world_from_bike, f_bike),
        t_car_s=sample.t_car_s,
        t_bike_s=sample.t_bike_s,
        gt=sample.gt,
        floor=method,
    )


_SOURCES: dict[str, Source] = {}


def _source(spec: dict[str, Any]) -> Source:
    key = repr(sorted(spec.items()))
    if key not in _SOURCES:
        _SOURCES[key] = open_source(spec)
    return _SOURCES[key]


def prepare(
    task: tuple[dict[str, Any], int, str, str | None],
) -> dict[str, Any]:
    """Load one frame, count the own points on the raw clouds and apply
    the floor (from the cache when it has the frame). Runs in a worker
    pool ahead of the GPU; returns plain arrays."""
    spec, i, floor, cache = task
    src = _source(spec)
    scene, frame = src.key(i)
    path = (
        Path(cache) / scene / f"{frame}.npz"
        if cache and floor != "raw"
        else None
    )
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files} | {"scene": scene, "frame": frame}
    raw = src[i]
    counts = own_counts(raw)
    s = floored(raw, floor)
    boxes, names, ignore = gt_arrays(s.gt)
    out: dict[str, Any] = {
        "scene": scene,
        "frame": frame,
        "car": s.car,
        "bike": s.bike,
        "car_from_bike": s.car_from_bike,
        "dt_ms": np.float64(s.dt_ms),
        "agent_gap_m": np.float64(s.agent_gap_m),
        "gt_boxes": boxes,
        "gt_names": np.asarray(names, str),
        "gt_ignore": ignore,
        "count_car": counts["car"],
        "count_bike": counts["bike"],
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{k: v for k, v in out.items() if k not in ("scene", "frame")},
        )
    return out
