"""Build a single OPV2V-format directory out of several datasets.

OpenCOOD's ``OPV2VBaseDataset`` takes exactly one ``root_dir`` and
enumerates the scenario folders directly under it. Training on more
than one dataset therefore needs a *view*: a directory of symlinks, one
per scenario, pointing at the real scenario folders on the NAS.

The prefix is not cosmetic. 31 of V2XSet's 33 training scenarios carry
the same timestamp name as an OPV2V scenario -- V2XSet is OPV2V's
CARLA scenarios re-simulated with an infrastructure agent added -- so
a flat merge would silently drop them. Every link is named
``<dataset>__<scene>``.

``V2XSETBaseDataset`` is a verbatim subclass of ``OPV2VBaseDataset``
for the LiDAR path, so a merged OPV2V + V2XSet view is served correctly
by ``fusion.dataset: 'opv2v'``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = _REPO / "config-global.json"
#: ``V2B_DATA_ROOT`` gives one job its own link tree (the cluster stages data
#: into the node-local ``$TMPDIR``); a shared ``resources/data`` link
#: repointed by a concurrent job would pull the data from under this one.
DATA_ROOT = Path(
    os.environ.get("V2B_DATA_ROOT") or _REPO / "resources" / "data"
)

#: Datasets this module can mix. They share the OPV2V on-disk layout
#: (``<scene>/<agent_id>/<frame>.pcd`` + ``.yaml``) and the same
#: world-frame label convention, which is what makes one loader enough.
MIXABLE: frozenset[str] = frozenset(
    {
        "opv2v",
        "v2xset",
        "opv2v-h",
        # derived data written by v2b/scripts/v2b-export-opv2v.py
        "nuscenes-opv2v",
        "nuscenes-mini-opv2v",
        "argoverse2-opv2v",
        "tumtraf-opv2v",
        "tumtraf-seq-opv2v",
        "tumtraf-veh-opv2v",
        "tumtraf-rsu-opv2v",
        "tumtraf-seq-opv2v-rebase-poisson",
        "tumtraf-veh-opv2v-rebase-poisson",
        "tumtraf-rsu-opv2v-rebase-poisson",
        "tumtraf-rsu-opv2v-rebase-mapfloor",  # one fixed RSU floor
        "waymo-opv2v",
        "v2v4real-opv2v",
        "v2xreal-opv2v",
        "v2xreal-opv2v-rebase-poisson",
        "v2xreal-vc-opv2v",
        "v2xreal-ic-opv2v",
        "v2xreal-vc-opv2v-rebase-poisson",
        "v2xreal-ic-opv2v-rebase-poisson",
        "v2xreal-v2v-opv2v",
        "v2xreal-v2v-opv2v-rebase-poisson",
        # V2V with our classes: car / bicycle / pedestrian (S32)
        "v2xreal-v2v-bike-opv2v",
        "v2xreal-v2v-bike-opv2v-rebase-poisson",
        "v2xreal-v2v-bike-opv2v-rebase-mapfloor",
        "v2xreal-v2v-bike-opv2v-rebase-grid",
        # V2V with the benchmark's classes on the yaw-fixed labels (S40)
        "v2xreal-v2v-truck-opv2v",
        "v2xreal-v2v-truck-opv2v-rebase-poisson",
        "waymo-opv2v-drop",
        "waymo-opv2v-rebase",
        "waymo-opv2v-both",
        "waymo-opv2v-synth",
        "waymo-opv2v-rebase2",
        "waymo-opv2v-synth2",
        "waymo-opv2v-hag",
        "waymo-opv2v-hag2",
        "waymo-opv2v-hag-poisson",
        "waymo-opv2v-rebase-learned",
        "waymo-opv2v-rebase3",
        "waymo-opv2v-rebase3-synth",
        "waymo-opv2v-rebase3-nowalls",
        "waymo-opv2v-rebase-delaunay",
        "waymo-opv2v-rebase-alpha",
        "waymo-opv2v-rebase-poisson",
        "waymo-opv2v-rebase-mapfloor",
        "waymo-opv2v-rebase-poisson-d5",
        "waymo-opv2v-rebase-poisson-d6",
        "waymo-opv2v-rebase-poisson-d7",
        "waymo-opv2v-rebase-poisson-d8",
        "v2b-scene_11-opv2v",
        "v2b-scene_11-opv2v-rebase3",
        "v2b-scene_11-opv2v-rebase-poisson",
        "v2b-scene_11-opv2v-rebase-mapfloor-loo",
        "v2b-scene_11-opv2v-rebase-mapfloor",
        "v2b-coop-opv2v",
        "v2b-coop-opv2v-rebase-poisson",
        "v2b-coop-v2-opv2v",
        "v2b-coop-v2-opv2v-rebase-poisson",
        # hand labels
        "v2b-coop-v2-hand-opv2v",
        "v2b-coop-v2-hand-opv2v-rebase-poisson",
        # one merged floor for both agents, test scenes
        "v2b-coop-v2-hand-opv2v-rebase-poisson-merged",
        # shared floors and densified labels
        "tumtraf-seq-opv2v-rebase-poisson-merged",
        "v2b-coop-v2-dense-opv2v",
        "v2b-coop-v2-dense-opv2v-rebase-poisson-merged",
        "v2b-coop-v2-pseudo-opv2v",
        "v2b-coop-v2-pseudo-opv2v-rebase-poisson",
        "v2b-coop-annotated-opv2v",
        "v2b-coop-annotated-opv2v-rebase-poisson",
        "waymo-opv2v-drop2",
        "waymo-opv2v-floor1",
        # code-space splits with twin renderings (v2b/train/twin_data.py)
        "waymo-opv2v-rebase-poisson-twin",
        "nuscenes-mini-opv2v-twin",
    }
)


@dataclass(frozen=True)
class Source:
    """One dataset split to pull scenarios from.

    ``scenes`` narrows the split to the named scenarios (``scene_14``
    matches the folder ``v2b_scene_14``, as in :func:`is_held_out`);
    empty means every scenario of the split. A split on disk is one
    layout of the shards, so a re-split
    is a view over the existing split directories instead of a second
    export.
    """

    dataset: str
    split: str
    scenes: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        """Prefix used for this source's scenario links."""
        return self.dataset.replace("-", "_")


@dataclass(frozen=True)
class MixtureView:
    """A materialised symlink view plus what went into it."""

    root: Path
    per_source: dict[str, int]

    @property
    def n_scenes(self) -> int:
        return sum(self.per_source.values())


def dataset_root(name: str) -> Path:
    """Absolute root of a registered dataset, via ``resources/data``.

    Goes through the per-machine symlink tree rather than re-deriving a
    NAS root, so there is exactly one place (``dm_link.py`` reading
    ``config-global.json``) that knows where data lives on this host.
    """
    registry = json.loads(CONFIG_PATH.read_text())["datasets"]
    if name not in registry:
        raise KeyError(
            f"{name!r} not in config-global.json "
            f"({', '.join(k for k in registry if not k.startswith('_'))})"
        )
    root = DATA_ROOT / name
    if not root.exists():
        raise FileNotFoundError(
            f"{root} is missing -- run v2b/scripts/dm_link.py"
        )
    return root


def scenario_dirs(source: Source) -> list[Path]:
    """Scenario folders of one source, sorted by name.

    A scenario is any directory directly under ``<root>/<split>``;
    OPV2V keeps a ``data_protocol.yaml`` file beside them, which is
    skipped because it is not a directory.
    """
    if source.dataset not in MIXABLE:
        raise ValueError(
            f"{source.dataset!r} is not OPV2V-format; mixable: "
            f"{', '.join(sorted(MIXABLE))}"
        )
    split_dir = dataset_root(source.dataset) / source.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"no such split: {split_dir}")
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def is_held_out(link_name: str, holdout: float | list[str]) -> bool:
    """Whether a scenario link belongs to the held-out share.

    A share is cut by the hash of the link name, so it is the same
    on every machine and for every run that names the same sources
    (a seed repeat selects its checkpoint on the same scenarios), and
    adding scenarios later moves none of the existing ones. A list
    names the held-out scenarios outright (``scene_14`` matches the
    folder ``v2b_scene_14``), the same for every dataset variant, which
    a hash over the link name is not.
    """
    if isinstance(holdout, list):
        scenario = link_name.split("__", 1)[-1]
        short = scenario.split("_", 1)[-1]
        return scenario in holdout or short in holdout
    digest = hashlib.sha1(link_name.encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return unit < holdout


def select_scenes(dirs: list[Path], scenes: tuple[str, ...]) -> list[Path]:
    """The scenario folders among ``dirs`` that ``scenes`` names.

    Raises when a named scene has no folder, so a typo in a spec fails
    the run at view time instead of silently shrinking the split.
    """
    if not scenes:
        return dirs
    wanted = set(scenes)
    hits = [
        d
        for d in dirs
        if d.name in wanted or d.name.split("_", 1)[-1] in wanted
    ]
    found = {d.name for d in hits} | {d.name.split("_", 1)[-1] for d in hits}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(
            f"scenes not found in the split: {', '.join(missing)} "
            f"(have {', '.join(d.name for d in dirs)})"
        )
    return hits


def build_view(
    sources: list[Source],
    out_dir: Path,
    overwrite: bool = True,
    holdout: float | list[str] | None = None,
    part: str = "train",
) -> MixtureView:
    """Materialise ``out_dir`` as one symlink per scenario.

    With ``holdout`` (a share in (0, 1)) the scenarios are cut in two
    by :func:`is_held_out`, and the view holds the ``part`` asked for:
    ``train`` (the rest) or ``holdout`` (the held-out share, the val
    view that selects the checkpoint).

    Raises if two sources would claim the same link name, which can
    only happen if the same ``(dataset, split)`` is listed twice --
    collisions *across* datasets are already prevented by the tag.
    """
    if part not in ("train", "holdout"):
        raise ValueError(f"part is train or holdout, not {part!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for link in out_dir.iterdir():
            if link.is_symlink():
                link.unlink()

    per_source: dict[str, int] = {}
    seen: dict[str, Path] = {}
    for src in sources:
        scenes = select_scenes(scenario_dirs(src), src.scenes)
        if holdout is not None:
            scenes = [
                s
                for s in scenes
                if is_held_out(f"{src.tag}__{s.name}", holdout)
                == (part == "holdout")
            ]
        for scene in scenes:
            name = f"{src.tag}__{scene.name}"
            if name in seen:
                raise ValueError(
                    f"duplicate scenario link {name!r} from {scene} and "
                    f"{seen[name]}"
                )
            seen[name] = scene
            (out_dir / name).symlink_to(scene, target_is_directory=True)
        key = f"{src.dataset}/{src.split}"
        per_source[key] = per_source.get(key, 0) + len(scenes)
    return MixtureView(root=out_dir, per_source=per_source)


def count_frames(view: MixtureView) -> int:
    """Number of ``.pcd`` frames reachable through the view.

    Counts every agent's frames, so this is sample count per agent, not
    cooperative frame count. Slow on the NAS -- for logging, not for
    anything in the training loop.
    """
    return sum(1 for _ in view.root.glob("*/*/*.pcd"))
