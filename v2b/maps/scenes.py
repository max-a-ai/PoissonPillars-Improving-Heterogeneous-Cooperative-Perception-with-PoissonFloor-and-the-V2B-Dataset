"""Scene registry: resolve V2B scenes to maps, trajectories and frames.

The V2B recordings are indexed by ``scenes_overview.json`` on the NAS,
which maps a scene name (``scene_04``, ``scene_14``, ...) to its
fused SLAM map plus, for each of the two agents (``bike`` and
``car``), the localised TUM trajectory and the nuScenes export.

Paths inside that file are NAS-absolute with a leading slash but are
really relative to the V2B dataset root, so they are joined here rather
than used directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from v2b.geometry.transforms import Trajectory, load_tum

AGENTS = ("bike", "car")


def default_v2b_root() -> Path:
    """Locate the V2B dataset root for this machine.

    Resolves ``resources/data/v2b`` relative to the repo root, which
    ``v2b/scripts/dm_link.py`` materialises from ``config-global.json``.
    """
    repo = Path(__file__).resolve().parents[2]
    return repo / "resources" / "data" / "v2b"


@dataclass(frozen=True)
class AgentPaths:
    """Per-agent artefacts for one scene."""

    agent: str
    tum: Path
    nusc_dir: Path
    dlio_map: Path
    dufo_map: Path

    def load_trajectory(self) -> Trajectory:
        """Read this agent's localised trajectory from its TUM file."""
        if not self.tum.exists():
            raise FileNotFoundError(f"missing trajectory: {self.tum}")
        return load_tum(self.tum, name=self.agent)


@dataclass(frozen=True)
class Scene:
    """One V2B recording scene."""

    name: str
    location_id: str
    fused_map: Path
    trajectory_png: Path
    own_map: bool
    agents: dict[str, AgentPaths]
    #: label deliveries of the merged (v2b) export, ``hand`` and
    #: ``pseudo``: the delivery root holding ``v1.0-mini/``, or None
    #: when the registry lists none
    labels: dict[str, Path | None] = field(default_factory=dict)

    def agent(self, which: str) -> AgentPaths:
        """Return the paths for ``bike`` or ``car``."""
        if which not in self.agents:
            raise KeyError(
                f"scene {self.name!r} has no agent {which!r}; "
                f"have {sorted(self.agents)}"
            )
        return self.agents[which]

    def has_trajectories(self) -> bool:
        """True when every agent's TUM file is present on disk."""
        return all(a.tum.exists() for a in self.agents.values())


def _join(root: Path, rel: str | None) -> Path:
    """Join a NAS-absolute-looking path from the registry onto ``root``."""
    return root / (rel or "").lstrip("/")


def load_scenes(root: Path | None = None) -> dict[str, Scene]:
    """Parse ``scenes_overview.json`` into :class:`Scene` objects."""
    base = root or default_v2b_root()
    registry = base / "scenes_overview.json"
    if not registry.exists():
        raise FileNotFoundError(
            f"scene registry not found: {registry}. Run "
            "`python3 v2b/scripts/dm_link.py` to build resources/."
        )
    raw = json.loads(registry.read_text())["scenes"]

    scenes: dict[str, Scene] = {}
    for name, entry in raw.items():
        agents: dict[str, AgentPaths] = {}
        for which in AGENTS:
            block = entry.get(which)
            if not block:
                continue
            agents[which] = AgentPaths(
                agent=which,
                tum=_join(base, block.get("to_nusc", {}).get("tum")),
                nusc_dir=_join(base, block.get("nusc")),
                dlio_map=_join(base, block.get("map", {}).get("dlio_map")),
                dufo_map=_join(base, block.get("map", {}).get("dufo_map")),
            )
        label_block = entry.get("v2b", {}).get("labels", {})
        labels = {
            kind: _join(base, rel) if rel else None
            for kind, rel in label_block.items()
        }
        scenes[name] = Scene(
            name=name,
            location_id=entry.get("location_id", ""),
            fused_map=_join(base, entry.get("map")),
            trajectory_png=_join(base, entry.get("trajectory")),
            own_map=bool(entry.get("own_map", False)),
            agents=agents,
            labels=labels,
        )
    return scenes
