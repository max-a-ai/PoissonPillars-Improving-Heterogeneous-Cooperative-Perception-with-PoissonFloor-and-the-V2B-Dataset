"""Render an OpenCOOD ``hypes`` dict from a v2b experiment spec.

An experiment spec (``configs/*.yaml``) says which published OpenCOOD
config to start from, which dataset splits to mix, what BEV grid to use
and how long to train. This module turns that into the full hypes dict
OpenCOOD's ``build_dataset`` / ``create_model`` expect.

**Why retargeting needs its own function.** The published configs use
YAML anchors -- ``&cav_lidar`` is referenced from six places. Anchors
are resolved at parse time, so after ``yaml.safe_load`` those six
places hold six independent copies. Changing only one leaves the rest
pointing at the old range, which is exactly how CoopDet3D's
``tumtraf_v2x/det/transfusion/default.yaml`` ended up with a nuScenes
``grid_size`` against a 150 m range and silently cropped its scenes.
:func:`retarget_grid` rewrites every copy, and :func:`stale_grid_keys`
fails the run if any survived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from v2b.data.mixture import Source

_REPO = Path(__file__).resolve().parents[2]
#: The OpenCOOD forks an experiment can run on, by the ``fork:`` key.
FORKS: dict[str, str] = {"codefilling": "CodeFilling", "heal": "HEAL"}
HYPES_ROOT = _REPO / "third_party" / "CodeFilling" / "opencood" / "hypes_yaml"


def fork_dir(fork: str) -> Path:
    """``third_party/<Fork>`` for a ``fork:`` key."""
    if fork not in FORKS:
        raise KeyError(f"fork {fork!r} not in {sorted(FORKS)}")
    return _REPO / "third_party" / FORKS[fork]


def hypes_root(fork: str) -> Path:
    """Where the fork keeps its published configs."""
    return fork_dir(fork) / "opencood" / "hypes_yaml"


#: Keys whose value is a 6-element point-cloud range and must all agree.
RANGE_KEYS: tuple[str, ...] = (
    "cav_lidar_range",
    "gt_range",
    "lidar_range",
)
#: Keys whose value is a 3-element voxel size and must all agree.
VOXEL_KEYS: tuple[str, ...] = ("voxel_size",)
#: HEAL camera grids: ``[lower, upper, cell]`` per axis, keyed by axis.
BOUND_KEYS: dict[str, tuple[int, int, int]] = {
    "xbound": (0, 3, 0),
    "ybound": (1, 4, 1),
}


def bound_for(
    key: str, point_cloud_range: list[float], voxel: list[float]
) -> list[float]:
    """The ``[lower, upper, cell]`` a camera grid key must hold."""
    lo, hi, ax = BOUND_KEYS[key]
    return [point_cloud_range[lo], point_cloud_range[hi], voxel[ax]]


#: keys a spec's ``codespace`` block may carry
CODESPACE_KEYS = frozenset(
    {
        "codes",
        "dim",
        "twin",
        "freeze",
        "decay",
        "beta",
        "vq_weight",
        "recon_weight",
        "sim_weight",
        "pair_tau",
    }
)


def _source(entry: dict[str, Any]) -> Source:
    """One ``sources`` entry of a spec: dataset, split, optional scenes."""
    scenes = entry.get("scenes") or []
    if not isinstance(scenes, list) or not all(
        isinstance(x, str) for x in scenes
    ):
        raise ValueError("a source's scenes is a list of scene names")
    return Source(str(entry["dataset"]), str(entry["split"]), tuple(scenes))


@dataclass(frozen=True)
class ExperimentSpec:
    """One v2b training experiment.

    Attributes
    ----------
    name:
        Run name. Also the ``outputs/<name>/`` directory and the wandb
        run name -- one string, three places.
    base:
        Path under the fork's ``opencood/hypes_yaml`` of the published
        config to start from.
    fork:
        Which OpenCOOD fork runs the experiment, ``codefilling`` (the
        default) or ``heal``; picks the venv, the config root and the
        HEAL-only data patches.
    classes:
        Which ``obj_type`` boxes of derived data count as targets, e.g.
        ``[car]``; ``None`` keeps every box (OPV2V's own files have no
        class and are always kept).
    anchors:
        Per-class anchor sizes ``{class: [l, w, h]}`` in metres, with
        an optional fourth entry, the anchor's centre height (HEAL's
        -1.0 m when absent: on a floor at -1.9 m that is a standing
        person or a car; a 3 m truck's centre sits at -0.45 m). Turns
        HEAL's single-class head into a class-major anchor set (one
        anchor per class and yaw, class = anchor index // yaws); see
        :func:`apply_anchor_classes`. Implies ``classes``.
    max_range:
        Metres; boxes farther than this from either agent are dropped
        when a label yaml is loaded (``dist_car`` / ``dist_bike`` of the
        V2B export). The hand labels end at 50 m of both agents, so a
        pseudo-labelled scene trimmed the same way teaches the same
        range; ``None`` keeps every box.
    train_sources, val_sources, test_sources:
        :class:`~v2b.data.mixture.Source` entries mixed into the train /
        val / test view; ``scenes`` on an entry narrows it to the named
        scenarios (the split-v3 views of V2B).
    point_cloud_range, voxel_size:
        The BEV grid, applied to every copy in the base config.
    epochs, batch_size, lr:
        Schedule overrides; ``None`` keeps the base config's value.
    init_from:
        Optional checkpoint directory to warm-start from.
    """

    name: str
    base: str
    train_sources: list[Source]
    val_sources: list[Source]
    point_cloud_range: list[float]
    voxel_size: list[float]
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    eval_freq: int | None = None
    # a random share of the train frames (0 < subset < 1), fixed seed
    subset: float | None = None
    # keep every epoch's weights (net_epoch<k>.pth) next to bestval and
    # latest, so a checkpoint can be re-picked on another split later
    keep_epochs: bool = False
    # held-out checkpoint selection (the paper's seed rows): this share
    # of the train SCENARIOS (fixed hash order) becomes the val view that
    # picks the best checkpoint, and ``sources.test`` is what
    # ``--eval-only`` scores; ``sources.val`` is then left out
    holdout: float | list[str] | None = None  # a share, or scene names
    test_sources: list[Source] = field(default_factory=list)
    init_from: str | None = None
    notes: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    fork: str = "codefilling"
    classes: list[str] | None = None
    max_range: float | None = None
    anchors: dict[str, list[float]] | None = None
    # per-class anchor matching thresholds {class: [pos_iou, neg_iou]};
    # classes left out keep HEAL's global pair (0.6 / 0.45 for cars)
    class_thresholds: dict[str, list[float]] | None = None
    # float channels per LiDAR point; 5 reads the ``hag`` (height above
    # ground) field of a split derived with ``--mode hag``
    point_channels: int = 4
    # the pillar net's pooling: max (HEAL), meanmax, attn or hist
    pillar_pooling: str = "max"
    # SECOND's voxel feature encoder (m3): mean (HEAL) or attn, the
    # voxel-level twin of pillar_pooling attn; see install_voxel_pooling
    voxel_pooling: str = "mean"
    # floor-consistency term on confident anchors: {floor_z, tau,
    # weight, classes}; see v2b/train/grounding.py
    grounding_loss: dict[str, Any] | None = None
    # self-supervised masked pretraining of the m1 trunk, no labels:
    # {mask_ratio, bins}; see v2b/train/pretrain.py
    pretrain: dict[str, Any] | None = None
    # a VQ code space between trunk and heads, trained with the twin
    # observations of a ``--twin-sweep`` split: {codes, dim, twin,
    # freeze, vq_weight, recon_weight, sim_weight}; see
    # v2b/train/codespace.py
    codespace: dict[str, Any] | None = None
    # evaluation-time box filter on the returns inside a box:
    # {intensity_min, classes, z_min, margin}; see v2b/train/box_filter.py
    box_filter: dict[str, Any] | None = None
    # fusion rows: 'single' puts our single-agent trunk per agent inside
    # HEAL's intermediate fusion model; see v2b/train/fusion_trunk.py
    fusion_trunk: str | None = None
    # fusion rows (v2b/train/fusion_data.py): one augmentation per
    # sample over the fused frame; every agent scored on the union of
    # the agents' labels; the warm-started tensors held fixed
    fused_augment: bool = False
    union_labels: bool = False
    # eval only: the single-agent model on every vehicle, boxes merged
    # in the ego frame (HEAL's late test collate); see
    # v2b/train/opencood_train.py bev_ap
    late_fusion: bool = False
    freeze_trunk: bool = False
    bev_attention: dict[str, Any] = field(default_factory=dict)
    # LitePT as the m1 encoder (v2b/models/litept_encoder.py): grid,
    # weights, model overrides
    litept: dict[str, Any] | None = None
    # a 3D voxel encoder (HEAL's SECOND, modality m3) voxelises finer
    # than the BEV grid: its own [dx, dy, dz]; the BEV grid above stays
    # the anchor and post-processing grid
    encoder_voxel: list[float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        """Read a ``configs/<name>.yaml`` experiment spec."""
        raw = yaml.safe_load(Path(path).read_text())
        grid = raw["grid"]
        sched = raw.get("train", {})
        sources = raw["sources"]
        holdout = sched.get("holdout")
        if holdout is not None:
            if isinstance(holdout, list):
                if not holdout or not all(isinstance(h, str) for h in holdout):
                    raise ValueError("train.holdout names scenes as strings")
            elif not 0 < float(holdout) < 1:
                raise ValueError("train.holdout is a share in (0, 1)")
            if sources.get("val"):
                raise ValueError(
                    "train.holdout makes the val view; drop sources.val"
                )
            if not sources.get("test"):
                raise ValueError("train.holdout needs sources.test")
        elif not sources.get("val"):
            raise ValueError("sources.val is required without holdout")
        return cls(
            name=raw["name"],
            base=raw["base"],
            train_sources=[_source(s) for s in sources["train"]],
            val_sources=[_source(s) for s in sources.get("val") or []],
            holdout=(
                [str(h) for h in holdout]
                if isinstance(holdout, list)
                else (float(holdout) if holdout is not None else None)
            ),
            test_sources=[_source(s) for s in sources.get("test") or []],
            point_cloud_range=[float(v) for v in grid["range"]],
            voxel_size=[float(v) for v in grid["voxel"]],
            epochs=sched.get("epochs"),
            batch_size=sched.get("batch_size"),
            lr=sched.get("lr"),
            eval_freq=sched.get("eval_freq"),
            subset=sched.get("subset"),
            keep_epochs=bool(sched.get("keep_epochs", False)),
            init_from=raw.get("init_from"),
            class_thresholds=(
                {
                    str(k): [float(x) for x in v]
                    for k, v in raw["class_thresholds"].items()
                }
                if raw.get("class_thresholds")
                else None
            ),
            point_channels=int(raw.get("point_channels", 4)),
            pillar_pooling=str(raw.get("pillar_pooling", "max")),
            voxel_pooling=str(raw.get("voxel_pooling", "mean")),
            grounding_loss=(
                dict(raw["grounding_loss"])
                if raw.get("grounding_loss")
                else None
            ),
            pretrain=(dict(raw["pretrain"]) if raw.get("pretrain") else None),
            codespace=(
                dict(raw["codespace"]) if raw.get("codespace") else None
            ),
            box_filter=(
                dict(raw["box_filter"]) if raw.get("box_filter") else None
            ),
            fusion_trunk=(
                str(raw["fusion_trunk"]) if raw.get("fusion_trunk") else None
            ),
            fused_augment=bool(raw.get("fused_augment", False)),
            union_labels=bool(raw.get("union_labels", False)),
            late_fusion=bool(raw.get("late_fusion", False)),
            freeze_trunk=bool(raw.get("freeze_trunk", False)),
            bev_attention=dict(raw.get("bev_attention") or {}),
            litept=(dict(raw["litept"]) if raw.get("litept") else None),
            encoder_voxel=(
                [float(v) for v in raw["encoder_voxel"]]
                if raw.get("encoder_voxel")
                else None
            ),
            notes=raw.get("notes", ""),
            overrides=raw.get("overrides", {}),
            fork=str(raw.get("fork", "codefilling")),
            classes=list(raw["classes"]) if raw.get("classes") else None,
            max_range=(
                float(raw["max_range"]) if raw.get("max_range") else None
            ),
            anchors=(
                {
                    str(k): [float(x) for x in v]
                    for k, v in raw["anchors"].items()
                }
                if raw.get("anchors")
                else None
            ),
        )

    @property
    def grid_cells(self) -> tuple[int, int]:
        """``(nx, ny)`` pillar counts implied by range and voxel."""
        r, v = self.point_cloud_range, self.voxel_size
        nx = (r[3] - r[0]) / v[0]
        ny = (r[4] - r[1]) / v[1]
        for n, axis in ((nx, "x"), (ny, "y")):
            if abs(n - round(n)) > 1e-6:
                raise ValueError(
                    f"{axis} extent is not an integer number of cells "
                    f"({n:.4f}) -- range and voxel disagree"
                )
        return int(round(nx)), int(round(ny))


class _BaseLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the Python tags of a dumped config.

    Some published configs are ``yaml.dump`` output of a parsed run:
    an ``OrderedDict`` becomes a plain dict, numpy objects (the parsed
    ``grid_size``, recomputed by the fork's yaml parser anyway) become
    ``None``.
    """


def _python_apply(loader: yaml.SafeLoader, suffix: str, node: Any) -> Any:
    if suffix == "collections.OrderedDict":
        seq = loader.construct_sequence(node, deep=True)
        pairs = seq[0] if seq and isinstance(seq[0], list) else []
        return {k: v for k, v in pairs}
    return None


_BaseLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _python_apply
)
_BaseLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


#: Published configs that do not parse as shipped, with the text repair
#: applied before the load (the forks live in the gitignored
#: ``third_party/``, so the fix has to travel with this loader). HEAL's
#: CoAlign config indents ``feat_dim`` under ``fusion_method: att``,
#: a YAML error; ``heter_model_baseline_ms`` reads it as
#: ``args['att']['feat_dim']``, which is what the repair restores.
BASE_REPAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "opv2v/LiDAROnly/lidar_coalign.yaml": (
        (
            "    fusion_method: att\n      feat_dim: [64, 128, 256]\n",
            "    fusion_method: att\n    att:\n"
            "      feat_dim: [64, 128, 256]\n",
        ),
    ),
}


def repair_base(base: str, text: str) -> str:
    """Apply the known text repairs of one published config.

    Every repair must match exactly once: an upstream edit that changes
    the broken lines is reported, not silently loaded half-fixed.
    """
    for old, new in BASE_REPAIRS.get(base, ()):
        if text.count(old) != 1:
            raise ValueError(
                f"{base}: the repair for {old.split()[0]!r} no longer "
                "matches the published config"
            )
        text = text.replace(old, new)
    return text


def load_base(base: str, fork: str = "codefilling") -> dict[str, Any]:
    """Read a published OpenCOOD config by fork-relative path."""
    path = hypes_root(fork) / base
    if not path.is_file():
        raise FileNotFoundError(f"no base config at {path}")
    text = repair_base(base, path.read_text())
    loaded: dict[str, Any] = yaml.load(text, Loader=_BaseLoader)
    return loaded


def _walk(node: Any, fn: Any) -> None:
    """Apply ``fn(container, key)`` to every mapping entry in a tree."""
    if isinstance(node, dict):
        for key in list(node):
            fn(node, key)
            _walk(node[key], fn)
    elif isinstance(node, list):
        for item in node:
            _walk(item, fn)


def retarget_grid(
    hypes: dict[str, Any], point_cloud_range: list[float], voxel: list[float]
) -> dict[str, Any]:
    """Rewrite every range / voxel copy in the tree, in place.

    Also drops the pre-baked ``anchor_args`` ``H``/``W``/``D``: the
    ``load_point_pillar_params`` parser recomputes them from the range
    and voxel, and a leftover literal would win over the new grid.
    """

    def _set(container: dict[str, Any], key: str) -> None:
        value = container[key]
        if key in RANGE_KEYS and isinstance(value, list) and len(value) == 6:
            container[key] = list(point_cloud_range)
        elif key in VOXEL_KEYS and isinstance(value, list) and len(value) == 3:
            container[key] = list(voxel)
        elif key in BOUND_KEYS and isinstance(value, list) and len(value) == 3:
            container[key] = bound_for(key, point_cloud_range, voxel)

    _walk(hypes, _set)

    anchors = hypes.get("postprocess", {}).get("anchor_args", {})
    for stale in ("H", "W", "D", "vh", "vw", "vd"):
        anchors.pop(stale, None)
    return hypes


def stale_grid_keys(
    hypes: dict[str, Any], point_cloud_range: list[float], voxel: list[float]
) -> list[str]:
    """Names of range / voxel keys that do *not* match the target.

    A non-empty result means the retarget missed a copy and the run
    would train one component against a different grid than another.
    """
    bad: list[str] = []
    # the ego encoder's own finer grid (apply_encoder_voxel) is not stale
    encoder = hypes.get("v2b_encoder_voxel")
    fine = [float(v) for v in encoder] if encoder else None
    # a range or voxel copy the spec pins through ``overrides`` (e.g.
    # 0.8 m cells keep gt_range at the benchmark's +-40 m inside a
    # +-41.6 m grid) is deliberate, not a missed copy
    pinned = set(hypes.get("v2b_pinned_grid_keys") or [])

    def _check(container: dict[str, Any], key: str) -> None:
        value = container[key]
        if not isinstance(value, list) or key in pinned:
            return
        want: list[float] | None = None
        if key in RANGE_KEYS and len(value) == 6:
            want = point_cloud_range
        elif key in VOXEL_KEYS and len(value) == 3:
            if fine is not None and [float(v) for v in value] == fine:
                return
            want = voxel
        elif key in BOUND_KEYS and len(value) == 3:
            want = bound_for(key, point_cloud_range, voxel)
        if want is not None and [float(v) for v in value] != want:
            bad.append(f"{key}={value}")

    _walk(hypes, _check)
    return bad


def _apply_overrides(hypes: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply dotted-path overrides, e.g. ``model.args.supervise_single``."""
    for dotted, value in overrides.items():
        node: Any = hypes
        parts = dotted.split(".")
        for part in parts[:-1]:
            # a block the base config lacks (``model.args.compressor``)
            # is created on the way down
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def apply_encoder_voxel(
    hypes: dict[str, Any], voxel: list[float], point_cloud_range: list[float]
) -> None:
    """Give the ego encoder its own, finer voxel grid.

    HEAL's SECOND base voxelises at 0.1 m and compresses the height
    axis into the BEV channels, so the encoder's ``voxel_size`` must
    stay apart from the BEV grid that :func:`retarget_grid` writes
    everywhere. The two copies that belong to the encoder (the
    modality's preprocess and its ``encoder_args``) are set here, after
    the retarget and its stale check; the range must be a whole number
    of encoder voxels on every axis.
    """
    if len(voxel) != 3:
        raise ValueError("encoder_voxel is [dx, dy, dz]")
    for i, axis in enumerate("xyz"):
        n = (point_cloud_range[i + 3] - point_cloud_range[i]) / voxel[i]
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"{axis} extent is not a whole number of encoder voxels "
                f"({n:.4f})"
            )
    ego = str(hypes["heter"]["ego_modality"]).split("&")[0]
    pre = hypes["heter"]["modality_setting"][ego]["preprocess"]["args"]
    pre["voxel_size"] = list(voxel)
    hypes["model"]["args"][ego]["encoder_args"]["voxel_size"] = list(voxel)
    hypes["v2b_encoder_voxel"] = list(voxel)


def render(
    spec: ExperimentSpec, train_dir: Path, val_dir: Path
) -> dict[str, Any]:
    """Full hypes dict for one experiment, ready for ``yaml.safe_dump``.

    ``test_dir`` is pointed at the validation view: this repo evaluates
    through the training loop, and a dangling ``test_dir`` from the
    published config would break OpenCOOD's post-training inference.
    """
    hypes = load_base(spec.base, spec.fork)
    hypes["name"] = spec.name
    hypes["root_dir"] = str(train_dir)
    hypes["validate_dir"] = str(val_dir)
    hypes["test_dir"] = str(val_dir)

    retarget_grid(hypes, spec.point_cloud_range, spec.voxel_size)
    stale = stale_grid_keys(hypes, spec.point_cloud_range, spec.voxel_size)
    if stale:
        raise ValueError(f"grid retarget missed: {', '.join(stale)}")
    if spec.encoder_voxel is not None:
        if spec.fork != "heal":
            raise ValueError("encoder_voxel is a HEAL-fork option")
        apply_encoder_voxel(hypes, spec.encoder_voxel, spec.point_cloud_range)

    params = hypes.setdefault("train_params", {})
    if spec.epochs is not None:
        params["epoches"] = spec.epochs
    if spec.batch_size is not None:
        params["batch_size"] = spec.batch_size
    if spec.eval_freq is not None:
        params["eval_freq"] = spec.eval_freq
    if spec.subset is not None:
        if not 0 < spec.subset <= 1:
            raise ValueError("train.subset is a share in (0, 1]")
        params["subset"] = float(spec.subset)
    if spec.keep_epochs:
        params["keep_epochs"] = True
    if spec.lr is not None:
        hypes.setdefault("optimizer", {})["lr"] = spec.lr

    # The merged view is served by the OPV2V loader regardless of which
    # datasets went into it -- V2XSETBaseDataset is a verbatim subclass
    # for the LiDAR path (see v2b.data.mixture).
    hypes.setdefault("fusion", {})["dataset"] = "opv2v"

    if spec.anchors:
        apply_anchor_classes(hypes, spec.anchors)
    if spec.class_thresholds:
        if not spec.anchors:
            raise ValueError("class_thresholds needs anchors: per class")
        unknown = set(spec.class_thresholds) - set(spec.anchors)
        if unknown:
            raise ValueError(f"class_thresholds for unknown classes {unknown}")
        hypes["v2b_class_thresholds"] = spec.class_thresholds
    if spec.point_channels != 4:
        if spec.fork != "heal":
            raise ValueError("point_channels is a HEAL-fork option")
        # read by the trainer: HEAL's reader, voxeliser and pillar net
        hypes["v2b_point_channels"] = spec.point_channels
    if spec.pillar_pooling != "max":
        if spec.fork != "heal":
            raise ValueError("pillar_pooling is a HEAL-fork option")
        hypes["v2b_pillar_pooling"] = spec.pillar_pooling
    if spec.voxel_pooling != "mean":
        if spec.fork != "heal":
            raise ValueError("voxel_pooling is a HEAL-fork option")
        hypes["v2b_voxel_pooling"] = spec.voxel_pooling
    if spec.grounding_loss:
        if not spec.anchors:
            raise ValueError("grounding_loss needs anchors: per class")
        hypes["v2b_grounding_loss"] = spec.grounding_loss
    if spec.pretrain:
        if spec.fork != "heal":
            raise ValueError("pretrain is a HEAL-fork option")
        hypes["v2b_pretrain"] = spec.pretrain
    if spec.codespace:
        if spec.fork != "heal":
            raise ValueError("codespace is a HEAL-fork option")
        if spec.pretrain:
            raise ValueError("codespace and pretrain exclude each other")
        unknown = set(spec.codespace) - CODESPACE_KEYS
        if unknown:
            raise ValueError(f"codespace: unknown keys {sorted(unknown)}")
        hypes["v2b_codespace"] = spec.codespace
        if spec.codespace.get("twin"):
            # the twin cloud is voxelised unaugmented; the ego cloud has
            # to match it cell for cell, so no flips or rotations
            hypes["data_augment"] = []
    if spec.box_filter:
        hypes["v2b_box_filter"] = spec.box_filter
    if spec.fusion_trunk:
        if spec.fork != "heal" or spec.fusion_trunk != "single":
            raise ValueError("fusion_trunk: single is a HEAL-fork option")
        hypes["v2b_fusion_trunk"] = spec.fusion_trunk
    if spec.bev_attention:
        if spec.fork != "heal":
            raise ValueError("bev_attention is a HEAL-fork option")
        hypes["v2b_bev_attention"] = spec.bev_attention
    if spec.litept is not None:
        if spec.fork != "heal":
            raise ValueError("litept is a HEAL-fork option")
        if spec.pillar_pooling != "max" or spec.point_channels != 4:
            raise ValueError(
                "litept replaces the pillar net: no pooling "
                "or point-channel options with it"
            )
        m1 = hypes["model"]["args"]["m1"]
        m1["core_method"] = "lite_pt_encoder"
        m1["encoder_args"]["litept"] = spec.litept
        hypes["v2b_litept"] = spec.litept
    if spec.fused_augment:
        if spec.fork != "heal":
            raise ValueError("fused_augment is a HEAL-fork option")
        if hypes.get("fusion", {}).get("core_method") != "intermediateheter":
            raise ValueError("fused_augment needs an intermediate fusion base")
        # the clouds meet in the ego frame before the pillars, so one
        # transform covers every agent and the fusion warps stay identity
        hypes["fusion"].setdefault("args", {})["proj_first"] = True
        hypes["v2b_fused_augment"] = True
    if spec.union_labels:
        if spec.fork != "heal":
            raise ValueError("union_labels is a HEAL-fork option")
        hypes["v2b_union_labels"] = True
    if spec.late_fusion:
        if spec.fork != "heal":
            raise ValueError("late_fusion is a HEAL-fork option")
        if hypes.get("fusion", {}).get("core_method") != "lateheter":
            raise ValueError(
                "late_fusion needs a late (single-agent) fusion base"
            )
        hypes["v2b_late_fusion"] = True
    if spec.freeze_trunk:
        if not spec.init_from:
            raise ValueError("freeze_trunk holds a warm start: init_from")
        hypes["v2b_freeze_trunk"] = True
    classes = spec.classes or (list(spec.anchors) if spec.anchors else None)
    if classes:
        # read by the trainer: HEAL's single-class loader keeps only these
        hypes["v2b_classes"] = list(classes)
    if spec.max_range:
        # read by the trainer: the loader drops boxes beyond the dual radius
        hypes["v2b_max_range"] = float(spec.max_range)
    _apply_overrides(hypes, spec.overrides)
    pinned = sorted(
        {
            k.rsplit(".", 1)[-1]
            for k in spec.overrides
            if k.rsplit(".", 1)[-1] in RANGE_KEYS + VOXEL_KEYS
        }
    )
    if pinned:
        hypes["v2b_pinned_grid_keys"] = pinned
    return hypes


def repair_preprocess_cfg(hypes: dict[str, Any]) -> dict[str, Any]:
    """Undo the degenerate voxel limits shipped inside HEAL's zip.

    The released ``config.yaml`` carries ``max_points_per_voxel: 1`` and
    ``max_voxel_train/test: 1``, which yields exactly *one* pillar for a
    200k-point frame and an empty BEV map. It is left over from a
    throughput-profiling run -- the sibling ``config_train_perf.txt``
    records a third, different range -- not the setting the weights were
    trained with. The repo's own configs use 32 and 70000, so those are
    restored. Range and voxel size are left alone: those must match the
    trained weights.
    """
    args = hypes.get("preprocess", {}).get("args", {})
    if args.get("max_points_per_voxel", 0) <= 1:
        args["max_points_per_voxel"] = 32
    for key in ("max_voxel_train", "max_voxel_test"):
        if args.get(key, 0) <= 1:
            args[key] = 70000
    return hypes


#: HEAL's anchor centre height (``generate_anchor_box``), every class
ANCHOR_Z = -1.0


def apply_anchor_classes(
    hypes: dict[str, Any], anchors: dict[str, list[float]]
) -> None:
    """Rewrite the anchor config for one anchor set per class.

    ``anchor_args.l/w/h`` become lists (one entry per class, in the
    order given), ``anchor_args.z`` the centre heights (the optional
    fourth entry, else HEAL's -1.0), ``num`` becomes classes x yaws, and every
    ``anchor_number`` in the tree (the heads' channel count) follows.
    ``dir_args.anchor_yaw`` stays the yaw list: HEAL's direction loss
    tiles it over the anchor axis, which matches the class-major,
    yaw-minor anchor order, and its reshapes assume
    ``len(anchor_yaw) == num_bins``. ``v2b_anchor_classes`` records the
    class order; the trainer installs the postprocessor patch when set.
    """
    args = hypes["postprocess"]["anchor_args"]
    yaws = [float(r) for r in args["r"]]
    names = list(anchors)
    args["l"] = [float(anchors[c][0]) for c in names]
    args["w"] = [float(anchors[c][1]) for c in names]
    args["h"] = [float(anchors[c][2]) for c in names]
    args["z"] = [
        float(anchors[c][3]) if len(anchors[c]) > 3 else ANCHOR_Z
        for c in names
    ]
    num = len(names) * len(yaws)
    args["num"] = num

    def _set(container: dict[str, Any], key: str) -> None:
        if key == "anchor_number":
            container[key] = num

    _walk(hypes, _set)
    hypes["v2b_anchor_classes"] = names
