"""LitePT-style attention stages for the m1 PointPillars BEV backbone.

LitePT (Yue et al., CVPR 2026, arXiv 2512.13689) keeps convolution in
the early, high-resolution stages of a point backbone and puts
attention, with a parameter-free rotary position encoding (PointROPE),
in the late, low-resolution ones. This module carries that split to
HEAL's m1 BEV backbone, whose stages are ``backbone_m1`` (stride 2,
128x128 cells) and ``layers_m1.resnet.layer1/2/...`` (stride 4, 8, ...).
:func:`install_bev_attention` swaps chosen stages for an
:class:`AttentionStage` with the same input / output channels and
stride, so the neck, shrink conv and heads keep their warm-started
weights.

Tokens are the BEV cells of a stage. They are serialised along a
Z-order curve and cut into groups of ``group`` (LitePT: N = 1024)
cells; attention runs inside a group. At stride 8 the 32x32 grid is
exactly one group, i.e. global attention over the scene.

Positional encodings (``pe``):

* ``rope2d``: RoPE on the cell's (x, y) grid index, LitePT's PointROPE
  restricted to the BEV plane: each head's channels are split into one
  part per axis and every channel pair of a part is rotated by an angle
  ``index * base ** (-2k / d_axis)``;
* ``rope3d``: the same with a third axis, the height of the cell's
  points (mean z of its pillars, in 0.4 m units), the piece BEV drops;
* ``abs``: absolute 2-D sinusoidal embeddings added to the tokens;
* ``none``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

PES = ("rope2d", "rope3d", "abs", "none")
BASE_CELL = 0.4  # m, the pillar size the height axis is measured in
GROUND = -1.9  # m, where the re-based ground sits (v2b.data.ground)


@dataclass
class AttnConfig:
    """One ablation setting; ``stages`` index m1's stages (0 = stride 2)."""

    stages: tuple[int, ...] = (2,)
    depth: tuple[int, ...] = (6,)  # one per stage, or one for all
    heads: int = 8
    mlp_ratio: float = 4.0
    pe: str = "rope2d"
    base: float = 100.0
    group: int = 1024

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AttnConfig:
        cfg = cls(
            stages=tuple(int(s) for s in raw.get("stages", (2,))),
            depth=tuple(
                int(d)
                for d in (
                    raw["depth"]
                    if isinstance(raw.get("depth"), (list, tuple))
                    else [raw.get("depth", 6)]
                )
            ),
            heads=int(raw.get("heads", 8)),
            mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
            pe=str(raw.get("pe", "rope2d")),
            base=float(raw.get("base", 100.0)),
            group=int(raw.get("group", 1024)),
        )
        if cfg.pe not in PES:
            raise ValueError(f"pe {cfg.pe!r}, expected one of {PES}")
        if len(cfg.depth) not in (1, len(cfg.stages)):
            raise ValueError(
                f"depth {list(cfg.depth)}: one value or one per stage "
                f"{list(cfg.stages)}"
            )
        return cfg

    def __post_init__(self) -> None:
        if isinstance(self.depth, int):  # AttnConfig(depth=2) in code
            self.depth = (self.depth,)

    def depth_of(self, i: int) -> int:
        """Blocks of the ``i``-th configured stage."""
        return self.depth[i] if len(self.depth) > 1 else self.depth[0]


def rope_angles(
    pos: torch.Tensor, dim: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin of RoPE angles for positions ``pos`` (..., ) over ``dim``.

    ``dim`` is even; channel pair k turns by ``pos * base ** (-2k/dim)``.
    """
    k = torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32)
    freq = base ** (-k / dim)
    ang = pos.float()[..., None] * freq
    return ang.cos(), ang.sin()


def rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate channel pairs (2k, 2k+1) of ``x`` (..., d) by the angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def axis_rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """PointROPE: one 1-D RoPE per axis on equal channel slices.

    ``x`` is (B, heads, n, d), ``pos`` (B, n, axes). Each axis gets
    ``2 * floor(d / (2 * axes))`` channels; any remainder is left as
    it is, so a head dimension need not divide by the axis count.
    """
    axes = pos.shape[-1]
    part = 2 * (x.shape[-1] // (2 * axes))
    pieces = []
    for a in range(axes):
        cos, sin = rope_angles(pos[..., a], part, base)
        sl = x[..., a * part : (a + 1) * part]
        pieces.append(rotate(sl, cos[:, None], sin[:, None]))
    rest = x[..., axes * part :]
    return torch.cat([*pieces, rest], dim=-1)


def sincos_2d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Absolute 2-D sinusoidal embedding of (B, n, 2) positions."""
    half = dim // 2
    k = torch.arange(0, half, 2, device=pos.device, dtype=torch.float32)
    freq = 10000.0 ** (-k / half)
    out = []
    for a in range(2):
        ang = pos[..., a].float()[..., None] * freq
        out += [ang.sin(), ang.cos()]
    emb = torch.cat(out, dim=-1)
    return functional.pad(emb, (0, dim - emb.shape[-1]))


def zorder(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Indices of an (h, w) grid in Z-order (Morton) sequence."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    code = torch.zeros_like(xs)
    for bit in range(max(h, w).bit_length()):
        code |= ((xs >> bit) & 1) << (2 * bit)
        code |= ((ys >> bit) & 1) << (2 * bit + 1)
    return torch.argsort(code.flatten())


class Block(nn.Module):  # type: ignore[misc]
    """Pre-norm transformer block with an optional RoPE on q and k."""

    def __init__(self, dim: int, cfg: AttnConfig) -> None:
        super().__init__()
        self.heads, self.cfg = cfg.heads, cfg
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        # every block starts as the identity on its residual path, so a
        # warm-started network is not disturbed while the stage learns
        for last in (self.proj, self.mlp[2]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, heads, n, d)
        if self.cfg.pe.startswith("rope"):
            q = axis_rope(q, pos, self.cfg.base)
            k = axis_rope(k, pos, self.cfg.base)
        att = functional.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(att.transpose(1, 2).reshape(b, n, c))
        return x + self.mlp(self.norm2(x))


class AttentionStage(nn.Module):  # type: ignore[misc]
    """A BEV stage: strided pooling to ``c_out``, then attention blocks.

    Stands in for a ResNet layer of the same ``c_in``, ``c_out`` and
    ``stride`` (LitePT's "pool stride 2" + attention blocks).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int,
        cfg: AttnConfig,
        heights: dict[str, torch.Tensor],
        depth: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg, self.stride, self.heights = cfg, stride, heights
        axes = 3 if cfg.pe == "rope3d" else 2
        head_dim = c_out // cfg.heads
        if cfg.pe.startswith("rope") and head_dim < 2 * axes:
            raise ValueError(
                f"{cfg.pe} needs >= {2 * axes} channels per head, "
                f"got {head_dim} ({c_out} / {cfg.heads} heads)"
            )
        self.pool = nn.Sequential(
            nn.Conv2d(c_in, c_out, stride, stride=stride, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        depth = cfg.depth_of(0) if depth is None else depth
        self.blocks = nn.ModuleList(Block(c_out, cfg) for _ in range(depth))
        self.out_norm = nn.LayerNorm(c_out)

    def positions(
        self, b: int, h: int, w: int, dev: torch.device
    ) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(h, device=dev),
            torch.arange(w, device=dev),
            indexing="ij",
        )
        pos = torch.stack([xs, ys], -1).reshape(1, h * w, 2).expand(b, -1, -1)
        if self.cfg.pe != "rope3d":
            return pos.float()
        hm = self.heights.get("map")
        if hm is None or hm.shape[0] != b:
            z = torch.zeros(b, h * w, 1, device=dev)
        else:
            z = functional.adaptive_max_pool2d(hm, (h, w)).reshape(b, h * w, 1)
        return torch.cat([pos.float(), z], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        b, c, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)  # (B, hw, C)
        pos = self.positions(b, h, w, x.device)
        if self.cfg.pe == "abs":
            tok = tok + sincos_2d(pos, c).to(tok.dtype)
        order = zorder(h, w, x.device)
        tok, pos = tok[:, order], pos[:, order]
        n, g = h * w, min(self.cfg.group, h * w)
        pad = (-n) % g
        if pad:
            tok = functional.pad(tok, (0, 0, 0, pad))
            pos = functional.pad(pos, (0, 0, 0, pad))
        groups = tok.shape[1] // g
        tok = tok.reshape(b * groups, g, c)
        pos = pos.reshape(b * groups, g, pos.shape[-1])
        for blk in self.blocks:
            tok = blk(tok, pos)
        tok = self.out_norm(tok).reshape(b, groups * g, c)[:, :n]
        out = torch.empty_like(tok)
        out[:, order] = tok
        return out.transpose(1, 2).reshape(b, c, h, w)


def _height_map(inputs: dict[str, Any], grid: tuple[int, int]) -> torch.Tensor:
    """Mean point height per pillar, in 0.4 m units above the ground.

    ``inputs`` is HEAL's ``inputs_m1`` (voxel_features (P, 32, >=3),
    voxel_num_points (P,), voxel_coords (P, 4) = batch, z, y, x).
    Empty cells read 0, the ground.
    """
    feats = inputs["voxel_features"]
    num = inputs["voxel_num_points"].clamp(min=1).to(feats.dtype)
    z = feats[:, :, 2].sum(dim=1) / num
    coords = inputs["voxel_coords"].long()
    b = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
    h, w = grid
    hm = torch.zeros(b, 1, h, w, device=feats.device, dtype=feats.dtype)
    hm[coords[:, 0], 0, coords[:, 2], coords[:, 3]] = (z - GROUND) / BASE_CELL
    return hm


def install_bev_attention(
    model: nn.Module, raw: dict[str, Any], modality: str = "m1"
) -> list[str]:
    """Swap the configured m1 stages for attention stages; return notes."""
    cfg = AttnConfig.from_dict(raw)
    heights: dict[str, torch.Tensor] = {}
    notes = []
    backbone = getattr(model, f"backbone_{modality}")
    layers = getattr(model, f"layers_{modality}")
    for i, s in enumerate(cfg.stages):
        if s == 0:
            owner, attr = backbone.resnet, "layer0"
        else:
            owner, attr = layers.resnet, f"layer{s}"
        old = getattr(owner, attr)
        first = old[0]
        c_in = first.conv1.in_channels
        c_out = first.conv1.out_channels
        stride = first.conv1.stride[0]
        depth = cfg.depth_of(i)
        stage = AttentionStage(c_in, c_out, stride, cfg, heights, depth)
        setattr(owner, attr, stage)
        notes.append(
            f"stage {s}: {len(old)} conv blocks -> {depth} attention "
            f"blocks ({c_in}->{c_out}, stride {stride}, pe {cfg.pe}, "
            f"base {cfg.base:g}, group {cfg.group})"
        )
    if cfg.pe == "rope3d":
        encoder = getattr(model, f"encoder_{modality}")

        def grab(_: nn.Module, args: tuple[Any, ...]) -> None:
            data = args[0]
            gy, gx = int(encoder.scatter.ny), int(encoder.scatter.nx)
            heights["map"] = _height_map(data[f"inputs_{modality}"], (gy, gx))

        encoder.register_forward_pre_hook(grab)
    return notes


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "PES",
    "AttentionStage",
    "AttnConfig",
    "axis_rope",
    "count_params",
    "install_bev_attention",
    "rope_angles",
    "zorder",
]
