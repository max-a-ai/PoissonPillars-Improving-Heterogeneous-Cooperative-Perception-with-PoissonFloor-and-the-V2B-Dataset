"""A learned code space on the BEV trunk (optional; not used in the
paper's experiments).

The detector's trunk (pillar net, backbone, multiscale layers, shrink
conv) stays as trained; on its 256-channel map sits an **adapter**
(to ``dim`` channels), a **vector-quantised codebook** of ``codes``
entries, and a **reconstructor** back to 256 channels that feeds the
anchor heads. What crosses the wire is the index map, ``log2(codes)``
bits per 0.8 m cell, against 512 bytes per cell for the fp16 map.

The pairing: a split derived with ``--twin-sweep`` holds, next to each
sweep, the same scene with the same synthetic bodies rendered as a
second mount sees them (``v2b/train/twin_data.py`` loads it as
``twin_inputs_m1``). Both clouds run through the trunk and the adapter;
on the cells where the two trunk maps differ by more than ``pair_tau``
of the map's mean activation (0.1: body cells sit at 0.2 to 0.6 with
a trained trunk, the point-order noise of the rest below 0.05), that
is where the twin rendering reached, a **similarity loss** pulls the
two codes together.
The claim under test is that a code learned this way names a person the
same way from either mount, so the detector reads it on a platform it
never trained on. The metrics that say so are logged every step:
``code_agreement`` (share of paired cells whose ego and twin indices
agree), ``perplexity`` (codes in use) and the detection loss through
the bottleneck.

Losses: HEAL's detection loss on the heads' outputs, plus
``vq_weight`` x commitment, ``recon_weight`` x smooth L1 between the
reconstructed and the trunk map (so frozen heads keep seeing what they
were trained on), ``sim_weight`` x the similarity term. The codebook
itself moves by exponential moving averages, dead codes are re-seeded
from the batch, both all-reduced under DDP.

Spec key::

    codespace: {codes: 256, dim: 64, twin: true, freeze: trunk+heads,
                vq_weight: 1.0, recon_weight: 1.0, sim_weight: 1.0,
                pair_tau: 0.1}

``freeze`` is ``trunk+heads`` (CodeAlign's setting: only the adapter,
the codebook and the reconstructor learn), ``trunk`` (heads learn too)
or ``none``. Attribute names of the detector are kept, so a checkpoint
of this wrapper warm-starts a plain detector and the other way round.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as functional

FREEZE_MODES = ("trunk+heads", "trunk", "none")


def _ddp() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


class VectorQuantizer(nn.Module):  # type: ignore[misc]
    """EMA codebook with straight-through gradients and dead-code
    re-seeding; codes are seeded from the first training batch."""

    def __init__(
        self,
        codes: int,
        dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        beta: float = 0.25,
        dead_after: float = 1e-2,
    ) -> None:
        super().__init__()
        self.codes = int(codes)
        self.dim = int(dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.beta = float(beta)
        self.dead_after = float(dead_after)
        self.register_buffer("embed", torch.randn(self.codes, self.dim))
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("cluster_size", torch.zeros(self.codes))
        self.register_buffer("seeded", torch.zeros(1))

    def _seed(self, flat: torch.Tensor) -> None:
        pick = torch.randint(
            0, flat.shape[0], (self.codes,), device=flat.device
        )
        seed = flat[pick].detach().clone()
        if _ddp():
            dist.broadcast(seed, 0)
        self.embed.copy_(seed)
        self.embed_avg.copy_(seed)
        self.cluster_size.fill_(1.0)
        self.seeded.fill_(1.0)

    def indices(self, z: torch.Tensor) -> torch.Tensor:
        """Nearest code of every cell, (B, h, w)."""
        flat = z.detach().permute(0, 2, 3, 1).reshape(-1, self.dim)
        dist2 = (
            (flat**2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + (self.embed**2).sum(1)[None, :]
        )
        idx: torch.Tensor = dist2.argmin(1)
        return idx.view(z.shape[0], z.shape[2], z.shape[3])

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(quantised with straight-through, commitment loss, indices,
        perplexity) for an adapter map ``z`` of shape (B, dim, h, w)."""
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        if self.training and float(self.seeded) == 0.0:
            self._seed(flat)
        idx = self.indices(z).view(-1)
        onehot = functional.one_hot(idx, self.codes).to(flat.dtype)
        quant = self.embed[idx].view(b, h, w, d).permute(0, 3, 1, 2)
        if self.training:
            with torch.no_grad():
                n = onehot.sum(0)
                dw = onehot.t() @ flat.detach()
                if _ddp():
                    dist.all_reduce(n)
                    dist.all_reduce(dw)
                self.cluster_size.mul_(self.decay).add_(
                    n, alpha=1 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                size = (
                    (self.cluster_size + self.eps)
                    / (total + self.codes * self.eps)
                    * total
                )
                self.embed.copy_(self.embed_avg / size.unsqueeze(1))
                dead = self.cluster_size < self.dead_after
                if bool(dead.any()):
                    pick = torch.randint(
                        0, flat.shape[0], (int(dead.sum()),), device=z.device
                    )
                    fresh = flat.detach()[pick]
                    if _ddp():
                        dist.broadcast(fresh, 0)
                    self.embed[dead] = fresh
                    self.embed_avg[dead] = fresh
                    self.cluster_size[dead] = 1.0
        commit = self.beta * functional.mse_loss(z, quant.detach())
        zq = z + (quant - z).detach()
        probs = onehot.mean(0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return zq, commit, idx.view(b, h, w), perplexity


def _out_channels(module: Any) -> int:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("shrink head without a conv: cannot size the adapter")
    return int(convs[-1].out_channels)


class CodeSpaceModel(nn.Module):  # type: ignore[misc]
    """The detector with the code bottleneck between trunk and heads.

    ``forward`` takes HEAL's ``batch["ego"]`` and returns the heads'
    ``cls_preds`` / ``reg_preds`` / ``dir_preds`` (so the post-processor
    and the AP evaluation run unchanged) plus ``codespace``: the
    auxiliary losses and metrics.
    """

    def __init__(
        self,
        detector: Any,
        codes: int = 256,
        dim: int = 64,
        freeze: str = "trunk+heads",
        decay: float = 0.99,
        beta: float = 0.25,
        pair_tau: float = 0.1,
    ) -> None:
        super().__init__()
        if freeze not in FREEZE_MODES:
            raise ValueError(
                f"freeze {freeze!r}, expected one of {FREEZE_MODES}"
            )
        self.encoder_m1 = detector.encoder_m1
        self.backbone_m1 = detector.backbone_m1
        self.layers_m1 = detector.layers_m1
        self.shrink_conv_m1 = detector.shrink_conv_m1
        self.layers_num_m1 = int(detector.layers_num_m1)
        self.cls_head_m1 = detector.cls_head_m1
        self.reg_head_m1 = detector.reg_head_m1
        self.dir_head_m1 = detector.dir_head_m1
        c = _out_channels(detector.shrink_conv_m1)
        self.adapter = nn.Sequential(
            nn.Conv2d(c, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
        self.vq = VectorQuantizer(codes, dim, decay=decay, beta=beta)
        self.reconstructor = nn.Sequential(
            nn.Conv2d(dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.freeze = freeze
        self.pair_tau = float(pair_tau)
        self.codes = int(codes)
        self.bits_per_cell = math.ceil(math.log2(max(codes, 2)))
        for m in self.frozen_modules():
            for p in m.parameters():
                p.requires_grad_(False)

    # -- which parts stay as trained ------------------------------------
    def trunk_modules(self) -> list[nn.Module]:
        return [
            self.encoder_m1,
            self.backbone_m1,
            self.layers_m1,
            self.shrink_conv_m1,
        ]

    def head_modules(self) -> list[nn.Module]:
        return [self.cls_head_m1, self.reg_head_m1, self.dir_head_m1]

    def frozen_modules(self) -> list[nn.Module]:
        if self.freeze == "none":
            return []
        if self.freeze == "trunk":
            return self.trunk_modules()
        return self.trunk_modules() + self.head_modules()

    def train(self, mode: bool = True) -> CodeSpaceModel:
        """Frozen parts stay in eval mode: their BatchNorm statistics
        are part of what was trained."""
        super().train(mode)
        for m in self.frozen_modules():
            m.eval()
        return self

    def model_train_init(self) -> None:
        return None

    # -- the pieces -------------------------------------------------------
    def trunk(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        scattered = self.encoder_m1({"inputs_m1": inputs}, "m1")
        feature = self.backbone_m1({"spatial_features": scattered})[
            "spatial_features_2d"
        ]
        feats = [feature]
        for i in range(1, self.layers_num_m1):
            feature = self.layers_m1.get_layer_i_feature(feature, layer_i=i)
            feats.append(feature)
        fused = self.layers_m1.decode_multiscale_feature(feats)
        out: torch.Tensor = self.shrink_conv_m1(fused)
        return out

    def heads(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cls_preds": self.cls_head_m1(feature),
            "reg_preds": self.reg_head_m1(feature),
            "dir_preds": self.dir_head_m1(feature),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        trunk_grad = self.freeze == "none"
        with torch.set_grad_enabled(trunk_grad and torch.is_grad_enabled()):
            f_ego = self.trunk(batch["inputs_m1"])
        z = self.adapter(f_ego)
        zq, commit, idx, perplexity = self.vq(z)
        recon = self.reconstructor(zq)
        out: dict[str, Any] = self.heads(recon)
        extras: dict[str, torch.Tensor] = {
            "vq": commit,
            "recon": functional.smooth_l1_loss(recon, f_ego.detach()),
            "perplexity": perplexity.detach(),
            "sim": z.sum() * 0.0,
            "paired_cells": torch.zeros((), device=z.device),
            "code_agreement": torch.zeros((), device=z.device),
        }
        twin = batch.get("twin_inputs_m1")
        if twin is not None:
            with torch.set_grad_enabled(
                trunk_grad and torch.is_grad_enabled()
            ):
                f_twin = self.trunk(twin)
            z_twin = self.adapter(f_twin)
            # the cells the twin rendering reached: where the trunk
            # maps differ clearly (the rest of both clouds is the same
            # sweep up to point order, which stays far below the bar)
            gap = (f_ego.detach() - f_twin.detach()).abs().mean(1)
            diff = gap > self.pair_tau * f_ego.detach().abs().mean()
            if bool(diff.any()):
                sel = diff.unsqueeze(1).expand_as(z)
                extras["sim"] = functional.smooth_l1_loss(z[sel], z_twin[sel])
                idx_twin = self.vq.indices(z_twin)
                extras["code_agreement"] = (
                    (idx[diff] == idx_twin[diff]).float().mean().detach()
                )
            extras["paired_cells"] = diff.float().sum() / max(z.shape[0], 1)
        out["codespace"] = extras
        out["code_indices"] = idx
        return out


class CodeSpaceLoss:
    """HEAL's criterion plus the code-space terms; same call signature."""

    def __init__(
        self,
        criterion: Any,
        vq_weight: float = 1.0,
        recon_weight: float = 1.0,
        sim_weight: float = 1.0,
    ) -> None:
        self.criterion = criterion
        self.vq_weight = float(vq_weight)
        self.recon_weight = float(recon_weight)
        self.sim_weight = float(sim_weight)
        self.loss_dict: dict[str, float] = {}

    def __call__(
        self, output: dict[str, Any], target: dict[str, Any], suffix: str = ""
    ) -> Any:
        base = self.criterion(output, target, suffix)
        ex = output["codespace"]
        total = (
            base
            + self.vq_weight * ex["vq"]
            + self.recon_weight * ex["recon"]
            + self.sim_weight * ex["sim"]
        )
        self.loss_dict = dict(getattr(self.criterion, "loss_dict", {}))
        self.loss_dict.update(
            {
                "vq_loss": float(ex["vq"].detach()),
                "recon_loss": float(ex["recon"].detach()),
                "sim_loss": float(ex["sim"].detach()),
                "perplexity": float(ex["perplexity"]),
                "paired_cells": float(ex["paired_cells"]),
                "code_agreement": float(ex["code_agreement"]),
                "total_loss": float(total.detach()),
            }
        )
        return total

    def logging(self, *args: Any, **kwargs: Any) -> Any:
        return self.criterion.logging(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.criterion, name)


def wrap_codespace(detector: Any, cfg: dict[str, Any]) -> CodeSpaceModel:
    """The wrapper for a spec's ``codespace`` block."""
    return CodeSpaceModel(
        detector,
        codes=int(cfg.get("codes", 256)),
        dim=int(cfg.get("dim", 64)),
        freeze=str(cfg.get("freeze", "trunk+heads")),
        decay=float(cfg.get("decay", 0.99)),
        beta=float(cfg.get("beta", 0.25)),
        pair_tau=float(cfg.get("pair_tau", 0.1)),
    )


def message_bytes(cfg: dict[str, Any], cells: int) -> float:
    """Bytes per frame of the index map over ``cells`` BEV cells."""
    bits = math.ceil(math.log2(max(int(cfg.get("codes", 256)), 2)))
    return cells * bits / 8
