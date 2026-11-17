"""Our single-agent trunk inside HEAL's intermediate-fusion models.

HEAL's fusion model (``heter_model_baseline``) computes one map per
agent as ``encoder -> BaseBEVBackbone -> shrinker`` and fuses those;
its single-agent model (``heter_model_late``) computes ``encoder ->
ResNetBEVBackbone -> multiscale layers -> shrink conv``. Both end in a
256-channel map at the head's stride, so the fusion net and the heads
never notice which trunk made it; but the two trunks share only the
pillar net, so a fusion run warm-started from a single-agent
checkpoint adopts 10 of 152 tensors and trains the rest from scratch
(the S25 and S27 rows), which is why fusion scored below no-fusion on
pedestrians on both sides.

:func:`install_single_trunk` gives the fusion model the single-agent
trunk under the single-agent names (``backbone_m1``, ``layers_m1``,
``shrink_conv_m1``, built from the same ``m1_pointpillar_pretrain``
settings), and replaces ``shrinker_m1`` with a neck that runs the
multiscale layers and the shrink conv. HEAL's forward is untouched,
and ``warm_start`` from a single-agent run then carries the whole trunk.
Spec key ``fusion_trunk: single``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch import nn

SINGLE_BASE = "hypes_yaml/opv2v/Single/m1_pointpillar_pretrain.yaml"


class LateNeck(nn.Module):  # type: ignore[misc]
    """The multiscale layers and the shrink conv of the single-agent
    model, applied after its ResNet backbone; the modules themselves are
    registered on the parent under their own names, this neck only
    runs them (held outside registration, so no key is duplicated)."""

    def __init__(
        self, layers: nn.Module, shrink: nn.Module, n: int, pool: int = 1
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_parts", (layers, shrink))
        self.n = int(n)
        # a fusion config whose head sits at a coarser stride (V2X-ViT:
        # its shrink header strides by 2) gets the trunk's 0.8 m map
        # pooled down to it, parameter-free
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()

    def forward(self, feature: Any) -> Any:
        layers, shrink = self._parts
        feats = [feature]
        for i in range(1, self.n):
            feature = layers.get_layer_i_feature(feature, layer_i=i)
            feats.append(feature)
        return self.pool(shrink(layers.decode_multiscale_feature(feats)))


def single_settings() -> dict[str, Any]:
    """The m1 trunk settings of HEAL's single-agent base config."""
    import opencood

    path = Path(opencood.__file__).parent / SINGLE_BASE
    doc = yaml.safe_load(path.read_text())
    return dict(doc["model"]["args"]["m1"])


def head_stride_factor(hypes: dict[str, Any], modality: str = "m1") -> int:
    """How much coarser the fusion config's per-agent map is than the
    single-agent trunk's: the stride of its shrink header (1 for
    AttFuse and F-Cooper, 2 for V2X-ViT)."""
    try:
        stride = hypes["model"]["args"][modality]["shrink_header"]["stride"]
    except (KeyError, TypeError):
        return 1
    return int(stride[0] if isinstance(stride, list) else stride)


def install_single_trunk(
    model: Any, modality: str = "m1", pool: int = 1
) -> str:
    """Swap the fusion model's per-agent trunk for the single-agent one;
    ``pool`` > 1 downsamples its map to a coarser fusion head."""
    from opencood.models.sub_modules.base_bev_backbone_resnet import (
        ResNetBEVBackbone,
    )
    from opencood.models.sub_modules.downsample_conv import DownsampleConv

    if not hasattr(model, f"shrinker_{modality}"):
        raise ValueError(
            "fusion_trunk: single needs HEAL's intermediate fusion model "
            f"(no shrinker_{modality} on {type(model).__name__})"
        )
    setting = single_settings()
    backbone = ResNetBEVBackbone(setting["backbone_args"])
    layers = ResNetBEVBackbone(setting["layers_args"])
    n = len(setting["layers_args"]["num_upsample_filter"])
    shrink = DownsampleConv(setting["shrink_header"])
    setattr(model, f"backbone_{modality}", backbone)
    setattr(model, f"layers_{modality}", layers)
    setattr(model, f"layers_num_{modality}", n)
    setattr(model, f"shrink_conv_{modality}", shrink)
    setattr(model, f"shrinker_{modality}", LateNeck(layers, shrink, n, pool))
    params = sum(
        p.numel() for m in (backbone, layers, shrink) for p in m.parameters()
    )
    return (
        f"fusion trunk: the single-agent trunk ({params / 1e6:.2f} M params: "
        "ResNet backbone, multiscale layers, shrink conv) per agent, under "
        "its own names, so a single-agent warm start adopts it whole"
        + (f"; pooled {pool}x to the fusion head's stride" if pool > 1 else "")
    )


def freeze_adopted(model: Any, keys: list[str]) -> str:
    """The frozen-trunk control: every tensor the warm start adopted
    stops training and its norm layers hold their running statistics,
    so only the fresh tensors (the fusion module, the heads) move.
    Install after any SyncBatchNorm conversion and before DDP."""
    names = set(keys)
    frozen = 0
    for name, p in model.named_parameters():
        if name in names:
            p.requires_grad_(False)
            frozen += p.numel()
    modules = {name.rsplit(".", 1)[0] for name in names}
    held = [
        m
        for name, m in model.named_modules()
        if name in modules and isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    original_train = model.train

    def train(mode: bool = True) -> Any:
        original_train(mode)
        for m in held:
            m.eval()
        return model

    model.train = train
    return (
        f"frozen trunk: {frozen / 1e6:.2f} M warm-started params fixed, "
        f"{len(held)} norm layers held; only the fresh tensors train"
    )
