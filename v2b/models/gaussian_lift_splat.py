"""GaussianLSS as a HEAL / CodeFilling camera encoder.

HEAL builds a modality's encoder by looking a class up by name in
``opencood.models.heter_encoders`` (``core_method`` with the
underscores removed, case-insensitive). ``register()`` puts
``GaussianLiftSplat`` there, so a config that says
``core_method: gaussian_lift_splat`` with the ordinary
``lift_splat_shoot`` ``encoder_args`` trains the GaussianLSS view
transform in place of the lift-splat one. Everything after the BEV
feature (backbone, shrink conv, pyramid fusion, heads, depth loss) is
untouched.

What changes against ``LiftSplatShoot``
---------------------------------------
Both share the EfficientNet trunk, the 1/8 feature map, the 1x1 image
and depth heads and HEAL's frustum geometry (``get_geometry`` gives
every (pixel, depth bin) its ego-frame 3D point, augmentation
included). Lift-splat then forms the D-fold outer product of features
and depth probabilities and sums it into the grid. Here each pixel
becomes one 3D Gaussian instead: its mean is the depth-probability
weighted 3D point, its covariance the weighted scatter of the bin
points about that mean (scaled by ``error_tolerance**2 / 9`` as in
GaussianLSS), its opacity a third 1x1 head, and the Gaussians are
rasterised onto the BEV grid by :class:`GaussianBevRenderer`. The
depth logits still feed HEAL's depth loss, so ``depth_supervision``
works unchanged; ``use_depth_gt`` is refused because a one-hot depth
has zero scatter and would collapse every Gaussian to a point.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

from v2b.models.gaussian_splat import GaussianBevRenderer


def pixel_gaussians(
    prob: torch.Tensor, geom: torch.Tensor, error_tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance of each pixel's depth distribution in 3D.

    ``prob (M, D, H, W)`` are the per-pixel depth probabilities,
    ``geom (M, D, H, W, 3)`` the 3D point of each depth bin. Returns
    ``mean (M, H, W, 3)`` and ``cov (M, H, W, 3, 3)``.
    """
    p = prob.unsqueeze(-1)
    mean = (p * geom).sum(1)
    delta = geom - mean.unsqueeze(1)
    cov = (p.unsqueeze(-1) * delta.unsqueeze(-1) * delta.unsqueeze(-2)).sum(1)
    return mean, cov * (error_tolerance**2 / 9.0)


def build_class(base: type) -> type:
    """``GaussianLiftSplat`` derived from the fork's ``LiftSplatShoot``.

    The base class lives in whichever ``opencood`` is on the path, so
    the subclass is made at registration time rather than at import.
    """

    class GaussianLiftSplat(base):  # type: ignore[misc]
        def __init__(self, args: dict[str, Any]) -> None:
            super().__init__(args)
            if args.get("use_depth_gt"):
                raise ValueError(
                    "gaussian_lift_splat needs a predicted depth "
                    "distribution; set use_depth_gt: false"
                )
            self.error_tolerance = float(args.get("error_tolerance", 1.0))
            self.opacity_head = nn.Conv2d(512, 1, kernel_size=1)
            gc = self.grid_conf
            self.renderer = GaussianBevRenderer(
                self.camC,
                tuple(gc["xbound"]),
                tuple(gc["ybound"]),
                opacity_min=float(args.get("opacity_min", 0.05)),
            )
            self.num_gaussians = 0.0

        def get_voxels(
            self,
            x: torch.Tensor,
            rots: torch.Tensor,
            trans: torch.Tensor,
            intrins: torch.Tensor,
            post_rots: torch.Tensor,
            post_trans: torch.Tensor,
        ) -> tuple[torch.Tensor, Any]:
            geom = self.get_geometry(
                rots, trans, intrins, post_rots, post_trans
            )
            b, n, c, im_h, im_w = x.shape
            x = x.reshape(b * n, c, im_h, im_w)
            cam = self.camencode
            feats = cam.get_eff_features(x[:, :3])
            feat = cam.image_head(feats)
            depth_logit = cam.depth_head(feats)
            opacity = self.opacity_head(feats).sigmoid()
            depth_items = None
            if self.depth_supervision:
                _, depth_gt_indices = cam.get_gt_depth_dist(x[:, 3])
                depth_items = (depth_logit, depth_gt_indices)
            f_h, f_w = feat.shape[-2:]
            mean, cov = pixel_gaussians(
                depth_logit.softmax(1),
                geom.reshape(b * n, self.D, f_h, f_w, 3),
                self.error_tolerance,
            )
            bev, drawn = self.renderer(
                feat.permute(0, 2, 3, 1).reshape(b, n * f_h * f_w, self.camC),
                mean.reshape(b, n * f_h * f_w, 3),
                cov.reshape(b, n * f_h * f_w, 3, 3),
                opacity.permute(0, 2, 3, 1).reshape(b, n * f_h * f_w, 1),
            )
            self.num_gaussians = float(drawn)
            return bev, depth_items

    return GaussianLiftSplat


def register() -> type:
    """Make ``core_method: gaussian_lift_splat`` resolvable in the fork.

    Idempotent; returns the registered class.
    """
    enc = importlib.import_module("opencood.models.heter_encoders")
    cls = getattr(enc, "GaussianLiftSplat", None)
    if cls is None:
        cls = build_class(enc.LiftSplatShoot)
        setattr(enc, "GaussianLiftSplat", cls)  # noqa: B010
    return cls
