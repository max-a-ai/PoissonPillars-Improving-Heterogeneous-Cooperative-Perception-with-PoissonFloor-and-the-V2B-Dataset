"""Run a trained learned-PoissonFloor model on raw sweeps.

Loads the run's rendered hypes and best-validation weights, builds HEAL's
pillar preprocessor for the run's grid and turns one raw cloud into a
:class:`v2b.data.ground.GroundField` through :meth:`FloorModel.height_field`;
``v2b-derive-ground.py --surface learned --model <run>`` uses it in place
of the geometric fits. Needs the HEAL fork (``V2B_FORK=heal``), a GPU and
the run's dataset root linked for the preprocessor's hypes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from v2b.data.ground import GroundField

F32 = npt.NDArray[np.float32]


class LearnedFloor:
    """The floor network of one run as a per-sweep surface fitter."""

    def __init__(self, run: Path, device: str = "cuda") -> None:
        from v2b.train.hypes import fork_dir
        from v2b.utils.opencood_shims import install as install_shims

        # the fork's opencood on the path, as the trainer does it
        install_shims(fork_dir(os.environ.setdefault("V2B_FORK", "heal")))

        from opencood.data_utils.pre_processor import build_preprocessor
        from opencood.tools import train_utils

        from v2b.train.floor_net import FloorModel
        from v2b.train.heal_data import install_point_channels
        from v2b.train.opencood_train import load_hypes, warm_start

        hypes = load_hypes(str(run / "config.yaml"))
        channels = int(hypes.get("v2b_point_channels", 4))
        if channels:
            install_point_channels(channels)
        self.channels = channels
        ego = str(hypes["heter"]["ego_modality"]).split("&")[0]
        setting = hypes["heter"]["modality_setting"][ego]["preprocess"]
        detector = train_utils.create_model(hypes)
        model = FloorModel(
            detector,
            list(setting["cav_lidar_range"]),
            modality=ego,
            voxel_size=[float(v) for v in setting["args"]["voxel_size"]],
            decoder_layers=int(
                (hypes.get("v2b_pretrain") or {}).get("decoder_layers", 1)
            ),
        )
        warm_start(model, str(run))
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.pre = build_preprocessor(setting, train=False)
        self.modality = ego

    def fit(self, points: F32) -> GroundField:
        """The predicted floor of one raw sweep (x, y, z, intensity, ...)
        in its sensor frame; extra channels beyond the run's are dropped,
        missing ones (the ``hag`` slot) are zero."""
        pts = np.asarray(points, np.float32)
        if pts.shape[1] < self.channels:
            pad = np.zeros(
                (pts.shape[0], self.channels - pts.shape[1]), np.float32
            )
            pts = np.concatenate([pts, pad], axis=1)
        pts = pts[:, : self.channels]
        vox: dict[str, Any] = self.pre.preprocess(pts)
        coords = np.pad(
            vox["voxel_coords"], ((0, 0), (1, 0)), constant_values=0
        )
        inputs = {
            "voxel_features": torch.from_numpy(vox["voxel_features"]).to(
                self.device
            ),
            "voxel_coords": torch.from_numpy(coords).to(self.device),
            "voxel_num_points": torch.from_numpy(vox["voxel_num_points"]).to(
                self.device
            ),
        }
        field: GroundField = self.model.height_field(inputs)
        return field
