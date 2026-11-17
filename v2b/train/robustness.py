"""Test-time robustness hooks and the message-size accounting.

Three things the paper's robustness block needs from the evaluation
path, none of which touches training:

* **Pose noise** (:func:`install_pose_noise`): Gaussian noise on the
  other agents' poses, the CoAlign protocol as HEAL's
  ``inference_w_noise.py`` runs it: ``x`` and ``y`` get
  ``N(0, pos_std)`` metres, the yaw ``N(0, rot_std)`` degrees, ``z``,
  roll and pitch stay (``pose_utils.generate_noise``). Two deliberate
  departures from HEAL: the ego's own pose stays clean (HEAL noises
  every agent, so the relative error is the difference of two draws),
  and the draw is seeded per frame and agent (``[seed, frame, agent]``
  through ``numpy.random.default_rng``) instead of one
  ``np.random.seed(303)`` in the main process, which the loader's
  worker processes do not inherit. The same frame, agent and sigma
  give the same noise on every machine and worker count. HEAL's own
  ``add_noise_data_dict`` is disarmed so it does not overwrite the
  clean pose this hook records.
* **Latency** (:func:`install_latency`): the ego frame ``t`` is paired
  with the other agents' frame ``t - k`` (their cloud and their pose
  at capture time), the labels stay those of ``t``. The V2X-Real V2V
  export numbers frames ``000000, 000001, ...`` per scenario at 10 Hz
  (the yaml ``timestamp`` steps by 0.1 s) and both agents carry the
  same frame list, so ``t - k`` is the other vehicle's k-th previous
  sweep. Frames at a scenario start without a predecessor use the
  oldest frame available (``k = 0`` at ``t = 0``) and are counted.
* **Bytes per message** (:func:`feature_bytes`, :func:`box_bytes`,
  :func:`codebook_bytes`): feature fusion sends one agent's BEV map
  (``C x H x W`` of the tensor the fusion module receives) at fp16;
  late fusion sends the box list, seven fp32 per box plus one fp32
  score; the codebook message is the box list plus 9.5 bytes per VRU.
  :class:`FeatureProbe` records the map's shape from the model's
  forward.

Pose rows follow HEAL: ``[x, y, z, roll, yaw, pitch]``, angles in
degrees.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

#: HEAL's seed in ``inference_w_noise.py``.
NOISE_SEED = 303
#: fp16 feature maps, fp32 box rows, the codebook's bytes per VRU.
FEATURE_BYTES_PER_VALUE = 2
BOX_FLOATS = 7 + 1
BOX_BYTES_PER_VALUE = 4
CODEBOOK_BYTES_PER_VRU = 9.5
#: Pose rows: x, y, z, roll, yaw, pitch.
POSE_X, POSE_Y, POSE_YAW = 0, 1, 4


def noise_vector(pos_std: float, rot_std: float, rng: Any) -> np.ndarray:
    """One pose perturbation ``[dx, dy, 0, 0, dyaw, 0]``, HEAL's layout
    and draw order (x, y, then yaw); metres and degrees."""
    xy = rng.normal(0.0, pos_std, size=2) if pos_std > 0 else np.zeros(2)
    yaw = rng.normal(0.0, rot_std, size=1) if rot_std > 0 else np.zeros(1)
    out = np.zeros(6)
    out[POSE_X], out[POSE_Y], out[POSE_YAW] = xy[0], xy[1], yaw[0]
    return out


def frame_rng(seed: int, frame: int, agent: int) -> Any:
    """The generator of one (frame, agent) cell; independent of the
    worker that loads the frame."""
    return np.random.default_rng([int(seed), int(frame), int(agent)])


def delayed_index(t: int, k: int) -> int:
    """The other agent's frame index for latency ``k``: ``t - k``,
    clipped at the scenario start."""
    return max(0, int(t) - int(k))


def feature_bytes(
    shape: tuple[int, ...], bytes_per_value: int = FEATURE_BYTES_PER_VALUE
) -> int:
    """Bytes of one agent's ``C x H x W`` feature map."""
    n = 1
    for s in shape:
        n *= int(s)
    return n * bytes_per_value


def box_bytes(n_boxes: float) -> float:
    """Bytes of a box list: seven fp32 per box and one fp32 score."""
    return float(n_boxes) * BOX_FLOATS * BOX_BYTES_PER_VALUE


def codebook_bytes(n_boxes: float, n_vru: float) -> float:
    """The codebook message: the box list plus 9.5 bytes per VRU."""
    return box_bytes(n_boxes) + float(n_vru) * CODEBOOK_BYTES_PER_VRU


def perturb_poses(
    data: Any, frame: int, pos_std: float, rot_std: float, seed: int
) -> Any:
    """Record every agent's clean pose and add noise to the non-ego
    poses; ``data`` is HEAL's ``retrieve_base_data`` dict."""
    for j, cav in enumerate(data.values()):
        params = cav["params"]
        pose = np.asarray(params["lidar_pose"], dtype=np.float64)
        params["lidar_pose_clean"] = pose.tolist()
        if cav.get("ego"):
            continue
        rng = frame_rng(seed, frame, j)
        params["lidar_pose"] = (
            pose + noise_vector(pos_std, rot_std, rng)
        ).tolist()
    return data


def _keep_clean(data: Any, noise_setting: Any) -> Any:
    """HEAL's ``add_noise_data_dict`` stand-in: fills the clean pose
    where a hook has not, and never draws noise itself."""
    for cav in data.values():
        params = cav["params"]
        params.setdefault("lidar_pose_clean", params["lidar_pose"])
    return data


def _disarm_heal_noise() -> None:
    from opencood.data_utils.datasets import (
        intermediate_heter_fusion_dataset,
        late_heter_fusion_dataset,
    )

    for mod in (intermediate_heter_fusion_dataset, late_heter_fusion_dataset):
        mod.add_noise_data_dict = _keep_clean


def install_pose_noise(
    pos_std: float, rot_std: float, seed: int = NOISE_SEED
) -> None:
    """Patch HEAL's OPV2V base loader (once per process) so the other
    agents' poses carry fixed-seed Gaussian noise at load time."""
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset

    if getattr(opv2v_basedataset, "_v2b_pose_noise", None) is not None:
        raise RuntimeError("pose noise already installed in this process")
    base = opv2v_basedataset.OPV2VBaseDataset
    original = base.retrieve_base_data

    def retrieve_base_data(self: Any, idx: int) -> Any:
        return perturb_poses(original(self, idx), idx, pos_std, rot_std, seed)

    base.retrieve_base_data = retrieve_base_data
    _disarm_heal_noise()
    opv2v_basedataset._v2b_pose_noise = (pos_std, rot_std, seed)


def scenario_frame(len_record: list[int], idx: int) -> tuple[int, int]:
    """HEAL's ``(scenario index, frame index)`` of a flat sample index,
    from the cumulative frame counts."""
    scenario = 0
    for i, end in enumerate(len_record):
        if idx < end:
            scenario = i
            break
    start = len_record[scenario - 1] if scenario else 0
    return scenario, idx - start


def install_latency(k: int) -> None:
    """Patch HEAL's OPV2V base loader (once per process): the non-ego
    agents' cloud and pose come from their frame ``t - k``."""
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset
    from opencood.utils import pcd_utils

    if getattr(opv2v_basedataset, "_v2b_latency", None) is not None:
        raise RuntimeError("latency already installed in this process")
    base = opv2v_basedataset.OPV2VBaseDataset
    original = base.retrieve_base_data

    def retrieve_base_data(self: Any, idx: int) -> Any:
        data = original(self, idx)
        if k <= 0:
            return data
        scenario, t = scenario_frame(list(self.len_record), idx)
        t_old = delayed_index(t, k)
        if t_old == t:
            return data
        database = self.scenario_database[scenario]
        for cav_id, cav in data.items():
            if cav.get("ego"):
                continue
            content = database[cav_id]
            keys = [key for key in content if key != "ego"]
            old = content[keys[t_old]]
            params_old = opv2v_basedataset.load_yaml(old["yaml"])
            cav["params"]["lidar_pose"] = params_old["lidar_pose"]
            cav["params"].pop("lidar_pose_clean", None)
            if "lidar_np" in cav:
                cav["lidar_np"] = pcd_utils.pcd_to_np(old["lidar"])
            cav["v2b_delayed_from"] = os.path.basename(old["yaml"])
        return data

    base.retrieve_base_data = retrieve_base_data
    _disarm_heal_noise()
    opv2v_basedataset._v2b_latency = k


def install_ego_only(model: Any) -> bool:
    """Run a feature-fusion model on the ego's map alone: the other
    agents' maps are dropped right before the fusion module, so every
    sample reaches it with one agent. Frames, labels (the union) and
    weights stay those of the fused run; the difference between the two
    evaluations is what the other agent's message is worth to this
    model. ``False`` when the model has no fusion module."""
    fusion = getattr(model, "fusion_net", None)
    if fusion is None:
        return False
    original = fusion.forward

    def forward(x: Any, record_len: Any, affine: Any, *rest: Any) -> Any:
        import torch

        starts = torch.cumsum(record_len, 0) - record_len
        ones = torch.ones_like(record_len)
        return original(x[starts.long()], ones, affine, *rest)

    fusion.forward = forward
    return True


def delayed_frames(len_record: list[int], k: int) -> int:
    """How many samples of a dataset fall back to an older ``k`` at a
    scenario start: ``min(k, frames)`` per scenario."""
    counts = [
        end - (len_record[i - 1] if i else 0)
        for i, end in enumerate(len_record)
    ]
    return sum(min(int(k), n) for n in counts)


class FeatureProbe:
    """Records the shape of the per-agent map the fusion module
    receives: ``(N_agents, C, H, W)`` on HEAL's baseline models."""

    def __init__(self) -> None:
        self.shape: tuple[int, ...] | None = None
        self._handle: Any = None

    def attach(self, model: Any) -> bool:
        fusion = getattr(model, "fusion_net", None)
        if fusion is None:
            return False

        def pre_hook(module: Any, inputs: Any) -> None:
            feature = inputs[0]
            self.shape = tuple(int(s) for s in feature.shape[1:])

        self._handle = fusion.register_forward_pre_hook(pre_hook)
        return True

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def message_bytes(self) -> int | None:
        return None if self.shape is None else feature_bytes(self.shape)
