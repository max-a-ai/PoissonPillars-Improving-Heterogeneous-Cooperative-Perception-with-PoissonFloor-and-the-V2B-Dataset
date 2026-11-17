"""The twin observation next to every training sweep, for HEAL's loader.

A split derived with ``v2b-derive-synth.py --twin-sweep`` holds, under
each agent folder, ``twin/<index>.pcd``: the same scene with the
same synthetic bodies rendered as another mount (other rings, other
height) would see them. The code-space runs (``v2b/train/codespace.py``)
need both clouds of a frame in one batch. Three hooks on HEAL's late
heterogeneous dataset do that without touching its files:

* ``retrieve_base_data`` also loads the twin cloud when the file exists
  (``lidar_twin_np``);
* ``get_item_single_car`` voxelises it with the agent's own
  pre-processor, after the same range and ego masks, without
  augmentation (``render`` empties ``data_augment`` for twin runs, so
  the two clouds stay aligned);
* ``collate_batch_train`` collates the twins of a batch into
  ``twin_inputs_<modality>``. The key does not start with ``inputs_``
  on purpose: ``heter_model_late.forward`` asserts exactly one such key.

Frames without a twin file (the validation split, a plain split) go
through unchanged and the batch carries no twin key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TWIN_DIR = "twin"


def twin_path(lidar_file: str | Path) -> Path:
    """``<agent>/twin/<index>.pcd`` for ``<agent>/<index>.pcd``; a
    subfolder, so frame listings (``*/*/*.pcd``) do not see it."""
    p = Path(lidar_file)
    return p.parent / TWIN_DIR / p.name


def install_twin_loader() -> None:
    """Patch HEAL's OPV2V base and late heterogeneous datasets (once)."""
    import opencood.data_utils.datasets as datasets
    from opencood.data_utils.datasets import late_heter_fusion_dataset
    from opencood.data_utils.datasets.basedataset import opv2v_basedataset
    from opencood.utils import pcd_utils

    if getattr(opv2v_basedataset, "_v2b_twin", False):
        return
    base = opv2v_basedataset.OPV2VBaseDataset
    original_retrieve = base.retrieve_base_data

    def retrieve_base_data(self: Any, idx: int) -> Any:
        data = original_retrieve(self, idx)
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        db = self.scenario_database[scenario_index]
        t_index = (
            idx
            if scenario_index == 0
            else idx - self.len_record[scenario_index - 1]
        )
        key = self.return_timestamp_key(db, t_index)
        for cav_id, content in db.items():
            if cav_id not in data or "lidar_np" not in data[cav_id]:
                continue
            twin = twin_path(content[key]["lidar"])
            if twin.exists():
                # the reader in place at call time (point_channels may
                # have swapped it)
                data[cav_id]["lidar_twin_np"] = pcd_utils.pcd_to_np(str(twin))
        return data

    base.retrieve_base_data = retrieve_base_data
    original_factory = late_heter_fusion_dataset.getLateheterFusionDataset

    def factory(cls: Any) -> Any:
        made = original_factory(cls)
        original_item = made.get_item_single_car
        original_collate = made.collate_batch_train

        class TwinLateheterFusionDataset(made):  # type: ignore[misc,valid-type]
            def get_item_single_car(self: Any, selected_cav_base: Any) -> Any:
                out = original_item(self, selected_cav_base)
                twin = selected_cav_base.get("lidar_twin_np")
                modality = selected_cav_base["modality_name"]
                if twin is None or self.sensor_type_dict[modality] != "lidar":
                    return out
                twin = pcd_utils.shuffle_points(twin)
                twin = pcd_utils.mask_points_by_range(
                    twin, self.params["preprocess"]["cav_lidar_range"]
                )
                twin = pcd_utils.mask_ego_points(twin)
                pre = getattr(self, f"pre_processor_{modality}")
                out[f"twin_features_{modality}"] = pre.preprocess(twin)
                return out

            def collate_batch_train(self: Any, batch: Any) -> Any:
                out = original_collate(self, batch)
                if out is None:
                    return out
                for modality in self.modality_name_list:
                    key = f"twin_features_{modality}"
                    feats = [b["ego"][key] for b in batch if key in b["ego"]]
                    if feats and len(feats) == len(batch):
                        pre = getattr(self, f"pre_processor_{modality}")
                        out["ego"][f"twin_inputs_{modality}"] = (
                            pre.collate_batch(feats)
                        )
                return out

        TwinLateheterFusionDataset.__name__ = made.__name__
        return TwinLateheterFusionDataset

    late_heter_fusion_dataset.getLateheterFusionDataset = factory
    # build_dataset resolves the factory by name in this namespace
    datasets.getLateheterFusionDataset = factory
    opv2v_basedataset._v2b_twin = True
