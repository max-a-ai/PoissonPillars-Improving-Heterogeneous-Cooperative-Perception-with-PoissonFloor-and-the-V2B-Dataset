# PoissonPillars: Improving Heterogeneous Cooperative Perception with PoissonFloor and the V2B Dataset

Code for an anonymous submission: PoissonPillars (PoissonFloor, attention
pooling, floor-anchored pretraining, shared floors for cooperative agents),
the training and evaluation pipeline on top of HEAL, the configs of the
paper's tables, and a viewer for V2B with five sample frames.

<p align="center"><img src="assets/poisson-floor-steps.gif" width="720" alt="PoissonFloor, step by step"><br>
<em>How a PoissonFloor is built from one LiDAR sweep.</em></p>

<p align="center"><img src="assets/floor-mesh-orbit.gif" width="720" alt="The floor under both agents' sweeps"><br>
<em>The floor as a mesh under the sweeps of the car (blue) and the bicycle
(orange), points coloured by height above the floor.</em></p>

## Quick look: the V2B viewer

```bash
uv run app.py
```

Requires [uv](https://docs.astral.sh/uv/). The viewer opens at
<http://127.0.0.1:5005> with `samples/scene_04/`: five annotated keyframes
(0, 40, 60, 80, 100) of scene 4 in the nuScenes format, the bicycle within
10 m of the car. Each frame holds the fused sweep of both agents
(`float32` `x, y, z, intensity, agent`, 1 car, 2 bicycle), the 3D boxes
and poses of these five frames, and the scene's floor mesh. No camera
images are included. The viewer shows what the car, the bicycle or both
see, draws the floor and colours points by their height above it.

## Code

```
v2b/
├── data/ground.py          PoissonFloor: seeds, screened Poisson surface, re-base
├── data/export_*.py        V2X-Real, TUMTraf V2X, Waymo, nuScenes, V2B -> OPV2V layout
├── train/heal_data.py      attention pooling and loader hooks, installed into HEAL
├── train/pretrain.py       floor-anchored masked pretraining
├── train/floor_net.py      learned PoissonFloor (real-time floor)
├── train/fusion_*.py       single-agent trunk inside HEAL's fusion models
├── train/hypes.py          experiment spec -> HEAL/OpenCOOD config
├── train/opencood_train.py training and evaluation loop
├── metrics/                BEV AP and the cooperative V2B scoring
└── scripts/                entry points (below)
configs/                    one spec per row of the paper's tables
config-global.json          dataset registry (set nas_root to your data)
```

### Setup

HEAL runs in its own environment; the code here patches it at runtime and
does not modify it.

```bash
git clone https://github.com/yifanlu0227/HEAL third_party/HEAL
git -C third_party/HEAL checkout 96812ed
cp third_party-heal-pyproject.toml third_party/HEAL/pyproject.toml
(cd third_party/HEAL && uv sync && uv run python opencood/utils/setup.py build_ext --inplace)
```

### Data

Run the scripts below with HEAL's Python, `PY=third_party/HEAL/.venv/bin/python`.

1. Export each dataset to the OPV2V layout:
   `$PY v2b/scripts/v2b-export-opv2v.py <dataset> --root <raw> --split <split> --out <derived>/<name>/<split>`.
2. Re-base it on the PoissonFloor:
   `$PY v2b/scripts/v2b-derive-ground.py --src <derived>/<name>/<split> --out <derived>/<name>-rebase-poisson/<split> --mode rebase --surface poisson`.
3. For cooperative rows on one shared floor (V2B, TUMTraf V2X fused):
   `$PY v2b/scripts/v2b-shared-floor.py --raw <name> --ego <i> --other <j> --out <derived>/<name>-rebase-poisson-merged`.
4. Point `nas_root` in `config-global.json` at the folder holding the
   derived datasets and link them: `$PY v2b/scripts/dm_link.py`.

### Train and evaluate

Each config is one row; `init_from` names the run it starts from, so
train the pretraining and single-agent runs before the fused ones.

```bash
$PY v2b/scripts/v2b-main.py --config configs/v2xreal/ours-fcooper.yaml \
    --wandb-project poissonpillars --wandb-name v2xreal-ours-fcooper
$PY v2b/scripts/v2b-main.py --config configs/v2xreal/ours-fcooper.yaml \
    --wandb-project poissonpillars --wandb-name v2xreal-ours-fcooper-eval \
    --eval-only --eval-ckpt outputs/v2xreal-ours-fcooper
```

`--dry-run` renders the full HEAL config into `outputs/<name>/` without
training.

| Folder | Rows |
|---|---|
| `configs/v2xreal/` | Tab. 3: HEAL and PoissonPillars without fusion and with AttFuse, F-Cooper, V2X-ViT, Where2comm; pretraining |
| `configs/v2b/` | Tab. 4, V2B: vehicle, bicycle and fused rows |
| `configs/tumtraf/` | Tab. 4, TUMTraf V2X: vehicle, RSU and fused rows; pretraining |
| `configs/waymo/` | Waymo and nuScenes runs the V2B rows start from |

## License

Code: MIT (see `LICENSE`). Data: CC BY-NC-SA 4.0.
