#!/usr/bin/env python3
"""Training entry point.

Reads an experiment spec from ``configs/``, materialises the mixed
dataset view, renders the full OpenCOOD config into
``outputs/<run-name>/config.yaml`` and hands the run to
``v2b/train/opencood_train.py``.

The actual loop runs in **another interpreter**:
the fork's venv (``third_party/HEAL/.venv`` for ``fork: heal``)
owns torch, spconv and the CUDA ops
built for this GPU, and those are deliberately not v2b dependencies
(the detector stack pins its own CUDA build, and mmcv/spconv have no
Python 3.13 wheels). This launcher stays pure stdlib so it type-checks
and runs in the project environment.

    uv run python v2b/scripts/v2b-main.py \
        --config configs/v2xreal/ours-fcooper.yaml \
        --wandb-project v2b \
        --wandb-name v2xreal-ours-fcooper

``--wandb-project`` and ``--wandb-name`` are mandatory: every run is logged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.data.mixture import build_view  # noqa: E402
from v2b.train.heal_data import write_assignment  # noqa: E402
from v2b.train.hypes import (  # noqa: E402
    ExperimentSpec,
    fork_dir,
    render,
)

TRAINER = REPO / "v2b" / "train" / "opencood_train.py"
VIEWS = REPO / "resources" / "data" / "generated" / "mixtures"


def fork_python(fork: str) -> Path:
    """The fork's venv interpreter (torch, spconv, CUDA ops)."""
    return fork_dir(fork) / ".venv" / "bin" / "python"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/<name>.yaml")
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--wandb-name", required=True)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--ap-samples", type=int, default=300)
    ap.add_argument(
        "--ddp",
        type=int,
        default=0,
        help="processes (GPUs) for torchrun; 0 = one process, no DDP",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="score the spec's init_from (or the run's bestval) on its "
        "val sources, no training; use --ap-samples -1 for every frame",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="render the config and the views, then stop",
    )
    # evaluation-only settings of the robustness harness
    # (v2b/scripts/v2b-robustness.py), passed through to the trainer
    ap.add_argument(
        "--eval-ckpt",
        default=None,
        help="eval only: score this checkpoint directory instead of the "
        "run's own or the spec's init_from",
    )
    ap.add_argument(
        "--eval-file",
        default=None,
        help="eval only: score this weight file (a run's net_latest.pth) "
        "instead of the bestval checkpoint of --eval-ckpt",
    )
    ap.add_argument(
        "--eval-out",
        default=None,
        help="eval only: write the metrics json here (default "
        "outputs/<run>/eval.json)",
    )
    ap.add_argument(
        "--pose-noise",
        nargs=2,
        type=float,
        default=None,
        metavar=("SIGMA_T", "SIGMA_R"),
        help="eval only: Gaussian noise on the other agents' pose, "
        "metres and degrees",
    )
    ap.add_argument(
        "--latency",
        type=int,
        default=0,
        help="eval only: the other agents' frame lags the ego by k frames",
    )
    ap.add_argument("--noise-seed", type=int, default=None)
    ap.add_argument(
        "--ego-only",
        action="store_true",
        help="eval only: a fusion model on the ego's map alone, the "
        "other agents dropped before the fusion module",
    )
    ap.add_argument(
        "--dump-detections",
        action="store_true",
        help="eval only: keep every decoded and labelled box in "
        "outputs/<run>/detections.npz",
    )
    ap.add_argument(
        "--dump-agent-detections",
        action="store_true",
        help="eval only, late fusion: keep every agent's own boxes in "
        "outputs/<run>/agent_detections.npz",
    )
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="eval only: fixed seeds and deterministic cuDNN",
    )
    return ap


def prepare(
    spec: ExperimentSpec, run_dir: Path, eval_only: bool = False
) -> Path:
    """Build the mixed views and write the rendered OpenCOOD config.

    With ``train.holdout`` the val view is the held-out share of the
    train scenarios (it selects the checkpoint) and an eval-only run
    scores ``sources.test`` instead, so no reported split ever picks
    the checkpoint.
    """
    train_sources = spec.train_sources
    train_view = build_view(
        train_sources, VIEWS / f"{spec.name}-train", holdout=spec.holdout
    )
    if eval_only and spec.test_sources:
        val_view = build_view(
            spec.test_sources,
            VIEWS / f"{spec.name}-test",
        )
    elif spec.holdout is not None:
        val_view = build_view(
            train_sources,
            VIEWS / f"{spec.name}-val",
            holdout=spec.holdout,
            part="holdout",
        )
    else:
        val_view = build_view(
            spec.val_sources,
            VIEWS / f"{spec.name}-val",
        )
    for label, view in (("train", train_view), ("val", val_view)):
        detail = ", ".join(f"{k} {v}" for k, v in view.per_source.items())
        print(f"{label:<6}view : {view.n_scenes} scenarios ({detail})")

    nx, ny = spec.grid_cells
    print(f"grid       : {nx} x {ny} pillars at {spec.voxel_size[0]} m")

    hypes = render(spec, train_view.root, val_view.root)
    run_dir.mkdir(parents=True, exist_ok=True)
    if spec.fork == "heal" and "heter" in hypes:
        # HEAL indexes its modality table by scenario name; the view's
        # prefixed names need a table of their own (one modality each)
        ego = str(hypes["heter"]["ego_modality"]).split("&")[0]
        hypes["heter"]["assignment_path"] = str(
            write_assignment(
                [train_view.root, val_view.root],
                run_dir / "modality_assignment.json",
                ego,
            )
        )
    out = run_dir / "config.yaml"
    out.write_text(yaml.safe_dump(hypes, sort_keys=False))
    return out


def main() -> int:
    args = build_parser().parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    run_dir = REPO / "outputs" / args.wandb_name
    print(f"experiment : {spec.name} ({spec.base})")
    if spec.notes:
        print(f"notes      : {spec.notes}")
    hypes_path = prepare(spec, run_dir, args.eval_only)
    print(f"config     : {hypes_path}")

    if args.dry_run:
        print("dry run; not launching")
        return 0
    python = fork_python(spec.fork)
    if not python.exists():
        raise SystemExit(
            f"missing {python} -- the training environment is the "
            f"{spec.fork} fork's venv (torch + spconv + CUDA ops)"
        )

    cmd = [str(python)]
    if args.ddp > 1:
        # two jobs on one node must not share torchrun's default
        # rendezvous port (29500); derive it from the job id
        port = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 10000
        cmd += [
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.ddp}",
            f"--master_port={port}",
        ]
    cmd += [
        str(TRAINER),
        "--hypes",
        str(hypes_path),
        "--run-dir",
        str(run_dir),
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        args.wandb_name,
        "--num-workers",
        str(args.num_workers),
        "--ap-samples",
        str(args.ap_samples),
    ]
    if args.wandb_entity:
        cmd += ["--wandb-entity", args.wandb_entity]
    if spec.init_from:
        cmd += ["--init-from", str(REPO / spec.init_from)]
    if args.resume:
        cmd.append("--resume")
    if args.eval_only:
        cmd.append("--eval-only")
    if args.eval_ckpt:
        cmd += ["--eval-ckpt", str(REPO / args.eval_ckpt)]
    if args.eval_file:
        cmd += ["--eval-file", str(REPO / args.eval_file)]
    if args.eval_out:
        cmd += ["--eval-out", str(REPO / args.eval_out)]
    if args.pose_noise is not None:
        cmd += ["--pose-noise", *(str(s) for s in args.pose_noise)]
    if args.latency:
        cmd += ["--latency", str(args.latency)]
    if args.noise_seed is not None:
        cmd += ["--noise-seed", str(args.noise_seed)]
    if args.ego_only:
        cmd.append("--ego-only")
    if args.val_batches is not None:
        cmd += ["--val-batches", str(args.val_batches)]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.dump_detections:
        cmd.append("--dump-detections")
    if args.dump_agent_detections:
        cmd.append("--dump-agent-detections")

    print(f"launching  : {' '.join(cmd)}", flush=True)
    env = {**os.environ, "V2B_FORK": spec.fork}
    return subprocess.call(cmd, cwd=str(REPO), env=env)


if __name__ == "__main__":
    sys.exit(main())
