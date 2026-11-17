#!/usr/bin/env python3
"""Re-score a stored cooperative-evaluation run, no data and no GPU.

``v2b-coop-eval.py`` writes every frame's labels and detections to
``outputs/coop-eval/<run>/<labels>/predictions.npz``; this reads them
back and recomputes ``results.json`` with ``v2b.metrics.coop_eval``,
which is numpy only, so it runs in the project venv::

    .venv/bin/python v2b/scripts/v2b-eval.py \\
        --run outputs/coop-eval/<run>/annotated

``--compare`` takes a second run directory and prints the largest
difference of any AP between the two result tables (the determinism
check: two inferences of one checkpoint must agree to four decimals).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2b.metrics.coop_eval import load_predictions, score_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", required=True, help="outputs/coop-eval/<run>/<labels>"
    )
    ap.add_argument("--compare", default=None, help="a second run directory")
    ap.add_argument("--min-points", type=int, default=10)
    ap.add_argument("--single-floor", type=int, default=1)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write", action="store_true", help="overwrite the run's results.json"
    )
    return ap.parse_args()


def flat_aps(table: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """``{stratum/class/ap@t: value}`` of a results ``ap`` table."""
    out: dict[str, float] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            out.update(flat_aps(value, f"{prefix}{key}/"))
        elif isinstance(value, float) and "ap@" in key:
            out[f"{prefix}{key}"] = value
    return out


def rescore(run: Path, args: argparse.Namespace) -> dict[str, Any]:
    labels, dets, meta = load_predictions(run / "predictions.npz")
    return score_run(
        labels,
        dets,
        meta["lidar_range"],
        args.min_points,
        args.single_floor,
        args.draws,
        args.seed,
    )


def main() -> int:
    args = parse_args()
    run = Path(args.run)
    results = rescore(run, args)
    row = results["ap"]["all"]
    print(
        f"{run}: {results['frames']} frames; AP@0.5 "
        + ", ".join(
            f"{c} {row[c]['ap@0.5']:.4f}"
            for c in ("car", "bicycle", "pedestrian")
        )
    )
    if args.write:
        stored = json.loads((run / "results.json").read_text())
        stored.update(results)
        (run / "results.json").write_text(json.dumps(stored, indent=1))
    if args.compare:
        other = rescore(Path(args.compare), args)
        a, b = flat_aps(results["ap"]), flat_aps(other["ap"])
        diffs = {
            k: abs(a[k] - b[k])
            for k in a
            if k in b and a[k] == a[k] and b[k] == b[k]
        }
        worst = max(diffs, key=lambda k: diffs[k]) if diffs else ""
        print(
            f"compared {len(diffs)} AP values with {args.compare}: "
            f"max |diff| = {diffs.get(worst, 0.0):.6f} ({worst})"
        )
        return 0 if diffs.get(worst, 0.0) < 5e-5 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
