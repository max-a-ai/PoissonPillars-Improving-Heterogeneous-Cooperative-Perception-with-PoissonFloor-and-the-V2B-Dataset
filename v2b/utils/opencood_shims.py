"""Make the vendored ``opencood`` importable in a modern environment.

``third_party/CodeFilling`` is a research fork pinned against an older
Python stack. Three of its module-level imports fail here, all on code
paths V2B never executes. Rather than installing an obsolete
dependency (or editing vendored code, which would drift from upstream),
each one is replaced by a stub that **raises on first use** -- so a
model that genuinely needs it fails loudly instead of quietly doing
something different.

``install()`` is idempotent and must be called before the first
``import opencood``.

===================  ======================================================
``mmcv.ops``         ``feature_alignnet`` imports ``DeformConv2dPack`` at
                     module level. The checkpoints we run configure no
                     aligner, so it is never constructed. Installing the
                     mm-stack would pin torch/CUDA back to what already
                     blocks ``coopdet3d`` on this GPU.
``pypcd``            ``pcd_utils`` imports the Python-2-era reader at
                     module level for ``read_pcd``, which only the
                     DAIR-V2X loader calls. OPV2V-family data goes
                     through ``pcd_to_np`` (open3d) instead.
``matplotlib.cm``    ``get_cmap`` was removed in matplotlib 3.9;
                     ``vis_utils`` calls it at import time.
===================  ======================================================
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODEFILLING = REPO / "third_party" / "CodeFilling"


def _raising_stub(name: str, hint: str) -> type:
    """A class that raises a pointed error the moment it is built."""

    class _Missing:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(f"{name} is not installed. {hint}")

    return _Missing


def _stub_mmcv() -> None:
    if "mmcv" in sys.modules:
        return
    mmcv = types.ModuleType("mmcv")
    ops = types.ModuleType("mmcv.ops")
    ops.DeformConv2dPack = _raising_stub(  # type: ignore[attr-defined]
        "mmcv.ops.DeformConv2dPack",
        "Install the mm-stack in third_party/CodeFilling, or pick a "
        "config whose aligner is not a deformable conv.",
    )
    mmcv.ops = ops  # type: ignore[attr-defined]
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.ops"] = ops


def _stub_pypcd() -> None:
    if "pypcd" in sys.modules:
        return
    pkg = types.ModuleType("pypcd")
    mod = types.ModuleType("pypcd.pypcd")
    mod.PointCloud = _raising_stub(  # type: ignore[attr-defined]
        "pypcd",
        "Only opencood's read_pcd (DAIR-V2X) needs it; OPV2V-family "
        "data reads through pcd_to_np. Use pypcd4 if you truly need it.",
    )
    pkg.pypcd = mod  # type: ignore[attr-defined]
    sys.modules["pypcd"] = pkg
    sys.modules["pypcd.pypcd"] = mod


def _patch_matplotlib_cm() -> None:
    import matplotlib.cm
    import matplotlib.pyplot as plt

    if not hasattr(matplotlib.cm, "get_cmap"):
        matplotlib.cm.get_cmap = plt.get_cmap  # type: ignore[attr-defined]


def install(repo: Path | None = None) -> None:
    """Apply every shim and put a vendored ``opencood`` on the path.

    ``repo`` selects the fork (``third_party/CodeFilling`` by default,
    ``third_party/HEAL`` is the other one that ships ``opencood``).
    """
    root = repo or CODEFILLING
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _stub_mmcv()
    _stub_pypcd()
    _patch_matplotlib_cm()
