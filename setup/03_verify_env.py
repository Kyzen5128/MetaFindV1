#!/usr/bin/env python3
"""Environment verification -- graph node ``n01_env_bootstrap``.

Implements L1-ENV-ULIP, L1-ENV-EGNN, L1-ENV-DET plus the FPS replacement checks
from docs/graph/validation_plan.yaml.

Every assertion here is about CONTENT, never about a file existing or a command
exiting 0 (rule V3). Where a check could pass vacuously, it carries a companion
assertion that would fail if the check were not really running (rule V2).

Usage::

    conda activate metafind
    python setup/03_verify_env.py            # skips the ~10GB ViT-bigG-14 download
    python setup/03_verify_env.py --full     # includes it
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ULIP_REPO = REPO / "metafind" / "vendor" / "ulip"
EGNN_REPO = REPO / "metafind" / "vendor" / "egnn"  # vendored upstream, for the drift check

# --------------------------------------------------------------------------- cache
# Point every model cache at the data volume BEFORE importing torch, open_clip or
# transformers -- they read these variables at import time, so setting them later
# has no effect.
#
# This is not a convenience. `/` has ~100GB free while the data volume has ~780GB,
# and ViT-bigG-14 alone is ~10GB. Relying on the user's shell to export HF_HOME
# means the script silently fills the wrong disk when run from a non-login shell,
# which is exactly what happened the first time this ran.
_CACHE = REPO / "data" / "cache"
if _CACHE.parent.exists():
    os.environ.setdefault("HF_HOME", str(_CACHE / "hf"))
    os.environ.setdefault("TORCH_HOME", str(_CACHE / "torch"))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator that records pass/fail instead of aborting on the first failure."""

    def deco(fn):
        def wrapped():
            try:
                detail = fn() or ""
                RESULTS.append((name, True, detail))
            except Exception as exc:  # noqa: BLE001 -- we want the full picture
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                if os.environ.get("METAFIND_VERBOSE"):
                    traceback.print_exc()

        return wrapped

    return deco


# --------------------------------------------------------------------------- L1-ENV-*


@check("L1-ENV-TORCH  torch + CUDA available")
def t_torch():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert total > 20, f"expected a >=24GB card, got {total:.1f}GB"
    return f"torch {torch.__version__}, {name}, {total:.1f}GB"


@check("L1-ENV-PATCH  compat shims install cleanly")
def t_patch():
    sys.path.insert(0, str(REPO))
    from metafind.compat import ulip_patch

    ulip_patch.apply(patch_fps=False)
    from torch._six import string_classes  # noqa: PLC0415

    assert string_classes is str
    import torch  # noqa: PLC0415

    assert torch._six.string_classes is str, "submodule not bound on parent package"
    import pointnet2_ops  # noqa: PLC0415
    import knn_cuda  # noqa: PLC0415

    # Companion assertion (V2): the stub must RAISE, not silently return something.
    try:
        pointnet2_ops.pointnet2_utils.furthest_point_sample(None, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("pointnet2 stub returned instead of raising -- it would hide a live call")
    return "torch._six / pointnet2_ops / knn_cuda shimmed"


@check("L1-FPS        pure-torch FPS matches the greedy algorithm")
def t_fps():
    import torch

    sys.path.insert(0, str(REPO))
    from metafind.compat.ulip_patch import farthest_point_sample_idx, fps

    torch.manual_seed(0)
    xyz = torch.rand(2, 1000, 3)

    idx = farthest_point_sample_idx(xyz, 64)
    assert idx.shape == (2, 64), f"bad shape {tuple(idx.shape)}"
    assert (idx[:, 0] == 0).all(), "must seed at index 0 to stay deterministic"
    # No duplicate centres -- the classic bug when the distance update is wrong.
    for b in range(idx.shape[0]):
        assert len(set(idx[b].tolist())) == 64, "duplicate centres selected"

    # Determinism: same input twice -> identical indices.
    assert torch.equal(idx, farthest_point_sample_idx(xyz, 64)), "FPS is not deterministic"

    # Coverage: FPS should spread out. Its minimum pairwise distance must beat
    # random sampling comfortably -- this is what distinguishes it from `xyz[:, :64]`.
    def min_pairwise(points):
        d = torch.cdist(points, points)
        d.fill_diagonal_(float("inf"))
        return d.min().item()

    picked = fps(xyz, 64)[0]
    random_pick = xyz[0][torch.randperm(1000)[:64]]
    assert min_pairwise(picked) > min_pairwise(random_pick), "FPS no better spread than random"

    assert fps(xyz, 64).shape == (2, 64, 3)
    return f"64 centres, min pairwise {min_pairwise(picked):.4f} vs random {min_pairwise(random_pick):.4f}"


@check("L1-ENV-EGNN   EGNN forward shapes (in_edge_nf=64, F8 projection)")
def t_egnn():
    import torch

    sys.path.insert(0, str(REPO))
    from metafind.vendor.egnn_clean import EGNN

    n, d_h, d_e = 20, 1280, 64
    egnn = EGNN(in_node_nf=d_h, hidden_nf=128, out_node_nf=d_h, in_edge_nf=d_e, n_layers=4)
    h = torch.randn(n, d_h)
    x = torch.randn(n, 3)
    rows, cols = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    edges = [torch.tensor(rows), torch.tensor(cols)]
    edge_attr = torch.randn(len(rows), d_e)

    h_out, x_out = egnn(h, x, edges, edge_attr)
    assert h_out.shape == (n, d_h), f"h {tuple(h_out.shape)} != {(n, d_h)}"
    assert x_out.shape == (n, 3), f"x {tuple(x_out.shape)} != {(n, 3)}"
    return f"h {tuple(h_out.shape)}, x {tuple(x_out.shape)}"


@check("L2-EQUIVAR    SE(3) equivariance of stock EGNN (smoke)")
def t_equivar():
    """Early smoke version of L2-EQUIVAR.

    Runs against the *stock* EGNN with h independent of x -- i.e. the formulation
    Appendix C actually requires. If this fails, the problem is our wiring, not
    MetaFind. The full 100x100 probe lives in graph node n11.
    """
    import torch

    sys.path.insert(0, str(REPO))
    from metafind.vendor.egnn_clean import EGNN

    torch.manual_seed(0)
    n, d_h, d_e = 12, 64, 16
    egnn = EGNN(in_node_nf=d_h, hidden_nf=64, out_node_nf=d_h, in_edge_nf=d_e, n_layers=3).double()
    h = torch.randn(n, d_h, dtype=torch.float64)
    x = torch.randn(n, 3, dtype=torch.float64)
    rows, cols = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    edges = [torch.tensor(rows), torch.tensor(cols)]
    edge_attr = torch.randn(len(rows), d_e, dtype=torch.float64)

    # Random rotation via QR, and a deliberately large translation: the paper's
    # motivation is unnormalised open-world coordinates.
    a = torch.randn(3, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    t = torch.randn(3, dtype=torch.float64) * 100.0

    h1, x1 = egnn(h, x, edges, edge_attr)
    h2, x2 = egnn(h, x @ q.T + t, edges, edge_attr)

    coord_err = (x2 - (x1 @ q.T + t)).abs().max().item()
    h_err = (h2 - h1).abs().max().item()
    assert coord_err < 1e-6, f"coord equivariance broken: {coord_err:.3e}"
    assert h_err < 1e-9, f"h invariance broken: {h_err:.3e}"

    # Companion assertion (V2): the check must be able to FAIL. Feeding a
    # rotation-dependent h must break invariance -- this is exactly finding F1.
    h_bad = torch.cat([x, h[:, 3:]], dim=1)
    h_bad_rot = torch.cat([x @ q.T + t, h[:, 3:]], dim=1)
    hb1, _ = egnn(h_bad, x, edges, edge_attr)
    hb2, _ = egnn(h_bad_rot, x @ q.T + t, edges, edge_attr)
    assert (hb2 - hb1).abs().max().item() > 1e-3, "injection did not break invariance -- test is vacuous"

    return f"coord err {coord_err:.2e}, h err {h_err:.2e}; Concat(x,t) injection breaks it as expected (F1)"


@check("L1-VENDOR     vendored EGNN matches upstream, no `models` collision")
def t_vendor():
    """Guards the fix for the ULIP/egnn top-level `models` package collision.

    Both clones ship a package called `models`. ULIP's lacks __init__.py
    (namespace package) while egnn's has one (regular package), so Python
    resolves `models` to egnn's regardless of sys.path order and
    `models.pointbert` becomes unreachable. We vendor egnn's single
    self-contained file so `models` belongs to ULIP alone.
    """
    vendored = (REPO / "metafind" / "vendor" / "egnn_clean.py").read_text()
    upstream = (EGNN_REPO / "models" / "egnn_clean" / "egnn_clean.py").read_text()
    assert vendored.endswith(upstream), "vendored EGNN has drifted from upstream"

    # Companion assertion (V2): prove the drift check can fail.
    assert not vendored.endswith(upstream + "\n# drift"), "drift check is vacuous"

    # The vendored module must not pull in a top-level `models` package at all.
    import metafind.vendor.egnn_clean as ve  # noqa: PLC0415

    assert ve.__name__.startswith("metafind."), f"unexpected module name {ve.__name__}"
    lic = REPO / "metafind" / "vendor" / "LICENSE.egnn"
    assert "MIT" in lic.read_text(), "MIT licence text missing alongside vendored code"
    return f"{len(upstream.splitlines())} lines vendored verbatim, MIT licence retained"


@check("L1-ENV-DET    determinism capability recorded")
def t_determinism():
    import torch

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        status = "deterministic algorithms enabled (warn_only)"
    except Exception as exc:  # noqa: BLE001
        status = f"NOT available: {exc}"
    finally:
        torch.use_deterministic_algorithms(False)
    # scatter_add_ on CUDA uses atomics -> nondeterminism source NS-5.
    return f"{status}; NS-5 (scatter_add_ atomics) stands regardless"


@check("L1-ENV-STORAGE  data symlink points at /mnt/data1")
def t_storage():
    link = REPO / "data"
    assert link.exists(), "run setup/01_storage.sh first"
    target = link.resolve()
    assert str(target).startswith("/mnt/data1"), f"data -> {target}, expected /mnt/data1/..."
    import shutil

    free_gb = shutil.disk_usage(target).free / 1024**3
    assert free_gb > 150, f"only {free_gb:.0f}GB free; need ~90GB plus headroom"

    # Model caches must resolve onto the data volume, not `/`. Asserting the
    # variable is merely SET would be verifying the wrong thing (rule V2) -- what
    # matters is where it actually points.
    for var in ("HF_HOME", "TORCH_HOME"):
        val = os.environ.get(var)
        assert val, f"{var} is unset; a download would land on /"
        assert Path(val).resolve().is_relative_to(target), (
            f"{var}={val} resolves outside {target}; ViT-bigG-14 (~10GB) would fill /"
        )
    return f"{target}, {free_gb:.0f}GB free, HF_HOME+TORCH_HOME on data volume"


@check("L1-ENV-ULIP   ULIP-2 model builds, pc_projection is 1280-d")
def t_ulip():
    import torch

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ULIP_REPO))
    from metafind.compat import ulip_patch

    ulip_patch.apply(patch_fps=True)

    from models.ULIP_models import ULIP2_PointBERT_Colored

    args = argparse.Namespace(npoints=10000)
    # ULIP hardcodes './models/pointbert/...yaml', so it must be built from the
    # repo root. Companion assertion below proves we restored the CWD.
    cwd_before = os.getcwd()
    with ulip_patch.ulip_cwd():
        model = ULIP2_PointBERT_Colored(args)
    assert os.getcwd() == cwd_before, "ulip_cwd leaked the working directory"

    dim = model.pc_projection.shape[1]
    assert dim == 1280, f"expected ULIP-2 embed dim 1280, got {dim} (512 would mean ULIP-1)"
    assert model.pc_projection.shape[0] == 768, "PointBERT feature dim should be 768"

    # Prove misc.fps really was replaced, not merely importable.
    from models.pointbert import misc

    assert misc.fps.__module__.startswith("metafind"), "misc.fps was not patched"

    # The collision fix must hold: `models` belongs to ULIP, not egnn.
    import models  # noqa: PLC0415

    resolved = [str(pth) for pth in models.__path__]
    expected = str(ULIP_REPO / "models")
    assert any(pth == expected for pth in resolved), (
        f"`models` resolved to {resolved}, expected the vendored {expected}"
    )
    return f"pc_projection {tuple(model.pc_projection.shape)}, misc.fps patched, models -> ULIP"


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include the ~10GB ViT-bigG-14 download")
    args = ap.parse_args()

    for fn in (t_torch, t_patch, t_fps, t_vendor, t_egnn, t_equivar, t_determinism, t_storage):
        fn()
    if args.full:
        t_ulip()

    width = max(len(n) for n, _, _ in RESULTS)
    print("\nn01_env_bootstrap -- verification")
    print("=" * (width + 60))
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    print("=" * (width + 60))

    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        print("Per graph spec: n01 failures are DETERMINISTIC_INPUT -- do NOT retry, fix versions.")
        return 2
    if not args.full:
        print("\nNote: L1-ENV-ULIP skipped. Re-run with --full once you are ready to")
        print("      download ViT-bigG-14 (~10GB into data/cache/hf).")
    print("\nn01_env_bootstrap postcondition satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
