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

# IMPLEMENTS-NODE: n01_env_bootstrap
# writes channel: run_progress -- this script is invoked by the operator rather
# than by a scheduler, so its record is the pass/fail summary it prints and the
# exit code, not a JSONL row. Named here so the registry and the source agree.

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ULIP_REPO = REPO / "metafind" / "vendor" / "ulip"

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

# The exact environment F24 and F25 were measured on. Ranges are not pins: a
# rebuild that takes ai2thor 5.1 renders fine, loads 12,000 houses, and reports
# a different asset database, with nothing saying the experiment changed.
PINNED_AI2THOR = "5.0.0"
PINNED_THOR_BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
PINNED_PROCTHOR_REV = "439193522244720b86d8c81cde2e51e3a4d150cf"

RESULTS: list[tuple[str, bool, str]] = []


# Decorated checks register themselves here. main() used to call them from a
# hand-written tuple, which meant a check could be written, decorated, and never
# run -- exactly what happened when L1-ENV-THOR was added and silently skipped.
# A test that does not run is worse than a missing one: the report says PASS for
# everything it did run and nothing says the list got shorter.
REGISTRY: list[tuple] = []


def check(name: str, full_only: bool = False):
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

        REGISTRY.append((name, wrapped, full_only))
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


@check("L1-VENDOR     vendored EGNN is unmodified, no `models` collision")
def t_vendor():
    """Guards the fix for the ULIP/egnn top-level `models` package collision.

    Both upstream repos ship a package called `models`. ULIP's lacks
    __init__.py (namespace package) while egnn's has one (regular package), so
    Python resolves `models` to egnn's regardless of sys.path order and
    `models.pointbert` becomes unreachable. Only the single self-contained EGNN
    file we actually use is vendored, so `models` belongs to ULIP alone.

    Integrity is pinned by hashing the body against the recorded upstream digest,
    rather than by keeping a second copy of the repo purely to diff against.
    """
    import hashlib  # noqa: PLC0415

    import metafind.vendor.egnn_clean as ve  # noqa: PLC0415

    path = REPO / "metafind" / "vendor" / "egnn_clean.py"
    body = path.read_text().split('"""\n', 2)[2]
    digest = hashlib.sha256(body.encode()).hexdigest()
    assert digest == ve.UPSTREAM_SHA256, (
        f"vendored EGNN body has been modified: {digest} != {ve.UPSTREAM_SHA256}"
    )

    # Companion assertion (V2): prove the digest check can fail.
    assert hashlib.sha256((body + "\n# drift").encode()).hexdigest() != ve.UPSTREAM_SHA256

    assert ve.__name__.startswith("metafind."), f"unexpected module name {ve.__name__}"
    lic = REPO / "metafind" / "vendor" / "LICENSE.egnn"
    assert "MIT" in lic.read_text(), "MIT licence text missing alongside vendored code"
    return f"{len(body.splitlines())} lines, sha256 {digest[:12]}, MIT licence retained"


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


@check("L1-ENV-STORAGE  data symlink is valid")
def t_storage():
    link = REPO / "data"
    assert link.exists(), "run setup/01_storage.sh first"
    target = link.resolve()
    assert link.is_symlink(), f"{link} is not a symlink"
    assert target.exists(), f"data target does not exist: {target}"
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


@check("L1-ENV-ULIP   ULIP-2 model builds, pc_projection is 1280-d", full_only=True)
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


@check("L1-ENV-THOR   AI2-THOR renders headless on the pinned build")
def t_ai2thor():
    """Importing ai2thor proves nothing -- the Unity build has to actually run.

    n07b, and therefore the whole Stage 2 target-modality branch, depends on
    CloudRendering working without an X server. That fails for reasons import
    cannot see: a missing build, a GPU the runtime will not take, a driver
    mismatch. It is cheap to find out here and expensive to find out inside a
    1,467-asset render loop.

    The build hash is asserted because F24 and F25's measurements are pinned to
    it: 1,934 asset-database entries, and the isolated-render recipe. A
    different build could change both without changing any number we print.
    """
    import ai2thor
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    assert ai2thor.__version__ == PINNED_AI2THOR, (
        f"expected ai2thor {PINNED_AI2THOR}, got {ai2thor.__version__}. "
        "F24 and F25 are pinned to this build; a different one may render fine and "
        "still report a different asset database."
    )

    thor_home = Path.home() / ".ai2thor"
    assert thor_home.is_symlink(), (
        f"{thor_home} is not a symlink; the ~800MB Unity build would land on /"
    )
    builds = [d.name for d in (thor_home / "releases").glob("thor-CloudRendering-*")
              if d.is_dir()]
    assert any(b.endswith(PINNED_THOR_BUILD) for b in builds), (
        f"expected CloudRendering build {PINNED_THOR_BUILD}, found {builds}"
    )

    # MEASURED: GetAssetDatabase returns nothing on a hand-authored iTHOR scene
    # such as FloorPlan1 -- it belongs to the PROCEDURAL API, so the scene has to
    # be a ProcTHOR house. The first version of this check used FloorPlan1 and
    # failed for that reason, which also confirms F25's 1,934 was measured in the
    # right context.
    import json

    house_path = Path(os.environ.get("METAFIND_DATA", REPO / "data")) / \
        "datasets" / "procthor-10k" / "train.jsonl"
    if not house_path.exists():
        return f"{ai2thor.__version__}, skipped the render probe (no procthor-10k yet)"
    with house_path.open() as fh:
        house = json.loads(fh.readline())

    c = Controller(scene=house, platform=CloudRendering,
                   width=64, height=64, quality="Low")
    try:
        frame = c.last_event.frame
        assert frame is not None and frame.shape == (64, 64, 3), (
            f"CloudRendering returned {None if frame is None else frame.shape}"
        )
        db = c.step(action="GetAssetDatabase").metadata["actionReturn"]
        assert db, "GetAssetDatabase returned nothing"
        # Reported, not asserted: U-08c's whole point is that this count is
        # derived per build rather than fixed. F25 recorded 1934 on
        # thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917.
        return f"{ai2thor.__version__}, {len(db)} assets in the database"
    finally:
        c.stop()


@check("L1-ENV-PRIOR  procthor-10k loads at the pinned revision")
def t_prior():
    """Split sizes alone are a weak pin -- a revision can change every house and
    still be 10k/1k/1k. `prior` clones a git repo, so the revision is checkable,
    and F25's 1,467 assets are a statement about THIS commit."""
    import prior

    ds = prior.load_dataset("procthor-10k")
    sizes = {k: len(ds[k]) for k in ("train", "val", "test")}
    assert sizes == {"train": 10000, "val": 1000, "test": 1000}, sizes

    cache = Path.home() / ".prior" / "datasets" / "allenai" / "procthor-10k" / "cache"
    assert cache.exists(), f"{cache} missing; cannot verify the dataset revision"
    rev = json.loads(cache.read_text())["main"]
    assert rev == PINNED_PROCTHOR_REV, (
        f"procthor-10k is at {rev}, expected {PINNED_PROCTHOR_REV}"
    )
    return f"{sum(sizes.values()):,} houses @ {rev[:12]}"



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include the ~10GB ViT-bigG-14 download")
    args = ap.parse_args()

    skipped = []
    for name, fn, full_only in REGISTRY:
        if full_only and not args.full:
            skipped.append(name)
            continue
        fn()

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
    if skipped:
        print(f"\nNote: skipped {', '.join(s.split()[0] for s in skipped)}. "
              "Re-run with --full once you are ready to")
        print("      download ViT-bigG-14 (~10GB into data/cache/hf).")
    print("\nn01_env_bootstrap postcondition satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
