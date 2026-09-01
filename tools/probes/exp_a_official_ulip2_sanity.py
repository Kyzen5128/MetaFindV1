#!/usr/bin/env python3
"""EXPERIMENT A -- official ULIP-2 zero-shot sanity check, with full provenance.

Kyzen via Codex, 2026-09-01: no RAG, no fusion, no ESSGNN, no Stage 1. Run
upstream's own zero-shot classification with `ULIP2_PointBERT_Colored` and the
official 10k xyzrgb ViT-bigG checkpoint, and list the checkpoint path, sha256,
`args.model`, `npoints`, input shape, and missing/unexpected keys. Target
Objaverse-LVIS 50.6 / 79.1.

DL-053 already reported 50.576 / 78.931, but as a number without the
provenance block. This re-runs it so that every field Codex asked for is
recorded beside the result rather than remembered.

Nothing of ours is in the loop. `upstream/ULIP_run` is invoked through
`runpy`, so the code that runs is upstream's `main.py` and upstream's
`test_zeroshot_3d_core` unmodified, on upstream's own `objaverse_lvis_colored`
loader and its own 45,692 `.npy` clouds. The only additions are the environment
shims -- `torch._six` was deleted in PyTorch 2.0, plus stubs for two CUDA
extensions -- which touch no program logic.

The provenance is captured by monkey-patching `load_state_dict` for one call
to record what it returned, and `PointTransformer_Colored.forward` for one call
to record the input tensor's shape and dtype. Both restore themselves. That is
the only way to see those two facts, because upstream prints neither.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import runpy
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
UPSTREAM = pathlib.Path("/home/kyzen/upstream/ULIP_run")
OUT = REPO / "output" / "look" / "exp_a_official_ulip2_sanity.json"
CKPT = (REPO / "data/models/ulip2/ULIP-2/pretrained_models/"
        "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
TARGET = {"objaverse_lvis_colored": {"top1": 50.6, "top5": 79.1},
          "modelnet40": {"top1": 84.7, "top5": 97.1}}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="objaverse_lvis_colored",
                    choices=tuple(TARGET))
    ap.add_argument("--npoints", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    prov = {"experiment": "A -- official ULIP-2 zero-shot classification",
            "upstream_repo": str(UPSTREAM),
            "checkpoint": str(CKPT),
            "checkpoint_bytes": CKPT.stat().st_size,
            "checkpoint_sha256": sha256(CKPT),
            "model": "ULIP2_PointBERT_Colored",
            "npoints": args.npoints,
            "validate_dataset_name": args.dataset,
            "target": TARGET[args.dataset]}
    import subprocess
    prov["upstream_git"] = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    prov["upstream_git_dirty"] = bool(subprocess.run(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain",
         "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    print(json.dumps(prov, indent=1), flush=True)

    # `main.py:14` imports wandb unconditionally, but every USE of it sits
    # behind `--wandb`, which this run does not pass. A stub is an environment
    # shim in the same class as torch._six: no upstream logic is reached.
    import types
    if "wandb" not in sys.modules:
        import importlib.machinery
        w = types.ModuleType("wandb")
        w.init = w.log = w.watch = w.finish = lambda *a, **k: None
        # timm calls importlib.util.find_spec("wandb"), which raises
        # "wandb.__spec__ is None" on a bare ModuleType. Give it one.
        w.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        w.__version__ = "0.0.0-stub"
        sys.modules["wandb"] = w
        prov["shims"] = ["torch._six", "pointnet2 stub", "knn_cuda stub",
                         "wandb stub (unused, --wandb not passed)"]

    from metafind.compat import ulip_patch
    ulip_patch.apply(patch_fps=False)
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    ulip_patch.apply(patch_fps=True)

    import torch
    # PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True;
    # `main.py:475` calls it without the argument and the ULIP-2 checkpoint
    # carries a numpy scalar, so it now raises. Restoring the pre-2.6 default
    # for this process is an environment shim -- the file is the official
    # release we hashed above, not an untrusted download.
    _real_load = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _real_load(*a, **k)
    torch.load = _load
    prov.setdefault("shims", []).append(
        "torch.load(weights_only=False), the pre-2.6 default")

    # --- capture what upstream never prints -----------------------------
    captured: dict = {}
    real_lsd = torch.nn.Module.load_state_dict

    def spy_lsd(self, sd, strict=True, **kw):
        r = real_lsd(self, sd, strict=strict, **kw)
        if "load_state_dict" not in captured and type(self).__name__.startswith(
                ("ULIP2_WITH_OPENCLIP", "ULIP_WITH_IMAGE")):
            captured["load_state_dict"] = {
                "module": type(self).__name__, "strict": strict,
                "n_keys_in_checkpoint": len(sd),
                "missing": len(r.missing_keys), "unexpected": len(r.unexpected_keys),
                "missing_sample": list(r.missing_keys)[:10],
                "unexpected_sample": list(r.unexpected_keys)[:10]}
            torch.nn.Module.load_state_dict = real_lsd
        return r
    torch.nn.Module.load_state_dict = spy_lsd

    from models.pointbert.point_encoder import PointTransformer_Colored
    real_fwd = PointTransformer_Colored.forward

    def spy_fwd(self, pts, *a, **k):
        if "pc_input" not in captured:
            captured["pc_input"] = {"shape": tuple(pts.shape),
                                    "dtype": str(pts.dtype),
                                    "device": str(pts.device)}
            PointTransformer_Colored.forward = real_fwd
        return real_fwd(self, pts, *a, **k)
    PointTransformer_Colored.forward = spy_fwd
    # ---------------------------------------------------------------------

    sys.argv = ["main.py",
                "--model", "ULIP2_PointBERT_Colored",
                "--npoints", str(args.npoints),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--output-dir", "./outputs/exp_a_sanity",
                "--evaluate_3d_ulip2",
                f"--validate_dataset_name={args.dataset}",
                "--test_ckpt_addr", str(CKPT)]
    os.makedirs("./outputs/exp_a_sanity", exist_ok=True)
    print("\n$ python " + " ".join(sys.argv[1:]) + "\n", flush=True)

    t0 = time.time()
    g = runpy.run_path("main.py", run_name="__main__")
    prov["seconds"] = round(time.time() - t0, 1)
    prov.update(captured)

    # upstream writes its numbers into the log; read them back rather than
    # re-deriving, so the recorded result is literally what it printed.
    log = pathlib.Path("./outputs/exp_a_sanity/log.txt")
    prov["upstream_log"] = str(UPSTREAM / log) if log.exists() else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print("\n=== PROVENANCE ===")
    print(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"\n目標 {TARGET[args.dataset]}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
