#!/usr/bin/env python3
"""Per-modality embedding NORM, released encoder vs a Stage 1 checkpoint.

WHY. The parameter-free control and the raw/no-fusion derangement arms both
build their gallery as an UNNORMALISED mean of the three modality vectors, so
whichever modality has the largest norm dominates that mean. On the TRAINED
checkpoint `cos(pc, mean_of_three)` is 0.9427 against 0.7168 untrained
(`diag_trained_fusion_identity_e25_500w.json`), i.e. the trained mean is nearly
parallel to pc -- which would make every raw derangement row a statement about
NORMS rather than about modality dependence.

That explanation was an INFERENCE. This measures it. Released-encoder norms are
already OBSERVED DATA (`diag_ulip_fingerprint.json`: text 37.13, image 40.23,
pc 27.86); only the trained point encoder's are missing, because
`diag_protocol_e_ulip_fingerprint.item3` is skipped under `--state`.

text and image are the frozen OpenCLIP cache and cannot move; they are measured
anyway, because "cannot move" is the kind of claim this project checks.

DIAGNOSTIC. One JSON under `output/look/`. Modifies no canonical artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metafind import paths                                      # noqa: E402
from metafind.models.fusion import MODALITIES                   # noqa: E402

from diag_text_shortcut import collect_inputs                   # noqa: E402

OUT = REPO / "output" / "look" / "diag_modality_norms.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="init0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    trained = args.state != "init0"

    import torch

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    dev_val = json.loads(
        (paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    sha = None
    if trained:
        # BEFORE collect_inputs: the checkpoint restores the fine-tuned PointBERT.
        from metafind.train.stage1 import load_stage1_checkpoint
        model, loss_fn = build_model(encoding, training, hyper)
        model.to(args.device)
        load_stage1_checkpoint(backbone, model, loss_fn, Path(args.state))
        sha = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        print(f"loaded {args.state}  sha256 {sha[:16]}…", flush=True)

    order, T, I, P = collect_inputs(backbone, dev_val,
                                    encoding["image_aggregation"], args.device, bs)
    assert order == dev_val, "loader reordered the split"
    raw = {m: x.numpy().astype(np.float64)
           for m, x in zip(MODALITIES, (T, I, P))}
    mean3 = np.mean([raw[m] for m in MODALITIES], axis=0)

    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    norms = {m: {"mean": float(np.linalg.norm(raw[m], axis=1).mean()),
                 "std": float(np.linalg.norm(raw[m], axis=1).std())}
             for m in MODALITIES}
    out = {
        "what": ("per-modality embedding norm on dev_val, and how parallel the "
                 "UNNORMALISED mean of the three is to each one"),
        "why": ("the parameter-free control and the raw derangement arms build "
                "their gallery as an unnormalised mean, so the largest-norm "
                "modality dominates it"),
        "state": "trained" if trained else "released encoder (no checkpoint)",
        "checkpoint": args.state if trained else None,
        "checkpoint_sha256": sha,
        "n": len(order),
        "norms": norms,
        "norm_share_of_mean": {m: norms[m]["mean"] / sum(
            norms[k]["mean"] for k in MODALITIES) for m in MODALITIES},
        "cos_mean_of_three_vs": {
            m: float((unit(mean3) * unit(raw[m])).sum(1).mean())
            for m in MODALITIES},
        "mean_paired_cosine": {
            "text_pc": float((unit(raw["text"]) * unit(raw["pc"])).sum(1).mean()),
            "image_pc": float((unit(raw["image"]) * unit(raw["pc"])).sum(1).mean()),
            "text_image": float((unit(raw["text"]) * unit(raw["image"])).sum(1).mean())},
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("what", "why")}, indent=1))
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[out["state"]] = out
    OUT.write_text(json.dumps(prev, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
