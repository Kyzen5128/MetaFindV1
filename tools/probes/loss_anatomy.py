#!/usr/bin/env python3
"""Is loss 2.39 the floor for this objective, or is there room left in it?

The claim under test: at batch 64 with a FIXED tau = 0.5, an InfoNCE loss near
2.3 is what a model that already ranks every positive first looks like, so the
plateau is arithmetic and not a training failure.

Two bounds, both exact, and they differ by what the negatives are assumed to do:

    negatives at cosine 0        L = log(e^2 + 63) - 2                = 2.2540
    negatives at the geometric   L = log(e^2 + 63 e^(-2/63)) - 2      = 2.2257
    minimum -1/(B-1)

The second is the true floor: 64 unit vectors cannot have mean pairwise cosine
below -1/(B-1), so no arrangement does better.

WHAT WAS NEVER MEASURED
-----------------------
`diag_trained_fusion_identity` recorded the positive and the HARDEST negative
per Table-1 condition, and the hardest negative came out at 0.82-0.83 -- nowhere
near the 0 the first bound assumes. The MEAN negative was never measured, and it
is the term that decides whether 2.39 sits on the floor or above it. This
measures it, on real batches, together with the loss those same batches produce,
so the arithmetic can be checked against itself rather than against an
assumption.

Reported per Table-1 condition, because a pc-bearing query and a text-only query
are not in the same place at all.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "loss_anatomy.json"
CONDS = {"text": (1, 0, 0), "image": (0, 1, 0), "pc": (0, 0, 1),
         "text+image": (1, 1, 0), "text+pc": (1, 0, 1),
         "image+pc": (0, 1, 1), "full": (1, 1, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind_out/checkpoints/"
                                      "sweep_lr/lr2.50e-4_s20260830/stage1_best.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    B, tau = args.batch, args.tau
    floor_zero = math.log(math.e ** (1 / tau) + (B - 1)) - 1 / tau
    floor_geom = math.log(math.e ** (1 / tau)
                          + (B - 1) * math.e ** (-(1 / tau) / (B - 1))) - 1 / tau
    chance = math.log(B)
    print(f"batch {B}, tau {tau}")
    print(f"  chance (all logits equal)        {chance:.4f}")
    print(f"  floor if negatives at cos 0      {floor_zero:.4f}")
    print(f"  floor at the geometric minimum   {floor_geom:.4f}\n")

    from metafind.train.stage1 import (build_model, load_protocols,
                                       load_stage1_checkpoint)
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    encoding, training, hyper = load_protocols()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope="point_encoder_and_fuser"))
    model, loss_fn = build_model(encoding, training, hyper)
    model.to(dev)
    load_stage1_checkpoint(bb, model, loss_fn, pathlib.Path(args.ckpt))
    model.eval()

    uids = sorted(json.loads((paths.OUTPUTS / "splits.json").read_text())
                  ["object"]["dev_val"])
    rng = np.random.default_rng(args.seed)
    picks = [rng.choice(len(uids), size=B, replace=False) for _ in range(args.batches)]

    stats = {c: {"pos": [], "neg": [], "hard": [], "loss": []} for c in CONDS}
    with torch.no_grad():
        for sel in picks:
            batch = [uids[i] for i in sel]
            txt, img, pc = [], [], []
            for u in batch:
                z = np.load(paths.EMBEDDINGS / f"{u}.npz")
                txt.append(z["text"].astype(np.float32))
                img.append(z["image"].astype(np.float32))
                c = np.load(paths.POINTCLOUDS / f"{u}.npz")
                pc.append(np.concatenate([c["xyz"].astype(np.float32),
                                          c["rgb"].astype(np.float32)], 1))
            T = torch.from_numpy(np.stack(txt)).to(dev)
            I = torch.from_numpy(np.stack(img)).to(dev)
            P = bb.encode_pc(np.stack(pc)).float()
            g = model.gallery({"text": T, "image": I, "pc": P})
            g = torch.nn.functional.normalize(g, dim=-1)

            for cond, mask in CONDS.items():
                present = torch.tensor(mask, dtype=torch.bool,
                                       device=dev).expand(B, -1).clone()
                q = model.query({"text": T, "image": I, "pc": P}, present=present)
                q = torch.nn.functional.normalize(q, dim=-1)
                s = q @ g.t()
                eye = torch.eye(B, dtype=torch.bool, device=dev)
                pos = s.diagonal()
                neg = s.masked_fill(eye, float("-inf"))
                stats[cond]["pos"].append(pos.mean().item())
                stats[cond]["neg"].append(s.masked_select(~eye).mean().item())
                stats[cond]["hard"].append(neg.max(1).values.mean().item())
                stats[cond]["loss"].append(torch.nn.functional.cross_entropy(
                    s / tau, torch.arange(B, device=dev)).item())

    res = {"batch": B, "tau": tau, "batches": args.batches,
           "checkpoint": args.ckpt,
           "chance": round(chance, 4),
           "floor_negatives_at_zero": round(floor_zero, 4),
           "floor_geometric": round(floor_geom, 4), "conditions": {}}
    print(f"{'條件':<12s}{'pos':>8s}{'mean neg':>10s}{'hardest':>9s}"
          f"{'margin':>8s}{'loss':>8s}{'距地板':>9s}")
    for cond, d in stats.items():
        p, n, h, l = (float(np.mean(d[k])) for k in ("pos", "neg", "hard", "loss"))
        res["conditions"][cond] = {"pos_cos": round(p, 4), "mean_neg_cos": round(n, 4),
                                   "hardest_neg_cos": round(h, 4),
                                   "margin": round(p - h, 4), "loss": round(l, 4),
                                   "above_geometric_floor": round(l - floor_geom, 4)}
        print(f"{cond:<12s}{p:8.4f}{n:10.4f}{h:9.4f}{p - h:8.4f}{l:8.4f}"
              f"{l - floor_geom:9.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
