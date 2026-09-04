#!/usr/bin/env python3
"""PAIRWISE CROSS-MODAL DIAGNOSTIC -- released ULIP-2 single tower, no Fusion, no training.

[KYZEN ✅ 2026-09-04] docs/audit/RETRIEVAL_EVALUATION_DEFINITION_20260904.md §4. This is NOT a
Table 1 reproduction: Table 1's gallery is the Fusion output over T+I+PC; here every cell is one
modality's ULIP-2 vector against another modality's ULIP-2 vector, positive = same UID.

Cells (query -> gallery):  T->I  I->T  T->PC  PC->T  I->PC  PC->I  and the within-modal
controls T->T  I->I  PC->PC.  Query image = ONE view (uid_seed % 12); gallery image = the 12-view
mean, so I->I is single view vs mean (not the same vector). T->T and PC->PC use the SAME vector on
both sides and must read 100: they are the identity-shortcut controls, not measurements.

Scored twice: cosine (our evaluator, float64, ties against) and Text2Shape's verbatim RR@k /
NDCG@5 on the raw (unnormalised) vectors -- the scorer upstream actually runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.eval.text2shape_eval import text2shape_metrics
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", help="val (dev) or test (sealed; needs --unseal)")
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/pairwise_crossmodal_diag.json")
    args = ap.parse_args()
    if args.split in ("test", "holdout", "full") and not args.unseal:
        raise SystemExit(f"--split {args.split} reads the sealed split; pass --unseal deliberately")
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())
    uids = sorted(sp["object"][args.split])
    n = len(uids)
    print(f"{n:,} {args.split} assets (scheme {sp.get('scheme')})", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    text = np.stack([emb(u, "text") for u in uids])
    views = [emb(u, "views") for u in uids]
    img_q = np.stack([v[uid_seed(u) % 12] for u, v in zip(uids, views)])
    img_g = np.stack([v.mean(0) for v in views])
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    clouds = []
    for u in uids:
        c = np.load(paths.POINTCLOUDS / f"{u}.npz")
        clouds.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
    out = []
    with torch.no_grad():
        for i in range(0, n, 32):
            out.append(bb.encode_pc(torch.from_numpy(np.stack(clouds[i:i + 32]))).float().cpu().numpy())
    pc = np.concatenate(out)
    print("encoded pc", flush=True)

    mods_q = {"T": text, "I": img_q, "PC": pc}
    mods_g = {"T": text, "I": img_g, "PC": pc}
    targets = np.arange(n)
    cells = {}
    for q in ("T", "I", "PC"):
        for g in ("T", "I", "PC"):
            name = f"{q}->{g}"
            qu, gu = normalize_for_scoring(mods_q[q]), normalize_for_scoring(mods_g[g])
            cos = recall_at_k(qu @ gu.T, targets, (1, 5))
            t2s_unit = text2shape_metrics(qu, gu, targets)
            t2s_raw = text2shape_metrics(mods_q[q].astype(np.float64), mods_g[g].astype(np.float64), targets)
            cells[name] = {
                "cosine": {"R@1": cos["R@1"], "R@5": cos["R@5"]},
                "text2shape_unit": {k: t2s_unit[k] for k in ("RR@1", "RR@5", "NDCG@5")},
                "text2shape_raw_dot": {k: t2s_raw[k] for k in ("RR@1", "RR@5", "NDCG@5")},
                "identity_control": (q == g and q != "I"),
            }
            c = cells[name]
            print(f"  {name:7s} cosine R@1 {c['cosine']['R@1']*100:5.1f} R@5 {c['cosine']['R@5']*100:5.1f}"
                  f" | t2s raw-dot RR@1 {t2s_raw['RR@1']*100:5.1f} RR@5 {t2s_raw['RR@5']*100:5.1f}"
                  f" NDCG@5 {t2s_raw['NDCG@5']*100:5.1f}"
                  + ("   [identity control: same vector both sides]" if c["identity_control"] else ""), flush=True)
    res = {"label": "PAIRWISE CROSS-MODAL DIAGNOSTIC -- not Table 1", "split": args.split, "n": n,
           "scheme": sp.get("scheme"), "split_seed": sp.get("split_seed"), "val_seed": sp.get("val_seed"),
           "backbone": "released ULIP-2 (PointBERT 10k xyzrgb, OpenCLIP ViT-bigG-14), no Fusion, no training",
           "query_image": "one view (uid_seed % 12)", "gallery_image": "12-view mean",
           "positive": "same uid", "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
