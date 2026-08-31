#!/usr/bin/env python3
"""Build MetaFind's baseline exactly as described, with an INDEPENDENT query cloud.

Kyzen, 2026-09-01: "it cannot be copied -- build what they built and run it, then
you will know." This is that build, with the one ingredient the 24-configuration
grid (DL-057) could not vary.

THE CONSTRUCTION, from `3experiments.tex` and nothing else
-----------------------------------------------------------
"we extend each baseline by adding a simple **mean pooling layer** to aggregate
available modalities, and use these fused embeddings to retrieve from a
pre-encoded gallery."

"their 'PC only' performance reflects retrieval using **identical embeddings for
both query and gallery**" -- which pins the gallery: for PC-only the query is the
point-cloud embedding alone, so query == gallery only if the gallery IS that
embedding. Gallery = released ULIP-2 point-cloud embedding.

WHAT IS DIFFERENT HERE
----------------------
DL-057 swept pooling, text rung, image construction and pool size, and the four
discriminative cells never moved: pc stayed at 100.0 in all 24, because the query
point cloud WAS the gallery entry -- cosine exactly 1, and mean-pooling something
that starts at 1 still beats every other asset. The paper needs 97.9 falling to
33.9 / 22.6 / 6.4.

So this run breaks the identity: the query point cloud is a SECOND, independent
10,000-point surface sample of the same mesh (`protocol_e_query_pc`, seed offset
1000003, same sampler and same `pc_norm`), and the query image is one held-out
view rather than the 12-view mean. Only the gallery keeps the canonical vectors.

It reports `cos(query pc, gallery pc)` FIRST, because that single number decides
whether the experiment can even reach the paper's row: if a resample lands at
0.999 then no query-side construction of this kind can produce the descent, and
the answer lies elsewhere.

Queries are the 4,569 dev_val assets -- the only split for which an independent
cloud exists. The gallery is the full 45,692 corpus.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "baseline_independent_query.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
QPC = paths.OUTPUTS / "_probe" / "protocol_e_query_pc" / "query_pc_offset1000003.npy"
QPC_MAN = QPC.with_suffix(".manifest.json")
LADDER = paths.OUTPUTS / "_probe" / "text_ladder"

CONDITIONS = ["text", "image", "pc", "text+image", "text+pc", "image+pc", "full"]
PAPER_ULIP = {
    "text": (0.1, 0.9), "image": (0.1, 1.3), "pc": (97.9, 99.4),
    "text+image": (0.0, 0.3), "text+pc": (33.9, 58.0),
    "image+pc": (22.6, 41.6), "full": (6.4, 15.9),
}


def corpus_uids() -> list[str]:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    return sorted(set(d["train"]) | set(d["test"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--text", default="L3_full_serialization",
                    choices=("L1_category_name", "L2_bare_description",
                             "L3_full_serialization"))
    args = ap.parse_args()

    uids = corpus_uids()
    pos = {u: i for i, u in enumerate(uids)}
    if not PC_CACHE.exists():
        sys.exit(f"{PC_CACHE} missing -- run tools/probes/table1_baseline_grid.py first")
    gal_pc = np.load(PC_CACHE)
    assert gal_pc.shape[0] == len(uids), (gal_pc.shape, len(uids))
    print(f"gallery {gal_pc.shape[0]:,} canonical pc embeddings", flush=True)

    man = json.loads(QPC_MAN.read_text())
    q_uids = man["uid_order"]
    q_rows = np.array([pos[u] for u in q_uids])
    print(f"queries {len(q_uids):,} dev_val assets with an independent cloud", flush=True)

    # --- encode the SECOND cloud with the same released encoder -------------
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    clouds = np.load(QPC, mmap_mode="r")
    q_pc = np.empty((len(q_uids), 1280), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(q_uids), args.batch):
            j = min(i + args.batch, len(q_uids))
            q_pc[i:j] = bb.encode_pc(np.asarray(clouds[i:j], dtype=np.float32)
                                     ).float().cpu().numpy()
            if i % 1024 == 0:
                print(f"  {i:,}/{len(q_uids):,} {time.time() - t0:.0f}s", flush=True)

    # --- THE GATING NUMBER --------------------------------------------------
    a = torch.from_numpy(q_pc)
    b = torch.from_numpy(gal_pc[q_rows])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    gate = {"mean": cos.mean().item(), "min": cos.min().item(),
            "p1": cos.quantile(0.01).item(), "median": cos.median().item()}
    print(f"\ncos(query pc, gallery pc): mean {gate['mean']:.6f}  "
          f"median {gate['median']:.6f}  p1 {gate['p1']:.6f}  min {gate['min']:.6f}\n",
          flush=True)

    # --- query modalities ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    q_txt = np.empty((len(q_uids), 1280), dtype=np.float32)
    q_img = np.empty((len(q_uids), 1280), dtype=np.float32)
    for k, u in enumerate(q_uids):
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        q_txt[k] = z["text"].astype(np.float32)
        v = z["views"]
        q_img[k] = v[rng.integers(v.shape[0])].astype(np.float32)   # one held-out view
    if args.text != "L3_full_serialization":
        sh = json.loads((LADDER / f"pack_{args.text}.json").read_text())["text"]["shards"][0]
        arr = np.load(sh["array"]).astype(np.float32)
        idx = {u: i for i, u in enumerate(sh["uid_order"])}
        hit = 0
        for k, u in enumerate(q_uids):
            if u in idx:
                q_txt[k] = arr[idx[u]]
                hit += 1
        print(f"text rung {args.text}: {hit:,}/{len(q_uids):,} overridden", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = torch.nn.functional.normalize(torch.from_numpy(gal_pc).to(dev), dim=-1)
    Q = {"text": torch.from_numpy(q_txt).to(dev),
         "image": torch.from_numpy(q_img).to(dev),
         "pc": torch.from_numpy(q_pc).to(dev)}
    target = torch.from_numpy(q_rows).to(dev)

    cells = {}
    for cond in CONDITIONS:
        keys = ["text", "image", "pc"] if cond == "full" else cond.split("+")
        q = torch.stack([Q[k] for k in keys], 0).mean(0)      # raw mean, as written
        q = torch.nn.functional.normalize(q, dim=-1)
        h1 = h5 = 0
        for i in range(0, q.shape[0], 1024):
            j = min(i + 1024, q.shape[0])
            sims = q[i:j] @ G.t()
            own = sims.gather(1, target[i:j].unsqueeze(1))
            higher = (sims > own).sum(dim=1)
            h1 += int((higher < 1).sum())
            h5 += int((higher < 5).sum())
        cells[cond] = (round(h1 / q.shape[0] * 100, 2), round(h5 / q.shape[0] * 100, 2))
        p = PAPER_ULIP[cond]
        print(f"  {cond:11s} R@1 {cells[cond][0]:6.2f}  R@5 {cells[cond][1]:6.2f}"
              f"   | paper {p[0]:5.1f} / {p[1]:5.1f}", flush=True)

    payload = {
        "what": "MetaFind's baseline construction (mean pool -> point-cloud "
                "gallery) with an INDEPENDENT query cloud and one held-out view",
        "gate": {"cos_query_pc_vs_gallery_pc": gate,
                 "reading": "a mean near 1.0 means a resample cannot break the "
                            "self-match, and no query-side construction of this "
                            "kind can reach the paper's descent"},
        "encoder": "released ULIP-2, no Stage 1 weights",
        "query_split": "dev_val", "n_query": len(q_uids),
        "gallery": "full corpus", "n_gallery": len(uids),
        "text_rung": args.text, "image": "one held-out view",
        "query_pc": str(QPC), "seed_offset": man.get("seed_offset"),
        "cells": {k: list(v) for k, v in cells.items()},
        "paper_ulip_row": {k: list(v) for k, v in PAPER_ULIP.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
