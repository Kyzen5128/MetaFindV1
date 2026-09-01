#!/usr/bin/env python3
"""Is the paper's ordering set by the fusion, or by the endpoints?

Codex, 2026-09-01, correcting the previous probe:

> No asymmetric fusion is required. T+PC is one vector and one cell; "adding PC
> helps text" and "adding text hurts PC" are the two halves of
> `text < T+PC < PC`. [...] The decisive discrepancy is upstream of fusion.
> Your image-only endpoint against the PC gallery is ~59.7, the paper reports
> 0.1.

He is right and `pc_gallery_weight_sweep`'s conclusion was wrong. That probe
reported "the ordering is inverted at every alpha" and read it as a property of
mean pooling. It is not. The sweep tied alpha_text and alpha_image together AND
carried a real, highly informative image embedding, so `image+pc` could never
fall below `text+pc`: our image endpoint alone is 59.7 where the paper's is 0.1.
The ordering follows the ENDPOINTS, not the fusion rule.

His synthetic check -- uninformative text/image at effective scales ~3.4 and
~3.8 relative to pc, plain linear addition, no learned fusion -- produced 33.90
/ 22.28 / 7.06 against the paper's 33.9 / 22.6 / 6.4.

This runs the same test on OUR REAL VECTORS rather than synthetic noise, which
is the part a synthetic control cannot settle. Two of his diagnostics at once:

  * each modality's query is either REAL (asset i's own) or SHUFFLED (a random
    other asset's, so the distribution and norms are untouched and only the
    instance signal is destroyed)
  * a two-dimensional (alpha_text, alpha_image) sweep instead of one shared
    alpha

Gallery is point-cloud only, released encoder, 9,138 test uids against all
45,692 -- the construction `IDesign/retrieve.py` uses, which is the only
asset-retrieval implementation in the cloned lineage.

If shuffled-image plus some (alpha_t, alpha_i) reproduces the paper's full
ordering on real vectors, then mean pooling is exonerated and the whole question
becomes why MetaFind's baseline image embedding carries no instance signal.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "uninformative_endpoint_grid.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
PAPER = {"text": 0.1, "image": 0.1, "pc": 97.9, "text+image": 0.0,
         "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}
CELLS = list(PAPER)
SCORED = ("text+pc", "image+pc", "full")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphas", default="1,2,3,3.4,3.8,4,5,6,8,12")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    q_uids = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"queries {len(q_uids):,}  gallery {len(corpus):,} (pc only)", flush=True)

    P_all = np.load(PC_CACHE).astype(np.float32)
    qi = np.array([pos[u] for u in q_uids])
    T = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["text"].astype(np.float32)
                  for u in q_uids])
    I = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["image"].astype(np.float32)
                  for u in q_uids])

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(q_uids))
    # A derangement, so no query keeps its own vector by accident.
    fixed = np.flatnonzero(perm == np.arange(len(q_uids)))
    if len(fixed):
        perm[fixed] = perm[(fixed + 1) % len(perm)]

    nrm = lambda a: torch.nn.functional.normalize(a, dim=-1)
    G = nrm(torch.from_numpy(P_all).to(dev))
    tgt = torch.tensor(qi, device=dev)
    t_real = nrm(torch.from_numpy(T).to(dev))
    i_real = nrm(torch.from_numpy(I).to(dev))
    p = nrm(torch.from_numpy(P_all[qi]).to(dev))
    t_shuf, i_shuf = t_real[perm], i_real[perm]

    def r1(q):
        s = nrm(q) @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        return ((s > own).sum(1) < 1).sum().item() / len(q_uids) * 100

    alphas = [float(a) for a in args.alphas.split(",")]
    res = {"n_query": len(q_uids), "n_gallery": len(corpus),
           "gallery": "pc only, released encoder", "paper_ulip_row": PAPER,
           "arms": {}}

    print(f"\n{'論文':<26s}" + "".join(f"{PAPER[c]:>10.1f}" for c in CELLS))
    for tm, im in itertools.product(("real", "shuffled"), repeat=2):
        t = t_real if tm == "real" else t_shuf
        i = i_real if im == "real" else i_shuf
        endpoints = {"text": r1(t), "image": r1(i)}
        best, best_err = None, float("inf")
        for at, ai in itertools.product(alphas, repeat=2):
            cells = {"text": endpoints["text"], "image": endpoints["image"],
                     "pc": r1(p),
                     "text+image": r1(at * t + ai * i),
                     "text+pc": r1(at * t + p),
                     "image+pc": r1(ai * i + p),
                     "full": r1(at * t + ai * i + p)}
            err = sum(abs(cells[c] - PAPER[c]) for c in SCORED)
            if err < best_err:
                best, best_err = (at, ai, cells), err
        at, ai, cells = best
        order_ok = (cells["full"] < cells["image+pc"] < cells["text+pc"]
                    < cells["pc"])
        key = f"text={tm}, image={im}"
        res["arms"][key] = {"alpha_text": at, "alpha_image": ai,
                            "abs_err_on_3_cells": round(best_err, 2),
                            "ordering_matches_paper": bool(order_ok),
                            "cells": {c: round(cells[c], 2) for c in CELLS}}
        print(f"{key:<26s}" + "".join(f"{cells[c]:>10.2f}" for c in CELLS)
              + f"   a_t={at} a_i={ai} err={best_err:.1f} "
              + ("順序✓" if order_ok else "順序✗"))

    win = min(res["arms"].items(), key=lambda kv: kv[1]["abs_err_on_3_cells"])
    res["closest_arm"] = win[0]
    print(f"\n最接近論文的一組: {win[0]}  誤差 {win[1]['abs_err_on_3_cells']:.1f} "
          f"（三格合計），順序{'符合' if win[1]['ordering_matches_paper'] else '不符'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
