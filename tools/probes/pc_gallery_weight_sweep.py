#!/usr/bin/env python3
"""How much text does it take to pull a query off a point-cloud gallery?

Codex, 2026-09-01, proposing a sixth reading of Table 1: the single-tower
baseline rows are scored against a POINT-CLOUD-ONLY gallery, not a
modality-complete one, and the descent

    pc 97.9  ->  text+pc 33.9  ->  image+pc 22.6  ->  full 6.4

is a multimodal query being dragged away from a pc gallery by unaligned text
and image vectors. The suggested control:

    gallery = normalize(pc)
    q_pc    = normalize(pc)
    q_tpc   = normalize(text + pc)
    q_full  = normalize(text + image + pc)

THAT EXACT CONTROL HAS ALREADY BEEN RUN. `table1_baseline_grid` swept 24
configurations of it -- "gallery = released ULIP-2 point-cloud embedding; query
= mean pool of the present modalities", 9,138 test uids against all 45,692 --
and its lowest `full` over all 24 was **97.47** against the paper's 6.4, with
text+pc 99.75 against 33.9. The mechanism does not descend at unit weight.

But the mechanism has a free parameter the grid held fixed, and Codex is
pointing straight at it: HOW MUCH the text and image weigh against the point
cloud in the sum. Under the released encoder the raw norms are text 37.07,
image 40.27, pc 27.86, so `text + pc` is already 57% text -- and still scores
99.75, because cos(that sum, its own pc) stays far above the cosine to any
other asset's pc in the pool.

So the question is not whether the mechanism exists. It is whether any weight
reproduces the paper's descent. This sweeps it:

    q = normalize( alpha * text_hat + alpha * image_hat + pc_hat )

with every modality unit-normalised first so `alpha` is the only scale, and
alpha running from 0 (pure pc) upward. Three cells are scored against the
paper's ULIP row at once, so a single alpha has to explain all three or none.

If some alpha lands near 33.9 / 22.6 / 6.4 together, Codex's reading is right
and the grid simply had the wrong scale. If the three cells cross their targets
at different alphas, the shape is not a scale effect and the reading is dead.

Released encoder, no Stage 1 weights, nothing trained.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402
from metafind.models import resolve_stage1 as R  # noqa: E402

OUT = REPO / "output" / "look" / "pc_gallery_weight_sweep.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
PAPER_ULIP = {"pc": 97.9, "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5,2,3,5,8,12,20,40")
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"queries {len(queries):,}  gallery {len(corpus):,}  (pc only)", flush=True)

    P_all = np.load(PC_CACHE).astype(np.float32)
    qi = np.array([pos[u] for u in queries])
    T = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["text"].astype(np.float32)
                  for u in queries])
    I = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["image"].astype(np.float32)
                  for u in queries])

    n = lambda a: torch.nn.functional.normalize(a, dim=-1)
    G = n(torch.from_numpy(P_all).to(dev))
    t = n(torch.from_numpy(T).to(dev))
    i = n(torch.from_numpy(I).to(dev))
    p = n(torch.from_numpy(P_all[qi]).to(dev))
    tgt = torch.tensor(qi, device=dev)

    print(f"\n未正規化的原始長度: text {np.linalg.norm(T,axis=1).mean():.2f}  "
          f"image {np.linalg.norm(I,axis=1).mean():.2f}  "
          f"pc {np.linalg.norm(P_all[qi],axis=1).mean():.2f}")

    def r1(q):
        s = n(q) @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        return ((s > own).sum(1) < 1).sum().item() / len(queries) * 100

    res = {"n_query": len(queries), "n_gallery": len(corpus),
           "gallery": "pc only, released encoder", "paper_ulip_row": PAPER_ULIP,
           "already_run": {"note": "table1_baseline_grid, 24 configs, unit "
                                   "weight", "best_full": 97.47,
                           "best_text_pc": 99.75},
           "sweep": {}}
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"\n{'alpha':>7s}{'pc':>9s}{'text+pc':>10s}{'image+pc':>10s}{'full':>9s}")
    print(f"{'論文':>7s}{PAPER_ULIP['pc']:9.1f}{PAPER_ULIP['text+pc']:10.1f}"
          f"{PAPER_ULIP['image+pc']:10.1f}{PAPER_ULIP['full']:9.1f}")
    for a in alphas:
        row = {"pc": r1(p),
               "text+pc": r1(a * t + p),
               "image+pc": r1(a * i + p),
               "full": r1(a * t + a * i + p)}
        res["sweep"][a] = {k: round(v, 2) for k, v in row.items()}
        print(f"{a:7.2f}{row['pc']:9.2f}{row['text+pc']:10.2f}"
              f"{row['image+pc']:10.2f}{row['full']:9.2f}")

    def crossing(cell):
        prev = None
        for a in alphas:
            v = res["sweep"][a][cell]
            if prev is not None and (prev[1] - PAPER_ULIP[cell]) * \
                                    (v - PAPER_ULIP[cell]) <= 0:
                return round(prev[0] + (a - prev[0]) *
                             (prev[1] - PAPER_ULIP[cell]) /
                             (prev[1] - v + 1e-12), 3)
            prev = (a, v)
        return None

    res["alpha_hitting_paper"] = {c: crossing(c) for c in
                                  ("text+pc", "image+pc", "full")}
    print(f"\n各格命中論文值所需的 alpha: {res['alpha_hitting_paper']}")
    print("三格要在同一個 alpha 命中，這個讀法才成立")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
