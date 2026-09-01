#!/usr/bin/env python3
"""What does dropping one of the twelve renders cost? Measured, all twelve ways.

Kyzen, 2026-09-01: 「那如果我隨便刪掉一張 Objaverse 的話呢?」

`2methdology.tex:28` says each asset is "rendered from 11 orthogonal
viewpoints"; `render_blender.py:103` renders 12, and DL-065 verified those 12
are OpenShape's own cameras with our index order rotated 180 degrees. Dropping
one closes the stated deviation. The question is what it costs, and it has only
ever been bounded by interpolation: `two_deviations` measured the 12-view mean
at 54.75 and a single view at 52.20, so 12 -> 11 was assumed to be a small
fraction of that 2.55.

This measures it instead, dropping each index in turn, because the twelve are
not interchangeable. Our own per-view profile has the bottom ring 13 points
below the top (DL-065: views 0-3 average 48.4, 4-7 46.1, 8-11 35.1), so which
one is dropped may matter more than that one is.

Gallery and query are both rebuilt from the same 11, which is the honest
version of the change: an 11-view corpus, not an 11-view query against a
12-view gallery.

Text query against the image-mean gallery would confound this with the text
question, so the query here is the image mean itself and the gallery is the
point-cloud embedding -- image -> pc, the same cell `two_deviations` priced.
9,138 test uids, all 45,692 as gallery, released encoder, nothing trained.
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

OUT = REPO / "output" / "look" / "drop_one_view.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
RINGS = {0: "上排 phi60", 1: "上排 phi60", 2: "上排 phi60", 3: "上排 phi60",
         4: "中排 phi90", 5: "中排 phi90", 6: "中排 phi90", 7: "中排 phi90",
         8: "下排 phi120", 9: "下排 phi120", 10: "下排 phi120", 11: "下排 phi120"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"queries {len(queries):,}  gallery {len(corpus):,}", flush=True)

    V = np.empty((len(queries), 12, 1280), np.float32)
    for a in range(0, len(queries), args.chunk):
        b = min(a + args.chunk, len(queries))
        for i in range(a, b):
            V[i] = np.load(paths.EMBEDDINGS / f"{queries[i]}.npz")["views"]
        print(f"  loaded {b:,}/{len(queries):,}", flush=True)

    Vt = torch.from_numpy(V).to(dev)
    G = torch.nn.functional.normalize(
        torch.from_numpy(np.load(PC_CACHE)).float().to(dev), dim=-1)
    tgt = torch.tensor([pos[u] for u in queries], device=dev)

    def score(q):
        q = torch.nn.functional.normalize(q, dim=-1)
        s = q @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        h = (s > own).sum(1)
        m = q.shape[0]
        return ((h < 1).sum().item() / m * 100, (h < 5).sum().item() / m * 100)

    full = Vt.sum(1)
    base1, base5 = score(full / 12.0)
    res = {"n_query": len(queries), "n_gallery": len(corpus),
           "all_12": {"R@1": round(base1, 2), "R@5": round(base5, 2)},
           "drop_one": {}}
    print(f"\n全部 12 張        R@1 {base1:6.2f}   R@5 {base5:6.2f}")
    print(f"\n{'刪掉哪一張':<18s}{'R@1':>8s}{'差':>8s}{'R@5':>8s}")
    d1 = []
    for k in range(12):
        a1, a5 = score((full - Vt[:, k]) / 11.0)
        d1.append(a1)
        res["drop_one"][k] = {"ring": RINGS[k], "R@1": round(a1, 2),
                              "delta_R1": round(a1 - base1, 2),
                              "R@5": round(a5, 2)}
        print(f"  {k:02d} {RINGS[k]:<13s}{a1:8.2f}{a1 - base1:+8.2f}{a5:8.2f}")

    res["worst_drop"] = {"view": int(np.argmin(d1)), "R@1": round(min(d1), 2)}
    res["best_drop"] = {"view": int(np.argmax(d1)), "R@1": round(max(d1), 2)}
    res["spread"] = round(max(d1) - min(d1), 2)
    print(f"\n最好刪 {np.argmax(d1):02d} ({max(d1):.2f})　"
          f"最差刪 {np.argmin(d1):02d} ({min(d1):.2f})　"
          f"差距 {max(d1) - min(d1):.2f} 分")
    print(f"12 張 vs 11 張，最大代價 {base1 - min(d1):.2f} 分")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
