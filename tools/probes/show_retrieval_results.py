#!/usr/bin/env python3
"""What actually came back. Top-k for a handful of queries, named.

Kyzen, 2026-09-01: 「我說三筆數據 列出來 我要看檢索結果」

`exp_c_our_corpus_rows.jsonl` records a rank per query per direction, which
says how far down the right answer was and nothing about what was above it. A
rank of 20 is a different thing when the nineteen ahead are other blankets than
when they are lampposts, and only one of those is a retrieval problem.

So this prints the ranked list: for each query and each direction, the top-k
gallery assets by category, marked against the target, with the cosine. Same
construction as Experiment C -- released ULIP-2 10k-xyzrgb, no Stage 1, no
fusion, gallery is the pure target modality, query text is a NON-canonical
`description_candidate` and query image is one view.

Reads only. Recomputes rather than caching, because a stale cache here would
mislabel which asset a rank belonged to.
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
from metafind.data.pointclouds import uid_seed  # noqa: E402
from metafind.models import resolve_stage1 as R  # noqa: E402

PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
ROWS = REPO / "output" / "look" / "exp_c_our_corpus_rows.jsonl"
OUT = REPO / "output" / "look" / "retrieval_results_sample.json"


def label(uid: str) -> str:
    try:
        a = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
        return a["category"]
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="0,4000,9000")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    picked = [int(x) for x in args.rows.split(",")]
    rows = [json.loads(l) for l in ROWS.read_text().splitlines()]
    sel = [rows[i] for i in picked]

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope="frozen"))

    G_p = np.load(PC_CACHE).astype(np.float32)
    G_t = np.empty((len(corpus), 1280), np.float32)
    G_i = np.empty((len(corpus), 1280), np.float32)
    for i, u in enumerate(corpus):
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        G_t[i] = z["text"]
        G_i[i] = z["image"]
        if (i + 1) % 15000 == 0:
            print(f"  gallery {i + 1:,}/{len(corpus):,}", flush=True)

    n = lambda a: torch.nn.functional.normalize(
        torch.from_numpy(np.asarray(a, np.float32)).to(dev), dim=-1)
    Gt, Gi, Gp = n(G_t), n(G_i), n(G_p)

    uids = [r["query_uid"] for r in sel]
    with torch.no_grad():
        Qt = n(bb.encode_text([r["caption"] for r in sel]).float().cpu().numpy())
    Qi = n(np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"]
                     .astype(np.float32)[uid_seed(u) % 12] for u in uids]))
    Qp = n(G_p[[pos[u] for u in uids]])

    dirs = [("T2S  文字 → 點雲", Qt, Gp), ("I2S  圖片 → 點雲", Qi, Gp),
            ("S2T  點雲 → 文字", Qp, Gt), ("T2I  文字 → 圖片", Qt, Gi)]

    dump = []
    for r_i, r in enumerate(sel):
        u = r["query_uid"]
        tgt = pos[u]
        print("\n" + "=" * 84)
        print(f"row {r['row']}   查詢資產 {u}   真實類別「{label(u)}」"
              f"   在畫廊第 {tgt:,} 列")
        print(f"查詢文字（第 {r['caption_rank_used']} 順位候選，"
              f"視角 {r['view']:02d}）:")
        c = r["caption"]
        for a in range(0, len(c), 78):
            print("   " + c[a:a + 78])
        entry = {"row": r["row"], "uid": u, "category": label(u),
                 "caption": c, "view": r["view"], "directions": {}}
        for name, Q, G in dirs:
            s = (Q[r_i:r_i + 1] @ G.t())[0]
            top = torch.topk(s, args.k)
            rank = int((s > s[tgt]).sum()) + 1
            print(f"\n  {name}    目標排名 {rank:,} / {len(corpus):,}"
                  f"    目標 cos {float(s[tgt]):.4f}")
            hits = []
            for p, (sc, ix) in enumerate(zip(top.values.tolist(),
                                             top.indices.tolist()), 1):
                gu = corpus[ix]
                mark = "  ← 目標" if ix == tgt else ""
                print(f"     {p}. {sc:.4f}  {label(gu):<26s} {gu[:16]}…{mark}")
                hits.append({"pos": p, "cos": round(sc, 4), "uid": gu,
                             "category": label(gu), "is_target": ix == tgt})
            entry["directions"][name] = {"target_rank": rank,
                                         "target_cos": round(float(s[tgt]), 4),
                                         "top": hits}
        dump.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_gallery": len(corpus), "queries": dump},
                              indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
