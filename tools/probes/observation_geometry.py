#!/usr/bin/env python3
"""How independent is the "second observation", and what does the ULIP row do with it?

[GPT via Kyzen 2026-09-03] "P5 跑完，不要先看 Stage1 R@1，先把三個 raw
observation 的 geometry 印出來." Two questions, both answered with the RELEASED
ULIP-2 encoder (no MetaFind training in the loop):

  A. Geometry of the query pack against the gallery's own record, per modality:
       paired cosine   cos(q_text, g_text), cos(q_image, g_image), cos(q_pc, g_pc)
       raw retrieval   q_text -> {g_text}, q_image -> {g_image}, q_pc -> {g_pc}
     over the whole gallery pool. If a "second" point-cloud sample still
     retrieves its own asset at 95%+, the bytes changed and the difficulty did
     not.

  B. The ULIP baseline row (gallery = PC embedding, query = mean of the
     available modality embeddings, raw and per-modality-L2) with the query
     built from the pack instead of the gallery's own record. Audit item
     P0-4': the paper's ULIP row has T+PC 33.9 below PC-only 97.9, which no
     text arm reproduced while the query pc was the gallery's own (98.7-99.3).

Runs under the desc_v1 overlay (METAFIND_DATA=/home/kyzen/metafind/metafind_data_desc,
METAFIND_TEXT_TEMPLATE=desc_v1): gallery text = canonical desc_v1 vector,
query text = the pack's alternate-description vector encoded by the same
frozen tower. Query image = the uid-seeded single view; gallery = 12-view mean.
Query pc = the pack's resampled cloud, gallery pc = the canonical cloud, both
through the released PointBERT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.eval.retrieval import QUERY_CONDITIONS, normalize_for_scoring, recall_at_k
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
from metafind.train.stage1 import QueryPack, load_protocols, protocol_n_views

PAPER_ULIP = dict(zip(QUERY_CONDITIONS, (0.1, 0.1, 97.9, 0.0, 33.9, 22.6, 6.4)))


def encode_pc(bb, clouds: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(clouds), batch):
            out.append(bb.encode_pc(torch.from_numpy(clouds[i:i + batch])).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_observation_geometry.json")
    args = ap.parse_args()

    enc, _, _ = load_protocols()
    pack = QueryPack(paths.OUTPUTS / "_probe" / "query_pack" / "query_pack.json",
                     protocol_n_views(enc))
    if not {"text", "pc"} <= set(pack.arms):
        raise SystemExit(f"pack arms {pack.arms}: need text and pc")
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    pack.require(q_uids)

    print(f"gallery {len(g_uids):,}  queries {len(q_uids):,}  text id {enc['text_serialization']}", flush=True)
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))

    # ---- gallery: canonical record, all three modalities -------------------
    g_text = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["text"] for u in g_uids]).astype(np.float32)
    g_img = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["image"] for u in g_uids]).astype(np.float32)
    print("encoding gallery pc (canonical clouds, released PointBERT)", flush=True)
    clouds = []
    for u in g_uids:
        c = np.load(paths.POINTCLOUDS / f"{u}.npz")
        clouds.append(np.concatenate([c["xyz"], c["rgb"]], axis=1).astype(np.float32))
    g_pc = encode_pc(bb, np.stack(clouds)); del clouds

    # ---- query: the pack's second observation per modality ------------------
    q_text = np.stack([pack.vector("text", u) for u in q_uids])
    q_img = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"][pack.view_index(u)]
                      for u in q_uids]).astype(np.float32)
    print("encoding query pc (resampled clouds from the pack)", flush=True)
    q_pc = encode_pc(bb, np.stack([pack.vector("pc", u) for u in q_uids]))

    # ---- A. geometry --------------------------------------------------------
    rows = targets
    out = {"n_gallery": len(g_uids), "n_query": len(q_uids), "geometry": {}, "ulip_row": {}}
    print("\n=== A. second observation vs the gallery's own record (released encoder) ===")
    print("modality   paired cos (mean / p5 / p95)    raw q->g R@1   R@5")
    for name, q, g in (("text", q_text, g_text), ("image", q_img, g_img), ("pc", q_pc, g_pc)):
        qn, gn = normalize_for_scoring(q), normalize_for_scoring(g)
        paired = (qn * gn[rows]).sum(1)
        r = recall_at_k(qn @ gn.T, targets)
        out["geometry"][name] = {"paired_cos_mean": float(paired.mean()),
                                 "paired_cos_p5": float(np.percentile(paired, 5)),
                                 "paired_cos_p95": float(np.percentile(paired, 95)),
                                 "raw_R@1": r["R@1"], "raw_R@5": r["R@5"]}
        print(f"{name:<9} {paired.mean():.4f} / {np.percentile(paired, 5):.4f} / "
              f"{np.percentile(paired, 95):.4f}        {r['R@1']*100:6.1f}  {r['R@5']*100:6.1f}")
    # the same-record reference, so the reader sees what "1.0" looks like here
    print("(same-record reference: paired cos = 1.0000 and raw R@1 = 100.0 by construction)")

    # ---- B. ULIP row with the pack as query ---------------------------------
    print("\n=== B. ULIP row: gallery = PC embedding, query = mean of available modalities, "
          "query observations from the PACK ===")
    print("paper ULIP  " + "".join(f"{PAPER_ULIP[c]:>8.1f}" for c in QUERY_CONDITIONS))
    gpc = normalize_for_scoring(g_pc)
    for mode in ("rawmean", "l2mean"):
        mods = {"text": q_text, "image": q_img, "pc": q_pc}
        if mode == "l2mean":
            mods = {k: normalize_for_scoring(v) for k, v in mods.items()}
        cells = {}
        line = f"{mode:<11} "
        for cond in QUERY_CONDITIONS:
            parts = [mods[m] for m in ("text", "image", "pc")
                     if m in cond.replace("full", "text+image+pc").split("+")]
            q = normalize_for_scoring(np.mean(parts, axis=0))
            r = recall_at_k(q @ gpc.T, targets)
            cells[cond] = r
            line += f"{r['R@1']*100:>8.1f}"
        out["ulip_row"][mode] = cells
        print(line)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
