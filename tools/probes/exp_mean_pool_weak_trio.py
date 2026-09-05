#!/usr/bin/env python3
"""Rung 0 of the fusion ladder (no trained fusion at all) x the weak-but-own query trio.

DIAGNOSTIC. Released ULIP-2, frozen. Gallery vector = pooled unit vectors of the
asset's own attrs text, 12-view mean image and canonical cloud. Query vector =
pooled unit vectors of whatever modalities are present, drawn from:

    text   own attrs (= gallery's) | category+size (Figure 1 form) | ULIP-2 BLIP caption | partner attrs
    image  own view | own Sketchfab thumbnail | partner view
    pc     canonical (= gallery's) | second surface sample of the same mesh (QueryPack)

Two poolings, both from DL-099: `mean` = L2-normalise each modality then average
(Table 3 "Fusion = Mean" read literally on unit vectors); `raw` = average the raw
ULIP-2 vectors (norms text ~38 / image ~43 / pc ~28, so text and image outweigh pc).

Question: does the paper's ordering (pc > full > T+PC ~ I+PC >> singles) appear
when the combiner does not learn to trust the cloud AND every query modality is a
weaker second observation of the same asset? Nothing here is a reproduction of
MetaFind's trained fusion; it bounds what an untrained combiner does.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import QUERY_CONDITIONS, normalize_for_scoring, recall_at_k
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2, "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}
PAPER_ULIP = {"text": 0.1, "image": 0.1, "pc": 97.9, "text+image": 0.0, "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}
U2_FEATS = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/ulip2_query_feats/ulip2_query_feats.npz")
PACK = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/query_pack/query_pack.json")


def mods_of(cond: str) -> list[str]:
    return ["text", "image", "pc"] if cond == "full" else cond.split("+")


def pool(parts: list[np.ndarray], how: str) -> np.ndarray:
    if how == "mean":
        return normalize_for_scoring(np.mean([normalize_for_scoring(p) for p in parts], axis=0))
    return normalize_for_scoring(np.mean(parts, axis=0))          # raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="dev_val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_mean_pool_weak_trio_val.json")
    args = ap.parse_args()
    from tools.probes.exp_query_pc_observation import encode_clouds, encode_gallery_pc, perturb
    from metafind.train.stage1 import QueryPack

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in q_uids}
    pools = defaultdict(list)
    for u in q_uids:
        pools[anns[u]["lvis_category"]].append(u)
    partner = {}
    for u in q_uids:
        rng = random.Random(uid_seed(u) + 11)
        pool_ = [x for x in pools[anns[u]["lvis_category"]] if x != u] or [x for x in q_uids if x != u]
        partner[u] = rng.choice(pool_)
    print(f"{len(q_uids):,} queries vs {len(g_uids):,} gallery", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))   # released weights, frozen
    with torch.no_grad():
        g_text = np.stack([emb(u, "text") for u in g_uids])
        g_img = np.stack([emb(u, "views").mean(0) for u in g_uids])
        g_pc = encode_gallery_pc(bb, g_uids, tag="released")
        texts = {"own(attrs)": g_text[targets], "partner(attrs)": np.stack([emb(partner[u], "text") for u in q_uids])}
        sents = [serialize_annotation(anns[u], template="{category} {{size: {width} x {length} x {height} cm}}") for u in q_uids]
        texts["cat_size"] = np.concatenate([bb.encode_text(sents[i:i + 256]).float().cpu().numpy() for i in range(0, len(sents), 256)])
        images = {"own view": np.stack([emb(u, "views")[uid_seed(u) % 12] for u in q_uids]),
                  "partner view": np.stack([emb(partner[u], "views")[uid_seed(partner[u]) % 12] for u in q_uids])}
        if U2_FEATS.exists():
            z = np.load(U2_FEATS); row = {u: i for i, u in enumerate(z["uids"].tolist())}
            idx = np.array([row[u] for u in q_uids])
            images["thumbnail(own)"] = z["thumbnail_feat"][idx].astype(np.float32)
            texts["u2 blip caption"] = z["blip_feat"][idx].astype(np.float32)
        pcs = {"canonical": g_pc[targets]}
        pack = QueryPack(PACK, n_views=12)
        missing = [u for u in q_uids if u not in pack.rows["pc"]]
        if missing:
            raise SystemExit(f"query pack pc arm lacks {len(missing)} query uids")
        clouds = []
        for u in q_uids:
            v = np.asarray(pack.vector("pc", u), dtype=np.float32)
            clouds.append(perturb(v[:, :3], v[:, 3:6], "resample", uid_seed(u) + 7))
        pcs["resample"] = encode_clouds(bb, clouds)
        print(f"  resample paired cos {float((normalize_for_scoring(pcs['resample']) * normalize_for_scoring(g_pc[targets])).sum(1).mean()):.3f}", flush=True)

    combos = [("own(attrs)", "own view"), ("cat_size", "own view"), ("cat_size", "thumbnail(own)"),
              ("u2 blip caption", "thumbnail(own)"), ("partner(attrs)", "partner view")]
    combos = [(t, i) for t, i in combos if t in texts and i in images]
    out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "query_split": args.query_split,
           "gallery_split": args.gallery_split, "paper": PAPER, "paper_ulip": PAPER_ULIP, "rows": {}}
    for how in ("mean", "raw"):
        G = pool([g_text, g_img, g_pc], how)
        print(f"\n=== pooling = {how}; gallery = pooled(text, 12-view image, canonical pc), released ULIP-2")
        print(f"{'query (text | image | pc)':<58}" + "".join(f"{c:>9}" for c in QUERY_CONDITIONS))
        print(f"{'paper w/o ESSGNN':<58}" + "".join(f"{PAPER[c]:>9.1f}" for c in QUERY_CONDITIONS))
        print(f"{'paper ULIP baseline':<58}" + "".join(f"{PAPER_ULIP[c]:>9.1f}" for c in QUERY_CONDITIONS))
        for pol, q_pc in pcs.items():
            for tn, im in combos:
                parts = {"text": texts[tn], "image": images[im], "pc": q_pc}
                cells = {c: recall_at_k(pool([parts[m] for m in mods_of(c)], how) @ G.T, targets) for c in QUERY_CONDITIONS}
                key = f"{how} | {tn} | {im} | pc={pol}"
                out["rows"][key] = {"pooling": how, "text": tn, "image": im, "pc": pol, "cells": cells}
                print(f"{(tn + ' | ' + im + ' | pc=' + pol):<58}" + "".join(f"{cells[c]['R@1']*100:>9.1f}" for c in QUERY_CONDITIONS), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
