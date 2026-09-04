#!/usr/bin/env python3
"""Type-level query text at TEST time on the P1 checkpoint (no training).

[KYZEN 2026-09-04 「好你先測吧」] 5i showed that scoring P1 with another
same-category asset's text+image flips the fused cells below pc (paper's
ordering) but drives text-only / image-only to ~0 (paper 13.8 / 11.7). The
paper's Figure 1 query text is `Platform Bed {size: ...}`: the TARGET's own
category and size, no description. So: query text = the target's own
type-level sentence built from the SAME annotation fields with a different
template (no materials, no placement, no description); image = a same-category
reference view (another asset) or the target's own view; pc = the target's own.

Gallery = P1's cached record (attrs_v1 text, 12-view mean, canonical pc via
P1's PointBERT). Parity rows reproduce the evaluator's numbers exactly.
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
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,
                                     normalize_for_scoring, recall_at_k)
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2, "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}
TEMPLATES = {
    "cat_size": "{category} {{size: {width} x {length} x {height} cm}}",   # Figure 1 form
    "cat_only": "{category}",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind_data_attrs/outputs/checkpoints/pilotP1_attrs_singleview_prefnorm_20260903/stage1_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gallery-split", default="train")
    ap.add_argument("--out", default="output/look/exp_type_level_query.json")
    args = ap.parse_args()
    from tools.probes.exp_query_observation import load_tower

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp["dev_val"])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in q_uids}
    # partner: same rule as Stage1Dataset._build_partners (same-category inside dev_val, Random(uid_seed+11))
    pools = defaultdict(list)
    for u in q_uids:
        pools[anns[u]["lvis_category"]].append(u)
    partner = {}
    for u in q_uids:
        rng = random.Random(uid_seed(u) + 11)
        pool = [x for x in pools[anns[u]["lvis_category"]] if x != u] or [x for x in q_uids if x != u]
        partner[u] = rng.choice(pool)
    print(f"{len(q_uids):,} queries vs {len(g_uids):,} gallery; {len(pools):,} categories", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="pointbert_and_fuser"))
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bb.model.load_state_dict(ck["backbone_trainable_state"], strict=False); bb.model.eval()
    model = load_tower(Path(args.ckpt), args.device)
    dev = args.device

    with torch.no_grad():
        # ---- query text variants
        texts = {"own(attrs)": np.stack([emb(u, "text") for u in q_uids]),
                 "partner(attrs)": np.stack([emb(partner[u], "text") for u in q_uids])}
        for name, tpl in TEMPLATES.items():
            sents = [serialize_annotation(anns[u], template=tpl) for u in q_uids]
            print(f"  {name} e.g. {sents[0]!r}", flush=True)
            vecs = []
            for i in range(0, len(sents), 256):
                vecs.append(bb.encode_text(sents[i:i + 256]).float().cpu().numpy())
            texts[name] = np.concatenate(vecs)
        images = {"own view": np.stack([emb(u, "views")[uid_seed(u) % 12] for u in q_uids]),
                  "partner view": np.stack([emb(partner[u], "views")[uid_seed(partner[u]) % 12] for u in q_uids])}
        # ---- gallery (P1's construction) and the query pc (= gallery pc, the asset's own)
        g_text = np.stack([emb(u, "text") for u in g_uids])
        g_img = np.stack([emb(u, "views").mean(0) for u in g_uids])
        g_pc, buf = [], []
        for i, u in enumerate(g_uids):
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            buf.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
            if len(buf) == 48 or i == len(g_uids) - 1:
                g_pc.append(bb.encode_pc(torch.from_numpy(np.stack(buf))).float().cpu().numpy()); buf = []
            if i % 6000 == 0:
                print(f"  gallery pc {i:,}/{len(g_uids):,}", flush=True)
        g_pc = np.concatenate(g_pc)
        G = []
        for i in range(0, len(g_uids), 512):
            s = slice(i, i + 512)
            G.append(model.gallery({"text": torch.from_numpy(g_text[s]).to(dev), "image": torch.from_numpy(g_img[s]).to(dev),
                                    "pc": torch.from_numpy(g_pc[s]).to(dev)}).float().cpu())
        G = normalize_for_scoring(torch.cat(G).numpy())
        q_pc = g_pc[targets]

        combos = [("own(attrs)", "own view", "parity: P1 as evaluated"),
                  ("partner(attrs)", "partner view", "parity: 5i (partner text+image)"),
                  ("cat_size", "partner view", "Figure-1 text of the TARGET + reference view"),
                  ("cat_only", "partner view", "category only + reference view"),
                  ("cat_size", "own view", "Figure-1 text + own view"),
                  ("cat_only", "own view", "category only + own view")]
        out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "paper": PAPER, "rows": {}}
        print(f"\n{'query (text | image); pc = own':<50}" + "".join(f"{c:>9}" for c in QUERY_CONDITIONS))
        print(f"{'paper w/o ESSGNN':<50}" + "".join(f"{PAPER[c]:>9.1f}" for c in QUERY_CONDITIONS))
        for tname, iname, label in combos:
            cells = {}
            for cond in QUERY_CONDITIONS:
                Q = []
                for i in range(0, len(q_uids), 512):
                    s = slice(i, i + 512)
                    e = {"text": torch.from_numpy(texts[tname][s]).to(dev), "image": torch.from_numpy(images[iname][s]).to(dev),
                         "pc": torch.from_numpy(q_pc[s]).to(dev)}
                    Q.append(model.query(e, present=condition_mask(cond, e["pc"].shape[0]).to(dev)).float().cpu())
                cells[cond] = recall_at_k(normalize_for_scoring(torch.cat(Q).numpy()) @ G.T, targets)
            out["rows"][f"{tname} | {iname}"] = {"label": label, "cells": cells}
            print(f"{(tname + ' | ' + iname):<50}" + "".join(f"{cells[c]['R@1']*100:>9.1f}" for c in QUERY_CONDITIONS) + f"   {label}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
