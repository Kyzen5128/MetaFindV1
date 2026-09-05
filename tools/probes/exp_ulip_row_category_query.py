#!/usr/bin/env python3
"""Can the paper's ULIP row be produced by CATEGORY-LEVEL query text and image?

Paper Table 1, ULIP row (no fusion, no training): text 0.1, image 0.1, pc 97.9,
T+I 0.0, T+PC 33.9, I+PC 22.6, full 6.4. Text-only and image-only at 0.1 are
chance level: the paper's query text and image do NOT identify the instance.
Figure 1 draws the text query as `Platform Bed {size: ...}` -- a category name
plus fields. So the hypothesis: the query's text and image describe the
CATEGORY (or another asset of it), only the point cloud is the asset's own.

Released ULIP-2, plain mean of unit vectors, gallery = pc embedding of dev_val
(4,569). Query variants, all seeded per uid:
  text : own attrs sentence | category name only
  image: own single view    | one view of ANOTHER asset in the same LVIS category
  pc   : own canonical cloud | the pack's resampled cloud
No training, nothing written to the dataset.
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
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER_ULIP = {"text": 0.1, "image": 0.1, "pc": 97.9, "text+image": 0.0, "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_ulip_row_category_query.json")
    args = ap.parse_args()

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp["dev_val"])
    n = len(uids); targets = np.arange(n)
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in uids}
    cat = {u: anns[u]["lvis_category"] for u in uids}
    by_cat = defaultdict(list)
    for u in uids:
        by_cat[cat[u]].append(u)
    print(f"{n:,} dev_val assets, {len(by_cat):,} categories present", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    with torch.no_grad():
        # text: own sentence (cached, attrs_v1 or whatever the overlay holds) and category-only
        t_own = np.stack([emb(u, "text") for u in uids])
        names = sorted(by_cat)
        cvec = {}
        for i in range(0, len(names), 128):
            chunk = names[i:i + 128]
            v = bb.encode_text([c.replace("_", " ") for c in chunk]).float().cpu().numpy()
            cvec.update({c: v[j] for j, c in enumerate(chunk)})
        t_cat = np.stack([cvec[cat[u]] for u in uids])
        # image: own single view and another same-category asset's view
        i_own, i_other, t_other, i_rand, t_rand = [], [], [], [], []
        for u in uids:
            rng = random.Random(uid_seed(u) + 11)
            v = emb(u, "views"); i_own.append(v[uid_seed(u) % 12])
            pool = [x for x in by_cat[cat[u]] if x != u]
            o = rng.choice(pool) if pool else u
            i_other.append(emb(o, "views")[uid_seed(o) % 12])
            t_other.append(emb(o, "text"))               # the OTHER asset's own sentence
            r = rng.choice(uids)                          # a RANDOM asset, any category
            i_rand.append(emb(r, "views")[uid_seed(r) % 12]); t_rand.append(emb(r, "text"))
        i_own, i_other, t_other = np.stack(i_own), np.stack(i_other), np.stack(t_other)
        i_rand, t_rand = np.stack(i_rand), np.stack(t_rand)
        g_img = np.stack([emb(u, "views").mean(0) for u in uids])   # gallery image = 12-view mean
        # pc: own canonical (gallery) and the pack's resampled cloud
        clouds = []
        for u in uids:
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            clouds.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
        def enc(cl):
            out = []
            for i in range(0, len(cl), 32):
                out.append(bb.encode_pc(torch.from_numpy(np.stack(cl[i:i + 32]))).float().cpu().numpy())
            return np.concatenate(out)
        print("encoding gallery pc", flush=True)
        g_pc = enc(clouds)
        pack_json = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/query_pack/query_pack.json")
        from metafind.train.stage1 import QueryPack
        pack = QueryPack(pack_json, n_views=12); pack.require(uids)
        print("encoding resampled query pc", flush=True)
        q_pc_res = enc([np.asarray(pack.vector("pc", u), dtype=np.float32) for u in uids])

    # [PAPER 3.1] baselines: "a simple mean pooling layer to aggregate available modalities, and use
    # these fused embeddings to retrieve from a pre-encoded gallery"; [PAPER 3.2] their PC-only uses
    # "identical embeddings for both query and gallery". The gallery's own construction is not stated,
    # so two readings are scored: gallery = pc vector, and gallery = mean of the asset's three vectors.
    galleries = {
        "gallery=pc": normalize_for_scoring(g_pc),
        "gallery=mean3(L2)": normalize_for_scoring(np.mean([normalize_for_scoring(t_own), normalize_for_scoring(g_img), normalize_for_scoring(g_pc)], axis=0)),
        "gallery=mean3(raw)": normalize_for_scoring(np.mean([t_own, g_img, g_pc], axis=0)),
    }
    variants = {
        "own text | own view | own pc": (t_own, i_own, g_pc),
        "category text | own view | own pc": (t_cat, i_own, g_pc),
        "category text | OTHER asset view | own pc": (t_cat, i_other, g_pc),
        "OTHER asset text | OTHER asset view | own pc": (t_other, i_other, g_pc),
        "OTHER asset text | OTHER asset view | resampled pc": (t_other, i_other, q_pc_res),
        "RANDOM asset text | RANDOM asset view | own pc": (t_rand, i_rand, g_pc),
        "OTHER asset text | RANDOM asset view | own pc": (t_other, i_rand, g_pc),
    }
    out = {"n": n, "paper_ulip": PAPER_ULIP, "rows": {}}
    for gname, G in galleries.items():
        print(f"\n=== {gname} ===")
        print(f"{'variant':<52}" + "".join(f"{c:>9}" for c in QUERY_CONDITIONS))
        print(f"{'paper ULIP row':<52}" + "".join(f"{PAPER_ULIP[c]:>9.1f}" for c in QUERY_CONDITIONS))
        for name, (t, im, pc) in variants.items():
            for mode in ("L2", "raw"):
                if mode == "L2":
                    mods = {"text": normalize_for_scoring(t), "image": normalize_for_scoring(im), "pc": normalize_for_scoring(pc)}
                else:
                    mods = {"text": t.astype(np.float64), "image": im.astype(np.float64), "pc": pc.astype(np.float64)}
                cells = {}
                for cond in QUERY_CONDITIONS:
                    parts = [mods[m] for m in ("text", "image", "pc") if m in cond.replace("full", "text+image+pc").split("+")]
                    cells[cond] = recall_at_k(normalize_for_scoring(np.mean(parts, axis=0)) @ G.T, targets)
                out["rows"][f"{gname} | {name} | {mode}mean"] = cells
                print(f"{name + ' | ' + mode:<52}" + "".join(f"{cells[c]['R@1']*100:>9.1f}" for c in QUERY_CONDITIONS), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
