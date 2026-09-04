#!/usr/bin/env python3
"""FUSION-CONVERGENCE LADDER, rung 0: mean-pooling fusion, NO training (released ULIP-2 vectors).
[KYZEN 「好」 2026-09-05 07:2x, on DL-098 item 3] Diagnostic only, never Table 1.
Query = mean of the L2-normalised AVAILABLE modality vectors (the paper's own baseline construction,
§3.1 "adding a simple mean pooling layer"); gallery = mean of all three (text, 12-view mean, pc).
Two query constructions: own observations (single view) and same-category partner text+image
(same rule as Stage1Dataset._build_partners: Random(uid_seed(uid)+11) over the val pool). val -> val.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
import numpy as np, torch
from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

CONDS = {"text": (1,0,0), "image": (0,1,0), "pc": (0,0,1), "text+image": (1,1,0),
         "text+pc": (1,0,1), "image+pc": (0,1,1), "full": (1,1,1)}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_fusion_ladder_rung0_mean.json")
    args = ap.parse_args()
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp["val"]); n = len(uids)
    emb = lambda u, k: np.load(paths.EMBEDDINGS / f"{u}.npz")[k].astype(np.float32)
    text = np.stack([emb(u, "text") for u in uids])
    views = [emb(u, "views") for u in uids]
    img_q = np.stack([v[uid_seed(u) % 12] for u, v in zip(uids, views)])
    img_g = np.stack([v.mean(0) for v in views])
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    clouds = [np.concatenate([(c := np.load(paths.POINTCLOUDS / f"{u}.npz"))["xyz"], c["rgb"]], 1).astype(np.float32) for u in uids]
    pcs = []
    with torch.no_grad():
        for i in range(0, n, 32):
            pcs.append(bb.encode_pc(torch.from_numpy(np.stack(clouds[i:i + 32]))).float().cpu().numpy())
    pc = np.concatenate(pcs)
    # partner: same LVIS category, never self, fixed per uid
    cat = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())["lvis_category"] for u in uids}
    pools = defaultdict(list)
    for u in uids: pools[cat[u]].append(u)
    partner = {}
    for u in uids:
        pool = [x for x in pools[cat[u]] if x != u] or [x for x in uids if x != u]
        partner[u] = random.Random(uid_seed(u) + 11).choice(pool)
    idx = {u: i for i, u in enumerate(uids)}
    p_text = np.stack([text[idx[partner[u]]] for u in uids])
    p_img = np.stack([views[idx[partner[u]]][uid_seed(partner[u]) % 12] for u in uids])
    N = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    gallery = normalize_for_scoring(N(text) + N(img_g) + N(pc))
    targets = np.arange(n)
    result = {"n_query": n, "n_gallery": n, "fusion": "mean of L2-normalised available vectors, untrained",
              "paper": {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2, "text+pc": 44.5, "image+pc": 45.8, "full": 51.7},
              "rows": {}}
    for name, (qt, qi) in {"own (single view)": (text, img_q), "partner text+image, pc own": (p_text, p_img)}.items():
        cells = {}
        for c, (a, b, d) in CONDS.items():
            q = a * N(qt) + b * N(qi) + d * N(pc)
            sim = normalize_for_scoring(q) @ gallery.T
            r = recall_at_k(sim, targets, (1, 5))
            cells[c] = {"R@1": float(r["R@1"]), "R@5": float(r["R@5"])}
        result["rows"][name] = cells
        print(f"{name:32s}", " ".join(f"{c}:{cells[c]['R@1']*100:5.1f}" for c in CONDS), flush=True)
    print("paper                           ", " ".join(f"{c}:{v:5.1f}" for c, v in result["paper"].items()))
    json.dump(result, open(args.out, "w"), indent=1); print("->", args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
