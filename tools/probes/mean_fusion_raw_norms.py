#!/usr/bin/env python3
"""Rung-0 companion: does UNnormalised mean pooling reproduce the paper's BASELINE rows?
Paper §3.1: baselines = pre-trained encoder + "a simple mean pooling layer to aggregate available
modalities", retrieving "from a pre-encoded gallery"; §3.2: their PC-only uses "identical embeddings
for both query and gallery". Here: released ULIP-2 vectors, val -> val, query = mean of the RAW
(unnormalised) available vectors, gallery = (a) the pc vector alone, (b) raw mean of the three.
Also prints the per-modality norms, which decide who dominates a raw mean."""
from __future__ import annotations
import json, random
from collections import defaultdict
import numpy as np, torch
from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
CONDS = {"text": (1,0,0), "image": (0,1,0), "pc": (0,0,1), "text+image": (1,1,0), "text+pc": (1,0,1), "image+pc": (0,1,1), "full": (1,1,1)}
sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]; uids = sorted(sp["val"]); n = len(uids)
emb = lambda u, k: np.load(paths.EMBEDDINGS / f"{u}.npz")[k].astype(np.float32)
text = np.stack([emb(u, "text") for u in uids]); views = [emb(u, "views") for u in uids]
img_q = np.stack([v[uid_seed(u) % 12] for u, v in zip(uids, views)]); img_g = np.stack([v.mean(0) for v in views])
bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
clouds = [np.concatenate([(c := np.load(paths.POINTCLOUDS / f"{u}.npz"))["xyz"], c["rgb"]], 1).astype(np.float32) for u in uids]
with torch.no_grad():
    pc = np.concatenate([bb.encode_pc(torch.from_numpy(np.stack(clouds[i:i+32]))).float().cpu().numpy() for i in range(0, n, 32)])
nrm = lambda a: float(np.linalg.norm(a, axis=1).mean())
print(f"raw norms  text {nrm(text):.2f}  image(single) {nrm(img_q):.2f}  image(12-mean) {nrm(img_g):.2f}  pc {nrm(pc):.2f}", flush=True)
cat = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())["lvis_category"] for u in uids}
pools = defaultdict(list); [pools[cat[u]].append(u) for u in uids]
partner = {u: random.Random(uid_seed(u) + 11).choice([x for x in pools[cat[u]] if x != u] or [x for x in uids if x != u]) for u in uids}
idx = {u: i for i, u in enumerate(uids)}
p_text = np.stack([text[idx[partner[u]]] for u in uids]); p_img = np.stack([views[idx[partner[u]]][uid_seed(partner[u]) % 12] for u in uids])
targets = np.arange(n); out = {"paper_baseline_ULIP": [0.1, 0.1, 97.9, 0, 33.9, 22.6, 6.4], "paper_metafind": [13.8, 11.7, 75.1, 17.2, 44.5, 45.8, 51.7], "rows": {}}
for gname, gal in {"gallery=pc only": pc, "gallery=raw mean(3)": (text + img_g + pc) / 3}.items():
    G = normalize_for_scoring(gal)
    for qname, (qt, qi) in {"own": (text, img_q), "partner": (p_text, p_img)}.items():
        cells = {}
        for c, (a, b, d) in CONDS.items():
            q = (a * qt + b * qi + d * pc) / (a + b + d)          # RAW mean, no per-modality normalisation
            r = recall_at_k(normalize_for_scoring(q) @ G.T, targets, (1, 5)); cells[c] = float(r["R@1"]) * 100
        out["rows"][f"{gname} | query {qname}"] = cells
        print(f"{gname:22s} query {qname:8s}", " ".join(f"{c}:{cells[c]:5.1f}" for c in CONDS), flush=True)
print("paper ULIP baseline          ", " ".join(f"{c}:{v:5.1f}" for c, v in zip(CONDS, out["paper_baseline_ULIP"])))
json.dump(out, open("output/look/exp_fusion_ladder_rung0_raw.json", "w"), indent=1)
