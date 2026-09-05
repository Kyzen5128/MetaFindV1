#!/usr/bin/env python3
"""Table 1 "w/ ESSGNN" row, second reading: the Stage 2 head WITH the ESSGNN fed a lone node.

Kyzen 2026-09-06 「2.」: besides scoring the Stage 2 shared head with the layout term absent
(paper: Objaverse-LVIS "lacks layout and disables ESSGNN"), score it with each Objaverse
asset pushed through ESSGNN as a graph of ONE node and no edges, so Eq. 6's
lambda * e_layout is present. With E = 0 the EGCL layers are the identity and the pooled
layout vector is a function of the node feature alone (normalised_sum pooling -> unit norm).

Node feature: the same construction Stage 2 used for ProcTHOR nodes
(`procthor_object_text.json`: "a {category}", frozen ULIP-2 / OpenCLIP text tower), here
"a {lvis_category}" of the asset itself. That is the asset's CATEGORY, not its identity;
still, the node is the target, so this variant is flagged as such.

Gallery, query constructions and parity rows follow `exp_type_level_query.py`; the
layout-absent rows must reproduce `table1_final_P1s_S2head_holdout.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import QUERY_CONDITIONS, condition_mask, normalize_for_scoring, recall_at_k
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
from metafind.train.stage1 import build_model, load_protocols, load_stage1_checkpoint
from metafind.train.stage2 import Stage2Data, build_stage2_model, load_stage2_protocols

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2, "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}
PAPER_W = {"text": 11.3, "image": 10.5, "pc": 63.2, "text+image": 15.9, "text+pc": 41.2, "image+pc": 42.0, "full": 48.2}
U2_FEATS = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/ulip2_query_feats/ulip2_query_feats.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt-record", required=True)
    ap.add_argument("--stage2-record", required=True, help="the arm's variant_ckpts.json")
    ap.add_argument("--query-split", default="holdout")
    ap.add_argument("--gallery-split", default="holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_stage2_head_lone_node_holdout.json")
    args = ap.parse_args()
    dev = args.device

    ckpt = json.loads(Path(args.stage1_ckpt_record).read_text())
    s2 = json.loads(Path(args.stage2_record).read_text())["full"]
    if s2["stage1_checkpoint_sha256"] != ckpt["sha256"]:
        raise SystemExit("Stage 2 record's parent is not this Stage 1 checkpoint")
    encoding, training, hyper = load_protocols()
    _s2p, _edge, arch = load_stage2_protocols()
    data = Stage2Data(dev)

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in q_uids}
    print(f"{len(q_uids):,} queries vs {len(g_uids):,} gallery", flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=dev, train_scope="fuser_only"))
    model = build_stage2_model(encoding, training, hyper, arch, node_feat_dim=data.node_dim,
                               edge_feat_dim=data.edge_dim, use_layout=True,
                               init_lambda=float(s2["lambda_init"]["init_lambda"])).to(dev)
    _, loss_fn = build_model(encoding, training, hyper)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "query.layout_weight"))
    s2_state = torch.load(s2["uri"], map_location=dev, weights_only=False)["trainable_state"]
    missing, unexpected = model.load_state_dict(s2_state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in the Stage 2 state: {unexpected[:4]}")
    model.eval()
    lam = float(model.query.layout_weight.item())
    print(f"Stage 2 state loaded ({len(s2_state)} tensors); lambda = {lam:.3f}", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    with torch.no_grad():
        # ---- gallery: own text, 12-view mean, canonical cloud through this checkpoint's Point-BERT
        g_text = np.stack([emb(u, "text") for u in g_uids])
        g_img = np.stack([emb(u, "views").mean(0) for u in g_uids])
        g_pc, buf = [], []
        for i, u in enumerate(g_uids):
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            buf.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
            if len(buf) == 48 or i == len(g_uids) - 1:
                g_pc.append(backbone.encode_pc(torch.from_numpy(np.stack(buf))).float().cpu().numpy()); buf = []
            if i % 3000 == 0:
                print(f"  gallery pc {i:,}/{len(g_uids):,}", flush=True)
        g_pc = np.concatenate(g_pc)
        G = []
        for i in range(0, len(g_uids), 512):
            s = slice(i, i + 512)
            G.append(model.gallery({"text": torch.from_numpy(g_text[s]).to(dev), "image": torch.from_numpy(g_img[s]).to(dev),
                                    "pc": torch.from_numpy(g_pc[s]).to(dev)}).float().cpu())
        G = normalize_for_scoring(torch.cat(G).numpy())
        q_pc = g_pc[targets]

        # ---- query text / image variants
        texts = {"own(attrs)": g_text[targets]}
        sents = [serialize_annotation(anns[u], template="{category} {{size: {width} x {length} x {height} cm}}") for u in q_uids]
        texts["cat_size"] = np.concatenate([backbone.encode_text(sents[i:i + 256]).float().cpu().numpy() for i in range(0, len(sents), 256)])
        images = {"own view": np.stack([emb(u, "views")[uid_seed(u) % 12] for u in q_uids])}
        if U2_FEATS.exists():
            z = np.load(U2_FEATS); row = {u: i for i, u in enumerate(z["uids"].tolist())}
            images["thumbnail(own)"] = z["thumbnail_feat"][np.array([row[u] for u in q_uids])].astype(np.float32)

        # ---- lone-node layout: node text "a {category}" as Stage 2 built ProcTHOR nodes
        node_sents = ["a " + anns[u]["lvis_category"].replace("_", " ") for u in q_uids]
        print(f"  node text e.g. {node_sents[0]!r}; ProcTHOR nodes were built the same way ('a counter top')", flush=True)
        node_feat = np.concatenate([backbone.encode_text(node_sents[i:i + 256]).float().cpu().numpy() for i in range(0, len(node_sents), 256)])
        if node_feat.shape[1] != data.node_dim:
            raise SystemExit(f"node feature dim {node_feat.shape[1]} != Stage 2 node_dim {data.node_dim}")
        no_edges = torch.zeros(2, 0, dtype=torch.long, device=dev)
        no_attr = torch.zeros(0, data.edge_dim, device=dev)
        no_missing = torch.zeros(0, dtype=torch.bool, device=dev)
        layouts = []
        for i in range(len(q_uids)):
            layouts.append(model.query.encode_layout(
                torch.from_numpy(node_feat[i:i + 1]).to(dev), torch.zeros(1, 3, device=dev),
                no_edges, no_attr, edge_missing=no_missing).float().cpu())
        layouts = torch.cat(layouts)               # (Q, D), unit norm each under normalised_sum pooling
        print(f"  lone-node e_layout norm mean {float(layouts.norm(dim=1).mean()):.3f}; lambda*|e_layout| = {lam * float(layouts.norm(dim=1).mean()):.1f}", flush=True)

        combos = [("own(attrs)", "own view", "own observations"),
                  ("cat_size", "thumbnail(own)", "weak own: category+size, own thumbnail")]
        combos = [c for c in combos if c[1] in images]
        out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "query_split": args.query_split,
               "gallery_split": args.gallery_split, "lambda": lam, "paper_wo": PAPER, "paper_w": PAPER_W, "rows": {}}
        print(f"\n{'query (text | image); pc = own':<44}{'layout':<10}" + "".join(f"{c:>9}" for c in QUERY_CONDITIONS))
        print(f"{'paper w/o ESSGNN':<54}" + "".join(f"{PAPER[c]:>9.1f}" for c in QUERY_CONDITIONS))
        print(f"{'paper w/ ESSGNN':<54}" + "".join(f"{PAPER_W[c]:>9.1f}" for c in QUERY_CONDITIONS))
        fused_norm = None
        for tname, iname, label in combos:
            for mode in ("absent", "lone-node"):
                cells = {}
                for cond in QUERY_CONDITIONS:
                    Q = []
                    for i in range(0, len(q_uids), 512):
                        s = slice(i, i + 512)
                        e = {"text": torch.from_numpy(texts[tname][s]).to(dev), "image": torch.from_numpy(images[iname][s]).to(dev),
                             "pc": torch.from_numpy(q_pc[s]).to(dev)}
                        lay = layouts[s].to(dev) if mode == "lone-node" else None
                        q = model.query(e, present=condition_mask(cond, e["pc"].shape[0]).to(dev), layout=lay).float().cpu()
                        if fused_norm is None and mode == "absent" and cond == "full":
                            fused_norm = float(q.norm(dim=1).mean())
                        Q.append(q)
                    cells[cond] = recall_at_k(normalize_for_scoring(torch.cat(Q).numpy()) @ G.T, targets)
                key = f"{tname} | {iname} | layout={mode}"
                out["rows"][key] = {"label": label, "layout": mode, "cells": cells}
                print(f"{(tname + ' | ' + iname):<44}{mode:<10}" + "".join(f"{cells[c]['R@1']*100:>9.1f}" for c in QUERY_CONDITIONS), flush=True)
        out["norm_fused_full_query"] = fused_norm
        print(f"\n|Fusion(full query)| mean {fused_norm:.1f} vs lambda*|e_layout| {lam:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
