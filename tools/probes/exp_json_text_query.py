#!/usr/bin/env python3
"""Does the TEXT the paper's Figure 2 shows (a structured JSON-like annotation) retrieve anything
when it is fed to a CLIP text tower as a query?  DIAGNOSTIC (Kyzen / Claude 2026-09-07).

Why: MetaFind's Table 1 baseline rows are text-only 0.1-5.3 and image-only 0.1-2.3 for ULIP,
OpenShape, Uni3D and OmniBind, while pc-only is 98-99. Healthy released ULIP-2 on our val gives
attrs-sentence -> pc 50.6 and own-view -> pc 70.4 (exp_mean_pool_weak_trio --gallery pc). A text
query that is nearly useless for every pretrained model is what the JSON annotation would be after
CLIP's 77-token truncation (100% of ours exceed it, mean ~160 tokens). This probe measures
text-only R@1 for several query TEXT constructions against three galleries, all released ULIP-2,
no training:
    galleries   pc-only | fused mean of unit (text, 12-view image, pc) | text-only (identity check)
    query texts attrs sentence (= gallery text) | Figure-2 JSON string, tokenizer-truncated at 77
                | JSON without description | description only | category name only
Nothing here is a reproduction of the paper's protocol; it bounds what each text construction can do.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.models.resolve_stage1 import figure2_json_string
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER_BASELINES = {"ULIP": (0.1, 0.1, 97.9), "OpenShape": (0.6, 0.3, 98.4), "Uni3D": (1.7, 1.2, 98.3),
                   "OmniBind (Full)": (5.3, 2.3, 99.0), "MetaFind w/o ESSGNN": (13.8, 11.7, 75.1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/look/exp_json_text_query_val.json")
    args = ap.parse_args()
    t0 = time.time()

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])
    targets = np.arange(len(uids))
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in uids}

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    g_text = np.stack([emb(u, "text") for u in uids])
    g_img = np.stack([emb(u, "views").mean(0) for u in uids])

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))   # released, frozen
    cache = paths.OUTPUTS / "_probe" / f"released_pc_{args.split}.npz"
    if cache.exists() and np.load(cache)["uids"].tolist() == uids:
        g_pc = np.load(cache)["pc"].astype(np.float32)
        print(f"released pc vectors from cache {cache}", flush=True)
    else:
        from tools.probes.exp_query_pc_observation import encode_gallery_pc
        with torch.no_grad():
            g_pc = encode_gallery_pc(bb, uids, tag="released")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, uids=np.array(uids), pc=g_pc)
        print(f"released pc vectors cached -> {cache}", flush=True)

    unit = normalize_for_scoring
    galleries = {
        "pc-only": unit(g_pc),
        "fused mean(text,image,pc)": unit(np.mean([unit(g_text), unit(g_img), unit(g_pc)], axis=0)),
        "text-only (identity check)": unit(g_text),
    }

    def json_without_description(a):
        b = dict(a); b["description"] = ""
        return figure2_json_string(b)

    texts = {
        "attrs sentence (= gallery text)": None,     # cached vector, no re-encode
        "figure2 JSON (77-token truncated)": [figure2_json_string(anns[u]) for u in uids],
        "figure2 JSON without description": [json_without_description(anns[u]) for u in uids],
        "description only": [anns[u].get("description") or anns[u]["category"] for u in uids],
        "category name only": [anns[u]["category"] for u in uids],
    }
    tok = bb.tokenizer
    n_over = sum(1 for s in texts["figure2 JSON (77-token truncated)"] if len(tok.encode(s)) + 2 > 77) \
        if hasattr(tok, "encode") else None
    print(f"{len(uids):,} queries; JSON strings over 77 tokens: {n_over}", flush=True)

    out = {"n": len(uids), "split": args.split, "paper_baselines_text_image_pc": PAPER_BASELINES, "rows": {}}
    print(f"\n{'query text':<40}" + "".join(f"{g:>32}" for g in galleries))
    with torch.no_grad():
        for name, strings in texts.items():
            if strings is None:
                Q = unit(g_text)
            else:
                Q = np.concatenate([bb.encode_text(strings[i:i + 256]).float().cpu().numpy()
                                    for i in range(0, len(strings), 256)])
                Q = unit(Q)
            row = {}
            for gname, G in galleries.items():
                row[gname] = recall_at_k(Q @ G.T, targets)
            out["rows"][name] = {"cells": row, "example": (strings[0][:200] if strings else None)}
            print(f"{name:<40}" + "".join(f"{row[g]['R@1']*100:>14.1f} / {row[g]['R@5']*100:<14.1f}" for g in galleries), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
