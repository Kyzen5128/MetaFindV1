#!/usr/bin/env python3
"""Does the SOURCE resolution of the query render move the image-only score?

Kyzen 2026-09-06: 「我的 image 分數比較高會不會是因為我的畫素比較高?」

The image tower is OpenCLIP ViT-bigG-14 at 224 x 224: every render is resized to
224 before encoding, whatever its source resolution (open_clip preprocess). So a
512 px render only differs from a low-res one through resampling quality. This
probe takes N val assets, encodes ONE own view at its native 512 px and after
downsampling to 128 / 64 px (then the same 224 preprocess), and scores each
against the cached 12-view-mean gallery of the whole val split with the frozen
CLIP tower (no fusion) -- the raw image->image identity that the Table 1 image
cell rests on. Gallery vectors are untouched; only the query's source pixels move.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.data.view_io import load_view_rgb
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resolutions", default="512,128,64")
    ap.add_argument("--out", default="output/look/exp_query_image_resolution_val.json")
    args = ap.parse_args()
    res = [int(r) for r in args.resolutions.split(",")]

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids = sorted(sp["dev_val"])
    rng = np.random.default_rng(20260906)
    q_uids = sorted(rng.choice(g_uids, size=args.n, replace=False).tolist())
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    G = normalize_for_scoring(np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"].mean(0) for u in g_uids]).astype(np.float32))
    print(f"{len(q_uids)} queries vs {len(g_uids):,} gallery (cached 12-view means)", flush=True)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    renders = {}
    for line in (paths.LOGS / "renders_index.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); renders[r["uid"]] = r
    out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "rows": {}}
    with torch.no_grad():
        for r in res:
            feats, t0 = [], time.time()
            for u in q_uids:
                vp = renders[u]["view_paths"][uid_seed(u) % len(renders[u]["view_paths"])]
                img = load_view_rgb(vp)                       # composited RGB, native 512
                if r != img.size[0]:
                    img = img.resize((r, r), Image.BILINEAR)  # lose pixels ...
                x = bb.preprocess(img).unsqueeze(0).to(args.device)   # ... then the tower's own 224 resize
                feats.append(bb.encode_image(x).float().cpu().numpy()[0])
            Q = normalize_for_scoring(np.stack(feats))
            cells = recall_at_k(Q @ G.T, targets)
            paired = float((Q * G[targets]).sum(1).mean())
            out["rows"][str(r)] = {"cells": cells, "paired_cos": paired, "seconds": time.time() - t0}
            print(f"  source {r:>3} px -> R@1 {cells['R@1']*100:5.1f}  R@5 {cells['R@5']*100:5.1f}  "
                  f"paired cos {paired:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    from pathlib import Path
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
