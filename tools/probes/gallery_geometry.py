#!/usr/bin/env python3
"""Where does the promoted gallery vector actually sit?

Kyzen, 2026-09-01, on the reported protocol: 「查」.

The paper's Table 1 has a shape our runs have never produced. pc alone is its
best cell at 75.1, and EVERY combination that adds a modality to pc is worse --
text+pc 44.5, image+pc 45.8, full 51.7. Ours goes the other way: pc 93.7 and
full 99.6 on protocol B.

One construction produces the paper's shape without any further assumption. If
the gallery vector lies close to the point-cloud embedding and far from text and
image, then a pc query is near-unimodal retrieval and scores high; a text or
image query is genuinely cross-modal and scores low; and mean-pooling a strong
pc with two weak others lands between them, below pc alone. `diag_modality_norms`
already measured pc at 72% of the trained gallery's unnormalised norm, which is
what suggested it, but that was the parameter-free mean of raw vectors -- not
the gallery TOWER's output, and not the index Table 1 actually retrieves from.

This measures the promoted index itself: for each asset, the cosine between its
stored gallery vector and its own raw text, image and pc embeddings, plus the
same after removing the across-asset mean, because a large shared component
inflates every raw cosine equally and says nothing about which modality the
vector follows per asset.

If the gallery follows pc, the paper's shape is explained and the remaining
question becomes why ours does not. If it is balanced, that reading is dead and
the shape needs another account.

Reads the index and the n06 sidecars. Encodes the point cloud through the
checkpoint's own backbone, because the index was built that way and the raw
released-encoder cloud is a different vector.
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

OUT = REPO / "output" / "look" / "gallery_geometry.json"
MODS = ("text", "image", "pc")


def cosines(A: torch.Tensor, B: torch.Tensor) -> dict:
    a = torch.nn.functional.normalize(A, dim=-1)
    b = torch.nn.functional.normalize(B, dim=-1)
    v = (a * b).sum(1)
    return {"mean": round(float(v.mean()), 4),
            "p5": round(float(v.quantile(0.05)), 4),
            "p95": round(float(v.quantile(0.95)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=None,
                    help="gallery_index_*.npz; the newest is used if omitted")
    ap.add_argument("--ckpt", default=None,
                    help="stage1_best.pt whose backbone encoded the index's clouds")
    ap.add_argument("-n", type=int, default=3000, help="assets to sample")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    idx_path = pathlib.Path(args.index) if args.index else sorted(
        paths.OUTPUTS.glob("gallery_index_*.npz"))[-1]
    z = np.load(idx_path, allow_pickle=True)
    print(f"index {idx_path.name}   keys {z.files}", flush=True)
    ids = [str(u) for u in z["ids"]]
    V = z["vectors"] if "vectors" in z.files else z["embeddings"]
    print(f"  {len(ids):,} assets, gallery vector {V.shape[1]}-d", flush=True)

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)
    uids = [ids[i] for i in sel]
    G = torch.from_numpy(V[sel].astype(np.float32))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    scope = "point_encoder_and_fuser"
    if args.ckpt:
        scope = torch.load(args.ckpt, map_location="cpu",
                           weights_only=False).get("train_scope", scope)
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope=scope))
    if args.ckpt:
        from metafind.train.stage1 import (build_model, load_protocols,
                                           load_stage1_checkpoint)
        enc, tr, hy = load_protocols()
        m, lf = build_model(enc, tr, hy)
        m.to(dev)
        load_stage1_checkpoint(bb, m, lf, pathlib.Path(args.ckpt))
        print(f"  loaded {args.ckpt} (train_scope {scope})", flush=True)
    else:
        print("  ⚠ no --ckpt: the pc row uses the RELEASED encoder, which is "
              "NOT what built the index unless the index is an untrained one",
              flush=True)

    raw = {"text": [], "image": []}
    clouds = []
    for u in uids:
        c = np.load(paths.EMBEDDINGS / f"{u}.npz")
        raw["text"].append(c["text"].astype(np.float32))
        raw["image"].append(c["image"].astype(np.float32))
        p = np.load(paths.POINTCLOUDS / f"{u}.npz")
        clouds.append(np.concatenate([p["xyz"], p["rgb"]], 1).astype(np.float32))
    pc = np.empty((len(uids), G.shape[1]), np.float32)
    with torch.no_grad():
        for i in range(0, len(uids), args.batch):
            j = min(i + args.batch, len(uids))
            pc[i:j] = bb.encode_pc(np.stack(clouds[i:j])).float().cpu().numpy()
    R = {"text": torch.from_numpy(np.stack(raw["text"])),
         "image": torch.from_numpy(np.stack(raw["image"])),
         "pc": torch.from_numpy(pc)}

    res = {"index": idx_path.name, "n": len(uids), "checkpoint": args.ckpt,
           "train_scope": scope, "raw_cosine": {}, "centred_cosine": {},
           "norms": {}}
    print(f"\n{'':<10s}{'cos(gallery, 這個模態)':>26s}{'去掉共同成分後':>18s}"
          f"{'‖模態‖':>12s}")
    Gc = G - G.mean(0, keepdim=True)
    for m in MODS:
        c = cosines(G, R[m])
        cc = cosines(Gc, R[m] - R[m].mean(0, keepdim=True))
        n = float(torch.linalg.norm(R[m], dim=1).mean())
        res["raw_cosine"][m] = c
        res["centred_cosine"][m] = cc
        res["norms"][m] = round(n, 2)
        print(f"{m:<10s}{c['mean']:26.4f}{cc['mean']:18.4f}{n:12.2f}")
    res["norms"]["gallery"] = round(float(torch.linalg.norm(G, dim=1).mean()), 2)
    print(f"{'gallery':<10s}{'':>26s}{'':>18s}{res['norms']['gallery']:12.2f}")

    lead = max(MODS, key=lambda m: res["centred_cosine"][m]["mean"])
    rest = [m for m in MODS if m != lead]
    gap = (res["centred_cosine"][lead]["mean"]
           - max(res["centred_cosine"][m]["mean"] for m in rest))
    res["closest_modality"] = lead
    res["lead_over_next"] = round(gap, 4)
    print(f"\n畫廊向量最靠近: {lead}，領先第二名 {gap:+.4f}（去共同成分後）")
    print("論文形狀需要 pc 明顯領先；平衡的話這個讀法就死了")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
