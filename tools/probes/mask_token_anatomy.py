#!/usr/bin/env python3
"""What did the mask tokens learn, and how much of a masked query is constant?

Kyzen, 2026-09-01: 「查」.

`fusion.py:188-200` fills an absent slot with a learned per-modality vector:

    fill = self.mask_tokens[i].expand_as(e)
    cols.append(torch.where(keep, e, fill))

and `include_absent_slots=True` keeps every slot in the pooled readout with
denom fixed at 3. So a text-only query is

    mean( head(e_text), head(mask_img), head(mask_pc) )

and two of those three terms are THE SAME VECTOR FOR EVERY ASSET. They are the
last part of the 23.63M-parameter query tower nobody has looked at.

Four things, and the last two are the point:

A  NORMS.  How big is each mask token beside the real embeddings it stands in
   for. A token much larger than the modality it replaces would dominate the
   mean regardless of what the present modality says.

B  DIRECTION.  cos(mask token, the across-asset mean of that modality's real
   embeddings). Near 1 would mean the token learned "the average asset" -- a
   prior standing in for the missing observation.

C  THE EMPTY QUERY.  `allow_all_masked=True`, so 2.7% of training queries lose
   all three modalities at p=0.3. Such a query is a pure constant and carries
   ZERO information about which asset it came from, so its R@1 must be chance,
   1/45,692 = 0.002%. Anything above that is a defect, and this is the cheapest
   place it would show.

D  SIGNAL AGAINST CONSTANT.  Per Table-1 condition, decompose the query output
   over assets into its across-asset mean (which cannot discriminate anything)
   and the residual (which is all the per-asset information there is). The
   ratio says how much of a single-modality query is actually the mask tokens.

Nothing is trained, nothing is written but one JSON. Uses the promoted index's
own checkpoint so the tower measured is the tower Table 1 was scored with.
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

OUT = REPO / "output" / "look" / "mask_token_anatomy.json"
MODS = ("text", "image", "pc")
CONDS = {"text": (1, 0, 0), "image": (0, 1, 0), "pc": (0, 0, 1),
         "text+image": (1, 1, 0), "text+pc": (1, 0, 1),
         "image+pc": (0, 1, 1), "full": (1, 1, 1), "none": (0, 0, 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind/metafind_out/checkpoints/"
                                      "qpack_ti_lr2.50e-04_s20260816/stage1_best.pt")
    ap.add_argument("--index", default=None)
    ap.add_argument("-n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    idx = pathlib.Path(args.index) if args.index else sorted(
        paths.OUTPUTS.glob("gallery_index_*.npz"))[-1]
    z = np.load(idx, allow_pickle=True)
    ids = [str(u) for u in z["ids"]]
    V = z["embeddings"]
    pos = {u: i for i, u in enumerate(ids)}
    print(f"index {idx.name}  {len(ids):,} assets", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scope = torch.load(args.ckpt, map_location="cpu",
                       weights_only=False).get("train_scope",
                                               "point_encoder_and_fuser")
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import (build_model, load_protocols,
                                       load_stage1_checkpoint)
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope=scope))
    enc, tr, hy = load_protocols()
    model, lf = build_model(enc, tr, hy)
    model.to(dev)
    load_stage1_checkpoint(bb, model, lf, pathlib.Path(args.ckpt))
    model.eval()
    print(f"loaded {args.ckpt} (train_scope {scope})", flush=True)

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)
    uids = [ids[i] for i in sel]

    txt, img, clouds = [], [], []
    for u in uids:
        c = np.load(paths.EMBEDDINGS / f"{u}.npz")
        txt.append(c["text"].astype(np.float32))
        img.append(c["image"].astype(np.float32))
        p = np.load(paths.POINTCLOUDS / f"{u}.npz")
        clouds.append(np.concatenate([p["xyz"], p["rgb"]], 1).astype(np.float32))
    T = torch.from_numpy(np.stack(txt)).to(dev)
    I = torch.from_numpy(np.stack(img)).to(dev)
    P = torch.empty(len(uids), T.shape[1], device=dev)
    with torch.no_grad():
        for i in range(0, len(uids), args.batch):
            j = min(i + args.batch, len(uids))
            P[i:j] = bb.encode_pc(np.stack(clouds[i:j])).float()
    RAW = {"text": T, "image": I, "pc": P}

    tokens = model.query.fusion.mask_tokens.detach()
    res = {"index": idx.name, "checkpoint": args.ckpt, "n": len(uids),
           "A_norms": {}, "B_direction": {}, "C_empty_query": {},
           "D_signal_vs_constant": {}}

    print(f"\n=== A 大小 ===\n{'':<8s}{'mask token':>13s}{'真實向量':>12s}{'比值':>9s}")
    for i, m in enumerate(MODS):
        nt = float(torch.linalg.norm(tokens[i]))
        nr = float(torch.linalg.norm(RAW[m], dim=1).mean())
        res["A_norms"][m] = {"mask_token": round(nt, 3),
                             "real_mean": round(nr, 3),
                             "ratio": round(nt / nr, 4)}
        print(f"{m:<8s}{nt:13.3f}{nr:12.3f}{nt / nr:9.4f}")

    print(f"\n=== B 方向：mask token 有沒有變成「平均資產」===")
    for i, m in enumerate(MODS):
        mu = RAW[m].mean(0)
        c = float(torch.nn.functional.cosine_similarity(
            tokens[i].flatten(), mu.flatten(), dim=0))
        res["B_direction"][m] = round(c, 4)
        print(f"  cos(mask_{m}, 該模態的跨資產平均) = {c:+.4f}")

    G = torch.nn.functional.normalize(
        torch.from_numpy(V.astype(np.float32)).to(dev), dim=-1)
    tgt = torch.tensor([pos[u] for u in uids], device=dev)

    def query_out(mask):
        present = torch.tensor(mask, dtype=torch.bool, device=dev) \
                       .expand(len(uids), -1).clone()
        with torch.no_grad():
            return model.query(RAW, present=present)

    def r1(q):
        q = torch.nn.functional.normalize(q, dim=-1)
        s = q @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        return ((s > own).sum(1) < 1).sum().item() / len(uids) * 100

    print(f"\n=== C 全遮的查詢（必須等於亂猜）===")
    qn = query_out(CONDS["none"])
    spread = float((qn - qn.mean(0)).norm(dim=1).mean())
    chance = 100.0 / len(ids)
    r_none = r1(qn)
    res["C_empty_query"] = {"R@1": round(r_none, 5),
                            "chance": round(chance, 5),
                            "residual_norm": round(spread, 6),
                            "verdict": "OK" if r_none <= max(chance * 5, 0.05)
                                       else "!! 高於亂猜"}
    print(f"  R@1 {r_none:.5f}%   亂猜 {chance:.5f}%   "
          f"跨資產殘差 {spread:.6f}   {res['C_empty_query']['verdict']}")

    print(f"\n=== D 訊號 vs 常數 ===")
    print(f"{'條件':<12s}{'‖常數‖':>10s}{'‖訊號‖':>10s}{'訊號佔比':>10s}{'R@1':>9s}")
    for cond, mask in CONDS.items():
        if cond == "none":
            continue
        q = query_out(mask)
        mu = q.mean(0)
        resid = q - mu
        nc = float(torch.linalg.norm(mu))
        ns = float(resid.norm(dim=1).mean())
        a1 = r1(q)
        res["D_signal_vs_constant"][cond] = {
            "constant_norm": round(nc, 3), "signal_norm": round(ns, 3),
            "signal_share": round(ns / (ns + nc), 4), "R@1": round(a1, 2)}
        print(f"{cond:<12s}{nc:10.3f}{ns:10.3f}{ns / (ns + nc):10.4f}{a1:9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
