#!/usr/bin/env python3
"""The ULIP row under the four cheap questions GPT put first (via Kyzen, 2026-09-03).

Released ULIP-2, no training. Gallery = the PC embedding of the 36,554 train
assets (reading B1). Query pc = the gallery's OWN cloud, as the paper says the
baselines' PC-only does ("identical embeddings for both query and gallery").

  P0-A  scorer 2x2: query fusion {raw mean, per-modality L2 then mean}
        x final scorer {cosine, raw dot}. Seven cells each. sec. 2.1 writes
        only sim(.,.); every production number so far is cosine.
  P0-B  image observation: 12-view mean vs the uid-seeded single view, under
        both scorers.
  P0-C  PC-only calibration: identical pc gives 100.0 under cosine here and
        97.9 in the paper. Under dot, own = |g|^2 and a larger-norm neighbour
        can win. Also how many gallery clouds have a near-duplicate.
  P0-D  margin decomposition, unit vectors: for query i with own pc p_i and
        competitor j, T+PC flips iff  t_i.(p_j - p_i) > 1 - p_i.p_j.
        Reported as the share of queries any competitor can flip, per
        modality and for full, with the margin and advantage percentiles.

Text arms: the corpus's canonical string (v2_cm template, cached) and
category-only (re-encoded). Output: output/look/exp_ulip_scorer_margin.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.eval.retrieval import QUERY_CONDITIONS, normalize_for_scoring, recall_at_k
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER = dict(zip(QUERY_CONDITIONS, (0.1, 0.1, 97.9, 0.0, 33.9, 22.6, 6.4)))
MODS = {"text": ("text",), "image": ("image",), "pc": ("pc",),
        "text+image": ("text", "image"), "text+pc": ("text", "pc"),
        "image+pc": ("image", "pc"), "full": ("text", "image", "pc")}


def encode_pc(bb, uids, batch=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(uids), batch):
            cl = []
            for u in uids[i:i + batch]:
                c = np.load(paths.POINTCLOUDS / f"{u}.npz")
                cl.append(np.concatenate([c["xyz"], c["rgb"]], axis=1).astype(np.float32))
            out.append(bb.encode_pc(torch.from_numpy(np.stack(cl))).float().cpu().numpy())
            if i % (batch * 200) == 0:
                print(f"    pc {i:,}/{len(uids):,}", flush=True)
    return np.concatenate(out)


def seven(q_mods: dict, gal: np.ndarray, targets: np.ndarray, fusion: str, scorer: str) -> dict:
    """q_mods: raw modality vectors of the queries. gal: raw gallery pc."""
    mods = dict(q_mods)
    if fusion == "l2mean":
        mods = {k: normalize_for_scoring(v) for k, v in mods.items()}
    g = normalize_for_scoring(gal) if scorer == "cosine" else gal.astype(np.float64)
    cells = {}
    for cond, parts in MODS.items():
        q = np.mean([mods[m] for m in parts], axis=0)
        q = normalize_for_scoring(q) if scorer == "cosine" else q.astype(np.float64)
        cells[cond] = recall_at_k(q @ g.T, targets)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_ulip_scorer_margin.json")
    args = ap.parse_args()

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))

    print(f"encoding {len(g_uids):,} gallery clouds with the released PointBERT", flush=True)
    g_pc = encode_pc(bb, g_uids)
    q_pc = g_pc[targets]                                   # IDENTICAL, as the paper says
    q_text = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["text"] for u in q_uids]).astype(np.float32)
    q_img_mean = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["image"] for u in q_uids]).astype(np.float32)
    from metafind.data.observation import view_indices
    q_img_single = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"][view_indices("single_view", u, 12)[0]]
                             for u in q_uids]).astype(np.float32)
    cats = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())["category"].strip() + "." for u in q_uids]
    with torch.no_grad():
        q_text_cat = np.concatenate([bb.encode_text(cats[i:i + 256]).float().cpu().numpy()
                                     for i in range(0, len(cats), 256)])

    out = {"n_gallery": len(g_uids), "n_query": len(q_uids), "tables": {}, "calibration": {}, "margin": {}}
    hdr = "".join(c.rjust(11) for c in QUERY_CONDITIONS)
    print("\n=== P0-A/B: ULIP row, gallery = PC, query pc IDENTICAL ===")
    print("paper".ljust(34) + "".join(f"{PAPER[c]:>11.1f}" for c in QUERY_CONDITIONS))
    for text_name, tvec in (("v2_template", q_text), ("category_only", q_text_cat)):
        for img_name, ivec in (("mean12", q_img_mean), ("single", q_img_single)):
            for fusion in ("rawmean", "l2mean"):
                for scorer in ("cosine", "dot"):
                    cells = seven({"text": tvec, "image": ivec, "pc": q_pc}, g_pc, targets, fusion, scorer)
                    key = f"{text_name}|{img_name}|{fusion}|{scorer}"
                    out["tables"][key] = cells
                    print(key.ljust(34) + "".join(f"{cells[c]['R@1']*100:>11.1f}" for c in QUERY_CONDITIONS), flush=True)

    # ---- P0-C: PC-only calibration ---------------------------------------
    gn = normalize_for_scoring(g_pc)
    S = gn[targets] @ gn.T
    S[np.arange(len(targets)), targets] = -np.inf
    nn_cos = S.max(1)
    norms = np.linalg.norm(g_pc, axis=1)
    out["calibration"] = {
        "pc_only_cosine_R@1": out["tables"]["v2_template|mean12|rawmean|cosine"]["pc"]["R@1"],
        "pc_only_dot_R@1": out["tables"]["v2_template|mean12|rawmean|dot"]["pc"]["R@1"],
        "nearest_other_cos_p50": float(np.percentile(nn_cos, 50)),
        "nearest_other_cos_p90": float(np.percentile(nn_cos, 90)),
        "share_nearest_other_cos_gt_0.98": float((nn_cos > 0.98).mean()),
        "share_nearest_other_cos_gt_0.99": float((nn_cos > 0.99).mean()),
        "gallery_pc_norm_p10_p50_p90": [float(np.percentile(norms, p)) for p in (10, 50, 90)],
    }
    print("\n=== P0-C: PC-only calibration (paper 97.9) ===")
    for k, v in out["calibration"].items():
        print(f"  {k}: {v}")

    # ---- P0-D: margin decomposition on unit vectors ----------------------
    print("\n=== P0-D: can text / image flip the pc top-1 while the pc is identical? (unit vectors) ===")
    p = gn[targets]                                        # own pc, unit
    margin = 1.0 - nn_cos                                  # 1 - max_j p_i.p_j
    res = {}
    for text_name, tvec in (("v2_template", q_text), ("category_only", q_text_cat)):
        for img_name, ivec in (("mean12", q_img_mean), ("single", q_img_single)):
            t = normalize_for_scoring(tvec); v = normalize_for_scoring(ivec)
            # advantage of the best competitor: max_j [x.(p_j - p_i)] with the
            # flip condition x.(p_j - p_i) > 1 - p_i.p_j checked per j
            def flips(x):
                Sx = x @ gn.T                                   # x_i . p_j
                own = (x * p).sum(1, keepdims=True)
                Sp = gn[targets] @ gn.T                         # p_i . p_j
                cond = (Sx - own) > (1.0 - Sp)
                cond[np.arange(len(targets)), targets] = False
                adv = Sx - own
                adv[np.arange(len(targets)), targets] = -np.inf
                return float(cond.any(1).mean()), adv.max(1)
            ft, at = flips(t); fi, ai = flips(v); ftv, atv = flips(t + v)
            key = f"{text_name}|{img_name}"
            res[key] = {"share_flip_text": ft, "share_flip_image": fi, "share_flip_text+image": ftv,
                        "text_adv_p50": float(np.percentile(at, 50)), "text_adv_p90": float(np.percentile(at, 90)),
                        "image_adv_p50": float(np.percentile(ai, 50)), "image_adv_p90": float(np.percentile(ai, 90))}
            print(f"  {key:<26} flip by text {ft*100:5.1f}%  by image {fi*100:5.1f}%  by text+image {ftv*100:5.1f}%"
                  f"   | best text adv p50/p90 {np.percentile(at,50):+.3f}/{np.percentile(at,90):+.3f}"
                  f"   image adv p50/p90 {np.percentile(ai,50):+.3f}/{np.percentile(ai,90):+.3f}")
    out["margin"] = res
    out["margin"]["pc_margin_p10_p50_p90"] = [float(np.percentile(margin, q)) for q in (10, 50, 90)]
    print(f"  pc top-1 margin (1 - nearest other cos) p10/p50/p90: "
          + " / ".join(f"{np.percentile(margin, q):.3f}" for q in (10, 50, 90)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
