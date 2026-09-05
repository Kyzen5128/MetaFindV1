#!/usr/bin/env python3
"""Which query point-cloud observation moves the fused cells DOWN, as Table 1 has them?

[KYZEN 2026-09-04] "既然你選定 P1 那我覺得你要先達到 MetaFind w/o ESSGNN."

Every arm so far hands the query the gallery's own 10k cloud (paired cos 1.00),
or a second surface sample of the same mesh (cos 0.99). With that, adding text
or image to a pc query can only raise R@1 (ARMS_TABLE.md, ULIP row hypothesis),
while the paper's w/o-ESSGNN row FALLS: pc 75.1 -> text+pc 44.5, full 51.7.

This probe tries qualitatively different observations of the SAME asset for
the query cloud -- partial (one-sided) scan, no colour, jitter, sparse, rotated
-- and scores each two ways, without any training:

  A. released ULIP-2, plain mean of the available unit vectors, gallery = pc
     embedding: the ULIP row (paper 0.1/0.1/97.9/0.0/33.9/22.6/6.4).
  B. the P1 checkpoint (its PointBERT + its two fusion heads), gallery = the
     canonical record, query = attrs text + single view + the perturbed cloud:
     the w/o-ESSGNN row (paper 13.8/11.7/75.1/17.2/44.5/45.8/51.7).

B is a re-score of a model trained on the canonical cloud, so it bounds what a
model retrained on that observation would do; it does not equal it. Query text
and view are P1's own (canonical attrs_v1 text; view = uid_seed % 12), so the
cloud is the only thing that changes between rows.

Nothing here is written into the dataset: perturbed clouds live in memory only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,
                                     normalize_for_scoring, recall_at_k)
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER_ULIP = {"text": 0.1, "image": 0.1, "pc": 97.9, "text+image": 0.0,
              "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}
PAPER_WO = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2,
            "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}

POLICIES = ("canonical", "resample", "nocolor", "jitter02", "sparse1k", "rotz",
            "half", "half_nocolor", "half_sparse1k")


def perturb(xyz: np.ndarray, rgb: np.ndarray, policy: str, seed: int) -> np.ndarray:
    """(N_POINTS, 6). `canonical` and `resample` are the cloud as given; the
    rest go through the SAME function the trainer and evaluator apply
    (`metafind.data.observation.perturb_cloud`), so a row here is what a
    `--query-pc-perturb` run would feed the query tower."""
    from metafind.data.observation import perturb_cloud
    cloud = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    if policy in ("canonical", "resample"):
        return cloud
    return perturb_cloud(cloud, policy, seed)


def encode_clouds(bb, clouds, batch: int = 48) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(clouds), batch):
            c = torch.from_numpy(np.stack(clouds[i:i + batch]))
            out.append(bb.encode_pc(c).float().cpu().numpy())
    return np.concatenate(out)


def encode_gallery_pc(bb, uids, batch: int = 48, tag: str = "") -> np.ndarray:
    out, buf = [], []
    with torch.no_grad():
        for i, u in enumerate(uids):
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            buf.append(np.concatenate([c["xyz"], c["rgb"]], axis=1).astype(np.float32))
            if len(buf) == batch or i == len(uids) - 1:
                out.append(bb.encode_pc(torch.from_numpy(np.stack(buf))).float().cpu().numpy())
                buf = []
            if i % 6000 == 0:
                print(f"  {tag} gallery pc {i:,}/{len(uids):,}", flush=True)
    return np.concatenate(out)


def row(cells, paper=None) -> str:
    return "".join(f"{cells[c]['R@1']*100:>8.1f}" for c in QUERY_CONDITIONS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="P1 stage1_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--out", default="output/look/exp_query_pc_observation.json")
    ap.add_argument("--smoke", action="store_true",
                    help="DEBUG RUN: 128 dev_val queries vs a 512-asset dev_val gallery")
    args = ap.parse_args()
    policies = args.policies.split(",")
    for p in policies:
        if p not in POLICIES:
            raise SystemExit(f"unknown policy {p!r}; have {POLICIES}")

    from metafind.train.stage1 import QueryPack
    from tools.probes.exp_query_observation import load_tower

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp["train"]), sorted(sp["dev_val"])
    if args.smoke:
        g_uids, q_uids = q_uids[:512], q_uids[:128]
        print("SMOKE: debug run, not evidence", flush=True)
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    print(f"{len(q_uids):,} dev_val queries vs {len(g_uids):,} train gallery; "
          f"policies {policies}", flush=True)

    # the pack was built once under the base root; the text-template overlays
    # (METAFIND_DATA=..._attrs etc.) do not link `_probe`
    pack_json = paths.OUTPUTS / "_probe" / "query_pack" / "query_pack.json"
    if not pack_json.exists():
        pack_json = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/query_pack/query_pack.json")
    pack = QueryPack(pack_json, n_views=12)
    pack.require(q_uids)

    # ---- query clouds per policy (memory only) -----------------------------
    q_clouds = {}
    for pol in policies:
        cl = []
        for u in q_uids:
            if pol == "canonical":
                c = np.load(paths.POINTCLOUDS / f"{u}.npz")
                xyz, rgb = c["xyz"], c["rgb"]
            else:
                v = np.asarray(pack.vector("pc", u), dtype=np.float32)
                xyz, rgb = v[:, :3], v[:, 3:6]
            cl.append(perturb(xyz, rgb, pol, uid_seed(u) + 7))
        q_clouds[pol] = cl
    print("query clouds built", flush=True)

    # ---- text / image, shared by A and B (frozen CLIP; cached vectors) -----
    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)
    g_text = np.stack([emb(u, "text") for u in g_uids])
    g_img = np.stack([emb(u, "views").mean(0) for u in g_uids])
    q_img = np.stack([emb(u, "views")[uid_seed(u) % 12] for u in q_uids])
    q_text = g_text[targets]
    print("text/image loaded", flush=True)

    out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "policies": policies,
           "paper_ulip": PAPER_ULIP, "paper_wo_essgnn": PAPER_WO, "A": {}, "B": {}}

    # ---- A. released ULIP-2, plain mean, gallery = pc ------------------------
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    g_pc = encode_gallery_pc(bb, g_uids, tag="A")
    gpc = normalize_for_scoring(g_pc)
    print("\n=== A. released ULIP-2, mean of unit vectors, gallery = pc embedding ===")
    print(f"{'policy':<14}{'cos':>7}" + "".join(f"{c:>8}" for c in QUERY_CONDITIONS))
    print(f"{'paper ULIP':<14}{'':>7}" + "".join(f"{PAPER_ULIP[c]:>8.1f}" for c in QUERY_CONDITIONS))
    for pol in policies:
        q_pc = encode_clouds(bb, q_clouds[pol])
        qn = normalize_for_scoring(q_pc)
        paired = float((qn * gpc[targets]).sum(1).mean())
        mods = {"text": normalize_for_scoring(q_text), "image": normalize_for_scoring(q_img), "pc": qn}
        cells = {}
        for cond in QUERY_CONDITIONS:
            parts = [mods[m] for m in ("text", "image", "pc")
                     if m in cond.replace("full", "text+image+pc").split("+")]
            cells[cond] = recall_at_k(normalize_for_scoring(np.mean(parts, axis=0)) @ gpc.T, targets)
        out["A"][pol] = {"paired_cos": paired, "cells": cells}
        print(f"{pol:<14}{paired:>7.3f}" + row(cells), flush=True)
    del bb; torch.cuda.empty_cache()

    # ---- B. P1: its PointBERT and its two fusion heads ------------------------
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="pointbert_and_fuser"))
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bb.model.load_state_dict(ck["backbone_trainable_state"], strict=False)
    bb.model.eval()
    model = load_tower(Path(args.ckpt), args.device)
    print(f"\nP1 tower: prefusion_norm={getattr(model.query.cfg, 'prefusion_norm', '?')}"
          if hasattr(model, "query") and hasattr(model.query, "cfg") else "\nP1 tower loaded", flush=True)
    g_pc1 = encode_gallery_pc(bb, g_uids, tag="B")
    dev = args.device
    with torch.no_grad():
        G = []
        for i in range(0, len(g_uids), 512):
            s = slice(i, i + 512)
            e = {"text": torch.from_numpy(g_text[s]).to(dev),
                 "image": torch.from_numpy(g_img[s]).to(dev),
                 "pc": torch.from_numpy(g_pc1[s]).to(dev)}
            G.append(model.gallery(e).float().cpu())
        G = normalize_for_scoring(torch.cat(G).numpy())
    print("\n=== B. P1 checkpoint re-scored: query = attrs text + single view + this cloud ===")
    print(f"{'policy':<14}{'cos':>7}" + "".join(f"{c:>8}" for c in QUERY_CONDITIONS))
    print(f"{'paper w/o':<14}{'':>7}" + "".join(f"{PAPER_WO[c]:>8.1f}" for c in QUERY_CONDITIONS))
    for pol in policies:
        q_pc = encode_clouds(bb, q_clouds[pol])
        paired = float((normalize_for_scoring(q_pc) * normalize_for_scoring(g_pc1)[targets]).sum(1).mean())
        cells = {}
        with torch.no_grad():
            for cond in QUERY_CONDITIONS:
                Q = []
                for i in range(0, len(q_uids), 512):
                    s = slice(i, i + 512)
                    e = {"text": torch.from_numpy(q_text[s]).to(dev),
                         "image": torch.from_numpy(q_img[s]).to(dev),
                         "pc": torch.from_numpy(q_pc[s]).to(dev)}
                    m = condition_mask(cond, e["pc"].shape[0]).to(dev)
                    Q.append(model.query(e, present=m).float().cpu())
                cells[cond] = recall_at_k(normalize_for_scoring(torch.cat(Q).numpy()) @ G.T, targets)
        out["B"][pol] = {"paired_cos": paired, "cells": cells}
        print(f"{pol:<14}{paired:>7.3f}" + row(cells), flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
