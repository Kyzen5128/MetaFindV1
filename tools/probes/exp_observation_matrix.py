#!/usr/bin/env python3
"""Every observation the corpus can express, crossed, measured against Table 1.

[KYZEN 2026-09-03] "列出所有可能性 ... 用實作去驗證". Not one axis at a time and
not a hypothesis -- the whole cross, with the paper's own numbers beside it.

THE THREE AXES, and why these are all of them
---------------------------------------------
The paper leaves three things unstated that this corpus CAN vary today without
re-rendering, re-annotating or retraining:

  text     canonical   the query reads the gallery's own text vector, byte for
                       byte -- what every number before 2026-09-03 was measured
                       under, and the reason the text column never moved in the
                       image-only experiment
           alternate   `description_candidates[1]`: a second sampled description
                       of the same asset, encoded through the same frozen tower.
                       A genuinely different sentence about the same object.

  image    same_mean       the gallery's own 12-view mean
           single_view     one view, by uid_seed
           disjoint_views  the mean of the other eleven -- no view shared with
                           the gallery at all

  gallery  dev_val   4,569
           full      45,692, the whole admitted corpus, same queries

A fourth axis -- a second point-cloud SAMPLE -- is absent because it does not
exist on disk; n03 wrote one sample per asset. A fifth -- which eleven cameras --
needs a re-render. Both are named so the cross is not mistaken for exhaustive.

WHAT THIS CANNOT SETTLE
-----------------------
Every cell is ONE checkpoint, trained under `same_record`, evaluated another
way. §十六: that is inference sensitivity and it cannot stand in for training
under a different observation. It bounds how much of the gap the OBSERVATION
can explain; it does not say what the authors did.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.observation import view_indices

# MetaFind w/o ESSGNN, 3experiments.tex:45, R@1 in percent.
PAPER = {"text": 13.8, "image": 11.7, "text+image": 17.2}
TEXT_POLICIES = ("canonical", "alternate")
IMAGE_POLICIES = ("same_mean", "single_view", "disjoint_views")
CONDITIONS = {"text": (True, False), "image": (False, True),
              "text+image": (True, True)}


def alt_text_vectors(uids: list[str], rank: int, device: str,
                     cache: Path) -> tuple[torch.Tensor, int]:
    """Encode the rank-N caption once and keep it; the tower is the slow part."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["uids"]) == uids:
            return torch.from_numpy(z["vecs"]).to(device), int(z["n_alt"])

    from metafind.models.resolve_stage1 import serialize_annotation
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    texts, n_alt = [], 0
    for uid in uids:
        a = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
        c = a.get("description_candidates") or []
        if rank < len(c) and c[rank].get("text"):
            # An asset with no candidate at this rank keeps its canonical text
            # and is COUNTED, never dropped: 111 assets have fewer than five
            # and 20 have exactly one, so dropping them would change the pool
            # and make the rows incomparable.
            texts.append(serialize_annotation({**a, "description": c[rank]["text"]}))
            n_alt += 1
        else:
            texts.append(serialize_annotation(a))

    bb = ULIPBackbone(BackboneConfig(device=device, train_scope="fuser_only"))
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            out.append(bb.encode_text(texts[i:i + 64]).float().cpu().numpy())
    vecs = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, uids=np.array(uids), vecs=vecs, n_alt=n_alt)
    return torch.from_numpy(vecs).to(device), n_alt


def query_images(uids: list[str], policy: str, device: str) -> torch.Tensor:
    img = []
    for uid in uids:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        if policy == "same_mean":
            # The stored fp16 mean, not a fp32 recomputation of it -- same bytes
            # or it is a different arm from every number measured before.
            img.append(z["image"].astype(np.float32))
        else:
            v = z["views"].astype(np.float32)
            img.append(v[view_indices(policy, uid, v.shape[0])].mean(axis=0))
    return torch.from_numpy(np.stack(img)).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_observation_matrix.json")
    args = ap.parse_args()

    from tools.probes.exp_query_observation import (
        gallery_vectors,
        load_tower,
        recall,
    )

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])
    full = sorted(set(sp["train"]) | set(sp["test"]))
    model = load_tower(Path(args.ckpt), args.device)

    canon = torch.from_numpy(np.stack(
        [np.load(paths.EMBEDDINGS / f"{u}.npz")["text"].astype(np.float32)
         for u in uids])).to(args.device)
    alt, n_alt = alt_text_vectors(
        uids, args.rank, args.device,
        Path(f"/tmp/claude-1002/alt_text_rank{args.rank}.npz"))
    texts = {"canonical": canon, "alternate": alt}
    images = {p: query_images(uids, p, args.device) for p in IMAGE_POLICIES}

    print(f"{len(uids):,} queries. {n_alt:,} have a rank-{args.rank} caption; "
          "the rest keep canonical and are counted, not dropped.")
    print(f"paper (w/o ESSGNN): text {PAPER['text']} · image {PAPER['image']} "
          f"· text+image {PAPER['text+image']}\n")

    rows, out = [], {}
    for g_name, g_uids in (("dev_val", uids), ("full", full)):
        where = {u: i for i, u in enumerate(g_uids)}
        target = torch.tensor([where[u] for u in uids]).to(args.device)
        gal = gallery_vectors(g_uids, args.device)
        for tp, ip in itertools.product(TEXT_POLICIES, IMAGE_POLICIES):
            cells = {}
            for cond, present in CONDITIONS.items():
                r1, r5 = recall(model, texts[tp], images[ip], gal, present,
                                args.device, target)
                cells[cond] = {"R@1": r1, "R@5": r5}
            key = f"{g_name}|{tp}|{ip}"
            out[key] = {"n_gallery": len(g_uids), **cells}
            rows.append((g_name, len(g_uids), tp, ip, cells))
            print(f"  {key}", flush=True)

    print("\n" + "gallery".ljust(9) + "text".ljust(11) + "image".ljust(16)
          + "".join(c.rjust(20) for c in CONDITIONS))
    for g, n, tp, ip, cells in rows:
        line = f"{g:<9}{tp:<11}{ip:<16}"
        for c in CONDITIONS:
            v = cells[c]["R@1"] * 100
            line += ("%.1f  (%.1fx)" % (v, v / PAPER[c])).rjust(20)
        print(line)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt, "n_query": len(uids), "rank": args.rank,
        "n_with_alternate": n_alt, "paper_w_o_essgnn_R@1_percent": PAPER,
        "axes_not_varied": {
            "point_cloud_observation": "a second sample does not exist on disk; "
                                       "n03 wrote one per asset",
            "camera_layout": "needs a re-render, which the spec defers"},
        "caveat": "one checkpoint trained under same_record, evaluated many "
                  "ways. Inference sensitivity, not a protocol comparison "
                  "(§十六).",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
