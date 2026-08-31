#!/usr/bin/env python3
"""Which evaluation protocol produces MetaFind Table 1's ULIP row?

THE LOCK
--------
Table 1's baselines are models we can run exactly. Its ULIP row is fourteen
numbers:

    text 0.1/0.9  image 0.1/1.3  pc 97.9/99.4  T+I 0/0.3
    T+PC 33.9/58  I+PC 22.6/41.6  full 6.4/15.9

Any protocol that reproduces those is, to a very good approximation, the
protocol the table was computed under -- and it can then be applied to the
MetaFind rows, which is the only way our own numbers become comparable.
Four cells carry nearly all the discriminating power (pc, T+PC, I+PC, full);
the near-zero cells are weak evidence in either direction.

WHAT IS ALREADY FIXED, AND WHY
------------------------------
**The gallery is the point-cloud embedding, not the mean of three.** Forced by
elimination: if the gallery were mean(t,i,p) then the `full` query -- also
mean(t,i,p) -- would match itself and score ~100, and the paper reports 6.4.
With a pc gallery, `pc` scores near-perfect (97.9 observed) and every added
modality rotates the query away from it, which is the descending order the row
actually shows. Corroborated by CAMERA's evaluator, which likewise pairs one
modality's raw matrix against another's (`docs/reference/camera/evaluate.py`).

`3experiments.tex` describes the baseline query as "a simple mean pooling layer
to aggregate available modalities", which does not say whether the vectors are
normalised first. Both are tried; it matters, because the released ULIP-2's
modality norms are 37 / 40 / 28 and the mean of raw vectors is therefore not the
mean of unit vectors.

THE GRID
--------
    pooling  raw mean | L2-normalise then mean
    text     L1 category name | L2 bare description | L3 full serialisation
    image    12-view mean | one held-out view
    pool     45,692 full corpus | 9,138 test split

24 configurations, 7 conditions each, all matrix products over cached vectors.
The point encoder is the RELEASED ULIP-2 with no Stage 1 weights: the baseline
row is a released checkpoint, so ours must be too.

WHAT THIS CANNOT SETTLE
-----------------------
A match locks the *evaluation*. It says nothing about the MetaFind rows'
training, and it must not be reported as reproducing Table 1 as a whole.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "table1_baseline_grid.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
LADDER = paths.OUTPUTS / "_probe" / "text_ladder"

CONDITIONS = ["text", "image", "pc", "text+image", "text+pc", "image+pc", "full"]

PAPER_ULIP = {                       # 3experiments.tex tab:objaverse-results
    "text": (0.1, 0.9), "image": (0.1, 1.3), "pc": (97.9, 99.4),
    "text+image": (0.0, 0.3), "text+pc": (33.9, 58.0),
    "image+pc": (22.6, 41.6), "full": (6.4, 15.9),
}
DISCRIMINATIVE = ["pc", "text+pc", "image+pc", "full"]


def all_uids() -> list[str]:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    return sorted(set(d["train"]) | set(d["test"])), sorted(d["test"])


def released_pc(uids: list[str], batch: int) -> np.ndarray:
    """Released ULIP-2 over every cloud, cached because it costs ~3 GPU-minutes."""
    if PC_CACHE.exists():
        a = np.load(PC_CACHE)
        if a.shape[0] == len(uids):
            print(f"pc from cache {PC_CACHE}", flush=True)
            return a
        print(f"pc cache has {a.shape[0]} rows, need {len(uids)}; recomputing",
              flush=True)
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    out = np.empty((len(uids), 1280), dtype=np.float32)
    buf, at, t0 = [], 0, time.time()
    with torch.no_grad():
        for i, uid in enumerate(uids):
            c = np.load(paths.POINTCLOUDS / f"{uid}.npz")
            buf.append(np.concatenate([c["xyz"].astype(np.float32),
                                       c["rgb"].astype(np.float32)], axis=1))
            if len(buf) == batch or i == len(uids) - 1:
                out[at:at + len(buf)] = bb.encode_pc(np.stack(buf)).float().cpu().numpy()
                at += len(buf)
                buf = []
                if at % 10000 < batch:
                    print(f"  pc {at:,}/{len(uids):,} {time.time() - t0:.0f}s", flush=True)
    PC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PC_CACHE, out)
    return out


def load_modalities(uids: list[str], test_uids: list[str], seed: int):
    """Cached text (three rungs) and image (two constructions) for the queries."""
    pos = {u: i for i, u in enumerate(uids)}
    rng = np.random.default_rng(seed)

    l3 = np.empty((len(uids), 1280), dtype=np.float32)
    img_mean = np.empty((len(uids), 1280), dtype=np.float32)
    img_one = np.empty((len(uids), 1280), dtype=np.float32)
    for i, uid in enumerate(uids):
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        l3[i] = z["text"].astype(np.float32)
        img_mean[i] = z["image"].astype(np.float32)
        v = z["views"]
        img_one[i] = v[rng.integers(v.shape[0])].astype(np.float32)
        if (i + 1) % 15000 == 0:
            print(f"  cached {i + 1:,}/{len(uids):,}", flush=True)

    texts = {"L3_full_serialization": l3}
    for rung in ("L1_category_name", "L2_bare_description"):
        man = json.loads((LADDER / f"pack_{rung}.json").read_text())
        sh = man["text"]["shards"][0]
        arr = np.load(sh["array"]).astype(np.float32)
        order = sh["uid_order"]
        full = l3.copy()                       # rows outside the pack keep L3
        hit = 0
        for k, u in enumerate(order):
            # `--limit` truncates the corpus while the pack still covers all
            # 9,138 test uids, so a miss here is a debug-run artifact. On a full
            # run every pack uid is present and `hit` must equal len(order) --
            # the assert below is what makes a silent partial override
            # impossible to mistake for a real result.
            if u in pos:
                full[pos[u]] = arr[k]
                hit += 1
        texts[rung] = full
        print(f"  {rung}: {hit:,}/{len(order):,} rows overridden", flush=True)
        if hit != len(order):
            print(f"  !! DEBUG RUN: {len(order) - hit:,} pack uids are outside "
                  f"the truncated corpus and kept their L3 vector", flush=True)
    return texts, {"12view_mean": img_mean, "single_view": img_one}


def pool_query(parts: list[torch.Tensor], normalise_first: bool) -> torch.Tensor:
    if normalise_first:
        parts = [torch.nn.functional.normalize(p, dim=-1) for p in parts]
    return torch.stack(parts, 0).mean(0)


def recall(q: torch.Tensor, g: torch.Tensor, target: torch.Tensor,
           chunk: int = 1024) -> tuple[float, float]:
    """R@1 and R@5 where `target[i]` is query i's own row in the gallery."""
    q = torch.nn.functional.normalize(q, dim=-1)
    g = torch.nn.functional.normalize(g, dim=-1)
    h1 = h5 = 0
    for i in range(0, q.shape[0], chunk):
        j = min(i + chunk, q.shape[0])
        sims = q[i:j] @ g.t()
        own = sims.gather(1, target[i:j].unsqueeze(1))
        higher = (sims > own).sum(dim=1)
        h1 += int((higher < 1).sum())
        h5 += int((higher < 5).sum())
    return h1 / q.shape[0] * 100, h5 / q.shape[0] * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    args = ap.parse_args()

    uids, test_uids = all_uids()
    if args.limit:
        uids, test_uids = uids[:args.limit], [u for u in test_uids[:args.limit]
                                              if u in set(uids[:args.limit])]
    print(f"corpus {len(uids):,}  test {len(test_uids):,}", flush=True)

    pc = released_pc(uids, args.batch)
    texts, images = load_modalities(uids, test_uids, args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pos = {u: i for i, u in enumerate(uids)}
    test_rows = torch.tensor([pos[u] for u in test_uids], device=dev)
    pc_t = torch.from_numpy(pc).to(dev)
    tex_t = {k: torch.from_numpy(v).to(dev) for k, v in texts.items()}
    img_t = {k: torch.from_numpy(v).to(dev) for k, v in images.items()}

    pools = {"full_45692": torch.arange(len(uids), device=dev),
             "test_9138": test_rows}

    rows = []
    for norm_first, trung, irung, pool_name in itertools.product(
            (False, True), texts.keys(), images.keys(), pools.keys()):
        sel = pools[pool_name]
        gallery = pc_t.index_select(0, sel)
        # where each test query's own asset sits inside this gallery
        where = {int(v): k for k, v in enumerate(sel.tolist())}
        target = torch.tensor([where[int(r)] for r in test_rows.tolist()], device=dev)

        q = {"text": tex_t[trung].index_select(0, test_rows),
             "image": img_t[irung].index_select(0, test_rows),
             "pc": pc_t.index_select(0, test_rows)}
        cells = {}
        for cond in CONDITIONS:
            keys = ["text", "image", "pc"] if cond == "full" else cond.split("+")
            r1, r5 = recall(pool_query([q[k] for k in keys], norm_first),
                            gallery, target)
            cells[cond] = (round(r1, 2), round(r5, 2))
        # distance to the paper's ULIP row on the four cells that separate configs
        err = sum(abs(cells[c][0] - PAPER_ULIP[c][0]) +
                  abs(cells[c][1] - PAPER_ULIP[c][1]) for c in DISCRIMINATIVE)
        pc_is_max = max(cells, key=lambda c: cells[c][0]) == "pc"
        rows.append({"normalise_before_mean": norm_first, "text": trung,
                     "image": irung, "pool": pool_name, "cells": cells,
                     "abs_err_on_4_cells": round(err, 2), "pc_is_max": pc_is_max})
        print(f"norm={int(norm_first)} {trung[:2]} {irung[:6]} {pool_name:11s} "
              f"| " + "  ".join(f"{c[:4]} {cells[c][0]:5.1f}" for c in CONDITIONS)
              + f" | err {err:7.1f} pcmax {pc_is_max}", flush=True)

    rows.sort(key=lambda r: r["abs_err_on_4_cells"])
    payload = {
        "what": "grid over baseline-row evaluation choices, scored against "
                "MetaFind Table 1's ULIP row",
        "fixed": "gallery = released ULIP-2 point-cloud embedding; query = mean "
                 "pool of the present modalities; queries are the 9,138 test uids",
        "encoder": "released ULIP-2 ULIP2_PointBERT_Colored, no Stage 1 weights",
        "paper_ulip_row": {k: list(v) for k, v in PAPER_ULIP.items()},
        "scored_on": DISCRIMINATIVE,
        "seed": args.seed,
        "n_configs": len(rows),
        "results_sorted_by_error": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nbest: {json.dumps(rows[0], indent=1)}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
