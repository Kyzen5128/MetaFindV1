#!/usr/bin/env python3
"""CAMERA's six-way object-level retrieval, run on OUR corpus with released ULIP-2.

WHY THIS EXISTS
---------------
Kyzen supplied CAMERA's evaluator (`evaluate.py`, and a 2025-08-12 working note
carrying its output verbatim). It is a THIRD protocol, independent of both
ULIP's zero-shot classification and MetaFind's Table 1, and it comes with a
published reference point. Running it on our data tells us whether our
text/image/pc alignment behaves like theirs under a protocol neither of us
designed.

WHAT CAMERA'S PROTOCOL IS (from their code, confirmed by their own analysis)
----------------------------------------------------------------------------
One point cloud, one caption, one rendered view per object. Encode all three,
L2-normalise each, then use the SAME three matrices as both query and gallery
in six pairings: S2T, T2S, S2I, I2S, T2I, I2T. Positives are the entries whose
`taxonomy_id-model_id` matches the query's; the rank is the BEST rank among
them. Metrics: MRR, R@1, R@5, R@10, mean/median rank, NDCG@5.

TWO DIFFERENCES FROM CAMERA, BOTH RECORDED RATHER THAN PAPERED OVER
-------------------------------------------------------------------
1. **Positives: one on both sides, so this is NOT a difference.** [CORRECTED
   2026-08-31] The first draft of this file recorded CAMERA as having ~5
   positives per query, from 75,361 captions over 15,033 models, and called our
   single-positive setting "stricter". That is wrong. Their loader emits ONE row
   per object -- `dataset_3d.py.__getitem__` draws one caption at random -- so
   `obj2idxs` holds a single index per object and the best-rank-over-positives
   machinery is a no-op. Their RR@1 is therefore plain single-positive instance
   retrieval, exactly like ours, and 13.50 IS comparable to our T2S modulo pool
   size and backbone. Do not reinstate the "stricter" caveat.
2. **Pool size.** Theirs is 14,966; our corpus is 45,692. Both are reported --
   `full` and a seeded 14,966 subset -- so the pool is controlled rather than
   argued about.

The point encoder is the RELEASED ULIP-2, no Stage 1 checkpoint: this measures
the backbone we start from, not anything we trained.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "camera_six_way.json"

PAIRS = [
    ("S2T", "pc", "text"),
    ("T2S", "text", "pc"),
    ("S2I", "pc", "image"),
    ("I2S", "image", "pc"),
    ("T2I", "text", "image"),
    ("I2T", "image", "text"),
]


def load_uids() -> list[str]:
    """Every admitted asset, train and test together.

    CAMERA evaluates over `train.txt + test.txt` merged (`whole=True`), so the
    matching choice here is the whole corpus rather than one split.
    """
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(set(d["train"]) | set(d["test"]))
    return uids


def load_text_image(uids: list[str], image_mode: str, seed: int):
    """Cached OpenCLIP vectors. `image_mode` is `mean` (12 views) or `single`."""
    rng = np.random.default_rng(seed)
    txt = np.empty((len(uids), 1280), dtype=np.float32)
    img = np.empty((len(uids), 1280), dtype=np.float32)
    for i, uid in enumerate(uids):
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        txt[i] = z["text"].astype(np.float32)
        if image_mode == "mean":
            img[i] = z["image"].astype(np.float32)
        else:
            v = z["views"]
            img[i] = v[rng.integers(v.shape[0])].astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"  text/image {i + 1:,}/{len(uids):,}", flush=True)
    return txt, img


def encode_pc(uids: list[str], batch: int) -> np.ndarray:
    """Released ULIP-2 PointBERT over every cloud. No Stage 1 weights."""
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    out = np.empty((len(uids), 1280), dtype=np.float32)
    buf: list[np.ndarray] = []
    at = 0
    t0 = time.time()
    with torch.no_grad():
        for i, uid in enumerate(uids):
            c = np.load(paths.POINTCLOUDS / f"{uid}.npz")
            xyz = c["xyz"].astype(np.float32)
            rgb = c["rgb"].astype(np.float32)
            buf.append(np.concatenate([xyz, rgb], axis=1))
            if len(buf) == batch or i == len(uids) - 1:
                e = bb.encode_pc(np.stack(buf)).float().cpu().numpy()
                out[at:at + len(buf)] = e
                at += len(buf)
                buf = []
                if at % 5000 < batch:
                    el = time.time() - t0
                    print(f"  pc {at:,}/{len(uids):,}  {el:.0f}s", flush=True)
    return out


def best_ranks(q: torch.Tensor, g: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Rank of each query's own index in its similarity row, 1-based.

    Our positives are singletons, so CAMERA's "best rank over all positives"
    reduces to "the rank of index i" -- computed here by counting how many
    gallery entries score strictly higher, which needs no sort and no N x N
    materialisation.
    """
    n = q.shape[0]
    ranks = torch.empty(n, dtype=torch.long)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        sims = q[i:j] @ g.t()                                   # (c, N)
        own = sims[torch.arange(j - i), torch.arange(i, j)]      # (c,)
        ranks[i:j] = (sims > own.unsqueeze(1)).sum(dim=1).cpu() + 1
    return ranks


def summarize(ranks: torch.Tensor) -> dict:
    r = ranks.float()
    return {
        "mrr": (1.0 / r).mean().item() * 100,
        "r1": (r <= 1).float().mean().item() * 100,
        "r5": (r <= 5).float().mean().item() * 100,
        "r10": (r <= 10).float().mean().item() * 100,
        # NDCG@5 with one binary positive collapses to 1/log2(rank+1) inside the
        # top 5 and 0 beyond it -- IDCG is 1 because the ideal puts the single
        # positive first. Written out rather than looped so the identity is
        # visible; it is exactly CAMERA's formula under |pos| = 1.
        "ndcg5": (torch.where(r <= 5, 1.0 / torch.log2(r + 1),
                              torch.zeros_like(r)).mean().item() * 100),
        "mean_rank": r.mean().item(),
        "median_rank": r.median().item(),
        "n": int(r.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-mode", default="mean", choices=("mean", "single"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--subset", type=int, default=14966,
                    help="second pool size, matching CAMERA's 14,966")
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    args = ap.parse_args()

    uids = load_uids()
    if args.limit:
        uids = uids[:args.limit]
    print(f"corpus {len(uids):,} assets", flush=True)

    txt, img = load_text_image(uids, args.image_mode, args.seed)
    pc = encode_pc(uids, args.batch)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feats = {}
    for name, a in (("text", txt), ("image", img), ("pc", pc)):
        t = torch.from_numpy(a).to(dev)
        feats[name] = torch.nn.functional.normalize(t, dim=-1)
        print(f"{name}: {tuple(t.shape)}  mean norm before normalise "
              f"{t.norm(dim=-1).mean():.2f}", flush=True)

    rng = np.random.default_rng(args.seed)
    pools = {"full": np.arange(len(uids))}
    if args.subset and args.subset < len(uids):
        pools[f"subset_{args.subset}"] = np.sort(
            rng.choice(len(uids), size=args.subset, replace=False))

    results = {}
    for pool_name, idx in pools.items():
        sel = torch.from_numpy(idx).to(dev)
        sub = {k: v.index_select(0, sel) for k, v in feats.items()}
        results[pool_name] = {}
        for tag, qk, gk in PAIRS:
            t0 = time.time()
            ranks = best_ranks(sub[qk], sub[gk])
            s = summarize(ranks)
            s["seconds"] = round(time.time() - t0, 2)
            results[pool_name][tag] = s
            print(f"{pool_name:16s} {tag}  R@1 {s['r1']:6.2f}  R@5 {s['r5']:6.2f}  "
                  f"NDCG@5 {s['ndcg5']:6.2f}  MRR {s['mrr']:6.2f}", flush=True)

    payload = {
        "what": "CAMERA's six-way object-level retrieval on our corpus, "
                "released ULIP-2 point encoder (no Stage 1 weights).",
        "protocol_source": "CAMERA evaluate.py / their 2025-08-12 working note; "
                           "metric convention RR@k + NDCG@5 is the Text2Shape "
                           "(Chen et al. 3DV 2018) lineage via TriCoLo and Parts2Words.",
        "differences_from_camera": [
            "positives: ONE on both sides. CAMERA's loader emits one row per "
            "object with a randomly drawn caption, so their best-rank-over-"
            "positives reduces to single-positive instance retrieval, same as "
            "ours. This is NOT a difference; an earlier note claiming ours was "
            "stricter was wrong and is withdrawn.",
            f"pool {len(uids):,} vs CAMERA's 14,966; a seeded {args.subset:,} "
            "subset is reported alongside",
            "point encoder is released ULIP-2 (ULIP2_PointBERT_Colored, 10k "
            "xyzrgb); CAMERA's checkpoints were shown by forensics to be "
            "ULIP_PointBERT (ULIP-1, 8192 xyz)",
        ],
        "image_mode": args.image_mode,
        "seed": args.seed,
        "n_assets": len(uids),
        "camera_reference_row": {
            "note": "their T2S on 14,966 ShapeNet chair/table, ULIP-1 backbone",
            "S2T": {"mrr": 21.54, "r1": 9.60, "r5": 32.25, "r10": 48.06, "ndcg5": 21.09},
            "T2S": {"mrr": 26.66, "r1": 13.50, "r5": 39.69, "r10": 55.89, "ndcg5": 26.89},
            "S2I": {"mrr": 48.76, "r1": 33.73, "r5": 67.01, "r10": 79.47, "ndcg5": 51.34},
            "I2S": {"mrr": 56.42, "r1": 41.48, "r5": 75.04, "r10": 85.94, "ndcg5": 59.46},
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
