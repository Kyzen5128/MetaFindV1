#!/usr/bin/env python3
"""EXPERIMENT C -- the same pure-modality evaluator, on our own 45,692.

Kyzen via Codex, 2026-09-01, third item:

> Run the same pure-modality evaluator on our own 45,692 corpus. test query
> 9,138, gallery 45,692, text query from an independent caption, image query
> from an independent view, pc query from an independent resample, plus a UID
> derangement control. **The result must not be used to claim a reproduction of
> CAMERA's 13.50** -- it exists only to validate the evaluator and the data
> alignment.

That constraint is the point of the probe, so it is repeated in the output and
not only here. Experiment B established that CAMERA's number comes from their
own fine-tuned `ULIP2_Instuct_c/checkpoint_last.pt`, which makes any comparison
with a released checkpoint meaningless in both directions.

WHAT THIS DOES VALIDATE
-----------------------
Two things nothing else has checked end to end:

1. **The evaluator.** Six directions, MRR / R@1 / R@5 / R@10 / NDCG@5, computed
   by the same code path as Experiment B so that a bug in the ranking would
   show up in both, and the derangement arm gives the floor each direction must
   collapse to.
2. **The alignment.** If `text[i]`, `views[i]`, the cloud for uid `i` and the
   gallery row for uid `i` ever came apart, the diagonal would stop being the
   answer and the deranged arm would stop being distinguishable from the real
   one. That is exactly the failure Codex flagged as unfalsifiable in MetaFind's
   own numbers, and here we can falsify it because we own both sides.

QUERY SIDE, INDEPENDENT ON ALL THREE
------------------------------------
  text   a NON-canonical `description_candidates` entry (rank >= 1),
         re-serialised through the production template and encoded by the same
         frozen tower. Assets with no second candidate are dropped and counted.
  image  ONE view, `views[uid_seed(uid) % n_views]`, against a gallery built from
         the 12-view mean -- so the query's own view is inside its own gallery
         entry at weight 1/12. That residue is measured, not assumed:
         `image_arm_anatomy` priced it at 10.09 points.
  pc     the query pack's independently resampled cloud where one exists,
         otherwise the canonical one, and which is which is recorded.

Gallery is the PURE target modality, never a fusion: text->pc scores against
the point-cloud embedding alone. No RAG, no fusion module, no ESSGNN, no Stage
1 weights -- the released ULIP-2 10k-xyzrgb encoder only.

Seeds fixed for random, numpy and torch. Every row's uid, caption, view, pc
source, positive count and six ranks are written out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402
from metafind.data.pointclouds import uid_seed  # noqa: E402
from metafind.models import resolve_stage1 as R  # noqa: E402
from metafind.train.stage1 import protocol_n_views  # noqa: E402

OUT = REPO / "output" / "look" / "exp_c_our_corpus_pure_modality.json"
ROWS = REPO / "output" / "look" / "exp_c_our_corpus_rows.jsonl"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
QPACK = paths.OUTPUTS / "_probe" / "query_pack" / "query_pack.json"
DISCLAIMER = ("evaluator and alignment validation ONLY. NOT a reproduction of "
              "CAMERA's 13.50: CAMERA scores its own fine-tuned "
              "ULIP2_Instuct_c/checkpoint_last.pt, this is a released "
              "zero-shot checkpoint, and the corpora differ.")


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k-ndcg", type=int, default=5)
    args = ap.parse_args()
    seed_all(args.seed)

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"gallery {len(corpus):,}  queries {len(queries):,}", flush=True)

    # ---- query-side text: a candidate the gallery does NOT carry -----------
    alt_text, dropped = {}, []
    for u in queries:
        a = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
        cands = [c for c in (a.get("description_candidates") or [])
                 if c.get("rank", 0) >= 1 and c.get("text")]
        if not cands:
            dropped.append(u)
            continue
        b = dict(a)
        b["description"] = cands[0]["text"]
        alt_text[u] = (R.serialize_annotation(b), cands[0]["rank"])
    queries = [u for u in queries if u in alt_text]
    print(f"  {len(dropped):,} queries dropped for having no alternate caption; "
          f"{len(queries):,} remain", flush=True)

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope="frozen"))
    ck = pathlib.Path(BackboneConfig().checkpoint)

    # ---- encode the query side ---------------------------------------------
    Q_t = np.empty((len(queries), 1280), np.float32)
    strs = [alt_text[u][0] for u in queries]
    with torch.no_grad():
        for i in range(0, len(strs), args.batch):
            j = min(i + args.batch, len(strs))
            Q_t[i:j] = bb.encode_text(strs[i:j]).float().cpu().numpy()
            if i % (args.batch * 40) == 0:
                print(f"  query text {i:,}/{len(strs):,}", flush=True)

    # [ULIP2 REVIEWER MINOR 4] Was `% 12`, fifteen lines above the
    # `protocol_n_views` call added to remove exactly this. It does not depend
    # on the query pack, so the "defer with the pack" rationale did not reach
    # it. Correct today only because the corpus is 12.
    _enc0 = json.loads((paths.OUTPUTS / "stage1_encoding_protocol.json").read_text())
    _n_views = protocol_n_views(_enc0)
    k_of = {u: uid_seed(u) % _n_views for u in queries}
    Q_i = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"]
                    .astype(np.float32)[k_of[u]] for u in queries])

    qpc_src = "canonical (no query pack on disk)"
    Q_p = np.load(PC_CACHE).astype(np.float32)[[pos[u] for u in queries]]
    if QPACK.exists():
        try:
            from metafind.train.stage1 import QueryPack
            # Reviewer 2026-09-03: this sits inside a try/except, so the
            # missing-argument TypeError was swallowed and the probe reported
            # "no query pack on disk" while the pack was present and
            # unbuildable -- a different query construction under a label
            # saying it had not been used.
            _enc = json.loads((paths.OUTPUTS / "stage1_encoding_protocol.json").read_text())
            qp = QueryPack(QPACK, protocol_n_views(_enc))
            n_hit = 0
            clouds, where = [], []
            for u in queries:
                v = qp.vector("pc", u) if hasattr(qp, "vector") else None
                if v is not None:
                    clouds.append(np.asarray(v, np.float32))
                    where.append(len(clouds) - 1)
                    n_hit += 1
                else:
                    where.append(None)
            if n_hit:
                enc = np.empty((len(clouds), 1280), np.float32)
                with torch.no_grad():
                    for i in range(0, len(clouds), args.batch):
                        j = min(i + args.batch, len(clouds))
                        enc[i:j] = bb.encode_pc(np.stack(clouds[i:j])).float().cpu().numpy()
                for r, w in enumerate(where):
                    if w is not None:
                        Q_p[r] = enc[w]
                qpc_src = f"query pack resample for {n_hit:,}/{len(queries):,}"
        except Exception as e:  # recorded, never silently swallowed
            qpc_src = f"canonical (query pack unusable: {type(e).__name__}: {e})"
    print(f"  pc query: {qpc_src}", flush=True)

    # ---- gallery: pure target modality, never a fusion ---------------------
    G_p = np.load(PC_CACHE).astype(np.float32)
    G_t = np.empty((len(corpus), 1280), np.float32)
    G_i = np.empty((len(corpus), 1280), np.float32)
    for i, u in enumerate(corpus):
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        G_t[i] = z["text"]
        G_i[i] = z["image"]
        if (i + 1) % 15000 == 0:
            print(f"  gallery {i + 1:,}/{len(corpus):,}", flush=True)

    n = lambda a: torch.nn.functional.normalize(
        torch.from_numpy(np.asarray(a, np.float32)).to(dev), dim=-1)
    Qt, Qi, Qp = n(Q_t), n(Q_i), n(Q_p)
    Gt, Gi, Gp = n(G_t), n(G_i), n(G_p)
    tgt = torch.tensor([pos[u] for u in queries], device=dev)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(queries))
    fixed = np.flatnonzero(perm == np.arange(len(queries)))
    if len(fixed):
        perm[fixed] = perm[(fixed + 1) % len(perm)]

    k = args.k_ndcg

    def evaluate(Q, G):
        """One positive per query here, so NDCG@k reduces to 1/log2(rank+1)."""
        ranks = torch.empty(len(queries), dtype=torch.long, device=dev)
        for a in range(0, len(queries), 1024):
            b = min(a + 1024, len(queries))
            s = Q[a:b] @ G.t()
            own = s.gather(1, tgt[a:b].unsqueeze(1))
            ranks[a:b] = (s > own).sum(1) + 1
        r = ranks.float()
        nd = torch.where(ranks <= k, 1.0 / torch.log2(r + 1), torch.zeros_like(r))
        return {"mrr": round(float((1 / r).mean()) * 100, 2),
                "r1": round(float((r <= 1).float().mean()) * 100, 2),
                "r5": round(float((r <= 5).float().mean()) * 100, 2),
                "r10": round(float((r <= 10).float().mean()) * 100, 2),
                f"ndcg@{k}": round(float(nd.mean()) * 100, 2),
                "median_rank": int(r.median()),
                "hits@1": int((r <= 1).sum()), "hits@5": int((r <= 5).sum()),
                "hits@10": int((r <= 10).sum()), "n": len(queries)}, ranks

    res = {"experiment": "C -- pure-modality six-way on our own corpus",
           "DISCLAIMER": DISCLAIMER,
           "checkpoint": str(ck), "checkpoint_sha256":
               hashlib.sha256(ck.read_bytes()).hexdigest(),
           "n_gallery": len(corpus), "n_query": len(queries),
           "dropped_no_alternate_caption": len(dropped),
           "seed": args.seed,
           "query_text": "non-canonical description_candidate, rank>=1",
           "query_image": f"single view, uid_seed(uid) % {_n_views}",
           "query_pc": qpc_src,
           "gallery": "pure target modality, no fusion",
           "results": {}}

    pairs = [("T2S (Text->PC)", Qt, Gp), ("S2T (PC->Text)", Qp, Gt),
             ("I2S (Image->PC)", Qi, Gp), ("S2I (PC->Image)", Qp, Gi),
             ("T2I (Text->Image)", Qt, Gi), ("I2T (Image->Text)", Qi, Gt)]
    print(f"\n{'方向':<20s}{'MRR':>8s}{'R@1':>8s}{'R@5':>8s}{'R@10':>8s}"
          f"{'NDCG@5':>9s}{'排名中位':>9s}")
    keep = {}
    for name, Q, G in pairs:
        m, rk = evaluate(Q, G)
        res["results"][name] = m
        keep[name] = rk.cpu().tolist()
        print(f"{name:<20s}{m['mrr']:8.2f}{m['r1']:8.2f}{m['r5']:8.2f}"
              f"{m['r10']:8.2f}{m[f'ndcg@{k}']:9.2f}{m['median_rank']:9d}")

    print(f"\n--- UID derangement 控制組（每個方向的地板）---")
    res["derangement_control"] = {}
    for name, Q, G in pairs:
        m, _ = evaluate(Q[perm], G)
        res["derangement_control"][name] = m
        print(f"{name:<20s}{m['mrr']:8.2f}{m['r1']:8.2f}{m['r5']:8.2f}"
              f"{m['r10']:8.2f}{m[f'ndcg@{k}']:9.2f}{m['median_rank']:9d}")
    res["chance_R1"] = round(100.0 / len(corpus), 5)
    print(f"{'隨機期望':<20s}{'':>8s}{res['chance_R1']:8.5f}")

    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("w") as f:
        for i, u in enumerate(queries):
            f.write(json.dumps({
                "row": i, "query_uid": u, "gallery_uid": u,
                "gallery_index": pos[u],
                "caption_rank_used": alt_text[u][1],
                "caption": alt_text[u][0],
                "view": int(k_of[u]),
                "rank": {name: keep[name][i] for name in keep},
            }, ensure_ascii=False) + "\n")
    res["rows_file"] = str(ROWS)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n{DISCLAIMER}")
    print(f"-> {OUT}\n-> {ROWS}  ({len(queries):,} 列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
