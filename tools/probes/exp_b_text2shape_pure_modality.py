#!/usr/bin/env python3
"""EXPERIMENT B -- CAMERA's no-RAG six-way retrieval on Text2Shape chair/table.

Kyzen via Codex, 2026-09-01: no RAG, no fusion, no ESSGNN, no Stage 1. ULIP's
own `ULIP_PointBERT` with the 8192 xyz SLIP ViT-B checkpoint, Text2Shape
chair/table, each modality L2-normalised on its own, gallery is the pure target
modality, six-way MRR / R@1 / R@5 / R@10 / NDCG@5, seeds fixed, and the query
uid, gallery uid, caption, view, rank and raw hit counts saved.

CAMERA reports T2S 26.66 / 13.50 / 39.69 / 55.89 / 26.89 over 14,966.

WHERE EVERY INPUT COMES FROM
----------------------------
  captions   text2shape.stanford.edu `captions.tablechair.csv`, downloaded
             2026-09-01. 75,360 human-written descriptions over 15,032 models,
             Chair 6,589 + Table 8,443. RAW ENGLISH -- the `.p` files hold
             vocabulary indices, which cannot be fed to a CLIP tokenizer.
  split      `text2shape-data/shapenet/processed_captions_{train,val,test}.p`
             from the same site, used ONLY for split membership: train 11,921
             models, val 1,486, test 1,492, union 14,899.
  clouds     `/mnt/data1/kyzen/datasets/shapenet-55/shapenet_pc/<synset>-<model>.npy`,
             (8192, 3) float64, already the count this checkpoint wants.
  images     `rendered_images/<synset>-<model>/<...>_r_NNN.png`, the RGB ones;
             the `_depth0001` siblings are excluded.
  model      `ULIP_PointBERT` from upstream, unmodified, with
             `ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt`.

`--split all` covers 14,899 models, of which 14,846 have a cloud. CAMERA says
14,966; the CSV alone gives 15,032 models and 14,979 with clouds. Those three
counts differ and this probe does NOT try to land on CAMERA's, because guessing
a filter to hit a target number is how a reproduction stops being one. The pool
it actually used is recorded in the output.

THE RANKING FOLLOWS CAMERA'S OWN evaluate.py
--------------------------------------------
One ROW PER CAPTION, so a model with five captions occupies five rows, and the
point cloud and image vectors are repeated across its rows. `obj2idxs` maps a
model to all of its rows, and a query's rank is the BEST rank over its
positives (`compute_best_ranks`, `docs/reference/camera/evaluate.py:150-171`).
Query and gallery are the same N. That is not the only defensible protocol but
it is the one whose numbers we are comparing against, so it is the one run.

NDCG@5 also follows theirs: IDCG uses min(#positives, k)
(`evaluate.py:174-196`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import pathlib
import pickle
import random
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
UPSTREAM = pathlib.Path("/home/kyzen/upstream/ULIP_run")
T2S = pathlib.Path("/mnt/data1/kyzen/datasets/text2shape")
SN = pathlib.Path("/mnt/data1/kyzen/datasets/shapenet-55")
CKPT = (REPO / "data/models/ulip2/ULIP-2/pretrained_models/"
        "ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt")
OUT = REPO / "output" / "look" / "exp_b_text2shape_pure_modality.json"
ROWS = REPO / "output" / "look" / "exp_b_text2shape_rows.jsonl"
CAMERA_T2S = {"mrr": 26.66, "r1": 13.50, "r5": 39.69, "r10": 55.89,
              "ndcg@5": 26.89, "n_models": 14966}


class Args:
    evaluate_3d = True
    npoints = 8192
    use_height = False


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model():
    from metafind.compat import ulip_patch
    ulip_patch.apply(patch_fps=False)
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    from models.ULIP_models import ULIP_PointBERT
    ulip_patch.apply(patch_fps=True)
    m = ULIP_PointBERT(Args())
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.get("state_dict", sd).items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    return m.eval(), {"missing": len(miss), "unexpected": len(unexp),
                      "missing_sample": list(miss)[:8],
                      "unexpected_sample": list(unexp)[:8]}


def split_models(which: str) -> set[str]:
    B = T2S / "bundle" / "text2shape-data" / "shapenet"
    names = ("train", "val", "test") if which == "all" else (which,)
    out: set[str] = set()
    for n in names:
        d = pickle.load(open(B / f"processed_captions_{n}.p", "rb"),
                        encoding="latin1")
        out |= set(d["caption_matches"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test", "all"))
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-captions", type=int, default=0,
                    help="0 = every caption the CSV has for a kept model")
    ap.add_argument("--k-ndcg", type=int, default=5)
    args = ap.parse_args()
    seed_all(args.seed)

    keep = split_models(args.split)
    rows = [r for r in csv.DictReader(
        (T2S / "captions.tablechair.csv").open(encoding="utf-8",
                                               errors="replace"))
            if r["modelId"] in keep]
    pcdir = SN / "shapenet_pc"
    have = {}
    for r in rows:
        syn, mid = r["topLevelSynsetId"], r["modelId"]
        p = pcdir / f"{syn}-{mid}.npy"
        if p.exists():
            have[mid] = syn
    rows = [r for r in rows if r["modelId"] in have]
    if args.max_captions:
        per = collections.defaultdict(int)
        kept = []
        for r in rows:
            if per[r["modelId"]] < args.max_captions:
                kept.append(r)
                per[r["modelId"]] += 1
        rows = kept
    models = sorted(have)
    print(f"split {args.split}: {len(keep):,} models in split, "
          f"{len(models):,} with a cloud, {len(rows):,} caption rows",
          flush=True)

    model, load_info = build_model()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    print(f"loaded {CKPT.name}  {load_info}", flush=True)

    from torchvision import transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    from models.pointbert.point_encoder import PointTransformer  # noqa: F401
    sys.path.insert(0, str(REPO))

    # ---- per-model pc and image, encoded ONCE then repeated across rows ----
    rng = np.random.default_rng(args.seed)
    view_of: dict[str, str] = {}
    pc_vec: dict[str, np.ndarray] = {}
    im_vec: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i in range(0, len(models), args.batch):
        chunk = models[i:i + args.batch]
        clouds = []
        for m in chunk:
            x = np.load(pcdir / f"{have[m]}-{m}.npy").astype(np.float32)
            x = x - x.mean(0)
            clouds.append(x / (np.sqrt((x ** 2).sum(1)).max() + 1e-9))
        with torch.no_grad():
            v = model.encode_pc(torch.from_numpy(np.stack(clouds)).to(dev))
        for m, e in zip(chunk, v.float().cpu().numpy()):
            pc_vec[m] = e
        imgs = []
        for m in chunk:
            cands = sorted(p for p in glob.glob(
                str(SN / "rendered_images" / f"{have[m]}-{m}" / "*.png"))
                if "depth" not in os.path.basename(p))
            pick = cands[int(rng.integers(len(cands)))] if cands else None
            view_of[m] = os.path.basename(pick) if pick else ""
            imgs.append(tf(Image.open(pick).convert("RGB")) if pick
                        else torch.zeros(3, 224, 224))
        with torch.no_grad():
            v = model.encode_image(torch.stack(imgs).to(dev))
        for m, e in zip(chunk, v.float().cpu().numpy()):
            im_vec[m] = e
        if (i + args.batch) % (args.batch * 40) == 0:
            print(f"  models {min(i + args.batch, len(models)):,}/{len(models):,}"
                  f"  {time.time() - t0:.0f}s", flush=True)

    # ---- captions, one row each -------------------------------------------
    from utils.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()
    txt = np.empty((len(rows), 512), np.float32)
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        t = torch.stack([tok(r["description"]) for r in chunk]).to(dev)
        if t.dim() == 3:
            t = t.squeeze(1)
        with torch.no_grad():
            txt[i:i + len(chunk)] = model.encode_text(t.long()).float().cpu().numpy()
        if (i + args.batch) % (args.batch * 100) == 0:
            print(f"  captions {min(i + args.batch, len(rows)):,}/{len(rows):,}",
                  flush=True)

    nrm = lambda a: torch.nn.functional.normalize(a, dim=-1)
    P = nrm(torch.from_numpy(np.stack([pc_vec[r["modelId"]] for r in rows])).to(dev))
    I = nrm(torch.from_numpy(np.stack([im_vec[r["modelId"]] for r in rows])).to(dev))
    Tx = nrm(torch.from_numpy(txt).to(dev))
    obj = [r["modelId"] for r in rows]
    obj2idx = collections.defaultdict(list)
    for i, m in enumerate(obj):
        obj2idx[m].append(i)

    def evaluate(Q, G, name):
        N = Q.shape[0]
        best = torch.empty(N, dtype=torch.long)
        ndcg = torch.empty(N)
        idx_of = {m: torch.tensor(v, device=dev) for m, v in obj2idx.items()}
        for a in range(0, N, 512):
            b = min(a + 512, N)
            s = Q[a:b] @ G.t()
            order = s.argsort(dim=1, descending=True)
            rankpos = order.argsort(dim=1)              # position of each idx
            for j in range(b - a):
                pos = idx_of[obj[a + j]]
                best[a + j] = int(rankpos[j][pos].min()) + 1
                top = order[j, :args.k_ndcg]
                hit = torch.isin(top, pos).float()
                dcg = (hit / torch.log2(torch.arange(
                    2, args.k_ndcg + 2, device=dev).float())).sum()
                Pn = min(len(pos), args.k_ndcg)
                idcg = (1.0 / torch.log2(torch.arange(
                    2, Pn + 2, device=dev).float())).sum()
                ndcg[a + j] = (dcg / idcg) if idcg > 0 else 0.0
        r = best.float()
        return {"mrr": round(float((1.0 / r).mean()) * 100, 2),
                "r1": round(float((r <= 1).float().mean()) * 100, 2),
                "r5": round(float((r <= 5).float().mean()) * 100, 2),
                "r10": round(float((r <= 10).float().mean()) * 100, 2),
                f"ndcg@{args.k_ndcg}": round(float(ndcg.mean()) * 100, 2),
                "median_rank": int(r.median()), "n": N}, best

    res = {"experiment": "B -- CAMERA no-RAG six-way, Text2Shape chair/table",
           "checkpoint": str(CKPT),
           "checkpoint_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
           "model": "ULIP_PointBERT", "npoints": 8192, "embed_dim": 512,
           "load_state_dict": load_info,
           "split": args.split, "n_models_in_split": len(keep),
           "n_models_with_cloud": len(models), "n_caption_rows": len(rows),
           "seed": args.seed,
           "camera_reported_T2S": CAMERA_T2S,
           "note": ("gallery is the pure target modality; one row per caption; "
                    "best rank over a model's positives, as CAMERA does"),
           "results": {}}

    pairs = [("S2T (PC->Text)", P, Tx), ("T2S (Text->PC)", Tx, P),
             ("S2I (PC->Image)", P, I), ("I2S (Image->PC)", I, P),
             ("T2I (Text->Image)", Tx, I), ("I2T (Image->Text)", I, Tx)]
    print(f"\n{'方向':<20s}{'MRR':>8s}{'R@1':>8s}{'R@5':>8s}{'R@10':>8s}"
          f"{'NDCG@5':>9s}{'排名中位':>9s}")
    ranks_out = {}
    for name, Q, G in pairs:
        m, best = evaluate(Q, G, name)
        res["results"][name] = m
        ranks_out[name] = best.tolist()
        print(f"{name:<20s}{m['mrr']:8.2f}{m['r1']:8.2f}{m['r5']:8.2f}"
              f"{m['r10']:8.2f}{m[f'ndcg@{args.k_ndcg}']:9.2f}"
              f"{m['median_rank']:9d}")
    print(f"{'CAMERA T2S':<20s}{CAMERA_T2S['mrr']:8.2f}{CAMERA_T2S['r1']:8.2f}"
          f"{CAMERA_T2S['r5']:8.2f}{CAMERA_T2S['r10']:8.2f}"
          f"{CAMERA_T2S['ndcg@5']:9.2f}")

    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "row": i, "model_id": r["modelId"], "synset": have[r["modelId"]],
                "category": r["category"], "caption": r["description"],
                "view": view_of[r["modelId"]],
                "n_positives": len(obj2idx[r["modelId"]]),
                "rank": {k: v[i] for k, v in ranks_out.items()},
            }, ensure_ascii=False) + "\n")
    res["rows_file"] = str(ROWS)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}\n-> {ROWS}  ({len(rows):,} 列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
