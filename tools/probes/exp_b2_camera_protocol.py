#!/usr/bin/env python3
"""EXPERIMENT B2 -- CAMERA's exact protocol, released checkpoint. One variable.

The Codex on CAMERA's machine answered every question B was blocked on, from
the original W&B run rather than from memory. The three that change this
experiment:

  POOL      14,966 = `shapenet-55/train.txt` + `test.txt`, kept if the
            (taxonomy, model) PAIR appears in `captions.tablechair.csv`
            (`dataset_3d.py:376`). Matching on model_id alone gives 14,986 --
            20 objects whose CSV taxonomy reads 04379243 while ShapeNet files
            them under 02933112 (10), 03001627 (8), 03636649 (1), 04460130 (1).
            Not the official Text2Shape split, not `processed_captions_*.p`,
            and the number counts OBJECTS.
  ROWS      ONE ROW PER OBJECT. `ShapeNet.__getitem__` returns one object with
            `random.choice(captions_of_this_object)`, so N = 14,966 for all
            three feature matrices. B built 7,414 caption rows with each object
            appearing ~5 times, which is a different and easier ranking problem.
  MODEL     `ULIP_PointBERT`, 8192 xyz, SLIP ViT-B, 512-d -- the same
            architecture B used. `ULIP2_Instuct_c` is only an output directory
            name; it is NOT ULIP2_PointBERT_Colored.

And the one that reframes the whole comparison:

  Their checkpoint was TRAINED, 250 epochs on a single 4090, lr 3e-3, batch 32,
  `--pretrain_dataset_name shapenet`, initialised from `point_bert_pretrained.pt`
  plus a frozen `slip_base_100ep.pt`, and its captions came from
  `captions.tablechair.csv` -- the same Text2Shape text the evaluation then
  retrieves against. So 26.66 / 13.50 is not a zero-shot number and no released
  checkpoint should be held to it.

WHAT THIS PROBE IS FOR
----------------------
With the pool, the row construction, the modality preprocessing and the
architecture now all matched to theirs, **the checkpoint is the only remaining
difference**. Whatever gap is left is attributable to 250 epochs of training on
this corpus, and nothing else. That is worth measuring exactly once.

Their reference, from the training log at epoch 247 / 249:

  T2S   MRR 26.43 / 26.40   R@1 13.16 / 13.17   R@5 39.83 / 40.13
        R@10 55.71 / 56.49  NDCG@5 26.80 / 26.83
  I2S   MRR 56.42   R@1 41.48 (6208/14966)   R@5 75.04   R@10 85.94
  S2I   R@1 33.73

PROTOCOL, matched item by item to what they reported
----------------------------------------------------
  pc      8192 xyz, no RGB; centroid subtracted then divided by max radius;
          no augmentation at evaluation
  image   one RGB view chosen at random from the 30 (`000, 012, ... 348`);
          `_depth0001` excluded. Their in-training six-way eval ran on
          `train_dataset`, so RandomResizedCrop(224, 0.5-1.0) + ImageNet
          statistics were still active -- reproduced here with a fixed seed
          rather than replaced by a deterministic crop, because matching the
          protocol matters more than tidiness.
  text    the CSV's raw English, one caption per object chosen at random,
          SimpleTokenizer, no prompt template
  score   each modality L2-normalised, dot product

One positive per query, so NDCG@5 reduces to 1/log2(rank+1) when rank <= 5.
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
import random
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
UPSTREAM = pathlib.Path("/home/kyzen/upstream/ULIP_run")
SN = pathlib.Path("/mnt/data1/kyzen/shapenet-55")
CSVP = pathlib.Path("/mnt/data1/kyzen/text2shape/captions.tablechair.csv")
CKPT = (REPO / "data/models/ulip2/ULIP-2/pretrained_models/"
        "ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt")
OUT_T = "output/look/exp_b2_camera_protocol_{tag}.json"
ROWS_T = "output/look/exp_b2_camera_protocol_rows_{tag}.jsonl"
CAMERA = {"T2S (Text->PC)": {"mrr": 26.43, "r1": 13.16, "r5": 39.83,
                             "r10": 55.71, "ndcg@5": 26.80},
          "I2S (Image->PC)": {"mrr": 56.42, "r1": 41.48, "r5": 75.04,
                              "r10": 85.94, "ndcg@5": 59.46},
          "S2I (PC->Image)": {"r1": 33.73}}


class Args:
    evaluate_3d = True
    npoints = 8192
    use_height = False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, help="CAMERA trained at seed 0")
    ap.add_argument("--ckpt", default=str(CKPT),
                    help="the released checkpoint by default; pass CAMERA's "
                         "checkpoint_last.pt to remove the last variable")
    ap.add_argument("--tag", default="released")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # [FIXED, CAMERA's Codex 2026-09-01] Keyed on the (taxonomy, model) PAIR,
    # which is what `dataset_3d.py:376` matches on. Keying on model_id alone
    # kept 20 extra objects whose CSV taxonomy says 04379243 while ShapeNet
    # files them under 02933112 (10), 03001627 (8), 03636649 (1), 04460130 (1)
    # -- 14,986 instead of 14,966. Pair matching reproduces their pool exactly.
    caps: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for r in csv.DictReader(CSVP.open(encoding="utf-8", errors="replace")):
        caps[(r["topLevelSynsetId"], r["modelId"])].append(r["description"])
    ids = []
    for f in ("train.txt", "test.txt"):
        ids += [l.strip() for l in (SN / f).read_text().splitlines() if l.strip()]
    pcdir = SN / "shapenet_pc"
    keys, seen = [], set()
    for x in ids:
        stem = x[:-4] if x.endswith(".npy") else x
        key = tuple(stem.split("-", 1))
        if key in caps and stem not in seen and (pcdir / f"{stem}.npy").exists():
            seen.add(stem)
            keys.append(stem)
    print(f"pool {len(keys):,}   (CAMERA log: 14,966; delta "
          f"{len(keys) - 14966:+d}, {abs(len(keys)-14966)/14966*100:.2f}%)",
          flush=True)

    from metafind.compat import ulip_patch
    ulip_patch.apply(patch_fps=False)
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    from models.ULIP_models import ULIP_PointBERT
    ulip_patch.apply(patch_fps=True)
    m = ULIP_PointBERT(Args())
    CK = pathlib.Path(args.ckpt)
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.get("state_dict", sd).items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    print(f"loaded {CK.name}  missing {len(miss)}  unexpected {len(unexp)}",
          flush=True)

    from torchvision import transforms as T
    from PIL import Image
    tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    from utils.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()

    P = np.empty((len(keys), 512), np.float32)
    I = np.empty((len(keys), 512), np.float32)
    Tx = np.empty((len(keys), 512), np.float32)
    chosen = []
    for a in range(0, len(keys), args.batch):
        chunk = keys[a:a + args.batch]
        clouds, imgs, texts, meta = [], [], [], []
        for stem in chunk:
            x = np.load(pcdir / f"{stem}.npy").astype(np.float32)
            if x.shape[0] > 8192:
                x = x[np.random.permutation(x.shape[0])[:8192]]
            x = x - x.mean(0)
            clouds.append(x / (np.sqrt((x ** 2).sum(1)).max() + 1e-9))
            cands = sorted(p for p in glob.glob(
                str(SN / "rendered_images" / stem / "*.png"))
                if "depth" not in os.path.basename(p))
            pick = random.choice(cands) if cands else None
            imgs.append(tf(Image.open(pick).convert("RGB")) if pick
                        else torch.zeros(3, 224, 224))
            cap = random.choice(caps[tuple(stem.split("-", 1))])
            texts.append(tok(cap))
            meta.append((os.path.basename(pick) if pick else "", cap))
        chosen += meta
        with torch.no_grad():
            P[a:a + len(chunk)] = m.encode_pc(
                torch.from_numpy(np.stack(clouds)).to(dev)).float().cpu().numpy()
            I[a:a + len(chunk)] = m.encode_image(
                torch.stack(imgs).to(dev)).float().cpu().numpy()
            Tx[a:a + len(chunk)] = m.encode_text(
                torch.stack(texts).to(dev).long()).float().cpu().numpy()
        if (a + args.batch) % (args.batch * 60) == 0:
            print(f"  {min(a + args.batch, len(keys)):,}/{len(keys):,}", flush=True)

    n = lambda x: torch.nn.functional.normalize(
        torch.from_numpy(x).to(dev), dim=-1)
    Pn, In, Tn = n(P), n(I), n(Tx)
    N = len(keys)
    tgt = torch.arange(N, device=dev)

    def ev(Q, G):
        rk = torch.empty(N, dtype=torch.long, device=dev)
        for a in range(0, N, 1024):
            b = min(a + 1024, N)
            s = Q[a:b] @ G.t()
            own = s.gather(1, tgt[a:b].unsqueeze(1))
            rk[a:b] = (s > own).sum(1) + 1
        r = rk.float()
        nd = torch.where(rk <= 5, 1.0 / torch.log2(r + 1), torch.zeros_like(r))
        return {"mrr": round(float((1 / r).mean()) * 100, 2),
                "r1": round(float((r <= 1).float().mean()) * 100, 2),
                "r5": round(float((r <= 5).float().mean()) * 100, 2),
                "r10": round(float((r <= 10).float().mean()) * 100, 2),
                "ndcg@5": round(float(nd.mean()) * 100, 2),
                "median_rank": int(r.median()),
                "hits@1": int((r <= 1).sum()), "n": N}, rk

    res = {"experiment": "B2 -- CAMERA's protocol, released checkpoint",
           "pool": N, "camera_pool": 14966,
           "checkpoint": CK.name, "checkpoint_path": str(CK),
           "checkpoint_sha256": hashlib.sha256(CK.read_bytes()).hexdigest(),
           "tag": args.tag,
           "camera_checkpoint": "ULIP2_Instuct_c/checkpoint_last.pt, "
                                "sha256 16203627a281ee20…, TRAINED 250 epochs "
                                "on Text2Shape captions + ShapeNet",
           "single_variable": "the checkpoint; pool, rows, preprocessing and "
                              "architecture all match CAMERA's",
           "model": "ULIP_PointBERT", "npoints": 8192, "embed_dim": 512,
           "seed": args.seed, "results": {}, "camera_reference": CAMERA}

    pairs = [("T2S (Text->PC)", Tn, Pn), ("S2T (PC->Text)", Pn, Tn),
             ("I2S (Image->PC)", In, Pn), ("S2I (PC->Image)", Pn, In),
             ("T2I (Text->Image)", Tn, In), ("I2T (Image->Text)", In, Tn)]
    print(f"\n{'方向':<20s}{'MRR':>8s}{'R@1':>8s}{'R@5':>8s}{'R@10':>8s}"
          f"{'NDCG@5':>9s}{'中位':>8s}   CAMERA R@1")
    keep = {}
    for name, Q, G in pairs:
        d, rk = ev(Q, G)
        res["results"][name] = d
        keep[name] = rk.cpu().tolist()
        ref = CAMERA.get(name, {}).get("r1")
        print(f"{name:<20s}{d['mrr']:8.2f}{d['r1']:8.2f}{d['r5']:8.2f}"
              f"{d['r10']:8.2f}{d['ndcg@5']:9.2f}{d['median_rank']:8d}"
              + (f"   {ref:>8.2f}" if ref else "          -"))

    OUT = REPO / OUT_T.format(tag=args.tag)
    ROWS = REPO / ROWS_T.format(tag=args.tag)
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with ROWS.open("w") as f:
        for i, stem in enumerate(keys):
            f.write(json.dumps({
                "row": i, "key": stem, "synset": stem.split("-", 1)[0],
                "model_id": stem.split("-", 1)[1],
                "view": chosen[i][0], "caption": chosen[i][1],
                "rank": {k: v[i] for k, v in keep.items()},
            }, ensure_ascii=False) + "\n")
    res["rows_file"] = str(ROWS)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}\n-> {ROWS}  ({N:,} 列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
