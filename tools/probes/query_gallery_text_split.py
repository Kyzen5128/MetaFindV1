#!/usr/bin/env python3
"""Does the gap close when the query's text is not the gallery's text?

Kyzen, 2026-09-01: 「有沒有可能官方 ulip2_objaverse_lvis 就是他們實際測試用的
方法啊 裡面什麼都有啊」

WHAT THE QUESTION IS REALLY ASKING
----------------------------------
`diag_untrained_fusion_identity` measured a zero-parameter control -- no
fusion, no module, no training, just the mean of the present raw ULIP vectors
scored against the mean of all three -- at text 99.56 and full 100.00 against
the paper's 13.8 and 51.7. The `full` cell is 1.0000 because the two means are
the SAME expression: query and gallery are built from one set of vectors, so
the target's own vector sits inside the target's own gallery entry. No amount
of training moves a number that is already 99.56 before the model exists.

ULIP-2's published shards break that, and they are the only material we have
that can: each asset carries FOUR independent descriptions -- the Objaverse
`name`, a BLIP caption of the thumbnail, an Azure caption of the same
thumbnail, and the captions of its LAION-5B nearest neighbours -- plus ours
makes five. Five different sentences about one object. Put one on the query
side and a different one in the gallery and the identity is gone, while
everything else about the construction is held fixed.

That is the experiment: a 5x5 sweep of query-text source against gallery-text
source. The diagonal is the construction we currently run. The off-diagonal is
the same construction with the identity removed. If the paper's 13.8 lives
anywhere in this table it lives off the diagonal, and the distance from the
diagonal to it is the size of what our protocol is missing.

WHAT THIS IS NOT
----------------
Not a claim that MetaFind uses these captions. It does not: `2methdology.tex:28`
says each asset is "rendered from 11 orthogonal viewpoints and annotated using
GPT-4o", which is our own text, not `blip_caption_feat`. The four upstream
captions are used here as INSTRUMENTS -- five sentences about one object are
what make the identity separable, whoever wrote them.

Image and point cloud are held fixed at ULIP-2's own published `image_feat`
and their `xyz`/`rgb` encoded by the released backbone, so the text source is
the only thing that varies across the sweep.

Reported for two Table-1 conditions:
  text  query = the query text alone
  full  query = mean(query text, image, pc)   gallery = mean(gallery text, image, pc)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "query_gallery_text_split.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shards")
SOURCES = ("name", "blip", "msft", "retrieval", "ours")
PAPER = {"text": 13.8, "full": 51.7}


def corpus_pos() -> dict[str, int]:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    return {u: i for i, u in enumerate(sorted(set(d["train"]) | set(d["test"])))}


def r1(q: torch.Tensor, g: torch.Tensor) -> float:
    """Row i's target is column i. Full-pool R@1, ties counted against us."""
    q = torch.nn.functional.normalize(q, dim=-1)
    g = torch.nn.functional.normalize(g, dim=-1)
    s = q @ g.t()
    own = s.diagonal().unsqueeze(1)
    return ((s > own).sum(1) < 1).sum().item() / q.shape[0] * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    theirs = {os.path.basename(f)[:-4]: f
              for f in glob.glob(os.path.join(args.shards, "*", "*.npy"))}
    pos = corpus_pos()
    keep = sorted(u for u in theirs if u in pos)
    print(f"they have {len(theirs):,}, we have {len(pos):,}, overlap {len(keep):,}",
          flush=True)
    if len(keep) < 500:
        sys.exit("overlap too small to read anything off -- extract more shards")

    txt = {s: [] for s in SOURCES}
    xyzrgb, img = [], []
    for u in keep:
        d = np.load(theirs[u], allow_pickle=True).item()
        xyzrgb.append(np.concatenate(
            [np.asarray(d["xyz"], np.float32), np.asarray(d["rgb"], np.float32)], 1))
        img.append(np.asarray(d["image_feat"], np.float32).mean(0))
        txt["name"].append(np.asarray(d["text_feat"][0]["original"], np.float32)[0])
        txt["blip"].append(np.asarray(
            np.asarray(d["blip_caption_feat"]).item()["original"], np.float32)[0])
        txt["msft"].append(np.asarray(
            np.asarray(d["msft_caption_feat"]).item()["original"], np.float32)[0])
        rt = d["retrieval_text_feat"]
        txt["retrieval"].append(
            np.asarray(rt[0]["original"], np.float32)[0] if len(rt)
            else np.zeros(1280, np.float32))
        txt["ours"].append(
            np.load(paths.EMBEDDINGS / f"{u}.npz")["text"].astype(np.float32))

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    pc = np.empty((len(keep), 1280), np.float32)
    with torch.no_grad():
        for i in range(0, len(keep), args.batch):
            j = min(i + args.batch, len(keep))
            pc[i:j] = bb.encode_pc(np.stack(xyzrgb[i:j])).float().cpu().numpy()
            if i % (args.batch * 20) == 0:
                print(f"  encoded {i:,}/{len(keep):,}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    E = {s: T(np.stack(v)) for s, v in txt.items()}
    I, P = T(np.stack(img)), T(pc)

    res = {"n": len(keep), "shards": args.shards,
           "note": "unnormalised mean, matching the parameter-free control",
           "paper_metafind_wo_essgnn": PAPER, "text": {}, "full": {}}

    for cond in ("text", "full"):
        print(f"\n=== {cond} · R@1 over {len(keep):,} ===")
        hdr = "查詢文字 (列) / 畫廊文字 (行)"
        print(f"{hdr:<22s}" + "".join(f"{g:>12s}" for g in SOURCES))
        for qs in SOURCES:
            row = {}
            cells = []
            for gs in SOURCES:
                g = (E[gs] + I + P) / 3.0
                q = E[qs] if cond == "text" else (E[qs] + I + P) / 3.0
                v = r1(q, g)
                row[gs] = round(v, 2)
                cells.append(f"{v:12.2f}" if qs != gs else f"{'['+f'{v:.2f}'+']':>12s}")
            res[cond][qs] = row
            print(f"{qs:<22s}" + "".join(cells))
        print(f"{'論文':<22s}{PAPER[cond]:12.1f}   ← [] 是對角線，也就是我們現在的做法")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
