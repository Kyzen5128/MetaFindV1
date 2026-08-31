#!/usr/bin/env python3
"""Ours against ULIP-2's own published data, on the assets both of us have.

Kyzen, 2026-09-01: point cloud, render, annotation, encoding -- all four done,
so whose is better?

"Better" needs a task, or it is just opinion. Two are used here, both of which
the upstream layer already answers correctly so neither favours us:

  * **zero-shot LVIS classification** for the point cloud and the image. Same
    1,156 prototypes, built by upstream's own 64-template recipe, so the only
    thing that varies is which vector is being classified.
  * **text -> point-cloud retrieval R@1** for the text. Both sides' text is
    scored against the SAME gallery, so the text is the only variable.

`SFXX/ulip`'s `ULIP-2/objaverse_lvis` shards carry OpenShape's preprocessing:
`xyz`/`rgb` (10k, unit sphere, 0-1 colour), `image_feat` (12 views, already
encoded by OpenCLIP ViT-bigG), and four text sources -- the Objaverse `name`,
a BLIP caption of the thumbnail, an Azure caption of the same thumbnail, and
the captions of LAION-5B nearest neighbours. Each text carries both `original`
and `prompt_avg` (the 64-template mean).

WHAT THIS CANNOT SAY
--------------------
The comparison runs on whatever assets the extracted shards and our corpus have
in common, which is a slice, not the corpus. And it compares INPUTS under a
fixed encoder; it says nothing about MetaFind's Table 1, whose numbers are set
by the query/gallery construction, not by input quality.
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

OUT = REPO / "output" / "look" / "ours_vs_ulip2_data.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
LVIS = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/objaverse-lvis/"
                    "objaverse_lvis_metadata.json")
TEMPLATES = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/templates.json")
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shard0")


def corpus_uids() -> list[str]:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    return sorted(set(d["train"]) | set(d["test"]))


def prototypes(bb, names, templates) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i, n in enumerate(names):
            e = bb.encode_text([t.format(n) for t in templates]).float()
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=-1))
            if (i + 1) % 300 == 0:
                print(f"  prototypes {i + 1}/{len(names)}", flush=True)
    return torch.stack(out)


def acc(feats, protos, target):
    f = torch.nn.functional.normalize(feats, dim=-1)
    lg = f @ protos.t()
    own = lg.gather(1, target.unsqueeze(1))
    higher = (lg > own).sum(1)
    n = f.shape[0]
    return (higher < 1).sum().item() / n * 100, (higher < 5).sum().item() / n * 100


def r1(q, g):
    """text -> pc R@1 inside this pool; row i's target is column i."""
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
    uids = corpus_uids()
    pos = {u: i for i, u in enumerate(uids)}
    both = sorted(u for u in theirs if u in pos)
    print(f"they have {len(theirs):,}, we have {len(uids):,}, overlap {len(both):,}",
          flush=True)
    if not both:
        sys.exit("no overlap -- extract a shard first")

    meta = json.loads(LVIS.read_text())
    names, v2k, k2id = meta["all_keys"], meta["value_to_key_mapping"], meta["key_to_id"]
    templates = json.loads(TEMPLATES.read_text())["modelnet40_64"]

    keep, labels = [], []
    for u in both:
        n = v2k.get(u)
        if n in k2id:
            keep.append(u)
            labels.append(k2id[n])
    print(f"{len(keep):,} of them carry an LVIS label", flush=True)

    # --- their side -----------------------------------------------------
    t_xyzrgb, t_img, t_txt = [], [], {k: [] for k in
                                      ("name", "blip", "msft", "retrieval")}
    for u in keep:
        d = np.load(theirs[u], allow_pickle=True).item()
        t_xyzrgb.append(np.concatenate(
            [np.asarray(d["xyz"], np.float32), np.asarray(d["rgb"], np.float32)], 1))
        t_img.append(np.asarray(d["image_feat"], np.float32).mean(0))
        t_txt["name"].append(np.asarray(d["text_feat"][0]["original"], np.float32)[0])
        t_txt["blip"].append(np.asarray(
            np.asarray(d["blip_caption_feat"]).item()["original"], np.float32)[0])
        t_txt["msft"].append(np.asarray(
            np.asarray(d["msft_caption_feat"]).item()["original"], np.float32)[0])
        rt = d["retrieval_text_feat"]
        t_txt["retrieval"].append(
            np.asarray(rt[0]["original"], np.float32)[0] if len(rt)
            else np.zeros(1280, np.float32))

    # --- our side -------------------------------------------------------
    o_txt = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["text"].astype(np.float32)
                      for u in keep])
    o_img = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["image"].astype(np.float32)
                      for u in keep])
    o_pc = np.load(PC_CACHE)[np.array([pos[u] for u in keep])]

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    t_pc = np.empty((len(keep), 1280), np.float32)
    with torch.no_grad():
        for i in range(0, len(keep), args.batch):
            j = min(i + args.batch, len(keep))
            t_pc[i:j] = bb.encode_pc(np.stack(t_xyzrgb[i:j])).float().cpu().numpy()
    print("encoded their clouds", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = prototypes(bb, names, templates).to(dev)
    tgt = torch.tensor(labels, device=dev)
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)

    res = {"n": len(keep), "classification_acc1_acc5": {}, "text_to_pc_R1": {}}
    print(f"\n{'—— zero-shot LVIS 分類（1,156 類）——':<42s} acc1     acc5")
    for tag, v in (("點雲 · ULIP-2 官方", t_pc), ("點雲 · 我們", o_pc),
                   ("圖片 · ULIP-2 官方 (12 視角平均)", T(np.stack(t_img))),
                   ("圖片 · 我們 (12 視角平均)", T(o_img))):
        a1, a5 = acc(T(v) if isinstance(v, np.ndarray) else v, P, tgt)
        res["classification_acc1_acc5"][tag] = [round(a1, 3), round(a5, 3)]
        print(f"{tag:<42s} {a1:7.2f}  {a5:7.2f}")

    print(f"\n{'—— 文字 → 點雲 R@1（池子 ' + str(len(keep)) + '）——':<42s} 對他們的雲  對我們的雲")
    rows = [("文字 · 物件名 (name)", t_txt["name"]),
            ("文字 · BLIP 描述", t_txt["blip"]),
            ("文字 · 微軟描述", t_txt["msft"]),
            ("文字 · LAION 撈的", t_txt["retrieval"]),
            ("文字 · 我們的 GPT 標註", o_txt)]
    for tag, v in rows:
        q = T(np.stack(v) if isinstance(v, list) else v)
        a, b = r1(q, T(t_pc)), r1(q, T(o_pc))
        res["text_to_pc_R1"][tag] = {"their_pc": round(a, 2), "our_pc": round(b, 2)}
        print(f"{tag:<42s} {a:9.2f}  {b:9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
