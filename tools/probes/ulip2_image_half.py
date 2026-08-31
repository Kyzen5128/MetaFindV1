#!/usr/bin/env python3
"""The half of ULIP-2 that 50.576 never touched: the image encoder.

WHY
---
Kyzen, 2026-09-01: finish ULIP-2 itself before touching anything MetaFind-side.
He is right that it is not finished. The upstream reproduction --
`{'acc1': 50.5756, 'acc5': 78.9307}` against a published 50.6 / 79.1 -- runs
`test_zeroshot_3d_core`, whose entire data path is

    point cloud -> point encoder
    category name x 64 templates -> text encoder -> normalise, mean, normalise
    cosine, top-1/top-5

**No image is loaded anywhere in it.** So that number verifies the point-cloud
half and the category-name half of the text encoder, and says nothing at all
about our renders, our image preprocessing, or `encode_image`.

WHAT THIS DOES
--------------
The same zero-shot classification, with the image embedding substituted for the
point-cloud embedding against the identical 1,156 Objaverse-LVIS prototypes.
It is CLIP zero-shot on our own renders, scored by ULIP-2's own text tower.
If the render pipeline and preprocessing are sound the number is high; if
anything upstream of the cached vector is wrong it collapses toward chance
(1/1156 = 0.087%).

Three image constructions are scored, because they are not interchangeable:
the 12-view mean we currently store, a single view, and each view in turn.

It also reports the TEXT side that classification never exercises: how many of
our GPT-4o descriptions hit CLIP's 77-token ceiling. A truncated caption is a
silently different input from the one the annotation file records.

Nothing here is MetaFind. No fusion, no dual tower, no Stage 1 weights --
released ULIP-2 only, and cached vectors that were produced by it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "ulip2_image_half.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
LVIS = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/objaverse-lvis/"
                    "objaverse_lvis_metadata.json")
TEMPLATES = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/templates.json")


def corpus_uids() -> list[str]:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    return sorted(set(d["train"]) | set(d["test"]))


def category_prototypes(bb, names: list[str], templates: list[str],
                        batch: int = 64) -> torch.Tensor:
    """Upstream's recipe verbatim: format, encode, normalise, mean, normalise.

    `ULIP/main.py:375-384`. Reimplemented rather than imported because upstream's
    version is welded into its own evaluation loop, but the four operations are
    the same four in the same order.
    """
    out = []
    with torch.no_grad():
        for i, name in enumerate(names):
            e = bb.encode_text([t.format(name) for t in templates]).float()
            e = torch.nn.functional.normalize(e, dim=-1)
            e = e.mean(dim=0)
            out.append(torch.nn.functional.normalize(e, dim=-1))
            if (i + 1) % 200 == 0:
                print(f"  prototypes {i + 1}/{len(names)}", flush=True)
    return torch.stack(out)


def topk(feats: torch.Tensor, protos: torch.Tensor, target: torch.Tensor,
         chunk: int = 2048) -> tuple[float, float]:
    feats = torch.nn.functional.normalize(feats, dim=-1)
    h1 = h5 = 0
    for i in range(0, feats.shape[0], chunk):
        j = min(i + chunk, feats.shape[0])
        logits = feats[i:j] @ protos.t()
        own = logits.gather(1, target[i:j].unsqueeze(1))
        higher = (logits > own).sum(dim=1)
        h1 += int((higher < 1).sum())
        h5 += int((higher < 5).sum())
    n = feats.shape[0]
    return h1 / n * 100, h5 / n * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-set", default="modelnet40_64",
                    help="upstream's default; its own LVIS script does not "
                         "override --validate_dataset_prompt (main.py:41)")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = json.loads(LVIS.read_text())
    names = meta["all_keys"]
    v2k, k2id = meta["value_to_key_mapping"], meta["key_to_id"]
    templates = json.loads(TEMPLATES.read_text())[args.prompt_set]
    print(f"{len(names)} categories, {len(templates)} templates", flush=True)

    uids = corpus_uids()
    if args.limit:
        uids = uids[:args.limit]

    # labels: uid -> LVIS category id. An asset we cannot label is dropped and
    # counted, never silently scored against an arbitrary class.
    keep, labels, unlabelled = [], [], 0
    for u in uids:
        name = v2k.get(u)
        if name is None or name not in k2id:
            unlabelled += 1
            continue
        keep.append(u)
        labels.append(k2id[name])
    print(f"{len(keep):,} labelled, {unlabelled:,} without an LVIS label", flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(keep)
    img_mean = np.empty((n, 1280), dtype=np.float32)
    img_one = np.empty((n, 1280), dtype=np.float32)
    views = np.empty((n, 12, 1280), dtype=np.float32)
    tok_lens, truncated = [], 0
    for i, u in enumerate(keep):
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        img_mean[i] = z["image"].astype(np.float32)
        v = z["views"].astype(np.float32)
        views[i] = v
        img_one[i] = v[rng.integers(v.shape[0])]
        rec = json.loads((paths.EMBEDDINGS / f"{u}.json").read_text())
        tok_lens.append(rec.get("text_tokens", -1))
        truncated += bool(rec.get("text_truncated"))
        if (i + 1) % 10000 == 0:
            print(f"  cached {i + 1:,}/{n:,}", flush=True)

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    protos = category_prototypes(bb, names, templates).to(dev)
    target = torch.tensor(labels, device=dev)

    res = {}
    for tag, arr in (("image_12view_mean", img_mean), ("image_single_view", img_one)):
        a1, a5 = topk(torch.from_numpy(arr).to(dev), protos, target)
        res[tag] = {"acc1": round(a1, 4), "acc5": round(a5, 4)}
        print(f"{tag:20s} acc1 {a1:7.4f}  acc5 {a5:7.4f}", flush=True)

    per_view = []
    for k in range(views.shape[1]):
        a1, a5 = topk(torch.from_numpy(views[:, k]).to(dev), protos, target)
        per_view.append({"view": k, "acc1": round(a1, 4), "acc5": round(a5, 4)})
        print(f"  view {k:2d}          acc1 {a1:7.4f}  acc5 {a5:7.4f}", flush=True)
    res["per_view"] = per_view

    if PC_CACHE.exists():
        pc = np.load(PC_CACHE)
        pos = {u: i for i, u in enumerate(corpus_uids())}
        rows = np.array([pos[u] for u in keep])
        a1, a5 = topk(torch.from_numpy(pc[rows]).to(dev), protos, target)
        res["pc_reference"] = {"acc1": round(a1, 4), "acc5": round(a5, 4)}
        print(f"{'pc (reference)':20s} acc1 {a1:7.4f}  acc5 {a5:7.4f}", flush=True)

    tl = np.array([t for t in tok_lens if t >= 0])
    res["text_tokens"] = {
        "n": int(tl.size), "mean": float(tl.mean()), "max": int(tl.max()),
        "at_or_over_77": int((tl >= 77).sum()),
        "flagged_truncated": truncated,
        "note": "CLIP's context length is 77. A description at the ceiling is "
                "not the description the annotation file records.",
    }
    print(f"\ntext tokens: mean {tl.mean():.1f}  max {tl.max()}  "
          f">=77: {(tl >= 77).sum():,}  flagged truncated: {truncated:,}", flush=True)

    payload = {
        "what": "zero-shot LVIS classification with the IMAGE embedding in place "
                "of the point cloud -- the half upstream's 50.576 never touches",
        "encoder": "released ULIP-2, no Stage 1 weights",
        "prompt_set": args.prompt_set, "n_categories": len(names),
        "n_scored": len(keep), "n_unlabelled": unlabelled,
        "chance_acc1": round(100 / len(names), 4),
        "upstream_pc_reference": {"acc1": 50.5756, "acc5": 78.9307,
                                  "source": "ULIP_run on our clouds, DL-053"},
        "results": res,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
