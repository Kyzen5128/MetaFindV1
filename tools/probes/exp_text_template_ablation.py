#!/usr/bin/env python3
"""Which part of our text makes text-only retrieval so easy? Ablate and measure.

[KYZEN 2026-09-03] "想辦法做出差不多的數據 用實驗證明架構符合論文設定."

The observation matrix closed off two axes: no combination of query observation
and gallery scope brings any condition within 3.2x of Table 1. So the difference
is not WHICH observation the query reads. This asks whether it is WHAT OUR TEXT
SAYS.

The serialized string is:

    {description} {category} made of {materials}, roughly {W} by {L} by {H}
    centimetres, {placement}.

and a real one reads

    "... Crossbar made of metal, plastic, roughly 100.2 by 30.9 by 15.2
    centimetres, typically placed on the floor."

**Three decimal-place dimensions in centimetres are close to a fingerprint.**
Two assets sharing a category, a material list and a placement will almost never
share 100.2 x 30.9 x 15.2. Nothing in the paper says its text carried dimensions
at that precision -- `2methdology.tex:28` says the annotations detail "size
dimensions" and stops. Our template is an IMPLEMENTATION CHOICE (D0-008), not a
paper fact, and this measures what that choice bought.

The arms, each a different serialization of the SAME annotation, encoded through
the same frozen tower:

    full          what the corpus uses today
    no_dims       the dimensions clause removed
    no_category   the category name removed
    no_dims_cat   both
    desc_only     the model's free description alone, no template at all

GALLERY UNCHANGED in every arm -- it keeps the canonical full-template vector.
Only the query's text is re-serialized, which is the same discipline as the
observation matrix: one thing varies.

[SENSITIVITY, NOT A RETRAIN] One checkpoint, evaluated many ways. A text
template is a training-time choice; changing it properly means re-encoding the
corpus and retraining. This bounds how much of the gap the TEXT CONTENT can
explain.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.models.resolve_stage1 import serialize_annotation

PAPER = {"text": 13.8, "text+image": 17.2}
ARMS = ("full", "no_dims", "no_category", "no_dims_cat", "desc_only",
        "category_only", "category_materials")
_DIMS = re.compile(r",? roughly [\d.]+ by [\d.]+ by [\d.]+ centimetres")


def serialize(a: dict, arm: str) -> str:
    """One annotation, five ways. Built by EDITING the produced string rather
    than by a second template, so no arm can drift from what n06 actually
    writes -- a parallel implementation is how the two would disagree."""
    s = serialize_annotation(a)
    if arm == "full":
        return s
    if arm == "desc_only":
        return a.get("description") or s
    if arm == "category_only":
        # The shortest thing a person would actually type. Our corpus text is a
        # 200-character machine fingerprint; Table 1's column is called
        # "Text Only" and the paper never says how long that text is.
        return (a.get("category") or "object").strip()
    if arm == "category_materials":
        cat = (a.get("category") or "object").strip()
        mats = ", ".join((a.get("materials") or [])[:3])
        return f"{cat} made of {mats}" if mats else cat
    if arm in ("no_dims", "no_dims_cat"):
        s = _DIMS.sub("", s)
    if arm in ("no_category", "no_dims_cat"):
        cat = (a.get("category") or "").strip()
        if cat:
            # The template capitalises the category and puts it after the
            # description; remove that occurrence only.
            s = s.replace(f"{cat[:1].upper()}{cat[1:]} made of", "Made of", 1)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--gallery-split", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-policy", default="same_mean",
                    choices=("same_mean", "single_view", "disjoint_views"),
                    help="the query image, so the text arms are CROSSED with "
                         "the image axis rather than measured beside it")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="output/look/exp_text_template.json")
    args = ap.parse_args()

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from tools.probes.exp_query_observation import (
        gallery_vectors,
        load_tower,
        recall,
    )

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])[: args.limit] if args.limit else sorted(sp[args.split])
    g_uids = (sorted(set(sp["train"]) | set(sp["test"]))
              if args.gallery_split == "full" else uids)
    where = {u: i for i, u in enumerate(g_uids)}
    target = torch.tensor([where[u] for u in uids]).to(args.device)

    anns = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in uids]
    print("one asset, five ways:")
    for arm in ARMS:
        print(f"  {arm:<12} {serialize(anns[0], arm)[130:250]!r}")

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    model = load_tower(Path(args.ckpt), args.device)
    gal = gallery_vectors(g_uids, args.device)
    from tools.probes.exp_observation_matrix import query_images
    q_img = query_images(uids, args.image_policy, args.device)

    print(f"\n{len(uids):,} queries against {len(g_uids):,} gallery items. "
          "The GALLERY keeps the full template in every arm.")
    print(f"paper: text {PAPER['text']} · text+image {PAPER['text+image']}\n")
    print("arm".ljust(14) + "text".rjust(20) + "text+image".rjust(20))

    out = {}
    for arm in ARMS:
        texts = [serialize(a, arm) for a in anns]
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                vecs.append(bb.encode_text(texts[i:i + 64]).float().cpu().numpy())
        t = torch.from_numpy(np.concatenate(vecs)).to(args.device)
        row, cells = arm.ljust(14), {}
        for cond, present in (("text", (True, False)),
                              ("text+image", (True, True))):
            r1, r5 = recall(model, t, q_img, gal, present, args.device, target)
            cells[cond] = {"R@1": r1, "R@5": r5}
            row += ("%.1f  (%.1fx)" % (r1 * 100, r1 * 100 / PAPER[cond])).rjust(20)
        print(row, flush=True)
        out[arm] = cells

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt, "n_query": len(uids), "n_gallery": len(g_uids),
        "paper_w_o_essgnn_R@1_percent": PAPER,
        "image_policy": args.image_policy,
        "gallery": "unchanged (full template) in every arm; only the QUERY text "
                   "is re-serialized",
        "caveat": "one checkpoint trained on the full template, evaluated with "
                  "other serializations. A template is a training-time choice; "
                  "changing it properly means re-encoding and retraining.",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
