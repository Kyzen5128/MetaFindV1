#!/usr/bin/env python3
"""Short fixed-format text built from the EXISTING annotation fields.

[KYZEN 2026-09-03] "我text欄位有紀錄嗎?像論文那樣 ... 可以利用固定的格式的短描述
去限制嗎?就不用重新生成了."

PAPER FACT, arXiv HTML sec. 2.3, verbatim:
    "These annotations provide rich textual descriptions detailing attributes
     such as object category, size dimensions, materials, and placement
     constraints."
That list is exactly four structured fields, all of which n06 already wrote to
every sidecar. It does NOT mention the long free-form visual description that
`TEXT_TEMPLATE` puts FIRST and that eats most of CLIP's 77-token window
(measured: 68 tokens median, 3.5% of the corpus truncated at the cap).

So the arms below are re-serializations of annotation data we already have. No
regeneration, no new LLM call.

BOTH SIDES are re-encoded, query and gallery. Changing only the query would
measure an observation MISMATCH; Kyzen asked to constrain the corpus format,
which is a change to the text both towers read.

Scoring is `metafind.eval.retrieval` -- the SAME code the Table 1 node uses.
A previous version of this experiment shipped its own recall() and its own
gallery construction; the gallery was built with a ZERO point cloud and ties
counted in the model's favour, and every number it produced was wrong.

[SENSITIVITY, NOT A RETRAIN] One checkpoint trained on the full template,
re-scored under other serializations. This bounds how much of the gap the TEXT
LENGTH can explain; it does not tell us what a model trained on short text does.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metafind import paths
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,
                                     normalize_for_scoring, recall_at_k)
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.train.stage1 import (Stage1Dataset, collate, load_protocols,
                                   modules_in_eval, split_embeds)

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2,
         "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}

# Every arm is `serialize_annotation(a, template=...)`, i.e. the SHIPPING
# serializer with a different template, so no arm can drift from what n06 does.
ARMS = {
    "full": None,                                    # what the corpus uses now
    "attrs_only": "{category} made of {materials}, roughly {width} by {length} "
                  "by {height} centimetres, {placement}.",
    "attrs_coarse": "{category} made of {materials}, roughly {width} by "
                    "{length} by {height} centimetres, {placement}.",
    "no_dims": "{category} made of {materials}, {placement}.",
    "cat_only": "{category}.",
    # --- the three arms that actually follow Figure 2 --------------------
    # [KYZEN 2026-09-03] "metafind論文不只有4項啊" -- correct. Figure 2 prints
    # the annotation record itself, and it carries THIRTEEN fields:
    #   category, synset, width, length, height, volume, mass, description,
    #   materials, onCeiling, onWall, onFloor, onObject
    # every one of which n06 already writes. `description` is among them, so
    # `attrs_only` above (which drops it) is NOT the paper's annotation; it is
    # a length ablation and is kept only as one.
    "figure2_json": "__json__",     # the record itself, serialised as Figure 2
    "figure2_prose": None,          # our template, but Figure 2's number format
    "desc_only": "{description}",
}
# Dimensions rounded to the nearest 10 cm: a length ablation, not the paper.
COARSE = {"attrs_coarse"}
# Figure 2 prints whole-number dimensions ("width": 30, "height": 40) and a
# description that ENDS ON A SENTENCE. Ours renders one decimal place
# (12.1 by 9.4 by 3.5) and hard-cuts at 160 characters, which lands mid-clause
# ("...They are finished."). Both are OBSERVED differences from the figure.
FIGURE2 = {"figure2_json", "figure2_prose"}
# Exactly the keys Figure 2 shows, in the order it shows them.
FIGURE2_KEYS = ("category", "synset", "width", "length", "height", "volume",
                "mass", "description", "materials",
                "onCeiling", "onWall", "onFloor", "onObject")


def _sentence_cap(text: str, limit: int) -> str:
    """Cut on a sentence boundary instead of mid-clause."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:stop + 1] if stop > 40 else cut).strip()


def text_for(ann: dict, arm: str) -> str:
    if arm in COARSE:
        ann = copy.deepcopy(ann)
        for k in ("width", "length", "height"):
            ann[k] = max(10.0, round(float(ann[k]) / 10.0) * 10.0)
    if arm in FIGURE2 or arm == "desc_only":
        ann = copy.deepcopy(ann)
        for k in ("width", "length", "height"):
            ann[k] = float(round(float(ann[k])))
        # 160, not 200: `serialize_annotation` re-caps at MAX_DESCRIPTION_CHARS
        # and would put the mid-clause cut straight back. The json arm keeps the
        # whole description because it does not go through that serializer.
        if arm != "figure2_json":
            ann["description"] = _sentence_cap(ann["description"], 160)
        if arm == "figure2_json":
            rec = {k: (round(ann[k], 2) if isinstance(ann.get(k), float)
                       else ann.get(k)) for k in FIGURE2_KEYS}
            return json.dumps({"annotations": rec}, ensure_ascii=False)
    return serialize_annotation(ann, template=ARMS[arm])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--arm", action="append",
                    help="repeatable; default every arm")
    ap.add_argument("--out", default="output/look/exp_text_length.json")
    args = ap.parse_args()
    arms = args.arm or list(ARMS)

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from tools.probes.exp_query_observation import load_tower

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    missing = set(q_uids) - set(g_uids)
    if missing:
        raise SystemExit(f"{len(missing):,} queries are not in the gallery; "
                         "R@1 against an absent target is undefined")
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    q_rows = torch.tensor(targets)

    bb = ULIPBackbone(BackboneConfig(device=args.device,
                                     train_scope="pointbert_and_fuser"))
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bb.model.load_state_dict(ck["backbone_trainable_state"], strict=False)
    model = load_tower(Path(args.ckpt), args.device)
    enc, _, _ = load_protocols()

    # --- image and point cloud, encoded ONCE and shared by every arm ---------
    print(f"encoding image+pc for {len(g_uids):,} gallery assets "
          "(once, shared by every arm)", flush=True)
    loader = DataLoader(Stage1Dataset(g_uids, enc["image_aggregation"]),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)
    IMG, PC = [], []
    with modules_in_eval(model, getattr(bb, "model", None)), torch.no_grad():
        for i, b in enumerate(loader):
            _q, g = split_embeds(b, bb, args.device)
            IMG.append(g["image"].cpu())
            PC.append(g["pc"].float().cpu())
            if i % 100 == 0:
                print(f"  batch {i}", flush=True)
    IMG, PC = torch.cat(IMG), torch.cat(PC)

    anns = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
            for u in g_uids]
    print("\none asset, every arm:")
    for a in arms:
        print(f"  {a:<14} {text_for(anns[0], a)!r}")

    out, hdr = {}, "arm".ljust(15) + "".join(c.rjust(13) for c in QUERY_CONDITIONS)
    print(f"\n{len(q_uids):,} held-out queries against {len(g_uids):,} gallery "
          "assets. Both sides carry the arm's text.")
    print("paper".ljust(15) + "".join(("%.1f" % PAPER[c]).rjust(13)
                                      for c in QUERY_CONDITIONS))
    print(hdr)

    for arm in arms:
        texts = [text_for(a, arm) for a in anns]
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                vecs.append(bb.encode_text(texts[i:i + 256]).float().cpu())
        TXT = torch.cat(vecs)

        G, Q = [], []
        with modules_in_eval(model, getattr(bb, "model", None)), torch.no_grad():
            for i in range(0, len(g_uids), 512):
                s = slice(i, i + 512)
                e = {"text": TXT[s].to(args.device),
                     "image": IMG[s].to(args.device),
                     "pc": PC[s].to(args.device)}
                G.append(model.gallery(e).float().cpu())
            G = torch.cat(G)
            cells = {}
            for cond in QUERY_CONDITIONS:
                Q = []
                for i in range(0, len(q_uids), 512):
                    r = q_rows[i:i + 512]
                    e = {"text": TXT[r].to(args.device),
                         "image": IMG[r].to(args.device),
                         "pc": PC[r].to(args.device)}
                    m = condition_mask(cond, len(r)).to(args.device)
                    Q.append(model.query(e, present=m).float().cpu())
                qv = normalize_for_scoring(torch.cat(Q).numpy())
                gv = normalize_for_scoring(G.numpy())
                cells[cond] = recall_at_k(qv @ gv.T, targets)
        out[arm] = cells
        print(arm.ljust(15) + "".join(("%.1f" % (cells[c]["R@1"] * 100)).rjust(13)
                                      for c in QUERY_CONDITIONS), flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt,
        "query_split": args.query_split, "gallery_split": args.gallery_split,
        "n_query": len(q_uids), "n_gallery": len(g_uids),
        "paper_w_o_essgnn_R@1_percent": PAPER,
        "templates": ARMS, "coarse_dimension_arms": sorted(COARSE),
        "both_sides_re_encoded": True,
        "scoring": "metafind.eval.retrieval (the Table 1 code path)",
        "caveat": "one checkpoint trained on the `full` template, re-scored "
                  "under other serializations. Not a retrain.",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
