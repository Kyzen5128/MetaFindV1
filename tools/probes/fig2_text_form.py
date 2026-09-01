#!/usr/bin/env python3
"""What the paper's Figure 2 string actually scores, once its numbers are its own.

`two_deviations.py` priced "Figure 2's JSON instead of our prose" at R@1 3.13
against our prose's 22.50, with the paper at 13.8, and concluded the JSON form
was worth about nineteen points. That arm was built wrong. It serialised OUR
stored numbers, which are unrounded:

    "width": 12.05325294373528, "length": 9.414963237058345,
    "volume": 397.1832667328153

CLIP's context is 77 tokens. Two seventeen-digit floats spend it before
`description` begins, so the 3.13 measured a string that had been cut down to
roughly a class name -- not the paper's format, an artefact of our precision.

Figure 2 prints its own numbers, and they are round:

    "category": "robot", "synset": "robot.n.01", "width": 30, "length": 30,
    "height": 40, "volume": 36000, "mass": 2.5, "description": "A small
    cubic-shaped robot with a smiling screen face, ...", "materials":
    ["metal", "glass", "plastic"], "onCeiling": false, "onWall": false,
    "onFloor": true, "onObject": true

Integers for the three dimensions and the volume, one decimal for mass. With
that precision the description survives, and the JSON form is a different
string entirely from the one that scored 3.13.

THE ARMS
--------
  prod_prose          what production encodes today (`TEXT_TEMPLATE`)
  fig2_json_raw       Figure 2's field order, OUR unrounded numbers  (the 3.13)
  fig2_json_rounded   Figure 2's field order AND Figure 2's precision

Token counts are reported beside every R@1, because on a 77-token budget the
count is the mechanism and the score is only its consequence.

WHAT IS AND IS NOT ESTABLISHED
------------------------------
PAPER FACT: Figure 2 prints that field set and those round numbers.
IMPLEMENTATION CHOICE: reading them as a rounding RULE. The figure shows one
example and states no rule; a 30x30x40 robot may simply have had round
dimensions. So `fig2_json_rounded` is a measurement of a plausible reading, not
a reproduction of a stated one.
UNKNOWN, and not addressed here: whether MetaFind encodes the JSON at all.
`2methdology.tex:28` says the annotations are produced by GPT-4o and says
nothing about what string reaches the text tower.

Protocol is `text_ladder_seven_pcgallery`'s, so the numbers line up with it:
the 9,138 test uids as queries, the point-cloud embedding of all 45,692 as the
gallery, released encoder, no Stage 1 weights.
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
from metafind.models import resolve_stage1 as R  # noqa: E402

OUT = REPO / "output" / "look" / "fig2_text_form.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
ANN = paths.OUTPUTS / "annotations"
# The order Figure 2 prints, so a truncation falls where the paper's would.
FIG2_KEYS = ["category", "synset", "width", "length", "height", "volume", "mass",
             "description", "materials", "onCeiling", "onWall", "onFloor",
             "onObject"]
PAPER_METAFIND_R1 = 13.8


def fig2_json(ann: dict, rounded: bool) -> str:
    """Figure 2's field order. `rounded` uses Figure 2's own precision."""
    d = {}
    for k in FIG2_KEYS:
        v = ann.get(k)
        if rounded and k in ("width", "length", "height", "volume"):
            v = int(round(float(v)))
        elif rounded and k == "mass":
            v = round(float(v), 1)
        d[k] = v
    return json.dumps({"annotations": d}, separators=(", ", ": "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    print(f"queries {len(queries):,}   gallery {len(corpus):,}", flush=True)

    anns = {u: json.loads((ANN / f"{u}.json").read_text()) for u in queries}
    arms = {
        "prod_prose": [R.serialize_annotation(anns[u]) for u in queries],
        "fig2_json_raw": [fig2_json(anns[u], False) for u in queries],
        "fig2_json_rounded": [fig2_json(anns[u], True) for u in queries],
    }

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    import open_clip
    tok = open_clip.get_tokenizer("ViT-bigG-14")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = torch.nn.functional.normalize(
        torch.from_numpy(np.load(PC_CACHE)).float().to(dev), dim=-1)
    tgt = torch.tensor([pos[u] for u in queries], device=dev)

    res = {"n_query": len(queries), "n_gallery": len(corpus),
           "paper_metafind_R1": PAPER_METAFIND_R1, "arms": {}}
    print(f"\n{'arm':<20s}{'tokens 平均':>12s}{'滿 77 的比例':>14s}"
          f"{'R@1':>9s}{'R@5':>9s}")
    for name, strings in arms.items():
        counts = np.array([int((tok([s])[0] != 0).sum()) for s in strings])
        E = torch.empty(len(strings), 1280, device=dev)
        with torch.no_grad():
            for i in range(0, len(strings), args.batch):
                j = min(i + args.batch, len(strings))
                E[i:j] = bb.encode_text(strings[i:j]).float()
        E = torch.nn.functional.normalize(E, dim=-1)
        s = E @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        higher = (s > own).sum(1)
        r1 = (higher < 1).sum().item() / len(strings) * 100
        r5 = (higher < 5).sum().item() / len(strings) * 100
        sat = float((counts >= 77).mean()) * 100
        res["arms"][name] = {"R@1": round(r1, 2), "R@5": round(r5, 2),
                             "tokens_mean": round(float(counts.mean()), 1),
                             "tokens_median": int(np.median(counts)),
                             "pct_truncated": round(sat, 2),
                             "example": strings[0][:400]}
        print(f"{name:<20s}{counts.mean():12.1f}{sat:13.1f}%{r1:9.2f}{r5:9.2f}")
    print(f"{'論文 MetaFind':<20s}{'':>12s}{'':>14s}{PAPER_METAFIND_R1:9.1f}")

    print("\n第一筆查詢在三種寫法下長什麼樣：")
    for name in arms:
        print(f"\n--- {name}  ({res['arms'][name]['tokens_mean']:.0f} tokens 平均) ---")
        print(res["arms"][name]["example"][:300])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
