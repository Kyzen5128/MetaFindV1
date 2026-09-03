#!/usr/bin/env python3
"""Every retrain arm against Table 1, all fourteen cells, one table.

[KYZEN 2026-09-03] "所有實驗應比較 Table 1 全七個 modality conditions x R@1 + R@5
... 不要只拿 Text R@1 判斷." Reads each arm's `table1.json` from the official
evaluator (metafind.eval.run_retrieval) and prints, per protocol, the seven
R@1 and seven R@5 cells beside the paper's MetaFind w/o ESSGNN row, plus one
distance: the mean over the 14 cells of |ln(ours / paper)|. A distance is a
ranking device between arms, not a claim that any arm is the paper's protocol.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

C = ("text", "image", "pc", "text+image", "text+pc", "image+pc", "full")
PAPER = {"R@1": dict(zip(C, (13.8, 11.7, 75.1, 17.2, 44.5, 45.8, 51.7))),
         "R@5": dict(zip(C, (23.1, 19.2, 78.0, 21.8, 71.3, 73.1, 76.5)))}
ROOTS = ["/home/kyzen/metafind_data"] + sorted(glob.glob("/home/kyzen/metafind_data_*"))
LABEL = {
    "pilot10b_official_CD": "pilot10b: same_record, v2_cm text, 12-view mean, raw inputs",
    "eval_pilotP1_attrs_singleview_prefnorm_20260903": "P1: attrs_v1 text + single_view query + prefusion L2",
    "eval_pilotP4_shared_fusion_20260903": "P4: P1 + ONE shared Fusion",
    "eval_pilotP3_view_tokens_20260903": "P3: P1 + 12 view tokens into Fusion",
    "eval_pilotP5": "P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2",
    "eval_pilotP7_no_prefusion_norm_20260903": "P7: P1 with prefusion L2 OFF",
    "eval_pilotP6_random_view_20260903": "P6: P1 with a fresh query view per step",
}


def distance(cells: dict) -> float:
    return sum(abs(math.log(max(cells[m][c], 1e-4) / PAPER[m][c]))
               for m in ("R@1", "R@5") for c in C) / 14


def main() -> int:
    arms = []
    for root in ROOTS:
        for p in sorted(glob.glob(f"{root}/outputs/eval/*/table1.json")):
            name = Path(p).parent.name
            if name not in LABEL:
                continue
            t = json.load(open(p))
            for proto, res in t["protocols"].items():
                cells = {m: {c: res["conditions"][c][m] * 100 for c in C} for m in ("R@1", "R@5")}
                arms.append((LABEL[name], proto, res["n_gallery"], cells, distance(cells)))
    out = ["# Stage 1 arms vs Table 1 (MetaFind w/o ESSGNN) -- fourteen cells",
           "",
           "Official evaluator (float64 cosine, ties against the model). C: 4,569 dev_val queries vs the 4,569 dev_val gallery. D: same queries vs the 36,554 train gallery. Paper: 20% test queries on Objaverse-LVIS, gallery size unstated. Distance = mean over the 14 cells of |ln(ours/paper)|; a ranking device, not a protocol claim.",
           ""]
    for proto in ("D_dev_val_vs_train", "C_dev_selection"):
        rows = [a for a in arms if a[1] == proto]
        if not rows:
            continue
        out += [f"## {proto}", "",
                "| arm | dist | " + " | ".join(C) + " |", "|---|---|" + "---|" * len(C),
                "| **paper R@1** | 0 | " + " | ".join(f"**{PAPER['R@1'][c]:.1f}**" for c in C) + " |",
                "| **paper R@5** | 0 | " + " | ".join(f"**{PAPER['R@5'][c]:.1f}**" for c in C) + " |"]
        for label, _, n, cells, d in sorted(rows, key=lambda a: a[4]):
            out.append(f"| {label} (n={n:,}) R@1 | {d:.2f} | " + " | ".join(f"{cells['R@1'][c]:.1f}" for c in C) + " |")
            out.append(f"| ↳ R@5 | | " + " | ".join(f"{cells['R@5'][c]:.1f}" for c in C) + " |")
        out.append("")
    out += ["## ULIP row hypothesis (2026-09-03): does a category-only text query explain ULIP's 0.1?", "",
            "Released ULIP-2, no training, gallery = PC embedding, query = raw mean of the available embeddings, 36,554 gallery, 4,569 queries (R@1):", "",
            "| text arm | text | image | pc | T+I | T+PC | I+PC | full |", "|---|---|---|---|---|---|---|---|",
            "| **paper, ULIP row** | **0.1** | **0.1** | **97.9** | **0.0** | **33.9** | **22.6** | **6.4** |",
            "| category only | 3.8 | 58.4 | 100.0 | 38.6 | 98.8 | 98.6 | 97.0 |",
            "| form-fill (attrs) | 4.7 | 58.4 | 100.0 | 39.7 | 98.7 | 98.6 | 96.6 |",
            "| description only | 24.1 | 58.4 | 100.0 | 52.8 | 98.4 | 98.6 | 96.5 |",
            "| full template | 24.5 | 58.4 | 100.0 | 52.3 | 98.6 | 98.6 | 96.6 |", "",
            "Category-only moves the text cell from 24.5 to 3.8 (paper 0.1) and nothing else: T+PC and full stay 98.7 / 96.6 in every arm against 33.9 / 6.4, because a query pc identical to the gallery pc dominates any mean. The paper's shape -- adding text or image to pc HURTS -- needs the query's pc and image to sit far from the gallery's. INFERENCE; the paper says what neither row was fed."]
    Path("output/look/ARMS_TABLE.md").write_text("\n".join(out) + "\n")
    print("\n".join(l for l in out if l.startswith("| ") or l.startswith("## ") or l.startswith("| **paper R@1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
