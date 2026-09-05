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
ROOTS = ["/home/kyzen/metafind/metafind_data"] + sorted(glob.glob("/home/kyzen/metafind/metafind_data_*"))
LABEL = {
    "pilot10b_official_CD": "pilot10b: same_record, v2_cm text, 12-view mean, raw inputs",
    "eval_pilotP1_attrs_singleview_prefnorm_20260903": "P1: attrs_v1 text + single_view query + prefusion L2",
    "eval_pilotP4_shared_fusion_20260903": "P4: P1 + ONE shared Fusion",
    "eval_pilotP3_view_tokens_20260903": "P3: P1 + 12 view tokens into Fusion",
    "eval_pilotP5": "P5: desc_v1 text; query = alternate description + resampled pc + single view; prefusion L2",
    "eval_pilotP7_no_prefusion_norm_20260903": "P7: P1 with prefusion L2 OFF",
    "eval_pilotP6_random_view_20260903": "P6: P1 with a fresh query view per step",
}
# Stage 2 heads over the P1 parent, layout off: the Table 1 'w/ ESSGNN' row.
STAGE2 = {
    "eval_stage2_pilot2_full_over_P1": "S2 pilot 2: full-T/I/P query, flat 5e-4, 1 ep / 1,500 houses",
    "eval_stage2_S2C_textonly_ft5e-5": "S2-C: text-only query, 5e-5 warmup+cosine, 1 ep / 1,500 houses",
    "eval_stage2_S2D_none_ft5e-5": "S2-D: full-T/I/P query, 5e-5 warmup+cosine, 1 ep / 1,500 houses",
}
PAPER_W = {"R@1": dict(zip(C, (11.3, 10.5, 63.2, 15.9, 41.2, 42.0, 48.2))),
           "R@5": dict(zip(C, (21.5, 15.9, 66.5, 20.3, 68.8, 70.4, 74.9)))}
PARENT_DIR = "eval_pilotP1_attrs_singleview_prefnorm_20260903"


def d_level(cells: dict) -> float:
    """mean |ln(ours/paper)| over the 14 cells: absolute level error."""
    return sum(abs(math.log(max(cells[m][c], 1e-4) / PAPER[m][c]))
               for m in ("R@1", "R@5") for c in C) / 14


def d_shape(cells: dict) -> float:
    """[GPT via Kyzen 2026-09-03] the same, with each table's overall level
    removed: compare the seven conditions' RELATIVE pattern. R@1 and R@5 each
    centred on their own mean log, then averaged."""
    tot = 0.0
    for m in ("R@1", "R@5"):
        ours = [math.log(max(cells[m][c], 1e-4)) for c in C]
        pap = [math.log(PAPER[m][c]) for c in C]
        mo, mp = sum(ours) / 7, sum(pap) / 7
        tot += sum(abs((o - mo) - (q - mp)) for o, q in zip(ours, pap)) / 7
    return tot / 2


def diagnostics(r1: dict, r5: dict) -> dict:
    """The interactions Table 1 is most discriminating on."""
    return {"T+PC/PC": r1["text+pc"] / r1["pc"],
            "I+PC/PC": r1["image+pc"] / r1["pc"],
            "Full/PC": r1["full"] / r1["pc"],
            "T+I/max(T,I)": r1["text+image"] / max(r1["text"], r1["image"]),
            "R@5/R@1": (sum(r5.values()) / 7) / (sum(r1.values()) / 7)}


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
                arms.append((LABEL[name], proto, res["n_gallery"], cells,
                             d_level(cells), d_shape(cells)))
    out = ["# Stage 1 arms vs Table 1 (MetaFind w/o ESSGNN) -- fourteen cells",
           "",
           "Official evaluator (float64 cosine, ties against the model). C: 4,569 dev_val queries vs the 4,569 dev_val gallery. D: same queries vs the 36,554 train gallery. Paper: 20% test queries on Objaverse-LVIS, gallery size unstated.",
           "",
           "Two scores per arm, never merged: **level** = mean |ln(ours/paper)| over the 14 cells; **shape** = the same after removing each table's overall level (R@1 and R@5 centred separately), i.e. only the relative pattern of the seven conditions. Plus the interaction ratios Table 1 is most discriminating on. Ranking devices, not protocol claims.",
           ""]
    pd = diagnostics(PAPER["R@1"], PAPER["R@5"])
    for proto in ("D_dev_val_vs_train", "C_dev_selection"):
        rows = [a for a in arms if a[1] == proto]
        if not rows:
            continue
        out += [f"## {proto}", "",
                "| arm | level | shape | " + " | ".join(C) + " |", "|---|---|---|" + "---|" * len(C),
                "| **paper R@1** | 0 | 0 | " + " | ".join(f"**{PAPER['R@1'][c]:.1f}**" for c in C) + " |",
                "| **paper R@5** | | | " + " | ".join(f"**{PAPER['R@5'][c]:.1f}**" for c in C) + " |"]
        for label, _, n, cells, dl, ds in sorted(rows, key=lambda a: a[5]):
            out.append(f"| {label} (n={n:,}) R@1 | {dl:.2f} | {ds:.2f} | " + " | ".join(f"{cells['R@1'][c]:.1f}" for c in C) + " |")
            out.append(f"| ↳ R@5 | | | " + " | ".join(f"{cells['R@5'][c]:.1f}" for c in C) + " |")
        keys = list(pd)
        out += ["", "Interaction ratios (R@1):", "",
                "| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys),
                "| **paper** | " + " | ".join(f"**{pd[k]:.2f}**" for k in keys) + " |"]
        for label, _, n, cells, dl, ds in sorted(rows, key=lambda a: a[5]):
            dg = diagnostics(cells["R@1"], cells["R@5"])
            out.append(f"| {label} | " + " | ".join(f"{dg[k]:.2f}" for k in keys) + " |")
        out.append("")
    out += ["## ULIP row hypothesis (2026-09-03): does a category-only text query explain ULIP's 0.1?", "",
            "Released ULIP-2, no training, gallery = PC embedding, query = raw mean of the available embeddings, 36,554 gallery, 4,569 queries (R@1):", "",
            "| text arm | text | image | pc | T+I | T+PC | I+PC | full |", "|---|---|---|---|---|---|---|---|",
            "| **paper, ULIP row** | **0.1** | **0.1** | **97.9** | **0.0** | **33.9** | **22.6** | **6.4** |",
            "| category only | 3.8 | 58.4 | 100.0 | 38.6 | 98.8 | 98.6 | 97.0 |",
            "| form-fill (attrs) | 4.7 | 58.4 | 100.0 | 39.7 | 98.7 | 98.6 | 96.6 |",
            "| description only | 24.1 | 58.4 | 100.0 | 52.8 | 98.4 | 98.6 | 96.5 |",
            "| full template | 24.5 | 58.4 | 100.0 | 52.3 | 98.6 | 98.6 | 96.6 |", "",
            "Category-only moves the text cell from 24.5 to 3.8 (paper 0.1) and nothing else: T+PC and full stay 98.7 / 96.6 in every arm against 33.9 / 6.4. Per-modality L2 before the mean does not change that either (T+PC 99.3, full 98.0, every text arm): with q = (p + t)/|p + t| and the gallery's own p, the own score (1 + p.t) exceeds every other (p.p_j + t.p_j) unless t prefers asset j over the own asset by MORE than the pc margin 1 - p.p_j, so an uninformative text cannot flip the ranking, only lower every score together. The paper's shape needs the query's pc (or image) to sit far from the gallery's own, or a text that is systematically anti-informative. INFERENCE; the paper says what neither row was fed. No ln-ratio score for this row (paper has a 0.0 cell); read the table."]
    # ---- Stage 2: w/ ESSGNN rows, and how much each lost against its parent --
    s2 = []
    parent = {}
    for root in ROOTS:
        for p in glob.glob(f"{root}/outputs/eval/*/table1.json"):
            name = Path(p).parent.name
            if name == PARENT_DIR:
                t = json.load(open(p))
                parent = {proto: {m: {c: res["conditions"][c][m] * 100 for c in C} for m in ("R@1", "R@5")}
                          for proto, res in t["protocols"].items()}
            if name in STAGE2:
                t = json.load(open(p))
                for proto, res in t["protocols"].items():
                    cells = {m: {c: res["conditions"][c][m] * 100 for c in C} for m in ("R@1", "R@5")}
                    s2.append((STAGE2[name], proto, cells))
    if s2 and parent:
        out += ["## Stage 2: the Table 1 'w/ ESSGNN' row (Stage 2 query head over the P1 parent, layout off)", "",
                "Paper: w/o 13.8/11.7/75.1/17.2/44.5/45.8/51.7 -> w/ 11.3/10.5/63.2/15.9/41.2/42.0/48.2. The ratio row is w/ divided by w/o, per cell: the paper loses 7-16%; the arms below are 1 epoch over 1,500 of 9,600 houses, so their loss is a lower bound on what a full Stage 2 would do.", ""]
        pr = {c: PAPER_W["R@1"][c] / PAPER["R@1"][c] for c in C}
        for proto in ("D_dev_val_vs_train", "C_dev_selection"):
            rows = [r for r in s2 if r[1] == proto]
            if not rows or proto not in parent:
                continue
            out += [f"### {proto}", "", "| head | " + " | ".join(C) + " |", "|---|" + "---|" * len(C),
                    "| **paper w/ ESSGNN R@1** | " + " | ".join(f"**{PAPER_W['R@1'][c]:.1f}**" for c in C) + " |",
                    "| **paper w/ ÷ w/o** | " + " | ".join(f"**{pr[c]:.2f}**" for c in C) + " |",
                    "| P1 parent (w/o) R@1 | " + " | ".join(f"{parent[proto]['R@1'][c]:.1f}" for c in C) + " |"]
            for label, _, cells in rows:
                out.append(f"| {label} R@1 | " + " | ".join(f"{cells['R@1'][c]:.1f}" for c in C) + " |")
                out.append("| ↳ ÷ parent | " + " | ".join(f"{cells['R@1'][c] / max(parent[proto]['R@1'][c], 1e-6):.2f}" for c in C) + " |")
            out.append("")
    Path("output/look/ARMS_TABLE.md").write_text("\n".join(out) + "\n")
    print("\n".join(l for l in out if l.startswith("| ") or l.startswith("## ") or l.startswith("| **paper R@1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
