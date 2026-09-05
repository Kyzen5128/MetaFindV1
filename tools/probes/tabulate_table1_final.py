#!/usr/bin/env python3
"""Markdown tables from `exp_type_level_query.py` outputs (R@1 / R@5, %), paper rows on top.

    python tools/probes/tabulate_table1_final.py output/look/table1_final_P1s_S1head_holdout.json [more.json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COND = ("text", "image", "pc", "text+image", "text+pc", "image+pc", "full")
PAPER = {
    "paper w/o ESSGNN": {"R@1": (13.8, 11.7, 75.1, 17.2, 44.5, 45.8, 51.7), "R@5": (23.1, 19.2, 78.0, 21.8, 71.3, 73.1, 76.5)},
    "paper w/ ESSGNN":  {"R@1": (11.3, 10.5, 63.2, 15.9, 41.2, 42.0, 48.2), "R@5": (21.5, 15.9, 66.5, 20.3, 68.8, 70.4, 74.9)},
}
ROWS = [  # (key in the JSON, label)
    ("own(attrs) | own view", "own observations (attrs text, own view, own cloud)"),
    ("own(attrs) | own view | pc=resample", "own text + own view, cloud = second surface sample"),
    ("cat_size | thumbnail(own)", "weak own: category+size text, own Sketchfab thumbnail, own cloud"),
    ("cat_size | thumbnail(own) | pc=resample", "weak own trio: category+size, thumbnail, second sample"),
    ("u2 blip caption | thumbnail(own)", "weak own: BLIP caption, own thumbnail, own cloud"),
    ("u2 blip caption | thumbnail(own) | pc=resample", "weak own trio: BLIP caption, thumbnail, second sample"),
    ("partner(attrs) | partner view", "partner: same-category other asset's text + view, own cloud"),
    ("partner(attrs) | partner view | pc=resample", "partner text + view, cloud = second sample"),
]


def fmt(vals):
    return " | ".join(f"{v:.1f}" for v in vals)


def main() -> int:
    for path in sys.argv[1:]:
        d = json.loads(Path(path).read_text())
        head = (f"### {Path(path).stem}  ({d.get('query_split', 'dev_val')} {d['n_query']:,} -> "
                f"{d.get('gallery_split', '?')} {d['n_gallery']:,}; ckpt {Path(d.get('ckpt', '?')).parent.name}"
                + (f"; Stage 2 head {Path(d['stage2_state']).parent.name}" if d.get("stage2_state") else "; Stage 1 head") + ")")
        print(head)
        for metric in ("R@1", "R@5"):
            print(f"\n**{metric} (%)**\n")
            print("| query construction | " + " | ".join(COND) + " |")
            print("|---|" + "---|" * len(COND))
            for name, p in PAPER.items():
                print(f"| {name} | {fmt(p[metric])} |")
            for key, label in ROWS:
                if key in d["rows"]:
                    cells = d["rows"][key]["cells"]
                    print(f"| {label} | {fmt([cells[c][metric] * 100 for c in COND])} |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
