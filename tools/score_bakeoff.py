#!/usr/bin/env python
"""Score the n05 bake-off arms and lay their descriptions side by side.

# SUPPORTS-NODE: n05_annotate

    python tools/score_bakeoff.py --arms gemma4_12b qwen38_27b

**The decision criterion is the mean CLIP score of the winning description**
(`USER`, 2026-08-23: 「100個平均分數最高就用它」). Everything else here is
reported beside it and does not decide anything.

What that number is and is not
------------------------------

It is CLIP-ViT-Large's image-text similarity, meaned over the eleven views and
then over the assets -- the same statistic ULIP-2 ranks candidates with
(`main.tex:677`). **It measures how well a sentence matches the pictures, not
whether the sentence is true.** A confident wrong noun can score well if it
looks like the thing. Both arms are scored by the same model on the same
assets, so the comparison between them is fair even where the absolute number
is not an accuracy.

`category_hits` is reported separately, against the Objaverse-LVIS label the
model was never shown. It is a **lower bound**: it counts a shared content
word, so "deer" against "elk" and "trowel" against "shovel" both score as
misses. Do not add it to the decision.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402
from metafind.data.annotate import _tokens  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "workflow/blocks/ULIP2/bakeoff/sample_100.jsonl"


def load_arm(arm: str) -> dict[str, dict]:
    d = paths.OUTPUTS / "bakeoff" / arm / "annotations"
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def summarise(arm: str, recs: dict[str, dict], truth: dict[str, str]) -> dict:
    wins = [r["description_candidates"][0]["clip_score"] for r in recs.values()
            if r.get("description_candidates")]
    # Rank-by-rank means say whether five candidates was enough: if the 5th is
    # still well below the 1st, more draws might still have found a better one.
    by_rank: dict[int, list[float]] = collections.defaultdict(list)
    for r in recs.values():
        for c in r.get("description_candidates", []):
            by_rank[c["rank"]].append(c["clip_score"])

    hits = [u for u, r in recs.items()
            if _tokens(r["category"]) & _tokens(truth.get(u, ""))]
    return {
        "arm": arm,
        "n": len(recs),
        "mean_clip": statistics.mean(wins) if wins else float("nan"),
        "median_clip": statistics.median(wins) if wins else float("nan"),
        "stdev_clip": statistics.stdev(wins) if len(wins) > 1 else 0.0,
        "mean_by_rank": {k: round(statistics.mean(v), 4) for k, v in sorted(by_rank.items())},
        "category_hits": len(hits),
        "one_attempt": sum(1 for r in recs.values() if r.get("attempts") == 1),
        "mean_height_cm": statistics.mean(r["height"] for r in recs.values()),
        "all_placements_false": sum(
            1 for r in recs.values()
            if not any(r[k] for k in ("onCeiling", "onWall", "onFloor", "onObject"))),
        "distinct_categories": len({r["category"].lower() for r in recs.values()}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--show", type=int, default=20,
                    help="assets to print side by side for the USER to read")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    truth = {}
    for line in SAMPLE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            truth[row["uid"]] = row["lvis_category"]

    arms = {a: load_arm(a) for a in args.arms}
    summaries = [summarise(a, r, truth) for a, r in arms.items()]

    print("=" * 92)
    print(f"{'arm':<16}{'n':>5}{'mean CLIP':>12}{'median':>10}{'sd':>8}"
          f"{'cat hits':>10}{'1-attempt':>11}{'distinct cat':>14}")
    print("-" * 92)
    for s in summaries:
        print(f"{s['arm']:<16}{s['n']:>5}{s['mean_clip']:>12.4f}{s['median_clip']:>10.4f}"
              f"{s['stdev_clip']:>8.4f}{s['category_hits']:>10}{s['one_attempt']:>11}"
              f"{s['distinct_categories']:>14}")
    print("=" * 92)

    complete = [s for s in summaries if s["n"]]
    if len(complete) > 1:
        best = max(complete, key=lambda s: s["mean_clip"])
        gap = best["mean_clip"] - min(s["mean_clip"] for s in complete)
        print(f"\nWINNER on the agreed criterion: {best['arm']}  "
              f"mean CLIP {best['mean_clip']:.4f}  (+{gap:.4f})")
        # A gap smaller than the within-arm spread of the per-asset scores is
        # not a result. Saying which is larger costs one line and stops a tie
        # being written up as a finding.
        sd = max(s["stdev_clip"] for s in complete)
        if gap < sd / (min(s["n"] for s in complete) ** 0.5):
            print("  ^ WITHIN NOISE: the gap is smaller than the standard error. "
                  "Report this as a tie, not a winner.")

    print("\ncandidate rank -> mean CLIP (is 5 enough? a flat tail says yes)")
    for s in summaries:
        print(f"  {s['arm']:<16}{s['mean_by_rank']}")

    print("\nplacement flags all false (should be near 0):")
    for s in summaries:
        print(f"  {s['arm']:<16}{s['all_placements_false']} / {s['n']}"
              f"   mean height {s['mean_height_cm']:.0f} cm")

    shared = sorted(set.intersection(*(set(r) for r in arms.values())) if arms else [])
    print(f"\n{'=' * 92}\nSIDE BY SIDE -- {min(args.show, len(shared))} of {len(shared)} "
          f"assets. The descriptions have no answer key; this is the USER's to judge.\n")
    for uid in shared[:args.show]:
        print(f"--- official label: {truth.get(uid, '?')}   uid {uid[:12]}")
        for a in args.arms:
            r = arms[a][uid]
            top = r["description_candidates"][0]["clip_score"]
            print(f"  [{a}]  cat={r['category']!r} synset={r['synset']}  clip={top:.4f}")
            print(f"      {r['description'][:190]}")
        print()

    if args.out:
        args.out.write_text(json.dumps({"summaries": summaries}, indent=1))
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
