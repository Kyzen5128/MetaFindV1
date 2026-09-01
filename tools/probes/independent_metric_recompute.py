#!/usr/bin/env python3
"""Recompute every metric from the saved ranks, importing none of the evaluator.

Codex, 2026-09-01, on what A and C need to stop being "provisional":

> an independent metric recomputation that does not import the evaluator

The point is that a bug in the ranking code must not be able to reproduce
itself inside its own check. So this file imports **no torch, no numpy, and
nothing from `metafind` or `tools`** -- only `json`, `math`, `pathlib` and
`argparse` from the standard library. It reads the per-query rank that each
experiment wrote to its `.jsonl` and recomputes

    MRR    = mean(1 / rank)
    R@k    = fraction with rank <= k
    NDCG@5 = mean over queries of DCG@5 / IDCG@5

from the definitions, then compares against the number the experiment
recorded in its own JSON. A disagreement means one of the two is wrong and
says which cell.

WHAT THIS CAN AND CANNOT CATCH
------------------------------
It catches an arithmetic or aggregation error in the evaluator: a wrong
denominator, an off-by-one in the rank, a mis-summed NDCG, a metric computed
over the wrong subset.

It CANNOT catch a wrong rank. The ranks are the evaluator's own output, so if
the similarity matrix or the target index were wrong, both sides would be
wrong together and agree. That failure mode is covered elsewhere and by a
different instrument -- the UID derangement control in experiment C, which
collapses every direction to a median target rank near half the gallery.

NDCG@5, and why the formula differs between experiments
--------------------------------------------------------
With ONE positive per query -- experiments B2 and C -- DCG@5 has at most one
non-zero term and IDCG@5 = 1, so NDCG@5 reduces to 1/log2(rank+1) when
rank <= 5 and 0 otherwise.

With SEVERAL positives per query -- experiment B, where a model owns about five
caption rows -- the saved rank is the BEST over the positives and the other
positives' positions were never recorded. NDCG@5 is therefore NOT recoverable
from the ranks alone, and this probe reports it as unrecoverable rather than
computing the single-positive formula and calling it agreement.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
LOOK = REPO / "output" / "look"
OUT = LOOK / "independent_metric_recompute.json"

# (label, rows file, result json, key path to the results dict, one positive?)
JOBS = [
    ("C  our corpus 45,692",
     "exp_c_our_corpus_rows.jsonl",
     "exp_c_our_corpus_pure_modality.json", ("results",), True),
    ("B2 CAMERA protocol, released",
     "exp_b2_camera_protocol_rows_released_pair.jsonl",
     "exp_b2_camera_protocol_released_pair.json", ("results",), True),
    ("B2 CAMERA protocol, their ckpt",
     "exp_b2_camera_protocol_rows_camera_pair.jsonl",
     "exp_b2_camera_protocol_camera_pair.json", ("results",), True),
    ("B  Text2Shape official test",
     "exp_b_text2shape_rows.jsonl",
     "exp_b_text2shape_pure_modality.json", ("results",), False),
]


def metrics(ranks: list[int], one_positive: bool) -> dict:
    n = len(ranks)
    out = {
        "mrr": round(sum(1.0 / r for r in ranks) / n * 100, 2),
        "r1": round(sum(r <= 1 for r in ranks) / n * 100, 2),
        "r5": round(sum(r <= 5 for r in ranks) / n * 100, 2),
        "r10": round(sum(r <= 10 for r in ranks) / n * 100, 2),
        "n": n,
    }
    if one_positive:
        out["ndcg@5"] = round(
            sum(1.0 / math.log2(r + 1) for r in ranks if r <= 5) / n * 100, 2)
    else:
        out["ndcg@5"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=0.011,
                    help="both sides are rounded to 2 dp, so anything at or "
                         "below one unit in the last place is agreement")
    args = ap.parse_args()

    report = {"tolerance": args.tol, "imports": ["json", "math", "pathlib",
                                                 "argparse"], "jobs": {}}
    worst = 0.0
    bad = []
    for label, rows_name, json_name, path, one_pos in JOBS:
        rp, jp = LOOK / rows_name, LOOK / json_name
        if not rp.exists() or not jp.exists():
            print(f"{label}: 缺檔，跳過")
            report["jobs"][label] = {"skipped": f"missing {rp.name}/{jp.name}"}
            continue
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        rec = json.loads(jp.read_text())
        for k in path:
            rec = rec[k]
        by_dir: dict[str, list[int]] = {}
        for r in rows:
            for d, v in r["rank"].items():
                by_dir.setdefault(d, []).append(int(v))
        print(f"\n=== {label} ===  {len(rows):,} 列")
        print(f"{'方向':<20s}{'指標':>9s}{'重算':>9s}{'原報告':>9s}{'差':>8s}")
        job = {}
        for d, ranks in by_dir.items():
            mine = metrics(ranks, one_pos)
            theirs = rec.get(d, {})
            cells = {}
            for m in ("mrr", "r1", "r5", "r10", "ndcg@5"):
                a, b = mine.get(m), theirs.get(m)
                if a is None:
                    cells[m] = {"recomputed": None, "reported": b,
                                "note": "multi-positive: not recoverable "
                                        "from best-rank alone"}
                    print(f"{d if m=='mrr' else '':<20s}{m:>9s}"
                          f"{'-':>9s}{(b if b is not None else 0):9.2f}"
                          f"{'n/a':>8s}")
                    continue
                delta = abs(a - b) if b is not None else None
                cells[m] = {"recomputed": a, "reported": b, "delta": delta}
                if delta is not None:
                    worst = max(worst, delta)
                    if delta > args.tol:
                        bad.append((label, d, m, a, b, delta))
                print(f"{d if m=='mrr' else '':<20s}{m:>9s}{a:9.2f}"
                      f"{(b if b is not None else 0):9.2f}"
                      + (f"{delta:8.2f}" if delta is not None else f"{'-':>8s}")
                      + ("" if delta is None or delta <= args.tol else "  ❌"))
            job[d] = cells
        report["jobs"][label] = {"n_rows": len(rows), "one_positive": one_pos,
                                 "directions": job}

    report["max_abs_delta"] = round(worst, 4)
    report["disagreements"] = [{"job": a, "direction": b, "metric": c,
                                "recomputed": d, "reported": e,
                                "delta": round(f, 4)}
                               for a, b, c, d, e, f in bad]
    report["verdict"] = ("every recoverable metric agrees within rounding"
                         if not bad else f"{len(bad)} cells disagree")
    print(f"\n最大差 {worst:.4f}   不一致 {len(bad)} 格")
    print(report["verdict"])
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
