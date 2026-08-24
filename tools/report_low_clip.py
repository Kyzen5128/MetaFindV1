"""List annotations whose WINNING description scored low against its own views.

[USER 2026-08-24] 「若產出的結果Clip分數記得紀錄 到時候把太低的抓出來」 -- the
scores were already recorded (every candidate carries `clip_score`; the winner
is rank 0), so this is the read-only extractor. It changes nothing on disk.

The threshold is the USER's to pick, not this script's: with no --threshold it
prints the distribution and the bottom N so the cut can be chosen from data
rather than assumed. A uid list written with --out feeds annotate_run
--uids-file --force for the re-run.

Usage:
  python tools/report_low_clip.py                     # distribution + bottom 50
  python tools/report_low_clip.py --threshold 0.22    # everything below 0.22
  python tools/report_low_clip.py --threshold 0.22 --out low_uids.txt
  python tools/report_low_clip.py --dir data/outputs/bakeoff/smoke_v8_5/annotations
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metafind import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=paths.ANNOTATIONS,
                    help="annotation directory (default: the corpus)")
    ap.add_argument("--threshold", type=float,
                    help="list every uid whose winner scored below this")
    ap.add_argument("--bottom", type=int, default=50,
                    help="with no --threshold: how many lowest to list")
    ap.add_argument("--out", type=Path,
                    help="also write the uids, one per line (annotate_run --uids-file)")
    args = ap.parse_args()

    rows = []
    unscored = 0
    for f in sorted(args.dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cands = rec.get("description_candidates") or []
        winner = next((c for c in cands if c.get("rank") == 0), None)
        if winner is None or "clip_score" not in winner:
            unscored += 1
            continue
        rows.append((float(winner["clip_score"]), rec["uid"]))
    if not rows:
        print(f"no scored annotations under {args.dir}")
        return 1
    rows.sort()
    scores = [s for s, _ in rows]
    q = statistics.quantiles(scores, n=20) if len(scores) >= 20 else []
    print(f"{len(rows):,} scored annotation(s)  ·  {unscored} without a score")
    print(f"min {scores[0]:.4f}   median {statistics.median(scores):.4f}   "
          f"max {scores[-1]:.4f}")
    if q:
        print(f"p5 {q[0]:.4f}   p10 {q[1]:.4f}   p25 {q[4]:.4f}")

    if args.threshold is not None:
        low = [(s, u) for s, u in rows if s < args.threshold]
        print(f"\n{len(low):,} below {args.threshold}:")
    else:
        low = rows[: args.bottom]
        print(f"\nbottom {len(low)}:")
    for s, u in low:
        print(f"  {s:.4f}  {u}")
    if args.out:
        args.out.write_text("".join(u + "\n" for _, u in low))
        print(f"\nwrote {len(low)} uid(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
