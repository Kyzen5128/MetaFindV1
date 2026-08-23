#!/usr/bin/env python
"""Hand-off report: which annotations the USER has to look at himself.

# SUPPORTS-NODE: n05_annotate

`category_relation` records how the model's `category` stands to the LVIS
anchor, and deliberately does not reject on it (see `annotate.category_relation`
-- separating a genuine downward refinement from a lateral replacement needs a
hypernym judgement no authoritative source here supplies). So the run RECORDS
the relation, and this script turns the recorded relation into something a
person can actually review: every `divergent` and `refined` row, with the twelve
view paths, so the reviewer opens the images rather than trusting the label.

    python tools/report_annotation_divergence.py

Writes `annotation_review.json` next to the corpus and prints the rates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402

paths.setup_env()

# Written in full so the reviewer can open an asset without joining files.
CARRY = ("uid", "lvis_category", "category", "synset", "synset_source",
         "category_relation", "identity_confirmed", "description")


def views_for(uid: str) -> list[str]:
    """The rendered views, read from n04's sidecar rather than guessed.

    Rebuilding the paths from a naming convention would silently produce
    plausible strings for an asset whose render failed; the sidecar exists only
    when all twelve landed.
    """
    sidecar = paths.OUTPUTS / "renders" / f"{uid}.json"
    if not sidecar.is_file():
        return []
    try:
        return json.loads(sidecar.read_text()).get("view_paths", [])
    except (OSError, json.JSONDecodeError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=paths.ANNOTATIONS,
                    help="annotations directory (a bake-off arm, to check this script)")
    ap.add_argument("--out", type=Path,
                    default=paths.OUTPUTS / "annotation_review.json")
    args = ap.parse_args()
    root = args.root
    files = sorted(root.glob("*.json"))
    if not files:
        print(f"no annotations under {root}")
        return 1

    buckets: dict[str, list[dict]] = {}
    counts: Counter[str] = Counter()
    unreadable: list[str] = []

    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            unreadable.append(f.name)
            continue
        rel = rec.get("category_relation", "unknown")
        counts[rel] += 1
        if rel in {"divergent", "refined"}:
            row = {k: rec.get(k) for k in CARRY}
            row["views"] = views_for(rec.get("uid", f.stem))
            buckets.setdefault(rel, []).append(row)

    total = sum(counts.values())
    out = args.out
    out.write_text(json.dumps({
        "total": total,
        "counts": dict(counts),
        "unreadable": unreadable,
        "divergent": buckets.get("divergent", []),
        "refined": buckets.get("refined", []),
    }, ensure_ascii=False, indent=1))

    print("=" * 60)
    print(f"annotations read: {total}")
    for rel, n in counts.most_common():
        print(f"  {rel:<12} {n:>7}  {n / total:6.2%}")
    if unreadable:
        print(f"  UNREADABLE   {len(unreadable):>7}  {unreadable[:5]}")
    print("=" * 60)
    print(f"written {out}")
    print("  `divergent` and `refined` carry their 12 view paths; open the "
          "images, the label alone is not the evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
