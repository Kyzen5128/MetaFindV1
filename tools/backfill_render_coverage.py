"""Backfill the alpha-coverage fields onto sidecars written before 2026-08-24.

# SUPPORTS-NODE: n04_render_views

WHY THIS EXISTS. `blank_views` changed meaning on 2026-08-24: it counted
`std()` of the black-composited image ("looks flat") and now counts alpha
coverage ("nothing was drawn"). Both definitions live under the same
`renderer_version: 6`, so without this the corpus carries a MIXTURE.

That is not cosmetic. `docs/graph/validation_plan.yaml` L1-RENDER-PARTIAL-BLANK
-- rank 5 in the project authority order, above repository implementation --
reports the corpus-wide `blank_views` distribution, and its own note says that
distribution is the ONLY signal that moves on the regression it exists to catch:

    "a regression that starts blanking views on ORDINARY assets would move this
     distribution and nothing else, since every other render check passes on a
     blank frame of the right shape."

The new definition is systematically LOOSER -- a black-but-present view counted
blank under the old rule is not blank under the new one -- so a mixed corpus
puts a directional bias in that detector's baseline. Found by the ULIP2 Block
Reviewer, who also supplied the remedy: the new fields are a function of alpha
in PNGs that ALREADY EXIST, so this is one pass over the images with no Blender,
no GPU, and no re-render.

Idempotent: an asset whose sidecar already carries `view_coverage` is skipped.
Atomic per sidecar: temp file then replace, so an interrupted run leaves whole
records rather than half-written ones.

    python tools/backfill_render_coverage.py [--limit N] [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metafind import paths  # noqa: E402
from metafind.data.renders import MIN_COVERAGE  # noqa: E402


def measure(sidecar: Path) -> tuple[str, dict | None, str]:
    """Return (uid, fields-to-merge, status). Never raises."""
    uid = sidecar.stem
    try:
        rec = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return uid, None, f"unreadable: {exc}"
    if "view_coverage" in rec:
        return uid, None, "already"
    views = rec.get("view_paths") or []
    if not views:
        return uid, None, "no view_paths"

    coverage, shas, dark = [], [], 0
    for v in views:
        try:
            rgba = np.asarray(Image.open(v).convert("RGBA"))
        except OSError as exc:
            return uid, None, f"missing png: {exc}"
        cov = float((rgba[..., 3] > 0).mean())
        coverage.append(round(cov, 8))
        if cov > 0:
            a = rgba[..., 3:4].astype(np.float32) / 255.0
            if float((rgba[..., :3].astype(np.float32) * a).std()) < 1.0:
                dark += 1
        shas.append(hashlib.sha256(Path(v).read_bytes()).hexdigest())

    return uid, {
        "view_coverage": coverage,
        "blank_views": sum(1 for c in coverage if c <= MIN_COVERAGE),
        "distinct_views": len(set(shas)),
        "dark_views": dark,
        # Says WHICH definition this record's `blank_views` is under, so the
        # mixture can never be silent again even if a backfill is interrupted.
        "coverage_backfilled": True,
    }, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sidecars = sorted(paths.RENDERS.glob("*.json"))
    if args.limit:
        sidecars = sidecars[: args.limit]
    print(f"{len(sidecars):,} sidecars", flush=True)

    counts, changed, started = {}, 0, time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (uid, fields, status) in enumerate(pool.map(measure, sidecars), 1):
            counts[status] = counts.get(status, 0) + 1
            if fields and not args.dry_run:
                sc = paths.RENDERS / f"{uid}.json"
                rec = json.loads(sc.read_text())
                rec.update(fields)
                tmp = sc.with_suffix(".json.part")
                tmp.write_text(json.dumps(rec))
                tmp.replace(sc)
                changed += 1
            elif fields:
                changed += 1
            if i % 2000 == 0:
                rate = i / max(time.time() - started, 1e-9)
                print(f"  [{i:6,}/{len(sidecars):,}] {rate:.0f}/s  "
                      f"remaining {(len(sidecars)-i)/max(rate,1e-9)/60:.1f} min", flush=True)

    print(f"\n{'DRY RUN, nothing written' if args.dry_run else 'written'}: {changed:,}")
    for k, v in sorted(counts.items()):
        print(f"  {k:24} {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
