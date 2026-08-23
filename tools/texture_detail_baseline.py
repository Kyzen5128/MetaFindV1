#!/usr/bin/env python
"""The comparison population for V1.0's null result.

# SUPPORTS-NODE: n03_sample_pointclouds

`V1.0` re-measures `D-12` on the 37 texture assets ULIP also publishes. Those 37
were selected by OVERLAP, not by texture detail -- and the whole reason the
re-measure exists is that the old per-face rule was a low-pass filter, so the
two rules diverge most where the texture HAS detail.

A null result on a low-detail sample is therefore ambiguous between "the rules
agree" and "this sample cannot tell them apart", and nothing in V1.0's own
output distinguishes those. This supplies the missing half: the same statistic
over a random sample of the whole `texture` class.

    python tools/texture_detail_baseline.py --n 300

Raised by ULIP2 Reviewer, 2026-08-23.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", default="v1_0_baseline")
    ap.add_argument("--out", type=Path,
                    default=paths.OUTPUTS / "texture_detail_baseline.json")
    args = ap.parse_args()

    from metafind.data import pointclouds as pc

    # Ordered by a hash of the uid, not by directory order and not by
    # random.shuffle: the same sample comes back on a re-run, on any machine,
    # without carrying a file of uids around.
    glbs = sorted(paths.OBJAVERSE_GLB.rglob("*.glb"),
                  key=lambda p: hashlib.sha256(
                      f"{args.seed}:{p.stem}".encode()).hexdigest())

    detail, scanned = [], 0
    for glb in glbs:
        if len(detail) >= args.n:
            break
        scanned += 1
        try:
            _parts, worst, _s, _m = pc.load_parts(glb)
            if worst != "texture":
                continue
            _xyz, rgb, *_ = pc.sample_mesh(glb, pc.uid_seed(glb.stem))
        except Exception:  # noqa: BLE001 -- an unopenable GLB is not evidence
            continue
        detail.append(float(np.var(rgb)))
        if len(detail) % 25 == 0:
            print(f"  {len(detail)}/{args.n} (scanned {scanned})", flush=True)

    if not detail:
        print("no texture assets sampled")
        return 2
    d = sorted(detail)
    rec = {
        "n": len(d), "scanned": scanned,
        "mean": statistics.mean(d), "median": statistics.median(d),
        "p10": d[len(d) // 10], "p90": d[(9 * len(d)) // 10],
        "min": d[0], "max": d[-1],
        "statistic": "variance of sampled RGB in [0,1], per asset",
        "sampler_version": pc.SAMPLER_VERSION,
    }
    print(json.dumps(rec, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
