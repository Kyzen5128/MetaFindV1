#!/usr/bin/env python
"""Freeze the 100 assets every bake-off arm annotates.

# SUPPORTS-NODE: n05_annotate_assets

`SPEC_M1` §4 requires `workflow/blocks/ULIP2/bakeoff/sample_100.jsonl` plus a
`.sha256`, shared by all three arms. **It does not say how to stratify**, so the
design below is an `IMPLEMENTATION CHOICE`, recorded here rather than left
implicit, and cheap to redo -- nothing has been annotated against it yet.

    python tools/build_bakeoff_sample.py --n 100
    python tools/build_bakeoff_sample.py --verify        # re-check the hash only

Why these strata
----------------

**`colour_source`, proportionally.** `FIND-4` compared our clouds against
ULIP's and reported the distributions "agree closely", but it **never
stratified by `colour_source`** -- and 8,853 assets are `gltf_default`, i.e.
plain white. An annotator asked to describe a white untextured mesh is doing a
different job from one describing a textured one, and a sample that happened to
under-draw the white class would hide it. This is the one blind spot the block
has already been caught by, so it is the one stratum that is not a guess.

**Distinct LVIS categories within each stratum.** The corpus is dominated by a
few categories; 100 assets drawn uniformly would repeat them. The bake-off
scores identification, so repeating a category buys a second measurement of the
same question instead of covering a new one. Categories are taken round-robin,
so the sample spans as many as the size allows.

**Not stratified by shape.** Flat and tall assets stress the dimension
estimate, which is `S-9`'s subject and has its own 103-asset control group.
Mixing that population in here would make one sample answer two questions and
neither cleanly.

Determinism
-----------

Selection is ordered by `sha256(seed:uid)` -- not `rng.choice(len(pool))`,
which moves when the pool changes size. The same seed against a corpus that has
gained or lost assets still picks the same ones, wherever they still exist.
That property was learned the hard way: the render-criteria tool drew indices
into a growing directory and returned two different answers from one seed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "workflow" / "blocks" / "ULIP2" / "bakeoff"
SAMPLE = OUT_DIR / "sample_100.jsonl"
SAMPLE_SHA = OUT_DIR / "sample_100.jsonl.sha256"
SEED = "20260822"


def _order_key(uid: str) -> str:
    return hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest()


def load_pool() -> list[dict]:
    """Assets that have BOTH a v4 render and a point-cloud sidecar, with metadata.

    Both, because an arm that cannot load an asset produces a quarantine record
    rather than an annotation, and `S-10` ("every arm produces 100 valid
    records") would then fail for a reason that has nothing to do with the
    annotator.
    """
    from metafind.data.annotate_run import load_lvis_categories

    lvis = load_lvis_categories()
    renders = {}
    for line in (paths.LOGS / "renders_index.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            renders[rec["uid"]] = rec

    pool = []
    for sc in paths.POINTCLOUDS.glob("*.json"):
        uid = sc.stem
        if uid not in renders:
            continue
        pc = json.loads(sc.read_text())
        pool.append({
            "uid": uid,
            "lvis_category": lvis.get(uid),
            "colour_source": pc["colour_source"],
            "color0_modulated": pc["color0_modulated"],
            "raw_bbox_extents": pc["raw_bbox_extents"],
            "n_views": len(renders[uid]["view_paths"]),
            "renderer_version": renders[uid]["renderer_version"],
            "sampler_version": pc["sampler_version"],
        })
    return pool


def stratify(pool: list[dict], n: int) -> list[dict]:
    """Proportional over `colour_source`, category round-robin inside each."""
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for a in pool:
        if a["lvis_category"]:  # an unlabelled asset cannot score identification
            by_source[a["colour_source"]].append(a)

    total = sum(len(v) for v in by_source.values())
    # Largest-remainder, so the quotas sum to exactly n instead of n-1 or n+1.
    exact = {k: len(v) * n / total for k, v in by_source.items()}
    quota = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - quota[k]))[:n - sum(quota.values())]:
        quota[k] += 1

    picked: list[dict] = []
    for source, assets in sorted(by_source.items()):
        by_cat: dict[str, list[dict]] = collections.defaultdict(list)
        for a in sorted(assets, key=lambda a: _order_key(a["uid"])):
            by_cat[a["lvis_category"]].append(a)
        # Round-robin across categories: one from each before a second from any.
        rounds, taken = 0, []
        cats = sorted(by_cat, key=lambda c: _order_key(c))
        while len(taken) < quota[source]:
            progressed = False
            for c in cats:
                if rounds < len(by_cat[c]):
                    taken.append(by_cat[c][rounds])
                    progressed = True
                    if len(taken) == quota[source]:
                        break
            if not progressed:
                break
            rounds += 1
        picked.extend(taken)
    return sorted(picked, key=lambda a: _order_key(a["uid"]))


def write(sample: list[dict]) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(a, sort_keys=True) + "\n" for a in sample)
    SAMPLE.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    SAMPLE_SHA.write_text(f"{digest}  {SAMPLE.name}\n")
    return digest


def verify() -> int:
    if not SAMPLE.exists() or not SAMPLE_SHA.exists():
        print(f"{SAMPLE} or its .sha256 is missing")
        return 2
    want = SAMPLE_SHA.read_text().split()[0]
    got = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    n = sum(1 for line in SAMPLE.read_text().splitlines() if line.strip())
    print(f"{n} assets  recorded {want[:16]}  actual {got[:16]}  "
          f"{'MATCH' if want == got else 'MISMATCH -- the sample changed under the arms'}")
    return 0 if want == got else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sample (refused by default: arms "
                         "already scored against it would become incomparable)")
    args = ap.parse_args()

    if args.verify:
        return verify()
    if SAMPLE.exists() and not args.force:
        print(f"{SAMPLE} already exists. Re-drawing it would silently change what "
              f"every arm was measured on -- pass --force if that is intended.")
        return verify()

    pool = load_pool()
    print(f"{len(pool):,} assets have both a render and a cloud", flush=True)
    sample = stratify(pool, args.n)
    digest = write(sample)

    src = collections.Counter(a["colour_source"] for a in sample)
    pop = collections.Counter(a["colour_source"] for a in pool)
    print(f"\n{len(sample)} assets, sha256 {digest[:16]}")
    print(f"  distinct LVIS categories : {len({a['lvis_category'] for a in sample})}")
    print(f"  colour_source            : "
          + " · ".join(f"{k} {v} ({v / len(sample):.0%} vs corpus "
                       f"{pop[k] / len(pool):.0%})" for k, v in sorted(src.items())))
    print(f"  color0_modulated         : {sum(a['color0_modulated'] for a in sample)}")
    print(f"  renderer_version         : "
          f"{dict(collections.Counter(a['renderer_version'] for a in sample))}")
    print(f"\nwritten {SAMPLE}\n        {SAMPLE_SHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
