#!/usr/bin/env python3
"""PROTOCOL E, PC arm -- a SECOND independent 10,000-point draw per dev_val asset.

DIAGNOSTIC DATA ONLY. Authorised by Kyzen via MASTER, 2026-08-31, for dev_val's
4,569 assets and for nothing else. It is not paper data and never becomes any.

WHAT THIS PRODUCES
------------------
For each dev_val uid, a second full surface sample of the SAME mesh at the SAME
point count, differing from the canonical cloud ONLY in the sampler seed:

    canonical (gallery)  seed = uid_seed(uid)                <- already on disk
    query                seed = uid_seed(uid) + 1_000_003    <- written here

Same density, same asset identity, independent draw. The rejected alternative --
splitting the stored 10,000 points into halves -- would have changed point
DENSITY at the same time as the sample, adding a confound instead of removing
one.

WHY THE SAMPLER IS NOT REIMPLEMENTED
-------------------------------------
`sample_mesh` and `pc_norm` from `metafind.data.pointclouds` are called
directly, so the two clouds differ in the seed and in nothing else. Neither
function writes anything -- only `process_one` does, and `process_one` is NOT
called, because it writes to the canonical `pointclouds/` path. The pipeline
`process_one` performs is reproduced here in the same order it performs it
(`pointclouds.py:678-684`):

    xyz, rgb = sample_mesh(glb, seed)          area-weighted, per-part seeded
    normed   = pc_norm(xyz.astype(float64)).astype(float32)
    cloud    = concatenate([normed, rgb], axis=1)

the last line being what `Stage1Dataset.__getitem__` builds from the stored npz
(`stage1.py`), so the query cloud reaches `encode_pc` in the same layout the
canonical one does.

WHERE IT WRITES, AND WHERE IT MUST NOT
---------------------------------------
One file under `data/outputs/_probe/protocol_e_query_pc/`, plus its manifest.
`paths.POINTCLOUDS` is asserted to be outside the output directory before
anything is written. Nothing under `pointclouds/` is read-modified, moved, or
regenerated; the canonical clouds are opened read-only, for hashing only.

ONE FILE, NOT 4,569. A single `(4569, 10000, 6)` float32 array (1.1 GB) rather
than per-asset npz files: it gives the provenance block ONE file hash to carry,
it re-runs the evaluation without re-sampling any mesh, and it avoids the
small-file write pattern this project has already measured as pathological on
the secondary volume. Written to NVMe (`/`, 299 GB free at time of writing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths                                           # noqa: E402
from metafind.data.pointclouds import N_POINTS, pc_norm, sample_mesh, uid_seed  # noqa: E402

# Fixed, and large enough that `uid_seed(uid) + OFFSET` cannot collide with the
# canonical run's per-part seeds, which are `uid_seed(uid) + part_index`.
SEED_OFFSET = 1_000_003
OUTDIR = paths.OUTPUTS / "_probe" / "protocol_e_query_pc"
ARRAY = OUTDIR / f"query_pc_offset{SEED_OFFSET}.npy"
MANIFEST = OUTDIR / f"query_pc_offset{SEED_OFFSET}.manifest.json"

_GLB: dict[str, Path] = {}


def _one(uid: str):
    """One asset's second draw. Returns ``(uid, (10000, 6) float32)``."""
    xyz, rgb, *_ = sample_mesh(_GLB[uid], uid_seed(uid) + SEED_OFFSET)
    normed = pc_norm(xyz.astype(np.float64)).astype(np.float32)
    if not np.isfinite(normed).all() or not np.isfinite(rgb).all():
        raise ValueError(f"{uid}: non-finite value after normalisation")
    return uid, np.concatenate([normed, rgb.astype(np.float32)], axis=1)


def _init(glb: dict[str, str]) -> None:
    _GLB.update({k: Path(v) for k, v in glb.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: first N dev_val assets.")
    args = ap.parse_args()

    # HARD GUARD. The authorisation's first condition is that nothing under
    # `pointclouds/` is overwritten. This asserts the destination cannot be it,
    # rather than trusting the constant above to stay correct.
    assert paths.POINTCLOUDS.resolve() not in OUTDIR.resolve().parents, \
        "refusing to write inside the canonical point-cloud directory"
    assert "pointclouds" not in OUTDIR.resolve().name, "destination looks canonical"

    dev_val = json.loads(
        (paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    if args.limit:
        dev_val = dev_val[:args.limit]
        print("⚠ --limit set: DEBUG RUN, not the Protocol E input.", flush=True)

    glb = {p.stem: str(p) for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    missing = [u for u in dev_val if u not in glb]
    if missing:
        raise SystemExit(f"{len(missing)} dev_val uid(s) have no GLB, "
                         f"e.g. {missing[:3]}")
    glb = {u: glb[u] for u in dev_val}

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        ARRAY, mode="w+", dtype=np.float32, shape=(len(dev_val), N_POINTS, 6))

    from multiprocessing import Pool

    index = {u: i for i, u in enumerate(dev_val)}
    per_uid: dict[str, str] = {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(glb,)) as pool:
        for k, (uid, cloud) in enumerate(
                pool.imap_unordered(_one, dev_val, chunksize=16), 1):
            out[index[uid]] = cloud
            per_uid[uid] = hashlib.sha256(cloud.tobytes()).hexdigest()
            if k % 500 == 0 or k == len(dev_val):
                el = time.time() - t0
                print(f"  {k:,}/{len(dev_val):,}  {el:.0f}s  "
                      f"eta {el/k*(len(dev_val)-k):.0f}s", flush=True)
    out.flush()
    del out

    # Canonical clouds: opened READ-ONLY, hashed so the provenance block can
    # show which pair of draws the numbers came from.
    canon = {u: hashlib.sha256((paths.POINTCLOUDS / f"{u}.npz").read_bytes()
                               ).hexdigest() for u in dev_val}
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())

    MANIFEST.write_text(json.dumps({
        "what": ("Protocol E query point clouds: a second independent "
                 "10,000-point surface sample per dev_val asset, same mesh, "
                 "same density, seed = uid_seed(uid) + SEED_OFFSET."),
        "status": "DIAGNOSTIC ONLY -- not paper data, not a canonical artifact",
        "seed_offset": SEED_OFFSET,
        "n_points": N_POINTS,
        "n_assets": len(dev_val),
        "array": str(ARRAY),
        "array_sha256": hashlib.sha256(ARRAY.read_bytes()).hexdigest(),
        "uid_order": dev_val,
        "query_pc_sha256_per_uid": per_uid,
        "canonical_pc_npz_sha256_per_uid": canon,
        "sampler": "metafind.data.pointclouds.sample_mesh + pc_norm (unmodified)",
        "code_revision": rev, "code_dirty": dirty,
        "debug_limit": args.limit,
    }, indent=1))
    print(f"\nwrote {ARRAY} ({ARRAY.stat().st_size/1e9:.2f} GB)\nwrote {MANIFEST}")
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
