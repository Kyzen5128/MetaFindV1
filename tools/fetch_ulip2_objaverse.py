#!/usr/bin/env python3
"""Fetch ULIP-2's own Objaverse-LVIS point clouds -- the ones its 50.6 was run on.

WHY
---
Our reproduction of upstream's zero-shot number is 50.5756 against a published
50.6, achieved by feeding OUR clouds to upstream's unmodified code. That 0.024
is the only thing standing between "the clouds are right" and "the clouds are
close enough that nothing showed". It cannot be narrowed further without the
clouds upstream actually used, and `SFXX/ulip` publishes them:
`ULIP-2/objaverse_lvis/000-NNN.tar.gz`, 160 shards, 185 GB.

With them, five things move from inferred to compared, per asset:

    point count, colour range and channel order, `pc_norm`'s centring and
    scaling, the up axis, and whether the sampler is FPS or uniform.

Every one of those is currently something we read out of upstream's code and
reimplemented. None has been checked against upstream's own output.

WHERE
-----
`/mnt/data1` by default. That volume is an SMR drive: sustained small-file
writes collapse to single-digit MB/s once its CMR cache fills. 160 large
tar.gz files are exactly the read-mostly bulk workload it suits, so the
archives live there and stay there -- do NOT extract 185 GB of small `.npy`
onto it. Extract per-shard into the scratch space when a shard is being read.

Resumable: `hf_hub_download` skips a file already present with the right size,
so an interrupted run continues rather than restarting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = "SFXX/ulip"
PREFIX = "ULIP-2/objaverse_lvis"
DEFAULT_DEST = Path("/mnt/data1/kyzen/ulip2_objaverse_lvis")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only the first N shards, to check the format "
                         "before committing 185 GB")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(PREFIX) and f.endswith(".tar.gz"))
    if not files:
        sys.exit(f"no .tar.gz under {PREFIX} in {REPO}")
    if args.limit:
        files = files[:args.limit]

    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    size = {s.rfilename: (s.size or 0) for s in info.siblings}
    total = sum(size.get(f, 0) for f in files)
    print(f"{len(files)} shards, {total / 1e9:.1f} GB -> {args.dest}", flush=True)
    if args.list_only:
        for f in files:
            print(f"  {f}  {size.get(f, 0) / 1e9:.2f} GB")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    done = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f,
                        local_dir=str(args.dest))
        done += size.get(f, 0)
        el = time.time() - t0
        rate = done / el / 1e6 if el else 0
        eta = (total - done) / (done / el) / 60 if done and el else 0
        print(f"  [{i}/{len(files)}] {f.split('/')[-1]}  "
              f"{done / 1e9:.1f}/{total / 1e9:.1f} GB  "
              f"{rate:.0f} MB/s  ETA {eta:.0f} min", flush=True)

    print(f"done: {done / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
