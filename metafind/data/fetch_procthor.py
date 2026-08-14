"""Fetch ProcTHOR-10K house layouts and persist them as JSONL.

Graph node: part of ``n02_acquire_sources`` / SG2.

MetaFind sec. 2.3 uses ProcTHOR for the scene-level data: "over 10,000 generated
houses constructed from a curated collection of more than 3,000 unique assets".
Only the layout JSON is needed -- objects, their positions and their semantic
metadata -- not the AI2-THOR simulator, since ESSGNN consumes the graph, not
renders.

Two revisions exist
-------------------

``prior`` warns on load that ProcTHOR-10K was updated for AI2-THOR 5.0+, with the
previous release pinned at ``ab3cacd0fc17754d4c080a3fd50b18395fae8647``. The
paper does not say which it used (U-13). The unique-asset count is a usable
discriminator, so this module reports it against the paper's "3,000+" claim
rather than assuming.

Splits
------

ProcTHOR ships 10k train / 1k val / 1k test, but sec. 3.1 says "We allocate 80%
of the data for training and reserve 20% for testing". Those disagree (U-14).
The native splits are preserved verbatim here; deciding how to slice them is
``n06_build_splits``'s job, not the fetcher's.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_DIR = DATA / "sources/procthor"
STATS_PATH = DATA / "runs/progress/procthor_stats.json"

PAPER_HOUSE_CLAIM = 10_000
PAPER_ASSET_CLAIM = 3_000
OLD_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"


def object_records(house: dict) -> list[dict]:
    """Flatten a house's object tree, including children.

    Objects nest: a cup on a table is a child of the table. A flat walk is what
    the scene graph needs, and dropping children would silently discard exactly
    the support relations sec. 2.3 calls out ("cup on table").
    """
    out: list[dict] = []
    stack = list(house.get("objects", []))
    while stack:
        obj = stack.pop()
        out.append(obj)
        stack.extend(obj.get("children", []) or [])
    return out


def house_stats(house: dict) -> dict:
    objs = object_records(house)
    return {
        "n_objects": len(objs),
        "n_rooms": len(house.get("rooms", []) or []),
        "asset_ids": {o.get("assetId") for o in objs if o.get("assetId")},
        "types": {o.get("id", "").split("|")[0] for o in objs if o.get("id")},
    }


def fetch(revision: str | None = None, limit: int | None = None) -> dict:
    import prior

    kwargs = {"revision": revision} if revision else {}
    dataset = prior.load_dataset("procthor-10k", **kwargs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_assets: set[str] = set()
    all_types: collections.Counter[str] = collections.Counter()
    per_split: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        houses = dataset[split]
        n = len(houses) if limit is None else min(limit, len(houses))
        path = OUT_DIR / f"{split}.jsonl"

        obj_counts: list[int] = []
        room_counts: list[int] = []
        with open(path, "w") as fh:
            for i in range(n):
                house = houses[i]
                fh.write(json.dumps(house) + "\n")
                st = house_stats(house)
                obj_counts.append(st["n_objects"])
                room_counts.append(st["n_rooms"])
                all_assets |= st["asset_ids"]
                all_types.update(st["types"])

        per_split[split] = {
            "houses": n,
            "path": str(path),
            "bytes": path.stat().st_size,
            "objects_total": sum(obj_counts),
            "objects_mean": sum(obj_counts) / max(len(obj_counts), 1),
            "objects_max": max(obj_counts, default=0),
            "rooms_mean": sum(room_counts) / max(len(room_counts), 1),
        }
        print(
            f"  {split:5s}: {n:5d} houses, "
            f"{per_split[split]['objects_mean']:.1f} objects/house "
            f"(max {per_split[split]['objects_max']}), "
            f"{path.stat().st_size / 1e6:.0f} MB",
            flush=True,
        )

    stats = {
        "revision": revision or "latest",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": per_split,
        "unique_assets": len(all_assets),
        "unique_object_types": len(all_types),
        "top_object_types": all_types.most_common(15),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=None, help=f"e.g. {OLD_REVISION} for the pre-5.0 release")
    ap.add_argument("--limit", type=int, default=None, help="houses per split, for a smoke run")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(DATA / "cache/hf"))

    print(f"ProcTHOR-10K  revision={args.revision or 'latest'}")
    stats = fetch(revision=args.revision, limit=args.limit)

    total_houses = sum(s["houses"] for s in stats["splits"].values())
    print()
    print(f"  houses 合計       : {total_houses}")
    print(f"  unique assetId    : {stats['unique_assets']}")
    print(f"  unique 物件類型   : {stats['unique_object_types']}")

    # Compare against the paper's claims rather than assuming they hold. A
    # mismatch is information about which revision was used (U-13), not a bug.
    print()
    train = stats["splits"]["train"]["houses"]
    if train < PAPER_HOUSE_CLAIM:
        print(f"  注意: train 只有 {train} 間，論文說 'over 10,000'")
    if stats["unique_assets"] < PAPER_ASSET_CLAIM:
        print(
            f"  注意: unique assets {stats['unique_assets']} < 論文的 '{PAPER_ASSET_CLAIM}+'"
            f" -> 可能是另一個 revision，見 U-13"
        )
    else:
        print(f"  unique assets {stats['unique_assets']} 符合論文的 '{PAPER_ASSET_CLAIM}+'")

    print(f"\n  stats -> {STATS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
