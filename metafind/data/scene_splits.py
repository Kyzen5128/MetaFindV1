"""Split the ProcTHOR houses into train and test.

# IMPLEMENTS-NODE: n09c_build_scene_splits

Writes ``scene_splits`` (train_houses, test_houses) and ``run_progress``.

[PAPER 3.1] "We allocate 80% of the data for training and reserve 20% for
testing." One sentence, covering both datasets; this node applies it to houses
and n09_build_splits applies it to objects.

Why this node exists separately from n09
-----------------------------------------

An earlier draft built the house split inside n09, which put Qwen semantic-edge
generation over ProcTHOR on the critical path to Stage 1. Stage 1 is
object-level pretraining on Objaverse-LVIS (paper 2.6) and needs no ProcTHOR
data at all, so a fault anywhere in the ProcTHOR branch used to stop training
that does not depend on it. The branches are decoupled; this is the ProcTHOR
half, and it can fail without touching Stage 1.

[UNKNOWN U-07] ProcTHOR ships its own 10k/1k/1k split
------------------------------------------------------

We do not use it. The paper states 80/20 and names no validation set;
ProcTHOR's own allocation is 83.3/8.3/8.3 with a val split the paper never
mentions, so honouring it would mean reporting numbers under an allocation the
paper does not describe. The split is drawn over all 12,000 houses with a
recorded seed, and the ProcTHOR split each house came from travels with it so a
reader can reconstruct either view.

What is deliberately NOT enforced
----------------------------------

Asset-level disjointness. An earlier draft forbade an asset appearing in both a
train and a test house, but the paper does not ask for it and ProcTHOR's 12,000
houses draw on only ~1,467 assets -- enforcing it would either fail outright or
change the scene distribution into something ProcTHOR did not generate. House
disjointness is what L2-LEAK-SCENE asserts and what G6_stage2_ready gates on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from metafind import paths, runlog

NODE = "n09c_build_scene_splits"

# [PAPER 3.1] "We allocate 80% of the data for training and reserve 20% for
# testing."
TRAIN_FRACTION = 0.8
DEFAULT_SEED = 20260816

SPLITS_PATH = paths.OUTPUTS / "scene_splits.json"
SEM_EDGE_CACHE = paths.OUTPUTS / "sem_edge_cache.json"


def split_houses(house_ids: list[str], seed: int,
                 train_fraction: float = TRAIN_FRACTION) -> tuple[list[str], list[str]]:
    """[PAPER 3.1] 80/20 over a sorted, seeded shuffle.

    Sorted first because the caller's order comes from a filesystem glob, and a
    split whose membership depends on directory iteration order is not a split
    anyone can reproduce.
    """
    ordered = sorted(house_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * train_fraction))
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def semantic_edge_coverage(house_ids: list[str], cache: dict, text_map: dict) -> dict:
    """How many of a split's semantic edges actually carry an embedding.

    G6_stage2_ready needs to know this before it lets Stage 2 train: a split
    whose edges are mostly degraded would train ESSGNN on a graph whose semantic
    half is largely absent, and the resulting Table 2 number would be measuring
    something other than what it claims.
    """
    entries = cache["entries"]
    total, degraded, missing = 0, 0, 0
    for house_id in house_ids:
        try:
            graph = json.loads((paths.SCENE_GRAPHS / f"{house_id}.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nodes = graph["nodes"]
        for i, j in graph["sem_edge_ids"]:
            total += 1
            ti = text_map.get(str(nodes[i]["asset_id"]))
            tj = text_map.get(str(nodes[j]["asset_id"]))
            if ti is None or tj is None:
                missing += 1
                continue
            from metafind.data.semantic_edges import cache_key

            key = cache_key(ti["text"], tj["text"], cache["prompt_version"],
                            cache["llm_model"], cache["text_encoder_version"])
            entry = entries.get(key)
            if entry is None:
                missing += 1
            elif entry.get("degraded"):
                degraded += 1
    return {"edges": total, "degraded": degraded, "uncached": missing,
            "covered_fraction": round((total - degraded - missing) / total, 6)
            if total else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip-coverage", action="store_true",
                    help="split without sem_edge_cache; coverage is recorded as null")
    args = ap.parse_args()

    house_ids = sorted(p.stem for p in paths.SCENE_GRAPHS.glob("*.json"))
    if not house_ids:
        print(f"no scene graphs in {paths.SCENE_GRAPHS} -- run n07 first", flush=True)
        return 2

    with runlog.run_progress(NODE):
        train, test = split_houses(house_ids, args.seed)

        # [L2-LEAK-SCENE] asserted here as well as measured downstream, because
        # a leaking split must not reach disk in the first place.
        leaked = set(train) & set(test)
        if leaked:
            raise AssertionError(f"{len(leaked)} houses in both splits")

        coverage = None
        if not args.skip_coverage:
            if not SEM_EDGE_CACHE.exists():
                print(f"{SEM_EDGE_CACHE} not found -- run n08 first, or pass "
                      "--skip-coverage to split without it", flush=True)
                return 2
            cache = json.loads(SEM_EDGE_CACHE.read_text())
            text_map = json.loads((paths.OUTPUTS / "procthor_object_text.json").read_text())
            coverage = {"train": semantic_edge_coverage(train, cache, text_map),
                        "test": semantic_edge_coverage(test, cache, text_map)}

    record = {
        "train_houses": train,
        "test_houses": test,
        "split_seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        # [U-07] recorded so a reader can reconstruct ProcTHOR's own view
        "procthor_split_of": {h: h.split("_")[0] for h in house_ids},
        "semantic_edge_coverage": coverage,
    }
    tmp = SPLITS_PATH.with_suffix(".json.part")
    with tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(SPLITS_PATH)

    print(f"{len(train):,} train / {len(test):,} test houses "
          f"(seed {args.seed}) -> {SPLITS_PATH}")
    if coverage:
        for name, c in coverage.items():
            print(f"  {name}: {c['covered_fraction']:.4f} of {c['edges']:,} "
                  f"semantic edges carry an embedding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
