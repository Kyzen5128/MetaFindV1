"""Fix the object-level 80/20 partition and define both evaluation protocols.

# IMPLEMENTS-NODE: n09_build_splits

Writes ``splits``, ``split_seed``, ``eval_protocols``, ``stage1_protocol`` and
``run_progress``.

[PAPER 3.1] "We allocate 80% of the data for training and reserve 20% for
testing." That is the whole statement, and it covers both datasets; houses are
n09c's, objects are this node's.

Why two evaluation protocols and not one
-----------------------------------------

[U-09] The paper says retrieval runs against a "pre-encoded asset database" and
separately that an 80/20 split exists. It never says which of the two the
gallery is, and the difference is large -- a gallery of 9,211 versus 46,052
changes R@1 substantially for the same model.

An earlier draft locked one integer and planned to back-solve it from the
baselines' 98-99% PC-Only figure. That is impossible: under PC-Only the query
embedding is the gallery entry's own embedding, so self-retrieval approaches
100% at either size and the number carries no information about the
denominator. Both protocols run, both are reported.

[U-09, widened] The QUERY set is unstated too. 3.1 gives the allocation and
never says Table 1's queries are the 20%. Both protocols below assume
query=test; that assumption is recorded here rather than deduced.

[U-28] ``layout_free_context`` records what happens to Eq. 6's lambda*e_layout
when Table 1 evaluates on Objaverse-LVIS, which has no scenes. Paper 3.2
acknowledges the "feature-attribution mismatch when evaluating on layout-free
datasets" without saying whether the term is omitted, zeroed or routed
elsewhere. We omit it, which affects 7 of Table 1's 14 MetaFind cells.

What this node does NOT decide
-------------------------------

The Stage 1 ENCODING decisions -- serialization, view aggregation, CLIP scope,
missing-modality representation -- belong to n05b and are already resolved
before n06 encodes anything. This node writes ``stage1_protocol``, which carries
the TRAINING-side fields: fusion, tower sharing, similarity, and the hash of the
hyperparameter artifact n05b produced.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from metafind import paths, runlog
from metafind.models.stage1_config import (
    SUPPORTED_SIMILARITY,
    canonical_hyperparameter_hash,
)

NODE = "n09_build_splits"

# [PAPER 3.1] Same sentence n09c applies to houses.
TRAIN_FRACTION = 0.8
DEFAULT_SEED = 20260816

SPLITS_PATH = paths.OUTPUTS / "splits.json"
EVAL_PROTOCOLS_PATH = paths.OUTPUTS / "eval_protocols.json"
STAGE1_PROTOCOL_PATH = paths.OUTPUTS / "stage1_protocol.json"

# [U-13] The paper lists five fusion strategies in 2.4 and never says which the
# full model uses. Table 3 ablates Mean and MLPs as separate rows, so the main
# line is none of those two. `masked_mlp` is ours and is recorded as such.
DEFAULT_FUSION = "masked_mlp"

# [U-16] 2.6 requires the gallery encoder frozen while the query fuser trains.
# If the towers were ONE module those cannot both hold, so `fully_shared` is a
# Table 1-only variant; G6 refuses it on a run that reaches Stage 2.
DEFAULT_TOWER_SHARING = "shared_backbone_separate_fusion"

# [U-23] 2.6 masks each modality independently at 30%, so all three are masked
# in 2.7% of samples. The paper does not say what an all-masked query means.
# Refusing it would change the stated masking procedure, so it is allowed and
# recorded.
DEFAULT_ALLOW_ALL_MASKED = True


def split_assets(uids: list[str], seed: int,
                 train_fraction: float = TRAIN_FRACTION) -> tuple[list[str], list[str]]:
    """[PAPER 3.1] 80/20 over a sorted, seeded shuffle.

    Sorted first for the same reason as n09c: the caller's order comes from a
    manifest or a glob, and a split whose membership depends on iteration order
    is not reproducible by anyone else.
    """
    ordered = sorted(uids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * train_fraction))
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def build_eval_protocols(train: list[str], test: list[str]) -> dict:
    """[U-09] Both gallery scopes, both reported.

    `gallery_size` is DERIVED from the split rather than written down. An
    earlier draft hardcoded 48,000 -- the paper's approximate figure -- into
    gallery arithmetic while the manifest holds 46,052 (U-01), which silently
    moved every denominator.
    """
    return {
        "A_test_gallery": {
            "query_split": "test",
            "gallery_split": "test",
            "gallery_size": len(test),
            "layout_free_context": "omitted",
        },
        "B_full_gallery": {
            "query_split": "test",
            "gallery_split": "full",
            "gallery_size": len(train) + len(test),
            "layout_free_context": "omitted",
        },
    }


def build_stage1_protocol(hyperparameters: dict, decided_by: str,
                          fusion: str = DEFAULT_FUSION,
                          tower_sharing: str = DEFAULT_TOWER_SHARING) -> dict:
    if "cosine" not in SUPPORTED_SIMILARITY:
        raise ValueError("cosine is no longer supported; U-24's reading changed")
    return {
        "status": "resolved",
        "fusion": fusion,
        "tower_sharing": tower_sharing,
        "allow_all_masked": DEFAULT_ALLOW_ALL_MASKED,
        # [U-24] sim(.,.) is never defined in the paper; cosine is our reading,
        # and the loss normalises both sides unconditionally, so any other value
        # here would produce cosine numbers under a different label.
        "similarity": "cosine",
        "hyperparameter_config_hash": hyperparameters["sha256"],
        "decided_by": decided_by,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def admitted_uids() -> list[str]:
    """Assets that survived every upstream node, not everything the manifest lists.

    An asset quarantined by n03, n04 or n05 has no point cloud, no views or no
    annotation, so putting it in the gallery would create an entry nothing can
    retrieve and a denominator that overstates the corpus.
    """
    def index_uids(name: str) -> set[str]:
        path = paths.LOGS / name
        return {json.loads(line)["uid"]
                for line in path.read_text().splitlines() if line.strip()}

    return sorted(index_uids("pointclouds_index.jsonl")
                  & index_uids("renders_index.jsonl")
                  & index_uids("annotations_index.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--decided-by", default=None)
    args = ap.parse_args()

    hp_path = paths.OUTPUTS / "stage1_hyperparameters.json"
    if not hp_path.exists():
        print(f"{hp_path} not found -- run n05b_resolve_stage1_encoding first",
              flush=True)
        return 2
    hyperparameters = json.loads(hp_path.read_text())
    recomputed = canonical_hyperparameter_hash(hyperparameters["values"])
    if recomputed != hyperparameters["sha256"]:
        print(f"stage1_hyperparameters sha256 does not match its own values: "
              f"{hyperparameters['sha256']} vs {recomputed}", flush=True)
        return 2

    decided_by = args.decided_by or getpass.getuser()

    with runlog.run_progress(NODE):
        uids = admitted_uids()
        if not uids:
            print("no asset survived all of n03, n04 and n05", flush=True)
            return 2
        train, test = split_assets(uids, args.seed)

        # [L2-LEAK-OBJECT] A leaking split must not reach disk.
        leaked = set(train) & set(test)
        if leaked:
            raise AssertionError(f"{len(leaked)} assets in both splits")

        _write(SPLITS_PATH, {
            "object": {"train": train, "test": test},
            "split_seed": args.seed,
            "train_fraction": TRAIN_FRACTION,
            "admitted_total": len(uids),
        })
        _write(EVAL_PROTOCOLS_PATH, build_eval_protocols(train, test))
        _write(STAGE1_PROTOCOL_PATH,
               build_stage1_protocol(hyperparameters, decided_by))

    print(f"{len(train):,} train / {len(test):,} test objects "
          f"(seed {args.seed}, {len(uids):,} admitted)")
    for name, p in build_eval_protocols(train, test).items():
        print(f"  {name}: query={p['query_split']}, gallery={p['gallery_split']} "
              f"({p['gallery_size']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
