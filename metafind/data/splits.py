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
gallery is, and the difference is large -- a gallery of 9,138 versus 45,692
changes R@1 substantially for the same model.

Which corpus size is which
--------------------------

Three sizes appear across this codebase and none of them is a typo. Counted
2026-08-30, OBSERVED DATA:

    46,052  the Objaverse-LVIS uid manifest (``download.py``), which is n03's
            input. Every count in ``data/pointclouds.py`` is against this one:
            ``logs/pointclouds_index.jsonl`` has 46,052 lines.
    46,024  what n04 actually rendered; ``logs/renders_index.jsonl``. The
            incident figures in ``data/renders.py`` are against the 46,052 it
            was handed, because they describe runs, not results.
    45,692  THIS corpus. 46,024 - 311 quarantined by n05 - 21 rejected in manual
            review, per ``outputs/annotation_exclusions.json``. It is what
            ``admitted_uids()`` returns, what ``splits.json`` records as
            ``admitted_total``, and the only one of the three that ever reaches
            a gallery. Splits: test 9,138 / train 36,554, and inside the train
            pool dev_train 31,985 / dev_val 4,569.

The prose above used to read "9,211 versus 46,052" -- an 80/20 of the manifest,
taken before rendering and annotation removed 360 assets from it. 46,052 is not
wrong about the manifest; it was wrong about the gallery.

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

# [DEVIATION D-3, Kyzen 2026-08-27] Development-time model selection happens
# INSIDE the paper's 80% training pool. The 20% test split is opened once, at
# the end, and never participates in choosing lr, epochs or a checkpoint.
#
# An earlier version halved the held-out 20% into val/test while keeping the
# whole 20% as the gallery, on the grounds that the val queries' answers were
# never used. That does not hold: a val query is ranked against a candidate pool
# roughly half of which is future test assets, so changing the checkpoint moves
# those assets' vectors, moves the val score, and the checkpoint is chosen by
# that score. Transductive contamination. Withdrawn.
#
# [USER-APPROVED, Kyzen 2026-08-27] 0.125 OF THE TRAINING POOL, which is 10% of
# the corpus: 70% dev-train / 10% dev-val / 20% test.
#
# He was offered 80/10/10 -- halving the paper's 20% into val and test -- and
# rejected it, consistent with the D-3 he approved the same day: that shape is
# the transductive contamination this deviation exists to avoid, and it would
# leave the reported gallery at 10% instead of the paper's 20%.
#
# So the fraction is expressed against the TRAINING POOL, not the corpus, and
# the arithmetic is stated because the two are easy to confuse:
#
#     0.125 x 0.80 = 0.10 of the corpus
#     train 80% = dev_train 70% + dev_val 10%      test 20%, sealed
#
# The three quantities the paper fixes are unchanged: 80% trains, 20% tests, and
# the test split takes no part in choosing anything.
DEV_VAL_FRACTION = 0.125
# A separate seed, not a reuse of the object seed: the same seed on a subset
# would make dev-val membership a deterministic function of train membership in
# a way nobody would have chosen deliberately.
DEFAULT_DEV_SEED = 20260827

SPLITS_PATH = paths.OUTPUTS / "splits.json"
EVAL_PROTOCOLS_PATH = paths.OUTPUTS / "eval_protocols.json"
STAGE1_PROTOCOL_PATH = paths.OUTPUTS / "stage1_protocol.json"

# [PAPER FACT 3experiments.tex:143] "MLP and the final selected Transformer
# outperform others". The three lines that stood here said the paper never names
# the full model's fusion. It does, and DECISION_LEDGER.md:723 (U-T) already
# corrected that false UNKNOWN back to a PAPER FACT -- but the fix landed in
# fusion.py:89, whose default the trainer never reads: stage1.py:322 and
# stage1_config.py:367 both construct FusionConfig with
# kind=training_protocol["fusion"].
#
# THIS constant is the one that reaches stage1_protocol.json and therefore the
# model. Said plainly because the same bug has already been fixed in the wrong
# half once: a default is only a default for callers that omit the argument,
# and here every caller supplies it.
DEFAULT_FUSION = "transformer"

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


def split_dev(train: list[str], seed: int,
              val_fraction: float = DEV_VAL_FRACTION) -> tuple[list[str], list[str]]:
    """[D-3] Carve a dev-val out of the training pool, same discipline as above.

    Returns (dev_train, dev_val) whose union is exactly `train`, so the
    development phase never sees an asset the paper allocated to testing.
    """
    ordered = sorted(train)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * (1.0 - val_fraction)))
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def build_eval_protocols(train: list[str], test: list[str],
                         dev_val: list[str] | None = None) -> dict:
    """[U-09] Both gallery scopes, both reported.

    `gallery_size` is DERIVED from the split rather than written down. An
    earlier draft hardcoded 48,000 -- the paper's approximate figure -- into
    gallery arithmetic while the manifest holds 46,052 (U-01), which silently
    moved every denominator. BOTH of those are now wrong for a gallery: n05
    admitted 45,692 of the manifest's 46,052 (see `admitted_uids` and the
    module docstring), so B is 45,692 and A is 9,138. Those figures are quoted
    to let a reader recognise a stale constant, never to be read back into code.
    """
    protocols = {
        "A_test_gallery": {
            "query_split": "test",
            "gallery_split": "test",
            "gallery_size": len(test),
            "layout_free_context": "omitted",
            "reported": True,
        },
        "B_full_gallery": {
            "query_split": "test",
            "gallery_split": "full",
            "gallery_size": len(train) + len(test),
            "layout_free_context": "omitted",
            "reported": True,
        },
    }
    if dev_val is not None:
        # [D-3] The development-phase selection protocol. `reported: False` is
        # the whole point: its numbers choose lr, epochs and checkpoint policy
        # and must never appear as a result.
        #
        # gallery = dev_val, NOT the whole training pool. Ranking a dev-val
        # query against all 36,554 training assets is a different task from the
        # final one (query 20%, gallery 20%, 9,138 candidates), and a duration
        # tuned against a pool an order of magnitude larger does not transfer.
        # The dev-val gallery is 4,569. Those three are splits.json as built on
        # the 45,692 corpus; the 36,819 and ~9,200 that stood here were the same
        # 80/20 taken against the 46,024 RENDER corpus, one node too early.
        # This is inference from D-3's logic rather than its text, and it is
        # recorded here so it can be overruled rather than discovered.
        #
        # Consequence of that choice: these numbers rank checkpoints against
        # each other and are NOT comparable to A or B, whose candidate pools are
        # different sizes. `reported: False` encodes that; this says why, because
        # the next person to see a dev-val recall figure will want to put it
        # beside a reported one.
        protocols["C_dev_selection"] = {
            "query_split": "dev_val",
            "gallery_split": "dev_val",
            "gallery_size": len(dev_val),
            "layout_free_context": "omitted",
            "reported": False,
        }
    return protocols


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

    Counted 2026-08-30 (OBSERVED DATA): pointclouds_index 46,052, renders_index
    46,024, annotations_index 45,692. The three-way intersection is 45,692,
    which matches both the 45,692 files in ``outputs/annotations/`` and
    ``splits.json``'s ``admitted_total``.
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
    ap.add_argument("--dev-seed", type=int, default=DEFAULT_DEV_SEED)
    ap.add_argument("--dev-val-fraction", type=float, default=DEV_VAL_FRACTION,
                    help="fraction of the 80%% training pool held out as dev-val "
                         "(NOT ratified; see DEV_VAL_FRACTION)")
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

    with runlog.run_progress(NODE) as progress:
        uids = admitted_uids()
        if not uids:
            print("no asset survived all of n03, n04 and n05", flush=True)
            progress.rc = 2
            return 2
        train, test = split_assets(uids, args.seed)
        dev_train, dev_val = split_dev(train, args.dev_seed,
                                       args.dev_val_fraction)

        # [L2-LEAK-OBJECT] A leaking split must not reach disk. Three pairs now,
        # and the middle one is the one D-3 exists to prevent.
        for a, b, why in ((train, test, "train/test"),
                          (dev_val, test, "dev_val/test"),
                          (dev_train, dev_val, "dev_train/dev_val")):
            if leaked := set(a) & set(b):
                raise AssertionError(f"{len(leaked)} assets in both {why}")
        if set(dev_train) | set(dev_val) != set(train):
            raise AssertionError(
                "dev_train + dev_val is not the training pool; the development "
                "phase would be training on something the paper did not allocate "
                "to training")

        _write(SPLITS_PATH, {
            "object": {"train": train, "test": test,
                       "dev_train": dev_train, "dev_val": dev_val},
            "split_seed": args.seed,
            "train_fraction": TRAIN_FRACTION,
            "dev_split_seed": args.dev_seed,
            "dev_val_fraction": args.dev_val_fraction,
            "admitted_total": len(uids),
        })
        _write(EVAL_PROTOCOLS_PATH,
               build_eval_protocols(train, test, dev_val))
        _write(STAGE1_PROTOCOL_PATH,
               build_stage1_protocol(hyperparameters, decided_by))

    print(f"{len(train):,} train / {len(test):,} test objects "
          f"(seed {args.seed}, {len(uids):,} admitted)")
    print(f"  of the train pool: {len(dev_train):,} dev_train / "
          f"{len(dev_val):,} dev_val "
          f"(seed {args.dev_seed}, fraction {args.dev_val_fraction})")
    for name, p in build_eval_protocols(train, test, dev_val).items():
        flag = "" if p["reported"] else "   [NOT REPORTED -- selection only]"
        print(f"  {name}: query={p['query_split']}, gallery={p['gallery_split']} "
              f"({p['gallery_size']:,}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
