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

# [DEVIATION D-3b, USER ORDER Kyzen 2026-09-04, verbatim 「80/10/10 就這樣拆」]
# SUPERSEDES the 70/10/20 shape above. The paper's 80/20 (seed 20260816) is
# kept byte-for-byte: the 80% training pool is unchanged and is now trained on
# in full. The paper's 20% ("holdout", 9,138) is halved into `val` (selection)
# and `test` (final), so the shape is 80 / 10 / 10 of the corpus.
#
# The 2026-08-27 objection to this shape is recorded in the audit
# (docs/audit/SPLIT_RETRIEVAL_FRESH_AUDIT_20260904.md §6): the final
# test->test gallery is 4,569 instead of the paper's ~20%, so a paper-size
# gallery is kept as protocol `A20_test_vs_holdout` (query = test, gallery =
# val + test = 9,138). Kyzen reaffirmed the order after reading that section.
#
# What is unchanged: no test asset takes part in selecting anything (`val` and
# `test` are disjoint, and `check_seal` treats `test`, `holdout` and `full` as
# sealed). What changed: every checkpoint trained under 70/10/20 was fitted on
# 31,985 assets and selected on a dev_val INSIDE the training pool; those are
# not comparable with runs under this scheme and their records say so through
# `pools_sha256`.
#
# `dev_train` / `dev_val` are still written, as ALIASES of `train` / `val`, so
# `stage1 --phase dev` and every probe reading `dev_val` keep working without a
# second code path.
SPLIT_SCHEME = "80/10/10"
VAL_FRACTION_OF_HOLDOUT = 0.5
DEFAULT_VAL_SEED = 20260904

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


def split_holdout(holdout: list[str], seed: int,
                  val_fraction: float = VAL_FRACTION_OF_HOLDOUT) -> tuple[list[str], list[str]]:
    """[D-3b] Halve the paper's 20% into (val, test), same discipline as above.

    Returns (val, test) whose union is exactly `holdout`. `val` selects
    checkpoints; `test` is opened once, at the end.
    """
    ordered = sorted(holdout)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * val_fraction))
    return sorted(ordered[:cut]), sorted(ordered[cut:])


def corpus_uids(obj: dict) -> list[str]:
    """The whole admitted corpus from a `splits.json` OBJECT dict, any scheme.

    Under 70/10/20 the corpus was train + test; under 80/10/10 it is
    train + val + test. Built from the primary keys only -- `dev_train` /
    `dev_val` are aliases and must not be double counted -- so the three
    consumers that used to write `train + test` by hand (`run_retrieval`,
    `gallery_index`, `g4_gallery_freeze`) cannot disagree with each other.

    Concatenated in split order, NOT sorted or de-duplicated: `run_retrieval`
    relies on `full` being the protocol's own uid order, and a duplicate is a
    corpus defect its gallery check must see rather than have hidden here.
    """
    return (list(obj.get("train", [])) + list(obj.get("val", []))
            + list(obj.get("test", [])))


def build_eval_protocols(train: list[str], test: list[str],
                         dev_val: list[str] | None = None,
                         holdout: list[str] | None = None) -> dict:
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
            "gallery_size": len(train) + len(test) + (
                len(holdout) - len(test) if holdout is not None else 0),
            "layout_free_context": "omitted",
            "reported": True,
        },
    }
    if holdout is not None:
        # [D-3b] The paper-size gallery. Under 80/10/10 `test` is half the
        # paper's 20%, so test->test ranks against 4,569 candidates where the
        # paper (if its gallery was its 20%) ranked against ~9.6K. This protocol
        # keeps that size: query = test, gallery = val + test. Caveat, recorded
        # here and in the audit: the `val` half of this gallery was used to
        # select the checkpoint (val->val), so this is a paper-size gallery
        # with a selection-side bias on half of it -- not leakage of any test
        # label, but not as clean as A either.
        protocols["A20_test_vs_holdout"] = {
            "query_split": "test",
            "gallery_split": "holdout",
            "gallery_size": len(holdout),
            "layout_free_context": "omitted",
            "reported": True,
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
        # [DL-044] C cannot tell a trained model from an untrained one. Measured
        # 2026-08-30: random fusion towers over the pretrained encoders score
        # full R@1 = 0.9989 on C, against 1.0000 trained, and four of seven
        # conditions move by less than a point. A protocol that a null model
        # passes is not measuring what it names.
        #
        # D is the cheapest way to ask whether that survives a larger gallery:
        # same 4,569 queries, gallery grown 8x to the whole training pool. If
        # the null still scores near 1.0 here, gallery size is not the cause and
        # the task itself is the problem; if it collapses, C was simply too
        # small and A/B may still be sound.
        #
        # It reads NOTHING sealed. `dev_val` is a subset of `train`
        # (asserted below in `split_dev`), so the query's own asset is in the
        # gallery, and `check_seal` looks for "test" or "full" in the splits
        # used -- ("dev_val", "train") is neither, so no --unseal.
        #
        # `reported: False` for the same reason as C, and one more: its gallery
        # is the training pool, so every candidate is an asset the model was
        # fitted on. It is a diagnostic, never a result.
        # [D-3b] Under 80/10/10 `dev_val` is OUTSIDE the training pool, so a
        # gallery of `train` alone would not contain the query's own asset and
        # the evaluator refuses (measured 2026-09-04 16:50 on P10: "4,569 query
        # assets are absent"). The gallery is then train + val: every training
        # asset plus the queries themselves, 41,123.
        d_gallery = "train_val" if holdout is not None else "train"
        protocols["D_dev_val_vs_train"] = {
            "query_split": "dev_val",
            "gallery_split": d_gallery,
            "gallery_size": len(train) + (len(dev_val) if holdout is not None else 0),
            "layout_free_context": "omitted",
            "reported": False,
        }
    return protocols


def build_stage1_protocol(hyperparameters: dict, decided_by: str,
                          fusion: str = DEFAULT_FUSION,
                          tower_sharing: str = DEFAULT_TOWER_SHARING,
                          prefusion_norm: bool = False,
                          image_tokens: int = 1) -> dict:
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
        # [AUDIT 2026-09-03 C8] see FusionConfig.prefusion_norm. Absent in
        # protocols written before this date; the trainer reads absence as False.
        "prefusion_norm": bool(prefusion_norm),
        # [AUDIT 2026-09-03 C4] see FusionConfig.image_tokens; 1 = pooled view.
        "image_tokens": int(image_tokens),
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


def filter_ladder() -> dict:
    """Every filtering stage as before / removed / after.

    [SPEC 20260903 §六, ULIP2 REVIEWER MAJOR 2, 2026-09-03] 「每一道 filter 都要
    輸出 before count / removed count / after count」and a final report of raw /
    usable / train / test / quarantined.

    The exclusion ledger's subtraction was live and silent. Until 2026-09-03 it
    removed nine metadata strings, i.e. nothing; the parse fix made it remove up
    to 332 real uids, and `admitted_uids` still returned a bare list. The
    docstring there asserts the corpus count does not move BECAUSE those 332
    have no annotation sidecar -- true when measured, and asserted by no code.
    If it is ever false by one asset, `admitted_total` leaves 45,692, the 80/20
    partition is taken over a different universe, and every archived
    `train_uid_set_sha256` stops matching with nothing printed. Three integers
    per stage is the whole cost of noticing.
    """
    def _index(name):
        p = paths.LOGS / name
        if not p.exists():
            return set()
        return {json.loads(l)["uid"] for l in p.read_text().splitlines() if l.strip()}

    manifest = set(json.loads(paths.LVIS_MANIFEST.read_text()))
    clouds = _index("pointclouds_index.jsonl")
    renders = _index("renders_index.jsonl")
    anns = _index("annotations_index.jsonl")

    ledger_path = paths.OUTPUTS / "annotation_exclusions.json"
    excluded = (ledger_excluded_uids(json.loads(ledger_path.read_text()))
                if ledger_path.exists() else set())

    stages, cur = [], manifest
    for name, have in (("n03_pointclouds", clouds), ("n04_renders", renders),
                       ("n05_annotate", anns)):
        after = cur & have
        stages.append({"stage": name, "before": len(cur),
                       "removed": len(cur - have), "after": len(after)})
        cur = after
    stages.append({"stage": "exclusion_ledger", "before": len(cur),
                   "removed": len(cur & excluded), "after": len(cur - excluded)})
    return {"raw_assets": len(manifest), "stages": stages,
            "usable_assets": len(cur - excluded),
            "ledger_uids_parsed": len(excluded)}


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

    admitted = (index_uids("pointclouds_index.jsonl")
                & index_uids("renders_index.jsonl")
                & index_uids("annotations_index.jsonl"))
    # The manually rejected assets stay out only while their annotation
    # sidecars are absent; a forced re-annotation or a restored sidecar would
    # readmit them through the index above. The exclusion ledger is the
    # authority, so it is applied here as well.
    ledger = paths.OUTPUTS / "annotation_exclusions.json"
    if ledger.exists():
        admitted -= ledger_excluded_uids(json.loads(ledger.read_text()))
    return sorted(admitted)


def ledger_excluded_uids(ledger: dict) -> set[str]:
    """The uids `annotation_exclusions.json` actually excludes.

    [FIXED 2026-09-03] This walked the ledger as `{uid: entry}`. It is not one:
    it is a metadata dict whose keys are `decided_at`, `decided_by`,
    `decision`, `git_commit`, `corpus_before`, `excluded_total`,
    `corpus_after`, `rendered_assets` and `groups`. The loop therefore
    subtracted nine strings -- 'Kyzen', '45692', '332', a timestamp, the word
    'groups' -- and not one of the 332 real uids, which live under
    `groups.<name>.uids` and were never read.

    The corpus count was nevertheless right, by luck: those 332 have no
    annotation sidecar, so the three-way intersection above had already dropped
    them. What was gone is the property the caller's docstring claims -- that
    the ledger is the authority and a restored sidecar cannot readmit a
    rejected asset. Restoring one sidecar would have readmitted it silently.

    Two shapes are on disk and both are handled, because the ledger records
    them differently and neither is wrong: `n05_quarantine.uids` is a list of
    uid strings, `manual_review_rejected.uids` is a list of dicts carrying
    `uid` beside the CLIP scores the rejection was decided on.
    """
    out: set[str] = set()
    for name, group in (ledger.get("groups") or {}).items():
        for e in (group or {}).get("uids") or []:
            uid = e.get("uid") if isinstance(e, dict) else e
            if not uid:
                raise ValueError(
                    f"annotation_exclusions.json groups.{name} has an entry "
                    f"with no uid: {e!r}. Refusing to guess: an unreadable "
                    "exclusion silently readmits a rejected asset.")
            out.add(str(uid))
    # The ledger states its own total. Cross-checking the parse against it is
    # what turns "the loop read something" into "the loop read the uids": the
    # defect above produced nine entries against a stated 332 and nothing
    # noticed for six days.
    # [ULIP2 REVIEWER MINOR 4] Per group as well as in total. A parse that read
    # the wrong field of the RIGHT NUMBER of entries passes a cardinality check;
    # the ledger already ships each group's own `n` (311 and 21), and comparing
    # those localises a wrong parse instead of only counting one.
    for name, group in (ledger.get("groups") or {}).items():
        n = (group or {}).get("n")
        got = len((group or {}).get("uids") or [])
        if n is not None and int(n) != got:
            raise ValueError(
                f"annotation_exclusions.json groups.{name} says n={n} but "
                f"carries {got} uid(s). Fix the ledger rather than the reader.")
    stated = ledger.get("excluded_total")
    if stated is not None and int(stated) != len(out):
        raise ValueError(
            f"annotation_exclusions.json says excluded_total {stated} but "
            f"groups.*.uids parses to {len(out)} distinct uid(s). The ledger "
            "and its own groups disagree; fix the ledger rather than the "
            "reader.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dev-seed", type=int, default=DEFAULT_DEV_SEED)
    ap.add_argument("--dev-val-fraction", type=float, default=DEV_VAL_FRACTION,
                    help="fraction of the 80%% training pool held out as dev-val "
                         "(NOT ratified; see DEV_VAL_FRACTION)")
    ap.add_argument("--decided-by", default=None)
    ap.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED,
                    help="[D-3b] seed that halves the paper's 20%% into val / test")
    ap.add_argument("--rewrite-stage1-protocol", action="store_true",
                    help="also re-materialise stage1_protocol.json (default: leave "
                         "an existing one untouched -- the split is not the protocol)")
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
        # [PAPER 3.1] 80/20, seed unchanged since 2026-08-16.
        train, holdout = split_assets(uids, args.seed)
        # [D-3b] The paper's 20% halved into val / test.
        val, test = split_holdout(holdout, args.val_seed)
        dev_train, dev_val = train, val          # aliases, see D-3b above

        # [L2-LEAK-OBJECT] A leaking split must not reach disk.
        for a, b, why in ((train, test, "train/test"),
                          (train, val, "train/val"),
                          (val, test, "val/test")):
            if leaked := set(a) & set(b):
                raise AssertionError(f"{len(leaked)} assets in both {why}")
        if set(val) | set(test) != set(holdout):
            raise AssertionError("val + test is not the paper's 20% holdout")
        if set(train) | set(holdout) != set(uids):
            raise AssertionError("train + holdout is not the admitted corpus")

        _write(SPLITS_PATH, {
            "object": {"train": train, "val": val, "test": test,
                       "holdout": holdout, "train_val": train + val,
                       "dev_train": dev_train, "dev_val": dev_val},
            "scheme": SPLIT_SCHEME,
            "split_seed": args.seed,
            "train_fraction": TRAIN_FRACTION,
            "val_seed": args.val_seed,
            "val_fraction_of_holdout": VAL_FRACTION_OF_HOLDOUT,
            # kept for readers of the 70/10/20 file; under 80/10/10 they
            # describe the SAME halving as the two fields above
            "dev_split_seed": args.val_seed,
            "dev_val_fraction": VAL_FRACTION_OF_HOLDOUT,
            "admitted_total": len(uids),
            "decided_by": decided_by,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        })
        _write(EVAL_PROTOCOLS_PATH,
               build_eval_protocols(train, test, dev_val, holdout=holdout))
        if args.rewrite_stage1_protocol or not STAGE1_PROTOCOL_PATH.exists():
            _write(STAGE1_PROTOCOL_PATH,
                   build_stage1_protocol(hyperparameters, decided_by))
        # Plain uid lists, one per line, so the split can be read with `wc -l`
        # and `comm` without Python. [KYZEN 2026-09-04] asked where the folders
        # are: membership is by LIST, the data directories are flat by uid; see
        # tools/materialize_split_dirs.py for a symlink view per split.
        lists = paths.OUTPUTS / "split_lists"
        lists.mkdir(parents=True, exist_ok=True)
        for name, ids in (("train", train), ("val", val), ("test", test),
                          ("holdout", holdout)):
            (lists / f"{name}.txt").write_text("\n".join(ids) + "\n")

    lad = filter_ladder()
    print("filtering, per §六 -- before / removed / after:")
    for st in lad["stages"]:
        print(f"  {st['stage']:<18} {st['before']:>7,} - {st['removed']:>5,} "
              f"= {st['after']:>7,}")
    print(f"  {'RAW':<18} {lad['raw_assets']:>7,}     "
          f"USABLE {lad['usable_assets']:,}   "
          f"quarantined or excluded {lad['raw_assets'] - lad['usable_assets']:,}")
    # The ledger's own claim, checked instead of believed: those 332 are
    # supposed to be already absent from the three-way intersection, so the
    # ledger stage should remove ZERO. If that ever changes, the corpus changed
    # and it says so here rather than in a silently different split.
    # [ULIP2 REVIEWER MINOR 2] Two independent computations of one universe:
    # `filter_ladder` starts from the LVIS manifest and intersects, while
    # `admitted_uids` intersects the three indexes without touching the
    # manifest. `main` PRINTS the first and SPLITS the second, so a uid present
    # in an index but absent from the manifest would make the printed USABLE
    # figure disagree with what was actually split -- and the operator would
    # read the ladder as confirmation of the split. They agree today, which is
    # luck, not a guarantee.
    if lad["usable_assets"] != len(uids):
        print(f"  MISMATCH the ladder says {lad['usable_assets']:,} usable but "
              f"admitted_uids() returned {len(uids):,}. These are two "
              "computations of one set and THIS RUN SPLIT THE SECOND.")
    led = next(st for st in lad["stages"] if st["stage"] == "exclusion_ledger")
    if led["removed"]:
        print(f"  NOTE the exclusion ledger removed {led['removed']:,} asset(s) "
              "that HAD survived n03/n04/n05. The corpus is not the one the "
              "audit measured; every archived train_uid_set_sha256 will differ.")
    print(f"[{SPLIT_SCHEME}] {len(train):,} train / {len(holdout):,} holdout "
          f"(seed {args.seed}, {len(uids):,} admitted)")
    print(f"  holdout halved: {len(val):,} val (selection) / {len(test):,} test "
          f"(final)  (val seed {args.val_seed})")
    for name, p in build_eval_protocols(train, test, dev_val, holdout=holdout).items():
        flag = "" if p["reported"] else "   [NOT REPORTED -- selection only]"
        print(f"  {name}: query={p['query_split']}, gallery={p['gallery_split']} "
              f"({p['gallery_size']:,}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
