"""n15 -- run Table 1's retrieval protocols and say what the numbers rest on.

# IMPLEMENTS-NODE: n15_eval_retrieval

Writes ``retrieval_metrics`` (``table1.json``; ``diagnostics.json`` is the
separate diagnostic artifact, not this channel), ``run_progress`` and
``cost_ledger``.

`metafind/eval/retrieval.py` has been the SCORER since the graph was written.
Nothing ever called it. `eval_protocols.json` has carried ``reported: true`` on
`A_test_gallery` and `B_full_gallery` for months and **no program reads that
file** -- verified 2026-08-30: the only references are `splits.py`, which writes
it, `tests/test_splits.py`, and `chain_overnight.sh`, which checks it exists.

So every number this project holds -- e5 0.9571, e10 0.9471, e25 0.9333 and
0.9321 -- is protocol **C**, hardcoded in `stage1.evaluate_dev_val`, whose
gallery is the 4,569 dev-val assets it also selected the checkpoint on. None of
them is comparable with the paper, and until this module existed none could be.

Three rules this module is built around
---------------------------------------
**Protocols are read, never named.** The first version of this evaluation put
the protocol into the code, which is why `eval_protocols.json` ended up with no
consumer. Conditions come from `retrieval.QUERY_CONDITIONS`; galleries and query
pools come from the artifact.

**The sealed split stays sealed.** [CODEX 2026-08-30] `A` and `B` query the test
split. A hyperparameter sweep that reads them has spent the test set, and no
later result from this corpus is a held-out number again. `--unseal` is required
and is recorded in the output, so breaking the seal is an act, not an accident.

**Paper metrics and diagnostics are separate artifacts.** [ULIP2 REVIEWER
2026-08-30] A later aggregator that finds `signed_target_margin` next to `R@1`
will eventually report one as the other.

**A reported gallery is READ, not re-encoded.** [DL-048] Until 2026-08-30 this
module re-encoded the whole gallery on every run and had never once opened
`gallery_index.json` -- so the artifact n11 stages, G4 verifies and n12 promotes
had **no consumer on the evaluation path**, and Table 1's A and B would have
been scored against bytes no gate ever saw. A `reported: true` protocol now
takes its gallery from the promoted index for this checkpoint's sha256 and
REFUSES if there is not one; see `gallery_from_promoted_index`. Every result
carries `gallery_source`, so "was this number index-backed?" is answerable from
one field.

Why the similarity matrix is never materialised
-----------------------------------------------
Protocol B is 9,138 queries x 45,692 gallery entries x 7 conditions = 2.9e9
scores. Everything below streams over gallery blocks: rank needs only the counts
of strictly-higher and tied entries, and every diagnostic here is an accumulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from metafind import paths, runlog
from metafind.eval.retrieval import (
    QUERY_CONDITIONS,
    condition_mask,
    normalize_for_scoring,
)

__all__ = ["ProtocolResult", "score_streaming", "load_protocols",
           "GALLERY_SOURCES", "gallery_source_for",
           "gallery_from_promoted_index", "main"]

NODE = "n15_eval_retrieval"

# The complete set. A result's `gallery_source` is one of these three and is
# never null: it is the field that makes "was this number index-backed?"
# answerable without reading provenance prose.
GALLERY_SOURCES = ("promoted_index", "direct_dev_encode",
                   "untrained_direct_encode")

# Cosine similarity is bounded above by 1, so the softmax shift needed for a
# stable exp() is a constant rather than a streaming max. One less pass.
_SHIFT = 1.0


def load_protocols(path: Path | None = None) -> dict:
    """The protocols as the artifact defines them.

    Read, not named. `A_test_gallery` / `B_full_gallery` / `C_dev_selection` are
    the artifact's current keys and this module must keep working when they are
    not.
    """
    p = path or (paths.OUTPUTS / "eval_protocols.json")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run n09_build_splits, which writes it")
    return json.loads(p.read_text())


def block_plan(ng: int, block: int) -> list[tuple[int, int, int]]:
    """`(start, end, count_from)` for every block, ALL THE SAME WIDTH.

    [ULIP2 REVIEWER 2026-08-30, MAJOR] The third exit of one root cause. Pass 1
    reads `own` out of a GEMM, and the comparisons read every other column out
    of a GEMM -- but a ragged final block is a DIFFERENT SHAPE, and a different
    shape can select a different BLAS kernel, which can return a different last
    bit for the same input.

    MEASURED on collapsed galleries (400 trials, ten block sizes each), before
    this: **143 of 400 gave a different answer depending on `block`**, and not
    by one -- a 27-row gallery of identical vectors scored rank 27 at block 8,
    rank 18 at block 3 and rank 8 at block 7. `block` is a performance knob and
    it was changing the reported metric, `tie_count` worst of all -- the
    diagnostic added specifically to detect collapse.

    So the final block OVERLAPS its predecessor instead of being short, and
    `count_from` says where its already-counted columns end. Every GEMM is
    `(nq, d) @ (d, block)`. `ng <= block` degenerates to a single block, where
    there is nothing to be inconsistent with.

    No epsilon anywhere: a tolerance chosen after seeing a score distribution is
    a fitted constant, and it would hide the case it was fitted to.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    if ng <= block:
        return [(0, ng, 0)]
    out, covered = [], 0
    for s in range(0, ng - block + 1, block):
        out.append((s, s + block, max(s, covered)))
        covered = s + block
    if covered < ng:
        out.append((ng - block, ng, covered))
    return out


def score_streaming(query: np.ndarray, gallery: np.ndarray,
                    targets: np.ndarray, block: int = 4096) -> dict:
    """Rank, margins and spread for every query, without an (nq, ng) array.

    `targets[i]` is the gallery COLUMN holding query i's own asset -- passed in,
    never assumed to be the diagonal, because under a full-corpus gallery a
    query's asset sits wherever the index put it.

    Ties count AGAINST the model, exactly as `rank_of_target` does: rank =
    strictly-higher + tied + 1. A collapsed model scoring everything equally
    would otherwise report R@1 = 100%.

    Every comparison in here is between numbers produced by the SAME arithmetic
    path. That is not fussiness; it is the whole correctness argument, and it
    was got wrong twice. See `block_plan` and the note on `own` below.

    Returns per-query arrays. The caller decides what to keep.
    """
    nq, ng = query.shape[0], gallery.shape[0]
    if targets.shape != (nq,):
        raise ValueError(f"{targets.shape} targets for {nq} queries")
    if targets.size and (targets.min() < 0 or targets.max() >= ng):
        raise ValueError("a target column is outside the gallery")
    # [ULIP2 REVIEWER 2026-08-30, MAJOR] Block-independence held only because
    # `encode_pools` happens to hand over float64. Nothing here required it, so
    # one test, one future caller, or one "save memory" refactor passing float32
    # brings the defect back in silence -- and `tie_count` is the diagnostic
    # added specifically to detect collapse.
    #
    # MEASURED on the current code, collapsed gallery, d=1280, L2-normalised:
    #     float32   tie_count changed with `block` in 6 of 6 trials at
    #               ng = 999, 4,569 and 9,138 alike (4568 vs 4567)
    #     float64   0 of 6 at every size
    #
    # Third time today a property was left to the caller to maintain instead of
    # enforced by the callee: `ARM_EXCLUDED` declared fields it did not contain,
    # and `ENFORCED_SINGLETONS` checked a merged dict so the encoding half was
    # never looked at. A guard is the difference between "nobody calls it that
    # way" and "it cannot be called that way".
    if query.dtype != np.float64 or gallery.dtype != np.float64:
        raise ValueError(
            f"score_streaming needs float64 (got {query.dtype}/{gallery.dtype}). "
            "In float32 tie_count moved with `block` in 6 of 6 trials at every "
            "production gallery size; in float64, 0 of 6. Normalise through "
            "retrieval.normalize_for_scoring.")

    plan = block_plan(ng, block)

    # [ULIP2 REVIEWER 2026-08-30, BLOCKER] `own` is read out of the SAME GEMM,
    # in the SAME block shape, that the comparisons below use. It used to be a
    # row-wise product, and the -inf mask that removed the target from its own
    # comparison did not fix the root cause: `own` also had to be compared
    # against every OTHER column, and a one-ULP difference between two
    # arithmetic paths is decisive when the columns are equal to each other.
    #
    # MEASURED, on a collapsed gallery (30 identical rows, 200 trials):
    #     own (row-wise) = 0.6679995781284078
    #     sim (GEMM)     = 0.6679995781284077   -- one ULP smaller, all 30
    #     -> higher = 0, tied = 0 -> rank 1, tie_count 0
    #     truth: rank 30, tie_count 29
    # 20 of 200 disagreed, and ONLY collapsed ones -- a totally collapsed model
    # reported as perfect retrieval, in the one situation n15 exists to
    # diagnose. It also disabled the negative control: with `higher = 0` and
    # `tied = 0` the answer does not depend on which column the target is, so
    # `shuffle_targets` scored identically to the real run.
    #
    # Not optimised into one pass by gathering only the target rows: a different
    # matrix shape may select a different kernel and the same one-ULP difference
    # returns through another door. MEASURED cost of the extra pass at protocol
    # B scale: 0.7 s per condition, ~10 s for all seven, against an encode of
    # 45,692 assets through PointBERT.
    own = np.empty(nq, dtype=np.float64)
    seen = np.zeros(nq, dtype=bool)
    for s, e, c in plan:
        rows = np.nonzero((targets >= c) & (targets < e) & ~seen)[0]
        if not rows.size:
            continue
        sim = query @ gallery[s:e].T
        own[rows] = sim[rows, targets[rows] - s]
        seen[rows] = True
        del sim
    if not seen.all():
        raise AssertionError(
            f"{(~seen).sum()} target columns were never visited; block_plan and "
            "the target range disagree")

    higher = np.zeros(nq, dtype=np.int64)
    tied = np.zeros(nq, dtype=np.int64)
    top1 = np.full(nq, -np.inf)
    top1_col = np.zeros(nq, dtype=np.int64)
    top2 = np.full(nq, -np.inf)
    # "hardest non-target" is the best score among entries that are NOT the
    # answer. It is the quantity a retrieval failure is made of, and it is not
    # the same as top2: when the target already ranks first, top2 IS the hardest
    # negative, but when it does not, top1 is.
    hardest = np.full(nq, -np.inf)
    off_sum = np.zeros(nq)
    off_sq = np.zeros(nq)
    exp_sum = np.zeros(nq)
    xexp_sum = np.zeros(nq)

    for s, e, c in plan:
        full = query @ gallery[s:e].T                     # (nq, block)
        # Only the columns this block is responsible for. The overlap with the
        # previous block was already counted there, and counting it twice would
        # inflate every rank near the end of the gallery.
        off = c - s
        sim = full[:, off:]

        in_block = (targets >= c) & (targets < e)
        blk = sim.copy()
        rows = np.nonzero(in_block)[0]
        if rows.size:
            blk[rows, targets[rows] - c] = -np.inf

        higher += (blk > own[:, None]).sum(axis=1)
        tied += (blk == own[:, None]).sum(axis=1)

        # Merging two (top1, top2) pairs. The second largest of
        # {top1, top2, b1, b2} is max(min(top1, b1), top2, b2) -- writing it out
        # because the obvious `if b1 > top1: top2 = top1` is wrong whenever the
        # block's own runner-up beats the incumbent leader.
        b1 = sim.max(axis=1)
        b1_col = sim.argmax(axis=1) + c
        b2 = (np.partition(sim, -2, axis=1)[:, -2] if sim.shape[1] >= 2
              else np.full(nq, -np.inf))
        new_top2 = np.maximum(np.minimum(top1, b1), np.maximum(top2, b2))
        top1_col = np.where(b1 > top1, b1_col, top1_col)
        top1 = np.maximum(top1, b1)
        top2 = new_top2
        hardest = np.maximum(hardest, blk.max(axis=1))

        # `blk` carries -inf at the target column, so the answer cannot inflate
        # its own spread statistics. The target's contribution is removed from
        # the counts instead (`n_off` below is ng - 1).
        #
        # [ULIP2 REVIEWER 2026-08-30, MINOR] The entropy accumulators read `sim`
        # -- UNMASKED -- while the std accumulators read `masked`. So
        # `off_target_std` excluded the answer and `off_target_entropy` did not,
        # under a name and a comment that both said it did. Both now read the
        # masked block: exp(-inf) is exactly 0.0, which drops the target from Z
        # with no branch, and `masked` is 0.0 at that column so the x*e^x term
        # contributes 0.0 there rather than the -inf * 0.0 = nan that `blk`
        # would give.
        masked = np.where(np.isfinite(blk), blk, 0.0)
        off_sum += masked.sum(axis=1)
        off_sq += (masked ** 2).sum(axis=1)
        ex = np.exp(blk - _SHIFT)
        exp_sum += ex.sum(axis=1)
        xexp_sum += (masked * ex).sum(axis=1)
        del full, sim, blk, ex, masked

    # Ties count AGAINST the model, as `rank_of_target` does. Both counts are
    # already over non-targets only, so there is nothing to subtract.
    rank = higher + tied + 1
    n_off = ng - 1
    off_mean = off_sum / max(n_off, 1)
    off_var = np.maximum(off_sq / max(n_off, 1) - off_mean ** 2, 0.0)
    # H = logZ - (sum x e^x)/Z, with the constant shift folded back in.
    # The clamp matters now that the target is masked out of Z: a one-entry
    # gallery has NO off-target column, so exp_sum is exactly 0 and an unclamped
    # log would emit -inf plus a RuntimeWarning. The same `max(n_off, 1)` clamp
    # two lines up already makes `off_target_std` meaningless-but-finite there;
    # this keeps the pair consistent rather than adding a second convention.
    logZ = np.log(np.maximum(exp_sum, 1e-300)) + _SHIFT
    entropy = logZ - xexp_sum / np.maximum(exp_sum, 1e-300)

    return {
        "rank": rank,
        "target_score": own,
        "top1_score": top1,
        "top1_col": top1_col,
        "top2_score": top2,
        "hardest_non_target_score": hardest,
        # The single most informative diagnostic: how far the answer beat the
        # best wrong answer. Negative means the query was lost, and by how much.
        "signed_target_margin": own - hardest,
        "top1_top2_gap": top1 - top2,
        # `tied` is already counted over NON-targets, so there is nothing to
        # subtract. It said `tied - 1` -- a leftover from when the target
        # compared against itself -- and under-reported every tie by one.
        "tie_count": tied,
        "off_target_std": np.sqrt(off_var),
        "off_target_entropy": entropy,
    }


def embedding_health(gallery: np.ndarray) -> dict:
    """Is the gallery still using its embedding space, or has it collapsed?

    `effective_rank` is exp(entropy of the normalised covariance eigenvalues)
    (Roy & Vetterli). A value near the full dimension means the space is used; a
    value near 1 means every asset embeds to almost the same direction, which is
    the shape of collapse that still scores well when query and gallery see
    identical inputs.

    Eigenvalues of the (d, d) covariance rather than an SVD of the (n, d)
    matrix: same spectrum, and d is 1,280 while n can be 45,692.
    """
    # UNCENTRED second moment, deliberately. Centring subtracts the mean
    # direction -- which under collapse is the ONLY direction there is, so a
    # centred spectrum measures the leftover jitter and reports a collapsed
    # space as full rank. Measured while writing the test for this: a gallery
    # of nearly identical unit vectors scored effective rank 15.4 out of 16
    # under centring, and the test that was supposed to catch collapse passed
    # on a fixture that was not collapsed.
    m = (gallery.T @ gallery) / max(gallery.shape[0], 1)
    ev = np.linalg.eigvalsh(m.astype(np.float64))
    ev = np.clip(ev, 0, None)
    p = ev / max(ev.sum(), 1e-300)
    nz = p[p > 0]
    return {
        "dim": int(gallery.shape[1]),
        "per_dim_std_min": float(gallery.std(axis=0).min()),
        "per_dim_std_median": float(np.median(gallery.std(axis=0))),
        "per_dim_std_max": float(gallery.std(axis=0).max()),
        # exp(entropy of the normalised spectrum) (Roy & Vetterli). Near the
        # full dimension: the space is used. Near 1: every asset embeds to
        # almost one direction -- the shape of collapse that still scores well
        # when query and gallery see identical inputs.
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "effective_rank_centred": _centred_effective_rank(gallery),
    }


def _centred_effective_rank(gallery: np.ndarray) -> float:
    """The same statistic with the mean removed. Reported beside, never instead.

    It answers a different question -- how much the assets differ FROM EACH
    OTHER -- and the pair is informative: a high centred rank with a low
    uncentred one is a gallery that varies only in a thin shell around one
    direction.
    """
    x = gallery - gallery.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(((x.T @ x) / max(x.shape[0] - 1, 1)
                                     ).astype(np.float64)), 0, None)
    p = ev / max(ev.sum(), 1e-300)
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def jsonable(x: float) -> float | None:
    """`None` for a non-finite score, because `Infinity` is not valid JSON.

    [ULIP2 REVIEWER 2026-08-30] With a one-entry gallery there is no non-target,
    so `hardest_non_target_score` is -inf and `signed_target_margin` is +inf.
    `json.dumps` writes `Infinity` happily and a strict parser then rejects the
    whole sidecar -- which is a machine-read file.
    """
    return float(x) if np.isfinite(x) else None


def quantiles(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if not a.size:
        return {}
    q = np.percentile(a, [1, 5, 50, 95, 99])
    return {"p1": float(q[0]), "p5": float(q[1]), "p50": float(q[2]),
            "p95": float(q[3]), "p99": float(q[4]),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def resolve_split(splits: dict, name: str) -> list[str]:
    """A protocol's split NAME to its uid list.

    `full` is not a key in `splits.json`; it is train + test, which is the whole
    admitted corpus. Built here rather than read from a constant so it cannot go
    stale -- `splits.py:17`, `pointclouds.py` and `renders.py` still carry
    "46,052" and "9,211" in prose, and the real counts are 45,692 and 9,138.
    """
    if name == "full":
        return list(splits["train"]) + list(splits["test"])
    if name not in splits:
        raise ValueError(
            f"protocol names split {name!r}, which splits.json does not have "
            f"({sorted(k for k in splits if isinstance(splits[k], list))})")
    return list(splits[name])


def degraded_render_uids() -> set[str]:
    """Assets whose own render sidecar reports an anomaly, from the manifest.

    [MASTER DECISION 2026-09-03, under Kyzen's delegation. The corpus is NOT
    changed.]

    253 admitted assets carry a degraded render -- 11 effectively blank, and 47
    of the anomalies sit inside the sealed test split. Their image vectors were
    computed from those frames, and they are live gallery candidates.

    Two ways to handle that, and only one is reversible. Removing them changes
    the corpus from 45,692, which changes the 80/20 partition, which invalidates
    every archived `train_uid_set_sha256` and every number measured so far --
    for an effect nobody has measured yet. Keeping them and making them
    FILTERABLE AT EVALUATION turns the question into a sensitivity axis, which
    is what §十六 is for: run the protocol twice and report the difference,
    instead of deciding it in advance.

    So the corpus keeps them, `filters.json` flags them, and this is the
    evaluator's opt-in exclusion. `--exclude-degraded-renders` records the count
    in the result, so a filtered number can never be mistaken for an unfiltered
    one.
    """
    import json

    m = paths.OUTPUTS / "manifest" / "assets.jsonl"
    if not m.exists():
        raise SystemExit(
            f"--exclude-degraded-renders needs {m}, which does not exist. "
            "Build it with `python tools/build_dataset_manifest.py`.")
    out = set()
    for line in m.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("admitted") and row.get("render_anomaly"):
            out.add(row["uid"])
    return out


def _splits_identity() -> dict:
    """Which split file a table was scored under: bytes, seeds, admitted count.

    A regenerated splits.json under another seed would move most of a new test
    set into an old checkpoint's training pool, and every size check would
    still pass. The table has to carry the split's identity for that to be
    detectable afterwards.
    """
    import hashlib

    p = paths.OUTPUTS / "splits.json"
    raw = json.loads(p.read_text())
    obj = raw.get("object", {})
    return {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "split_seed": obj.get("split_seed", raw.get("split_seed")),
            "dev_split_seed": obj.get("dev_split_seed", raw.get("dev_split_seed")),
            "admitted_total": obj.get("admitted_total", raw.get("admitted_total"))}


def check_seal(protocol_name: str, protocol: dict, unsealed: bool) -> bool:
    """Refuse a sealed-split protocol unless the operator said so, in words.

    [CODEX 2026-08-30] `A` and `B` query the test split. A hyperparameter sweep
    that reads them has spent the test set: no later number from this corpus is
    a held-out number again, and nothing about the artifacts would show it.

    The guard lives in code rather than in the plan because a procedural rule
    survives exactly until someone is in a hurry. It returns whether the seal
    was broken, and the caller records that in the output -- a run that touched
    the test split must say so on its face.
    """
    # [ULIP2 REVIEWER 2026-08-30] `full` was checked on the GALLERY side only, so
    # a protocol with `query_split: "full"` walked straight past the guard --
    # and `full` is train + test by definition. Reachable precisely because
    # protocols are read rather than named: a new protocol key is the expected
    # case, not an exotic one.
    splits_used = (protocol.get("query_split"), protocol.get("gallery_split"))
    touches_test = "test" in splits_used or "full" in splits_used
    if touches_test and not unsealed:
        raise SystemExit(
            f"{protocol_name} reads the sealed test split "
            f"(query={protocol.get('query_split')!r}, "
            f"gallery={protocol.get('gallery_split')!r}) and --unseal was not "
            "given. This is the final evaluation, not a development metric: "
            "using it to choose a learning rate, an epoch count or a checkpoint "
            "spends the test set permanently. Use C_dev_selection for "
            "development. Pass --unseal only for a reported result.")
    return touches_test


def gallery_source_for(protocol: dict, untrained: bool) -> str:
    """Where this protocol's gallery comes from -- FROM FIELDS, never a name.

    Same rule as `protocol_caveat`, for the same reason: a protocol the artifact
    adds carrying `reported: true` must not silently get the development path
    because its key is not one this module happens to know. `reported` decides
    it.

    `--ckpt-record none` wins over `reported`, and that is not a loophole: an
    untrained run reads no checkpoint record, so it has no Stage 1 sha256, so
    there is no promoted index it could be bound to. It encodes its own gallery
    and the returned value says so on its face.
    """
    if untrained:
        return "untrained_direct_encode"
    if protocol.get("reported", False):
        return "promoted_index"
    return "direct_dev_encode"


def gallery_from_promoted_index(name: str, gallery_uids: list[str], ckpt: dict,
                                backbone, model) -> tuple[dict, np.ndarray]:
    """A reported protocol's gallery: the bytes a gate verified. FAIL CLOSED.

    [DL-048] n15 re-encoded the gallery on every run and never opened
    `gallery_index.json`, so the promoted index had no consumer here at all.
    Table 1's A and B must be scored against the artifact n11 staged, G4
    verified and n12 promoted -- not against a fresh encode nothing checked.

    Every one of these REFUSES. None of them falls back to re-encoding: a silent
    fallback would put a number in `table1.json` that nothing verified, and
    nothing in the artifact would say so.

    Raised by `load_promoted_index_for_checkpoint`, which does the lookup, the
    bytes re-verification and the read as ONE call, and which n15 lets
    propagate untouched -- `FileNotFoundError` (no registry, or the record names
    a file that is gone), `KeyError` (no promoted index for this checkpoint),
    `ValueError` (bytes do not hash to the record, the array disagrees with the
    record's own count/dim, an asset id repeats, an identity field is missing).

    ⚠ **n15 does not re-hash the file itself, and that is deliberate.** A verify
    in one module and an open in another are two separate opens of the same
    path, and the file can change between them: verification that does not hand
    back the bytes it verified has not verified the bytes you score. The
    re-verify is live on every read because it is inside the read.

    Raised here, because they need this protocol's uids or this run's weights:

    * the record's `stage1_checkpoint_sha256` is not this checkpoint. The lookup
      is already keyed by that sha; this reads the FIELD, because
      `gallery_index.main` writes it precisely on the ground that "a key is not
      a field", and `verified_index` checks the field is PRESENT without
      checking it AGREES
    * the live gallery encoder does not hash to the record's
      `gallery_encoder_sha256`
    * the record NAMES NO GATE RECORD -- see the guard below for exactly how
      much and how little that proves
    * a uid this protocol's gallery needs is absent from the index

    ⚠ **ROW ORDER.** `gallery_index.main` writes `sorted(train + test)`;
    `resolve_split(splits, "full")` returns `list(train) + list(test)`. Those
    are DIFFERENT orders and `targets` is built from the protocol's order. Rows
    are gathered by uid lookup, so the matrix returned is in the protocol's
    order. Using the index's own order instead would score every query against
    the wrong row and still produce a complete, plausible Table 1.

    ⚠ **The uid relation is SUPERSET, not equality.** Protocol A's gallery is
    the 9,138 test uids taken out of a 45,692-row index. Requiring the sets to
    be equal would refuse A.

    ⚠ **Whether this path and the re-encode path produce the same vectors is
    UNVERIFIED, and nothing here should be read as claiming they do.** The
    INPUTS match (OBSERVED IMPLEMENTATION: under `image_aggregation: mean`
    `Stage1Dataset.__getitem__` returns `cached["image"]`, the array
    `gallery_index` reads directly, and both build the cloud as xyz concatenated
    with rgb). That is a statement about inputs only. `gallery_index` encodes
    one asset at a time and `encode_pools` encodes in batches of 64, and a
    different batch shape can select a different kernel -- the same class of
    difference `block_plan` exists for. Only a synthetic parity test exists
    today. Measuring the real delta needs a real promoted index.

    Returns `(record, gallery)` where `gallery` is float64 and L2-normalised by
    `normalize_for_scoring`, exactly as `encode_pools.norm` produces. The index
    stores raw float32; this is the SAME single normalisation, not a second one.
    """
    # Owned by `metafind/train/gallery_index.py`. n15 does NOT parse
    # `gallery_index.json` itself: two readers of one registry is how the two
    # halves drift apart.
    from metafind.train.gallery_index import (
        gallery_encoder_sha256, load_promoted_index_for_checkpoint)

    sha = ckpt.get("sha256")
    if not sha:
        raise ValueError(
            f"{name} is reported, so its gallery must come from the promoted "
            "index -- but the checkpoint record carries no sha256, so there is "
            "nothing to look one up by.")

    # (record, ids, embeddings) -- the loader raises if there is no promoted
    # index for this sha, if the file is gone, or if its bytes no longer match
    # the record. Refusing to re-encode after any of those is the whole point:
    # Table 1's reported galleries are the bytes G4 verified and n12 published.
    loaded = load_promoted_index_for_checkpoint(sha)
    required = ("uri", "sha256", "gallery_encoder_sha256",
                "stage1_checkpoint_sha256")
    try:
        record, ids, embeddings = loaded
        assert isinstance(record, dict) and all(k in record for k in required)
    except (TypeError, ValueError, AssertionError):
        raise ValueError(
            "load_promoted_index_for_checkpoint did not return the agreed "
            f"(record, ids, embeddings), where record carries {list(required)}. "
            f"Got {type(loaded).__name__}: {loaded!r:.200}.") from None

    if record["stage1_checkpoint_sha256"] != sha:
        raise ValueError(
            f"the promoted index found for {sha[:16]}... states "
            f"stage1_checkpoint_sha256 {record['stage1_checkpoint_sha256'][:16]}"
            ".... The registry key and the record disagree about which "
            "checkpoint built this index.")

    # [REVIEWER MINOR-2] n15 used to accept ANY registry entry whose bytes
    # hashed correctly, whether or not a G4 PASS had ever existed for it.
    # `promote()` is the only sanctioned writer of the registry and cannot
    # reach its write without a terminal G4 record saying PASS -- but that
    # check fired at PROMOTION time, and this read happens later, against a
    # JSON file on a shared volume that nothing makes immutable. A hand-added
    # key with a matching `sha256` satisfies every other check in this
    # function. So this is not a duplicate of promotion's check; at read time
    # it is the only check of this property anywhere.
    #
    # ⚠ WHAT THIS PROVES, EXACTLY: that the entry NAMES a gate record. Not
    # that the gate passed, and not that the named record exists or says so --
    # someone editing the registry can type a hex string as easily as a
    # digest. Do not upgrade this sentence.
    #
    # ponytail: presence only. The real read-time enforcement is to open
    # `gate_record_uri`, hash it against `gate_record_sha256` and require
    # verdict == PASS and gate_id == G4_gallery_freeze. Not done here because
    # it makes n15 a SECOND parser of G4's record format, which is the failure
    # the single-loader ruling exists to prevent. Upgrade path: a
    # `verified_gate_record()` in `metafind/gates/`, called from here.
    if not record.get("gate_record_sha256"):
        raise ValueError(
            f"the promoted index for {name} names no gate record "
            "(gate_record_sha256 is absent or empty), so nothing in it says "
            "which G4 verdict cleared these vectors. `promote()` writes that "
            "field on every entry it publishes, so an entry without it was "
            "not written by promotion. Refusing: a reported Table 1 number "
            "must be able to name the verdict that cleared its gallery.")

    live = gallery_encoder_sha256(
        backbone, model,
        include_buffers=bool(record.get("gallery_encoder_hash_includes_buffers")))
    if live != record["gallery_encoder_sha256"]:
        raise ValueError(
            f"the gallery encoder loaded here hashes to {live[:16]}... but the "
            f"index was built by {record['gallery_encoder_sha256'][:16]}.... "
            "The vectors in the index are not the ones this model would "
            "produce, and scoring against them would report a model that was "
            "never measured.")

    # No duplicate-id check here, and no ids/rows length check: `verified_index`
    # raises on both, and its docstring says so. A second copy in this module is
    # the drift the single-loader ruling exists to prevent -- these are
    # guaranteed by the callee, not merely true of today's registry.
    at = {u: i for i, u in enumerate(ids)}
    absent = [u for u in gallery_uids if u not in at]
    if absent:
        raise ValueError(
            f"{len(absent):,} of {name}'s {len(gallery_uids):,} gallery uids "
            f"are absent from the promoted index (e.g. {absent[:3]}). The "
            "index does not cover this protocol's gallery; it is not the index "
            "this protocol should be scored against.")

    # By uid, in the PROTOCOL's order. See the ROW ORDER note above.
    rows = np.fromiter((at[u] for u in gallery_uids), dtype=np.int64,
                       count=len(gallery_uids))
    return record, normalize_for_scoring(embeddings[rows])


def protocol_caveat(protocol: dict, splits: dict,
                    gallery_source: str | None = None,
                    n_excluded_g: int = 0) -> str:
    """What a reader has to be told about this protocol's number -- FROM FIELDS.

    [ULIP2 REVIEWER 2026-08-30, MAJOR] This was a
    ``{"A_test_gallery": ..., "B_full_gallery": ...}.get(name, <development>)``
    lookup in `main`, i.e. the module's own opening rule ("Protocols are read,
    never named") broken twelve lines from where the artifact is read. The cost
    is not stylistic: a protocol the artifact adds gets the *development* caveat
    -- "selects checkpoints, never reported" -- printed next to a number that is
    reported. A wrong caveat is worse than no caveat, because it is read.

    `reported` is the field that decides it, and until this function existed it
    had **no consumer anywhere on the evaluation path**: the only reader in the
    repo is a `print` in `splits.py`, which is the file that writes it.

    The two clauses that used to be hardcoded per name are derived instead:

    * "query = test is this project's assumption" from ``query_split == "test"``
    * the distractor count by intersecting the resolved gallery with `train`,
      so the "36,554" that used to be a literal here is counted. `resolve_split`'s
      own docstring lists three files still carrying stale corpus counts in
      prose; this is how a fourth is not created.

    [U-09] is unconditional on a reported protocol: the paper does not state its
    gallery for any of them.

    `gallery_source` is a third derived clause, added for the same reason: a C
    or D number is scored against a gallery this run encoded itself, and nothing
    in the prose said so. It is derived by `gallery_source_for` from `reported`
    and `--ckpt-record none`, never from the protocol's key, and it is passed in
    rather than recomputed so the sentence and the `gallery_source` FIELD in the
    same artifact cannot disagree.
    """
    if not protocol.get("reported", False):
        base = ("development protocol -- eval_protocols.json marks it "
                "reported: false; it selects checkpoints and is never reported")
        return f"{base}; {_source_clause(gallery_source)}" if gallery_source else base
    parts = ["the paper does not state its gallery [U-09]"]
    if protocol.get("query_split") == "test":
        parts.append("query = test is this project's assumption")
    train = set(splits.get("train", ()))
    distractors = sum(1 for u in resolve_split(splits, protocol["gallery_split"])
                      if u in train)
    if distractors:
        parts.append(f"the gallery contains {distractors:,} training assets "
                     "as distractors")
    # [ULIP2 REVIEWER MINOR 6] The distractor count above is over the UNFILTERED
    # split, so under --exclude-degraded-renders it names more assets than the
    # gallery holds -- protocol B would read "36,554 training assets" beside a
    # gallery of 36,348. The docstring says this count is derived rather than
    # literal "so a fourth stale corpus count is not created"; the flag created
    # one by another route. Said in the same sentence rather than plumbed
    # through, because the number is already in hand at the call site.
    if n_excluded_g:
        parts.append(f"{n_excluded_g:,} degraded-render assets were excluded "
                     "from both pools, so the distractor count above is the "
                     "unfiltered split's")
    if gallery_source:
        parts.append(_source_clause(gallery_source))
    return "; ".join(parts)


def _source_clause(gallery_source: str) -> str:
    """One sentence per `GALLERY_SOURCES` value. Unknown values are not silent."""
    return {
        "promoted_index":
            "the gallery is the PROMOTED INDEX, verified by sha256 against the "
            "record n12 published for this checkpoint",
        "direct_dev_encode":
            "the gallery was ENCODED BY THIS RUN and is not the promoted index, "
            "so this number is not index-backed and no gate has seen it",
        "untrained_direct_encode":
            "the gallery was ENCODED BY THIS RUN from untrained fusion towers; "
            "there is no Stage 1 checkpoint and therefore no promoted index",
    }.get(gallery_source, f"unrecognised gallery_source {gallery_source!r}")


def encode_pools(backbone, model, query_uids, gallery_uids, aggregation,
                 device, batch_size, query_pack=None, observation=None,
                 image_tokens: int = 1):
    """Query embeddings per condition, plus gallery embeddings, plus the map.

    The gallery is encoded ONCE and shared by all seven conditions: the gallery
    tower sees every modality regardless of what the query withheld, which is
    what makes a condition a statement about the QUERY.

    ``gallery_uids=None`` returns ``(queries, None)`` and **does not construct a
    dataset, open a file or call the gallery tower at all.** That is how a
    reported protocol's promise is kept: its gallery comes from the promoted
    index, and "the encoder was not called" has to be a property of this
    function rather than a habit of its caller. `run_protocol` is what passes
    None, and `test_a_reported_protocol_never_calls_the_gallery_encoder` is what
    holds it there.

    ``query_pack`` reaches the QUERY dataset only. The gallery pass is
    constructed with ``query_pack=None`` in the same expression that decides the
    conditions, so the two cannot come apart: a gallery built from the query's
    own observations is the leak this argument exists to remove, and it would
    score beautifully. [MASTER ruling 2026-08-31, option (a)] the gallery image
    stays the 12-view mean, so `gallery_index.py` and the promoted index that
    backs every REPORTED protocol need no change and cannot diverge from this
    path.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.train.stage1 import (Stage1Dataset, collate, modules_in_eval,
                                       split_embeds)

    def embed(uids, conditions):
        # `conditions` is empty exactly on the gallery pass, so the pack is tied
        # to the same condition that decides which tower runs. One expression,
        # not two statements a later edit could separate.
        pack = query_pack if conditions else None
        # The observation, like the pack, is a QUERY-side construction: the
        # gallery pass (conditions empty) reads the promoted 12-view mean, per
        # sec. 2.4 "modality-complete and frozen after pretraining".
        obs = observation if conditions else None
        extra = {}
        if obs is not None:
            extra["observation"] = obs
        if image_tokens != 1:
            extra["image_tokens"] = image_tokens
        loader = DataLoader(Stage1Dataset(uids, aggregation, query_pack=pack, **extra),
                            batch_size=batch_size,
                            shuffle=False, collate_fn=collate, num_workers=4,
                            drop_last=False)
        gal, per_cond = [], {c: [] for c in conditions}
        with modules_in_eval(model, getattr(backbone, "model", None)), torch.no_grad():
            for i, batch in enumerate(loader):
                query_embeds, gallery_embeds = split_embeds(batch, backbone, device)
                n = gallery_embeds["text"].size(0)
                if not conditions:
                    gal.append(model.gallery(gallery_embeds).float().cpu())
                for cond in conditions:
                    mask = condition_mask(cond, n).to(device)
                    per_cond[cond].append(
                        model.query(query_embeds, present=mask).float().cpu())
                if i % 20 == 0:
                    print(f"    batch {i}", flush=True)
        return gal, per_cond

    if gallery_uids is None:
        gal = None
        print("  gallery NOT encoded -- it comes from the promoted index",
              flush=True)
    else:
        print(f"  encoding gallery ({len(gallery_uids):,} assets)", flush=True)
        gal, _ = embed(gallery_uids, [])
    print(f"  encoding queries ({len(query_uids):,} assets, 7 conditions)",
          flush=True)
    _, per_cond = embed(query_uids, list(QUERY_CONDITIONS))

    def norm(chunks):
        """Normalise, then score in float64.

        [CODEX 2026-08-30 + ULIP2 REVIEWER 2026-08-30] MEASURED at production
        shape -- d=1280, L2-normalised, a fully collapsed gallery, real gallery
        sizes 999 / 4,569 / 9,138, twelve trials each over four block sizes:

            float32   rank / tie_count changed with `block` in  7-9 of 12
            float64   rank / tie_count changed with `block` in  0 of 12

        R@1 was 0.0000 in every configuration and both dtypes, so the REPORTED
        metric was never at risk -- but `tie_count` is the diagnostic added to
        detect collapse, and in float32 it moved with a performance knob.

        Cost: 45,692 x 1280 float64 is 468 MB, against an encode of the same
        45,692 assets through PointBERT.

        ⚠ This is a precision change, not a definition change. Nothing about how
        a rank is counted differs; the same arithmetic is done with more bits.

        ⚠ **`output/look/dtype_effect.json` is STALE. Do not cite it.**
        [ULIP2 REVIEWER 2026-08-30, MINOR] OBSERVED DATA, 2026-08-30: it is
        byte-identical to `output/look/dtype_effect_prehelper.json`
        (sha256 0a1223c5..., both from run_id 1788026953-186207-187e52,
        code_revision bdc7b8d3) -- the output of `tools/measure_dtype_effect.py`
        BEFORE it gained a baseline, so it compares the two rescorings with each
        other and nothing else. Its verdict field reads "2 cell(s) differ -- this
        IS a decision", which contradicts every docstring that cites it.

        The correct conclusion is in **`output/look/dtype_effect_helper.json`**
        (sha256 e73fe4fc..., run_id 1788027667-208636-5b450e, code_revision
        554d23cf), which compares each dtype against the RECORDED run
        (`baseline_record: data/outputs/ladder/e25_500w/stage1_best_ckpt.json`)
        and reports `bit_exact_agreement_with_baseline: {float32: 6,
        float64: 7}`. The two files are not in conflict: they answer different
        questions, and the stale one answers the question nobody asked. The
        deltas are identical in both -- `image+pc` R@1 0.9879623550 vs
        0.9877434887, one query in 4,569.

        `stage1.evaluate_dev_val`'s docstring cites the same stale filename and
        is in the Stage 1 import closure, so it is NOT edited from here.
        """
        import torch
        return normalize_for_scoring(torch.cat(chunks).numpy())

    return ({c: norm(v) for c, v in per_cond.items()},
            None if gal is None else norm(gal))


def apply_control(control: str, targets: np.ndarray, n_gallery: int,
                  seed: int) -> tuple[np.ndarray, str]:
    """Negative controls. Without one, a high score proves nothing.

    [CODEX + ULIP2 REVIEWER 2026-08-30] Every checkpoint this project holds
    reports `full` R@1 = 1.0000 while the paper reports 0.517.

    ``shuffle_targets``
        Each query is scored against somebody else's asset. **This is a WIRING
        CHECK, not a discriminator between explanations.** See the section
        below: its green light is not a conclusion, and it must still be run.

    ``none``
        The real measurement.

    What `shuffle_targets` can and cannot detect
    --------------------------------------------
    [ULIP2 REVIEWER 2026-08-30, BLOCKER] This docstring used to claim
    `shuffle_targets` tests the "both towers see identical input" hypothesis. It
    does not, and the refutation was already sitting in this module's own test
    file: `test_shuffling_the_targets_collapses_the_metric_to_chance` builds
    ``g = q.copy()`` -- literally two towers seeing identical input, real
    R@1 = 1.0 -- and the control scores below 0.05. **Green.** The control passes
    hardest exactly when the defect it was said to test is total.

    The reason is arithmetic, not luck. When q_i is close to g_i, permuting the
    target column moves the target from the argmax to a random column, so the
    rank goes to O(n_gallery) and R@1 to ~1/n_gallery **whatever produced the
    similarity**. A model that is genuinely good, a model whose towers are the
    same function, and a model whose queries are all identical to their own
    gallery rows all give the same answer here.

    * **CAN detect:** that the rank arithmetic depends on which column is the
      target. This repo has shipped a bug of exactly that shape once -- when
      `own` came from a row-wise product and the comparisons from a GEMM,
      `higher` and `tied` were both 0, the rank did not depend on `targets` at
      all, and the shuffled run scored identically to the real one. That is what
      the control caught, and it is worth keeping for.
    * **CANNOT detect (no discriminative power, all three):**
      "both towers see identical input" -- shown above; "the embedding space has
      collapsed" -- a collapsed gallery ranks last both shuffled and unshuffled,
      so both cells read 0.0 and neither is informative about the other; "the
      task is saturated at this gallery size" -- chance is 1/n_gallery in both
      the saturated and the honest case, so the control cannot tell a gallery
      that is too small from one that is not.

    ⚠ Run it anyway. A failing `shuffle_targets` is decisive (the metric is not
    retrieval); a passing one is only the absence of that one defect.

    **Candidate explanations for `full` R@1 = 1.0000, and their status:**

    * **float32 rounding inflates it** -- **ELIMINATED 2026-08-30.** Rescoring
      `e25_500w` in float64 gives `full` = 1.000000000, bit-exact.
      ⚠ The evidence is `output/look/dtype_effect_helper.json`, NOT
      `output/look/dtype_effect.json` -- see the note in `norm()` above.
    * **Both towers see identical input** -- `INFERENCE`, and **still open**.
      The mechanism is confirmed in code (p_mask leaves all three modalities
      present in 0.7^3 = 34.3% of steps, and GalleryTower calls its fusion
      without `present`), but a mechanism being real does not make it the only
      cause. Nothing in this module tests it; `shuffle_targets` does not.
    * **The embedding space has collapsed** -- open. `embedding_health`'s
      uncentred effective rank addresses it; not yet run on a real checkpoint.
    * **The task is saturated at this gallery size** -- open. Needs protocol A
      or B, where the gallery is 2x and 10x larger.
    * **Stage 1 training is not what produced it** -- addressable with
      ``--ckpt-record none`` (see `main`), which scores the two fusion towers at
      random initialisation. This is the one control here with discriminative
      power against the first candidate, which is why it stopped being a
      docstring suggestion and became a code path.

    ⚠ **`exclude_target` was specified and is NOT implemented, deliberately.**
    Removing the answer and asking what is retrieved instead is already measured
    by every normal run: `hardest_non_target_score` IS the best score with the
    target excluded, and `signed_target_margin` is the gap it leaves, both per
    query. A separate mode would recompute the same numbers and report an "R@1"
    whose target does not exist -- a probe that cannot return a positive, which
    is the defect class this project has now hit six times. If the intent was
    something else, say what, and it gets built.
    """
    if control == "none":
        return targets, "none"
    if control == "shuffle_targets":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(targets)
        # A derangement is not enforced: at n = 9,138 the expected number of
        # accidental fixed points is 1, and forcing zero would make the control
        # slightly EASIER than chance rather than equal to it.
        return shuffled, "shuffle_targets"
    raise ValueError(f"unknown control {control!r}")


def load_stage2_over_stage1(record_path: str, stage1_ckpt: dict,
                            variant: str = "full") -> dict:
    """Read a Stage 2 record, refuse a wrong parent, build the tower it fits.

    The tower is built WITH the ESSGNN branch (`use_layout=True`) so the Stage 2
    state's `query.layout_encoder.*` and `query.layout_weight` have somewhere to
    land; evaluation then passes no layout, which is exactly what sec. 3.2
    describes for the w/ ESSGNN row. The loss module is the Stage 1 one, built
    only so `load_stage1_checkpoint` can restore its temperature buffer.
    """
    from metafind.train.stage1 import build_model
    from metafind.train.stage1 import load_protocols as load_stage1_protocols
    from metafind.train.stage2 import (Stage2Data, build_stage2_model,
                                       load_stage2_protocols)

    rec = json.loads(Path(record_path).read_text())
    # stage2.py writes variant_ckpts.json, one record per Table 3 variant id;
    # a single-record file is accepted too.
    if "uri" not in rec:
        if variant not in rec:
            raise SystemExit(f"{record_path} holds variants {sorted(rec)}; "
                             f"{variant!r} is not among them")
        rec = rec[variant]
    parent = rec.get("stage1_checkpoint_sha256")
    if parent != stage1_ckpt.get("sha256"):
        raise SystemExit(
            f"{record_path} was fine-tuned from Stage 1 checkpoint "
            f"{str(parent)[:16]} but --ckpt-record is {str(stage1_ckpt.get('sha256'))[:16]}. "
            "The w/ ESSGNN row must lay the Stage 2 query weights over the towers "
            "they were fine-tuned from; pass that checkpoint's record.")
    for k in ("uri", "sha256", "lambda_init"):
        if k not in rec:
            raise SystemExit(f"{record_path} lacks {k!r}; not a Stage 2 record")
    if hashlib.sha256(Path(rec["uri"]).read_bytes()).hexdigest() != rec["sha256"]:
        raise SystemExit(f"{rec['uri']} does not match the sha256 in its record")
    encoding, training, hyperparameters = load_stage1_protocols()
    _stage2_proto, _edge_proto, arch_proto = load_stage2_protocols()
    data = Stage2Data("cpu")               # only for node_dim / edge_dim
    model = build_stage2_model(encoding, training, hyperparameters, arch_proto,
                               node_feat_dim=data.node_dim, edge_feat_dim=data.edge_dim,
                               use_layout=True,
                               init_lambda=float(rec["lambda_init"]["init_lambda"]))
    _, loss_fn = build_model(encoding, training, hyperparameters)
    print(f"stage2 record {record_path}\n  variant {rec.get('variant_id')}  "
          f"parent {str(parent)[:16]}  lambda_0 {rec['lambda_init']['init_lambda']:.4f}  "
          f"steps {rec.get('steps')}", flush=True)
    return {"record": rec, "model": model, "loss_fn": loss_fn}


def overlay_stage2_weights(model, rec: dict, device: str) -> None:
    """Lay the Stage 2 trainable state over the Stage 1 towers, and check it
    covers what Stage 2 trains -- the query fusion, the ESSGNN, lambda."""
    import torch
    state = torch.load(rec["uri"], map_location=device, weights_only=False)["trainable_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"Stage 2 state has keys the tower lacks: {sorted(unexpected)[:5]}")
    should = {n for n, _ in model.named_parameters()
              if n.startswith("query.fusion") or n.startswith("query.layout_encoder")
              or n == "query.layout_weight"}
    absent = {n for n in should if n not in state
              and not n.endswith("fusion.mask_tokens")}   # frozen under masking=none
    if absent:
        raise SystemExit(f"Stage 2 state does not cover {sorted(absent)[:5]}; the "
                         "w/ ESSGNN row would be scored with Stage 1 query weights")
    lam = float(model.query.layout_weight.item())
    print(f"  Stage 2 query weights laid over the Stage 1 parent: {len(state)} tensors, "
          f"lambda {lam:.4f} (unused at evaluation: layout=None)", flush=True)


def _construction_kwargs(observation, image_tokens: int) -> dict:
    """Only what differs from the pre-2026-09-03 construction, as keywords."""
    out = {}
    if observation is not None:
        out["observation"] = observation
    if image_tokens != 1:
        out["image_tokens"] = image_tokens
    return out


def run_protocol(name: str, protocol: dict, splits: dict, backbone, model,
                 aggregation: str, device: str, batch_size: int,
                 control: str, seed: int, block: int,
                 untrained: bool = False,
                 ckpt: dict | None = None,
                 query_pack=None,
                 exclude_uids: set | None = None,
                 observation=None, image_tokens: int = 1) -> tuple[dict, list]:
    """One protocol, seven conditions. Returns (core result, per-query rows).

    The gallery's SOURCE is decided here, from the protocol's own fields, not by
    the caller handing one in -- see `gallery_source_for`. A reported protocol
    gets the promoted index or an exception; nothing in between.
    """
    query_uids = resolve_split(splits, protocol["query_split"])
    gallery_uids = resolve_split(splits, protocol["gallery_split"])

    # BOTH sides, unlike the query-pack drop below. A degraded asset left in the
    # gallery is still a candidate every other query can be scored against, so
    # excluding it from the query pool alone would measure something in between
    # and call it either.
    n_excluded_q = n_excluded_g = 0
    if exclude_uids:
        n_excluded_q = sum(1 for u in query_uids if u in exclude_uids)
        n_excluded_g = sum(1 for u in gallery_uids if u in exclude_uids)
        query_uids = [u for u in query_uids if u not in exclude_uids]
        gallery_uids = [u for u in gallery_uids if u not in exclude_uids]
        print(f"  degraded-render exclusion: {n_excluded_q} from the query pool, "
              f"{n_excluded_g} from the gallery. This number is NOT comparable "
              "with an unfiltered run of the same protocol.", flush=True)

    # [MASTER ruling 2026-08-31] Assets with no second observation are dropped
    # from the QUERY pool only. The GALLERY keeps every uid, so the denominator
    # -- the thing an R@1 is a fraction of -- is unchanged and this protocol
    # stays comparable to the same protocol without a pack on that axis. What
    # changes is `n_query`, which is recorded below.
    dropped_queries = []
    if query_pack is not None:
        query_uids, dropped_queries = query_pack.covered(query_uids)
        if dropped_queries:
            print(f"  {len(dropped_queries)} query asset(s) dropped: no second "
                  f"observation. Gallery unchanged at {len(gallery_uids):,}.",
                  flush=True)
            # A dropped query is neither a hit nor a miss; on a development
            # protocol that is recorded and tolerated, on a REPORTED one it
            # would make the cell an R@k over a subset of the test split while
            # every size check stayed green. A reported number is over the
            # whole split or it is not the protocol's number.
            if protocol.get("reported", False):
                raise ValueError(
                    f"{name} is a reported protocol and {len(dropped_queries)} "
                    "test query asset(s) have no second observation in the "
                    "query pack. Build the pack over the whole test split, or "
                    "score this protocol without a pack.")

    # [ULIP2 REVIEWER 2026-08-30, MINOR] An empty pool used to WRITE A TABLE.
    # `R@1: float((ranks <= 1).mean()) if ranks.size else 0.0` turned "there was
    # nothing to score" into a reported 0.0000, in a file whose whole purpose is
    # to be read as a result. `stage1.evaluate_dev_val:770` returns `{}` on an
    # empty pool precisely to keep "no measurement" distinguishable from "a
    # measurement of zero"; n15 wrote out the indistinguishable version instead.
    # Refusing is the n15 equivalent: this node's output is Table 1, and there
    # is no honest Table 1 row for a protocol with no queries.
    if not query_uids or not gallery_uids:
        raise ValueError(
            f"{name} resolves to {len(query_uids):,} queries and "
            f"{len(gallery_uids):,} gallery entries "
            f"(query_split={protocol['query_split']!r}, "
            f"gallery_split={protocol['gallery_split']!r}). An empty pool has no "
            "R@1: reporting 0.0000 for it would put 'nothing was measured' and "
            "'the model got everything wrong' in the same cell.")

    # [ULIP2 REVIEWER 2026-08-30, MINOR] A duplicated gallery uid was COUNTED
    # AND ALLOWED. `col` is a dict comprehension, so it keeps the LAST index of a
    # repeated uid -- and the earlier copy is then a gallery row bit-identical to
    # the target, which ties with it, which counts against the model (ties count
    # against, by design). So R@1 is depressed by exactly the duplicated queries
    # and the only trace is an integer in a field nobody diffs. Refuse: the
    # gallery comes from `splits.json`, so a duplicate there is a corpus defect
    # to fix at the source, not a condition to score under.
    dupes = len(gallery_uids) - len(set(gallery_uids))
    if dupes:
        seen, repeated = set(), []
        for u in gallery_uids:
            if u in seen and u not in repeated:
                repeated.append(u)
            seen.add(u)
        raise ValueError(
            f"{name}'s gallery ({protocol['gallery_split']!r}) has {dupes:,} "
            f"duplicate uid(s), e.g. {repeated[:3]}. Each duplicate is a row "
            "bit-identical to some query's target, so it ties with that target "
            "and -- ties counting against the model -- pushes its rank to 2 or "
            "worse. R@1 would be depressed by a data defect and nothing in the "
            "metric would say so. Fix splits.json.")
    col = {u: i for i, u in enumerate(gallery_uids)}
    missing = [u for u in query_uids if u not in col]
    if missing:
        raise ValueError(
            f"{len(missing):,} query assets are absent from the {name} gallery "
            f"(e.g. {missing[:3]}). Every query's own asset must be findable, "
            "or its rank is undefined and the metric silently measures nothing.")
    targets = np.array([col[u] for u in query_uids], dtype=np.int64)

    declared = protocol.get("gallery_size")
    # [BLOCKER, ULIP2 Reviewer 2026-09-03] Against the UNFILTERED size. This
    # compared the declared size to the length AFTER the degraded-render
    # exclusion, so the flag that exclusion exists for raised on every one of
    # the four protocols -- A by 47, B by 253, C by 20, D by 206 -- with a
    # message blaming a stale artifact. The staleness guard is the point and
    # stays live; what it must not do is read a deliberate filter as decay.
    resolved = len(gallery_uids) + n_excluded_g
    if declared is not None and declared != resolved:
        raise ValueError(
            f"{name} declares gallery_size {declared:,} but the split resolves "
            f"to {resolved:,}. One of them is stale."
            + (f" ({n_excluded_g:,} degraded assets were then excluded, leaving "
               f"{len(gallery_uids):,}; that is not the mismatch.)"
               if n_excluded_g else ""))

    # Decided from the protocol's own fields, here in the callee. A reported
    # protocol gets the promoted index or an exception; there is no third
    # outcome and no caller can arrange one. Note the ordering: the three
    # refusals above -- empty pool, duplicate gallery uid, a query whose own
    # asset is not in the gallery -- all fire BEFORE this, so they hold on the
    # index-backed path exactly as they did on the re-encode path.
    source = gallery_source_for(protocol, untrained)
    index_record = None
    if source == "promoted_index":
        index_record, gallery = gallery_from_promoted_index(
            name, gallery_uids, ckpt or {}, backbone, model)
        # `None`, not `gallery_uids`: the gallery encoder is not called at all.
        queries, _ = encode_pools(backbone, model, query_uids, None,
                                  aggregation, device, batch_size, query_pack,
                                  **_construction_kwargs(observation, image_tokens))
    else:
        queries, gallery = encode_pools(backbone, model, query_uids,
                                        gallery_uids, aggregation, device,
                                        batch_size, query_pack,
                                        **_construction_kwargs(observation, image_tokens))

    eff_targets, control_used = apply_control(control, targets,
                                              len(gallery_uids), seed)

    conditions, rows = {}, []
    for cond, q in queries.items():
        r = score_streaming(q, gallery, eff_targets, block=block)
        ranks = r["rank"]
        conditions[cond] = {
            # No `if ranks.size else 0.0` fallback any more. `ranks.size ==
            # len(query_uids)`, and an empty query pool is refused at the top of
            # this function -- that raise is where the property now lives. The
            # fallback was the thing that let "nothing was measured" be written
            # out as the number 0.0.
            "R@1": float((ranks <= 1).mean()),
            "R@5": float((ranks <= 5).mean()),
            "hits@1": int((ranks <= 1).sum()),
            "hits@5": int((ranks <= 5).sum()),
            "n_query": int(len(query_uids)),
            "n_gallery": int(len(gallery_uids)),
            # Always present, so an unfiltered run says 0 rather than saying
            # nothing -- an absent field reads as "not applicable", and here it
            # would read as "not filtered" on a run that was.
            "degraded_renders_excluded": {"query": int(n_excluded_q),
                                          "gallery": int(n_excluded_g)},
            "error_count": int((ranks > 1).sum()),
        }
        for i in range(len(query_uids)):
            rows.append({
                "protocol": name, "condition": cond, "control": control_used,
                "query_uid": query_uids[i],
                "target_uid": gallery_uids[int(eff_targets[i])],
                "target_column": int(eff_targets[i]),
                "target_rank": int(ranks[i]),
                "top1_uid": gallery_uids[int(r["top1_col"][i])],
                "target_score": jsonable(r["target_score"][i]),
                "top1_score": jsonable(r["top1_score"][i]),
                "top2_score": jsonable(r["top2_score"][i]),
                "hardest_non_target_score":
                    jsonable(r["hardest_non_target_score"][i]),
                "signed_target_margin": jsonable(r["signed_target_margin"][i]),
                "top1_top2_gap": jsonable(r["top1_top2_gap"][i]),
                "tie_count": int(r["tie_count"][i]),
                "off_target_std": float(r["off_target_std"][i]),
                "off_target_entropy": float(r["off_target_entropy"][i]),
            })
        conditions[cond]["diagnostics"] = {
            "signed_target_margin": quantiles(r["signed_target_margin"]),
            "target_score": quantiles(r["target_score"]),
            "hardest_non_target_score": quantiles(r["hardest_non_target_score"]),
            "top1_top2_gap": quantiles(r["top1_top2_gap"]),
            "off_target_std": quantiles(r["off_target_std"]),
            "off_target_entropy": quantiles(r["off_target_entropy"]),
            "exact_tie_rate": float((r["tie_count"] > 0).mean()),
            "n_nonfinite_scores": int((~np.isfinite(r["target_score"])).sum()),
        }
        # [ULIP2 REVIEWER 2026-08-30] A cell at the ceiling is reported WITH the
        # reason it cannot be read, beside the number, rather than left for a
        # reader to notice.
        if conditions[cond]["R@1"] >= 0.98:
            conditions[cond]["ceiling_warning"] = (
                f"R@1 {conditions[cond]['R@1']:.4f} with "
                f"{conditions[cond]['error_count']} errors out of "
                f"{len(query_uids)}. A saturated cell cannot separate two "
                "models. Read signed_target_margin, not R@1.")

    core = {
        "protocol": name,
        "query_split": protocol["query_split"],
        "gallery_split": protocol["gallery_split"],
        "n_query": len(query_uids),
        "n_gallery": len(gallery_uids),
        # Always 0 now -- a non-zero value raises above. Kept in the artifact
        # because a reader of table1.json can then see that the check RAN, and
        # because removing a field changes the schema of a file other nodes read.
        "duplicate_gallery_uids": dupes,
        "control": control_used,
        # ⚠ The fields that make "was this number index-backed, and what
        # cleared it?" answerable without a manual join back through
        # `gallery_index.json`. `gallery_source` is one of `GALLERY_SOURCES`
        # and is never null; the rest are null exactly when there is no index
        # and no checkpoint behind them.
        #
        # [REVIEWER MINOR-2] `gate_record_uri` / `gate_record_sha256` had no
        # consumer at all: `promote()` wrote them and nothing read them, so a
        # reported number could not name the verdict that cleared its gallery
        # without a human opening the registry.
        #
        # `gallery_encoder_sha256` is the RECORD's, and it is only non-null on
        # the promoted path. IMPLEMENTATION CHOICE, stated rather than left to
        # be discovered: computing it on a direct encode would hash the whole
        # of `backbone.model` -- OpenCLIP ViT-bigG-14 included -- on every
        # development run, to produce a value nothing compares against.
        # `stage1_checkpoint_sha256` already identifies the weights there.
        # Which observations the QUERY side used. `null` is the pre-2026-08-31
        # construction, where the query read the gallery's own cached vectors --
        # so a table without this field is not "unknown", it is that one, and a
        # reader can tell the two apart without opening the checkpoint.
        # BY UID. A count says something was removed; the uids say which, which
        # is what it takes to check whether two runs dropped the same assets.
        "dropped_query_uids": dropped_queries,
        # Stated so no image cell under a pack is read as held-out: the pack's
        # query view is one of the twelve averaged into the gallery image
        # vector (ruling of 2026-08-31, option (a)); the E1 protocol's
        # leave-one-view-out gallery is a separate index, not this one.
        "query_observation_note": (
            "query image = one of the 12 views; the gallery image vector is the "
            "mean of all 12 INCLUDING that view" if query_pack else None),
        "query_construction": (query_pack.identity() if query_pack
                               else {"arms": [], "note": "query reads the "
                                     "gallery's own observations"}),
        "gallery_source": source,
        "gallery_index_uri": index_record["uri"] if index_record else None,
        "gallery_index_sha256": index_record["sha256"] if index_record else None,
        "gallery_encoder_sha256": (index_record["gallery_encoder_sha256"]
                                   if index_record else None),
        "stage1_checkpoint_sha256": (ckpt or {}).get("sha256"),
        "gate_record_uri": index_record["gate_record_uri"] if index_record else None,
        "gate_record_sha256": (index_record["gate_record_sha256"]
                               if index_record else None),
        "conditions": conditions,
        "embedding_health": embedding_health(gallery),
    }
    return core, rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="n15 -- run the Table 1 retrieval protocols.")
    ap.add_argument("--ckpt-record", default=None,
                    help="Stage 1 checkpoint record. Defaults to the canonical "
                         "stage1_ckpt.json. Its sha256 is verified against the "
                         "weights before anything is encoded. Pass the literal "
                         "'none' for the UNTRAINED control: no record is read "
                         "and no Stage 1 weights are loaded, so the two fusion "
                         "towers stay at random initialisation. --init-seed is "
                         "then required.")
    ap.add_argument("--protocol", action="append", default=None,
                    help="protocol key from eval_protocols.json; repeatable. "
                         "Default: every protocol the artifact defines.")
    ap.add_argument("--unseal", action="store_true",
                    help="permit protocols that read the sealed test split. "
                         "Required for a reported result; never for development.")
    ap.add_argument("--control", default="none",
                    choices=("none", "shuffle_targets"),
                    help="negative control. shuffle_targets is a WIRING CHECK: "
                         "a failure is decisive (the metric is not retrieval), "
                         "a pass discriminates between none of the standing "
                         "explanations for R@1 = 1.0000. See apply_control. "
                         "For a control with discriminative power use "
                         "--ckpt-record none.")
    ap.add_argument("--out-dir", default=None,
                    help="directory for this evaluation's artifacts, relative "
                         "to data/outputs/eval. Default: a name built from the "
                         "checkpoint's arm hash and the control.")
    ap.add_argument("--query-pack", default=None,
                    help="query_pack.json from tools/make_query_pack.py: the "
                         "query side then uses a SECOND observation of each "
                         "asset (an alternate caption, one held-out view, a "
                         "second point sample) instead of the gallery's own "
                         "cached vectors. Omit for the pre-2026-08-31 "
                         "construction. The gallery is unchanged either way.")
    ap.add_argument("--query-pc-perturb", default=None,
                    help="perturbation of the query pack's pc arm "
                         "(metafind.data.observation.PC_PERTURBATIONS). "
                         "Default: what the checkpoint record says it was "
                         "trained under.")
    ap.add_argument("--query-image-policy", default=None,
                    choices=("same_mean", "single_view", "held_out_view",
                             "disjoint_views"),
                    help="which observation the QUERY image reads (the gallery "
                         "keeps the 12-view mean). Default: whatever the "
                         "checkpoint record says it trained under, so the "
                         "evaluation construction matches the training one "
                         "unless you say otherwise; same_mean for records "
                         "written before 2026-09-03.")
    # [PAPER 3.2] "Using the Stage-1 head reproduces the 'w/o ESSGNN'" and the
    # w/ ESSGNN row is that same evaluation "on Objaverse-LVIS (which lacks
    # layout and disables ESSGNN)" with the Stage-2 head: the query fusion
    # after layout-aware fine-tuning, no layout vector, the gallery unchanged
    # because Stage 2 froze it. So the w/ ESSGNN row is: the Stage 1 parent's
    # towers, with the Stage 2 checkpoint's query-side weights laid over them.
    ap.add_argument("--stage2-ckpt-record", default=None,
                    help="stage2_<variant>.json from metafind.train.stage2. Its "
                         "stage1_checkpoint_sha256 must equal --ckpt-record's "
                         "sha256. Produces the Table 1 'MetaFind w/ ESSGNN' row: "
                         "Stage 2 query fusion, layout=None, gallery = the Stage 1 "
                         "parent's (frozen in Stage 2).")
    ap.add_argument("--stage2-variant", default="full",
                    help="which record to take from a variant_ckpts.json")
    ap.add_argument("--exclude-degraded-renders", action="store_true",
                    help="drop the 253 admitted assets whose render sidecar "
                         "reports blank, dark or fewer-distinct-than-listed "
                         "views, from BOTH the query and the gallery. Off by "
                         "default: the corpus keeps them and this is a "
                         "sensitivity axis, not a corpus decision. The count "
                         "is recorded in the result.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block", type=int, default=4096,
                    help="gallery columns scored at once. The similarity matrix "
                         "is never materialised: protocol B is 2.9e9 scores.")
    ap.add_argument("--seed", type=int, default=20260830,
                    help="only used by --control shuffle_targets.")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="torch seed for the RANDOM INITIALISATION of the two "
                         "fusion towers. Required with --ckpt-record none, and "
                         "meaningless without it. No default on purpose: the "
                         "towers are two independent draws, so a single "
                         "untrained run can be luck, and the seed has to be a "
                         "stated condition rather than a hidden one. Run "
                         "several.")
    args = ap.parse_args()

    untrained = args.ckpt_record == "none"
    if untrained and args.init_seed is None:
        raise SystemExit(
            "--ckpt-record none needs --init-seed. The result depends on which "
            "random draw the towers got; without the seed on record the run "
            "cannot be repeated and a single number cannot be told from luck. "
            "Run at least three seeds before reading anything into it.")
    if not untrained and args.init_seed is not None:
        raise SystemExit(
            "--init-seed only has meaning with --ckpt-record none. With a real "
            "checkpoint every trainable weight is overwritten by "
            "load_stage1_checkpoint, so the seed would appear in the "
            "provenance of a run it did not affect.")

    from metafind.train.gallery_index import load_checkpoint_record
    from metafind.train.stage1 import (build_model, load_protocols as
                                       load_stage1_protocols,
                                       load_stage1_checkpoint)
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    if runlog.runtime_source_status() != "ok":
        raise SystemExit("cannot fingerprint the source tree; refusing to start.")

    protocols = load_protocols()
    wanted = args.protocol or list(protocols)
    unknown = [p for p in wanted if p not in protocols]
    if unknown:
        raise SystemExit(
            f"{unknown} not in eval_protocols.json ({sorted(protocols)})")

    splits = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    encoding, training, hyperparameters = load_stage1_protocols()

    # Verified before a single asset is encoded: the record is a CLAIM about a
    # file, and an evaluation attributed to the wrong checkpoint is worse than
    # no evaluation.
    #
    # ⚠ `--ckpt-record none` is the ONE path that skips this, and it skips
    # `load_stage1_checkpoint` with it. There is nothing to verify: no record is
    # read and no trained bytes are loaded. Everything the record would have
    # supplied is written as null rather than invented, and `untrained: true`
    # plus `init_seed` say why.
    ckpt = {} if untrained else load_checkpoint_record(args.ckpt_record)
    # [AUDIT 2026-09-03 E1] the query's observation is part of the
    # construction, and it follows the checkpoint unless overridden.
    from metafind.data.observation import Observation, ObservationProtocol
    image_policy = args.query_image_policy or ckpt.get("query_image_policy") or "same_mean"
    if image_policy == "random_view":
        # a stochastic query is not a reported number; the deterministic
        # counterpart of "one view per query" is the uid-seeded single view
        print("checkpoint trained under random_view; evaluating under single_view",
              flush=True)
        image_policy = "single_view"
    observation = ObservationProtocol(
        positive_policy="same_uid",
        query=Observation(image=image_policy), gallery=Observation())
    if observation.is_same_observation():
        observation = None
    image_tokens = int(training.get("image_tokens", 1))
    print(f"query image observation: {image_policy}"
          f"{'' if args.query_image_policy else ' (from the checkpoint record)'}"
          f"; image_tokens={image_tokens}", flush=True)
    seals = {n: check_seal(n, protocols[n], args.unseal) for n in wanted}

    # ---- where these paths actually land (checked 2026-08-30, so the next
    # reader does not check it again) ----
    # There is ONE outputs tree, not two. `data` is a symlink and so is one
    # directory inside it:
    #     data                     -> /home/kyzen/metafind_data
    #     data/outputs/checkpoints -> /home/kyzen/metafind_out/checkpoints
    # `findmnt -T` puts BOTH roots on /dev/nvme0n1p2 (ext4). So a path written
    # `data/outputs/checkpoints/...` and one written
    # `/home/kyzen/metafind_out/checkpoints/...` are the same bytes, and a
    # reviewer quoting either is quoting the same file. This was mistaken for
    # two divergent trees once. `find` does NOT follow a symlinked start point
    # and returns the empty set silently: use `find -L`.
    #
    # [ULIP2 REVIEWER 2026-08-30, MINOR] The canonical
    # `data/outputs/checkpoints/stage1_ckpt.json` -- OBSERVED DATA, read
    # 2026-08-30 -- carries NONE of `arm_config_hash`,
    # `base_hyperparameter_sha256`, `runtime_source_sha256` or
    # `checkpoint_schema`. It has `config_hash` instead, which is a different
    # field with a different definition, so it is deliberately NOT substituted
    # here. Consequences, both intended:
    #
    #   * the provenance block records four nulls. That is correct: the record
    #     does not know its arm, and writing "unknown" as if it were a value, or
    #     silently reusing `config_hash`, would put a false identity in a file
    #     whose purpose is identity.
    #   * the default out-dir becomes `unknown_ep24_none`. The word `unknown` is
    #     load-bearing -- ⚠ **an evaluation in a directory named `unknown_*` has
    #     no arm identity**, so it cannot be compared with an arm from the sweep,
    #     which does write `arm_config_hash`. Give `--out-dir` a name that says
    #     what was scored, or point `--ckpt-record` at a sweep record.
    #
    # The artifact is NOT edited to fix this: a checkpoint record is written by
    # the run that produced the checkpoint, and back-filling identity fields
    # after the fact invents the very provenance they exist to carry.
    arm = ("untrained" if untrained
           else ckpt.get("arm_config_hash", "unknown")[:12])
    suffix = f"_seed{args.init_seed}" if untrained else ""
    out = paths.OUTPUTS / "eval" / (
        args.out_dir or f"{arm}{suffix}_ep{ckpt.get('epoch')}_{args.control}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"refusing to start: {out} is not empty. Give --out-dir a fresh "
            "name; an evaluation overwritten in place cannot be compared with "
            "the one it replaced.")
    out.mkdir(parents=True, exist_ok=True)

    # [FIX 2026-08-30] n15 was the only implemented node that wrote neither
    # `run_progress` nor `cost_ledger`, though the registry declares both and
    # `graph_spec.yaml` lists them under `writers: [ALL_NODES]`. The record is
    # not decoration: a protocol-B pass is the longest single node in the
    # graph, and one that died left nothing behind saying it had ever started.
    started = time.time()
    with runlog.run_progress(NODE):
        # [FIXED 2026-09-01] This was the literal `"point_encoder_and_fuser"`,
        # the same hardcode the trainer carried. It decides which parameters
        # come back with `requires_grad`, and `load_stage1_checkpoint` refuses
        # any trainable parameter its section does not cover -- so a
        # `fuser_only` checkpoint, whose `backbone_trainable_state` is correctly
        # empty, failed to load against a backbone built as if the point encoder
        # had trained. The scope the run actually used is on the record, so read
        # it. An untrained control has no record and keeps the default; nothing
        # is restored into it anyway.
        scope = (ckpt or {}).get("train_scope", "point_encoder_and_fuser")
        backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                               train_scope=scope))
        if untrained:
            # Seeded HERE, immediately before `build_model`, because that call is
            # what draws the towers' weights out of the global torch RNG. The
            # backbone above is loaded from a file and draws nothing, so its
            # position relative to this line does not matter -- stated because
            # "seed at the top of main" is the habit that makes it matter later.
            import torch
            torch.manual_seed(args.init_seed)
        stage2 = None
        if args.stage2_ckpt_record:
            if untrained:
                raise SystemExit("--stage2-ckpt-record needs a real Stage 1 parent, "
                                 "not --ckpt-record none")
            stage2 = load_stage2_over_stage1(args.stage2_ckpt_record, ckpt,
                                             variant=args.stage2_variant)
            model, loss_fn = stage2["model"], stage2["loss_fn"]
            model.to(args.device)
            load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                                   new_prefixes=("query.layout_encoder",
                                                 "query.layout_weight"))
            overlay_stage2_weights(model, stage2["record"], args.device)
        else:
            model, loss_fn = build_model(encoding, training, hyperparameters)
            model.to(args.device)
            if not untrained:
                load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]))

        provenance = {
            "run_id": runlog.run_id(),
            "code_revision": runlog.code_revision(),
            "code_dirty": runlog.code_dirty(),
            "runtime_source_sha256": runlog.runtime_source_sha256(),
            "runtime_source_status": runlog.runtime_source_status(),
            "started_at": time.time(),
            "query_image_policy": image_policy,
            # resolved again (and enforced) where the pack is built, below
            "query_pc_perturb": (args.query_pc_perturb
                                 or ckpt.get("query_pc_perturb") or "none"),
            "image_tokens": image_tokens,
            # None for the w/o ESSGNN row. For w/ ESSGNN: which Stage 2 run laid
            # its query-side weights over this Stage 1 parent, and with what
            # lambda -- the layout vector itself is never computed here.
            "stage2_checkpoint": (None if stage2 is None else
                                  {k: stage2["record"].get(k) for k in
                                   ("uri", "sha256", "variant_id",
                                    "stage1_checkpoint_sha256", "effective_values",
                                    "lambda_init", "steps", "code_revision")}),
            "checkpoint": {k: ckpt.get(k) for k in
                           ("uri", "sha256", "epoch", "run_id", "seed",
                            "arm_config_hash", "base_hyperparameter_sha256",
                            "code_revision", "checkpoint_schema", "phase",
                            "query_observation", "query_construction")},
            "control": args.control,
            "unsealed": bool(args.unseal),
            # The development rule opens the test split once, on a --phase
            # final checkpoint. A sealed read on a dev-phase checkpoint is a
            # diagnostic, and the table must say so on its face rather than
            # leave it to whoever remembers which checkpoint this was.
            "sealed_read_on_nonfinal_checkpoint": bool(
                any(seals.values()) and not untrained
                and ckpt.get("phase") != "final"),
            "splits": _splits_identity(),
            "device": args.device,
            "ckpt_record": "none" if untrained else str(
                args.ckpt_record or paths.CHECKPOINTS / "stage1_ckpt.json"),
            "untrained": untrained,
            "init_seed": args.init_seed,
        }
        if untrained:
            # ⚠ The single sentence this whole control turns on. "Untrained"
            # here means NO STAGE 1 TRAINING. It does not mean "no pretraining",
            # and stating it one step wider is the error this run exists to
            # avoid making.
            #
            # OBSERVED IMPLEMENTATION (`models/ulip_backbone.py:52-53`, 348-367):
            # the ULIP-2 checkpoint supplies `point_encoder` (226 tensors),
            # `pc_projection` and `logit_scale`, and `ULIPBackbone.__init__`
            # loads them regardless of this flag. Text and image go through the
            # same pretrained OpenCLIP ViT-bigG-14. So of everything that
            # produces an embedding on this run, ONLY the two fusion towers are
            # random.
            provenance["untrained_caveat"] = (
                "Stage 1 is UNTRAINED: no checkpoint record was read and no "
                "Stage 1 weights were loaded, so both fusion towers are at "
                "random initialisation from torch seed "
                f"{args.init_seed}. This is NOT a zero-pretraining baseline -- "
                "the point encoder still carries ULIP-2's pretrained PointBERT "
                "and pc_projection, and text/image still go through pretrained "
                "OpenCLIP ViT-bigG-14. Any score here is what the pretrained "
                "encoders plus two random fusions achieve. The two towers are "
                "two independent draws, so one seed is one sample: run several "
                "before reading a difference into it.")

        # Built ONCE, before the protocol loop, so every protocol in one run
        # shares one construction. Per-protocol packs would let two rows of the
        # same table be produced by two different query constructions.
        query_pack = None
        if args.query_pack:
            from metafind.train.stage1 import QueryPack, protocol_n_views

            # `n_views` comes from the SAME encoding artifact the trainer read.
            # [BLOCKER, ULIP2 Reviewer 2026-09-03] This call was left at one
            # argument when `QueryPack` gained the parameter, so `--query-pack`
            # raised TypeError here -- after the backbone and checkpoint had
            # loaded, on the one path that turns a checkpoint into a Table 1
            # number. A pack could be trained with and not scored with, which
            # is precisely the split QueryPack's own docstring exists to
            # prevent.
            query_pack = QueryPack(args.query_pack, protocol_n_views(encoding))
            print(f"query pack {query_pack.path} arms={list(query_pack.arms)} "
                  f"sha256={query_pack.sha256[:12]}", flush=True)
        pc_perturb = args.query_pc_perturb or ckpt.get("query_pc_perturb") or "none"
        if pc_perturb != "none":
            if query_pack is None or "pc" not in query_pack.arms:
                raise SystemExit(
                    f"the checkpoint was trained with query pc perturbation "
                    f"{pc_perturb!r} but no --query-pack with a pc arm was given; "
                    "scoring it on the gallery's own cloud would be a different "
                    "construction from the one it was trained under")
            query_pack.pc_perturb = pc_perturb
            print(f"query pc perturbation: {pc_perturb}"
                  f"{'' if args.query_pc_perturb else ' (from the checkpoint record)'}",
                  flush=True)

        exclude_uids = (degraded_render_uids()
                        if args.exclude_degraded_renders else None)

        if exclude_uids:
            print(f"excluding {len(exclude_uids):,} assets with a degraded "
                  "render [MASTER 2026-09-03; the corpus is unchanged, this is "
                  "an evaluation-time sensitivity axis]", flush=True)

        results = {}
        for name in wanted:
            print(f"\n=== {name} ===", flush=True)
            core, rows = run_protocol(
                name, protocols[name], splits, backbone, model,
                encoding["image_aggregation"], args.device, args.batch_size,
                args.control, args.seed, args.block, untrained, ckpt,
                query_pack, exclude_uids,
                **_construction_kwargs(observation, image_tokens))
            # From the protocol's FIELDS, never its name. See protocol_caveat:
            # the name lookup that used to be here gave any protocol the
            # artifact adds the "never reported" caveat, printed beside a
            # reported number. `gallery_source` is passed rather than
            # recomputed, so the sentence and the field cannot disagree.
            core["caveat"] = protocol_caveat(
                protocols[name], splits, core.get("gallery_source"),
                # From the result the run just produced, not from the flag, so
                # the sentence and the `degraded_renders_excluded` field in the
                # same artifact cannot disagree -- the reason `gallery_source`
                # is passed rather than recomputed, one line above.
                (core.get("degraded_renders_excluded") or {}).get("gallery", 0))
            core["sealed_split_read"] = seals[name]
            results[name] = core

            with (out / f"per_query_{name}.jsonl").open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            for cond, cell in core["conditions"].items():
                print(f"  {cond:12s} R@1 {cell['R@1']:.4f}  R@5 {cell['R@5']:.4f}  "
                      f"margin p5 "
                      f"{cell['diagnostics']['signed_target_margin'].get('p5', 0):+.4f}",
                      flush=True)

        # Two artifacts, not one. [ULIP2 REVIEWER 2026-08-30] An aggregator that
        # finds `signed_target_margin` beside `R@1` will eventually report one as
        # the other. `table1.json` is the `retrieval_metrics` channel n20 and
        # n21 read; `diagnostics.json` is not that channel and no aggregator
        # should key off it.
        # DEEP copy. [ULIP2 REVIEWER MINOR 5, 2026-09-03] The dict comprehension
        # is shallow, so `table[...]["conditions"]` was the SAME OBJECT as
        # `results[...]["conditions"]`, and `cell.pop("diagnostics")` below
        # stripped it from `results` too. `diagnostics.json`, whose entire
        # purpose is to hold the block `table1.json` must not carry, was then
        # written with that block already removed. Recoverable from the
        # per-query rows, which is why it went unnoticed.
        table = {"provenance": provenance,
                 "protocols": copy.deepcopy(
                     {n: {k: v for k, v in r.items() if k != "embedding_health"}
                      for n, r in results.items()})}
        for r in table["protocols"].values():
            for cell in r["conditions"].values():
                cell.pop("diagnostics", None)
        (out / "table1.json").write_text(json.dumps(table, indent=1))
        (out / "diagnostics.json").write_text(json.dumps(
            {"provenance": provenance, "protocols": results}, indent=1))
        print(f"\nwrote {out}/table1.json and diagnostics.json", flush=True)

    # `cost_ledger` merges by numeric_add, so a re-run of one protocol adds to
    # the total rather than replacing it. Comparisons, not just wallclock: the
    # cost of this node is set by gallery size x conditions, and protocol B is
    # three orders of magnitude bigger than protocol C at the same wallclock
    # budget per query.
    runlog.cost_ledger(
        wallclock_s=round(time.time() - started, 1),
        protocols_run=len(results),
        queries_scored=sum(r["n_query"] * len(r["conditions"])
                           for r in results.values()),
        gallery_comparisons=sum(r["n_query"] * r["n_gallery"] * len(r["conditions"])
                                for r in results.values()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
