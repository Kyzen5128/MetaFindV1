"""n15 -- run Table 1's retrieval protocols and say what the numbers rest on.

# IMPLEMENTS-NODE: n15_eval_retrieval

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

Why the similarity matrix is never materialised
-----------------------------------------------
Protocol B is 9,138 queries x 45,692 gallery entries x 7 conditions = 2.9e9
scores. Everything below streams over gallery blocks: rank needs only the counts
of strictly-higher and tied entries, and every diagnostic here is an accumulator.
"""

from __future__ import annotations

import argparse
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

__all__ = ["ProtocolResult", "score_streaming", "load_protocols", "main"]

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
        masked = np.where(np.isfinite(blk), blk, 0.0)
        off_sum += masked.sum(axis=1)
        off_sq += (masked ** 2).sum(axis=1)
        ex = np.exp(sim - _SHIFT)
        exp_sum += ex.sum(axis=1)
        xexp_sum += (sim * ex).sum(axis=1)
        del full, sim, blk, ex, masked

    # Ties count AGAINST the model, as `rank_of_target` does. Both counts are
    # already over non-targets only, so there is nothing to subtract.
    rank = higher + tied + 1
    n_off = ng - 1
    off_mean = off_sum / max(n_off, 1)
    off_var = np.maximum(off_sq / max(n_off, 1) - off_mean ** 2, 0.0)
    # H = logZ - (sum x e^x)/Z, with the constant shift folded back in.
    logZ = np.log(exp_sum) + _SHIFT
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


def encode_pools(backbone, model, query_uids, gallery_uids, aggregation,
                 device, batch_size):
    """Query embeddings per condition, plus gallery embeddings, plus the map.

    The gallery is encoded ONCE and shared by all seven conditions: the gallery
    tower sees every modality regardless of what the query withheld, which is
    what makes a condition a statement about the QUERY.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.train.stage1 import Stage1Dataset, collate, modules_in_eval

    def embed(uids, conditions):
        loader = DataLoader(Stage1Dataset(uids, aggregation), batch_size=batch_size,
                            shuffle=False, collate_fn=collate, num_workers=4,
                            drop_last=False)
        gal, per_cond = [], {c: [] for c in conditions}
        with modules_in_eval(model, getattr(backbone, "model", None)), torch.no_grad():
            for i, batch in enumerate(loader):
                embeds = {"text": batch["text"].to(device),
                          "image": batch["image"].to(device),
                          "pc": backbone.encode_pc(batch["pc"].to(device))}
                n = embeds["text"].size(0)
                if not conditions:
                    gal.append(model.gallery(embeds).float().cpu())
                for cond in conditions:
                    mask = condition_mask(cond, n).to(device)
                    per_cond[cond].append(
                        model.query(embeds, present=mask).float().cpu())
                if i % 20 == 0:
                    print(f"    batch {i}", flush=True)
        return gal, per_cond

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
        """
        import torch
        return normalize_for_scoring(torch.cat(chunks).numpy())

    return {c: norm(v) for c, v in per_cond.items()}, norm(gal)


def apply_control(control: str, targets: np.ndarray, n_gallery: int,
                  seed: int) -> tuple[np.ndarray, str]:
    """Negative controls. Without one, a high score proves nothing.

    [CODEX + ULIP2 REVIEWER 2026-08-30] Every checkpoint this project holds
    reports `full` R@1 = 1.0000 while the paper reports 0.517. A mechanism for
    that is understood -- `p_mask` leaves all three modalities present in 0.7^3 =
    34.3% of steps, and with `GalleryTower` calling its fusion without `present`,
    both towers then see bit-identical input -- but a mechanism being real does
    not make it the only cause. That is an `INFERENCE`, and these turn it into a
    measurement:

    ``shuffle_targets``
        Each query is scored against somebody else's asset. A retrieval metric
        must collapse to chance (~1/n_gallery). If it does not, the number was
        never measuring retrieval.

    ``none``
        The real measurement.

    **Candidate explanations for `full` R@1 = 1.0000, and their status:**

    * **float32 rounding inflates it** -- **ELIMINATED 2026-08-30.** Rescoring
      `e25_500w` in float64 gives `full` = 1.000000000, bit-exact
      (`output/look/dtype_effect.json`). Closed by measurement, not argument.
      The first candidate for this cell to be actually closed.
    * **Both towers see identical input** -- `INFERENCE`. The mechanism is
      confirmed in code (p_mask leaves all three modalities present in
      0.7^3 = 34.3% of steps, and GalleryTower calls its fusion without
      `present`), but a mechanism being real does not make it the only cause.
      `shuffle_targets` tests it.
    * **The embedding space has collapsed** -- open. `embedding_health`'s
      uncentred effective rank addresses it; not yet run on a real checkpoint.
    * **The task is saturated at this gallery size** -- open. Needs protocol A
      or B, where the gallery is 2x and 10x larger.

    ⚠ **`exclude_target` was specified and is NOT implemented, deliberately.**
    Removing the answer and asking what is retrieved instead is already measured
    by every normal run: `hardest_non_target_score` IS the best score with the
    target excluded, and `signed_target_margin` is the gap it leaves, both per
    query. A separate mode would recompute the same numbers and report an "R@1"
    whose target does not exist -- a probe that cannot return a positive, which
    is the defect class this project has now hit six times. If the intent was
    something else, say what, and it gets built.

    The third control the Reviewer named -- scoring an INITIALISATION checkpoint
    -- needs no code: point `--ckpt-record` at an untrained checkpoint. If
    `full` is near 1.0 there, training did not cause it.
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


def run_protocol(name: str, protocol: dict, splits: dict, backbone, model,
                 aggregation: str, device: str, batch_size: int,
                 control: str, seed: int, block: int) -> tuple[dict, list]:
    """One protocol, seven conditions. Returns (core result, per-query rows)."""
    query_uids = resolve_split(splits, protocol["query_split"])
    gallery_uids = resolve_split(splits, protocol["gallery_split"])

    dupes = len(gallery_uids) - len(set(gallery_uids))
    col = {u: i for i, u in enumerate(gallery_uids)}
    missing = [u for u in query_uids if u not in col]
    if missing:
        raise ValueError(
            f"{len(missing):,} query assets are absent from the {name} gallery "
            f"(e.g. {missing[:3]}). Every query's own asset must be findable, "
            "or its rank is undefined and the metric silently measures nothing.")
    targets = np.array([col[u] for u in query_uids], dtype=np.int64)

    declared = protocol.get("gallery_size")
    if declared is not None and declared != len(gallery_uids):
        raise ValueError(
            f"{name} declares gallery_size {declared:,} but the split resolves "
            f"to {len(gallery_uids):,}. One of them is stale.")

    queries, gallery = encode_pools(backbone, model, query_uids, gallery_uids,
                                    aggregation, device, batch_size)

    eff_targets, control_used = apply_control(control, targets,
                                              len(gallery_uids), seed)

    conditions, rows = {}, []
    for cond, q in queries.items():
        r = score_streaming(q, gallery, eff_targets, block=block)
        ranks = r["rank"]
        conditions[cond] = {
            "R@1": float((ranks <= 1).mean()) if ranks.size else 0.0,
            "R@5": float((ranks <= 5).mean()) if ranks.size else 0.0,
            "hits@1": int((ranks <= 1).sum()),
            "hits@5": int((ranks <= 5).sum()),
            "n_query": int(len(query_uids)),
            "n_gallery": int(len(gallery_uids)),
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
        "duplicate_gallery_uids": dupes,
        "control": control_used,
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
                         "weights before anything is encoded.")
    ap.add_argument("--protocol", action="append", default=None,
                    help="protocol key from eval_protocols.json; repeatable. "
                         "Default: every protocol the artifact defines.")
    ap.add_argument("--unseal", action="store_true",
                    help="permit protocols that read the sealed test split. "
                         "Required for a reported result; never for development.")
    ap.add_argument("--control", default="none",
                    choices=("none", "shuffle_targets"),
                    help="negative control. shuffle_targets must collapse to "
                         "chance; if it does not, the metric is not retrieval.")
    ap.add_argument("--out-dir", default=None,
                    help="directory for this evaluation's artifacts, relative "
                         "to data/outputs/eval. Default: a name built from the "
                         "checkpoint's arm hash and the control.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block", type=int, default=4096,
                    help="gallery columns scored at once. The similarity matrix "
                         "is never materialised: protocol B is 2.9e9 scores.")
    ap.add_argument("--seed", type=int, default=20260830,
                    help="only used by --control shuffle_targets.")
    args = ap.parse_args()

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
    ckpt = load_checkpoint_record(args.ckpt_record)
    seals = {n: check_seal(n, protocols[n], args.unseal) for n in wanted}

    arm = ckpt.get("arm_config_hash", "unknown")[:12]
    out = paths.OUTPUTS / "eval" / (
        args.out_dir or f"{arm}_ep{ckpt.get('epoch')}_{args.control}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"refusing to start: {out} is not empty. Give --out-dir a fresh "
            "name; an evaluation overwritten in place cannot be compared with "
            "the one it replaced.")
    out.mkdir(parents=True, exist_ok=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model, loss_fn = build_model(encoding, training, hyperparameters)
    model.to(args.device)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]))

    provenance = {
        "run_id": runlog.run_id(),
        "code_revision": runlog.code_revision(),
        "code_dirty": runlog.code_dirty(),
        "runtime_source_sha256": runlog.runtime_source_sha256(),
        "runtime_source_status": runlog.runtime_source_status(),
        "started_at": time.time(),
        "checkpoint": {k: ckpt.get(k) for k in
                       ("uri", "sha256", "epoch", "run_id", "seed",
                        "arm_config_hash", "base_hyperparameter_sha256",
                        "code_revision", "checkpoint_schema")},
        "control": args.control,
        "unsealed": bool(args.unseal),
        "device": args.device,
    }

    results = {}
    for name in wanted:
        print(f"\n=== {name} ===", flush=True)
        core, rows = run_protocol(
            name, protocols[name], splits, backbone, model,
            encoding["image_aggregation"], args.device, args.batch_size,
            args.control, args.seed, args.block)
        # Both labels stated, symmetrically, so neither protocol looks like the
        # trustworthy one by omission.
        core["caveat"] = {
            "A_test_gallery": "the paper does not state its gallery [U-09]; "
                              "query = test is this project's assumption",
            "B_full_gallery": "the gallery contains the 36,554 training assets "
                              "as distractors; the paper does not state its "
                              "gallery [U-09]",
        }.get(name, "development protocol -- selects checkpoints, never reported")
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
    # the other.
    table = {"provenance": provenance,
             "protocols": {n: {k: v for k, v in r.items()
                               if k != "embedding_health"}
                           for n, r in results.items()}}
    for r in table["protocols"].values():
        for cell in r["conditions"].values():
            cell.pop("diagnostics", None)
    (out / "table1.json").write_text(json.dumps(table, indent=1))
    (out / "diagnostics.json").write_text(json.dumps(
        {"provenance": provenance, "protocols": results}, indent=1))
    print(f"\nwrote {out}/table1.json and diagnostics.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
