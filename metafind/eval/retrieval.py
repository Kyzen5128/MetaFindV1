"""Table 1: instance-level retrieval, 7 query conditions x the gallery protocols.

# IMPLEMENTS-NODE: n15_eval_retrieval

Written from scratch, as `docs/graph/node_registry.yaml:903` says it must be:
ULIP's `test_zeroshot_3d_core` scores zero-shot CLASSIFICATION over the 1,156
LVIS category names (`upstream/ULIP/main.py:350-428`), and OpenShape's
`src/train.py:215-235` does the same. Neither ranks an asset against a gallery
of assets, so there is no upstream scorer to inherit -- only the vocabulary.

What the paper fixes
--------------------
[PAPER 3experiments.tex:24] Seven conditions: text, image, pc, text+image,
text+pc, image+pc, and full. [PAPER :18] R@1 and R@5. The correct answer is the
query's own asset, retrieved across modalities -- 3.1's "pre-encoded asset
database" ranked against a query built from a subset of that same asset.

What the paper does not fix
---------------------------
[U-09] Whether the gallery is the 20% test split or all 46,052 assets. The
difference moves R@1 substantially and the paper never says, so BOTH protocols
run and both are reported (`metafind/data/splits.py`). Trying to back-solve it
from the baselines' 98-99% PC-only figure is impossible: under PC-only the query
embedding is the gallery entry's own embedding, so self-retrieval approaches
100% at either size and the number carries no information about the denominator.

[U-09, widened] The QUERY set is unstated too. Both reported protocols assume
query = test; that assumption is recorded here rather than deduced.

This module holds the SCORER. The masks are deterministic -- one fixed
combination per condition -- and share nothing with `sample_modality_mask`,
which draws sec. 2.6's stochastic 30% training mask. Reusing that here would
evaluate a random subset per query and call it "text-only".
"""

from __future__ import annotations

import numpy as np

from metafind.models.fusion import MODALITIES

__all__ = [
    "QUERY_CONDITIONS",
    "condition_mask",
    "normalize_for_scoring",
    "recall_at_k",
    "rank_of_target",
]

# [PAPER 3experiments.tex:24] The seven, in the paper's own order. Keys are the
# labels Table 1 prints; values are presence flags in `MODALITIES` order
# ("text", "image", "pc").
QUERY_CONDITIONS: dict[str, tuple[bool, bool, bool]] = {
    "text":        (True,  False, False),
    "image":       (False, True,  False),
    "pc":          (False, False, True),
    "text+image":  (True,  True,  False),
    "text+pc":     (True,  False, True),
    "image+pc":    (False, True,  True),
    "full":        (True,  True,  True),
}


def normalize_for_scoring(embeddings: np.ndarray) -> np.ndarray:
    """Return one shared float64, L2-normalised retrieval representation.

    Stage 1 checkpoint selection, n15, and the dtype comparison harness must not
    each implement a slightly different meaning of "float64 scoring".  Before
    this helper existed they did: Torch-f64 normalisation/GEMM, Torch-f32
    normalisation followed by NumPy-f64 GEMM, and NumPy-f64 throughout.  The
    measurement in ``tools/measure_dtype_effect.py`` exercised only the last of
    those paths, so it could not validate the other two.

    Model outputs arrive as float32 CPU arrays.  Conversion happens *before*
    normalisation, and all subsequent similarity GEMMs are NumPy float64.  The
    rank/tie definition remains exact comparison; this only makes the numerical
    path explicit and shared.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain a non-finite value")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot score a zero-norm embedding")
    return x / norms


def condition_mask(condition: str, batch_size: int):
    """A ``(batch_size, 3)`` bool tensor: which modalities this condition gives.

    Deterministic on purpose. `sample_modality_mask` implements sec. 2.6's
    stochastic training mask; using it here would hand each query a random
    subset and report the result under a fixed label.
    """
    import torch

    if condition not in QUERY_CONDITIONS:
        raise ValueError(
            f"unknown query condition {condition!r}; Table 1 has "
            f"{sorted(QUERY_CONDITIONS)}")
    flags = QUERY_CONDITIONS[condition]
    return torch.tensor(flags, dtype=torch.bool).expand(batch_size, len(MODALITIES))


def rank_of_target(similarity: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each query's own asset within its gallery row.

    `similarity` is (n_query, n_gallery); `targets[i]` is the gallery COLUMN
    holding query i's asset. The column is passed in rather than assumed to be
    the diagonal, because under `B_full_gallery` the gallery is the whole corpus
    and a query's asset sits wherever the index put it -- assuming the diagonal
    there would score row i against asset i and silently measure nothing.

    Ties count AGAINST the model: a rank is the number of gallery entries
    scoring strictly higher, plus the ties, plus one. A degenerate model that
    returns identical scores everywhere would otherwise report R@1 = 100%.

    ⚠ **CORRECTED 2026-08-30. This paragraph used to claim the tie test
    "catches EXACT collapse and not NEAR collapse". That is false, and was
    measured false on the production path.**

    The tie test is bit-equality, and bit-equality does not hold even between
    gallery entries that are byte-identical, because a single BLAS `gemm` does
    not return the same last bit for every output column:

        one `q @ g[:3].T`, three IDENTICAL gallery rows, d=10
            5.938673867335661  and  5.938673867335662
        61 of 400 random collapsed galleries showed >1 distinct value
        numpy 2.4.6, scipy-openblas

    ⚠ **The failure rate depends entirely on the shape, and the first numbers
    written here were measured in a shape this project never runs.**

    In a TOY regime -- `d` between 4 and 40, embeddings NOT normalised -- the
    exact rank of a fully collapsed gallery came out wrong in 20 of 200 trials
    (the ULIP2 Block Reviewer measured 42/200 on a different draw). Those are
    the numbers that first stood here, and extrapolating them to production was
    an error the Reviewer made and then withdrew.

    At PRODUCTION shape -- `d = 1280`, L2-normalised, gallery sizes 999 / 4,569
    / 9,138 / 45,692 -- the picture is different, and the question that matters
    is whether a collapsed model can score R@1 > 0:

        float32   0 trials with R@1 > 0, out of 1,120, at every gallery size
        float64   0 at every real gallery size; fails only at ng <= 13
                  (ng=5: 14/200, ng=13: 6/200), which no protocol uses --
                  the smallest real gallery is 4,569

    **So at the shapes actually used, a totally collapsed model reports R@1 = 0
    and the tie mechanism does its job.** The docstring's original promise was
    not false in production; it was unproven, and the disproof came from a
    regime that does not occur here.

    What does survive at production shape is smaller and is a DIAGNOSTIC issue:
    in float32 the exact `rank` and `tie_count` of a collapsed gallery move with
    the caller's block size (7-9 of 12 trials); in float64 they do not (0 of
    12). `run_retrieval` therefore scores in float64. R@1 is unaffected either
    way.

    What this does and does not change:

    * **Whether any existing number moves is `UNKNOWN`.** [CODEX 2026-08-30]
      This paragraph asserted "No existing number moves", on the reasoning that
      a healthy model does not produce bit-identical scores. That reasoning is
      plausible and it is not evidence: nobody has recomputed e5, e10 or e25
      under a tolerant tie policy, so the claim is withdrawn rather than
      softened. It becomes answerable only by rerunning them.
    * **What changes is what may be CLAIMED.** "Ties count against the model, so
      a collapsed model cannot report 100%" has been used as partial reassurance
      about `full` R@1 = 1.0000. That reassurance is withdrawn: the guarantee
      does not exist. Whether `full` is saturated or collapsed is `UNKNOWN` and
      needs a negative control, not this function.

    A derived tolerance (the `gamma_d` dot-product backward error bound -- NOT
    an epsilon fitted to an observed score distribution) was proposed and
    measured to be monotonically unfavourable to the model: over 1,200 queries
    it never produced a rank BETTER than bit-equality, so it could not flatter a
    result. **STATUS: NOT ADOPTED, 2026-08-30.** Scoring in float64 closed every
    symptom actually observed at production shape -- R@1 was never affected, and
    `tie_count`'s drift with the caller's block size went from 7-9 of 12 trials
    to 0 of 12. A tolerance would additionally change the DEFINITION of three
    quantities that go into Table 1, out of proportion to a symptom that higher
    precision already removes. The monotonicity measurement above is kept so the
    option can be revisited with evidence if it is ever needed.

    ⚠ **What NOTHING here catches: NEAR collapse.** Scores differing by 1e-9 are
    not tied under bit-equality, and would not be tied under any tolerance
    derived from floating-point error either -- 1e-9 is four orders of magnitude
    above the 2.8e-13 such a bound gives at d=1280. Near-collapse is the
    commoner way a contrastive model fails, and the tie mechanism cannot see it
    at all.

    The two are not substitutes: **the tie mechanism answers EXACT collapse, the
    margin diagnostics answer NEAR collapse.** For the latter read
    `signed_target_margin` and `top1_top2_gap` quantiles from
    `run_retrieval.py`, and run a negative control.
    """
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2-D, got {similarity.shape}")
    if targets.shape != (similarity.shape[0],):
        raise ValueError(
            f"{targets.shape[0] if targets.ndim else '?'} targets for "
            f"{similarity.shape[0]} queries")
    if targets.size and (targets.min() < 0 or targets.max() >= similarity.shape[1]):
        raise ValueError("a target column is outside the gallery")

    own = similarity[np.arange(similarity.shape[0]), targets][:, None]
    higher = (similarity > own).sum(axis=1)
    tied = (similarity == own).sum(axis=1) - 1      # minus the target itself
    return higher + tied + 1


def recall_at_k(similarity: np.ndarray, targets: np.ndarray,
                ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """R@k as a fraction in [0, 1], one entry per k, plus the counts.

    Reported with `n_query` and `n_gallery` because the registry's postcondition
    for this node requires every cell to carry them, and because an R@1 without
    its denominator cannot be compared to anything -- which is the whole of U-09.
    """
    ranks = rank_of_target(similarity, targets)
    out: dict[str, float] = {}
    for k in ks:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        out[f"R@{k}"] = float((ranks <= k).mean()) if ranks.size else 0.0
    out["n_query"] = int(similarity.shape[0])
    out["n_gallery"] = int(similarity.shape[1])
    return out
