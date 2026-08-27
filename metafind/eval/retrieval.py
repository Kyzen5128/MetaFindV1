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

    The tie test is BIT-EQUALITY, so it catches EXACT collapse and not NEAR
    collapse: cosines at 0.5 exactly are caught, cosines differing by 1e-12 are
    not, and near-collapse is the commoner way a contrastive model fails. No
    tolerance is applied deliberately -- an epsilon chosen after seeing a score
    distribution is a fitted constant, and it would hide the very case it was
    fitted to. If an R@1 looks impossibly high, the score spread is where to
    look; this function will not flag it.
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
