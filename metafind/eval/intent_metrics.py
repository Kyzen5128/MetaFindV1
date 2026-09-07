"""Multi-positive intent retrieval over one explicitly judged gallery.

IMPLEMENTATION CHOICE: cosine scores use float64. Each query has one ranking,
ordered by descending score and then ascending gallery UID (Python's Unicode
lexical order). Relevance labels never participate in tie breaking. This is
different from the pessimistic exact-target tie policy of the Table 1 scorer.

The caller supplies one modality condition and must establish that the frozen
gallery is fully judged. Vectors and positive UID lists alone cannot establish
annotation completeness or input provenance; this module does not infer either.
"""
from __future__ import annotations

import numpy as np


TIE_RULE = "cosine_float64_descending_then_gallery_uid_lexical_ascending"


def _validate_ids(values: list[str], name: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of identifiers")
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in values):
        raise ValueError(f"{name} must contain nonempty strings without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains repeated identifiers")


def _normalise_vectors(vectors: np.ndarray, count: int, name: str) -> np.ndarray:
    raw = np.asarray(vectors)
    if raw.ndim != 2 or raw.shape[0] != count or raw.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({count}, positive embedding dimension)")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    values = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    # Scaling each row first avoids overflow/underflow in the norm of otherwise
    # finite, nonzero inputs. The division also avoids mutating caller arrays.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f"{name} contains a zero-norm vector")
    values = values / scale
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _cosine_scores(query: np.ndarray, gallery_block: np.ndarray) -> np.ndarray:
    # A fixed row-wise reduction, rather than a shape-dependent BLAS GEMM,
    # gives identical vectors the same score after gallery reordering/blocking.
    return np.einsum("ij,j->i", gallery_block, query, optimize=False)


def evaluate_rankings(query_vectors: np.ndarray, gallery_vectors: np.ndarray, *,
                      gallery_uids: list[str], query_ids: list[str],
                      relevant_uids: list[list[str]], block: int = 4096) -> dict:
    """Return macro-averaged Hit/Recall percentages and per-query evidence.

    Per-query hits are booleans and recalls are fractions in [0, 1]. Recall@k
    divides the number of relevant retrieved assets by *all* relevant assets
    for that query, even when there are more than k positives. Hit@k only asks
    whether at least one positive was retrieved. Every query has equal weight
    in each aggregate; this is not a pooled (micro) recall.

    When the gallery contains fewer than five assets, top 5 contains the whole
    gallery. Scores are computed for one query and at most ``block`` gallery
    rows at a time. Only five candidates survive between blocks; no query by
    gallery score matrix is allocated. Exact computed score ties use TIE_RULE.
    """
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError("block must be a positive integer")
    _validate_ids(gallery_uids, "gallery_uids")
    _validate_ids(query_ids, "query_ids")
    if not isinstance(relevant_uids, list) or len(relevant_uids) != len(query_ids):
        raise ValueError("relevant_uids must provide one positive UID list per query")
    gallery_set = set(gallery_uids)
    positives = []
    for query_id, uids in zip(query_ids, relevant_uids):
        _validate_ids(uids, f"relevant_uids[{query_id!r}]")
        if not set(uids) <= gallery_set:
            raise ValueError(f"relevant_uids[{query_id!r}] contains an asset outside the gallery")
        positives.append(set(uids))
    query = _normalise_vectors(query_vectors, len(query_ids), "query_vectors")
    gallery = _normalise_vectors(gallery_vectors, len(gallery_uids), "gallery_vectors")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")

    lexical_rank = np.empty(len(gallery_uids), dtype=np.int64)
    for rank, index in enumerate(sorted(range(len(gallery_uids)), key=gallery_uids.__getitem__)):
        lexical_rank[index] = rank
    keep = min(5, len(gallery_uids))
    records = []
    for query_id, vector, relevant in zip(query_ids, query, positives):
        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float64)
        for start in range(0, len(gallery_uids), block):
            stop = min(start + block, len(gallery_uids))
            scores = _cosine_scores(vector, gallery[start:stop])
            indices = np.concatenate((best_indices, np.arange(start, stop, dtype=np.int64)))
            scores = np.concatenate((best_scores, scores))
            order = np.lexsort((lexical_rank[indices], -scores))[:keep]
            best_indices, best_scores = indices[order], scores[order]
        top_uids = [gallery_uids[index] for index in best_indices]
        hits_1 = int(top_uids[0] in relevant)
        hits_5 = len(set(top_uids) & relevant)
        records.append({
            "query_id": query_id,
            "top5_uids": top_uids,
            "n_relevant": len(relevant),
            "hit_at_1": bool(hits_1),
            "hit_at_5": bool(hits_5),
            "recall_at_1": hits_1 / len(relevant),
            "recall_at_5": hits_5 / len(relevant),
        })
    metrics = {name: float(np.mean([record[key] for record in records]) * 100.0)
               for name, key in (("Hit@1", "hit_at_1"), ("Hit@5", "hit_at_5"),
                                 ("Recall@1", "recall_at_1"), ("Recall@5", "recall_at_5"))}
    return {"metrics": metrics, "records": records, "tie_rule": TIE_RULE}
