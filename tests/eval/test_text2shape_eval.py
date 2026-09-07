"""Text2Shape's verbatim metrics agree with ours where they must, and differ where upstream differs."""
from __future__ import annotations

import numpy as np
import pytest

from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.eval.text2shape_eval import text2shape_metrics
from metafind.eval import text2shape_eval


def test_rr_at_k_equals_our_recall_at_k_on_unit_vectors_without_ties():
    rng = np.random.default_rng(0)
    g = normalize_for_scoring(rng.normal(size=(300, 16)))
    q = normalize_for_scoring(g + 0.8 * rng.normal(size=g.shape))
    t = np.arange(300)
    sim = q @ g.T
    ours = recall_at_k(sim, t, (1, 5)); ours1, ours5 = ours["R@1"], ours["R@5"]
    m = text2shape_metrics(q, g, t)
    assert abs(m["RR@1"] - ours1) < 1e-12 and abs(m["RR@5"] - ours5) < 1e-12
    assert 0.0 < m["RR@1"] < 1.0          # a case where something is actually measured
    assert m["NDCG@5"] >= m["RR@1"] and m["NDCG@5"] <= m["RR@5"]


def test_raw_dot_can_rank_differently_from_cosine():
    """Upstream scores the unnormalised dot product: a long wrong gallery vector wins."""
    q = np.array([[1.0, 0.0]])
    g = np.array([[0.9, 0.1], [3.0, 3.0]])     # row 0 is the cosine match, row 1 the dot match
    t = np.array([0])
    unit = text2shape_metrics(normalize_for_scoring(q), normalize_for_scoring(g), t, n_neighbors=2)
    raw = text2shape_metrics(q, g, t, n_neighbors=2)
    assert unit["RR@1"] == 1.0 and raw["RR@1"] == 0.0


def test_query_batches_bound_dense_work_and_aggregate_all_queries(monkeypatch):
    """Uneven batches and repeated targets retain upstream's full-query means."""
    rng = np.random.default_rng(71)
    gallery = rng.normal(size=(19, 8))
    targets = np.array([1, 1, 3, 4, 6, 0, 8, 9, 9, 12, 15])
    query = gallery[targets] + rng.normal(size=(11, 8))
    expected = text2shape_metrics(query, gallery, targets, query_block_size=len(query))
    original = text2shape_eval._compute_nearest_neighbors_cosine
    calls = []

    def bounded(fit, queries, *args, **kwargs):
        calls.append(queries.shape[0])
        assert queries.shape[0] <= 4
        assert fit is gallery
        return original(fit, queries, *args, **kwargs)

    monkeypatch.setattr(text2shape_eval, "_compute_nearest_neighbors_cosine", bounded)
    actual = text2shape_metrics(query, gallery, targets, query_block_size=4)
    assert calls == [4, 4, 3]
    assert actual.pop("query_block_size") == 4
    expected.pop("query_block_size")
    assert actual == expected


@pytest.mark.parametrize("block", [1, 2, 4])
def test_query_batches_preserve_upstream_ties(block):
    # Exact integer dot products isolate upstream's partition tie policy from
    # floating-point matrix multiplication differences across batch shapes.
    query = np.ones((7, 2))
    gallery = np.ones((9, 2))
    targets = np.arange(7)
    expected = text2shape_metrics(query, gallery, targets, query_block_size=7)
    actual = text2shape_metrics(query, gallery, targets, query_block_size=block)
    for result in (expected, actual):
        result.pop("query_block_size")
    assert actual == expected


@pytest.mark.parametrize("change", ["empty", "nan", "width", "fractional_target",
                                    "negative_target", "oversize_target", "block", "neighbors"])
def test_invalid_inputs_fail_before_upstream_dense_allocation(monkeypatch, change):
    query, gallery, targets = np.ones((2, 3)), np.ones((5, 3)), np.array([0, 1])
    kwargs = {}
    if change == "empty":
        query, targets = query[:0], targets[:0]
    elif change == "nan":
        query[0, 0] = np.nan
    elif change == "width":
        gallery = gallery[:, :2]
    elif change == "fractional_target":
        targets = np.array([0., 1.5])
    elif change == "negative_target":
        targets[0] = -1
    elif change == "oversize_target":
        targets[1] = 5
    elif change == "block":
        kwargs["query_block_size"] = 0
    else:
        kwargs["n_neighbors"] = 0

    def must_not_allocate(*args, **kwargs):
        pytest.fail("invalid inputs reached dense upstream computation")

    monkeypatch.setattr(text2shape_eval, "_compute_nearest_neighbors_cosine", must_not_allocate)
    with pytest.raises(ValueError):
        text2shape_metrics(query, gallery, targets, **kwargs)
