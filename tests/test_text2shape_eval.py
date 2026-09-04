"""Text2Shape's verbatim metrics agree with ours where they must, and differ where upstream differs."""
from __future__ import annotations

import numpy as np

from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.eval.text2shape_eval import text2shape_metrics


def test_rr_at_k_equals_our_recall_at_k_on_unit_vectors_without_ties():
    rng = np.random.default_rng(0)
    g = normalize_for_scoring(rng.normal(size=(300, 16)))
    q = normalize_for_scoring(g + 0.8 * rng.normal(size=g.shape))
    t = np.arange(300)
    sim = q @ g.T
    ours1, ours5 = recall_at_k(sim, t, 1), recall_at_k(sim, t, 5)
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
