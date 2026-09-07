"""Multi-positive metrics checked against independently computed full rankings."""
from __future__ import annotations

import math

import numpy as np
import pytest

from metafind.eval import intent_metrics
from metafind.eval.intent_metrics import TIE_RULE, evaluate_rankings


def _reference(query, gallery, gallery_uids, query_ids, relevant_uids):
    """Small brute-force oracle: Python scalar cosine and complete sorting."""
    records = []
    for query_id, vector, relevant in zip(query_ids, query, relevant_uids):
        qnorm = math.sqrt(math.fsum(float(x) ** 2 for x in vector))
        scored = []
        for uid, candidate in zip(gallery_uids, gallery):
            gnorm = math.sqrt(math.fsum(float(x) ** 2 for x in candidate))
            score = math.fsum(float(x) * float(y) for x, y in zip(vector, candidate)) / (qnorm * gnorm)
            scored.append((score, uid))
        top = [uid for _, uid in sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:5]]
        count_1 = int(top[0] in relevant)
        count_5 = len(set(top) & set(relevant))
        records.append({"query_id": query_id, "top5_uids": top, "n_relevant": len(relevant),
                        "hit_at_1": bool(count_1), "hit_at_5": bool(count_5),
                        "recall_at_1": count_1 / len(relevant),
                        "recall_at_5": count_5 / len(relevant)})
    metrics = {name: sum(row[key] for row in records) / len(records) * 100
               for name, key in (("Hit@1", "hit_at_1"), ("Hit@5", "hit_at_5"),
                                 ("Recall@1", "recall_at_1"), ("Recall@5", "recall_at_5"))}
    return records, metrics


def test_unequal_positive_counts_distinguish_hit_recall_and_macro_denominators():
    # Both queries rank a..f. Query 1 finds one of two positives, query 2
    # finds all of its one positive. Hit@1 is 100%, macro Recall@1 is 75%,
    # whereas pooled recall would be 2/3. The sixth positive cannot enter top 5.
    angles = np.linspace(0, 1, 6)
    gallery = np.column_stack((np.cos(angles), np.sin(angles)))
    result = evaluate_rankings(np.array([[1., 0.], [1., 0.]]), gallery,
                               gallery_uids=list("abcdef"), query_ids=["two", "one"],
                               relevant_uids=[["a", "f"], ["a"]], block=2)
    assert result["metrics"] == {"Hit@1": 100., "Hit@5": 100., "Recall@1": 75., "Recall@5": 75.}
    assert result["records"][0] == {"query_id": "two", "top5_uids": list("abcde"),
                                    "n_relevant": 2, "hit_at_1": True, "hit_at_5": True,
                                    "recall_at_1": .5, "recall_at_5": .5}
    assert result["tie_rule"] == TIE_RULE


@pytest.mark.parametrize("block", [1, 2, 3, 7, 50])
def test_top_five_and_tail_match_independent_full_cosine_ranking(block):
    rng = np.random.default_rng(234)
    gallery = rng.normal(size=(19, 11))
    query = rng.normal(size=(7, 11))
    uids = [f"asset_{i:02}" for i in rng.permutation(len(gallery))]
    query_ids = [f"intent_{i}" for i in range(len(query))]
    labels = [[uids[i], uids[-i - 1]] if i % 2 else [uids[i]] for i in range(len(query))]
    expected_records, expected_metrics = _reference(query, gallery, uids, query_ids, labels)
    result = evaluate_rankings(query, gallery, gallery_uids=uids, query_ids=query_ids,
                               relevant_uids=labels, block=block)
    assert result["records"] == expected_records
    assert result["metrics"] == pytest.approx(expected_metrics)


@pytest.mark.parametrize("block", [1, 3, 8, 4096])
def test_tied_scores_follow_uid_order_after_gallery_shuffle_and_relabeling(block):
    # Nontrivial identical vectors exercise the row reduction, not just exact
    # integer products. The label is last lexically and receives no tie priority.
    rng = np.random.default_rng(31)
    vector = rng.normal(size=1280)
    uids = ["z", "a", "f", "c", "b", "e", "d", "g"]
    gallery = np.tile(vector, (len(uids), 1))
    query = vector[None, :]
    for order in (np.arange(len(uids)), rng.permutation(len(uids))):
        ordered_uids = [uids[i] for i in order]
        result = evaluate_rankings(query, gallery[order], gallery_uids=ordered_uids,
                                   query_ids=["intent"], relevant_uids=[["z"]], block=block)
        assert result["records"][0]["top5_uids"] == list("abcde")
        assert result["metrics"] == dict.fromkeys(("Hit@1", "Hit@5", "Recall@1", "Recall@5"), 0.)
        relabeled = evaluate_rankings(query, gallery[order], gallery_uids=ordered_uids,
                                     query_ids=["intent"], relevant_uids=[["a", "b"]], block=block)
        assert relabeled["records"][0]["top5_uids"] == result["records"][0]["top5_uids"]
        assert relabeled["metrics"] == {"Hit@1": 100., "Hit@5": 100., "Recall@1": 50., "Recall@5": 100.}


def test_more_than_five_positives_and_small_gallery_keep_full_recall_denominator():
    large = evaluate_rankings(np.ones((1, 2)), np.ones((8, 2)), gallery_uids=list("abcdefgh"),
                              query_ids=["q"], relevant_uids=[list("abcdefgh")])
    assert large["metrics"] == {"Hit@1": 100., "Hit@5": 100., "Recall@1": 12.5, "Recall@5": 62.5}
    small = evaluate_rankings(np.ones((1, 2)), np.ones((3, 2)), gallery_uids=list("cba"),
                              query_ids=["q"], relevant_uids=[["c", "b"]])
    assert small["records"][0]["top5_uids"] == list("abc")
    assert small["metrics"] == {"Hit@1": 0., "Hit@5": 100., "Recall@1": 0., "Recall@5": 100.}


def test_single_positive_reduces_to_top_k_success_with_non_diagonal_targets():
    gallery = np.eye(6)
    query = gallery[[5, 2]]
    result = evaluate_rankings(query, gallery, gallery_uids=list("abcdef"), query_ids=["first", "second"],
                               relevant_uids=[["f"], ["d"]], block=2)
    assert result["records"][0]["top5_uids"][0] == "f"
    assert result["metrics"] == {"Hit@1": 50., "Hit@5": 100., "Recall@1": 50., "Recall@5": 100.}
    for row in result["records"]:
        assert row["recall_at_1"] == float(row["hit_at_1"])
        assert row["recall_at_5"] == float(row["hit_at_5"])


def test_cosine_is_scale_invariant_and_inputs_are_not_mutated():
    query = np.array([[1., 0.]])
    gallery = np.array([[1., .1], [3., 3.], [0., 1.]])
    original_query, original_gallery = query.copy(), gallery.copy()
    kwargs = dict(gallery_uids=["best_cosine", "best_dot", "other"], query_ids=["q"],
                  relevant_uids=[["best_cosine"]])
    expected = evaluate_rankings(query, gallery, **kwargs)
    actual = evaluate_rankings(query * 1e-300, gallery * np.array([[1e300], [1e-300], [7.]]), **kwargs)
    assert actual == expected
    assert actual["records"][0]["top5_uids"][0] == "best_cosine"
    np.testing.assert_array_equal(query, original_query)
    np.testing.assert_array_equal(gallery, original_gallery)


def test_scoring_memory_is_bounded_by_one_query_and_gallery_block(monkeypatch):
    original = intent_metrics._cosine_scores
    shapes = []

    def bounded(query, gallery_block):
        assert query.shape == (3,)
        assert 1 <= len(gallery_block) <= 4
        assert query.dtype == gallery_block.dtype == np.float64
        shapes.append(len(gallery_block))
        return original(query, gallery_block)

    monkeypatch.setattr(intent_metrics, "_cosine_scores", bounded)
    evaluate_rankings(np.ones((3, 3)), np.ones((11, 3)), gallery_uids=[str(i) for i in range(11)],
                      query_ids=["a", "b", "c"], relevant_uids=[["0"], ["1"], ["2"]], block=4)
    assert shapes == [4, 4, 3] * 3


@pytest.mark.parametrize("change", [
    "empty_query", "empty_gallery", "no_query_ids", "no_gallery_uids", "duplicate_query",
    "duplicate_gallery", "empty_id", "whitespace_id", "nonstring_id", "string_instead_of_ids",
    "positive_empty", "positive_duplicate", "positive_unknown", "positive_not_list", "positive_count",
    "positive_outer_not_list", "query_count", "gallery_count", "query_1d", "gallery_3d", "zero_width",
    "width_mismatch", "query_zero", "gallery_zero", "query_nan", "gallery_inf", "complex", "string_vectors",
    "block_zero", "block_negative", "block_float", "block_bool",
])
def test_invalid_inputs_fail_before_any_scoring(monkeypatch, change):
    q, g = np.ones((2, 3)), np.ones((6, 3))
    kwargs = dict(gallery_uids=list("abcdef"), query_ids=["q1", "q2"], relevant_uids=[["a"], ["b"]])
    if change == "empty_query": q, kwargs["query_ids"], kwargs["relevant_uids"] = q[:0], [], []
    elif change == "empty_gallery": g, kwargs["gallery_uids"] = g[:0], []
    elif change == "no_query_ids": kwargs["query_ids"] = []
    elif change == "no_gallery_uids": kwargs["gallery_uids"] = []
    elif change == "duplicate_query": kwargs["query_ids"] = ["q1", "q1"]
    elif change == "duplicate_gallery": kwargs["gallery_uids"][-1] = "a"
    elif change == "empty_id": kwargs["gallery_uids"][0] = ""
    elif change == "whitespace_id": kwargs["query_ids"][0] = " q1"
    elif change == "nonstring_id": kwargs["gallery_uids"][0] = 1
    elif change == "string_instead_of_ids": kwargs["gallery_uids"] = "abcdef"
    elif change == "positive_empty": kwargs["relevant_uids"][0] = []
    elif change == "positive_duplicate": kwargs["relevant_uids"][0] = ["a", "a"]
    elif change == "positive_unknown": kwargs["relevant_uids"][0] = ["missing"]
    elif change == "positive_not_list": kwargs["relevant_uids"][0] = "a"
    elif change == "positive_count": kwargs["relevant_uids"] = [["a"]]
    elif change == "positive_outer_not_list": kwargs["relevant_uids"] = {"q1": ["a"], "q2": ["b"]}
    elif change == "query_count": q = q[:1]
    elif change == "gallery_count": g = g[:5]
    elif change == "query_1d": q = q[0]
    elif change == "gallery_3d": g = g[..., None]
    elif change == "zero_width": q, g = q[:, :0], g[:, :0]
    elif change == "width_mismatch": g = g[:, :2]
    elif change == "query_zero": q[0] = 0
    elif change == "gallery_zero": g[0] = 0
    elif change == "query_nan": q[0, 0] = np.nan
    elif change == "gallery_inf": g[0, 0] = np.inf
    elif change == "complex": q = q.astype(complex)
    elif change == "string_vectors": g = g.astype(str)
    elif change == "block_zero": kwargs["block"] = 0
    elif change == "block_negative": kwargs["block"] = -2
    elif change == "block_float": kwargs["block"] = 1.5
    elif change == "block_bool": kwargs["block"] = True

    def must_not_score(*args):
        pytest.fail("invalid inputs reached scoring")

    monkeypatch.setattr(intent_metrics, "_cosine_scores", must_not_score)
    with pytest.raises(ValueError):
        evaluate_rankings(q, g, **kwargs)
