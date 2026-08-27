"""Tests for n15_eval_retrieval's scorer.

The scorer is the only part of Table 1 that can be wrong quietly: a wrong
protocol produces a number, a wrong rank produces a number, and both look like
results. Every test here fixes a number by hand rather than by running the code
under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from metafind.eval.retrieval import (
    QUERY_CONDITIONS,
    condition_mask,
    rank_of_target,
    recall_at_k,
)


# --- the seven conditions ----------------------------------------------------

def test_table_one_has_exactly_the_paper_s_seven_conditions():
    """[PAPER 3experiments.tex:24] text, image, pc, and the four combinations."""
    assert set(QUERY_CONDITIONS) == {
        "text", "image", "pc", "text+image", "text+pc", "image+pc", "full"}


def test_every_condition_gives_at_least_one_modality():
    """An all-absent query is sec. 2.6's 2.7% training edge case (U-23), not a
    Table 1 row."""
    assert all(any(flags) for flags in QUERY_CONDITIONS.values())


def test_a_condition_name_matches_the_modalities_it_switches_on():
    """The names are the table's labels; a mismatch would mislabel a whole
    column and nothing downstream would notice."""
    for name, flags in QUERY_CONDITIONS.items():
        wanted = set(name.split("+")) if name != "full" else {"text", "image", "pc"}
        got = {m for m, on in zip(("text", "image", "pc"), flags) if on}
        assert got == wanted, name


def test_the_mask_is_deterministic_not_the_training_sampler():
    """Reusing `sample_modality_mask` here would hand each query a random subset
    and report it under a fixed label."""
    a = condition_mask("text+pc", 16)
    b = condition_mask("text+pc", 16)
    assert a.shape == (16, 3)
    assert bool((a == b).all())
    assert [bool(v) for v in a[0]] == [True, False, True]


def test_an_unknown_condition_is_refused():
    with pytest.raises(ValueError, match="unknown query condition"):
        condition_mask("text+layout", 4)


# --- ranking -----------------------------------------------------------------

def test_the_target_column_is_used_not_the_diagonal():
    """Under B_full_gallery the query's asset sits wherever the index put it.

    Assuming the diagonal would score query i against asset i and measure
    nothing -- and it would still return a plausible number.
    """
    sim = np.array([[0.1, 0.9, 0.2],
                    [0.8, 0.1, 0.3]])
    assert list(rank_of_target(sim, np.array([1, 0]))) == [1, 1]
    assert list(rank_of_target(sim, np.array([0, 1]))) == [3, 3]


def test_ties_count_against_the_model():
    """A model returning identical scores everywhere must not score R@1 = 100%."""
    sim = np.ones((4, 10))
    ranks = rank_of_target(sim, np.arange(4))
    assert list(ranks) == [10, 10, 10, 10]
    assert recall_at_k(sim, np.arange(4))["R@1"] == 0.0


def test_a_perfect_model_scores_one():
    sim = np.eye(5) * 2 - 1
    m = recall_at_k(sim, np.arange(5))
    assert m["R@1"] == 1.0
    assert m["R@5"] == 1.0


def test_recall_at_five_counts_ranks_two_through_five():
    """Hand-fixed: the target is the 3rd best for every query."""
    sim = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), (4, 1))
    m = recall_at_k(sim, np.full(4, 2))
    assert m["R@1"] == 0.0
    assert m["R@5"] == 1.0


def test_every_cell_carries_its_denominators():
    """[registry:891] the postcondition -- an R@1 without n_gallery cannot be
    compared to anything, which is the whole of U-09."""
    m = recall_at_k(np.zeros((3, 40)), np.arange(3))
    assert m["n_query"] == 3
    assert m["n_gallery"] == 40


def test_a_target_outside_the_gallery_is_refused():
    """Silently clipping would score against the wrong asset."""
    with pytest.raises(ValueError, match="outside the gallery"):
        rank_of_target(np.zeros((2, 3)), np.array([0, 7]))


def test_a_target_count_that_does_not_match_the_queries_is_refused():
    with pytest.raises(ValueError, match="targets for"):
        rank_of_target(np.zeros((3, 5)), np.array([0, 1]))


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(np.zeros((2, 4)), np.arange(2), ks=(0,))
