"""Tests for n09c_build_scene_splits.

The check that matters is L2-LEAK-SCENE: train_houses and test_houses disjoint,
exact tolerance, full sample. Its negative injection -- moving one test house
into the train set -- is exercised against the production splitter.
"""

from __future__ import annotations

import pytest

from metafind.data.scene_splits import (
    DEFAULT_SEED,
    TRAIN_FRACTION,
    split_houses,
)


def houses(n: int) -> list[str]:
    return [f"train_{i:05d}" for i in range(n)]


# --- L2-LEAK-SCENE ---------------------------------------------------------

def test_train_and_test_are_disjoint():
    """[L2-LEAK-SCENE] Exact tolerance, full sample."""
    train, test = split_houses(houses(12000), DEFAULT_SEED)
    assert set(train) & set(test) == set()


def test_moving_one_test_house_into_train_is_detectable():
    """[L2-LEAK-SCENE negative injection] leakage_count == 1.

    The injection is applied to the production splitter's OUTPUT, which is where
    a leak would actually appear -- a hand-built pair of lists would only prove
    that set intersection works.
    """
    train, test = split_houses(houses(1000), DEFAULT_SEED)
    injected_train = train + [test[0]]
    assert len(set(injected_train) & set(test)) == 1


def test_every_house_lands_in_exactly_one_split():
    all_ids = houses(1000)
    train, test = split_houses(all_ids, DEFAULT_SEED)
    assert sorted(train + test) == sorted(all_ids)
    assert len(train) + len(test) == len(all_ids)


# --- the 80/20 the paper states -------------------------------------------

@pytest.mark.parametrize("n", [10, 100, 1000, 12000])
def test_the_train_share_is_eighty_percent(n):
    """[PAPER 3.1] "We allocate 80% of the data for training"."""
    train, test = split_houses(houses(n), DEFAULT_SEED)
    assert len(train) == round(n * TRAIN_FRACTION)
    assert abs(len(train) / n - TRAIN_FRACTION) <= 0.05


def test_the_fraction_is_configurable_but_defaults_to_the_paper():
    assert TRAIN_FRACTION == 0.8
    train, _ = split_houses(houses(100), DEFAULT_SEED, train_fraction=0.5)
    assert len(train) == 50


# --- reproducibility -------------------------------------------------------

def test_the_same_seed_gives_the_same_split():
    a = split_houses(houses(1000), 42)
    b = split_houses(houses(1000), 42)
    assert a == b


def test_a_different_seed_gives_a_different_split():
    a, _ = split_houses(houses(1000), 42)
    b, _ = split_houses(houses(1000), 43)
    assert a != b


def test_input_order_does_not_change_the_split():
    """A split that depends on glob order is not reproducible by anyone else."""
    ids = houses(1000)
    forward = split_houses(ids, DEFAULT_SEED)
    reversed_ = split_houses(list(reversed(ids)), DEFAULT_SEED)
    shuffled = split_houses(ids[500:] + ids[:500], DEFAULT_SEED)
    assert forward == reversed_ == shuffled


def test_houses_from_all_three_procthor_files_can_mix():
    """[U-07] We split over all 12,000, not along ProcTHOR's own boundaries."""
    ids = ([f"train_{i:05d}" for i in range(100)]
           + [f"val_{i:05d}" for i in range(20)]
           + [f"test_{i:05d}" for i in range(20)])
    train, test = split_houses(ids, DEFAULT_SEED)
    prefixes = {h.split("_")[0] for h in test}
    # a split drawn over the union puts more than one ProcTHOR file in test;
    # if this ever holds only one, the shuffle has stopped mixing them
    assert len(prefixes) > 1


def test_an_empty_input_yields_two_empty_splits():
    assert split_houses([], DEFAULT_SEED) == ([], [])
