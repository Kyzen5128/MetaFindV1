"""Tests for n09_build_splits.

Two things matter: object-level leakage (L2-LEAK-OBJECT, gated by G3), and that
both U-09 protocols exist with DERIVED gallery sizes. The second is where an
earlier draft went wrong by hardcoding the paper's approximate 48,000 against a
manifest of 46,052.
"""

from __future__ import annotations

import pytest

from metafind.data.splits import (
    DEFAULT_FUSION,
    DEFAULT_SEED,
    DEFAULT_TOWER_SHARING,
    TRAIN_FRACTION,
    build_eval_protocols,
    build_stage1_protocol,
    split_assets,
)


def uids(n: int) -> list[str]:
    return [f"{i:032x}" for i in range(n)]


# --- L2-LEAK-OBJECT --------------------------------------------------------

def test_train_and_test_are_disjoint():
    train, test = split_assets(uids(46052), DEFAULT_SEED)
    assert set(train) & set(test) == set()


def test_moving_one_test_asset_into_train_is_detectable():
    """[L2-LEAK-OBJECT negative injection] Applied to the production splitter's
    output, which is where a leak would actually appear."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    assert len(set(train + [test[0]]) & set(test)) == 1


def test_every_asset_lands_in_exactly_one_split():
    all_ids = uids(1000)
    train, test = split_assets(all_ids, DEFAULT_SEED)
    assert sorted(train + test) == sorted(all_ids)


# --- the 80/20 -------------------------------------------------------------

@pytest.mark.parametrize("n", [10, 1000, 46052])
def test_the_train_share_is_eighty_percent(n):
    train, _ = split_assets(uids(n), DEFAULT_SEED)
    assert len(train) == round(n * TRAIN_FRACTION)


def test_input_order_does_not_change_the_split():
    ids = uids(1000)
    assert (split_assets(ids, DEFAULT_SEED)
            == split_assets(list(reversed(ids)), DEFAULT_SEED)
            == split_assets(ids[500:] + ids[:500], DEFAULT_SEED))


def test_the_same_seed_reproduces_the_split():
    assert split_assets(uids(500), 42) == split_assets(uids(500), 42)


# --- U-09: both protocols --------------------------------------------------

def test_both_gallery_protocols_are_defined():
    """[U-09] The paper never says which of the two the gallery is, and the
    difference moves R@1 substantially, so both run and both are reported."""
    train, test = split_assets(uids(46052), DEFAULT_SEED)
    p = build_eval_protocols(train, test)
    assert set(p) == {"A_test_gallery", "B_full_gallery"}


def test_gallery_sizes_are_derived_from_the_split_not_hardcoded():
    """An earlier draft hardcoded 48,000 -- the paper's approximate figure --
    while the manifest holds 46,052 (U-01), moving every denominator."""
    train, test = split_assets(uids(46052), DEFAULT_SEED)
    p = build_eval_protocols(train, test)
    assert p["A_test_gallery"]["gallery_size"] == len(test)
    assert p["B_full_gallery"]["gallery_size"] == len(train) + len(test) == 46052

    # and it tracks a different corpus rather than staying at 46,052
    t2, e2 = split_assets(uids(1000), DEFAULT_SEED)
    assert build_eval_protocols(t2, e2)["B_full_gallery"]["gallery_size"] == 1000


def test_the_two_protocols_have_different_gallery_sizes():
    """If they ever coincide the pair has stopped answering U-09."""
    train, test = split_assets(uids(46052), DEFAULT_SEED)
    p = build_eval_protocols(train, test)
    assert p["A_test_gallery"]["gallery_size"] < p["B_full_gallery"]["gallery_size"]


def test_both_protocols_query_the_test_split_and_say_so():
    """[U-09 widened] 3.1 never says Table 1's queries are the 20%. This is our
    assumption and it is recorded per protocol, not implied."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    for proto in build_eval_protocols(train, test).values():
        assert proto["query_split"] == "test"


def test_layout_free_context_is_recorded_on_every_protocol():
    """[U-28] 3.2 admits the mismatch without saying what happens to
    lambda*e_layout. Omitting it is a choice affecting 7 of Table 1's 14 cells."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    for proto in build_eval_protocols(train, test).values():
        assert proto["layout_free_context"] == "omitted"


# --- stage1_protocol -------------------------------------------------------

def hyperparameters() -> dict:
    return {"sha256": "deadbeef" * 8, "values": {}}


def test_the_protocol_carries_the_hyperparameter_hash():
    """G3 dereferences this; a protocol whose hash points nowhere is a run whose
    hyperparameters cannot be stated afterwards."""
    p = build_stage1_protocol(hyperparameters(), "tester")
    assert p["hyperparameter_config_hash"] == hyperparameters()["sha256"]
    assert p["status"] == "resolved"


def test_similarity_is_cosine_and_recorded():
    """[U-24] sim(.,.) is never defined. The loss normalises both sides
    unconditionally, so any other label here would be a lie about the numbers."""
    assert build_stage1_protocol(hyperparameters(), "tester")["similarity"] == "cosine"


def test_the_main_line_is_not_a_table3_fusion_ablation():
    """[U-13] Table 3 ablates Mean and MLPs as separate rows, so the full model
    is neither. Defaulting to one of them would run an ablation as the main
    line -- the same class of error as the earlier `train fuser only` default."""
    assert DEFAULT_FUSION not in ("mean", "mlp")


def test_the_towers_are_not_fully_shared():
    """[U-16] 2.6 requires the gallery encoder frozen while the query fuser
    trains. One module cannot satisfy both."""
    assert DEFAULT_TOWER_SHARING != "fully_shared"


def test_an_unknown_similarity_is_refused():
    import metafind.data.splits as s

    original = s.SUPPORTED_SIMILARITY
    try:
        s.SUPPORTED_SIMILARITY = ("dot_product",)
        with pytest.raises(ValueError):
            s.build_stage1_protocol(hyperparameters(), "tester")
    finally:
        s.SUPPORTED_SIMILARITY = original
