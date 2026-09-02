"""Tests for n09_build_splits.

Two things matter: object-level leakage (L2-LEAK-OBJECT, gated by G3), and that
both U-09 protocols exist with DERIVED gallery sizes. The second is where an
earlier draft went wrong by hardcoding the paper's approximate 48,000 against a
manifest of 46,052.

The synthetic corpus below is 45,692, not 46,052, and the difference is the
point. 46,052 is the Objaverse-LVIS uid MANIFEST and is still the right
denominator inside n03/n04; 45,692 is what n05 admitted (46,024 rendered - 311
quarantined - 21 rejected, per outputs/annotation_exclusions.json) and is the
only one that ever reaches a gallery. Sizing the fixture at the live corpus
makes the derived counts below equal splits.json's real ones -- train 36,554,
test 9,138, dev_val 4,569 -- so a drift in the splitter shows up as a number
somebody can recognise instead of an abstract one.
"""

from __future__ import annotations

import pytest

from metafind.data.splits import (
    DEFAULT_DEV_SEED,
    DEFAULT_FUSION,
    DEFAULT_SEED,
    DEFAULT_TOWER_SHARING,
    DEV_VAL_FRACTION,
    TRAIN_FRACTION,
    build_eval_protocols,
    build_stage1_protocol,
    split_assets,
    split_dev,
)


# The admitted corpus, OBSERVED DATA 2026-08-30: `splits.json`'s
# `admitted_total`, the 45,692 files in `outputs/annotations/`, and the
# three-way intersection `splits.admitted_uids()` returns all agree.
CORPUS = 45692


def uids(n: int) -> list[str]:
    return [f"{i:032x}" for i in range(n)]


# --- L2-LEAK-OBJECT --------------------------------------------------------

def test_train_and_test_are_disjoint():
    train, test = split_assets(uids(CORPUS), DEFAULT_SEED)
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

@pytest.mark.parametrize("n", [10, 1000, CORPUS])
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
    train, test = split_assets(uids(CORPUS), DEFAULT_SEED)
    p = build_eval_protocols(train, test)
    assert set(p) == {"A_test_gallery", "B_full_gallery"}


def test_gallery_sizes_are_derived_from_the_split_not_hardcoded():
    """An earlier draft hardcoded 48,000 -- the paper's approximate figure --
    while the manifest held 46,052 (U-01), moving every denominator. The
    admitted corpus is 45,692, so a constant would now be wrong twice over.

    The split sizes asserted here are the ones splits.json actually holds.
    """
    train, test = split_assets(uids(CORPUS), DEFAULT_SEED)
    p = build_eval_protocols(train, test)
    assert p["A_test_gallery"]["gallery_size"] == len(test) == 9138
    assert p["B_full_gallery"]["gallery_size"] == len(train) + len(test) == 45692
    assert len(train) == 36554

    # and it tracks a different corpus rather than staying at 45,692
    t2, e2 = split_assets(uids(1000), DEFAULT_SEED)
    assert build_eval_protocols(t2, e2)["B_full_gallery"]["gallery_size"] == 1000


def test_the_two_protocols_have_different_gallery_sizes():
    """If they ever coincide the pair has stopped answering U-09."""
    train, test = split_assets(uids(CORPUS), DEFAULT_SEED)
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


# --- [D-3] dev-val, carved out of the 80% -------------------------------------
#
# The whole point of these is that the 20% test split never touches model
# selection. Each one fails if it starts to.

def test_dev_train_and_dev_val_partition_the_training_pool():
    train, _ = split_assets(uids(1000), DEFAULT_SEED)
    dev_train, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    assert set(dev_train) | set(dev_val) == set(train)
    assert not set(dev_train) & set(dev_val)


def test_dev_val_never_contains_a_test_asset():
    """[D-3] The failure this deviation exists to prevent."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    _, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    assert not set(dev_val) & set(test)


def test_moving_one_test_asset_into_dev_val_is_detectable():
    """The check above passes trivially unless it can also fail."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    _, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    assert set(dev_val + test[:1]) & set(test)


def test_the_dev_val_share_matches_the_recorded_fraction():
    train, _ = split_assets(uids(1000), DEFAULT_SEED)
    _, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    assert len(dev_val) == round(len(train) * DEV_VAL_FRACTION)


def test_the_same_dev_seed_reproduces_the_dev_split():
    train, _ = split_assets(uids(500), DEFAULT_SEED)
    a = split_dev(train, 7)
    b = split_dev(train, 7)
    c = split_dev(train, 8)
    assert a == b
    assert a != c


def test_the_selection_protocol_is_marked_not_reported():
    """[D-3] Its numbers choose lr and checkpoints; they are never a result."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    _, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    p = build_eval_protocols(train, test, dev_val)
    assert p["C_dev_selection"]["reported"] is False
    assert p["A_test_gallery"]["reported"] is True
    assert p["B_full_gallery"]["reported"] is True


def test_the_selection_gallery_is_dev_val_not_the_training_pool():
    """Ranking against 36k candidates is a different task from ranking against 9k.

    A training duration tuned against a pool an order of magnitude larger than
    the final one does not transfer, so this is a property of the protocol and
    not a detail of it.
    """
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    _, dev_val = split_dev(train, DEFAULT_DEV_SEED)
    p = build_eval_protocols(train, test, dev_val)
    assert p["C_dev_selection"]["gallery_split"] == "dev_val"
    assert p["C_dev_selection"]["gallery_size"] == len(dev_val)
    assert p["C_dev_selection"]["gallery_size"] != len(train)


def test_no_selection_protocol_without_a_dev_val():
    """Callers that pass no dev-val must not silently get a third protocol."""
    train, test = split_assets(uids(1000), DEFAULT_SEED)
    assert "C_dev_selection" not in build_eval_protocols(train, test)


def test_the_exclusion_ledger_yields_uids_not_its_own_metadata_keys():
    """The defect: `annotation_exclusions.json` was walked as `{uid: entry}`.

    It is a metadata dict, so the loop subtracted its KEYS -- 'Kyzen',
    'groups', '45692', a timestamp -- and never touched the 332 real uids under
    `groups.<name>.uids`. The corpus count stayed right by luck, because those
    332 had no annotation sidecar and the three-way intersection had already
    dropped them; what was lost is the property that the ledger is the
    authority, so restoring one sidecar would have silently readmitted an asset
    Kyzen rejected.

    Both on-disk shapes are exercised: a list of uid strings, and a list of
    dicts carrying `uid` beside the scores the rejection was decided on.
    """
    from metafind.data.splits import ledger_excluded_uids

    ledger = {
        "decided_at": "2026-08-28T14:44:25+08:00",
        "decided_by": "Kyzen",
        "decision": "delete these",
        "git_commit": "9e91457220fe625b3313d30c387364d592ee20a0",
        "corpus_before": 46024,
        "corpus_after": 45692,
        "excluded_total": 3,
        "groups": {
            "n05_quarantine": {"uids": ["aaa", "bbb"]},
            "manual_review_rejected": {"uids": [{"uid": "ccc", "clip_score": 0.1}]},
        },
    }
    got = ledger_excluded_uids(ledger)
    assert got == {"aaa", "bbb", "ccc"}
    for key in ledger:
        assert key not in got, f"the ledger's own metadata key {key!r} leaked in"


def test_the_ledger_parse_is_checked_against_the_ledgers_own_total():
    """Nine parsed entries against a stated 332 went unnoticed for six days.

    The cross-check is what turns "the loop read something" into "the loop read
    the uids", so a parse that silently reads the wrong field now fails loudly.
    """
    from metafind.data.splits import ledger_excluded_uids

    with pytest.raises(ValueError, match="excluded_total"):
        ledger_excluded_uids({"excluded_total": 332,
                              "groups": {"g": {"uids": ["only-one"]}}})

    with pytest.raises(ValueError, match="no uid"):
        ledger_excluded_uids({"groups": {"g": {"uids": [{"clip_score": 0.1}]}}})
