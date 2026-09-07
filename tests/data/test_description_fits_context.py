"""[PROMPT_VERSION 9] The description must leave the serialized string inside
CLIP's 77-token context.

v8 had no such bound. 2,095 of 45,692 records (4.59%, OBSERVED DATA 2026-08-28)
serialized to more than 77 tokens, and `encode_text_image` quarantines those --
so running the encoder as it stood would have taken the corpus to 43,597 without
a word. The loss was not uniform: 82.58% of four-placement-flag assets exceeded
the bound against 1.52% of single-flag ones, an 18x enrichment, because the
placement clause grows with the number of flags.

Every test here that asserts a guard holds is followed by one that shows the
guard can fail, because a green assertion over a bound that is never reached
proves nothing.
"""
from __future__ import annotations

import pytest

from metafind.data.annotate import (
    MAX_DESCRIPTION_WORDS,
    annotation_contract_id,
    build_description_prompt,
)
from metafind.data.annotate_run import _fit_description
from metafind.data.encode_text_image import TEXT_CONTEXT_LENGTH, true_token_count
from metafind.models.resolve_stage1 import serialize_annotation

ANCHOR = "chair"
PROPS = (1.0, 0.5, 0.5)
MODEL = "gemma-4-12B-it"

SHORT = "A wooden dining chair with a slatted back and worn green paint."
LONG = (
    "This is a low-poly, blocky three-dimensional model of a wooden dining "
    "chair with a slatted back, four turned legs, a woven rush seat, worn "
    "green paint across the arms, and a small carved rosette at the crest rail"
)


def _parsed(**over):
    obj = {
        "category": "dining chair",
        "identity_confirmed": True,
        "height": 90.0,
        "width_axis": "x",
        "mass": 6.0,
        "materials": ["wood", "fabric"],
        # All four flags: the placement clause is at its longest, which is the
        # population the v8 bound actually hit.
        "onCeiling": True, "onWall": True, "onFloor": True, "onObject": True,
    }
    obj.update(over)
    return obj


def _ranked(*texts):
    return [{"text": t, "clip_score": 1.0 - i / 100, "rank": i}
            for i, t in enumerate(texts)]


def _tokens(description, **over):
    ann, _ = _fit_description(_parsed(**over), ANCHOR, PROPS,
                              _ranked(description), MODEL)
    if ann is None:
        return None
    return true_token_count(serialize_annotation(ann.as_record(MODEL)))


# ---------------------------------------------------------------- the fixture

def test_the_long_fixture_really_does_overflow():
    """Without this, every test below could pass on a bound that never bites."""
    assert _tokens(LONG) is None, (
        "LONG no longer overflows 77 tokens, so the tests that rely on it "
        "being rejected are asserting nothing")


def test_the_short_fixture_really_does_fit():
    n = _tokens(SHORT)
    assert n is not None and n <= TEXT_CONTEXT_LENGTH


# ------------------------------------------------------------------ selection

def test_the_best_fitting_candidate_is_chosen_not_the_best_ranked():
    ann, fit = _fit_description(_parsed(), ANCHOR, PROPS,
                                _ranked(LONG, SHORT), MODEL)
    assert ann is not None
    assert ann.description == SHORT
    assert fit["rank_used"] == 1, "rank 0 overflows; rank 1 fits"
    assert fit["candidates_tried"] == 2
    assert fit["tokens"] <= TEXT_CONTEXT_LENGTH


def test_taking_the_top_ranked_unconditionally_would_have_overflowed():
    """The v8 behaviour, run on the same input, as the failing counterpart."""
    from metafind.data.annotate import validate_annotation

    v8 = validate_annotation(_parsed(), lvis_category=ANCHOR,
                             proportions=PROPS, description=LONG)
    assert true_token_count(
        serialize_annotation(v8.as_record(MODEL))) > TEXT_CONTEXT_LENGTH


def test_a_fitting_winner_is_kept_and_nothing_is_swapped():
    ann, fit = _fit_description(_parsed(), ANCHOR, PROPS,
                                _ranked(SHORT, LONG), MODEL)
    assert ann.description == SHORT
    assert fit["rank_used"] == 0 and fit["candidates_tried"] == 1


def test_no_candidate_fits_returns_none_rather_than_truncating():
    ann, fit = _fit_description(_parsed(), ANCHOR, PROPS,
                                _ranked(LONG, LONG + " and a brass ferrule"),
                                MODEL)
    assert ann is None
    assert fit["rank_used"] is None
    # `over_context`, not `tokens`: the failure record now names WHICH reason
    # each candidate failed for, so a run that quarantined on validation errors
    # cannot be reported as a run that quarantined on length.
    assert fit["over_context"] and all(n > TEXT_CONTEXT_LENGTH
                                       for n in fit["over_context"])
    assert fit["rejected_by_validator"] == []


def test_the_bound_is_checked_against_the_real_serialized_string():
    """Not against a description-only proxy. The same description fits under a
    short placement clause and overflows under the longest one -- which is why
    a single fixed description budget cannot be the gate."""
    middle = LONG[:LONG.rindex(" ", 0, 150)]
    one_flag = _tokens(middle, onCeiling=False, onWall=False, onObject=False)
    four_flag = _tokens(middle)
    assert one_flag is not None, "the short placement clause should leave room"
    assert four_flag is None, "the long placement clause should not"


# ------------------------------------------------------------------- contract

def test_the_description_prompt_is_now_fingerprinted():
    """It was not. Measured 2026-08-28: editing DESCRIPTION_PROMPT left the id
    at metafind_annot_v8@95e37eb05182d364 while bumping a dimension bound moved
    it -- and the description is the one field retrieval consumes."""
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = a.DESCRIPTION_PROMPT
    try:
        a.DESCRIPTION_PROMPT = original + "\nAnswer in rhyme."
        assert annotation_contract_id() != before
    finally:
        a.DESCRIPTION_PROMPT = original
    assert annotation_contract_id() == before


def test_moving_the_word_budget_moves_the_contract_id():
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = a.MAX_DESCRIPTION_WORDS
    try:
        a.MAX_DESCRIPTION_WORDS = original + 1
        assert annotation_contract_id() != before
    finally:
        a.MAX_DESCRIPTION_WORDS = original
    assert annotation_contract_id() == before


def test_the_prompt_states_the_budget():
    prompt = build_description_prompt(11, ANCHOR)
    assert str(MAX_DESCRIPTION_WORDS) in prompt
    assert "ONE sentence" in prompt


def test_the_corpus_split_is_visible_on_disk():
    """v8 records and v9 records must be distinguishable by a field that is
    PRESENT, not by one that is absent -- the absence rule is what created the
    provenance hazard `annotate_run` already documents.

    [FIXED 2026-08-29] This carried
    `@pytest.mark.parametrize("field", ["prompt_version", "contract_family"])`
    and the body never read `field`. It therefore ran the SAME two assertions
    twice and reported two passes -- the identical shape as
    `test_essgnn.py:259`'s `for seed in range(6)` with an unused `seed`, which
    this project has already documented once. Found by an AST scan for tests
    that cannot fail; it was written the same day the scan was written."""
    from metafind.data.annotate import PROMPT_VERSION

    assert PROMPT_VERSION == 9
    assert annotation_contract_id().startswith("metafind_annot_v9@")


# ----------------------------------------------------- validator, not just fit

def test_a_candidate_the_validator_refuses_is_skipped_not_raised():
    """[ESSGNN Reviewer 2026-08-28] `_fit_description` runs OUTSIDE the repair
    loop's `try`, so an escaping AnnotationError would crash the whole run
    instead of quarantining one asset.

    Today no candidate can trigger one -- the draw loop pre-filters with
    `non_english_characters`, and that is the same function the validator's
    language rule calls. The guard is here because that argument is about
    ANOTHER module's state, not about this one."""
    ann, fit = _fit_description(_parsed(), ANCHOR, PROPS,
                                _ranked("   ", SHORT), MODEL)
    assert ann is not None, "a blank candidate must be skipped, not fatal"
    assert ann.description == SHORT
    assert fit["rank_used"] == 1


def test_without_the_guard_that_same_input_raises():
    """The failing counterpart: the bare call the reviewer flagged."""
    from metafind.data.annotate import AnnotationError, validate_annotation

    with pytest.raises(AnnotationError):
        validate_annotation(_parsed(), lvis_category=ANCHOR,
                            proportions=PROPS, description="   ")


def test_the_quarantine_record_separates_the_two_reasons():
    ann, fit = _fit_description(_parsed(), ANCHOR, PROPS,
                                _ranked(LONG, "   "), MODEL)
    assert ann is None
    assert fit["over_context"] and all(n > TEXT_CONTEXT_LENGTH
                                       for n in fit["over_context"])
    assert fit["rejected_by_validator"] == [1]
