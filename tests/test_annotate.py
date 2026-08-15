"""L1 tests for n05_annotate's schema and repair loop.

Covers L1-ANNOT-SCHEMA's truth table, L1-ANNOT-REPAIR, and the prompt property
that a finding asked for and no test previously enforced: the prompt must tell
the annotator the renders are scale-normalised.

None of these need the model. The parts that do -- generation, the C1 loop
running end to end, quarantine on exhaustion -- are exercised separately once
the GPU is free.
"""

from __future__ import annotations

import pytest

from metafind.data.annotate import (
    MAX_ATTEMPTS,
    PLACEMENT_VOCABULARY,
    AnnotationError,
    build_prompt,
    build_repair_prompt,
    parse_annotation,
    validate_annotation,
)


def _valid(**over):
    obj = {
        "category": "dining chair",
        "description": "A wooden dining chair with a slatted back.",
        "dimensions": {"length_m": 0.5, "width_m": 0.5, "height_m": 0.9},
        "materials": ["wood", "fabric"],
        "placement_constraints": ["floor"],
    }
    obj.update(over)
    return obj


# ----------------------------------------------------------------- the prompt


def test_prompt_says_the_renders_are_scale_normalised():
    """[F13] n04 fits every asset to a unit sphere, so size is not in the image.

    Without this sentence the prompt asks for dimensions from pictures that
    contain no scale, and a confident answer looks like a measurement. The
    estimate is a category prior and the prompt has to say so.
    """
    p = build_prompt(11).lower()
    assert "scale-normalised" in p or "scale normalised" in p
    assert "not from the picture" in p
    assert "metres" in p


def test_prompt_lists_the_closed_vocabulary():
    """A closed set the model is never shown is not closed, only enforced."""
    p = build_prompt(11)
    for term in PLACEMENT_VOCABULARY:
        assert term in p, term


def test_prompt_is_stable():
    """[cache_key: prompt_version] Two calls must not differ, or every asset
    re-annotates and the cache key means nothing."""
    assert build_prompt(11) == build_prompt(11)


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize("wrapper", [
    '{body}',
    'Here is the annotation:\n{body}',
    '```json\n{body}\n```',
    '```\n{body}\n```\nHope that helps!',
])
def test_json_is_recovered_from_whatever_the_model_wraps_it_in(wrapper):
    """Formatting is not a schema failure.

    Treating a fenced block as invalid spends one of two repair attempts on
    punctuation rather than on the content that was actually wrong.
    """
    body = '{"category": "chair", "x": 1}'
    assert parse_annotation(wrapper.format(body=body))["category"] == "chair"


def test_unparseable_response_names_the_problem():
    with pytest.raises(AnnotationError, match="no JSON object"):
        parse_annotation("I'm sorry, I can't help with that.")
    with pytest.raises(AnnotationError, match="malformed"):
        parse_annotation('{"category": "chair",}')


# ------------------------------------------------------ L1-ANNOT-SCHEMA


def test_a_valid_annotation_passes():
    a = validate_annotation(_valid())
    assert a.category == "dining chair"
    assert a.dimensions["height_m"] == 0.9
    assert a.placement_constraints == ["floor"]


@pytest.mark.parametrize("field", ["category", "description", "dimensions",
                                   "materials", "placement_constraints"])
def test_each_required_field_missing_is_rejected(field):
    obj = _valid()
    del obj[field]
    with pytest.raises(AnnotationError, match=field):
        validate_annotation(obj)


def test_placement_outside_the_vocabulary_is_rejected():
    """The whole point of a closed set."""
    with pytest.raises(AnnotationError, match="not allowed"):
        validate_annotation(_valid(placement_constraints=["on the roof"]))


def test_millimetres_are_rejected_with_the_unit_named():
    """The commonest failure: 900 for a chair rather than 0.9.

    The message must say METRES, because it is fed straight back into the
    repair prompt and "value out of range" tells the model nothing.
    """
    with pytest.raises(AnnotationError, match="METRES"):
        validate_annotation(_valid(dimensions={"length_m": 500, "width_m": 500,
                                               "height_m": 900}))


@pytest.mark.parametrize("dims", [
    {"length_m": 0.5, "width_m": 0.5},                       # missing a key
    {"length_m": "0.5", "width_m": 0.5, "height_m": 0.9},    # string
    {"length_m": True, "width_m": 0.5, "height_m": 0.9},     # bool is not a number
    {"length_m": 0.0, "width_m": 0.5, "height_m": 0.9},      # zero
])
def test_bad_dimensions_are_rejected(dims):
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(dimensions=dims))


def test_a_single_string_is_accepted_where_a_list_is_expected():
    """Models return "wood" for a one-material object constantly. Spending a
    repair attempt on that would be spending it on nothing."""
    a = validate_annotation(_valid(materials="wood", placement_constraints="floor"))
    assert a.materials == ["wood"] and a.placement_constraints == ["floor"]


def test_duplicate_placement_values_are_collapsed():
    a = validate_annotation(_valid(placement_constraints=["floor", "floor"]))
    assert a.placement_constraints == ["floor"]


def test_empty_lists_are_rejected():
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(materials=[]))
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(placement_constraints=[]))


# ------------------------------------------------------ L1-ANNOT-REPAIR


def test_repair_prompt_differs_and_names_the_error():
    """[L1-ANNOT-REPAIR] Resending the same prompt reproduces the same mistake.

    A loop whose retry is identical to its first attempt is not a repair loop;
    it is a delay with an attempt counter.
    """
    original = build_prompt(11)
    err = "`dimensions.height_m` = 900 is outside 0.001-100.0 m"
    repair = build_repair_prompt(original, err, '{"height_m": 900}')
    assert repair != original
    assert err in repair
    assert "900" in repair


def test_repair_prompt_refuses_to_be_built_without_an_error():
    with pytest.raises(ValueError, match="just a resend"):
        build_repair_prompt(build_prompt(11), "", "whatever")


def test_the_repair_budget_is_bounded():
    """[L1-ANNOT-EXHAUST] C1's hard bound. Two attempts, then quarantine --
    an exhausted item must never be admitted."""
    assert MAX_ATTEMPTS == 2


# --------------------------------------------------- L1-ANNOT-NO-FALLBACK


def test_no_ulip_caption_fallback_exists():
    """[L1-ANNOT-NO-FALLBACK] ULIP-2's shipped captions carry no placement
    constraints, so substituting them answers a different question."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "metafind" / "data" / "annotate.py"
    body = src.read_text()
    for banned in ("ulip_caption", "shipped_caption", "fallback_caption"):
        assert banned not in body, f"a caption fallback path reappeared: {banned}"
