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
    PLACEMENT_FLAGS,
    AnnotationError,
    build_prompt,
    build_repair_prompt,
    parse_annotation,
    validate_annotation,
)


def _valid(**over):
    obj = {
        "category": "dining chair",
        "synset": "chair.n.01",
        "description": "A wooden dining chair with a slatted back.",
        "width": 50.0, "length": 50.0, "height": 90.0,
        "mass": 6.0,
        "materials": ["wood", "fabric"],
        "onCeiling": False, "onWall": False, "onFloor": True, "onObject": False,
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
    assert "centimetres" in p


def test_prompt_names_all_four_placement_flags():
    """[PAPER Figure 2] Four INDEPENDENT booleans, not a category choice.

    A flag the model is never shown cannot be answered, and v1's single-choice
    vocabulary is exactly what put `handheld` on a gaming chair.
    """
    p = build_prompt(11)
    for flag in PLACEMENT_FLAGS:
        assert flag in p, flag
    assert "INDEPENDENT" in p


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
    assert a.synset == "chair.n.01"
    assert a.height == 90.0
    assert (a.on_floor, a.on_object, a.on_wall, a.on_ceiling) == (True, False, False, False)


def test_volume_is_derived_not_asked():
    """[PAPER Figure 2] volume 36000 = width 30 * length 30 * height 40.

    Asking the model for a fourth number invites one that disagrees with the
    other three, with no way afterwards to tell which was wrong.
    """
    a = validate_annotation(_valid(width=30, length=30, height=40))
    assert a.volume == 36000
    assert "volume" not in build_prompt(11)


@pytest.mark.parametrize("field", ["category", "synset", "description", "width",
                                   "length", "height", "mass", "materials",
                                   "onCeiling", "onWall", "onFloor", "onObject"])
def test_each_required_field_missing_is_rejected(field):
    obj = _valid()
    del obj[field]
    with pytest.raises(AnnotationError, match=field):
        validate_annotation(obj)


@pytest.mark.parametrize("flag", list(PLACEMENT_FLAGS))
def test_placement_flags_must_be_booleans(flag):
    """A string where a boolean belongs is v1's category-choice habit leaking
    back in. `"onFloor": "yes"` is truthy and would pass a bare `if`."""
    with pytest.raises(AnnotationError, match="true or false"):
        validate_annotation(_valid(**{flag: "yes"}))


def test_all_four_false_is_accepted():
    """An abstract shape belongs nowhere in particular, and that is an answer.

    v1 demanded a positive value and `unconstrained` absorbed 30.7% of the
    corpus as a result.
    """
    a = validate_annotation(_valid(onCeiling=False, onWall=False,
                                   onFloor=False, onObject=False))
    assert not any((a.on_ceiling, a.on_wall, a.on_floor, a.on_object))


def test_all_four_true_is_accepted():
    """The flags are independent, so every combination is representable."""
    a = validate_annotation(_valid(onCeiling=True, onWall=True,
                                   onFloor=True, onObject=True))
    assert all((a.on_ceiling, a.on_wall, a.on_floor, a.on_object))


def test_metres_are_rejected_with_the_unit_named():
    """v1's failure inverted. That schema was metres and kept receiving
    millimetres; this one is centimetres and will receive metres -- 0.9 for a
    chair rather than 90. The message must say CENTIMETRES, because it goes
    straight back into the repair prompt.
    """
    with pytest.raises(AnnotationError, match="CENTIMETRES"):
        validate_annotation(_valid(width=0.5, length=0.5, height=0.9))


@pytest.mark.parametrize("bad", ["0.5", True, 0.0, None])
def test_bad_dimensions_are_rejected(bad):
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(height=bad))


@pytest.mark.parametrize("bad", ["2.5", True, 0.0, None])
def test_bad_mass_is_rejected(bad):
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(mass=bad))


@pytest.mark.parametrize("bad", ["robot", "robot.n", "robot.x.01", "robot.n.aa"])
def test_malformed_synset_is_rejected(bad):
    with pytest.raises(AnnotationError, match="WordNet"):
        validate_annotation(_valid(synset=bad))


def test_synset_shape_is_checked_but_existence_is_not():
    """No WordNet corpus here, so a well-formed invention passes.

    Recorded rather than hidden: the check is a SHAPE check, and validity is
    measured downstream instead of being assumed.
    """
    a = validate_annotation(_valid(synset="notathing.n.07"))
    assert a.synset == "notathing.n.07"


def test_a_single_string_is_accepted_where_a_list_is_expected():
    """Models return "wood" for a one-material object constantly. Spending a
    repair attempt on that would be spending it on nothing."""
    a = validate_annotation(_valid(materials="wood"))
    assert a.materials == ["wood"]


def test_material_spelling_variants_are_folded():
    """[MATERIAL_SYNONYMS] v1 emitted `metal` 34.3% AND `metallic` 10.7% as
    separate tokens; a text encoder reads those as two different materials."""
    a = validate_annotation(_valid(materials=["Metallic", "metal", "WOODEN"]))
    assert a.materials == ["metal", "wood"]


def test_nothing_is_dropped_from_materials():
    """Only spellings are merged. Deciding what "is not a material" is a
    judgement the paper does not license, so `textured` survives."""
    a = validate_annotation(_valid(materials=["textured", "plastic"]))
    assert a.materials == ["textured", "plastic"]


def test_empty_materials_is_rejected():
    with pytest.raises(AnnotationError):
        validate_annotation(_valid(materials=[]))


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
