"""L1 tests for annotation parsing and the repair loop (sec. 2.3).

A fake backend stands in for the vision-language model, so the schema, the
prompt and the bounded repair loop are all exercised without a 16 GB checkpoint.
"""

from __future__ import annotations

import json

import pytest

from metafind.data.annotate import (
    PLACEMENT_VOCABULARY,
    AnnotationConfig,
    SchemaError,
    annotate_one,
    build_prompt,
    parse_annotation,
)


def good(**overrides) -> dict:
    base = {
        "category": "dining chair",
        "description": "A wooden dining chair with a slatted back.",
        "dimensions": {"width_m": 0.45, "height_m": 0.9, "depth_m": 0.5},
        "materials": ["wood", "fabric"],
        "placement_constraints": ["floor_standing"],
    }
    base.update(overrides)
    return base


class FakeBackend:
    """Replies with a scripted sequence, recording the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, images, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


# --------------------------------------------------------------- parsing


def test_valid_annotation_parses():
    out = parse_annotation(json.dumps(good()))
    assert out["category"] == "dining chair"
    assert out["dimensions"]["height_m"] == 0.9
    assert out["placement_constraints"] == ["floor_standing"]


def test_markdown_fences_are_tolerated():
    """Models wrap JSON in fences constantly; rejecting that wastes repair budget."""
    text = "```json\n" + json.dumps(good()) + "\n```"
    assert parse_annotation(text)["category"] == "dining chair"


def test_surrounding_prose_is_tolerated():
    text = "Sure! Here is the description:\n" + json.dumps(good()) + "\nHope that helps."
    assert parse_annotation(text)["category"] == "dining chair"


def test_non_json_is_rejected():
    with pytest.raises(SchemaError, match="no JSON object"):
        parse_annotation("I'm afraid I can't tell what this object is.")


def test_json_array_is_rejected():
    with pytest.raises(SchemaError, match="expected a JSON object"):
        parse_annotation("[1, 2, 3]")


# --------------------------------------------------------------- required fields


@pytest.mark.parametrize(
    "field", ["category", "description", "dimensions", "materials", "placement_constraints"]
)
def test_each_paper_named_field_is_required(field: str):
    """sec. 2.3 names category, size, materials and placement constraints."""
    payload = good()
    del payload[field]
    with pytest.raises(SchemaError, match="missing required field"):
        parse_annotation(json.dumps(payload))


def test_empty_strings_are_rejected():
    with pytest.raises(SchemaError, match="non-empty string"):
        parse_annotation(json.dumps(good(category="   ")))


def test_empty_lists_are_rejected():
    with pytest.raises(SchemaError, match="non-empty list"):
        parse_annotation(json.dumps(good(materials=[])))
    with pytest.raises(SchemaError, match="non-empty list"):
        parse_annotation(json.dumps(good(placement_constraints=[])))


# --------------------------------------------------------------- dimensions


@pytest.mark.parametrize("axis", ["width_m", "height_m", "depth_m"])
def test_every_axis_is_required(axis: str):
    dims = {"width_m": 0.4, "height_m": 0.9, "depth_m": 0.5}
    del dims[axis]
    with pytest.raises(SchemaError, match=f"missing {axis}"):
        parse_annotation(json.dumps(good(dimensions=dims)))


def test_millimetre_answers_are_caught():
    """A model reporting 450 for a chair means millimetres; that must not pass."""
    with pytest.raises(SchemaError, match="outside"):
        parse_annotation(json.dumps(good(dimensions={"width_m": 450, "height_m": 900, "depth_m": 500})))


def test_absurd_dimensions_are_caught():
    with pytest.raises(SchemaError, match="outside"):
        parse_annotation(json.dumps(good(dimensions={"width_m": 0.4, "height_m": 400.0, "depth_m": 0.5})))


def test_non_numeric_dimensions_are_caught():
    with pytest.raises(SchemaError, match="not a number"):
        parse_annotation(json.dumps(good(dimensions={"width_m": "wide", "height_m": 0.9, "depth_m": 0.5})))


def test_numeric_strings_are_accepted_and_coerced():
    """Rejecting "0.45" would burn a repair attempt over formatting, not content."""
    out = parse_annotation(json.dumps(good(dimensions={"width_m": "0.45", "height_m": "0.9", "depth_m": "0.5"})))
    assert out["dimensions"]["width_m"] == 0.45
    assert isinstance(out["dimensions"]["width_m"], float)


# --------------------------------------------------------------- placement vocabulary


def test_placement_vocabulary_is_closed():
    """The layout signal is only usable if the values are drawn from a fixed set."""
    with pytest.raises(SchemaError, match="outside the allowed set"):
        parse_annotation(json.dumps(good(placement_constraints=["on_a_desk_maybe"])))


def test_every_allowed_placement_value_passes():
    """Negative injection above proves nothing unless the whole vocabulary works."""
    for value in PLACEMENT_VOCABULARY:
        out = parse_annotation(json.dumps(good(placement_constraints=[value])))
        assert out["placement_constraints"] == [value]


def test_error_message_names_the_offending_value():
    """The message is fed back to the model, so it has to be specific."""
    with pytest.raises(SchemaError) as exc:
        parse_annotation(json.dumps(good(placement_constraints=["floor_standing", "in_the_sky"])))
    assert "in_the_sky" in str(exc.value)
    assert "floor_standing" in str(exc.value), "the allowed set should be listed"


# --------------------------------------------------------------- repair loop


def test_repair_loop_recovers_from_one_bad_reply():
    backend = FakeBackend(["not json at all", json.dumps(good())])
    out, errors = annotate_one(backend, images=[], cfg=AnnotationConfig())
    assert out["category"] == "dining chair"
    assert len(errors) == 1, "the recovered failure must still be recorded"
    assert len(backend.prompts) == 2


def test_repair_prompt_differs_and_carries_the_reason():
    """Resending the identical prompt would just reproduce the identical mistake."""
    backend = FakeBackend([json.dumps(good(placement_constraints=["nowhere"])), json.dumps(good())])
    annotate_one(backend, images=[], cfg=AnnotationConfig())
    first, second = backend.prompts
    assert first != second
    assert "nowhere" in second
    assert "rejected" in second


def test_repair_loop_is_bounded():
    """MODEL_RECOVERABLE has a budget; exhausting it must fail, not loop."""
    cfg = AnnotationConfig(max_repair_attempts=2)
    backend = FakeBackend(["garbage"] * 10)
    with pytest.raises(SchemaError, match="after 3 attempts"):
        annotate_one(backend, images=[], cfg=cfg)
    assert len(backend.prompts) == 3, "should stop at 1 + max_repair_attempts"


def test_successful_first_reply_makes_no_repair_call():
    backend = FakeBackend([json.dumps(good())])
    out, errors = annotate_one(backend, images=[], cfg=AnnotationConfig())
    assert errors == []
    assert len(backend.prompts) == 1


# --------------------------------------------------------------- prompt


def test_prompt_states_the_view_count_and_vocabulary():
    prompt = build_prompt(AnnotationConfig())
    assert "11" in prompt
    for value in PLACEMENT_VOCABULARY:
        assert value in prompt, f"{value} is accepted but never offered to the model"


def test_prompt_explains_that_renders_are_scale_normalised():
    """F13: absolute scale is destroyed by normalisation, so size is a prior."""
    assert "scale-normalised" in build_prompt(AnnotationConfig())


def test_prompt_asks_for_all_four_paper_attributes():
    prompt = build_prompt(AnnotationConfig())
    for field in ("category", "dimensions", "materials", "placement_constraints"):
        assert field in prompt
