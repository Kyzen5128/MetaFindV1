"""Tests for n05b_resolve_stage1_encoding.

The check that matters is L1-TEXT-SERIALIZATION: the annotation -> string
template is pinned, and a fixed golden annotation produces a byte-identical
string. Its negative injection is reordering two fields, which is exactly the
kind of edit that looks harmless and silently moves every text embedding.
"""

from __future__ import annotations

import pytest

from metafind.models.resolve_stage1 import (
    DEFAULT_HYPERPARAMETERS,
    IMAGE_AGGREGATION,
    MAX_DESCRIPTION_CHARS,
    MISSING_MODALITY,
    TEXT_SERIALIZATION,
    VARIANTS,
    build_hyperparameters,
    build_protocol,
    serialize_annotation,
)
from metafind.models.stage1_config import (
    KNOWN_MISSING_MODALITY,
    PAPER_P_MASK,
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
    REQUIRED_HYPERPARAMETERS,
)

GOLDEN_ANNOTATION = {
    "category": "dining chair",
    "synset": "chair.n.01",
    "description": "A wooden dining chair with a slatted back and four tapered legs",
    "width": 50.0, "length": 45.0, "height": 90.0,
    "mass": 6.0,
    "materials": ["wood", "fabric"],
    "onCeiling": False, "onWall": False, "onFloor": True, "onObject": False,
}

# [L1-TEXT-SERIALIZATION] The golden string. If this test fails, either the
# template changed on purpose -- in which case update it here AND note it in the
# report, because every text embedding moves with it -- or it changed by
# accident, which is what the test is for.
GOLDEN_STRING = (
    "A wooden dining chair with a slatted back and four tapered legs. "
    "A dining chair made of wood, fabric, "
    "roughly 50 by 45 by 90 centimetres, "
    "typically placed on the floor."
)


# --- L1-TEXT-SERIALIZATION -------------------------------------------------

def test_the_golden_annotation_serialises_byte_identically():
    assert serialize_annotation(GOLDEN_ANNOTATION) == GOLDEN_STRING


def test_reordering_two_fields_changes_the_string():
    """[L1-TEXT-SERIALIZATION negative injection] Swap materials and dimensions.

    Applied to the template the production function accepts, so the injected
    path is the real one.
    """
    swapped = (
        "{description} A {category} "
        "roughly {length:.2f} by {width:.2f} by {height:.2f} metres, "
        "made of {materials}, typically placed {placement}."
    )
    assert serialize_annotation(GOLDEN_ANNOTATION, swapped) != GOLDEN_STRING


def test_serialisation_is_stable_across_calls():
    a = serialize_annotation(GOLDEN_ANNOTATION)
    b = serialize_annotation(dict(GOLDEN_ANNOTATION))
    assert a == b


def test_a_description_already_ending_in_a_period_gains_no_second_one():
    ann = dict(GOLDEN_ANNOTATION, description="A plain stool.")
    assert ".. " not in serialize_annotation(ann)
    assert serialize_annotation(ann).startswith("A plain stool. A dining chair")


def test_two_placement_flags_read_as_prose():
    """[PAPER Figure 2] The flags are independent, so combinations are normal.

    A book is on a table or a shelf but can also sit on the floor: onFloor and
    onObject are both true and the sentence has to say so without reading as a
    serialised list.
    """
    ann = dict(GOLDEN_ANNOTATION, onFloor=True, onObject=True)
    assert "typically placed on the floor or on other objects." in serialize_annotation(ann)


def test_all_flags_false_reads_as_prose_not_as_an_error():
    """v1 demanded a positive placement value and `unconstrained` absorbed
    30.7% of the corpus. An abstract shape genuinely belongs nowhere."""
    ann = dict(GOLDEN_ANNOTATION, onFloor=False, onObject=False,
               onWall=False, onCeiling=False)
    assert "with no typical placement." in serialize_annotation(ann)


@pytest.mark.parametrize("field", ["materials"])
def test_an_empty_list_field_is_refused_not_rendered_as_punctuation(field):
    """"made of ," and "typically placed ." encode fine and rank badly.

    n05 already refuses both, but a guard living in another module vanishes the
    first time this function is called from somewhere else.
    """
    ann = dict(GOLDEN_ANNOTATION, **{field: []})
    with pytest.raises(ValueError):
        serialize_annotation(ann)


def test_underscores_never_reach_the_encoder():
    """The flag NAMES are camelCase identifiers (`onCeiling`); the sentence must
    carry their prose rendering, never the identifier."""
    ann = dict(GOLDEN_ANNOTATION, onFloor=False, onCeiling=True)
    out = serialize_annotation(ann)
    assert "_" not in out and "onCeiling" not in out
    assert "typically mounted on a ceiling." in out


# --- the 77-token budget ---------------------------------------------------

def test_an_overlong_description_is_capped_at_a_word_boundary():
    """CLIP truncates at 77 tokens silently, and the tail is what it drops.

    The tail is where placement_constraints lives -- the single most
    retrieval-relevant field -- so it must survive a long description rather
    than be the thing that falls off.
    """
    ann = dict(GOLDEN_ANNOTATION, description="wooden " * 200)
    out = serialize_annotation(ann)
    assert "typically placed on the floor." in out
    assert len(out) < 400
    # capped at a boundary, not mid-word
    assert "wooden woode." not in out


def test_a_short_description_is_untouched():
    assert GOLDEN_ANNOTATION["description"] in serialize_annotation(GOLDEN_ANNOTATION)
    assert len(GOLDEN_ANNOTATION["description"]) < MAX_DESCRIPTION_CHARS


def test_the_cap_falls_on_a_space_not_inside_a_word():
    ann = dict(GOLDEN_ANNOTATION,
               description="alpha bravo charlie delta echo foxtrot " * 10)
    out = serialize_annotation(ann)
    head = out.split(". A dining chair")[0]
    assert head.rstrip(".").split()[-1] in {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot"}


# --- the recorded decisions ------------------------------------------------

def test_image_aggregation_is_one_the_pipeline_implements():
    assert IMAGE_AGGREGATION in PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS


def test_missing_modality_is_a_known_representation_and_not_zero_padding():
    """[PAPER 2.6] "Rather than zero-padding, we apply masked embeddings"."""
    assert MISSING_MODALITY in KNOWN_MISSING_MODALITY
    assert "zero" not in MISSING_MODALITY


def test_the_protocol_records_every_field_the_channel_declares():
    p = build_protocol("trainable", "frozen", "tester")
    for field in ("status", "text_serialization", "image_aggregation",
                  "paper_clip_train_scope", "actual_clip_train_scope",
                  "missing_modality_representation", "decided_by", "decided_at"):
        assert field in p
    assert p["status"] == "resolved"
    assert p["text_serialization"] == TEXT_SERIALIZATION


@pytest.mark.parametrize("field", ["paper_clip_train_scope", "actual_clip_train_scope"])
def test_an_unknown_clip_scope_is_refused(field):
    kwargs = {"paper_clip_train_scope": "frozen",
              "actual_clip_train_scope": "frozen", "decided_by": "tester"}
    kwargs[field] = "partially"
    with pytest.raises(ValueError):
        build_protocol(**kwargs)


def test_d1_is_active_exactly_when_the_two_readings_disagree_that_way():
    """D-1's whole content is the GAP: the paper wants CLIP trained, we froze it."""
    def active(paper, actual):
        p = build_protocol(paper, actual, "tester")
        return (p["paper_clip_train_scope"] == "trainable"
                and p["actual_clip_train_scope"] == "frozen")

    assert active("trainable", "frozen") is True
    assert active("frozen", "frozen") is False
    assert active("trainable", "trainable") is False


# --- hyperparameters [U-22] ------------------------------------------------

def test_every_required_hyperparameter_is_named():
    """[U-22] The paper gives none, so the artifact must name each one."""
    h = build_hyperparameters("tester")
    for field in REQUIRED_HYPERPARAMETERS:
        assert field in h["values"], field


def test_p_mask_matches_the_paper_not_an_ablation():
    """[PAPER 2.6] 30% is stated. Table 3 ablates 0.10 and 0.50."""
    assert DEFAULT_HYPERPARAMETERS["p_mask"] == PAPER_P_MASK == 0.30


def test_the_hyperparameter_hash_changes_with_the_values():
    a = build_hyperparameters("tester")
    b = build_hyperparameters("tester", {"learning_rate": 3e-4})
    assert a["sha256"] != b["sha256"]


def test_a_missing_hyperparameter_is_refused():
    """The run must not start on a partial artifact -- U-22's whole point is
    that the report can name every value, and a silent default names none."""
    import metafind.models.resolve_stage1 as r
    original = r.DEFAULT_HYPERPARAMETERS
    try:
        r.DEFAULT_HYPERPARAMETERS = {k: v for k, v in original.items()
                                     if k != "learning_rate"}
        with pytest.raises(ValueError) as exc:
            r.build_hyperparameters("tester")
        assert "learning_rate" in str(exc.value)
    finally:
        r.DEFAULT_HYPERPARAMETERS = original


# --- Table 3 variants ------------------------------------------------------

def test_there_is_one_variant_per_table3_row():
    assert len(VARIANTS) == 10
    assert len({v["variant_id"] for v in VARIANTS}) == 10
    assert len({v["table3_row"] for v in VARIANTS}) == 10


def test_the_inference_only_variant_reuses_the_full_checkpoint():
    """Table 3's "w/o iterative retrieval" is the same model, composed differently.

    Retraining it would answer a question Table 3 is not asking.
    """
    v = next(v for v in VARIANTS if v["variant_id"] == "no_iterative")
    assert v["requires_training"] is False
    assert v["reuses_ckpt"] == "full"
    assert v["composition_mode"] == "parallel"


def test_every_other_variant_needs_its_own_training_run():
    for v in VARIANTS:
        if v["variant_id"] != "no_iterative":
            assert v["requires_training"] is True, v["variant_id"]
            assert v["train_scope"] is not None, v["variant_id"]


def test_the_full_variant_leaves_fusion_to_the_main_line():
    """[U-13] The paper lists five fusion strategies and never says which is Full."""
    full = next(v for v in VARIANTS if v["variant_id"] == "full")
    assert full["fusion"] is None
    assert full["dropout"] == PAPER_P_MASK
    assert full["layout_encoder"] == "essgnn"


def test_only_the_zero_pad_variant_overrides_the_missing_modality_rule():
    overriding = [v for v in VARIANTS if "missing_modality_representation" in v]
    assert [v["variant_id"] for v in overriding] == ["zero_pad"]


def test_the_dropout_ablations_carry_the_rates_table3_names():
    rates = {v["variant_id"]: v["dropout"] for v in VARIANTS}
    assert rates["dropout_10"] == 0.10
    assert rates["dropout_50"] == 0.50
