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
    TEXT_SERIALIZATION_FAMILY,
    TEXT_TEMPLATE,
    VARIANTS,
    build_hyperparameters,
    build_protocol,
    serialization_probes,
    serialize_annotation,
    text_serialization_id,
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
#
# Updated DELIBERATELY for D0-008 (ratified 2026-08-21). Two of the four
# approved edits are visible in this string: E-2 removed the fixed "A " article
# and S-2 capitalised the category. All 45,952 text embeddings move with it,
# which is exactly what this test exists to force someone to notice.
GOLDEN_STRING = (
    "A wooden dining chair with a slatted back and four tapered legs. "
    "Dining chair made of wood, fabric, "
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

    The injected template carries no `:.2f` any more: after D0-008 E-1/S-1 the
    dimensions reach `.format()` already rendered as strings, because "one
    decimal, trailing .0 stripped" is not expressible as a format spec. That is
    also why the cache identity hashes the emitted string rather than this
    template constant.
    """
    swapped = (
        "{description} {category} "
        "roughly {length} by {width} by {height} centimetres, "
        "made of {materials}, {placement}."
    )
    assert serialize_annotation(GOLDEN_ANNOTATION, swapped) != GOLDEN_STRING


def test_serialisation_is_stable_across_calls():
    a = serialize_annotation(GOLDEN_ANNOTATION)
    b = serialize_annotation(dict(GOLDEN_ANNOTATION))
    assert a == b


def test_a_description_already_ending_in_a_period_gains_no_second_one():
    ann = dict(GOLDEN_ANNOTATION, description="A plain stool.")
    assert ".. " not in serialize_annotation(ann)
    assert serialize_annotation(ann).startswith("A plain stool. Dining chair")


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
    head = out.split(". Dining chair")[0]
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
    assert p["text_serialization"] == text_serialization_id()
    assert p["text_template"] == TEXT_TEMPLATE


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


# --- D0-008 E-1 / S-1: the dimension formatter -----------------------------

def test_a_sub_centimetre_dimension_keeps_its_decimal():
    """[E-1] `:.0f` rendered 161 records' stored dimension as `0`.

    The string then asserted something the annotation did not say, which is a
    false statement handed to a frozen encoder rather than a rounding nicety.
    """
    ann = dict(GOLDEN_ANNOTATION, width=4.0, length=4.0, height=0.5)
    assert "roughly 4 by 4 by 0.5 centimetres" in serialize_annotation(ann)


def test_the_formatter_applies_above_one_centimetre_too():
    """[S-1] Kyzen's wording scoped it below 1 cm; the corpus holds one 2.5.

    Under a threshold reading that value renders `2` and loses a fifth of the
    dimension, so the formatter is uniform at every magnitude.
    """
    ann = dict(GOLDEN_ANNOTATION, width=25.0, length=25.0, height=2.5)
    assert "roughly 25 by 25 by 2.5 centimetres" in serialize_annotation(ann)


def test_an_integer_dimension_stays_bare():
    """No `.0` tail: the trailing zero is stripped, so integers read as before
    E-1 and only the fractional values changed."""
    assert "roughly 50 by 45 by 90 centimetres" in serialize_annotation(GOLDEN_ANNOTATION)


def test_a_stored_zero_dimension_still_renders_zero():
    """The rule is "never render a stored NON-ZERO value as 0", not "never emit
    a zero". A record that says 0 must keep saying 0."""
    ann = dict(GOLDEN_ANNOTATION, height=0.0)
    assert "roughly 50 by 45 by 0 centimetres" in serialize_annotation(ann)


# --- D0-008 E-2 / S-2: the article and the capital --------------------------

def test_no_article_precedes_the_category():
    """[E-2] The fixed "A " produced 3,643 "A airplane" / "A umbrella" strings.

    It is REMOVED rather than repaired: a first-letter a/an rule is wrong for
    "hour", "university", "MRI" and "USB", and Kyzen forbade the heuristic.
    """
    for category in ("airplane", "umbrella", "apple", "hour", "MRI"):
        out = serialize_annotation(dict(GOLDEN_ANNOTATION, category=category))
        assert " A " not in out
        assert " An " not in out


def test_the_category_starts_the_clause_with_a_capital():
    """[S-2] Removing "A " left the second sentence opening on a lower-case
    noun. Capitalising is not an a/an heuristic and carries no vocabulary."""
    out = serialize_annotation(dict(GOLDEN_ANNOTATION, category="flip-flop"))
    assert ". Flip-flop made of" in out


def test_capitalising_does_not_lower_case_the_rest_of_the_category():
    """`str.capitalize()` would turn "LED lamp" into "Led lamp"."""
    out = serialize_annotation(dict(GOLDEN_ANNOTATION, category="LED lamp"))
    assert ". LED lamp made of" in out


def test_a_non_alphabetic_leading_character_is_left_alone():
    """`"3d printer"[:1].upper()` is `"3"`. No branch, no crash, no change."""
    out = serialize_annotation(dict(GOLDEN_ANNOTATION, category="3d printer"))
    assert ". 3d printer made of" in out


# --- R-3: the deleted PLACEMENT_PHRASES entry -------------------------------

def test_wall_and_ceiling_together_read_as_prose():
    """[R-3, D0-008 §12.3] The ("onWall", "onCeiling") entry was UNREACHABLE:
    placement_phrase() only ever builds ("onCeiling", "onWall").

    Master ruled delete rather than fix, so this combination takes the fallback
    join -- which is what it already did, and what those records already say.
    """
    ann = dict(GOLDEN_ANNOTATION, onFloor=False, onWall=True, onCeiling=True)
    out = serialize_annotation(ann)
    assert "typically mounted on a ceiling or on a wall." in out
    assert "typically mounted on a wall or ceiling" not in out


# --- B-3: the cache identity ------------------------------------------------

def test_the_retired_identifier_is_gone():
    """[B-3] "metafind_v1_natural" labelled BOTH the metre and the centimetre
    template, so 5,276 sidecars carry a name that identifies no transformation.
    A name that can mean two things is not a cache identity."""
    assert TEXT_SERIALIZATION_FAMILY != "metafind_v1_natural"
    assert "metafind_v1_natural" not in build_protocol(
        "frozen", "frozen", "tester")["text_serialization"]


def test_the_identity_is_content_addressed_not_a_version_string():
    """It must move when the emitted string moves, whether or not anyone
    remembers to bump a number. That is the whole content of B-3."""
    import metafind.models.resolve_stage1 as r

    before = text_serialization_id()
    original = r.TEXT_TEMPLATE
    try:
        r.TEXT_TEMPLATE = original.replace("centimetres", "metres")
        assert text_serialization_id() != before
    finally:
        r.TEXT_TEMPLATE = original
    assert text_serialization_id() == before


def test_the_identity_is_stable_across_calls():
    assert text_serialization_id() == text_serialization_id()
    assert text_serialization_id().startswith(TEXT_SERIALIZATION_FAMILY + "@")


def test_the_protocol_records_the_strings_the_serializer_emits():
    """The artifact must DESCRIBE the encoder, not merely name it. Before D10
    it recorded the metre template while the code emitted centimetres."""
    p = build_protocol("frozen", "frozen", "tester")
    assert p["text_serialization_probes"] == [
        serialize_annotation(probe) for probe in serialization_probes()]
    assert all("centimetres" in s for s in p["text_serialization_probes"])


@pytest.mark.parametrize("knob", [
    "TEXT_TEMPLATE", "NO_PLACEMENT_PHRASE",
    "MAX_DESCRIPTION_CHARS", "MAX_CATEGORY_CHARS", "MAX_MATERIALS",
])
def test_moving_any_knob_moves_the_identity(knob):
    """[B-3] Adversarial review found a single probe left most of this module
    invisible to the hash: editing PLACEMENT_PHRASES[("onFloor",)] moved every
    floor-standing record's string while the identity sat still, so the protocol
    would have gone on certifying a serializer that no longer existed.

    The probe SUITE closes that. Each knob below changes the emitted text for
    some record, so each must change the identity.

    Integer knobs are HALVED rather than decremented: 40 -> 39 on the category
    cap happens to trim the probe at the same word boundary, so a one-step
    mutation is not evidence either way.
    """
    import metafind.models.resolve_stage1 as r

    before = text_serialization_id()
    original = getattr(r, knob)
    try:
        setattr(r, knob, original.replace("centimetres", "cm")
                if knob == "TEXT_TEMPLATE" else
                "nowhere in particular" if isinstance(original, str) else
                original // 2)
        assert text_serialization_id() != before
    finally:
        setattr(r, knob, original)
    assert text_serialization_id() == before


def test_editing_a_placement_phrase_moves_the_identity():
    import metafind.models.resolve_stage1 as r

    before = text_serialization_id()
    original = dict(r.PLACEMENT_PHRASES)
    try:
        r.PLACEMENT_PHRASES[("onFloor",)] = "usually on the ground"
        assert text_serialization_id() != before
    finally:
        r.PLACEMENT_PHRASES.clear()
        r.PLACEMENT_PHRASES.update(original)
    assert text_serialization_id() == before


def test_re_adding_the_deleted_dead_entry_moves_the_identity():
    """[R-3] Deleting it changed 0 strings, so nothing else would notice it
    coming back. The identity does."""
    import metafind.models.resolve_stage1 as r

    before = text_serialization_id()
    try:
        r.PLACEMENT_PHRASES[("onWall", "onCeiling")] = "typically mounted on a wall or ceiling"
        assert text_serialization_id() != before
    finally:
        del r.PLACEMENT_PHRASES[("onWall", "onCeiling")]
    assert text_serialization_id() == before


def test_the_probe_suite_covers_every_placement_branch():
    """A knob the suite never exercises is a knob the identity cannot see."""
    import metafind.models.resolve_stage1 as r

    emitted = {r.placement_phrase(p) for p in serialization_probes()}
    for phrase in r.PLACEMENT_PHRASES.values():
        assert phrase in emitted, phrase
    assert r.NO_PLACEMENT_PHRASE in emitted
    assert "typically mounted on a ceiling or on a wall" in emitted  # the fallback


def test_a_one_step_cap_change_still_moves_the_identity():
    """[B-3] Adversarial review round 2: 160 -> 161 changed a real corpus record
    while every probe truncated at the same word boundary, so a probe-only hash
    sat still and load_protocol() kept accepting the old protocol.

    A sample cannot cover a continuous parameter, so the constants are hashed
    alongside the emitted strings.
    """
    import metafind.models.resolve_stage1 as r

    before = text_serialization_id()
    try:
        r.MAX_DESCRIPTION_CHARS = r.MAX_DESCRIPTION_CHARS + 1
        assert text_serialization_id() != before
    finally:
        r.MAX_DESCRIPTION_CHARS = r.MAX_DESCRIPTION_CHARS - 1
    assert text_serialization_id() == before


def test_the_contract_manifest_names_every_constant_the_string_depends_on():
    from metafind.models.resolve_stage1 import serialization_contract

    c = serialization_contract()
    assert set(c) == {"family", "template", "max_description_chars",
                      "max_category_chars", "max_materials",
                      "placement_phrases", "no_placement_phrase"}
    assert c["template"] == TEXT_TEMPLATE
    assert "onWall+onCeiling" not in c["placement_phrases"]   # R-3 stays deleted


# --- tau [C-001, D2a] ------------------------------------------------------

def test_the_emitted_temperature_is_the_papers_value():
    """[PAPER FACT] 3experiments.tex:15 -- "The temperature is 0.5 for all
    experiments." n05b had no path to it at all: 0.07 was hardcoded and no CLI
    flag reached the field, so the artifact could only ever record CLIP's
    convention."""
    from metafind.models.losses import PAPER_TAU

    assert DEFAULT_HYPERPARAMETERS["init_temperature"] == 0.5 == PAPER_TAU
    assert build_hyperparameters("tester")["values"]["init_temperature"] == 0.5


def test_temperature_is_fixed_not_learned():
    """[USER-RATIFIED IMPLEMENTATION CHOICE, not a paper statement] MetaFind
    nowhere states that tau is non-learnable. The inference rests on the authors'
    own vocabulary: they call f_h/f_x "learnable functions" (2methdology.tex:54)
    and lambda "a learnable scalar" (:87), but name tau "a temperature
    hyperparameter" at both :79 and :99 -- and a value that is optimised does not
    stay fixed "for all experiments" (3experiments.tex:15). Strong, and still an
    inference. Ratified by the user 2026-08-21 as a choice, not as paper wording."""
    assert DEFAULT_HYPERPARAMETERS["learnable_temperature"] is False
    assert build_hyperparameters("tester")["values"]["learnable_temperature"] is False


def test_the_paper_temperature_survives_the_loss_constructor_without_a_deviation_warning():
    """The artifact and the trainer have to agree. `losses.py:114` warns whenever
    tau deviates; emitting 0.5 fixed is precisely the case that must NOT warn."""
    import warnings

    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss

    values = build_hyperparameters("tester")["values"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any deviation warning fails here
        MetaFindContrastiveLoss(ContrastiveConfig(
            learnable_temperature=values["learnable_temperature"],
            init_temperature=values["init_temperature"],
            max_logit_scale=values["max_logit_scale"]))


def test_an_explicit_override_can_still_reach_the_temperature():
    """C-001 makes 0.5 the default; it must not make it the only possibility. A
    deviation experiment stays expressible -- and `losses.py` will flag it."""
    h = build_hyperparameters("tester", {"init_temperature": 0.07,
                                         "learnable_temperature": True})
    assert h["values"]["init_temperature"] == 0.07
    assert h["sha256"] != build_hyperparameters("tester")["sha256"]


# --- v3_fit: the sentence trimmed to CLIP's 77 tokens [KYZEN 2026-09-06] --------------

def _fit_annotation(n_words: int) -> dict:
    return {"category": "table lamp", "description": " ".join(["ornate"] * n_words) + ".",
            "materials": ["ceramic", "fabric", "metal", "glass"], "width": 35.0, "length": 35.0,
            "height": 60.0, "onCeiling": False, "onWall": False, "onFloor": False, "onObject": True}


def test_v3_fit_keeps_every_structured_field_and_stays_within_77_tokens():
    from metafind.models.resolve_stage1 import serialize_annotation, true_token_count
    s = serialize_annotation(_fit_annotation(120), template="__fit__")
    assert true_token_count(s) <= 77
    assert s.endswith("Table lamp made of ceramic, fabric, metal, roughly 35 by 35 by 60 centimetres, "
                      "typically placed on top of other objects.")
    assert s.startswith("ornate ornate")
    assert ". Table lamp" in s                      # the trimmed description still ends with a period


def test_v3_fit_leaves_a_short_description_untouched():
    from metafind.models.resolve_stage1 import serialize_annotation, TEXT_TEMPLATES
    ann = _fit_annotation(12)
    fitted = serialize_annotation(ann, template="__fit__")
    plain = serialize_annotation(ann, template=TEXT_TEMPLATES["v2_cm"])
    assert fitted == plain


def test_v3_fit_is_deterministic_and_trims_at_word_boundaries():
    from metafind.models.resolve_stage1 import serialize_annotation, true_token_count
    ann = _fit_annotation(200)
    a, b = serialize_annotation(ann, template="__fit__"), serialize_annotation(ann, template="__fit__")
    assert a == b
    head = a.split(" Table lamp")[0]
    assert set(head.rstrip(".").split()) == {"ornate"}
    # one more word would not fit
    assert true_token_count(a) <= 77 < true_token_count(a.replace(". Table lamp", " ornate. Table lamp", 1))


def test_v3_fit_without_materials_drops_the_made_of_clause():
    from metafind.models.resolve_stage1 import serialize_annotation
    ann = _fit_annotation(5); ann["materials"] = []
    s = serialize_annotation(ann, template="__fit__")
    assert "made of" not in s and "Table lamp, roughly 35 by 35 by 60 centimetres" in s


def test_v3_fit_refuses_when_the_structured_fields_alone_overflow():
    from metafind.models.resolve_stage1 import serialize_annotation
    import pytest as _pt
    ann = _fit_annotation(3)
    # three absurd multi-word "materials": the tail alone is well over budget
    ann["materials"] = [" ".join(f"material{i}" for i in range(40))] * 3
    with _pt.raises(ValueError, match="structured fields alone"):
        serialize_annotation(ann, template="__fit__")
