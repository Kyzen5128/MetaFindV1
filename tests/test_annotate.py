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
    PROMPT_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    AnnotationError,
    annotation_contract_id,
    build_prompt,
    build_repair_prompt,
    non_english_characters,
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


# ====================================================================== P-1
# The prompt states the output language.

def test_the_prompt_requires_english_and_names_the_fields():
    """[P-1] v3 never named a language. Qwen2.5-VL is multilingual, so a Chinese
    description was a CORRECT answer to the prompt as written, and 3 of 45,952
    records came back part-Chinese. The requirement has to be stated, not hoped
    for."""
    p = build_prompt(11)
    assert "ENGLISH" in p
    for field in ("category", "description", "materials"):
        assert field in p
    lower = p.lower()
    for script in ("chinese", "japanese", "korean", "cyrillic", "arabic"):
        assert script in lower, script


def test_the_prompt_still_permits_accented_latin():
    """[P-1] The instruction must not read as "ASCII only" -- "Pokémon" is
    correct English spelling and 7 records legitimately use accents."""
    assert "Pok\u00e9mon" in build_prompt(11) or "Pokémon" in build_prompt(11)


def test_the_prompt_version_moved_because_the_prompt_changed():
    """[P-1] v4 asks for something v3 did not. Pretending they are one contract
    is what makes a corpus unfalsifiable later."""
    assert PROMPT_VERSION == 4


def test_editing_the_prompt_moves_the_contract_id():
    """The old `test_prompt_is_stable` only checked that two calls agree, so a
    prompt EDIT was invisible to the whole suite. The contract fingerprint is
    what actually pins the text."""
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = a.build_prompt
    try:
        a.build_prompt = lambda n: original(n) + "\nAnswer in rhyme."
        assert annotation_contract_id() != before
    finally:
        a.build_prompt = original
    assert annotation_contract_id() == before


# ====================================================================== P-2
# Validation refuses non-English text.

@pytest.mark.parametrize("text", [
    "Pokemon", "Pokémon", "Poké Ball", "Raphaël", "Carmín", "naïve café",
    "5¢", "a price of €3", "curly “quotes” and an em—dash", "20°C",
    "3d printer", "LED lamp", "",
])
def test_english_including_accents_and_currency_is_accepted(text):
    """[P-2] `.isascii()` was rejected as the rule precisely because it fails
    here. The complete non-ASCII vocabulary of the corpus is é, í, ë and ¢ --
    from Pokémon, Carmín, Raphaël and a 5¢ price tag -- and all of it is correct
    English."""
    assert non_english_characters(text) == []


@pytest.mark.parametrize("text,script", [
    ("abc抽吸液体", "CJK"),
    ("slight凹陷", "CJK"),
    ("登多利卷心糖", "CJK"),
    ("由塑料或金属制成，带有针头。", "CJK punctuation"),
    ("привет", "Cyrillic"),
    ("مرحبا", "Arabic"),
    ("こんにちは", "Kana"),
    ("한국어", "Hangul"),
])
def test_non_latin_scripts_are_refused(text, script):
    assert non_english_characters(text), script


def test_the_rule_is_script_based_not_ascii_based():
    """The distinction that matters: same non-ASCII status, opposite verdicts."""
    assert not "Pokémon".isascii() and non_english_characters("Pokémon") == []
    assert not "凹陷".isascii() and non_english_characters("凹陷") != []


@pytest.mark.parametrize("field", ["category", "description"])
def test_a_non_english_text_field_is_refused_by_validation(field):
    """[P-2] Under VALIDATOR_VERSION 1 this passed on attempt 1 and the repair
    loop never fired, which is why three records had to be repaired by hand."""
    with pytest.raises(AnnotationError) as exc:
        validate_annotation(_valid(**{field: "凹陷"}))
    assert field in str(exc.value)
    assert "English" in str(exc.value)


def test_a_non_english_material_is_refused_by_validation():
    with pytest.raises(AnnotationError, match="English"):
        validate_annotation(_valid(materials=["wood", "塑料"]))


def test_the_language_error_tells_the_model_what_to_do():
    """The message is fed back verbatim by build_repair_prompt(), so "invalid
    schema" would spend a repair attempt on nothing."""
    with pytest.raises(AnnotationError) as exc:
        validate_annotation(_valid(description="a device 由塑料制成"))
    msg = str(exc.value)
    assert "must be written in English" in msg
    assert "Chinese" in msg and "Pok" in msg      # names the rule AND the exception
    assert "由" in msg or "塑" in msg               # names the offending characters


def test_an_accented_annotation_still_validates_end_to_end():
    ann = validate_annotation(_valid(category="Pokémon figure",
                                     description="A Poké Ball replica.",
                                     materials=["plastic"]))
    assert ann.category == "Pokémon figure"


# ====================================================================== P-5
# The record can say where its text came from.

def test_a_model_record_declares_itself_model_generated():
    """[P-5] "model-generated" must be a RECORDED fact, not the absence of a
    note -- otherwise a hand-repaired record is indistinguishable from an
    untouched one."""
    rec = validate_annotation(_valid()).as_record("Qwen/Qwen2.5-VL-7B-Instruct")
    assert rec["description_source"] == "model"
    assert rec["description_original"] is None
    assert rec["description_translation_authority"] is None
    assert rec["description_translated_by"] is None


def test_a_record_carries_all_three_contract_axes_and_a_fingerprint():
    """[P-5] prompt_version alone cannot express "which validator admitted
    this"."""
    rec = validate_annotation(_valid()).as_record("m")
    assert rec["prompt_version"] == PROMPT_VERSION
    assert rec["validator_version"] == VALIDATOR_VERSION
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["annotation_contract"] == annotation_contract_id()
    assert rec["annotation_contract"].startswith(f"metafind_annot_v{PROMPT_VERSION}@")


@pytest.mark.parametrize("knob", [
    "MIN_DIM_CM", "MAX_MASS_KG", "NON_LATIN_SYMBOL_CUTOFF", "MAX_ATTEMPTS",
    "VALIDATOR_VERSION", "SCHEMA_VERSION",
])
def test_moving_any_admission_rule_moves_the_contract_id(knob):
    """Three integers can be forgotten; a fingerprint over the actual rules
    cannot. Same argument as D10's text_serialization_id()."""
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = getattr(a, knob)
    try:
        setattr(a, knob, original + 1)
        assert annotation_contract_id() != before
    finally:
        setattr(a, knob, original)
    assert annotation_contract_id() == before


def test_changing_a_material_synonym_moves_the_contract_id():
    import metafind.data.annotate as a

    before = annotation_contract_id()
    try:
        a.MATERIAL_SYNONYMS["tin"] = "metal"
        assert annotation_contract_id() != before
    finally:
        del a.MATERIAL_SYNONYMS["tin"]
    assert annotation_contract_id() == before


# ====================================================================== P-3
# A language failure must reach the repair loop, not be accepted as-is.

CJK_RESPONSE = ('{"category": "syringe", "synset": "syringe.n.01", "width": 5, '
                '"length": 10, "height": 2, "mass": 0.1, "description": '
                '"a medical device used for injecting or抽吸液体", '
                '"materials": ["plastic"], "onCeiling": false, "onWall": false, '
                '"onFloor": false, "onObject": true}')
ENGLISH_RESPONSE = CJK_RESPONSE.replace(
    "injecting or抽吸液体", "injecting or drawing fluids")


class FakeAnnotator:
    """Records the prompt it was given on each attempt. No model, no GPU."""

    model_id = "fake-vlm"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, image_paths, prompt):  # noqa: D102 -- matches Annotator
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _render_rec():
    return {"view_paths": [f"view_{i:02d}.png" for i in range(11)],
            "raw_bbox_extents": [1.0, 1.0, 1.0]}


def test_a_chinese_annotation_goes_to_the_repair_path_not_straight_into_the_corpus():
    """[P-3] THE regression this whole extension exists for.

    Under VALIDATOR_VERSION 1 the CJK response passed on attempt 1 and was
    admitted: one generate call, `attempts: 1`, Chinese text in the corpus. The
    repair loop could not help, because nothing had failed.
    """
    from metafind.data.annotate_run import annotate_one

    ann = FakeAnnotator(CJK_RESPONSE, ENGLISH_RESPONSE)
    rec, bad = annotate_one(ann, "uid0", _render_rec())

    assert bad is None and rec is not None
    assert len(ann.prompts) == 2, "the repair attempt was never invoked"
    assert rec["attempts"] == 2
    assert non_english_characters(rec["description"]) == []
    assert "drawing fluids" in rec["description"]


def test_the_repair_prompt_carries_the_language_failure_verbatim():
    """[P-3] The loop is only as good as what it feeds back. A repair prompt
    that does not name the language problem spends the attempt on nothing."""
    from metafind.data.annotate_run import annotate_one

    ann = FakeAnnotator(CJK_RESPONSE, ENGLISH_RESPONSE)
    annotate_one(ann, "uid0", _render_rec())

    original, repair = ann.prompts
    assert repair != original
    assert "REJECTED" in repair
    assert "must be written in English" in repair
    assert "`description`" in repair


def test_an_unrepaired_language_failure_is_quarantined_never_admitted():
    """[L1-ANNOT-EXHAUST] Two attempts, both Chinese. The exhausted item must not
    reach the corpus -- a bound treated as success is a bound that does nothing."""
    from metafind.data.annotate_run import annotate_one

    ann = FakeAnnotator(CJK_RESPONSE, CJK_RESPONSE)
    rec, bad = annotate_one(ann, "uid0", _render_rec())

    assert rec is None and bad is not None
    assert bad["terminated_by"] == "repair_budget"
    assert bad["attempts"] == MAX_ATTEMPTS
    assert "English" in bad["exception_msg"]


def test_a_clean_english_annotation_still_costs_one_attempt():
    """The language rule must not tax the 45,942 records that were already fine."""
    from metafind.data.annotate_run import annotate_one

    ann = FakeAnnotator(ENGLISH_RESPONSE)
    rec, bad = annotate_one(ann, "uid0", _render_rec())
    assert bad is None and rec["attempts"] == 1
    assert len(ann.prompts) == 1


def test_the_repaired_record_is_marked_model_generated_not_human_repaired():
    """[P-5] A model that fixed itself is still a model annotation. Only a human
    edit may claim otherwise."""
    from metafind.data.annotate_run import annotate_one

    rec, _ = annotate_one(FakeAnnotator(CJK_RESPONSE, ENGLISH_RESPONSE),
                          "uid0", _render_rec())
    assert rec["description_source"] == "model"
    assert rec["description_translated_by"] is None


# --- AC-1: no existing record is automatic work [D2a] ----------------------
#
# The hazard these cover, measured on the real corpus 2026-08-21: a bare
# `annotate_run` queued 45,955 records -- the 45,952 accepted legacy-v3 corpus
# AND the 3 legacy-v1 residuals whose fate D0-003 has not decided. A run would
# have rewritten both, settling D0-003 by mutation.

V1_RESIDUALS = ("6c7db00cc164467ebac356a5ca67368b",
                "8a0192eee6fb4140bb3e9696b3dbae5a",
                "a397b648d6eb48d7909d1ee11235e78f")


def _corpus(tmp_path, monkeypatch):
    """A miniature of the real three-population corpus, in a tmp dir."""
    import json as _json

    from metafind.data import annotate_run as R

    ann = tmp_path / "annotations"
    ann.mkdir()
    monkeypatch.setattr(R.paths, "ANNOTATIONS", ann)

    v3 = {"category": "chair", "synset": "chair.n.01", "width": 50.0, "length": 45.0,
          "height": 90.0, "mass": 6.0, "description": "A wooden dining chair.",
          "materials": ["wood"], "onCeiling": False, "onWall": False,
          "onFloor": True, "onObject": True, "volume": 202500.0,
          "dimension_unit": "cm", "mass_unit": "kg", "prompt_version": 3,
          "annotator_model": "Qwen/Qwen2.5-VL-7B-Instruct", "attempts": 1}
    v1 = {"category": "Pole Dancer", "description": "A stylised figure.",
          "dimensions": {"length_m": 2.5, "width_m": 0.5, "height_m": 1.8},
          "materials": ["fabric"], "placement_constraints": ["ceiling_mounted"],
          "annotator_model": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt_version": 1,
          "attempts": 1}

    for uid in ("legacy_a", "legacy_b"):
        (ann / f"{uid}.json").write_text(_json.dumps(dict(v3, uid=uid)))
    for uid in V1_RESIDUALS:
        (ann / f"{uid}.json").write_text(_json.dumps(dict(v1, uid=uid)))

    def digest(uid):
        import hashlib
        return hashlib.sha256((ann / f"{uid}.json").read_bytes()).hexdigest()

    registry = tmp_path / "annotation_provenance.json"
    registry.write_text(_json.dumps({
        "registry_version": 1,
        "populations": [
            {"state": R.ACCEPTED_LEGACY_V3, "prompt_version": 3,
             "records": {u: digest(u) for u in ("legacy_a", "legacy_b")}},
            {"state": R.LEGACY_V1_RESIDUAL, "prompt_version": 1,
             "records": {u: digest(u) for u in V1_RESIDUALS}},
        ]}))
    return R, ann, registry


def test_a_bare_run_queues_nothing_at_all_not_merely_no_legacy_v3(tmp_path, monkeypatch):
    """[AC-1.a] Zero records TOTAL. Protecting the accepted corpus while letting
    the 3 residuals through would resolve D0-003 by rewriting them."""
    R, _, registry = _corpus(tmp_path, monkeypatch)

    candidates = ["legacy_a", "legacy_b", *V1_RESIDUALS]
    states = R.classify_all(candidates, R.load_provenance_registry(registry))

    assert [u for u, st in states.items() if st is None] == []
    assert [u for u, st in states.items() if st == R.UNACCOUNTED] == []


def test_each_population_is_named_not_lumped_together(tmp_path, monkeypatch):
    """[AC-1.c / AC-1.e] The 3 residuals must never be reported as legacy-v3."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    states = R.classify_all(["legacy_a", *V1_RESIDUALS],
                            R.load_provenance_registry(registry))

    assert states["legacy_a"] == R.ACCEPTED_LEGACY_V3
    for uid in V1_RESIDUALS:
        assert states[uid] == R.LEGACY_V1_RESIDUAL
        assert states[uid] != R.ACCEPTED_LEGACY_V3


def test_an_undeclared_legacy_record_is_unaccounted_not_accepted(tmp_path, monkeypatch):
    """[R-A.2] The whole hazard was a missing field being read as a verdict. An
    undeclared record must not become "accepted" by omission -- it becomes
    UNACCOUNTED, which stops the run."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    import json as _json
    stray = _json.loads((ann / "legacy_a.json").read_text()) | {"uid": "stray"}
    (ann / "stray.json").write_text(_json.dumps(stray))

    states = R.classify_all(["stray"], R.load_provenance_registry(registry))
    assert states["stray"] == R.UNACCOUNTED
    assert states["stray"] is not None          # and so it is never silent work


def test_deleting_the_registry_fails_closed_not_open(tmp_path, monkeypatch):
    """A protection that evaporates when its own file goes missing is not a
    protection. With no registry every stored record is UNACCOUNTED, which
    refuses the run -- it does not queue 45,955 assets."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    registry.unlink()

    states = R.classify_all(["legacy_a", *V1_RESIDUALS],
                            R.load_provenance_registry(registry))
    assert set(states.values()) == {R.UNACCOUNTED}
    assert [u for u, st in states.items() if st is None] == []


def test_a_declaration_stops_describing_a_record_that_changed(tmp_path, monkeypatch):
    """The registry declares something about a specific record. If that record's
    schema moves, the declaration no longer covers it."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    import json as _json
    rec = _json.loads((ann / "legacy_a.json").read_text())
    (ann / "legacy_a.json").write_text(_json.dumps(rec | {"prompt_version": 9}))

    states = R.classify_all(["legacy_a"], R.load_provenance_registry(registry))
    assert states["legacy_a"] == R.UNACCOUNTED


def test_a_truncated_record_is_not_treated_as_done(tmp_path, monkeypatch):
    """A half-written sidecar was the original reason `is_complete()` parsed at
    all. It must stay visible, and it must not be silently re-annotated either."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    (ann / "legacy_a.json").write_text('{"category": "cha')

    states = R.classify_all(["legacy_a"], R.load_provenance_registry(registry))
    assert states["legacy_a"] == R.UNACCOUNTED
    assert R.is_complete("legacy_a") is False


def test_a_genuinely_new_asset_is_still_work(tmp_path, monkeypatch):
    """[R-C] The gate removes the accident, not the pipeline. A uid with no
    record at all is the one thing a bare run must still pick up."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    states = R.classify_all(["never_seen"], R.load_provenance_registry(registry))
    assert states["never_seen"] is None


def test_explicit_force_still_reaches_every_record(tmp_path, monkeypatch):
    """[AC-1.b] The negative test, through `build_work_list()` -- the same
    function `main()` calls, so this exercises the real force branch rather than
    restating it. Capability gated, not deleted. No annotation is run here; only
    the work list is computed."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    candidates = ["legacy_a", "legacy_b", *V1_RESIDUALS]

    todo, blocked, _ = R.build_work_list(candidates, force=True,
                                         registry=R.load_provenance_registry(registry))
    assert todo == candidates and len(todo) == 5
    assert blocked == []

    # ...and the same call without force reaches nothing. Same function, same
    # inputs: the only difference is the flag.
    assert R.build_work_list(candidates, force=False,
                             registry=R.load_provenance_registry(registry))[0] == []


def test_a_named_migration_reaches_exactly_the_uids_it_names(tmp_path, monkeypatch):
    """[AC-1.b] `--uids-file <list> --force` is the named-migration form: the same
    explicit capability aimed at one declared population instead of the whole
    corpus. `main()` narrows `candidates` to the file's uids, then hands them to
    the same `build_work_list()`."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    wanted = ["legacy_a", "legacy_b"]            # what --uids-file would contain

    todo, _, _ = R.build_work_list(wanted, force=True,
                                   registry=R.load_provenance_registry(registry))
    assert todo == wanted
    assert not set(todo) & set(V1_RESIDUALS)     # D0-003 stays out of it


def test_the_bare_run_refuses_rather_than_queueing_an_unaccounted_record(tmp_path,
                                                                        monkeypatch):
    """`build_work_list` must hand `main()` a non-empty `blocked` list, which is
    what turns into exit code 3. An unaccounted record neither runs nor vanishes."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    import json as _json
    (ann / "stray.json").write_text(
        _json.dumps(_json.loads((ann / "legacy_a.json").read_text()) | {"uid": "stray"}))

    todo, blocked, _ = R.build_work_list(["legacy_a", "stray"], force=False,
                                         registry=R.load_provenance_registry(registry))
    assert blocked == ["stray"]
    assert todo == []


def test_limit_cannot_resurrect_work_the_gate_removed(tmp_path, monkeypatch):
    """`--limit N` slices the work list AFTER it is built (annotate_run.py, the
    `if args.limit` block). An empty list stays empty at every N."""
    R, _, registry = _corpus(tmp_path, monkeypatch)
    todo, _, _ = R.build_work_list(["legacy_a", "legacy_b", *V1_RESIDUALS], force=False,
                                   registry=R.load_provenance_registry(registry))
    for n in (1, 10, 45955):
        assert todo[:n] == []


def test_the_registry_may_not_declare_a_state_it_has_no_standing_to_declare(tmp_path):
    """`annotated_under_current_contract` is read off the record's contract id,
    and `unaccounted` is the absence of a declaration. A registry asserting
    either would be asserting something it cannot know."""
    import json as _json

    import pytest as _pytest

    from metafind.data import annotate_run as R

    for state in (R.CURRENT_CONTRACT, R.UNACCOUNTED, "whatever"):
        bad = tmp_path / f"{state}.json"
        bad.write_text(_json.dumps({"populations": [
            {"state": state, "prompt_version": 3, "records": {"x": "d"}}]}))
        with _pytest.raises(R.ProvenanceRegistryError, match="provenance state"):
            R.load_provenance_registry(bad)


def test_a_declaration_may_not_move_a_record_between_schema_generations(tmp_path):
    """[AC-1.e] Without this, a registry could declare a legacy-v1 residual as
    `accepted_legacy_v3` and the loader would take it -- exactly the conflation
    AC-1.e forbids, achieved without touching a single record."""
    import json as _json

    import pytest as _pytest

    from metafind.data import annotate_run as R

    bad = tmp_path / "mislabelled.json"
    bad.write_text(_json.dumps({"populations": [
        {"state": R.ACCEPTED_LEGACY_V3, "prompt_version": 1,
         "records": {V1_RESIDUALS[0]: "d"}}]}))
    with _pytest.raises(R.ProvenanceRegistryError, match="schema generations"):
        R.load_provenance_registry(bad)


def test_a_uid_declared_twice_is_refused_rather_than_last_write_wins(tmp_path):
    """Silent last-write-wins would let a second population sweep a residual into
    the accepted corpus. A record belongs to exactly one population."""
    import json as _json

    import pytest as _pytest

    from metafind.data import annotate_run as R

    bad = tmp_path / "dup.json"
    bad.write_text(_json.dumps({"populations": [
        {"state": R.LEGACY_V1_RESIDUAL, "prompt_version": 1, "records": {"dup": "d"}},
        {"state": R.ACCEPTED_LEGACY_V3, "prompt_version": 3, "records": {"dup": "d"}}]}))
    with _pytest.raises(R.ProvenanceRegistryError, match="twice"):
        R.load_provenance_registry(bad)


def test_a_declaration_without_a_records_mapping_is_refused(tmp_path):
    """A declaration that names no digest cannot be checked against the record it
    claims to describe, so it would protect whatever sits at that uid."""
    import json as _json

    import pytest as _pytest

    from metafind.data import annotate_run as R

    bad = tmp_path / "no_records.json"
    bad.write_text(_json.dumps({"populations": [
        {"state": R.ACCEPTED_LEGACY_V3, "prompt_version": 3, "uids": ["x"]}]}))
    with _pytest.raises(R.ProvenanceRegistryError, match="records"):
        R.load_provenance_registry(bad)


def test_a_corrupt_registry_raises_rather_than_loading_empty(tmp_path):
    """Truncated JSON, or a document of the wrong shape, must not degrade into an
    empty registry -- and must not crash with a bare JSONDecodeError either.
    `main()` turns this into a refusal with exit code 3."""
    import pytest as _pytest

    from metafind.data import annotate_run as R

    for text in ('{"populations": [', '[]', '"just a string"', '{"nope": 1}'):
        bad = tmp_path / "corrupt.json"
        bad.write_text(text)
        with _pytest.raises(R.ProvenanceRegistryError):
            R.load_provenance_registry(bad)


def test_a_record_whose_bytes_changed_loses_its_declaration(tmp_path, monkeypatch):
    """The declaration says a SPECIFIC record was classified and re-validated. If
    the bytes move, that statement no longer describes what is on disk -- and a
    same-prompt_version substitution must not inherit it."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    import json as _json

    reg = R.load_provenance_registry(registry)
    assert R.provenance_state("legacy_a", reg) == R.ACCEPTED_LEGACY_V3

    # Same uid, same prompt_version, different content.
    (ann / "legacy_a.json").write_text(_json.dumps({"prompt_version": 3}))
    assert R.provenance_state("legacy_a", reg) == R.UNACCOUNTED


def test_a_sidecar_containing_json_null_is_present_not_absent(tmp_path, monkeypatch):
    """`json.loads("null")` is `None`. Returning that as "no record" would put an
    existing file back in the work queue -- a real hole in AC-1.a, and the reason
    `_record()` guards on `isinstance(rec, dict)`."""
    R, ann, registry = _corpus(tmp_path, monkeypatch)
    reg = R.load_provenance_registry(registry)

    for text in ("null", "[]", '"x"', "42"):
        (ann / "legacy_a.json").write_text(text)
        assert R.provenance_state("legacy_a", reg) == R.UNACCOUNTED, text
        assert R.build_work_list(["legacy_a"], force=False, registry=reg)[0] == []
