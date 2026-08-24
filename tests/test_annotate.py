"""L1 tests for n05_annotate's schema and repair loop.

Covers L1-ANNOT-SCHEMA's truth table, L1-ANNOT-REPAIR, and the prompt property
that a finding asked for and no test previously enforced: the prompt must tell
the annotator the renders are scale-normalised.

None of these need the model. The parts that do -- generation, the C1 loop
running end to end, quarantine on exhaustion -- are exercised separately once
the GPU is free.
"""

from __future__ import annotations

import json

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
    category_relation,
    derive_dimensions,
    lvis_synset,
    non_english_characters,
    parse_annotation,
    validate_annotation,
    LVIS_SYNSETS,
)

# --- v5 fixtures ------------------------------------------------------------
# The anchor and the proportions are now INPUTS to both the prompt and the
# validator, so every test states them explicitly rather than inheriting a
# default that could drift.
ANCHOR = "chair"
# (y, x, z) normalised, largest = 1.0. y is the up axis.
PROPS = (1.0, 0.5, 0.5)


def _valid(**over):
    """A v5 model response: ONE absolute dimension, no synset, no width/length."""
    obj = {
        "category": "dining chair",
        "identity_confirmed": True,
        "description": "A wooden dining chair with a slatted back.",
        "height": 90.0,
        "width_axis": "x",
        "mass": 6.0,
        "materials": ["wood", "fabric"],
        "onCeiling": False, "onWall": False, "onFloor": True, "onObject": False,
    }
    obj.update(over)
    return obj


def _validate(obj, lvis_category=ANCHOR, proportions=PROPS, description=None):
    # [PROMPT_VERSION 8] `description` no longer comes back in the response --
    # it is generated five times separately and chosen by CLIP. Tests that were
    # written about the OTHER fields pass the response's own description
    # through, so they keep asserting what they were written to assert; a test
    # about the description itself passes its own.
    if description is None:
        description = obj.get("description") or "A syringe with a clear barrel."
    return validate_annotation(obj, lvis_category=lvis_category,
                               proportions=proportions, description=description)


# ----------------------------------------------------------------- the prompt


def test_prompt_says_the_renders_are_scale_normalised():
    """[F13] n04 fits every asset to a unit sphere, so size is not in the image.

    Without this sentence the prompt asks for dimensions from pictures that
    contain no scale, and a confident answer looks like a measurement. The
    estimate is a category prior and the prompt has to say so.
    """
    p = build_prompt(11, ANCHOR, PROPS).lower()
    assert "scale-normalised" in p or "scale normalised" in p
    assert "not from the picture" in p
    assert "centimetres" in p


def test_prompt_names_all_four_placement_flags():
    """[PAPER Figure 2] Four INDEPENDENT booleans, not a category choice.

    A flag the model is never shown cannot be answered, and v1's single-choice
    vocabulary is exactly what put `handheld` on a gaming chair.
    """
    p = build_prompt(11, ANCHOR, PROPS)
    for flag in PLACEMENT_FLAGS:
        assert flag in p, flag
    assert "INDEPENDENT" in p


def test_prompt_is_stable():
    """[cache_key: prompt_version] Two calls must not differ, or every asset
    re-annotates and the cache key means nothing."""
    assert build_prompt(11, ANCHOR, PROPS) == build_prompt(11, ANCHOR, PROPS)


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
    a = _validate(_valid())
    assert a.category == "dining chair"
    assert a.synset == "chair.n.01"
    assert a.height == 90.0
    assert (a.on_floor, a.on_object, a.on_wall, a.on_ceiling) == (True, False, False, False)


def test_volume_is_derived_not_asked():
    """[PAPER Figure 2] volume 36000 = width 30 * length 30 * height 40.

    Asking the model for a fourth number invites one that disagrees with the
    other three, with no way afterwards to tell which was wrong.
    """
    # v5: only `height` is asked for. 40 cm tall on a 1.0 : 0.75 : 0.75 mesh
    # gives 30 x 30 x 40, and volume is still the product of the three.
    a = _validate(_valid(height=40), proportions=(1.0, 0.75, 0.75))
    assert (a.width, a.length, a.height) == (30.0, 30.0, 40.0)
    assert a.volume == 36000
    # v6 SHOWS the paper's Figure 2 example, which prints `volume: 36000`, so
    # the word is now in the prompt as an illustration. What must stay true is
    # that the model is not ASKED for it -- checked on the instruction, which is
    # the thing that decides what comes back.
    p = build_prompt(11, ANCHOR, PROPS)
    ask = p.split("Return one JSON object")[1]
    assert '"volume"' not in ask
    assert "do not include `description`, `width`, `length` or `volume`" in p


# [PROMPT_VERSION 8] `description` is not in this list any more: it does not
# come back in the response, it is injected from the CLIP ranking. Its absence
# is covered by `test_no_ranked_description_is_refused` below.
@pytest.mark.parametrize("field", ["category", "identity_confirmed",
                                   "height", "width_axis", "mass", "materials",
                                   "onCeiling", "onWall", "onFloor", "onObject"])
def test_each_required_field_missing_is_rejected(field):
    obj = _valid()
    del obj[field]
    with pytest.raises(AnnotationError, match=field):
        _validate(obj)


@pytest.mark.parametrize("flag", list(PLACEMENT_FLAGS))
def test_placement_flags_must_be_booleans(flag):
    """A string where a boolean belongs is v1's category-choice habit leaking
    back in. `"onFloor": "yes"` is truthy and would pass a bare `if`."""
    with pytest.raises(AnnotationError, match="true or false"):
        _validate(_valid(**{flag: "yes"}))


def test_all_four_false_is_accepted():
    """An abstract shape belongs nowhere in particular, and that is an answer.

    v1 demanded a positive value and `unconstrained` absorbed 30.7% of the
    corpus as a result.
    """
    a = _validate(_valid(onCeiling=False, onWall=False,
                                   onFloor=False, onObject=False))
    assert not any((a.on_ceiling, a.on_wall, a.on_floor, a.on_object))


def test_all_four_true_is_accepted():
    """The flags are independent, so every combination is representable."""
    a = _validate(_valid(onCeiling=True, onWall=True,
                                   onFloor=True, onObject=True))
    assert all((a.on_ceiling, a.on_wall, a.on_floor, a.on_object))


def test_metres_are_rejected_with_the_unit_named():
    """v1's failure inverted. That schema was metres and kept receiving
    millimetres; this one is centimetres and will receive metres -- 0.9 for a
    chair rather than 90. The message must say CENTIMETRES, because it goes
    straight back into the repair prompt.
    """
    with pytest.raises(AnnotationError, match="CENTIMETRES"):
        _validate(_valid(height=0.9))


@pytest.mark.parametrize("bad", ["0.5", True, 0.0, None])
def test_bad_dimensions_are_rejected(bad):
    with pytest.raises(AnnotationError):
        _validate(_valid(height=bad))


@pytest.mark.parametrize("bad", ["2.5", True, 0.0, None])
def test_bad_mass_is_rejected(bad):
    with pytest.raises(AnnotationError):
        _validate(_valid(mass=bad))


def test_the_model_cannot_supply_a_synset_at_all():
    """[Design Decision 4] v4 read `synset` from the response and checked SHAPE
    only, so `robot.n.01` on a teapot passed. v5 does not read the field: an
    invented one in the response is ignored, and the stored id is LVIS's."""
    a = _validate(_valid(synset="robot.n.01"))
    assert a.synset == lvis_synset(ANCHOR) == "chair.n.01"


def test_the_synset_follows_the_model_and_records_where_it_came_from():
    """[USER DECISION `U-SY`, 2026-08-23] Reversed. This test used to assert the
    opposite -- that the synset follows the ANCHOR -- and that rule produced
    `category: "centipede"` beside `synset: "snake.n.01"` on an asset whose LVIS
    label was simply wrong. Two fields of one schema contradicting each other.

    Nothing is minted: WordNet answers, or a recorded fallback does. Which rung
    answered is stored, because an id alone cannot distinguish "LVIS's own
    table" from "we gave up and kept the anchor".
    """
    from metafind.data.annotate import resolve_synset

    # An LVIS term: the table wins, it is the most authoritative source here.
    assert resolve_synset("toy", "toy") == ("toy.n.03", "lvis_table")

    # Not an LVIS term, but WordNet knows the phrase.
    syn, src = resolve_synset("centipede", "snake")
    assert (syn, src) == ("centipede.n.01", "wordnet_phrase")

    # A compound: English is head-final, so the LAST word resolves it. This is
    # deliberately conservative -- "toy dinosaur" lands on the dinosaur, not on
    # the toy, and never on an invented `toy_dinosaur.n.01`.
    syn, src = resolve_synset("toy dinosaur", "toy")
    assert src == "wordnet_head" and syn.startswith("dinosaur.n.")

    # End to end through the validator, on the case that forced the change.
    a = _validate(_valid(category="centipede"), lvis_category="snake")
    assert a.category == "centipede"
    assert a.synset == "centipede.n.01"
    assert a.synset_source == "wordnet_phrase"


def test_every_synset_comes_from_the_lvis_table():
    """[UPSTREAM FACT] 1,156 entries, all copied from the LVIS v1 release."""
    assert len(LVIS_SYNSETS) == 1156
    assert lvis_synset("horned cow") == "bull.n.11"      # via LVIS's synonym table
    with pytest.raises(AnnotationError, match="not one of"):
        lvis_synset("a category LVIS has never heard of")


def test_an_invented_synset_can_no_longer_reach_the_corpus():
    """v4's own comment admitted "a well-formed but invented synset passes
    here". `notathing.n.07` was admissible. Under v5 the field is not read."""
    a = _validate(_valid(synset="notathing.n.07"))
    assert a.synset == "chair.n.01"
    # Still true under `U-SY`: the model's `synset` field is never read on the
    # anchored path. What changed is that the id is now resolved from the
    # model's CATEGORY, which is evidence about the object, rather than from its
    # `synset` field, which is a free-text guess at a database key.
    #
    # The fixture's category is "dining chair", which is not an LVIS term, so
    # the head noun answers -- and WordNet independently lands on the same
    # `chair.n.01` LVIS's own table holds for the anchor. That agreement is
    # worth asserting: the two authorities are not being played off each other.
    assert a.synset_source == "wordnet_head"
    assert LVIS_SYNSETS["chair"] == "chair.n.01"


def test_a_single_string_is_accepted_where_a_list_is_expected():
    """Models return "wood" for a one-material object constantly. Spending a
    repair attempt on that would be spending it on nothing."""
    a = _validate(_valid(materials="wood"))
    assert a.materials == ["wood"]


def test_material_spelling_variants_are_folded():
    """[MATERIAL_SYNONYMS] v1 emitted `metal` 34.3% AND `metallic` 10.7% as
    separate tokens; a text encoder reads those as two different materials."""
    a = _validate(_valid(materials=["Metallic", "metal", "WOODEN"]))
    assert a.materials == ["metal", "wood"]


def test_nothing_is_dropped_from_materials():
    """Only spellings are merged. Deciding what "is not a material" is a
    judgement the paper does not license, so `textured` survives."""
    a = _validate(_valid(materials=["textured", "plastic"]))
    assert a.materials == ["textured", "plastic"]


def test_empty_materials_is_rejected():
    with pytest.raises(AnnotationError):
        _validate(_valid(materials=[]))


# ------------------------------------------------------ L1-ANNOT-REPAIR


def test_repair_prompt_differs_and_names_the_error():
    """[L1-ANNOT-REPAIR] Resending the same prompt reproduces the same mistake.

    A loop whose retry is identical to its first attempt is not a repair loop;
    it is a delay with an attempt counter.
    """
    original = build_prompt(11, ANCHOR, PROPS)
    err = "`dimensions.height_m` = 900 is outside 0.001-100.0 m"
    repair = build_repair_prompt(original, err, '{"height_m": 900}')
    assert repair != original
    assert err in repair
    assert "900" in repair


def test_repair_prompt_refuses_to_be_built_without_an_error():
    with pytest.raises(ValueError, match="just a resend"):
        build_repair_prompt(build_prompt(11, ANCHOR, PROPS), "", "whatever")


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
    p = build_prompt(11, ANCHOR, PROPS)
    assert "ENGLISH" in p
    for field in ("category", "description", "materials"):
        assert field in p
    lower = p.lower()
    for script in ("chinese", "japanese", "korean", "cyrillic", "arabic"):
        assert script in lower, script


def test_the_prompt_still_permits_accented_latin():
    """[P-1] The instruction must not read as "ASCII only" -- "Pokémon" is
    correct English spelling and 7 records legitimately use accents."""
    assert "Pok\u00e9mon" in build_prompt(11, ANCHOR, PROPS) or "Pokémon" in build_prompt(11, ANCHOR, PROPS)


def test_the_prompt_version_moved_because_the_prompt_changed():
    """[P-1] v4 asks for something v3 did not. Pretending they are one contract
    is what makes a corpus unfalsifiable later."""
    assert PROMPT_VERSION == 8


def test_editing_the_prompt_moves_the_contract_id():
    """The old `test_prompt_is_stable` only checked that two calls agree, so a
    prompt EDIT was invisible to the whole suite. The contract fingerprint is
    what actually pins the text."""
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = a.build_prompt
    try:
        a.build_prompt = lambda n, c, p: original(n, c, p) + "\nAnswer in rhyme."
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
        _validate(_valid(**{field: "凹陷"}))
    assert field in str(exc.value)
    assert "English" in str(exc.value)


def test_a_non_english_material_is_refused_by_validation():
    with pytest.raises(AnnotationError, match="English"):
        _validate(_valid(materials=["wood", "塑料"]))


def test_the_language_error_tells_the_model_what_to_do():
    """The message is fed back verbatim by build_repair_prompt(), so "invalid
    schema" would spend a repair attempt on nothing."""
    with pytest.raises(AnnotationError) as exc:
        _validate(_valid(description="a device 由塑料制成"))
    msg = str(exc.value)
    assert "must be written in English" in msg
    assert "Chinese" in msg and "Pok" in msg      # names the rule AND the exception
    assert "由" in msg or "塑" in msg               # names the offending characters


def test_an_accented_annotation_still_validates_end_to_end():
    ann = _validate(_valid(category="Pokémon figure",
                                     description="A Poké Ball replica.",
                                     materials=["plastic"]))
    assert ann.category == "Pokémon figure"


# ====================================================================== P-5
# The record can say where its text came from.

def test_a_model_record_declares_itself_model_generated():
    """[P-5] "model-generated" must be a RECORDED fact, not the absence of a
    note -- otherwise a hand-repaired record is indistinguishable from an
    untouched one."""
    rec = _validate(_valid()).as_record("Qwen/Qwen2.5-VL-7B-Instruct")
    assert rec["description_source"] == "model"
    assert rec["description_original"] is None
    assert rec["description_translation_authority"] is None
    assert rec["description_translated_by"] is None


def test_a_record_carries_all_three_contract_axes_and_a_fingerprint():
    """[P-5] prompt_version alone cannot express "which validator admitted
    this"."""
    rec = _validate(_valid()).as_record("m")
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

CJK_RESPONSE = ('{"category": "syringe", "identity_confirmed": true, '
                '"height": 2, "width_axis": "x", "mass": 0.1, "description": '
                '"a medical device used for injecting or抽吸液体", '
                '"materials": ["plastic"], "onCeiling": false, "onWall": false, '
                '"onFloor": false, "onObject": true}')
ENGLISH_RESPONSE = CJK_RESPONSE.replace(
    "injecting or抽吸液体", "injecting or drawing fluids")

UNANCHORED_RESPONSE = json.dumps({
    # [PROMPT_VERSION 7] No `identity_confirmed` and no `description`: there is
    # no anchor to confirm, and the description arrives from the CLIP ranking.
    "category": "syringe", "synset": "syringe.n.01",
    "height": 15.0, "width_axis": "x", "mass": 0.02,
    "materials": ["plastic"],
    "onCeiling": False, "onWall": False, "onFloor": False, "onObject": True,
})



class FakeAnnotator:
    """Records the prompt it was given on each attempt. No model, no GPU.

    v6 makes TWO kinds of call per asset: one blind turn, then one or more
    anchored attempts. `responses` stays the list of ANNOTATION replies, so
    every test still reads as "attempt 1 returns this, attempt 2 returns that";
    the blind turn is served separately from `blind`. Prepending a dummy to
    every call site instead would have made each list mean something different
    from what it says.
    """

    model_id = "fake-vlm"

    def __init__(self, *responses: str,
                 description: str = "A syringe with a clear barrel.") -> None:
        self.responses = list(responses)
        self.description = description
        self.prompts: list[str] = []             # every call, in order
        self.annotation_prompts: list[str] = []  # the structured ones only
        self.description_prompts: list[str] = []  # the sampled draws

    def generate(self, image_paths, prompt, *, sample=False, seed=None):  # noqa: D102
        self.prompts.append(prompt)
        # [PROMPT_VERSION 8] Description draws are the sampled calls. Identified
        # by what they ask rather than by call order, so a test that changes the
        # number of attempts still works.
        if sample or "Return one JSON object" not in prompt:
            self.description_prompts.append(prompt)
            return self.description
        self.annotation_prompts.append(prompt)
        return self.responses.pop(0)


def _render_rec():
    return {"view_paths": [f"view_{i:02d}.png" for i in range(11)],
            "raw_bbox_extents": [1.0, 1.0, 1.0]}


def test_no_ranked_description_is_refused():
    """[PROMPT_VERSION 8] The description is injected, so its absence is a
    caller bug rather than a model failure -- and must still not pass."""
    for bad in (None, "", "   "):
        with pytest.raises(AnnotationError, match="no ranked description"):
            validate_annotation(_valid(), lvis_category=ANCHOR,
                                proportions=PROPS, description=bad)


def test_a_chinese_description_candidate_is_dropped_before_it_can_be_ranked():
    """[P-3] THE regression this whole extension exists for, moved to where the
    description now lives.

    Under `VALIDATOR_VERSION 1` a CJK response passed on attempt 1 and was
    admitted: Chinese text in the corpus. v2 made it a validation failure that
    the repair loop could fix, because the description came back in the
    structured response.

    In v8 it does not. Re-prompting the structured call cannot change a
    description that came from a different call, so the repair loop would spend
    both attempts on a field it cannot reach and quarantine a usable asset. The
    rule therefore moved to the candidate stage: a non-English draw is dropped,
    and four of five candidates are still four candidates.
    """
    from metafind.data import annotate_run as R

    draws = {"n": 0}

    class MixedLanguageDraws(FakeAnnotator):
        def generate(self, image_paths, prompt, *, sample=False, seed=None):
            if sample:
                draws["n"] += 1
                if draws["n"] in (2, 5):
                    return "\u91cd\u5316\u5b78\u88dd\u7f6e\uff0c\u900f\u660e\u7ba1\u8eab"
                return f"A syringe with a clear barrel, draw {draws['n']}."
            return super().generate(image_paths, prompt)

    seen = []

    def fake_rank(views, candidates):
        seen.append(list(candidates))
        return candidates[0], [{"text": c, "clip_score": 0.3 - i * 0.01, "rank": i}
                               for i, c in enumerate(candidates)]

    R.rank_descriptions = fake_rank
    ann = MixedLanguageDraws(ENGLISH_RESPONSE)
    rec, bad = R.annotate_one(ann, "uid0", _render_rec(),
                              lvis_category="syringe", proportions=PROPS)

    assert bad is None and rec is not None
    assert len(seen[0]) == 3, f"the two CJK draws should not reach the ranker: {seen[0]}"
    assert non_english_characters(rec["description"]) == []
    assert rec["description_candidates_rejected_non_english"] == 2, (
        "the rejection must be COUNTED -- a model that keeps answering in "
        "Chinese is telling the USER something, and a silent drop hides it"
    )


def test_every_candidate_non_english_quarantines_rather_than_admitting_one():
    """The other side: with nothing English left there is no description to
    rank, and admitting a CJK one is the exact defect `P-3` exists for."""
    from metafind.data import annotate_run as R

    class AllChinese(FakeAnnotator):
        def generate(self, image_paths, prompt, *, sample=False, seed=None):
            if sample:
                return "\u91cd\u5316\u5b78\u88dd\u7f6e"
            return super().generate(image_paths, prompt)

    rec, bad = R.annotate_one(AllChinese(ENGLISH_RESPONSE), "uid0", _render_rec(),
                              lvis_category="syringe", proportions=PROPS)
    assert rec is None and bad is not None
    assert bad["exception_type"] == "NoDescriptionCandidates"
    assert "non-English" in bad["exception_msg"]


def test_a_clean_english_annotation_still_costs_one_structured_attempt():
    """The language rule must not tax the records that were already fine."""
    from metafind.data import annotate_run as R

    R.rank_descriptions = lambda views, c: (c[0], [{"text": t, "clip_score": 0.3,
                                                    "rank": i} for i, t in enumerate(c)])
    ann = FakeAnnotator(ENGLISH_RESPONSE)
    rec, bad = R.annotate_one(ann, "uid0", _render_rec(),
                              lvis_category="syringe", proportions=PROPS)
    assert bad is None and rec["attempts"] == 1
    assert len(ann.annotation_prompts) == 1
    assert len(ann.description_prompts) == 5, "five independent draws, per main.tex:677"
    assert rec["description_source"] == "model"
    assert rec["description_translated_by"] is None


def test_a_record_carries_all_three_contract_axes_and_a_fingerprint():
    """[P-5] prompt_version alone cannot express "which validator admitted
    this"."""
    rec = _validate(_valid()).as_record("m")
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

CJK_RESPONSE = ('{"category": "syringe", "identity_confirmed": true, '
                '"height": 2, "width_axis": "x", "mass": 0.1, "description": '
                '"a medical device used for injecting or抽吸液体", '
                '"materials": ["plastic"], "onCeiling": false, "onWall": false, '
                '"onFloor": false, "onObject": true}')
ENGLISH_RESPONSE = CJK_RESPONSE.replace(
    "injecting or抽吸液体", "injecting or drawing fluids")

UNANCHORED_RESPONSE = json.dumps({
    # [PROMPT_VERSION 7] No `identity_confirmed` and no `description`: there is
    # no anchor to confirm, and the description arrives from the CLIP ranking.
    "category": "syringe", "synset": "syringe.n.01",
    "height": 15.0, "width_axis": "x", "mass": 0.02,
    "materials": ["plastic"],
    "onCeiling": False, "onWall": False, "onFloor": False, "onObject": True,
})



class FakeAnnotator:
    """Records the prompt it was given on each attempt. No model, no GPU.

    v6 makes TWO kinds of call per asset: one blind turn, then one or more
    anchored attempts. `responses` stays the list of ANNOTATION replies, so
    every test still reads as "attempt 1 returns this, attempt 2 returns that";
    the blind turn is served separately from `blind`. Prepending a dummy to
    every call site instead would have made each list mean something different
    from what it says.
    """

    model_id = "fake-vlm"

    def __init__(self, *responses: str,
                 description: str = "A syringe with a clear barrel.") -> None:
        self.responses = list(responses)
        self.description = description
        self.prompts: list[str] = []             # every call, in order
        self.annotation_prompts: list[str] = []  # the structured ones only
        self.description_prompts: list[str] = []  # the sampled draws

    def generate(self, image_paths, prompt, *, sample=False, seed=None):  # noqa: D102
        self.prompts.append(prompt)
        # [PROMPT_VERSION 8] Description draws are the sampled calls. Identified
        # by what they ask rather than by call order, so a test that changes the
        # number of attempts still works.
        if sample or "Return one JSON object" not in prompt:
            self.description_prompts.append(prompt)
            return self.description
        self.annotation_prompts.append(prompt)
        return self.responses.pop(0)


def _render_rec():
    return {"view_paths": [f"view_{i:02d}.png" for i in range(11)],
            "raw_bbox_extents": [1.0, 1.0, 1.0]}


def test_the_repaired_record_is_marked_model_generated_not_human_repaired():
    """[P-5] A model that fixed itself is still a model annotation. Only a human
    edit may claim otherwise."""
    from metafind.data.annotate_run import annotate_one

    rec, _ = annotate_one(FakeAnnotator(CJK_RESPONSE, ENGLISH_RESPONSE),
                          "uid0", _render_rec(),
                          lvis_category="syringe", proportions=PROPS)
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


# ====================================================================== v5
# PROMPT_VERSION 5 / VALIDATOR_VERSION 3 / SCHEMA_VERSION 3.
#
# The rules this task exists to add. Each test names the failure it prevents,
# because "the category is now anchored" is a claim, and a claim with no test
# under it is what let 45,952 records describe the wrong object.

def test_the_prompt_states_the_catalogued_identity():
    """[Design Decision 1] v4 never told the model what the object was, so a
    7B model asked to IDENTIFY a 224x224 render collapsed onto `toy`."""
    p = build_prompt(11, "pinecone", PROPS)
    assert "pinecone" in p
    assert "Objaverse-LVIS" in p


def test_the_prompt_hands_over_the_exact_proportions_and_forbids_re_estimating():
    """[Design Decision 3] Three unknowns become one. The mesh is exact; the
    renders are scale-normalised and carry no absolute size at all."""
    p = build_prompt(11, ANCHOR, (1.0, 0.25, 0.4))
    assert "1.000 : 0.250 : 0.400" in p
    assert "EXACT" in p
    assert "Do not re-estimate" in p


def test_the_prompt_asks_for_refinement_and_forbids_replacement():
    p = build_prompt(11, "toy", PROPS)
    assert "MORE SPECIFIC" in p
    assert "NEVER name a different object" in p


def test_the_prompt_asks_only_one_absolute_measurement():
    """v5 must not reinstate the three-guess habit through the back door."""
    # Asserted on the field list, not on the whole prompt: v6 shows Figure 2's
    # worked example, which necessarily prints all three dimensions. The example
    # is what the answer should LOOK like; the field list is what is asked for.
    p = build_prompt(11, ANCHOR, PROPS)
    ask = p.split("Return one JSON object")[1]
    assert '"height"' in ask
    assert '"width":' not in ask and '"length":' not in ask


def test_identity_confirmed_false_is_ADMITTED_not_quarantined():
    """[TASK.md R-B] THE rule of this run. LVIS's own error rate has never been
    measured; a corpus filtered on an unmeasured threshold cannot measure it.
    Flag, do not filter.
    """
    a = _validate(_valid(identity_confirmed=False))
    assert a.identity_confirmed is False
    rec = a.as_record("m")
    assert rec["identity_confirmed"] is False


@pytest.mark.parametrize("bad", ["true", "yes", 1, 0, None])
def test_identity_confirmed_must_be_a_real_boolean(bad):
    """A coerced answer is indistinguishable afterwards from a real one."""
    with pytest.raises(AnnotationError, match="identity_confirmed"):
        _validate(_valid(identity_confirmed=bad))


@pytest.mark.parametrize("bad", ["y", "width", "", None, 1, "xz"])
def test_width_axis_must_name_one_of_the_two_horizontal_axes(bad):
    with pytest.raises(AnnotationError, match="width_axis"):
        _validate(_valid(width_axis=bad))


@pytest.mark.parametrize("ok", ["x", "X", " x ", "Z"])
def test_width_axis_case_and_padding_are_normalised_not_refused(ok):
    """Deliberate: "X" is the same answer as "x". The stored value is
    canonical, so the record never carries two spellings of one axis."""
    a = _validate(_valid(width_axis=ok))
    assert a.width_axis == ok.strip().lower()


def test_width_and_length_come_from_the_mesh_not_from_the_model():
    """[Design Decision 3] The model may state them; they are ignored. A model
    that also emits width/length must not be able to contradict the mesh."""
    a = _validate(_valid(height=100.0, width=999.0, length=999.0),
                  proportions=(1.0, 0.6, 0.3))
    assert (a.height, a.width, a.length) == (100.0, 60.0, 30.0)


def test_the_horizontal_axis_assignment_is_the_models_call():
    """x vs z -> width vs length is NOT determined by the mesh: the canonical
    facing of an Objaverse asset is unknown, so this stays a judgement."""
    kw = dict(proportions=(1.0, 0.6, 0.3))
    ax = _validate(_valid(height=100.0, width_axis="x"), **kw)
    az = _validate(_valid(height=100.0, width_axis="z"), **kw)
    assert (ax.width, ax.length) == (60.0, 30.0)
    assert (az.width, az.length) == (30.0, 60.0)


def test_height_is_the_y_axis():
    """[OBSERVED DATA] Y-up, reproduced independently for this task over 1,365
    unambiguously tall and 962 unambiguously flat assets. A wrong axis here
    silently rescales every dimension in the corpus and raises no error."""
    assert derive_dimensions(10.0, (1.0, 0.5, 0.25), "x") == (5.0, 2.5)
    # a flat mesh: y is the SHORT axis, so the same height yields a big footprint
    w, l = derive_dimensions(10.0, (0.2, 1.0, 0.8), "x")
    assert (round(w, 3), round(l, 3)) == (50.0, 40.0)


def test_a_degenerate_up_axis_is_refused_not_divided_by():
    with pytest.raises(AnnotationError, match="up axis"):
        derive_dimensions(10.0, (0.0, 1.0, 1.0), "x")


def test_a_derived_dimension_outside_the_bounds_blames_the_height():
    """The model chose one number; the error must point at that number."""
    with pytest.raises(AnnotationError, match="height"):
        _validate(_valid(height=1.0), proportions=(1.0, 1e-6, 1.0))


@pytest.mark.parametrize("anchor,category,expected", [
    ("chair", "chair", "exact"),
    ("chair", "Chair", "exact"),                       # case is not a refinement
    ("car (automobile)", "car", "exact"),              # LVIS disambiguator is apparatus
    ("toy", "toy dinosaur", "refined"),
    ("chair", "wooden dining chair", "refined"),
    ("motor vehicle", "coffee machine", "divergent"),
    ("motor vehicle", "pickup truck", "divergent"),    # a real refinement, unprovable here
])
def test_category_relation_is_computed_and_recorded(anchor, category, expected):
    assert category_relation(anchor, category) == expected


def test_a_divergent_category_is_recorded_not_rejected():
    """[IMPLEMENTATION CHOICE -- UNRESOLVED ENFORCEMENT] "motor vehicle" ->
    "pickup truck" is a legitimate downward refinement and "motor vehicle" ->
    "coffee machine" is a lateral replacement, and no authoritative source in
    this project separates them. Rejecting `divergent` would throw away the
    good ones; accepting it silently would hide the bad ones. So it is
    MEASURED -- the same reasoning as identity_confirmed.
    """
    a = _validate(_valid(category="coffee machine"), lvis_category="motor vehicle")
    assert a.category_relation == "divergent"
    assert a.as_record("m")["category_relation"] == "divergent"


def test_the_record_carries_the_anchor_so_disagreement_stays_measurable():
    """Without the anchor on disk, nobody can afterwards tell a refinement from
    a replacement, and the entire point of anchoring is lost."""
    rec = _validate(_valid(category="dining chair")).as_record("m")
    assert rec["lvis_category"] == "chair"
    assert rec["category"] == "dining chair"
    assert rec["synset"] == "chair.n.01"


def test_all_three_contract_axes_moved_for_v8():
    # SCHEMA_VERSION 4 -> 5 with `synset_source` (`U-SY`, 2026-08-23).
    assert (PROMPT_VERSION, VALIDATOR_VERSION, SCHEMA_VERSION) == (8, 4, 5)
    assert annotation_contract_id().startswith("metafind_annot_v8@")


def test_swapping_the_synset_table_moves_the_contract_id():
    """The table decides every `synset` in the corpus. Changing it without
    moving the fingerprint would leave two different corpora indistinguishable."""
    import metafind.data.annotate as a

    before = annotation_contract_id()
    original = dict(a.LVIS_SYNSETS)
    try:
        a.LVIS_SYNSETS["chair"] = "stool.n.01"
        assert annotation_contract_id() != before
    finally:
        a.LVIS_SYNSETS.clear()
        a.LVIS_SYNSETS.update(original)
    assert annotation_contract_id() == before


def test_the_synset_table_covers_the_whole_lvis_vocabulary():
    """[Design Decision 4] 1,156 categories, 1,156 entries, no gaps. A gap
    would mean some assets get a looked-up synset and others a guessed one."""
    import json as _json
    from pathlib import Path as _Path
    meta = _json.loads(
        (_Path("data/datasets/objaverse-lvis/objaverse_lvis_metadata.json")).read_text()
    )
    assert set(meta["all_keys"]) == set(LVIS_SYNSETS)


def test_the_bakeoff_cannot_write_into_the_corpus(tmp_path, monkeypatch):
    """`SPEC_M1` §4: `data/outputs/annotations/` holds 0 files at every point in M1.

    The bake-off writes 100 records per arm. Without redirection they land in
    the directory the full run owns, and afterwards **nothing distinguishes an
    experiment from the corpus** -- the records carry no field saying which they
    are, so the contamination is not recoverable by inspection.

    Expected truth is `paths.ANNOTATIONS` itself, read through the same module
    the runner uses, so the test cannot pass by agreeing with a stale copy.
    """
    from metafind import paths
    from metafind.data import annotate_run as R

    monkeypatch.setattr(R, "_ARM_ROOT", None)
    monkeypatch.setattr(paths, "ANNOTATIONS", tmp_path / "corpus")
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path)

    assert R.out_root() == tmp_path / "corpus"
    assert R.sidecar_path("u") == tmp_path / "corpus" / "u.json"

    arm = R.use_arm("qwen38_27b")
    assert arm == tmp_path / "bakeoff" / "qwen38_27b" / "annotations"
    assert R.sidecar_path("u") == arm / "u.json"
    assert paths.ANNOTATIONS not in R.sidecar_path("u").parents

    # Traversal, absolute paths and the two directory shorthands are refused by
    # NAME, before any resolution -- a check that only compared resolved paths
    # would depend on where `data/outputs` happens to be linked today.
    monkeypatch.setattr(R, "_ARM_ROOT", None)
    for bad in ["", ".", "..", "../annotations", "a/b", "/tmp"]:
        with pytest.raises(ValueError, match="plain directory name"):
            R.use_arm(bad)
    assert R._ARM_ROOT is None, "a refused arm must not leave the writer redirected"
    assert R.out_root() == tmp_path / "corpus"


def test_the_blind_turn_never_leaks_the_answer_it_is_meant_to_test():
    """[PROMPT_VERSION 6] Turn 1 exists to make `identity_confirmed` falsifiable.

    `W-7`: we feed the LVIS label in and ask the model to confirm it, and
    **there is no ground truth telling us a `true` is really true**. A guess made
    before the label exists either matches the Objaverse-LVIS catalogue or does
    not, and the catalogue is a source this project did not write -- so it is the
    one automatic accuracy signal the bake-off has.

    All of which collapses if the anchor appears in turn 1. What must not appear
    is the IDENTITY -- the category, the catalogue it came from -- and the
    schema's *structure*, because a turn 1 that knows it is filling a form
    becomes a form-filling task rather than a look-and-say one.

    `materials` is deliberately NOT on that list even though it is a schema
    field. Turn 1 asks what the object appears to be made of because that is
    part of looking at an object, and the word carries no information about
    WHICH object it is. This test was written with `materials` included, caught
    the blind prompt, and the assertion was the thing that was wrong -- recorded
    here so the next reader does not re-add it.
    """
    from metafind.data.annotate import build_blind_prompt, build_prompt

    blind = build_blind_prompt(11)
    anchored = build_prompt(11, "syringe", PROPS, blind_guess="a hypodermic needle")

    for leak in ["syringe", "Objaverse-LVIS", "catalogued", "correct unless",
                 "JSON", "onCeiling", "onFloor", "synset", "proportions",
                 "centimetres", "mass"]:
        assert leak not in blind, f"turn 1 leaks {leak!r}, which is what it must not know"
    assert "OBJECT:" in blind

    # And the anchored turn must actually carry the blind answer back, or the
    # model is simply being asked twice rather than being asked to reconsider.
    assert "a hypodermic needle" in anchored
    assert "syringe" in anchored


def test_one_failed_description_draw_does_not_cost_the_asset():
    """[PROMPT_VERSION 7] A candidate that does not come back is one fewer
    candidate, not one fewer asset.

    v6's blind turn is gone -- v7 supplies no identity at all, so the whole run
    is blind and there is nothing to be blind about separately. What that test
    protected still applies here: dropping an asset because a MEASUREMENT threw
    would remove it for a reason that has nothing to do with the corpus, and
    would do it selectively -- the assets the model struggles with most.

    Only when EVERY draw fails is the asset quarantined, because then there is
    no description to rank and the record cannot be completed.
    """
    from metafind.data import annotate_run as R

    calls = {"n": 0}

    class FlakyDraws(FakeAnnotator):
        def generate(self, image_paths, prompt, *, sample=False, seed=None):
            if sample:
                calls["n"] += 1
                if calls["n"] in (2, 4):
                    raise RuntimeError("CUDA hiccup")
                return f"A syringe, draw {calls['n']}."
            return super().generate(image_paths, prompt)

    ann = FlakyDraws(UNANCHORED_RESPONSE)
    ranked = []

    def fake_rank(views, candidates):
        ranked.append(list(candidates))
        return candidates[0], [{"text": c, "clip_score": 0.3 - i * 0.01, "rank": i}
                               for i, c in enumerate(candidates)]

    R.rank_descriptions = fake_rank
    rec, bad = R.annotate_one(ann, "uid0", _render_rec(),
                              lvis_category=None, proportions=PROPS)

    assert bad is None and rec is not None, "two failed draws must not lose the asset"
    assert len(ranked[0]) == 3, f"3 of 5 draws survived, got {ranked[0]}"
    assert len(rec["description_candidates"]) == 3
    assert rec["description"] == "A syringe, draw 1."


def test_every_failed_draw_quarantines_rather_than_inventing_a_description():
    """The other side of the same rule: with no candidate there is nothing to
    rank, and writing a record with an empty description would put a blank
    field into the corpus under the same schema as a real one."""
    from metafind.data import annotate_run as R

    class AllDrawsFail(FakeAnnotator):
        def generate(self, image_paths, prompt, *, sample=False, seed=None):
            if sample:
                raise RuntimeError("CUDA hiccup")
            return super().generate(image_paths, prompt)

    rec, bad = R.annotate_one(AllDrawsFail(UNANCHORED_RESPONSE), "uid0", _render_rec(),
                              lvis_category=None, proportions=PROPS)
    assert rec is None and bad is not None
    assert bad["exception_type"] == "NoDescriptionCandidates"


# --- C4: CUDA OOM is not just another exception ----------------------------
# The batch fallback answers a failed draw by issuing five more prefills. That
# is right for a per-asset failure and catastrophic for an OOM, and MEASURED
# 2026-08-24 the batched draw peaks at 31,932 MiB of 32,607 -- 675 MiB of
# headroom -- so OOM is the likeliest thing that `except` will ever catch.

def _oom() -> RuntimeError:
    """The shape torch raises when the allocator fails from inside a kernel."""
    return RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
        "capacity of 31.36 GiB of which 512.00 MiB is free.")


def test_an_out_of_memory_error_is_recognised_by_message_and_by_type():
    """Both branches, because neither alone is reliable.

    torch raises `torch.cuda.OutOfMemoryError` for the allocator's own
    failures, but an OOM surfacing from a kernel or a cuBLAS call arrives as a
    plain `RuntimeError` whose message is the only evidence. Matching on type
    alone misses the second; matching on the string alone breaks the moment the
    typed error stops carrying that wording.
    """
    from metafind.data.annotate_run import _is_cuda_oom

    assert _is_cuda_oom(_oom()) is True
    assert _is_cuda_oom(RuntimeError("OUT OF MEMORY")) is True, "match is case-insensitive"

    import torch
    typed = getattr(torch.cuda, "OutOfMemoryError", None)
    if typed is not None:
        assert _is_cuda_oom(typed("no message that mentions the words")) is True


def test_an_ordinary_failure_is_not_treated_as_an_out_of_memory():
    """The guard must stay narrow.

    If everything counted as an OOM the cache would be flushed on every
    malformed prompt and the distinction recorded in `draw_mode` would be
    worthless -- which is the whole reason the OOM branch exists separately.
    """
    from metafind.data.annotate_run import _is_cuda_oom

    assert _is_cuda_oom(RuntimeError("CUDA hiccup")) is False
    assert _is_cuda_oom(ValueError("the prompt was malformed")) is False


class _BatchOOMs(FakeAnnotator):
    """Accepts `n`, OOMs on the batched draw, succeeds one at a time."""

    def __init__(self, *responses: str, single_fails: int = 0, **kw) -> None:
        super().__init__(*responses, **kw)
        self.singles = 0
        self.single_fails = single_fails

    def generate(self, image_paths, prompt, *, sample=False, seed=None, n=1):
        if sample and n > 1:
            raise _oom()
        if sample:
            self.singles += 1
            if self.singles <= self.single_fails:
                raise _oom()
            return f"A syringe, draw {self.singles}."
        return super().generate(image_paths, prompt)


def _rank_first(monkeypatch):
    from metafind.data import annotate_run as R

    monkeypatch.setattr(R, "rank_descriptions", lambda views, c: (
        c[0], [{"text": t, "clip_score": 0.3 - i * 0.01, "rank": i}
               for i, t in enumerate(c)]))


def test_an_out_of_memory_batch_frees_the_cache_before_retrying(monkeypatch):
    """[C4] Free first, THEN fall back.

    Without this the five sequential prefills run on a card still holding the
    workspace that just failed, so the recovery inherits the condition it is
    recovering from. `empty_cache()` cannot free live tensors -- it releases the
    allocator's unused blocks, which after a failed generation is exactly the
    transient prefill workspace.
    """
    from metafind.data import annotate_run as R

    _rank_first(monkeypatch)
    freed: list[int] = []
    monkeypatch.setattr(R, "_release_cuda", lambda: freed.append(1))

    ann = _BatchOOMs(UNANCHORED_RESPONSE)
    rec, bad = R.annotate_one(ann, "uid0", _render_rec(),
                              lvis_category=None, proportions=PROPS)

    assert bad is None and rec is not None, "an OOM must not lose the asset"
    assert freed, "the batch OOMed and the cache was never released"
    assert ann.singles == R.N_CANDIDATES, "the fallback still draws every candidate"


def test_the_out_of_memory_fallback_is_recorded_as_its_own_mode(monkeypatch):
    """[C4] An OOM-driven fallback must not read as a per-asset one.

    `draw_mode` reaches the sidecar. While both causes wrote
    `sequential_fallback`, the run's OOM rate was unmeasurable after the fact --
    and at 675 MiB of headroom over 5.4 days that rate is the number that
    decides whether the configuration was survivable.
    """
    from metafind.data import annotate_run as R

    _rank_first(monkeypatch)
    monkeypatch.setattr(R, "_release_cuda", lambda: None)

    rec, _ = R.annotate_one(_BatchOOMs(UNANCHORED_RESPONSE), "uid0", _render_rec(),
                            lvis_category=None, proportions=PROPS)
    assert rec["description_sampling"]["draw"] == "oom_sequential_fallback"
    assert rec["description_sampling"]["batch_shape"] == 1, "the batch did not happen"


def test_a_non_memory_failure_keeps_the_plain_fallback_and_frees_nothing(monkeypatch):
    """[C4] The negative case, so the OOM branch cannot quietly swallow the other.

    A malformed prompt or a transient kernel fault is genuinely per-asset. It
    must still fall back, must NOT be labelled an OOM, and must not flush a
    cache that is not the problem.
    """
    from metafind.data import annotate_run as R

    _rank_first(monkeypatch)
    freed: list[int] = []
    monkeypatch.setattr(R, "_release_cuda", lambda: freed.append(1))

    class _BatchFails(_BatchOOMs):
        def generate(self, image_paths, prompt, *, sample=False, seed=None, n=1):
            if sample and n > 1:
                raise RuntimeError("CUDA hiccup")
            return super().generate(image_paths, prompt, sample=sample, seed=seed, n=n)

    rec, _ = R.annotate_one(_BatchFails(UNANCHORED_RESPONSE), "uid0", _render_rec(),
                            lvis_category=None, proportions=PROPS)
    assert rec["description_sampling"]["draw"] == "sequential_fallback"
    assert not freed, "a non-OOM failure must not flush the allocator"


def test_a_single_draw_that_ooms_frees_again_so_one_oom_does_not_become_five(monkeypatch):
    """[C4] The per-draw release, which is a separate failure from the batch one.

    The batch OOMs and is freed; then the FIRST single draw OOMs too. Without a
    release inside the loop its workspace stays cached and draws 2-5 inherit a
    full card, so one OOM reliably becomes five and the asset is quarantined for
    a memory condition that had already been recovered from once.
    """
    from metafind.data import annotate_run as R

    _rank_first(monkeypatch)
    freed: list[int] = []
    monkeypatch.setattr(R, "_release_cuda", lambda: freed.append(1))

    ann = _BatchOOMs(UNANCHORED_RESPONSE, single_fails=1)
    rec, bad = R.annotate_one(ann, "uid0", _render_rec(),
                              lvis_category=None, proportions=PROPS)

    assert bad is None and rec is not None
    assert len(freed) >= 2, "the batch was freed but the failed single draw was not"
    assert len(rec["description_candidates"]) == R.N_CANDIDATES - 1, \
        "four of five draws survived"


# --- C1 / D-2: the default model must be the one that actually runs ---------

def test_the_default_model_is_not_on_the_archive_drive():
    """[C1] A default that is safe only because one caller overrides it.

    `MODEL_ID` was `/mnt/data1/kyzen/models/Qwen3.8-27B` -- 56 GB of bf16
    against a 32,607 MiB card, on the SMR drive that `CLAUDE.md` §9 reserves for
    cold read-mostly bulk. `tools/run_ulip_full.sh` passed `--model` and was the
    only thing standing between that and an OOM, so every direct invocation --
    a resume, a debug run, a validation batch, the timing arm -- loaded 56 GB
    onto a 32 GB card, slowly, off SMR.

    The assertion is the PROPERTY, not the path: a working checkpoint may not
    live on the archive drive. Re-pointing D-2 to another SMR copy would pass a
    string comparison and fail this.
    """
    from metafind.data.annotate_run import MODEL_ID

    assert not MODEL_ID.startswith("/mnt/data1"), (
        f"the default annotator {MODEL_ID} is on the SMR archive drive; "
        "CLAUDE.md §9 reserves it for cold read-mostly bulk")


def test_the_cli_default_and_the_deviation_record_name_the_same_model():
    """[C1, D-2] The record and the runnable default may not drift apart.

    `MODEL_ID`'s comment IS the registered `D-2` deviation record -- what stands
    in for the paper's GPT-4o. While the record said `Qwen3.8-27B` and the chain
    ran `gemma-4-12B-it`, every artifact was stamped honestly and the deviation
    record described a model that had never run and could not run. The run was
    correct and undescribable.
    """
    import argparse

    from metafind.data.annotate_run import MODEL_ID

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    assert ap.parse_args([]).model == MODEL_ID
