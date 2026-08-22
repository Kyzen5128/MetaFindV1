"""Structured annotations for each asset, from its 11 rendered views.

COMPONENT-OF-NODE: n05_annotate  (NOT the node -- see below)

This module is the deterministic half of n05: the prompt, the schema, the
repair-prompt construction and the closed vocabulary. It writes NONE of the
node's channels, so it deliberately carries no IMPLEMENTS-NODE marker -- the
structural check that compares a node's declared writes against its source
would otherwise pass on a file that produces no annotations at all, which is
the exact defect that check exists to catch.

The half that is missing needs the model: generation over the 11 views, the C1
repair loop end to end, and quarantine on exhaustion.

[CORRECTED] This module used to say the paper gives no schema. Paper 2.3's prose
does only say "attributes **such as** object category, size dimensions,
materials, and placement constraints" -- but Figure 2 (``data-preprocess.png``)
PRINTS THE SCHEMA, and v1 of this file was written without ever seeing it.

The figure files are absent from the extracted arXiv source, and
``SOURCE_MANIFEST.json`` records only ``.tex`` hashes -- it never noted that six
referenced images were missing. So "we have read the paper" was true of the
text and false of the figures, and every audit inherited that gap.

The field set is therefore a PAPER FACT (see PLACEMENT_FLAGS below for the
verbatim JSON). What remains ours:

  * the units, which the figure does not label (INFERENCE: centimetres)
  * ``volume`` computed rather than asked
  * material spelling normalisation
  * the serialisation into a sentence for CLIP, which lives in n05b

One caveat on authority: the figure shows ONE example. "These fields exist" is
a paper fact; "these are all the fields" is not.

Why the size field can only ever be a guess
-------------------------------------------

n04 fits every asset to a unit sphere before rendering, so the annotator sees no
absolute scale at all (F13). Its size estimate is therefore a category prior --
"a dining chair is about 90 cm tall" -- and the prompt says so explicitly rather
than inviting it to pretend it measured something. ``raw_bbox_extents`` from
n04's sidecar is carried alongside, so the estimate can be audited against the
mesh's own bounding box. That is a weak ground truth: Objaverse authors choose
their own units, and nothing here verifies a metre is a metre.

Deviation D-2
-------------

Qwen2.5-VL stands in for GPT-4o, recorded per asset as ``annotator_model``.
There is deliberately NO fallback to ULIP-2's shipped captions: they carry no
placement constraints, so substituting them would answer a different question
cheaply rather than this one. L1-ANNOT-NO-FALLBACK scans this file to keep it
that way.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PLACEMENT_FLAGS",
    "MATERIAL_SYNONYMS",
    "PROMPT_VERSION",
    "blind_agrees",
    "build_blind_prompt",
    "parse_blind_guess",
    "VALIDATOR_VERSION",
    "SCHEMA_VERSION",
    "AnnotationError",
    "annotation_contract_id",
    "build_prompt",
    "CANONICAL_LVIS_CATEGORY",
    "CANONICAL_PROPORTIONS",
    "category_relation",
    "derive_dimensions",
    "lvis_synset",
    "build_repair_prompt",
    "non_english_characters",
    "parse_annotation",
    "validate_annotation",
]

# --- annotation contract versioning ---------------------------------------
#
# THREE versions, not one. `prompt_version` alone used to stand for the whole
# annotation contract, and it cannot: the prompt, the validator and the record
# schema change independently, and two records both stamped `prompt_version: 3`
# could have been admitted under different acceptance rules with nothing on disk
# to tell them apart. That is the same defect class as the retired
# `"metafind_v1_natural"` serialization label, which named two different
# transformations -- see D0-008 §11.2 B-3 and D10.
#
#   PROMPT_VERSION     what was ASKED of the model
#   VALIDATOR_VERSION  what was ACCEPTED from it
#   SCHEMA_VERSION     what the stored record can EXPRESS
#
# and `annotation_contract_id()` binds all three to the actual text of the
# prompt and the actual admission bounds, so an edit that someone forgets to
# version still moves the identity.
PROMPT_VERSION = 6  # v6 asks BLIND first, then anchors; shows the paper's
                    # Figure 2 example instead of describing it; and asks the
                    # model to look again before describing. v5 anchors the identity on the Objaverse-LVIS label and
                    # supplies the exact mesh proportions, so the model answers
                    # ONE scale question instead of guessing three dimensions
                    # from scale-normalised renders. v4 stated the output
                    # language; v3 did not, and 3 records came back
                    # part-Chinese. v2 followed Figure 2 but left 41% with all
                    # four flags false
VALIDATOR_VERSION = 3  # v3 admits `identity_confirmed`, derives width/length
                       # from the mesh instead of accepting them, and looks
                       # `synset` up rather than reading it from the model.
                       # v2 refused non-Latin script; v1 had no language rule
SCHEMA_VERSION = 4  # v4 stores the turn-1 blind guess, its raw text and
                    # whether it agrees with the anchor -- the only
                    # automatic accuracy signal in the bake-off, and the
                    # answer to W-7's "is identity_confirmed a rubber
                    # stamp?". v3 stores the LVIS anchor, `identity_confirmed`, the
                    # derived category relation and the horizontal axis
                    # assignment. v2 recorded translation/repair provenance and
                    # the contract identity; v1 had neither
MAX_ATTEMPTS = 2  # C1's hard bound; the third outcome is quarantine

# [PAPER FACT -- Figure 2, `data-preprocess.png`] The paper DOES show the
# annotation schema. It is printed inside the "Structured Detailed Description"
# box of the asset-level pipeline, and reads:
#
#   {"annotations": {
#     "category": "robot", "synset": "robot.n.01",
#     "width": 30, "length": 30, "height": 40, "volume": 36000, "mass": 2.5,
#     "description": "A small cubic-shaped robot with a smiling screen face,
#                     two antennae on top, and rounded side arms and feet with
#                     spring-like connectors.",
#     "materials": ["metal", "glass", "plastic"],
#     "onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}}
#
# v1 of this module predates that reading. The figure files are absent from the
# extracted arXiv source (see SOURCE_MANIFEST.json -- it records only .tex
# hashes and never noted that six referenced images were missing), so every
# earlier audit read the paper as text alone and treated the schema as
# unspecified. It is not unspecified.
#
# What v1 got wrong, specifically:
#   * placement was a single 8-label list WE invented. The paper uses FOUR
#     INDEPENDENT BOOLEANS. v1's vocabulary mixed two orthogonal axes -- where
#     an object rests (floor/tabletop/wall/ceiling/shelf/outdoor) and whether it
#     is portable (handheld) -- and forced one choice between them. MEASURED
#     consequence: 61.3% of assets landed in the two catch-alls
#     (unconstrained 30.7% + handheld 30.6%), and chairs were labelled
#     `handheld`. Against AI2-THOR ground truth on 1,930 assets the field was
#     67.2% correct on a binary whose majority class is ~70%, i.e. it carried
#     almost no signal (Box 2%, Lettuce 4%, Book 10%).
#   * synset, volume and mass were absent entirely.
#   * dimensions were in metres; the figure's numbers are 30/30/40 with
#     volume 36000 = 30*30*40, which is only coherent in CENTIMETRES.
#
# The figure shows ONE example, so "these fields exist" is a PAPER FACT while
# "these are ALL the fields" is not -- a figure may abbreviate for layout.
PLACEMENT_FLAGS = ("onCeiling", "onWall", "onFloor", "onObject")

# Units. MetaFind's figure prints bare numbers and labels none of them.
#
#   SCHEMA MATCH, PROVENANCE UNKNOWN. The field set matches Holodeck's
#   (AllenAI, CVPR 2024) Appendix A.6 in 13 of 14 attributes -- only
#   `frontView` differs -- and A.6 states "physical dimensions in
#   centimeters", "cubic centimeters (cm3)", "kilograms (kg)". But MetaFind
#   cites neither Holodeck nor ObjaTHOR anywhere: 51 bib entries, zero hits
#   for holodeck/objathor/allenai across .tex, .bib and .bbl, and
#   2methdology.tex's data-preparation section carries no citation at all.
#   So the relationship is UNSTATED, and this may never be written as
#   "MetaFind specifies centimetres".
#
# The two units are NOT equally supported, and the report must keep them apart:
#
#   cm   UPSTREAM-SUPPORTED INFERENCE, plus MetaFind-internal arithmetic:
#        the figure's own 30 * 30 * 40 = 36000 closes only in cubic
#        centimetres. That second leg does not depend on Holodeck at all.
#   kg   UPSTREAM-SUPPORTED INFERENCE only. The figure prints mass 2.5 and
#        nothing in it can verify the unit. Weaker than cm; say so.
#
# ObjaTHOR is read for meaning and never copied, because the lineage has moved:
# current ObjaTHOR uses `depth` where the figure uses `length` and documents
# volume in LITRES where the figure's arithmetic is cubic centimetres. Nor does
# any released annotation version carry this schema -- 2023_07_28 has no
# placement booleans at all, and 2023_09_23 has the four booleans but stores
# geometry as a `size` triple with no volume, mass or materials. MetaFind's
# figure matches the Holodeck PAPER, not any shipped dataset. Where sources
# disagree, the figure wins: it is the paper being reproduced.
DIMENSION_UNIT = "cm"
MASS_UNIT = "kg"

# [UNKNOWN -- U-NEW] Holodeck's schema has FOURTEEN fields; MetaFind's figure
# shows thirteen. The missing one is `front view`, "an integer denoting the view
# representing the front of the object, often the most symmetrical view".
#
# Two readings, and nothing distinguishes them: the figure abbreviated for
# layout, or MetaFind dropped a field it had no use for. Holodeck needs a
# canonical front to orient assets when placing them in a room; MetaFind
# retrieves rather than places, so dropping it is plausible.
#
# NOT implemented, because implementing an unstated field is inventing one.
# Recorded here so the gap is visible rather than absent.
UPSTREAM_FIELD_NOT_IMPLEMENTED = "front view"

# [PROMPT_VERSION 5] `synset` is no longer asked of the model -- it is looked
# up (Design Decision 4). `width` and `length` are no longer asked either: the
# mesh already carries the exact proportions, so the model supplies absolute
# scale for `height` alone and names which horizontal axis reads as width.
REQUIRED_FIELDS = ("category", "identity_confirmed", "height", "width_axis",
                   "mass", "description", "materials", *PLACEMENT_FLAGS)
DIMENSION_KEYS = ("width", "length", "height")
HORIZONTAL_AXES = ("x", "z")

# --- Design Decision 4: `synset` is mapped, not generated -------------------
#
# [UPSTREAM FACT] Every entry is copied verbatim from the LVIS v1 release; see
# `lvis_synsets.json` `_provenance`. Nothing here is inferred or hand-authored.
#
# [IMPLEMENTATION CHOICE] The synset follows the LVIS ANCHOR, never the model's
# refined `category`. "toy" -> "toy dinosaur" is a better retrieval string but
# there is no authoritative synset for "toy dinosaur", and minting one would
# re-introduce exactly the invented-synset error class this lookup removes.
_LVIS_SYNSET_PATH = Path(__file__).with_name("lvis_synsets.json")
with _LVIS_SYNSET_PATH.open(encoding="utf-8") as _fh:
    LVIS_SYNSETS: dict[str, str] = json.load(_fh)["synsets"]


def lvis_synset(lvis_category: str) -> str:
    """The WordNet id LVIS itself assigns to this category. KeyError is correct:
    an unknown category means the anchor is not from the 1,156-term vocabulary,
    and guessing past that is what v4 did."""
    try:
        return LVIS_SYNSETS[lvis_category]
    except KeyError:
        raise AnnotationError(
            f"{lvis_category!r} is not one of the {len(LVIS_SYNSETS)} "
            "Objaverse-LVIS categories, so no authoritative synset exists for it"
        ) from None


_PAREN = re.compile(r"\s*\([^)]*\)")


def _tokens(name: str) -> set[str]:
    """Content words of a category name. LVIS parenthetical disambiguators
    ("car (automobile)") are annotation apparatus, not part of the name."""
    return {t for t in re.split(r"[^a-z0-9]+", _PAREN.sub("", name.lower())) if t}


def category_relation(lvis_category: str, category: str) -> str:
    """How the model's `category` stands to the LVIS anchor.

    `exact`      the same name
    `refined`    every anchor word survives and the name is longer
                 -- "toy" -> "toy dinosaur"
    `divergent`  neither -- includes both genuine downward refinements that
                 change vocabulary ("motor vehicle" -> "pickup truck") AND
                 lateral replacements ("motor vehicle" -> "coffee machine")

    [IMPLEMENTATION CHOICE -- UNRESOLVED ENFORCEMENT] `divergent` deliberately
    does NOT distinguish those last two. Separating them needs a hypernym
    judgement over free text that no authoritative source in this project
    supplies, and inventing one would be exactly the silent guess the design
    forbids. So this run RECORDS the relation and does not reject on it, on the
    same reasoning as Design Decision 2: measure the rate first, then decide
    whether the rule needs teeth. See TASK.md 14.
    """
    a, c = _tokens(lvis_category), _tokens(category)
    if a == c:
        return "exact"
    if a and a <= c:
        return "refined"
    return "divergent"


def derive_dimensions(height_cm: float, proportions: tuple[float, float, float],
                      width_axis: str) -> tuple[float, float]:
    """(width, length) in cm from the exact mesh proportions.

    `proportions` is (y, x, z) normalised so the largest is 1.0, with y the
    up axis -- verified independently against the corpus, not assumed. Only
    ONE number in the record is an estimate; these two are measurements scaled
    by it.
    """
    py, px, pz = proportions
    if py <= 0:
        raise AnnotationError(
            "the mesh has zero extent on the up axis, so height cannot set the scale"
        )
    scale = height_cm / py
    x_cm, z_cm = px * scale, pz * scale
    return (x_cm, z_cm) if width_axis == "x" else (z_cm, x_cm)

# Objaverse holds everything from a bead to a building. Anything outside this
# is a unit error, not an object. Widened from v1's metre bounds by 100x
# because the unit changed, not because the corpus did.
MIN_DIM_CM, MAX_DIM_CM = 0.1, 10_000.0
MIN_MASS_KG, MAX_MASS_KG = 0.001, 100_000.0
# Water is 0.001 kg/cm^3, osmium 0.0225. Loose by design: a bounding box is
# mostly air for most objects, so the FLOOR matters more than the ceiling.
MIN_DENSITY_KG_CM3, MAX_DENSITY_KG_CM3 = 1e-9, 1.0

# [IMPLEMENTATION CHOICE] Normalisation, NOT a closed vocabulary. The figure's
# example is lowercase free text (["metal", "glass", "plastic"]) and the paper
# defines no material vocabulary, so forcing a closed list would be our
# invention. Merging spellings of one word is not: v1 produced `metal` 34.3%
# AND `metallic` 10.7% as separate tokens, which the text encoder sees as two
# different words for one material. Only unambiguous spelling variants are
# merged; nothing is dropped, because deciding what "is not a material" is a
# judgement the paper does not license.
MATERIAL_SYNONYMS = {
    "metallic": "metal",
    "metals": "metal",
    "wooden": "wood",
    "plastics": "plastic",
    "fabrics": "fabric",
    "textile": "fabric",
    "glassy": "glass",
    "ceramics": "ceramic",
    "rubbery": "rubber",
    "leathery": "leather",
    "papery": "paper",
    "stones": "stone",
}


# --- P-2: the English text contract ----------------------------------------
#
# [IMPLEMENTATION CHOICE, VALIDATOR_VERSION 2] The frozen CLIP text tower was
# trained on English captions, and Stage 1 serializes `category`, `description`
# and `materials` straight into the string it sees. A Chinese description is not
# a schema error -- it is a correct answer to a prompt that never named a
# language -- but it is text the encoder cannot place.
#
# The rule is an ALLOW-LIST on script, not `.isascii()`. The corpus contains
# legitimate accented English, and an ASCII rule would reject all of it:
# measured over all 45,952 v3 records, the complete non-ASCII vocabulary is
#
#     U+00E9 'é' x5   U+00ED 'í' x1   U+00EB 'ë' x1   U+00A2 '¢' x1
#
# from "Pokémon", "Carmín", "Raphaël" and a "5¢" price tag. All four must pass.
# What must not pass is a different SCRIPT: CJK, Cyrillic, Arabic, Hebrew,
# Devanagari, Hangul, Kana and the rest.
#
# Implemented on `unicodedata` rather than on hand-written codepoint ranges: a
# letter is admitted when its Unicode NAME begins with "LATIN", which is exactly
# the property wanted and stays correct as Unicode grows. Symbols and
# punctuation are admitted below U+2100, which covers Latin-1, General
# Punctuation and the currency block (so '¢', '€', an em dash and curly quotes
# pass) while excluding ideographic punctuation, arrows and emoji.
NON_LATIN_SYMBOL_CUTOFF = 0x2100


def non_english_characters(text: str) -> list[str]:
    """The characters in `text` that break the English text contract.

    Returns them in order of first appearance, deduplicated, so the repair
    prompt can name them. Empty list means the text conforms.
    """
    bad: list[str] = []
    for ch in text:
        if ch.isspace() or ord(ch) < 128:
            continue
        category = unicodedata.category(ch)
        name = unicodedata.name(ch, "")
        if category[0] in "LM":                       # letters and marks
            ok = name.startswith("LATIN") or name.startswith("COMBINING")
        elif category[0] == "N":                      # numbers
            ok = name.startswith(("DIGIT", "LATIN", "SUPERSCRIPT", "VULGAR"))
        elif category[0] in "PSZ":                    # punctuation, symbols
            ok = ord(ch) < NON_LATIN_SYMBOL_CUTOFF
        else:
            ok = False
        if not ok and ch not in bad:
            bad.append(ch)
    return bad


def _refuse_non_english(field: str, value: str) -> None:
    """[L1-ANNOT-LANGUAGE] Raise with a message the repair prompt can act on.

    The message is fed back verbatim by `build_repair_prompt()`, so it names the
    field, the offending characters and the rule -- "invalid schema" would give
    the model nothing to fix.
    """
    bad = non_english_characters(value)
    if bad:
        raise AnnotationError(
            f"`{field}` contains non-English text: {''.join(bad)!r}. "
            "The annotation text must be written in English. Non-English "
            "scripts are not allowed in category, description, or materials. "
            "Accented Latin letters such as 'Pokémon' are fine; Chinese, "
            "Japanese, Korean, Cyrillic and Arabic characters are not. "
            "Rewrite the field in English and return the corrected JSON."
        )


class AnnotationError(Exception):
    """A schema violation, carrying the message that goes into the repair prompt."""


@dataclass(frozen=True)
class Annotation:
    """One asset's annotation, in the field set paper Figure 2 prints."""

    category: str
    synset: str
    width: float
    length: float
    height: float
    mass: float
    description: str
    materials: list[str]
    on_ceiling: bool
    on_wall: bool
    on_floor: bool
    on_object: bool
    # --- SCHEMA_VERSION 3 ---------------------------------------------------
    # The anchor is stored, not just consumed. Without it on disk nobody can
    # afterwards tell a refinement from a replacement, and the whole point of
    # anchoring is that the disagreement becomes MEASURABLE.
    lvis_category: str
    identity_confirmed: bool
    category_relation: str
    width_axis: str

    @property
    def volume(self) -> float:
        """[DERIVED, never asked of the model] Figure 2 prints volume 36000
        beside width 30, length 30, height 40, and 30*30*40 = 36000 exactly.
        The field is therefore a product of the other three, not an independent
        observation. Asking the model for it invites a fourth number that does
        not agree with the first three, and there is no way to tell afterwards
        which of the four was wrong.
        """
        return self.width * self.length * self.height

    def as_record(self, annotator_model: str) -> dict[str, Any]:
        return {
            "category": self.category,
            "synset": self.synset,
            # [SCHEMA_VERSION 3] The LVIS anchor and what the model did with it.
            # `identity_confirmed` is RECORDED ONLY on this run -- nothing is
            # quarantined, dropped or repaired on it. See TASK.md R-B: LVIS's
            # own error rate has never been measured, and a corpus filtered on
            # an unmeasured rule cannot measure anything.
            "lvis_category": self.lvis_category,
            "identity_confirmed": self.identity_confirmed,
            "category_relation": self.category_relation,
            "width_axis": self.width_axis,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "volume": self.volume,
            "mass": self.mass,
            "description": self.description,
            "materials": self.materials,
            "onCeiling": self.on_ceiling,
            "onWall": self.on_wall,
            "onFloor": self.on_floor,
            "onObject": self.on_object,
            # Provenance, outside the paper's field set. `dimension_unit` is an
            # INFERENCE (see DIMENSION_UNIT) and must travel with the numbers,
            # or a later reader repeats the metres/centimetres mistake v1 made.
            "dimension_unit": DIMENSION_UNIT,
            "mass_unit": MASS_UNIT,
            # [P-5] The annotation contract, in three independent axes plus a
            # fingerprint that moves even when someone forgets to bump one.
            "prompt_version": PROMPT_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "annotation_contract": annotation_contract_id(),
            "annotator_model": annotator_model,
            # [P-5] Description provenance, always present so that "this text is
            # the model's" is a RECORDED fact rather than the absence of a note.
            # A human repair overwrites these four; `annotator_model` stays,
            # because every other field is still the model's output.
            "description_source": "model",
            "description_original": None,
            "description_translation_authority": None,
            "description_translated_by": None,
        }


# [PROMPT_VERSION 6] Turn 1. No anchor, no schema, no JSON.
#
# v5 hands the model the Objaverse-LVIS label and asks it to confirm. `W-7`
# named the problem that creates: **there is no ground truth telling us a
# `true` is really true**, so `identity_confirmed` could be a rubber stamp and
# nothing on disk would show it. Asking BEFORE the label exists is what makes
# the confirmation falsifiable -- the blind answer either agrees with the
# catalogue or it does not, and the catalogue is a source we did not write.
#
# It is also the only automatic accuracy signal in the bake-off. Description
# quality, mass and materials have no answer key; `U-Q`'s 20 hand-adjudicated
# assets are the only ground truth for those and they are the USER's to judge.
BLIND_PROMPT = (
    "You are looking at {n_views} rendered views of a single 3D asset, from "
    "different angles around it.\n"
    "\n"
    "First, LOOK. Write two or three short lines describing only what is "
    "actually visible: overall shape, distinct parts, colours, apparent "
    "materials, any text or markings.\n"
    "\n"
    "Then, on a final line by itself, write:\n"
    "OBJECT: <what this object is, in one to three words>\n"
    "\n"
    "Name the object, not the scene. If you are unsure, give your single best "
    "guess -- an honest wrong guess is more useful here than a refusal."
)


def build_blind_prompt(n_views: int) -> str:
    """[PROMPT_VERSION 6] Turn 1: identify with nothing supplied."""
    return BLIND_PROMPT.format(n_views=n_views)


def parse_blind_guess(raw: str) -> str:
    """The `OBJECT:` line, verbatim. Empty string when the model did not give one.

    `S-11` requires the blind guess be recorded VERBATIM, so this extracts and
    does not normalise: no lowercasing, no article stripping, no lemmatising.
    Whatever cleaning a later analysis wants, it can do on a value that still
    says what the model said.
    """
    for line in reversed(raw.strip().splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("OBJECT:"):
            return stripped[len("OBJECT:"):].strip()
    return ""


def blind_agrees(blind_guess: str, lvis_category: str) -> bool:
    """Does the unprompted answer share a content word with the catalogue label?

    Deliberately generous, and the direction of its error matters. "wooden
    chair" against "chair" counts; "seat" against "chair" does not, because
    nothing here knows they are synonyms. So the measured agreement rate is a
    LOWER BOUND on blind identification accuracy, not an estimate of it -- a
    model penalised for using a different word for the right object.

    Tightening it would need a synonym judgement over free text that no
    authoritative source in this project supplies, which is the same reason
    `category_relation` refuses to split `divergent`.
    """
    if not blind_guess or not lvis_category:
        return False
    return bool(_tokens(blind_guess) & _tokens(lvis_category))


# [PAPER FACT -- Figure 2] The paper's own worked example, shown to the model
# rather than paraphrased. It is the one example of this schema that carries any
# authority, and every earlier prompt described the fields in prose while the
# figure sat unused. `volume` is omitted from what the model is asked for -- see
# `Annotation.volume`, it is a product of the other three -- but it is left in
# the example because removing it would misrepresent the figure.
FIGURE_2_EXAMPLE = (
    '{"category": "robot", "synset": "robot.n.01",\n'
    '  "width": 30, "length": 30, "height": 40, "volume": 36000, "mass": 2.5,\n'
    '  "description": "A small cubic-shaped robot with a smiling screen face, '
    'two antennae on top, and rounded side arms and feet with spring-like '
    'connectors.",\n'
    '  "materials": ["metal", "glass", "plastic"],\n'
    '  "onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}'
)


def build_prompt(n_views: int, lvis_category: str,
                 proportions: tuple[float, float, float],
                 blind_guess: str = "") -> str:
    """[PROMPT_VERSION 6] Turn 2: anchored, exemplified, observe-before-answer.

    What v6 changes and why each change exists:

    * **The model has already answered blind** (`build_blind_prompt`). Its own
      words are quoted back, so the anchor arrives as a second opinion rather
      than as the first thing it ever hears about the object. v5 stated the
      answer and then asked for agreement, which is what `W-7` calls a rubber
      stamp.
    * **The paper's Figure 2 example is shown**, not described. It is the only
      authoritative instance of this schema and every prompt before v6 left it
      on the page.
    * **Observe before answering.** The fields are asked for after an explicit
      look, in the order a reader would need them, instead of as a bare list.

    What v6 does NOT change: the FIELD SET. `SPEC_M1` and the USER both hold
    that the stored record must carry the paper's fields, so prompt engineering
    moves what is ASKED and never what is KEPT.

    Constrained JSON decoding was specified for v6 and is NOT implemented:
    `outlines`, `lm-format-enforcer`, `jsonformer` and `xgrammar` are all absent
    and transformers has no built-in grammar. It is also not obviously wanted --
    `S-10` measures parse failures per arm, and a model that cannot emit valid
    JSON unaided is telling the USER something about itself that constrained
    decoding would hide. Recorded as an IMPLEMENTATION CHOICE, revisit if parse
    failures dominate the result.

    Inherited from v5 and unchanged:

    [DEVIATION -- see TASK.md R-E] The paper has the VLM generate the
    annotations with GPT-4o (`2methdology.tex:28`, `neurips_2025.tex:100`).
    Supplying the dataset's own label is a DEPARTURE from that, adopted because
    a 7B-class model asked to IDENTIFY an object in a 224x224 scale-normalised
    render collapses onto high-frequency priors -- `toy` was the single most
    common answer at 3.4%, a word v4's prompt explicitly forbade. This is not
    paper-faithful and must never be described as such.

    Why proportions are handed over rather than asked for: n04 fits every asset
    to a unit sphere, so the renders carry no absolute scale, and the previous
    prompt's own instruction was to estimate size "from what kind of object it
    is, not from the picture" -- i.e. dimensions were already category-derived.
    The mesh bounding box is exact and covers the corpus, so three guesses
    become one.
    """
    py, px, pz = proportions
    recall = (f'A moment ago, looking at these same views with nothing supplied, '
              f'you said this was: "{blind_guess}"\n\n'
              if blind_guess else "")
    return (
        f"You are looking at {n_views} rendered views of a single 3D asset.\n"
        "\n"
        + recall +
        f'This asset is catalogued in Objaverse-LVIS as: "{lvis_category}"\n'
        "Treat that identity as CORRECT unless the images clearly contradict it.\n"
        "\n"
        "Its true proportions, measured from the mesh itself, are\n"
        f"    height : x-axis : z-axis  =  {py:.3f} : {px:.3f} : {pz:.3f}\n"
        "(largest = 1.000). These are EXACT. Do not re-estimate them from the "
        "images.\n"
        "\n"
        "IMPORTANT: write every text field in ENGLISH. `category`, `description` "
        "and every entry of `materials` must be English. Do not use Chinese, "
        "Japanese, Korean, Cyrillic or Arabic characters. Accented Latin "
        "spellings such as \"Pokemon\"/\"Pok\u00e9mon\" are fine.\n"
        "\n"
        "IMPORTANT: these renders are SCALE-NORMALISED. Every asset is fitted to "
        "the same size before rendering, so the images contain no information "
        "about how large the object really is. You are asked for exactly ONE "
        "absolute measurement, and you should answer it from what the object IS, "
        "not from the picture.\n"
        "\n"
        "Before you answer, LOOK at the views once more and note to yourself: "
        "what parts does it have, what colours, what does the surface look like, "
        "is there any text or marking. The description below should come from "
        "that, not from what the category name suggests.\n"
        "\n"
        "This is the exact format, shown with a worked example:\n"
        f"{FIGURE_2_EXAMPLE}\n"
        "\n"
        "Return one JSON object and nothing else, with exactly these fields "
        "(do not include `width`, `length` or `volume` -- they are computed "
        "from the proportions above):\n"
        f'  "category": the catalogued identity "{lvis_category}", OR a STRICTLY '
        "MORE SPECIFIC term for the same object. Refine it when the images "
        'support that: "toy" -> "toy dinosaur", "motor vehicle" -> "pickup '
        'truck". NEVER name a different object.\n'
        '  "identity_confirmed": true if the images are consistent with the '
        "catalogued identity, false if they clearly show something else. Answer "
        "honestly -- false is a useful answer, not a failure.\n"
        f'  "height": number, CENTIMETRES. How tall is a real {lvis_category}? '
        "This one number sets the scale; the other two dimensions follow from "
        "the proportions above.\n"
        '  "width_axis": "x" or "z" -- which of the two horizontal axes reads as '
        "left-to-right WIDTH in these views. The other becomes front-to-back "
        "length.\n"
        '  "mass": number, KILOGRAMS\n'
        '  "description": the identity is already given, so do not spend the '
        "sentence restating it. Describe what makes THIS instance distinctive: "
        "colour, style, finish, condition, ornament, distinguishing detail. One "
        "or two sentences.\n"
        '  "materials": a list of material names, most prominent first\n'
        '  "onCeiling": true if this object is typically mounted on a ceiling\n'
        '  "onWall": true if it is typically mounted on a wall\n'
        '  "onFloor": true if it typically rests directly on the floor or ground\n'
        '  "onObject": true if it could sit on top of a table, desk, shelf, '
        "counter, cabinet or any other supporting surface\n"
        "\n"
        "The four placement fields are INDEPENDENT booleans, not a choice. Answer "
        "each one separately, and ask yourself where this object would be found "
        "in a real room.\n"
        "\n"
        "ALMOST EVERY OBJECT HAS AT LEAST ONE OF THE FOUR SET TO TRUE. All four "
        "false means the object belongs nowhere at all, which is true only of "
        "abstract shapes with no real-world counterpart. If you are unsure, ask "
        "the simpler question: is it small enough to put on a table? Then "
        "onObject is true. Is it large enough to stand on the ground? Then "
        "onFloor is true.\n"
        "\n"
        "Worked examples:\n"
        "  chair          onFloor true, everything else false\n"
        "  mug            onObject true, onFloor false -- it lives on surfaces\n"
        "  book           onFloor AND onObject both true\n"
        "  teddy bear     onFloor AND onObject both true\n"
        "  ceiling lamp   onCeiling true, everything else false\n"
        "  framed picture onWall true, everything else false\n"
        "  car            onFloor true (the ground counts as floor)\n"
        "Do not omit any of the four."
    )


def build_repair_prompt(original: str, error: str, raw_response: str) -> str:
    """The retry prompt for C1.

    [L1-ANNOT-REPAIR] It must NAME the specific failure and differ from the
    original. Re-sending an identical prompt reproduces the identical mistake,
    which turns a repair loop into a delay with a retry count attached.
    """
    if not error:
        raise ValueError("a repair prompt with no error to repair is just a resend")
    return (
        f"{original}\n"
        "\n"
        "YOUR PREVIOUS ANSWER WAS REJECTED.\n"
        f"Reason: {error}\n"
        f"What you returned: {raw_response[:600]}\n"
        "\n"
        "Fix exactly that problem and return the corrected JSON object alone."
    )


ANNOTATION_CONTRACT_FAMILY = "metafind_annot"
CANONICAL_N_VIEWS = 11  # n04 renders 11; the prompt text depends on the count
# [PROMPT_VERSION 5] The prompt now varies per asset, so the contract hashes a
# FIXED instantiation. These two values are arbitrary but pinned: change them
# and the fingerprint moves for a reason that has nothing to do with the
# contract, which is the opposite of what the fingerprint is for.
CANONICAL_LVIS_CATEGORY = "chair"
CANONICAL_PROPORTIONS = (1.0, 0.5, 0.5)


def annotation_contract() -> dict[str, Any]:
    """Everything that decides whether an annotation is admitted, in one mapping.

    Hashed rather than trusted to three hand-maintained integers: an edit to the
    prompt text, to a dimension bound, to the material synonyms or to the
    language rule changes what the corpus MEANS, and a version someone forgot to
    bump would otherwise hide it. Same reasoning as D10's
    `text_serialization_id()` on the encoder side.
    """
    return {
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt": build_prompt(CANONICAL_N_VIEWS, CANONICAL_LVIS_CATEGORY,
                               CANONICAL_PROPORTIONS),
        "required_fields": list(REQUIRED_FIELDS),
        "horizontal_axes": list(HORIZONTAL_AXES),
        # The synset table is part of what the corpus MEANS: swap it and every
        # record's `synset` changes without a single version constant moving.
        "lvis_synset_table_sha256": hashlib.sha256(
            json.dumps(LVIS_SYNSETS, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "placement_flags": list(PLACEMENT_FLAGS),
        "dimension_unit": DIMENSION_UNIT,
        "mass_unit": MASS_UNIT,
        "dim_bounds_cm": [MIN_DIM_CM, MAX_DIM_CM],
        "mass_bounds_kg": [MIN_MASS_KG, MAX_MASS_KG],
        "density_bounds_kg_cm3": [MIN_DENSITY_KG_CM3, MAX_DENSITY_KG_CM3],
        "material_synonyms": dict(sorted(MATERIAL_SYNONYMS.items())),
        "non_latin_symbol_cutoff": NON_LATIN_SYMBOL_CUTOFF,
        "max_attempts": MAX_ATTEMPTS,
    }


def annotation_contract_id() -> str:
    """`metafind_annot_v<prompt>@<hash16>` -- stamped on every record."""
    payload = json.dumps(annotation_contract(), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{ANNOTATION_CONTRACT_FAMILY}_v{PROMPT_VERSION}@{digest[:16]}"


def parse_annotation(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model response.

    Models wrap JSON in prose or fences however they like, and treating that as
    a schema failure would spend a repair attempt on formatting rather than on
    content.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise AnnotationError("the response contained no JSON object")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"the JSON was malformed: {exc}") from exc
    if not isinstance(obj, dict):
        raise AnnotationError("the top-level JSON value was not an object")
    return obj


def validate_annotation(obj: dict[str, Any], *, lvis_category: str,
                        proportions: tuple[float, float, float]) -> Annotation:
    """[L1-ANNOT-SCHEMA, VALIDATOR_VERSION 3] Every rule, with the message the
    repair prompt needs.

    Each message names the offending field and value, because it is fed back
    verbatim and a message like "invalid schema" gives the model nothing to act
    on.

    v3 differs from v2 in three ways, and all three narrow what the model is
    trusted with:

      * `synset` is LOOKED UP from the LVIS anchor, not read from the response.
        v2 checked the SHAPE only, so a well-formed but invented id passed.
      * `width` and `length` are DERIVED from the exact mesh proportions. v2
        accepted three independent guesses about an object whose renders carry
        no scale at all.
      * `identity_confirmed` is required and recorded. It is NOT a gate --
        `false` is admitted exactly like `true` (TASK.md R-B).
    """
    missing = [f for f in REQUIRED_FIELDS if f not in obj]
    if missing:
        raise AnnotationError(f"required field(s) missing: {', '.join(missing)}")

    for field in ("category", "description"):
        if not isinstance(obj[field], str) or not obj[field].strip():
            raise AnnotationError(f"`{field}` must be a non-empty string")

    # [L1-ANNOT-LANGUAGE] Before anything else about the content: is it English?
    for field in ("category", "description"):
        _refuse_non_english(field, obj[field])

    # [VALIDATOR_VERSION 3] The identity question. A non-boolean is refused
    # rather than coerced: "yes", 1 and "true" all mean the model answered a
    # different question than the one asked, and a coerced answer is
    # indistinguishable afterwards from a real one.
    identity_confirmed = obj["identity_confirmed"]
    if not isinstance(identity_confirmed, bool):
        raise AnnotationError(
            f"`identity_confirmed` must be true or false, got "
            f"{identity_confirmed!r}"
        )

    width_axis = obj["width_axis"]
    if not isinstance(width_axis, str) or width_axis.strip().lower() not in HORIZONTAL_AXES:
        raise AnnotationError(
            f"`width_axis` = {width_axis!r} must be exactly \"x\" or \"z\" -- "
            "which of the two horizontal axes reads as left-to-right width"
        )
    width_axis = width_axis.strip().lower()

    # [VALIDATOR_VERSION 3] ONE estimate. The other two dimensions are the
    # mesh's own proportions scaled by it, so they cannot disagree with the
    # shape the renders show -- which is what v2's three independent guesses
    # could and did.
    height = obj["height"]
    if isinstance(height, bool) or not isinstance(height, (int, float)):
        raise AnnotationError(f"`height` must be a number, got {height!r}")
    height = float(height)
    if not MIN_DIM_CM <= height <= MAX_DIM_CM:
        raise AnnotationError(
            f"`height` = {height} is outside {MIN_DIM_CM}-{MAX_DIM_CM} cm. "
            "Sizes must be in CENTIMETRES, not metres or millimetres."
        )
    width, length = derive_dimensions(height, proportions, width_axis)
    dims = {"width": width, "length": length, "height": height}
    # The derived pair can still leave the admissible range -- a needle-thin
    # mesh scaled by a plausible height produces a sub-millimetre width. The
    # message points at `height`, because that is the only number the model
    # chose.
    for k in ("width", "length"):
        if not MIN_DIM_CM <= dims[k] <= MAX_DIM_CM:
            raise AnnotationError(
                f"a height of {height} cm implies {k} = {dims[k]:.4g} cm from "
                f"the mesh proportions, which is outside "
                f"{MIN_DIM_CM}-{MAX_DIM_CM} cm. Re-check the height in "
                "CENTIMETRES."
            )

    mass = obj["mass"]
    if isinstance(mass, bool) or not isinstance(mass, (int, float)):
        raise AnnotationError(f"`mass` must be a number, got {mass!r}")
    mass = float(mass)
    if not MIN_MASS_KG <= mass <= MAX_MASS_KG:
        raise AnnotationError(
            f"`mass` = {mass} is outside {MIN_MASS_KG}-{MAX_MASS_KG} kg. "
            "Mass must be in KILOGRAMS."
        )

    # [IMPLEMENTATION CHOICE] Cross-field consistency. Unchanged from v2 in
    # intent, but it now checks a mass estimate against a bounding box that is
    # exact in shape and estimated in one scalar, rather than against three
    # independent guesses. A UNIT-ERROR detector, not a physics check.
    density = mass / (dims["width"] * dims["length"] * dims["height"])
    if not MIN_DENSITY_KG_CM3 <= density <= MAX_DENSITY_KG_CM3:
        raise AnnotationError(
            f"`mass` = {mass} kg with the implied "
            f"{dims['width']:.4g} x {dims['length']:.4g} x {dims['height']:.4g} cm "
            f"bounding box gives {density:.3g} kg/cm^3, which is not physical "
            f"(water is 0.001). Check the height is in CENTIMETRES and the mass "
            f"in KILOGRAMS."
        )

    materials = obj["materials"]
    if isinstance(materials, str):
        materials = [materials]
    if not isinstance(materials, list) or not materials:
        raise AnnotationError("`materials` must be a non-empty list of strings")
    if any(not isinstance(m, str) or not m.strip() for m in materials):
        raise AnnotationError("every entry in `materials` must be a non-empty string")
    normalised = [MATERIAL_SYNONYMS.get(m.strip().lower(), m.strip().lower())
                  for m in materials]
    for material in normalised:
        _refuse_non_english("materials", material)

    flags: dict[str, bool] = {}
    for f in PLACEMENT_FLAGS:
        v = obj[f]
        if not isinstance(v, bool):
            raise AnnotationError(
                f"`{f}` must be true or false, got {v!r}. The four placement "
                "fields are independent booleans, not a category choice."
            )
        flags[f] = v
    # Deliberately NOT requiring at least one true. See v2's reasoning.

    category = obj["category"].strip()
    # [TASK.md R-B applied to `category`] The relation is COMPUTED and STORED.
    # `divergent` is not rejected on this run: the rule "refinement, never
    # lateral replacement" cannot be enforced mechanically without a hypernym
    # judgement over free text that no authoritative source here supplies.
    # Recording the rate is what makes the enforcement question answerable.
    return Annotation(
        category=category,
        synset=lvis_synset(lvis_category),
        width=width,
        length=length,
        height=height,
        mass=mass,
        description=obj["description"].strip(),
        materials=list(dict.fromkeys(normalised)),
        on_ceiling=flags["onCeiling"],
        on_wall=flags["onWall"],
        on_floor=flags["onFloor"],
        on_object=flags["onObject"],
        lvis_category=lvis_category,
        identity_confirmed=identity_confirmed,
        category_relation=category_relation(lvis_category, category),
        width_axis=width_axis,
    )
