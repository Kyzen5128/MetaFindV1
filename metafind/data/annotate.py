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

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PLACEMENT_FLAGS",
    "MATERIAL_SYNONYMS",
    "PROMPT_VERSION",
    "AnnotationError",
    "build_prompt",
    "build_repair_prompt",
    "parse_annotation",
    "validate_annotation",
]

PROMPT_VERSION = 3  # v2 followed Figure 2 but left 41% with all four flags false
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

REQUIRED_FIELDS = ("category", "synset", "width", "length", "height", "mass",
                   "description", "materials", *PLACEMENT_FLAGS)
DIMENSION_KEYS = ("width", "length", "height")

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
            "prompt_version": PROMPT_VERSION,
            "annotator_model": annotator_model,
        }


def build_prompt(n_views: int) -> str:
    """The annotation prompt. Pinned, versioned, and tested against a golden string.

    The scale paragraph is not politeness. n04 normalises every asset to a unit
    sphere, so nothing in these images carries absolute size; a prompt that asks
    for dimensions without saying so invites a confident measurement of an
    object whose scale is not in the picture.
    """
    return (
        f"You are looking at {n_views} rendered views of a single 3D asset.\n"
        "\n"
        "IMPORTANT: these renders are SCALE-NORMALISED. Every asset is fitted to "
        "the same size before rendering, so the images contain no information "
        "about how large the object really is. Estimate its size from what kind "
        "of object it is, not from the picture, and do not describe the estimate "
        "as a measurement. The PROPORTIONS between width, length and height ARE "
        "visible and should be read from the images.\n"
        "\n"
        "Return one JSON object and nothing else, with exactly these fields:\n"
        '  "category": a SPECIFIC noun phrase naming the object. Name the thing '
        'itself, not the class it belongs to: "sofa" not "furniture", '
        '"rabbit" not "animal", "toy dinosaur" not "toy"\n'
        '  "synset": the closest WordNet synset id, e.g. "robot.n.01"\n'
        '  "width": number, CENTIMETRES (left-to-right extent)\n'
        '  "length": number, CENTIMETRES (front-to-back extent)\n'
        '  "height": number, CENTIMETRES (floor-to-top extent)\n'
        '  "mass": number, KILOGRAMS\n'
        '  "description": one or two sentences covering shape, style and colour\n'
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


def validate_annotation(obj: dict[str, Any]) -> Annotation:
    """[L1-ANNOT-SCHEMA] Every rule, with the message the repair prompt needs.

    Each message names the offending field and value, because it is fed back
    verbatim and a message like "invalid schema" gives the model nothing to act
    on.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in obj]
    if missing:
        raise AnnotationError(f"required field(s) missing: {', '.join(missing)}")

    for field in ("category", "description", "synset"):
        if not isinstance(obj[field], str) or not obj[field].strip():
            raise AnnotationError(f"`{field}` must be a non-empty string")

    # WordNet ids are `lemma.pos.NN`. Checked for SHAPE only -- verifying the
    # id exists needs a WordNet corpus this environment does not carry, so a
    # well-formed but invented synset passes here and is measured downstream
    # rather than silently trusted.
    synset = obj["synset"].strip()
    parts = synset.split(".")
    if len(parts) != 3 or parts[1] not in ("n", "v", "a", "s", "r") or not parts[2].isdigit():
        raise AnnotationError(
            f"`synset` = {synset!r} is not a WordNet id of the form "
            '"lemma.n.01" (lemma, part-of-speech, two-digit sense number)'
        )

    dims: dict[str, float] = {}
    for k in DIMENSION_KEYS:
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnnotationError(f"`{k}` must be a number, got {v!r}")
        v = float(v)
        if not MIN_DIM_CM <= v <= MAX_DIM_CM:
            # v1's failure mode inverted: the old schema was metres and got
            # millimetres, this one is centimetres and will get metres (0.9 for
            # a chair rather than 90).
            raise AnnotationError(
                f"`{k}` = {v} is outside {MIN_DIM_CM}-{MAX_DIM_CM} cm. "
                "Sizes must be in CENTIMETRES, not metres or millimetres."
            )
        dims[k] = v

    mass = obj["mass"]
    if isinstance(mass, bool) or not isinstance(mass, (int, float)):
        raise AnnotationError(f"`mass` must be a number, got {mass!r}")
    mass = float(mass)
    if not MIN_MASS_KG <= mass <= MAX_MASS_KG:
        raise AnnotationError(
            f"`mass` = {mass} is outside {MIN_MASS_KG}-{MAX_MASS_KG} kg. "
            "Mass must be in KILOGRAMS."
        )

    # [IMPLEMENTATION CHOICE] Cross-field consistency, which v1 could not do
    # because it had no mass. The range check alone cannot catch a metres
    # answer: 0.5 x 0.5 x 0.9 is inside 0.1-10000 cm, so a chair given in
    # metres passes every per-field rule and lands in the corpus silently.
    # Density catches it -- 0.225 cm^3 weighing 6 kg is 26 kg/cm^3, about
    # 26,000x water. The bounds are deliberately loose (water is 0.001, osmium
    # 0.0225) because this is a UNIT-ERROR detector, not a physics check, and a
    # bounding box is mostly air for most objects.
    density = mass / (dims["width"] * dims["length"] * dims["height"])
    if not MIN_DENSITY_KG_CM3 <= density <= MAX_DENSITY_KG_CM3:
        raise AnnotationError(
            f"`mass` = {mass} kg with a "
            f"{dims['width']} x {dims['length']} x {dims['height']} cm bounding "
            f"box implies {density:.3g} kg/cm^3, which is not physical "
            f"(water is 0.001). Sizes must be in CENTIMETRES and mass in "
            f"KILOGRAMS -- check you did not give sizes in metres."
        )

    materials = obj["materials"]
    if isinstance(materials, str):
        materials = [materials]
    if not isinstance(materials, list) or not materials:
        raise AnnotationError("`materials` must be a non-empty list of strings")
    if any(not isinstance(m, str) or not m.strip() for m in materials):
        raise AnnotationError("every entry in `materials` must be a non-empty string")
    # Normalise spelling, preserve order and content. See MATERIAL_SYNONYMS:
    # nothing is dropped, only `metallic`-style variants folded onto one token.
    normalised = [MATERIAL_SYNONYMS.get(m.strip().lower(), m.strip().lower())
                  for m in materials]

    flags: dict[str, bool] = {}
    for f in PLACEMENT_FLAGS:
        v = obj[f]
        if not isinstance(v, bool):
            raise AnnotationError(
                f"`{f}` must be true or false, got {v!r}. The four placement "
                "fields are independent booleans, not a category choice."
            )
        flags[f] = v
    # Deliberately NOT requiring at least one true. An asset with all four false
    # is a real answer -- an abstract shape belongs nowhere in particular -- and
    # v1's equivalent escape hatch (`unconstrained`) absorbed 30.7% of the
    # corpus precisely because the schema demanded a positive answer.

    return Annotation(
        category=obj["category"].strip(),
        synset=synset,
        width=dims["width"],
        length=dims["length"],
        height=dims["height"],
        mass=mass,
        description=obj["description"].strip(),
        materials=list(dict.fromkeys(normalised)),
        on_ceiling=flags["onCeiling"],
        on_wall=flags["onWall"],
        on_floor=flags["onFloor"],
        on_object=flags["onObject"],
    )
