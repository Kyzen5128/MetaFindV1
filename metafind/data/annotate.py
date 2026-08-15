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

Paper 2.3 asks for "rich textual descriptions detailing attributes **such as**
object category, size dimensions, materials, and placement constraints".
"Such as" gives examples, not a schema, so everything below is ours:

  * exactly these four fields plus a free-text description
  * all of them required
  * ``placement_constraints`` drawn from a CLOSED vocabulary
  * ``dimensions`` in metres

Keeping them is right -- without a schema there is no verifiable artifact, and
``placement_constraints`` is the signal that makes layout-aware retrieval mean
anything. But they are reported as implementation choices, never as paper
requirements.

Why the size field can only ever be a guess
-------------------------------------------

n04 fits every asset to a unit sphere before rendering, so the annotator sees no
absolute scale at all (F13). Its size estimate is therefore a category prior --
"a dining chair is about 0.9 m tall" -- and the prompt says so explicitly rather
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
    "PLACEMENT_VOCABULARY",
    "PROMPT_VERSION",
    "AnnotationError",
    "build_prompt",
    "build_repair_prompt",
    "parse_annotation",
    "validate_annotation",
]

PROMPT_VERSION = 1
MAX_ATTEMPTS = 2  # C1's hard bound; the third outcome is quarantine

# [IMPLEMENTATION CHOICE, U-11-adjacent but distinct] The paper names
# "placement constraints" and stops. A closed set is what makes the field
# checkable at all -- free text would pass any schema and mean nothing to a
# layout encoder. These describe what an object RESTS ON or ATTACHES TO, which
# is the reading that connects to scene layout: paper 2.3's own example of a
# physical relation is "cup on table".
PLACEMENT_VOCABULARY = (
    "floor",          # stands on the ground: chairs, tables, cabinets
    "tabletop",       # sits on a horizontal surface above the floor
    "wall_mounted",   # attaches to a vertical surface: pictures, shelves
    "ceiling_mounted",  # hangs: lamps, fans
    "shelf",          # stored inside or on shelving
    "handheld",       # has no resting place of its own: tools, cups in use
    "outdoor_ground",  # belongs outside: vehicles, plants, rocks
    "unconstrained",  # genuinely no constraint, e.g. an abstract shape
)

REQUIRED_FIELDS = ("category", "description", "dimensions", "materials",
                   "placement_constraints")
DIMENSION_KEYS = ("length_m", "width_m", "height_m")
# Objaverse holds everything from a bead to a building. Anything outside this
# is a unit error, not an object.
MIN_DIM_M, MAX_DIM_M = 0.001, 100.0


class AnnotationError(Exception):
    """A schema violation, carrying the message that goes into the repair prompt."""


@dataclass(frozen=True)
class Annotation:
    category: str
    description: str
    dimensions: dict[str, float]
    materials: list[str]
    placement_constraints: list[str]

    def as_record(self, annotator_model: str) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "dimensions": self.dimensions,
            "materials": self.materials,
            "placement_constraints": self.placement_constraints,
            "annotator_model": annotator_model,
        }


def build_prompt(n_views: int) -> str:
    """The annotation prompt. Pinned, versioned, and tested against a golden string.

    The scale paragraph is not politeness. n04 normalises every asset to a unit
    sphere, so nothing in these images carries absolute size; a prompt that asks
    for dimensions without saying so invites a confident measurement of an
    object whose scale is not in the picture.
    """
    vocab = ", ".join(PLACEMENT_VOCABULARY)
    return (
        f"You are looking at {n_views} rendered views of a single 3D asset.\n"
        "\n"
        "IMPORTANT: these renders are SCALE-NORMALISED. Every asset is fitted to "
        "the same size before rendering, so the images contain no information "
        "about how large the object really is. Estimate its dimensions from what "
        "kind of object it is, not from the picture, and do not describe the "
        "estimate as a measurement.\n"
        "\n"
        "Return one JSON object and nothing else, with exactly these fields:\n"
        '  "category": a short noun phrase naming the object\n'
        '  "description": one or two sentences covering shape, style and colour\n'
        '  "dimensions": {"length_m": float, "width_m": float, "height_m": float} '
        "in METRES\n"
        '  "materials": a list of material names, most prominent first\n'
        f'  "placement_constraints": a list drawn ONLY from: {vocab}\n'
        "\n"
        "Use no other placement values. If none applies, use "
        '["unconstrained"].'
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

    if not isinstance(obj["category"], str) or not obj["category"].strip():
        raise AnnotationError("`category` must be a non-empty string")
    if not isinstance(obj["description"], str) or not obj["description"].strip():
        raise AnnotationError("`description` must be a non-empty string")

    dims = obj["dimensions"]
    if not isinstance(dims, dict):
        raise AnnotationError("`dimensions` must be an object with length_m, width_m, height_m")
    missing_d = [k for k in DIMENSION_KEYS if k not in dims]
    if missing_d:
        raise AnnotationError(f"`dimensions` is missing {', '.join(missing_d)}")
    clean_dims: dict[str, float] = {}
    for k in DIMENSION_KEYS:
        v = dims[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnnotationError(f"`dimensions.{k}` must be a number, got {v!r}")
        v = float(v)
        if not MIN_DIM_M <= v <= MAX_DIM_M:
            # Almost always millimetres: 900 for a chair rather than 0.9.
            raise AnnotationError(
                f"`dimensions.{k}` = {v} is outside {MIN_DIM_M}-{MAX_DIM_M} m. "
                "Dimensions must be in METRES, not millimetres or centimetres."
            )
        clean_dims[k] = v

    materials = obj["materials"]
    if isinstance(materials, str):
        materials = [materials]
    if not isinstance(materials, list) or not materials:
        raise AnnotationError("`materials` must be a non-empty list of strings")
    if any(not isinstance(m, str) or not m.strip() for m in materials):
        raise AnnotationError("every entry in `materials` must be a non-empty string")

    placement = obj["placement_constraints"]
    if isinstance(placement, str):
        placement = [placement]
    if not isinstance(placement, list) or not placement:
        raise AnnotationError(
            "`placement_constraints` must be a non-empty list drawn from "
            f"{', '.join(PLACEMENT_VOCABULARY)}"
        )
    bad = [p for p in placement if p not in PLACEMENT_VOCABULARY]
    if bad:
        raise AnnotationError(
            f"placement value(s) {bad} are not allowed. Use only: "
            f"{', '.join(PLACEMENT_VOCABULARY)}"
        )

    return Annotation(
        category=obj["category"].strip(),
        description=obj["description"].strip(),
        dimensions=clean_dims,
        materials=[m.strip() for m in materials],
        placement_constraints=list(dict.fromkeys(placement)),
    )
