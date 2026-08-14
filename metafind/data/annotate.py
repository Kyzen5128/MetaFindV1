"""Annotate assets from their rendered views (paper sec. 2.3).

    "Each asset is rendered from 11 orthogonal viewpoints and annotated using
    GPT-4o. These annotations provide rich textual descriptions detailing
    attributes such as object category, size dimensions, materials, and
    placement constraints."

GPT-4o is replaced by a local Qwen2.5-VL. That is a declared deviation: the
captions are what the text tower trains on, so a different annotator gives a
different text distribution. It is recorded per asset via ``annotator_model``.

Schema, and why it is validated
-------------------------------

The four attributes the paper names become required fields. A vision-language
model asked for JSON will sometimes wrap it in prose, emit a trailing comma, or
invent a fifth field; none of that raises an exception anywhere, it just
produces a caption that is subtly not what was asked for. So the response is
parsed and checked, and a failure feeds the specific error back for one bounded
retry -- MODEL_RECOVERABLE in the graph spec, not a blind retry, since resending
the identical prompt would only reproduce the identical mistake.

Placement constraints matter most
---------------------------------

They are the signal that makes layout-aware retrieval possible at all: whether a
thing stands on the floor, sits on a surface, mounts to a wall. That is exactly
what ULIP-2's shipped captions do not have, and the reason this stage cannot be
skipped in favour of them.

The size field is a category prior, not a measurement (F13)
-----------------------------------------------------------

Meshes are normalised into a unit sphere before rendering, so the images
preserve aspect ratio but destroy absolute scale: a 1.8 m table and a 0.1 m cup
land at the same size. Whatever dimensions the model reports are inferred from
what the object *is*, not seen. The render sidecar stores the true bounding box
alongside, which makes the estimate auditable rather than merely plausible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "AnnotationConfig",
    "ANNOTATION_SCHEMA",
    "build_prompt",
    "parse_annotation",
    "SchemaError",
    "VisionBackend",
]


class SchemaError(ValueError):
    """The response did not satisfy the annotation schema."""


ANNOTATION_SCHEMA: dict[str, Any] = {
    "category": "str, the object class in a word or two",
    "description": "str, one or two sentences on appearance",
    "dimensions": {
        "width_m": "float, estimated real-world width in metres",
        "height_m": "float",
        "depth_m": "float",
    },
    "materials": "list[str], the main materials",
    "placement_constraints": (
        "list[str], drawn from: floor_standing, tabletop, wall_mounted, "
        "ceiling_mounted, handheld, outdoor, stackable, requires_support"
    ),
}

PLACEMENT_VOCABULARY = {
    "floor_standing",
    "tabletop",
    "wall_mounted",
    "ceiling_mounted",
    "handheld",
    "outdoor",
    "stackable",
    "requires_support",
}

# Loose bounds only. They exist to catch a model emitting millimetres, or a
# hallucinated 400 m chair -- not to second-guess a plausible estimate.
MIN_DIM_M = 0.001
MAX_DIM_M = 100.0


@dataclass
class AnnotationConfig:
    """Settings for the annotation pass.

    Attributes:
        model_id: HuggingFace id of the vision-language model.
        n_views: how many of the rendered views to show. All 11 follows the
            paper; fewer cuts cost roughly linearly.
        max_new_tokens: generation cap.
        temperature: 0 for greedy decoding. The annotations become a frozen
            artifact everything downstream reads, so run-to-run variation buys
            nothing and costs reproducibility.
        max_repair_attempts: bounded MODEL_RECOVERABLE retries after a schema
            failure, with the error fed back.
    """

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    n_views: int = 11
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_repair_attempts: int = 2
    prompt_version: str = "annotate/1"
    placement_vocabulary: set[str] = field(default_factory=lambda: set(PLACEMENT_VOCABULARY))


class VisionBackend(Protocol):
    """Anything that turns (images, prompt) into text.

    Defined as a protocol so the schema, prompt and repair loop are testable
    without a 16 GB checkpoint, and so the annotator is not welded to Qwen.
    """

    def generate(self, images: list, prompt: str) -> str: ...


def build_prompt(cfg: AnnotationConfig, repair_error: str | None = None) -> str:
    """Build the annotation prompt, optionally with a repair instruction.

    Args:
        cfg: annotation settings.
        repair_error: the schema error from the previous attempt. Included so
            the retry differs from the original request -- resending the same
            prompt would just reproduce the same malformed output.
    """
    schema = json.dumps(ANNOTATION_SCHEMA, indent=2, ensure_ascii=False)
    parts = [
        f"You are shown {cfg.n_views} rendered views of a single 3D asset from "
        "different angles. They are all the same object.",
        "",
        "Describe the object as JSON with exactly these fields:",
        schema,
        "",
        "Rules:",
        "- Reply with the JSON object only. No prose, no markdown fences.",
        "- placement_constraints must use only these values: "
        + ", ".join(sorted(cfg.placement_vocabulary)),
        "- dimensions are your best estimate of the real object in metres. The "
        "renders are scale-normalised, so judge by what the object is.",
        "- materials should describe the surfaces you can see.",
    ]
    if repair_error:
        parts += [
            "",
            "Your previous reply was rejected for this reason:",
            f"    {repair_error}",
            "Return corrected JSON that fixes exactly that problem.",
        ]
    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in ``` fences or a sentence of preamble often enough that
    refusing anything but a bare object would send most valid answers into the
    repair loop for no reason.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise SchemaError("no JSON object found in the reply") from None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaError(f"reply is not valid JSON: {exc}") from None

    if not isinstance(obj, dict):
        raise SchemaError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def parse_annotation(text: str, cfg: AnnotationConfig | None = None) -> dict:
    """Parse and validate one annotation.

    Args:
        text: the raw model reply.
        cfg: settings, for the placement vocabulary.

    Returns:
        The validated annotation.

    Raises:
        SchemaError: with a message specific enough to be worth feeding back.
    """
    cfg = cfg or AnnotationConfig()
    obj = _extract_json(text)

    missing = [k for k in ("category", "description", "dimensions", "materials",
                           "placement_constraints") if k not in obj]
    if missing:
        raise SchemaError(f"missing required field(s): {', '.join(missing)}")

    for key in ("category", "description"):
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise SchemaError(f"{key} must be a non-empty string")

    dims = obj["dimensions"]
    if not isinstance(dims, dict):
        raise SchemaError("dimensions must be an object with width_m, height_m, depth_m")
    for axis in ("width_m", "height_m", "depth_m"):
        if axis not in dims:
            raise SchemaError(f"dimensions is missing {axis}")
        try:
            value = float(dims[axis])
        except (TypeError, ValueError):
            raise SchemaError(f"dimensions.{axis} is not a number: {dims[axis]!r}") from None
        if not MIN_DIM_M <= value <= MAX_DIM_M:
            raise SchemaError(
                f"dimensions.{axis} = {value} is outside {MIN_DIM_M}-{MAX_DIM_M} m; "
                "give metres, not millimetres or centimetres"
            )
        dims[axis] = value

    for key in ("materials", "placement_constraints"):
        if not isinstance(obj[key], list) or not obj[key]:
            raise SchemaError(f"{key} must be a non-empty list")
        if not all(isinstance(v, str) and v.strip() for v in obj[key]):
            raise SchemaError(f"{key} must contain only non-empty strings")

    unknown = [v for v in obj["placement_constraints"] if v not in cfg.placement_vocabulary]
    if unknown:
        raise SchemaError(
            f"placement_constraints contains value(s) outside the allowed set: "
            f"{', '.join(unknown)}. Allowed: {', '.join(sorted(cfg.placement_vocabulary))}"
        )

    return obj


def annotate_one(
    backend: VisionBackend, images: list, cfg: AnnotationConfig | None = None
) -> tuple[dict, list[str]]:
    """Annotate a single asset, repairing bounded schema failures.

    Args:
        backend: the vision-language model.
        images: rendered views, most informative first.
        cfg: annotation settings.

    Returns:
        ``(annotation, errors)`` where ``errors`` lists the schema failures that
        were repaired along the way -- kept so a high repair rate is visible
        rather than hidden behind eventual success.

    Raises:
        SchemaError: if every attempt failed.
    """
    cfg = cfg or AnnotationConfig()
    errors: list[str] = []
    last: SchemaError | None = None

    for attempt in range(cfg.max_repair_attempts + 1):
        prompt = build_prompt(cfg, repair_error=errors[-1] if errors else None)
        try:
            return parse_annotation(backend.generate(images, prompt), cfg), errors
        except SchemaError as exc:
            last = exc
            errors.append(str(exc))
            if attempt == cfg.max_repair_attempts:
                break

    raise SchemaError(
        f"schema validation failed after {cfg.max_repair_attempts + 1} attempts: {last}"
    )
