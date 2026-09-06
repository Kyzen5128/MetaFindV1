"""PROMPT_VERSION 10 -- the paper's Figure 2 annotation, asked for in one call.

# SUPPORTS-NODE: n05_annotate

[DL-103, Kyzen 2026-09-06 「這次資料集處理方法直接採用論文所描述的 ... 除了 LLM
其他一律按照論文要求」] The paper's data preparation (`2methdology.tex` §2.3):
"Each asset is rendered from 11 orthogonal viewpoints and annotated using
GPT-4o. These annotations provide rich textual descriptions detailing
attributes such as object category, size dimensions, materials, and placement
constraints." Figure 2 prints the schema (see `annotate.FIGURE_2_EXAMPLE`).

What v10 does, and what each choice rests on
--------------------------------------------
* ONE structured call returns the whole Figure 2 object.            PAPER (Figure 2)
  v8/v9 drew five descriptions, ranked them with CLIP-ViT-L and
  kept the one that fit 77 tokens; the paper has one description.
* The VLM estimates width / length / height itself (centimetres). PAPER (§2.3 "size
  v5-v9 asked for the height only and derived the other two from   dimensions"; Figure 2
  the mesh's exact proportions.  The mesh proportions still travel  prints 30 / 30 / 40)
  in the record as a DIAGNOSTIC field; they are not used.
* volume = width x length x height, computed.                       PAPER (Figure 2:
                                                                     36000 = 30 * 30 * 40)
* `synset` is asked of the model (Figure 2 prints it); a malformed  IMPLEMENTATION CHOICE
  or unknown id falls back to `annotate.resolve_synset`.
* The Objaverse-LVIS label is still shown as the catalogue identity. IMPLEMENTATION CHOICE
  The paper does not say whether GPT-4o saw it; without it gemma      (paper silent; v7
  misidentified ~28% of a 97-asset sample (PROMPT_VERSION 7 note).    measurement)
* All 11 views go in one turn (`Annotator.generate`).               PAPER (§2.3)
* Description: one sentence, English, like the Figure 2 example.    PAPER (Figure 2)
  Length is instructed, not refused: the text encoder truncates at
  77 tokens whatever we do, and the paper's own example is ~25 words.
* Retries: a failed parse or validation gets a repair prompt; the   IMPLEMENTATION CHOICE
  last attempt samples with a per-attempt seed, so "retry" is not     (fixes the 311
  a byte-identical resend.                                            no-op retries)

Units: centimetres and kilograms, as `annotate.DIMENSION_UNIT` / `MASS_UNIT`
record (INFERENCE from Figure 2's arithmetic; see that module).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from metafind.data.annotate import (
    DIMENSION_UNIT, FIGURE_2_EXAMPLE, MASS_UNIT, MATERIAL_SYNONYMS, MAX_DENSITY_KG_CM3,
    MAX_DIM_CM, MAX_MASS_KG, MIN_DENSITY_KG_CM3, MIN_DIM_CM, MIN_MASS_KG, PLACEMENT_FLAGS,
    SYNSET_PATTERN, AnnotationError, build_repair_prompt, category_relation,
    non_english_characters, parse_annotation, resolve_synset,
)

PROMPT_VERSION = 10
VALIDATOR_VERSION = 6
SCHEMA_VERSION = 7
MAX_ATTEMPTS = 3
DESCRIPTION_WORDS_HINT = 30           # instruction to the model, not an admission rule
MAX_DESCRIPTION_CHARS = 600           # admission: anything longer is not "one sentence"
MIN_DESCRIPTION_CHARS = 12
MAX_MATERIALS = 6                     # more than this is refused (repair), never silently cut
REQUIRED_N_VIEWS = 11                 # PAPER 2methdology.tex:28; renderer v7 layout (DL-077 Q2)
REQUIRED_RENDERER_VERSION = 7
# [ULIP2 reviewer 2026-09-06, MINOR 5] the sampled retry seed is uid_seed(uid) + attempt + RETRY_SALT;
# a re-run over v10's own quarantine passes a new salt (`annotate_run --retry-salt`) so it is not
# the same three attempts again. Recorded in every record.
RETRY_SALT = 0
ANNOTATION_CONTRACT_FAMILY = "metafind_annot"

# The paper's field set, in Figure 2 order. `RECORD_REQUIRED_FIELDS` is what a
# stored v10 record must carry to count as complete (the runner's contract test).
FIGURE2_FIELDS = ("category", "synset", "width", "length", "height", "volume", "mass",
                  "description", "materials", *PLACEMENT_FLAGS)
RESPONSE_REQUIRED = ("category", "synset", "width", "length", "height", "mass",
                     "description", "materials", *PLACEMENT_FLAGS)
RECORD_REQUIRED_FIELDS = FIGURE2_FIELDS

CANONICAL_N_VIEWS = 11
CANONICAL_LVIS_CATEGORY = "chair"


def build_prompt(n_views: int, lvis_category: str) -> str:
    """One turn, the whole Figure 2 object. The LVIS label is the catalogue identity."""
    return (
        f"You are looking at {n_views} rendered views of a single 3D asset, taken from "
        "evenly spaced directions around it.\n"
        "\n"
        f'This asset is catalogued in Objaverse-LVIS as: "{lvis_category}"\n'
        "Treat that identity as correct unless the images clearly contradict it; if they "
        "do, name what you actually see.\n"
        "\n"
        "Describe the asset as a structured annotation. Here is the exact format, shown "
        "with a worked example for a different object:\n"
        f"{FIGURE_2_EXAMPLE}\n"
        "\n"
        "Return ONE JSON object and nothing else, with exactly these fields in this order:\n"
        "  category     the specific object type, in English (1-4 words)\n"
        "  synset       the WordNet id of that category, like \"robot.n.01\"\n"
        "  width        real-world width in centimetres (a number)\n"
        "  length       real-world length / depth in centimetres (a number)\n"
        "  height       real-world height in centimetres (a number)\n"
        "  volume       width x length x height, in cubic centimetres\n"
        "  mass         typical mass in kilograms (a number)\n"
        f"  description  ONE English sentence of at most about {DESCRIPTION_WORDS_HINT} words "
        "describing what you see: shape, parts, colours, surface, any text or marking\n"
        "  materials    a JSON list of 1-6 English material words (e.g. \"wood\", \"metal\")\n"
        "  onCeiling, onWall, onFloor, onObject   booleans: where this object is typically placed\n"
        "\n"
        "The renders are scale-normalised, so estimate the real-world size from what the "
        "object IS, not from how large it appears. Write every text field in English only "
        "(no Chinese, Japanese, Korean, Cyrillic or Arabic characters). Use plain numbers, "
        "not strings, for width, length, height, volume and mass."
    )


def contract() -> dict[str, Any]:
    """Everything that decides admission, hashed into the contract id."""
    return {
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt": build_prompt(CANONICAL_N_VIEWS, CANONICAL_LVIS_CATEGORY),
        "response_required": list(RESPONSE_REQUIRED),
        "record_required": list(RECORD_REQUIRED_FIELDS),
        "placement_flags": list(PLACEMENT_FLAGS),
        "dimension_unit": DIMENSION_UNIT,
        "mass_unit": MASS_UNIT,
        "dim_bounds_cm": [MIN_DIM_CM, MAX_DIM_CM],
        "mass_bounds_kg": [MIN_MASS_KG, MAX_MASS_KG],
        "density_bounds_kg_cm3": [MIN_DENSITY_KG_CM3, MAX_DENSITY_KG_CM3],
        "max_description_chars": MAX_DESCRIPTION_CHARS,
        "max_materials": MAX_MATERIALS,
        "material_synonyms": dict(sorted(MATERIAL_SYNONYMS.items())),
        "max_attempts": MAX_ATTEMPTS,
        "min_description_chars": MIN_DESCRIPTION_CHARS,
        "required_n_views": REQUIRED_N_VIEWS,
        "required_renderer_version": REQUIRED_RENDERER_VERSION,
        "synset_existence": "wordnet lookup of a well-formed id; lookup fallback otherwise",
        "volume": "computed = width * length * height",
    }


def contract_id() -> str:
    payload = json.dumps(contract(), sort_keys=True, ensure_ascii=False)
    return f"{ANNOTATION_CONTRACT_FAMILY}_v{PROMPT_VERSION}@{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _synset_exists(synset_id: str) -> bool:
    """[ULIP2 reviewer MINOR 4] a well-formed id that names no WordNet synset is not a synset.
    WordNet unavailable -> accept the shape (recorded through the source string upstream)."""
    from metafind.data.annotate import _wordnet
    wn = _wordnet()
    if wn is None:
        return True
    try:
        wn.synset(synset_id)
        return True
    except Exception:  # noqa: BLE001 -- nltk raises WordNetError / ValueError for unknown ids
        return False


def _number(obj: dict, key: str, lo: float, hi: float, unit: str) -> float:
    v = obj.get(key)
    if isinstance(v, str):
        try:
            v = float(v.replace(",", "").split()[0])
        except (ValueError, IndexError):
            raise AnnotationError(f"`{key}` must be a number, got {obj.get(key)!r}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AnnotationError(f"`{key}` must be a number, got {v!r}")
    v = float(v)
    if not lo <= v <= hi:
        raise AnnotationError(f"`{key}` = {v} is outside {lo}-{hi} {unit}. Re-check the unit.")
    return v


def validate(obj: dict[str, Any], *, lvis_category: str) -> dict[str, Any]:
    """The model's object -> the Figure 2 fields plus what the validator decided.

    Raises AnnotationError with a message the repair prompt can quote.
    """
    # Figure 2 prints the object under an "annotations" wrapper, and the model often
    # returns it that way; the fields are the same either way.
    if set(obj) == {"annotations"} and isinstance(obj["annotations"], dict):
        obj = obj["annotations"]
    missing = [f for f in RESPONSE_REQUIRED if f not in obj]
    if missing:
        raise AnnotationError(f"required field(s) missing: {', '.join(missing)}")
    for field in ("category", "description"):
        if not isinstance(obj[field], str) or not obj[field].strip():
            raise AnnotationError(f"`{field}` must be a non-empty string")
        bad = non_english_characters(obj[field])
        if bad:
            raise AnnotationError(f"`{field}` contains non-English characters {bad[:5]!r}; write it in English")
    category = " ".join(obj["category"].strip().lower().split())
    if len(category.split()) > 6:
        raise AnnotationError("`category` must be a short object type (at most a few words)")
    description = " ".join(obj["description"].strip().split())
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise AnnotationError(f"`description` is {len(description)} characters; write ONE sentence")
    if len(description) < MIN_DESCRIPTION_CHARS:
        raise AnnotationError("`description` is too short to describe the object; write ONE full sentence")

    width = _number(obj, "width", MIN_DIM_CM, MAX_DIM_CM, "cm")
    length = _number(obj, "length", MIN_DIM_CM, MAX_DIM_CM, "cm")
    height = _number(obj, "height", MIN_DIM_CM, MAX_DIM_CM, "cm")
    mass = _number(obj, "mass", MIN_MASS_KG, MAX_MASS_KG, "kg")
    volume = width * length * height
    density = mass / volume
    if not MIN_DENSITY_KG_CM3 <= density <= MAX_DENSITY_KG_CM3:
        raise AnnotationError(
            f"`mass` = {mass} kg with a {width} x {length} x {height} cm box gives "
            f"{density:.3g} kg/cm^3, which is not physical (water is 0.001). Check that the "
            "dimensions are in CENTIMETRES and the mass in KILOGRAMS.")

    materials = obj["materials"]
    if isinstance(materials, str):
        materials = [m for m in materials.replace(";", ",").split(",")]
    if not isinstance(materials, list) or not materials:
        raise AnnotationError("`materials` must be a non-empty JSON list of material words")
    norm = []
    for m in materials:
        if not isinstance(m, str) or not m.strip():
            raise AnnotationError("`materials` entries must be non-empty strings")
        if non_english_characters(m):
            raise AnnotationError(f"material {m!r} is not English")
        w = " ".join(m.strip().lower().split())
        w = MATERIAL_SYNONYMS.get(w, w)
        if w not in norm:
            norm.append(w)
    if len(norm) > MAX_MATERIALS:
        raise AnnotationError(f"`materials` lists {len(norm)} entries; give at most {MAX_MATERIALS} material words")
    materials = norm

    flags = {}
    for f in PLACEMENT_FLAGS:
        v = obj[f]
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if not isinstance(v, bool):
            raise AnnotationError(f"`{f}` must be true or false, got {obj[f]!r}")
        flags[f] = v

    synset_raw = str(obj.get("synset", "")).strip().lower()
    if SYNSET_PATTERN.match(synset_raw) and _synset_exists(synset_raw):
        synset, synset_source = synset_raw, "model"
    else:
        synset, synset_source = resolve_synset(category, lvis_category)
        why = "malformed" if not SYNSET_PATTERN.match(synset_raw) else "unknown_to_wordnet"
        synset_source = f"{synset_source}_after_{why}_model_synset"

    return {
        "category": category, "synset": synset, "synset_source": synset_source,
        "width": width, "length": length, "height": height, "volume": volume, "mass": mass,
        "description": description, "materials": materials, **flags,
        "lvis_category": lvis_category,
        "category_relation": category_relation(lvis_category, category),
    }


def build_record(fields: dict[str, Any], *, uid: str, model_id: str, attempt: int,
                 sampled_seed: int | None, render_rec: dict, proportions,
                 prefix_reuse: bool, raw_response: str, first_error: str = "") -> dict[str, Any]:
    from metafind.data.view_io import image_identity
    return {
        **fields,
        "dimension_unit": DIMENSION_UNIT,
        "mass_unit": MASS_UNIT,
        "volume_source": "computed_width_x_length_x_height",
        "dimensions_source": "vlm_estimate",           # v5-v9: mesh proportions x height
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "annotation_contract": contract_id(),
        "annotator_model": model_id,
        "description_source": "model",
        "uid": uid,
        "attempts": attempt,
        "first_error": first_error,                    # why attempt 1 was refused, if it was
        "sampled_seed": sampled_seed,                  # None = greedy attempt
        "retry_salt": RETRY_SALT,
        "vision_prefix_reuse": prefix_reuse,
        "n_views_shown": len(render_rec["view_paths"]),
        "image_identity": image_identity(render_rec),
        "renderer_version": render_rec.get("renderer_version"),
        "raw_bbox_extents": render_rec.get("raw_bbox_extents"),
        "mesh_proportions_yxz": list(proportions) if proportions is not None else None,
        "raw_response_tail": raw_response[-400:],
    }


def annotate_one(ann, uid: str, render_rec: dict, *, lvis_category: str, proportions,
                 preloaded_images=None) -> tuple[dict | None, dict | None]:
    """One asset under v10. Returns ``(record, quarantine_entry)`` -- exactly one is None."""
    from metafind.data.annotate_run import _is_cuda_oom, _release_cuda
    from metafind.data.pointclouds import uid_seed

    views = render_rec["view_paths"]
    # [ULIP2 reviewer MAJOR 3] the contract is hashed on 11 views; refuse anything else.
    if len(views) != REQUIRED_N_VIEWS or render_rec.get("renderer_version") != REQUIRED_RENDERER_VERSION:
        return None, {"uid": uid, "failure_class": "DETERMINISTIC_INPUT", "exception_type": "WrongRenderProtocol",
                      "exception_msg": (f"{len(views)} views, renderer_version {render_rec.get('renderer_version')}; "
                                        f"v10 requires {REQUIRED_N_VIEWS} views from renderer v{REQUIRED_RENDERER_VERSION}"),
                      "attempts": 0, "traceback": ""}
    prompt = build_prompt(len(views), lvis_category)
    # No SharedViewPrefix here. It exists to reuse one prefill across v9's five description
    # draws plus the structured call; v10 makes ONE call per asset, so there is nothing to
    # share -- and measured on the 2026-09-06 smoke, the cached-prefix path returned malformed
    # JSON on the first call of every asset (plain `generate` returned valid JSON first time).
    prefix_reuse = False

    def gen(p, **kw):
        try:
            return ann.generate(views, p, **kw)
        except Exception as exc:  # noqa: BLE001 -- an OOM must free the card before the next try
            if _is_cuda_oom(exc):
                _release_cuda()
            raise

    current, last_error, last_raw, first_error = prompt, "", "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # attempts 1-2 greedy (the repair prompt already differs); the last one samples
        # with a per-attempt seed so a deterministic failure is not resent byte for byte.
        sampled_seed = (uid_seed(uid) + attempt + RETRY_SALT) if attempt == MAX_ATTEMPTS else None
        try:
            raw = gen(current, sample=True, seed=sampled_seed) if sampled_seed is not None else gen(current)
        except Exception as exc:  # noqa: BLE001 -- surface as quarantine, not a crash
            import traceback as _tb
            return None, {"uid": uid, "failure_class": "RESOURCE" if _is_cuda_oom(exc) else "UNKNOWN",
                          "exception_type": type(exc).__name__, "exception_msg": str(exc)[:400],
                          "attempts": attempt, "traceback": _tb.format_exc()[-1500:]}
        last_raw = raw if isinstance(raw, str) else (raw[0] if raw else "")
        try:
            fields = validate(parse_annotation(last_raw), lvis_category=lvis_category)
        except AnnotationError as exc:
            last_error = str(exc)
            first_error = first_error or last_error
            current = build_repair_prompt(prompt, last_error, last_raw)
            continue
        return build_record(fields, uid=uid, model_id=ann.model_id, attempt=attempt,
                            sampled_seed=sampled_seed, render_rec=render_rec,
                            proportions=proportions, prefix_reuse=prefix_reuse,
                            raw_response=last_raw, first_error=first_error), None
    return None, {"uid": uid, "failure_class": "MODEL_RECOVERABLE", "terminated_by": "repair_budget",
                  "attempts": MAX_ATTEMPTS, "exception_type": "AnnotationError",
                  "exception_msg": last_error[:400], "raw_response": last_raw[:1000]}
