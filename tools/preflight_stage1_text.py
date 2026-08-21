#!/usr/bin/env python
"""[B-4, D0-008 §11.2] The gate that runs before n06 spends any GPU time.

# CHECKS-NODE: n06_encode_text_image

Three questions, all answerable in seconds over the full corpus, none of which
requires loading an encoder:

  1. does every serialized string match the template ratified by D0-008 §11.3?
  2. does any record still render a stored non-zero dimension as ``0``?
  3. does any record still exceed CLIP's 77-token context?

and one more that is the reason D10 exists at all:

  4. under the ratified protocol, how many of the embeddings already on disk are
     still cache-valid?

Token counting uses the UNTRUNCATED BPE path -- ``SimpleTokenizer.encode`` plus
2 for SOT/EOT -- not ``open_clip.tokenize``, which pads to exactly 77 and would
report the corpus's 89-token record as 77. That is Codex finding C-3, confirmed.

Read-only. This script opens annotations and sidecars and writes nothing.

    python tools/preflight_stage1_text.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402

paths.setup_env()

from metafind.data.annotate import (  # noqa: E402
    annotation_contract_id,
    non_english_characters,
)
from metafind.data.encode_text_image import (  # noqa: E402
    TEXT_CONTEXT_LENGTH,
    expected_text_for,
    is_complete,
    true_token_count,
)
from metafind.models.resolve_stage1 import text_serialization_id  # noqa: E402

# The fields a record must carry for this serializer to handle it. Keyed on the
# SHAPE, not on a version number: `prompt_version` moves for reasons that have
# nothing to do with serializability (v4 changed the prompt's language clause and
# not one field name), and a gate that keys on it starts rejecting a corpus it
# can serialize perfectly well.
SERIALIZABLE_FIELDS = ("width", "length", "height",
                       "description", "category", "materials")

# --- the ratified contract, transcribed from D0-008, imported from nothing ---
#
# Everything below is a LOCAL COPY on purpose. An earlier version of this file
# imported the caps, `_cap()` and `placement_phrase()` from the module it was
# meant to be checking, so an unauthorised edit to a placement phrase moved both
# sides of the comparison together and the gate reported zero mismatches. An
# oracle that shares the implementation's semantics is not an oracle.
#
# The cost of the copy is that a LEGITIMATE change here must be made twice. That
# is the intended cost: D0-008 §12.3's scope guard says no serialization change
# beyond E-1/E-2/S-1/S-2 is authorised, so this gate should fail loudly when one
# happens, and a second edit is exactly the moment someone has to justify it.
RATIFIED_MAX_DESCRIPTION_CHARS = 160
RATIFIED_MAX_CATEGORY_CHARS = 40
RATIFIED_MAX_MATERIALS = 3
RATIFIED_PLACEMENT_PHRASES = {
    ("onCeiling",): "typically mounted on a ceiling",
    ("onWall",): "typically mounted on a wall",
    ("onFloor",): "typically placed on the floor",
    ("onObject",): "typically placed on top of other objects",
    ("onFloor", "onObject"): "typically placed on the floor or on other objects",
}
RATIFIED_NO_PLACEMENT_PHRASE = "with no typical placement"
RATIFIED_FLAG_ORDER = ("onCeiling", "onWall", "onFloor", "onObject")

# n05's own admission bounds (annotate.py MIN_DIM_CM / MAX_DIM_CM). n06 reads the
# annotation JSON directly and never revalidates it, so a hand-edited or
# corrupted record could otherwise reach the encoder as "roughly nan by inf".
RATIFIED_MIN_DIM_CM, RATIFIED_MAX_DIM_CM = 0.1, 10_000.0

# CLIP's context length, transcribed rather than imported for the same reason as
# everything else here: if production moved TEXT_CONTEXT_LENGTH to 88 the gate
# would move with it and stop seeing the 88-token record, while CLIP would go on
# truncating at 77. 77 is CLIP's number, not ours, so the gate also fails if the
# production constant no longer agrees with it.
RATIFIED_TEXT_CONTEXT_LENGTH = 77


def ratified_cap(text: str, limit: int) -> str:
    """Trim at a word boundary, keeping a trailing period. Local copy."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "."


def ratified_placement(ann: dict) -> str:
    """The placement clause, rebuilt from the transcribed vocabulary."""
    on = tuple(f for f in RATIFIED_FLAG_ORDER if ann.get(f))
    if not on:
        return RATIFIED_NO_PLACEMENT_PHRASE
    for key in (on, tuple(sorted(on))):
        if key in RATIFIED_PLACEMENT_PHRASES:
            return RATIFIED_PLACEMENT_PHRASES[key]
    parts = [RATIFIED_PLACEMENT_PHRASES[(f,)] for f in on]
    return parts[0] + " or ".join([""] + [p.split(" ", 2)[-1] for p in parts[1:]])


def ratified_string(ann: dict) -> str:
    """D0-008 §11.3's template, written out and importing nothing from n05b.

    Field order, the E-1/S-1 dimension formatter, the E-2 absent article, the
    S-2 capitalisation, the unit, the caps and the placement vocabulary are all
    transcribed above, so this function and ``serialize_annotation()`` share no
    code at all. They can only agree by both being right.
    """
    description = ratified_cap(ann["description"].strip(),
                               RATIFIED_MAX_DESCRIPTION_CHARS)
    if description and not description.endswith("."):
        description += "."
    category = ratified_cap(ann["category"], RATIFIED_MAX_CATEGORY_CHARS).rstrip(".")
    category = category[:1].upper() + category[1:]              # S-2
    materials = ", ".join(ann["materials"][:RATIFIED_MAX_MATERIALS])
    dims = [f"{float(ann[f]):.1f}".removesuffix(".0")           # E-1 + S-1
            for f in ("width", "length", "height")]
    return (f"{description} {category} made of {materials}, "
            f"roughly {dims[0]} by {dims[1]} by {dims[2]} centimetres, "
            f"{ratified_placement(ann)}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="first N annotations, for a smoke run")
    args = ap.parse_args()

    index = paths.LOGS / "renders_index.jsonl"
    if not index.exists():
        print(f"{index} not found -- run n04_render_views first")
        return 2
    rendered = {json.loads(l)["uid"] for l in index.read_text().splitlines() if l.strip()}

    annotations = sorted(paths.ANNOTATIONS.glob("*.json"))
    if args.limit:
        annotations = annotations[: args.limit]

    work_list = [p for p in annotations if p.stem in rendered]

    identity = text_serialization_id()
    contract = annotation_contract_id()
    v3 = v_other = 0
    residual_v1: list[str] = []
    serializable_non_v3: list[str] = []
    unserializable_v3: list[str] = []
    language_violations: list[tuple[str, str]] = []
    stale_contract = 0
    prompt_versions: dict = {}
    template_mismatch: list[str] = []
    zero_dim: list[str] = []
    bad_dim: list[str] = []
    overlong: list[tuple[str, int]] = []
    cache_valid = 0
    independent_valid = 0
    foreign_but_valid: list[str] = []
    max_tokens, max_uid = 0, ""

    for path in work_list:
        ann = json.loads(path.read_text())
        uid = path.stem
        # Version FIRST, then serializability. The other order let a v3 record
        # that could not be serialized fall into the "not v3" bucket and quietly
        # shrink the validated population: a single corrupt record could take the
        # total to 0 and the gate would still print PASSED.
        prompt_versions[ann.get("prompt_version")] = (
            prompt_versions.get(ann.get("prompt_version"), 0) + 1)
        if ann.get("annotation_contract") != contract:
            stale_contract += 1
        is_v3 = all(f in ann for f in SERIALIZABLE_FIELDS)
        text = expected_text_for(path)
        if not is_v3:
            v_other += 1
            (residual_v1 if not text else serializable_non_v3).append(uid)
            continue
        if not text:
            unserializable_v3.append(uid)
            continue
        v3 += 1

        if text != ratified_string(ann):
            template_mismatch.append(uid)

        # n06 reads annotation JSON straight off disk and never revalidates it,
        # so a corrupted record would serialize as "roughly nan by inf" and be
        # embedded without a word. n05 refuses these values; this gate is the
        # place that notices if one got past n05 or was edited afterwards.
        for field in ("width", "length", "height"):
            v = float(ann[field])
            if not math.isfinite(v) or not (
                    RATIFIED_MIN_DIM_CM <= v <= RATIFIED_MAX_DIM_CM):
                bad_dim.append(uid)
                break

        rendered_dims = text.split(", roughly ", 1)[1].split(" centimetres,", 1)[0]
        for shown, field in zip(rendered_dims.split(" by "),
                                ("width", "length", "height")):
            try:
                shown_value = float(shown)
            except ValueError:          # "nan"/"inf" already counted above
                zero_dim.append(uid)
                break
            if shown_value == 0.0 and float(ann[field]) != 0.0:
                zero_dim.append(uid)
                break

        # [P-2] One definition of "English", shared with the validator that
        # admits annotations. Unlike the template rules above, this is not a
        # D0-008 constant to transcribe -- a second copy would be a second
        # answer to the same question.
        for field in ("category", "description", *ann["materials"]):
            bad = non_english_characters(str(ann.get(field, field)))
            if bad:
                language_violations.append((uid, "".join(bad)))
                break

        n = true_token_count(text)
        if n > max_tokens:
            max_tokens, max_uid = n, uid
        if n > RATIFIED_TEXT_CONTEXT_LENGTH:
            overlong.append((uid, n))

        # Two independent counts of the same thing. `is_complete()` is the gate
        # n06 will actually apply, so it is the one that decides the run; the
        # second reconstructs the answer from the sidecar without calling it, so
        # the reported proof is not simply the implementation restating itself.
        if is_complete(uid, text):
            cache_valid += 1
            rec = json.loads((paths.EMBEDDINGS / f"{uid}.json").read_text())
            if rec.get("text_serialization") != identity:
                foreign_but_valid.append(uid)
        sc = paths.EMBEDDINGS / f"{uid}.json"
        if sc.exists():
            try:
                rec = json.loads(sc.read_text())
            except (OSError, json.JSONDecodeError):
                rec = {}
            npz = rec.get("embedding_uri")
            if (rec.get("text") == ratified_string(ann)
                    and isinstance(npz, str) and npz and Path(npz).is_file()):
                independent_valid += 1

    print(f"serializer identity      {identity}")
    print(f"annotation files         {len(annotations):,}")
    print(f"n06 work list            {len(work_list):,}   "
          f"(annotations INTERSECT renders_index)")
    print(f"  valid v3               {v3:,}")
    print(f"  prompt_version 1 residuals{v_other:>3,}   {sorted(residual_v1)}")
    print()
    print(f"prompt_version spread     "
          f"{ {k: f'{v:,}' for k, v in sorted(prompt_versions.items(), key=lambda x: (x[0] is None, x[0]))} }")
    print(f"stale annotation contract{stale_contract:>7,}   "
          f"(current: {contract})")
    print()
    print(f"language violations      {len(language_violations)}")
    print(f"template mismatches      {len(template_mismatch)}")
    print(f"zero-dimension renders   {len(zero_dim)}")
    print(f"over {RATIFIED_TEXT_CONTEXT_LENGTH} true tokens      {len(overlong)}   "
          f"(max {max_tokens} on {max_uid})")
    print()
    on_disk = len(list(paths.EMBEDDINGS.glob("*.json")))
    print(f"{'embedding sidecars on disk:':<38}{on_disk:>8,}")
    print(f"{'  of those still cache-valid:':<38}{cache_valid:>8,}")
    print(f"{'  independent recount:':<38}{independent_valid:>8,}")
    print()
    print(f"{'total records:':<38}{v3:>8,}")
    print(f"{'cache-valid under ratified protocol:':<38}{cache_valid:>8,}")
    print(f"{'requires encoding:':<38}{v3 - cache_valid:>8,}")

    failures = []
    if template_mismatch:
        failures.append(f"{len(template_mismatch)} strings do not match the "
                        f"ratified template, e.g. {template_mismatch[:3]}")
    if zero_dim:
        failures.append(f"{len(zero_dim)} records render a stored non-zero "
                        f"dimension as 0, e.g. {zero_dim[:3]}")
    if overlong:
        failures.append(f"{len(overlong)} records exceed "
                        f"{RATIFIED_TEXT_CONTEXT_LENGTH} true tokens: {overlong[:3]}")
    if TEXT_CONTEXT_LENGTH != RATIFIED_TEXT_CONTEXT_LENGTH:
        failures.append(f"n06's TEXT_CONTEXT_LENGTH is {TEXT_CONTEXT_LENGTH}, but "
                        f"CLIP truncates at {RATIFIED_TEXT_CONTEXT_LENGTH}")
    if unserializable_v3:
        failures.append(f"{len(unserializable_v3)} prompt_version:3 records cannot "
                        f"be serialized at all, e.g. {unserializable_v3[:3]}. n06 "
                        f"would quarantine them and emit no embedding")
    if serializable_non_v3:
        failures.append(f"{len(serializable_non_v3)} non-v3 records now serialize; "
                        f"the population this gate validates has changed shape: "
                        f"{serializable_non_v3[:3]}")
    if language_violations:
        failures.append(f"{len(language_violations)} records contain non-English "
                        f"script in category/description/materials, e.g. "
                        f"{language_violations[:3]}")
    if v3 == 0:
        failures.append("no prompt_version:3 records were validated at all -- an "
                        "empty gate is not a passing gate")
    if bad_dim:
        failures.append(f"{len(bad_dim)} records carry a non-finite or "
                        f"out-of-range dimension, e.g. {bad_dim[:3]}")
    if foreign_but_valid:
        failures.append(f"{len(foreign_but_valid)} sidecars are judged COMPLETE "
                        f"while recording a different serialization identity, "
                        f"e.g. {foreign_but_valid[:3]}. That is the two-distribution "
                        f"gallery this contract exists to prevent")
    if cache_valid != independent_valid:
        failures.append(f"is_complete() says {cache_valid:,} cache-valid, an "
                        f"independent recount of the same sidecars says "
                        f"{independent_valid:,}")
    if failures:
        print("\nPRE-FLIGHT FAILED -- do not start n06:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPRE-FLIGHT PASSED for the text contract.")
    if stale_contract:
        # A WARNING, not a failure. Whether n06 may encode a corpus annotated
        # under an older contract is a research decision about comparability, not
        # a defect this gate can adjudicate.
        print(f"\nWARNING: {stale_contract:,} of {len(work_list):,} records were "
              f"annotated under an OLDER annotation contract than the one this "
              f"code declares.\n"
              f"         current: {contract}\n"
              f"         The serialized text is still valid and every check above "
              f"passed. Whether a\n"
              f"         full re-annotation must precede n06 is a decision for "
              f"Master, not for this gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
