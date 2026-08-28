"""Run Qwen2.5-VL over each asset's 11 views and produce a validated annotation.

# IMPLEMENTS-NODE: n05_annotate

Writes ``objaverse_annotations`` (one sidecar per asset, plus the derived
index), and ``quarantine`` / ``run_progress`` / ``cost_ledger`` via runlog.

The generating half of n05. The schema, prompt and repair-prompt construction
live in ``annotate.py``, which is deterministic and testable without a GPU;
this module is the part that needs the model, and it implements subgraph SG1:

    sg1_generate  ->  sg1_validate  ->  admit
                          |
                          +-> C1 repair loop (2 attempts) -> quarantine

Why the loop is bounded at two and the third outcome is quarantine
------------------------------------------------------------------

C1's four-piece set, from the graph spec:

    progress measure    item_attempt, incremented per attempt
    semantic exit       the annotation passes schema validation
    hard bound          MAX_ATTEMPTS = 2
    exhaustion outcome  quarantine with terminated_by=repair_budget

An exhausted item must never be admitted. That is L1-ANNOT-EXHAUST, and its
negative injection is "mark the exhausted item as admitted" -- a bound treated
as success is a bound that does nothing.

Deviation D-2
-------------

Qwen2.5-VL replaces GPT-4o, recorded per asset as ``annotator_model``. There is
no fallback to ULIP-2's shipped captions; see annotate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from metafind import paths, runlog
from metafind.data.view_io import image_identity

# BEFORE any transformers/torch import anywhere in this process. HF_HOME is read
# at IMPORT time, not at from_pretrained time, so setting it inside Annotator
# after `import transformers` had no effect: the run started re-downloading 16 GB
# of weights into ~/.cache on the 100 GB root partition while a complete copy sat
# on the data volume. It looked like a slow load for six minutes.
paths.setup_env()

# n03's seed function, reused rather than reinvented: the two nodes must not
# disagree about what "this asset's seed" means.
from metafind.data.pointclouds import uid_seed
from metafind.data.describe_rank import N_CANDIDATES, RANKER_MODEL, RANKER_VERSION
from metafind.data.describe_rank import rank as rank_descriptions
from metafind.data.annotate import (
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    AnnotationError,
    annotation_contract_id,
    blind_agrees,
    build_blind_prompt,
    build_description_prompt,
    build_prompt,
    build_unanchored_prompt,
    non_english_characters,
    parse_blind_guess,
    build_repair_prompt,
    parse_annotation,
    validate_annotation,
    LVIS_SYNSETS,
)

NODE = "n05_annotate"
MODEL_ID = "/home/kyzen/metafind_out/gemma-4-12B-it"  # D-2: stands in for GPT-4o.
# [DEVIATION D-2, re-pointed 2026-08-24 on the USER's decision -- his words,
# verbatim: "D-2 改成 gemma"] The paper annotates with GPT-4o
# (`2methdology.tex:28`, `neurips_2025.tex:100`). The stand-in was
# Qwen2.5-VL-7B-Instruct, then Qwen3.8-27B, and is now gemma-4-12B-it.
# None of them is GPT-4o; the deviation is RE-POINTED, not discharged, and
# must never be written up as paper-faithful.
#
# [C1, 2026-08-24] This default was `/mnt/data1/kyzen/models/Qwen3.8-27B`:
# 56 GB of bf16 against a 32,607 MiB card, on the SMR drive. It was safe only
# because `tools/run_ulip_full.sh` overrode it with `--model`, and a default
# that is safe only when one caller overrides it is not a default. Any direct
# invocation -- a resume, a debug, a validation batch, the timing arm -- loaded
# 56 GB onto a 32 GB card and OOMed slowly, off SMR. The record and the default
# now name the same model, so `--model` no longer has to be remembered for the
# run to be BOTH runnable and correctly described.
#
# The bake-off arms recorded `/mnt/data1/kyzen/models/gemma-4-12B-it`. MEASURED
# 2026-08-24: that copy and this one are the same weights -- identical byte
# size (23,951,779,497), identical `config.json` sha256, and identical sha256
# over 400 MB of head+tail of `model.safetensors`. Same model, different path
# string; `annotator_model` will differ textually between the arms and the run.
MAX_NEW_TOKENS = 512

# [PROMPT_VERSION 7] Sampling for the description candidates only. ULIP-2 does
# not publish BLIP-2's decoding settings, so these are an IMPLEMENTATION CHOICE
# and must never be reported as upstream values. They are chosen to give the
# five candidates room to differ without drifting into nonsense; the spread of
# their CLIP scores is recorded, which is what says afterwards whether the
# setting was reasonable.
SAMPLING_TEMPERATURE = 0.9
SAMPLING_TOP_P = 0.95

# Stamped on every record: which model chose the description, and which
# version of the ranking rule. Two corpora ranked by different CLIPs are
# not comparable and nothing else on the record would say so.
DESCRIPTION_RANKER = {"model": RANKER_MODEL, "version": RANKER_VERSION}


# Set only by `use_arm`. `None` means "the corpus", and that is resolved at
# CALL time rather than captured here: binding `paths.ANNOTATIONS` at import
# would freeze whatever it pointed at then, which silently defeats every test
# that redirects it and would make this module disagree with `paths` about
# where the corpus is.
_ARM_ROOT: Path | None = None


def out_root() -> Path:
    """Where n05 writes: the corpus, unless `--arm` redirected it."""
    return paths.ANNOTATIONS if _ARM_ROOT is None else _ARM_ROOT


def use_arm(arm: str) -> Path:
    """Point every write at ``data/outputs/bakeoff/<arm>/`` instead of the corpus.

    `SPEC_M1` §4: *"`data/outputs/annotations/` must hold 0 files at every point
    in M1. It belongs to the full run alone."* The bake-off writes 100 records
    per arm, and without this they land in the directory the full run owns,
    where nothing afterwards can tell an experiment from the corpus.

    The name is restricted to a plain directory, and the result is compared
    against `paths.ANNOTATIONS` **after resolving symlinks** -- `data/outputs`
    is itself a link on this machine, so comparing the unresolved paths would
    let `--arm ../../annotations` through.
    """
    global _ARM_ROOT
    if not arm or arm != Path(arm).name or arm in {".", ".."}:
        raise ValueError(f"--arm must be a plain directory name, got {arm!r}")
    root = paths.OUTPUTS / "bakeoff" / arm / "annotations"
    if root.resolve() == paths.ANNOTATIONS.resolve():
        raise ValueError(
            f"--arm {arm!r} resolves onto the corpus annotations directory; "
            "the bake-off may not write there"
        )
    root.mkdir(parents=True, exist_ok=True)
    _ARM_ROOT = root
    return root


def sidecar_path(uid: str) -> Path:
    return out_root() / f"{uid}.json"


def _record(uid: str) -> tuple[dict, str] | None:
    """``(record, sha256-of-the-bytes)``, or ``None`` when no file is there at all.

    "No record" and "a record I could not read" lead to OPPOSITE decisions below,
    so they are kept apart: absence is the only thing a bare run treats as work.
    An unreadable file yields ``({}, digest)`` -- present, and therefore never
    silent work.

    The `isinstance` guard matters: `json.loads("null")` returns `None`, and
    returning that would have made a corrupt-but-parseable sidecar look ABSENT
    and put an existing file back in the queue. Anything that is not a JSON
    object is an unreadable record, not a missing one.
    """
    sc = sidecar_path(uid)
    if not sc.exists():
        return None
    try:
        raw = sc.read_bytes()
    except OSError:
        return {}, ""
    digest = hashlib.sha256(raw).hexdigest()
    try:
        rec = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}, digest
    return (rec if isinstance(rec, dict) else {}), digest


def _under_current_contract(rec: dict, image_id: str | None = None) -> bool:
    # Keyed on the ANNOTATION CONTRACT, not on prompt_version alone. A v1 sidecar
    # has every field name it shares with v2 but a different schema, and treating
    # it as done would silently mix two annotation generations in one corpus --
    # and prompt_version cannot express the other two axes: the same
    # `prompt_version: 3` could have been admitted by a validator with or without
    # the language rule. The contract id folds prompt, validator and schema
    # semantics into one comparison.
    # [ADDED 2026-08-24] The renders are half of what produced this record and
    # were not compared at all. A record that cannot say which images it saw is
    # not current -- absence fails CLOSED, into UNACCOUNTED, which stops the run
    # rather than either re-annotating or skipping it silently. `image_id` is
    # supplied by `main`, which has n04's index; the presence rule holds even
    # when a caller cannot supply it.
    if not rec.get("image_identity"):
        return False
    if image_id is not None and rec.get("image_identity") != image_id:
        return False
    return (all(k in rec for k in REQUIRED_FIELDS)
            and rec.get("annotation_contract") == annotation_contract_id()
            and "annotator_model" in rec)


def is_complete(uid: str) -> bool:
    """Completion is a parseable sidecar carrying the fields the channel declares.

    Same contract as n03 and n04, and for the same reason: the record IS the
    artifact here, so a half-written one is the only failure worth guarding.
    """
    found = _record(uid)
    return found is not None and _under_current_contract(found[0])


# --- AC-1: an existing record is never automatic work ----------------------
#
# [D2a, AC-1] `is_complete()` alone decides "done under the CURRENT contract",
# which is the right question for resuming an interrupted run and the wrong one
# for a corpus that already exists under an older contract. After D10 introduced
# contract stamping, no stored record carried a contract id, so a bare run
# queued all 45,955: the 45,952 the user accepted as legacy-v3, and the 3
# legacy-v1 residuals D0-003 has not decided. Re-running would have rewritten
# both -- and rewriting the residuals would have settled D0-003 by mutation,
# ahead of any decision.
#
# The gate is NOT "skip records that look done". A missing field is what created
# this hazard, so a missing field may not be what clears it: a record is passed
# over only when a DECLARED registry names it, or when it carries the current
# contract id. Any other existing record is UNACCOUNTED, and an unaccounted
# record STOPS the run rather than joining the queue -- so deleting, truncating
# or shortening the registry fails closed, never open.
#
# Nothing is removed: `--force` still re-annotates anything, and
# `--uids-file <list> --force` is the named-migration form.

PROVENANCE_REGISTRY = paths.OUTPUTS / "annotation_provenance.json"

CURRENT_CONTRACT = "annotated_under_current_contract"
ACCEPTED_LEGACY_V3 = "accepted_legacy_v3"
LEGACY_V1_RESIDUAL = "legacy_v1_residual_unresolved"
UNACCOUNTED = "unaccounted"

# Only these two may be DECLARED. `CURRENT_CONTRACT` is self-evidencing -- it is
# read off the record's own `annotation_contract` field, which is where AC-1.c's
# "explicit in the record" half is satisfied -- and `UNACCOUNTED` is the ABSENCE
# of a declaration, so neither is something a registry may assert.
#
# Each declarable state is bound to the schema generation its name refers to.
# Without this, a registry could declare a legacy-v1 residual as
# `accepted_legacy_v3` and the loader would take it, which is exactly the
# conflation AC-1.e forbids.
STATE_PROMPT_VERSION = {
    ACCEPTED_LEGACY_V3: 3,
    LEGACY_V1_RESIDUAL: 1,
}
DECLARABLE_STATES = tuple(STATE_PROMPT_VERSION)


class ProvenanceRegistryError(ValueError):
    """The registry cannot be trusted. Always fatal -- never a reason to proceed."""


def load_provenance_registry(path: Path = PROVENANCE_REGISTRY
                             ) -> dict[str, tuple[str, int, str]]:
    """uid -> (declared state, prompt_version, sha256 of the declared record).

    A missing file is an EMPTY registry, not a permissive one: with nothing
    declared, every pre-existing record classifies UNACCOUNTED and the run
    refuses. Every other problem -- unparseable, wrong shape, an undeclarable
    state, a state bound to the wrong schema, a uid declared twice -- raises.
    Both directions are closed; there is no reading of a damaged registry under
    which work gets queued.
    """
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProvenanceRegistryError(f"{path} could not be read: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("populations"), list):
        raise ProvenanceRegistryError(
            f"{path} has no `populations` list; it is not a provenance registry")

    out: dict[str, tuple[str, int, str]] = {}
    for pop in doc["populations"]:
        if not isinstance(pop, dict):
            raise ProvenanceRegistryError(
                f"{path} has a population entry that is not an object: {pop!r}")
        state = pop.get("state")
        if state not in STATE_PROMPT_VERSION:
            raise ProvenanceRegistryError(
                f"{path} declares provenance state {state!r}; declarable states are "
                f"{DECLARABLE_STATES}")
        pv = pop.get("prompt_version")
        # `type(pv) is int` rather than `isinstance`: `True == 1` and `1.0 == 1`
        # both hold in Python, and a schema generation named by a bool is not a
        # schema generation.
        if type(pv) is not int or pv != STATE_PROMPT_VERSION[state]:
            raise ProvenanceRegistryError(
                f"{path} declares {state!r} at prompt_version {pv!r}, but that state "
                f"means prompt_version {STATE_PROMPT_VERSION[state]}. A declaration "
                "may not move a record between schema generations.")
        records = pop.get("records")
        if not isinstance(records, dict):
            raise ProvenanceRegistryError(
                f"{path} population {state!r} has no `records` mapping of "
                "uid -> sha256")
        for uid, digest in records.items():
            if uid in out:
                raise ProvenanceRegistryError(
                    f"{path} declares {uid!r} twice ({out[uid][0]!r} and {state!r}). "
                    "A record belongs to exactly one population; a second "
                    "declaration would silently overwrite the first.")
            if not isinstance(uid, str) or not isinstance(digest, str) or not digest:
                raise ProvenanceRegistryError(
                    f"{path} population {state!r} has a malformed entry for {uid!r}")
            out[uid] = (state, pv, digest)
    return out


def provenance_state(uid: str, registry: dict[str, tuple[str, int, str]],
                     image_id: str | None = None) -> str | None:
    """Which declared population accounts for this uid, or ``None`` if no record.

    ``None`` -- and only ``None`` -- is what a bare run treats as work.

    A declaration is about ONE specific record: it says that record was seen,
    classified and (for legacy-v3) re-validated. So it is honoured only while the
    bytes still hash to what was declared. If the file changed, the declaration
    no longer describes what is on disk and the uid becomes UNACCOUNTED -- which
    stops the run rather than either re-annotating it or quietly skipping it.
    """
    found = _record(uid)
    if found is None:
        return None
    rec, digest = found
    if _under_current_contract(rec, image_id):
        # Self-evidencing, and it must win: a residual that a NAMED MIGRATION
        # legitimately re-annotated will carry the current contract while an
        # older declaration still names it. The record is the newer fact.
        return CURRENT_CONTRACT
    declared = registry.get(uid)
    if declared is None:
        return UNACCOUNTED
    state, pv, declared_digest = declared
    if declared_digest != digest or rec.get("prompt_version") != pv:
        return UNACCOUNTED
    # [ADDED 2026-08-24, Codex CHANGES REQUIRED] A declaration says a record was
    # seen and classified. It cannot say that about images rendered afterwards.
    # Without this, a byte-matching legacy record survives a re-render and
    # `build_work_list` puts its state in NEITHER list -- `todo` is
    # `state is None` and `blocked` is `state is UNACCOUNTED` -- so it is not
    # annotated, not blocked, not counted, and `blocked == 0` reads as "nothing
    # is stuck". A silent skip is the one outcome AC-1 exists to prevent, and
    # the ENGINEER reported to MASTER that a missing identity routes to
    # UNACCOUNTED unconditionally. It did not.
    #
    # [CORRECTED 2026-08-27] The sentence that stood here was "It does now."
    # That claim was still stronger than the mechanism. The comparison is
    # `image_id is not None and ...`, so a caller that passes no id disables it,
    # and until today NO test passed one: `grep -rn 'image_ids' tests/` and
    # `grep -rn 'image_id=' tests/` both returned nothing, across 14 call sites.
    # A behavioural claim written in the file with nothing in the suite able to
    # falsify it is a green light wired to nothing.
    #
    # The default stays. `main()` supplies the ids from n04's index, and the
    # callers that legitimately have none -- an audit over a registry alone --
    # must not be forced to invent one. What changed is that the comparison half
    # now has tests that fail if it stops comparing, and this comment states the
    # condition instead of asserting the outcome.
    if image_id is not None and rec.get("image_identity") != image_id:
        return UNACCOUNTED
    return state


def classify_all(candidates, registry, image_ids=None) -> dict[str, str | None]:
    """The predicate a bare run builds its work list from. No model, no GPU."""
    image_ids = image_ids or {}
    return {uid: provenance_state(uid, registry, image_ids.get(uid))
            for uid in candidates}


def build_work_list(candidates, force: bool, registry=None,
                    image_ids=None) -> tuple[list[str], list[str], dict[str, str | None]]:
    """The WHOLE work-list decision, both branches. Returns (todo, blocked, states).

    `main()` calls exactly this and does nothing else to choose what gets
    annotated, so a proof that exercises this function is a proof about the real
    run -- not a re-statement of it. Nothing here loads a model or touches a GPU.
    """
    if force:
        # [AC-1.b] The explicit path, unfiltered. `--force` re-annotates the whole
        # corpus; `--uids-file <list> --force` is the same capability aimed at a
        # named population. The gate removes the ACCIDENT, not this.
        return list(candidates), [], {}
    states = classify_all(candidates,
                          load_provenance_registry() if registry is None else registry,
                          image_ids)
    return ([uid for uid, st in states.items() if st is None],
            [uid for uid, st in states.items() if st == UNACCOUNTED],
            states)


def _fit_description(parsed: dict, lvis_category, proportions, ranked: list[dict],
                     model_id: str):
    """[PROMPT_VERSION 9] The highest-ranked description whose SERIALIZED string
    fits CLIP's 77-token context, and the record of which one that was.

    Returns ``(Annotation, fit_record)``, or ``(None, fit_record)`` when no
    candidate fits.

    Why here and not in the prompt. `MAX_DESCRIPTION_WORDS` is an INSTRUCTION --
    the model overruns a stated word count routinely, and a word count is not a
    token count in any case. This is the bound that is actually enforced, and it
    is checked against the real serialized sentence rather than a proxy, because
    the remainder differs per asset: over the corpus the non-description part
    costs between 31 and 52 tokens, so a single fixed description budget is
    either unsafe for the worst asset or wasteful for the median.

    Why it can be done by SELECTION rather than by cutting. `describe_rank`
    already draws five candidates and ranks them, so taking the best one that
    fits costs no model call and produces no mid-sentence tail -- which is the
    whole point of v9, since the 160-character cap it replaces produced
    "It features a." on 95.8% of the v8 corpus.

    [ULIP2 Block Reviewer 2026-08-28] This used to say the reason was that
    "every one of them is a complete sentence". **That sentence is false** and
    the conclusion does not need it. Generation is capped at MAX_NEW_TOKENS, so
    a repetition loop returns a candidate cut mid-word; rare under "ONE sentence
    of at most 15 words", not impossible. The real reason is stronger: a
    runaway candidate is LONG, so it fails the token bound and is skipped. **A
    malformed candidate can only be discarded, never selected.** Safety comes
    from the bound, not from the candidates being well-formed -- which is the
    property worth having, because it does not depend on the generator
    behaving.

    The rank actually used is RECORDED. If this routinely lands on rank 3, the
    word budget is too loose and the number is how anyone would find that out;
    a silent fallback would look exactly like a run where every winner fitted.
    """
    from copy import deepcopy

    from metafind.data.encode_text_image import TEXT_CONTEXT_LENGTH, true_token_count
    from metafind.models.resolve_stage1 import serialize_annotation

    tried = []
    for candidate in ranked:
        text = candidate["text"]
        # [ESSGNN Reviewer 2026-08-28] This call is OUTSIDE the repair loop's
        # `try`, so an escaping AnnotationError would leave `annotate_one`
        # entirely -- skipping both the repair loop and the quarantine path
        # below, and crashing the run instead of losing one asset.
        #
        # Today it cannot fire, and I checked rather than assumed. Three rules
        # in `validate_annotation` read `description`: the two emptiness checks
        # and `_refuse_non_english`. The draw loop's pre-filter is
        # `if text and not non_english_characters(text)`, and
        # `_refuse_non_english` is `non_english_characters` plus a raise -- the
        # SAME function, so every candidate here already satisfies all three.
        # Every other rule in that function reads a field the winner already
        # passed on this same `parsed`.
        #
        # It is caught anyway, because that argument is about the state of
        # ANOTHER module: it holds only while the pre-filter and the validator
        # keep applying the identical rule, and nothing makes them. This file
        # already states the principle at `serialize_annotation` -- "a guard
        # that depends on a check in another module is a guard that disappears".
        # A candidate that will not validate is the same fact to the caller as
        # a candidate that will not fit: this one cannot be used.
        try:
            annotation = validate_annotation(deepcopy(parsed),
                                             lvis_category=lvis_category,
                                             proportions=proportions,
                                             description=text)
        except AnnotationError as exc:
            tried.append({"rank": candidate["rank"], "rejected": str(exc)[:200]})
            continue
        n = true_token_count(serialize_annotation(annotation.as_record(model_id)))
        tried.append({"rank": candidate["rank"], "tokens": n})
        if n <= TEXT_CONTEXT_LENGTH:
            return annotation, {"rank_used": candidate["rank"],
                                "tokens": n,
                                "candidates_tried": len(tried),
                                "context_length": TEXT_CONTEXT_LENGTH}
    # Two reasons reach here and the record must say WHICH: "none fitted" and
    # "none validated" call for different responses, and one message covering
    # both would state the wrong cause for whichever it was not.
    return None, {"rank_used": None,
                  "tried": tried,
                  "over_context": [t["tokens"] for t in tried if "tokens" in t],
                  "rejected_by_validator": [t["rank"] for t in tried
                                            if "rejected" in t],
                  "candidates_tried": len(tried),
                  "context_length": TEXT_CONTEXT_LENGTH}


# --- v5 anchors -----------------------------------------------------------
#
# Two inputs that PROMPT_VERSION 5 requires and v4 never read: the dataset's own
# category, and the mesh's own proportions. Both were on disk the whole time.

def load_lvis_categories() -> dict[str, str]:
    """uid -> Objaverse-LVIS category. `value_to_key_mapping` was downloaded by
    `download.py:70` and, until this task, read by nothing in the pipeline."""
    src = paths.DATASETS / "objaverse-lvis" / "objaverse_lvis_metadata.json"
    with src.open(encoding="utf-8") as fh:
        mapping = json.load(fh)["value_to_key_mapping"]
    unknown = {c for c in mapping.values() if c not in LVIS_SYNSETS}
    if unknown:
        raise SystemExit(
            f"{len(unknown)} LVIS categories have no synset in the lookup table, "
            f"e.g. {sorted(unknown)[:5]}. Rebuild the table rather than annotating "
            "past it."
        )
    return mapping


def load_proportions() -> dict[str, tuple[float, float, float]]:
    """uid -> (y, x, z) normalised so the largest is 1.0.

    [OBSERVED DATA] The axis convention is Y-up. Reproduced independently for
    this task over 1,365 assets whose LVIS category is unambiguously tall and
    962 unambiguously flat: tall meshes average [x .543, y .946, z .474] and are
    y-longest 83.6% of the time; flat meshes average [x .882, y .372, z .716]
    and are y-longest 11.5% of the time. `height` is the y axis.
    """
    out: dict[str, tuple[float, float, float]] = {}
    src = paths.LOGS / "pointclouds_index.jsonl"
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            ext = rec.get("raw_bbox_extents")
            if not ext or len(ext) != 3:
                continue
            longest = max(ext)
            if longest <= 0:
                continue
            x, y, z = (v / longest for v in ext)
            out[rec["uid"]] = (y, x, z)
    return out


def _is_cuda_oom(exc: BaseException) -> bool:
    """[C4, 2026-08-24] Is this exception the card running out of memory?

    MEASURED during the 2026-08-24 timing arm: the batched n=5 draw peaks at
    31,932 MiB of 32,607 -- **675 MiB of headroom**, worse than the 30.02 GB
    this file's own docstring records. At that margin CUDA OOM is not one of the
    things the fallback's `except Exception` might catch; over 46,024 assets and
    5.4 days it is the LIKELIEST thing it will ever catch.

    That matters because the fallback's response is to issue five more prefills.
    On a genuinely per-asset failure (a very detailed mesh, an unusually long
    prompt) that is right and it is why the fallback exists. On an OOM it is the
    least survivable possible response: five sequential prefills on a card that
    has just fragmented, each of which can OOM again. Freeing the cache first is
    what makes the retry meaningful instead of five more failures.

    Both branches, because neither alone is reliable: torch raises
    `torch.cuda.OutOfMemoryError` for the allocator's own failures, but an OOM
    surfacing from inside a kernel, a cuBLAS call, or a lower-level RuntimeError
    reaches here as a plain exception whose message is the only evidence. A
    string match alone would miss the typed case on a torch that renames it; the
    typed check alone misses everything that is not raised by the allocator.
    """
    import torch

    typed = getattr(torch.cuda, "OutOfMemoryError", None)
    if typed is not None and isinstance(exc, typed):
        return True
    return "out of memory" in str(exc).lower()


def _release_cuda() -> None:
    """Return the allocator's cached blocks to the driver. Best-effort.

    `empty_cache()` does not free memory held by live tensors -- it releases
    what the caching allocator is holding unused, which after a failed
    generation is the transient prefill workspace. It is the only thing that can
    be done from here without unloading the model, and it is what makes the
    sequential retry a different attempt rather than a repeat of the same one.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Annotator:
    """Holds the model. One instance per process."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 quant: str = "none") -> None:
        """`quant` is recorded, not inferred, because it is not free.

        The card is 32 GB. `gemma-4-12B-it` (23 GB bf16) and
        `gemma-4-31B-it-qat-w4a16` (22 GB, already w4a16 in the checkpoint) fit
        as published; `Qwen3.8-27B` is 56 GB of bf16 and does not fit at all.
        Loading it 4-bit is what makes that arm exist, and it means the bake-off
        compares **what fits on this card**, not the models at equal precision.
        A result from it may never be written up as "model A beats model B".
        """
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_id = model_id
        self.quant = quant
        self.processor = AutoProcessor.from_pretrained(model_id)

        kwargs: dict = {"dtype": torch.bfloat16, "device_map": device}
        if quant == "native-compressed-tensors":
            # MEASURED, twice. Forcing bf16 on a w4a16 checkpoint OOMed at 27.56
            # GB of a 31.36 GB card. "auto" -- the checkpoint's own dtype --
            # loaded at 23.02 GB and then OOMed anyway on the FIRST image.
            #
            # The reason is in the checkpoint: its `ignore` list holds the whole
            # vision tower, so 4-bit covers the language layers and the image
            # encoder stays bf16. 22 GB on disk becomes 23 GB resident with ~8
            # GB left, and encoding eleven 224px views does not fit in 8 GB.
            #
            # `max_memory` caps the card and spills the remainder to host RAM.
            # It is SLOWER -- every offloaded layer crosses PCIe once per
            # forward pass -- and that cost is this arm's, not the other two's,
            # which is one more reason the bake-off compares what fits on this
            # card rather than the models.
            # The load log says what is actually happening: "Decompressing
            # model: 410 items". transformers UNPACKS the 4-bit weights to bf16
            # by default, which is why a 22 GB checkpoint arrives as 23 GB
            # resident and then cannot encode a single image. `run_compressed`
            # keeps them packed and uses the compressed kernels instead.
            #
            # Built from the checkpoint's OWN quantization_config rather than
            # from scratch: `config_groups` and `ignore` describe which tensors
            # are quantised and which are not, and a config assembled here would
            # lose both and quantise the vision tower the checkpoint exempts.
            import json as _json

            from transformers import CompressedTensorsConfig

            qc = _json.loads((Path(model_id) / "config.json").read_text())["quantization_config"]
            qc["run_compressed"] = True
            kwargs["dtype"] = "auto"
            kwargs["quantization_config"] = CompressedTensorsConfig(**qc)
        if quant == "bnb-nf4":
            from transformers import BitsAndBytesConfig

            # nf4 + double quantisation, compute in bf16: the configuration
            # bitsandbytes documents as the accuracy-preserving one. Recorded in
            # the arm's metrics so a reader knows which 4-bit this was -- "4-bit"
            # alone names several quite different things.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant not in {"none", "native-compressed-tensors"}:
            raise ValueError(f"unknown quant {quant!r}")
        # `native-compressed-tensors` needs no kwargs: the checkpoint carries its
        # own `quantization_config`. It DOES need the `compressed_tensors`
        # package, whose absence is what U-R recorded -- imported here so the
        # failure names itself instead of surfacing as a config parse error.
        if quant == "native-compressed-tensors":
            import compressed_tensors  # noqa: F401

        # Resolved from the checkpoint's own `architectures`, not pinned to one
        # class: D-2 has now named two different model families, and a hardcoded
        # class turns "swap the annotator" into an edit of the loader.
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()

        # [ULIP2 Reviewer, 2026-08-23] **"we did not set it" is not "there is no
        # value".** `generation/utils.py:1780-1782` merges user kwargs OVER
        # `model.generation_config` over library defaults, so every sampling
        # parameter this code does not name is silently supplied by the
        # checkpoint -- and the checkpoints DISAGREE:
        #
        #     gemma-4-12B-it             temperature 1.0  top_p 0.95  top_k 64
        #     gemma-4-31B-it-qat-w4a16   temperature 1.0  top_p 0.95  top_k 64
        #     Qwen3.8-27B                temperature 1.0  top_p 0.95  top_k 20
        #
        # `gen_kwargs` overrides temperature and top_p but never top_k, so the
        # 2026-08-22 bake-off compared three models under two different sampling
        # distributions while every record said the settings matched. That
        # experiment cannot be attributed, and `experiments.md` §6 forbids
        # treating a library default as a specified parameter.
        #
        # This captures the EFFECTIVE merged values, read back off the loaded
        # model rather than restated from constants in this file -- restating
        # them would reproduce exactly the blind spot that caused this, because
        # the parameters at issue are the ones the code never mentions.
        #
        # [USER DECISION `U-TK`, 2026-08-23] **top_k keeps the checkpoint's own
        # value and is not overridden.** Options put to the USER were (A) leave
        # it, now that it is recorded, or (B) set it explicitly. A was chosen.
        #
        # This is a decision, not the absence of one, and the distinction is the
        # whole point: the value is identical either way, but until today
        # nothing in the record said which. Production runs ONE model, so the
        # cross-arm disagreement (gemma 64 / Qwen 20) is a retrospective defect
        # in the 2026-08-22 bake-off -- already void for an unrelated reason,
        # the renderer changed -- and not a forward one. Overriding it now would
        # make the new corpus incomparable with that bake-off a SECOND time and
        # buy nothing.
        #
        # An `n05` run that compares models again MUST set top_k explicitly for
        # every arm. That is the condition this decision is scoped to.
        self.effective_generation_config = {
            k: getattr(self.model.generation_config, k, None)
            for k in ("do_sample", "temperature", "top_p", "top_k", "num_beams",
                      "repetition_penalty", "max_new_tokens")
        }
        print(f"[annotator] effective generation_config: "
              f"{self.effective_generation_config}", flush=True)

    def generate(self, image_paths: list[str], prompt: str, *,
                 sample: bool = False, seed: int | None = None,
                 n: int = 1) -> str | list[str]:
        """One forward pass over all views at once. ``str`` if ``n == 1``, else a list.

        `n > 1` draws `n` sampled continuations from a SINGLE prefill instead of
        re-encoding the views once per draw. The draws are still independent --
        multinomial sampling per sequence at the same temperature and top_p --
        so this is an implementation change, not a method change. `V3.1` is the
        gate that has to demonstrate that on real assets before the full run.

        Measured 2026-08-23, gemma-4-12B-it, 12 views, 3193 prompt tokens:

            5 x generate(n=1)   10.94 s      <- five prefills of the same images
            1 x generate(n=5)    5.50 s      5/5 distinct, peak 30.02 GB

        The peak is why `describe_rank` parks its ranker on the CPU: with the
        ranker resident, n=5 does not fit on a 32 GB card in either precision.

        All views go in a single conversation turn, not one call per view: the
        annotation is about ONE object, and asking per view produces per-view
        opinions to reconcile rather than one description informed by every
        angle. Measured on five assets, per-view captioning called one shovel a
        shovel, a hand trowel, a chisel and an axe.
        """
        import torch

        # [2026-08-23] `{"url": p}` was here, which lets the processor open the
        # file itself and call `.convert("RGB")` -- alpha DROPPED, not
        # composited. n04 now writes transparent RGBA, so every anti-aliased
        # silhouette edge would reach the model as its unblended foreground
        # colour. `view_io` composites onto the recorded background and is the
        # single place that decision lives, shared with the CLIP ranker and n06.
        from metafind.data.view_io import load_views_rgb

        content = [{"type": "image", "image": im}
                   for im in load_views_rgb(image_paths)]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # The processor's own chat template, for every arm.
        #
        # This used to go through `qwen_vl_utils.process_vision_info`, which is
        # Qwen's helper and does not know the other two checkpoints. Keeping it
        # for one arm and a different path for the others would have made the
        # arms differ in how they are FED as well as in which model they are --
        # a confound sitting underneath the measurement the bake-off exists for.
        # `enable_thinking=False` on every arm that understands it.
        #
        # MEASURED 2026-08-23, Qwen3.8-27B, same 11 views and same prompt:
        #
        #     thinking on    11.8 s   264 tokens   22.4 tok/s
        #     thinking off    2.9 s    62 tokens   21.8 tok/s
        #
        # The tokens-per-second are the same, so 4-bit decoding is NOT the cost
        # -- an earlier note in this file blaming it was wrong. The whole
        # difference is that a thinking model narrates the task before answering
        # and then hits `max_new_tokens`. It also changes WHAT comes back: with
        # thinking on the reply was the model reasoning about the instruction,
        # not a description, so this is a correctness fix and not only a speed
        # one.
        #
        # Passed through `**` because a processor whose template has no such
        # variable raises on an unexpected keyword, and gemma and Qwen do not
        # have to agree about it for the arms to stay comparable -- what has to
        # match is that neither narrates.
        template_kwargs: dict = {}
        if "enable_thinking" in (getattr(self.processor, "chat_template", "") or ""):
            template_kwargs["enable_thinking"] = False
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        ).to(self.model.device)

        # Greedy by default. The repair loop feeds the specific error back, so
        # a retry must differ because the PROMPT differs -- not because the
        # sampler rolled differently. Temperature there would make a failure
        # that repeats look like one that was fixed.
        #
        # `sample=True` is for the description candidates alone. ULIP-2
        # generates its candidates "independently" (main.tex:677), and greedy
        # decoding would return the SAME sentence five times -- a ranking over
        # five identical strings is not a ranking. The seed is per candidate and
        # recorded, so independent still means reproducible.
        gen_kwargs: dict = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": sample}
        if sample:
            gen_kwargs |= {"temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P}
            if seed is not None:
                torch.manual_seed(seed)
        if n > 1:
            # [ULIP2 Reviewer, 2026-08-23 -- read against transformers 5.15.0]
            # The independence claim in this docstring holds ONLY for
            # `do_sample=True, num_beams=1`. `_expand_inputs_for_generation`
            # (generation/utils.py:929) repeat_interleaves the prompt into `n`
            # rows and `_sample` (2921-2923) draws with `torch.multinomial` over
            # `(n, vocab)`, which samples each ROW independently -- that is the
            # mechanism, and it is why this is an implementation change.
            #
            # Beam search takes a different path where the rows are beam
            # hypotheses and are NOT independent. Nothing here sets num_beams,
            # so the premise holds today; the assert is so that adding it later
            # fails loudly instead of silently turning five independent draws
            # into five beams while every record still says "sampled".
            # [CORRECTED by ULIP2 Reviewer, 2026-08-23] This checked
            # `gen_kwargs`, which is built two lines above and into which
            # nothing ever puts `num_beams` -- so the check was constant-true
            # and could not fire. What actually decides beam search is the
            # MERGED config, and a checkpoint whose `generation_config.json`
            # carries `num_beams: 4` was precisely the scenario the guard's own
            # comment claimed to cover. Reading the effective value is the
            # difference between a guard and a comment.
            effective_beams = gen_kwargs.get(
                "num_beams",
                getattr(self.model.generation_config, "num_beams", 1) or 1)
            if effective_beams != 1 or not sample:
                raise ValueError(
                    "n > 1 draws independent samples and requires "
                    "do_sample=True with num_beams=1; got "
                    f"sample={sample} effective num_beams={effective_beams}")
            gen_kwargs["num_return_sequences"] = n
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        replies = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return replies if n > 1 else replies[0]


class SharedViewPrefix:
    """[SPEEDUP 2026-08-24, USER: 方案一] Encode this asset's views ONCE and let
    every call for the asset continue from that encoding.

    MEASURED before writing this: the draw prompt and the structured prompt
    tokenize to 3,195 and 4,106 tokens, of which the first **3,121 are
    identical** -- 97.7% of the draw call and 76.0% of the structured call. The
    shared part is exactly the twelve images plus the common instruction
    opening, and the images are what a prefill spends its time on. n05's
    9.5 s/asset was paying that vision cost twice per asset (and the batched
    n=5 draw expands its inputs BEFORE the prefill, so the draw was paying it
    five times over rows).

    Mechanism: prefill the common prefix once into a `DynamicCache` built with
    the model's own text config -- **the same construction `generate()` uses
    internally** (`_prepare_cache_for_generation`:
    `DynamicCache(config=self.config.get_text_config(decoder=True))`), so the
    40-of-48 sliding-window layers this checkpoint declares (window 1024) get
    sliding layers here exactly as they do on the plain path. A config-less
    cache would silently give all 48 layers full history: ~3x the memory, and
    a cache structure the normal path never produces. [ULIP2 Reviewer,
    2026-08-24 -- the BLOCKER on this class's first version.]

    Every call then generates on a COPY of the primed cache (`deepcopy`, plus
    `batch_repeat_interleave` for the n=5 draw), and the copy is discarded.
    The base itself is never generated on, so it needs no restoring -- the
    first version crop()ed the base back after each call, which cannot work on
    sliding layers (evicted states are gone; `DynamicSlidingWindowLayer.crop`
    raises at window overflow) and had already tripped the 4.x/5.x crop-sign
    shim. Copies cost ~0.5 GB transient at this window size, against the
    ~1.2 GB per asset the config-less full cache held for the asset's whole
    lifetime.

    **Numerical status: IMPLEMENTATION CHOICE, recorded per record.** A cached
    prefix changes reduction order the same way batch shape does -- this file
    already records `batch_shape` because batch=1 and batch=5 disagree by up to
    1.25e-1 in logits. The same honesty applies here: every record written
    through this path carries `vision_prefix_reuse: true`, so a reproduction
    knows which numerics produced it. The distribution is unchanged by design:
    same model, same tokens, same sampler, same seed handling.

    MEASURED 2026-08-24, 2 assets, both paths: the GREEDY structured call is
    **identical character for character** (~500 argmax tokens, zero
    divergence), so the logits agree wherever a deterministic decode looks.
    The SAMPLED draws at the same seed differ (0/5 identical strings) -- the
    two paths consume randomness against microscopically different logits, a
    reseed-like effect, not a distribution change. A draw is therefore
    reproducible given its recorded configuration (seed, batch_shape,
    prefix_reuse), the same standard batch_shape was accepted under.
    [ULIP2 Reviewer ruling, 2026-08-24: no version bump -- PROMPT_VERSION
    gates the prompt/validator/schema contract and none of those changed.]

    **Fallback is structural, not exceptional.** `ok` is False when the
    annotator is a test fake, the tokenizations disagree with the primed
    prefix, or the cache APIs are missing -- those fall back to the plain path
    silently and the record says so. Runtime exceptions (OOM included) are NOT
    swallowed here: they propagate to `annotate_one`, whose C4 handling is the
    place that decides what an OOM means.
    """

    def __init__(self, ann, image_paths: list[str], prompts: list[str],
                 preloaded_images=None) -> None:
        self.ann = ann
        self.ok = False
        self.base = None
        self.base_len = 0
        self._image_paths = list(image_paths)
        self._full = {}          # prompt -> tokenized inputs (on device)
        model = getattr(ann, "model", None)
        processor = getattr(ann, "processor", None)
        if model is None or processor is None or not hasattr(model, "device"):
            return               # a test fake; plain path
        try:
            import torch
            from transformers import DynamicCache
        except Exception:  # noqa: BLE001 -- no torch, no cache path
            return
        self._torch = torch
        self._DynamicCache = DynamicCache
        if not (hasattr(DynamicCache, "batch_repeat_interleave")
                and hasattr(DynamicCache, "crop")):
            return               # cache API moved; plain path is always correct

        from metafind.data.view_io import load_views_rgb

        images = (preloaded_images if preloaded_images is not None
                  else load_views_rgb(image_paths))

        def tokenize(prompt):
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            kw = {}
            if "enable_thinking" in (getattr(processor, "chat_template", "") or ""):
                kw["enable_thinking"] = False
            return processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt", **kw,
            ).to(model.device)

        try:
            toks = [tokenize(pr) for pr in prompts]
        except Exception:  # noqa: BLE001 -- tokenizer quirk: plain path
            return
        ids = [t["input_ids"][0] for t in toks]
        n = 0
        limit = min(len(i) for i in ids)
        while n < limit and all(bool(i[n] == ids[0][n]) for i in ids[1:]):
            n += 1
        if n < 16:
            return               # no meaningful shared prefix; nothing to reuse
        first = toks[0]
        seq = first["input_ids"].shape[1]
        # Every image must live INSIDE the prefix -- a continuation never gets
        # pixel_values, so an image token past the cut would be a token with no
        # features. `image_position_ids` marks image positions with >= 0.
        # `mm_token_type_ids` is (1, seq) aligned with input_ids, image tokens
        # marked 1 -- the per-TOKEN channel. `image_position_ids` is (12, 280, 2),
        # PER-IMAGE grid coordinates: indexing [0] takes one image, its max is
        # bounded by the grid, and a guard built on it could never fail.
        # [ULIP2 Reviewer, 2026-08-24 -- named as a could-not-fail candidate;
        # confirmed, replaced.] No per-token channel -> structural fallback.
        mm = first.get("mm_token_type_ids")
        if mm is None or mm.shape[1] != seq:
            return
        img_pos = (mm[0] == 1).nonzero()
        if len(img_pos) == 0 or int(img_pos.max()) >= n:
            return
        # Slice every per-token tensor to the prefix; whole-image tensors
        # (pixel_values) pass through untouched.
        prefix = {}
        for k, v in first.items():
            if hasattr(v, "ndim") and v.ndim >= 2 and v.shape[1] == seq:
                prefix[k] = v[:, :n]
            else:
                prefix[k] = v
        try:
            cfg = model.config.get_text_config(decoder=True)
            with torch.no_grad():
                out = model(**prefix, use_cache=True,
                            past_key_values=DynamicCache(config=cfg))
            self.base = out.past_key_values
        except Exception:  # noqa: BLE001 -- prefill refused; plain path
            return
        self.base_len = n
        self._prefix_ids = ids[0][:n]
        self._full = {pr: t for pr, t in zip(prompts, toks)}
        self._tokenize = tokenize
        self.ok = True

    def _inputs_for(self, prompt):
        t = self._full.get(prompt)
        if t is None:
            t = self._tokenize(prompt)
            self._full[prompt] = t
        ids = t["input_ids"]
        if (ids.shape[1] < self.base_len
                or not bool((ids[0, :self.base_len] == self._prefix_ids).all())):
            return None          # this prompt does not share the prefix
        return t

    def _copy(self, repeats: int = 1):
        # transformers 5 removed the legacy tuple API; the supported route is a
        # deep copy (clones the KV tensors, base untouched), expanded in place
        # for n > 1. The copy is the whole isolation story: the base is never
        # generated on, so sliding-layer eviction during a call can never
        # corrupt the prefix the next call starts from.
        import copy

        grown = copy.deepcopy(self.base)
        if repeats > 1:
            grown.batch_repeat_interleave(repeats)
        return grown

    def generate(self, prompt: str, *, sample: bool = False,
                 seed: int | None = None, n: int = 1):
        """Same contract as `Annotator.generate`, minus re-encoding the views."""
        if not self.ok:
            raise RuntimeError("SharedViewPrefix used while not ok")
        torch = self._torch
        t = self._inputs_for(prompt)
        if t is None:            # unshared prompt: pay the full price, stay correct
            return self.ann.generate(self._image_paths, prompt,
                                     sample=sample, seed=seed, n=n)
        ids = t["input_ids"]
        mask = t.get("attention_mask")
        gen_kwargs: dict = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": sample}
        if sample:
            gen_kwargs |= {"temperature": SAMPLING_TEMPERATURE,
                           "top_p": SAMPLING_TOP_P}
            if seed is not None:
                torch.manual_seed(seed)
        if n > 1:
            effective_beams = getattr(
                self.ann.model.generation_config, "num_beams", 1) or 1
            if effective_beams != 1 or not sample:
                raise ValueError(
                    "n > 1 draws independent samples and requires do_sample=True "
                    f"with num_beams=1; got sample={sample} "
                    f"effective num_beams={effective_beams}")
            cache = self._copy(n)
            ids = ids.repeat(n, 1)
            mask = mask.repeat(n, 1) if mask is not None else None
        else:
            cache = self._copy()
        with torch.no_grad():
            out = self.ann.model.generate(
                input_ids=ids, attention_mask=mask,
                past_key_values=cache, **gen_kwargs)
        del cache
        trimmed = out[:, t["input_ids"].shape[1]:]
        replies = self.ann.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return replies if n > 1 else replies[0]


def annotate_one(ann: Annotator, uid: str, render_rec: dict, *,
                 lvis_category: str | None,
                 proportions: tuple[float, float, float],
                 preloaded_images=None) -> tuple[dict | None, dict | None]:
    """SG1 for one asset. Returns ``(record, quarantine_entry)`` -- exactly one is None.

    `lvis_category=None` is the UNANCHORED path (`PROMPT_VERSION 7`): no
    identity is supplied, the model answers `synset` itself, and the
    description comes from ULIP-2's generate-many-and-rank rather than from the
    structured response.
    """
    views = render_rec["view_paths"]
    anchored = lvis_category is not None

    # [SPEEDUP 2026-08-24] Both prompts are known before any call, so the
    # shared image prefix can be primed once and every call for this asset --
    # draw, structured, and any repair -- continues from it. `sv.ok` False
    # (a test fake, a moved cache API, a tokenizer that disagrees) falls back
    # to the plain per-call path, and the record says which path ran.
    draw_prompt = build_description_prompt(len(views), lvis_category)
    struct_prompt = (build_prompt(len(views), lvis_category, proportions)
                     if anchored else build_unanchored_prompt(len(views), proportions))
    sv = SharedViewPrefix(ann, views, [draw_prompt, struct_prompt],
                          preloaded_images=preloaded_images)
    prefix_reuse = sv.ok

    def gen(prompt, **kw):
        return sv.generate(prompt, **kw) if sv.ok else ann.generate(views, prompt, **kw)

    # --- description candidates, ULIP-2's method (main.tex:677) --------------
    candidates: list[str] = []
    ranked: list[dict] = []
    # [PROMPT_VERSION 8] Both modes now go through the ranking. Seeded from the
    # uid, so the same asset draws the same five candidates on a re-run while
    # different assets do not share a draw. `uid_seed` is n03's, reused rather
    # than reinvented so the two nodes cannot disagree about what "this asset's
    # seed" means.
    rejected_non_english: list[str] = []
    base = uid_seed(uid)
    # [2026-08-23] All N candidates come from ONE prefill of the views instead
    # of N. Sampling is unchanged -- N independent multinomial draws at the same
    # temperature and top_p -- so the candidate set has the same distribution;
    # what changes is that the vision tower runs once instead of N times.
    # Measured 10.94 s -> 5.50 s at N=5 on 12 views. `V3.1` verifies the
    # equivalence on real assets before the full run.
    #
    # ONE seed for the batch, not one per candidate: `torch.manual_seed` is set
    # once and the N sequences then diverge from the shared generator, so a
    # per-candidate seed no longer describes anything. `description_sampling`
    # records the batch seed, which is what actually reproduces the draw.
    draw_mode = "batched"
    try:
        drawn = gen(draw_prompt, sample=True, seed=base, n=N_CANDIDATES)
    except Exception as exc:  # noqa: BLE001 -- see below; a failed batch is not the asset
        # [C4, 2026-08-24] OOM is handled BEFORE the fallback, not folded into
        # it. See `_is_cuda_oom`: at 675 MiB of headroom this is the likeliest
        # exception here, and issuing five more prefills on a fragmented card
        # without freeing the workspace first is the one response that cannot
        # work. The distinction is also RECORDED -- `draw_mode` reaches the
        # sidecar, so an OOM-driven fallback is no longer indistinguishable from
        # a per-asset one, and the run's OOM rate becomes countable afterwards
        # instead of invisible.
        oom = _is_cuda_oom(exc)
        draw_mode = "oom_sequential_fallback" if oom else "sequential_fallback"
        if oom:
            _release_cuda()
        # Falling back to one-at-a-time rather than quarantining: the batch can
        # fail for a reason that is about THIS asset's memory footprint (a very
        # detailed mesh, an unusually long prompt) while the same asset renders
        # fine drawn singly, and losing it would be a data loss caused by an
        # optimisation.
        drawn = []
        for k in range(N_CANDIDATES):
            try:
                drawn.append(gen(draw_prompt, sample=True, seed=base + k))
            except Exception as inner:  # noqa: BLE001 -- one bad draw is not the asset
                # Free again per draw: without this the FIRST single draw to OOM
                # leaves its workspace cached and the remaining draws inherit a
                # card that is still full, so one OOM would reliably become five.
                if _is_cuda_oom(inner):
                    _release_cuda()
                drawn.append("")
    if isinstance(drawn, str):
        drawn = [drawn]
    for text in drawn:
        text = (text or "").strip()
        # [PROMPT_VERSION 8] The language rule has to be applied HERE, not in
        # the validator. The description no longer comes back in the structured
        # response, so a non-English one cannot be repaired by re-prompting the
        # structured call -- the repair loop would spend both attempts on a
        # field it cannot reach and quarantine a perfectly good asset. Dropping
        # the candidate costs one of five draws instead.
        if text and not non_english_characters(text):
            candidates.append(text)
        elif text:
            rejected_non_english.append(text)
    if not candidates:
        return None, {
            "uid": uid, "failure_class": "DETERMINISTIC_INPUT",
            "exception_type": "NoDescriptionCandidates",
            "exception_msg": (
                f"no usable description from {N_CANDIDATES} draws"
                + (f"; {len(rejected_non_english)} were rejected as non-English, "
                   f"e.g. {rejected_non_english[0][:80]!r}" if rejected_non_english else "")),
            "traceback": "",
        }
    winner, ranked = rank_descriptions(views, candidates)

    prompt = struct_prompt
    current = prompt
    last_error = ""
    last_raw = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = gen(current)
        last_raw = raw
        try:
            parsed = parse_annotation(raw)
            annotation = validate_annotation(
                parse_annotation(raw),
                lvis_category=lvis_category,
                proportions=proportions,
                description=winner,
            )
        except AnnotationError as exc:
            last_error = str(exc)
            if attempt < MAX_ATTEMPTS:
                # The specific error goes back in. Re-sending the original
                # reproduces the original mistake.
                current = build_repair_prompt(prompt, last_error, raw)
            continue

        # [PROMPT_VERSION 9] The winner is the best DESCRIPTION; it is not
        # necessarily one that fits. Swap in the best that does, before the
        # record is built -- after it, the string is already the artifact.
        annotation, description_fit = _fit_description(
            parsed, lvis_category, proportions, ranked, ann.model_id)
        if annotation is None:
            return None, {
                "uid": uid,
                "failure_class": "DETERMINISTIC_INPUT",
                "exception_type": "NoDescriptionFitsContext",
                "exception_msg": (
                    f"none of {description_fit['candidates_tried']} ranked "
                    f"descriptions could be used. "
                    f"{len(description_fit['over_context'])} exceeded "
                    f"{description_fit['context_length']} tokens "
                    f"({description_fit['over_context']}); "
                    f"{len(description_fit['rejected_by_validator'])} were "
                    f"refused by the validator (ranks "
                    f"{description_fit['rejected_by_validator']}). The asset is "
                    "QUARANTINED rather than truncated: a mid-sentence tail is "
                    "what PROMPT_VERSION 9 exists to stop."),
                "traceback": "",
            }

        rec = annotation.as_record(ann.model_id)
        rec |= {
            "uid": uid,
            "prompt_version": PROMPT_VERSION,
            # Which ranked candidate actually survived the context bound, and
            # how long it came out. `rank_used` > 0 on many assets means the
            # word budget is too loose; without the field that is invisible.
            "description_fit": description_fit,
            "attempts": attempt,
            # [PROMPT_VERSION 7] Every candidate and its score, not just the
            # winner. `E-10` promises the spread is recorded so "would ten have
            # been better than five" is answerable from the data; keeping only
            # the winner throws that away and it cannot be recomputed without
            # re-running the model.
            "description_candidates": ranked,
            "description_ranker": DESCRIPTION_RANKER,
            # Recorded, not silently dropped: a model that keeps producing
            # Chinese is telling the USER something about itself, and a run
            # where this is nonzero everywhere means the language instruction
            # is not landing.
            "description_candidates_rejected_non_english": len(rejected_non_english),
            # `seed_base` is now the seed for the WHOLE batch, not the first of
            # N per-candidate seeds: one `manual_seed(base)` precedes a single
            # generate that returns N sequences. `draw` records which of the two
            # paths ran, because the fallback still seeds per candidate and a
            # record that did not say so would not reproduce.
            #
            # [ULIP2 Reviewer, 2026-08-23] `batch_shape` is the real cost of the
            # batched draw and it needs its own field. Reproducing candidate 3
            # now requires re-running the WHOLE batch at the SAME shape: a batch
            # of 5 and a batch of 3 do not share a candidate 3, because the
            # sampler consumes the generator differently and bf16 matmul
            # reduction order changes with the batch (measured: 1.25e-1 max
            # logit difference between batch=1 and batch=5, large enough to flip
            # a token). Without this field, "same base seed" means different
            # things in two records that look identical.
            #
            # `effective` is the MERGED config read back off the loaded model,
            # not this file's constants. It is what catches the parameters the
            # code never names -- `top_k` is supplied by the checkpoint (64 for
            # gemma, 20 for Qwen) and went unrecorded through an entire
            # model comparison. A record listing only what the code set cannot
            # detect that class of confound, which is why the two fields are
            # kept side by side rather than merged.
            "description_sampling": {"temperature": SAMPLING_TEMPERATURE,
                                     "top_p": SAMPLING_TOP_P,
                                     "seed_base": uid_seed(uid),
                                     "n": N_CANDIDATES,
                                     "draw": draw_mode,
                                     "batch_shape": (N_CANDIDATES
                                                     if draw_mode == "batched" else 1),
                                     "effective": getattr(
                                         ann, "effective_generation_config", None),
                                     # Which numerics produced this record --
                                     # a cached prefix shifts logits the way
                                     # batch shape does (see SharedViewPrefix).
                                     "prefix_reuse": prefix_reuse},
        "vision_prefix_reuse": prefix_reuse,
            # F13: the annotator saw scale-normalised renders, so its size
            # estimate is a category prior. The mesh's own bounding box travels
            # with it so the estimate can be audited -- and it is a WEAK ground
            # truth, since Objaverse authors choose their own units.
            "raw_bbox_extents": render_rec.get("raw_bbox_extents"),
            # [ADDED 2026-08-24] WHICH images the model was shown. Completion
            # compared the annotation contract and nothing about the renders, so
            # a re-rendered corpus (the OptiX swap, `RENDERER_VERSION` 5 -> 6)
            # left every record "current" while describing images that had been
            # deleted. See `view_io.image_identity`.
            "image_identity": image_identity(render_rec),
            "renderer_version": render_rec.get("renderer_version"),
            # The exact triple the prompt showed the model, so the derived
            # width/length can be recomputed from the record alone.
            "mesh_proportions_yxz": list(proportions),
        }
        return rec, None

    # Exhausted. Quarantine, never admit: L1-ANNOT-EXHAUST.
    return None, {
        "uid": uid,
        "failure_class": "MODEL_RECOVERABLE",
        "terminated_by": "repair_budget",
        "attempts": MAX_ATTEMPTS,
        "exception_type": "AnnotationError",
        "exception_msg": last_error[:400],
        "raw_response": last_raw[:1000],
    }


def rebuild_index(index_path: Path) -> int:
    """Derive annotations_index.jsonl from the per-asset sidecars."""
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as f:
        for sc in sorted(out_root().glob("*.json")):
            try:
                f.write(json.dumps(json.loads(sc.read_text())) + "\n")
                n += 1
            except (OSError, json.JSONDecodeError):
                continue
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    # A specific uid list, for validation batches. `--limit N` takes the first N
    # of a SORTED corpus, which is fine for a smoke test and useless for
    # measuring against ground truth: the assets that happen to sort first are
    # not the assets AI2-THOR can adjudicate.
    ap.add_argument("--uids-file", type=Path,
                    help="annotate exactly these uids, one per line")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", default="none",
                    choices=["none", "bnb-nf4", "native-compressed-tensors"],
                    help="how to load the weights; recorded per arm because it "
                         "is not free -- see Annotator.__init__")
    ap.add_argument("--arm", help="write to data/outputs/bakeoff/<arm>/ instead "
                                  "of the corpus annotations directory")
    args = ap.parse_args()

    if args.arm:
        print(f"arm {args.arm!r} -> {use_arm(args.arm)}", flush=True)

    out_root().mkdir(parents=True, exist_ok=True)
    renders = {}
    index = paths.LOGS / "renders_index.jsonl"
    if not index.exists():
        print(f"{index} not found -- run n04_render_views first", flush=True)
        return 2
    for line in index.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            renders[r["uid"]] = r

    if args.uids_file:
        candidates = [u for u in args.uids_file.read_text().split() if u]
        missing = [u for u in candidates if u not in renders]
        if missing:
            print(f"{len(missing)} uid(s) have no render, e.g. {missing[:3]}", flush=True)
            return 2
    else:
        candidates = sorted(renders)

    try:
        todo, blocked, states = build_work_list(
            candidates, args.force,
            image_ids={u: image_identity(r) for u, r in renders.items()})
    except ProvenanceRegistryError as exc:
        print(f"{exc}\nRefusing: the provenance registry is what accounts for the "
              "existing corpus, so a registry that cannot be trusted is a reason to "
              "stop, never a reason to proceed. Rebuild it with "
              "tools/declare_annotation_provenance.py --declare.", flush=True)
        return 3
    if blocked:
        print(f"{len(blocked):,} existing annotation record(s) carry neither the "
              f"current contract {annotation_contract_id()} nor a declaration in "
              f"{PROVENANCE_REGISTRY}, e.g. {blocked[:3]}.\n"
              "Refusing: an unclassified record must be neither silently re-annotated "
              "nor silently skipped. Declare it "
              "(tools/declare_annotation_provenance.py --declare) or re-annotate it "
              "explicitly with --force.", flush=True)
        return 3
    for state in (CURRENT_CONTRACT, ACCEPTED_LEGACY_V3, LEGACY_V1_RESIDUAL):
        n = sum(1 for st in states.values() if st == state)
        if n:
            print(f"  {state:<34} {n:>7,}  (not queued)", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    # [PROMPT_VERSION 5] Both anchors are resolved BEFORE the model loads. An
    # asset with no LVIS category or no mesh proportions cannot be annotated
    # under v5 at all, and discovering that 19 hours into a run -- or silently
    # falling back to a v4-style guess -- are both worse than stopping here.
    # [PROMPT_VERSION 7] The LVIS category is no longer supplied to the model.
    # It is still LOADED, because the mesh-proportion guard below shares this
    # pass and because an asset outside the LVIS subset is outside the corpus
    # this milestone is defined on -- but it is not passed to `annotate_one`.
    lvis_categories = load_lvis_categories()
    proportions = load_proportions()
    no_anchor = [u for u in todo if u not in lvis_categories or u not in proportions]
    if no_anchor:
        print(f"{len(no_anchor):,} queued uid(s) have no LVIS category or no mesh "
              f"proportions, e.g. {no_anchor[:3]}.\n"
              "Refusing: an asset outside the Objaverse-LVIS subset is outside "
              "would have to be annotated under a different contract than the rest "
              "of the corpus. Exclude them explicitly or resolve the gap.",
              flush=True)
        return 3

    print(f"{len(renders):,} rendered assets, {len(todo):,} to annotate", flush=True)
    if not todo:
        return 0

    ann = Annotator(args.model, quant=args.quant)
    done, quarantined, started = 0, 0, time.time()
    # [SPEEDUP 2026-08-24, USER: 方案三] Decode the NEXT asset's twelve PNGs on
    # a CPU thread while the GPU is busy with THIS one. Pure pipelining: the
    # images that reach the model are byte-identical to a synchronous load, and
    # a preload failure degrades to the synchronous path inside annotate_one
    # (preloaded_images=None), never to a skipped asset.
    from concurrent.futures import ThreadPoolExecutor

    from metafind.data.view_io import load_views_rgb

    def _preload(u):
        try:
            return load_views_rgb(renders[u]["view_paths"])
        except Exception:  # noqa: BLE001 -- the real load will report it properly
            return None

    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(_preload, todo[0]) if todo else None
    with runlog.run_progress(NODE):
        for i, uid in enumerate(todo):
            images = pending.result() if pending is not None else None
            pending = (pool.submit(_preload, todo[i + 1])
                       if i + 1 < len(todo) else None)
            try:
                rec, bad = annotate_one(
                    ann, uid, renders[uid],
                    preloaded_images=images,
                    # [PROMPT_VERSION 8] The LVIS label IS supplied, and
                    # `annotate.anchored` is therefore True for every asset --
                    # 46,207 entries in this table, 0 of them empty. The
                    # `resolve_synset` ladder is the only path a synset takes.
                    #
                    # This comment previously read "None = the unanchored
                    # path", left over from v7, while the line below it passed
                    # the real category. A reader who trusted it would conclude
                    # the ladder was dead code -- one did (ULIP2 Reviewer,
                    # 2026-08-23). `code-changes.md` §14: a comment that
                    # contradicts the line under it is worse than no comment.
                    lvis_category=lvis_categories[uid],
                    proportions=proportions[uid],
                )
            except Exception as exc:  # noqa: BLE001 -- one asset must not stop the run
                runlog.quarantine(NODE, [{
                    "uid": uid,
                    "failure_class": ("RESOURCE" if "memory" in str(exc).lower()
                                      else "TRANSIENT"),
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                quarantined += 1
                continue

            if bad is not None:
                runlog.quarantine(NODE, [bad])
                quarantined += 1
                continue

            sc = sidecar_path(uid)
            tmp = sc.with_suffix(".json.part")
            with tmp.open("w") as fh:
                json.dump(rec, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            if args.force:
                tmp.replace(sc)          # explicit: overwrite whatever is there
            else:
                # Classification decided this uid had NO record. Between then and
                # now another writer could have created one, and a bare run may
                # not overwrite an existing record -- that is the whole of AC-1.
                # `os.link` refuses to clobber, so the check and the create are one
                # atomic step rather than a window; `sc.exists()` here would still
                # leave a race.
                try:
                    os.link(tmp, sc)
                except FileExistsError:
                    tmp.unlink()
                    print(f"  {uid} gained a record after classification; skipping "
                          "(use --force to overwrite)", flush=True)
                    continue
                tmp.unlink()
            done += 1
            if done % 100 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                print(f"  [{done:6d}/{len(todo)}] {rate:.1f}/min, "
                      f"剩餘約 {(len(todo)-done)/max(rate,1e-9):.0f} 分, "
                      f"quarantine {quarantined}", flush=True)

    # The index follows the records. An arm writing its index into
    # `logs/annotations_index.jsonl` would overwrite the corpus index with
    # 100 experimental rows, which is the same contamination one directory up.
    index_path = (paths.LOGS / "annotations_index.jsonl"
                  if _ARM_ROOT is None
                  else _ARM_ROOT.parent / "annotations_index.jsonl")
    n_indexed = rebuild_index(index_path)
    runlog.cost_ledger(
        wallclock_s=round(time.time() - started, 1),
        assets_annotated=done,
        vlm_calls=done + quarantined * MAX_ATTEMPTS,
    )
    print(f"\n{done:,} annotated this run, {n_indexed:,} complete on disk, "
          f"{quarantined:,} quarantined -> {out_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
