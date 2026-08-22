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

# BEFORE any transformers/torch import anywhere in this process. HF_HOME is read
# at IMPORT time, not at from_pretrained time, so setting it inside Annotator
# after `import transformers` had no effect: the run started re-downloading 16 GB
# of weights into ~/.cache on the 100 GB root partition while a complete copy sat
# on the data volume. It looked like a slow load for six minutes.
paths.setup_env()

from metafind.data.annotate import (
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    AnnotationError,
    annotation_contract_id,
    build_prompt,
    build_repair_prompt,
    parse_annotation,
    validate_annotation,
    LVIS_SYNSETS,
)

NODE = "n05_annotate"
MODEL_ID = "/mnt/data1/kyzen/models/Qwen3.8-27B"  # D-2: stands in for GPT-4o.
# [DEVIATION D-2, re-pointed 2026-08-21 on the user's decision] The paper
# annotates with GPT-4o (`2methdology.tex:28`, `neurips_2025.tex:100`).
# It did so with Qwen2.5-VL-7B-Instruct before this task. Neither is GPT-4o;
# the deviation is RE-POINTED, not discharged, and must never be written up
# as paper-faithful.
MAX_NEW_TOKENS = 512


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


def _under_current_contract(rec: dict) -> bool:
    # Keyed on the ANNOTATION CONTRACT, not on prompt_version alone. A v1 sidecar
    # has every field name it shares with v2 but a different schema, and treating
    # it as done would silently mix two annotation generations in one corpus --
    # and prompt_version cannot express the other two axes: the same
    # `prompt_version: 3` could have been admitted by a validator with or without
    # the language rule. The contract id folds prompt, validator and schema
    # semantics into one comparison.
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


def provenance_state(uid: str, registry: dict[str, tuple[str, int, str]]) -> str | None:
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
    if _under_current_contract(rec):
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
    return state


def classify_all(candidates, registry) -> dict[str, str | None]:
    """The predicate a bare run builds its work list from. No model, no GPU."""
    return {uid: provenance_state(uid, registry) for uid in candidates}


def build_work_list(candidates, force: bool,
                    registry=None) -> tuple[list[str], list[str], dict[str, str | None]]:
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
                          load_provenance_registry() if registry is None else registry)
    return ([uid for uid, st in states.items() if st is None],
            [uid for uid, st in states.items() if st == UNACCOUNTED],
            states)


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


class Annotator:
    """Holds the model. One instance per process."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Resolved from the checkpoint's own `architectures`, not pinned to one
        # class: D-2 has now named two different model families, and a hardcoded
        # class turns "swap the annotator" into an edit of the loader.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

    def generate(self, image_paths: list[str], prompt: str) -> str:
        """One forward pass over all 11 views at once.

        All views in a single conversation turn, not eleven separate calls: the
        annotation is about ONE object, and asking eleven times would produce
        eleven opinions to reconcile rather than one description informed by
        every angle.
        """
        import torch

        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                # Greedy. The repair loop feeds the specific error back, so a
                # retry must differ because the PROMPT differs -- not because
                # the sampler rolled differently. Temperature here would make a
                # failure that repeats look like one that was fixed.
                do_sample=False,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def annotate_one(ann: Annotator, uid: str, render_rec: dict, *,
                 lvis_category: str,
                 proportions: tuple[float, float, float]) -> tuple[dict | None, dict | None]:
    """SG1 for one asset. Returns ``(record, quarantine_entry)`` -- exactly one is None."""
    views = render_rec["view_paths"]
    prompt = build_prompt(len(views), lvis_category, proportions)
    current = prompt
    last_error = ""
    last_raw = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = ann.generate(views, current)
        last_raw = raw
        try:
            annotation = validate_annotation(
                parse_annotation(raw),
                lvis_category=lvis_category,
                proportions=proportions,
            )
        except AnnotationError as exc:
            last_error = str(exc)
            if attempt < MAX_ATTEMPTS:
                # The specific error goes back in. Re-sending the original
                # reproduces the original mistake.
                current = build_repair_prompt(prompt, last_error, raw)
            continue

        rec = annotation.as_record(ann.model_id)
        rec |= {
            "uid": uid,
            "prompt_version": PROMPT_VERSION,
            "attempts": attempt,
            # F13: the annotator saw scale-normalised renders, so its size
            # estimate is a category prior. The mesh's own bounding box travels
            # with it so the estimate can be audited -- and it is a WEAK ground
            # truth, since Objaverse authors choose their own units.
            "raw_bbox_extents": render_rec.get("raw_bbox_extents"),
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
        todo, blocked, states = build_work_list(candidates, args.force)
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
    lvis_categories = load_lvis_categories()
    proportions = load_proportions()
    no_anchor = [u for u in todo if u not in lvis_categories or u not in proportions]
    if no_anchor:
        print(f"{len(no_anchor):,} queued uid(s) have no LVIS category or no mesh "
              f"proportions, e.g. {no_anchor[:3]}.\n"
              "Refusing: v5 is category-anchored, and an asset without an anchor "
              "would have to be annotated under a different contract than the rest "
              "of the corpus. Exclude them explicitly or resolve the gap.",
              flush=True)
        return 3

    print(f"{len(renders):,} rendered assets, {len(todo):,} to annotate", flush=True)
    if not todo:
        return 0

    ann = Annotator(args.model)
    done, quarantined, started = 0, 0, time.time()
    with runlog.run_progress(NODE):
        for uid in todo:
            try:
                rec, bad = annotate_one(
                    ann, uid, renders[uid],
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
