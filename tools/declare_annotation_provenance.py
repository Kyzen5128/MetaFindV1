#!/usr/bin/env python
"""Declare, and re-check, which population accounts for every stored annotation.

# SUPPORTS-NODE: n05_annotate

[D2a, AC-1.c / AC-1.e] The annotation corpus predates contract stamping, so no
stored record carries an `annotation_contract`. That absence is a fact about
when the records were written, and it must not be read as a verdict in either
direction: not as "these are done" (which would let a stale corpus masquerade as
current) and not as "these need redoing" (which is what queued all 45,955 for a
~19.6 GPU-hour rewrite, including 3 records whose fate D0-003 has not decided).

So the populations are DECLARED, once, in an explicit registry, rather than
inferred from a missing field. `annotate_run` consults that registry; a record
neither declared nor carrying the current contract is UNACCOUNTED, and an
unaccounted record stops the run instead of joining the queue.

    --declare   build the registry from the corpus and write it
    (default)   re-check the registry against the corpus and against
                annotate_run's real work-list predicate

Neither mode loads the annotation model, touches the GPU, or writes a single
annotation record.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402
from metafind.data import annotate_run as R  # noqa: E402
from metafind.data.annotate import (  # noqa: E402
    REQUIRED_FIELDS,
    VALIDATOR_VERSION,
    AnnotationError,
    annotation_contract_id,
    validate_annotation,
)

# [TASK.md D2a §6] The three legacy-v1 residuals, named by the contract. They are
# listed here so the classifier below can be CHECKED against the population the
# project says exists, rather than quietly declaring whatever it happens to find.
# They are NOT legacy-v3, they are NOT migrated, and D0-003 is UNRESOLVED.
DECLARED_V1_RESIDUALS = frozenset({
    "6c7db00cc164467ebac356a5ca67368b",
    "8a0192eee6fb4140bb3e9696b3dbae5a",
    "a397b648d6eb48d7909d1ee11235e78f",
})

LEGACY_V3_PROMPT_VERSION = 3
LEGACY_V1_PROMPT_VERSION = 1


def _records() -> dict[str, tuple[dict, str]]:
    """uid -> (record, sha256 of the exact bytes on disk).

    The digest is what makes the declaration a statement about a RECORD rather
    than about a filename. Hashing all 45,955 costs ~0.3 s.
    """
    out = {}
    for sc in sorted(paths.ANNOTATIONS.glob("*.json")):
        try:
            raw = sc.read_bytes()
        except OSError:
            out[sc.stem] = ({}, "")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            rec = {}
        out[sc.stem] = (rec if isinstance(rec, dict) else {}, digest)
    return out


def classify_for_declaration(records: dict[str, dict]) -> tuple[dict[str, list[str]], list[str]]:
    """Partition the corpus, refusing anything the declaration cannot justify.

    Returns ``(populations, undeclarable)``. A uid lands in `undeclarable` when
    no rule covers it -- the tool then refuses to write, because a registry that
    guesses is the very failure it exists to prevent.
    """
    v3, v1, current, undeclarable = [], [], [], []
    for uid, (rec, _digest) in sorted(records.items()):
        pv = rec.get("prompt_version")
        if (all(k in rec for k in REQUIRED_FIELDS)
                and rec.get("annotation_contract") == annotation_contract_id()):
            current.append(uid)          # self-evidencing; nothing to declare
        elif pv == LEGACY_V3_PROMPT_VERSION and all(k in rec for k in REQUIRED_FIELDS):
            v3.append(uid)
        elif pv == LEGACY_V1_PROMPT_VERSION:
            v1.append(uid)
        else:
            undeclarable.append(uid)
    return {"accepted_legacy_v3": v3,
            "legacy_v1_residual_unresolved": v1,
            "annotated_under_current_contract": current}, undeclarable


def revalidate(records: dict[str, dict], uids: list[str]) -> tuple[int, list[tuple[str, str]]]:
    """Run VALIDATOR_VERSION 2 over the declared v3 population, here and now.

    The registry says these records satisfy today's validator. It says so because
    this ran, not because a prior document asserted it.
    """
    ok, failed = 0, []
    for uid in uids:
        try:
            validate_annotation(records[uid][0])
            ok += 1
        except AnnotationError as exc:
            failed.append((uid, str(exc)[:200]))
    return ok, failed


def declare(path) -> int:
    records = _records()
    pops, undeclarable = classify_for_declaration(records)
    if undeclarable:
        print(f"REFUSING to declare: {len(undeclarable)} record(s) match no declaration "
              f"rule, e.g. {undeclarable[:5]}. A registry that guesses is worse than no "
              f"registry. Resolve them first.")
        return 3

    v1 = set(pops["legacy_v1_residual_unresolved"])
    if v1 != set(DECLARED_V1_RESIDUALS):
        print(f"REFUSING to declare: the prompt_version {LEGACY_V1_PROMPT_VERSION} population "
              f"is {sorted(v1)}, but TASK.md D2a §6 names {sorted(DECLARED_V1_RESIDUALS)}. "
              "A difference here is a MASTER-IMPACTING FINDING, not something to declare over.")
        return 3

    v3 = pops["accepted_legacy_v3"]
    ok, failed = revalidate(records, v3)
    if failed:
        print(f"REFUSING to declare: {len(failed)} of {len(v3)} legacy-v3 records do NOT pass "
              f"VALIDATOR_VERSION {VALIDATOR_VERSION}, e.g. {failed[:3]}. They cannot be "
              "declared as validated under it.")
        return 3

    doc = {
        "registry_version": 1,
        "declared_by": getpass.getuser(),
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "authority": ("workflow/tasks/D2a_stage1-protocol-refresh/TASK.md §7.1, AC-1.c and "
                      "AC-1.e. Provenance: D10 USER_REVIEW.md §7.0."),
        "purpose": ("Declare, explicitly, which population accounts for each stored annotation "
                    "record. `annotate_run` reads this instead of inferring acceptance from a "
                    "missing `annotation_contract` field. A record that is neither declared "
                    "here nor stamped with the current contract is UNACCOUNTED and stops a "
                    "bare run."),
        "current_annotation_contract": annotation_contract_id(),
        "not_declared_here": {
            "state": R.CURRENT_CONTRACT,
            "count": len(pops["annotated_under_current_contract"]),
            "why": ("Self-evidencing: the record carries the current contract id, so it needs "
                    "no declaration. Listed as a count only."),
        },
        "populations": [
            {
                "state": R.ACCEPTED_LEGACY_V3,
                "count": len(v3),
                "prompt_version": LEGACY_V3_PROMPT_VERSION,
                "annotation_contract": None,
                "annotation_contract_note": (
                    "These records were generated before contract stamping existed and carry "
                    "no `annotation_contract`. They MUST NOT be given one: a v4 contract id "
                    "on a v3 record would present legacy output as v4-generated. The absence "
                    "is preserved deliberately; THIS registry, not the absence, is what "
                    "accounts for them."),
                "generated_under_prompt_version": LEGACY_V3_PROMPT_VERSION,
                "revalidated_under_validator_version": VALIDATOR_VERSION,
                "revalidation_evidence": (
                    f"OBSERVED DATA, measured by this tool at declaration time: {ok} of "
                    f"{len(v3)} records pass `validate_annotation()` under VALIDATOR_VERSION "
                    f"{VALIDATOR_VERSION}, 0 failures. Note the distinction: they were ADMITTED "
                    "at generation time by the validator then in force, which had no language "
                    "rule; VALIDATOR_VERSION 2 is applied to them retrospectively and they "
                    "satisfy it."),
                "acceptance": (
                    "Accepted by the user 2026-08-21 as the Stage 1 annotation corpus. No "
                    "re-annotation is authorised."),
                "digest_note": (
                    "`records` maps uid -> sha256 of the exact bytes that were classified "
                    "and re-validated above. A record whose bytes no longer match is "
                    "UNACCOUNTED, so this validation claim cannot silently outlive the "
                    "record it was measured on."),
                "records": {uid: records[uid][1] for uid in v3},
            },
            {
                "state": R.LEGACY_V1_RESIDUAL,
                "count": len(pops["legacy_v1_residual_unresolved"]),
                "prompt_version": LEGACY_V1_PROMPT_VERSION,
                "annotation_contract": None,
                "decision_status": (
                    "D0-003 is UNRESOLVED. These 3 records carry the v1 schema "
                    "(`dimensions.{length_m,width_m,height_m}`, `placement_constraints`) and "
                    "are NOT part of the accepted legacy-v3 corpus, NOT migrated, and NOT "
                    "validated under VALIDATOR_VERSION 2. Declaring them here records only "
                    "that they exist and are not to be touched; it decides nothing. Any "
                    "migration is a user decision under D0-003."),
                "digest_note": (
                    "`records` maps uid -> sha256 of the exact bytes declared. A "
                    "record whose bytes no longer match is UNACCOUNTED, so the "
                    "declaration cannot outlive the record it describes."),
                "why_declared_rather_than_left_undeclared": (
                    "Leaving them out would make them UNACCOUNTED, which blocks every bare "
                    "run. Declaring their unresolved state protects them without resolving "
                    "them, and without mutating a single record."),
                "records": {uid: records[uid][1] for uid in sorted(v1)},
            },
        ],
    }

    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    print(f"declared -> {path}")
    for pop in doc["populations"]:
        print(f"  {pop['state']:<34} {len(pop['records']):>7,}  "
              f"prompt_version {pop['prompt_version']}")
    print(f"  {R.CURRENT_CONTRACT:<34} "
          f"{doc['not_declared_here']['count']:>7,}  (self-evidencing, not declared)")
    return 0


def audit(path) -> int:
    """Re-check the registry, then exercise annotate_run's REAL bare-run predicate."""
    registry = R.load_provenance_registry(path)
    renders = {json.loads(l)["uid"]
               for l in (paths.LOGS / "renders_index.jsonl").read_text().splitlines()
               if l.strip()}

    # `main()`'s own work-list call, verbatim, with force off. `main()` chooses
    # what to annotate through this function and nothing else, so this is the
    # real predicate rather than a re-implementation of it. No model is
    # constructed to get here and CUDA is never initialised.
    todo, blocked, states = R.build_work_list(sorted(renders), force=False,
                                              registry=registry)

    v1 = {u for u, d in registry.items() if d[0] == R.LEGACY_V1_RESIDUAL}
    v3 = {u for u, d in registry.items() if d[0] == R.ACCEPTED_LEGACY_V3}

    print(f"registry                               {path}")
    print(f"current annotation contract            {annotation_contract_id()}")
    print(f"rendered assets                        {len(renders):>7,}")
    print(f"bare annotate_run todo (no --force)    {len(todo):>7,}")
    print(f"  accepted legacy-v3 queued            "
          f"{len([u for u in todo if u in v3]):>7,}")
    print(f"  legacy-v1 residuals queued           "
          f"{len([u for u in todo if u in v1]):>7,}")
    print(f"  unaccounted (would REFUSE the run)   {len(blocked):>7,}")
    for state in (R.CURRENT_CONTRACT, R.ACCEPTED_LEGACY_V3, R.LEGACY_V1_RESIDUAL):
        print(f"  {state:<34} {sum(1 for st in states.values() if st == state):>7,}")

    problems = []
    if todo:
        problems.append(f"AC-1.a FAILS: {len(todo)} record(s) would be queued by a bare run")
    if blocked:
        problems.append(f"{len(blocked)} record(s) are UNACCOUNTED; a bare run refuses to start")
    if v3 & set(DECLARED_V1_RESIDUALS):
        problems.append("AC-1.e FAILS: a legacy-v1 residual is declared as legacy-v3")
    if v1 != set(DECLARED_V1_RESIDUALS):
        problems.append(f"the declared v1 residual set {sorted(v1)} is not the contract's "
                        f"{sorted(DECLARED_V1_RESIDUALS)}")

    # AC-1.b: the capability the gate must NOT have removed -- same function, same
    # inputs, force on. Computing the list only; nothing is annotated.
    forced, _, _ = R.build_work_list(sorted(renders), force=True, registry=registry)
    print(f"\nwith explicit --force                  {len(forced):>7,}  (AC-1.b capability intact)")
    if len(forced) != len(renders):
        problems.append("AC-1.b FAILS: --force no longer reaches every record")

    for p in problems:
        print(f"FAIL  {p}")
    print("\nAC-1 audit: " + ("FAILED" if problems else "PASSED"))
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--declare", action="store_true",
                    help="build the registry from the corpus and write it")
    ap.add_argument("--registry", type=Path, default=R.PROVENANCE_REGISTRY)
    args = ap.parse_args()
    return declare(args.registry) if args.declare else audit(args.registry)


if __name__ == "__main__":
    raise SystemExit(main())
