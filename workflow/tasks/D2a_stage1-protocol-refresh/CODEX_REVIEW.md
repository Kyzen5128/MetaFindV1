# Codex Independent Review

> Independent review record for one formal D-task.
> Codex is a reviewer, not project authority.
> Claude must verify material findings before they affect research-significant work.

---

## Task ID

`D2a_stage1-protocol-refresh`

---

## Review Target

- `TASK.md` (§7.1 `AC-1`, §8 R-A…R-D, §14 attack list)
- the `AC-1` gate in `metafind/data/annotate_run.py` (the whole of the new code, inline)
- `main()`'s work-list construction and the annotation write path
- `tools/declare_annotation_provenance.py`
- the registry format `data/outputs/annotation_provenance.json`
- the C-001 τ change in `metafind/models/resolve_stage1.py`
- the new tests

Excluded from review scope: D10's serializer, `encode_text_image.py`, the pre-flight gate,
`essgnn.py` (D0-009's scope wall).

---

## Review Mode

`ADVERSARIAL REVIEW`

Two rounds, per `codex-reviewer` skill: round 1 attack, Claude verified and fixed, round 2
targeted re-review of the fixes.

---

## Review Status

`COMPLETED`

**Post-remediation note, 2026-08-21.** After this review closed, the user returned three decisions
(registry route approved · τ wording corrected in the contract · a narrow `docs/graph/README.md`
line-270 exception granted). The remediation changed **one code comment, one test docstring, and
one README line** — no reviewed logic. The verification state below is re-run and current;
`check_graph.py` now reports **2275 checks, all pass**, and the suite is **547 passed**. Findings
F-4 and F-5 are updated in §6A with their dispositions.

**Operational note, disclosed for honesty.** Three earlier invocations
(`codex exec` backgrounded with `nohup`) read the files and then exited with code 0 **without
emitting a findings report**. Those runs produced no review and were not counted as one. The
review below comes from foreground invocations with the code passed inline. Reviewer:
`codex-cli 0.148.0`, model `gpt-5.6-sol`, `--sandbox read-only`.

---

## 1. Review Brief Given to Codex

### Objective

Falsify the claim that, absent explicit `--force`, a bare
`python -m metafind.data.annotate_run` queues **zero** existing annotation records — while
re-annotation stays reachable, the three populations stay distinguishable, and no annotation
record is mutated.

### Claimed Behavior

- bare run: `todo = 0`, `unaccounted = 0`, over 45,955 rendered assets
- `--force`: 45,955 (capability intact)
- three states explicit in the record (`annotated_under_current_contract`) or in a declared
  registry (`accepted_legacy_v3`, `legacy_v1_residual_unresolved`)
- no model load, no GPU, no record mutated
- `init_temperature` 0.07 → 0.5, `learnable_temperature` True → False, nothing else changed

### Relevant Evidence

Corpus fingerprint, population counts, the emitted artifacts, the tool's audit output, and the
full gate source were supplied inline.

### Changes / Result

See `HANDOFF.md` §Changes.

### Known Uncertainty

Disclosed to Codex up front: the registry binds `(uid, prompt_version)` and not record content
(round 1); the threat model is **accident**, not an adversary with write access to
`data/outputs/`.

### Review Request

"Do not confirm. Find what is wrong." Codex was given the eight attack vectors of `TASK.md`
§14 plus TOCTOU, symlinks, `METAFIND_DATA` repointing, path traversal and duplicate uids.

---

## 2. Codex Original Findings

Round 1 produced 15 findings. The material ones are preserved below; the remainder were
restatements of these or explicit non-findings.

### Finding 1 — registry may bind a residual to the accepted population

Severity: `BLOCKER`

Finding: `load_provenance_registry()` accepted any `(state, prompt_version)` pair. Nothing
bound `accepted_legacy_v3` to prompt_version 3.

Evidence / reasoning: a registry declaring
`{"state": "accepted_legacy_v3", "prompt_version": 1, "uids": ["6c7db00c…"]}` loads without
complaint, labelling a legacy-v1 residual as legacy-v3 — the exact conflation `AC-1.e` forbids,
achieved without touching a record.

Suggested action: bind each declarable state to its schema generation.

### Finding 2 — a uid declared twice is silently last-write-wins

Severity: `MAJOR`

Finding: `out[uid] = (state, pv)` overwrites. Listing a residual under
`legacy_v1_residual_unresolved` and again under `accepted_legacy_v3` silently yields the latter.

### Finding 3 — a sidecar containing JSON `null` is treated as ABSENT and queued

Severity: `MAJOR` (Codex filed under A8 as a type bug; its real consequence is an `AC-1.a` hole)

Finding: `_record()` returned `json.loads(...)` unguarded. `json.loads("null")` is `None`,
which `provenance_state()` reads as "no record" — so an existing file re-entered the work queue.

### Finding 4 — same-`prompt_version` content substitution inherits the declaration

Severity: `BLOCKER`

Finding: the declaration binds `(uid, prompt_version)` only. Replacing a declared legacy-v3
record with `{"prompt_version": 3}` still clears it. "No record identity, digest, required
fields, schema, or content is authenticated."

### Finding 5 — a corrupt registry crashes instead of refusing

Severity: `MAJOR`

Finding: truncated JSON, a missing `populations` key, or a scalar document raise uncaught
exceptions rather than the declared refusal path with exit code 3.

### Finding 6 — check/use race between classification and write

Severity: `BLOCKER`

Finding: a uid classified as absent can gain a record before `tmp.replace(sc)` overwrites it.

### Finding 7 — `_under_current_contract()` is checked before the registry

Severity: `MAJOR`

Finding: a record carrying the current contract id overrides an explicit unresolved
declaration, "so the registry is not authoritative over the declared state".

### Finding 8 — path traversal in a uid

Severity: `BLOCKER`

Finding: `sidecar_path()` does not validate the uid; `../` components escape
`paths.ANNOTATIONS`.

### Finding 9 — `AC-1.c` is not literally met

Severity: `MAJOR`

Finding: `CURRENT_CONTRACT` is expressly *forbidden* in the registry and inferred from record
content, so "all three states explicit in the declared registry" does not hold.

### Finding 10 — duplicate JSON keys collapse

Severity: `MINOR`

Finding: duplicate `"populations"` keys, or duplicate uid keys inside one `records` object,
collapse last-write-wins in `json.loads`.

### Finding 11 — `prompt_version` type laxity

Severity: `MINOR`

Finding: `True == 1` and `1.0 == 1` in Python, so a bool or float passes the schema binding.

### Finding 12 — a non-object population entry crashes

Severity: `MAJOR` (round 2, newly introduced by the round-1 fix)

Finding: `{"populations": [null]}` raises an uncaught `AttributeError` on `pop.get("state")`.

### Finding 13 — the "whole work-list decision" guarantee is comment-only

Severity: `NIT`

Finding: nothing structurally prevents a future `todo.append(...)` after `build_work_list()`.

### Finding 14 — duplicate uids under `--force` are annotated twice

Severity: `MINOR`

Finding: `--uids-file` with a repeated uid re-annotates it twice under `--force`.

---

## 3. Claude Verification

### Finding 1 — `CONFIRMED`

Verification performed: constructed the registry Codex describes and loaded it.
Result: `{'6c7db00cc164467ebac356a5ca67368b': ('accepted_legacy_v3', 1)}` — accepted, exactly as
claimed.

Higher-authority evidence checked: `TASK.md` §7.1 `AC-1.e` — "The 3 residuals must not be
labelled legacy-v3". The defect defeats a binding sub-condition.

Conclusion: real, and material. Fixed.

### Finding 2 — `CONFIRMED`

Verification performed: two populations declaring `"dup"`; loader returned
`{'dup': ('accepted_legacy_v3', 3)}`. Last-write-wins reproduced.

Conclusion: real. Fixed.

### Finding 3 — `CONFIRMED`, and more serious than Codex graded it

Verification performed: wrote `null` into a sidecar.
`_record()` → `None`; `provenance_state()` → `None`; `build_work_list(force=False)` → **queued**.

Conclusion: this is a genuine hole in `AC-1.a` — the one claim the task exists to establish —
reachable by a corrupt-but-parseable file with no adversary involved. Fixed, and given its own
regression test.

### Finding 4 — `CONFIRMED`

Verification performed: reasoned from the code and reproduced after the fix (see §4).
Note: Claude had independently identified this limitation before the review and had provisionally
classified it out of scope. Codex's framing changed that judgement — see §4.

Conclusion: real. Fixed by binding the declaration to a sha256 of the record.

### Finding 5 — `CONFIRMED`

Verification performed: `'{"populations": ['`, `'[]'`, `'"just a string"'`, `'{"nope": 1}'` all
raised uncaught `JSONDecodeError`/`KeyError`/`TypeError`.

Note on severity: every one of these fails **closed** — nothing is annotated. So this was a
diagnosability defect, not a safety hole. Fixed anyway, because a mechanism whose failure mode is
a traceback invites someone to delete the file to "fix" it.

### Finding 6 — `CONFIRMED`

Verification performed: read the write path. The window is real.

Scope note: it cannot touch any of the 45,955 — all of them exist, so none is in `todo` — and it
requires two concurrent `annotate_run` processes, which this pipeline does not run. Fixed anyway
(data-loss class), atomically rather than with a re-check.

### Finding 7 — `CONFIRMED` as behaviour, `REJECTED` as a defect

Verification performed: the ordering is as described.

Reasoning: the ordering is **correct for the legitimate case**. If `D0-003` is ever resolved and a
residual is re-annotated through a named migration, that record will carry the current contract
while an older registry entry still names it as a residual. The record is the newer fact and must
win. The attack case requires write access to `data/outputs/annotations/` — an actor who has that
already has `--force`. It never queues work in either case.

Codex round 2, when given this reasoning: `ACCEPT`.

### Finding 8 — `CONFIRMED` as behaviour, `REJECTED` as in-scope for D2a

Verification performed: `sidecar_path()` does concatenate lexically.

Reasoning: `sidecar_path()` is **unchanged by this task** — it is pre-existing behaviour, and
uids come from a project-generated render index (all 45,955 verified as 32-character hex).
`TASK.md` §9 authorises `annotate_run.py` for "work-list / completion semantics", not for
broadening input validation. Fixing it here would be an unrequested scope expansion.

Codex round 2: `ACCEPT WITH RESIDUAL RISK` — "the safety claim depends entirely on trusted
render-index generation". Claude agrees with that caveat and records it as a finding for Master.

### Finding 9 — `REJECTED`

Verification performed: read `AC-1.c` verbatim in `TASK.md` §7.1.

Higher-authority evidence: `AC-1.c` reads "explicit **in the record or in a declared registry**".
`annotated_under_current_contract` is explicit *in the record* — the record literally carries the
`annotation_contract` field. The other two are in the registry. The "or" is satisfied by the split.

Also decisive: the alternative Codex assumed — all three declarable in the registry — would require
either stamping a v4 contract id onto v3 records (forbidden by R-A.1) or mutating them (forbidden
by §9.3). Codex was reasoning from D10 §7.0's *earlier* wording, which says only "explicit in the
record". See `HANDOFF.md` FIND-1.

Codex round 2, when given the verbatim text: `ACCEPT`.

### Finding 10 — `CONFIRMED` as behaviour, `REJECTED` as a defect

Verification performed: duplicate-key collapse is standard `json.loads` behaviour.

Reasoning: both collapse directions **fail closed**. A collapsed `populations` key yielding `[]`
produces an empty registry → every record `UNACCOUNTED` → the run refuses. A collapsed uid key
inside `records` loses a declaration → that uid becomes `UNACCOUNTED` → the run refuses. Neither
queues work. Closing it would need a custom JSON parser, which is disproportionate.

### Finding 11 — `CONFIRMED`

Verification performed: `True == 1` and `1.0 == 1` both hold; both passed the binding. Fixed with
`type(pv) is not int`.

### Finding 12 — `CONFIRMED`

Verification performed: `{"populations": [null]}` raised `AttributeError`. This was introduced by
the round-1 hardening, and round 2 caught it — which is the point of a second round. Fixed.

### Finding 13 — `CONFIRMED`, no change

Verification performed: true as stated. `main()` calls `build_work_list()` and does nothing else
to choose work; `--limit` can only slice. The guarantee is pinned by tests rather than by
structure. Accepted as a documentation-level observation.

### Finding 14 — `CONFIRMED`, no change

Verification performed: true. Re-annotating the same uid twice under explicit `--force` is
idempotent (the second write produces the same record) and is pre-existing `--uids-file`
behaviour. `NIT`.

---

## 4. Resulting Changes

Changes made because of CONFIRMED findings — all in `metafind/data/annotate_run.py` unless noted:

- **F1** `STATE_PROMPT_VERSION` binds each declarable state to its schema generation;
  `load_provenance_registry()` refuses a state declared at the wrong `prompt_version`.
  *(Finding 1)*
- **F2** a uid declared twice raises `ProvenanceRegistryError` instead of overwriting.
  *(Finding 2)*
- **F3** `_record()` guards on `isinstance(rec, dict)`, so JSON `null`/`[]`/`"x"`/`42` is a
  **present but unreadable** record, never an absent one. *(Finding 3)*
- **F4** the registry now stores `records: {uid: sha256}` and a declaration is honoured only while
  the record's bytes still hash to what was declared. Also changes
  `tools/declare_annotation_provenance.py` to emit digests. Measured cost: 0.28 s to hash all
  45,955; registry 1.8 MB → 5.0 MB. *(Finding 4)*
- **F5** `ProvenanceRegistryError` replaces every raw exception; `main()` catches it and returns
  exit code 3 with a refusal message. *(Finding 5)*
- **F6** the non-force write path uses `os.link()`, which refuses to clobber, making the
  check and the create one atomic step. `--force` keeps `tmp.replace()`. *(Finding 6)*
- **F7** a non-object `populations` entry raises `ProvenanceRegistryError`. *(Finding 12)*
- **F8** `type(pv) is not int` rejects `True` and `1.0` as schema generations. *(Finding 11)*

Test coverage added for every one of F1…F5 and F8 (`tests/test_annotate.py`). Suite: 525 → 547
(435 `def test_` functions). Re-run after the remediation: **547 passed, 0 failed**;
`check_graph.py` **2275 checks, all pass**; `preflight_stage1_text.py` **PRE-FLIGHT PASSED**;
annotation corpus checksum `30b2737c95152043762ce25fcabe7a0e`, equal to baseline.

**One change was made from Claude's own audit, before Codex reported:** `build_work_list()` was
extracted so the AC-1.b tests exercise `main()`'s real force branch instead of re-typing
`list(candidates)`. Codex round 1's attack #10 named this exact weakness independently.

---

## 5. Findings Not Adopted

### REJECTED

- **Finding 7** — `_under_current_contract()` before the registry. Correct for the legitimate
  named-migration case; never queues work. Codex accepted on re-review.
- **Finding 8** — uid path traversal. Real, pre-existing, and outside `TASK.md` §9's authorised
  scope for this task. Escalated to Master instead (`HANDOFF.md` FIND-4).
- **Finding 9** — `AC-1.c` not literally met. Contradicted by the verbatim text of `TASK.md`
  §7.1, which is the authoritative execution contract. Codex accepted on re-review.
- **Finding 10** — duplicate JSON keys. Both collapse directions fail closed.

### PLAUSIBLE

`None.` Every material finding was resolved to `CONFIRMED` or `REJECTED` by direct execution.

### UNVERIFIED

`None.`

---

## 6. Remaining Disagreement

`None.` Codex's round-2 re-review confirmed all six fixes closed
(C-1, C-2, C-4, C-7, C-8) and accepted all three deferral rationales, one of them
(`sidecar_path()` traversal) `WITH RESIDUAL RISK` — a caveat Claude agrees with and has carried
into the HANDOFF as FIND-4.

---

## 6A. Material Finding Traceability

| ID | Claim attacked | Evidence | Codex finding | Claude verification | Impact | Decision implication |
|---|---|---|---|---|---|---|
| F-1 | "a bare run queues 0 records TOTAL" | `annotate_run.py` `_record()`; reproduced by writing `null` into a sidecar | a parseable-but-non-object sidecar reads as ABSENT and is queued | `CONFIRMED` | `AC-1.a` | `None` — fixed within scope, regression test added |
| F-2 | "the 3 residuals cannot be labelled legacy-v3" | `load_provenance_registry()`; reproduced with a hand-built registry | the registry could bind a residual to `accepted_legacy_v3` | `CONFIRMED` | `AC-1.e`, `D0-003` | `None` — fixed within scope |
| F-3 | "the declaration accounts for these records" | registry keyed on `(uid, prompt_version)` | a same-version content substitution silently inherits the "validated under VALIDATOR_VERSION 2" claim | `CONFIRMED` | `AC-1.c`, `AC-1.e` provenance integrity | `None` — fixed by digest binding |
| F-4 | "uids are safe to path-join" | `sidecar_path()`, unchanged by this task | `../` in a uid escapes `paths.ANNOTATIONS` | `CONFIRMED` as behaviour; `REJECTED` as in-scope | `annotate_run` input validation, all populations | **Master decision required**: pre-existing defect, outside D2a's §9 scope. Claude did not fix it |
| F-5 | "`AC-1.c` requires all three states in the registry" | `TASK.md` §7.1 vs `D10/USER_REVIEW.md` §7.0 | the implementation does not put all three in the registry | `REJECTED` | `AC-1.c` reading | **RESOLVED 2026-08-21.** The user confirmed §7.1 is authoritative and **approved the declared-registry route**; the registry path is now named in `TASK.md` §9.2. §7.0's narrower wording was stale and Master corrected it. Claude's rejection of this finding stands |
| F-6 | "no existing record is overwritten" | classify→write window | TOCTOU race | `CONFIRMED` | `AC-1.a` under concurrency | `None` — closed atomically with `os.link()` |

**A Codex finding is not a decision.** Rows F-4 and F-5 carry decision implications and appear in
`HANDOFF.md` §15 `USER REVIEW INPUT`. **F-5 has since been decided by the user (2026-08-21):
rejection upheld, registry route approved. F-4 remains open for Master** — the pre-existing
`sidecar_path()` traversal was not fixed by this task and no decision on it has been returned.

---

## 7. Final Review Outcome

`PASS WITH FOLLOW-UP`

The review found eight confirmed defects, two of them genuine holes in the `AC-1.a` claim
(Findings 3 and 6). All eight are fixed and covered by regression tests; the round-2 re-review
confirmed each closure and introduced one new finding (12), which was also fixed.

One item remains for someone other than this task:

- **F-4**, uid path traversal in `sidecar_path()` — pre-existing, outside D2a's authorised scope,
  **still undecided**.

**F-5** (the `AC-1.c` wording difference) was resolved by the user on 2026-08-21 in favour of
`TASK.md` §7.1, upholding Claude's rejection of the finding and approving the registry route.

`PASS` here describes the **review**. It accepts, approves and finalises nothing.
