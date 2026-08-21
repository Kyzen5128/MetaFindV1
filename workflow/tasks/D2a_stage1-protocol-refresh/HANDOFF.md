# D-Task Handoff

> Execution record returned to Master.
> Findings and decisions are stated separately (`WORKFLOW.md` §13A).
> This task does not accept itself, does not mark D10 accepted, and does not unblock D1.

---

## Task ID

`D2a_stage1-protocol-refresh`

---

## Status

`COMPLETE — AWAITING MASTER INTEGRATION REVIEW`

Executed 2026-08-21, then revised the same day under a **USER DECISION carrying three
remediations** (registry route approved · τ wording corrected · a narrow `docs/graph/README.md`
scope exception granted).

**All 14 Definition-of-Done items are now fully met.** The one failing check reported in the first
return — `check_graph.py`'s README unit-test counter — is resolved: the user granted a narrow
exception, the edit was made, and `check_graph.py` now reports **2275 checks, all pass**.

Every figure in this document is **post-remediation**. Nothing here quotes a pre-edit count.

---

## 1. Objective Result

`load_protocol()` no longer refuses: the Stage 1 protocol artifacts were re-emitted through n05b
under τ = 0.5 and the ratified serializer, and the existing annotation corpus is now protected
from accidental re-annotation.

Measured, on the real corpus:

```
                                    before D2a      after D2a
rendered assets                         45,955         45,955
bare annotate_run todo (no --force)     45,955              0
  accepted legacy-v3 queued             45,952              0
  legacy-v1 residuals queued                 3              0
  unaccounted (run refuses, exit 3)          -              0
with explicit --force                   45,955         45,955
```

The "before" row is not quoted from `TASK.md` §6; it was recomputed here with the pre-D2a
predicate (`[u for u in sorted(renders) if not R.is_complete(u)]`) and independently reproduced
§6's three numbers exactly.

---

## 2. Scope Compliance

### Completed In Scope

- **C-001** — τ = 0.5 is reachable through n05b and is now its default.
- **C-002** — n05b re-emitted both protocol artifacts in one call.
- **AC-1.a…e** — mechanism built, demonstrated, and adversarially reviewed twice.
- **AC-1.e provenance** — formalised in a declared registry.

### Explicit Non-Scope Respected

Stated explicitly, as `TASK.md` §16 requires:

**No n06 run. No n09 run. No training. No gallery indexing. No GPU embedding generation. No
annotation mutation of any kind — including the 3 legacy-v1 residuals.**

- `data/outputs/annotations/**` — all 45,955 records **byte-identical** to task start.
  `md5sum` of the sorted per-file `md5sum` list: `30b2737c95152043762ce25fcabe7a0e` at start and
  at finish.
- `data/outputs/embeddings/` — still 5,276 `.npz`, untouched, not deleted.
- `data/outputs/checkpoints/` — still empty.
- `data/outputs/logs/annotations_index.jsonl` — **not rebuilt.** `main()` returns at
  `if not todo: return 0`, before `rebuild_index()` is reached.
- `D0-002` and `D0-003` — not decided, not touched.
- No v4 re-annotation of any population.
- The ratified template and every serialized string — unchanged.
- `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md`, `workflow/decisions/**` —
  not touched.
- **D0-009 scope wall** — `metafind/models/essgnn.py`, the equivariance tests, `docs/audit/` and
  `docs/graph/` were not modified. One incidental observation touching D0-009's area is reported
  as MIF-3 and **was not acted on**.

### Scope Deviations

**None.** The one item first reported as a scope conflict is now authorised.

- `docs/graph/README.md` line 270 was edited (413 → 435, 525 → 547) under the **narrow scope
  exception the user granted on 2026-08-21**, now written into `TASK.md` §9.1. Verified to touch
  **exactly one line** (`diff` reports a single `270c270` hunk). No other file under `docs/**` was
  read for modification or written; `docs/audit/**`, the rest of `docs/graph/`, and `docs/paper/**`
  remain untouched.
- On the first return this was escalated, not edited, because §9.3 then protected it. That
  escalation is retained as MIF-1 with a `RESOLVED` disposition, so the record shows why the file
  changed.

One artifact was written that §9.2 does not list — unavoidably, as a consequence of the run the
contract mandates. See §12 MIF-2. Its content is byte-identical to before.

**No D10 file was modified by this task**, as `AC-1.c` requires. `D10/USER_REVIEW.md` shows a
later mtime (11:45) because **Master** corrected its stale §7.0 wording; this executor did not
open it for writing.

---

## 3. Files Changed

Verified by mtime against session start; `git status` alone is misleading here because the working
tree already carried D10's uncommitted changes before this task began.

| Path | §9 basis | What changed |
|---|---|---|
| `metafind/data/annotate_run.py` | §9.1 | the AC-1 provenance gate; work-list construction; atomic non-force write |
| `metafind/models/resolve_stage1.py` | §9.1 | C-001: `DEFAULT_HYPERPARAMETERS` τ |
| `tests/test_annotate.py` | §9.1 | 18 new tests for AC-1 and the Codex findings |
| `tests/test_resolve_stage1.py` | §9.1 | 4 new tests for τ |
| `tools/declare_annotation_provenance.py` | §9.1 (new file under `tools/`) | declares and re-checks the registry; the runnable AC-1 proof |
| `docs/graph/README.md` | §9.1 **narrow scope exception**, user-approved 2026-08-21 | **line 270 only** — stale test counts 413 → 435 and 525 → 547. Documentation only |
| `workflow/tasks/D2a_stage1-protocol-refresh/TASK.md` | §9.2 | status line only, `READY` → `ACTIVE` |
| `workflow/tasks/D2a_stage1-protocol-refresh/HANDOFF.md`, `CODEX_REVIEW.md` | §9.2 | this handoff and the review record |

**Not touched** (pre-session mtimes, D10's work): `metafind/data/annotate.py` (03:36),
`metafind/data/encode_text_image.py` (03:38), `tests/test_encode_text_image.py` (03:39),
`tools/preflight_stage1_text.py` (03:37) — the last of these is D10's gate, which had to keep
passing **unmodified**, and did. `annotate.py` was *authorised* by §9.1 but proved unnecessary:
the registry lives in `annotate_run.py`, which already owns the filesystem, leaving `annotate.py`
pure and import-free of `paths`.

---

## 4. Artifacts Produced

| Path | §9 basis | Note |
|---|---|---|
| `data/outputs/stage1_hyperparameters.json` | §9.2 | rewritten by n05b. `init_temperature: 0.5`, `learnable_temperature: false` |
| `data/outputs/stage1_encoding_protocol.json` | §9.2 | rewritten by n05b. `metafind_v2_cm@8e4b1fcc66c7f48c` |
| `data/outputs/annotation_provenance.json` | §9.2 (provenance registry) | new, 5.0 MB, declares all 45,955 uids by population with a sha256 each |
| `data/outputs/protocol_backup_pre_D2a_20260821/` | §9.2 (backup location, §10.3) | pre-run copies of both protocol artifacts, plus `variant_registry.json` and its md5 |
| `data/outputs/variant_registry.json` | **not listed in §9.2** | rewritten by n05b as an unavoidable side effect; **byte-identical**, md5 `5c84f5be5738c0ab374cb91568bddcca` before and after. See MIF-2 |

n05b command, exactly as executed:

```
python -m metafind.models.resolve_stage1 \
  --paper-clip-train-scope frozen --actual-clip-train-scope frozen --confidence moderate \
  --decided-by "D2a_stage1-protocol-refresh (Claude, executing). Authority: Kyzen -- tau=0.5 ratified 2026-08-21, text template D0-008 USER_APPROVED 2026-08-21; CLIP train scope carried forward unchanged from the 2026-08-16 resolution."
```

`--paper-clip-train-scope frozen`, `--actual-clip-train-scope frozen` and `--confidence moderate`
were read off the pre-existing artifact and carried forward deliberately. They are **not** this
task's decisions; changing them silently would have been a research-significant edit outside scope.

`decided_by` names the executing agent and the authority separately, so the record does not credit
the approver with the work.

**What n05b changed, field by field** — `stage1_hyperparameters.json`:

| | before | after |
|---|---|---|
| `values.init_temperature` | `0.07` | `0.5` |
| `values.learnable_temperature` | `true` | `false` |
| the other 9 `values` keys | — | **identical**, no key added or removed |
| `sha256`, `decided_by`, `decided_at` | — | changed, as they must |

`stage1_encoding_protocol.json`: `text_serialization`, `text_template`, `decided_by`,
`decided_at` changed; `text_serialization_contract`, `text_serialization_family`,
`text_serialization_probes` were **added** (D10 introduced them after the old artifact was
written); `image_aggregation`, `paper_clip_train_scope`, its basis text, `confidence`,
`actual_clip_train_scope`, `missing_modality_representation` all unchanged. Nothing else moved.

---

## 5. Evidence Used

| Claim | Class | Source |
|---|---|---|
| "The temperature is 0.5 for all experiments." | **PAPER FACT** | `docs/paper/metafind_source/3experiments.tex:15`, read directly this session |
| the paper never calls τ learnable | **PAPER FACT as to silence** | `learnable` occurs at `2methdology.tex:54` and `:87` only; τ is named "a temperature **hyperparameter**" at `:79` and `:99`. All four lines read directly this session |
| `learnable_temperature = false` | **USER-RATIFIED IMPLEMENTATION CHOICE** on a strongly-supported INFERENCE | `TASK.md` §7 C-001, user ruling 2026-08-21 |
| 45,952 v3 + 3 v1 = 45,955 | **OBSERVED DATA** | recounted this session; `prompt_version` histogram `{3: 45952, 1: 3}`; the pv==1 set equals the three uids §6 names |
| 0 records carry an `annotation_contract` | **OBSERVED DATA** | histogram `{None: 45955}` |
| all 45,952 v3 records pass `VALIDATOR_VERSION 2` | **OBSERVED DATA**, measured here | 45,952 / 45,952 through `validate_annotation()`, 0 failures |
| the pre-D2a bare run queued 45,955 | **OBSERVED DATA**, reproduced here | pre-change predicate re-run against the real corpus |
| serializer identity `metafind_v2_cm@8e4b1fcc66c7f48c` | **OBSERVED IMPLEMENTATION** | emitted artifact; matches `TASK.md` §6 |
| the registry mechanism | **IMPLEMENTATION CHOICE** | this task, per R-D |

**The τ inference, in its ratified form.** An earlier draft of `TASK.md` §7 said *"The paper never
uses the word 'learnable'."* That was false, this task reported it, and the **user corrected the
contract on 2026-08-21**. The ratified argument, re-verified here against the primary source:

| line | what the paper says |
|---|---|
| `2methdology.tex:54` | `f_h`, `f_x` are "two **learnable** functions" |
| `2methdology.tex:87` | λ is "a **learnable** scalar" |
| `2methdology.tex:79` | τ is "a temperature **hyperparameter**" |
| `2methdology.tex:99` | τ is "a temperature **hyperparameter**" |
| `3experiments.tex:15` | "The temperature is 0.5 for all experiments." |

The authors distinguish learnable quantities from hyperparameters **in their own vocabulary**, and
place τ on the hyperparameter side **twice**. With a value fixed "for all experiments", that is a
strong reading — and it remains an **INFERENCE**. The paper nowhere states that τ is non-learnable.

`learnable_temperature: false` is therefore a **USER-RATIFIED IMPLEMENTATION CHOICE** and is
recorded as one in `resolve_stage1.py`'s C-001 comment and in the τ tests. The false sentence never
appeared in any code, test, or artifact this task produced — verified by grep. See FIND-2.

---

## 6. Decisions Made Within Task Scope

Per R-D, the mechanism is an `IMPLEMENTATION CHOICE`. Classified and argued:

**A declared registry, not a record edit.** `data/outputs/annotation_provenance.json` enumerates
every uid with its population and a sha256 of its bytes. `annotate_run` skips a record only when
the registry names it, or when the record itself carries the current contract id.

Why this and not the alternatives:

| Alternative | Rejected because |
|---|---|
| stamp a v4 contract id on the v3 records | R-A.1 forbids it; they did not earn it; and §9.3 requires all 45,955 to stay byte-identical |
| treat "missing `annotation_contract`" as "accepted" | R-A.2 forbids it. Absence is what created this hazard; it may not be what clears it |
| mark the 3 residuals in-record | R-B: they are under an unresolved decision. This mechanism mutates **nothing** |
| remove or narrow `--force` | R-C forbids it. `--force` is untouched |
| skip any record whose file exists | loses the half-written-record guard that `is_complete()` existed for, and cannot tell the three populations apart |

Satisfaction of each sub-condition:

- **AC-1.a** — 0 total. Not 0 legacy-v3: 0 legacy-v3 **and** 0 residuals **and** 0 unaccounted.
- **AC-1.b** — `--force` reaches all 45,955; `--uids-file <list> --force` is the named-migration
  form, reaching exactly the uids named. Both tested through the same `build_work_list()` that
  `main()` calls.
- **AC-1.c** — `annotated_under_current_contract` is explicit **in the record** (its
  `annotation_contract` field); `accepted_legacy_v3` and `legacy_v1_residual_unresolved` are
  explicit **in the declared registry**. Nothing is inferred from a missing field: an existing
  record that is neither declared nor stamped is `UNACCOUNTED`, which **stops the run** (exit 3).
- **AC-1.d** — proved via `tools/declare_annotation_provenance.py` and the direct
  `build_work_list()` call. No model constructed; `torch.cuda.is_initialized()` → `False`.
- **AC-1.e** — the registry records the **45,952 only** as `accepted_legacy_v3`,
  `generated_under_prompt_version: 3`, `revalidated_under_validator_version: 2`, with the
  measurement that established it. `annotation_contract` is explicitly `null` with a note saying
  it must stay that way. The 3 residuals are a separate population whose `decision_status` opens
  "D0-003 is UNRESOLVED" and closes "it decides nothing".

**Fail-closed by construction.** Deleting, truncating, corrupting or shortening the registry does
not re-open the queue — it makes records `UNACCOUNTED`, which refuses the run. Verified by
deleting the registry (all `UNACCOUNTED`, `todo` empty) and by repointing `METAFIND_DATA` at a
corpus with no registry (existing records blocked; only a genuinely new uid was work).

**A precision the contract did not require.** The registry's "validated under VALIDATOR_VERSION 2"
claim is bound to a sha256 per record, so it cannot silently outlive the record it was measured
on. Adopted after Codex demonstrated a same-`prompt_version` substitution that inherited the
claim. Cost: 0.28 s to hash all 45,955; registry 1.8 MB → 5.0 MB.

---

## 7. Verification Performed

### Tests

- `python -m pytest tests/ -q`, **no `--ignore`** → **547 passed**, 0 failed. Baseline 525;
  +22 new (18 in `test_annotate.py`, 4 in `test_resolve_stage1.py`).
- `def test_` functions defined in `tests/`: **435**. Both figures now match
  `docs/graph/README.md:270`, which is what `check_graph.py:415` asserts.
- New coverage includes the AC-1.b negative test, a test that the 3 residuals are never
  auto-queued and never labelled legacy-v3, and one regression test per confirmed Codex finding.

### Runtime Checks

- **AC-1.a**, through `main()`'s own `build_work_list()` call: `todo 0`, legacy-v3 `0`,
  residuals `0`, unaccounted `0`, over 45,955 rendered assets. `torch.cuda.is_initialized()`
  → `False`.
- **AC-1.b**: `build_work_list(..., force=True)` → 45,955. Computation only; no annotation run.
- `load_protocol()` called directly → returns, prints `metafind_v2_cm@8e4b1fcc66c7f48c`.
- `tools/preflight_stage1_text.py`, **unmodified** → `PRE-FLIGHT PASSED`; 0 language violations,
  0 template mismatches, 0 zero-dimension renders, 0 over 77 tokens (max 72).
- `tools/check_graph.py` → **2275 checks, all pass.** (On the first return this reported 1 failure;
  the user-approved README edit resolved it.)

### Artifact Checks

- annotations: 45,955 files, corpus fingerprint `30b2737c95152043762ce25fcabe7a0e`, **identical
  to task start**.
- embeddings: 5,276 `.npz`, unchanged. checkpoints: empty.
- `annotations_index.jsonl`: not rebuilt.
- `variant_registry.json`: rewritten byte-identical (md5 unchanged).

### Research Fidelity Verification

- τ = 0.5 traced to `3experiments.tex:15` by reading the file, not by citation.
- The emitted artifact records `0.5` / `false`; the other 9 hyperparameter values are unchanged,
  so no library default reintroduced `0.07`.
- `build_hyperparameters()` output feeds `MetaFindContrastiveLoss` without raising the deviation
  warning at `losses.py:114` — a test asserts this under `warnings.simplefilter("error")`.
- An explicit override to 0.07 still works and produces a different `sha256`, so C-001 made 0.5
  the default without making it the only possibility.
- **The two authority tiers were grepped for conflation.** The code comment states
  `init_temperature = 0.5` as PAPER FACT with the verbatim quote, and `learnable_temperature =
  false` as a USER-RATIFIED IMPLEMENTATION CHOICE resting on an inference, explicitly "NOT a paper
  statement". No wording anywhere in the changed files lets the second be read as the first.
  The emitted artifact carries only values, asserting no authority claim at all.

---

## 8. Verification Result

`PASS`

Everything the task set out to establish is established and independently measured, and every gate
passes. Full post-remediation battery, run in the order `TASK.md` §12.1 specifies:

| Check | Result |
|---|---|
| `pytest tests/ -q` (no `--ignore`) | **547 passed**, 0 failed |
| `tools/check_graph.py` | **2275 checks, all pass** |
| `tools/preflight_stage1_text.py` (unmodified) | **PRE-FLIGHT PASSED** — 0 language violations, 0 template mismatches, 0 zero-dimension renders, 0 over 77 tokens (max 72) |
| annotation corpus checksum | `30b2737c95152043762ce25fcabe7a0e` — **equals baseline** |
| `data/outputs/embeddings/*.npz` | **5,276** |
| `data/outputs/checkpoints/` | **0** |
| AC-1 audit | **PASSED** — bare todo 0, legacy-v3 0, residuals 0, unaccounted 0; `--force` 45,955 |
| `load_protocol()` | returns `metafind_v2_cm@8e4b1fcc66c7f48c` |
| τ artifact | `init_temperature 0.5`, `learnable_temperature false` |
| `git status` / `git diff --stat` | only §9.1 / §9.2 paths; `docs/graph/README.md` shows `2 +-`, i.e. the single line |

The remediation touched only a code comment, a test docstring, and one README line, so the AC-1
and τ measurements are unchanged from the first return — re-run and reconfirmed rather than
carried over.

---

## 9. Codex Review Result

`COMPLETED` — `PASS WITH FOLLOW-UP`. Two rounds. Full record in `CODEX_REVIEW.md`.

Disclosed: three earlier backgrounded `codex exec` invocations read the files and exited without
emitting a report. Those produced no review and were **not** counted as one; the review came from
foreground runs with the code inline.

Codex found **8 confirmed defects**, two of them genuine holes in the AC-1.a claim. All are fixed
with regression tests. Round 2 confirmed each closure, found one newly-introduced defect (also
fixed), and accepted all three of Claude's deferral rationales.

---

## 10. Confirmed Codex Findings

| # | Finding | Fix |
|---|---|---|
| 1 | the registry could bind a legacy-v1 residual to `accepted_legacy_v3` | `STATE_PROMPT_VERSION` binds each state to its schema generation |
| 2 | a uid declared twice was silent last-write-wins | duplicate declaration raises |
| 3 | **a sidecar containing JSON `null` read as ABSENT and was queued** — a real AC-1.a hole | `_record()` guards on `isinstance(rec, dict)` |
| 4 | a same-`prompt_version` content substitution inherited the declaration | registry binds a sha256 per record |
| 5 | a corrupt registry raised an uncaught exception instead of refusing | `ProvenanceRegistryError`; `main()` returns exit 3 |
| 6 | **check/use race between classification and write** | non-force writes use `os.link()`, which refuses to clobber |
| 11 | `True == 1` and `1.0 == 1` passed the schema binding | `type(pv) is not int` |
| 12 | `{"populations": [null]}` crashed (introduced by the round-1 fix, caught in round 2) | non-object population entry raises |

Claude independently found and fixed a ninth before Codex reported it: the AC-1.b tests re-typed
`list(candidates)` instead of exercising the real force branch. `build_work_list()` was extracted
so both branches are tested through the function `main()` actually calls.

---

## 11. Rejected / Unverified Codex Findings

### REJECTED

- **`_under_current_contract()` before the registry.** Correct for the legitimate case: a residual
  re-annotated through a named migration must report as current-contract rather than be overridden
  by a stale declaration. It never queues work. Codex accepted on re-review.
- **uid path traversal in `sidecar_path()`.** Real, but **pre-existing and unchanged by this
  task**, and outside §9's authorisation. Escalated as FIND-4 instead of fixed. Codex re-review:
  `ACCEPT WITH RESIDUAL RISK`.
- **"AC-1.c is not literally met."** Contradicted by the verbatim text of `TASK.md` §7.1 — see
  FIND-1. Codex accepted once shown the exact wording.
- **duplicate JSON keys collapse.** True of `json.loads`, but both collapse directions fail
  closed: the run refuses rather than queues.

### PLAUSIBLE

`None.` Every material finding resolved to `CONFIRMED` or `REJECTED` by direct execution.

### UNVERIFIED

`None.`

---

## 12. Master-Impacting Findings

### MIF-1 — `check_graph.py` failed on a PROTECTED file — **RESOLVED by user decision 2026-08-21**

> Retained as a record of what was escalated and why the file later changed. **The failure described
> here no longer exists.**

**Finding (as first reported).** `tools/check_graph.py:415` asserts that `docs/graph/README.md`
states the current unit-test count. `README.md:270` said
`413 個測試函式 … （pytest 參數化後展開成 525 個 case）`. This task's 22 new tests moved the counts
to 435 and 547, so the assertion failed.

**Evidence.** `check_graph.py` → `1 FAILURES — README unit-test count: README says 413, tests/
defines 435`. The check passed at task start, and the task's own tests are what broke it.

**Why it was not fixed on the first return.** `docs/**` was `PROTECTED` under §9.3, which closes
with "Anything not listed in 9.1 or 9.2 is protected by default. Writing to it is a scope violation
— escalate instead." DoD 10 (`check_graph.py` passes) and §9.3 could not both be satisfied, so the
conflict was escalated rather than resolved locally.

**Disposition — RESOLVED.** The user granted a **narrow scope exception** on 2026-08-21, now
recorded in `TASK.md` §9.1: `docs/graph/README.md` **line 270 only**, updating the stale counts
413 → 435 and 525 → 547, independently verified by Master against `pytest --collect-only` (547)
and a `def test_` count (435). The edit was made and confirmed to touch exactly one line;
`check_graph.py` now reports 2275 checks, all pass. The exception lifts nothing else under
`docs/**`.

**Still open for Master, not for this task:** whether `check_graph.py:415` coupling a prose counter
to the live test count should keep blocking any task that legitimately adds tests. This task took
no view and changed no checker.

### MIF-2 — n05b writes three artifacts, not the two §9.2 authorises

**Finding.** `resolve_stage1.py:629` writes `data/outputs/variant_registry.json` in the same call
as the two protocol artifacts. `TASK.md` §6 says "n05b rewrites two artifacts at once" and §9.2
authorises exactly those two. Running n05b — which C-002 mandates — necessarily writes a third.

**Evidence.** `resolve_stage1.py:627-629`; the file exists and was rewritten.

**Mitigation taken.** Backed up before the run; verified **byte-identical** afterwards
(md5 `5c84f5be5738c0ab374cb91568bddcca` both sides), because `VARIANTS` is a module constant that
this task did not change.

**Decision required:** none urgent. `TASK.md` §6 and §9.2 should say "three" for future tasks.

### MIF-3 — an observation adjacent to D0-009, reported and NOT acted on

**Finding.** While verifying that MetaFind never uses "learnable" of τ, the two actual occurrences
of the word were located: `2methdology.tex:54` (`f_h`, `f_x`) and `:87` (λ). Line 54 is the
sentence D0-009 is auditing.

**What was observed:** only that the word "learnable" appears there. **Nothing about the `f_x`
codomain was examined, concluded, or recorded.** No file in D0-009's scope was opened or modified.

**Why reported:** `TASK.md` §4's scope wall requires reporting anything bearing on D0-009's
question rather than acting on it.

**Can this task continue?** Yes; it did, without touching that area.

---

## 13. Remaining Risks

- **The registry is a declaration, and declarations can be re-made.** Anyone who can run
  `--declare` can re-partition the corpus. The tool refuses to declare over an unclassifiable
  record, refuses if the v1 population is not exactly the three uids `TASK.md` §6 names, and
  refuses if any v3 record fails `VALIDATOR_VERSION 2` — but it is a tool, not an immutable fact.
- **`sidecar_path()` does not validate uids** (FIND-4). Pre-existing. The safety of every claim
  here rests on `renders_index.jsonl` being project-generated; all 45,955 uids in it were verified
  as 32-character hex.
- **The 5,276 stale embeddings remain on disk and remain cache-invalid.** Untouched by this task,
  by design.
- **`D0-003` is still unresolved**, and `stage1.py:109` still has no existence guard, so the 3
  residuals remain a live `FileNotFoundError` for D3. This task protected them; it decided nothing.

---

## 14. Blocked Items

- **None within this task.** DoD 10 is fully met; MIF-1 is resolved.
- `D1_n06-reencode` remains **BLOCKED**, correctly. It unblocks only on **D10's** `USER_APPROVED`,
  not on this task's.

---

## 15. USER REVIEW INPUT

> **Remediation round, 2026-08-21.** The user returned three decisions on the first submission:
> `AC-1.c`'s registry route **approved**; the τ supporting sentence **corrected in the contract**;
> a **narrow `docs/graph/README.md` line-270 exception granted**. All three are applied below and
> reflected in `TASK.md` §7, §7.1, §9.1, §9.2, §12.1. FIND-1 and FIND-2 are retained with a
> `RESOLVED` disposition rather than deleted, so the record shows what was asked and what was
> ruled.

### Material Findings

- **FIND-1 — `AC-1.c` is worded differently in the two documents. RESOLVED.**
  `TASK.md` §7.1 (the authoritative execution contract, §5 row 2) reads *"explicit in the record
  **or in a declared registry**"*. `D10/USER_REVIEW.md` §7.0 (retained as provenance) reads
  *"explicit **in the record**"*.
  Evidence: `TASK.md` §7.1 AC-1.c vs `D10/USER_REVIEW.md` §7.0 AC-1.c, both read this session.
  This task proceeded on §7.1, because §5 designates it authoritative and §7.1 self-identifies as
  "as amended by the user 2026-08-21". **The delivered mechanism depends on the broader wording.**
  The narrower wording is not satisfiable alongside the rest of the contract: putting all three
  states in the record would require editing the records (§9.3 forbids) or stamping a v4 contract
  id on v3 records (R-A.1 forbids). Codex independently raised the narrow reading as a `MAJOR`
  finding; Claude rejected it on the verbatim §7.1 text and Codex accepted.
  **Resolved, user decision 2026-08-21:** §7.1 is the authoritative wording and the **declared
  registry route is approved**. `data/outputs/annotation_provenance.json` is now named explicitly
  in `TASK.md` §9.2. `D10/USER_REVIEW.md` §7.0's narrower "in the record" wording is stale and
  **Master corrected it** — this task modified no D10 file.
  The three prohibitions were re-checked against the delivered mechanism and all hold: a missing
  `annotation_contract` never means accepted (an undeclared record is `UNACCOUNTED` and stops the
  run); no v4 contract id is written onto any legacy-v3 record (`annotation_contract` is explicitly
  `null` in the registry, and all 45,955 records are byte-identical); the 3 residuals are a
  separate population, and the loader now *refuses* a registry that tries to declare them as
  legacy-v3.

- **FIND-2 — a supporting sentence in `TASK.md` §7 C-001 was factually wrong. RESOLVED.**
  The old sentence "The paper never uses the word 'learnable'" is false; the word appears at
  `2methdology.tex:54` and `:87`.
  Evidence: `grep -rn "learnable" docs/paper/metafind_source/*.tex`, run this session.
  **Resolved, user decision 2026-08-21:** the contract now carries the accurate and stronger
  argument — the paper uses "learnable" precisely, and names τ "a temperature **hyperparameter**"
  at `:79` and `:99`, so the authors place τ on the hyperparameter side of their own distinction
  **twice**. Re-verified here by reading all four lines.
  **The authority classification is unchanged**: τ = 0.5 is PAPER FACT; `learnable_temperature =
  false` is a USER-RATIFIED IMPLEMENTATION CHOICE on an INFERENCE and **must never be presented as
  a PAPER FACT**. `resolve_stage1.py`'s C-001 comment and the τ test docstring now carry the
  ratified formulation. The false sentence never entered any code, test, or artifact this task
  produced — confirmed by grep.

- **FIND-3 — the legacy-v3 corpus passes `VALIDATOR_VERSION 2`, measured here.**
  45,952 / 45,952 through `validate_annotation()`, 0 failures.
  Evidence: OBSERVED DATA, run this session; recorded in the registry with the distinction that
  these records were **admitted** at generation time by the validator then in force (which had no
  language rule) and satisfy `VALIDATOR_VERSION 2` **retrospectively**.

- **FIND-4 — `sidecar_path()` does not validate uids; `../` escapes `paths.ANNOTATIONS`.**
  Evidence: `annotate_run.py:74-75`, unchanged by this task. Codex `BLOCKER`, verified as real
  behaviour, classified out of D2a's authorised scope and **not fixed**.

- **FIND-5 — two genuine holes existed in the first AC-1 implementation and were found only by
  adversarial review.** A sidecar containing JSON `null` re-entered the work queue; a
  same-`prompt_version` content substitution inherited the "validated" declaration.
  Evidence: `CODEX_REVIEW.md` Findings 3 and 4, both reproduced by execution before fixing.
  Stated because the AC-1.a claim would otherwise appear to have been correct from the start.

### Material Decisions / Implementation Choices

- **What was done:** τ recorded as `init_temperature: 0.5`, `learnable_temperature: false`.
  **Authority:** `TASK.md` §7 C-001 — 0.5 is PAPER FACT (`3experiments.tex:15`);
  non-learnable is a **user ruling of 2026-08-21**, an IMPLEMENTATION CHOICE on an inference.

- **What was done:** AC-1 implemented as a **declared provenance registry** at
  `data/outputs/annotation_provenance.json`, with no annotation record mutated.
  **Authority:** `TASK.md` §8 R-D left the mechanism to the task as an IMPLEMENTATION CHOICE;
  the route is now **user-approved, 2026-08-21**, and the path is named in `TASK.md` §9.2.
  The choice of registry *design* — enumerated uids, per-record digests, fail-closed — remains
  this task's implementation choice and is argued in §6.

- **What was done:** the registry binds each declaration to a **sha256 of the record**, so the
  "validated under VALIDATOR_VERSION 2" claim cannot silently outlive the record it describes.
  **Authority:** **proposed, not yet authorised.** Beyond AC-1's literal requirement; adopted
  after Codex demonstrated the substitution. Cost 0.28 s and 3.2 MB.

- **What was done:** an existing record that is neither declared nor stamped makes
  `annotate_run` **refuse to start** (exit 3) rather than skip it.
  **Authority:** **proposed, not yet authorised.** This is Claude's reading of R-A.2 — treating
  an unclassified record as "fine" would re-create the absence-means-acceptance defect in a new
  place. It means a future genuinely-new corpus needs a `--declare` run before a bare
  `annotate_run` will proceed.

- **What was done:** the non-force write path changed from `tmp.replace(sc)` to `os.link(tmp, sc)`.
  **Authority:** **proposed, not yet authorised.** Closes the classify→write race. It touches the
  write path, which is arguably beyond "work-list semantics" in §9.1.

- **What was done:** n05b re-run carrying forward `paper_clip_train_scope: frozen`,
  `actual_clip_train_scope: frozen`, `confidence: moderate` from the previous artifact.
  **Authority:** carried forward deliberately, not decided here. Flagged because n05b requires
  `--paper-clip-train-scope` and a different value would have been a silent research change.

### Claude ↔ Codex Material Disagreement

- **Disagreement:** Codex held that AC-1.c requires all three states declarable in the registry,
  making the implementation non-compliant. Claude held that `AC-1.c`'s "or in a declared registry"
  permits the current-contract state to be self-evidenced in the record.
  **Verified disposition:** `REJECTED` (Codex's finding). Resolved against the verbatim text of
  `TASK.md` §7.1; Codex accepted on re-review. **Recorded even though it was resolved**, because
  it turns on the same wording difference as FIND-1, which the user has not yet confirmed.

- **Disagreement:** Codex graded uid path traversal a `BLOCKER`. Claude classified it real but
  out of scope.
  **Verified disposition:** `CONFIRMED` as behaviour, `REJECTED` as in-scope. Codex re-review:
  `ACCEPT WITH RESIDUAL RISK`. Carried as FIND-4.

No other material disagreement.

### Impact

- `D10_stage1-encoding-contract` — AC-1 evidence is ready to return with it.
- `D1_n06-reencode` — still blocked; `load_protocol()` no longer refuses on protocol grounds.
- `data/outputs/stage1_hyperparameters.json` — τ semantics changed; **n09 has not run**, so
  nothing downstream has baked the old hash.
- `data/outputs/annotation_provenance.json` — a new shared artifact `annotate_run` now depends on.
- `docs/graph/README.md` — line 270 counters updated under the granted exception; `check_graph.py`
  passes again.
- `metafind/data/annotate_run.py` — a bare run now refuses on an unaccounted record.

### Remaining UNKNOWN / Blocker

- `D0-002` and `D0-003` remain undecided. Untouched.
- Retrieval impact of the ratified template remains **UNKNOWN**, as D0-008 states.
- ~~Whether §7.1's or §7.0's `AC-1.c` wording is intended~~ — **resolved 2026-08-21**, FIND-1.

### Items Requiring USER Awareness / Decision

**Resolved in the 2026-08-21 remediation round — listed so the record is complete:**

- **FIND-1** — `AC-1.c` wording. **APPROVED**: §7.1 authoritative, registry route permitted.
- **FIND-2** — the "never uses the word 'learnable'" sentence. **CORRECTED** in the contract; the
  accurate argument is now in the code comment and the test.
- **MIF-1** — `docs/graph/README.md` line 270. **NARROW EXCEPTION GRANTED**; edit made, all gates
  pass.

**Still requiring user awareness or decision:**

- The **registry design** — a new shared artifact and a new dependency for `annotate_run`. The
  *route* is approved; the design below is not separately ratified.
- The **sha256 binding**, the **refuse-on-unaccounted** behaviour, and the **`os.link` write** —
  three implementation choices that go beyond AC-1's literal text. Each is argued in §6; none is
  ratified.
- **MIF-1** — `check_graph.py` fails; the fix is in a PROTECTED file.
- **MIF-2** — n05b writes three artifacts, not two.
- **FIND-4** — pre-existing uid path-traversal, deliberately not fixed here.
- **FIND-5** — the first AC-1 implementation had two real holes, found only by adversarial review.

---

## 16. Recommended Master Update

For Master's consideration after integration review — **not** enacted here:

- carry this task's AC-1 evidence back to `D10_stage1-encoding-contract`'s final USER REVIEW;
- correct `TASK.md` §6 / §9.2 to say n05b writes **three** artifacts (MIF-2) — the one item from
  the first return that the remediation round did not address;
- consider whether `check_graph.py:415`'s coupling of a prose counter to the live test count should
  keep blocking tasks that legitimately add tests (MIF-1 residual; this task changed no checker);
- once the user has ruled, record in `CONTEXT.md` §5 that C-001 and C-002 are **executed** and
  that the τ row has moved out of "established but not yet reflected in artifacts";
- register `data/outputs/annotation_provenance.json` in `CONTEXT.md` §10's file map;
- route MIF-3 to `D0-009` as an observation only.

**`D1_n06-reencode` must remain BLOCKED.** Acceptance of this task does not unblock it.

---

## 17. Recommended Next Action

Return D10 to final USER REVIEW carrying this task's AC-1 evidence.

Not started from this task. Master decides.

---

## 18. Completion Statement

This task has stopped execution after producing this HANDOFF, which is the **post-remediation**
version: the three USER DECISIONS of 2026-08-21 have been applied and the full verification battery
re-run afterwards. No figure in this document predates that edit.

**Explicitly, as `TASK.md` §16 requires: no n06 run, no n09 run, no training, no GPU embedding
generation, and no annotation mutation occurred.** All 45,955 annotation records are byte-identical
to task start, verified by checksum.

No D10 file was modified. `D0-009`'s scope wall was respected.

This task does not claim project-level or milestone-level completion, does not mark `D10` accepted,
and does not claim that any material decision is accepted. **`D2a` is not `USER_APPROVED` and not
`FINAL ACCEPTED`. `D1_n06-reencode` was not started and remains BLOCKED.**

Master must review and return one MASTER RECOMMENDATION: `ACCEPT` / `ACCEPT WITH FOLLOW-UP` /
`REWORK` / `REJECT` / `BLOCKED`. Those are recommendations only; this task becomes `DONE` on the
user's `APPROVE`.
