# MetaFindV1 Task Index

> Master-maintained registry of all formal workflow tasks and decision candidates.
> Execution instructions belong in each task's `TASK.md`. Results belong in each task's `HANDOFF.md`.
> Only Master edits this file.

**Initialized:** 2026-08-20
**Corrected:** 2026-08-20 (Master correction pass)
**Active formal work:** `D1_n06-reencode` — approved 2026-08-21, **starting now**. `D0-009` is **`USER_APPROVED`** (`DL-004`). Four decisions are FINAL ACCEPTED: `DL-001`, `DL-002`, `DL-003`, `DL-004`. **`D0-008` (`DL-001`), `D10` (`DL-002`) and `D2a` (`DL-003`) are all `USER_APPROVED` as of 2026-08-21.** The migration backlog is empty. **`D1_n06-reencode` is UNBLOCKED and READY** — TASK.md proposed, awaiting user review; not started.
**Execution policy:** sequential — one ACTIVE task at a time.

---

## Status Definitions

**Execution status and integration status are two different facts. Do not merge them.**

> `task execution complete` **≠** `project decision accepted`
>
> A task can legitimately be execution `COMPLETE` and integration `AWAITING_USER_REVIEW` at the same time.

### Execution status

`PLANNED` · `READY` · `ACTIVE` · `BLOCKED` · `REVIEW` · `COMPLETE` · `REWORK` · `REJECTED`

- `PLANNED` — scope depends on results or decisions that do not exist yet; the contract cannot be written honestly today.
- `READY` — every known dependency is satisfied and no unresolved decision gates it.
- `BLOCKED` — must name the blocker explicitly.
- `COMPLETE` — the work is finished and verified. **This is not acceptance.**

### Integration status

`—` · `AWAITING_USER_REVIEW` · `USER_APPROVED` · `USER_REJECTED` · `REWORK` · `BLOCKED`

- `AWAITING_USER_REVIEW` — Master has recommended; the user has not decided. **Not project state.**
- `USER_APPROVED` — the user approved. FINAL ACCEPTED, integrated.
- `USER_REJECTED` — the user rejected. The finding may stand; the remedy does not.
- `REWORK` / `BLOCKED` — returned to its owner by Master routing; changes no project state.

`DONE` requires execution `COMPLETE` **and** integration `USER_APPROVED`.

Governing rules: `workflow/WORKFLOW.md` §13B. Project-level record: `workflow/DECISION_LEDGER.md`.

---

## Role Definitions

- `USER` — **final research / project authority.** The only role that can convert a recommendation into FINAL ACCEPTED
- `MASTER` — orchestration, integration review, recommendation, status, dependency control
- `D0` — research / architecture / evidence decisions requested by Master
- `D1+` — bounded stage / work-package execution
- `CODEX` — independent reviewer

---

## Active Tasks

| ID | Task | Role | Execution | Integration | Depends On | Parallel Safe | Path |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

`D2a_stage1-protocol-refresh` — **execution `COMPLETE`, integration `USER_APPROVED` 2026-08-21** (`DL-003`), MIF-2 ratified. Only `D0-009` remains running.

**Two concurrent work items were approved 2026-08-21; D2a has since completed.** Master verified `WORKFLOW.md` §7's five conditions for this pair: no dependency, no filesystem overlap (D0-009 writes only its decision file; D2a touches nothing under `workflow/decisions/`), no scientific conflict (Stage 1 encoding vs Stage 2 geometry), each independently verifiable, and the audit is long enough that concurrency saves real time.

**Scope wall — neither may modify the other's scope.** A finding that bears on the other is a `MASTER-IMPACTING FINDING`, reported and not acted on.

`D0-008_stage1-text-template` — execution `COMPLETE`, integration **`USER_APPROVED`** (2026-08-21, ledger `DL-001`). Migration through the user review gate is complete.
`D10_stage1-encoding-contract` — execution **`COMPLETE`**, integration **`AWAITING_USER_REVIEW / MODIFIED`** (`DL-002`). Master reviewed 2026-08-21 and recommended ACCEPT WITH FOLLOW-UP; user returned **`MODIFY`**, approving the implementation in principle but **withholding acceptance** pending condition **`AC-1`** (legacy-v3 rerun protection). Brief: `workflow/tasks/D10_stage1-encoding-contract/USER_REVIEW.md` §7.0.

**Dependency correction accepted (user decision #5, 2026-08-21):** the Stage 1 critical path is `n05b (C-001 + C-002) → n06`, **not** `D1 → D2`. Verified: `splits.py` never reads an embedding and `encode_text_image.py` never reads anything n09 writes, so **n06 and n09 are independent**. `D0-002` and `D0-003` gate **n09 only**.

```
[D2a: n05b — C-001 τ=0.5 + C-002 protocol + AC-1 + provenance]  ──►  D1 (n06)
[D0-002 tower_sharing · D0-003 the 3 v1 records]                ──►  n09 (splits)
                        D1 + n09                                ──►  D3
```

`D2a_stage1-protocol-refresh` — **READY, user-approved 2026-08-21.** Carries `AC-1`.
`D10_stage1-encoding-contract` — execution `COMPLETE`, integration `AWAITING_USER_REVIEW / MODIFIED`, blocked on `AC-1` which `D2a` carries.

---

## Work Packages

| ID | Task | Role | Status | Depends On | Blocks | Conv. Mode | Parallel Safe |
|---|---|---|---|---|---|---|---|
| `D9_paper-figures-audit` | Read all 38 extracted paper figures; mark each U-register entry resolved / refuted / untouched; register new contradictions | D1+ | **READY** | — | evidence for D0-002, D0-004 | NEW CONVERSATION | **YES** (writes only `docs/`) |
| `D10_stage1-encoding-contract` | Clear the cache-validity BLOCKER (B-1…B-4); implement the ratified U-15 template (E-1, E-2, S-1, S-2); apply R-1/R-2/R-3; re-annotate the one truncated record; update the golden test; add the pre-flight gate | D1+ | **READY** — approved 2026-08-21, contract written, conversation not yet started | D0-008 (accepted) | D1, D2 | NEW CONVERSATION | NO |
| `D1_n06-reencode` | Full n06 re-encode of text + image embeddings over the admitted corpus | D1+ | **`READY` — user-approved 2026-08-21**; conversation sets `ACTIVE` | D0-008 ✅ · D2a ✅ · D10 ✅ | D3 | NEW CONVERSATION | `YES WITH WORKTREE ISOLATION` for D0-009 only | `workflow/tasks/D1_n06-reencode/TASK.md` |
| `D2_stage1-prereq` | Apply C-001; re-run n05b (carries C-002); run n09_build_splits; produce `splits.json`, `eval_protocols.json`, `stage1_protocol.json`; verify G3 | D1+ | **BLOCKED** | D0-002, D0-003, D10 | D3 | NEW CONVERSATION | NO |
| `D3_stage1-train` | Stage 1 smoke (limited assets, 1 epoch) then full training; checkpoint, curves, full provenance | D1+ | **BLOCKED** | D1, D2, D0-003 (hard) (+ D0-005 conditionally) | D4 | NEW CONVERSATION | NO |
| `D4_gallery-index` | n11 staging → G4 freeze → n12 promote; encoder fingerprint cross-check | D1+ | **BLOCKED** | D3 | D5, D7 | NEW CONVERSATION | NO |
| `D5_stage2-prereq` | ESSGNN axis resolution, n08 node-text handling, n11b stage-2 gallery index | D1+ | **BLOCKED** | D0-004, D0-006, D4 | D6 | NEW CONVERSATION | NO |
| `D6_stage2-train` | n13 Stage 2 training | D1+ | **BLOCKED** | D5 | D8 | NEW CONVERSATION | NO |
| `D7_eval-table1` | Implement and run n15_eval_retrieval; produce Table 1 | D1+ | **BLOCKED** | D4 | — | NEW CONVERSATION | YES with D5/D6 once D4 lands (writes eval outputs only) |
| `D8_eval-table2` | n15a/n15b/G7/n15c, n16 compose, n17 judge; produce Table 2 | D1+ | **BLOCKED** | D0-007, D6 | — | NEW CONVERSATION | NO |

Not yet scoped as work packages, deliberately: n14 equivariance probe, n18/n19 ablations, n20 aggregate, n21 compare-to-paper, G5, n22 publish. Their scope depends on decisions and artifacts that do not exist. Writing contracts for them now would be invention.

---

## Blocked Tasks

| ID | Blocked By | Required Resolution |
|---|---|---|
| ~~`D1_n06-reencode`~~ | — | **UNBLOCKED 2026-08-21.** All three prerequisites `USER_APPROVED` |
| `D2_stage1-prereq` | D0-002, D0-003, D10 | `tower_sharing` mode; disposition of the 3 `prompt_version:1` annotations. Plus correction C-001 (τ = 0.5), in-scope execution. All land in artifacts n09 writes or hashes |
| `D3_stage1-train` | D1, D2, D0-003 | Complete embedding cache (45,952 `.npz`); three protocol files present and G3-valid; splits must not contain uids with no embedding, or `stage1.py:109` raises `FileNotFoundError`. Plus D0-005 if D0-002 selects `fully_separate` |
| `D4_gallery-index` | D3 | A Stage 1 checkpoint must exist |
| `D5_stage2-prereq` | D0-004, D0-006, D4 | ESSGNN axis coupling; node-text handling; promoted gallery index |
| `D6_stage2-train` | D5 | Stage 2 inputs resolved and n11b index built |
| `D7_eval-table1` | D4 | Promoted, fingerprint-verified gallery index |
| `D8_eval-table2` | D0-007, D6 | Evaluation scene construction protocol; a trained Stage 2 model |

---

## Decision Queue

Candidates identified by Master. **`D0-008` is `USER_APPROVED` (2026-08-21, ledger `DL-001`).** The other six have no decision file and none is open.

A candidate becomes a formal decision only when Master creates its file under `workflow/decisions/` after user approval. Master prepares sections 1–5 (framing and evidence pointers); D0 owns sections 6–11; Master fills section 12 on review.

| ID | Question | Status | Decision File | Blocks | On Critical Path |
|---|---|---|---|---|---|
| D0-008 | Ratify the Stage 1 text serialization template (U-15) | **`USER_APPROVED`** 2026-08-21 (`DL-001`) | `workflow/decisions/D0-008_stage1-text-template.md` | resolved | done |
| D0-002 | U-16 `tower_sharing`: `shared_backbone_separate_fusion` / `fully_shared` / `fully_separate` | `OPEN` | — | D2, D3, Stage 2 feasibility | **YES** |
| D0-003 | The 3 `prompt_version:1` annotations: admit, drop, or re-annotate. **Hard blocker for D3** — if admitted, `stage1.py:109` raises `FileNotFoundError` | `OPEN` | — | D2, D3 | **YES** |
| D0-005 | `build_model()` bypasses `Stage1RuntimeConfig`; single backbone; shared `FusionConfig` object | `OPEN` | — | D3 | Conditional on D0-002 |
| D0-004 | ESSGNN `coord_feat` / `architecture_family` coupling | `OPEN` | — | D5, ablation design | No |
| D0-009 | MetaFind §2.5 `f_x → R³` | **`USER_APPROVED`** 2026-08-21 (`DL-004`) — verdict `PAPER-AMBIGUOUS`, Option A adopted as a USER-RATIFIED IMPLEMENTATION CHOICE | `workflow/decisions/D0-009_essgnn-fx-codomain.md` | resolved | done |
| D0-006 | n08 node-text information collapse (`object_text()` is category-only) | `OPEN` | — | D5 | No |
| D0-007 | Table 2 protocol: 200-scene construction, 1–5 vs 0–10 scale comparability | `OPEN` | — | D8 | No |

Full evidence for each candidate: `workflow/MASTER.md` §8.

**Reclassified out of the decision queue.** τ was previously registered here as `D0-001`. It is not a research adjudication: `3experiments.tex:15` states "The temperature is 0.5 for all experiments", it is the only value the paper gives, and no conflicting statement exists in the paper source. It is now tracked as an implementation correction.

### Implementation Corrections — execution, not adjudication

| ID | Correction | Established requirement | Owner task | Status |
|---|---|---|---|---|
| C-001 | `stage1_hyperparameters.json` records `init_temperature: 0.07`, `learnable_temperature: true`; `resolve_stage1.py:197-211` hardcodes them with no override path | **PAPER FACT τ = 0.5** (`3experiments.tex:15`). Non-learnable is a strongly supported INFERENCE | `D2_stage1-prereq` | OPEN |
| C-002 | `stage1_encoding_protocol.json` records the v1 metre template while `serialize_annotation()` emits the v3 centimetre template | The artifact must describe runtime truth. Previously recorded in `_workflow_old_20260820/任務/A_n06-reencode/TASK.md` | `D1_n06-reencode` (refresh) / `D2_stage1-prereq` (n05b re-run) | OPEN |

C-001 requires a code change, not only an artifact edit. Departing from τ = 0.5 is permitted only as a registered **DEVIATION** and is a user ruling, not a task-local choice.

**Not a D0 candidate.** The following were reviewed and are not research adjudications — they are execution detail belonging inside a D-task, or already-settled positions:

- `renders_index.jsonl` (45,955) versus `data/outputs/renders/` directory count (46,000). A bookkeeping discrepancy to resolve inside D1 or D2, not a research question.
- n04 unit-sphere normalisation, `f_x` scalar output, absence of a no-relation filter in n08, undirected semantic edges. Settled positions with recorded reasoning; see `CONTEXT.md` §5.
- `onFloor` / `onObject` accuracy ceiling. A known limitation to report, not a defect to fix.

---

## Parallelization Rules

**Policy: sequential. One ACTIVE execution task.** Parallel requires Master to verify all five conditions in `workflow/WORKFLOW.md` §7 **and** obtain separate user approval.

**No parallel execution is proposed at this time.** The next action is a single task.

Pairs Master has pre-assessed as non-conflicting, held for future consideration only — listing them here is not a recommendation to run them:

| Pair | Why non-conflicting | When Master would propose it |
|---|---|---|
| D1 ∥ D9 | D1 writes only `data/outputs/embeddings/`; D9 writes only `docs/` | Once `D1_n06-reencode` is ACTIVE and running as a multi-hour GPU job, Master will propose marking D9 `PARALLEL SAFE` and ask the user to approve it separately |
| D1 ∥ D0-002 / D0-003 | Decision investigation does not mutate `data/outputs/` | Same trigger. **Caveat:** if D0-003 changes the admitted corpus, D1's output must be re-checked against the final admitted set before D3 |

Known conflict risk to watch: several candidate tasks want to write `docs/audit/C_PAPER_CONTRADICTIONS.md`. D9 and any later D0-004 work both touch it. Never run those concurrently.

---

## Conversation Mode

Default `NEW CONVERSATION` for every task above. A task reads `CLAUDE.md`, applicable `.claude/rules/`, `workflow/CONTEXT.md`, its own `TASK.md`, and only the files `TASK.md` names.

No task currently requires `FORK REQUIRED`. Mark it only when a task depends on conversation-only reasoning that genuinely cannot be captured in `CONTEXT.md`, `TASK.md`, or a decision document, and state why.

---

## Completed Tasks

> A row here means **execution complete**. It does not mean the project accepted the result — check the integration column.

| ID | Task | Accepted By Master | Result Artifact | Codex Review |
|---|---|---|---|---|
| `D0-008_stage1-text-template` | Ratify the Stage 1 text serialization template (U-15) | Master recommended ACCEPT WITH FOLLOW-UP; **user `MODIFY` → `APPROVE` 2026-08-21**. Integration: **`USER_APPROVED`** (`DL-001`) |
| `D10_stage1-encoding-contract` | Stage 1 encoding contract: cache validity, ratified serializer, P-1…P-5 | Master recommended ACCEPT WITH FOLLOW-UP; **user `MODIFY` → `APPROVE` 2026-08-21** after `AC-1` was demonstrated. Integration: **`USER_APPROVED`** (`DL-002`). **`G-7` not ratified — independently OPEN** |
| `D2a_stage1-protocol-refresh` | τ = 0.5 · protocol refresh · AC-1 rerun protection · legacy-v3 provenance | Master recommended ACCEPT WITH FOLLOW-UP; **user `APPROVE` 2026-08-21, MIF-2 ratified**. Integration: **`USER_APPROVED`** (`DL-003`) | `workflow/decisions/D0-008_stage1-text-template.md` §11 | 2 rounds, `gpt-5.6-sol` xhigh; 13 findings, 11 confirmed/partially confirmed; recorded in §9–§10 of the decision file |

Follow-ups carried from D0-008: **F-1** (cache-validity BLOCKER) through **F-5** are D10's scope; **F-6** done by Master at acceptance; **F-7** routes to D0-003; **F-8**/**F-9** deferred. Full table in §12.4 of the decision file.

---

## Task Naming Convention

Work packages: `D<number>_<short-slug>` → `workflow/tasks/<task-id>/`
Decisions: `D0-<number>_<short-slug>` → `workflow/decisions/<decision-id>.md`

---

## Master Rules

1. Only Master changes task status in this file.
2. A task becomes `ACTIVE` only after user approval.
3. `READY` means all known dependencies are satisfied and no unresolved decision gates it.
4. `BLOCKED` must name the blocker explicitly.
5. `DONE` means **all** of: Definition of Done satisfied · verification completed · Codex review completed · Master issued an ACCEPT-class **recommendation** · a USER REVIEW BRIEF was delivered · **the user returned `APPROVE`** · Master integrated the result.
   Master's recommendation alone is not `DONE`. Neither is Claude + Codex consensus.
6. Sequential execution is the default.
7. Parallel execution requires explicit `Parallel Safe = YES` with verified dependency and filesystem non-conflict.
8. D-task conversations must not modify this file.
9. Execution status and integration status are recorded separately. Never write `DONE` for a task whose integration is `AWAITING_USER_REVIEW`.
10. Every material decision recorded here must have a matching entry in `workflow/DECISION_LEDGER.md`. If the two disagree, the ledger is the project-level record.
