# MetaFindV1 Task Index

> Master-maintained registry of all formal workflow tasks and decision candidates.
> Execution instructions belong in each task's `TASK.md`. Results belong in each task's `HANDOFF.md`.
> Only Master edits this file.

**Initialized:** 2026-08-20
**Corrected:** 2026-08-20 (Master correction pass)
**Active formal work:** none. `D0-008` accepted with follow-up 2026-08-21. `D10_stage1-encoding-contract` is **READY** — approved 2026-08-21, `TASK.md` written, awaiting user approval to start the task conversation.
**Execution policy:** sequential — one ACTIVE task at a time.

---

## Status Definitions

`PLANNED` · `READY` · `ACTIVE` · `BLOCKED` · `REVIEW` · `DONE` · `REWORK` · `REJECTED`

- `READY` — every known dependency is satisfied and no unresolved decision gates it.
- `BLOCKED` — must name the blocker explicitly.
- `PLANNED` — scope depends on results or decisions that do not exist yet; the contract cannot be written honestly today.

---

## Role Definitions

- `MASTER` — orchestration, integration, status, dependency control
- `D0` — research / architecture / evidence decisions requested by Master
- `D1+` — bounded stage / work-package execution
- `CODEX` — independent reviewer

---

## Active Tasks

| ID | Task | Role | Status | Depends On | Parallel Safe | Path |
|---|---|---|---|---|---|---|
| — | None | — | — | — | — | — |

`D0-008_stage1-text-template` is **DONE** — accepted with follow-up 2026-08-21.
`D10_stage1-encoding-contract` is **READY** — contract at `workflow/tasks/D10_stage1-encoding-contract/TASK.md`. Becomes ACTIVE when the user approves starting its conversation. Nothing is ACTIVE now.

---

## Work Packages

| ID | Task | Role | Status | Depends On | Blocks | Conv. Mode | Parallel Safe |
|---|---|---|---|---|---|---|---|
| `D9_paper-figures-audit` | Read all 38 extracted paper figures; mark each U-register entry resolved / refuted / untouched; register new contradictions | D1+ | **READY** | — | evidence for D0-002, D0-004 | NEW CONVERSATION | **YES** (writes only `docs/`) |
| `D10_stage1-encoding-contract` | Clear the cache-validity BLOCKER (B-1…B-4); implement the ratified U-15 template (E-1, E-2, S-1, S-2); apply R-1/R-2/R-3; re-annotate the one truncated record; update the golden test; add the pre-flight gate | D1+ | **READY** — approved 2026-08-21, contract written, conversation not yet started | D0-008 (accepted) | D1, D2 | NEW CONVERSATION | NO |
| `D1_n06-reencode` | Full n06 re-encode of text + image embeddings over the admitted corpus | D1+ | **BLOCKED** | **D10** | D3 | NEW CONVERSATION | YES once unblocked (writes only `data/outputs/embeddings/`) |
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
| `D1_n06-reencode` | **D10** | D0-008 is accepted, but ratification alone does not unblock D1. Follow-up F-1 — the cache completion/validity BLOCKER, exit criteria B-1…B-4 — must be cleared, and the ratified template must actually be implemented |
| `D2_stage1-prereq` | D0-002, D0-003, D10 | `tower_sharing` mode; disposition of the 3 `prompt_version:1` annotations. Plus correction C-001 (τ = 0.5), in-scope execution. All land in artifacts n09 writes or hashes |
| `D3_stage1-train` | D1, D2, D0-003 | Complete embedding cache (45,952 `.npz`); three protocol files present and G3-valid; splits must not contain uids with no embedding, or `stage1.py:109` raises `FileNotFoundError`. Plus D0-005 if D0-002 selects `fully_separate` |
| `D4_gallery-index` | D3 | A Stage 1 checkpoint must exist |
| `D5_stage2-prereq` | D0-004, D0-006, D4 | ESSGNN axis coupling; node-text handling; promoted gallery index |
| `D6_stage2-train` | D5 | Stage 2 inputs resolved and n11b index built |
| `D7_eval-table1` | D4 | Promoted, fingerprint-verified gallery index |
| `D8_eval-table2` | D0-007, D6 | Evaluation scene construction protocol; a trained Stage 2 model |

---

## Decision Queue

Candidates identified by Master. **`D0-008` is ACCEPTED (2026-08-21).** The other six have no decision file and none is open.

A candidate becomes a formal decision only when Master creates its file under `workflow/decisions/` after user approval. Master prepares sections 1–5 (framing and evidence pointers); D0 owns sections 6–11; Master fills section 12 on review.

| ID | Question | Status | Decision File | Blocks | On Critical Path |
|---|---|---|---|---|---|
| D0-008 | Ratify the Stage 1 text serialization template (U-15) | **`ACCEPTED`** 2026-08-21, with follow-up | `workflow/decisions/D0-008_stage1-text-template.md` | resolved | done |
| D0-002 | U-16 `tower_sharing`: `shared_backbone_separate_fusion` / `fully_shared` / `fully_separate` | `OPEN` | — | D2, D3, Stage 2 feasibility | **YES** |
| D0-003 | The 3 `prompt_version:1` annotations: admit, drop, or re-annotate. **Hard blocker for D3** — if admitted, `stage1.py:109` raises `FileNotFoundError` | `OPEN` | — | D2, D3 | **YES** |
| D0-005 | `build_model()` bypasses `Stage1RuntimeConfig`; single backbone; shared `FusionConfig` object | `OPEN` | — | D3 | Conditional on D0-002 |
| D0-004 | ESSGNN `coord_feat` / `architecture_family` coupling | `OPEN` | — | D5, ablation design | No |
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

| ID | Task | Accepted By Master | Result Artifact | Codex Review |
|---|---|---|---|---|
| `D0-008_stage1-text-template` | Ratify the Stage 1 text serialization template (U-15) | **ACCEPT WITH FOLLOW-UP**, 2026-08-21 | `workflow/decisions/D0-008_stage1-text-template.md` §11 | 2 rounds, `gpt-5.6-sol` xhigh; 13 findings, 11 confirmed/partially confirmed; recorded in §9–§10 of the decision file |

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
5. `DONE` means: Definition of Done satisfied, verification completed, Codex review completed, and Master accepted the HANDOFF.
6. Sequential execution is the default.
7. Parallel execution requires explicit `Parallel Safe = YES` with verified dependency and filesystem non-conflict.
8. D-task conversations must not modify this file.
