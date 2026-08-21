# MetaFindV1 Decision Ledger

> **Material decisions only.**
>
> This is not a research log, not a conversation log, and not a finding dump. Findings live in decision files, HANDOFFs, and audit documents. Only decisions that change project state are recorded here.
>
> Master maintains this file. D0 and D-tasks do **not** write their own recommendations here as FINAL.
>
> Governing rules: `workflow/WORKFLOW.md` §13A (Finding vs Decision), §13B (User Review Gate), §13C (USER REVIEW BRIEF).

---

## The rule

> A material decision reaches `USER_APPROVED` **only** when the user approves it.
>
> Claude + Codex + Master consensus does **not** substitute for user approval.

Master's `ACCEPT` / `ACCEPT WITH FOLLOW-UP` enters this ledger as `AWAITING_USER_REVIEW`, never as `USER_APPROVED`.

---

## Status vocabulary

| Status | Meaning |
|---|---|
| `PROPOSED` | A decision has been formulated but no Master integration review has completed |
| `AWAITING_USER_REVIEW` | Master has reviewed and recommended. A USER REVIEW BRIEF is owed or delivered. **Not yet project state** |
| `USER_APPROVED` | The user approved. **FINAL ACCEPTED.** Master may integrate into MASTER.md, CONTEXT.md, INDEX.md, dependency state, and project-wide contracts |
| `USER_REJECTED` | The user rejected. The underlying finding may still stand; the proposed remedy does not |
| `SUPERSEDED` | Replaced by a later decision. **Never delete a superseded entry** — mark it and point to the replacement |

`MODIFY` is recorded as `USER_APPROVED` with the **user's** wording as the decision, and Master's original proposal retained in the notes.

`INVESTIGATE MORE` leaves the entry at `AWAITING_USER_REVIEW` with the additional investigation named.

---

## What belongs here

Any addition, modification, confirmation, or reversal of:

paper interpretation · architecture · implementation choice · deviation · dataset semantics · annotation semantics · preprocessing · training protocol · evaluation protocol · shared artifact semantics · cache validity · checkpoint validity · dependency ordering · scientifically meaningful assumption · anything that can materially change reproduction results

**When in doubt, record it.**

## What does not belong here

Routine execution that changes no scientific behaviour and no shared contract: local refactors inside an approved scope, test scaffolding, log formatting, documentation corrections that only make a comment describe existing code accurately.

---

## Entry format

Each entry records:

```
### <Decision ID>

| | |
|---|---|
| Source | task / D0 that produced it |
| Issue / Finding | what was discovered (a FINDING, not a remedy) |
| Evidence references | file:line, paper section, measurement, decision file section |
| Decision / Resolution | what is to be done (a DECISION) |
| Authority classification | PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION / OBSERVED DATA / INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / USER DECISION |
| USER final decision | APPROVE / REJECT / MODIFY / INVESTIGATE MORE / — |
| Affected components | tasks, artifacts, stages |
| Status | see vocabulary above |
| Date | proposed / resolved |
```

Keep the Issue and the Decision in separate rows. Do not merge them.

---

# Ledger

> Entries are added by Master at integration review and updated when the user acts.
> **Ordered newest first.**

---

### `DL-006` — `D0-003` resolved: the 3 legacy-v1 residuals are deleted and re-annotated under v5

| | |
|---|---|
| **Source** | USER, 2026-08-22, following the 「不准使用舊模型產出的東西」 directive |
| **Issue / Finding** | Three uids carried `prompt_version 1` records long after the rest of the corpus moved to v3: `6c7db00cc164467ebac356a5ca67368b` (pole dancer), `8a0192eee6fb4140bb3e9696b3dbae5a` (pinecone), `a397b648d6eb48d7909d1ee11235e78f` (train). `D0-003` asked whether to admit, drop, or re-annotate them, and the question was **hard-blocking `D3`**: `splits.py:169-171` admitted all 45,955 while `stage1.py:109` loads the `.npz` with no existence guard, so Stage 1 would raise `FileNotFoundError` mid-epoch |
| **Why they were stuck — established 2026-08-22, not previously recorded** | **They are not broken assets.** Renders and point clouds exist for all three. They failed the v3 re-annotation on a **validator rule**: `quarantine_n05_annotate.jsonl` records `terminated_by: repair_budget`, `failure_class: MODEL_RECOVERABLE`, and for `6c7db00c…` the exception is verbatim ``synset` = 'pom.pom.n.01' is not a WordNet id of the form "lemma.n.01"`. The model invented a malformed synset, exhausted the repair budget, and the old v1 record was never overwritten |
| **Why the question dissolves under v5** | **v5 does not ask the model for `synset` at all.** Design Decision 4 replaced it with a deterministic lookup over the 1,156-term LVIS vocabulary (`metafind/data/lvis_synsets.json`, built and cross-checked against detectron2, 0 invented entries). The exact failure mode that produced these three residuals **no longer exists in the v5 pipeline** |
| **Evidence references** | `data/outputs/logs/quarantine_n05_annotate.jsonl` (3 of its 5 records) · `metafind/data/annotate.py` `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` / `SCHEMA_VERSION 3` · `metafind/data/lvis_synsets.json` · `D14/ESCALATION.md` "synset table 1,156 / 1,156 … 0 invented entries" · the three records read verbatim before deletion |
| **Decision / Resolution** | **Delete them.** They are re-annotated under v5 alongside the other 45,952, with no special handling. If v5 fails on any of them, they are quarantined by the ordinary path like any other failure — there is no longer a legacy schema to preserve |
| **Authority classification** | **USER DECISION.** The supporting facts are OBSERVED DATA (the quarantine records) and OBSERVED IMPLEMENTATION (v5's synset lookup). **This is not resolution-by-mutation**: the deletion follows from the v5 design the user already ratified, and the reason the residuals existed was established from evidence *before* deleting them |
| **USER final decision** | **`APPROVE`** — 2026-08-22 (「如果可以就刪」, conditional on the question being answerable in the new work; Master established that it is, and recorded the basis above) |
| **Affected components** | `data/outputs/annotations/` (now **0 files**) · `D0-003` closed · `D3`'s hard blocker cleared · `D2`'s corpus denominator becomes 45,955 uniformly · `annotation_provenance.json` must be rebuilt · `MASTER.md` §8, `INDEX.md` Decision Queue, `CONTEXT.md` §5/§6/§7 all still describe `D0-003` as UNRESOLVED and must be corrected |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

**Supersedes** the "explicitly NOT resolved: `D0-003`" clauses in `DL-002` and `DL-003`, and the
`legacy_v1_residual_unresolved` state in `data/outputs/annotation_provenance.json`. Those entries
are **not deleted** — they were true when written.

**`AC-1` no longer has a subject.** It protected an accepted legacy corpus from accidental
re-annotation. That corpus was deliberately deleted on 2026-08-22, so AC-1.a ("a bare
`annotate_run` queues 0 records") is now expected to be **false** — a bare run should queue the
whole corpus, because that is the intent. `DL-003-A1` must be rewritten to say so rather than
re-proving a guarantee whose purpose has ended. **Do not treat a bare run queuing 45,955 as a
regression.**

---

### `DL-007` — n05 v5 anchors object identity on the Objaverse-LVIS label. **This is a DEVIATION.**

**Registered by Master 2026-08-21 during the re-initialization audit. It had no ledger entry, which was a gap: it is the most scientifically material change made to the pipeline this week.**

| | |
|---|---|
| **Source** | `D14_n05-v5-reannotate`, design `workflow/n05_v5_design.md`, approved by the user 2026-08-21 |
| **Issue / Finding** | n05 asked the VLM to *identify* objects from 224×224 renders it often cannot read, so it collapsed onto high-frequency priors. **LVIS ground truth:** 1,156 distinct categories, top-20 share **7.1%**. **Qwen output:** 3,036 categories, top-20 share **22.3%** — 3× more concentrated than the truth. `toy` is the single most common answer at 1,542 (3.4%), a word `build_prompt` **explicitly forbids**. Agreement with LVIS: category matches 29.0%, LVIS word appears in the description 28.4%, **neither 67.8%** |
| **Why it is not a `category`-only defect** | `build_prompt` says *"Estimate its size from what kind of object it is, not from the picture"* — dimensions and placement are **by design** derived from the category. Observed: `LVIS pinecone → "a dark brown hairbrush"`, `LVIS mug → "a cylindrical pillow"`, `LVIS truck → "a modern air conditioner unit"`. **A wrong category is a wrong record, not a wrong field** |
| **Evidence references** | `workflow/MIF_n05_diagnosis.md` · `workflow/MIF_n05_category_vs_lvis.md` · `objaverse_lvis_metadata.json → value_to_key_mapping` (46,207 uid→category, fetched by `download.py:70`, **read by nothing**) · `annotate.py:366` `build_prompt(n_views)` receives only `n_views` · `annotate.py:510` `validate_annotation()` never checks semantic correctness |
| **Resolution not driven by resolution** | correlation(best-view occupancy, LVIS agreement) = **+0.054**, agreement flat at ~28–30% across a ~100× range of effective object pixels. **Re-rendering would not have fixed it** and is explicitly out of scope |
| **Decision / Resolution** | n05 v5 supplies the LVIS category to the model as the anchored identity; the model may **refine downward** but not replace laterally; it also emits `identity_confirmed`. `PROMPT_VERSION 5`, `SCHEMA_VERSION 3`, `VALIDATOR_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7` |
| **Authority classification** | **DEVIATION.** The paper has the VLM *generate* the category: `2methdology.tex:28` and `neurips_2025.tex:100` both say GPT-4o produces the structured annotations; Figure 2's caption (`2methdology.tex:24`) says the VLM *generates* attributes including category. **Feeding the dataset label in is a departure and must never be described as paper-faithful.** `D14/TASK.md` `R-E` binds |
| **What remains UNRESOLVED** | **`D0-010` has not been researched** — its §6–§11 are empty. The choice between *prompt hint* / *hard value* / *cross-check* / *record-only* was made by design ratification, **not** by a completed evidence audit. Also unresolved: whether anchoring merely substitutes LVIS's own errors for Qwen's, and whether `identity_confirmed` detects that or rubber-stamps the anchor (`IC-1`) |
| **USER final decision** | design **approved** 2026-08-21; **the deviation registration itself is pending** and reaches the user through `D14`'s acceptance brief |
| **Affected components** | `annotate.py`, `annotate_run.py`, `lvis_synsets.json`, the entire annotation corpus, every Stage 1 text embedding, both tables' comparability with the paper |
| **Status** | `AWAITING_USER_REVIEW` |
| **Date** | design approved 2026-08-21 · registered here 2026-08-21 |

**Registry gap, open:** `docs/graph/graph_spec.yaml` carries **no deviation entry** for LVIS anchoring. And `tools/check_graph.py:373-383` compares deviation **ids only, never the `what:` text** (`FU-A`), so a missing or falsified deviation description passes every gate silently.

---

### `DL-006` — the n05 annotation model is `Qwen3.8-27B`

**Registered by Master 2026-08-21 during the re-initialization audit. The user's decision (U-6) was made in conversation on 2026-08-21 and had no ledger entry.**

| | |
|---|---|
| **Source** | User decision **U-6**, 「走本地 Qwen3.8-27B」, recorded in `workflow/tasks/D14_n05-v5-reannotate/USER_DIRECTIVES.md` |
| **Issue / Finding** | n05 ran `Qwen/Qwen2.5-VL-7B-Instruct` as a stand-in for GPT-4o. A 7B model was a candidate cause of the identity-collapse defect in `DL-007` |
| **Decision / Resolution** | The annotation model becomes local **`Qwen3.8-27B`**, weights at `/mnt/data1/kyzen/models/Qwen3.8-27B`. Enacted at `annotate_run.py:72`. Master verified 2026-08-21: **18/18 shards, 56 GB, download complete** |
| **Authority classification** | **USER DECISION.** The substitution itself remains a **DEVIATION** — the paper says GPT-4o twice. Recorded under the split `D-2` (see `DL-005`). Reaching GPT-4o would narrow `D-2`, never discharge it |
| **Correction of record** | `D14/TASK.md` §7 originally justified its model prohibition with *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from the code comment at `annotate_run.py:71`. D14's finding F-2 shows OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **The conflict is UNRESOLVED, not resolved**, and the API has never been exercised. It must not be restated as settled |
| **UNRESOLVED and material** | **The model has never been loaded.** 56 GB at bf16 does not fit the RTX 5090's 32,607 MiB, so **quantization is required and has not been tested.** Quantization changes annotation quality, which makes it an **experimental condition**, not an engineering detail. **No Phase 3 runtime estimate is evidence-backed** — the "~19.6 GPU-h" figure belongs to the 7B model. Tracked as blocker **R2** (`MASTER.md` §11) |
| **USER final decision** | **`APPROVE`** — the model choice (U-6). The quantization condition and the runtime estimate are **not** covered by it |
| **Affected components** | `annotate_run.py:72` · deviation `D-2` · `D14` Phase 2 and Phase 3 · every annotation record |
| **Status** | **`USER_APPROVED`** for the model choice; the quantization condition is **OPEN** |
| **Date** | 2026-08-21 |

---

### `DL-005` — deviation `D-2` split into `D-2` (annotation) and `D-8` (scene judging)

| | |
|---|---|
| **Source** | Master, executing D14's escalation **P-2**, 2026-08-21 |
| **Issue / Finding** | `graph_spec.yaml` recorded one deviation — *"Qwen2.5-VL replaces GPT-4o for annotation **and** scene judging"*. After user decision **U-6** the annotation model became `Qwen3.8-27B` while the judge (n17) stayed `Qwen2.5-VL`. **One id would denote two different substitutions** |
| **Evidence references** | `graph_spec.yaml:130` (pre-split) · `annotate_run.py:72` (`MODEL_ID` already changed) · `2methdology.tex:28`, `neurips_2025.tex:100` (paper says GPT-4o) |
| **Decision / Resolution** | **Split, not rewrite-in-place.** `D-2` = Qwen3.8-27B for **asset annotation (n05)**. New `D-8` = Qwen2.5-VL for **scene judging (n17)**. Deviation count 六項 → 七項, synchronised across 5 files: `graph_spec.yaml`, root `README.md`, `docs/graph/README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`. `check_graph.py` → **2275 checks, all pass** |
| **Why split rather than rewrite** | One id covering two different substitute models is the exact ambiguity that produced this escalation, and the two will diverge further — n17 may change model later. D14 recommended splitting; Master agreed |
| **Authority classification** | The model change itself is a **USER DECISION** (U-6). This split is **bookkeeping that follows from it** — it records an existing fact accurately, it does not make a new choice |
| **USER final decision** | **pending** — reaches the user through D14's acceptance brief |
| **Affected components** | `docs/graph/graph_spec.yaml`, both `README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`; the reproduction report's deviation section |
| **Status** | `AWAITING_USER_REVIEW` — executed, not yet ratified |
| **Date** | 2026-08-21 |

**Correction carried in the same edit.** `D-2`'s stated reason was *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from a code comment. D14's finding F-2 showed OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **That conflict is UNRESOLVED.** The registry now records availability as UNRESOLVED rather than as established, in `graph_spec.yaml`, both READMEs, and `TASK.md` §7.

**Open gate weakness, registered not fixed:** `tools/check_graph.py:373-383` compares deviation **ids only** — regex `\|\s*\*\*(D-[0-9])\*\*` — and never reads the `what:` text. **A deviation whose description has gone false passes every gate silently.** That is how `D-2` stayed wrong. Found by D14. Not in D14's scope; unassigned.

---

### `DL-004` — MetaFind §2.5 `f_x → R³`: verdict and implementation

| | |
|---|---|
| **Source** | `D0-009_essgnn-fx-codomain` |
| **Issue / Finding** | MetaFind states `f_x: R^(2d+1+e) → R³` (`2methdology.tex:54`) and claims equivariance for **any orthogonal** `Q ∈ R^{3×3}` (`appendix.tex:23`), but **never defines the `·`** in the coordinate update (`2methdology.tex:52`). Master confirmed the silence exhaustively: zero hits for `hadamard` / `element-wise` / `inner product` / `dot product` / `contraction` across all five `.tex` files |
| **Evidence references** | `2methdology.tex:52`, `:54` · `appendix.tex:23`, `:29`, `:53`, `:61`, `:68` · `essgnn.py:311-312`, `:353`, `:358-359` · `C_PAPER_CONTRADICTIONS.md:114` (C3, **SEVERE**) · decision file §6–§11 · Codex round 1 (its BLOCKER changed the verdict) |
| **Verdict** | **`PAPER-AMBIGUOUS`.** MetaFind alone does not uniquely determine how the `R³` output participates in the coordinate update. **This must never be rewritten as "MetaFind explicitly got it wrong."** |
| **Decision / Resolution** | **Option A** — retain the scalar `f_x` coordinate multiplier. **`essgnn.py` behaviour is NOT modified**; what changed is the authority classification |
| **Authority classification** | `f_x → R³` stated = **PAPER FACT** · the operator is undefined = **PAPER FACT (as to silence)** · `h` invariance = **PAPER FACT** (`appendix.tex:29`, `:68`) · **scalar `f_x` = USER-RATIFIED IMPLEMENTATION CHOICE under a PAPER-AMBIGUOUS specification.** Not a PAPER FACT |
| **Rationale, recorded** | Not chosen because upstream EGNN is more sensible. Chosen because the operator semantics are undefined; Hadamard closes dimensionally but breaks the paper's own general-orthogonal equivariance claim; `R³` + contraction invents an operator MetaFind never defines; the scalar preserves equivariance and invents nothing |
| **Binding prohibition** | **"upstream EGNN settles it" may no longer be used as paper-interpretation authority anywhere in this project.** `E_GRAPH_REVALIDATION.md:175` must be corrected |
| **USER final decision** | **`APPROVE` with implementation decision A** |
| **Affected components** | `essgnn.py` (classification only, no behaviour change) · `docs/audit/` C3 · `docs/graph/` U-26 · `CONTEXT.md` §5 · Stage 2 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-21 |

**`MIF-1` REJECTED as a blocker.** D0-009 asked to be gated on `D0-004`'s "unresolved `h` semantics". Master rejected it: `appendix.tex:29` **assumes** `h^0` invariant and `:68` concludes the feature update invariant — `h` invariance is not an open question in MetaFind. `essgnn.py:353` feeds `f_h` only `h`, `radial`, and `edge_attr`, all invariant. `D0-004` concerns which layer's `h` reaches `f_x`; both are invariant. Option B's conflict is independent of `h`.

**Follow-ups registered, none blocking:**

| ID | Item |
|---|---|
| MIF-3 | `F_CODE_GRAPH_CONSISTENCY.md` — its `CONSISTENT` column is unlabelled and appears to mean "code matches the graph spec", not "code matches the paper". Terminology ambiguity |
| MIF-4a | `2.2e-16` vs `0.43` — **must not be described as a repo-verified measurement.** Cannot be reproduced: the `R³` variant does not exist in code |
| MIF-4b | No `R³`-variant equivariance test exists. **Narrower than reported:** `test_se3_equivariance` runs at `n_layers=3` (`test_essgnn.py:102`, `:112`) and `test_equivariance_negative_injection` (`:129`) proves it non-vacuous |
| MIF-5 | `normalize_coord_diff` has no MetaFind authority. `essgnn.py:189` defaults `False`, zero current impact. **Independent candidate — not D0-009's to touch** |
| — | Correct `E_GRAPH_REVALIDATION.md:175`'s `[UPSTREAM] settles it` |

---

### `DL-003-A1` — PREPARED AMENDMENT to `DL-003` / AC-1, to land WITH D14 Phase 3

**Status: `PREPARED, NOT IN FORCE`.** Drafted by Master 2026-08-21 at D14's request (P-5), so the
amendment lands **with** Phase 3 rather than being retrofitted after it. `DL-003` stands unchanged
until Phase 3 completes.

**Why an amendment is needed.** `DL-003` records the corpus as **accepted legacy-v3 validated under
`VALIDATOR_VERSION 2`**, and the provenance registry declares that as a population of **45,952**.
D14 Phase 3 re-annotates those same 45,952 under `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` /
`SCHEMA_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7`. **The moment Phase 3 finishes,
`DL-003`'s description of the corpus becomes false.**

### What changes

| | before Phase 3 | after Phase 3 |
|---|---|---|
| 45,952 records | `accepted_legacy_v3`, `VALIDATOR_VERSION 2` | annotated under the **current** contract `metafind_annot_v5@…` |
| how they satisfy AC-1.a | by **registry declaration** | by `is_complete()` — they carry the current `annotation_contract` |
| 3 legacy-v1 residuals | `legacy_v1_residual_unresolved` | **unchanged.** `D0-003` still UNRESOLVED |
| registry `accepted_legacy_v3` population | 45,952 | **0** — the state becomes historical |

### What must NOT change

- **AC-1.a still holds: a bare `annotate_run` queues 0 records TOTAL.** After Phase 3 the 45,952
  satisfy it through `is_complete()` instead of through the registry, and the 3 residuals still
  satisfy it through their declaration. **The guarantee is identical; only the mechanism moves.**
- **AC-1.b still holds:** `--force` still re-annotates; the named-migration form still works.
- **AC-1.c still holds:** three states remain explicit, never inferred from a missing field.
- **`D0-003` remains UNRESOLVED.** Nothing in Phase 3 or this amendment resolves it, and the
  3 residuals must be byte-identical at Phase 3's end.

### Required of D14 at Phase 3

1. Update the registry so `accepted_legacy_v3` no longer claims a population that has moved on.
   **Do not delete the state** — mark it historical, with the date and the contract that superseded it.
2. Re-prove AC-1.a **after** the registry update: bare run queues **0 TOTAL**, 0 legacy-v3, 0 residuals.
3. Re-prove AC-1.b: `--force` still yields a non-empty work list.
4. Confirm the 3 residuals' declaration survived untouched.

**If AC-1.a cannot be re-proved after the registry update, that is a `MASTER-IMPACTING FINDING`, not
a registry-editing problem.** The protection is the point; the registry is only how it is expressed.

### Authority

The re-annotation itself is a **USER DECISION** (U-6, U-10, and the design ratified 2026-08-21).
This amendment is **bookkeeping that follows from it** — it records a consequence, it does not make
a new choice. It still reaches the user through D14's acceptance brief.

**Superseded on landing:** `DL-003`'s "legacy-v3 validated under `VALIDATOR_VERSION 2`" description
of the 45,952, and `AC-1.e`'s instruction to record them as such. **`DL-003` is not deleted** — it is
the true record of what was accepted on 2026-08-21.

---

### `DL-003` — Stage 1 protocol refresh, τ = 0.5, and legacy-corpus rerun protection

| | |
|---|---|
| **Source** | `D2a_stage1-protocol-refresh` |
| **Issue / Finding** | `load_protocol()` refused the on-disk protocol after D10's B-2. τ = 0.5 had no code path (`resolve_stage1.py` hardcoded 0.07, 0 CLI flags). And `annotate_run.is_complete()` keyed on a contract **no** record carried, so a bare invocation would have queued **45,955** records — overwriting a corpus the user decided to preserve, and resolving `D0-003` by mutation |
| **Evidence references** | Master re-ran, read-only: `build_work_list(force=False)` → todo **0**; `force=True` → **45,955**; state histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}`; `load_protocol()` → `metafind_v2_cm@8e4b1fcc66c7f48c`; artifacts `0.5` / `False`; `prompt_version {3: 45952, 1: 3}`; 547 passed · 2275 checks · PRE-FLIGHT PASSED |
| **Decision / Resolution** | Accept the task. τ = 0.5 with `learnable_temperature: false` written through n05b; protocol refreshed to the ratified serializer; AC-1 satisfied via a **declared registry** (`data/outputs/annotation_provenance.json`) plus a relocated work-list predicate; the 45,952 formalized as accepted legacy-v3 under `VALIDATOR_VERSION 2`; the 3 residuals explicitly `legacy_v1_residual_unresolved` |
| **Authority classification** | τ = 0.5 = **PAPER FACT** (`3experiments.tex:15`) · `learnable_temperature: false` = **USER-RATIFIED IMPLEMENTATION CHOICE** on a strongly-supported inference — the paper uses "learnable" for `f_h`/`f_x`/λ but calls τ a "temperature hyperparameter" twice, and never states τ is non-learnable · AC-1 mechanism = IMPLEMENTATION CHOICE within the approved menu · registry choice, legacy-v3 formalization = **USER DECISION** |
| **Scope deviation — ratified** | **MIF-2.** `n05b` writes **three** artifacts (`resolve_stage1.py:660-662`); the contract declared two. `variant_registry.json` was rewritten. Master proved the rewrite byte-identical (`VARIANTS` absent from the diff; `_write()` is a deterministic `json.dump(indent=1)`). Root cause: **Master's contract under-declared the write surface**, not executor conduct. **USER RATIFIED 2026-08-21.** The contract was **not** retroactively edited |
| **USER final decision** | **`APPROVE`** — task accepted, MIF-2 ratified |
| **Affected components** | `resolve_stage1.py` · `annotate_run.py` · `annotate.py` · tests · `stage1_hyperparameters.json` · `stage1_encoding_protocol.json` · `variant_registry.json` · new `annotation_provenance.json` · new `tools/declare_annotation_provenance.py` · `docs/graph/README.md:270` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · approved 2026-08-21 |

**Codex findings retained, not deleted because fixed** (user instruction): a JSON-`null` sidecar could be re-queued; unchanged `prompt_version` with changed content could leave provenance stale. Both fixed with regression tests. Full record in `CODEX_REVIEW.md`.

**Explicitly NOT resolved:** `D0-003`. The 3 legacy-v1 residuals remain unresolved and are labelled as such in the registry. Nothing claims otherwise.

**Does NOT unblock `D1`.** Chain: `D2a USER_APPROVED → D10 final USER REVIEW with AC-1 evidence → D10 USER_APPROVED → D1_n06-reencode`.

**Open follow-ups:** F-2 `sidecar_path()` uid validation (pre-existing, LOW, unassigned) · F-3 Master under-declared `tools/` scope for the registry tool, same class as MIF-2 · F-4 TASK §12.1's AC-1.a snippet is stale, now marked SUPERSEDED · F-5 Master holds no pre-task corpus fingerprint.

---

### `DL-002` — Stage 1 encoding contract implementation

| | |
|---|---|
| **Source** | `D10_stage1-encoding-contract` |
| **Issue / Finding** | The cache-validity blocker (a resumed n06 would build a two-distribution gallery), the unimplemented ratified template, and four open annotation-pipeline gaps. Plus **FIND-8**, discovered at Master review: `annotate_run.is_complete()` keys on `annotation_contract`, which **no** existing record carries, so a bare invocation would queue **45,955** records for re-annotation |
| **Evidence references** | `encode_text_image.py:73-83`, `:86-108` · `annotate_run.py:98`, `:250` · `resolve_stage1.py:243` · `tools/preflight_stage1_text.py` (run by Master) · `CODEX_REVIEW.md` (2 rounds, 12 findings) · `HANDOFF.md` |
| **Decision / Resolution** | Ratify the implementation in principle: serializer, `text_serialization_id` / cache validity, `load_protocol` mismatch rejection, Stage 1 text pre-flight, >77 true-token hard gate, P-1…P-5, annotation contract versioning, and 3 user-authorised manual translations. **Acceptance withheld** pending `AC-1` |
| **Authority classification** | B-1/B-3 mechanisms = IMPLEMENTATION CHOICE · late template binding = **bug fix** · P-1…P-5 = IMPLEMENTATION CHOICE, **USER DECISION** to ratify the scope extension · 3 translations = **DEVIATION**, user-authorised · legacy-v3 retention = **USER DECISION** |
| **Acceptance condition** | **`AC-1`** — absent explicit force or migration intent, the accepted legacy-v3 corpus must not be automatically treated by `annotate_run` as requiring re-annotation. Sub-conditions AC-1.a…AC-1.e in the brief §7.0. Assigned to `D2a` |
| **USER final decision** | `MODIFY` 2026-08-21 (acceptance withheld pending `AC-1`) → **`APPROVE` 2026-08-21** after `D2a` (`DL-003`) demonstrated `AC-1` |
| **Affected components** | `resolve_stage1.py` · `encode_text_image.py` · `annotate.py` · `annotate_run.py` · `tools/preflight_stage1_text.py` · 3 annotation records (backed up) · `D1`, `D2a`, `D3`, `D4`, `D7` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · MODIFY 2026-08-21 · **APPROVED 2026-08-21** |

**`AC-1` — CLEARED.** All five sub-conditions verified by Master through the **production** work-list path `build_work_list()` (`annotate_run.py:439`), not through the superseded `is_complete()` predicate: bare todo TOTAL **0** · legacy-v3 queued **0** · residuals queued **0** · `--force` **45,955** · histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}` · no fake v4 `annotation_contract`.

**`G-7` is NOT ratified by this acceptance — it remains independently OPEN.** The user declined to rule on it in this decision, and D10's FINAL ACCEPTED status must **not** be read as approving it:

| | Item | Impact today |
|---|---|---|
| G-7 §5 | Four in-code comment corrections outside R-1/R-2/R-3 | zero behavioural |
| G-7 §6 | Whether the `math.isfinite()` guard belongs in `serialize_annotation()` rather than only in the pre-flight | 0 records |

**Other items explicitly not blocking, carried forward:** `D0-003` **UNRESOLVED** · MIF-D10-3 routed to `D3`/`D4` (re-verified 2026-08-21: `stage1.py:110` and `gallery_index.py:215` load NPZ directly; neither file mentions `text_serialization`) · **F-2** `sidecar_path()` uid validation, LOCAL, pre-existing · template retrieval impact **UNKNOWN**.

**Unblocks:** `D1_n06-reencode`.

**Briefs:** `USER_REVIEW.md` (Rev 2), `USER_REVIEW_FINAL.md`

---

### `DL-001` — Stage 1 text serialization design (U-15)

| | |
|---|---|
| **Source** | `D0-008_stage1-text-template` |
| **Issue / Finding** | The paper specifies no text serialization format (PAPER FACT as to silence). The recorded protocol artifact describes a different template from the one the encoder runs. The running serializer rendered 161 real dimensions as `0 centimetres`, emitted 3,643 ungrammatical articles, and knowingly encoded 1 over-length record. Four in-code justifications did not describe the code |
| **Evidence references** | `2methdology.tex:28` and caption `:24` · Figure 2 (`data-preprocess.png`) · `resolve_stage1.py:96-100`, `:102-115`, `:116-128`, `:162` · `encode_text_image.py:73-83`, `:86-108`, `:194`, `:233` · full-corpus scan of 45,952 v3 records · decision file §6, §9, §10, §11 |
| **Decision / Resolution** | Ratify the design in §11.3: template form, field set and order, centimetres, `width → length → height`, E-1 (dimension precision), E-2 (article removal), E-3 (re-annotate the CJK record), S-1 (uniform formatter, no threshold), S-2 (category capitalisation), R-3 (**delete** the unreachable placement entry), and omission of `synset` / `volume` / `mass` |
| **Authority classification** | Field **set** = PAPER FACT · centimetres = INFERENCE as to intent, OBSERVED DATA as to this corpus · `width,length,height` = INFERENCE from Figure 2 · everything else = **IMPLEMENTATION CHOICE** · E-1/E-2/E-3/S-1/S-2 = **USER DECISION** |
| **Binding user modification** | Omission of `synset` / `volume` / `mass` is an **IMPLEMENTATION CHOICE** with **UNKNOWN retrieval impact**. Must **not** be stated as a PAPER FACT or described as proven redundant. The redundancy argument is WITHDRAWN (Codex C-6) |
| **USER final decision** | `MODIFY` → **`APPROVE`** |
| **Affected components** | `resolve_stage1.py` · `stage1_encoding_protocol.json` · every Stage 1 text embedding · `D10`, `D1`, `D2`, `D3`, `D4`, `D7` · every text-conditioned column of Table 1 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | proposed 2026-08-21 · approved 2026-08-21 |

**Explicitly outside this decision:** the n06 cache completion/validity gate (decision file §11.2, B-1…B-4). It is an execution question owned by `D10_stage1-encoding-contract`. This approval neither clears nor waives it.

**Outstanding follow-ups:** FU-2…FU-5 (`D10`) · FU-6 (Master, done at approval) · FU-7 (Master → `D0-003`) · FU-8, FU-9 (deferred). FU-1 is not carried by this decision.

**Integrated at approval:** `workflow/CONTEXT.md` §5 (FU-6), `workflow/MASTER.md`, `workflow/INDEX.md`, this ledger.

**Not yet implemented:** the ratified template does not yet exist in `resolve_stage1.py`. That is FU-2, owned by `D10`.

---

## Migration backlog

**Migration backlog is now empty.** Both pre-gate items — `D0-008` and `D10` — have passed through the gate and reached `USER_APPROVED`.

`workflow/WORKFLOW.md` §19 governs work completed **before** the user review gate was adopted.

Such work is **not** re-run: evidence surveys, Codex reviews, Claude verification of Codex findings, tests, and implementations all stand as produced. What is owed is the tail of the flow:

```
existing completed artifacts
→ Master integration review
→ USER REVIEW BRIEF
→ USER final decision
```

Any acceptance Master recorded before the gate existed is **reclassified as a MASTER RECOMMENDATION** and carries status `AWAITING_USER_REVIEW` here.

| Decision / Task | Pre-gate state | Ledger status | What is owed |
|---|---|---|---|
| ~~`D0-008_stage1-text-template`~~ | Master recorded `ACCEPT WITH FOLLOW-UP` pre-gate | **CLEARED 2026-08-21** → `DL-001`, `USER_APPROVED` | Migration complete. Brief delivered, user returned MODIFY then APPROVE |
| ~~`D10_stage1-encoding-contract`~~ | Executed against D0-008's pre-gate acceptance | **CLEARED 2026-08-21** → `DL-002`, `USER_APPROVED` | Migration complete. `AC-1` satisfied by `D2a`; implementation and Codex review were **not** re-run |

Master must not treat a pre-gate acceptance as though the gate had been satisfied.

---

## Maintenance rules

1. Only Master edits this file.
2. A D0 decision file's status and this ledger must agree. If they disagree, the ledger is the project-level record and the decision file is corrected.
3. `USER_APPROVED` is written only after the user's actual approval, with its date.
4. Superseded entries are **marked**, never deleted, and must name their replacement.
5. When a decision is approved, Master integrates it and records which files were updated.
6. If a decision is later found to rest on a mistaken finding, add a new entry that supersedes it. Do not edit the original's conclusion.
