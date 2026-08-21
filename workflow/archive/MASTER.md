# MetaFindV1 Master Control

> Maintained by the Master / Orchestrator.
> Project-level state, dependencies, assignments, integration status, next actions.
> Not scientific authority. Primary evidence, audit documents, implementation contracts, and runtime artifacts override this file.

**Initialized:** 2026-08-20
**Repository state at initialization:** `4a4ebbe` + untracked `workflow/`, `_workflow_old_20260820/`; deleted `TASKS.md`, `主線.md`, `支線任務.md`

> **⚠ STALE — do not act on §4, §6, §11 or §12 without checking.** This file still describes the
> pre-Block D-task world of 2026-08-20/21. Since then: the project moved to **two blocks**
> (`workflow/BLOCKS.md`), the engineering skills were integrated (`workflow/SKILLS.md`), `D0-003`
> was resolved (`DL-006`), **every old-model artifact was deleted** — `annotations/` and
> `embeddings/` are now **empty** — and `D15`'s findings verified n03/n04 against upstream.
> **Current state: `workflow/MASTER_INITIALIZATION_REPORT.md` and
> `workflow/MASTER_SESSION_HANDOFF.md`.** Known-wrong lines here include the n06 count (5,276),
> the test count (442), and `MASTER.md:273`'s claim that τ = 0.5 has no code path — `D2a` fixed
> that on 2026-08-21. Rewriting this file is queued, not done.

**Last corrected:** 2026-08-21 (Master re-initialization audit). `git HEAD 468bbac` + 13 modified / 7 untracked, **none committed**.
Every count in §4, §6, §11 and §12 was re-measured against disk on that date. Sections that had gone stale are marked
**[CORRECTED 2026-08-21]**; the superseded figure is stated rather than deleted, so the drift stays auditable.

---

## 1. Project Goal

Reproduce the MetaFind paper (dual-tower multimodal 3D asset retrieval with layout context) with evidence-backed correspondence between the published method and this implementation.

Target deliverables:

- Stage 1 object-level contrastive pretraining (paper 2.6, Eq. 5)
- Stage 2 layout-conditioned training with ESSGNN (paper 2.5, Eq. 3 / appendix Eq. 13, Eq. 6/7)
- Gallery index + retrieval evaluation reproducing Table 1
- Scene-composition evaluation reproducing Table 2
- Explicit registry of PAPER FACT / IMPLEMENTATION CHOICE / DEVIATION / UNKNOWN throughout

Working code alone is not the deliverable. Traceable reproduction fidelity is.

---

## 2. Current Project Phase **[CORRECTED 2026-08-21]**

**Phase: n05 annotation corpus is being rebuilt. Everything downstream is halted behind it.**

> Superseded wording: *"Stage 1 data preparation — INCOMPLETE … n06 has produced 5,276 of the expected embeddings."*
> That described the project before the n05 category defect was found on 2026-08-21.

Three things must not be collapsed into one statement:

- **Raw / upstream preprocessing: COMPLETE.** Download, point clouds, renders, scene graphs, ProcTHOR modalities, semantic edges, and the Stage 2 protocol resolvers have all run and their artifacts are on disk.
- **The n05 annotation corpus is DEFECTIVE and is being replaced.** Qwen was asked to *identify* objects it could not read and collapsed onto high-frequency priors; because `build_prompt` derives dimensions and placement from the category, a wrong category is a **wrong record, not a wrong field**. Diagnosis: `workflow/MIF_n05_diagnosis.md`. Replacement design: `workflow/n05_v5_design.md`. Executing task: **`D14_n05-v5-reannotate`, ACTIVE**, Phase 1 complete, holding at the Phase 2 gate.
- **Stage 1 prerequisite data preparation: HALTED, not merely incomplete.** `D1_n06-reencode` was **STOPPED** at `2026-08-21T14:15:48` after **20,053 / 45,952** `.npz`. Those embeddings encode the defective categories and are **invalid**, not partial progress. n09 has never run, so `splits.json`, `eval_protocols.json` and `stage1_protocol.json` are all absent.

Stage 1 has never trained. `data/outputs/checkpoints/` is empty. Everything from Stage 1 training onward is unexecuted.

---

## 3. Overall Pipeline

```
n01 env ─ n02 download ─ n03 pointclouds ─ G2 ─ n04 renders ─ n05 annotate
                                                                  │
                                    ┌─────────────────────────────┤
                                    ▼                             ▼
                          n05b resolve_stage1            n06 encode_text_image
                                    │                             │
                                    └──────────► n09 build_splits ◄┘
                                                 (+ G3 object corpus)
                                                       │
                                                       ▼
                                              n10 train_stage1
                                                       │
                                    ┌──────────────────┴────────────┐
                                    ▼                               ▼
                          n11 gallery_index_staging          n15 eval_retrieval
                          G4 gallery_freeze                  (Table 1)
                          n12 promote_index
                                    │
   n07 scene_graphs ─ n07b modalities ─ n08 semantic_edges ─ n09b/n09c ─ G6
                                    │
                                    ▼
                          n11b stage2_gallery_index ─ n13 train_stage2
                                                            │
                                                            ▼
                          n15a/n15b/G7/n15c ─ n16 compose ─ n17 judge  (Table 2)
                                                            │
                                                            ▼
                          n18/n19 ablations ─ n20 aggregate ─ n21 compare ─ G5 ─ n22
```

Node registry: `docs/graph/node_registry.yaml` (38 nodes, 7 gates).
Structural checker: `tools/check_graph.py` — **2275 checks, all pass** (verified 2026-08-20).

---

## 4. Current Status

Status classification is deliberately strict. Artifacts on disk are OBSERVED DATA. Existing code is OBSERVED IMPLEMENTATION. Neither is evidence of paper fidelity.

### DONE — artifact verified on disk (2026-08-20)

| Node | Evidence |
|---|---|
| n02–n04 | `pointclouds_index.jsonl` 46,052 · `renders_index.jsonl` 45,955 |
| n05 annotate | `data/outputs/annotations/` = 45,955 files: **45,952 `prompt_version:3` + 3 `prompt_version:1`**. **[CORRECTED 2026-08-21] "DONE" here means the files exist. The v3 corpus is semantically DEFECTIVE** — see §2 and `MIF_n05_diagnosis.md`. It is being replaced by `D14`. Do not treat this row as an accepted corpus |
| n05b resolve_stage1_encoding | `stage1_encoding_protocol.json`, `stage1_hyperparameters.json` (sha256 self-consistent) |
| n07 scene_graphs | `data/outputs/scene_graphs/` = 12,000 files |
| n07b procthor_modalities | `procthor_node_embeddings.{json,npz}`, `procthor_object_text.json` |
| n08 semantic_edges | `sem_edge_cache.json`, `sem_edge_embeddings.npz`, `sem_edge_sentences.jsonl` |
| n09b resolve_stage2_protocol | `stage2_protocol.json`, `essgnn_arch_protocol.json`, `essgnn_edge_protocol.json`, `stage2_positive_map.json` |
| n09c build_scene_splits | `scene_splits.json` |

Repository health, **re-run by Master 2026-08-21** (supersedes the 2026-08-20 figures):

- `python -m pytest tests/ -q` → **582 passed**, 0 failed, 0 skipped, 22 warnings (all τ-deviation warnings, raised from a **test fixture** — the production artifact carries τ = 0.5 per `DL-003`).
- `tools/check_graph.py` → **2275 checks, all pass**.

> **[CORRECTED 2026-08-21] The suite grew 442 → 547 → 582** as `D2a` and then `D14` added coverage. `442` remains the correct record for 2026-08-20 and for every document dated then; it is **not** the current baseline. Use `pytest tests/ -q` with no `--ignore` flag as the standard health check.

These establish that the code executes and is internally consistent. They do **not** establish paper fidelity for any component.

### NOT DONE

| Node | Observed state |
|---|---|
| n06 encode_text_image | **[CORRECTED 2026-08-21] 20,053 `.npz` present** (was 5,276 before `D1` ran), and **all of them are invalid** — they encode the defective v3 categories, and every one is already cache-invalid under `DL-002`'s text-bound `is_complete()`. `D1` was **STOPPED** at `2026-08-21T14:15:48`; nothing was deleted. Expected successful output of a future full run is **45,952** — see the count breakdown below |
| n09 build_splits | **never executed.** `splits.json`, `eval_protocols.json`, `stage1_protocol.json` all absent |
| n10 train_stage1 | **never executed.** `data/outputs/checkpoints/` is empty |
| n10b post_stage1_encode | **not implemented** (no `IMPLEMENTS-NODE` marker anywhere) |
| n11 / n11b / n12 gallery index | implemented (`train/gallery_index.py`), never executed |
| n13 train_stage2 | implemented (`train/stage2.py`), never executed, no `IMPLEMENTS-NODE` marker |
| n14 equivariance_probe | not implemented |
| n15 eval_retrieval | **zero implementation code.** Spec only |
| n15a / n15b / n15c / n16 / n17 | **zero implementation code.** Spec only |
| n18 – n22 | not implemented |

#### n06 count breakdown — do not conflate these three numbers

| Quantity | Value | Basis |
|---|---|---|
| annotation files on disk | **45,955** | `ls data/outputs/annotations` |
| valid v3 annotations | **45,952** | `prompt_version == 3` |
| old v1 residuals | **3** | `prompt_version == 1`, listed under D0-003 in §8 |
| uids n06 *attempts* | **45,955** | `encode_text_image.py:177-179` — `annotations` glob ∩ `renders_index.jsonl` |
| **expected successful `.npz`** | **45,952** | the 3 v1 records raise `KeyError: 'width'` in `serialize_annotation()` and are quarantined without output (`encode_text_image.py:213-221`), verified by direct call |
| expected quarantine records | **3** | same |

This expected-output figure is a **runtime property of n06 and holds regardless of how D0-003 is decided.** D0-003 governs something different: whether those 3 uids are admitted by n09 into the splits and gallery. If they are, Stage 1 crashes — see D0-003 in §8.

### ACTIVE / READY **[CORRECTED 2026-08-21]**

| ID | Work package | State |
|---|---|---|
| **`D14_n05-v5-reannotate`** | Category-anchored re-annotation | **`ACTIVE`.** Approved 2026-08-21 (U-10 authorises Phase 1 + Phase 2 **only**). **Phase 1 COMPLETE** — v5 prompt, validator v3, `identity_confirmed`, 1,156-entry synset lookup, contract `metafind_annot_v5@f5b2bfb2e5f61fe7`, 582 tests pass. **Holding at the Phase 2 gate.** Five escalations `P-1`…`P-5` filed; `IC-1` and `IC-2` **await a Master or user ruling** |
| `D15_n03-n04-code-audit` | Read-only audit of n03/n04 against `/home/kyzen/upstream/ULIP @ 95d480fe` | **`PLANNED`** — contract written, not approved. **§6 of its contract contains a factual error corrected 2026-08-21**; see §11 R1 |
| `D16_gates-g1-g2` | First gates G1/G2 + the `gate_record` schema | **`BLOCKED`** on `D15` and on user resolution of `OQ-1` / `OQ-2` |
| ~~`D1_n06-reencode`~~ | Full n06 re-encode | **`STOPPED` 2026-08-21T14:15:48**, 20,053 npz, nothing deleted. **Must re-run after `D14`.** Its `TASK.md` preconditions (5,276 npz, 547 tests) are stale and marked as such |
| `D9_paper-figures-audit` | Read all 38 extracted paper figures against the U-register | **`READY`.** Zero cost, no GPU, read-only on `data/`, writes only `docs/` |

### BLOCKED

| ID | Work package | Blocked by |
|---|---|---|
| `D10_stage1-encoding-contract` | Clear the cache-validity BLOCKER; implement the ratified template; re-annotate the truncated record; add the pre-flight gate | **READY** — approved 2026-08-21, contract written, awaiting approval to start the conversation |
| `D1_n06-reencode` | Full text/image re-encode | **D10** (D0-008 is accepted but does not unblock D1 — follow-up F-1 does) |
| `D2_stage1-prereq` | Apply C-001, re-run n05b, run n09, verify G3 | D0-002, D0-003, D10 |
| `D3_stage1-train` | Stage 1 smoke → full training | D1, D2 (hard: D0-003, or the DataLoader raises `FileNotFoundError`; plus D0-005 if D0-002 selects `fully_separate`) |
| `D4_gallery-index` | n11 → G4 → n12 | D3 |
| `D5_stage2-prereq` | ESSGNN axis + n08 node text + n11b | D0-004, D0-006, D4 |
| `D6_stage2-train` | n13 | D5 |
| `D7_eval-table1` | Implement and run n15_eval_retrieval | D4 |
| `D8_eval-table2` | n15a/b/c, n16, n17 | D0-007, D6 |

### DECISION REQUIRED

**[CORRECTED 2026-08-21]** Nine registered candidates, two resolved. Full registry: `workflow/INDEX.md` §Decision Queue. Critical-path summary:

| ID | Question | Blocks |
|---|---|---|
| **D0-010** | **How the Objaverse-LVIS ground-truth category enters n05** — prompt hint / hard value / cross-check / record-only; and whether the GPT-4o → Qwen substitution is a material cause. **Decision file exists; §6–§11 are EMPTY — no research has been done** | **`D14` Phase 3, therefore everything downstream** |
| D0-002 | U-16 tower sharing mode | D2 → D3, and Stage 2 feasibility |
| D0-003 | The 3 `prompt_version:1` annotations — admit, drop, or re-annotate | D2 (corpus denominator) and **D3 (hard: `stage1.py:109` raises `FileNotFoundError` if admitted)** |
| D0-004 | ESSGNN `coord_feat` / `architecture_family` coupling | D5, ablation design |
| D0-005 | `build_model()` bypasses `Stage1RuntimeConfig` | D3, conditional on D0-002 |
| D0-006 | n08 node-text information collapse | D5 |
| D0-007 | Table 2 evaluation protocol (200 scenes, 1–5 scale) | D8 |
| ~~D0-009~~ | ~~MetaFind §2.5 `f_x → R³`~~ — **`USER_APPROVED` 2026-08-21 (`DL-004`).** Verdict `PAPER-AMBIGUOUS`; Option A adopted as a USER-RATIFIED IMPLEMENTATION CHOICE | resolved |
| ~~D0-008~~ | ~~Ratify the Stage 1 text serialization template (U-15)~~ — **`USER_APPROVED` 2026-08-21** (`DL-001`) | resolved. Implementation is FU-2, owned by D10 |

### Implementation Corrections — not D0 decisions

Known mismatches between an established requirement and the current artifact/code. These need execution, not research adjudication. They are carried out inside the task named below.

| ID | Correction | Established requirement | Owner task |
|---|---|---|---|
| C-001 | `stage1_hyperparameters.json` records `init_temperature: 0.07`, `learnable_temperature: true` | **PAPER FACT: τ = 0.5** (`3experiments.tex:15`) | D2 |
| C-002 | `stage1_encoding_protocol.json` records the v1 metre template; `serialize_annotation()` produces the v3 centimetre template | The artifact must describe runtime truth | D1 (refresh) / D2 (re-run n05b) |

---

## 5. Dependency / Execution Order **[CORRECTED 2026-08-21]**

> Superseded: the chain that began `D0-008 → D10 → D1`. All three are `USER_APPROVED` and `D1` has since been stopped. The head of the critical path is now the annotation corpus.

```
R1 orientation verification (read-only, cheap) ──► must land before D16 writes G2
                                                   and before D3 spends a training run
D0-010 (category source) ──┐
                           ├──► D14 Phase 2 (300-500 sample) ──► [HARD HOLD: user's go only]
D14 Phase 1 COMPLETE ──────┘                                            │
                                                                        ▼
                                                   D14 Phase 3 (full re-annotation)
                                                                        │
                                                                        ▼
                                                   D1 (n06 re-encode, 45,952 npz)
                                                                        │
D0-002 U-16 ─────────┐                                                  │
D0-003 stragglers ───┴──► D2 (n05b + n09 + G3)  ────────────────────────┴─► D3
                          (gate n09 ONLY - independent of n06)             │
                                                                          ▼
                                                    D4 (gallery index n11/G4/n12)
                                                         │              │
                                    D0-004 ──► D5 (Stage 2 prereq)      └──► D7 (Table 1 eval)
                                    D0-006 ──►        │
                                                      ▼
                                               D6 (Stage 2 train)
                                                      │
                                    D0-007 ──►        ▼
                                               D8 (Table 2 eval)

D9 (figures audit) · D15 (n03/n04 audit) · D0-002 / D0-003 / D0-010 research
        └── all PARALLEL SAFE with the chain above; none writes data/outputs/
```

Critical path: **R1 → D0-010 → D14 Phase 2 → [user gate] → D14 Phase 3 → D1 → D2 → D3 → D4 → {D7, D5 → D6 → D8}**, with D0-002 and D0-003 required before n09 (D2).

Execution is **sequential by default** for anything that writes `data/outputs/` or holds the GPU. Read-only research and audits may run alongside.

Ordering constraint: D2 must re-run n05b to refresh `stage1_encoding_protocol.json`, and `resolve_stage1.py:443-444` rewrites `stage1_hyperparameters.json` in the same call. So the template ratification (D0-008) and the τ correction (C-001) must both be settled before n05b runs, or n05b runs twice.

---

## 6. Active Assignment **[CORRECTED 2026-08-21]**

> Superseded: this section described `D2a` and `D0-009` as the live pair. Both closed on 2026-08-21 (`DL-003`, `DL-004`).

**One ACTIVE work item: `D14_n05-v5-reannotate`.**

| | `D14_n05-v5-reannotate` |
|---|---|
| Role | `D1+` execution |
| Status | **`ACTIVE`.** Phase 1 COMPLETE, holding at the Phase 2 gate |
| Carries | v5 category-anchored annotation · LVIS anchor · `identity_confirmed` · exact mesh proportions · synset lookup · model change to `Qwen3.8-27B` (U-6) |
| Writes | `annotate.py`, `annotate_run.py`, `tests/test_annotate.py`, `lvis_synsets.json`, its own task dir. Phase 3 only: `data/outputs/annotations/**` |
| Authorisation | **U-10 covers Phase 1 + Phase 2 only.** `R-A` makes the Phase 3 gate absolute — good Phase 2 numbers are **not** permission |

**Owed to D14 by Master, still outstanding:**

| | Item | State |
|---|---|---|
| `P-1` | Amend `TASK.md` §7 for U-6, and qualify the unverified "GPT-4o is unavailable" premise | **GRANTED**, applied in `D14/TASK.md` §7 |
| `P-2` | Split deviation `D-2` into `D-2` (annotation) + `D-8` (scene judge) | **EXECUTED** across 5 files; ledger `DL-005` is `AWAITING_USER_REVIEW` |
| `P-3` | Two integers in `docs/graph/README.md` | **GRANTED**; `check_graph.py` back to 2275 all-pass |
| `P-4` | Correct the `/mnt/data1` claim in `CLAUDE.md` §9 / `CONTEXT.md` §9 | `CONTEXT.md` **DONE**. `CLAUDE.md` is project instruction — **user only** |
| `P-5` | Prepare the `DL-003` / AC-1 amendment for Phase 3 | **DRAFTED** as `DL-003-A1`, `PREPARED, NOT IN FORCE` |
| `IC-1` | Record `exact` / `refined` / `divergent` category relations and reject nothing | **UNRULED — owed to the user** |
| `IC-2` | `synset` follows the LVIS anchor, not the model's refined category | **UNRULED — owed to the user** |

**Concurrent, permitted, none writing `data/outputs/`:** `D0-010` research · `D0-002` / `D0-003` research · `D15` (once its §6 error is fixed) · R1 orientation verification.

**Scope wall.** No item may modify another's scope. A finding bearing on another is a `MASTER-IMPACTING FINDING`, reported and not acted on.

`D0-008_stage1-text-template` was returned `RECOMMENDED` by D0 on 2026-08-21 and Master recorded `ACCEPT WITH FOLLOW-UP` the same day — **before the user review gate was adopted.**

**Migration backlog is empty.** Both pre-gate items reached `USER_APPROVED` on 2026-08-21: `D0-008` (`DL-001`) and `D10` (`DL-002`).

Migration completed 2026-08-21. Master delivered the USER REVIEW BRIEF (`workflow/decisions/D0-008_USER_REVIEW.md`); the user returned `MODIFY`, Master applied the corrections, and the user returned **`APPROVE`**.

**Status: `USER_APPROVED`. FINAL ACCEPTED.** Ledger entry `DL-001`.

Its investigation, Codex adversarial review, and Claude's verification were **not** re-run.

`D10_stage1-encoding-contract` executed against that pre-gate acceptance. Master reviewed it on 2026-08-21 and recommended `ACCEPT WITH FOLLOW-UP`; the user returned **`MODIFY`**.

**`D10` is `USER_APPROVED` 2026-08-21 (`DL-002`). FINAL ACCEPTED.** `AC-1` — its sole blocking condition — was satisfied by `D2a` and verified by Master through the production work-list path.

**`G-7` is NOT ratified by that acceptance.** It stays independently OPEN: D10's Decisions §5 (four comment corrections, zero behavioural impact) and §6 (`math.isfinite()` guard placement, 0 records affected). D10 being FINAL must not be read as approving them.

**`D1_n06-reencode` is now UNBLOCKED.**

Decision-state files: `workflow/DECISION_LEDGER.md` (project-level record), `workflow/USER_REVIEW_TEMPLATE.md` (brief format).

`D10_stage1-encoding-contract` — execution `COMPLETE`, integration `AWAITING_USER_REVIEW / MODIFIED`, blocked on `AC-1`, which `D2a` carries.

---

## 7. Pending Assignments

See `workflow/INDEX.md`.

---

## 8. D0 Research / Architecture Decisions

Seven candidates registered. **One is open: `D0-008`** — approved by the user on 2026-08-20 and created at `workflow/decisions/D0-008_stage1-text-template.md`. The remaining six have no decision file.

τ was previously registered here as `D0-001`; it is now correction `C-001` and is documented immediately below.

The old workflow (`_workflow_old_20260820/`) recorded D-α through D-η. Those labels are **not** carried forward. Each candidate below was independently re-derived from current repository evidence during this initialization.

**C-001 — Temperature τ. Reclassified: this is not a D0 decision.**

The primary evidence is sufficient and uncontested, so there is no two-way research question to adjudicate.

- **PAPER FACT — τ = 0.5.** `docs/paper/metafind_source/3experiments.tex:15`: "The temperature is 0.5 for all experiments." Verified verbatim. A grep for `emperature` across `docs/paper/metafind_source/*.tex` returns three hits: `2methdology.tex:79` and `:99` both say only "τ is a temperature hyperparameter" and give no value; `3experiments.tex:15` gives 0.5. **No conflicting statement exists in the paper source.**
- **INFERENCE (strongly supported) — `learnable_temperature: false`.** The paper states a fixed value held across all experiments. A learnable τ initialised at 0.5 would not remain 0.5, so a learnable temperature cannot satisfy the sentence. The paper does not use the word "learnable"; this is inference, not paper fact.
- ~~**ARTIFACT MISMATCH.**~~ **RESOLVED by `D2a`, 2026-08-21.** `stage1_hyperparameters.json` now records `init_temperature: 0.5`, `learnable_temperature: false`, `decided_by: "D2a_stage1-protocol-refresh"`. Re-verified 2026-08-21.
- ~~**IMPLEMENTATION GAP.**~~ **RESOLVED by `D2a`, 2026-08-21.** The paragraph below is retained as a record of the original finding; **its concluding claim is now false.**
  > `metafind/models/resolve_stage1.py:197-211` hardcodes `init_temperature: 0.07` and `learnable_temperature: True` in `DEFAULT_HYPERPARAMETERS` … **There is currently no supported way to produce τ = 0.5 through n05b.**

  Current state, verified 2026-08-21: `resolve_stage1.py:272-273` defaults to `init_temperature: 0.5`, `learnable_temperature: False`, and `stage1.py:335-336` reads those values out of the artifact. **The training path uses τ = 0.5 fixed.** The `tau deviates from the paper` warnings still visible in `pytest` output come from *tests* constructing `ContrastiveConfig` with library defaults (0.07 / learnable), **not** from the training path — do not read them as evidence that the gap is open. Evidence: `workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md`, FIND-13.
- **OBSERVED IMPLEMENTATION.** `metafind/models/losses.py:70` already defines `PAPER_TAU = 0.5` and warns on every deviating construction (`losses.py:114-120`). All 22 warnings in the test run come from this.

**What is actually required:** an implementation correction inside `D2_stage1-prereq` — add an override path for the hyperparameters, set τ = 0.5, re-run n05b, and re-run n09 so `stage1_protocol.json` hashes the corrected artifact. No D0 investigation is needed.

**The one thing reserved for the user:** whether the mainline run uses the paper-faithful τ = 0.5, or deliberately departs from it. The default is τ = 0.5. Any other value is a **DEVIATION** that must be registered and must be stated wherever results are compared with the paper's tables. If τ = 0.5 trains poorly, that is a reportable finding, not grounds for a silent substitution.

**D0-002 — U-16 tower sharing**

- Conflict between the architecture figure (`MetaFind.drawio.png`, labelled `ULIP-2 (Shared)`, one Fusion Layer) and paper 2.6 ("Only the query-side fusion layer and the ESSGNN module are updated; the gallery encoder is frozen").
- OBSERVED IMPLEMENTATION: `metafind/models/dual_tower.py:315-321` raises on `freeze_gallery()` under `fully_shared`, explicitly on the grounds that the two readings cannot both hold.
- OBSERVED IMPLEMENTATION: `metafind/data/splits.py:80` defaults `tower_sharing = "shared_backbone_separate_fusion"` and records that as an implementation choice.
- Consequence: written into `stage1_protocol.json` by n09; determines whether Stage 2 can freeze the gallery at all; conditions D0-005.
- The figure claim was re-verified only as file presence this initialization. The figure has **not** been read image-by-image by Master — D9 exists for that.

**D0-003 — Three `prompt_version:1` annotations**

- OBSERVED DATA: `data/outputs/annotations/` holds 45,952 v3 + 3 v1.
- OBSERVED DATA: `data/outputs/logs/annotations_index.jsonl` has 45,955 entries, so `metafind/data/splits.py:169-171` (`pointclouds ∩ renders ∩ annotations`) admits **45,955** — the 3 v1 records included.
- OBSERVED DATA: the three are `a397b648d6eb48d7909d1ee11235e78f`, `6c7db00cc164467ebac356a5ca67368b`, `8a0192eee6fb4140bb3e9696b3dbae5a`. They carry the v1 schema — `dimensions: {length_m, width_m, height_m}` and `placement_constraints: [...]` — not the v3 flat fields.
- **VERIFIED RUNTIME FACT:** `serialize_annotation()` on one of these raises `KeyError: 'width'`. n06 catches it (`encode_text_image.py:213-221`), writes a quarantine record, and produces no `.npz`. So n06 cannot and will not embed them.
- **VERIFIED RUNTIME FACT — this is a hard blocker for D3, not only a denominator question.** `metafind/data/splits.py:169-171` admits all 45,955, so n09 would place these three uids into a split. `metafind/train/stage1.py:109` then does `np.load(paths.EMBEDDINGS / f"{uid}.npz")` with **no existence guard**, so Stage 1 raises `FileNotFoundError` the moment the DataLoader reaches one of them.
- Consequence: the corpus denominator, both `eval_protocols.json` gallery sizes, all downstream recall denominators, **and whether Stage 1 can complete an epoch at all**.

**D0-004 — ESSGNN `coord_feat` / `architecture_family` coupling**

- OBSERVED IMPLEMENTATION: `metafind/models/essgnn.py:191-195` derives `coord_feat` from `architecture_family`; `essgnn.py:491-503` raises if `appendix_shared_msg` is paired with `coord_feat != "current"`.
- OBSERVED DATA: `essgnn_arch_protocol.json` already records `architecture_family: appendix_shared_msg`, `coord_feat: current`, `decided_by: "Kyzen (2026-08-19)"`. That combination is admitted by the current code, so **Stage 2 training is not blocked by this**.
- Open question: whether the coupling is a defect that prevents isolating `coord_feat` as an ablation axis, given that `coord_feat` is the only axis with paper backing (2.5 Eq. 3 vs appendix Eq. 13) while MLP depth is not mentioned by the paper at all.
- The old workflow's premise that this makes ablation impossible was contradicted by a prior session's experiment. That contradiction is itself unverified this session and is part of what D0-004 must settle.

**D0-005 — `build_model()` bypasses `Stage1RuntimeConfig`**

- OBSERVED IMPLEMENTATION: `metafind/train/stage1.py:309-338` builds `DualTowerConfig` from raw protocol dicts, not from `Stage1RuntimeConfig`; passes the **same `FusionConfig` object** as both `query_fusion` and `gallery_fusion` (`stage1.py:322,325`); `main()` constructs exactly one `ULIPBackbone` (`stage1.py:368`).
- Consequence: `tower_sharing="fully_separate"` cannot be honoured by the trainer as written. Becomes a hard blocker only if D0-002 selects `fully_separate`.
- Secondary: `tests/test_dual_tower.py` exercises a construction path the trainer does not use. Test coverage does not cover the executed path.

**D0-006 — n08 node-text information collapse**

- OBSERVED IMPLEMENTATION (verified): `metafind/data/scene_graphs.py:96-104` — `object_text()` returns `f"a {humanise(category)}"`, category only, and its own docstring registers this as U-12.
- Old-workflow counts (1,467 assetIds collapsing to 93 unique node texts) are **not re-verified this initialization.**
- Registered as a candidate because it is an IMPLEMENTATION CHOICE affecting Stage 2 input quality, not a paper-fidelity defect. Requires re-verification before any n08 rerun.

**D0-007 — Table 2 evaluation protocol**

- U-27: how the 200 evaluation scenes are constructed. Old workflow states 60 I-Design prompts are available in `idesign_source`. **Not re-verified this initialization.**
- Also open: MetaFind Table 2 uses a 1–5 scale, I-Design uses 0–10 — comparability claim needs establishing before D8.
- Far from the critical path. Registered so it is not rediscovered later.

**D0-009 — MetaFind §2.5 `f_x → R³`: how to reproduce it faithfully** · **OPEN — registered 2026-08-21, awaiting user approval to open the D0 conversation**

Formal decision file: `workflow/decisions/D0-009_essgnn-fx-codomain.md`. **Master performed no paper audit** — sections 1–5 are framing and evidence pointers only.

- OBSERVED IMPLEMENTATION: `essgnn.py:311-312` uses a **scalar** `f_x` (`_mlp(..., 1)`), hardcoded with no configuration flag. `essgnn.py:39-44` records the reason as "it is simply an error in the paper", citing the Appendix proof **and the reference EGNN**.
- **The records disagree with each other.** `docs/audit/D_IMPLEMENTATION_FORMULA_CONTRACT.md:124` calls it `[PAPER CONTRADICTION] — C3`; `docs/audit/E_GRAPH_REVALIDATION.md:175` calls C3 `VERIFIED`, settled by `[UPSTREAM]`; `docs/audit/F_CODE_GRAPH_CONSISTENCY.md:27` calls it `CONSISTENT`.
- **Why it is being reopened.** The user's governing principle for this decision is that **upstream is supplementary evidence and may not override MetaFind's main text.** The `[UPSTREAM] settles it` reasoning at `E_GRAPH_REVALIDATION.md:175` is inadmissible as a settlement under that principle, so the question returns to MetaFind's own evidence. It may reach the same conclusion.
- `CONTEXT.md` §5 records the position as a DEVIATION, resting on equivariance figures (`2.2e-16 vs 0.43`) that are **UNVERIFIED** in this repository.
- Out of scope, already decided by the user: `h^l` vs `h^{l+1}`. §2.5's sequential update stands.
- Cheap to settle now: Stage 2 has never run, `checkpoints/` is empty, and no Stage 2 artifact depends on it yet.

**D0-008 — Ratify the Stage 1 text serialization template (U-15)** · **OPEN — assigned to D0**

Formal decision file: `workflow/decisions/D0-008_stage1-text-template.md`. **That file is the authority for this decision from here on**; the summary below is Master's framing only and must not be edited to contradict it.

**Not a new project finding.** `_workflow_old_20260820/任務/A_n06-reencode/TASK.md` already recorded the artifact-versus-code template mismatch (its lines 18, 32-37, and acceptance item 133 — "`stage1_encoding_protocol.json` 的 `text_template` == `resolve_stage1.TEXT_TEMPLATE`"). What this initialization adds is independent re-verification against the current repository, plus the τ interaction below that the old card did not cover.

Scope split:

- The **artifact refresh** (making `stage1_encoding_protocol.json` describe runtime truth) is execution work, tracked as correction **C-002**, carried out in D1/D2. It is not a research question.
- The **open question for D0** is narrower: ratify `TEXT_TEMPLATE` as the recorded Stage 1 IMPLEMENTATION CHOICE, because the code explicitly asks for that sign-off and paper 2.3 supplies no format. This is a thin evidence dossier and a ratification, not a deep investigation.

- OBSERVED IMPLEMENTATION: `metafind/models/resolve_stage1.py:102` carries an explicit in-code marker: `# [U-15, IMPLEMENTATION CHOICE -- CONFIRM BEFORE THE FULL RUN]`. The template has never been formally confirmed.
- OBSERVED IMPLEMENTATION: `TEXT_TEMPLATE` (`resolve_stage1.py:96-100`) is
  `"{description} A {category} made of {materials}, roughly {width:.0f} by {length:.0f} by {height:.0f} centimetres, {placement}."`
- OBSERVED DATA: `data/outputs/stage1_encoding_protocol.json` records a **different** template —
  `"... roughly {length:.2f} by {width:.2f} by {height:.2f} metres, typically placed {placement}."`
  Different unit, different precision, different field order, different placement phrasing.
- OBSERVED IMPLEMENTATION: n06 uses the **code** constant, not the artifact. `encode_text_image.py:194` calls `serialize_annotation(annotation)` and `resolve_stage1.py:278` defaults `template=TEXT_TEMPLATE`. The artifact's `text_template` field is written but never read back.
- Consequence 1: the recorded encoding provenance misdescribes what the encoder actually does. `tools/check_graph.py` does not catch it — 2275 checks pass with the mismatch in place.
- Consequence 2: running D1 now spends ~4 GPU-hours producing embeddings under a template the code itself flags as unconfirmed, with wrong recorded provenance.
- Consequence 3 (**ordering constraint, ties to C-001**): re-running n05b to refresh the protocol also rewrites `stage1_hyperparameters.json` in the same call (`resolve_stage1.py:443-444`). So the template ratification and the τ correction must be settled before n05b is re-run, or n05b has to run twice.
- Question for D0: ratify the current centimetre template — including its field order and the deliberate omission of `synset`, `volume`, and `mass` (`resolve_stage1.py:102-115`) — as the recorded Stage 1 IMPLEMENTATION CHOICE.

---

## 9. Integration Status

**`D0-008_stage1-text-template` — `USER_APPROVED` 2026-08-21. FINAL ACCEPTED.** Ledger `DL-001`. Integrated into `CONTEXT.md` §5 (FU-6), `MASTER.md`, `INDEX.md`, and the Decision Ledger.

**Ratified in design only.** The template is **not yet implemented** in `resolve_stage1.py` (FU-2, owned by D10), and this approval does **not** authorise n06 to run — the cache completion/validity gate is D10's. Master independently re-verified the load-bearing claims before resolution rather than accepting on assertion — verification log in Section 12.1 of the decision file. `CONTEXT.md` §5 and §6 updated per follow-up F-6.

`workflow/tasks/` still contains templates only; no D1+ execution task has run.

Prior work under `_workflow_old_20260820/` was used as a **navigation aid only**. Every claim promoted into this file, `CONTEXT.md`, or `INDEX.md` was re-verified against the current repository this session, or is explicitly marked as unverified.

Corrections found against the old workflow during re-verification:

| Old claim | Verified value |
|---|---|
| 39 extracted paper figures | **38** (metafind 6, ulip2 15, egnn 8, idesign 9) |
| 42 entries in the U-register | **35** distinct `U-*` identifiers appear across `docs/graph/*.md`. Discrepancy unresolved |
| n06 target = 45,952 embeddings | **Confirmed, by a different route than the old card gave.** n06 *attempts* `annotations ∩ renders_index` = 45,955 (`encode_text_image.py:177-179`), but the 3 v1 records raise `KeyError: 'width'` in `serialize_annotation()` and are quarantined without producing a `.npz` (`encode_text_image.py:213-221`). Expected **successful output = 45,952 `.npz` + 3 quarantine records**. See §4 note |
| Branch C (n09) depends on Branch A (n06) | **False.** `splits.py:169-171` reads the three index files, never the embeddings. n09 is independent of n06 |

---

## 10. Review Status

### Task-level Codex Reviews

| Work | Reviewer | Rounds | Outcome | Master's verification |
|---|---|---|---|---|
| `D0-008_stage1-text-template` | Codex `gpt-5.6-sol`, xhigh, read-only sandbox | 2 (round 1 budget-exhausted, honestly not counted; round 2 complete) | `BLOCKED BY UNKNOWN` / REJECT-as-written, 13 findings | 11 CONFIRMED or PARTIALLY CONFIRMED, 2 rejected/reduced with stated reasons. Codex surfaced the project's critical blocker (MIF-4), which D0's own scope had missed |

### Milestone / Integration Reviews

None.

### User Review Gate status

| Item | Master recommendation | User decision | Ledger status |
|---|---|---|---|
| `D0-008_stage1-text-template` | ACCEPT WITH FOLLOW-UP (2026-08-21) | **`MODIFY` → `APPROVE`** 2026-08-21 | **`USER_APPROVED`** (`DL-001`) |
| `D10_stage1-encoding-contract` | ACCEPT WITH FOLLOW-UP (2026-08-21) | `MODIFY` → **`APPROVE`** 2026-08-21 | **`USER_APPROVED`** (`DL-002`). `AC-1` cleared. **`G-7` NOT ratified — independently OPEN** |
| `D2a_stage1-protocol-refresh` | ACCEPT WITH FOLLOW-UP (2026-08-21) | **`APPROVE`** 2026-08-21, MIF-2 ratified | **`USER_APPROVED`** (`DL-003`) |

Full record: `workflow/DECISION_LEDGER.md`.

A milestone completion is always material. Task-level user approvals do **not** aggregate into milestone approval — the user approves each milestone as its own decision (`WORKFLOW.md` §14).

Scheduled milestone reviews (per `workflow/WORKFLOW.md` §14):

- after D3 — Stage 1 complete
- after D6 — Stage 2 complete
- after D7 + D8 — evaluation complete
- before D9 release path — final reproduction

---

## 11. Current Blockers **[CORRECTED 2026-08-21]**

### The four live blockers, highest first

**R1 — the point-cloud corpus may be systematically mis-oriented, and the two measurements in existence contradict each other. UNVERIFIED. CRITICAL.**

| Source | Population | Result |
|---|---|---|
| `docs/graph/00_FINDINGS.md` **F21** | **6** assets | same-asset Chamfer median **0.00318** vs cross-asset baseline **0.05880** → "geometry matches" |
| A prior session's measurement, 2026-08-21 (**not recorded in the repository until now**) | **286** overlapping assets from the official ULIP-2 `000-009` shard | median Chamfer **0.0903 at 0°** vs **0.0230 at 180° yaw**; **269/286 (94.1%)** improve under 180°; poor-tier share 47.9% → 2.4% |

Both cannot be right. Untested candidate explanations: (a) the 180° result is an artifact — official `.npy` clouds are **uncentered** (centroid up to 0.588) while ours are **pre-centered**, so omitting `pc_norm` on the official side shifts the whole distribution; (b) F21's n=6 is too small; (c) a genuine frame divergence between `pointclouds.py:118` (per-geometry `apply_transform`) and `renders.py:150` (per-vertex `transform_points` + `scene.apply_transform(fit)`).

A decisive test — encode both orientations through the frozen ULIP-2 checkpoint and compare against the official image embeddings — was written and **timed out without a result**.

**Why it outranks everything else:** if real, all 46,052 point clouds are invalid, therefore Stage 1, therefore both tables — **and no error is raised anywhere in the chain.** It must be settled before `D16` writes G2's criteria and before `D3` spends a training run.

**Consequential correction:** `workflow/tasks/D15_n03-n04-code-audit/TASK.md` §6 stated *"`L2-PC-ULIP-REF` has never been run and the reference clouds are not on disk."* **Both halves are false** — F21 ran a version of it, and the reference shard was extracted on 2026-08-21. Corrected in that file on 2026-08-21.

**R2 — `D14` Phase 3's cost is not evidence-backed. HIGH.**
`Qwen3.8-27B` is **56 GB** at bf16 (18/18 shards present at `/mnt/data1/kyzen/models/Qwen3.8-27B`) on a **32,607 MiB** RTX 5090. Quantization is **required and has never been loaded or benchmarked**. The "~19.6 GPU-h" figure belongs to the **7B** model. Quantization also changes annotation quality, which makes it an **experimental condition**, not an engineering detail.

**R3 — the whole dataset now lives on an SMR drive. HIGH.**
`data → /mnt/data1/kyzen/MetaFind` since 2026-08-21 21:32; `/home/kyzen/data` no longer exists. `/mnt/data1` is `ST4000DM004`, **SMR**, measured `w_await` above **5,000 ms** under mixed small-file write load. n05 writes 45,952 small `.json`; n06 writes 45,952 small `.npz` — the drive's worst case. **Every runtime estimate in this project predates the migration and is unmeasured against it.**

**R4 — Table 1 and Table 2 have zero implementation. HIGH.**
n15, n15a/b/c, n16, n17, n18–n22 are spec only. The reproduction **cannot produce its headline result today**, independent of training, and no task owns it.

### Carried blockers

**B1 — Stage 1 has no usable embedding cache. [CORRECTED: the count was 5,276; it is now 20,053, and all of them are invalid.]**
`data/outputs/embeddings/` holds **20,053** `.npz` against an expected successful output of 45,952. They were produced by the stopped `D1` run against the **defective v3 annotation corpus**, so they are invalid for a second and stronger reason than the template staleness described below. Nothing has been deleted. The original staleness evidence, verified directly rather than inferred from timestamps:

```
cached  (embeddings/000074a3....json)
        "... roughly 0.25 by 0.15 by 0.05 metres, typically placed floor."
current (annotations/000074a3....json, prompt_version 3)
        length 10.0  width 25.0  height 4.0  dimension_unit "cm"
```

Both the template and the underlying values differ. A mixed cache would train Stage 1 on two different text distributions. Full re-encode required — this is `D1_n06-reencode`.

**B2 — Stage 1 cannot start: three protocol files absent.**
`metafind/train/stage1.py:70` raises without `stage1_protocol.json`; `stage1.py:357-360` returns 2 without `splits.json`. n09 has never run. Cleared by `D2_stage1-prereq`.

**B3 — Two open decisions and two corrections gate n09.**
n09 writes `tower_sharing` and the hyperparameter hash into `stage1_protocol.json`. Running it before D0-002 and D0-003 are accepted would bake unresolved choices into the artifact Stage 1 hashes against, and running it before corrections C-001 and C-002 land would hash an artifact that contradicts a paper fact.

**B4 — Table 1 evaluation has no implementation.**
n15_eval_retrieval is spec only. The reproduction cannot produce Table 1 in its current state, independent of training.

**B6 — A resumed n06 would silently build a two-distribution gallery. THE CRITICAL BLOCKER.**
`is_complete()` (`encode_text_image.py:73-83`) compares nothing about the text — only sidecar existence, `encoder_version`, and NPZ existence. A plain re-run skips all 5,276 metre-derived embeddings as "complete" and encodes the rest in centimetres. No error, no warning, identical `text_serialization` on both halves; `gallery_index.py` fingerprints the checkpoint, not the text. **Table 1 would be self-consistent and wrong.** Surfaced by Codex adversarial review during D0-008, confirmed by Master by direct code reading, classified BLOCKER by Kyzen. Exit criteria B-1…B-4 in decision §11.2. Cleared by `D10`. **This is the only blocker in the project capable of producing confident wrong numbers with no error anywhere in the chain.**

**B5 — The recorded Stage 1 encoding protocol does not describe what the encoder does.**
`stage1_encoding_protocol.json` records the v1 metre-based template; `serialize_annotation()` produces the v3 centimetre template, and that is what n06 uses. The code additionally flags the template `CONFIRM BEFORE THE FULL RUN` (`resolve_stage1.py:102`). Re-encoding before this is settled burns ~4 GPU-hours against an unratified template with mis-recorded provenance. **Previously recorded** in `_workflow_old_20260820/任務/A_n06-reencode/TASK.md`; re-verified against the current repository during this initialization. Ratification is done (D0-008 ACCEPTED 2026-08-21). The artifact refresh (C-002) and the template implementation are now `D10`'s scope.

---

## 12. Next Recommended Action **[CORRECTED 2026-08-21]**

> Superseded: *"`D10_stage1-encoding-contract` is READY … then `D1_n06-reencode`."* `D10` is `USER_APPROVED` (`DL-002`) and `D1` has been stopped. Acting on the old text would have restarted a 4-hour encode against a corpus that is about to be replaced.

**Three actions, in this order. Only the third costs meaningful GPU time.**

| # | Action | Cost | Why it is first |
|---|---|---|---|
| **1** | **Verify R1** — finish the frozen-checkpoint orientation test over a proper sample, with `pc_norm` applied identically to both sides | minutes of GPU, read-only | It can invalidate all 46,052 point clouds. Every downstream hour is wasted if it is real, and it raises no error on its own |
| **2** | **Open `D0-010`** — the actual research for §6–§11 of `workflow/decisions/D0-010_n05-category-source.md` | no GPU | It is the declared blocker on an irreversible whole-corpus re-annotation |
| **3** | **`D14` Phase 2** — the stratified 300–500 asset sample | small GPU | It produces the numbers the Phase 3 go/no-go needs. **R2 must be settled first**: the 27B model has never been loaded, and quantization is untested |

**Parallel-safe alongside all three, writing nothing under `data/outputs/`:** `D15` (after its §6 correction) · `D0-002` / `D0-003` research · `D9_paper-figures-audit` · B4 evaluation design.

**Standing prohibitions:**

- **`D14` Phase 3 does not begin without the user's explicit go.** `R-A` is absolute; good Phase 2 numbers are not permission.
- **`D1_n06-reencode` does not restart automatically** after `D14`, and its `TASK.md` preconditions must be refreshed first.
- Master must not start any task conversation without user approval.

---

## Master Operating Rules

The Master is the project manager and integration owner, responsible for:

- maintaining the global project view;
- tracking DONE / ACTIVE / READY / BLOCKED / DECISION REQUIRED;
- maintaining task dependencies and execution order;
- deciding when D0 research/architecture support is required;
- preparing self-contained task contracts;
- receiving and validating task HANDOFFs;
- integrating accepted results;
- requesting milestone-level Codex review;
- proposing the next task to the user.

The Master does not spend its context on long bounded implementation work, prolonged debugging, dataset processing, or deep single-question research. Those belong to D0 or D1+ task conversations.

Execution flow:

1. Master proposes the next task.
2. User approves.
3. A dedicated task conversation executes it.
4. Task performs its own verification.
5. Codex performs independent review.
6. Task writes `HANDOFF.md`.
7. User tells Master the task is finished.
8. Master reads `TASK.md`, `HANDOFF.md`, `CODEX_REVIEW.md`, and re-verifies the load-bearing claims.
9. Master returns a MASTER RECOMMENDATION: ACCEPT / ACCEPT WITH FOLLOW-UP / REWORK / REJECT / BLOCKED.
10. For ACCEPT-class recommendations, Master writes a USER REVIEW BRIEF and records the entry as `AWAITING_USER_REVIEW` in `workflow/DECISION_LEDGER.md`.
11. User returns APPROVE / REJECT / MODIFY / INVESTIGATE MORE.
12. Only after the user's APPROVE does Master update global project state and mark the task DONE.

Execution is sequential by default. Parallel execution requires explicit `PARALLEL SAFE: YES` with verified dependency and filesystem non-conflict.
