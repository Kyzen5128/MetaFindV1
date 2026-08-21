# MetaFindV1 Shared Project Context

> Compressed orientation for D0 / D1 / D2 / D3... task conversations.
> Orientation only. Task authority, scope, evidence, and Definition of Done live in each task's `TASK.md`.
> Master owns this file. Tasks must not edit it.

**Last updated:** 2026-08-20 (Master initialization)

---

## 1. Project Objective

Reproduce the MetaFind paper — dual-tower multimodal 3D asset retrieval with layout context — with evidence-backed correspondence between the published method and this implementation.

The deliverable is a traceable reproduction, not working code. Every research-significant behaviour must be classifiable as PAPER FACT, UPSTREAM FACT, OBSERVED IMPLEMENTATION, OBSERVED DATA, INFERENCE, IMPLEMENTATION CHOICE, DEVIATION, or UNKNOWN.

---

## 2. System / Research Pipeline

Two training stages feeding two evaluation tables.

**Object corpus (Objaverse):**
`n02 download → n03 pointclouds → n04 renders (11 views) → n05 annotate (LLM, 13-field schema) → n05b resolve encoding protocol → n06 encode text+image (frozen CLIP) → n09 build splits (80/20) → n10 Stage 1 contrastive training`

**Layout corpus (ProcTHOR / AI2-THOR):**
`n07 scene graphs → n07b asset modalities → n08 semantic edges (LLM sentences → embeddings) → n09b/n09c stage-2 protocol + scene splits → n11b stage-2 gallery index → n13 Stage 2 training (ESSGNN + layout-conditioned contrastive)`

**Evaluation:**
`n11 gallery staging → G4 freeze → n12 promote → n15 retrieval eval (Table 1)`
`n15a/n15b/G7/n15c → n16 compose scenes → n17 LLM judge (Table 2)`
`n18/n19 ablations → n20 aggregate → n21 compare to paper → G5 → n22 publish`

Node registry: `docs/graph/node_registry.yaml` — 38 nodes, 7 gates.
Structural checker: `tools/check_graph.py` — 2275 checks, all passing as of 2026-08-20.

---

## 3. Current Architecture

**Dual tower** — `metafind/models/dual_tower.py`
`MetaFindDualTower` / `DualTowerConfig`. Query tower and gallery tower over a ULIP-2 backbone. `tower_sharing` ∈ `shared_backbone_separate_fusion` | `fully_shared` | `fully_separate`. `freeze_gallery()` refuses `fully_shared` (`dual_tower.py:315-321`) — the paper's 2.6 requirement and a single shared module cannot both hold.

**Backbone** — `metafind/models/ulip_backbone.py`
`ULIPBackbone` / `BackboneConfig`, `EMBED_DIM`. Vendored ULIP-2 + PointBERT under `metafind/vendor/ulip/`. CLIP frozen for n06 encoding.

**Fusion** — `metafind/models/fusion.py`
`FusionConfig`, `sample_modality_mask()`. Modality masking per 2.6 at p=0.3, independently per modality.

**Loss** — `metafind/models/losses.py`
`MetaFindContrastiveLoss` / `ContrastiveConfig`. Stage 1 is query→gallery only (Eq. 5, unidirectional); Stage 2 is symmetric (Eq. 7a/7b). `PAPER_TAU = 0.5` is defined at `losses.py:70` and warns on deviation.

**ESSGNN** — `metafind/models/essgnn.py`
`ESSGNNConfig`, `ESSGCL` / `ESSGCLShared`. Two axes: `architecture_family` ∈ `sec25_two_mlp` | `appendix_shared_msg`, and `coord_feat` ∈ `current` | `updated`. They are currently **coupled** (`essgnn.py:191-195`, `essgnn.py:491-503`) — see D0-004.

**Protocol resolvers** — `metafind/models/resolve_stage1.py` (n05b), `resolve_stage2.py` (n09b). These write the JSON artifacts that trainers are not allowed to decide for themselves.

**Trainers** — `metafind/train/stage1.py` (n10), `stage2.py` (n13), `gallery_index.py` (n11/n11b/n12).

**Paths** — `metafind/paths.py`. Use it. Do not hardcode absolute paths.

---

## 4. Authority Hierarchy

Defined in `/home/kyzen/MetaFindV1/CLAUDE.md` §3. Summary, highest first:

1. Original MetaFind source / supplementary material (`docs/paper/metafind_source/`)
2. Published MetaFind paper
3. Upstream papers and official implementations (`docs/paper/{ulip2,egnn,idesign}_source/`)
4. Verified project audit / implementation contract (`docs/audit/`)
5. Graph specification (`docs/graph/`)
6. Current repository implementation
7. Tests and observed runtime/data
8. Reasoned inference
9. `Session Handoff.md` / conversational memory

A lower-authority source must never silently override a higher one. Conflicts stay explicit until Master resolves them.

`_workflow_old_20260820/` is **historical reference only** — not authority at any level. It may be mined for leads; every claim taken from it must be re-verified.

---

## 5. Stable Decisions

Accepted and reflected in on-disk artifacts. These are IMPLEMENTATION CHOICES unless marked otherwise.

| Decision | Value | Where recorded | Class |
|---|---|---|---|
| Stage 1 loss direction | query→gallery only (Eq. 5) | `losses.py`, `L1-LOSS-STAGE1-UNIDIRECTIONAL` | PAPER FACT |
| Stage 2 loss direction | symmetric (Eq. 7a/7b) | `losses.py` | PAPER FACT |
| Object split | 80/20, sorted+seeded shuffle, seed 20260816 | `splits.py:65-66` | PAPER 3.1 + choice of seed |
| Missing-modality representation | learned token (not zero-pad) | `stage1_encoding_protocol.json`, U-11 | IMPLEMENTATION CHOICE |
| Fusion strategy | `masked_mlp` | `splits.py:75` (`DEFAULT_FUSION`), U-13 | IMPLEMENTATION CHOICE |
| All-masked queries allowed | `true` | `splits.py:86`, U-23 | IMPLEMENTATION CHOICE |
| ESSGNN architecture | `appendix_shared_msg`, `coord_feat: current`, `hidden_dim 128`, `n_layers 4`, `distance: squared` | `essgnn_arch_protocol.json`, decided by Kyzen 2026-08-19 | IMPLEMENTATION CHOICE |
| Semantic edges undirected | yes | U-19; `scene_graphs.py`, `semantic_edges_run.py` | IMPLEMENTATION CHOICE |
| `f_x` output: scalar coordinate multiplier | scalar retained | **`D0-009` `USER_APPROVED` 2026-08-21 (`DL-004`). Verdict `PAPER-AMBIGUOUS`** — MetaFind states `f_x → R³` (`2methdology.tex:54`) and **never defines the `·`** in the coordinate update. **Do not write "the paper is wrong", and do not cite upstream EGNN as settling it.** The `2.2e-16 vs 0.43` figures are **UNVERIFIED here and unreproducible** — no `R³` variant exists in code | **USER-RATIFIED IMPLEMENTATION CHOICE under a PAPER-AMBIGUOUS specification.** Not a PAPER FACT |
| n04 unit-sphere normalisation | kept | Objaverse units span ~1.3e5×; absolute scale never usable | IMPLEMENTATION CHOICE |
| Gallery scopes reported | both A_test_gallery and B_full_gallery, `gallery_size` derived not hardcoded | `splits.py:104-124`, U-09 | IMPLEMENTATION CHOICE |

**Stage 1 text serialization (U-15) — RATIFIED. `USER_APPROVED` 2026-08-21.**

Decision: `workflow/decisions/D0-008_stage1-text-template.md` §14. Brief: `D0-008_USER_REVIEW.md`. Ledger: `workflow/DECISION_LEDGER.md`.

**Ratified in design only.** This does not authorise n06 to run — the cache completion/validity gate is owned by `D10_stage1-encoding-contract`. **The ratified template is not yet implemented in `resolve_stage1.py`.**

```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```

`{Category}` = category with first character upper-cased. `W`/`L`/`H` render at one decimal with a trailing `.0` stripped, **uniformly at every magnitude** — no `< 1 cm` threshold. No `"A "` article and **no a/an heuristic** (explicit user constraint).

| Element | Class |
|---|---|
| That a serialization format exists at all | IMPLEMENTATION CHOICE — the paper is silent (PAPER FACT as to silence, verified across all 5 `.tex` files + Figure 2) |
| Field set (description, category, materials, dimensions, placement) | PAPER FACT (`2methdology.tex:28` + Figure 2); concatenating them into one prompt is an IMPLEMENTATION CHOICE |
| Field order | IMPLEMENTATION CHOICE. **The paper does not constrain serialization order.** The paper's attribute order is category → *dimensions* → materials; the code emits category → *materials* → dimensions. Any future claim that §2.3 mandates an order must cite new evidence |
| Centimetres | **INFERENCE** as to MetaFind's intent (density plausibility of Figure 2's mass/size pairing + unstated Holodeck schema match). **OBSERVED DATA** as to this corpus: all 45,952 v3 records store `dimension_unit: "cm"`. The volume-arithmetic support is **WITHDRAWN** — `30×30×40=36000` is unit-invariant |
| `width, length, height` ordering | INFERENCE from Figure 2, which prints that order |
| Dimension precision, article removal | IMPLEMENTATION CHOICE, user-approved (E-1, E-2, S-1, S-2). **Not** bug fixes — the prior behaviour did what it said; the choice was defective |
| Omission of `synset` / `volume` / `mass` | **IMPLEMENTATION CHOICE. Retrieval impact UNKNOWN.** *Binding user wording, 2026-08-21:* must **not** be stated as a PAPER FACT, and must **not** be described as proven redundant. The "volume is redundant" justification is **WITHDRAWN** (Codex C-6): a frozen text tower is not guaranteed to multiply three numerals. `resolve_stage1.py:111`'s `r = 0.52-0.62` is **UNVERIFIED** here and must not be reported as MEASURED |
| Placement phrasing | IMPLEMENTATION CHOICE. `onWall`→"mounted", `onObject`→"on top of", all-false→"no typical placement" are **inventions**, not schema-preserving paraphrases |

Decision file: `workflow/decisions/D0-008_stage1-text-template.md`. **The ratified template is not yet implemented** — see §6.

Rows sourced from the previous workflow rather than re-verified this session: the Objaverse scale-span figure (~1.3e5×). The decisions are reflected in code; the supporting numbers are unverified.

**Corrections C-001 and C-002 — COMPLETED 2026-08-21 by `D2a_stage1-protocol-refresh` (`DL-003`):**

| | Established | Current artifact / code | Owner |
|---|---|---|---|
| Temperature τ | **RESOLVED 2026-08-21 (`DL-003`).** `stage1_hyperparameters.json` now records `init_temperature: 0.5`, `learnable_temperature: false` | τ = 0.5 is a **PAPER FACT** (`3experiments.tex:15`). `learnable_temperature: false` is a **USER-RATIFIED IMPLEMENTATION CHOICE** — the paper uses "learnable" for `f_h`/`f_x` (`2methdology.tex:54`) and λ (`:87`), but calls τ a "temperature **hyperparameter**" twice (`:79`, `:99`) and never states it is non-learnable. **Never write this as a PAPER FACT** |
| Encoding protocol record | **RESOLVED 2026-08-21 (`DL-003`).** `stage1_encoding_protocol.json` now records the ratified template and `metafind_v2_cm@8e4b1fcc66c7f48c`. `load_protocol()` passes | — |

Do not re-open τ as a research question. Using anything other than 0.5 is a **DEVIATION** requiring registration and disclosure wherever results are compared with the paper's tables.

**Genuinely open — do not treat as decided:** `tower_sharing`, handling of the 3 `prompt_version:1` annotations, the ESSGNN axis coupling, `build_model()` construction path, node-text enrichment, Table 2 protocol. See `workflow/INDEX.md` §Decision Queue.

---

## 6. Current Project State

**Executed and on disk:** n02, n03, n04, n05, n05b, n07, n07b, n08, n09b, n09c.

**Not executed:** n06 (partial and stale), n09, n10, n11, n11b, n12, n13, n14+.

**Not implemented at all:** n10b, n14, n15, n15a, n15b, n15c, n16, n17, n18–n22.

Key artifact counts, **re-measured by Master 2026-08-21** (supersedes the 2026-08-20 snapshot):

```
pointclouds_index.jsonl        46,052
renders_index.jsonl            45,955
annotations_index.jsonl        45,955
data/outputs/annotations/      45,955  (45,952 prompt_version 3 + 3 prompt_version 1)
                                       ← the v3 corpus is SEMANTICALLY DEFECTIVE; D14 is replacing it
admitted (pc ∩ renders ∩ ann)  45,955
data/outputs/embeddings/       20,053 npz  ← [CORRECTED, was 5,276] D1 STOPPED 2026-08-21T14:15:48.
                                             ALL INVALID: stale template AND defective categories.
                                             Nothing deleted. Do not resume from them.
data/outputs/scene_graphs/     12,000
data/outputs/checkpoints/           0  ← empty, never trained
splits.json / eval_protocols.json / stage1_protocol.json   ABSENT
```

**The n05 annotation corpus is defective.** Qwen was asked to *identify* objects it could not read and collapsed onto high-frequency priors (`toy` 3.4%, `bookshelf` 2.4% vs LVIS's most common class `chair` at 1.0%; top-20 share 22.3% vs the ground truth's 7.1%). Because `build_prompt` derives dimensions and placement from the category, **a wrong category is a wrong record, not a wrong field.** Diagnosis: `workflow/MIF_n05_diagnosis.md`. Replacement design: `workflow/n05_v5_design.md`. Executing task: `D14_n05-v5-reannotate`, ACTIVE, holding at its Phase 2 gate. **Never describe the on-disk v3 corpus as accepted.**

Repository health: `python -m pytest tests/ -q` → **582 passed, 0 skipped** (`[CORRECTED, was 442]` — the suite grew 442 → 547 → 582 as `D2a` and `D14` added coverage). `tools/check_graph.py` → 2275 checks, all pass. **Neither is evidence of paper fidelity.** Use the plain command with no `--ignore`; `test_cuda_smoke.py` genuinely runs.

**n06 expected successful output = 45,952 `.npz` + 3 quarantine records.** n06 attempts 45,955 (`annotations` glob ∩ `renders_index.jsonl`, `encode_text_image.py:177-179`); the 3 `prompt_version:1` records carry a different schema and raise `KeyError: 'width'` in `serialize_annotation()`, so they are quarantined without output (`encode_text_image.py:213-221`). Keep the three numbers distinct: **45,955 annotation files · 45,952 valid v3 · 3 v1 residuals.**

**RESOLVED 2026-08-21 (`DL-003`). The annotation corpus is now protected, and the protocol is refreshed.**

```
bare `annotate_run` (no --force) queues     0 records TOTAL
  accepted legacy-v3                        0
  legacy-v1 residuals                       0
--force still queues                   45,955   (capability preserved)
state histogram   {accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}
load_protocol()   PASS -> metafind_v2_cm@8e4b1fcc66c7f48c
```

**The declared registry is `data/outputs/annotation_provenance.json`.** Three states are explicit there — `accepted_legacy_v3` (45,952), `legacy_v1_residual_unresolved` (3), and annotated-under-current-contract — never inferred from a missing field. **The corpus is `legacy-v3 validated under VALIDATOR_VERSION 2`. It is not v4-generated and must never be described as such.** `D0-003` remains **UNRESOLVED**; the 3 residuals are labelled accordingly.

The work-list predicate now lives in `build_work_list()` (`annotate_run.py:439`), **not** in `is_complete()` alone. Any check written against `is_complete()` gives a false negative.

**Historical — the defect this replaced:**

**BLOCKER (resolved) — a resumed n06 would silently build a two-distribution gallery.** `is_complete()` (`encode_text_image.py:73-83`) checks only sidecar existence, `encoder_version`, and NPZ existence. It compares **nothing about the text**. A plain re-run would skip all 5,276 metre-derived embeddings as "complete" and encode only the rest in centimetres — no error, no warning, and an identical `text_serialization` label on both halves. `gallery_index.py` fingerprints the checkpoint, not the text, so Table 1 would come out self-consistent and wrong. `"metafind_v1_natural"` already labels two different transformations and is not a valid cache identity. `D0-008` (`DL-001`), `D10` (`DL-002`) and `D2a` (`DL-003`) are all `USER_APPROVED`; `load_protocol()` passes, the pre-flight passes, and every stale embedding is cache-invalid without being deleted.

**[CORRECTED 2026-08-21] `D1_n06-reencode` is NOT runnable.** This paragraph previously ended *"`D1_n06-reencode` is UNBLOCKED as of 2026-08-21."* That was true for about six hours. D1 then ran, was **STOPPED at `2026-08-21T14:15:48`** at 20,053 npz on the user's order, and is now blocked behind `D14`'s replacement of the annotation corpus. Its `TASK.md` preconditions (5,276 npz at start, 547 tests) are stale.

**The recorded Stage 1 encoding protocol does not describe the encoder.** `data/outputs/stage1_encoding_protocol.json` records
`"... roughly {length:.2f} by {width:.2f} by {height:.2f} metres, typically placed {placement}."`
while `resolve_stage1.py:96-100` defines
`"... roughly {width:.0f} by {length:.0f} by {height:.0f} centimetres, {placement}."`
n06 uses the **code** constant (`encode_text_image.py:194` → `serialize_annotation`, default `TEXT_TEMPLATE`); the artifact field is written but never read back. `tools/check_graph.py` does not catch the mismatch. The code additionally carries `# [U-15, IMPLEMENTATION CHOICE -- CONFIRM BEFORE THE FULL RUN]` at `resolve_stage1.py:102`; that marker is discharged by D0-008 but the code still holds the pre-ratification template.

**Measured defects in the current serializer** (full corpus, 45,952 v3 records, re-verified by Master 2026-08-21): **161** records render a stored non-zero dimension as `0` under `:.0f`; **3,643** emit `"A airplane"`-style ungrammatical articles; **1** record (`3e91980a22da4c0da975cc8ef776972c`, 89 true BPE tokens) exceeds CLIP's 77-token limit and is **recorded and then encoded anyway**. The ratified template removes all three at zero token cost.

**Four in-code justifications do not describe the code**: the field-order claim (`resolve_stage1.py:93-95`), the "every variable-length part is bounded" claim (`:141-149`, `MAX_PLACEMENT` at `:162` is unused), the unreachable `PLACEMENT_PHRASES[("onWall","onCeiling")]` entry, and the withdrawn volume-redundancy argument. Do not trust in-code rationale in this module without checking it.

**The embedding cache is stale, not merely incomplete.** Cached text uses the old metre-based template and pre-v3 dimension values; current v3 annotations use centimetres and different numbers. Mixing the two would train Stage 1 on two text distributions. `data/outputs/annotations_v1_prompt1/` (45,953) and `annotations_v2_sample/` (200) are backups — **do not delete**.

---

## 7. Cross-Task Dependencies

```
D9 figures audit ──────────┐ may refine evidence for D0-002, D0-004; not yet scheduled
D0-008 ratify template ──► D1 n06 re-encode ─┐
D0-002 / D0-003 ─────────────────────────────┴─► D2 (C-001 + C-002 + n09) ──► D3 Stage 1 train
                                                     │
                                        D4 gallery index (n11/G4/n12)
                                             │                 │
                          D0-004/006 ──► D5 Stage 2 prereq     └──► D7 Table 1 eval
                                             │
                                        D6 Stage 2 train ──► D8 Table 2 eval (needs D0-007)
```

Facts every task owner should know:

- **n09 does not depend on n06.** `splits.py:169-171` reads the three index files, never the embeddings. Stage 1 *training* needs both.
- **n09 bakes decisions into artifacts.** `stage1_protocol.json` carries `tower_sharing` and the sha256 of `stage1_hyperparameters.json`; `stage1.py:81-86` refuses to run if the hash does not dereference. Changing τ after n09 requires rerunning n09.
- **n05b rewrites two artifacts at once.** `resolve_stage1.py:443-444` writes `stage1_encoding_protocol.json` and `stage1_hyperparameters.json` in the same call, and `build_hyperparameters()` is invoked with no overrides (`resolve_stage1.py:442`). `DEFAULT_HYPERPARAMETERS` (`resolve_stage1.py:197-211`) hardcodes `init_temperature: 0.07`, `learnable_temperature: True` — **there is no supported way to produce τ = 0.5 through n05b today.** Corrections C-001 and C-002 must therefore land together, or n05b runs twice.
- **The 3 v1 annotations crash Stage 1 if n09 admits them.** `splits.py:169-171` admits all 45,955; `stage1.py:109` loads `EMBEDDINGS/<uid>.npz` with no existence guard; n06 never produces those three. Result: `FileNotFoundError` mid-epoch. This is a hard dependency of D3 on D0-003, not a reporting nicety.
- **Gallery index requires an encoder fingerprint.** `gallery_index.py` writes `gallery_index_<ckpt-sha16>.npz` and cross-checks `stage1_ckpt.json`. An index built by a drifted encoder produces self-consistent wrong numbers with no error.
- **`fully_shared` cannot reach Stage 2.** `dual_tower.py:315-321` raises on `freeze_gallery()`.

---

## 8. Global Constraints

- Tests passing is not proof of paper fidelity.
- Do not infer paper requirements from the current implementation.
- Do not silently replace missing evidence with assumptions.
- Mark unsupported or unresolved research claims explicitly.
- Codex is an independent reviewer, not scientific authority; Claude must verify material Codex findings against stronger evidence before adopting them.
- A task must not expand into another task's scope without Master approval.
- A task must not start the next stage on its own.
- Do not delete or regenerate datasets, annotation backups, checkpoints, embeddings, or experiment outputs without explicit authorization.
- Report `MASTER-IMPACTING FINDING` rather than rewriting global project state locally.

---

## 9. Runtime / Environment Facts

- Repository: `/home/kyzen/MetaFindV1`
- **Data root: `/mnt/data1/kyzen/MetaFind`, linked as `/home/kyzen/MetaFindV1/data`. [CORRECTED 2026-08-21]**
  Was `/home/kyzen/data/MetaFind`. **The migration is complete** — the symlink was repointed 2026-08-21 21:32 and `/home/kyzen/data` **no longer exists**. `CLAUDE.md` §9 still carries the old path; it is project instruction and only the user may correct it.
  **Consequence, unmeasured:** the dataset now sits on the SMR drive described below. Every runtime estimate recorded anywhere in this project predates the move.
- Python: `/home/kyzen/miniconda3/envs/MetaFind/bin/python` (conda env `MetaFind`)
- Run modules as `python -m metafind.<module>` from the repository root.
- **`/mnt/data1` IS valid on this host** — corrected 2026-08-21 (D14 finding F-1, re-verified by Master). It is a mounted 3.6 TB ext4 volume, `3.4 TB free`, `/mnt/data1/kyzen` writable, filesystem created 2026-08-20. **`CLAUDE.md` §9 still carries the stale claim that it belongs to a previous machine; that file is project instruction and only the user may correct it.** Whether `/mnt/data1` becomes a sanctioned location for project artifacts is **undecided**.
- **`/mnt/data1` is an SMR drive** (`ST4000DM004`, measured 2026-08-21). Sustained small-file writes collapse to single-digit MB/s once its CMR cache fills — `w_await` measured above **5,000 ms** under mixed load, versus <10 ms normal. **Suits cold, read-mostly bulk data; poor for high-frequency small-file work.** n06 writes 45,952 small `.npz`, which is the worst case for this drive. **Decide placement per directory, not for `data/` as a whole.**
- Use `metafind/paths.py` for project paths regardless.
- GPU: **verified 2026-08-20 — `NVIDIA GeForce RTX 5090`, 31.4 GB, `torch.cuda.is_available() → True`.** Documentation elsewhere in the repo has been reported as stating different hardware; treat those statements as stale. Record the actual device for any experiment.
- Observed n06 throughput: ~190–197 assets/min (partial run, 2026-08-17). A projection, not a guarantee.
- Observed n08 runtime: ~22 min (old workflow claim, not re-verified).
- `graphify-out/graph.json` exists; `graphify query "..."` is available for codebase navigation. Navigation only — conclusions must return to source.

---

## 10. Shared File Map

### Project control
- `CLAUDE.md` — project research / engineering rules
- `.claude/rules/` — research-rigor, experiments, paper-reproduction, code-changes
- `workflow/WORKFLOW.md` — the operating protocol
- `workflow/MASTER.md` — Master global control state
- `workflow/CONTEXT.md` — this file
- `workflow/INDEX.md` — task registry and decision queue
- `workflow/decisions/` — accepted D0 decisions
- `workflow/tasks/<task-id>/` — `TASK.md`, `HANDOFF.md`, `CODEX_REVIEW.md`

### Research evidence
- `docs/paper/metafind_source/` — MetaFind TeX + 6 figures (`2methdology.tex`, `3experiments.tex`, `appendix.tex`, `MetaFind.drawio.png`, `data-preprocess.png`)
- `docs/paper/{ulip2,egnn,idesign}_source/` — upstream sources; 15 / 8 / 9 figures
- `docs/audit/` — A_FORMULA_INVENTORY, B_UPSTREAM_TO_METAFIND_MAP, C_PAPER_CONTRADICTIONS, D_IMPLEMENTATION_FORMULA_CONTRACT, E_GRAPH_REVALIDATION, F_CODE_GRAPH_CONSISTENCY
- `docs/graph/` — 00_FINDINGS, 01_GRAPH_SPEC, 02_BUILD_STEPS, `node_registry.yaml`, `graph_spec.yaml`, `validation_plan.yaml`
- `essgnn.md` — ESSGNN axis working notes (working context, not authority)

### Implementation
- `metafind/data/` — n02–n09c preprocessing
- `metafind/models/` — dual_tower, fusion, essgnn, losses, ulip_backbone, resolve_stage1/2, stage1_config
- `metafind/train/` — stage1, stage2, gallery_index
- `metafind/vendor/` — vendored ULIP-2 / PointBERT / EGNN. Prefer adapters over editing.
- `tests/` — 21 test modules
- `tools/check_graph.py` — structural gate checker. Run after graph/spec/code changes.

### Historical (reference only, not authority)
- `_workflow_old_20260820/` — previous mainline, support-line, TASKS registry, and A–F task cards
- `Session Handoff.md` — **deleted from the repository** in commit `1837477` ("chore: establish multi-agent research workflow"), superseded by this workflow. Recoverable from history with `git show 169bd5b:"Session Handoff.md"` if ever needed. It was never authority; Master session continuity now belongs in `workflow/MASTER_SESSION_HANDOFF.md`

---

## 11. Task Conversation Startup Rule

A new D-task conversation reads, in order:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. applicable `.claude/rules/`
3. this file
4. its own `workflow/tasks/<task-id>/TASK.md` (or `workflow/decisions/<id>.md` for D0)
5. only the additional evidence / implementation files that TASK.md explicitly names

Do not automatically re-read the entire repository. Do not read other task folders.

If the task discovers something that changes shared architecture, dependencies, or an accepted Master assumption, report `MASTER-IMPACTING FINDING` with evidence, affected tasks, and whether the current task can safely continue. Do not rewrite global project state.

---

## 12. Context Maintenance Rule

Master owns this file. Update it only when shared project understanding changes: an architecture decision is accepted, a major dependency changes, a project-wide constraint changes, a stable runtime fact changes, or a milestone moves the shared state.

Do not update it for routine debugging, minor implementation edits, individual test failures, or temporary experiment results.
