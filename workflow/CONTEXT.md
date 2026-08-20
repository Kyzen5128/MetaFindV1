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
| `f_x` output is scalar, not R³ | scalar | 2.5 literal text is wrong; scalar gives equivariance error 2.2e-16 vs 0.43 | DEVIATION, recorded |
| n04 unit-sphere normalisation | kept | Objaverse units span ~1.3e5×; absolute scale never usable | IMPLEMENTATION CHOICE |
| Gallery scopes reported | both A_test_gallery and B_full_gallery, `gallery_size` derived not hardcoded | `splits.py:104-124`, U-09 | IMPLEMENTATION CHOICE |

Rows sourced from the previous workflow rather than re-verified this session: the `f_x` scalar equivariance figures (2.2e-16 vs 0.43) and the Objaverse scale-span figure (~1.3e5×). The decisions are reflected in code; the supporting numbers are unverified.

**Established but not yet reflected in artifacts — corrections pending, not open questions:**

| | Established | Current artifact / code | Owner |
|---|---|---|---|
| Temperature τ | **PAPER FACT τ = 0.5** (`3experiments.tex:15`, the only value the paper states; no conflicting statement in the source). Non-learnable is a strongly supported INFERENCE, not paper wording | `stage1_hyperparameters.json` records `0.07` / `learnable: true`; `resolve_stage1.py:197-211` hardcodes it with no override path | C-001, executed in D2 |
| Encoding protocol record | The artifact must describe what `serialize_annotation()` actually emits | `stage1_encoding_protocol.json` records the v1 metre template | C-002, executed in D1/D2 |

Do not re-open τ as a research question. Using anything other than 0.5 is a **DEVIATION** requiring registration and disclosure wherever results are compared with the paper's tables.

**Genuinely open — do not treat as decided:** `tower_sharing`, handling of the 3 `prompt_version:1` annotations, ratification of the Stage 1 text serialization template, the ESSGNN axis coupling, `build_model()` construction path, node-text enrichment, Table 2 protocol. See `workflow/INDEX.md` §Decision Queue.

---

## 6. Current Project State

**Executed and on disk:** n02, n03, n04, n05, n05b, n07, n07b, n08, n09b, n09c.

**Not executed:** n06 (partial and stale), n09, n10, n11, n11b, n12, n13, n14+.

**Not implemented at all:** n10b, n14, n15, n15a, n15b, n15c, n16, n17, n18–n22.

Key artifact counts, verified 2026-08-20:

```
pointclouds_index.jsonl        46,052
renders_index.jsonl            45,955
annotations_index.jsonl        45,955
data/outputs/annotations/      45,955  (45,952 prompt_version 3 + 3 prompt_version 1)
admitted (pc ∩ renders ∩ ann)  45,955
data/outputs/embeddings/        5,276 npz  ← PARTIAL AND STALE
data/outputs/scene_graphs/     12,000
data/outputs/checkpoints/           0  ← empty
splits.json / eval_protocols.json / stage1_protocol.json   ABSENT
```

Repository health: `python -m pytest tests/ -q` → **442 passed, 0 skipped, 0 deselected** (442 collected). `tools/check_graph.py` → 2275 checks, all pass. **Neither is evidence of paper fidelity.** Use the plain command with no `--ignore`; `test_cuda_smoke.py` contributes 5 of the 442 and CUDA is available, so those tests really run.

**n06 expected successful output = 45,952 `.npz` + 3 quarantine records.** n06 attempts 45,955 (`annotations` glob ∩ `renders_index.jsonl`, `encode_text_image.py:177-179`); the 3 `prompt_version:1` records carry a different schema and raise `KeyError: 'width'` in `serialize_annotation()`, so they are quarantined without output (`encode_text_image.py:213-221`). Keep the three numbers distinct: **45,955 annotation files · 45,952 valid v3 · 3 v1 residuals.**

**The recorded Stage 1 encoding protocol does not describe the encoder.** `data/outputs/stage1_encoding_protocol.json` records
`"... roughly {length:.2f} by {width:.2f} by {height:.2f} metres, typically placed {placement}."`
while `resolve_stage1.py:95-99` defines
`"... roughly {width:.0f} by {length:.0f} by {height:.0f} centimetres, {placement}."`
n06 uses the **code** constant (`encode_text_image.py:194` → `serialize_annotation`, default `TEXT_TEMPLATE`); the artifact field is written but never read back. `tools/check_graph.py` does not catch the mismatch. The code additionally carries `# [U-15, IMPLEMENTATION CHOICE -- CONFIRM BEFORE THE FULL RUN]` at `resolve_stage1.py:101`. **Do not re-run n06 until D0-008 is accepted.**

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
- Data root: `/home/kyzen/data/MetaFind`, linked as `/home/kyzen/MetaFindV1/data`
- Python: `/home/kyzen/miniconda3/envs/MetaFind/bin/python` (conda env `MetaFind`)
- Run modules as `python -m metafind.<module>` from the repository root.
- Paths under `/mnt/data1` belong to a previous machine and are invalid. Use `metafind/paths.py`.
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
- `Session Handoff.md` — session working memory only. **Currently absent from the working tree** (deleted after commit `4a4ebbe`, not by any workflow task; recoverable with `git restore "Session Handoff.md"`). It is not authority either way

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
