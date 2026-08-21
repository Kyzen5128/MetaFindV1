# MASTER INITIALIZATION REPORT

**Date:** 2026-08-21 · **git HEAD:** `468bbac` (+ uncommitted D14 Phase 1 work)
**Author:** Master (this session) · **Status:** for USER review. **No scientific implementation was changed.**

> **SUPERSEDED IN PART, same day.** A parallel Master session executed the n03/n04 upstream
> verification directly and wrote `workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md` at 22:47.
> It **resolves risk R1** (§3.3, §10): the 180° yaw is **real** but was measured to have **no effect**
> on the embeddings the pipeline consumes — matched cosine 0.4513 (0°) vs 0.4512 (180°), R@1 98.0% vs
> 97.5% against ULIP-2's own 99.0%. **The point-cloud corpus does not need regenerating.** The yaw
> still matters for `n16_compose_scenes`, where assets are placed with real geometry. Read
> `FINDINGS.md` before acting on §3.3 or §10 of this report.

> Everything below is either (a) re-measured by Master this session, or (b) quoted from a repository
> document with its source named. Where a document and the disk disagree, the disk is reported and the
> document is listed in §9 as needing correction.

---

## 1. Current Authoritative Project State

### 1.1 What the project is

Reproduce MetaFind — a dual-tower multimodal 3D asset retrieval system with layout context —
with **evidence-backed correspondence** between the published method and this implementation.
Working code is not the deliverable. Traceable reproduction fidelity is.

Two training stages feeding two evaluation tables:

```
Stage 1 (object)  n02 download → n03 pointclouds → n04 renders(11) → n05 annotate(VLM)
                  → n05b protocol → n06 encode(text+image) → n09 splits → n10 train
                  → n11/G4/n12 gallery → n15 eval  ..................... TABLE 1

Stage 2 (layout)  n07 scene graphs → n07b modalities → n08 semantic edges
                  → n09b/n09c protocol+splits → n11b index → n13 train(ESSGNN)
                  → n15a/b/c → n16 compose → n17 judge  ................ TABLE 2

Then              n18/n19 ablations → n20 aggregate → n21 compare → G5 → n22 publish
```

Registry: `docs/graph/node_registry.yaml` — **38 nodes, 7 gates.**

### 1.2 Execution state (Master-verified 2026-08-21)

| | Nodes |
|---|---|
| **Executed, artifacts on disk** | n02, n03, n04, n05 (v3 corpus), n05b, n07, n07b, n08, n09b, n09c |
| **Started and STOPPED** | n06 — 20,053 / 45,952 `.npz`, halted `2026-08-21T14:15:48` on the user's order. **These embeddings are invalid** once annotations change |
| **Never executed** | n09, n10, n11, n11b, n12, n13, n14+ |
| **Zero implementation code** | n10b, n14, **n15 (Table 1)**, n15a, n15b, n15c, n16, n17, n18–n22 |
| **`data/outputs/checkpoints/`** | **empty.** No model has ever been trained |

### 1.3 Repository health, re-run by Master this session

```
pytest tests/ -q          582 passed, 22 warnings (all τ-deviation warnings)
tools/check_graph.py      2275 checks, all pass
```

**Neither is evidence of paper fidelity.** They establish that the code executes and that the
spec documents are internally consistent with each other.

The 22 warnings are `stage1.py:331` reporting `init_temperature=0.07 learnable=True` inside a
**test fixture**; the production artifact `stage1_hyperparameters.json` was corrected to `0.5 / false`
by `DL-003`. This is expected, not a regression.

### 1.4 Working tree

`git status` shows **13 modified + 7 untracked** paths. All of it is real, in-flight work:

- `metafind/data/annotate.py`, `annotate_run.py`, `tests/test_annotate.py`, `metafind/data/lvis_synsets.json`
  — **D14 Phase 1 (n05 v5), complete but uncommitted and unaccepted**
- `docs/graph/*`, `README.md`, `workflow/*` — the `D-2`/`D-8` deviation split (`DL-005`)
- `workflow/tasks/D14…`, `D15…`, `D16…`, `workflow/decisions/D0-010…`, two `MIF_*` files, `n05_v5_design.md`

**Nothing here is committed. A single `git checkout` would destroy D14 Phase 1.**

---

## 2. Accepted Scientific Decisions

### 2.1 Ratified through the user review gate

| ID | Decision | Class | Status |
|---|---|---|---|
| `DL-001` | Stage 1 text serialization template (U-15): centimetres, `{description} {Category} made of {materials}, roughly W by L by H centimetres, {placement}.` | Field **set** = PAPER FACT; format, order, units = IMPLEMENTATION CHOICE / INFERENCE | `USER_APPROVED` |
| `DL-002` | Stage 1 encoding contract: `text_serialization_id` cache validity, `load_protocol` mismatch rejection, >77-token hard gate, pre-flight | IMPLEMENTATION CHOICE + 3 user-authorised DEVIATIONs | `USER_APPROVED` |
| `DL-003` | τ = 0.5, `learnable_temperature: false`, protocol refresh, AC-1 corpus rerun protection via `annotation_provenance.json` | τ=0.5 = **PAPER FACT** (`3experiments.tex:15`); non-learnable = USER-RATIFIED IMPLEMENTATION CHOICE | `USER_APPROVED` |
| `DL-004` | ESSGNN `f_x` stays a **scalar** coordinate multiplier | Verdict **`PAPER-AMBIGUOUS`**. USER-RATIFIED IMPLEMENTATION CHOICE — **not** a PAPER FACT | `USER_APPROVED` |

**Binding prohibition from `DL-004`:** *"upstream EGNN settles it"* may no longer be used as
paper-interpretation authority anywhere in this project.

### 2.2 User decisions taken in conversation, not yet through the gate

| ID | Decision | Where recorded |
|---|---|---|
| **U-6** | Annotation model → **local `Qwen3.8-27B`** (was Qwen2.5-VL-7B) | `D14/USER_DIRECTIVES.md`, `annotate_run.py:72` |
| **U-7** | `/mnt/data1` sanctioned; project data to be migrated there | same. **Migration is already done** — see §4 |
| **U-10** | Phase 1 + Phase 2 of D14 authorised. **Not Phase 3** | same |
| — | n05 v5 **category-anchored** design (`n05_v5_design.md`) approved 2026-08-21 | `D14/TASK.md` §2 |
| `DL-005` | Deviation `D-2` split into `D-2` (annotation model) + `D-8` (scene judge model) | ledger, `AWAITING_USER_REVIEW` |
| `DL-003-A1` | Prepared amendment to AC-1 for when D14 Phase 3 lands | ledger, **`PREPARED, NOT IN FORCE`** |

### 2.3 Standing stable decisions (`CONTEXT.md` §5)

Stage 1 loss unidirectional (PAPER FACT) · Stage 2 loss symmetric (PAPER FACT) · 80/20 object split,
seed 20260816 · missing modality = learned token · fusion `masked_mlp` · all-masked queries allowed ·
ESSGNN `appendix_shared_msg` / `coord_feat: current` / hidden 128 / 4 layers / squared distance ·
semantic edges undirected · n04 unit-sphere normalisation kept · both gallery scopes reported.

### 2.4 Registered deviations

Seven (`D-2`…`D-8`) plus one conditional (`D-1`, not activated). The two live ones:

- **`D-2`** — annotation model is not GPT-4o. Now `Qwen3.8-27B`. Paper says GPT-4o twice
  (`2methdology.tex:28`, `neurips_2025.tex:100`).
- **`D-8`** — scene judging (n17) is `Qwen2.5-VL`, also not GPT-4o.
- **New, not yet assigned a deviation id: LVIS category anchoring.** The paper has the VLM
  *generate* the category. v5 feeds the dataset's ground-truth label in. `D14/TASK.md` R-E records
  this as a DEVIATION; **`graph_spec.yaml` does not yet carry it.**

**Correction of record.** `D-2`'s stated reason was *"GPT-4o is unavailable"*. Master wrote that
without verifying it. OpenAI's deprecation page does not list base `gpt-4o`; secondary sources
disagree. **That conflict is UNRESOLVED.** It must not be restated as settled.

---

## 3. Current Unresolved Scientific Issues

### 3.1 Blocking the critical path

| ID | Question | Blocks |
|---|---|---|
| **`D0-010`** | How does the Objaverse-LVIS ground-truth category enter n05? Prompt hint / hard value / cross-check / record-only. **And is the GPT-4o→Qwen substitution a material cause of the defect?** | D14 Phase 3, therefore everything |
| **`D0-003`** | The 3 `prompt_version:1` residual annotations — admit, drop, or re-annotate. **Hard crash risk:** `splits.py:169-171` admits all 45,955; `stage1.py:109` loads the `.npz` with no existence guard → `FileNotFoundError` mid-epoch | n09 (D2), D3 |
| **`D0-002`** | `tower_sharing`: `shared_backbone_separate_fusion` / `fully_shared` / `fully_separate`. Figure says "ULIP-2 (Shared)"; §2.6 says the gallery encoder is frozen. **`fully_shared` cannot reach Stage 2** (`dual_tower.py:315-321`) | n09 (D2), D3, Stage 2 feasibility |

`D0-010` has a decision file with §1–§5 written by Master; **§6–§11 are empty — no research has been done.**
`D0-002` and `D0-003` have **no decision file at all**.

### 3.2 Open, off the immediate critical path

| ID | Question | Blocks |
|---|---|---|
| `D0-005` | `build_model()` bypasses `Stage1RuntimeConfig`; one backbone; one shared `FusionConfig` object → `fully_separate` is unimplementable as written | D3, conditional on D0-002 |
| `D0-004` | ESSGNN `coord_feat` / `architecture_family` coupling — can `coord_feat` be isolated as an ablation axis? | D5, ablation design |
| `D0-006` | n08 `object_text()` returns `f"a {category}"` only — information collapse into Stage 2 input | D5 |
| `D0-007` | Table 2 protocol: how the 200 evaluation scenes are built; MetaFind 1–5 vs I-Design 0–10 comparability | D8 |
| `G-7` | D10's four comment corrections + `math.isfinite()` guard placement. **Explicitly NOT ratified by `DL-002`** | nothing today |
| `IC-1` | D14 records `exact`/`refined`/`divergent` category relations and **rejects nothing** — no hypernym source exists in this project to separate `motor vehicle→pickup truck` (wanted) from `motor vehicle→coffee machine` (forbidden) | D14 acceptance |
| `IC-2` | `synset` follows the LVIS anchor, not the model's refined `category` | D14 acceptance |

### 3.3 **NEW — surfaced this session, unrecorded anywhere in `workflow/` or `docs/`**

> **R1 — the point-cloud corpus may be systematically mis-oriented, and the two measurements in
> existence contradict each other.**

| Source | Population | Result |
|---|---|---|
| `docs/graph/00_FINDINGS.md` **F21** | **6** assets with matching GLBs | same-asset Chamfer median **0.00318** vs cross-asset baseline **0.05880** → "geometry matches, 18× separation" |
| A prior session's measurement (2026-08-21 ~14:20, memory only) | **286** overlapping assets from the official `000-009` shard | median Chamfer **0.0903 at 0°** vs **0.0230 at 180° yaw**; **269/286 (94.1%)** improve under a 180° rotation; poor-tier share falls 47.9% → 2.4% |

Both cannot be right. Candidate explanations, none tested:

1. The 180° result is an artifact — official ULIP-2 `.npy` clouds are **uncentered** (`centroid` up to 0.588)
   while our `.npz` are **pre-centered**; if the comparison omitted `pc_norm` on the official side, the
   whole distribution shifts and a spurious best rotation can appear.
2. F21's n=6 is simply too small to see a 94%-prevalence effect that has a 6%-ish exception rate.
3. A genuine frame-convention divergence between `pointclouds.py:118` (per-geometry `apply_transform`)
   and `renders.py:150` (per-vertex `transform_points` + `scene.apply_transform(fit)`).

A decisive test was written (`orient_test.py` — encode both orientations through the frozen ULIP-2
checkpoint, compare to the official image embeddings) and **timed out without producing a result.**

**Why this is the project's highest technical risk.** If real, it invalidates all 46,052 point clouds,
therefore Stage 1 training, therefore both tables — and it produces **no error anywhere in the chain**,
exactly the failure class the B6 blocker was created to prevent.

**Second-order problem:** `D15/TASK.md` §6 states *"`L2-PC-ULIP-REF` has never been run and the
reference clouds are not on disk."* **Both halves are false** — F21 ran a version of it, and the
reference shard was extracted this same day. D15's contract must be corrected before it runs.

---

## 4. Current Runtime / Data State

**All counts measured by Master 2026-08-21, read-only.**

```
data/outputs/annotations/           45,955   = 45,952 prompt_version 3 + 3 prompt_version 1
data/outputs/embeddings/*.npz       20,053   ← D1 halted; invalid once annotations change
data/outputs/checkpoints/                0   ← never trained
data/outputs/scene_graphs/          12,000
data/outputs/pointclouds/           92,104   = 46,052 .npz + 46,052 sidecars
data/outputs/renders/               92,000   = 46,000 dirs + sidecars (index admits 45,955)
splits.json / eval_protocols.json / stage1_protocol.json     ABSENT — n09 never ran
```

Backups present and **must not be deleted**: `annotations_v1_prompt1/`, `annotations_v2_sample/`,
`annotations_v3_pre_D10/`, `protocol_backup_pre_D2a_20260821/`.

### 4.1 Environment — **three documented facts are now stale**

| | Documents say | Measured today |
|---|---|---|
| Data root | `/home/kyzen/data/MetaFind` (`CLAUDE.md` §9, `CONTEXT.md` §9) | **`/mnt/data1/kyzen/MetaFind`.** `data →` symlink repointed 2026-08-21 21:32. **`/home/kyzen/data` no longer exists — the migration is complete** |
| Embeddings | 5,276 (`CONTEXT.md` §6) | **20,053** |
| Test count | 442 (`CONTEXT.md` §6, `MASTER.md` §4) | **582** |

```
GPU        NVIDIA GeForce RTX 5090, 32,607 MiB, 106 MiB used (idle)
/          937 G, 825 G free
/mnt/data1 3.6 T, 3.0 T free, 449 G used
annotation model  /mnt/data1/kyzen/models/Qwen3.8-27B — 18/18 shards, 56 GB, COMPLETE
upstream refs     /home/kyzen/upstream/ULIP @ 95d480fe · /home/kyzen/upstream/egnn @ e9ca6c0c
```

**`/mnt/data1` is an SMR drive (`ST4000DM004`).** Measured `w_await` above **5,000 ms** under mixed
small-file write load. **The entire project dataset now lives on it**, and n05 (45,952 small `.json`)
and n06 (45,952 small `.npz`) are precisely its worst case. No throughput measurement has been taken
since the migration; every existing runtime estimate predates it.

---

## 5. Proposed Block Mapping

Adopting the USER's B0–B4 structure. Two adjustments proposed, both structural, neither scientific.

| Block | Owns (nodes / gates) | Current live work |
|---|---|---|
| **B0** Research / Paper Fidelity | `D0-002` `D0-003` `D0-004` `D0-005` `D0-006` `D0-007` `D0-010`, `docs/audit/**`, the U-register, deviation registry, authority-conflict adjudication | `D0-010` (opened, no research done); `D0-002`/`D0-003` not opened |
| **B1** ULIP-2 / Asset Retrieval | n02→n06, n09, n10, n10b, n11, n12, gates G1 G2 G3 G4 | **D14** (ACTIVE, Phase 1 done), **D15** (planned), **D16** (blocked on D15), D1, D2, D3, D4 |
| **B2** ESSGNN / Layout | n07, n07b, n08, n09b, n09c, n11b, n13, n14, gate G6 | none — artifacts exist, nothing running |
| **B3** Retrieval / Composition | query encoder, layout-aware retrieval, n15a, n15b, n15c, n16, I-Design integration, gate G7 | none — **zero code exists** |
| **B4** Evaluation / Reproduction | n15 (Table 1), n17 (Table 2 judge), n18–n22, gates G5, metrics, compare-to-paper | none — **zero code exists** |

### Adjustment 1 — B1 is currently ~80% of the project and needs an explicit internal split

B1 as drawn spans the annotation corpus, the encoder, the splits, Stage 1 training and the gallery.
That is correct as an *ownership* boundary — one person must hold the whole chain — but it currently
carries **three simultaneous work fronts**. Proposed internal lanes, one Owner over all three:

- **B1-a Corpus** — n02…n05 + the LVIS anchor + gates G1/G2 (D14, D15, D16)
- **B1-b Encoding & splits** — n05b, n06, n09 (D1, D2)
- **B1-c Stage 1 training & gallery** — n10, n11/G4/n12 (D3, D4)

### Adjustment 2 — B4 should be staffed **now**, not last

B4 owns **n15 (Table 1)**, which has **zero implementation code**. It is the project's headline
deliverable and it is not on anybody's list. It is also fully parallel-safe: it writes only new
evaluation code and reads a promoted gallery index that does not exist yet, so it can be **designed
and unit-tested against synthetic inputs** long before D4 lands. Leaving it until after Stage 2 is
how a reproduction discovers at the end that it cannot produce its own table.

### Not a technical block — Master-owned

Infrastructure and governance: the `/mnt/data1` SMR placement decision, workflow-file consistency,
the deviation registry, and `graphify update`. Master keeps these; no Owner is assigned.

---

## 6. Current Critical Path

```
                          ┌── B0: D0-010 (category source)      ─┐
                          │                                       │  both must land
D14 Phase 2 (sample, 300-500 assets, no GPU cost decision yet) ◄──┘  before Phase 3
        │
        ▼   [HARD HOLD GATE — USER's explicit go only]
D14 Phase 3  full re-annotation, 45,952 assets
        │    cost NOT evidence-backed: 27B model, 56 GB bf16 on a 32 GB GPU
        ▼
D1  n06 re-encode  45,952 npz  (previous estimate ~4 GPU-h, predates the SMR migration)
        │
        │   ┌── B0: D0-002 tower_sharing ──┐
        │   └── B0: D0-003 the 3 residuals ─┤ (gate n09 only — independent of n06)
        ▼                                   ▼
D2  n05b re-run + n09 build_splits  →  splits.json / eval_protocols.json / stage1_protocol.json
        │
        ▼
D3  Stage 1 training  ← first checkpoint the project has ever produced
        │
        ▼
D4  gallery index  n11 → G4 → n12
        │
        ├──────────────► D7  n15 eval  →  TABLE 1     ← B4, zero code today
        │
        └── D5 Stage 2 prereq → D6 Stage 2 train → D8 n16/n17 → TABLE 2   ← B2/B3/B4
```

**Parallel and safe today, blocking nothing:**
`D15` (read-only n03/n04 upstream audit) · **R1 orientation verification** · `D0-002` / `D0-003` /
`D0-010` research · B4 evaluation design.

**The single longest pole is not on this diagram:** B3 and B4 together are ~10 unimplemented nodes.

---

## 7. Which Block Should Be ACTIVE First

**B1 — with B0 running concurrently, and R1 verified before either commits GPU time.**

Reasoning:

1. **Everything is behind B1's corpus.** n06, n09, Stage 1, the gallery, both tables. Nothing
   downstream can be trusted until the annotation corpus is settled.
2. **B0 is cheap, parallel-safe and unblocks the step after.** `D0-002` and `D0-003` gate n09, not
   n06 — verified: `splits.py` never reads an embedding, `encode_text_image.py` never reads anything
   n09 writes. Running them while D14 works costs nothing and removes the next wall.
3. **R1 must be answered before D16 writes G2 and before D3 spends a training run.** If the point
   clouds are mis-oriented, G2 would certify the defect and Stage 1 would train on it silently.
   The verification is read-only and cheap.

**Recommended order of the first three actions:**

| # | Action | Cost | Why now |
|---|---|---|---|
| 1 | **Verify R1** — finish the frozen-checkpoint orientation test, over a proper sample, with `pc_norm` applied identically to both sides | minutes of GPU | It can invalidate the corpus. Everything else is wasted if it is real |
| 2 | **Open `D0-010`** — B0 does the actual research §6–§11 | no GPU | It is the stated blocker on a 19.6 GPU-hour irreversible run |
| 3 | **D14 Phase 2** — the 300–500 asset sample | small GPU | Produces the numbers Phase 3's go/no-go needs |

`D15` may run alongside all three — **after its §6 factual error is corrected.**

---

## 8. Proposed Owner / Reviewer Arrangement

| Block | Owner | Reviewer | Staff now? |
|---|---|---|---|
| **B0** | Research conversation. Owns all `D0-*` decision files §6–§11 | Codex adversarial, **plus** an independent Claude reviewer for `D0-010` — it decides a 19.6 GPU-hour irreversible run | **YES** |
| **B1** | One conversation holding n02→gallery. Continues D14; then D15/D16, D1, D2, D3, D4 | Independent, read-only, **synchronous**. First assignments: (a) attack D14's Phase 2 sample before Phase 3 is proposed; (b) own the R1 orientation verification; (c) audit `D15/TASK.md` §6 | **YES** |
| **B2** | — | — | not yet |
| **B3** | — | — | not yet |
| **B4** | Evaluation conversation. Designs and unit-tests n15 against synthetic inputs | Codex on metric definitions and protocol alignment vs the paper | **YES — design only, no execution** |

**Reviewer working rules (per USER's §6):** read-only by default; own worktree or isolated output
directory if a check must execute; never touches Owner production files; reports
`FINDING / EVIDENCE / CLASSIFICATION / IMPACT / SEVERITY`; may **not** decide a material remedy.

**Codex:** targeted reviews for high-risk questions; a full adversarial review at each Block
milestone. Codex PASS ≠ project PASS. Every Codex finding is independently verified and classified
`CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED`.

---

## 9. Workflow Files Needing Restructuring

### 9.1 Stale — states a fact that the disk contradicts

| File | Problem |
|---|---|
| `workflow/MASTER.md` §4, §6, §11, §12 | Says n06 = **5,276** (is 20,053); tests = **442** (are 582); "Next Recommended Action: D10 is READY … then D1". **D10 is approved, D1 is stopped, and D14 is what is actually active.** §11 blockers B1/B5/B6 describe a world before `DL-002`/`DL-003`. **Does not mention D14, D15, D16, `D0-010`, or the n05 defect at all** |
| `workflow/CONTEXT.md` §6, §9 | Same 5,276 / 442; data root still `/home/kyzen/data/MetaFind` |
| `workflow/INDEX.md` | "Active Tasks" table is literally `— — —` while **D14 is ACTIVE**. D14 appears only in a prose line |
| `docs/PROGRESS.md` | Snapshot from 2026-08-17: "n05 running, 29,598 / 45,955", 417 tests, 2,276 checks, data 392 GB. **Every number is wrong.** Either regenerate or stamp HISTORICAL |
| `CLAUDE.md` §9 | Data root `/home/kyzen/data/MetaFind` and the repo symlink line. **Project instruction — only the USER may edit it** |
| `workflow/tasks/D15…/TASK.md` §6 | Claims `L2-PC-ULIP-REF` never ran and the reference clouds are absent. Both false (§3.3) |
| `graphify-out/` | Stale relative to `metafind/data/annotate*.py` and `workflow/**` |

### 9.2 Missing structure the Block-centric model needs

```
workflow/BLOCKS.md                          ← block registry, owner/reviewer, state
workflow/blocks/B0_research-fidelity/{BLOCK.md,REVIEW.md}
workflow/blocks/B1_ulip2-asset-retrieval/{BLOCK.md,REVIEW.md}
workflow/blocks/B2_essgnn-layout/{BLOCK.md,REVIEW.md}
workflow/blocks/B3_scene-retrieval-composition/{BLOCK.md,REVIEW.md}
workflow/blocks/B4_evaluation/{BLOCK.md,REVIEW.md}
```

**Proposal, not yet executed:** existing `workflow/tasks/D*` become **internal work items** of their
block, keeping their directories and history. `INDEX.md` stays the registry; `BLOCKS.md` becomes the
ownership layer above it. No task is renamed and no decision file moves.

### 9.3 Open ledger items needing Master action

- `DL-005` — `AWAITING_USER_REVIEW` (the `D-2`/`D-8` split, already executed in the files)
- `DL-003-A1` — `PREPARED, NOT IN FORCE`; must land **with** D14 Phase 3
- **No ledger entry exists for `U-6`** (the Qwen3.8-27B model decision) or for the **LVIS anchoring
  deviation**. Both are material and both are currently only in conversation and task files
- D14's escalation asks Master to rule on **`IC-1`** and **`IC-2`** — unanswered
- D14's `P-1` (amend `TASK.md` §7 for U-6) is granted in the task file but §7's false premise
  *"GPT-4o is unavailable"* still needs its qualification propagated

### 9.4 Uncommitted

**13 modified + 7 untracked files, none committed.** D14 Phase 1, the `D-2`/`D-8` split, three task
contracts and a decision file all live only in the working tree. Recommend a checkpoint commit
before any further work, whatever the Block decision is.

---

## 10. Immediate Risks / Blockers

Ordered by how much they can silently corrupt.

| # | Risk | Severity | Status |
|---|---|---|---|
| **R1** | **Point-cloud orientation.** Two measurements contradict each other (n=6 says fine, n=286 says 94.1% need 180°). If real: 46,052 clouds invalid, Stage 1 invalid, both tables invalid, **and no error is raised anywhere** | **CRITICAL** | **UNVERIFIED, unrecorded in the repository.** Decisive test written, timed out |
| **R2** | **D14 Phase 3 cost is not evidence-backed.** `Qwen3.8-27B` is **56 GB** at bf16 on a **32 GB** GPU → quantization is required and has **never been loaded or benchmarked**. The "~19.6 GPU-h" figure is the **7B** model's. Quantization also changes annotation quality, which makes it an experimental condition, not an engineering detail | **HIGH** | Model download complete (18/18). No load test run |
| **R3** | **SMR write cliff.** All project data now sits on `/mnt/data1` (`ST4000DM004`, SMR, `w_await` >5,000 ms measured). n05 writes 45,952 small `.json`; n06 writes 45,952 small `.npz`. **Every runtime estimate in the project predates the migration** | **HIGH** | Unmeasured since migration |
| **R4** | **Table 1 and Table 2 have zero implementation.** n15, n15a/b/c, n16, n17, n18–n22 are spec only. The reproduction **cannot produce its headline result today**, independent of training, and no Block owns it | **HIGH** | Unassigned |
| **R5** | **`D0-003` is a hard crash.** `splits.py:169-171` admits all 45,955; `stage1.py:109` has no existence guard → `FileNotFoundError` mid-epoch the first time Stage 1 reaches one of the 3 residuals | **MEDIUM** | Known, unresolved, no decision file |
| **R6** | **Governance drift.** `MASTER.md` no longer describes reality (§9.1). A task conversation reading it would act on a false project state — the exact failure the workflow exists to prevent | **MEDIUM** | Fixable immediately |
| **R7** | **20,053 embeddings look like progress and are not.** They encode the v3 categories that D14 is replacing. They must not be counted, reused, or resumed from | **MEDIUM** | Correctly protected by `DL-002`'s cache-validity mechanism |
| **R8** | **All in-flight work is uncommitted.** One careless `git checkout`/`stash` loses D14 Phase 1 and three task contracts | **MEDIUM** | Trivially fixable |
| **R9** | `graph_spec.yaml` carries no deviation entry for **LVIS category anchoring**, and `check_graph.py:373-383` compares deviation **ids only, never the text** — so a deviation whose description has gone false passes every gate silently (`FU-A`) | **LOW-MEDIUM** | Registered, unassigned |

---

## What Master is asking the USER to decide

1. **Approve or amend the Block mapping** in §5, including the two proposed adjustments
   (B1 internal lanes; staffing B4 early).
2. **Approve the first-ACTIVE ordering** in §7 — R1 verification first, then `D0-010`, then D14 Phase 2.
3. **Approve the Owner / Reviewer staffing** in §8 (B0, B1, B4 now; B2, B3 later).
4. **Rule on `IC-1` and `IC-2`**, which D14 escalated and which are still unanswered.
5. Confirm Master may **correct the stale workflow files** in §9.1 and **commit** the working tree (§9.4).

**Master has changed no code, no data, no annotation, no protocol, and no scientific artifact this
session.** The only file written is this report.
