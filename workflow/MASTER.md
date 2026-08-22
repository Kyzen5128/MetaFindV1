# MetaFindV1 — Project State

> Master maintains this file. It is the single place to look for **where the project is**.
> It is not scientific authority: the paper, the upstream sources, `DECISION_LEDGER.md`, and
> the artifacts on disk all outrank it.
>
> Structure and rules: `workflow/BLOCKS.md` · Method: `workflow/SKILLS.md`
>
> **Verified 2026-08-22.** Every number below was measured, not carried forward.

---

## 1. Goal

Reproduce MetaFind — dual-tower multimodal 3D asset retrieval with layout context — with
**evidence-backed correspondence** between the published method and this implementation.

Deliverables: Stage 1 object-level contrastive pretraining · Stage 2 layout-conditioned training
with ESSGNN · gallery index and retrieval evaluation (**Table 1**) · scene-composition evaluation
(**Table 2**) · an explicit registry of what is PAPER FACT, IMPLEMENTATION CHOICE, DEVIATION and
UNKNOWN throughout.

**Working code is not the deliverable. A traceable reproduction is.**

---

## 2. Where the project is

**Data preparation. Nothing has ever been trained.**

```
pointclouds     46,052   .npz + .json each; verified against official ULIP-2 artifacts
renders         46,045   directories, 506,495 PNG = 46,045 x 11 exactly
    of which    45,955   USABLE -- in renders_index.jsonl, with a sidecar.  <-- the real population
    and             90   directories with 11 blank PNGs, no index entry, no sidecar
scene_graphs    12,000
procthor_modalities      1,467

annotations          0   the corpus was deleted; a new one has not been produced
embeddings           0
sem_edge_*      deleted  n08 must re-run
checkpoints          0

splits.json · eval_protocols.json · stage1_protocol.json      absent
```

```
pytest tests/ -q          582 passed
tools/check_graph.py      2275 checks, all pass
GPU                       NVIDIA GeForce RTX 5090, 32.6 GB, idle
data root                 /mnt/data1/kyzen/MetaFind   (SMR drive)
free                      3.0 TB
```

**Neither the test suite nor the graph checker is evidence of paper fidelity.** They establish
that the code executes and that the spec documents agree with each other.

**The corpus denominator is 45,955** — `DL-006` (2026-08-22), `renders_index.jsonl`,
`annotate_run.py:142`, `tools/status.sh:55`, `tools/chain_after_n05.sh:56` all agree. `45,952`
is the pre-`DL-006` figure (45,955 minus the 3 deleted residuals) and is **stale wherever it
still appears** outside a historical record. The 97-asset gap from 46,052 point clouds is n04's
quarantine: `quarantine_n04_render_views.jsonl`, 143 lines / 99 unique uids, all
`DETERMINISTIC_INPUT`, e.g. *"every view is blank -- the asset never entered frame"*.

### Pipeline

```
ULIP2    n02 download → n03 pointclouds → n04 renders(11) → n05 annotate
         → n05b protocol → n06 encode → n09 splits → n10 Stage 1 train
         → n11 / G4 / n12 gallery → n15 eval ............................ TABLE 1

ESSGNN   n07 scene graphs → n07b modalities → n08 semantic edges
         → n09b / n09c → n11b index → n13 Stage 2 train → n14 probe
         → n15a/b/c → n16 compose → n17 judge ........................... TABLE 2

then     n18/n19 ablations → n20 aggregate → n21 compare → G5 → n22
```

Registry: `docs/graph/node_registry.yaml` — 38 nodes, 7 gates.

### Implementation coverage

| | |
|---|---|
| implemented and executed | n02, n03, n04, n07, n07b, n09b, n09c |
| implemented, must re-run | n05 (v5 written, never run), n05b, n06, n08 |
| implemented, never executed | n09, n10, n11, n11b, n12, n13 |
| **no implementation at all** | n10b, n14, **n15 (Table 1)**, n15a, n15b, n15c, n16, n17, n18–n22 |

---

## 3. Blocks

| Block | State | Engineer | Reviewer |
|---|---|---|---|
| **ULIP2** | **ACTIVE** — milestone 1, the annotator bake-off | unassigned | unassigned |
| **ESSGNN** | **ON HOLD** — USER decision 2026-08-22, not staffed | — | — |
| **INTEGRATOR** | **ON HOLD** — USER decision 2026-08-22, not staffed | — | — |

**Only ULIP2 is open.** The USER opens the others; Master does not staff a block on its own.
When ESSGNN opens, its first work is the Table 2 chain (n15a/b/c, n16, n17, n14) — no GPU, no
pending decision, and the longest pole in the project.

### ULIP2's milestones — each one accepted by the USER before the next begins

```
M1  annotator selection    three candidates, 300-500 sampled assets each
M2  full annotation        45,955 assets, multi-day, runs once
M3  encode + splits        needs Q-TOWER decided first
M4  Stage 1 training       the project's first checkpoint
M5  gallery index + Table 1
```

Scope, open items and evidence: `workflow/blocks/<BLOCK>/BLOCK.md`.

---

## 4. Decisions in force

Full record with evidence: `workflow/DECISION_LEDGER.md`.

| | Decision |
|---|---|
| `DL-001` | Stage 1 text serialization template — centimetres, ratified form |
| `DL-002` | Stage 1 encoding contract — cache validity, mismatch rejection, 77-token gate, pre-flight |
| `DL-003` | τ = 0.5 with `learnable_temperature: false`; protocol refreshed |
| `DL-004` | ESSGNN `f_x` stays a **scalar** coordinate multiplier. Verdict **`PAPER-AMBIGUOUS`** |
| `DL-005` | Deviation `D-2` split into `D-2` (annotation model) and `D-8` (scene judge model) — *awaiting USER* |
| `DL-006` (08-22) | The three legacy annotation residuals are deleted and re-annotated with everything else |
| ~~`DL-006` (08-21)~~ | ~~n05's annotation model is `Qwen3.8-27B`~~ — **SUPERSEDED by `DL-008`** |
| `DL-007` | n05 v5 anchors object identity on the Objaverse-LVIS label. **A DEVIATION** — *awaiting USER* |
| `DL-008` | The annotator is chosen by a lightweight bake-off. **Procedure approved; the winner is PENDING USER** |
| `DL-009` | Execution order: `ULIP2` runs to completion before `ESSGNN` opens |
| `DL-010` | MetaFind silent + component unmodified → **the official upstream implementation IS the reference** |
| `DL-011` | The ULIP2 session's USER decisions `U-A` … `U-AC`, ratified as a block. Includes **`U-A` no post-processing repairs** (project-wide) and the **`n03`/`n04` re-run authorisation**, which supersedes `BLOCK.md` §7 |
| `DL-012` | Deviations **`D-9` … `D-12`** registered under `U-Z`. Gate regex fixed — it could not see two-digit ids |
| `DL-013` | The `texture` carve-out (`D-12`) reverses a USER-confirmed choice on an inconclusive measurement — **`AWAITING_USER_REVIEW`** |

> **`DL-006` is used twice.** Two different decisions share the id. Always cite it by date.
> Registered at the top of `DECISION_LEDGER.md`. `DL-003-A1` is `PREPARED, NOT IN FORCE`.

**Standing prohibitions carried by these decisions:**

- τ = 0.5 is a **PAPER FACT** (`3experiments.tex:15`). Anything else is a registered DEVIATION.
- `learnable_temperature: false` is a **USER-RATIFIED IMPLEMENTATION CHOICE**, never a PAPER FACT.
- On `f_x`: **never write "the paper is wrong"**, and **never cite upstream EGNN as settling a
  MetaFind interpretation**. The `2.2e-16 vs 0.43` figures are unverified here and unreproducible.
- The annotation model is not GPT-4o (`D-2`), and feeding the LVIS label into the prompt is a
  **DEVIATION**. Neither may be described as paper-faithful.

### Deviations

**Eleven registered** (`D-2`…`D-12`) plus one conditional (`D-1`, not activated), in
`docs/graph/graph_spec.yaml`. Four were added 2026-08-22 under `U-Z` (`DL-012`): `D-9` LVIS
anchoring · `D-10` contrastive negatives · `D-11` white background · `D-12` the `COLOR_0`
texture carve-out.

**Still open:** n08's LLM belongs to no deviation id since `DL-005` split `D-2`. ESSGNN's, and
that block is on hold. **Debt `D-2`/`FU-A` is not discharged** — the gate still compares ids and
never reads the `what:` text, so a deviation whose description goes false passes silently.

---

## 5. Open questions

| ID | Question | Owner |
|---|---|---|
| ~~`Q-CATEGORY`~~ | **DISCHARGED 2026-08-22 (`R-6`).** It and `DL-007`'s `D0-010` are the same question, listing the identical four options. The true state is **decided, implemented, unaudited** — not "uninvestigated". `U-AB` requires the missing evidence audit (ULIP2 `W-6`) before the full annotation run | — |
| `Q-TOWER` | `tower_sharing`: `shared_backbone_separate_fusion` / `fully_shared` / `fully_separate`. Determines whether Stage 2 can freeze the gallery at all | INTEGRATOR |
| `Q-BUILDMODEL` | The Stage 1 trainer builds its model from raw protocol dicts, not from the runtime config, and passes one fusion object to both towers. `fully_separate` is therefore unimplementable as written | INTEGRATOR |
| `Q-ESSGNN-AXIS` | Whether the coupling between `coord_feat` and `architecture_family` prevents isolating `coord_feat` as an ablation axis | ESSGNN |
| `Q-NODETEXT` | n08's node text is category-only, which collapses distinct assets into the same string | ESSGNN |
| `Q-TABLE2` | How the 200 evaluation scenes are constructed, and whether a 1–5 scale is comparable with I-Design's 0–10 | ESSGNN |
| `Q-JUDGE-MODEL` | Which model scores scenes in n17. The previous one's weights were deleted | ESSGNN |
| `Q-N08-MODEL` | Which model generates n08's semantic-edge sentences | ESSGNN |
| `Q-YAW-PLACEMENT` | Our assets sit 180° yawed about Y relative to ULIP-2's released clouds. This does **not** move the embedding, but scene composition places assets with real geometry | ESSGNN |

Every one of these is a **USER decision**. Blocks investigate and recommend; they do not choose.

---

## 6. Known debts

| | Item |
|---|---|
| D-1 | `data/outputs/annotation_provenance.json` (4.9 MB) still declares `accepted_legacy_v3: 45,952` + `legacy_v1_residual_unresolved: 3` — **45,955 records, none of which exist**; `data/outputs/annotations/` is empty. Rebuild it through `tools/declare_annotation_provenance.py` — never by hand. Assigned: ULIP2 W-3 |
| D-2 | The gate checker compares deviation **ids** only and never reads their text, so a deviation whose description has gone false passes silently |
| D-3 | `tools/check_graph.py` asserts a unit-test count recorded in `docs/graph/README.md`; adding tests moves it and the gate fails until the figure is updated |
| D-4 | `sidecar_path()` performs no uid validation. Not reachable in this pipeline without a hand-written malicious input |
| D-5 | `docs/audit/F_CODE_GRAPH_CONSISTENCY.md`'s `CONSISTENT` column means "code matches the graph spec", not "code matches the paper", and does not say so |
| D-6 | `docs/PROGRESS.md` is a snapshot from 2026-08-17 and every number in it is wrong |

---

## 7. Critical path

```
ULIP2 annotator bake-off  (sample only, 300-500 assets per candidate)
   → USER picks the winner
   → n05 full run          45,955 assets, multi-day, runs ONCE
   → n06 encode            45,955 embeddings
   → Q-TOWER decided → n09 splits
   → n10 Stage 1 training  ← the project's first checkpoint
   → n11 / G4 / n12 gallery
        ├→ n15 eval  → TABLE 1        (no implementation exists)
        └→ ESSGNN: n08 rerun → n11b → n13 → n16 → n17 → TABLE 2
```

**The longest pole is not on the GPU.** Table 1 and Table 2 together are roughly ten
unimplemented nodes, and none of them needs a trained model to be designed and unit-tested.

---

## 7b. ULIP2 M1 — where it actually is, 2026-08-22 18:00

**Master's earlier W-1…W-8 list was written from a stale picture and is superseded by
`SPEC_M1_corpus_and_annotator.md`.** M1 is **corpus correction AND annotator selection**, not the
bake-off alone.

```
n03   COMPLETE   46,052 .npz + 46,052 .json, quarantine 0, ~897 assets/min flat
                 sampler_version 6 uniform -- so DEVIATION D-12 is BAKED IN
n04   RUNNING    6,280 / 46,052 at 18:44, ~730/min, quarantine 0.2% (the same
                 "blank in every view" assets as before). ETA ~54 min
n05+  NOT STARTED. No model has ever been loaded, and `compressed-tensors` is
                 absent -- so the "READY" QAT arm cannot load today (U-R)
```

**Verified by Master over the whole corpus, not taken from the block's report:**

```
sidecars / npz     46,052 / 46,052      quarantine file absent -> 0
sampler_version    {6: 46052}           no v3 or v4 survived
frame_correction   yaw180_about_y@ulip2_frame  on all 46,052
colour_source      texture 23,675 · flat 13,524 · gltf_default 8,853
color0_modulated   2,257  (4.90%)
```

**The `colour_source` distribution is identical class-for-class to the pre-regeneration corpus.**
That is not luck — it is the property the *"`colour_source` names the BASE"* choice was made to
preserve. Had a modulated texture been relabelled `vertex`, 2,257 assets would have migrated
between classes and every colour statistic, plus `G3`, would silently have read a different corpus.

⚠️ **The "21 zero-variance assets" figure is not robust and must not be diffed.** Master measured
**18** at exact equality — but the population is a **continuum**: 18 at `== 0`, **84** at `≤ 1e-30`,
106 at `≤ 1e-12`. Each is still a flat asset with one near-zero axis and two around 0.17. **18 vs
21 is a threshold effect, not a regression.** Comparing the two numbers as if the corpus changed
would be reading noise.

**The bake-off is still PAUSED** by USER decision until the `n04` finding is adjudicated —
`width_axis` and `identity_confirmed` are contaminated by tumbled renders, so all three arms
would return noise and **a bug would read as "this criterion has no discriminating power"**.
`n04` finishing is what lifts that.

**Earlier estimates in this file were wrong and are corrected:** the run is **not** 3.3 hours.
Splitting output onto NVMe took `n03` from 391 assets/min and falling to ~897 flat, so `n03` is
~51 min and `n04` ~63 min. See `CONTEXT.md` §6b — and note the trap: write bandwidth measured
only 796 kB/s, which made *"moving the output cannot help"* look correct. It is **seek**
contention, not bandwidth.

---

## 8. Immediate risks

| | Risk |
|---|---|
| **R-1** | **No annotator has ever been loaded on this machine at any scale.** Measured 2026-08-22: the two Gemma arms are on disk and fit (`gemma-4-31B-it-qat-w4a16` 22,188 MiB official QAT · `gemma-4-12B-it` 22,812 MiB bf16), but **neither has been loaded, and `Qwen3.8-27B` exists only as 55.56 GB bf16 — its w4a16 build has never been produced.** No full-run cost estimate is evidence-backed; the `~19.6 GPU-h` figure in `evidence/n05_v5_design.md:141` belongs to the old 7B model |
| **R-2** | **The data root is an SMR drive.** Small-file write bursts collapse to single-digit MB/s once its cache fills. n05 and n06 each write ~46,000 small files. Every runtime estimate predates the move |
| **R-3** | **Table 1 and Table 2 have no implementation and no owner's attention.** The reproduction cannot produce its headline result today, independent of training |
| **R-4** | **The project has no annotation corpus at all.** Deliberate — the old one disagreed with the dataset's own labels on most of the corpus — but nothing downstream can run until a full run succeeds |

---

## 9. Master's operating rules

- Maintain the global view; do not spend this context on long implementation, training, data
  processing, or deep single-question research.
- Propose the next work; the USER approves scope before a block starts.
- Receive block handoffs, re-verify the load-bearing claims yourself, integrate.
- Request Codex adversarial review at milestones.
- Run the **USER Acceptance Grill** at every milestone — one criterion per round.
- Report **FINDING** and **DECISION** separately, always.
- Write anything that must survive into `workflow/`. Conversation is not storage.
