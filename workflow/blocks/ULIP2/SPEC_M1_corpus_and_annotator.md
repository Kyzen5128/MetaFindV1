# SPEC — ULIP2 / M1 · corpus correction and annotator selection

> Written by the Block Owner before implementation, from the USER grill of 2026-08-22.
> The single reference the 4-axis review, the Reviewer, Codex and USER acceptance all read from.
> Decisions of record and their evidence: `workflow/blocks/ULIP2/HANDOFF.md`.

**Block:** ULIP2 · **Milestone:** M1 · **Owner:** ULIP2 Engineer · **Reviewer:** *unassigned* ·
**Date:** 2026-08-22 · **Baseline commit:** `adf3df2`

---

## 1. OBJECTIVE

Today the render corpus is geometrically wrong in a way nothing detects: `n04`'s camera orbits
`+Z` while the meshes are `Y`-up, so every asset tumbles instead of turning, and the sidecars
describe an orbit that was never performed. The annotator has never run at all under
`PROMPT_VERSION 5`, no candidate model has ever been loaded, and the training hyperparameters are
invented rather than measured.

After M1: **the object corpus is geometrically correct and its correctness is demonstrated against
an official upstream artifact**, and **one annotator has been chosen on measured evidence from
this hardware**, with the full-corpus run still gated behind an explicit USER authorisation.

M1 does **not** produce the annotation corpus. It produces the corrected inputs and the evidence
needed to choose how to produce it.

---

## 2. SOURCE OF TRUTH

Authority order for this milestone, highest first:

| # | Source | Used for |
|---|---|---|
| 1 | `docs/paper/metafind_source/{2methdology,3experiments,neurips_2025}.tex` | view count (11), the Figure 2 annotation field set, the fusion selection, τ, the 80/20 split |
| 2 | `docs/paper/metafind_source/data-preprocess.png` (Figure 2) | the annotation JSON, read directly from the figure |
| 3 | `docs/paper/ulip2_source/main.tex:677` and its `top_k_captions` ablation | render count/spacing upstream, the generate-then-CLIP-rank method |
| 4 | `/home/kyzen/upstream/ULIP` @ `95d480fe` | `pc_norm`, colour flag, point count, the all-gather loss, official hyperparameters |
| 5 | **ULIP-2's released Objaverse renders**, `SFXX/ulip` `render_images_resized_224/objaverse_rgb_chunk_0000.tar.gz` | the measured camera target: orbit axis, background, framing, elevation |
| 6 | `workflow/DECISION_LEDGER.md` | `DL-001` … `DL-009` |
| 7 | `docs/graph/01_GRAPH_SPEC.md` §11, `docs/graph/node_registry.yaml` | gate criteria and node contracts |
| 8 | the repository, then tests, then runtime | current behaviour only |

**Standing precedence rule, USER-adopted 2026-08-22 (`U-O`):** where MetaFind states something,
MetaFind governs; where MetaFind is silent, ULIP-2 governs. A value chosen under `U-O` is an
`IMPLEMENTATION CHOICE` with upstream provenance and **never** a `PAPER FACT`.

### How conformance with ULIP-2 is reported — `R-11`, USER-ruled 2026-08-22

> 「他是你參考的架構那必須知道啊 除非你有特別做什麼設定」

**ULIP-2 is this project's reference architecture, not a third party.** Our point encoder *is* its
frozen checkpoint. Three consequences bind every claim in this milestone:

| | Rule |
|---|---|
| **Default** | Agreeing with ULIP-2 is the expected state. **State it with the measurement and its `n`** — not as something requiring justification, and not as a deviation |
| **Divergence** | Only a place we **deliberately chose differently** is a `DEVIATION` and needs a registry id |
| **Never claimable** | That we ran the same code. **ULIP-2's rendering and sampling code was never published**, so procedural identity cannot be asserted in either direction |

**Supersedes** the earlier Reviewer position that agreement with ULIP-2 may never be stated.
Procedural identity stays unclaimable; **artifact agreement is measurable and must be quantified.**

Applied to this milestone:

| | Claim | Status |
|---|---|---|
| Point-cloud geometry | `FIND-7`: R@1 **98.0%** against ULIP's official features | conformance, stated |
| Renders | `S-5`: R@1 **97.2%** (arm E, n=286) against 83.2% for the v2 corpus | conformance, stated |
| `COLOR_0` handling | `R-10`: cosine mean **0.9004**, median **0.9195**, n=130 through the frozen encoder | conformance, stated |
| **White background** | Chosen **against** upstream's black, on upstream's own metric | **DEVIATION — needs a registry id (`U-Z`)** |

**Only the last line is a deviation.** The first three are the reference architecture behaving as
the reference architecture, and each carries its measurement and its population.

**Unresolved conflict, carried:** `U-09` — whether the paper's retrieval gallery is the 20% test
split or the full corpus. Both protocols are produced; neither is presented as the paper's.

---

## 3. INPUTS

| Artifact | Count | Producer | State |
|---|---|---|---|
| `data/datasets/objaverse-lvis/glbs/**.glb` | **46,052** | n02 | verified complete, 0 missing, 0 extra |
| `lvis.json` | 46,052 uids | n02 | `sha256 ba1bb191e1d98252e19e59aa18185ea46b83d9955b28ba895133536974e8bf9f` |
| `objaverse_lvis_metadata.json` | 46,207 uid→category | n02 | `sha256 38e66d5b6cf38f19b1bd174943caa14c2f8952006d943e2f99ba0826b82d095e` |
| `data/outputs/pointclouds/*.npz` + sidecars | 46,052 + 46,052 | n03 | **to be regenerated** (yaw) |
| `data/outputs/renders/**` | 46,045 dirs · 506,495 PNG · 45,955 sidecars | n04 | **to be regenerated** (up-axis, background, framing) |
| `metafind/data/lvis_synsets.json` | 1,156 | D14 | accepted |
| ULIP-2 checkpoint | 402 MB | n02 | behaviour evidenced by `FIND-7` |
| ULIP-2 official renders, chunk 0000 | 50,000 PNG · 4,167 assets · **217 overlap ours** | upstream | measurement target only, never an input to our corpus |
| `gemma-4-31B-it-qat-w4a16` · `gemma-4-12B-it` · `Qwen3.8-27B` | 22,187 · 22,811 MiB · 55.56 GB | n02 | **none has ever been loaded** |

---

## 4. OUTPUTS

| Path | Count | Consumed by |
|---|---|---|
| `data/outputs/pointclouds/` | 46,052 `.npz` + sidecars, corrected frame | n06, n10, n11, G2 |
| `data/outputs/renders/` | 46,045 dirs × 11 PNG, corrected camera | n05, n06, G3 |
| `data/outputs/logs/{pointclouds,renders}_index.jsonl` | 46,052 / 45,955 | n05, n06, n09 |
| `data/outputs/gate_records/G1_*.json` | 1 | G5, the report |
| `data/outputs/bakeoff/<arm>/{annotations,raw,metrics.json,timing.jsonl,quarantine.jsonl}` | 100 records per arm | USER selection |
| `workflow/blocks/ULIP2/bakeoff/sample_100.jsonl` + `.sha256` | 100 | all three arms |
| `workflow/blocks/ULIP2/bakeoff/human_adjudication.jsonl` | 20 | scoring all three arms |
| `data/outputs/annotation_provenance.json` | rebuilt, 0 declared records | `annotate_run` |
| `metafind/eval/` + tests | new module | n15 |

**`data/outputs/annotations/` must hold 0 files at every point in M1.** It belongs to the full
run alone.

---

## 5. SCOPE

1. **Correct `n04`'s camera**: orbit about the mesh up axis, black background, ≈0.60 frame fit.
2. **Correct the 180° yaw at the mesh-load layer**, so `n03` and `n04` inherit one fix.
3. **Solve ULIP's elevation** from its released renders instead of guessing it.
4. **Verify on 100 assets, then regenerate** `n03` (46,052) and `n04` (46,045).
5. **Settle `F-N03-1`** — the 8,853 glTF-default-white clouds — against ULIP's official clouds,
   **before** the regeneration, so a fix lands in the same pass.
6. **Implement `G1_sources_valid`** and the gate-record schema the five later gates inherit.
7. **Rebuild `annotation_provenance.json`** (debt `D-1`).
8. **Build the multi-arm annotator runner** with per-arm output isolation.
9. **Build the `n05` v6 prompt**: two-turn blind-then-anchored identification, the paper's
   Figure 2 exemplar, an observe-before-answer段, constrained JSON decoding, five description
   candidates re-ranked by CLIP ViT-Large.
10. **Widen the validator's dimension floor** so the 103 currently-impossible assets pass.
11. **Run the bake-off**: 100 stratified assets × 3 arms, one arm at a time.
12. **Design and unit-test `metafind/eval/` (n15)** against synthetic inputs.
13. **Correct the fusion default to `transformer`** and fill `variant_registry.json`'s `full` row.
14. **Correct the four stale `24 GB` comments.**

## 6. NON-SCOPE

- **The full annotation run.** Gated on the USER, on the bake-off result, and on `W-5`.
- `n06`, `n09`, `n10` execution. `n09` is additionally blocked on `Q-TOWER`, which is Master's.
- Anything in ESSGNN. `DL-009` holds that block closed.
- Re-deciding `DL-001` … `DL-009`.
- Resolving `U-01`, `U-09`, `Q-CATEGORY`, `Q-TOWER`, `Q-BUILDMODEL`.
- Choosing the winning annotator. The bake-off measures; the USER selects (`DL-008`).

---

## 7. PAPER / UPSTREAM AUTHORITY

| Behaviour | Authority | Citation | Class |
|---|---|---|---|
| 11 views per asset | MetaFind | `2methdology.tex:28`, `neurips_2025.tex:100` | **PAPER FACT** |
| Annotation field set (13 fields) | MetaFind | Figure 2, `data-preprocess.png` | **PAPER FACT** |
| `volume` is `w × l × h` | MetaFind | Figure 2: `30 × 30 × 40 = 36000` | **PAPER FACT** |
| Fusion default = `transformer` | MetaFind | `3experiments.tex:143` *"the final selected Transformer"* | **PAPER FACT** — corrects a false `UNKNOWN` at `fusion.py:22-29` |
| Masking, not zero-padding; p = 0.3 | MetaFind | `2methdology.tex:75`, `3experiments.tex:143` | **PAPER FACT** |
| τ = 0.5 | MetaFind | `3experiments.tex:15` | **PAPER FACT** |
| 80/20 split | MetaFind | `3experiments.tex:8` | **PAPER FACT** |
| R@1 / R@5, instance-level | MetaFind | `3experiments.tex:18`, `:24` | **PAPER FACT** |
| Single-axis orbit, 30° spacing | ULIP-2 | `ulip2_source/main.tex:677` | **UPSTREAM FACT** |
| Orbit is about the asset's **up** axis | ULIP-2 renders | vase `9f1335d8…` holds `h/w = 1.94` across all 12 views | **OBSERVED DATA** |
| Black background | ULIP-2 renders | measured corner luminance `0` on both assets | **OBSERVED DATA** |
| ≈0.60 frame fit | ULIP-2 renders | longest side ÷ 224 = 0.61 / 0.60 on two very different shapes | **OBSERVED DATA** |
| Generate-many, rank by CLIP, keep top-1 | ULIP-2 | `main.tex:677`; ablation 69.7 / 66.7 / 66.4 / 66.3 | **UPSTREAM FACT** |
| Rank with CLIP ViT-Large | ULIP-2 | `main.tex:677` | **UPSTREAM FACT** |
| 10,000 points, colour on, `pc_norm` then concat rgb | ULIP-2 | `dataset_3d.py:456-505` | **UPSTREAM FACT** |
| Contrastive negatives gathered across GPUs | ULIP | `models/losses.py:38-40` | **UPSTREAM FACT** |
| `betas (0.9, 0.98)`, `warmup 1`, `wd 0.1` | ULIP | `main.py:47-60` | **UPSTREAM FACT**, adopted under `U-O` |
| LVIS category anchoring | — | `DL-007` | **DEVIATION** |
| Annotator is not GPT-4o | — | `DL-005`, `DL-008`, deviation `D-2` | **DEVIATION** |

---

## 8. IMPLEMENTATION CHOICES

| Choice | Reason | What would revisit it |
|---|---|---|
| Camera **elevation = solved from ULIP's renders**, not chosen | The vase is rotationally symmetric, so its constant `h/w = 1.94` is a function of elevation alone | A better upstream source, or MetaFind stating one |
| Orbit is a **single horizontal ring**, not a sphere lattice | ULIP-2's *"spaced equally by 360/12 degrees"* describes a circle. The Fibonacci variant was the Owner's preference and had no provenance | MetaFind stating a layout |
| **Five** description candidates | ULIP-2 generates ten with a small model; ten from a 27B over 45,955 assets is unaffordable. **A cost choice, not an upstream figure** | The bake-off's recorded CLIP-score spread showing five is still on the improving part of the curve |
| Bake-off sample **4 strata × 25** | 100 assets over the original 12 cells is ~6 per cell and measures nothing | A larger sample budget |
| Sample selection by `sha256(f"{SEED}:{uid}")` rank, `SEED = 20260822` | Independent of language version, list order and platform, unlike `random.sample` | — |
| Validator dimension floor widened | 103 assets currently have an **empty** feasible height band for **any** model output. They are posters and decals; a sub-millimetre thickness is correct for them | Evidence that any of the 103 is malformed rather than flat |
| Training split becomes **72 / 8 / 20** | `lr` and `epochs` have no source at all; a validation split is the only way to measure rather than invent them. **The paper's 20% test is untouched** | MetaFind publishing implementation details |
| `n10b` certifies before re-encoding | OpenCLIP is frozen in Stage 1, so `n06`'s text/image vectors are already final. A silent no-op and a silent stale-embedding bug are indistinguishable otherwise | `actual_clip_train_scope` becoming `trainable` |

---

## 9. KNOWN DEVIATIONS

| | Expected | Reproduced | Reason | Impact on comparability | Registry id |
|---|---|---|---|---|---|
| `D-2` | GPT-4o annotates | a local VLM annotates | GPT-4o availability is **UNRESOLVED**, never established | Table 1's absolute values shift | `D-2` ✅ |
| **LVIS anchoring** | the VLM generates the category | the dataset's label is supplied and only refined downward | a 7B-class model asked to identify a 224×224 render collapses onto priors — `toy` at 3.4% | every text embedding | **`DL-007`; NO registry entry — must be created** |
| **`F-N10-1` negatives** | 512 in-batch negatives (64 × 8 GPUs, all-gathered) | one GPU's batch | 512 clouds of 10,000 points with backprop will not fit 32,607 MiB. **Gradient accumulation does not raise the negative count** | first-order term of the contrastive objective; a candidate explanation for any Table 1 shortfall | **none — must be created, with its measured value** |
| **Corpus size** | ~48,000 assets | 46,052 → 45,955 usable | `U-01`; the manifest is what exists | ~1,558 fewer training and ~390 fewer test assets under the 80/20 split | **none — carried as a stated Table 1 limitation** |

**Three of the four have no registry id. `check_graph.py:373-383` compares deviation ids only and
never reads the `what:` text (`D-2`/`FU-A`), so a missing or falsified entry passes every gate
silently.** Registry ownership sits with the Integrator, which `DL-009` holds closed — this is a
`MASTER-IMPACTING` routing item, not an engineering task.

---

## 10. UNKNOWN

| | What is unknown | Checked | What would resolve it |
|---|---|---|---|
| `U-01` | Why 46,052 vs "approximately 48,000" | ULIP's `lvis.json` is the source and ULIP uses its keys identically (`dataset_3d.py:463-470`) | Objaverse's own LVIS annotation file, or MetaFind stating its manifest. **USER decided to carry it** |
| `U-03a` | Projection: orthographic or perspective | MetaFind says *"orthogonal viewpoints"*, which is not a projection model; ULIP publishes no render code | Parallax measurement across ULIP's 12 released views — **available and not yet performed** |
| `U-09` | Gallery scope for Table 1 | Table 1's values sit far above chance for either pool | Nothing available. **Both protocols reported** |
| `U-13` | ~~Fusion default~~ | **RESOLVED** — `3experiments.tex:143` names the Transformer | — |
| `U-14` | How 11 views become one vector | MetaFind silent; **ULIP never faces the question** — it keeps 12 and samples one per step, so `U-O` does not reach it | Storing all 11 (`U-V`) keeps every option open at 2.6 GB |
| `F-N03-1` | Whether ULIP's clouds for the same uid are also white | 8,853 assets are glTF-default; **`FIND-4` never stratified by `colour_source`** | The differential comparison in scope item 5 |
| `IC-1` | Whether `identity_confirmed` was ever more than a rubber stamp | It was asked **after** the answer was supplied | The two-turn design makes it computed rather than asked. Reviewer `W-7` still audits it |

**Absence of evidence is not evidence of absence.** No item above may be written up as settled.

---

## 11. SUCCESS CRITERIA

| # | Criterion | Measurement | Population |
|---|---|---|---|
| **S-1** | Tall assets render upright | image `h/w` tracks `extent_y / extent_x`, not `extent_z / extent_y`. **Current: log-log correlation `+0.893` for the wrong model and `−0.671` for the right one — the signs must swap** | ≥120 assets, `default_rng(20260822)` |
| **S-2** | The orbit is about the up axis | a rotationally symmetric asset holds a constant image `h/w` across all views, as ULIP's vase does at `1.94 ± 0.01` | the 217 assets shared with ULIP's chunk |
| **S-3** | **Background is WHITE, and the divergence from upstream is recorded** — `[255,255,255,255]`, mean corner luminance `= 255` | `USER DECISION U-W`. **Rewritten 2026-08-22; as originally written this criterion demanded luminance `0` and would now fail on correct artifacts.** The engineer had changed the background to black to match ULIP, and the Reviewer measured that on `S-5` black **costs 1.4 points** (95.8% against white's 97.2%). The USER's reasoning, recorded verbatim: a criterion cannot be used when it wins and set aside when it loses. **This is a deliberate divergence from upstream on upstream's own metric, and therefore a DEVIATION needing a registry id** — routed to Master, see §9 | all regenerated renders |
| **S-4** | **Framing stays at `xmag 1.10`, and what it produces is recorded rather than targeted** | `USER DECISION U-X`. **Rewritten 2026-08-22.** The original `0.60 ± 0.03` was fitted to `xmag 1.20` on 8 assets and **would now fail** under 1.10. ULIP's own longest-side ÷ 224 ranges 0.405–0.701 across 8 assets and the per-asset ratio to ours spans 1.16–1.83, so upstream has no single framing constant to match — 1.20 reproduced its *mean*, not its rule, and bought nothing on `S-5`. The criterion is now: measure it, record it, and assert only that **no asset is clipped** (foreground must not touch the frame edge), which is the property that would actually corrupt data | ≥200 assets |
| **S-5** | **The correction is confirmed against an official upstream artifact** | `FIND-9`'s harness re-run: R@1 vs ULIP's `image_feat` rises from the v2 corpus's **83.2%** toward the point tower's 98.0%. **Already measured on the decided configuration (`U-W` + `U-X`) by the Reviewer: R@1 97.2%, R@5 99.3%, matched 0.9160, gap 0.3734, n=286** — a **14.0**-point rise. Both parties' independent measurements of v2 and v3 agree to four decimals | 286 assets, mismatched pairs as control, chance R@1 0.35% |
| **S-6** | The yaw is gone | median Chamfer distance vs ULIP's clouds **drops from 0.0903 to ≈0.0230**, and the ">0.1" count from 137/286 to ≈7/286 | the 286 uids overlapping shard 000-009 |
| **S-7** | Nothing else moved | point count, `rgb_scale`, `coloured_point_fraction`, `max_radius`, `centroid_offset` unchanged; **flat clouds are explained by flat meshes** — `L1-PC-NONDEGENERATE`. **Rewritten 2026-08-22; the original demanded "the 21 zero-variance assets stay exactly 21" and FAILS on a correct corpus (18).** See §11.1 | all 46,052 |
| **S-8** | `G1` produces the project's first gate record | verdict, criteria, measurements, `is_terminal`, and a schema the five later gates inherit | — |
| **S-9** | No model can defeat the dimension floor | the "empty feasible band" count falls **from 103 to 0** | all 45,955 |
| **S-10** | Every arm produces 100 valid records | parse failures, repair exhaustions, quarantines all recorded per arm | 3 × 100 |
| **S-11** | `identity_confirmed` is not a rubber stamp | the blind turn-1 guess is recorded **verbatim**, and its agreement rate with the LVIS anchor is reported per arm | 3 × 100 |
| **S-12** | `data/outputs/annotations/` holds 0 files | counted **before and after every arm**, recorded in `metrics.json` | — |
| **S-13** | `n15` runs end-to-end on synthetic inputs | 7 conditions × 2 protocols; every cell carries `n_gallery` and `n_query` | synthetic |
| **S-14** | `pytest tests/ -q` stays green and grows | current baseline **582 passed** | — |

**Pre-registered auto-proceed criteria** (the USER's compression of stop points 1–3): the Owner
continues without asking **only** if `S-1` … `S-6` all pass on the 100-asset trial. **Any single
failure stops and reports.** Good numbers are permission to continue; they are never permission to
skip a later gate.

### 11.1 Why `S-7`'s zero-variance line was replaced rather than re-fitted

**The criterion as written fails, and the failure is real: `21 → 18`.** That is recorded here and
not deleted. What follows is why 18 is the correct number and 21 was never a measurement of the
property `S-7` exists to protect.

**`FRAME_CORRECTION` is `(x, y, z) → (−x, y, −z)`. Negating an axis leaves that axis's variance
unchanged in exact arithmetic**, so the correction cannot alter per-axis variance by any geometric
mechanism. It alters the *arithmetic path* — the correction composes into the scene graph's node
transforms, so vertex coordinates differ in their last bits. Three assets sitting at the `1e-33`
boundary crossed from exactly `0.0` to approximately `0.0`. **`== 0` on a float variance is a
threshold on a continuum**, measured over all 46,052:

```
min per-axis variance, cumulative        == 0       18
                                         <= 1e-30   84
                                         <= 1e-20   84
                                         <= 1e-15   88
                                         <= 1e-12  106
```

**The trap in picking a different epsilon.** `U-AE` deleted the old corpus, so no *new* statistic
has a baseline — swapping `== 0` for `<= 1e-20` trades the one before/after that exists for a
number comparable to nothing. Choosing an epsilon after seeing the distribution is also the exact
move `S-3` records the USER forbidding: a criterion cannot be used when it wins and set aside when
it loses.

**The replacement is not a new criterion.** `docs/graph/validation_plan.yaml`
`L1-PC-NONDEGENERATE` already defines this check, it predates the regeneration, and
`docs/graph/` outranks this SPEC in the project authority order. Its own note says why:

> *An absolute variance floor cannot tell a flat ASSET from a flattening BUG. … Quarantining them
> would discard valid data, and raising the floor until they pass would blind the check to the
> failure it exists for. Comparing against the mesh separates the two.*

**`S-7` was restating that check badly.** A count of flat clouds is a proxy; the property is
*whether a flat cloud is explained by a flat mesh*. Measured on all 46,052 sidecars — both
`per_axis_variance` and `raw_bbox_extents` are recorded, so no `.npz` is read:

```
assets with >=1 axis variance <= 1e-12       106
of those, mesh NOT correspondingly flat        0      <-- the failure condition
flat-axis extent / largest extent    max 7.004e-04    median 2.220e-16
```

**0 violations at every threshold from `0` to `1e-12`**, which is the property that makes this
criterion usable where the count is not: it does not move when the epsilon moves.

**What is lost and is not hidden.** `21 → 18` cannot be verified asset-by-asset, because the old
corpus is deleted under `U-AE`. The explanation above is the algebra plus the value distribution,
**not a paired comparison**. `INFERENCE`, not `OBSERVED DATA`. The `106 / 0` measurement above is
`OBSERVED DATA` on the new corpus alone.

## 12. FAILURE CONDITIONS

Loud failures: `G1` FAIL · no arm loads · regeneration changes the corpus count · `pytest` regresses.

**Silent failures — the ones this milestone actually exists to prevent:**

1. **Renders look plausible and are still wrong.** A vertical orbit at the wrong elevation, or a
   correct orbit about the wrong axis for asymmetric meshes, produces perfectly reasonable images.
   `S-2` and `S-5` are the guard; a visual check is not.
2. **`S-5` is achieved by fitting to ULIP rather than by being correct.** The elevation is solved
   from **one** rotationally symmetric asset and verified on **held-out asymmetric** assets. If the
   two disagree, that is a finding, not a tuning knob.
3. **The bake-off measures prompt compliance and is read as identification accuracy.** Under v5/v6
   the anchor is supplied, so `category`-vs-LVIS and top-20 concentration are near-tautological.
   **They are reported as compliance checks and must never be used to rank the arms.**
4. **A v6 pipeline defect is read as a model weakness.** All three arms fail the same way. The
   stub-model pre-flight, the cross-arm comparison and the 103-asset control group separate them;
   **if all three arms are simply poor, this experiment cannot tell a bad prompt from three weak
   models, and that must be stated rather than resolved.**
5. **The description passes every check and is factually wrong about colour or material.** No
   automatic check catches it. **The 20 human adjudications are the only defence.**
6. **`n10b` silently does nothing** because OpenCLIP was frozen — indistinguishable from silently
   using stale embeddings unless it certifies.
7. **A deviation with no registry entry.** Three of four currently have none, and the gate checker
   reads ids only.

## 13. SELF-VERIFICATION REQUIREMENTS

| Seam | Existing / new | What it tests | Expected-truth source |
|---|---|---|---|
| `renders.azimuth_orbit_directions()` | existing | directions vary in the two axes ⟂ to up; the up component is constant | Pure geometry — a ring about `u` has constant `d·u` |
| `renders.normalised_scene()` | existing | the corrective rotation is applied once, is a rotation (`det = +1`) and is not a reflection | `FIND-6`: `(x,y,z) → (−x,y,−z)`, determinant `+1` |
| `renders.render_views()` → PNG | existing | background luminance `0`; longest side ÷ 224 ≈ 0.60 | **ULIP's released renders**, measured |
| whole-pipeline, pixels | new | a tall asset's image `h/w` tracks `extent_y/extent_x` | **LVIS categories as external labels**, never our own annotations |
| `annotate.validate_annotation()` | existing | the 103 previously-impossible assets now admit at least one height | Mesh `raw_bbox_extents` — measured, not asked |
| `annotate.build_prompt()` | existing | turn 1 contains **no** LVIS category; turn 2 does | The two-turn design's whole purpose |
| `annotate_run.build_work_list()` | existing | `--arm` refuses when the out-dir resolves to `paths.ANNOTATIONS` | The isolation requirement itself |
| `metafind/eval/` | new | 7 conditions × 2 protocols on synthetic embeddings with a known answer | A hand-built fixture whose correct ranking is known by construction |

**Expected-Truth Provenance Rule.** Every expected value above comes from the paper, from ULIP's
released artifacts, from the source dataset's own metadata, or from geometry. **None comes from
what our code currently returns.** Where no seam exists — the factual truth of a free-text
description — that is stated in §12.5 rather than papered over with a weaker test.

Beyond the seams, the Owner verifies: artifact integrity · provenance · dataset consistency ·
upstream and downstream consistency · semantic sanity · paper consistency · failure cases · resume
and cache correctness · silent failure.

## 14. INDEPENDENT REVIEW REQUIREMENTS

**Reviewer is unassigned.** `BLOCKS.md` requires an independent, **synchronous** review before any
expensive run. **This is a staffing blocker the USER must resolve**, not something the Owner may
waive.

Before the 100-asset trial: the corrective rotation is a rotation and not a reflection · the
elevation solution is not fitted to the same asset it is validated on · the sample-selection rule
is genuinely reproducible.

Before the full regeneration: `S-1` … `S-6` independently reproduced, not read from the Owner's
report.

Before the bake-off: `W-5` (is `Q-CATEGORY` the same question `DL-007` already answered?) ·
`W-6` (the `D0-010` evidence audit that was never done) · `W-7` (can `identity_confirmed` be
checked at all at n=100?).

**A review that begins after the run is a post-mortem.**

## 15. MILESTONE CRITERIA

- [ ] `S-1` … `S-14` met, each with its measurement and population
- [ ] Owner self-verification complete
- [ ] Reviewer independent verification complete — **blocked: unassigned**
- [ ] 4-axis code review, all four reported separately
- [ ] Codex milestone review, or `CODEX REVIEW UNAVAILABLE` stated
- [ ] every UNKNOWN resolved or explicitly carried
- [ ] every DEVIATION registered — **three currently have no registry id**
- [ ] experiment provenance recorded per `.claude/rules/experiments.md`

**Execution complete is not acceptance.** Only the USER's `APPROVE`, item by item, is.
