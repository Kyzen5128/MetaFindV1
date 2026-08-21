# D-Task Execution Contract

> This file is the authoritative execution contract for one bounded work package.
> The task owner must stay within scope, satisfy the Definition of Done, perform verification, obtain Codex review, and return a HANDOFF to Master.

---

## Task ID

`D15_n03-n04-code-audit`

---

## Status

Current:

`PLANNED` — contract written by Master 2026-08-21. **Not approved. Do not start.**

> Finishing execution does not make a task `DONE`. See Section 17 of `workflow/tasks/TEMPLATE.md`.

---

## 1. Objective

Establish, against primary upstream and paper sources, **what `n03_sample_pointclouds` and `n04_render_views` actually implement**, and classify every research-significant behaviour of both as `UPSTREAM FACT` / `PAPER FACT` / `IMPLEMENTATION CHOICE` / `DEVIATION` / `UNKNOWN`.

**This task is READ-ONLY.** It changes no code, no data, no artifact, no test. Its single deliverable is a findings document.

---

## 2. Why This Task Exists

Both nodes have already produced the corpus the whole reproduction rests on:

- `pointclouds/` — 46,052 `.npz`, 100% of the manifest
- `renders/` — 46,045 directories, 45,955 admitted to the index

Both have unit tests that pass (16 for n03, 11 for n04, inside the 582-test suite). **Passing unit tests is evidence that the code does what the test says. It is not evidence that the test asserts the right thing.**

Two of those tests make explicit upstream/paper claims that have never been checked against the actual source:

| Test | Claim | Checked against source? |
|---|---|---|
| `tests/test_pointclouds.py:51` `test_pc_norm_matches_ulips_definition` | our normalisation == ULIP's | **No** |
| `tests/test_renders.py:47` `test_primary_layout_is_the_ulip2_style_orbit` | our camera layout == ULIP-2's | **No** |

Until 2026-08-21 the upstream source was not on this machine. It now is:

```
/home/kyzen/upstream/ULIP    commit 95d480fe2b16c06d0558c60b5cfea981b4cdc8eb
/home/kyzen/upstream/egnn    commit e9ca6c0c3e1d30a7598efbd66034121b4af8dccc
```

**Why now.** `n03`'s output is the input to Stage 1 training, and it is the artifact `G2_pc_sanity` (`D16`) is supposed to gate. If G2's criteria are written against an unverified understanding of what n03 produces, the gate bakes in the error and then certifies it. **The audit must precede the gate.**

**Downstream.** `D16_gates-g1-g2` depends on this task's findings for G2's criteria. `D3_stage1-train` consumes n03's clouds directly.

---

## 3. Required Shared Context

Before execution, read:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/research-rigor.md` and `.claude/rules/paper-reproduction.md`
3. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
4. this `TASK.md`

Then read only the sources listed under Section 5.

Do not automatically re-read the entire repository.

---

## 4. Dependencies

### Required Before Start

- Upstream repositories present on disk (**satisfied** 2026-08-21, commits recorded above)
- No other task dependency

### Blocks

- `D16_gates-g1-g2` — G2's criteria depend on FIND-n from this task

### Parallel Safety

`PARALLEL SAFE: YES`

Reason: the task is read-only. It writes exactly one new file, inside its own task directory, that no other task reads while running.

Potential filesystem conflicts: **none.** Writes only `workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md` and `HANDOFF.md`.

---

## 5. Authoritative Inputs

Ordered by the authority hierarchy in `CLAUDE.md` §3.

| # | Source | Why it matters |
|---|---|---|
| 1 | `docs/paper/metafind_source/2methdology.tex` §2.3 | The only place the paper constrains rendering: "11 orthogonal viewpoints". The count is stated; nothing else is |
| 2 | ULIP-2 paper (arXiv 2305.08275) | MetaFind's point encoder is ULIP-2's. What ULIP-2 specifies about its own preprocessing is UPSTREAM FACT — but only becomes MetaFind-relevant where MetaFind adopts it |
| 3 | `/home/kyzen/upstream/ULIP/data/dataset_3d.py` | The official implementation. `pc_normalize()` at **:33-38**; further `pc_norm` methods at **:381** and **:496**. **Which one the Objaverse/ULIP-2 path actually uses is the first thing to determine — they may differ** |
| 4 | `/home/kyzen/upstream/ULIP/` — sampling, view rendering, dataset construction | Whether ULIP-2 uses farthest-point sampling, how many points, whether it renders its own views and with what camera layout |
| 5 | `docs/graph/node_registry.yaml` — `n03_sample_pointclouds`, `n04_render_views` | The project's recorded postconditions, UNKNOWNs U-02, U-03, U-03a, and finding F13 |
| 6 | `metafind/data/pointclouds.py` | Current implementation. OBSERVED IMPLEMENTATION only |
| 7 | `metafind/data/renders.py` | Current implementation. OBSERVED IMPLEMENTATION only |
| 8 | `tests/test_pointclouds.py`, `tests/test_renders.py` | What is currently asserted — and, critically, what is **not** |
| 9 | `data/outputs/pointclouds/*.json` sidecars, `data/outputs/logs/pointclouds_index.jsonl` | OBSERVED DATA. 46,052 records carrying `centroid_offset`, `max_radius`, `per_axis_variance`, `raw_bbox_extents` |
| 10 | `data/outputs/renders/<uid>.json`, `data/outputs/logs/renders_index.jsonl` | OBSERVED DATA. 45,955 admitted records |

---

## 6. Current Relevant State

**n03 — OBSERVED IMPLEMENTATION / OBSERVED DATA**

- 46,052 `.npz` + 46,052 sidecars on disk. `pointclouds_index.jsonl` = 46,052 lines. No quarantine file for n03.
- Sampling is **area-weighted over mesh faces** with a per-uid seed (`test_allocation_is_area_weighted`, `test_seed_depends_on_uid_only`). Whether ULIP-2 samples this way is **unverified**.
- 10,000 points, xyz+rgb. The `(10000, 6)` shape is in the node postcondition. Whether 10,000 is ULIP-2's number or ours is **unverified**.
- Sidecar statistics are computed in **float64** while the cloud is stored **float32**. `pointclouds.py:380-389` records the reason: float32 summation over 10,000 values accumulates ~1e-5 of error, which exceeds G2's 1e-5 centroid tolerance, and **8 of 46,052 assets were recorded as failing a check their data passes**. One was re-verified in float64: offset `5.2e-09`, not `1.15e-05`. **This is a live interaction with D16 and must be carried into that task's criteria.**
- The completion marker is the **sidecar**, written last (`is_complete()`, `pointclouds.py:310`).

**n04 — OBSERVED IMPLEMENTATION / OBSERVED DATA**

- 46,045 render directories on disk; `renders_index.jsonl` admits **45,955**; `quarantine_n04_render_views.jsonl` holds **143** records. These three numbers do not reconcile by simple arithmetic — retries are involved. **Reconciling them is in scope.**
- 11 views per asset, `view_00.png` … `view_10.png`, 224 px.
- Default projection **orthographic**, default camera placement **Fibonacci lattice**. Both are recorded as UNKNOWN in the registry (U-03a, U-03) — the paper names neither.
- Pre-normalisation mesh extents are recorded per asset (`test_pre_normalisation_extents_are_recorded`). Registry finding **F13**: unit-sphere normalisation destroys absolute scale, so the annotator's size estimate is a category prior and the true extents exist so the estimate stays auditable. **This is the same mechanism that made the n05 defect total rather than partial** (`workflow/MIF_n05_diagnosis.md`) — relevant context, not this task's subject.

**Registry UNKNOWN U-02, verbatim, because it sets this task's boundary**

> The ULIP-2 checkpoint was trained on point clouds sampled by ULIP's own pipeline, so sampling differently shifts every embedding with no error raised. Measured by L2-PC-ULIP-REF and reported. It does not gate: the paper never says MetaFind reuses ULIP's clouds, and Stage 1 trains the point encoder anyway.

> **[CORRECTED BY MASTER 2026-08-21 — this paragraph was factually wrong.]**
> It previously read: *"`L2-PC-ULIP-REF` **has never been run and the reference clouds are not on disk** (registry note on `G2_pc_sanity`). Whether obtaining them is feasible is a question this task may answer; running the comparison is not in scope."*
>
> **Both halves are false.** A version of `L2-PC-ULIP-REF` **has** been run — `docs/graph/00_FINDINGS.md` **F21** reports it over 6 assets (same-asset Chamfer median `0.00318` vs cross-asset baseline `0.05880`) and it is the finding that fixed the `fallback_grey` colour bug. And the official ULIP-2 `000-009` reference shard **was extracted on 2026-08-21** from the local HF cache (4,999 clouds), so reference data is obtainable on this machine.

**What this task must therefore treat as an open contradiction, not as a settled fact:**

A second measurement taken 2026-08-21 over **286** overlapping assets reports median Chamfer **0.0903 at 0°** versus **0.0230 after a 180° yaw**, with **269/286 (94.1%)** improving under the rotation. **F21 (n=6, "geometry matches") and this (n=286, "94% need 180°") cannot both be right.** Untested explanations include: the official `.npy` clouds are **uncentered** (centroid up to 0.588) while ours are **pre-centered**, so omitting `pc_norm` on the official side shifts the comparison; F21's n=6 being too small; or a genuine frame divergence between `pointclouds.py:118` and `renders.py:150`.

**Scope ruling.** Reporting and characterising this contradiction **is in scope** — it bears directly on scope items 1–4 and on G2's criteria. **Resolving it is not**: Master tracks it as blocker **R1** (`MASTER.md` §11) and it is being verified separately through the frozen ULIP-2 checkpoint. If this audit produces evidence either way, that is a `MASTER-IMPACTING FINDING`.

---

## 7. Scope

### In Scope

**n03 — determine and classify each of:**

1. Which `pc_norm` / `pc_normalize` variant ULIP-2's Objaverse path actually applies, and whether `pointclouds.py` matches it operation-for-operation.
2. The sampling method ULIP-2 uses (farthest-point? uniform-area? something else) versus our area-weighted sampler.
3. The point count and channel layout ULIP-2 expects, versus our `(10000, 6)`.
4. RGB scale and colour-source handling versus upstream.
5. Whether `test_pc_norm_matches_ulips_definition` asserts the right proposition — and if it does not, say exactly what it does assert.
6. Whether the float32/float64 sidecar behaviour at `pointclouds.py:380-389` is correctly described by its own comment, verified on more than the one asset already checked.

**n04 — determine and classify each of:**

7. What §2.3's "11 orthogonal viewpoints" can and cannot constrain, given that 11 mutually orthogonal directions do not exist in 3D.
8. Whether ULIP-2 renders its own views, and if so with what layout, resolution and projection.
9. Whether `test_primary_layout_is_the_ulip2_style_orbit` asserts what its name claims.
10. The status of U-03 (camera placement) and U-03a (projection) after reading upstream — resolved, still open, or newly contradicted.
11. Reconciliation of `46,045 dirs / 45,955 admitted / 143 quarantined`, with the disposition of every non-admitted asset stated.

**Both:**

12. For every item above, state the evidence class and the concrete file/line or paper location it rests on.
13. State explicitly which items remain `UNKNOWN` after the audit, and what would resolve each.

### Explicit Non-Scope

- **No code changes.** Not to `pointclouds.py`, not to `renders.py`, not to any test — including tests this audit finds to be asserting the wrong thing. Report them; do not fix them.
- **No data changes.** No re-sampling, no re-rendering, no re-writing of sidecars or indexes.
- **Do not run `L2-PC-ULIP-REF`.** The reference clouds are not on disk; obtaining them is a separate decision.
- **Do not implement G1 or G2.** That is `D16`.
- **Do not resolve U-02, U-03 or U-03a as project decisions.** Report what the evidence supports; the resolution is the user's.
- **Do not re-open the n05 annotation defect.** It is `D14`'s.
- **No re-render proposal.** `MIF_n05_diagnosis.md` Evidence 2 already measured that framing does not drive annotation agreement (correlation `+0.054`). If this audit surfaces a *different* reason to re-render, that is a `MASTER-IMPACTING FINDING`, not an action.

If a blocker makes completion impossible, report to Master instead of silently expanding scope.

---

## 8. Expected Deliverables

1. `workflow/tasks/D15_n03-n04-code-audit/FINDINGS.md` — one entry per scope item 1–13, each carrying:
   - the claim
   - the evidence, with file:line or paper section
   - the evidence class
   - implementation impact
   - whether it changes G2's criteria (flag for `D16`)
2. `workflow/tasks/D15_n03-n04-code-audit/HANDOFF.md` per `workflow/tasks/HANDOFF_TEMPLATE.md`.

No other file may be created or modified.

---

## 9. Likely Files / Areas

- `/home/kyzen/upstream/ULIP/data/dataset_3d.py` and the ULIP-2 dataset/pretrain path
- `metafind/data/pointclouds.py`, `metafind/data/renders.py`
- `tests/test_pointclouds.py`, `tests/test_renders.py`
- `docs/graph/node_registry.yaml` (n03, n04)
- `docs/paper/metafind_source/2methdology.tex`
- `data/outputs/logs/{pointclouds_index,renders_index,quarantine_n04_render_views}.jsonl`

This list is guidance, not permission to expand scope.

---

## 10. Execution Requirements

1. Read the upstream source before making any claim about upstream behaviour. A test name is not evidence of what upstream does.
2. Distinguish "ULIP does X" from "ULIP-2's Objaverse path does X" from "MetaFind adopts X". `dataset_3d.py` holds several variants; naming the wrong one is the most likely failure of this task.
3. Where the paper is silent, say so and label the current behaviour `IMPLEMENTATION CHOICE` — never convert silence into endorsement (memory: `paper-silence-is-not-method-validation`).
4. Verify sidecar claims on a real sample, not on the single asset already checked at `pointclouds.py:384-386`.
5. Make no change of any kind. This includes "obvious" fixes.
6. Record any Master-impacting discovery immediately.
7. Stop if a required authority decision is missing.

---

## 11. Master-Impacting Finding Rule

Report as `MASTER-IMPACTING FINDING`, and do not act on, anything that would change:

- G2's criteria or thresholds (→ `D16`)
- whether the existing 46,052 clouds or 45,955 render sets must be regenerated (→ `D3` feasibility, and a very large GPU/IO cost)
- the standing of U-02, U-03, U-03a
- a test that asserts a false proposition about upstream

Include: finding, evidence, affected tasks, and whether this task can safely continue.

Do not make a new project-wide decision locally.

---

## 12. Verification Requirements

### Required Checks

- Every upstream claim cites `/home/kyzen/upstream/<repo>/<path>:<line>` at the recorded commit.
- Every paper claim cites section or equation.
- Every implementation claim cites `metafind/...:<line>`.
- Every data claim states the population it was measured over.

### Required Tests

- **None are to be added or modified.** Existing tests may be read and may be *run* to observe behaviour, never edited.

### Runtime / Artifact Checks

- The float64/float32 sidecar finding re-measured over **at least 200 assets**, reporting how many would cross a 1e-5 centroid threshold under each dtype.
- The `46,045 / 45,955 / 143` reconciliation computed from the actual JSONL files, not asserted.

### Research Fidelity Check

Required. For each of scope items 1–4 and 7–10, state whether correspondence to the authoritative source was **established**, **refuted**, or **not determinable**, and on what evidence. Do not claim fidelity because a test passes.

---

## 13. Definition of Done

- [ ] Objective achieved — items 1–13 each answered or explicitly marked UNKNOWN with what would resolve it.
- [ ] Scope respected — `git status` shows only the two new files in this task directory.
- [ ] `FINDINGS.md` produced with evidence classes and citations.
- [ ] No code, test, or data modified.
- [ ] Runtime/artifact verification completed (the two checks in §12).
- [ ] Research fidelity independently checked.
- [ ] Items affecting G2 explicitly flagged for `D16`.
- [ ] Codex review completed.
- [ ] Material Codex findings independently verified by Claude.
- [ ] `HANDOFF.md` written.

The task owner does not mark the project stage DONE. Master reviews; the **user** accepts.

---

## 14. Codex Review Requirement

Scope Codex to this task. Provide: this `TASK.md`, `FINDINGS.md`, and the upstream files cited.

Ask Codex specifically to attack the upstream-correspondence claims — whether the cited `pc_norm` variant is the one ULIP-2 actually uses on Objaverse, and whether the camera-layout claim survives reading the upstream renderer.

If Codex is unavailable, that fact must reach the user review brief. It may not be silently omitted.

---

## 15A. Required Completion Reporting — Finding vs Decision

This task produces **FINDINGS ONLY**. It may recommend; it may not decide.

It may not: change code, change data, change a test, resolve an UNKNOWN, alter G2's criteria, or mark any project item resolved.

---

## 17. Master Recommendation and the User Review Gate

Execution `COMPLETE` is not acceptance. Master reviews and issues a **MASTER RECOMMENDATION**; only the user's `APPROVE` makes any finding project state. See `workflow/WORKFLOW.md` §13A/§13B.

---

## Contract provenance

Written by Master 2026-08-21 at the user's instruction ("04 程式碼有驗證過嗎? … 06 也是 將這幾個驗證指派任務"), against artifact items **04** and **06**.
Baseline commit `468bbac`. Upstream commits recorded in §2.
