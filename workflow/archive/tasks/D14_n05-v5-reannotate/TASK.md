# D-Task Execution Contract

## Task ID

`D14_n05-v5-reannotate`

---

## Status

`ACTIVE` — **APPROVED by the user 2026-08-21.** Design ratified the same day (`n05_v5_design.md`).

Set to `ACTIVE` by the D14 executor 2026-08-21T17:0x+0800 on the user's explicit authorisation. That is the only edit to this file the executor may make.

---

## 1. Objective

**Implement n05 v5 — category-anchored annotation — validate it on a sample, and then re-annotate the corpus.**

One correctness boundary: after this task, every annotation record is anchored on the Objaverse-LVIS ground-truth identity rather than on a free-form guess, and the disagreement between the two is **measured** rather than assumed.

**This task contains a hard HOLD GATE.** The full run costs ~19.6 GPU-hours and **must not be run twice**.

---

## 2. Why This Task Exists

Kyzen found that n05's categories disagree with the dataset's own ground truth. Master diagnosed it. Full diagnosis: `workflow/MIF_n05_diagnosis.md`. Approved design: `workflow/n05_v5_design.md`.

**The root cause, measured by Master:**

Qwen is asked to *identify* an object from 224×224 renders it often cannot read, so it falls back on high-frequency priors.

| | distinct categories | top-20 share |
|---|---|---|
| LVIS ground truth | 1,156 | **7.1%** |
| Qwen output | 3,036 | **22.3%** |

`toy` is Qwen's single most common answer at 1,542 (3.4%) — a word the current prompt **explicitly forbids**.

**And it is not a `category`-only defect.** `build_prompt` says *"Estimate its size from what kind of object it is, not from the picture"* — so dimensions are **by design** derived from the category. Placement likewise. Descriptions were observed describing the hallucinated object:

```
LVIS pinecone -> "A dark brown hairbrush with a circular handle and bristles"
LVIS mug      -> "A cylindrical pillow with a striped pattern"
LVIS truck    -> "a modern air conditioner unit with multiple vents"
```

**A wrong category is a wrong record, not a wrong field.**

---

## 3. Required Shared Context

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/code-changes.md`
3. `/home/kyzen/MetaFindV1/.claude/rules/experiments.md`
4. `/home/kyzen/MetaFindV1/workflow/WORKFLOW.md` §13A, §13B
5. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
6. **`/home/kyzen/MetaFindV1/workflow/n05_v5_design.md`** — the approved design
7. `/home/kyzen/MetaFindV1/workflow/MIF_n05_diagnosis.md` — the evidence
8. this `TASK.md`

Then only the files named in §5 and §9.

---

## 4. Dependencies

### Required Before Start

- Design approved by Kyzen 2026-08-21 (`n05_v5_design.md`). Satisfied.
- `D1_n06-reencode` **STOPPED** at `2026-08-21T14:15:48`, 20,053 npz. Must stay stopped.

### Blocks

`D1_n06-reencode` (must re-run afterwards) → `D2` n09 → `D3` → everything downstream.

### Parallel Safety

`PARALLEL SAFE: NO`. This task holds the GPU for ~19.6 h and mutates the annotation corpus.

---

## 5. Authoritative Inputs

| # | Source | Why |
|---|---|---|
| 1 | `workflow/n05_v5_design.md` | **The design. Decisions 1–4 are settled; do not re-litigate them** |
| 2 | `data/datasets/objaverse-lvis/objaverse_lvis_metadata.json` → `value_to_key_mapping` | The anchor. 46,207 uid → category, **100% coverage** of our 45,952 |
| 3 | `data/outputs/logs/pointclouds_index.jsonl` → `raw_bbox_extents` | Exact mesh proportions, 46,041/46,052. **Y-up verified by Master**: tall objects (n=651) mean `[x .515, y .960, z .402]`; flat (n=497) `[x .865, y .318, z .738]`. **`height` = y** |
| 4 | `metafind/data/annotate.py` | `build_prompt` (366), `validate_annotation` (510), `PROMPT_VERSION`/`VALIDATOR_VERSION`/`SCHEMA_VERSION`, `annotation_contract_id()` |
| 5 | `metafind/data/annotate_run.py` | `build_work_list` (270), `is_complete` (120), `--force` (414), the AC-1 registry machinery |
| 6 | `tools/declare_annotation_provenance.py` | The registry writer. The registry **must** be updated after re-annotation |
| 7 | `workflow/DECISION_LEDGER.md` `DL-003` | AC-1 and the legacy-v3 provenance contract this task changes |

---

## 6. Current Relevant State

Master-verified 2026-08-21, read-only.

```
annotations                45,955   = 45,952 v3 + 3 legacy-v1 residuals
LVIS label coverage        45,952 / 45,952 = 100.00%
LVIS vague catch-alls      2.0%   (figurine 125, sculpture 115, motor vehicle 104, toy 80)
raw_bbox_extents           46,041 / 46,052 = 100%
embeddings                 20,053 npz   (D1 halted; invalid once annotations change)
checkpoints                0
```

**Agreement between LVIS and current Qwen output:**

```
category matches                          29.0%
LVIS word appears in description          28.4%
either                                    32.2%
neither                                   67.8%
```

**67.8% overstates the true error.** Part is vocabulary, not error: `headset→headphones`, `chair→stool`, `pastry→donut`, `Bible→book`, `softball→baseball` — several where Qwen is *more* specific, which the prompt asked for. **The prior-collapse evidence is what independently establishes a large genuine error component.**

**Resolution is NOT the cause.** correlation(best-view occupancy, agreement) = **+0.054**, agreement flat at ~28-30% across a ~100× range of effective object pixels. **Do not propose re-rendering.**

**AC-1 does not block this.** `build_work_list(force=True)` (`annotate_run.py:278-282`) is the AC-1.b explicit path, and `--force` is the sanctioned named-migration form. The gate removes the accident, not this.

---

## 7. Scope

### Phase 1 — implement v5

- `build_prompt` takes the LVIS category and the mesh proportions.
- New output field **`identity_confirmed`** (boolean).
- `category` rule: the LVIS label **or a strictly more specific term for the same object**. Lateral replacement is invalid.
- Dimensions: the model supplies **absolute scale for `height` only**; width/length follow from the exact proportions. Horizontal axis assignment (x vs z → width vs length) remains the model's call from the images.
- Version bumps: `PROMPT_VERSION` → 5, `SCHEMA_VERSION` → 3, `VALIDATOR_VERSION` → 3 as the validator changes. The contract fingerprint must move.
- **`synset` becomes a lookup** over the 1,156-term vocabulary (Design Decision 4). Build the table, review it, apply deterministically. **If building it turns out to need its own research decision, escalate — do not guess 1,156 entries.**

### Phase 2 — sample validation, then **HARD STOP**

Run v5 on a **stratified sample of 300–500 assets** covering: vague LVIS labels, specific LVIS labels, low frame occupancy, high frame occupancy.

Report, on that sample:

- `identity_confirmed` false rate — **the number that tells us whether LVIS is a trustworthy anchor**
- category agreement with LVIS (exact / refined-downward / lateral — count them separately)
- category distribution concentration vs the 22.3% baseline
- dimension plausibility against the exact proportions
- side-by-side of old vs new record for ~30 assets, including the known failures (`pinecone`, `mug`, `chocolate cake`, `truck`, `saddle`)

**Then STOP and report to Master. Do not start Phase 3.**

The full run proceeds **only** on the user's explicit go. This gate exists because 19.6 GPU-hours must not be spent on an unvalidated prompt.

### Phase 3 — full re-annotation (only after the user's go)

- Back up the whole v3 corpus **before** any mutation.
- `--force` over the full corpus.
- Update the provenance registry so AC-1's declarations remain true.

### Explicit Non-Scope

- ❌ **Do not re-render.** Evidence: correlation +0.054.
- ✅ **The annotation model IS changed** — `Qwen/Qwen3.8-27B`, local weights at `/mnt/data1/kyzen/models/Qwen3.8-27B`. **User decision U-6, 2026-08-21.**

  **P-1 GRANTED by Master 2026-08-21.** This line previously read *"Do not change the annotation model. GPT-4o is unavailable; deviation `D-2` stands."* **Both halves are withdrawn.**

  **The prohibition** is superseded by U-6.

  **The premise is worse than superseded — it was never established.** Master wrote "GPT-4o is unavailable" in `n05_v5_design.md` and in this contract **without verifying it**, inferring it from a code comment (`annotate_run.py:71`, "D-2: stands in for GPT-4o"). D14's finding F-2 is correct: OpenAI's official deprecation page does not list base `gpt-4o`, and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**; secondary sources say otherwise. **That conflict is UNRESOLVED, not resolved**, and no one has exercised the API. It must not be restated as settled anywhere.

  **The model change remains a DEVIATION.** The paper says GPT-4o (`2methdology.tex:28`, `neurips_2025.tex:100`). Swapping in Qwen3.8-27B is a departure — a user-decided one, recorded under the split `D-2` (see P-2). **R-E still binds.** Reaching GPT-4o would narrow `D-2`, never discharge it.
- ❌ **Do not quarantine or drop records on `identity_confirmed == false`.** Design Decision 2: **flag only** on this run. We have no measurement of LVIS's own error rate, and inventing a threshold before measuring it is forbidden.
- ❌ **Do not touch the 3 legacy-v1 residuals.** `D0-003` stays UNRESOLVED.
- ❌ **Do not run n06, n09, training, or gallery indexing.**
- ❌ **Do not delete the 20,053 halted embeddings.** They self-invalidate; deletion is not authorised.
- ❌ **Do not re-litigate Design Decisions 1–4.** Kyzen approved them. New *evidence* against one is a `MASTER-IMPACTING FINDING`; preference is not.

---

## 8. Master's Standing Rulings

**R-A — the HOLD GATE is absolute.** Phase 3 does not begin without the user's explicit go, whatever Phase 2's numbers look like. Good numbers are not permission.

**R-B — flag, do not filter.** `identity_confirmed == false` is recorded and counted. Nothing is dropped, quarantined, or repaired on it in this run. **The point of the field is to measure LVIS, and a filtered corpus cannot measure anything.**

**R-C — back up before mutating.** The v3 corpus is the only record of what the un-anchored pipeline produced. It is evidence. Back it up whole, verify the backup, then mutate.

**R-D — the registry must stay true.** After Phase 3 the corpus is no longer `accepted_legacy_v3`. `DL-003`'s AC-1 declarations must be updated in the same breath, or the provenance registry starts lying.

**R-E — this is a DEVIATION and must be recorded as one.** The paper has the VLM generate annotations with GPT-4o. Supplying the dataset label is a departure. It is the right call and it is **not** paper-faithful. Do not let any comment, field, or report blur that.

---

## 9. Allowed Writes / Protected Files

### 9.1 ALLOWED

| Path | For |
|---|---|
| `metafind/data/annotate.py` | prompt, validator, versions, contract, synset table |
| `metafind/data/annotate_run.py` | passing the LVIS category and proportions through |
| `tests/test_annotate.py` | coverage for every v5 rule |
| `data/outputs/annotations/**` | **Phase 3 only, after the user's go, after the backup is verified** |
| a v3 backup directory under `data/outputs/` | Required by R-C. Name it in the HANDOFF |
| `data/outputs/annotation_provenance.json` via `tools/declare_annotation_provenance.py` | R-D |
| `data/outputs/logs/` — n05 run log, `quarantine_n05_annotate.jsonl`, `run_progress.jsonl`, `cost_ledger.jsonl` | **The last two are shared append-only ledgers. Append only — never truncate, rewrite, or clean** |
| a synset lookup table under `data/outputs/` or `metafind/data/` | Design Decision 4 |
| `workflow/tasks/D14_n05-v5-reannotate/HANDOFF.md`, `CODEX_REVIEW.md`, `TASK.md` status line | required |
| `docs/graph/README.md` **line 270 only** | **P-3 GRANTED, standing for this task's duration.** Update the two stale integers whenever adding tests moves them — currently **435 → 456** (test functions) and **547 → 582** (parametrised cases), both re-verified by Master. `check_graph.py:415` asserts the README figure against `tests/`, so DoD #13 (add tests) and DoD #14 (gates pass) are otherwise in direct contradiction. **Documentation only. Two integers. This lifts protection on nothing else under `docs/**`** |

### 9.2 PROTECTED

| Path | Why |
|---|---|
| `data/outputs/annotations/{6c7db00c…,8a0192ee…,a397b648…}.json` | The 3 legacy-v1 residuals. **`D0-003` UNRESOLVED. Do not touch, do not migrate** |
| `data/outputs/embeddings/**` | 20,053 halted npz. Self-invalidating; **deletion not authorised** |
| `data/outputs/checkpoints/**` | Must stay empty |
| `data/outputs/renders/**`, `pointclouds/**` | **No re-render.** Read-only |
| `data/outputs/stage1_*.json`, `variant_registry.json` | Ratified by `DL-003` |
| `metafind/models/**`, `metafind/train/**`, `metafind/data/encode_text_image.py` | Out of scope |
| `tools/preflight_stage1_text.py` | D10's gate |
| `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md`, `decisions/**` | Master's |
| `docs/**` | Untouched |

Anything not in 9.1 is protected by default.

---

## 10. Execution Requirements

1. Read `n05_v5_design.md` before writing any code. Decisions 1–4 are settled.
2. Verify LVIS coverage **per uid** before relying on it.
3. Verify the Y-up axis mapping independently — Master's evidence is in §5, reproduce it.
4. **Phase 2 stops. It does not roll into Phase 3.**
5. Back up the v3 corpus and **verify the backup** before any mutation.
6. Never touch the 3 legacy-v1 residuals.
7. Report any Master-impacting discovery immediately.

---

## 11. Master-Impacting Finding Rule

Report `MASTER-IMPACTING FINDING` for anything affecting architecture, accepted interpretation, cross-task dependency, milestone feasibility, or a global runtime assumption.

**Escalate rather than decide:** an `identity_confirmed` false rate high enough to question LVIS as an anchor · evidence that a Design Decision is wrong · any need to touch the 3 residuals · a synset table that cannot be built without a research decision.

---

## 12. Verification Requirements

`PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python`

### Before anything

```bash
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # baseline
ls data/outputs/embeddings/*.npz | wc -l          # 20053, must not change
ls data/outputs/checkpoints | wc -l               # 0
git rev-parse HEAD; git status --porcelain
```

### Phase 2 — the numbers that decide whether Phase 3 runs

```
sample size, and how it was stratified
identity_confirmed == false          n and %          <-- the headline number
category vs LVIS:  exact / refined-downward / lateral   counted separately
category top-20 concentration        vs the 22.3% baseline
`toy` / `bookshelf` / `pillow` frequency   vs 3.4% / 2.4% / 2.1%
dimension check against exact proportions
30 old-vs-new side-by-side, including pinecone / mug / chocolate cake / truck / saddle
```

**Then STOP.**

### Phase 3 — after the run

```bash
ls data/outputs/annotations | wc -l                # 45,955
$PY -c "..."                                       # 45,952 at PROMPT_VERSION 5, 3 still v1
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # differs; backup matches baseline
ls data/outputs/embeddings/*.npz | wc -l           # still 20053
ls data/outputs/checkpoints | wc -l                # 0
$PY -m pytest tests/ -q
$PY tools/check_graph.py
```

Plus: the 3 residuals byte-identical to baseline · the provenance registry updated and self-consistent · full-corpus `identity_confirmed` distribution · full-corpus category concentration vs 22.3%.

---

## 13. Definition of Done

- [ ] **1.** v5 implemented: LVIS anchor, exact proportions, `identity_confirmed`, refinement-not-replacement rule, version bumps, contract fingerprint moved.
- [ ] **2.** Y-up axis mapping independently reproduced.
- [ ] **3.** `synset` lookup built over the 1,156 vocabulary and reviewed — or escalated with reasons if it needs its own decision.
- [ ] **4.** Phase 2 run on a stratified 300–500 sample; every §12 metric reported.
- [ ] **5.** **Task STOPPED after Phase 2 and reported.** Phase 3 began only on the user's explicit go.
- [ ] **6.** v3 corpus backed up **and the backup verified** before any mutation.
- [ ] **7.** Full run complete: 45,952 at `PROMPT_VERSION 5`.
- [ ] **8.** The 3 legacy-v1 residuals **byte-identical**. Nothing claims `D0-003` is resolved.
- [ ] **9.** Provenance registry updated; AC-1's declarations still true; `--force` still works; a bare run still queues 0.
- [ ] **10.** Embeddings still 20,053, none deleted. `checkpoints/` empty. n06/n09/training never invoked.
- [ ] **11.** `identity_confirmed` distribution reported for the full corpus.
- [ ] **12.** Category concentration reported vs the 22.3% baseline.
- [ ] **13.** `pytest tests/ -q` and `check_graph.py` pass. New tests cover every v5 rule.
- [ ] **14.** `git diff` touches only §9.1 paths.
- [ ] **15.** Experiment provenance recorded per `.claude/rules/experiments.md`.
- [ ] **16.** The DEVIATION (R-E) recorded explicitly.
- [ ] **17.** Codex review completed; material findings verified by Claude.
- [ ] **18.** `HANDOFF.md` + `CODEX_REVIEW.md` written, including `USER REVIEW INPUT`.

---

## 14. Codex Review Requirement

Ask Codex to attack:

- whether anchoring merely **replaces** Qwen's errors with LVIS's, and whether `identity_confirmed` actually detects that or just rubber-stamps the anchor;
- whether the model can still laterally replace the category despite the rule, and whether the validator enforces it or only asks nicely;
- whether the exact-proportions path is axis-correct, and what a wrong axis mapping would silently produce;
- whether `identity_confirmed` is answerable at all from 224×224 renders, or is a field the model will always answer `true`;
- whether the sample stratification could make Phase 2 look better than the corpus;
- whether the synset lookup introduces a new error class;
- whether the backup is verified or merely made;
- whether AC-1 still holds after the registry update — bare run queues 0, `--force` still works;
- whether any part of the output implies `D0-003` is resolved.

Codex must not be asked merely to confirm.

---

## 15. Claude Verification of Codex Findings

Classify each: `CONFIRMED` · `PLAUSIBLE` · `REJECTED` · `UNVERIFIED`. Unavailable review is `CODEX REVIEW UNAVAILABLE` — **not a PASS** — and must reach the HANDOFF.

---

## 16. Required Handoff

`HANDOFF.md` + `CODEX_REVIEW.md`, per `HANDOFF_TEMPLATE.md`, including §15 `USER REVIEW INPUT`. Findings and decisions reported **separately** (`WORKFLOW.md` §13A).

State explicitly: **no re-render, no n06, no n09, no training, the 3 residuals untouched, `D0-003` still UNRESOLVED**, and that the LVIS anchoring is a recorded **DEVIATION**.

Then stop. Do not start `D1_n06-reencode`.

---

## 17. Master Recommendation and the User Review Gate

Master reviews and returns a **MASTER RECOMMENDATION**, not an acceptance. Only the user's `APPROVE` makes it FINAL ACCEPTED.

`D1_n06-reencode` must re-run after this task and does **not** restart automatically.
