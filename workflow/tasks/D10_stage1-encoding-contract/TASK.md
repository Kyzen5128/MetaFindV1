# D-Task Execution Contract

> This file is the authoritative execution contract for one bounded work package.
> Stay within scope, satisfy the Definition of Done, perform verification, obtain Codex review, and return a HANDOFF to Master.

---

## Task ID

`D10_stage1-encoding-contract`

---

## Status

`READY`

Approved by user 2026-08-21. Contract finalized by Master 2026-08-21.
Becomes `ACTIVE` when the user approves starting the task conversation.

---

## 1. Objective

**Establish a safe, verified Stage 1 encoding contract before n06 runs.**

One objective, one correctness boundary: after this task, it must be impossible for `n06` to produce a gallery built from two different text distributions, and the serializer must emit the template ratified by `D0-008`.

This task does **not** encode anything. It makes encoding safe and proves that it is.

---

## 2. Why This Task Exists

`D0-008` was accepted with follow-up on 2026-08-21. Acceptance ratified the serialization design but deliberately did **not** unblock `D1_n06-reencode`.

The reason is follow-up **F-1**. `is_complete()` (`metafind/data/encode_text_image.py:73-83`) compares nothing about the text — only sidecar existence, `encoder_version`, and NPZ existence. A plain `python -m metafind.data.encode_text_image` would skip all **5,276** metre-derived embeddings as "complete" and encode only the remainder under the new centimetre template.

The result: a gallery whose text embeddings come from two distributions, with no error, no warning, and an identical `text_serialization` label on both halves. `metafind/train/gallery_index.py` fingerprints the checkpoint, not the text, so Table 1 would be **self-consistent and wrong**.

This is the only known defect in the project capable of producing confident wrong numbers with no error anywhere in the chain. It was surfaced by Codex adversarial review during D0-008, confirmed by Master by direct code reading, and classified BLOCKER by Kyzen.

`D1_n06-reencode` is ~4 GPU-hours. Everything in this task is minutes of work. Doing it first is not optional.

**Milestone contribution:** Stage 1. **Downstream:** D1 → D2 → D3 → D4 → D7 all inherit this contract.

---

## 3. Required Shared Context

Read, in order:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/code-changes.md`
3. `/home/kyzen/MetaFindV1/.claude/rules/research-rigor.md`
4. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
5. this `TASK.md`

Then read only the files listed under §5 and §9.

Do not re-read the entire repository. Do not read other task folders.

---

## 4. Dependencies

### Required Before Start

- `D0-008_stage1-text-template` — **ACCEPTED WITH FOLLOW-UP, 2026-08-21.** Satisfied.
  All serialization policy is user-decided. **No research decision remains open in this task's scope.**

### Blocks

- `D1_n06-reencode` — cannot start until this task is accepted by Master.
- `D2_stage1-prereq`, `D3_stage1-train`, `D4_gallery-index`, `D7_eval-table1` — transitively.

### Parallel Safety

`PARALLEL SAFE: NO`

This task modifies `metafind/models/resolve_stage1.py`, `metafind/data/encode_text_image.py`, `tests/test_resolve_stage1.py`, and mutates one annotation record. Nothing else may touch those while it runs.

---

## 5. Authoritative Inputs

Ordered by authority.

| # | Source | Why it matters |
|---|---|---|
| 1 | `workflow/decisions/D0-008_stage1-text-template.md` | **The contract for this task.** §11.1 (user decisions E-1, E-2, E-3, S-1, S-2), §11.2 (blocker exit criteria B-1…B-4), §11.3 (ratified template, golden string, R-1…R-4), §11.4 (measured effect), §12 (Master's resolution and the R-3 ruling) |
| 2 | `docs/paper/metafind_source/2methdology.tex` §2.3 | Only if a documentation correction needs the paper's wording. The paper does **not** constrain serialization order — do not reopen this |
| 3 | `metafind/models/resolve_stage1.py` | `TEXT_TEMPLATE` (96-100), the `CONFIRM` marker (102), omission rationale (102-115), `PLACEMENT_PHRASES` (116-128), `MAX_PLACEMENT` (162), `serialize_annotation()` (278+), `build_protocol()` (~377), `main()` (416+) |
| 4 | `metafind/data/encode_text_image.py` | `is_complete()` (73-83), `load_protocol()` (86-108), work-list construction (177-179), serializer call (194), sidecar stamping (233) |
| 5 | `tests/test_resolve_stage1.py` | `GOLDEN_ANNOTATION`, `L1-TEXT-SERIALIZATION`, the negative-injection test at 60-69 |
| 6 | `data/outputs/annotations/` | 45,955 records: 45,952 `prompt_version:3` + 3 `prompt_version:1` |
| 7 | `data/outputs/embeddings/` | 5,276 stale sidecars + NPZ. **Read-only. Do not delete.** |

---

## 6. Current Relevant State

Verified by Master 2026-08-21 at commit `1837477`.

| | |
|---|---|
| annotation files | 45,955 = **45,952 v3** + 3 v1 |
| stale embeddings | **5,276** `.npz` + 5,276 sidecars, metre-template text |
| records rendering a `0` dimension under `:.0f` | **161** (163 non-integer values across 162 records: `{0.5:155, 0.2:7, 2.5:1}`) |
| records with ungrammatical `"A "` article | **3,643** |
| records over CLIP's 77-token limit | **1** — `3e91980a22da4c0da975cc8ef776972c`, true length **89** BPE tokens |
| test suite | `python -m pytest tests/ -q` → 442 passed, 0 skipped |
| structural checker | `tools/check_graph.py` → 2275 checks, all pass |
| `data/outputs/checkpoints/` | empty — Stage 1 has never trained |

**Ratified template (D0-008 §11.3):**

```
{description} {Category} made of {materials}, roughly {W} by {L} by {H} centimetres, {placement}.
```

`{Category}` = category with first character upper-cased.
`W`/`L`/`H` = one decimal with a trailing `.0` stripped, **applied uniformly at every magnitude**. No `< 1 cm` threshold branch.
No `"A "` article, and **no a/an heuristic** — explicit user constraint.

**Expected golden string** after the change, from the existing `GOLDEN_ANNOTATION`:

```
A wooden dining chair with a slatted back and four tapered legs. Dining chair made of wood, fabric, roughly 50 by 45 by 90 centimetres, typically placed on the floor.
```

---

## 7. Scope

### In Scope

**F-1 — the BLOCKER.** All four exit criteria from D0-008 §11.2:

| | Requirement |
|---|---|
| B-1 | Every one of the 5,276 stale sidecars is treated as **incomplete** — via a bumped `ENCODER_VERSION`, a fresh embeddings namespace, or `is_complete()` binding to the serialized text. **Deleting them is not required and is not authorised.** |
| B-2 | `load_protocol()` **refuses to run** when the protocol's recorded template does not match the executed serializer — exact comparison or hash |
| B-3 | `"metafind_v1_natural"` is **retired** as a cache identity. It already labels two different transformations, so versioning alone is insufficient |
| B-4 | A pre-flight check confirms 0 zero-dimension renderings, 0 records over 77 true tokens, and output matching the ratified template — runs in seconds, before any GPU time |

**F-2 — implement the ratified template** in `resolve_stage1.py`: E-1 (dimension precision), E-2 (article removal), S-1 (uniform formatter, no threshold), S-2 (category capitalisation). **Nothing else.**

**F-3 — documentation and dead code:**
- R-1: correct the false field-order comment at `resolve_stage1.py:93-95`. The paper's order is category → *dimensions* → materials; the code emits category → *materials* → dimensions. Do not change the order — correct the comment.
- R-2: correct the "EVERY variable-length part is bounded" claim at `resolve_stage1.py:141-149`, and either enforce or delete the unused `MAX_PLACEMENT` (`:162`).
- R-3: **DELETE** the unreachable `PLACEMENT_PHRASES[("onWall","onCeiling")]` entry. Master ruled delete, not fix — see §8 below.

**F-4 — re-annotate `3e91980a22da4c0da975cc8ef776972c`** to English under the current v3 prompt, then re-verify its token count.

**F-5 — update `L1-TEXT-SERIALIZATION`** deliberately, plus the coverage Codex found missing: sub-centimetre dimensions, the absent-article form, the wall+ceiling combination, protocol/serializer mismatch, and cache invalidation.

### Explicit Non-Scope

- ❌ **Do not run n06.** No encoder invocation, no GPU embedding generation, no partial run, no `--limit` smoke run of the encoder.
- ❌ **Do not re-run n05b** (`resolve_stage1.main()`). It rewrites `stage1_hyperparameters.json` in the same call and would clobber the τ decision. That is D2's job.
- ❌ **Do not touch τ.** Correction C-001 belongs to D2.
- ❌ **Do not delete the 5,276 stale embeddings or sidecars.** Invalidate them; do not destroy them.
- ❌ **No serialization change beyond E-1, E-2, S-1, S-2.** Master's scope guard.
- ❌ Do not decide `D0-002` (tower sharing) or `D0-003` (the 3 v1 annotations).
- ❌ Do not re-open the field order, the unit, the omission of `synset`/`volume`/`mass`, or the placement phrasing. All ratified.
- ❌ Do not fix the latent defects deferred as F-9 (`_cap()` word-boundary, doubled period, uncapped material strings). Zero corpus impact; deliberately deferred.
- ❌ Do not touch `workflow/MASTER.md`, `CONTEXT.md`, or `INDEX.md`.

If a blocker makes completion impossible, report to Master. Do not silently expand scope.

---

## 8. Master's Standing Rulings

**R-3 — delete, do not fix.** The fallback already emits grammatical prose (`"typically mounted on a ceiling or on a wall"`) for all 90 affected records. "Fixing" the entry would change 90 serialized strings **that were not present in the configuration measured in D0-008 §11.4**, weakening the evidence the ratification rests on. Deleting changes 0 strings.

**Scope guard.** If you believe any string change beyond E-1/E-2/S-1/S-2 is warranted, report `MASTER-IMPACTING FINDING` and stop. Do not extend the template.

**F-4 dependency.** Re-annotation requires the annotation model to be available. If it is not, **report and stop** — do not skip the item, and do not hand-write the annotation.

---

## 9. Likely Files / Areas

- `metafind/data/encode_text_image.py` — `is_complete()`, `load_protocol()`, `ENCODER_VERSION`, sidecar stamping
- `metafind/models/resolve_stage1.py` — `TEXT_TEMPLATE`, `serialize_annotation()`, `PLACEMENT_PHRASES`, `MAX_PLACEMENT`, `TEXT_SERIALIZATION`, comments at 93-95 and 141-149, `build_protocol()`
- `tests/test_resolve_stage1.py` — golden string and new coverage
- `tests/test_encode_text_image.py` — cache-validity and protocol-binding coverage
- possibly a new pre-flight script under `tools/` for B-4
- `data/outputs/annotations/3e91980a22da4c0da975cc8ef776972c.json` — the single re-annotation

Guidance, not permission to expand.

---

## 10. Execution Requirements

1. Read `workflow/decisions/D0-008_stage1-text-template.md` §11 and §12 before editing anything. That file is this task's contract.
2. Make the smallest coherent change per item. No unrelated refactoring, renaming, or reformatting.
3. Preserve research semantics outside the four approved serialization changes.
4. Back up the single annotation record before mutating it (F-4).
5. Verify each assumption that affects scientific behaviour before relying on it.
6. Report any Master-impacting discovery immediately.
7. Stop if a required authority decision turns out to be missing.

---

## 11. Master-Impacting Finding Rule

Report `MASTER-IMPACTING FINDING` if execution discovers anything affecting project architecture, accepted research interpretation, a cross-task dependency, another task's contract, milestone feasibility, or a global runtime assumption.

Include: the finding, the evidence, affected tasks, and whether this task can safely continue.

Do not make a new project-wide decision locally.

---

## 12. Verification Requirements

### Required Checks

- **The cache-validity proof (see DoD item 2)** — the central verification of this task.
- Pre-flight over the full v3 corpus: 0 zero-dimension renderings, 0 records over 77 true tokens, output matches the ratified template.
- Token counting must use the **untruncated** BPE path (`SimpleTokenizer.encode` + 2 for SOT/EOT). The padded tokenizer saturates at 77 and would mask a 89-token record — this is Codex finding C-3, confirmed.
- Negative test: `load_protocol()` **fails** on a deliberately mismatched protocol.
- `git diff --stat` touches only files named in §9.

### Required Tests

- `python -m pytest tests/ -q` → all pass. **Use no `--ignore` flag**; the suite is 442 and `test_cuda_smoke.py` contributes 5 that genuinely run.
- `tools/check_graph.py` → all checks pass.
- Updated `L1-TEXT-SERIALIZATION` matches the golden string in §6 exactly.
- New coverage per F-5.

### Runtime / Artifact Checks

- The 5,276 stale `.npz` and sidecars still exist on disk, unmodified.
- `data/outputs/checkpoints/` still empty; no embeddings created.
- The re-annotated record is valid v3 and under 77 true tokens.

### Research Fidelity Check

The template must match D0-008 §11.3 **exactly**, including uniform formatter application (S-1) and category capitalisation (S-2). Do not claim fidelity because tests pass — compare the emitted string to the ratified specification directly.

---

## 13. Definition of Done

- [ ] **1.** All four blocker exit criteria B-1…B-4 demonstrably hold — shown with evidence, not asserted.
- [ ] **2.** **Without running the encoder or performing GPU embedding generation**, the completion / cache-validity logic demonstrates that a resumed n06 would classify **all 45,952 records as requiring encoding** under the ratified protocol, rather than skipping the existing ~5,276 stale embeddings.

      The verification must explicitly report:

      ```
      total records:                          45,952
      cache-valid under ratified protocol:          0
      requires encoding:                       45,952
      ```

      Proven through completion / cache-validity / pre-flight logic **only**. Do not launch actual n06 embedding generation.

      Additionally reconcile the corpus arithmetic in the HANDOFF so the figures are unambiguous downstream: n06's work list is `annotations ∩ renders_index` = **45,955**, of which the 3 `prompt_version:1` records fail `serialize_annotation()` with `KeyError: 'width'` and are quarantined without output. The 45,952 above is the valid-v3 population. Report both, and state which population each number describes.
- [ ] **3.** `load_protocol()` fails on a deliberately mismatched protocol — negative test present and passing.
- [ ] **4.** Pre-flight over the full v3 corpus reports 0 zero-dimension renderings, 0 records over 77 true tokens, and output matching the ratified template.
- [ ] **5.** `3e91980a22da4c0da975cc8ef776972c` is English, valid v3, and under 77 true tokens. Original backed up.
- [ ] **6.** Golden string updated deliberately and matches §6 exactly; F-5 coverage added.
- [ ] **7.** `pytest tests/ -q` passes with no ignore flag; `tools/check_graph.py` passes.
- [ ] **8.** `git diff` touches only files named in §9. Unexpected files investigated before completion.
- [ ] **9.** The 5,276 stale embeddings still exist on disk; no embeddings created; `checkpoints/` still empty.
- [ ] **10.** No serialization change beyond E-1, E-2, S-1, S-2.
- [ ] **11.** Codex review completed; material findings independently verified by Claude.
- [ ] **12.** `HANDOFF.md` written; `CODEX_REVIEW.md` written.

The task owner does not mark the Stage 1 milestone DONE. Master decides acceptance.

---

## 14. Codex Review Requirement

Scope the review to this task. Provide Codex with: this `TASK.md`, the diff, the verification output, D0-008 §11.2 and §11.3, and the known uncertainties.

Ask Codex to attack specifically:

- whether B-1 actually invalidates **every** stale sidecar, including any path where a sidecar could still be judged complete;
- whether B-2's binding can be bypassed — e.g. a caller passing a non-default `template` argument to `serialize_annotation()`;
- whether retiring the identifier (B-3) leaves any code path still reading or writing the old value;
- whether the pre-flight check (B-4) could pass while the real encode would diverge;
- whether the uniform formatter (S-1) is correct for fractional values **absent from this corpus** — Python's ties-to-even renders `0.25` as `0.2`;
- whether capitalisation (S-2) handles non-alphabetic or multi-byte leading characters correctly;
- whether the golden-string update silently changed anything beyond the four approved edits;
- whether the DoD item 2 verification genuinely proves the claim, or merely restates the implementation.

Codex must not be asked merely to confirm the work.

---

## 15. Claude Verification of Codex Findings

Classify each material finding: `CONFIRMED` · `PLAUSIBLE` · `REJECTED` · `UNVERIFIED`.

Only verified findings drive research-significant corrections.

If review is unavailable through quota, auth, timeout, or runtime failure, report `CODEX REVIEW UNAVAILABLE`. **That is not PASS.**

---

## 16. Required Handoff

Write `workflow/tasks/D10_stage1-encoding-contract/HANDOFF.md` and `CODEX_REVIEW.md`.

The HANDOFF must cover: Task ID · Status · Objective Result · Files Changed · Artifacts Produced · Evidence Used · Decisions Made Within Scope · Verification Performed · Verification Result · Codex Review Result · Confirmed Findings · Rejected/Unverified Findings · Master-Impacting Findings · Remaining Risks · Blocked Items · Recommended Master Update · Recommended Next Action.

State explicitly in the HANDOFF that **no encoder run and no GPU embedding generation occurred**.

Then stop. Do not start `D1_n06-reencode`.

---

## 17. Master Acceptance

Master reviews `TASK.md`, `HANDOFF.md`, `CODEX_REVIEW.md`, and the repository state, then returns one of `ACCEPT` · `ACCEPT WITH FOLLOW-UP` · `REWORK` · `REJECT` · `BLOCKED`.

Only Master updates `workflow/INDEX.md`.

On acceptance, `D1_n06-reencode` unblocks.
