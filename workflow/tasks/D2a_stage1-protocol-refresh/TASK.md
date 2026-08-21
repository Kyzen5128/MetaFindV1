# D-Task Execution Contract

> Authoritative execution contract for one bounded work package.
> Stay within scope, satisfy the Definition of Done, verify, obtain Codex review, return a HANDOFF to Master.

---

## Task ID

`D2a_stage1-protocol-refresh`

---

## Status

**Execution: `COMPLETE`. Integration: `USER_APPROVED` 2026-08-21. FINAL ACCEPTED.**

Written 2026-08-21; approved after three rounds of contract review (two `MODIFY` cycles) plus one mid-execution `USER DECISION` remediation. Executed, Codex-reviewed twice, Master-verified independently, and accepted by the user with **MIF-2 ratified**.

Ledger entry `DL-003`. Brief: `USER_REVIEW.md`.

---

## 1. Objective

**Make `load_protocol()` stop refusing — by re-emitting the Stage 1 protocol under τ = 0.5 and the ratified serializer — and make the existing annotation corpus safe from accidental re-annotation.**

One correctness boundary: after this task, `n06` is runnable on protocol grounds, and no ordinary command can overwrite annotation records that the project has decided to preserve or has not yet decided about.

This task does **not** run n06, n09, or any training.

---

## 2. Why This Task Exists

Three things converged.

**(a) `load_protocol()` refuses today.** D10 implemented B-2, so n06 hard-fails against the on-disk protocol. Verified by Master:

```
ValueError: stage1_encoding_protocol records text_serialization 'metafind_v1_natural',
but this process's serializer is 'metafind_v2_cm@8e4b1fcc66c7f48c'.
```

**(b) The recorded critical path was wrong.** `CONTEXT.md` §7 said `D1 → D2`; D0-008 item 6 says `n05b → n06`. The user resolved this on 2026-08-21 in favour of D0-008: **the new protocol artifact must precede n06.** `D2` is split so `D1` does not inherit `D0-002` and `D0-003`, which n06 never touches.

Verified code facts behind the split:

| # | Fact | Evidence |
|---|---|---|
| 1 | n06 and n09 are mutually independent | `splits.py` reads only the three index files, never an embedding; `encode_text_image.py` reads nothing n09 writes |
| 2 | n05b writes both artifacts in one call — C-001 and C-002 cannot be separated | `resolve_stage1.py:443-444` |
| 3 | τ = 0.5 has **no code path**: `DEFAULT_HYPERPARAMETERS` hardcodes `0.07`, and there are **0** CLI flags for it | `resolve_stage1.py:243`; `grep` over `main()` |
| 4 | `D0-002` binds in n09, not n05b | `splits.py:130,136` |
| 5 | `D0-003` binds in n09, not n05b | `splits.py` admits all 45,955; the crash is in `stage1.py`'s loader |

**(c) `AC-1` blocks D10's acceptance.** D10's contract-versioning change, combined with the user's decision to preserve the legacy-v3 corpus, left the code in active opposition to that decision. This task carries `AC-1`.

**Downstream — must match §4 exactly:**

```
D2a USER_APPROVED
   → D10 returns to final USER REVIEW with AC-1 evidence
      → D10 USER_APPROVED
         → D1_n06-reencode unblocks
```

---

## 3. Required Shared Context

Read, in order:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. `/home/kyzen/MetaFindV1/.claude/rules/code-changes.md`
3. `/home/kyzen/MetaFindV1/.claude/rules/research-rigor.md`
4. `/home/kyzen/MetaFindV1/workflow/WORKFLOW.md` §13A, §13B
5. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
6. this `TASK.md`

Then only the files named in §5 and §9. Do not re-read the repository.

---

## 4. Dependencies

### Required Before Start

- `D0-008_stage1-text-template` — **`USER_APPROVED` 2026-08-21** (ledger `DL-001`). Satisfied.
- `D10_stage1-encoding-contract` — implementation approved in principle by the user 2026-08-21; integration `AWAITING_USER_REVIEW / MODIFIED` (`DL-002`).

  **Do not proactively reopen D10's existing scope.** But `settled` is not a reason to ignore evidence: if this task or Codex finds **new contradictory evidence or a correctness defect** in D10's implementation, **escalate it** as a `MASTER-IMPACTING FINDING` with evidence. Do not suppress a real defect behind the label, and do not silently fix it either — report and let Master decide.

### Blocks

Acceptance of this task does **not** unblock `D1`. The chain is:

```
D2a USER_APPROVED
   → D10 returns to final USER REVIEW carrying this task's AC-1 evidence
      → D10 USER_APPROVED
         → D1_n06-reencode unblocks
```

- `D10`'s final acceptance — `AC-1` is demonstrated here and returns with D10.
- `D1_n06-reencode` — unblocks only on **D10's** `USER_APPROVED`, not on this task's.

**`D1` must not be started merely because `D2a` was accepted.**

### Parallel Safety

`PARALLEL SAFE: NO` — against any other **execution** task. This task mutates `resolve_stage1.py`, `annotate_run.py`, `annotate.py`, tests, and writes two protocol artifacts.

**One verified exception, approved by the user 2026-08-21:**

`D2a` may run concurrently with `D0-009_essgnn-fx-codomain`. Master verified all five conditions of `WORKFLOW.md` §7:

| | |
|---|---|
| 1. No dependency | Neither needs the other's result. D0-009 is a paper audit; D2a is a protocol refresh |
| 2. No filesystem conflict | D0-009 writes **only** `workflow/decisions/D0-009_essgnn-fx-codomain.md`. It is forbidden from touching code, tests, protocol, or graph contract. D2a touches none of `workflow/decisions/` |
| 3. No scientific conflict | D2a settles τ and the Stage 1 encoding protocol. D0-009 examines ESSGNN's `f_x` codomain — Stage 2 geometry. Disjoint |
| 4. Independently verifiable | D2a by its DoD; D0-009 by its verdict and Codex review |
| 5. Materially saves time | Yes — D0-009 is a long read-only audit |

**Scope wall.** `D2a` must not touch `metafind/models/essgnn.py`, the equivariance tests, `docs/audit/`, `docs/graph/`, or `workflow/decisions/**` — all already in §9.3. If D2a finds something bearing on D0-009's question, **report it as a `MASTER-IMPACTING FINDING`. Do not act on it.**

---

## 5. Authoritative Inputs

| # | Source | Why |
|---|---|---|
| 1 | `workflow/decisions/D0-008_stage1-text-template.md` §11.3, §14 | The ratified template and the binding user classifications. **τ = 0.5 is a PAPER FACT** (`3experiments.tex:15`) |
| 2 | `workflow/tasks/D10_stage1-encoding-contract/USER_REVIEW.md` §7.0 | **Provenance / historical source of `AC-1`.** It is **not** the execution contract. **`TASK.md` §7.1 is the authoritative execution contract for this task.** If the two ever disagree: report a `MASTER-IMPACTING FINDING` and stop. Do not pick one |
| 3 | `workflow/DECISION_LEDGER.md` `DL-001`, `DL-002` | Decision state |
| 4 | `metafind/models/resolve_stage1.py` | `DEFAULT_HYPERPARAMETERS` (~243), `build_hyperparameters()`, `main()` (~416-460), `_write()` (~443-444) |
| 5 | `metafind/data/annotate_run.py` | `is_complete()` (78-100), `main()` work-list construction (~248-253), `--force` (227) |
| 6 | `metafind/data/annotate.py` | `annotation_contract_id()`, `PROMPT_VERSION`, `VALIDATOR_VERSION`, `SCHEMA_VERSION` |
| 7 | `metafind/data/encode_text_image.py` | `load_protocol()` (86-108) — the consumer this task must satisfy |
| 8 | `tools/preflight_stage1_text.py` | The B-4 gate. Must still pass afterwards |

---

## 6. Current Relevant State

Verified by Master 2026-08-21, read-only.

### The three populations — never conflate them

| count | population | status |
|---|---|---|
| **45,952** | **accepted legacy-v3 corpus** | validated under `VALIDATOR_VERSION 2`; 0 language violations, 0 validator failures |
| **3** | **legacy-v1 residuals** — `6c7db00cc164467ebac356a5ca67368b`, `8a0192eee6fb4140bb3e9696b3dbae5a`, `a397b648d6eb48d7909d1ee11235e78f` | **`D0-003` UNRESOLVED.** `prompt_version: 1`, v1 schema. **Not legacy-v3. Not migrated** |
| **45,955** | total render / work population | 45,952 + 3 |

### The hazard, measured

```
rendered assets                                        45,955
bare `annotate_run` todo (no --force)                  45,955
  of those, accepted legacy-v3                         45,952
  of those, legacy-v1 residuals                             3   <-- all three
current contract id      metafind_annot_v4@52f6b2c72fce2950
records carrying that contract                              0
```

Each of the 3 residuals: `in_renders=True`, `is_complete=False`, `annotation_contract` absent.

At n05's observed 39/min a bare run is ~19.6 GPU-hours, and it would rewrite the residuals into the current schema — **resolving `D0-003` by mutation.**

### Other state

- `stage1_hyperparameters.json`: `init_temperature: 0.07`, `learnable_temperature: true`. Untouched by D10.
- `stage1_encoding_protocol.json`: records `metafind_v1_natural`. Stale.
- Serializer identity today: `metafind_v2_cm@8e4b1fcc66c7f48c`.
- Pre-flight PASSES. Test suite **525 passed**. `check_graph.py` 2275 all pass.
- `data/outputs/embeddings/` = 5,276 stale `.npz`, all cache-invalid. `checkpoints/` empty.
- No script auto-invokes n05; `tools/` references are monitoring only.

---

## 7. Scope

### In Scope

**C-001 — τ = 0.5.** Make it reachable through n05b.

**User decision, 2026-08-21 — binding, and the two halves have different authority:**

| Element | Authority | Wording that must be used |
|---|---|---|
| `init_temperature = 0.5` | **PAPER FACT** | `3experiments.tex:15`, "The temperature is 0.5 for all experiments." The paper states this explicitly |
| `learnable_temperature = false` | **USER-RATIFIED IMPLEMENTATION CHOICE**, resting on a strongly-supported INFERENCE | **Correction, user decision 2026-08-21.** An earlier draft of this contract said "the paper never uses the word learnable". **That was false.** The paper does use it — `2methdology.tex:54` types `f_h` and `f_x` as "two **learnable** functions", and `:87` calls λ "a **learnable** scalar". What the paper never does is call the contrastive temperature τ learnable: `2methdology.tex:79` and `:99` both name τ "a temperature **hyperparameter**". The authors therefore distinguish learnable quantities from hyperparameters in their own vocabulary, and placed τ on the hyperparameter side **twice**. Combined with "The temperature is 0.5 for all experiments", that is a strong inference — but it remains an **INFERENCE**. The paper nowhere states that τ is non-learnable. **`learnable_temperature: false` must never be presented as a PAPER FACT** |

This reproduction adopts a **fixed τ = 0.5**. Any code comment, artifact field, docstring, or report wording that blurs the two rows above is a defect.

**C-002 — protocol refresh.** Run n05b so `stage1_encoding_protocol.json` records the ratified template and the current serializer identity, and `stage1_hyperparameters.json` records τ = 0.5. Both land in one call.

**AC-1 — legacy rerun protection.** Full text in §7.1. This contract is self-contained: `D10/USER_REVIEW.md` §7.0 is retained as **provenance**, but you do not need it to know the acceptance contract.

---

### 7.1 `AC-1` — full text, as amended by the user 2026-08-21

> **`AC-1`.** Before `D10_stage1-encoding-contract` may be marked `USER_APPROVED`, a safety mechanism must exist and be **demonstrated** such that, absent explicit force or a named migration intent, **no** existing annotation record — neither the accepted legacy-v3 corpus nor the legacy-v1 residuals — is automatically treated by `annotate_run` as requiring re-annotation.

| | Requirement |
|---|---|
| **AC-1.a** | **A bare `python -m metafind.data.annotate_run` queues 0 records TOTAL** — not merely 0 legacy-v3. Verified 2026-08-21: it queues **45,955**, and **all 3 v1 residuals are in that queue**. If the 45,952 were protected but the 3 slipped through, `annotate_run` would rewrite them into the current schema and thereby **resolve `D0-003` by mutation**, before any decision was taken |
| **AC-1.b** | Re-annotation remains reachable through **explicit** force or a **named migration intent**. The mechanism removes the accident, never the capability |
| **AC-1.c** | Three states are **explicit in the record or in a declared registry**, not inferred from a missing field: annotated-under-current-contract · accepted-legacy-v3 · legacy-v1-residual-unresolved.<br><br>**User confirmation, 2026-08-21: this section is the authoritative wording, and the "declared registry" route is approved.** `data/outputs/annotation_provenance.json` as a declared registry is **permitted**.<br><br>Still prohibited: using a missing `annotation_contract` to mean accepted · stamping a v4 `annotation_contract` onto legacy-v3 records · placing the 3 v1 residuals in the legacy-v3 population.<br><br>`D10/USER_REVIEW.md` §7.0's narrower "in the record" wording is **stale**; Master updates it. **The D2a executor must not modify any D10 file.** |
| **AC-1.d** | Demonstrated **without** loading the annotation model or consuming GPU time |
| **AC-1.e** | The **45,952 only** are recorded as **legacy-v3 validated under `VALIDATOR_VERSION 2`**. They must not be relabelled as v4-generated and must not be given a v4 contract id they did not earn. **The 3 residuals must not be labelled legacy-v3**, and nothing may present `D0-003` as resolved |

Provenance: `workflow/tasks/D10_stage1-encoding-contract/USER_REVIEW.md` §7.0, and the user's amendment of 2026-08-21. If that file and this section ever disagree, **report it** — do not pick one.

### Explicit Non-Scope

- ❌ **Do not run n06** (`metafind.data.encode_text_image`), with or without `--limit`.
- ❌ **Do not run n09** (`metafind.data.splits`).
- ❌ **Do not run any training or gallery indexing.**
- ❌ **Do not re-annotate anything.** No `annotate_run` invocation that mutates a record — including the 3 residuals.
- ❌ **Do not decide `D0-002`** (tower sharing). It binds in n09.
- ❌ **Do not decide or resolve `D0-003`.** The 3 residuals stay `legacy-v1 residual / unresolved / not migrated`.
- ❌ **Do not perform a v4 re-annotation** of any population. The user ruled it out.
- ❌ **Do not proactively reopen D10's existing scope** — serializer, `text_serialization_id`, cache validity, `load_protocol` rejection, pre-flight, the >77 gate, P-1…P-5, contract versioning, the 3 authorised translations.
  **However:** if this task or Codex finds **new contradictory evidence or a correctness defect**, report a `MASTER-IMPACTING FINDING` and **stop**. Do not silently suppress it, and do not silently repair it. (Same rule as §4 — stated here so a reader of §7 alone does not miss the exception.)
- ❌ **Do not change the ratified template or any serialized string.** D0-008 §12.3's scope guard still holds.
- ❌ Do not touch `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, or `DECISION_LEDGER.md`.

---

## 8. Master's Standing Rulings

**R-A — two prohibited shortcuts for `AC-1`.** Both would satisfy the letter and violate the decision:

1. **Do not stamp a v4 `annotation_contract` onto v3 records.** They did not earn it; it is precisely the "disguised as v4-generated" the user forbade (decision #3, AC-1.e).
2. **Do not make "missing `annotation_contract`" implicitly mean "accepted".** AC-1.c requires the three states to be **explicit in the record or in a declared registry**, not inferred from absence. Absence is what created this hazard.

**R-B — least mutation.** The 3 residuals are under an unresolved decision. **Prefer a mechanism that does not mutate them at all.** If you conclude that marking them is unavoidable, that is a `MASTER-IMPACTING FINDING`: report it with evidence and stop rather than deciding it locally.

**R-C — capability, not prohibition.** AC-1.b requires re-annotation to stay reachable through explicit force or a named migration intent. Do not solve AC-1 by deleting the ability to re-annotate.

**R-D — mechanism is an IMPLEMENTATION CHOICE.** This contract states conditions, not a design. Whatever mechanism you choose must be classified and justified in the HANDOFF, and must satisfy every sub-condition of `AC-1` rather than the one that is easiest to demonstrate.

---

## 9. Allowed Writes / Protected Files

This section is authoritative for what may be written. DoD 12 is checked against **this** list, not against a narrower one.

### 9.1 ALLOWED — source and tests

| Path | For |
|---|---|
| `metafind/models/resolve_stage1.py` | τ path (C-001) |
| `metafind/data/annotate_run.py` | work-list / completion semantics (AC-1.a, AC-1.b) |
| `metafind/data/annotate.py` | contract / provenance vocabulary, if AC-1.c needs it |
| `tests/test_resolve_stage1.py`, `tests/test_annotate.py` | coverage |
| `tests/test_encode_text_image.py` | only if the τ or protocol change touches its fixtures |
| a new file under `tools/` | only if the AC-1 proof needs a runnable check |
| `docs/graph/README.md` **line 270 only** | **NARROW SCOPE EXCEPTION, user-approved 2026-08-21.** Update the stale test counts **413 → 435** and **525 → 547**, verified by Master against `pytest --collect-only` (547 collected) and a `def test_` count (435). `tools/check_graph.py:415` asserts the README figure equals the count in `tests/`, so adding tests breaks the checker until this is updated. **Documentation only.** This does **not** lift protection on anything else under `docs/**` |

### 9.2 ALLOWED — artifacts and task outputs

| Path | Conditions |
|---|---|
| `data/outputs/stage1_hyperparameters.json`<br>`data/outputs/stage1_encoding_protocol.json` | **Written by n05b only, never by hand.** Back up first (§10.3) |
| a backup location for the two protocol artifacts above | Required by §10.3. Name it in the HANDOFF |
| `data/outputs/annotation_provenance.json` — or another provenance registry under `data/outputs/` | **Explicitly approved by the user 2026-08-21** as the AC-1.c "declared registry" route. Must not be a v4 contract id on v3 records (R-A.1), and must not place the 3 residuals in the legacy-v3 population |
| `workflow/tasks/D2a_stage1-protocol-refresh/HANDOFF.md`<br>`workflow/tasks/D2a_stage1-protocol-refresh/CODEX_REVIEW.md` | Required by §16 |
| `workflow/tasks/D2a_stage1-protocol-refresh/TASK.md` | Status line only — `PROPOSED` → `ACTIVE` at start |

### 9.3 PROTECTED — do not write, move, delete, or regenerate

| Path | Why |
|---|---|
| `data/outputs/annotations/**` | **All 45,955 must be byte-identical at completion** (DoD 9). Covers both the 45,952 and the 3 residuals |
| `data/outputs/annotations_v1_prompt1/**`, `annotations_v2_sample/**`, `annotations_v3_pre_D10/**` | Backups. Never delete |
| `data/outputs/embeddings/**` | 5,276 stale `.npz` — invalidated, **not** to be deleted |
| `data/outputs/checkpoints/**` | Empty; must stay empty |
| `data/outputs/logs/annotations_index.jsonl` | Rebuilding it is a mutation of corpus state. If you believe it must change, escalate |
| `metafind/data/encode_text_image.py` | D10's consumer. This task satisfies it, does not edit it |
| `metafind/models/dual_tower.py`, `fusion.py`, `losses.py`, `essgnn.py`, `metafind/train/**` | Out of scope entirely |
| `tools/preflight_stage1_text.py` | D10's gate. Must still pass unmodified |
| `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md` | Master's |
| `workflow/decisions/**`, `workflow/tasks/D10_stage1-encoding-contract/**` | Another task's / a settled decision's record |
| `docs/**` — **except** `docs/graph/README.md` line 270, see §9.1 | Untouched by this task. The single-line exception is narrow and does not generalise: `docs/audit/**`, `docs/graph/` everything else, and `docs/paper/**` remain fully protected |
| `data/outputs/renders/**`, `pointclouds/**`, `scene_graphs/**` | Untouched by this task |

Anything not listed in 9.1 or 9.2 is protected by default. Writing to it is a scope violation — escalate instead.

---

## 10. Execution Requirements

1. Read `D10/USER_REVIEW.md` §7.0 for **provenance / historical context**, not to discover the acceptance contract. **§7.1 of this file is self-contained and authoritative.**
2. **Establish AC-1's mechanism and demonstrate it BEFORE running n05b.** n05b is a one-way artifact write; the protection should exist first.
3. Back up `stage1_hyperparameters.json` and `stage1_encoding_protocol.json` before n05b overwrites them.
4. Smallest coherent change per item. No unrelated refactoring.
5. Never mutate an annotation record in this task.
6. Report any Master-impacting discovery immediately.
7. Stop if a required authority decision is missing.

---

## 11. Master-Impacting Finding Rule

Report `MASTER-IMPACTING FINDING` for anything affecting project architecture, accepted research interpretation, a cross-task dependency, another task's contract, milestone feasibility, or a global runtime assumption.

Include the finding, the evidence, affected tasks, and whether this task can safely continue. Do not decide it locally.

Specifically escalate rather than decide: any need to mutate the 3 residuals (R-B), and any conflict between `AC-1`'s sub-conditions.

---

## 12. Verification Requirements

### Required Checks

- **`load_protocol()` returns without raising** — call it directly.
- **AC-1.a — the central proof.** Reproduce `annotate_run`'s work-list computation **without loading the model**, and show:

  ```
  rendered assets                              45,955
  bare annotate_run todo (no --force)               0
    accepted legacy-v3 queued                       0
    legacy-v1 residuals queued                      0
  ```

- **AC-1.b — negative test.** Explicit force / named migration intent still yields a non-empty work list. Demonstrate the *computation*, never an actual annotation run.
- **AC-1.c** — the three states are explicitly distinguishable, and the distinction does not rest on a missing field.
- **AC-1.e** — the 45,952 carry a legacy-v3 / `VALIDATOR_VERSION 2` provenance statement; the 3 residuals do **not**, and nothing anywhere presents `D0-003` as resolved.
- `stage1_hyperparameters.json` records `init_temperature: 0.5`, `learnable_temperature: false`.
- `stage1_encoding_protocol.json` records the ratified template and `metafind_v2_cm@8e4b1fcc66c7f48c`.
- `tools/preflight_stage1_text.py` still PASSES.
- `git diff --stat` touches only paths permitted by §9.1 / §9.2, and nothing in §9.3.

### 12.1 Exact verification commands

`PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python`, run from the repository root.

> These commands implement the requirements above. **They do not specify how `AC-1` is to be built.** Where a command reaches into module internals it is doing so to *observe* the work list, not to mandate where the filter lives — see the adaptation clause under AC-1.a.

**Before any edit — capture the baseline:**

```bash
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # corpus fingerprint
ls data/outputs/embeddings/*.npz | wc -l                                     # expect 5276
ls data/outputs/checkpoints | wc -l                                          # expect 0
cp data/outputs/stage1_hyperparameters.json  <backup>/                       # §10.3
cp data/outputs/stage1_encoding_protocol.json <backup>/                      # §10.3
```

**DoD 3 — `load_protocol()` no longer raises:**

```bash
$PY -c "from metafind.data.encode_text_image import load_protocol; p=load_protocol(); print('OK', p['text_serialization'])"
```

**DoD 1 / 2 — the emitted artifacts:**

```bash
$PY -c "import json;d=json.load(open('data/outputs/stage1_hyperparameters.json'))['values'];print(d['init_temperature'], d['learnable_temperature'])"
$PY -c "import json;d=json.load(open('data/outputs/stage1_encoding_protocol.json'));print(d['text_serialization']);print(d['text_template'])"
```

**DoD 4 — `AC-1.a`, the central proof. No model load, no GPU.**

Reproduce `annotate_run`'s own work-list computation read-only and report the three populations separately:

```bash
$PY - <<'EOF'
import json
from metafind.data import annotate_run as R
from metafind import paths
renders = {json.loads(l)["uid"] for l in (paths.LOGS/"renders_index.jsonl").read_text().splitlines() if l.strip()}
V1 = {"6c7db00cc164467ebac356a5ca67368b","8a0192eee6fb4140bb3e9696b3dbae5a","a397b648d6eb48d7909d1ee11235e78f"}
todo = [u for u in sorted(renders) if not R.is_complete(u)]          # the bare-run predicate
print(f"rendered assets                        {len(renders):>7,}")
print(f"bare annotate_run todo (no --force)    {len(todo):>7,}")
print(f"  accepted legacy-v3 queued            {len([u for u in todo if u not in V1]):>7,}")
print(f"  legacy-v1 residuals queued           {len([u for u in todo if u in V1]):>7,}")
EOF
```

Required output: `0`, `0`, `0` on the last three lines.

> **SUPERSEDED — historical record, do not reuse.** The mechanism relocated the work-list predicate to `build_work_list()` (`annotate_run.py:439`), so the snippet below now checks a path the real run no longer takes and returns a **false negative** (Master reproduced 45,955 with it before switching to the real predicate). It is retained because it is what the contract said at execution time. **The equivalent that actually exercises `main()`'s predicate is recorded in `USER_REVIEW.md` §2.**

**Adaptation clause.** The snippet above mirrors the `annotate_run.main()` predicate **as it stood when this contract was written** (`annotate_run.py:250`). If your mechanism relocates the filter, **do not** leave this snippet checking a path the real run no longer takes. Supply an equivalent read-only command that exercises whatever `main()` actually uses to build `todo`, and argue the equivalence in the HANDOFF. What is fixed is the *claim* — a bare run queues 0 total — not this snippet.

**Never** satisfy this by invoking `annotate_run` itself: that loads the model and, at `todo > 0`, would begin writing.

**DoD 5 — `AC-1.b`, the negative test.** Show that an explicit force / named migration intent still produces a **non-empty** work list, by computing it — never by running an annotation. Assert it in `tests/test_annotate.py`, and state the invocation form in the HANDOFF.

**DoD 7 / 8 — `AC-1.e` and `D0-003`:**

```bash
grep -rniE "legacy.?v3|validator_version" data/outputs/*.json* metafind/ tests/ | head -40
grep -rniE "d0-003|residual" metafind/ tests/ tools/ | grep -iE "resolv|fixed|closed|done" || echo "no claim that D0-003 is resolved"
```

**DoD 9 — the corpus is untouched:**

```bash
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # must equal the baseline
```

**DoD 10 — gates and suites:**

```bash
$PY tools/preflight_stage1_text.py     # must still print PRE-FLIGHT PASSED
$PY -m pytest tests/ -q                # no --ignore flag; baseline 525 passed
$PY tools/check_graph.py               # all pass
```

**After the `docs/graph/README.md` line-270 edit — re-run the full battery.** The counts move only if tests changed, so this is the moment the whole verification state must be re-established, not just the checker:

```bash
$PY -m pytest tests/ -q                                                       # counts must match the README
$PY tools/check_graph.py                                                      # must pass with the new figures
$PY tools/preflight_stage1_text.py                                            # must still print PRE-FLIGHT PASSED
find data/outputs/annotations -name '*.json' | sort | xargs md5sum | md5sum   # must equal the baseline
ls data/outputs/embeddings/*.npz | wc -l                                      # still 5276
ls data/outputs/checkpoints | wc -l                                           # still 0
git status --porcelain
git diff --stat
```

If all of these pass, update the **final verification state** in `HANDOFF.md` and `CODEX_REVIEW.md` to the post-edit figures before returning to Master. Do not leave a HANDOFF quoting pre-edit counts.

**DoD 11 / 12 — artifacts and scope:**

```bash
ls data/outputs/embeddings/*.npz | wc -l    # still 5276
ls data/outputs/checkpoints | wc -l         # still 0
git status --porcelain
git diff --stat
```

---

### Required Tests

- `python -m pytest tests/ -q` — no `--ignore` flag. Baseline **525 passed**.
- `tools/check_graph.py` — all checks pass.
- New coverage for the AC-1 mechanism, including the negative test for AC-1.b and a test that the 3 residuals are never auto-queued.

### Runtime / Artifact Checks

- `data/outputs/embeddings/` still **5,276** `.npz`, unmodified. No new embeddings.
- `data/outputs/checkpoints/` still empty.
- **All 45,955 annotation records byte-identical to task start.** Verify by checksum.
- Protocol artifacts backed up before n05b.

### Research Fidelity Check

- τ = 0.5 traceable to `3experiments.tex:15`, not to a convention. Confirm the emitted artifact records 0.5 and that no library default reintroduced 0.07.
- **The two authorities stay distinct** (§7 C-001): `0.5` is a PAPER FACT; `learnable_temperature: false` is a USER-RATIFIED IMPLEMENTATION CHOICE on a strongly-supported inference. Grep whatever you write for wording that would let a reader take the second for the first.

---

## 13. Definition of Done

- [ ] **1.** `stage1_hyperparameters.json` records `init_temperature: 0.5`, `learnable_temperature: false`.
- [ ] **2.** `stage1_encoding_protocol.json` records the ratified template and `metafind_v2_cm@8e4b1fcc66c7f48c`.
- [ ] **3.** `load_protocol()` returns without raising — demonstrated by direct call.
- [ ] **4. AC-1.a** — a bare `annotate_run` queues **0 records TOTAL**: 0 legacy-v3 **and** 0 legacy-v1 residuals. Demonstrated **without loading the annotation model and without GPU time**.
- [ ] **5. AC-1.b** — explicit force / named migration intent still produces a non-empty work list. Negative test present and passing.
- [ ] **6. AC-1.c** — three states explicit in the record or a declared registry: annotated-under-current-contract · accepted-legacy-v3 · legacy-v1-residual-unresolved. **Not inferred from a missing field.**
- [ ] **7. AC-1.e** — the **45,952 only** recorded as legacy-v3 validated under `VALIDATOR_VERSION 2`. No v4 contract id on any v3 record. The 3 residuals **not** labelled legacy-v3.
- [ ] **8.** Nothing anywhere states or implies that `D0-003` is resolved.
- [ ] **9.** All 45,955 annotation records byte-identical to task start.
- [ ] **10.** Pre-flight still PASSES; `pytest tests/ -q` and `check_graph.py` pass. If `docs/graph/README.md:270` was updated under the §9.1 exception, the **whole** battery was re-run afterwards and `HANDOFF.md` / `CODEX_REVIEW.md` quote the post-edit figures.
- [ ] **11.** Embeddings still 5,276; checkpoints still empty; n06, n09, and training never invoked.
- [ ] **12.** `git diff` touches only paths permitted by **§9.1 and §9.2**. No path in §9.3 is modified. Unexpected files investigated before completion.
- [ ] **13.** Codex review completed; material findings independently verified by Claude.
- [ ] **14.** `HANDOFF.md` and `CODEX_REVIEW.md` written, including the `USER REVIEW INPUT` section.

The task owner does not mark the Stage 1 milestone DONE, and does not mark `D10` accepted. Master reviews; the user decides.

---

## 14. Codex Review Requirement

Scope to this task. Provide Codex with this `TASK.md`, the diff, the verification output, `D10/USER_REVIEW.md` §7.0, and known uncertainties.

Ask Codex to attack:

- whether AC-1.a's proof is genuine or merely restates the implementation;
- **whether any path still queues a record without explicit force** — a different entry point, a flag combination, an empty/corrupt sidecar, a uid absent from the registry;
- whether the mechanism silently makes "missing field" mean "accepted" after all (R-A.2);
- whether the 3 residuals could be swept into the legacy-v3 population by any code path;
- whether anything now implies `D0-003` is resolved;
- whether AC-1.b's capability survives, or was removed rather than gated;
- whether the τ change can be defeated by a library default, an override path, or a later n05b call;
- whether n05b's write silently changed anything besides τ and the template record.

Codex must not be asked merely to confirm.

---

## 15. Claude Verification of Codex Findings

Classify each material finding: `CONFIRMED` · `PLAUSIBLE` · `REJECTED` · `UNVERIFIED`.

If review is unavailable through quota, auth, timeout, or runtime failure: `CODEX REVIEW UNAVAILABLE`. **That is not PASS**, and it must reach the HANDOFF's `USER REVIEW INPUT`.

---

## 16. Required Handoff

Write `HANDOFF.md` and `CODEX_REVIEW.md` in this directory, per `workflow/tasks/HANDOFF_TEMPLATE.md` — including §15 `USER REVIEW INPUT`.

Report findings and decisions **separately** (`WORKFLOW.md` §13A): FINDING · EVIDENCE · IMPLEMENTATION / PROPOSED DECISION · AUTHORITY · IMPACT · UNKNOWN.

State explicitly in the HANDOFF that **no n06 run, no n09 run, no training, no GPU embedding generation, and no annotation mutation occurred.**

Then stop. Do not start `D1_n06-reencode`.

---

## 17. Master Recommendation and the User Review Gate

Master reviews and returns a **MASTER RECOMMENDATION**, not an acceptance.

```
task completion
→ Master integration review
→ MASTER RECOMMENDATION
→ USER REVIEW BRIEF
→ USER decision: APPROVE / REJECT / MODIFY / INVESTIGATE MORE
→ on APPROVE: FINAL ACCEPTED
```

Until the user approves: integration is `AWAITING_USER_REVIEW`, the task is **not** DONE, `D1` is **not** unblocked, and no global state file records the result.

**On `D2a` `USER_APPROVED`:**

- `D10_stage1-encoding-contract` returns for final USER REVIEW carrying this task's `AC-1` evidence.
- **`D1_n06-reencode` remains BLOCKED.**
- **Only `D10`'s `USER_APPROVED` unblocks `D1_n06-reencode`.**

Acceptance of this task does not, by itself, unblock `D1`.
