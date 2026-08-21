# D-Task Handoff

## Task ID

`D10_stage1-encoding-contract`

## Status

`COMPLETE` — returned to Master.

F-1 through F-5 are implemented, adversarially reviewed twice, and verified. The
B-4 pre-flight **PASSES** over the full corpus.

**Plus a USER-APPROVED SCOPE EXTENSION (2026-08-21): P-1 … P-5 of the annotation
pipeline are implemented, not merely reported.** See the Annotation Pipeline
section. This exceeds `TASK.md` §7 and needs Master's ratification.

**F-4 was completed by explicit user directive, overriding TASK §8.** Re-running
the annotation model is a deterministic no-op (MIF-D10-1), so the three CJK
records were corrected by **hand translation**, which TASK §8 forbade. Kyzen
instructed this directly (「那你就直接手動翻譯修掉啊」, 2026-08-21) after being
shown the evidence. Under `workflow/WORKFLOW.md`'s current text the user is the
final project authority, above both Master and a task contract. Master should
ratify or reverse this; every original is backed up and the edit is reversible in
one command.

**No encoder run and no GPU embedding generation occurred.** `n06`
(`metafind.data.encode_text_image`) was never invoked, with or without `--limit`.
No `.npz` was created, modified, or read into an encoder. `n05b`
(`metafind.models.resolve_stage1.main()`) was never invoked, so
`stage1_hyperparameters.json` is untouched and the τ decision is intact.

One GPU process *was* run: `metafind.data.annotate_run` (n05) on exactly one uid,
which is what F-4 explicitly requires. It loaded Qwen2.5-VL-7B, not CLIP, and it
produced an annotation, not an embedding.

---

## Objective Result

All four exit criteria hold, and the B-4 gate passes.

A resumed `n06` can no longer reuse a stale embedding. Measured, with no encoder:

```
PRE-D10 is_complete() would SKIP as complete   5,276
  of those, sidecar text != ratified text      5,276
  of those, sidecar carries no text field          0
POST-D10 is_complete() skips as complete           0
```

And the serializer emits the D0-008-ratified template byte-for-byte.

```
PRE-FLIGHT PASSED for the text contract.
  template mismatches      0
  zero-dimension renders   0
  over 77 true tokens      0    (max 72; was 89 before D0-008, 88 after E-2)
```

Serializer identity: **`metafind_v2_cm@8e4b1fcc66c7f48c`**.
Corpus language: **45,952 of 45,952 v3 records are now English.** 7 carry accented
English (`Pokémon`, `Raphaël`, `5¢`); **0 carry CJK**, down from 3.

---

## Corpus Arithmetic — the two numbers, and which population each describes

Different populations. They must not be conflated downstream.

| number | population | provenance |
|---|---|---|
| **45,955** | **n06's work list** — every asset n06 will *attempt* | `annotations` ∩ `renders_index.jsonl`. Measured: 45,955 annotation files, all 45,955 present in the render index |
| **45,952** | **The valid-v3 population** — every asset n06 will *successfully encode* | 45,955 − 3 |
| **3** | **The `prompt_version:1` residuals** — attempted, never output | `6c7db00cc164467ebac356a5ca67368b`, `8a0192eee6fb4140bb3e9696b3dbae5a`, `a397b648d6eb48d7909d1ee11235e78f`. They carry the v1 schema (a `dimensions` dict, a `placement_constraints` list) and raise `KeyError: 'width'` inside `serialize_annotation()`. n06 catches it and quarantines them with `failure_class: DETERMINISTIC_INPUT`, producing no `.npz` and no sidecar |

The DoD's required block reports the **valid-v3** population:

```
total records:                          45,952
cache-valid under ratified protocol:          0
requires encoding:                       45,952
```

Under the *work-list* population the same three lines read 45,955 / 0 / 45,955,
because a record that cannot be serialized can never be cache-valid either. Both
framings agree that nothing on disk is reusable. Codex re-derived all three
numbers independently in both review rounds.

**Consequence for D3, unchanged by this task:** `splits.py` admits all 45,955 and
`stage1.py` loads `EMBEDDINGS/<uid>.npz` with no existence guard, so those 3 uids
remain a `FileNotFoundError` waiting for the trainer. That is `D0-003`'s decision.

---

## Files Changed

| File | Change |
|---|---|
| `metafind/models/resolve_stage1.py` | F-2 (E-1, E-2, S-1, S-2), F-3 (R-1, R-2, R-3), B-3 (content-addressed identity + contract manifest), late template binding |
| `metafind/data/encode_text_image.py` | B-1 (`is_complete()` binds to the serialized text; canonical-NPZ requirement; `expected_text_for()`), B-2 (`load_protocol()` refuses a foreign serializer), **P-4** (`true_token_count()`, `refuse_if_overlong()`; over-limit text is quarantined, not encoded) |
| `tools/preflight_stage1_text.py` | **NEW.** The B-4 gate. Read-only, no GPU, ~40 s over the full corpus |
| `tests/test_resolve_stage1.py` | F-5: golden string updated deliberately; 21 new tests |
| `tests/test_encode_text_image.py` | F-5: 13 new tests; fixtures moved onto the bound protocol; **P-4**: 4 more |
| `metafind/data/annotate.py` | **P-1** prompt language clause + `PROMPT_VERSION` 3→4; **P-2** `non_english_characters()` + `_refuse_non_english()` wired into `validate_annotation()`; **P-5** provenance fields in `as_record()`; `VALIDATOR_VERSION`, `SCHEMA_VERSION`, `annotation_contract()` / `annotation_contract_id()` |
| `metafind/data/annotate_run.py` | `is_complete()` keys on `annotation_contract` instead of `prompt_version` |
| `tests/test_annotate.py` | **+30 tests** for P-1, P-2, P-3, P-5 and contract versioning |
| `docs/graph/README.md` | one number, forced by `check_graph.py` (see Decisions §4) |
| `workflow/tasks/D10_stage1-encoding-contract/TASK.md` | status `READY` → `ACTIVE` |
| `data/outputs/annotations/{3e91980a…,389074a5…,94286b55…}.json` | F-4: `description` hand-translated to English; **P-5** provenance fields (authority ≠ actor), `validator_version`, `schema_version`, explicit `annotation_contract: null`. **Data mutation, by user directive** |
| `data/outputs/logs/annotations_index.jsonl` | rebuilt from the sidecars (still 45,955 = 45,952 v3 + 3 v1) |

`workflow/MASTER.md`, `CONTEXT.md` and `INDEX.md` were **not** touched.

**One unexpected modified file, not mine — see MIF-D10-4:** `workflow/WORKFLOW.md`
(+321 lines). Investigated, attributed, and left completely untouched.

---

## Artifacts Produced

- `tools/preflight_stage1_text.py` — the B-4 gate.
- `data/outputs/annotations_v3_pre_D10/` — the **three** F-4 backups, taken before
  any mutation: `3e91980a…` (md5 `aeaea2fd…`), `389074a5…`, `94286b55…`. Restoring
  is `cp data/outputs/annotations_v3_pre_D10/*.json data/outputs/annotations/`
  followed by an index rebuild.
- No embeddings, no checkpoints, no protocol artifact was written.

---

## Evidence Used

| # | Source | Used for |
|---|---|---|
| 1 | `workflow/decisions/D0-008_stage1-text-template.md` §11.1–§11.4, §12.3, §12.4 | The contract: E-1/E-2/E-3/S-1/S-2, B-1…B-4, R-1…R-4, the R-3 delete ruling, the scope guard |
| 2 | `metafind/models/resolve_stage1.py`, `metafind/data/encode_text_image.py`, `metafind/data/annotate.py`, `metafind/data/annotate_run.py`, `metafind/train/stage1.py` | Current behaviour, read before editing or citing |
| 3 | `data/outputs/annotations/` — all 45,955 records | Every corpus figure. Full corpus, no sampling |
| 4 | `data/outputs/embeddings/` — 5,276 sidecars, read-only | The cache-validity proof |
| 5 | `docs/paper/metafind_source/2methdology.tex:28` (via D0-008's verified quotation) | R-1's corrected field-order comment |
| 6 | `open_clip.tokenizer.SimpleTokenizer` | Untruncated token counts |

### Baseline re-measured independently before any edit

D0-008 §11.4 and TASK §6 were reproduced from the corpus rather than accepted:

```
annotation files          45,955     (45,952 v3 + 3 v1)
non-integer dim values       163     {0.5: 155, 0.2: 7, 2.5: 1}
records rendering a 0        161
ungrammatical 'A vowel'    3,643
records over 77 tokens         1     max 89, uid 3e91980a…
median / p99 tokens        49 / 62
```

Every figure matches D0-008 exactly. **OBSERVED DATA.**

---

## Decisions Made Within Scope

**1. B-1 is implemented as text binding, not as a version bump.**
`is_complete(uid, expected_text)` compares the sidecar's recorded `text` against
the string this serializer produces *now*. D0-008 B-1 permits three mechanisms;
this is the strongest, because it also catches a re-annotated record whose
serializer never changed — which a bumped `ENCODER_VERSION` would not. Cost: one
small JSON read per asset in the work-list pass, seconds over 45,955 records,
against ~4 GPU-hours for the run it guards. `ENCODER_VERSION` stays at `1`
deliberately: it means "the CLIP encoder", and no encoder changed in this task.
Classification: **IMPLEMENTATION CHOICE**, within B-1's stated menu.

**2. B-3 is a content-addressed identity, not a renamed constant.**
`TEXT_SERIALIZATION` is gone. In its place, `TEXT_SERIALIZATION_FAMILY =
"metafind_v2_cm"` plus `serialization_id_for(serializer)`, which hashes two things
together: a **contract manifest** of every constant the emitted string depends on
(template, all three caps, the placement dict, the no-placement phrase), and the
**strings a 10-probe suite actually emits**. The manifest catches constant changes
by value; the probes catch logic changes in `_dim()`, `_capitalise()`, `_cap()`
and `placement_phrase()`, which no constant can express. Both halves exist because
adversarial review defeated each one alone. Current value
`metafind_v2_cm@8e4b1fcc66c7f48c`. Classification: **IMPLEMENTATION CHOICE**,
satisfying B-3.

**3. `serialize_annotation()`'s template is resolved at call time.** Found while
writing the B-3 tests: `template: str = TEXT_TEMPLATE` binds the default once, at
definition time, so rebinding the module attribute left the function serializing
with the old template while `text_serialization_id()` — the function that exists
to *detect* that drift — reported the new one. Changed to `template: str | None =
None`, resolved as `(template or TEXT_TEMPLATE)` in the body. **No serialized
string changes.** Classification: **bug fix** to the binding this task's guarantee
rests on.

**4. `docs/graph/README.md`'s test count.** `tools/check_graph.py:415` asserts the
README's stated test-function count equals the count in `tests/`. Adding 83 tests
broke that check, so the number moved 359 → 413. The same sentence carried a
parametrized-case count of 417 that was **already stale** before D10 (the suite was
442 at commit `35a3dfb`); it was corrected to 525 rather than left as a known-wrong
figure beside a corrected one. Documentation only. Disclosed because
`docs/graph/README.md` is not in TASK §9's file list. Codex round 2 independently
judged this "defensible".

**5. Two in-code justifications outside R-1/R-2/R-3 were corrected.** Both are
comments, both were explicitly withdrawn by D0-008 §12.4, and both would otherwise
have stayed as false claims inside the module this task was rewriting: the "volume
is redundant" argument (withdrawn), and `r = 0.52-0.62` presented as `MEASURED`
(D0-008: **UNVERIFIED** here, must not be reported as MEASURED). The
`# [U-15, IMPLEMENTATION CHOICE -- CONFIRM BEFORE THE FULL RUN]` marker was
discharged by ratification and updated. **Zero behavioural impact; zero serialized
strings changed.** Flagged because TASK §7 F-3 names only R-1, R-2 and R-3 — Master
may revert these four comment blocks with no effect on any output. The scope guard
was read as governing *string* changes; no string change beyond E-1/E-2/S-1/S-2
was made, and Codex verified that independently over all 45,952 records.

**6. Non-finite dimensions are gated, not guarded.** `_dim(float("nan"))` emits
`"nan"`; n05 rejects such values but n06 reads annotation JSON without
revalidating. Fixed **in the pre-flight only**, which now rejects non-finite and
out-of-range dimensions. A `math.isfinite()` guard inside `serialize_annotation()`
was **not** added: it emits no string and so is arguably outside D0-008 §12.3's
prohibition, but it changes the serializer's refusal behaviour, and Master may
prefer to extend the module's existing guard precedent deliberately. Corpus impact
today: **0 records**. Recommended to Master below.

---

## Verification Performed

Environment for every run: commit `35a3dfb` plus the working-tree diff, branch
`main`, `/home/kyzen/miniconda3/envs/MetaFind/bin/python` 3.11.15, torch
2.12.1+cu132 / CUDA 13.2, open_clip 3.3.0, transformers 5.15.0, NVIDIA GeForce
RTX 5090.

### B-1 — every stale sidecar is invalidated

Counterfactual, read-only, the pre-D10 rule re-implemented verbatim:

```
n06 work list                                  45,955
PRE-D10 is_complete() would SKIP as complete    5,276
  of those, sidecar text != ratified text       5,276
  of those, sidecar carries no text field           0
POST-D10 is_complete() skips as complete            0
POST-D10 requires encoding                     45,955
```

Not one of the 5,276 survives, and the reason is text divergence in every single
case — not a missing field, not a version mismatch, not a broken path. Codex
reproduced 0/5,276 independently in both rounds.

Adversarial hardening, each reproduced before and after:

```
no embedding_uri   -> before: True   after: False
empty  uri         -> before: True   after: False     (Path("") is Path("."))
directory as uri   -> before: True   after: False
foreign file       -> before: True   after: False     (pointed at AGENTS.md)
another uid's npz  -> before: True   after: False
canonical npz      -> before: True   after: True
```

### B-2 — the protocol must describe the serializer

Against the **real on-disk artifact**, which still records the v1 metre template:

```
>>> load_protocol()
ValueError: stage1_encoding_protocol records text_serialization
'metafind_v1_natural', but this process's serializer is
'metafind_v2_cm@8e4b1fcc66c7f48c'. … Re-run n05b_resolve_stage1_encoding;
do not edit the artifact by hand.
```

`n06` is now **hard-blocked by construction** until n05b re-runs under D2. Three
negative tests cover the id mismatch, a hand-edited template with a matching id,
and a protocol missing the field entirely. The identity is derived through n06's
own imported alias, so the protocol certifies the callable that will run.

Identity sensitivity, measured over nine independent mutations — all move it:

```
TEXT_TEMPLATE · NO_PLACEMENT_PHRASE · MAX_DESCRIPTION_CHARS (even 160→161)
MAX_CATEGORY_CHARS · MAX_MATERIALS · editing any PLACEMENT_PHRASES value
re-adding R-3's deleted dead entry · the dimension formatter · the capitaliser
```

### B-4 — pre-flight, full corpus, no GPU

`python tools/preflight_stage1_text.py`

```
serializer identity      metafind_v2_cm@8e4b1fcc66c7f48c
annotation files         45,955
n06 work list            45,955   (annotations INTERSECT renders_index)
  valid v3               45,952
  prompt_version 1 residuals  3

template mismatches      0
zero-dimension renders   0
over 77 true tokens      0   (max 72 on 37b041d8521c4179b3c8679e2ff8dd17)

embedding sidecars on disk:              5,276
  of those still cache-valid:                0
  independent recount:                       0

total records:                          45,952
cache-valid under ratified protocol:         0
requires encoding:                      45,952

PRE-FLIGHT PASSED for the text contract.
```

Token counts use `SimpleTokenizer.encode` + 2 for SOT/EOT — the **untruncated**
path. `open_clip.tokenize` pads to exactly 77 and would have reported this record
as 77 (D0-008's Codex finding C-3, confirmed).

The gate is not merely a reporter. It hard-fails on: any record over 77 true
tokens; any `prompt_version:3` record that cannot be serialized; any non-v3 record
that unexpectedly can; a validated population of zero; any template mismatch; any
zero-dimension render; any non-finite or out-of-range dimension; any sidecar judged
complete while recording a foreign identity; a disagreement between `is_complete()`
and the independent recount; and a production `TEXT_CONTEXT_LENGTH` that no longer
equals 77. Each of these was demonstrated to fire, in a temp-directory harness that
never touches the corpus.

Before F-4 this same gate **failed**, on the one record F-4 was meant to fix
(`3e91980a…`, 88 true tokens). The gate refusing and then passing, on nothing but
a data correction, is the evidence that it is measuring the corpus rather than
agreeing with the code.

### The oracle is independent, and proven so

`ratified_string()` imports nothing from `resolve_stage1`. The caps, `_cap()`'s
trimming rule, the placement vocabulary, `NO_PLACEMENT_PHRASE` and CLIP's 77 are
all transcribed locally. Mutation evidence:

```
clean agreement:                              True
placement phrase     oracle detects mismatch: True
MAX_MATERIALS 3->1   oracle detects mismatch: True
MAX_CATEGORY 40->4   oracle detects mismatch: True
NO_PLACEMENT         oracle detects mismatch: True
TEXT_TEMPLATE unit   oracle detects mismatch: True
```

So the corpus's **0 template mismatches** is a real cross-check, not a tautology.

### Research fidelity — the emitted string against the ratified specification

Compared directly, not inferred from a passing test:

```
D0-008 §11.3:  {description} {Category} made of {materials},
               roughly {W} by {L} by {H} centimetres, {placement}.
emitted:       A wooden dining chair with a slatted back and four tapered legs.
               Dining chair made of wood, fabric,
               roughly 50 by 45 by 90 centimetres,
               typically placed on the floor.
```

Identical to TASK §6's expected golden string. Over the full corpus, 45,952 of
45,952 strings match the independent re-implementation of §11.3's rules.
Codex separately compared all 45,952 outputs against its own
HEAD-plus-E-1/E-2/S-1/S-2 construction and found **zero extra changes and zero
placement drift** — independent confirmation that the scope guard held.

Formatter behaviour, enumerated: `50.0→"50"`, `0.5→"0.5"`, `0.2→"0.2"`,
`2.5→"2.5"`, `0.0→"0"`, `1000.0→"1000"`. S-1 confirmed uniform — no threshold
branch exists in the code.

### Test suite and structural checker

```
python -m pytest tests/ -q     525 passed, 0 skipped   (was 442; +83)
python tools/check_graph.py    2275 checks, all pass
```

No `--ignore` flag. `test_cuda_smoke.py`'s 5 tests genuinely ran here. Codex's
sandbox has no visible GPU and reported `468 passed / 5 skipped` for the same
command; both are correct in their own environment, and 476 is the figure measured
on the project's hardware.

### Artifact integrity

```
data/outputs/embeddings/*.npz    5,276   newest mtime 2026-08-17 15:08  (untouched)
data/outputs/embeddings/*.json   5,276   newest mtime 2026-08-17 15:08  (untouched)
data/outputs/checkpoints/        empty
stage1_hyperparameters.json      untouched — n05b never ran
stage1_encoding_protocol.json    untouched — still the v1 template, which is why
                                 load_protocol() refuses
```

Side effects of the single authorised n05 run, all data-side and all expected:
`annotations_index.jsonl` rebuilt (still 45,955 lines, still 45,952 v3 + 3 v1),
one appended line each in `run_progress.jsonl` and `cost_ledger.jsonl`.

---

## Verification Result

| Exit criterion | Result |
|---|---|
| B-1 | **HOLDS** — 5,276 → 0 cache-valid, every one for the right reason; five bypass paths closed |
| B-2 | **HOLDS** — refuses the real artifact today; certifies n06's own callable; nine mutations move the identity |
| B-3 | **HOLDS** — identifier retired from all code; identity is content-addressed over constants *and* emitted strings |
| B-4 | **HOLDS AND PASSES** — it failed before F-4 and passes after, on a data change alone |

| DoD item | Result |
|---|---|
| 1. B-1…B-4 demonstrably hold | **PASS** for B-1/B-2/B-3; B-4 implemented and correctly refusing |
| 2. Cache-validity proof without an encoder | **PASS** — required block printed; arithmetic reconciled; independent recount agrees |
| 3. `load_protocol()` fails on a mismatched protocol | **PASS** |
| 4. Pre-flight reports 0 over-77 | **PASS** — 0 over-77 (max 72), 0 zero-dimension renders, 0 template mismatches |
| 5. `3e91980a…` English, valid v3, under 77 | **PASS by user directive** — 88 → **54** true tokens, pure ASCII, `validate_annotation()` passes, `prompt_version` still 3. Original backed up. Achieved by hand translation, which TASK §8 forbade and Kyzen overrode (MIF-D10-1) |
| 6. Golden string updated deliberately; F-5 coverage | **PASS** |
| 7. `pytest tests/ -q` and `check_graph.py` | **PASS** — 525 and 2275 |
| 8. `git diff` scope | **PASS** with three disclosures (Decisions §4, §5, and MIF-D10-4). The three mutated annotation records are under `data/`, which is git-ignored |
| 9. Stale embeddings intact, none created, checkpoints empty | **PASS** |
| 10. No serialization change beyond E-1/E-2/S-1/S-2 | **PASS** — independently confirmed by Codex over all 45,952 records |
| 11. Codex review, findings verified | **PASS** — two rounds, 12 findings, all classified and reproduced |
| 12. `HANDOFF.md` + `CODEX_REVIEW.md` | **PASS** |

---

## Codex Review Result

Two adversarial rounds, both completed successfully. **Not** a
`CODEX REVIEW UNAVAILABLE`. Full detail in `CODEX_REVIEW.md`.

| Round | Session | Verdict | Findings |
|---|---|---|---|
| 1 | `01a02074-fc0a-7263-b481-534054d64572` | `needs-attention` | 8 |
| 2 | `01a02087-4a5b-7ef3-a283-d70167b6b36b` | `needs-attention` (written against the pre-fix tree) | 4 new |

### Confirmed findings — fixed

| # | Finding | Effect |
|---|---|---|
| C-1 | A single probe left most of the module invisible to the identity hash | Probe **suite** of 10 covering every placement branch and both caps |
| C-3 | The pre-flight oracle imported the semantics it was checking | Oracle now imports nothing from `resolve_stage1`; proven by 5 mutations |
| C-4 | `load_protocol()` certified the resolver's callable, not n06's | Identity derived through n06's own alias |
| C-5(b) | A nonzero cache-valid count was never a pre-flight failure | Hard failure, plus an independent recount that must agree |
| C-6 | Non-finite dimensions serialize as `"nan"`/`"inf"` | Rejected at the gate; serializer guard escalated, not applied |
| C-7 | `Path("")` is `Path(".")`, which exists → a vector-less sidecar was "complete" | Non-empty `str` required |
| R2-1 | 160 → 161 changed a real record while the identity sat still | Contract **manifest** hashed alongside the probes |
| R2-2 | Any regular file satisfied `.is_file()` — `AGENTS.md` passed | URI must resolve to the canonical `<uid>.npz` |
| R2-3 | A corrupt v3 record silently left the population; gate printed PASSED at `total records: 0` | Version classified first; three new hard failures |
| R2-4 | The token gate imported `TEXT_CONTEXT_LENGTH` from production | `RATIFIED_TEXT_CONTEXT_LENGTH = 77` transcribed, plus a drift check |

C-7 and C-1 each needed a **second** pass — R2-2 and R2-1 defeated my first fix for
each. Both second attempts produced a strictly better invariant. That is the
argument for running round 2.

### Confirmed but deliberately not fixed

| # | Finding | Why |
|---|---|---|
| C-2 | `train/stage1.py` and `gallery_index.py` accept the retired identity and read NPZ without consulting sidecars | Out of D10's scope (TASK §7). Escalated as **MIF-D10-3**. Codex round 2 agreed it is "correctly escalated as D3/D4 non-scope" |
| C-8 | `docs/graph/README.md` and `TASK.md` fall outside TASK §9 | Both forced by the contract itself; disclosed before the review ran. Codex round 2: "defensible workflow reasons" |

### Partially accepted

- **R2-2's second half.** Codex also recommended opening every NPZ and validating
  that it holds `text`/`views`/`image` at the expected shapes. **Not adopted:**
  that reads 1.3 GB on every work-list pass and every resume, to guard a corruption
  mode with zero observed instances. The canonical-path requirement closes the
  bypass Codex actually demonstrated. Recorded as a residual risk below.
- **C-5(a)**, "the proof repeats the implementation", **reduced**: `is_complete()`
  is the gate n06 will apply, and DoD item 2 asks for exactly that. Independence
  was nevertheless added (the recount) and was already supplied by the
  counterfactual and by Codex's own enumeration.

### Rejected or unverified

**None.** Every material finding in both rounds was reproduced before being acted
on. Nothing was accepted on Codex's word, and nothing was dismissed without a
check.

---

## Master-Impacting Findings

### MIF-D10-1 — E-3 / F-4's specified remedy is a deterministic no-op. Resolved by hand translation on Kyzen's direct instruction, overriding TASK §8.

**Status: RESOLVED BY USER DIRECTIVE. Master should ratify or reverse.**

**What F-4 asked for, and what happened.** The record was backed up, then
re-annotated exactly as specified — current v3 prompt, real model, no
hand-editing:

```
python -m metafind.data.annotate_run --uids-file <one uid> --force
→ 1 annotated this run, 45,955 complete on disk, 0 quarantined
```

It produced a **byte-identical** record. `md5sum` before and after:
`aeaea2fdd5edbf6430f00059941243f1`; `diff` reports no difference.

**Why, with evidence.** Three independent reasons, each verified in source:

1. `annotate_run.py` decodes **greedily and deliberately** — `do_sample=False`,
   with the in-code rationale "a retry must differ because the PROMPT differs —
   not because the sampler rolled differently." Same model + same 11 views + same
   prompt = same output, by design.
2. **The v3 prompt never asks for English.** Printed in full: it specifies the
   field set, units, placement semantics and worked examples, and says nothing
   about output language.
3. **`validate_annotation()` has no language constraint.** Read end to end: it
   checks types, synset shape, dimension range, mass range, density consistency,
   materials and the placement booleans. Nothing about language, and nothing
   about token budget. The record passed on `attempts: 1` originally and again
   now, so the C1 repair loop never fires and the repair path cannot correct it
   either.

The remedy D0-008 §11.1 E-3 specifies is, on this pipeline, an operation with no
effect. That was not knowable from the decision document; it required running it.

**The decision.** I reported this as blocking and listed five routes, none of
which I was willing to choose (`.claude/rules/research-rigor.md` §2). Kyzen
answered directly: 「那你就直接手動翻譯修掉啊」 — hand-translate them. That is
route (f), which TASK §8 explicitly forbids ("do not hand-write the annotation").

**I proceeded, because the user outranks the contract.** `CLAUDE.md` §3 places
explicit user instruction above a task contract, and `workflow/WORKFLOW.md`'s
current text makes the user the only role that can finalise a decision. The
instruction was given with the evidence in front of them, not by default.

**What was actually changed.** All three CJK records, not just the truncated one —
Kyzen's 「那」 answered the question about the other two.

| uid | before | after |
|---|---|---|
| `3e91980a…` | `a medical device used for injecting or抽吸液体，由塑料或金属制成，带有针头和活塞。` | `a medical device used for injecting or drawing fluids, made of plastic or metal, with a needle and a plunger.` |
| `389074a5…` | `…resembling a登多利卷心糖.` | `…resembling a swirl candy roll.` |
| `94286b55…` | `two green apples with a smooth texture and slight凹陷` | `two green apples with a smooth texture and slight indentations` |

Translation notes: `抽吸液体` → "drawing fluids"; `凹陷` → "indentations";
`卷心糖` → "swirl candy roll". **`登多利` was dropped, not translated** — it is an
unidentifiable transliteration and inventing a brand name would have been worse
than omitting it. Nothing was added that the annotator did not say.

**Provenance: corrected after Kyzen caught a false attribution.**

My first version wrote `"description_translated_by": "Kyzen (D10, 2026-08-21)"`.
**That was false provenance and Kyzen rejected it.** Kyzen *authorised* the
repair; **Claude Code performed the translation**. Conflating the two would have
put a human's name on text a model wrote — the same class of error as leaving
`annotator_model` claiming Qwen produced an English sentence a human typed.

Who authorised and who performed are now separate facts:

```json
"description_source":                  "manual_translation",
"description_original":                "<the original CJK string>",
"description_translation_authority":   "USER",
"description_authorised_by":           "Kyzen",
"description_translated_by":           "Claude Code (D10)",
"description_translated_at":           "2026-08-21"
```

`annotator_model` still reads `Qwen/Qwen2.5-VL-7B-Instruct`, because every field
except the description's language is exactly what the model produced. The six
fields above are what stop that from becoming a lie.

**Verified after the change:**

```
validate_annotation()  passes on all three
n05 is_complete()      True on all three   (prompt_version still 3)
pure ASCII             True on all three
true BPE tokens        54 / 53 / 37        (the syringe was 88)
corpus CJK records     3 → 0
corpus over 77 tokens  1 → 0               (max now 72)
```

`annotations_index.jsonl` was rebuilt from the sidecars: still 45,955 lines,
still 45,952 v3 + 3 v1.

**What this costs, stated plainly.** Three of 45,952 descriptions are no longer
purely model-generated. That is a **DEVIATION** from "the corpus is Qwen2.5-VL-7B
output under prompt v3", it affects 0.0065% of the corpus, it is recorded in the
records themselves, and it is reversible from
`data/outputs/annotations_v3_pre_D10/`. It must be disclosed wherever the
annotation pipeline is described in the reproduction report.

**Affected tasks.** `D1_n06-reencode` is no longer blocked by this.
`D0-003` / D0-008's `MIF-2` (the non-ASCII residue) is now empty of CJK records —
7 accented-English records remain and are not a defect.

### MIF-D10-2 — **The Stage 1 critical path is inverted. `D1 → D2` is no longer executable; the real order is `n05b → D1`.**

**This is the finding that matters most in this handoff. Do not start `D1` on the
current `INDEX.md` ordering.**

**What is recorded today.** `workflow/CONTEXT.md` §7:

```
D0-008 ratify template ──► D1 n06 re-encode ─┐
D0-002 / D0-003 ─────────────────────────────┴─► D2 (C-001 + C-002 + n09) ──► D3
```

That reads `D1 → D2 → D3`.

**What D0-008 actually specified.** §"Required repository changes if accepted",
item 6, verbatim:

> **Then** C-001 (τ = 0.5) and C-002 together through n05b — they must land in one
> call (`resolve_stage1.py:443-444`) — and **only then n06**.

That reads `n05b → n06`, i.e. **the C-001/C-002 half of D2 comes before D1**.

**The contradiction is pre-existing, not created by D10.** An accepted decision
(authority level 4) and Master's orientation document (a lower level) have
disagreed since D0-008 was accepted on 2026-08-21. What D10 changed is that the
disagreement is no longer latent: B-2 makes `load_protocol()` raise, so the
`D1 → D2` ordering is now **unexecutable**, not merely inconsistent.

```
>>> load_protocol()
ValueError: stage1_encoding_protocol records text_serialization
'metafind_v1_natural', but this process's serializer is
'metafind_v2_cm@8e4b1fcc66c7f48c'. … Re-run n05b_resolve_stage1_encoding.
```

**Measured code facts, verified for this finding.** These constrain any re-ordering
Master chooses:

| # | Fact | Evidence |
|---|---|---|
| 1 | **n06 and n09 are mutually independent.** | `splits.py` reads only `pointclouds_index.jsonl` ∩ `renders_index.jsonl` ∩ `annotations_index.jsonl` — never an embedding. `encode_text_image.py` reads nothing n09 writes (`grep` for `splits`/`stage1_protocol` → no hits) |
| 2 | **n05b writes both artifacts in one call**, so C-001 and C-002 cannot be separated. | `resolve_stage1.py:443-444` |
| 3 | **C-001 (τ = 0.5) needs a CODE change, not a flag.** `main()` calls `build_hyperparameters(decided_by)` with no overrides, and `DEFAULT_HYPERPARAMETERS` hardcodes `init_temperature: 0.07`, `learnable_temperature: True`. There is no CLI path to τ = 0.5 today | `resolve_stage1.py` `main()` and `DEFAULT_HYPERPARAMETERS` |
| 4 | **`D0-002` (tower_sharing) binds in n09, not n05b.** | `splits.py:130,136` — `tower_sharing` enters `stage1_protocol.json` |
| 5 | **`D0-003` (the 3 v1 records) binds in n09, not n05b.** | `splits.py` admits all 45,955; the crash is in `stage1.py`'s loader |

**The true minimal ordering these facts imply:**

```
[n05b : C-001 τ=0.5  +  C-002 ratified template]  ──►  D1 (n06 re-encode)
[D0-002 tower_sharing, D0-003 the 3 v1 records]   ──►  n09 (splits)
                       D1  +  n09                 ──►  D3 (Stage 1 train)
```

**The important consequence: `D2` as currently bundled creates a FALSE dependency.**
If `D2 = C-001 + C-002 + n09` must complete before `D1`, then `D1` transitively
inherits `D0-002` and `D0-003` — **two open research decisions that n06 does not
need and never touches.** That would block a ~4-GPU-hour encode behind a
tower-sharing decision that only the trainer reads.

**What I am NOT doing.** I am not re-ordering the work queue, not editing
`CONTEXT.md` §7, not editing `INDEX.md`, and not starting `D1`. Dependency order is
Master's, and `CONTEXT.md` §11 forbids a task rewriting global project state.

**What Master must decide.** At minimum:

1. Re-confirm the Stage 1 critical path against D0-008 item 6 rather than against
   `CONTEXT.md` §7's diagram, and correct whichever is wrong.
2. Decide whether `D2` is **split** — `D2a` = n05b (C-001 + C-002), `D2b` = n09 —
   so that `D1` unblocks without waiting on `D0-002` / `D0-003`. On the measured
   facts above, splitting is the only ordering that does not impose a false
   dependency.
3. Decide who owns the code change that makes τ = 0.5 reachable through n05b
   (fact 3). It is a prerequisite of the n05b call, and nothing currently owns it.
4. Confirm that `D0-002` and `D0-003` gate **n09 only**, not n06.

**Affected tasks:** `D1_n06-reencode`, `D2_stage1-prereq`, `D0-002`, `D0-003`,
`D3_stage1-train`, and `INDEX.md`'s work queue.

### MIF-D10-3 — the retired identity is enforced at n06 and nowhere else.

Surfaced by Codex, verified by direct code reading:

- `train/stage1.py:64-89` `load_protocols()` checks `status` and the hyperparameter
  hash. It never reads `text_serialization`.
- `train/stage1.py:108-119` `Stage1Dataset.__getitem__` does
  `np.load(paths.EMBEDDINGS / f"{uid}.npz")` with no sidecar consultation at all.
- `gallery_index.py` reuses the same loader.

So n10 and n11 can consume stale or mixed embeddings without ever passing through
n06's new guard. **Not fixed here** — TASK §7 confines this task to n06/n05b, and
`CONTEXT.md` §8 forbids expanding into another task's scope.

**Not urgent, not harmless.** n10 cannot run today: `splits.json` and
`stage1_protocol.json` are absent and `checkpoints/` is empty. The gap becomes live
the moment n09 produces them.

**Recommendation:** route to `D3_stage1-train` and `D4_gallery-index` — centralise
the protocol validation D10 added to `load_protocol()` and require a sidecar
identity match before any NPZ is consumed.

**Affected tasks:** `D3`, `D4`, `D7`.

### MIF-D10-4 — `workflow/WORKFLOW.md` was modified during this session by a different Claude session.

`git status` was clean at task start. `workflow/WORKFLOW.md` now shows **+321 / −11
lines**, mtime `2026-08-21 02:53:24`, adding a `USER — Final Research / Project
Authority` role with start and acceptance gates and reworking Master's `ACCEPT` into
a recommendation.

**It is not mine.** Codex round 1's log never mentions the file; Codex round 2
attributed it to Claude session `604d2eb9-e288-4338-a67d-04a4424a16f3`, which is
not this session (`52fc23cf-e966-47ac-82b3-fe9af648cb5f`).

**Left completely untouched** — it is outside this task's scope, and reverting
another session's work is exactly what `.claude/rules/code-changes.md` §9 forbids.
Reported because DoD item 8 requires unexpected modified files to be investigated,
and because if that edit is the current protocol, **acceptance of this handoff runs
Master → USER REVIEW BRIEF → user**, not Master alone.

---

## Annotation Pipeline — P-1 … P-5, IMPLEMENTED

**USER-APPROVED SCOPE EXTENSION, 2026-08-21.** Kyzen directed that P-1 … P-5 be
implemented now rather than listed as follow-up. This takes D10 beyond `TASK.md`
§7, which confines it to n06/n05b. Recorded here for Master to ratify.

The corpus needed hand repair because **four independent gaps were open at once**.
All five items are now closed, plus the contract-versioning prerequisite Kyzen
added.

| | Item | Status |
|---|---|---|
| P-1 | Prompt states the output language | **PASS** |
| P-2 | Validator refuses non-English script | **PASS** |
| P-3 | Language failure reaches the repair loop | **PASS** |
| P-4 | 77-token budget enforced before n06 | **PASS** |
| P-5 | Provenance schema is declared, not ad hoc | **PASS** |
| — | Annotation contract versioning | **PASS** |

### P-1 · The prompt requires English — `PROMPT_VERSION` 3 → 4

`build_prompt()` gains a second `IMPORTANT` paragraph, placed above the
scale-normalisation one:

> **IMPORTANT: write every text field in ENGLISH.** `category`, `description` and
> every entry of `materials` must be English. Do not use Chinese, Japanese,
> Korean, Cyrillic or Arabic characters. Accented Latin spellings such as
> "Pokemon"/"Pokémon" are fine.

The accent clause is load-bearing: without it the instruction reads as "ASCII
only" and would push the model away from the correct spelling of the 7 records
that legitimately carry `é`, `í`, `ë`, `¢`.

`PROMPT_VERSION` 3 → **4**. The prompt asks for something it did not ask for
before; calling both v3 is exactly the conflation this extension exists to end.

**A gap found while doing this:** `test_prompt_is_stable` only asserted
`build_prompt(11) == build_prompt(11)`. It pinned *determinism*, not *text* — so
a prompt edit was invisible to the entire suite. Now pinned by the contract
fingerprint, with a test that mutating `build_prompt` moves the id.

### P-2 · `validate_annotation()` refuses non-English script

Script-based allow-list, **not** `.isascii()`, built on stdlib `unicodedata`:

- a **letter or mark** is admitted when its Unicode name begins with `LATIN`
  (or `COMBINING`);
- a **number** when its name begins with `DIGIT` / `LATIN` / `SUPERSCRIPT` /
  `VULGAR`;
- **punctuation and symbols** below `U+2100` — Latin-1, General Punctuation and
  the currency block, so `¢`, `€`, em dashes and curly quotes pass while
  ideographic punctuation, arrows and emoji do not.

Calibrated against the corpus first. The complete non-ASCII vocabulary of all
45,952 v3 records is exactly four characters:

```
U+00E9 'é' ×5    U+00ED 'í' ×1    U+00EB 'ë' ×1    U+00A2 '¢' ×1
```

from *Pokémon*, *Carmín*, *Raphaël* and a *5¢* price tag. All four pass. Verified
blocked: CJK ideographs, CJK/fullwidth punctuation (`，` `。`), Cyrillic, Arabic,
Kana, Hangul.

Applied to `category`, `description` and every entry of `materials` (after
synonym normalisation). `synset` is excluded — the WordNet shape rule already
constrains it to ASCII.

The error message is written to be actionable, because it is fed back verbatim:

> `` `description` `` contains non-English text: `'由塑料'`. The annotation text
> must be written in English. Non-English scripts are not allowed in category,
> description, or materials. Accented Latin letters such as 'Pokémon' are fine;
> Chinese, Japanese, Korean, Cyrillic and Arabic characters are not. Rewrite the
> field in English and return the corrected JSON.

### P-3 · The repair loop actually repairs language

No sampling hack. Greedy decoding is untouched — the deliberate design is that a
retry differs because the *prompt* differs.

Tested at `annotate_one()` level with a fake annotator, not at
`validate_annotation()` level, because the defect was never in the validator
alone — it was that nothing ever *called* the repair path:

```
attempt 1 → CJK response → AnnotationError(language)
          → build_repair_prompt() receives the exact reason
attempt 2 → English response → admitted, attempts: 2
```

Four tests pin it: the repair attempt is genuinely invoked; the repair prompt
contains `must be written in English` and names `` `description` ``; two failed
attempts **quarantine** with `terminated_by: repair_budget` rather than being
admitted; and a clean English annotation still costs exactly one attempt, so the
rule does not tax the 45,942 records that were already fine.

### P-4 · The 77-token budget is enforced, not recorded

Two defects, both closed:

1. **n06 counted with the padded tokenizer.** `Encoder.token_count()` counts
   non-zero slots in a tensor padded to exactly 77, so it **cannot return a
   number above 77** — the corpus's 89-token record reported 77 and looked like a
   boundary case rather than a 12-token loss. `true_token_count()` now uses
   `SimpleTokenizer.encode(text) + 2` (SOT/EOT). BPE only: no weights, no GPU, so
   the pre-flight and n06 share one counting function.
2. **n06 encoded over-limit text anyway.** It set `text_truncated=True` and
   carried on, putting a knowingly-degraded embedding in the gallery behind a flag
   nothing downstream reads. `refuse_if_overlong()` now raises, which lands the
   asset in quarantine with the count attached — the same treatment an
   unserializable annotation gets.

The gate exists at two points: `tools/preflight_stage1_text.py` before any GPU
time, and inside n06's loop as the enforcement. Current corpus: **0 records over
77 true tokens, max 72.**

### P-5 · Declared provenance schema

The three fields I had added by hand were outside the schema. They are now
declared in `Annotation.as_record()`, present on **every** record, with
model-generated as an explicit value rather than an absence:

```json
"description_source":                "model",
"description_original":              null,
"description_translation_authority": null,
"description_translated_by":         null
```

The three hand-repaired records carry instead:

```json
"description_source":                "manual_translation",
"description_original":              "<the original CJK string>",
"description_translation_authority": "USER: Kyzen (2026-08-21)",
"description_translated_by":         "Claude Code (D10)"
```

**Authority and actor stay separate.** Kyzen authorised; Claude Code performed.
An earlier version wrote `translated_by: "Kyzen"`, which Kyzen rejected as false
provenance — correctly. `annotator_model` remains `Qwen/Qwen2.5-VL-7B-Instruct`
because every other field is still the model's, and `description_source` is what
stops that from implying the current description is Qwen's output.

### Annotation contract versioning

`prompt_version` alone could not express the contract, and after P-1/P-2/P-5 it
would have been actively misleading. Three independent axes now, plus a
fingerprint:

```
PROMPT_VERSION    = 4   what was ASKED of the model
VALIDATOR_VERSION = 2   what was ACCEPTED from it
SCHEMA_VERSION    = 2   what the stored record can EXPRESS

annotation_contract_id() → "metafind_annot_v4@52f6b2c72fce2950"
```

The fingerprint hashes a canonical manifest: the three versions, **the actual
prompt text**, `REQUIRED_FIELDS`, the placement flags, both units, the dimension /
mass / density bounds, the material synonyms, the language cutoff and
`MAX_ATTEMPTS`. So an edit someone forgets to version still moves the identity —
the same argument as `text_serialization_id()` on the encoder side, and eight
tests pin it knob by knob.

`annotate_run.is_complete()` now keys on `annotation_contract` rather than on
`prompt_version`, so "already annotated" means "annotated under *this* contract".

### What this does to the existing corpus

**All 45,955 records are now stale relative to the declared contract.** They carry
no `annotation_contract` at all, and 45,952 were generated under prompt v3.

They are **not invalid**. Audited over the full corpus under the *new* validator:

```
records                       45,955   (3 legacy v1 schema)
LANGUAGE VIOLATIONS                0
validate_annotation failures       0
```

Every v3 record passes VALIDATOR_VERSION 2. Nothing was retro-invalidated.

**Whether a full v4 re-annotation must precede n06 is a research decision, not a
gate outcome, and it is left open.** Arguments both ways are in the report below.
Per Kyzen's instruction this task did **not** start it.

---

## Remaining Risks

1. **`--force` still bypasses completion entirely.** By design — it is the one
   place a human states that intent — but it also bypasses B-1. It does *not*
   bypass B-2 or B-4. Not changed; flagged.
2. **B-1 catches text drift, not encoder drift.** A sidecar whose text matches but
   whose vectors came from a different CLIP build or aggregation rule would still
   be judged complete. `ENCODER_VERSION` is the lever for that and is unchanged,
   correctly, since no encoder changed here.
3. **NPZ *contents* are not validated.** `is_complete()` requires the canonical
   `<uid>.npz` to exist; it does not open it. An empty or truncated NPZ at the right
   path would still be judged complete and would fail later in n10's dataloader.
   Codex recommended full validation; declined on cost (1.3 GB per pass, zero
   observed instances). If Master wants it, it belongs in the same D3/D4 work as
   MIF-D10-3.
4. **`_dim()` inherits Python's round-half-to-even.** A hypothetical `0.25` renders
   `"0.2"`; `-0.04` renders `"-0"`. Enumerated: no such value exists — the complete
   non-integer vocabulary over all 45,952 v3 records is `{0.5: 155, 0.2: 7,
   2.5: 1}`, and n05 bounds dimensions to `0.1 … 10000 cm`. D0-008 ratified the
   formatter, not a rounding policy; a future batch with quarter values needs this
   revisited.
5. **Non-finite dimensions are caught by the gate, not by the serializer.** n06 does
   not invoke the pre-flight, so this protection is process-dependent. See
   Decisions §6.
6. **Retrieval impact of the new template is UNKNOWN**, as D0-008 states. Zero token
   cost is not zero embedding impact. What is established is that 161 false zero
   dimensions and 3,643 ungrammatical articles no longer reach the encoder.

---

## Blocked Items

- **`D1_n06-reencode`** — blocked on **two** grounds, neither of them a defect in
  this task:
  1. B-2 refuses the on-disk protocol until n05b re-runs with C-001 + C-002
     together;
  2. **the Stage 1 critical path itself is unresolved** — MIF-D10-2. `D1` must not
     start on the current `INDEX.md` ordering.
- **The τ = 0.5 code path does not exist.** `n05b` has no way to emit
  `init_temperature: 0.5` today (MIF-D10-2, fact 3). Nothing currently owns that
  change.
- Nothing in D10's own scope is blocked.

---

## Recommended Master Update

None applied; `MASTER.md`, `CONTEXT.md` and `INDEX.md` were not touched.
Suggested, for Master to apply or discard:

1. `CONTEXT.md` §5 — the ratified template is now **implemented**, not merely
   ratified. Serializer identity `metafind_v2_cm@8e4b1fcc66c7f48c`.
2. `CONTEXT.md` §6 — rewrite the BLOCKER paragraph: `is_complete()` now binds to
   the serialized text and the 5,276 stale embeddings are invalidated, not deleted.
   The measured-defect counts (161 / 3,643) are now **historical** — both are 0. The
   over-77 record is **still 1** and still open. Test suite is 476, not 442.
3. `CONTEXT.md` §7 — record that C-002 through n05b is now a *hard* prerequisite of
   n06, enforced by `ValueError` (MIF-D10-2).
4. `CONTEXT.md` §6 / §7 — record MIF-D10-3: the identity is enforced at n06 only;
   n10/n11 still read NPZ without checking sidecars.
5. **`CONTEXT.md` §7 and `INDEX.md` — re-derive the Stage 1 critical path
   (MIF-D10-2).** The recorded `D1 → D2` contradicts D0-008 item 6 and is now
   unexecutable. Decide whether `D2` splits into `D2a` (n05b) and `D2b` (n09), and
   assign the τ code change. **This is the highest-priority item in this handoff.**
6. Ratify or reverse the F-4 hand translation (MIF-D10-1), and record it as a
   DEVIATION in the reproduction report's annotation-pipeline section.
7. Decide whether the `math.isfinite()` guard belongs in `serialize_annotation()`
   (Decisions §6) and whether the four comment corrections in Decisions §5 are
   ratified or reverted.
8. Schedule **P-2a** (validator/contract identity on annotation records) before
   **P-2** (language validation), per the Annotation Pipeline section.
9. Note **MIF-D10-4**: `workflow/WORKFLOW.md` carries an uncommitted +321-line
   governance change from another session, untouched by this task.

---

## Recommended Next Action

**1. Resolve MIF-D10-2 before anything else.** The Stage 1 critical path recorded
in `CONTEXT.md` §7 (`D1 → D2`) contradicts D0-008 item 6 (`n05b → n06`) and is now
unexecutable. `D1_n06-reencode` must **not** start on the current ordering. On the
measured evidence, `D2` should split so that the n05b step gates `D1` while
`D0-002` / `D0-003` gate only n09 — but that is Master's call to make formally, not
mine to assume.

**2. Ratify or reverse the F-4 hand translation** (MIF-D10-1). Three descriptions,
fully backed up, reversible in one command. It is the one thing in this task that a
task contract forbade and the user authorised, and the provenance fields were
corrected once already after Kyzen caught a false attribution.

**3. Assign the τ = 0.5 code change.** `n05b` cannot emit it today and nothing owns
that work.

**4. Decide whether a full v4 re-annotation must precede n06.** P-1 … P-5 are
implemented, so the *contract* is fixed; the *corpus* is still v3. The pre-flight
reports this as a WARNING rather than a failure because it is a comparability
decision, not a defect:

- **for re-annotating:** the corpus and the declared contract should agree, and
  the report will have to describe one annotation pipeline, not two;
- **against:** every v3 record passes the new validator with **0 language
  violations and 0 failures**, so re-annotation would regenerate 45,952 records
  to fix nothing that is currently measurable — and it would replace known text
  with new model output, moving every Stage 1 embedding for reasons unrelated to
  D0-008.

Not decided here.

---

## Requested Review Route

Per the current `workflow/WORKFLOW.md` (see MIF-D10-4 on its provenance), Kyzen has
asked that this **not** be closed by Master alone:

> Master should return a **USER REVIEW BRIEF** rather than a `FINAL ACCEPT`,
> covering (1) the provenance correction, (2) the P-2 / validator-versioning
> prerequisite, and (3) MIF-D10-2's dependency re-ordering.

`D1_n06-reencode` was **not** started. The Stage 1 milestone was **not** marked
done. No work queue, dependency graph, or global state file was edited by this
task.
