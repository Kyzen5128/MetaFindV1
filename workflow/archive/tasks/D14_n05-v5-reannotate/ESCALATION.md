# ESCALATION TO MASTER — D14_n05-v5-reannotate

**From:** D14 executor · **Date:** 2026-08-21 · **Task status:** `ACTIVE`
**git HEAD:** `468bbac999d0c064fbc1bd098910f82b980dd18e`

Filed under `WORKFLOW.md` §18.2. **This is a request for permission and triage, not a
notification.** Six paths D14 is forbidden to touch now carry statements that are either
false or that contradict a user decision. D14 has edited none of them.

Companion documents in this directory:
`USER_DIRECTIVES.md` (what the user instructed, 560 lines) ·
`MASTER_IMPACTING_FINDING.md` (F-1 `/mnt/data1`, F-2 GPT-4o, F-3 the download).

---

# PART 1 — What the user authorised

Recorded verbatim. Full detail in `USER_DIRECTIVES.md`.

| id | The user's words | Effect |
|---|---|---|
| U-1 | 「等等先停下不要馬上做」 | D14 held before starting |
| U-2 | 「模型設定我們先不要定」 | Re-opened the model question that §7 had closed |
| U-3 | 「先下載Qwen3.8-27B…我要載到/mnt/data1/kyzen/」 | Weights acquisition |
| U-4 | 「幫我設定好這台以後用這個hf-token」 | HF token installed, 3 paths, chmod 600 |
| U-5 | 「下載給我指令 我來跑 你先中斷」 | Download handed to the user |
| **U-6** | **「走本地 Qwen3.8-27B」** | **MODEL DECIDED. Contradicts `TASK.md` §7** |
| **U-7** | **「這個路徑是我新裝的硬碟我後續要將檔案也搬過去」** | **`/mnt/data1` sanctioned; migration intended** |
| U-8 | 「不用換」 | HF token not rotated; risk accepted |
| **U-10** | **「我允許你 現在將標註這流程完全處理修好 讓05能夠產出正確的標註」** | **Phase 1 + Phase 2 authorised** |

### What U-10 does NOT authorise

**Phase 3 — the 19.6-GPU-hour, whole-corpus, irreversible re-annotation.**

`TASK.md` R-A makes that gate absolute, and Phase 2 exists precisely to produce the numbers
that decision needs. "修好標註流程" is authorisation to build and validate, not to spend the
run. **D14 will stop after Phase 2 and ask.** Master should not read U-10 as a Phase 3 go.

---

# PART 2 — Permissions D14 needs

Every path below is protected by `TASK.md` §9.2 or is Master's. **D14 has touched none of them.**
Ordered by whether it blocks.

## P-1 — `TASK.md` §7 Explicit Non-Scope · BLOCKING · Master's call

§7 reads:

> ❌ **Do not change the annotation model.** GPT-4o is unavailable; deviation `D-2` stands.

U-6 directs a model change. **The contract and the user's instruction contradict each other**,
and `WORKFLOW.md` §13B makes the user's instruction govern. But D14 has now *enacted* that
change at `annotate_run.py:71` under §9.1's allowance to edit that file — so the repository
currently does something §7 forbids.

**Requested:** Master amends §7 to record the model change as authorised by U-6, and re-words
it so the deviation is described as **re-pointed, not discharged.**

**Also, for the record:** the premise §7 gives for the prohibition — "GPT-4o is unavailable" —
is **not established**. See `MASTER_IMPACTING_FINDING.md` F-2: OpenAI's own deprecation page
does not list base `gpt-4o`, and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**.
Secondary sources contradict it. **The user declined to resolve the conflict, which is not the
same as the conflict being resolved.** §7's stated reason should not survive unqualified.

## P-2 — the `D-2` deviation record is now false · BLOCKING PROVENANCE · Master's call

`docs/graph/graph_spec.yaml:129-131` is the authoritative numbering:

```yaml
- id: D-2
  what: "Qwen2.5-VL replaces GPT-4o for annotation and scene judging"
```

Two problems, and the second is the serious one.

1. **It is factually wrong for annotation** as of this task. n05 now runs Qwen3.8-27B.
2. **D-2 bundles two nodes that no longer share a model.** Scene judging (n17) is untouched
   and still Qwen2.5-VL. One deviation id now needs to name two different substitutes.

`tools/check_graph.py:373-383` only verifies that the deviation **ids** match across
documents — **it never checks the text.** So this wrong statement passes every gate silently.
That is the same defect class as the retired `"metafind_v1_natural"` label.

The same sentence is repeated in files D14 also cannot touch:

```
README.md:60                      | D-2 | Qwen2.5-VL 取代 GPT-4o（資產標註與場景評分）|
docs/graph/02_BUILD_STEPS.md:22   same table
docs/graph/02_BUILD_STEPS.md:116  [偏離 D-2] Qwen2.5-VL 取代 GPT-4o（§2.3 明寫 GPT-4o）
docs/graph/02_BUILD_STEPS.md:772  scene scoring — this one stays correct
docs/graph/01_GRAPH_SPEC.md       same id
```

**Requested:** Master decides whether D-2 is **split** (annotation vs scene judging) or
**re-worded to name both substitutes**, then updates `graph_spec.yaml` and the three
documents. **D14 recommends splitting** — one id covering two models is how a record stops
being checkable — but this is a numbering decision that belongs to whoever owns the spec.

**D14 will not write up Phase 2 or Phase 3 while the deviation record says something untrue.**

## P-3 — `docs/graph/README.md` test counts · BLOCKS DoD 13 · one-line fix

`tools/check_graph.py` now reports **1 FAILURE**:

```
README unit-test count: README says 435, tests/ defines 456
```

`docs/graph/README.md:270` also says "展開成 547 個 case"; the suite now runs **582**.

**Verified this did not pre-exist.** Restoring the v4 test file returns `2275 checks, all
pass`. **D14 caused it by adding the tests that `DoD 13` also requires** — the item cannot be
satisfied and obeyed at the same time. `docs/**` is protected outright by §9.2.

**Requested:** permission to change two integers in `docs/graph/README.md`
(`435` → `456`, `547` → `582`), or have Master do it. Nothing else in that file needs to move.

## P-4 — `CLAUDE.md` §9 and `workflow/CONTEXT.md` §9 · NOT BLOCKING · Master → User

Both state:

> Paths under `/mnt/data1` belong to the previous machine and must not be assumed valid.

**Measured 2026-08-21:** `/dev/sda1`, 3.6 TB ext4, mounted at `/mnt/data1`, 3.4 TB free,
`lost+found` created `Aug 20 16:08`. `/mnt/data1/kyzen` is owned by `kyzen` and writable.
The user confirmed (U-7) it is a **new disk** and that **project files will be migrated there.**

`CLAUDE.md` is project instruction; D14 cannot edit it, and neither can it edit `CONTEXT.md`.

**Requested:** correct both, and register the intended migration as its own task.
**D14's engineering objection, for the record:** a migration moves `data_root`, and
`.claude/rules/experiments.md` §5 makes that a change of experimental condition. Moving 413 GB
while a 20–45 h annotation job reads from it is an avoidable failure. **The migration should
not overlap D14 Phase 3.**

## P-5 — `workflow/DECISION_LEDGER.md` `DL-003` · NOT YET · needed before Phase 3 closes

`TASK.md` R-D requires the provenance registry to stop describing the corpus as
`accepted_legacy_v3` the moment Phase 3 rewrites it. `DECISION_LEDGER.md` is Master's.

**Requested:** Master prepares the `DL-003` / AC-1 amendment now, so it lands in the same
breath as Phase 3 rather than after it. D14 will run
`tools/declare_annotation_provenance.py` and prove afterwards that a bare run still queues 0
and `--force` still works — but the ledger sentence is Master's to write.

---

# PART 3 — Everything D14 changed

All inside `TASK.md` §9.1 ALLOWED. `git status` confirms nothing else moved.

| Path | What | §9.1 line |
|---|---|---|
| `metafind/data/annotate.py` | v5 prompt, validator v3, versions 5/3/3, synset lookup, `category_relation`, `derive_dimensions` | "prompt, validator, versions, contract, synset table" |
| `metafind/data/annotate_run.py` | LVIS anchor + proportions passed through; `MODEL_ID`; generic model class | "passing the LVIS category and proportions through" |
| `tests/test_annotate.py` | 113 → 148 tests; every v5 rule covered | "coverage for every v5 rule" |
| `metafind/data/lvis_synsets.json` | **new.** 1,156 entries + provenance | "a synset lookup table under … `metafind/data/`" |
| `workflow/tasks/D14_n05-v5-reannotate/*.md` | this file, `USER_DIRECTIVES.md`, `MASTER_IMPACTING_FINDING.md` | "`HANDOFF.md`, `CODEX_REVIEW.md`, `TASK.md` status line" |
| `TASK.md` | Status line `READY` → `ACTIVE` only | the one edit §9.1 permits |

Outside the repository, on user instruction: the Qwen3.8-27B download under
`/mnt/data1/kyzen/models/`, and the HF token at three `chmod 600` paths.

### Contract identity moved

```
PROMPT_VERSION 4 -> 5 · VALIDATOR_VERSION 2 -> 3 · SCHEMA_VERSION 2 -> 3
metafind_annot_v5@f5b2bfb2e5f61fe7
```

### Verification done

```
pytest tests/ -q            582 passed
tools/check_graph.py        2275 checks, 1 FAILURE  (P-3 above, and only that)
LVIS coverage               45,952 / 45,952 = 100.00%   verified per uid by D14
Y-up axis                   reproduced independently: tall n=1,365 [x .543 y .946 z .474],
                            flat n=962 [x .882 y .372 z .716].  height = y
synset table                1,156 / 1,156 from LVIS v1, cross-checked against detectron2,
                            0 disagreements, 0 invented entries.  Escalation NOT required
```

### Two IMPLEMENTATION CHOICES needing user approval

**IC-1 — `divergent` categories are recorded, not rejected.** §7 says lateral replacement is
invalid, but `motor vehicle → pickup truck` (which the design calls **required**) and
`motor vehicle → coffee machine` (forbidden) cannot be separated without a hypernym judgement
no source in this project supplies. D14 computes `exact` / `refined` / `divergent`, stores it,
and rejects nothing — the same reasoning as Design Decision 2, and §12 requires those three
counts anyway. **This is `TASK.md` §14's second Codex attack point and Codex must be aimed at it.**

**IC-2 — `synset` follows the LVIS anchor, not the model's refined `category`.** No
authoritative synset exists for "toy dinosaur"; minting one restores the error class the
lookup was adopted to remove.

---

# PART 4 — Still blocked on evidence, not on permission

The Qwen3.8-27B download is at **9 / 18 shards**. Until it finishes D14 cannot measure:

- whether the model loads on 32,607 MiB — **it will not at bf16 (55.6 GB); quantization is
  required and has not been done or tested**
- VRAM peak with 11 images · seconds per asset · whether the quantized model emits parseable JSON

**No Phase 3 runtime estimate is currently evidence-backed.** The 20–45 h figure is an
`INFERENCE` and could be materially worse.

---

# PART 5 — What Master is asked to do

1. **P-1** amend `TASK.md` §7 for U-6, and qualify the "GPT-4o is unavailable" premise (F-2).
2. **P-2** decide how `D-2` is split or re-worded, then fix `graph_spec.yaml`, `README.md`,
   `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`. **Blocks Phase 2 write-up.**
3. **P-3** grant two integers in `docs/graph/README.md`, or make the edit. **Blocks DoD 13.**
4. **P-4** correct `CLAUDE.md` §9 and `CONTEXT.md` §9; register the `/mnt/data1` migration as
   its own task, sequenced clear of D14 Phase 3.
5. **P-5** prepare the `DL-003` / AC-1 amendment for Phase 3.
6. **Rule on IC-1 and IC-2**, or route them to the user.
7. **Confirm that U-10 is not a Phase 3 go.** D14 reads it as Phase 1 + Phase 2 only.

---

## Standing honesty statements

No re-render. n06 / n09 / training / gallery indexing never invoked. The 3 legacy-v1
residuals are byte-identical and untouched — **`D0-003` remains UNRESOLVED**. The 20,053
halted embeddings are untouched; none deleted. `checkpoints/` empty. The annotation corpus is
unchanged, md5 `30b2737c95152043762ce25fcabe7a0e`. `docs/`, `README.md` and every Master file
untouched. Codex review has not run — **that is not a PASS**. LVIS anchoring and the
annotation model are both recorded **DEVIATIONS**. **No claim of paper fidelity is made.**
