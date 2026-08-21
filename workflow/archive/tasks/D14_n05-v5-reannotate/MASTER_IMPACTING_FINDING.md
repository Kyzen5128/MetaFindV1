# MASTER-IMPACTING FINDING — D14_n05-v5-reannotate

**Raised by:** D14 executor, 2026-08-21.
**Task status at time of writing:** `READY` (unchanged — Phase 1 has **not** started; no code, prompt, validator, or annotation record has been modified).
**Trigger:** the user re-opened the annotation-model question, which `TASK.md` §7 currently places in Explicit Non-Scope.

Reported per `WORKFLOW.md` §18.2. This file lives inside the task directory. Master's files
(`MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md`, `decisions/**`) are untouched.

**Findings and decisions are stated separately (`WORKFLOW.md` §13A). No decision is enacted here.**

---

## Provenance

```
git HEAD            468bbac999d0c064fbc1bd098910f82b980dd18e
working tree        workflow/INDEX.md, workflow/MASTER.md modified (Master's, pre-existing)
                    D14 task dir untracked
date                2026-08-21
host                RTX 5090, 32,607 MiB; torch 2.12.1+cu132; transformers 5.15.0
python              /home/kyzen/miniconda3/envs/MetaFind/bin/python
```

---

# F-1 — `/mnt/data1` exists on this machine and is a mounted 3.6 TB volume

## Finding

`CLAUDE.md` §9 states:

> Paths under `/mnt/data1` belong to the previous machine and must not be assumed valid.

**That statement is stale.** `/mnt/data1` is present, mounted, and writable on the current host.

## Evidence — OBSERVED DATA, measured 2026-08-21

```
$ lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
sda           3.6T
└─sda1        3.6T ext4     /mnt/data1

$ df -h /mnt/data1
/dev/sda1       3.6T  2.1M  3.4T   1% /mnt/data1

$ ls -la /mnt/data1
drwxr-xr-x 4 root  root  4096 Aug 20 16:16 .
drwxr-xr-x 2 kyzen kyzen 4096 Aug 20 16:16 kyzen
drwx------ 2 root  root 16384 Aug 20 16:08 lost+found

write test on /mnt/data1/kyzen   -> WRITABLE
```

The `lost+found` mtime (`Aug 20 16:08`) indicates the filesystem was created on **2026-08-20** — i.e. this
is a **new disk on the current machine**, not a survival of the previous machine's path.

## Evidence class

`OBSERVED DATA` (mount state, capacity, ownership, writability).
`INFERENCE` only as to *why* the doc is stale (a new disk was added after `CLAUDE.md` §9 was written).

## Affected

- `CLAUDE.md` §9 — a project-instruction statement that is now factually wrong.
- `CONTEXT.md` §9 Runtime / Environment Facts — carries the same claim.
- Any future task that treats a `/mnt/data1` path as evidence of a stale machine-specific path.

## Current task impact

None on scientific behaviour. It affects **where large artifacts may be stored**: 3.4 TB free on
`/mnt/data1` versus 423 GB free on `/` (which holds `/home/kyzen/data/MetaFind`, currently 413 GB).

## Can the current task continue safely

**Yes.** No project data lives on `/mnt/data1`.

## FINDING / DECISION separation

```
FINDING:   /mnt/data1 is a mounted, writable 3.6 TB ext4 volume on this host,
           created 2026-08-20.                       [OBSERVED DATA] [CONFIRMED]
DECISION:  Whether CLAUDE.md §9 and CONTEXT.md §9 should be corrected, and whether
           /mnt/data1 becomes a sanctioned location for model weights or project data.
           [proposed by: D14 executor] [requires user approval: YES — CLAUDE.md is
           project instruction, and CONTEXT.md is Master's file]
```

**D14 has not edited either file.**

---

# F-2 — the "GPT-4o is unavailable" premise underpinning deviation `D-2` may be false, and the paper-era snapshot has a 63-day window

## Finding

`n05_v5_design.md` and `TASK.md` §7 both rest on:

> "Do not change the annotation model. **GPT-4o is unavailable**; deviation `D-2` stands."

OpenAI's **official** API deprecations page does **not** list the base `gpt-4o` alias in any
deprecation table as of 2026-08-21, and lists this row:

| Shutdown date | Model snapshot | Substitute model |
|---|---|---|
| **2026-10-23** | `gpt-4o-2024-05-13` | `gpt-5.6-sol` |

If `gpt-4o` is reachable, the reproduction has a **63-day window** (2026-08-21 → 2026-10-23) in which the
paper's stated annotation model — or at minimum a GPT-4o-family snapshot — is still callable.

## Evidence

| Source | Says | Class |
|---|---|---|
| `2methdology.tex:28` | "annotated using **GPT-4o**" | PAPER FACT (already established by Master) |
| `neurips_2025.tex:100` | "processed with **GPT-4o**" | PAPER FACT (already established by Master) |
| OpenAI `developers.openai.com/api/docs/deprecations` | base `gpt-4o` absent from all deprecation tables; `gpt-4o-2024-05-13` shutdown **2026-10-23** | UPSTREAM FACT (vendor documentation), **retrieved via web, not exercised against the API** |
| Multiple secondary blogs | claim GPT-4o was removed from the **API** on 2026-02-16 | **secondary sources — not authoritative under `research-rigor.md` §5** |
| Local environment | **no OpenAI API key present** (`env` scan, names only) | OBSERVED DATA |

### ⚠ UNRESOLVED CONFLICT

The vendor deprecation page and the secondary sources **disagree** about whether base `gpt-4o` is still
served by the API. Per `research-rigor.md` §4 this conflict is **left explicit and not resolved by
D14**. It is resolvable in ~5 minutes and for a few cents by issuing one authenticated request — which
requires an API key this environment does not have.

## Cost of the paper-faithful route — ESTIMATE, from measured token counts

Measured on this corpus (not assumed):

```
assets                       45,952
views per asset                  11  (224x224 RGB, verified by PIL)
v4 prompt                       625 tokens   (Qwen tokenizer, build_prompt(11))
model output                    111 tokens   mean over 399 real annotations (p95 123)
v5 prompt assumed              ~700 tokens   (+ LVIS category and proportions)
v5 output assumed              ~120 tokens   (+ identity_confirmed)
```

Applying published list prices (`$2.50` / `$10.00` per M for `gpt-4o`):

| Configuration | Cost |
|---|---|
| `detail=low` (85 tok/img) + Batch API −50% | **US$121** |
| `detail=high` (255 tok/img) + Batch API −50% | US$229 |
| `detail=low`, realtime | US$243 |
| `detail=high`, realtime | US$458 |

**Both the image-token formula and the price table are taken from web sources and are UNVERIFIED
against a vendor invoice.** The token counts for prompt and output are **measured**.

For comparison, the local route (Qwen-class 27B on the existing RTX 5090) costs approximately
**NT$70–160 of electricity** (14–31 kWh at ~700 W wall over an estimated 20–45 h). The 20–45 h figure is
an **INFERENCE** — no 27B model has been benchmarked on this host with 11-image inputs.

## Evidence class

```
The paper specifies GPT-4o                            PAPER FACT
Base gpt-4o absent from OpenAI's deprecation tables    UPSTREAM FACT (vendor doc, web-retrieved)
gpt-4o-2024-05-13 shuts down 2026-10-23                UPSTREAM FACT (vendor doc, web-retrieved)
gpt-4o is actually callable today                      UNKNOWN — not exercised
Which GPT-4o snapshot MetaFind used                    UNKNOWN — the paper does not state it
Corpus token counts (625 / 111)                        OBSERVED DATA
API dollar cost                                        INFERENCE from unverified list prices
Local runtime 20-45 h for a 27B model                  INFERENCE — unbenchmarked
```

**Note for the record:** even if `gpt-4o` is reachable, MetaFind never names the snapshot. Selecting one
is an `IMPLEMENTATION CHOICE`, and every non-GPT-4o option remains a `DEVIATION`. Reaching GPT-4o would
**narrow** `D-2`, not automatically discharge it.

## Affected

- `n05_v5_design.md` — "No model change. GPT-4o is unavailable; deviation `D-2` stands."
- `TASK.md` §7 Explicit Non-Scope — "Do not change the annotation model."
- Deviation `D-2` (`annotate_run.py:71`).
- `TASK.md` §8 **R-E** — the DEVIATION disclosure. Its wording changes if the model changes.
- Table 1 / Table 2 comparability with the paper's reported results.
- The ~19.6 GPU-h budget assumption, and D14's Phase 3 execution plan.

## Current task impact

**Phase 1 has not started, so nothing is invalidated.** But the annotation model is an input to
`build_prompt`, the validator, `PROMPT_VERSION`, and the contract fingerprint. Implementing v5 against
Qwen and *then* switching models would mean rewriting Phase 1 and re-running Phase 2.

## Can the current task continue safely

**Not past Phase 1 design, no.** Per `WORKFLOW.md` §18.3 the stop-safe condition applies: continuing
would build an approved design around a model the user is actively reconsidering, and would risk
spending Phase 2 (and possibly Phase 3's 19.6 GPU-h) on a configuration chosen under a premise that
may be false.

**D14 is therefore holding at `READY`.**

## Recommended action — PROPOSED, not enacted

```
FINDING:   OpenAI's official deprecation documentation does not list base gpt-4o as
           deprecated, and schedules gpt-4o-2024-05-13 for shutdown on 2026-10-23.
           Secondary sources contradict this. The premise "GPT-4o is unavailable"
           is therefore not established.
                                    [UPSTREAM FACT + UNRESOLVED CONFLICT] [PLAUSIBLE]

DECISION:  (a) Resolve the conflict empirically with one authenticated API call
               before D14 Phase 1 begins.
           (b) If gpt-4o is reachable: decide whether the annotation model changes,
               which requires amending TASK.md §7 Explicit Non-Scope and re-wording
               R-E's DEVIATION statement.
           (c) If it is not reachable: record the conflict as resolved-by-observation
               and let D-2 stand as designed.
           [proposed by: D14 executor] [requires user approval: YES — this is a
            paper-fidelity decision and a task-contract change. WORKFLOW.md §13B
            makes it material.]
```

**Cheap disambiguation available:** Phase 2's stratified 300–500 sample can be run against more than one
model. At the measured token rates that is roughly **US$1–3** of API spend per model. This measures the
choice instead of arguing it, and it is already inside D14's approved Phase 2 scope — *except* that
using a non-Qwen model requires the §7 amendment above.

---

# F-3 — action taken: Qwen3.8-27B download (user-instructed)

## What was done

On the user's explicit instruction, the official weights `Qwen/Qwen3.8-27B` are being downloaded to
`/mnt/data1/kyzen/models/Qwen3.8-27B`.

```
repo            Qwen/Qwen3.8-27B     (Apache 2.0, released 2026-08-14)
architecture    Qwen3_5ForConditionalGeneration
weights         55.6 GB across 18 safetensors shards (32 files total)
destination     /mnt/data1/kyzen/models/Qwen3.8-27B     (outside the repo, outside data/outputs)
log             /mnt/data1/kyzen/models/qwen38_download.log
command         huggingface_hub.snapshot_download(repo_id=..., local_dir=..., max_workers=8)
```

**Nothing under `data/outputs/`, `metafind/`, or `tests/` was written. No protected path was touched.**

## Two facts the user should have

1. **55.6 GB bf16 does not fit in 32,607 MiB of VRAM.** Published estimates put bf16 inference at
   ~56–68 GB. Running this model on this host **requires quantization**. Third-party quantized repos
   exist (`huginnfork/Qwen3.8-27B-NVFP4A16` at 31.0 GB, `-FP8` at 38.5 GB) but are **not vendor-published**
   and introduce a provenance question of their own.
2. **`transformers 5.15.0` supports the architecture.** Verified directly on this host:
   `Qwen3_5ForConditionalGeneration`, `Qwen3VLForConditionalGeneration`,
   `Qwen3VLMoeForConditionalGeneration`, and `InternVLForConditionalGeneration` all resolve.
   `[OBSERVED IMPLEMENTATION]`

## Evidence class

`OBSERVED DATA` (repo size, architecture, transformers support, destination path).
`INFERENCE` (VRAM requirement, from vendor-adjacent web sources, not measured on this host).

---

# Summary for Master triage

| # | Finding | Class | Status | Blocks D14? |
|---|---|---|---|---|
| F-1 | `/mnt/data1` exists, mounted, 3.4 TB free, writable | OBSERVED DATA | CONFIRMED | No |
| F-2 | "GPT-4o is unavailable" is not established; vendor doc vs secondary sources conflict; paper-era snapshot dies 2026-10-23 | UPSTREAM FACT + UNRESOLVED CONFLICT | PLAUSIBLE | **Yes — holding at READY** |
| F-3 | Qwen3.8-27B downloaded on user instruction; bf16 will not fit 32 GB VRAM | OBSERVED DATA | CONFIRMED | No |

## What D14 has NOT done

- `TASK.md` Status remains `READY`. Phase 1 has not begun.
- No re-render. No n06, n09, training, or gallery indexing.
- The 3 legacy-v1 residuals untouched; **`D0-003` remains UNRESOLVED**.
- The 20,053 halted embeddings untouched; none deleted.
- No annotation record read for mutation, written, or backed up.
- `annotate.py`, `annotate_run.py`, the prompt, the validator, and all version constants unchanged.
- No Master file edited.
- Design Decisions 1–4 not re-litigated. **F-2 is new evidence about a stated premise, not a preference.**

## What Master must decide

1. Correct `CLAUDE.md` §9 / `CONTEXT.md` §9 regarding `/mnt/data1`? (F-1)
2. Authorise one authenticated `gpt-4o` API call to resolve the F-2 conflict?
3. If GPT-4o is reachable — amend `TASK.md` §7 to permit a model change, and re-word R-E?
4. Does D14 resume Phase 1 on Qwen, or hold until the model question is settled?

**D14 is holding and awaits triage.**
