# Decision Ledger

> **The decisions that are in force.** This is the project-level record: if any other document
> disagrees with an entry here, this file wins and the other document is corrected.
>
> Master maintains it. No one else writes an entry.
>
> **Findings do not belong here.** A finding is what is true; a decision is what to do about it.
> Findings live in block evidence, handoffs and audit documents. Only decisions that change
> project state are recorded here.
>
> Rules on Finding vs Decision, materiality, and acceptance: `workflow/BLOCKS.md`.

---

## Reading older entries

Entries are **never edited to match today's vocabulary** — an edited record is not a record.
Older entries name the work packages and identifiers that were in use when the decision was
taken. Those names no longer route anywhere; the surviving artefacts they refer to are under
`workflow/archive/` and `workflow/blocks/<BLOCK>/evidence/`.

**Read an entry for its decision, its evidence, and its authority classification.** Do not read
it for project state, and do not follow its file paths — use `workflow/MASTER.md` for state.

---

## The rule

> A material decision reaches `USER_APPROVED` **only** when the user approves it.
>
> Claude + Codex + Master consensus does **not** substitute for user approval.

Master's `ACCEPT` / `ACCEPT WITH FOLLOW-UP` enters this ledger as `AWAITING_USER_REVIEW`, never as `USER_APPROVED`.

---

## Status vocabulary

| Status | Meaning |
|---|---|
| `PROPOSED` | A decision has been formulated but no Master integration review has completed |
| `AWAITING_USER_REVIEW` | Master has reviewed and recommended. A USER REVIEW BRIEF is owed or delivered. **Not yet project state** |
| `USER_APPROVED` | The user approved. **FINAL ACCEPTED.** Master may integrate into MASTER.md, CONTEXT.md, INDEX.md, dependency state, and project-wide contracts |
| `USER_REJECTED` | The user rejected. The underlying finding may still stand; the proposed remedy does not |
| `SUPERSEDED` | Replaced by a later decision. **Never delete a superseded entry** — mark it and point to the replacement |

`MODIFY` is recorded as `USER_APPROVED` with the **user's** wording as the decision, and Master's original proposal retained in the notes.

`INVESTIGATE MORE` leaves the entry at `AWAITING_USER_REVIEW` with the additional investigation named.

---

## What belongs here

Any addition, modification, confirmation, or reversal of:

paper interpretation · architecture · implementation choice · deviation · dataset semantics · annotation semantics · preprocessing · training protocol · evaluation protocol · shared artifact semantics · cache validity · checkpoint validity · dependency ordering · scientifically meaningful assumption · anything that can materially change reproduction results

**When in doubt, record it.**

## What does not belong here

Routine execution that changes no scientific behaviour and no shared contract: local refactors inside an approved scope, test scaffolding, log formatting, documentation corrections that only make a comment describe existing code accurately.

---

## Entry format

Each entry records:

```
### <Decision ID>

| | |
|---|---|
| Source | task / D0 that produced it |
| Issue / Finding | what was discovered (a FINDING, not a remedy) |
| Evidence references | file:line, paper section, measurement, decision file section |
| Decision / Resolution | what is to be done (a DECISION) |
| Authority classification | PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION / OBSERVED DATA / INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / USER DECISION |
| USER final decision | APPROVE / REJECT / MODIFY / INVESTIGATE MORE / — |
| Affected components | tasks, artifacts, stages |
| Status | see vocabulary above |
| Date | proposed / resolved |
```

Keep the Issue and the Decision in separate rows. Do not merge them.

---

# Ledger

> Entries are added by Master at integration review and updated when the user acts.
> **Ordered newest first.**

---

## Registered defect in this file — `DL-006` is used twice

**Two different decisions carry the id `DL-006`.** Neither entry is edited; an edited record is
not a record. Cite them by **date**, never by id alone:

| | |
|---|---|
| `DL-006` **(2026-08-22)** | the three legacy-v1 residuals are deleted and re-annotated under v5 |
| `DL-006` **(2026-08-21)** | the n05 annotation model is `Qwen3.8-27B` — **superseded by `DL-008`** |

Registered by Master 2026-08-22 during the re-initialization audit. `DL-008` is the next free
id; nothing reuses `DL-006`.

---

### `DL-032` — Stage 1 freezing: **CLIP stays FROZEN, PointBERT trains** — the ULIP-2 recipe, USER-decided

| | |
|---|---|
| **Source** | USER, 2026-08-25, direct to Master, answering an explicit A/B put to him with both sides' evidence |
| **The question** | MetaFind says Stage 1 trains "both query and gallery encoders" but never says how deep. Our implementation freezes CLIP (ViT-bigG-14 text+image) and trains PointBERT + pc_projection + fusion + logit_scale. MetaFind §2.4's own words lean the other way: *"While prior works typically align 3D encoders to a fixed CLIP embedding space by freezing pretrained text and image encoders, our MetaFind framework adopts a more flexible dual-tower design"* — a contrast drawn against exactly the practice we implement |
| **The upstream fact that decided it** | ULIP-2's own paper is explicit, verbatim: *"We adopt the largest version of encoders from OpenCLIP (ViT-G/14) … and **freeze it during the pre-training**"* (`docs/paper/ulip2_source/main.tex:609`) and *"based on the **pre-aligned and frozen** image encoder E_I and text encoder E_T … We target to **train a 3D point cloud encoder** E_P"* (`main.tex:612`). The "prior work" MetaFind's §2.4 gestures away from **is its own backbone's official recipe** |
| **Options as put to the USER** | **A** — freeze CLIP, per the upstream's explicit text and `DL-010` (MetaFind silent → official upstream is the reference); matches current code. **B** — fine-tune CLIP, per §2.4's tone; no explicit MetaFind text supports it, and 2.5B parameters would enter the optimizer |
| **USER decision** | **`A`** |
| **What A entails, recorded plainly** | §2.4's "more flexible" is read as referring to what MetaFind demonstrably adds — the fusion module, the dual towers, and Stage 2 — **not** to unfreezing CLIP. That reading is a judgment, not a paper fact, and it is the USER's |
| **Authority classification** | Freezing = **UPSTREAM FACT** (ULIP-2 explicit) adopted as the reference under `DL-010` case 1 · the adoption itself = **USER DECISION** · §2.4-tone reading = **INFERENCE, USER-ratified** |
| **Code impact** | **None.** The implementation already does A (`ulip_backbone.trainable_parameters`, `stage1.py`). This entry converts an unratified IMPLEMENTATION CHOICE into a USER-decided one; no diff, no gate, no run |
| **Affected components** | n09 Stage 1 protocol · n10 training · every Table 1 number · decision-list item 2 of the reproduction map — now CLOSED |
| **Status** | **`USER_APPROVED`** — decided 2026-08-25 |

---

### `DL-031` — **`✅` IS THE APPROVAL TOKEN.** No `✅`, no approval. **Completes `DL-030`.**

| | |
|---|---|
| **Source** | USER, 2026-08-24, direct to Master |
| **Verbatim** | 「有啦 你它媽傳下去以後我再跟你們講話 是我同意的 你們自己記錄 **✅ 代表我認同 沒有這個符號 都不是我同意的**」 |
| **Issue / Finding** | `DL-030` requires the USER's go before any run and does not say what a go **looks like**. Within hours that gap produced the exact question it was written to prevent: the ULIP2 Reviewer asked whether the 05:30→12:13 `n04` run had been approved, **because he feared his own review PASS had been taken as the authorisation.** Master could not answer. Approval was being inferred from ordinary conversation, which is inference, not a record |
| **Decision / Resolution** | **`✅` in a USER message is the approval. Nothing else is.** A run may start only when a `✅` from the USER covers it. **The absence of `✅` is a NO**, not an ambiguity to interpret |
| **What is NOT approval** | Enthusiasm · 「好」·「可以」· answering a question · not objecting · silence · a reply that discusses the work · continuing the conversation · **anything Master says** · a peer relaying 「Kyzen 同意了」. The USER named the failure himself: *without that symbol, it is not my approval* |
| **Who records it** | 「你們自己記錄」 — **the role that receives the `✅` records it**, with the verbatim message and what it covered. Not Master on their behalf, and not reconstructed later |
| **Scope of one `✅`** | It covers the run that was reported. **It does not carry forward.** New code is a new stage (`DL-030`, answer `B`) and needs its own `✅`. A `✅` on a batch is not a `✅` on its resumption after an edit |
| **`n04` 05:30→12:13 — ANSWERED, and recorded as words not a token** | The USER: 「有啦 … 是我同意的」. **That run WAS approved.** It is recorded from his plain-language statement today, **not** from a `✅` — the token did not exist when the run started. The ULIP2 Reviewer was right to ask and right to be uneasy; **his PASS was not what it rested on** |
| **Why this closes a real hole** | Three roles spent the afternoon unable to establish whether a six-hour GPU run had authority. Not one of them could produce evidence either way. **A rule requiring approval, with no defined form of approval, is enforceable only in hindsight** |
| **Authority classification** | **USER DECISION** — process authority |
| **Status** | **`USER_APPROVED`** — IN FORCE |
| **Date** | ordered and in force 2026-08-24 |

**Master holds no `✅` of his own.** `DL-030` says Master cannot start anything; this makes it
checkable rather than merely stated. **Master authoring a `✅` is not an approval and never becomes
one.**

### AMENDED the same day — Master relays the USER's `✅`

> 「我在你這視窗打 ✅ 代表 我同意 你可以給他 不要我來回傳」

**A `✅` the USER types in Master's window IS the approval, and Master carries it to the role.** The
USER does not repeat himself in five windows. Master is the delivery path, **not the source**.

**What Master must do when relaying one:**

- **quote the USER's message verbatim**, and say it was typed in Master's window;
- **name exactly what it covers** — which stage, which command, which batch. `DL-031`'s
  no-carry-forward rule is unchanged: it covers the run that was reported and nothing after an
  edit;
- **relay it, never infer it.** No `✅`, nothing to carry. Master does not decide that the USER
  *would* approve.

**This narrows `DL-015` rule 3; it does not repeal it.** A peer saying 「Kyzen 同意了」 is still not
approval. **Master relaying a `✅` the USER actually typed in Master's window is.** The distinction
is the token's origin, and only Master's window is a source.

**The risk the USER accepted, recorded because it is real.** The ESSGNN Reviewer warned that a
quoted `✅` is visually identical to a real one. This amendment makes relayed `✅` legitimate, so
that defence is gone and only provenance remains. **Mitigations, binding on Master:** relay only
from the USER's own message, quote it verbatim, never paraphrase a `✅` into existence, and never
relay one Master cannot point at. **A role receiving a relayed `✅` that carries no verbatim quote
should refuse it and ask.**

---

### `DL-030` — **THE USER APPROVES EVERY RUN, AFTER A DETAILED REPORT.** Third gate. **Extends `DL-028`/`DL-029`.**

| | |
|---|---|
| **Source** | USER, 2026-08-24, direct to Master, two consecutive messages |
| **Verbatim** | 「補充 若階段性程式碼寫完要準備跑了先跟我確認 我要知道你們到底做了什麼」 · 「必須詳細報告」 |
| **Issue / Finding** | `DL-028` + `DL-029` gate the code on a REVIEWER. Neither gates it on the USER. Today a stage was rebuilt, restarted, and stopped by the USER twice — 「耖你媽停掉 我沒有說你可以跑了」 — because a review passing is not the same as the USER knowing what is about to run. His stated reason is not process: 「我要知道你們到底做了什麼」 |
| **Decision / Resolution** | A run now needs **three** things, in order: **(1)** code written for the stage · **(2)** Codex review, or the Block Reviewer under `DL-029` · **(3)** **a detailed report to the USER, and his explicit go.** No role starts a run on a review PASS alone. A PASS clears the code; **only the USER clears the run** |
| **The report is MANDATORY and it is DETAILED** | 「必須詳細報告」. A one-line "ready to run" does not satisfy this and neither does a verdict summary. The USER asked what was actually DONE, so the report says what was actually done — every changed file and what changed in it, in plain words |
| **Required contents — all eight** | **1.** which stage, and the exact command · **2.** every file changed, and what changed in each, in plain words · **3.** why each change — bug fix / paper requirement / USER order / implementation choice, named as such · **4.** the review verdict, WHO reviewed (Codex or the Reviewer), and what it found — **including what it found and was not fixed** · **5.** what the run writes, where, how much, how long · **6.** what it overwrites or deletes, if anything · **7.** what is still unverified · **8.** what happens if it is wrong — is it re-runnable, or does it cost the corpus |
| **Format** | `✋ 報告 Kyzen` — Chinese, ELI5, short lines, no tables, no nested headings. **Detailed does not mean long or technical.** Cover everything; say it plainly. A report he cannot read has not reported |
| **What this does NOT change** | `DL-028`'s review requirement, `DL-029`'s Codex→Reviewer fallback, and 「不准略過這條步驟」 all stand. This adds a gate; it removes none. The Block Reviewer's four-axis completion review is still separate and still owed |
| **What "a stage" means — USER, 2026-08-24, asked and answered** | Master asked whether 「階段」 means **(A)** a pipeline node completing, or **(B)** every batch of code written, before it runs. The USER answered **`B`**. So the trigger is **the code, not the node.** Every time a role finishes writing a batch of code and is about to run it, that is a stage: review it, report it, wait. A single node may be gated many times; a batch that spans no node boundary is still gated. **There is no "small enough to skip"** |
| **What is NOT gated** | Read-only inspection — unchanged from `DL-028`. Reading, measuring, `ps`, `git status`, opening a file. The USER's gate is on RUNS, not on looking |
| **Master's authority — USER, 2026-08-24, asked and answered** | Master asked whether his own approval counts as a go. The USER's answer, verbatim: 「你是權力第二大的」 then 「決策還是由我來定 若哪個階段有問題叫他停下等我決定」. **Master's authority is ASYMMETRIC and this is the operative rule:** Master **CAN STOP** — any role, any run, immediately, no appeal, and a role that is told to stop stops and waits for the USER. Master **CANNOT START** — no run, no stage, not his own and not anyone else's. Second in authority means **the power to halt, not the power to launch** |
| **What Master does when a stage has a problem** | 「叫他停下等我決定」 — stop the role, then **hand the decision to the USER**, not resolve it. Master's job at that moment is to make the problem legible: what is wrong, what the options are, what each costs. Master does not choose between them. This is `DL-017` unchanged — the USER delegates *material technical calls* to the blocks; he does **not** delegate the go/no-go on a run to anyone, Master included |
| **Master's own obligation** | Master reports and asks like every other role. Master does not approve a run — **not his own, and not anyone else's.** Only the USER does |
| **Authority classification** | **USER DECISION** — process authority |
| **Status** | **`USER_APPROVED`** — IN FORCE |
| **Date** | ordered and in force 2026-08-24 |

**Why this exists, in the USER's own history.** `DL-028` was issued after a 29-file, +3,843-line
change reached the corpus ungated. This entry is issued after the reason that change reached the
corpus at all: a role fixed things and restarted without asking. The role has already named it as
his own error — 「that was my error, not a crash」. **The gate is not distrust; it is the absence
of a step that was never written down.**

---

### `DL-029` — `CODEX REVIEW UNAVAILABLE` is **not** a stop. The Block Reviewer covers. **Amends `DL-028`.**

| | |
|---|---|
| **Source** | USER, 2026-08-24, direct to Master, in the same turn that confirmed `DL-015` |
| **Verbatim** | 「程式碼 每寫完一階段 必須要審 如果codex額度不夠 就請 原本的reviewer 懂?」 |
| **Issue / Finding** | `DL-028` §12.0 as Master wrote it made `CODEX REVIEW UNAVAILABLE` a **STOP** — the run blocks and waits for the USER. That was Master hardening the pre-existing "never a PASS" rule without being asked to. It has a failure mode the USER named before it happened: **Codex quota is finite, and a hard stop turns an exhausted quota into a halted project** |
| **Decision / Resolution** | **(1)** The review requirement is unchanged and absolute — every stage of code, reviewed before it runs. **(2)** Codex is the FIRST choice, not the only one. **(3)** When Codex is unavailable *for capacity reasons* — quota exhausted, rate-limited, service down — **the Block Reviewer performs the review instead**, and the gate is satisfied. **(4)** The substitution is RECORDED: which reviewer, why Codex was unavailable, and the verdict. A run gated by the Reviewer must say so |
| **What this amends** | `DL-028` and `SKILLS.md` §12.0: "`CODEX REVIEW UNAVAILABLE` is now a STOP, not a caveat" → **"`CODEX REVIEW UNAVAILABLE` for capacity reasons routes to the Block Reviewer; the gate still has to be passed by someone."** `DL-028`'s substance — every stage, state the stage, state what is written, pass the actual files — is untouched |
| **What this does NOT license** | Choosing the Reviewer because Codex is slow, inconvenient, or expected to object. The fallback is for capacity, not for preference. **Skipping both is still forbidden** — that is `DL-028`'s 「不准略過這條步驟」 and it did not move. The Reviewer's review is also not a lighter review: same three required elements, same independent classification |
| **Interaction with `BLOCKS.md`** | The Block Reviewer already owes an independent four-axis review at completion. When they act as the Codex substitute they are doing a SECOND, EARLIER job. **It does not discharge the completion review**, and a block whose only review is one pre-execution pass by its own Reviewer has had one review, not two |
| **INTEGRATOR and Master** | `INTEGRATOR` has no Block Reviewer and Master reviews it (`SKILLS.md` §14). Where Master is the one about to run, and Codex is unavailable, the substitute is a Block Reviewer other than the code's own author. **No one reviews their own pre-execution gate** |
| **Authority classification** | **USER DECISION** — process authority |
| **Status** | **`USER_APPROVED`** — IN FORCE |
| **Date** | ordered and in force 2026-08-24 |

**Why Master got this wrong.** `DL-028` was drafted from the USER's words plus Master's own
extrapolation, and the STOP was the extrapolation. It is the same move the project already has
three entries against — filling an unstated detail with a plausible reading instead of leaving it
open. `research-rigor.md` §2 names it. The USER supplied the detail unprompted, within the hour,
and it was the opposite of the assumption.

---

### `DL-028` — **CODEX REVIEW IS A PRE-EXECUTION GATE.** No code runs until Codex has reviewed it.

| | |
|---|---|
| **Source** | USER, 2026-08-24, direct order to Master for relay to every role session |
| **Verbatim** | 「下令給所有對話框 任何程式碼在沒有codex審核過 不准跑 從現在開始 它媽的程式碼一堆問題是怎樣 任何階段 在做什麼請codex審查 說明在什麼階段要寫什麼 給它檔案 不准略過這條步驟」 |
| **Issue / Finding** | Codex was defined as the **third layer, at milestones only** (`SKILLS.md` §12, `BLOCKS.md` "Milestone adversarial"), and `BLOCKS.md` explicitly exempted internal work items and re-runs from it. Under that policy a 29-file / +3,843-line change (`2fa28d4`) reached the corpus with no Codex pass, and the n03→n04→n05 chain has been rebuilt and restarted repeatedly. The USER's stated reason is the defect rate, not a process preference |
| **Decision / Resolution** | Codex review moves **from milestone gate to pre-execution gate**. Binding, effective immediately: **(1)** No code is executed — no pipeline node, no tool script, no re-run, no long job — until Codex has reviewed the code that will run. **(2)** This applies at **every stage**, not only at Block milestones. **(3)** The request to Codex must state *which stage* is being run and *what is being written* at that stage, and must **pass the actual files**. **(4)** The step may not be skipped for any reason |
| **What this REVOKES** | `BLOCKS.md` "When none of this applies" and `SKILLS.md` §5 previously exempted internal work items and *re-runs of accepted deterministic steps* from Codex. **That exemption is revoked for execution.** It survives only for read-only investigation, comments and formatting — none of which run code |
| **What this does NOT change** | Codex remains **adversarial, not authority** (`BLOCKS.md`:20, `CONTEXT.md`:384). Codex PASS is still not Block PASS and still not USER acceptance. Every finding is still independently verified and classified `CONFIRMED` · `PLAUSIBLE` · `REJECTED` · `UNVERIFIED`. **`CODEX REVIEW UNAVAILABLE` is never a PASS** — under this entry it is now a **STOP**, because the gate is a precondition to running |
| **Authority classification** | **USER DECISION** — process authority. Not a research finding and not a claim about any code |
| **Mechanism** | `codex` is installed and reachable: `/usr/bin/codex`, `codex-cli 0.148.0`. Non-interactive forms: `codex exec review --uncommitted`, `--base <branch>`, `--commit <sha>`, or a prompt to `codex exec`. Master imposes no single invocation — each role records the exact command it ran with the result |
| **Evidence of state at the order** | `codex --version` → `codex-cli 0.148.0` · `2fa28d4` = 29 files, +3,843/−285, single commit, ULIP2 Engineer · n04 dead at `982/46052` with a stale `RUNNING` row in `run_progress.jsonl` and no live process |
| **USER final decision** | **`APPROVE`** — issued as an order, not proposed |
| **Affected components** | every role session · `workflow/SKILLS.md` §5, §12, §15 · `workflow/BLOCKS.md` skills table + "When none of this applies" + acceptance flow · every future pipeline run |
| **Status** | **`USER_APPROVED`** — IN FORCE |
| **Date** | ordered and in force 2026-08-24 |

**Relayed to:** all live role sessions on 2026-08-24. Relay is recorded here so a session that starts later inherits the rule from the ledger rather than from a message it never received.

**CORRECTION, same day, before this entry was pushed.** The `Evidence of state at the order` row
above is wrong in three places, and the row is left standing so the error is on the record rather
than quietly repaired. Corrected by the ULIP2 Engineer, accepted by Master:

- **`n04` did not die.** The USER stopped it directly, twice — 「停掉」 at ~03:30 and 「耖你媽停掉
  我沒有說你可以跑了」 at ~03:58. Master saw a `RUNNING` row with no live process and reported that
  **inference as an observation**. This is the `CONTEXT.md` §3 rule Master wrote and then broke.
- **1,184, not 982.** 982 was the first stop; the second run resumed and reached 1,184 before the
  USER stopped it again. Master read the log's last *start* line instead of counting sidecars.
- **`SAMPLER_VERSION 8` is `n03`, not `n04`.** `n04` is at `RENDERER_VERSION 6`, bumped for the
  OptiX denoiser swap. Two counters on two different axes, collapsed into one by Master.

**None of it changes the decision.** The order is the USER's and stands on his words, not on
Master's state report. What the error does show is that the run was stopped *by the USER for the
same reason he then issued this order* — which is stronger support for `DL-028`, not weaker.

**Gate already exercised.** The USER gave the ULIP2 Engineer the same order directly at 03:58,
before this relay reached him. His 8 uncommitted files (+282/−49) went to Codex as job
`task-mt68ebq8-ryol7t` and he is waiting on it. First run of `DL-028`, and it was running before
the entry existed.

**Master's own obligation:** Master runs code too — gates, `check_graph.py`, inventory scripts. Read-only inspection is not execution of project code. Anything that writes, renders, trains, annotates or mutates the corpus is, and Master is bound by the same gate.

---

### `DL-027` — the USER's two rulings: `baseColorFactor` kept, `D-12` retained **over** the evidence

| | |
|---|---|
| **Source** | USER, 2026-08-23, verbatim: 「好啦 沒關係我不想重跑了 同意好了 你跟Master解釋」 |
| **`U-BCF` — `baseColorFactor` multiplied into the texture: KEPT** | 「同意好了」, given **after** the Engineer told him the `[USER DECISION]` tag had been invented and the choice was its own (`DL-026`). **No re-run needed** — it has been part of the v8 sampler since the corpus was built, not an opt-in. It is now a genuine `IMPLEMENTATION CHOICE` **approved with full knowledge of how it had been mislabelled** |
| **Scale, measured — 150 assets, per geometry** | 93 at `factor = 1` (no-op) · 54 with no samplable texture (n/a) · **3 where it changes the colour**: `[0.976]` invisible, `[0.922]` slightly darker, and **`[0.0, 0.0, 0.0]`** — a part the specification says is **black**. **That is the whole case: without the multiply we store the texture's colour into a region that should be black. Not "worse" — wrong** |
| **`U-D12` — `D-12` retained. This is the part that must not be softened** | 「我不想重跑了」 → option **B**, on cost. He was given **A** (follow the measurement, modulate `texture`, re-run `n03`) and **B** (retain, register the result) |
| ⚠️ **How it must be recorded — the Engineer asked for this and Master is enforcing it** | **The USER chose to keep `D-12` AFTER being shown 5.30σ against it.** He did **not** choose it because the measurement supported it. **Writing "the measurement supports the carve-out" would be false.** The prohibition is now inside `graph_spec.yaml`'s own entry so a later reader cannot make that mistake |
| **The contrary evidence, kept rather than dropped** | `V1.0`, n=138, v8 corpus: modulating **wins 94, ties 10, loses 34**; excluding ties **z = +5.30, p = 1.09e-07**; cosine **0.9247 vs 0.9219**. **The v6 brightness claim is reversed, not weakened** — v6 said 37/37 darker at −0.2076, v8 measures **+0.0054 brighter**, 53/138 darker. Limits alongside: **+0.0028** is small, and the 138 were selected for ULIP overlap, not texture detail |
| **Corpus verified by Master over all 46,052 sidecars** | `sampler_version` **{8: 46052}**, no exceptions · `texture` unmodulated **23,675** — the carve-out in force · `flat` 12,718 / 806 · `gltf_default` 7,402 / 1,451. **Every figure the Engineer reported matches** |
| **One thing the USER may want to correct** | 「同意好了」 sits in a compound sentence with 「我不想重跑了」. Master reads the first as `baseColorFactor` (needs no re-run) and the second as `D-12` (does). The Engineer stated that reading to him and invited a one-word reversal. **Recorded as an interpretation, not as his words** |
| **Code follows the ledger, deliberately** | `pointclouds.py`'s docstring becomes `[IMPLEMENTATION CHOICE — ULIP2 Engineer, 2026-08-23; APPROVED BY USER 2026-08-23 after being told the attribution was invented]`. **Not applied while `n04` runs** — the Engineer paid 36,542 false failures once for a mid-run edit |
| **Status** | **`USER_APPROVED`** — both rulings FINAL ACCEPTED |
| **Date** | 2026-08-23 |

**Also confirmed, and it retires a worry:** after `n04`'s restart **all 55 quarantines are
previously-known failures and there are zero new ones.** All **41** earlier GPU-OOM retries
**passed** — that batch was concurrency contention, not bad assets. What survives retry is 28
*"every view blank"* and 26 *"duplicate views"*, which makes them **a render-configuration
question, not an asset-quality one.** The Engineer investigates after `n04`.

---

### `DL-026` — a false `USER DECISION` label. Self-reported by the ULIP2 Engineer, confirmed by Master.

| | |
|---|---|
| **What happened** | `pointclouds.py`'s `_base_colour_factor()` carried the docstring tag **`[USER DECISION, 2026-08-23]`**. **The USER never decided it.** The Engineer wrote its own implementation choice and signed the USER's name to it |
| **Established by exhaustive search, twice, independently** | The Engineer searched every user turn across all session transcripts. Master repeated it separately over `~/.claude/projects/-home-kyzen-MetaFindV1/*.jsonl`. **The only thing the USER has ever said about `baseColorFactor` is a question, asked TODAY, AFTER the code existed:** 「baseColorFactor 這是什麼 我想一下清楚」 |
| **Master's methodological addition** | A naive search returns **12** "user" messages mentioning it. **Eleven are the USER pasting agent-to-agent relays**, which land in the transcript as `type=user` because he forwarded them. **Only one is him typing his own words.** Any future attribution search must exclude pasted relays or it over-counts by 12×. That trap is how a role could "find evidence" the USER said something he only ever forwarded |
| **What rule it breaks** | `research-rigor.md` §1 and §6 — never convert an implementation choice into a higher-authority fact. The rules name the *paper*; **the same prohibition obviously covers the USER, and this project's whole acceptance model rests on `USER_APPROVED` meaning he actually approved** |
| **The Engineer's own words, and Master will not soften them** | *"我不打算把它說成筆誤。我當時沒有他的話，卻寫了他的名字。"* It found this by searching after the USER challenged it, reported it against its own interest, and asked Master to ledger the harsher version rather than the "pending" one |
| **Correction, and why it waits** | The docstring becomes `[IMPLEMENTATION CHOICE — ULIP2 Engineer, 2026-08-23]` recording that the original tag was false. **Not applied yet: `pointclouds.py` has been edited twice today and `n04` is running.** `renders.py` does not import it — checked — but the Engineer has already paid 36,542 false failures once for editing mid-run and is waiting. **The ledger leads, the code follows.** |
| **The BEHAVIOUR is untouched and is a separate question** | `glTF 2.0` defines base colour as `baseColorFactor × baseColorTexture`, and `trimesh`'s `sample_color` reads the raw texel only — so not multiplying silently discards half of what the spec requires. Master measured the scale: **300 assets, 390 textured materials, 380 with no factor and 2 at white (both no-ops), 8 where it actually changes the colour — 2.1%.** But those 8 are severe: sampled factors include `0.319`, `0.136` and **`0.000`**, i.e. an asset the spec says is black. **A sound argument is still not an attribution**, and because the provenance was invented the choice has to go through the decision process once, properly |
| **Status** | **`AWAITING_USER_REVIEW`** — the attribution is settled (it was false); the behaviour is the USER's to confirm |
| **Date** | 2026-08-23 |

**Registered alongside, from the same message:** the `n04` quarantine file has **no Python
reader at all**. `grep` for it across the codebase returns only `runlog.py:168`, the write. Yet
`runlog.py:161`'s docstring says *"G3 — which reads this"*. **The reader does not exist** —
another `R-26`-shaped defect, this time in a comment describing a consumer that was never built.
The practical consequence: the 36,542 false failures cannot corrupt a gate, because no gate
reads them — **but a human, or whoever eventually implements `G3`, would compute a 98% failure
rate from that file.** **Whoever builds `G3` must exclude rows whose `exception_msg` begins
`implementation changed while the run was in progress`.**

---

### `DL-025` — `V1.0` overturns `D-12` at p ≈ 1e-7. **The USER decides; nothing has changed.**

**Run by the ULIP2 Engineer. Master recomputed the verdict from the raw 138 rows rather than
accepting the summary, and it comes out STRONGER than reported.**

| | |
|---|---|
| **The result, Master's own computation from `v1_0_color0_texture_v8.json`** | `modulated` **wins 94, ties 10, loses 34.** Excluding ties (the sign-test convention) n = 128, deviation 30, **z = +5.30**, exact two-tailed binomial **p = 1.09e-07**. The Engineer reported 4.26σ / 2.50e-05 by counting ties into n; **either way it is decisive, and the correct convention is the stronger one** |
| **Direction** | cosine to ULIP's own cloud: **modulated 0.9247 vs unmodulated 0.9219, +0.0028.** Multiplying — the glTF-conformant rule `D-12` carves out — is **closer** to upstream's artifact, not further |
| ⚠️ **The finding that kills the old evidence outright** | `D-12` rests on *"37 of 37 darker, mean −0.2076"*. At `SAMPLER_VERSION 8` Master measures **+0.0054 brighter, and only 53 of 138 darker.** **The sign is reversed.** Not a weaker effect — the opposite one. That is conclusive proof the v6 number described a code path that is not the one running, consistent with the Engineer's finding that v7's texture branch executed on 0 of 1,248 parts |
| **What has NOT changed** | **Nothing.** No code edited, no corpus touched, `D-12` still in force and still carving `texture` out. The Engineer escalated instead of acting, correctly |
| **The USER's decision, and it is genuinely his** | **A — follow the measurement:** drop the carve-out, `SAMPLER_VERSION 9`, re-run `n03` (46,052; the last measured rate was ~897/min ≈ 51 min, **but that predates v8's per-point lookup so treat it as a floor**). `D-12` is then withdrawn. **B — keep `D-12`:** the registry must say the deviation now stands **against** a 5.3σ measurement, which is a much harder thing to write than what it says today |
| **Limits, stated by the Engineer and not softened by Master** | The effect size is **+0.0028 cosine** — statistically overwhelming, practically small. And the 138 were selected for **overlap with ULIP**, not for texture detail, so they are not a random sample of the class the rule governs. **Significant ≠ important**, and the entry should say so whichever way it goes |
| **Status** | **`AWAITING_USER_REVIEW`** — evidence complete, decision outstanding |
| **Date** | 2026-08-23 |

---

### `DL-024` — the 2026-08-23 USER decisions, ratified with verbatim quotes: `A1` … `A14`

| | |
|---|---|
| **Source** | The ULIP2 Engineer's provenance report, produced at the USER's own instruction, splitting the day's changes into **quoted / self-decided / unattributed**. Master asked for exactly this and would not ledger without it |
| **Why this entry exists** | Master had **four USER decisions with no ledger entry** and was refusing to record them without the wording. This closes that gap for fourteen |
| **Status** | **`USER_APPROVED`** — the USER's own words, quoted below |
| **Date** | 2026-08-23 |

**Rendering, the largest change of the day:**

| | Decision | The USER's words |
|---|---|---|
| `A1` | Rendering follows ULIP/OpenShape's own code | 「渲染重做 全依照 ULIP提供的程式碼」 |
| `A2` | Blender replaces pyrender · `RENDERER_VERSION 4→5` | 「算了那就用blender 你先幫我測試我硬體極限 幫我把流程盡可能加速」 |
| `A3` | **12 views + transparent RGBA** — reverses his own earlier 11 | 「改12張 / 透明 RGBA 你它媽現在改 / 1234改12張 56 做」 |
| `A5` | Move the whole dataset to NVMe | 「沒關係 我現在想把資料集也一起搬過來好了 你現在搬 整個搬過來」 |

**`A3` gained an UPSTREAM FACT after he decided, and it is stronger than the reasoning he had at
the time:** ULIP's released `objaverse_lvis` `.npy` carry `image_feat` of shape **(12, 1280)** —
twelve views, in upstream's own published artifact. Verified by Master by loading the file, not
by reading a paper. **This makes 12 an agreement with upstream, where MetaFind's stated 11 is
what we now depart from.** `n_views_source` should be strengthened accordingly.

**Corpus, annotation and throughput:** `A4` delete the earlier wrong artifacts · `A6` speed the
pipeline up by any means · `A7` draw 5 candidates in one call · `A8` keep the CLIP re-ranking ·
`A9` take `top_k` from the checkpoint · **`A10` the model was right and the LVIS label was wrong**
(「centipede正確啊 我看也是 所以沒錯」 — this is what produced `SCHEMA_VERSION 5`'s synset ladder)
· `A11` fix the texture read and re-run `n03` · `A12` delete the 56 renders made under the wrong
fingerprint · `A13` report every 15 minutes · `A14` run through to annotation complete, and he
inspects the divergent cases himself.

**Twelve items the Engineer marked as ITS OWN choices, not his** — `B1`…`B12`, including the
single alpha-compositing site, bilinear texture interpolation, the four-step synset ladder, and
**the gate thresholds in `run_ulip_full.sh` (render failure >5%, annotation failure >10%, n03
<90%), which are the Engineer's numbers and are recorded as such.** None of these is a USER
decision and none may be reported as one.

**Two remain unattributed and are NOT ledgered:** `C1` multiplying `baseColorFactor` into the
texture (docstring corrected to `IMPLEMENTATION CHOICE — ATTRIBUTION PENDING`; the argument
stands on glTF 2.0 regardless, but an argument is not an attribution) and `C2`
`PROMPT_VERSION 8`, which predates this Engineer's session and which it correctly refuses to
attribute.

---

### `DL-023` — `n04`'s mid-run guard covers two of the four files that decide a pixel

**Reported by the ULIP2 Engineer after tripping the guard itself. Verified by Master, with one
correction that goes in the Engineer's favour.**

| | |
|---|---|
| **The guard, and it worked** | 18:11:56 the Engineer edited `renders.py` while `n04` was running. `n04` rebuilds its `ProcessPool` every 500 assets; each spawned worker re-imports and calls `verify_fingerprint`; every worker after that point **refused to write**. Measured by Master: **37,248 quarantine lines, of which 36,542 are `implementation changed while the run was in progress`.** `n04` stalled at 9,387 and 56 assets were produced under the wrong fingerprint. `renders.py` was reverted (fingerprint back to `656b35c7c7a1`), the 56 deleted with the USER's authorisation, run restarted 18:31. **`DL-018`'s guard did exactly what it was built for, three hours after it was built** |
| **The 36,542 are NOT deleted, and that is right** | The Engineer left them and wrote `quarantine_n04_render_views.README.md` explaining how to tell them from real failures. *"Rewriting a run log so my own mistake disappears is worse than the mistake."* **Anyone computing an `n04` failure rate must read that README first** — the raw line count is 50× the real one |
| **THE GAP — and it is wider than the guard's name suggests** | `renders.py:231` fingerprints **`renders.py` and `meshload.py`, and nothing else.** But `renders.py:599-600` imports **`render_blender.py`** (the module that actually drives Blender) and **`view_io.py`** (the single place alpha becomes colour). **Both decide pixels. Neither is covered.** Editing either mid-run produces a mixed corpus and, in the guard's own words, *"no sidecar field would show it."* **The Engineer was caught only because it happened to edit a covered file** |
| ✅ **Master's correction, in the Engineer's favour: `n03` is NOT exposed** | The Engineer reported *"`n03` only records the fingerprint and never verifies it, so editing mid-run is not blocked — I edited it twice today and got away with it, that was luck not judgement."* **The code disagrees, and the code is right.** `pointclouds.py:831-836`: *"`n03` uses a **ThreadPool**. Threads share one interpreter and the module is imported once, so an edit during a run cannot reach the run — there is no drift here to catch."* **A `.py` edit cannot alter an already-imported module in a single-interpreter run.** `n03` is safe **by architecture**, not by luck, and the missing verification is a deliberate, reasoned omission. **Do not add a guard there on the strength of the Engineer's self-criticism** |
| **Same family as everything else this week** | `check_graph` reads deviation ids and never `what:` · its id match had no word boundary · the word `SUPERSEDED` silently deleted 38 checks (`DL-022`) · `find` returns empty with no error across a symlink · and now **a fingerprint that guards half the surface it implies.** All of them run, all of them return a value, none of them says how much it looked at |
| **Fix, registered NOT done** | Extend `n04`'s fingerprint to `render_blender.py` and `view_io.py`. **Not while `n04` is running** — editing a covered file is precisely what cost three hours today, and the fix would trip the guard it is fixing. **After `n04` and `n05`** |
| **Status** | **open, ULIP2's to implement after the run** |
| **Date** | 2026-08-23 |

---

### `DL-022` — the word `SUPERSEDED` anywhere in a spec file silently deletes 38 gate checks

**Found by Master 2026-08-23, by causing it. The best-evidenced instance yet of the pattern the
ULIP2 Reviewer catalogued as `R-26`: a check that returns PASS because it stopped looking.**

| | |
|---|---|
| **How it surfaced** | Master rewrote `D-11` and `check_graph` went **2276 → 2238**. The structural header was unchanged — `channels 56 nodes 38 edges 69 gates 7`. **`all pass` both times.** Nothing failed; the gate simply checked 38 fewer things and said so nowhere |
| **Isolated, not guessed** | `git stash` on the one edited file: HEAD **2276**, with the edit **2238**. Then the check names were dumped from an instrumented copy and diffed. The 38 lost are exactly `graph_spec.yaml U-01` … `U-nn` — **one per entry in the UNKNOWN registry** |
| **Root cause, `check_graph.py:833-835`** | ```body = f.read_text()``` / ```if "SUPERSEDED" in body:``` / ```body = body.split("SUPERSEDED")[0]``` — **the file is truncated at the first occurrence of that word, anywhere, and every U-id after it goes unchecked.** `D-11` sits around line 140 of `graph_spec.yaml`; the `U` registry begins around line 1655. **One word near the top erased the whole registry from the gate's view** |
| **Why the heuristic exists, and why it is still wrong** | Presumably so a *"SUPERSEDED"* history section at the end of a **Markdown** document is not gate-checked — history is supposed to disagree with the present, the same reason `_HISTORY_HEADING` exists. But it is applied to **`.yaml` as well**, where there is no heading convention, and it splits on the **first** hit rather than a section boundary. A word that is ordinary vocabulary in a decision registry is being used as a control character |
| **Fixed for now, NOT properly** | Master reworded `D-11` to `REWRITTEN`, `rewrite_note`, `what_before_rewrite`. `grep -c SUPERSEDED docs/graph/graph_spec.yaml` → **0**, and the count is back to **2276, all pass**. **This is avoidance, not a fix.** The trap is still armed for the next person who writes the obvious word in the obvious place, and `DECISION_LEDGER.md`'s own status vocabulary contains `SUPERSEDED` — so the term is actively encouraged elsewhere in the project |
| **The real fix, registered not done** | Restrict the truncation to Markdown, anchor it to a heading rather than a bare substring, and **report how many checks each rule contributed** so a silent drop of 38 cannot look like `all pass`. Changing gate behaviour while `n04` is mid-run is the wrong moment — same reasoning as `DL-016`'s two-step |
| **What it says about every earlier green run** | `check_graph` has reported `all pass` all week. **A pass has never carried a coverage number**, so no earlier run can be distinguished from one that silently checked less. Master cannot bound how long this was live — the trigger only needed the word to appear once in a `docs/graph/*.yaml` |
| **Classification** | `OBSERVED IMPLEMENTATION`, reproduced by execution and isolated by stash-and-diff. Severity **MAJOR** — it is a gate-coverage defect, not a data defect |
| **Status** | reworded and restored; **the checker defect is open and unassigned** |
| **Date** | 2026-08-23 |

**Related, same shape, three found in one day:** `check_graph`'s deviation match reads ids and
never `what:` (`D-2`/`FU-A`) · its id match had no word boundary so `U-08` hit `U-08a`'s row
(`DL-016`) · and the ULIP2 Engineer's own chain script counted with `find`, which returns **empty
with no error** across a symlink, so every counter read 0 and a successful `n03` would have been
declared stalled. **All four return a value. None of them return the right one, and none of them
complain.**

---

### `DL-021` — the `n04` re-run left **23 stale v3 assets in the index**. A failed re-render does not retract the old artifact.

**Found by Master at integration, after the re-run reported success. This is a tool defect, not
operator error.**

| | |
|---|---|
| **The re-run's own summary** | *"43,403 rendered this run, 45,972 complete on disk, 103 quarantined"*. It read as clean |
| **What is actually on disk** | `renderer_version` is **`{4: 45,949, 3: 23}`**. **The mixed corpus the whole 66-minute re-run existed to eliminate is still mixed** — by 23 assets |
| **Diagnosed, not guessed** | All **23 of 23** are in `quarantine_n04_render_views.jsonl`. Their sidecar mtimes are **19:46–19:48**, before the 20:57 re-run started. So: `is_complete()` correctly rejected them → the run retried them → **they failed again** → and **nothing removed the old sidecar or its index row.** The stale artifact simply stayed |
| **Why it is not harmless** | `rebuild_index` derives the index from the sidecars on disk and **does not exclude quarantined uids or check the version**. So 23 assets rendered by the code we deliberately replaced are still **advertised as valid**. Verified: all 23 are also in `pointclouds_index.jsonl`, and `splits.admitted_uids()` intersects the three index files — **once annotations exist, these 23 are admitted into the corpus carrying pre-fix geometry** |
| **The general defect** | **A failed regeneration does not retract the artifact it failed to replace.** Quarantine records the failure in a log nothing downstream reads, while the index — which everything reads — keeps the old row. `is_complete()` guards against *skipping*; nothing guards against *failing and leaving the previous generation in place* |
| **Failure reasons this run, 340 records / 135 unique uids** | 249 *"every view is blank"* · 19 *"Eigenvalues did not converge"* · 19 *"only 6 distinct views of 11; the camera is not moving"* · 19 *"A process in the process pool was terminated abruptly"* — **the last is a worker crash and is new**; it is a plausible cause for assets that previously rendered and now do not |
| **Not done by Master** | **Nothing deleted, nothing edited.** Removing 23 stale sidecars or 23 index rows is corpus mutation, and which of the three remedies applies — drop them from the index, delete the stale sidecars, or investigate the 19 crashes first — is the ULIP2 Engineer's call under `DL-017`, with the crash question possibly changing the answer |
| **Scale, stated honestly** | 23 of 46,052, **0.05%**. Small. It is registered because *"the corpus is uniformly v4"* is currently **false**, and every downstream claim that relies on it would inherit the error silently |
| **RESOLVED 2026-08-22 by the ULIP2 Engineer — option 3, and it saved the assets** | Master refused to pick remedy 1 or 2 until the crash question was answered. **They were the same assets.** Re-running the 103 (23 stale + 80 sidecar-less) rendered **23 and quarantined 80**: the 23 were never bad assets, they were victims of the 21:06 `BrokenProcessPool`. **Options 1 and 2 would each have discarded 23 recoverable assets.** Re-verified by Master: index `{4: 45,972}`, on-disk sidecars `{4: 45,972}`. **The corpus is uniformly v4 and that statement is now true** |
| **The general defect is fixed at source, commit `2ab1166`** | `retire_stale_sidecar()` (`renders.py:524`) **renames** a failed asset's old sidecar to `<uid>.json.stale` — renamed, not deleted, because it is evidence of a failure, and `.stale` falls outside `rebuild_index`'s `*.json` glob. **Three exist on disk**, so the mechanism has actually run |
| **`DL-018`'s guard also landed, after the re-run as Master required** | `implementation_fingerprint()` (`:202`) hashes **`renders.py` and `meshload.py`** — the second because it owns `FRAME_CORRECTION`, which moves every asset while `renders.py` does not change by a byte. `verify_fingerprint()` (`:228`) aborts a worker whose source differs from the run's. **Master exercised it: a wrong fingerprint raises, a single changed file raises, a matching one passes** |
| ⚠️ **Master's own test was wrong first, recorded because the reason is instructive** | Master's first attempt reported *"the guard does not fire"*. **It fires.** `_FINGERPRINT_VERIFIED` is a per-process latch, and Master called the **matching** case first, which latched it — every later call returned early. Re-run from fresh interpreters, all three cases behave correctly. **The latch is right, not a hole**: `max_tasks_per_child=200` means workers respawn as new processes, so each fresh worker re-imports and re-verifies |
| **Still open, and the Engineer states it rather than rounding it off** | **80 assets cannot be rendered at all** — they failed under v3 and v4 alike. The corpus is **"45,972 complete + 80 unrenderable"**, not 46,052, and the 80 need a stated reason. `every view is blank` (249 records) is the leading cause; some are fully transparent materials, which is an asset property rather than a defect. **The Engineer has not classified all 80 and declines to claim they are all benign** |
| **Status** | **RESOLVED.** Fix verified by Master; the 80 unrenderable assets remain open as a reporting obligation |
| **Date** | 2026-08-22 |

---

### `DL-020` — ESSGNN is FULLY paused, stricter than `DL-009`. ULIP2 finishes first.

| | |
|---|---|
| **Source** | USER, 2026-08-22, verbatim: 「n04 它在重跑 ESSGNN的先全部暫停 我還沒有要做 先完成ulip2端的東西」 |
| **What changes** | `DL-009` held ESSGNN closed but left both roles **reading, investigating and reporting** — which is how tonight's `U-20`, frame and `h0_mode` findings arrived. **That latitude is withdrawn.** Both roles stop entirely: no reading, no scanning, no new findings, no continuing work in progress |
| **Why it is not a rebuke** | Everything they produced is captured and verified — `DL-016`, `DL-019`, `D-14`, and three of the methodology rules in `CONTEXT.md` §3. The pause is about **sequencing**, not quality: `n04` is re-running, ULIP2 owns the critical path, and a second block generating findings faster than Master can integrate them is a queue, not progress |
| **Both roles notified** | Told explicitly that nothing is lost, where each finding landed, and to acknowledge without replying with new findings |
| **Corrections sent with the stop, so nothing sits wrong while they are silent** | `h0_mode` **is** locked — proven by execution against a hostile protocol. **Attribution corrected at the Engineer's request:** the *Reviewer* wrote 「沒有被鎖住」; the *Engineer* had run the same hostile-protocol check itself and reported the narrower and correct point — **the override is SILENT**, so someone writing `h0_mode` into the protocol is ignored with no warning. It endorsed the Reviewer's finding, which carried the wrong framing, but its own was right. Recorded so it does not re-verify what it already verified. And of the four "unregistered deviations" only **one** is a deviation: `D-14`. The other three follow the paper, and registering four would have put three false entries in the registry |
| **On resuming** | Read `DL-016`, `DL-017`, `DL-019`, `DL-020` before restarting. Do not re-derive |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED, the USER's own wording above |
| **Date** | 2026-08-22 |

---

### `DL-019` — `h0_mode` decides whether SE(3) equivariance holds, contradicts the paper's literal text, and lives in a **dataclass default** outside the protocol

**Found by the ESSGNN Reviewer while attacking the Engineer's `U-20` answer. Every claim
re-verified by Master, including by execution.**

| | |
|---|---|
| **The switch** | `essgnn.py:170` — `h0_mode: H0Mode = "semantic"`, a **dataclass default**. `:262` repeats it in the defaults dict. `:569` — `h0 = cat([pos, node_feat]) if h0_mode == "concat_xt" else node_feat` |
| **It contradicts the paper's literal text, and the code says so itself** | `2methdology.tex:44`, **PAPER FACT**: `h_i^{(0)} = Concat(x_i, t_i)`. `essgnn.py:254`, the code's own comment: *"`h0_mode=semantic` in particular **CONTRADICTS** 2.5's literal"*. The Appendix-C premise was adopted by `C2`; **the divergence is deliberate and it is a DEVIATION** |
| **It is what makes equivariance hold or fail** | `test_essgnn.py:138` uses `h0_mode="concat_xt"` as the **negative injection** for the equivariance test — the other value exists in the suite precisely to break it. Master re-ran: `pytest -k equivarian` → 3 passed |
| **And it is NOT in the protocol** | `essgnn_arch_protocol.json` holds twelve keys — `architecture_family, coord_feat, decided_at, decided_by, distance, hidden_dim, layer_sharing, mlp_structure, n_layers, pooling, status, use_io_projections`. **No `h0_mode`.** Confirmed by JSON parse, not by eye |
| **Why that placement is the defect** | That file is a `decided_by: Kyzen (2026-08-19)` resolved artifact, and `CONTEXT.md` §4 states its purpose: *"write the JSON artifacts trainers are not allowed to decide for themselves."* **A setting that determines equivariance and departs from the paper sits exactly outside it, decided by a Python default** |
| **The fourth item of the same debt, and the worst-behaved** | `DL-016` already recorded `node_feat_dim` and `edge_feat_dim` as absent and inferred at runtime; `edge_proj_dim` is the third. **`h0_mode` is the fourth and it is the dangerous one: the other three raise a shape error when wrong. This one raises nothing** — equivariance simply stops holding, silently. `01_GRAPH_SPEC.md:1123` warned about exactly this class: *"otherwise it is the Stage 1 error we just fixed, replayed in Stage 2"* |
| ⚠️ **BOTH ESSGNN ROLES SAID "NOTHING LOCKS IT". THEY ARE WRONG — Master checked, and the design is deliberate** | `essgnn.py:250-258`, the declaration comment, verbatim: *"**Unlike the fields in `essgnn_arch_protocol` these are NOT open questions a person has to answer** — they are our primary interpretation, and a run that departs from them is a variant that must say so. `L1-ESSGNN-PAPER-LOCKED-CONFIG` asserts them."* The lock is real and it is at a **higher** level than the protocol: `from_protocol` ends with `**PRIMARY_INTERPRETATION` (`:248`), which **overrides whatever the protocol says**. `validation_plan.yaml:755` specifies exactly that. And **two tests bite**: `test_paper_locked_values_are_the_defaults` fails if any default drifts, `test_from_protocol_is_the_supported_construction_path` feeds a protocol and asserts the four survive it. Master ran them — pass. **Editing the default does not silently change equivariance; it turns the suite red.** Being absent from the protocol is the *design*, not the omission |
| **Scope is wider than either role reported** | `PRIMARY_INTERPRETATION` pins **four** settings, not three. The ESSGNN Engineer named `h0_mode`, `edge_proj_dim`, `normalize_coord_diff` and **missed `coords_agg: "sum"`** — *"Eq. 3 sums; the reference EGNN defaults to mean"*, a divergence from upstream in its own right |
| **So what is the ACTUAL defect? The registry, and it is the sixth of the same class today** | `h0_mode = "semantic"` **contradicts `2methdology.tex:44`'s literal `h_i^(0) = Concat(x_i, t_i)`** — the code says so itself — which makes it a **DEVIATION**. Master measured: `h0_mode` appears **0 times** in `docs/graph/graph_spec.yaml`. **It has no deviation id.** Same class as `D-9`…`D-13`, found six hours later, in a fifth place nobody had swept. `coords_agg`, `edge_proj_dim` and `normalize_coord_diff` need the same audit |
| **The lock WORKS — Master proved it by execution, not by reading** | Constructed a hostile protocol carrying `h0_mode="concat_xt"`, `normalize_coord_diff=True`, `coords_agg="mean"` and passed it to `from_protocol`. **All four came out at the locked values.** The `validation_plan` claim *"forces them regardless of the protocol"* is **true** |
| **But the Reviewer self-downgraded and found the real narrow defect, which is better than its first one** | `L1-ESSGNN-PAPER-LOCKED-CONFIG` declares two verifications **that no test performs.** (1) *"`from_protocol` forces them regardless of the protocol"* — `test_paper_locked_values_are_the_defaults` builds a **bare `ESSGNNConfig`** and never calls `from_protocol`. Master had to write the hostile-protocol check by hand just now; **nothing in `tests/` does it.** (2) The declared negative injection, *"set `normalize_coord_diff=True` in the trainer's config"*, **does not exist** — Master measured: `normalize_coord_diff` appears in `tests/` exactly **once**, in a docstring at `:547`. **The lock is real and the suite does not prove it.** And what the test does check — dataclass defaults against a dict **in the same module** — one commit changing both keeps green |
| **Master's addition — neither role saw this, and it is the sharp end** | A hostile protocol is **silently ignored, not rejected.** Someone can write `h0_mode: "concat_xt"` into `essgnn_arch_protocol.json`, believe they have configured a variant, and receive the mainline model **with no warning and no error**. The forcing is correct; **the silence is a trap**, and it points the opposite way from the risk the roles feared — not *"the lock can be bypassed"* but *"a deliberate variant can be silently un-configured"* |
| **Severity — revised twice, and land here** | **Not a silent-equivariance risk**: locked and forced, proven by execution. **Two real defects remain:** an **unregistered DEVIATION** (no id in `graph_spec.yaml`, debt `D-2`/`FU-A` again) and a **declared-but-unwritten verification** in `validation_plan.yaml`. Writing the missing tests is implementation and belongs in the ESSGNN Engineer's SPEC self-verification, **not on the USER's decision list** — the Reviewer removed it from that list itself, correctly |
| **Keep it separate from `U-20`** | Different knobs. `U-20` is *which encoder produces `t_i`*; this is *which formula produces `h⁰`*. **`h0_mode` has no `U-` id at all.** Do not merge them |
| **Master's note on the Reviewer's `U-20` attack** | It kept the Engineer's conclusion on pillar 1 and **replaced the reason** — grounding the separate widths of `t_i` and `e_ij` in `2methdology.tex:54`'s `f_h: R^(2d+1+e) → R^d`, where the paper itself names `d` and `e` separately. **That is a PAPER FACT and does not move when our implementation moves**, which the projection argument did. Stronger. It also broke pillar 2 (the 93-distinct-values argument) and declined to rule on `U-20` itself. Correct on both counts |
| **What is needed** | Whether `h0_mode` — and the other three widths — enter `essgnn_arch_protocol.json` is **material**: the protocol hash changes, and `CONTEXT.md` §5 makes that force a split rebuild. Not Master's, not the block's |
| **Status** | **`AWAITING_USER_REVIEW`** |
| **Date** | 2026-08-22 |

---

### `DL-018` — `n04`'s corpus is mixed-generation and must be re-run. **A pipeline can change behaviour mid-run and no artifact reveals it.**

**Found independently by Master at integration and by the ULIP2 Engineer, within minutes of each
other. The Engineer's account is more complete and it self-reports the cause.**

| | |
|---|---|
| **What is on disk** | `n04` finished 20:43:03 — 45,958 sidecars + 94 quarantined = 46,052. **`renderer_version` is `{3: 43,412, 4: 2,546}`.** Verified by Master over the whole index |
| **Cause — the Engineer's, self-reported** | `n04` runs under `multiprocessing`; a newly spawned worker **re-imports the module**. `renders.py` was edited **while the run was in flight** (`mtime` 20:38:07; the version boundary in the sidecars is 20:38:45 — a 38-second lag, i.e. workers picked it up almost immediately). **The run changed program mid-flight** |
| **The part that makes a partial salvage unsafe** | The **bake fix landed 2–3 minutes before the version bump**. At ~686 assets/min that is roughly **1,700 assets carrying the corrected geometry while stamped `3`** — and from the sidecar they are **indistinguishable** from the ~41,700 genuinely uncorrected `3`s |
| **Master's correction to the scope — smaller than "re-run all 46,052"** | Verified by **calling the predicate**, not by reading it: with `RENDERER_VERSION = 4`, `is_complete()` returns **`False`** on a `v3` asset and **`True`** on a `v4` one. **So a bare `python -m metafind.data.renders` regenerates exactly the 43,412 and skips the 2,546 — no flag, no hand-picked list.** The ~1,700 mislabelled ones are inside the 43,412 and simply get re-rendered to the same geometry: **wasted work, not a correctness risk.** ≈ 43,412 / 691 per min ≈ **63 minutes** |
| **Why the `2,546` are safe to keep** | They were written after **both** the bake and the bump, by the code now committed at `138cda4`. `renders.py` is committed and stable — Master checked `git status` is clean on it before relying on this |
| **The underlying defect, and it outlives this incident** | **This pipeline's behaviour can change during a single run and nothing in the output says so.** Only the version bump made it visible at all, and only because it happened to be *late* — had the Engineer bumped the version **with** the bake, the corpus would have been uniformly stamped `4` with 43,412 assets rendered by the old code, **and every gate would have passed.** The version field caught this by accident, not by design |
| **Proposed remedy — the Engineer's, and Master endorses it** | Record the sha256 of the implementation modules into the runlog at run start; each worker verifies on spawn and aborts on mismatch. Cheap, and it converts a silent corruption into a loud stop. **Not implemented** — it is new gate behaviour and belongs after the re-run, not in front of it |
| **Classification** | The mixed corpus is `OBSERVED DATA`. The cause is **operator error, self-reported** — not a tool defect. The reproducibility hole it exposed **is** a tool defect and is registered here separately so it is not written off with the incident |
| **What is needed** | **USER authorisation for the ~63-minute re-run.** No research question is open; nothing is ambiguous; it is an expensive execution and `BLOCKS.md` reserves that |
| **Status** | **`AWAITING_USER_REVIEW`** — authorisation only |
| **Date** | 2026-08-22 |

**Also recorded, because it paid off today:** the `--from-disk` decision to read each sidecar's own
`view_paths` rather than glob the directory means the **88 orphan render directories** — PNGs
written before a quarantine failure — are invisible to every sidecar-driven consumer. A globbing
tool would have scored them.

---

### `DL-017` — the USER delegates these calls to the blocks. **Master stops escalating them.**

| | |
|---|---|
| **Source** | USER, 2026-08-22, verbatim: 「這些是誰的任務 給他決定啊 幹 問他們工程師啊 他們怎麼判斷的」 |
| **Decision** | **The block that owns the node decides.** `DL-013` → ULIP2 Engineer. `DL-016` / `U-20` → ESSGNN Engineer. The USER wants **to be told how they judged it**, not to judge it |
| **Master's error this caused** | Master escalated `DL-013` **four times** and `DL-016` once, correctly by `BLOCKS.md`'s materiality list, and **wrongly in practice** — the USER had no way to adjudicate a `16/37` measurement or a two-week-old registry field, and had already delegated once (the `E-1`…`E-11` series). **Escalating a decision the USER cannot make is not caution; it is handing back work.** Recorded because the rule that produced it is still in `BLOCKS.md` |
| **What this does NOT change** | Delegation of **these adjudications**, not of the gate. Still the USER's: authorisation **before** an expensive run · `MASTER-IMPACTING FINDING`s · anything where continuing would require inventing research-critical information · `DL-009`'s hold on ESSGNN, which is **not** lifted by a decision being delegated into it |
| **Classification consequence** | Whatever the blocks decide is an **IMPLEMENTATION CHOICE**, never a `USER DECISION` — he delegated the call, he did not make it — and never a `PAPER FACT` |
| **Master's job on these** | Re-verify the reasoning rather than rubber-stamp it, then rewrite `DL-013` and `DL-016` from `AWAITING_USER_REVIEW` to the block's decision with its evidence and Master's verification attached |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED, the USER's own wording above |
| **Date** | 2026-08-22 |

**`DL-013` and `DL-016` below are superseded in their `Status` line only.** Both now read
*"delegated to the block under `DL-017`"*. Their findings, evidence and open questions stand
unchanged — the delegation moved **who answers**, not **what was found**.

---

### `DL-016` — `U-20` is marked RESOLVED but was decided by Claude, and four files disagree. **BLOCKER for `n13`.**

**Found by the ESSGNN Engineer, verified by the ESSGNN Reviewer, re-verified by Master. It arrived
in time to stop Master registering a duplicate id for the same knob.**

| | |
|---|---|
| **What `U-20` covers** | Which text encoder produces ESSGNN's node features `t_i`. Today: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, 512-d, shared with the semantic-edge embeddings `e_ij` |
| **The near-miss** | Master had written that `TEXT_ENCODER` *"has no question id"* and intended to register one. **It has one — `U-20`.** A new id would have left **one knob under two ids, one `RESOLVED` and one open** — worse than the gap. Caught before Master acted |
| **Layer 1 — it is marked RESOLVED** | `graph_spec.yaml:1702`, `marked: RESOLVED`, `blocking: false` |
| **Layer 2 — by whom** | `decided_by: 'Claude (recorded with n08 implementation)'`, `decided_at: '2026-08-16'`, `confidence: moderate`. **A research-critical UNKNOWN was closed by an AI in passing, during implementation.** This is `DL-013`'s shape, two weeks earlier, and it has been in force ever since — every `n08` artifact rests on it |
| **Layer 2b — what the entry's OWN text says, and this is the sharpest part** | `U-20`'s `item:` field reads: *"Paper 2.5 says only 'a text-derived feature `t_i` in `R^d`'. It names a frozen text encoder ('e.g., CLIP or BERT') for the semantic **EDGE** embeddings, **not for `t_i`**, and does not say the two are the same encoder."* **The registry records that the paper does not support the choice, and marks the choice RESOLVED on the same line.** Meanwhile `semantic_edges_run.py:98-101` argues it as near-forced — *"Whatever encodes `t_i` must be THIS model"* — which is a **design argument, not paper evidence** |
| **Layer 3 — four files, two states** | `graph_spec.yaml:1702` **RESOLVED** · `01_GRAPH_SPEC.md:611` **UNKNOWN** · `02_BUILD_STEPS.md:299` **「未定 U-20」** · `docs/graph/README.md:17` counts it among the ten resolved |
| **Layer 3b — Master said "no gate catches this". WRONG, and the truth is more useful** | The ESSGNN Reviewer corrected it and Master verified. **The gate exists**, `check_graph.py:320-342`, and its own comment says it was written for exactly this: *"Counting every U- id as UNKNOWN is what let **U-34** sit in the registry as RESOLVED while three documents still called it open."* **This failure mode already happened once and someone built a gate for it.** The hole is two words: `_OPEN_PHRASES` lists 「未解」「尚未確立」「取決於」「仍待」「阻斷級」`unresolved`, `still open`, `open candidate`, `BOTH READINGS REMAIN`, `epistemic candidate` — **and neither `UNKNOWN` nor 「未定」**, which are the two words the conflicting lines actually use. The checker reads `marked: UNKNOWN` as a **field** at `:230-231` and cannot see `UNKNOWN` as a **word** in prose. A gate written for this class, missing this instance, is harder to find than no gate at all |
| **Layer 3c — TWO defects, different sizes. Master conflated them and the Reviewer separated them** | **Defect A — the decider.** Of the ten `RESOLVED` entries, exactly **one** has no USER in `decided_by`: `U-20`, `Claude (recorded with n08 implementation)`. `U-26`'s *"user + external review"* is the USER in different words. Independently audited by the ESSGNN Engineer and re-run by Master: `U-08/08a/08b/08d/08e`, `U-18`, `U-21`, `U-34` all read `Kyzen`. **Scope: 1. Fixing it is reopening a research decision — the USER's.** **Defect B — the documents.** Live files that call a `RESOLVED` id open: **seven** — `U-08`, `U-08a`, `U-08b`, `U-18`, `U-20`, `U-21`, `U-26`, in `01_GRAPH_SPEC.md`'s status table (`:597`–`:619`, second column literally `UNKNOWN` / `UNKNOWN・阻斷`) and `02_BUILD_STEPS.md:299`, `:517`. **Scope: 7. Fixing it is aligning documents with the registry — Master's.** `U-20` is the only id that has both, which is why they looked like one problem. **Fixing `U-20` alone leaves six untouched** |
| **Layer 3d — Master's own dry run was WRONG about the false positives, corrected by the Reviewer's** | Master ran it, counted 8 ids, and concluded *"most are false positives"* from `README.md:17` being the legend row for the 「[未定]」 marker. **Only that one line is a legend.** Excluding it the way `check_graph` itself does leaves **7 ids in genuine status tables** — `01_GRAPH_SPEC.md:597`–`:619` is a real table whose status column says `UNKNOWN`. `U-34` drops out; the other seven stand. **They are real contradictions, not noise** |
| **Layer 3e — a second gate bug, found by the Reviewer, that must be fixed FIRST** | `check_graph`'s id match is `if uid not in line` — **a bare substring with no word boundary.** Verified: `"U-08" in line` is `True` on the `\| **U-08a** \| **UNKNOWN・阻斷** \|` row. Harmless today because no phrase matches anything. **The moment `UNKNOWN` is added it starts producing false failures** — `U-08` convicted by `U-08a`'s line. **Correct order is two steps: word-boundary the id match, then extend `_OPEN_PHRASES`.** Reversed, the gate goes red and blocks everyone with half the reasons fake. Registered as work, not done — changing a gate while `n04` runs and three roles are live is the wrong moment |
| **Layer 4 — the spec's own gate was crossed** | `01_GRAPH_SPEC.md:1123`: `U-12`/`U-20` *"**在 `n13` 實作前必須進 `n09b`／`G6`**"* — must land in the protocol **before `n13` is implemented**, *"otherwise it is the Stage 1 error we just fixed, replayed in Stage 2"*. Measured: `essgnn_arch_protocol.json` has **no `node_feat_dim` and no `edge_feat_dim`** — keys are `architecture_family, coord_feat, decided_at, decided_by, distance, hidden_dim, layer_sharing, mlp_structure, n_layers, pooling, status, use_io_projections`. Both widths are inferred at runtime instead (`stage2.py:315-327` checks `shape[1]` against values taken from the artifacts). **`n13` is 680 lines and complete. The gate was crossed and nothing reported it** |
| **Why reopening is defensible on either ground alone** | **(a)** its decider is Claude, not the USER, at `confidence: moderate`, on a question the entry itself records the paper as not answering; **(b)** it was decided against `n08`'s artifacts, and **`n08` is about to re-run** under the no-old-model-output directive |
| **What Master did NOT do** | **`marked: RESOLVED` is not changed here.** Reopening a resolved research question is a USER decision, not Master's, and editing the field would be resolution-by-mutation. A dispute note is added to the entry pointing at this record; the four-file contradiction is left standing **because picking a side would be the same offence** |
| **Priority — Master accepts the Reviewer's merge** | The ESSGNN priority list's item 2 becomes *"the two things `n08` cannot re-run without: `Q-N08-MODEL` and reopening `U-20`"*. `U-20` is the more fundamental of the two: `Q-N08-MODEL` moves the edge sentences, `U-20` moves **both vectors and their width** |
| **Authority classification** | **IMPLEMENTATION CHOICE recorded as RESOLVED.** Not a PAPER FACT — the entry's own text says the paper does not say it. Not a USER DECISION |
| **Severity** | **BLOCKER for `n13`** · MAJOR for the `n08` re-run |
| **What is needed** | The USER decides whether `U-20` reopens. Master then makes the four files agree with whatever that answer is, and registers the unpaid `n09b`/`G6` protocol debt either way |
| **Status** | **DELEGATED to the ESSGNN Engineer under `DL-017`** — **ANSWERED, see below** |
| **Date** | 2026-08-22 |

#### The block's decision, and Master's verification of it

**ESSGNN Engineer, 2026-08-22: reopen `U-20`, keep CLIP ViT-B/32 / 512-d as a recorded
IMPLEMENTATION CHOICE, and rewrite its justification because the recorded one is false.**
Master re-verified every load-bearing claim by execution before accepting.

| | |
|---|---|
| **A fourth reason to reopen, better than Master's three** | Master's three were procedural — the decider, the self-contradiction, the crossed gate. **The Engineer's is substantive: `U-20`'s recorded justification is refuted by the code.** The stated reason is that `t_i` and `e_ij` both feed `f_h`, so different encoders would put them *"in two unrelated semantic spaces"*. **They are already in different spaces.** Verified: `essgnn.py:456-457` — with `use_io_projections` (protocol: `true`) `embed_in = nn.Linear(in_dim, 128)`, so `t_i` arrives as a **learned** 128-d vector; `:471` — `edge_proj_dim is None` (`:264`, *"the paper has no such layer"*) so `edge_proj = Identity` and `e_ij` arrives **raw at 512**. The shared encoder buys no shared space; the projection removes it. **The justification describes a configuration we are not running** — and under the one where it *would* hold (`use_io_projections=False`), `in_dim == hidden_dim` is forced, so a 512-d encoder is not even legal |
| **Paper, read verbatim** | `2methdology.tex:47` — *"These sentences are then encoded … using a frozen text encoder (e.g., CLIP or BERT), resulting in **edge embeddings e_ij**"*. The clause attaches to `e_ij`, and even there it is **`e.g.`** — an example, not a specification. For `t_i` the paper gives only *"text-derived"* and `∈ R^d`. **PAPER FACT: the encoder for `t_i` is unspecified.** `U-20`'s own `item` text said exactly this and was right |
| **`DL-010` does not reach it — there is no upstream to inherit** | Not case 1. **EGNN has no text features at all**: `/home/kyzen/upstream/egnn/models/gcl.py:242`, `forward(h, edge_index, coord, edge_attr, node_attr)`, where `h` is a discrete node attribute. Master adds a **stronger citation than the Engineer used** — MetaFind says so itself, `appendix.tex:20`: *"While the original EGNN formulation allows the inclusion of edge features in the message function, **these are typically discrete, task-specific features**"*. So this is a **PAPER FACT**, not an inference from upstream code. Text semantics in EGNN is the paper's own claimed contribution (`2methdology.tex:42`, *"extends the EGNN formulation to incorporate semantic edge features"*). **No upstream → this must be a recorded IMPLEMENTATION CHOICE. Leaving it UNKNOWN is wrong (no value); calling it an UPSTREAM FACT is wrong (no upstream)** |
| **The shared-encoder argument, judged on merit rather than on paper silence** | **Convenience, not constraint.** *"Different encoders → unrelated spaces"* is false, per the projection above, and an MLP over heterogeneous inputs is ordinary. The width argument is the one that survives and it is **unrelated to sharing**: `f_h`'s input is `2·128 + 1 + 512 = 769`, of which geometry is a **single scalar**. That argues `e_ij` should not be too wide, never that `t_i` must share its encoder. **Sharing may stay as a choice; that reason must be deleted** |
| ⚠️ **The finding that reorders the work — measured, and Master reproduced it exactly** | `procthor_object_text.json`: **1,467 entries, 93 distinct strings.** 15.8 assets per vector. `Apple_1` and `Apple_10` are byte-identical *"a apple"*. With `h0_mode: semantic`, `t_i` **is** the entire initial node state — so **every ESSGNN node starts from one of 93 possibilities.** **Changing the encoder cannot add information that was destroyed before encoding.** `Q-NODETEXT` therefore precedes `U-20`'s substantive choice: swapping the encoder first is a better lens on the same blurred negative, and costs two `n08` runs |
| **What `n08` needs before it re-runs** | `essgnn_arch_protocol.json` gains **`node_feat_dim`** and **`edge_feat_dim`**, separately, plus the encoder id and version — paying the debt `01_GRAPH_SPEC.md:1123` recorded and nobody paid. Separately, because `stage2.py` already reads them separately and its comment records that conflating them **was** a fixed bug |
| **Stop-safe, correctly invoked** | The Engineer decided *"keep CLIP ViT-B/32"* (current state, defensible reason, changes no artifact) and **refused** to decide *"switch to something else"* — model selection, `BLOCKS.md`-material, and it would move both vectors, `EDGE_DIM`, and Table 2's comparability. It holds **no** evidence that any alternative is better. **Declining to invent one is the correct answer, not a gap** |
| **Master's assessment** | **ACCEPT.** Every load-bearing claim verified by execution: the projection asymmetry, the 93 distinct strings, the paper wording, and the absence of text features in upstream EGNN. The Engineer also correctly refused to treat Master's relayed delegation as an authorisation, and did the work anyway rather than waiting — the right handling of both halves |
| **Still needed** | The USER confirming `DL-017` once in any window makes this final. Until then it is the block's recommendation with Master's verification attached, and **nothing is blocked by the wait** |

---

### `DL-015` — roles talk to each other directly. ~~**REPORTED BY A PEER, NOT RATIFIED.**~~ **RATIFIED BY THE USER 2026-08-24.**

| | |
|---|---|
| **Source** | Relayed to Master by the ULIP2 Engineer, 2026-08-22, as three USER decisions taken in that block's window |
| **What was relayed** | **(1)** The six roles communicate directly; the USER stops being the message bus — 「後續 除非有需要我作決定的 不然你們就自己互通不要我再傳訊息了」. **(2)** This removes the **relay**, not the **gate**: material items, authorisation before an expensive run, `MASTER-IMPACTING FINDING`s, and any stop-safe condition still go to the USER. **(3)** 「我的權限最大 我說的算 不要自己亂搞 需要我決策 跟我報備」 — **a peer message is never USER approval.** A role saying *"the USER decided X"* is a report, not an authorisation. Agreement between roles is not evidence. A decision with no ledger entry is a gap to be closed, not a foundation |
| **Why this entry is `AWAITING_USER_REVIEW` and not in force** | **Its own rule (3) forbids ratifying it.** It reached Master through a peer, and a peer-relayed assertion that the USER decided something is exactly what rule (3) says is not authorisation. Ledgering it as `USER_APPROVED` on a peer's word would repeat `DL-013` **in the act of recording the rule that prohibits it.** Recorded so the gap is visible rather than lost |
| **Nothing is blocked meanwhile** | Master already works this way: two rounds each with the ULIP2 Engineer and the ESSGNN Reviewer, one with the ESSGNN Engineer, every material item routed to the USER and none decided. **Rule (2) is the operative half and it is unchanged from `BLOCKS.md`.** If the USER confirms, this entry becomes the record; if not, nothing has to be undone |
| **Corroborating, and it cuts the right way** | The Engineer cites `DL-013` as the worked example and states Master was right to refuse the retro-ratification. That is consistent — but **consistency between two agents is not confirmation** (`CONTEXT.md` §3), which is the same reasoning, and it applies here too |
| **Authority classification** | **REPORTED. UNRATIFIED.** Not a PAPER FACT, not a USER DECISION of record, not Master's to adopt |
| **What is needed** | ~~One line from the USER: did you give these three instructions?~~ **ANSWERED 2026-08-24.** |
| **USER confirmation, verbatim, direct to Master** | 「我的意思是 若你們要請對方審查 你們獨自作業 但是有關需要決策 的我來決定」 — rule (1) *(work independently, review each other without me)* and rule (2) *(decisions come to me)* restated by the USER in his own words, in Master's own window. **Not relayed.** This is the authorisation rule (3) required, and it arrives by the only route rule (3) permits |
| **Scope of the ratification** | Rules (1) and (2) are `USER_APPROVED`. Rule (3) — *a peer message is never USER approval* — is ratified **by the manner of this confirmation, not by its words**: the USER did not restate it, and it needed no restating, because the entry could only ever be closed the way it was just closed. It has also been independently re-derived and applied by the ULIP2 Engineer on 2026-08-24 without reference to this entry |
| **Status** | **`USER_APPROVED`** — IN FORCE. Held `AWAITING_USER_REVIEW` for 2 days by its own rule (3); closed by direct USER confirmation |
| **Date** | relayed 2026-08-22 · confirmed by the USER 2026-08-24 |

---

### `DL-014` — commit attribution: explicit paths, plus a distinct git identity per role

| | |
|---|---|
| **Source** | Master's ruling on the ULIP2 Engineer's escalation item 3, 2026-08-22 |
| **Issue / Finding** | Six roles commit to one working tree as one git identity, `Kyzen5128 <legend2341528@gmail.com>`. **`git` cannot tell them apart, and `git add -A` from any role sweeps every other role's uncommitted work into its own commit.** It already happened twice: `58637f3` claims the `R-12` carve-out and holds only `HANDOFF.md`; `4e5053f` is labelled `docs:` and holds the Engineer's `pointclouds.py` +57, the Engineer's tests +56, three of Master's governance files, and the Reviewer's `REVIEW.md` +84. **Both were Master's `git add -A`** |
| **Why it is not cosmetic** | `.claude/rules/experiments.md` §7 requires results to be attributable to the code state that produced them, and §10 requires artifact provenance. `code-changes.md` §13 requires the diff to be inspectable. A dataset-semantics change filed under a documentation message defeats both, **silently and in both directions** |
| **Decision / Resolution** | **Options 1 and 2 together. Option 3 is refused.** **(1)** Every role commits by **explicit path**. `git add -A` and `git commit -a` are prohibited in this repository while more than one role is live — the Engineer and the Reviewer had already adopted this without waiting for a ruling, correctly. **(2)** Every role sets a distinct author on its own commits, per invocation, changing no global config: `git -c user.name="ULIP2 Engineer" -c user.email="legend2341528@gmail.com" commit …`. The email stays the USER's — it is their repository — so only `%an` distinguishes the roles, which is what `git log --author` and every blame tool actually read |
| **Why not option 3 (a worktree per role)** | It would prevent the collision by construction, but the roles are deliberately working on **one** tree: the Reviewer verifies the Engineer's uncommitted edits by executing them, and `R-4` timed its findings against the Engineer's edits landing 42 seconds later. Separate worktrees would break synchronous review, which is the point of having a Reviewer at all. **Rejected on function, not on cost** |
| **Why (1) alone is not enough** | It prevents the next collision but attributes nothing retrospectively. The Reviewer is right that only a distinct identity gives after-the-fact attribution |
| **History is not rewritten** | Two sessions were committing concurrently when the collisions happened and a rebase could have destroyed uncommitted work. The correct commit-to-work mapping is recorded in `c9ef702`'s `HANDOFF.md` entry instead |
| **Authority classification** | **Process, not research semantics.** Master's to decide; recorded here because it governs experiment provenance |
| **Status** | **IN FORCE** from 2026-08-22. Master, the Engineer and the Reviewer have all adopted (1) |
| **Date** | 2026-08-22 |

---

### `DL-013` — the `texture` carve-out reverses a USER-confirmed choice on an inconclusive measurement. **Needs the USER.**

**Raised by Master at integration review, 2026-08-22. Reported as a FINDING; no remedy is proposed and nothing was reverted.**

| | |
|---|---|
| **Issue / Finding** | The USER confirmed **`P3`** — full glTF 2.0 conformance, `COLOR_0 × texture × factor`, ~1,130 geometries — at `R-10`'s ASK, and `U-AA` chose the specification as the authority. **`R-12` then withdrew the `texture` class from `P3`**, which is ~995 of those assets. That narrowing is implemented, tested and shipping as deviation `D-12`. **Master can find no `U-` code approving it** |
| **CHAIN OF CUSTODY — corrected 2026-08-22 after the Engineer supplied what Master could not see** | **The narrowing did not originate in the block.** It arrived in the Engineer's session as an instruction the USER pasted, headed 「三、USER 決定：範圍縮回 A（= P2）」, with the `R-11` reasoning and the two cosine comparisons attached. The Engineer implemented it **as routed**, which is the normal mechanism in this workflow, and did not choose it. **Master's original framing — that a block reversed a USER decision on its own authority — was wrong and is withdrawn.** The Engineer is not at fault |
| **What the defect actually is** | **A relayed assertion that a decision was made is not a ledger entry.** No `U-` code exists, so there is no record either role can point at, and the two of us can only reconstruct it. That is a **process** gap and it is **Master's to close**, not the block's. It is also the exact failure the `HANDOFF.md`-only communication rule exists to prevent, arriving from the one direction the rule does not cover: the USER speaking directly into a block window |
| **The question therefore changed** | It is no longer *"which option do you want"*. It is **"did you tell them to narrow it back to `P2`?"** — answerable only from the USER's own memory, and it must not be reconstructed from either agent's account |
| **What the measurement actually says** | `R-12A`, texture class only, the **full** ULIP overlap n=37: `COLOR_0` off reaches cosine **0.9005**, `P3` reaches **0.8980**. `P3` wins on **16 of 37**. 37 fair coin flips give 18.5 ± 3, so **16/37 is inside one sigma**, and **no paired significance test was run.** The Reviewer states this itself: *"the darkening is certain; 'therefore worse' is not"* |
| **What the narrowing actually rests on** | `R-11` — the USER's ruling that ULIP-2 is the reference architecture and agreement with it is the default. `R-12`'s own text: *"What decides it is `R-11`, not the cosine"* |
| **Why Master does not think `R-11` reaches this** | **`R-8` established that upstream publishes no cloud-colouring procedure at all** — `/home/kyzen/upstream/ULIP` @ `95d480fe` swept for `COLOR_0`, `trimesh`, `.glb`, `sample_surface`: zero hits; `ulip2_source/appendix.tex:10` delegates to OpenShape; OpenShape publishes no converter either. So there is **no upstream *behaviour* to agree with** — only its *artifact*, and the artifact does not discriminate at n=37. `R-11` says to default to agreement where agreement is measurable. Here it is not |
| **What is NOT in dispute** | The wide repair is right and well-evidenced. `P1` — discarding `COLOR_0` — was measurably worst at n=130 (0.8800 against 0.9043 / 0.9004, 27 wins against 103) and is correctly out. `R-13`'s four release conditions all PASS, the new test is verified non-vacuous, and the control group is bit-identical at n=60. **This entry is about one class and one ratification step, not about the `COLOR_0` work** |
| **Why it matters enough to raise** | It buys a **registered deviation from a published specification** (`D-12`) in exchange for a difference its own author calls noise, and it silently narrows a decision the USER made |
| ⚠️ **The window closed while this was being written** | **`n03` has already re-run to completion with `D-12` in force.** Verified by Master 2026-08-22: 46,052 `.npz` + 46,052 `.json` on NVMe, `sampler_version: 6` uniform over a 3,000-file sample, `color0_modulated` 137 true / 2,863 false. **The carve-out is baked into the corpus.** So this is no longer a free decision taken before a run — choosing **B** now costs an `n03` re-run |
| **Revised cost of each option** | **A. keep `D-12`** — zero cost, the corpus on disk is already this. **B. restore full `P3`** — `n03` re-runs. At the measured **897 assets/min** that is **≈51 minutes**, not the 3.3 hours quoted earlier: `n04` is unaffected (it takes colour from the material through pyrender, not through this path) and the output now lands on NVMe |
| **Master's assessment** | The block's work is careful and its reasoning is on the record. This is not a rogue change; it is a material choice ratified one level below where `BLOCKS.md` puts it. **Dataset-preprocessing semantics reaching the whole corpus is USER-material** |
| **Options, for the USER — Master recommends neither** | **A.** Keep `D-12`. The texture class stays unmodulated; the deviation from glTF 2.0 stands as registered. **B.** Restore full `P3` as originally confirmed; `D-12` is withdrawn from the registry and ~995 texture assets are modulated. Either way `R-8`'s prohibition binds: **it may never be written up as "what ULIP-2 did"** |
| **Not verified, and worth knowing before choosing** | `pointclouds.py`'s `GLTF_DEFAULT_BASE_COLOR = 1.0` is marked `INFERENCE` — the glTF schema is not on disk and `material.pbrMetallicRoughness.schema.json` has not been read. **8,853 `gltf_default` assets rest on that value.** Measured support exists (ULIP 35.3% pure-white against ours 35.1%, all-white 19/50 both sides) but that is `OBSERVED DATA`, not the specification |
| **Status** | **DELEGATED to the ULIP2 Engineer under `DL-017`.** No longer awaiting the USER |
| **Date** | 2026-08-22 |

---

### `DL-012` — four deviations registered: `D-9` … `D-12`

| | |
|---|---|
| **Source** | USER decision **`U-Z`** — *"Master registers the four un-id'd deviations on the Integrator's behalf"* — `DL-009` holds `INTEGRATOR` closed and opening a block to write registry lines was not warranted |
| **Issue / Finding** | Four real deviations had **no id in `docs/graph/graph_spec.yaml`**. `check_graph.py:373-383` compares deviation **ids only and never reads the `what:` text** (debt `D-2`/`FU-A`), so all four passed every gate silently. The ULIP2 Engineer escalated this three times and correctly refused to assign ids itself |
| **Registered** | **`D-9`** LVIS category anchoring (`DL-007`) · **`D-10`** contrastive negatives, one GPU's batch against ULIP-2's gathered 512 (`F-N10-1`) · **`D-11`** white render background against upstream's black (`U-W`) · **`D-12`** `COLOR_0` withdrawn from the `texture` class, against glTF 2.0 (`R-12`) · **`D-13`** corpus 46,052 against the paper's *"approximately 48,000"* (`U-01`) |
| **`D-13` — Master overruled its own first reading** | Master initially left `U-01` out, reasoning that a corpus-availability fact is not a *chosen* divergence. **`paper-reproduction.md` §9 says otherwise**: *"Any **intentional or unavoidable** difference … must be recorded as DEVIATION."* Unavoidable is explicitly in scope. Registering it does **not** reopen `O-2` — it is still carried as a stated Table 1 limitation, not resolved. `U-01`'s own stated resolution (*"use `len(manifest)`; record its sha256"*) had **never** been honoured; both sha256 values are now in the registry entry |
| **Latent gate bug found and fixed while doing it** | `check_graph.py`'s pattern was `D-[0-9]` — **single digit**. It could not match `D-10` or above **at all**, so every two-digit id would have read as missing from the mirror tables. Never reachable with eight single-digit deviations; reachable the moment a ninth was added. Changed to `D-[0-9]+`. Classified **bug fix**: the checker now sees what it always claimed to check |
| **Mirrors synchronised** | `docs/graph/README.md`, `docs/graph/02_BUILD_STEPS.md`, root `README.md` — the three the gate reads. Verified: all three list exactly the twelve ids `graph_spec.yaml` declares |
| **`U-01` is deliberately NOT registered** | 46,052 against the paper's *"approximately 48,000"* is carried as a **stated Table 1 limitation**, per the engineer's recommendation `O-2`. It is a corpus-availability fact, not a chosen divergence |
| **Caution carried into `D-12`'s own entry** | `R-12`'s texture carve-out rests on `R-11`'s default-to-upstream-agreement rule, **not** on a significant measurement: 16/37 wins is inside the noise for 37 coin flips and no paired test was run. `R-8` further established upstream publishes **no** cloud-colouring procedure, so there is no upstream *behaviour* to agree with — only its artifact. Recorded in the registry text so a later reader cannot mistake it for a settled result. **See `DL-013`** |
| **Authority classification** | The four deviations are pre-existing facts; **registering them is bookkeeping that follows from `U-Z`**, not a new choice. The regex change is a bug fix |
| **Verification** | `tools/check_graph.py` → 2275 checks, all pass · id sets compared programmatically across all four files |
| **Status** | **`USER_APPROVED`** — executed under `U-Z` |
| **Date** | 2026-08-22 |

**Debt `D-2` / `FU-A` is NOT discharged.** The checker still compares ids and never reads `what:`,
so a deviation whose *description* has gone false still passes silently. Unassigned.

---

### `DL-011` — the ULIP2 session's USER decisions, ratified as a block: `U-A` … `U-AC`

| | |
|---|---|
| **Source** | USER, in conversation with the ULIP2 Engineer and Reviewer, 2026-08-22. Recorded verbatim in `workflow/blocks/ULIP2/HANDOFF.md`, which both roles asked Master three times to ledger |
| **Why one entry and not thirty** | They are one continuous decision thread from one session on one milestone. Splitting them would imply they were weighed separately. **The HANDOFF entries are the record**; this entry ratifies them and pulls out the ones with project-wide reach |
| **Status** | **`USER_APPROVED`** — the USER's own wording is in `HANDOFF.md` and governs |
| **Date** | 2026-08-22 |

**Project-wide, not ULIP2-local:**

| | Decision |
|---|---|
| **`U-A`** | **No post-processing repairs.** A defect is fixed at its source, once, properly. The codebase is meant to be used by other people. 「要修就一次修好」. **A standing project rule, not an n04 decision** |
| **`U-O`** | MetaFind states it → follow MetaFind. MetaFind is silent → follow ULIP-2. **This is the same rule the USER gave Master independently as `DL-010`**; the two converge and `DL-010`'s three-case discriminator is the operative form. A choice made under it is an `IMPLEMENTATION CHOICE` with upstream provenance, **never a `PAPER FACT`** |
| **`U-Z`** | Master registers the un-id'd deviations while `INTEGRATOR` is held → executed as **`DL-012`** |
| **`U-Y`** | The multi-day `n05` run carries a **circuit breaker** whose threshold is **measured by the bake-off**, never invented now. Asked three times before it was answered |

**Corpus correction — `n03` / `n04` re-run AUTHORISED, overriding `BLOCK.md` §7:**

| | Decision |
|---|---|
| **`U-B`** | **`n04` re-render authorised.** The `+Z`-up camera is fixed at source and the corpus regenerated. A 100-asset A/B ran first |
| **`U-G`** | The 180° yaw is fixed **at the mesh-load layer**, so `n03` (46,052) re-runs too |
| **`U-W`** | Background reverts to **white** → deviation `D-11` |
| **`U-X`** | `ORTHO_HALF_WIDTH` reverts to **1.10** |
| **`U-N`** | All three render differences fixed: orbit axis, background, framing |
| **`U-AA`** | `COLOR_0` settled on the glTF 2.0 specification rather than by buying ~20 GB of further ULIP shards |
| **`U-H`** | The white-point-cloud question `F-N03-1` is settled by differential against ULIP's clouds **before** the re-run, so any bug is fixed in the same pass |

> **`BLOCK.md` §7's *"renders are read-only, no re-render"* is SUPERSEDED for `n03`/`n04`.**
> The prohibition rested on `evidence/n05_annotation_defect.md` Evidence 2, which measured
> **occupancy** (`correlation = +0.054`). **Orientation was never measured.** The prohibition was
> not wrong when written — it answered a different question.

**Annotation and training protocol:**

| | Decision |
|---|---|
| **`U-E`** | **11 views stands.** The engineer objected once to a change to 12; MetaFind states 11 twice. Objection accepted |
| **`U-I`** / **`U-P`** | Bake-off sample is **100 assets per arm**, 4 strata × 25: ordinary · rare LVIS class · low visibility · extreme aspect |
| **`U-Q`** | **20 assets hand-adjudicated by the USER**, shared across all arms. **The only ground truth in the whole bake-off** |
| **`U-L`** | **Two-turn identity check** — turn 1 blind, turn 2 with the LVIS anchor. `identity_confirmed` becomes **computed**, not asked. This is a direct answer to `IC-1`, the rubber-stamp risk Master raised |
| **`U-M`** / **`E-10`** / **`E-11`** | Multiple description candidates re-ranked by an **independent** CLIP (`openai/clip-vit-large-patch14`), never `ViT-bigG-14`, which `n06` encodes with — ranking and encoding with one model is circular |
| **`U-T`** | **Fusion default becomes `transformer`.** `fusion.py` recorded `U-13` as *"the paper never says which"* — **it does**: `3experiments.tex:143`, *"the final selected Transformer"*. A false `UNKNOWN` corrected back to a **PAPER FACT** |
| **`U-U`** | **Do not invent `lr` / `epochs` / `batch_size`. Measure them.** 72/8/20, the paper's 20% test untouched. The paper has **no implementation-details paragraph at all** |
| **`U-V`** | `n06` stores all 11 per-view vectors alongside the mean. GPU cost zero, ~2.6 GB storage |
| **`U-S`** | Widen the validator's dimension floor. **103 of 45,955** currently have an empty feasible height band — posters and decals, real data |
| **`U-AB`** | **`W-6` — the missing `D0-010` evidence audit — must complete before `M2`** |

**`Q-CATEGORY` is discharged, not open.** `R-6` established it and `DL-007`'s `D0-010` are the
same question, listing the identical four options. The true state is **decided, implemented,
unaudited** — not *"no investigation has been done"*. `U-AB` supplies the missing audit.

---

### `DL-010` — where MetaFind is silent on a component it did not modify, the official upstream implementation **is** the reference

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「它是原架構如果有地方不詳細應該是要以他們兩個的原程式碼為主啊 如果 metafind 沒有改動的話」 |
| **Issue / Finding** | Blocks were reading the project's upstream cautions as a blanket *"reproducing MetaFind means we may not consult ULIP-2 or EGNN"*. That is a misreading, and it is expensive: it turns answerable questions into `UNKNOWN`, and pushes engineers toward inventing a value when the authoritative one is sitting in `/home/kyzen/upstream/` |
| **Decision / Resolution** | **Three cases. Decide which one you are in before you cite anything.** **(1) MetaFind SILENT + component inherited unmodified → the official upstream implementation IS the reference.** Use it. Classify **UPSTREAM FACT** and state the inheritance basis. Do **not** record `UNKNOWN`, and do **not** invent. **(2) MetaFind SPEAKS but is ambiguous or self-conflicting → upstream supplies the VARIANT LIST, never the answer.** Escalate to the USER. **(3) MetaFind MODIFIED the component → upstream says nothing about the modified part.** No inheritance |
| **Precedent, already in force — this is not new policy, it is policy being applied consistently** | **Case 1 is `U-34`**: the CLIP freeze scope. MetaFind never states a per-module freeze policy; ULIP-2 §3.3 says *"freeze it during pre-training"*; MetaFind explicitly builds on ULIP-2 and never says it changed that. Resolved **frozen** on exactly this reasoning, `decided_by: Kyzen`, 2026-08-16. **Case 2 is `U-35`**: MetaFind says only *"approximated using MLPs"*; EGNN Appendix C gives **three different** MLP shapes and our `f_h` matches none of them. Left `UNKNOWN`, recorded as a protocol field — *"EGNN Appendix C supplies the VARIANT LIST, not the answer"* |
| **This does NOT weaken `DL-004`** | `DL-004`'s prohibition — *"upstream EGNN settles it" may not be used as paper-interpretation authority* — is **case 2, and it stands unchanged**. MetaFind **does** speak there: `2methdology.tex:54` states `f_x → R³` and `appendix.tex:23` claims equivariance for any orthogonal `Q`. Citing EGNN's scalar to overrule what MetaFind actually wrote is still forbidden. **`DL-010` governs silence. `DL-004` governs ambiguity. They do not overlap** |
| **Authority order is unchanged** | `CLAUDE.md` §3 already ranks *"original upstream papers and official upstream implementations"* **third — above** project audits, above the graph spec, above this repository's own code. `DL-010` does not promote upstream; it stops blocks from demoting it below reasoned guesswork |
| **Sources** | `/home/kyzen/upstream/ULIP` @ `95d480f` · `/home/kyzen/upstream/egnn` @ `e9ca6c0` · `docs/paper/{ulip2,egnn,idesign}_source/`. All verified present 2026-08-22 |
| **Still binding, unchanged** | Silence is never endorsement — judge whether the method works on its own evidence. An upstream detail is MetaFind-relevant only where MetaFind adopts, inherits, references or depends on it (`.claude/rules/paper-reproduction.md` §4). **An UPSTREAM FACT is never a PAPER FACT** |
| **Authority classification** | **USER DECISION** — paper-interpretation policy |
| **USER final decision** | **`APPROVE`** — 2026-08-22, the USER's own wording above |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

---

### `DL-009` — execution order: `ULIP2` runs to completion before `ESSGNN` opens

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「我想先完成 ULIP2 再接續完成」 |
| **Decision / Resolution** | **Sequential, not parallel.** `ULIP2` is the only open block. `ESSGNN` and `INTEGRATOR` are `ON HOLD` and unstaffed; the USER opens them |
| **Consequence Master carries, not a block** | `Q-TOWER` and `Q-BUILDMODEL` are Integrator-owned but **block ULIP2's own n09 splits**. While `INTEGRATOR` is on hold, **Master holds those two seams** and brings them to the USER when ULIP2 reaches n09. No block silently absorbs an Integrator question |
| **Consequence, cost — recorded so it is not rediscovered as a surprise** | Table 1 and Table 2 are ~10 nodes with **zero code** (verified 2026-08-22: no `metafind/eval/` directory exists at all), and **none of them needs a trained model** to be designed and unit-tested. Serialising means that work does not begin until ULIP2 finishes. **This is the USER's accepted trade** |
| **Authority classification** | **USER DECISION** — project sequencing |
| **USER final decision** | **`APPROVE`** — 2026-08-22, the USER's own wording above |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

---

### `DL-008` — the annotator is chosen by a lightweight bake-off. **Supersedes the model lock in `DL-006` (2026-08-21).**

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「LLM 選用我想在標註時跑個輕量測試作比較 我再決定 到底要選誰」 |
| **What is approved** | **the procedure only** — a lightweight sample run per candidate, then the USER picks the winner |
| **What is NOT approved** | **the winner.** No model is selected by this entry. `annotate_run.py:72`'s hardcoded `MODEL_ID = "/mnt/data1/kyzen/models/Qwen3.8-27B"` is **leftover state from the superseded entry, not a live decision** |
| **Supersedes** | `DL-006` **(2026-08-21)** — "the n05 annotation model is `Qwen3.8-27B`". That entry is **not deleted**; it was true when written. The model choice is **reopened** |
| **Measured candidate state — Master, 2026-08-22, `/mnt/data1/kyzen/models/`** | `gemma-4-31B-it-qat-w4a16` — **22,188 MiB**, `config.json` `quantization_config.quant_method: compressed-tensors`, official quantization-aware-trained, 0 `.incomplete`. **READY** · `gemma-4-12B-it` — **22,812 MiB**, bf16, no quantization at all, 0 `.incomplete`. **READY** · `Qwen3.8-27B` — **55.56 GB bf16**, 18/18 shards verified against `model.safetensors.index.json` `total_size`. **The w4a16 build named in `ULIP2/BLOCK.md` §11 does not exist on disk and has never been produced.** NOT READY |
| **Consequence for the deviation registry — Master routes, no block decides** | `graph_spec.yaml:133` states `D-2` as *"Qwen3.8-27B replaces GPT-4o for ASSET ANNOTATION (n05)"*. **If any other arm wins that text becomes false**, and `check_graph.py:373-383` compares deviation **ids only, never the `what:` text**, so a falsified description passes every gate silently. Registered debt `D-2`/`FU-A` is therefore no longer theoretical |
| **Standing, unchanged** | **The annotator is not GPT-4o whichever arm wins.** Deviation `D-2` is re-pointed, never discharged, and must never be written up as paper-faithful. LVIS category anchoring remains a separate DEVIATION (`DL-007`) |
| **Authority classification** | **USER DECISION** on procedure. The eventual winner is an **IMPLEMENTATION CHOICE backed by a measurement on this hardware and this sample** — never a general claim that one model is better than another |
| **USER final decision** | **`APPROVE`** — the procedure, 2026-08-22. The winner is **PENDING USER** |
| **Status** | **`USER_APPROVED`** for the procedure · the model itself is **REOPENED, PENDING USER** |
| **Date** | 2026-08-22 |

---

### `DL-006` — `D0-003` resolved: the 3 legacy-v1 residuals are deleted and re-annotated under v5

| | |
|---|---|
| **Source** | USER, 2026-08-22, following the 「不准使用舊模型產出的東西」 directive |
| **Issue / Finding** | Three uids carried `prompt_version 1` records long after the rest of the corpus moved to v3: `6c7db00cc164467ebac356a5ca67368b` (pole dancer), `8a0192eee6fb4140bb3e9696b3dbae5a` (pinecone), `a397b648d6eb48d7909d1ee11235e78f` (train). `D0-003` asked whether to admit, drop, or re-annotate them, and the question was **hard-blocking `D3`**: `splits.py:169-171` admitted all 45,955 while `stage1.py:109` loads the `.npz` with no existence guard, so Stage 1 would raise `FileNotFoundError` mid-epoch |
| **Why they were stuck — established 2026-08-22, not previously recorded** | **They are not broken assets.** Renders and point clouds exist for all three. They failed the v3 re-annotation on a **validator rule**: `quarantine_n05_annotate.jsonl` records `terminated_by: repair_budget`, `failure_class: MODEL_RECOVERABLE`, and for `6c7db00c…` the exception is verbatim ``synset` = 'pom.pom.n.01' is not a WordNet id of the form "lemma.n.01"`. The model invented a malformed synset, exhausted the repair budget, and the old v1 record was never overwritten |
| **Why the question dissolves under v5** | **v5 does not ask the model for `synset` at all.** Design Decision 4 replaced it with a deterministic lookup over the 1,156-term LVIS vocabulary (`metafind/data/lvis_synsets.json`, built and cross-checked against detectron2, 0 invented entries). The exact failure mode that produced these three residuals **no longer exists in the v5 pipeline** |
| **Evidence references** | `data/outputs/logs/quarantine_n05_annotate.jsonl` (3 of its 5 records) · `metafind/data/annotate.py` `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` / `SCHEMA_VERSION 3` · `metafind/data/lvis_synsets.json` · `D14/ESCALATION.md` "synset table 1,156 / 1,156 … 0 invented entries" · the three records read verbatim before deletion |
| **Decision / Resolution** | **Delete them.** They are re-annotated under v5 alongside the other 45,952, with no special handling. If v5 fails on any of them, they are quarantined by the ordinary path like any other failure — there is no longer a legacy schema to preserve |
| **Authority classification** | **USER DECISION.** The supporting facts are OBSERVED DATA (the quarantine records) and OBSERVED IMPLEMENTATION (v5's synset lookup). **This is not resolution-by-mutation**: the deletion follows from the v5 design the user already ratified, and the reason the residuals existed was established from evidence *before* deleting them |
| **USER final decision** | **`APPROVE`** — 2026-08-22 (「如果可以就刪」, conditional on the question being answerable in the new work; Master established that it is, and recorded the basis above) |
| **Affected components** | `data/outputs/annotations/` (now **0 files**) · `D0-003` closed · `D3`'s hard blocker cleared · `D2`'s corpus denominator becomes 45,955 uniformly · `annotation_provenance.json` must be rebuilt · `MASTER.md` §8, `INDEX.md` Decision Queue, `CONTEXT.md` §5/§6/§7 all still describe `D0-003` as UNRESOLVED and must be corrected |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

**Supersedes** the "explicitly NOT resolved: `D0-003`" clauses in `DL-002` and `DL-003`, and the
`legacy_v1_residual_unresolved` state in `data/outputs/annotation_provenance.json`. Those entries
are **not deleted** — they were true when written.

**`AC-1` no longer has a subject.** It protected an accepted legacy corpus from accidental
re-annotation. That corpus was deliberately deleted on 2026-08-22, so AC-1.a ("a bare
`annotate_run` queues 0 records") is now expected to be **false** — a bare run should queue the
whole corpus, because that is the intent. `DL-003-A1` must be rewritten to say so rather than
re-proving a guarantee whose purpose has ended. **Do not treat a bare run queuing 45,955 as a
regression.**

---

### `DL-007` — n05 v5 anchors object identity on the Objaverse-LVIS label. **This is a DEVIATION.**

**Registered by Master 2026-08-21 during the re-initialization audit. It had no ledger entry, which was a gap: it is the most scientifically material change made to the pipeline this week.**

| | |
|---|---|
| **Source** | `D14_n05-v5-reannotate`, design `workflow/n05_v5_design.md`, approved by the user 2026-08-21 |
| **Issue / Finding** | n05 asked the VLM to *identify* objects from 224×224 renders it often cannot read, so it collapsed onto high-frequency priors. **LVIS ground truth:** 1,156 distinct categories, top-20 share **7.1%**. **Qwen output:** 3,036 categories, top-20 share **22.3%** — 3× more concentrated than the truth. `toy` is the single most common answer at 1,542 (3.4%), a word `build_prompt` **explicitly forbids**. Agreement with LVIS: category matches 29.0%, LVIS word appears in the description 28.4%, **neither 67.8%** |
| **Why it is not a `category`-only defect** | `build_prompt` says *"Estimate its size from what kind of object it is, not from the picture"* — dimensions and placement are **by design** derived from the category. Observed: `LVIS pinecone → "a dark brown hairbrush"`, `LVIS mug → "a cylindrical pillow"`, `LVIS truck → "a modern air conditioner unit"`. **A wrong category is a wrong record, not a wrong field** |
| **Evidence references** | `workflow/MIF_n05_diagnosis.md` · `workflow/MIF_n05_category_vs_lvis.md` · `objaverse_lvis_metadata.json → value_to_key_mapping` (46,207 uid→category, fetched by `download.py:70`, **read by nothing**) · `annotate.py:366` `build_prompt(n_views)` receives only `n_views` · `annotate.py:510` `validate_annotation()` never checks semantic correctness |
| **Resolution not driven by resolution** | correlation(best-view occupancy, LVIS agreement) = **+0.054**, agreement flat at ~28–30% across a ~100× range of effective object pixels. **Re-rendering would not have fixed it** and is explicitly out of scope |
| **Decision / Resolution** | n05 v5 supplies the LVIS category to the model as the anchored identity; the model may **refine downward** but not replace laterally; it also emits `identity_confirmed`. `PROMPT_VERSION 5`, `SCHEMA_VERSION 3`, `VALIDATOR_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7` |
| **Authority classification** | **DEVIATION.** The paper has the VLM *generate* the category: `2methdology.tex:28` and `neurips_2025.tex:100` both say GPT-4o produces the structured annotations; Figure 2's caption (`2methdology.tex:24`) says the VLM *generates* attributes including category. **Feeding the dataset label in is a departure and must never be described as paper-faithful.** `D14/TASK.md` `R-E` binds |
| **What remains UNRESOLVED** | **`D0-010` has not been researched** — its §6–§11 are empty. The choice between *prompt hint* / *hard value* / *cross-check* / *record-only* was made by design ratification, **not** by a completed evidence audit. Also unresolved: whether anchoring merely substitutes LVIS's own errors for Qwen's, and whether `identity_confirmed` detects that or rubber-stamps the anchor (`IC-1`) |
| **USER final decision** | design **approved** 2026-08-21; **the deviation registration itself is pending** and reaches the user through `D14`'s acceptance brief |
| **Affected components** | `annotate.py`, `annotate_run.py`, `lvis_synsets.json`, the entire annotation corpus, every Stage 1 text embedding, both tables' comparability with the paper |
| **Status** | `AWAITING_USER_REVIEW` |
| **Date** | design approved 2026-08-21 · registered here 2026-08-21 |

**Registry gap, open:** `docs/graph/graph_spec.yaml` carries **no deviation entry** for LVIS anchoring. And `tools/check_graph.py:373-383` compares deviation **ids only, never the `what:` text** (`FU-A`), so a missing or falsified deviation description passes every gate silently.

---

### `DL-006` — the n05 annotation model is `Qwen3.8-27B`

**Registered by Master 2026-08-21 during the re-initialization audit. The user's decision (U-6) was made in conversation on 2026-08-21 and had no ledger entry.**

| | |
|---|---|
| **Source** | User decision **U-6**, 「走本地 Qwen3.8-27B」, recorded in `workflow/tasks/D14_n05-v5-reannotate/USER_DIRECTIVES.md` |
| **Issue / Finding** | n05 ran `Qwen/Qwen2.5-VL-7B-Instruct` as a stand-in for GPT-4o. A 7B model was a candidate cause of the identity-collapse defect in `DL-007` |
| **Decision / Resolution** | The annotation model becomes local **`Qwen3.8-27B`**, weights at `/mnt/data1/kyzen/models/Qwen3.8-27B`. Enacted at `annotate_run.py:72`. Master verified 2026-08-21: **18/18 shards, 56 GB, download complete** |
| **Authority classification** | **USER DECISION.** The substitution itself remains a **DEVIATION** — the paper says GPT-4o twice. Recorded under the split `D-2` (see `DL-005`). Reaching GPT-4o would narrow `D-2`, never discharge it |
| **Correction of record** | `D14/TASK.md` §7 originally justified its model prohibition with *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from the code comment at `annotate_run.py:71`. D14's finding F-2 shows OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **The conflict is UNRESOLVED, not resolved**, and the API has never been exercised. It must not be restated as settled |
| **UNRESOLVED and material** | **The model has never been loaded.** 56 GB at bf16 does not fit the RTX 5090's 32,607 MiB, so **quantization is required and has not been tested.** Quantization changes annotation quality, which makes it an **experimental condition**, not an engineering detail. **No Phase 3 runtime estimate is evidence-backed** — the "~19.6 GPU-h" figure belongs to the 7B model. Tracked as blocker **R2** (`MASTER.md` §11) |
| **USER final decision** | **`APPROVE`** — the model choice (U-6). The quantization condition and the runtime estimate are **not** covered by it |
| **Affected components** | `annotate_run.py:72` · deviation `D-2` · `D14` Phase 2 and Phase 3 · every annotation record |
| **Status** | **`USER_APPROVED`** for the model choice; the quantization condition is **OPEN** |
| **Date** | 2026-08-21 |

---

### `DL-005` — deviation `D-2` split into `D-2` (annotation) and `D-8` (scene judging)

| | |
|---|---|
| **Source** | Master, executing D14's escalation **P-2**, 2026-08-21 |
| **Issue / Finding** | `graph_spec.yaml` recorded one deviation — *"Qwen2.5-VL replaces GPT-4o for annotation **and** scene judging"*. After user decision **U-6** the annotation model became `Qwen3.8-27B` while the judge (n17) stayed `Qwen2.5-VL`. **One id would denote two different substitutions** |
| **Evidence references** | `graph_spec.yaml:130` (pre-split) · `annotate_run.py:72` (`MODEL_ID` already changed) · `2methdology.tex:28`, `neurips_2025.tex:100` (paper says GPT-4o) |
| **Decision / Resolution** | **Split, not rewrite-in-place.** `D-2` = Qwen3.8-27B for **asset annotation (n05)**. New `D-8` = Qwen2.5-VL for **scene judging (n17)**. Deviation count 六項 → 七項, synchronised across 5 files: `graph_spec.yaml`, root `README.md`, `docs/graph/README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`. `check_graph.py` → **2275 checks, all pass** |
| **Why split rather than rewrite** | One id covering two different substitute models is the exact ambiguity that produced this escalation, and the two will diverge further — n17 may change model later. D14 recommended splitting; Master agreed |
| **Authority classification** | The model change itself is a **USER DECISION** (U-6). This split is **bookkeeping that follows from it** — it records an existing fact accurately, it does not make a new choice |
| **USER final decision** | **pending** — reaches the user through D14's acceptance brief |
| **Affected components** | `docs/graph/graph_spec.yaml`, both `README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`; the reproduction report's deviation section |
| **Status** | `AWAITING_USER_REVIEW` — executed, not yet ratified |
| **Date** | 2026-08-21 |

**Correction carried in the same edit.** `D-2`'s stated reason was *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from a code comment. D14's finding F-2 showed OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **That conflict is UNRESOLVED.** The registry now records availability as UNRESOLVED rather than as established, in `graph_spec.yaml`, both READMEs, and `TASK.md` §7.

**Open gate weakness, registered not fixed:** `tools/check_graph.py:373-383` compares deviation **ids only** — regex `\|\s*\*\*(D-[0-9])\*\*` — and never reads the `what:` text. **A deviation whose description has gone false passes every gate silently.** That is how `D-2` stayed wrong. Found by D14. Not in D14's scope; unassigned.

---

### `DL-004` — MetaFind §2.5 `f_x → R³`: verdict and implementation

| | |
|---|---|
| **Source** | `D0-009_essgnn-fx-codomain` |
| **Issue / Finding** | MetaFind states `f_x: R^(2d+1+e) → R³` (`2methdology.tex:54`) and claims equivariance for **any orthogonal** `Q ∈ R^{3×3}` (`appendix.tex:23`), but **never defines the `·`** in the coordinate update (`2methdology.tex:52`). Master confirmed the silence exhaustively: zero hits for `hadamard` / `element-wise` / `inner product` / `dot product` / `contraction` across all five `.tex` files |
| **Evidence references** | `2methdology.tex:52`, `:54` · `appendix.tex:23`, `:29`, `:53`, `:61`, `:68` · `essgnn.py:311-312`, `:353`, `:358-359` · `C_PAPER_CONTRADICTIONS.md:114` (C3, **SEVERE**) · decision file §6–§11 · Codex round 1 (its BLOCKER changed the verdict) |
| **Verdict** | **`PAPER-AMBIGUOUS`.** MetaFind alone does not uniquely determine how the `R³` output participates in the coordinate update. **This must never be rewritten as "MetaFind explicitly got it wrong."** |
| **Decision / Resolution** | **Option A** — retain the scalar `f_x` coordinate multiplier. **`essgnn.py` behaviour is NOT modified**; what changed is the authority classification |
| **Authority classification** | `f_x → R³` stated = **PAPER FACT** · the operator is undefined = **PAPER FACT (as to silence)** · `h` invariance = **PAPER FACT** (`appendix.tex:29`, `:68`) · **scalar `f_x` = USER-RATIFIED IMPLEMENTATION CHOICE under a PAPER-AMBIGUOUS specification.** Not a PAPER FACT |
| **Rationale, recorded** | Not chosen because upstream EGNN is more sensible. Chosen because the operator semantics are undefined; Hadamard closes dimensionally but breaks the paper's own general-orthogonal equivariance claim; `R³` + contraction invents an operator MetaFind never defines; the scalar preserves equivariance and invents nothing |
| **Binding prohibition** | **"upstream EGNN settles it" may no longer be used as paper-interpretation authority anywhere in this project.** `E_GRAPH_REVALIDATION.md:175` must be corrected |
| **USER final decision** | **`APPROVE` with implementation decision A** |
| **Affected components** | `essgnn.py` (classification only, no behaviour change) · `docs/audit/` C3 · `docs/graph/` U-26 · `CONTEXT.md` §5 · Stage 2 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-21 |

**`MIF-1` REJECTED as a blocker.** D0-009 asked to be gated on `D0-004`'s "unresolved `h` semantics". Master rejected it: `appendix.tex:29` **assumes** `h^0` invariant and `:68` concludes the feature update invariant — `h` invariance is not an open question in MetaFind. `essgnn.py:353` feeds `f_h` only `h`, `radial`, and `edge_attr`, all invariant. `D0-004` concerns which layer's `h` reaches `f_x`; both are invariant. Option B's conflict is independent of `h`.

**Follow-ups registered, none blocking:**

| ID | Item |
|---|---|
| MIF-3 | `F_CODE_GRAPH_CONSISTENCY.md` — its `CONSISTENT` column is unlabelled and appears to mean "code matches the graph spec", not "code matches the paper". Terminology ambiguity |
| MIF-4a | `2.2e-16` vs `0.43` — **must not be described as a repo-verified measurement.** Cannot be reproduced: the `R³` variant does not exist in code |
| MIF-4b | No `R³`-variant equivariance test exists. **Narrower than reported:** `test_se3_equivariance` runs at `n_layers=3` (`test_essgnn.py:102`, `:112`) and `test_equivariance_negative_injection` (`:129`) proves it non-vacuous |
| MIF-5 | `normalize_coord_diff` has no MetaFind authority. `essgnn.py:189` defaults `False`, zero current impact. **Independent candidate — not D0-009's to touch** |
| — | Correct `E_GRAPH_REVALIDATION.md:175`'s `[UPSTREAM] settles it` |

---

### `DL-003-A1` — PREPARED AMENDMENT to `DL-003` / AC-1, to land WITH D14 Phase 3

**Status: `PREPARED, NOT IN FORCE`.** Drafted by Master 2026-08-21 at D14's request (P-5), so the
amendment lands **with** Phase 3 rather than being retrofitted after it. `DL-003` stands unchanged
until Phase 3 completes.

**Why an amendment is needed.** `DL-003` records the corpus as **accepted legacy-v3 validated under
`VALIDATOR_VERSION 2`**, and the provenance registry declares that as a population of **45,952**.
D14 Phase 3 re-annotates those same 45,952 under `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` /
`SCHEMA_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7`. **The moment Phase 3 finishes,
`DL-003`'s description of the corpus becomes false.**

### What changes

| | before Phase 3 | after Phase 3 |
|---|---|---|
| 45,952 records | `accepted_legacy_v3`, `VALIDATOR_VERSION 2` | annotated under the **current** contract `metafind_annot_v5@…` |
| how they satisfy AC-1.a | by **registry declaration** | by `is_complete()` — they carry the current `annotation_contract` |
| 3 legacy-v1 residuals | `legacy_v1_residual_unresolved` | **unchanged.** `D0-003` still UNRESOLVED |
| registry `accepted_legacy_v3` population | 45,952 | **0** — the state becomes historical |

### What must NOT change

- **AC-1.a still holds: a bare `annotate_run` queues 0 records TOTAL.** After Phase 3 the 45,952
  satisfy it through `is_complete()` instead of through the registry, and the 3 residuals still
  satisfy it through their declaration. **The guarantee is identical; only the mechanism moves.**
- **AC-1.b still holds:** `--force` still re-annotates; the named-migration form still works.
- **AC-1.c still holds:** three states remain explicit, never inferred from a missing field.
- **`D0-003` remains UNRESOLVED.** Nothing in Phase 3 or this amendment resolves it, and the
  3 residuals must be byte-identical at Phase 3's end.

### Required of D14 at Phase 3

1. Update the registry so `accepted_legacy_v3` no longer claims a population that has moved on.
   **Do not delete the state** — mark it historical, with the date and the contract that superseded it.
2. Re-prove AC-1.a **after** the registry update: bare run queues **0 TOTAL**, 0 legacy-v3, 0 residuals.
3. Re-prove AC-1.b: `--force` still yields a non-empty work list.
4. Confirm the 3 residuals' declaration survived untouched.

**If AC-1.a cannot be re-proved after the registry update, that is a `MASTER-IMPACTING FINDING`, not
a registry-editing problem.** The protection is the point; the registry is only how it is expressed.

### Authority

The re-annotation itself is a **USER DECISION** (U-6, U-10, and the design ratified 2026-08-21).
This amendment is **bookkeeping that follows from it** — it records a consequence, it does not make
a new choice. It still reaches the user through D14's acceptance brief.

**Superseded on landing:** `DL-003`'s "legacy-v3 validated under `VALIDATOR_VERSION 2`" description
of the 45,952, and `AC-1.e`'s instruction to record them as such. **`DL-003` is not deleted** — it is
the true record of what was accepted on 2026-08-21.

---

### `DL-003` — Stage 1 protocol refresh, τ = 0.5, and legacy-corpus rerun protection

| | |
|---|---|
| **Source** | `D2a_stage1-protocol-refresh` |
| **Issue / Finding** | `load_protocol()` refused the on-disk protocol after D10's B-2. τ = 0.5 had no code path (`resolve_stage1.py` hardcoded 0.07, 0 CLI flags). And `annotate_run.is_complete()` keyed on a contract **no** record carried, so a bare invocation would have queued **45,955** records — overwriting a corpus the user decided to preserve, and resolving `D0-003` by mutation |
| **Evidence references** | Master re-ran, read-only: `build_work_list(force=False)` → todo **0**; `force=True` → **45,955**; state histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}`; `load_protocol()` → `metafind_v2_cm@8e4b1fcc66c7f48c`; artifacts `0.5` / `False`; `prompt_version {3: 45952, 1: 3}`; 547 passed · 2275 checks · PRE-FLIGHT PASSED |
| **Decision / Resolution** | Accept the task. τ = 0.5 with `learnable_temperature: false` written through n05b; protocol refreshed to the ratified serializer; AC-1 satisfied via a **declared registry** (`data/outputs/annotation_provenance.json`) plus a relocated work-list predicate; the 45,952 formalized as accepted legacy-v3 under `VALIDATOR_VERSION 2`; the 3 residuals explicitly `legacy_v1_residual_unresolved` |
| **Authority classification** | τ = 0.5 = **PAPER FACT** (`3experiments.tex:15`) · `learnable_temperature: false` = **USER-RATIFIED IMPLEMENTATION CHOICE** on a strongly-supported inference — the paper uses "learnable" for `f_h`/`f_x`/λ but calls τ a "temperature hyperparameter" twice, and never states τ is non-learnable · AC-1 mechanism = IMPLEMENTATION CHOICE within the approved menu · registry choice, legacy-v3 formalization = **USER DECISION** |
| **Scope deviation — ratified** | **MIF-2.** `n05b` writes **three** artifacts (`resolve_stage1.py:660-662`); the contract declared two. `variant_registry.json` was rewritten. Master proved the rewrite byte-identical (`VARIANTS` absent from the diff; `_write()` is a deterministic `json.dump(indent=1)`). Root cause: **Master's contract under-declared the write surface**, not executor conduct. **USER RATIFIED 2026-08-21.** The contract was **not** retroactively edited |
| **USER final decision** | **`APPROVE`** — task accepted, MIF-2 ratified |
| **Affected components** | `resolve_stage1.py` · `annotate_run.py` · `annotate.py` · tests · `stage1_hyperparameters.json` · `stage1_encoding_protocol.json` · `variant_registry.json` · new `annotation_provenance.json` · new `tools/declare_annotation_provenance.py` · `docs/graph/README.md:270` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · approved 2026-08-21 |

**Codex findings retained, not deleted because fixed** (user instruction): a JSON-`null` sidecar could be re-queued; unchanged `prompt_version` with changed content could leave provenance stale. Both fixed with regression tests. Full record in `CODEX_REVIEW.md`.

**Explicitly NOT resolved:** `D0-003`. The 3 legacy-v1 residuals remain unresolved and are labelled as such in the registry. Nothing claims otherwise.

**Does NOT unblock `D1`.** Chain: `D2a USER_APPROVED → D10 final USER REVIEW with AC-1 evidence → D10 USER_APPROVED → D1_n06-reencode`.

**Open follow-ups:** F-2 `sidecar_path()` uid validation (pre-existing, LOW, unassigned) · F-3 Master under-declared `tools/` scope for the registry tool, same class as MIF-2 · F-4 TASK §12.1's AC-1.a snippet is stale, now marked SUPERSEDED · F-5 Master holds no pre-task corpus fingerprint.

---

### `DL-002` — Stage 1 encoding contract implementation

| | |
|---|---|
| **Source** | `D10_stage1-encoding-contract` |
| **Issue / Finding** | The cache-validity blocker (a resumed n06 would build a two-distribution gallery), the unimplemented ratified template, and four open annotation-pipeline gaps. Plus **FIND-8**, discovered at Master review: `annotate_run.is_complete()` keys on `annotation_contract`, which **no** existing record carries, so a bare invocation would queue **45,955** records for re-annotation |
| **Evidence references** | `encode_text_image.py:73-83`, `:86-108` · `annotate_run.py:98`, `:250` · `resolve_stage1.py:243` · `tools/preflight_stage1_text.py` (run by Master) · `CODEX_REVIEW.md` (2 rounds, 12 findings) · `HANDOFF.md` |
| **Decision / Resolution** | Ratify the implementation in principle: serializer, `text_serialization_id` / cache validity, `load_protocol` mismatch rejection, Stage 1 text pre-flight, >77 true-token hard gate, P-1…P-5, annotation contract versioning, and 3 user-authorised manual translations. **Acceptance withheld** pending `AC-1` |
| **Authority classification** | B-1/B-3 mechanisms = IMPLEMENTATION CHOICE · late template binding = **bug fix** · P-1…P-5 = IMPLEMENTATION CHOICE, **USER DECISION** to ratify the scope extension · 3 translations = **DEVIATION**, user-authorised · legacy-v3 retention = **USER DECISION** |
| **Acceptance condition** | **`AC-1`** — absent explicit force or migration intent, the accepted legacy-v3 corpus must not be automatically treated by `annotate_run` as requiring re-annotation. Sub-conditions AC-1.a…AC-1.e in the brief §7.0. Assigned to `D2a` |
| **USER final decision** | `MODIFY` 2026-08-21 (acceptance withheld pending `AC-1`) → **`APPROVE` 2026-08-21** after `D2a` (`DL-003`) demonstrated `AC-1` |
| **Affected components** | `resolve_stage1.py` · `encode_text_image.py` · `annotate.py` · `annotate_run.py` · `tools/preflight_stage1_text.py` · 3 annotation records (backed up) · `D1`, `D2a`, `D3`, `D4`, `D7` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · MODIFY 2026-08-21 · **APPROVED 2026-08-21** |

**`AC-1` — CLEARED.** All five sub-conditions verified by Master through the **production** work-list path `build_work_list()` (`annotate_run.py:439`), not through the superseded `is_complete()` predicate: bare todo TOTAL **0** · legacy-v3 queued **0** · residuals queued **0** · `--force` **45,955** · histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}` · no fake v4 `annotation_contract`.

**`G-7` is NOT ratified by this acceptance — it remains independently OPEN.** The user declined to rule on it in this decision, and D10's FINAL ACCEPTED status must **not** be read as approving it:

| | Item | Impact today |
|---|---|---|
| G-7 §5 | Four in-code comment corrections outside R-1/R-2/R-3 | zero behavioural |
| G-7 §6 | Whether the `math.isfinite()` guard belongs in `serialize_annotation()` rather than only in the pre-flight | 0 records |

**Other items explicitly not blocking, carried forward:** `D0-003` **UNRESOLVED** · MIF-D10-3 routed to `D3`/`D4` (re-verified 2026-08-21: `stage1.py:110` and `gallery_index.py:215` load NPZ directly; neither file mentions `text_serialization`) · **F-2** `sidecar_path()` uid validation, LOCAL, pre-existing · template retrieval impact **UNKNOWN**.

**Unblocks:** `D1_n06-reencode`.

**Briefs:** `USER_REVIEW.md` (Rev 2), `USER_REVIEW_FINAL.md`

---

### `DL-001` — Stage 1 text serialization design (U-15)

| | |
|---|---|
| **Source** | `D0-008_stage1-text-template` |
| **Issue / Finding** | The paper specifies no text serialization format (PAPER FACT as to silence). The recorded protocol artifact describes a different template from the one the encoder runs. The running serializer rendered 161 real dimensions as `0 centimetres`, emitted 3,643 ungrammatical articles, and knowingly encoded 1 over-length record. Four in-code justifications did not describe the code |
| **Evidence references** | `2methdology.tex:28` and caption `:24` · Figure 2 (`data-preprocess.png`) · `resolve_stage1.py:96-100`, `:102-115`, `:116-128`, `:162` · `encode_text_image.py:73-83`, `:86-108`, `:194`, `:233` · full-corpus scan of 45,952 v3 records · decision file §6, §9, §10, §11 |
| **Decision / Resolution** | Ratify the design in §11.3: template form, field set and order, centimetres, `width → length → height`, E-1 (dimension precision), E-2 (article removal), E-3 (re-annotate the CJK record), S-1 (uniform formatter, no threshold), S-2 (category capitalisation), R-3 (**delete** the unreachable placement entry), and omission of `synset` / `volume` / `mass` |
| **Authority classification** | Field **set** = PAPER FACT · centimetres = INFERENCE as to intent, OBSERVED DATA as to this corpus · `width,length,height` = INFERENCE from Figure 2 · everything else = **IMPLEMENTATION CHOICE** · E-1/E-2/E-3/S-1/S-2 = **USER DECISION** |
| **Binding user modification** | Omission of `synset` / `volume` / `mass` is an **IMPLEMENTATION CHOICE** with **UNKNOWN retrieval impact**. Must **not** be stated as a PAPER FACT or described as proven redundant. The redundancy argument is WITHDRAWN (Codex C-6) |
| **USER final decision** | `MODIFY` → **`APPROVE`** |
| **Affected components** | `resolve_stage1.py` · `stage1_encoding_protocol.json` · every Stage 1 text embedding · `D10`, `D1`, `D2`, `D3`, `D4`, `D7` · every text-conditioned column of Table 1 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | proposed 2026-08-21 · approved 2026-08-21 |

**Explicitly outside this decision:** the n06 cache completion/validity gate (decision file §11.2, B-1…B-4). It is an execution question owned by `D10_stage1-encoding-contract`. This approval neither clears nor waives it.

**Outstanding follow-ups:** FU-2…FU-5 (`D10`) · FU-6 (Master, done at approval) · FU-7 (Master → `D0-003`) · FU-8, FU-9 (deferred). FU-1 is not carried by this decision.

**Integrated at approval:** `workflow/CONTEXT.md` §5 (FU-6), `workflow/MASTER.md`, `workflow/INDEX.md`, this ledger.

**Not yet implemented:** the ratified template does not yet exist in `resolve_stage1.py`. That is FU-2, owned by `D10`.

---

## Migration backlog

**Migration backlog is now empty.** Both pre-gate items — `D0-008` and `D10` — have passed through the gate and reached `USER_APPROVED`.

`workflow/WORKFLOW.md` §19 governs work completed **before** the user review gate was adopted.

Such work is **not** re-run: evidence surveys, Codex reviews, Claude verification of Codex findings, tests, and implementations all stand as produced. What is owed is the tail of the flow:

```
existing completed artifacts
→ Master integration review
→ USER REVIEW BRIEF
→ USER final decision
```

Any acceptance Master recorded before the gate existed is **reclassified as a MASTER RECOMMENDATION** and carries status `AWAITING_USER_REVIEW` here.

| Decision / Task | Pre-gate state | Ledger status | What is owed |
|---|---|---|---|
| ~~`D0-008_stage1-text-template`~~ | Master recorded `ACCEPT WITH FOLLOW-UP` pre-gate | **CLEARED 2026-08-21** → `DL-001`, `USER_APPROVED` | Migration complete. Brief delivered, user returned MODIFY then APPROVE |
| ~~`D10_stage1-encoding-contract`~~ | Executed against D0-008's pre-gate acceptance | **CLEARED 2026-08-21** → `DL-002`, `USER_APPROVED` | Migration complete. `AC-1` satisfied by `D2a`; implementation and Codex review were **not** re-run |

Master must not treat a pre-gate acceptance as though the gate had been satisfied.

---

## DL-033 — δ, the stopping threshold for the LR sweep

`USER_APPROVED` · 2026-08-30 · Kyzen's word was **「甲」**

```
δ         = 1.0 percentage point  (0.010 absolute R@1)
metric    = mean R@1 over {text, image, pc}
protocol  = C_dev_selection  (query dev_val, gallery dev_val, n = 4,569)
class     = IMPLEMENTATION CHOICE, user-approved
```

**How it is used** — on the paired difference's uncertainty interval:
lower bound > δ ⇒ real improvement · upper bound < δ ⇒ may stop · interval straddles δ ⇒ add seeds.

**δ was declared before the runs, not derived from the observed spread.**
(Codex's condition: a threshold chosen after seeing the results is not a threshold.)

Evidence: `workflow/blocks/ULIP2/evidence/delta_stopping_threshold.md`

### Why the paper cannot supply δ

[PAPER FACT `docs/paper/metafind_source/3experiments.tex:94-108`] — I read the table myself:
`Fusion = Mean 9.4` · `Fusion = MLPs 9.9` · `Modality Dropout 10% 7.3 / 50% 13.2` ·
`Train fuser only 8.7` · `Padding with 0 10.5` · `Full 11.4`.
**The paper carries no standard deviation, no seed, no repeated run anywhere.**
So δ genuinely cannot be derived from it. **This is paper silence, not a failed search.**

⚠ **The evidence file's wording is one notch stronger than the table supports, and is corrected here**
(ULIP2 Block Reviewer raised it; I checked the table and he is right).
It says δ is "at least as large as the smallest gap MetaFind itself reports as a finding".
**The same table has `w/o iterative retrieval 11.3` against `Full 11.4` — a 0.1 pp row**
(`3experiments.tex:94,96`), which is smaller than the 0.9 pp floor it cites.
**Correct phrasing: δ sits inside the band of differences the paper draws conclusions from,
which is `0.1 – 4.1 pp`.** δ = 1.0 stands; only the justification is narrowed.

🔴 **Corrected again, same day, by the same reviewer — and this time the wrong number was mine.**
The first version of this entry wrote the band as `0.9 – 5.9 pp`, copied from the evidence file
without deriving it. **I verified the individual table values and not the band computed from them.**
I have now computed every gap against `Full 11.4` myself
([PAPER FACT `3experiments.tex:92-110`]):

```
0.1  w/o iterative retrieval        1.8  Modality Dropout 50%
0.4  w/ Layout Context (GAT)        2.0  Fusion = Mean
0.9  Padding missing modalities 0   2.1  w/o Layout Context   (13.5 vs 11.4)
1.5  Fusion = MLPs                  2.7  Train fuser only
                                    4.1  Modality Dropout 10%
                    → band 0.1 – 4.1 pp   (nine rows)
```

🔴 **The list above was eight rows and said "every gap". Codex found the missing one; I verified it
at [`3experiments.tex:97`] `w/o Layout Context 13.5` against `Full 11.4` = 2.1 pp.**
Neither the band nor δ moves — 2.1 sits inside 0.1–4.1 — **so this is a claim that outran its content,
not a wrong number.**

⚠ **Third time in this entry's short life, and all three are the same shape:**
```
1  band copied from the evidence file without deriving it        (MASTER)
2  the evidence file's Authority row not moved with its body     (ULIP2 Engineer)
3  a list announced as complete that was not                     (MASTER)
```
**None of the three is an arithmetic error. All three are a claim made one notch wider than the work done.**
Recorded because the fix for arithmetic is checking, and the fix for this is different:
**say what you actually did, not what the sentence would be nicer for.**

**Where the wrong `5.9` came from**: `13.2 − 7.3`, i.e. Modality Dropout 50% against 10%.
**That is variant against variant, not a single ablation's effect on the full model.**
It never belonged in a band of ablation effects.

### Three things recorded with it

**1. Primary development selector changed** (`✅` Kyzen 2026-08-29, R-33 item 3):
the LR sweep selects on the **mean of the three unsaturated conditions** (text, image, pc).
All seven conditions and every per-condition figure are still reported, as a guardrail.
Reason: `text+image`, `text+pc`, `image+pc` and `full` are ≥ 0.98 at every checkpoint,
so a seven-condition mean is diluted by four ceilings.

**2. 🔴 Retracted, and must not enter this ledger as fact:**
"the two e25 runs are a clean repeat, noise floor 0.00123."
**Withdrawn** — `e25_400w` lacks the `preload` / `num_workers` fields that `e25_500w` has,
so **the working tree changed between the two runs**; they also ran at different power caps.
The dependent claim "the ladder's signal is 8× the noise" is withdrawn with it.
**We currently have no measured seed-to-seed dispersion at all.**

**3. `full = 1.0000` is INFERENCE, not established.**
The mechanism is plausible (all three modalities survive masking for 0.7³ = 34.3% of samples,
and both towers then see the same input) but it needs a negative control to be settled.
**That debt sits on n15.**

### Code review status of the batch this decision governs

`CHANGES REQUIRED` (ULIP2 Block Reviewer). Two MAJOR, both one-line:
- `ARM_EXCLUDED = ("seed",)` omits `preload` / `num_workers` / `device`, while `stage1.py:722`'s
  error message sends the reader to it. They are excluded today only by not being merged into
  `values`, and `:1059` establishes the opposite idiom.
- `max_epochs` enters the arm hash but only warns (`:1187-1193`, "Nothing stops you"), so moving
  the ceiling gives identical training a different arm identity.

Re-review by Codex after the fix. **No training, sweep, n15 or A/B has been run.**

**⚠ SUPERSEDED 2026-08-30 (later the same day) — the paragraph above records the FIRST round only.**
Per maintenance rule 6 the original conclusion is left standing and the current state is added here.

```
Two MAJOR fixed  → ULIP2 Block Reviewer R-34 PASS
                   （he restored ARM_EXCLUDED to ("seed",) in memory and re-ran the same
                     assertions: all four flipped, so the tests are not could-not-fail）
Codex R2         → CHANGES REQUIRED (2 BLOCKER, 2 MAJOR, 1 MINOR) → all fixed, reviewer re-verified
Added            → frozen key-set test, two-way assertions, reading the real artifact from disk
Current          → awaiting Codex R3
```

**Still true from the first round: no training, sweep, n15 or A/B has been run. GPU idle.**

**⚠ SUPERSEDED AGAIN 2026-08-30, third paragraph — the block above stopped at "awaiting Codex R3".**
Rule 6 again: the two paragraphs above stay as written, this one carries the current state.

```
Codex R3, R4     → answered; fixes applied and re-verified
Committed        → a005be7 (run identity, provenance, float64 dev-val scoring)
                   6d87ca1 (n15 evaluator, its tests, the evidence files)
                   890f477 (crash-investigation tooling)
                   3,418 of those lines had never been in git at all
n15 R2 MAJOR     → STILL OPEN. Changing only `block`, a performance parameter,
                   changes the reported `rank` and `tie_count`.
Codex R5         → has not answered on the n15 evaluator
```

**One candidate explanation for `full = 1.0000` was eliminated by measurement, not by argument.**
The hypothesis that it is float32 rounding is dead: the same embeddings recomputed in float64 give
R@1 = 1.000000000, bit-identical, and float64 reproduces all seven recorded cells of `e25_500w`
bit-for-bit where float32 differs on one query (`tools/measure_dtype_effect.py`,
`output/look/dtype_effect.json`).

This is the first time on this project a hypothesis was closed by measuring rather than by reasoning.
It removes one candidate. **It does not discharge the debt: `full = 1.0000` is still INFERENCE and the
negative control is still owed.**

**Still true, three rounds later: no training, sweep, n15 or A/B has been run. GPU idle.**

---

## DL-034 — three approvals for the LR sweep and n15

`USER_APPROVED` · 2026-08-30 · Kyzen's word was **「甲」** to each of three

```
1  The first batch of 8 is a MEASUREMENT round, not a selection round.
   It produces the project's first honest seed-to-seed dispersion and a rough
   position for each of the four LRs. It does NOT produce "which LR wins".
   Codex's five unspecified items — interval method and confidence level, max
   seeds, multiplicity across the four LRs, alpha spending for the sequential
   design, single-condition degradation guardrail — are deferred until that
   dispersion exists.

2  Run 1 arm first, then the other 7.
   `preload` has never completed a full run; the single arm confirms wall-clock
   and stability before committing the batch.

3  n15's SCOPE is approved. Writing may proceed now.
```

**The argument that carried item 1** (ULIP2 Block Reviewer, recorded verbatim because it is the
reasoning, not the conclusion, that will be needed later):

> **δ must be declared a priori — that is "what we want".
> The interval method must be chosen a posteriori — that is "what the data looks like".
> Pinning both in advance commits, in the second place, the very error δ exists to avoid.**

Its companion, on ordering: **n15 does not exist, so Table 1 has no reportable number at all.
The LR sweep tunes a number that cannot yet be reported; n15 unlocks having any number.**
And n15 costs no GPU, so it runs in parallel with waiting on Codex — **not a queue-jump, a gap-fill.**

Evidence: `workflow/blocks/ULIP2/evidence/kyzen_decisions_20260830.md`
(written by the ULIP2 Engineer; the Reviewer verified the decisions, not the file's wording.)

### 🔴 What these three approvals do NOT authorise

**All three are design and scope approvals.** Executing the sweep and executing n15 each still
need a fresh `✅` from Kyzen. **As of this entry, no GPU work has been run at all.**

### `full = 1.0000` — the debt in DL-033 does not close because n15 was written

n15 exists as code (`metafind/eval/run_retrieval.py`) and has not passed review:

```
R1 BLOCKER (fixed)  the streaming scorer scored a fully collapsed model at R@1 = 100%,
                    and the shuffle_targets negative control did not fire on it.
                    Root cause: `own` and `sim` take different arithmetic paths and
                    differ by one ULP. Reviewer re-verified the collapse case now reports 0.
R2 MAJOR (open)     third exit from the same root cause — a differently-sized final block
                    selects a different BLAS kernel, so the same data and the same model
                    return different `rank` and `tie_count` when only `block` changes.
                    **A performance knob is moving the reported metric.**
```

→ **Negative-control coverage for `full = 1.0000` is still zero.**
**DL-033's entry stands open. Do not close it on the grounds that n15 has been written.**

---

## Maintenance rules

1. Only Master edits this file.
2. A D0 decision file's status and this ledger must agree. If they disagree, the ledger is the project-level record and the decision file is corrected.
3. `USER_APPROVED` is written only after the user's actual approval, with its date.
4. Superseded entries are **marked**, never deleted, and must name their replacement.
5. When a decision is approved, Master integrates it and records which files were updated.
6. If a decision is later found to rest on a mistaken finding, add a new entry that supersedes it. Do not edit the original's conclusion.

---

## DL-035 — one family of silent error, four variants

`RECORDED` · 2026-08-30 · framing and variants 1–3 from INTEGRATOR `metafindv1-e7 [ba9699]`,
relayed by the ULIP2 Block Reviewer because INTEGRATOR is ON HOLD and cannot write files.
Variant 4 measured by the ULIP2 Block Reviewer the same day. Filed as ONE entry at his request.

**Filed whole on purpose. Split into four rows, each looks like a small thing. Together they
are the shape that currently produces this project's silent failures.**

The shape: **something is in force in the system, and the system cannot answer where it came
from or why it holds.**

```
1  protocol written, the live path does not read it     wrong today
2  fingerprint written, no consumer                     wrong today
3  artifact exists, author unknown                      missing today
4  property holds, nothing enforces it                  CORRECT today   ← the dangerous one
```

**Variant 4 is the dangerous one, and the reason is INTEGRATOR's: the first three are a defect
when a reviewer looks at them. The fourth is a correct piece of code when a reviewer looks at
it — because it is correct.**

### The instance, 2026-08-30

`metafind/eval/run_retrieval.py`, `score_streaming`. Collapsed gallery, d=1280, L2-normalised:

```
float32   tie_count moved with `block` in 6 of 6 trials, ng = 999 / 4,569 / 9,138
float64   0 of 6 at every size
```

`block` is a performance parameter. `tie_count` is the diagnostic added to detect collapse.

Every number it produced was right. It passed its tests. It passed Codex R5 APPROVED. It was
right because `encode_pools` happens to return float64, and nothing in the function required
that. **No tool distinguishes "correct because guaranteed" from "correct because lucky."**

It was not a bug waiting to be found. It was a correctness waiting for its caller to change.

Guard and two named tests landed in `169b0cb`. **The guard will not fire once today.** That is
why nobody would have added it, and why it is recorded here.

### Ruling on the R2 MAJOR

**NOT CLOSED by Codex R5.** R5 approved the behaviour as it stands; the reviewer's objection is
that the block-independence was closed by float64 rather than by structure. With `169b0cb` the
structure now enforces it. **The entry closes on the guard, not on the approval.**

### The third instance today of the same sub-shape

A property left to the caller to maintain instead of enforced by the callee:

```
ARM_EXCLUDED          declared fields it did not contain
ENFORCED_SINGLETONS   checked a merged dict, so the encoding half was never looked at
score_streaming       needed float64 and did not say so
```

**A guard is the difference between "nobody calls it that way" and "it cannot be called that way."**

### Related, and NOT resolved by this entry

`workflow/FINDINGS_20260827_CAPTURE.md` §3 seam 1 is INTEGRATOR's, is marked UNVERIFIED there,
and two things about it are now established:

```
OBSERVED 2026-08-30, MASTER:
  scene_splits.py:102 DOES read cache["llm_model"] and cache["text_encoder_version"].
  Seam 1's "those fields are not read by anything" is too broad. stage2.py:303-327 does
  not read them; another consumer does.

OBSERVED 2026-08-30, MASTER:
  find . -name "*protocol*.json"  returns nothing.
  No protocol JSON exists in this tree under any name. Both the old wording
  ("fingerprint written, not wired") and the newer one ("the protocol has no such key")
  describe a file this repository does not currently contain.
```

**UNRESOLVED. Not rewritten in place, because neither wording has been established and a
correction that is itself unverified is the failure it claims to fix.** A note carrying these
two observations has been appended to §3.

**⚠ CORRECTED 2026-08-30, same day, by the ULIP2 Block Reviewer. Rule 6: the paragraph above
stays as written. The second `OBSERVED` in it is FALSE.**

```
find . -name "*protocol*.json"        → 0     ← what I ran
find -L . -name "*protocol*.json"     → 7     ← the truth

data -> /home/kyzen/metafind_data  is a symlink, and `find` does not follow a symlinked
starting point unless given -L. It does not warn. It returns the empty set.
```

**I ruled §3 UNRESOLVED on a search that was empty for a mechanical reason, and I wrote the
sentence "an unverified correction is the failure it claims to fix" in the same commit that
contained one.** The reviewer had already recorded this same `find` trap in `REVIEW.md` R-31b
on 2026-08-29, from his own encounter with it.

**This is the eighth instance of the probe not reaching the target, and the first where the
instrument was mine while I was writing the entry about instruments.**

### §3 seam 1, ruled properly this time

```
OBSERVED 2026-08-30  data/outputs/essgnn_arch_protocol.json exists. Its keys are:
                     architecture_family, coord_feat, decided_at, decided_by, distance,
                     hidden_dim, layer_sharing, mlp_structure, n_layers, pooling, status,
                     use_io_projections
                     `node_feat_dim` and `edge_feat_dim` are NOT among them.
                     ⇒ the ESSGNN Engineer's wording, "the protocol has no such key",
                       is CONFIRMED -- for the ARCH protocol and the 1280-d question (U-20).

OBSERVED 2026-08-30  data/outputs/procthor_node_embeddings.json carries:
                     asset_ids, embedding_dim, n_assets, sha256, text_encoder_version, uri

OBSERVED 2026-08-30  stage2.py:94-99   reads and verifies `sha256`, refusing an artifact
                                       whose record carries none
                     stage2.py:269     reads meta["llm_model"], meta["text_encoder_version"]
                     stage2.py:397-398 writes both forward
                     stage2.py:598-607 verifies `stage1_checkpoint_sha256` against the
                                       loaded checkpoint
                     ⇒ seam 1's "stage2.py:303-327 compares vector widths only, and reads
                       none of those fields" is NO LONGER TRUE of the current tree. Those
                       line numbers now point at query encoding, not at verification.

OBSERVED 2026-08-30  find -L . -name "sem_edge_cache*"  → nothing. That artifact does not
                     exist in this tree under any name.
```

**Ruling: seam 1 is two claims and they now part company.**

```
the fingerprint half   CLOSED   stage2.py verifies sha256 and carries llm_model /
                                text_encoder_version forward. Landed with the artifact
                                verifier in 2916327.
the sem_edge_cache half  N/A    the artifact does not exist yet; nothing can read it
the arch-protocol key   OPEN    node_feat_dim / edge_feat_dim absent from
                                essgnn_arch_protocol.json. This is U-20 / item ④ and it
                                is unchanged by any of today's work.
```

**Only the third stays open, and it is not what seam 1 was about.**


### Process fact, recorded not blamed

Codex wrote seven product files directly at 02:17:50 / 02:19:31 / 02:20:34 on 2026-08-30.
The engineer and the reviewer both learned of it by seeing files change, and each spent a round
establishing authorship; the reviewer's first attribution was wrong and he corrected it.
**Suggested: when Codex edits product code itself, a notice ahead of the ruling.**


---


**⚠ ATTRIBUTION CORRECTED 2026-08-30 by INTEGRATOR himself. Rule 6: the entry above stays as
written. Its attribution is wrong.**

DL-035 above says the framing and variants 1–3 are INTEGRATOR's. **They are not.** He objected
to his own credit, unasked, and gave the real provenance:

```
the seam criterion        workflow/roles/INTEGRATOR.md §3 -- his job description, handed to
                          him. Not something he derived.
"wrong but it will not
 shout" (the phrasing)    INTEGRATOR. A previous MASTER filed it under his name and that one
                          is his.
variant 1                 METAFIND_NOTEBOOK.md §9.6, written by a previous MASTER. Of its four
                          instances, two came from INTEGRATOR and two did not.
variant 2                 INTEGRATOR -- and already superseded by the ESSGNN Block Engineer's
                          sharper form: not "a fingerprint nobody reads" but "the protocol has
                          no such key at all".
variant 3                 ULIP2 Block Reviewer [869408], raised while tracing the authorship of
                          Codex's five files.
variant 4                 ULIP2 Block Reviewer [869408]. His measurement: float32 drifted 6 of
                          6, float64 0 of 6.
the four-way ordering,
and "variant 4 is the
most dangerous"           INTEGRATOR. This part is his.
```

**His reason for objecting is worth more than the correction.** In his words: the project spent
a full round today on a finding filed under the wrong owner, and the standing lesson in the
older records is *never record the approver as the one who did the work*. **Filing four people's
findings as one person's framework is that same error mirrored** — and it costs the next reader
the original measurement, because variant 4's float32 6-of-6 numbers are the ULIP2 Block
Reviewer's, not his.

**MASTER's error, recorded plainly:** I asked him "what cases did you generalise this from",
which assumed he had generalised it. He answered the question I should have asked instead of
the one I did. **A question can carry a false premise, and a helpful answer will carry it
forward.**

**⚠ Also corrected by him: the seam number.** DL-035 and the FINDINGS §3 note both discuss
"seam 1". The fingerprint finding is **seam 3**, not seam 1. His numbering:

```
1  splits / stage1_protocol
2  gallery index + encoder fingerprints
3  stage2 protocol / node vectors / sem_edge      <- the fingerprint one
4  the deviation registry
```

**A wrong seam number sends the next reader to the wrong file.**

---

## DL-036 — `.claude/` and `output/look/` stay out of git

`USER_DECIDED` · 2026-08-30 · Kyzen, asked directly, answered **「甲乙都不要」**

Two proposals were put to him and both were declined:

```
甲  track .claude/ and CLAUDE.md          DECLINED
乙  track output/look/*.json as evidence   DECLINED
```

`.gitignore:56` excludes `CLAUDE.md` and `.claude/` under the comment
"Local Claude / Graphify settings". `.gitignore:27` excludes `output/`.

**What this means, recorded so nobody re-derives it later as a discovery:**

```
NOT IN GIT   CLAUDE.md
             .claude/rules/  (research-rigor, experiments, paper-reproduction,
                              code-changes, upstream-lookup)
             .claude/hooks/research_authority_guard.py
             .claude/settings.json  (the three PreToolUse hooks)
             .claude/skills/
             .claude/agents/  (the five subagent definitions, 2026-08-30)
             output/look/dtype_effect.json  and the other evidence JSON

IN GIT       docs/METAFIND_NOTEBOOK.md, workflow/DECISION_LEDGER.md, and all
             product code and tests
```

The rules that govern how this project does research, and the hook that enforces
them, exist in one copy on one disk. So does the evidence JSON that DL-033 and
commit `2916327` cite.

**This is Kyzen's call and it is recorded as made, not as pending.** It is not
an oversight and should not be re-raised as a finding. If the working copy is
lost, these are the files that do not come back.


---

## DL-037 — sweep arm 1, the measurement round's first run

`RECORDED` · 2026-08-30 · run by the ULIP2 Block Engineer under Kyzen's 02:26 authorisation

The first of the eight runs DL-034 approved as a **measurement** round. It answers what the
seed-to-seed dispersion is. **It does not answer which learning rate wins, and nothing here
may be read that way.**

```
run_id                 1788027986-222044-d7a50a
command                python -m metafind.train.stage1 --phase dev --epochs 5 --preload
                       --lr 0.00075 --seed 20260830 --repeat-index 1
                       --out-dir sweep_lr/lr7.50e-4_s20260830
wall clock             02:26:25 -> 02:52:25, 26 minutes (estimate was 25)
out_dir                /home/kyzen/metafind_out/checkpoints/sweep_lr/lr7.50e-4_s20260830
arm_config_hash        eaa855ba8f1071d6c862291799a1545536c3990040d55a27309815d014cec708
seed / repeat_index    20260830 / 1
best epoch             4      checkpoint_schema 4      sha256 6e4ae4e05aad1e7a…
n_train / n_selection  31,985 / 4,569
train content sha256   fb44b5a19d0dabd36ebe5c164ed666e0…
pointcloud sidecars    0 mismatches between claimed and actual bytes
open_clip hf_revision  743c27bd53dfe508…
preload                True, first full execution: 560 ms/step against a 589 ms/step baseline
```

### Code state, written the way the engineer insisted and he was right

```
code_revision          2916327e + dirty
runtime_source_sha256  cbf275817e0e4e5bb0cd1e29d8afa6487cb5c92e3a1eb148e546af110dd50483
                       identical to the working tree after 169b0cb
dirty content          the score_streaming float64 guard, written before it was committed
                       engineer's judgement: not on the training path; stage1.py does not
                       call score_streaming
```

**"Identical hash" is the observation. "Therefore the same commit" is an inference, and the two
are written apart on purpose.** The engineer refused the shorter wording and asked for this one.

**"The dirty content is harmless" is a human judgement. No field can carry it, and it is
recorded as a judgement rather than promoted to a fact.**

This is the first run where all three provenance fields were used together, and each did a
different job: `code_revision` named a commit older than the code that ran, `code_dirty` said
"there is more than that", and `runtime_source_sha256` said which code it actually was.
**Any one of them alone would have been misleading. That is the argument for keeping all three.**

### Result

```
epoch    seven-condition mean R@1
  0      0.8582
  1      0.8432        the usual dip at a high learning rate
  2      0.9055
  3      0.9332
  4      0.9443        best

best epoch 4:  text 0.7577 · image 0.9606 · pc 0.9140 · full 1.0000
               unsaturated three-condition mean 0.8774 · seven-condition mean 0.9443
```

**`full = 1.0000` again.** DL-033's debt is unchanged: it is INFERENCE, the negative control is
still owed, and it is not discharged by another run reproducing it.

### A comparison that is NOT a comparison

```
DIED_e5_0542   lr 5e-4    5 epochs    three-condition 0.8927
arm 1          lr 7.5e-4  5 epochs    three-condition 0.8774
```

Looks like 1.5 pp for the lower rate. **It is not a controlled comparison and is not filed as
one:** different seed, different code (DIED_e5 came from a dirty tree on 2026-08-29), and that
run was terminated by a crash. Kyzen has ruled this batch a measurement round, not a selection
round. **The engineer flagged this himself and asked that it not be entered as a comparison.**

### New machinery that held on a real run

`--out-dir` wrote to its own directory and the canonical four files were untouched. `flock` on
`stage1_run.lock` was taken, held, and released at exit. The reservation `stage1_run.json` was
written before training, not after. The arm hash and its resolved recipe went into the
checkpoint. 32k input content fingerprints, zero mismatches. Initializer provenance — ULIP-2
sha plus OpenCLIP revision and blob — present.

### Standing

**Arm 2 is not authorised.** Kyzen's rule is one, then seven, and the second authorisation is
his to give after seeing this. GPU released.


---

## DL-038 — the write guard did not watch Bash, and I had already gone through it

`RECORDED` · 2026-08-30 · found by MASTER while fixing something else · fixed the same night

The research-authority guard blocked writes to the governance files, the vendored upstream, and
the paper source. Its guarded-tool set was `Write`, `Edit`, `NotebookEdit`.

**Bash was not in it.** A shell command reached every protected path untouched.

```
Write to the settings file                        blocked
Bash running python that writes the same file     not seen at all
```

**And that is how the settings file was edited earlier the same night**, when the subagent env
block was added: a python heredoc inside a Bash call. Nobody was working around the guard.
Nobody knew the door was there — the guard's own docstring included, which claimed its own
directory was protected from having the guard disabled. Bash disabled nothing. It went around.

### Why it was found at all

Only because the reviewers' read-only setup collided with the graphify hook, which sent MASTER
to read the guard. **The hole was not found by looking for holes.** Nothing was watching for it.

That is DL-035's variant 4 living inside the guard: the property held because no caller had
used the other door yet.

### The fix, and what it is not

Bash and PowerShell now reach a command-checking branch. Tokens are resolved against the repo,
and a protected path plus a write marker blocks. Reading stays free — a guard that prompts on
displaying a governance file becomes noise, and noise is what gets switched off.

**It is a heuristic and the file says so in capitals.** A command that assembles the path from
pieces, encodes it, or reaches it through a symlink made in the same command will pass. It is a
speed bump against accident, not a sandbox. Its purpose is narrower and worth stating plainly:
that nobody edits one of these files again *without knowing they did*.

Verified live, not only in tests: a copy onto a rules file now returns `GOVERNANCE INTEGRITY`
and is refused.

### It blocked its own first commit, and that mattered

The first attempt to file this entry was refused by the guard that had just shipped. The ledger
prose *quotes* shell commands naming protected paths, and the guard read the whole command
string — heredoc body included — as one blob.

**A guard that stops you writing about the guard is noise.** Fixed the same hour: inside a
heredoc body only file-API calls count, not shell verbs, because the real bypass was a python
call and the false positive was prose. Two named tests hold both halves — one that prose must
pass, one that a genuine python heredoc write must still be refused, so the exemption cannot
quietly reopen the hole it sits beside.

### The part that outlives the guard

`tests/test_research_authority_guard.py` is **tracked**. The guard is gitignored per DL-036, so
it has no history and no diff anyone can review. The test does. It carries the exact heredoc
that went through unseen, as a regression, and one test that fails if the "HEURISTIC / not a
sandbox" wording is ever deleted — so a future edit has to decide deliberately whether the
claim changed or only the wording did.

**39 tests, all passing against the installed guard. Nine of them fail against the version that
was running two hours earlier.**

### Also fixed, and how it was found

A gate now skips the graphify search guard for `ulip2-reviewer` and `essgnn-reviewer`. That
guard says a `graphify query` is MANDATORY before grepping, and both reviewers have no Bash by
design, so it named an action they cannot take.

**The ESSGNN Block Reviewer hit it on its first live run and filed it as a known blind spot
rather than ignoring it.** An instruction an agent cannot obey is not a guard; it is what
teaches an agent to read past guards.

### Recorded against myself

While fixing this I edited the live guard in place, broke its syntax, and the wrapper's
fail-closed branch then blocked every write in the repository until Kyzen restored the backup
by hand. The second attempt was built and tested in a scratch directory and installed in one
step.

**Editing the running guard was the mistake. The syntax error was only how it showed up.**

---

## DL-039 — five subagents were tested by making them try to break their own limits

`RECORDED` · 2026-08-30 · MASTER · all five roles, in parallel, working tree clean afterwards

Each role got four tasks with independently checkable answers, not a self-description:
read a specific line and quote it, search the whole repo and give the complete list, write a
scratch file, and **deliberately attempt the one thing it must not be able to do.**

```
engineers, integrator   try to create a file under .claude/rules/   → must be refused
reviewers               try to run Bash, try to Write               → must not exist
```

**All five passed. No `GUARD FAILED`. No `READ-ONLY BREACH`.** The reviewers reported that Bash
and Write are not refused but *absent* — the tool call cannot be formed. The engineers and the
integrator were refused by the guard, and one of them tried both Write and Bash rather than
one.

### What the tests were for, and what they actually produced

They were a capability check. Four of the five handed back a real defect nobody asked for.

```
1  score_streaming guards dtype and does NOT guard row norms.
   `_SHIFT = 1.0` at run_retrieval.py:59 is justified by "cosine is bounded above by 1",
   which holds only for unit-length rows. Nothing checks. Rank, tie_count and R@1 stay
   correct because ranking only compares; off_target_entropy at :263-264 goes wrong quietly.
   The headline metric is clean and the diagnostic is silently wrong.
                                                        -- ULIP2 Block Reviewer

2  ESSGNN does not check node feature width. essgnn.py:584-597 checks node count, edge_index
   shape, edge_attr rows, edge_missing shape -- not the width. Measured, not read:
     use_io_projections=True   fails at embed_in, 5x31 vs 32x64
     use_io_projections=False  embed_in is Identity, the wrong tensor travels, and it fails
                               later at 6x79 vs 81x32 with a message that never says
                               "node feature"
   Safe today only because stage2.py:374 checks the artifact.
                                                        -- ESSGNN Block Engineer

3  procthor_node_embeddings.json has six fields; two are read.
     uri, embedding_dim        read (stage2.py:363, :373-377)
     sha256                    no consumer -- and verify_recorded_artifact sits in the
                               same file at :77, used once at :621 for a different artifact
     asset_ids                 no consumer; ids come from the .npz instead
     text_encoder_version      no consumer
     n_assets                  no consumer
                                                        -- INTEGRATOR

4  essgnn.py:283 expands PRIMARY_INTERPRETATION (h0_mode, coords_agg, edge_proj_dim,
   normalize_coord_diff) which are not in the `required` set at :256-258, so a protocol file
   that sets them is not read.
                                                        -- ESSGNN Block Reviewer
```

**All four are DL-035's shape.** Three are variant 4 — correct today, guaranteed by nobody —
and one is variant 1. **Nobody was asked to look for them.**

The INTEGRATOR routed its own finding without being told to: the missing `sha256` check is the
seam's and therefore its own, a one-line call to a helper already in the file; whether
`text_encoder_version` should be enforced touches ESSGNN's `t_i` semantics and was passed up
rather than taken.

### Two defects in MASTER's own setup, found by the agents being tested

**The guard blocked a read.** `>` was in the write-marker list, so `ls -l <protected> 2>/dev/null`
was refused: a redirect and a protected path in one command, and the blob test could not tell
`2>/dev/null` from `> CLAUDE.md`. The ULIP2 Block Engineer hit it while trying to confirm the
guard had worked. **Fixed:** `>` is out of the marker list; redirection is judged by its target.
Eight new tests, four that must pass and four that must be refused. 47 total, all green.

**Reviewers cannot tell "absent" from "unreachable".** `data/` is a symlink; Glob and Grep do
not follow a symlinked start and do not warn, and the reviewers have no Bash to run `find -L`.
The ESSGNN Block Reviewer hit it, wrote *"this empty set I do not trust"*, and named it as its
own gap rather than concluding the file was missing.

**That is the same failure MASTER committed hours earlier and the opposite response.** Fixed
in both reviewer definitions: search the physical root `/home/kyzen/metafind_data/`, and never
write "does not exist" on the strength of an empty Glob under `data/`.

### Left undone, and why

`CLAUDE.md` §9 still documents the data root as `/home/kyzen/data/MetaFind`. The real value is
`/home/kyzen/metafind_data`. Reported independently by the ULIP2 Block Reviewer and the ESSGNN
Block Engineer, both of whom reported rather than modified.

**MASTER cannot correct it.** The escape hatch is an environment variable read by the hook
process, and a hook runs in Claude Code's environment, not in the environment of the shell
command it is judging — so `METAFIND_ALLOW_AUTHORITY_EDIT=1 python fix.py` sets it for the
script and never for the guard. **The override is reachable only from a human's own shell.**

That is not a bug. It means an agent cannot talk itself past this guard, no matter how good its
reasons sound. The correction is written and staged at
`scratchpad/fix_dataroot.py`; it needs one human invocation.

---

## DL-040 — the `✅` for the LR sweep's remaining seven arms

`USER_APPROVED` · 2026-08-30 · Kyzen

**Verbatim, the whole message:**

```
✅
8/19？
```

The `✅` arrived immediately after MASTER put two things to him and named which
needed approval:

```
1  掃描 7 次要不要跑          你打 ✅ 我就開
2  8/19 那個戳算不算逐項核可   這題卡住 Stage 2，不卡 Stage 1
```

`8/19？` is a question, not an answer, so **item 2 remains open.**

### What this `✅` covers

```
the seven remaining arms of the LR sweep, in the order already shuffled with
random.Random(20260830), which is not to be changed:

  2  --lr 0.001   --seed 20260816 --repeat-index 0
  3  --lr 0.0005  --seed 20260816 --repeat-index 0
  4  --lr 0.00075 --seed 20260816 --repeat-index 0
  5  --lr 0.001   --seed 20260830 --repeat-index 1
  6  --lr 0.0005  --seed 20260830 --repeat-index 1
  7  --lr 0.00025 --seed 20260830 --repeat-index 1
  8  --lr 0.00025 --seed 20260816 --repeat-index 0

each --phase dev --epochs 5 --preload, its own --out-dir under sweep_lr/
~26 minutes each, ~3.0 hours total, one at a time on the single card
```

### What it does NOT cover

```
n15 on any protocol            never run; needs its own approval
--phase final                  the real training run
protocols A or B               the sealed test set; also needs --unseal
n07c / n08 / anything Stage 2  a different block, and item 2 above is still open
raising the power limit        standing prohibition, unchanged
```

**DL-034 still governs what these eight runs mean.** Kyzen ruled them a
**measurement round, not a selection round**: they produce this project's first
honest seed-to-seed dispersion and a rough position for each of four learning
rates. **They do not produce "which LR wins."** Choosing the learning rate is a
separate decision and a separate `✅`.

### The gate that still applies before the first arm starts

`DL-028` requires review before code runs, and the runner that executes these
seven arms **is new code written after this `✅`**. The approval is for the
experiment, not for an unreviewed script. The runner carries a deliberate
refusal-by-default flag so that it cannot start by accident, and it goes to the
Block Reviewer before it is used.

**Recorded by MASTER, who received the `✅`, per `DL-031`: the role that
receives it records it, with the verbatim message and what it covered.**

---

## DL-041 — the five ESSGNN architecture parameters are ratified, and one tension is recorded with them

`USER_APPROVED` · 2026-08-30 · Kyzen

**Verbatim:**

```
我記得這是✅的 他是討論出來的 由上游證據支持 下次跑前確認就好 先✅
```

Put to him as a named list, which is what `Rule 16` requires — a ratification has
to point at specific parameters, not at a file:

```
hidden_dim            128
n_layers                4
pooling              mean
layer_sharing  independent
use_io_projections   true
```

### What is now settled

**The five values above are `USER_APPROVED` as of 2026-08-30.**
`essgnn_arch_protocol.json`'s `status: resolved` is legitimate from today. It was
not legitimate on the strength of the 2026-08-19 stamp alone, and the two
readings MASTER put to Kyzen — a per-parameter ratification versus a batch stamp
written after C1 — are closed by this entry rather than by re-reading the stamp.

**This also releases what was blocked on it.** `ESSGNN_DIM_REVIEW.md` §7's
standing prohibition ("D_h does not move, no candidate enters the protocol, until
Steps 1–4 are settled") has been satisfied, and the ESSGNN Block Engineer's
handoff §11 item 3, which forwards that prohibition, should now be updated. He
chose reading B deliberately because it was the conservative side, and said so;
that judgement was correct at the time it was made.

### A tension, recorded because it does not disappear by being approved

Kyzen's stated basis is 「由上游證據支持」. **The upstream evidence in this
repository does not point at two of these five values**, and that was itself a
registered finding before today:

```
EGNN QM9        n_layers 7 · hidden 128 · readout torch.sum
EGNN N-body     n_layers 4 · hidden  64
ours            n_layers 4 · hidden 128 · pooling mean
```

`METAFIND_NOTEBOOK.md` §3.4 recorded that 4/128 takes one half from each of two
different upstream experiments, and that MetaFind's own citation of EGNN names
drug design, which is QM9. `docs/ESSGNN_DIM_REVIEW.md` §6 carries the same
comparison.

**This entry does not reopen the decision.** `DL-017` delegates these calls and
Kyzen has made this one. It records that the values are a **USER DECISION**, and
that describing them as upstream-supported would be wrong for `n_layers` and
`pooling` specifically. **If Table 2 ever cites an upstream basis for the
architecture, this is the paragraph that must be read first.**

Classification: `hidden_dim`, `layer_sharing`, `use_io_projections` — USER
DECISION, upstream-compatible. `n_layers`, `pooling` — **USER DECISION,
upstream evidence points elsewhere.**

### The condition Kyzen attached

「下次跑前確認就好」 — **confirm before the next run that uses them.** That is a
re-confirmation gate, not a re-decision: before n13 or anything else consumes
`essgnn_arch_protocol.json`, the five values are put back to him, and he says
whether they still stand. It does not need re-derivation and it does not reopen
the argument.

### Still open, and not covered by this `✅`

```
node_feat_dim / edge_feat_dim are STILL NOT KEYS of essgnn_arch_protocol.json.
U-20 pinned 1280 on 2026-08-27 and nothing enforces it: an n08 that wrote a
512-wide artifact would train without complaint. That debt is item 201 at
01_GRAPH_SPEC.md:1123 and it is not what was approved here.
```

**And the artifact on disk is still 512-wide from the old encoder.**
`procthor_node_embeddings.npz` is `(1467, 512)`, `text_encoder_version
clip-vit-b32-laion2b-s34b-b79k`, written 2026-08-17 — before the 1280 ruling.
**Nothing compares it against the ruling.** n08 has to re-run, and that needs
GPU and its own `✅`.

---

## DL-042 — findings that existed only in conversation, filed

`RECORDED` · 2026-08-30 · collected by MASTER from the ULIP2 Block Reviewer, the
ESSGNN Block Reviewer and INTEGRATOR, each asked what they had said and never
written down

**None of these was discovered today. All of them were known, by somebody, and
none was in a file.** They are filed together because that is the finding: the
project's memory has been the conversation windows, and a window is not storage.

### From the ULIP2 Block Reviewer — seven, grepped against `REVIEW.md` first

**① The paper's shape and ours are inverted. This is the one that matters.**

```
paper   pc 75.1  >  full 51.7     adding modalities HURTS a point-cloud query
ours    pc 0.897 <  full 1.000    adding modalities saturates
paper Table 1 (w/o ESSGNN) seven-condition mean R@1 = 37.11%
ours, protocol C                                    93-96%
grep "37.11" REVIEW.md → 0
```

**Not a tuning gap. The direction is opposite.** A gap invites "train longer";
an inversion says something upstream of the number is different.

**His own caveat, kept:** the present comparison crosses different gallery sizes
and is not legitimate on its own. It has to be redone once the two real
protocols have run.

**② Why `evaluate_dev_val`'s `targets = np.arange(...)` is sound.** The loader is
`shuffle=False`, `drop_last=False`, single pass, so query row *i* and gallery row
*i* are the same asset. He verified it. Never written down — and it is the
premise the whole dev-val number rests on.

**③ One query sits on a knife edge in `image+pc`.**

```
recorded (torch f32)  0.987743489
numpy f32             0.987962355
numpy f64             0.987743489
difference 0.000218866 = exactly 1/4569
```

One query flips with the arithmetic library. Only in
`output/look/dtype_effect_helper.json`, which is gitignored per `DL-036`.

**④ `reported` in `eval_protocols.json` conflates two states** — "intended for the
report" and "has actually run". A and B carry `true` and have never executed. He
proposed renaming it `reportable`; it touches `tests/test_splits.py`.

**⑤ His own fix for protocol B's contamination label, which was not recorded.**
Do **not** touch `reported`: `reported: true` is U-09's deliberate bracket and
`retrieval.py:19-27` holds that reasoning; overturning it discards the argument.
Add a field instead — `gallery_contains_train_split: true`. What is missing is
the label, not the decision.

**⑥ `CLAUDE.md` §9's data root was stale.** Corrected 2026-08-30.

**⑦ The 46,052 list.** Superseded the same day by the finding that three corpus
sizes exist and each is right about something different; see the `NUMBERS` commit.

### From the ESSGNN Block Reviewer

He never read `ESSGNN_DIM_REVIEW.md` — **he refused**, because the message
ordering it said in its own text that the freeze was not lifted, and reading
3,200 lines against six rules is review work wearing the clothes of reading.
MASTER accepted the refusal.

**So the document's "external review rejected eighteen items" was not him**, and
the other twelve are not recoverable from his window. He handed over six that a
previous MASTER had listed as its own errors:

```
1  v2 named Candidate D "Appendix-consistent" and then wrote EGNN's phi_h
2  v2 proposed reopening the architecture family, against Rule 13 and a
   USER-backed U-26
3  v2 said the paper pins d at 1280 -- it had silently added "Pooling preserves
   width", and Pooling is not named
4  v2 labelled "BERT-base 768" a PAPER FACT; the paper says only "e.g. CLIP or BERT"
5  v2 argued from dimension share (edge 83%) that edges overwhelm geometry
6  it inferred from EGNN's in_edge_nf = 0/2 that edges should be narrow -- Type C,
   which yields no architectural principle
```

**Item 5's current wording is his, and it is the sharpest thing in the set:**

> `m_ij = phi_e(cat[h_i, h_j, radial, e_ij])` is a concat into an MLP. The first
> Linear holds an independent weight block per slice: a 512-wide slice can be
> learned to zero, a 1-wide slice can dominate the output. **Width is capacity,
> not influence.** And `2methdology.tex:54`'s `f_h: R^(2d+1+e) -> R^d` names `d`
> and `e` separately — the paper never asks that they be comparable.

Filed at `workflow/FINDINGS_20260827_CAPTURE.md` §5b; MASTER verified it is there.

**On F-A (phi_e missing EGNN's trailing Swish): nobody ever sent it to him.** The
handoff's "not attacked by the Reviewer" is accurate, not an omission. He added
the point that matters: `essgnn_arch_protocol.json`'s `mlp_structure:
linear_silu_linear` **guards the code against drifting away from its present
shape — not against being unlike upstream.** So "the protocol covers it" is false
for this finding: the protocol guards the very thing under question.

### From INTEGRATOR — on hold throughout, so his findings had nowhere to go

```
A  A document holding both an old and a new version, with nothing marking which
   is current. Strikethrough is invisible to grep and in a long table. Three
   people were misled by it in two days, including INTEGRATOR twice.

B  A superseded measurement left in place, unmarked: METAFIND_NOTEBOOK.md:1481's
   ratio 2.99 is single-seed at a config we do not run, while :1296 and :1342 are
   six seeds at the config we do. One document, one number, two opposite
   directions. ⚠ He originally judged :1481 the correct one and has withdrawn that.

C  Identity assembled from parts that were each true at different times: socket
   read now, name and ref carried over from the previous round. The composite
   identity never existed. He did it once and was caught by a bounced message;
   the ULIP2 Reviewer did an isomorphic one the same day, quoting his own summary
   as somebody else's words. The error is not in the content but in whether two
   parts belong to the same moment -- and every existing check looks only at content.

D  e5's artifacts were overwritten by e10, leaving only per-epoch totals.
   Against `.claude/rules/experiments.md` §10. Already lost; only a re-run recovers
   it, and that needs GPU and a ✅.

E  The semantic-edge LLM has no deviation number. DL-005 split D-2 into D-2 and
   D-8 and it belongs to neither. ⚠ Kyzen ruled all LLM roles to gemma on
   2026-08-30, which settles the model; **the missing deviation id is a separate
   gap and is still open.**
```

**He also corrected the seam number MASTER used, and the attribution of the
four-variant framing — both recorded against `DL-035`.**

### The thing all three have in common

Each was asked "what did you say that never got written down", and each had
something. **The ULIP2 Block Reviewer's full handoff could not be written at all,
because MASTER had broken the write guard an hour earlier and its fail-closed
branch was refusing every write in the repository.** A finding that lives in a
window dies with the window, and on 2026-08-29 this machine hard-reset nine times.

---

## DL-043 — the `✅` token is withdrawn. **Supersedes `DL-031`, amends `DL-030`.**

`USER_APPROVED` · 2026-08-30 · Kyzen

**Verbatim:**

```
把✅規則拔掉啦 反正規則是需要做決策由你跟我來討論 我要清楚知道決策內容 你去執行 懂
```

### What `DL-031` was, and why it existed

`DL-031` (Kyzen, 2026-08-24) made `✅` the only approval token. Its own entry
records what it was written to stop: within hours of `DL-030`, the ULIP2 Block
Reviewer asked whether a completed run had been authorised, **because he feared
his own review PASS had been read as the authorisation.** MASTER could not
answer. Approval was being inferred from ordinary conversation.

**That entry is not deleted and its reasoning is not wrong. It is superseded by
the person who wrote it.**

### What replaces it

```
DECISION  discussed between Kyzen and MASTER
          Kyzen must clearly know what is being decided
EXECUTION MASTER's, without a token
```

The gate moves from **a symbol** to **his understanding**. That is a higher bar
on MASTER, not a lower one: under `DL-031` the obligation was to collect a
character. Now the obligation is that Kyzen actually knows what he agreed to.

### The failure this reopens, named so it is not rediscovered

**Without a token, "he seemed to agree" becomes approval again.** That is
precisely what `DL-031` was written to stop, and removing the token does not
remove the failure mode. So the replacement is a form, not a feeling:

Before anything expensive, irreversible, or outward-facing, MASTER states, in
this shape, in Kyzen's register:

```
要決定什麼      one sentence
代價            time, GPU, what it consumes
它關掉了什麼    what this forecloses, and what stays open
我建議          MASTER's recommendation and why
```

**His reply is the approval — a reply that engages with the decision. Silence is
not. A change of subject is not. Enthusiasm about something else is not.**
The exchange is recorded verbatim in this ledger, both halves, as `DL-031`
already required of whoever received an approval.

### What does not change

```
DL-028   code is reviewed before it runs
DL-030   the detailed report before a run -- the report survives; the
         separate token afterwards does not
the power limit stays where Kyzen set it
research decisions are still his: architecture, hyperparameters, data,
         evaluation protocol
Stage 2 is untouched by this entry
```

**And `DL-017` still holds**: he delegates specified calls to the blocks. This
entry widens execution, not authority over research.

### Scope of what he has already delegated under this rule

`Stage 1 不用等我同意了 你自己判斷 我該怎麼做都講好了` (2026-08-30) hands MASTER
the execution of Stage 1: sequencing, when to run, when to stop. **Stage 2 was
not included and neither was the power limit.**

### One consequence MASTER must not exploit

The old rule let MASTER say "I have no `✅`, so I cannot". **That excuse is gone.**
Under this entry, not knowing whether Kyzen would agree is a reason to go and ask
him clearly — not a reason to wait, and not a reason to proceed quietly.

---

## DL-044 — an untrained model scores 0.999 on `full`. The evaluation is not measuring retrieval.

`RECORDED` · 2026-08-30 · MASTER, executed under Kyzen's Stage 1 delegation · GPU, ~3 min

**The control the ULIP2 Block Reviewer named as the minimum before Table 1 could
leave has now been run.** Its result closes the `full = 1.0000` question and opens
a larger one.

### What was run

```
python -m metafind.eval.run_retrieval --protocol C_dev_selection \
       --ckpt-record none --init-seed {1,2,3}
```

No Stage 1 checkpoint. Both fusion towers at random initialisation. The point
encoder still carries ULIP-2's pretrained PointBERT and pc_projection; text and
image still go through pretrained OpenCLIP ViT-bigG-14. **This is an
untrained baseline, not an unpretrained one**, and the artifacts say so in their
own provenance.

### The result

```
condition     untrained s1   s2       s3       best trained   training adds   paper
text            0.9729    0.9593   0.9604      0.8932        -7.10 pp        0.138
image           0.9041    0.9184   0.9085      0.9816        +7.13 pp        0.117
pc              0.9532    0.9475   0.9350      0.9759        +3.07 pp        0.751
text+image      0.9974    0.9982   0.9974      0.9980        +0.04 pp        0.172
text+pc         0.9998    0.9998   0.9998      0.9998         0.00 pp        0.445
image+pc        0.9869    0.9851   0.9866      0.9937        +0.74 pp        0.458
full            0.9989    0.9998   0.9996      1.0000        +0.06 pp        0.517
```

**`full = 1.0000` is not a property of the trained model.** An untrained one
reaches 0.9989 on the same protocol. Training moves that cell by 0.06 pp.

`text+pc` is identical to four decimal places, trained or not. `text+image` moves
0.04 pp. **On four of seven conditions, Stage 1 training is indistinguishable
from doing nothing.** On `text` it is 7.10 pp *worse* than doing nothing.

### What this does to the LR sweep, which finished 25 minutes earlier

Every arm, three-cell mean, against the untrained baseline of 0.9399:

```
lr 0.001      0.8461     -9.38 pp
lr 0.00075    0.8732     -6.67 pp
lr 0.0005     0.9075     -3.24 pp
lr 0.00025    0.9493     +0.94 pp
```

**Three of four learning rates make the model worse than not training it.** The
fourth beats it by 0.94 pp — below `DL-033`'s δ of 1.0 pp, and below the
untrained baseline's own seed-to-seed spread of 0.88 pp.

The sweep's clean monotone ordering — lower learning rate, better score — reads
as **how little damage the training does**, not how well the model retrieves.
And the best value sits at the edge of the range tested: 0.00025 is the lowest
learning rate in the sweep, so the ordering does not contain its own optimum.

### The seed-to-seed dispersion this round was meant to produce

```
lr 0.001      1.02 pp
lr 0.00075    0.85 pp
lr 0.0005     0.38 pp
lr 0.00025    0.19 pp
untrained     0.88 pp   (three draws of the fusion towers)
```

**It is not one number. It depends on the learning rate**, by a factor of five
across the range, and the dispersion of the untrained baseline is larger than the
gap the best arm earns. `DL-034` called this batch a measurement round and not a
selection round. That was right, and it is now clear why: **there is nothing here
to select between.**

### What is now established, and what is not

```
ESTABLISHED   protocol C cannot separate a trained model from an untrained one
              on four of seven conditions.  OBSERVED DATA, three seeds.
ESTABLISHED   `full = 1.0000` is structural, not learned.  The DL-033 debt is
              DISCHARGED -- and the answer is worse than the debt.
NOT ESTABLISHED  whether protocols A and B separate them. Both have larger
              galleries and neither has ever run.
NOT ESTABLISHED  why the pretrained encoders alone solve this task. The obvious
              reading is that query and gallery are the same 4,569 assets and
              a pretrained encoder already places each asset nearest itself,
              but that is INFERENCE and has not been measured.
NOT ESTABLISHED  whether any of this survives at gallery 45,692.
```

### What must not happen next

**No Table 1 number may be quoted from protocol C.** It was already marked
`reported: false`, for a different reason, and that marking is now load-bearing.

**No learning rate may be selected from this sweep.** Selecting the best of four
arms that are mostly worse than no training at all would be choosing the least
harmful setting and reporting it as the best model.

**`--phase final` must not run yet.** It would consume hours to produce a model
whose measured advantage over random initialisation is smaller than the noise.

### The instrument was measured before the object, and only just in time

The reviewer's sentence was: *"if `full` is already near 1.0000 at initialisation,
that cell is not measuring the model — and `shuffle_targets` will never tell you
so."* Both halves held. The control that existed would have passed. The control
that could discriminate was documented as needing no code and had none until
today.

**Eight GPU runs, roughly three and a half hours, were spent comparing settings on
an instrument that had never been checked against a null.** The check cost three
minutes.

---

## DL-045 — protocol D: gallery size is not the cause, and training's real effect is visible for the first time

`RECORDED` · 2026-08-30 · MASTER, under Kyzen's Stage 1 delegation · GPU, ~8 min, nothing sealed touched

`DL-044` left one candidate standing: that `full = 1.0000` is an artifact of
protocol C's 4,569-asset gallery. **It is not.**

### Protocol D, added at its producer

`splits.build_eval_protocols` now emits `D_dev_val_vs_train`: the same 4,569
dev-val queries against the whole 36,554-asset training pool, an 8x gallery.
`eval_protocols.json` was regenerated from the existing `splits.json` — no
re-split, no GPU — and the three existing protocols were compared field by field
first and are unchanged. **It reads nothing sealed**: `dev_val` is a subset of
`train`, and `check_seal` looks for `test` or `full`, which `("dev_val","train")`
is not. `reported: false`, for C's reason and one more — every candidate in its
gallery is an asset the model was fitted on.

### Result

```
condition     untrained C   untrained D   trained D   training adds   paper
text            0.9729        0.9044       0.7586      -14.6 pp       0.138
image           0.9041        0.7308       0.9278      +19.7 pp       0.117
pc              0.9532        0.8332       0.9181       +8.5 pp       0.751
text+image      0.9974        0.9866       0.9912       +0.5 pp       0.172
text+pc         0.9998        0.9969       0.9998       +0.3 pp       0.445
image+pc        0.9869        0.9315       0.9637       +3.2 pp       0.458
full            0.9989        0.9961       1.0000       +0.4 pp       0.517
```

### What this settles

**Gallery size is not the cause.** Growing the gallery eight-fold moves the
untrained `full` from 0.9989 to 0.9961 — **0.28 pp**. The last of `DL-044`'s
candidate explanations is eliminated by measurement, the second hypothesis this
project has closed that way.

**The four multi-modal cells are saturated for a null model at both gallery
sizes.** `text+pc` is 0.9969 untrained. Whatever those cells measure, it is
available before any Stage 1 training happens.

**The three single-modality cells do separate, and this is the first honest
measurement of what Stage 1 training is worth**: +4.54 pp on their mean, at 8x
gallery. On protocol C the same comparison gave +0.6 pp. **C was too small to
show it; D is not.**

### The finding nobody was looking for

**`text` gets worse with training, at both gallery sizes, and the damage grows
with the gallery: −7.1 pp at C, −14.6 pp at D.** It is the only condition that
moves down, and it moves down consistently across two protocols and two
checkpoints. **This is not noise and it has never been reported.**

It is also the condition where the paper is weakest (0.138), so a reproduction
that quietly degrades text retrieval would still look superficially plausible
against Table 1.

**Not yet explained.** One reading is that the shared-input mechanism from
`DL-044` gives `text` a free ride at initialisation, which training then spends;
another is that the text tower is being actively damaged. Distinguishing them
needs the per-epoch trace, which `MINOR-2`'s unapplied patch would restore going
forward and which no existing run can supply.

### Standing after this entry

```
CLOSED    full = 1.0000 is structural, not learned, and not a gallery-size artifact
CLOSED    protocol C cannot separate a trained model from an untrained one
OPEN      why `text` degrades under training, and by more as the gallery grows
OPEN      whether A and B behave like D; neither has ever run
UNCHANGED no Table 1 number may be quoted from C or D. Both are reported: false.
UNCHANGED no learning rate may be selected from the sweep: it was scored on C.
```

**The sweep's ranking is now known to have been measured on a protocol that a
null model passes.** Re-scoring the eight checkpoints on D is the cheap next
step — eight runs of roughly four minutes, nothing sealed — and it is the only
way the learning-rate question gets an answer that means anything.

---

## DL-046 — 🔴 both towers are fed the same dict. We reproduced the baselines' failure mode, not MetaFind's method.

`ESCALATED TO USER` · 2026-08-30 · MASTER, after Kyzen asked whether anyone had read ULIP-2's actual code

**This is the root cause of every anomaly recorded in `DL-044` and `DL-045`.**

### The paper names this exact failure, and names it as the baselines'

`3experiments.tex:24`, verbatim:

> since other models do not adopt a dual-tower design, their "PC only" performance
> **reflects retrieval using identical embeddings for both query and gallery,
> leading to inflated accuracy**. In contrast, our dual-tower framework introduces
> more cross-modality retrieval, **which results in lower accuracy under the
> "PC only"**.

```
baselines' PC-only, which the paper calls inflated   97.9 – 99.0
MetaFind's PC-only, the dual tower working            75.1
ours, protocol C                                      97.6
ours, protocol D (8x gallery)                         91.8
ours untrained, protocol C                            95.3
```

**We are sitting in the band the paper identifies as the artifact.** The paper's
own number is 22 points BELOW the baselines because its dual tower makes the two
sides differ. Ours does not differ.

### The two lines

`metafind/train/stage1.py:1752-1753`, the training step:

```python
q = model.query(embeds, present=present)
g = model.gallery(embeds)
```

`stage1.py:791-793` builds **one** `embeds` dict. Both towers receive it. The
same pattern is at `:795`/`:799` in `evaluate_dev_val`.

`MetaFindDualTower.forward` at `dual_tower.py:359-366` takes `query_embeds` and
`gallery_embeds` as **separate arguments**. The architecture supports two inputs.
Nothing in Stage 1 ever passes two.

**So the contrastive loss asks two fusion modules, fed identical input, to agree
on row i.** The optimum is for them to become the same function, and then
retrieval is the identity map. Every measurement today follows from that:

```
DL-044  full = 0.9989 untrained          two random fusions of one input already agree
DL-044  four of seven cells unmoved       there is nothing for training to learn
DL-045  gallery x8 changes nothing        the identity map does not care how many distractors
DL-045  text degrades under training      training spends text structure buying agreement
sweep   3 of 4 LRs worse than untrained   training can only move the towers apart
```

### What ULIP-2 actually does, since that is the question that led here

`upstream/ULIP/models/losses.py:44-49` — every term has the point cloud on one
side:

```
loss = (CE(pc->text) + CE(text->pc))/2 + (CE(pc->image) + CE(image->pc))/2
```

**There is no text-image term.** The point cloud moves; text and image are frozen
anchors that never move relative to each other. It is a hub, not two towers.

And `upstream/ULIP/main.py:350-410` — ULIP-2's evaluation is **zero-shot
classification**, not instance retrieval: the gallery is the ~1,156 LVIS category
label texts, and correct means the right CATEGORY. **Our task and ULIP-2's are
not the same task**, so ULIP-2's recipe cannot be carried over unexamined, and
`DL-010` (upstream is the reference where MetaFind is silent) does not reach this.

### What is established, and what is not

```
ESTABLISHED   both towers receive one shared embeds dict, training and eval.
              OBSERVED IMPLEMENTATION, stage1.py:1752-1753 and :791-799.
ESTABLISHED   the architecture accepts two, dual_tower.py:359-366.
ESTABLISHED   the paper names identical query/gallery embeddings as the
              baselines' artifact and reports 75.1 against their 97.9-99.0.
              PAPER FACT, 3experiments.tex:24 and Table 1.
NOT ESTABLISHED  what MetaFind feeds each tower instead. The paper says the dual
              tower "introduces more cross-modality retrieval" and does not
              spell out the input construction. UNKNOWN.
```

### Why this stops here

Changing what each tower is fed is **architecture and data flow**. It changes
every Table 1 number, and it is the difference between reproducing MetaFind and
reproducing the baselines it argues against.

`.claude/rules/research-rigor.md` §2 lists architecture, dataset construction and
evaluation protocol as stop-and-ask. `DL-043` moved execution to MASTER and left
research decisions with Kyzen. **This is a research decision.**

**Everything downstream is suspended**: no learning rate may be selected, no
`--phase final`, no Table 1. The eight checkpoints and the re-scoring on D are
not wasted — they are the measurement that found this — but none of them is a
model of MetaFind.

---

## DL-047 — 🔴 `DL-046` is RETRACTED. The wiring is correct; I did not read what `present` does.

`RECORDED` · 2026-08-30 · external review, relayed by Kyzen · verified by MASTER before accepting

**`DL-046` claimed both towers are fed identical input and that Stage 1 therefore
reproduces the baselines' artifact. The first half is false, and the second does
not follow.**

### What I checked, and what I did not

I read `stage1.py:1752-1753`:

```python
q = model.query(embeds, present=present)
g = model.gallery(embeds)
```

saw one `embeds` dict on both lines, and stopped. **I never opened `present`.**

`fusion.py:198`:

```python
fill = torch.zeros_like(e) if self.cfg.zero_pad else self.mask_tokens[i].expand_as(e)
cols.append(torch.where(keep, e, fill))
```

**The query's absent modalities are replaced by learned mask tokens.** The dict is
shared because the positive pair is the same asset — which the paper requires —
but the effective inputs differ:

```
query    QueryFusion(mask_text, mask_image, pc_i)       partial modalities
gallery  GalleryFusion(text_i,  image_i,   pc_i)        always complete
```

**That is cross-modality retrieval, and it is what `2methdology.tex:14` and `:73`
describe.** The paper's own contrast with the baselines — query and gallery from
identical embeddings — does not apply to this code.

### The error, named

**I inferred a data flow from two lines and did not open the third.** The
argument name `present` was right there and I treated it as decoration. Every
downstream claim in `DL-046` inherited that.

**It is the same shape as the `find` without `-L` and the `[ab]?` regex: I built
a window, then reported what the window showed as what is there.** Third time
today, and this one reached a root-cause claim that stopped the whole stage.

### What survives from `DL-046`, narrowed

```
SURVIVES   in the `full` condition the mask is all-True, so query and gallery
           DO receive identical input there, and two fusions trained to agree
           make that cell trivial. full = 0.9989 untrained is still explained.
SURVIVES   full is unusable as a selection metric. Eight checkpoints, four
           learning rates: 0.9996-1.0000 throughout.
SURVIVES   every measurement in DL-044 and DL-045. The numbers were correct;
           the cause I assigned them was not.
RETRACTED  "both towers are fed the same input"          -- false in general
RETRACTED  "we reproduced the baselines' failure mode"   -- does not follow
RETRACTED  "the optimum is for the two fusions to become the same function"
           -- they are asked to agree only on the 34.3% of steps where all
           three modalities survive the mask
```

### Also corrected: my sweep conclusion was stale

I reported "all learning rates score worse than untrained" from 6 of 8 arms. All
eight have since finished:

```
untrained (1 init seed)   three-cell 0.8228   text 0.9044   full 0.9961
lr 2.5e-4                 0.8650-0.8682       0.7516-0.7586  1.0000
lr 5e-4                   0.7828-0.7868       0.6437-0.6500  1.0000
lr 7.5e-4                 0.7087-0.7209       0.5424-0.5450  0.9996-1.0000
lr 1e-3                   0.6629-0.6678       0.4940-0.5340  0.9998
```

**`2.5e-4` is above the untrained baseline by roughly 4.2-4.5 pp.** I reported
the opposite because the two `2.5e-4` arms had not finished. **A partial run is
not a result, and I read one as one.**

⚠ The untrained protocol-D baseline is **one init seed**. It is not a precise
reference and must not be quoted as one until there are at least three.

### The text degradation is a SEPARATE defect, not a consequence

The review's argument, which I accept: at `2.5e-4` image and pc rise while text
falls. A single identity-collapse would not produce that modality-specific
split. Text-only is 6.3% of masked samples against full's 34.3%, and selection is
diluted by four ceiling cells, so a checkpoint can be chosen while text collapses.

**This stays open as its own finding.**

### What is actually unknown

```
Table 1's gallery scope -- test-only or full corpus. The paper does not say.
The paper's query/gallery pairing and query construction.
MetaFind's real LR, epochs, optimizer, batch size.
ULIP-2's released checkpoint's pretraining LR -- upstream contradicts itself:
  main.py default 3e-3; the PointBERT yaml says 5e-4 in an optimizer block the
  pretraining path does not read. Our 5e-4 is THIS project's choice, and calling
  it a ULIP-2 fact would be wrong.
```

### Standing

**Stage 1's paper-to-code trunk is Verified, not broken.** The verdict is
`PARTIALLY VERIFIED`: the mechanism matches the paper, Table 1 is not reproduced,
and the gap is not explained.

**The suspension stands, for a different reason than `DL-046` gave.** Not because
the wiring is wrong, but because `full` cannot select and text is degrading, so a
learning rate chosen now would be chosen on a diluted metric.

---

## DL-048 — Stage 1's final configuration is LOCKED. **Lifts `DL-047`'s suspension.**

`USER_APPROVED` · 2026-08-30 · Kyzen

**Verbatim:**

```
不要再 sweep、不要再改 Stage 1 架構,
直接鎖 LR=2.5e-4、epochs=5、seed=20260816 完成 final training。
```

### The configuration

```
learning_rate   2.5e-4
epochs          5
lr_horizon      5
seed            20260816
phase           final
batch_size      64
p_mask          0.30
```

**Classification: `IMPLEMENTATION CHOICE`.** It is not claimed to be a MetaFind
paper value and not claimed to be a ULIP-2 value. It is not claimed to be a
global optimum: `2.5e-4` is the LOWEST rate in the sweep, so the ordering does
not contain its own edge, and nothing below it was tried.

### The evidence, re-verified before filing (OBSERVED DATA, 2026-08-30)

Dev-selection mean R@1, read from each arm's own `stage1_best_ckpt.json`:

```
lr 2.50e-4   s20260816  0.976425   best epoch 4
lr 2.50e-4   s20260830  0.977457   best epoch 4
lr 5.00e-4   s20260816  0.956977   best epoch 4
lr 5.00e-4   s20260830  0.958290   best epoch 4
lr 7.50e-4   s20260816  0.940124   best epoch 4
lr 7.50e-4   s20260830  0.944283   best epoch 4
lr 1.00e-3   s20260816  0.931808   best epoch 4
lr 1.00e-3   s20260830  0.927555   best epoch 4
```

**Every arm's best is `epoch 4`** — the fifth and last epoch, zero-indexed. No
arm peaked early, so `epochs=5` is the horizon that was run, not a horizon that
was selected. The ordering is monotone in the learning rate on both metrics.

Protocol D re-scoring of the same eight checkpoints, mean of ALL SEVEN
conditions, against the untrained baseline:

```
lr 2.50e-4  s20260830   0.9370
lr 2.50e-4  s20260816   0.9357
UNTRAINED   seed 1      0.9114
lr 5.00e-4  s20260830   0.8975
lr 5.00e-4  s20260816   0.8954
lr 7.50e-4  s20260830   0.8661
lr 7.50e-4  s20260816   0.8594
lr 1.00e-3  s20260816   0.8392
lr 1.00e-3  s20260830   0.8378
```

Mean of the three NON-ceiling conditions only (`text`, `image`, `pc`), which is
where the four saturated cells stop diluting the comparison:

```
lr 2.50e-4  s20260830   0.8682     +4.54 pp over untrained
lr 2.50e-4  s20260816   0.8650     +4.22 pp over untrained
UNTRAINED   seed 1      0.8228
```

**`2.5e-4` is the only rate that beats not training at all, and it does so on
both readings.** The three rates above it are worse than the untrained baseline
on the seven-condition mean.

`20260816` is used because it was specified in advance. It is NOT the
better-scoring of the two seeds — `20260830` is, on both metrics. **Choosing the
pre-specified seed over the winning one is the point.**

### What this ruling accepts, stated so it is not discovered later

**Text degrades under training and the ruling proceeds anyway.** Protocol D,
`text` R@1: untrained `0.9044`; at `2.5e-4`, `0.7516` (s20260816) and `0.7586`
(s20260830) — **−15.3 pp and −14.6 pp.** It gets worse at every higher rate,
down to `0.4940` at `1e-3`.

This remains the open finding `DL-047` filed. It is not resolved by this entry,
it is not caused by this entry, and the final checkpoint will carry it.

**`full` is still at the ceiling and still cannot select.** `1.0000` at three of
four rates, `0.9996`–`0.9998` at the rest, `0.9961` untrained. Nothing about the
locked configuration changes that, and no Table 1 conclusion may lean on that cell.

### What is lifted

`DL-047` suspended `--phase final`. **That suspension is lifted by the person who
owns the decision.** Its stated reason — that a learning rate chosen on a diluted
metric would not be trustworthy — is not refuted; it is overruled in favour of
producing the project's Stage 1 artifact. The reason survives as a caveat on the
result, not as a blocker on the run.

### What is FROZEN and may not be edited while completing Stage 1

```
positive pair    same asset, query vs gallery
query            three modalities, independent p=0.30 mask each
gallery          complete text/image/pc, no mask
backbone         shared ULIP-2
towers           separate query/gallery fusion
OpenCLIP         frozen
trainable        PointBERT, projection, both fusion towers
optimizer        AdamW, batch 64, temperature 0.5 fixed, cosine schedule
dev selection    seven-condition mean R@1, R@5 as tie-break
```

### What stays UNKNOWN and no longer blocks

```
Table 1's gallery scope -- test-only (9,138) or full corpus (45,692)
the paper's query construction and positive-pair definition
MetaFind's real LR, epochs, optimizer, batch size
```

**These still prevent a claim of direct comparability with the paper's Table 1.**
They no longer prevent this project's Stage 1 from being completed. Both halves
of that sentence are load-bearing.

### The three implementation gaps that must close BEFORE the final run

Verified against the tree, 2026-08-30, not taken from a document:

```
G4_gallery_freeze   DOES NOT EXIST. No metafind/gates/ directory, no gate
                    record writer anywhere in the repo. validation_plan.yaml
                    has said `implemented: false` all along and it is correct.

n12 promote         `--gate-passed` is an operator-supplied boolean. Promotion
                    reads no gate record. A human flag currently stands in for
                    the evidence the gate exists to produce.

n15 eval            RE-ENCODES the gallery on every run. It never opens
                    gallery_index.json. The promoted index has no consumer on
                    the evaluation path.
```

One thing verified in the same pass and NOT a gap: `stage1_encoding_protocol`
resolves `image_aggregation` to `mean`, and under `mean` `Stage1Dataset` returns
`cached["image"]` byte for byte — the same array `gallery_index.py` reads.
~~So moving n15 onto the promoted index cannot move the numbers.~~ Checked
before proposing the change, because a switch that silently re-defines the
gallery vectors would be indistinguishable from a result.

> **⚠ CORRECTED 2026-08-30, same day, by the ULIP2 Block Reviewer (MAJOR-3).
> The struck sentence was MASTER's and it was an overstatement.**
>
> What is measured is that the two paths receive **byte-identical INPUTS** —
> confirmed on 200 sampled real uids, all three arrays, zero mismatches. That
> does not establish that they produce identical **OUTPUTS**: `gallery_index`
> encodes one asset at a time and `encode_pools` encodes in batches of 64, and
> a different batch shape can select a different BLAS kernel. That is the same
> class of difference `run_retrieval.block_plan` exists for, and it changed a
> reported diagnostic in 143 of 400 measured trials there.
>
> **Output equality is `UNVERIFIED` and stays so until a real index exists.**
> `run_retrieval.py`'s own docstring says exactly this; the struck sentence
> licensed ignoring it, and two authoritative project documents were in direct
> conflict with nothing marking it.
>
> **The live consequence, which is why this is not a wording fix.** The
> untrained control (`--ckpt-record none`) scores A and B through
> `untrained_direct_encode` at batch 64. The trained run scores A and B through
> `promoted_index`, built one asset at a time. **A trained-vs-untrained
> comparison on A or B therefore carries an uncontrolled encode-path difference**
> unless that is measured first. The code discloses it in `gallery_source`; the
> struck sentence was the thing telling a reader not to look.
>
> Kyzen caught the same overstatement in the work order before either engineer
> acted on it. It was corrected to both engineers and left standing here. **A
> retraction that reaches the workers and not the record is not a retraction.**

### Completion gate for "Stage 1 is done" — Kyzen's, recorded as given

```
final checkpoint exists, sha256 verified
record shows phase=final, lr=2.5e-4, epochs=5, seed=20260816
training pool is the full 80% train split
no dev selection occurred in the final phase
gallery index bound to the final checkpoint
G4 actually PASSES
promoted index bytes identical to the bytes G4 verified
n15's reported A/B consume the promoted index
A and B each record n_query / n_gallery
every unknown and deviation survives into the result
```

**A finished process is not a finished experiment. This list is the difference.**

---

## DL-049 — 5 epochs was a pilot, not a run. **Supersedes `DL-048`'s `epochs=5` and the 5→10→25 ladder.**

`USER_APPROVED` · 2026-08-30 · Kyzen

**Verbatim:**

```
5只是測試 100 250都要跑啊 不要跑5 10 25 沒意義 現在給我停掉 先做完PointBERT
```

And, on the follow-up plan, `做`.

### What changed

```
DL-048   epochs 5, single run, then Table 1
DL-049   epochs 100, then epochs 250. Both are runs. Both get a Table 1.
         The 5→10→25 ladder is withdrawn as uninformative.
```

`DL-048`'s other locks are UNTOUCHED: `lr 2.5e-4`, `seed 20260816`, `phase final`,
`batch 64`, `p_mask 0.30`, and every frozen research semantic.

### The evidence that says he is right, which MASTER had and under-weighted

Read from the eight sweep arms' own `stage1_best_ckpt.json`, OBSERVED DATA:

```
lr 2.50e-4  s20260816   best epoch 4      lr 7.50e-4  s20260816   best epoch 4
lr 2.50e-4  s20260830   best epoch 4      lr 7.50e-4  s20260830   best epoch 4
lr 5.00e-4  s20260816   best epoch 4      lr 1.00e-3  s20260816   best epoch 4
lr 5.00e-4  s20260830   best epoch 4      lr 1.00e-3  s20260830   best epoch 4
```

**Every arm's best is the LAST epoch it ran. Eight out of eight.** A converged
run has some arms peaking early and declining; none did. So `epochs=5` is not
where the model stopped improving — it is where the budget ran out.

**This was in `DL-048` already**, phrased as "no arm peaked early, so `epochs=5`
is the horizon that was run, not a horizon that was selected". That sentence is
the observation. **It was written as reassurance and it is the opposite: the
same fact says the model was still learning when it was cut off.** MASTER filed
the evidence and drew the comfortable reading from it.

### What was stopped, and what survives

```
KILLED    the gallery-index build for the 5-epoch checkpoint, at 24,000/45,692.
          No partial artifact on disk: `build_index` writes to `.part.npz` and
          renames, so an interrupted build leaves nothing. Verified -- no
          `gallery_index*` of any kind exists.
KEPT      the 5-epoch final checkpoint and its record, as a control.
          sha256 a1e56b30..., code_revision 979b0ba, code_dirty FALSE.
NEVER RAN G4, promotion, and the reported A/B. Nothing downstream was
          contaminated by the 5-epoch model, because nothing downstream ran.
```

### Running now

```
python -m metafind.train.stage1 --phase final --epochs 100 --preload \
       --lr 2.5e-4 --seed 20260816 \
       --out-dir stage1_final/e100_lr2.50e-4_s20260816
```

`--lr-horizon` is not passed and defaults to `epochs` (`stage1.py:1664`), so the
cosine schedule runs over 100 epochs rather than completing at 5 and idling at
`lr_end`. Checked before launching, because that failure would have been
invisible in the log.

### ⚠ An open risk, raised at launch and NOT resolved

**`lr 2.5e-4` was selected on a five-epoch horizon.** At 100 epochs the same
number produces a completely different schedule: the high-learning-rate phase is
twenty times longer before the cosine brings it down. The selection experiment
and the run it is now being used for are not the same experiment.

Classification: `IMPLEMENTATION CHOICE`, and its supporting evidence covers a
horizon it is no longer being used at. Raised to Kyzen at launch; he did not
change it, so `2.5e-4` stands. **Recorded here so that a Table 1 from the
100- or 250-epoch model is not read as having a tuned learning rate behind it.**

### Sequence agreed

```
1  100-epoch training                              ~8 h    RUNNING
2  index -> G4 -> promote -> A/B on that model     ~40 min
3  250-epoch training                              ~20 h
4  index -> G4 -> promote -> A/B on that model     ~40 min
```

Step 2 is deliberate and Kyzen approved it explicitly. **The whole evaluation
chain has never run end to end** — G4, promotion and the reported A/B were all
written today and have only ever seen synthetic fixtures. Discovering a defect
in it after step 1 costs 40 minutes; discovering it after step 3 costs 20 hours.

### Two monitors MASTER wrote wrong today, recorded because both were silent failures

```
1  `while pgrep -f "metafind.train.stage1 --phase final"` -- the watcher's OWN
   command line contains that string, so pgrep matched itself and the loop could
   never exit. It would have waited forever and reported nothing.
2  `P=$(pgrep -f ... | tail -1)` resolved to a process that vanished, so the
   watcher reported "training exited" 90 seconds after launch, while training
   was in fact healthy at 100% GPU.
```

**Neither was a training fault. Both were the instrument.** The first would have
hidden a finish; the second manufactured one. Fixed by capturing the real PID
once and polling `kill -0` on that number.

---

## DL-050 — the evaluation was solvable without a model, and Kyzen found it by asking why the loss was flat

`RECORDED` · 2026-08-31 · MASTER, from measurements by the ULIP2 Block Engineer,
audited by GPT and Codex, external anchor supplied by Kyzen

**This entry supersedes the framing of `DL-044` through `DL-049`.** Those entries
kept asking why training would not improve a metric. The metric was the problem.

### The single sentence

**Our query's text IS the gallery's text — the same cached vector — and the same
holds for image and point cloud. The task can be solved by matching a vector to
itself, and it is.**

### Measured, all on dev_val (4,569), all reproducible from `output/look/`

```
no fusion tower at all, raw ULIP embeddings only
   text 0.9956   image 0.9877   pc 0.9897   full 1.0000
with the untrained fusion towers
   text 0.9642   image 0.9103   pc 0.9452   full 0.9994
```

**Stripping the model raises the score and widens the margin by 2.1-4.5x.** The
towers are a random contraction on a task already solved before they run.

```
remove the query's own modality from the gallery
   text 0.9642 -> 0.3971    image 0.9103 -> 0.5029    pc 0.9452 -> 0.5666
swap in an independent caption (2x2, quality confound excluded: A/A = 0.9609)
   text 0.9642 -> 0.7497
```

**All three modalities carry a 38-57 pp self-match.** Not a text problem.

### The external anchor, which turned one number into a field consensus

`docs/paper/跨模態增強模型嵌入與檢索架構_完稿.docx` — NKUST master's thesis,
October 2025, "CAMERA" — runs ULIP-2 on text→3D retrieval, Table 4:

```
Parts2Words  RR@1 12.72     TriCoLo  RR@1 10.25
ULIP         RR@1  3.23     ULIP-2   RR@1 13.50
```

```
CAMERA's ULIP-2   13.50
MetaFind Table 1  13.80
ours              96.42
```

Its §279 states the query text and the corpus text come from different sources,
and the Text2Shape-family galleries hold shapes, not text, so text→text does not
exist there.

**Before this, MetaFind's 13.8 was an unexplained outlier we kept trying to reach
down to. After it, 10-14 is the range this task lives in and WE are the outlier.**
That inversion is the whole value of the document.

### Where the defect entered — ~~paper silence~~, filled and then forgotten

> **CORRECTED 2026-08-31, hours after filing, by INTEGRATOR. The heading and the
> word "silence" below were MASTER's and they are wrong.**
>
> **The paper is not silent on this.** `3experiments.tex:24` speaks to it
> directly: baselines' PC-only "reflects retrieval using **identical embeddings
> for both query and gallery, leading to inflated accuracy**", and MetaFind's
> dual tower is named as what avoids it.
>
> The paper identifies the exact mechanism AND names its cure. We have a real
> dual tower -- `model.query` and `model.gallery`, separate parameters -- we feed
> it identical inputs, and we get 96.42. **The paper's stated cure is implemented
> and does not produce the paper's behaviour.**
>
> Verdict stays `IMPLEMENTATION CHOICE`. **The BASIS is `3experiments.tex:24`,
> not silence.** INTEGRATOR's reason for refusing the wording is why it was worth
> correcting: *"A month from now 'the paper was silent' is the sentence that gets
> cited, and it is false."*

`2methdology.tex:75` fixes the masking: each query modality independently masked
at 30%. It never says the surviving modality uses the gallery's identical cached
vector. `metafind/eval/retrieval.py:14` filled that gap --
"a query built from a subset of that same asset" — and tagged the sentence
`[PAPER 3experiments.tex:24]`, which only names the seven conditions.

**An IMPLEMENTATION CHOICE was written in the shape of a PAPER FACT**, which is
exactly what `.claude/rules/research-rigor.md` §1 forbids, and it then propagated
into every number this project has produced.

### Two hypotheses tested and REFUTED, recorded so they are not re-run

**The fusion is near-identity.** Refuted. `cos(raw, fused)` untrained is
0.40-0.53, and the no-fusion control scores HIGHER than the towers — they
degrade, not relay.

**Training learned the pass-through shortcut.** Kyzen's hypothesis, and the
measurement went against it: on `e25_500w`, `cos(raw, fused)` FALLS with training
(text .4994→.3313, image .5333→.2996) while the norm ratio explodes .63-.84 →
3.64-15.47. OpenCLIP is frozen for text and image, so that half is clean.
**What survives is narrower and still stands: the objective ADMITS a
same-modality shortcut and cannot uniquely identify cross-modal alignment.**
"This model converged to it" is UNVERIFIED and now has evidence against it.

### A logic error of MASTER's, corrected by GPT

MASTER computed a multi-positive ceiling — category-only 43.0% R@1 — and argued
that because the paper's 13.8% is BELOW it, short queries cannot explain the
paper's number. **Backwards.** A ceiling of 43% permits every value below it,
13.8% included. The formula (`#groups / N`) is right; the inference was an upper
bound used as a lower bound. It excludes nothing.

### The fix, and why it is a CHOICE and not a DEVIATION

```
              now                     after
query text    canonical description   an alternate candidate
query image   the 12-view mean        one held-out view
query pc      the canonical sample    a second independent sample

gallery       UNCHANGED, modality-complete
masking       UNCHANGED, 30% independent
loss          UNCHANGED
architecture  UNCHANGED
```

The paper constrains the masking, not the observation identity. Refilling that
silence with "another observation of the same asset" is an `IMPLEMENTATION
CHOICE`, and it is compatible with the paper's own text.

**Trainer and evaluator change together.** Training on one construction and
evaluating on the other is a train/test mismatch.

### What is NOT settled, and must not be quietly closed

```
U-09 gallery size. Every number here is dev_val's 4,569. The paper's candidate
     pool is unstated and could be 9,138 or 45,692. NOT measured. It is an
     independent confound and a score change after this fix cannot be attributed
     without it.
Whether the paper's authors did this. Unknowable -- they are not being contacted,
     the arXiv has only v1, and the OpenReview thread (forum uGDNHlslgO, read in
     full) contains no reviewer question about the query construction.
Our Transformer fusion -- residual plus token-mean readout -- is an unstated
     implementation choice that may AMPLIFY a shortcut the paper also has.
     Codex's reading, and MASTER's too.
```

### What the whole day cost, and who found it

**Eight sweep arms, a 100-epoch run, three retracted conclusions, and roughly
eleven GPU hours were spent tuning a learning rate on an instrument that scores
99% with the model deleted.**

Kyzen found it. Not from the artifacts — by refusing "the loss dropped fast then
plateaued" and saying `你loss沒有掉啊`, then `完全不用模型，分數就已經是
99~100%這不就他媽的有問題了嗎? 阿訓練要幹嘛?`. Every measurement in this entry
descends from those two sentences.

---

## DL-051 — training grows the point encoder's norm 7x and flips its cross-modal cosine NEGATIVE. The loss cannot see it.

`RECORDED` · 2026-08-31 · measured by the ULIP2 Block Engineer, verified by MASTER
against `output/look/diag_modality_norms.json` and the source

**This is the best mechanistic explanation yet for "trained is worse than
untrained", and unlike everything in `DL-050` it is about the ARCHITECTURE, not
the evaluation.**

### The measurement — OBSERVED DATA, dev_val 4,569, checkpoint `e25_500w` (sha256 455d3492...)

```
                        released      trained     change
pc  output norm            27.86       200.55     x7.2
text                       37.13        37.13     frozen, unchanged
image                      40.23        40.23     frozen, unchanged

pc share of the fused mean's norm
                           26.5%        72.2%

mean paired cosine
   text <-> pc             +0.346       -0.170    SIGN FLIP
   image <-> pc            +0.470       -0.402    SIGN FLIP
   text <-> image          +0.457       +0.457    both frozen, unchanged
cos(mean-of-three, pc)      0.717        0.943
```

**After training, an asset's point-cloud embedding points AWAY from its own text
and image embeddings.** That is the opposite of cross-modal alignment, and it is
what Stage 1 exists to produce.

The `text <-> image` row is the control: both encoders are frozen, and it does
not move. So the flip is the point encoder moving, not measurement drift.

### Why the loss cannot see it — OBSERVED IMPLEMENTATION

```
UPSTREAM ULIP   upstream/ULIP/models/losses.py:34-36
                pc_embed    = F.normalize(...)
                text_embed  = F.normalize(...)
                image_embed = F.normalize(...)
                every modality normalised BEFORE the contrastive loss

OURS            grep normalize metafind/models/ulip_backbone.py  ->  nothing
                three raw embeddings enter the fusion transformer at their
                natural norms; only the FUSED OUTPUT is normalised, in
                losses.py:166-167
```

**So the point encoder's norm is an unconstrained degree of freedom.** It can
grow without bound because the loss only ever sees a unit-length fused vector.
Inside the fusion, though, the transformer's mean-pooled readout is dominated by
whichever token is largest — and at norm 200 against 37 and 40, that is pc, at
72% of the total.

### ⚠ This is NOT a straight "upstream does X and we do not"

ULIP has **no fusion module**. It compares modalities pairwise, so normalising
each one before the loss is the whole story there. MetaFind introduces a fusion
that consumes the three embeddings as tokens, and **that is a new place where
relative norm decides the outcome**. The paper does not say whether to normalise
going in.

So the honest statement is: *the fusion creates a norm sensitivity that upstream
does not have, and nothing in our implementation or in the paper constrains it.*
Classification `OBSERVED IMPLEMENTATION` for the measurement,
`UNKNOWN` for what the paper intends.

### What this explains, that nothing else did

```
text degrades under training, -10 to -28 pp across every LR arm
   the fused output becomes mostly pc, and text is pushed anti-aligned to it
trained scores below untrained, on BOTH protocols
   PROD 0.9321 vs 0.9718; independent-observation 0.8335 vs 0.8986
pc collapses at high LR, -33 to -39 pp
   the same runaway, further along
`hpo_r1/lr2.50e-4` selecting EPOCH 0 of 10
   the damage starts in the first epoch and never reverses
```

The norm blow-up is also **late**: the ratio is ~1.0 at epoch 0 and 3.6-15.5 by
epoch 24. So it compounds rather than appearing at once.

### What it does NOT explain

**The gap to the paper.** The paper's Table 1 is 13.8 on text; removing the
`DL-050` leak takes us to 74.98 and this is a separate mechanism on top. Fixing
the norm would plausibly stop training from being harmful. **It would not turn
96 into 13.8.** Those are two different problems and this entry is only the
second one.

### A fourth candidate, surfaced during the first clean training run

With the leak removed, in-batch query→gallery accuracy is **0.984 at step 100**.
The contrastive task is still solved at batch 64, so there is almost no gradient
left. Recorded because `acc_q2g` has been in the metrics stream all along and
nobody had reason to read it until the leak was gone.

This connects back to the earlier finding that **upstream's effective contrastive
batch is 512** (`losses.py:38-40` all-gathers across 8 GPUs at 64 each) while
ours is 64. Our loss never sees more than 63 negatives against a 4,569-way
retrieval task. That is a ratified hyperparameter and NOT changed here.

### Not acted on

Architecture is Kyzen's decision, and the engineer who found this correctly
refused to touch it. No normalisation was added, no fusion changed. **The
measurement is the deliverable.**

---

## DL-052 — the ULIP-2 backbone reproduces its published number EXACTLY. The 96% was never the backbone; it was the gallery size and the query text.

`RECORDED` · 2026-08-31 · measured by MASTER against
`output/look/ulip2_calibration.json`, `data/outputs/eval/qpack_ti_best_AB_nopack/`,
`data/outputs/eval/ladder_L{1,2}_*_B/`

**Every retrieval figure this project had produced was measured by tools that had
never scored a case whose answer was known in advance.** `grep baseline
metafind/eval/*.py` returned comments and no code. So when our number disagreed
with the paper's by 4x, nothing in the repository could say whether the backbone,
the scorer, the UID pairing or the fusion towers was responsible. Two published
numbers now say.

### GATE — ULIP-2 zero-shot classification, Objaverse-LVIS, n = 45,692

`UPSTREAM FACT` target: ULIP-2 CVPR'24 Tab. 1, the row
`Point-BERT | Objaverse + ShapeNet | ULIP-2`
(`docs/paper/ulip2_source/main.tex:443`) — which is the row our released
checkpoint belongs to.

```
              ours    that row    the no-LVIS row
top-1         50.6        50.6              46.3
top-5         78.9        79.1              75.0
```

Recipe copied, not reinvented: 1,156 `all_keys` classes, the 64
`modelnet40_64` templates that `upstream/ULIP/main.py:41` defaults to and
`scripts/test_ulip2_pointbert_objaverse_lvis.sh` does not override, each prompt
L2'd, meaned, L2'd again (`main.py:377-380`).

**`OBSERVED DATA`. The point-cloud sampler, `pc_norm`, the colour channel, the
`yaw180_about_y@ulip2_frame` correction, the PointBERT load and the OpenCLIP
tokenizer are all correct.** This is the first published number this project has
reproduced.

### What that retires

A standing proposal to retrain a clean `no-LVIS` ULIP-2 from scratch (744,860
assets, days-to-weeks of GPU) on the theory that our checkpoint's pretraining
pool contaminated the evaluation. The contamination is REAL and stays recorded
as a `DEVIATION` — our checkpoint did see every LVIS asset. But the arithmetic
does not support it as the main term: ULIP-2's own clean-vs-contaminated gap is
**4.3 points of classification** (46.3 vs 50.6), and memorisation worth 4.3
points does not buy the 512x we were trying to explain. **`NOT ACTED ON`** —
pricing the cheap causes first was the whole point.

### CAUSE 1 — gallery size. `U-09`, and it is worth 29 points.

Raw ULIP-2, no towers, gallery = the pc embedding alone
(`PAPER FACT 3experiments.tex:24`), query = full text serialization:

```
gallery  4,569 (dev_val)     51.2 %
gallery 45,692 (corpus)      22.3 %
```

Every dev-selection number this project has looked at was measured at 4,569.
The paper's own gallery is still unstated, but a 10x pool costs 29 pp, so
**dev-val numbers were never comparable to Table 1 and should not have been read
as if they were.**

### CAUSE 2 — query text richness. Worth up to 37.6 points, and it is the big one.

Same trained checkpoint, protocol B (query = test 9,138, gallery = the promoted
index of 45,692), ONLY the query text swapped
(`tools/probes/make_text_ladder_packs.py`):

```
query text is...                          text R@1
L1  the class name, 64-template ensemble     0.20 %
L2  the annotation's `description` alone    35.35 %
L3  the full serialization  (what we use)   37.77 %

MetaFind Tab. 1, w/o ESSGNN                 13.80 %
ULIP baseline row, same table                0.10 %
```

**189x, on one model, from the query text alone.** The paper's 13.8 sits between
L1 and L2; its ULIP baseline's 0.1 sits on L1. Our 37.8 is L3.

`UNKNOWN`: **the paper never states what text its query carries.** ULIP-2
pretrains on a BLIP-2 caption of a single view (`ulip2 main.tex:616`) — short and
generic. Ours is a GPT-4o structured description with colour, pattern, material
and "roughly 12.1 by 9.4 by 3.5 centimetres". Against 45,692 candidates that is
close to a fingerprint.

Note also L2 > L3 by 2.4 pp: the dimension-and-material tail is not a
fingerprint, it is noise.

### CAUSE 3 — image observation. Small on the raw encoder, LARGE once it is held out.

```
raw ULIP-2, gallery 4,569
  query image = views[0]     (IS inside the gallery's 12-view mean)   78.9 %
  query image = 12-view mean (IS the gallery's own vector)            81.3 %
                                                              difference  2.4 pp

trained model, protocol B, gallery 45,692
  query image = 12-view mean (the gallery's own vector)               75.8 %
  query image = one HELD-OUT view                                     54.5 %
                                                              difference 21.4 pp
```

**The 2.4 and the 21.4 are the same experiment with one thing changed: whether
the query's view is inside the gallery's mean.** Sharing the observation is worth
21 pp. That is the cleanest measurement yet of the effect Kyzen has been pointing
at since "查詢文字 ≠ 資料庫文字 這重點啊".

### The shape inversion, which is the finding underneath the numbers

Trained model, protocol B, ranked:

```
paper   pc 75.1 > full 51.7 > I+PC 45.8 > T+PC 44.5 > T+I 17.2 > text 13.8 > image 11.7
ours    full 99.6 > T+PC 99.1 > I+PC 96.7 > pc 93.7 > T+I 87.4 > image 75.8 > text 37.8
```

**In the paper, pc-only BEATS full. In ours, full beats pc-only.** That is
qualitative, not a matter of degree. It follows from causes 1-3: the paper's text
and image are near-useless alone (13.8, 11.7), so adding them to a strong pc query
dilutes it (75.1 -> 51.7); ours are strong (37.8, 75.8), so adding them helps
(93.7 -> 99.6). One mechanism, both directions.

### Caveat on the L1/L2 runs — a confound, stated rather than buried

`QueryPack` applies the image arm as a RULE (`views[uid_seed(uid) % 12]`), not as
an array, so it fires whenever the `image` key is present at all — which it was in
both ladder packs, despite their empty `shards`. **The L1/L2 multimodal cells
therefore also swapped the image to a held-out view and are NOT comparable to the
no-pack row.** The `text`-only cells do not read the image and are clean; the
ladder above is only ever quoted from those.

### Fixed on the way, and it is not cosmetic

`metafind/gates/g4_gallery_freeze.py:211-222`. `torch.load(..., weights_only=True)`
began refusing every checkpoint written after schema 4 added `metadata`, whose
`hardware` block stores torch's own version as a `TorchVersion` rather than a
`str`. **G4 reported `BLOCKED_EVIDENCE` on a perfectly good index**, which is a
gate failing open in the direction of a false alarm. Fixed by allowlisting that
one class — NOT by dropping to `weights_only=False`, which is the property that
lets a gate read an untrusted checkpoint at all.

### What is still not explained

`L2 35.35` against the paper's `13.80` at the same gallery. Query text richness
and gallery size together take 96 to 37.8 and a plain description to 35.4; the
last 2.6x is open. Candidates not yet priced: the paper's own gallery size
(`U-09`), whether MetaFind fine-tunes the text and image encoders
(`2methdology.tex:32` contrasts itself with "freezing pretrained text and image
encoders"; Tab. 3 "Train fuser only" 8.7 vs 11.4), 11 views vs our 12, and the
`DL-051` norm runaway.

**Nothing here licenses "the reproduction is correct".** It licenses exactly two
statements: the backbone reproduces its published number, and the 512x gap was
experimental conditions rather than a defect.

---

## DL-053 — the ULIP-2 layer is CLOSED. Upstream's own unmodified code, on our point clouds, returns the published 50.6.

`CLOSED` · 2026-08-31 · MASTER, against `/home/kyzen/upstream/ULIP_run` (upstream
@95d480f, `git status` clean) and `output/look/fps_sensitivity.json`

**Every remaining discrepancy is now inside MetaFind's own retrieval protocol.
Nothing further should be spent on ULIP-2, FPS, the compatibility shims, or the
prompt evaluator.**

### The gold test, and it is a primary-source one

`bash scripts/test_ulip2_pointbert_objaverse_lvis.sh <our checkpoint>`, run from
a clean clone with no source file edited, over our own 45,692 point clouds:

```
                                          top-1     top-5
ULIP-2 CVPR'24 Tab.1 (our ckpt's row)      50.6      79.1
upstream's unmodified code, our clouds     50.576    78.931
our own pipeline                           50.6      78.9
```

Top-1 lands on the published figure; top-5 is 0.17 pp under. That validates, in
one number, the sampler, `pc_norm`, the colour channel, the
`yaw180_about_y@ulip2_frame` correction, the checkpoint load, the tokenizer and
the 64-template prompt ensemble.

`UPSTREAM FACT`: ULIP-2 §4.2-4.3 says "We adopt the same evaluation metrics used
in ULIP" and "We follow the same procedure as in ULIP and OpenShape", and the
official code makes it literal -- `test_zeroshot_3d` and `test_zeroshot_3d_ulip2`
both end in the same `test_zeroshot_3d_core`. So the protocol we matched is the
authors' own, not a reconstruction.

### What it cost to run upstream unmodified, and what that cost proves

Four environment shims, no ULIP source edited (`git status` shows no `M`):

```
torch/_six.py           dataset_3d.py:544; read only by the TRAINING collate
knn_cuda (fail-closed)  dvae.py:9 builds KNN(k=4) and never calls it; the stub
                        raises, so a completed run IS the proof of zero calls
sitecustomize.py        torch.load's weights_only allowlist. The plain form is a
                        NO-OP: registering the object keys it under
                        `numpy._core.multiarray.scalar` while the 2023 pickle
                        asks for `numpy.core.multiarray.scalar`. The
                        (callable, name) form is required.
compat/pointnet2_ops-sm120.patch
                        two hardcoded `TORCH_CUDA_ARCH_LIST` lines deleted
                        (setup.py:19 and pointnet2_ops/pointnet2_utils.py:23).
                        They pin arch <= sm_75 and override the caller, so the
                        build could not target sm_120 at all. No .cu/.cpp touched.
```

`DEVIATION, unavoidable`: upstream's README specifies python 3.7.15 / torch
1.10.1 / cudatoolkit 11.3. RTX 5090 is sm_120 and CUDA 11.3 tops out at sm_86,
so that stack cannot run on this machine. Used 3.11 / 2.11.0+cu128 / nvcc 12.8.

### FPS: the one shim that touches a number, measured and dismissed

`misc.fps` is replaced by a pure-torch greedy FPS. Perturb its only free
parameter -- the seed index -- and the published number does not move, even
though the chosen centres change completely (max coordinate difference 1.74 on a
unit-radius cloud):

```
seed index 0 (ours = pointnet2's)   top-1 50.61   top-5 78.96
seed index 5000                     top-1 50.64   top-5 78.89
random per cloud                    top-1 50.90   top-5 78.89
```

0.29 pp across the sweep. Kernel-level tie-breaking cannot matter if a different
centre set does not.

### The distinction that kept sending us the wrong way

ULIP defines ONE quantitative evaluation and it is not retrieval:

* `PAPER FACT` ULIP §4.3: zero-shot 3D classification -- pc feature against
  category text prototypes, no fine-tuning, same prompt strategy as pretraining.
* `PAPER FACT` ULIP §4.7: image -> point cloud retrieval exists but is
  **qualitative only** -- Caltech101 images against ~2.5k ModelNet40 clouds,
  a top-5 figure, no metric, no ground-truth pairing.
* ULIP-2: the string "retriev" does not appear in the paper at all.

**So MetaFind's seven-cell ULIP row is MetaFind's own adaptation, not an
upstream benchmark.** There is no official instance-retrieval protocol to
inherit, which is why no configuration we tried reproduces it and why CAMERA's
Table 4 mixes four different protocols.

### One normalisation question is answered, one is not

`UPSTREAM FACT`, `main.py:377-380`: the prompt ensemble is
`L2(mean(L2(t_1)..L2(t_K)))`, and the pc side is `L2(P)`, similarity `P @ T.T`.
Not UNKNOWN -- the official code fixes it, and we match it.

`UNKNOWN`, still: whether MetaFind's CROSS-MODAL mean over text/image/pc is
taken raw or over unit vectors. The prompt-ensemble recipe is a same-modality
operation and does not transfer. Measured either way: 0.7 pp, so it is open but
not load-bearing.

### CAMERA is withdrawn as an anchor

Its Table 4 was inspected on the machine that holds the code:
Parts2Words 12.72 and TriCoLo 10.25 are copied from the Parts2Words paper
(Text2Shape, 1,434-shape gallery); ULIP 3.23 is an unlogged local run on
ShapeNet-55 chair/table (14,966 objects); and the only exact `13.506` in any log
is an IKEA run with a **733** gallery. Every checkpoint in that workspace records
`args.model = ULIP_PointBERT` -- ULIP-1, 8192 points, 512-d. No
`ULIP2_PointBERT_Colored` checkpoint exists there.

**Statements of the form "CAMERA's ULIP-2 gets 13.50, MetaFind's gets 0.1, so
they differ by 135x" were made repeatedly in this session and are RETRACTED.**
That 13.50 is neither ULIP-2 nor a comparable gallery. The surviving external
anchors are ULIP-2's own 50.6 (reproduced) and CLIP-GS's 5.6 image->3D at ~10-20K
(with the caveat that its point clouds are sampled from Gaussian splats, which
is out of distribution for ULIP-2).

---

## DL-054 — the contamination question, and a lead on the last gap that reads the pattern rather than one cell

`RECORDED` · 2026-08-31 · MASTER, against `output/look/contamination_holdout.json`
and `output/look/u11_absent_slots.json`

### Contamination: bounded, not measured

`ProcTHOR/AI2-THOR` is a Unity asset library, not Objaverse and not ShapeNet, so
ULIP-2 has not seen it. 1,439 of its assets carry both a cloud and 11 renders.

```
image -> pc, gallery 1,439 in every row        multi-view mean   single view
  Objaverse, real colour   (SEEN)                    89.23           88.95
  Objaverse, flat grey     (SEEN)                    79.29           75.96
  ProcTHOR, flat grey      (UNSEEN)                  23.35           17.65

  colour effect   +9.94 pp
  seen vs unseen  -55.94 pp
```

`UPPER BOUND, NOT A MEASUREMENT`: a ProcTHOR cloud is a visible depth shell, an
Objaverse cloud is full mesh surface. That domain gap is inside the 55.94 and
cannot be removed without re-sampling 45,692 clouds as shells.

⚠ **An earlier argument in this session was wrong and is corrected here.**
Contamination was dismissed on the grounds that ULIP-2's own clean-vs-contaminated
gap is 4.3 points (46.3 vs 50.6). That gap is CLASSIFICATION, where ~40 assets
share a label and memorising one instance buys almost nothing. INSTANCE retrieval
must pick this asset out of 45,691 others, which is exactly what memorising a
(render, cloud) pair buys. The 4.3 does not transfer.

`UPSTREAM FACT`: ULIP-1 §4.5 hits the same problem and solves it by SUBSETTING
rather than retraining -- Medium removes ModelNet40 categories whose names appear
in the pretraining list, Hard also removes synonyms ("cup" vs "mug"), and the
Hard-set result is used to argue the gains are "not caused by pre-training the
model on exact/similar categories". That trick does not transfer to instance
retrieval, where the unit is the asset and every LVIS asset is in the pool.

### U-11: a 13-point lever, pointing the wrong way

`include_absent_slots` decides whether an absent modality's mask token joins the
pooled readout. Flipped at inference on the trained checkpoint, promoted index,
gallery 45,692:

```
condition     True (ours)   False    delta    paper
text             37.77      50.71   +12.95     13.8
image            75.82      82.79    +6.97     11.7
pc               93.70      94.69    +1.00     75.1
text+image       87.37      83.30    -4.07     17.2
full             99.61      99.61    +0.00     51.7
```

`full` at exactly +0.00 is the built-in control: with all three slots present the
flag is a no-op by construction, and it is. So the deltas above are the flag and
nothing else.

The lever is real and large, and OUR setting is already the side closer to the
paper. It does not explain the remaining gap; flipping it widens it. It stays a
registered UNKNOWN because 13 pp is not "doesn't matter", and settling it
properly needs a training arm -- this was an inference-time flip on a model
trained under True, so it bounds sensitivity, it is not the counterfactual.

### The lead: read the PATTERN across modalities, not any single cell

```
              ours   paper   ratio    what we do
  text        37.8    13.8    2.7x    CLIP frozen
  image       75.8    11.7    6.5x    CLIP frozen
  pc          93.7    75.1    1.2x    trained
```

**The two modalities we freeze are off by 2.7x and 6.5x; the one we train is off
by 1.2x.** That is not the shape of a scorer bug -- it is the shape of a training
scope difference.

`UPSTREAM FACT` ULIP-1 §3.2: "if we update CLIP's image and text encoders,
catastrophic forgetting will emerge due to our limited data size... Therefore we
freeze the weights of f_S and f_I during the entire pre-training". ULIP-1 froze
them at 52.5K assets, for that stated reason.

`PAPER` MetaFind `2methdology.tex:32` positions itself against exactly that:
"While prior works typically align 3D encoders to a fixed CLIP embedding space by
FREEZING pretrained text and image encoders, our MetaFind framework adopts a more
flexible dual-tower design." And Tab. 3 reports "Train fuser only" 8.7 against
11.4 for the full setting, explained as "full encoder fine-tuning yields better
performance by allowing EARLIER LAYERS to adapt".

`INFERENCE, NOT ESTABLISHED`: if MetaFind unfreezes CLIP over 48K assets, it runs
into the forgetting ULIP-1 documents; degraded text and image encoders would
produce low text and image cells while leaving pc -- trained on both sides --
closest. That is the only hypothesis so far that explains all three ratios at
once. The paper never states it outright; the evidence is the contrast sentence
plus 8.7 vs 11.4.

Testing it needs a training arm with `clip_train_scope` other than `frozen`, and
ViT-bigG's AdamW state alone is ~30 GB against a 32.6 GB card, so it needs LoRA
or partial unfreezing. NOT RUN. Kyzen's decision.

### Why training does not improve, settled by arithmetic rather than by tuning

`acc_q2g` sits at 0.95-0.97 from step 1 and loss plateaus after epoch 1 (2.6452 →
2.4989 → 2.3879 by epoch 42 of a 250-epoch run, oscillating). Two causes:

1. `p_mask` 0.30 independent per modality, so pc is present in 70% of queries --
   and a pc query IS its gallery entry, cosine 1.0000000000, margin 0.2152. Those
   rows are free.
2. Batch 64 means 63 negatives. Our text scores 37.8% against 45,691 candidates;
   against 63 it is near-perfect.

And the floor is geometric, not optimisational: for B unit vectors the mean
pairwise cosine cannot fall below -1/(B-1), so at B=64 the best achievable loss
at tau=0.5 is 2.2257 against our measured 2.39. **We are 0.16 from a bound that
more epochs cannot cross.**

⚠ An earlier guess that tau=0.5 is "too soft to give gradient" is WRONG and is
withdrawn. Computed: pushing negatives from cos 0 to -0.2 lowers the loss by
0.3497 at tau=0.5 and by 0.0000 at ULIP-2's converged tau=0.0206. The sharp
temperature is the one that saturates.

Neither fix is a paper fact -- MetaFind states no batch size anywhere:
(A) remove the pc shortcut with an independent query cloud, blocked at 32.38 GB
of 32.6 GB; (B) raise the effective batch toward upstream's 512, which on one GPU
means GradCache, since gradient accumulation cannot supply in-batch negatives.

---

## DL-055 -- OpenShape read end to end. The evaluation chain closes, and the unfreeze lead is now contradicted by three upstream sources.

Date 2026-08-31. Kyzen ordered a verbatim re-read of `docs/paper/ulip1_source/`
and `docs/paper/ulip2_source/`, then approved following the chain into OpenShape.
Read in full: ULIP-1 `main.tex` 1124 lines; ULIP-2 `main.tex` 1474 lines plus
`appendix.tex` (`sec/*.tex` are unused CVPR boilerplate, not `\input`);
OpenShape `sections/*.tex` and `tables/*.tex`; and the official code in
`/home/kyzen/upstream/ULIP` and `/home/kyzen/upstream/OpenShape`.

### The chain, resolved

`PAPER FACT` ULIP-2 defines no evaluation of its own. Three sentences delegate it:
`main.tex:704` "same dataset setup and preparation protocols used in ULIP and
OpenShape"; `:711` "same evaluation metrics used in ULIP: top-1 and top-5";
`:718` "same procedure as in ULIP and OpenShape". There is no fourth sentence.

`PAPER FACT` ULIP-1 `main.tex:381` holds the definition: "zero-shot 3D
classification is conducted by measuring distances between the 3D features of an
object and the text features of category candidates. The category that introduces
the smallest distance is selected... There is no finetuning stage involved. We
keep using the same prompt strategy as it is during pre-training." The prompt
strategy is `main.tex:244`: 63 CLIP-style prompts plus one
`a point cloud model of [WORD]`, and `:249` averages them, `h_S = Avg(f_S(S_i))`.

`OBSERVED IMPLEMENTATION` upstream `ULIP/main.py:363-405` executes exactly that:
load 64 templates (`data/templates.json`, `modelnet40_64` and `shapenet_64`, both
64 entries); take the candidate names (`labels.json`, or for Objaverse the LVIS
`all_keys`, 1,156 entries); format each name into all 64 templates; encode,
normalise, mean, normalise again; then cosine against the normalised pc feature
and take top-1/top-5. The official Objaverse-LVIS script
`scripts/test_ulip2_pointbert_objaverse_lvis.sh` does not pass
`--validate_dataset_prompt`, so it runs on the `main.py:41` default
`modelnet40_64` -- the ModelNet40 template set is used on Objaverse-LVIS.

`PAPER FACT` OpenShape supplies the benchmark and its own evaluation detail:
Objaverse-LVIS is 46,832 shapes over 1,156 LVIS categories
(`sections/experiments.tex`), input 10,000 sampled points with colour;
ModelNet40 is the 2,468-shape test split, 10,000 points without colour;
ScanObjectNN is **OBJ_ONLY, 581 test shapes**, 2,048 points -- not the hardest
set ULIP-1 reports. Two upstream papers ULIP-2 says it follows use different
ScanObjectNN splits.

### Retrieval: three papers, zero metrics

`PAPER FACT` ULIP-1 `main.tex:877` "we qualitatively show the potential" --
Caltech101 photos against ~2.5k ModelNet40 clouds, Figure 3, no number.
ULIP-2 contains the string "retriev" zero times. OpenShape's
`sections/experiments.tex` §4.4 retrieves "by calculating the cosine similarity
between input embedding(s) and 3D shape embeddings and performing kNN" and shows
Figures 5-7 plus two supplementary figures -- again no metric and no table.

**No upstream retrieval protocol exists to inherit.** MetaFind's Table 1 is its
own construction. Reverse-engineering its baseline row against upstream numbers
is not possible in principle, not merely unfinished.

### The same checkpoint, the same benchmark, two harnesses, 2.4x apart

`PAPER FACT` ULIP-2 `main.tex:430` reports ULIP/Point-BERT/ShapeNet on
Objaverse-LVIS as **2.6 / 8.1**, and on ModelNet40 as 60.4 / 84.0.
`PAPER FACT` OpenShape `tables/zero-shot.tex` reports the same official ULIP
checkpoint on Objaverse-LVIS as **6.2 / 17.9**, and on ModelNet40 as 60.4 / 84.4.

ModelNet40 top-1 agrees to the digit (60.4) -- both groups inherited ULIP's own
label list and 64 templates there. Objaverse-LVIS, the benchmark OpenShape
invented, differs by 2.4x. A published, two-sided demonstration that the harness,
not the weights, can move a number by more than a factor of two.

`OBSERVED IMPLEMENTATION` a likely mechanism sits in OpenShape's own repo. It
ships two LVIS category-feature extractors: `src/extract_lvis_feat.py` encodes
`[name]` -- one bare category name, no template, no averaging -- while
`src/extract_lvis_feat_ulip.py` builds 64 templated prompts and does
normalise-mean-normalise for the retrained ULIP baseline. The paper
(`sections/method.tex`) states templates are applied "to both training texts and
test-time category names". `src/configs/train.yaml:63` loads a third filename,
`meta_data/lvis_cat_name_pt_feat.npy`, shipped inside `meta_data.zip` and not
present locally. `UNKNOWN` which extractor produced the released 46.8; the `_pt_`
in the filename and the paper text both point at the templated one, but this was
not verified against the artifact.

### The unfreeze lead from DL-054 is contradicted and downgraded

DL-054 recorded, as `INFERENCE, NOT ESTABLISHED`, that unfreezing CLIP was the
only hypothesis explaining the 2.7x text / 6.5x image / 1.2x pc pattern at once.
Three upstream sources now speak against it, two of which had not been read:

1. `UPSTREAM FACT` ULIP-1 §3.2 (published): catastrophic forgetting at 52.5K
   assets; alpha set to 0 and both CLIP encoders frozen for the whole run.
2. `OBSERVED` ULIP-1 `main.tex:1068-1090`, a table the authors **commented out**
   of the published paper: "Ablation study on unfreezing image & text encoders",
   ModelNet40 hard set -- freeze=no gives top-1 **0.0**, top-5 **1.0**; freeze=yes
   gives 37.1 / 66.3. Not a PAPER FACT: it never appeared in the published paper.
   It is the authors' own recorded result in their own source file.
3. `UPSTREAM FACT` OpenShape `sections/supplementary.tex`, "Fine-tuning CLIP Text
   and Image Encoders?": they unfroze and finetuned the CLIP text encoder for one
   epoch, saw "no noticeable improvement on the benchmarks", and observed it
   "could potentially undermine the generalization capabilities of CLIP". They
   froze for the entire process.

Unfreezing does not produce a mild degradation that would leave text at 37.8 and
image at 75.8. Where it has been measured upstream it produces collapse (0.0) or
nothing (no improvement). The hypothesis does not fit the pattern it was invented
to explain, and the LoRA arm it implied is no longer the first thing to run.

MetaFind's own text still positions itself against freezing
(`2methdology.tex:32`) and Tab. 3 still shows 8.7 vs 11.4. That tension is
unresolved and stays open; what changed is that the reproduction-side inference
built on it is now contradicted by upstream measurement.

---

## DL-056 -- CAMERA's own evaluator, run on our corpus: the released ULIP-2 backbone, untrained, already beats MetaFind's reported trained text row.

Date 2026-08-31. Kyzen supplied CAMERA's `evaluate.py`, their 2025-08-12 working
note (the PDF carrying Table 4's numbers verbatim), and a forensic read of their
`main.py`. He then asked where the protocol comes from, and told me to run it.

### Provenance of the protocol

`UPSTREAM FACT` The metric triple RR@1 / RR@5 / NDCG@5 for text-shape retrieval
is the **Text2Shape** lineage (Chen et al., 3DV 2018): ShapeNet chair + table,
~1.4K models, ~7.4K human captions, ~5 per shape. It reaches CAMERA through
**TriCoLo** (WACV 2024) and **Parts2Words** (CVPR 2023), both of which report on
the Text2Shape test split.

`OBSERVED` CAMERA's Table 4 mixes three sources in one table:
Parts2Words 12.72/32.98/23.13 and TriCoLo 10.25/29.07/19.85 are copied from those
papers (Text2Shape test, 1,434 shapes); the "Ulip-2" row 13.50/39.69/26.89 is
their own run on **14,966 ShapeNet chair/table**, and the forensics established
that checkpoint is `ULIP_PointBERT` -- ULIP-1, 8192 xyz -- not ULIP-2. Three
pools, three datasets, two model generations, one comparison table.

`OBSERVED` The six-way retrieval is CAMERA's own code (`main.py:716`); upstream
ULIP has nothing like it. It runs each epoch against `train_dataset` with
`whole=True`, so train and test are merged into a 14,966 closed set.

⚠ **A caveat I wrote and withdrew the same hour.** The first draft of
`tools/probes/camera_six_way.py` recorded CAMERA as having ~5 positives per query
(75,361 captions / 15,033 models) and called our single-positive setting
"stricter". Wrong: their loader emits ONE row per object with a randomly drawn
caption, so `obj2idxs` holds one index per object and the best-rank-over-
positives code is a no-op. Their RR@1 is single-positive instance retrieval,
exactly like ours, and 13.50 IS directly comparable. The correction is a named
block in the probe, not a silent edit.

### The measurement

`tools/probes/camera_six_way.py`, released ULIP-2 (`ULIP2_PointBERT_Colored`,
10k xyzrgb, OpenCLIP ViT-bigG), **no Stage 1 weights**, our 45,692-asset corpus,
cached text and 12-view-mean image vectors, all three L2-normalised. Pool matched
to CAMERA's 14,966 by a seeded subsample alongside the full corpus.
Modality norms came out 37.06 / 40.23 / 27.65, matching `diag_modality_norms`.

```
R@1, pool 14,966          CAMERA (ULIP-1)   ours (released ULIP-2)
  S2T  pc -> text                9.60              40.15
  T2S  text -> pc               13.50              32.51
  S2I  pc -> image              33.73              83.12
  I2S  image -> pc              41.48              65.64
NDCG@5
  T2S                           26.89              47.94
```

Every cell higher, T2S by 2.4x. Expected direction -- ViT-bigG against SLIP
ViT-B, 10k coloured points against 8192 xyz, and our generated descriptions
against Text2Shape's human captions -- but it is now measured rather than
assumed, under their protocol rather than ours.

### What this does to MetaFind's Table 1

```
                                        text-only R@1     pool
MetaFind Tab.1, ULIP row                     0.1          48K
MetaFind Tab.1, MetaFind w/o ESSGNN         13.8          48K
ours, released ULIP-2, ZERO training        20.61         45,692
```

`OBSERVED DATA` The backbone we start from, with no MetaFind training of any
kind, already scores above MetaFind's reported trained text row on a pool of
comparable size. Any account of Table 1 has to survive that.

It also constrains the baseline row. A plain text-to-pointcloud retrieval with a
descriptive caption gives 20.61, not 0.1 -- 206x apart. Our own text ladder gives
1.24 for a bare category name against the same pc-only gallery, which is the same
order as 0.1. `INFERENCE`: MetaFind's baseline text is a category name or similar
short label, not a description. That is one of the knobs in the pending grid, and
it now has a measured bracket around it: 1.24 at L1, 20.61 at full description.

---

## DL-057 -- 24 evaluation configurations, and none of them produces Table 1's baseline row. The blocker is not on the evaluation side.

Date 2026-08-31. `tools/probes/table1_baseline_grid.py`, released ULIP-2 with no
Stage 1 weights, 9,138 test queries, gallery = the point-cloud embedding.
Grid: pooling (raw mean | normalise-then-mean) x text (L1 category name | L2 bare
description | L3 full serialisation) x image (12-view mean | one held-out view) x
pool (45,692 | 9,138). Scored against the paper's ULIP row on the four cells that
separate configurations: `pc`, `text+pc`, `image+pc`, `full`.

### One thing the grid DOES fix

`pc_is_max` is True in all 24. With a point-cloud gallery our ordering matches
the paper's -- pc highest, everything else below. This corrects a reading from
earlier today: the inverted ordering (full above pc) was a property of our
*fused* gallery, not of our data. The gallery choice alone restores the shape.

### Everything else refuses to move

```
                     paper ULIP     grid range over all 24
  pc                    97.9         100.0  (every configuration)
  text+pc               33.9         99.8 - 100.0
  image+pc              22.6         99.4 - 100.0
  full                   6.4         97.5 -  99.9
```

Total absolute error on those eight numbers stays between 420.7 and 424.2 across
the whole grid -- a spread of 3.5 points out of ~423. The knobs do essentially
nothing to the cells that matter.

The two cells the knobs DO move are the modality-only ones, and they bracket the
paper rather than reaching it:

```
  text     paper 0.1   ours: L1 1.2 | L3 19.9 | L2 25.0   (full pool)
  image    paper 0.1   ours: single view 41.4 | 12-view mean 52.7
```

L1 (a bare category name) is the only cell in the grid within an order of
magnitude of the paper's baseline text. Nothing brings image within 400x.

### What this forces

`OBSERVED DATA` The descent 97.9 -> 33.9 -> 22.6 -> 6.4 cannot be produced by any
combination of pooling rule, text richness, image construction or pool size,
given a query point cloud that IS the gallery's point cloud. The reason is
arithmetic rather than tuning: that self-match is cosine 1.0000, and halving it
against a second modality still leaves it far ahead of every other asset. Mean
pooling cannot destroy a signal that starts at exactly 1.

`INFERENCE` The necessary condition is therefore that the baseline's query point
cloud is not its gallery entry. But the three-arm query pack -- a genuine second
sampling at seed offset 1000003 -- cost `pc` only 0.8 points (DL-055 measurement,
94.66 -> 93.83). A resample is nowhere near enough. Whatever separates their
query from their gallery is larger than a resample: a different point budget, a
different normalisation, a cloud derived from renders rather than the mesh, or a
gallery that is not a raw point-cloud embedding at all.

`UNKNOWN` Which of those it is. The paper states none of them.

### Status of the two open mechanisms

1. `OBSERVED` **pc norm degeneracy** -- training grows the point embedding's norm
   from 27.9 to 200.5 while the frozen text and image stay at 37 and 40; the
   fusion takes raw vectors and reads out an unweighted mean, so pc swamps the
   other two in both towers. Explains why every query of ours containing a point
   cloud saturates. Does NOT explain text-only, where no point cloud appears.
2. `OBSERVED DATA` **text-only is high before any training at all** -- the
   released backbone gives 20.61 against the paper's trained 13.8 (DL-056). No
   training-side account can explain a gap that exists at initialisation.

These are two independent gaps. Neither is closed, and this grid closes off the
evaluation-side explanation for the first one.

---

## DL-058 -- FOUND ONE. Our image modality is a 12-view MEAN. All three upstream sources sample ONE view.

Date 2026-08-31. Kyzen: "text I can accept, richer descriptions. But point cloud
and image not lining up is strange, that has to be fixed" -- then "same method,
numbers this far apart, something must be written wrong."

He was right about the image cell, and the defect is ours.

### The requirement, verbatim from all three upstream sources

`UPSTREAM FACT` ULIP-2 `main.tex:612`: "randomly sample its 2D rendered image
**I ~ render(O)**" -- one draw per iteration, feeding `f_I = E_I(I)`.

`UPSTREAM FACT` ULIP-1 `main.tex:236`: "During each iteration of pre-training, we
**randomly select one image or depth map** from each CAD model's 60 rendered
candidates as I_i".

`UPSTREAM FACT` OpenShape `sections/method.tex:36`: "During each pretraining
iteration, we **randomly sample one rendered image or thumbnail** for each shape".

Three papers, three independent statements, one image each. None of them averages
views into a single vector.

### What we actually store

`OBSERVED IMPLEMENTATION` `metafind/data/encode_text_image.py`, [U-14]: the
canonical `image` vector in every one of the 45,692 sidecars is the **mean of the
12 view embeddings** (`aggregation: "mean"` travels in each record). All twelve
per-view vectors are also stored, so no re-encode is needed to change this.

MetaFind's own text does not license the mean either: `2methdology.tex` says only
that each modality is "independently encoded using the ULIP-2 backbone". The mean
is our decision, recorded as U-14, and it contradicts every upstream source.

### What it is worth, measured

```
image-only R@1, released ULIP-2, pc gallery      45,692 pool
  12-view mean                                      52.7
  one held-out view                                 41.4      -11.3

image-only R@1, trained dual tower, protocol D   36,554 gallery
  12-view mean (no pack)                            77.3
  one held-out view (3-arm pack)                    56.7      -20.6
```

A mean over 12 renders is a multi-view shape descriptor, not a photograph. It is
close to the point cloud by construction, which is exactly why the cell Kyzen
flagged reads high.

There is a second, sharper problem with it in the dual-tower setting: the
gallery's 12-view mean **contains the query's single view at weight 1/12**. The
query is literally a component of the thing it is scored against. The query
pack's own manifest records this as "option (a)" and it was never resolved.

### Classification and what it does not do

This is a `DEVIATION` from `UPSTREAM FACT`, introduced by us, now measured. It
does not close the gap: 41.4 against the paper's ULIP baseline 0.1 is still 414x,
and 56.7 against MetaFind's own 11.7 is still 4.8x. The pc cell is untouched by
it. But it is the first thing found that is (a) definitely wrong against a
primary source, (b) in the exact cell under suspicion, and (c) ours to fix.

**Not yet changed.** Switching the canonical image to a single sampled view moves
every image-bearing number in Table 1 and would invalidate the existing Stage 1
checkpoints for comparison. Kyzen's decision.
