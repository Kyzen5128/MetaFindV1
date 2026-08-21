# USER DIRECTIVES LOG — D14_n05-v5-reannotate

**Filed by:** D14 executor, at the user's explicit instruction ("先把我做的決定先跟 master 報備一下").
**Date:** 2026-08-21T16:28+0800
**git HEAD:** `468bbac999d0c064fbc1bd098910f82b980dd18e`
**Task status:** `READY` — **unchanged.** Phase 1 has not begun.

Companion document: `MASTER_IMPACTING_FINDING.md` (same directory) carries the *findings*.
This file carries the *user's instructions* and what D14 did in response.
They are kept apart per `WORKFLOW.md` §13A.

---

## Reading rule for Master

Everything in §1 is **what the user actually instructed**, recorded in the user's own words.

Everything in §3 is **what the user has NOT ruled on**. D14 makes no claim of approval there.

**Nothing D14 or Master recommended in conversation has been approved.** Where a recommendation was
made, it is marked as such and remains a *proposal*. Per `WORKFLOW.md` §13B, only the user's
`APPROVE` makes a material decision `FINAL ACCEPTED`, and no such approval has been given for
anything in this log beyond the operational instructions listed in §1.

---

## 1. Directives issued by the user

### U-1 — Hold. Do not begin execution.

> 「等等先停下不要馬上做」

**Classification:** execution-control directive.
**Effect:** D14 read the §3 required context and then stopped. `TASK.md` Status was **not** changed
from `READY` to `ACTIVE`, because Phase 1 never started.
**Status:** in force.

---

### U-2 — The annotation model is NOT to be settled yet. Survey the options first.

> 「模型設定我們先不要定 幫我查宜下目前最新的模型可以選用哪些?」

**Classification: MATERIAL.** This is the one directive Master must act on.

**Why it is material:** `TASK.md` §7 *Explicit Non-Scope* currently reads

> ❌ **Do not change the annotation model.** GPT-4o is unavailable; deviation `D-2` stands.

and `n05_v5_design.md` states the same. The user has **re-opened** that question.
A closed non-scope item and an open user question cannot both stand.

**What D14 did:** produced a read-only survey of currently available models — no code, no config,
no model selection. **No model was chosen.**

**What Master must reconcile:** whether §7's Explicit Non-Scope line is now suspended, amended, or
whether the survey is informational only and §7 stands. **D14 did not decide this and did not edit
`TASK.md` §7.**

**Related finding:** `MASTER_IMPACTING_FINDING.md` **F-2** — the premise "GPT-4o is unavailable"
is not established, and the paper-era snapshot `gpt-4o-2024-05-13` is scheduled for shutdown
**2026-10-23**. That finding is `PLAUSIBLE`, not `CONFIRMED`; the vendor documentation and secondary
sources conflict and D14 could not resolve the conflict without an API key.

---

### U-3 — Download Qwen3.8-27B to `/mnt/data1/kyzen/`, and report to Master.

> 「先下載Qwen3.8-27B 順便回報給Master 我要載到/mnt/data1/kyzen/」

**Classification:** operational instruction. **Not** a model selection.

**Explicitly recorded:** obtaining the weights is **not** the same as adopting the model.
The user issued U-2 (do not settle the model) before U-3. D14 treats U-3 as acquisition, not
adoption, and **no code, prompt, validator, or version constant references this model.**

**What D14 did:**
```
repo         Qwen/Qwen3.8-27B     Apache 2.0, released 2026-08-14
destination  /mnt/data1/kyzen/models/Qwen3.8-27B
method       huggingface_hub.snapshot_download(local_dir=..., max_workers=8)
size         55.6 GB / 18 safetensors shards (32 files)
reached      3.6 GB before being stopped under U-5
```

**Verified from the downloaded `config.json`, not from documentation:**
```
architectures        ['Qwen3_5ForConditionalGeneration']
language_model_only  False          <- vision path is enabled
vision_config        depth 27, hidden 1152
text_config          hidden 5120, 64 layers, bfloat16
transformers_version 5.8.0.dev0     (host has 5.15.0)
```
`[OBSERVED DATA]`

**Constraint the user was told:** 55.6 GB at bf16 **does not fit** the host's 32,607 MiB of VRAM.
Running it requires quantization. `[INFERENCE — VRAM figure is from vendor-adjacent web sources,
not measured on this host]`

**Path note:** the destination is outside the repository and outside `data/outputs/`.
No path listed in `TASK.md` §9.2 PROTECTED was touched.
See `MASTER_IMPACTING_FINDING.md` **F-1** — `/mnt/data1` exists on this host, contradicting
`CLAUDE.md` §9. **D14 did not edit `CLAUDE.md` or `CONTEXT.md`.**

---

### U-4 — Install the supplied HuggingFace token permanently on this machine.

> 「幫我設定好這台以後用這個hf-token」

**Classification:** environment configuration. **Not research-relevant.** No scientific behaviour changes.

**What D14 did:**
```
/home/kyzen/.hf_token.env                        chmod 600   sourced from ~/.bashrc
/home/kyzen/.cache/huggingface/token             chmod 600
/home/kyzen/data/MetaFind/models/hf-cache/token  chmod 600   (the HF_HOME paths.py sets)
```
`~/.bashrc` gained one `source` line; the secret itself is **not** written into `.bashrc`.
Verified resolving in three contexts (fresh login shell · project `HF_HOME` · default `HF_HOME`).
All paths are outside the repository and cannot be committed.

**Reported to the user:** the token was pasted in plaintext and now sits in this session's
transcript. Rotation was recommended. **The user has not said whether they will rotate.**

---

### U-5 — Stop D14's download; hand over the command; the user will run it.

> 「下載給我指令 我來跑 你先中斷 我想自己看進度」

**What D14 did:** terminated the download process. **3.6 GB and 8 `.incomplete` files were
deliberately retained** so the user's run resumes rather than restarts. Handed over the `hf download`
command and the completion checks.

**Status:** the download is now the user's process. D14 is not monitoring it.

---

## 2. What D14 has changed on disk

| Path | Change | Authority |
|---|---|---|
| `/mnt/data1/kyzen/models/Qwen3.8-27B` | partial download, 3.6 GB, incomplete | U-3 |
| `/home/kyzen/.hf_token.env` | created, 600 | U-4 |
| `/home/kyzen/.bashrc` | +1 `source` line | U-4 |
| `/home/kyzen/.cache/huggingface/token` | created, 600 | U-4 |
| `/home/kyzen/data/MetaFind/models/hf-cache/token` | created, 600 | U-4 |
| `workflow/tasks/D14_n05-v5-reannotate/MASTER_IMPACTING_FINDING.md` | created | `TASK.md` §11, `WORKFLOW.md` §18.2 |
| `workflow/tasks/D14_n05-v5-reannotate/USER_DIRECTIVES.md` | this file | user instruction |

**Nothing else.** Specifically **not** changed:

- `metafind/data/annotate.py`, `annotate_run.py` — prompt, validator, `PROMPT_VERSION`,
  `SCHEMA_VERSION`, `VALIDATOR_VERSION`, contract fingerprint: all untouched
- `data/outputs/annotations/**` — not read for mutation, not written, not backed up
- the 3 legacy-v1 residuals — untouched. **`D0-003` remains UNRESOLVED**
- the 20,053 halted embeddings — untouched, none deleted
- `data/outputs/checkpoints/**` — still empty
- `data/outputs/renders/**`, `pointclouds/**` — **no re-render**
- `workflow/MASTER.md`, `CONTEXT.md`, `INDEX.md`, `DECISION_LEDGER.md`, `decisions/**` — untouched
- `CLAUDE.md` — untouched, **including the stale `/mnt/data1` statement in §9**
- `TASK.md` — Status line **not** edited; still `READY`
- n06, n09, training, gallery indexing — **never invoked**

---

## 3. What the user has NOT decided

Recorded so no later reader mistakes silence for approval.

| # | Open question | Whose call |
|---|---|---|
| 1 | Whether the annotation model changes at all | **User** — U-2 explicitly deferred it |
| 2 | Whether `TASK.md` §7's "Do not change the annotation model" is suspended or stands | Master → User |
| 3 | Whether to spend one authenticated API call to resolve the F-2 conflict about `gpt-4o` | **User** |
| 4 | Whether `Qwen3.8-27B` is adopted. **Downloading it is not adopting it** | **User** |
| 5 | If a model change happens: how R-E's DEVIATION wording is rewritten | **User** |
| 6 | Whether `CLAUDE.md` §9 / `CONTEXT.md` §9 are corrected re `/mnt/data1` | Master → User |
| 7 | Whether `/mnt/data1` becomes a sanctioned location for project artifacts | Master → User |
| 8 | Whether D14 resumes Phase 1 now, or holds | **User** |
| 9 | Whether the HF token is rotated | **User** |

**Design Decisions 1–4 in `n05_v5_design.md` were not re-litigated and are not in question here.**
F-2 concerns a *premise stated alongside* those decisions ("GPT-4o is unavailable"), not the
decisions themselves.

---

## 4. D14's position

D14 is **holding at `READY`** under `WORKFLOW.md` §18.3 (stop-safe).

Rationale, stated as a proposal and not as an accomplished decision:

```
FINDING:   The annotation model is an input to build_prompt, the validator, PROMPT_VERSION and
           the contract fingerprint. The user has re-opened the model question (U-2) while
           TASK.md §7 still forbids changing it.
                                        [OBSERVED IMPLEMENTATION + user directive] [CONFIRMED]

DECISION:  D14 should not implement Phase 1 until the model question is settled, because building
           v5 against one model and then switching would invalidate Phase 1 and require re-running
           Phase 2.
           [proposed by: D14 executor] [requires user approval: YES]
```

**This is D14's recommendation. It has not been approved by anyone.**

If Master or the user directs D14 to proceed on Qwen regardless, D14 will do so and record the
model choice as an open `IMPLEMENTATION CHOICE` carried into Phase 2.

---

## 5. Honesty statements required by `TASK.md` §16

- **No re-render was performed.** The +0.054 correlation finding stands unchallenged.
- **n06, n09, training, and gallery indexing were never invoked.**
- **The 3 legacy-v1 residuals are untouched. `D0-003` remains UNRESOLVED.** Nothing in this log or
  in `MASTER_IMPACTING_FINDING.md` implies otherwise.
- **The 20,053 halted embeddings are untouched; none were deleted.**
- **LVIS anchoring remains a recorded DEVIATION** (`n05_v5_design.md` Decision 1, `TASK.md` R-E).
  It has not been implemented, and nothing here softens that classification.
- **Codex review has not been performed.** That is neither a PASS nor a failure — Phase 1 has
  not produced anything to review.
- **No claim of paper fidelity is made anywhere in this task's current state.**

---

**D14 awaits Master triage and the user's decisions on §3.**

---
---

# ADDENDUM — 2026-08-21T16:5x+0800

D14 put its open questions to the user in one batch. The user answered three of four and
returned a question on the fourth. Recorded below in the user's own words.

**This addendum resolves §3 items 1, 4, 7 and 9. Items 2, 3, 5, 6 and 8 change status — see §7.**

---

## U-6 — the annotation model: **local Qwen3.8-27B**

> 「走本地 Qwen3.8-27B」

**Classification: MATERIAL. USER-DECIDED.** This closes the question U-2 opened.

**What it resolves:**

| §3 item | Was | Now |
|---|---|---|
| 1. Whether the annotation model changes | open | **Yes — it changes** |
| 4. Whether Qwen3.8-27B is adopted | open | **Adopted** |

**What it does NOT resolve — Master must still act:**

`TASK.md` §7 *Explicit Non-Scope* still reads:

> ❌ **Do not change the annotation model.** GPT-4o is unavailable; deviation `D-2` stands.

The user has now directed a model change. **The contract line and the user's directive contradict
each other, and the user's instruction governs (`WORKFLOW.md` §13B).** Master must amend §7.
**D14 has not edited `TASK.md`.**

**Consequence for deviation `D-2`.** `D-2` is currently recorded at `annotate_run.py:71` as

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # D-2: stands in for GPT-4o
```

Under U-6 the substitute becomes `Qwen/Qwen3.8-27B`. **`D-2` is not discharged — it is re-pointed.**
The paper specifies GPT-4o (`2methdology.tex:28`, `neurips_2025.tex:100`); Qwen3.8-27B is not GPT-4o.
**This remains a DEVIATION and must never be described as paper-faithful.**

**What the user was told before deciding, and which remains true:**

- `MASTER_IMPACTING_FINDING.md` **F-2** stays **UNRESOLVED**. The user chose the local route without
  resolving whether `gpt-4o` is reachable. That conflict is **not** closed by U-6 — it is
  **declined**, which is a different thing. The 2026-10-23 shutdown of `gpt-4o-2024-05-13`
  proceeds regardless. **Recorded so nobody later reads U-6 as evidence that GPT-4o was unavailable.**
- Qwen3.8-27B at bf16 is **55.6 GB** and **does not fit** the host's 32,607 MiB of VRAM.
  **Running it requires quantization, which has not been done and not been tested.**
  `[OBSERVED DATA` for the size; `INFERENCE` for the VRAM requirement`]`
- The 20–45 h runtime figure is an **INFERENCE**. No 27B model has been benchmarked on this host
  with 11-image inputs. **It could be materially worse.**

---

## U-7 — `/mnt/data1` is a new disk; project files will be migrated there

> 「這個路徑是我新裝的硬碟我後續要將檔案也搬過去 你可以跟Master報備」

**Classification:** infrastructure / data-location decision. **USER-DECIDED**, reported to Master
as instructed.

**What it resolves:**

| §3 item | Was | Now |
|---|---|---|
| 7. Whether `/mnt/data1` is sanctioned for project artifacts | open | **Yes — and a migration is intended** |

**What Master must act on:**

1. **`CLAUDE.md` §9 is now factually wrong** and the user has confirmed the disk is intentional:
   > Paths under `/mnt/data1` belong to the previous machine and must not be assumed valid.

   `CONTEXT.md` §9 carries the same claim. **D14 cannot edit either file.** See `MASTER_IMPACTING_FINDING.md` **F-1**.

2. **A migration is planned but not scheduled, scoped, or authorised as a task.** D14 flags the
   research risk rather than acting on it:

   > **A data migration changes `data_root`. Under `.claude/rules/experiments.md` §5, moving
   > experiment inputs between comparable runs makes the result a different condition.**
   > `metafind/paths.py` centralises this (`DATA`, `MODELS`, `HF_CACHE`, `PROCTHOR`), so a move is
   > tractable — but the annotation corpus, the 20,053 halted embeddings, the renders and the
   > pointclouds all live under `/home/kyzen/data/MetaFind` today.

   **D14 recommends the migration NOT overlap D14's Phase 3 run.** Moving 413 GB while a 20-45 h
   annotation job reads from it is an avoidable failure mode.
   `[proposed by: D14 executor] [requires user approval: YES]`

3. **D14 has not moved anything and will not.** The only thing D14 placed on `/mnt/data1` is the
   model download, at the user's instruction (U-3).

---

## U-8 — the HF token is not rotated

> 「不用換」

**Classification:** accepted risk, user-decided. §3 item 9 closed.

The token remains installed at the three 600-permission paths listed under U-4. The plaintext value
remains in this session's transcript. **The user was informed and declined rotation.**

---

## U-9 — OPEN: the user asked D14 what it actually intends to do

> 「你先說到底要幹嗎?」

In response to D14's question about whether Phase 1 should begin now.

**Status: the user has NOT authorised Phase 1 to start.** D14 owes a concrete work plan first.
**§3 item 8 (does D14 resume Phase 1) remains OPEN.**

`TASK.md` Status remains `READY`.

---

## 7. Revised open-question table

| # | Question | Status | Whose call |
|---|---|---|---|
| 1 | Whether the annotation model changes | **CLOSED — yes** (U-6) | — |
| 2 | Whether `TASK.md` §7's model prohibition is amended | **OPEN — now urgent.** U-6 contradicts it | Master → User |
| 3 | Whether to spend one API call resolving F-2 (`gpt-4o` reachable?) | **DECLINED, not resolved.** F-2 stays open | User |
| 4 | Whether Qwen3.8-27B is adopted | **CLOSED — yes** (U-6) | — |
| 5 | How `D-2`'s deviation wording is rewritten for the new substitute | **OPEN** | Master → User |
| 6 | Whether `CLAUDE.md` §9 / `CONTEXT.md` §9 are corrected | **OPEN — user confirmed the disk is real** (U-7) | Master → User |
| 7 | Whether `/mnt/data1` is sanctioned for project artifacts | **CLOSED — yes, migration intended** (U-7) | — |
| 8 | Whether D14 resumes Phase 1 | **OPEN** — user asked for a plan first (U-9) | User |
| 9 | Whether the HF token is rotated | **CLOSED — no** (U-8) | — |
| 10 | **NEW.** Whether Qwen3.8-27B can actually run on 32 GB VRAM, and at what speed | **UNKNOWN — untested** | evidence needed, not a decision |
| 11 | **NEW.** When the `/mnt/data1` migration happens relative to D14 Phase 3 | **OPEN** | Master → User |

---

## 8. Standing honesty statements — unchanged by this addendum

- **No re-render.** No n06, n09, training, or gallery indexing.
- **The 3 legacy-v1 residuals are untouched. `D0-003` remains UNRESOLVED.**
- **The 20,053 halted embeddings are untouched; none deleted.**
- **LVIS anchoring remains a recorded DEVIATION.** So does the annotation model, now re-pointed
  from Qwen2.5-VL-7B to Qwen3.8-27B. **Neither is paper-faithful.**
- **Codex review has not been performed.** Not a PASS — there is nothing yet to review.
- **`TASK.md` is still `READY`. Phase 1 has not begun.**
- **No claim of paper fidelity is made anywhere in this task's current state.**

---
---

# ADDENDUM 2 — 2026-08-21, Phase 1 executed

## U-10 — the user authorised execution

> 「沒關係 我允許你 現在將標註這流程完全處理修好 讓05能夠產出正確的標註 你後續再跟Master說就好」

`TASK.md` Status set `READY` -> `ACTIVE`. §3 item 8 closed.

**The HOLD GATE is NOT closed by this.** "修好標註流程" authorises implementation and
sample validation. It is not the explicit go for a 19.6-GPU-hour, whole-corpus,
irreversible re-annotation, and `TASK.md` R-A makes that gate absolute. Phase 2 must
produce its numbers first, because informing exactly that decision is what Phase 2 is for.
**D14 will stop after Phase 2 and ask.**

---

## Completed — model-independent verification (`TASK.md` §10.2, §10.3, DoD 2, DoD 3)

### V-1 LVIS coverage, verified per uid, not accepted on report

```
annotation files                 45,955   (45,952 pv3 + 3 pv1 residuals)
v3 uids WITH an LVIS label       45,952 / 45,952 = 100.00%
v3 uids WITHOUT                       0
distinct LVIS categories used     1,156
LVIS top-20 share                  7.1%   (Master reported 7.1% -- reproduced)
```
`[OBSERVED DATA]` `CONFIRMED`

### V-2 Y-up axis — independently reproduced, not inherited

Master's evidence used category sets D14 never saw. D14 chose its own: LVIS categories
whose real-world aspect ratio is unambiguous.

```
TALL  n=1,365   mean normalised [x .543, y .946, z .474]   y-longest 83.6%
FLAT  n=  962   mean normalised [x .882, y .372, z .716]   y-longest 11.5%
separation:  y +0.574   x -0.339   z -0.243
```
Master reported `[.515 .960 .402]` / `[.865 .318 .738]`. Different method, different
sample, same conclusion. **`height` = y.** `[OBSERVED DATA]` `CONFIRMED`

### V-3 synset table — built from the authoritative source, escalation NOT required

`TASK.md` §7 said to escalate if the table needed a research decision. **It did not.**
LVIS itself publishes a synset per category, so this is a copy, not a judgement.

```
primary   https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip   1,203 categories
cross-chk detectron2 lvis_v1_categories.py                           1,203, 0 disagreements
coverage  1,156 / 1,156 = 100.00%
rules     1,155 match an LVIS category name; 1 ('horned cow') via LVIS's own synonym table
invented  0        malformed 0        not-from-LVIS 0
```
Written to `metafind/data/lvis_synsets.json` with provenance and the primary sha256.
`[UPSTREAM FACT]` `CONFIRMED`

---

## Completed — Phase 1, v5 implemented

```
PROMPT_VERSION     4 -> 5
VALIDATOR_VERSION  2 -> 3
SCHEMA_VERSION     2 -> 3
contract id        metafind_annot_v5@f5b2bfb2e5f61fe7      (fingerprint moved)
tests              tests/test_annotate.py 113 -> 148;  full suite 582 passed
```

What changed, and the failure each change removes:

| Change | What it prevents |
|---|---|
| `build_prompt` receives the LVIS category | the model never has to guess an identity, so it cannot collapse onto `toy` |
| `build_prompt` receives the exact mesh proportions | three guesses about a scale-normalised render become one |
| `identity_confirmed` required, **recorded, never filtered** | makes LVIS's own error rate measurable (R-B) |
| `synset` looked up, not read from the response | v4's own comment admitted "a well-formed but invented synset passes here" |
| `width`/`length` derived from the mesh | the model can no longer contradict the shape the renders show |
| `lvis_category` + `category_relation` stored on every record | refinement vs replacement stays distinguishable after the fact |
| `MODEL_ID` -> `/mnt/data1/kyzen/models/Qwen3.8-27B` | user decision U-6. **D-2 re-pointed, NOT discharged** |
| model class -> `AutoModelForImageTextToText` | D-2 has now named two families; a hardcoded class makes swapping the annotator a loader edit |

---

## Two IMPLEMENTATION CHOICES that need review

### IC-1 `divergent` categories are recorded, not rejected

`TASK.md` §7 says lateral replacement is invalid. **It cannot be enforced mechanically.**

```
"motor vehicle" -> "pickup truck"    a genuine downward refinement   } indistinguishable
"motor vehicle" -> "coffee machine"  a lateral replacement           } without a hypernym
                                                                       judgement over free
                                                                       text that no
                                                                       authoritative source
                                                                       in this project supplies
```

A token-containment rule catches `toy -> toy dinosaur` but rejects `motor vehicle ->
pickup truck`, which the design explicitly calls **required**.

**Chosen:** compute `category_relation` deterministically (`exact` / `refined` /
`divergent`), store it, reject nothing. Same reasoning as Design Decision 2: measure the
rate, then decide whether the rule needs teeth. `TASK.md` §12 already requires those three
counts to be reported separately, so the classifier was required regardless.

```
FINDING:   The refinement-not-replacement rule is not mechanically enforceable
           from any source this project holds.        [OBSERVED IMPLEMENTATION] [CONFIRMED]
DECISION:  Record the relation and measure the divergent rate in Phase 2 rather
           than rejecting on it.
           [proposed by: D14] [requires user approval: YES]
```

This is exactly `TASK.md` §14's second Codex attack point. **Codex must be pointed at it.**

### IC-2 `synset` follows the LVIS anchor, not the model's refined category

`toy -> toy dinosaur` is the better retrieval string, but no authoritative synset exists
for "toy dinosaur", and minting one re-creates the invented-synset error class the lookup
was adopted to remove. So `category` may be refined while `synset` stays LVIS's.
`[IMPLEMENTATION CHOICE]` `requires user approval: YES`

---

## LOCAL TASK ISSUE — a Definition-of-Done item collides with a protected path

`DoD 13` requires `tools/check_graph.py` to pass. It now reports **1 failure**:

```
README unit-test count: README says 435, tests/ defines 456
```

The counter lives in **`docs/graph/README.md`**, and `TASK.md` §9.2 protects `docs/**`
outright. Verified this failure did **not** pre-exist: restoring the v4 test file returns
`2275 checks, all pass`. **D14 caused it by adding the tests DoD 13 also requires.**

**D14 did not edit it.** One stale integer (and a second, "547 個 case", now 582) needs a
one-line correction from whoever owns `docs/`. `check_graph.py` is otherwise fully green.

---

## BLOCKED — Step 0 cannot run yet

The Qwen3.8-27B download is at **9 / 18 shards (35 GB of 55.6 GB)**; the user is running it.
Until it completes, D14 cannot measure:

- whether the model loads at all on 32,607 MiB (**it will not at bf16 -- quantization is
  required and has not been done**)
- VRAM peak with 11 images
- seconds per asset, hence whether Phase 3 is 20 h or 100 h
- whether the quantized model emits parseable JSON

**No estimate of Phase 3's runtime is currently evidence-backed.** The 20-45 h figure
remains an `INFERENCE`.

---

## State verified after Phase 1

```
annotations                45,955          unchanged, not read for mutation
corpus md5 (baseline)      30b2737c95152043762ce25fcabe7a0e
3 legacy-v1 residuals      prompt_version 1, untouched   D0-003 STILL UNRESOLVED
embeddings                 20,053 npz      none deleted
checkpoints                0
docs/ , README.md          untouched
Master's workflow files    untouched
git diff (D14 only)        metafind/data/annotate.py, annotate_run.py,
                           tests/test_annotate.py, metafind/data/lvis_synsets.json (new),
                           workflow/tasks/D14_n05-v5-reannotate/ (new)
```

**No re-render. n06 / n09 / training / gallery indexing never invoked. LVIS anchoring and
the annotation model are both recorded DEVIATIONS. No claim of paper fidelity is made.**
