# Skill Integration Policy

> How the engineering skills are used inside this workflow: which one, at which layer, by whom,
> and when it is worth it. Master owns this file.
>
> Structure and rules: `workflow/BLOCKS.md` · Project state: `workflow/MASTER.md`

---

## 0. Standing — skills are method tools, not authority

A skill can change **how** we work. It can never change **what is true** or **who decides**.

No skill output overrides:

- USER-approved decisions (`workflow/DECISION_LEDGER.md`)
- MetaFind paper / source / supplementary material
- upstream authoritative sources and official implementations
- Block boundaries
- the separation of **FINDING** from **DECISION**
- the USER's final authority

```
Skill PASS  ≠  scientific PASS
Tests PASS  ≠  reproduction fidelity
Codex PASS  ≠  Block PASS
Reviewer PASS ≠ USER acceptance
```

**The point of this document is to say which skill is used at which layer, by whom, and when it
is worth it.** Work is owned by whole blocks, not by a swarm of tiny tickets. Do not rebuild a
swarm of tiny tickets out of skills. §5 is the list of things that need none of this.

---

## 1. Invocation reality — verified on this machine 2026-08-22

`mattpocock-skills@claude-plugins-official` v1.2.3,
`~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/1.2.3/`.

**Six of the requested skills carry `disable-model-invocation: true` in their frontmatter.**
Claude **cannot** call them; only the USER can, by typing the slash command. This is a hard
constraint on the workflow, not a preference.

| Skill | Claude may invoke | How we use it |
|---|---|---|
| `mattpocock-skills:research` | **yes** | as-is |
| `mattpocock-skills:tdd` | **yes** | as-is, plus the Expected-Truth rule (§7) |
| `mattpocock-skills:diagnosing-bugs` | **yes** | as-is |
| `mattpocock-skills:code-review` | **yes** | **extended 2 axes → 4** (§9) |
| `mattpocock-skills:domain-modeling` | **yes** | as-is |
| `mattpocock-skills:codebase-design` | **yes** | as-is |
| `mattpocock-skills:grilling` | **yes** | as-is, and it is the engine of the Acceptance Grill (§11) |
| `grill-with-docs` | **no — USER types `/grill-with-docs`** | its SKILL.md is literally *"Call the Skill tool twice, for grilling and domain-modeling"*. **Claude reproduces it directly by calling those two.** No capability is lost |
| `implement` | **no — USER types `/implement`** | thin wrapper: tdd at seams → typecheck → full suite → code-review → commit. **Claude reproduces the sequence directly**, minus the commit rule (§8) |
| `improve-codebase-architecture` | **no — USER types `/improve-codebase-architecture`** | **genuinely unavailable to Claude.** Reviewer must ask the USER to run it at a milestone (§10) |
| `to-spec` | **no — USER types `/to-spec`** | **and it publishes to an issue tracker we do not have.** We take its method, not the skill (§6) |
| `handoff` | **no — USER types `/handoff`** | **not used.** It writes to the OS temp directory, explicitly outside the workspace, which contradicts our persistence rule. We use `session-handoff` + block `HANDOFF.md` (§12) |
| `grill-me` | **no — USER types `/grill-me`** | not needed; `grilling` covers it |

**Consequence for Block Owners and Reviewers:** when a step calls for `grill-with-docs` or
`implement`, do not call the Skill tool and do not report a failure — perform the composed
sequence. When a step calls for `improve-codebase-architecture`, **ask the USER to run it.**

---

## 2. Mandatory

Five gates. Each applies to a **material** change or a **Block milestone**, never to an internal
work item.

| # | Gate | Skill | Who | When |
|---|---|---|---|---|
| M1 | **Block start grill** | `grilling` + `domain-modeling` (= `grill-with-docs`) | Master, with Owner and Reviewer present | before a new Block or a major Block milestone begins |
| M2 | **SPEC** | our template (§6) | Owner, from the grill output | after M1, before any implementation |
| M3 | **4-axis completion review** | `code-review`, extended | Owner, at the completion claim | whenever a material implementation is claimed done |
| M4 | **Source verification** | `research` | whoever makes the claim | before any paper / upstream / dataset-semantics claim becomes project state |
| M5 | **USER Acceptance Grill** | `grilling` mode, one item at a time | Master → USER | at every Block milestone |

`tdd` is mandatory **only** where a behavioural seam exists *and* the change is research-critical.
Where no seam exists, say so in the SPEC rather than inventing one.

## 3. Conditional

Run when the trigger fires, not on a schedule.

| Skill | Trigger |
|---|---|
| `diagnosing-bugs` | runtime bug · semantic contradiction · unexpected output · performance regression · suspected data corruption · **source vs generated-artifact mismatch** · **two measurements that conflict** · suspected silent pipeline failure |
| `codebase-design` | designing or reshaping a module interface; deciding where a seam goes |
| `domain-modeling` | terminology has drifted, or a new domain term is being coined |
| `improve-codebase-architecture` | Reviewer, at a **stable milestone only**, or when block code has visibly become hard to navigate. **USER-invoked** |
| `research` beyond M4 | any time a primary source would settle an argument faster than debate |

## 4. Not used

- **Matt's `handoff`** — writes outside the workspace. Use `session-handoff` and block `HANDOFF.md`.
- **`to-spec` as published** — requires an issue tracker (`docs/agents/issue-tracker.md`,
  `/setup-matt-pocock-skills`). We have no tracker; specs live in `workflow/`.
- **`implement`'s commit rule** — *"Commit your work to the current branch"* does not bind here (§8).
- **`prototype`, `wizard`, `resolving-merge-conflicts`, `writing-for-agents`** — outside this integration.

---

## 5. When **not** to run any of this

Heavyweight process is for material change. It is not for:

- an internal work item **inside an already-approved SPEC** that changes no scientific behaviour
- comments, docstrings, log formatting, file moves, typo fixes
- test scaffolding that introduces **no new expected-truth claim**
- read-only investigation that changes nothing on disk
- re-running an already-accepted deterministic step with unchanged inputs
- anything the Owner can verify, correct, and describe in one sentence

**The Block Owner manages internal work items alone.** If a task would need a grill, a spec, a TDD
cycle, a reviewer, Codex and a USER gate to change one function, the classification is wrong —
it is an internal work item, so treat it as one.

Escalate to heavyweight only for: a material change · a high-risk pipeline stage · **before an
expensive run** (full annotation, corpus generation, n06 full encode, multi-hour GPU job, full
training, full evaluation) · a major internal milestone · a Block milestone.

---

## 6. Block start → SPEC

```
MASTER
  ↓
BLOCK OWNER + BLOCK REVIEWER
  ↓
grilling + domain-modeling          ← M1
  ↓
Block Plan
  ↓
SPEC                                ← M2
  ↓
USER approves scope
  ↓
execution begins
```

### The grill's own rule, which is also ours

`grilling` states: *"Finding facts is your job, never the user's. When a frontier question needs a
fact from the environment, dispatch a sub-agent to find it; don't ask the user for anything you
could look up yourself."*

**That is binding.** Anything answerable from the repo, the paper, the upstream source, the
runtime, or the data on disk is **Master's job to look up before asking.** Only genuine decisions
reach the USER. It also works in rounds — ask the whole settled frontier at once, then wait — so
the grill does not degenerate into twenty separate messages.

### SPEC — 15 required sections, USER-specified

Written to `workflow/blocks/<BLOCK>/SPEC_<milestone>.md`, from
`workflow/blocks/SPEC_TEMPLATE.md`.

```
OBJECTIVE · SOURCE OF TRUTH · INPUTS · OUTPUTS · SCOPE · NON-SCOPE ·
PAPER / UPSTREAM AUTHORITY · IMPLEMENTATION CHOICES · KNOWN DEVIATIONS · UNKNOWN ·
SUCCESS CRITERIA · FAILURE CONDITIONS · SELF-VERIFICATION REQUIREMENTS ·
INDEPENDENT REVIEW REQUIREMENTS · MILESTONE CRITERIA
```

**Adopted from `to-spec`, because it is genuinely useful:** sketch the **test seams** before
writing the spec. Prefer existing seams; use the highest seam available; the fewer seams, the
better — one is ideal. Confirm the seams with the USER. This lands in
`SELF-VERIFICATION REQUIREMENTS`.

**Not adopted:** the User-Stories section, and publication to an issue tracker. A
paper-reproduction pipeline has no user actors, and "As a user I want a point cloud" is noise.

**The SPEC is not bureaucracy.** It is the single reference that the 4-axis review, the Reviewer,
Codex, and the USER acceptance all read from. Without it, Axis 2 has nothing to check against.

---

## 7. Owner workflow

```
SPEC
  ↓
implement  ─── tdd at pre-agreed seams
  │            research when a source question blocks progress
  │            diagnosing-bugs when something contradicts
  ↓
self-verification
  ↓
4-axis code-review                  ← M3
  ↓
completion claim → HANDOFF.md
```

The Owner holds **the whole block chain**, not one node. A B1 Owner who understands n06 but not
where its input comes from is not doing the job.

### Self-verification the Owner owns regardless of any reviewer

implementation correctness · unit and integration tests · runtime verification · artifact
integrity · provenance · dataset consistency · upstream/downstream consistency · semantic
sanity · paper consistency · failure cases · resume and cache correctness · silent failure.

**Having a Reviewer never excuses skipping this.**

### The Expected-Truth Provenance Rule — our addition to `tdd`

`tdd`'s own rules bind: red before green, test public behaviour, never private implementation
detail, no tautological tests, no implementation-coupled tests.

On top of them, for this project:

> **Every test whose expected value encodes a claim about the world must name where that expected
> value came from — and it must not be the implementation under test.**

Applies to every test touching: dataset · annotation · geometry · paper formula · protocol ·
evaluation · units · coordinate frames.

**Why.** A validator that accepts any English string, plus a test that only feeds it English
strings, produces a green suite over a corpus where `chocolate cake → mosaic`. That is not
hypothetical — it is what 582 passing tests did here. **The code and the test shared one wrong
assumption**, so the test could not fail.

For each such test, answer in one line in the test or the SPEC: *where does the expected truth
come from?* Acceptable answers: the paper's equation · the official upstream implementation ·
the source dataset's own metadata · an independent measurement. Unacceptable: "what the function
currently returns".

---

## 8. Commit rule — we override `implement`

`implement` ends *"Commit your work to the current branch."* **That is not our rule.**

- Unaccepted scientific work **may** be checkpoint-committed — losing it is worse.
- Such a commit **must** be marked `WIP` / `UNACCEPTED` in its message.
- A commit is **never** acceptance. Only the USER's `APPROVE` makes anything FINAL.
- Never commit as a way of closing a review.

---

## 9. Four-axis code review

Matt's `code-review` ships **two** axes as parallel sub-agents. We run **four**. Axes 3 and 4
have no upstream equivalent and are ours.

**Report the four separately. Never merge them into one PASS.** A change can pass three and be
scientifically worthless.

### AXIS 1 — Standards
Upstream axis, unchanged. Repo standards plus the Fowler smell baseline the skill carries
(mysterious name, duplicated code, feature envy, data clumps, primitive obsession, repeated
switches, shotgun surgery, divergent change, speculative generality, message chains, middle man,
refused bequest). A documented repo standard overrides the baseline; smells are judgement calls.

### AXIS 2 — Spec
Upstream axis, with our SPEC as the spec source instead of an issue tracker. Reports: missing or
partial requirements · behaviour nobody asked for (scope creep) · requirements that look
implemented but are implemented wrongly. **Quote the SPEC line for every finding.**

### AXIS 3 — Source / Evidence Fidelity *(ours)*
Does the implementation match its authority?

> paper · upstream authoritative source · source dataset · USER-approved decision · recorded
> protocol · artifact provenance

Sub-agent brief: *"For each research-significant behaviour in the diff, name the authority it
rests on and cite it — paper section, upstream `file:line` at the recorded commit, dataset field,
or ledger entry. Flag every behaviour that has no authority, cites the wrong one, or cites an
authority that does not actually say what is claimed. Flag any INFERENCE or IMPLEMENTATION CHOICE
written as a PAPER FACT. Under 400 words."*

### AXIS 4 — Scientific / Semantic Validity *(ours)*
**Assume the code runs, the tests pass, and Axis 2 passes. How could the result still be
scientifically wrong?**

Actively hunt: semantic contradiction · silent corruption · generated artifact vs source
mismatch · wrong units · coordinate or frame mismatch · label noise · invalid assumptions ·
downstream contamination · evaluation leakage.

Sub-agent brief: *"Take as given that this code runs and its tests pass. Your job is to find how
the output could still be scientifically invalid. Prefer a concrete failing case — inputs, and
the wrong output or wrong conclusion they produce — over a general worry. Check units, coordinate
frames, whether generated artifacts still agree with their source, whether any error path fails
silently, and whether anything here could contaminate a downstream stage or leak into evaluation.
Under 400 words."*

### Output

```
## STANDARDS            n findings, worst:
## SPEC                 n findings, worst:
## SOURCE / EVIDENCE    n findings, worst:
## SCIENTIFIC / SEMANTIC n findings, worst:
```

No single winner across axes. Ranking them is exactly what the separation prevents.

---

## 10. Reviewer workflow

**The Block Reviewer is a separate Claude context, not Codex and not a second Owner.**

```
research                        source-of-truth and contract audit
diagnosing-bugs                 differential tests, conflicting measurements
code-review (4-axis)            independent pass, not the Owner's
improve-codebase-architecture   milestone only — USER-invoked (§1)
```

### Synchronous, not terminal

The Reviewer does **not** wait for the Owner to finish. Before any of these, the Reviewer must
already have audited the sources, the contract, a real sample, and semantic consistency:

full annotation · large corpus processing · n06 full encode · any multi-hour GPU run · full
training · full evaluation.

**A review that begins after the run is a post-mortem, not a review.**

### The Reviewer's standing question

> **"If every one of the Owner's tests passes, how could this still be wrong?"**

And the ten attacks: is the Owner's own contract wrong? · was an upstream source-of-truth missed? ·
do the generated artifacts actually match the source data? · could a schema PASS still be
semantically wrong? · could all tests PASS and the science still be wrong? · is there silent
corruption? · are the block's work items consistent with each other? · could this output
contaminate downstream? · did the Owner state an INFERENCE as a FACT? · which failure modes do the
Owner's tests not cover?

### Differential testing — the Reviewer's sharpest tool

From `diagnosing-bugs`: build a red-capable feedback loop, reproduce, minimise, form several
**falsifiable** hypotheses, instrument, fix, regression-test, then re-run the original failing
loop. **Do not read the code and guess the cause first.**

Compare two things that should agree:

```
official ULIP-2 artifact   vs   our generated artifact
source dataset metadata    vs   our generated annotation
before                     vs   after
configuration A            vs   configuration B
```

This is not theoretical. It is how the 180° yaw was found, and how it was then shown **not** to
move the embedding (`workflow/blocks/ULIP2/evidence/n03_n04_upstream_verification.md`).

### Boundaries

Read-only by default. To execute a check: read-only commands, an isolated output directory, or a
separate git worktree — never the Owner's production files. **The Reviewer may not decide a
material remedy.** Findings go to Master through `HANDOFF.md`.

---

## 11. USER Acceptance Grill

At a Block milestone, Master must **not** say *"everything passed, please approve."*

Master runs `grilling` mode against the USER: **one material criterion at a time.**

### Before asking anything

Master looks up the evidence itself — repo, paper, runtime, data, Reviewer report, Codex report.
**Never ask the USER to look up a fact for the AI.**

### Format

```
[Acceptance i/N]

REQUIREMENT        what was originally required
OWNER CLAIM        what the Owner says was done
EVIDENCE           what Master actually verified, with file:line / measurement / population
SELF VERIFICATION  how the Owner verified it
BLOCK REVIEWER     the Reviewer's verdict
CODEX              material findings, if any
MASTER ASSESSMENT  PASS / FAIL / INVESTIGATE MORE
REMAINING UNKNOWN  what is still not known

YOUR DECISION
  A. ACCEPT THIS ITEM
  B. REJECT
  C. INVESTIGATE MORE
  D. SHOW MORE EVIDENCE
```

**Then wait.** One item per message. Never ten questions at once.

### Rules

1. One material criterion per round.
2. Master looks up everything lookup-able. The USER decides; the USER does not research.
3. **An Owner claim is not evidence.**
4. **A passing test is not a substitute for scientific evidence.**
5. **Reviewer PASS is not USER acceptance.**
6. **Codex PASS is not USER acceptance.**
7. Nothing is marked `FINAL ACCEPTED` until **every** material criterion has been through a round.
8. The USER may at any point demand `SHOW EVIDENCE`, `INVESTIGATE MORE`, or `RETURN TO OWNER`.

---

## 12. Codex

### 12.0 The pre-execution gate — `DL-028`, USER order 2026-08-24

> **No code runs until Codex has reviewed the code that will run.** Every stage. No exceptions.

Ordered by the USER on 2026-08-24 after a 29-file, +3,843-line change reached the corpus ungated
and the n03→n04→n05 chain was rebuilt and restarted repeatedly. The reason given is the defect
rate, not a process preference.

```
about to run anything
        ↓
   CODEX REVIEW      ← state the stage · state what is being written · pass the files
        ↓
   findings verified and classified by the role that asked
        ↓
      RUN
```

Binding form of the request. All three are required:

1. **which stage** is being run (`n03`, `n04`, `n05`, a tool, a re-run — name it);
2. **what is being written** at that stage — the artefacts, the versions, the paths;
3. **the actual files**, passed to Codex, not described to it.

`codex` is installed: `/usr/bin/codex`, `codex-cli 0.148.0`. Non-interactive forms —
`codex exec review --uncommitted` · `--base <branch>` · `--commit <sha>` · `codex exec "<prompt>"`.
No single invocation is mandated; **record the exact command and its result** with the run.

**Scope.** Anything that writes, renders, trains, annotates, or mutates the corpus is execution.
Read-only inspection is not. The `SKILLS.md` §5 / `BLOCKS.md` exemption for internal work items
and *re-runs of accepted deterministic steps* **is revoked for execution** — a re-run of accepted
code still runs code, and `2fa28d4` is the case that proves a "re-run" can be a rebuild.

**`CODEX REVIEW UNAVAILABLE` is now a STOP, not a caveat.** It was already never a PASS. Under a
pre-execution gate it blocks the run: report it to the USER and wait.

### 12.1 Codex is still not authority

Codex stays the **third** layer. It does not replace the Block Reviewer.

```
BLOCK OWNER  ↕  BLOCK REVIEWER
                     ↓
              pre-execution Codex   (mandatory, before ANY run — 12.0)
                     ↓
              targeted Codex        (high-risk question, any time)
                     ↓
              milestone Codex       (mandatory, adversarial, at every Block milestone)
                     ↓
                  MASTER
                     ↓
              USER ACCEPTANCE GRILL
```

**Wrong:** `Owner → Codex` used as the review. Codex is adversarial, not authoritative, and it has
no standing project context. A Codex PASS clears the gate to *run*; it does not clear the Block,
and it is not USER acceptance.

Every Codex finding is independently verified and classified `CONFIRMED` · `PLAUSIBLE` ·
`REJECTED` · `UNVERIFIED`, and it must reach the USER brief.

---

## 13. Handoff

Use `session-handoff` (this project's own skill) and the block `HANDOFF.md`. **Not** Matt's
`handoff`, which writes outside the workspace.

A handoff is **temporary continuity only**. It is never a formal finding, a formal decision, or
an acceptance. Anything that must persist goes back into the block and Master workflow files.

---

## 14. Per-block skill matrix

| Block | Owner | Reviewer |
|---|---|---|
| **ULIP2** | `implement`-sequence · `tdd` · `research` · `diagnosing-bugs` | `research` · `code-review` (4-axis) · `diagnosing-bugs` · `improve-codebase-architecture` (milestone, USER-invoked) |
| **ESSGNN** | `implement`-sequence · `tdd` · `research` · `diagnosing-bugs` | same as ULIP2 |
| **INTEGRATOR** | `research` · `grilling` · `diagnosing-bugs` | — (Master reviews) |

**Evaluation work carries extra weight on Axes 3 and 4.** An evaluation implemented perfectly
against the wrong protocol still produces an invalid comparison with the paper's tables, and every
test will pass while it does. Table 1 lives in ULIP2, Table 2 in ESSGNN — both Owners inherit this.

---

## 15. The formal acceptance flow

```
Block Plan
  → USER approves scope
  → Owner implementation + self-verification
       ↕  Reviewer synchronous independent verification
  → 4-axis completion review
  → Codex milestone adversarial review
  → Master integration
  → USER Acceptance Grill          (one item at a time)
  → USER FINAL ACCEPTED
```

No step may be skipped at a Block milestone. Every step **is** skipped for an internal work item —
see §5.
