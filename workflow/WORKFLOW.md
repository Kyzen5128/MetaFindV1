# MetaFindV1 Multi-Agent Workflow

> Operating protocol for Master, D0, D1/D2/D3..., and Codex.
> This file defines how work moves through the project.
> It does not contain scientific conclusions or task-specific implementation details.
>
> **Superseded in part, 2026-08-21/22.** The project moved to a **Block-centric** structure
> (`workflow/BLOCKS.md`) and formally integrated the engineering skills
> (`workflow/SKILLS.md`). Where this file's D-task role model conflicts with those two, **they
> win** — see §20. What still binds unchanged: §13A Finding vs Decision, §13B User Review Gate,
> §13C USER REVIEW BRIEF, §18 escalation, and the USER's final authority.

---

# 1. Roles

## USER — Final Research / Project Authority

The user is the final decision maker for this project.

Master, D0, D-tasks, and Codex investigate, implement, review, and recommend. **None of them decides.**

The user holds two distinct gates:

1. **Start gate** — nothing formal begins without user approval (Section 5).
2. **Acceptance gate** — no material decision becomes project state without user approval (Section 13B).

The user is the only role that can convert a recommendation into `FINAL ACCEPTED`.

The user is not required to read full investigations. Master must present decisions in a compact, decision-ready form — the USER REVIEW BRIEF (Section 13C).

---

## MASTER — Project Lead / Integration Owner

Master maintains the full project view.

Master is responsible for:

- understanding the complete project pipeline;
- tracking current stage and milestone status;
- knowing what is DONE, ACTIVE, READY, BLOCKED, or awaiting a decision;
- maintaining dependencies and execution order;
- deciding when a research/architecture question must go to D0;
- defining bounded D1/D2/D3... work packages;
- preparing self-contained TASK.md files;
- proposing the next task to the user;
- receiving HANDOFF results;
- validating task completion;
- producing the USER REVIEW BRIEF for every material decision;
- integrating results **after** user approval;
- updating MASTER.md, CONTEXT.md, and INDEX.md;
- requesting milestone-level Codex review;
- preserving the global architecture and research direction.

Master should not consume its context on long bounded execution work.

Master normally does not:

- perform long implementation tasks;
- run long experiments;
- perform prolonged debugging;
- deeply investigate one research question;
- process large datasets;
- silently execute the next task.

Master proposes.
The user approves.
A dedicated task conversation executes.

Master then reviews and **recommends**.
The user makes the final decision.

Master's `ACCEPT` is a recommendation, not an acceptance. See Section 13B.

---

## D0 — Research / Architecture Lead

D0 handles questions that require evidence-backed investigation before Master can safely decide.

Examples:

- paper interpretation;
- conflicting formulas or sections;
- architecture decisions;
- source/provenance questions;
- research-significant configuration choices;
- cross-stage assumptions;
- decisions affecting several later tasks.

D0 is not a normal implementation worker.

D0 should:

1. define the exact decision question;
2. inspect primary evidence;
3. inspect relevant supporting/conflicting evidence;
4. inspect current repository behavior;
5. identify viable options;
6. analyze consequences;
7. make a recommendation;
8. request Codex adversarial review;
9. verify Codex findings;
10. return a recommendation to Master.

D0 does not mark its own recommendation ACCEPTED.

Master reviews and recommends.

**The user decides acceptance.** A D0 recommendation that Master endorses is still only a recommendation until the user approves it through the gate in Section 13B.

Formal accepted decisions live under:

`workflow/decisions/`

---

## D1 / D2 / D3... — Stage / Work-Package Owners

Each D-task is a bounded execution owner.

Default granularity:

> One independently verifiable stage or coherent work package.

Do not create tasks so small that Master spends more effort coordinating than the task requires.

Do not create tasks so large that the task becomes another Master.

A D-task should understand:

- its objective;
- required project context;
- inputs and authority;
- dependencies;
- scope;
- explicit non-scope;
- expected deliverables;
- verification requirements;
- Definition of Done.

A D-task does not manage the overall project.

It does not automatically start the next task.

---

## CODEX — Independent Reviewer

Codex provides independent review.

Codex is not:

- scientific authority;
- project manager;
- final decision maker.

Codex reviews:

- implementation correctness;
- assumptions;
- regressions;
- missing verification;
- design choices;
- source/implementation mismatch;
- hidden failure modes.

Claude must independently verify material Codex findings.

---

# 2. Persistent Project Memory

Important workflow state must live in files, not only conversation context.

## Global state

`workflow/MASTER.md`

Contains:

- overall project status;
- current phase;
- dependencies;
- active assignment;
- blockers;
- next recommended action.

## Shared compressed context

`workflow/CONTEXT.md`

Contains:

- project goal;
- architecture;
- stable accepted decisions;
- shared constraints;
- cross-task dependencies;
- stable runtime facts.

This is the main mechanism used to avoid repeatedly loading the full Master conversation.

## Task registry

`workflow/INDEX.md`

Contains:

- formal tasks;
- task status;
- dependencies;
- active/blocked/completed state.

## Research decisions

`workflow/decisions/`

Contains formal D0 decisions.

## Execution workspaces

`workflow/tasks/<task-id>/`

Contains:

- `TASK.md`
- `HANDOFF.md`
- `CODEX_REVIEW.md`

---

# 3. Conversation Strategy

## Default: NEW CONVERSATION

Formal D0 and D1/D2/D3 execution should normally use a new Claude conversation.

Reason:

- reduce unnecessary inherited tokens;
- keep task context focused;
- prevent old debugging/discussion history from dominating execution;
- force Master to create a clear, self-contained task contract.

A new D-task normally reads:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. applicable `.claude/rules/`
3. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
4. its assigned `TASK.md` or D0 decision file
5. only the explicitly referenced source / code / evidence files

It should not automatically read the entire repository.

---

## Exception: FORK REQUIRED

Fork from Master only when a task critically depends on conversation-only reasoning that cannot reasonably be captured in:

- CONTEXT.md;
- TASK.md;
- decision documents;
- repository evidence.

This should be exceptional.

Master must explicitly mark:

`Conversation Mode: FORK REQUIRED`

and explain why.

Otherwise:

`Conversation Mode: NEW CONVERSATION`

---


## Session Continuity

Conversation continuity is temporary working memory and must remain separate
from formal project state and formal task completion.

### Master session continuity

If the Master conversation is unfinished but must move to a fresh conversation,
use:

`/session-handoff`

Target:

`workflow/MASTER_SESSION_HANDOFF.md`

A fresh Master conversation should read this file only when resuming that
unfinished Master session.

It does not replace:

- `workflow/MASTER.md`
- `workflow/CONTEXT.md`
- `workflow/INDEX.md`

After the continuity state is no longer needed, the file may be replaced by a
later session handoff.

It is local working memory and should not be committed.

### D0 decision session continuity

If a formal D0 research / architecture decision is unfinished but must continue
in a fresh conversation, use:

`/session-handoff`

Target:

`workflow/decisions/<decision-id>_SESSION_HANDOFF.md`

The resumed D0 conversation should read:

1. `CLAUDE.md`
2. applicable `.claude/rules/`
3. `workflow/WORKFLOW.md`
4. `workflow/MASTER.md`
5. `workflow/CONTEXT.md`
6. `workflow/INDEX.md`
7. its formal `workflow/decisions/<decision-id>.md`
8. its `<decision-id>_SESSION_HANDOFF.md`
9. only the additional evidence required by the decision

The session handoff records temporary investigation continuity only.

It must not replace or redefine the formal decision file.

When D0 finishes its investigation, the formal result must be written into:

`workflow/decisions/<decision-id>.md`

including:

- evidence;
- analysis;
- recommendation;
- Codex adversarial review;
- Claude verification of Codex findings;
- final recommendation to Master.

D0 does not mark its own recommendation as accepted.

Master reviews the formal decision and owns final resolution.

The D0 session handoff is local working memory and should not be committed.

### D-task session continuity

If a formal D-task is unfinished but must continue in a fresh conversation,
use:

`/session-handoff`

Target:

`workflow/tasks/<task-id>/SESSION_HANDOFF.md`

The resumed D-task conversation should read:

1. `CLAUDE.md`
2. applicable `.claude/rules/`
3. `workflow/CONTEXT.md`
4. its `TASK.md`
5. its `SESSION_HANDOFF.md`
6. only the additional evidence/files required by TASK.md

`SESSION_HANDOFF.md` means:

> where this unfinished task currently stands

It is not a formal completion artifact.

It is local working memory and should not be committed.

### Formal task completion

When a D-task is formally finished or reaches a formal blocked return point,
do not use `SESSION_HANDOFF.md` as the result.

Formal return to Master requires:

- `workflow/tasks/<task-id>/CODEX_REVIEW.md` when applicable
- `workflow/tasks/<task-id>/HANDOFF.md`

`HANDOFF.md` means:

> the formal result submitted to Master for integration review

Therefore:

unfinished conversation continuity
→ `SESSION_HANDOFF.md`

formal completed/blocked task return
→ `HANDOFF.md`

These meanings must never be merged.

---

# 4. Task Creation

Master creates a task only after determining:

- objective;
- dependency;
- scope;
- non-scope;
- authority;
- expected deliverables;
- verification;
- Definition of Done;
- Codex review requirement.

Master creates:

`workflow/tasks/<task-id>/TASK.md`

from:

`workflow/tasks/TEMPLATE.md`

Then Master registers the task in:

`workflow/INDEX.md`

Possible initial states:

- PLANNED
- READY
- BLOCKED

---

# 5. User Approval Gate

The user holds **two** gates. Do not conflate them.

## 5.1 Start gate — may this work begin?

Master does not automatically launch formal work.

Flow:

Master
→ proposes NEXT TASK
→ explains why
→ user approves
→ task becomes ACTIVE
→ dedicated conversation starts

## 5.2 Acceptance gate — does this result become project state?

Master does not automatically accept a finished result.

Flow:

Task or D0 returns
→ Master integration review
→ Master produces a MASTER RECOMMENDATION
→ Master writes a USER REVIEW BRIEF
→ user returns APPROVE / REJECT / MODIFY / INVESTIGATE MORE
→ only on APPROVE does the decision become `FINAL ACCEPTED`

The full acceptance gate is specified in Sections 13A–13C.

The user remains the final execution-control gate **and** the final research-decision authority.

---

# 6. Sequential Execution Policy

Default:

`ONE ACTIVE EXECUTION TASK`

Example:

D1 ACTIVE
D2 READY
D3 PLANNED
D4 BLOCKED

After D1 is accepted:

D1 DONE
D2 ACTIVE

This reduces:

- unfinished parallel work;
- dependency drift;
- conflicting modifications;
- coordination overhead.

---

# 7. Parallel Execution Exception

Parallel execution is allowed only when Master explicitly verifies:

1. neither task depends on the other's unfinished result;
2. filesystem modifications do not conflict;
3. scientific decisions do not conflict;
4. both tasks can be independently validated;
5. parallel execution materially saves time.

Master then marks:

`PARALLEL SAFE: YES`

Typical acceptable case:

- one GPU preprocessing job runs for hours;
- D0 independently investigates paper evidence;
- they modify unrelated areas.

If uncertain:

do not parallelize.

---

# 8. D0 Decision Flow

When Master encounters a research/architecture uncertainty:

MASTER
→ create D0 decision
→ D0 investigates
→ Codex adversarial review
→ Claude verifies findings
→ D0 recommends
→ MASTER reviews
→ MASTER RECOMMENDATION: ACCEPT / ACCEPT WITH FOLLOW-UP / REWORK / REJECT / BLOCKED
→ MASTER writes USER REVIEW BRIEF
→ USER: APPROVE / REJECT / MODIFY / INVESTIGATE MORE
→ on APPROVE: `FINAL ACCEPTED`

Until the **user** approves it:

the recommendation is not project-wide truth.

Master's endorsement narrows the question and vouches for the evidence. It does not settle it.

Finally accepted decisions may update:

- `workflow/decisions/`
- `workflow/CONTEXT.md`
- affected TASK.md files
- dependency order.

---

# 9. D-Task Execution Flow

For D1/D2/D3...:

MASTER creates TASK.md

↓

USER approves

↓

NEW CONVERSATION starts

↓

Task reads:
CLAUDE.md
+ applicable rules
+ CONTEXT.md
+ TASK.md
+ explicitly required files

↓

Task executes only its bounded scope

↓

Task performs verification

↓

Codex independent review

↓

Claude verifies Codex findings

↓

Task writes:
CODEX_REVIEW.md
HANDOFF.md

↓

Task stops

↓

USER tells Master:
"<task-id> finished; review its handoff."

↓

MASTER performs integration review

↓

MASTER RECOMMENDATION:
ACCEPT / ACCEPT WITH FOLLOW-UP / REWORK / REJECT / BLOCKED

↓

MASTER writes USER REVIEW BRIEF
(required whenever the result carries a material decision — Section 13C)

↓

USER: APPROVE / REJECT / MODIFY / INVESTIGATE MORE

↓

on APPROVE: `FINAL ACCEPTED`
Master integrates into global project state

---

# 10. Scope Control

A D-task must not silently expand into another task.

If it discovers another issue:

### If not required to finish current task

Record it in HANDOFF as:

`FOLLOW-UP CANDIDATE`

Continue current task.

### If it blocks current task

Report:

`TASK BLOCKER`

and stop unsafe execution.

### If it changes project-wide assumptions

Report:

`MASTER-IMPACTING FINDING`

Provide:

- evidence;
- affected tasks;
- whether current task can continue safely.

Master decides what happens next.

---

# 11. Codex Task Review

Every formal D-task receives Codex review before handoff.

Order:

Claude implementation / investigation

↓

Claude verification

↓

Codex review

↓

Claude verification of Codex findings

↓

final corrections if justified

↓

HANDOFF

Codex should receive focused context:

- TASK.md;
- relevant diff/result;
- verification results;
- required authority evidence;
- known uncertainty.

Do not dump unrelated Master history into Codex.

---

# 12. Codex Finding Classification

Claude classifies each material Codex finding as:

- CONFIRMED
- PLAUSIBLE
- REJECTED
- UNVERIFIED

Research-significant changes should normally require CONFIRMED evidence.

Codex review failure due to:

- quota;
- authentication;
- timeout;
- runtime failure;

must be reported:

`CODEX REVIEW UNAVAILABLE`

It is not PASS.

It must also be stated in the USER REVIEW BRIEF (Section 13C). The user decides whether to proceed without independent review; Master does not make that call silently.

---

# 13. Master Integration Review

Master does not accept HANDOFF blindly.

Master reviews:

- TASK.md;
- HANDOFF.md;
- CODEX_REVIEW.md;
- relevant repository state;
- task Definition of Done;
- Master-impacting findings.

Master returns one **MASTER RECOMMENDATION**:

- ACCEPT
- ACCEPT WITH FOLLOW-UP
- REWORK
- REJECT
- BLOCKED

**This is a recommendation, not an acceptance.**

`REWORK`, `REJECT`, and `BLOCKED` may be acted on immediately — they return work to the task owner and change no project state. Master should still inform the user.

`ACCEPT` and `ACCEPT WITH FOLLOW-UP` **do change project state** and therefore do not take effect on their own. They proceed to the user review gate in Section 13B.

Only after the **user** approves may Master:

- mark a task DONE;
- unblock downstream tasks;
- update CONTEXT.md;
- change overall pipeline status;
- record any decision as `FINAL ACCEPTED`.

---

# 13A. Finding vs Decision

These are different things and must never be merged into one statement.

## FINDING

A finding is **what was discovered**.

- an observation, a contradiction, a measurement, a defect, a piece of evidence;
- it has an evidence class (PAPER FACT, OBSERVED IMPLEMENTATION, OBSERVED DATA, INFERENCE, UNKNOWN, …);
- it has a verification status (CONFIRMED / PLAUSIBLE / REJECTED / UNVERIFIED);
- it is true or false independently of what anyone wants to do about it.

Claude, Codex, D0, and D-tasks may all **produce and verify findings**. That is their job.

## DECISION

A decision is **what will be done about a finding**.

- adopt, reject, defer, deviate, correct, re-scope, re-order;
- it is a choice, not an observation;
- it changes project state, artifacts, or scientific interpretation.

**Only the user makes material decisions.** See Section 13B for what counts as material.

## Why the separation matters

A confirmed finding does not imply its remedy. "The serializer renders 0.5 cm as 0" is a finding. "Therefore change the formatter" is a decision — and a different remedy, or accepting the defect as a disclosed limitation, may be the right call.

Collapsing the two lets an AI-selected remedy enter the project wearing the authority of a verified measurement.

When reporting, always state them separately:

```
FINDING:   <what is true>  [evidence class] [verification status]
DECISION:  <what to do>    [proposed by]    [requires user approval: yes/no]
```

An AI may state a **proposed** decision and argue for it. It may not enact a material one.

---

# 13B. User Review Gate

## The rule

> A material decision becomes `FINAL ACCEPTED` **only** when the user approves it.
>
> Unanimity among Master, D0, the D-task, and Codex does **not** substitute for user approval.

Agreement between models is not independent confirmation. Models share training data, share framing, and are given the same brief. Convergence is weak evidence, and it is never authority.

## What is material

Any decision that could materially change the reproduced results, or that fixes an interpretation the reproduction will be judged on. Including:

- paper interpretation;
- architecture;
- implementation choice;
- deviation;
- dataset / annotation semantics;
- preprocessing;
- training protocol;
- evaluation protocol;
- shared artifact semantics;
- cache / checkpoint validity;
- dependency ordering;
- scientifically meaningful assumptions;
- anything else that could materially change reproduced results.

**When in doubt, treat it as material.** The cost of an unnecessary brief is a short paragraph. The cost of a silently adopted decision is a reproduction nobody can defend.

## What is not material

Routine execution that changes no scientific behaviour and no shared contract: local refactors within an approved scope, test scaffolding, log formatting, documentation corrections that only make a comment describe existing code accurately.

These are reported in the HANDOFF and do not require their own brief. They may still be mentioned in a brief's summary if they help the user judge the whole.

## The four user actions

| Action | Meaning | Effect |
|---|---|---|
| `APPROVE` | The proposed decision is adopted | Becomes `FINAL ACCEPTED`. Master integrates into global project state |
| `REJECT` | The proposed decision is not adopted | Master records the rejection and its reason. The finding stands; the remedy does not |
| `MODIFY` | Adopted with changes the user specifies | Master records the user's version as the decision, not the proposed one. If the modification is substantial, Master re-verifies before integrating |
| `INVESTIGATE MORE` | Not enough evidence to decide | Master defines the additional investigation and returns it to D0 or a D-task. State stays unresolved |

The user's wording is authoritative. If the user's instruction and Master's recommendation differ, the user's instruction governs. Master may state a disagreement once, with evidence, and must then proceed as instructed.

## Recording

- The user's action, its date, and any modification are recorded in the decision file's Master Resolution section, or in the task's acceptance record.
- A decision file's status becomes `ACCEPTED` only after user `APPROVE`.
- Master's recommendation is retained alongside it. If the user's decision differs, both are kept — the disagreement is part of the record.

## What this does not change

This gate adds a step. It removes nothing.

Verification, Codex adversarial review, Claude's verification of Codex findings, Master integration review, impact-check escalation, sequential execution, scope control, and session-continuity rules all apply **unchanged and in full, before** the gate is reached. The gate is the last step, not a replacement for the earlier ones.

A brief must never be used to hand the user an unreviewed result. If Codex review was unavailable, the brief says so and says it is not a PASS.

---

# 13C. USER REVIEW BRIEF

Master writes this after integration review, whenever the result carries a material decision.

## Design constraint

**Keep it short.** The brief is a decision aid, not a re-narration of the investigation.

Do not restate the full research process. The decision file and the HANDOFF hold the detail; the brief points at them. Aim for something the user can act on without opening anything else — while making it obvious where to look if they want to.

If a section has nothing in it, write `None` and move on. Do not pad.

## Required structure

```
# USER REVIEW BRIEF — <decision-id or task-id>

## What was found
<the findings, briefly. Separate each finding from its remedy.>

## Evidence
<where it came from: file, line, paper section, measurement.
 Say what was measured directly and what was accepted on report.>

## Claude / Codex disagreement
<any unresolved disagreement, and each side's position.
 "None" if they converged — and say so plainly, since convergence
 is not confirmation.>

## Finding verification status
CONFIRMED:   <...>
PLAUSIBLE:   <...>
REJECTED:    <...>  (with why)
UNVERIFIED:  <...>

## Proposed decision
<what Master recommends doing. One paragraph.>

## Decision authority / classification
<PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION / OBSERVED DATA /
 INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / UNKNOWN>
<and: is this Master's call, or genuinely the user's? Say which.>

## Impact
<tasks, artifacts, stages affected. What becomes unblocked or blocked.>

## Remaining UNKNOWN
<what is still not established, and whether it matters for this decision.>

## What you need to decide
<the specific question, stated as a question.
 APPROVE / REJECT / MODIFY / INVESTIGATE MORE>
```

## Honesty requirements

The brief must not oversell. Specifically:

- Distinguish what Master **re-verified directly** from what it **accepted on the task's report**.
- State disagreement even when Master thinks Codex was wrong.
- Never present an INFERENCE or IMPLEMENTATION CHOICE as a PAPER FACT.
- Never let "all tests pass" stand in for research fidelity.
- If the recommendation rests on an unverified number, say so in the brief, not only in the decision file.
- If Master's own earlier statement was corrected during review, say so.

---

# 14. Milestone-Level Review

Task-level Codex review is not enough for major milestones.

At major milestones such as:

- Stage 1 complete;
- Stage 2 complete;
- evaluation pipeline complete;
- final reproduction complete;

Master requests a separate integration-level Codex review.

The review asks whether:

- all required work packages integrate correctly;
- accepted D0 decisions are reflected consistently;
- implementation and tests agree with the intended contract;
- runtime artifacts support the claimed completion;
- paper fidelity has been independently verified;
- no task-local assumption became an unnoticed global assumption.

A milestone is marked complete only after milestone review, Master's recommendation, **and user approval through the Section 13B gate.**

A milestone completion is always material: it asserts that a stage of the reproduction is done. Master writes a USER REVIEW BRIEF for it like any other material decision, and the brief must state plainly what remains UNVERIFIED at the point of completion.

Task-level user approvals do not aggregate into milestone approval. The user approves the milestone as its own decision.

---

# 15. Context Maintenance

Master updates `workflow/CONTEXT.md` only for information that future tasks genuinely need.

Examples:

- accepted architecture decision;
- changed shared pipeline;
- changed dependency;
- stable environment fact;
- cross-task constraint.

Do not place:

- temporary debugging logs;
- long experiment traces;
- conversation summaries;
- task-local implementation details;

into CONTEXT.md.

Keep shared context compact.

---

# 16. Master Context Conservation

Master should avoid repeatedly re-reading the entire repository.

Master should rely on:

- MASTER.md
- CONTEXT.md
- INDEX.md
- accepted D0 decisions
- task HANDOFFs

and only inspect deeper evidence when required for integration or decision-making.

This keeps Master useful as a long-lived project supervisor.

---

# 17. Core Principle

The system should behave like:

USER
= principal investigator / final authority

MASTER
= project supervisor

D0
= research / architecture lead

D1/D2/D3...
= senior engineers owning bounded stages

Codex
= independent reviewer

Files
= persistent shared memory

Conversation context
= temporary working memory

The architecture and project state must survive even if any individual conversation is discarded.

And one thing more:

**No AI in this system decides what is scientifically true or what the reproduction will claim.**

They find, verify, challenge, and recommend. The user decides. A workflow that lets agreement between models stand in for that is not a research workflow — it is a machine for laundering assumptions into conclusions.

---

# 18. Escalation / Objection Protocol

A D-task owner is expected to challenge the current plan when execution reveals
evidence that the plan, architecture, research interpretation, dependency
structure, or task contract may be wrong.

A D-task must not silently follow a known-invalid plan merely because it appears
in TASK.md.

At the same time, a D-task must not silently redesign the project.

Default escalation path:

D-task
→ detect inconsistency
→ `/impact-check` when impact is uncertain
→ Master triage
→ local fix / task rework / new task / D0 decision

---

## 18.1 Local Task Issue

A D-task may resolve an issue locally only when all of the following are true:

- the issue is inside the existing TASK.md scope;
- intended behavior is already unambiguous;
- it does not change research interpretation;
- it does not change project architecture;
- it does not change another task's assumptions or dependencies;
- it does not materially alter the Definition of Done;
- it does not introduce a new project-wide behavior.

Examples:

- incorrect local path;
- straightforward implementation defect with an already-defined intended behavior;
- missing defensive handling for an input already covered by the task contract;
- missing task-local verification.

The task must document the correction and verification in HANDOFF.md.

If any of these conditions are uncertain, use `/impact-check`.

---

## 18.2 Master-Impacting Finding

A D-task must report:

`MASTER-IMPACTING FINDING`

when evidence may affect:

- project architecture;
- accepted research interpretation;
- paper-fidelity assumptions;
- cross-task dependency;
- another task's contract;
- milestone feasibility;
- shared artifact semantics;
- evaluation validity;
- reproducibility claims;
- global runtime assumptions.

Required report:

Finding:

Evidence:

Evidence class:

Affected task(s):

Current task impact:

Can current task continue safely:

Recommended action:

The D-task must not directly modify:

- `workflow/MASTER.md`
- `workflow/CONTEXT.md`
- `workflow/INDEX.md`
- another task's TASK.md

Master owns project-level integration and triage.

---

## 18.3 Stop-Safe Rule

If continuing execution may:

- generate scientifically invalid artifacts;
- waste substantial compute on a known-questionable configuration;
- corrupt or overwrite important artifacts;
- make downstream results uninterpretable;
- violate stronger authority;
- violate an accepted project decision;
- cross an unauthorized research or architecture boundary;

the task must stop at the nearest safe point.

Report:

`TASK BLOCKER — MASTER REVIEW REQUIRED`

Do not continue merely to satisfy the original Definition of Done.

A safely blocked task is preferable to a completed invalid task.

---

## 18.4 Master Triage

Master is the first project-level escalation point.

After receiving a finding, Master chooses one of four paths.

### A. LOCAL FIX

Use when intended behavior is already unambiguous and the correction remains
inside the current task.

Master may authorize the same D-task to continue.

### B. TASK REWORK / CONTRACT CHANGE

Use when the current task remains the correct owner but its scope,
verification requirements, assumptions, or Definition of Done must change.

Master updates the formal TASK.md before execution resumes.

### C. NEW EXECUTION TASK

Use when the discovered issue is implementation, data, operational, or
verification work that should be isolated from the current task.

Master creates a separate bounded work package.

### D. D0 DECISION

Use when resolution requires research / architecture adjudication.

Examples:

- conflicting primary evidence;
- ambiguous architecture;
- unsupported scientific assumption;
- cross-stage methodological choice;
- deliberate deviation from the paper;
- multiple technically valid implementations with different scientific meaning.

Master creates or activates a formal D0 decision.

---

## 18.5 D0 Is Not the Default Escalation Target

D0 is the Research / Architecture Lead.

D0 is not:

- a general debugger;
- a second implementation engineer;
- the default destination for difficult bugs.

A D-task normally reports project-impacting uncertainty to Master first.

Master decides whether D0 is required.

Ordinary implementation defects should not be sent to D0 merely because they
are difficult.

---

## 18.6 Engineering Objection

A D-task owner may explicitly challenge:

- an implementation approach assigned in TASK.md;
- an assumption in TASK.md;
- a dependency declared by Master;
- a supposedly settled interpretation contradicted by new evidence;
- a verification requirement that cannot prove the property it claims to verify.

Use:

`ENGINEERING OBJECTION`

Required format:

Claim being challenged:

Observed evidence:

Evidence class:

Why the current plan may be invalid:

Potential consequence if ignored:

Can execution continue safely:

Suggested Master triage:

The objection must be evidence-backed.

An objection does not itself overturn current project state.

Master must adjudicate or escalate it.

---

## 18.7 New Evidence Overrides Obedience

TASK.md is an execution contract.

It is not permission to ignore stronger evidence.

If new primary-source, repository, runtime, or artifact evidence contradicts an
assumption inside TASK.md:

1. preserve the evidence;
2. stop unsafe work if necessary;
3. run `/impact-check` when impact is uncertain;
4. report the contradiction to Master when project-impacting;
5. wait for triage when required;
6. resume only under the resulting contract or accepted decision.

The authority hierarchy in CLAUDE.md still applies.

A lower-authority task instruction must not override higher-authority evidence.

---

## 18.8 User Escalation

The user may challenge any level directly.

If the user identifies a possible project-wide error, Master should first
classify the concern.

Master may then:

- verify it directly when the answer is unambiguous;
- assign a bounded execution task;
- request `/impact-check` on an existing task finding;
- create a D0 decision;
- pause affected downstream work.

A previous PASS result does not invalidate a new evidence-backed concern.

---

## 18.9 Codex Findings

A potentially project-impacting Codex finding follows the same escalation path:

Codex finding
→ Claude verification
→ CONFIRMED / PLAUSIBLE / REJECTED / UNVERIFIED
→ project-impacting CONFIRMED finding
→ Master triage

Codex must not directly:

- change project architecture;
- change dependencies;
- create accepted D0 decisions;
- alter global workflow state.

Codex is an independent reviewer, not project or scientific authority.

---

## 18.10 Core Escalation Principle

Use this decision rule:

Local implementation certainty
→ engineer may fix within scope

Impact uncertain
→ `/impact-check`

Project-wide impact
→ Master

Research / architecture uncertainty
→ Master delegates to D0

Unsafe continuation
→ stop first, escalate second

The D-task is responsible for detecting inconsistencies in the evidence it can
see.

It is not responsible for already knowing every downstream consequence.

No agent should knowingly produce invalid downstream work merely to preserve
the original schedule.

---

## 18.11 Impact-Check Skill

Formal D-tasks may use the user-level Claude skill:

`/impact-check`

Purpose:

> Perform scoped cross-task impact triage when a D-task discovers an
> inconsistency but cannot confidently determine whether the issue is local or
> project-impacting.

The skill does not replace Master.

The skill does not make project-wide decisions.

It provides structured triage before escalation.

---

### When to Invoke

A D-task should invoke `/impact-check` when it observes evidence such as:

- TASK.md assumptions disagree with repository reality;
- implementation disagrees with specification or primary evidence;
- runtime behavior contradicts the expected contract;
- an artifact's semantics differ from what the task expects;
- a declared dependency appears false;
- verification cannot prove the property it claims to verify;
- a fix may affect another stage or task;
- continuing may waste substantial compute;
- continuing may produce scientifically invalid results;
- the engineer cannot confidently determine whether the issue is local.

The engineer does not need full project knowledge before invoking the skill.

Its responsibility is to detect inconsistency, not to already know every
downstream consequence.

---

### Context Loaded by Impact Check

The impact check should inspect:

1. `CLAUDE.md`
2. applicable `.claude/rules/`
3. `workflow/WORKFLOW.md`
4. `workflow/MASTER.md`
5. `workflow/CONTEXT.md`
6. `workflow/INDEX.md`
7. the current task's `TASK.md`

Then perform only scoped retrieval needed for the finding.

Use Graphify for navigation when useful, then verify conclusions against actual
repository files and stronger authority evidence.

Do not automatically load the entire repository.

---

### Impact-Check Classifications

`/impact-check` returns one primary classification:

#### LOCAL

The issue remains inside the current task and intended behavior is unambiguous.

Action:

Current engineer may fix it within scope, verify the correction, and document
it in HANDOFF.md.

#### MASTER-IMPACTING

The issue may affect project state, another task, shared artifact semantics,
dependencies, milestone validity, or evaluation validity.

Action:

Return a `MASTER-IMPACTING FINDING` to Master.

#### D0-CANDIDATE

The issue requires research / architecture adjudication.

Action:

Return the finding to Master.

Only Master decides whether to create a formal D0 decision.

#### BLOCKER

Continuing may create invalid results, waste substantial compute, corrupt
artifacts, or cross an unauthorized research boundary.

Action:

Stop at the nearest safe point and report:

`TASK BLOCKER — MASTER REVIEW REQUIRED`

---

### Authority Boundary

`/impact-check` performs triage only.

It must not:

- alter `workflow/MASTER.md`;
- alter `workflow/CONTEXT.md`;
- alter `workflow/INDEX.md`;
- alter another task's TASK.md;
- create or accept a D0 decision;
- change task status;
- silently redesign project architecture;
- continue the underlying implementation while performing the check.

The escalation chain remains:

D-task
→ `/impact-check` when needed
→ Master
→ Master triage
→ local fix / task rework / new task / D0

---

### Core Principle

Bounded task context is intentional.

A D-task should not load the entire project merely so it can detect every
possible downstream consequence.

Instead:

Engineer detects inconsistency
→ Impact Check expands context only as needed
→ Master owns global consequences
→ D0 handles research / architecture ambiguity when delegated

This preserves focused task execution without sacrificing project-level error
detection.

---

---

## 18.x Escalation and the User Review Gate

Escalation decides **who handles** a finding. It does not decide **what is done** about it.

An escalation that ends in a local fix, a task rework, or a new task is routing, and needs no brief of its own.

An escalation that produces a **material decision** under Section 13B — a changed research interpretation, a changed architecture, a changed dependency order, a changed artifact contract — still passes through the user review gate before it becomes project state. Master triage selects the route; it does not confer acceptance.

`/impact-check` classifies impact. Classification is a finding, not a decision (Section 13A).

---

# 19. Migration

The user review gate (Sections 13A–13C) applies to all work from the point it was adopted.

**It is not retroactive.** Completed evidence gathering and completed Codex reviews are not re-run because the gate was added afterwards.

## D0-008 — migration case

`D0-008_stage1-text-template` completed its investigation, its Codex adversarial review, and Claude's verification of Codex findings before this gate existed.

It is **not** re-opened. Specifically:

- its evidence survey is not re-gathered;
- its Codex adversarial review is not re-run;
- its Claude verification of Codex findings is not repeated;
- Section 11 stands as returned.

From Section 11 onward it follows the current flow:

```
D0-008 Section 11 (returned)
→ Master integration review
→ MASTER RECOMMENDATION
→ USER REVIEW BRIEF
→ USER final decision
→ on APPROVE: FINAL ACCEPTED
```

If Master already recorded an acceptance in that decision's Master Resolution section before this gate was adopted, that acceptance is reclassified as a **MASTER RECOMMENDATION**. Master owes the user a USER REVIEW BRIEF for it, and the decision reaches `FINAL ACCEPTED` only on the user's `APPROVE`.

Master must not treat a pre-gate acceptance as though the gate had been satisfied.

## General rule for other pre-gate work

Any decision recorded as accepted before this gate was adopted, and which is **material** under Section 13B, is subject to the same treatment: reclassified as a recommendation, briefed to the user, and finalised only on approval.

Work that is not material does not need retrospective briefing.


---

# 20. Block-Centric Workflow and Skill Integration

**USER decision 2026-08-21 / 2026-08-22.** Two documents now sit above this one for anything
concerning structure and method:

| Document | Governs |
|---|---|
| `workflow/BLOCKS.md` | block ownership, roles, the `HANDOFF.md` communication rule |
| `workflow/SKILLS.md` | which skill is used at which layer, by whom, and when it is worth it |
| `workflow/blocks/SPEC_TEMPLATE.md` | the 15-section implementation contract |

## 20.1 What replaced what

**Replaced.** Fine-grained `D<n>` task cards as the unit of work. Work is now owned by a **Block**
that holds an entire technical chain. Existing `workflow/tasks/D*` directories are unchanged and
become internal work items of their block; `workflow/INDEX.md` remains the registry.

**Unchanged, and still binding.** §13A Finding vs Decision · §13B User Review Gate · §13C USER
REVIEW BRIEF · §18 escalation and Master-impacting findings · the authority hierarchy in
`CLAUDE.md` §3 · `workflow/DECISION_LEDGER.md` as the project-level record · the USER as final
research authority.

## 20.2 Roles

`USER` · `MASTER` · **`BLOCK OWNER`** · **`BLOCK REVIEWER`** · **`INTEGRATOR`** · `CODEX`.

The Block Reviewer is an independent Claude context working **synchronously** with the Owner.
**Codex does not replace the Block Reviewer** — it is the third layer, adversarial and
context-free, and Codex PASS is not Block PASS.

## 20.3 The acceptance flow

```
Block Plan
  → USER approves scope
  → Owner implementation + self-verification
       ↕  Reviewer synchronous independent verification
  → 4-axis completion review
  → Codex milestone adversarial review
  → Master integration
  → USER Acceptance Grill          (one material criterion per round)
  → USER FINAL ACCEPTED
```

This **extends** §13B rather than replacing it. §13B said the user returns
APPROVE / REJECT / MODIFY / INVESTIGATE MORE on a brief. It now happens **item by item**, in
`grilling` mode: Master looks up the evidence first, presents one criterion, and waits. Nothing is
`FINAL ACCEPTED` until every material criterion has been through a round. Full format:
`SKILLS.md` §11.

## 20.4 Four-axis review replaces the single completion claim

`STANDARDS` · `SPEC` · `SOURCE / EVIDENCE` · `SCIENTIFIC / SEMANTIC`, **reported separately**.
Axis 4 assumes the code runs, the tests pass and the SPEC is met, then asks how the result could
still be scientifically wrong. `SKILLS.md` §9.

## 20.5 Do not rebuild fine-grained process out of skills

Skills are method tools, not authority. They apply to **material** changes, high-risk pipeline
stages, the moment **before an expensive run**, and Block milestones.

They do **not** apply to internal work items inside an approved SPEC, comments, formatting,
read-only investigation, or re-runs of accepted deterministic steps. The Block Owner manages those
alone. `SKILLS.md` §5.

## 20.6 Handoff

`session-handoff` and the block `HANDOFF.md` only. A handoff is temporary continuity — never a
formal finding, decision, or acceptance. Persistent state goes back into the block and Master
workflow files.
