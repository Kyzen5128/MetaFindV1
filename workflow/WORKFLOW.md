# MetaFindV1 Multi-Agent Workflow

> Operating protocol for Master, D0, D1/D2/D3..., and Codex.
> This file defines how work moves through the project.
> It does not contain scientific conclusions or task-specific implementation details.

---

# 1. Roles

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
- integrating accepted results;
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

Master decides acceptance.

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

Master does not automatically launch formal work.

Flow:

Master
→ proposes NEXT TASK
→ explains why
→ user approves
→ task becomes ACTIVE
→ dedicated conversation starts

The user remains the final execution-control gate.

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
→ ACCEPT / REWORK / REJECT / BLOCKED

Until Master accepts it:

the recommendation is not project-wide truth.

Accepted decisions may update:

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

ACCEPT / ACCEPT WITH FOLLOW-UP / REWORK / REJECT / BLOCKED

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

Master returns one:

- ACCEPT
- ACCEPT WITH FOLLOW-UP
- REWORK
- REJECT
- BLOCKED

Only after acceptance may Master:

- mark task DONE;
- unblock downstream tasks;
- update CONTEXT.md;
- change overall pipeline status.

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

Only after milestone review and Master acceptance should a milestone be marked complete.

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
