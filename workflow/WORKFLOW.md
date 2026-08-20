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
