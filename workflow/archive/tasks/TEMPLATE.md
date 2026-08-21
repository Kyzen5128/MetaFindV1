# D-Task Execution Contract

> This file is the authoritative execution contract for one bounded work package.
> The task owner must stay within scope, satisfy the Definition of Done, perform verification, obtain Codex review, and return a HANDOFF to Master.

---

## Task ID

`D<number>_<short-slug>`

---

## Status

Use only:

- `PLANNED`
- `READY`
- `ACTIVE`
- `BLOCKED`
- `REVIEW`
- `AWAITING_USER_REVIEW` — execution finished and reviewed; the user has not yet decided
- `DONE` — user-approved and integrated. **Nothing else means DONE**
- `REWORK`
- `REJECTED`

Current:

`PLANNED`

> Finishing execution does not make a task `DONE`. See Section 17.

---

## 1. Objective

State one bounded, independently verifiable objective.

The task should normally correspond to one stage or one coherent work package.

If the task contains multiple unrelated root causes or cannot be independently verified, split it before execution.

---

## 2. Why This Task Exists

Explain:

- why this task is needed;
- why it should happen now;
- which project milestone it contributes to;
- what downstream work depends on it.

---

## 3. Required Shared Context

Before execution, read:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. applicable `.claude/rules/`
3. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
4. this `TASK.md`
5. if this is a resumed unfinished task and
   `workflow/tasks/<task-id>/SESSION_HANDOFF.md` exists, read it before resuming

Then read only the additional files listed under Authoritative Inputs / Relevant Files.

Do not automatically re-read the entire repository.

---

## 4. Dependencies

### Required Before Start

- ...

### Blocks

- ...

### Parallel Safety

`NO` by default.

If Master explicitly verifies that this task can run concurrently with another task:

`PARALLEL SAFE: YES`

Reason:

Potential filesystem conflicts:

---

## 5. Authoritative Inputs

List only sources needed to perform this task correctly.

For research-significant work, order by authority.

Examples:

- primary paper / TeX source;
- supplementary material;
- verified audit document;
- implementation contract;
- graph/spec;
- current implementation;
- tests;
- runtime evidence.

For each important source, state why it matters.

---

## 6. Current Relevant State

Summarize only the repository/runtime state needed for this task.

Do not copy unrelated project history.

Include known facts such as:

- current implementation behavior;
- existing artifacts;
- current configuration;
- known failures;
- relevant accepted D0 decisions.

---

## 7. Scope

### In Scope

- ...

### Explicit Non-Scope

- ...

Do not modify or investigate Non-Scope items unless a blocker makes completion impossible.

If that happens, report to Master instead of silently expanding scope.

---

## 8. Expected Deliverables

List concrete outputs.

Examples:

- implementation changes;
- generated artifacts;
- tests;
- experiment results;
- documentation updates;
- verified runtime output.

---

## 9. Likely Files / Areas

Expected files or directories:

- ...

This list is guidance, not permission to expand scope.

---

## 10. Execution Requirements

The task owner must:

1. confirm dependencies before modifying anything;
2. inspect authoritative evidence before research-significant changes;
3. make the smallest coherent change needed;
4. avoid unrelated cleanup or refactoring;
5. verify intermediate assumptions when they affect scientific behavior;
6. record any Master-impacting discovery immediately;
7. stop if a required authority decision is missing.

---

## 11. Master-Impacting Finding Rule

If execution discovers information that may change:

- project architecture;
- accepted research interpretation;
- cross-task dependency;
- another task's execution contract;
- milestone feasibility;
- global runtime assumptions;

report:

`MASTER-IMPACTING FINDING`

Include:

- finding;
- evidence;
- affected task(s);
- whether current task can safely continue.

Do not make a new project-wide decision locally.

---

## 12. Verification Requirements

Define exact verification before execution begins.

### Required Checks

- ...

### Required Tests

- ...

### Runtime / Artifact Checks

- ...

### Research Fidelity Check

If research-significant:

- verify against the authoritative source independently of tests;
- do not claim paper fidelity solely because tests pass.

---

## 13. Definition of Done

The task is not complete until all applicable items are satisfied.

- [ ] Objective achieved.
- [ ] Scope respected.
- [ ] Required deliverables produced.
- [ ] Required tests/checks pass.
- [ ] Runtime/artifact verification completed.
- [ ] Research fidelity independently checked where applicable.
- [ ] No unresolved BLOCKER remains inside task scope.
- [ ] Codex review completed.
- [ ] Material Codex findings independently verified by Claude.
- [ ] HANDOFF.md written.

The task owner does not mark the project stage DONE.
Master decides acceptance.

---

## 14. Codex Review Requirement

Every formal D-task receives an independent Codex review.

Review should be scoped to this task, not the entire repository.

Provide Codex with:

- this TASK.md;
- relevant diff / implementation;
- verification results;
- necessary authority evidence;
- known uncertainties.

Ask Codex to look for:

- correctness defects;
- regression risks;
- scope violations;
- hidden assumptions;
- missing verification;
- mismatch between claimed behavior and implementation;
- research/source mismatch where applicable.

Codex must not be asked merely to confirm Claude's work.

---

## 15. Claude Verification of Codex Findings

Classify each material finding:

- `CONFIRMED`
- `PLAUSIBLE`
- `REJECTED`
- `UNVERIFIED`

Only verified findings should automatically drive research-significant corrections.

If review is unavailable because of quota/auth/runtime failure, report:

`CODEX REVIEW UNAVAILABLE`

Do not treat unavailable review as PASS.

---

## 15A. Required Completion Reporting — Finding vs Decision

`WORKFLOW.md` §13A. At completion, the task must report these **separately**. Do not merge them.

| | |
|---|---|
| **FINDING** | What is true. Observation, contradiction, measurement, defect |
| **EVIDENCE** | Where it came from — file:line, paper section, measurement, runtime output |
| **IMPLEMENTATION / PROPOSED DECISION** | What was done, or what is proposed to be done |
| **AUTHORITY** | Under whose authority — an accepted D0 decision, this TASK.md, a user ruling, or **proposed and not yet authorised** |
| **IMPACT** | Affected tasks, artifacts, stages |
| **UNKNOWN / UNRESOLVED** | What is still not established |

### What a D-task may do

- discover a problem;
- verify a problem;
- implement a choice **this TASK.md already authorises**;
- propose a recommendation.

### What a D-task may not do

A D-task must **not** declare a material project decision FINAL on the strength of:

- tests passing;
- Codex returning PASS;
- Claude and Codex agreeing.

None of those is authority. Convergence between models is not independent confirmation.

A material decision (`WORKFLOW.md` §13B) becomes project state only after Master's integration review, a USER REVIEW BRIEF, and the **user's** approval.

If execution reveals that a material choice is needed which this TASK.md does not authorise, report it under `Items Requiring USER Awareness / Decision` in the HANDOFF and, where it affects other work, as a `MASTER-IMPACTING FINDING`. Do not decide it locally.

---

## 16. Required Handoff

At completion, write:

`workflow/tasks/<task-id>/HANDOFF.md`

If Codex review ran, also write:

`workflow/tasks/<task-id>/CODEX_REVIEW.md`

The HANDOFF must summarize:

- Task ID
- Status
- Objective Result
- Files Changed
- Artifacts Produced
- Evidence Used
- Decisions Made Within Scope
- Verification Performed
- Verification Result
- Codex Review Result
- Confirmed Findings
- Rejected / Unverified Findings
- Master-Impacting Findings
- Remaining Risks
- Blocked Items
- Recommended Master Update
- Recommended Next Action

Then stop.

Do not start the next task.

---

## 17. Master Recommendation and the User Review Gate

After task completion, Master reviews:

- TASK.md
- HANDOFF.md
- CODEX_REVIEW.md
- relevant repository state

Master returns one **MASTER RECOMMENDATION**:

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`

**This is a recommendation, not an acceptance.**

`REWORK` / `REJECT` / `BLOCKED` are immediate routing results — they return work to the owner and change no project state. Master still informs the user, and must not smuggle a new material scientific decision in through routing.

`ACCEPT` / `ACCEPT WITH FOLLOW-UP` **change project state** and therefore proceed to the gate:

```
task completion
→ Master integration review
→ MASTER RECOMMENDATION
→ USER REVIEW BRIEF   (workflow/USER_REVIEW_TEMPLATE.md)
→ USER decision: APPROVE / REJECT / MODIFY / INVESTIGATE MORE
→ on APPROVE: FINAL ACCEPTED
```

Until the user approves:

- the task's integration status is `AWAITING_USER_REVIEW`;
- the task is **not** DONE;
- downstream tasks are **not** unblocked;
- `MASTER.md`, `CONTEXT.md`, and `INDEX.md` are **not** updated with the result;
- the entry sits at `AWAITING_USER_REVIEW` in `workflow/DECISION_LEDGER.md`.

> **Execution complete ≠ decision accepted.** A task may be `execution: COMPLETE` and `integration: AWAITING_USER_REVIEW` at the same time. These are two different facts and INDEX.md records them separately.

Only Master updates global task state in `workflow/INDEX.md`, and only after the user's approval.

---

## 18. Impact Check / Escalation Trigger

A D-task is responsible for its bounded work package.

It is not expected to understand every downstream consequence in the project.

However, the task must actively detect when its own assumptions, evidence,
runtime behavior, interfaces, or task contract appear inconsistent.

### Use `/impact-check` when any of the following occurs

- TASK.md assumptions do not match repository reality;
- implementation does not match the referenced specification;
- implementation appears inconsistent with primary evidence;
- runtime behavior contradicts expected behavior;
- an input or output artifact has unexpected semantics;
- a declared dependency appears incorrect;
- fixing the issue may affect another task or stage;
- verification cannot establish the property it claims to verify;
- continuing may waste substantial compute;
- continuing may produce scientifically invalid artifacts;
- the task owner cannot confidently determine whether the issue is local.

The engineer does NOT need to prove project-wide impact before invoking the
skill.

A concrete inconsistency plus evidence is sufficient.

Run:

`/impact-check`

The skill performs scoped cross-task impact triage using:

- `workflow/MASTER.md`
- `workflow/CONTEXT.md`
- `workflow/INDEX.md`
- this TASK.md
- scoped repository / Graphify retrieval
- relevant authority evidence when necessary

It must classify the finding as one of:

- `LOCAL`
- `MASTER-IMPACTING`
- `D0-CANDIDATE`
- `BLOCKER`

---

### LOCAL

If `/impact-check` returns `LOCAL`:

- the intended behavior is unambiguous;
- the issue remains inside task scope;
- no research meaning changes;
- no other task contract or dependency changes.

The current D-task may fix the issue, verify the correction, and record it in
HANDOFF.md.

---

### MASTER-IMPACTING

If `/impact-check` returns `MASTER-IMPACTING`:

report the generated:

`MASTER-IMPACTING FINDING`

to Master.

Do not silently update:

- `workflow/MASTER.md`
- `workflow/CONTEXT.md`
- `workflow/INDEX.md`
- another task's TASK.md

Master performs project-level triage.

---

### D0-CANDIDATE

If `/impact-check` returns `D0-CANDIDATE`:

report it to Master.

The D-task does not directly start D0.

Master decides whether the issue requires a formal D0 research / architecture
decision.

---

### BLOCKER

If `/impact-check` returns `BLOCKER`:

stop at the nearest safe point.

Report:

`TASK BLOCKER — MASTER REVIEW REQUIRED`

Do not continue merely to satisfy the original Definition of Done.

A safely blocked task is preferable to a completed invalid task.

---

### Engineer Detection Rule

The engineer's responsibility is not:

> know the whole project.

The engineer's responsibility is:

> notice when the evidence it observes does not match the contract it was given.

When uncertain whether an inconsistency is local or project-impacting:

use `/impact-check`.

Do not guess.
