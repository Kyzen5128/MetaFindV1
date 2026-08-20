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
- `DONE`
- `REWORK`
- `REJECTED`

Current:

`PLANNED`

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

## 17. Master Acceptance

After task completion, Master reviews:

- TASK.md
- HANDOFF.md
- CODEX_REVIEW.md
- relevant repository state

Master returns one of:

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`

Only Master updates global task state in `workflow/INDEX.md`.
