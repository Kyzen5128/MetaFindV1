# MetaFindV1 Task Index

> Master-maintained registry of all formal workflow tasks.
> Detailed execution instructions belong in each task's TASK.md.
> Detailed results belong in each task's HANDOFF.md.

---

## Status Definitions

Use only:

- `PLANNED`
- `READY`
- `ACTIVE`
- `BLOCKED`
- `REVIEW`
- `DONE`
- `REWORK`
- `REJECTED`

---

## Role Definitions

- `MASTER` — project-level orchestration, integration, status, dependency control
- `D0` — research / architecture / evidence decisions requested by Master
- `D1+` — bounded stage / work-package execution tasks
- `CODEX` — independent reviewer

---

## Active Tasks

| ID | Task | Role | Status | Depends On | Parallel Safe | Task Path |
|---|---|---|---|---|---|---|
| — | None | — | — | — | — | — |

---

## Planned Tasks

| ID | Task | Role | Status | Depends On | Notes |
|---|---|---|---|---|---|
| — | None initialized yet | — | PLANNED | — | — |

---

## Completed Tasks

| ID | Task | Accepted By Master | Handoff | Codex Review |
|---|---|---|---|---|
| — | None | — | — | — |

---

## Blocked Tasks

| ID | Task | Blocked By | Required Resolution |
|---|---|---|---|
| — | None | — | — |

---

## Decision Queue

Research / architecture decisions that Master has delegated to D0.

| Decision ID | Question | Status | Decision File | Blocks |
|---|---|---|---|---|
| — | None | — | — | — |

---

## Task Naming Convention

Use:

`D<number>_<short-slug>`

Examples:

- `D1_stage1-prerequisites`
- `D2_stage1-training`
- `D3_gallery-index`
- `D4_stage2-preparation`

D0 decisions use:

`D0-<number>_<short-slug>`

Examples:

- `D0-001_tau`
- `D0-002_u16-sharing`

Task directory:

`workflow/tasks/<task-id>/`

Decision file:

`workflow/decisions/<decision-id>.md`

---

## Master Rules

1. Only Master changes task status in this file.
2. A task becomes `ACTIVE` only after user approval.
3. `READY` means all known dependencies are satisfied.
4. `BLOCKED` must name the blocker explicitly.
5. `DONE` means:
   - task Definition of Done satisfied;
   - verification completed;
   - Codex review completed;
   - Master accepted the HANDOFF.
6. Sequential execution is the default.
7. Parallel execution requires explicit `Parallel Safe = YES`.
8. D-task conversations must not modify this file directly.
