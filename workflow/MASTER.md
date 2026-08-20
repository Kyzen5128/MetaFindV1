# MetaFindV1 Master Control

> This document is maintained by the Master / Orchestrator.
> It records project-level state, dependencies, assignments, integration status, and next actions.
> It is not scientific authority and must not replace primary evidence, audit documents, implementation contracts, or runtime verification.

---

## 1. Project Goal

TO BE INITIALIZED

---

## 2. Current Project Phase

TO BE INITIALIZED

---

## 3. Overall Pipeline

TO BE INITIALIZED

---

## 4. Current Status

### DONE

None recorded yet.

### ACTIVE

None.

### READY

None.

### BLOCKED

None.

### DECISION REQUIRED

None.

---

## 5. Dependency / Execution Order

TO BE INITIALIZED

---

## 6. Active Assignment

**Task:** None  
**Owner:** None  
**Started:** —  
**Status:** —  

---

## 7. Pending Assignments

See:

`workflow/INDEX.md`

---

## 8. D0 Research / Architecture Decisions

No active D0 decision.

Formal accepted decisions are stored under:

`workflow/decisions/`

---

## 9. Integration Status

No task result has been integrated into the new workflow yet.

---

## 10. Review Status

### Task-level Codex Reviews

None.

### Milestone / Integration Reviews

None.

---

## 11. Current Blockers

None recorded yet.

---

## 12. Next Recommended Action

TO BE INITIALIZED

The Master may recommend the next task, but must not start a new D-task without user approval.

---

## Master Operating Rules

The Master acts as the project manager and integration owner.

The Master is responsible for:

- maintaining the global project view;
- tracking what is DONE, ACTIVE, READY, BLOCKED, or awaiting a decision;
- maintaining task dependencies and execution order;
- deciding when D0 research/architecture support is required;
- preparing self-contained task contracts for D1, D2, D3...;
- receiving and validating task HANDOFFs;
- integrating accepted results;
- requesting milestone-level Codex review;
- proposing the next task to the user.

The Master should not spend its context on long bounded implementation work, prolonged debugging, dataset processing, or deep single-question research.

Those belong to D0 or D1/D2/D3 task conversations.

Default execution policy:

1. Master proposes the next task.
2. User approves.
3. A dedicated task conversation executes it.
4. Task performs its own verification.
5. Codex performs independent review.
6. Task writes HANDOFF.md.
7. User tells Master the task is finished.
8. Master reads TASK.md, HANDOFF.md, and CODEX_REVIEW.md.
9. Master returns ACCEPT / ACCEPT WITH FOLLOW-UP / REWORK / REJECT / BLOCKED.
10. Only after acceptance does Master update global project state.

Execution is sequential by default.

Parallel execution is allowed only when Master explicitly marks tasks `PARALLEL SAFE` and confirms that their dependencies and filesystem modifications do not conflict.
