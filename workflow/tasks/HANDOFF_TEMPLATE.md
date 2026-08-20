# D-Task Handoff

> Formal result returned from one D-task to Master.
> This file records the current completed/blocked state of this task only.
> It does not update global project state by itself.

---

## Task ID

`D<number>_<short-slug>`

---

## Status

Use one:

- `DONE`
- `BLOCKED`
- `NEEDS MASTER DECISION`
- `REWORK REQUIRED`

Current:

`DONE`

---

## 1. Objective Result

State whether the task objective was achieved.

Summary:

---

## 2. Scope Compliance

### Completed In Scope

- ...

### Explicit Non-Scope Respected

- ...

### Scope Deviations

None.

If any deviation occurred, explain why.

---

## 3. Files Changed

- ...

If none:

`None.`

---

## 4. Artifacts Produced

- ...

Examples:

- checkpoints
- generated protocol files
- logs
- indexes
- plots
- reports

If none:

`None.`

---

## 5. Evidence Used

List only evidence materially used to make decisions or verify correctness.

For research-significant work, distinguish:

- PAPER FACT
- UPSTREAM-SUPPORTED
- VERIFIED RUNTIME FACT
- IMPLEMENTATION FACT
- INFERENCE
- UNKNOWN / UNVERIFIED

---

## 6. Decisions Made Within Task Scope

Record only decisions explicitly permitted by TASK.md.

Do not introduce new project-wide architecture decisions here.

- ...

---

## 7. Verification Performed

### Tests

- ...

### Runtime Checks

- ...

### Artifact Checks

- ...

### Research Fidelity Verification

- ...

---

## 8. Verification Result

Overall:

- `PASS`
- `PARTIAL`
- `FAIL`
- `BLOCKED`

Details:

---

## 9. Codex Review Result

Review status:

- `PASS`
- `ISSUES FOUND`
- `CODEX REVIEW UNAVAILABLE`
- `NOT APPLICABLE`

Codex review file:

`workflow/tasks/<task-id>/CODEX_REVIEW.md`

Summary:

---

## 10. Confirmed Codex Findings

Findings independently verified by Claude:

- ...

If none:

`None.`

---

## 11. Rejected / Unverified Codex Findings

### REJECTED

- ...

### PLAUSIBLE

- ...

### UNVERIFIED

- ...

---

## 12. Master-Impacting Findings

Anything discovered that may affect:

- project architecture;
- accepted research interpretation;
- another task;
- dependency order;
- milestone feasibility;
- shared runtime assumptions.

If none:

`None.`

Otherwise, for each finding provide:

Finding:
Evidence:
Affected tasks:
Can current workflow continue safely:

---

## 13. Remaining Risks

- ...

If none:

`None known within task scope.`

---

## 14. Blocked Items

- ...

If none:

`None.`

---

## 15. Recommended Master Update

State exactly what Master should consider updating after acceptance.

Examples:

- mark task DONE;
- unblock D2;
- update CONTEXT.md;
- create D0 decision;
- change dependency order.

---

## 16. Recommended Next Action

Recommendation:

Do not start it from this task.

Master decides the next action.

---

## 17. Completion Statement

This task has stopped execution after producing this HANDOFF.

The task does not claim project-level or milestone-level completion.

Master must review and return:

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`
