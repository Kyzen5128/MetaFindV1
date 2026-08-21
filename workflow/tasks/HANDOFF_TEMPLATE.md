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

- `EXECUTION COMPLETE` — the task's work is finished and verified. **This is the normal completion status**
- `BLOCKED`
- `NEEDS MASTER DECISION`
- `REWORK REQUIRED`

Current:

`EXECUTION COMPLETE`

> A task does not declare itself `DONE`. `DONE` means user-approved and integrated, and only Master records it after the user's approval (`WORKFLOW.md` §13B).
> **Execution complete ≠ decision accepted.**

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

## 15. USER REVIEW INPUT

> **This is not the USER REVIEW BRIEF.** It is the input Master uses to write one (`workflow/USER_REVIEW_TEMPLATE.md`).
>
> Keep it short and factual. Master compresses further.
>
> **A material item must not be omitted because every AI agreed on it.** Consensus between Claude, Codex, and Master is exactly when the user most needs to see the item. Convergence between models is not independent confirmation.

### Material Findings

State what is **true**, not what to do about it.

- Finding:
  Evidence / provenance: (file:line · paper section · measurement · runtime output)

### Material Decisions / Implementation Choices

Everything materially decided or implemented during this task — **including choices already implemented but not yet ratified by the user.**

- What was done:
  Authority: (accepted D0 decision · this TASK.md · user ruling · **proposed, not yet authorised**)

### Claude ↔ Codex Material Disagreement

- Disagreement:
  Verified disposition: `CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED`

If none:

`No material disagreement.`

State this even when you believe Codex was wrong.

### Impact

- Affected task / artifact / stage:

### Remaining UNKNOWN / Blocker

If none:

`None known.`

### Items Requiring USER Awareness / Decision

List every material item under `WORKFLOW.md` §13B, even under full consensus:

paper interpretation · architecture · implementation choice · deviation · dataset semantics · annotation semantics · preprocessing · training protocol · evaluation protocol · shared artifact semantics · cache validity · checkpoint validity · dependency ordering · scientifically meaningful assumption · anything that can materially change reproduction results

When in doubt, list it.

- ...

---

## 16. Recommended Master Update

State exactly what Master should consider updating after acceptance.

Examples:

- mark task DONE;
- unblock D2;
- update CONTEXT.md;
- create D0 decision;
- change dependency order.

---

## 17. Recommended Next Action

Recommendation:

Do not start it from this task.

Master decides the next action.

---

## 18. Completion Statement

This task has stopped execution after producing this HANDOFF.

The task does not claim project-level or milestone-level completion, and does not claim that any material decision is accepted.

Master must review and return one **MASTER RECOMMENDATION**:

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`

`ACCEPT` and `ACCEPT WITH FOLLOW-UP` are recommendations only. They proceed to a USER REVIEW BRIEF and the user's decision. This task becomes `DONE` only on the user's `APPROVE` (`WORKFLOW.md` §13B).
