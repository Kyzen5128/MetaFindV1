# Codex Independent Review

> Independent review record for one formal D-task.
> Codex is a reviewer, not project authority.
> Claude must verify material findings before they affect research-significant work.

---

## Task ID

`D<number>_<short-slug>`

---

## Review Target

Describe exactly what Codex reviewed:

- TASK.md
- diff / implementation
- tests
- runtime output
- relevant evidence
- specific uncertainty

---

## Review Mode

Use one:

- `CODE REVIEW`
- `ADVERSARIAL REVIEW`
- `RESEARCH / SPECIFICATION REVIEW`
- `INTEGRATION REVIEW`

---

## Review Status

- `COMPLETED`
- `UNAVAILABLE`
- `FAILED`

Reason if unavailable/failed:

---

## 1. Review Brief Given to Codex

### Objective

...

### Claimed Behavior

...

### Relevant Evidence

...

### Changes / Result

...

### Known Uncertainty

...

### Review Request

Ask Codex to challenge rather than confirm.

---

## 2. Codex Original Findings

Preserve the substantive findings accurately.

### Finding 1

Severity:

- `BLOCKER`
- `MAJOR`
- `MINOR`
- `NIT`

Finding:

Evidence / reasoning:

Suggested action:

---

Add additional findings as needed.

---

## 3. Claude Verification

For every material Codex finding classify:

- `CONFIRMED`
- `PLAUSIBLE`
- `REJECTED`
- `UNVERIFIED`

### Finding 1

Classification:

Verification performed:

Higher-authority evidence checked:

Conclusion:

---

## 4. Resulting Changes

Changes made because of CONFIRMED findings:

- ...

If none:

`None.`

---

## 5. Findings Not Adopted

### REJECTED

- ...

### PLAUSIBLE

- ...

### UNVERIFIED

- ...

---

## 6. Remaining Disagreement

If Claude and Codex still disagree:

Question:

Claude position:

Codex position:

Evidence on each side:

What remains unresolved:

If none:

`None.`

---

## 6A. Material Finding Traceability

One row per **material** finding, so Master and the user can trace it without reading the whole review.

| ID | Claim attacked | Evidence | Codex finding | Claude verification | Impact | Decision implication |
|---|---|---|---|---|---|---|
| F-1 | what Claude asserted | file:line · paper section · measurement | what Codex says is wrong with it | `CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED` | tasks / artifacts / stages | what *would* follow, if anything — or `None` |

**A Codex finding is not a decision.**

The `Decision implication` column states what a finding *would* imply. It does not enact anything. A material decision requires Master's integration review, a USER REVIEW BRIEF, and the **user's** approval (`WORKFLOW.md` §13A, §13B).

Codex must not fill a FINAL decision anywhere in this file. Neither may Claude, on the strength of a Codex finding.

Every row whose `Decision implication` is not `None` must reach the HANDOFF's `USER REVIEW INPUT` section.

---

## 7. Final Review Outcome

This outcome describes the **review**, not the project. `PASS` means the review found nothing outstanding — it does not accept, approve, or finalise anything.

Use one:

- `PASS`
- `PASS WITH FOLLOW-UP`
- `REWORK REQUIRED`
- `BLOCKED`
- `REVIEW UNAVAILABLE`

Reason:

---

## Review Rules

- Codex findings are not automatically accepted.
- **A Codex finding is a FINDING, never a DECISION** (`WORKFLOW.md` §13A). It reports what is true; it does not choose what is done about it.
- Codex must not record a FINAL decision in this file, and no `PASS` here accepts anything.
- Research-significant findings require evidence verification.
- Tests passing do not override primary-source mismatch.
- Codex failure/quota/auth errors are not PASS. `CODEX REVIEW UNAVAILABLE` must also be stated in the USER REVIEW BRIEF — whether to proceed without independent review is the user's call, not Master's.
- **Codex agreeing with Claude is not confirmation.** Both are models given the same brief. Convergence is weak evidence and never authority. Say so plainly rather than reporting agreement as corroboration.
- Review should remain scoped to the task unless an integration review was explicitly requested.
