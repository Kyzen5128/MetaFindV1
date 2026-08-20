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

## 7. Final Review Outcome

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
- Research-significant findings require evidence verification.
- Tests passing do not override primary-source mismatch.
- Codex failure/quota/auth errors are not PASS.
- Review should remain scoped to the task unless an integration review was explicitly requested.
