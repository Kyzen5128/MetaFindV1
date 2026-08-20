# D0 Decision Template

> D0 is the Research / Architecture Lead.
> It handles questions that Master should not resolve from intuition alone.
> A D0 decision must be evidence-backed, independently reviewed, and accepted by Master before becoming project state.

---

## Decision ID

`D0-XXX_short-name`

---

## Status

Use only:

- `OPEN`
- `INVESTIGATING`
- `REVIEW`
- `RECOMMENDED`
- `ACCEPTED`
- `REWORK`
- `REJECTED`

Current:

`OPEN`

---

## Session Continuity

If this formal D0 decision is resumed in a new conversation and:

`workflow/decisions/<decision-id>_SESSION_HANDOFF.md`

exists, read it after the formal workflow state and this decision file.

The session handoff is temporary working memory only.

It must not override:

- primary evidence;
- accepted project state;
- this formal decision record;
- verified repository / runtime evidence.

When the investigation is complete, record the formal result in this decision
file and return it to Master.

Do not use the session handoff as the completed decision artifact.

---

## 1. Question

State exactly what must be decided.

Do not combine unrelated questions.

---

## 2. Why This Decision Exists

Explain:

- what triggered the question;
- which task or milestone it affects;
- what will remain blocked until it is resolved.

---

## 3. Decision Scope

### In Scope

- ...

### Explicit Non-Scope

- ...

---

## 4. Authority / Evidence

### Primary Evidence

List the highest-authority evidence first.

For each item record:

- source;
- exact location;
- what it supports;
- evidence classification.

### Supporting Evidence

- upstream papers;
- official implementations;
- verified runtime observations;
- other relevant evidence.

### Conflicting Evidence

Record contradictions explicitly.

Do not silently reconcile conflicting sources.

---

## 5. Current Repository State

Describe only what is relevant to this decision:

- current implementation behavior;
- current config / protocol;
- current tests;
- current audit interpretation;
- known runtime behavior.

Current implementation is evidence about repository state, not proof of paper intent.

---

## 6. Options

### Option A

Description:

Evidence supporting:

Evidence against:

Consequences:

Risks:

---

### Option B

Description:

Evidence supporting:

Evidence against:

Consequences:

Risks:

---

Add more options only when genuinely necessary.

---

## 7. Analysis

Compare the options against:

- primary evidence;
- reproducibility;
- implementation impact;
- downstream dependencies;
- scientific validity;
- uncertainty.

Do not choose based on model confidence alone.

---

## 8. Recommended Decision

Recommendation:

Confidence:

Evidence classification:

Reason:

---

## 9. Codex Adversarial Review

Codex must independently challenge:

- evidence interpretation;
- hidden assumptions;
- missing alternatives;
- downstream consequences;
- whether the recommendation is actually supported.

Codex is reviewer, not authority.

---

## 10. Claude Verification of Codex Findings

Classify each material finding as:

- `CONFIRMED`
- `PLAUSIBLE`
- `REJECTED`
- `UNVERIFIED`

Only verified findings should change a research-significant recommendation.

---

## 11. Final Recommendation to Master

Decision proposed:

Why:

Remaining uncertainty:

Required repository changes if accepted:

Tasks affected:

---

## 12. Master Resolution

Master fills this section after review.

Resolution:

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`

Accepted decision:

Date:

Affected tasks:

Required follow-up:

---

## D0 Operating Rules

D0:

- investigates research, evidence, architecture, and cross-task decisions;
- does not execute unrelated implementation work;
- does not silently update project-wide accepted state;
- does not mark its own recommendation as accepted;
- must return the result to Master;
- must use Codex adversarial review for formal decisions;
- must clearly distinguish paper fact, upstream-supported inference, implementation choice, runtime fact, and unresolved interpretation.

Master remains the final integration owner.