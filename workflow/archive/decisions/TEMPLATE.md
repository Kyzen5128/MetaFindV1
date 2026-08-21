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
- `RECOMMENDED` — returned to Master by D0
- `AWAITING_USER_REVIEW` — Master has recommended; the user has not yet decided
- `USER_APPROVED` — **FINAL ACCEPTED.** The only status that makes this project state
- `USER_REJECTED`
- `REWORK`
- `REJECTED`

Current:

`OPEN`

> `RECOMMENDED` and Master's endorsement are **not** acceptance. Only `USER_APPROVED` is.
> See `workflow/WORKFLOW.md` §13B.

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

## Finding vs Decision — read before filling this file

`WORKFLOW.md` §13A. These are different things and must never be merged into one statement.

| | |
|---|---|
| **FINDING** | What is **true**. An observation, contradiction, measurement, or defect. Has an evidence class and a verification status. True or false independently of what anyone wants to do about it |
| **DECISION** | What will be **done** about a finding. Adopt, reject, defer, deviate, correct. A choice, not an observation |

A confirmed finding does not imply its remedy. Keep them in separate sentences throughout this file, and especially in Sections 8, 11, 12, and 13.

D0 produces findings and **proposes** decisions. D0 does not make material decisions.

---

## 1. Question

State exactly what must be decided.

Separate the two halves explicitly:

```
FINDING UNDER EXAMINATION:  <what is or may be true>
DECISION REQUIRED:          <what must be chosen about it>
```

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

State the finding and the proposed remedy separately. A confirmed finding does not imply its remedy.

**FINDINGS established by this investigation**

| # | Finding | Evidence class | Verification |
|---|---|---|---|
| | | | CONFIRMED / PLAUSIBLE / UNVERIFIED |

**PROPOSED DECISION** — what D0 recommends doing about them

Recommendation:

Confidence:

Evidence classification of what would be adopted:

Reason:

Requires user approval: **YES** if material under `WORKFLOW.md` §13B / NO

> This is a proposal. D0 does not mark it accepted, and neither does Master.

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

Material under `WORKFLOW.md` §13B: YES / NO

If YES, this decision cannot become project state without a USER REVIEW BRIEF and the user's approval — regardless of how strongly D0, Codex, and Master agree.

**D0 stops here.** Sections 12, 13, and 14 are not D0's to fill.

---

## 12. Master Integration Recommendation

**Master fills this section. It is a recommendation, not an acceptance.**

Master reviews Sections 1–11, the repository state, and re-verifies the load-bearing claims rather than accepting them on assertion.

### Master's independent verification

| Claim | Source checked | Result |
|---|---|---|
| | | CONFIRMED / not reproduced / accepted on report |

State plainly which claims Master re-verified **directly** and which it accepted **on D0's report**.

### MASTER RECOMMENDATION

- `ACCEPT`
- `ACCEPT WITH FOLLOW-UP`
- `REWORK`
- `REJECT`
- `BLOCKED`

Recommendation:

Reason:

Date:

Affected tasks:

Required follow-up:

### Routing effect

`REWORK` / `REJECT` / `BLOCKED` may be acted on immediately — they return work to the owner and change no project state. Master still informs the user.

`ACCEPT` / `ACCEPT WITH FOLLOW-UP` **change project state** and therefore do not take effect here. They proceed to Section 13.

> Master must not write "accepted", update `MASTER.md` / `CONTEXT.md` / `INDEX.md`, unblock a downstream task, or mark anything DONE on the strength of this section alone.

Set Status to `AWAITING_USER_REVIEW` and add the entry to `workflow/DECISION_LEDGER.md`.

---

## 13. USER REVIEW BRIEF

**Master fills this section**, following `workflow/USER_REVIEW_TEMPLATE.md`.

**KEEP IT SHORT.** Do not restate Sections 1–11. The user should be able to act on this without opening anything else, while knowing where to look.

Required content:

1. What was found — material findings only
2. Evidence / provenance — paper / code / test / runtime / data / D0 / Codex, and what Master verified directly
3. Claude ↔ Codex disagreement — or `No material disagreement.`
4. Verified conclusion — CONFIRMED / PLAUSIBLE / REJECTED / UNVERIFIED
5. Proposed / implemented decisions — each with Decision, Authority, Classification
6. Impact — task / artifact / dataset / training / evaluation / dependency / downstream result
7. Remaining UNKNOWN — or `None known.`
8. USER ACTION REQUIRED — the specific question

Material items must appear here **even under full Claude + Codex + Master consensus**. Convergence between models is not independent confirmation.

If Codex review did not run, state `CODEX REVIEW UNAVAILABLE`. It is not a PASS.

---

## 14. USER Final Decision

**Only the user fills this section.**

D0 must not fill it. Master must not fill it. Codex must not fill it. Do not pre-populate it with an expected answer.

User action:

- `APPROVE`
- `REJECT`
- `MODIFY`
- `INVESTIGATE MORE`

Decision as decided by the user:

Date:

User's modifications, if any:

### Effect

| Action | Effect |
|---|---|
| `APPROVE` | Status becomes `USER_APPROVED`. **FINAL ACCEPTED.** Master integrates into global project state and records it in the Decision Ledger |
| `REJECT` | Status becomes `USER_REJECTED`. The finding stands; the proposed remedy does not |
| `MODIFY` | The **user's** version is the decision, not the proposal. Master re-verifies if the modification is substantial, then integrates |
| `INVESTIGATE MORE` | Status returns to `INVESTIGATING`. Master defines the additional investigation |

If the user's instruction and Master's recommendation differ, **the user's instruction governs.** Master may state a disagreement once, with evidence, then proceeds as instructed. Both are retained in the record — the disagreement is part of it.

Master's recommendation from Section 12 is never overwritten. It is kept alongside the user's decision.

---

## D0 Operating Rules

D0:

- investigates research, evidence, architecture, and cross-task decisions;
- does not execute unrelated implementation work;
- does not silently update project-wide accepted state;
- does not mark its own recommendation as accepted;
- does not fill Sections 12, 13, or 14;
- does not treat Master's endorsement, Codex agreement, or passing tests as acceptance;
- must return the result to Master;
- must use Codex adversarial review for formal decisions;
- must clearly distinguish paper fact, upstream-supported inference, implementation choice, runtime fact, and unresolved interpretation.

Master remains the final **integration** owner.

**The user remains the final research and project authority.** A material decision becomes project state only on the user's `APPROVE` (`WORKFLOW.md` §13B).