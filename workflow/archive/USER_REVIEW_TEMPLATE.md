# USER REVIEW BRIEF — Template

> Master writes this after integration review, for every result carrying a **material decision** (`WORKFLOW.md` §13B).
>
> **KEEP IT SHORT.** This is a decision aid, not a summary of the investigation. The decision file and the HANDOFF hold the detail; this points at them.
>
> Write `None` for empty sections. Do not pad.
>
> Save as `workflow/tasks/<task-id>/USER_REVIEW.md` or `workflow/decisions/<decision-id>_USER_REVIEW.md`, or deliver inline to the user. Either way, record the user's action in the Decision Ledger.

---

# USER REVIEW BRIEF

**Task / Decision ID:**
**Master Recommendation:** `ACCEPT` / `ACCEPT WITH FOLLOW-UP` / `REWORK` / `REJECT` / `BLOCKED`

---

## 1. What was found

Material findings only. State what is **true**, not what to do about it — remedies belong in §5.

- ...

---

## 2. Evidence / provenance

For each finding, name the actual source: **paper / code / test / runtime / data / D0 / Codex**.

Also mark which items Master **re-verified directly** and which it **accepted on the task's report**.

| Finding | Source | Location | Master verified? |
|---|---|---|---|
| | | | direct / on report |

---

## 3. Claude ↔ Codex disagreement

Material disagreement only. State both positions.

Report it even when Master believes Codex was wrong.

If none:

`No material disagreement.`

> Convergence between models is **not** independent confirmation. If all agreed, say so plainly — do not present it as corroboration.

---

## 4. Verified conclusion

```
CONFIRMED:   ...
PLAUSIBLE:   ...
REJECTED:    ...   (state why)
UNVERIFIED:  ...
```

If Codex review did not run, state `CODEX REVIEW UNAVAILABLE` here. **It is not a PASS**, and proceeding without it is the user's call.

---

## 5. Proposed / implemented decisions

One row per decision. **Anything already implemented but not yet ratified by the user must appear here** — do not hide it in the HANDOFF.

| # | Decision | Authority | Classification |
|---|---|---|---|
| 1 | | who it belongs to | PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION / OBSERVED DATA / INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / USER DECISION / MASTER RECOMMENDATION / UNKNOWN |

Never present an INFERENCE or IMPLEMENTATION CHOICE as a PAPER FACT.

---

## 6. Impact

Affected: **task / artifact / dataset / training / evaluation / dependency / downstream result**

- What becomes unblocked:
- What becomes blocked:

---

## 7. Remaining UNKNOWN / unresolved

State whether each one matters for *this* decision.

If none:

`None known.`

---

## 8. USER ACTION REQUIRED

State the specific question as a question.

- `APPROVE` — adopt as proposed
- `REJECT` — do not adopt; the finding stands, the remedy does not
- `MODIFY` — adopt with your changes
- `INVESTIGATE MORE` — insufficient evidence to decide

---

## Rules for writing this brief

- **Do not** restate the full HANDOFF.
- **Do not** substitute "tests PASS" for a research summary.
- **Do not** hide a decision because every AI agreed on it — consensus is exactly when the user most needs to see it.
- **Do not** hide a material choice that is already implemented but not yet ratified.
- **Do not** let Master's recommendation read as though it were already accepted.
- If Master's own earlier statement was corrected during review, say so.
- If the recommendation rests on an unverified number, say so **here**, not only in the decision file.

---

## Material reporting rule

Any addition, modification, confirmation, or reversal of the following is **material** and must appear in this brief, even under full Claude + Codex + Master consensus:

paper interpretation · architecture · implementation choice · deviation · dataset semantics · annotation semantics · preprocessing · training protocol · evaluation protocol · shared artifact semantics · cache validity · checkpoint validity · dependency ordering · scientifically meaningful assumption · anything that can materially change reproduction results

When in doubt, treat it as material.
