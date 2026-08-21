# SPEC — <block> / <milestone>

> Written by the Block Owner after the start grill (`grilling` + `domain-modeling`), **before**
> implementation. Approved by the USER before execution begins.
>
> This is the single reference the 4-axis review, the Reviewer, Codex, and the USER acceptance
> all read from. Policy: `workflow/SKILLS.md` §6.
>
> Copy to `workflow/blocks/<BLOCK>/SPEC_<milestone>.md`.

**Block:** · **Milestone:** · **Owner:** · **Reviewer:** · **Date:** · **Baseline commit:**

---

## 1. OBJECTIVE

One paragraph. What is true after this milestone that is not true now.

## 2. SOURCE OF TRUTH

What decides correctness here, in authority order. Name the actual file, section, or dataset
field — not "the paper". If two sources conflict, record both and say the conflict is unresolved.

## 3. INPUTS

Every artifact read, with its expected count, its producer, and its acceptance state.

## 4. OUTPUTS

Every artifact written, with path, expected count, schema, and who consumes it downstream.

## 5. SCOPE

## 6. NON-SCOPE

What this milestone deliberately does not do. Anything appearing here that later shows up in the
diff is scope creep and Axis 2 must catch it.

## 7. PAPER / UPSTREAM AUTHORITY

Per research-significant behaviour: the behaviour, its authority, and the exact citation
(paper section / equation, upstream `file:line` at a recorded commit, dataset field, ledger entry).

| Behaviour | Authority | Citation | Class |
|---|---|---|---|

Class ∈ PAPER FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION · OBSERVED DATA · INFERENCE ·
IMPLEMENTATION CHOICE · DEVIATION · UNKNOWN.

## 8. IMPLEMENTATION CHOICES

Choices made because the source underspecifies. Each with its reason and what it would take to
revisit it. **Never write one of these as a PAPER FACT.**

## 9. KNOWN DEVIATIONS

Each deviation: expected behaviour · reproduced behaviour · reason · expected scientific impact ·
whether it affects comparability with the paper's reported results · its registry id in
`docs/graph/graph_spec.yaml`. **A deviation with no registry id is not recorded.**

## 10. UNKNOWN

What is not known, what was checked, and what would resolve it. Absence of evidence is not
evidence of absence.

## 11. SUCCESS CRITERIA

Checkable statements, each with the measurement that decides it and the population it is measured
over. "It works" is not a criterion.

## 12. FAILURE CONDITIONS

What makes this milestone a failure, including the **silent** ones — the outputs that would look
fine and be wrong.

## 13. SELF-VERIFICATION REQUIREMENTS

What the Owner will verify and how.

**Test seams** (adopted from `to-spec`): sketch them here before implementing. Prefer existing
seams. Use the highest seam available. Fewer is better; one is ideal. **Confirm the seams with the
USER before implementation.**

| Seam | Existing or new | What it tests | Expected-truth source |
|---|---|---|---|

**Expected-Truth Provenance Rule** (`SKILLS.md` §7): every test whose expected value encodes a
claim about the world must name where that value came from, and it must not be the implementation
under test. If a behaviour has no seam, say so here rather than inventing one.

## 14. INDEPENDENT REVIEW REQUIREMENTS

What the Reviewer must check independently, and **which checks must happen before** any expensive
run rather than after it.

## 15. MILESTONE CRITERIA

What must all be true for this to reach the USER Acceptance Grill:

- [ ] every SUCCESS CRITERION met, with evidence
- [ ] Owner self-verification complete
- [ ] Reviewer independent verification complete
- [ ] 4-axis code review complete, all four axes reported separately
- [ ] Codex milestone review complete, or `CODEX REVIEW UNAVAILABLE` stated
- [ ] every UNKNOWN either resolved or explicitly carried forward
- [ ] every DEVIATION registered
- [ ] experiment provenance recorded per `.claude/rules/experiments.md`

**Execution complete is not acceptance.** Only the USER's `APPROVE`, item by item, is.
