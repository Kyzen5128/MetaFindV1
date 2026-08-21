# REVIEW — ESSGNN

**Reviewer:** unassigned · **Mode:** independent, read-only, **synchronous with the engineer**

## What the reviewer is for

Not a second engineer. Attacks the engineer's **contract**, not only its output:

1. Is the contract the engineer defined itself wrong?
2. Was an upstream source-of-truth missed?
3. Do the generated artifacts actually match the source data?
4. Could a schema PASS still be semantically wrong?
5. Could all tests PASS and the science still be wrong?
6. Is there silent corruption?
7. Are the block's work items consistent with each other?
8. Could this block's output contaminate downstream?
9. Did the engineer state an INFERENCE as a FACT?
10. Which failure modes do the engineer's tests not cover?

**Review early.** Before any long GPU job, corpus generation, full training, or expensive
evaluation, the reviewer must already have audited the sources, the contract, a real sample,
and the semantic consistency. A review that starts after the run is a post-mortem.

## Boundaries

Read-only by default. To execute a check, use a read-only command, an isolated output
directory, or a separate git worktree — never the engineer's production files.

**The reviewer may not decide a material remedy.** Findings go to Master via `HANDOFF.md`;
only the USER makes anything FINAL.

## Skills the Reviewer uses

Policy: `workflow/SKILLS.md` §10. Skills are method tools, never authority.

| Skill | Use it for | Claude may invoke |
|---|---|---|
| `mattpocock-skills:research` | source-of-truth and contract audit against primary sources | yes |
| `mattpocock-skills:diagnosing-bugs` | contradictions, conflicting measurements, suspected silent failure | yes |
| `mattpocock-skills:code-review` | an independent 4-axis pass, separate from the Owner's | yes |
| `improve-codebase-architecture` | milestone-only architecture survey | **no — ask the USER to run `/improve-codebase-architecture`** |

### Four axes, reported separately

`STANDARDS` · `SPEC` · `SOURCE / EVIDENCE` · `SCIENTIFIC / SEMANTIC`.
Axis 4 assumes the code runs, the tests pass and the SPEC is met, then asks how the result could
**still** be scientifically wrong: wrong units, coordinate or frame mismatch, generated artifact
disagreeing with its source, label noise, silent corruption, downstream contamination, evaluation
leakage. Never merge the four into one verdict.

### Differential testing is the sharpest tool here

Compare two things that should agree — official upstream artifact vs ours, source metadata vs
generated annotation, before vs after, configuration A vs B. Build a red-capable loop, reproduce,
minimise, form **falsifiable** hypotheses, instrument, then fix. **Do not read the code and guess
the cause first.** This is how the 180 degree yaw was found, and how it was then shown not to move
the embedding (`workflow/blocks/ULIP2/evidence/n03_n04_upstream_verification.md`).

### Review early, not at the end

Before full annotation, corpus generation, n06 full encode, any multi-hour GPU run, full training,
or full evaluation, the sources, the contract, a real sample and the semantic consistency must
**already** have been audited. A review that starts after the run is a post-mortem.

---

## Finding format

```
FINDING          what is true
EVIDENCE         file:line / paper section / measurement + the population it was measured over
CLASSIFICATION   PAPER FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION · OBSERVED DATA ·
                 INFERENCE · IMPLEMENTATION CHOICE · DEVIATION · UNKNOWN
IMPACT           tasks, artifacts, stages
SEVERITY         BLOCKER · MAJOR · MINOR · NOTE
```

A material finding must carry real evidence. "Looks wrong" is not a finding.

---

## Findings

_None yet._
