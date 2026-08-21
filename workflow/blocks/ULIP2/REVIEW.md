# REVIEW — ULIP2

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
