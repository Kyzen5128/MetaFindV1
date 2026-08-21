# HANDOFF — ESSGNN

> **Binding USER rule:** everything goes through this file. In-conversation reporting does not count.
> Write here **before** calling Master, not after.
> Append newest at the top. Never delete an entry — mark it `RESOLVED` and say by whom.

## When to write here

- You need Master's permission, a contract change, or a ruling.
- You are stuck and it touches the other block or the Integrator.
- You found something that changes shared architecture, a dependency, or an accepted assumption
  (`MASTER-IMPACTING FINDING`) — report it, do not act on it.
- You finished a work item and it needs integration review.

## Entry format

```
### <date> · <FROM role> → <TO role> · <BLOCKING | INFO | FINDING>

FINDING     what is true, with evidence (file:line, paper section, measurement, population)
DECISION    what you propose — kept separate from the finding, never merged
EVIDENCE    how it was verified, and what remains unverified
IMPACT      which tasks, artifacts, stages
ASK         exactly what you need from the recipient
STATE       can this block safely continue meanwhile? yes / no, and why
```

Classify every claim: PAPER FACT · UPSTREAM FACT · OBSERVED IMPLEMENTATION · OBSERVED DATA ·
INFERENCE · IMPLEMENTATION CHOICE · DEVIATION · UNKNOWN. Never promote an inference to a fact.

---

_No entries yet._
