---
name: impact-check
description: "Performs read-only, scoped impact triage for an evidence-backed inconsistency discovered during a formal MetaFindV1 D-task. Use when task assumptions, implementation, runtime behavior, artifacts, dependencies, or verification may be local, Master-impacting, a D0 candidate, or unsafe to continue. Does not modify global workflow state or continue implementation."
---

# Impact Check

Perform scoped, read-only impact triage for a finding discovered during a formal MetaFindV1 D-task.

Answer:

> Is this finding local to the current task, or must it be escalated to Master for project-level or D0 review?

The current D-task is responsible for detecting inconsistencies. It is not expected to already know every downstream consequence.

Do not continue the underlying implementation while performing this check.

## Preconditions

Identify an exact formal task:

`workflow/tasks/<task-id>/TASK.md`

If the task ID or corresponding `TASK.md` cannot be identified:

1. ask for the exact task ID;
2. do not invent or infer one from the current project assignment;
3. stop the impact check.

This skill does not create a task, activate a task, or change task status.

## Trigger conditions

Use this skill when a formal D-task observes evidence that:

- a `TASK.md` assumption does not match repository reality;
- implementation does not match the referenced specification;
- implementation appears inconsistent with primary evidence;
- runtime behavior contradicts the expected contract;
- an artifact has unexpected or unclear semantics;
- a declared dependency may be wrong;
- a correction may affect another task or stage;
- verification cannot establish the property it claims to verify;
- continuing may waste substantial compute;
- continuing may generate scientifically invalid artifacts;
- continuing may corrupt, overwrite, or contaminate important artifacts;
- the task owner cannot confidently determine whether the issue is local.

The task does not need to prove project-wide impact before invoking the skill.

A concrete inconsistency plus evidence is sufficient.

## Required context

Read the current repository state independently in this order:

1. the repo root `AGENTS.md`
2. `workflow/WORKFLOW.md`
3. `workflow/MASTER.md`
4. `workflow/CONTEXT.md`
5. `workflow/INDEX.md`
6. the current task's `workflow/tasks/<task-id>/TASK.md`
7. only the additional source, specification, implementation, test, runtime, artifact, or decision files needed to evaluate the finding

If an accepted D0 decision is relevant, read its formal file under `workflow/decisions/` and verify its acceptance status. Do not infer acceptance from a summary elsewhere.

Do not automatically read the entire repository or unrelated task directories.

Read another task's `TASK.md` only when that task is directly implicated and the shared workflow files do not establish the relevant input, output, dependency, or contract.

## Independent verification

Verify the finding against the current repository.

Do not accept a claim merely because it appears in:

- a prior Claude or Codex response;
- a conversation summary;
- a session handoff;
- an earlier impact-check report;
- a README;
- a code comment;
- a test name;
- `MASTER.md`, `CONTEXT.md`, or `INDEX.md`;
- an old workflow file.

These sources may provide navigation or project-state context, but material claims must be checked against the appropriate current evidence.

Current code and resolved configuration can establish `OBSERVED IMPLEMENTATION`. They cannot by themselves establish `PAPER FACT`, paper intent, specification correctness, or reproduction correctness.

In particular:

- prior Claude analysis is not evidence by itself;
- Codex is an independent reviewer, not scientific authority;
- workflow state is not paper authority;
- `OBSERVED IMPLEMENTATION` does not establish paper intent or paper fidelity;
- passing tests prove only their scoped assertions;
- historical workflow material is a lead only and must be re-verified.

Record what was checked, how it was checked, and what remains unverified.

## Scoped retrieval

Expand context only far enough to determine impact.

Prefer read-only repository navigation such as:

- searching definitions and call sites;
- tracing producers and consumers;
- inspecting configuration flow;
- checking artifact provenance;
- checking relevant task dependencies;
- reading the exact specification or authority source.

Any index, search result, generated graph, comment, or summary is navigation evidence only. Verify the conclusion against actual repository files, runtime artifacts, or primary sources.

Do not run expensive training, evaluation, downloads, destructive operations, or artifact regeneration during impact triage.

## Evidence classification

Use the project evidence taxonomy from `AGENTS.md`:

- `PAPER FACT`: explicitly stated in the relevant paper-content authority.
- `UPSTREAM FACT`: explicitly stated in an upstream source and applicable only where MetaFind inherits it.
- `OBSERVED IMPLEMENTATION`: directly observed in current code or resolved configuration.
- `OBSERVED DATA`: directly measured from current runtime behavior, data, logs, checkpoints, manifests, or artifacts.
- `INFERENCE`: a reasoned conclusion not explicitly stated by stronger evidence.
- `IMPLEMENTATION CHOICE`: a selected behavior where the paper leaves room or is silent.
- `DEVIATION`: a deliberate or observed difference from a supported paper requirement.
- `UNKNOWN`: evidence is absent, contradictory, inaccessible, stale, or insufficient.

When older workflow text uses legacy labels, interpret them as follows:

- `PRIMARY` maps to the applicable primary-source class, usually `PAPER FACT`.
- `IMPLEMENTATION FACT` maps to `OBSERVED IMPLEMENTATION`.
- `VERIFIED RUNTIME FACT` and `VERIFIED DATA FACT` map to `OBSERVED DATA`, with the verification method recorded.
- `UNKNOWN / UNVERIFIED` maps to `UNKNOWN`.

`ACCEPTED D0 DECISION` is governance status, not a substitute for evidence classification. Record it separately and only when the formal decision was accepted by Master. An accepted D0 decision defines project behavior but does not become a `PAPER FACT`.

Keep conflicting evidence in separate entries. Do not resolve a conflict merely by selecting the currently implemented behavior.

## Impact questions

Answer all of the following:

1. Does the finding remain entirely within the current `TASK.md` scope?
2. Is intended behavior already unambiguous from stronger authority or an accepted decision?
3. Would correcting it change scientific meaning, paper interpretation, architecture, or evaluation semantics?
4. Would it change another task's input, output, assumption, dependency, scope, verification requirement, or Definition of Done?
5. Would it affect shared artifact semantics, milestone validity, evaluation validity, or reproducibility claims?
6. Does resolution require research or architecture adjudication?
7. Could continuing produce invalid or uninterpretable results, waste substantial compute, corrupt artifacts, violate stronger authority, or cross an unauthorized boundary?
8. Can unaffected work continue safely without crossing the uncertain boundary?

## Classification

Return exactly one primary classification:

- `LOCAL`
- `MASTER-IMPACTING`
- `D0-CANDIDATE`
- `BLOCKER`

When more than one description applies, use this priority:

1. `BLOCKER`
2. `D0-CANDIDATE`
3. `MASTER-IMPACTING`
4. `LOCAL`

Record secondary characteristics in the reasoning, but do not return multiple primary classifications.

### LOCAL

Use `LOCAL` only when all of the following are true:

- the issue is entirely inside the current task's existing scope;
- intended behavior is unambiguous;
- no research interpretation changes;
- no project architecture changes;
- no other task contract or dependency changes;
- no shared artifact semantics change;
- the Definition of Done does not materially change;
- continuing under the current task contract is safe.

Recommended action:

The current engineer may correct the issue later within the existing task authority, verify it, and document it in `HANDOFF.md`.

The impact check itself does not implement the correction.

### MASTER-IMPACTING

Use `MASTER-IMPACTING` when the finding affects or may affect:

- another task;
- a cross-task dependency;
- shared artifact semantics;
- project architecture considered settled;
- a global runtime assumption;
- milestone feasibility or validity;
- evaluation validity;
- paper-fidelity assumptions;
- reproducibility claims;
- another task's scope or Definition of Done.

Recommended action:

Return a `MASTER-IMPACTING FINDING` to Master.

Do not update global workflow state locally.

### D0-CANDIDATE

Use `D0-CANDIDATE` when resolution requires research or architecture adjudication, including:

- conflicting primary evidence;
- ambiguous paper interpretation;
- ambiguous architecture;
- an unsupported scientific assumption;
- multiple technically valid implementations with different scientific meaning;
- a deliberate deviation choice;
- a cross-stage methodological choice.

Recommended action:

Return the finding to Master as a D0 candidate.

Only Master decides whether to create or activate a formal D0 decision. The current task and this skill must not appoint D0 directly.

### BLOCKER

Use `BLOCKER` when continuing before resolution may:

- generate scientifically invalid artifacts or results;
- waste substantial compute on a known-questionable configuration;
- corrupt, overwrite, or contaminate important artifacts;
- make downstream artifacts uninterpretable;
- violate stronger authority or an accepted project decision;
- cross an unauthorized research or architecture boundary;
- irreversibly commit an unresolved assumption into shared artifacts.

Recommended action:

Stop at the nearest safe point and report:

`TASK BLOCKER — MASTER REVIEW REQUIRED`

A blocker may also be project-impacting or require D0 review. Its primary classification remains `BLOCKER` because execution must stop first.

Do not continue merely to satisfy the original Definition of Done.

## Stop-safe behavior

While the impact check is active:

- pause the underlying implementation;
- do not begin another task or stage;
- do not launch expensive computation;
- do not regenerate or overwrite artifacts;
- do not make speculative corrective edits;
- preserve the observed evidence;
- identify the last safe state;
- distinguish work that is safe to retain from work that may be invalidated;
- state whether the current task may continue fully, only in unaffected areas, or not at all.

A safely blocked task is preferable to a completed invalid task.

## Required output

Return:

# IMPACT CHECK REPORT

Task ID:

Finding:

Observed Evidence:

Evidence Class:

Evidence Verification:

Relevant Governance or Accepted Decision Status:

Contradicted Assumption / Contract:

Primary Classification:
`LOCAL | MASTER-IMPACTING | D0-CANDIDATE | BLOCKER`

Secondary Characteristics:

Affected Components:

Potentially Affected Tasks:

Shared Artifacts or Interfaces Affected:

Can Current Task Continue Safely:
`YES | YES WITH RESTRICTIONS | NO`

Safe Continuation Boundary:

Reason:

Recommended Engineer Action:

Recommended Master Action:

D0 Needed:
`YES | NO | MASTER TO DECIDE`

Files / Evidence Checked:

Unverified Items:

## Escalation message

If the primary classification is not `LOCAL`, also return this compact block:

`MASTER-IMPACTING FINDING`

Finding:

Evidence:

Evidence class:

Evidence verification:

Affected task(s):

Affected artifact(s) or interface(s):

Current task impact:

Can current task continue safely:

Impact-check classification:

Recommended Master triage:

If the primary classification is `BLOCKER`, prepend:

`TASK BLOCKER — MASTER REVIEW REQUIRED`

Do not write the escalation into global project files. Return it to the current task owner for delivery to Master.

## Scope and authority boundaries

This skill performs triage only.

It must not:

- modify any repository file or runtime artifact;
- continue or repair the underlying implementation;
- modify `workflow/MASTER.md`;
- modify `workflow/CONTEXT.md`;
- modify `workflow/INDEX.md`;
- modify the current or another task's `TASK.md`;
- modify a decision file;
- create, activate, accept, reject, or close a D0 decision;
- change task status, dependencies, execution order, or parallel-safety status;
- change project architecture or scientific interpretation;
- mark a task, stage, gate, or milestone complete;
- silently fix a project-wide issue;
- treat Claude, Codex, workflow summaries, tests, or implementation as scientific authority.

Master owns project-level triage, integration, task contracts, dependencies, and global state.

D0 owns research or architecture investigation only when delegated by Master.

The user remains the execution-control authority.

## Core rule

The task owner's responsibility is:

> detect inconsistency and preserve evidence.

The task owner is not required to already know:

> every downstream consequence.

When impact is uncertain, use this skill and expand context only as needed.

When continuation may be unsafe:

> stop first, then escalate.
