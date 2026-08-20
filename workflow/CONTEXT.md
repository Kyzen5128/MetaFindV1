# MetaFindV1 Shared Project Context

> Compressed project context for D0 / D1 / D2 / D3 task conversations.
> This file provides orientation, not detailed task instructions.
> Task-specific authority, scope, evidence, and Definition of Done belong in each task's TASK.md.

---

## 1. Project Objective

TO BE INITIALIZED

---

## 2. System / Research Pipeline

TO BE INITIALIZED

---

## 3. Current Architecture

TO BE INITIALIZED

---

## 4. Authority Hierarchy

All research-significant work must follow the project authority hierarchy defined in:

`/home/kyzen/MetaFindV1/CLAUDE.md`

General direction:

Primary paper / source
→ verified evidence
→ interpretation / audit
→ implementation contract
→ graph / specification
→ implementation
→ tests
→ observed runtime / data
→ reasoned inference

Lower-authority artifacts must not redefine higher-authority evidence.

---

## 5. Stable Decisions

TO BE INITIALIZED

Only decisions already accepted by Master should appear here.

Do not put unresolved alternatives here.

Formal research / architecture decisions are stored in:

`workflow/decisions/`

---

## 6. Current Project State

TO BE INITIALIZED

Keep this section concise.

Include only project state that materially affects multiple tasks.

Detailed task status belongs in:

`workflow/INDEX.md`

---

## 7. Cross-Task Dependencies

TO BE INITIALIZED

Only record dependencies that multiple task owners need to understand.

Task-local dependencies belong in TASK.md.

---

## 8. Global Constraints

Current global rules include:

- Do not treat tests passing as proof of paper fidelity.
- Do not infer paper requirements from the current implementation.
- Do not silently replace missing evidence with assumptions.
- Mark unsupported or unresolved research claims explicitly.
- Codex is an independent reviewer, not scientific authority.
- Research-significant Codex findings must be verified by Claude against stronger evidence before adoption.
- A task must not expand into another task's scope without Master approval.
- A task must not start the next stage on its own.

---

## 9. Runtime / Environment Facts

TO BE INITIALIZED

Only include stable environment facts needed across multiple tasks.

Do not copy transient logs here.

---

## 10. Shared File Map

### Project control

- `CLAUDE.md` — project research / engineering rules
- `workflow/MASTER.md` — Master global control state
- `workflow/CONTEXT.md` — compressed shared context
- `workflow/INDEX.md` — task registry
- `workflow/decisions/` — accepted D0 decisions
- `workflow/tasks/` — execution task workspaces

### Research evidence

- `docs/paper/`
- `docs/audit/`
- `docs/graph/`

### Implementation

- `metafind/`
- `tests/`
- `tools/`
- `setup/`

---

## 11. Task Conversation Startup Rule

A new D-task conversation should normally read:

1. `/home/kyzen/MetaFindV1/CLAUDE.md`
2. applicable `.claude/rules/`
3. `/home/kyzen/MetaFindV1/workflow/CONTEXT.md`
4. its own `workflow/tasks/<task>/TASK.md`
5. only the additional evidence / implementation files explicitly referenced by TASK.md

Do not automatically re-read the entire repository.

Do not automatically read other task folders.

If the task discovers information that changes shared architecture, dependencies, or an accepted Master assumption, report:

`MASTER-IMPACTING FINDING`

The task should not rewrite global project state on its own.

---

## 12. Context Maintenance Rule

Master owns this file.

Update CONTEXT.md only when shared project understanding changes, such as:

- an architecture decision is accepted;
- a major dependency changes;
- a project-wide constraint changes;
- a stable runtime/environment fact changes;
- a milestone changes the shared current state.

Do not update it for routine debugging, minor implementation edits, individual test failures, or temporary experiment results.
