---
name: paper-audit
description: "Performs a read-only, source-first audit of MetaFind paper claims, formulas, equations, sections, appendices, tables, figures, and research interpretations. Use when a claim must be verified against paper content authority. Do not use to implement fixes or to treat code, tests, README files, or prior agent statements as paper facts."
---

# Paper Audit

Perform a focused, read-only evidence audit. Determine what the relevant paper source actually states, what it leaves unspecified, and whether a proposed interpretation is supported.

Do not modify repository files, paper sources, metadata, audits, specifications, code, tests, configuration, or generated artifacts during this workflow.

## Independence requirement

Verify every material conclusion independently.

Do not accept a statement as true merely because it appears in:

- a prior Claude or Codex response;
- a hook result;
- a README;
- a code comment;
- an existing audit conclusion;
- a graph specification;
- a test name or passing test;
- the current implementation.

These may provide leads, but the supporting evidence must be inspected directly.

## Paper-content authority

Apply these roles without collapsing them into one authority level:

1. `docs/paper/metafind_source/**`, excluding `SOURCE_MANIFEST.json`, is the primary content authority for MetaFind claims.
2. Other `docs/paper/*_source/**` directories, excluding their `SOURCE_MANIFEST.json` files, are content authority for their respective upstream papers.
3. Upstream evidence applies to MetaFind only when the MetaFind source explicitly adopts or depends on that upstream method.
4. Official or pinned upstream code may establish `OBSERVED IMPLEMENTATION`; it does not establish what the MetaFind paper says.
5. `docs/audit/**` contains derived inventories, audits, mappings, and implementation contracts.
6. `docs/graph/**` contains derived project decisions and specifications.
7. Repository code, tests, runtime artifacts, README files, comments, handoffs, and conversations are lower-level evidence and must not overwrite paper content.

Use `SOURCE_MANIFEST.json` to identify provenance, include trees, and source integrity. It is regenerable metadata, not paper content, and does not itself prove a scientific claim.

Generated formula inventories may be used for navigation and exact-string checks. Confirm material formula claims in the included TeX source and surrounding context.

## Protected paths

- Treat `docs/paper/*_source/**` as protected paper content, except for `SOURCE_MANIFEST.json`, which is regenerable metadata rather than paper content. This audit is read-only, so do not regenerate or modify the manifest during the audit.
- Treat `docs/paper/*.gz` as protected.
- Never directly modify `metafind/vendor/**`. Any compatibility fix must be placed in `metafind/compat/` and handled outside this audit.
- Do not modify `CLAUDE.md` or `.claude/**` during normal research work.
- Claude hooks do not enforce Codex actions. Apply these restrictions as explicit stop conditions.

## Evidence classifications

Label every material statement with one of these classifications:

- `PAPER FACT`: explicitly stated in the relevant paper source.
- `UPSTREAM FACT`: explicitly stated in an upstream paper or its authoritative source.
- `OBSERVED IMPLEMENTATION`: directly observed in code, configuration, dependency behavior, or an executed path.
- `OBSERVED DATA`: directly measured from a dataset, checkpoint, manifest, log, or produced artifact.
- `INFERENCE`: a reasoned conclusion not explicitly stated by the source.
- `IMPLEMENTATION CHOICE`: a project decision where the paper leaves room or is silent.
- `DEVIATION`: a deliberate or observed difference from a supported paper requirement.
- `UNKNOWN`: evidence is absent, contradictory, inaccessible, or insufficient.

Do not relabel an inference or project decision as a paper fact because it appears reasonable or has been implemented.

Paper silence remains `UNKNOWN` until a project decision is made. After a decision, the selected behavior is an `IMPLEMENTATION CHOICE`, not a retroactive `PAPER FACT`.

## Audit procedure

### 1. Freeze the target

State the narrow audit target before drawing conclusions:

- claim or question;
- paper and source directory;
- relevant section, equation, appendix, table, figure, or algorithm;
- whether code comparison is in scope;
- what decision the audit is expected to inform.

Break compound claims into separately verifiable claims.

### 2. Locate the primary source

Inspect the applicable source manifest and included TeX files.

For each claim:

- record the exact source file;
- record the section, equation, table, figure, appendix, algorithm, or line location;
- read sufficient surrounding text to preserve definitions and qualifiers;
- follow relevant `\input` or `\include` relationships;
- inspect captions, footnotes, proof assumptions, and appendices when they change the meaning.

Do not rely on memory, converted summaries, search snippets, or a detached formula without context.

### 3. Build an evidence ledger

For every material claim, record:

| Field | Required content |
|---|---|
| Classification | One evidence class |
| Claim | The precise proposition being checked |
| Source | File or observed artifact |
| Location | Section, equation, table, figure, appendix, symbol, or code location |
| Evidence | Concise paraphrase or minimal excerpt |
| Limitation | Silence, ambiguity, contradiction, scope boundary, or missing evidence |

Keep conflicting evidence in separate rows.

### 4. Resolve source roles

Check whether the claim is:

- explicitly stated by MetaFind;
- inherited from an upstream method;
- merely similar to an upstream method;
- specified only by a project audit or graph decision;
- observed only in the implementation;
- inferred from multiple sources;
- not specified anywhere authoritative.

Resemblance is not inheritance. An upstream convention cannot fill MetaFind silence unless MetaFind explicitly adopts it.

If MetaFind and an upstream source differ, MetaFind controls the MetaFind reproduction requirement.

### 5. Audit contradictions and silence

For a contradiction:

- cite every conflicting passage independently;
- explain exactly what cannot simultaneously be true;
- identify whether the conflict changes shapes, update ordering, parameters, training, evaluation, or invariance;
- preserve each source statement as evidence;
- classify any selected resolution as `INFERENCE` or `IMPLEMENTATION CHOICE` unless the paper resolves it.

For paper silence:

- document what was searched;
- state what remains unspecified;
- identify downstream behavior that still needs a choice;
- do not fill the gap from code defaults, tests, README text, or convenience.

### 6. Audit formulas and algorithms

For each relevant formula or algorithm, verify:

- symbol definitions and domains;
- tensor shapes and dimensional closure;
- indexing and neighborhood definitions;
- norm versus squared norm;
- scalar versus vector outputs;
- aggregation rule and normalization;
- residual terms;
- update ordering and old-state versus new-state inputs;
- parameter sharing or independence;
- initialization assumptions;
- pooling and output semantics;
- frozen versus trainable components;
- loss direction, denominators, negatives, similarity, and temperature;
- transformation assumptions and whether the claimed invariant or equivariant output follows.

When code comparison is in scope, map paper symbols to concrete code symbols and call sites. Keep the paper result and implementation observation in separate evidence rows.

### 7. Audit experimental claims

When the target involves data, training, or evaluation, verify:

- dataset identity and scope;
- split unit and leakage boundary;
- preprocessing and modality construction;
- training stages and trainable components;
- sampling, masking, dropout, seeds, and negatives;
- checkpoint identity;
- gallery and query protocol;
- candidate pool;
- metric definition and aggregation;
- baseline protocol compatibility;
- evaluator and human-evaluation assumptions.

Numerical similarity alone does not establish paper fidelity.

### 8. Assess impact

Assign one impact level:

- `CRITICAL`: the issue can invalidate the scientific conclusion or core reproduction claim.
- `MATERIAL`: it can meaningfully change architecture, training, evaluation, or reported results.
- `MINOR`: it affects precision or documentation without changing the central conclusion.
- `NONE KNOWN`: no supported downstream effect was found.

State the reasoning and affected downstream components.

### 9. Stop conditions

Stop and request a user decision when:

- authoritative source passages conflict and the choice affects scientific behavior;
- the paper is silent and alternatives materially change results;
- a proposed resolution would create or remove a deviation;
- required source evidence is missing or inaccessible;
- continuing would require modifying protected authority or governance files;
- the audit target expands into implementation work.

Do not make the decision implicitly by selecting the easiest or currently implemented behavior.

## Required output

Return these sections:

1. `Audit Target`
2. `Evidence Ledger`
3. `Paper Findings`
4. `Conflicts`
5. `Unknowns`
6. `Current Implementation` when comparison was requested
7. `Reproduction Impact`
8. `Verdict`
9. `Required User Decision`

Use exactly one overall verdict:

- `VERIFIED`
- `PARTIALLY VERIFIED`
- `UNVERIFIED`
- `CONTRADICTED`
- `BLOCKED BY UNKNOWN`

If a section has no findings, state that explicitly. Do not implement fixes or modify files as part of the audit.
