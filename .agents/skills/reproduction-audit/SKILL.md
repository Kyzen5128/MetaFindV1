---
name: reproduction-audit
description: "Performs a read-only, evidence-driven audit of whether MetaFind code, configuration, data flow, tests, and validation artifacts reproduce the paper and project specification. Use for reproduction status, code-vs-paper analysis, deviations, missing implementation, runtime tracing, validation, and reproducibility review. Do not implement fixes during the audit."
---

# Reproduction Audit

Audit whether the current MetaFind system implements and validates the intended scientific reproduction.

The required reasoning chain is:

`paper → specification → implementation → validation`

Keep all four stages separate. Evidence from a later stage must not silently redefine an earlier stage.

This is a read-only workflow. Do not repair code, change configuration, regenerate authority files or metadata, weaken tests, alter gates, or modify repository state during the audit.

## Independence requirement

Independently inspect the evidence supporting every material conclusion.

A prior Claude or Codex statement, existing audit, README, graph document, implementation choice, test, or runtime result is not sufficient merely because it already exists. Determine its evidence class and verify the underlying source or behavior.

Claude hooks do not protect Codex actions. Treat protected-path rules as explicit stop conditions.

## Authority model

Use the following roles:

1. `docs/paper/metafind_source/**`, excluding `SOURCE_MANIFEST.json`, defines MetaFind paper content.
2. Other `docs/paper/*_source/**` directories, excluding their `SOURCE_MANIFEST.json` files, define their respective upstream papers.
3. Each `SOURCE_MANIFEST.json` is regenerable provenance and integrity metadata, not paper content and not evidence for a scientific claim by itself.
4. Upstream facts apply only where MetaFind explicitly inherits them.
5. `docs/audit/**` provides derived formula inventories, upstream mappings, contradiction records, implementation contracts, and consistency audits.
6. `docs/graph/**` provides derived project specifications, decisions, validation plans, and gate definitions.
7. The current repository provides `OBSERVED IMPLEMENTATION`.
8. Tests and executed checks provide scoped validation evidence.
9. Runtime data and experiment artifacts provide `OBSERVED DATA`.
10. README files, comments, handoffs, conversations, and memory provide context only.

Do not treat these sources as equal authority.

If a derived audit or graph specification conflicts with paper source, report the conflict. Do not silently follow the derived document.

## Protected paths

- Treat `docs/paper/*_source/**` as protected paper content, except for `SOURCE_MANIFEST.json`, which is regenerable metadata rather than paper content. This audit is read-only, so do not regenerate or modify the manifest during the audit.
- Treat `docs/paper/*.gz` as protected.
- Never directly modify `metafind/vendor/**`. Any compatibility fix must be placed in `metafind/compat/` and handled outside this audit.
- Do not modify `CLAUDE.md` or `.claude/**` during normal research work.
- Do not claim Codex was protected by the Claude research-authority hook.

## Evidence classifications

Use these classifications throughout:

- `PAPER FACT`
- `UPSTREAM FACT`
- `OBSERVED IMPLEMENTATION`
- `OBSERVED DATA`
- `INFERENCE`
- `IMPLEMENTATION CHOICE`
- `DEVIATION`
- `UNKNOWN`

A passing test is validation evidence for its explicit assertion, not a paper fact.

## Audit procedure

### 1. Define scope

State:

- reproduction claim or subsystem;
- applicable paper sections, equations, tables, or algorithms;
- implementation entry points;
- configuration and data scope;
- validation artifacts in scope;
- whether execution is authorized and safe.

Prefer a narrow, falsifiable audit target over a general statement that the repository “matches the paper.”

### 2. Establish the paper requirement

Apply the `paper-audit` methodology to the relevant claims.

For each requirement:

- cite the MetaFind source location;
- identify any explicit upstream inheritance;
- record contradictions and silence;
- distinguish mandatory behavior from optional examples;
- classify unresolved details as `UNKNOWN`;
- classify selected behavior for a silent detail as `IMPLEMENTATION CHOICE`.

Do not use current code to decide what the paper meant.

### 3. Establish the project specification

Inspect relevant material under `docs/audit/**` and `docs/graph/**`.

For each specification item, determine whether it is:

- a direct transcription of a paper requirement;
- an upstream requirement explicitly inherited by MetaFind;
- a resolution of a paper contradiction;
- an implementation choice for paper silence;
- a declared deviation;
- a validation or promotion rule;
- an unresolved unknown.

Verify important paper-backed assertions against the paper source. An existing decision record may establish the chosen project behavior, but it cannot turn that behavior into a paper fact.

Record specification conflicts instead of selecting one silently.

### 4. Trace the implementation

Trace the actual producer-to-consumer path:

`producer → persisted artifact → loader → transformation → consumer → model/loss/retrieval/evaluation`

Inspect:

- definitions and concrete call sites;
- constructors and resolved configuration;
- defaults and environment overrides;
- checkpoint loading and trainable parameters;
- tensor shapes and transformations;
- data identity, splits, filtering, and joins;
- query and gallery construction;
- loss construction and negative sets;
- retrieval candidate pools;
- metric computation and aggregation;
- fallback, degraded, cache, resume, and error paths;
- whether the relevant branch is reachable in the configured run.

A function that exists but is never called is not implemented behavior.

A configuration field that exists but is not consumed is not an implemented specification.

A test fixture or synthetic path is not evidence that the production path behaves identically.

### 5. Protect pinned and governance content

- Never directly modify `metafind/vendor/**`. Treat it as pinned upstream and verify integrity or usage read-only.
- Any future compatibility fix must be placed in `metafind/compat/`.
- Treat `docs/paper/*_source/**` as protected paper content except for `SOURCE_MANIFEST.json`, which is regenerable metadata and not paper content. This audit remains read-only and does not regenerate it.
- Treat `docs/paper/*.gz` as protected.
- Do not modify `CLAUDE.md` or `.claude/**` during this audit.
- Do not claim Codex was protected by the Claude research-authority hook.

### 6. Build the audit matrix

Create one row per independently testable requirement:

| Requirement | Paper evidence | Specification | Implementation | Validation | Status | Evidence class | Impact |
|---|---|---|---|---|---|---|---|

Use row statuses such as:

- `MATCHED`
- `PARTIAL`
- `DEVIATED`
- `NOT IMPLEMENTED`
- `UNREACHABLE`
- `UNKNOWN`

Do not mark `NOT IMPLEMENTED` until definitions, call sites, configuration branches, and alternative paths have been searched.

Do not mark `MATCHED` when only a symbol name or test name matches.

### 7. Audit formulas and architecture

For equation-to-code mappings, verify:

- symbol-to-variable mapping;
- tensor shapes;
- distance definition;
- neighborhood and directionality;
- message construction;
- aggregation and normalization;
- scalar or vector coordinate weights;
- residuals;
- old-state versus updated-state ordering;
- parameter and tower sharing;
- projection layers;
- initialization and pooling;
- invariance and equivariance assumptions;
- freezing and optimizer parameter groups.

Report paper contradictions separately from implementation deviations.

If the project selects one side of a paper contradiction, preserve the selected behavior as an `INFERENCE` or `IMPLEMENTATION CHOICE` unless stronger evidence exists.

### 8. Audit data, training, retrieval, and evaluation

Verify:

- source dataset and revision;
- manifest-derived counts rather than convenient constants;
- object-level versus scene-level split units;
- leakage boundaries;
- preprocessing and modality provenance;
- annotation model, prompt, revision, and cache identity;
- sampling units and positive mappings;
- masking and dropout granularity;
- trainable and frozen components by stage;
- optimizer, scheduler, batch size, epochs, temperature, and seed;
- query/gallery encoder identity;
- checkpoint and gallery-index identity;
- candidate pool and retrieval protocol;
- evaluation inputs, metrics, judges, aggregation, and baseline comparability.

Do not claim reproduction from similar final numbers alone. Reproduction requires protocol alignment and traceable evidence.

### 9. Audit validation

Distinguish:

- static inspection;
- an existing but unexecuted test;
- an executed test result;
- runtime instrumentation;
- artifact-level evidence;
- an explicitly recorded gate result.

When safe and within authorization, run only narrow, non-destructive checks. Do not launch expensive downloads, training, large evaluations, publication, or destructive commands without explicit authorization.

Record the exact command, scope, environment, result, and limitations of any executed validation.

A test proves only what its assertion observes. In particular, verify whether it proves that:

- the intended production node executed;
- the expected branch was reached;
- the real artifact was consumed;
- the scientific property, rather than a mock or shape check, was tested.

### 10. Preserve MANY TESTS, FEW GATES

Apply the project principle exactly:

`MANY TESTS, FEW GATES`

- Keep broad L1/L2 checks for local correctness, seams, diagnostics, and failure detection.
- Recognize a gate only when the specification explicitly designates it as a gate.
- Do not promote every test to a gate.
- Do not weaken gate criteria to accommodate a failing implementation.
- Do not treat a missing gate record as a pass.
- Preserve gate return-code and evidence semantics.
- Required Audits must run and be recorded, whatever their outcome, but they do not block.
- A paper contradiction must not become a gate merely because it is important; doing so can incentivize weakening the audit criterion.
- Validation count is not evidence coverage. Map each test and gate to the exact claim it controls.

### 11. Audit deviations and degraded behavior

For each difference between paper/specification and implementation, determine whether it is:

- a supported alternative;
- an `IMPLEMENTATION CHOICE`;
- a declared `DEVIATION`;
- an accidental divergence;
- a fallback or degraded mode;
- dead or unreachable code;
- `UNKNOWN`.

For every deviation, report:

- evidence for the expected behavior;
- observed behavior;
- reason, if recorded;
- affected stages, tables, metrics, or claims;
- whether the deviation was visible in outputs;
- whether it invalidates comparison with the paper.

Do not conceal a deviation because it is necessary for current hardware, dependencies, cost, or test stability.

### 12. Audit reproducibility provenance

For a meaningful experiment or scientific result, check for:

- code revision and dirty-worktree state;
- resolved configuration and hashes;
- exact command;
- random seeds;
- dataset and split identity;
- checkpoint identity;
- dependency and environment versions;
- hardware;
- metrics and aggregation;
- output and log locations;
- degraded flags and known deviations.

Missing provenance limits the verdict even when the reported numbers look plausible.

### 13. Assess impact

Use:

- `CRITICAL`: invalidates or can invert a core scientific conclusion.
- `MATERIAL`: can materially change architecture, training, evaluation, or reported results.
- `MINOR`: affects precision, maintainability, or documentation without changing the central result.
- `NONE KNOWN`: no supported downstream impact was identified.

Trace the impact through downstream consumers rather than guessing from the local change.

### 14. Stop conditions

Stop and request an explicit user decision when:

- paper sources conflict on scientifically material behavior;
- paper silence permits alternatives that materially change results;
- specification records conflict;
- a new or removed deviation is required;
- validation would require weakening a gate or audit;
- required evidence is missing;
- continuing would require modifying protected sources or pinned vendor files;
- expensive or irreversible execution is needed;
- the request would transition from audit into implementation.

Do not repair a discrepancy merely to make tests, imports, shapes, training, or metrics pass.

## Required output

Return:

1. `Audit Scope`
2. `Authority and Evidence Ledger`
3. `Paper → Specification → Implementation → Validation Matrix`
4. `Runtime Trace`
5. `Validation Evidence`
6. `Implementation Choices`
7. `Deviations`
8. `Missing or Unreachable Implementation`
9. `Unknowns and Conflicts`
10. `Reproducibility Provenance`
11. `Reproduction Impact`
12. `Verdict`
13. `Required User Decision`

Use exactly one overall verdict:

- `VERIFIED`
- `PARTIALLY VERIFIED`
- `UNVERIFIED`
- `DEVIATED`
- `NOT IMPLEMENTED`
- `BLOCKED BY UNKNOWN`

State explicitly when a section has no findings. Do not implement fixes as part of this audit.
