---
name: reproduction-audit
description: Perform a rigorous read-only audit of whether the current MetaFindV1 implementation, data pipeline, training procedure, or evaluation procedure corresponds to the evidence-backed MetaFind reproduction target.
argument-hint: "[component, pipeline stage, implementation claim, or reproduction scope]"
disable-model-invocation: true
---

# Reproduction Audit

Audit the requested MetaFindV1 reproduction scope:

$ARGUMENTS

The purpose of this skill is to determine how faithfully the current repository corresponds to the evidence-backed MetaFind specification.

This is a read-only reproduction audit.

Do not modify code, configuration, data, documentation, tests, checkpoints, caches, experiment outputs, or any other repository artifact during the audit.

Do not repair problems while auditing.

If a problem is found, record it and report it.

## 1. Define the Audit Scope

State exactly what is being audited.

The scope may be:

- one function
- one module
- one equation-to-code mapping
- one model component
- one dataset stage
- one preprocessing stage
- one training stage
- one evaluation stage
- one complete pipeline
- one reproduction claim

Do not silently expand the audit to the entire repository.

If the requested scope is ambiguous and different interpretations would materially change the result, ask the user before continuing.

## 2. Define the Reproduction Target First

Before judging the implementation, establish what behavior the authoritative evidence actually requires.

Use the source authority defined in CLAUDE.md.

Prioritize:

1. MetaFind original source or supplementary material
2. Published MetaFind paper
3. Original upstream papers or official implementations
4. Official project documentation
5. Current repository implementation
6. Reasoned inference

The repository must not be used as evidence for what the paper intended.

If the reproduction target itself is unresolved, record the unresolved item as UNKNOWN before auditing correspondence.

## 3. Use the Project Evidence Classes

Every research-relevant claim must remain classified as one of:

- PAPER FACT
- UPSTREAM FACT
- OBSERVED IMPLEMENTATION
- OBSERVED DATA
- INFERENCE
- IMPLEMENTATION CHOICE
- DEVIATION
- UNKNOWN

Do not collapse these categories.

In particular:

- existing code is not automatically PAPER FACT
- upstream behavior is not automatically a MetaFind requirement
- an IMPLEMENTATION CHOICE is not automatically a DEVIATION
- successful execution is not evidence of paper fidelity
- an UNKNOWN must not be silently resolved by current implementation behavior

## 4. Establish the Expected Behavior

For the audited component, determine the expected behavior from authoritative evidence.

Record when applicable:

- component purpose
- inputs
- outputs
- architecture
- equations
- variable semantics
- tensor semantics
- dimensions
- operation ordering
- graph construction
- feature construction
- preprocessing
- dataset source
- training behavior
- loss functions
- optimizer settings
- inference procedure
- evaluation procedure
- metrics
- frozen versus trainable components
- upstream inheritance
- MetaFind-specific modifications

If a required detail is not specified, mark it UNKNOWN.

Do not invent a target merely to make correspondence auditable.

## 5. Inspect the Current Implementation

Inspect the actual repository implementation within the requested scope.

Record when applicable:

- file path
- class
- function
- method
- configuration source
- relevant constants
- relevant command
- data flow
- tensor flow
- control flow
- model dependency
- external dependency
- checkpoint behavior
- relevant generated artifacts

Use direct repository evidence.

Do not rely only on file names, comments, documentation, or symbol names when the actual behavior can be inspected.

Comments may describe intended behavior but do not override executable behavior.

## 6. Build a Requirement-to-Implementation Matrix

For every material reproduction requirement, map:

- expected behavior
- evidence classification
- authoritative source
- implementation location
- observed implementation behavior
- correspondence status
- remaining uncertainty

Use one of the following correspondence statuses:

- VERIFIED
- PARTIALLY VERIFIED
- UNVERIFIED
- DEVIATED
- NOT IMPLEMENTED
- NOT APPLICABLE
- BLOCKED BY UNKNOWN

Do not assign VERIFIED unless the relevant implementation behavior was actually inspected and corresponds to an evidence-backed target.

## 7. Equation-to-Code Correspondence

When equations are relevant, explicitly map:

- mathematical variable -> code tensor or object
- mathematical operation -> code operation
- function -> module or callable
- coefficient -> literal, parameter, or configuration
- aggregation -> reduction operation
- normalization -> normalization operation
- index relationship -> graph or tensor indexing
- coordinate dependency -> implemented dependency
- trainable term -> trainable parameter
- frozen term -> frozen component
- equation order -> execution order

Check semantics, not only tensor shapes.

Matching dimensions without matching mathematical meaning is not equation fidelity.

Matching symbol names without matching operations is not verification.

## 8. Architecture Correspondence

When architecture is relevant, verify when applicable:

- required components exist
- components are connected correctly
- inputs correspond
- outputs correspond
- operation ordering corresponds
- number of layers corresponds when specified
- dimensions correspond when specified
- parameter sharing corresponds
- message passing corresponds
- aggregation corresponds
- normalization corresponds
- coordinate handling corresponds
- frozen/trainable status corresponds
- upstream components are inherited correctly
- MetaFind modifications are implemented correctly

If the paper leaves an architectural parameter unspecified, do not fail the implementation solely for choosing a value.

Instead classify the value as an IMPLEMENTATION CHOICE unless contrary evidence exists.

## 9. Data Pipeline Correspondence

For dataset and preprocessing stages, verify when applicable:

- dataset identity
- source
- version
- data root
- scene selection
- object selection
- filtering
- exclusions
- preprocessing sequence
- split construction
- annotation generation
- feature generation
- semantic edge generation
- point-cloud construction
- normalization
- caching
- sample counts
- generated metadata

Separate:

- paper-required data behavior
- implementation behavior
- currently observed data on disk

Existing data files are OBSERVED DATA.

Their existence alone does not prove they were generated using the correct procedure.

## 10. Training Correspondence

When training is within scope, verify when applicable:

- model initialization
- checkpoint initialization
- frozen parameters
- trainable parameters
- optimizer
- learning rate
- scheduler
- loss functions
- loss weights
- batch size
- epoch count
- step count
- sampling
- gradient handling
- random seeds
- model-selection procedure
- resume behavior

A training script that executes successfully is EXECUTABLE, not automatically PAPER-ALIGNED.

If the paper does not specify a parameter, record the implementation value as an IMPLEMENTATION CHOICE unless stronger evidence exists.

## 11. Evaluation Correspondence

When evaluation is within scope, verify when applicable:

- evaluation dataset
- split
- preprocessing
- query construction
- candidate construction
- inference procedure
- ranking
- filtering
- thresholds
- metric definition
- metric implementation
- averaging
- checkpoint selection
- number of samples

A matching metric name is insufficient.

A similar numerical result is insufficient.

Paper-result comparability requires protocol correspondence.

## 12. Upstream Correspondence

If MetaFind relies on an upstream method or official implementation:

1. determine exactly what MetaFind claims to inherit
2. determine what MetaFind claims to modify
3. inspect the relevant upstream source when necessary
4. compare the current implementation to the inherited requirement
5. do not import unrelated upstream defaults

Classify upstream-supported behavior as UPSTREAM FACT unless MetaFind itself explicitly specifies it.

If MetaFind modifies the upstream method, audit the modification separately.

## 13. Configuration Audit

Configuration is part of reproduction behavior.

Inspect relevant values such as:

- hidden dimensions
- edge dimensions
- feature dimensions
- layer counts
- seeds
- checkpoint paths
- model names
- encoder names
- batch sizes
- optimizer settings
- learning rates
- loss weights
- thresholds
- split definitions
- dataset paths

For every material configuration value, determine whether it is:

- explicitly paper-specified
- upstream-specified and inherited
- implementation-specific
- a compatibility choice
- a deviation
- unknown in the target specification

Do not present current configuration values as paper parameters without evidence.

## 14. Dependency and Environment Audit

When relevant, inspect whether compatibility changes preserve scientific behavior.

Relevant factors may include:

- Python version
- PyTorch version
- CUDA version
- GPU architecture
- library versions
- changed APIs
- deprecated operations
- replacement implementations
- local patches
- vendor patches

A compatibility modification is acceptable only if the scientific semantics remain equivalent for the audited scope.

If equivalence cannot be established, classify the issue as PARTIALLY VERIFIED, UNVERIFIED, or DEVIATED as appropriate.

## 15. Validation Evidence

Inspect existing validation relevant to the reproduction claim.

Examples include:

- import tests
- unit tests
- shape tests
- invariance tests
- equivariance tests
- deterministic tests
- dataset integrity checks
- forward-pass checks
- checkpoint loading checks
- paper-equation tests
- evaluation tests

For each relevant test, state exactly what it establishes.

Do not extrapolate beyond the test scope.

For example:

- an import test establishes importability
- an equivariance test establishes the tested equivariance property
- a dataset count establishes the observed count
- a shape check establishes shape correspondence
- none of these individually establishes complete reproduction fidelity

## 16. Detect Unsupported Assumptions

Search for implementation behavior that may depend on unsupported assumptions.

Examples:

- undocumented hyperparameters
- guessed dimensions
- guessed encoders
- inferred preprocessing
- library defaults
- silent fallbacks
- fabricated missing values
- zero-filling unspecified features
- upstream defaults automatically inherited
- assumptions embedded in comments
- assumptions embedded in configuration
- assumptions embedded in data generation scripts

Classify each according to the project evidence system.

If scientifically consequential, include it in the final audit even when the code currently works.

## 17. Detect Deviations

A DEVIATION exists when the implementation differs from an evidence-backed expected behavior.

For every identified deviation, record:

- expected behavior
- source
- implementation behavior
- implementation location
- reason if known
- scientific impact
- whether paper-result comparability is affected

Do not classify unspecified paper behavior as a deviation merely because the implementation made a choice.

That is usually an IMPLEMENTATION CHOICE unless contrary evidence exists.

## 18. Detect Missing Implementation

Check whether evidence-backed required behavior is absent.

Possible cases include:

- missing architecture component
- missing loss term
- missing feature
- missing preprocessing stage
- missing dataset filter
- missing evaluation step
- missing normalization
- missing frozen component
- missing checkpoint initialization
- missing metric behavior

Use NOT IMPLEMENTED only when authoritative evidence establishes that the behavior is required.

Do not convert paper ambiguity into a missing-feature finding.

## 19. Detect Dead or Unused Reproduction Code

When relevant, determine whether code believed to implement a paper component is actually used.

Check when necessary:

- import path
- call path
- configuration selection
- runtime branch
- instantiated class
- executed function
- generated artifact consumption

A correct implementation that is never invoked does not establish pipeline-level correspondence.

Distinguish:

- implemented
- reachable
- selected
- executed
- validated

## 20. Detect Placeholder or Degraded Modes

Look for scientifically relevant fallback behavior such as:

- placeholders
- mock data
- dummy values
- skipped stages
- reduced models
- debug branches
- degraded checkpoints
- missing-feature fallbacks
- optional dependency fallbacks
- partial dataset operation

If such behavior is active within the audited scope, record it explicitly.

Do not describe a degraded or debug pathway as the full reproduction pathway.

## 21. Data-to-Code Consistency

When generated data is part of the audited scope, verify whether the code currently expected to generate or consume the data is consistent with the observed artifact format.

Check when applicable:

- schema
- dimensions
- keys
- file names
- counts
- IDs
- ordering
- metadata
- checkpoint compatibility

Do not assume old generated artifacts correspond to the current code state.

If provenance cannot be established, classify it as UNKNOWN or UNVERIFIED.

## 22. Code-State Awareness

Determine whether repository state affects the audit.

Inspect when relevant:

- git branch
- commit
- uncommitted changes
- untracked research files
- local patches

Do not describe current behavior as belonging exactly to a commit if research-critical uncommitted changes affect the audited scope.

Do not modify git state during the audit.

## 23. No Repair During Audit

Do not:

- edit code
- create patches
- rewrite configuration
- regenerate data
- rerun preprocessing that changes artifacts
- retrain models
- overwrite checkpoints
- update documentation
- fix tests
- install or change dependencies
- delete files

Read-only commands and non-mutating inspections are allowed.

If execution of a test or command may create or modify persistent artifacts, do not run it as part of this audit without explicit user authorization.

Instead report what validation would be needed.

## 24. Safe Validation Boundary

Non-destructive validation may be used only when clearly safe and useful.

Before running a validation command, determine whether it can:

- modify repository files
- update caches
- write logs
- create checkpoints
- generate outputs
- mutate data
- install packages
- change environment state

If there is uncertainty about side effects, do not run it.

Record the proposed validation instead.

## 25. Impact Classification

For each unresolved issue, implementation choice, missing component, or deviation, classify reproduction impact as:

- CRITICAL
- MATERIAL
- MINOR
- NONE KNOWN

Use:

CRITICAL
when the issue can invalidate the core reproduced method or make major scientific claims incomparable.

MATERIAL
when the issue can meaningfully affect architecture, training, evaluation, or reported numerical results.

MINOR
when the difference is real but unlikely to change the main scientific interpretation.

NONE KNOWN
only when no meaningful reproduction impact is currently supported by evidence.

Do not use NONE KNOWN merely because the impact has not been measured.

## 26. Scope-Level Verdict

At the end of the audit, assign exactly one verdict to the requested scope:

- VERIFIED
- PARTIALLY VERIFIED
- UNVERIFIED
- DEVIATED
- NOT IMPLEMENTED
- BLOCKED BY UNKNOWN

Definitions:

VERIFIED
The relevant implementation was inspected and sufficient evidence establishes correspondence for the requested scope.

PARTIALLY VERIFIED
Some important correspondence is established, but material parts remain unverified or implementation-dependent.

UNVERIFIED
The implementation may exist, but available evidence or validation is insufficient to establish correspondence.

DEVIATED
The implementation materially differs from an evidence-backed requirement.

NOT IMPLEMENTED
An evidence-backed required behavior is absent.

BLOCKED BY UNKNOWN
The target specification itself is too uncertain to determine correspondence without a research decision.

Do not use VERIFIED merely because:

- code runs
- tests pass
- shapes match
- metrics look plausible
- a previous audit said PASS
- repository documentation claims correctness

## 27. Reproduction Coverage

When useful, summarize the audited requirements numerically.

For example:

- total material requirements inspected
- VERIFIED count
- PARTIALLY VERIFIED count
- UNVERIFIED count
- DEVIATED count
- NOT IMPLEMENTED count
- BLOCKED BY UNKNOWN count

This is a coverage summary only.

Do not convert the counts into a scientific confidence percentage unless a valid methodology for doing so exists.

## 28. Stop Conditions

Stop and ask the user before resolving an issue if:

- the reproduction target is ambiguous
- authoritative sources conflict
- multiple plausible implementations remain
- an UNKNOWN affects architecture
- an UNKNOWN affects training
- an UNKNOWN affects dataset construction
- an UNKNOWN affects evaluation
- a required comparison depends on missing provenance
- choosing one interpretation could materially alter reproduction fidelity

When stopping, report:

- what is known
- what is unknown
- evidence checked
- implementation behavior
- available interpretations
- impact
- exact user decision required

Do not silently choose a path.

## 29. Audit Output

Return the result using this structure.

### Reproduction Audit Scope

State exactly what was audited.

### Reproduction Target

State the evidence-backed expected behavior.

List unresolved target specification items separately.

### Requirement-to-Implementation Matrix

For each material requirement include:

- requirement
- classification
- source
- implementation location
- observed implementation
- correspondence status
- impact
- notes

### Validation Evidence

State:

- validation inspected
- what each validation establishes
- what remains unvalidated

Do not overstate test coverage.

### Implementation Choices

List material implementation choices that fill paper silence.

### Deviations

List all identified deviations or state that none were identified within the audited scope.

### Missing Implementation

List required behavior that is not implemented or state that none was identified.

### Unknowns

List every unresolved research-relevant item.

### Provenance Issues

Include when relevant:

- code state
- data provenance
- checkpoint provenance
- generated artifact provenance

### Reproduction Impact

Summarize CRITICAL, MATERIAL, and MINOR issues.

### Verdict

Use exactly one:

- VERIFIED
- PARTIALLY VERIFIED
- UNVERIFIED
- DEVIATED
- NOT IMPLEMENTED
- BLOCKED BY UNKNOWN

Scope the verdict only to the requested audit target.

### Required User Decision

Include only when a stop condition was reached.

Ask the minimum precise question required to proceed.

## 30. Audit Boundary

This skill determines reproduction status.

It does not authorize subsequent corrective action.

After identifying:

- bugs
- deviations
- missing components
- unsupported assumptions
- unknowns
- failed validation

report them and stop.

Do not modify the repository until the user explicitly requests the next action.
