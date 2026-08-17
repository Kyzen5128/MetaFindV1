---
name: paper-audit
description: Perform a rigorous evidence-based audit of a MetaFind paper claim, component, equation, architecture detail, dataset procedure, training setting, or evaluation setting before it is treated as established reproduction evidence.
argument-hint: "[claim, component, section, equation, or research question]"
disable-model-invocation: true
---

# Paper Audit

Audit the requested MetaFind research target:

$ARGUMENTS

The purpose of this skill is to determine exactly what the available authoritative evidence establishes before implementation or scientific claims are made.

This is an evidence audit, not an implementation task.

Do not modify code, configuration, data, documentation, or experiment artifacts while performing this audit unless the user separately requests such modification.

## 1. Define the Audit Target

First state the exact question being audited.

Make the target narrow enough that evidence can resolve it.

Examples:

- What architecture does MetaFind specify for the layout encoder?
- How is semantic edge embedding e_ij constructed?
- Is a component frozen or trainable?
- What dataset split is used?
- What optimizer and learning rate are specified?
- Does the current implementation correspond to Equation X?
- Is a repository behavior supported by the paper?
- Is a claimed reproduction detail actually unknown?

If the user's target is ambiguous and different interpretations would materially change the audit, ask the user before continuing.

## 2. Establish Source Scope

Identify the authoritative sources relevant to the target.

Follow the project source authority defined in CLAUDE.md.

Prioritize:

1. MetaFind original source or supplementary material
2. Published MetaFind paper
3. Original upstream papers or official implementations
4. Official project documentation
5. Current repository implementation
6. Reasoned inference

Secondary sources may be used only to locate primary sources.

Do not use blogs, Reddit, forum posts, AI summaries, or unsourced explanations as research authority.

## 3. Separate Source Roles

Do not merge different source roles.

For every relevant finding, determine whether it is:

- PAPER FACT
- UPSTREAM FACT
- OBSERVED IMPLEMENTATION
- OBSERVED DATA
- INFERENCE
- IMPLEMENTATION CHOICE
- DEVIATION
- UNKNOWN

A repository observation is not automatically a paper fact.

An upstream implementation detail is not automatically a MetaFind requirement.

An inference is not evidence that the authors used that design.

## 4. Read the Relevant Primary Evidence

Inspect the actual relevant source material rather than relying on memory or summaries.

For a paper claim, inspect the relevant combination of:

- main text
- equations
- figure captions
- tables
- appendix
- supplementary material
- implementation notes
- referenced upstream work when required

Do not stop at the first matching keyword if surrounding context can change the interpretation.

For an equation, inspect definitions of its variables and any preceding or following equations required to interpret it.

For a table or figure, inspect its caption and surrounding explanation.

For an upstream dependency, determine exactly what MetaFind claims to inherit or modify before importing upstream details.

## 5. Build an Evidence Ledger

For each material claim, record:

- claim
- classification
- source
- exact location when available
- what the source establishes
- what the source does not establish

Prefer page, section, equation, table, figure, appendix, file, function, or line references when available.

Do not cite a broad source when a more precise location is available.

## 6. Check for Paper Silence

Explicitly test whether the authoritative sources specify the detail being audited.

If they do not, classify the detail as UNKNOWN unless a stronger authoritative source resolves it.

Do not fill paper silence using:

- conventional defaults
- library defaults
- current repository behavior
- upstream defaults
- related papers
- plausible engineering practice
- values that make the code execute
- values that produce reasonable results

These may become candidate IMPLEMENTATION CHOICE items, but not paper facts.

## 7. Check for Source Conflicts

Compare relevant sources for contradictions.

Potential conflicts include:

- main paper vs appendix
- paper vs supplementary material
- MetaFind vs upstream paper
- MetaFind text vs MetaFind equations
- MetaFind description vs released source
- paper specification vs current repository implementation

If a conflict exists:

1. identify each conflicting statement
2. identify its source
3. apply the source authority from CLAUDE.md
4. explain whether the conflict can be resolved
5. explain the reproduction impact

Do not silently select one interpretation.

If the conflict remains unresolved and affects research-critical behavior, stop and ask the user.

## 8. Audit Current Implementation Only When Relevant

If the question concerns reproduction correspondence, inspect the current repository implementation separately.

Record:

- file
- symbol or function
- observed behavior
- relevant parameters
- relevant tensor or data flow
- whether the behavior is supported by authoritative evidence

Classify repository findings as OBSERVED IMPLEMENTATION unless stronger evidence establishes another category.

Do not infer paper intent from implementation structure.

## 9. Equation-to-Code Audit

When auditing correspondence between an equation and code, explicitly map:

- equation variable -> code object or tensor
- mathematical operation -> implementation operation
- aggregation -> implemented reduction
- normalization -> implemented normalization
- ordering -> execution ordering
- constants or coefficients -> configuration or literal values
- trainable quantities -> parameters
- frozen quantities -> non-trainable components
- dimensional semantics -> tensor dimensions when specified

Then classify each mapping as:

- VERIFIED
- PARTIALLY VERIFIED
- UNVERIFIED
- DEVIATED
- UNKNOWN

Naming similarity alone is not verification.

## 10. Architecture Audit

For architecture claims, check when applicable:

- component existence
- component purpose
- inputs
- outputs
- feature semantics
- dimensions
- number of layers
- connectivity
- message passing
- aggregation
- coordinate use
- normalization
- parameter sharing
- frozen versus trainable status
- upstream inheritance
- MetaFind-specific modifications

Do not infer unspecified architecture parameters from typical implementations.

## 11. Dataset Audit

For dataset-related claims, check when applicable:

- dataset identity
- source
- version
- scene or sample selection
- filtering
- preprocessing
- annotation generation
- train/validation/test split
- external generated data
- normalization
- augmentation
- exclusions
- sample counts

Distinguish paper-defined dataset behavior from the data currently present on disk.

Matching file counts alone does not establish dataset equivalence.

## 12. Training Audit

For training claims, check when applicable:

- loss functions
- loss weights
- optimizer
- learning rate
- scheduler
- batch size
- epoch or step count
- initialization
- parameter freezing
- checkpoint initialization
- sampling
- gradient handling
- random seed
- model selection procedure

Any unresolved parameter that can materially affect the result must remain UNKNOWN.

Do not select a plausible value during the audit.

## 13. Evaluation Audit

For evaluation claims, check when applicable:

- evaluation dataset
- split
- preprocessing
- query construction
- candidate pool
- inference procedure
- metric definition
- ranking logic
- thresholds
- filtering
- averaging
- checkpoint selection

A matching metric name or similar numerical result does not establish protocol equivalence.

## 14. Determine Reproduction Impact

For every unresolved or conflicting item, determine whether it can affect:

- architecture fidelity
- training behavior
- dataset equivalence
- evaluation comparability
- numerical results
- scientific conclusions

Use one of:

- CRITICAL
- MATERIAL
- MINOR
- NONE KNOWN

Do not assign NONE KNOWN merely because the impact has not yet been tested.

## 15. Stop Conditions

Stop and ask the user before making a research decision if:

- authoritative sources conflict and cannot resolve the issue
- multiple plausible implementations remain
- a required architecture detail is missing
- a training-critical parameter is missing
- dataset provenance is unresolved
- evaluation protocol is ambiguous
- adopting one interpretation could materially affect reproduction fidelity

When stopping, report:

- what is established
- what is unknown
- evidence checked
- available interpretations
- likely impact
- exact decision required from the user

Do not continue by guessing.

## 16. Audit Output

Return the audit using this structure.

### Audit Target

State the exact research question.

### Evidence Summary

For each material finding provide:

- classification
- claim
- source
- location
- evidence
- limitation

### Current Implementation

Include only when relevant.

State what the repository actually does without presenting it as paper intent.

### Conflicts

List conflicts or state that none were identified within the checked evidence.

### Unknowns

List every unresolved research-relevant detail.

Do not omit unknowns because they are inconvenient.

### Reproduction Impact

State the impact of each conflict, unknown, implementation choice, or deviation.

### Verdict

Use one of:

- VERIFIED
- PARTIALLY VERIFIED
- UNVERIFIED
- CONTRADICTED
- BLOCKED BY UNKNOWN

Scope the verdict narrowly to the audited claim.

Never use VERIFIED to mean that the entire MetaFind reproduction is correct.

### Required User Decision

Include this section only if a stop condition was reached.

Ask the minimum precise question required to continue.

## 17. Audit Boundary

The audit establishes evidence status only.

It does not authorize:

- implementation choices
- code modifications
- dataset changes
- training runs
- evaluation runs
- destructive operations

After the audit, wait for the user's instruction before taking a research-critical action that depends on unresolved evidence.
