# Paper Reproduction

This project aims to reproduce MetaFind as faithfully as the available evidence permits.

The target is not merely functional similarity. The target is evidence-backed correspondence between the published method and the reproduced implementation.

## 1. Define the Reproduction Target

Before implementing or changing a research-critical component, determine what the target paper actually specifies.

For each component, identify when available:

- purpose
- inputs
- outputs
- architecture
- equations
- tensor or feature semantics
- preprocessing
- dataset source
- dataset split
- training objective
- optimizer
- learning rate
- scheduler
- batch size
- number of epochs
- random seed
- initialization
- checkpoint
- inference procedure
- evaluation protocol
- metrics
- ablations
- implementation-specific details

Do not assume that unspecified items have conventional or obvious values.

## 2. Paper Specification vs Implementation

Keep the paper specification separate from the current repository implementation.

For a research-critical component, reason using this structure when applicable:

1. What the paper explicitly requires.
2. What supplementary or author-provided material adds.
3. What relevant upstream work specifies.
4. What the current repository actually does.
5. Whether the repository behavior is supported by the paper.
6. What remains unknown.
7. What implementation choices or deviations exist.

Repository behavior must not retroactively become evidence for what the paper intended.

## 3. Paper Silence

If the target paper does not specify a detail, classify it as UNKNOWN unless another authoritative source resolves it.

Do not infer a value merely because:

- it is common practice
- an upstream repository uses it
- another related paper uses it
- the current implementation uses it
- a library defaults to it
- it makes the code run
- it produces plausible results

Any adopted value not established by authoritative evidence must remain labeled as an IMPLEMENTATION CHOICE or INFERENCE as appropriate.

## 4. Use of Upstream Work

MetaFind may depend on methods, models, datasets, or implementations from upstream work.

Do not automatically copy all upstream behavior into MetaFind.

An upstream detail may be treated as MetaFind-relevant only when there is evidence that MetaFind adopts, inherits, references, or depends on that detail.

Distinguish:

- what the upstream method defines
- what MetaFind explicitly adopts
- what MetaFind modifies
- what MetaFind leaves unspecified
- what the reproduction chooses to inherit

If MetaFind modifies an upstream component, do not assume unchanged upstream behavior for the modified portion without evidence.

## 5. Equations and Architecture

For equations and architecture descriptions:

- preserve variable meaning
- preserve operation order
- preserve dependency relationships
- preserve dimensional meaning when specified
- preserve normalization and aggregation behavior when specified
- preserve trainable versus frozen components when specified

Do not replace a stated operation with a merely similar operation without recording a DEVIATION.

If implementation behavior is claimed to match an equation, verify the correspondence explicitly rather than relying on naming similarity.

## 6. Dataset and Preprocessing Fidelity

Dataset construction and preprocessing are part of the reproduced method.

Do not treat them as incidental implementation details.

When available, track:

- dataset identity and version
- scene or sample selection
- filtering
- train/validation/test split
- preprocessing sequence
- feature generation
- annotations
- external generated data
- caching
- normalization
- augmentation
- exclusions

If the reproduced dataset differs from the paper's dataset or cannot be verified as identical, record the difference explicitly.

## 7. Training Fidelity

Do not claim training fidelity based only on successful optimization or decreasing loss.

Training reproduction requires evidence for relevant choices such as:

- objective functions
- loss weights
- optimizer
- learning rate
- scheduling
- batch construction
- epoch or step count
- parameter freezing
- initialization
- checkpoint loading
- gradient handling
- random seeds
- sampling procedures

If a required training detail is missing and different plausible choices may change the result, stop and ask the user before selecting one.

## 8. Evaluation Fidelity

Do not compare reproduced results to paper results unless the evaluation protocols are meaningfully aligned.

Verify when applicable:

- evaluation dataset
- split
- preprocessing
- query construction
- candidate pool
- metric definition
- ranking procedure
- averaging method
- filtering
- thresholds
- checkpoint selection
- inference settings

A matching metric name does not guarantee a matching evaluation protocol.

## 9. Deviations

Any intentional or unavoidable difference from the evidence-backed reproduction target must be recorded as DEVIATION.

For each important deviation, record:

- original expected behavior
- reproduced behavior
- reason for the deviation
- evidence supporting the comparison
- expected scientific impact
- whether the deviation affects comparability with reported paper results

Do not hide deviations behind refactoring, compatibility fixes, hardware constraints, or library changes.

## 10. Validation Standard

Validation should test correspondence to the research claim being made.

Examples:

- import success validates importability
- unit tests validate the tested behavior
- equivariance tests validate the tested equivariance property
- dataset counts validate the counted dataset state
- matching tensor shapes validate shape correspondence
- matching equations and verified operations support method correspondence
- matching evaluation protocol supports result comparability

No single successful test establishes full paper reproduction.

## 11. Reproduction Status

Use precise status language.

A component may be described as:

- IMPLEMENTED
- EXECUTABLE
- UNIT-TESTED
- BEHAVIOR-VERIFIED
- PAPER-ALIGNED
- DEVIATED
- PARTIALLY VERIFIED
- UNVERIFIED
- BLOCKED BY UNKNOWN

Use PAPER-ALIGNED only when sufficient evidence establishes correspondence between the implementation and the paper specification for the relevant scope.

Do not use a stronger status than the evidence supports.

## 12. Completion Gate

A reproduction task is not complete merely because the implementation executes.

Before calling a research-critical component complete, determine:

- what evidence supports it
- what was tested
- what remains unverified
- whether any implementation choices remain
- whether any deviations remain
- whether any unresolved unknown could change the scientific result

If an unresolved uncertainty could materially affect reproduction fidelity, do not close the issue as fully reproduced.
