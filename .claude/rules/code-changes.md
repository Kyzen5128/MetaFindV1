# Code Changes

Code changes in this repository must preserve research traceability and minimize unintended changes to scientific behavior.

A successful edit is not defined only by whether the code runs. The edit must also preserve or explicitly document its effect on reproduction fidelity.

## 1. Understand Before Editing

Before modifying research-relevant code, determine:

- what component is being changed
- why the change is required
- what behavior currently exists
- what behavior is expected
- what evidence defines the expected behavior
- whether the change affects paper reproduction
- which files are actually necessary to modify

Do not edit research-critical behavior before understanding its role in the MetaFind pipeline.

If expected behavior is uncertain and the choice may affect research results, stop and ask the user.

## 2. Minimal Change Principle

Make the smallest change that correctly addresses the requested task.

Do not:

- refactor unrelated code
- rename unrelated symbols
- reformat unrelated files
- reorganize directories without need
- replace working components merely because another design looks cleaner
- introduce abstractions unrelated to the task
- change defaults that are outside the requested scope
- modify multiple research components when one targeted change is sufficient

Keep the causal relationship between the requested change and the resulting diff easy to inspect.

## 3. Preserve Research Semantics

Do not change scientific behavior as a side effect of maintenance work.

Research-relevant semantics include, when applicable:

- model architecture
- layer ordering
- tensor operations
- graph construction
- feature definitions
- coordinate handling
- message passing
- aggregation
- normalization
- loss functions
- loss weights
- sampling
- preprocessing
- dataset filtering
- training procedure
- inference logic
- ranking
- evaluation metrics
- thresholds

If any of these must change, explicitly identify the change before treating it as a routine code fix.

## 4. Paper-Critical Changes

For changes affecting paper reproduction, establish the evidence basis before implementation.

Classify the intended behavior using the project evidence categories.

A code change may implement:

- PAPER FACT
- UPSTREAM FACT
- IMPLEMENTATION CHOICE
- DEVIATION

Do not implement an unresolved UNKNOWN as though it were a paper requirement.

If multiple plausible implementations exist and the choice can materially affect reproduction fidelity, stop and ask the user before choosing one.

## 5. No Silent Behavioral Repair

Do not silently change research behavior merely to make:

- tests pass
- imports succeed
- training start
- tensor shapes align
- metrics improve
- outputs look plausible
- external libraries stop raising errors

First determine whether the failure reveals:

- an environment issue
- an implementation bug
- a data issue
- an unsupported assumption
- a reproduction mismatch
- an unresolved specification

Fix the actual cause when evidence supports it.

## 6. Existing Behavior Is Not Automatically Correct

Do not assume existing repository code is correct simply because it:

- already exists
- executes successfully
- has tests
- produces reasonable output
- was previously accepted
- resembles an upstream implementation

Existing code is OBSERVED IMPLEMENTATION unless stronger evidence supports it.

When changing existing behavior, distinguish between correcting an implementation error and introducing a new implementation choice.

## 7. Dependency and API Changes

Be cautious when modifying:

- package versions
- CUDA behavior
- PyTorch behavior
- third-party APIs
- model libraries
- dataset libraries
- serialization formats
- checkpoint loading
- external repositories
- vendor code

Do not change dependency versions merely to bypass a failure without understanding the compatibility impact.

When adapting code to a newer dependency or hardware environment, preserve the original scientific semantics whenever possible.

If exact behavior cannot be preserved, record the difference as a possible DEVIATION.

## 8. Vendor and Upstream Code

Avoid modifying vendored or upstream code unless necessary.

Prefer local adapters, wrappers, or narrowly scoped compatibility changes when they preserve upstream behavior more clearly.

If upstream code must be modified:

- identify the original source
- explain why modification is required
- keep the patch minimal
- preserve the original behavior where possible
- record any behavioral deviation

Do not present locally patched upstream code as the original upstream implementation.

## 9. Data Safety

Treat research data and generated artifacts as valuable state.

Do not delete, overwrite, move, regenerate, or mutate datasets, checkpoints, embeddings, caches, experiment outputs, or generated annotations unless the user explicitly authorizes the action or it is clearly required by the requested task.

Before destructive operations, identify:

- target path
- expected contents
- scope of deletion or overwrite
- whether the data is reproducible
- whether a backup or regeneration path exists

Do not use broad destructive commands when a narrower operation is sufficient.

## 10. Configuration Changes

Configuration is part of scientific behavior.

Do not silently change:

- paths
- seeds
- hyperparameters
- batch sizes
- model dimensions
- feature dimensions
- split definitions
- checkpoint paths
- thresholds
- optimizer settings
- scheduler settings
- evaluation settings

If a configuration value differs from the paper or cannot be verified, retain the correct evidence classification.

## 11. Validation After Changes

Validation must be proportional to the scope of the change.

Prefer targeted validation first.

Depending on the edit, validation may include:

- syntax or import checks
- targeted unit tests
- tensor shape checks
- deterministic behavior checks
- equivariance tests
- dataset integrity checks
- small forward passes
- checkpoint loading
- targeted evaluation
- comparison against known outputs

Do not automatically run expensive full training or evaluation when a targeted test is sufficient.

Do not treat a passing test as evidence for claims outside the scope of that test.

## 12. Failure Handling

If validation fails:

1. report the failing command or test
2. identify the observed failure
3. determine whether the failure was introduced by the change
4. distinguish environment, data, implementation, and research-specification causes
5. avoid stacking speculative fixes

Do not repeatedly modify research logic until a test happens to pass.

If the correct fix is uncertain and scientifically consequential, stop and ask the user.

## 13. Diff Discipline

Keep diffs inspectable.

After a meaningful code change, verify which files changed.

Unexpected modified files must be investigated before completion.

Do not include unrelated generated files, caches, temporary files, editor state, or environment artifacts in a research change unless explicitly required.

## 14. Comments and Documentation

Do not write comments that state uncertain research claims as facts.

Comments describing paper behavior must be supported by evidence.

When useful, distinguish clearly between:

- paper-required behavior
- upstream-required behavior
- implementation-specific behavior
- compatibility workaround
- intentional deviation

Do not use documentation language to hide uncertainty.

## 15. Reporting a Code Change

For research-relevant modifications, report concisely:

- files changed
- purpose of the change
- important behavior changed
- evidence basis
- validation performed
- validation result
- remaining uncertainty
- implementation choices or deviations

Do not claim that a change is paper-faithful unless that correspondence was actually established.

## 16. Completion Gate

Do not mark a code-change task complete until:

- the requested change is implemented
- unrelated files were not modified
- relevant validation was performed
- failures are resolved or explicitly reported
- research-critical assumptions are identified
- unresolved unknowns are not hidden
- deviations are documented
- the claimed level of correctness matches the available evidence

If the code works but reproduction correctness remains unresolved, report that distinction explicitly.
