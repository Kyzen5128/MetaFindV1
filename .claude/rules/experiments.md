# Experiments

Experiments in this repository must be reproducible, traceable, and scientifically interpretable.

An experiment is not only a command execution. It is a controlled comparison between a defined configuration, defined data, defined code state, and observed results.

## 1. Define the Experiment Before Running

Before running a research-relevant experiment, identify when applicable:

- experiment objective
- hypothesis or question
- code version or git state
- dataset and split
- preprocessing state
- model configuration
- checkpoint
- training configuration
- evaluation configuration
- random seed
- hardware
- software environment
- expected outputs
- comparison baseline
- success or failure criteria

Do not launch an expensive or research-critical experiment when the experimental condition itself is ambiguous.

If an unresolved choice may materially affect the scientific interpretation, stop and ask the user.

## 2. One Question Per Experiment

Prefer experiments that answer a clearly defined question.

Avoid changing multiple independent research variables simultaneously unless the experiment is intentionally designed to study their combined effect.

When comparing two results, identify exactly what changed between them.

If multiple uncontrolled differences exist, do not attribute the result to a single factor without evidence.

## 3. Baseline Discipline

A comparison requires an appropriate baseline.

Before claiming improvement, regression, or equivalence, verify that the compared runs are aligned in relevant conditions such as:

- dataset
- split
- preprocessing
- model initialization
- checkpoint
- training budget
- evaluation procedure
- candidate pool
- metrics
- hardware-sensitive behavior when relevant

Do not compare results from materially different protocols as though they were directly comparable.

## 4. Randomness and Seeds

Randomness is part of experimental state.

When randomness can affect the result, record the relevant seed or seeds.

Relevant sources may include:

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- dataloader workers
- sampling procedures
- dataset shuffling
- augmentation
- model initialization

Do not claim deterministic reproduction merely because a single seed was set.

If deterministic behavior depends on backend or library settings, record those conditions.

## 5. Data State

Record the exact data state used by an experiment when practical.

This may include:

- dataset identity
- dataset version
- data root
- split definition
- file counts
- preprocessing version
- generated annotations
- embeddings
- caches
- filtered samples
- excluded samples

Do not silently regenerate or replace experiment inputs between comparable runs.

If data state changes, treat the resulting experiment as a different condition.

## 6. Configuration Capture

Record research-relevant configuration values rather than relying on memory or implicit defaults.

Important configuration may include:

- model dimensions
- number of layers
- feature definitions
- optimizer
- learning rate
- scheduler
- loss weights
- batch size
- gradient accumulation
- number of epochs or steps
- sampling settings
- thresholds
- checkpoint path
- evaluation settings

Library defaults must not be treated as paper-specified parameters unless supported by evidence.

## 7. Code State

Results must be associated with the code state that produced them.

When practical, record:

- git commit
- git branch
- uncommitted changes
- relevant modified files
- local patches

If the repository contains uncommitted research-critical changes, do not describe the experiment as corresponding exactly to a known commit without noting those changes.

## 8. Environment State

Record environment information when it may affect reproducibility.

This may include:

- Python version
- PyTorch version
- CUDA version
- GPU model
- relevant library versions
- environment variables
- deterministic backend settings

Do not assume results from different hardware or software stacks are scientifically identical when numerical or implementation differences may matter.

## 9. Experiment Commands

Preserve the command used to execute a research-relevant experiment when practical.

Include relevant command-line arguments and environment variables.

Avoid reporting only a paraphrased description when the exact command is available.

Do not modify the command after the run and present the modified version as the executed command.

## 10. Output and Artifact Provenance

Experiment outputs must remain traceable to the run that produced them.

Relevant artifacts may include:

- checkpoints
- logs
- metrics
- predictions
- embeddings
- visualizations
- generated annotations
- cached features
- evaluation reports

Do not overwrite prior experiment outputs when they are needed for comparison or provenance.

Use distinct output locations or identifiers when multiple runs must be preserved.

## 11. Checkpoints

For checkpoint-based experiments, record when applicable:

- checkpoint path
- checkpoint origin
- training stage
- epoch or step
- model configuration
- whether loading was strict or partial
- missing keys
- unexpected keys

Do not assume a checkpoint is compatible solely because loading succeeds.

Partial loading or ignored parameters must be reported when scientifically relevant.

## 12. Metrics

Metric values must be interpreted together with their definitions and evaluation protocol.

Record when applicable:

- metric name
- metric definition
- aggregation method
- dataset split
- filtering
- thresholds
- ranking procedure
- candidate set
- number of evaluated samples

Do not claim agreement with paper results based only on a similar numeric value when the evaluation protocol is not verified.

## 13. Repeated Runs

A single run may be insufficient for conclusions affected by stochastic variation.

When scientifically relevant, use repeated runs or multiple seeds.

Report when applicable:

- number of runs
- individual results
- mean
- variance or standard deviation
- best result
- selection rule

Do not report the best run alone as representative unless that selection procedure is part of the defined protocol.

## 14. Failed Experiments

Failed experiments are evidence and must not be silently discarded when they are relevant.

Record:

- command
- configuration
- failure point
- error
- whether the failure is reproducible
- suspected cause
- confirmed cause if established

Distinguish between:

- environment failure
- data failure
- implementation failure
- numerical failure
- specification uncertainty

Do not repeatedly change research parameters merely to obtain a successful run.

## 15. Debug Runs vs Scientific Runs

Keep debugging runs separate from scientific experiments.

A debug run may use:

- reduced data
- fewer epochs
- fewer samples
- altered batch size
- temporary configuration
- synthetic input
- shortened evaluation

Do not use debug-run results as scientific evidence unless the changed conditions are explicitly relevant to the claim.

Label debug results clearly.

## 16. Sanity Checks

Before expensive training or evaluation, prefer inexpensive sanity checks when they can detect obvious problems.

Examples include:

- import checks
- one batch loading
- tensor shape inspection
- small forward pass
- loss finiteness
- gradient existence
- checkpoint loading
- small-sample evaluation
- deterministic repeat check

Passing sanity checks allows progression to larger experiments but does not establish scientific correctness.

## 17. Resource Use

Do not start expensive training, full-dataset preprocessing, or large evaluation runs without understanding their expected purpose and outputs.

Before a costly operation, identify when relevant:

- expected duration
- GPU usage
- memory usage
- disk usage
- output path
- whether existing artifacts may be overwritten

Do not delete existing experiment artifacts to free space without explicit authorization.

## 18. Result Interpretation

Separate observation from interpretation.

Examples:

- "validation recall@10 = 0.42" is OBSERVED DATA.
- "this is higher than run A under the same protocol" is a comparison supported only if the protocols are aligned.
- "the architectural change caused the improvement" is an inference unless the experimental design isolates that cause.

Do not convert correlation into causal explanation without adequate experimental control.

## 19. Comparison with Paper Results

Before comparing reproduced results with the MetaFind paper, verify when possible:

- same dataset
- same split
- same preprocessing
- same model setting
- same evaluation protocol
- same metric definition
- same candidate construction
- comparable training condition

If any required condition is unknown, state the limitation.

Numerical similarity alone does not establish faithful reproduction.

## 20. Experiment Reporting

For a research-relevant experiment, report concisely when applicable:

- experiment objective
- command
- configuration
- data
- checkpoint
- code state
- environment
- seed
- result
- validation
- deviations
- failures
- unresolved uncertainty

Do not report only the final metric while omitting conditions necessary to interpret it.

## 21. Experiment Completion Gate

Do not mark an experiment complete until:

- the intended run actually finished or its failure is documented
- relevant outputs were located
- metrics were extracted correctly
- experiment conditions are traceable
- unexpected behavior is reported
- comparison claims are supported by aligned protocols
- unresolved uncertainties are not hidden

A completed process is not automatically a completed scientific experiment.
