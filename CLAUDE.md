# MetaFindV1 Research Instructions

MetaFindV1 is a scientific paper reproduction project.

The primary objective is research correctness, reproducibility, and traceability.
Working code alone is not sufficient evidence of correctness.

## 1. Core Rule

Accuracy takes priority over speed.

Never fabricate, silently guess, or fill in missing research-critical information.

If information is unknown, ambiguous, contradictory, or insufficiently supported and the choice may affect:

- architecture
- dataset construction
- preprocessing
- training
- evaluation
- metrics
- reproduction fidelity
- scientific conclusions

STOP and ask the user before making the decision.

Do not silently choose between multiple plausible interpretations.

## 2. Evidence Classification

For every important technical conclusion, distinguish between:

- PAPER FACT: explicitly stated by the target paper.
- UPSTREAM FACT: explicitly supported by an upstream paper or official implementation.
- OBSERVED IMPLEMENTATION: confirmed from the current repository code.
- OBSERVED DATA: confirmed from actual project outputs or datasets.
- INFERENCE: logically inferred but not explicitly stated.
- IMPLEMENTATION CHOICE: selected because the source is underspecified.
- DEVIATION: intentionally differs from the source.
- UNKNOWN: available evidence is insufficient.

Never present an inference, implementation choice, deviation, or unknown as a paper fact.

## 3. Source Authority

For MetaFind reproduction, prefer sources in this order:

1. Original MetaFind source / supplementary material
2. Published MetaFind paper
3. Original upstream papers and official upstream implementations
4. Official project documentation
5. Current repository implementation
6. Reasoned inference

A lower-priority source must not silently override a higher-priority source.

If authoritative sources contradict each other:
- identify both sources
- show the conflicting definitions
- explain the implementation impact
- do not resolve the contradiction silently

## 4. Research Verification

Before claiming what a paper does:

- read the relevant original section
- inspect equations
- inspect appendix/supplementary material
- inspect figures and captions when relevant
- inspect referenced upstream work when inherited behavior matters

Do not rely only on:
- README files
- summaries
- comments
- previous AI-generated notes
- memory

Whenever possible, report exact evidence:
- section
- equation
- appendix
- table
- figure
- file
- line number

## 5. Runtime Truth

Never infer actual model behavior only from:
- filenames
- variable names
- comments
- documentation
- JSON schemas

Trace the actual execution path.

For questions such as "does the model use field X?":

1. Find where X is produced.
2. Trace where X is loaded.
3. Trace transformations applied to X.
4. Find where X is consumed.
5. Confirm whether it reaches the model, loss, or evaluation.

A field existing in a JSON file does not prove that the model uses it.

## 6. Code Changes

Do not make research-significant changes silently.

Before changing research-critical code:

1. Identify current behavior.
2. Identify source requirement.
3. Show the discrepancy.
4. Classify the proposed change as:
   - bug fix
   - implementation choice
   - deviation
   - experiment
5. Define how the change will be verified.
6. Ask the user when the decision is research-significant or ambiguous.

Do not:
- commit
- push
- delete datasets
- delete checkpoints
- overwrite expensive experiment outputs
- restart expensive preprocessing
- start long training jobs

unless explicitly requested.

## 7. Verification Standard

Do not report success only because a command completed without an exception.

Use appropriate verification, such as:

- unit tests
- tensor shape checks
- numerical checks
- invariance/equivariance tests
- data counts
- manifest validation
- checkpoint inspection
- retrieval metrics
- independent implementation comparison
- paper-equation comparison

For substantial work, report:

- What changed?
- Why?
- What evidence justified it?
- How was it verified?
- What remains unverified?

If something cannot be verified, explicitly say so.

## 8. Experiments

Experiments must be reproducible.

Record when relevant:

- git commit SHA
- configuration
- command
- random seed
- dataset split
- checkpoint
- model version
- software environment
- GPU/hardware
- metrics
- output path

Do not compare experiments without identifying uncontrolled differences.

## 9. Current Machine

Repository:

/home/kyzen/MetaFindV1

Data root:

/home/kyzen/data/MetaFind

Repository data link:

/home/kyzen/MetaFindV1/data
-> /home/kyzen/data/MetaFind

Conda environment:

MetaFind

Python:

/home/kyzen/miniconda3/envs/MetaFind/bin/python

Historical paths such as /mnt/data1 belong to the previous machine.

Do not assume old machine-specific absolute paths are valid.
Prefer project path abstractions over hard-coded paths.

If an old absolute path is found, report it before modifying it.

## 10. Communication

Be precise rather than agreeable.

If the user's hypothesis conflicts with evidence, state the disagreement clearly and show the evidence.

Report:
- contradictions
- failed tests
- negative results
- suspicious data
- unsupported assumptions
- uncertainty

Do not hide problems to make progress appear better than it is.

When evidence is insufficient, say:

"I cannot determine this reliably from the available evidence."

Then identify exactly what information is missing and ask the user.
