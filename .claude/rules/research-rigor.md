# Research Rigor

This project is scientific research and paper reproduction work.

Research correctness, traceability, and reproducibility take priority over speed, convenience, and task completion.

## 1. Evidence Before Conclusion

Do not present a technical claim as established unless it is supported by identifiable evidence.

For research-relevant claims, distinguish the evidence class defined by the project instructions:

- PAPER FACT
- UPSTREAM FACT
- OBSERVED IMPLEMENTATION
- OBSERVED DATA
- INFERENCE
- IMPLEMENTATION CHOICE
- DEVIATION
- UNKNOWN

Never convert an inference, implementation choice, or observed implementation into a paper fact.

Working code is evidence only that the code executed successfully. It is not sufficient evidence that the implementation matches the paper.

## 2. No Silent Guessing

Never silently fill in missing research-critical information.

If information is unknown, ambiguous, contradictory, or insufficiently supported, explicitly state that condition.

If resolving the uncertainty may affect any of the following, STOP and ask the user before making the decision:

- architecture
- model structure
- dataset construction
- data preprocessing
- feature construction
- loss functions
- optimization
- hyperparameters
- training procedure
- evaluation protocol
- metrics
- checkpoints
- reproduction fidelity
- scientific conclusions

Do not choose between multiple plausible implementations merely to continue the task.

## 3. Missing Information

If the paper or authoritative source does not specify a required detail:

1. State exactly what information is missing.
2. State which sources were checked.
3. Explain why the missing information matters.
4. List plausible interpretations only if useful.
5. Label those interpretations as INFERENCE or IMPLEMENTATION CHOICE.
6. Ask the user before adopting one when the choice can affect research results.

Absence of evidence must not be converted into evidence of absence.

## 4. Conflicting Evidence

When authoritative sources disagree:

1. Do not silently select one source.
2. Identify the conflicting statements.
3. Identify the source of each statement.
4. Preserve the distinction between paper, supplementary material, upstream work, documentation, repository behavior, and inference.
5. Explain the possible impact on reproduction.
6. Ask the user when the conflict affects a research-critical decision and cannot be resolved from stronger evidence.

## 5. Source Discipline

Prefer primary sources for research claims.

Priority should be given to:

- original paper and supplementary material
- official author-provided material
- original upstream papers
- official upstream implementations
- official documentation
- directly observed repository behavior

Do not use blogs, forum posts, Reddit, AI-generated summaries, or unsourced secondary explanations as authoritative research evidence.

Secondary sources may be used only as navigation aids for locating primary sources.

## 6. Observation vs Interpretation

Keep direct observations separate from interpretation.

Examples:

- "The repository sets hidden_dim=256" is OBSERVED IMPLEMENTATION.
- "The paper uses hidden_dim=256" is PAPER FACT only if the paper explicitly supports it.
- "The authors probably used hidden_dim=256" is INFERENCE.

Do not merge these categories.

## 7. Reproducibility

For research-relevant actions, preserve enough information to reproduce the result when practical, including:

- command executed
- relevant configuration
- input data or dataset version
- checkpoint or model version
- software environment when relevant
- random seed when relevant
- produced artifact or result
- validation performed

Do not claim reproducibility when required provenance is missing.

## 8. Scientific Claims

Do not make stronger conclusions than the evidence supports.

Avoid statements such as:

- "correct reproduction"
- "paper-faithful"
- "matches the paper"
- "verified"
- "equivalent"
- "same implementation"

unless the required evidence has actually been established.

Use narrower statements when appropriate, such as:

- "the implementation runs"
- "this behavior was observed"
- "this matches the stated equation"
- "this remains unverified"
- "this is an implementation choice"
- "the paper does not specify this detail"

## 9. Stop Condition

When continuing would require inventing, assuming, or silently selecting research-critical information, stop.

Report:

- what is known
- what is unknown
- available evidence
- why the uncertainty matters
- what decision is required from the user

Do not optimize for uninterrupted task completion at the expense of research correctness.
