# MetaFind Reproduction — Research Rigor & Decision Authority Rules

These rules govern all research, reproduction, implementation, auditing, and decision-making for the MetaFind project.

The primary goal is not merely to make the code run.

The goal is to reproduce MetaFind as faithfully as possible while maintaining strict provenance between:

1. what the MetaFind paper explicitly states,
2. what upstream papers explicitly state,
3. what upstream official implementations actually do,
4. what the current MetaFindV1 implementation actually does,
5. what is inferred,
6. what is chosen by the USER.

Never collapse these levels together.

---

# Rule 0 — Highest-Level Principle

Evidence discovery is NOT decision authority.

Finding a value in an upstream paper or repository does NOT automatically mean:

- MetaFind used it,
- this project should use it,
- it may be written into the reproduction protocol.

Always distinguish:

> "What the source did"

from

> "What MetaFind did"

from

> "What we will do"

Only explicit evidence or explicit USER approval may bridge these levels.

---

# Rule 1 — Evidence Classification

Every non-trivial technical claim must be classified into one of the following evidence levels.

## 1. PAPER FACT

The MetaFind paper itself explicitly states the claim.

Valid sources include:

- MetaFind main text
- equations
- tables
- captions
- appendix
- algorithms
- supplementary material

A claim may only be labeled PAPER FACT if MetaFind itself supports it.

Upstream papers do NOT create MetaFind PAPER FACTs.

---

## 2. UPSTREAM-PAPER FACT

A directly inherited upstream paper explicitly states the claim.

Examples:

- ULIP-2
- ULIP-1
- Point-BERT
- EGNN
- OpenShape
- ProcTHOR
- I-Design

This means:

> "the upstream paper explicitly does this"

It does NOT mean:

> "MetaFind explicitly does this"

or:

> "we automatically adopt this"

---

## 3. UPSTREAM-OFFICIAL-IMPL FACT

The official repository of a directly relevant upstream method explicitly implements the behavior.

Examples:

- Salesforce ULIP official repo
- official OpenShape repo
- official EGNN repo

This means:

> "the official upstream implementation does this"

It does NOT mean:

> "MetaFind necessarily did this"

or:

> "our reproduction automatically adopts this"

---

## 4. UPSTREAM-EXPERIMENT-SPECIFIC FACT

A setting appears in a specific upstream experiment but is not intrinsic to the architecture.

Examples:

- EGNN uses 7 layers on QM9
- a specific OpenShape experiment uses a certain learning rate
- a particular ULIP script uses a certain batch size

These settings have weaker inheritance authority than architectural definitions.

Do NOT silently treat them as MetaFind defaults.

---

## 5. LIBRARY / PARSER DEFAULT

A framework, parser, library, CLI, or generic training script provides a default value.

Examples:

- argparse default
- PyTorch default
- optimizer default
- generic config default

A library/parser default alone is NEVER an authoritative reproduction value.

It may only be reported as implementation context.

---

## 6. OBSERVED IMPLEMENTATION

The current MetaFindV1 source code actually behaves this way.

This means:

> "the current project code does this"

It does NOT prove:

> "the MetaFind paper did this"

Existing code must not be used as retrospective evidence for the paper.

---

## 7. INFERENCE

A conclusion is derived from paper structure, mathematics, wording, consistency constraints, or comparison with upstream work, but is not explicitly stated.

Inference must remain visibly labeled.

Never rewrite an inference as a paper fact.

---

## 8. IMPLEMENTATION CHOICE

The paper does not uniquely specify the value or behavior, so the project must choose one.

Only the USER may promote an unresolved implementation choice into the official reproduction protocol.

Claude may recommend a choice.

Claude may NOT finalize it without USER approval.

---

## 9. DEVIATION

The implementation intentionally differs from the MetaFind paper or the selected reproduction interpretation.

Every deviation must be explicitly recorded.

Never hide a deviation as if it were faithful reproduction.

---

## 10. UNKNOWN

The available evidence is insufficient to determine the answer.

UNKNOWN is a valid and preferred answer when the evidence is incomplete.

Do not guess merely to avoid writing UNKNOWN.

---

# Rule 2 — Paper Silence

When MetaFind does not specify a detail:

DO NOT:

- guess,
- silently use a default,
- infer a value merely because an upstream repository uses it,
- infer a value merely because a library uses it,
- infer a value merely because current MetaFindV1 code already uses it.

Instead follow the Upstream Lookup Procedure defined below.

After the lookup, classify the missing detail by type.

---

# Rule 3 — Upstream Inheritance

MetaFind explicitly builds on several upstream systems.

Therefore upstream sources are mandatory evidence sources when MetaFind is silent.

However:

> Upstream evidence is not automatic inheritance.

The inheritance rule depends on the type of missing information.

---

## Type A — Architecture / Mathematics / Module Mechanics

Examples:

- EGNN message passing equations
- coordinate update form
- distance representation
- Point-BERT grouping/tokenization mechanics
- ULIP-2 encoder architecture
- intrinsic normalization inside an inherited module
- mathematical behavior required by an upstream module

If ALL of the following are true:

1. MetaFind explicitly says it uses, adopts, extends, or builds upon the upstream method,
2. MetaFind does not state that this specific mechanism was modified,
3. the upstream definition is part of the method itself rather than a task-specific experiment,

then the upstream definition may be treated as the preferred reconstruction candidate.

It must still be labeled:

- UPSTREAM-PAPER FACT
or
- UPSTREAM-OFFICIAL-IMPL FACT

If the project adopts it, additionally record:

- IMPLEMENTATION CHOICE / inherited from upstream

Never relabel it as PAPER FACT.

---

## Type B — Numerical Hyperparameters / Training Recipe / Experiment-Specific Values

Examples:

- learning rate
- batch size
- epochs
- number of layers
- hidden dimensions
- warmup duration
- weight decay
- optimizer betas
- gradient clipping
- scheduler
- random seed
- checkpoint selection
- early stopping
- validation frequency
- initialization values
- dropout probabilities not stated by MetaFind

These MUST NOT be automatically inherited merely because:

- an upstream paper used them,
- an upstream repository used them,
- a training script used them,
- a parser default contains them.

They may only be recorded as:

> UPSTREAM CANDIDATE

Claude may recommend one.

Claude may explain why it is a reasonable starting point.

Claude may NOT write it into the official MetaFind reproduction protocol without explicit USER approval.

---

## Type C — Task-Specific Upstream Experimental Settings

A setting used by an upstream method on a different task must receive especially weak inheritance authority.

Example:

> EGNN uses 7 layers on QM9.

This proves only:

> EGNN's QM9 experiment used 7 layers.

It does NOT justify:

> MetaFind ESSGNN should use 7 layers.

Unless MetaFind explicitly imports that setting, it remains an UPSTREAM CANDIDATE at most.

---

## Type D — Library / Parser Defaults

Library defaults and parser defaults never resolve MetaFind paper silence by themselves.

They may only be reported as:

> implementation context / default

They must NOT become protocol values without USER approval.

---

# Rule 4 — Mandatory Upstream Lookup Procedure

For every unresolved MetaFind detail that may affect research results:

## Step 1 — Re-check MetaFind completely

Search:

- main text
- equations
- appendix
- tables
- captions
- algorithms
- supplementary material

Do not conclude "paper silence" from one paragraph alone.

---

## Step 2 — Identify the direct upstream dependency

Examples:

- ULIP-2 for multimodal encoding
- Point-BERT for the point-cloud branch
- EGNN for equivariant message passing
- OpenShape for relevant 3D retrieval practices
- ProcTHOR for dataset semantics
- I-Design for scene-generation/evaluation integration

Do not use unrelated literature merely because it looks similar.

---

## Step 3 — Read the upstream paper

Determine whether the answer is:

- architectural,
- mathematical,
- training-specific,
- dataset-specific,
- experiment-specific.

---

## Step 4 — Inspect official upstream code if needed

Use official repositories only.

Separate:

- actual implementation behavior,
- experiment script values,
- parser defaults,
- library defaults.

Do not merge them into one "upstream setting".

---

## Step 5 — Report the evidence

For every unresolved item report:

1. What MetaFind says
2. What MetaFind does not say
3. What the upstream paper says
4. What the upstream official code does
5. Whether the upstream evidence is architectural or experiment-specific
6. Evidence classification
7. Whether inheritance is justified
8. Remaining uncertainty
9. Candidate recommendation if appropriate

---

## Step 6 — USER decision gate

If the item still requires an IMPLEMENTATION CHOICE and may affect research results:

STOP after presenting the evidence.

Do not silently implement or finalize the choice.

Mark:

> STATUS: PENDING USER DECISION

---

# Rule 5 — Decision Authority

Claude's authority is limited to:

1. searching evidence,
2. reading evidence,
3. classifying evidence,
4. identifying contradictions,
5. deriving mathematical consequences,
6. presenting candidate implementations,
7. comparing alternatives,
8. recommending a preferred option,
9. explaining the consequences of each option.

Claude does NOT have authority to:

1. promote an IMPLEMENTATION CHOICE into the official protocol,
2. treat an UPSTREAM FACT as a MetaFind PAPER FACT,
3. treat an UPSTREAM value as automatically inherited,
4. override a USER-approved decision,
5. silently change an approved protocol,
6. declare a candidate "final",
7. declare a recommendation "decided",
8. modify research-critical settings merely because they appear more reasonable,
9. use current code as proof that the paper intended the same behavior.

Only the USER may approve unresolved research-impacting implementation choices.

---

# Rule 6 — USER Decisions and Decision Ledger

An explicit USER decision has implementation authority for this reproduction project.

A USER-approved decision may override an upstream candidate.

However:

A USER decision does NOT change source provenance.

For example:

> USER chooses λ_init = 0.

Correct classification:

- MetaFind PAPER: λ is learnable scalar
- MetaFind PAPER: initialization unspecified
- Flamingo: zero-gated new branch is an upstream/domain precedent
- PROJECT: λ_init = 0 is USER-approved IMPLEMENTATION CHOICE

Incorrect classification:

> MetaFind uses λ_init = 0.

Every approved research-critical decision should be recorded in the Decision Ledger.

---

# Rule 7 — No Silent Promotion

The following silent transitions are forbidden:

UNKNOWN
→ PAPER FACT

INFERENCE
→ PAPER FACT

UPSTREAM-PAPER FACT
→ PAPER FACT

UPSTREAM-OFFICIAL-IMPL FACT
→ PAPER FACT

UPSTREAM CANDIDATE
→ OFFICIAL PROTOCOL

IMPLEMENTATION CHOICE
→ OFFICIAL PROTOCOL

OBSERVED IMPLEMENTATION
→ PAPER FACT

LIBRARY DEFAULT
→ IMPLEMENTATION CHOICE

A recommendation is not a decision.

A default is not a decision.

An upstream setting is not a decision.

Existing code is not automatically authoritative.

---

# Rule 8 — Contradictions Inside MetaFind

If two parts of MetaFind disagree:

Examples:

- Method vs Appendix
- text vs figure
- equation vs proof
- experiment prose vs table
- architecture diagram vs implementation description

DO NOT choose one silently.

Create a contradiction entry containing:

1. source A
2. source B
3. exact disagreement
4. mathematical/implementation consequences
5. upstream evidence
6. possible reconciliations
7. recommended interpretation
8. evidence level

If the contradiction changes implementation behavior, ask the USER to approve the adjudication unless an earlier USER decision already exists.

---

# Rule 9 — Mathematical Claims

For every mathematical equivalence claim:

Examples:

- equation ↔ cross entropy
- bidirectional loss ↔ logits transpose
- SE(3)/E(3) equivariance
- sparse-neighborhood adaptation
- scalar/vector coordinate update

Explicitly state all assumptions needed for the equivalence.

Never say:

> "equivalent"

when the equivalence only holds under unstated conditions.

Preferred form:

> Equivalent under assumptions A, B, C.

If an implementation modifies the paper equation, determine whether the proof assumptions still hold.

---

# Rule 10 — Training and Evaluation Integrity

Any choice involving:

- train/validation/test split
- checkpoint selection
- early stopping
- hyperparameter tuning
- model selection
- evaluation gallery
- positive/negative definition
- random seed
- metric computation

is research-critical.

Never use a paper-designated test set for hyperparameter selection or early stopping unless the paper explicitly does so.

If the reproduction changes the evaluation protocol, label the change clearly as an IMPLEMENTATION CHOICE or DEVIATION.

Do not report a model-selection-contaminated test result as if it were an untouched paper-comparable test result.

---

# Rule 11 — Current Code Is Not Authority

When inspecting MetaFindV1:

Current code may establish:

> OBSERVED IMPLEMENTATION

but NOT:

> PAPER FACT

If current code conflicts with:

- MetaFind,
- approved Decision Ledger entries,
- current reproduction protocol,

the conflict must be surfaced.

Do not defend existing code merely because it already exists.

Implementation inertia is not evidence.

---

# Rule 12 — STOP Conditions

Do NOT immediately interrupt the USER whenever paper silence is found.

First complete the upstream lookup procedure.

After evidence collection, STOP and ask the USER only when the unresolved item:

- can materially affect reported Table 1 / Table 2 / Table 3 results,
- changes architecture,
- changes mathematical behavior,
- changes trainable/frozen parameters,
- changes loss definition,
- changes positive/negative pairing,
- changes data split,
- changes gallery/evaluation universe,
- changes optimizer/training recipe,
- changes initialization with plausible optimization consequences,
- introduces a deviation from MetaFind.

When asking the USER, do NOT ask a bare question.

Present:

1. unresolved item,
2. MetaFind evidence,
3. upstream evidence,
4. candidate A,
5. candidate B if relevant,
6. expected consequences,
7. your recommendation,
8. evidence level.

Then ask for the decision.

---

# Rule 13 — Do Not Ask When Evidence Already Resolves It

If MetaFind explicitly states the answer:

use the MetaFind answer.

Do not ask the USER to choose something already specified by the paper.

If an existing Decision Ledger entry already resolves the implementation choice:

follow the Decision Ledger.

Do not reopen settled decisions unless new evidence materially contradicts them.

---

# Rule 14 — Reproduction Priority Order

When deciding how to reconstruct a missing detail, use this priority:

1. MetaFind explicit statement
2. MetaFind mathematical consistency / appendix / algorithm
3. Existing USER-approved adjudication
4. Direct upstream paper definition
5. Direct upstream official implementation
6. Closely related first-party upstream experiment
7. General domain convention
8. Project implementation choice
9. UNKNOWN

This priority is NOT automatic inheritance.

It is the order in which evidence should be considered.

---

# Rule 15 — Architectural Inheritance vs Hyperparameter Inheritance

This distinction is mandatory.

Example:

MetaFind says it extends EGNN.

Reasonable upstream inheritance candidate:

- EGNN-style invariant distance input
- scalar coordinate coefficient
- equivariant coordinate update structure

Not automatically inherited:

- EGNN QM9 layer count
- QM9 hidden dimension
- QM9 optimizer
- QM9 learning rate
- QM9 epochs

Likewise:

MetaFind says it uses ULIP-2.

Reasonable upstream inheritance candidate:

- ULIP-2 encoder structure
- Point-BERT branch behavior
- frozen CLIP behavior if supported by ULIP-2 source

Not automatically inherited:

- a random ULIP script's epochs
- generic parser batch size
- generic parser scheduler
- unrelated ULIP experiment hyperparameters

---

# Rule 16 — Official Protocol Promotion

An item may enter the official reproduction protocol only through one of these paths:

## Path A
MetaFind PAPER FACT directly specifies it.

## Path B
The USER explicitly approves an IMPLEMENTATION CHOICE.

## Path C
A previously approved Decision Ledger entry already specifies it.

No other path is valid.

UPSTREAM FACT alone is insufficient.

---

# Rule 17 — Language Discipline

Use precise wording.

Allowed:

- "MetaFind explicitly states..."
- "ULIP-2 explicitly states..."
- "The official ULIP implementation does..."
- "This is an upstream candidate."
- "This is the most likely interpretation."
- "This remains unknown."
- "The project currently chooses..."
- "The current code implements..."

Avoid unsupported wording such as:

- "MetaFind definitely uses..."
- "the paper specifies..."
- "this is equivalent..."
- "this is validated..."
- "this is the official setting..."
- "this is the final design..."

unless the corresponding evidence actually supports that strength.

---

# Rule 18 — Evidence Citation

For every disputed or research-critical claim, cite the primary source.

Preferred citation format:

- `2methdology.tex:87`
- `appendix.tex:49-59`
- `ULIP-2 main.tex:612-616`
- official repo file + relevant line/function

Do not cite:

- blogs
- Reddit
- forum posts
- secondary summaries
- AI-generated summaries

when a primary source exists.

---

# Rule 19 — Researcher Behavior

When uncertain:

ASK AFTER EVIDENCE SEARCH.

Do not:

- improvise,
- hide uncertainty,
- optimize for convenience,
- modify architecture silently,
- make the code run by changing the science.

A failed reproduction with clearly identified uncertainty is preferable to a successful run whose provenance is unclear.

---

# Rule 20 — Required Output for an Unresolved Decision

When presenting a research-critical unresolved item, use this structure:

### Issue
What is unresolved?

### MetaFind
What does the MetaFind paper explicitly say?

### MetaFind Silence
What exactly is not specified?

### Upstream Paper
What does the direct upstream paper say?

### Upstream Official Code
What does the official implementation do?

### Evidence Classification
PAPER FACT / UPSTREAM-PAPER FACT / UPSTREAM-OFFICIAL-IMPL FACT / UPSTREAM-EXPERIMENT-SPECIFIC FACT / INFERENCE / UNKNOWN

### Candidate A
Description + consequences.

### Candidate B
Description + consequences.

### Recommendation
Claude's preferred option and why.

### Decision Status
STATUS: PENDING USER DECISION

Do not modify the official protocol until the USER approves one candidate.

---

# Final Standing Rule

When MetaFind does not specify an answer:

> FIRST search the direct upstream paper and official implementation.

But:

> UPSTREAM IS A MANDATORY EVIDENCE SOURCE, NOT AN AUTOMATIC DECISION SOURCE.

Architecture, mathematics, and intrinsic module mechanics may preferentially inherit from a directly adopted upstream method when MetaFind does not state a modification.

Numerical hyperparameters, training recipes, experiment-specific settings, checkpoint policies, and library defaults must NOT be automatically inherited.

They remain candidates until explicitly approved by the USER.

When in doubt:

SEARCH → CLASSIFY → EXPLAIN → RECOMMEND → ASK.

Never:

SEARCH → FIND DEFAULT → SILENTLY ADOPT.