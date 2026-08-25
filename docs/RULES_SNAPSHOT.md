# MetaFindV1 規則快照

repo commit `e789d13`　·　由 `tools/dump_rules.py` 產生

**不要直接編輯本檔。** 要改請改下列來源檔，再重新執行該腳本。

---

## 生效中的規則檔

| # | 名稱 | 來源路徑 | 行數 | sha256(前16) |
|---|---|---|---|---|
| 1 | 專案指令 | `CLAUDE.md` | 184 | `f27e3af2caefd05d` |
| 2 | 規則 1：研究嚴謹度 | `.claude/rules/research-rigor.md` | 156 | `597bbe4599103f23` |
| 3 | 規則 2：論文復現 | `.claude/rules/paper-reproduction.md` | 235 | `b59cae6d73d7db6e` |
| 4 | 規則 3：實驗 | `.claude/rules/experiments.md` | 376 | `98491f1fd9c00c47` |
| 5 | 規則 4：程式碼變更 | `.claude/rules/code-changes.md` | 289 | `b5fd170803ecf83d` |
| 6 | 規則 5：上游查找 | `.claude/rules/upstream-lookup.md` | 84 | `203861fec9f8a535` |

`.claude/` 依 `.gitignore:56` 不進版控，五個規則檔都是本機檔案，每次 session 由 Claude Code 自動載入。本快照是它們唯一的可攜副本。

---

## 產生本快照時發現的規則衝突（**需 Kyzen 裁決**）

### 衝突 C-R1：上游值到底能不能採用

**規則 2（論文復現）§3 Paper Silence** 明文禁止：

> Do not infer a value merely because:
> - **an upstream repository uses it**
> - a library defaults to it

**但 Kyzen 的 standing rule**（2026-08-25 口頭、2026-08-26 再次下令並要求寫入規則）是：

> 「若 metafind論文找不到答案去找上游的論文」
> 「找不到答案高機率是照原架構的方法」

**這兩條在決策方向上相反。** 規則 2 說「上游用了不構成採用理由」，standing rule 說「論文沒講就照上游」。

**影響範圍不小。** 本輪依 standing rule 採用或建議採用的值，全部落在規則 2 §3 的禁止清單裡：

| 值 | 來源 | 規則 2 §3 的字面判定 |
|---|---|---|
| Stage 1 epochs 250 | ULIP `main.py:47` default ＋ 官方腳本 | 「a library defaults to it」→ 禁止 |
| 不早停、取 best checkpoint | ULIP `main.py:212-231`、OpenShape `train.py:190-201` | 「an upstream repository uses it」→ 禁止 |
| ESSGNN 層數 7 | EGNN `main_qm9.py:34` | 同上 |
| ESSGNN pooling sum | EGNN `qm9/models.py:83` | 同上 |
| lr 建議 5e-4 | OpenShape supp:190 | 這條是**論文**不是 repo，規則 2 §3 不禁 |

**可能的調和讀法**（我的判讀，非裁決）：規則 2 §3 第一句其實留了門——
"classify it as UNKNOWN **unless another authoritative source resolves it**"，
而規則 2 §4 又說上游細節在「有證據 MetaFind 採用／繼承／依賴」時即為 MetaFind-relevant。
MetaFind 明文建構在 ULIP-2 與 EGNN 之上，所以上游**可以**算 authoritative source。
若採此讀法，§3 真正禁止的是「照抄後當成已解決、標成 PAPER FACT」，
而不是「採用上游值並標成 UPSTREAM FACT」。

**我目前的做法一律標 UPSTREAM FACT ＋ 附檔案行號，從未標成 PAPER FACT。**
但字面衝突仍在，需要 Kyzen 用一句話定調，二選一：

- **甲**：規則 2 §3 加註「上游官方論文與程式碼算 authoritative source；採用時標 UPSTREAM FACT，不得標 PAPER FACT」。standing rule 勝出。
- **乙**：standing rule 只適用於「架構與方法」，不適用於「超參數數值」；數值一律 UNKNOWN 並上呈 Kyzen。規則 2 §3 勝出。

**在 Kyzen 定調前，本輪所有依 standing rule 採用的值都維持「建議」狀態，不寫入協定。**

### 衝突 C-R2（較輕）：什麼時候該停下來問人

**規則 1（研究嚴謹度）§2** 列出一長串「會影響研究結果就 STOP 並詢問使用者」的清單，
其中包含 `hyperparameters`、`training procedure`、`optimization`。
**規則 5（上游查找）** 則要求四步查完才准上呈。

這兩條**不矛盾、是順序關係**（先查完再問），但規則 1 §2 字面沒有「先查上游」這一步。
建議在規則 1 §2 加一句指向規則 5，避免下一個讀到的人又直接跳去問人。
此項不影響現有結論，屬文件整併。

---


==============================================================================
## 專案指令

來源：`CLAUDE.md`

==============================================================================

# MetaFindV1 Research Instructions

MetaFindV1 is a scientific paper reproduction project.

The priority is research correctness, reproducibility, and traceability.
Working code or passing tests alone are not sufficient evidence of reproduction fidelity.

Project-specific instructions here override conflicting general user defaults.

## 1. Core Research Rule

Accuracy takes priority over speed.

Never fabricate, silently guess, or fill in missing research-critical information.

If an unknown, ambiguity, or contradiction could materially affect architecture, dataset construction, preprocessing, training, evaluation, metrics, reproduction fidelity, or scientific conclusions:

- identify the uncertainty;
- show the available evidence;
- do not silently choose between plausible interpretations;
- ask the user before making a research-significant choice.

## 2. Evidence Classification

Classify important technical conclusions as one of:

- **PAPER FACT** — explicitly stated by MetaFind.
- **UPSTREAM FACT** — supported by an upstream paper or official implementation.
- **OBSERVED IMPLEMENTATION** — confirmed from current repository code.
- **OBSERVED DATA** — confirmed from actual project data or outputs.
- **INFERENCE** — logically inferred but not explicitly stated.
- **IMPLEMENTATION CHOICE** — selected because the source is underspecified.
- **DEVIATION** — intentionally differs from the source.
- **UNKNOWN** — evidence is insufficient.

Never present an inference, implementation choice, deviation, or unknown as a paper fact.

## 3. Authority and Provenance

For MetaFind reproduction, use this authority order:

1. Original MetaFind source / supplementary material
2. Published MetaFind paper
3. Original upstream papers and official upstream implementations
4. Verified project audit / implementation contract
5. Graph specification
6. Current repository implementation
7. Tests and observed runtime/data
8. Reasoned inference
9. Session Handoff / conversational memory

A lower-authority source must not silently override a higher-authority source.

If authoritative sources conflict:

- identify both sources;
- show the conflicting definitions;
- explain the implementation impact;
- leave the conflict explicit until it is resolved.

`Session Handoff.md`, previous AI notes, README files, comments, and conversational memory are working context, not scientific authority.

## 4. Paper Verification

Before claiming that a paper specifies a behavior, verify the relevant primary source.

Check the relevant section, equations, appendix/supplementary material, figures/captions, and inherited upstream behavior when applicable.

Prefer precise provenance such as:

- section;
- equation;
- appendix;
- table;
- figure;
- authoritative file/location.

Do not establish paper behavior from summaries, previous AI notes, README text, or memory alone.

## 5. Runtime Truth

Do not infer actual model behavior from filenames, variable names, comments, schemas, or documentation alone.

For questions such as whether field `X` is actually used, trace:

1. where `X` is produced;
2. where it is loaded;
3. transformations applied to it;
4. where it is consumed;
5. whether it reaches the model, loss, retrieval, or evaluation path.

Existence in a file or schema does not prove runtime use.

## 6. Research-Significant Code Changes

Before changing research-critical behavior:

1. establish current behavior;
2. establish the source requirement;
3. identify the discrepancy;
4. classify the change as `bug fix`, `implementation choice`, `deviation`, or `experiment`;
5. define how the change will be verified.

Ask the user before resolving a research-significant ambiguity or intentionally deviating from the authoritative source.

Do not start long training/preprocessing jobs, delete datasets/checkpoints, or overwrite expensive experiment outputs unless explicitly requested.

## 7. Verification Standard

A successful command or green test suite is not sufficient by itself.

Use verification appropriate to the claim, including when relevant:

- unit or regression tests;
- tensor shape / numerical checks;
- invariance or equivariance tests;
- dataset and manifest validation;
- checkpoint inspection;
- retrieval/evaluation metrics;
- paper-equation comparison;
- independent implementation comparison.

Before reporting substantial work complete, establish:

- what changed;
- why it changed;
- evidence supporting it;
- how it was verified;
- what remains unverified.

## 8. Experiment Reproducibility

For meaningful experiments, preserve the information needed to reproduce the result:

- git commit SHA;
- configuration and command;
- random seed and dataset split;
- checkpoint/model version;
- environment and hardware;
- metrics and output path.

Do not compare experiments without identifying relevant uncontrolled differences.

## 9. Current Environment

- Repository: `/home/kyzen/MetaFindV1`
- Data root: `/home/kyzen/data/MetaFind`
- Repository data link: `/home/kyzen/MetaFindV1/data -> /home/kyzen/data/MetaFind`
- Conda environment: `MetaFind`
- Python: `/home/kyzen/miniconda3/envs/MetaFind/bin/python`

- Secondary volume: `/mnt/data1` — **valid on this host.** `/dev/sda1`, 3.6 TB ext4, mounted, `/mnt/data1/kyzen` writable. Filesystem created 2026-08-20; it is a **new disk on this machine**, not a survival of the previous one.

**Corrected 2026-08-21.** This section previously read *"Paths under `/mnt/data1` belong to the previous machine and must not be assumed valid."* That statement is false and any reasoning that relied on it should be re-checked.

**`/mnt/data1` is an SMR drive** (`ST4000DM004`). Sustained small-file writes collapse to single-digit MB/s once its CMR cache fills — measured `w_await` above 5,000 ms under a mixed write load. It suits **cold, read-mostly bulk data**. It is a poor host for high-frequency small-file work such as writing 45,952 `.npz` embeddings. Decide placement per directory, not for `data/` as a whole.

Prefer project-relative paths or current path abstractions. Report stale machine-specific absolute paths before modifying them.

## 10. Graphify

A project knowledge graph exists under `graphify-out/`.

For codebase questions when `graphify-out/graph.json` exists:

- use `graphify query "<question>"` for scoped codebase retrieval;
- use `graphify path "<A>" "<B>"` for relationships;
- use `graphify explain "<concept>"` for focused concepts;
- use `graphify-out/wiki/index.md` for broad navigation when available.

Use `GRAPH_REPORT.md` only when scoped retrieval is insufficient or a broad architecture review is required.

After meaningful code changes, run:

`graphify update .`

## 11. Session Handoff

`Session Handoff.md` contains only the latest cross-session working state.

It is not scientific authority and must yield to the authority hierarchy above.

If a handoff conflicts with a higher-authority source or current repository state, use the higher-authority information and correct the handoff when the next handoff is generated.



==============================================================================
## 規則 1：研究嚴謹度

來源：`.claude/rules/research-rigor.md`

==============================================================================

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



==============================================================================
## 規則 2：論文復現

來源：`.claude/rules/paper-reproduction.md`

==============================================================================

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



==============================================================================
## 規則 3：實驗

來源：`.claude/rules/experiments.md`

==============================================================================

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



==============================================================================
## 規則 4：程式碼變更

來源：`.claude/rules/code-changes.md`

==============================================================================

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



==============================================================================
## 規則 5：上游查找

來源：`.claude/rules/upstream-lookup.md`

==============================================================================

# Upstream Lookup (mandatory before escalating any unknown)

USER ORDER, Kyzen, 2026-08-26, verbatim:
「若 metafind論文找不到答案去找上游的論文」

This rule exists because the order had to be given twice. Both times the failure
was the same: declaring "no source" after searching only one place, then
escalating a question upstream had already answered.

## The sequence. No step may be skipped.

1. **MetaFind** -- paper, appendix, figures, supplementary.
2. **Upstream paper** -- ULIP-2 first (the direct backbone), then ULIP-1,
   Point-BERT, EGNN, OpenShape, ProcTHOR, whichever owns the component.
3. **Upstream official code** -- the repo, and specifically:
   - the argument parser and its **defaults**
   - the **launch scripts** (`scripts/*.sh`), which may override those defaults
   - **config files** (`*.yaml`, `*.json`) **read end to end, not just the field
     you came for**
   - the **training loop itself**: what it validates on, how it selects a
     checkpoint, whether it early-stops
4. **Released artifacts** -- the checkpoint. Tensor shapes and key counts are
   OBSERVED DATA and outrank every document about them.
5. **Only now** may the question go to the USER, and the report must state which
   of steps 1 to 4 were searched and what each returned.

## Recorded failures. Do not repeat them.

- **2026-08-25** -- Declared "epochs has no source" after grepping the ULIP-2
  paper for four words. The answer was `main.py:47`, default 250, and the
  official pretrain script does not override it. Step 3 was skipped.
- **2026-08-26** -- Quoted `ULIP_2_PointBERT_10k_colored_pointclouds.yaml` for
  the point count and stopped reading. The same file carries the optimizer
  block, the scheduler block, and `depth: 18`. Step 3 done partially is the same
  as step 3 not done.
- **2026-08-26** -- Wrote "Transformer depth 12" from the Point-BERT paper while
  the checkpoint we actually load has **18** blocks (counted from
  `ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`).
  Step 4 outranks step 2 and was not run.
- **2026-08-26** -- Tagged ESSGNN's `n_layers` and `pooling` as "our choice, the
  paper gives no value" after reading only the EGNN paper's equations. The EGNN
  repo has both: QM9 uses `n_layers 7` (`main_qm9.py:34`) and its readout is
  `torch.sum` (`qm9/models.py:83`). Worse, the `4` in our config is the
  **N-body** default (`main_nbody.py:35`), so two different tasks were mixed.
  When upstream ships several task configs, pick the one MetaFind's own citation
  points at: it says drug design, which is QM9.
- **Earlier** -- Built n04 on pyrender without reading how ULIP and OpenShape
  render. Cost a full 46K re-render.

## Reading a config file

Read the whole file. Then establish **which parts the code actually consumes**,
because a config can carry dead sections:

> `ULIP_models.py:364` passes only `config.model` into `PointTransformer_Colored`.
> So `depth: 18` is live, and the `optimizer` / `scheduler` blocks in the same
> yaml are dead for tri-modal training (they are Point-BERT reconstruction
> leftovers, as `consider_metric: CDL1` confirms).

Quoting a dead config section as an upstream recipe is the same class of error
as inventing one.

## What still counts as escalation-worthy

Escalate only when one of these is true, and say which:

- **All four steps returned nothing.**
- **Upstream contradicts itself**, and both sides are primary. Example: ULIP-1's
  paper states `lr 1e-3` while `scripts/pretrain_pointbert.sh` explicitly passes
  `--lr 3e-3` and `main.py:52` defaults to 3e-3. The standing rule cannot pick
  for us here.
- **Upstream's answer rests on a premise we do not have.** Example: upstream
  selects checkpoints on an independent benchmark because it pretrains on one
  corpus and validates on another; MetaFind trains and evaluates on the same
  corpus, so the mechanism transfers but the ingredient does not.
- **Upstream's answer conflicts with MetaFind's own text.** MetaFind wins; record
  the deviation.

## Classification

An adopted upstream answer is `UPSTREAM FACT` with its file and line, never a
`PAPER FACT` about MetaFind. A value read off a checkpoint is `OBSERVED DATA`.
Neither may be reported as something MetaFind states.

