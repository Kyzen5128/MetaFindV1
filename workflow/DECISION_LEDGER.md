# Decision Ledger

> **The decisions that are in force.** This is the project-level record: if any other document
> disagrees with an entry here, this file wins and the other document is corrected.
>
> Master maintains it. No one else writes an entry.
>
> **Findings do not belong here.** A finding is what is true; a decision is what to do about it.
> Findings live in block evidence, handoffs and audit documents. Only decisions that change
> project state are recorded here.
>
> Rules on Finding vs Decision, materiality, and acceptance: `workflow/BLOCKS.md`.

---

## Reading older entries

Entries are **never edited to match today's vocabulary** — an edited record is not a record.
Older entries name the work packages and identifiers that were in use when the decision was
taken. Those names no longer route anywhere; the surviving artefacts they refer to are under
`workflow/archive/` and `workflow/blocks/<BLOCK>/evidence/`.

**Read an entry for its decision, its evidence, and its authority classification.** Do not read
it for project state, and do not follow its file paths — use `workflow/MASTER.md` for state.

---

## The rule

> A material decision reaches `USER_APPROVED` **only** when the user approves it.
>
> Claude + Codex + Master consensus does **not** substitute for user approval.

Master's `ACCEPT` / `ACCEPT WITH FOLLOW-UP` enters this ledger as `AWAITING_USER_REVIEW`, never as `USER_APPROVED`.

---

## Status vocabulary

| Status | Meaning |
|---|---|
| `PROPOSED` | A decision has been formulated but no Master integration review has completed |
| `AWAITING_USER_REVIEW` | Master has reviewed and recommended. A USER REVIEW BRIEF is owed or delivered. **Not yet project state** |
| `USER_APPROVED` | The user approved. **FINAL ACCEPTED.** Master may integrate into MASTER.md, CONTEXT.md, INDEX.md, dependency state, and project-wide contracts |
| `USER_REJECTED` | The user rejected. The underlying finding may still stand; the proposed remedy does not |
| `SUPERSEDED` | Replaced by a later decision. **Never delete a superseded entry** — mark it and point to the replacement |

`MODIFY` is recorded as `USER_APPROVED` with the **user's** wording as the decision, and Master's original proposal retained in the notes.

`INVESTIGATE MORE` leaves the entry at `AWAITING_USER_REVIEW` with the additional investigation named.

---

## What belongs here

Any addition, modification, confirmation, or reversal of:

paper interpretation · architecture · implementation choice · deviation · dataset semantics · annotation semantics · preprocessing · training protocol · evaluation protocol · shared artifact semantics · cache validity · checkpoint validity · dependency ordering · scientifically meaningful assumption · anything that can materially change reproduction results

**When in doubt, record it.**

## What does not belong here

Routine execution that changes no scientific behaviour and no shared contract: local refactors inside an approved scope, test scaffolding, log formatting, documentation corrections that only make a comment describe existing code accurately.

---

## Entry format

Each entry records:

```
### <Decision ID>

| | |
|---|---|
| Source | task / D0 that produced it |
| Issue / Finding | what was discovered (a FINDING, not a remedy) |
| Evidence references | file:line, paper section, measurement, decision file section |
| Decision / Resolution | what is to be done (a DECISION) |
| Authority classification | PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION / OBSERVED DATA / INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / USER DECISION |
| USER final decision | APPROVE / REJECT / MODIFY / INVESTIGATE MORE / — |
| Affected components | tasks, artifacts, stages |
| Status | see vocabulary above |
| Date | proposed / resolved |
```

Keep the Issue and the Decision in separate rows. Do not merge them.

---

# Ledger

> Entries are added by Master at integration review and updated when the user acts.
> **Ordered newest first.**

---

## Registered defect in this file — `DL-006` is used twice

**Two different decisions carry the id `DL-006`.** Neither entry is edited; an edited record is
not a record. Cite them by **date**, never by id alone:

| | |
|---|---|
| `DL-006` **(2026-08-22)** | the three legacy-v1 residuals are deleted and re-annotated under v5 |
| `DL-006` **(2026-08-21)** | the n05 annotation model is `Qwen3.8-27B` — **superseded by `DL-008`** |

Registered by Master 2026-08-22 during the re-initialization audit. `DL-008` is the next free
id; nothing reuses `DL-006`.

---

### `DL-018` — `n04`'s corpus is mixed-generation and must be re-run. **A pipeline can change behaviour mid-run and no artifact reveals it.**

**Found independently by Master at integration and by the ULIP2 Engineer, within minutes of each
other. The Engineer's account is more complete and it self-reports the cause.**

| | |
|---|---|
| **What is on disk** | `n04` finished 20:43:03 — 45,958 sidecars + 94 quarantined = 46,052. **`renderer_version` is `{3: 43,412, 4: 2,546}`.** Verified by Master over the whole index |
| **Cause — the Engineer's, self-reported** | `n04` runs under `multiprocessing`; a newly spawned worker **re-imports the module**. `renders.py` was edited **while the run was in flight** (`mtime` 20:38:07; the version boundary in the sidecars is 20:38:45 — a 38-second lag, i.e. workers picked it up almost immediately). **The run changed program mid-flight** |
| **The part that makes a partial salvage unsafe** | The **bake fix landed 2–3 minutes before the version bump**. At ~686 assets/min that is roughly **1,700 assets carrying the corrected geometry while stamped `3`** — and from the sidecar they are **indistinguishable** from the ~41,700 genuinely uncorrected `3`s |
| **Master's correction to the scope — smaller than "re-run all 46,052"** | Verified by **calling the predicate**, not by reading it: with `RENDERER_VERSION = 4`, `is_complete()` returns **`False`** on a `v3` asset and **`True`** on a `v4` one. **So a bare `python -m metafind.data.renders` regenerates exactly the 43,412 and skips the 2,546 — no flag, no hand-picked list.** The ~1,700 mislabelled ones are inside the 43,412 and simply get re-rendered to the same geometry: **wasted work, not a correctness risk.** ≈ 43,412 / 691 per min ≈ **63 minutes** |
| **Why the `2,546` are safe to keep** | They were written after **both** the bake and the bump, by the code now committed at `138cda4`. `renders.py` is committed and stable — Master checked `git status` is clean on it before relying on this |
| **The underlying defect, and it outlives this incident** | **This pipeline's behaviour can change during a single run and nothing in the output says so.** Only the version bump made it visible at all, and only because it happened to be *late* — had the Engineer bumped the version **with** the bake, the corpus would have been uniformly stamped `4` with 43,412 assets rendered by the old code, **and every gate would have passed.** The version field caught this by accident, not by design |
| **Proposed remedy — the Engineer's, and Master endorses it** | Record the sha256 of the implementation modules into the runlog at run start; each worker verifies on spawn and aborts on mismatch. Cheap, and it converts a silent corruption into a loud stop. **Not implemented** — it is new gate behaviour and belongs after the re-run, not in front of it |
| **Classification** | The mixed corpus is `OBSERVED DATA`. The cause is **operator error, self-reported** — not a tool defect. The reproducibility hole it exposed **is** a tool defect and is registered here separately so it is not written off with the incident |
| **What is needed** | **USER authorisation for the ~63-minute re-run.** No research question is open; nothing is ambiguous; it is an expensive execution and `BLOCKS.md` reserves that |
| **Status** | **`AWAITING_USER_REVIEW`** — authorisation only |
| **Date** | 2026-08-22 |

**Also recorded, because it paid off today:** the `--from-disk` decision to read each sidecar's own
`view_paths` rather than glob the directory means the **88 orphan render directories** — PNGs
written before a quarantine failure — are invisible to every sidecar-driven consumer. A globbing
tool would have scored them.

---

### `DL-017` — the USER delegates these calls to the blocks. **Master stops escalating them.**

| | |
|---|---|
| **Source** | USER, 2026-08-22, verbatim: 「這些是誰的任務 給他決定啊 幹 問他們工程師啊 他們怎麼判斷的」 |
| **Decision** | **The block that owns the node decides.** `DL-013` → ULIP2 Engineer. `DL-016` / `U-20` → ESSGNN Engineer. The USER wants **to be told how they judged it**, not to judge it |
| **Master's error this caused** | Master escalated `DL-013` **four times** and `DL-016` once, correctly by `BLOCKS.md`'s materiality list, and **wrongly in practice** — the USER had no way to adjudicate a `16/37` measurement or a two-week-old registry field, and had already delegated once (the `E-1`…`E-11` series). **Escalating a decision the USER cannot make is not caution; it is handing back work.** Recorded because the rule that produced it is still in `BLOCKS.md` |
| **What this does NOT change** | Delegation of **these adjudications**, not of the gate. Still the USER's: authorisation **before** an expensive run · `MASTER-IMPACTING FINDING`s · anything where continuing would require inventing research-critical information · `DL-009`'s hold on ESSGNN, which is **not** lifted by a decision being delegated into it |
| **Classification consequence** | Whatever the blocks decide is an **IMPLEMENTATION CHOICE**, never a `USER DECISION` — he delegated the call, he did not make it — and never a `PAPER FACT` |
| **Master's job on these** | Re-verify the reasoning rather than rubber-stamp it, then rewrite `DL-013` and `DL-016` from `AWAITING_USER_REVIEW` to the block's decision with its evidence and Master's verification attached |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED, the USER's own wording above |
| **Date** | 2026-08-22 |

**`DL-013` and `DL-016` below are superseded in their `Status` line only.** Both now read
*"delegated to the block under `DL-017`"*. Their findings, evidence and open questions stand
unchanged — the delegation moved **who answers**, not **what was found**.

---

### `DL-016` — `U-20` is marked RESOLVED but was decided by Claude, and four files disagree. **BLOCKER for `n13`.**

**Found by the ESSGNN Engineer, verified by the ESSGNN Reviewer, re-verified by Master. It arrived
in time to stop Master registering a duplicate id for the same knob.**

| | |
|---|---|
| **What `U-20` covers** | Which text encoder produces ESSGNN's node features `t_i`. Today: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, 512-d, shared with the semantic-edge embeddings `e_ij` |
| **The near-miss** | Master had written that `TEXT_ENCODER` *"has no question id"* and intended to register one. **It has one — `U-20`.** A new id would have left **one knob under two ids, one `RESOLVED` and one open** — worse than the gap. Caught before Master acted |
| **Layer 1 — it is marked RESOLVED** | `graph_spec.yaml:1702`, `marked: RESOLVED`, `blocking: false` |
| **Layer 2 — by whom** | `decided_by: 'Claude (recorded with n08 implementation)'`, `decided_at: '2026-08-16'`, `confidence: moderate`. **A research-critical UNKNOWN was closed by an AI in passing, during implementation.** This is `DL-013`'s shape, two weeks earlier, and it has been in force ever since — every `n08` artifact rests on it |
| **Layer 2b — what the entry's OWN text says, and this is the sharpest part** | `U-20`'s `item:` field reads: *"Paper 2.5 says only 'a text-derived feature `t_i` in `R^d`'. It names a frozen text encoder ('e.g., CLIP or BERT') for the semantic **EDGE** embeddings, **not for `t_i`**, and does not say the two are the same encoder."* **The registry records that the paper does not support the choice, and marks the choice RESOLVED on the same line.** Meanwhile `semantic_edges_run.py:98-101` argues it as near-forced — *"Whatever encodes `t_i` must be THIS model"* — which is a **design argument, not paper evidence** |
| **Layer 3 — four files, two states** | `graph_spec.yaml:1702` **RESOLVED** · `01_GRAPH_SPEC.md:611` **UNKNOWN** · `02_BUILD_STEPS.md:299` **「未定 U-20」** · `docs/graph/README.md:17` counts it among the ten resolved |
| **Layer 3b — Master said "no gate catches this". WRONG, and the truth is more useful** | The ESSGNN Reviewer corrected it and Master verified. **The gate exists**, `check_graph.py:320-342`, and its own comment says it was written for exactly this: *"Counting every U- id as UNKNOWN is what let **U-34** sit in the registry as RESOLVED while three documents still called it open."* **This failure mode already happened once and someone built a gate for it.** The hole is two words: `_OPEN_PHRASES` lists 「未解」「尚未確立」「取決於」「仍待」「阻斷級」`unresolved`, `still open`, `open candidate`, `BOTH READINGS REMAIN`, `epistemic candidate` — **and neither `UNKNOWN` nor 「未定」**, which are the two words the conflicting lines actually use. The checker reads `marked: UNKNOWN` as a **field** at `:230-231` and cannot see `UNKNOWN` as a **word** in prose. A gate written for this class, missing this instance, is harder to find than no gate at all |
| **Layer 3c — TWO defects, different sizes. Master conflated them and the Reviewer separated them** | **Defect A — the decider.** Of the ten `RESOLVED` entries, exactly **one** has no USER in `decided_by`: `U-20`, `Claude (recorded with n08 implementation)`. `U-26`'s *"user + external review"* is the USER in different words. Independently audited by the ESSGNN Engineer and re-run by Master: `U-08/08a/08b/08d/08e`, `U-18`, `U-21`, `U-34` all read `Kyzen`. **Scope: 1. Fixing it is reopening a research decision — the USER's.** **Defect B — the documents.** Live files that call a `RESOLVED` id open: **seven** — `U-08`, `U-08a`, `U-08b`, `U-18`, `U-20`, `U-21`, `U-26`, in `01_GRAPH_SPEC.md`'s status table (`:597`–`:619`, second column literally `UNKNOWN` / `UNKNOWN・阻斷`) and `02_BUILD_STEPS.md:299`, `:517`. **Scope: 7. Fixing it is aligning documents with the registry — Master's.** `U-20` is the only id that has both, which is why they looked like one problem. **Fixing `U-20` alone leaves six untouched** |
| **Layer 3d — Master's own dry run was WRONG about the false positives, corrected by the Reviewer's** | Master ran it, counted 8 ids, and concluded *"most are false positives"* from `README.md:17` being the legend row for the 「[未定]」 marker. **Only that one line is a legend.** Excluding it the way `check_graph` itself does leaves **7 ids in genuine status tables** — `01_GRAPH_SPEC.md:597`–`:619` is a real table whose status column says `UNKNOWN`. `U-34` drops out; the other seven stand. **They are real contradictions, not noise** |
| **Layer 3e — a second gate bug, found by the Reviewer, that must be fixed FIRST** | `check_graph`'s id match is `if uid not in line` — **a bare substring with no word boundary.** Verified: `"U-08" in line` is `True` on the `\| **U-08a** \| **UNKNOWN・阻斷** \|` row. Harmless today because no phrase matches anything. **The moment `UNKNOWN` is added it starts producing false failures** — `U-08` convicted by `U-08a`'s line. **Correct order is two steps: word-boundary the id match, then extend `_OPEN_PHRASES`.** Reversed, the gate goes red and blocks everyone with half the reasons fake. Registered as work, not done — changing a gate while `n04` runs and three roles are live is the wrong moment |
| **Layer 4 — the spec's own gate was crossed** | `01_GRAPH_SPEC.md:1123`: `U-12`/`U-20` *"**在 `n13` 實作前必須進 `n09b`／`G6`**"* — must land in the protocol **before `n13` is implemented**, *"otherwise it is the Stage 1 error we just fixed, replayed in Stage 2"*. Measured: `essgnn_arch_protocol.json` has **no `node_feat_dim` and no `edge_feat_dim`** — keys are `architecture_family, coord_feat, decided_at, decided_by, distance, hidden_dim, layer_sharing, mlp_structure, n_layers, pooling, status, use_io_projections`. Both widths are inferred at runtime instead (`stage2.py:315-327` checks `shape[1]` against values taken from the artifacts). **`n13` is 680 lines and complete. The gate was crossed and nothing reported it** |
| **Why reopening is defensible on either ground alone** | **(a)** its decider is Claude, not the USER, at `confidence: moderate`, on a question the entry itself records the paper as not answering; **(b)** it was decided against `n08`'s artifacts, and **`n08` is about to re-run** under the no-old-model-output directive |
| **What Master did NOT do** | **`marked: RESOLVED` is not changed here.** Reopening a resolved research question is a USER decision, not Master's, and editing the field would be resolution-by-mutation. A dispute note is added to the entry pointing at this record; the four-file contradiction is left standing **because picking a side would be the same offence** |
| **Priority — Master accepts the Reviewer's merge** | The ESSGNN priority list's item 2 becomes *"the two things `n08` cannot re-run without: `Q-N08-MODEL` and reopening `U-20`"*. `U-20` is the more fundamental of the two: `Q-N08-MODEL` moves the edge sentences, `U-20` moves **both vectors and their width** |
| **Authority classification** | **IMPLEMENTATION CHOICE recorded as RESOLVED.** Not a PAPER FACT — the entry's own text says the paper does not say it. Not a USER DECISION |
| **Severity** | **BLOCKER for `n13`** · MAJOR for the `n08` re-run |
| **What is needed** | The USER decides whether `U-20` reopens. Master then makes the four files agree with whatever that answer is, and registers the unpaid `n09b`/`G6` protocol debt either way |
| **Status** | **DELEGATED to the ESSGNN Engineer under `DL-017`** — **ANSWERED, see below** |
| **Date** | 2026-08-22 |

#### The block's decision, and Master's verification of it

**ESSGNN Engineer, 2026-08-22: reopen `U-20`, keep CLIP ViT-B/32 / 512-d as a recorded
IMPLEMENTATION CHOICE, and rewrite its justification because the recorded one is false.**
Master re-verified every load-bearing claim by execution before accepting.

| | |
|---|---|
| **A fourth reason to reopen, better than Master's three** | Master's three were procedural — the decider, the self-contradiction, the crossed gate. **The Engineer's is substantive: `U-20`'s recorded justification is refuted by the code.** The stated reason is that `t_i` and `e_ij` both feed `f_h`, so different encoders would put them *"in two unrelated semantic spaces"*. **They are already in different spaces.** Verified: `essgnn.py:456-457` — with `use_io_projections` (protocol: `true`) `embed_in = nn.Linear(in_dim, 128)`, so `t_i` arrives as a **learned** 128-d vector; `:471` — `edge_proj_dim is None` (`:264`, *"the paper has no such layer"*) so `edge_proj = Identity` and `e_ij` arrives **raw at 512**. The shared encoder buys no shared space; the projection removes it. **The justification describes a configuration we are not running** — and under the one where it *would* hold (`use_io_projections=False`), `in_dim == hidden_dim` is forced, so a 512-d encoder is not even legal |
| **Paper, read verbatim** | `2methdology.tex:47` — *"These sentences are then encoded … using a frozen text encoder (e.g., CLIP or BERT), resulting in **edge embeddings e_ij**"*. The clause attaches to `e_ij`, and even there it is **`e.g.`** — an example, not a specification. For `t_i` the paper gives only *"text-derived"* and `∈ R^d`. **PAPER FACT: the encoder for `t_i` is unspecified.** `U-20`'s own `item` text said exactly this and was right |
| **`DL-010` does not reach it — there is no upstream to inherit** | Not case 1. **EGNN has no text features at all**: `/home/kyzen/upstream/egnn/models/gcl.py:242`, `forward(h, edge_index, coord, edge_attr, node_attr)`, where `h` is a discrete node attribute. Master adds a **stronger citation than the Engineer used** — MetaFind says so itself, `appendix.tex:20`: *"While the original EGNN formulation allows the inclusion of edge features in the message function, **these are typically discrete, task-specific features**"*. So this is a **PAPER FACT**, not an inference from upstream code. Text semantics in EGNN is the paper's own claimed contribution (`2methdology.tex:42`, *"extends the EGNN formulation to incorporate semantic edge features"*). **No upstream → this must be a recorded IMPLEMENTATION CHOICE. Leaving it UNKNOWN is wrong (no value); calling it an UPSTREAM FACT is wrong (no upstream)** |
| **The shared-encoder argument, judged on merit rather than on paper silence** | **Convenience, not constraint.** *"Different encoders → unrelated spaces"* is false, per the projection above, and an MLP over heterogeneous inputs is ordinary. The width argument is the one that survives and it is **unrelated to sharing**: `f_h`'s input is `2·128 + 1 + 512 = 769`, of which geometry is a **single scalar**. That argues `e_ij` should not be too wide, never that `t_i` must share its encoder. **Sharing may stay as a choice; that reason must be deleted** |
| ⚠️ **The finding that reorders the work — measured, and Master reproduced it exactly** | `procthor_object_text.json`: **1,467 entries, 93 distinct strings.** 15.8 assets per vector. `Apple_1` and `Apple_10` are byte-identical *"a apple"*. With `h0_mode: semantic`, `t_i` **is** the entire initial node state — so **every ESSGNN node starts from one of 93 possibilities.** **Changing the encoder cannot add information that was destroyed before encoding.** `Q-NODETEXT` therefore precedes `U-20`'s substantive choice: swapping the encoder first is a better lens on the same blurred negative, and costs two `n08` runs |
| **What `n08` needs before it re-runs** | `essgnn_arch_protocol.json` gains **`node_feat_dim`** and **`edge_feat_dim`**, separately, plus the encoder id and version — paying the debt `01_GRAPH_SPEC.md:1123` recorded and nobody paid. Separately, because `stage2.py` already reads them separately and its comment records that conflating them **was** a fixed bug |
| **Stop-safe, correctly invoked** | The Engineer decided *"keep CLIP ViT-B/32"* (current state, defensible reason, changes no artifact) and **refused** to decide *"switch to something else"* — model selection, `BLOCKS.md`-material, and it would move both vectors, `EDGE_DIM`, and Table 2's comparability. It holds **no** evidence that any alternative is better. **Declining to invent one is the correct answer, not a gap** |
| **Master's assessment** | **ACCEPT.** Every load-bearing claim verified by execution: the projection asymmetry, the 93 distinct strings, the paper wording, and the absence of text features in upstream EGNN. The Engineer also correctly refused to treat Master's relayed delegation as an authorisation, and did the work anyway rather than waiting — the right handling of both halves |
| **Still needed** | The USER confirming `DL-017` once in any window makes this final. Until then it is the block's recommendation with Master's verification attached, and **nothing is blocked by the wait** |

---

### `DL-015` — roles talk to each other directly. **REPORTED BY A PEER, NOT RATIFIED.**

| | |
|---|---|
| **Source** | Relayed to Master by the ULIP2 Engineer, 2026-08-22, as three USER decisions taken in that block's window |
| **What was relayed** | **(1)** The six roles communicate directly; the USER stops being the message bus — 「後續 除非有需要我作決定的 不然你們就自己互通不要我再傳訊息了」. **(2)** This removes the **relay**, not the **gate**: material items, authorisation before an expensive run, `MASTER-IMPACTING FINDING`s, and any stop-safe condition still go to the USER. **(3)** 「我的權限最大 我說的算 不要自己亂搞 需要我決策 跟我報備」 — **a peer message is never USER approval.** A role saying *"the USER decided X"* is a report, not an authorisation. Agreement between roles is not evidence. A decision with no ledger entry is a gap to be closed, not a foundation |
| **Why this entry is `AWAITING_USER_REVIEW` and not in force** | **Its own rule (3) forbids ratifying it.** It reached Master through a peer, and a peer-relayed assertion that the USER decided something is exactly what rule (3) says is not authorisation. Ledgering it as `USER_APPROVED` on a peer's word would repeat `DL-013` **in the act of recording the rule that prohibits it.** Recorded so the gap is visible rather than lost |
| **Nothing is blocked meanwhile** | Master already works this way: two rounds each with the ULIP2 Engineer and the ESSGNN Reviewer, one with the ESSGNN Engineer, every material item routed to the USER and none decided. **Rule (2) is the operative half and it is unchanged from `BLOCKS.md`.** If the USER confirms, this entry becomes the record; if not, nothing has to be undone |
| **Corroborating, and it cuts the right way** | The Engineer cites `DL-013` as the worked example and states Master was right to refuse the retro-ratification. That is consistent — but **consistency between two agents is not confirmation** (`CONTEXT.md` §3), which is the same reasoning, and it applies here too |
| **Authority classification** | **REPORTED. UNRATIFIED.** Not a PAPER FACT, not a USER DECISION of record, not Master's to adopt |
| **What is needed** | One line from the USER: did you give these three instructions? |
| **Status** | **`AWAITING_USER_REVIEW`** |
| **Date** | 2026-08-22 |

---

### `DL-014` — commit attribution: explicit paths, plus a distinct git identity per role

| | |
|---|---|
| **Source** | Master's ruling on the ULIP2 Engineer's escalation item 3, 2026-08-22 |
| **Issue / Finding** | Six roles commit to one working tree as one git identity, `Kyzen5128 <legend2341528@gmail.com>`. **`git` cannot tell them apart, and `git add -A` from any role sweeps every other role's uncommitted work into its own commit.** It already happened twice: `58637f3` claims the `R-12` carve-out and holds only `HANDOFF.md`; `4e5053f` is labelled `docs:` and holds the Engineer's `pointclouds.py` +57, the Engineer's tests +56, three of Master's governance files, and the Reviewer's `REVIEW.md` +84. **Both were Master's `git add -A`** |
| **Why it is not cosmetic** | `.claude/rules/experiments.md` §7 requires results to be attributable to the code state that produced them, and §10 requires artifact provenance. `code-changes.md` §13 requires the diff to be inspectable. A dataset-semantics change filed under a documentation message defeats both, **silently and in both directions** |
| **Decision / Resolution** | **Options 1 and 2 together. Option 3 is refused.** **(1)** Every role commits by **explicit path**. `git add -A` and `git commit -a` are prohibited in this repository while more than one role is live — the Engineer and the Reviewer had already adopted this without waiting for a ruling, correctly. **(2)** Every role sets a distinct author on its own commits, per invocation, changing no global config: `git -c user.name="ULIP2 Engineer" -c user.email="legend2341528@gmail.com" commit …`. The email stays the USER's — it is their repository — so only `%an` distinguishes the roles, which is what `git log --author` and every blame tool actually read |
| **Why not option 3 (a worktree per role)** | It would prevent the collision by construction, but the roles are deliberately working on **one** tree: the Reviewer verifies the Engineer's uncommitted edits by executing them, and `R-4` timed its findings against the Engineer's edits landing 42 seconds later. Separate worktrees would break synchronous review, which is the point of having a Reviewer at all. **Rejected on function, not on cost** |
| **Why (1) alone is not enough** | It prevents the next collision but attributes nothing retrospectively. The Reviewer is right that only a distinct identity gives after-the-fact attribution |
| **History is not rewritten** | Two sessions were committing concurrently when the collisions happened and a rebase could have destroyed uncommitted work. The correct commit-to-work mapping is recorded in `c9ef702`'s `HANDOFF.md` entry instead |
| **Authority classification** | **Process, not research semantics.** Master's to decide; recorded here because it governs experiment provenance |
| **Status** | **IN FORCE** from 2026-08-22. Master, the Engineer and the Reviewer have all adopted (1) |
| **Date** | 2026-08-22 |

---

### `DL-013` — the `texture` carve-out reverses a USER-confirmed choice on an inconclusive measurement. **Needs the USER.**

**Raised by Master at integration review, 2026-08-22. Reported as a FINDING; no remedy is proposed and nothing was reverted.**

| | |
|---|---|
| **Issue / Finding** | The USER confirmed **`P3`** — full glTF 2.0 conformance, `COLOR_0 × texture × factor`, ~1,130 geometries — at `R-10`'s ASK, and `U-AA` chose the specification as the authority. **`R-12` then withdrew the `texture` class from `P3`**, which is ~995 of those assets. That narrowing is implemented, tested and shipping as deviation `D-12`. **Master can find no `U-` code approving it** |
| **CHAIN OF CUSTODY — corrected 2026-08-22 after the Engineer supplied what Master could not see** | **The narrowing did not originate in the block.** It arrived in the Engineer's session as an instruction the USER pasted, headed 「三、USER 決定：範圍縮回 A（= P2）」, with the `R-11` reasoning and the two cosine comparisons attached. The Engineer implemented it **as routed**, which is the normal mechanism in this workflow, and did not choose it. **Master's original framing — that a block reversed a USER decision on its own authority — was wrong and is withdrawn.** The Engineer is not at fault |
| **What the defect actually is** | **A relayed assertion that a decision was made is not a ledger entry.** No `U-` code exists, so there is no record either role can point at, and the two of us can only reconstruct it. That is a **process** gap and it is **Master's to close**, not the block's. It is also the exact failure the `HANDOFF.md`-only communication rule exists to prevent, arriving from the one direction the rule does not cover: the USER speaking directly into a block window |
| **The question therefore changed** | It is no longer *"which option do you want"*. It is **"did you tell them to narrow it back to `P2`?"** — answerable only from the USER's own memory, and it must not be reconstructed from either agent's account |
| **What the measurement actually says** | `R-12A`, texture class only, the **full** ULIP overlap n=37: `COLOR_0` off reaches cosine **0.9005**, `P3` reaches **0.8980**. `P3` wins on **16 of 37**. 37 fair coin flips give 18.5 ± 3, so **16/37 is inside one sigma**, and **no paired significance test was run.** The Reviewer states this itself: *"the darkening is certain; 'therefore worse' is not"* |
| **What the narrowing actually rests on** | `R-11` — the USER's ruling that ULIP-2 is the reference architecture and agreement with it is the default. `R-12`'s own text: *"What decides it is `R-11`, not the cosine"* |
| **Why Master does not think `R-11` reaches this** | **`R-8` established that upstream publishes no cloud-colouring procedure at all** — `/home/kyzen/upstream/ULIP` @ `95d480fe` swept for `COLOR_0`, `trimesh`, `.glb`, `sample_surface`: zero hits; `ulip2_source/appendix.tex:10` delegates to OpenShape; OpenShape publishes no converter either. So there is **no upstream *behaviour* to agree with** — only its *artifact*, and the artifact does not discriminate at n=37. `R-11` says to default to agreement where agreement is measurable. Here it is not |
| **What is NOT in dispute** | The wide repair is right and well-evidenced. `P1` — discarding `COLOR_0` — was measurably worst at n=130 (0.8800 against 0.9043 / 0.9004, 27 wins against 103) and is correctly out. `R-13`'s four release conditions all PASS, the new test is verified non-vacuous, and the control group is bit-identical at n=60. **This entry is about one class and one ratification step, not about the `COLOR_0` work** |
| **Why it matters enough to raise** | It buys a **registered deviation from a published specification** (`D-12`) in exchange for a difference its own author calls noise, and it silently narrows a decision the USER made |
| ⚠️ **The window closed while this was being written** | **`n03` has already re-run to completion with `D-12` in force.** Verified by Master 2026-08-22: 46,052 `.npz` + 46,052 `.json` on NVMe, `sampler_version: 6` uniform over a 3,000-file sample, `color0_modulated` 137 true / 2,863 false. **The carve-out is baked into the corpus.** So this is no longer a free decision taken before a run — choosing **B** now costs an `n03` re-run |
| **Revised cost of each option** | **A. keep `D-12`** — zero cost, the corpus on disk is already this. **B. restore full `P3`** — `n03` re-runs. At the measured **897 assets/min** that is **≈51 minutes**, not the 3.3 hours quoted earlier: `n04` is unaffected (it takes colour from the material through pyrender, not through this path) and the output now lands on NVMe |
| **Master's assessment** | The block's work is careful and its reasoning is on the record. This is not a rogue change; it is a material choice ratified one level below where `BLOCKS.md` puts it. **Dataset-preprocessing semantics reaching the whole corpus is USER-material** |
| **Options, for the USER — Master recommends neither** | **A.** Keep `D-12`. The texture class stays unmodulated; the deviation from glTF 2.0 stands as registered. **B.** Restore full `P3` as originally confirmed; `D-12` is withdrawn from the registry and ~995 texture assets are modulated. Either way `R-8`'s prohibition binds: **it may never be written up as "what ULIP-2 did"** |
| **Not verified, and worth knowing before choosing** | `pointclouds.py`'s `GLTF_DEFAULT_BASE_COLOR = 1.0` is marked `INFERENCE` — the glTF schema is not on disk and `material.pbrMetallicRoughness.schema.json` has not been read. **8,853 `gltf_default` assets rest on that value.** Measured support exists (ULIP 35.3% pure-white against ours 35.1%, all-white 19/50 both sides) but that is `OBSERVED DATA`, not the specification |
| **Status** | **DELEGATED to the ULIP2 Engineer under `DL-017`.** No longer awaiting the USER |
| **Date** | 2026-08-22 |

---

### `DL-012` — four deviations registered: `D-9` … `D-12`

| | |
|---|---|
| **Source** | USER decision **`U-Z`** — *"Master registers the four un-id'd deviations on the Integrator's behalf"* — `DL-009` holds `INTEGRATOR` closed and opening a block to write registry lines was not warranted |
| **Issue / Finding** | Four real deviations had **no id in `docs/graph/graph_spec.yaml`**. `check_graph.py:373-383` compares deviation **ids only and never reads the `what:` text** (debt `D-2`/`FU-A`), so all four passed every gate silently. The ULIP2 Engineer escalated this three times and correctly refused to assign ids itself |
| **Registered** | **`D-9`** LVIS category anchoring (`DL-007`) · **`D-10`** contrastive negatives, one GPU's batch against ULIP-2's gathered 512 (`F-N10-1`) · **`D-11`** white render background against upstream's black (`U-W`) · **`D-12`** `COLOR_0` withdrawn from the `texture` class, against glTF 2.0 (`R-12`) · **`D-13`** corpus 46,052 against the paper's *"approximately 48,000"* (`U-01`) |
| **`D-13` — Master overruled its own first reading** | Master initially left `U-01` out, reasoning that a corpus-availability fact is not a *chosen* divergence. **`paper-reproduction.md` §9 says otherwise**: *"Any **intentional or unavoidable** difference … must be recorded as DEVIATION."* Unavoidable is explicitly in scope. Registering it does **not** reopen `O-2` — it is still carried as a stated Table 1 limitation, not resolved. `U-01`'s own stated resolution (*"use `len(manifest)`; record its sha256"*) had **never** been honoured; both sha256 values are now in the registry entry |
| **Latent gate bug found and fixed while doing it** | `check_graph.py`'s pattern was `D-[0-9]` — **single digit**. It could not match `D-10` or above **at all**, so every two-digit id would have read as missing from the mirror tables. Never reachable with eight single-digit deviations; reachable the moment a ninth was added. Changed to `D-[0-9]+`. Classified **bug fix**: the checker now sees what it always claimed to check |
| **Mirrors synchronised** | `docs/graph/README.md`, `docs/graph/02_BUILD_STEPS.md`, root `README.md` — the three the gate reads. Verified: all three list exactly the twelve ids `graph_spec.yaml` declares |
| **`U-01` is deliberately NOT registered** | 46,052 against the paper's *"approximately 48,000"* is carried as a **stated Table 1 limitation**, per the engineer's recommendation `O-2`. It is a corpus-availability fact, not a chosen divergence |
| **Caution carried into `D-12`'s own entry** | `R-12`'s texture carve-out rests on `R-11`'s default-to-upstream-agreement rule, **not** on a significant measurement: 16/37 wins is inside the noise for 37 coin flips and no paired test was run. `R-8` further established upstream publishes **no** cloud-colouring procedure, so there is no upstream *behaviour* to agree with — only its artifact. Recorded in the registry text so a later reader cannot mistake it for a settled result. **See `DL-013`** |
| **Authority classification** | The four deviations are pre-existing facts; **registering them is bookkeeping that follows from `U-Z`**, not a new choice. The regex change is a bug fix |
| **Verification** | `tools/check_graph.py` → 2275 checks, all pass · id sets compared programmatically across all four files |
| **Status** | **`USER_APPROVED`** — executed under `U-Z` |
| **Date** | 2026-08-22 |

**Debt `D-2` / `FU-A` is NOT discharged.** The checker still compares ids and never reads `what:`,
so a deviation whose *description* has gone false still passes silently. Unassigned.

---

### `DL-011` — the ULIP2 session's USER decisions, ratified as a block: `U-A` … `U-AC`

| | |
|---|---|
| **Source** | USER, in conversation with the ULIP2 Engineer and Reviewer, 2026-08-22. Recorded verbatim in `workflow/blocks/ULIP2/HANDOFF.md`, which both roles asked Master three times to ledger |
| **Why one entry and not thirty** | They are one continuous decision thread from one session on one milestone. Splitting them would imply they were weighed separately. **The HANDOFF entries are the record**; this entry ratifies them and pulls out the ones with project-wide reach |
| **Status** | **`USER_APPROVED`** — the USER's own wording is in `HANDOFF.md` and governs |
| **Date** | 2026-08-22 |

**Project-wide, not ULIP2-local:**

| | Decision |
|---|---|
| **`U-A`** | **No post-processing repairs.** A defect is fixed at its source, once, properly. The codebase is meant to be used by other people. 「要修就一次修好」. **A standing project rule, not an n04 decision** |
| **`U-O`** | MetaFind states it → follow MetaFind. MetaFind is silent → follow ULIP-2. **This is the same rule the USER gave Master independently as `DL-010`**; the two converge and `DL-010`'s three-case discriminator is the operative form. A choice made under it is an `IMPLEMENTATION CHOICE` with upstream provenance, **never a `PAPER FACT`** |
| **`U-Z`** | Master registers the un-id'd deviations while `INTEGRATOR` is held → executed as **`DL-012`** |
| **`U-Y`** | The multi-day `n05` run carries a **circuit breaker** whose threshold is **measured by the bake-off**, never invented now. Asked three times before it was answered |

**Corpus correction — `n03` / `n04` re-run AUTHORISED, overriding `BLOCK.md` §7:**

| | Decision |
|---|---|
| **`U-B`** | **`n04` re-render authorised.** The `+Z`-up camera is fixed at source and the corpus regenerated. A 100-asset A/B ran first |
| **`U-G`** | The 180° yaw is fixed **at the mesh-load layer**, so `n03` (46,052) re-runs too |
| **`U-W`** | Background reverts to **white** → deviation `D-11` |
| **`U-X`** | `ORTHO_HALF_WIDTH` reverts to **1.10** |
| **`U-N`** | All three render differences fixed: orbit axis, background, framing |
| **`U-AA`** | `COLOR_0` settled on the glTF 2.0 specification rather than by buying ~20 GB of further ULIP shards |
| **`U-H`** | The white-point-cloud question `F-N03-1` is settled by differential against ULIP's clouds **before** the re-run, so any bug is fixed in the same pass |

> **`BLOCK.md` §7's *"renders are read-only, no re-render"* is SUPERSEDED for `n03`/`n04`.**
> The prohibition rested on `evidence/n05_annotation_defect.md` Evidence 2, which measured
> **occupancy** (`correlation = +0.054`). **Orientation was never measured.** The prohibition was
> not wrong when written — it answered a different question.

**Annotation and training protocol:**

| | Decision |
|---|---|
| **`U-E`** | **11 views stands.** The engineer objected once to a change to 12; MetaFind states 11 twice. Objection accepted |
| **`U-I`** / **`U-P`** | Bake-off sample is **100 assets per arm**, 4 strata × 25: ordinary · rare LVIS class · low visibility · extreme aspect |
| **`U-Q`** | **20 assets hand-adjudicated by the USER**, shared across all arms. **The only ground truth in the whole bake-off** |
| **`U-L`** | **Two-turn identity check** — turn 1 blind, turn 2 with the LVIS anchor. `identity_confirmed` becomes **computed**, not asked. This is a direct answer to `IC-1`, the rubber-stamp risk Master raised |
| **`U-M`** / **`E-10`** / **`E-11`** | Multiple description candidates re-ranked by an **independent** CLIP (`openai/clip-vit-large-patch14`), never `ViT-bigG-14`, which `n06` encodes with — ranking and encoding with one model is circular |
| **`U-T`** | **Fusion default becomes `transformer`.** `fusion.py` recorded `U-13` as *"the paper never says which"* — **it does**: `3experiments.tex:143`, *"the final selected Transformer"*. A false `UNKNOWN` corrected back to a **PAPER FACT** |
| **`U-U`** | **Do not invent `lr` / `epochs` / `batch_size`. Measure them.** 72/8/20, the paper's 20% test untouched. The paper has **no implementation-details paragraph at all** |
| **`U-V`** | `n06` stores all 11 per-view vectors alongside the mean. GPU cost zero, ~2.6 GB storage |
| **`U-S`** | Widen the validator's dimension floor. **103 of 45,955** currently have an empty feasible height band — posters and decals, real data |
| **`U-AB`** | **`W-6` — the missing `D0-010` evidence audit — must complete before `M2`** |

**`Q-CATEGORY` is discharged, not open.** `R-6` established it and `DL-007`'s `D0-010` are the
same question, listing the identical four options. The true state is **decided, implemented,
unaudited** — not *"no investigation has been done"*. `U-AB` supplies the missing audit.

---

### `DL-010` — where MetaFind is silent on a component it did not modify, the official upstream implementation **is** the reference

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「它是原架構如果有地方不詳細應該是要以他們兩個的原程式碼為主啊 如果 metafind 沒有改動的話」 |
| **Issue / Finding** | Blocks were reading the project's upstream cautions as a blanket *"reproducing MetaFind means we may not consult ULIP-2 or EGNN"*. That is a misreading, and it is expensive: it turns answerable questions into `UNKNOWN`, and pushes engineers toward inventing a value when the authoritative one is sitting in `/home/kyzen/upstream/` |
| **Decision / Resolution** | **Three cases. Decide which one you are in before you cite anything.** **(1) MetaFind SILENT + component inherited unmodified → the official upstream implementation IS the reference.** Use it. Classify **UPSTREAM FACT** and state the inheritance basis. Do **not** record `UNKNOWN`, and do **not** invent. **(2) MetaFind SPEAKS but is ambiguous or self-conflicting → upstream supplies the VARIANT LIST, never the answer.** Escalate to the USER. **(3) MetaFind MODIFIED the component → upstream says nothing about the modified part.** No inheritance |
| **Precedent, already in force — this is not new policy, it is policy being applied consistently** | **Case 1 is `U-34`**: the CLIP freeze scope. MetaFind never states a per-module freeze policy; ULIP-2 §3.3 says *"freeze it during pre-training"*; MetaFind explicitly builds on ULIP-2 and never says it changed that. Resolved **frozen** on exactly this reasoning, `decided_by: Kyzen`, 2026-08-16. **Case 2 is `U-35`**: MetaFind says only *"approximated using MLPs"*; EGNN Appendix C gives **three different** MLP shapes and our `f_h` matches none of them. Left `UNKNOWN`, recorded as a protocol field — *"EGNN Appendix C supplies the VARIANT LIST, not the answer"* |
| **This does NOT weaken `DL-004`** | `DL-004`'s prohibition — *"upstream EGNN settles it" may not be used as paper-interpretation authority* — is **case 2, and it stands unchanged**. MetaFind **does** speak there: `2methdology.tex:54` states `f_x → R³` and `appendix.tex:23` claims equivariance for any orthogonal `Q`. Citing EGNN's scalar to overrule what MetaFind actually wrote is still forbidden. **`DL-010` governs silence. `DL-004` governs ambiguity. They do not overlap** |
| **Authority order is unchanged** | `CLAUDE.md` §3 already ranks *"original upstream papers and official upstream implementations"* **third — above** project audits, above the graph spec, above this repository's own code. `DL-010` does not promote upstream; it stops blocks from demoting it below reasoned guesswork |
| **Sources** | `/home/kyzen/upstream/ULIP` @ `95d480f` · `/home/kyzen/upstream/egnn` @ `e9ca6c0` · `docs/paper/{ulip2,egnn,idesign}_source/`. All verified present 2026-08-22 |
| **Still binding, unchanged** | Silence is never endorsement — judge whether the method works on its own evidence. An upstream detail is MetaFind-relevant only where MetaFind adopts, inherits, references or depends on it (`.claude/rules/paper-reproduction.md` §4). **An UPSTREAM FACT is never a PAPER FACT** |
| **Authority classification** | **USER DECISION** — paper-interpretation policy |
| **USER final decision** | **`APPROVE`** — 2026-08-22, the USER's own wording above |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

---

### `DL-009` — execution order: `ULIP2` runs to completion before `ESSGNN` opens

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「我想先完成 ULIP2 再接續完成」 |
| **Decision / Resolution** | **Sequential, not parallel.** `ULIP2` is the only open block. `ESSGNN` and `INTEGRATOR` are `ON HOLD` and unstaffed; the USER opens them |
| **Consequence Master carries, not a block** | `Q-TOWER` and `Q-BUILDMODEL` are Integrator-owned but **block ULIP2's own n09 splits**. While `INTEGRATOR` is on hold, **Master holds those two seams** and brings them to the USER when ULIP2 reaches n09. No block silently absorbs an Integrator question |
| **Consequence, cost — recorded so it is not rediscovered as a surprise** | Table 1 and Table 2 are ~10 nodes with **zero code** (verified 2026-08-22: no `metafind/eval/` directory exists at all), and **none of them needs a trained model** to be designed and unit-tested. Serialising means that work does not begin until ULIP2 finishes. **This is the USER's accepted trade** |
| **Authority classification** | **USER DECISION** — project sequencing |
| **USER final decision** | **`APPROVE`** — 2026-08-22, the USER's own wording above |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

---

### `DL-008` — the annotator is chosen by a lightweight bake-off. **Supersedes the model lock in `DL-006` (2026-08-21).**

| | |
|---|---|
| **Source** | USER, 2026-08-22: 「LLM 選用我想在標註時跑個輕量測試作比較 我再決定 到底要選誰」 |
| **What is approved** | **the procedure only** — a lightweight sample run per candidate, then the USER picks the winner |
| **What is NOT approved** | **the winner.** No model is selected by this entry. `annotate_run.py:72`'s hardcoded `MODEL_ID = "/mnt/data1/kyzen/models/Qwen3.8-27B"` is **leftover state from the superseded entry, not a live decision** |
| **Supersedes** | `DL-006` **(2026-08-21)** — "the n05 annotation model is `Qwen3.8-27B`". That entry is **not deleted**; it was true when written. The model choice is **reopened** |
| **Measured candidate state — Master, 2026-08-22, `/mnt/data1/kyzen/models/`** | `gemma-4-31B-it-qat-w4a16` — **22,188 MiB**, `config.json` `quantization_config.quant_method: compressed-tensors`, official quantization-aware-trained, 0 `.incomplete`. **READY** · `gemma-4-12B-it` — **22,812 MiB**, bf16, no quantization at all, 0 `.incomplete`. **READY** · `Qwen3.8-27B` — **55.56 GB bf16**, 18/18 shards verified against `model.safetensors.index.json` `total_size`. **The w4a16 build named in `ULIP2/BLOCK.md` §11 does not exist on disk and has never been produced.** NOT READY |
| **Consequence for the deviation registry — Master routes, no block decides** | `graph_spec.yaml:133` states `D-2` as *"Qwen3.8-27B replaces GPT-4o for ASSET ANNOTATION (n05)"*. **If any other arm wins that text becomes false**, and `check_graph.py:373-383` compares deviation **ids only, never the `what:` text**, so a falsified description passes every gate silently. Registered debt `D-2`/`FU-A` is therefore no longer theoretical |
| **Standing, unchanged** | **The annotator is not GPT-4o whichever arm wins.** Deviation `D-2` is re-pointed, never discharged, and must never be written up as paper-faithful. LVIS category anchoring remains a separate DEVIATION (`DL-007`) |
| **Authority classification** | **USER DECISION** on procedure. The eventual winner is an **IMPLEMENTATION CHOICE backed by a measurement on this hardware and this sample** — never a general claim that one model is better than another |
| **USER final decision** | **`APPROVE`** — the procedure, 2026-08-22. The winner is **PENDING USER** |
| **Status** | **`USER_APPROVED`** for the procedure · the model itself is **REOPENED, PENDING USER** |
| **Date** | 2026-08-22 |

---

### `DL-006` — `D0-003` resolved: the 3 legacy-v1 residuals are deleted and re-annotated under v5

| | |
|---|---|
| **Source** | USER, 2026-08-22, following the 「不准使用舊模型產出的東西」 directive |
| **Issue / Finding** | Three uids carried `prompt_version 1` records long after the rest of the corpus moved to v3: `6c7db00cc164467ebac356a5ca67368b` (pole dancer), `8a0192eee6fb4140bb3e9696b3dbae5a` (pinecone), `a397b648d6eb48d7909d1ee11235e78f` (train). `D0-003` asked whether to admit, drop, or re-annotate them, and the question was **hard-blocking `D3`**: `splits.py:169-171` admitted all 45,955 while `stage1.py:109` loads the `.npz` with no existence guard, so Stage 1 would raise `FileNotFoundError` mid-epoch |
| **Why they were stuck — established 2026-08-22, not previously recorded** | **They are not broken assets.** Renders and point clouds exist for all three. They failed the v3 re-annotation on a **validator rule**: `quarantine_n05_annotate.jsonl` records `terminated_by: repair_budget`, `failure_class: MODEL_RECOVERABLE`, and for `6c7db00c…` the exception is verbatim ``synset` = 'pom.pom.n.01' is not a WordNet id of the form "lemma.n.01"`. The model invented a malformed synset, exhausted the repair budget, and the old v1 record was never overwritten |
| **Why the question dissolves under v5** | **v5 does not ask the model for `synset` at all.** Design Decision 4 replaced it with a deterministic lookup over the 1,156-term LVIS vocabulary (`metafind/data/lvis_synsets.json`, built and cross-checked against detectron2, 0 invented entries). The exact failure mode that produced these three residuals **no longer exists in the v5 pipeline** |
| **Evidence references** | `data/outputs/logs/quarantine_n05_annotate.jsonl` (3 of its 5 records) · `metafind/data/annotate.py` `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` / `SCHEMA_VERSION 3` · `metafind/data/lvis_synsets.json` · `D14/ESCALATION.md` "synset table 1,156 / 1,156 … 0 invented entries" · the three records read verbatim before deletion |
| **Decision / Resolution** | **Delete them.** They are re-annotated under v5 alongside the other 45,952, with no special handling. If v5 fails on any of them, they are quarantined by the ordinary path like any other failure — there is no longer a legacy schema to preserve |
| **Authority classification** | **USER DECISION.** The supporting facts are OBSERVED DATA (the quarantine records) and OBSERVED IMPLEMENTATION (v5's synset lookup). **This is not resolution-by-mutation**: the deletion follows from the v5 design the user already ratified, and the reason the residuals existed was established from evidence *before* deleting them |
| **USER final decision** | **`APPROVE`** — 2026-08-22 (「如果可以就刪」, conditional on the question being answerable in the new work; Master established that it is, and recorded the basis above) |
| **Affected components** | `data/outputs/annotations/` (now **0 files**) · `D0-003` closed · `D3`'s hard blocker cleared · `D2`'s corpus denominator becomes 45,955 uniformly · `annotation_provenance.json` must be rebuilt · `MASTER.md` §8, `INDEX.md` Decision Queue, `CONTEXT.md` §5/§6/§7 all still describe `D0-003` as UNRESOLVED and must be corrected |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-22 |

**Supersedes** the "explicitly NOT resolved: `D0-003`" clauses in `DL-002` and `DL-003`, and the
`legacy_v1_residual_unresolved` state in `data/outputs/annotation_provenance.json`. Those entries
are **not deleted** — they were true when written.

**`AC-1` no longer has a subject.** It protected an accepted legacy corpus from accidental
re-annotation. That corpus was deliberately deleted on 2026-08-22, so AC-1.a ("a bare
`annotate_run` queues 0 records") is now expected to be **false** — a bare run should queue the
whole corpus, because that is the intent. `DL-003-A1` must be rewritten to say so rather than
re-proving a guarantee whose purpose has ended. **Do not treat a bare run queuing 45,955 as a
regression.**

---

### `DL-007` — n05 v5 anchors object identity on the Objaverse-LVIS label. **This is a DEVIATION.**

**Registered by Master 2026-08-21 during the re-initialization audit. It had no ledger entry, which was a gap: it is the most scientifically material change made to the pipeline this week.**

| | |
|---|---|
| **Source** | `D14_n05-v5-reannotate`, design `workflow/n05_v5_design.md`, approved by the user 2026-08-21 |
| **Issue / Finding** | n05 asked the VLM to *identify* objects from 224×224 renders it often cannot read, so it collapsed onto high-frequency priors. **LVIS ground truth:** 1,156 distinct categories, top-20 share **7.1%**. **Qwen output:** 3,036 categories, top-20 share **22.3%** — 3× more concentrated than the truth. `toy` is the single most common answer at 1,542 (3.4%), a word `build_prompt` **explicitly forbids**. Agreement with LVIS: category matches 29.0%, LVIS word appears in the description 28.4%, **neither 67.8%** |
| **Why it is not a `category`-only defect** | `build_prompt` says *"Estimate its size from what kind of object it is, not from the picture"* — dimensions and placement are **by design** derived from the category. Observed: `LVIS pinecone → "a dark brown hairbrush"`, `LVIS mug → "a cylindrical pillow"`, `LVIS truck → "a modern air conditioner unit"`. **A wrong category is a wrong record, not a wrong field** |
| **Evidence references** | `workflow/MIF_n05_diagnosis.md` · `workflow/MIF_n05_category_vs_lvis.md` · `objaverse_lvis_metadata.json → value_to_key_mapping` (46,207 uid→category, fetched by `download.py:70`, **read by nothing**) · `annotate.py:366` `build_prompt(n_views)` receives only `n_views` · `annotate.py:510` `validate_annotation()` never checks semantic correctness |
| **Resolution not driven by resolution** | correlation(best-view occupancy, LVIS agreement) = **+0.054**, agreement flat at ~28–30% across a ~100× range of effective object pixels. **Re-rendering would not have fixed it** and is explicitly out of scope |
| **Decision / Resolution** | n05 v5 supplies the LVIS category to the model as the anchored identity; the model may **refine downward** but not replace laterally; it also emits `identity_confirmed`. `PROMPT_VERSION 5`, `SCHEMA_VERSION 3`, `VALIDATOR_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7` |
| **Authority classification** | **DEVIATION.** The paper has the VLM *generate* the category: `2methdology.tex:28` and `neurips_2025.tex:100` both say GPT-4o produces the structured annotations; Figure 2's caption (`2methdology.tex:24`) says the VLM *generates* attributes including category. **Feeding the dataset label in is a departure and must never be described as paper-faithful.** `D14/TASK.md` `R-E` binds |
| **What remains UNRESOLVED** | **`D0-010` has not been researched** — its §6–§11 are empty. The choice between *prompt hint* / *hard value* / *cross-check* / *record-only* was made by design ratification, **not** by a completed evidence audit. Also unresolved: whether anchoring merely substitutes LVIS's own errors for Qwen's, and whether `identity_confirmed` detects that or rubber-stamps the anchor (`IC-1`) |
| **USER final decision** | design **approved** 2026-08-21; **the deviation registration itself is pending** and reaches the user through `D14`'s acceptance brief |
| **Affected components** | `annotate.py`, `annotate_run.py`, `lvis_synsets.json`, the entire annotation corpus, every Stage 1 text embedding, both tables' comparability with the paper |
| **Status** | `AWAITING_USER_REVIEW` |
| **Date** | design approved 2026-08-21 · registered here 2026-08-21 |

**Registry gap, open:** `docs/graph/graph_spec.yaml` carries **no deviation entry** for LVIS anchoring. And `tools/check_graph.py:373-383` compares deviation **ids only, never the `what:` text** (`FU-A`), so a missing or falsified deviation description passes every gate silently.

---

### `DL-006` — the n05 annotation model is `Qwen3.8-27B`

**Registered by Master 2026-08-21 during the re-initialization audit. The user's decision (U-6) was made in conversation on 2026-08-21 and had no ledger entry.**

| | |
|---|---|
| **Source** | User decision **U-6**, 「走本地 Qwen3.8-27B」, recorded in `workflow/tasks/D14_n05-v5-reannotate/USER_DIRECTIVES.md` |
| **Issue / Finding** | n05 ran `Qwen/Qwen2.5-VL-7B-Instruct` as a stand-in for GPT-4o. A 7B model was a candidate cause of the identity-collapse defect in `DL-007` |
| **Decision / Resolution** | The annotation model becomes local **`Qwen3.8-27B`**, weights at `/mnt/data1/kyzen/models/Qwen3.8-27B`. Enacted at `annotate_run.py:72`. Master verified 2026-08-21: **18/18 shards, 56 GB, download complete** |
| **Authority classification** | **USER DECISION.** The substitution itself remains a **DEVIATION** — the paper says GPT-4o twice. Recorded under the split `D-2` (see `DL-005`). Reaching GPT-4o would narrow `D-2`, never discharge it |
| **Correction of record** | `D14/TASK.md` §7 originally justified its model prohibition with *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from the code comment at `annotate_run.py:71`. D14's finding F-2 shows OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **The conflict is UNRESOLVED, not resolved**, and the API has never been exercised. It must not be restated as settled |
| **UNRESOLVED and material** | **The model has never been loaded.** 56 GB at bf16 does not fit the RTX 5090's 32,607 MiB, so **quantization is required and has not been tested.** Quantization changes annotation quality, which makes it an **experimental condition**, not an engineering detail. **No Phase 3 runtime estimate is evidence-backed** — the "~19.6 GPU-h" figure belongs to the 7B model. Tracked as blocker **R2** (`MASTER.md` §11) |
| **USER final decision** | **`APPROVE`** — the model choice (U-6). The quantization condition and the runtime estimate are **not** covered by it |
| **Affected components** | `annotate_run.py:72` · deviation `D-2` · `D14` Phase 2 and Phase 3 · every annotation record |
| **Status** | **`USER_APPROVED`** for the model choice; the quantization condition is **OPEN** |
| **Date** | 2026-08-21 |

---

### `DL-005` — deviation `D-2` split into `D-2` (annotation) and `D-8` (scene judging)

| | |
|---|---|
| **Source** | Master, executing D14's escalation **P-2**, 2026-08-21 |
| **Issue / Finding** | `graph_spec.yaml` recorded one deviation — *"Qwen2.5-VL replaces GPT-4o for annotation **and** scene judging"*. After user decision **U-6** the annotation model became `Qwen3.8-27B` while the judge (n17) stayed `Qwen2.5-VL`. **One id would denote two different substitutions** |
| **Evidence references** | `graph_spec.yaml:130` (pre-split) · `annotate_run.py:72` (`MODEL_ID` already changed) · `2methdology.tex:28`, `neurips_2025.tex:100` (paper says GPT-4o) |
| **Decision / Resolution** | **Split, not rewrite-in-place.** `D-2` = Qwen3.8-27B for **asset annotation (n05)**. New `D-8` = Qwen2.5-VL for **scene judging (n17)**. Deviation count 六項 → 七項, synchronised across 5 files: `graph_spec.yaml`, root `README.md`, `docs/graph/README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`. `check_graph.py` → **2275 checks, all pass** |
| **Why split rather than rewrite** | One id covering two different substitute models is the exact ambiguity that produced this escalation, and the two will diverge further — n17 may change model later. D14 recommended splitting; Master agreed |
| **Authority classification** | The model change itself is a **USER DECISION** (U-6). This split is **bookkeeping that follows from it** — it records an existing fact accurately, it does not make a new choice |
| **USER final decision** | **pending** — reaches the user through D14's acceptance brief |
| **Affected components** | `docs/graph/graph_spec.yaml`, both `README.md`, `02_BUILD_STEPS.md`, `01_GRAPH_SPEC.md`; the reproduction report's deviation section |
| **Status** | `AWAITING_USER_REVIEW` — executed, not yet ratified |
| **Date** | 2026-08-21 |

**Correction carried in the same edit.** `D-2`'s stated reason was *"GPT-4o is unavailable"*. **Master wrote that without verifying it**, inferring it from a code comment. D14's finding F-2 showed OpenAI's official deprecation page does not list base `gpt-4o` and schedules `gpt-4o-2024-05-13` for shutdown **2026-10-23**, while secondary sources disagree. **That conflict is UNRESOLVED.** The registry now records availability as UNRESOLVED rather than as established, in `graph_spec.yaml`, both READMEs, and `TASK.md` §7.

**Open gate weakness, registered not fixed:** `tools/check_graph.py:373-383` compares deviation **ids only** — regex `\|\s*\*\*(D-[0-9])\*\*` — and never reads the `what:` text. **A deviation whose description has gone false passes every gate silently.** That is how `D-2` stayed wrong. Found by D14. Not in D14's scope; unassigned.

---

### `DL-004` — MetaFind §2.5 `f_x → R³`: verdict and implementation

| | |
|---|---|
| **Source** | `D0-009_essgnn-fx-codomain` |
| **Issue / Finding** | MetaFind states `f_x: R^(2d+1+e) → R³` (`2methdology.tex:54`) and claims equivariance for **any orthogonal** `Q ∈ R^{3×3}` (`appendix.tex:23`), but **never defines the `·`** in the coordinate update (`2methdology.tex:52`). Master confirmed the silence exhaustively: zero hits for `hadamard` / `element-wise` / `inner product` / `dot product` / `contraction` across all five `.tex` files |
| **Evidence references** | `2methdology.tex:52`, `:54` · `appendix.tex:23`, `:29`, `:53`, `:61`, `:68` · `essgnn.py:311-312`, `:353`, `:358-359` · `C_PAPER_CONTRADICTIONS.md:114` (C3, **SEVERE**) · decision file §6–§11 · Codex round 1 (its BLOCKER changed the verdict) |
| **Verdict** | **`PAPER-AMBIGUOUS`.** MetaFind alone does not uniquely determine how the `R³` output participates in the coordinate update. **This must never be rewritten as "MetaFind explicitly got it wrong."** |
| **Decision / Resolution** | **Option A** — retain the scalar `f_x` coordinate multiplier. **`essgnn.py` behaviour is NOT modified**; what changed is the authority classification |
| **Authority classification** | `f_x → R³` stated = **PAPER FACT** · the operator is undefined = **PAPER FACT (as to silence)** · `h` invariance = **PAPER FACT** (`appendix.tex:29`, `:68`) · **scalar `f_x` = USER-RATIFIED IMPLEMENTATION CHOICE under a PAPER-AMBIGUOUS specification.** Not a PAPER FACT |
| **Rationale, recorded** | Not chosen because upstream EGNN is more sensible. Chosen because the operator semantics are undefined; Hadamard closes dimensionally but breaks the paper's own general-orthogonal equivariance claim; `R³` + contraction invents an operator MetaFind never defines; the scalar preserves equivariance and invents nothing |
| **Binding prohibition** | **"upstream EGNN settles it" may no longer be used as paper-interpretation authority anywhere in this project.** `E_GRAPH_REVALIDATION.md:175` must be corrected |
| **USER final decision** | **`APPROVE` with implementation decision A** |
| **Affected components** | `essgnn.py` (classification only, no behaviour change) · `docs/audit/` C3 · `docs/graph/` U-26 · `CONTEXT.md` §5 · Stage 2 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | 2026-08-21 |

**`MIF-1` REJECTED as a blocker.** D0-009 asked to be gated on `D0-004`'s "unresolved `h` semantics". Master rejected it: `appendix.tex:29` **assumes** `h^0` invariant and `:68` concludes the feature update invariant — `h` invariance is not an open question in MetaFind. `essgnn.py:353` feeds `f_h` only `h`, `radial`, and `edge_attr`, all invariant. `D0-004` concerns which layer's `h` reaches `f_x`; both are invariant. Option B's conflict is independent of `h`.

**Follow-ups registered, none blocking:**

| ID | Item |
|---|---|
| MIF-3 | `F_CODE_GRAPH_CONSISTENCY.md` — its `CONSISTENT` column is unlabelled and appears to mean "code matches the graph spec", not "code matches the paper". Terminology ambiguity |
| MIF-4a | `2.2e-16` vs `0.43` — **must not be described as a repo-verified measurement.** Cannot be reproduced: the `R³` variant does not exist in code |
| MIF-4b | No `R³`-variant equivariance test exists. **Narrower than reported:** `test_se3_equivariance` runs at `n_layers=3` (`test_essgnn.py:102`, `:112`) and `test_equivariance_negative_injection` (`:129`) proves it non-vacuous |
| MIF-5 | `normalize_coord_diff` has no MetaFind authority. `essgnn.py:189` defaults `False`, zero current impact. **Independent candidate — not D0-009's to touch** |
| — | Correct `E_GRAPH_REVALIDATION.md:175`'s `[UPSTREAM] settles it` |

---

### `DL-003-A1` — PREPARED AMENDMENT to `DL-003` / AC-1, to land WITH D14 Phase 3

**Status: `PREPARED, NOT IN FORCE`.** Drafted by Master 2026-08-21 at D14's request (P-5), so the
amendment lands **with** Phase 3 rather than being retrofitted after it. `DL-003` stands unchanged
until Phase 3 completes.

**Why an amendment is needed.** `DL-003` records the corpus as **accepted legacy-v3 validated under
`VALIDATOR_VERSION 2`**, and the provenance registry declares that as a population of **45,952**.
D14 Phase 3 re-annotates those same 45,952 under `PROMPT_VERSION 5` / `VALIDATOR_VERSION 3` /
`SCHEMA_VERSION 3`, contract `metafind_annot_v5@f5b2bfb2e5f61fe7`. **The moment Phase 3 finishes,
`DL-003`'s description of the corpus becomes false.**

### What changes

| | before Phase 3 | after Phase 3 |
|---|---|---|
| 45,952 records | `accepted_legacy_v3`, `VALIDATOR_VERSION 2` | annotated under the **current** contract `metafind_annot_v5@…` |
| how they satisfy AC-1.a | by **registry declaration** | by `is_complete()` — they carry the current `annotation_contract` |
| 3 legacy-v1 residuals | `legacy_v1_residual_unresolved` | **unchanged.** `D0-003` still UNRESOLVED |
| registry `accepted_legacy_v3` population | 45,952 | **0** — the state becomes historical |

### What must NOT change

- **AC-1.a still holds: a bare `annotate_run` queues 0 records TOTAL.** After Phase 3 the 45,952
  satisfy it through `is_complete()` instead of through the registry, and the 3 residuals still
  satisfy it through their declaration. **The guarantee is identical; only the mechanism moves.**
- **AC-1.b still holds:** `--force` still re-annotates; the named-migration form still works.
- **AC-1.c still holds:** three states remain explicit, never inferred from a missing field.
- **`D0-003` remains UNRESOLVED.** Nothing in Phase 3 or this amendment resolves it, and the
  3 residuals must be byte-identical at Phase 3's end.

### Required of D14 at Phase 3

1. Update the registry so `accepted_legacy_v3` no longer claims a population that has moved on.
   **Do not delete the state** — mark it historical, with the date and the contract that superseded it.
2. Re-prove AC-1.a **after** the registry update: bare run queues **0 TOTAL**, 0 legacy-v3, 0 residuals.
3. Re-prove AC-1.b: `--force` still yields a non-empty work list.
4. Confirm the 3 residuals' declaration survived untouched.

**If AC-1.a cannot be re-proved after the registry update, that is a `MASTER-IMPACTING FINDING`, not
a registry-editing problem.** The protection is the point; the registry is only how it is expressed.

### Authority

The re-annotation itself is a **USER DECISION** (U-6, U-10, and the design ratified 2026-08-21).
This amendment is **bookkeeping that follows from it** — it records a consequence, it does not make
a new choice. It still reaches the user through D14's acceptance brief.

**Superseded on landing:** `DL-003`'s "legacy-v3 validated under `VALIDATOR_VERSION 2`" description
of the 45,952, and `AC-1.e`'s instruction to record them as such. **`DL-003` is not deleted** — it is
the true record of what was accepted on 2026-08-21.

---

### `DL-003` — Stage 1 protocol refresh, τ = 0.5, and legacy-corpus rerun protection

| | |
|---|---|
| **Source** | `D2a_stage1-protocol-refresh` |
| **Issue / Finding** | `load_protocol()` refused the on-disk protocol after D10's B-2. τ = 0.5 had no code path (`resolve_stage1.py` hardcoded 0.07, 0 CLI flags). And `annotate_run.is_complete()` keyed on a contract **no** record carried, so a bare invocation would have queued **45,955** records — overwriting a corpus the user decided to preserve, and resolving `D0-003` by mutation |
| **Evidence references** | Master re-ran, read-only: `build_work_list(force=False)` → todo **0**; `force=True` → **45,955**; state histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}`; `load_protocol()` → `metafind_v2_cm@8e4b1fcc66c7f48c`; artifacts `0.5` / `False`; `prompt_version {3: 45952, 1: 3}`; 547 passed · 2275 checks · PRE-FLIGHT PASSED |
| **Decision / Resolution** | Accept the task. τ = 0.5 with `learnable_temperature: false` written through n05b; protocol refreshed to the ratified serializer; AC-1 satisfied via a **declared registry** (`data/outputs/annotation_provenance.json`) plus a relocated work-list predicate; the 45,952 formalized as accepted legacy-v3 under `VALIDATOR_VERSION 2`; the 3 residuals explicitly `legacy_v1_residual_unresolved` |
| **Authority classification** | τ = 0.5 = **PAPER FACT** (`3experiments.tex:15`) · `learnable_temperature: false` = **USER-RATIFIED IMPLEMENTATION CHOICE** on a strongly-supported inference — the paper uses "learnable" for `f_h`/`f_x`/λ but calls τ a "temperature hyperparameter" twice, and never states τ is non-learnable · AC-1 mechanism = IMPLEMENTATION CHOICE within the approved menu · registry choice, legacy-v3 formalization = **USER DECISION** |
| **Scope deviation — ratified** | **MIF-2.** `n05b` writes **three** artifacts (`resolve_stage1.py:660-662`); the contract declared two. `variant_registry.json` was rewritten. Master proved the rewrite byte-identical (`VARIANTS` absent from the diff; `_write()` is a deterministic `json.dump(indent=1)`). Root cause: **Master's contract under-declared the write surface**, not executor conduct. **USER RATIFIED 2026-08-21.** The contract was **not** retroactively edited |
| **USER final decision** | **`APPROVE`** — task accepted, MIF-2 ratified |
| **Affected components** | `resolve_stage1.py` · `annotate_run.py` · `annotate.py` · tests · `stage1_hyperparameters.json` · `stage1_encoding_protocol.json` · `variant_registry.json` · new `annotation_provenance.json` · new `tools/declare_annotation_provenance.py` · `docs/graph/README.md:270` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · approved 2026-08-21 |

**Codex findings retained, not deleted because fixed** (user instruction): a JSON-`null` sidecar could be re-queued; unchanged `prompt_version` with changed content could leave provenance stale. Both fixed with regression tests. Full record in `CODEX_REVIEW.md`.

**Explicitly NOT resolved:** `D0-003`. The 3 legacy-v1 residuals remain unresolved and are labelled as such in the registry. Nothing claims otherwise.

**Does NOT unblock `D1`.** Chain: `D2a USER_APPROVED → D10 final USER REVIEW with AC-1 evidence → D10 USER_APPROVED → D1_n06-reencode`.

**Open follow-ups:** F-2 `sidecar_path()` uid validation (pre-existing, LOW, unassigned) · F-3 Master under-declared `tools/` scope for the registry tool, same class as MIF-2 · F-4 TASK §12.1's AC-1.a snippet is stale, now marked SUPERSEDED · F-5 Master holds no pre-task corpus fingerprint.

---

### `DL-002` — Stage 1 encoding contract implementation

| | |
|---|---|
| **Source** | `D10_stage1-encoding-contract` |
| **Issue / Finding** | The cache-validity blocker (a resumed n06 would build a two-distribution gallery), the unimplemented ratified template, and four open annotation-pipeline gaps. Plus **FIND-8**, discovered at Master review: `annotate_run.is_complete()` keys on `annotation_contract`, which **no** existing record carries, so a bare invocation would queue **45,955** records for re-annotation |
| **Evidence references** | `encode_text_image.py:73-83`, `:86-108` · `annotate_run.py:98`, `:250` · `resolve_stage1.py:243` · `tools/preflight_stage1_text.py` (run by Master) · `CODEX_REVIEW.md` (2 rounds, 12 findings) · `HANDOFF.md` |
| **Decision / Resolution** | Ratify the implementation in principle: serializer, `text_serialization_id` / cache validity, `load_protocol` mismatch rejection, Stage 1 text pre-flight, >77 true-token hard gate, P-1…P-5, annotation contract versioning, and 3 user-authorised manual translations. **Acceptance withheld** pending `AC-1` |
| **Authority classification** | B-1/B-3 mechanisms = IMPLEMENTATION CHOICE · late template binding = **bug fix** · P-1…P-5 = IMPLEMENTATION CHOICE, **USER DECISION** to ratify the scope extension · 3 translations = **DEVIATION**, user-authorised · legacy-v3 retention = **USER DECISION** |
| **Acceptance condition** | **`AC-1`** — absent explicit force or migration intent, the accepted legacy-v3 corpus must not be automatically treated by `annotate_run` as requiring re-annotation. Sub-conditions AC-1.a…AC-1.e in the brief §7.0. Assigned to `D2a` |
| **USER final decision** | `MODIFY` 2026-08-21 (acceptance withheld pending `AC-1`) → **`APPROVE` 2026-08-21** after `D2a` (`DL-003`) demonstrated `AC-1` |
| **Affected components** | `resolve_stage1.py` · `encode_text_image.py` · `annotate.py` · `annotate_run.py` · `tools/preflight_stage1_text.py` · 3 annotation records (backed up) · `D1`, `D2a`, `D3`, `D4`, `D7` |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | reviewed 2026-08-21 · MODIFY 2026-08-21 · **APPROVED 2026-08-21** |

**`AC-1` — CLEARED.** All five sub-conditions verified by Master through the **production** work-list path `build_work_list()` (`annotate_run.py:439`), not through the superseded `is_complete()` predicate: bare todo TOTAL **0** · legacy-v3 queued **0** · residuals queued **0** · `--force` **45,955** · histogram `{accepted_legacy_v3: 45952, legacy_v1_residual_unresolved: 3}` · no fake v4 `annotation_contract`.

**`G-7` is NOT ratified by this acceptance — it remains independently OPEN.** The user declined to rule on it in this decision, and D10's FINAL ACCEPTED status must **not** be read as approving it:

| | Item | Impact today |
|---|---|---|
| G-7 §5 | Four in-code comment corrections outside R-1/R-2/R-3 | zero behavioural |
| G-7 §6 | Whether the `math.isfinite()` guard belongs in `serialize_annotation()` rather than only in the pre-flight | 0 records |

**Other items explicitly not blocking, carried forward:** `D0-003` **UNRESOLVED** · MIF-D10-3 routed to `D3`/`D4` (re-verified 2026-08-21: `stage1.py:110` and `gallery_index.py:215` load NPZ directly; neither file mentions `text_serialization`) · **F-2** `sidecar_path()` uid validation, LOCAL, pre-existing · template retrieval impact **UNKNOWN**.

**Unblocks:** `D1_n06-reencode`.

**Briefs:** `USER_REVIEW.md` (Rev 2), `USER_REVIEW_FINAL.md`

---

### `DL-001` — Stage 1 text serialization design (U-15)

| | |
|---|---|
| **Source** | `D0-008_stage1-text-template` |
| **Issue / Finding** | The paper specifies no text serialization format (PAPER FACT as to silence). The recorded protocol artifact describes a different template from the one the encoder runs. The running serializer rendered 161 real dimensions as `0 centimetres`, emitted 3,643 ungrammatical articles, and knowingly encoded 1 over-length record. Four in-code justifications did not describe the code |
| **Evidence references** | `2methdology.tex:28` and caption `:24` · Figure 2 (`data-preprocess.png`) · `resolve_stage1.py:96-100`, `:102-115`, `:116-128`, `:162` · `encode_text_image.py:73-83`, `:86-108`, `:194`, `:233` · full-corpus scan of 45,952 v3 records · decision file §6, §9, §10, §11 |
| **Decision / Resolution** | Ratify the design in §11.3: template form, field set and order, centimetres, `width → length → height`, E-1 (dimension precision), E-2 (article removal), E-3 (re-annotate the CJK record), S-1 (uniform formatter, no threshold), S-2 (category capitalisation), R-3 (**delete** the unreachable placement entry), and omission of `synset` / `volume` / `mass` |
| **Authority classification** | Field **set** = PAPER FACT · centimetres = INFERENCE as to intent, OBSERVED DATA as to this corpus · `width,length,height` = INFERENCE from Figure 2 · everything else = **IMPLEMENTATION CHOICE** · E-1/E-2/E-3/S-1/S-2 = **USER DECISION** |
| **Binding user modification** | Omission of `synset` / `volume` / `mass` is an **IMPLEMENTATION CHOICE** with **UNKNOWN retrieval impact**. Must **not** be stated as a PAPER FACT or described as proven redundant. The redundancy argument is WITHDRAWN (Codex C-6) |
| **USER final decision** | `MODIFY` → **`APPROVE`** |
| **Affected components** | `resolve_stage1.py` · `stage1_encoding_protocol.json` · every Stage 1 text embedding · `D10`, `D1`, `D2`, `D3`, `D4`, `D7` · every text-conditioned column of Table 1 |
| **Status** | **`USER_APPROVED`** — FINAL ACCEPTED |
| **Date** | proposed 2026-08-21 · approved 2026-08-21 |

**Explicitly outside this decision:** the n06 cache completion/validity gate (decision file §11.2, B-1…B-4). It is an execution question owned by `D10_stage1-encoding-contract`. This approval neither clears nor waives it.

**Outstanding follow-ups:** FU-2…FU-5 (`D10`) · FU-6 (Master, done at approval) · FU-7 (Master → `D0-003`) · FU-8, FU-9 (deferred). FU-1 is not carried by this decision.

**Integrated at approval:** `workflow/CONTEXT.md` §5 (FU-6), `workflow/MASTER.md`, `workflow/INDEX.md`, this ledger.

**Not yet implemented:** the ratified template does not yet exist in `resolve_stage1.py`. That is FU-2, owned by `D10`.

---

## Migration backlog

**Migration backlog is now empty.** Both pre-gate items — `D0-008` and `D10` — have passed through the gate and reached `USER_APPROVED`.

`workflow/WORKFLOW.md` §19 governs work completed **before** the user review gate was adopted.

Such work is **not** re-run: evidence surveys, Codex reviews, Claude verification of Codex findings, tests, and implementations all stand as produced. What is owed is the tail of the flow:

```
existing completed artifacts
→ Master integration review
→ USER REVIEW BRIEF
→ USER final decision
```

Any acceptance Master recorded before the gate existed is **reclassified as a MASTER RECOMMENDATION** and carries status `AWAITING_USER_REVIEW` here.

| Decision / Task | Pre-gate state | Ledger status | What is owed |
|---|---|---|---|
| ~~`D0-008_stage1-text-template`~~ | Master recorded `ACCEPT WITH FOLLOW-UP` pre-gate | **CLEARED 2026-08-21** → `DL-001`, `USER_APPROVED` | Migration complete. Brief delivered, user returned MODIFY then APPROVE |
| ~~`D10_stage1-encoding-contract`~~ | Executed against D0-008's pre-gate acceptance | **CLEARED 2026-08-21** → `DL-002`, `USER_APPROVED` | Migration complete. `AC-1` satisfied by `D2a`; implementation and Codex review were **not** re-run |

Master must not treat a pre-gate acceptance as though the gate had been satisfied.

---

## Maintenance rules

1. Only Master edits this file.
2. A D0 decision file's status and this ledger must agree. If they disagree, the ledger is the project-level record and the decision file is corrected.
3. `USER_APPROVED` is written only after the user's actual approval, with its date.
4. Superseded entries are **marked**, never deleted, and must name their replacement.
5. When a decision is approved, Master integrates it and records which files were updated.
6. If a decision is later found to rest on a mistaken finding, add a new entry that supersedes it. Do not edit the original's conclusion.
