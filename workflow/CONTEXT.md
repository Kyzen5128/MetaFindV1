# Shared Context

> Orientation for every block conversation. Read this once at the start.
> Master owns this file. Blocks do not edit it.
>
> **Orientation only.** Scope, evidence and definition of done live in each block's
> `BLOCK.md` and `SPEC_*.md`.
>
> **Verified 2026-08-22.**

---

## 1. What we are doing

Reproducing MetaFind — dual-tower multimodal 3D asset retrieval conditioned on scene layout —
so that every research-significant behaviour can be traced to evidence.

The deliverable is a **traceable reproduction**, not working code.

---

## 2. Authority order

From `CLAUDE.md` §3, highest first:

```
1  MetaFind source / supplementary material     docs/paper/metafind_source/
2  the published MetaFind paper
3  upstream papers and official implementations  docs/paper/{ulip2,egnn,idesign}_source/
                                                 /home/kyzen/upstream/{ULIP,egnn}
4  verified project audits and contracts         docs/audit/
5  the graph specification                       docs/graph/
6  the current repository implementation
7  tests and observed runtime / data
8  reasoned inference
9  handoffs and conversational memory
```

**A lower source never silently overrides a higher one.** Conflicts stay explicit until the
USER resolves them.

Handoffs, README files, code comments and previous AI notes are **working context, never
scientific authority.**

---

## 3. Evidence classification — use it on every technical claim

```
PAPER FACT              explicitly stated by MetaFind
UPSTREAM FACT           supported by an upstream paper or official implementation
OBSERVED IMPLEMENTATION confirmed from current repository code
OBSERVED DATA           confirmed from actual project data or output
INFERENCE               logically inferred, not stated
IMPLEMENTATION CHOICE   chosen because the source underspecifies
DEVIATION               intentionally differs from the source
UNKNOWN                 evidence is insufficient
```

**Never present an inference, an implementation choice, a deviation, or an unknown as a
PAPER FACT.**

**Never use "the paper does not say" to justify a method.** Silence is not endorsement. Judge
whether the method actually works, on its own evidence.

### Upstream is a source, not a forbidden zone — `DL-010`, USER, 2026-08-22

MetaFind **builds on** ULIP-2 and EGNN. Reproducing MetaFind does **not** mean you may not read
them. Reading them is often the correct move. **Work out which of three cases you are in
before you cite anything:**

| | MetaFind | What upstream is worth |
|---|---|---|
| **1** | **SILENT**, component inherited unmodified | **The official upstream implementation IS the reference.** Use it. Classify **UPSTREAM FACT** and state the inheritance basis. Do **not** write `UNKNOWN`, and do **not** invent a value |
| **2** | **SPEAKS**, but ambiguous or self-conflicting | Upstream gives you the **variant list, never the answer.** Escalate to the USER |
| **3** | **MODIFIED** the component | Upstream says nothing about the modified part |

Precedent, both already in force: **`U-34`** is case 1 — CLIP freeze scope resolved to ULIP-2
§3.3 because MetaFind builds on ULIP-2 and never says it changed that. **`U-35`** is case 2 —
EGNN Appendix C has three MLP shapes and our `f_h` matches none, so it stayed `UNKNOWN`.

**`DL-004`'s prohibition still stands and is case 2**, not case 1: MetaFind *does* state
`f_x → R³`, so upstream EGNN may not be cited to overrule it. `DL-010` governs **silence**;
`DL-004` governs **ambiguity**.

```
/home/kyzen/upstream/ULIP  @ 95d480f      docs/paper/ulip2_source/
/home/kyzen/upstream/egnn  @ e9ca6c0      docs/paper/egnn_source/  idesign_source/
```

**An UPSTREAM FACT is never a PAPER FACT.** Keep the labels apart.

```
Tests PASS      ≠  reproduction fidelity
Code exists     ≠  paper intent
Codex PASS      ≠  block PASS
Reviewer PASS   ≠  USER acceptance
AI agreement    ≠  evidence
```

---

## 4. Architecture

| Component | Where | Notes |
|---|---|---|
| Dual tower | `metafind/models/dual_tower.py` | query and gallery towers over a ULIP-2 backbone. `freeze_gallery()` refuses `fully_shared` — the paper's frozen-gallery requirement and a single shared module cannot both hold |
| Backbone | `metafind/models/ulip_backbone.py` | vendored ULIP-2 + PointBERT under `metafind/vendor/ulip/`. CLIP frozen during encoding |
| Fusion | `metafind/models/fusion.py` | modality masking, p = 0.3, independently per modality |
| Loss | `metafind/models/losses.py` | Stage 1 query→gallery only; Stage 2 symmetric. τ = 0.5 |
| ESSGNN | `metafind/models/essgnn.py` | `architecture_family` ∈ {`sec25_two_mlp`, `appendix_shared_msg`}, `coord_feat` ∈ {`current`, `updated`}. Currently coupled |
| Protocol resolvers | `metafind/models/resolve_stage1.py`, `resolve_stage2.py` | write the JSON artifacts trainers are not allowed to decide for themselves |
| Trainers | `metafind/train/{stage1,stage2,gallery_index}.py` | |
| Paths | `metafind/paths.py` | **use it.** Never hardcode absolute paths |

---

## 5. Facts every block owner should know

- **Splits do not depend on embeddings.** The split builder reads the three index files and
  never touches an embedding. Stage 1 *training* needs both.
- **Splits bake decisions into artifacts.** `stage1_protocol.json` carries `tower_sharing` and
  a hash of the hyperparameters, and the trainer refuses to run if the hash does not
  dereference. Changing a hyperparameter after the split means rebuilding the split.
- **The encoding protocol resolver writes three artifacts in one call.** Corrections that touch
  any of them must land together, or it runs twice.
- **The gallery index is fingerprinted to the checkpoint.** An index built by a drifted encoder
  produces self-consistent wrong numbers with no error anywhere.
- **`fully_shared` cannot reach Stage 2.**
- **Point clouds and renders are verified** against official ULIP-2 artifacts and do **not**
  need regenerating. Evidence: `workflow/blocks/ULIP2/evidence/n03_n04_upstream_verification.md`.
- **Our assets sit 180° yawed about Y** relative to ULIP-2's released clouds. Measured: this does
  **not** move the embedding. It **does** matter for scene composition, where assets are placed
  with real geometry.

---

## 6. Environment

```
repository       /home/kyzen/MetaFindV1
data root        /mnt/data1/kyzen/MetaFind      reached via the ./data symlink
python           /home/kyzen/miniconda3/envs/MetaFind/bin/python     (conda env MetaFind)
run modules as   python -m metafind.<module>    from the repository root
GPU              NVIDIA GeForce RTX 5090, 32,607 MiB
upstream refs    /home/kyzen/upstream/ULIP @ 95d480fe
                 /home/kyzen/upstream/egnn @ e9ca6c0c
models           /mnt/data1/kyzen/models/
graph            graphify-out/graph.json — navigation only; conclusions return to source
```

**`/mnt/data1` is an SMR drive** (`ST4000DM004`). Sustained small-file writes collapse to
single-digit MB/s once its cache fills — write latency above 5,000 ms has been measured under
mixed load. Large sequential writes are fine. **It now holds the whole dataset**, and both n05
and n06 write ~46,000 small files each. Treat every inherited runtime estimate as unmeasured.

---

## 7. Global constraints

- Do not silently replace missing evidence with an assumption. Mark uncertainty explicitly.
- Do not infer a paper requirement from the current implementation.
- Do not change scientific behaviour to make a test pass, an import succeed, or a shape align.
  Find the actual cause first.
- Do not delete or regenerate datasets, checkpoints, embeddings, caches or experiment outputs
  without explicit authorisation.
- A block must not expand into another block's scope, and must not start the next stage alone.
- Anything affecting shared architecture, a dependency, or an accepted assumption is a
  **MASTER-IMPACTING FINDING**: write it to `HANDOFF.md` with evidence, say whether work can
  safely continue, and **do not act on it**.
- Codex is an independent reviewer, not authority. Verify its findings against stronger evidence.

---

## 8. Starting a block conversation

Read, in order:

```
1  CLAUDE.md
2  the applicable .claude/rules/
3  this file
4  workflow/BLOCKS.md          structure and rules
5  workflow/SKILLS.md          method
6  your own workflow/blocks/<BLOCK>/BLOCK.md
7  only the files your BLOCK.md and SPEC name
```

Do not re-read the whole repository. Do not read another block's directory.

---

## 9. Where things live

```
workflow/MASTER.md                  project state
workflow/BLOCKS.md                  blocks, roles, communication and review rules
workflow/SKILLS.md                  which skill, by whom, when
workflow/CONTEXT.md                 this file
workflow/DECISION_LEDGER.md         decisions in force
workflow/roles/                     conversation-role prompts
workflow/blocks/<BLOCK>/            BLOCK.md · SPEC_*.md · REVIEW.md · HANDOFF.md · evidence/
workflow/blocks/SPEC_TEMPLATE.md    the 15-section contract
workflow/archive/                   history only. never authority, never project state

docs/paper/                         paper sources and figures — the top of the authority order
docs/audit/                         formula inventory, upstream map, contradictions, contracts
docs/graph/                         node registry, graph spec, validation plan, findings
metafind/                           implementation
tests/                              582 tests
tools/check_graph.py                structural gate checker — run after spec or code changes
```
