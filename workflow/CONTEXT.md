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

### Agreement on a shared wrong premise — the failure that LOOKS like cross-validation

**Before you verify a finding, ask: what is this claim's upstream, and did anyone read it?**

Worked example, 2026-08-22, both roles self-reported. The ESSGNN Reviewer claimed
`Q-N08-MODEL` — the n08 **LLM** — also determines the node features `t_i`. The ESSGNN Engineer
independently checked it and **confirmed and strengthened it**: `essgnn_arch_protocol.json` has no
`node_feat_dim` / `edge_feat_dim`, `stage2.py` reads the two widths separately, and there is a
comment recording that `edge_dim` was once assigned from the node record. **Every one of those
facts is true.**

The conclusion was still wrong. `semantic_edges_run.py:371` reads node texts from
`procthor_object_text.json` — n07's **rule-based** `"a {category}"`. **Changing the LLM does not
move `t_i` at all.** The coupling is real but it lives in `TEXT_ENCODER`, a different knob.

**Both roles correctly verified the downstream. Neither verified the upstream** — nobody asked
who produces the node text. Two independent measurements agreed because they were measuring the
same true downstream hanging off the same false premise, **and the agreement raised both their
confidences.**

This is a **different failure mode** from trusting a non-authoritative source, and harder to
catch:

```
one role believes a declaration instead of the code      -> caught by a second reader
two roles each verify the downstream correctly, nobody
reads the upstream, and consistency is mistaken for
corroboration                                            -> a second reader makes it WORSE
```

**Independent confirmation of a claim's consequences is not confirmation of the claim.**

### Walk the authority order TOP-DOWN before you measure anything

The two rules above stop you **stating** something wrong. This one stops you **doing work that
was already done** — and doing it worse.

**Ask "does this question have a `U-` number?" BEFORE asking "what does the code do?"**

Worked example, 2026-08-22, self-reported by both ESSGNN roles. Across four rounds of mutual
verification, **none of the three of us read the `U` registry in `docs/graph/graph_spec.yaml`** —
which is **rank 5** in §2 above, over the repository implementation (6) and observed data (7). All
three jumped straight to 6 and 7. Two rounds went into a danger that `U-21` had **closed two weeks
earlier**, and the recommendation we reached — *I-Design first* — is the **opposite** of what
`U-27`'s own `do_not_rush` field instructs.

Everything measured was true. It was simply a worse answer than the one already on file at a
higher rank.

```
registry declarations are not behaviour evidence  ->  don't treat LOW authority as high
AI agreement is not evidence                      ->  don't treat CONSENSUS as corroboration
walk the authority order top-down                 ->  don't RE-DERIVE at rank 6 what
                                                       rank 5 already answered
assert the PROPERTY, not a count of it            ->  don't tune the threshold to the data
```

### Assert the property, not a count of it

A count is a **proxy**. When it moves you cannot tell whether the world moved or your threshold
did, and the temptation is to move the threshold until the number behaves — which blinds the
check to the thing it exists for.

Worked example, 2026-08-22. The `n03` regeneration moved *"assets with exactly zero variance on
one axis"* from **21** to **18**. Master proposed re-cutting at `≤1e-12 & others >0.05` (102);
the ESSGNN Reviewer proposed `≤1e-20` (84). **The ULIP2 Engineer rejected both, and was right on
three counts:**

1. **`validation_plan.yaml:101` had already defined the real check** — `L1-PC-NONDEGENERATE`:
   *"for every axis where variance is at or near zero, the mesh's own `raw_bbox_extents` is
   correspondingly flat."* Its own note says an absolute floor **cannot** tell a flat **asset**
   from a flattening **bug**, and *"raising the floor until they pass would blind the check to the
   failure it exists for."* Rank 5 had the answer; two of us re-derived it worse at rank 7.
2. **The property is threshold-free.** Re-measured by Master: 106 assets at `≤1e-12`, **0**
   whose mesh is not correspondingly flat, worst ratio `7.004e-04`. **Violations are 0 at every
   cut from `0` to `1e-12`.** The epsilon moves; the conclusion does not.
3. **Changing the metric would have destroyed the only comparison available.** `21` survived the
   corpus deletion **only because it was written down** in `validation_plan.yaml`. The corpus it
   described is gone. So `== 0` has a before and after, and every replacement statistic has
   **none**.

And picking a cut **after** looking at the distribution is precisely what `S-3` records the USER
prohibiting. **Master proposed it anyway.**

---

### The notch — a claim stated one step stronger than its mechanism supports

**Five instances on 2026-08-24, from the two most active roles.** Not carelessness about facts.
Every one was *true-adjacent* and stated **one notch stronger than the mechanism behind it**.
That notch is what turns an observation into a defect, and it reads identically in a code comment,
a research note, a registry entry and a message to a peer.

| what was written | what the mechanism supported |
|---|---|
| `failure_class` *"decides whether an asset is ever tried again"* | it is written to a log and **read by nothing** |
| `splits.py:72-74` *"the paper never says which fusion the full model uses"* | `3experiments.tex:143` says **"the final selected Transformer"** |
| *"the two-verdicts case **cannot** arise"* — a prose exclusion in a Codex prompt | a prompt line is an instruction, not enforcement; the file is still in the diff |
| Master: *"n04 **died**"* | a stale `RUNNING` row and no process — the USER had stopped it, twice |
| Master: *"**zero** pipeline processes"* | a `ps` pattern that could not match `envs/MetaFind/bin/python`; 45 were alive |

**The tell is the strong word.** `decides` · `never` · `cannot` · `died` · `zero`. Each is a claim
about a MECHANISM, and in every case the mechanism was one step weaker than the word.

**The check is one question, asked before the sentence is written:**

> *What would have to be true for this word to hold — and did I verify that, or the thing next to it?*

Two of the five were caught by the author, two by a peer, one by Codex. **None was caught by a
test, a gate, or a review of the code alone** — they are claims *about* code, and only reading the
mechanism catches them.

**Enforcement beats instruction, and it is usually available.** `--sandbox read-only` is
enforcement; "please don't read that file" is not. `find -L` is enforcement; a comment saying the
path is a symlink is not. When both are available at similar cost, the instruction is the wrong
one — and describing an instruction as a guarantee is the notch itself.

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
- **Point clouds and renders are BEING REGENERATED.** Corrected 2026-08-22 — the previous text
  here read *"verified … and do not need regenerating"*, and that is now false. `n04`'s camera
  orbited **`+Z`** while the meshes are **`Y`-up**, so every asset tumbled instead of turning and
  the sidecars described an orbit that was never performed. Confirmed three ways: code reading,
  a falsifiable pixel prediction (H-A `+0.893` against H-B `−0.671`, 120 assets), and ULIP-2's own
  released renders. Authorised by `U-B` / `U-G` (`DL-011`), which **supersede** `BLOCK.md` §7's
  no-re-render rule. `S-5` for the corrected configuration is already measured: **R@1 97.2%**
  against the v2 corpus's 83.2%, n=286, target ULIP-2's own `image_feat`.
- **The 180° yaw is being corrected in the same pass**, at the mesh-load layer. The old
  measurement — that the yaw does not move the embedding — still stands as a measurement, but it
  describes a state that is ending. `Q-YAW-PLACEMENT` shrinks accordingly.
- **`n07b` was always in the correct frame.** It orbits `+Y`, for a reason its docstring states
  incorrectly (*"trimesh's z-up"* — `n04` was never z-up). So `n04` and `n07b` have been in
  **different frames** since both were produced, and `test_the_orbit_uses_n04s_constants_not_copies`
  is structurally blind to it: it compares two scalars and not the direction. The `n04` correction
  repairs the mismatch as a side effect.

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

## 6a. ⚠️ THE STORAGE PICTURE CHANGED AGAIN, 2026-08-23. Read this before §6b.

**Measured by Master 2026-08-23. §6b below describes 2026-08-22 and is now partly historical.**

```
data -> /home/kyzen/metafind_data        NVMe.  Was /mnt/data1/kyzen/MetaFind.
NVMe   /  937 G   515 G used   375 G free   (was 816 G free)
SMR    /mnt/data1  3.6 T  455 G used  3.0 T free
```

**The 328 GB of Objaverse GLBs now exist on BOTH disks, in full, as separate copies.**
Verified: 46,052 `.glb` on each, `links=1` on both, different devices — **not hardlinks, not
symlinks. 328 GB of NVMe is a duplicate of 328 GB of SMR.**

**[CORRECTED within the hour — Master's first version of this paragraph was wrong and is
withdrawn.]** It said this duplicate was created today and was *"what the USER stopped on
2026-08-22"*. **It is not.** File timestamps, both volumes:

```
NVMe  first .glb   2026-08-15 14:11      glbs dir  2026-08-15 11:37
SMR   first .glb   2026-08-15 13:53      glbs dir  2026-08-15 11:37
either volume, .glb written after 2026-08-20:   0
```

**Both copies are eight days old, made ~18 minutes apart, and nothing has been written to
either since.** The only thing that changed on 2026-08-23 is the **`data` symlink**, repointed
from SMR to NVMe at 08:13. **No 328 GB copy happened today and no instruction was violated.**
Master inferred a fresh copy from the free-space drop without checking a single mtime.

**ATTRIBUTION CLOSED 2026-08-23.** The Engineer supplied the USER's verbatim order for
repointing `data` to NVMe: 「沒關係 我現在想把資料集也一起搬過來好了 你現在搬 整個搬過來」, after
asking 「會不會比較快?」 and being shown the SSD/HDD difference. **Explicit, not inferred.** It
copied rather than moved — deleting the source before verification is irreversible — and
verified: 46,052 GLB, 60 sampled byte-identical, total bytes equal at 351,831,395,378.

The Engineer notes its memory of copying today conflicts with the 2026-08-15 mtimes above and
**defers to the measurement**: what it did today may have been verifying an existing copy. **The
timestamps are the record.**

What remains true: the duplicate is real, it costs 328 GB of NVMe, and **nobody should delete
either side without the USER saying which** — the SMR copy is now the unused one.

**Headroom is not a concern — measured, not estimated.** 9,455 renders occupy 15.45 GB, i.e.
**1.67 MB/asset → 75.3 GB for the full corpus, 59.8 GB still to write, against 374 GB free.
Six times the need.** And deleting 328 GB mid-run is actively harmful: `n04`'s 65
resource-class failures cluster in high-concurrency bursts (hourly correlation between the
`gpu_oom` and `no_output` buckets **+0.949**, five hours both at zero), and a mass delete
manufactures exactly such a burst. `/mnt/data1` is also SMR, where deleting 46,052 files is not
free. **If either copy goes, it goes after `n04` and `n05`.**

**Operationally it is not yet a problem:** `n04` has ~34 h left and 12 views × 46,052 at 512 px
projects to well under 375 GB. It is recorded because a 328 GB duplicate is not something anyone
should discover by running out of space at hour 30.

---

## 6b. Storage is SPLIT — USER decision 2026-08-22

> 「之後記得都先把檔案放在這邊 除非做完了或空間不足」 · 「大型資料集不要搬過來喔」

**Everything the pipeline produces goes to NVMe. Everything it downloaded stays on the SMR drive.**
Implemented purely as symlinks under `data/outputs/`, so `metafind/` and `paths.py` are unchanged
and still see one tree. **Do not hardcode either root — keep using `paths.py`.**

```
NVMe  /home/kyzen/metafind_out/{pointclouds,renders,annotations,embeddings,checkpoints}
      816 GB free.  n03 n04 n05 n06 n10 all land here.

SMR   data/datasets/objaverse-lvis/glbs   328 GB   sequential reads, fine where it is
      data/models/{hf-cache,ulip2}         12.4 GB
      /mnt/data1/kyzen/models             100 GB   the three annotator candidates
      data/outputs/{logs,scene_graphs,procthor_modalities}
```

**Do not copy the 328 GB of GLBs to NVMe.** It was attempted and the USER stopped it. Reading them
is sequential and SMR handles that, as long as nothing else competes for the head.

### Why it was worth doing — measured, n03 run twice, same code, same corpus

```
read and write both on SMR    573 -> 391 assets/min, still falling
split, output on NVMe         ~897 assets/min, flat for the whole run
                              SMR 86% busy but READING only; NVMe 4% busy
```

⚠️ **The trap, recorded so nobody re-derives it wrong.** Measured write bandwidth was only
**796 kB/s**, which makes *"moving the output somewhere faster cannot help — we are not
bandwidth-bound"* look like sound reasoning. **It is wrong.** The constraint is **head seek**, not
bandwidth: small writes interleaved with heavy random reads collapse an SMR drive. Splitting the
two workloads across two physical devices is what recovered the throughput, and it more than
doubled it.

**`CLAUDE.md` §9 still names `/home/kyzen/data/MetaFind` as the data root. That path does not
exist** — the real link is `data -> /mnt/data1/kyzen/MetaFind`. `CLAUDE.md` is guard-protected and
is the USER's to correct. §9 itself says *"Report stale machine-specific absolute paths before
modifying them"*, so it is reported here and not touched.

**It has a real consequence, not just a documentation one.** `n03`'s sidecars record
`path: /home/kyzen/MetaFindV1/data/outputs/pointclouds/<uid>.npz` — an absolute path that resolves
through **two** symlinks:

```
/home/kyzen/MetaFindV1/data/outputs/pointclouds/<uid>.npz
  -> data       -> /mnt/data1/kyzen/MetaFind
  -> pointclouds -> /home/kyzen/metafind_out/pointclouds     <- where the bytes actually are
```

Verified resolvable today. **But it depends on both links surviving**, and anyone following
`CLAUDE.md` §9 to `/home/kyzen/data/MetaFind` finds nothing at all.

**`/mnt/data1` is an SMR drive** (`ST4000DM004`). Sustained small-file writes collapse to
single-digit MB/s once its cache fills — write latency above 5,000 ms has been measured under
mixed load. Large sequential writes are fine.

---

## 7. Global constraints

- **`node_registry.yaml`'s `reads` / `writes` declarations are NOT evidence of what the code does.**
  Established 2026-08-22 by **two independent cases**, both self-caught: the registry declares
  `n08` `read_before_write` for `procthor_node_embeddings.npz`, but `semantic_edges_run.py` writes
  it **unconditionally** — encode → `.part.npz` → replace, no `exists()` check anywhere; and the
  registry lists `sem_edge_cache` in `n09c`'s `reads`, while `scene_splits.py` computes coverage
  *after* the split and never uses it to choose houses. **Both mismatches happen to be safe. The
  next one will not be.** `CLAUDE.md` §5 already forbids inferring runtime behaviour from a
  schema; this is the concrete form it takes in this repository, and **six roles read that
  registry**. Trace the code.
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
