# E. GRAPH_REVALIDATION

Every step of `docs/graph/02_BUILD_STEPS.md`, re-checked against the authors'
arXiv TeX source rather than against the converted Markdown the earlier audit
used.

Verdicts:

| verdict | meaning |
|---|---|
| `VERIFIED` | the TeX supports it; unchanged |
| `NEEDS_CHANGE` | wrong or mislabelled; changed this pass |
| `UNRESOLVED` | the paper is silent and the decision is still open |
| `OBSOLETE` | rested on a Markdown artifact; removed |

**Headline: the build steps came through almost intact.** Of the graph's
decisions, the substantive changes are five code bugs (G below) and four
labelling corrections. Nothing in the pipeline design was invalidated by moving
to the TeX — which is itself worth recording, because the natural fear on
discovering a corrupt source is that everything built on it is worthless.

---

## E.1 Authority

| item | before | after | verdict |
|---|---|---|---|
| formula authority | `docs/paper/*_paper.md` (converted) | `docs/paper/*_source/` (arXiv TeX) | `NEEDS_CHANGE` |
| paper Markdown | "Level 0 authority" | **deleted** — a lossy copy kept for convenience is still a second authority | `OBSOLETE` |
| `docs/audit/repaired/` | repaired Markdown, treated as source | deleted | `OBSOLETE` |
| `repair_report.json`, `latex_sanity.json` | repair provenance | deleted; superseded by `formula_inventory_validation.json` | `OBSOLETE` |
| A/B/C/D | built on repaired Markdown | rebuilt from TeX | `NEEDS_CHANGE` |

The Markdown was not merely lossy at the margins. `\frac` arrived as a form feed
plus "rac"; `\neq` arrived as a **real newline**, which is a legal character, so
a control-byte census skipped it and that whole command class survived two
rounds of "repaired". A conversion that loses characters silently cannot be an
authority no matter how carefully it is patched.

---

## E.2 Phase 0 — environment and acquisition

| step | verdict | note |
|---|---|---|
| 0.1 environment | `VERIFIED` | no paper content |
| 0.2 download (Objaverse-LVIS) | `VERIFIED` | TeX: "approximately 48,000 distinct 3D assets" |

Measured 46,052 GLBs against "approximately 48,000". Already recorded as a
measured deviation with its cause; the TeX changes nothing.

---

## E.3 Phase 1 — data processing

| step | claim | verdict |
|---|---|---|
| 1.1 point clouds | 10,000 points, `pc_norm` | `VERIFIED` for the normalisation (`[UPSTREAM]`, ULIP's dataset); **the point count is `[IMPLEMENTATION CHOICE]`** — MetaFind states none, and ULIP-2 ablates 10k/8k/2k rather than fixing one |
| 1.2 renders | 11 views | `VERIFIED` — "Each asset is rendered from 11 orthogonal viewpoints" |
| 1.3 annotation | GPT-4o structured text | `VERIFIED` — "annotated using GPT-4o … object category, size dimensions, materials, and placement constraints". Our local-Qwen substitution remains a recorded `[DEVIATION]`, unaffected. |
| 1.4 scene graphs | ProcTHOR, support ∪ adjacency | `VERIFIED` with a caveat, below |
| 1.5 splits | 80/20 | `VERIFIED` — "In both datasets, we allocate 80% … 20%" |

**Caveat on 1.4 — S5, which is U-29 and predates this pass.** The TeX describes
two edge types:
"(i) physical-relation edges … (ii) semantic-relation edges … obtained by
prompting an LLM on object pairs". But `MF-2`, `MF-3` and `MF-10` carry exactly
one edge term, `e_ij`, and both §2.5 and the appendix describe it as the
semantic embedding. **The paper never says how physical edges enter the math.**

Our `essgnn_edge_protocol` already records
`physical_relation_encoding = "neighbourhood_only"` — physical edges determine
`\mathcal{N}(i)`, semantic edges supply `e_ij`. That is the reading requiring no
invention, and it is now confirmed as `UNRESOLVED` in the paper rather than
settled by it. **Verdict: the decision stands; its label was too strong.**

---

## E.4 Phase 2 — training

### Step 2.1 Stage 1

| item | verdict | basis |
|---|---|---|
| both towers trained | `VERIFIED` | "both query and gallery encoders are trained" |
| 30% per-modality masking | `VERIFIED` | "each modality in the query has a 30% probability of being independently masked" |
| masked embeddings, not zero-padding | `VERIFIED` | "Rather than zero-padding, we apply masked embeddings" |
| gallery modality-complete | `VERIFIED` | "The gallery encoder is trained to be modality-complete" |
| Eq. 5 unidirectional | `VERIFIED` | C7 — and left alone deliberately |
| OpenCLIP frozen (U-34) | `VERIFIED`, still `[INFERENCE]` | ULIP-2's TeX settles the upstream half outright: "the pre-aligned and **frozen** image encoder $E_I$ and text encoder $E_T$", "freeze it during the pre-training". MetaFind never states a change. |
| `τ` learnable | `VERIFIED` as `[UPSTREAM]` | ULIP-2 TeX line 616: "$\tau$ is a learnable temperature parameter". Our `losses.py` docstring claim was checked and is correct. |
| `τ = 0.07` init | `VERIFIED` as `[IMPLEMENTATION CHOICE]` | not in either paper; CLIP's convention |
| checkpoint contents | **`NEEDS_CHANGE`** | P0-1 — see G |
| RNG device | **`NEEDS_CHANGE`** | P0-5 — see G |

### Step 2.2 gallery index

| item | verdict |
|---|---|
| index pinned to the checkpoint | `VERIFIED` |
| staging → gate → promote | `VERIFIED` |
| encoder identity hash | **`NEEDS_CHANGE`** — P0-2, covered fusion only |
| Objaverse and ProcTHOR indices never merge (U-08a) | `VERIFIED` |

### Step 2.3 Stage 2

| item | verdict | basis |
|---|---|---|
| only query fuser + ESSGNN update | `VERIFIED` | "Only the query-side fusion layer and the ESSGNN module are updated; the gallery encoder is frozen" |
| 30% scene dropout, per batch | `VERIFIED` | "the layout vector $e_{\text{layout}}$ is omitted in 30% of batches" |
| scene dropout independent of `p_mask` | **`NEEDS_CHANGE`** | aliased to `p_mask`; Table 3's sweep would have moved both |
| Eq. 7/8 bidirectional | `VERIFIED` | "We adopt a **bidirectional contrastive learning** objective" |
| ESSGNN present in the model | **`NEEDS_CHANGE`** | P0-3 — built with `use_layout=False` |
| U-08a same-assetId positive | `VERIFIED` | nothing in the TeX binds the Stage 2 gallery to Objaverse |
| U-08d target removed from context | `VERIFIED` as `[IMPLEMENTATION CHOICE]` | the paper does not describe the training sample at all |
| U-08e unique positive per batch | `VERIFIED` as `[IMPLEMENTATION CHOICE]` | ditto |
| ProcTHOR point normalisation | **`NEEDS_CHANGE`** | P0-4 |

---

## E.5 Phase 3 — evaluation

| step | verdict | note |
|---|---|---|
| 3.1 Table 1 | `UNRESOLVED` | S3/S4 — `Fusion` unpicked and no hyperparameters, so Table 1 cannot be attributed to the paper |
| 3.2 equivariance validation | `VERIFIED`, strengthened | now runs on CUDA too; `L ≥ 2` requirement recorded (S6) |
| 3.3 Table 2 / 3 | `UNRESOLVED` | U-27, below |

**On 3.2 — C8 is RA-4 and predates this pass.** The paper motivates ESSGNN by
GAT sensitivity to "global translation **and scaling**", but SE(3) contains no
scaling. RA-4 already had the right disposition and it stands: **measure how far
`e_layout` moves under scaling; do not predict, and do not attribute any
robustness found to the SE(3) proof.**

---

## E.6 U-27 — materially advanced, still open

`3experiments.tex` says only "a set of 200 randomly sampled scenes". No prompts,
no room dimensions, no seed. **`UNRESOLVED` — confirmed, not resolved.**

Two changes from reading I-Design's own arXiv source:

1. **A citable basis now exists.** `tabs/tab_promptlist_minimal.tex` (20 prompts,
   "Design me a bedroom." / "Design me a living room.", each with `[x, y, z]`
   room dimensions) and `tabs/tab_promptlist_others.tex` (40 detailed prompts
   across living room, home office, kitchen, bedroom, bathroom, dining room,
   playroom, fitness room, …). Constructing 200 by extending a published
   distribution is defensible in a way that inventing 200 is not.
2. **One unknown disappears.** I-Design's paper describes its input as "an
   unstructured, grammar-free natural language user input"; object count is
   decided by the agents, not supplied. The registry's
   `IDesign(no_of_objects, user_input, room_dimensions)` describes the released
   **code's** entry point — a measured implementation fact, and it should say so
   rather than read as the paper's interface.

**Scope, stated precisely:** U-27 blocks the Table 2 branch. It blocks neither
stage of training, neither gallery index, nor Table 1. Both "U-27 blocks
nothing" and "U-27 blocks everything" have been asserted in this project and
both were wrong.

---

## E.7 Contradictions: before and after

Every C-entry was re-derived from the TeX. **None was a conversion artifact.**
Two entries an earlier draft called "new" were not: C8 is RA-4 and S5 is U-29,
both found by rereading the paper before this audit began. The TeX confirmed
them; it did not discover them.

| id | before | after |
|---|---|---|
| C1 two ESSGNNs | suspected | `VERIFIED` — §2.5 has `f_h`/`f_x`, the appendix has `φ_e`/`φ_x`/`φ_h` |
| C2 `h^(0)` contains `x` | suspected | `VERIFIED` — "We begin by assuming that $\rm h^0$ is invariant to SE(3)" |
| C3 `f_x → R^3` | suspected | `VERIFIED`, and `[UPSTREAM]` settles it: EGNN's `φ_x` "outputs a scalar value" |
| C4 width mismatch | derived | `VERIFIED` — `h^(0) ∈ R^{d+3}`, `f_h: R^{2d+1+e} → R^d`, residual |
| C5 `N(i)` vs `j ≠ i` | suspected | `VERIFIED` |
| C6 distance vs squared | suspected | `VERIFIED` — `\|x_i^l - x_j^l\|_2` vs `\|x_i^l - x_j^l\|^2` |
| C7 Stage 1 asymmetry | suspected | `VERIFIED` |
| C8 SE(3) vs scaling | registered as **RA-4** | `VERIFIED` — the TeX confirms the wording and adds nothing |
| S5 physical edges | registered as **U-29** | `VERIFIED` — still `UNKNOWN`, as U-29 already said |
| S6 `t_i` encoder | partial | `VERIFIED` — "a text-derived feature", nothing more |
| old "Eq. 7a/7b" | treated as the paper's numbering | **corrected** — the TeX has one numbered Eq. (7); the split was the converter's |

---

## E.8 Artifacts: regenerate or keep

| artifact | count | verdict |
|---|---|---|
| n02 Objaverse GLBs | 46,052 | **keep** — acquisition unaffected |
| n03 point clouds | 46,052 | **keep** — `pc_norm` at write time is right |
| n04 renders | 45,955 | **keep** |
| n05 annotations | in progress | **keep** — schema unaffected |
| n07 scene graphs | 12,000 | **keep** |
| n07b ProcTHOR modalities | 1,467 | **keep** — world-frame storage is correct; the fix is at encode time |
| n09c scene splits | 9,600 / 2,400 | **keep** |

**No preprocessing needs rerunning.** P0-4 changes how stored clouds are
*consumed*, not how they were produced — deliberately, so the AI2-THOR
bounding-box provenance survives.

---

## E.9 Summary

| verdict | count |
|---|---|
| `VERIFIED` | 26 |
| `NEEDS_CHANGE` | 9 (5 code, 4 labelling) |
| `UNRESOLVED` | 4 (C1/U-26, U-27, Table 1 attribution, S5 label) |
| `OBSOLETE` | 3 (repaired Markdown, repair report, latex sanity) |

The pipeline design survived. The code did not.
