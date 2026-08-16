# F. CODE_GRAPH_CONSISTENCY

Subsystem by subsystem: does the paper, the graph, the code and the tests say
the same thing?

A row is `CONSISTENT` only when all four agree **and something executes the
check**. A decision recorded in a protocol that no test reads is `SPEC ONLY` —
it may be right, but nothing would notice if the code drifted. That distinction
is the point of this document: the previous pass had four rows that read as
consistent and were not, and each was found by comparing code to code rather
than doc to doc.

| status | meaning |
|---|---|
| `CONSISTENT` | paper/graph/code/test agree, and a test asserts it |
| `SPEC ONLY` | graph and code agree; no test would catch drift |
| `FIXED` | inconsistent before this pass; corrected |
| `OPEN` | still inconsistent, recorded not fixed |

---

## F.1 ESSGNN

| item | paper | graph | code | test | status |
|---|---|---|---|---|---|
| `h^(0) = t_i` | contradicts itself (C2) | `h0_mode=semantic` | `essgnn.py` | positive + negative injection | `CONSISTENT` |
| `f_x → R^1` | says `R^3` (C3) | no flag, audit-only | `_mlp(..., 1)` | equivariance, CPU + CUDA | `CONSISTENT` |
| `\mathcal{N}(i)` | contradicts itself (C5) | edge-driven | `edge_index` | edge-count tests | `CONSISTENT` |
| squared distance | contradicts itself (C6) | `distance=squared` | `radial` | `squared_and_euclidean_produce_different_layouts` | `CONSISTENT` |
| `coords_agg=sum` | Eq. 3 sums | `sum` | `unsorted_segment_sum` | `coords_agg_defaults_to_sum_per_eq3` | `CONSISTENT` |
| no `C` normalisation | absent (B.2.2) | absent | absent | — | `SPEC ONLY` |
| pooling | unspecified (S2) | `pooling=mean` | `_pool` | protocol-locked-values test | `CONSISTENT` |
| `L`, widths | unspecified (S4) | protocol | `from_protocol` | `from_protocol_refuses_*` | `CONSISTENT` |
| missing-edge token | unspecified (U-30) | `learned_missing_token` | **`nn.Parameter`** | `missing_edge_token_gets_gradient` | `FIXED` |
| equivariance at `L ≥ 2` | Eq. 4 | — | — | `n_layers=3` in both tests | `CONSISTENT` |
| `x^(L)` equivariance | Eq. 4 | — | not exposed by `forward` | **not asserted** | `OPEN` — D.4 |
| **C1 family** | specifies two | `architecture_family` | both implemented | `test_se3_equivariance` runs both | `FIXED` |
| `coord_feat` per family | — | resolved, not defaulted | `__post_init__` | refusal test | `FIXED` |
| F8's scope | — | — | measured on both | pinned + companion | `FIXED` |

**`OPEN` row.** `test_se3_equivariance` asserts h-invariance only. Coverage is
better than it sounds — `x` re-enters through `d_{ij}` at the next layer, so a
distance-changing breakage is caught — but a distance-*preserving* one
(reflection, coordinate permutation) would pass. Fixing it means `forward`
returning `x^(L)`, which changes the signature every caller uses; recorded
rather than done on the eve of the first training run.

---

## F.2 Stage 1

| item | paper | graph | code | test | status |
|---|---|---|---|---|---|
| both towers trained | §2.6 | ✓ | optimizer holds all three modules | `assert_checkpoint_covers_optimizer` | `FIXED` |
| **PointBERT saved** | — | ✓ | **3-section checkpoint** | round-trip, bit-identical | `FIXED` |
| `logit_scale` saved | — | — | `loss_trainable_state` | round-trip | `FIXED` |
| OpenCLIP frozen | inference (U-34) | ✓ | `train_scope` | backbone scope tests | `CONSISTENT` |
| 30% per-modality mask | §2.6 | ✓ | `sample_modality_mask` | rate + independence tests | `CONSISTENT` |
| masked embedding not zero-pad | §2.6 | ✓ | `FusionConfig.zero_pad` | `masked_modality_uses_a_learned_token_not_zero` | `CONSISTENT` |
| Eq. 5 unidirectional | §2.6 | ✓ | `bidirectional=False` | direction test | `CONSISTENT` |
| **CUDA RNG** | — | — | draws on the generator's device | **CUDA smoke, ran on the 4090** | `FIXED` |
| `Stage1RuntimeConfig` is the construction path | — | declared authority | **n10 does not use it** | — | `OPEN` |
| **backbone actually frozen** | §2.6 | — | `set_train_scope` + `is_frozen` | 9 tests, incl. non-determinism injection | `FIXED` |
| checkpoint guard reads `sections` | — | — | compares against the saved keys | guard tests | `FIXED` |

**The `OPEN` row.** `stage1_config.py` declares itself the single path that
validates and hashes a run's configuration; `stage1.py` reads the protocol dicts
directly. Nothing is wrong today — same artifact, same values — but a class
nothing calls cannot enforce anything, and the hash it would compute is the
thing G3 dereferences. Routing n10 through it is a real change to the trainer
and is left for a deliberate commit rather than folded into this audit.

---

## F.3 Gallery index

| item | paper | graph | code | test | status |
|---|---|---|---|---|---|
| gallery frozen | §2.4, §2.6 | ✓ | `freeze_gallery(True)` | freeze tests | `CONSISTENT` |
| **encoder identity hash** | — | "gallery encoder matches Stage 1" | **backbone + fusion** | — | `FIXED`, `SPEC ONLY` |
| index pinned to checkpoint | — | ✓ | digest compared at promote | promote tests | `CONSISTENT` |
| **index filename is immutable** | — | write_once | named per checkpoint sha | — | `FIXED`, `SPEC ONLY` |
| staging → gate → promote | — | ✓ | `promote(gate_passed)` | refuses without the gate | `CONSISTENT` |
| Objaverse ≠ ProcTHOR index | U-08a | ✓ | separate paths | — | `SPEC ONLY` |
| **ProcTHOR `pc_norm`** | — | — | `prepare_depth_shell` | — | `FIXED`, `SPEC ONLY` |

Two `FIXED, SPEC ONLY` rows are honest about their limits: both fixes are
correct in the code and neither has a test, because both need a loaded ULIP-2
backbone (9.5 GB) to exercise end to end. They will be covered by the first
smoke run, and until then this table says so rather than implying coverage.

---

## F.4 Stage 2

| item | paper | graph | code | test | status |
|---|---|---|---|---|---|
| **ESSGNN exists in the model** | §2.6 | ✓ | **`build_stage2_model`** | — | `FIXED`, `SPEC ONLY` |
| only fuser + ESSGNN update | §2.6 | ✓ | `freeze_for_stage2` | frozen-scope tests | `CONSISTENT` |
| query encoders frozen | §2.6 | ✓ | `no_grad` + `requires_grad_(False)` | `L1-STAGE2-QUERY-ENCODERS-FROZEN` | `CONSISTENT` |
| Eq. 7/8 bidirectional | §2.6 | ✓ | `bidirectional=True` | symmetry test | `CONSISTENT` |
| **scene dropout ≠ `p_mask`** | §2.6, two rates | ✓ | `PAPER_SCENE_DROPOUT` | — | `FIXED`, `SPEC ONLY` |
| dropout per batch | §2.6 | `granularity=batch` | one draw per batch | `L1-SCENE-DROPOUT-30` | `CONSISTENT` |
| target removed (U-08d) | silent | ✓ | `build_context_graph` | target-absence, edge-removal, remap | `CONSISTENT` |
| unique positive (U-08e) | silent | ✓ | `unique_positive_batches` | 6 tests incl. the degenerate case | `CONSISTENT` |
| same-assetId positive (U-08a) | silent | ✓ | `positive_map` lookup | — | `SPEC ONLY` |
| **batches reshuffled per epoch** | silent | — | inside the epoch loop | — | `FIXED`, `SPEC ONLY` |
| **layout with nodes but no edges** | silent | — | `if keep:` | — | `FIXED`, `SPEC ONLY` |
| **ESSGNN widths** | — | measured, not decided | from n08's artifacts | — | `FIXED`, `SPEC ONLY` |
| **Stage 2 saves `loss_fn`** | — | — | `loss_trainable_state` | — | `FIXED`, `SPEC ONLY` |
| **`--variant` reaches the model** | Table 3 | `variant_registry` | `load_variant` | 6 tests | `FIXED` |
| n13 has ever run | — | — | **never** | — | `OPEN` |

---

## F.5 Data pipeline

| item | paper | graph | code | test | status |
|---|---|---|---|---|---|
| 11 views | §2.3 | ✓ | `renders.py` | render tests | `CONSISTENT` |
| GPT-4o annotation | §2.3 | recorded `[DEVIATION]` (local Qwen) | `annotate.py` | schema tests | `CONSISTENT` |
| 10,000 points | **unstated** | 10k | `pointclouds.py` | count tests | `CONSISTENT`, relabelled `[IMPLEMENTATION CHOICE]` |
| `pc_norm` at write | `[UPSTREAM]` | ✓ | `pointclouds.py` | matches-ULIP tests | `CONSISTENT` |
| 80/20 split | §2.3 | ✓ | `splits.py`, `scene_splits.py` | split tests | `CONSISTENT` |
| bbox cross-check | — | ✓ | `procthor_modalities.py` | bbox tests | `CONSISTENT` |
| physical edges → `\mathcal{N}(i)` | **silent** (S5) | `neighbourhood_only` | `scene_graphs.py` | — | `SPEC ONLY`, relabelled |
| `e_ij` never sees `x` | appendix premise | ✓ | text-only encoder | — | `SPEC ONLY` |

The last row deserves naming: the entire equivariance argument rests on `e_ij`
being independent of `x`, and nothing currently asserts it. It holds by
construction — `semantic_edges.py` takes two text descriptions — but "holds by
construction" is what was said about the four things this pass found.

---

## F.6 Audit tooling

| item | code | test | status |
|---|---|---|---|
| TeX include-tree resolution | `build_source_manifest.py` | comments stripped before walking | `FIXED` — counted `%\input` as included |
| formula extraction | `build_formula_inventory.py` | substring + SHA256 + round-trip + uniqueness | `FIXED` — `$$` was never scanned |
| arXiv ids | `build_source_manifest.py` | — | `FIXED` — MetaFind's pointed at another paper |
| commented-out equations excluded | `strip_comments` | 3 excluded in EGNN, counted | `CONSISTENT` |
| equation numbers are ours | documented in A | — | `SPEC ONLY` |

---

## F.7 Scoreboard

| status | count |
|---|---|
| `CONSISTENT` | 30 |
| `FIXED` | 22 |
| `SPEC ONLY` | 16 |
| `OPEN` | 3 |

The three `OPEN` rows — `x^(L)` untested, `Stage1RuntimeConfig` uncalled, n13
never run — are recorded rather than fixed, and each says why.

The sixteen `SPEC ONLY` rows are the honest weak spot, and it grew: several of
this round's fixes are correct in the code with no test, because they need a
loaded 9.5 GB backbone or a completed run. That is the same reason the bugs
survived in the first place. The queued Stage 1 smoke converts most of them, and
until it passes this table says `SPEC ONLY` rather than implying coverage.

**`FIXED` more than doubled between rounds.** Nine of those came from an
external review of the first nine; two came from the first round's own fix. A
fix is not a place to stop checking.
