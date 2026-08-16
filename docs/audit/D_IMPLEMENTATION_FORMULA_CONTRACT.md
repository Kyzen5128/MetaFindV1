# D. IMPLEMENTATION_FORMULA_CONTRACT

The formulas the code actually computes, and where each departs from the paper.

This is the operative document: A says what the papers contain, B says what is
inherited, C says where the paper conflicts with itself, and D says **what we
run**. Where they disagree, D is what a reader should check the code against.

Formula ids are A's; contradiction ids `C1`–`C8` and silences `S1`–`S6` are C's.
Every departure carries a label:

| label | meaning |
|---|---|
| `[PAPER]` | the paper states it and we do it |
| `[PAPER CONTRADICTION]` | the paper states two incompatible things; we chose |
| `[IMPLEMENTATION CHOICE]` | the paper is silent; the choice is ours |
| `[UPSTREAM]` | evidence from ULIP-2 or EGNN, used only where noted |
| `[DEVIATION]` | we knowingly differ from a clear statement |

**There are no `[DEVIATION]` rows.** Everywhere the paper is unambiguous we
follow it, including where following it is worse (Stage 1's asymmetry, C7).

---

## D.0 Status

| formula | what it is | code | status |
|---|---|---|---|
| `MF-1` | retrieval argmax | [gallery_index.py](../../metafind/train/gallery_index.py) | `[PAPER]` |
| `MF-U1` | `h^(0) = Concat(x_i, t_i)` | [essgnn.py](../../metafind/models/essgnn.py) | `[PAPER CONTRADICTION]` — C2 |
| `MF-14` | feature update (**primary**) | `ESSGCLShared` in [essgnn.py](../../metafind/models/essgnn.py) | `[PAPER]` |
| `MF-10` | shared message `m_ij` (**primary**) | `ESSGCLShared` | `[PAPER]` |
| `MF-13` | coordinate update (**primary**) | `ESSGCLShared` | `[PAPER CONTRADICTION]` — C3 |
| `MF-2` | feature update (2.5, competing) | `ESSGCL` | `[PAPER]` |
| `MF-3` | coordinate update | [essgnn.py](../../metafind/models/essgnn.py) | `[PAPER CONTRADICTION]` — C3 |
| `MF-U4` | `e_layout = Pooling(...)` | [essgnn.py](../../metafind/models/essgnn.py) | `[IMPLEMENTATION CHOICE]` — S2 |
| `MF-4` | SE(3) equivariance | [test_essgnn.py](../../tests/test_essgnn.py), [test_cuda_smoke.py](../../tests/test_cuda_smoke.py) | tested, CPU and CUDA |
| `MF-5` | Stage 1 loss | [losses.py](../../metafind/models/losses.py) | `[PAPER]` |
| `MF-6` | Eq. 6 residual fusion | [dual_tower.py](../../metafind/models/dual_tower.py) | `[PAPER]` |
| `MF-7` | q2g and g2q | [losses.py](../../metafind/models/losses.py) | `[PAPER]`, g2q via transpose |
| `MF-8` | ½(q2g + g2q) | [losses.py](../../metafind/models/losses.py) | `[PAPER]` |
| `MF-9`–`MF-15` | appendix proof | — | obligations, D.4 |

---

## D.1 ESSGNN

### `MF-U1` — `h^(0)` — `[PAPER CONTRADICTION]`

```latex
h_i^{(0)} = \text{Concat}(x_i, t_i).
```

**We compute `h^(0) = t_i`.** Default `h0_mode="semantic"`.

Why: C2 and C4. The appendix's own premise requires `h^0` invariant to SE(3) on
`x`; `Concat(x_i, t_i)` is not, and the widths do not close either.

The printed form remains runnable as `h0_mode="concat_xt"` and is the **negative
control**: `test_equivariance_negative_injection` asserts it *breaks* invariance
(`err > 1e-3`). Without that assertion the positive test would pass on a model
that ignored coordinates entirely.

### `MF-2` — feature update — `[PAPER]`

```latex
h_i^{l+1} &= h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h), \\
```

```python
m_h = self.f_h(torch.cat([h[row], h[col], radial, edge_attr], dim=-1))
h_next = h + unsorted_segment_sum(m_h, row, num_segments=h.size(0))
```

Residual outside `f_h`; sum over `edge_index`, i.e. `\mathcal{N}(i)` (resolving
C5 toward §2.5); one MLP seeing the raw tuple — the §2.5 architecture, not the
appendix's.

`_mlp` is `Linear→SiLU→Linear`. §2.5 says only "which we approximate using
multilayer perceptrons (MLPs)" — no depth, no activation. **U-35,
`[IMPLEMENTATION CHOICE]`.** EGNN's appendix gives three different shapes; a
dependency's appendix supplies a variant list, never the answer.

### `MF-3` — coordinate update — `[PAPER CONTRADICTION]` on the output type

```latex
x_i^{l+1} &= x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x),
```

```python
self.f_x = _mlp(2 * h + 1 + edge_dim, h, 1)      # -> R^1, NOT R^3
w = self.f_x(torch.cat([h_for_x[row], h_for_x[col], radial, edge_attr], dim=-1))
trans = coord_diff * w                            # scalar * vector
```

| aspect | paper | ours | label |
|---|---|---|---|
| `f_x` codomain | `\mathbb{R}^{3}` | `\mathbb{R}^1` | `[PAPER CONTRADICTION]` — C3 |
| features into `f_x` | `h^{l+1}` | `h^{l+1}` (`coord_feat="updated"`) | `[PAPER]` |
| aggregation | sum | sum (`coords_agg="sum"`) | `[PAPER]`; reference EGNN defaults to mean |
| normalising `C` | absent | absent | `[PAPER]`; EGNN has `1/(M-1)` (B.2.2) |

**`f_x`'s codomain gets no config flag, deliberately.** The other §2.5-vs-appendix
splits are flags because both readings are mathematically valid. This one is
not: the literal reading breaks Eq. (4)'s own claim, and a flag would imply a
legitimate choice exists. `[UPSTREAM]` agrees in prose — EGNN's `φ_x` "outputs a
scalar value" — but the resolution rests on MetaFind's own proof, not on EGNN.

### `MF-U4` — pooling — `[IMPLEMENTATION CHOICE]` (S2)

Default `mean`; `sum` and `max` available; recorded in `essgnn_arch_protocol`.
Mean because `sum` makes `e_layout`'s magnitude scale with room size, entangling
the pooling choice with the learnable `λ`. **Not attributable to the paper.**

### The missing semantic-edge token — `[IMPLEMENTATION CHOICE]` (U-30)

The paper never mentions absent relations. Our rule (`L1-SEMEDGE-NO-ZEROFILL`):
zero is a valid point in the edge space and a zero-fill is indistinguishable
from a real relation, so absence must be represented explicitly.

`build_context_graph` returns an `edge_missing` bool mask; `ESSGNN.forward`
substitutes `self.missing_edge_token`, an `nn.Parameter`. It was a seeded numpy
constant while `essgnn_edge_protocol` already recorded
`semantic_missing_representation = learned_missing_token` — the protocol
described something that got no gradient, entered no optimizer and reached no
checkpoint. `test_missing_edge_token_gets_gradient` is the assertion.

### Other ESSGNN settings

| setting | value | basis |
|---|---|---|
| `distance` | `squared` | C6 — appendix and EGNN; `euclidean` follows §2.5. `[PAPER CONTRADICTION]` |
| `n_layers` | 4 | S4 — no `L` in the paper. `[IMPLEMENTATION CHOICE]` |
| `hidden_dim` | 128 | S4 |
| `layer_sharing` | `independent` | U-31 — `θ_h`/`θ_x` carry no layer index |
| `use_io_projections` | True | U-33 — §2.5 has no projections; reference EGNN does |
| `edge_proj_dim` | None | the paper has no such layer |
| `normalize_coord_diff` | False | nor any normalisation of `(x_i - x_j)` |

---

## D.2 Eq. 6 — `MF-6` — `[PAPER]`

```latex
e_{\text{query}} = \text{Fusion}(e_{\text{text}}, e_{\text{img}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}},
```

```python
self.layout_weight = nn.Parameter(torch.tensor(float(cfg.init_lambda)))
```

- `λ` is **learnable**, which the paper states outright, and used **directly**:
  no `exp`, no `sigmoid`, no clamp. The paper gives no positivity constraint.
- Named `layout_weight` **because** an earlier name, `log_lambda`, described a
  log while the code used it linearly — a name that invited a "fix" adding
  `exp()`, moving `λ` from 1.0 to *e* at initialisation. The name is part of the
  contract.
- `init_lambda = 1.0` is ours (S4).
- **Enforced:** `essgnn.out_dim == cfg.dim`, raised at construction. Eq. 6 adds;
  a width mismatch is a different model, not a broadcast.

---

## D.3 The losses

| | Stage 1 | Stage 2 |
|---|---|---|
| formula | `MF-5` | `MF-7`, `MF-8` |
| direction | **unidirectional** | **bidirectional**, averaged |
| `bidirectional` | `False` | `True` |
| basis | `[PAPER]` — C7 | `[PAPER]` |

Three things to hold onto:

1. **The asymmetry is the paper's** (C7). Making Stage 1 symmetric "because
   ULIP-2 is" would silently change the experiment; B.1.1 shows why the
   resemblance is not authority.
2. **`MF-7`'s g2q direction is computed by transposing the q2g logit matrix.**
   Exact only because `sim` is symmetric — true for our cosine choice (S1) and
   false for an asymmetric similarity. A protocol naming a non-cosine `sim` is
   refused rather than quietly transposed.
   *(Note on naming: "Eq. 7a/7b" is **ours**. The TeX has a single numbered
   equation (7) holding both directions; the split came from the Markdown
   conversion. Code comments still use 7a/7b and that is fine as long as nobody
   cites it as the paper's numbering.)*
3. **`τ` is learnable, initialised at 0.07.** Learnable has `[UPSTREAM]` support —
   ULIP-2's TeX says "$\tau$ is a learnable temperature parameter". `0.07` does
   **not**: it is CLIP's convention via ULIP's code, `[IMPLEMENTATION CHOICE]`.
   MetaFind calls `τ` "a temperature hyperparameter", which if anything reads as
   fixed. Both arrive through a hashed hyperparameter artifact.
   Stage 2 **inherits** Stage 1's learned `τ` as its initialisation — the paper
   says nothing about `τ` across stages, so `[IMPLEMENTATION CHOICE]`.

---

## D.4 Appendix — obligations, not code

| formula | obligation |
|---|---|
| `MF-10` | `e_ij` must never be a function of `x`. The semantic-edge encoder sees text only. |
| `MF-11` | the distance fed to the message MLP is recomputed from live coordinates, never cached across a transform |
| `MF-13`, `MF-U15` | `φ_x` scalar-valued (D.1, C3) |
| `MF-14` | residual outside the message function |
| `MF-9`, `MF-15` | the end-to-end property `MF-4` asserts |

`MF-4` is tested at `n_layers=3` on CPU and CUDA. **`L ≥ 2` is required** (S6):
`x^(L)` is never read, so at `L = 1` the coordinate channel is write-only and the
whole class of coordinate bugs is invisible. At `L ≥ 2`, `x^l` re-enters through
`d_{ij}^l`, so a coordinate update that changes pairwise distances breaks
h-invariance and is caught.

**Known limit:** the test asserts h-invariance only. `ESSGNN.forward` returns
pooled `h` and never exposes `x^(L)`, so a coordinate breakage that *preserves*
pairwise distances — a reflection, a coordinate permutation — would pass. Stated
rather than papered over.

---

## D.5 Data-path contracts

### Point cloud normalisation — `[IMPLEMENTATION CHOICE]`

Two corpora, one PointBERT:

| corpus | stored as | fed to `encode_pc` as |
|---|---|---|
| Objaverse (n03) | already `pc_norm`'d | unchanged |
| ProcTHOR (n07b) | **world frame**, asset lifted to y = 40 m | `prepare_depth_shell` → `pc_norm` + grey |

The world frame stays on disk because that is what the AI2-THOR bounding-box
check compares against — the check that caught the orthographic-depth bug.
Normalisation happens at encode time instead.

Without it the two galleries sat in different input distributions — **a shift
our own preprocessing created**, distinct from the unavoidable one (complete
mesh surface versus visible depth shell), which is declared and not fixable.

### Checkpoint completeness — `[IMPLEMENTATION CHOICE]`

Stage 1's optimizer moves three modules, so the checkpoint has three sections:

```
stage1.pt
├── backbone_trainable_state    point_encoder.*, pc_projection
├── tower_trainable_state       query.fusion.*, gallery.fusion.*
├── loss_trainable_state        logit_scale (when learnable)
└── trainer_version, epoch, train_scope
```

`assert_checkpoint_covers_optimizer` compares by tensor **identity** on every
save. `load_stage1_checkpoint` refuses a file missing a section, refuses a
section that omits a trainable tensor, and admits genuinely-new modules only
through a declared `new_prefixes` — Stage 2 passes
`("query.layout_encoder", "layout_weight")` because 2.6 introduces them there.

`gallery_encoder_sha256` covers backbone **and** gallery fusion. Hashing the
fusion alone gave two runs with different fine-tuned PointBERTs the same digest.

### Scene dropout is not modality masking — `[PAPER]`

Both are 30%, and that is what made the alias invisible. §2.6 masks each
**modality** independently at 30% in Stage 1, and drops the layout vector in 30%
of **batches** in Stage 2. Separate constants (`PAPER_P_MASK`,
`PAPER_SCENE_DROPOUT`); Table 3's `p_mask` sweep no longer moves scene dropout.

---

## D.6 What is NOT implemented

### n13 has never run

[stage2.py](../../metafind/train/stage2.py) is complete — sampling, batching,
context graphs, forward, Eq. 7/8, backward, checkpoint — and carries **no
`IMPLEMENTS-NODE` marker**, because it needs `stage1_ckpt` (n10 has not run) and
`sem_edge_cache` (n08 has not run). The marker is a claim and the README's count
is computed from it; it goes on when a smoke run passes, not when the code looks
finished.

### `Stage1RuntimeConfig` is declared authority and n10 does not use it

`metafind/models/stage1_config.py` defines it as the single construction path;
`metafind/train/stage1.py` builds its configuration from the raw protocol dicts
instead. Nothing is wrong today — both read the same artifact — but the class
cannot enforce what nothing calls. Recorded in F, not silently fixed, because
routing n10 through it changes the trainer on the eve of its first run.

### U-27 — the 200 I-Design scenes

Open, and blocking **the Table 2 branch only**. New evidence this pass: I-Design's
own arXiv source publishes **60 prompts with room dimensions**
(`tabs/tab_promptlist_minimal.tex`, 20 short-form; `tab_promptlist_others.tex`,
40 detailed), which gives a citable basis for constructing 200 rather than
inventing them. It also **removes one unknown**: I-Design's paper describes its
input as "an unstructured, grammar-free natural language user input" plus room
dimensions — **object count is not a user input**, it is decided by the agents.
The registry's `IDesign(no_of_objects, user_input, room_dimensions)` describes
the released code's entry point, which is a measured implementation fact and
should be labelled as one.

---

## D.7 Admissibility

No claim that MetaFind's mathematics is fully reproduced is admissible while:

- **C1 is decided but is an `[INFERENCE]`** (U-26, 2026-08-17). The primary is
  the appendix's shared-message form; 2.5 remains implemented and must be
  measured against it. The paper never says which it ran, so no result may be
  reported as "the paper's architecture" -- only as one of two readings.
- **Neither stage has run.** Every contract here is checked by tests, not by a
  completed training run.
- **S4 stands.** No layer count, widths, batch size, learning rate, optimiser,
  epochs, `τ` value or OpenCLIP variant exists in the paper. Every number in a
  results table of ours comes from a configuration the paper does not specify.
- **C8 stands.** No evaluation of ours may report scale robustness as a
  reproduced property; SE(3) contains no scaling.
