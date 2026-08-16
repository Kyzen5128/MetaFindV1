# D. IMPLEMENTATION_FORMULA_CONTRACT

For every MetaFind formula: which code implements it, exactly how it departs
from the printed form, and what test would catch the departure being wrong.

The contract this document enforces: **no line of model code may exist without a
row here.** A formula with no code is unimplemented and must say so. Code with
no formula is ours, and must be labelled `[IMPLEMENTATION CHOICE]`, never
`[PAPER]`.

Formula ids are A's. Contradiction ids `C1`–`C7` and silence ids `S1`–`S5` are
C's. Upstream relationships are B's.

---

## D.0 Status at a glance

| formula | what it is | code | status |
|---|---|---|---|
| `MF-1` | retrieval argmax | [gallery_index.py](../../metafind/train/gallery_index.py) | implemented |
| `MF-U1` | `h^(0) = Concat(x_i, t_i)` | [essgnn.py:403](../../metafind/models/essgnn.py#L403) | **deviates — C2** |
| `MF-2` | feature update | [essgnn.py:311-313](../../metafind/models/essgnn.py#L311-L313) | implemented |
| `MF-3` | coordinate update | [essgnn.py:315-324](../../metafind/models/essgnn.py#L315-L324) | **deviates — C3** |
| `MF-U4` | `e_layout = Pooling(...)` | [essgnn.py:414](../../metafind/models/essgnn.py#L414) | choice — S2 |
| `MF-4` | SE(3) equivariance claim | [test_essgnn.py:90](../../tests/test_essgnn.py#L90) | **partially tested** |
| `MF-5` | Stage 1 loss | [losses.py:141-144](../../metafind/models/losses.py#L141-L144) | implemented |
| `MF-6` | Eq. 6 residual fusion | [dual_tower.py:201-216](../../metafind/models/dual_tower.py#L201-L216) | implemented |
| `MF-7a` | q→g direction | [losses.py:134](../../metafind/models/losses.py#L134) | implemented |
| `MF-7b` | g→q direction | [losses.py:149](../../metafind/models/losses.py#L149) | implemented, **via transpose** |
| `MF-8` | ½(7a+7b) | [losses.py:152](../../metafind/models/losses.py#L152) | implemented |
| `MF-9`–`MF-15` | Appendix C proof | — | proof obligations, see D.4 |

Two rows are not green and are stated plainly in D.5: Stage 2's **training
loop** ([stage2.py](../../metafind/train/stage2.py)) has no forward/loss/backward and
therefore carries no `IMPLEMENTS-NODE` marker; and `MF-4`'s test asserts only
half of what its own docstring claims.

---

## D.1 ESSGNN — `MF-U1`, `MF-2`, `MF-3`, `MF-U4`

### `MF-U1` — `h^(0) = Concat(x_i, t_i)` — **DEVIATION, recorded**

```latex
h_i^{(0)} = \operatorname{Concat}(x_i, t_i)
```

Code ([essgnn.py:403](../../metafind/models/essgnn.py#L403)):

```python
h0 = torch.cat([pos, node_feat], dim=-1) if self.cfg.h0_mode == "concat_xt" else node_feat
```

Default is `h0_mode="semantic"`, i.e. **`h^(0) = t_i`** — not what §2.5 prints.

- **Why:** C2 and C4. Appendix C's own premise requires `h^0` invariant; the
  printed form is not, and the widths do not close either.
- **Label:** `[PAPER CONTRADICTION]`, resolved toward the theorem.
- **The printed form is still runnable** — `h0_mode="concat_xt"` — and it is not
  there for completeness. It is the negative control.
- **Test:** [test_equivariance_negative_injection](../../tests/test_essgnn.py#L111)
  asserts the literal §2.5 form **breaks** invariance (`err > 1e-3`). That is
  what makes the positive test non-vacuous. Without it, an implementation that
  ignored coordinates entirely would also pass.

### `MF-2` — feature update — implemented

```latex
h_i^{l+1} = h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h) \quad (2)
```

```python
m_h = self.f_h(torch.cat([h[row], h[col], radial, edge_attr], dim=-1))
h_next = h + unsorted_segment_sum(m_h, row, num_segments=h.size(0))
```

Faithful: residual outside `f_h`, sum over `edge_index` (i.e. `\mathcal{N}(i)`,
resolving C5 toward §2.5), `f_h` sees `(h_i, h_j, d, e_ij)` with no shared
message variable — the §2.5 architecture, not Appendix C's.

Unstated and therefore ours: `_mlp` is `Linear→SiLU→Linear`. §2.5 says only
"approximated using MLPs" — no depth, no activation. Registered **U-35**; note
that this shape matches EGNN's `φ_x` and matches none of EGNN's three shapes for
the feature path. That is recorded, not corrected: a dependency's appendix
supplies a variant list, never the answer.

### `MF-3` — coordinate update — **DEVIATION on the output type**

```latex
x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x) \quad (3)
```

```python
self.f_x = _mlp(2 * h + 1 + edge_dim, h, 1)          # -> R^1, NOT R^3
...
w = self.f_x(torch.cat([h_for_x[row], h_for_x[col], radial, edge_attr], dim=-1))
trans = coord_diff * w                                # scalar * vector
```

- **Departure:** §2.5's prose types `f_x : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^3`.
  We emit `\mathbb{R}^1`.
- **Why:** C3. A vector-valued `f_x` makes Eq. (3) non-equivariant, contradicting
  Eq. (4) in the same section and `MF-U16`'s factorisation in the appendix.
- **No config flag, deliberately.** The other §2.5-vs-Appendix-C splits are
  flags because both readings are mathematically valid. This one is not: the
  literal reading breaks the paper's central claim. A flag would imply a
  legitimate choice exists.
- `h_for_x` **is** `h^{l+1}` by default (`coord_feat="updated"`), following §2.5
  against Appendix C's `h^l`. That one *is* a flag, because both are equivariant.

### `MF-U4` — pooling — `[IMPLEMENTATION CHOICE]` (S2)

```latex
e_{\text{layout}} = \operatorname{Pooling}(\{h_i^{(L)}\})
```

Default `mean`; `sum` and `max` available. Mean because `e_layout` is added to
the fused embedding under a learnable `λ`, and `sum` makes the term's magnitude
scale with room size, entangling the pooling choice with `λ`'s learned value.
Recorded in `essgnn_arch_protocol`; **not** attributable to the paper.

---

## D.2 Eq. 6 — `MF-6`

```latex
e_{\text{query}} = \operatorname{Fusion}(e_{\text{text}}, e_{\text{image}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}} \quad (6)
```

```python
self.layout_weight = nn.Parameter(torch.tensor(float(cfg.init_lambda)))
```

- `λ` is a **learnable scalar**, which the paper states outright. Used directly:
  no `exp`, no `sigmoid`, no clamp. The paper gives no positivity constraint, so
  a log-parameterisation would be an unlicensed addition.
- The parameter is named `layout_weight` **because** an earlier name,
  `log_lambda`, described a log while the code used it linearly. That naming
  invited a "fix" adding `exp()`, which would have moved `λ` from 1.0 to *e* at
  initialisation and shifted every downstream number. The name is now part of
  the contract.
- `init_lambda = 1.0` is ours (S4 — the paper gives no initialisation).
- **Enforced invariant:** `essgnn.out_dim == cfg.dim`, raised at construction.
  Eq. 6 *adds*; a width mismatch is not a broadcast, it is a different model.

---

## D.3 The losses — `MF-5`, `MF-7a`, `MF-7b`, `MF-8`

`MF-5` (Stage 1) is **unidirectional**:

```python
if not self.cfg.bidirectional:
    out["loss"] = loss_q2g          # Eq. 5 -- query->gallery only
    return out
```

`MF-8` (Stage 2) averages both directions:

```python
loss_g2q = F.cross_entropy(logits_q2g.t(), labels)
out["loss"] = 0.5 * (loss_q2g + loss_g2q)
```

Three things to hold onto:

1. **The asymmetry between the stages is intentional and is the paper's** (C7).
   `bidirectional` defaults to `False`. Making Stage 1 symmetric "because ULIP-2
   is" would silently change the experiment; B.1.1 shows why the resemblance is
   not authority.
2. **`MF-7b` is computed by transposing `MF-7a`'s logit matrix.** This is exact
   only because `sim` is symmetric — which holds for our cosine choice (S1) and
   would not for an asymmetric similarity. The code carries that caveat; a
   protocol naming a non-cosine `sim` is refused rather than quietly transposed.
3. **`τ` is learnable, initialised at 0.07.** Learnable has dependency-paper
   support (ULIP-2 Eq. 1/2 say so). `0.07` does **not** — it is CLIP's
   convention, reached through ULIP's code, and is labelled `[IMPLEMENTATION
   CHOICE]` accordingly. MetaFind itself calls `τ` "a temperature
   hyperparameter", which if anything reads as fixed. Both arrive through a
   hashed hyperparameter artifact so a run cannot inherit either silently.

---

## D.4 Appendix C — `MF-9` … `MF-15` are obligations, not code

None of these compile to anything. They constrain what the code may do:

| formula | obligation on the implementation |
|---|---|
| `MF-10` | `e_ij` must never be a function of `x`. The semantic-edge encoder sees text only; a coordinate reaching it voids the equivariance argument. |
| `MF-11` | the distance fed to the message MLP must be recomputed from live coordinates, never cached across a transform |
| `MF-13`, `MF-U16` | `φ_x` scalar-valued (D.1, C3) |
| `MF-14` | residual outside the message function |
| `MF-9`, `MF-15` | the end-to-end property `MF-4` asserts |

The `MF-10` obligation is also the reason for `L1-SEMEDGE-NO-ZEROFILL`. A
missing semantic edge cannot be zero-filled: zero is a valid point in the edge
embedding space, so a zero-filled edge is indistinguishable from a real relation
that happens to embed near the origin — degraded input made invisible rather
than visible.

---

## D.5 What is NOT implemented, stated plainly

### D.5.1 Stage 2's training loop

[metafind/train/stage2.py](../../metafind/train/stage2.py) has its batching
(`unique_positive_batches`) and its context-graph construction
(`build_context_graph`) written and tested, and **no forward/loss/backward**.

It therefore carries **no `IMPLEMENTS-NODE:` marker**. The marker is a claim,
and the README's implementation count is computed from it, so a file that cannot
train must not carry one. Saving weights from this file would produce an
untrained checkpoint with a valid shape — the exact failure this project keeps
rediscovering, in the same shape as the `torch.zeros(1, 1280)` placeholder that
would have yielded a correctly-shaped gallery index of garbage.

Its two tested pieces enforce decisions that would otherwise fail silently:

- **U-08e** — no asset appears twice in a batch. With a frozen encoder one
  `assetId` has one embedding, so a duplicate is a negative bit-identical to the
  positive, and the gradient asks the model to separate two identical vectors.
- **U-08d** — the target node is removed from its own context graph, with edges
  dropped and indices remapped. Leaving it in lets ESSGNN read the answer off
  its own input; the loss falls and nothing distinguishes that from learning.

### D.5.2 `MF-4`'s test asserts half of its docstring

[test_se3_equivariance](../../tests/test_essgnn.py#L90) is documented as
"coords are equivariant and h is invariant" and asserts **only h-invariance**,
because `ESSGNN.forward` returns the pooled `h` and never exposes `x^(L)`.

In practice the coverage is better than it sounds: the test runs `n_layers=3`,
and `x^l` re-enters the feature path through `d_{ij}^l` at every subsequent
layer (S5), so a coordinate update that changed pairwise distances would break
h-invariance at layer 2 and be caught. But a breakage that *preserves* pairwise
distances — a reflection, a coordinate permutation — would pass.

**Two consequences, both recorded rather than fixed in passing:**

1. Any equivariance test must use `L ≥ 2`. At `L = 1` the coordinate channel is
   write-only and the whole class of bugs is invisible.
2. The docstring overclaims. Either the test gains a direct assertion on
   `x^(L)` — which needs `forward` to expose it — or the docstring is narrowed
   to what it checks.

### D.5.3 `U-27` — the 200 I-Design prompts

Open, and blocking the Table 2 branch **only**. It blocks nothing in Table 1 or
Table 3. Recorded here because "U-27 blocks nothing" and "U-27 blocks
everything" have both been asserted in this project and both were wrong.

---

## D.6 Admissibility

No claim that MetaFind's mathematics is fully reproduced is admissible while:

- **C1 is open.** §2.5 and Appendix C are two different ESSGNNs (U-26); until
  one is chosen there is no single specification to have reproduced.
- **Stage 2 does not train** (D.5.1).
- **S4 stands.** The paper contains no layer count, no widths, no batch size, no
  learning rate, no optimiser, no epoch count, no `τ` value and no OpenCLIP
  variant. Every number in a results table of ours is therefore produced by a
  configuration the paper does not specify, and must be reported as such rather
  than as a reproduction of the paper's numbers.
