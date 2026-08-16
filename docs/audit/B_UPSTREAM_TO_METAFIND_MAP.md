# B. UPSTREAM_TO_METAFIND_MAP

What MetaFind takes from ULIP-2 and EGNN, formula by formula, from the authors'
arXiv TeX source.

Formula ids are A's (`A_FORMULA_INVENTORY.md`); every quoted expression is in
`formula_inventory.json` with a SHA256 that the build asserts against the `.tex`.
Sources: `docs/paper/{metafind,ulip2,egnn}_source/`.

The rule this document enforces: **resemblance is not inheritance.** MetaFind's
Eq. (10) and EGNN's Eq. (3) differ by one symbol, and that symbol changes what
the term means. MetaFind's Eq. (5) and ULIP-2's Eq. (1) look like the same
InfoNCE and are not the same objective. Reading either as "same as upstream, so
use the upstream code" is the mistake this audit exists to stop.

---

## B.0 What each upstream supplies

| upstream | what MetaFind takes | what it does NOT take |
|---|---|---|
| ULIP-2 | a pre-trained **object-level** tri-modal embedding space, used as a frozen-except-`E_P` feature extractor | ULIP-2's losses. MetaFind never restates Eq. (1)/(2)/(3); its Stage 1 loss is its own. |
| EGNN | the **equivariant message-passing scheme** and its equivariance proof structure | task heads, velocity channel, edge inference, and the normalising constant `C` |

Neither upstream is a scene model; neither is a retrieval model. They meet only
at MetaFind Eq. (6):

```latex
e_{\text{query}} = \text{Fusion}(e_{\text{text}}, e_{\text{img}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}},
```

The left summand is where ULIP-2 enters; `e_layout` is where EGNN enters. Before
that point they never interact, and no upstream fact about one constrains the
other.

MetaFind's own words, `2methdology.tex`: ESSGNN is "a **modified** Equivariant
Graph Convolutional Layer (EGCL) structure", and the towers are "both leveraging
the ULIP-2 embedding backbone". `MODIFIED` is the paper's word, not our label.

---

## B.1 ULIP-2 → MetaFind

### B.1.1 `U2-1`, `U2-2` → `MF-5` — **CONCEPTUAL_SOURCE**

ULIP-2 Eq. (1), `main.tex` line 614:

```latex
\mathcal{L}_\text{P2I}=-\frac{1}{2}\sum_i\log\frac{\exp(\mathbf{f}_i^\text{P} \mathbf{f}_i^\text{I}/\tau)}{\sum_j\exp(\mathbf{f}_i^\text{P} \mathbf{f}_j^\text{I}/\tau)}+\log\frac{\exp(\mathbf{f}_i^\text{P} \mathbf{f}_i^\text{I}/\tau)}{\sum_j\exp(\mathbf{f}_j^\text{P} \mathbf{f}_i^\text{I}/\tau)},
```

MetaFind Eq. (5), `2methdology.tex`:

```latex
\mathcal{L}_{\text{pre}} = -\log \frac{\exp(\text{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A)) / \tau)}{\sum_{A' \in \mathcal{B}} \exp(\text{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A')) / \tau)},
```

| | ULIP-2 (1)/(2) | MetaFind (5) |
|---|---|---|
| direction | **bidirectional** — two terms, one normalised over `j` in each index position, with the ½ | **unidirectional** — query→gallery only |
| what is aligned | two views of the **same object** (its cloud, its render, its caption) | a **query tower** against a **gallery tower** |
| similarity | dot product of `\mathbf{f}` vectors, written out | `\text{sim}(\cdot,\cdot)`, never defined ([S1](C_PAPER_CONTRADICTIONS.md#s1)) |
| temperature | "$\tau$ is a **learnable** temperature parameter" (`main.tex` line 616) | "$\tau$ is a temperature **hyperparameter**" |

The direction row matters most and MetaFind never comments on it. Its own
Stage 2 (`MF-7`, `MF-8`) *is* bidirectional and says so. So Stage 1 as written
is strictly weaker than both its backbone and its own second stage. See
[C7](C_PAPER_CONTRADICTIONS.md#c7).

**Verdict:** `CONCEPTUAL_SOURCE`. Implementing Eq. (5) by calling a ULIP-2 loss
would silently make Stage 1 bidirectional.

### B.1.2 `U2-3` → MetaFind Stage 1 — **CONCEPTUAL_SOURCE**

```latex
\min_{E_\text{P}}\mathcal{L}_\text{P2I}+\mathcal{L}_\text{P2T}.
```

The subscript is the content. ULIP-2's TeX is explicit twice over:

> "based on the pre-aligned and **frozen** image encoder $E_\text{I}$ and text
> encoder $E_\text{T}$ in OpenCLIP" — `main.tex`
>
> "We adopt the largest version of encoders from OpenCLIP (ViT-G/14) … and
> **freeze it during the pre-training**." — `main.tex`

MetaFind Stage 1: "**both** query and gallery encoders are trained … both towers
share the contrastive retrieval objective."

Compatible only if "trained" means the fusers and the point tower while
text/image stay frozen. That is **U-34**, and the TeX now settles the upstream
half of it: ULIP-2 unambiguously freezes OpenCLIP and optimises only `E_P`. The
MetaFind half remains an inference — the paper never states a change of policy,
and its §2.4 calls the towers "leveraging the ULIP-2 embedding backbone", which
a fine-tuned CLIP would no longer be. Recorded `[INFERENCE]`, not `[PAPER]`.

### B.1.3 What is genuinely inherited

The encoders, and only at object level:

| ULIP-2 component | TeX evidence | MetaFind use | relationship |
|---|---|---|---|
| OpenCLIP ViT-G/14 `E_I`, `E_T` | "freeze it during the pre-training" | text/image towers | `DIRECTLY_REUSED`, frozen |
| point encoder `E_P` | "We target to train a 3D point cloud encoder $E_\text{P}$" | pc tower | `DIRECTLY_REUSED`, trainable in Stage 1 |
| the pre-aligned joint space | "The feature space, already pre-aligned by OpenCLIP, serves as the target space" | what Eq. (6) adds into | `CONCEPTUAL_SOURCE` |
| `L_P2I`, `L_P2T`, `min_{E_P}` | Eq. (1)–(3) | — | `CONCEPTUAL_SOURCE`, not reused |
| BLIP-2 caption generation | §3.2 | — | `NOT_USED` (MetaFind annotates with GPT-4o) |
| 10k xyzrgb / 8k / 2k point sampling | ablation table | our n03 samples 10k | `UNKNOWN` — MetaFind states no point count |
| X-InstructBLIP 3D captioning | downstream experiment | — | `NOT_USED` |

**`PointBERT` is not named in ULIP-2's method section** — it appears in related
work and in a downstream-comparison context. Which point encoder MetaFind uses
is therefore not settled by ULIP-2's paper either; our choice follows the
released ULIP-2 checkpoint and is an `[IMPLEMENTATION CHOICE]`.

Any argument of the form "ULIP-2 does X, therefore MetaFind does X" is invalid
for everything except the object-level embedding space itself.

---

## B.2 EGNN → ESSGNN

### B.2.1 `EG-3` → `MF-10` — **MODIFIED** (one symbol, changed meaning)

```latex
\rmm_{ij} &=\phi_{e}\left(\rmh_{i}^{l}, \rmh_{j}^{l},\left\|\rmx_{i}^{l}-\rmx_{j}^{l}\right\|^{2}, a_{i j}\right)   % EGNN (3)
\rm m_{ij} = \phi_e\left(\rm h_i^l, \rm h_j^l, \| \rm x_i^l - \rm x_j^l \|^2, e_{ij} \right)                        % MetaFind (10)
```

Identical but for `a_{ij}` → `e_{ij}`. The substitution is the contribution:

- EGNN's `a_ij`: MetaFind's own appendix describes these as "typically discrete,
  task-specific features such as bond types or edge labels".
- MetaFind's `e_ij`: "edge embeddings derived from LLM-generated natural language
  relation descriptions, which are subsequently encoded via a frozen text
  encoder".

The appendix's argument for why the swap is legal is one property only: `e_ij`
is "**invariant to the input node positions**". That is what our pipeline must
preserve — the semantic-edge encoder may never see a coordinate. It is also why
a missing semantic edge may not be zero-filled: zero is a valid point in the
embedding space and would read as a real relation, not as absence
(`L1-SEMEDGE-NO-ZEROFILL`).

### B.2.2 `EG-4` → `MF-13` — **MODIFIED** (`C` removed)

```latex
\rmx_{i}^{l+1} &=\rmx_{i}^{l}+ C\sum_{j \neq i}\left(\rmx_{i}^{l}-\rmx_{j}^{l}\right) \phi_{x}\left(\rmm_{ij}\right)   % EGNN (4)
\rm x_i^{l+1} = \rm x_i^l + \sum_{j \ne i} (\rm x_i^l - \rm x_j^l) \cdot \phi_x(\rm m_{ij})                            % MetaFind (13)
```

EGNN's TeX defines it: "$C$ is chosen to be $1/(M-1)$, which divides the sum by
its number of elements." MetaFind drops it silently. ProcTHOR rooms vary widely
in object count, so with and without `C` are measurably different models.
Restoring it is **not** licensed — an upstream implementation detail cannot fill
in what MetaFind wrote differently. Ours follows MetaFind; if instability
appears, adding `C` becomes a recorded `[DEVIATION]`.

### B.2.3 `EG-5` + `EG-6` → `MF-14` — **MODIFIED** (residual added, signature cut)

```latex
\rmm_{i} &=\sum_{j \neq i} \rmm_{ij}                        % EGNN (5)
\rmh_{i}^{l+1} &=\phi_{h}\left(\rmh_{i}^l, \rmm_{i}\right)  % EGNN (6)
\rm h_i^{l+1} = \rm h_i^l + \sum_{j \ne i} \phi_h(\rm m_{ij})   % MetaFind (14)
```

Two changes: MetaFind makes the update **residual**, and moves `φ_h` inside the
sum with `h_i^l` no longer among its arguments. EGNN aggregates then transforms
once; MetaFind transforms each message then aggregates. Different functions, and
`φ_h`'s input width differs accordingly.

### B.2.4 `EG-1` → `MF-4`, `MF-9`, `MF-15` — **CONCEPTUAL_SOURCE**

```latex
\phi(T_g(\rmx)) = S_g(\phi(\rmx))
```

The general definition. MetaFind instantiates it at SE(3) and adds the edge set
as an explicit argument — `\text{ESSGNN}(R x^l + T, h^l, E)` versus
`\mathrm{EGCL}(Q\rmx^l + g, \rmh^l)` — which is the notational marker of the
`a_ij → e_ij` substitution.

### B.2.5 `φ_x` output type — the one upstream fact that decides a MetaFind contradiction

EGNN's TeX states it in prose, not only in a formula:

> "The weights of this sum are provided as the output of the function
> $\phi_x: \mathbb{R}^{\text{nf}} \rightarrow \mathbb{R}^1$ that takes as input
> the edge embedding $\rmm_{ij}$ … and **outputs a scalar value**."

MetaFind's §2.5 types `f_x: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^{3}`.
See [C3](C_PAPER_CONTRADICTIONS.md#c3). This is `UPSTREAM` evidence and is used
only to *characterise* MetaFind's internal contradiction — MetaFind's own
appendix proof independently requires the scalar, so the resolution does not
rest on EGNN.

### B.2.6 NOT_USED — 21 of EGNN's 33

Appearing in the EGNN paper is not evidence MetaFind uses it.

| EGNN | why not used |
|---|---|
| `EG-7` (velocity) | MetaFind's scene graphs have no velocity channel; objects are static |
| `EG-8` (edge inference) | defines `m_i = \sum_{j \neq i} e_{ij}\rmm_{ij}` — `e_ij` as a **scalar gate**. MetaFind's `e_ij` is a sentence **vector** inside `φ_e`. Same two characters, incompatible types. |
| graph autoencoder, QM9 / N-body heads | different tasks |
| Appendix A / B.1 proof steps | proof obligations, not layers |
| Appendix E norm-uniqueness proofs | not referenced by MetaFind |

`EG-8` is the trap worth naming twice: porting EGNN's edge-inference code
because the symbol matched would typecheck in neither direction.

---

## B.3 Where the two upstreams are wired together

Only Eq. (6), and only additively. Consequences:

1. `e_layout` must be **the same width** as the fused ULIP-2 embedding, since it
   is added, not concatenated. Nothing in either upstream fixes that width, and
   `Pooling` is unspecified ([S2](C_PAPER_CONTRADICTIONS.md#s2)).
2. `λ` is **learnable** — "where $\lambda$ is a learnable scalar controlling the
   contribution of layout information". A plain scalar coefficient; not a log,
   an exponent, or a gate.
3. "This residual design allows layout reasoning to enhance retrieval without
   disrupting the original embedding space" holds only if the ULIP-2 space is in
   fact preserved — independent support for the U-34 reading in B.1.2.

---

## B.4 Summary

| upstream | MetaFind | relationship |
|---|---|---|
| `U2-1`, `U2-2` | `MF-5` | CONCEPTUAL_SOURCE — bidirectional → unidirectional; same-object → cross-tower |
| `U2-3` | Stage 1 prose | CONCEPTUAL_SOURCE — `min` over `E_P` only → "both towers" |
| OpenCLIP `E_I`/`E_T` | text/image towers | DIRECTLY_REUSED, frozen |
| point encoder `E_P` | pc tower | DIRECTLY_REUSED, trainable in Stage 1 |
| `EG-1` | `MF-4`/`MF-9`/`MF-15` | CONCEPTUAL_SOURCE — instantiated at SE(3), `E` added |
| `EG-3` | `MF-10` | MODIFIED — `a_ij` (discrete) → `e_ij` (sentence embedding) |
| `EG-4` | `MF-13` | MODIFIED — `C = 1/(M-1)` removed |
| `EG-5`+`EG-6` | `MF-14` | MODIFIED — residual added; `φ_h` moved inside the sum |
| `EG-2` | — | CONCEPTUAL_SOURCE — the generic GNN recap |
| `φ_x → R^1` (prose) | contradicts `MF-3` | UPSTREAM evidence for [C3](C_PAPER_CONTRADICTIONS.md#c3) |
| 21 others | — | NOT_USED |

**Absent from this table:** `MF-2` and `MF-3` have no upstream row. They are not
a modification of any EGNN equation — they are a second, different ESSGNN, and
reconciling them with the appendix is [C1](C_PAPER_CONTRADICTIONS.md#c1).
