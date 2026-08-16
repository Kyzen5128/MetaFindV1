# B. UPSTREAM_TO_METAFIND_MAP

What MetaFind actually takes from ULIP-2 and EGNN, formula by formula.

Formula ids are A's (`docs/audit/A_FORMULA_INVENTORY.md`); every quoted
expression is in A verbatim with a SHA256. Section references are to
`docs/audit/repaired/`.

The rule this document exists to enforce: **resemblance is not inheritance.**
MetaFind's Eq. (10) and EGNN's Eq. (3) differ by one symbol, and that symbol
changes what the term means. MetaFind's Eq. (5) and ULIP-2's Eq. (1) look like
the same InfoNCE and are not the same objective. Reading either as "same as
upstream, so use the upstream code" is the specific mistake this audit was
commissioned to stop.

---

## B.0 What each upstream actually supplies

| upstream | what MetaFind takes | what it does NOT take |
|---|---|---|
| ULIP-2 | a pre-trained **object-level** tri-modal embedding space, used as a frozen feature extractor | ULIP-2's losses. MetaFind never restates Eq. (1)/(2)/(3). Its Stage 1 loss is its own. |
| EGNN | the **equivariant message-passing scheme** and its equivariance proof structure | EGNN's task heads, velocity channel, edge inference, and normalising constant `C` |

Neither upstream is a scene model, and neither is a retrieval model. They meet
only at MetaFind Eq. (6):

```latex
e_{\text{query}} = \operatorname{Fusion}(e_{\text{text}}, e_{\text{image}}, e_{\text{pc}}) + \lambda \cdot e_{\text{layout}} \quad (6)
```

The left summand is where ULIP-2 enters; `e_layout` is where EGNN enters. Before
that point the two never interact, and no upstream fact about one constrains
the other.

---

## B.1 ULIP-2 → MetaFind

### B.1.1 `U2-1`, `U2-2` (L_P2I, L_P2T) → `MF-5` — **CONCEPTUAL_SOURCE, not reused**

ULIP-2 Eq. (1):

```latex
\mathcal{L}_{P2I} = -\frac{1}{2} \sum_{i} \left[ \log \frac{\exp(f_{P,i} \cdot f_{I,i} / \tau)}{\sum_{j} \exp(f_{P,i} \cdot f_{I,j} / \tau)} + \log \frac{\exp(f_{P,i} \cdot f_{I,i} / \tau)}{\sum_{j} \exp(f_{P,j} \cdot f_{I,i} / \tau)} \right] \quad (1)
```

MetaFind Eq. (5):

```latex
\mathcal{L}_{\text{pre}} = -\log \frac{\exp(\operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A)) / \tau)}{\sum_{A' \in \mathcal{B}} \exp(\operatorname{sim}(f_{\text{query}}(Q), f_{\text{gallery}}(A')) / \tau)} \quad (5)
```

Four differences, each independently load-bearing:

| | ULIP-2 (1)/(2) | MetaFind (5) |
|---|---|---|
| direction | **bidirectional** — the bracket holds both normalisations, with the ½ | **unidirectional** — query→gallery only |
| what is aligned | two views of the **same object** (its cloud, its render, its caption) | a **query tower** against a **gallery tower**, different objects per side |
| similarity | dot product, written out | `sim(·,·)`, never defined ([C7](#c7)) |
| temperature | "a **learnable** temperature parameter" | "a temperature **hyperparameter**" |

The direction row is the one that matters most and the paper never comments on
it. MetaFind's own Stage 2 (`MF-7a`/`MF-7b`/`MF-8`) *is* bidirectional, and says
so — "we adopt a bidirectional contrastive learning objective to symmetrically
align query and gallery embeddings". So Stage 1 as literally written is a
strictly weaker objective than both its own backbone and its own Stage 2. See
[C6](#c6).

**Verdict:** ULIP-2's losses are `CONCEPTUAL_SOURCE`. Implementing Eq. (5) by
calling a ULIP-2 loss would silently make Stage 1 bidirectional.

### B.1.2 `U2-3` (`\min_{E_P}`) → MetaFind Stage 1 — **CONCEPTUAL_SOURCE**

```latex
\min_{E_P} \mathcal{L}_{P2I} + \mathcal{L}_{P2T} \quad (3)
```

The subscript is the content: ULIP-2 optimises **only the point encoder**, with
OpenCLIP ViT-G/14 frozen, because the pre-aligned language-image space is the
target being joined.

MetaFind Stage 1: "**Both** query and gallery encoders are trained … both towers
share the contrastive retrieval objective." Different set of trainable
parameters, on both sides.

These are compatible only if "trained" means the fusers and the point tower
while the text/image encoders stay frozen. That reading is **U-34**, resolved
in favour of frozen OpenCLIP — the deciding evidence is MetaFind's own §2.4
("Each available modality is independently encoded using the ULIP-2 backbone"):
a backbone that is fine-tuned is no longer the ULIP-2 space that the gallery
side is defined in. Recorded as `[INFERENCE]`, not `[PAPER]`.

### B.1.3 What is genuinely inherited

Only the **encoders**, as frozen weights, and only at object level. MetaFind's
§2.4 names the backbone; it never claims ULIP-2 knows anything about scenes.
Any argument of the shape "ULIP-2 does X, so MetaFind does X" is invalid for
everything except the object-level embedding space itself.

---

## B.2 EGNN → MetaFind

MetaFind states ESSGNN "follows a **modified** EGCL structure", so `MODIFIED` is
the paper's own word, not our label.

### B.2.1 `EG-3` → `MF-10` — **MODIFIED** (one symbol, changed meaning)

```latex
m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, a_{ij} \right) \quad (3)     % EGNN
m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, e_{ij} \right) \quad (10)    % MetaFind
```

Character-for-character identical but for `a_{ij}` → `e_{ij}`. The substitution
is the paper's contribution:

- EGNN's `a_ij` is a **discrete, task-specific** edge feature (a bond type).
- MetaFind's `e_ij` is a **dense sentence embedding** of an LLM-written relation,
  encoded by a frozen text encoder.

Appendix C's argument for why the swap is legal is one property only: `e_ij`
is independent of `x`. That property is what our pipeline must preserve — the
semantic edge encoder may never see a coordinate. It is also why a missing
semantic edge may not be filled with zeros: zero is a valid point in the
embedding space and would be read as a real relation, not as absence
(`L1-SEMEDGE-NO-ZEROFILL`).

### B.2.2 `EG-4` → `MF-13` — **MODIFIED** (the constant `C` is dropped)

```latex
x_i^{l+1} = x_i^l + C \sum_{j \neq i} (x_i^l - x_j^l) \phi_x(m_{ij}) \quad (4)   % EGNN
x_i^{l+1} = x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij}) \quad (13) % MetaFind
```

EGNN's `C = 1/(M-1)` divides by the neighbour count; without it the coordinate
update grows with node degree. MetaFind drops it silently. Since ProcTHOR rooms
vary widely in object count, restoring `C` and not restoring it are measurably
different models. Restoring it is **not** licensed: an upstream implementation
detail cannot fill in what MetaFind chose to write differently. Ours follows
MetaFind and is labelled `[PAPER]`; if instability appears, adding `C` becomes
a recorded `[DEVIATION]`.

### B.2.3 `EG-5` + `EG-6` → `MF-14` — **MODIFIED** (residual added, signature cut)

```latex
m_i = \sum_{j \neq i} m_{ij} \quad (5)          % EGNN
h_i^{l+1} = \phi_h(h_i^l, m_i) \quad (6)        % EGNN
h_i^{l+1} = h_i^l + \sum_{j \neq i} \phi_h(m_{ij}) \quad (14)   % MetaFind
```

Two changes. MetaFind makes the update **residual**, and it moves `φ_h` inside
the sum with `h_i^l` no longer among its arguments. EGNN aggregates first and
transforms once; MetaFind transforms each message and aggregates after. These
are different functions, not a rearrangement, and `φ_h`'s input width differs
accordingly.

### B.2.4 `EG-1` → `MF-4`, `MF-9`, `MF-15` — **CONCEPTUAL_SOURCE**

```latex
\phi(T_g(x)) = S_g(\phi(x)) \quad (1)
```

The general definition of equivariance. MetaFind instantiates it for SE(3) and
adds the edge set as an explicit argument (`ESSGNN(R x^l + T, h^l, E)` versus
`EGCL(Qx^l + g, h^l)`), which is the notational marker of the `a_ij → e_ij`
substitution. `EG-14@3.1` and `EG-15@AppendixA` are the same statement in
EGNN's own notation.

### B.2.5 `EG-11` → `MF-11` — **CONCEPTUAL_SOURCE** (proof step, reproduced)

```latex
\|(Qx_i^l + g) - (Qx_j^l + g)\|^2 = \|Q(x_i^l - x_j^l)\|^2 = (x_i^l - x_j^l)^T Q^T Q (x_i^l - x_j^l) = \|x_i^l - x_j^l\|^2 \quad (11)   % EGNN
\|Q x_i^l + g - (Q x_j^l + g)\|^2 = \|Q(x_i^l - x_j^l)\|^2 = \|x_i^l - x_j^l\|^2 \quad (11)                                              % MetaFind
```

MetaFind reproduces EGNN's step, dropping the `Q^T Q = I` middle term. Same
mathematics. Nothing to implement — it is a proof obligation, and what it
obliges us to is that the distance fed to `φ_e` is a genuine pairwise distance
computed from live coordinates, never a cached or precomputed scalar.

### B.2.6 The NOT_USED set — 33 of EGNN's 44

Appearing in the EGNN paper is not evidence MetaFind uses it. In particular:

| EGNN formulas | why not used |
|---|---|
| `EG-7a`, `EG-7b` | velocity-type representations. MetaFind's scene graphs have no velocity channel; objects are static. |
| `EG-8` | edge **inference** — a learned `e_ij ∈ {0,1}` gate. MetaFind's `e_ij` is given by an LLM, not inferred. Identical symbol, unrelated construct. |
| `EG-9` | graph autoencoder head |
| `EG-12@B.1` | proof step for the velocity variant |
| `EG-13`, `EG-14@E.2`, `EG-15@E.2` | Appendix E distance-norm uniqueness proofs |
| remainder | QM9/N-body task heads, MLP implementation details, generic equivariance examples, Appendix A proof steps |

`EG-8` is the trap worth naming twice. It defines `m_i = \sum_{j \neq i} e_{ij} m_{ij}`
— `e_ij` as a **scalar gate**. MetaFind's `e_ij` is a sentence **vector** inside
`φ_e`. Same two characters, incompatible types. Porting EGNN's edge-inference
code because the symbol matched would typecheck in neither direction.

---

## B.3 Where the two upstreams are wired together

Only Eq. (6), and only additively. Consequences that follow directly:

1. `e_layout` must be **the same width** as the fused ULIP-2 embedding, since it
   is added, not concatenated. Nothing in either upstream fixes that width;
   `Pooling` is unspecified ([C8](#c8)).
2. `λ` is **learnable** — the paper says so explicitly ("a learnable scalar
   controlling the contribution of layout information"). It is a plain scalar
   coefficient. It is **not** described as a log, an exponent, or a gate, so a
   parameter named `log_lambda` and used linearly is a bug in either direction.
   Ours is `layout_weight`, used directly, initialised to 1.0.
3. The "residual design allows layout reasoning to enhance retrieval without
   disrupting the original embedding space" claim only holds if the ULIP-2 space
   is in fact preserved — i.e. if the backbone stays frozen. This is independent
   support for the U-34 resolution in B.1.2.

---

## B.4 Summary table

| upstream | MetaFind | relationship | the delta in one line |
|---|---|---|---|
| `U2-1` | `MF-5` | CONCEPTUAL_SOURCE | bidirectional → unidirectional; same-object → cross-tower |
| `U2-2` | `MF-5` | CONCEPTUAL_SOURCE | as above |
| `U2-3` | Stage 1 prose | CONCEPTUAL_SOURCE | `min` over `E_P` only → "both towers" |
| `EG-1` | `MF-4`/`MF-9`/`MF-15` | CONCEPTUAL_SOURCE | instantiated at SE(3); `E` added as an argument |
| `EG-3` | `MF-10` | MODIFIED | `a_ij` (discrete) → `e_ij` (sentence embedding) |
| `EG-4` | `MF-13` | MODIFIED | normalising constant `C` dropped |
| `EG-5`+`EG-6` | `MF-14` | MODIFIED | residual added; `φ_h` moved inside the sum |
| `EG-11` | `MF-11` | CONCEPTUAL_SOURCE | proof step reproduced, `Q^T Q` term elided |
| `EG-14@3.1` | `MF-4` | CONCEPTUAL_SOURCE | notational restatement |
| `EG-15@AppendixA` | `MF-15` | CONCEPTUAL_SOURCE | notational restatement |
| 33 others | — | NOT_USED | see B.2.6 |

Note what is **absent** from this table: MetaFind §2.5's `MF-2` and `MF-3` have
no upstream row. They are not a modification of any EGNN equation — they are a
second, different ESSGNN, and reconciling them with Appendix C is [C1](#c1).

---

Contradiction ids `C1`–`C9` are defined in `C_PAPER_CONTRADICTIONS.md`.
