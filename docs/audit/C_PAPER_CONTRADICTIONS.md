# C. PAPER_CONTRADICTIONS

Places where MetaFind states two things that cannot both hold, and places where
it states nothing that must be stated for the model to exist.

Two classes, kept apart because they license different actions:

- **C1–C7 — contradictions.** Two passages conflict. Choosing one is *not* an
  implementation choice; it is a decision about which of the paper's own claims
  to honour, and it must be recorded with its cost.
- **S1–S5 — silences.** The paper is simply mute. Filling one in is an
  `[IMPLEMENTATION CHOICE]` and can never be labelled `[PAPER]`.

Every quotation below is from `docs/audit/repaired/metafind_paper.md`; formula
ids are A's.

---

## Contradictions

<a id="c1"></a>
### C1 — §2.5 and Appendix C describe two different ESSGNNs — **STRUCTURAL, blocking**

§2.5 (`MF-2`, `MF-3`) — **two** independent MLPs, no message variable:

```latex
h_i^{l+1} = h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h) \quad (2)
x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x) \quad (3)
```

Appendix C (`MF-10`, `MF-13`, `MF-14`) — **three** functions around one shared
message:

```latex
m_{ij} = \phi_e \left( h_i^l, h_j^l, \|x_i^l - x_j^l\|^2, e_{ij} \right) \quad (10)
x_i^{l+1} = x_i^l + \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij}) \quad (13)
h_i^{l+1} = h_i^l + \sum_{j \neq i} \phi_h(m_{ij}) \quad (14)
```

These are not two notations for one layer. In §2.5 the two branches share no
computation and `f_h`, `f_x` each see the raw `(h_i, h_j, e_ij)` tuple. In
Appendix C a single `φ_e` produces `m_ij`, and `φ_x`, `φ_h` see **only** that
message. Different parameter counts, different gradient paths, different
functions.

C2–C6 below are each a separate consequence of this split; they are listed
individually because each one can be got wrong on its own even after picking a
side.

**Registered as U-26. This is the only genuinely blocking contradiction.**
Everything downstream of `e_layout` depends on which layer is built.

<a id="c2"></a>
### C2 — `h^{(0)}` contains `x`, and Appendix C's proof assumes it does not — **SEVERE**

§2.5, unnumbered but load-bearing (`MF-U1`):

```latex
h_i^{(0)} = \operatorname{Concat}(x_i, t_i)
```

Appendix C, in the same paper:

> **Assuming that $h^0$ is invariant to $SE(3)$ transformations on $x$**, and
> that semantic edge embeddings $e_{ij}$ are derived solely from object-level
> textual descriptions and thus independent of spatial coordinates, the pairwise
> edge message is: …

`Concat(x_i, t_i)` is a direct function of `x_i`. Under `x ↦ Qx + g` it becomes
`Concat(Qx_i + g, t_i) ≠ h_i^{(0)}`. **The proof's stated premise is violated by
the paper's own initialisation, three pages earlier.**

The damage is not confined to the appendix. `e_layout = Pooling({h_i^(L)})` is
the entire output of ESSGNN, and the paper's motivation for ESSGNN is that GATs
are "highly sensitive to global translation and scaling". If `h` carries raw
world coordinates, `e_layout` inherits exactly the sensitivity ESSGNN exists to
remove — a ProcTHOR room shifted 10 m gets a different layout embedding, which
is the failure mode named in §2.5's own opening.

Two readings, and they are not equally supported:

| reading | consequence |
|---|---|
| `h^(0) = t_i` only, `x_i` held in the separate coordinate channel (as EGNN does) | the proof holds; the ESSGNN motivation holds; **contradicts the printed formula** |
| `h^(0) = Concat(x_i, t_i)` as printed | matches the formula; **the appendix proof is false and the stated motivation fails** |

Neither can be labelled `[PAPER]` unqualified. The second is unusable — it
makes the paper's own theorem wrong — so a reproduction that adopts the first
is choosing between the paper's formula and the paper's theorem, and must say
which. Record as `[PAPER CONTRADICTION]`, resolved toward the theorem, with the
printed formula noted as the discarded side.

<a id="c3"></a>
### C3 — `f_x` outputs `\mathbb{R}^3`, but the proof needs a scalar — **SEVERE**

§2.5, in prose:

> $f_h : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^d$, $f_x : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^3$

EGNN's corresponding `φ_x` outputs `\mathbb{R}^1`. The difference decides
whether the model is equivariant at all.

Appendix C's proof step (`MF-U16`) pulls `Q` out of the sum:

```latex
Q x_i^l + g + \sum_{j \neq i} (Q x_i^l + g - Q x_j^l - g) \cdot \phi_x(m_{ij}) = Q x_i^l + g + Q \sum_{j \neq i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})
```

That factorisation is valid **only if `φ_x(m_ij)` is a scalar.** If `f_x` returns
a vector in `\mathbb{R}^3` and `·` is the elementwise product, then
`Q(x_i - x_j) ⊙ f_x ≠ Q((x_i - x_j) ⊙ f_x)` for general `Q`, and rotation
equivariance is lost — Eq. (4)'s claim fails. If `·` is instead a dot product,
the result is a scalar and the coordinate update has the wrong shape.

So the `\mathbb{R}^3` annotation is inconsistent with Eq. (4), with Eq. (13),
and with EGNN. It is most likely a typo for `\mathbb{R}^1`, but "most likely a
typo" is an inference, and it must be recorded as one. Adopt `\mathbb{R}^1`;
label `[PAPER CONTRADICTION]`.

**Verification obligation:** an equivariance unit test — feed a scene, feed the
same scene under a random `(Q, g)`, assert `h^(L)` is unchanged and `x^(L)`
transforms — is the only thing that distinguishes the two readings at runtime.
Numbers passing is not the same as equivariance holding.

<a id="c4"></a>
### C4 — `h`'s width is self-inconsistent — **SEVERE**

Three statements, pairwise incompatible:

1. `h_i^{(0)} = \operatorname{Concat}(x_i, t_i)` with `x_i ∈ \mathbb{R}^3`, `t_i ∈ \mathbb{R}^d` ⇒ `h^{(0)} ∈ \mathbb{R}^{d+3}`
2. `f_h : \mathbb{R}^{(2d+1+e)} \to \mathbb{R}^d` — its input width `2d` counts `h_i` and `h_j` as `d` each, and its output is `d`
3. `h_i^{l+1} = h_i^l + \sum f_h(...)` — a residual sum requires output width == `h` width

(1) says `h` is `d+3` wide. (2) says `f_h` reads `2d` of `h` and writes `d`.
(3) then adds a `d`-vector to a `(d+3)`-vector. **The three cannot all hold.**

Dropping the `x_i` from `h^(0)` — i.e. the resolution already forced by C2 —
makes all three consistent at once. That is independent corroboration for C2's
resolution, from pure dimensional analysis, and it is the strongest evidence in
this document that `Concat(x_i, t_i)` is the erroneous line.

<a id="c5"></a>
### C5 — `j ∈ \mathcal{N}(i)` versus `j \neq i` — **MODERATE**

§2.5 sums over `\sum_{j \in \mathcal{N}(i)}` — the graph neighbourhood, i.e. the
scene graph's actual edges. Appendix C sums over `\sum_{j \neq i}` — **all**
other nodes, a complete graph.

This is not cosmetic. MetaFind's whole §2.5 setup is a scene graph `G = (V, E)`
with spatial edges from "physical layout constraints (e.g., adjacency, support)"
and LLM semantic edges. Under `j ≠ i` that graph structure is discarded and
every object talks to every other, which would make the edge construction
pipeline — a substantial part of the paper — pointless. Worse, `e_ij` is only
defined for pairs that *have* an edge; under `j ≠ i` most pairs have no `e_ij`
at all, and the paper offers no fill value.

Resolve toward `\mathcal{N}(i)`: it is the reading under which the rest of the
paper is not vestigial. Label `[PAPER CONTRADICTION]`, and note that our
`L1-SEMEDGE-NO-ZEROFILL` rule exists precisely because the alternative reading
would have demanded a silent zero fill.

<a id="c6"></a>
### C6 — distance versus squared distance — **MINOR but not free**

§2.5: `d_{ij}^l = \|x_i^l - x_j^l\|_2` — a Euclidean **distance**.
Appendix C `MF-10`: `\|x_i^l - x_j^l\|^2` — a **squared** distance.

Both are SE(3)-invariant, so neither breaks the proof; the MLP can absorb the
difference in principle. It is still a different input distribution — squared
distances in a 10 m room span two orders of magnitude more range — and so a
different optimisation problem. Pick one, state it, keep it fixed.

<a id="c7"></a>
### C7 — Stage 1 is unidirectional, Stage 2 is bidirectional — **MODERATE, uncommented**

`MF-5` has a single `-\log` with one normalisation over `A' ∈ \mathcal{B}`.
`MF-7a`/`MF-7b`/`MF-8` are explicitly symmetric, and the paper says so:

> We adopt a **bidirectional** contrastive learning objective to symmetrically
> align query and gallery embeddings

ULIP-2's Eq. (1)/(2), which MetaFind's backbone is trained with, are also
bidirectional (see `B.1.1`). So Stage 1 as literally written is weaker than both
its own backbone and its own second stage, and the paper never remarks on it.

Unlike C1–C5, both readings are *runnable*. Follow the paper literally —
unidirectional Stage 1 — because a "fix" here silently changes the reported
experiment. If a symmetric Stage 1 is later wanted, it is a `[DEVIATION]` with
its own row, not a correction.

---

## Silences

<a id="s1"></a>
### S1 — `\operatorname{sim}` is never defined

Introduced in Eq. (1) as "$\operatorname{sim}(\cdot, \cdot)$ denotes the
similarity function" and used in Eqs. (1), (5), (7a), (7b) without ever being
given. Cosine and dot product are not interchangeable here: with unnormalised
embeddings the temperature `τ` has no fixed scale, and Eq. (5) becomes
ill-conditioned. ULIP-2 uses a dot product over normalised features.
`[IMPLEMENTATION CHOICE]` — cosine, i.e. L2-normalise then dot.

<a id="s2"></a>
### S2 — `\operatorname{Pooling}` is unspecified

```latex
e_{\text{layout}} = \operatorname{Pooling}(\{h_i^{(L)}\})
```

Mean, max, sum and attention pooling all satisfy the sentence. They differ in
whether `e_layout` scales with room size — sum does, mean does not — and
`e_layout` is added to the fused embedding with a learnable `λ`, so the pooling
choice and `λ`'s learned value are entangled. `[IMPLEMENTATION CHOICE]`.

<a id="s3"></a>
### S3 — `\operatorname{Fusion}` is a list, not a specification

§2.4 offers "mean pooling, MLP, masked MLP, gated fusion, or Transformer-based
fusion" and never picks. Since Table 1's numbers depend on the pick, no
reproduction of Table 1 can be attributed to the paper.
`[IMPLEMENTATION CHOICE]`.

<a id="s4"></a>
### S4 — no training hyperparameters exist in the paper

Searched and absent: layer count `L`, hidden width `d`, edge-embedding width
`e`, batch size, learning rate, optimiser, epochs, `τ`'s value or whether it is
learnable, `λ`'s initialisation, and the OpenCLIP variant. The only stated
numbers are the two 30% dropout rates (modality masking in Stage 1, scene
dropout in Stage 2).

Every one of these is `[IMPLEMENTATION CHOICE]`. **No quantitative result of
this reproduction may be presented as reproducing the paper's numbers**, because
the paper does not contain the configuration that produced them.

<a id="s5"></a>
### S5 — the coordinate output is computed and never consumed

Not a contradiction, but the reason C3 matters more than it first appears.
`e_layout` is built from `h^(L)` alone; `x^(L)` is never read. So one might
think a broken coordinate update is harmless.

It is not, for `L ≥ 2`: `MF-2` feeds `d_{ij}^l` into the feature update, and
`d_{ij}^l` is computed from `x^l`, which `MF-3` produced at the previous layer.
A non-equivariant coordinate update therefore contaminates `h` from layer 2
onward. With `L = 1` the bug is invisible. **Any equivariance test must use
`L ≥ 2`,** or it will pass on a model that is not equivariant.

---

## Disposition

| id | severity | blocks | resolution |
|---|---|---|---|
| C1 | structural | **yes** — U-26 | open |
| C2 | severe | no | toward the theorem: `h^(0) = t_i` |
| C3 | severe | no | `f_x → \mathbb{R}^1`; guard with an equivariance test |
| C4 | severe | no | subsumed by C2's resolution |
| C5 | moderate | no | `\mathcal{N}(i)` |
| C6 | minor | no | pick one, fix it, state it |
| C7 | moderate | no | follow the paper literally |
| S1–S5 | — | no | `[IMPLEMENTATION CHOICE]`, each recorded |

C2, C3 and C4 all point the same way, from three independent directions — the
appendix's stated premise, the equivariance algebra, and dimensional analysis.
That convergence is why the resolution is defensible; a single argument would
not have been.

**C1 remains open.** Until it is decided, ESSGNN has no single specification and
no claim of a fully reproduced MetaFind is admissible.
