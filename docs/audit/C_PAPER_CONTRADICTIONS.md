# C. PAPER_CONTRADICTIONS

Contradictions and silences in MetaFind, **re-verified against the authors' arXiv
TeX source**. Every entry here was checked a second time after the earlier audit
was found to rest on a lossy PDF→Markdown conversion.

**Result of that re-check: every contradiction survived.** None was a conversion
artifact. The paper really does say these things.

Two classes, kept apart because they license different actions:

- **C1–C8 — contradictions.** Two passages conflict. Choosing one is a decision
  about which of the paper's own claims to honour, recorded with its cost.
- **S1–S6 — silences.** The paper is mute. Filling one in is an
  `[IMPLEMENTATION CHOICE]` and can never be labelled `[PAPER]`.

Sources: `docs/paper/metafind_source/{2methdology,appendix,neurips_2025}.tex`.
Formula ids are A's.

---

## Contradictions

<a id="c1"></a>
### C1 — §2.5 and Appendix A describe two different ESSGNNs — **STRUCTURAL, blocking**

§2.5 (`MF-2`, `MF-3`) — **two** independent MLPs, no message variable:

```latex
h_i^{l+1} &= h_i^l + \sum_{j \in \mathcal{N}(i)} f_h(d_{ij}^l, h_i^l, h_j^l, e_{ij}; \theta_h), \\
    x_i^{l+1} &= x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x),
```

Appendix (`MF-10`, `MF-13`, `MF-14`) — **three** functions around one shared
message:

```latex
\rm m_{ij} = \phi_e\left(\rm h_i^l, \rm h_j^l, \| \rm x_i^l - \rm x_j^l \|^2, e_{ij} \right)
\rm x_i^{l+1} = \rm x_i^l + \sum_{j \ne i} (\rm x_i^l - \rm x_j^l) \cdot \phi_x(\rm m_{ij})
\rm h_i^{l+1} = \rm h_i^l + \sum_{j \ne i} \phi_h(\rm m_{ij})
```

Not two notations for one layer. In §2.5 the branches share no computation and
`f_h`, `f_x` each see the raw `(h_i, h_j, e_ij)` tuple. In the appendix a single
`φ_e` produces `m_ij` and `φ_x`, `φ_h` see **only** that message. Different
parameter counts, different gradient paths, different functions.

The appendix does flag itself as an adaptation — "The position update in ESSGNN
(**adapted from EGNN**)" — but it never says the method section's `f_h`/`f_x`
form is being replaced, and it proves equivariance for the appendix form only.

C2–C6 are separate consequences of this split, listed individually because each
can be got wrong on its own after picking a side.

**Registered as U-26. The only genuinely blocking contradiction.**

<a id="c2"></a>
### C2 — `h^{(0)}` contains `x`, and the proof assumes it does not — **SEVERE**

`2methdology.tex` (`MF-U1`):

```latex
h_i^{(0)} = \text{Concat}(x_i, t_i).
```

`appendix.tex`, same paper:

> "We begin by **assuming that $\rm h^0$ is invariant to SE(3) transformations
> on $\rm x$**, and that semantic edge embeddings $e_{ij}$ are derived solely
> from object-level textual descriptions and thus independent of spatial
> coordinates."

`Concat(x_i, t_i)` is a direct function of `x_i`. Under `x ↦ Qx + g` it becomes
`Concat(Qx_i + g, t_i) ≠ h_i^{(0)}`. **The proof's stated premise is violated by
the method section's own initialisation.**

Not confined to the appendix. `e_layout = Pooling({h_i^(L)})` is the entire
output of ESSGNN, and §2.5's motivation is that GATs are "highly sensitive to
global translation and scaling variations across scenes". If `h` carries raw
world coordinates, `e_layout` inherits exactly the sensitivity ESSGNN exists to
remove.

| reading | consequence |
|---|---|
| `h^(0) = t_i`, `x_i` in the separate coordinate channel (as EGNN does) | proof holds; motivation holds; **contradicts the printed formula** |
| `h^(0) = Concat(x_i, t_i)` as printed | matches the formula; **the appendix proof is false and the stated motivation fails** |

The second is unusable — it makes the paper's own theorem wrong. Adopting the
first chooses the paper's theorem over the paper's formula, and must say so.
`[PAPER CONTRADICTION]`, resolved toward the theorem.

<a id="c3"></a>
### C3 — `f_x` outputs `\mathbb{R}^3`, but the proof needs a scalar — **SEVERE**

`2methdology.tex`, in prose:

> `\( f_h: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^d \)`,
> `\( f_x: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^{3} \)`

The appendix's own proof step (`MF-U15`) factors `Q` out of the sum:

```latex
Q\rm x_i^l + g + \sum_{j \ne i} \left( Q\rm x_i^l + g - Q\rm x_j^l - g \right) \cdot \phi_x(\rm m_{ij}) &= Q\rm x_i^l + g + Q \sum_{j \ne i} (\rm x_i^l - \rm x_j^l) \cdot \phi_x(\rm m_{ij})
```

That factorisation is valid **only if `φ_x(m_ij)` is a scalar.** With `f_x`
returning `\mathbb{R}^3` and `·` elementwise, `Q(x_i - x_j) ⊙ f_x ≠ Q((x_i - x_j) ⊙ f_x)`
for general `Q`, and Eq. (4)'s equivariance claim fails. If `·` were a dot
product the coordinate update would have the wrong shape.

**Upstream agrees**, and says it in prose (`egnn_source/sections/model.tex`):

> "the function $\phi_x: \mathbb{R}^{\text{nf}} \rightarrow \mathbb{R}^1$ …
> **outputs a scalar value**"

Most likely a typo for `\mathbb{R}^1`, but "most likely a typo" is an inference
and is recorded as one. Adopt `\mathbb{R}^1`; `[PAPER CONTRADICTION]`.

**Verification obligation:** an equivariance test at `L ≥ 2` (see [S6](#s6)) is
the only thing that distinguishes the readings at runtime.

<a id="c4"></a>
### C4 — `h`'s width is self-inconsistent — **SEVERE**

Three statements, pairwise incompatible:

1. `h_i^{(0)} = \text{Concat}(x_i, t_i)` with `x_i ∈ \mathbb{R}^3`, `t_i ∈ \mathbb{R}^d` ⇒ `h^{(0)} ∈ \mathbb{R}^{d+3}`
2. `f_h: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^d` — input counts `h_i`, `h_j` as `d` each; output is `d`
3. `h_i^{l+1} = h_i^l + \sum f_h(...)` — a residual sum needs output width == `h` width

(1) says `h` is `d+3`. (2) says `f_h` reads `2d` and writes `d`. (3) then adds a
`d`-vector to a `(d+3)`-vector. **The three cannot all hold.**

Dropping `x_i` from `h^(0)` — the resolution already forced by C2 — makes all
three consistent at once. Independent corroboration from pure dimensional
analysis, and the strongest evidence that `Concat(x_i, t_i)` is the erroneous
line.

<a id="c5"></a>
### C5 — `j \in \mathcal{N}(i)` versus `j \ne i` — **MODERATE**

§2.5 sums over `\sum_{j \in \mathcal{N}(i)}` — the scene graph's actual edges.
The appendix sums over `\sum_{j \ne i}` — all other nodes, a complete graph.

Not cosmetic. §2.5 sets up a scene graph `G = (\mathcal{V}, \mathcal{E})` whose
edges are built from "physical-relation edges" and LLM "semantic-relation
edges". Under `j ≠ i` that structure is discarded and every object talks to
every other, making a substantial part of the paper's pipeline vestigial. Worse,
`e_ij` is only defined for pairs that *have* an edge; under `j ≠ i` most pairs
have no `e_ij` and the paper offers no fill value.

Resolve toward `\mathcal{N}(i)` — the reading under which the rest of the paper
is not vestigial. `[PAPER CONTRADICTION]`.

*(Upstream note, not evidence: EGNN's own §2.2 recap uses `\sum_{j \in \mathcal{N}(i)}`
for generic GNNs and `\sum_{j \neq i}` for its own EGCL, so MetaFind's two forms
each track a different part of EGNN's text.)*

<a id="c6"></a>
### C6 — distance versus squared distance — **MINOR but not free**

§2.5: `\( d_{ij}^l = \|x_i^l - x_j^l\|_2 \)` — a Euclidean **distance**.
Appendix `MF-10`: `\| \rm x_i^l - \rm x_j^l \|^2` — a **squared** distance.

Both SE(3)-invariant, so neither breaks the proof; an MLP can absorb the
difference in principle. Still a different input distribution — squared
distances in a 10 m room span two orders of magnitude more range — hence a
different optimisation problem. Pick one, state it, keep it fixed.

<a id="c7"></a>
### C7 — Stage 1 unidirectional, Stage 2 bidirectional, uncommented — **MODERATE**

`MF-5` has a single `-\log` with one normalisation over `A' \in \mathcal{B}`.
Eq. (7)/(8) are explicitly symmetric:

> "We adopt a **bidirectional contrastive learning** objective to symmetrically
> align query and gallery embeddings"

ULIP-2's Eq. (1)/(2) are also bidirectional (B.1.1). So Stage 1 as written is
weaker than both its backbone and its own second stage, and the paper never
remarks on it.

Unlike C1–C5, both readings are runnable. **Follow the paper literally** —
unidirectional Stage 1 — because a "fix" here silently changes the reported
experiment. A symmetric Stage 1 would be a `[DEVIATION]` with its own row.

<a id="c8"></a>
### C8 — SE(3) is claimed as the answer to *scaling* sensitivity — **MODERATE, already registered as RA-4**

`2methdology.tex`, motivating ESSGNN:

> "we observed that GATs were highly sensitive to global translation **and
> scaling** variations across scenes, resulting in unstable layout embeddings
> and poor generalization"

And `neurips_2025.tex`:

> "with built-in SE(3) equivariance to prevent degradation under arbitrary scene
> rotations or global shifts in coordinate systems"

**SE(3) is rotation and translation only. It does not contain scaling.** The
appendix proof takes `Q` orthogonal and `g` a translation; a uniform scaling
`x ↦ sx` changes `\|x_i - x_j\|^2` by `s^2` and the messages change with it.

So the stated motivation names a failure mode the method provably does not
address. The contributions list's narrower claim (rotations, global shifts) is
correct; only the §2.5/§3.4 motivation over-reaches.

**This was already found, before this audit.** `docs/graph/README.md` registers
it as **RA-4**, from a line-by-line reread of the same passage. The TeX confirms
the wording and adds nothing. It is listed here so C is complete, not because it
is new — and the registry's disposition is the better one and is adopted
unchanged:

> "沒有保證" is not "cannot work". An MLP may still learn behaviour insensitive
> to scale within its training range. **RA-4 measures how far `e_layout`
> actually moves; it does not predict.**

An earlier draft of this document said no evaluation of ours may report scale
robustness. That was stricter than the evidence supports: what is forbidden is
attributing scale robustness to the SE(3) proof, not measuring whether it
happens to hold.

---

## Silences

<a id="s1"></a>
### S1 — `\text{sim}` is never defined

Used in Eq. (1), (5), (7) and never given; the only gloss is "$\text{sim}(\cdot, \cdot)$
denotes the similarity function". Cosine and dot product are not
interchangeable: with unnormalised embeddings `τ` has no fixed scale. ULIP-2
uses a dot product over its `\mathbf{f}` features. `[IMPLEMENTATION CHOICE]` —
cosine (L2-normalise, then dot).

<a id="s2"></a>
### S2 — `\text{Pooling}` is unspecified

```latex
e_{\text{layout}} = \text{Pooling}(\{h_i^{(L)}\}).
```

Mean, max, sum and attention all satisfy the sentence, and they differ in
whether `e_layout` scales with room size — which entangles the choice with the
learnable `λ`. `[IMPLEMENTATION CHOICE]`.

<a id="s3"></a>
### S3 — `\text{Fusion}` is a list, not a specification

§2.4 offers "mean pooling, MLP, masked MLP, gated fusion, or Transformer-based
fusion" and never picks. Table 1's numbers depend on the pick, so no
reproduction of Table 1 can be attributed to the paper.
`[IMPLEMENTATION CHOICE]`.

<a id="s4"></a>
### S4 — no training hyperparameters exist in the paper

Searched across all five included `.tex` files and absent: layer count `L`,
hidden width `d`, edge width `e`, batch size, learning rate, optimiser, epochs,
`τ`'s value or whether it is learnable, `λ`'s initialisation, and the OpenCLIP
variant. The only stated numbers are the two 30% rates (modality masking in
Stage 1, scene dropout in Stage 2).

Every one is `[IMPLEMENTATION CHOICE]`. **No quantitative result of this
reproduction may be presented as reproducing the paper's numbers.**

<a id="s5"></a>
### S5 — two edge types are described, one enters the math — **registered as U-29**

§2.4 describes the scene graphs as having

> "(i) physical-relation edges that capture spatial dependencies (e.g., "cup on
> table"); and (ii) semantic-relation edges that capture functional or
> contextual associations (e.g., "microscope–lab bench"), obtained by prompting
> an LLM on object pairs."

But `MF-2`, `MF-3` and `MF-10` carry exactly one edge term, `e_ij`, and both
§2.5 and the appendix describe it as the **semantic** embedding from the frozen
text encoder. **The paper never says how physical-relation edges are numerically
encoded** — whether they contribute their own `e_ij`, share the channel, are
concatenated, or only determine `\mathcal{N}(i)`.

`UNKNOWN`, and **already registered as U-29** in `02_BUILD_STEPS.md` — again
found before this audit, not by it. `essgnn_edge_protocol` records
`physical_relation_encoding = "neighbourhood_only"`: physical edges define
`\mathcal{N}(i)`, semantic edges supply `e_ij`. That is the reading requiring no
invention, but the paper does not say so and it must not be written as though it
did.

<a id="s6"></a>
### S6 — `t_i` has no stated encoder, and the coordinate output is never consumed

Two silences with one shared consequence.

**`t_i`.** §2.5 introduces "a text-derived feature $t_i \in \mathbb{R}^d$" and
never says what produces it. The frozen text encoder is named **only** for
`e_ij`, in the appendix. Assuming the same encoder for `t_i` would be inventing
a paper fact. `UNKNOWN` (U-20); its width `d` is unstated too (S4).

**The coordinate channel.** `e_layout` is built from `h^{(L)}` alone; `x^{(L)}`
is never read. One might think a broken coordinate update is harmless. It is
not, for `L ≥ 2`: `MF-2` feeds `d_{ij}^l` into the feature update, and
`d_{ij}^l` comes from `x^l`, which `MF-3` produced at the previous layer. A
non-equivariant coordinate update contaminates `h` from layer 2 onward. **At
`L = 1` the entire class of bugs is invisible, so any equivariance test must use
`L ≥ 2`.**

---

## Disposition

| id | severity | blocks | resolution |
|---|---|---|---|
| C1 | structural | **yes** — U-26 | open |
| C2 | severe | no | toward the theorem: `h^(0) = t_i` |
| C3 | severe | no | `f_x → \mathbb{R}^1`; guarded by an `L ≥ 2` equivariance test |
| C4 | severe | no | subsumed by C2's resolution |
| C5 | moderate | no | `\mathcal{N}(i)` |
| C6 | minor | no | pick one, fix it, state it |
| C7 | moderate | no | follow the paper literally |
| C8 | moderate | no | never report scale robustness |
| S1–S6 | — | no | `[IMPLEMENTATION CHOICE]` or `UNKNOWN`, each recorded |

C2, C3 and C4 point the same way from three independent directions — the
appendix's stated premise, the equivariance algebra, and dimensional analysis.
That convergence is why the resolution is defensible.

**C1 remains open.** Until it is decided, ESSGNN has no single specification and
no claim of a fully reproduced MetaFind is admissible.
