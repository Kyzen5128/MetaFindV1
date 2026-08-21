# D0 Decision — MetaFind §2.5 `f_x → R³`: how to reproduce it faithfully

> **Sections 1–5 were prepared by Master** as framing and evidence pointers. They are **not** a completed evidence survey, and Master deliberately did **not** perform the paper audit.
> **Sections 6–11 are D0's work and are intentionally empty.**
> **Sections 12–14** are Master's / the user's, per `WORKFLOW.md` §13B.

---

## Decision ID

`D0-009_essgnn-fx-codomain`

---

## Status

`INVESTIGATION COMPLETE` — 2026-08-21. Sections §6–§11 filled by the D0-009 conversation; Codex adversarial review completed (round 1 of 2) and its BLOCKER accepted, changing the verdict.

**Verdict: `PAPER-AMBIGUOUS`. NOT ACCEPTED.** Awaiting Master integration review (§12) and the `WORKFLOW.md` §13B user gate (§13, §14). D0 has not marked its own recommendation accepted.

**`PARALLEL SAFE: YES` with `D2a_stage1-protocol-refresh`** — verified by Master against `WORKFLOW.md` §7. D0-009 writes **only this file**; it touches no code, test, protocol, or graph contract. `D2a` touches nothing under `workflow/decisions/`.

**Scope wall.** D0-009 must not touch anything in `D2a`'s scope — `resolve_stage1.py`, `annotate_run.py`, `annotate.py`, the protocol artifacts, or the Stage 1 encoding contract. If this investigation finds something bearing on `D2a`, report a `MASTER-IMPACTING FINDING`. Do not act on it.

---

## Governing principle for this decision

Stated by the user, 2026-08-21. **Binding on the whole investigation.**

> The goal is **not** to hunt for errors in the paper. It is to establish faithfully what MetaFind actually requires.

| | |
|---|---|
| 1 | If MetaFind itself gives a definition that is **explicit, executable, and internally consistent**, MetaFind governs. Use it |
| 2 | **Do not** change a clear statement in MetaFind's main text because EGNN, an upstream work, or the Appendix looks more sensible |
| 3 | Escalate to the user **only** when MetaFind is internally contradictory, admits multiple irreducible readings, is dimensionally impossible, or cannot hold together with its own claims |
| 4 | **Upstream is supplementary evidence only. It may not override MetaFind's main text** |

---

## 1. Question

**How is MetaFind §2.5's `f_x: R^(2d+1+e) → R³` to be reproduced faithfully?**

```
FINDING UNDER EXAMINATION:  What MetaFind itself specifies for f_x's codomain and
                            for the operator `·` in the coordinate update.
DECISION REQUIRED:          What this reproduction implements, and under what
                            evidence class.
```

MetaFind §2.5 types the function as `→ R³` and writes the coordinate update as:

```
x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i^l − x_j^l) · f_x(d_ij^l, h_i^{l+1}, h_j^{l+1}, e_ij)
```

**Already decided by the user; not in scope.** `h` follows §2.5's sequential per-layer update: `h^l → h^{l+1} → x^{l+1} uses h^{l+1}`. The `h^l` vs `h^{l+1}` question is **closed**. Do not reopen it.

---

## 2. Why This Decision Exists

The repository has already resolved this — and resolved it by a route the user's governing principle above puts in question.

The current implementation uses a **scalar** `f_x`, and the recorded justification appeals to (a) the Appendix's Eq. 13 proof and (b) the reference EGNN implementation. Under principle 4, upstream cannot settle a MetaFind main-text statement. Under principle 2, the Appendix being more sensible is not by itself a reason to override §2.5.

Whether the current implementation is nevertheless correct is exactly what this decision must establish **from MetaFind's own evidence** — not assume, and not overturn on reflex either.

**What is blocked until it resolves:** nothing is executing today. Stage 2 (`n13`) has never run and `D5_stage2-prereq` is not started, so this can be settled before any Stage 2 artifact exists. That is the cheap moment.

**What it affects if it changes:** `metafind/models/essgnn.py`, the equivariance tests, `docs/audit/` C3, `docs/graph/` U-26, `CONTEXT.md` §5, and every Stage 2 result.

---

## 3. Decision Scope

### In Scope

- What MetaFind's own text, appendix, figures, captions, and supplementary material define for `f_x`'s codomain.
- What MetaFind defines — or fails to define — for the operator `·` in the §2.5 coordinate update: dot product, Hadamard product, scalar multiplication, or something else.
- Whether MetaFind anywhere re-defines or corrects `→ R³`.
- Whether any MetaFind-internal evidence closes the dimensions.
- Dimensional executability of each candidate reading.
- Whether a dimensionally-closing reading is compatible with MetaFind's **own** SE(3) equivariance claim.
- A verdict in exactly one of three classes (§6 format).

### Explicit Non-Scope

- ❌ **Do not change `f_x` to `R¹` scalar on your own authority.**
- ❌ **Do not use "EGNN is scalar anyway" as the deciding reason.** Upstream is supplementary evidence only.
- ❌ **Do not modify code, protocol, tests, or artifacts.** This is an audit. Implementation follows only after the user rules.
- ❌ **Do not reopen `h^l` vs `h^{l+1}`.** The user decided: §2.5 sequential update.
- ❌ Do not decide `D0-004` (the `coord_feat` / `architecture_family` coupling), though you may note any interaction.
- ❌ Do not repair a contradiction by choosing the convenient side. If MetaFind contradicts itself, **record the contradiction** and escalate.

---

## 4. Authority / Evidence

> Master-prepared pointers. **Master did not perform this audit.** D0 must gather the evidence itself.

Authority order for this decision, per the governing principle:

1. **MetaFind main text** — `docs/paper/metafind_source/2methdology.tex`, `3experiments.tex`, `neurips_2025.tex`
2. **MetaFind appendix / supplementary** — `docs/paper/metafind_source/appendix.tex`
3. **MetaFind figures and captions** — `MetaFind.drawio.png`, `data-preprocess.png`, `scene1_*.png`, `scene2_*.png`
4. **Upstream, supplementary only** — `docs/paper/egnn_source/`, `docs/paper/ulip2_source/`

Nothing below level 3 may override levels 1–2.

**Required searches** (D0 must perform them; Master has not):

- every occurrence of `f_x`, `\phi_x`, `phi_x`, `R^3`, `\mathbb{R}^{3}`, `\cdot`, `\odot`, `\times` in all MetaFind `.tex` files;
- any definition of notation, an operator table, or a "notation" paragraph anywhere in the source;
- figure captions and any inline math in them;
- the appendix equivariance proof, read for what it **requires** of `f_x` rather than for what EGNN does.

### Known prior claims — treat as claims to be tested, not as evidence

`docs/audit/E_GRAPH_REVALIDATION.md:175` records C3 as `VERIFIED` with the reason `[UPSTREAM] settles it: EGNN's φ_x "outputs a scalar value"`. **Under principle 4 that reasoning is inadmissible as a settlement.** D0 must re-derive from MetaFind's own evidence and may reach the same or a different conclusion.

---

## 5. Current Repository State

Master-verified 2026-08-21, read-only. **This is repository state, not evidence of paper intent.**

**The implementation uses a scalar, hardcoded, with no configuration flag.**

| Location | Content |
|---|---|
| `metafind/models/essgnn.py:311-312` | `# f_x : R^(2d + 1 + e) -> R   (Eq. 3, corrected to scalar per Eq. 13)`<br>`self.f_x = _mlp(2 * h + 1 + edge_dim, h, 1)` |
| `metafind/models/essgnn.py:358-359` | `w = self.f_x(...)`<br>`trans = coord_diff * w  # scalar * vector -> stays equivariant` |
| `metafind/models/essgnn.py:39-44` | Records the position as: **"F10 has no flag: it is simply an error in the paper."** Cites Eq. 13's proof factoring the rotation out, and adds "The reference EGNN agrees (`coord_mlp` ends in `Linear(hidden, 1)`)" |

**How it is recorded elsewhere — the records do not agree with each other:**

| Document | Recorded as |
|---|---|
| `docs/audit/D_IMPLEMENTATION_FORMULA_CONTRACT.md:124` | `f_x` codomain: paper `R³`, implementation `R¹` → **`[PAPER CONTRADICTION]` — C3** |
| `docs/audit/E_GRAPH_REVALIDATION.md:175` | C3 **`VERIFIED`**, settled by `[UPSTREAM]` |
| `docs/audit/F_CODE_GRAPH_CONSISTENCY.md:27` | `CONSISTENT`, "no flag, audit-only" |
| `workflow/CONTEXT.md` §5 | **DEVIATION, recorded** — "2.5 literal text is wrong; scalar gives equivariance error 2.2e-16 vs 0.43" |

**Two things D0 must know about that last row.** The equivariance figures `2.2e-16 vs 0.43` were inherited from the previous workflow and are **UNVERIFIED** in this repository — `CONTEXT.md` §5 says so explicitly. And one document calls the same item a `PAPER CONTRADICTION` while another calls it `VERIFIED`; that disagreement is itself unresolved.

**Nothing is executing on this today.** Stage 2 (`n13`) has never run. `data/outputs/checkpoints/` is empty. `essgnn_arch_protocol.json` records `architecture_family: appendix_shared_msg`, `coord_feat: current` — a separate axis, governed by `D0-004`.

---

## 6. Options

*D0 fills this section.*

**Required report format — the user specified it exactly. Produce it verbatim:**

```text
ISSUE
MetaFind §2.5 f_x → R³

PAPER FACT
[quote the original text, item by item]

INTERNAL EVIDENCE
[whether any other MetaFind section / appendix explains it]

EXECUTABILITY CHECK
[for dot / Hadamard / other readings: does each close dimensionally?]

CONTRADICTION CHECK
[does it conflict with MetaFind's own SE(3) claim?]

VERDICT
PAPER-RESOLVED / PAPER-AMBIGUOUS / PAPER-CONTRADICTORY

USER DECISION REQUIRED
[if a ruling is still needed, list the genuine options. Do not choose for the user]
```

### Verdict classes — use exactly one

| Class | Meaning |
|---|---|
| **PAPER-RESOLVED** | MetaFind's own evidence suffices to define how `R³` operates |
| **PAPER-AMBIGUOUS** | MetaFind does not define it clearly, but several executable readings exist |
| **PAPER-CONTRADICTORY / BLOCKING** | The main text's `R³` cannot hold together with the equation's dimensions or with MetaFind's own equivariance claim |

### Specific checks the user requires

1. **If `·` is a standard dot product:** `R³ · R³ → R`, so adding a scalar to `x_i^l ∈ R³` does not close. **Record this explicitly** if it is what the text implies.
2. **If `·` is a Hadamard product:** dimensions close. Then separately check compatibility with MetaFind's own SE(3) equivariance claim. **If incompatible, do not silently switch to scalar — mark it a paper contradiction.**

---

### D0-009 REPORT — required format, produced verbatim

> **Revision 2, after Codex adversarial review.** Revision 1 returned `PAPER-CONTRADICTORY`.
> Codex finding `D0-009-A1` (BLOCKER) was **verified and CONFIRMED**: Revision 1's claim that an
> `R³` codomain is *unreachable* was wrong. The verdict below is the corrected one. Revision 1's
> reasoning and what broke it are recorded in §10 rather than deleted.

```text
ISSUE
MetaFind §2.5 f_x → R³

PAPER FACT
P1  2methdology.tex:52 — coordinate update, verbatim:
      x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \cdot f_x(d_{ij}^l, h_i^{l+1}, h_j^{l+1}, e_{ij}; \theta_x)

P2  2methdology.tex:54 — verbatim:
      "f_h: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^d,  f_x: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^{3}
       are two learnable functions parameterized by \theta_h and \theta_x, respectively,
       which we approximate using multilayer perceptrons (MLPs)."
      This is the ONLY occurrence of `f_x` outside the equation itself, in all 5 .tex files.
      MetaFind never says what the three output components MEAN.

P3  2methdology.tex:38 — "ESSGNN is designed to maintain equivariance to SE(3) transformations
      during message passing while incorporating semantic relationships between objects
      through learned edge representations."

P4  2methdology.tex:61-64 — "Our model retains full SE(3)-equivariance concerning input
      transformations. Specifically, for any rotation operator R \in SO(3) and translation
      vector T \in \mathbb{R}^3, the following condition holds:
        (R x^{l+1} + T, h^{l+1}) = ESSGNN(R x^l + T, h^l, E)"
      Note "any rotation operator R ∈ SO(3)" — the whole group, not a subgroup. Note also that
      the equation writes h^{l+1} UNCHANGED on both sides: MetaFind's MAIN TEXT, not only its
      appendix, asserts that h is invariant.

P5  2methdology.tex:44 — h_i^{(0)} = Concat(x_i, t_i), with x_i ∈ R³ and t_i ∈ R^d.
      This CONFLICTS with P4's invariance of h. See CONTRADICTION CHECK item 2 — it belongs
      to D0-004 / RA-1 and is not decided here, but it makes P4's premise conditional.

P6  2methdology.tex:47 — e_ij comes from LLM relation sentences encoded by a frozen text
      encoder. No coordinates enter. (MAIN TEXT source for e_ij's invariance.)

P7  appendix.tex:50 — position update restated in MetaFind's own proof:
      x_i^{l+1} = x_i^l + \sum_{j \ne i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})
      φ_x is NEVER given a codomain type anywhere in the appendix.

P8  appendix.tex:56-58 — MetaFind's own algebraic step, verbatim:
      Q x_i^l + g + \sum_{j\ne i} (Q x_i^l + g - Q x_j^l - g) \cdot \phi_x(m_{ij})
        = Q x_i^l + g + Q \sum_{j\ne i} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})
        = ... = Q x_i^{l+1} + g
      Q is pulled out THROUGH the `·` product.

P9  appendix.tex:20,29 — e_ij is "invariant to the input node positions x"; h^0 is ASSUMED
      invariant to SE(3) transformations on x. (An assumption, stated as such by MetaFind.)

P10 appendix.tex:38-44 — ||Qx_i + g - (Qx_j + g)||² = ||x_i - x_j||², "Hence the edge message
      is preserved: m_ij' = m_ij."

P11 PAPER FACT AS TO SILENCE — MetaFind never defines the operator `·`. Verified exhaustively
    over the whole archive:
      • \odot         : 0 occurrences in all 5 .tex files
      • \times        : 1 occurrence, only "Q \in \mathbb{R}^{3\times3}" (appendix.tex:23)
      • \cdot         : 9 TOKENS across 7 lines — full enumeration, corrected per Codex A7:
          2methdology.tex:10   sim(\cdot,\cdot)          2 tokens — argument placeholders,
                                                          not products
          2methdology.tex:52   the equation in P1        1 token
          2methdology.tex:85   + \lambda \cdot e_layout  1 token
          2methdology.tex:128  + \lambda \cdot e_layout  1 token  (Algorithm 1, line 5)
          appendix.tex:50      the equation in P7        1 token
          appendix.tex:56      the proof step in P8      2 tokens
          appendix.tex:57      the proof step in P8      1 token
        Verified by `grep -o '\cdot' *.tex | wc -l` = 9. Revision 1 said "6 occurrences",
        counting grouped syntactic sites; the token count is 9. The SILENCE conclusion is
        unaffected — none of the 9 is a definition.
      • no \section/\paragraph named notation; no operator table; no symbol list
      • no \newcommand / \renewcommand / \def / \DeclareMathOperator touching math operators
        in neurips_2025.tex or neurips_2025.sty (only font/section/float macros)
      • figures carry no inline math: MetaFind.drawio.png was opened and read directly
        (2026-08-21) — a block diagram; data-preprocess.png and the 4 scene*.png are
        pipeline/render images
      • SOURCE_MANIFEST.json: "supplement_files": [] — no supplementary material beyond
        appendix.tex. arXiv 2510.04057, archive sha256 61351d34…38b4e5, 5 .tex files,
        541 lines, all searched.
      f_x → R³ is never redefined, corrected, erratum'd, or restated anywhere.

INTERNAL EVIDENCE
E1  MetaFind's OWN usage of `\cdot` as a product, elsewhere in the same paper, is scalar ×
    vector. 2methdology.tex:85-87 writes "+ \lambda \cdot e_layout" and then states "where λ
    is a learnable scalar". That is the only other genuine product written with `\cdot` in
    MetaFind. House usage, not a definition.
    [INFERENCE from MetaFind-internal usage — suggestive, NOT decisive]

E2  MetaFind's appendix proof step (P8) pulls Q out through the `·`. This constrains `·` to
    satisfy  (Qv) ⋆ w = Q(v ⋆ w)  for all Q ∈ SO(3).
    It does NOT force w to be a scalar — see EXECUTABILITY CHECK Reading E.
    PROVENANCE CAVEAT (Codex A3/A5): that step is a near-verbatim adaptation of EGNN's proof
    (docs/paper/egnn_source/sections/appendix.tex:18-20), which MetaFind states openly
    ("the original proof structure holds. We now restate and extend the proof below",
    appendix.tex:20). MetaFind restating it makes it MetaFind's assertion; it does not make
    it independent evidence.
    [PAPER FACT that the step is asserted; INFERENCE as to what it constrains]

E3  f_x's arguments are SE(3)-invariant, so f_x's OUTPUT does not rotate.
    MAIN-TEXT sources (re-sourced per Codex A3, which correctly showed Revision 1's
    "appendix-free" claim was false because it still cited the appendix):
      • d_ij  — a Euclidean distance, invariant by construction (2methdology.tex:54)
      • e_ij  — from text only, no coordinates (2methdology.tex:47 = P6)
      • h     — 2methdology.tex:63 (= P4) writes h^{l+1} unchanged on both sides of the
                transformation law. That IS a main-text assertion of h's invariance.
    CONDITIONAL (Codex A4, CONFIRMED): P5 initialises h^{(0)} = Concat(x_i, t_i), which
    rotates with x. So MetaFind's main text asserts h-invariance (P4) and specifies an
    initialisation that violates it (P5). E3 therefore holds only under the invariant-h
    reading. That reading is the one D0-004 / RA-1 owns; D0-009 does not decide it, and
    D0-009's result is CONDITIONAL ON IT. This is NOT orthogonal to D0-004 — Revision 1's
    claim of orthogonality is retracted.
    [INFERENCE, CONDITIONAL]

E4  The SAME "where …" clause (P2) is ALREADY dimensionally inconsistent independently of the
    R³ question. By P5, h^{(0)} ∈ R^{d+3}; so f_h's stated input width 2d+1+e should be
    2d+7+e, and its stated codomain R^d cannot be added to h_i^l ∈ R^{d+3}. MetaFind's type
    annotations in that one clause do not close for f_h either. Cited ONLY as evidence about
    the reliability of that clause. The underlying h^{(0)} question belongs to D0-004 / RA-1
    and is not decided here. (Independently recorded at docs/audit/C_PAPER_CONTRADICTIONS.md
    C4.)
    [OBSERVED — arithmetic on P2 + P5]

E5  Nothing in MetaFind explains, motivates, or uses a 3-dimensional f_x output. There is no
    per-axis or anisotropic-scaling story anywhere. ESSGNN's only stated novelty over EGNN is
    "semantic-aware edge modulation" (2methdology.tex:59), which concerns e_ij, not the
    coordinate channel. Absence of an explanation is NOT evidence that R³ was unintended.
    [PAPER FACT as to silence]

EXECUTABILITY CHECK

  THE GOVERNING THEOREM (assumption-free; replaces Revision 1's Schur/bilinearity argument,
  which Codex A1 correctly broke).

    Fix the invariant output w. Define g(v) = v ⋆ w, a map R³ → R³. Equivariance (P4)
    requires g(Qv) = Q g(v) for every Q ∈ SO(3). Take any Q that fixes v (a rotation about
    the axis v): then g(v) = g(Qv) = Q g(v), so g(v) is fixed by every rotation about v,
    hence g(v) ∈ span(v). Write g(v) = λ(v) v. For arbitrary Q, g(Qv) = Q g(v) = λ(v) Qv and
    also g(Qv) = λ(Qv) Qv, so λ(Qv) = λ(v), i.e. λ depends on v only through ||v||. Also
    g(0) = Q g(0) ∀Q ⇒ g(0) = 0.

      CONCLUSION:  v ⋆ w  =  λ(||v||, w) · v   for some scalar function λ.

    No linearity, bilinearity, or continuity is assumed — only that ⋆ is a function.
    Since ||v|| = ||x_i - x_j|| = d_ij is already an argument of f_x, the admissible family
    is exactly: SCALE (x_i - x_j) BY A SCALAR DERIVED FROM THE INVARIANT INPUTS.

    WHAT THIS DOES NOT SAY (Codex A1, CONFIRMED): it does not say f_x's CODOMAIN must be R¹.
    λ may be any scalar function of w ∈ R³. An R³ codomain is therefore REACHABLE.
    Revision 1's "R³ is unreachable" and "the extra two components can carry no equivariant
    effect" are RETRACTED.

  Reading A — `·` = standard dot product, f_x → R³
    dimensional closure : NO.  (x_i-x_j)·f_x has shape (), x_i^l has shape (3,).
                          R³ + R does not close.  ← the user's required explicit record
    equivariance        : NO.  |(Qv)·w - v·w| = 1.36e-2 ≠ 0 for invariant w.
    verdict             : not executable, and not equivariant either. EXCLUDED.

  Reading B — `·` = Hadamard (element-wise) product, f_x → R³
    dimensional closure : YES. R³ ⊙ R³ = R³, added to x_i ∈ R³. ✔
    equivariance        : NO under the full group. max |Q(v⊙w) - (Qv)⊙w| = 8.21 over 2000
                          random SO(3). Fails P4, which demands "any R ∈ SO(3)".
    subgroup note       : NARROWED per Codex A11 (CONFIRMED). Hadamard is NOT "equivariant
                          only in trivial cases" — diag(1,-1,-1) ∈ SO(3) is diagonal and
                          commutes with diag(w), so a nontrivial subgroup (the π-rotations
                          about the coordinate axes, and axis permutations when w's entries
                          permute accordingly) does commute. The correct statement is simply:
                          Hadamard is not equivariant under ALL of SO(3).
    verdict             : executable, but incompatible with P3/P4.

  Reading C — `·` = scalar multiplication, f_x's codomain read as R¹
    dimensional closure : YES. ✔     equivariance : YES. max err 1.78e-15 (float64 noise).
    verdict             : executable and equivariant — but contradicts the literal "→ R³".

  Reading E — `·` = an UNSTATED scalarising product, f_x → R³ kept literally  [NEW,
              raised by Codex A1/A6, verified and ADMITTED]
    Form              : (x_i - x_j) ⋆ w  :=  λ(w) · (x_i - x_j),  w = f_x(...) ∈ R³.
                        Examples: λ = aᵀw for a learned or fixed a ∈ R³ (e.g. a = (1,1,1));
                        λ = ||w||²; λ = α(d_ij, w).
    dimensional closure : YES. ✔
    equivariance        : YES — exactly the family the governing theorem admits.
    consumes all three outputs : YES.
    satisfies MetaFind's own proof step P8 : YES — B(v,w) = (aᵀw)v gives (Qv)⋆w = Q(v⋆w).
    cost                : `·` must denote an operator MetaFind never names and never uses
                          elsewhere. No standard notation reads `v · w` as "scalarise w, then
                          scale v". MetaFind gives no reducer, so the reducer is INVENTED.
    verdict             : EXECUTABLE, EQUIVARIANT, AND KEEPS "→ R³". Not excluded.

  ── A result that materially lowers the stakes ──────────────────────────────────────────
  If the unstated reducer in Reading E is LINEAR — λ = aᵀw + b, whether a is learned or a
  fixed vector such as (1,1,1) — then Reading E and Reading C define EXACTLY THE SAME
  FUNCTION CLASS. f_x is an MLP (P2); its final layer is Linear(hidden, 3) with weight
  W ∈ R^{3×hidden} and bias b₃, and the composite coefficient is aᵀ(Wz + b₃) + b = (aᵀW)z +
  (aᵀb₃ + b). aᵀW ranges over all of R^{1×hidden} as (a, W) vary, and conversely any
  Linear(hidden,1) is realised by a = (1,0,0) with W's first row set to its weight. So the
  set of realisable coordinate-update functions is identical.
  [DERIVED — linear algebra]
  CAVEAT: identical FUNCTION CLASS is not identical TRAINED MODEL. The R³ form is an
  over-parameterisation; initialisation scale and optimisation trajectory differ, and this
  reproduction has never measured that. A NON-linear reducer (λ = ||w||², say) is a genuinely
  different function class.
  ────────────────────────────────────────────────────────────────────────────────────────

CONTRADICTION CHECK
1. f_x → R³ vs SE(3) equivariance — NOT a strict contradiction. It is a contradiction under
   every NAMED product (Readings A, B, C exhaust dot / Hadamard / scalar), but Reading E
   reconciles both statements using an operator MetaFind leaves undefined. Since P11
   establishes that MetaFind never defines `·`, the honest description is UNDER-SPECIFICATION,
   not contradiction. Revision 1 called this a contradiction; that was wrong.

2. A REAL contradiction does exist nearby, and it is NOT the one this decision was opened
   for: P4 asserts h^{l+1} is invariant, and P5 initialises h^{(0)} = Concat(x_i, t_i), which
   is not. That contradiction is D0-004 / RA-1's, is already recorded at
   docs/audit/C_PAPER_CONTRADICTIONS.md C2/C4, and is NOT decided here. D0-009's result is
   conditional on its resolution (E3).

3. MetaFind-internal weight, for the user's use — NOT a proof of intent:
     favouring a scalar coefficient : P3, P4, P7, P8, E1, E2, E5, plus the SE(3) claim in the
                                      abstract, in contribution (2) of the introduction
                                      (neurips_2025.tex:103) and in 3experiments.tex:8
     favouring a literal R³ codomain: P2 — one type annotation, in one subordinate clause,
                                      in a clause already arity-inconsistent for f_h (E4)
   Every admissible reading (C and E alike) uses a scalar COEFFICIENT. The open question is
   only the CODOMAIN of f_x and the identity of the reducer. That is a narrower question than
   Revision 1 posed.

VERDICT
PAPER-AMBIGUOUS

  Justification against the other two classes:
  • not PAPER-RESOLVED — P11 is a positive finding of silence. MetaFind never defines `·` and
    never says what the three outputs mean. Two executable, equivariant readings survive
    (C and E). Calling this RESOLVED would smuggle a choice into a class label.
  • not PAPER-CONTRADICTORY — Revision 1's grounds for that class were destroyed by Codex A1.
    Reading E keeps BOTH P2 and P4. The class definition ("the main text's R³ cannot hold
    together with … MetaFind's own equivariance claim") is therefore not satisfied.
  • PAPER-AMBIGUOUS matches: "MetaFind does not define it clearly, but several executable
    readings exist."

USER DECISION REQUIRED
YES. MetaFind does not determine f_x's codomain or the reducer. The reproduction must choose.
The genuine options are the A / B / C / D table below. D0-009 does not choose; §8 states a
PROPOSED decision, separated from the findings as WORKFLOW.md §13A requires.
```

---

### The genuine options

| | Option | What it implements | What it keeps | What it gives up | Evidence class of the result |
|---|---|---|---|---|---|
| **A** | **`f_x → R¹` (scalar), keep current behaviour** | `_mlp(..., 1)`; `trans = (x_i − x_j) * w` — exactly `essgnn.py:311-312, 358-359` today | P3, P4 and the appendix proof; correspondence with the only reading MetaFind's `\cdot` usage elsewhere supports (E1) | The literal string "R³" in `2methdology.tex:54` | `IMPLEMENTATION CHOICE` where MetaFind is silent about the codomain, **plus** a `DEVIATION` from P2's literal annotation. **Not** "the paper is wrong" |
| **B** | **`f_x → R³` with `·` read as Hadamard** | `_mlp(..., 3)`; `trans = (x_i − x_j) * w`, `w ∈ R³` | The literal type annotation and the most common plain reading of "·" between two 3-vectors | P3, P4, the appendix proof, and ESSGNN's stated headline property | `DEVIATION` from MetaFind's SE(3) claim. The model **could not be reported as SE(3)-equivariant** |
| **D** | **`f_x → R³` with an explicit scalar reducer (Reading E)** | `_mlp(..., 3)` then a declared reducer `λ = aᵀw` (learned or fixed) or a declared nonlinear reducer; `trans = (x_i − x_j) * λ` | **Both** P2 and P4 simultaneously. The only option that contradicts nothing MetaFind states | Nothing MetaFind states — but the reducer itself is **invented**, since MetaFind names none | `IMPLEMENTATION CHOICE` — the reducer is unsupported by any MetaFind text. **If the reducer is linear this is the same function class as A** (see the boxed result above), so it is a reparameterisation, not a different method |
| **C** | **Flag it; measure; let the user rule with numbers** | `f_x_codomain: scalar \| r3_hadamard \| r3_reduced` behind one config axis + protocol field | All readings available; the ambiguity preserved in the artifact rather than erased | Nothing scientific; costs one config axis, one protocol field, one measurement run | The measurement is `OBSERVED DATA`. Choosing a primary afterwards is still a user decision |

**On Option C specifically.** `CONTEXT.md` §5 justifies the present scalar implementation with "equivariance error 2.2e-16 vs 0.43", and `CONTEXT.md` §5 itself marks those figures **UNVERIFIED in this repository**. The algebra checks in this document do **not** reproduce them — they measure operator identities in isolation, not an end-to-end ESSGNN forward pass; 8.21 and 1.78e-15 are different quantities from 0.43 and 2.2e-16. `docs/audit/C_PAPER_CONTRADICTIONS.md` C3 already records a standing **"verification obligation: an equivariance test at L ≥ 2"** for exactly this. Option C is the only option that discharges it.

**Cheapest moment.** Stage 2 has never run and `data/outputs/checkpoints/` is empty, so no Stage 2 artifact depends on the current choice. A, B, C and D are all free of migration cost today.

---

## 7. Analysis

Compare readings against: MetaFind's own text first · dimensional closure · MetaFind's own equivariance claim · reproducibility · downstream impact · residual uncertainty.

Upstream may appear here only as supplementary corroboration, clearly labelled, never as the deciding reason.

### 7.1 How the paper search was conducted, and what would falsify "exhaustive"

`SOURCE_MANIFEST.json` fixes the corpus: arXiv `2510.04057`, archive sha256 `61351d34…38b4e5`, main tex `neurips_2025.tex`, include tree `{neurips_2025 → 2methdology, 3experiments, appendix → 4backgound}`, `orphan_tex_files: []`, `macro_files: []`, **`supplement_files: []`**. 541 lines of TeX. All 5 files were searched for `f_x`, `\phi_x`, `phi_x`, `R^3`, `\mathbb{R}^{3}`, `\cdot`, `\odot`, `\times`, and for *notation / denote / operator / element-wise / elementwise / Hadamard / scalar*. Every hit is enumerated in P11; the `\cdot` list is complete at 9 tokens over 7 lines, verified by token count, not by eye. `neurips_2025.sty` and the preamble were checked for macro redefinitions — only font, section-heading and float parameters. `MetaFind.drawio.png` was opened and read directly: a block diagram, no mathematical notation.

**Independently re-run by Codex** (2026-08-21), which confirmed zero `\odot`, one `\times`, two `f_x`, four `\phi_x`, and **no omitted definition** across the five TeX files, preamble, `.bbl`, `.sty`, metadata and six figures — and which corrected the `\cdot` count from 6 grouped sites to 9 tokens (finding A7, accepted). Two independent searches agreeing on the silence is the strongest form this claim can take from this corpus.

**What would falsify it.** A published erratum, an arXiv v2, an author repository, or reviewer-response text outside this archive. None is in `docs/paper/`, and `supplement_files: []` says the archive carries none. `UNKNOWN` as to anything outside `docs/paper/MetaFind.gz`.

### 7.2 The chain, and exactly where each link comes from

`docs/audit/E_GRAPH_REVALIDATION.md:175` settles C3 with `[UPSTREAM] settles it: EGNN's φ_x "outputs a scalar value"`. Under governing principle rule 4 that is **inadmissible as a settlement**, and it is not used here.

| Step | Source | Authority |
|---|---|---|
| `d_ij` is invariant | `2methdology.tex:54` — a Euclidean distance | 1 — main text |
| `e_ij` is invariant | `2methdology.tex:47` — text-derived, no coordinates | 1 — main text |
| `h` is invariant | `2methdology.tex:63` — the transformation law writes `h^{l+1}` unchanged on both sides | 1 — main text |
| …but `h^{(0)} = Concat(x_i, t_i)` is **not** invariant | `2methdology.tex:44` | 1 — main text, **conflicting** |
| therefore `f_x`'s output does not rotate, **conditional on the invariant-h reading** | arithmetic on the above | derived, conditional |
| the update must hold for **any** `R ∈ SO(3)` | `2methdology.tex:61-64` | 1 — main text |
| ⇒ `v ⋆ w = λ(‖v‖, w)·v` — a scalar coefficient, codomain undetermined | stabiliser argument (§6) | mathematics |
| MetaFind's own proof performs exactly that commutation | `appendix.tex:56-58` | 2 — corroboration only, see §7.5 |

**No step is upstream.** The conditionality flagged in row 4 is real and is owned by `D0-004`/RA-1.

**A correction to Revision 1.** Revision 1 claimed the chain closes with "no appendix, no EGNN". Codex A3 showed that was false — Revision 1's F-3 cited `appendix.tex:20,29,38-44`. The chain above re-sources every invariance premise to the **main text**, which does support the claim; but Revision 1's phrasing asserted something it had not done, and that is recorded rather than quietly repaired.

### 7.3 Why PAPER-AMBIGUOUS, and why the earlier PAPER-CONTRADICTORY was wrong

Revision 1 reasoned: f_x's output is invariant; the only SO(3)-commuting product is scalar multiplication; therefore an `R³` codomain is unreachable; therefore P2 and P4 contradict. The third step does not follow. The theorem constrains the **coefficient** to be a scalar; it says nothing about the **codomain** of the function that produces the coefficient. `λ = aᵀw` with `w ∈ R³` satisfies every constraint. Codex A1 is CONFIRMED and the verdict changed.

- **Not PAPER-RESOLVED.** P11 is a positive finding of silence. Nothing in MetaFind defines `·` or says what f_x's three outputs are. Readings C and E both survive. RESOLVED would also be the *convenient* verdict — it ratifies the code that already exists and requires no user gate.
- **Not PAPER-CONTRADICTORY.** Its grounds are retracted. It is worth stating that this class was the one Revision 1 chose, and it was *less* convenient than RESOLVED, so the error was not one of convenience — it was an overreach in the mathematics.
- **PAPER-AMBIGUOUS** matches word for word: the operator is undefined and several executable readings exist.

There is one narrower claim that survives intact and should not be lost in the reclassification: **under every product MetaFind's notation could plausibly be naming — dot, Hadamard, scalar — an `R³` codomain and full SE(3)-equivariance cannot both hold.** Reconciling them requires an operator MetaFind never names.

### 7.4 "The paper is simply wrong" — assumed, not demonstrated

`metafind/models/essgnn.py:39-44` records: **"F10 has no flag: it is simply an error in the paper."**

```
FINDING:  MetaFind never defines `·` and never says what f_x's three outputs mean (P11).
          Under any named product, P2's R³ and P4's equivariance cannot both hold; under an
          unnamed scalarising product they can.
          [PAPER FACT for the quotations and the silence; DERIVED for the rest]  [CONFIRMED]

NOT A FINDING: that "→ R³" is an error at all. Reading E keeps it. Even setting Reading E
          aside, nothing identifies WHICH of the two statements the authors mis-stated.
```

The comment collapses §13A's finding/decision separation: an AI-selected remedy is stated with the authority of a demonstrated fact. It then adds "The reference EGNN agrees", corroboration written adjacent to a justification — which is how `[UPSTREAM] settles it` entered `docs/audit/E`. The *behaviour* (scalar) remains defensible; the *stated reason* is not. Note that `docs/audit/C_PAPER_CONTRADICTIONS.md` C3 is more careful than the code comment: it says "Most likely a typo for `\mathbb{R}^1`, but 'most likely a typo' is an inference and is recorded as one."

### 7.5 Upstream — supplementary only, explicitly not load-bearing

`[UPSTREAM FACT — verified here, 2026-08-21]` `docs/paper/egnn_source/sections/model.tex:44`: *"the function $\phi_x: \mathbb{R}^{\text{nf}} \rightarrow \mathbb{R}^1$ that takes as input the edge embedding $m_{ij}$ … and outputs a scalar value."* EGNN's update (`model.tex:15`) writes the product by juxtaposition, with no `\cdot`. MetaFind cites EGNN as the model it "extends" (`2methdology.tex:42`) and states its own novelty as semantic edge modulation (`2methdology.tex:59`), which does not touch the coordinate channel.

**This corroborates Options A and D. It decides neither, and §7.2 does not use it.** Under rule 4 its correct weight is: *consistent with, but incapable of settling*.

**Provenance caveat.** MetaFind's appendix proof (`appendix.tex:56-58`) is a near-verbatim adaptation of EGNN's (`egnn_source/sections/appendix.tex:18-20`), which MetaFind states openly. Codex A5 further showed the step does not even require a scalar codomain — `B(v,w) = (aᵀw)v` satisfies it exactly. So E2/P8 is **corroboration only** and carries less weight than Revision 1 gave it.

### 7.6 Residual uncertainty

| # | Item | Status |
|---|---|---|
| R-1 | MetaFind material outside `docs/paper/MetaFind.gz` (v2, erratum, author code, reviewer response) defining `·` or correcting `→ R³` | `UNKNOWN`. Not checked — no network use. Manifest reports `supplement_files: []` |
| R-2 | `CONTEXT.md` §5's "2.2e-16 vs 0.43" | Still `UNVERIFIED`. D0-009 did **not** reproduce it and does not rely on it. `docs/audit/C_PAPER_CONTRADICTIONS.md` C3 records a standing verification obligation (equivariance test at L ≥ 2) that remains open |
| R-3 | What the authors intended — a typo'd codomain, an unstated reducer, or a proof error | `UNKNOWN` and unknowable from this corpus. This is what the user must rule on |
| R-4 | Interaction with `D0-004` (`architecture_family`, `coord_feat`, `h^{(0)}`) | **NOT orthogonal** — Revision 1's orthogonality claim is RETRACTED per Codex A4. E3's invariant-`h` premise is exactly what `h^{(0)} = Concat(x_i, t_i)` violates. D0-009's conclusion is conditional on D0-004's resolution. `MASTER-IMPACTING` |
| R-5 | `normalize_coord_diff` (`essgnn.py:337`) | Appears nowhere in MetaFind §2.5 or the appendix. Out of D0-009 scope; flagged for Master |
| R-6 | Whether a linear vs nonlinear reducer changes trained behaviour under Option D | `UNKNOWN`. Function classes are identical for a linear reducer (proved in §6); optimisation behaviour is not, and has never been measured here |

---

## 8. Recommended Decision

State the FINDING and the PROPOSED DECISION separately (`WORKFLOW.md` §13A).

The verdict is PAPER-AMBIGUOUS, so D0 **presents the options and does not choose.** The user rules.

Requires user approval: **YES** — paper interpretation + architecture, `WORKFLOW.md` §13B.

### FINDINGS

```
F-1  MetaFind §2.5 types f_x : R^(2d+1+e) → R³ (2methdology.tex:54) and claims full
     SE(3)-equivariance for any R ∈ SO(3) (2methdology.tex:61-64).
     [PAPER FACT]  [CONFIRMED — quoted verbatim]

F-2  MetaFind never defines the operator `·` and never says what f_x's three outputs mean.
     All 9 `\cdot` tokens across 7 lines are enumerated; `\odot` occurs 0 times; no notation
     section, no operator table, no math macro redefinition, no math in any figure;
     SOURCE_MANIFEST.json records `supplement_files: []`.
     [PAPER FACT as to silence]  [CONFIRMED — two independent exhaustive searches, Claude
     and Codex, agreeing]

F-3  Conditional on the invariant-h reading, every argument of f_x is SE(3)-invariant, so
     f_x's output does not rotate. Sourced to MetaFind MAIN TEXT (2methdology.tex:47, :54,
     :63). The condition is real: 2methdology.tex:44 initialises h^{(0)} = Concat(x_i, t_i),
     which is not invariant. That conflict belongs to D0-004 / RA-1.
     [INFERENCE, CONDITIONAL]  [CONFIRMED as conditional]

F-4  Given F-3, the coordinate update's operator must reduce to scaling (x_i − x_j) by a
     scalar λ(d_ij, w). Proved by a stabiliser argument assuming only that the operator is a
     function — no linearity, bilinearity or continuity.
     This constrains the COEFFICIENT, not the CODOMAIN of f_x.
     [DERIVED — mathematics]  [CONFIRMED]

F-5  An R³ codomain is NOT excluded. λ = aᵀw with w ∈ R³ closes dimensionally, is fully
     SE(3)-equivariant, consumes all three outputs, and satisfies MetaFind's own appendix
     step. Revision 1 asserted the opposite; that assertion is RETRACTED.
     [DERIVED]  [CONFIRMED — Codex A1, verified by D0]

F-6  If that reducer is LINEAR, the R³ reading and the scalar reading define exactly the same
     function class; the difference is a reparameterisation, not a different method.
     Optimisation behaviour may still differ and has never been measured here.
     [DERIVED — linear algebra]  [CONFIRMED]

F-7  Under every product MetaFind's notation could plausibly be NAMING — dot, Hadamard,
     scalar — R³ and full SE(3)-equivariance cannot both hold. Dot does not close
     dimensionally (R³ + R); Hadamard is not equivariant under all of SO(3) (max error 8.21);
     scalar requires abandoning R³. Only an operator MetaFind never names reconciles them.
     [DERIVED]  [CONFIRMED]

F-8  MetaFind's appendix step (appendix.tex:56-58) is a near-verbatim adaptation of EGNN's
     proof, and does not by itself require a scalar codomain. It is CORROBORATION ONLY.
     [PAPER FACT for the step; INFERENCE for what it constrains]  [CONFIRMED, non-load-bearing]

F-9  Which of MetaFind's statements — if either — is erroneous is NOT determinable from this
     corpus. "It is simply an error in the paper" (metafind/models/essgnn.py:39-44) is an
     assumption presented as a demonstrated result.
     [UNKNOWN]  [CONFIRMED that it is undetermined]

F-10 FIVE repository records cover this item, not four. Master's §5 inventory omits
     docs/audit/C_PAPER_CONTRADICTIONS.md C3, which is the most disciplined of them
     ("'most likely a typo' is an inference and is recorded as one"). The defects in the
     others are specific, not a generic four-way disagreement (evidence classes corrected
     per Codex A9):
       • docs/audit/E_GRAPH_REVALIDATION.md:175 — settles C3 by `[UPSTREAM]`, inadmissible
         under governing principle rule 4.                    [DERIVED AUDIT RECORD]
       • docs/audit/F_CODE_GRAPH_CONSISTENCY.md:27 — says CONSISTENT although its own
         definition at :6-18 requires paper agreement.        [DERIVED AUDIT RECORD]
       • docs/audit/D_IMPLEMENTATION_FORMULA_CONTRACT.md:124 `[PAPER CONTRADICTION]` and
         workflow/CONTEXT.md §5 `DEVIATION` are NOT in conflict with each other.
       • metafind/models/essgnn.py:39-44, :311-312 — the only OBSERVED IMPLEMENTATION here.
     [MASTER-IMPACTING]  [CONFIRMED]

F-11 CONTEXT.md §5's supporting figures "2.2e-16 vs 0.43" remain UNVERIFIED in this
     repository, and docs/audit/C_PAPER_CONTRADICTIONS.md C3 records a still-open
     verification obligation (equivariance test at L ≥ 2). D0-009 did not discharge it.
     [UNVERIFIED — CONTEXT.md §5 says so itself]

F-12 data/outputs/essgnn_arch_protocol.json EXISTS and records
     {architecture_family: appendix_shared_msg, coord_feat: current, distance: squared,
      hidden_dim: 128, n_layers: 4, pooling: mean, decided_by: Kyzen 2026-08-19}.
     Codex A10 reported it missing; that report is REJECTED — `data` is a symlink to
     /home/kyzen/data/MetaFind, so a repo-confined path search does not see it.
     [OBSERVED DATA]  [CONFIRMED — file read 2026-08-21]

VERDICT:  PAPER-AMBIGUOUS
```

### PROPOSED DECISION

```
PROPOSED DECISION:  Option A — keep f_x's effective codomain at R¹ (scalar), i.e. leave the
                    computational behaviour of metafind/models/essgnn.py:311-312 and
                    :358-359 UNCHANGED, and replace the recorded justification.
PROPOSED BY:        D0-009 (Claude), after Codex adversarial review
EVIDENCE CLASS:     IMPLEMENTATION CHOICE where MetaFind is silent (F-2), plus a DEVIATION
                    from 2methdology.tex:54's literal annotation.
                    NOT a PAPER FACT. NOT "the paper is wrong".
CONDITIONAL ON:     D0-004 / RA-1 resolving h^{(0)} toward the invariant-h reading (F-3).
                    If h carries raw coordinates, F-3's premise fails and this must be redone.
REQUIRES USER APPROVAL:  YES  (WORKFLOW.md §13B)
D0 DOES NOT ENACT THIS.
```

**Why A rather than B, C or D.**

- **vs B** — B is the only option that abandons a property MetaFind states in its abstract, in contribution (2) of its introduction, in §2.5's motivation, in §2.5's formal claim, in §3.1, and throughout Appendix C. It buys the literal string "R³" at the cost of ESSGNN's stated reason to exist.
- **vs D** — D contradicts nothing MetaFind states, which is a genuine advantage and the user should weigh it. Against it: the reducer is **invented**; MetaFind names none, so D trades a deviation-from-a-type-annotation for an invention-with-no-textual-support. And by F-6, a linear reducer gives the same function class as A anyway, so D's fidelity gain is largely notational. A nonlinear reducer would be a real difference — and would be entirely unsupported by the paper.
- **vs C** — C is not exclusive with A and is the honest way to close R-2 and C3's standing obligation. **If the user wants the number before ruling, C then A is the better sequence, and D0-009 says so explicitly rather than hiding it inside a recommendation for A.**

**This is a recommendation, not a resolution.** The verdict is PAPER-AMBIGUOUS; MetaFind cannot make this call. The user rules under §13B.

### If the user rules for Option A — required repository changes

No change to computed behaviour. All changes are record-accuracy.

| File | Change |
|---|---|
| `metafind/models/essgnn.py:39-44` | Replace "it is simply an error in the paper" with the F-2/F-4/F-5/F-9 framing: MetaFind does not define `·`; an R³ codomain is *not* excluded; this reproduction chooses scalar. Move the EGNN sentence into an explicitly-labelled *supplementary corroboration* note, or delete it |
| `metafind/models/essgnn.py:311-312` | "corrected to scalar per Eq. 13" asserts an error. Reword — the appendix step does not require a scalar codomain (F-8) |
| `docs/audit/E_GRAPH_REVALIDATION.md:175` | Replace `[UPSTREAM] settles it` with the MetaFind-internal chain in §7.2 |
| `docs/audit/F_CODE_GRAPH_CONSISTENCY.md:27` | `CONSISTENT` conflicts with its own definition at `:6-18`; re-state |
| `docs/audit/D_IMPLEMENTATION_FORMULA_CONTRACT.md:124` | C3 stays flagged, but the label should become PAPER-AMBIGUOUS rather than PAPER CONTRADICTION |
| `docs/audit/C_PAPER_CONTRADICTIONS.md` C3 | Closest to correct already. Add Reading E — its "most likely a typo" inference is sound but not the only reconciliation |
| `docs/graph/` U-26 | Align |
| `workflow/CONTEXT.md` §5 | Keep the row as a recorded choice; replace "2.5 literal text is wrong; scalar gives equivariance error 2.2e-16 vs 0.43" — the first clause is F-9 (undetermined) and the second is F-11 (unverified). Point at this decision file |
| `workflow/DECISION_LEDGER.md` | Add the ratified entry |

**No code, test, protocol, artifact, or data file was modified by D0-009.** Only this decision file was written.

---

## 9. Codex Adversarial Review

**Status: COMPLETED.** `codex-cli 0.148.0`, `codex exec --sandbox read-only`, 2026-08-21, 207,627 tokens. Codex was given the binding rules (MetaFind primary; upstream supplementary only; repo code is not paper intent) and was asked to falsify, not to agree. Brief and raw output retained in the session scratchpad.

Codex was required to attack: search exhaustiveness; verdict-class convenience; upstream laundering; whether "the paper is wrong" was assumed; dismissed dimensionally-closing readings; the provenance of the equivariance argument.

**Codex overall verdict:** *"No. The audit proves neither that an R³ codomain is impossible nor that one paper statement must be discarded; the undefined operator admits dimensionally closed, fully equivariant R³ readings, so PAPER-AMBIGUOUS survives and PAPER-CONTRADICTORY does not."*

| ID | Sev | Attack |
|---|---|---|
| A1 | BLOCKER | Reading D's Schur argument assumes linearity in `v` and never shows `w` must be scalar. `B(v,w) = (w₁+w₂+w₃)v` closes, consumes all three outputs, and is fully equivariant |
| A2 | MAJOR | The verdict depends on A1's false impossibility result; PAPER-AMBIGUOUS is the surviving class |
| A3 | MAJOR | "Strike the appendix and the chain still closes" is false — the fallback still invokes F-3, which Revision 1 sourced to the appendix |
| A4 | MAJOR | F-3 is not CONFIRMED-unconditional: `h^{(0)} = Concat(x_i, t_i)` is not invariant; the appendix only *assumes* invariance. The claimed orthogonality to the other ESSGNN axes is unsupported |
| A5 | MAJOR | F-5's claim that the appendix step permits only a scalar φ_x is false; `B(v,w) = (aᵀw)v` satisfies it exactly. Only *upstream* types φ_x as R¹ |
| A6 | MAJOR | The A/B/C options assume one paper statement must be erroneous, and wrongly equate "keep R³" with Hadamard. A fourth option — R³ plus an explicit scalar reducer — is missing |
| A7 | MINOR | The `\cdot` count is 9 tokens over 7 lines, not 6 occurrences |
| A8 | MINOR | F-4's numeric maxima are cited as CONFIRMED with no reproducible script, seed distribution, or environment recorded; and a Hadamard-only test cannot quantify over all products |
| A9 | MINOR | The four-record "disagreement" claim is loose, and derived audit records were misclassified as OBSERVED IMPLEMENTATION |
| A10 | MINOR | `essgnn_arch_protocol.json` does not exist anywhere in the repository |
| A11 | NIT | "Hadamard is equivariant only in trivial/degenerate cases" is too broad — `diag(1,−1,−1) ∈ SO(3)` commutes with `diag(w)` |

---

## 10. Claude Verification of Codex Findings

| ID | Verdict | Verification performed | Action taken |
|---|---|---|---|
| **A1** | **CONFIRMED — BLOCKER** | Re-derived from scratch. Codex is right that the bilinearity assumption was unjustified and, more importantly, right that the conclusion was wrong regardless: `λ = aᵀw` is a counterexample. D0 replaced the argument with an assumption-free stabiliser proof, which yields `v ⋆ w = λ(‖v‖,w)·v` — constraining the *coefficient*, not f_x's *codomain*. Numerically re-checked (`equivmap.py` below): a rotation about the axis `v` fixes `v` to 5.6e-17 but moves the Hadamard output by 6.99e-2, so `v ⊙ w ∉ span(v)`; the scalar output's out-of-span component is exactly 0 | **Verdict changed from PAPER-CONTRADICTORY to PAPER-AMBIGUOUS.** "R³ is unreachable" and "the extra two components can carry no equivariant effect" retracted throughout |
| **A2** | **CONFIRMED — MAJOR** | Follows from A1. Checked the class definitions in §6: PAPER-CONTRADICTORY requires that R³ "cannot hold together with … MetaFind's own equivariance claim". Reading E makes both hold, so the class does not apply | Verdict changed. §7.3 rewritten to argue the new class and to record why the old one failed |
| **A3** | **CONFIRMED — MAJOR** | Read Revision 1's own text: the fallback claimed "no appendix, no EGNN" while F-3 cited `appendix.tex:20,29,38-44`. Genuine self-contradiction | Re-sourced every invariance premise to MetaFind **main text** (`2methdology.tex:47`, `:54`, `:63`). `2methdology.tex:63` writes `h^{l+1}` unchanged on both sides of the transformation law, which is a main-text assertion of h-invariance. The claim now matches what was actually done |
| **A4** | **CONFIRMED — MAJOR** | Verified `2methdology.tex:44` (`h^{(0)} = Concat(x_i, t_i)`) against `appendix.tex:29` ("We begin by assuming that h⁰ is invariant"). The paper does assert both. Independently corroborated by `docs/audit/C_PAPER_CONTRADICTIONS.md` C2/C4, which D0 had not read before Codex surfaced it | F-3 marked CONDITIONAL. **R-4's "orthogonal to D0-004" claim RETRACTED** and re-flagged `MASTER-IMPACTING`. D0-009's result now states its dependency on D0-004 explicitly |
| **A5** | **CONFIRMED — MAJOR** | Checked `B(v,w) = (aᵀw)v` against `appendix.tex:56-58` by hand: `(Qv)⋆w = (aᵀw)Qv = Q((aᵀw)v) = Q(v⋆w)`. The step is satisfied with `w ∈ R³` | F-8 downgraded to CORROBORATION ONLY, non-load-bearing. §7.5 rewritten |
| **A6** | **CONFIRMED — MAJOR** | Option B did conflate "keep R³" with Hadamard. The reducer option is genuinely distinct | **Option D added.** Also derived and added a result Codex did not raise: for a *linear* reducer, Option D and Option A are the same function class (F-6), which materially lowers the stakes of the whole decision |
| **A7** | **CONFIRMED — MINOR** | `grep -o '\cdot' *.tex \| wc -l` → 9; `grep -c` → 4 lines in `2methdology.tex`, 3 in `appendix.tex` | P11 corrected to 9 tokens / 7 lines, with per-line attribution. The silence conclusion is unaffected |
| **A8** | **CONFIRMED — MINOR** | Correct on both counts: no script or environment was recorded, and a Hadamard sample cannot support a universal claim | Both scripts embedded verbatim below with seeds, environment and commit. The universal claim is now carried by the stabiliser **proof**, not by sampling; the numbers are demonstrations of specific readings only |
| **A9** | **PARTIALLY CONFIRMED — MINOR** | Confirmed that `D:124` and `CONTEXT.md` §5 are compatible, that `F:27` conflicts with its own definition at `:6-18`, and that derived audit records are not OBSERVED IMPLEMENTATION. **Codex missed something larger:** there are FIVE records, not four — `docs/audit/C_PAPER_CONTRADICTIONS.md` C3, absent from Master's §5 inventory, and the most disciplined of them | F-10 rewritten with per-record defects and corrected evidence classes, and flagged `MASTER-IMPACTING` for the missing fifth record |
| **A10** | **REJECTED** | The file exists: `/home/kyzen/data/MetaFind/outputs/essgnn_arch_protocol.json`, reachable as `data/outputs/essgnn_arch_protocol.json` because `data` is a symlink to `/home/kyzen/data/MetaFind` (`ls -la data` confirms). Contents match Master's §5 claim exactly. Codex's repo-confined path search could not follow the symlink | No change to §5. Recorded as F-12 so the rejection is auditable |
| **A11** | **CONFIRMED — NIT** | `diag(1,−1,−1)` has determinant +1 and is diagonal, so it commutes with `diag(w)`. Revision 1's "trivial or degenerate" was wrong | Narrowed to "Hadamard is not equivariant under all of SO(3)", which is the claim P4 actually needs |

**Not raised by Codex; found by D0 during the same round.** The linear-reducer equivalence (F-6). It weakens D0's own recommendation for Option A by showing Option D is largely a reparameterisation rather than a rival method, and it is recorded for that reason.

**Rounds used: 1 of 2.** A second round was not run. Every BLOCKER and MAJOR finding was accepted and the verdict changed as a result, so there was no disagreement left to arbitrate; the remaining items are corrections already applied. A re-review of Revision 2 would be a reasonable Master-level request but is not required by the two-round limit.

### Reproducibility of the numeric checks

```
commit      35a3dfb3ab7cae6e1b6c58c4794d745852e822cc  (working tree dirty; no file touched
                                                       by D0-009 other than this one)
interpreter /home/kyzen/miniconda3/envs/MetaFind/bin/python  — Python 3.11.15
numpy       2.4.6
run at      2026-08-21
scope       standalone; imports NO repository code and writes NO repository artifact
```

`dcheck.py` — enumerates the readings. `rand_SO3` draws a 3×3 standard-normal matrix, QR-factorises it, fixes the sign convention from `diag(R)`, and flips a column if `det < 0`. `v`, `w3`, `c` are standard normal. 2000 draws, `np.random.default_rng(20260821)`.

```python
import numpy as np
rng = np.random.default_rng(20260821)
def rand_SO3(rng):
    A = rng.normal(size=(3, 3)); Q, R = np.linalg.qr(A); Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0: Q[:, 0] *= -1
    return Q
worst_had = worst_sca = 0.0
for _ in range(2000):
    Q = rand_SO3(rng); v = rng.normal(size=3); w3 = rng.normal(size=3); c = rng.normal()
    worst_had = max(worst_had, np.abs(Q @ (v * w3) - (Q @ v) * w3).max())   # Hadamard
    worst_sca = max(worst_sca, np.abs(Q @ (v * c)  - (Q @ v) * c ).max())   # scalar
v = rng.normal(size=3); w3 = rng.normal(size=3); Q = rand_SO3(rng)
print(np.shape(np.dot(v, w3)))                       # () -- dot does not close against (3,)
print(abs(np.dot(Q @ v, w3) - np.dot(v, w3)))        # 0.013587904376220372
print(worst_had, worst_sca)                          # 8.20752060049811  1.7763568394002505e-15
```

`equivmap.py` — the stabiliser test behind F-4. Builds `Q` as a Rodrigues rotation of 0.7 rad about the axis `v`, so `Q v = v`.

```python
import numpy as np
rng = np.random.default_rng(7)
v = rng.normal(size=3); w = rng.normal(size=3); had = v * w
axis = v / np.linalg.norm(v); th = 0.7
K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
Qv = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K @ K)
print(np.abs(Qv @ v - v).max())        # 5.551115123125783e-17   -- Q fixes v
print(np.abs(Qv @ had - had).max())    # 0.06992726901008257     -- but not v*w
# out-of-span(v) component: Hadamard 0.10846125518645355 ; scalar exactly 0.0
```

These demonstrate specific readings. The **general** claim in F-4 rests on the stabiliser proof in §6, not on these samples.

---

## 11. Final Recommendation to Master

### Verdict

**`PAPER-AMBIGUOUS`.** MetaFind types `f_x → R³` (`2methdology.tex:54`), claims full SE(3)-equivariance for any `R ∈ SO(3)` (`2methdology.tex:61-64`), and **never defines the operator `·`** or says what the three outputs mean. Two executable, fully equivariant readings survive: scalar `f_x` (Reading C), and literal `R³` reduced to a scalar by an unstated reducer (Reading E). MetaFind cannot choose between them.

The §6 report in the user's required format is above and is not repeated here.

### What changed during review, and why it matters

D0-009 first returned `PAPER-CONTRADICTORY` on the ground that an `R³` codomain was mathematically unreachable. **That was wrong**, and Codex's BLOCKER `A1` broke it. The corrected mathematics (assumption-free stabiliser argument) constrains only the *coefficient* to be a scalar; f_x's codomain is left free. Master should note that the first answer would have handed the user a false dichotomy, and that the correction came from adversarial review rather than from D0.

The narrower claim that does survive: **under every product MetaFind's notation could plausibly be naming — dot, Hadamard, scalar — an `R³` codomain and full SE(3)-equivariance cannot both hold.** Reconciling them requires an operator MetaFind never names.

### Proposed decision

`Option A` — leave the scalar behaviour at `metafind/models/essgnn.py:311-312, :358-359` **unchanged**, and rewrite the justification. Class: `IMPLEMENTATION CHOICE` where MetaFind is silent, plus a `DEVIATION` from the literal annotation. **Not** a PAPER FACT, and **not** "the paper is wrong". Requires user approval. **If the user wants the missing equivariance number before ruling, run `Option C` first, then `A`.**

### Remaining uncertainty

`R-1` material outside the archive · `R-2` the unreproduced `2.2e-16 / 0.43` and C3's still-open equivariance obligation · `R-3` authorial intent, unknowable here · `R-4` the D0-004 dependency · `R-6` whether a linear vs nonlinear reducer changes trained behaviour.

### MASTER-IMPACTING FINDINGS

| # | Finding | Why Master must handle it |
|---|---|---|
| **MIF-1** | **D0-009's conclusion is CONDITIONAL on `D0-004` / RA-1.** F-3's invariant-`h` premise is precisely what `h^{(0)} = Concat(x_i, t_i)` (`2methdology.tex:44`) violates. Revision 1 claimed the two were orthogonal; **retracted** | Dependency ordering. If D0-004 resolves toward coordinate-carrying `h`, D0-009 must be redone |
| **MIF-2** | **Master's §5 record inventory is incomplete.** A fifth record exists — `docs/audit/C_PAPER_CONTRADICTIONS.md` C3 — and is the most disciplined of the five. §5 lists only four | The framing given to D0 omitted the best existing evidence. Worth checking whether other decision files inherit the same gap |
| **MIF-3** | `docs/audit/F_CODE_GRAPH_CONSISTENCY.md:27` records `CONSISTENT` although its own definition at `:6-18` requires paper agreement | A self-inconsistent audit record, independent of how the user rules here |
| **MIF-4** | `docs/audit/C_PAPER_CONTRADICTIONS.md` C3's **verification obligation — "an equivariance test at L ≥ 2" — is still open**, and `CONTEXT.md` §5's `2.2e-16 / 0.43` still rests on it | The present position is partly supported by a number this repository has never reproduced |
| **MIF-5** | `normalize_coord_diff` (`essgnn.py:337`) has no basis in MetaFind §2.5 or the appendix | Outside D0-009's scope; needs an owner |

**Nothing found here touches `D2a_stage1-protocol-refresh`.** D0-009 read no file in D2a's scope and modified none. The scope wall held.

### Files modified by D0-009

`workflow/decisions/D0-009_essgnn-fx-codomain.md` — this file only. No code, test, protocol, artifact, dataset, or audit document was touched.

**D0 stops here.**

---

## 12. Master Integration Recommendation

*Master fills this after review. A recommendation, not an acceptance.*

---

## 13. USER REVIEW BRIEF

*Master fills this per `workflow/USER_REVIEW_TEMPLATE.md`.*

---

## 14. USER Final Decision

*Only the user fills this.* `APPROVE` · `REJECT` · `MODIFY` · `INVESTIGATE MORE`

---

## D0 Operating Rules

D0 investigates and recommends. D0 does not decide, does not mark its own recommendation accepted, does not fill Sections 12–14, and does not modify code, tests, protocol, or artifacts.

If D0 finds something that changes project architecture, another task's contract, dependency order, or a shared assumption, report `MASTER-IMPACTING FINDING` with evidence.

Master is the integration owner. **The user is the final research authority.**
