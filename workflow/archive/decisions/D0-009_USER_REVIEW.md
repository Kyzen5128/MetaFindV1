# USER REVIEW BRIEF

**Decision ID:** `D0-009_essgnn-fx-codomain`
**D0 verdict:** `PAPER-AMBIGUOUS`
**Master Recommendation:** `ACCEPT WITH FOLLOW-UP` — accept the research finding. **Do not adopt any option yet.**
**Integration status:** `AWAITING_USER_REVIEW`

---

## 1. What was found

| # | Finding |
|---|---|
| FIND-1 | **MetaFind writes `f_x: R^(2d+1+e) → R³`** — `2methdology.tex:54`, verbatim. This is not a transcription artefact; it is typed alongside `f_h → R^d` in the same sentence |
| FIND-2 | **MetaFind never defines the `·` in the coordinate update** (`2methdology.tex:52`). Not as dot product, Hadamard, scalar multiplication, or any contraction |
| FIND-3 | **MetaFind claims equivariance for *any orthogonal* `Q ∈ R^{3×3}`** — `appendix.tex:23` — which is broader than SO(3): it includes reflections |
| FIND-4 | Three readings are executable, and they are not equivalent. **A** scalar (abandons the literal `R³`) · **B** Hadamard (closes dimensionally, conflicts with the equivariance claim) · **D** `R³` + an explicit contraction (keeps both, but the contraction is not in the paper) |
| FIND-5 | **Upstream EGNN can no longer settle this.** `E_GRAPH_REVALIDATION.md:175` marked C3 `VERIFIED` on `[UPSTREAM] settles it`. Under your governing principle, upstream is supplementary only |
| FIND-6 | **The existing scalar code is not evidence of paper intent.** It is OBSERVED IMPLEMENTATION and a recorded deviation |

**The verdict is not "the paper is wrong."** It is that MetaFind states `R³` clearly and leaves the operator undefined, so more than one faithful implementation exists.

---

## 2. Evidence / provenance

| Claim | Master verified? | Result |
|---|---|---|
| `f_x → R³` verbatim | **direct** | `2methdology.tex:54` — `f_x: \mathbb{R}^{(2d + 1 + e)} \to \mathbb{R}^{3}` |
| The `·` in the update | **direct** | `2methdology.tex:52` — `(x_i^l - x_j^l) \cdot f_x(...)` |
| The operator is undefined | **direct, exhaustive** | `grep -riE "hadamard\|element-?wise\|inner product\|dot product\|contraction"` across **all five** `.tex` files → **zero hits.** The only `\times` occurrence is `Q ∈ R^{3×3}` at `appendix.tex:23`, a matrix shape, not an operator definition |
| Equivariance claimed for any orthogonal `Q` | **direct** | `appendix.tex:23`, `:53`, `:61` |
| `h` invariance | **direct** | `appendix.tex:29` **assumes** `h^0` invariant; `:68` concludes the feature update is invariant |
| Implementation feeds nothing coordinate-dependent into `h` | **direct** | `essgnn.py:353` — `f_h(cat([h[row], h[col], radial, edge_attr]))`. `radial` is a distance; `edge_attr` is text-derived. Both invariant |

**Codex:** round 1 completed and its BLOCKER **changed the verdict**. Round 2 not run. Master found no new contradictory evidence, so nothing was re-run.

---

## 3. Master's adjudication of the D0-004 dependency (MIF-1)

D0-009 asked that its final decision be blocked pending `D0-004`, on the ground that *if D0-004 concludes `h` carries coordinate-dependent information, the equivariance analysis must be redone.*

**Master's ruling: the premise is contradicted by both the paper and the code. `MIF-1` is REJECTED as a blocker, and retained as a caution.**

| | Evidence |
|---|---|
| **Paper** | `appendix.tex:29` — "We begin by **assuming** that `h^0` is invariant to SE(3) transformations on `x`". `appendix.tex:68` — "the feature update is **invariant** to SE(3) transformations of positions". `h` invariance is not an open question in MetaFind; it is an explicit assumption carried through the proof |
| **Code** | `essgnn.py:353` — the only inputs to `f_h` are `h`, `radial`, and `edge_attr`. Nothing coordinate-dependent enters `h` |
| **Scope** | `D0-004` concerns the `coord_feat` / `architecture_family` coupling — **which layer's `h` feeds `f_x`** (`h^l` vs `h^{l+1}`). Under the paper's proof **both are invariant**, so the equivariance analysis is unchanged either way |
| **Already closed** | You closed `h^l` vs `h^{l+1}` on 2026-08-21 in favour of §2.5's sequential update |

**Also independent of `h`:** Option B's equivariance conflict arises because element-wise scaling of a 3-vector does not commute with an orthogonal `Q`. That holds whatever `h` is.

**What survives as a caution.** `D0-004` is genuinely open, and if it ever produced evidence that `h` is coordinate-dependent, that would contradict `appendix.tex:29` and would be a `MASTER-IMPACTING FINDING` in its own right — reopening far more than D0-009. On current evidence it will not.

**Consequence:** the final option choice is **available to you now**. It does not have to wait for `D0-004`. You may still prefer `INVESTIGATE MORE` for other reasons — see §7.

---

## 4. MIF-2 … MIF-5 impact classification

| | Claim | Master's finding | Class |
|---|---|---|---|
| **MIF-2** | The audit inventory omits `C_PAPER_CONTRADICTIONS.md` C3 | **PARTIALLY CONFIRMED — and the target was wrong.** C3 is **not** missing from that file: it has a full section at line 114, rated **SEVERE**, plus a summary row at 348 and grouping at 59/356. What was missing is that **Master's own §5 framing** omitted the primary contradiction registry. **Master's bookkeeping defect, now corrected** in the decision file §5. The D0-009 paper result was not touched | **bookkeeping, Master's** — fixed |
| **MIF-3** | `F_CODE_GRAPH_CONSISTENCY.md:27` contradicts its own classification | **CONFIRMED.** The row reads `f_x → R^1 \| says R^3 (C3) \| … \| CONSISTENT`. **Context Master adds:** every row in that table follows the same pattern, so the file's `CONSISTENT` appears to mean "code matches the graph spec", not "code matches the paper". The ambiguity is real and the column is unlabelled | **documentation consistency follow-up** |
| **MIF-4a** | `2.2e-16` vs `0.43` not reproduced in this repo | **CONFIRMED**, and it **cannot** be reproduced: the `R³` variant does not exist in code. Must not be described as a verified measurement. Already flagged in `CONTEXT.md` §5 | **UNVERIFIED figure** — must stay labelled |
| **MIF-4b** | "L≥2 equivariance test debt stays OPEN" | **PARTIALLY REJECTED.** `test_se3_equivariance` **exists and runs at `n_layers=3`** (`test_essgnn.py:102`, `:112`), and `test_equivariance_negative_injection` (`:129`) proves it is not vacuous. What is genuinely absent is a test of the **`R³` interpretation** — impossible while only the scalar is implemented. **The debt is narrower than stated** | **narrowed** — real debt is R³-variant coverage |
| **MIF-5** | `normalize_coord_diff` has no MetaFind authority | **CONFIRMED**, low urgency. `essgnn.py:189` defaults it to `False`, and `:265` records "nor any normalisation of `(x_i - x_j)`". **Off today, zero current effect.** Registered as a separate reproduction-fidelity candidate — **not** for D0-009 to decide | **new candidate, registered** |

---

## 5. Proposed / implemented decisions

| # | Item | Authority | Classification |
|---|---|---|---|
| 1 | Verdict `PAPER-AMBIGUOUS` | D0-009 evidence, Master-verified | **FINDING**, not a decision |
| 2 | `f_x → R³` is what MetaFind states | `2methdology.tex:54` | **PAPER FACT** |
| 3 | The `·` is undefined by MetaFind | exhaustive search, zero hits | **PAPER FACT (as to silence)** |
| 4 | `h` is invariant | `appendix.tex:29`, `:68` | **PAPER FACT** (assumed and carried through the proof) |
| 5 | Options A / B / D | — | **NONE ADOPTED.** No option is proposed for acceptance in this decision |
| 6 | `[UPSTREAM] settles it` is inadmissible as a settlement | your governing principle | **USER DECISION** |
| 7 | If a contraction is linear and sits after `f_x`'s final layer, it composes with a scalar formulation | D0-009 analysis | **MATHEMATICAL / IMPLEMENTATION ANALYSIS — not a PAPER FACT.** Requires confirming the actual final-layer structure satisfies the equivalence |

---

## 6. Impact

- **Nothing changes today.** `essgnn.py`, tests, and the protocol are untouched, as instructed.
- **Stage 2 has never run.** `checkpoints/` is empty, `D5_stage2-prereq` has not started. This is still the cheap moment.
- **`D1_n06-reencode` is unaffected** — different stage, different module. D1 is `READY` and may proceed regardless of this decision.
- **If an option other than A is eventually adopted:** `essgnn.py`, the equivariance tests, `docs/audit/` C3, `docs/graph/` U-26, `CONTEXT.md` §5, and every Stage 2 result would change.

---

## 7. Remaining UNKNOWN / follow-up

| ID | Item | Owner |
|---|---|---|
| **Open** | **Which option this reproduction adopts.** No option is adopted. `A` is **not** pre-selected despite being what the code does today | **you** |
| MIF-3 | `F_CODE_GRAPH_CONSISTENCY.md` — label what its `CONSISTENT` column means | follow-up |
| MIF-4a | `2.2e-16` / `0.43` — reproduce or retire. Cannot be reproduced without an `R³` variant | deferred |
| MIF-4b | `R³`-variant equivariance coverage — only meaningful if an `R³` option is adopted | conditional |
| MIF-5 | `normalize_coord_diff` — separate reproduction-fidelity candidate. Off by default | registered, unassigned |
| — | `E_GRAPH_REVALIDATION.md:175`'s `[UPSTREAM] settles it` reasoning needs correcting whatever you decide | Master |
| — | `D0-004` remains open. On current evidence it does **not** gate this decision | — |

---

## 8. USER ACTION REQUIRED

**Two separable questions.** Master recommends splitting them.

**(a) Do you accept the research finding — `PAPER-AMBIGUOUS`, with no option adopted?**

Master verified the load-bearing evidence independently: `R³` is stated verbatim, the operator is undefined across all five source files, and `h` invariance is a paper assumption rather than an open question.

**(b) Do you want to rule on the option now, or defer?**

D0-009 recommended `INVESTIGATE MORE` **on the ground that `D0-004` blocks it. Master rejects that ground** — see §3. If you defer, it should be for a reason you choose, not because of a dependency that the paper and the code both contradict.

- `APPROVE` — accept the finding; no option adopted; the option ruling stays open as a tracked item
- `REJECT`
- `MODIFY` — e.g. accept the finding and rule on the option in the same breath, or attach conditions
- `INVESTIGATE MORE` — name what should be investigated, given that the `D0-004` ground is rejected

**Master has adopted nothing, and specifically has not adopted Option A.** `essgnn.py` is untouched.
