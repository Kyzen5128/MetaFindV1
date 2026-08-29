# δ — the smallest improvement worth acting on, for the Stage 1 LR sweep

**Status: IN FORCE.** Kyzen chose option 甲 on 2026-08-30. Entered in
`workflow/DECISION_LEDGER.md` as **`DL-033`** by Master (`bd5ca70`, band corrected in
`65078f5`). The ledger is the authority; this file is the block evidence behind it.

---

## The decision

```
δ = 1.0 percentage point  (0.010 absolute R@1)
metric   = mean R@1 over {text, image, pc}
protocol = C_dev_selection  (query dev_val, gallery dev_val, 4,569)
```

Applied to the **paired** difference between two arms, per the rule the Reviewer relayed
from Codex on 2026-08-30 (`REVIEW.md` R-33):

```
d_s  = metric(LR_a, seed_s) - metric(LR_b, seed_s)     ← same seed, both arms
compute the mean paired difference and its uncertainty interval

interval lower bound > δ   → real improvement
interval upper bound < δ   → stop
interval straddles δ       → add a seed
```

**The invariant this supports is `same arm_config_hash AND same pools_sha256 ⇒ same
treatment`** — not `same arm_config_hash ⇒ same treatment`. [ULIP2 REVIEWER 2026-08-30]
Split identity lives in the run record, per Codex's schema, so the hash alone does not
survive a change of pools. Stated here so the hash is not read as a stronger guarantee
than it is.

δ is declared **before** the sweep runs and is **not** derived from observed scatter.

⚠ If the first 4×2 round measures seed-to-seed scatter **wider than 1.0 pp**, this rule
will never return "real improvement". That is the rule working, not failing — but the
response is to take it back to Kyzen, **never** to move δ after seeing the numbers.
That is Codex's requirement and the reason is that a threshold chosen after seeing the
results is not a threshold.

---

## Why this metric and not `mean_R@1`

Four of the seven reported conditions are pinned at ≥0.98 on protocol C
(`text+image`, `text+pc`, `image+pc`, `full`), so the seven-cell mean is mostly
constant and a real difference in the three live cells is divided by seven before it
is read. **Kyzen approved the selector change on 2026-08-29** (relayed by the Reviewer,
R-33 item 3). The seven-cell mean and every individual cell are still reported in full,
as a guardrail.

`full` is at exactly 1.0000 in every checkpoint we hold. The mechanism is understood
(`INFERENCE`, not established as the sole cause — see the negative-control work owed by
n15), but a cell that cannot move cannot contribute to a comparison either way.

---

## What the paper gives, and what it does not

`OBSERVED` — read directly from `docs/paper/metafind_source/3experiments.tex`:

| Fact | Location |
|---|---|
| Table 1 prints R@1/R@5 to **one decimal place** (`13.8 / 23.1`) | `:36-46` |
| Table 2 prints to two decimals (`3.42`) | `:69-72` |
| Table 3 ablations print to one decimal | `:94-108` |
| **No standard deviation anywhere** | whole file |
| **No seed, no repeated runs, no "averaged over N runs"** | whole file |

So the paper supplies **no noise information at all**, and δ **cannot be derived from it**.
This is `UNKNOWN` in the paper-silence sense, not a value we failed to find.

The nearest anchor the paper does give is the size of the gaps it is willing to draw a
conclusion from in its own ablation table (`3experiments.tex:94-108`):

Ablation effects **against the full model (11.4)**, which is the comparison the table is
making:

```
w/o iterative retrieval    11.3   →  0.1 pp
w/ Layout Context (GAT)    11.0   →  0.4 pp
Padding with 0             10.5   →  0.9 pp
Fusion = MLPs               9.9   →  1.5 pp
Modality Dropout = 50%     13.2   →  1.8 pp   (above the full model)
Fusion = Mean               9.4   →  2.0 pp
w/o Layout Context         13.5   →  2.1 pp   (above the full model)
Train fuser only            8.7   →  2.7 pp
Modality Dropout = 10%      7.3   →  4.1 pp
```

All nine rows of Table 3, independently recomputed by the ULIP2 Block Reviewer from the
paper on 2026-08-30. An earlier version of this list omitted the two Layout Context rows.

⚠ **Corrected 2026-08-30 (Codex).** An earlier version of this file quoted the band as
"2.0 – 5.9 pp". Both ends were wrong: it omitted the four effects below 2.0, and its
5.9 pp was `Dropout 10%` measured against `Dropout 50%` — **two variants compared with
each other**, not a single ablation effect against the full model. The corrected range of
single-ablation effects is **0.1 – 4.1 pp**.

δ = 1.0 pp **falls inside the 0.1 – 4.1 pp band of effects the paper draws conclusions
from**, and is ten times the resolution its tables are printed at. That is the whole of
the claim: the band is a sanity check on the order of magnitude, **not** a derivation and
**not** a noise estimate. `PAPER SILENT / IMPLEMENTATION CHOICE` (Codex 2026-08-30).

⚠ **Corrected 2026-08-30 (ULIP2 Block Reviewer).** This paragraph previously read
"at least as large as the smallest one MetaFind itself reports as a finding". False: the
same table carries a 0.1 pp row. δ = 1.0 pp is unaffected; the sentence was.

**This is an anchor, not a derivation.** The paper's numbers are Table 1 test-set R@1 on
a different protocol and a different scale (roughly 10–50%) from our development metric
(roughly 71–96%). The band is used to check that δ is not absurd, not to compute it.

**Classification: `IMPLEMENTATION CHOICE`, user-approved.** It is not a `PAPER FACT` and
it is not an `INFERENCE` from the paper.

---

## What δ = 1.0 pp can and cannot detect

`OBSERVED DATA` — the ladder, protocol C, three-cell mean:

```
e10       0.8858
e25_500w  0.8555     difference 3.0 pp   → δ = 1.0 pp detects this comfortably
```

⚠ Those two runs are **not** a controlled comparison: they differ in epochs, in
`lr_horizon`, and in the working tree that produced them. The number is quoted here only
to show the scale δ has to work at.

**We do not have a measured seed-repeat spread.** The 0.00123 figure previously quoted as
a noise floor was **withdrawn on 2026-08-30**: `e25_400w` and `e25_500w` do not carry the
same checkpoint record fields, so the working tree changed between them and they were
never a repeat. Producing a real noise floor is one of the things the 4×2 paired design
exists to do.

---

## Authority

| Item | Class | Source |
|---|---|---|
| δ = 1.0 pp | `IMPLEMENTATION CHOICE`, user-approved | Kyzen, 2026-08-30 |

⚠ **This file is block evidence, not the decision ledger.** [CODEX 2026-08-30] An
untracked evidence file cannot be the authority for a research-critical threshold. It is
now entered as **`DL-033`**; the ledger wins if the two ever disagree. δ must never be
reported as paper-derived.

⚠ **The retracted band reached the ledger too.** Master's first `DL-033` also carried
`0.9–5.9 pp`, corrected in `65078f5`. Master's own note on why: *"I checked the table's
values and not the band I derived from them."* The same wrong number therefore existed in
two places at once — the reason a correction has to be chased into every artifact that
copied it, not just the one where it was found.

**Still unspecified in the stopping rule** (Codex 2026-08-30, not yet decided): the
interval method and confidence level, the maximum number of seeds, how multiplicity /
winner's curse across four LRs is handled, alpha spending for the sequential
"straddle → add a seed" step, and a per-cell degradation guardrail beside the three-cell
mean.
| three-cell selector | `IMPLEMENTATION CHOICE` | Kyzen, 2026-08-29 (R-33 item 3) |
| paired-difference stopping rule | `IMPLEMENTATION CHOICE` | Codex 2026-08-30, relayed R-33 |
| paper prints one decimal, reports no variance | `OBSERVED` | `3experiments.tex:36-46, 69-72, 94-108` |
| ablation effect band **0.1 – 4.1 pp** | `OBSERVED` | `3experiments.tex:94-108` |

⚠ **This row read `0.9–5.9 pp` until 2026-08-30.** That figure was retracted in the body
of this file on the same day, but survived here — in the one table a ledger entry copies
whole, still carrying an `OBSERVED` classification. [ULIP2 REVIEWER 2026-08-30] It is the
standard way a correction fails: the prose is fixed, the summary is not, and the summary
is what gets read. Recorded rather than quietly overwritten.
| e10 vs e25 three-cell gap 3.0 pp | `OBSERVED DATA` | `data/outputs/ladder/*/stage1_best_ckpt.json` |
