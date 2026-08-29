# Stage 1 LR sweep — the plan, ready for Kyzen's execution `✅`

**Status: NOT AUTHORISED TO RUN.** Reviewer R-34 passed the CODE; execution needs a
separate `✅` from Kyzen. DL-028 also still needs Codex R2.

---

## Design

4 learning rates × 2 shared seeds = **8 runs**, paired (both seeds run at every LR),
execution order randomised. Codex's design, adopted over the Reviewer's region-search
alternative because that one assumed a smooth LR response and adjacent optima with no
evidence for either, and the paired form is what the stopping rule needs.

```
LRs    2.5e-4   5e-4 (the ratified value)   7.5e-4   1e-3
seeds  20260816 (repeat_index 0)   20260830 (repeat_index 1)
each   --epochs 5, cosine annealing fully within 5 (no --lr-horizon)
```

`20260830` follows the project's date convention (`split_seed 20260816`,
`dev_split_seed 20260827`). Declared here **before** the runs, so it cannot be chosen
after seeing a result.

**Protocol C only.** Protocols A and B use the sealed test split and must never appear
in a sweep — Codex 2026-08-30. The hard check for this belongs in n15, which does not
exist yet; until it does, the constraint is procedural.

---

## Execution order

Shuffled with `random.Random(20260830)` over `[(lr, seed) for lr in LRS for seed in SEEDS]`,
so the order is reproducible rather than merely arbitrary. Randomised because this machine
crashed nine times on 2026-08-29 and the fix (500 W) is a mitigation, not a repair: a
correlated failure must not land on one LR.

```
1  --lr 0.00075 --seed 20260830 --repeat-index 1 --out-dir sweep_lr/lr7.50e-4_s20260830
2  --lr 0.001   --seed 20260816 --repeat-index 0 --out-dir sweep_lr/lr1.00e-3_s20260816
3  --lr 0.0005  --seed 20260816 --repeat-index 0 --out-dir sweep_lr/lr5.00e-4_s20260816
4  --lr 0.00075 --seed 20260816 --repeat-index 0 --out-dir sweep_lr/lr7.50e-4_s20260816
5  --lr 0.001   --seed 20260830 --repeat-index 1 --out-dir sweep_lr/lr1.00e-3_s20260830
6  --lr 0.0005  --seed 20260830 --repeat-index 1 --out-dir sweep_lr/lr5.00e-4_s20260830
7  --lr 0.00025 --seed 20260830 --repeat-index 1 --out-dir sweep_lr/lr2.50e-4_s20260830
8  --lr 0.00025 --seed 20260816 --repeat-index 0 --out-dir sweep_lr/lr2.50e-4_s20260816
```

Full command shape:

```
python -m metafind.train.stage1 --phase dev --epochs 5 --preload \
  --lr <LR> --seed <SEED> --repeat-index <I> --out-dir sweep_lr/<TAG>
```

---

## Cost

`OBSERVED DATA` — from the dev_val logs, 31,985 assets, batch 64, 499 steps/epoch:

```
e25_500w   4.90 min/epoch        e10   4.71 min/epoch        e25_400w   5.51 min/epoch
->  ~25 min per 5-epoch run  ->  ~3.3 h for 8 runs
```

⚠ At 500 W. 600 W crashed the machine nine times on 2026-08-29 and is measured no faster
(561 ms/step at both). 400 W is 12% slower.

### ⚠ The estimate was measured in a different execution configuration

[ULIP2 REVIEWER 2026-08-30] Every run above was `preload: false` / `num_workers: 4`.
Verified across every checkpoint record we hold:

```
e25_500w      preload=False  num_workers=4
canonical     preload=False  num_workers=4
e10, e25_400w, DIED_e5_0542   fields absent (predate the flag)
```

**No complete run has ever used `--preload`**, and this plan uses it on every line.
`stage1.py:1190` is `workers = 0 if args.preload else 4`, so the whole data-loading path
changes — throughput could move in either direction.

That matters more than it normally would, because this batch is simultaneously the first
full `--preload` execution AND the source of this project's first honest seed-repeat
spread. A throughput or stability effect would land exactly on the measurement we least
want to repeat.

**Mitigation, at zero extra cost: run arm 1 alone, measure it, then release the other
seven.** Arm 1 has to run regardless; this only splits "approve 8" into "approve 1, then
7". Kyzen's call.

### ⚠ What this batch does NOT answer

All 8 runs are `--epochs 5`. This answers **which learning rate**, not **how many epochs**.
The epoch question is a second layer and needs its own runs.

---

## ⚠ This is a MEASUREMENT round, not a SELECTION round

[ULIP2 REVIEWER 2026-08-30] Codex listed five things the stopping rule leaves unspecified:
the interval method and confidence level, the maximum number of seeds, multiplicity /
winner's curse across four LRs, alpha spending for the sequential "straddle → add a seed"
step, and a per-cell degradation guardrail beside the three-cell mean.

**None of them can be answered yet, and answering them now would repeat the exact error
δ exists to prevent — from the other side.** All five presuppose that these 8 runs produce
a decision. At n=2 per arm, with **no measured seed-to-seed spread anywhere in this
project**, no interval is trustworthy: the stopping rule does not run on n=2.

```
δ                 declared BEFORE   because it states what we WANT
interval method   chosen AFTER      because it depends on what the data LOOKS LIKE
```

Fixing both in advance would be guessing a confidence level against a dispersion nobody
has measured. So:

```
this round PRODUCES      the first honest seed-repeat spread, and a rough
                         picture of where the four LRs sit
this round DOES NOT      declare a winning LR
```

The five items are decided once the spread exists. **Kyzen needs to agree to that framing,
not to five separate parameters.**

---

## Reading the result

```
δ    = 1.0 pp on mean R@1 over {text, image, pc}, protocol C   (Kyzen 2026-08-30)
d_s  = metric(LR_a, seed_s) - metric(LR_b, seed_s)             ← paired, same seed
interval lower > δ → real   |   upper < δ → stop   |   straddles → add a seed
```

Single-seed results are used **only** to reject NaN / divergence / OOM, never for
statistical elimination.

The seven-cell mean and every individual cell are reported alongside, as a guardrail.

---

## What this sweep also produces

The **first honest seed-repeat spread this project has**. The 0.00123 figure previously
quoted as a noise floor was withdrawn on 2026-08-30: `e25_400w` and `e25_500w` do not
carry the same checkpoint record fields, so the tree changed between them and they were
never a repeat.

⚠ If that spread comes back **wider than δ**, the rule will never return "real
improvement". That is the rule being correctly conservative. Report it to Kyzen; do not
move δ.

---

## Still open before this can run

| Item | Owner |
|---|---|
| Codex R2 on DL-028 | Codex, via Kyzen |
| Execution `✅` for the 8 runs | Kyzen |
| n15 evaluator — scope not approved, blocks Table 1 | Kyzen |
| A/B-excluded-from-sweep as a hard check in code | n15 |
