# Kyzen's three decisions, 2026-08-30

Put as one block, answered `甲 / 甲 / 甲`. Block evidence; the ledger is Master's.
Submitted for `DECISION_LEDGER.md` entry alongside `DL-033`.

---

## 1 · The first 8 runs are a MEASUREMENT round, not a SELECTION round

```
produces      the first honest seed-to-seed spread this project has,
              and a rough picture of where the four LRs sit
does not      declare a winning learning rate
```

The five things Codex left unspecified — interval method and confidence level, maximum
seeds, multiplicity / winner's curse across four LRs, alpha spending for the sequential
"straddle → add a seed" step, and a per-cell degradation guardrail — are decided **after**
the spread exists, not now.

**Why this is not a loophole:** δ and the interval method are opposites, and fixing both
in advance would repeat δ's own error from the other side.

```
δ                 declared BEFORE   it states what we WANT
interval method   chosen AFTER      it depends on what the data LOOKS LIKE
```

At n=2 per arm with no measured dispersion anywhere in this project, no interval is
trustworthy. Guessing a confidence level against a dispersion nobody has measured is not
rigour.

⚠ The corollary, which must not be quietly dropped: **this round cannot be used to pick a
learning rate.** If someone later reads a winner out of it, that is the failure this
framing exists to prevent.

## 2 · Release arm 1 first, then the remaining seven

Zero extra cost — arm 1 runs either way. The reason is that the cost estimate
(4.90 min/epoch) was measured at `preload: false` / `num_workers: 4`, and **no complete
run has ever used `--preload`**, which every line of this plan does. `stage1.py:1190` is
`workers = 0 if args.preload else 4`, so the whole data-loading path differs.

This batch is simultaneously `--preload`'s first full execution and the source of the
first honest dispersion measurement. A throughput or stability effect would land exactly
on the measurement least worth repeating.

## 3 · n15, the evaluation runner, is written NOW

**Scope approved. Execution is not** — it still needs Reviewer + Codex review, then a
separate `✅`.

It is the only route to a number comparable with the paper. Protocols `A_test_gallery`
and `B_full_gallery` have `reported: true` and **have never run**, because no executable
consumes `eval_protocols.json`; every score this project holds (e5 0.9571, e10 0.9471,
e25 0.9333 / 0.9321) is protocol C, hardcoded in `stage1.evaluate_dev_val`.

Writing it costs no GPU, so it proceeds in parallel with the sweep.
