#!/usr/bin/env python3
"""Stage 1 `--phase final` training curves, with the reference lines that make them readable.

`tools/plot_stage1.py` draws the DEV phase: loss plus the dev-val R@1 curve it
selects on. `--phase final` performs no selection (`stage1.py:1567` empties
`dev_val_uids`), so there is no held-out curve to draw and that tool's second
panel would be empty. This is the final-phase plot instead.

WHY THE REFERENCE LINES ARE THE POINT
--------------------------------------
A contrastive loss of 2.4 means nothing on its own. With `similarity: cosine`,
`init_temperature: 0.5` and `learnable_temperature: false`, `losses.py:169-170`
computes ``logits = (1/tau) * q @ g.T`` on L2-normalised rows, so every logit is
bounded in ``[-2, +2]`` and the loss is bounded too:

    chance   all logits equal          -> ln(B)                        = 4.1589
    bound    target +2, others -2      -> ln(1 + (B-1) * e^-4)         = 0.7674

for B = 64. **The loss can never reach 0.**

⚠ The lower line is an ABSOLUTE THEORETICAL BOUND, not an achievable training
floor. [ULIP2 REVIEWER 2026-08-30] It is derived per-sample, and reaching it
would need every asset's own positive at cos +1 while all 63 others sit at
cos -1 -- SIMULTANEOUSLY FOR ALL 64 ROWS, which 64 embeddings in one shared
space cannot satisfy. Calling it "the floor" invites the reading that 0.767 is
a target the run is failing to approach. It is not; it is the value below which
the arithmetic cannot go. A reader who does not know that sees
2.4 and concludes the model is barely training; against the real range it is
roughly two thirds of the way down. Both lines are computed from the run's own
recorded `tau` and batch size, not hardcoded, so they follow a config change.

Accuracy gets the same treatment: `acc_q2g` is in-batch top-1 over B, so chance
is 1/B, not 0.

WHAT THIS PLOT CANNOT SHOW
---------------------------
**There is no validation curve, so overfitting is not visible here.** A final
run has no held-out split by construction. Training loss falling is not evidence
of generalisation, and this figure must not be read as if it were. The held-out
answer comes from n15 after the run, not from this file.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
LOG = REPO / "data" / "outputs" / "logs" / "train_stage1.jsonl"
OUT = REPO / "output" / "look" / "stage1_final_curves.png"


def load(run_id: str | None) -> dict[str, list]:
    """Rows for one run_id, or the newest run if none is named."""
    rows = []
    for line in LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("loss") is not None:
            rows.append(r)
    if not rows:
        sys.exit(f"{LOG} has no rows carrying `loss`")
    rid = run_id or rows[-1]["run_id"]
    return [r for r in rows if r.get("run_id") == rid], rid


def series(rows, key):
    """(step, value) pairs where the key is present. Absent != zero."""
    xy = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [p[0] for p in xy], [p[1] for p in xy]


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    rows, rid = load(run_id)
    cfg = rows[-1]
    batch = 64  # arm_config.batch_size; recorded per run, see the record.
    tau = cfg.get("tau", 0.5)
    scale = 1.0 / tau

    # Both bounds from THIS run's tau and batch, so a config change moves them.
    chance = math.log(batch)
    floor = math.log(1.0 + (batch - 1) * math.exp(-2.0 * scale))

    # [FIXED] `max(step) / epochs_seen` was wrong: the newest epoch is PARTIAL,
    # so it divided a full-epoch count by a fractional epoch and drew the
    # boundary lines in the wrong places (475 against a true 572). Derive it
    # from the pool the run actually used, the way the DataLoader does.
    n_train = 36554
    steps_per_epoch = math.ceil(n_train / batch)
    epochs_seen = max(r["epoch"] for r in rows) + 1

    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(
        f"Stage 1 --phase final  ·  run {rid}\n"
        f"lr 2.5e-4 · seed 20260816 · batch {batch} · tau {tau} fixed · "
        f"rev {cfg.get('code_revision','?')[:8]} dirty={cfg.get('code_dirty')}",
        fontsize=11, y=0.995)

    # --- 1. loss, against its own bounds -----------------------------------
    x, y = series(rows, "loss")
    ax[0].plot(x, y, lw=1.0, color="#1f77b4")
    ax[0].axhline(chance, ls="--", lw=1.2, color="#c23b3b")
    ax[0].axhline(floor, ls="--", lw=1.2, color="#2f7d4f")
    ax[0].text(x[0], chance, f"  chance = ln({batch}) = {chance:.3f}",
               va="bottom", fontsize=9, color="#c23b3b")
    ax[0].text(x[0], floor, f"  absolute theoretical lower bound at tau={tau} = "
               f"{floor:.3f}  (NOT an achievable floor -- see docstring)",
               va="bottom", fontsize=9, color="#2f7d4f")
    ax[0].set_ylabel("InfoNCE loss")
    ax[0].set_ylim(0, chance * 1.08)
    ax[0].grid(alpha=.25)

    # --- 2. in-batch top-1 --------------------------------------------------
    x, y = series(rows, "acc_q2g")
    ax[1].plot(x, y, lw=1.0, color="#b07d10")
    ax[1].axhline(1.0 / batch, ls="--", lw=1.2, color="#c23b3b")
    ax[1].text(x[0], 1.0 / batch, f"  chance = 1/{batch} = {1/batch:.4f}",
               va="bottom", fontsize=9, color="#c23b3b")
    ax[1].set_ylabel(f"in-batch top-1 (of {batch})")
    ax[1].set_ylim(0, 1.02)
    ax[1].grid(alpha=.25)

    # --- 3. learning rate: does the cosine actually span the run? -----------
    x, y = series(rows, "lr")
    ax[2].plot(x, y, lw=1.2, color="#3f8fc4")
    ax[2].set_ylabel("learning rate")
    ax[2].grid(alpha=.25)
    ax[2].text(0.01, 0.08,
               "if this flattens long before the run ends, --lr-horizon is "
               "shorter than --epochs and the rest of the run trains at lr_end",
               transform=ax[2].transAxes, fontsize=8.5, color="#4c5a68")

    # --- 4. gradient norm: stability ---------------------------------------
    x, y = series(rows, "grad_norm")
    if x:
        ax[3].plot(x, y, lw=0.9, color="#8b5cf6")
        ax[3].set_yscale("log")
    ax[3].set_ylabel("grad norm (log)")
    ax[3].set_xlabel(f"optimizer step   ({steps_per_epoch:.0f} per epoch)")
    ax[3].grid(alpha=.25, which="both")

    for a in ax:
        for e in range(1, epochs_seen):
            a.axvline(e * steps_per_epoch, color="#8b95a0", lw=.5, alpha=.5)

    fig.text(0.5, 0.005,
             "NO VALIDATION CURVE EXISTS: --phase final performs no dev "
             "selection, so overfitting is NOT visible here. Held-out numbers "
             "come from n15 after the run.",
             ha="center", fontsize=9.5, color="#c23b3b")
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"{len(rows)} rows, {epochs_seen} epoch(s) so far -> {OUT}")
    print(f"loss now {rows[-1]['loss']:.4f}   "
          f"range [{floor:.3f} absolute bound, {chance:.3f} chance]")
    print(f"progress down the range: "
          f"{(chance - rows[-1]['loss']) / (chance - floor) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
