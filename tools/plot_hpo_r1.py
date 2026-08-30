#!/usr/bin/env python3
"""Round-1 LR sweep: dev-val retrieval per epoch, per condition, against the untrained baseline.

WHY THE UNTRAINED LINE IS ON EVERY PANEL
-----------------------------------------
The seven-condition mean hides the result. At `lr 2.5e-4` the mean moves -0.1 pp
against no training at all -- which reads as "training does nothing". It is not:
`image` gains 7.0 pp, `pc` gains 3.0 pp, and `text` loses 11.1 pp. A mean over
four saturated cells and three moving ones in opposite directions is not a
summary, it is a cancellation. So every condition is drawn separately and every
panel carries its own untrained reference.

WHERE THE UNTRAINED NUMBERS COME FROM, AND THE CAVEAT THAT GOES WITH THEM
--------------------------------------------------------------------------
`DL-044`, three initialisation seeds, protocol C (query and gallery both
dev_val, 4,569). ⚠ They were produced by `metafind/eval/run_retrieval.py`,
while the curves here come from `stage1.evaluate_dev_val`. Both implement
protocol C and both score in float64 through `normalize_for_scoring`, but
**the two tools have never been measured against each other**. The comparison
is therefore INFERENCE, not OBSERVED DATA, and the band -- not a line -- is
drawn from the three seeds so the reader can see the spread it has to beat.

The four saturated conditions share one panel deliberately: they are flat at
0.99+ everywhere, they carry no selection signal, and putting them on their own
axes would give four uninformative cells the same visual weight as the three
that decide the answer.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
LOG = REPO / "data" / "outputs" / "logs" / "train_stage1_dev_val.jsonl"
OUT = REPO / "output" / "look" / "hpo_r1_lr_sweep.png"

# The commit all four arms must share. A row from any other revision is a
# different program and is dropped rather than plotted beside these.
REV = "7734f06"

# [DL-044] protocol C, three init seeds, UNTRAINED (Stage 1 never ran; the point
# encoder still carries ULIP-2's pretrained PointBERT and text/image still go
# through pretrained OpenCLIP).
UNTRAINED = {
    "text":       (0.9729, 0.9593, 0.9604),
    "image":      (0.9041, 0.9184, 0.9085),
    "pc":         (0.9532, 0.9475, 0.9350),
    "text+image": (0.9974, 0.9982, 0.9974),
    "text+pc":    (0.9998, 0.9998, 0.9998),
    "image+pc":   (0.9869, 0.9851, 0.9866),
    "full":       (0.9989, 0.9998, 0.9996),
}
MOVING = ["text", "image", "pc"]
SATURATED = ["text+image", "text+pc", "image+pc", "full"]
COLOURS = {"2.5e-4": "#2f7d4f", "5.0e-4": "#3f8fc4",
           "7.5e-4": "#b07d10", "1.0e-3": "#c23b3b"}


def load():
    """Runs at REV, in launch order, paired with the LR the driver used.

    Order rather than a recorded lr: `train_stage1_dev_val.jsonl` carries
    `arm_config_hash` but not the rate itself, and the driver ran the four
    sequentially. Stated because it is an assumption a re-ordered re-run breaks.
    """
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("code_revision", "").startswith(REV)]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["run_id"], []).append(r)
    lrs = ["2.5e-4", "5.0e-4", "7.5e-4", "1.0e-3"]
    return [(lr, sorted(v, key=lambda r: r["epoch"]))
            for lr, v in zip(lrs, by.values())]


def band(ax, cond_or_list):
    """The untrained range as a band, not a line: three seeds, and the spread
    is the thing an arm has to clear before its difference means anything."""
    if isinstance(cond_or_list, str):
        vals = UNTRAINED[cond_or_list]
    else:
        vals = [sum(UNTRAINED[c][i] for c in cond_or_list) / len(cond_or_list)
                for i in range(3)]
    lo, hi = min(vals), max(vals)
    ax.axhspan(lo, hi, color="#8b95a0", alpha=.28, zorder=0)
    ax.axhline(sum(vals) / 3, color="#4c5a68", ls="--", lw=1.1, zorder=1)
    return lo, hi


def main() -> int:
    arms = load()
    if not arms:
        sys.exit(f"no rows at revision {REV} in {LOG}")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()
    fig.suptitle(
        "Stage 1 round-1 LR sweep · phase=dev · epochs 10 · lr_horizon 250 · "
        "seed 20260816 · batch 64 · protocol C (4,569)\n"
        "grey band = UNTRAINED, three init seeds (DL-044). "
        "Every arm at every epoch that sits BELOW the band is worse than not training at all.",
        fontsize=11.5, y=0.985)

    panels = [("mean of all seven", None)] + [(c, c) for c in MOVING]
    for i, (title, cond) in enumerate(panels):
        a = ax[i]
        band(a, list(UNTRAINED) if cond is None else cond)
        for lr, rows in arms:
            xs = [r["epoch"] for r in rows]
            key = "mean_R@1" if cond is None else f"cond_{cond}_R@1"
            ys = [r.get(key) for r in rows]
            a.plot(xs, ys, "-o", ms=3.5, lw=1.6, color=COLOURS[lr], label=f"lr {lr}")
        a.set_title(title + ("   ← the selection metric" if cond is None else ""),
                    fontsize=11, fontweight="bold")
        a.set_xlabel("epoch"); a.set_ylabel("R@1")
        a.grid(alpha=.25); a.set_xlim(-0.3, 9.3)
        if i == 0:
            a.legend(fontsize=9, loc="lower left")

    # --- the four saturated cells, together, to show they carry no signal ----
    a = ax[4]
    for lr, rows in arms:
        for c in SATURATED:
            a.plot([r["epoch"] for r in rows],
                   [r.get(f"cond_{c}_R@1") for r in rows],
                   "-", lw=1.1, alpha=.75, color=COLOURS[lr])
    a.set_title("the four saturated conditions (all arms)", fontsize=11, fontweight="bold")
    a.set_xlabel("epoch"); a.set_ylabel("R@1"); a.set_ylim(0.96, 1.005)
    a.grid(alpha=.25); a.set_xlim(-0.3, 9.3)
    a.text(0.03, 0.08, "text+image · text+pc · image+pc · full\n"
                       "never leave 0.97–1.00 at any rate or epoch.\n"
                       "Four of the seven cells the mean averages\n"
                       "cannot separate these models.",
           transform=a.transAxes, fontsize=9, color="#4c5a68")

    # --- best epoch per arm, against untrained, per condition ----------------
    a = ax[5]
    a.axis("off")
    lines = ["BEST EPOCH PER ARM, vs untrained (pp)", ""]
    lines.append(f"{'arm':>9} {'ep':>3} {'mean':>8} {'text':>8} {'image':>8} {'pc':>8}")
    for lr, rows in arms:
        b = max(rows, key=lambda r: (r["mean_R@1"], r["mean_R@5"]))
        u = {c: sum(UNTRAINED[c]) / 3 for c in MOVING}
        um = sum(sum(v) / 3 for v in UNTRAINED.values()) / 7
        lines.append(
            f"{lr:>9} {b['epoch']:3d} {(b['mean_R@1']-um)*100:+8.1f} "
            + " ".join(f"{(b.get(f'cond_{c}_R@1', 0)-u[c])*100:+8.1f}" for c in MOVING))
    lines += ["", "Every arm's mean is at or below untrained.",
              "image and pc GAIN; text LOSES more than both gain.",
              "", "⚠ untrained came from run_retrieval.py, the curves from",
              "   stage1.evaluate_dev_val. Same protocol C, same float64,",
              "   but the two tools have never been measured against each",
              "   other. This comparison is INFERENCE, not OBSERVED DATA."]
    a.text(0.0, 0.98, "\n".join(lines), transform=a.transAxes, va="top",
           family="monospace", fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"{len(arms)} arm(s) at rev {REV} -> {OUT}")
    for lr, rows in arms:
        b = max(rows, key=lambda r: (r["mean_R@1"], r["mean_R@5"]))
        print(f"  lr {lr}: {len(rows)} epochs, best ep {b['epoch']} "
              f"mean_R@1 {b['mean_R@1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
