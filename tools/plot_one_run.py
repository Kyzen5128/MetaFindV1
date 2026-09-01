#!/usr/bin/env python3
"""One Stage 1 run's curves, drawn against the bounds that make them readable.

`plot_stage1.py` overlays every run in the log, which is right for comparing
arms and wrong for reading one. This draws a single `run_id`.

WHERE EACH PIECE COMES FROM, AND WHY IT IS NOT ALL ONE PLACE
------------------------------------------------------------
`train_stage1.jsonl` carries only per-step scalars -- `loss`, `acc_q2g`, `lr`,
`grad_norm`, `tau`, `step`, `epoch` -- plus `arm_config_hash`. It does NOT carry
the arm itself, and the per-epoch dev-val score is printed to stdout rather than
logged as a row. A first version of this plot read the arm from the jsonl and
titled the figure `train_scope = ?  lr None  None epochs`, and left the dev-val
panel empty. So:

    per-step curves   train_stage1.jsonl, filtered by run_id
    arm (scope, lr, epochs)  the checkpoint record's `arm_config`, via --ckpt
    dev-val R@1 per epoch    parsed from the run's stdout log, via --train-log

Each is optional; whatever is missing is stated on the figure instead of being
guessed at or silently dropped.

THE REFERENCE LINES ARE THE POINT
---------------------------------
A contrastive loss of 2.3 means nothing on its own. Against its own bounds it
means the objective is finished:

    chance, all logits equal        ln(B)                        = 4.1589
    floor, negatives at cosine 0    ln(e^(1/tau) + B-1) - 1/tau  = 2.2540
    floor, geometric minimum        using cos = -1/(B-1)         = 2.2257

`loss_anatomy.py` measured the mean negative cosine at 0.0007, so the middle
line is the operative one: the negatives really are orthogonal, and with tau
pinned at 0.5 the softmax cannot sharpen past it however well the model ranks.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
LOG = REPO / "data" / "outputs" / "logs" / "train_stage1.jsonl"
DEV_RE = re.compile(r"epoch (\d+) dev-val: mean R@1 ([\d.]+)\s+mean R@5 ([\d.]+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--ckpt", type=pathlib.Path, default=None,
                    help="stage1_best_ckpt.json -- supplies train_scope, lr and "
                         "epochs, which the per-step rows do not carry")
    ap.add_argument("--train-log", type=pathlib.Path, default=None,
                    help="the run's stdout log -- supplies the per-epoch dev-val "
                         "R@1, which is printed rather than logged as a row")
    args = ap.parse_args()
    out = args.out or REPO / "output" / "look" / f"run_{args.run_id[:16]}.png"

    rows = []
    for line in LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("run_id") == args.run_id:
            rows.append(r)
    if not rows:
        sys.exit(f"no rows for run_id {args.run_id}")
    cfg = rows[-1]

    arm, missing = {}, []
    if args.ckpt and args.ckpt.exists():
        arm = json.loads(args.ckpt.read_text()).get("arm_config", {})
    else:
        missing.append("arm (pass --ckpt)")
    B = arm.get("batch_size", 64)
    tau = arm.get("init_temperature", cfg.get("tau", 0.5))
    scope = arm.get("train_scope") or "not recorded"
    lr0 = arm.get("learning_rate")
    eps = arm.get("epochs")

    dev = []
    if args.train_log and args.train_log.exists():
        for m in DEV_RE.finditer(args.train_log.read_text()):
            dev.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    else:
        missing.append("dev-val (pass --train-log)")

    scale = 1.0 / tau
    chance = math.log(B)
    floor0 = math.log(math.e ** scale + (B - 1)) - scale
    floorg = math.log(math.e ** scale + (B - 1) * math.e ** (-scale / (B - 1))) - scale

    def series(key):
        xy = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
        return [p[0] for p in xy], [p[1] for p in xy]

    n_ep = max(r["epoch"] for r in rows) + 1
    firsts = {}
    for r in rows:
        firsts.setdefault(r["epoch"], r["step"])
    spe = min(firsts[e] / e for e in firsts if e > 0) if len(firsts) > 1 else float("nan")

    fig, ax = plt.subplots(4, 1, figsize=(11, 13.5), sharex=False)
    fig.suptitle(
        f"Stage 1  ·  train_scope = {scope}  ·  run {args.run_id}\n"
        f"lr {lr0} · batch {B} · tau {tau} fixed · {eps} epochs · "
        f"seed {cfg.get('seed')} · rev {cfg.get('code_revision','?')[:8]} "
        f"dirty={cfg.get('code_dirty')}", fontsize=11, y=0.995)

    x, y = series("loss")
    ax[0].plot(x, y, lw=1.0, color="#1f77b4")
    for val, col, lab in (
            (chance, "#c23b3b", f"chance = ln({B}) = {chance:.4f}"),
            (floor0, "#2f7d4f", f"floor with negatives at cos 0 = {floor0:.4f}"
                                "   (measured mean negative: 0.0007)"),
            (floorg, "#7a5cbf", f"geometric floor = {floorg:.4f}")):
        ax[0].axhline(val, ls="--", lw=1.2, color=col)
    ax[0].legend(["loss", f"chance {chance:.3f}",
                  f"floor, negatives at cos 0 = {floor0:.4f}  (measured: 0.0007)",
                  f"geometric floor {floorg:.4f}"], fontsize=8.5, loc="upper right")
    ax[0].set_ylabel("InfoNCE loss")
    ax[0].set_ylim(floorg - 0.05, chance * 1.03)
    ax[0].set_xlabel("optimizer step")
    ax[0].grid(alpha=.25)

    x, y = series("acc_q2g")
    ax[1].plot(x, y, lw=1.0, color="#b07d10")
    ax[1].axhline(1.0 / B, ls="--", lw=1.2, color="#c23b3b")
    ax[1].text(x[0], 1.0 / B, f"  chance = 1/{B} = {1/B:.4f}", va="bottom",
               fontsize=9, color="#c23b3b")
    ax[1].set_ylabel(f"in-batch top-1 (of {B})")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xlabel("optimizer step")
    ax[1].grid(alpha=.25)

    if dev:
        e = [d[0] for d in dev]
        ax[2].plot(e, [d[1] for d in dev], "o-", lw=1.6, ms=6, color="#2f7d4f",
                   label="mean R@1")
        ax[2].plot(e, [d[2] for d in dev], "s--", lw=1.2, ms=5, color="#8b95a0",
                   label="mean R@5")
        best = max(dev, key=lambda d: d[1])
        ax[2].annotate(f"best epoch {best[0]}: {best[1]:.4f}",
                       (best[0], best[1]), textcoords="offset points",
                       xytext=(-104, 6), fontsize=9, color="#2f7d4f")
        ax[2].set_ylim(min(d[1] for d in dev) - 0.004, 1.001)
        ax[2].legend(fontsize=9, loc="lower right")
        ax[2].set_xticks(e)
    else:
        ax[2].text(0.5, 0.5, "dev-val not available: pass --train-log",
                   ha="center", transform=ax[2].transAxes, color="#8b95a0")
    ax[2].set_ylabel("dev-val R@1 / R@5  (4,569)")
    ax[2].set_xlabel("epoch")
    ax[2].grid(alpha=.25)

    x, y = series("lr")
    ax[3].plot(x, y, lw=1.3, color="#3f8fc4")
    xg, yg = series("grad_norm")
    if xg:
        a2 = ax[3].twinx()
        a2.plot(xg, yg, lw=0.8, color="#8b5cf6", alpha=.65)
        a2.set_yscale("log")
        a2.set_ylabel("grad norm (log)", color="#8b5cf6")
    ax[3].set_ylabel("learning rate", color="#3f8fc4")
    ax[3].set_xlabel(f"optimizer step   ({spe:.0f} per epoch)")
    ax[3].grid(alpha=.25)

    for a in (ax[0], ax[1], ax[3]):
        for e in range(1, n_ep):
            a.axvline(e * spe, color="#8b95a0", lw=.5, alpha=.5)

    lo = min(series("loss")[1])
    note = (f"lowest loss {lo:.4f} — {lo - floorg:.4f} above the geometric floor, "
            f"{lo - floor0:.4f} above the measured-negatives floor. With tau pinned "
            f"at {tau} the softmax cannot sharpen further, so this is the objective "
            f"finishing, not training stalling.")
    if missing:
        note += "  [not shown: " + "; ".join(missing) + "]"
    fig.text(0.5, 0.006, note, ha="center", fontsize=9.5, color="#4c5a68", wrap=True)
    fig.tight_layout(rect=[0, 0.028, 1, 0.982])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"{len(rows)} rows, {n_ep} epochs, {len(dev)} dev-val points -> {out}")
    print(f"loss {rows[0].get('loss'):.4f} -> {rows[-1].get('loss'):.4f}, lowest {lo:.4f}"
          f"   floors: geometric {floorg:.4f}, at-zero {floor0:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
