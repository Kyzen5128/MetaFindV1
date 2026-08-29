#!/usr/bin/env python3
"""Draw the Stage 1 loss curve and the dev-val R@1 curve, one PNG.

Written because the numbers were only readable as jsonl. The file is what gets
plotted, and until now nobody could look at it without running a parser.

Runs are separated by `run_id` where it exists. For the six pre-2026-08-29 runs
it does not, so they fall back to splitting on `step` going backwards -- the
only separator those files have, which is exactly why `run_id` was added.

Smoke rows are dropped by `n_gallery != 4569`: a 1,500-item gallery inflates
R@1 mechanically, and the two highest numbers in the dev-val log are smoke rows
written with the same field names as the real ones. Plotting them together
draws a dip that never happened.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False

# Run as a plain script (`python tools/plot_stage1.py`), not `-m`, so the repo
# root is not on the path. This is the same ModuleNotFoundError that killed the
# 06:57:30 systemd-run launch in 19 ms; a tool meant to be run by hand should
# not require the caller to know that.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from metafind import paths

DEV_VAL_GALLERY = 4569
STEPS_PER_EPOCH = 499     # 31,985 train assets / batch 64, drop_last


def load(path: pathlib.Path) -> list[list[dict]]:
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for r in rows:
        new = (cur and r.get("run_id") != cur[-1].get("run_id")) or \
              (cur and r["step"] < cur[-1]["step"])
        if new:
            runs.append(cur); cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def describe(run: list[dict], live_ids: set[str]) -> str | None:
    """Label for a run worth drawing, or None to skip it.

    [KYZEN 2026-08-29] Two rounds of trimming. First the chart drew all seven
    runs found in the logs under meaningless names (`舊 run 1`..`舊 run 6`);
    four were the truncated remains of machine crashes and two were 1-epoch
    smoke fragments. Then, with those named honestly, he cut further:
    一崩掉的不用寫啦 寫跑成功的就好.

    So the rule is now: the run in progress, plus runs that REACHED THEIR LAST
    STEP. A crashed run's curve is a prefix of a successful one -- it adds a
    line that ends in the middle of the chart and says nothing a completed run
    does not already say.

    Labels are English because the fonts on this box render it without
    surprises; the Chinese ones needed an explicit CJK family and still look
    wrong wherever this PNG is opened next.
    """
    epochs = max(r["epoch"] for r in run) + 1
    steps = run[-1]["step"]
    start = time.strftime("%H:%M", time.localtime(run[0]["ts"]))
    if run[0].get("run_id") in live_ids:
        return f"{start} running now - step {steps}"
    if epochs < 2:
        return None
    # STEPS_PER_EPOCH*epochs is what separates the 05:20 run the machine killed
    # at step 2320 from the 06:17 run that finished at 2480; both reached epoch
    # index 4, so epoch count alone cannot tell them apart.
    if steps < epochs * STEPS_PER_EPOCH - STEPS_PER_EPOCH // 8:
        return None
    return f"{start} completed - {epochs} epochs"


def main() -> int:
    L = paths.LOGS
    tr = load(L / "train_stage1.jsonl") + load(L / "train_stage1.pre_runid.jsonl")
    dv = load(L / "train_stage1_dev_val.jsonl") + load(L / "train_stage1_dev_val.pre_runid.jsonl")
    if not tr:
        print("no training rows yet", file=sys.stderr); return 1

    # A run is "live" if it is the newest one and still growing. Everything with
    # a run_id was written by the current code; the newest of those is the one
    # in progress.
    # The NEWEST run, by timestamp -- not `tr[0]`, which is whichever run happens
    # to sit first in the file. That bug labelled a finished 10-epoch run from
    # 08:17 as "running now" while the actual live run was at epoch 16, and the
    # chart looked entirely reasonable while doing it.
    newest = max((r for run in tr for r in run), key=lambda r: r["ts"], default=None)
    live_ids = {newest["run_id"]} if newest and newest.get("run_id") else set()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8))

    drawn = 0
    # Older runs first, the live one last: the curves lie almost on top of each
    # other (that overlap is the reproducibility evidence), so whichever is
    # drawn last is the only one visible.
    for run in sorted(tr, key=lambda r: r[0]["ts"]):
        name = describe(run, live_ids)
        if name is None:
            continue
        live = run[0].get("run_id") in live_ids
        a1.plot([r["step"] for r in run], [r["loss"] for r in run],
                lw=2.6 if live else 1.2, alpha=1.0 if live else 0.55,
                zorder=3 if live else 2, label=name)
        drawn += 1
    a1.set_title("Stage 1 training loss")
    a1.set_xlabel("step"); a1.set_ylabel("loss"); a1.grid(alpha=.3); a1.legend(fontsize=9)

    # A dev-val run inherits its parent's fate. On its own it cannot be judged:
    # the 05:20 run the machine killed mid-epoch-4 had already WRITTEN four
    # dev-val rows, so by dev-val's own step count it looks complete -- and it
    # was drawn as "05:25 completed - 4 epochs" until this pairing was added.
    # Each dev-val run is matched to the loss run it belongs to by start time
    # (they begin within a minute of each other) and kept only if that one was.
    kept_starts = [r[0]["ts"] for r in tr if describe(r, live_ids) is not None]

    any_dv = False
    for run in sorted(dv, key=lambda r: r[0]["ts"]):
        pts = [r for r in run if r.get("n_gallery") == DEV_VAL_GALLERY]
        if len(pts) < 2:
            continue
        if not any(0 <= pts[0]["ts"] - t <= 1800 for t in kept_starts):
            continue
        name = describe(pts, live_ids)
        if name is None:
            continue
        any_dv = True
        live = pts[0].get("run_id") in live_ids
        a2.plot([r["epoch"] + 1 for r in pts], [r["mean_R@1"] for r in pts], "o-",
                lw=2.6 if live else 1.2, alpha=1.0 if live else 0.55,
                zorder=3 if live else 2, label=name)
    a2.set_title("dev-val retrieval, mean R@1 over 7 conditions (higher is better)")
    a2.set_xlabel("epoch"); a2.set_ylabel("mean R@1"); a2.grid(alpha=.3)
    if any_dv:
        a2.legend(fontsize=9)
    else:
        a2.text(.5, .5, "no epoch finished yet", ha="center", transform=a2.transAxes)

    out = paths.REPO / "output" / "look" / "stage1_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"{out.relative_to(paths.REPO)}  drew {drawn} loss curves"
          f" (read {len(tr)}; the rest are smoke checks or crashed runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
