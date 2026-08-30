#!/usr/bin/env python3
"""Step 1 — UNTRAINED dev-val, through `stage1.evaluate_dev_val` itself.

WHY THIS EXISTS
---------------
Every `vs_untrained` number reported so far compares two DIFFERENT programs:
the untrained side came from `metafind/eval/run_retrieval.py` (DL-044), the
trained curves from `stage1.evaluate_dev_val`. Both implement protocol C and
both score in float64 through `normalize_for_scoring`, so they *should* agree
-- but nobody has measured it, and the headline claim is a **+0.1 pp**
difference at `lr 2.5e-4`. A gap that small may not be compared across two
tools that have never been checked against each other.

This runs the untrained model through the trained side's own evaluator, so the
comparison stops being INFERENCE.

WHAT "UNTRAINED" MEANS HERE, EXACTLY
-------------------------------------
Zero optimizer steps. No Stage 1 checkpoint is read. **It is NOT an
un-pretrained model**: the point encoder still carries ULIP-2's pretrained
PointBERT and pc_projection, and text/image still go through pretrained
OpenCLIP ViT-bigG-14. Only the two fusion towers are random, drawn from
`--seeds`. Two towers are two draws, so one seed is one sample -- hence three.

WHAT THIS SCRIPT MUST NOT DO
-----------------------------
Diagnostic only. It writes ONE json under `output/look/` and touches no
protocol artifact, no embedding cache, no checkpoint, and no trainer state.
`evaluate_dev_val` is imported and called unmodified -- if it needed changing
to run here, that would itself be the finding.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths                                    # noqa: E402
from metafind.eval.retrieval import QUERY_CONDITIONS          # noqa: E402

OUT = REPO / "output" / "look" / "diag_untrained_same_evaluator.json"

# [DL-044] the same quantity measured by the OTHER tool, for the delta.
RUN_RETRIEVAL = {
    "text":       (0.9729, 0.9593, 0.9604),
    "image":      (0.9041, 0.9184, 0.9085),
    "pc":         (0.9532, 0.9475, 0.9350),
    "text+image": (0.9974, 0.9982, 0.9974),
    "text+pc":    (0.9998, 0.9998, 0.9998),
    "image+pc":   (0.9869, 0.9851, 0.9866),
    "full":       (0.9989, 0.9998, 0.9996),
}


def main() -> int:
    seeds = [int(s) for s in (sys.argv[1:] or ["1", "2", "3"])]

    import torch
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import (build_model, evaluate_dev_val,
                                       load_protocols)

    encoding, training, hyper = load_protocols()
    # `build_model` unwraps `hyperparameters["values"]` itself
    # (stage1.py:1445), so it gets the WHOLE artifact; `values` here is
    # only for reading batch_size.
    values = hyper["values"]
    dev_val = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    print(f"dev_val {len(dev_val):,} assets · seeds {seeds}", flush=True)

    # Built ONCE: it is loaded from a file and draws nothing from the RNG, so
    # its position relative to the seed does not matter. Rebuilding it per seed
    # would only add three ViT-bigG loads to the wall clock.
    backbone = ULIPBackbone(BackboneConfig(
        device="cuda", train_scope="point_encoder_and_fuser"))

    runs = {}
    for seed in seeds:
        # Seeded immediately before `build_model`, because that call is what
        # draws the two fusion towers out of the global torch RNG.
        torch.manual_seed(seed)
        model, _loss = build_model(encoding, training, hyper)
        model.to("cuda")
        t0 = time.time()
        scores = evaluate_dev_val(backbone, model, dev_val,
                                  encoding["image_aggregation"], "cuda",
                                  values["batch_size"])
        runs[seed] = scores
        print(f"  seed {seed}: mean_R@1 {scores['mean_R@1']:.6f}  "
              f"mean_R@5 {scores['mean_R@5']:.6f}  "
              f"gallery {scores['n_gallery']:,}  ({time.time()-t0:.0f}s)",
              flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'condition':>12} " + "".join(f"{'seed '+str(s):>12}" for s in seeds)
          + f"{'this mean':>12}{'run_retr mean':>15}{'delta pp':>10}")
    table = {}
    for c in QUERY_CONDITIONS:
        mine = [runs[s][c]["R@1"] for s in seeds]
        m = sum(mine) / len(mine)
        o = sum(RUN_RETRIEVAL[c]) / 3
        table[c] = {"per_seed_R@1": mine, "mean_R@1": m,
                    "run_retrieval_mean_R@1": o, "delta_pp": (m - o) * 100}
        print(f"{c:>12} " + "".join(f"{v:12.4f}" for v in mine)
              + f"{m:12.4f}{o:15.4f}{(m-o)*100:+10.2f}")

    mm = sum(runs[s]["mean_R@1"] for s in seeds) / len(seeds)
    om = sum(sum(v) / 3 for v in RUN_RETRIEVAL.values()) / 7
    print(f"{'MEAN OF 7':>12} " + "".join(f"{runs[s]['mean_R@1']:12.4f}" for s in seeds)
          + f"{mm:12.4f}{om:15.4f}{(mm-om)*100:+10.2f}")
    spread = max(runs[s]["mean_R@1"] for s in seeds) - min(runs[s]["mean_R@1"] for s in seeds)
    print(f"\nseed spread on mean_R@1: {spread*100:.2f} pp")
    print("⚠ any trained-vs-untrained difference smaller than this spread is "
          "not a difference.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": "UNTRAINED (0 optimizer steps) through stage1.evaluate_dev_val",
        "caveat": ("NOT un-pretrained: PointBERT and OpenCLIP are still ULIP-2 / "
                   "OpenCLIP pretrained. Only the two fusion towers are random."),
        "seeds": seeds, "n_dev_val": len(dev_val),
        "per_seed": {str(s): runs[s] for s in seeds},
        "per_condition": table,
        "mean_of_seven": {"this_tool": mm, "run_retrieval": om,
                          "delta_pp": (mm - om) * 100},
        "seed_spread_mean_R@1_pp": spread * 100,
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
