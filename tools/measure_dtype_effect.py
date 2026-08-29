"""Does scoring dev_val in float64 instead of float32 move any reported number?

# READ-ONLY MEASUREMENT. Trains nothing, writes no checkpoint, touches no
# artifact under data/outputs except one small JSON report it creates.

Why this exists
---------------
`run_retrieval` (n15) now scores in float64, because at production shape the
collapse diagnostic `tie_count` moved with the caller's block size in 7-9 of 12
float32 trials and in 0 of 12 float64 trials. `stage1.evaluate_dev_val` still
scores in float32, and two evaluators with different numerical semantics is the
thing we already refused once.

Changing it looks research-significant, because dev_val R@1 is the metric that
SELECTS checkpoints and the ladder numbers (e5 0.9571, e10 0.9471, e25 0.9333 /
0.9321) were all produced in float32. But "the numbers would become
incomparable" is currently an assumption, not a measurement -- and it is cheap
to settle:

    encode dev_val ONCE with an existing checkpoint
    score the same embeddings twice, float32 and float64
    compare all seven conditions, exactly

If nothing moves, this is a precision fix and nobody has to decide anything. If
something moves, the decision comes with the actual difference attached instead
of a guess. [ULIP2 REVIEWER 2026-08-30]

⚠ This answers the question for ONE checkpoint. It cannot prove the two dtypes
agree for every model; it can only find a disagreement if one is there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from metafind import paths, runlog


def score(queries: dict, gallery: np.ndarray, dtype) -> dict:
    from metafind.eval.retrieval import recall_at_k

    g = gallery.astype(dtype)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    targets = np.arange(g.shape[0])
    out = {}
    for cond, q in queries.items():
        q = q.astype(dtype)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        out[cond] = recall_at_k(q @ g.T, targets, ks=(1, 5))
    out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in queries]))
    out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in queries]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/outputs/ladder/e25_500w/stage1_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--baseline-record",
                    default="data/outputs/ladder/e25_500w/stage1_best_ckpt.json",
                    help="the record of the run that PRODUCED the checkpoint. "
                         "This is the baseline that matters: the question is "
                         "whether a number already on record would move, not "
                         "whether two new computations agree with each other.")
    ap.add_argument("--out", default="output/look/dtype_effect.json")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from metafind.eval.retrieval import QUERY_CONDITIONS, condition_mask
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import (Stage1Dataset, build_model, collate,
                                       load_protocols, load_stage1_checkpoint,
                                       modules_in_eval)

    encoding, training, hyperparameters = load_protocols()
    uids = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    print(f"dev_val: {len(uids):,} assets", flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model, loss_fn = build_model(encoding, training, hyperparameters)
    model.to(args.device)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(args.ckpt))

    loader = DataLoader(Stage1Dataset(uids, encoding["image_aggregation"]),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4, drop_last=False)
    gal, per_cond = [], {c: [] for c in QUERY_CONDITIONS}
    with modules_in_eval(model, getattr(backbone, "model", None)), torch.no_grad():
        for i, batch in enumerate(loader):
            e = {"text": batch["text"].to(args.device),
                 "image": batch["image"].to(args.device),
                 "pc": backbone.encode_pc(batch["pc"].to(args.device))}
            n = e["text"].size(0)
            gal.append(model.gallery(e).float().cpu())
            for cond in QUERY_CONDITIONS:
                per_cond[cond].append(
                    model.query(e, present=condition_mask(cond, n).to(args.device)
                                ).float().cpu())
            if i % 10 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)

    # ONE encode, two scorings. Anything else and a difference could be the
    # encoder rather than the arithmetic.
    gallery = torch.cat(gal).numpy()
    queries = {c: torch.cat(v).numpy() for c, v in per_cond.items()}
    f32 = score(queries, gallery, np.float32)
    f64 = score(queries, gallery, np.float64)

    print(f"\n{'condition':12s} {'f32 R@1':>9s} {'f64 R@1':>9s} {'delta':>10s}"
          f" {'f32 R@5':>9s} {'f64 R@5':>9s}")
    diffs = {}
    for c in list(QUERY_CONDITIONS) + ["mean_R@1", "mean_R@5"]:
        if c in QUERY_CONDITIONS:
            a, b = f32[c]["R@1"], f64[c]["R@1"]
            a5, b5 = f32[c]["R@5"], f64[c]["R@5"]
            print(f"{c:12s} {a:9.6f} {b:9.6f} {b - a:+10.2e} {a5:9.6f} {b5:9.6f}")
            if a != b or a5 != b5:
                diffs[c] = {"R@1": [a, b], "R@5": [a5, b5]}
        else:
            print(f"{c:12s} {f32[c]:9.6f} {f64[c]:9.6f} {f64[c] - f32[c]:+10.2e}")
            if f32[c] != f64[c]:
                diffs[c] = [f32[c], f64[c]]

    # [ULIP2 REVIEWER 2026-08-30] The first version compared f32 against f64 and
    # called any difference a decision. That is the wrong baseline. The question
    # is not "do the two new computations agree with each other" but "does
    # either of them move a number ALREADY ON RECORD" -- and the answer turned
    # out to be the opposite of what the two-way comparison suggested.
    baseline = args.baseline_record and Path(args.baseline_record)
    recorded = (json.loads(baseline.read_text()).get("dev_val")
                if baseline and baseline.exists() else None)
    agree = {}
    if recorded:
        for dt, got in (("float32", f32), ("float64", f64)):
            agree[dt] = sum(
                recorded[c]["R@1"] == got[c]["R@1"]
                and recorded[c]["R@5"] == got[c]["R@5"]
                for c in QUERY_CONDITIONS)
        print(f"\nbit-exact agreement with {baseline}:")
        for dt, n in agree.items():
            print(f"  {dt}: {n}/{len(QUERY_CONDITIONS)} conditions")

    if recorded:
        verdict = (f"float64 reproduces {agree['float64']}/7 recorded conditions, "
                   f"float32 reproduces {agree['float32']}/7. The baseline is the "
                   f"RECORDED run, not the other dtype.")
    else:
        verdict = ("no baseline record given -- f32 vs f64 alone cannot say "
                   "whether an existing number would move")
    print(f"\nVERDICT: {verdict}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.ckpt,
        "n_dev_val": len(uids),
        "run_id": runlog.run_id(),
        "code_revision": runlog.code_revision(),
        "runtime_source_sha256": runlog.runtime_source_sha256(),
        "float32": f32, "float64": f64,
        "differing_cells": diffs,
        "baseline_record": str(baseline) if recorded else None,
        "bit_exact_agreement_with_baseline": agree,
        "verdict": verdict}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
