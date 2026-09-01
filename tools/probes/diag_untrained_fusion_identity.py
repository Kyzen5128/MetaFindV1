#!/usr/bin/env python3
"""ITEM 1 -- is the UNTRAINED fusion tower a near-identity map?

WHY THIS EXISTS
---------------
Untrained, dev_val scores text 0.964, image 0.910, pc 0.945, full 0.999
(`output/look/diag_untrained_same_evaluator.json`, OBSERVED DATA). Nothing is
trained, so that is not learned alignment. Two explanations are still open and
they lead to different conclusions about MetaFind's Table 1 protocol:

  LEAKAGE   query and gallery are built from the same cached vectors, so the
            task is partly vector-to-itself matching.
  IDENTITY  both untrained towers pass the ULIP embedding through almost
            unchanged, so query and gallery emit ~the same vector for the same
            asset and retrieval is trivial for a structural reason.

They are not exclusive, but IDENTITY is cheaper to test and reframes the other.
This script measures it and changes nothing.

⚠ NOT AN ARCHITECTURE CHANGE. [Kyzen 2026-08-30, verbatim] 「禁止自行改
architecture。先量。」 Nothing here mutates a module, a config, a checkpoint,
a split, or a protocol artifact. One JSON under `output/look/`.

THE MECHANISM UNDER TEST (OBSERVED IMPLEMENTATION, verified for this run)
-------------------------------------------------------------------------
`fusion.py:268-269`::

    h = self.head(x + self.modality_pos, src_key_padding_mask=pad)
    return (h * w).sum(dim=1) / denom

* `head` is `nn.TransformerEncoder` of `nn.TransformerEncoderLayer(...,
  norm_first=True)` (`fusion.py:174-182`). PyTorch's encoder layer is always
  residual, so an identity path from input to output exists by construction.
* the pool is a masked MEAN over the three modality slots, not a CLS token.
* `include_absent_slots` defaults True (`fusion.py:94`), so `active` is
  all-ones (`fusion.py:239-241`), `w` is all-ones and `denom` is 3 REGARDLESS
  of `present`. Every output is therefore a mean over three slots, two of
  which -- under a single-modality condition -- carry the per-modality mask
  token, which is the SAME vector for every asset.

So if `head` were exactly the identity, a text-only query would be
``e_text/3 + (shared constant)``. A shared constant carries no per-asset
information, which is why the centred statistic below is reported alongside
the raw one: raw cosine is depressed by that constant even when the tower is
behaving as a scaled identity on the part that varies across assets.

WHAT IS MEASURED
----------------
A  IDENTITY.  cos(fusion output, raw modality embedding), single-modality
   input, reported separately for the QUERY tower and the GALLERY tower and
   per modality. Also the same cosine after removing the across-asset mean
   from both sides ("centred"), which isolates the per-asset signal, and
   ||out|| / ||raw||. The gallery's PRODUCTION path (all three present, which
   is the only way `GalleryTower.forward` runs) is reported too, because that
   is the vector retrieval actually scores against.

B  MARGIN.  Per Table-1 condition: the positive similarity sim(Q(A), G(A)) and
   the hardest negative max_{B != A} sim(Q(A), G(B)), and their difference.
   Distributions, not just means.

C  CONTROL.  B recomputed with NO FUSION AT ALL and no parameters of any kind:
   query = mean of the raw ULIP embeddings the condition makes present,
   gallery = mean of all three raw ULIP embeddings. If the control already
   reproduces the margin, the towers are contributing nothing.

   This control is the parameter-free limit of the real module: it is what
   `ModalityFusion(kind="mean", include_absent_slots=False)` computes, since
   there the mask tokens are multiplied by zero. It is NOT the default
   configuration, which averages the mask tokens in -- that difference is the
   point of reporting both.

"UNTRAINED" means what Step 1 meant: zero optimizer steps, no Stage 1
checkpoint read. NOT un-pretrained -- PointBERT and OpenCLIP still carry
ULIP-2 / OpenCLIP weights. Only the two fusion towers are random, drawn from
`--seeds`; two towers are two draws, so one seed is one sample.

Scoring arithmetic is the production helper `normalize_for_scoring` (NumPy
float64) and the production `condition_mask`, so the similarities here are on
the same numerical footing as `evaluate_dev_val`.

================================================================================
`--state <checkpoint>` -- THE SAME MEASUREMENT ON A TRAINED MODEL
================================================================================
[Kyzen, via MASTER, 2026-08-31] The FILENAME IS HISTORICAL: with `--state` this
runs on a trained checkpoint. The default `init0` path is unchanged, byte for
byte, and still writes `diag_untrained_fusion_identity.json`; a checkpoint
writes a separate file and overwrites nothing.

THE HYPOTHESIS UNDER TEST -- Kyzen's, not ours
-----------------------------------------------
Stage 1's query is a masked SUBSET of asset A and its gallery is the FULL asset
A, so both sides contain the same modality vectors. The cheapest way to drive
that loss down may therefore be for both towers to learn PASS-THROUGH -- emit
the input largely unchanged. Text-only query then matches the gallery's text
component, image-only matches image, pc-only matches pc, and the loss falls
with NO cross-modal alignment learned at all.

This measures whether that happened. It does not measure whether it is the only
possible explanation.

⚠ SUPPORTING DIAGNOSTIC, NOT THE DISCRIMINATING TEST
-----------------------------------------------------
[GPT via MASTER, 2026-08-31 -- DEMOTING an earlier framing this file carried,
which called it "the most important measurement".] The identity cosine is
NEITHER SUFFICIENT NOR NECESSARY for the pass-through hypothesis:

  not sufficient  a RISE in cos(raw, fused) is also produced by a model that
                  genuinely learned a shared space and happens to retain raw
                  direction; by the fusion barely moving while PointBERT or the
                  projection does all the work; and by a learned projection
                  that lifts the cosine while the retrieval gain is still
                  cross-modal.
  not necessary   a model relying ENTIRELY on a same-modality shortcut but
                  applying a common rotation, F(t) = R t, shows a LOW
                  cos(t, R t) while preserving instance identity perfectly.

The discriminating test is E_same vs E_cross across training
(`diag_caption_pair_geometry.py`), not this. Read this file as supporting
evidence and nothing more.

PRE-REGISTERED READING -- fixed BEFORE the trained numbers were produced
-------------------------------------------------------------------------
Against the UNTRAINED baseline already on disk
(`diag_untrained_fusion_identity.json`, 3 seeds):

    query   text 0.4994  image 0.5333  pc 0.3963      cos_raw, mean
    gallery text 0.4984  image 0.5204  pc 0.3968

  pass-through INCREASES with training
      CONSISTENT WITH the degenerate-solution hypothesis. Not support for it,
      for the "not sufficient" reason above.
  pass-through DECREASES
      the towers relay less than they did. This does NOT refute the hypothesis
      either, for the "not necessary" reason above -- a common rotation hides a
      perfect shortcut behind a low cosine.
  UNCHANGED
      training moved something the identity cosine cannot see. Say exactly
      that. Do not force either reading onto it.

In every branch the verdict comes from E_same/E_cross, and this file only says
which way the cosine went.

A delta smaller than the untrained SEED-TO-SEED SPREAD is not distinguishable
from initialisation noise; the output carries the spread next to every delta so
that check cannot be skipped. `delta_exceeds_seed_spread` is a coarse guard, not
a significance test.

⚠ THE PC ROW'S INPUT MAY NOT BE THE SAME VECTOR IN THE TWO RUNS.
`train_scope` is a property of the CHECKPOINT, not a constant, and it is read
back from the checkpoint here rather than assumed. Under
`point_encoder_and_fuser` a Stage 1 checkpoint restores a FINE-TUNED PointBERT
as well as the towers, so the checkpoint is loaded BEFORE `collect_inputs` and
the trained run's `raw["pc"]` is the trained point encoder's output. That is the
correct comparison for pass-through -- each tower is measured against ITS OWN
input -- but it means the pc delta mixes "the tower relays more" with "the
tower's input moved". Under `fuser_only` the backbone is frozen, so that
confound is absent and the pc row is directly comparable. `text` and `image`
never have the problem: OpenCLIP is frozen and both runs read the identical
cached vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metafind import paths                                            # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,  # noqa: E402
                                     normalize_for_scoring)
from metafind.models.fusion import MODALITIES                         # noqa: E402

# Reused verbatim so the inputs are collected by the SAME code path Step 2/4
# used: same Stage1Dataset, same loader flags, same `modules_in_eval`.
from diag_text_shortcut import build_untrained, collect_inputs        # noqa: E402

LOOK = REPO / "output" / "look"
CONDS = tuple(QUERY_CONDITIONS)
QUANTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def dist(x: np.ndarray) -> dict:
    """Mean/std/min/max plus quantiles. MASTER asked for the distribution."""
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "q": {f"p{int(p*100)}": float(v)
                  for p, v in zip(QUANTS, np.quantile(x, QUANTS))}}


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return ((a * b).sum(1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))


def pos_and_hardest(q: np.ndarray, g: np.ndarray) -> dict:
    """Positive similarity, hardest-negative similarity, and the margin.

    Row i of the query stack and row i of the gallery stack are the same asset
    -- `collect_inputs` uses `shuffle=False, drop_last=False` and its caller
    asserts the loader did not reorder the split -- so the positive is the
    diagonal. Stated rather than assumed, for the same reason
    `rank_of_target` takes an explicit target column.

    `frac_margin_gt_0` is R@1 under a strict-greater rule; it is reported to
    show the margin distribution is the same object the recall came from, not
    as a replacement for `recall_at_k` (which counts ties against the model).
    """
    qn, gn = normalize_for_scoring(q), normalize_for_scoring(g)
    sim = qn @ gn.T
    n = sim.shape[0]
    pos = sim[np.arange(n), np.arange(n)].copy()
    sim[np.arange(n), np.arange(n)] = -np.inf
    hard = sim.max(axis=1)
    margin = pos - hard
    return {"positive": dist(pos), "hardest_negative": dist(hard),
            "margin": dist(margin),
            "frac_margin_gt_0": float((margin > 0).mean())}


def delta_vs_untrained(trained_ident: dict, n: int, limit) -> dict | None:
    """Trained identity cosines minus the recorded untrained ones.

    The untrained side is the mean over its three seeds, and its seed-to-seed
    SPREAD travels next to every delta: two random draws of the same
    architecture already disagree, and a delta inside that disagreement is not
    evidence of anything training did. Returns ``None`` rather than a
    misleading zero when the baseline is absent or was produced on a different
    population.
    """
    prior = LOOK / "diag_untrained_fusion_identity.json"
    if not prior.exists():
        print(f"no untrained baseline at {prior}; skipping the delta block")
        return None
    d = json.loads(prior.read_text())
    if d.get("debug_limit") != limit or d.get("n_dev_val") != n:
        print(f"untrained baseline is n={d.get('n_dev_val')} "
              f"limit={d.get('debug_limit')}, this run is n={n} limit={limit}; "
              "NOT comparable, skipping the delta block")
        return None
    out: dict = {}
    for tw in ("query", "gallery"):
        out[tw] = {}
        for m in MODALITIES:
            out[tw][m] = {}
            for st in ("cos_raw", "cos_centred", "norm_ratio"):
                v = [d["per_seed"][k]["identity"][tw][m][st]["mean"]
                     for k in d["per_seed"]]
                u, t = float(np.mean(v)), float(trained_ident[tw][m][st]["mean"])
                spread = float(max(v) - min(v))
                out[tw][m][st] = {
                    "untrained_per_seed": [float(x) for x in v],
                    "untrained_mean_over_seeds": u,
                    "untrained_seed_spread": spread,
                    "trained": t, "delta": t - u,
                    "delta_exceeds_seed_spread": abs(t - u) > spread,
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3", help="init0 only")
    ap.add_argument("--state", default="init0",
                    help="'init0' (untrained, --seeds draws) or a Stage 1 "
                         "checkpoint path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: first N dev_val assets. Any number "
                         "produced with this set is a debug run, not evidence.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    trained = args.state != "init0"

    import torch

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import load_protocols, modules_in_eval

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    dev_val = json.loads(
        (paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    if args.limit:
        dev_val = dev_val[:args.limit]
        print("⚠ --limit set: DEBUG RUN. These numbers are not evidence.",
              flush=True)
    print(f"dev_val {len(dev_val):,} · state {args.state} · "
          f"{'checkpoint' if trained else f'seeds {seeds}'} · batch {bs}",
          flush=True)

    # The checkpoint stores only its TRAINABLE parameters, so a backbone built
    # with the wrong scope has a different trainable set and the load is
    # rejected. Read the scope back rather than assuming one.
    scope = "point_encoder_and_fuser"
    if trained:
        scope = torch.load(Path(args.state), map_location="cpu",
                           weights_only=False).get("train_scope", scope)
        print(f"checkpoint train_scope = {scope}", flush=True)
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope=scope))
    model_t, ckpt_sha = None, None
    if trained:
        # BEFORE `collect_inputs`. Under `point_encoder_and_fuser` the
        # checkpoint restores a fine-tuned PointBERT as well as the two towers,
        # and collecting inputs first would measure trained towers against
        # RELEASED-encoder point clouds -- a comparison of two different
        # models. Under `fuser_only` the order does not matter; loading first
        # is correct either way.
        from metafind.train.stage1 import (build_model,
                                           load_stage1_checkpoint)
        model_t, loss_fn = build_model(encoding, training, hyper)
        model_t.to(args.device)
        load_stage1_checkpoint(backbone, model_t, loss_fn, Path(args.state))
        ckpt_sha = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        print(f"loaded {args.state}  sha256 {ckpt_sha[:16]}…", flush=True)
    t0 = time.time()
    order, T, I, P = collect_inputs(backbone, dev_val,
                                    encoding["image_aggregation"],
                                    args.device, bs)
    assert order == dev_val, "loader reordered the split"
    print(f"inputs collected in {time.time()-t0:.0f}s  "
          f"text {tuple(T.shape)} image {tuple(I.shape)} pc {tuple(P.shape)}",
          flush=True)

    raw = {"text": T.numpy().astype(np.float64),
           "image": I.numpy().astype(np.float64),
           "pc": P.numpy().astype(np.float64)}

    # ---- C: the parameter-free control, computed once (seed-independent) ----
    # No module, no mask token, no learned weight: the mean of the raw ULIP
    # vectors the condition makes present, against the mean of all three.
    g_raw = np.mean([raw[m] for m in MODALITIES], axis=0)
    control = {}
    for cond in CONDS:
        present = [m for m, f in zip(MODALITIES, QUERY_CONDITIONS[cond]) if f]
        control[cond] = pos_and_hardest(
            np.mean([raw[m] for m in present], axis=0), g_raw)
        c = control[cond]
        print(f"  CONTROL {cond:>10}  pos {c['positive']['mean']:.4f}  "
              f"hard {c['hardest_negative']['mean']:.4f}  "
              f"margin {c['margin']['mean']:+.4f}  "
              f"R@1~ {c['frac_margin_gt_0']:.4f}", flush=True)

    per_seed: dict[str, dict] = {}
    states = [("trained", None)] if trained else [(str(s), s) for s in seeds]
    for label, seed in states:
        model = model_t if seed is None else build_untrained(
            seed, encoding, training, hyper, args.device)
        n = len(order)

        # ---- A: identity cosines ------------------------------------------
        # Single-modality input to BOTH towers, so "the raw embedding" is
        # unambiguous. The gallery is driven through `.fusion` directly because
        # `GalleryTower.forward` refuses an incomplete gallery by design
        # (dual_tower.py:175); `model.gallery.fusion` is the same module that
        # `forward` calls, so this measures the gallery tower, not a stand-in.
        towers = {"query": model.query.fusion, "gallery": model.gallery.fusion}
        outs = {t: {m: [] for m in MODALITIES} for t in towers}
        gal_full = []
        with modules_in_eval(model), torch.no_grad():
            for i in range(0, n, bs):
                sl = slice(i, min(i + bs, n))
                b = sl.stop - sl.start
                e = {"text": T[sl].to(args.device), "image": I[sl].to(args.device),
                     "pc": P[sl].to(args.device)}
                gal_full.append(model.gallery(e).float().cpu())
                for m in MODALITIES:
                    mask = condition_mask(m, b).to(args.device)
                    for tname, tmod in towers.items():
                        outs[tname][m].append(
                            tmod(e, present=mask).float().cpu())
        gal_full = np.concatenate([x.numpy() for x in gal_full]).astype(np.float64)

        ident: dict = {}
        for tname in towers:
            ident[tname] = {}
            for m in MODALITIES:
                o = np.concatenate([x.numpy() for x in outs[tname][m]]
                                   ).astype(np.float64)
                r = raw[m]
                # Centred: the across-asset mean is a constant every query
                # shares -- two of the three pooled slots ARE that constant --
                # and a constant cannot help retrieval. Removing it isolates
                # the per-asset signal, which is what "identity" has to mean
                # for the trivial-retrieval story to hold.
                ident[tname][m] = {
                    "cos_raw": dist(rowwise_cos(o, r)),
                    "cos_centred": dist(rowwise_cos(o - o.mean(0), r - r.mean(0))),
                    "norm_ratio": dist(np.linalg.norm(o, axis=1)
                                       / np.linalg.norm(r, axis=1)),
                }
                z = ident[tname][m]
                print(f"  {label:>8} {tname:>7} {m:>5}  "
                      f"cos {z['cos_raw']['mean']:+.4f}  "
                      f"centred {z['cos_centred']['mean']:+.4f}  "
                      f"|out|/|raw| {z['norm_ratio']['mean']:.3f}", flush=True)
        # The gallery vector retrieval actually scores against.
        ident["gallery_production_all_present"] = {
            m: {"cos_raw": dist(rowwise_cos(gal_full, raw[m])),
                "cos_centred": dist(rowwise_cos(gal_full - gal_full.mean(0),
                                                raw[m] - raw[m].mean(0)))}
            for m in MODALITIES}
        ident["gallery_production_all_present"]["mean_of_three"] = {
            "cos_raw": dist(rowwise_cos(gal_full, g_raw)),
            "cos_centred": dist(rowwise_cos(gal_full - gal_full.mean(0),
                                            g_raw - g_raw.mean(0)))}

        # ---- B: margins through the production towers ----------------------
        margins = {}
        with modules_in_eval(model), torch.no_grad():
            for cond in CONDS:
                q = []
                for i in range(0, n, bs):
                    sl = slice(i, min(i + bs, n))
                    b = sl.stop - sl.start
                    e = {"text": T[sl].to(args.device),
                         "image": I[sl].to(args.device),
                         "pc": P[sl].to(args.device)}
                    q.append(model.query(
                        e, present=condition_mask(cond, b).to(args.device)
                    ).float().cpu())
                margins[cond] = pos_and_hardest(
                    np.concatenate([x.numpy() for x in q]).astype(np.float64),
                    gal_full)
                mm = margins[cond]
                print(f"  {label:>8} MARGIN {cond:>10}  "
                      f"pos {mm['positive']['mean']:.4f}  "
                      f"hard {mm['hardest_negative']['mean']:.4f}  "
                      f"margin {mm['margin']['mean']:+.4f}  "
                      f"R@1~ {mm['frac_margin_gt_0']:.4f}", flush=True)

        per_seed[label] = {"identity": ident, "margins": margins}
        del model
        torch.cuda.empty_cache()

    delta = delta_vs_untrained(per_seed["trained"]["identity"], len(order),
                               args.limit) if trained else None
    if delta:
        print("\nDELTA vs untrained (mean over 3 seeds)  "
              "-- + = MORE pass-through, - = LESS")
        print(f"{'tower/mod':>16} {'cos_raw':>22} {'cos_centred':>22} "
              f"{'norm_ratio':>22}")
        for tw in ("query", "gallery"):
            for m in MODALITIES:
                row = ""
                for st in ("cos_raw", "cos_centred", "norm_ratio"):
                    z = delta[tw][m][st]
                    flag = "*" if z["delta_exceeds_seed_spread"] else " "
                    row += (f"  {z['untrained_mean_over_seeds']:.4f}->"
                            f"{z['trained']:.4f} {z['delta']:+.4f}{flag}")
                print(f"{tw + '/' + m:>16}{row}")
        print("  * = |delta| exceeds the untrained seed-to-seed spread "
              "(coarse guard, not a significance test)")

    tag = "init0" if not trained else (Path(args.state).parent.name
                                       or Path(args.state).stem)
    OUT = LOOK / ("diag_untrained_fusion_identity.json" if not trained
                  else f"diag_trained_fusion_identity_{tag}.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": (("TRAINED" if trained else "UNTRAINED") +
                 " fusion identity test. A: cos(fusion out, raw modality "
                 "embedding) per tower per modality, single-modality input. "
                 "B: positive vs hardest-negative similarity per Table-1 "
                 "condition through the production towers. C: the same, with "
                 "no fusion and no parameters (mean of the present raw ULIP "
                 "embeddings vs mean of all three)."),
        "caveat": ((
            "TRAINED = a Stage 1 checkpoint is loaded, which restores the "
            "fine-tuned PointBERT as well as both towers, so this run's raw "
            "`pc` input is NOT the untrained run's. text and image are: "
            "OpenCLIP is frozen." if trained else
            "UNTRAINED = zero optimizer steps, no Stage 1 checkpoint. NOT "
            "un-pretrained: PointBERT and OpenCLIP still carry ULIP-2 / "
            "OpenCLIP weights; only the two fusion towers are random.") +
            " dev_val is a 4,569-asset HPO selector split, NOT the paper's "
            "20% test population."),
        "n_dev_val": len(order), "seeds": None if trained else seeds,
        "batch_size": bs, "debug_limit": args.limit,
        "state": "trained" if trained else "INIT-0 (zero optimizer steps)",
        "checkpoint": args.state if trained else None,
        "checkpoint_sha256": ckpt_sha,
        "hypothesis": (
            "Kyzen 2026-08-31: Stage 1's query is a masked SUBSET of asset A "
            "and its gallery is the FULL asset A, so the cheapest way to lower "
            "the loss may be for both towers to learn PASS-THROUGH -- relaying "
            "the input rather than aligning modalities."),
        "status": ("SUPPORTING DIAGNOSTIC. The identity cosine is neither "
                   "sufficient nor necessary for the pass-through hypothesis: "
                   "a rise is also produced by a genuinely learned shared "
                   "space, and a pure same-modality shortcut composed with a "
                   "common rotation F(t)=Rt shows a LOW cosine while "
                   "preserving instance identity perfectly. The "
                   "discriminating test is E_same vs E_cross across training."),
        "prereg_reading": {
            "increase": ("CONSISTENT WITH the degenerate-solution hypothesis; "
                         "not support for it"),
            "decrease": ("the towers relay less than they did; this does NOT "
                         "refute the hypothesis, because a common rotation "
                         "hides a perfect shortcut behind a low cosine"),
            "unchanged": ("training moved something the identity cosine cannot "
                          "see; say exactly that and force neither reading"),
            "noise_floor": ("a delta smaller than the untrained seed-to-seed "
                            "spread is not distinguishable from initialisation "
                            "noise; `delta_exceeds_seed_spread` is a coarse "
                            "guard, not a significance test"),
        },
        "train_scope": scope,
        "pc_row_caveat": (
            "train_scope is point_encoder_and_fuser, so the checkpoint restores "
            "a fine-tuned PointBERT and the trained run's raw pc input is not "
            "the untrained run's. Each tower is measured against ITS OWN input, "
            "which is the right comparison for pass-through, but the pc delta "
            "mixes 'the tower relays more' with 'the tower's input moved'. text "
            "and image are free of this: OpenCLIP is frozen and both runs read "
            "the identical cached vectors."
            if scope == "point_encoder_and_fuser" else
            f"train_scope is {scope}, so the backbone is frozen and both runs "
            "read the identical cached pc vectors. The pc delta is a clean "
            "measurement of the tower, with no input-moved confound."),
        "control_no_fusion": control,
        "delta_vs_untrained": delta,
        "per_seed": per_seed,
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
