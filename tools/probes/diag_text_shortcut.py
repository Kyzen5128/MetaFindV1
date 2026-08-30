#!/usr/bin/env python3
"""Steps 2-4 — is dev-val R@1 measuring RETRIEVAL, or an instance SELF-MATCH?

WHY THIS EXISTS
---------------
An UNTRAINED model scores mean_R@1 = 0.9718 on dev_val (Step 1,
`output/look/diag_untrained_same_evaluator.json`, OBSERVED DATA). Nothing is
trained, so whatever that number measures, it is not learned alignment.

Codex's diagnosis, accepted by MASTER: for each asset the query and the gallery
are built from THE SAME cached modality vectors. The hypothesis under test is
therefore NOT about strings -- it is that **the query and the gallery use the
identical cached text embedding for the same asset**, so "find this asset from
its text" is partly matching a vector to itself. String counting cannot show
that; only swapping the vector can.

WHAT THIS SCRIPT MUST NOT DO
-----------------------------
It writes JSON under `output/look/` and NOTHING else. It does not touch the
serializer, `resolve_stage1.py`, the 45,692-asset embedding cache, any
checkpoint, split, HPO output, the trainer, or `evaluate_dev_val`.

STEP 2 -- THE 2x2 [MASTER amendment, before any number was seen]
----------------------------------------------------------------
    C = the canonical description. NOT merely "the string the gallery holds":
        it is the caption a CLIP ranker SELECTED as best of five (`rank: 0`).
    A = the deterministic highest-ranked NON-canonical alternate -- the lowest
        `rank` whose `text` differs from the canonical description.

A single `A -> G(C)` arm is not interpretable, because A is BY CONSTRUCTION a
lower-ranked caption: a drop is equally consistent with "exact-text identity was
removed" and with "the alternate is simply a worse description". The diagonal
separates them, so all four cells run:

    C -> G(C)   the current protocol, the control
    A -> G(C)
    C -> G(A)
    A -> G(A)

PRE-REGISTERED READING. Written here before the run, so it cannot be fitted
to the result afterwards:

  * `C/C` and `A/A` both high while `C/A` and `A/C` both fall sharply
        -> strong evidence for an exact same-text shortcut. The diagonal
           survives because the strings match; the off-diagonal dies because
           they do not.
  * `A/A` itself falls a lot
        -> the caption-QUALITY confound is not excluded, and nothing about the
           shortcut is established.
  * cross-caption (`A/C`, `C/A`) stays high
        -> the same-text shortcut is not the main cause, and Step 4 is the
           measurement that matters.

STEP 4 IS NOT OPTIONAL [MASTER amendment]. `image` (0.9103) and `pc` (0.9452)
are also abnormally high untrained and the 2x2 only probes text. Order of
execution is 2, then 4, then 3.

WHY THIS DOES NOT CALL `evaluate_dev_val`
------------------------------------------
It cannot. Every step overrides an input `evaluate_dev_val` does not expose:
Step 2 needs a different text vector on the query side than on the gallery
side, Step 3 needs a different text vector entirely, Step 4 needs an incomplete
gallery -- which `GalleryTower.forward` refuses outright (dual_tower.py:175,
"gallery tower is modality-complete").

So the scoring loop is re-implemented and the ARITHMETIC is not:
`normalize_for_scoring`, `condition_mask`, `rank_of_target` and `recall_at_k`
are imported from `metafind.eval.retrieval`, the same objects `evaluate_dev_val`
imports, in the same float64 order, at the same batch size, over the same
loader ordering.

Two gates prove the re-implementation is the same experiment. If either fails
the run stops, because a harness that cannot reproduce the control is not
measuring the thing the control measured:

  GATE 1  re-encoding the canonical serialized string and applying the cache's
          own float16 round-trip must reproduce the cached `text` vector BIT
          FOR BIT, all 4,569 assets.
  GATE 2  this harness's canonical control must reproduce `evaluate_dev_val`'s
          own seven conditions BIT FOR BIT for the same seed -- and (2b) the
          recorded Step 1 values for that seed, which were produced by another
          process on another day.

WHY TEXT IS ENCODED ONE STRING AT A TIME
-----------------------------------------
MEASURED, and it is the reason the first version of GATE 1 failed. n06 encodes
one caption per call (`encode_text_image.py:374`, `encode_text([text])`).
Re-encoding the same strings in batches changes the float32 output enough that
the float16 store lands on a different value:

    batch    f16 elements differing from cache (64 assets, 81,920 elements)
      1                0          max |delta| 0
      8              407          max |delta| 1.953e-03  (one f16 ULP)
     64              689          max |delta| 1.953e-03

identical under both `fuser_only` and `point_encoder_and_fuser`. So every text
here is encoded at batch 1: it is what makes "the only difference is the
string" true rather than approximately true. Cost is ~31 s per 4,569 strings.

"UNTRAINED" means exactly what Step 1 meant: zero optimizer steps, no Stage 1
checkpoint. PointBERT and OpenCLIP are still ULIP-2 / OpenCLIP pretrained; only
the two fusion towers are random, drawn from `--seeds`.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths                                          # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,  # noqa: E402
                                     normalize_for_scoring, rank_of_target,
                                     recall_at_k)
from metafind.models.fusion import MODALITIES                       # noqa: E402
from metafind.models.resolve_stage1 import (TEXT_TEMPLATE,          # noqa: E402
                                            serialize_annotation)

LOOK = REPO / "output" / "look"
CONDS = tuple(QUERY_CONDITIONS)

# --- Step 3 -----------------------------------------------------------------
# IMPLEMENTATION CHOICE, flagged. The paper specifies no sub-serialization; only
# S4 is an authoritative artifact (it IS `TEXT_TEMPLATE`, asserted at run time).
# The increments are grammatical sentences rather than literal string prefixes
# of S4, because a prefix cut would end "made of metal," with a dangling comma
# and hand CLIP a fragment -- the failure `_cap` exists to avoid.
FIELD_VARIANTS = {
    "S0_description": "{description}",
    "S1_+category":   "{description} {category}.",
    "S2_+materials":  "{description} {category} made of {materials}.",
    "S3_+dimensions": ("{description} {category} made of {materials}, "
                       "roughly {width} by {length} by {height} centimetres."),
    "S4_+placement":  TEXT_TEMPLATE,
    "D_dimensions_only": "Roughly {width} by {length} by {height} centimetres.",
}

# --- Step 4 -----------------------------------------------------------------
# DIAGNOSTIC ONLY. Each row removes the query's own modality FROM THE GALLERY,
# which violates the paper's modality-complete gallery encoder (sec. 2.6) and
# can never be a reported MetaFind protocol. [Kyzen 2026-08-30, verbatim]
# 「明確標記 DIAGNOSTIC ONLY，不得作為 MetaFind Table 1 reported protocol」
LEAVE_ONE_OUT = {          # query condition -> modalities the GALLERY keeps
    "text":  ("image", "pc"),
    "image": ("text", "pc"),
    "pc":    ("text", "image"),
}


def f16_roundtrip(x: np.ndarray) -> np.ndarray:
    """The cache's own dtype path, applied to a freshly encoded vector.

    n06 stores `text_vec.astype(np.float16)` (encode_text_image.py:530) and
    `Stage1Dataset.__getitem__` reads it back with `.astype(np.float32)`
    (stage1.py:328). A vector encoded now and left at float32 would differ from
    a cached one by the float16 rounding ALONE, and the claim under test is that
    the only difference is the string. So the round-trip is applied, not skipped.
    """
    return x.astype(np.float16).astype(np.float32)


def collect_inputs(backbone, uids, aggregation, device, batch_size):
    """The three modality matrices `evaluate_dev_val` would build, in its order.

    Same `Stage1Dataset`, same DataLoader flags, same batch size, and
    `encode_pc` inside `modules_in_eval` -- that context is not cosmetic, it is
    what keeps DropPath off and BatchNorm from writing dev-val statistics into
    the backbone (stage1.py:426). Collected ONCE: nothing here depends on the
    seed (the backbone is loaded from a checkpoint and draws nothing from the
    RNG), so every seed and every variant below reuses these bytes.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.train.stage1 import Stage1Dataset, collate, modules_in_eval

    loader = DataLoader(Stage1Dataset(uids, aggregation), batch_size=batch_size,
                        shuffle=False, collate_fn=collate, num_workers=4,
                        drop_last=False)
    order, text, image, pc = [], [], [], []
    with modules_in_eval(getattr(backbone, "model", None)), torch.no_grad():
        for batch in loader:
            order.extend(batch["uid"])
            text.append(batch["text"])
            image.append(batch["image"])
            pc.append(backbone.encode_pc(batch["pc"].to(device)).float().cpu())
    return order, torch.cat(text), torch.cat(image), torch.cat(pc)


def encode_texts(backbone, strings, label=""):
    """`backbone.encode_text` ONE STRING AT A TIME, then the f16 round-trip.

    Batch 1 is load-bearing, not a default -- see the module docstring's
    measurement. Batching changes the stored value on ~0.5-0.8% of elements.
    """
    import torch

    out, t0 = [], time.time()
    with torch.no_grad():
        for s in strings:
            out.append(backbone.encode_text([s]).float().cpu())
    v = torch.from_numpy(f16_roundtrip(torch.cat(out).numpy()))
    print(f"  encoded {len(strings):,} strings {label} ({time.time()-t0:.0f}s)",
          flush=True)
    return v


def score(model, gallery_text, query_text, image, pc, device, batch_size,
          conditions=CONDS, gallery_present=None):
    """`evaluate_dev_val`'s scoring, with the gallery and query texts separable.

    Returns ``(scores, ranks)``. Everything numerical is the imported production
    helper. The three departures from `evaluate_dev_val` are the three the
    diagnostic needs and no others:

      * `gallery_text` may differ from `query_text` (Step 2's 2x2);
      * `conditions` may be a subset (Step 4 asks three cells, not seven);
      * `gallery_present` may mask a modality out of the GALLERY (Step 4), which
        goes through `model.gallery.fusion` directly because
        `GalleryTower.forward` refuses an incomplete gallery by design. With
        `gallery_present=None` the call is `model.gallery(embeds)` -- the
        production path, unchanged.

    `ranks` is returned so a subset recall can be taken WITHOUT re-scoring:
    `rank_of_target` is per-row against the whole gallery, so selecting rows
    after ranking equals selecting them before it, and the gallery stays 4,569
    wide either way.
    """
    import torch

    from metafind.train.stage1 import modules_in_eval

    n = gallery_text.size(0)
    gal, per_cond = [], {c: [] for c in conditions}
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            b = sl.stop - sl.start
            img, cloud = image[sl].to(device), pc[sl].to(device)
            g_embeds = {"text": gallery_text[sl].to(device), "image": img, "pc": cloud}
            if gallery_present is None:
                gal.append(model.gallery(g_embeds).float().cpu())
            else:
                mask = torch.tensor(gallery_present, dtype=torch.bool,
                                    device=device).expand(b, len(MODALITIES))
                gal.append(model.gallery.fusion(g_embeds, present=mask).float().cpu())
            q_embeds = {"text": query_text[sl].to(device), "image": img, "pc": cloud}
            for cond in conditions:
                per_cond[cond].append(model.query(
                    q_embeds, present=condition_mask(cond, b).to(device)).float().cpu())

    g = normalize_for_scoring(torch.cat(gal).numpy())
    targets = np.arange(g.shape[0])
    out, ranks = {}, {}
    for cond in conditions:
        sim = normalize_for_scoring(torch.cat(per_cond[cond]).numpy()) @ g.T
        out[cond] = recall_at_k(sim, targets, ks=(1, 5))
        ranks[cond] = rank_of_target(sim, targets)
    if len(conditions) == len(CONDS):
        out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in conditions]))
        out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in conditions]))
    out["n_gallery"] = int(g.shape[0])
    return out, ranks


def build_untrained(seed, encoding, training, hyper, device):
    import torch

    from metafind.train.stage1 import build_model

    torch.manual_seed(seed)                 # this call is what draws the towers
    model, _ = build_model(encoding, training, hyper)
    return model.to(device)


def avg(per_seed, seeds, path):
    """Mean over seeds of a nested value addressed by `path`."""
    vals = []
    for s in seeds:
        v = per_seed[s]
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="2,4,3",
                    help="comma list; executed in the order given (MASTER: 2, "
                         "then 4, then 3)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: first N dev_val assets. Any number "
                         "produced with this set is a debug run, not evidence.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip GATE 2 (one extra full pass); GATE 1 still runs")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch

    from metafind.data.encode_text_image import true_token_count
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import evaluate_dev_val, load_protocols

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    dev_val = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    if args.limit:
        dev_val = dev_val[:args.limit]
        print("⚠ --limit set: DEBUG RUN. These numbers are not evidence.", flush=True)
    print(f"dev_val {len(dev_val):,} · seeds {seeds} · steps {steps} · batch {bs}",
          flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))

    t0 = time.time()
    order, C_vec, image_cached, pc_enc = collect_inputs(
        backbone, dev_val, encoding["image_aggregation"], args.device, bs)
    assert order == dev_val, "loader reordered the split"
    print(f"inputs collected in {time.time()-t0:.0f}s  text {tuple(C_vec.shape)} "
          f"image {tuple(image_cached.shape)} pc {tuple(pc_enc.shape)}", flush=True)

    # ---- strings -----------------------------------------------------------
    annotations = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
                   for u in order]
    canonical = [serialize_annotation(a) for a in annotations]

    alt_strings, alt_rank = [], []
    for a in annotations:
        cands = sorted((c for c in (a.get("description_candidates") or [])
                        if c["text"] != a["description"]), key=lambda c: c["rank"])
        if not cands:
            raise SystemExit(f"{a['uid']}: no non-canonical description candidate")
        alt_strings.append(serialize_annotation({**a, "description": cands[0]["text"]}))
        alt_rank.append(cands[0]["rank"])
    assert all(x != y for x, y in zip(alt_strings, canonical)), \
        "an alternate serialized to the canonical string"

    # ---- GATE 1 ------------------------------------------------------------
    canon_re = encode_texts(backbone, canonical, "(canonical, for GATE 1)")
    exact = bool(torch.equal(canon_re, C_vec))
    d = (canon_re - C_vec).abs()
    print(f"GATE 1  re-encoded canonical vs cached text\n"
          f"  bit-identical: {exact}   max|delta| {d.max().item():.3e}   "
          f"rows differing {int((d.sum(1) > 0).sum())}/{len(order)}", flush=True)
    if not exact:
        raise SystemExit("GATE 1 FAILED: the harness cannot reproduce the cache, "
                         "so a text swap would not be the only difference. STOP.")
    del canon_re

    # ---- GATE 2 ------------------------------------------------------------
    verify_seed, control = seeds[0], {}
    if not args.no_verify:
        t0 = time.time()
        ref = evaluate_dev_val(
            backbone, build_untrained(verify_seed, encoding, training, hyper,
                                      args.device),
            dev_val, encoding["image_aggregation"], args.device, bs)
        mine, _ = score(build_untrained(verify_seed, encoding, training, hyper,
                                        args.device),
                        C_vec, C_vec, image_cached, pc_enc, args.device, bs)
        bad = [c for c in CONDS
               if ref[c]["R@1"] != mine[c]["R@1"] or ref[c]["R@5"] != mine[c]["R@5"]]
        print(f"\nGATE 2  harness vs evaluate_dev_val, seed {verify_seed} "
              f"({time.time()-t0:.0f}s): bit-identical on {7-len(bad)}/7"
              + (f"   MISMATCH {bad}" if bad else ""))
        for c in CONDS:
            print(f"    {c:>12}  evaluate_dev_val {ref[c]['R@1']:.9f}   "
                  f"harness {mine[c]['R@1']:.9f}")
        if bad:
            raise SystemExit("GATE 2 FAILED: this harness and the trainer's own "
                             "evaluator disagree on the canonical control. That "
                             "disagreement is itself the finding. STOP.")
        rec = LOOK / "diag_untrained_same_evaluator.json"
        if rec.exists() and not args.limit:
            step1 = json.loads(rec.read_text())
            i = step1["seeds"].index(verify_seed)
            worst = max(abs(mine[c]["R@1"] - step1["per_condition"][c]["per_seed_R@1"][i])
                        for c in CONDS)
            print(f"  GATE 2b vs Step 1's RECORDED seed {verify_seed} (another "
                  f"process, another day): max |delta| {worst:.3e}"
                  + ("  bit-identical" if worst == 0 else "  ⚠ NOT bit-identical"))
            control["gate2b_max_abs_delta_vs_recorded_step1"] = float(worst)

    results = {"n_dev_val": len(order), "seeds": seeds, "batch_size": bs,
               "debug_limit": args.limit, "gate1_f16_bit_identical": exact,
               "gate2_matches_evaluate_dev_val": None if args.no_verify else True,
               **control}

    def dump(step: int):
        p = LOOK / f"diag_text_shortcut_step{step}.json"
        p.write_text(json.dumps(results, indent=1, default=float))
        print(f"\nwrote {p}", flush=True)

    # ======================= STEP 2 -- the 2x2 ==============================
    if 2 in steps:
        over = np.array([true_token_count(s) > 77 for s in alt_strings])
        keep = np.where(~over)[0]
        print(f"\nSTEP 2  alternate-rank histogram "
              f"{dict(sorted(collections.Counter(alt_rank).items()))}   "
              f"alternates over CLIP's 77-token context: {int(over.sum())} "
              f"(the tokenizer truncates their tail, as it would in production)",
              flush=True)
        A_vec = encode_texts(backbone, alt_strings, "(alternate)")
        cells = {"C/C": (C_vec, C_vec), "A/C": (C_vec, A_vec),
                 "C/A": (A_vec, C_vec), "A/A": (A_vec, A_vec)}

        per_seed = {}
        for seed in seeds:
            m = build_untrained(seed, encoding, training, hyper, args.device)
            per_seed[seed], t0 = {}, time.time()
            for name, (gal_t, qry_t) in cells.items():
                sc, rk = score(m, gal_t, qry_t, image_cached, pc_enc, args.device, bs)
                sc["inbudget_R@1"] = {c: float((rk[c][keep] <= 1).mean()) for c in CONDS}
                per_seed[seed][name] = sc
            print("  seed {}: ".format(seed)
                  + "  ".join(f"{n} {per_seed[seed][n]['mean_R@1']:.4f}" for n in cells)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            del m
            torch.cuda.empty_cache()

        print(f"\nSTEP 2  2x2, notation QUERY/GALLERY, mean over seeds {seeds}\n"
              f"{'condition':>12}" + "".join(f"{n:>12}" for n in cells)
              + f"{'A/C - C/C':>12}{'A/A - C/C':>12}")
        table = {}
        for c in list(CONDS) + ["mean_R@1", "mean_R@5"]:
            path = (lambda n: (n, c)) if c.startswith("mean") else \
                   (lambda n: (n, c, "R@1"))
            v = {n: avg(per_seed, seeds, path(n)) for n in cells}
            table[c] = {**v, "A/C_minus_C/C_pp": (v["A/C"] - v["C/C"]) * 100,
                        "A/A_minus_C/C_pp": (v["A/A"] - v["C/C"]) * 100}
            print(f"{c:>12}" + "".join(f"{v[n]:12.4f}" for n in cells)
                  + f"{(v['A/C']-v['C/C'])*100:+12.2f}"
                  + f"{(v['A/A']-v['C/C'])*100:+12.2f}")

        print(f"\n  R@5, text row only: "
              + "  ".join(f"{n} {avg(per_seed, seeds, (n, 'text', 'R@5')):.4f}"
                          for n in cells))
        print(f"  the {int(over.sum())} over-budget alternates excluded "
              f"({len(keep)} queries, gallery still {len(order)}), text R@1: "
              + "  ".join(f"{n} {avg(per_seed, seeds, (n, 'inbudget_R@1', 'text')):.4f}"
                          for n in cells))

        results["step2"] = {
            "what": ("2x2 over description choice. C = canonical (CLIP-ranker "
                     "rank 0); A = lowest-rank candidate whose text differs from "
                     "the canonical description, substituted into "
                     "serialize_annotation's `description` field. Notation "
                     "QUERY/GALLERY. image and pc identical in every cell; every "
                     "text encoded at batch 1 through backbone.encode_text."),
            "pre_registered_reading": [
                "C/C and A/A high, C/A and A/C fall sharply -> exact same-text "
                "shortcut",
                "A/A itself falls a lot -> caption-quality confound not excluded, "
                "shortcut NOT established",
                "cross-caption stays high -> shortcut is not the main cause; "
                "Step 4 is the measurement that matters",
            ],
            "alternate_rank_histogram": dict(sorted(collections.Counter(alt_rank).items())),
            "alternates_over_77_tokens": int(over.sum()),
            "n_query_in_budget": int(len(keep)),
            "per_condition": table,
            "per_seed": {str(s): per_seed[s] for s in seeds},
        }
        results["step2"]["inbudget_text_R@1"] = {
            n: avg(per_seed, seeds, (n, "inbudget_R@1", "text")) for n in cells}
        dump(2)

    # ================= STEP 4 -- same-modality removal ======================
    if 4 in steps:
        print("\n⚠ STEP 4 IS DIAGNOSTIC ONLY. Removing the query's own modality "
              "from the gallery violates the paper's modality-complete gallery "
              "encoder (sec. 2.6) and MUST NOT be reported as a MetaFind Table 1 "
              "protocol. 【明確標記 DIAGNOSTIC ONLY，不得作為 MetaFind Table 1 "
              "reported protocol】", flush=True)
        per_seed = {}
        for seed in seeds:
            m = build_untrained(seed, encoding, training, hyper, args.device)
            per_seed[seed], t0 = {}, time.time()
            for cond, kept in LEAVE_ONE_OUT.items():
                present = tuple(mod in kept for mod in MODALITIES)
                sc, _ = score(m, C_vec, C_vec, image_cached, pc_enc, args.device,
                              bs, conditions=(cond,), gallery_present=present)
                per_seed[seed][cond] = sc
            print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
            del m
            torch.cuda.empty_cache()

        base = json.loads((LOOK / "diag_untrained_same_evaluator.json").read_text())
        print(f"\nSTEP 4  DIAGNOSTIC ONLY -- gallery WITHOUT the query's own "
              f"modality\n{'query':>12}{'gallery keeps':>18}{'R@1':>10}{'R@5':>10}"
              f"{'full-gallery R@1':>20}{'delta pp':>11}")
        table = {}
        for cond, kept in LEAVE_ONE_OUT.items():
            r1 = avg(per_seed, seeds, (cond, cond, "R@1"))
            r5 = avg(per_seed, seeds, (cond, cond, "R@5"))
            b = float(np.mean([base["per_condition"][cond]["per_seed_R@1"][
                base["seeds"].index(s)] for s in seeds]))
            table[cond] = {"gallery_keeps": list(kept), "R@1": r1, "R@5": r5,
                           "full_gallery_R@1": b, "delta_pp": (r1 - b) * 100}
            print(f"{cond:>12}{'+'.join(kept):>18}{r1:10.4f}{r5:10.4f}"
                  f"{b:20.4f}{(r1-b)*100:+11.2f}")
        results["step4"] = {
            "DIAGNOSTIC_ONLY": ("violates sec. 2.6's modality-complete gallery "
                                "encoder; never a reported MetaFind protocol"),
            "how": ("model.gallery.fusion(embeds, present=mask) -- "
                    "GalleryTower.forward is bypassed because it refuses an "
                    "incomplete gallery by design (dual_tower.py:175)"),
            "baseline": "Step 1's recorded untrained full-gallery R@1, same seeds",
            "per_condition": table,
            "per_seed": {str(s): per_seed[s] for s in seeds},
        }
        dump(4)

    # ==================== STEP 3 -- field ablation ==========================
    if 3 in steps:
        assert all(serialize_annotation(a, template=FIELD_VARIANTS["S4_+placement"]) == c
                   for a, c in zip(annotations, canonical)), \
            "S4 is not byte-identical to the production serializer"
        vecs = {}
        for name, tmpl in FIELD_VARIANTS.items():
            if name == "S4_+placement":
                vecs[name] = C_vec       # proven bit-identical by GATE 1
                print(f"  {name:<20} reuses the cached canonical vectors")
                continue
            strings = [serialize_annotation(a, template=tmpl) for a in annotations]
            print(f"  {name:<20} e.g. {strings[0][:88]!r}", flush=True)
            vecs[name] = encode_texts(backbone, strings, f"({name})")

        per_seed = {}
        for seed in seeds:
            m = build_untrained(seed, encoding, training, hyper, args.device)
            per_seed[seed], t0 = {}, time.time()
            for name, v in vecs.items():
                # BOTH sides: this is a SERIALIZER ablation, so S4 must reproduce
                # today's numbers exactly -- which is what checks the machinery.
                per_seed[seed][name] = score(m, v, v, image_cached, pc_enc,
                                             args.device, bs)[0]
                # and the asymmetric reading, query-only, for the same cost.
                per_seed[seed][name + "|query_only"] = score(
                    m, C_vec, v, image_cached, pc_enc, args.device, bs)[0]
            print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
            del m
            torch.cuda.empty_cache()

        print(f"\nSTEP 3  field ablation, untrained, seeds {seeds}. NOT A FIX -- "
              f"see MASTER's dev_val STRING-UNIQUENESS measurement.\n"
              f"{'variant':>20}{'text R@1':>11}{'increment pp':>14}"
              f"{'mean7 R@1':>12}{'text R@1 query-only':>21}")
        table, prev = {}, None
        for name in FIELD_VARIANTS:
            t1 = avg(per_seed, seeds, (name, "text", "R@1"))
            m7 = avg(per_seed, seeds, (name, "mean_R@1"))
            qo = avg(per_seed, seeds, (name + "|query_only", "text", "R@1"))
            inc = None if (prev is None or name.startswith("D_")) else (t1 - prev) * 100
            table[name] = {"text_R@1_both_sides": t1, "mean7_R@1_both_sides": m7,
                           "increment_pp": inc, "text_R@1_query_only": qo}
            col = f"{'-':>14}" if inc is None else f"{inc:+14.2f}"
            print(f"{name:>20}{t1:11.4f}{col}{m7:12.4f}{qo:21.4f}")
            if not name.startswith("D_"):
                prev = t1
        results["step3"] = {
            "what": ("six text serializations. `both_sides` swaps the serializer "
                     "for gallery AND query (S4 must then reproduce today's "
                     "numbers); `query_only` leaves the gallery canonical."),
            "not_a_fix": ("MASTER measured 4,569 distinct descriptions over 4,569 "
                          "dev_val assets -- STRING uniqueness, not embedding "
                          "behaviour. Removing a field cannot repair a shortcut "
                          "that runs on the shared cached embedding."),
            "templates": FIELD_VARIANTS,
            "per_variant": table,
            "per_seed": {str(s): per_seed[s] for s in seeds},
        }
        dump(3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
