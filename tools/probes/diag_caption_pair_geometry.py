#!/usr/bin/env python3
"""Raw-encoder GEOMETRY per modality, and THREE GALLERY ARMS over Protocol E.

The FILENAME IS HISTORICAL. This began as a text-only caption-pair probe; it now
covers all three modalities and the three gallery arms. The name is kept so
MASTER's work order and this file still name the same thing.

WHY IT EXISTS
-------------
Protocol E, INIT-0, dev_val: a TEXT-ONLY query finds its own asset out of 4,569
roughly 75% of the time after ZERO optimizer steps
(`output/look/diag_protocol_e_init0.json`, `E_independent_observation`). Two
separate questions follow, and this file answers both.

1. GEOMETRY -- is that separability already in the released encoder?
   Per modality, no fusion tower, no trained parameter:

       positive_i         = cos(query_i, gallery_i)
       hardest_negative_i = max_{j != i} cos(query_i, gallery_j)
       margin_i           = positive_i - hardest_negative_i

   A large margin establishes that the PRETRAINED representation already
   separates many assets BEFORE MetaFind's fusion runs -- that is, that this
   was not learned in Stage 1. It establishes nothing else.

   ⚠ [MASTER 2026-08-31, correcting a reading this file previously carried]
   A high margin does NOT show the score is "mostly literal overlap" and does
   NOT show it is illegitimate. Caption geometry cannot speak to cross-modal
   ability in either direction.

2. THREE GALLERY ARMS -- geometry alone cannot say how much of the score is
   same-modality. The gallery side can:

       SAME-MODALITY-ONLY   gallery keeps ONLY the query's own modality
       CROSS-MODAL-ONLY     gallery DROPS the query's own modality
       COMPLETE             all three modalities -- the real task

   ⚠ [Codex, via MASTER, 2026-08-31] The differences between arms are
   SENSITIVITY, nothing more. The fusion is nonlinear, so an arm value is NOT an
   additive contribution: it may not be written as "X% of the score comes
   from ...", and the arms may not be summed. Removing a modality is not even
   guaranteed to lower the score -- the paper's own Table 1 has `full` (51.7)
   BELOW `pc` (75.1), i.e. adding modalities hurting.

   CROSS-MODAL-ONLY is a DIAGNOSTIC and NOT A BOUND in either direction. It may
   not train, it may not select a checkpoint, and it is only ever reported
   alongside the complete-gallery number.

3. E_same vs E_cross ACROSS TRAINING -- THE HEADLINE, and the only test here
   that discriminates. [GPT via MASTER, 2026-08-31]

       E_same    text query -> gallery(T+I+P)   the complete-gallery task
       E_cross   text query -> gallery(I+P)     own modality removed
       and the image and pc equivalents

   PRE-REGISTERED READING, written before the trained numbers existed:

     training loss falls, E_same holds or rises, E_cross DOES NOT RISE or FALLS
         strong evidence that Stage 1 optimisation reduced the objective
         WITHOUT learning cross-modal retrieval -- it took the same-modality
         route.
     E_cross RISES with training
         the objective IS teaching cross-modal alignment and the shortcut is
         not what the optimiser took.

   Why this and not the identity cosine: a rise in cos(raw, fused) is neither
   sufficient (a genuinely learned shared space can also retain raw direction)
   nor necessary (a pure same-modality shortcut composed with a common rotation
   F(t) = R t shows a LOW cosine while preserving instance identity perfectly).
   `diag_untrained_fusion_identity.py --state ...` is SUPPORTING evidence only.

   ⚠ ALIGNED PROTOCOLS OR NOTHING. The untrained baseline this is compared
   against is `diag_text_shortcut.py` Step 4, which ran on PROD inputs -- query
   and gallery built from the SAME cached vectors. So the arms run on BOTH
   input sets here, and the trained-vs-untrained comparison is made WITHIN one
   input set, never across the two. Running INIT-0 on PROD also re-derives Step
   4's own numbers (text 0.9642 -> 0.3971, image 0.9103 -> 0.5029,
   pc 0.9452 -> 0.5666) from a second implementation, which is the check that
   this file builds the arms the same way that one did.

4. GALLERY MODALITY DERANGEMENT -- the CAUSAL test. [Codex via MASTER,
   2026-08-31, superseding E_same/E_cross as the headline: a cosine falling
   does not exclude a rotation that preserves instance identity, and a masked
   gallery changes the gallery's STRUCTURE as well as its content.]

   Query fixed. UID labels fixed. The gallery stays MODALITY-COMPLETE and goes
   through the production `model.gallery(...)` entry point. One gallery
   modality is permuted across UIDs by a DERANGEMENT -- a permutation with no
   fixed point, so no asset keeps its own vector -- and the other two stay
   correctly aligned. Repeated for text, image and pc, on INIT-0 and TRAINED.

       if a text-only query collapses ONLY when the gallery's TEXT is deranged
           strong evidence the model depends on the own-modality route
       if deranging image or pc hurts comparably
           it does not

   This is strictly better than the masking arms for the causal question: it
   changes WHICH asset a gallery modality describes without changing how many
   modalities the gallery has, so the fusion sees the same shape either way and
   the mask tokens never enter.

   Built with Sattolo's algorithm, which yields a single n-cycle and therefore
   has no fixed point BY CONSTRUCTION; the absence is also asserted, and the
   seed is recorded.

THE STANDING PROTOCOL IS UNCHANGED
----------------------------------
    Stage 1 primary = independent query observation + MODALITY-COMPLETE gallery
    training, checkpoint selection, reported R@1/R@5 = complete gallery, always

Nothing in this file changes that. Both non-complete arms exist only here.

TEXT IS MEASURED TWICE, AND THE PAIR IS THE POINT
--------------------------------------------------
    text_bare        C = annotation["description"]
                     A = the highest-ranked NON-canonical description candidate
                     -- the raw strings, NO template.
    text_templated   both put through `serialize_annotation`, which is what the
                     model actually consumes.

The bare pair is the fix for this file's original bug: it used to serialize BOTH
sides, so C and A shared an identical category, materials, dimension string and
placement clause, and the measured similarity was dominated by the template
rather than by the description. `text_bare` isolates the description; the gap
between the two rows is how much of the text margin the template supplies.

`text_templated` is also what the ARMS consume, so the two halves of this file
are computed on the same vectors.

⚠ THE BARE PATH GOES AROUND A PRODUCTION GATE. n06's encode path calls
`refuse_if_overlong` (`encode_text_image.py`), and `serialize_annotation` caps
the description, so a templated string never exceeds CLIP's 77-token context.
A BARE description can: measured on dev_val, 14 of 4,569 canonical and 7 of
4,569 alternate descriptions exceed 77 tokens and are silently truncated by
`open_clip`'s tokenizer. The counts are recomputed and reported every run
rather than dropped, because dropping them would change the gallery size.

TWO INPUT SETS
--------------
    PROD_shared_observation      query and gallery are the SAME vectors --
                                 what `evaluate_dev_val` scores, and what the
                                 untrained E_same/E_cross baseline used.
    E_independent_observation    query and gallery observe the same asset
                                 through DIFFERENT samples (below).

INPUTS -- Protocol E's, unchanged
----------------------------------
    GALLERY   text  cached canonical serialization (the vector on disk)
              image mean of the ELEVEN views != held-out
              pc    the canonical 10,000-point sample (read-only)
    QUERY     text  the highest-ranked NON-canonical description, serialized
              image the ONE held-out view, index sha256(uid) % 12
              pc    a SECOND 10,000-point sample, seed offset +1_000_003, built
                    by `tools/probes/build_protocol_e_query_pc.py`

`heldout_index` and `collect` are IMPORTED from
`diag_protocol_e_ulip_fingerprint` rather than restated, so the two files cannot
drift into building different queries under the same name.

`--state <checkpoint>` runs the trained side. The checkpoint is loaded BEFORE
`collect`, because `train_scope` is `point_encoder_and_fuser` and a Stage 1
checkpoint restores a fine-tuned PointBERT as well as the towers; collecting
first would score trained towers against released-encoder point clouds. One
consequence: the `pc` GEOMETRY row is not the same measurement in the two runs
(its inputs moved), while `text` and `image` are -- OpenCLIP is frozen.

DEVICE
------
NOT CPU-ONLY, and it was wrong to describe an earlier version as such.
[MASTER 2026-08-31: "GPU is allowed and expected here."]

    text    GPU -- four batch-1 encode passes over 4,569 strings
    pc      GPU -- PointBERT runs on both draws; neither is cached
    image   CPU -- the per-view embeddings are cached, so this arm really is CPU

WHAT IT WRITES
--------------
Two JSON files under `output/look/`, and MODIFIES NO CANONICAL ARTIFACT. It
reads the annotations, the embedding cache, the canonical point clouds and the
`_probe/` query clouds; it writes none of them, and it touches no checkpoint,
no split, no protocol artifact and no trainer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metafind import paths                                            # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,  # noqa: E402
                                     normalize_for_scoring, rank_of_target,
                                     recall_at_k)
from metafind.models.fusion import MODALITIES                         # noqa: E402
from metafind.models.resolve_stage1 import serialize_annotation       # noqa: E402

from diag_text_shortcut import encode_texts                           # noqa: E402
from diag_protocol_e_ulip_fingerprint import (N_VIEWS, QPC, collect,  # noqa: E402
                                              heldout_index, sha)

LOOK = REPO / "output" / "look"
CONDS = tuple(QUERY_CONDITIONS)
ARMS = ("same_modality_only", "cross_modal_only", "complete")
PCTS = (1, 5, 25, 50, 75, 95, 99)


def q(x: np.ndarray, name: str) -> dict:
    p = np.percentile(x, PCTS)
    return {"name": name, "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            **{f"p{k}": float(v) for k, v in zip(PCTS, p)}}


def geometry(Q: np.ndarray, G: np.ndarray, label: str) -> dict:
    """Positive, hardest negative, margin and raw-encoder R@1/R@5 for one modality.

    Row i of `Q` and row i of `G` are the same asset, so the positive is the
    diagonal. The gallery is the full split -- 4,569 wide -- for every modality.
    """
    Qn, Gn = normalize_for_scoring(Q), normalize_for_scoring(G)
    sim = Qn @ Gn.T
    d = np.arange(sim.shape[0])
    pos = sim[d, d]
    off = sim.copy()
    off[d, d] = -np.inf
    neg = off.max(axis=1)
    margin = pos - neg
    r = rank_of_target(sim, d)
    frac_pos = float((margin > 0).mean())
    rec = recall_at_k(sim, d, ks=(1, 5))
    # `margin > 0` means no other gallery column scores >= the positive, and
    # `rank_of_target` counts ties AGAINST the model, so rank == 1 is the same
    # condition. Exact equality, not a tolerance: they are two fractions of the
    # same integer count. This is the indexing check for the block above.
    assert frac_pos == rec["R@1"], \
        f"{label}: frac(margin>0) {frac_pos} != R@1 {rec['R@1']}"
    return {"positive": q(pos, "cos(query_i, gallery_i)"),
            "hardest_negative": q(neg, "max_{j!=i} cos(query_i, gallery_j)"),
            "margin": q(margin, "positive - hardest_negative"),
            "frac_margin_gt_0": frac_pos,
            "raw_encoder_R@1": rec["R@1"], "raw_encoder_R@5": rec["R@5"],
            "median_rank": float(np.median(r)),
            "n_query": int(sim.shape[0]), "n_gallery": int(sim.shape[1])}


def arm_gallery_flags(arm: str, cond: tuple[bool, bool, bool]):
    if arm == "complete":
        return (True,) * len(MODALITIES)
    if arm == "same_modality_only":
        return cond
    if arm == "cross_modal_only":
        return tuple(not f for f in cond)
    raise ValueError(arm)


def raw_mean(mats: dict, flags) -> np.ndarray | None:
    """Parameter-free mean over the modalities `flags` makes present.

    `None` when no modality is present -- which happens for exactly one cell,
    CROSS-MODAL-ONLY under `full`, where the mean of an empty set is not a
    number and reporting a zero there would be an invented value.
    """
    keep = [m for m, f in zip(MODALITIES, flags) if f]
    return None if not keep else np.mean([mats[m] for m in keep], axis=0)


def arms_raw(gal: dict, qry: dict) -> dict:
    """The three arms with NO FUSION and no parameter of any kind."""
    out: dict = {a: {} for a in ARMS}
    n = next(iter(gal.values())).shape[0]
    tgt = np.arange(n)
    for c in CONDS:
        qv = normalize_for_scoring(raw_mean(qry, QUERY_CONDITIONS[c]))
        for a in ARMS:
            flags = arm_gallery_flags(a, QUERY_CONDITIONS[c])
            gv = raw_mean(gal, flags)
            if gv is None:
                out[a][c] = {"R@1": None, "R@5": None,
                             "undefined": ("the gallery keeps zero modalities; "
                                           "the mean of an empty set is not a "
                                           "vector")}
                continue
            cell = recall_at_k(qv @ normalize_for_scoring(gv).T, tgt, ks=(1, 5))
            cell["gallery_modalities"] = [m for m, f in zip(MODALITIES, flags) if f]
            out[a][c] = cell
    return out


def fuse_gallery(model, gal: dict, flags, device, bs):
    """One gallery matrix under one presence mask.

    A COMPLETE mask goes through `model.gallery(...)`, the production entry
    point, unchanged. Any other mask goes through `model.gallery.fusion(...)`
    directly, because `GalleryTower.forward` refuses an incomplete gallery by
    design (`dual_tower.py:175-180`) -- the gallery is modality-complete in the
    paper's construction, so masking it is a DIAGNOSTIC bypass of a guard that
    is doing its job, not a defect in the guard. Absent slots carry the GALLERY
    fusion's own learned mask tokens (`tower_sharing` is
    `shared_backbone_separate_fusion`, so those are not the query's tokens).
    """
    import torch

    from metafind.train.stage1 import modules_in_eval

    n = gal["text"].size(0)
    complete = all(flags)
    out = []
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            b = sl.stop - sl.start
            g = {m: gal[m][sl].to(device) for m in MODALITIES}
            if complete:
                v = model.gallery(g)
            else:
                mask = torch.tensor(flags, dtype=torch.bool, device=device
                                    ).expand(b, len(MODALITIES))
                v = model.gallery.fusion(g, present=mask)
            out.append(v.float().cpu())
    return torch.cat(out)


def fuse_query(model, qry: dict, cond: str, device, bs):
    import torch

    from metafind.train.stage1 import modules_in_eval

    n = qry["text"].size(0)
    out = []
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            b = sl.stop - sl.start
            e = {m: qry[m][sl].to(device) for m in MODALITIES}
            out.append(model.query(
                e, present=condition_mask(cond, b).to(device)).float().cpu())
    return torch.cat(out)


def arms_fused(model, gal: dict, qry: dict, device, bs) -> dict:
    """The three arms through the towers, one forward per DISTINCT mask.

    The 7 conditions and 3 arms name 21 cells but only 8 distinct gallery masks
    (every subset of the three modalities) and 7 distinct query vectors, so 15
    tower passes cover all 21 cells and the rest is GEMMs.
    """
    masks = sorted({arm_gallery_flags(a, QUERY_CONDITIONS[c])
                    for a in ARMS for c in CONDS})
    G = {f: normalize_for_scoring(fuse_gallery(model, gal, f, device, bs).numpy())
         for f in masks}
    Q = {c: normalize_for_scoring(fuse_query(model, qry, c, device, bs).numpy())
         for c in CONDS}
    tgt = np.arange(next(iter(G.values())).shape[0])
    out: dict = {a: {} for a in ARMS}
    for a in ARMS:
        for c in CONDS:
            f = arm_gallery_flags(a, QUERY_CONDITIONS[c])
            cell = recall_at_k(Q[c] @ G[f].T, tgt, ks=(1, 5))
            cell["gallery_modalities"] = [m for m, k in zip(MODALITIES, f) if k]
            if not any(f):
                # Every slot absent, so the gallery is the fused mask tokens --
                # the SAME vector for every asset. It carries no asset identity
                # at all. Reported because a constant gallery scoring R@1 = 0 is
                # the tie mechanism working, not because the number measures
                # retrieval.
                cell["degenerate_constant_gallery"] = True
            out[a][c] = cell
    return out


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation of ``range(n)`` with NO fixed point.

    Sattolo's algorithm -- the inner draw is strictly BELOW the current index,
    not at-or-below as in Fisher-Yates -- produces a single n-cycle, so every
    element moves by construction rather than by rejection sampling. The caller
    still asserts it: "by construction" is what the Fisher-Yates version also
    looked like.
    """
    if n < 2:
        raise ValueError("a derangement needs n >= 2")
    rng = np.random.default_rng(seed)
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))          # 0 <= j < i, never j == i
        p[i], p[j] = p[j], p[i]
    return p


def derange_raw(gal: dict, qry: dict, perm: np.ndarray) -> dict:
    """The derangement with NO fusion and no parameter of any kind."""
    out = {}
    for name in ("aligned",) + tuple(f"derange_{m}" for m in MODALITIES):
        g = dict(gal)
        if name != "aligned":
            g[name[len("derange_"):]] = gal[name[len("derange_"):]][perm]
        G = np.mean([g[m] for m in MODALITIES], axis=0)
        out[name] = {c: geometry(np.mean([qry[m] for m, f
                                          in zip(MODALITIES, QUERY_CONDITIONS[c])
                                          if f], axis=0), G, f"{name}/{c}")
                     for c in CONDS}
    return out


def derange_fused(model, gal: dict, qry: dict, perm, device, bs) -> dict:
    """The derangement through the towers.

    The gallery stays MODALITY-COMPLETE and goes through `model.gallery(...)`,
    the production entry point -- unlike the masking arms, nothing here bypasses
    `GalleryTower.forward` and no mask token is ever used. Only the CONTENT of
    one gallery modality is re-pointed at other assets.

    The query is fused once per condition and reused across all four gallery
    variants, because the query side is identical in every one of them.
    """
    Q = {c: normalize_for_scoring(fuse_query(model, qry, c, device, bs).numpy())
         for c in CONDS}
    complete = (True,) * len(MODALITIES)
    out = {}
    for name in ("aligned",) + tuple(f"derange_{m}" for m in MODALITIES):
        g = dict(gal)
        if name != "aligned":
            m = name[len("derange_"):]
            g[m] = gal[m][perm]
        G = fuse_gallery(model, g, complete, device, bs).numpy()
        out[name] = {c: geometry(Q[c], G, f"{name}/{c}") for c in CONDS}
    return out


def gallery_path_equivalence(model, gal, device, bs) -> float:
    """`model.gallery(g)` vs `model.gallery.fusion(g, present=all-True)`.

    The arms reach the gallery through the fusion module directly. This shows
    that on a COMPLETE mask the bypass IS the production path, so the only
    difference between the arms is the mask -- checked on real inputs rather
    than argued from reading `GalleryTower.forward`.
    """
    import torch

    from metafind.train.stage1 import modules_in_eval

    with modules_in_eval(model), torch.no_grad():
        g = {m: gal[m][:bs].to(device) for m in MODALITIES}
        b = g["text"].size(0)
        a = model.gallery(g)
        mask = torch.ones(b, len(MODALITIES), dtype=torch.bool, device=device)
        c = model.gallery.fusion(g, present=mask)
    return float((a - c).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3", help="init0 only")
    ap.add_argument("--state", default="init0",
                    help="'init0' (untrained, --seeds draws) or a Stage 1 "
                         "checkpoint path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--derange-seed", type=int, default=20260831,
                    help="seed for the gallery derangement")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: first N dev_val assets. Debug, not evidence.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    trained = args.state != "init0"

    import torch

    from metafind.data.encode_text_image import true_token_count
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    dev_val = json.loads(
        (paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    if args.limit:
        dev_val = dev_val[:args.limit]
        print("⚠ --limit set: DEBUG RUN. These numbers are not evidence.", flush=True)

    man = json.loads(next(QPC.glob("*.manifest.json")).read_text())
    if man["debug_limit"] is not None and not args.limit:
        raise SystemExit(f"{man['array']} was built with --limit "
                         f"{man['debug_limit']}; rebuild it in full.")
    qpc_all = np.load(man["array"], mmap_mode="r")
    pos = {u: k for k, u in enumerate(man["uid_order"])}
    if missing := [u for u in dev_val if u not in pos]:
        raise SystemExit(f"query PC missing for {len(missing)} uid(s)")

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model_t, ckpt_sha = None, None
    if trained:
        # BEFORE `collect`. `train_scope` is `point_encoder_and_fuser`, so the
        # checkpoint restores a fine-tuned PointBERT as well as the towers.
        from metafind.train.stage1 import load_stage1_checkpoint
        model_t, loss_fn = build_model(encoding, training, hyper)
        model_t.to(args.device)
        load_stage1_checkpoint(backbone, model_t, loss_fn, Path(args.state))
        ckpt_sha = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        print(f"loaded {args.state}  sha256 {ckpt_sha[:16]}…", flush=True)
    t0 = time.time()
    order, C_tpl, img_cached, P_canon, views = collect(
        backbone, dev_val, encoding["image_aggregation"], args.device, bs)
    assert order == dev_val, "loader reordered the split"
    print(f"inputs collected in {time.time()-t0:.0f}s  views {views.shape}",
          flush=True)

    # ---- IMAGE arm. CPU: the per-view embeddings are cached. ---------------
    held = np.array([heldout_index(u) for u in order])
    v32 = views.astype(np.float32)
    rows = np.arange(len(order))
    v_held = v32[rows, held]
    keep = np.array([[j for j in range(N_VIEWS) if j != h] for h in held])
    img11 = v32[rows[:, None], keep].mean(axis=1)
    # The held-out view must not be inside the gallery average. GATHERED, not
    # `(sum - held)/11`: the subtraction reads the held-out value into the
    # arithmetic. Exact equality, not a tolerance.
    probe = v32.copy()
    probe[rows, held] = 12345.0
    a_img = float(np.abs(probe[rows[:, None], keep].mean(axis=1) - img11).max())
    assert a_img == 0.0, f"gallery mean depends on the held-out view: {a_img}"
    del probe
    print(f"IMAGE arm  held-out view excluded, perturbation residual {a_img:.1e}  OK",
          flush=True)

    # ---- TEXT arms. Bare AND templated. -----------------------------------
    ann = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in order]
    bare_C, bare_A, tpl_C, tpl_A, alt_rank = [], [], [], [], []
    for a in ann:
        c = a["description"]
        cands = [x for x in (a.get("description_candidates") or [])
                 if x["text"] != c]
        if not cands:
            raise SystemExit(f"{a['uid']}: no non-canonical description candidate")
        best = min(cands, key=lambda x: x["rank"])
        bare_C.append(c)
        bare_A.append(best["text"])
        tpl_C.append(serialize_annotation(a))
        tpl_A.append(serialize_annotation({**a, "description": best["text"]}))
        alt_rank.append(best["rank"])
    assert all(x != y for x, y in zip(bare_A, bare_C)), "an alternate equals its canonical"
    assert all(x != y for x, y in zip(tpl_A, tpl_C)), "an alternate serialized to the canonical"

    # The bare path bypasses n06's `refuse_if_overlong`, so count what CLIP
    # will silently truncate instead of letting it pass unrecorded.
    over = {k: int(sum(true_token_count(s) > 77 for s in v))
            for k, v in (("bare_C", bare_C), ("bare_A", bare_A),
                         ("templated_C", tpl_C), ("templated_A", tpl_A))}
    print(f"TEXT  strings over CLIP's 77-token context (silently truncated): "
          f"{over}", flush=True)
    # Two different assets sharing one alternate string are indistinguishable to
    # a single-positive R@1 in `text_bare`. Counted, not assumed absent.
    dup_bare_A = len(bare_A) - len(set(bare_A))
    dup_bare_C = len(bare_C) - len(set(bare_C))
    print(f"TEXT  duplicate bare strings -- canonical {dup_bare_C}, "
          f"alternate {dup_bare_A}", flush=True)

    A_tpl = encode_texts(backbone, tpl_A, "(templated alternate)")
    # GATE: the harness must reproduce the CACHED canonical vector bit for bit,
    # or "the only difference is the description" is not true of the arms.
    re_C = encode_texts(backbone, tpl_C, "(templated canonical, for the gate)")
    gate = bool(torch.equal(re_C, C_tpl))
    print(f"GATE  re-encoded templated canonical == cached text, bit-identical: "
          f"{gate}", flush=True)
    if not gate:
        raise SystemExit("GATE FAILED: the harness cannot reproduce the cache. STOP.")
    del re_C
    n_diff = int((A_tpl != C_tpl).any(dim=1).sum())
    assert n_diff == len(order), f"only {n_diff}/{len(order)} alternate vectors differ"

    C_bare = encode_texts(backbone, bare_C, "(bare canonical)")
    A_bare = encode_texts(backbone, bare_A, "(bare alternate)")

    # ---- PC arm. GPU: neither draw is cached. ------------------------------
    qpc = np.asarray(qpc_all[[pos[u] for u in order]])
    n_same = int(sum(np.array_equal(
        qpc[k, :, :3], np.load(paths.POINTCLOUDS / f"{order[k]}.npz")["xyz"])
        for k in range(len(order))))
    assert n_same == 0, f"{n_same} query clouds are byte-identical to the canonical"
    P_query = []
    with torch.no_grad():
        for i in range(0, len(order), bs):
            P_query.append(backbone.encode_pc(
                torch.from_numpy(qpc[i:i + bs]).to(args.device)).float().cpu())
    P_query = torch.cat(P_query)
    print(f"PC arm  {len(order)} independent redraws, 0 byte-identical  OK", flush=True)

    # ---- PART 1: GEOMETRY --------------------------------------------------
    f64 = lambda x: (x.numpy() if isinstance(x, torch.Tensor) else x).astype(np.float64)
    geo = {
        "text_bare": geometry(f64(A_bare), f64(C_bare), "text_bare"),
        "text_templated": geometry(f64(A_tpl), f64(C_tpl), "text_templated"),
        "image": geometry(f64(v_held), f64(img11), "image"),
        "pc": geometry(f64(P_query), f64(P_canon), "pc"),
    }
    print(f"\nGEOMETRY -- raw released encoder, no fusion, gallery {len(order):,}")
    print(f"{'modality':>16} {'pos(p50)':>10} {'hard-neg(p50)':>14} "
          f"{'margin(p50)':>12} {'frac>0':>8} {'R@1':>7} {'R@5':>7}")
    for k, v in geo.items():
        print(f"{k:>16} {v['positive']['p50']:10.4f} "
              f"{v['hardest_negative']['p50']:14.4f} {v['margin']['p50']:12.4f} "
              f"{v['frac_margin_gt_0']:8.4f} {v['raw_encoder_R@1']:7.4f} "
              f"{v['raw_encoder_R@5']:7.4f}")

    # ---- PART 2: THREE ARMS, on BOTH input sets ----------------------------
    ten = torch.from_numpy
    INPUTS = {
        # Query and gallery are the SAME vectors. This is what `evaluate_dev_val`
        # scores and what the untrained E_same/E_cross baseline used, so it is
        # the set the trained-vs-untrained comparison is made within.
        "PROD_shared_observation": (
            {"text": C_tpl, "image": img_cached, "pc": P_canon},
            {"text": C_tpl, "image": img_cached, "pc": P_canon}),
        "E_independent_observation": (
            {"text": C_tpl, "image": ten(img11), "pc": P_canon},
            {"text": A_tpl, "image": ten(v_held), "pc": P_query}),
    }

    def show(tag, cells):
        print(f"\n{tag}")
        print(f"{'arm':>20} " + "".join(f"{c:>12}" for c in CONDS))
        for a in ARMS:
            row = ""
            for c in CONDS:
                r = cells[a][c]["R@1"]
                row += f"{'--':>12}" if r is None else f"{r:12.4f}"
            print(f"{a:>20} " + row)

    arms = {k: {} for k in INPUTS}
    for k, (gal, qry) in INPUTS.items():
        arms[k]["raw_no_fusion"] = arms_raw(
            {m: f64(gal[m]) for m in MODALITIES},
            {m: f64(qry[m]) for m in MODALITIES})
        show(f"ARMS  {k}  raw / no fusion  R@1", arms[k]["raw_no_fusion"])

    states = [("trained", None)] if trained else [(f"INIT-0_seed{s}", s)
                                                  for s in seeds]
    equiv = None
    for label, seed in states:
        if seed is None:
            model = model_t
        else:
            torch.manual_seed(seed)
            model, _ = build_model(encoding, training, hyper)
            model.to(args.device)
        if equiv is None:
            equiv = gallery_path_equivalence(
                model, INPUTS["E_independent_observation"][0], args.device, bs)
            print(f"\nGALLERY PATH  max|model.gallery(g) - "
                  f"model.gallery.fusion(g, present=all-True)| = {equiv:.3e}",
                  flush=True)
            assert equiv == 0.0, "the masked bypass is not the production path"
        for k, (gal, qry) in INPUTS.items():
            arms[k][label] = arms_fused(model, gal, qry, args.device, bs)
            show(f"ARMS  {k}  {label}  R@1", arms[k][label])
        if seed is not None:
            del model
        torch.cuda.empty_cache()

    # ---- THE HEADLINE: E_same vs E_cross, per input set, per state ---------
    headline = {k: {lab: {c: {
        "E_same_complete_gallery": arms[k][lab]["complete"][c]["R@1"],
        "E_cross_own_modality_removed": arms[k][lab]["cross_modal_only"][c]["R@1"],
        "E_sameonly_gallery_is_own_modality": arms[k][lab]["same_modality_only"][c]["R@1"],
    } for c in ("text", "image", "pc")} for lab in arms[k]} for k in INPUTS}
    for k in INPUTS:
        print(f"\nHEADLINE  {k}   E_same -> E_cross   R@1")
        print(f"{'state':>18}" + "".join(f"{c:>26}" for c in ("text", "image", "pc")))
        for lab in arms[k]:
            row = ""
            for c in ("text", "image", "pc"):
                h = headline[k][lab][c]
                row += (f"{h['E_same_complete_gallery']:11.4f} ->"
                        f"{h['E_cross_own_modality_removed']:11.4f}  ")
            print(f"{lab:>18}" + row)

    # ---- PART 3: GALLERY MODALITY DERANGEMENT ------------------------------
    perm = derangement(len(order), args.derange_seed)
    n_fixed = int((perm == np.arange(len(order))).sum())
    assert n_fixed == 0, f"the permutation has {n_fixed} fixed point(s)"
    assert len(set(perm.tolist())) == len(order), "not a permutation"
    print(f"\nDERANGEMENT  seed {args.derange_seed}, {len(order):,} elements, "
          f"fixed points {n_fixed}  OK", flush=True)

    der: dict = {k: {} for k in INPUTS}
    for k, (gal, qry) in INPUTS.items():
        der[k]["raw_no_fusion"] = derange_raw(
            {m: f64(gal[m]) for m in MODALITIES},
            {m: f64(qry[m]) for m in MODALITIES}, perm)
    for label, seed in states:
        if seed is None:
            model = model_t
        else:
            torch.manual_seed(seed)
            model, _ = build_model(encoding, training, hyper)
            model.to(args.device)
        for k, (gal, qry) in INPUTS.items():
            der[k][label] = derange_fused(model, gal, qry, perm, args.device, bs)
        if seed is not None:
            del model
        torch.cuda.empty_cache()

    VAR = ("aligned",) + tuple(f"derange_{m}" for m in MODALITIES)
    for k in INPUTS:
        print(f"\nDERANGEMENT  {k}   R@1 (margin p50)   "
              f"-- own modality marked <<>>")
        print(f"{'state':>18} {'query':>7} " + "".join(f"{v:>22}" for v in VAR))
        for lab in der[k]:
            for c in ("text", "image", "pc"):
                row = ""
                for v in VAR:
                    z = der[k][lab][v][c]
                    own = v == f"derange_{c}"
                    cell = (f"{z['raw_encoder_R@1']:.4f} "
                            f"({z['margin']['p50']:+.4f})")
                    row += f"{('<<'+cell+'>>') if own else cell:>22}"
                print(f"{lab:>18} {c:>7} " + row)

    # ---- output ------------------------------------------------------------
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    enc_sha = hashlib.sha256(backbone.cfg.checkpoint.read_bytes()).hexdigest()

    tag = "init0" if not trained else (Path(args.state).parent.name
                                       or Path(args.state).stem)
    out_path = LOOK / f"diag_caption_pair_geometry_{tag}.json"
    prov_path = LOOK / f"diag_caption_pair_geometry_{tag}_provenance.json"
    LOOK.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps({
        "split": "dev_val", "n_assets": len(order), "uid_order": order,
        "state": "trained" if trained else "INIT-0 (zero optimizer steps)",
        "seeds": None if trained else seeds,
        "stage1_checkpoint": args.state if trained else None,
        "stage1_checkpoint_sha256": ckpt_sha,
        "encoder": str(backbone.cfg.checkpoint), "encoder_sha256": enc_sha,
        "text_template_applied_to": ["text_templated", "all three arms"],
        "text_bare_applied_to": ["text_bare geometry only"],
        "query_pc_seed_offset": man["seed_offset"],
        "query_pc_array": man["array"],
        "query_pc_array_sha256": man["array_sha256"],
        "code_revision": rev, "code_dirty": dirty,
        "per_uid": {u: {
            "canonical_text_bare_sha256": sha(bare_C[k]),
            "alternate_text_bare_sha256": sha(bare_A[k]),
            "canonical_text_templated_sha256": sha(tpl_C[k]),
            "alternate_text_templated_sha256": sha(tpl_A[k]),
            "alternate_text_rank": int(alt_rank[k]),
            "heldout_view_index": int(held[k]),
            "heldout_view_sha256": sha(views[k, held[k]]),
            "gallery_view_indices": [int(j) for j in range(N_VIEWS) if j != held[k]],
            "query_pc_sha256": man["query_pc_sha256_per_uid"][u],
            "canonical_pc_npz_sha256": man["canonical_pc_npz_sha256_per_uid"][u],
        } for k, u in enumerate(order)},
    }, indent=1))

    out_path.write_text(json.dumps({
        "what": ("Raw-encoder geometry per modality, plus three gallery arms "
                 "over Protocol E's inputs, at INIT-0."),
        "NOT_THE_PAPER_PROTOCOL": ("Protocol E is a diagnostic implementation "
                                   "choice, not MetaFind's evaluation "
                                   "construction. The two non-complete gallery "
                                   "arms exist only in this diagnostic."),
        "standing_protocol_unchanged": ("Stage 1 primary = independent query "
                                        "observation + MODALITY-COMPLETE "
                                        "gallery. Training, checkpoint "
                                        "selection and every reported R@1/R@5 "
                                        "use the complete gallery, always."),
        "caveat": ("dev_val is a 4,569-asset HPO selector split. It is NOT the "
                   "paper's 20% test population and no number here may be set "
                   "against Table 1."),
        "how_to_read_the_geometry": (
            "A large margin establishes that the PRETRAINED representation "
            "already separates many assets BEFORE MetaFind's fusion runs -- "
            "i.e. that this was not learned in Stage 1. It does NOT establish "
            "that the separation is illegitimate, and caption geometry cannot "
            "speak to cross-modal ability in either direction."),
        "how_to_read_the_arms": (
            "SENSITIVITY, nothing more. The fusion is nonlinear, so an arm "
            "value is NOT an additive contribution: it may not be expressed as "
            "'X% of the score comes from ...' and the arms may not be summed. "
            "Removing a modality is not guaranteed to lower the score -- the "
            "paper's Table 1 has `full` (51.7) below `pc` (75.1). "
            "CROSS-MODAL-ONLY is a DIAGNOSTIC and NOT A BOUND in either "
            "direction; it may not train, may not select a checkpoint, and is "
            "only ever reported alongside the complete-gallery number."),
        "text_bare_vs_templated": (
            "text_bare uses the raw description strings on BOTH sides, so the "
            "template's shared category / materials / dimensions / placement "
            "cannot carry the match. text_templated is what the model consumes "
            "and what all three arms use."),
        "bare_path_bypasses_the_token_gate": {
            "gate": "encode_text_image.refuse_if_overlong, 77-token CLIP context",
            "strings_over_77_tokens_silently_truncated": over,
            "note": ("kept rather than dropped: dropping them would change the "
                     "gallery size"),
        },
        "duplicate_bare_strings": {"canonical": dup_bare_C, "alternate": dup_bare_A,
                                   "note": ("two assets sharing one string are "
                                            "indistinguishable to a "
                                            "single-positive R@1")},
        "assertions": {
            "heldout_view_perturbation_residual": a_img,
            "gate_templated_canonical_bit_identical": gate,
            "alternate_text_vectors_differing": n_diff,
            "query_clouds_byte_identical_to_canonical": n_same,
            "gallery_masked_bypass_equals_production_path_maxabs": equiv,
        },
        "headline_E_same_vs_E_cross": {
            "definition": ("E_same = query -> COMPLETE gallery (the real task); "
                           "E_cross = query -> gallery with the query's own "
                           "modality removed (DIAGNOSTIC, not a bound)."),
            "prereg_reading": {
                "E_cross_does_not_rise_or_falls": (
                    "with the loss falling and E_same holding or rising, strong "
                    "evidence that Stage 1 optimisation reduced the objective "
                    "WITHOUT learning cross-modal retrieval -- it took the "
                    "same-modality route"),
                "E_cross_rises": ("the objective IS teaching cross-modal "
                                  "alignment and the shortcut is not what the "
                                  "optimiser took"),
            },
            "protocol_alignment": ("trained-vs-untrained is compared WITHIN one "
                                   "input set. The untrained baseline "
                                   "(diag_text_shortcut Step 4) used PROD "
                                   "inputs, so PROD is the comparable set."),
            "values": headline,
        },
        "derangement": {
            "what": ("Query fixed, UID labels fixed, gallery MODALITY-COMPLETE "
                     "and through the production model.gallery(...) entry "
                     "point. One gallery modality is permuted across UIDs by a "
                     "derangement; the other two stay aligned."),
            "why_this_and_not_the_masking_arms": (
                "it changes WHICH asset a gallery modality describes without "
                "changing how many modalities the gallery has, so the fusion "
                "sees the same shape either way and no mask token is used"),
            "reading": ("if a text-only query collapses ONLY when the "
                        "gallery's text is deranged, that is strong evidence "
                        "the model depends on the own-modality route; if "
                        "deranging image or pc hurts comparably, it does not"),
            "seed": args.derange_seed,
            "algorithm": ("Sattolo -- a single n-cycle, so no fixed point by "
                          "construction; asserted as well"),
            "fixed_points": n_fixed,
            "values": der,
        },
        "n_assets": len(order), "batch_size": bs, "debug_limit": args.limit,
        "seeds": None if trained else seeds,
        "state": "trained" if trained else "INIT-0 (zero optimizer steps)",
        "checkpoint": args.state if trained else None,
        "checkpoint_sha256": ckpt_sha,
        "encoder_sha256": enc_sha,
        "code_revision": rev, "code_dirty": dirty,
        "geometry": geo,
        "arms": arms,
    }, indent=1, default=float))
    print(f"\nwrote {out_path}\nwrote {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
