#!/usr/bin/env python3
"""Why does our ULIP baseline score 51.2% text where every published source says 0.1-4.5%?

WHAT IS ALREADY KNOWN, AND IS NOT IN DISPUTE
---------------------------------------------
`output/look/diag_ulip_fingerprint.json` (rev 7734f06, dev_val, 4,569) already
ran the ULIP baseline the paper's way -- gallery = the point-cloud embedding
alone [PAPER FACT 3experiments.tex:24], query = "a simple mean pooling layer"
over the available modalities [PAPER FACT :15], no MetaFind tower anywhere:

    condition   ours   MetaFind Tab.1   CLIP-GS's ULIP-2
    text        51.2        0.1              4.5
    image       81.3        0.1              5.6
    pc         100.0       97.9               --

`pc` lands on the paper. So the scorer, the UID pairing and the gallery
construction are not what is wrong -- a pc query IS its own gallery entry and
the measurement says so. `text` is off by 512x with no tower, no training and
no shared observation involved. The excess is in the RAW ULIP-2 embeddings.

FOUR CANDIDATE CAUSES. THIS PROBE PRICES THREE OF THEM FOR FREE.
-----------------------------------------------------------------
1. QUERY TEXT RICHNESS. ULIP-2 pretrains on a BLIP-2 caption of one view
   [UPSTREAM FACT ulip2 main.tex:616] -- short and generic. Our gallery text is
   a structured GPT-4o description carrying colour, pattern, material and
   "roughly 12.1 by 9.4 by 3.5 centimetres". Against 4,569 candidates that is
   close to a fingerprint. What MetaFind fed ITS baseline is UNKNOWN; the paper
   never says. Priced by a three-rung ladder, below.
2. IMAGE OBSERVATION. ULIP-2 samples ONE random render per step. Our `image` is
   the mean of twelve. A twelve-view mean is cleaner than anything the encoder
   saw in training. Priced by scoring a single view.
3. GALLERY SIZE [U-09]. Ours is 4,569; the paper never states its own. Priced
   by rescoring the same queries against 9,138 and 45,692.
4. CHECKPOINT CONTAMINATION. Our checkpoint is the `objaverse_shapenet` release
   [OBSERVED IMPLEMENTATION paths.py:45], i.e. ULIP-2's "Objaverse + ShapeNet"
   row, whose pretraining pool includes every LVIS asset we evaluate on. That is
   a real DEVIATION and it is recorded. It is NOT priced here, and the arithmetic
   argues it is not the main term: ULIP-2's own table puts clean vs contaminated
   at 46.3 vs 50.6 zero-shot top-1 [UPSTREAM FACT ulip2 main.tex:434-444], and
   memorisation worth 4.3 points of classification does not buy 512x of instance
   retrieval. Pricing it costs a full ULIP-2 pretrain on 744,860 assets; causes
   1-3 cost one encoding pass, so they go first.

AND THE GATE NOBODY HAS RUN
----------------------------
ULIP-2 zero-shot classification on Objaverse-LVIS. Our checkpoint's published
number is top-1 50.6 / top-5 79.1 [UPSTREAM FACT ulip2 main.tex:443]. It is the
only figure that tests the BACKBONE rather than anything we built on it, and
`grep -r zeroshot metafind/` finds nothing. Upstream's recipe is copied, not
reinvented: 1,156 `all_keys` classes, the 64 `modelnet40_64` templates that
`main.py:41`'s default selects and the official LVIS script does not override,
each prompt L2'd, meaned, then L2'd again (`main.py:377-380`).

WHAT THIS IS NOT
----------------
Diagnostic. No training, no checkpoint, no protocol artifact, no write under
`data/`. Point-cloud embeddings are cached to the session scratchpad so the
rescoring ladders are free to re-run. One json under `output/look/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths                                      # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS,          # noqa: E402
                                     normalize_for_scoring, rank_of_target)

UPSTREAM = Path("/home/kyzen/upstream/ULIP")
OUT = REPO / "output" / "look" / "ulip2_calibration.json"
SCRATCH = Path("/tmp/claude-1002/-home-kyzen-MetaFindV1"
               "/ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad")

PAPER_ZEROSHOT = {"top1": 50.6, "top5": 79.1}          # our checkpoint's row
CLEAN_ZEROSHOT = {"top1": 46.3, "top5": 75.0}          # the no-LVIS row
PAPER_ULIP_ROW = {"text": 0.1, "image": 0.1, "pc": 97.9, "text+image": 0.0,
                  "text+pc": 33.9, "image+pc": 22.6, "full": 6.4}
PRIOR_RUN = {"text": 51.19, "image": 81.27, "pc": 100.0}   # diag_ulip_fingerprint


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
def encode_texts(backbone, strings, batch=256, tag=""):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), batch):
            out.append(backbone.encode_text(strings[i:i + batch]).float().cpu().numpy())
            if i and i % (batch * 40) == 0:
                log(f"    {tag} {i:,}/{len(strings):,}")
    return np.concatenate(out)


def class_features(backbone, labels, templates):
    """(n_class, 1280) prompt ensemble -- upstream `main.py:377-380` exactly.

    Each of the 64 prompts is L2'd, the 64 are meaned, and the mean is L2'd
    again. The second normalise is not redundant: the mean of unit vectors is
    not a unit vector.
    """
    flat = [t.format(l) for l in labels for t in templates]
    e = encode_texts(backbone, flat, tag="class prompts")
    e = e.reshape(len(labels), len(templates), -1)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True)
    e = e.mean(axis=1)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def encode_point_clouds(backbone, uids, batch=64):
    """(n, 1280). Clouds on disk are ALREADY pc_norm'd -- measured, not assumed:
    |centroid| ~ 1e-10, max radius 1.000000, rgb in [0,1], which is exactly
    `Objaverse_Lvis_Colored.pc_norm`'s output. Re-normalising would be a no-op,
    and skipping it keeps the input byte-identical to what the rest of the
    pipeline feeds `encode_pc`."""
    import torch
    root = paths.OUTPUTS / "pointclouds"
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(uids), batch):
            clouds = np.stack([
                np.concatenate([z["xyz"], z["rgb"]], axis=1)
                for z in (np.load(root / f"{u}.npz") for u in uids[i:i + batch])
            ]).astype(np.float32)
            out.append(backbone.encode_pc(clouds).float().cpu().numpy())
            if i and i % (batch * 100) == 0:
                rate = i / (time.time() - t0)
                log(f"    pc {i:,}/{len(uids):,}  {rate:.0f}/s  "
                    f"eta {(len(uids)-i)/rate/60:.1f} min")
    return np.concatenate(out)


# --------------------------------------------------------------------------
def zero_shot(pc, text_feat, target):
    q, g = normalize_for_scoring(pc), normalize_for_scoring(text_feat)
    t1 = t5 = 0
    for i in range(0, len(q), 4096):
        order = np.argsort(-(q[i:i + 4096] @ g.T), axis=1)[:, :5]
        tgt = target[i:i + 4096][:, None]
        t1 += int((order[:, :1] == tgt).any(axis=1).sum())
        t5 += int((order == tgt).any(axis=1).sum())
    return {"top1": 100.0 * t1 / len(q), "top5": 100.0 * t5 / len(q)}


def recall(query, gallery, target_col, block=1024):
    """R@1 / R@5 via the shared `rank_of_target`, so ties count against the
    model here exactly as they do in n15."""
    q, g = normalize_for_scoring(query), normalize_for_scoring(gallery)
    r = np.concatenate([rank_of_target(q[i:i + block] @ g.T, target_col[i:i + block])
                        for i in range(0, len(q), block)])
    return {"R@1": 100.0 * float((r <= 1).mean()),
            "R@5": 100.0 * float((r <= 5).mean())}


def seven(mods, gallery, target_col):
    """The paper's seven query conditions against a pc-only gallery.

    [PAPER 3experiments.tex:15] fixes the aggregator as "a simple mean pooling
    layer" and no more. Whether each modality is unit-normalised BEFORE the mean
    is unstated -- searched MetaFind, ULIP and ULIP-2 text and ULIP's
    `models/losses.py`, none of which fixes a cross-modal mean -- so it stays an
    IMPLEMENTATION CHOICE and both orders run. It only moves the four
    multimodal cells: for a single modality the mean is a no-op under the final
    L2, and the `/2` of a mean cancels there too, so it is dropped.
    """
    out = {}
    for variant in ("B0_raw_mean", "B1_unit_mean"):
        rows = {}
        for cond, flags in QUERY_CONDITIONS.items():
            parts = [mods[m] for m, on in zip(("text", "image", "pc"), flags) if on]
            if variant == "B1_unit_mean":
                parts = [normalize_for_scoring(p) for p in parts]
            rows[cond] = recall(sum(parts), gallery, target_col)
        out[variant] = rows
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    meta = json.loads(paths.LVIS_METADATA.read_text())
    labels, uid2name, name2id = (meta["all_keys"], meta["value_to_key_mapping"],
                                 meta["key_to_id"])
    templates = json.loads((UPSTREAM / "data" / "templates.json").read_text())["modelnet40_64"]

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(sp["train"]) | set(sp["test"]))
    query_uids = sorted(sp["dev_val"])          # matches diag_ulip_fingerprint
    if args.limit:
        # Smoke mode: shrink the corpus first, then keep only the dev_val queries
        # that survive in it -- a query outside the gallery has no target column.
        corpus = corpus[:args.limit]
        inside = set(corpus)
        query_uids = [u for u in query_uids if u in inside][:512]
        if not query_uids:
            sys.exit("--limit too small: no dev_val query landed in the corpus slice")
    pos = {u: i for i, u in enumerate(corpus)}
    log(f"corpus {len(corpus):,} · queries {len(query_uids):,} (dev_val) · "
        f"{len(labels):,} classes x {len(templates)} templates")

    bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    assert bb.is_frozen(), "a calibration must run the released encoder frozen"

    # ---- one GPU pass ----------------------------------------------------
    cache = SCRATCH / "ulip2_pc_embed.npz"
    pc_all = None
    if cache.exists() and not args.limit:
        z = np.load(cache, allow_pickle=True)
        if list(z["uids"]) == corpus:
            log(f"reusing pc embeddings from {cache}")
            pc_all = z["embed"]
    if pc_all is None:
        log("encoding point clouds ...")
        pc_all = encode_point_clouds(bb, corpus, args.batch)
        if not args.limit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, uids=np.array(corpus), embed=pc_all)

    log("encoding class prompts ...")
    cls_feat = class_features(bb, labels, templates)

    log("encoding bare descriptions for the query ladder ...")
    ann = paths.OUTPUTS / "annotations"
    desc = [json.loads((ann / f"{u}.json").read_text())["description"] for u in query_uids]
    desc_emb = encode_texts(bb, desc, tag="descriptions")
    del bb

    # ---- GATE: zero-shot classification ----------------------------------
    tgt_cls = np.array([name2id[uid2name[u]] for u in corpus])
    zs = zero_shot(pc_all, cls_feat, tgt_cls)
    log("GATE done")

    # ---- the three ladders, all CPU --------------------------------------
    emb = paths.OUTPUTS / "embeddings"
    cached = [np.load(emb / f"{u}.npz") for u in query_uids]
    rich_text = np.stack([c["text"] for c in cached]).astype(np.float32)
    img_mean = np.stack([c["image"] for c in cached]).astype(np.float32)
    img_one = np.stack([c["views"][0] for c in cached]).astype(np.float32)
    q_pc = np.stack([pc_all[pos[u]] for u in query_uids])
    cat_text = np.stack([cls_feat[name2id[uid2name[u]]] for u in query_uids])

    galleries = {"dev_val_4569": query_uids,
                 "test_9138": sorted(sp["test"]),
                 "corpus_45692": corpus}
    if args.limit:
        galleries = {"dev_val_4569": query_uids}

    results = {}
    for gname, guids in galleries.items():
        gpos = {u: i for i, u in enumerate(guids)}
        if not all(u in gpos for u in query_uids):
            log(f"  skip {gname}: not every query is in it")
            continue
        gal = normalize_for_scoring(np.stack([pc_all[pos[u]] for u in guids]))
        col = np.array([gpos[u] for u in query_uids])
        results[gname] = {
            "seven_conditions": seven({"text": rich_text, "image": img_mean,
                                       "pc": q_pc}, gal, col),
            "text_ladder": {
                "L1_category_name_64_templates": recall(cat_text, gal, col),
                "L2_bare_description":           recall(desc_emb, gal, col),
                "L3_full_serialization":         recall(rich_text, gal, col),
            },
            "image_ladder": {
                "single_view_0":  recall(img_one, gal, col),
                "mean_12_views":  recall(img_mean, gal, col),
            },
            "n_gallery": len(guids),
        }
        log(f"  {gname}: text L3 {results[gname]['text_ladder']['L3_full_serialization']['R@1']:.1f} "
            f"L2 {results[gname]['text_ladder']['L2_bare_description']['R@1']:.1f} "
            f"L1 {results[gname]['text_ladder']['L1_category_name_64_templates']['R@1']:.1f}")

    # ---- report -----------------------------------------------------------
    print(f"\n{'='*74}\nGATE  ULIP-2 zero-shot, Objaverse-LVIS, n={len(corpus):,}")
    print(f"{'':10}{'ours':>9}{'our ckpt':>11}{'clean ckpt':>12}")
    for k in ("top1", "top5"):
        print(f"  {k:8}{zs[k]:9.1f}{PAPER_ZEROSHOT[k]:11.1f}{CLEAN_ZEROSHOT[k]:12.1f}")

    d = results.get("dev_val_4569")
    if d:
        print(f"\nSEVEN CONDITIONS, gallery = pc only, n={d['n_gallery']:,}")
        print(f"{'condition':>12}{'B0 raw':>9}{'B1 unit':>9}{'paper':>8}{'prior run':>11}")
        for c in QUERY_CONDITIONS:
            print(f"{c:>12}{d['seven_conditions']['B0_raw_mean'][c]['R@1']:9.1f}"
                  f"{d['seven_conditions']['B1_unit_mean'][c]['R@1']:9.1f}"
                  f"{PAPER_ULIP_ROW[c]:8.1f}"
                  f"{PRIOR_RUN.get(c, float('nan')):11.1f}")

        print(f"\nCAUSE 1  query text richness   (gallery = pc only, {d['n_gallery']:,})")
        for k, v in d["text_ladder"].items():
            print(f"  {k:34}{v['R@1']:8.1f}{v['R@5']:8.1f}")
        print("CAUSE 2  image observation")
        for k, v in d["image_ladder"].items():
            print(f"  {k:34}{v['R@1']:8.1f}{v['R@5']:8.1f}")

    print("\nCAUSE 3  gallery size   (text L3 -> pc gallery)")
    for g, r in results.items():
        print(f"  {g:22}{r['n_gallery']:>8,}"
              f"{r['text_ladder']['L3_full_serialization']['R@1']:9.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": "why our ULIP baseline text is 51.2% where the paper is 0.1%",
        "checkpoint": str(paths.ULIP2_CKPT),
        "checkpoint_deviation": ("this is ULIP-2's 'Objaverse + ShapeNet' row; its "
                                 "pretraining pool includes the LVIS assets we "
                                 "evaluate on. DEVIATION, recorded, NOT priced here "
                                 "-- ULIP-2's own clean-vs-contaminated gap is "
                                 "46.3 vs 50.6 zero-shot top-1."),
        "gate_zero_shot": {"ours": zs, "our_checkpoints_published_row": PAPER_ZEROSHOT,
                           "no_lvis_row": CLEAN_ZEROSHOT,
                           "provenance": "ULIP-2 CVPR'24 Tab.1 (main.tex:434-444)"},
        "metafind_paper_ulip_row_R@1": PAPER_ULIP_ROW,
        "by_gallery": results,
        "normalization_order": "IMPLEMENTATION CHOICE; no primary source fixes a "
                               "cross-modal mean's order",
        "u09": "the paper never states its gallery; sizes reported side by side",
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
