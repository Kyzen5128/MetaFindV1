#!/usr/bin/env python3
"""Short fixed-format text built from the EXISTING annotation fields.

[KYZEN 2026-09-03] "我text欄位有紀錄嗎?像論文那樣 ... 可以利用固定的格式的短描述
去限制嗎?就不用重新生成了."

PAPER FACT, arXiv HTML sec. 2.3, verbatim:
    "These annotations provide rich textual descriptions detailing attributes
     such as object category, size dimensions, materials, and placement
     constraints."
That list is exactly four structured fields, all of which n06 already wrote to
every sidecar. It does NOT mention the long free-form visual description that
`TEXT_TEMPLATE` puts FIRST and that eats most of CLIP's 77-token window
(measured: 68 tokens median, 3.5% of the corpus truncated at the cap).

So the arms below are re-serializations of annotation data we already have. No
regeneration, no new LLM call.

BOTH SIDES are re-encoded, query and gallery. Changing only the query would
measure an observation MISMATCH; Kyzen asked to constrain the corpus format,
which is a change to the text both towers read.

Scoring is `metafind.eval.retrieval` -- the SAME code the Table 1 node uses.
A previous version of this experiment shipped its own recall() and its own
gallery construction; the gallery was built with a ZERO point cloud and ties
counted in the model's favour, and every number it produced was wrong.

[SENSITIVITY, NOT A RETRAIN] One checkpoint trained on the full template,
re-scored under other serializations. This bounds how much of the gap the TEXT
LENGTH can explain; it does not tell us what a model trained on short text does.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metafind import paths
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,
                                     normalize_for_scoring, recall_at_k)
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.train.stage1 import (Stage1Dataset, collate, load_protocols,
                                   modules_in_eval, split_embeds)

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2,
         "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}

# Every arm is `serialize_annotation(a, template=...)`, i.e. the SHIPPING
# serializer with a different template, so no arm can drift from what n06 does.
ARMS = {
    "full": None,                                    # what the corpus uses now
    "attrs_only": "{category} made of {materials}, roughly {width} by {length} "
                  "by {height} centimetres, {placement}.",
    "attrs_coarse": "{category} made of {materials}, roughly {width} by "
                    "{length} by {height} centimetres, {placement}.",
    "no_dims": "{category} made of {materials}, {placement}.",
    "cat_only": "{category}.",
    # --- the three arms that actually follow Figure 2 --------------------
    # [KYZEN 2026-09-03] "metafind論文不只有4項啊" -- correct. Figure 2 prints
    # the annotation record itself, and it carries THIRTEEN fields:
    #   category, synset, width, length, height, volume, mass, description,
    #   materials, onCeiling, onWall, onFloor, onObject
    # every one of which n06 already writes. `description` is among them, so
    # `attrs_only` above (which drops it) is NOT the paper's annotation; it is
    # a length ablation and is kept only as one.
    "figure2_json": "__json__",     # the record itself, serialised as Figure 2
    "figure2_prose": None,          # our template, but Figure 2's number format
    "desc_only": "{description}",
    # --- the form-fill ladder --------------------------------------------
    # [KYZEN 2026-09-03] "文字都先採用固定填表的方式". Every rung is the SAME
    # fixed form -- description, category, materials, whole-centimetre
    # dimensions, placement -- and the only thing that moves is how much of the
    # description the form admits. `fill0` is the pure form-fill (no free
    # prose at all); the rest cap it on a WORD boundary at N characters.
    # Figure 2's own record carries a description, so a rung that keeps one is
    # closer to the figure than `fill0` is; the ladder exists to find which cap
    # lands on Table 1's 13.8 rather than to argue for one.
    "fill0": "{category} made of {materials}, roughly {width} by {length} by "
             "{height} centimetres, {placement}.",
    "fill30": None, "fill50": None, "fill70": None,
    "fill100": None, "fill140": None,
}
FILL = {"fill0": 0, "fill30": 30, "fill50": 50, "fill70": 70, "fill100": 100,
        "fill140": 140}
# Dimensions rounded to the nearest 10 cm: a length ablation, not the paper.
COARSE = {"attrs_coarse"}
# Figure 2 prints whole-number dimensions ("width": 30, "height": 40) and a
# description that ENDS ON A SENTENCE. Ours renders one decimal place
# (12.1 by 9.4 by 3.5) and hard-cuts at 160 characters, which lands mid-clause
# ("...They are finished."). Both are OBSERVED differences from the figure.
FIGURE2 = {"figure2_json", "figure2_prose"}
# Exactly the keys Figure 2 shows, in the order it shows them.
FIGURE2_KEYS = ("category", "synset", "width", "length", "height", "volume",
                "mass", "description", "materials",
                "onCeiling", "onWall", "onFloor", "onObject")


# --- the image axis ---------------------------------------------------------
# [KYZEN 2026-09-03] "補影像嗎?" -- yes. image-only is the worst cell (84.6
# against Table 1's 11.7) and no text arm moves it, because the query image IS
# the gallery's own vector: sec. 2.3 renders each asset from several views and
# `n07` caches their mean, and `split_embeds` hands that one mean to both towers.
#
# The 12 per-view vectors are already in every `.npz` (`views`, 12x1280), so
# every policy below is a re-read of cached data. No re-render, no re-encode.
#
# `same_mean` is the shipping construction. `single_view` and `four_views` change
# the QUERY only and leave the gallery exactly as the promoted index has it,
# which is what sec. 2.4 requires -- "The gallery encoder is modality-complete
# and frozen after pretraining." `disjoint_views` also rebuilds the GALLERY from
# the complementary half; it is the only policy that touches the gallery, and it
# is declared in the output for that reason.
IMAGE_POLICIES = ("same_mean", "single_view", "four_views", "disjoint_views",
                  "eleven_views")
# `eleven_views` rebuilds BOTH sides from eleven of the twelve cached views, the
# dropped index chosen per uid. It is the cheapest reading of sec. 2.3's "11
# orthogonal viewpoints" against our 12 (a DEVIATION, sec. 6.1 of the
# notebook): if dropping any one view leaves the fingerprint where it was, the
# camera count is not the axis that separates us from Table 1; if it moves it,
# the view protocol is promoted to a blocker before anyone re-renders 46K.
GALLERY_CHANGING = {"disjoint_views", "eleven_views"}


def _views(uid: str) -> np.ndarray:
    v = np.load(paths.EMBEDDINGS / f"{uid}.npz")["views"].astype(np.float32)
    if v.shape[0] != 12:
        raise SystemExit(f"{uid} caches {v.shape[0]} views, not 12")
    return v


def _pick(uid: str, k: int) -> list[int]:
    """Which view indices this asset offers the query. Deterministic per uid --
    a seeded global RNG would make the arm depend on evaluation order."""
    import hashlib
    h = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
    return sorted((h + i * 12 // k) % 12 for i in range(k)) if k > 1 else [h % 12]


def image_pair(uids: list[str], policy: str):
    """(query image, gallery image) for every uid, as float32 tensors."""
    if policy not in IMAGE_POLICIES:
        raise SystemExit(f"unknown image policy {policy!r}")
    q, g = [], []
    for u in uids:
        v = _views(u)
        if policy == "same_mean":
            m = v.mean(0)
            q.append(m); g.append(m)
        elif policy == "single_view":
            q.append(v[_pick(u, 1)[0]]); g.append(v.mean(0))
        elif policy == "four_views":
            q.append(v[_pick(u, 4)].mean(0)); g.append(v.mean(0))
        elif policy == "eleven_views":
            keep = [i for i in range(12) if i != _pick(u, 1)[0]]
            m = v[keep].mean(0)
            q.append(m); g.append(m)
        else:                                    # disjoint_views
            idx = set(_pick(u, 6))
            other = [i for i in range(12) if i not in idx]
            q.append(v[sorted(idx)].mean(0)); g.append(v[other].mean(0))
    return (torch.from_numpy(np.stack(q)), torch.from_numpy(np.stack(g)))


def _sentence_cap(text: str, limit: int) -> str:
    """Cut on a sentence boundary instead of mid-clause."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:stop + 1] if stop > 40 else cut).strip()


def text_for(ann: dict, arm: str) -> str:
    if arm in COARSE:
        ann = copy.deepcopy(ann)
        for k in ("width", "length", "height"):
            ann[k] = max(10.0, round(float(ann[k]) / 10.0) * 10.0)
    if arm in FILL:
        ann = copy.deepcopy(ann)
        for k in ("width", "length", "height"):
            ann[k] = float(round(float(ann[k])))
        d = ann["description"].strip()
        n = FILL[arm]
        if n == 0:
            return serialize_annotation(ann, template=ARMS["fill0"])
        if len(d) > n:                      # cut on a word, never mid-word
            d = d[:n].rsplit(" ", 1)[0].rstrip(" ,;:")
        ann["description"] = d
        return serialize_annotation(ann, template=None)
    if arm in FIGURE2 or arm == "desc_only":
        ann = copy.deepcopy(ann)
        for k in ("width", "length", "height"):
            ann[k] = float(round(float(ann[k])))
        # 160, not 200: `serialize_annotation` re-caps at MAX_DESCRIPTION_CHARS
        # and would put the mid-clause cut straight back. The json arm keeps the
        # whole description because it does not go through that serializer.
        if arm != "figure2_json":
            ann["description"] = _sentence_cap(ann["description"], 160)
        if arm == "figure2_json":
            rec = {k: (round(ann[k], 2) if isinstance(ann.get(k), float)
                       else ann.get(k)) for k in FIGURE2_KEYS}
            return json.dumps({"annotations": rec}, ensure_ascii=False)
    return serialize_annotation(ann, template=ARMS[arm])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-policy", default="same_mean",
                    help="which observation the QUERY image is; "
                         "same_mean is the shipping construction")
    ap.add_argument("--prefusion-norm", action="store_true",
                    help="L2-normalise each modality vector before it enters "
                         "Fusion, on BOTH towers. `fusion.py` does not: raw "
                         "vectors go straight into the Transformer, and the "
                         "three towers are trained differently, so their scales "
                         "drift. DIAGNOSTIC ONLY -- this checkpoint was trained "
                         "without it, so a drop here measures scale sensitivity, "
                         "not what a model trained with it would score.")
    ap.add_argument("--derange", default="none",
                    choices=("none", "text", "image", "pc"),
                    help="permute ONE modality's asset identity on the GALLERY "
                         "side only, keeping the query intact. How far R@1 "
                         "falls is how much of the gallery vector's identity "
                         "that modality carries. A diagnostic of the learned "
                         "Fusion_G, not a protocol.")
    ap.add_argument("--similarity", default="cosine", choices=("cosine", "dot"),
                    help="cosine is the production scorer (normalize_for_scoring "
                         "then GEMM). dot skips the normalisation: sec. 2.1 "
                         "writes only 'sim(.,.)', and DPR, which the paper cites "
                         "for the dual-tower paradigm, uses the unnormalised "
                         "inner product. A SENSITIVITY on one checkpoint trained "
                         "under cosine; not a claim about the paper.")
    ap.add_argument("--arm", action="append",
                    help="repeatable; default every arm")
    ap.add_argument("--out", default="output/look/exp_text_length.json")
    args = ap.parse_args()
    arms = args.arm or list(ARMS)

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from tools.probes.exp_query_observation import load_tower

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    missing = set(q_uids) - set(g_uids)
    if missing:
        raise SystemExit(f"{len(missing):,} queries are not in the gallery; "
                         "R@1 against an absent target is undefined")
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    q_rows = torch.tensor(targets)

    bb = ULIPBackbone(BackboneConfig(device=args.device,
                                     train_scope="pointbert_and_fuser"))
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bb.model.load_state_dict(ck["backbone_trainable_state"], strict=False)
    model = load_tower(Path(args.ckpt), args.device)
    enc, _, _ = load_protocols()

    # --- image and point cloud, encoded ONCE and shared by every arm ---------
    print(f"encoding image+pc for {len(g_uids):,} gallery assets "
          "(once, shared by every arm)", flush=True)
    loader = DataLoader(Stage1Dataset(g_uids, enc["image_aggregation"]),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=4)
    IMG, PC = [], []
    with modules_in_eval(model, getattr(bb, "model", None)), torch.no_grad():
        for i, b in enumerate(loader):
            _q, g = split_embeds(b, bb, args.device)
            IMG.append(g["image"].cpu())
            PC.append(g["pc"].float().cpu())
            if i % 100 == 0:
                print(f"  batch {i}", flush=True)
    IMG, PC = torch.cat(IMG), torch.cat(PC)

    # `same_mean` keeps the dataloader's own vectors, so the canonical arm stays
    # bit-for-bit the construction `run_retrieval.py` scores and the parity check
    # keeps meaning something. Any other policy rebuilds from the cached views.
    if args.image_policy == "same_mean":
        QIMG = GIMG = IMG
    else:
        QIMG, GIMG = image_pair(g_uids, args.image_policy)
        cached_mean, _ = image_pair(g_uids[:256], "same_mean")
        drift = (cached_mean - IMG[:256]).abs().max().item()
        if drift > 2e-2:
            raise SystemExit(
                f"mean(views) and the cached `image` differ by {drift:.4f}; the "
                "view-derived policies would not be measuring the same object "
                "as the shipping aggregation")
        print(f"  image policy {args.image_policy}: query rebuilt from cached "
              f"views (mean(views) vs cached image, max |diff| {drift:.5f})"
              + ("; GALLERY ALSO REBUILT" if args.image_policy
                 in GALLERY_CHANGING else "; gallery unchanged"), flush=True)

    # What actually enters Fusion. The three towers are trained differently --
    # PointBERT moves, the two CLIP towers do not -- so their scales drift
    # apart, and `fusion.py` applies no per-modality normalisation before the
    # Transformer. Printed here because a modality that is numerically larger
    # dominates attention regardless of what it means.
    print("\nmodality norms entering Fusion (this checkpoint, gallery side):")
    for nm, arr in (("image", IMG), ("pc", PC)):
        n = arr.float().norm(dim=1)
        print(f"  {nm:<6} mean {n.mean():7.2f}  std {n.std():6.2f}")

    anns = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
            for u in g_uids]
    print("\none asset, every arm:")
    for a in arms:
        print(f"  {a:<14} {text_for(anns[0], a)!r}")

    out, hdr = {}, "arm".ljust(15) + "".join(c.rjust(13) for c in QUERY_CONDITIONS)
    print(f"\n{len(q_uids):,} held-out queries against {len(g_uids):,} gallery "
          "assets. Both sides carry the arm's text.")
    print("paper".ljust(15) + "".join(("%.1f" % PAPER[c]).rjust(13)
                                      for c in QUERY_CONDITIONS))
    print(hdr)

    for arm in arms:
        texts = [text_for(a, arm) for a in anns]
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                vecs.append(bb.encode_text(texts[i:i + 256]).float().cpu())
        TXT = torch.cat(vecs)

        n = TXT.float().norm(dim=1)
        print(f"  text   mean {n.mean():7.2f}  std {n.std():6.2f}"
              f"   (arm {arm})", flush=True)

        # The derangement acts on the GALLERY only, after the query rows are
        # taken, so a query still meets its own text/image/pc -- only the
        # gallery entry it should match now carries someone else's.
        gTXT, gIMG, gPC = TXT, IMG, PC
        if args.derange != "none":
            rng = np.random.default_rng(20260903)
            perm = rng.permutation(len(g_uids))
            fixed = int((perm == np.arange(len(g_uids))).sum())
            print(f"  derange {args.derange}: {fixed} of {len(g_uids):,} rows "
                  "happen to map to themselves", flush=True)
            pt = torch.from_numpy(perm)
            if args.derange == "text":
                gTXT = TXT[pt]
            elif args.derange == "image":
                gIMG = IMG[pt]
            else:
                gPC = PC[pt]

        def _pf(t):
            return (torch.nn.functional.normalize(t, dim=-1)
                    if args.prefusion_norm else t)

        G, Q = [], []
        with modules_in_eval(model, getattr(bb, "model", None)), torch.no_grad():
            for i in range(0, len(g_uids), 512):
                s = slice(i, i + 512)
                e = {"text": _pf(gTXT[s].to(args.device)),
                     "image": _pf((GIMG if args.derange != "image" else gIMG)[s]
                                  .to(args.device)),
                     "pc": _pf(gPC[s].to(args.device))}
                G.append(model.gallery(e).float().cpu())
            G = torch.cat(G)
            cells = {}
            for cond in QUERY_CONDITIONS:
                Q = []
                for i in range(0, len(q_uids), 512):
                    r = q_rows[i:i + 512]
                    e = {"text": _pf(TXT[r].to(args.device)),
                         "image": _pf(QIMG[r].to(args.device)),
                         "pc": _pf(PC[r].to(args.device))}
                    m = condition_mask(cond, len(r)).to(args.device)
                    Q.append(model.query(e, present=m).float().cpu())
                if args.similarity == "cosine":
                    qv = normalize_for_scoring(torch.cat(Q).numpy())
                    gv = normalize_for_scoring(G.numpy())
                else:   # dot: float64 GEMM, same rank/tie rule, no normalisation
                    qv = torch.cat(Q).numpy().astype(np.float64)
                    gv = G.numpy().astype(np.float64)
                cells[cond] = recall_at_k(qv @ gv.T, targets)
        out[arm] = cells
        print(arm.ljust(15) + "".join(("%.1f" % (cells[c]["R@1"] * 100)).rjust(13)
                                      for c in QUERY_CONDITIONS), flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt,
        "query_split": args.query_split, "gallery_split": args.gallery_split,
        "n_query": len(q_uids), "n_gallery": len(g_uids),
        "paper_w_o_essgnn_R@1_percent": PAPER,
        "templates": ARMS, "coarse_dimension_arms": sorted(COARSE),
        "both_sides_re_encoded": True,
        "image_policy": args.image_policy,
        "similarity": args.similarity, "derange": args.derange,
        "prefusion_norm": args.prefusion_norm,
        "gallery_image_rebuilt": args.image_policy in GALLERY_CHANGING,
        "scoring": "metafind.eval.retrieval (the Table 1 code path)",
        "caveat": "one checkpoint trained on the `full` template, re-scored "
                  "under other serializations. Not a retrain.",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
