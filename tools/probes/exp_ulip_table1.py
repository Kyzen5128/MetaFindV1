#!/usr/bin/env python3
"""ULIP baseline: which gallery construction reproduces Table 1's ULIP row?

[KYZEN 2026-09-03] "先復現 ULIP Table1：PC-gallery / mean-gallery、raw-mean /
normalized-mean、test/full gallery，全是 cheap inference；先把 baseline
mechanism 搞清楚."

A0 = arXiv HTML. Verbatim, sec. 3.1:
    "we extend each baseline by adding a simple mean pooling layer to
     aggregate available modalities, and use these fused embeddings to
     retrieve from a pre-encoded gallery."
so the QUERY side is fixed by the paper: mean of the available modalities. The
paper does not state what the pre-encoded gallery holds. Sec. 3.2, verbatim:
    "since other models do not adopt a dual-tower design, their 'PC only'
     performance reflects retrieval using identical embeddings for both query
     and gallery, leading to inflated accuracy."
That sentence is a property of a PC-only gallery (B1) and not of a mean gallery
(B2); the table is the test.

    Table 1, ULIP row (R@1 / R@5):
      text 0.1/0.9   image 0.1/1.3   pc 97.9/99.4   T+I 0/0.3
      T+PC 33.9/58   I+PC 22.6/41.6  full 6.4/15.9

Two forks the paper leaves open, both run:
    gallery      B1 = PC embedding alone      B2 = mean(text, image, pc)
    pooling      raw mean                      mean of L2-normalised vectors

No MetaFind fusion, no Stage 1 weights, no ESSGNN: the released ULIP-2 encoder
only (sha verified by ULIPBackbone). Text and image are the n06 cache, i.e. the
same frozen towers; pc is encoded live by the released PointBERT.

Scoring is metafind.eval.retrieval, the Table 1 code path. Item 3 of
diag_protocol_e_ulip_fingerprint.py holds the same arms but scores a
query-equals-gallery diagonal; this runs them on explicit targets so the query
pool can be a subset of the gallery (protocol D shape), and on the whole corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metafind import paths
from metafind.eval.retrieval import (MODALITIES, QUERY_CONDITIONS,
                                     normalize_for_scoring, recall_at_k)
from metafind.train.stage1 import (Stage1Dataset, collate, load_protocols,
                                   modules_in_eval)

PAPER_ULIP = {"text": (0.1, 0.9), "image": (0.1, 1.3), "pc": (97.9, 99.4),
              "text+image": (0.0, 0.3), "text+pc": (33.9, 58.0),
              "image+pc": (22.6, 41.6), "full": (6.4, 15.9)}


def collect(backbone, uids, aggregation, device, bs):
    loader = DataLoader(Stage1Dataset(uids, aggregation), batch_size=bs,
                        shuffle=False, collate_fn=collate, num_workers=4)
    order, T, I, P = [], [], [], []
    with modules_in_eval(getattr(backbone, "model", None)), torch.no_grad():
        for i, b in enumerate(loader):
            order.extend(b["uid"])
            T.append(b["text"].float()); I.append(b["image"].float())
            P.append(backbone.encode_pc(b["pc"].to(device)).float().cpu())
            if i % 100 == 0:
                print(f"  batch {i}", flush=True)
    return order, torch.cat(T).numpy(), torch.cat(I).numpy(), torch.cat(P).numpy()


def fingerprint(T, I, P, q_rows, targets, QT=None, QI=None):
    """QT / QI: the QUERY side's text / image when they differ from the
    gallery's (an observation arm). Default: the same arrays, i.e. the paper's
    'identical embeddings' situation."""
    QT = T if QT is None else QT
    QI = I if QI is None else QI
    tn, In, pn = (normalize_for_scoring(x) for x in (T, I, P))
    out = {"modality_geometry": {
        "norms": {m: float(np.linalg.norm(x, axis=1).mean())
                  for m, x in (("text", T), ("image", I), ("pc", P))},
        "mean_paired_cosine": {
            "text_pc": float((tn * pn).sum(1).mean()),
            "image_pc": float((In * pn).sum(1).mean()),
            "text_image": float((tn * In).sum(1).mean())}}}
    qtn, qIn = normalize_for_scoring(QT), normalize_for_scoring(QI)
    for norm in (False, True):
        t, i, p = (tn, In, pn) if norm else (T, I, P)
        raw = {"text": qtn if norm else QT, "image": qIn if norm else QI,
               "pc": p}          # the query reads ITS observation; the gallery its own
        for name, g in (("B1_gallery_pc", p),
                        ("B2_gallery_mean3", np.mean([t, i, p], axis=0))):
            gn = normalize_for_scoring(g)
            cell = {}
            for c in QUERY_CONDITIONS:
                present = [m for m, f in zip(MODALITIES, QUERY_CONDITIONS[c]) if f]
                q = normalize_for_scoring(
                    np.mean([raw[m][q_rows] for m in present], axis=0))
                cell[c] = recall_at_k(q @ gn.T, targets, ks=(1, 5))
            out[f"{name}__{'l2mean' if norm else 'rawmean'}"] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-split", default="dev_val")
    ap.add_argument("--gallery-split", default="train_val",   # [D-3b]
                    help="a split key, or 'full' for train+test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    # The released encoder is FIXED, so if the paper's ULIP row cannot be
    # reproduced on our corpus the difference is the corpus, and these two axes
    # are the cheap ones: re-serialize the text (frozen text tower, minutes) and
    # re-read the query image from the cached per-view matrix (no re-render).
    # Both reuse exp_text_length's arms so a number here and a number there
    # mean the same construction.
    ap.add_argument("--text-arm", default="full",
                    help="a key of exp_text_length.ARMS; applied to BOTH sides")
    ap.add_argument("--image-policy", default="same_mean",
                    help="a key of exp_text_length.IMAGE_POLICIES; QUERY side")
    ap.add_argument("--out", default="output/look/exp_ulip_table1.json")
    args = ap.parse_args()

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids = (sorted(set(sp["train"]) | set(sp["test"]))
              if args.gallery_split == "full" else sorted(sp[args.gallery_split]))
    q_uids = (g_uids if args.query_split == "gallery"
              else sorted(sp[args.query_split]))
    missing = set(q_uids) - set(g_uids)
    if missing:
        raise SystemExit(f"{len(missing):,} queries are not in the gallery")
    where = {u: k for k, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])

    # fuser_only: PointBERT stays at the released ULIP-2 weights, nothing trains
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    enc, _, _ = load_protocols()
    print(f"encoding {len(g_uids):,} gallery assets with the released encoder",
          flush=True)
    order, T, I, P = collect(bb, g_uids, enc["image_aggregation"],
                             args.device, args.batch_size)
    assert order == g_uids

    from tools.probes.exp_text_length import (GALLERY_CHANGING, image_pair,
                                              text_for)
    if args.text_arm != "full":
        anns = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
                for u in g_uids]
        texts = [text_for(a, args.text_arm) for a in anns]
        print(f"  text arm {args.text_arm}: {texts[0]!r}", flush=True)
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                vecs.append(bb.encode_text(texts[i:i + 256]).float().cpu())
        T = torch.cat(vecs).numpy()
    QI = I
    if args.image_policy != "same_mean":
        q_img, g_img = image_pair(g_uids, args.image_policy)
        QI = q_img.numpy()
        if args.image_policy in GALLERY_CHANGING:
            I = g_img.numpy()
        print(f"  image policy {args.image_policy}: query rebuilt from views"
              + ("; GALLERY ALSO REBUILT" if args.image_policy in GALLERY_CHANGING
                 else "; gallery unchanged"), flush=True)
    fp = fingerprint(T, I, P, targets, targets, QT=T, QI=QI)

    print(f"\n{len(q_uids):,} queries against {len(g_uids):,} gallery assets")
    g = fp["modality_geometry"]
    print("released-encoder geometry on this corpus:")
    print("  norms   " + "  ".join(f"{m} {v:.2f}" for m, v in g["norms"].items()))
    print("  cosine  " + "  ".join(f"{m} {v:.3f}"
                                   for m, v in g["mean_paired_cosine"].items()))
    hdr = "arm".ljust(28) + "".join(c.rjust(11) for c in QUERY_CONDITIONS)
    print("\nR@1"); print(hdr)
    print("paper ULIP".ljust(28) + "".join(
        ("%.1f" % PAPER_ULIP[c][0]).rjust(11) for c in QUERY_CONDITIONS))
    for k, cell in fp.items():
        if k == "modality_geometry":
            continue
        print(k.ljust(28) + "".join(
            ("%.1f" % (cell[c]["R@1"] * 100)).rjust(11) for c in QUERY_CONDITIONS))
    print("\nR@5"); print(hdr)
    print("paper ULIP".ljust(28) + "".join(
        ("%.1f" % PAPER_ULIP[c][1]).rjust(11) for c in QUERY_CONDITIONS))
    for k, cell in fp.items():
        if k == "modality_geometry":
            continue
        print(k.ljust(28) + "".join(
            ("%.1f" % (cell[c]["R@5"] * 100)).rjust(11) for c in QUERY_CONDITIONS))

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "query_split": args.query_split, "gallery_split": args.gallery_split,
        "text_arm": args.text_arm, "image_policy": args.image_policy,
        "n_query": len(q_uids), "n_gallery": len(g_uids),
        "encoder": "released ULIP-2, no Stage 1 weights, no fusion",
        "paper_ulip_row": PAPER_ULIP,
        "scoring": "metafind.eval.retrieval", "results": fp}, indent=1))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
