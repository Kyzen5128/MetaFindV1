#!/usr/bin/env python3
"""The two upstream deviations found by reading `upstream/ULIP/` end to end, priced.

DEVIATION A -- the query text is a SENTENCE here and a JSON BLOB in the paper
--------------------------------------------------------------------------------
MetaFind's Figure 2 (`data-preprocess.png`) prints the annotation it feeds, and
it is raw JSON:

    {"annotations": {"category": "robot", "synset": "robot.n.01", "width": 30,
     "length": 30, "height": 40, "volume": 36000, "mass": 2.5,
     "description": "A small cubic-shaped robot with ...", "materials": [...],
     "onCeiling": false, "onWall": false, "onFloor": true, "onObject": true}}

We serialise the same fields into fluent English instead (`metafind_v2_cm`).
That is an IMPLEMENTATION CHOICE -- the paper never states what string reaches
the text encoder -- and it is not cosmetic, because CLIP truncates at 77 tokens.
MEASURED on one asset: our prose is 74/77 and survives; the JSON is 77/77 and is
CUT, and what survives is

    {" annotations ": {" category ": " flip - flop ( sandal )", " synset ":
     " flip - flop . n . 0 2 ", " width ": 1 2 . 0 5 3 2 5 2 9 4 3 7 3 5 2 8 ,
     " length ": 9 . 4 1 4 9 6 3 2 3 7 0 5 8 3 4 5 , " height

-- the entire `description` is gone, spent on two 17-digit floats. So the JSON
form carries roughly a class name, which is where the paper's 13.8 sits on the
ladder we already measured (class name 0.2, description 35.4, ours 37.8).

DEVIATION B -- image preprocessing
-----------------------------------
[UPSTREAM FACT] `ULIP_models.py:354` takes open_clip's `preprocess` and DISCARDS
it; `main.py:176-182` normalises images with ImageNet statistics
(0.485/0.456/0.406, 0.229/0.224/0.225) and a RandomResizedCrop(224, 0.5-1.0).
ULIP-2's point encoder is therefore aligned against image embeddings produced
that way. We feed open_clip's own transform: Resize+CenterCrop and
0.4815/0.4578/0.4082, 0.2686/0.2613/0.2758. The stds differ by ~17%.

Priced on a query subset against the full 45,692 pc gallery, which is enough to
see direction and sign; the gallery never changes, so the comparison is clean.

WHAT THIS IS NOT
----------------
Diagnostic. Trains nothing, writes no protocol artifact, no checkpoint. Runs the
released encoder frozen. One json under `output/look/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metafind import paths                                       # noqa: E402
from metafind.eval.retrieval import normalize_for_scoring, rank_of_target  # noqa: E402

SCRATCH = Path("/tmp/claude-1002/-home-kyzen-MetaFindV1"
               "/ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad")
OUT = REPO / "output" / "look" / "two_deviations.json"

# The field order Figure 2 prints, so the truncation falls where the paper's would.
FIG2_KEYS = ["category", "synset", "width", "length", "height", "volume", "mass",
             "description", "materials", "onCeiling", "onWall", "onFloor", "onObject"]
# [UPSTREAM FACT main.py:176-177] what ULIP actually trained against.
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-image", type=int, default=2000,
                    help="query assets for deviation B; the gallery stays 45,692")
    args = ap.parse_args()

    import torch
    from torchvision import transforms
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    z = np.load(SCRATCH / "ulip2_pc_embed.npz", allow_pickle=True)
    corpus, pc_all = list(z["uids"]), z["embed"]
    pos = {u: i for i, u in enumerate(corpus)}
    qu = sorted(json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]["test"])
    gal = normalize_for_scoring(pc_all)
    log(f"gallery {len(corpus):,} (pc only) · query {len(qu):,} (test)")

    bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    assert bb.is_frozen()

    def recall(q, col, block=512):
        a = normalize_for_scoring(q)
        r = np.concatenate([rank_of_target(a[i:i + block] @ gal.T, col[i:i + block])
                            for i in range(0, len(a), block)])
        return {"R@1": 100.0 * float((r <= 1).mean()),
                "R@5": 100.0 * float((r <= 5).mean())}

    def enc_text(strings, tag):
        out = []
        with torch.no_grad():
            for i in range(0, len(strings), 256):
                out.append(bb.encode_text(strings[i:i + 256]).float().cpu().numpy())
                if i and i % 5120 == 0:
                    log(f"  {tag} {i:,}/{len(strings):,}")
        return np.concatenate(out)

    res = {}

    # ---- DEVIATION A -------------------------------------------------------
    ann = paths.OUTPUTS / "annotations"
    raw = [json.loads((ann / f"{u}.json").read_text()) for u in qu]
    fig2 = [json.dumps({"annotations": {k: a[k] for k in FIG2_KEYS}}) for a in raw]
    col = np.array([pos[u] for u in qu])
    log("A: encoding Figure-2-style JSON queries ...")
    res["A_text_form"] = {
        "fig2_raw_json": recall(enc_text(fig2, "json"), col),
        "ours_prose_serialization": recall(
            np.stack([np.load(paths.OUTPUTS / "embeddings" / f"{u}.npz")["text"]
                      for u in qu]).astype(np.float32), col),
        "paper_metafind_R@1": 13.8, "paper_ulip_row_R@1": 0.1,
    }
    log(f"A: json {res['A_text_form']['fig2_raw_json']['R@1']:.2f}  "
        f"ours {res['A_text_form']['ours_prose_serialization']['R@1']:.2f}")

    # ---- DEVIATION B -------------------------------------------------------
    sub = qu[:args.n_image]
    subcol = np.array([pos[u] for u in sub])
    renders = paths.OUTPUTS / "renders"
    openclip_tf = bb.preprocess
    imagenet_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),   # upstream's, verbatim
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET),
    ])
    # RandomResizedCrop is stochastic; seed so the comparison is reproducible and
    # both transforms see the same PIL images in the same order.
    torch.manual_seed(0)

    def enc_views(tf, tag):
        """(n, 12, 1280) -- every view, so mean and single-view both fall out."""
        out = []
        with torch.no_grad():
            for i, u in enumerate(sub):
                ims = [Image.open(renders / u / f"view_{v:02d}.png").convert("RGB")
                       for v in range(12)]
                out.append(bb.encode_image(torch.stack([tf(im) for im in ims]))
                           .float().cpu().numpy())
                if i and i % 400 == 0:
                    log(f"  {tag} {i:,}/{len(sub):,}")
        return np.stack(out)

    log("B: encoding renders under BOTH transforms ...")
    v_oc = enc_views(openclip_tf, "openclip")
    v_in = enc_views(imagenet_tf, "imagenet")
    res["B_image_preprocessing"] = {
        "n_query": len(sub), "n_gallery": len(corpus),
        "openclip_resize_centercrop": {
            "mean_12_views": recall(v_oc.mean(1), subcol),
            "single_view_0": recall(v_oc[:, 0], subcol)},
        "imagenet_randomresizedcrop": {
            "mean_12_views": recall(v_in.mean(1), subcol),
            "single_view_0": recall(v_in[:, 0], subcol)},
        "paper_metafind_R@1": 11.7, "paper_ulip_row_R@1": 0.1,
    }

    b = res["B_image_preprocessing"]
    print(f"\n{'='*70}\nDEVIATION A -- query text form  (gallery pc only, 45,692)")
    print(f"  {'Figure-2 raw JSON':32}{res['A_text_form']['fig2_raw_json']['R@1']:8.2f}")
    print(f"  {'our prose serialization':32}"
          f"{res['A_text_form']['ours_prose_serialization']['R@1']:8.2f}")
    print(f"  {'paper MetaFind':32}{13.8:8.2f}\n  {'paper ULIP row':32}{0.1:8.2f}")

    print(f"\nDEVIATION B -- image preprocessing  (n_query {len(sub):,})")
    print(f"{'':34}{'12-view mean':>14}{'single view':>13}")
    for k, label in (("openclip_resize_centercrop", "open_clip (ours)"),
                     ("imagenet_randomresizedcrop", "ImageNet (upstream's)")):
        print(f"  {label:32}{b[k]['mean_12_views']['R@1']:14.2f}"
              f"{b[k]['single_view_0']['R@1']:13.2f}")
    print(f"  {'paper MetaFind / ULIP row':32}{11.7:14.2f}{0.1:13.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
