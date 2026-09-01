#!/usr/bin/env python3
"""Their pixels against our pixels, both ranked over the whole point-cloud gallery.

Codex, 2026-09-01, third correction:

> Do not just measure image-image cosine 0.82-0.86 between the two renders.
> Encode both sets of pixels and rank each against the full 45,692 PC gallery,
> with R@1, R@5, median target rank, positive cosine, hardest negative cosine
> and the top-1 margin.

He is right that the cosine between the two sides' view vectors cannot separate
"different render domain" from "same domain, different retrieval behaviour".
A rank is the thing under dispute, so a rank is what has to be measured.

The question this settles: our image-only query retrieves its own point cloud
at 52.7 R@1; the paper's ULIP row reports 0.1 with a checkpoint `custom.bib`
resolves to ULIP-2, the one we reproduce to 0.02 on zero-shot LVIS. If ULIP-2's
OWN published renders also retrieve at ~50, the render domain is eliminated and
the discrepancy is upstream of the pixels -- a join, a wrapper, or a query
construction we have not seen. If theirs ranks far worse, the domain is the
answer.

Both sides go through OUR encoder and OUR preprocessing (`view_io.load_views_rgb`
plus open_clip's transform, the path n06 uses), so the encoder is held fixed and
the pixels are the only variable.

  ours     data/outputs/renders/<uid>/view_NN.png     12, Blender, 3 rings of 4
  theirs   ULIP_Objaverse_Triplets/render_images_resized_224, chunk 0000,
           <uid>/NNN.png, 12 per asset

Gallery is the released encoder's point-cloud embedding for all 45,692. A
deranged arm is scored alongside as the floor, since the whole question is
whether the paper's endpoint is at that floor.

Small n by construction: one 2.5 GB chunk of 193, intersected with our corpus.
Reported as a directional read against a full-size gallery, not as a corpus
measurement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "render_domain_ranking.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
THEIRS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/their_renders")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--their-renders", default=THEIRS)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    have = {pathlib.Path(p).parent.name for p in
            glob.glob(os.path.join(args.their_renders, "**", "*.png"),
                      recursive=True)}
    uids = sorted(u for u in have if u in pos
                  and (paths.RENDERS / u).is_dir())
    print(f"兩邊都有像素的資產 {len(uids)}", flush=True)
    if len(uids) < 30:
        sys.exit("too few to read anything off")

    from metafind.data.encode_text_image import Encoder
    enc = Encoder(device="cuda" if torch.cuda.is_available() else "cpu")

    def encode_dir(paths_12: list[str]) -> np.ndarray:
        return enc.encode_views(paths_12).mean(0)

    ours, theirs = [], []
    for k, u in enumerate(uids):
        ours.append(encode_dir([str(paths.RENDERS / u / f"view_{i:02d}.png")
                                for i in range(12)]))
        tp = sorted(glob.glob(os.path.join(args.their_renders, "**", u, "*.png"),
                              recursive=True))
        theirs.append(encode_dir(tp[:12]))
        if (k + 1) % 50 == 0:
            print(f"  encoded {k + 1}/{len(uids)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = lambda a: torch.nn.functional.normalize(a, dim=-1)
    G = n(torch.from_numpy(np.load(PC_CACHE).astype(np.float32)).to(dev))
    tgt = torch.tensor([pos[u] for u in uids], device=dev)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(uids))

    res = {"n_query": len(uids), "n_gallery": len(corpus),
           "encoder": "ours, held fixed for both sides", "arms": {}}
    print(f"\n{'來源':<26s}{'R@1':>7s}{'R@5':>7s}{'目標排名中位':>13s}"
          f"{'正例 cos':>10s}{'最難負例':>10s}{'邊際':>9s}")

    hits = {}

    def score(name, Q):
        """[Codex 2026-09-01] frac(margin > 0) is reported beside R@1 as a
        SELF-CHECK: with margin = positive - max_negative the two must agree,
        and a disagreement would mean the rank or the negative exclusion is
        wrong rather than the model."""
        q = n(torch.from_numpy(np.stack(Q)).to(dev))
        s = q @ G.t()
        own = s.gather(1, tgt.unsqueeze(1)).squeeze(1)
        rank = (s > own.unsqueeze(1)).sum(1) + 1
        hard = s.scatter(1, tgt.unsqueeze(1), -2.0).max(1).values
        marg = own - hard
        m = len(uids)
        hits[name] = (rank == 1).cpu().numpy()
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        r = {"R@1": round((rank == 1).sum().item() / m * 100, 2),
             "R@5": round((rank <= 5).sum().item() / m * 100, 2),
             "median_target_rank": int(rank.median()),
             "positive_cos": round(float(own.mean()), 4),
             "hardest_negative_cos": round(float(hard.mean()), 4),
             "top1_margin": round(float(marg.mean()), 4),
             "margin_pct": {f"p{int(x*100):02d}": round(
                 float(marg.quantile(x)), 4) for x in qs},
             "frac_margin_gt_0": round(float((marg > 0).float().mean()) * 100, 2)}
        res["arms"][name] = r
        print(f"{name:<26s}{r['R@1']:7.2f}{r['R@5']:7.2f}"
              f"{r['median_target_rank']:13d}{r['positive_cos']:10.4f}"
              f"{r['hardest_negative_cos']:10.4f}{r['top1_margin']:9.4f}"
              f"{r['frac_margin_gt_0']:12.2f}")
        return r

    score("我們的 render", ours)
    score("ULIP-2 官方 render", theirs)
    score("我們的 render, 打亂", [ours[i] for i in perm])

    print(f"\n邊際分位數 (positive - 最難負例):")
    for k in ("我們的 render", "ULIP-2 官方 render"):
        print(f"  {k:<22s}" + "  ".join(
            f"{a}={b:+.4f}" for a, b in res["arms"][k]["margin_pct"].items()))

    # [Codex] 217 paired samples: 3.7 points is not obviously significant.
    a, b = hits["我們的 render"], hits["ULIP-2 官方 render"]
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    if b01 + b10:
        chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from math import erfc, sqrt
        pval = erfc(sqrt(chi / 2))
    else:
        chi, pval = 0.0, 1.0
    res["mcnemar"] = {"ours_only": b01, "theirs_only": b10,
                      "chi2_cc": round(chi, 3), "p": round(pval, 4)}
    print(f"\nMcNemar 配對檢定 (n={len(uids)}): 只有我們對 {b01}，只有他們對 "
          f"{b10}，chi2 {chi:.3f}, p {pval:.4f}  "
          f"-> {'不顯著' if pval > 0.05 else '顯著'}")

    ch1, ch5 = 100.0 / len(corpus), 500.0 / len(corpus)
    res["chance"] = {"R@1": round(ch1, 5), "R@5": round(ch5, 5),
                     "paper_image_R1_over_chance": round(0.1 / ch1, 1),
                     "paper_image_R5_over_chance": round(1.3 / ch5, 1)}
    print(f"\n此池子的隨機期望 R@1 {ch1:.5f}%  R@5 {ch5:.5f}%")
    print(f"論文 ULIP image 0.1 / 1.3 = 隨機的 {0.1/ch1:.0f} 倍 / {1.3/ch5:.0f} 倍"
          " -- 是極弱訊號，不是完全打亂")

    a, b = res["arms"]["我們的 render"], res["arms"]["ULIP-2 官方 render"]
    if min(a["R@1"], b["R@1"]) > 25:
        v = ("兩組都高 -> render domain 排除，可疑的是 UID join 或未公開的 wrapper")
    elif b["R@1"] > 25 >= a["R@1"]:
        v = "官方高、我們低 -> 我們的 render/preprocessing 有 domain shift"
    elif a["R@1"] > 25 >= b["R@1"]:
        v = "我們高、官方低 -> 論文的 render domain 可能就是答案"
    else:
        v = "兩組都低 -> 之前 52.7 的 image 路徑跟這批 pixels 不是同一條"
    res["reading"] = v
    print(f"\n判讀: {v}")
    print(f"論文 ULIP 列的 image-only 是 0.1；打亂那一組是這個池子的地板")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
