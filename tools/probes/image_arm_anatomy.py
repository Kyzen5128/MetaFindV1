#!/usr/bin/env python3
"""Why is our image query strong? Split the answer into leak and likeness.

Kyzen, 2026-09-01: 「image 我不懂 我是強在哪裡?」

The query pack's image arm draws ONE view, `views[uid_seed(uid) % 12]`; the
gallery keeps the 12-view mean. So the query's own view sits inside its own
gallery entry at weight 1/12, and `QueryPack`'s docstring names that as an
unremoved leak. Independent-query image measured 56.7 R@1 against the paper's
11.7, and the open question is whether the 1/12 is what carries it.

Two candidate explanations, and they call for different work:

  LEAK      the query view is inside the gallery mean. Removable in principle,
            at the cost of a per-query gallery.
  LIKENESS  twelve renders of one object simply resemble each other and not
            other objects. Nothing to remove -- it is what a render IS.

They are separable. Score the same query view against a gallery whose target
entry excludes exactly that view and changes nothing else:

  A  gallery_j = mean of asset j's 12 views                     (production)
  B  identical, except gallery_i = mean of asset i's OTHER 11   (leak removed)

The drop from A to B is the whole leak. Whatever B still scores is likeness,
and no gallery construction can take it away.

Reported with the cosines behind the ranks: cos(view_k, mean12) against
cos(view_k, mean11-without-k), and cos(view_k, view_j) for j != k, which is
likeness measured directly with no mean involved at all.

Protocol matches `fig2_text_form`: the 9,138 test uids as queries, all 45,692
as the gallery, n06's cached view vectors, no Stage 1 weights, nothing trained.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "image_arm_anatomy.json"
PAPER_IMAGE_R1 = 11.7


# The view rule is IMPORTED, not transcribed. A first draft of this probe
# rewrote it as int(sha256(uid).hexdigest()[:8], 16) -- four bytes where the
# real one takes eight -- which picks a different view for most assets and
# would have measured a leak the query pack never had.
from metafind.data.pointclouds import uid_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"queries {len(queries):,}  gallery {len(corpus):,}  device {dev}",
          flush=True)

    n, D = len(corpus), 1280
    sum12 = np.empty((n, D), np.float32)
    for a in range(0, n, args.chunk):
        b = min(a + args.chunk, n)
        for i in range(a, b):
            sum12[i] = np.load(paths.EMBEDDINGS / f"{corpus[i]}.npz")["views"] \
                         .astype(np.float32).sum(0)
        if a % (args.chunk * 4) == 0:
            print(f"  loaded {b:,}/{n:,}", flush=True)

    k_of = {u: uid_seed(u) % 12 for u in queries}
    qv = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"]
                   .astype(np.float32)[k_of[u]] for u in queries])

    S = torch.from_numpy(sum12).to(dev)
    Qraw = torch.from_numpy(qv).to(dev)              # UNnormalised, for the subtraction
    Q = torch.nn.functional.normalize(Qraw, dim=-1)
    tgt = torch.tensor([pos[u] for u in queries], device=dev)

    G12 = torch.nn.functional.normalize(S / 12.0, dim=-1)
    # The target row only: the sum of its twelve minus the one the query drew,
    # over eleven. Every other row is untouched, so A and B differ in exactly
    # one number per query and the difference between them is the whole leak.
    G11_diag = torch.nn.functional.normalize((S[tgt] - Qraw) / 11.0, dim=-1)

    res = {"n_query": len(queries), "n_gallery": n,
           "paper_image_R1": PAPER_IMAGE_R1, "arms": {}, "cosines": {}}

    def rank_r1_r5(scores, diag_override=None):
        own = scores.gather(1, tgt.unsqueeze(1))
        if diag_override is not None:
            own = diag_override.unsqueeze(1)
            scores = scores.clone()
            scores.scatter_(1, tgt.unsqueeze(1), own)
        higher = (scores > own).sum(1)
        m = scores.shape[0]
        return ((higher < 1).sum().item() / m * 100,
                (higher < 5).sum().item() / m * 100)

    s = Q @ G12.t()
    a1, a5 = rank_r1_r5(s)
    res["arms"]["A_gallery_mean12"] = {"R@1": round(a1, 2), "R@5": round(a5, 2)}

    diag11 = (Q * G11_diag).sum(1)
    b1, b5 = rank_r1_r5(s, diag_override=diag11)
    res["arms"]["B_target_excludes_the_query_view"] = {"R@1": round(b1, 2),
                                                       "R@5": round(b5, 2)}

    cos12 = (Q * G12[tgt]).sum(1)
    idx = torch.arange(len(queries), device=dev)
    other = torch.stack([
        torch.nn.functional.normalize(
            torch.from_numpy(np.load(paths.EMBEDDINGS / f"{u}.npz")["views"]
                             .astype(np.float32)[(k_of[u] + 6) % 12]).to(dev),
            dim=0)
        for u in queries])
    cos_vv = (Q * other).sum(1)
    for tag, v in (("view_k vs its own mean12", cos12),
                   ("view_k vs its own mean11 without k", diag11),
                   ("view_k vs the opposite view k+6", cos_vv)):
        res["cosines"][tag] = {"mean": round(float(v.mean()), 4),
                               "p5": round(float(v.quantile(0.05)), 4),
                               "p95": round(float(v.quantile(0.95)), 4)}

    print(f"\n{'arm':<38s}{'R@1':>9s}{'R@5':>9s}")
    for k, v in res["arms"].items():
        print(f"{k:<38s}{v['R@1']:9.2f}{v['R@5']:9.2f}")
    print(f"{'論文 image':<38s}{PAPER_IMAGE_R1:9.1f}")
    print(f"\n洩漏值 = A - B = {a1 - b1:.2f} 分")
    print(f"\n{'cosine':<38s}{'平均':>9s}{'p5':>9s}{'p95':>9s}")
    for k, v in res["cosines"].items():
        print(f"{k:<38s}{v['mean']:9.4f}{v['p5']:9.4f}{v['p95']:9.4f}")
    res["leak_points"] = round(a1 - b1, 2)
    res["likeness_points"] = round(b1, 2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
