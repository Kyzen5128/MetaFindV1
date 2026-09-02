#!/usr/bin/env python3
"""Can ULIP-2's released per-view image features stand in for ours? Measured.

Codex, 2026-09-02: "1280-d and a same-asset diagonal support a compatible
ViT-bigG space, but the file carries no checkpoint id. Before use, check
normalisation, feature-norm distribution, same-image cosine against the
current query features, retrieval ranking, and whether an extra projection is
needed." This is that check, on every asset both sides hold.

THEIRS   `objaverse_lvis/<uid>.npy` -> `image_feat` (12, 1280) float32, from the
         extracted shards in the scratchpad. Cameras: a single ring at 30
         degree steps (ULIP-2 section 4.1).
OURS     `embeddings/<uid>.npz` -> `views` (12, 1280) float16, ViT-bigG-14 via
         the ULIP-2 tower, three polar rings of four (OpenShape layout).

The camera sets are NOT the same twelve poses, so a per-index comparison is
meaningless. Per asset, each of their views is matched to our most similar
view (max cosine over our twelve); the matched cosine says how close the
nearest pose is in the shared space. A retrieval test then asks the only
question that matters for a gallery: with OUR canonical text as the query,
does a gallery built from THEIR twelve-view mean rank the right asset as
well as one built from OURS.

Reads cached features only; no encoder runs. CPU.
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

OUT = REPO / "output" / "look" / "official_image_feat_compat.json"
SHARDS = pathlib.Path("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
                      "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shards")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-assets", type=int, default=3000)
    ap.add_argument("--shards", default=str(SHARDS))
    args = ap.parse_args()

    theirs = {p.stem: p for p in pathlib.Path(args.shards).glob("*/*.npy")}
    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    both = [u for u in corpus if u in theirs][: args.max_assets]
    if not both:
        raise SystemExit("no overlap between the extracted shards and the corpus")
    print(f"{len(both):,} assets held by both sides (of {len(theirs):,} extracted)")

    T_views, O_views, O_text = [], [], []
    for u in both:
        d = np.load(theirs[u], allow_pickle=True).item()
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        T_views.append(np.asarray(d["image_feat"], np.float32))
        O_views.append(z["views"].astype(np.float32))
        O_text.append(z["text"].astype(np.float32))
    T = torch.from_numpy(np.stack(T_views))   # (N, 12, 1280)
    O = torch.from_numpy(np.stack(O_views))   # (N, 12, 1280)
    X = torch.from_numpy(np.stack(O_text))    # (N, 1280)

    # --- norms: are theirs unit-normalised, raw, or scaled? ------------------
    tn, on = T.norm(dim=-1), O.norm(dim=-1)
    norms = {"theirs_mean": round(float(tn.mean()), 4), "theirs_std": round(float(tn.std()), 4),
             "ours_mean": round(float(on.mean()), 4), "ours_std": round(float(on.std()), 4)}

    # --- nearest-pose cosine, per their view --------------------------------
    n = torch.nn.functional.normalize
    Tn, On = n(T, dim=-1), n(O, dim=-1)
    cross = torch.einsum("nid,njd->nij", Tn, On)          # (N, 12, 12)
    best = cross.max(dim=2).values                        # their view -> our closest
    same_asset_best = {"mean": round(float(best.mean()), 4),
                       "p05": round(float(best.flatten().kthvalue(int(0.05 * best.numel())).values), 4),
                       "min": round(float(best.min()), 4)}
    # against a DIFFERENT asset's closest view, as the floor
    perm = torch.roll(torch.arange(len(both)), 1)
    other = torch.einsum("nid,njd->nij", Tn, On[perm]).max(dim=2).values
    other_asset_best = round(float(other.mean()), 4)

    # --- mean-of-views gallery, text query ----------------------------------
    G_theirs, G_ours = n(T.mean(1), dim=-1), n(O.mean(1), dim=-1)
    Q = n(X, dim=-1)
    tgt = torch.arange(len(both))

    def r_at(G):
        s = Q @ G.t()
        own = s.gather(1, tgt.unsqueeze(1))
        rank = (s > own).sum(1) + 1
        return {"r1": round(float((rank <= 1).float().mean()) * 100, 2),
                "r5": round(float((rank <= 5).float().mean()) * 100, 2),
                "median_rank": int(rank.median())}

    gallery_mean_cos = round(float((G_theirs * G_ours).sum(-1).mean()), 4)
    res = {"n_assets": len(both), "pool": len(both),
           "norms": norms,
           "nearest_pose_cosine_same_asset": same_asset_best,
           "nearest_pose_cosine_other_asset": other_asset_best,
           "twelve_view_mean_cosine_theirs_vs_ours": gallery_mean_cos,
           "text_query_R": {"gallery_theirs": r_at(G_theirs),
                            "gallery_ours": r_at(G_ours)},
           "reading": "same space if the same-asset nearest-pose cosine sits far "
                      "above the other-asset floor, the two twelve-view means "
                      "agree, and text->image retrieval lands within a point or "
                      "two of ours. A norm scale is harmless after normalisation;"
                      " a projection would show as low cosine with intact ranking"}
    print(json.dumps(res, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
