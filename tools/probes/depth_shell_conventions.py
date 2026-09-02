#!/usr/bin/env python3
"""Two conventions in the ProcTHOR depth-shell path, measured through PointBERT.

Codex, 2026-09-02, on two audit findings that were argued and not measured:

  * grey  -- `prepare_depth_shell` colours the shell 0.5; ULIP's stand-in for a
             cloud with no colour is 0.4 (`vendor/ulip/data/dataset_3d.py:292`,
             `pointclouds.DEFAULT_GREY`). "Measure the embedding effect first,
             then decide the priority."
  * hand  -- AI2-THOR is a left-handed y-up frame and nothing mirrors the
             unprojected points into the right-handed frame Objaverse clouds
             are loaded in (`meshload.py`). "Needs a geometric unit test."

Both are answered the same way: encode the same clouds under both settings
with the FROZEN released PointBERT and report (a) the cosine between the two
embeddings of each asset and (b) whether the asset would still find itself --
for every asset, is its own other-setting embedding the nearest neighbour
among all 1,439, and what rank does it get otherwise.

For the handedness test the mirror is applied to OBJAVERSE clouds (whose frame
is known to be the encoder's), so the question "does a reflection change what
PointBERT returns" is asked on data the encoder was trained on, not on data
whose frame is already in doubt. Bilaterally symmetric objects are expected
to be invariant; chiral ones are not, and the spread says how much.

Reads only. Writes its own JSON under output/look.
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

OUT = REPO / "output" / "look" / "depth_shell_conventions.json"


def encode(bb, clouds: list[np.ndarray], batch: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for a in range(0, len(clouds), batch):
            out.append(bb.encode_pc(np.stack(clouds[a:a + batch])).float())
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


def self_rank(A: torch.Tensor, B: torch.Tensor) -> dict:
    """For each row i of A, where does B[i] rank among all rows of B."""
    s = A @ B.t()
    own = s.diag().unsqueeze(1)
    rank = (s > own).sum(1) + 1
    cos = s.diag()
    return {"cos_mean": round(float(cos.mean()), 4),
            "cos_min": round(float(cos.min()), 4),
            "cos_p05": round(float(cos.kthvalue(max(1, int(0.05 * len(cos)))).values), 4),
            "self_is_top1": round(float((rank == 1).float().mean()) * 100, 2),
            "median_rank": int(rank.median()),
            "worst_rank": int(rank.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-objaverse", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from metafind.models.ulip_backbone import (BackboneConfig, ULIPBackbone,
                                               pc_norm)
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    assert bb.is_frozen()

    # ---- grey: the 1,439 ProcTHOR shells under 0.5 and 0.4 -----------------
    recs = [json.loads(p.read_text())
            for p in sorted(paths.PROCTHOR_MODALITIES.glob("*.json"))]
    recs = [r for r in recs if r["pointcloud_uri"] is not None]
    xyz = [pc_norm(np.load(r["pointcloud_uri"])["xyz"].astype(np.float32))
           for r in recs]
    shells = {}
    for grey in (0.5, 0.4):
        shells[grey] = encode(
            bb, [np.concatenate([x, np.full_like(x, grey)], 1) for x in xyz],
            args.batch)
    grey_stats = self_rank(shells[0.5], shells[0.4])
    print(f"grey 0.5 vs 0.4 over {len(recs):,} ProcTHOR shells: {grey_stats}")

    # ---- handedness: Objaverse clouds, original vs z-mirrored ---------------
    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    rng = np.random.default_rng(20260902)
    uids = [split["test"][i] for i in sorted(rng.choice(len(split["test"]),
                                                       args.n_objaverse,
                                                       replace=False))]
    orig, mirr = [], []
    for u in uids:
        c = np.load(paths.POINTCLOUDS / f"{u}.npz")
        pc = np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32)
        orig.append(pc)
        m = pc.copy()
        m[:, 2] *= -1.0
        mirr.append(m)
    E_o, E_m = encode(bb, orig, args.batch), encode(bb, mirr, args.batch)
    hand_stats = self_rank(E_o, E_m)
    print(f"z-mirror over {len(uids)} Objaverse test clouds: {hand_stats}")

    res = {"grey_0.5_vs_0.4_procthor": {"n": len(recs), **grey_stats},
           "z_mirror_objaverse": {"n": len(uids), **hand_stats},
           "reading": {
               "grey": "cos_mean near 1 and self_is_top1 near 100 mean the "
                       "constant is immaterial to the encoder; otherwise "
                       "0.4 (ULIP's convention) should be adopted and the "
                       "Stage 2 index rebuilt",
               "hand": "cos_mean near 1 means PointBERT is reflection-"
                       "insensitive on this corpus and the Unity frame needs "
                       "no mirror; a low tail means chiral assets are "
                       "encoded differently and the shell must be mirrored"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
