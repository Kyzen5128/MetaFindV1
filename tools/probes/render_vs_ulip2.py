#!/usr/bin/env python3
"""n04: do our 12 renders sit at the same 12 cameras as ULIP-2's published ones?

The layout was already checked by reading code -- `render_blender.py:103` lists
the same phi 60/90/120 rings of four that OpenShape's official
`render_single_glb.py:172` does, degree for degree. This checks the OUTPUT
instead, which is what actually matters, using two signals that a code diff
cannot give:

1. **Per-view accuracy profile.** Our own 45,692-asset run found a strong
   structure: views 0-3 average 48.4 acc1, views 4-7 46.1, views 8-11 **35.1**.
   The bottom ring is 13 points down because looking up at an object from
   underneath is a poor view. If their `image_feat` shows the same profile at
   the same indices, the camera ORDER matches too, not just the set of angles.
   Same profile at shifted indices would mean the same cameras in a different
   order; no profile at all would mean a different layout.

2. **Per-view cosine, ours against theirs, same asset same index.** Two renders
   of one object from one camera encode to nearly the same vector; two different
   cameras do not. The full 12x12 matrix is printed so a permutation would show
   up as an off-diagonal maximum rather than being hidden by the average.

Both sides go through the same OpenCLIP ViT-bigG, so the encoder cannot be the
difference. Theirs is already encoded, ours is read from the n06 sidecars.
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

OUT = REPO / "output" / "look" / "render_vs_ulip2.json"
LVIS = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/objaverse-lvis/"
                    "objaverse_lvis_metadata.json")
TEMPLATES = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/templates.json")
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shard0")
RINGS = {"0-3 上排 phi60": range(0, 4), "4-7 中排 phi90": range(4, 8),
         "8-11 下排 phi120": range(8, 12)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=SHARDS)
    args = ap.parse_args()

    theirs = {os.path.basename(f)[:-4]: f
              for f in glob.glob(os.path.join(args.shards, "*", "*.npy"))}
    meta = json.loads(LVIS.read_text())
    names, v2k, k2id = meta["all_keys"], meta["value_to_key_mapping"], meta["key_to_id"]

    keep, labels = [], []
    for u in sorted(theirs):
        p = paths.EMBEDDINGS / f"{u}.npz"
        if p.exists() and v2k.get(u) in k2id:
            keep.append(u)
            labels.append(k2id[v2k[u]])
    print(f"{len(keep):,} assets with both sides' views and an LVIS label", flush=True)

    tv = np.stack([np.asarray(np.load(theirs[u], allow_pickle=True).item()["image_feat"],
                              np.float32) for u in keep])          # (N,12,1280)
    ov = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"].astype(np.float32)
                   for u in keep])
    print(f"their views {tv.shape}   our views {ov.shape}", flush=True)
    if tv.shape[1] != ov.shape[1]:
        print(f"!! view counts differ: {tv.shape[1]} vs {ov.shape[1]}", flush=True)

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    bb = ULIPBackbone(BackboneConfig(train_scope="frozen"))
    templates = json.loads(TEMPLATES.read_text())["modelnet40_64"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    protos = []
    with torch.no_grad():
        for i, n in enumerate(names):
            e = bb.encode_text([t.format(n) for t in templates]).float()
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            protos.append(torch.nn.functional.normalize(e, dim=-1))
            if (i + 1) % 300 == 0:
                print(f"  prototypes {i + 1}/{len(names)}", flush=True)
    P = torch.stack(protos).to(dev)
    tgt = torch.tensor(labels, device=dev)

    def acc1(a):
        f = torch.nn.functional.normalize(torch.from_numpy(a).to(dev), dim=-1)
        lg = f @ P.t()
        own = lg.gather(1, tgt.unsqueeze(1))
        return ((lg > own).sum(1) < 1).sum().item() / a.shape[0] * 100

    res = {"n": len(keep), "per_view_acc1": {}, "ring_mean_acc1": {}}
    print(f"\n{'視角':>6s} {'ULIP-2 官方':>12s} {'我們':>10s}")
    ta, oa = [], []
    for k in range(tv.shape[1]):
        a, b = acc1(tv[:, k]), acc1(ov[:, k])
        ta.append(a)
        oa.append(b)
        res["per_view_acc1"][k] = {"theirs": round(a, 2), "ours": round(b, 2)}
        print(f"{k:6d} {a:12.2f} {b:10.2f}")
    for tag, r in RINGS.items():
        m_t = float(np.mean([ta[i] for i in r]))
        m_o = float(np.mean([oa[i] for i in r]))
        res["ring_mean_acc1"][tag] = {"theirs": round(m_t, 2), "ours": round(m_o, 2)}
        print(f"{tag:>16s} {m_t:8.2f} {m_o:8.2f}")

    # 12x12: is view k of ours the same camera as view k of theirs?
    T = torch.nn.functional.normalize(torch.from_numpy(tv).to(dev), dim=-1)
    O = torch.nn.functional.normalize(torch.from_numpy(ov).to(dev), dim=-1)
    M = torch.einsum("nid,njd->ij", T, O).cpu().numpy() / len(keep)
    res["cross_view_cosine"] = M.round(4).tolist()
    res["argmax_per_their_view"] = M.argmax(1).tolist()
    print("\n12x12 平均 cosine（列=他們的視角，行=我們的視角）")
    print("      " + "".join(f"{j:6d}" for j in range(M.shape[1])))
    for i in range(M.shape[0]):
        star = M[i].argmax()
        print(f"  {i:3d} " + "".join(
            (f"[{M[i, j]:4.2f}]" if j == star else f" {M[i, j]:5.2f}")
            for j in range(M.shape[1])))
    print(f"\n每一列的最大值落在: {M.argmax(1).tolist()}")
    print("對角線 = 相機順序一致；其他 = 順序被打亂或相機不同")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
