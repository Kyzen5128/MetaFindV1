#!/usr/bin/env python3
"""Which modality does the TRAINED gallery tower actually use? Derange its inputs.

Codex, 2026-09-01, rejecting the cosine reading as insufficient:

> gallery-side derangement: G(T,I,P), G(shuffle(T),I,P), G(T,shuffle(I),P),
> G(T,I,shuffle(P)). It answers directly which modality the trained gallery
> uses, more reliably than looking at fusion attention or cosine.

He is right that `gallery_geometry`'s centred cosine (text .454, image .559,
pc .801) shows where the output SITS, not what it USES. A tower could sit near
pc and still route decisive information from text. Deranging one input at a
time and re-scoring separates the two.

It also settles a claim of mine he rejected. I wrote that a modality-complete
fused gallery "should" preserve its inputs, so MetaFind's 13.8 could not come
from one. He pointed out that a learned non-linear fusion is free to be
pc-dominant -- and that we had already measured exactly that at 0.801 without
noticing it was the counterexample to our own argument. If deranging text moves
retrieval by almost nothing, then a fused gallery CAN behave like a pc gallery,
MetaFind's 13.8 is compatible with the modality-complete text, and the
"pure-gallery only" reading dies here.

WHAT IS DERANGED, AND WHAT IS NOT
---------------------------------
Only the GALLERY's input for one modality, replaced by another asset's vector
of the same modality -- same distribution, same norms, no instance signal. The
QUERY is untouched in every arm, and so is the gallery's other two modalities.
So the drop from the intact arm is that modality's contribution to the gallery
representation, per query direction.

Read it against two references printed alongside:

  intact          what the gallery normally scores
  all deranged    the floor: a gallery carrying no instance signal at all

A modality whose derangement costs nothing is a modality the gallery ignores.
A modality whose derangement costs everything is the one it runs on.

Trained checkpoint, its own promoted index's checkpoint, so the tower measured
is the tower Table 1 was scored with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "gallery_modality_ablation.json"
MODS = ("text", "image", "pc")
CONDS = {"text": (1, 0, 0), "image": (0, 1, 0), "pc": (0, 0, 1),
         "full": (1, 1, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind_out/checkpoints/"
                                      "qpack_ti_lr2.50e-04_s20260816/stage1_best.pt")
    ap.add_argument("-n", type=int, default=4000, help="assets, query AND gallery")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    pool = sorted(set(split["train"]) | set(split["test"]))
    rng = np.random.default_rng(args.seed)
    sel = [pool[i] for i in sorted(rng.choice(len(pool), min(args.n, len(pool)),
                                              replace=False))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(sel):,} assets, used as both query and gallery", flush=True)

    ck = pathlib.Path(args.ckpt)
    scope = torch.load(ck, map_location="cpu",
                       weights_only=False).get("train_scope",
                                               "point_encoder_and_fuser")
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import (build_model, load_protocols,
                                       load_stage1_checkpoint)
    bb = ULIPBackbone(BackboneConfig(device=dev, train_scope=scope))
    enc, tr, hy = load_protocols()
    model, lf = build_model(enc, tr, hy)
    model.to(dev)
    load_stage1_checkpoint(bb, model, lf, ck)
    model.eval()
    print(f"loaded {ck.name}  train_scope {scope}", flush=True)

    txt, img, clouds = [], [], []
    for u in sel:
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        txt.append(z["text"].astype(np.float32))
        img.append(z["image"].astype(np.float32))
        c = np.load(paths.POINTCLOUDS / f"{u}.npz")
        clouds.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
    T = torch.from_numpy(np.stack(txt)).to(dev)
    I = torch.from_numpy(np.stack(img)).to(dev)
    P = torch.empty(len(sel), T.shape[1], device=dev)
    with torch.no_grad():
        for a in range(0, len(sel), args.batch):
            b = min(a + args.batch, len(sel))
            P[a:b] = bb.encode_pc(np.stack(clouds[a:b])).float()
    RAW = {"text": T, "image": I, "pc": P}

    perm = rng.permutation(len(sel))
    fixed = np.flatnonzero(perm == np.arange(len(sel)))
    if len(fixed):
        perm[fixed] = perm[(fixed + 1) % len(perm)]
    perm_t = torch.from_numpy(perm).to(dev)

    n = lambda x: torch.nn.functional.normalize(x, dim=-1)
    tgt = torch.arange(len(sel), device=dev)

    def gallery(deranged: tuple[str, ...]) -> torch.Tensor:
        embeds = {m: (RAW[m][perm_t] if m in deranged else RAW[m]) for m in MODS}
        with torch.no_grad():
            out = torch.empty(len(sel), T.shape[1], device=dev)
            for a in range(0, len(sel), args.batch):
                b = min(a + args.batch, len(sel))
                out[a:b] = model.gallery({m: v[a:b] for m, v in embeds.items()})
        return n(out)

    def query(cond) -> torch.Tensor:
        present = torch.tensor(CONDS[cond], dtype=torch.bool,
                               device=dev).expand(len(sel), -1).clone()
        with torch.no_grad():
            out = torch.empty(len(sel), T.shape[1], device=dev)
            for a in range(0, len(sel), args.batch):
                b = min(a + args.batch, len(sel))
                out[a:b] = model.query({m: v[a:b] for m, v in RAW.items()},
                                       present=present[a:b])
        return n(out)

    Q = {c: query(c) for c in CONDS}

    arms = [("intact", ()), ("text deranged", ("text",)),
            ("image deranged", ("image",)), ("pc deranged", ("pc",)),
            ("all deranged", ("text", "image", "pc"))]
    res = {"checkpoint": str(ck), "train_scope": scope, "n": len(sel),
           "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
           "note": "only the GALLERY's input is deranged; the query is intact "
                   "in every arm",
           "chance_R1": round(100.0 / len(sel), 4), "arms": {}}

    print(f"\n{'畫廊輸入':<18s}" + "".join(f"{c:>12s}" for c in CONDS))
    base = {}
    for name, der in arms:
        G = gallery(der)
        row = {}
        for c in CONDS:
            s = Q[c] @ G.t()
            own = s.gather(1, tgt.unsqueeze(1))
            row[c] = round(((s > own).sum(1) < 1).sum().item() / len(sel) * 100, 2)
        res["arms"][name] = row
        if name == "intact":
            base = row
        delta = "" if name == "intact" else "   Δ " + " ".join(
            f"{c}:{row[c]-base[c]:+.1f}" for c in CONDS)
        print(f"{name:<18s}" + "".join(f"{row[c]:12.2f}" for c in CONDS) + delta)
    print(f"{'隨機':<18s}" + "".join(f"{100.0/len(sel):12.4f}" for _ in CONDS))

    d = {m: {c: round(base[c] - res["arms"][f"{m} deranged"][c], 2)
             for c in CONDS} for m in MODS}
    res["contribution"] = d
    lead = max(MODS, key=lambda m: d[m]["full"])
    res["gallery_runs_on"] = lead
    print(f"\n每個模態對畫廊的貢獻（intact 減去打亂該模態）:")
    for m in MODS:
        print(f"  {m:<8s}" + "  ".join(f"{c} {d[m][c]:+7.2f}" for c in CONDS))
    print(f"\n畫廊主要靠: {lead}")
    print("若打亂 text 幾乎不掉 -> 融合後的畫廊確實可以表現得像純點雲畫廊，"
          "MetaFind 的 13.8 就與 modality-complete 的說法相容")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
