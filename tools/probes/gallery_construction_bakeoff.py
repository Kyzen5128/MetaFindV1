#!/usr/bin/env python3
"""Four gallery constructions, one query, one pool. Which one lands on 13.8?

Codex, 2026-09-01, after checking ULIP-2's architecture and revising toward
Kyzen's reading:

> 最有判別力的下一個測試只有三組:
>   A. text query -> pure PC gallery
>   B. text query -> raw mean(T,I,PC) gallery
>   C. text query -> trained gallery-fuser(T,I,PC)
> 再對 C 做一次: C1 G(T,I,P), C2 G(mask_text,I,P)
> 如果 C1 約 13.8、C2 幾乎不變, 就是 gallery fuser 實際由 PC 主導;
> 如果 A 約 13.8、C 明顯更高, 則 Table 1 很可能實際用了純 PC gallery.

Two things he established that reframe this, and both correct me.

**ULIP-2 has no fused gallery at all.** It aligns three encoders into one space
and returns three vectors (`ULIP_models.py:194`); there is no
`Fusion(text, image, pc)` anywhere upstream. So CAMERA's `text @ pc.T` is not
an alternative reading, it is the ordinary way this backbone is used.

**And `gallery = mean(T,I,P)` was our over-reading.** MetaFind has learned
query and gallery fusion heads. With `q = F_query(text)` and
`g = F_gallery(text, image, pc)` there is no `text·text = 1` term even though
the gallery's input contains the query's text -- two learned heads can rotate
or compress it away. Our 99.56 control therefore proves the RAW MEAN
implementation has a shortcut; it does not prove a learned fuser must.

He also withdrew "coincidence" on the CAMERA number: 13.16 against MetaFind's
13.80 is 0.64 of a point, from a pure-PC gallery, one positive per object, UID
ranking, in the same aligned space. That is a data point, not a numerical
accident.

THE ARMS
--------
Identical query, identical pool, identical checkpoint. Only the gallery differs.

  A   pure point-cloud embedding                        (CAMERA's construction)
  B   raw unnormalised mean of the three ULIP vectors   (our parameter-free control)
  C   the TRAINED gallery fuser, all three present      (what we report)
  C2  the trained gallery fuser with text MASKED        (its own mask token in
                                                         the text slot -- not a
                                                         derangement, an absence)

Each run twice on the query side, because the two answer different questions:

  same caption        the asset's canonical text, i.e. the text the gallery
                      also holds. This is what protocol B reports.
  independent caption a non-canonical `description_candidate`. Removes the
                      exact identity; leaves the 0.85 correlate Codex flagged.

9,138 test queries against all 45,692, so the pool is MetaFind's scale.
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
from metafind.models import resolve_stage1 as R  # noqa: E402

OUT = REPO / "output" / "look" / "gallery_construction_bakeoff.json"
PC_CACHE = paths.OUTPUTS / "_probe" / "released_pc_embeddings.npy"
MODS = ("text", "image", "pc")
PAPER = {"metafind_text_only": 13.8, "camera_T2S": 13.16}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind/metafind_out/checkpoints/"
                                      "qpack_ti_lr2.50e-04_s20260816/stage1_best.pt")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    queries = sorted(split["test"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"loaded {ck.name}  scope {scope}", flush=True)

    # --- gallery inputs, whole corpus --------------------------------------
    G_t = np.empty((len(corpus), 1280), np.float32)
    G_i = np.empty((len(corpus), 1280), np.float32)
    for k, u in enumerate(corpus):
        z = np.load(paths.EMBEDDINGS / f"{u}.npz")
        G_t[k] = z["text"]
        G_i[k] = z["image"]
        if (k + 1) % 15000 == 0:
            print(f"  cached {k + 1:,}/{len(corpus):,}", flush=True)
    # The pc the TRAINED backbone produces, not the released one: the gallery
    # fuser was trained against its own point encoder's output.
    G_p = np.empty((len(corpus), 1280), np.float32)
    with torch.no_grad():
        buf = []
        for k, u in enumerate(corpus):
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            buf.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
            if len(buf) == args.batch or k == len(corpus) - 1:
                G_p[k - len(buf) + 1:k + 1] = bb.encode_pc(
                    np.stack(buf)).float().cpu().numpy()
                buf = []
            if (k + 1) % 15000 == 0:
                print(f"  encoded pc {k + 1:,}/{len(corpus):,}", flush=True)

    Tt = torch.from_numpy(G_t).to(dev)
    Ti = torch.from_numpy(G_i).to(dev)
    Tp = torch.from_numpy(G_p).to(dev)
    n = lambda x: torch.nn.functional.normalize(x, dim=-1)

    def fuse(mask_text: bool) -> torch.Tensor:
        out = torch.empty(len(corpus), 1280, device=dev)
        with torch.no_grad():
            for a in range(0, len(corpus), args.batch):
                b = min(a + args.batch, len(corpus))
                e = {"text": Tt[a:b], "image": Ti[a:b], "pc": Tp[a:b]}
                if mask_text:
                    p = torch.tensor([0, 1, 1], dtype=torch.bool,
                                     device=dev).expand(b - a, -1).clone()
                    out[a:b] = model.gallery.fusion(e, present=p)
                else:
                    out[a:b] = model.gallery(e)
        return n(out)

    GAL = {
        "A  pure pc": n(Tp),
        "B  raw mean(T,I,P)": n((Tt + Ti + Tp) / 3.0),
        "C  trained fuser": fuse(False),
        "C2 trained fuser, text masked": fuse(True),
    }
    print("galleries built", flush=True)

    # --- query text: canonical and independent ------------------------------
    same, indep, keep = [], [], []
    for u in queries:
        a = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
        alt = [c for c in (a.get("description_candidates") or [])
               if c.get("rank", 0) >= 1 and c.get("text")]
        if not alt:
            continue
        keep.append(u)
        same.append(R.serialize_annotation(a))
        b = dict(a)
        b["description"] = alt[0]["text"]
        indep.append(R.serialize_annotation(b))
    print(f"{len(keep):,} queries ({len(queries) - len(keep)} dropped, "
          f"no alternate caption)", flush=True)

    def enc_text(strs):
        out = torch.empty(len(strs), 1280, device=dev)
        with torch.no_grad():
            for a in range(0, len(strs), args.batch):
                b = min(a + args.batch, len(strs))
                out[a:b] = bb.encode_text(strs[a:b]).float()
        return out

    tgt = torch.tensor([pos[u] for u in keep], device=dev)
    raw_q = {"same caption": enc_text(same), "independent caption": enc_text(indep)}

    res = {"checkpoint": str(ck), "train_scope": scope,
           "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
           "n_query": len(keep), "n_gallery": len(corpus),
           "reference": PAPER, "chance_R1": round(100.0 / len(corpus), 5),
           "results": {}}

    print(f"\n{'畫廊建構':<32s}" + "".join(f"{k:>22s}" for k in raw_q))
    print(f"{'':<32s}" + "".join(f"{'R@1':>10s}{'MRR':>12s}" for _ in raw_q))
    for gname, G in GAL.items():
        row, cells = {}, ""
        for qname, Qraw in raw_q.items():
            # the query goes through the trained QUERY tower for C/C2, and raw
            # for A/B, because A and B have no learned gallery to match.
            if gname.startswith(("A", "B")):
                Q = n(Qraw)
            else:
                p = torch.tensor([1, 0, 0], dtype=torch.bool,
                                 device=dev).expand(len(keep), -1).clone()
                zero = torch.zeros_like(Qraw)
                with torch.no_grad():
                    o = torch.empty(len(keep), 1280, device=dev)
                    for a in range(0, len(keep), args.batch):
                        b = min(a + args.batch, len(keep))
                        o[a:b] = model.query(
                            {"text": Qraw[a:b], "image": zero[a:b],
                             "pc": zero[a:b]}, present=p[a:b])
                Q = n(o)
            s = Q @ G.t()
            own = s.gather(1, tgt.unsqueeze(1))
            rank = (s > own).sum(1) + 1
            r = rank.float()
            row[qname] = {"r1": round(float((r <= 1).float().mean()) * 100, 2),
                          "mrr": round(float((1 / r).mean()) * 100, 2),
                          "r5": round(float((r <= 5).float().mean()) * 100, 2),
                          "median_rank": int(r.median())}
            cells += f"{row[qname]['r1']:10.2f}{row[qname]['mrr']:12.2f}"
        res["results"][gname] = row
        print(f"{gname:<32s}{cells}")
    print(f"{'MetaFind text-only':<32s}{13.8:10.2f}")
    print(f"{'CAMERA T2S (pure pc gallery)':<32s}{13.16:10.2f}")
    print(f"{'chance':<32s}{100.0/len(corpus):10.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
