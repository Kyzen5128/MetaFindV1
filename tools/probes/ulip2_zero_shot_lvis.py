#!/usr/bin/env python3
"""Is our ULIP-2 backbone the paper's ULIP-2? Zero-shot Objaverse-LVIS classification.

[KYZEN 2026-09-04] "你是不是有動 ulip2 的設計 導致分數異常過高 請去比對官方作法."

ULIP-2 (arXiv 2305.08275 v4), Table 10: Point-BERT pre-trained on Objaverse +
ShapeNet with OpenCLIP ViT-G, 10k xyzrgb input, zero-shot classification on
Objaverse-LVIS: top-1 50.6, top-5 79.1. That is the paper's own number for the
checkpoint we load, on the benchmark our corpus is drawn from, with NO fusion
and NO training of ours in the loop. If the released weights, our point-cloud
preprocessing and our text tower reproduce it, the backbone is intact; if we
had altered ULIP-2, this is where it would show.

Protocol, following ULIP's own zero-shot code: every LVIS category name is
put through the prompt templates ULIP ships (data/templates.json), the
template embeddings are averaged and L2-normalised, the point cloud embedding
is L2-normalised, and the prediction is the argmax cosine over all 1,156
categories. Point clouds are our canonical 10k xyzrgb clouds through the
released PointBERT + projection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

TEMPLATES = Path("/home/kyzen/upstream/ULIP/data/templates.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_ulip2_zero_shot_lvis.json")
    args = ap.parse_args()

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in uids}
    labels = [anns[u]["lvis_category"] for u in uids]
    # every LVIS class in the corpus, not only those present in the split --
    # the paper's benchmark scores over all 1,156 (ours holds what the 45,692
    # admitted assets cover)
    classes = sorted({json.loads(p.read_text())["lvis_category"]
                      for p in paths.ANNOTATIONS.glob("*.json")})
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[l] for l in labels])
    print(f"{len(uids):,} clouds, {len(classes):,} LVIS classes, "
          f"{len(set(labels)):,} present in {args.split}", flush=True)

    tpl = json.loads(TEMPLATES.read_text())
    # ULIP ships modelnet40_64 and shapenet_64; its Objaverse-LVIS zero-shot
    # run uses the 64 ModelNet templates ("a point cloud model of {}." etc.)
    key = "objaverse_lvis" if "objaverse_lvis" in tpl else "modelnet40_64"
    templates = tpl[key]
    print(f"templates: {key} ({len(templates)}), e.g. {templates[0]!r}", flush=True)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    with torch.no_grad():
        text = []
        for c in classes:
            name = c.replace("_", " ")
            e = bb.encode_text([t.format(name) for t in templates]).float()
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            text.append(torch.nn.functional.normalize(e, dim=0))
        text = torch.stack(text)                                    # (C, D)
        pcs = []
        for i in range(0, len(uids), 32):
            cl = []
            for u in uids[i:i + 32]:
                c = np.load(paths.POINTCLOUDS / f"{u}.npz")
                cl.append(np.concatenate([c["xyz"], c["rgb"]], axis=1).astype(np.float32))
            pcs.append(torch.nn.functional.normalize(bb.encode_pc(torch.from_numpy(np.stack(cl))).float(), dim=-1))
            if i % 1600 == 0:
                print(f"  pc {i:,}/{len(uids):,}", flush=True)
        pc = torch.cat(pcs)
        logits = (pc @ text.T).cpu().numpy()
    top5 = np.argsort(-logits, axis=1)[:, :5]
    top1 = float((top5[:, 0] == y).mean()); t5 = float((top5 == y[:, None]).any(1).mean())
    print(f"\nzero-shot Objaverse-LVIS on {args.split}: top-1 {top1*100:.1f}  top-5 {t5*100:.1f}")
    print("paper ULIP-2 Table 10 (Point-BERT, 10k xyzrgb, ViT-G):  top-1 50.6  top-5 79.1")
    out = {"split": args.split, "n": len(uids), "n_classes": len(classes), "top1": top1, "top5": t5,
           "paper_top1": 50.6, "paper_top5": 79.1, "templates": key}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
