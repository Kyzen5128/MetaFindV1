#!/usr/bin/env python3
"""Encode an alternative QUERY text per asset with the frozen CLIP text tower.

[KYZEN 2026-09-04] The paper's query text is not the gallery's description
(ULIP row text-only 0.1); Figure 1 shows `Platform Bed {size: ...}`, i.e. the
Sketchfab model NAME plus size. Sources (from OpenShape's objaverse_meta.json,
which covers all 45,692 assets):
  sketchfab_name        "Rusty Lantern"
  sketchfab_name_size   "Rusty Lantern {size: 30 x 30 x 60 cm}"
Writes data/outputs/_probe/text_override/<source>.npz with `uids` and `vecs`
(float16, 1280-d, the raw CLIP output like the n06 cache). Consumed by
Stage1Dataset(text_override=...) on the query side only; gallery untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

META = Path("/home/kyzen/upstream/openshape-objaverse-embeddings/objaverse_meta.json")
SOURCES = ("sketchfab_name", "sketchfab_name_size", "cat_size", "cat_only")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="sketchfab_name", choices=SOURCES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(set(sp["train"]) | set(sp["test"]))
    meta = {}
    if args.source.startswith("sketchfab"):
        meta = {e["u"]: e for e in json.loads(META.read_text())["entries"]}
        missing = [u for u in uids if u not in meta]
        if missing:
            raise SystemExit(f"{len(missing)} uids without Sketchfab metadata, e.g. {missing[:3]}")

    def size_of(u):
        a = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
        return f"{float(a['width']):.0f} x {float(a['length']):.0f} x {float(a['height']):.0f} cm"
    from metafind.models.resolve_stage1 import serialize_annotation
    if args.source == "sketchfab_name":
        sents = [meta[u]["name"] for u in uids]
    elif args.source == "sketchfab_name_size":
        sents = [f"{meta[u]['name']} {{size: {size_of(u)}}}" for u in uids]
    else:   # Figure-1 forms from the annotation fields: "Platform Bed {size: ...}" / category only
        tpl = {"cat_size": "{category} {{size: {width} x {length} x {height} cm}}", "cat_only": "{category}"}[args.source]
        sents = [serialize_annotation(json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()), template=tpl) for u in uids]
    print(f"{len(uids):,} assets; e.g. {sents[0]!r} / {sents[1]!r}", flush=True)
    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    vecs = []
    with torch.no_grad():
        for i in range(0, len(sents), 256):
            vecs.append(bb.encode_text(sents[i:i + 256]).half().cpu().numpy())
    vecs = np.concatenate(vecs)
    out = paths.OUTPUTS / "_probe" / "text_override" / f"{args.source}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, uids=np.array(uids), vecs=vecs, source=args.source, meta_sha=str(META))
    print(f"-> {out} {vecs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
