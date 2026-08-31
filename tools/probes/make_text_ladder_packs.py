#!/usr/bin/env python3
"""Query packs that vary ONE thing: how much the query text says.

WHY
---
`output/look/ulip2_calibration.json` measured, on the raw ULIP-2 encoder with a
point-cloud-only gallery of 45,692:

    query text = category name (64 templates)    1.1 %
    query text = bare description               28.3 %
    query text = full serialization             22.3 %

MetaFind's ULIP baseline is 0.1 % [PAPER FACT 3experiments.tex:36], which sits
next to the first rung and nowhere near the third. Our own trained model scores
37.8 % text at the same gallery against the paper's 13.8 %. The obvious
candidate is that the paper's query text is far poorer than ours -- but that
was measured WITHOUT the towers, and a tower could absorb the difference. This
builds the packs that put the same ladder through the trained model.

Only `text` is swapped. Image and pc stay canonical, so the one moving part is
the query text; `run_retrieval` applies a pack to the query pass alone and
leaves the promoted gallery index untouched (`run_retrieval.py:709-710`).

WHAT THE RUNGS ARE
------------------
L1  the class name through upstream's own 64-template ensemble
    (`main.py:377-380`: L2 each prompt, mean, L2 again). This is roughly the
    information a ULIP-2-era caption carried.
L2  the annotation's `description` field alone -- prose, no dimensions, no
    material list.
    (L3, the full serialization, is what the canonical cache already holds, so
    it needs no pack: run `run_retrieval` without one.)

NOT A CANONICAL ARTIFACT. Writes only under `_probe/`, changes no protocol, no
checkpoint, no gallery index.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metafind import paths                                       # noqa: E402

UPSTREAM = Path("/home/kyzen/upstream/ULIP")
DEST = paths.OUTPUTS / "_probe" / "text_ladder"


def main() -> int:
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp["test"])                       # protocol A and B query set
    meta = json.loads(paths.LVIS_METADATA.read_text())
    templates = json.loads((UPSTREAM / "data" / "templates.json").read_text())["modelnet40_64"]
    ann = paths.OUTPUTS / "annotations"
    print(f"{len(uids):,} test uids · {len(meta['all_keys']):,} classes "
          f"x {len(templates)} templates", flush=True)

    bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    assert bb.is_frozen()
    import torch

    def enc(strings, tag):
        out, t0 = [], time.time()
        with torch.no_grad():
            for i in range(0, len(strings), 256):
                out.append(bb.encode_text(strings[i:i + 256]).float().cpu().numpy())
                if i and i % 10240 == 0:
                    print(f"  {tag} {i:,}/{len(strings):,} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        return np.concatenate(out)

    # ---- L1: class name, upstream's 64-template ensemble ------------------
    flat = [t.format(l) for l in meta["all_keys"] for t in templates]
    e = enc(flat, "class prompts").reshape(len(meta["all_keys"]), len(templates), -1)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    e = e.mean(axis=1)
    cls_feat = e / np.linalg.norm(e, axis=-1, keepdims=True)
    l1 = np.stack([cls_feat[meta["key_to_id"][meta["value_to_key_mapping"][u]]]
                   for u in uids]).astype(np.float32)

    # ---- L2: the bare description -----------------------------------------
    desc = [json.loads((ann / f"{u}.json").read_text())["description"] for u in uids]
    l2 = enc(desc, "descriptions").astype(np.float32)

    DEST.mkdir(parents=True, exist_ok=True)
    for name, arr, what in (
        ("L1_category_name", l1,
         "class name through upstream's 64-template ensemble (main.py:377-380)"),
        ("L2_bare_description", l2,
         "the annotation's `description` field alone -- no dimensions, no materials"),
    ):
        npy = DEST / f"query_text_{name}.npy"
        np.save(npy, arr)
        (DEST / f"pack_{name}.json").write_text(json.dumps({
            "what": f"TEXT-ONLY query pack, rung {name}",
            "status": "DIAGNOSTIC ONLY -- not a canonical artifact",
            "gallery": "UNCHANGED: the promoted index",
            "text": {"shards": [{"array": str(npy), "uid_order": uids}],
                     "rule": what},
            "image": {"shards": []},
            "pc": {"shards": [], "omitted_because":
                   "this pack isolates the TEXT variable; image and pc stay canonical"},
            "written_at": time.time(),
        }, indent=1))
        print(f"wrote {DEST/f'pack_{name}.json'}  {arr.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
