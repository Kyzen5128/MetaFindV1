#!/usr/bin/env python3
"""U-11: do the absent modalities' mask tokens take part in the pooled readout?

WHY THIS ONE
------------
[PAPER 3experiments.tex:143] fixes exactly one thing about the fusion module --
that it is "the final selected Transformer". Everything else in
`metafind/models/fusion.py` is a default we wrote:

    hidden 2048 · n_heads 8 · n_layers 2 · dropout 0.0 · norm_first True
    readout = masked mean over the 3 tokens (not a CLS token)
    learned per-modality positional embedding
    include_absent_slots True          <- this one

`include_absent_slots` decides what a PARTIAL query looks like, and five of
Table 1's seven cells are partial:

    True   text-only  ->  tokens [text, mask_img, mask_pc], attention over all
                          three, output = mean of three. Diluted 2/3 by two
                          learned constants.
    False  text-only  ->  the other two are key-padded and dropped from the
                          mean, output = the text token's own transformer
                          output. No dilution.

`fusion.py:238` argues for True: sec. 2.6 contrasts masked embeddings with
"rather than zero-padding", and a slot dropped from the readout has nothing to
pad. That reading is defensible and it is still a reading.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
-----------------------------------------
The flag is flipped AT INFERENCE on a checkpoint trained under True, so this is
a SENSITIVITY measurement, not the counterfactual. A model trained under False
would place its mask tokens differently. If the sensitivity is small the choice
does not matter and U-11 can be closed cheaply; if it is large, U-11 needs a
training arm and cannot be settled by argument.

The gallery is untouched: it is modality-complete, all three slots present, so
`active` is all-ones either way and the promoted index stays valid.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metafind import paths                                            # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,  # noqa: E402
                                     normalize_for_scoring, rank_of_target)
from metafind.train.gallery_index import load_promoted_index_for_checkpoint  # noqa: E402

REC = Path("/home/kyzen/metafind_data/outputs/checkpoints/"
           "qpack_ti_lr2.50e-04_s20260816/stage1_best_ckpt.json")
OUT = REPO / "output" / "look" / "u11_absent_slots.json"
PAPER = {"text":13.8,"image":11.7,"pc":75.1,"text+image":17.2,
         "text+pc":44.5,"image+pc":45.8,"full":51.7}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import build_model, load_stage1_checkpoint, load_protocols

    record = json.loads(REC.read_text())
    sha, weights = record["sha256"], Path(record["uri"])   # `uri` is the .pt
    _rec, gids, gemb = load_promoted_index_for_checkpoint(sha)
    gal = normalize_for_scoring(gemb)
    gpos = {u: i for i, u in enumerate(gids)}
    qu = sorted(json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]["test"])
    col = np.array([gpos[u] for u in qu])
    log(f"gallery {len(gids):,} (promoted index) · query {len(qu):,} (test)")

    encoding, training, hyper = load_protocols()
    backbone = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    model, loss_fn = build_model(encoding, training, hyper)
    model.to("cuda")
    load_stage1_checkpoint(backbone, model, loss_fn, path=weights)
    model.eval()

    emb = paths.OUTPUTS / "embeddings"
    cached = [np.load(emb / f"{u}.npz") for u in qu]
    text = torch.tensor(np.stack([c["text"] for c in cached]).astype(np.float32))
    image = torch.tensor(np.stack([c["image"] for c in cached]).astype(np.float32))
    root = paths.OUTPUTS / "pointclouds"

    log("encoding query point clouds once (shared by both settings) ...")
    pcs = []
    with torch.no_grad():
        for i in range(0, len(qu), 64):
            cl = np.stack([np.concatenate([z["xyz"], z["rgb"]], 1)
                           for z in (np.load(root / f"{u}.npz") for u in qu[i:i+64])])
            pcs.append(backbone.encode_pc(cl.astype(np.float32)).float().cpu())
            if i and i % 2560 == 0: log(f"  pc {i}/{len(qu)}")
    pc = torch.cat(pcs)

    res = {}
    for flag in (True, False):
        model.query.fusion.cfg.include_absent_slots = flag
        rows = {}
        for cond in QUERY_CONDITIONS:
            outs = []
            with torch.no_grad():
                for i in range(0, len(qu), 256):
                    sl = slice(i, i + 256)
                    n = len(qu[sl])
                    outs.append(model.query(
                        {"text": text[sl].cuda(), "image": image[sl].cuda(),
                         "pc": pc[sl].cuda()},
                        present=condition_mask(cond, n).cuda()).float().cpu().numpy())
            q = normalize_for_scoring(np.concatenate(outs))
            r = np.concatenate([rank_of_target(q[i:i+512] @ gal.T, col[i:i+512])
                                for i in range(0, len(q), 512)])
            rows[cond] = {"R@1": 100.0*float((r <= 1).mean()),
                          "R@5": 100.0*float((r <= 5).mean())}
        res["include_absent_slots=%s" % flag] = rows
        log(f"  include_absent_slots={flag}: text {rows['text']['R@1']:.2f} "
            f"full {rows['full']['R@1']:.2f}")

    a, b = res["include_absent_slots=True"], res["include_absent_slots=False"]
    print(f"\n{'='*66}\nU-11 sensitivity, protocol B geometry, gallery {len(gids):,}")
    print(f"{'condition':>12}{'True (ours)':>13}{'False':>9}{'delta':>9}{'paper':>8}")
    for c in QUERY_CONDITIONS:
        print(f"{c:>12}{a[c]['R@1']:13.2f}{b[c]['R@1']:9.2f}"
              f"{b[c]['R@1']-a[c]['R@1']:+9.2f}{PAPER[c]:8.1f}")
    print("\nthe gallery is modality-complete, so the flag changes the QUERY side only.")
    print("flipped at inference on a checkpoint trained under True: a SENSITIVITY,")
    print("not the counterfactual. A large delta means U-11 needs a training arm.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": "U-11: do absent modalities' mask tokens join the pooled readout?",
        "checkpoint_record": str(REC), "n_gallery": len(gids), "n_query": len(qu),
        "caveat": "flag flipped at inference on a model trained under True; "
                  "this bounds sensitivity, it is not the trained counterfactual",
        "results": res, "paper_R@1": PAPER,
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
