#!/usr/bin/env python3
"""Where does the Stage 2 loss start, and what do the first optimizer steps do to it?

Pilot 2 (builder fixed) still logged loss ~2.65 at step 50 and flat to step
1,500, while the same parent head retrieves its ProcTHOR target at 82.4% R@1
(tools/probes/stage2_procthor_retrieval). Those two numbers cannot both
describe step 0. This replays the trainer's own construction -- same
Stage2Data, encode_query, gallery index, batching, loss -- and prints the
loss BEFORE any update, then after 1, 2, 5, 10, 20, 50 AdamW steps at the
recipe's rate on a fixed probe batch, with and without the layout term.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.train.stage1 import build_model, load_protocols, load_stage1_checkpoint
from metafind.train.stage2 import (Stage2Data, build_stage2_model, encode_query,
                                   enumerate_samples, freeze_for_stage2,
                                   load_asset_modality_vectors, load_stage2_protocols,
                                   unique_positive_batches, usable_batches)
from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt-record", required=True)
    ap.add_argument("--hyperparameters", required=True)
    ap.add_argument("--lr", type=float, default=None, help="override the recipe's rate")
    ap.add_argument("--houses", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = json.loads(Path(args.stage1_ckpt_record).read_text())
    values = json.loads(Path(args.hyperparameters).read_text())["values"]
    lr = args.lr if args.lr is not None else float(values["learning_rate"])
    encoding, training, hyper = load_protocols()
    _s2p, _e, arch = load_stage2_protocols()
    data = Stage2Data(args.device)
    index = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    arr = np.load(index["uri"]); ids = arr["ids"].tolist()
    gallery_vecs = torch.from_numpy(arr["embeddings"]).to(args.device)
    data.asset_vectors = load_asset_modality_vectors(arr)
    row = {a: i for i, a in enumerate(ids)}

    backbone = ULIPBackbone(BackboneConfig(device=args.device, train_scope="point_encoder_and_fuser"))
    model = build_stage2_model(encoding, training, hyper, arch, node_feat_dim=data.node_dim,
                               edge_feat_dim=data.edge_dim, use_layout=True, init_lambda=93.0).to(args.device)
    _, loss_fn = build_model(encoding, training, hyper)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "query.layout_weight"))
    loss_fn = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True, learnable_temperature=False,
                                                        init_temperature=0.5)).to(args.device)
    freeze_for_stage2(model, backbone, query_modality_masking="none")

    sp = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    houses = sorted(sp["train_houses"])[: args.houses]
    graphs = {h: json.loads((paths.OUTPUTS / "scene_graphs" / f"{h}.json").read_text()) for h in houses}
    samples = enumerate_samples(houses, set(ids))
    rng = np.random.default_rng(int(values["seed"]))
    batches, _, _ = usable_batches(unique_positive_batches(samples, int(values["batch_size"]), rng))
    probe, train_batches = batches[0], batches[1:]

    def batch_loss(batch, drop):
        model.query.train() if not drop else model.query.train()
        qs, gs = [], []
        for i in batch:
            h, t, a = samples[i]
            qs.append(encode_query(model, graphs[h], t, a, drop, args.device, data))
            gs.append(gallery_vecs[row[a]])
        q, g = torch.stack(qs), torch.stack(gs)
        if q.dim() == 3:
            q = q.squeeze(1)
        out = loss_fn(q, g)
        return out

    with torch.no_grad():
        for drop in (True, False):
            o = batch_loss(probe, drop)
            print(f"step 0  layout {'dropped' if drop else 'used'}: loss {o['loss'].item():.4f}  "
                  f"acc_q2g {o['acc_q2g'].item():.3f}  q shape {tuple(torch.stack([encode_query(model, graphs[samples[probe[0]][0]], samples[probe[0]][1], samples[probe[0]][2], drop, args.device, data)]).shape)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=float(values["weight_decay"]))
    print(f"\nAdamW lr {lr:g} flat (no warmup), {len(params)} tensors; loss on the fixed probe batch (layout used):")
    marks = {1, 2, 5, 10, 20, 50}
    for step, b in enumerate(train_batches[:50], start=1):
        drop = bool(rng.random() < 0.3)
        out = batch_loss(b, drop)
        opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
        if step in marks:
            with torch.no_grad():
                p = batch_loss(probe, False); p2 = batch_loss(probe, True)
            print(f"  after {step:2d} steps: probe loss layout-used {p['loss'].item():.4f} (acc {p['acc_q2g'].item():.3f})"
                  f"   layout-dropped {p2['loss'].item():.4f}   lambda {model.query.layout_weight.item():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
