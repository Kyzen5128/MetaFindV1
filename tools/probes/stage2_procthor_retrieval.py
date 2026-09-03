#!/usr/bin/env python3
"""Did Stage 2 learn layout-aware retrieval, and did its layout-free head survive?

Three heads scored on the SAME leave-one-out queries from held-out ProcTHOR
test houses, against the 1,439-asset ProcTHOR gallery the Stage 2 run trained
on (the Stage 1 parent's gallery tower, frozen in Stage 2):

  S1        Stage 1 parent's query fusion, no layout       (what Stage 2 started from)
  S2-off    Stage 2 query fusion, no layout                (the Table 1 w/ ESSGNN head)
  S2-on     Stage 2 query fusion + lambda * ESSGNN(context) (Eq. 6, what Stage 2 trained)

If S2-on >> S2-off the layout branch carries the retrieval; if S2-off << S1
the fine-tuning wrecked the layout-free head (the paper's 'feature-attribution
mismatch', measured on the training distribution itself rather than on
Objaverse). Query construction is the trainer's own: the target's full
T/I/P from the gallery index plus the house minus the target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.train.stage1 import load_protocols as load_stage1_protocols
from metafind.train.stage1 import load_stage1_checkpoint
from metafind.train.stage2 import (Stage2Data, build_stage2_model, encode_query,
                                   enumerate_samples, load_asset_modality_vectors,
                                   load_stage2_protocols)
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt-record", required=True)
    ap.add_argument("--stage2-record", required=True, help="variant_ckpts.json")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--houses", type=int, default=300, help="test houses to query from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_stage2_procthor_retrieval.json")
    args = ap.parse_args()

    ckpt = json.loads(Path(args.stage1_ckpt_record).read_text())
    s2 = json.loads(Path(args.stage2_record).read_text())[args.variant]
    if s2["stage1_checkpoint_sha256"] != ckpt["sha256"]:
        raise SystemExit("Stage 2 record's parent is not this Stage 1 checkpoint")
    encoding, training, hyper = load_stage1_protocols()
    _s2p, _edge, arch = load_stage2_protocols()

    data = Stage2Data(args.device)
    index = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    arr = np.load(index["uri"])
    ids = arr["ids"].tolist()
    gallery = torch.from_numpy(arr["embeddings"]).to(args.device)
    data.asset_vectors = load_asset_modality_vectors(arr)
    row = {a: i for i, a in enumerate(ids)}

    sp = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    houses = sorted(sp["test_houses"])[: args.houses]
    graphs = {h: json.loads((paths.OUTPUTS / "scene_graphs" / f"{h}.json").read_text())
              for h in houses}
    samples = enumerate_samples(houses, set(ids))
    print(f"{len(houses)} test houses, {len(samples):,} leave-one-out queries, "
          f"gallery {len(ids):,}", flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device, train_scope="fuser_only"))
    model = build_stage2_model(encoding, training, hyper, arch,
                               node_feat_dim=data.node_dim, edge_feat_dim=data.edge_dim,
                               use_layout=True,
                               init_lambda=float(s2["lambda_init"]["init_lambda"])).to(args.device)
    from metafind.train.stage1 import build_model
    _, loss_fn = build_model(encoding, training, hyper)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "query.layout_weight"))
    s1_query_state = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("query.fusion")}
    s2_state = torch.load(s2["uri"], map_location=args.device, weights_only=False)["trainable_state"]

    targets = np.array([row[a] for _, _, a in samples])
    g = normalize_for_scoring(gallery.cpu().numpy())

    def run(label: str, drop_layout: bool):
        model.eval()
        qs = []
        with torch.no_grad():
            for house_id, tidx, asset_id in samples:
                q = encode_query(model, graphs[house_id], tidx, asset_id,
                                 drop_layout, args.device, data)
                qs.append(q.float().cpu())
        qv = normalize_for_scoring(torch.stack(qs).numpy())
        r = recall_at_k(qv @ g.T, targets)
        print(f"  {label:<8} R@1 {r['R@1']*100:5.1f}  R@5 {r['R@5']*100:5.1f}", flush=True)
        return r

    out = {"n_query": len(samples), "n_gallery": len(ids), "heads": {}}
    print("\nProcTHOR leave-one-out retrieval (held-out houses):")
    out["heads"]["S1_no_layout"] = run("S1", drop_layout=True)
    model.load_state_dict(s2_state, strict=False)
    out["heads"]["S2_no_layout"] = run("S2-off", drop_layout=True)
    out["heads"]["S2_with_layout"] = run("S2-on", drop_layout=False)
    lam = float(model.query.layout_weight.item())
    out["lambda"] = lam
    # how big is the layout term relative to the fused query, after training?
    with torch.no_grad():
        h, t, a = samples[0]
        q_off = encode_query(model, graphs[h], t, a, True, args.device, data).float()
        q_on = encode_query(model, graphs[h], t, a, False, args.device, data).float()
    out["norm_fused"] = float(q_off.norm()); out["norm_layout_term"] = float((q_on - q_off).norm())
    print(f"  lambda {lam:.3f}; |Fusion| {out['norm_fused']:.1f}  |lambda*e_layout| {out['norm_layout_term']:.1f} (one sample)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
