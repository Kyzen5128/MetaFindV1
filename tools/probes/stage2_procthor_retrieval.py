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
the target's DECLARED modalities (stage2_protocol asset_modalities) from the gallery index plus the house minus the target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths, runlog
from metafind.eval.retrieval import normalize_for_scoring, recall_at_k
from metafind.train.stage1 import (load_protocols as load_stage1_protocols,
                                   load_stage1_checkpoint, load_stage1_model_config,
                                   effective_stage1_model_inputs, stage1_backbone_kwargs)
from metafind.train.stage2 import (Stage2Data, build_stage2_model, encode_query,
                                   enumerate_samples, load_asset_modality_vectors,
                                   verify_stage2_input_identity)
from metafind.train.gallery_index import (load_checkpoint_record, load_stage2_checkpoint_record,
                                          verified_stage2_index, verify_gallery_encoder)
from metafind.eval.run_retrieval import overlay_stage2_weights
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt-record", required=True)
    ap.add_argument("--stage2-record", required=True, help="variant_ckpts.json")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--allow-legacy-stage2-inputs", action="store_true",
                    help="explicitly replay old house graphs without bound input provenance")
    ap.add_argument("--allow-legacy-gallery-index", action="store_true")
    ap.add_argument("--houses", type=int, default=300, help="test houses to query from")
    ap.add_argument("--query-mode", default="none", choices=("none", "text_only"),
                    help="how the query is built, matching the Stage 2 run's "
                         "query_modality_masking: none = the target's declared modalities; "
                         "text_only = text alone, image and pc as mask tokens")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/exp_stage2_procthor_retrieval.json")
    args = ap.parse_args()

    ckpt = load_checkpoint_record(args.stage1_ckpt_record)
    ckpt = load_stage1_model_config(ckpt["uri"], ckpt)
    s2 = load_stage2_checkpoint_record(args.stage2_record, ckpt, args.variant,
                                       allow_legacy=args.allow_legacy_stage2_inputs)
    encoding, training, hyper = effective_stage1_model_inputs(ckpt, *load_stage1_protocols())
    if "input_identity" in s2:
        identity = verify_stage2_input_identity(s2, current_inputs=True)
        _s2p = identity["protocols"]["stage2"]
        arch = identity["protocols"]["arch"]
        graph_unit = identity["graph_unit"]
    elif args.allow_legacy_stage2_inputs:
        identity = None
        _s2p = s2.get("stage2_protocol") or json.loads((paths.OUTPUTS / "stage2_protocol.json").read_text())
        arch = s2.get("arch_protocol") or json.loads((paths.OUTPUTS / "essgnn_arch_protocol.json").read_text())
        graph_unit = s2.get("graph_unit", "house")
    else:
        raise SystemExit("Stage 2 record has no input identity; explicit legacy replay is required")

    data = Stage2Data(args.device, graph_unit=graph_unit)
    index = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    declared = tuple(_s2p["asset_modalities"])
    if training["tower_sharing"] == "fully_separate" and "pc" in declared:
        raise SystemExit("legacy pc-bearing gallery caches do not contain the separate query point path")
    ids, embeddings, arr = verified_stage2_index(index, ckpt["sha256"], declared)
    if s2.get("gallery_index_sha256") and s2["gallery_index_sha256"] != index["sha256"]:
        raise SystemExit("Stage 2 checkpoint was trained against a different gallery index")
    gallery = torch.from_numpy(embeddings).to(args.device)
    data.asset_vectors = load_asset_modality_vectors(
        arr, declared)
    row = {a: i for i, a in enumerate(ids)}

    sp = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    houses = sorted(sp["test_houses"])[: args.houses]
    graphs = data.graphs_for(houses)
    samples = enumerate_samples(houses, set(ids), graph_unit=graph_unit)
    if not samples:
        raise SystemExit("no eligible held-out queries")
    print(f"{len(houses)} test houses, {len(samples):,} leave-one-out queries, "
          f"gallery {len(ids):,}", flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device, train_scope=training["train_scope"],
                                         **stage1_backbone_kwargs(ckpt)))
    query_backbone = (backbone.clone_point_path()
                      if training["tower_sharing"] == "fully_separate" else None)
    model = build_stage2_model(encoding, training, hyper, arch,
                               node_feat_dim=data.node_dim, edge_feat_dim=data.edge_dim,
                               use_layout=True,
                               init_lambda=float(s2["lambda_init"]["init_lambda"])).to(args.device)
    from metafind.train.stage1 import build_model
    _, loss_fn = build_model(encoding, training, hyper)
    if training["freeze_gallery"]:
        model.freeze_gallery(True)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "query.layout_weight"),
                           query_backbone=query_backbone)
    verify_gallery_encoder(index, backbone, model, parent_checkpoint=ckpt,
                           allow_legacy=args.allow_legacy_gallery_index, declared_modalities=declared)

    targets = np.array([row[a] for _, _, a in samples])
    g = normalize_for_scoring(gallery.cpu().numpy())

    present = (torch.tensor([[True, False, False]]).to(args.device)
               if args.query_mode == "text_only" else None)

    def run(label: str, drop_layout: bool):
        model.eval()
        qs = []
        with torch.no_grad():
            for house_id, tidx, asset_id in samples:
                q = encode_query(model, graphs[house_id], tidx, asset_id,
                                 drop_layout, args.device, data, present=present)
                qs.append(q.float().cpu())
        qv = normalize_for_scoring(torch.stack(qs).numpy())
        r = recall_at_k(qv @ g.T, targets)
        print(f"  {label:<8} R@1 {r['R@1']*100:5.1f}  R@5 {r['R@5']*100:5.1f}", flush=True)
        return r

    out = {"n_query": len(samples), "n_gallery": len(ids), "query_mode": args.query_mode,
           "checkpoint_sha256": ckpt["sha256"], "stage2_sha256": s2["sha256"],
           "gallery_index_sha256": index["sha256"], "graph_unit": graph_unit,
           "runtime_source_sha256": runlog.runtime_source_sha256(),
           "runtime_source_status": runlog.runtime_source_status(),
           "allow_legacy_stage2_inputs": args.allow_legacy_stage2_inputs,
           "allow_legacy_gallery_index": args.allow_legacy_gallery_index, "heads": {}}
    print(f"query mode: {args.query_mode}")
    print("\nProcTHOR leave-one-out retrieval (held-out houses):")
    out["heads"]["S1_no_layout"] = run("S1", drop_layout=True)
    overlay_stage2_weights(model, s2, args.device)
    out["heads"]["S2_no_layout"] = run("S2-off", drop_layout=True)
    out["heads"]["S2_with_layout"] = run("S2-on", drop_layout=False)
    lam = float(model.query.layout_weight.item())
    out["lambda"] = lam
    # how big is the layout term relative to the fused query, after training?
    with torch.no_grad():
        h, t, a = samples[0]
        q_off = encode_query(model, graphs[h], t, a, True, args.device, data, present=present).float()
        q_on = encode_query(model, graphs[h], t, a, False, args.device, data, present=present).float()
    out["norm_fused"] = float(q_off.norm()); out["norm_layout_term"] = float((q_on - q_off).norm())
    print(f"  lambda {lam:.3f}; |Fusion| {out['norm_fused']:.1f}  |lambda*e_layout| {out['norm_layout_term']:.1f} (one sample)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
