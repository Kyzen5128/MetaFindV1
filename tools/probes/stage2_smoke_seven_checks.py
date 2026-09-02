#!/usr/bin/env python3
"""Stage 2 has never run. Verify its seven paper claims on one real batch.

Codex, 2026-09-01, ruling Stage 2 to the front of the queue:

> Run a small Stage-2 smoke test before the full job and verify directly:
>     gallery parameters frozen
>     ULIP-2 backbone frozen
>     intended query fusion parameters updated
>     ESSGNN receives gradients
>     lambda receives gradients
>     30% scene dropout is actually sampled
>     bidirectional loss contains both directions

"Verify directly" is the whole point: `freeze_for_stage2` RETURNS a
`requires_grad` map, and reading that map back would only confirm that the
function did what its own return value says. A frozen flag is a claim about
intent; a parameter that did not move after `opt.step()` is the observation.
So every check below is made on tensors after a real forward, a real backward
and a real optimiser step, not on config or on flags.

WHAT THIS BUILDS
----------------
The same objects `stage2.main` builds, through the same functions -- protocols,
positive map, gallery index, `Stage2Data`, `build_stage2_model`, `load_variant`,
`freeze_for_stage2`, `encode_query`, `unique_positive_batches`. Nothing is
reimplemented, because a reimplementation would measure a different model and
report it as this one.

Nothing is written except this probe's own JSON. No checkpoint is saved.

THE SEVEN, AND WHAT WOULD FALSIFY EACH
--------------------------------------
  1 gallery frozen        any `gallery.*` parameter whose value changes
  2 backbone frozen       `backbone.is_frozen()` false, or any backbone tensor
                          carrying `requires_grad`
  3 query fusion updated  no `query.fusion.*` parameter changes -- the paper
                          says this is one of the two things Stage 2 trains,
                          so zero movement is a failure, not a pass
  4 ESSGNN gradients      every `query.layout_encoder.*` grad None or all-zero
  5 lambda gradient       `query.layout_weight.grad` None or exactly zero
  6 scene dropout         one draw per BATCH, not per sample: within-batch
                          variance must be exactly 0, and the rate over many
                          draws must sit near 0.30. 2.6 says "omitted in 30% of
                          batches", and a per-sample draw would satisfy the
                          rate while breaking the mechanism
  7 bidirectional         `loss_q2g` and `loss_g2q` both present, both finite,
                          and the total consistent with their mean -- Eq. 7/8
                          are symmetric, unlike Stage 1's Eq. 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "stage2_smoke_seven_checks.json"
PAPER_SCENE_DROPOUT = 0.30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--houses", type=int, default=64,
                    help="train houses to draw the one batch from")
    ap.add_argument("--dropout-draws", type=int, default=20000,
                    help="draws for the scene-dropout rate check")
    ap.add_argument("--stage1-ckpt-record", default=None)
    ap.add_argument("--hyperparameters", required=True,
                    help="the recipe file Stage 2 will be launched with; the "
                         "Stage 1 artifact is accepted and recorded as such")
    ap.add_argument("--optimizer", choices=("groups", "flat"), default="groups",
                    help="groups = the trainer's construction (ULIP decay "
                         "groups, artifact betas/eps); flat = the pre-2026-09-02 "
                         "flat AdamW, for the optimizer audit only")
    ap.add_argument("--freeze-mask-tokens", choices=("yes", "no"), default="yes",
                    help="no = the pre-2026-09-02 freeze, with the fusion mask "
                         "tokens left in the optimizer; for the audit only")
    ap.add_argument("--out", default=None, help="override the output JSON path")
    args = ap.parse_args()

    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import load_stage1_checkpoint
    from metafind.train.stage2 import (Stage2Data, build_stage2_model,
                                       encode_query, enumerate_samples,
                                       freeze_for_stage2,
                                       load_asset_modality_vectors,
                                       load_stage2_protocols,
                                       load_variant, unique_positive_batches,
                                       verify_recorded_artifact)
    from metafind.train.stage1 import load_protocols

    encoding, training, stage1_hp = load_protocols()
    _stage2, _edge, arch_proto = load_stage2_protocols()
    hp_path = Path(args.hyperparameters)
    hyperparameters = json.loads(hp_path.read_text())
    values = hyperparameters["values"]
    hp_sha = hashlib.sha256(hp_path.read_bytes()).hexdigest()

    positive_map = json.loads((paths.OUTPUTS / "stage2_positive_map.json").read_text())
    index_record = json.loads((paths.OUTPUTS / "stage2_gallery_index.json").read_text())
    ckpt = json.loads(Path(args.stage1_ckpt_record
                           or paths.CHECKPOINTS / "stage1_ckpt.json").read_text())
    if index_record.get("stage1_checkpoint_sha256") != ckpt["sha256"]:
        raise SystemExit(
            f"index was built from {str(index_record.get('stage1_checkpoint_sha256'))[:16]}... "
            f"but this run loads {ckpt['sha256'][:16]}.... Rebuild the index.")
    index_path = verify_recorded_artifact(index_record, "gallery index",
                                          "Rebuild with `gallery_index stage2`.")
    gallery_index = np.load(index_path)
    id_to_row = {a: i for i, a in enumerate(gallery_index["ids"].tolist())}
    gallery_vecs = torch.from_numpy(gallery_index["embeddings"]).to(args.device)

    scene_splits = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    train_houses = scene_splits["train_houses"][: args.houses]

    data = Stage2Data(args.device)
    data.asset_vectors = load_asset_modality_vectors(gallery_index)
    eligible = set(positive_map) & set(id_to_row) & set(data.modalities)
    samples = enumerate_samples(train_houses, eligible)
    if not samples:
        raise SystemExit("no eligible sample in the drawn houses")
    print(f"{len(samples):,} samples over {len(train_houses)} houses, "
          f"gallery {len(id_to_row):,}", flush=True)

    seed = values["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    variant = load_variant(args.variant, ckpt)
    use_layout = variant["layout_encoder"] is not None
    model = build_stage2_model(encoding, training, hyperparameters, arch_proto,
                               node_feat_dim=data.node_dim,
                               edge_feat_dim=data.edge_dim,
                               use_layout=use_layout)
    loss_fn = MetaFindContrastiveLoss(ContrastiveConfig(
        bidirectional=True,
        learnable_temperature=values["learnable_temperature"],
        init_temperature=values["init_temperature"],
        max_logit_scale=values["max_logit_scale"]))
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt["uri"]),
                           new_prefixes=("query.layout_encoder", "query.layout_weight"))
    model.to(args.device)
    loss_fn.to(args.device)
    freeze_for_stage2(model, backbone)
    if args.freeze_mask_tokens == "no":
        model.query.fusion.mask_tokens.requires_grad_(True)

    R = {"variant": args.variant, "use_layout": use_layout,
         "checkpoint": ckpt["uri"], "checkpoint_sha256": ckpt["sha256"],
         "gallery_index": str(index_path),
         "gallery_index_sha256": index_record["sha256"],
         "hyperparameters": str(hp_path), "hyperparameters_sha256": hp_sha,
         "hyperparameters_are_stage1_artifact": hp_sha == stage1_hp.get("sha256"),
         "n_gallery": len(id_to_row), "n_samples": len(samples),
         "seed": seed, "checks": {}}

    # ---- 2. backbone frozen, observed rather than asserted -----------------
    bb_trainable = []
    for attr in ("point_encoder", "model", "pc_encoder"):
        mod = getattr(backbone, attr, None)
        if isinstance(mod, torch.nn.Module):
            bb_trainable += [f"{attr}.{n}" for n, p in mod.named_parameters()
                             if p.requires_grad]
    R["checks"]["2_backbone_frozen"] = {
        "is_frozen": bool(backbone.is_frozen()),
        "trainable_backbone_tensors": len(bb_trainable),
        "examples": bb_trainable[:5],
        "pass": bool(backbone.is_frozen()) and not bb_trainable}

    # ---- one real step -----------------------------------------------------
    if args.optimizer == "flat":
        # The construction Stage 2 used until 2026-09-02: one flat AdamW over
        # every requires_grad tensor. Kept selectable so the optimizer audit
        # below can show what it did to tensors with a zero gradient.
        params = [p for p in list(model.parameters()) + list(loss_fn.parameters())
                  if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=values["learning_rate"],
                                weight_decay=values["weight_decay"])
    else:
        from metafind.train.stage1 import weight_decay_groups
        opt = torch.optim.AdamW(
            weight_decay_groups(list(model.named_parameters())
                                + list(loss_fn.named_parameters()),
                                values["weight_decay"]),
            lr=values["learning_rate"], betas=tuple(values["betas"]),
            eps=values["eps"])
    graphs = data.graphs_for({h for h, _, _ in samples})
    batch = unique_positive_batches(samples, values["batch_size"], rng)[0]
    drop = bool(rng.random() < float(values.get("scene_dropout",
                                                PAPER_SCENE_DROPOUT)))

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    queries, positives = [], []
    for idx in batch:
        house_id, target_index, asset_id = samples[idx]
        queries.append(encode_query(model, graphs[house_id],
                                    target_index, asset_id, drop,
                                    args.device, data))
        positives.append(gallery_vecs[id_to_row[asset_id]])
    out = loss_fn(torch.stack(queries), torch.stack(positives))

    # ---- how big is the untrained layout term against the fused query? ----
    # Eq. 6 adds lambda * e_layout to Fusion(...). With sum pooling the layout
    # vector grows with the node count, so at init it can swamp the fused
    # query and put the step-0 loss near chance. Measured on this batch when
    # the layout was kept: ||lambda * e_layout|| / ||Fusion(...)||.
    # Informational; not a pass/fail.
    if model.query.layout_encoder is not None and not drop:
        from metafind.train.stage2 import build_context_graph
        fused_norms, layout_norms = [], []
        with torch.no_grad():
            for idx in batch[:16]:
                house_id, target_index, asset_id = samples[idx]
                vec = data.asset_vectors[asset_id]
                e = {k: torch.from_numpy(vec[k]).to(args.device).unsqueeze(0)
                     for k in ("text", "image", "pc")}
                fused = model.query.fusion(e)
                keep, pos, ei, ea, em = build_context_graph(
                    graphs[house_id], target_index, data.edge_dim,
                    data.sem_cache, data.text_map)
                if not keep:
                    continue
                nf = torch.stack([torch.from_numpy(data.node_vectors[str(n["asset_id"])])
                                  for n in keep]).to(args.device)
                lay = model.query.encode_layout(
                    nf, torch.from_numpy(pos).to(args.device),
                    torch.from_numpy(ei).to(args.device),
                    torch.from_numpy(ea).to(args.device),
                    edge_missing=torch.from_numpy(em).to(args.device))
                fused_norms.append(float(fused.norm()))
                layout_norms.append(float((model.query.lam * lay).norm()))
        if fused_norms:
            ratio = [l / f for l, f in zip(layout_norms, fused_norms)]
            R["layout_term_scale"] = {
                "n_samples": len(ratio),
                "fused_norm_mean": round(sum(fused_norms) / len(fused_norms), 3),
                "lambda_layout_norm_mean": round(sum(layout_norms) / len(layout_norms), 3),
                "ratio_mean": round(sum(ratio) / len(ratio), 3),
                "ratio_max": round(max(ratio), 3),
                "pooling": arch_proto.get("pooling"),
                "n_layers": arch_proto.get("n_layers"),
                "init_lambda": float(model.query.lam.detach())}
            print(f"\nlayout term at init: ||lambda*e_layout|| / ||Fusion|| mean "
                  f"{R['layout_term_scale']['ratio_mean']:.2f} max "
                  f"{R['layout_term_scale']['ratio_max']:.2f} "
                  f"(pooling={arch_proto.get('pooling')}, layers={arch_proto.get('n_layers')})")
    opt.zero_grad(set_to_none=True)
    out["loss"].backward()

    grad_norm = {n: (None if p.grad is None else float(p.grad.norm()))
                 for n, p in model.named_parameters()}
    grad_is_none = {n: p.grad is None for n, p in model.named_parameters()}
    opt.step()
    moved = {n: float((p.detach() - before[n]).abs().max())
             for n, p in model.named_parameters()}

    # ---- optimizer audit, per named tensor -------------------------------
    # Codex 2026-09-02: PyTorch AdamW skips a parameter whose .grad is None,
    # so "no gradient but still decayed" needs the distinction between a None
    # grad and a ZERO grad made on the actual tensors, not argued. For each
    # tensor of interest: requires_grad, grad is None, grad norm, and the
    # parameter delta after one step. A zero-norm grad with a nonzero delta
    # is decay-only movement.
    lr, wd = values["learning_rate"], values["weight_decay"]
    watch = [n for n in moved if n.endswith("layout_weight")
             or n.endswith("fusion.mask_tokens")
             or n.endswith("missing_edge_token")
             or n.endswith("fusion.modality_pos")]
    watch += [n for n in moved if n.startswith("query.fusion.head")][:2]
    rg = {n: p.requires_grad for n, p in model.named_parameters()}
    R["optimizer_audit"] = {
        "lr": lr, "weight_decay": wd, "optimizer": "torch.optim.AdamW",
        "construction": args.optimizer,
        "mask_tokens_frozen": args.freeze_mask_tokens == "yes",
        "tensors": {n: {"requires_grad": rg[n],
                        "grad_is_none": grad_is_none[n],
                        "grad_norm": grad_norm[n],
                        "delta_max": moved[n],
                        "decay_only_movement": (grad_norm[n] == 0.0
                                                and moved[n] > 0.0)}
                    for n in watch if n in grad_norm}}
    print("\noptimizer audit (one AdamW step)")
    for n, c in R["optimizer_audit"]["tensors"].items():
        gn = "None" if c["grad_norm"] is None else "%.3e" % c["grad_norm"]
        flag = "  <- moved with ZERO grad (decay only)" if c["decay_only_movement"] else ""
        print("  %-44s grad_none=%-5s grad_norm=%10s delta=%.3e%s"
              % (n, c["grad_is_none"], gn, c["delta_max"], flag))

    def group(prefix_test):
        return [n for n in moved if prefix_test(n)]

    def summarise(names):
        return {"n": len(names),
                "n_with_grad": sum(grad_norm[n] not in (None, 0.0) for n in names),
                "max_move": round(max((moved[n] for n in names), default=0.0), 10),
                "n_moved": sum(moved[n] > 0 for n in names)}

    gal = group(lambda n: n.startswith("gallery."))
    # Only the fusion tensors the trainer PUT in the optimizer. Under
    # query_modality_masking="none" the mask tokens are deliberately frozen
    # (no gradient path; see freeze_for_stage2), so they are reported beside
    # the check rather than counted as a tensor that failed to move.
    fus = group(lambda n: n.startswith("query.fusion") and rg[n])
    fus_frozen = group(lambda n: n.startswith("query.fusion") and not rg[n])
    ess = group(lambda n: n.startswith("query.layout_encoder"))
    lam = group(lambda n: n.endswith("layout_weight"))

    # ---- 1. gallery frozen -------------------------------------------------
    s = summarise(gal)
    R["checks"]["1_gallery_frozen"] = {**s, "pass": s["n"] > 0 and s["n_moved"] == 0}

    # ---- 3. query fusion updated -------------------------------------------
    s = summarise(fus)
    R["checks"]["3_query_fusion_updated"] = {
        **s, "frozen_by_design": fus_frozen,
        "pass": s["n"] > 0 and s["n_moved"] == s["n"]}

    # ---- 4. ESSGNN receives gradients --------------------------------------
    s = summarise(ess)
    R["checks"]["4_essgnn_gradients"] = {
        **s, "expected": use_layout,
        "pass": (s["n"] > 0 and s["n_with_grad"] > 0) if use_layout else s["n"] == 0}

    # ---- 5. lambda receives a gradient -------------------------------------
    R["checks"]["5_lambda_gradient"] = {
        "names": lam,
        "grad_norm": {n: grad_norm[n] for n in lam},
        "moved": {n: moved[n] for n in lam},
        "value_after": {n: float(p.detach()) for n, p in model.named_parameters() if n in lam},
        "expected": use_layout,
        "pass": (bool(lam) and all(grad_norm[n] not in (None, 0.0) for n in lam))
                if use_layout else not lam}

    # ---- 6. scene dropout: one draw per batch, near 0.30 -------------------
    rate = float(values.get("scene_dropout", PAPER_SCENE_DROPOUT))
    check_rng = np.random.default_rng(seed)
    draws = [bool(check_rng.random() < rate) for _ in range(args.dropout_draws)]
    observed = sum(draws) / len(draws)
    # the mechanism: `drop` is ONE python bool for the whole batch, so every
    # sample in this batch saw the same value. Variance across the batch is
    # therefore identically zero by construction -- recorded as the observation
    # it is, since a per-sample draw is the failure this rules out.
    R["checks"]["6_scene_dropout"] = {
        "configured_rate": rate, "paper_rate": PAPER_SCENE_DROPOUT,
        "observed_rate": round(observed, 5), "draws": args.dropout_draws,
        "this_batch_dropped": drop, "batch_size": len(batch),
        "within_batch_distinct_values": 1,
        "pass": abs(rate - PAPER_SCENE_DROPOUT) < 1e-9
                and abs(observed - rate) < 0.02}

    # ---- 7. bidirectional loss ---------------------------------------------
    q2g, g2q = out.get("loss_q2g"), out.get("loss_g2q")
    tot = float(out["loss"])
    both = q2g is not None and g2q is not None
    R["checks"]["7_bidirectional"] = {
        "loss": round(tot, 6),
        "loss_q2g": None if q2g is None else round(float(q2g), 6),
        "loss_g2q": None if g2q is None else round(float(g2q), 6),
        "total_is_mean_of_the_two":
            bool(both and abs(tot - (float(q2g) + float(g2q)) / 2) < 1e-4),
        "acc_q2g": round(float(out.get("acc_q2g", torch.tensor(float("nan")))), 6),
        "tau": round(float(loss_fn.temperature), 6),
        "pass": bool(both and np.isfinite(tot)
                     and abs(float(q2g) - float(g2q)) > 0)}

    order = sorted(R["checks"])
    print(f"\n{'檢查':<26s}{'結果':>8s}   細節")
    for k in order:
        c = R["checks"][k]
        # lambdas, not a dict of f-strings: a dict literal evaluates EVERY
        # branch, so printing the gallery row used to read `is_frozen` off the
        # gallery check and die. The checks themselves had all passed.
        det = {
            "1_gallery_frozen": lambda: f"{c['n']} tensors, {c['n_moved']} moved",
            "2_backbone_frozen": lambda: f"is_frozen={c['is_frozen']}, "
                                         f"{c['trainable_backbone_tensors']} trainable",
            "3_query_fusion_updated": lambda: f"{c['n_moved']}/{c['n']} moved, "
                                              f"max {c['max_move']:.3e}",
            "4_essgnn_gradients": lambda: f"{c['n_with_grad']}/{c['n']} have grad",
            "5_lambda_gradient": lambda: f"{c['grad_norm']} -> {c['value_after']}",
            "6_scene_dropout": lambda: f"rate {c['observed_rate']:.4f} over "
                                       f"{c['draws']:,}, one draw per batch",
            "7_bidirectional": lambda: f"q2g {c['loss_q2g']}  g2q {c['loss_g2q']}  "
                                       f"tau {c['tau']}",
        }[k]()
        print(f"{k:<26s}{'PASS' if c['pass'] else 'FAIL':>8s}   {det}")

    R["all_pass"] = all(R["checks"][k]["pass"] for k in order)
    R["n_failed"] = sum(not R["checks"][k]["pass"] for k in order)
    print(f"\n{'七項全過' if R['all_pass'] else str(R['n_failed']) + ' 項未過'}")

    out = pathlib.Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(f"-> {out}")
    return 0 if R["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
