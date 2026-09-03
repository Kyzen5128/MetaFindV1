#!/usr/bin/env python3
"""Answer the five questions the ten-epoch pilot was run to answer.

[KYZEN 2026-09-03] The run is not for a score. He named five questions and the
run exists to answer them before 60+ hours are committed:

    1. Is PointBERT actually learning?
    2. Does the mask embedding grow out of near-zero?
    3. Are text / image / full abnormally high, i.e. shortcut-shaped?
    4. Does PC-only move sensibly from the official ULIP-2 initialisation?
    5. Over ten epochs, does validation improve, saturate fast, or collapse?

This reads the run's own logs and its final checkpoint. It trains nothing and
changes nothing.

THE DENOMINATOR THIS TOOL EXISTS TO FIX
---------------------------------------
The live metric records `query_norm`, which is the FUSED OUTPUT. The mask token
substitutes for a fusion INPUT, so `mask_token_norm / query_norm` compares two
things at different points in the network. In epoch 0 that ratio fell from
2.60% to 0.77% purely because the output norm tripled while the token held
still -- a statement about the fusion head, not about the mask token.

The right denominator is the modality vectors that ENTER the fusion. Text and
image are frozen, so theirs are read straight from n06's cache. The point cloud
is not: PointBERT is training, so its norm moves, and it has to be measured
through the checkpoint being analysed. That is why this is a separate tool and
not a log column -- the answer needs the model, not just the run's numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metafind import paths

CONDITIONS = ("text", "image", "pc", "text+image", "text+pc", "image+pc", "full")


def _rows(name: str, revision: str) -> list[dict]:
    p = paths.LOGS / name
    if not p.exists():
        return []
    out = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in out if str(r.get("code_revision", "")).startswith(revision)]


def modality_norms(ckpt_path: Path, uids: list[str], device: str,
                   n: int = 256) -> dict:
    """The norms of the three vectors that ENTER the fusion, under this ckpt.

    Text and image come from n06's cache because they are frozen and the
    checkpoint cannot have moved them -- reading them from the cache is reading
    exactly what the tower saw. The point cloud is encoded live, because
    PointBERT is in the optimizer and its output norm is the thing that moved.
    """
    import torch

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import (
        build_model,
        load_protocols,
        load_stage1_checkpoint,
    )

    encoding, training, hyperparameters = load_protocols()
    values = hyperparameters["values"]
    backbone = ULIPBackbone(BackboneConfig(
        device=device, train_scope=training.get("train_scope",
                                                "point_encoder_and_fuser")))
    model = build_model(encoding, training, values, backbone.dim).to(device)
    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
    loss_fn = MetaFindContrastiveLoss(ContrastiveConfig(
        bidirectional=False,
        learnable_temperature=values["learnable_temperature"],
        init_temperature=values["init_temperature"],
        max_logit_scale=values["max_logit_scale"])).to(device)
    load_stage1_checkpoint(backbone, model, loss_fn, ckpt_path)

    text, image, pcs = [], [], []
    sample = uids[:n]
    for uid in sample:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        text.append(float(np.linalg.norm(z["text"].astype(np.float32))))
        image.append(float(np.linalg.norm(z["image"].astype(np.float32))))
    from metafind.train.stage1 import modules_in_eval
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, len(sample), 32):
            batch = []
            for uid in sample[i:i + 32]:
                c = np.load(paths.POINTCLOUDS / f"{uid}.npz")
                xyz = c["xyz"].astype(np.float32)
                rgb = c["rgb"].astype(np.float32) if "rgb" in c else None
                batch.append(xyz if rgb is None
                             else np.concatenate([xyz, rgb], axis=1))
            t = torch.from_numpy(np.stack(batch)).to(device)
            pcs += backbone.encode_pc(t).norm(dim=-1).tolist()

    mask = None
    for n_, p in model.named_parameters():
        if n_.endswith("mask_tokens") and n_.startswith("query"):
            mask = p.detach().cpu().norm(dim=-1).tolist()
    return {"n_sampled": len(sample),
            "text_norm": float(np.median(text)),
            "image_norm": float(np.median(image)),
            "pc_norm": float(np.median(pcs)),
            "query_mask_token_norms": mask}


def answer(revision: str, ckpt: Path | None, device: str) -> dict:
    train = _rows("train_stage1.jsonl", revision)
    val = _rows("train_stage1_dev_val.jsonl", revision)
    if not val:
        raise SystemExit(f"no dev-val rows for code_revision {revision}")

    def series(k):
        return [r[k] for r in train if k in r]

    out = {
        "code_revision": revision,
        "epochs": len(val),
        "train_rows": len(train),
        # --- 1 -----------------------------------------------------------
        "q1_pointbert_learning": {
            "grad_norm_first": series("pointbert_grad_norm")[:1],
            "grad_norm_last": series("pointbert_grad_norm")[-1:],
            "grad_norm_median": float(np.median(series("pointbert_grad_norm"))),
            "always_nonzero": all(v > 0 for v in series("pointbert_grad_norm")),
            "fusion_grad_norm_median": float(np.median(series("fusion_grad_norm"))),
            # The control. Frozen means frozen, and this is the only line that
            # would say otherwise.
            "clip_grad_norm_always_zero": all(
                v == 0.0 for v in series("clip_grad_norm")),
        },
        # --- 2 -----------------------------------------------------------
        "q2_mask_embedding": {
            "norm_first": series("mask_token_norm")[:1],
            "norm_last": series("mask_token_norm")[-1:],
            "norm_min_last": series("mask_token_norm_min")[-1:],
            "norm_max_last": series("mask_token_norm_max")[-1:],
            "_note": "the RATIO that matters is under q2_ratio_corrected; "
                     "mask_token_norm / query_norm compares a fusion INPUT "
                     "against a fusion OUTPUT and is not the question",
        },
        # --- 3 and 4 -------------------------------------------------------
        "q3_q4_conditions": {
            c: {"first": val[0][f"cond_{c}_R@1"],
                "last": val[-1][f"cond_{c}_R@1"],
                "delta": round(val[-1][f"cond_{c}_R@1"]
                               - val[0][f"cond_{c}_R@1"], 4)}
            for c in CONDITIONS},
        # --- 5 -------------------------------------------------------------
        "q5_validation_trend": {
            "mean_R@1_per_epoch": [r["mean_R@1"] for r in val],
            "mean_R@5_per_epoch": [r["mean_R@5"] for r in val],
            "n_gallery_每epoch": sorted({r["n_gallery"] for r in val}),
            "best_epoch": max(range(len(val)), key=lambda i: val[i]["mean_R@1"]),
        },
        # --- the protocol's own invariants, checked not assumed -------------
        "invariants": {
            "all_masked_frac_mean": float(np.mean(series("all_masked_frac"))),
            "all_masked_frac_expected": 0.027,
            "query_norm_first_last": [series("query_norm")[0],
                                      series("query_norm")[-1]],
            "gallery_norm_first_last": [series("gallery_norm")[0],
                                        series("gallery_norm")[-1]],
            "tau": sorted({r["tau"] for r in train}),
        },
    }

    if ckpt is not None and ckpt.exists():
        sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
        m = modality_norms(ckpt, sorted(sp["dev_val"]), device)
        mask = m["query_mask_token_norms"] or []
        out["q2_ratio_corrected"] = {
            **{k: v for k, v in m.items() if k != "query_mask_token_norms"},
            "mask_token_norms": mask,
            # Per modality, in the fusion's own input order, because a single
            # mean would hide one stand-in learning while another does not.
            "ratio_vs_text": (mask[0] / m["text_norm"]) if mask else None,
            "ratio_vs_image": (mask[1] / m["image_norm"]) if len(mask) > 1 else None,
            "ratio_vs_pc": (mask[2] / m["pc_norm"]) if len(mask) > 2 else None,
            "_why": "the mask token replaces a fusion INPUT, so the denominator "
                    "is the modality vector it stands in for -- not the fused "
                    "output, which is what the live metric divides by",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True,
                    help="code_revision prefix identifying the run")
    ap.add_argument("--ckpt", default=None, help="the run's best checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/look/pilot10.json")
    args = ap.parse_args()

    rec = answer(args.revision, Path(args.ckpt) if args.ckpt else None,
                 args.device)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
