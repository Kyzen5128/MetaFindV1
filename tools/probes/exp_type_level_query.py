#!/usr/bin/env python3
"""Type-level query text at TEST time on the P1 checkpoint (no training).

[KYZEN 2026-09-04 「好你先測吧」] 5i showed that scoring P1 with another
same-category asset's text+image flips the fused cells below pc (paper's
ordering) but drives text-only / image-only to ~0 (paper 13.8 / 11.7). The
paper's Figure 1 query text is `Platform Bed {size: ...}`: the TARGET's own
category and size, no description. So: query text = the target's own
type-level sentence built from the SAME annotation fields with a different
template (no materials, no placement, no description); image = a same-category
reference view (another asset) or the target's own view; pc = the target's own.

Gallery = P1's cached record (attrs_v1 text, 12-view mean, canonical pc via
P1's PointBERT). Parity rows reproduce the evaluator's numbers exactly.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metafind import paths, runlog
from metafind.data.pointclouds import uid_seed
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,
                                     normalize_for_scoring, recall_at_k)
from metafind.models.resolve_stage1 import serialize_annotation
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

PAPER = {"text": 13.8, "image": 11.7, "pc": 75.1, "text+image": 17.2, "text+pc": 44.5, "image+pc": 45.8, "full": 51.7}
TEMPLATES = {
    "cat_size": "{category} {{size: {width} x {length} x {height} cm}}",   # Figure 1 form
    "cat_only": "{category}",
}


def optional_fields_text(cache_path, uids: list[str]):
    """An optional diagnostic row cannot change the primary query population."""
    if cache_path is None:
        return None, None
    root = Path(cache_path)
    missing = [u for u in uids if not (root / f"{u}.npz").is_file()]
    if missing:
        return None, {"reason": "incomplete fields-text-cache", "uri": str(root),
                      "missing_count": len(missing), "missing_examples": missing[:5]}
    try:
        vectors = np.stack([np.load(root / f"{u}.npz", allow_pickle=False)["text"].astype(np.float32)
                            for u in uids])
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("text vectors must be a finite 2-D array")
    except (KeyError, ValueError, OSError) as exc:
        return None, {"reason": str(exc), "uri": str(root)}
    return vectors, None


def cosine_stats(S: np.ndarray, targets: np.ndarray, tau: float = 0.5, batch: int = 64, seed: int = 0) -> dict:
    """What the InfoNCE loss is made of, on the FULL cosine matrix S (queries x gallery).

    pos      mean cos(q_i, g_target(i))
    neg_mean mean cos(q_i, g_j), j != target
    hard_neg mean over queries of max_{j != target} cos(q_i, g_j)
    margin   mean of pos - hard_neg;  margin_pos_frac = fraction with margin > 0 (= R@1)
    loss_full     mean -log softmax at tau over the WHOLE gallery as negatives
    loss_batch64  the same with 63 random negatives per query (the training batch shape), one draw
    (Kyzen 2026-09-07: the plateau at ~2.4 needs these, not a floor argument.)"""
    n, m = S.shape
    rows = np.arange(n)
    pos = S[rows, targets]
    S_neg = S.copy(); S_neg[rows, targets] = -np.inf
    hard = S_neg.max(1)
    neg_mean = (S.sum(1) - pos) / (m - 1)
    margin = pos - hard
    z = S / tau
    lse_full = np.logaddexp.reduce(z, axis=1)
    loss_full = float(np.mean(lse_full - pos / tau))
    rng = np.random.default_rng(seed)
    losses = []
    for i in range(n):
        cand = rng.choice(m - 1, size=batch - 1, replace=False)
        cand = cand + (cand >= targets[i])          # skip the target column
        zi = np.concatenate([[z[i, targets[i]]], z[i, cand]])
        losses.append(float(np.logaddexp.reduce(zi) - zi[0]))
    return {"pos": float(pos.mean()), "neg_mean": float(neg_mean.mean()), "hard_neg": float(hard.mean()),
            "margin": float(margin.mean()), "margin_pos_frac": float((margin > 0).mean()),
            "pos_p10": float(np.percentile(pos, 10)), "margin_p10": float(np.percentile(margin, 10)),
            "loss_full": loss_full, "loss_batch64": float(np.mean(losses)), "tau": tau, "n": int(n), "m": int(m)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/home/kyzen/metafind/metafind_data_attrs/outputs/checkpoints/pilotP1_attrs_singleview_prefnorm_20260903/stage1_best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gallery-split", default="train_val")   # [D-3b] dev_val is outside train now
    ap.add_argument("--query-split", default="dev_val",
                    help="splits.json key for the query set (dev_val = val). `holdout` (val+test, 9,138) is the "
                         "20%% Kyzen's diagram reports on; every read of it is recorded in the ledger.")
    ap.add_argument("--out", default="output/look/exp_type_level_query.json")
    ap.add_argument("--fields-text-cache", default=None,
                    help="explicit optional FIELDS text cache; incomplete coverage skips this row and is recorded")
    ap.add_argument("--no-sketchfab", action="store_true")
    ap.add_argument("--pc-policies", default="canonical",
                    help="comma list from exp_query_pc_observation.POLICIES; `canonical` = the gallery's own "
                         "cloud, the rest are a SECOND surface sample of the same mesh (QueryPack) perturbed "
                         "by metafind.data.observation.perturb_cloud. Non-canonical policies score a focused "
                         "row subset (2026-09-06 weak-trio test).")
    ap.add_argument("--attribution", action="store_true",
                    help="single-modality replacement attribution (Kyzen 2026-09-07): rows A own/own/own, "
                         "B partner text only, C partner image only, D partner pc only, E partner text+image, "
                         "F all partner; text = attrs sentence, image = 12-view mean, pc = canonical cloud")
    ap.add_argument("--stage2-state", default=None,
                    help="a Stage 2 `stage2_full.pt` whose parent is --ckpt: its trained `query.fusion.*` "
                         "weights replace the Stage 1 query fusion (the gallery tower is frozen in Stage 2, "
                         "so the gallery stays Stage 1's). Layout branch not loaded: Objaverse assets have no "
                         "scene, so this is the Table 1 'w/ ESSGNN' shared head with the layout term absent.")
    ap.add_argument("--stage2-record", default=None,
                    help="record identifying --stage2-state (default: sibling variant_ckpts.json)")
    ap.add_argument("--stage2-variant", default="full")
    ap.add_argument("--allow-legacy-stage2-inputs", action="store_true",
                    help="explicitly permit old Stage 2 state with no embedded model metadata")
    args = ap.parse_args()
    policies = args.pc_policies.split(",")
    from metafind.train.stage1 import (build_model, load_protocols, load_stage1_checkpoint,
                                      load_stage1_model_config, effective_stage1_model_inputs,
                                      stage1_backbone_kwargs)
    from metafind.train.gallery_index import load_stage2_checkpoint_record
    from metafind.eval.run_retrieval import overlay_stage2_weights
    ckpt_record = load_stage1_model_config(args.ckpt)
    encoding, training, hyper = effective_stage1_model_inputs(ckpt_record, *load_protocols())
    if int(training["image_tokens"]) != 1:
        raise SystemExit("this single-vector observation probe requires image_tokens=1")
    s2_record = None
    if args.stage2_state:
        rec_path = args.stage2_record or Path(args.stage2_state).with_name("variant_ckpts.json")
        s2_record = load_stage2_checkpoint_record(rec_path, ckpt_record, args.stage2_variant,
                                                state_path=args.stage2_state,
                                                allow_legacy=args.allow_legacy_stage2_inputs)
    elif args.stage2_record:
        raise SystemExit("--stage2-record requires --stage2-state")

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    g_uids, q_uids = sorted(sp[args.gallery_split]), sorted(sp[args.query_split])
    where = {u: i for i, u in enumerate(g_uids)}
    targets = np.array([where[u] for u in q_uids])
    anns = {u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in q_uids}
    # partner: same rule as Stage1Dataset._build_partners (same-category inside dev_val, Random(uid_seed+11))
    pools = defaultdict(list)
    for u in q_uids:
        pools[anns[u]["lvis_category"]].append(u)
    partner = {}
    for u in q_uids:
        rng = random.Random(uid_seed(u) + 11)
        pool = [x for x in pools[anns[u]["lvis_category"]] if x != u] or [x for x in q_uids if x != u]
        partner[u] = rng.choice(pool)
    print(f"{len(q_uids):,} queries vs {len(g_uids):,} gallery; {len(pools):,} categories", flush=True)

    def emb(u, key):
        return np.load(paths.EMBEDDINGS / f"{u}.npz")[key].astype(np.float32)

    bb = ULIPBackbone(BackboneConfig(device=args.device, train_scope=training["train_scope"],
                                   **stage1_backbone_kwargs(ckpt_record)))
    bb_q = bb.clone_point_path() if training["tower_sharing"] == "fully_separate" else None
    model, loss = build_model(encoding, training, hyper)
    if training["freeze_gallery"]:
        model.freeze_gallery(True)
    load_stage1_checkpoint(bb, model, loss, Path(args.ckpt), query_backbone=bb_q)
    bb.set_train_scope("fuser_only")
    if bb_q is not None:
        bb_q.set_train_scope("fuser_only")
    model.to(args.device).eval()
    if s2_record is not None:
        overlay_stage2_weights(model, s2_record, args.device, fusion_only=True)
        print(f"  verified Stage 2 query fusion loaded from {s2_record['uri']}", flush=True)
    dev = args.device

    with torch.no_grad():
        # ---- query text variants
        texts = {"own(attrs)": np.stack([emb(u, "text") for u in q_uids]),
                 "partner(attrs)": np.stack([emb(partner[u], "text") for u in q_uids])}
        # Objaverse / Sketchfab metadata (OpenShape's `objaverse_meta.json`): the asset's own NAME,
        # tags, description -- about the target, but not the gallery's GPT-style form-fill.
        meta_path = Path("/home/kyzen/upstream/openshape-objaverse-embeddings/objaverse_meta.json")
        meta = {e["u"]: e for e in json.loads(meta_path.read_text())["entries"]} if meta_path.exists() else {}
        def size_of(u):
            a = anns[u]; return f"{float(a['width']):.0f} x {float(a['length']):.0f} x {float(a['height']):.0f} cm"
        variants = {name: [serialize_annotation(anns[u], template=tpl) for u in q_uids] for name, tpl in TEMPLATES.items()}
        if meta and not args.no_sketchfab:
            variants["sketchfab_name"] = [meta[u]["name"] for u in q_uids]
            variants["sketchfab_name_size"] = [f"{meta[u]['name']} {{size: {size_of(u)}}}" for u in q_uids]
            variants["sketchfab_name_tags"] = [", ".join([meta[u]["name"]] + list(meta[u].get("tags") or [])[:5]) for u in q_uids]
            variants["sketchfab_desc_or_name"] = [(meta[u].get("desc") or meta[u]["name"])[:300] for u in q_uids]
        fields, fields_skipped = optional_fields_text(args.fields_text_cache, q_uids)
        if fields is not None:
            texts["fields(attrs cache)"] = fields
        if fields_skipped:
            print(f"  fields(attrs cache) skipped: {fields_skipped}", flush=True)
        for name, sents in variants.items():
            print(f"  {name} e.g. {sents[0]!r}", flush=True)
            vecs = []
            for i in range(0, len(sents), 256):
                vecs.append(bb.encode_text(sents[i:i + 256]).float().cpu().numpy())
            texts[name] = np.concatenate(vecs)
        def one_view(u):
            v = emb(u, "views")                    # (n_views, D): 12 on the v6 corpus, 11 on v7
            return v[uid_seed(u) % v.shape[0]]
        images = {"own view": np.stack([one_view(u) for u in q_uids]),
                  "partner view": np.stack([one_view(partner[u]) for u in q_uids])}
        # ULIP-2 / OpenShape per-object observations (extract_ulip2_query_feats.py): the target's OWN
        # Sketchfab thumbnail (CLIP feature), its Sketchfab name, BLIP / Azure captions -- all ViT-bigG.
        u2p = Path("/home/kyzen/metafind/metafind_data/outputs/_probe/ulip2_query_feats/ulip2_query_feats.npz")
        covered = False
        if u2p.exists():
            z = np.load(u2p); row = {u: i for i, u in enumerate(z["uids"].tolist())}
            missing = [u for u in q_uids if u not in row or partner[u] not in row]
            covered = not missing
            if missing:
                print(f"  ULIP-2 feature cache lacks {len(missing)} of {len(q_uids)} query uids "
                      f"(or their partners); thumbnail / caption rows skipped", flush=True)
        if covered:
            idx = np.array([row[u] for u in q_uids])
            images["thumbnail(own)"] = z["thumbnail_feat"][idx].astype(np.float32)
            images["thumbnail(partner)"] = z["thumbnail_feat"][np.array([row[partner[u]] for u in q_uids])].astype(np.float32)
            for key, name in (("name_feat", "u2 name"), ("blip_feat", "u2 blip caption"), ("msft_feat", "u2 msft caption")):
                texts[name] = z[key][idx].astype(np.float32)
        # ---- gallery (P1's construction) and the query pc (= gallery pc, the asset's own)
        g_text = np.stack([emb(u, "text") for u in g_uids])
        g_img = np.stack([emb(u, "views").mean(0) for u in g_uids])
        g_pc, buf = [], []
        for i, u in enumerate(g_uids):
            c = np.load(paths.POINTCLOUDS / f"{u}.npz")
            buf.append(np.concatenate([c["xyz"], c["rgb"]], 1).astype(np.float32))
            if len(buf) == 48 or i == len(g_uids) - 1:
                g_pc.append(bb.encode_pc(torch.from_numpy(np.stack(buf))).float().cpu().numpy()); buf = []
            if i % 6000 == 0:
                print(f"  gallery pc {i:,}/{len(g_uids):,}", flush=True)
        g_pc = np.concatenate(g_pc)
        G = []
        for i in range(0, len(g_uids), 512):
            s = slice(i, i + 512)
            G.append(model.gallery({"text": torch.from_numpy(g_text[s]).to(dev), "image": torch.from_numpy(g_img[s]).to(dev),
                                    "pc": torch.from_numpy(g_pc[s]).to(dev)}).float().cpu())
        G = normalize_for_scoring(torch.cat(G).numpy())
        query_pc = g_pc
        if bb_q is not None:
            from tools.probes.exp_query_pc_observation import encode_clouds
            encoded = []
            for start in range(0, len(g_uids), 48):
                clouds = []
                for uid in g_uids[start:start + 48]:
                    with np.load(paths.POINTCLOUDS / f"{uid}.npz") as cloud:
                        clouds.append(np.concatenate([cloud["xyz"], cloud["rgb"]], 1).astype(np.float32))
                encoded.append(encode_clouds(bb_q, clouds))
            query_pc = np.concatenate(encoded)
        q_pcs = {"canonical": query_pc[targets]}
        # Table 1 row 1 as evaluated (DL-102): same_record text + same_mean image (the 12-view mean the
        # gallery itself used) + canonical cloud.
        images["own mean"] = g_img[targets]
        p_idx = np.array([where[partner[u]] for u in q_uids])
        images["partner mean"] = g_img[p_idx]
        q_pcs["partner"] = query_pc[p_idx]
        if [p for p in policies if p != "canonical"]:
            from metafind.train.stage1 import QueryPack
            from tools.probes.exp_query_pc_observation import perturb, encode_clouds
            pack = QueryPack(Path("/home/kyzen/metafind/metafind_data/outputs/_probe/query_pack/query_pack.json"), n_views=12)
            # only the pc arm is consumed here (the text variants above are built independently), so
            # require coverage of that arm alone: the text arm refuses assets without a second caption
            # (7 of 4,569 in val), which is irrelevant to a cloud-only draw.
            missing_pc = [u for u in q_uids if u not in pack.rows["pc"]]
            if missing_pc:
                raise SystemExit(f"query pack pc arm lacks {len(missing_pc)} of {len(q_uids)} query uids, e.g. {missing_pc[:3]}")
            for pol in policies:
                if pol == "canonical":
                    continue
                clouds = []
                for u in q_uids:
                    v = np.asarray(pack.vector("pc", u), dtype=np.float32)
                    clouds.append(perturb(v[:, :3], v[:, 3:6], pol, uid_seed(u) + 7))
                q_pcs[pol] = encode_clouds(bb_q or bb, clouds)
                cos = float((normalize_for_scoring(q_pcs[pol]) * normalize_for_scoring(g_pc[targets])).sum(1).mean())
                print(f"  query pc policy {pol}: paired cos to the gallery cloud {cos:.3f}", flush=True)

        combos = [("own(attrs)", "own mean", "Table 1 row 1 construction (same_record, same_mean)"),
                  ("own(attrs)", "own view", "parity: P1 as evaluated"),
                  ("partner(attrs)", "partner view", "parity: 5i (partner text+image)"),
                  ("cat_size", "partner view", "Figure-1 text of the TARGET + reference view")]
        for tn in ("fields(attrs cache)", "cat_only", "sketchfab_name", "sketchfab_name_size", "sketchfab_name_tags", "sketchfab_desc_or_name"):
            if tn in texts:
                combos += [(tn, "partner view", f"{tn} + reference view"), (tn, "own view", f"{tn} + own view")]
        if "thumbnail(own)" in images:
            combos += [("own(attrs)", "thumbnail(own)", "own fields text + OWN THUMBNAIL"),
                       ("own(attrs)", "thumbnail(partner)", "own fields text + partner thumbnail"),
                       ("u2 name", "thumbnail(own)", "Sketchfab name (u2 feat) + own thumbnail"),
                       ("u2 blip caption", "thumbnail(own)", "BLIP caption + own thumbnail"),
                       ("u2 msft caption", "thumbnail(own)", "Azure caption + own thumbnail"),
                       ("cat_size", "thumbnail(own)", "Figure-1 fields + own thumbnail")]
        focus = [c for c in combos if c[0] in ("own(attrs)", "partner(attrs)", "cat_size", "u2 name", "u2 blip caption")
                 and c[1] in ("own view", "own mean", "partner view", "thumbnail(own)")
                 and not (c[0] == "own(attrs)" and c[1] == "partner view")
                 and not (c[0] == "partner(attrs)" and c[1] != "partner view")
                 and not (c[0] == "cat_size" and c[1] == "partner view")]
        out = {"n_query": len(q_uids), "n_gallery": len(g_uids), "query_split": args.query_split,
               "gallery_split": args.gallery_split, "paper": PAPER, "pc_policies": policies,
               "ckpt": args.ckpt, "checkpoint_sha256": ckpt_record["sha256"],
               "stage2_state": args.stage2_state,
               "stage2_sha256": s2_record["sha256"] if s2_record else None,
               "allow_legacy_stage2_inputs": args.allow_legacy_stage2_inputs,
               "effective_model_inputs": {"encoding": encoding, "training": training},
               "runtime_source_sha256": runlog.runtime_source_sha256(),
               "runtime_source_status": runlog.runtime_source_status(),
               "fields_text_cache": args.fields_text_cache,
               "skipped_variants": {"fields(attrs cache)": fields_skipped} if fields_skipped else {},
               "rows": {}}
        print(f"\n{'query (text | image | pc)':<50}" + "".join(f"{c:>9}" for c in QUERY_CONDITIONS))
        print(f"{'paper w/o ESSGNN':<50}" + "".join(f"{PAPER[c]:>9.1f}" for c in QUERY_CONDITIONS))
        if args.attribution:
            plan = [("own(attrs)", "own mean", "canonical", "A  own text + own image + own pc"),
                    ("partner(attrs)", "own mean", "canonical", "B  partner TEXT only"),
                    ("own(attrs)", "partner mean", "canonical", "C  partner IMAGE only"),
                    ("own(attrs)", "own mean", "partner", "D  partner PC only"),
                    ("partner(attrs)", "partner mean", "canonical", "E  partner text + image"),
                    ("partner(attrs)", "partner mean", "partner", "F  all three partner (sanity)")]
        else:
            plan = [(t, i, pol, lab) for pol in policies for t, i, lab in (combos if pol == "canonical" else focus)]
        for tname, iname, pol, label in plan:
            q_pc = q_pcs[pol]
            if True:
                cells = {}
                for cond in QUERY_CONDITIONS:
                    Q = []
                    for i in range(0, len(q_uids), 512):
                        s = slice(i, i + 512)
                        e = {"text": torch.from_numpy(texts[tname][s]).to(dev), "image": torch.from_numpy(images[iname][s]).to(dev),
                             "pc": torch.from_numpy(q_pc[s]).to(dev)}
                        Q.append(model.query(e, present=condition_mask(cond, e["pc"].shape[0]).to(dev)).float().cpu())
                    S = normalize_for_scoring(torch.cat(Q).numpy()) @ G.T
                    cells[cond] = recall_at_k(S, targets)
                    cells[cond]["cos_stats"] = cosine_stats(S, targets, tau=0.5, batch=64, seed=20260907)
                key = f"{tname} | {iname}" + ("" if pol == "canonical" else f" | pc={pol}")
                out["rows"][key] = {"label": label, "pc_policy": pol, "cells": cells}
                print(f"{key:<50}" + "".join(f"{cells[c]['R@1']*100:>9.1f}" for c in QUERY_CONDITIONS) + f"   {label}", flush=True)
                for c in QUERY_CONDITIONS:
                    cs = cells[c]["cos_stats"]
                    print(f"    cos {c:<10} pos {cs['pos']:.3f}  neg {cs['neg_mean']:.3f}  hard {cs['hard_neg']:.3f}  "
                          f"margin {cs['margin']:.3f} (p10 {cs['margin_p10']:.3f})  loss full {cs['loss_full']:.2f}  b64 {cs['loss_batch64']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
