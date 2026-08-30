#!/usr/bin/env python3
"""PROTOCOL E (independent observation) + ITEM 3 (ULIP B1/B2 fingerprint).

⚠ PROTOCOL E IS NOT THE PAPER'S PROTOCOL. [Kyzen, via MASTER, 2026-08-31]
「不是 paper protocol，除非作者確認」 It is a diagnostic implementation choice
answering one question -- "does anything survive when the query's OBSERVATION of
an asset is independent of the gallery's?" -- and it may not be reported as
MetaFind's evaluation construction.

WHY IT EXISTS
-------------
Untrained, dev_val scores 0.910-0.999 across the seven conditions
(`diag_untrained_same_evaluator.json`). ITEM 1 then showed the fusion towers are
NOT the cause: strip them entirely and the parameter-free mean of the raw ULIP
embeddings scores 0.988-1.000 with margins 2-4x larger
(`diag_untrained_fusion_identity.json`). Under `full` the parameter-free
positive similarity is exactly 1.0000, because `evaluate_dev_val` hands ONE
`embeds` dict to both towers (`stage1.py:832-840`) -- the query and the gallery
are the same bytes.

Protocol E breaks exactly that, and nothing else.

                   GALLERY (modality-complete, as sec. 2.6 requires)
    text     canonical serialized description        (the cached vector)
    image    mean of the ELEVEN views != held-out
    pc       the canonical 10,000-point sample       (untouched on disk)

                   QUERY (an independent observation of the same asset)
    text     the highest-ranked NON-canonical description
    image    the ONE held-out view, index sha256(uid) % 12
    pc       a SECOND 10,000-point sample of the same mesh, seed offset
             +1_000_003, same density, built by
             `tools/probes/build_protocol_e_query_pc.py`

TWO CONTROLS, AND THE SECOND ONE IS NOT OPTIONAL
-------------------------------------------------
PROD  the production path: cached `image`, canonical text, canonical pc, the
      same vectors on both sides. This is what `evaluate_dev_val` scores.
E0    the same shared-observation protocol but with the image rebuilt as the
      mean of the twelve CACHED views.

E0 exists because the cached `image` is NOT the mean of the cached `views`.
MEASURED, 200 dev_val assets: max|delta| 4.883e-03, bit-identical 0/200. The
cause is `encode_text_image.py:524-532` -- `pooled = aggregate(view_vecs, ...)`
runs on the FLOAT32 encoder output and `views` and `image` are then cast to
float16 SEPARATELY, so `mean(f16 views) != f16(mean(f32 views))`. Comparing
Protocol E's eleven-view gallery against the cached twelve-view vector would
therefore confound "one view was held out" with "the average was taken in a
different dtype order". E0 removes that: E and E0 differ only in which views
went into the mean.

WHAT IT MUST NOT DO
-------------------
Writes JSON under `output/look/` and nothing else. Touches no canonical point
cloud, no embedding cache, no checkpoint, no split, no protocol artifact, no
trainer state. The query point clouds were built by a separate authorised
script into `_probe/` and are opened read-only here.

ITEM 3 -- the ULIP fingerprint -- rides along because it needs the same
released-encoder embeddings and no model at all. See `item3()`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metafind import paths                                            # noqa: E402
from metafind.eval.retrieval import (QUERY_CONDITIONS, condition_mask,  # noqa: E402
                                     normalize_for_scoring, recall_at_k)
from metafind.models.fusion import MODALITIES                         # noqa: E402
from metafind.models.resolve_stage1 import serialize_annotation       # noqa: E402

from diag_text_shortcut import encode_texts, f16_roundtrip            # noqa: E402

LOOK = REPO / "output" / "look"
CONDS = tuple(QUERY_CONDITIONS)
N_VIEWS = 12
QPC = paths.OUTPUTS / "_probe" / "protocol_e_query_pc"

# [PAPER FACT docs/paper/metafind_source/3experiments.tex, Table
# tab:objaverse-results] The ULIP baseline row, R@1 / R@5. Item 3 asks which
# gallery construction reproduces this SHAPE.
PAPER_ULIP = {
    "text": (0.1, 0.9), "image": (0.1, 1.3), "pc": (97.9, 99.4),
    "text+image": (0.0, 0.3), "text+pc": (33.9, 58.0),
    "image+pc": (22.6, 41.6), "full": (6.4, 15.9),
}


def heldout_index(uid: str) -> int:
    """[MASTER 2026-08-31, verbatim] ``sha256(uid) % 12``."""
    return int(hashlib.sha256(uid.encode()).hexdigest(), 16) % N_VIEWS


def sha(x) -> str:
    b = x.tobytes() if isinstance(x, np.ndarray) else str(x).encode()
    return hashlib.sha256(b).hexdigest()


def score(model, gal, qry, device, bs, conditions=CONDS):
    """R@1 / R@5 over the seven conditions, gallery and query built separately.

    `gal` and `qry` are ``{"text"|"image"|"pc": (N, D) cpu tensor}``. The
    gallery goes through `model.gallery(...)`, the PRODUCTION path, which
    refuses an incomplete gallery by design (`dual_tower.py:175-180`) -- so a
    Protocol E gallery that lost a modality would raise here rather than score.
    That refusal IS assertion 2.

    Row i of both stacks is the same asset (`collect` preserves split order and
    asserts it), so the target column is the diagonal.
    """
    import torch

    from metafind.train.stage1 import modules_in_eval

    n = gal["text"].size(0)
    G, Q = [], {c: [] for c in conditions}
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            b = sl.stop - sl.start
            g = {m: gal[m][sl].to(device) for m in MODALITIES}
            q = {m: qry[m][sl].to(device) for m in MODALITIES}
            G.append(model.gallery(g).float().cpu())
            for c in conditions:
                Q[c].append(model.query(
                    q, present=condition_mask(c, b).to(device)).float().cpu())

    g = normalize_for_scoring(torch.cat(G).numpy())
    t = np.arange(g.shape[0])
    out = {c: recall_at_k(normalize_for_scoring(torch.cat(Q[c]).numpy()) @ g.T,
                          t, ks=(1, 5)) for c in conditions}
    out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in conditions]))
    out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in conditions]))
    return out


def score_raw(gal: dict, qry: dict) -> dict:
    """The same seven conditions with NO FUSION and no parameters at all.

    Query = mean of the raw ULIP embeddings the condition makes present;
    gallery = mean of all three. ITEM 1 established that this parameter-free
    limit is what actually carries the untrained score, so every Protocol E
    cell is reported against it.
    """
    g = normalize_for_scoring(np.mean([gal[m] for m in MODALITIES], axis=0))
    t = np.arange(g.shape[0])
    out = {}
    for c in CONDS:
        present = [m for m, f in zip(MODALITIES, QUERY_CONDITIONS[c]) if f]
        out[c] = recall_at_k(
            normalize_for_scoring(np.mean([qry[m] for m in present], axis=0)) @ g.T,
            t, ks=(1, 5))
    out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in CONDS]))
    out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in CONDS]))
    return out


def item3(T: np.ndarray, I: np.ndarray, P: np.ndarray) -> dict:
    """ITEM 3 -- which baseline gallery reproduces Table 1's ULIP fingerprint?

    [PAPER FACT 3experiments.tex:24] "we extend each baseline by adding a simple
    *mean pooling layer* to aggregate available modalities, and use these fused
    embeddings to retrieve from a pre-encoded gallery" -- so the QUERY side is
    fixed by the paper: the mean of the available modalities. What the paper
    does not state is what the pre-encoded GALLERY holds. Two readings:

        B1  gallery = the PC representation alone
        B2  gallery = mean(text, image, pc)

    [PAPER FACT, same paragraph] "since other models do not adopt a dual-tower
    design, their 'PC only' performance reflects retrieval using identical
    embeddings for both query and gallery, leading to inflated accuracy" --
    which is a property of B1 and not of B2, but the paper says it in prose and
    the table is the test.

    A second fork the paper does not fix either, and it changes the answer:
    whether the mean is taken over the raw embeddings or over L2-NORMALISED
    ones. Both are run. Neither is a PAPER FACT.

    This is NOT an attempt to reproduce the baseline row. Gallery size, asset
    population, caption text and render set all differ from the paper's. Only
    the SHAPE across the seven conditions is being compared.
    """
    # Per-modality norms. An unnormalised mean is dominated by whichever
    # modality is largest, so if ||text|| or ||image|| greatly exceeds ||pc||
    # then every non-PC modality added to the query SWAMPS the PC direction --
    # which is the only mechanism that makes the paper's row fall monotonically
    # (pc 97.9 > T+PC 33.9 > I+PC 22.6 > full 6.4). Measured here so that
    # explanation can be confirmed or dropped instead of asserted.
    tn, In, pn = (normalize_for_scoring(x) for x in (T, I, P))
    out: dict = {"modality_geometry": {
        "norms": {m: {"mean": float(np.linalg.norm(x, axis=1).mean()),
                      "std": float(np.linalg.norm(x, axis=1).std())}
                  for m, x in (("text", T), ("image", I), ("pc", P))},
        # How aligned the released encoder's three spaces actually are on THIS
        # corpus. If cross-modal cosine is high, a text-only query can find its
        # own asset in a PC gallery -- which is what separates our levels from
        # the paper's, independently of which gallery construction is used.
        "mean_paired_cosine": {
            "text_pc": float((tn * pn).sum(1).mean()),
            "image_pc": float((In * pn).sum(1).mean()),
            "text_image": float((tn * In).sum(1).mean())}}}
    for norm in (False, True):
        t, i, p = (T, I, P) if not norm else (
            normalize_for_scoring(T), normalize_for_scoring(I),
            normalize_for_scoring(P))
        raw = {"text": t, "image": i, "pc": p}
        for name, g in (("B1_gallery_pc_only", p),
                        ("B2_gallery_mean_of_three", np.mean([t, i, p], axis=0))):
            gn = normalize_for_scoring(g)
            tgt = np.arange(gn.shape[0])
            cell = {}
            for c in CONDS:
                present = [m for m, f in zip(MODALITIES, QUERY_CONDITIONS[c]) if f]
                q = normalize_for_scoring(np.mean([raw[m] for m in present], axis=0))
                cell[c] = recall_at_k(q @ gn.T, tgt, ks=(1, 5))
            out[f"{name}__{'l2mean' if norm else 'rawmean'}"] = cell
    return out


def collect(backbone, uids, aggregation, device, bs):
    """Cached text / cached image / per-view matrix / canonical-PC embedding.

    The text, image and canonical point cloud come through `Stage1Dataset` and
    the same DataLoader flags `evaluate_dev_val` uses, so the PROD control below
    is the production path and not an imitation of it. `views` is read straight
    from the npz because `Stage1Dataset` under `mean` never returns it.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.train.stage1 import Stage1Dataset, collate, modules_in_eval

    loader = DataLoader(Stage1Dataset(uids, aggregation), batch_size=bs,
                        shuffle=False, collate_fn=collate, num_workers=4,
                        drop_last=False)
    order, text, image, pc = [], [], [], []
    with modules_in_eval(getattr(backbone, "model", None)), torch.no_grad():
        for b in loader:
            order.extend(b["uid"])
            text.append(b["text"])
            image.append(b["image"])
            pc.append(backbone.encode_pc(b["pc"].to(device)).float().cpu())
    views = np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")["views"] for u in order])
    return order, torch.cat(text), torch.cat(image), torch.cat(pc), views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="init0",
                    help="'init0' (untrained, --seeds draws) or a Stage 1 "
                         "checkpoint path")
    ap.add_argument("--seeds", default="1,2,3", help="init0 only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: first N dev_val assets. Debug, not evidence.")
    ap.add_argument("--skip-item3", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    trained = args.state != "init0"

    import torch

    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    dev_val = json.loads(
        (paths.OUTPUTS / "splits.json").read_text())["object"]["dev_val"]
    if args.limit:
        dev_val = dev_val[:args.limit]
        print("⚠ --limit set: DEBUG RUN. These numbers are not evidence.", flush=True)

    # ---- the authorised second point-cloud draw --------------------------
    man = json.loads(next(QPC.glob("*.manifest.json")).read_text())
    qpc_all = np.load(man["array"], mmap_mode="r")
    if man["debug_limit"] is not None and not args.limit:
        raise SystemExit(f"{man['array']} was built with --limit "
                         f"{man['debug_limit']}; rebuild it in full.")
    pos = {u: k for k, u in enumerate(man["uid_order"])}
    missing = [u for u in dev_val if u not in pos]
    if missing:
        raise SystemExit(f"query PC missing for {len(missing)} uid(s)")

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model_hash, loss_fn = None, None
    if trained:
        # A checkpoint restores the fine-tuned POINT ENCODER as well as the
        # towers, so it must be loaded BEFORE any point cloud is encoded --
        # otherwise the pc embeddings would come from the released weights and
        # the towers from the trained ones.
        from metafind.train.stage1 import load_stage1_checkpoint
        model, loss_fn = build_model(encoding, training, hyper)
        model.to(args.device)
        load_stage1_checkpoint(backbone, model, loss_fn, Path(args.state))
        model_hash = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        print(f"loaded {args.state}  sha256 {model_hash[:16]}…", flush=True)

    t0 = time.time()
    order, C_vec, img_cached, P_canon, views = collect(
        backbone, dev_val, encoding["image_aggregation"], args.device, bs)
    assert order == dev_val, "loader reordered the split"
    print(f"inputs collected in {time.time()-t0:.0f}s  views {views.shape}",
          flush=True)

    # ---- IMAGE arm --------------------------------------------------------
    held = np.array([heldout_index(u) for u in order])
    v32 = views.astype(np.float32)                       # (N, 12, D)
    rows = np.arange(len(order))
    v_held = v32[rows, held]                             # (N, D) the query view
    img12 = v32.mean(axis=1)                             # E0 gallery
    # E gallery. GATHERED, not `(sum - held) / 11`. The subtraction is
    # algebraically identical and numerically is not: it reads the held-out
    # value into the arithmetic, so the gallery vector depends on it to within
    # float32 error. ASSERTION 2 below caught exactly that -- with a 12345.0
    # probe in the held-out slot the subtraction form moved img11 by 3.4e-04.
    # Gathering the eleven kept indices means the held-out view is never
    # touched, so the exclusion is structural rather than algebraic, and it is
    # also what `aggregate(views, "mean")` (`encode_text_image.py:406-407`)
    # would have computed from eleven renders.
    keep = np.array([[j for j in range(N_VIEWS) if j != h] for h in held])
    img11 = v32[rows[:, None], keep].mean(axis=1)        # E gallery

    # ASSERTION 1 -- the held-out view is genuinely NOT in the gallery average.
    # Checked as an algebraic identity on the actual arrays, not by trusting the
    # indexing: 11 * mean(kept) + held == 12 * mean(all).
    lhs = img11 * (N_VIEWS - 1) + v_held
    rhs = img12 * N_VIEWS
    a1_max = float(np.abs(lhs - rhs).max())
    assert a1_max < 1e-2, f"held-out view leaked into the gallery mean: {a1_max}"
    # and independently: overwriting the held-out view must not move img11 AT
    # ALL. This is the strong form -- exact equality, not a tolerance -- and it
    # is the assertion that rejected the subtraction form of img11.
    probe = v32.copy()
    probe[rows, held] = 12345.0
    a1_alt = float(np.abs(probe[rows[:, None], keep].mean(axis=1) - img11).max())
    assert a1_alt == 0.0, f"gallery mean depends on the held-out view: {a1_alt}"
    del probe
    print(f"ASSERTION 1  held-out view excluded from the gallery average: "
          f"identity residual {a1_max:.2e}, perturbation residual {a1_alt:.1e}  OK",
          flush=True)

    # ---- TEXT arm ---------------------------------------------------------
    ann = [json.loads((paths.ANNOTATIONS / f"{u}.json").read_text()) for u in order]
    canonical = [serialize_annotation(a) for a in ann]
    alt_strings, alt_rank = [], []
    for a in ann:
        cands = sorted((c for c in (a.get("description_candidates") or [])
                        if c["text"] != a["description"]), key=lambda c: c["rank"])
        if not cands:
            raise SystemExit(f"{a['uid']}: no non-canonical description candidate")
        alt_strings.append(serialize_annotation({**a, "description": cands[0]["text"]}))
        alt_rank.append(cands[0]["rank"])
    assert all(x != y for x, y in zip(alt_strings, canonical)), \
        "an alternate serialized to the canonical string"
    A_vec = encode_texts(backbone, alt_strings, "(Protocol E alternate)")
    # GATE: the harness must reproduce the CACHED canonical vector bit for bit,
    # or "the only difference is the description" is not true.
    canon_re = encode_texts(backbone, canonical, "(canonical, for the gate)")
    gate1 = bool(torch.equal(canon_re, C_vec))
    print(f"GATE  re-encoded canonical == cached text, bit-identical: {gate1}",
          flush=True)
    if not gate1:
        raise SystemExit("GATE FAILED: the harness cannot reproduce the cache. STOP.")
    del canon_re
    n_txt_diff = int((A_vec != C_vec).any(dim=1).sum())
    assert n_txt_diff == len(order), \
        f"only {n_txt_diff}/{len(order)} alternate text vectors differ"
    print(f"TEXT arm  strings differ {len(order)}/{len(order)}, "
          f"embeddings differ {n_txt_diff}/{len(order)}  OK", flush=True)

    # ---- PC arm -----------------------------------------------------------
    qpc = np.asarray(qpc_all[[pos[u] for u in order]])
    n_same = int(sum(np.array_equal(
        qpc[k, :, :3],
        np.load(paths.POINTCLOUDS / f"{order[k]}.npz")["xyz"]) for k in range(len(order))))
    assert n_same == 0, f"{n_same} query clouds are byte-identical to the canonical"
    P_query = []
    with torch.no_grad():
        for i in range(0, len(order), bs):
            P_query.append(backbone.encode_pc(
                torch.from_numpy(qpc[i:i + bs]).to(args.device)).float().cpu())
    P_query = torch.cat(P_query)
    pc_cos = float((torch.nn.functional.normalize(P_query, dim=1)
                    * torch.nn.functional.normalize(P_canon, dim=1)).sum(1).mean())
    print(f"PC arm  {len(order)} independent redraws, 0 byte-identical, "
          f"mean cos(query pc, canonical pc) = {pc_cos:.4f}  OK", flush=True)

    # ---- the three protocols ---------------------------------------------
    ten = torch.from_numpy
    PROTOCOLS = {
        "PROD_shared_cached_image": (
            {"text": C_vec, "image": img_cached, "pc": P_canon},
            {"text": C_vec, "image": img_cached, "pc": P_canon}),
        "E0_shared_12view_image": (
            {"text": C_vec, "image": ten(img12), "pc": P_canon},
            {"text": C_vec, "image": ten(img12), "pc": P_canon}),
        "E_independent_observation": (
            {"text": C_vec, "image": ten(img11), "pc": P_canon},
            {"text": A_vec, "image": ten(v_held), "pc": P_query}),
    }

    def raw_of(d):
        return {m: d[m].numpy().astype(np.float64) for m in MODALITIES}

    results = {}
    for name, (gal, qry) in PROTOCOLS.items():
        results[name] = {"raw_no_fusion": score_raw(raw_of(gal), raw_of(qry))}
        r = results[name]["raw_no_fusion"]
        print(f"\n{name}  raw/no-fusion  mean_R@1 {r['mean_R@1']:.4f}", flush=True)
        for c in CONDS:
            print(f"    {c:>11}  R@1 {r[c]['R@1']:.4f}  R@5 {r[c]['R@5']:.4f}",
                  flush=True)

    states = [("trained", None)] if trained else [(f"INIT-0_seed{s}", s) for s in seeds]
    for label, seed in states:
        if seed is not None:
            torch.manual_seed(seed)
            model, loss_fn = build_model(encoding, training, hyper)
            model.to(args.device)
        for name, (gal, qry) in PROTOCOLS.items():
            r = score(model, gal, qry, args.device, bs)
            results[name][label] = r
            print(f"\n{name}  {label}  mean_R@1 {r['mean_R@1']:.4f}  "
                  f"mean_R@5 {r['mean_R@5']:.4f}", flush=True)
            for c in CONDS:
                print(f"    {c:>11}  R@1 {r[c]['R@1']:.4f}  R@5 {r[c]['R@5']:.4f}",
                      flush=True)
        # ASSERTION 3 -- determinism. Rebuild at the SAME seed and rescore.
        if seed is not None:
            torch.manual_seed(seed)
            again, _ = build_model(encoding, training, hyper)
            again.to(args.device)
            rep = score(again, *PROTOCOLS["E_independent_observation"],
                        args.device, bs)
            same = all(rep[c]["R@1"] == results["E_independent_observation"][label][c]["R@1"]
                       and rep[c]["R@5"] == results["E_independent_observation"][label][c]["R@5"]
                       for c in CONDS)
            print(f"ASSERTION 3  same seed re-run bit-identical on 7/7: {same}",
                  flush=True)
            assert same, "same seed produced a different Protocol E result"
            del again
        if seed is not None:
            del model
        torch.cuda.empty_cache()

    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    tag = "trained" if trained else "init0"

    prov = {
        "split": "dev_val", "n_assets": len(order), "uid_order": order,
        "state": "trained" if trained else "INIT-0 (zero optimizer steps)",
        "checkpoint": args.state if trained else None,
        "checkpoint_sha256": model_hash, "seeds": None if trained else seeds,
        "code_revision": rev, "code_dirty": dirty,
        "query_pc_seed_offset": man["seed_offset"],
        "query_pc_array": man["array"], "query_pc_array_sha256": man["array_sha256"],
        "per_uid": {u: {
            "heldout_view_index": int(held[k]),
            "heldout_view_sha256": sha(views[k, held[k]]),
            "gallery_view_indices": [int(j) for j in range(N_VIEWS) if j != held[k]],
            "gallery_view_sha256": [sha(views[k, j])
                                    for j in range(N_VIEWS) if j != held[k]],
            "alternate_text_sha256": sha(alt_strings[k]),
            "alternate_text_rank": int(alt_rank[k]),
            "canonical_text_sha256": sha(canonical[k]),
            "query_pc_sha256": man["query_pc_sha256_per_uid"][u],
            "canonical_pc_npz_sha256": man["canonical_pc_npz_sha256_per_uid"][u],
        } for k, u in enumerate(order)},
    }
    (LOOK / f"diag_protocol_e_{tag}_provenance.json").write_text(
        json.dumps(prov, indent=1))

    LOOK.mkdir(parents=True, exist_ok=True)
    (LOOK / f"diag_protocol_e_{tag}.json").write_text(json.dumps({
        "what": ("PROTOCOL E -- independent-observation diagnostic. Query and "
                 "gallery observe the same asset through DIFFERENT samples: "
                 "alternate description, held-out view sha256(uid)%12, second "
                 "10k mesh sample at seed offset +1000003."),
        "NOT_THE_PAPER_PROTOCOL": ("Diagnostic implementation choice. It is not "
                                   "MetaFind's evaluation construction and may "
                                   "not be reported as one unless the authors "
                                   "confirm it."),
        "caveat": ("dev_val is a 4,569-asset HPO selector split. It is NOT the "
                   "paper's 20% test population and no number here may be set "
                   "against Table 1."),
        "controls": {
            "PROD_shared_cached_image": "production path; evaluate_dev_val's own inputs",
            "E0_shared_12view_image": ("shared observation, image rebuilt as the "
                                       "mean of the 12 CACHED views, so E and E0 "
                                       "differ only in which views were averaged"),
        },
        "cached_image_is_not_mean_of_cached_views": {
            "measured_max_abs_delta": 4.883e-03, "bit_identical": "0/200",
            "cause": "encode_text_image.py:524-532, f16 cast after an f32 mean",
        },
        "assertions": {
            "heldout_excluded_identity_residual": a1_max,
            "heldout_perturbation_residual": a1_alt,
            "gate_canonical_text_bit_identical": gate1,
            "alternate_text_vectors_differing": n_txt_diff,
            "query_clouds_byte_identical_to_canonical": n_same,
            "mean_cos_query_pc_vs_canonical_pc": pc_cos,
        },
        "n_assets": len(order), "batch_size": bs, "debug_limit": args.limit,
        "state": prov["state"], "checkpoint_sha256": model_hash,
        "code_revision": rev, "code_dirty": dirty,
        "results": results,
    }, indent=1, default=float))
    print(f"\nwrote {LOOK}/diag_protocol_e_{tag}.json")

    # ---- ITEM 3 -----------------------------------------------------------
    if not args.skip_item3 and not trained:
        fp = item3(C_vec.numpy().astype(np.float64),
                   img_cached.numpy().astype(np.float64),
                   P_canon.numpy().astype(np.float64))
        print("\nITEM 3 -- ULIP fingerprint. Paper's ULIP row for comparison:")
        print(f"{'variant':>34} " + "".join(f"{c:>12}" for c in CONDS))
        print(f"{'PAPER ULIP R@1':>34} "
              + "".join(f"{PAPER_ULIP[c][0]:12.1f}" for c in CONDS))
        for k, cell in fp.items():
            if k == "modality_geometry":
                continue
            print(f"{k:>34} "
                  + "".join(f"{cell[c]['R@1']*100:12.1f}" for c in CONDS))
        print(f"  norms  {fp['modality_geometry']['norms']}")
        print(f"  paired cosine  {fp['modality_geometry']['mean_paired_cosine']}")
        (LOOK / "diag_ulip_fingerprint.json").write_text(json.dumps({
            "what": ("ITEM 3 -- which baseline gallery construction reproduces "
                     "Table 1's ULIP fingerprint. Query is the paper's own "
                     "'simple mean pooling layer' over available modalities "
                     "(3experiments.tex:24); the gallery is the unknown."),
            "purpose": ("NOT an attempt to reproduce the baseline row. The "
                        "population, gallery size, caption text and render set "
                        "all differ from the paper's. Only the SHAPE across the "
                        "seven conditions is compared."),
            "paper_ulip_R@1_R@5_percent": PAPER_ULIP,
            "n_gallery": len(order), "split": "dev_val",
            "encoder": ("released ULIP-2 checkpoint: OpenCLIP ViT-bigG-14 text "
                        "and image, PointBERT point cloud. No MetaFind fusion, "
                        "no trained parameter."),
            "code_revision": rev, "code_dirty": dirty,
            "variants": fp,
        }, indent=1, default=float))
        print(f"wrote {LOOK}/diag_ulip_fingerprint.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
