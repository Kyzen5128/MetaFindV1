#!/usr/bin/env python3
"""U-09 GALLERY-SIZE SENSITIVITY DIAGNOSTIC.

⚠ NOT A PAPER REPRODUCTION. [Codex via MASTER, 2026-08-31] The paper's candidate
pool is still UNKNOWN -- that is what U-09 IS. This measures how the reported
numbers move as the pool grows, and nothing here may be set against Table 1.

THE QUESTION
------------
dev_val scores 0.90-1.00 across the seven conditions against a 4,569-asset
gallery. Does that survive a larger pool, or is it mostly an artefact of the
denominator? U-09 records that the paper never says whether the gallery is the
20% test split or the whole corpus (`metafind/eval/retrieval.py`), and the two
readings are 9,138 and 45,692 here.

THE CONTROLS THAT MAKE IT A SIZE EXPERIMENT AND NOT A DIFFERENT ONE
--------------------------------------------------------------------
    same 4,569 QUERIES at every size          -- the query set never moves
    every gallery is a STRICT SUBSET of the   -- nested index prefixes, so this
      next one                                   is true by construction
    a UID in two galleries uses the IDENTICAL -- the gallery matrix is built
      vector in both                             ONCE over 45,692 and the
                                                 smaller sizes are COLUMN
                                                 SLICES of that same array, so
                                                 the vectors are the same bytes
                                                 rather than merely equal
    the only change permitted is added        -- asserted below
      distractors

dev_val occupies index 0..4568, so the target column of query i is i at every
size, and the smaller galleries are `sim[:, :N]`. Slicing columns AFTER the
GEMM is what makes "identical vector" exact instead of a claim about
determinism: there is only one gallery encode.

TWO PROTOCOLS, REPORTED SEPARATELY
-----------------------------------
    PROD    query and gallery are the SAME cached vectors -- templated text,
            cached image, canonical point cloud. What `evaluate_dev_val`
            scores.
    BARE    the corrected description protocol. Query text is the BARE
            highest-ranked NON-canonical description; gallery text is the BARE
            canonical description -- no template on either side, so the
            template's shared category / materials / dimensions / placement
            cannot carry the match. Image and pc are the independent
            observation: the query is the held-out view sha256(uid)%12 and a
            second 10,000-point mesh sample, the gallery is the mean of the
            other eleven views and the canonical cloud.

⚠ BARE GOES AROUND A PRODUCTION GATE. n06 refuses a string over CLIP's
77-token context (`refuse_if_overlong`) and `serialize_annotation` caps the
description so a templated string never is. A bare one can be, and is silently
truncated by the tokenizer. The count is measured and reported, not dropped:
dropping those assets would change the gallery size, which is the one variable
this diagnostic is controlling.

EVERY GALLERY ASSET DROPS A VIEW UNDER BARE, not just the dev_val ones, so the
eleven-view average is the same construction for a positive and for a
distractor. A gallery whose distractors were twelve-view and whose positives
were eleven-view would hand the positives a systematic disadvantage that has
nothing to do with pool size.

WHAT IT WRITES
--------------
One JSON per state under `output/look/`, plus a cached bare-caption matrix
under `data/outputs/_probe/`. MODIFIES NO CANONICAL ARTIFACT: annotations, the
embedding cache, the point clouds, the splits, the protocols and every
checkpoint are opened read-only.
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
                                     normalize_for_scoring, rank_of_target,
                                     recall_at_k)
from metafind.models.fusion import MODALITIES                         # noqa: E402

from diag_text_shortcut import encode_texts                           # noqa: E402
from diag_protocol_e_ulip_fingerprint import N_VIEWS, QPC, heldout_index  # noqa: E402

LOOK = REPO / "output" / "look"
CONDS = tuple(QUERY_CONDITIONS)
PROBE = paths.OUTPUTS / "_probe"
# 4,569 is the current protocol; 9,138 and 45,692 are U-09's two readings of
# "the gallery" (the 20% test split, and the whole admitted corpus); 22,846 is
# the midpoint, so the trend has four points rather than a line through two.
SIZES = (4569, 9138, 22846, 45692)


def stats_from_sim(sim: np.ndarray, targets: np.ndarray) -> dict:
    """R@1/R@5, positive rank and hardest-negative margin from one sim matrix.

    Takes the matrix rather than the two embedding stacks so a smaller gallery
    can be scored as a COLUMN SLICE of the big one -- which is what makes the
    nested-subset guarantee exact.
    """
    n = sim.shape[0]
    r = rank_of_target(sim, targets)
    pos = sim[np.arange(n), targets]
    off = sim.copy()
    off[np.arange(n), targets] = -np.inf
    neg = off.max(axis=1)
    margin = pos - neg
    rec = recall_at_k(sim, targets, ks=(1, 5))
    # margin > 0 iff no other column scores >= the positive iff rank == 1, and
    # `rank_of_target` counts ties against the model. Two fractions of the same
    # integer count, so exact equality -- the indexing check for this block.
    assert float((margin > 0).mean()) == rec["R@1"], "margin and R@1 disagree"
    return {"R@1": rec["R@1"], "R@5": rec["R@5"],
            "n_query": rec["n_query"], "n_gallery": rec["n_gallery"],
            "positive_rank": {"median": float(np.median(r)),
                              "mean": float(r.mean()), "p95": float(np.percentile(r, 95)),
                              "max": int(r.max())},
            "positive_sim_mean": float(pos.mean()),
            "hardest_negative_sim_mean": float(neg.mean()),
            "margin": {"mean": float(margin.mean()), "std": float(margin.std()),
                       "p5": float(np.percentile(margin, 5)),
                       "p50": float(np.percentile(margin, 50)),
                       "p95": float(np.percentile(margin, 95))}}


def ladder(Q: np.ndarray, G: np.ndarray, sizes) -> dict:
    """One GEMM against the FULL gallery, then a column slice per size."""
    sim = normalize_for_scoring(Q) @ normalize_for_scoring(G).T
    t = np.arange(sim.shape[0])
    out = {str(n): stats_from_sim(sim[:, :n], t) for n in sizes}
    del sim
    return out


def load_cached(uids, key):
    """`key` out of every asset's embedding npz, in `uids` order."""
    return np.stack([np.load(paths.EMBEDDINGS / f"{u}.npz")[key] for u in uids])


def encode_pc_all(backbone, uids, device, bs):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(uids), bs):
            batch = np.stack([
                (lambda z: np.concatenate([z["xyz"], z["rgb"]], axis=1))(
                    np.load(paths.POINTCLOUDS / f"{u}.npz"))
                for u in uids[i:i + bs]]).astype(np.float32)
            out.append(backbone.encode_pc(
                torch.from_numpy(batch).to(device)).float().cpu())
            if (i // bs) % 100 == 0:
                print(f"    pc {i:,}/{len(uids):,}", flush=True)
    return torch.cat(out)


def fuse(model, embeds: dict, cond: str | None, device, bs):
    """Gallery (`cond is None`, production path) or query under one condition."""
    import torch

    from metafind.train.stage1 import modules_in_eval

    n = embeds["text"].size(0)
    out = []
    with modules_in_eval(model), torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            e = {m: embeds[m][sl].to(device) for m in MODALITIES}
            v = (model.gallery(e) if cond is None else
                 model.query(e, present=condition_mask(
                     cond, sl.stop - sl.start).to(device)))
            out.append(v.float().cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="init0",
                    help="'init0' (untrained, --seeds draws) or a checkpoint path")
    ap.add_argument("--seeds", default="1,2,3", help="init0 only")
    ap.add_argument("--distractor-seed", type=int, default=20260831)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-gallery", type=int, default=None,
                    help="SMOKE ONLY: cap the largest gallery. Debug, not evidence.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    trained = args.state != "init0"

    import torch

    from metafind.data.encode_text_image import true_token_count
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyper = load_protocols()
    bs = hyper["values"]["batch_size"]
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    dev_val = sp["dev_val"]
    corpus = sorted(set(sp["train"]) | set(sp["test"]))

    # ---- the nested index. dev_val FIRST, so the target column is the row. ----
    rest = [u for u in corpus if u not in set(dev_val)]
    rng = np.random.default_rng(args.distractor_seed)
    rng.shuffle(rest)
    order = list(dev_val) + rest
    sizes = [n for n in SIZES if n <= len(order)]
    if args.limit_gallery:
        sizes = [n for n in sizes if n <= args.limit_gallery] or [len(dev_val)]
        order = order[:max(sizes)]
        print("⚠ --limit-gallery set: DEBUG RUN. Not evidence.", flush=True)
    assert order[:len(dev_val)] == dev_val, "dev_val is not the index prefix"
    assert len(set(order)) == len(order), "the gallery index repeats a uid"
    for a, b in zip(sizes, sizes[1:]):
        assert set(order[:a]) < set(order[:b]), f"gallery {a} is not a strict subset of {b}"
    print(f"gallery ladder {sizes}  queries {len(dev_val):,}  "
          f"distractor seed {args.distractor_seed}", flush=True)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model_t, ckpt_sha = None, None
    if trained:
        # BEFORE any point cloud is encoded: the checkpoint restores the
        # fine-tuned PointBERT as well as the towers.
        from metafind.train.stage1 import load_stage1_checkpoint
        model_t, loss_fn = build_model(encoding, training, hyper)
        model_t.to(args.device)
        load_stage1_checkpoint(backbone, model_t, loss_fn, Path(args.state))
        ckpt_sha = hashlib.sha256(Path(args.state).read_bytes()).hexdigest()
        print(f"loaded {args.state}  sha256 {ckpt_sha[:16]}…", flush=True)

    t0 = time.time()
    T_tpl = torch.from_numpy(load_cached(order, "text").astype(np.float32))
    I_prod = torch.from_numpy(load_cached(order, "image").astype(np.float32))
    views = load_cached(order, "views").astype(np.float32)
    print(f"cached vectors loaded in {time.time()-t0:.0f}s  views {views.shape}",
          flush=True)

    # ---- BARE image: every gallery asset drops its own held-out view ----
    held = np.array([heldout_index(u) for u in order])
    rows = np.arange(len(order))
    keep = np.array([[j for j in range(N_VIEWS) if j != h] for h in held])
    I_bare = views[rows[:, None], keep].mean(axis=1)
    probe = views.copy()
    probe[rows, held] = 12345.0
    a_img = float(np.abs(probe[rows[:, None], keep].mean(axis=1) - I_bare).max())
    assert a_img == 0.0, f"gallery mean depends on the held-out view: {a_img}"
    del probe
    v_held = views[rows[:len(dev_val)], held[:len(dev_val)]]
    del views
    print(f"BARE image  held-out view excluded, perturbation residual {a_img:.1e}  OK",
          flush=True)

    # ---- BARE text. Cached: CLIP is frozen, so this is state-independent. ----
    bare_C, bare_A = [], []
    for u in order:
        a = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
        bare_C.append(a["description"])
    for u in dev_val:
        a = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
        c = a["description"]
        cands = [x for x in (a.get("description_candidates") or []) if x["text"] != c]
        if not cands:
            raise SystemExit(f"{u}: no non-canonical description candidate")
        bare_A.append(min(cands, key=lambda x: x["rank"])["text"])
    over = {"bare_gallery_C": int(sum(true_token_count(s) > 77 for s in bare_C)),
            "bare_query_A": int(sum(true_token_count(s) > 77 for s in bare_A))}
    dup = {"bare_gallery_C": len(bare_C) - len(set(bare_C)),
           "bare_query_A": len(bare_A) - len(set(bare_A))}
    print(f"BARE text  over CLIP's 77-token context (silently truncated) {over}\n"
          f"BARE text  duplicate strings {dup}", flush=True)

    PROBE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\x00".join(bare_C)).encode()).hexdigest()[:16]
    cache = PROBE / f"bare_caption_text_{key}.npy"
    if cache.exists():
        T_bare = torch.from_numpy(np.load(cache))
        print(f"BARE text  reusing {cache.name} ({len(bare_C):,} vectors)", flush=True)
    else:
        T_bare = encode_texts(backbone, bare_C, f"(bare canonical, {len(bare_C):,})")
        np.save(cache, T_bare.numpy())
        print(f"BARE text  wrote {cache}", flush=True)
    assert T_bare.shape[0] == len(order)
    A_bare = encode_texts(backbone, bare_A, "(bare alternate, dev_val)")

    # ---- BARE query point clouds: the authorised second draw ----
    man = json.loads(next(QPC.glob("*.manifest.json")).read_text())
    if man["debug_limit"] is not None:
        raise SystemExit(f"{man['array']} is a debug build; rebuild it in full.")
    qpc_all = np.load(man["array"], mmap_mode="r")
    pos = {u: k for k, u in enumerate(man["uid_order"])}
    qpc = np.asarray(qpc_all[[pos[u] for u in dev_val]])

    # ---- point clouds, per state (the checkpoint moves PointBERT) ----
    print("encoding canonical point clouds over the full gallery…", flush=True)
    t0 = time.time()
    P_canon = encode_pc_all(backbone, order, args.device, bs)
    P_query = []
    with torch.no_grad():
        for i in range(0, len(dev_val), bs):
            P_query.append(backbone.encode_pc(
                torch.from_numpy(qpc[i:i + bs]).to(args.device)).float().cpu())
    P_query = torch.cat(P_query)
    print(f"  point clouds encoded in {time.time()-t0:.0f}s", flush=True)

    nq = len(dev_val)
    ten = torch.from_numpy
    PROTOCOLS = {
        "PROD": ({"text": T_tpl, "image": I_prod, "pc": P_canon},
                 {"text": T_tpl[:nq], "image": I_prod[:nq], "pc": P_canon[:nq]}),
        "BARE": ({"text": T_bare, "image": ten(I_bare), "pc": P_canon},
                 {"text": A_bare, "image": ten(v_held), "pc": P_query}),
    }

    results: dict = {p: {} for p in PROTOCOLS}
    for p, (gal, qry) in PROTOCOLS.items():
        g64 = {m: gal[m].numpy().astype(np.float64) for m in MODALITIES}
        q64 = {m: qry[m].numpy().astype(np.float64) for m in MODALITIES}
        G = np.mean([g64[m] for m in MODALITIES], axis=0)
        cells = {}
        for c in CONDS:
            present = [m for m, f in zip(MODALITIES, QUERY_CONDITIONS[c]) if f]
            cells[c] = ladder(np.mean([q64[m] for m in present], axis=0), G, sizes)
        results[p]["raw_no_fusion"] = cells
        del g64, q64, G

    states = [("trained", None)] if trained else [(f"INIT-0_seed{s}", s)
                                                  for s in seeds]
    for label, seed in states:
        if seed is None:
            model = model_t
        else:
            torch.manual_seed(seed)
            model, _ = build_model(encoding, training, hyper)
            model.to(args.device)
        for p, (gal, qry) in PROTOCOLS.items():
            G = fuse(model, gal, None, args.device, bs).numpy()
            cells = {}
            for c in CONDS:
                cells[c] = ladder(fuse(model, qry, c, args.device, bs).numpy(),
                                  G, sizes)
            results[p][label] = cells
            del G
        if seed is not None:
            del model
        torch.cuda.empty_cache()
        print(f"  {label} done", flush=True)

    for p in PROTOCOLS:
        print(f"\nU-09 {p}   R@1 by gallery size")
        print(f"{'state':>16} {'cond':>11} " + "".join(f"{n:>10,}" for n in sizes))
        for lab in results[p]:
            for c in CONDS:
                print(f"{lab:>16} {c:>11} " + "".join(
                    f"{results[p][lab][c][str(n)]['R@1']:10.4f}" for n in sizes))

    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    tag = "init0" if not trained else (Path(args.state).parent.name
                                       or Path(args.state).stem)
    out = LOOK / f"diag_u09_gallery_size_{tag}.json"
    LOOK.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": "U-09 gallery-size sensitivity diagnostic.",
        "NOT_A_PAPER_REPRODUCTION": (
            "The paper's candidate pool is UNKNOWN -- that is what U-09 is. No "
            "number here may be set against Table 1."),
        "controls": {
            "queries": f"the same {nq} dev_val assets at every size",
            "nesting": ("gallery N is the first N of one fixed index whose "
                        "prefix is dev_val, so each is a strict subset of the "
                        "next; asserted"),
            "identical_vectors": ("the gallery is encoded ONCE over the full "
                                  "index and each size is a COLUMN SLICE of "
                                  "that array, so a shared UID's vector is the "
                                  "same bytes, not merely equal"),
            "only_variable": "added distractors",
        },
        "protocols": {
            "PROD": "query and gallery are the same cached vectors",
            "BARE": ("bare canonical description in the gallery, bare "
                     "highest-ranked non-canonical description in the query, "
                     "no template on either side; image and pc are the "
                     "independent observation"),
        },
        "bare_path_bypasses_the_token_gate": {
            "gate": "encode_text_image.refuse_if_overlong, 77-token CLIP context",
            "strings_over_77_tokens_silently_truncated": over,
            "note": "kept: dropping them would change the gallery size",
        },
        "duplicate_bare_strings": dup,
        "assertions": {"heldout_view_perturbation_residual": a_img},
        "sizes": sizes, "n_query": nq, "batch_size": bs,
        "distractor_seed": args.distractor_seed,
        "gallery_uid_order_sha256": hashlib.sha256(
            "\x00".join(order).encode()).hexdigest(),
        "bare_caption_cache": str(cache),
        "bare_caption_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "query_pc_array": man["array"],
        "query_pc_array_sha256": man["array_sha256"],
        "query_pc_seed_offset": man["seed_offset"],
        "encoder": str(backbone.cfg.checkpoint),
        "encoder_sha256": hashlib.sha256(
            backbone.cfg.checkpoint.read_bytes()).hexdigest(),
        "state": "trained" if trained else "INIT-0 (zero optimizer steps)",
        "checkpoint": args.state if trained else None,
        "checkpoint_sha256": ckpt_sha,
        "seeds": None if trained else seeds,
        "debug_limit_gallery": args.limit_gallery,
        "code_revision": rev, "code_dirty": dirty,
        "results": results,
    }, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
