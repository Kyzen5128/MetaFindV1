#!/usr/bin/env python3
"""Does the query's observation change Table 1's shape? Measure it.

[KYZEN 2026-09-03] He asked for experiments on the unresolved settings, not for
another report saying they are unresolved. 問題 4 and 問題 6 are exactly that:
the paper never states what the query observes, and `metafind/data/observation.py`
was built to express the alternatives and then never run.

This runs it. One checkpoint, one gallery, one set of queries; the ONLY thing
that varies is which image the query draws:

    same_mean       the 12-view mean, i.e. the query sees the gallery's own
                    image vector -- what every number so far was measured under
    single_view     one view, chosen by uid_seed
    held_out_view   the same single view, named for what it is on the query side
    disjoint_views  the mean of the other eleven, so query and gallery share no
                    view at all

`disjoint_views` is the interesting one: it is the closest thing this corpus can
express to "the query is a different observation of the same asset", without
re-rendering anything and without a query pack.

NO BACKBONE IS LOADED. Text and image are frozen, so their vectors come from
n06's cache -- exactly what the towers saw. That is what lets this run on CPU
beside a training job, and it is also why the point-cloud conditions are absent:
PointBERT is trainable, its output is not cached, and encoding 4,569 clouds
through it is not free. The four conditions here are the ones the image
observation actually moves.

[SENSITIVITY, NOT A PROTOCOL COMPARISON] One checkpoint evaluated four ways is
inference sensitivity. §十六 is explicit that it cannot stand in for training
under a different observation -- that needs a retrain. What it CAN say is
whether the saturation is a property of the observation or of the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.observation import view_indices

CONDITIONS = {"text": ("text",), "image": ("image",),
              "text+image": ("text", "image")}
POLICIES = ("same_mean", "single_view", "held_out_view", "disjoint_views")


def load_tower(ckpt_path: Path, device: str):
    """The two fusion heads from a Stage 1 checkpoint. No backbone."""
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyperparameters = load_protocols()
    model, _loss = build_model(encoding, training, hyperparameters)
    model = model.to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["tower_trainable_state"],
                                                strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    gap = trainable - set(ck["tower_trainable_state"])
    if gap:
        raise SystemExit(f"the checkpoint does not cover {sorted(gap)[:4]}")
    if unexpected:
        raise SystemExit(f"unexpected keys in the tower state: {unexpected[:4]}")
    model.eval()
    return model


def gallery_vectors(uids: list[str], device: str):
    """The gallery bank. Unchanged by every policy -- only the QUERY varies."""
    text, img = [], []
    for uid in uids:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        text.append(z["text"].astype(np.float32))
        img.append(z["image"].astype(np.float32))
    return (torch.from_numpy(np.stack(text)).to(device),
            torch.from_numpy(np.stack(img)).to(device))


def vectors(uids: list[str], policy: str, device: str):
    """Query and gallery vectors for one image policy. Cache only."""
    text, q_img, g_img = [], [], []
    for uid in uids:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        text.append(z["text"].astype(np.float32))
        g_img.append(z["image"].astype(np.float32))   # the gallery is unchanged
        views = z["views"].astype(np.float32)
        if policy == "same_mean":
            # NOT views.mean(): the stored `image` is the fp16 mean n06 wrote,
            # and recomputing it in fp32 differs in the last bits from every
            # number measured so far. Same bytes or it is a different arm.
            q_img.append(z["image"].astype(np.float32))
        else:
            idx = view_indices(policy, uid, views.shape[0])
            q_img.append(views[idx].mean(axis=0))
    t = torch.from_numpy(np.stack(text)).to(device)
    return t, torch.from_numpy(np.stack(q_img)).to(device), \
        torch.from_numpy(np.stack(g_img)).to(device)


def recall(model, text, q_img, gal, present, device, target_row, batch=256):
    """R@1 / R@5 of each query against the whole gallery.

    `target_row[i]` is where query i's own asset sits in the gallery, so the
    gallery may be LARGER than the query pool -- which is the point of the
    full-corpus arm. Reading the diagonal instead would silently score against
    the wrong asset the moment the two pools differ.
    """
    g_text, g_img = gal
    n = text.size(0)
    with torch.no_grad():
        q = model.query({"text": text if present[0] else None,
                         "image": q_img if present[1] else None,
                         "pc": None},
                        present=torch.tensor([[present[0], present[1], False]]
                                             ).repeat(n, 1).to(device))
        q = torch.nn.functional.normalize(q.float(), dim=-1)
        gs = []
        for i in range(0, g_text.size(0), 4096):
            gs.append(model.gallery({"text": g_text[i:i + 4096],
                                     "image": g_img[i:i + 4096],
                                     "pc": torch.zeros_like(g_text[i:i + 4096])}))
        g = torch.nn.functional.normalize(torch.cat(gs).float(), dim=-1)
        r1 = r5 = 0
        for i in range(0, n, batch):
            sims = q[i:i + batch] @ g.T
            rows = torch.arange(sims.size(0))
            tgt = sims[rows, target_row[i:i + batch]]
            rank = (sims > tgt.unsqueeze(1)).sum(dim=1) + 1
            r1 += int((rank <= 1).sum())
            r5 += int((rank <= 5).sum())
    return r1 / n, r5 / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--gallery-split", default=None,
                    help="default: the same pool as the queries. `full` is the "
                         "whole admitted corpus, which is the other axis §十六 "
                         "asks about -- ten times the distractors, same queries.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="output/look/exp_query_observation.json")
    args = ap.parse_args()

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])[: args.limit] if args.limit else sorted(sp[args.split])
    if args.gallery_split == "full":
        g_uids = sorted(set(sp["train"]) | set(sp["test"]))
    elif args.gallery_split:
        g_uids = sorted(sp[args.gallery_split])
    else:
        g_uids = uids
    where = {u: i for i, u in enumerate(g_uids)}
    missing = [u for u in uids if u not in where]
    if missing:
        raise SystemExit(f"{len(missing)} query assets are not in the gallery, "
                         f"e.g. {missing[:3]} -- every query needs its positive")
    target_row = torch.tensor([where[u] for u in uids]).to(args.device)
    model = load_tower(Path(args.ckpt), args.device)
    gal = gallery_vectors(g_uids, args.device)

    print(f"checkpoint {args.ckpt}")
    print(f"{len(uids):,} queries against {len(g_uids):,} gallery items, "
          f"image policy varied, GALLERY UNCHANGED\n")
    head = "policy".ljust(16) + "".join(c.rjust(14) for c in CONDITIONS)
    print(head)
    out = {}
    for policy in POLICIES:
        text, q_img, _ = vectors(uids, policy, args.device)
        row, cells = policy.ljust(16), {}
        for name, mods in CONDITIONS.items():
            r1, r5 = recall(model, text, q_img, gal,
                            ("text" in mods, "image" in mods), args.device,
                            target_row)
            cells[name] = {"R@1": r1, "R@5": r5}
            row += ("%.4f / %.4f" % (r1, r5)).rjust(14)
        print(row)
        out[policy] = cells

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt, "split": args.split, "n": len(uids),
        "gallery_split": args.gallery_split or args.split,
        "n_gallery": len(g_uids),
        "gallery": "unchanged (12-view mean) in every row; only the QUERY varies",
        "caveat": "inference sensitivity on ONE checkpoint. It cannot stand in "
                  "for training under a different observation (§十六).",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
