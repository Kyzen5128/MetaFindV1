#!/usr/bin/env python3
"""The arm the image experiment could not reach: a DIFFERENT TEXT observation.

[KYZEN 2026-09-03] "列出所有可能性" and test them, not guess.

`exp_query_observation.py` varied only the image, so its text column was
constant across all four rows -- and it was constant for a reason worth stating
plainly: under `same_record` the query's text vector is the SAME BYTES as the
gallery's. Text-only retrieval there is asking "which gallery row has my exact
vector", which no observation policy over images can perturb. That is why text
never moved, and why the experiment could not speak to the text axis at all.

Every annotation carries `description_candidates`: five independently sampled
descriptions of the same asset, ranked by CLIP score, of which rank 0 is the
canonical one already in the corpus. Ranks 1..4 are genuine second observations
that exist on disk and have never been encoded.

So this encodes them through the SAME frozen text tower the corpus was built
with, and runs text-only and text+image with:

    query text   = an alternate caption (rank 1 by default)
    gallery text = the canonical one, unchanged

That is a real second observation of the same asset: a different sentence,
written by the same annotator about the same renders, describing the same thing.
It is the closest this corpus comes to the construction the paper's notation
implies -- `f_query(Q)` against `f_gallery(A)` with Q not derived from A's
stored record.

[SENSITIVITY, NOT A TRAINED ARM] One checkpoint evaluated two ways. §十六 is
explicit that a training-time observation change needs a retrain before the two
are comparable as protocols. What this can settle is whether the text column's
height is a property of the observation or of the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.models.resolve_stage1 import serialize_annotation


def alternate_text(uid: str, rank: int) -> tuple[str, bool]:
    """The asset's rank-N caption, serialized through the same template.

    Returns (text, is_alternate). An asset with no candidate at that rank falls
    back to its canonical text and says so, because dropping it would change
    the pool and make the two rows incomparable -- 111 assets have fewer than
    five candidates and 20 have exactly one.
    """
    a = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
    cands = a.get("description_candidates") or []
    if rank < len(cands) and cands[rank].get("text"):
        return serialize_annotation({**a, "description": cands[rank]["text"]}), True
    return serialize_annotation(a), False


def encode_texts(texts: list[str], device: str, batch: int = 64) -> np.ndarray:
    """Through the SAME frozen tower n06 used. Not a re-implementation."""
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    bb = ULIPBackbone(BackboneConfig(device=device, train_scope="fuser_only"))
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            out.append(bb.encode_text(texts[i:i + batch]).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev_val")
    ap.add_argument("--gallery-split", default=None)
    ap.add_argument("--rank", type=int, default=1, help="which alternate caption")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="output/look/exp_text_observation.json")
    args = ap.parse_args()

    from tools.probes.exp_query_observation import (
        gallery_vectors,
        load_tower,
        recall,
    )

    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    uids = sorted(sp[args.split])[: args.limit] if args.limit else sorted(sp[args.split])
    g_uids = (sorted(set(sp["train"]) | set(sp["test"]))
              if args.gallery_split == "full" else uids)
    where = {u: i for i, u in enumerate(g_uids)}
    target_row = torch.tensor([where[u] for u in uids]).to(args.device)

    texts, is_alt = [], []
    for uid in uids:
        t, alt = alternate_text(uid, args.rank)
        texts.append(t)
        is_alt.append(alt)
    n_alt = sum(is_alt)
    print(f"{n_alt:,} of {len(uids):,} assets have a rank-{args.rank} caption; "
          f"the rest keep their canonical text and are counted as such")

    print("encoding the alternate captions through the frozen tower...", flush=True)
    q_text = torch.from_numpy(encode_texts(texts, args.device)).to(args.device)

    model = load_tower(Path(args.ckpt), args.device)
    gal = gallery_vectors(g_uids, args.device)
    # The query's own canonical text and image, straight from the cache.
    canon_text, q_img = [], []
    for uid in uids:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        canon_text.append(z["text"].astype(np.float32))
        q_img.append(z["image"].astype(np.float32))
    canon_text = torch.from_numpy(np.stack(canon_text)).to(args.device)
    q_img = torch.from_numpy(np.stack(q_img)).to(args.device)

    print(f"\n{len(uids):,} queries against {len(g_uids):,} gallery items. "
          "The GALLERY keeps its canonical text in both rows.\n")
    print("query text".ljust(22) + "text".rjust(15) + "text+image".rjust(15))
    out = {}
    for name, t in (("canonical (same bytes)", canon_text),
                    (f"alternate rank {args.rank}", q_text)):
        row, cells = name.ljust(22), {}
        for cond, present in (("text", (True, False)), ("text+image", (True, True))):
            r1, r5 = recall(model, t, q_img, gal, present, args.device, target_row)
            cells[cond] = {"R@1": r1, "R@5": r5}
            row += ("%.4f / %.4f" % (r1, r5)).rjust(15)
        print(row)
        out[name] = cells

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "checkpoint": args.ckpt, "split": args.split,
        "gallery_split": args.gallery_split or args.split,
        "n": len(uids), "n_gallery": len(g_uids),
        "rank": args.rank, "n_with_alternate": n_alt,
        "caveat": "inference sensitivity on ONE checkpoint trained under "
                  "same_record. A training-time observation change needs a "
                  "retrain before the two are comparable as protocols (§十六).",
        "results": out}, indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
