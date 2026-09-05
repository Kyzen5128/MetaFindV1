#!/usr/bin/env python3
"""Pull the QUERY-side observations ULIP-2 / OpenShape ship per Objaverse object out of the
ULIP-2 HF shards Kyzen downloaded on 2026-09-01 (/mnt/data1/kyzen/datasets/ulip2_objaverse_lvis).

Each <uid>.npy (OpenShape format) holds, in the same OpenCLIP ViT-bigG-14 space as our cache:
  thumbnail_feat (1280,)        CLIP feature of the Sketchfab THUMBNAIL  -> a target image that is not our render
  text / text_feat              the Sketchfab NAME (+ 'original' and 'prompt_avg' features)
  blip_caption(_feat), msft_caption(_feat)   short captions of the thumbnail/renders
  retrieval_text(_feat)         LAION captions retrieved for the renders
  image_feat (12, 1280)         their 12 renders; xyz/rgb (10k)  -- not extracted (we have our own)

Writes ONE npz per field set to <OUT>/ulip2_query_feats.npz keyed by uid, for our 45,692 assets only.
Read-only on the shards; nothing under /mnt/data1 is modified.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SHARDS = Path("/mnt/data1/kyzen/datasets/ulip2_objaverse_lvis/ULIP-2/objaverse_lvis")
META = Path("/home/kyzen/upstream/openshape-objaverse-embeddings/objaverse_meta.json")
OUT = Path("/home/kyzen/metafind_data/outputs/_probe/ulip2_query_feats")


def pull(shard: str, wanted: set[str]) -> dict:
    got = {}
    with tarfile.open(SHARDS / f"{shard}.tar.gz", "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            uid = Path(m.name).stem
            if uid not in wanted:
                continue
            d = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True).item()
            got[uid] = {
                "thumbnail_feat": np.asarray(d["thumbnail_feat"], np.float16),
                "name": d["text"][0] if d.get("text") else "",
                "name_feat": np.asarray(d["text_feat"][0]["original"], np.float16).reshape(-1) if d.get("text_feat") else None,
                "name_feat_prompt_avg": np.asarray(d["text_feat"][0]["prompt_avg"], np.float16).reshape(-1) if d.get("text_feat") else None,
                "blip_caption": d.get("blip_caption", ""),
                "blip_feat": np.asarray(d["blip_caption_feat"]["original"], np.float16).reshape(-1) if d.get("blip_caption") else None,
                "msft_caption": d.get("msft_caption", ""),
                "msft_feat": np.asarray(d["msft_caption_feat"]["original"], np.float16).reshape(-1) if d.get("msft_caption") else None,
                "retrieval_text": list(d.get("retrieval_text") or []),
                "retrieval_feat0": (np.asarray(d["retrieval_text_feat"][0]["original"], np.float16).reshape(-1)
                                    if d.get("retrieval_text") else None),
            }
            if len(got) == len(wanted):
                break
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--splits", default="/home/kyzen/metafind_data/outputs/splits.json")
    args = ap.parse_args()
    sp = json.loads(Path(args.splits).read_text())["object"]
    from metafind.data.splits import corpus_uids
    uids = sorted(corpus_uids(sp))          # [D-3b] train + val + test
    meta = {e["u"]: e for e in json.loads(META.read_text())["entries"]}
    by_shard: dict[str, set] = {}
    for u in uids:
        shard = meta[u]["glb"].split("/")[1]            # "glbs/000-017/<uid>.glb"
        by_shard.setdefault(shard, set()).add(u)
    print(f"{len(uids):,} assets over {len(by_shard)} shards", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    allgot: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(pull, s, w): s for s, w in sorted(by_shard.items())}
        for i, f in enumerate(as_completed(futs), 1):
            got = f.result(); allgot.update(got)
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} shards, {len(allgot):,} assets", flush=True)
    missing = [u for u in uids if u not in allgot]
    print(f"missing {len(missing)} (e.g. {missing[:3]})")
    order = [u for u in uids if u in allgot]
    D = 1280
    def stack(key):
        return np.stack([allgot[u][key] if allgot[u][key] is not None else np.zeros(D, np.float16) for u in order])
    has = lambda key: np.array([allgot[u][key] is not None for u in order])
    np.savez(OUT / "ulip2_query_feats.npz", uids=np.array(order),
             thumbnail_feat=stack("thumbnail_feat"),
             name_feat=stack("name_feat"), has_name=has("name_feat"),
             name_feat_prompt_avg=stack("name_feat_prompt_avg"),
             blip_feat=stack("blip_feat"), has_blip=has("blip_feat"),
             msft_feat=stack("msft_feat"), has_msft=has("msft_feat"),
             retrieval_feat0=stack("retrieval_feat0"), has_retrieval=has("retrieval_feat0"))
    (OUT / "ulip2_query_texts.json").write_text(json.dumps(
        {u: {"name": allgot[u]["name"], "blip": allgot[u]["blip_caption"], "msft": allgot[u]["msft_caption"],
             "retrieval_text": allgot[u]["retrieval_text"][:5]} for u in order}, ensure_ascii=False))
    print(f"-> {OUT} ({len(order):,} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
