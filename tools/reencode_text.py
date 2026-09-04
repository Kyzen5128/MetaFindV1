#!/usr/bin/env python3
"""Re-encode the TEXT of an existing embedding cache under the active template.

[KYZEN 2026-09-03] "文字都先採用固定填表的方式" / "全程交給你處理".

The n06 cache holds, per asset, `text (1280,)`, `views (12,1280)` and
`image (1280,)`. Only `text` depends on the serialization template; the view
vectors come from the frozen image tower and the renders, neither of which
changes. Re-running n06 would push 45,692 x 12 images through ViT-bigG again
for nothing, so this copies `views` / `image` and re-encodes `text` only.

It writes into `paths.EMBEDDINGS` of the CURRENT data root. Run it under
`METAFIND_DATA=<overlay>` so the original cache is never touched -- the two
caches are different corpora as far as `check_embedding_sidecars` is concerned,
and it refuses a mismatch by `text_serialization`, which is content-addressed.

The overlay's `stage1_encoding_protocol.json` must already be resolved under
the same `METAFIND_TEXT_TEMPLATE`; `load_protocol()` checks that the live
serializer emits what the protocol recorded, and this tool goes through it.

Sidecars are the source sidecar with the text fields replaced, so every other
provenance field (image identity, renderer version, checkpoint sha) travels
unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from metafind import paths
from metafind.data.encode_text_image import (Encoder, load_protocol,
                                             refuse_if_overlong, sidecar_path)
from metafind.models.resolve_stage1 import (TEXT_TEMPLATE_NAME,
                                            serialize_annotation)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="the embeddings directory to copy views/image from")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--truncate", action="store_true",
                    help="[KYZEN 2026-09-04, figure2_json] encode over-length text "
                         "anyway (CLIP keeps the first 77 tokens) and record "
                         "text_truncated=True with the true count, instead of "
                         "refusing (P-4). The poster's gallery text is the whole "
                         "annotation JSON, which is what the paper's CLIP saw.")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    dst = paths.EMBEDDINGS.resolve()
    if src == dst:
        raise SystemExit(f"source and destination are the same directory "
                         f"({src}); run under METAFIND_DATA=<overlay>")
    protocol = load_protocol()          # refuses if the template drifted
    print(f"template {TEXT_TEMPLATE_NAME}  ->  "
          f"{protocol['text_serialization']}\n  from {src}\n  to   {dst}",
          flush=True)
    dst.mkdir(parents=True, exist_ok=True)

    uids = sorted(p.stem for p in src.glob("*.json"))
    if args.limit:
        uids = uids[: args.limit]
    todo = [u for u in uids if not sidecar_path(u).exists()]
    print(f"{len(uids):,} in source, {len(todo):,} to write", flush=True)
    if not todo:
        return 0

    enc = Encoder(device=args.device)
    started, done, overlong = time.time(), 0, 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        texts, keep = [], []
        for u in chunk:
            ann = json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())
            t = serialize_annotation(ann)
            try:
                n_tok = refuse_if_overlong(t)
            except ValueError:
                if not args.truncate:
                    overlong += 1
                    continue
                from metafind.data.encode_text_image import true_token_count
                n_tok = true_token_count(t); overlong += 1
            texts.append((u, t, n_tok)); keep.append(u)
        if not texts:
            continue
        with enc.torch.no_grad():
            vecs = enc.backbone.encode_text([t for _, t, _ in texts]).float().cpu().numpy()
        for (u, t, n_tok), v in zip(texts, vecs):
            old = np.load(src / f"{u}.npz")
            rec = json.loads((src / f"{u}.json").read_text())
            npz = dst / f"{u}.npz"
            tmp = dst / f"{u}.part.npz"
            np.savez_compressed(tmp, text=v.astype(np.float16),
                                views=old["views"], image=old["image"])
            tmp.replace(npz)
            rec.update({"embedding_uri": str(npz), "text": t,
                        "text_tokens": int(n_tok), "text_truncated": bool(n_tok > 77),
                        "text_serialization": protocol["text_serialization"],
                        "reencoded_text_from": str(src)})
            sc = sidecar_path(u)
            tmp = sc.with_suffix(".json.part")
            with tmp.open("w") as fh:
                json.dump(rec, fh, ensure_ascii=False); fh.flush(); os.fsync(fh.fileno())
            tmp.replace(sc)
            done += 1
        if done and done % 2000 < args.batch:
            rate = done / max(time.time() - started, 1e-9) * 60
            print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, over-length {overlong}",
                  flush=True)
    print(f"done {done:,}, over-length refused {overlong}, "
          f"{time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
