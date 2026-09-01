#!/usr/bin/env python3
"""How long is the string upstream actually encodes, and how long is ours?

Kyzen, 2026-09-01: 「你不是有載下來官方的嗎? 你看他怎麼做的啊」

`fig2_text_form` established that both JSON arms are cut for all 9,138 queries
and that the paper's 13.8 sits between a truncated JSON (8.09) and our prose
(19.88). It could not say what a text of the right length looks like, because
every string it had was ours.

ULIP-2's published shards carry the RAW STRINGS, not only their embeddings:

    text            ['Desktop Screen Low Poly']              the Objaverse name
    blip_caption    'a white sign on a stand'                BLIP on the thumbnail
    msft_caption    'a white computer monitor with a ...'    Azure on the thumbnail
    retrieval_text  13 strings                               LAION-5B neighbours

So the question is directly measurable: on the assets both sides have, how many
tokens does each upstream source spend, how many does ours spend, and how often
does each hit CLIP's 77.

This is a length measurement, not a retrieval one. `query_gallery_text_split`
already scored these same sources; what was never measured is the property that
`fig2_text_form` showed to be the mechanism -- whether the string fits.

Both `original` and `prompt_avg` embeddings ship with each source. Only the
strings are read here; nothing is encoded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402
from metafind.models import resolve_stage1 as R  # noqa: E402

OUT = REPO / "output" / "look" / "upstream_text_length.json"
ANN = paths.OUTPUTS / "annotations"
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shards")
LIMIT = 77


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=SHARDS)
    args = ap.parse_args()

    theirs = {os.path.basename(f)[:-4]: f
              for f in glob.glob(os.path.join(args.shards, "*", "*.npy"))}
    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = set(split["train"]) | set(split["test"])
    keep = sorted(u for u in theirs if u in corpus)
    print(f"overlap {len(keep):,}", flush=True)

    import open_clip
    tok = open_clip.get_tokenizer("ViT-bigG-14")

    src: dict[str, list[str]] = {k: [] for k in
                                 ("name", "blip", "msft", "retrieval",
                                  "ours_prose", "ours_description")}
    for u in keep:
        d = np.load(theirs[u], allow_pickle=True).item()
        src["name"].append(d["text"][0] if len(d["text"]) else "")
        src["blip"].append(d.get("blip_caption") or "")
        src["msft"].append(d.get("msft_caption") or "")
        rt = d.get("retrieval_text") or []
        src["retrieval"].append(rt[0] if len(rt) else "")
        a = json.loads((ANN / f"{u}.json").read_text())
        src["ours_prose"].append(R.serialize_annotation(a))
        src["ours_description"].append(a["description"])

    res = {"n": len(keep), "clip_limit": LIMIT, "sources": {}}
    print(f"\n{'來源':<20s}{'字元中位':>10s}{'token 中位':>11s}"
          f"{'token 平均':>11s}{'最長':>7s}{'滿 77':>8s}")
    for name, strings in src.items():
        n_tok = np.array([int((tok([s])[0] != 0).sum()) for s in strings])
        n_ch = np.array([len(s) for s in strings])
        sat = float((n_tok >= LIMIT).mean()) * 100
        res["sources"][name] = {
            "chars_median": int(np.median(n_ch)),
            "tokens_median": int(np.median(n_tok)),
            "tokens_mean": round(float(n_tok.mean()), 1),
            "tokens_max": int(n_tok.max()),
            "pct_at_limit": round(sat, 2),
            "examples": strings[:3],
        }
        print(f"{name:<20s}{np.median(n_ch):10.0f}{np.median(n_tok):11.0f}"
              f"{n_tok.mean():11.1f}{n_tok.max():7d}{sat:7.1f}%")

    print("\n各來源前三筆：")
    for name in src:
        print(f"\n--- {name} ---")
        for s in res["sources"][name]["examples"]:
            print(f"  {s[:150]!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
