#!/usr/bin/env python3
"""One-sentence gemma description per ProcTHOR asset from its unified renders (DL-103 R6b).

DL-077 Q10 (Kyzen 甲): gemma captions from the unified-protocol renders. Since 2026-09-06
the PRIMARY node / gallery text comes from ProcTHOR's own metadata
(`tools/procthor_metadata_text.py`); this adds the Figure 2 `description` field, the one
thing the metadata lacks, so a ProcTHOR record has the same shape as an Objaverse one.

Prompt: the asset's category (from the metadata, shown as the identity) and its 11 views;
the model returns ONE English sentence about what it sees. Same admission rules as
annotate_v10's description (English, one sentence, bounded length). Output:
`outputs/procthor_captions.json` {assetId: sentence}; feed it to
`procthor_metadata_text.py --captions`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from metafind import paths
from metafind.data.annotate import non_english_characters
from metafind.data.annotate_v10 import DESCRIPTION_WORDS_HINT, MAX_DESCRIPTION_CHARS, MIN_DESCRIPTION_CHARS

PROMPT_VERSION = 1


def build_prompt(n_views: int, category: str) -> str:
    return (
        f"You are looking at {n_views} rendered views of a single 3D asset, taken from evenly "
        f"spaced directions around it. It is a {category}.\n"
        "\n"
        f"Write ONE English sentence of at most about {DESCRIPTION_WORDS_HINT} words describing what "
        "you see: its shape, parts, colours, surface, and any text or marking. Reply with the "
        "sentence alone -- no preamble, no quotes, no list."
    )


def clean(text: str) -> str | None:
    t = " ".join((text or "").strip().strip('"').split())
    if len(t) < MIN_DESCRIPTION_CHARS or len(t) > MAX_DESCRIPTION_CHARS or non_english_characters(t):
        return None
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from metafind.data.annotate_run import Annotator

    out_path = args.out or (paths.OUTPUTS / "procthor_captions.json")
    recs = json.loads((paths.OUTPUTS / "procthor_asset_annotations.json").read_text())
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [a for a in sorted(recs) if a not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(recs):,} assets, {len(done):,} captioned, {len(todo):,} to do", flush=True)
    if not todo:
        return 0
    ann = Annotator()
    failed, started = [], time.time()
    for i, aid in enumerate(todo, 1):
        sc = paths.PROCTHOR_MODALITIES / f"{aid}.json"
        if not sc.exists():
            failed.append((aid, "no modality sidecar")); continue
        rec = json.loads(sc.read_text())
        views = rec["view_paths"]
        cat = recs[aid]["category"]
        text = None
        for attempt in range(2):
            raw = ann.generate(views, build_prompt(len(views), cat), sample=attempt == 1, seed=1000 + i)
            text = clean(raw if isinstance(raw, str) else raw[0])
            if text:
                break
        if text:
            done[aid] = text
        else:
            failed.append((aid, "no admissible sentence"))
        if i % 50 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            rate = i / (time.time() - started) * 60
            print(f"  [{i:>5}/{len(todo)}] {rate:.0f}/min, failed {len(failed)}", flush=True)
    out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    print(f"{len(done):,} captions -> {out_path}; failed {len(failed)} e.g. {failed[:3]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
