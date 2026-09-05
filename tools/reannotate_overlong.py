#!/usr/bin/env python
"""Prepare the PROMPT_VERSION 9 re-annotation of the assets CLIP cannot read.

# SUPPORTS-NODE: n05_annotate

[Kyzen 2026-08-28] Option D-small. The 2,095 assets whose v8 serialized string
exceeds CLIP's 77-token context are re-annotated under v9, which asks for ONE
sentence of at most `MAX_DESCRIPTION_WORDS` words. The alternatives were
rejected by the USER after measurement:

  A  let n06 quarantine them            corpus 45,692 -> 43,597, and NOT
                                        uniformly: 82.58% of four-placement-flag
                                        assets against 1.52% of single-flag ones
  B  truncate the description tail       keeps the corpus but keeps the
                                        mid-sentence tail ("It features a.")

This script does the three things that must happen BEFORE the model runs, and
nothing that needs a GPU:

  1. recompute the population from the corpus, so the list is never stale
  2. move the v8 records aside -- moved, not deleted; they are the only
     evidence of what the corpus said before
  3. write the ledger, the same shape as `annotation_exclusions.json`

It then PRINTS the annotate command. It does not run it: the run is ~3.7 GPU
hours and starting one is the USER's call, not this script's.

    python tools/reannotate_overlong.py            # list and count only
    python tools/reannotate_overlong.py --prepare  # also move + write the ledger
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402
from metafind.data.annotate import (  # noqa: E402
    MAX_DESCRIPTION_WORDS,
    PROMPT_VERSION,
    annotation_contract_id,
)
from metafind.data.encode_text_image import (  # noqa: E402
    TEXT_CONTEXT_LENGTH,
    true_token_count,
)
from metafind.models.resolve_stage1 import serialize_annotation  # noqa: E402

SUPERSEDED = Path("/home/kyzen/metafind/metafind_out/annotations_superseded_v8")
LEDGER = paths.OUTPUTS / "reannotation_v9.json"
UID_LIST = paths.OUTPUTS / "reannotate_v9_uids.txt"


def overlong() -> list[tuple[str, int, str]]:
    """(uid, tokens, serialized) for every record CLIP cannot read whole.

    Recomputed from the corpus every run rather than read from a stored list: a
    list on disk is a claim about a corpus that may have moved since, and this
    one decides what gets overwritten.
    """
    out = []
    for path in sorted(paths.ANNOTATIONS.glob("*.json")):
        ann = json.loads(path.read_text())
        try:
            text = serialize_annotation(ann)
        except (KeyError, ValueError):
            continue          # unserializable is n06's quarantine, not this
        n = true_token_count(text)
        if n > TEXT_CONTEXT_LENGTH:
            out.append((path.stem, n, text))
    return out


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true",
                    help="move the v8 records aside and write the ledger")
    args = ap.parse_args()

    rows = overlong()
    total = len(list(paths.ANNOTATIONS.glob("*.json")))
    if not rows:
        print(f"{total:,} records, none over {TEXT_CONTEXT_LENGTH} tokens. "
              "Nothing to re-annotate.")
        return 0

    counts: dict[int, int] = {}
    for _, n, _ in rows:
        counts[n] = counts.get(n, 0) + 1
    print(f"corpus                 {total:,}")
    print(f"over {TEXT_CONTEXT_LENGTH} true BPE tokens  {len(rows):,}"
          f"   ({len(rows) / total * 100:.2f}%)")
    print(f"  worst              {max(n for _, n, _ in rows)} tokens")
    print(f"  exactly +1         {counts.get(TEXT_CONTEXT_LENGTH + 1, 0):,}")
    print(f"contract now           {annotation_contract_id()}")
    print(f"asking for             ONE sentence, <= {MAX_DESCRIPTION_WORDS} words "
          f"(PROMPT_VERSION {PROMPT_VERSION})")

    if not args.prepare:
        print("\n--prepare would move these aside and write the ledger. "
              "Nothing was written.")
        return 0

    UID_LIST.write_text("\n".join(uid for uid, _, _ in rows) + "\n")
    SUPERSEDED.mkdir(parents=True, exist_ok=True)
    moved = []
    for uid, n, text in rows:
        src = paths.ANNOTATIONS / f"{uid}.json"
        dst = SUPERSEDED / f"{uid}.json"
        if dst.exists():
            print(f"refusing: {dst} already exists -- an earlier prepare would "
                  "be overwritten, and it is the only copy of that record")
            return 3
        shutil.copy2(src, dst)     # copy, not move: `--force` overwrites in place
        moved.append({"uid": uid, "v8_tokens": n, "v8_text": text})

    LEDGER.write_text(json.dumps({
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "decided_by": "Kyzen",
        "decision": "我只想跑D 小 / 原標註結果 + D 小重跑結果",
        "option": "D-small: re-annotate only the overlong assets under v9",
        "rejected": {
            "A": "let n06 quarantine them; corpus 45,692 -> 43,597",
            "B": "truncate the description tail; keeps the mid-sentence cut",
        },
        "git_commit": git_commit(),
        "corpus": total,
        "reannotated": len(rows),
        "context_length": TEXT_CONTEXT_LENGTH,
        "max_description_words": MAX_DESCRIPTION_WORDS,
        "contract_before": "metafind_annot_v8@95e37eb05182d364",
        "contract_after": annotation_contract_id(),
        "superseded_records": str(SUPERSEDED),
        "uid_list": str(UID_LIST),
        "records": moved,
    }, ensure_ascii=False, indent=1))

    print(f"\ncopied {len(moved):,} v8 records -> {SUPERSEDED}")
    print(f"ledger -> {LEDGER}")
    print(f"uid list -> {UID_LIST}")
    print("\nthe run, which this script deliberately does NOT start:\n")
    print(f"  {sys.executable} -m metafind.data.annotate_run \\")
    print(f"      --uids-file {UID_LIST} --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
