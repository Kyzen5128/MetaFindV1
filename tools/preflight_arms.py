#!/usr/bin/env python
"""Load each bake-off arm and make it emit one token. On-disk is not loadable.

# SUPPORTS-NODE: n05_annotate_assets

`U-R` recorded that `gemma-4-31B-it-qat-w4a16` was listed READY while
`compressed_tensors` was absent, so the arm could not load at all. This is the
check that would have caught it, and it runs before the bake-off rather than
during it: an arm that dies on load at asset 40 has already spent forty assets
of the other arms' comparability.

Each arm is loaded EXACTLY as `annotate_run.Annotator` loads it, imports and
kwargs included, so a pass here is a statement about the runner and not about
this file.

    python tools/preflight_arms.py                 # all three
    python tools/preflight_arms.py --arm qwen38_27b

The 5090 has 32 GB and `Qwen3.8-27B` is 56 GB of bf16, so it cannot be loaded
the way the other two are. It is loaded 4-bit. **That makes this a comparison
of what fits on this card, not of the models at equal precision**, and no
result may be written up as "model A is better than model B".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# uid -> the arm definition. `quant` is the ONLY thing that differs between
# arms besides the checkpoint, and it is recorded per arm so the write-up
# cannot lose it.
ARMS = {
    "qwen38_27b": {
        "model_id": "/mnt/data1/kyzen/models/Qwen3.8-27B",   # DELETED 2026-09-05 (Kyzen); this arm can no longer run
        "quant": "bnb-nf4",
        "why": "56 GB of bf16 against a 32 GB card -- 4-bit is what makes it loadable at all",
    },
    # RULED OUT on this card, 2026-08-22, USER decision to proceed with two arms.
    # Five attempts, each aimed at a distinct MEASURED cause, each improving and
    # none sufficient:
    #
    #   1  dtype=bfloat16                    27.56 GB resident, OOM
    #   2  dtype="auto"                      23.02 GB resident, OOM on image 1
    #   3  max_memory {0: 24GiB, cpu: 40}    ignored; 23.02 GB, OOM on image 1
    #   4  run_compressed=True               "Decompressing model" STILL logged;
    #                                        OOM 442 MiB short, 3.23 GB fragmented
    #   5  expandable_segments:True          fragmentation 3.23 -> 1.65 GB,
    #                                        allocated 28.95 GB, OOM 112 MiB short
    #
    # The checkpoint's `ignore` list exempts the whole vision tower, so 4-bit
    # covers the language layers only and the image encoder stays bf16: 22 GB on
    # disk arrives as ~23 GB resident, leaving ~8 GB, and one 224px view does not
    # fit in what is left. It is CLOSE -- 112 MiB -- and it is still short.
    #
    # Kept here so the next reader does not spend an hour rediscovering it.
    # Re-enable only with a bigger card or a checkpoint whose vision tower is
    # also quantised.
    "gemma4_31b_qat": {
        "model_id": "/mnt/data1/kyzen/models/gemma-4-31B-it-qat-w4a16",   # DELETED 2026-09-05 (Kyzen); this arm can no longer run
        "quant": "native-compressed-tensors",
        "why": "already w4a16 in the checkpoint -- but its vision tower is not",
        "ruled_out": "OOM on a 32 GB card; see the note above, five attempts",
    },
    "gemma4_12b": {
        "model_id": "/mnt/data1/kyzen/models/gemma-4-12B-it",
        "quant": "none",
        "why": "23 GB of bf16 fits on the card as published",
    },
}


def probe(name: str, spec: dict, image: Path | None) -> dict:
    import torch

    from metafind.data.annotate_run import Annotator

    t0 = time.time()
    out: dict = {"arm": name, **{k: v for k, v in spec.items()}}
    try:
        ann = Annotator(model_id=spec["model_id"], quant=spec["quant"])
        out["load_seconds"] = round(time.time() - t0, 1)
        out["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        # A load that succeeds and a forward pass that succeeds are different
        # claims. `U-R`'s failure was the first; an unsupported image path or a
        # chat template without an image slot is the second, and only a real
        # generation separates them.
        t1 = time.time()
        reply = ann.generate([str(image)] if image else [],
                             "Reply with exactly one word: OK")
        out["generate_seconds"] = round(time.time() - t1, 1)
        out["reply"] = reply.strip()[:120]
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- the whole point is to report, not raise
        out["ok"] = False
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:300]
    finally:
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--out", type=Path,
                    default=Path("data/outputs/bakeoff/preflight.json"))
    args = ap.parse_args()

    from metafind import paths

    # A real render, because a model that loads may still reject the image path
    # or the chat template's image slot, and a synthetic array would not
    # exercise either.
    sample = Path("workflow/blocks/ULIP2/bakeoff/sample_100.jsonl")
    image = None
    if sample.exists():
        uid = json.loads(sample.read_text().splitlines()[0])["uid"]
        cand = paths.RENDERS / uid / "view_00.png"
        image = cand if cand.exists() else None
    print(f"probe image: {image}", flush=True)

    names = ([args.arm] if args.arm
             else [k for k, v in ARMS.items() if "ruled_out" not in v])
    results = []
    for name in names:
        print(f"\n=== {name} ({ARMS[name]['quant']}) ===", flush=True)
        r = probe(name, ARMS[name], image)
        results.append(r)
        print(json.dumps(r, indent=1), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    bad = [r["arm"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} arms loadable; written {args.out}")
    if bad:
        print(f"NOT loadable: {', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
