"""Run Qwen2.5-VL over each asset's 11 views and produce a validated annotation.

# IMPLEMENTS-NODE: n05_annotate

Writes ``objaverse_annotations`` (one sidecar per asset, plus the derived
index), and ``quarantine`` / ``run_progress`` / ``cost_ledger`` via runlog.

The generating half of n05. The schema, prompt and repair-prompt construction
live in ``annotate.py``, which is deterministic and testable without a GPU;
this module is the part that needs the model, and it implements subgraph SG1:

    sg1_generate  ->  sg1_validate  ->  admit
                          |
                          +-> C1 repair loop (2 attempts) -> quarantine

Why the loop is bounded at two and the third outcome is quarantine
------------------------------------------------------------------

C1's four-piece set, from the graph spec:

    progress measure    item_attempt, incremented per attempt
    semantic exit       the annotation passes schema validation
    hard bound          MAX_ATTEMPTS = 2
    exhaustion outcome  quarantine with terminated_by=repair_budget

An exhausted item must never be admitted. That is L1-ANNOT-EXHAUST, and its
negative injection is "mark the exhausted item as admitted" -- a bound treated
as success is a bound that does nothing.

Deviation D-2
-------------

Qwen2.5-VL replaces GPT-4o, recorded per asset as ``annotator_model``. There is
no fallback to ULIP-2's shipped captions; see annotate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from metafind import paths, runlog

# BEFORE any transformers/torch import anywhere in this process. HF_HOME is read
# at IMPORT time, not at from_pretrained time, so setting it inside Annotator
# after `import transformers` had no effect: the run started re-downloading 16 GB
# of weights into ~/.cache on the 100 GB root partition while a complete copy sat
# on the data volume. It looked like a slow load for six minutes.
paths.setup_env()

from metafind.data.annotate import (
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    REQUIRED_FIELDS,
    AnnotationError,
    build_prompt,
    build_repair_prompt,
    parse_annotation,
    validate_annotation,
)

NODE = "n05_annotate"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # D-2: stands in for GPT-4o
MAX_NEW_TOKENS = 512


def sidecar_path(uid: str) -> Path:
    return paths.ANNOTATIONS / f"{uid}.json"


def is_complete(uid: str) -> bool:
    """Completion is a parseable sidecar carrying the fields the channel declares.

    Same contract as n03 and n04, and for the same reason: the record IS the
    artifact here, so a half-written one is the only failure worth guarding.
    """
    sc = sidecar_path(uid)
    if not sc.exists():
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # Includes prompt_version: a v1 sidecar has every v2 field name it shares
    # but a different schema, and treating it as done would silently mix two
    # annotation generations in one corpus.
    return (all(k in rec for k in REQUIRED_FIELDS)
            and rec.get("prompt_version") == PROMPT_VERSION
            and "annotator_model" in rec)


class Annotator:
    """Holds the model. One instance per process; loading costs ~30 s and 16 GB."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

    def generate(self, image_paths: list[str], prompt: str) -> str:
        """One forward pass over all 11 views at once.

        All views in a single conversation turn, not eleven separate calls: the
        annotation is about ONE object, and asking eleven times would produce
        eleven opinions to reconcile rather than one description informed by
        every angle.
        """
        import torch

        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                # Greedy. The repair loop feeds the specific error back, so a
                # retry must differ because the PROMPT differs -- not because
                # the sampler rolled differently. Temperature here would make a
                # failure that repeats look like one that was fixed.
                do_sample=False,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def annotate_one(ann: Annotator, uid: str, render_rec: dict) -> tuple[dict | None, dict | None]:
    """SG1 for one asset. Returns ``(record, quarantine_entry)`` -- exactly one is None."""
    views = render_rec["view_paths"]
    prompt = build_prompt(len(views))
    current = prompt
    last_error = ""
    last_raw = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = ann.generate(views, current)
        last_raw = raw
        try:
            annotation = validate_annotation(parse_annotation(raw))
        except AnnotationError as exc:
            last_error = str(exc)
            if attempt < MAX_ATTEMPTS:
                # The specific error goes back in. Re-sending the original
                # reproduces the original mistake.
                current = build_repair_prompt(prompt, last_error, raw)
            continue

        rec = annotation.as_record(ann.model_id)
        rec |= {
            "uid": uid,
            "prompt_version": PROMPT_VERSION,
            "attempts": attempt,
            # F13: the annotator saw scale-normalised renders, so its size
            # estimate is a category prior. The mesh's own bounding box travels
            # with it so the estimate can be audited -- and it is a WEAK ground
            # truth, since Objaverse authors choose their own units.
            "raw_bbox_extents": render_rec.get("raw_bbox_extents"),
        }
        return rec, None

    # Exhausted. Quarantine, never admit: L1-ANNOT-EXHAUST.
    return None, {
        "uid": uid,
        "failure_class": "MODEL_RECOVERABLE",
        "terminated_by": "repair_budget",
        "attempts": MAX_ATTEMPTS,
        "exception_type": "AnnotationError",
        "exception_msg": last_error[:400],
        "raw_response": last_raw[:1000],
    }


def rebuild_index(index_path: Path) -> int:
    """Derive annotations_index.jsonl from the per-asset sidecars."""
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as f:
        for sc in sorted(paths.ANNOTATIONS.glob("*.json")):
            try:
                f.write(json.dumps(json.loads(sc.read_text())) + "\n")
                n += 1
            except (OSError, json.JSONDecodeError):
                continue
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    # A specific uid list, for validation batches. `--limit N` takes the first N
    # of a SORTED corpus, which is fine for a smoke test and useless for
    # measuring against ground truth: the assets that happen to sort first are
    # not the assets AI2-THOR can adjudicate.
    ap.add_argument("--uids-file", type=Path,
                    help="annotate exactly these uids, one per line")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    paths.ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    renders = {}
    index = paths.LOGS / "renders_index.jsonl"
    if not index.exists():
        print(f"{index} not found -- run n04_render_views first", flush=True)
        return 2
    for line in index.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            renders[r["uid"]] = r

    if args.uids_file:
        wanted = [u for u in args.uids_file.read_text().split() if u]
        missing = [u for u in wanted if u not in renders]
        if missing:
            print(f"{len(missing)} uid(s) have no render, e.g. {missing[:3]}", flush=True)
            return 2
        todo = [u for u in wanted if args.force or not is_complete(u)]
    else:
        todo = [u for u in sorted(renders) if args.force or not is_complete(u)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(renders):,} rendered assets, {len(todo):,} to annotate", flush=True)
    if not todo:
        return 0

    ann = Annotator(args.model)
    done, quarantined, started = 0, 0, time.time()
    with runlog.run_progress(NODE):
        for uid in todo:
            try:
                rec, bad = annotate_one(ann, uid, renders[uid])
            except Exception as exc:  # noqa: BLE001 -- one asset must not stop the run
                runlog.quarantine(NODE, [{
                    "uid": uid,
                    "failure_class": ("RESOURCE" if "memory" in str(exc).lower()
                                      else "TRANSIENT"),
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                quarantined += 1
                continue

            if bad is not None:
                runlog.quarantine(NODE, [bad])
                quarantined += 1
                continue

            sc = sidecar_path(uid)
            tmp = sc.with_suffix(".json.part")
            with tmp.open("w") as fh:
                json.dump(rec, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(sc)
            done += 1
            if done % 100 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                print(f"  [{done:6d}/{len(todo)}] {rate:.1f}/min, "
                      f"剩餘約 {(len(todo)-done)/max(rate,1e-9):.0f} 分, "
                      f"quarantine {quarantined}", flush=True)

    n_indexed = rebuild_index(paths.LOGS / "annotations_index.jsonl")
    runlog.cost_ledger(
        wallclock_s=round(time.time() - started, 1),
        assets_annotated=done,
        vlm_calls=done + quarantined * MAX_ATTEMPTS,
    )
    print(f"\n{done:,} annotated this run, {n_indexed:,} complete on disk, "
          f"{quarantined:,} quarantined -> {paths.ANNOTATIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
