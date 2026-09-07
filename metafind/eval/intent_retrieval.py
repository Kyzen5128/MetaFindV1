"""Evaluate independently authored furniture needs against reviewed acceptable assets.

IMPLEMENTATION CHOICE: this is a multi-positive intent benchmark, not MetaFind
Table 1's unavailable complete protocol. No query is constructed from its answer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from metafind.eval.custom_table1 import (
    MODALITIES, _cloud, _numpy, _read_array_source, available_mean,
    validate_cache_compatibility,
)
from metafind.eval.retrieval import QUERY_CONDITIONS, normalize_for_scoring


def _vectors(value, rows, label, width=None):
    value = _numpy(value)
    if (value.ndim != 2 or value.shape[0] != rows or not value.shape[1]
            or (width is not None and value.shape[1] != width)
            or not np.isfinite(value).all()
            or np.any(np.linalg.norm(value.astype(np.float64), axis=1) == 0)):
        raise ValueError(f"{label}: invalid encoded vectors")
    return value


def encode_gallery(protocol, backbone, *, batch_size=16):
    """Encode a fixed complete gallery once; never depend on the current query."""
    import torch
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = {m: [] for m in MODALITIES}
    uids = protocol["gallery_uids"]
    width = None
    with torch.inference_mode():
        for start in range(0, len(uids), batch_size):
            records = [protocol["records"][u] for u in uids[start:start + batch_size]]
            text = _vectors(backbone.encode_text([r["canonical_text"] for r in records]), len(records), "gallery text", width)
            width = text.shape[1]
            pc = _vectors(backbone.encode_pc(torch.from_numpy(np.stack([_cloud(r) for r in records]))), len(records), "gallery pc", width)
            images = []
            for r in records:
                with np.load(io.BytesIO(_read_array_source(r["embedding"])), allow_pickle=False) as z:
                    image = np.asarray(z["views"], dtype=np.float32).mean(axis=0)
                if image.shape != (width,) or not np.isfinite(image).all() or np.linalg.norm(image) == 0:
                    raise ValueError("gallery image cache disagrees with loaded encoder width")
                images.append(image)
            for m, values in (("text", text), ("image", np.stack(images)), ("pc", pc)):
                result[m].append(values)
    return {m: np.concatenate(v) for m, v in result.items()}


def encode_queries(protocol, backbone, *, query_backbone=None):
    """Read literal external query text/images/clouds; never read qrels for inputs."""
    import torch
    from PIL import Image, ImageOps
    from metafind.data.pointclouds import pc_norm
    from metafind.eval.intent_protocol import verified_source_bytes

    encoded, token_records = {}, {}
    qbb = query_backbone if query_backbone is not None else backbone
    with torch.inference_mode():
        for query in protocol["queries"]:
            parts = {}
            if "text" in query:
                parts["text"] = _vectors(backbone.encode_text([query["text"]]), 1, "query text")
                token_records[query["query_id"]] = {
                    "text": query["text"],
                    "effective_token_ids": torch.as_tensor(backbone.tokenizer([query["text"]])).tolist()[0],
                }
            if "images" in query:
                images = []
                for source in query["images"]:
                    with Image.open(io.BytesIO(verified_source_bytes(source))) as image:
                        batch = backbone.preprocess(ImageOps.exif_transpose(image).convert("RGB"))[None]
                    images.append(_vectors(backbone.encode_image(batch), 1, "query image"))
                parts["image"] = np.mean(np.stack(images), axis=0, dtype=np.float32)
            if "pointcloud" in query:
                raw = np.load(io.BytesIO(verified_source_bytes(query["pointcloud"])), allow_pickle=False)
                cloud = np.asarray(raw, dtype=np.float32).copy()
                if cloud.shape != (10000, 6) or not np.isfinite(cloud).all():
                    raise ValueError("query pointcloud must be finite (10000, 6)")
                if not np.any(cloud[:, :3] != cloud[0, :3]):
                    raise ValueError("query pointcloud has no spatial extent")
                cloud[:, :3] = pc_norm(np.asarray(cloud[:, :3], dtype=np.float64)).astype(np.float32)
                parts["pc"] = _vectors(qbb.encode_pc(torch.from_numpy(cloud[None])), 1, "query pc")
            if not parts or len({v.shape[1] for v in parts.values()}) != 1:
                raise ValueError("query modality widths disagree or no input was provided")
            encoded[query["query_id"]] = parts
    return encoded, token_records


def fuse_gallery(parts, *, model=None, device="cpu", batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model is None:
        return available_mean(parts, "full")
    import torch
    values = []
    with torch.inference_mode():
        for start in range(0, len(parts["text"]), batch_size):
            inputs = {m: torch.as_tensor(v[start:start + batch_size], device=device) for m, v in parts.items()}
            values.append(_vectors(model.gallery(inputs), len(inputs["text"]), "gallery fusion"))
    return normalize_for_scoring(np.concatenate(values))


def fuse_queries(protocol, encoded, *, model=None, device="cpu"):
    """Each modality condition gets only its declared inputs, and its own labels."""
    import torch
    result = {}
    with torch.inference_mode():
        for condition, flags in QUERY_CONDITIONS.items():
            selected = [q for q in protocol["queries"] if condition in q["conditions"]]
            if not selected:
                continue
            values = []
            for query in selected:
                parts = encoded[query["query_id"]]
                for m, yes in zip(MODALITIES, flags):
                    if yes and m not in parts:
                        raise ValueError(f"{query['query_id']}: requested {condition} has no {m}")
                if model is None:
                    value = available_mean(parts, condition)
                else:
                    inputs = {m: torch.as_tensor(parts[m], device=device) if yes else None
                              for m, yes in zip(MODALITIES, flags)}
                    present = torch.tensor([flags], dtype=torch.bool, device=device)
                    value = _vectors(model.query(inputs, present=present, layout=None), 1, "query fusion")
                values.append(value)
            result[condition] = {"query_ids": [q["query_id"] for q in selected],
                                 "vectors": normalize_for_scoring(np.concatenate(values))}
    return result


def evaluate_method(protocol, gallery, queries, output, method, *, block=4096):
    from metafind.eval.intent_metrics import evaluate_rankings
    records = {q["query_id"]: q for q in protocol["queries"]}
    results = {}
    for condition, group in queries.items():
        ids = group["query_ids"]
        positive = [[uid for uid in protocol["gallery_uids"] if records[qid]["qrels"][condition][uid]] for qid in ids]
        outcome = evaluate_rankings(group["vectors"], gallery, gallery_uids=protocol["gallery_uids"],
                                    query_ids=ids, relevant_uids=positive, block=block)
        results[condition] = {"n_query": len(ids), "query_ids": ids, "metrics": outcome["metrics"]}
        path = output / f"{method}__{condition.replace('+', '_')}.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for row in outcome["records"]:
                f.write(json.dumps({**row, "condition": condition}, ensure_ascii=False, allow_nan=False) + "\n")
    return results


def markdown_report(result):
    lines = ["# Furniture-intent retrieval", "", "IMPLEMENTATION CHOICE: reviewed multi-positive intent queries; not the author's Table 1.",
             "", "Hit@k: at least one acceptable asset in top k. Recall@k: fraction of all acceptable assets retrieved.",
             "All scores below are query-macro percentages. Unjudged candidates are not treated as negatives.",
             "Evaluation role: development comparison; independent test status is not certified.", "",
             "| Method | Query condition | Queries | Hit@1 | Hit@5 | Recall@1 | Recall@5 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method, groups in result["methods"].items():
        for condition, row in groups.items():
            m = row["metrics"]
            lines.append(f"| {method} | {condition} | {row['n_query']} | " + " | ".join(f"{m[k]:.2f}" for k in ("Hit@1", "Hit@5", "Recall@1", "Recall@5")) + " |")
    lines += ["", "Conditions may use different explicit query sets and relevance judgments; their raw differences do not isolate fusion gains.",
              "Stage 2 uses no layout input and preserves the Stage 1 gallery; this does not assess scene benefit.",
              "Input provenance and declared human/assessor judgments are recorded, not independently certified by code."]
    return "\n".join(lines) + "\n"


def _json_write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage1-record", type=Path, required=True)
    parser.add_argument("--stage2-record", type=Path)
    parser.add_argument("--stage2-variant", default="full")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.block < 1:
        parser.error("batch-size and block must be positive")
    from metafind.eval.intent_protocol import load_protocol
    from metafind.eval.custom_models import load_mean, load_stage1, apply_stage2, verify_mean_initializer
    from metafind.train.gallery_index import load_checkpoint_record, load_stage2_checkpoint_record
    from metafind.train import stage1

    protocol_bytes = args.protocol.read_bytes()
    protocol = load_protocol(args.protocol)
    if args.protocol.read_bytes() != protocol_bytes:
        raise ValueError("protocol changed during preflight")
    record = load_checkpoint_record(args.stage1_record)
    record = stage1.load_stage1_model_config(record["uri"], record)
    encoding, training, _ = stage1.effective_stage1_model_inputs(record, *stage1.load_protocols())
    validate_cache_compatibility(protocol, encoding)
    if training.get("train_scope") == "full" or encoding.get("actual_clip_train_scope") != "frozen":
        raise ValueError("intent evaluation v1 requires frozen CLIP")
    if any(stage1.fusion_config_for(encoding, training, gallery=g).image_tokens != 1 for g in (False, True)):
        raise ValueError("intent evaluation v1 requires one pooled image token")
    child = load_stage2_checkpoint_record(args.stage2_record, record, args.stage2_variant, allow_legacy=False) if args.stage2_record else None
    if child and training.get("tower_sharing") == "fully_shared":
        raise ValueError("Stage 2 requires separate query/gallery fusion weights")
    stage1.stage1_backbone_kwargs(record)
    verify_mean_initializer()
    if args.check_only:
        print(json.dumps({"status": "inputs_verified", "n_query": len(protocol["queries"]),
                          "n_gallery": len(protocol["gallery_uids"]),
                          "judgment_validity": "coverage verified; relevance remains assessor supplied",
                          "effective_tokens": "not encoded in check-only"}))
        return 0

    import torch
    from metafind import runlog
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source_tree_hash = runlog.runtime_source_sha256()
    if not source_tree_hash:
        raise RuntimeError("cannot fingerprint evaluation source tree")
    source_files = [Path(__file__), Path(__file__).with_name("intent_protocol.py"),
                    Path(__file__).with_name("intent_metrics.py"), Path(__file__).with_name("custom_models.py"),
                    Path(__file__).with_name("custom_table1.py")]
    provenance = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                  "stage1": record, "stage2": child, "command": sys.argv if argv is None else list(argv),
                  "seed": args.seed, "device": args.device, "batch_size": args.batch_size, "block": args.block,
                  "code_revision": runlog.code_revision(), "code_dirty": runlog.code_dirty(),
                  "runtime_source_sha256": source_tree_hash,
                  "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "query_image_preprocessing": "EXIF transpose, PIL RGB, restored CLIP preprocess, arithmetic mean of explicit images",
                  "query_pointcloud_preprocessing": "pc_norm(float64 xyz) to float32; RGB unchanged",
                  "image_cache_encoder_identity": "gallery per-view bytes/sidecars bound; historical CLIP weights not independently certified"}
    with (output / "protocol.json").open("xb") as f:
        f.write(protocol_bytes)
    _json_write(output / "provenance.json", provenance)
    result = {"schema": "metafind.intent_retrieval.results.v1", "claim": "custom_furniture_intent_retrieval",
              "evaluation_role": "development_comparison", "independent_test_status": "not_certified",
              "evidence_class": "OBSERVED DATA", "n_gallery": len(protocol["gallery_uids"]),
              "methods": {}, "provenance": provenance,
              "scoring": "float64 cosine descending; ties gallery UID lexical ascending; Hit and Recall query-macro percentages"}
    try:
        baseline = load_mean(args.device)
        result["baseline_initializers"] = {"ulip2_sha256": stage1.OFFICIAL_ULIP2_SHA256,
                                            "open_clip": stage1._open_clip_weight_identity()}
        gallery = fuse_gallery(encode_gallery(protocol, baseline, batch_size=args.batch_size))
        encoded, tokens = encode_queries(protocol, baseline)
        vectors = fuse_queries(protocol, encoded)
        result["methods"]["ulip2_available_mean"] = evaluate_method(protocol, gallery, vectors, output, "ulip2_available_mean", block=args.block)
        _json_write(output / "ulip2_query_tokens.json", tokens)
        del gallery, encoded, vectors, baseline
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        loaded = load_stage1(args.stage1_record, args.device)
        if loaded.record["sha256"] != record["sha256"]:
            raise ValueError("Stage 1 checkpoint changed after preflight")
        parts = encode_gallery(protocol, loaded.backbone, batch_size=args.batch_size)
        gallery = fuse_gallery(parts, model=loaded.model, device=args.device, batch_size=args.batch_size)
        encoded, tokens = encode_queries(protocol, loaded.backbone, query_backbone=loaded.query_backbone)
        vectors = fuse_queries(protocol, encoded, model=loaded.model, device=args.device)
        result["methods"]["stage1"] = evaluate_method(protocol, gallery, vectors, output, "stage1", block=args.block)
        _json_write(output / "stage1_query_tokens.json", tokens)
        if child:
            applied = apply_stage2(loaded, args.stage2_record, args.stage2_variant)
            if applied["sha256"] != child["sha256"]:
                raise ValueError("Stage 2 checkpoint changed after preflight")
            child_gallery = fuse_gallery(parts, model=loaded.model, device=args.device, batch_size=args.batch_size)
            if not np.array_equal(gallery, child_gallery):
                raise ValueError("Stage 2 overlay changed the parent gallery")
            vectors = fuse_queries(protocol, encoded, model=loaded.model, device=args.device)
            result["methods"]["stage2_layout_off"] = evaluate_method(protocol, gallery, vectors, output, "stage2_layout_off", block=args.block)
        result["elapsed_seconds"] = time.time() - started
        result["stage2_parent_gallery"] = "unchanged" if child else "not_applicable"
        _json_write(output / "results.json", result)
        (output / "table.md").write_text(markdown_report(result), encoding="utf-8")
        print(f"Completed furniture-intent evaluation: {output / 'table.md'}")
    except BaseException as error:
        _json_write(output / "failure.json", {"status": "failed", "exception": type(error).__name__,
                                             "error": str(error), "elapsed_seconds": time.time() - started})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
