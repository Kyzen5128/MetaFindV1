"""Seven-modality, same-UID retrieval under a frozen custom protocol.

IMPLEMENTATION CHOICE: this evaluates our declared observation construction;
it does not claim to recover the authors' unpublished Table 1 protocol.
Preparation is CPU-only: python -m metafind.eval.custom_protocol --help.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from metafind.eval.retrieval import QUERY_CONDITIONS, normalize_for_scoring
from metafind.eval.run_retrieval import score_streaming

ARMS = ("same_observation", "different_observations")
MODALITIES = ("text", "image", "pc")


def _read_array_source(source: dict) -> bytes:
    """Consume exactly the bytes whose identity was fixed at preparation."""
    data = Path(source["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError(f"input changed after protocol preparation: {source['path']}")
    return data


def _cloud(record: dict, query: bool = False) -> np.ndarray:
    key = "query_cloud" if query else "cloud"
    raw = _read_array_source(record[key])
    if query:
        cloud = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            cloud = np.concatenate([z["xyz"], z["rgb"]], axis=1)
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.shape != (10000, 6) or not np.isfinite(cloud).all():
        raise ValueError(f"{key} must be finite (10000, 6): {record[key]['path']}")
    return cloud


def validate_cache_compatibility(protocol: dict, encoding: dict | None = None) -> None:
    """A new query protocol must not disguise a different cache construction."""
    frozen = protocol["encoding_protocol"]["content"]
    if frozen.get("image_aggregation") != "mean":
        raise ValueError("custom evaluation v1 requires a mean-image cache")
    if encoding is not None:
        for key in ("text_serialization", "text_template", "image_aggregation"):
            if encoding.get(key) != frozen.get(key):
                raise ValueError(f"checkpoint and evaluation cache disagree on {key}")
        from metafind.train.stage1 import protocol_n_views
        if protocol_n_views(encoding) != protocol["n_views"]:
            raise ValueError("checkpoint and evaluation cache disagree on n_views")


def available_mean(embeds: dict[str, np.ndarray], condition: str = "full") -> np.ndarray:
    """L2-normalise each available modality, average, then L2-normalise.

    No learned modules, mask tokens, trained point encoder, or missing slots.
    This fixed baseline is an IMPLEMENTATION CHOICE, recorded in every result.
    """
    present = QUERY_CONDITIONS[condition]
    vectors = [normalize_for_scoring(embeds[m])
               for m, yes in zip(MODALITIES, present) if yes]
    return normalize_for_scoring(np.mean(vectors, axis=0))


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _tokens_differ(backbone, records: list[dict]) -> None:
    """Different strings can become identical after CLIP's truncation."""
    import torch

    a = backbone.tokenizer([r["canonical_text"] for r in records])
    b = backbone.tokenizer([r["query_text"] for r in records])
    same = torch.as_tensor(a).eq(torch.as_tensor(b)).all(dim=-1)
    if bool(same.any()):
        raise ValueError("query/canonical text has identical effective CLIP tokens; "
                         "provide a different description within the text context")


def encode_observations(protocol: dict, backbone, *, device: str,
                        batch_size: int, query_backbone=None) -> dict:
    """Encode each point path separately; share frozen CLIP observations.

    Gallery candidates each have a fixed held-out view, including candidates
    which never appear as queries. Neither gallery depends on the current query.
    Canonical text is encoded from its frozen source string, so both sides use
    the actual restored CLIP tokenizer/weights. Image vectors come from the
    content-verified per-view cache; trainable-CLIP checkpoints are unsupported.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    gallery_uids, query_uids = protocol["gallery_uids"], protocol["query_uids"]
    records = protocol["records"]
    gallery = {arm: {m: [] for m in MODALITIES} for arm in ARMS}
    with torch.no_grad():
        for start in range(0, len(gallery_uids), batch_size):
            uids = gallery_uids[start:start + batch_size]
            recs = [records[u] for u in uids]
            text = _numpy(backbone.encode_text([r["canonical_text"] for r in recs]))
            pc = _numpy(backbone.encode_pc(torch.from_numpy(np.stack([_cloud(r) for r in recs]))))
            same, different = [], []
            for r in recs:
                with np.load(io.BytesIO(_read_array_source(r["embedding"])), allow_pickle=False) as z:
                    # n06 rounds its precomputed mean and per-view vectors
                    # separately to float16. Derive BOTH arms from the same
                    # stored view matrix to keep their arithmetic comparable.
                    same.append(np.asarray(z["views"], dtype=np.float32).mean(axis=0))
                    different.append(np.asarray(z["views"][r["gallery_views"]],
                                                dtype=np.float32).mean(axis=0))
            for arm, images in zip(ARMS, (same, different)):
                for m, values in (("text", text), ("pc", pc), ("image", np.stack(images))):
                    gallery[arm][m].append(values)
            print(f"gallery observations {start + len(uids)}/{len(gallery_uids)}", flush=True)
        gallery = {arm: {m: np.concatenate(v) for m, v in parts.items()}
                   for arm, parts in gallery.items()}
        where = {uid: i for i, uid in enumerate(gallery_uids)}
        targets = np.array([where[u] for u in query_uids], dtype=np.int64)
        # A always uses the very same observations, even with separate weights.
        query = {"same_observation": {m: v[targets].copy()
                                     for m, v in gallery[ARMS[0]].items()},
                 "different_observations": {m: [] for m in MODALITIES}}
        qbb = query_backbone if query_backbone is not None else backbone
        q_same_pc = []
        for start in range(0, len(query_uids), batch_size):
            uids = query_uids[start:start + batch_size]
            recs = [records[u] for u in uids]
            _tokens_differ(backbone, recs)
            text = _numpy(backbone.encode_text([r["query_text"] for r in recs]))
            pc = _numpy(qbb.encode_pc(torch.from_numpy(np.stack([_cloud(r, True) for r in recs]))))
            if query_backbone is not None:
                q_same_pc.append(_numpy(qbb.encode_pc(torch.from_numpy(np.stack([_cloud(r) for r in recs])))))
            images = []
            for r in recs:
                with np.load(io.BytesIO(_read_array_source(r["embedding"])), allow_pickle=False) as z:
                    images.append(np.asarray(z["views"][r["query_view"]], dtype=np.float32))
            for m, values in (("text", text), ("image", np.stack(images)), ("pc", pc)):
                query[ARMS[1]][m].append(values)
            print(f"query observations {start + len(uids)}/{len(query_uids)}", flush=True)
        query[ARMS[1]] = {m: np.concatenate(v) for m, v in query[ARMS[1]].items()}
        if query_backbone is not None:
            query[ARMS[0]]["pc"] = np.concatenate(q_same_pc)
    return {"gallery": gallery, "query": query, "targets": targets}


def fuse_observations(observations: dict, *, model=None, device="cpu",
                      batch_size=256) -> dict:
    """Encode both galleries once per method; each serves all seven queries."""
    import torch

    def fused(parts, condition=None):
        if model is None:
            return available_mean(parts, condition or "full")
        chunks = []
        n = len(parts["text"])
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                inputs = {m: torch.as_tensor(v[start:stop], device=device) for m, v in parts.items()}
                if condition is None:
                    value = model.gallery(inputs)
                else:
                    flags = QUERY_CONDITIONS[condition]
                    present = torch.tensor(flags, dtype=torch.bool, device=device).expand(stop - start, -1)
                    # Never provide an absent modality's actual feature vector.
                    inputs = {m: inputs[m] if yes else None for m, yes in zip(MODALITIES, flags)}
                    value = model.query(inputs, present=present, layout=None)
                chunks.append(_numpy(value))
        return normalize_for_scoring(np.concatenate(chunks))

    return {arm: {"gallery": fused(observations["gallery"][arm]),
                  "query": {c: fused(observations["query"][arm], c)
                            for c in QUERY_CONDITIONS}}
            for arm in ARMS}


def evaluate_vectors(vectors: dict, protocol: dict, output: Path, method: str,
                     *, block: int = 4096) -> dict:
    """Shared scorer, explicit UID targets, per-query evidence, no pool changes."""
    queries, gallery = protocol["query_uids"], protocol["gallery_uids"]
    if not queries or not gallery or len(set(queries)) != len(queries) or len(set(gallery)) != len(gallery):
        raise ValueError("query and gallery UID lists must be nonempty and unique")
    where = {uid: i for i, uid in enumerate(gallery)}
    if not set(queries) <= where.keys():
        raise ValueError("every query target must occur in the gallery")
    targets = np.array([where[u] for u in queries], dtype=np.int64)
    result = {}
    for arm in ARMS:
        g = normalize_for_scoring(vectors[arm]["gallery"])
        if len(g) != len(gallery):
            raise ValueError("gallery vectors do not match the fixed UID list")
        result[arm] = {}
        for condition in QUERY_CONDITIONS:
            q = normalize_for_scoring(vectors[arm]["query"][condition])
            if len(q) != len(queries) or q.shape[1] != g.shape[1]:
                raise ValueError("query vectors do not match the fixed UID list/gallery dimensions")
            scores = score_streaming(q, g, targets, block=block)
            ranks = scores["rank"]
            metrics = {"R@1": float(np.mean(ranks <= 1) * 100),
                       "R@5": float(np.mean(ranks <= 5) * 100)}
            result[arm][condition] = metrics
            evidence = output / f"{method}__{arm}__{condition.replace('+', '_')}.jsonl"
            with evidence.open("x", encoding="utf-8") as f:
                for i, uid in enumerate(queries):
                    f.write(json.dumps({"query_uid": uid, "target_uid": uid,
                                        "target_column": int(targets[i]),
                                        "top1_uid": gallery[int(scores["top1_col"][i])],
                                        "rank": int(ranks[i]),
                                        "tie_count": int(scores["tie_count"][i])}) + "\n")
            print(f"{method} {arm} {condition}: {metrics['R@1']:.2f}/{metrics['R@5']:.2f}", flush=True)
        for condition, flags in QUERY_CONDITIONS.items():
            if sum(flags) > 1:
                best = max(result[arm][m]["R@1"] for m, yes in zip(MODALITIES, flags) if yes)
                result[arm][condition]["gain_over_best_constituent_R@1_pp"] = result[arm][condition]["R@1"] - best
    return result


def markdown_report(result: dict) -> str:
    lines = ["# Custom seven-modality retrieval", "",
             "IMPLEMENTATION CHOICE: fixed same-UID retrieval, not an exact reproduction of the authors' Table 1.",
             f"Queries: {result['n_query']}; gallery candidates: {result['n_gallery']}. Values are R@1 / R@5 (%).",
             "Evaluation role: development comparison; independent test status is not certified.", ""]
    for arm in ARMS:
        lines += [f"## {arm}", "", "| Method | T | I | P | T+I | T+P | I+P | T+I+P |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, rows in result["methods"].items():
            cells = [f"{rows[arm][c]['R@1']:.2f} / {rows[arm][c]['R@5']:.2f}" for c in QUERY_CONDITIONS]
            lines.append("| " + " | ".join([method] + cells) + " |")
        lines.append("")
    lines += ["Stage 2 is evaluated with layout=None and the Stage 1 parent gallery. This does not measure scene/layout benefit.",
              "Different observations share the target mesh/UID; they do not establish statistical independence or semantic substitute relevance.", ""]
    return "\n".join(lines)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage1-record", type=Path, required=True)
    parser.add_argument("--stage2-record", type=Path)
    parser.add_argument("--stage2-variant", default="full")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--check-only", action="store_true", help="Verify frozen inputs/configurations without constructing models or writing results")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.block < 1:
        parser.error("batch-size and block must be positive")

    from metafind.eval.custom_protocol import load_protocol
    from metafind.train.gallery_index import load_checkpoint_record, load_stage2_checkpoint_record
    from metafind.train.stage1 import load_stage1_model_config

    protocol_bytes = args.protocol.read_bytes()
    protocol = load_protocol(args.protocol)
    if args.protocol.read_bytes() != protocol_bytes:
        raise ValueError("protocol changed while it was being verified")
    record = load_checkpoint_record(args.stage1_record)
    record = load_stage1_model_config(record["uri"], record)
    # Embedded configuration is authoritative; current training files are not.
    from metafind.train.stage1 import effective_stage1_model_inputs, load_protocols
    encoding, training, _ = effective_stage1_model_inputs(record, *load_protocols())
    validate_cache_compatibility(protocol, encoding)
    from metafind.train.stage1 import fusion_config_for
    if (any(fusion_config_for(encoding, training, gallery=g).image_tokens != 1
            for g in (False, True)) or training.get("train_scope") == "full"
            or encoding.get("actual_clip_train_scope") != "frozen"):
        raise ValueError("custom evaluation v1 requires frozen CLIP and one image token")
    stage2_record = None
    if args.stage2_record:
        stage2_record = load_stage2_checkpoint_record(args.stage2_record, record, args.stage2_variant,
                                                     allow_legacy=False)
        if training.get("tower_sharing") == "fully_shared":
            raise ValueError("Stage 2 requires separate query/gallery fusion weights")
    from metafind.train.stage1 import stage1_backbone_kwargs
    from metafind.eval.custom_models import verify_mean_initializer
    stage1_backbone_kwargs(record)
    verify_mean_initializer()
    if args.check_only:
        print(json.dumps({"status": "inputs_verified", "n_query": len(protocol["query_uids"]),
                          "n_gallery": len(protocol["gallery_uids"]), "n_views": protocol["n_views"],
                          "effective_token_check": "required during model encoding",
                          "model_execution": "not_performed"}, indent=2))
        return 0

    import torch
    from metafind import runlog
    from metafind.eval.custom_models import load_mean, load_stage1, apply_stage2

    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}; use a new run directory")
    out.mkdir(parents=True)
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    source_hash = runlog.runtime_source_sha256()
    if not source_hash:
        raise RuntimeError("cannot fingerprint the evaluation source tree")
    provenance = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                  "stage1": record, "stage2": stage2_record,
                  "command": sys.argv if argv is None else argv,
                  "seed": args.seed, "device": args.device, "batch_size": args.batch_size, "block": args.block,
                  "code_revision": runlog.code_revision(), "code_dirty": runlog.code_dirty(),
                  "runtime_source_sha256": source_hash, "python": platform.python_version(),
                  "numpy": np.__version__, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "platform": platform.platform()}
    with (out / "protocol.json").open("xb") as f:
        f.write(protocol_bytes)
    _write_json(out / "provenance.json", provenance)
    result = {"schema": "metafind.custom_table1.results.v1", "claim": "custom_same_uid_retrieval",
              "evidence_class": "OBSERVED DATA", "evaluation_role": "development_comparison",
              "independent_test_status": "not_certified", "n_query": len(protocol["query_uids"]),
              "n_gallery": len(protocol["gallery_uids"]), "methods": {},
              "mean_baseline": "official ULIP-2; normalise each available modality, mean, normalise; full-modality own gallery",
              "scoring": "float64 cosine; exact UID; pessimistic ties; R@k percentages",
              "provenance": provenance}
    try:
        # Release the pretrained model before loading the trained point paths.
        baseline = load_mean(args.device)
        from metafind.train.stage1 import OFFICIAL_ULIP2_SHA256, _open_clip_weight_identity
        result["baseline_initializers"] = {"ulip2_sha256": OFFICIAL_ULIP2_SHA256,
                                           "open_clip": _open_clip_weight_identity()}
        with torch.no_grad():
            observations = encode_observations(protocol, baseline, device=args.device, batch_size=args.batch_size)
            vectors = fuse_observations(observations)
            result["methods"]["ulip2_available_mean"] = evaluate_vectors(vectors, protocol, out, "ulip2_available_mean", block=args.block)
        del observations, vectors, baseline
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        loaded = load_stage1(args.stage1_record, args.device)
        if loaded.record["sha256"] != record["sha256"]:
            raise ValueError("Stage 1 checkpoint changed after preflight")
        observations = encode_observations(protocol, loaded.backbone, device=args.device,
                                           batch_size=args.batch_size, query_backbone=loaded.query_backbone)
        vectors = fuse_observations(observations, model=loaded.model, device=args.device, batch_size=args.batch_size)
        result["methods"]["stage1"] = evaluate_vectors(vectors, protocol, out, "stage1", block=args.block)
        if args.stage2_record:
            frozen_gallery = {arm: vectors[arm]["gallery"].copy() for arm in ARMS}
            applied = apply_stage2(loaded, args.stage2_record, args.stage2_variant)
            if applied["sha256"] != stage2_record["sha256"]:
                raise ValueError("Stage 2 checkpoint changed after preflight")
            vectors = fuse_observations(observations, model=loaded.model, device=args.device, batch_size=args.batch_size)
            for arm in ARMS:
                if not np.array_equal(frozen_gallery[arm], vectors[arm]["gallery"]):
                    raise ValueError("Stage 2 overlay changed the parent gallery")
                vectors[arm]["gallery"] = frozen_gallery[arm]
            result["methods"]["stage2_layout_off"] = evaluate_vectors(vectors, protocol, out, "stage2_layout_off", block=args.block)
        for method, rows in result["methods"].items():
            for arm in ARMS:
                for condition, metrics in rows[arm].items():
                    metrics["gain_over_mean_R@1_pp"] = metrics["R@1"] - result["methods"]["ulip2_available_mean"][arm][condition]["R@1"]
        result["elapsed_seconds"] = time.time() - started
        result["validation"] = {"effective_text_tokens": "different for every query on each loaded CLIP tokenizer",
                                "input_files": "verified against prepared hashes",
                                "query_gallery_views": "disjoint in different_observations",
                                "image_cache_encoder_identity": "producer sidecars recorded; historical OpenCLIP weight bytes not independently certified",
                                "stage2_parent_gallery": "unchanged" if args.stage2_record else "not_applicable"}
        _write_json(out / "results.json", result)
        (out / "table.md").write_text(markdown_report(result), encoding="utf-8")
        print(f"Completed custom evaluation: {out / 'table.md'}", flush=True)
    except BaseException as exc:
        _write_json(out / "failure.json", {"status": "failed", "error": str(exc),
                                            "exception": type(exc).__name__, "elapsed_seconds": time.time() - started})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
