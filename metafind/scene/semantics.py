"""Generate only explicitly requested scene relations with the existing n08 SG2.

All artifacts go to a new directory. Historical Stage 2 evidence identifies the
LLM by its cache-key name, not historical weight bytes; this bridge preserves
that limitation instead of asserting an unavailable byte-equivalence proof.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from metafind.data import semantic_edges_run as n08
from metafind.data.semantic_edges import cache_key, relation_text_for
from metafind.data.semantic_provenance import (
    build_node_source, build_edge_source, read_verified_edge_source,
    source_identity_sha256, text_encoder_identity,
)
from metafind.eval.custom_models import load_stage1
from metafind.scene.compose import _publish_json
from metafind.scene.prepare import _json, _read, _arrays, _relative, _training_encoder, SEMANTIC_KEY_FIELDS
from metafind.train import gallery_index, stage1


def _reference(stage1_record, stage2_record, variant, sources):
    for path in (stage1_record, stage2_record):
        _read(path, sources)
    parent = gallery_index.load_checkpoint_record(stage1_record)
    parent = stage1.load_stage1_model_config(parent["uri"], parent)
    child = gallery_index.load_stage2_checkpoint_record(stage2_record, parent, variant,
                                                       allow_legacy=False)
    for record in (parent, child):
        _read(record["uri"], sources, record["sha256"])
    if not gallery_index.stage2_layout_settings(child)[0]:
        raise ValueError("semantic generation requires a reference checkpoint with layout")
    expected, meta = _training_encoder(child, sources)
    current = dict(zip(SEMANTIC_KEY_FIELDS,
                       (n08.PROMPT_VERSION, n08.LLM_MODEL, n08.TEXT_ENCODER_VERSION)))
    if meta != current or Path(n08.LLM_MODEL_PATH).name != meta["llm_model"]:
        raise ValueError("reference semantic cache does not match the current n08 prompt/model/version")
    encoding, training, _ = stage1.effective_stage1_model_inputs(parent, {}, {}, {})
    if training["train_scope"] == "full" or encoding["actual_clip_train_scope"] != "frozen":
        raise ValueError("scene semantics requires the reference frozen CLIP text encoder")
    stage1.stage1_backbone_kwargs(parent)  # initializer preflight, no model constructed
    return parent, child, expected, meta


def _pairs(requested, texts, expected, meta):
    if not isinstance(requested, list) or not requested:
        raise ValueError("required_pairs must be a nonempty explicit list of [UID, UID]")
    if not isinstance(texts, dict):
        raise ValueError("asset_texts must be an asset UID map")
    pairs, mappings = {}, []
    for pair in requested:
        if not isinstance(pair, list) or len(pair) != 2 \
                or any(not isinstance(uid, str) or not uid.strip() or uid not in texts for uid in pair):
            raise ValueError("each required pair must contain two known nonempty asset UIDs")
        # A repeated UID is valid: two placed objects may use the same asset.
        build_node_source({uid: texts[uid] for uid in pair}, expected)
        a, b = sorted(relation_text_for(texts[uid]) for uid in pair)
        key = cache_key(a, b, **meta)
        pairs[key] = (a, b)
        mappings.append({"asset_ids": pair, "key": key, "descriptions": [a, b]})
    return pairs, mappings


def _base_cache(path, sources, expected, meta):
    if path is None:
        return {}, {}, None
    cache = _json(path, sources)
    if {k: cache[k] for k in SEMANTIC_KEY_FIELDS} != meta:
        raise ValueError("base semantic cache uses a different prompt/model/version")
    artifact = cache["embedding_artifact"]
    arrays = _arrays(_read(_relative(artifact["uri"], Path(path).parent), sources,
                          artifact["sha256"]))
    proof = read_verified_edge_source(cache, arrays)
    if proof["encoder_identity"] != expected:
        raise ValueError("base semantic cache uses a different text encoder")
    vectors = {key: row.copy() for key, row in zip(arrays["keys"].tolist(), arrays["embeddings"])}
    settled = {key: {"sentence": entry.get("sentence"), "degraded": bool(entry.get("degraded")),
                     "generation_origin": "verified_base_cache"}
               for key, entry in cache["entries"].items()}
    for key, entry in cache["entries"].items():
        if entry.get("degraded") and entry.get("embedding_uri") is not None:
            raise ValueError("degraded base relation still claims an embedding")
    return settled, vectors, arrays["embeddings"].shape[1]


def generate(asset_texts: Path, required_pairs: Path, stage1_record: Path,
             stage2_record: Path, out_dir: Path, *, variant="full", device="cpu",
             base_cache: Path | None = None) -> Path:
    """Return a new n08-compatible cache; perform no Cartesian pair expansion."""
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    sources = {}
    parent, child, expected, meta = _reference(stage1_record, stage2_record, variant, sources)
    texts, requested = _json(asset_texts, sources), _json(required_pairs, sources)
    pairs, mappings = _pairs(requested, texts, expected, meta)
    settled, vectors, base_dim = _base_cache(base_cache, sources, expected, meta)
    edge_dim = child["layout_input_dims"]["edge_feat_dim"]
    if not isinstance(edge_dim, int) or isinstance(edge_dim, bool) or edge_dim <= 0 \
            or edge_dim != n08.EDGE_DIM or base_dim not in (None, edge_dim):
        raise ValueError("reference/base semantic width does not match the n08 text encoder")
    missing = sorted(set(pairs) - set(settled))
    out_dir.mkdir(parents=True, exist_ok=False)
    sentences_path = out_dir / "sentences.jsonl"
    new_good = []
    if missing:
        # Never keep the full ULIP backbone resident alongside the LLM.
        writer = n08.RelationWriter(model_id=n08.LLM_MODEL_PATH, device=device)
        try:
            if Path(writer.model_id).resolve() != Path(n08.LLM_MODEL_PATH).resolve():
                raise ValueError("loaded relation writer differs from the configured n08 model ID")
            with sentences_path.open("x", encoding="utf-8") as stream:
                for key in missing:
                    a, b = pairs[key]
                    sentence, error = n08.write_one(writer, a, b)
                    record = {"key": key, "desc_a": a, "desc_b": b, "sentence": sentence,
                              "degraded": sentence is None, "reason": error,
                              "generation_origin": "n08_sg2_write_one",
                              "max_attempts": n08.MAX_ATTEMPTS}
                    settled[key] = record
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if sentence is not None:
                        new_good.append(key)
        finally:
            del writer
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
    else:
        sentences_path.write_text("")

    encoder_status = "verified_existing_vectors_no_new_encoding" if vectors else "not_used_no_successful_sentences"
    if new_good:
        loaded = load_stage1(stage1_record, device)
        if loaded.record["sha256"] != parent["sha256"]:
            raise ValueError("Stage 1 checkpoint changed before semantic encoding")
        if text_encoder_identity(loaded.backbone) != expected:
            raise ValueError("loaded text encoder differs from the Stage 2 semantic encoder")
        encoded = n08.encode_sentences([settled[key]["sentence"] for key in new_good],
                                       backbone=loaded.backbone)
        vectors.update(zip(new_good, encoded))
        if text_encoder_identity(loaded.backbone) != expected:
            raise ValueError("text encoder changed during semantic encoding")
        encoder_status = "verified_loaded_text_tower"
        del loaded

    embedding_path = out_dir / "sem_edge_embeddings.npz"
    cache_path = out_dir / "sem_edge_cache.json"
    entries = n08.build_cache(settled, str(embedding_path))
    identity = build_edge_source(entries, expected)
    keys = identity["keys"]
    array = np.stack([vectors[key] for key in keys]) if keys else np.empty((0, edge_dim), dtype=np.float32)
    payload = {"keys": np.array(keys), "embeddings": array.astype(np.float32),
               "source_identity_sha256": np.array(source_identity_sha256(identity))}
    cache = {**meta, "text_encoder": n08.TEXT_ENCODER, "edge_dim": edge_dim,
             "entries": entries, "source_identity": identity,
             "embedding_artifact": {"uri": str(embedding_path)}}
    read_verified_edge_source(cache, payload)
    for source, digest in list(sources.items()):
        _read(source, sources, digest)
    np.savez_compressed(embedding_path, **payload)
    cache["embedding_artifact"]["sha256"] = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    _publish_json(cache_path, cache)
    record = {"schema": "metafind.scene_semantics.v1", "status": "complete",
              "classification": "IMPLEMENTATION CHOICE", "sources": sources,
              "stage1_checkpoint_sha256": parent["sha256"], "stage2_checkpoint_sha256": child["sha256"],
              "requested_pairs": mappings, "distinct_required_keys": len(pairs),
              "generated_keys": missing, "generated_successes": len(new_good),
              "generated_degraded": len(missing) - len(new_good),
              "base_cache": str(Path(base_cache).resolve()) if base_cache is not None else None,
              "text_encoder_status": encoder_status, "text_encoder_identity": expected,
              "llm": {"model_id": str(n08.LLM_MODEL_PATH), "cache_key_name": meta["llm_model"],
                      "invoked": bool(missing),
                      "historical_weight_byte_equivalence": "UNKNOWN",
                      "binding": "existing n08 configured path and cache-key model name"},
              "sg2": {"max_attempts": n08.MAX_ATTEMPTS, "max_new_tokens": n08.MAX_NEW_TOKENS,
                      "producer_sha256": hashlib.sha256(Path(n08.__file__).read_bytes()).hexdigest()},
              "device": device,
              "cache": {"uri": str(cache_path), "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()},
              "embeddings": cache["embedding_artifact"],
              "sentences": {"uri": str(sentences_path),
                            "sha256": hashlib.sha256(sentences_path.read_bytes()).hexdigest()},
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _publish_json(out_dir / "record.json", record)
    return cache_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-texts", "required-pairs", "stage1-record", "stage2-record", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--base-cache", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(generate(args.asset_texts, args.required_pairs, args.stage1_record, args.stage2_record,
                   args.out_dir, variant=args.variant, base_cache=args.base_cache, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
