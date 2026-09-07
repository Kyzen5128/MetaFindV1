"""Bind n08 text inputs and the loaded text encoder to its cached vectors.

File SHA verification remains the caller's responsibility. These pure checks
prove the producer/consumer relationship that hashing a sidecar alone cannot.
Legacy artifacts have no such evidence and are never labelled verified.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from importlib.metadata import version

import numpy as np


def source_identity_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _implementation(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def text_encoder_identity(backbone) -> dict:
    """Fingerprint the *loaded* OpenCLIP text path, without loading a model.

    Vision parameters, point parameters, logit scale and the tokenizer's mutable
    BPE result cache do not participate in text encoding and are excluded.
    Nonpersistent buffers (notably the attention mask) do participate.
    """
    import torch

    clip = backbone.model.open_clip_model
    tokenizer = backbone.tokenizer
    if any(module.training for name, module in clip.named_modules()
           if not name.startswith("visual")):
        raise ValueError("semantic text encoder must be in eval mode")
    fields = ("encoder", "bpe_ranks", "byte_encoder", "context_length",
              "all_special_ids", "sot_token_id", "eot_token_id", "pat", "clean_fn")
    if any(not hasattr(tokenizer, key) for key in fields):
        raise ValueError("semantic provenance requires an inspectable OpenCLIP SimpleTokenizer")
    if getattr(tokenizer, "reduction_fn", None) is not None:
        raise ValueError("stochastic/reduction tokenizers are not supported for semantic caches")
    token_config = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "implementation_sha256": _implementation(type(tokenizer)),
        "encoder": sorted(tokenizer.encoder.items()),
        "bpe_ranks": sorted((list(k), v) for k, v in tokenizer.bpe_ranks.items()),
        "byte_encoder": sorted(tokenizer.byte_encoder.items()),
        "context_length": tokenizer.context_length,
        "all_special_ids": list(tokenizer.all_special_ids),
        "sot_token_id": tokenizer.sot_token_id, "eot_token_id": tokenizer.eot_token_id,
        "pattern": tokenizer.pat.pattern, "pattern_flags": tokenizer.pat.flags,
        "clean_fn_sha256": _implementation(tokenizer.clean_fn),
        "reduction_fn": None,
    }
    # Hash tokenizer helpers too: e.g. cleaning delegates to basic_clean.
    token_module = inspect.getmodule(type(tokenizer))
    token_config["module_sha256"] = _implementation(token_module)
    digest = hashlib.sha256()
    count = 0
    for label, values in (("parameter", clip.named_parameters()),
                          ("buffer", clip.named_buffers())):
        for name, tensor in sorted(values):
            if name.startswith("visual.") or name in {"logit_scale", "logit_bias"}:
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([label, name, str(value.dtype), list(value.shape)],
                                     separators=(",", ":")).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
    if not count:
        raise ValueError("text encoder has no inspectable text parameters/buffers")
    architecture = {
        "modules": {name: repr(module) for name, module in clip.named_children()
                    if name != "visual"},
        "options": {name: getattr(clip, name, None) for name in
                    ("context_length", "vocab_size", "text_pool_type", "text_eos_id")},
    }
    implementation = {
        "class": f"{type(clip).__module__}.{type(clip).__qualname__}",
        "open_clip_version": version("open_clip_torch"),
        "torch_version": torch.__version__,
        "clip_module_sha256": _implementation(inspect.getmodule(type(clip))),
        "text_forward_sha256": _implementation(type(clip).encode_text),
        "wrapper_forward_sha256": _implementation(type(backbone).encode_text),
        "ulip_forward_sha256": _implementation(type(backbone.model).encode_text),
    }
    # Child transformer/activation implementation can change without CLIP's
    # own forward changing. Record their source modules once, independent of
    # module traversal order or the tokenizer's growing result cache.
    modules = {type(m).__module__: inspect.getmodule(type(m))
               for name, m in clip.named_modules()
               if name and not name.startswith("visual")}
    implementation["text_module_sha256"] = {name: _implementation(module)
                                            for name, module in sorted(modules.items())}
    return {"version": 1, "kind": "open_clip_loaded_text_tower",
            "state_sha256": digest.hexdigest(),
            "tokenizer_sha256": source_identity_sha256(token_config),
            "architecture_sha256": source_identity_sha256(architecture),
            "implementation": implementation,
            "output": {"normalization": "l2", "dtype": "float32"}}


def _validate_encoder(identity: dict) -> None:
    if not isinstance(identity, dict) or identity.get("version") != 1 \
            or identity.get("kind") != "open_clip_loaded_text_tower":
        raise ValueError("unsupported semantic text encoder identity")
    for key in ("state_sha256", "tokenizer_sha256", "architecture_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"semantic encoder identity lacks valid {key}")
    if not isinstance(identity.get("implementation"), dict) \
            or not identity["implementation"] \
            or identity.get("output") != {"normalization": "l2", "dtype": "float32"}:
        raise ValueError("semantic encoder implementation/output identity is incomplete")


def build_node_source(text_map: dict, encoder_identity: dict) -> dict:
    _validate_encoder(encoder_identity)
    if not isinstance(text_map, dict) or not text_map:
        raise ValueError("node text map must be nonempty")
    if any(not isinstance(a, str) or not a for a in text_map):
        raise ValueError("node asset IDs must be nonempty strings")
    ids = sorted(text_map)
    rows = []
    for asset_id in ids:
        rec = text_map[asset_id]
        if not isinstance(rec, dict) or not isinstance(rec.get("text"), str) \
                or not rec["text"].strip():
            raise ValueError(f"node {asset_id} lacks exact nonempty text")
        rows.append([asset_id, rec["text"]])
    return {"version": 1, "kind": "node_text", "asset_ids": ids,
            "texts_sha256": source_identity_sha256({"rows": rows}),
            "encoder_identity": encoder_identity}


def build_edge_source(entries: dict, encoder_identity: dict) -> dict:
    _validate_encoder(encoder_identity)
    rows = []
    for key, entry in sorted(entries.items()):
        if entry.get("degraded"):
            continue
        if not isinstance(entry.get("sentence"), str) or not entry["sentence"].strip():
            raise ValueError(f"semantic edge {key} lacks its encoded sentence")
        rows.append([key, entry["sentence"]])
    return {"version": 1, "kind": "edge_sentence", "keys": [r[0] for r in rows],
            "texts_sha256": source_identity_sha256({"rows": rows}),
            "encoder_identity": encoder_identity}


def _verify(record, arrays, expected, row_key, *, allow_legacy):
    identity = record.get("source_identity")
    embedded = arrays.get("source_identity_sha256")
    if identity is None and embedded is None:
        if allow_legacy:
            return {"status": "legacy_unbound", "source_identity": None,
                    "encoder_identity": None}
        raise ValueError("semantic cache has no bound source identity; rerun n08 or explicitly "
                         "allow legacy semantic inputs")
    if not isinstance(identity, dict) or identity.get("version") != 1 \
            or identity.get("kind") != expected["kind"]:
        raise ValueError("unsupported semantic source identity")
    if embedded is None or np.asarray(embedded).shape != () \
            or str(np.asarray(embedded).item()) != source_identity_sha256(identity):
        raise ValueError("semantic source identity is not bound to its vector bytes")
    if identity != expected:
        raise ValueError("semantic source text or membership changed since encoding")
    ids = np.asarray(arrays[row_key])
    vectors = np.asarray(arrays["embeddings"])
    expected_ids = expected["asset_ids" if row_key == "ids" else "keys"]
    if ids.ndim != 1 or ids.tolist() != expected_ids \
            or len(set(ids.tolist())) != len(ids):
        raise ValueError("semantic vector row identities do not match their source")
    if vectors.ndim != 2 or len(vectors) != len(ids) or vectors.shape[1] == 0 \
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("semantic vectors must be finite float32 rows matching their source")
    return {"status": "verified", "source_identity": identity,
            "encoder_identity": identity["encoder_identity"]}


def read_verified_node_source(record: dict, current_text_map: dict, arrays,
                              allow_legacy: bool = False) -> dict:
    identity = record.get("source_identity")
    if identity is None and arrays.get("source_identity_sha256") is None:
        return _verify(record, arrays, {"kind": "node_text"}, "ids",
                       allow_legacy=allow_legacy)
    if not isinstance(identity, dict):
        raise ValueError("unsupported semantic node source identity")
    expected = build_node_source(current_text_map, identity.get("encoder_identity"))
    result = _verify(record, arrays, expected, "ids", allow_legacy=allow_legacy)
    if "asset_ids" in record and record["asset_ids"] != expected["asset_ids"] \
            or "n_assets" in record and record["n_assets"] != len(expected["asset_ids"]):
        raise ValueError("node record membership disagrees with its bound source")
    return result


def read_verified_edge_source(record: dict, arrays, allow_legacy: bool = False) -> dict:
    identity = record.get("source_identity")
    if identity is None and arrays.get("source_identity_sha256") is None:
        return _verify(record, arrays, {"kind": "edge_sentence"}, "keys",
                       allow_legacy=allow_legacy)
    if not isinstance(identity, dict):
        raise ValueError("unsupported semantic edge source identity")
    expected = build_edge_source(record["entries"], identity.get("encoder_identity"))
    result = _verify(record, arrays, expected, "keys", allow_legacy=allow_legacy)
    for row, key in enumerate(expected["keys"]):
        uri = record["entries"][key].get("embedding_uri", "")
        if not isinstance(uri, str) or uri.rsplit("#", 1)[-1] != str(row):
            raise ValueError("semantic edge row pointer does not match its bound key order")
        artifact = record.get("embedding_artifact")
        if artifact is not None and uri.rsplit("#", 1)[0] != artifact.get("uri"):
            raise ValueError("semantic edge row pointer names a different embedding artifact")
    return result
