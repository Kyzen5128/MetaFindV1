"""Encode explicit raw scene queries and freeze a replayable composition bundle.

This is an input bridge, not an implementation of the unpublished 200-scene
sampling protocol. It consumes prepared point clouds and existing n08 sentences;
it never fabricates absent relation evidence or calls the annotation producer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from metafind.data.semantic_provenance import (
    build_node_source, read_verified_node_source, read_verified_edge_source,
    text_encoder_identity,
)
from metafind.eval.custom_models import load_stage1
from metafind.scene.compose import (
    SCHEMA, MissingSemanticPairsError, _publish_json, _slot, export_query_model, run_manifest,
)
from metafind.train import gallery_index

REQUEST_SCHEMA = "metafind.scene_request.v1"
SEMANTIC_KEY_FIELDS = ("prompt_version", "llm_model", "text_encoder_version")


def _read(path, sources, expected=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"source hash mismatch: {path}")
    previous = sources.get(str(path))
    if previous is not None and previous != digest:
        raise ValueError(f"source changed during preparation: {path}")
    sources[str(path)] = digest
    return raw


def _json(path, sources, expected=None):
    return json.loads(_read(path, sources, expected))


def _arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
        return dict(stored)


def _relative(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def _vector_batch(value, rows, width, label):
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if value.shape != (rows, width) or not torch.isfinite(value).all():
        raise ValueError(f"{label} needs {rows} finite vectors of width {width}")
    if (value.norm(dim=1) == 0).any():
        raise ValueError(f"{label} contains a zero vector")
    return value


@torch.inference_mode()
def encode_query(query, loaded, base, sources):
    """Use the restored query point path and unchanged raw CLIP vectors.

    Mean over the explicitly supplied image views matches the supported S1
    pooled-image recipe. There is no hidden view selection, PC sampling, or
    extra retrieval-vector normalisation here.
    """
    if not isinstance(query, dict) or set(query) - {"slot", "text", "images", "pointcloud"}:
        raise ValueError("query accepts slot, text, images and pointcloud only")
    _slot(query["slot"])
    embeds = {}
    backbone, width = loaded.backbone, loaded.model.cfg.dim
    if "text" in query:
        if not isinstance(query["text"], str) or not query["text"].strip():
            raise ValueError("query text must be nonempty; omit an absent modality")
        embeds["text"] = _vector_batch(backbone.encode_text([query["text"]]), 1, width, "text")[0]
    if "images" in query:
        if not isinstance(query["images"], list) or not query["images"]:
            raise ValueError("query images must be an explicit nonempty ordered path list")
        if loaded.encoding["image_aggregation"] != "mean":
            raise ValueError("raw scene image encoding requires the recorded mean aggregation")
        views = []
        for path in query["images"]:
            raw = _read(_relative(path, base), sources)
            with Image.open(io.BytesIO(raw)) as image:
                batch = backbone.preprocess(image.convert("RGB"))[None]
            views.append(_vector_batch(backbone.encode_image(batch), 1, width, "image")[0])
        embeds["image"] = torch.stack(views).mean(0)
    if "pointcloud" in query:
        arrays = _arrays(_read(_relative(query["pointcloud"], base), sources))
        xyz, rgb = arrays["xyz"], arrays["rgb"]
        if any(a.shape != (10000, 3) or a.dtype != np.float32 or not np.isfinite(a).all()
               for a in (xyz, rgb)):
            raise ValueError("query PC requires prepared finite float32 xyz/rgb arrays (10000,3)")
        point_path = loaded.query_backbone if loaded.query_backbone is not None else backbone
        points = torch.from_numpy(np.concatenate((xyz, rgb), axis=1))[None]
        embeds["pc"] = _vector_batch(point_path.encode_pc(points), 1, width, "pc")[0]
    if not embeds:
        raise ValueError("query has no available raw modality")
    return {"slot": query["slot"], "embeds": embeds, "query_text": query.get("text")}


def _training_encoder(child, sources):
    """Recover producer evidence from the child-bound training artifacts."""
    artifacts = child["input_identity"]["artifacts"]
    def artifact(name):
        rec = artifacts[name]
        return _read(rec["uri"], sources, rec["sha256"])
    node, texts = json.loads(artifact("node_record")), json.loads(artifact("object_text"))
    node_arrays = _arrays(artifact("node_embeddings"))
    # The node record's own vector claim must identify these same bytes too.
    if Path(node["uri"]).resolve() != Path(artifacts["node_embeddings"]["uri"]).resolve() \
            or node["sha256"] != artifacts["node_embeddings"]["sha256"]:
        raise ValueError("training node record does not identify its bound embeddings")
    node_source = read_verified_node_source(node, texts, node_arrays)
    cache = json.loads(artifact("semantic_cache"))
    edge_arrays = _arrays(artifact("semantic_embeddings"))
    edge_artifact = cache.get("embedding_artifact", {})
    if edge_artifact != artifacts["semantic_embeddings"]:
        raise ValueError("training semantic record does not identify its bound embeddings")
    edge_source = read_verified_edge_source(cache, edge_arrays)
    if node_source["encoder_identity"] != edge_source["encoder_identity"]:
        raise ValueError("training node and edge text encoders differ")
    return node_source["encoder_identity"], {k: cache[k] for k in SEMANTIC_KEY_FIELDS}


@torch.inference_mode()
def semantic_inputs(child, backbone, asset_texts, cache_path, sources, *, node_dim, edge_dim):
    expected, training_meta = _training_encoder(child, sources)
    actual = text_encoder_identity(backbone)
    if actual != expected:
        raise ValueError("loaded text encoder differs from the Stage 2 semantic encoder")
    cache = _json(cache_path, sources)
    artifact = cache["embedding_artifact"]
    edge_arrays = _arrays(_read(_relative(artifact["uri"], Path(cache_path).parent), sources, artifact["sha256"]))
    proof = read_verified_edge_source(cache, edge_arrays)
    if proof["encoder_identity"] != expected:
        raise ValueError("scene semantic cache uses a different text encoder")
    meta = {k: cache[k] for k in SEMANTIC_KEY_FIELDS}
    if any(meta[k] != training_meta[k] for k in meta):
        raise ValueError("scene relation prompt/model/version differs from Stage 2")
    if edge_arrays["embeddings"].shape[1] != edge_dim:
        raise ValueError("scene semantic vectors differ from the ESSGNN edge width")
    edges = {key: torch.from_numpy(row.copy()) for key, row in
             zip(edge_arrays["keys"].tolist(), edge_arrays["embeddings"])}
    for key, entry in cache["entries"].items():
        if entry.get("degraded"):
            if entry.get("embedding_uri") is not None:
                raise ValueError("degraded semantic relation still claims an embedding")
            edges[key] = None
    node_source = build_node_source(asset_texts, actual)
    ids = node_source["asset_ids"]
    nodes = {}
    for start in range(0, len(ids), 128):
        batch = ids[start:start + 128]
        values = _vector_batch(backbone.encode_text([asset_texts[a]["text"] for a in batch]),
                               len(batch), node_dim, "node text")
        # n08 normalises node/edge semantics; raw query vectors above stay raw.
        values = F.normalize(values, dim=-1)
        nodes.update(zip(batch, values.unbind()))
    if text_encoder_identity(backbone) != actual:
        raise ValueError("semantic encoder changed during preparation")
    return nodes, edges, meta, node_source


def prepare(request_path: Path, out_dir: Path, *, device="cpu") -> Path:
    """Publish manifest.json only after an actual CPU composition replay passes."""
    request_path, out_dir = Path(request_path).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    sources = {}
    spec = _json(request_path, sources)
    required = {"schema", "provenance", "stage1_record", "stage2_record", "variant",
                "gallery_registry", "gallery_ids", "asset_texts", "initial_graph",
                "queries", "mode", "use_layout"}
    if not isinstance(spec, dict) or not required <= spec.keys() \
            or set(spec) - required - {"semantic_cache"} or spec["schema"] != REQUEST_SCHEMA:
        raise ValueError(f"scene request requires explicit fields {sorted(required)}; optional semantic_cache")
    if not isinstance(spec["provenance"], dict) or not spec["provenance"]:
        raise ValueError("scene request requires source provenance")
    if not isinstance(spec["use_layout"], bool) or spec["mode"] not in ("iterative", "parallel"):
        raise ValueError("scene request requires explicit layout boolean and iterative/parallel mode")
    if not isinstance(spec["queries"], list) or not spec["queries"]:
        raise ValueError("scene request needs nonempty ordered queries")
    base = request_path.parent
    path = lambda name: _relative(spec[name], base).resolve()
    for name in ("stage1_record", "stage2_record", "gallery_registry"):
        _read(path(name), sources)
    loaded = load_stage1(path("stage1_record"), device)
    child = gallery_index.load_stage2_checkpoint_record(
        path("stage2_record"), loaded.record, spec["variant"], allow_legacy=False)
    has_layout, _ = gallery_index.stage2_layout_settings(child)
    if spec["use_layout"] and not has_layout:
        raise ValueError("no_layout checkpoint cannot enable layout")
    record, ids, vectors = gallery_index.load_promoted_index_for_checkpoint(
        loaded.record["sha256"], path("gallery_registry"))
    gallery_index.verify_gallery_encoder(record, loaded.backbone, loaded.model,
                                         parent_checkpoint=loaded.record, allow_legacy=False)
    _read(record["uri"], sources, record["sha256"])
    selected = ids if spec["gallery_ids"] == "all" else spec["gallery_ids"]
    if not isinstance(selected, list) or not selected or any(not isinstance(x, str) for x in selected) \
            or len(set(selected)) != len(selected) or set(selected) - set(ids):
        raise ValueError("gallery_ids must be 'all' or an explicit nonempty unique admitted UID list")
    index = {uid: i for i, uid in enumerate(ids)}
    gallery = torch.from_numpy(vectors[[index[uid] for uid in selected]].copy())
    texts = _json(path("asset_texts"), sources)
    assets = set(selected) | {node["asset_id"] for node in spec["initial_graph"]["nodes"]}
    if not isinstance(texts, dict) or assets - texts.keys():
        raise ValueError("asset_texts must cover selected gallery and initial placed assets")
    texts = {uid: texts[uid] for uid in sorted(assets)}
    queries = [encode_query(q, loaded, base, sources) for q in spec["queries"]]
    model_path = export_query_model(path("stage1_record"), path("stage2_record"),
                                    out_dir / "model", variant=spec["variant"])
    model = json.loads(model_path.read_text())
    semantic_source = {"status": "not_used", "reason": "checkpoint_has_no_layout_branch"}
    nodes, edges, meta = {}, {}, {}
    if has_layout:
        if "semantic_cache" not in spec:
            raise ValueError("layout checkpoint requires an explicit prepared n08 semantic_cache")
        arch = model["config"]["essgnn"]
        nodes, edges, meta, semantic_source = semantic_inputs(
            child, loaded.backbone, texts, path("semantic_cache"), sources,
            node_dim=arch["node_feat_dim"], edge_dim=arch["edge_feat_dim"])
    inputs = {"gallery_ids": selected, "gallery": gallery,
              "asset_node_embeddings": nodes, "asset_texts": texts,
              "initial_graph": spec["initial_graph"], "queries": queries,
              "semantic_cache": edges, "semantic_meta": meta,
              "mode": spec["mode"], "use_layout": spec["use_layout"]}
    input_path = out_dir / "inputs.pt"
    with input_path.open("xb") as stream:
        torch.save(inputs, stream)
    model["weights"]["path"] = "model/query_tower.pt"
    manifest = {"schema": SCHEMA, "status": "frozen", "model": model,
                "inputs": {"path": "inputs.pt", "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()},
                "provenance": {"request": spec, "sources": sources, "semantic_source": semantic_source,
                               "gallery_record": record, "encoding_device": device,
                               "query_image_rule": "mean_raw_explicit_views",
                               "query_pc_rule": "caller_prepared_xyz_rgb_10000_no_resampling",
                               "prepare_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    # A candidate is intentionally not the published manifest: missing semantic
    # keys or invalid geometry must not leave a usable-looking frozen result.
    candidate = out_dir / "candidate.json"
    _publish_json(candidate, manifest)
    try:
        result = run_manifest(candidate)
    except MissingSemanticPairsError as exc:
        _publish_json(out_dir / "required_pairs.json", exc.pairs)
        _publish_json(out_dir / "missing_semantics.json", {
            "status": "incomplete", "pairs": exc.pairs, "keys": exc.keys,
            "request_sha256": sources[str(request_path)],
            "reason": "current trajectory requires relations not present in the supplied cache",
        })
        raise
    for source, digest in list(sources.items()):
        _read(source, sources, digest)
    dest = out_dir / "manifest.json"
    _publish_json(dest, manifest)
    _publish_json(out_dir / "validation_composition.json", result)
    candidate.unlink()
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(prepare(args.request, args.out_dir, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
