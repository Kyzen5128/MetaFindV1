"""Algorithm 1 over encoded inputs; no planner, model download, or placement search.

U-18/U-21: use I-Design slots, only already placed nodes, and the retrieved
asset's text features. n07 supplies kNN adjacency; n08 supplies semantic keys.
Missing keys fail; explicit None uses the existing U-30 learned missing token.
This implementation is not a claim that the author's 200-scene protocol is known.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from metafind.data.scene_graphs import ADJACENCY_K, _knn_pairs
from metafind.data.semantic_edges import cache_key, relation_text_for
from metafind.eval.retrieval import normalize_for_scoring
from metafind.models.dual_tower import DualTowerConfig, QueryTower
from metafind.models.essgnn import ESSGNNConfig
from metafind.models.fusion import FusionConfig, MODALITIES
from metafind.scene.placement import _number

SCHEMA = "metafind.scene_composition.v1"


class MissingSemanticPairsError(KeyError):
    """Unprepared relations in the current trajectory, never degraded edges."""
    def __init__(self, pairs, keys):
        self.pairs, self.keys = pairs, keys
        super().__init__(f"semantic cache lacks {len(pairs)} pair(s): {pairs}; prepare n08 evidence first")


def _vector(value, size: int, label: str) -> Tensor:
    value = torch.as_tensor(value).detach()
    if value.shape != (size,) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be a finite floating vector of width {size}")
    return value


def _slot(slot: dict) -> str:
    if not isinstance(slot, dict) or not isinstance(slot.get("new_object_id"), str) or not slot["new_object_id"].strip():
        raise ValueError("slot requires a nonempty I-Design new_object_id")
    try:
        # Preserve the planned values, and share the downstream placement
        # contract: float() accepted strings/bools that could never be placed.
        for axis in ("x", "y", "z"):
            _number(slot["position"][axis], "slot position")
        _number(slot["rotation"]["z_angle"], "slot yaw")
        for axis in ("length", "width", "height"):
            _number(slot["size_in_meters"][axis], "slot size", positive=True)
        relations = slot["placement"]["objects_in_room"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("slot requires explicit finite numeric position, rotation, positive size and placement relations") from exc
    if not isinstance(relations, list) or any(
        not isinstance(r, dict) or not isinstance(r.get("object_id"), str)
        or not isinstance(r.get("preposition"), str) for r in relations
    ):
        raise ValueError("slot placement.objects_in_room must contain explicit object relations")
    return slot["new_object_id"]


def _physical_graph(nodes):
    """The n07 physical-pair rule, independent of optional semantic encoders."""
    ids = [n["slot"]["new_object_id"] for n in nodes]
    index = {sid: i for i, sid in enumerate(ids)}
    positions = np.array([[float(n["slot"]["position"][a]) for a in ("x", "y", "z")]
                         for n in nodes], dtype=np.float64).reshape(-1, 3)
    support = set()
    for i, node in enumerate(nodes):
        for relation in node["slot"]["placement"]["objects_in_room"]:
            j = index.get(relation["object_id"])
            if j is not None and relation["preposition"] in ("on", "under"):
                support.add(tuple(sorted((i, j))))
    adjacency = set(_knn_pairs(positions, ADJACENCY_K)) - support
    pairs = sorted(support | adjacency)
    record = {"node_ids": ids, "asset_ids": [n["asset_id"] for n in nodes],
              "positions": positions.tolist(), "support": [list(p) for p in sorted(support)],
              "adjacency": [list(p) for p in sorted(adjacency)]}
    return positions, pairs, record


def _graph(nodes, asset_node_embeddings, asset_texts, semantic_cache, semantic_meta,
           cfg, device, dtype):
    """Build physical trace and, only for an existing branch, semantic tensors."""
    positions, pairs, record = _physical_graph(nodes)
    if cfg is None:
        return None, {**record, "semantic_status": "not_used",
                      "semantic_keys": [], "semantic_missing": []}
    edges, values, missing, keys = [], [], [], []
    pair_keys, absent_pairs, absent_keys = {}, [], []
    for i, j in pairs:
        a, b = (nodes[k]["asset_id"] for k in (i, j))
        key = cache_key(relation_text_for(asset_texts[a]), relation_text_for(asset_texts[b]),
                        semantic_meta["prompt_version"], semantic_meta["llm_model"],
                        semantic_meta["text_encoder_version"])
        pair_keys[i, j] = key
        if key not in semantic_cache:
            if key not in absent_keys:
                absent_pairs.append([a, b])
                absent_keys.append(key)
    if absent_pairs:
        raise MissingSemanticPairsError(absent_pairs, absent_keys)
    for i, j in pairs:
        key = pair_keys[i, j]
        vector = semantic_cache[key]
        absent = vector is None
        vector = (torch.zeros(cfg.edge_feat_dim) if absent else
                  _vector(vector, cfg.edge_feat_dim, "semantic edge"))
        for src, dst in ((i, j), (j, i)):
            edges.append((src, dst))
            values.append(vector.to(device=device, dtype=dtype))
            missing.append(absent)
            keys.append(key)
    features = torch.stack([asset_node_embeddings[n["asset_id"]].to(device=device, dtype=dtype)
                            for n in nodes]) if nodes else torch.empty((0, cfg.node_feat_dim), device=device, dtype=dtype)
    tensors = (features, torch.as_tensor(positions, device=device, dtype=dtype),
               torch.tensor(edges, device=device, dtype=torch.long).reshape(-1, 2).T,
               torch.stack(values) if values else torch.empty((0, cfg.edge_feat_dim), device=device, dtype=dtype),
               torch.tensor(missing, device=device, dtype=torch.bool))
    record.update(semantic_keys=keys, semantic_missing=missing)
    return tensors, record


@torch.inference_mode()
def compose(tower: QueryTower, *, gallery_ids: list[str], gallery: Tensor,
            asset_node_embeddings: dict[str, Tensor] | None = None, asset_texts: dict[str, dict],
            initial_graph: dict, queries: list[dict], semantic_cache: dict[str, Tensor | None] | None = None,
            semantic_meta: dict | None = None, mode: str = "iterative", use_layout: bool = True) -> dict:
    """Retrieve and place a fixed ordered query list, without mutating inputs.

    queries contain ``slot`` (I-Design object), ``embeds`` (available (D,)
    modality vectors), and optional ``query_text`` for traceability only.
    ``parallel`` reuses G0 for every query; weights and gallery stay identical.
    The caller supplies a loaded, eval-mode QueryTower. A tower without an
    ESSGNN branch requires use_layout=False and does not consume node/edge
    vectors or their cache metadata. A layout-trained tower keeps the existing
    semantic validation even when use_layout=False, preserving that control.
    """
    if mode not in ("iterative", "parallel"):
        raise ValueError("mode must be iterative or parallel")
    if not isinstance(use_layout, bool):
        raise ValueError("use_layout must be an explicit boolean")
    if any(m.training for m in tower.modules()):
        raise ValueError("composition requires the complete query tower in eval mode")
    has_layout = tower.layout_encoder is not None
    if use_layout and not has_layout:
        raise ValueError("use_layout=True requires a query tower with an ESSGNN branch")
    if has_layout != tower.cfg.use_layout or (has_layout and tower.cfg.essgnn is None):
        raise ValueError("query tower layout branch disagrees with its configuration")
    cfg = tower.cfg.essgnn if has_layout else None
    if tower.cfg.query_fusion.image_tokens != 1:
        raise ValueError("encoded composition currently requires image_tokens=1")
    if not gallery_ids or any(not isinstance(s, str) or not s.strip() for s in gallery_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("gallery IDs must be nonempty and unique")
    gallery = torch.as_tensor(gallery).detach()
    if gallery.shape != (len(gallery_ids), tower.cfg.dim) or not gallery.is_floating_point() or not torch.isfinite(gallery).all():
        raise ValueError("gallery must have one finite nonzero floating vector per ID")
    if not isinstance(initial_graph.get("room_id"), str) or not initial_graph["room_id"].strip():
        raise ValueError("initial_graph requires an explicit room_id")
    if not isinstance(initial_graph.get("nodes"), list) or not isinstance(queries, list) or not queries:
        raise ValueError("initial nodes and a nonempty ordered query list are required")
    nodes = copy.deepcopy(initial_graph["nodes"])
    slots = [n["slot"] for n in nodes] + [q["slot"] for q in queries]
    slot_ids = [_slot(slot) for slot in slots]
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("initial/query slot IDs must be unique; future nodes cannot be in G0")
    for slot in slots:
        for relation in slot["placement"]["objects_in_room"]:
            if relation["object_id"] not in slot_ids or relation["object_id"] == slot["new_object_id"]:
                raise ValueError("placement relation references an unknown or identical slot")
    required_assets = set(gallery_ids) | {n["asset_id"] for n in nodes}
    node_vectors = {}
    for asset in required_assets:
        if asset not in asset_texts or (has_layout and (
                asset_node_embeddings is None or asset not in asset_node_embeddings)):
            raise ValueError(f"missing retrieved-asset node semantics: {asset}")
        if has_layout:
            node_vectors[asset] = _vector(asset_node_embeddings[asset], cfg.node_feat_dim, f"node {asset}")
        if not isinstance(asset_texts[asset].get("text"), str) or not asset_texts[asset]["text"].strip():
            raise ValueError(f"asset {asset} requires its canonical node text")
        if has_layout:
            relation_text_for(asset_texts[asset])
    if has_layout and (semantic_meta is None or set(semantic_meta) != {
            "prompt_version", "llm_model", "text_encoder_version"}):
        raise ValueError("semantic_meta requires the three original n08 key fields")
    if has_layout and semantic_cache is None:
        raise ValueError("semantic_cache is required for an ESSGNN query tower")
    prepared = []
    parameter = next(tower.parameters())
    device, dtype = parameter.device, parameter.dtype
    for query in queries:
        embeds = query["embeds"]
        if not isinstance(embeds, dict) or set(embeds) - set(MODALITIES) or not any(v is not None for v in embeds.values()):
            raise ValueError("each query needs explicit available text/image/pc embeddings")
        prepared.append({m: (_vector(embeds[m], tower.cfg.dim, f"query {m}").to(device=device, dtype=dtype)[None]
                             if embeds.get(m) is not None else None) for m in MODALITIES})
    # Scoring copies to CPU float64; the cached gallery is never rewritten.
    normalized_gallery = normalize_for_scoring(gallery.to(device="cpu", dtype=torch.float64).numpy())
    initial_nodes = copy.deepcopy(nodes)
    trace = []
    for step, (query, embeds) in enumerate(zip(queries, prepared)):
        context = nodes if mode == "iterative" else initial_nodes
        tensors, graph_record = _graph(context, node_vectors, asset_texts,
                                      semantic_cache, semantic_meta, cfg, device, dtype)
        layout = tower.encode_layout(*tensors[:4], edge_missing=tensors[4]) if context and use_layout else None
        fused = tower.fusion(embeds)
        encoded = tower(embeds, layout=layout)
        if not torch.isfinite(encoded).all():
            raise ValueError("query tower produced a zero or nonfinite vector")
        scores = normalized_gallery @ normalize_for_scoring(encoded.to(device="cpu", dtype=torch.float64).numpy())[0]
        # Stable ties preserve the explicit gallery order, recorded below.
        order = np.argsort(-scores, kind="stable")[:min(5, len(gallery_ids))]
        selected = gallery_ids[int(order[0])]
        trace.append({"step": step, "slot_id": query["slot"]["new_object_id"],
                      "query_text": query.get("query_text"), "context": graph_record,
                      "query_embedding": encoded[0].cpu().tolist(),
                      "selected_asset_id": selected, "selected_node_text": asset_texts[selected]["text"],
                      "top5": [{"asset_id": gallery_ids[int(i)], "score": float(scores[i])} for i in order],
                      "lambda": float(tower.lam) if has_layout else 0., "fused_norm": float(fused.norm()),
                      "layout_norm": float(layout.norm()) if layout is not None else 0.,
                      "lambda_layout_norm": float((tower.lam * layout).norm()) if layout is not None else 0.})
        nodes.append({"asset_id": selected, "slot": copy.deepcopy(query["slot"])})
    _, final_edges = _graph(nodes, node_vectors, asset_texts, semantic_cache,
                            semantic_meta, cfg, device, dtype)
    result = {"schema": SCHEMA, "status": "complete", "classification": "IMPLEMENTATION CHOICE",
            "mode": mode, "use_layout": use_layout, "room_id": initial_graph["room_id"], "gallery_ids": list(gallery_ids),
            "tie_policy": "stable_gallery_order", "adjacency_k": ADJACENCY_K,
            "trace": trace, "final_graph": {"nodes": nodes, **final_edges}}
    if not has_layout:
        result["semantic_status"] = "not_used"
    return result


def _read_artifact(record: dict, base: Path) -> bytes:
    path = Path(record["path"])
    if not path.is_absolute():
        path = base / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return raw


def _complete_config(cls, value: dict, label: str) -> dict:
    expected = {f.name for f in fields(cls)}
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label} requires complete dataclass fields; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return copy.deepcopy(value)


def _float32_state(state: dict) -> None:
    if not isinstance(state, dict) or any(not isinstance(t, Tensor) for t in state.values()):
        raise ValueError("model state must be a tensor mapping")
    for name, tensor in state.items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"model supports float32 state only; {name} has {tensor.dtype}")
        if tensor.is_complex() or not torch.isfinite(tensor).all():
            raise ValueError(f"model state is complex or nonfinite: {name}")


def _publish_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(part, path)
    part.unlink()


def run_manifest(path: Path) -> dict:
    """CPU entry for a frozen JSON manifest and weights-only PyTorch bundles.

    model.config is dataclasses.asdict(QueryTower.cfg); model.weights contains
    the complete QueryTower.state_dict(). inputs contains compose keyword args.
    Both external artifacts require path+sha256; provenance is mandatory.
    """
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "frozen" or not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ValueError("manifest must be frozen, versioned, and contain source provenance")
    if manifest["model"].get("dtype") != "float32":
        raise ValueError("manifest model.dtype must explicitly be float32")
    config = _complete_config(DualTowerConfig, manifest["model"]["config"], "DualTowerConfig")
    if config["essgnn"] is None:
        if config["use_layout"] is not False:
            raise ValueError("model use_layout=True requires a complete ESSGNNConfig")
    else:
        config["essgnn"] = ESSGNNConfig(**_complete_config(ESSGNNConfig, config["essgnn"], "ESSGNNConfig"))
    for key in ("query_fusion", "gallery_fusion"):
        config[key] = FusionConfig(**_complete_config(FusionConfig, config[key], key))
    weights = torch.load(io.BytesIO(_read_artifact(manifest["model"]["weights"], path.parent)),
                         map_location="cpu", weights_only=True)
    _float32_state(weights)
    tower = QueryTower(DualTowerConfig(**config)).to(device="cpu", dtype=torch.float32).eval()
    tower.load_state_dict(weights, strict=True)
    inputs = torch.load(io.BytesIO(_read_artifact(manifest["inputs"], path.parent)),
                        map_location="cpu", weights_only=True)
    result = compose(tower, **inputs)
    result["provenance"] = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                            "manifest": manifest, "device": "cpu",
                            "compose_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def export_query_model(stage1_record_path: Path, stage2_record_path: Path,
                       out_dir: Path, *, variant: str = "full") -> Path:
    """Export complete query fusion/ESSGNN state from existing records, no backbone.

    This exports model.json only. Raw queries, gallery vectors, node embeddings
    and semantic caches remain separately supplied and provenance-bound inputs.
    """
    from metafind.train import gallery_index, stage1
    from metafind.eval.run_retrieval import load_stage2_over_stage1, overlay_stage2_weights

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    paths = [Path(stage1_record_path), Path(stage2_record_path)]
    record_bytes = [p.read_bytes() for p in paths]
    parent = gallery_index.load_checkpoint_record(paths[0])
    parent_bytes = gallery_index.verified_checkpoint_bytes(parent)
    parent_payload = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=False)
    parent_state = parent_payload["tower_trainable_state"]
    _float32_state(parent_state)
    loaded = load_stage2_over_stage1(str(paths[1]), parent, variant, allow_legacy_inputs=False)
    model, child = loaded["model"], loaded["record"]
    if model.fusion_is_tied():
        raise ValueError("composition export requires separate query/gallery fusion")
    child_bytes = gallery_index.verified_checkpoint_bytes(child)
    child_payload = torch.load(io.BytesIO(child_bytes), map_location="cpu", weights_only=False)
    _float32_state(child_payload["trainable_state"])
    stage1.validate_stage1_forward_config(model, parent_payload["metadata"])
    # Restore from the very bytes verified above, never reopen the S1 path.
    stage1.load_stage1_tower_state(model, parent_state,
                                  new_prefixes=("query.layout_encoder", "query.layout_weight"))
    overlay_stage2_weights(model, child, "cpu", fusion_only=False)
    model.query.to(device="cpu", dtype=torch.float32).requires_grad_(False).eval()
    state = model.query.state_dict()
    _float32_state(state)
    if any(p.read_bytes() != raw for p, raw in zip(paths, record_bytes)):
        raise ValueError("checkpoint record changed during model export")
    out_dir.mkdir(parents=True, exist_ok=False)
    weight_path = out_dir / "query_tower.pt"
    with weight_path.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    provenance = {"scope": "query-model-only; no backbone or input encoding",
                  "stage1_checkpoint": {"path": parent["uri"], "sha256": hashlib.sha256(parent_bytes).hexdigest()},
                  "stage2_checkpoint": {"path": child["uri"], "sha256": hashlib.sha256(child_bytes).hexdigest()},
                  "records": [{"path": str(p.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                               "content": json.loads(raw)} for p, raw in zip(paths, record_bytes)],
                  "compose_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "torch_version": str(torch.__version__)}
    result = {"config": asdict(model.query.cfg), "dtype": "float32",
              "weights": {"path": str(weight_path.resolve()),
                          "sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest()},
              "provenance": provenance}
    dest = out_dir / "model.json"
    _publish_json(dest, result)
    return dest


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export-model":
        parser = argparse.ArgumentParser(description="Export verified S1/S2 query model; no backbone")
        parser.add_argument("--stage1-record", required=True, type=Path)
        parser.add_argument("--stage2-record", required=True, type=Path)
        parser.add_argument("--out-dir", required=True, type=Path)
        parser.add_argument("--variant", default="full")
        args = parser.parse_args(argv[1:])
        path = export_query_model(args.stage1_record, args.stage2_record, args.out_dir, variant=args.variant)
        print(f"query model exported -> {path}", flush=True)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    result = run_manifest(args.manifest)
    _publish_json(args.out, result)
    print(f"composed {len(result['trace'])} slots -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
