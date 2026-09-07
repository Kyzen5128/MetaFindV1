"""Isolated diagnostic: real ULIP/CLIP, n11b, Stage2.main, one update, strict restore.

Each phase runs in a separate process. No legacy permission, LLM, GPU, or writes
to the original corpus. Historical category relation sentences are explicit
diagnostic inputs, not the current canonical relation research protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
DATA = WORK / "data"
OUT = DATA / "outputs"
REPO = Path("/home/kyzen/MetaFindV1")
os.environ.update({"METAFIND_DATA": str(DATA), "METAFIND_TEXT_TEMPLATE": "v3_fit",
                   "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                   "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "PYTHONPATH": str(REPO)})
os.environ.setdefault("HF_HOME", "/home/kyzen/metafind/metafind_data/models/hf-cache")
os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"] + "/torch")
sys.path.insert(0, str(REPO))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"uri": str(path), "sha256": digest(path)}


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(WORK):
        raise ValueError(f"output outside diagnostic root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_prepared():
    snapshot = json.loads((WORK / "prepared_inputs.json").read_text())
    for rec in snapshot.values():
        if digest(rec["uri"]) != rec["sha256"]:
            raise ValueError(f"prepared input changed: {rec['uri']}")
    return snapshot


def cli(module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("CLI", command, flush=True)
    subprocess.run(command, cwd=REPO, env=os.environ, check=True)


def parent():
    from metafind.train.stage1 import load_stage1_model_config
    rec = json.loads((WORK / "parent_record.json").read_text())
    return load_stage1_model_config(rec["uri"], rec)


def bind(parent_record):
    """Metadata-only preparation. The parent must already be a completed run."""
    from metafind.train.stage1 import (load_stage1_model_config,
                                       effective_stage1_model_inputs, load_protocols)
    from metafind.models.resolve_stage1 import canonical_hyperparameter_hash, VARIANTS
    if (WORK / "parent_record.json").exists():
        raise ValueError("parent already bound; do not silently replace the diagnostic parent")
    original = json.loads(Path(parent_record).read_text())
    rec = load_stage1_model_config(original["uri"], original)
    enc, training, hp = effective_stage1_model_inputs(rec, {}, {}, {"values": {}})
    if enc["actual_clip_train_scope"] != "frozen":
        raise ValueError("this diagnostic encodes semantic text from a frozen parent CLIP")
    hp["sha256"] = canonical_hyperparameter_hash(hp["values"])
    training.update(status="resolved", hyperparameter_config_hash=hp["sha256"])
    enc["status"] = "resolved"
    for name, value in (("stage1_encoding_protocol.json", enc),
                        ("stage1_protocol.json", training),
                        ("stage1_hyperparameters.json", hp),
                        ("variant_registry.json", VARIANTS)):
        write(OUT / name, value)
    write(WORK / "parent_record.json", original)
    load_protocols()
    variant = "full" if training["train_scope"] == "point_encoder_and_fuser" else "fuser_only"
    if training["train_scope"] not in ("point_encoder_and_fuser", "fuser_only"):
        raise ValueError("unsupported diagnostic Stage1 train scope")
    write(WORK / "parent_binding.json", {"source_record": artifact(parent_record),
          "checkpoint": artifact(rec["uri"]), "variant": variant,
          "diagnostic_data_manifest": artifact(WORK / "source_manifest.json")})
    cli("metafind.models.resolve_stage2", "--decided-by", "isolated CPU diagnostic; existing resolved decisions")
    from metafind.train.stage2 import load_stage2_protocols, load_variant
    load_stage2_protocols()
    load_variant(variant, rec, training=training, encoding=enc, values=hp["values"])
    print("parent and protocols bound", variant, rec["sha256"], flush=True)


def semantics():
    """Re-encode exact existing sentences and canonical node text, no LLM."""
    import numpy as np
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.train.stage1 import stage1_backbone_kwargs
    from metafind.data.semantic_edges_run import encode_sentences, build_cache
    from metafind.data.semantic_provenance import (text_encoder_identity,
        build_node_source, build_edge_source, source_identity_sha256,
        read_verified_node_source, read_verified_edge_source)
    destination = OUT / "procthor_node_embeddings.json"
    if destination.exists():
        raise ValueError("semantic output already exists")
    node_map = json.loads((OUT / "procthor_object_text.json").read_bytes())
    settled = {r["key"]: r for r in map(json.loads,
               (OUT / "sem_edge_sentences.jsonl").read_text().splitlines())}
    edge_path, node_path = OUT / "sem_edge_embeddings.npz", OUT / "procthor_node_embeddings.npz"
    entries = build_cache(settled, str(edge_path))
    backbone = ULIPBackbone(BackboneConfig(device="cpu", train_scope="fuser_only",
                                          **stage1_backbone_kwargs(parent())))
    backbone.model.eval()
    encoder = text_encoder_identity(backbone)
    ns, es = build_node_source(node_map, encoder), build_edge_source(entries, encoder)
    nodes = encode_sentences([node_map[a]["text"] for a in ns["asset_ids"]], backbone=backbone)
    edges = encode_sentences([entries[k]["sentence"] for k in es["keys"]], backbone=backbone)
    del backbone
    gc.collect()
    verify_prepared()
    np.savez_compressed(node_path, ids=np.array(ns["asset_ids"]), embeddings=nodes,
                        source_identity_sha256=np.array(source_identity_sha256(ns)))
    np.savez_compressed(edge_path, keys=np.array(es["keys"]), embeddings=edges,
                        source_identity_sha256=np.array(source_identity_sha256(es)))
    meta = json.loads((OUT / "historical_semantic_source.json").read_text())
    nr = {**artifact(node_path), "asset_ids": ns["asset_ids"], "n_assets": len(nodes),
          "embedding_dim": int(nodes.shape[1]), "text_encoder_version": meta["text_encoder_version"],
          "source_identity": ns}
    er = {**meta, "entries": entries, "edge_dim": int(edges.shape[1]),
          "source_identity": es, "embedding_artifact": artifact(edge_path),
          "diagnostic": {"relation_protocol": "historical category descriptions",
                         "historical_llm_weight_bytes": "UNKNOWN",
                         "source_manifest": artifact(WORK / "source_manifest.json")}}
    write(destination, nr)
    write(OUT / "sem_edge_cache.json", er)
    with np.load(node_path, allow_pickle=False) as stored:
        nstatus = read_verified_node_source(nr, node_map, dict(stored))
    with np.load(edge_path, allow_pickle=False) as stored:
        estatus = read_verified_edge_source(er, dict(stored))
    write(WORK / "semantic_result.json", {"node": artifact(node_path), "edge": artifact(edge_path),
          "node_status": nstatus["status"], "edge_status": estatus["status"],
          "encoder_identity": encoder, "node_shape": list(nodes.shape), "edge_shape": list(edges.shape)})


def gallery():
    if (OUT / "stage2_gallery_index.json").exists():
        raise ValueError("gallery output already exists")
    cli("metafind.train.gallery_index", "stage2", "--device", "cpu",
        "--stage1-ckpt-record", WORK / "parent_record.json")
    cli("metafind.data.scene_splits", "--seed", "20260816")


def preflight():
    import numpy as np
    from metafind.train.stage2 import (load_stage2_protocols, Stage2Data, enumerate_samples,
        training_batches, capture_stage2_input_identity, verify_stage2_input_identity)
    from metafind.train.gallery_index import verified_stage2_index
    rec = parent()
    proto, edge, arch = load_stage2_protocols()
    index = json.loads((OUT / "stage2_gallery_index.json").read_text())
    ids, vectors, arrays = verified_stage2_index(index, rec["sha256"], tuple(proto["asset_modalities"]))
    data = Stage2Data("cpu", graph_unit=proto["graph_unit"])
    # n08 normalizes the same real frozen CLIP text vectors that n11b stores
    # raw. Batch shapes can change floating-point rounding, so compare at a
    # declared tolerance rather than pretending that this is bitwise arithmetic.
    expected_nodes = arrays["text"] / np.linalg.norm(arrays["text"], axis=1, keepdims=True)
    actual_nodes = np.stack([data.node_vectors[a] for a in ids])
    np.testing.assert_allclose(actual_nodes, expected_nodes, rtol=1e-5, atol=1e-6)
    houses = json.loads((OUT / "scene_splits.json").read_text())["train_houses"]
    samples = enumerate_samples(houses, set(ids), graph_unit=proto["graph_unit"])
    rng = np.random.default_rng(20260816)
    batches, dropped, dropped_samples = training_batches(samples, 8, rng)
    layout_draw = float(rng.random())
    assert len(houses) == len(batches) == 1 and len(samples) == 8
    assert dropped == dropped_samples == 0 and layout_draw >= .3
    identity = capture_stage2_input_identity(proto, edge, arch, houses)
    verify_stage2_input_identity({"input_identity": identity}, current_inputs=True)
    write(WORK / "preflight_result.json", {"samples": samples, "batches": batches,
          "layout_draw": layout_draw, "semantic_status": data.semantic_source_status,
          "gallery_shape": list(vectors.shape), "input_identity": identity,
          "node_vs_normalized_raw_gallery_text_max_abs_error": float(np.max(np.abs(actual_nodes-expected_nodes))),
          "source_status": "freshly encoded; strict validation; no legacy flag"})


def query_matrix(model, rec):
    import numpy as np
    import torch
    from metafind.train import stage2
    from metafind.train.gallery_index import verified_stage2_index
    index = json.loads((OUT / "stage2_gallery_index.json").read_text())
    ids, gallery_vectors, arrays = verified_stage2_index(
        index, rec["stage1_checkpoint_sha256"], ("text", "image"))
    data = stage2.Stage2Data("cpu", graph_unit="room")
    data.asset_vectors = stage2.load_asset_modality_vectors(arrays, ("text", "image"))
    houses = ["test_00757"]
    graphs = data.graphs_for(houses)
    samples = stage2.enumerate_samples(houses, set(ids), graph_unit="room")
    model.eval()
    with torch.no_grad():
        queries = torch.stack([stage2.encode_query(model, graphs[h], i, a, False, "cpu", data)
                               for h, i, a in samples]).float().numpy()
    assert np.isfinite(queries).all()
    return queries, samples


def train():
    """Observe actual AdamW and save seam; do not replace its math or trainer."""
    import numpy as np
    import torch
    from metafind.train import stage2
    binding = json.loads((WORK / "parent_binding.json").read_text())
    history = []
    original_step = torch.optim.AdamW.step
    original_payload = stage2.stage2_checkpoint_payload
    original_freeze = stage2.freeze_for_stage2
    parameter_names, forbidden_optimizer_ids, freeze_evidence = {}, set(), {}

    def observed_freeze(model, backbone, *args, **kwargs):
        result = original_freeze(model, backbone, *args, **kwargs)
        parameter_names.update({id(p): name for name, p in model.named_parameters()})
        forbidden_optimizer_ids.update(id(p) for p in backbone.model.parameters())
        forbidden_optimizer_ids.update(id(p) for p in model.gallery.parameters())
        freeze_evidence.update(backbone_is_frozen=backbone.is_frozen(),
            gallery_has_trainable_parameter=any(p.requires_grad for p in model.gallery.parameters()),
            gallery_training=model.gallery.training)
        assert freeze_evidence == {"backbone_is_frozen": True,
                                  "gallery_has_trainable_parameter": False, "gallery_training": False}
        return result

    def tensor_sha(p):
        return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def observed_step(optimizer, *args, **kwargs):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        assert not forbidden_optimizer_ids.intersection(id(p) for p in params)
        names = [parameter_names[id(p)] for p in params]
        assert all(name.startswith("query.") for name in names)
        grads = [p.grad for p in params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() for g in grads)
        before = [tensor_sha(p) for p in params]
        value = original_step(optimizer, *args, **kwargs)
        after = [tensor_sha(p) for p in params]
        changed = sum(a != b for a, b in zip(before, after))
        assert changed > 0
        components = {}
        for label, prefix in (("query_fusion", "query.fusion."),
                              ("layout_encoder", "query.layout_encoder."),
                              ("layout_weight", "query.layout_weight")):
            indices = [i for i, name in enumerate(names) if name.startswith(prefix)]
            used = [params[i].grad for i in indices if params[i].grad is not None]
            components[label] = {"optimizer_tensors": len(indices), "gradient_tensors": len(used),
                "finite_gradients": bool(used) and all(bool(torch.isfinite(g).all()) for g in used),
                "nonzero_gradient": any(bool(torch.count_nonzero(g).item()) for g in used),
                "changed_tensors": sum(before[i] != after[i] for i in indices)}
        history.append({"n_optimizer_tensors": len(params), "n_gradient_tensors": len(grads),
                        "all_gradients_finite": True, "changed_tensors": changed,
                        "learning_rates": [g["lr"] for g in optimizer.param_groups],
                        "parameter_names": names, "components": components,
                        "freeze_evidence": freeze_evidence,
                        "frozen_components_absent_from_optimizer": True,
                        "parameter_sha_before": before, "parameter_sha_after": after})
        write(WORK / "optimizer_observation.json", history)
        assert all(c["finite_gradients"] and c["nonzero_gradient"] and c["changed_tensors"] > 0
                   for c in components.values()), components
        return value

    def observed_payload(model, loss_fn, record):
        assert len(history) == record["steps"] == 1
        assert record["semantic_source_status"] == {"node": "verified", "edge": "verified", "encoder_match": "verified"}
        assert record["gallery_source_status"] == "verified"
        record["diagnostic"] = {"purpose": "one real CPU optimizer update; no benchmark result",
                               "source_manifest": artifact(WORK / "source_manifest.json"),
                               "driver": artifact(__file__)}
        queries, samples = query_matrix(model, record)
        np.savez_compressed(WORK / "before_save_queries.npz", queries=queries)
        write(WORK / "update_result.json", {"updates": history, "samples": samples,
                                            "query_shape": list(queries.shape)})
        return original_payload(model, loss_fn, record)

    torch.optim.AdamW.step = observed_step
    stage2.stage2_checkpoint_payload = observed_payload
    stage2.freeze_for_stage2 = observed_freeze
    old_argv = sys.argv
    sys.argv = ["stage2", "--variant", binding["variant"], "--device", "cpu",
                "--stage1-ckpt-record", str(WORK / "parent_record.json"),
                "--hyperparameters", str(WORK / "stage2_hyperparameters.json"),
                "--epochs", "1", "--query-modality-masking", "none"]
    try:
        rc = stage2.main()
        assert rc == 0 and len(history) == 1
    finally:
        torch.optim.AdamW.step = original_step
        stage2.stage2_checkpoint_payload = original_payload
        stage2.freeze_for_stage2 = original_freeze
        sys.argv = old_argv


def restore():
    import numpy as np
    import torch
    from metafind.train import stage1, stage2
    from metafind.train.gallery_index import (load_stage2_checkpoint_record, verify_gallery_encoder,
                                              verified_checkpoint_bytes)
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
    from metafind.eval.run_retrieval import overlay_stage2_weights
    binding = json.loads((WORK / "parent_binding.json").read_text())
    parent_rec = parent()
    rec = load_stage2_checkpoint_record(OUT / "variant_ckpts.json", parent_rec, binding["variant"])
    stage2.verify_stage2_input_identity(rec, current_inputs=True)
    enc, training, hp = stage1.effective_stage1_model_inputs(parent_rec, *stage1.load_protocols())
    bb = ULIPBackbone(BackboneConfig(device="cpu", train_scope=training["train_scope"],
                                    **stage1.stage1_backbone_kwargs(parent_rec)))
    bb_q = bb.clone_point_path() if training["tower_sharing"] == "fully_separate" else None
    dims = rec["layout_input_dims"]
    model = stage2.build_stage2_model(enc, training, rec["hyperparameters"], rec["arch_protocol"],
        node_feat_dim=dims["node_feat_dim"], edge_feat_dim=dims["edge_feat_dim"],
        use_layout=True, init_lambda=rec["lambda_init"]["init_lambda"])
    if training["freeze_gallery"]:
        model.freeze_gallery(True)
    _, parent_loss = stage1.build_model(enc, training, hp)
    stage1.load_stage1_checkpoint(bb, model, parent_loss, Path(parent_rec["uri"]),
         query_backbone=bb_q, new_prefixes=("query.layout_encoder", "query.layout_weight"))
    index = json.loads((OUT / "stage2_gallery_index.json").read_text())
    verify_gallery_encoder(index, bb, model, parent_checkpoint=parent_rec,
                           declared_modalities=("text", "image"))
    overlay_stage2_weights(model, rec, "cpu")
    import io
    saved = torch.load(io.BytesIO(verified_checkpoint_bytes(rec)), map_location="cpu", weights_only=False)
    state = model.state_dict()
    assert all(torch.equal(state[k], value) for k, value in saved["trainable_state"].items())
    queries, samples = query_matrix(model, rec)
    with np.load(WORK / "before_save_queries.npz") as expected:
        np.testing.assert_array_equal(queries, expected["queries"])
    write(WORK / "restore_result.json", {"checkpoint": artifact(rec["uri"]),
          "steps": rec["steps"], "saved_tensors": len(saved["trainable_state"]),
          "all_saved_tensors_exact": True, "query_vectors_bitwise_equal": True,
          "query_shape": list(queries.shape), "samples": samples,
          "input_identity_verified": True, "gallery_encoder_verified": True,
          "semantic_source_status": rec["semantic_source_status"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("bind", "semantics", "gallery", "preflight", "train", "restore"))
    parser.add_argument("--parent-record", type=Path)
    args = parser.parse_args()
    verify_prepared()
    started = time.time()
    environment = {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE",
                   "TRANSFORMERS_OFFLINE", "METAFIND_DATA", "METAFIND_TEXT_TEMPLATE", "HF_HOME")}
    write(WORK / f"{args.phase}_execution.json", {"phase": args.phase, "start": started,
          "environment": environment, "driver": artifact(__file__),
          "production_sources": {name: artifact(REPO / name) for name in (
              "metafind/train/stage2.py", "metafind/train/stage1.py", "metafind/train/gallery_index.py",
              "metafind/models/ulip_backbone.py", "metafind/models/fusion.py", "metafind/models/essgnn.py")}})
    if args.phase == "bind":
        if args.parent_record is None:
            raise ValueError("bind requires --parent-record")
        bind(args.parent_record)
    else:
        globals()[args.phase]()
    verify_prepared()
    print(json.dumps({"phase": args.phase, "status": "complete", "seconds": time.time()-started}), flush=True)


if __name__ == "__main__":
    main()
