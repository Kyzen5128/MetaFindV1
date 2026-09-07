"""Stage 2 producer/consumer regressions with tiny, local artifacts."""

import hashlib
import json

import numpy as np
import pytest

from metafind.train import stage2 as s2


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    graphs = tmp_path / "scene_graphs"
    modalities = tmp_path / "modalities"
    graphs.mkdir()
    modalities.mkdir()
    monkeypatch.setattr(s2.paths, "OUTPUTS", tmp_path)
    monkeypatch.setattr(s2.paths, "SCENE_GRAPHS", graphs)
    monkeypatch.setattr(s2.paths, "PROCTHOR_MODALITIES", modalities)

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    text = {a: {"text": f"node description {a}", "relation_text": f"relation {a}"}
            for a in ("A", "B", "C")}
    write("procthor_object_text.json", text)
    for a in text:
        (modalities / f"{a}.json").write_text(json.dumps({"asset_id": a}))
    graph = {"graph_unit": "room", "builder_version": 2,
             "nodes": [{"index": i, "asset_id": a, "room_id": "room|0",
                        "position": [float(i), 0, 0]} for i, a in enumerate(text)],
             "sem_edge_ids": [[0, 1]]}
    (graphs / "h.json").write_text(json.dumps(graph))
    nodes = tmp_path / "nodes.npz"
    node_vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(nodes, ids=np.array(list(text)), embeddings=node_vectors)
    write("procthor_node_embeddings.json", {
        "uri": str(nodes), "embedding_dim": 4,
        "sha256": hashlib.sha256(nodes.read_bytes()).hexdigest()})
    from metafind.data.semantic_edges import cache_key
    key = cache_key("relation A", "relation B", 1, "m", "v")
    np.savez(tmp_path / "sem_edge_embeddings.npz", keys=np.array([key]),
             embeddings=np.array([[1, 2, 3, 4]], dtype=np.float32))
    write("sem_edge_cache.json", {
        "edge_dim": 4, "prompt_version": 1, "llm_model": "m", "text_encoder_version": "v",
        "entries": {key: {"embedding_uri": str(tmp_path / "sem_edge_embeddings.npz") + "#0"}}})
    protocols = ({"graph_unit": "room"}, {"directionality": "symmetric"},
                 {"pooling": "normalised_sum"})
    for name, protocol in zip(("stage2_protocol.json", "essgnn_edge_protocol.json",
                               "essgnn_arch_protocol.json"), protocols):
        write(name, protocol)
    write("scene_splits.json", {"train_houses": ["h"]})
    write("stage2_positive_map.json", {a: a for a in text})
    write("stage2_gallery_index.json", {"sha256": "gallery"})
    return tmp_path, protocols, graph, node_vectors


def test_stage2_uses_relation_cache_keys_and_preserves_node_vectors(inputs):
    _, _, graph, nodes = inputs
    data = s2.Stage2Data("cpu", graph_unit="room")
    np.testing.assert_array_equal(data.node_vectors["A"], nodes[0])
    _, _, _, attrs, missing = s2.build_context_graph(
        graph, 2, data.edge_dim, data.sem_cache, data.text_map)
    assert not missing.any()
    np.testing.assert_array_equal(attrs, [[1, 2, 3, 4], [1, 2, 3, 4]])


def test_scope_is_checked_during_enumeration_and_graph_loading(inputs):
    root, _, graph, _ = inputs
    graph.pop("graph_unit")
    graph["builder_version"] = 1
    (root / "scene_graphs" / "h.json").write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="graph_unit"):
        s2.enumerate_samples(["h"], {"A"})
    data = s2.Stage2Data("cpu", graph_unit="room")
    with pytest.raises(ValueError, match="graph_unit"):
        data.graphs_for(["h"])
    assert s2.enumerate_samples(["h"], {"A"}, graph_unit="house") == [("h", 0, "A")]
    data.graph_unit = "house"
    assert data.graphs_for(["h"])["h"] == graph


@pytest.mark.parametrize("filename", ["scene_splits.json", "procthor_object_text.json",
    "sem_edge_cache.json", "sem_edge_embeddings.npz", "nodes.npz",
    "scene_graphs/h.json", "essgnn_arch_protocol.json", "stage2_protocol.json"])
def test_replay_refuses_replaced_input_bytes(inputs, filename):
    root, protocols, _, _ = inputs
    identity = s2.capture_stage2_input_identity(*protocols, ["h"])
    record = {"input_identity": identity}
    assert s2.verify_stage2_input_identity(record) == identity
    path = root / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since it was recorded"):
        s2.verify_stage2_input_identity(record)


def test_replay_refuses_added_modality_records(inputs):
    root, protocols, _, _ = inputs
    identity = s2.capture_stage2_input_identity(*protocols, ["h"])
    (root / "modalities" / "D.json").write_text('{"asset_id": "D"}')
    with pytest.raises(ValueError, match="membership changed"):
        s2.verify_stage2_input_identity({"input_identity": identity})


def test_verified_archived_inputs_cannot_authorize_a_different_current_root(inputs, monkeypatch):
    import shutil

    root, protocols, _, _ = inputs
    identity = s2.capture_stage2_input_identity(*protocols, ["h"])
    record = {"input_identity": identity}
    assert s2.verify_stage2_input_identity(record, current_inputs=True) == identity
    other = root / "other_outputs"
    other.mkdir()
    shutil.copyfile(root / "procthor_node_embeddings.json", other / "procthor_node_embeddings.json")
    monkeypatch.setattr(s2.paths, "OUTPUTS", other)
    # The archived root is intact, but it is no longer what Stage2Data would load.
    assert s2.verify_stage2_input_identity(record) == identity
    with pytest.raises(ValueError, match="current input path differs"):
        s2.verify_stage2_input_identity(record, current_inputs=True)


def test_legacy_record_is_not_claimed_to_have_verified_inputs():
    with pytest.raises(ValueError, match="legacy replay must be selected explicitly"):
        s2.verify_stage2_input_identity({"sha256": "old"})


def test_checkpoint_embeds_the_protocols_and_input_identity(inputs, tmp_path):
    import torch

    _, protocols, _, _ = inputs
    identity = s2.capture_stage2_input_identity(*protocols, ["h"])
    record = {"variant_id": "full", "input_identity": identity,
              "graph_unit": "room", "stage2_protocol": protocols[0],
              "edge_protocol": protocols[1], "arch_protocol": protocols[2],
              "layout_input_dims": {"node_feat_dim": 4, "edge_feat_dim": 4}}
    checkpoint = tmp_path / "stage2.pt"
    torch.save(s2.stage2_checkpoint_payload(torch.nn.Linear(4, 4),
                                           torch.nn.Linear(1, 1), record), checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    assert restored["metadata"] == record
    assert s2.verify_stage2_input_identity(restored["metadata"]) == identity


def test_archived_checkpoint_location_can_change_but_bytes_and_recipe_cannot(inputs):
    import shutil
    import torch
    from metafind.train.gallery_index import load_stage2_checkpoint_record

    root, protocols, _, _ = inputs
    checkpoint = root / "stage2.pt"
    record = {"variant_id": "full", "uri": str(checkpoint),
              "sha256": "unfinished", "size_bytes": 0,
              "stage1_checkpoint_sha256": "parent",
              "lambda_init": {"init_lambda": 1.0}, "graph_unit": "room",
              "input_identity": s2.capture_stage2_input_identity(*protocols, ["h"]),
              "stage2_protocol": protocols[0], "edge_protocol": protocols[1],
              "arch_protocol": protocols[2],
              "layout_input_dims": {"node_feat_dim": 4, "edge_feat_dim": 4}}
    payload = s2.stage2_checkpoint_payload(torch.nn.Linear(4, 4), torch.nn.Linear(1, 1), record)
    assert not {"uri", "sha256", "size_bytes"} & payload["metadata"].keys()
    torch.save(payload, checkpoint)
    record["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record["size_bytes"] = checkpoint.stat().st_size
    archived = root / "archive" / "stage2.pt"
    archived.parent.mkdir()
    shutil.move(checkpoint, archived)
    record["uri"] = str(archived)
    sidecar = root / "variant_ckpts.json"
    sidecar.write_text(json.dumps({"full": record}))
    parent = {"sha256": "parent"}
    assert load_stage2_checkpoint_record(sidecar, parent)["uri"] == str(archived)

    for field, changed in (("graph_unit", "house"),
                           ("lambda_init", {"init_lambda": 2.0}),
                           ("arch_protocol", {"pooling": "mean"})):
        sidecar.write_text(json.dumps({"full": {**record, field: changed}}))
        with pytest.raises(ValueError, match="checkpoint and record disagree"):
            load_stage2_checkpoint_record(sidecar, parent)
    sidecar.write_text(json.dumps({"full": record}))
    archived.write_bytes(archived.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match the sha256"):
        load_stage2_checkpoint_record(sidecar, parent)


def test_stage2_keeps_the_parents_distinct_gallery_constructor(monkeypatch):
    from dataclasses import asdict
    from metafind.models.fusion import FusionConfig
    from metafind.models import ulip_backbone

    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 8)
    query = FusionConfig(dim=8, hidden=16, n_heads=2, n_layers=1,
                         prefusion_norm=True)
    gallery = FusionConfig(dim=8, kind="mean", prefusion_norm=False)
    training = {"fusion": "transformer", "tower_sharing": "fully_separate",
                "query_fusion_config": asdict(query), "gallery_fusion_config": asdict(gallery)}
    arch = {"status": "resolved", "architecture_family": "appendix_shared_msg", "use_io_projections": True,
            "distance": "squared", "coord_feat": "current", "layer_sharing": "independent",
            "pooling": "normalised_sum", "hidden_dim": 16, "n_layers": 1,
            "mlp_structure": "egnn_appendix"}
    model = s2.build_stage2_model({"missing_modality_representation": "learned_token"},
                                 training, {"values": {}}, arch, node_feat_dim=4,
                                 edge_feat_dim=4)
    assert model.query.fusion.cfg == query
    assert model.gallery.fusion.cfg == gallery
