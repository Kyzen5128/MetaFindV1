"""Raw input bridge: real encoding control flow, source proofs and CPU replay.

Tiny encoders replace only the large ULIP constructor. These tests make no
claim about a real pretrained model's retrieval quality.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from metafind.scene import prepare as module
from metafind.data.semantic_edges import cache_key
from metafind.data.semantic_provenance import build_node_source, build_edge_source, source_identity_sha256
from .test_scene_composition import case, existing_checkpoint_records, no_layout_case, slot


IDENTITY = {"version": 1, "kind": "open_clip_loaded_text_tower", "state_sha256": "1"*64,
            "tokenizer_sha256": "2"*64, "architecture_sha256": "3"*64,
            "implementation": {"fixture": "small_cpu_encoder"},
            "output": {"normalization": "l2", "dtype": "float32"}}


class TinyBackbone:
    def __init__(self):
        self.texts, self.clouds = [], []

    def encode_text(self, texts):
        self.texts += texts
        return torch.tensor([[3., 0.] if text.startswith("A") else [0., 4.] for text in texts])

    def preprocess(self, image):
        return torch.tensor(np.array(image).copy()).permute(2, 0, 1).float()

    def encode_image(self, batch):
        return batch[:, :2].mean((2, 3))

    def encode_pc(self, points):
        self.clouds.append(points.clone())
        return torch.tensor([[5., 6.]])


def loaded_stub(tower, record=None):
    return SimpleNamespace(backbone=TinyBackbone(), query_backbone=TinyBackbone(),
                           model=SimpleNamespace(cfg=tower.cfg), record=record,
                           encoding={"image_aggregation": "mean"})


def test_raw_query_preserves_text_views_and_query_point_path(case, tmp_path):
    loaded = loaded_stub(case[0])
    for name, rgb in (("first", [9, 0, 0]), ("second", [0, 3, 0])):
        Image.new("RGB", (2, 2), tuple(rgb)).save(tmp_path / f"{name}.png")
    xyz = np.zeros((10000, 3), dtype=np.float32)
    xyz[0] = [.1, -.2, .3]
    rgb = np.full_like(xyz, .25)
    np.savez(tmp_path / "pc.npz", xyz=xyz, rgb=rgb)
    sources = {}
    query = {"slot": slot("q", 0), "text": "A  verbatim caption. ",
             "images": ["first.png", "second.png"], "pointcloud": "pc.npz"}
    result = module.encode_query(query, loaded, tmp_path, sources)
    assert loaded.backbone.texts == [query["text"]]
    torch.testing.assert_close(result["embeds"]["text"], torch.tensor([3., 0.]))
    torch.testing.assert_close(result["embeds"]["image"], torch.tensor([4.5, 1.5]))
    torch.testing.assert_close(result["embeds"]["pc"], torch.tensor([5., 6.]))
    assert not loaded.backbone.clouds
    torch.testing.assert_close(loaded.query_backbone.clouds[0][0], torch.from_numpy(np.concatenate((xyz, rgb), 1)))
    assert len(sources) == 3
    assert all(hashlib.sha256(open(p, "rb").read()).hexdigest() == sha for p, sha in sources.items())


@pytest.mark.parametrize("query", [{}, {"text": " "}, {"images": []}, {"bogus": "text"}])
def test_missing_or_invalid_modalities_never_become_zero_observations(case, tmp_path, query):
    with pytest.raises(ValueError):
        module.encode_query({"slot": slot("q", 0), **query}, loaded_stub(case[0]), tmp_path, {})


def semantic_fixture(tmp_path):
    texts = {"A": {"text": "A original annotation", "relation_text": "chair"},
             "B": {"text": "B original annotation", "relation_text": "table"}}
    meta = {"prompt_version": 1, "llm_model": "fixture", "text_encoder_version": "tiny"}
    def artifact(name, data, *, arrays=False):
        path = tmp_path / name
        if arrays: np.savez(path, **data)
        else: path.write_text(json.dumps(data))
        return {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    node_source = build_node_source(texts, IDENTITY)
    node = artifact("nodes.npz", {"ids": np.array(["A", "B"]), "embeddings": np.eye(2, dtype=np.float32),
                    "source_identity_sha256": np.array(source_identity_sha256(node_source))}, arrays=True)
    entries = {cache_key("chair", "table", **meta): {"sentence": "A relation", "degraded": False,
                      "embedding_uri": str(tmp_path / "edges.npz") + "#0"},
               "known-degraded": {"sentence": None, "degraded": True, "embedding_uri": None}}
    edge_source = build_edge_source(entries, IDENTITY)
    edge = artifact("edges.npz", {"keys": np.array(edge_source["keys"]),
                    "embeddings": np.array([[.6, .8]], dtype=np.float32),
                    "source_identity_sha256": np.array(source_identity_sha256(edge_source))}, arrays=True)
    node_record = artifact("nodes.json", {**node, "source_identity": node_source})
    cache_record = artifact("cache.json", {**meta, "entries": entries, "embedding_artifact": edge,
                                           "source_identity": edge_source})
    child = {"input_identity": {"artifacts": {"node_record": node_record, "node_embeddings": node,
              "object_text": artifact("text.json", texts), "semantic_cache": cache_record,
              "semantic_embeddings": edge}}}
    return child, texts, tmp_path / "cache.json"


def test_semantic_encoding_binds_real_n08_schema_and_distinguishes_query_text(tmp_path, monkeypatch):
    child, texts, cache = semantic_fixture(tmp_path)
    monkeypatch.setattr(module, "text_encoder_identity", lambda _: copy.deepcopy(IDENTITY))
    backbone = TinyBackbone()
    nodes, edges, meta, proof = module.semantic_inputs(child, backbone, texts, cache, {}, node_dim=2, edge_dim=2)
    assert backbone.texts == [texts[uid]["text"] for uid in ("A", "B")]
    torch.testing.assert_close(nodes["A"], torch.tensor([1., 0.]))
    torch.testing.assert_close(nodes["B"], torch.tensor([0., 1.]))
    assert proof == build_node_source(texts, IDENTITY)
    torch.testing.assert_close(edges[cache_key("chair", "table", **meta)], torch.tensor([.6, .8]))
    assert edges["known-degraded"] is None


@pytest.mark.parametrize("fault", ["encoder", "training_text", "scene_sentence", "dimension"])
def test_semantic_bridge_refuses_incompatible_inputs(tmp_path, monkeypatch, fault):
    child, texts, cache = semantic_fixture(tmp_path)
    encoder = copy.deepcopy(IDENTITY)
    if fault == "encoder": encoder["state_sha256"] = "f"*64
    monkeypatch.setattr(module, "text_encoder_identity", lambda _: encoder)
    if fault == "training_text":
        path = tmp_path / "text.json"
        path.write_text(json.dumps({**texts, "A": {"text": "changed"}}))
    if fault == "scene_sentence":
        # The child keeps its original training cache at a separate path.
        other = tmp_path / "scene-cache.json"
        value = json.loads(cache.read_text())
        key = next(iter(value["entries"]))
        value["entries"][key]["sentence"] = "new unencoded sentence"
        other.write_text(json.dumps(value))
        cache = other
    with pytest.raises(ValueError):
        module.semantic_inputs(child, TinyBackbone(), texts, cache, {}, node_dim=2,
                               edge_dim=3 if fault == "dimension" else 2)


def raw_request(case, tmp_path, monkeypatch):
    tower, inputs = no_layout_case(case)
    parent, child = existing_checkpoint_records((tower, inputs), tmp_path, monkeypatch)
    parent_record = json.loads(parent.read_text())
    loaded = loaded_stub(tower, parent_record)
    monkeypatch.setattr(module, "load_stage1", lambda *_: loaded)
    calls = []
    monkeypatch.setattr(module.gallery_index, "verify_gallery_encoder", lambda *a, **kw: calls.append((a, kw)))
    np.savez(tmp_path / "gallery.npz", ids=np.array(["B", "A"]), embeddings=np.array([[0., 1.], [1., 0.]], dtype=np.float32))
    record = {"uri": str(tmp_path / "gallery.npz"), "sha256": hashlib.sha256((tmp_path / "gallery.npz").read_bytes()).hexdigest(),
              "count": 2, "dim": 2, "gallery_encoder_sha256": "fixture-helper-seam",
              "stage1_checkpoint_sha256": parent_record["sha256"]}
    (tmp_path / "gallery.json").write_text(json.dumps({parent_record["sha256"]: record}))
    (tmp_path / "texts.json").write_text(json.dumps(inputs["asset_texts"]))
    spec = {"schema": module.REQUEST_SCHEMA, "provenance": {"fixture": "raw driver CPU"},
            "stage1_record": "s1.json", "stage2_record": "s2.json", "variant": "no_layout",
            "gallery_registry": "gallery.json", "gallery_ids": ["A", "B"], "asset_texts": "texts.json",
            "initial_graph": inputs["initial_graph"], "mode": "iterative", "use_layout": False,
            "queries": [{"slot": q["slot"], "text": "A wanted object"} for q in inputs["queries"]]}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(spec))
    return path, loaded, calls


def test_raw_request_real_export_and_cpu_replay(case, tmp_path, monkeypatch):
    request, loaded, calls = raw_request(case, tmp_path, monkeypatch)
    dest = module.prepare(request, tmp_path / "bundle")
    result = module.run_manifest(dest)
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A"]*3
    assert result["gallery_ids"] == ["A", "B"]
    manifest = json.loads(dest.read_text())
    assert manifest["provenance"]["semantic_source"]["status"] == "not_used"
    assert calls[0][0][1] is loaded.backbone
    assert calls[0][1]["allow_legacy"] is False
    assert (dest.parent / "validation_composition.json").is_file()
    assert not (dest.parent / "candidate.json").exists()
    with pytest.raises(FileExistsError): module.prepare(request, dest.parent)


def test_input_changed_during_encoding_never_publishes_manifest(case, tmp_path, monkeypatch):
    request, loaded, _ = raw_request(case, tmp_path, monkeypatch)
    original = loaded.backbone.encode_text
    def drift(texts):
        request.write_text(request.read_text() + " ")
        return original(texts)
    loaded.backbone.encode_text = drift
    with pytest.raises(ValueError, match="source hash mismatch"):
        module.prepare(request, tmp_path / "bundle")
    assert not (tmp_path / "bundle" / "manifest.json").exists()


def test_missing_real_trajectory_pairs_are_exported_without_complete_manifest(case, tmp_path, monkeypatch):
    from metafind.scene.compose import compose
    request, _, _ = raw_request(case, tmp_path, monkeypatch)
    tower, inputs = case
    inputs["semantic_cache"] = {}
    # Use a real ESSGNN trajectory at the replay seam, rather than inventing
    # the KeyError or parsing an exception's human-readable message.
    monkeypatch.setattr(module, "run_manifest", lambda _: compose(tower, **inputs))
    with pytest.raises(module.MissingSemanticPairsError) as error:
        module.prepare(request, tmp_path / "bundle")
    assert error.value.pairs == [["A", "B"]]
    out = tmp_path / "bundle"
    assert json.loads((out / "required_pairs.json").read_text()) == [["A", "B"]]
    assert json.loads((out / "missing_semantics.json").read_text())["status"] == "incomplete"
    assert not (out / "manifest.json").exists()
