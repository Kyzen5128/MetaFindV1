"""Real n08 SG2/encoding and source checks; only large model loads are faked."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from metafind.data.semantic_edges import cache_key
from metafind.data.semantic_provenance import read_verified_edge_source
from metafind.models.dual_tower import QueryTower
from metafind.scene import semantics as module
from .test_scene_composition import case, existing_checkpoint_records
from .test_scene_prepare import IDENTITY, semantic_fixture


@pytest.fixture
def setup(case, tmp_path, monkeypatch):
    semantic_dir = tmp_path / "reference_semantics"
    semantic_dir.mkdir()
    proof, texts, cache_path = semantic_fixture(semantic_dir)
    tower = QueryTower(replace(case[0].cfg, essgnn=replace(case[0].cfg.essgnn, edge_feat_dim=2))).eval()
    parent, child = existing_checkpoint_records((tower, case[1]), tmp_path, monkeypatch)
    child_rec = json.loads(child.read_text())
    child_payload = torch.load(child_rec["uri"], weights_only=False)
    child_payload["metadata"].update(proof)
    child_payload["metadata"]["layout_input_dims"]["edge_feat_dim"] = 2
    torch.save(child_payload, child_rec["uri"])
    child_rec.update(proof)
    child_rec["layout_input_dims"]["edge_feat_dim"] = 2
    child_rec["sha256"] = hashlib.sha256(Path(child_rec["uri"]).read_bytes()).hexdigest()
    child.write_text(json.dumps(child_rec))
    text_path, pairs_path = tmp_path / "asset_texts.json", tmp_path / "pairs.json"
    text_path.write_text(json.dumps(texts))
    pairs_path.write_text(json.dumps([["A", "B"], ["B", "A"]]))
    monkeypatch.setattr(module.n08, "LLM_MODEL", "fixture")
    monkeypatch.setattr(module.n08, "LLM_MODEL_PATH", str(tmp_path / "models" / "fixture"))
    monkeypatch.setattr(module.n08, "TEXT_ENCODER_VERSION", "tiny")
    monkeypatch.setattr(module.n08, "EDGE_DIM", 2)
    monkeypatch.setattr(module.stage1, "stage1_backbone_kwargs", lambda _: {})
    monkeypatch.setattr(module, "text_encoder_identity", lambda _: IDENTITY)
    parent_record = json.loads(parent.read_text())
    events, prompts, writer_refs = [], [], []
    answers = ["", "Sure: The chair accompanies the table."]

    class Writer:
        def __init__(self, model_id, device):
            self.model_id = model_id
            self.answers = iter(answers)
            writer_refs.append(weakref.ref(self))
            events.append("writer_loaded")

        def generate(self, prompt):
            prompts.append(prompt)
            return next(self.answers)

    class Encoder:
        def encode_text(self, sentences):
            events.append(("encoded", sentences))
            return torch.tensor([[3., 4.]]).repeat(len(sentences), 1)

    def load_model(path, device):
        assert all(ref() is None for ref in writer_refs), "LLM and ULIP must not coexist"
        events.append("encoder_loaded")
        return SimpleNamespace(record=parent_record, backbone=Encoder())

    monkeypatch.setattr(module.n08, "RelationWriter", Writer)
    monkeypatch.setattr(module, "load_stage1", load_model)
    return SimpleNamespace(parent=parent, child=child, texts=text_path, pairs=pairs_path,
                           base=cache_path, answers=answers, events=events, prompts=prompts,
                           out=tmp_path / "new_cache")


def run(setup, **kwargs):
    return module.generate(setup.texts, setup.pairs, setup.parent, setup.child, setup.out, **kwargs)


def read_cache(path):
    cache = json.loads(path.read_text())
    with np.load(cache["embedding_artifact"]["uri"]) as stored:
        arrays = dict(stored)
    assert read_verified_edge_source(cache, arrays)["status"] == "verified"
    return cache, arrays


def test_real_repair_then_encoding_deduplicates_only_requested_pairs(setup):
    path = run(setup)
    cache, arrays = read_cache(path)
    assert len(setup.prompts) == module.n08.MAX_ATTEMPTS == 2
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in setup.prompts[1]
    assert "no sentence" in setup.prompts[1]
    assert setup.events[:2] == ["writer_loaded", "encoder_loaded"]
    assert setup.events[2] == ("encoded", ["The chair accompanies the table."])
    np.testing.assert_allclose(arrays["embeddings"], [[.6, .8]], atol=1e-7)
    assert len(cache["entries"]) == 1
    record = json.loads((setup.out / "record.json").read_text())
    assert record["distinct_required_keys"] == record["generated_successes"] == 1
    assert len(record["requested_pairs"]) == 2
    assert record["generated_degraded"] == 0
    assert record["llm"]["historical_weight_byte_equivalence"] == "UNKNOWN"
    assert record["text_encoder_status"] == "verified_loaded_text_tower"
    with pytest.raises(FileExistsError):
        run(setup)


def test_exhausted_real_sg2_budget_emits_only_genuine_degraded_record(setup):
    setup.answers[:] = ["", "1234"]
    path = run(setup)
    cache, arrays = read_cache(path)
    assert len(setup.prompts) == 2
    assert setup.events == ["writer_loaded"]
    assert arrays["embeddings"].shape == (0, 2)
    entry = next(iter(cache["entries"].values()))
    assert entry["degraded"] and entry["embedding_uri"] is None
    sentence = json.loads((setup.out / "sentences.jsonl").read_text())
    assert "no words" in sentence["reason"] and sentence["generation_origin"] == "n08_sg2_write_one"
    assert json.loads((setup.out / "record.json").read_text())["generated_degraded"] == 1


def test_all_verified_base_hits_load_neither_model_and_preserve_vectors(setup):
    path = run(setup, base_cache=setup.base)
    cache, arrays = read_cache(path)
    assert setup.events == setup.prompts == []
    assert cache["entries"]["known-degraded"]["degraded"]
    np.testing.assert_array_equal(arrays["embeddings"], np.array([[.6, .8]], dtype=np.float32))
    record = json.loads((setup.out / "record.json").read_text())
    assert record["generated_keys"] == [] and not record["llm"]["invoked"]
    assert record["text_encoder_status"] == "verified_existing_vectors_no_new_encoding"


def test_base_extension_allows_same_asset_pair_without_expanding_gallery(setup):
    setup.pairs.write_text(json.dumps([["A", "A"], ["B", "A"]]))
    path = run(setup, base_cache=setup.base)
    cache, arrays = read_cache(path)
    meta = {k: cache[k] for k in module.SEMANTIC_KEY_FIELDS}
    assert len(setup.prompts) == 2  # one missing description pair, two real attempts
    assert set(arrays["keys"].tolist()) == {cache_key("chair", "chair", **meta),
                                           cache_key("chair", "table", **meta)}
    assert len(cache["entries"]) == 3  # two vectors plus copied explicit degraded evidence


@pytest.mark.parametrize("pairs", [[], [["A"]], [["A", "absent"]], [["A", 2]], "all"])
def test_bad_explicit_pairs_fail_before_models_and_output(setup, pairs):
    setup.pairs.write_text(json.dumps(pairs))
    with pytest.raises(ValueError, match="pairs|pair"):
        run(setup)
    assert not setup.events and not setup.out.exists()


def test_different_training_model_key_refused_before_models(setup, monkeypatch):
    monkeypatch.setattr(module.n08, "LLM_MODEL", "different-model")
    with pytest.raises(ValueError, match="prompt/model/version"):
        run(setup)
    assert not setup.events and not setup.out.exists()


def test_loaded_writer_model_id_cannot_impersonate_cache_key(setup, monkeypatch):
    monkeypatch.setattr(module.n08, "RelationWriter", lambda **kwargs: SimpleNamespace(model_id="wrong"))
    with pytest.raises(ValueError, match="configured n08 model ID"):
        run(setup)
    assert not (setup.out / "sem_edge_cache.json").exists()


def test_wrong_loaded_text_encoder_cannot_publish_vectors(setup, monkeypatch):
    monkeypatch.setattr(module, "text_encoder_identity", lambda _: {**IDENTITY, "state_sha256": "4"*64})
    with pytest.raises(ValueError, match="text encoder differs"):
        run(setup)
    assert len(setup.prompts) == 2
    assert not (setup.out / "sem_edge_cache.json").exists()
    assert (setup.out / "sentences.jsonl").exists()


def test_source_change_during_generation_cannot_publish_cache(setup, monkeypatch):
    real = module.n08.write_one
    def changed(*args):
        result = real(*args)
        setup.texts.write_text(setup.texts.read_text() + " ")
        return result
    monkeypatch.setattr(module.n08, "write_one", changed)
    with pytest.raises(ValueError, match="source hash mismatch"):
        run(setup)
    assert not (setup.out / "sem_edge_cache.json").exists()


def test_cli_defaults_to_cpu_and_accepts_requested_pair_file(setup):
    assert module.main(["--asset-texts", str(setup.texts), "--required-pairs", str(setup.pairs),
                        "--stage1-record", str(setup.parent), "--stage2-record", str(setup.child),
                        "--base-cache", str(setup.base), "--out-dir", str(setup.out)]) == 0
    assert json.loads((setup.out / "record.json").read_text())["device"] == "cpu"


def test_generated_cache_reaches_actual_raw_scene_semantic_consumer(setup, monkeypatch):
    from metafind.scene import prepare
    from .test_scene_prepare import TinyBackbone

    cache_path = run(setup)
    monkeypatch.setattr(prepare, "text_encoder_identity", lambda _: IDENTITY)
    texts = json.loads(setup.texts.read_text())
    child = json.loads(setup.child.read_text())
    nodes, edges, meta, source = prepare.semantic_inputs(
        child, TinyBackbone(), texts, cache_path, {}, node_dim=2, edge_dim=2)
    torch.testing.assert_close(nodes["A"], torch.tensor([1., 0.]))
    torch.testing.assert_close(nodes["B"], torch.tensor([0., 1.]))
    key = cache_key("chair", "table", **meta)
    torch.testing.assert_close(edges[key], torch.tensor([.6, .8]))
    assert source["encoder_identity"] == IDENTITY


def test_base_sidecar_cannot_relabel_unencoded_sentence(setup):
    cache = json.loads(setup.base.read_text())
    entry = next(entry for entry in cache["entries"].values() if not entry.get("degraded"))
    entry["sentence"] = "changed without encoding"
    other = setup.base.with_name("changed-cache.json")
    other.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="source text"):
        run(setup, base_cache=other)
    assert not setup.events and not setup.out.exists()
