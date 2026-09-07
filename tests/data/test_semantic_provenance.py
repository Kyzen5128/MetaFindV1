"""Small CPU regressions for n08's actual source/encoder binding."""
import copy
import json
import re

import numpy as np
import pytest
import torch

from metafind.data import semantic_provenance as provenance


ENCODER = {"version": 1, "kind": "open_clip_loaded_text_tower",
           "state_sha256": "1" * 64, "tokenizer_sha256": "2" * 64,
           "architecture_sha256": "3" * 64, "implementation": {"test": "tiny"},
           "output": {"normalization": "l2", "dtype": "float32"}}
TEXT = {"b": {"text": " B chair.", "relation_text": "chair"},
        "a": {"text": "A table.", "relation_text": "table"}}


def node_fixture():
    source = provenance.build_node_source(TEXT, ENCODER)
    arrays = {"ids": np.array(["a", "b"]),
              "embeddings": np.eye(2, dtype=np.float32),
              "source_identity_sha256": np.array(provenance.source_identity_sha256(source))}
    return {"source_identity": source}, arrays


def test_exact_node_text_bound_but_json_order_and_relation_text_independent():
    rec, arrays = node_fixture()
    text = dict(reversed(list(TEXT.items())))
    assert provenance.read_verified_node_source(rec, text, arrays)["status"] == "verified"
    text = copy.deepcopy(text)
    text["a"]["relation_text"] = "new relation description"
    assert provenance.read_verified_node_source(rec, text, arrays)["status"] == "verified"
    text["a"]["text"] += " "
    with pytest.raises(ValueError, match="source text or membership changed"):
        provenance.read_verified_node_source(rec, text, arrays)


def test_backfilled_sidecar_cannot_relabel_existing_node_vectors():
    rec, arrays = node_fixture()
    text = copy.deepcopy(TEXT)
    text["a"]["text"] = "Unrelated new node text"
    rec["source_identity"] = provenance.build_node_source(text, ENCODER)
    with pytest.raises(ValueError, match="not bound to its vector bytes"):
        provenance.read_verified_node_source(rec, text, arrays, allow_legacy=True)


@pytest.mark.parametrize("mutation", ["reordered", "duplicate", "missing", "nan", "dtype"])
def test_invalid_vector_identity_or_rows_refused(mutation):
    rec, arrays = node_fixture()
    if mutation == "reordered":
        arrays["ids"] = arrays["ids"][::-1]
    elif mutation == "duplicate":
        arrays["ids"] = np.array(["a", "a"])
    elif mutation == "missing":
        arrays["embeddings"] = arrays["embeddings"][:1]
    elif mutation == "nan":
        arrays["embeddings"][0, 0] = np.nan
    else:
        arrays["embeddings"] = arrays["embeddings"].astype(np.float64)
    with pytest.raises(ValueError, match="semantic vector"):
        provenance.read_verified_node_source(rec, TEXT, arrays)


def test_legacy_opt_in_is_explicit_and_does_not_accept_partial_modern_metadata():
    rec, arrays = node_fixture()
    legacy = {k: v for k, v in arrays.items() if k != "source_identity_sha256"}
    with pytest.raises(ValueError, match="no bound source identity"):
        provenance.read_verified_node_source({}, TEXT, legacy)
    assert provenance.read_verified_node_source({}, TEXT, legacy, True) == {
        "status": "legacy_unbound", "source_identity": None, "encoder_identity": None}
    with pytest.raises(ValueError, match="not bound"):
        provenance.read_verified_node_source(rec, TEXT, legacy, True)
    with pytest.raises(ValueError, match="unsupported"):
        provenance.read_verified_node_source({}, TEXT, arrays, True)


def test_edges_bind_exact_sentences_and_positional_pointers():
    entries = {"b": {"sentence": "B edge.", "embedding_uri": "cache.npz#1"},
               "a": {"sentence": "A edge.", "embedding_uri": "cache.npz#0"},
               "bad": {"degraded": True, "sentence": None, "embedding_uri": None}}
    source = provenance.build_edge_source(entries, ENCODER)
    record = {"entries": entries, "source_identity": source}
    arrays = {"keys": np.array(["a", "b"]), "embeddings": np.eye(2, dtype=np.float32),
              "source_identity_sha256": np.array(provenance.source_identity_sha256(source))}
    assert provenance.read_verified_edge_source(record, arrays)["status"] == "verified"
    entries["a"]["embedding_uri"] = "cache.npz#1"
    with pytest.raises(ValueError, match="row pointer"):
        provenance.read_verified_edge_source(record, arrays)
    entries["a"]["embedding_uri"] = "cache.npz#0"
    entries["a"]["sentence"] += " "
    with pytest.raises(ValueError, match="source text"):
        provenance.read_verified_edge_source(record, arrays)


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2)
        self.transformer = torch.nn.Linear(2, 2)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer("attn_mask", torch.eye(2), persistent=False)
        self.context_length = 2
        self.text_pool_type = "argmax"

    def encode_text(self, tokens):
        return self.transformer(tokens)


class TinyULIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.open_clip_model = TinyClip()

    def encode_text(self, tokens):
        return self.open_clip_model.encode_text(tokens)


def lower(text):
    return text.lower()


class TinyTokenizer:
    def __init__(self):
        self.encoder = {"a": 1, "b": 2}
        self.bpe_ranks = {("a", "b"): 0}
        self.byte_encoder = {97: "a", 98: "b"}
        self.context_length = 2
        self.all_special_ids = [0, 3]
        self.sot_token_id, self.eot_token_id = 0, 3
        self.pat = re.compile("[a-z]+")
        self.clean_fn = lower
        self.cache = {}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyULIP()
        self.tokenizer = TinyTokenizer()
        self.eval()

    def encode_text(self, texts):
        return self.model.encode_text(torch.ones(len(texts), 2))


def test_loaded_encoder_identity_ignores_mutable_cache_vision_and_logit_scale():
    backbone = TinyBackbone()
    before = provenance.text_encoder_identity(backbone)
    backbone.tokenizer.cache["new"] = "cached BPE output"
    with torch.no_grad():
        backbone.model.open_clip_model.visual.weight.add_(1)
        backbone.model.open_clip_model.logit_scale.add_(1)
    assert provenance.text_encoder_identity(backbone) == before


def test_encoder_must_be_in_evaluation_mode():
    backbone = TinyBackbone().train()
    with pytest.raises(ValueError, match="eval mode"):
        provenance.text_encoder_identity(backbone)


def test_real_tiny_openclip_identity_survives_tokenization_and_forward():
    from open_clip.model import CLIP
    from open_clip.tokenizer import SimpleTokenizer

    backbone = TinyBackbone()
    backbone.tokenizer = SimpleTokenizer(context_length=8)
    backbone.model.open_clip_model = CLIP(
        embed_dim=4,
        vision_cfg={"layers": 1, "width": 8, "head_width": 4,
                    "patch_size": 2, "image_size": 4},
        text_cfg={"context_length": 8, "vocab_size": backbone.tokenizer.vocab_size,
                  "width": 8, "heads": 2, "layers": 1},
    ).eval()
    before = provenance.text_encoder_identity(backbone)
    tokens = backbone.tokenizer(["A chair beside a table"])
    with torch.no_grad():
        assert backbone.model.encode_text(tokens).shape == (1, 4)
    assert provenance.text_encoder_identity(backbone) == before


@pytest.mark.parametrize("change", ["weight", "nonpersistent_mask", "vocab", "bpe", "context", "pool"])
def test_loaded_encoder_identity_detects_effective_text_changes(change):
    backbone = TinyBackbone()
    before = provenance.text_encoder_identity(backbone)
    clip = backbone.model.open_clip_model
    if change == "weight":
        with torch.no_grad():
            clip.transformer.weight.add_(1)
    elif change == "nonpersistent_mask":
        clip.attn_mask[0, 0] = -100
    elif change == "vocab":
        backbone.tokenizer.encoder["a"] = 4
    elif change == "bpe":
        backbone.tokenizer.bpe_ranks[("a", "b")] = 2
    elif change == "context":
        backbone.tokenizer.context_length = 3
    else:
        clip.text_pool_type = "last"
    assert provenance.text_encoder_identity(backbone) != before


@pytest.mark.parametrize("mutate_input", [False, True])
def test_n08_real_producer_uses_one_encoder_and_rechecks_source(tmp_path, monkeypatch, mutate_input):
    from metafind.data import semantic_edges_run as producer
    from metafind.models import ulip_backbone

    text_file = tmp_path / "procthor_object_text.json"
    text_file.write_text(json.dumps(TEXT))
    monkeypatch.setattr(producer.paths, "OUTPUTS", tmp_path)
    for attr, name in (("EMBEDDINGS_PATH", "edges.npz"), ("CACHE_PATH", "edges.json"),
                       ("NODE_EMB_PATH", "nodes.npz"), ("NODE_EMB_RECORD", "nodes.json")):
        monkeypatch.setattr(producer, attr, tmp_path / name)
    settled = {"z": {"sentence": "Z relation", "degraded": False},
               "a": {"sentence": "A relation", "degraded": False}}
    monkeypatch.setattr(producer, "collect_pairs", lambda *args: dict.fromkeys(settled))
    monkeypatch.setattr(producer, "load_sentences", lambda: settled)
    monkeypatch.setattr(producer.runlog, "cost_ledger", lambda **kwargs: None)
    loaded = []
    def construct(cfg):
        loaded.append(TinyBackbone())
        return loaded[-1]
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", construct)
    monkeypatch.setattr(provenance, "text_encoder_identity", lambda _: ENCODER)
    calls = []
    real_encode = producer.encode_sentences
    monkeypatch.setattr(producer, "EDGE_DIM", 2)
    def encode(texts, *, backbone):
        calls.append((texts, backbone))
        if mutate_input and len(calls) == 2:
            changed = copy.deepcopy(TEXT)
            changed["a"]["text"] = "New text while encoding"
            text_file.write_text(json.dumps(changed))
        return real_encode(texts, backbone=backbone)
    monkeypatch.setattr(producer, "encode_sentences", encode)
    monkeypatch.setattr("sys.argv", ["semantic_edges_run"])
    if mutate_input:
        with pytest.raises(ValueError, match="changed during semantic encoding"):
            producer.main()
        assert not producer.NODE_EMB_RECORD.exists()
        assert not producer.EMBEDDINGS_PATH.exists()
    else:
        assert producer.main() == 0
        node = json.loads(producer.NODE_EMB_RECORD.read_text())
        edge = json.loads(producer.CACHE_PATH.read_text())
        with np.load(node["uri"]) as arrays:
            assert provenance.read_verified_node_source(node, TEXT, arrays)["status"] == "verified"
            np.testing.assert_allclose(np.linalg.norm(arrays["embeddings"], axis=1), 1, atol=1e-6)
        with np.load(producer.EMBEDDINGS_PATH) as arrays:
            assert provenance.read_verified_edge_source(edge, arrays)["status"] == "verified"
    assert len(loaded) == 1 and len(calls) == 2
    assert calls[0][1] is calls[1][1] is loaded[0]
    assert calls[0][0] == ["A relation", "Z relation"]
    assert calls[1][0] == [TEXT["a"]["text"], TEXT["b"]["text"]]
