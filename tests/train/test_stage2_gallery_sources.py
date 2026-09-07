"""Producer/consumer source binding and empty-training regressions; CPU fixtures only."""

import json

import numpy as np
import pytest

from metafind.train import gallery_index as gi
from metafind.train import stage2


def canonical_texts(tmp_path):
    return json.loads((tmp_path / "procthor_object_text.json").read_text())


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    from metafind.models.resolve_stage1 import serialize_fitted

    root = tmp_path / "modalities"
    root.mkdir()
    monkeypatch.setattr(gi.paths, "PROCTHOR_MODALITIES", root)
    monkeypatch.setattr(gi.paths, "OUTPUTS", tmp_path)
    annotations, texts = {}, {}
    for asset in ("a", "b"):
        image = tmp_path / f"{asset}.png"
        image.write_bytes(f"image {asset}".encode())
        (root / f"{asset}.json").write_text(json.dumps({
            "asset_id": asset, "text": f"old description {asset}",
            "view_paths": [str(image)], "pointcloud_uri": str(tmp_path / "unused.npz")}))
        annotations[asset] = {
            "category": f"asset {asset}", "width": 10, "length": 20, "height": 30,
            "description": f"A newly captioned asset {asset}.", "materials": ["wood"],
            "onFloor": True, "source": "procthor_metadata_v1",
        }
        texts[asset] = {"text": serialize_fitted(annotations[asset]),
                       "source": "procthor_metadata_v1@0123456789abcdef+gemma_caption"}
    (tmp_path / "procthor_object_text.json").write_text(json.dumps(texts))
    (tmp_path / "procthor_asset_annotations.json").write_text(json.dumps(annotations))
    return root


def index(tmp_path, identity=None):
    vec = np.eye(2, dtype=np.float32)
    record = gi.build_index(vec, ["a", "b"], tmp_path / "gallery.npz",
                            extra={"text": vec, "image": vec}, source_identity=identity)
    record.update(stage1_checkpoint_sha256="parent", gallery_encoder_sha256="encoder",
                  modality_completeness={"declared_modalities": ["text", "image"]})
    return record


def test_current_bound_index_loads_same_verified_arrays(corpus, tmp_path):
    identity = gi.capture_stage2_gallery_sources(("text", "image"))
    record = index(tmp_path, identity)
    ids, embeddings, arrays = gi.verified_stage2_index(record, "parent", ("text", "image"))
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(arrays["text"], embeddings)
    assert gi.verify_stage2_gallery_sources(record, arrays) == "verified"
    assert all("pointcloud" not in source for source in identity["encoded_inputs"].values())
    assert identity["version"] == 2
    assert identity["text_source"]["template"] == "v3_fit"
    assert identity["encoded_inputs"]["a"]["text"] == canonical_texts(tmp_path)["a"]["text"]
    # The undeclared cloud is never opened, and changing its bytes is immaterial.
    (tmp_path / "unused.npz").write_bytes(b"an undeclared cloud")
    assert gi.verify_stage2_gallery_sources(record, arrays) == "verified"


def test_sorted_json_keys_do_not_change_asset_identity_or_index_row_order(corpus, tmp_path):
    # File order and asset-ID order need not coincide in a legitimate corpus.
    for filename, asset in (("a.json", "b"), ("b.json", "a")):
        path = corpus / filename
        rec = json.loads(path.read_text())
        path.write_text(json.dumps({**rec, "asset_id": asset}))
    identity = gi.capture_stage2_gallery_sources(("text", "image"))
    assert list(identity["encoded_inputs"]) == ["b", "a"]
    vec = np.eye(2, dtype=np.float32)
    record = gi.build_index(vec, ["b", "a"], tmp_path / "gallery.npz",
                            extra={"text": vec, "image": vec}, source_identity=identity)
    record.update(stage1_checkpoint_sha256="parent", gallery_encoder_sha256="encoder",
                  modality_completeness={"declared_modalities": ["text", "image"]})
    sorted_record = json.loads(json.dumps(record, sort_keys=True))
    assert list(sorted_record["source_identity"]["encoded_inputs"]) == ["a", "b"]
    ids, embeddings, _ = gi.verified_stage2_index(sorted_record, "parent", ("text", "image"))
    assert ids == ["b", "a"]
    np.testing.assert_array_equal(embeddings, vec)


@pytest.mark.parametrize("changed", ["text", "image", "membership", "view_order"])
def test_source_drift_is_refused_even_with_legacy_flag(corpus, tmp_path, changed):
    record = index(tmp_path, gi.capture_stage2_gallery_sources(("text", "image")))
    path = corpus / "a.json"
    rec = json.loads(path.read_text())
    if changed == "text":
        rec["text"] = "new blue cabinet"
        path.write_text(json.dumps(rec))
    elif changed == "image":
        (tmp_path / "a.png").write_bytes(b"a replacement image at the same path")
    elif changed == "membership":
        (corpus / "c.json").write_text(json.dumps({**rec, "asset_id": "c"}))
    else:
        rec["view_paths"] = [str(tmp_path / "b.png"), str(tmp_path / "a.png")]
        path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="source bytes or membership changed"):
        gi.verified_stage2_index(record, "parent", ("text", "image"), allow_legacy_sources=True)


def test_new_json_snapshot_cannot_backfill_old_vectors(corpus, tmp_path):
    record = index(tmp_path)  # Legacy vectors predate source binding.
    path = corpus / "a.json"
    rec = json.loads(path.read_text())
    rec["text"] = "new blue cabinet"
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="no bound source_identity"):
        gi.verified_stage2_index(record, "parent", ("text", "image"))
    _, _, arrays = gi.verified_stage2_index(record, "parent", ("text", "image"),
                                           allow_legacy_sources=True)
    assert gi.verify_stage2_gallery_sources(record, arrays, allow_legacy=True) == "legacy_unbound"
    record["source_identity"] = gi.capture_stage2_gallery_sources(("text", "image"))
    with pytest.raises(ValueError, match="not bound to its index bytes"):
        gi.verified_stage2_index(record, "parent", ("text", "image"), allow_legacy_sources=True)


def test_sidecar_source_identity_cannot_change_under_same_index_bytes(corpus, tmp_path):
    record = index(tmp_path, gi.capture_stage2_gallery_sources(("text", "image")))
    record["source_identity"]["selection_limit"] = 2
    with pytest.raises(ValueError, match="not bound to its index bytes"):
        gi.verified_stage2_index(record, "parent", ("text", "image"))
    record.pop("source_identity")
    with pytest.raises(ValueError, match="no bound source_identity"):
        gi.verified_stage2_index(record, "parent", ("text", "image"), allow_legacy_sources=True)


def test_declared_cloud_and_exact_image_bytes_are_verified(corpus, tmp_path):
    cloud = tmp_path / "unused.npz"
    cloud.write_bytes(b"cloud original")
    identity = gi.capture_stage2_gallery_sources(("text", "image", "pc"))
    source = identity["encoded_inputs"]["a"]
    assert gi.verified_source_bytes(source["images"][0]) == b"image a"
    assert gi.verified_source_bytes(source["pointcloud"]) == b"cloud original"
    cloud.write_bytes(b"cloud replacement")
    with pytest.raises(ValueError, match="source changed"):
        gi.verified_source_bytes(source["pointcloud"])


def test_exclusions_and_limit_only_hash_encoded_sources(corpus, tmp_path):
    a = corpus / "a.json"
    rec = json.loads(a.read_text())
    rec["text"] = ""  # The obsolete copied text no longer controls eligibility.
    rec["view_paths"] = []
    a.write_text(json.dumps(rec))
    identity = gi.capture_stage2_gallery_sources(("text", "image"), limit=1)
    assert len(identity["modality_records"]) == 1
    assert identity["encoded_inputs"] == {}
    record = {"source_identity": identity,
              "modality_completeness": {"declared_modalities": ["text", "image"]}}
    (corpus / "b.json").write_bytes(b"outside the selected prefix")
    assert gi.verify_stage2_gallery_sources(record) == "verified"


def test_duplicate_asset_ids_are_refused(corpus):
    (corpus / "b.json").write_bytes((corpus / "a.json").read_bytes())
    with pytest.raises(ValueError, match="duplicate.*asset_id"):
        gi.capture_stage2_gallery_sources(("text", "image"))


def test_current_root_cannot_substitute_another_corpus(corpus, tmp_path, monkeypatch):
    record = index(tmp_path, gi.capture_stage2_gallery_sources(("text", "image")))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(gi.paths, "PROCTHOR_MODALITIES", other)
    with pytest.raises(ValueError, match="current modality root differs"):
        gi.verified_stage2_index(record, "parent", ("text", "image"))


@pytest.mark.parametrize("changed_at", [None, "before_image", "after_image",
                                        "before_text", "after_text", "during_text_read"])
def test_real_stage2_index_entry_point_binds_the_encoded_bytes(
        corpus, tmp_path, monkeypatch, changed_at):
    """Run the CLI orchestration with tiny encoders, including the final recheck."""
    import contextlib
    import sys
    from types import SimpleNamespace
    from PIL import Image
    import torch
    from metafind.models import ulip_backbone
    from metafind.train import stage1

    expected_texts = [rec["text"] for rec in canonical_texts(tmp_path).values()]
    text_inputs = []
    sidecar_bytes = {path: path.read_bytes() for path in corpus.glob("*.json")}

    def replace_canonical_text():
        path = tmp_path / "procthor_object_text.json"
        data = json.loads(path.read_text())
        data["a"]["text"] = "a changed sentence during encoding"
        path.write_text(json.dumps(data))

    for asset in ("a", "b"):
        Image.new("RGB", (2, 2), color=(20, 40, 60)).save(tmp_path / f"{asset}.png")
    monkeypatch.setattr(gi.paths, "OUTPUTS", tmp_path)
    monkeypatch.setattr(gi, "STAGE2_PATH", tmp_path / "stage2_gallery_index.json")
    (tmp_path / "stage2_protocol.json").write_text(json.dumps({"asset_modalities": ["text", "image"]}))
    parent = {"uri": str(tmp_path / "unused-parent.pt"), "sha256": "parent"}
    monkeypatch.setattr(gi, "load_checkpoint_record", lambda p: parent)
    monkeypatch.setattr(stage1, "load_stage1_model_config", lambda *a: parent)
    monkeypatch.setattr(stage1, "load_protocols", lambda: ({}, {}, {}))
    monkeypatch.setattr(stage1, "effective_stage1_model_inputs", lambda *a: (
        {}, {"train_scope": "fuser_only", "tower_sharing": "shared_backbone_separate_fusion"}, {}))
    monkeypatch.setattr(stage1, "stage1_backbone_kwargs", lambda *a: {})
    monkeypatch.setattr(stage1, "load_stage1_checkpoint", lambda *a, **kw: None)

    class Gallery(torch.nn.Module):
        def forward(self, embeds, declared=None):
            return embeds["text"] + embeds["image"]

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gallery = Gallery()

        def freeze_gallery(self, frozen):
            self.gallery.eval()

    class Backbone:
        def __init__(self, config):
            self.model = torch.nn.Linear(2, 2)
            self.cfg = SimpleNamespace(dtype=torch.float32)

        def set_train_scope(self, scope):
            self.model.eval()

        def is_frozen(self):
            return True

        def encode_text(self, texts):
            text_inputs.extend(texts)
            if changed_at == "after_text":
                replace_canonical_text()
            return torch.tensor([[float(len(texts[0])), 1.]])

        def preprocess(self, image):
            return torch.tensor(list(image.getpixel((0, 0)))[:2], dtype=torch.float32)

        def encode_image(self, images):
            if changed_at == "after_image":
                (tmp_path / "a.png").write_bytes(b"changed during encode")
            return images

    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", Backbone)
    monkeypatch.setattr(stage1, "build_model", lambda *a: (Tower(), torch.nn.Linear(1, 1)))
    monkeypatch.setattr(gi.runlog, "run_progress", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(gi.runlog, "cost_ledger", lambda **kw: None)
    if changed_at in ("before_image", "before_text"):
        capture = gi.capture_stage2_gallery_sources
        def changed_snapshot(*args, **kwargs):
            identity = capture(*args, **kwargs)
            if changed_at == "before_image":
                (tmp_path / "a.png").write_bytes(b"changed before encode")
            else:
                replace_canonical_text()
            return identity
        monkeypatch.setattr(gi, "capture_stage2_gallery_sources", changed_snapshot)
    if changed_at == "during_text_read":
        from pathlib import Path
        read_bytes = Path.read_bytes
        replaced = False
        def swap_after_read(path):
            nonlocal replaced
            raw = read_bytes(path)
            if path == tmp_path / "procthor_object_text.json" and not replaced:
                replaced = True
                replace_canonical_text()
            return raw
        monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    monkeypatch.setattr(sys, "argv", ["gallery_index", "stage2", "--device", "cpu"])
    if changed_at is not None:
        with pytest.raises(ValueError, match="source (changed|bytes or membership changed)"):
            gi.main()
        assert not gi.STAGE2_PATH.exists()
        if changed_at in ("before_text", "after_text", "during_text_read"):
            assert text_inputs == expected_texts  # Exact captured bytes, not a later reopen.
    else:
        assert gi.main() == 0
        record = json.loads(gi.STAGE2_PATH.read_text())
        ids, embeddings, arrays = gi.verified_stage2_index(record, "parent", ("text", "image"))
        assert ids == ["a", "b"]
        np.testing.assert_array_equal(embeddings, arrays["text"] + arrays["image"])
        assert text_inputs == expected_texts
        np.testing.assert_array_equal(arrays["text"], [[len(t), 1.] for t in expected_texts])
        assert "stage2_source_identity_sha256" in arrays
        # The trainer/probe both reuse these raw vectors for queries. Exercise
        # the real lookup and encode_query seam without an unused layout branch.
        class Query(torch.nn.Module):
            layout_encoder = None

            def forward(self, embeds, *, present, layout):
                assert layout is None
                assert embeds["pc"] is None
                return embeds["text"]

        data = SimpleNamespace(asset_vectors=stage2.load_asset_modality_vectors(
            arrays, ("text", "image")))
        query = stage2.encode_query(SimpleNamespace(query=Query()), {}, 0, "a",
                                    True, "cpu", data)
        np.testing.assert_array_equal(query.numpy(), arrays["text"][0])
    assert all(path.read_bytes() == old for path, old in sidecar_bytes.items())


@pytest.mark.parametrize("legacy", [False, True])
def test_version_one_sidecar_text_cannot_be_relabelled_canonical(corpus, tmp_path, legacy):
    identity = gi.capture_stage2_gallery_sources(("text", "image"))
    identity["version"] = 1
    identity.pop("text_source")
    for source in identity["encoded_inputs"].values():
        source.pop("text")
    record = index(tmp_path, identity)
    with pytest.raises(ValueError, match="version 1 binds renderer-sidecar text"):
        gi.verified_stage2_index(record, "parent", ("text", "image"),
                                 allow_legacy_sources=legacy)


@pytest.mark.parametrize("change", ["canonical_text", "annotation", "canonical_root"])
def test_consumer_rechecks_canonical_text_sources(corpus, tmp_path, monkeypatch, change):
    record = index(tmp_path, gi.capture_stage2_gallery_sources(("text", "image")))
    if change == "canonical_root":
        other = tmp_path / "other"
        other.mkdir()
        for name in ("procthor_object_text.json", "procthor_asset_annotations.json"):
            (other / name).write_bytes((tmp_path / name).read_bytes())
        monkeypatch.setattr(gi.paths, "OUTPUTS", other)
    else:
        name = "procthor_object_text.json" if change == "canonical_text" else "procthor_asset_annotations.json"
        path = tmp_path / name
        data = json.loads(path.read_text())
        data["a"]["text" if change == "canonical_text" else "description"] = "different source"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source bytes or membership changed"):
        gi.verified_stage2_index(record, "parent", ("text", "image"), allow_legacy_sources=True)


@pytest.mark.parametrize("change", ["missing_map", "missing_asset", "empty_text", "wrong_template",
                                    "generic_source", "wrong_annotation"])
def test_new_snapshot_requires_current_canonical_metadata(corpus, tmp_path, change):
    path = tmp_path / "procthor_object_text.json"
    data = canonical_texts(tmp_path)
    if change == "missing_map":
        path.unlink()
    elif change == "missing_asset":
        data.pop("a")
    elif change == "empty_text":
        data["a"]["text"] = ""
    elif change == "wrong_template":
        data["a"]["text"] = json.loads((corpus / "a.json").read_text())["text"]
    elif change == "generic_source":
        data["a"]["source"] = "a different producer"
    else:
        annotation_path = tmp_path / "procthor_asset_annotations.json"
        annotations = json.loads(annotation_path.read_text())
        annotations["a"]["width"] = 999
        annotation_path.write_text(json.dumps(annotations))
    if change != "missing_map":
        path.write_text(json.dumps(data))
    with pytest.raises((FileNotFoundError, ValueError)):
        gi.capture_stage2_gallery_sources(("text", "image"))


def test_obsolete_sidecar_text_does_not_exclude_valid_canonical_text(corpus):
    path = corpus / "a.json"
    data = json.loads(path.read_text())
    data.pop("text")
    path.write_text(json.dumps(data))
    identity = gi.capture_stage2_gallery_sources(("text", "image"))
    assert set(identity["encoded_inputs"]) == {"a", "b"}
    assert identity["encoded_inputs"]["a"]["text"]


def test_image_only_snapshot_does_not_consume_undeclared_text(corpus, tmp_path):
    (tmp_path / "procthor_object_text.json").unlink()
    (tmp_path / "procthor_asset_annotations.json").unlink()
    identity = gi.capture_stage2_gallery_sources(("image",))
    assert identity["text_source"] is None
    assert all("text" not in source for source in identity["encoded_inputs"].values())


@pytest.mark.parametrize("pattern,batch_size", [(["A"] * 12, 64),
                                               (list("ABCDEFG"), 64),
                                               (list("ABCDEFGHIJKL"), 4), ([], 64)])
def test_zero_update_epoch_is_rejected(pattern, batch_size):
    samples = [("h", i, a) for i, a in enumerate(pattern)]
    with pytest.raises(ValueError, match="no usable batches.*zero optimizer steps"):
        stage2.training_batches(samples, batch_size, np.random.default_rng(7))


def test_prevalidated_first_batch_preserves_sampler_and_rng():
    samples = [("h", i, str(i % 13)) for i in range(50)]
    old_rng = np.random.default_rng(7)
    expected = stage2.usable_batches(stage2.unique_positive_batches(samples, 8, old_rng))
    new_rng = np.random.default_rng(7)
    actual = stage2.training_batches(samples, 8, new_rng)
    assert actual == expected
    assert new_rng.random() == old_rng.random()


def test_zero_batch_cli_refuses_before_backbone_or_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    from metafind.models import ulip_backbone
    from metafind.train import stage1

    monkeypatch.setattr(stage2.paths, "OUTPUTS", tmp_path)
    monkeypatch.setattr(stage2, "CKPT_DIR", tmp_path)
    values = {"optimizer": "adamw", "learning_rate": .001, "weight_decay": 0.,
              "batch_size": 64, "epochs": 1, "seed": 7, "init_temperature": .5,
              "learnable_temperature": False, "max_logit_scale": 100.}
    hyper = {"values": values}
    hp = tmp_path / "hyper.json"
    hp.write_text(json.dumps(hyper))
    parent_weights = tmp_path / "parent.pt"
    parent_weights.write_bytes(b"unused parent bytes")
    parent = {"uri": str(parent_weights), "sha256": hashlib.sha256(parent_weights.read_bytes()).hexdigest()}
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    for name, value in {
        "scene_splits.json": {"train_houses": ["h"]}, "stage2_positive_map.json": {"a": "a"},
        "stage2_gallery_index.json": {"stage1_checkpoint_sha256": parent["sha256"],
                                     "modality_completeness": {"declared_modalities": ["text", "image"]}},
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    recipe = ({}, {"tower_sharing": "shared_backbone_separate_fusion", "train_scope": "fuser_only"}, hyper)
    monkeypatch.setattr(stage1, "load_protocols", lambda: recipe)
    monkeypatch.setattr(stage1, "load_stage1_model_config", lambda *a: parent)
    monkeypatch.setattr(stage1, "effective_stage1_model_inputs", lambda *a: recipe)
    monkeypatch.setattr(stage2, "load_stage2_protocols", lambda: (
        {"asset_modalities": ["text", "image"], "graph_unit": "room"}, {}, {}))
    monkeypatch.setattr(stage2, "capture_stage2_input_identity", lambda *a: {})
    monkeypatch.setattr(stage2, "verify_stage2_input_identity", lambda *a, **kw: {})
    vectors = {"ids": np.array(["a"]), "text": np.ones((1, 2), dtype=np.float32),
               "image": np.ones((1, 2), dtype=np.float32)}
    monkeypatch.setattr(gi, "verified_stage2_index", lambda *a, **kw: (["a"], vectors["text"], vectors))
    monkeypatch.setattr(stage2, "Stage2Data", lambda *a, **kw: SimpleNamespace(
        asset_vectors=None, modalities={"a": {}}, graphs_for=lambda houses: {}))
    monkeypatch.setattr(stage2, "enumerate_samples", lambda *a, **kw: [("h", 0, "a")])
    def forbidden_backbone(*a, **kw):
        raise AssertionError("zero-batch validation must precede backbone allocation")
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", forbidden_backbone)
    monkeypatch.setattr(sys, "argv", ["stage2", "--device", "cpu", "--hyperparameters", str(hp),
                                      "--stage1-ckpt-record", str(parent_path)])
    with pytest.raises(ValueError, match="no usable batches"):
        stage2.main()
    assert not (tmp_path / "stage2_full.pt").exists()
