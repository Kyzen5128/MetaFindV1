"""Checkpoint reconstruction must survive unrelated current protocol changes."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch

from metafind.models.fusion import FusionConfig, ModalityFusion
from metafind.models.stage1_config import UnsupportedProtocol
from metafind.train import stage1


def recipe():
    return ({"missing_modality_representation": "learned_token",
             "actual_clip_train_scope": "frozen", "image_aggregation": "mean"},
            {"fusion": "transformer", "tower_sharing": "shared_backbone_separate_fusion",
             "train_scope": "fuser_only", "prefusion_norm": True,
             "gallery_fusion": "mean", "freeze_gallery": True},
            {"values": {"learnable_temperature": False, "init_temperature": 0.5,
                        "max_logit_scale": 100}})


def legacy_record():
    e, t, h = recipe()
    arm = {**{f"encoding.{k}": v for k, v in e.items()},
           **{f"training.{k}": v for k, v in t.items()}, **h["values"],
           "train_scope": t["train_scope"]}
    return {"arm_config": arm, **t, "clip_train_scope": "frozen"}


def test_legacy_recipe_wins_over_current_protocol_without_mutating_it():
    e, t, h = recipe()
    t.update(prefusion_norm=False, gallery_fusion="transformer", freeze_gallery=False,
             train_scope="point_encoder_and_fuser", image_tokens=12)
    e["missing_modality_representation"] = "zero_pad"
    original = copy.deepcopy((e, t, h))
    got_e, got_t, _ = stage1.effective_stage1_model_inputs(legacy_record(), e, t, h)
    assert (e, t, h) == original
    assert got_t["train_scope"] == "fuser_only"
    assert got_t["freeze_gallery"] is True
    query = stage1.fusion_config_for(got_e, got_t)
    gallery = stage1.fusion_config_for(got_e, got_t, gallery=True)
    assert query.prefusion_norm and query.image_tokens == 1 and not query.zero_pad
    assert gallery.kind == "mean"


def test_legacy_missing_flags_use_historical_defaults():
    record = legacy_record()
    del record["prefusion_norm"]
    del record["arm_config"]["training.prefusion_norm"]
    e, t, h = recipe()
    _, restored, _ = stage1.effective_stage1_model_inputs(record, e, t, h)
    assert restored["prefusion_norm"] is False


def test_parent_modality_mask_rate_is_restored_for_variant_checks():
    record = legacy_record()
    record["arm_config"]["p_mask"] = 0.3
    e, t, h = recipe()
    h["values"]["p_mask"] = 0.0
    _, _, restored = stage1.effective_stage1_model_inputs(record, e, t, h)
    assert restored["values"]["p_mask"] == 0.3


def test_partial_checkpoint_uses_its_recorded_backbone_initializer(tmp_path, monkeypatch):
    from metafind.models import ulip_backbone
    original = tmp_path / "custom.pt"
    original.write_bytes(b"custom frozen point backbone")
    default = tmp_path / "default.pt"
    default.write_bytes(b"different released backbone")
    class Config:
        checkpoint = default
    monkeypatch.setattr(ulip_backbone, "BackboneConfig", Config)
    record = {"initializers": {"ulip2": {
        "uri": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}}}
    assert stage1.stage1_backbone_kwargs(record)["checkpoint"] == original
    original.unlink()
    with pytest.raises(ValueError, match="no available file matches"):
        stage1.stage1_backbone_kwargs(record)
    default.write_bytes(b"custom frozen point backbone")
    assert stage1.stage1_backbone_kwargs(record)["checkpoint"] == default


def test_changed_openclip_initializer_is_refused(tmp_path, monkeypatch):
    original = tmp_path / "custom.pt"
    original.write_bytes(b"point backbone")
    record = {"initializers": {
        "ulip2": {"uri": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()},
        "open_clip": {"weight_blob_sha256": "recorded"}}}
    monkeypatch.setattr(stage1, "_open_clip_weight_identity", lambda: {"weight_blob_sha256": "changed"})
    with pytest.raises(ValueError, match="OpenCLIP initializer differs"):
        stage1.stage1_backbone_kwargs(record)


def test_unknown_recipe_is_refused_and_conflicting_record_is_refused():
    e, t, h = recipe()
    with pytest.raises(ValueError, match="no bound"):
        stage1.effective_stage1_model_inputs({}, e, t, h)
    record = legacy_record()
    record["prefusion_norm"] = False
    with pytest.raises(ValueError, match="prefusion_norm"):
        stage1.effective_stage1_model_inputs(record, e, t, h)


def test_snapshot_restores_actual_forward_after_protocol_drift(monkeypatch):
    from metafind.models import ulip_backbone
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 16)
    e, t, h = recipe()
    original, _ = stage1.build_model(e, t, h)
    record = {"model_inputs": stage1.model_input_snapshot(e, t, h, original)}
    t["prefusion_norm"] = False
    t["gallery_fusion"] = "transformer"
    e2, t2, h2 = stage1.effective_stage1_model_inputs(record, e, t, h)
    restored, _ = stage1.build_model(e2, t2, h2)
    restored.load_state_dict(original.state_dict())
    original.eval(); restored.eval()
    embeds = {name: torch.randn(3, 16) * scale
              for name, scale in zip(("text", "image", "pc"), (1, 10, 100))}
    present = torch.ones(3, 3, dtype=torch.bool)
    with torch.no_grad():
        for side in ("query", "gallery"):
            assert torch.equal(getattr(original, side).fusion(embeds, present),
                               getattr(restored, side).fusion(embeds, present))


def test_direct_loader_rejects_parameter_free_protocol_drift(monkeypatch):
    from metafind.models import ulip_backbone
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 16)
    e, t, h = recipe()
    model, _ = stage1.build_model(e, t, h)
    record = {"model_inputs": stage1.model_input_snapshot(e, t, h, model)}
    stage1.validate_stage1_forward_config(model, record)
    model.query.fusion.cfg.prefusion_norm = False
    with pytest.raises(ValueError, match="query fusion configuration differs"):
        stage1.validate_stage1_forward_config(model, record)


def test_frozen_transformer_gallery_is_saved_and_required_on_restore(tmp_path, monkeypatch):
    from metafind import paths
    from metafind.models import ulip_backbone
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 16)
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    e, t, h = recipe()
    t.update(gallery_fusion="transformer", hyperparameter_config_hash="b" * 64,
             _arm_config_hash="a" * 64, _arm_config={})
    original, loss = stage1.build_model(e, t, h)
    original.freeze_gallery(True)
    class Backbone:
        model = torch.nn.Linear(16, 16)
        def trainable_parameters(self):
            return list(self.model.parameters())
    rp = stage1.resolve_run_paths(None)
    record = stage1.save_checkpoint(Backbone(), original, loss, h, e, t, 1, 0, rp)
    state = torch.load(rp.latest_checkpoint, weights_only=False)["tower_trainable_state"]
    restored, _ = stage1.build_model(e, t, h)
    restored.freeze_gallery(True)
    stage1.load_stage1_tower_state(restored, state)
    assert record["tower_state_includes_frozen_gallery"] is True
    for name, value in original.gallery.state_dict().items():
        assert torch.equal(value, restored.gallery.state_dict()[name])
    old_state = {k: v for k, v in state.items() if not k.startswith("gallery.")}
    with pytest.raises(ValueError, match="does not cover"):
        stage1.load_stage1_tower_state(restored, old_state)


def test_embedded_recipe_is_bound_to_bytes_and_sidecar(tmp_path):
    path = tmp_path / "stage1.pt"
    metadata = legacy_record()
    torch.save({"metadata": metadata}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = stage1.load_stage1_model_config(path)
    assert loaded["sha256"] == digest and loaded["uri"] == str(path)
    with pytest.raises(ValueError, match="hashes to"):
        stage1.load_stage1_model_config(path, {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="metadata disagree"):
        stage1.load_stage1_model_config(path, {"sha256": digest, "prefusion_norm": False})


def test_data_root_relocation_preserves_initializer_identity(tmp_path):
    path = tmp_path / "stage1.pt"
    metadata = {**legacy_record(), "embeddings_dir": "/old/embeddings",
                "initializers": {"ulip2": {"uri": "/old/ulip.pt", "sha256": "a" * 64},
                                 "open_clip": {"hf_cache": "/old/hf", "weight_blob_sha256": "b" * 64}}}
    torch.save({"metadata": metadata}, path)
    record = copy.deepcopy(metadata)
    record.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), embeddings_dir="/new/embeddings")
    record["initializers"]["ulip2"]["uri"] = "/new/ulip.pt"
    record["initializers"]["open_clip"]["hf_cache"] = "/new/hf"
    got = stage1.load_stage1_model_config(path, record)
    assert got["initializers"]["ulip2"]["uri"] == "/new/ulip.pt"
    record["initializers"]["ulip2"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="initializers"):
        stage1.load_stage1_model_config(path, record)


@pytest.mark.parametrize("sharing", ["fully_separate", "shared_backbone_separate_fusion"])
def test_frozen_gallery_rejects_trainable_gallery_backbone(sharing):
    with pytest.raises(UnsupportedProtocol, match="fuser_only"):
        stage1.validate_frozen_gallery_scope(
            {"freeze_gallery": True, "tower_sharing": sharing,
             "train_scope": "point_encoder_and_fuser"})
    stage1.validate_frozen_gallery_scope(
        {"freeze_gallery": True, "train_scope": "fuser_only"})


@pytest.mark.parametrize("key,value", [
    ("_query_observation", "second_observation"),
    ("_query_image_policy", "random_view"), ("_query_pc_perturb", "rotate")])
def test_query_observations_change_experiment_identity(key, value):
    e, t, _ = recipe()
    t.update(similarity="cosine", _epoch_count=1, _lr_horizon=1, _num_workers=4)
    values = {"optimizer": "adamw", "scheduler": "cosine"}
    first, _ = stage1.arm_config_hash(values, t, e, "dev")
    t[key] = value
    changed, _ = stage1.arm_config_hash(values, t, e, "dev")
    assert first != changed


def test_main_process_random_observations_repeat_after_seed():
    from metafind.data.observation import view_indices
    def draw():
        return ([view_indices("random_view", "asset", 12) for _ in range(20)],
                np.random.rand(4).tolist(), torch.rand(4).tolist())
    random.seed(999)
    stage1.seed_training(20260816)
    first = draw()
    stage1.seed_training(20260816)
    assert first == draw()


@pytest.mark.parametrize("zero_pad", [False, True])
def test_absent_multiview_image_has_same_forward_for_none_or_tensor(zero_pad):
    fusion = ModalityFusion(FusionConfig(
        dim=16, hidden=32, n_heads=2, n_layers=1, image_tokens=3,
        zero_pad=zero_pad, prefusion_norm=True)).eval()
    present = torch.tensor([[True, False, True], [False, False, True]])
    embeddings = {"text": torch.randn(2, 16), "pc": torch.randn(2, 16), "image": None}
    with torch.no_grad():
        a = fusion(embeddings, present)
        b = fusion({**embeddings, "image": torch.randn(2, 3, 16)}, present)
    assert torch.equal(a, b)
