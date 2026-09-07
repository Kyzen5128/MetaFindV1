"""Custom retrieval loader seams with temporary checkpoints and tiny models."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from metafind.eval import custom_models as cm
from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
from metafind.models.fusion import FusionConfig


class TinyBackbone:
    events = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.model.weight)
        self.set_train_scope(cfg.train_scope)
        self.events.append(("construct", cfg.train_scope))

    def set_train_scope(self, scope):
        self.cfg.train_scope = scope
        self.model.requires_grad_(scope != "fuser_only")
        self.model.train(scope != "fuser_only")
        self.events.append(("scope", scope))

    def clone_point_path(self):
        self.events.append(("clone", self.cfg.train_scope))
        return TinyBackbone(replace(self.cfg))


@pytest.fixture
def tiny_runtime(monkeypatch):
    events = []
    monkeypatch.setattr(TinyBackbone, "events", events)
    monkeypatch.setattr(cm, "ULIPBackbone", TinyBackbone)
    monkeypatch.setattr(cm.stage1, "load_protocols", lambda: ({}, {}, {"values": {}}))

    def build(encoding, training, hp):
        model = MetaFindDualTower(DualTowerConfig(
            dim=8, tower_sharing=training["tower_sharing"], use_layout=False,
            query_fusion=cm.stage1.fusion_config_for(encoding, training),
            gallery_fusion=cm.stage1.fusion_config_for(encoding, training, gallery=True)))
        return model, torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(cm.stage1, "build_model", build)
    real_load = cm.stage1.load_stage1_checkpoint

    def checked_load(backbone, model, loss_fn, path, *, query_backbone=None):
        events.append(("load", backbone.cfg.train_scope))
        return real_load(backbone, model, loss_fn, path, query_backbone=query_backbone)

    monkeypatch.setattr(cm.stage1, "load_stage1_checkpoint", checked_load)
    return events


def write_parent(tmp_path, *, scope="point_encoder_and_fuser", sharing="shared_backbone_separate_fusion",
                 freeze_gallery=False, token_side=None, clip_scope="frozen"):
    initializer = tmp_path / "initializer.pt"
    initializer.write_bytes(b"tiny initializer")
    fusion = FusionConfig(dim=8, kind="transformer", hidden=16, n_heads=2,
                          n_layers=1, prefusion_norm=True)
    encoding = {"missing_modality_representation": "learned_token",
                "actual_clip_train_scope": clip_scope}
    training = {"train_scope": scope, "tower_sharing": sharing, "fusion": "transformer",
                "freeze_gallery": freeze_gallery, "image_tokens": 1,
                "query_fusion_config": asdict(fusion), "gallery_fusion_config": asdict(fusion)}
    if token_side:
        training[f"{token_side}_fusion_config"]["image_tokens"] = 2
    hp = {"values": {"learnable_temperature": False, "init_temperature": .5,
                     "max_logit_scale": 100}}
    model, loss_fn = cm.stage1.build_model(encoding, training, hp)
    metadata = {"model_inputs": cm.stage1.model_input_snapshot(encoding, training, hp, model),
                "initializers": {"ulip2": {"uri": str(initializer),
                    "sha256": hashlib.sha256(initializer.read_bytes()).hexdigest()}}}
    payload = {"metadata": metadata, "tower_trainable_state": model.state_dict(),
               "loss_trainable_state": loss_fn.state_dict(),
               "backbone_trainable_state": {} if scope == "fuser_only" else
                   {"weight": torch.full((2, 2), 4.)}}
    if sharing == "fully_separate":
        payload["query_backbone_trainable_state"] = ({} if scope == "fuser_only" else
                                                     {"weight": torch.full((2, 2), 8.)})
    weights = tmp_path / "stage1.pt"
    torch.save(payload, weights)
    record = {**metadata, "uri": str(weights), "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    record_path = tmp_path / "stage1.json"
    record_path.write_text(json.dumps(record))
    return record_path, record


@pytest.mark.parametrize("scope,sharing,freeze", [
    ("point_encoder_and_fuser", "shared_backbone_separate_fusion", False),
    ("fuser_only", "shared_backbone_separate_fusion", True),
    ("point_encoder_and_fuser", "fully_separate", False),
])
def test_stage1_restores_original_scope_then_freezes(tmp_path, tiny_runtime, scope, sharing, freeze):
    path, _ = write_parent(tmp_path, scope=scope, sharing=sharing, freeze_gallery=freeze)
    loaded = cm.load_stage1(path, "cpu")
    assert loaded.training["train_scope"] == scope
    assert ("load", scope) in tiny_runtime
    assert tiny_runtime.index(("construct", scope)) < tiny_runtime.index(("load", scope))
    assert tiny_runtime[-1] == ("scope", "fuser_only")
    assert torch.all(loaded.backbone.model.weight == (0 if scope == "fuser_only" else 4))
    assert loaded.model.query.fusion.cfg.prefusion_norm
    modules = [loaded.backbone.model, loaded.model, loaded.loss_fn]
    if sharing == "fully_separate":
        assert loaded.query_backbone is not None
        assert torch.all(loaded.query_backbone.model.weight == 8)
        modules.append(loaded.query_backbone.model)
    else:
        assert loaded.query_backbone is None
    assert all(not p.requires_grad for m in modules for p in m.parameters())
    assert all(not child.training for m in modules for child in m.modules())


@pytest.mark.parametrize("kwargs,match", [
    ({"scope": "full"}, "trainable CLIP"),
    ({"clip_scope": "trainable"}, "trainable CLIP"),
    ({"token_side": "query"}, "image_tokens=1"),
    ({"token_side": "gallery"}, "image_tokens=1"),
])
def test_cached_input_limits_fail_before_backbone(tmp_path, tiny_runtime, kwargs, match):
    path, _ = write_parent(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        cm.load_stage1(path, "cpu")
    assert not tiny_runtime


def test_stage1_rejects_replaced_checkpoint_before_construction(tmp_path, tiny_runtime):
    path, record = write_parent(tmp_path)
    from pathlib import Path
    Path(record["uri"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="hashes to"):
        cm.load_stage1(path, "cpu")
    assert not tiny_runtime


def test_mean_only_loads_verified_official_initializer(tmp_path, tiny_runtime, monkeypatch):
    path = tmp_path / "official.pt"
    path.write_bytes(b"pretend official bytes")
    monkeypatch.setattr(cm.stage1, "OFFICIAL_ULIP2_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    backbone = cm.load_mean("cpu", path)
    assert backbone.cfg.train_scope == "fuser_only"
    assert torch.all(backbone.model.weight == 0)
    assert not any(event[0] == "load" for event in tiny_runtime)
    assert not backbone.model.training


def test_mean_refuses_nonofficial_bytes_before_construction(tmp_path, tiny_runtime):
    path = tmp_path / "finetuned.pt"
    path.write_bytes(b"not official")
    with pytest.raises(SystemExit, match="not the official"):
        cm.load_mean("cpu", path)
    assert not tiny_runtime


def write_stage2(tmp_path, loaded, *, parent=None, legacy=False, gallery_change=False):
    metadata = {"stage1_checkpoint_sha256": parent or loaded.record["sha256"],
                "stage1_model_inputs": loaded.record["model_inputs"],
                "lambda_init": {"init_lambda": .3}}
    state = {k: v.clone() + 1 for k, v in loaded.model.state_dict().items()
             if k.startswith("query.fusion.")}
    # A layout branch can exist in the trained checkpoint without being built.
    state["query.layout_weight"] = torch.tensor(.4)
    state["query.layout_encoder.dummy"] = torch.tensor(2.)
    if gallery_change:
        state["gallery.fusion.mask_tokens"] = torch.zeros(3, 8)
    payload = {"trainable_state": state}
    if not legacy:
        payload["metadata"] = metadata
    path = tmp_path / "stage2.pt"
    torch.save(payload, path)
    record = {**metadata, "uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record_path = tmp_path / "stage2.json"
    record_path.write_text(json.dumps({"full": record}))
    return record_path, record


def test_stage2_updates_only_query_without_scene_inputs(tmp_path, tiny_runtime, monkeypatch):
    path, _ = write_parent(tmp_path)
    loaded = cm.load_stage1(path, "cpu")
    gallery = {k: v.clone() for k, v in loaded.model.gallery.state_dict().items()}
    query = {k: v.clone() for k, v in loaded.model.query.state_dict().items()}
    from metafind.train import stage2
    def no_scene_inputs(*args, **kwargs):
        pytest.fail("layout-off overlay must not read scene inputs")
    monkeypatch.setattr(stage2, "Stage2Data", no_scene_inputs)
    monkeypatch.setattr(stage2, "verify_stage2_input_identity", no_scene_inputs)
    path, expected = write_stage2(tmp_path, loaded)
    assert cm.apply_stage2(loaded, path) == expected
    assert loaded.model.query.layout_encoder is None
    assert loaded.record["sha256"] == expected["stage1_checkpoint_sha256"]
    assert all(torch.equal(v, loaded.model.gallery.state_dict()[k]) for k, v in gallery.items())
    assert all(torch.equal(v + 1, loaded.model.query.state_dict()[k]) for k, v in query.items())
    assert not loaded.model.training


@pytest.mark.parametrize("kwargs,error,match", [
    ({"parent": "wrong"}, ValueError, "fine-tuned"),
    ({"legacy": True}, ValueError, "legacy replay"),
    ({"gallery_change": True}, SystemExit, "non-query"),
])
def test_stage2_refuses_wrong_parent_legacy_or_gallery_state(tmp_path, tiny_runtime, kwargs, error, match):
    path, _ = write_parent(tmp_path)
    loaded = cm.load_stage1(path, "cpu")
    old = {k: v.clone() for k, v in loaded.model.state_dict().items()}
    path, _ = write_stage2(tmp_path, loaded, **kwargs)
    with pytest.raises(error, match=match):
        cm.apply_stage2(loaded, path)
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in old.items())


def test_stage2_refuses_tied_gallery_before_overlay(tmp_path, tiny_runtime):
    path, _ = write_parent(tmp_path, sharing="fully_shared")
    loaded = cm.load_stage1(path, "cpu")
    with pytest.raises(ValueError, match="separate query/gallery"):
        cm.apply_stage2(loaded, tmp_path / "unused.json")
