"""CPU seams for checkpoint reconstruction and gallery/query provenance.

These use temporary bytes and tiny towers, never pretrained weights or a corpus.
"""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
from metafind.models.fusion import FusionConfig
from metafind.train import gallery_index as gi
from metafind.eval.run_retrieval import overlay_stage2_weights


def pair():
    f = FusionConfig(dim=8, hidden=16, n_heads=2, n_layers=1, dropout=0.)
    model = MetaFindDualTower(DualTowerConfig(dim=8,
        tower_sharing="shared_backbone_separate_fusion", query_fusion=f,
        gallery_fusion=replace(f), use_layout=False))
    backbone = SimpleNamespace(model=torch.nn.Linear(8, 8), cfg=SimpleNamespace(dtype=torch.float32))
    return backbone, model


@pytest.mark.parametrize("flag,value", [("prefusion_norm", True), ("zero_pad", True), ("image_tokens", 2)])
def test_hash_binds_parameter_free_gallery_forward(flag, value):
    bb, model = pair()
    old_v1 = gi.gallery_encoder_sha256(bb, model, hash_version=1)
    old_v2 = gi.gallery_encoder_sha256(bb, model)
    setattr(model.gallery.fusion.cfg, flag, value)
    assert gi.gallery_encoder_sha256(bb, model, hash_version=1) == old_v1
    assert gi.gallery_encoder_sha256(bb, model) != old_v2


def test_query_updates_do_not_change_gallery_identity():
    bb, model = pair()
    digest = gi.gallery_encoder_sha256(bb, model)
    model.query.fusion.cfg.prefusion_norm = True
    with torch.no_grad():
        next(model.query.parameters()).add_(1.)
    assert gi.gallery_encoder_sha256(bb, model) == digest
    assert gi.gallery_encoder_sha256(bb, model, declared_modalities=("text", "image")) != digest


def test_v1_is_never_silently_reinterpreted_as_v2():
    bb, model = pair()
    record = {"gallery_encoder_sha256": gi.gallery_encoder_sha256(bb, model, hash_version=1)}
    with pytest.raises(ValueError, match="legacy"):
        gi.verify_gallery_encoder(record, bb, model)
    with pytest.raises(ValueError, match="requires its parent"):
        gi.verify_gallery_encoder(record, bb, model, allow_legacy=True)


def test_explicit_v1_compatibility_checks_the_parent_forward_flags(tmp_path, monkeypatch):
    from metafind.train import stage1
    bb, model = pair()
    snapshot = {
        "version": 1,
        "encoding": {"missing_modality_representation": "learned_token", "actual_clip_train_scope": "frozen"},
        "training": {"fusion": "transformer", "tower_sharing": "shared_backbone_separate_fusion",
                     "train_scope": "fuser_only", "query_fusion_config": asdict(model.query.fusion.cfg),
                     "gallery_fusion_config": asdict(model.gallery.fusion.cfg)},
        "loss_values": {"learnable_temperature": False, "init_temperature": .5, "max_logit_scale": 100}}
    p = tmp_path / "s1.pt"
    torch.save({"metadata": {"model_inputs": snapshot}}, p)
    parent = {"uri": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    monkeypatch.setattr(stage1, "load_protocols", lambda: ({}, {}, {}))
    record = {"gallery_encoder_sha256": gi.gallery_encoder_sha256(bb, model, hash_version=1)}
    assert gi.verify_gallery_encoder(record, bb, model, parent_checkpoint=parent,
                                     allow_legacy=True) == record["gallery_encoder_sha256"]
    model.gallery.fusion.cfg.prefusion_norm = True
    with pytest.raises(ValueError, match="forward configuration differs"):
        gi.verify_gallery_encoder(record, bb, model, parent_checkpoint=parent, allow_legacy=True)


def write_s2(tmp_path, metadata=None, state=None):
    p = tmp_path / "s2.pt"
    rec = {"uri": str(p), "stage1_checkpoint_sha256": "a" * 64,
           "lambda_init": {"init_lambda": .3}, **(metadata or {})}
    torch.save({"trainable_state": state or {}, "metadata": dict(rec)}, p)
    rec["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    rp = tmp_path / "variant_ckpts.json"
    rp.write_text(json.dumps({"full": rec}))
    return rp, rec


def test_stage2_record_checks_parent_state_path_and_embedded_config(tmp_path):
    rp, rec = write_s2(tmp_path)
    assert gi.load_stage2_checkpoint_record(rp, {"sha256": "a" * 64}) == rec
    with pytest.raises(ValueError, match="fine-tuned"):
        gi.load_stage2_checkpoint_record(rp, {"sha256": "b" * 64})
    with pytest.raises(ValueError, match="stage2-state"):
        gi.load_stage2_checkpoint_record(rp, {"sha256": "a" * 64}, state_path=tmp_path / "other.pt")
    rec["lambda_init"]["init_lambda"] = .9
    rp.write_text(json.dumps({"full": rec}))
    with pytest.raises(ValueError, match="disagree: lambda_init"):
        gi.load_stage2_checkpoint_record(rp, {"sha256": "a" * 64})


def test_stage2_overlay_checks_bytes_and_refuses_gallery_changes(tmp_path):
    bb, model = pair()
    state = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("query.fusion.")}
    rp, rec = write_s2(tmp_path, state=state)
    overlay_stage2_weights(model, rec, "cpu", fusion_only=True)
    state["gallery.fusion.mask_tokens"] = torch.zeros(3, 8)
    _, malicious = write_s2(tmp_path, state=state)
    with pytest.raises(ValueError, match="sha256"):
        overlay_stage2_weights(model, rec, "cpu", fusion_only=True)
    with pytest.raises(SystemExit, match="non-query"):
        overlay_stage2_weights(model, malicious, "cpu", fusion_only=True)


def test_declared_absent_modality_requires_stage2_mask_tokens(tmp_path):
    _, model = pair()
    state = {k: v for k, v in model.state_dict().items()
             if k.startswith("query.fusion.") and not k.endswith("mask_tokens")}
    _, rec = write_s2(tmp_path, {"query_modality_masking": "none",
                                "stage2_protocol": {"asset_modalities": ["text", "image"]}}, state)
    with pytest.raises(SystemExit, match="does not cover.*mask_tokens"):
        overlay_stage2_weights(model, rec, "cpu", fusion_only=True)


def test_stage2_gallery_checks_parent_declaration_and_bytes(tmp_path):
    vec = np.eye(2, dtype=np.float32)
    rec = gi.build_index(vec, ["a", "b"], tmp_path / "index.npz", extra={"text": vec, "image": vec})
    rec.update(stage1_checkpoint_sha256="parent", gallery_encoder_sha256="encoder",
               modality_completeness={"declared_modalities": ["text", "image"]})
    ids, embeddings, arrays = gi.verified_stage2_index(rec, "parent", ("text", "image"))
    assert ids == ["a", "b"] and np.array_equal(arrays["image"], embeddings)
    with pytest.raises(ValueError, match="different Stage 1"):
        gi.verified_stage2_index(rec, "other", ("text", "image"))
    with pytest.raises(ValueError, match="declaration"):
        gi.verified_stage2_index(rec, "parent", ("text", "image", "pc"))
    (tmp_path / "index.npz").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="hashes to"):
        gi.verified_stage2_index(rec, "parent", ("text", "image"))


def test_fields_cache_is_explicit_and_never_reduces_the_query_pool(tmp_path):
    from tools.probes.exp_type_level_query import optional_fields_text
    uids = ["a", "b"]
    assert optional_fields_text(None, uids) == (None, None)
    np.savez(tmp_path / "a.npz", text=np.ones(8))
    vectors, skip = optional_fields_text(tmp_path, uids)
    assert vectors is None and skip["missing_count"] == 1
    assert uids == ["a", "b"]
    np.savez(tmp_path / "b.npz", text=np.zeros(8))
    vectors, skip = optional_fields_text(tmp_path, uids)
    assert vectors.shape == (2, 8) and skip is None


def test_load_tower_uses_checkpoint_flags_over_current_protocols(tmp_path, monkeypatch):
    from metafind.train import stage1
    from metafind.models import ulip_backbone
    from tools.probes.exp_query_observation import load_tower
    _, model = pair()
    model.query.fusion.cfg.prefusion_norm = True
    model.gallery.fusion.cfg.prefusion_norm = True
    encoding = {"missing_modality_representation": "learned_token", "actual_clip_train_scope": "frozen"}
    training = {"fusion": "transformer", "tower_sharing": "shared_backbone_separate_fusion",
                "train_scope": "fuser_only", "prefusion_norm": True, "image_tokens": 1,
                "freeze_gallery": False, "gallery_fusion": "transformer"}
    hyper = {"values": {"learnable_temperature": False, "init_temperature": .5, "max_logit_scale": 100}}
    snapshot = stage1.model_input_snapshot(encoding, training, hyper, model)
    p = tmp_path / "s1.pt"
    torch.save({"metadata": {"model_inputs": snapshot}, "tower_trainable_state": model.state_dict()}, p)
    monkeypatch.setattr(stage1, "load_protocols", lambda: ({}, {"prefusion_norm": False}, {}))
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 8)
    restored = load_tower(p, "cpu")
    assert restored.gallery.fusion.cfg.prefusion_norm is True
    assert restored.query.fusion.cfg.prefusion_norm is True
    assert asdict(restored.gallery.fusion.cfg) == snapshot["training"]["gallery_fusion_config"]
