"""The Table 1 'w/ ESSGNN' row: Stage 2 query weights over their own Stage 1 parent.

[PAPER 3.2] "Using the Stage-1 head reproduces the 'w/o ESSGNN'"; the w/ ESSGNN
row evaluates the Stage 2 head "on Objaverse-LVIS (which lacks layout and
disables ESSGNN)". Two refusals are pinned here because each would otherwise
produce a plausible-looking number under the wrong label.
"""
from __future__ import annotations

import json
import hashlib

import pytest

from metafind.eval import run_retrieval as rr


def test_a_stage2_record_from_another_parent_is_refused(tmp_path):
    rec = tmp_path / "stage2_full.json"
    rec.write_text(json.dumps({"uri": str(tmp_path / "x.pt"), "sha256": "0" * 64,
                               "stage1_checkpoint_sha256": "a" * 64,
                               "lambda_init": {"init_lambda": 0.3}}))
    with pytest.raises(SystemExit) as e:
        rr.load_stage2_over_stage1(str(rec), {"sha256": "b" * 64})
    assert "fine-tuned from Stage 1 checkpoint" in str(e.value)


def test_a_record_that_is_not_a_stage2_record_is_refused(tmp_path):
    rec = tmp_path / "stage1_ckpt.json"            # a Stage 1 record by mistake
    rec.write_text(json.dumps({"uri": str(tmp_path / "x.pt"), "sha256": "0" * 64,
                               "stage1_checkpoint_sha256": "b" * 64}))
    with pytest.raises(SystemExit) as e:
        rr.load_stage2_over_stage1(str(rec), {"sha256": "b" * 64})
    assert "lambda_init" in str(e.value)


def test_overlay_refuses_a_state_that_skips_the_query_fusion():
    """Scoring the w/ ESSGNN row with Stage 1 query weights is the failure the
    coverage check exists for."""
    import torch
    from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
    from metafind.models.fusion import FusionConfig

    f = FusionConfig(dim=8, kind="transformer", hidden=16, n_heads=2, n_layers=1)
    model = MetaFindDualTower(DualTowerConfig(dim=8, tower_sharing="shared_backbone_separate_fusion",
                                              query_fusion=f, gallery_fusion=f,
                                              use_layout=False))
    # a state carrying only lambda-shaped junk: no query.fusion keys
    state = {}
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s2.pt")
        torch.save({"trainable_state": state}, p)
        with pytest.raises(SystemExit) as e:
            rr.overlay_stage2_weights(model, {"uri": p, "sha256": hashlib.sha256(
                open(p, "rb").read()).hexdigest()}, "cpu")
    assert "does not cover" in str(e.value)


def _no_layout_checkpoints(tmp_path, *, explicit_layout_flag):
    import torch
    from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
    from metafind.models.fusion import FusionConfig
    from metafind.models.losses import MetaFindContrastiveLoss
    from metafind.train.stage1 import model_input_snapshot
    from metafind.train.stage2 import stage2_checkpoint_payload

    f = FusionConfig(dim=8, kind="mlp", hidden=8)
    model = MetaFindDualTower(DualTowerConfig(
        dim=8, tower_sharing="shared_backbone_separate_fusion",
        query_fusion=f, gallery_fusion=f, use_layout=False))
    encoding = {"missing_modality_representation": "learned_token",
                "actual_clip_train_scope": "frozen"}
    training = {"fusion": "mlp", "tower_sharing": "shared_backbone_separate_fusion",
                "train_scope": "fuser_only"}
    hyper = {"values": {"init_temperature": .5, "learnable_temperature": False,
                        "max_logit_scale": 100.}}
    snapshot = model_input_snapshot(encoding, training, hyper, model)
    parent_path = tmp_path / "stage1.pt"
    torch.save({"metadata": {"model_inputs": snapshot}}, parent_path)
    parent = {"uri": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()}
    stage2_path = tmp_path / "no_layout.pt"
    record = {"uri": str(stage2_path), "variant_id": "no_layout", "lambda_init": None,
              "stage1_checkpoint_sha256": parent["sha256"], "stage1_model_inputs": snapshot}
    if explicit_layout_flag:
        record["use_layout"] = False
    model.gallery.requires_grad_(False)
    with torch.no_grad():
        for p in model.query.fusion.parameters():
            p.add_(.125)
    torch.save(stage2_checkpoint_payload(model, MetaFindContrastiveLoss(), record), stage2_path)
    record["sha256"] = hashlib.sha256(stage2_path.read_bytes()).hexdigest()
    record_path = tmp_path / "variant_ckpts.json"
    record_path.write_text(json.dumps({"no_layout": record}))
    return model, parent, record_path, record, (encoding, training, hyper)


@pytest.mark.parametrize("explicit_layout_flag", [False, True])
def test_no_layout_checkpoint_loads_without_graph_artifacts_and_overlays_query_only(
        tmp_path, monkeypatch, explicit_layout_flag):
    import copy
    import torch
    from metafind.models.losses import MetaFindContrastiveLoss
    from metafind.train import stage1, stage2

    trained, parent, record_path, record, protocols = _no_layout_checkpoints(
        tmp_path, explicit_layout_flag=explicit_layout_flag)
    target = copy.deepcopy(trained)
    with torch.no_grad():
        for p in target.query.fusion.parameters():
            p.sub_(.125)
    monkeypatch.setattr(stage1, "load_protocols", lambda: protocols)
    monkeypatch.setattr(stage1, "build_model", lambda *args: (target, MetaFindContrastiveLoss()))

    def graph_forbidden(*args, **kwargs):
        raise AssertionError("no-layout replay must not require graph artifacts or ESSGNN")

    monkeypatch.setattr(stage2, "Stage2Data", graph_forbidden)
    monkeypatch.setattr(stage2, "load_stage2_protocols", graph_forbidden)
    monkeypatch.setattr(stage2, "build_stage2_model", graph_forbidden)
    loaded = rr.load_stage2_over_stage1(str(record_path), parent, variant="no_layout")
    assert loaded["model"].query.layout_encoder is None
    assert loaded["model"].query.layout_weight is None
    gallery_before = {k: v.clone() for k, v in target.gallery.state_dict().items()}
    rr.overlay_stage2_weights(target, loaded["record"], "cpu")
    for key, value in trained.query.state_dict().items():
        assert torch.equal(value, target.query.state_dict()[key])
    for key, value in gallery_before.items():
        assert torch.equal(value, target.gallery.state_dict()[key])


def test_full_checkpoint_with_missing_lambda_is_not_reinterpreted_as_no_layout(tmp_path):
    import torch
    from metafind.train.gallery_index import load_stage2_checkpoint_record

    p = tmp_path / "full.pt"
    record = {"uri": str(p), "variant_id": "full", "lambda_init": None,
              "stage1_checkpoint_sha256": "a" * 64}
    torch.save({"metadata": dict(record), "trainable_state": {}}, p)
    record["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    rp = tmp_path / "record.json"
    rp.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="with layout requires lambda_init"):
        load_stage2_checkpoint_record(rp, {"sha256": "a" * 64})


@pytest.mark.parametrize("record", [
    {"variant_id": "full", "use_layout": False, "lambda_init": None},
    {"variant_id": "no_layout", "use_layout": True, "lambda_init": {"init_lambda": .3}},
    {"variant_id": "no_layout", "use_layout": None, "lambda_init": None},
])
def test_explicit_layout_flag_must_match_the_named_ablation(record):
    from metafind.train.gallery_index import stage2_layout_settings
    with pytest.raises(ValueError, match="use_layout"):
        stage2_layout_settings(record)


def test_no_layout_checkpoint_cannot_smuggle_layout_weights_through_fusion_only_overlay(tmp_path):
    import torch
    from pathlib import Path

    model, _, _, record, _ = _no_layout_checkpoints(tmp_path, explicit_layout_flag=True)
    p = Path(record["uri"])
    payload = torch.load(p, weights_only=False)
    payload["trainable_state"]["query.layout_weight"] = torch.tensor(1.)
    torch.save(payload, p)
    record["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="no-layout record carries layout weights"):
        rr.overlay_stage2_weights(model, record, "cpu", fusion_only=True)


def test_procthor_probe_reports_only_available_heads_for_no_layout(tmp_path, monkeypatch):
    """Run the real CLI control flow with small model/array fixtures on CPU."""
    import copy
    import sys
    from types import SimpleNamespace
    import numpy as np
    import torch
    from metafind.models.losses import MetaFindContrastiveLoss
    from metafind.train import stage1
    from tools.probes import stage2_procthor_retrieval as probe

    trained, parent, record_path, _, protocols = _no_layout_checkpoints(
        tmp_path, explicit_layout_flag=True)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for filename, value in {
        "stage2_gallery_index.json": {"sha256": "index"},
        "stage2_protocol.json": {"asset_modalities": ["text", "image"]},
        "essgnn_arch_protocol.json": {},
        "scene_splits.json": {"test_houses": ["h"]},
    }.items():
        (outputs / filename).write_text(json.dumps(value))
    monkeypatch.setattr(probe.paths, "OUTPUTS", outputs)
    monkeypatch.setattr(probe, "load_checkpoint_record", lambda *a: parent)
    monkeypatch.setattr(probe, "load_stage1_protocols", lambda: protocols)
    monkeypatch.setattr(probe, "stage1_backbone_kwargs", lambda *a: {})

    class Data:
        node_dim = edge_dim = 2
        semantic_source_status = "legacy_unbound"

        def __init__(self, *args, **kwargs):
            pass

        def graphs_for(self, houses):
            return {house: {} for house in houses}

    monkeypatch.setattr(probe, "Stage2Data", Data)
    monkeypatch.setattr(probe, "verified_stage2_index", lambda *a, **k:
                        (["a", "b"], np.eye(2, dtype=np.float32), {}))
    monkeypatch.setattr(probe, "load_asset_modality_vectors", lambda *a: {})
    monkeypatch.setattr(probe, "enumerate_samples", lambda *a, **k: [("h", 0, "a"), ("h", 1, "b")])
    monkeypatch.setattr(probe, "ULIPBackbone", lambda cfg: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(probe, "load_stage1_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(probe, "verify_gallery_encoder", lambda *a, **k: None)
    monkeypatch.setattr(probe, "verify_stage2_gallery_sources", lambda *a, **k: None)
    monkeypatch.setattr(probe.runlog, "runtime_source_sha256", lambda: "test")
    monkeypatch.setattr(probe.runlog, "runtime_source_status", lambda: "fixture")
    target = copy.deepcopy(trained)

    def build(*args, **kwargs):
        assert kwargs["use_layout"] is False
        return target

    monkeypatch.setattr(probe, "build_stage2_model", build)
    monkeypatch.setattr(stage1, "build_model", lambda *a: (target, MetaFindContrastiveLoss()))

    def query(model, graph, target_index, asset_id, drop_layout, *args, **kwargs):
        assert drop_layout is True, "no-layout probe must not label an unchanged query S2-on"
        return torch.eye(2)[target_index]

    monkeypatch.setattr(probe, "encode_query", query)
    output = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["stage2_procthor_retrieval", "--stage1-ckpt-record", "unused",
                        "--stage2-record", str(record_path), "--variant", "no_layout",
                        "--allow-legacy-stage2-inputs", "--device", "cpu", "--out", str(output)])
    assert probe.main() == 0
    result = json.loads(output.read_text())
    assert set(result["heads"]) == {"S1_no_layout", "S2_no_layout"}
    assert result["use_layout"] is False
    assert result["layout_status"] == "disabled_by_checkpoint"
    assert result["lambda"] is None
    assert "norm_layout_term" not in result
