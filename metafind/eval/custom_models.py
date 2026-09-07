"""Verified model loading for custom asset retrieval with cached CLIP inputs.

These loaders restore model identity. The evaluation runner remains responsible
for binding its query/gallery observations and cache contents to the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from torch import nn

from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
from metafind.train import gallery_index, stage1


@dataclass
class LoadedModel:
    backbone: ULIPBackbone
    query_backbone: ULIPBackbone | None
    model: nn.Module
    record: dict
    encoding: dict
    training: dict
    hyperparameters: dict
    loss_fn: nn.Module


def load_fusion_tower(checkpoint_path: str | Path, device: str,
                      record: dict | None = None) -> nn.Module:
    """Restore both Stage 1 fusion heads without constructing a backbone.

    Historical probes supply their own modality vectors. Their caller remains
    responsible for restoring the matching point encoder and checking inputs.
    """
    import torch

    record = stage1.load_stage1_model_config(checkpoint_path, record)
    encoding, training, hyperparameters = stage1.effective_stage1_model_inputs(
        record, *stage1.load_protocols())
    model, _loss = stage1.build_model(encoding, training, hyperparameters)
    if training.get("freeze_gallery"):
        model.freeze_gallery(True)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stage1.load_stage1_tower_state(model, checkpoint["tower_trainable_state"])
    model.eval()
    return model


def load_stage1(record_path: str | Path, device: str) -> LoadedModel:
    """Restore the recorded training scope before loading, then freeze for eval."""
    record = gallery_index.load_checkpoint_record(record_path)
    record = stage1.load_stage1_model_config(record["uri"], record)
    encoding, training, hyperparameters = stage1.effective_stage1_model_inputs(
        record, *stage1.load_protocols())
    if (training["train_scope"] == "full" or
            encoding["actual_clip_train_scope"] != "frozen"):
        raise ValueError("custom cached-CLIP evaluation does not support trainable CLIP")
    for gallery in (False, True):
        if stage1.fusion_config_for(encoding, training, gallery=gallery).image_tokens != 1:
            raise ValueError("custom evaluation requires image_tokens=1 on both towers")

    backbone = ULIPBackbone(BackboneConfig(
        device=device, train_scope=training["train_scope"],
        **stage1.stage1_backbone_kwargs(record)))
    query_backbone = (backbone.clone_point_path()
                      if training["tower_sharing"] == "fully_separate" else None)
    model, loss_fn = stage1.build_model(encoding, training, hyperparameters)
    if training.get("freeze_gallery"):
        model.freeze_gallery(True)
    model.to(device)
    stage1.load_stage1_checkpoint(backbone, model, loss_fn, Path(record["uri"]),
                                 query_backbone=query_backbone)

    backbone.set_train_scope("fuser_only")
    if query_backbone is not None:
        query_backbone.set_train_scope("fuser_only")
    model.requires_grad_(False).eval()
    loss_fn.to(device).requires_grad_(False).eval()
    return LoadedModel(backbone, query_backbone, model, record, encoding,
                       training, hyperparameters, loss_fn)


def verify_mean_initializer(checkpoint: str | Path | None = None) -> Path:
    """Verify pretrained bytes without constructing a model."""
    config = BackboneConfig(train_scope="fuser_only")
    if checkpoint is not None:
        config.checkpoint = Path(checkpoint)
    with config.checkpoint.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    stage1.assert_official_initialiser({"ulip2": {"sha256": digest}}, allow_other=False)
    return config.checkpoint


def load_mean(device: str, checkpoint: str | Path | None = None) -> ULIPBackbone:
    """Load only the official pretrained backbone, for available-modality means."""
    config = BackboneConfig(device=device, train_scope="fuser_only",
                            checkpoint=verify_mean_initializer(checkpoint))
    backbone = ULIPBackbone(config)
    backbone.set_train_scope("fuser_only")
    return backbone


def apply_stage2(loaded: LoadedModel, record_path: str | Path,
                 variant: str = "full") -> dict:
    """Replace only the query fusion; layout stays disabled and gallery fixed.

    The verified Stage 2 record binds its parent and saved metadata. No scene
    inputs are consumed when evaluating this head without a layout branch.
    """
    from metafind.eval.run_retrieval import overlay_stage2_weights

    if loaded.model.fusion_is_tied():
        raise ValueError("Stage 2 overlay requires separate query/gallery fusion weights")
    record = gallery_index.load_stage2_checkpoint_record(
        record_path, loaded.record, variant, allow_legacy=False)
    overlay_stage2_weights(loaded.model, record, loaded.backbone.cfg.device,
                           fusion_only=True)
    loaded.model.requires_grad_(False).eval()
    return record
