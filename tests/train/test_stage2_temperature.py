"""Stage 2 temperature transfer through the real partial-checkpoint loader.

Tiny CPU modules exercise fixed/learnable scope changes without ULIP weights,
training data, or bypassing Stage 1's checkpoint coverage checks.
"""
from __future__ import annotations

import io
import math
from types import SimpleNamespace

import pytest
import torch

from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
from metafind.train.stage1 import trainable_state_dict
from metafind.train.stage2 import restore_stage1_for_stage2, stage2_checkpoint_payload


pytestmark = pytest.mark.filterwarnings("ignore:tau deviates from the paper")


def _values(learnable=False, tau=0.5, clamp=100.0):
    return {"learnable_temperature": learnable, "init_temperature": tau,
            "max_logit_scale": clamp}


def _parent_checkpoint(values, *, trained_tau=None):
    backbone = SimpleNamespace(model=torch.nn.Linear(2, 2))
    model = torch.nn.Linear(2, 2)
    parent_loss = MetaFindContrastiveLoss(ContrastiveConfig(**values))
    if trained_tau is not None:
        with torch.no_grad():
            parent_loss.logit_scale.fill_(math.log(1 / trained_tau))
    state = {"backbone_trainable_state": trainable_state_dict(backbone.model),
             "tower_trainable_state": trainable_state_dict(model),
             "loss_trainable_state": trainable_state_dict(parent_loss)}
    checkpoint = io.BytesIO()
    torch.save(state, checkpoint)
    checkpoint.seek(0)
    return backbone, model, checkpoint


@pytest.mark.parametrize("parent_learnable", [False, True])
@pytest.mark.parametrize("child_learnable", [False, True])
def test_temperature_transfer_respects_both_training_scopes(parent_learnable,
                                                           child_learnable):
    parent = _values(parent_learnable, tau=0.07)
    backbone, model, checkpoint = _parent_checkpoint(
        parent, trained_tau=0.03 if parent_learnable else None)
    child, record = restore_stage1_for_stage2(
        backbone, model, parent, _values(child_learnable), checkpoint)
    expected = (0.03 if parent_learnable else 0.07) if child_learnable else 0.5
    assert child.cfg.bidirectional is True
    assert isinstance(child.logit_scale, torch.nn.Parameter) is child_learnable
    assert child.temperature.item() == pytest.approx(expected)
    out = child(torch.eye(2), torch.eye(2))
    assert out["temperature"].item() == pytest.approx(expected)
    assert record["requested_init_temperature"] == 0.5
    assert record["raw_initial_temperature"] == pytest.approx(expected)
    assert record["effective_initial_temperature"] == pytest.approx(expected)
    source = ("stage1_checkpoint" if parent_learnable else "stage1_fixed_recipe")
    assert record["source"] == (source if child_learnable else "stage2_recipe")
    assert record["learnable_temperature"] is child_learnable


def test_learnable_child_preserves_raw_parent_scale_and_records_its_own_clamp():
    parent = _values(True, clamp=100.0)
    backbone, model, checkpoint = _parent_checkpoint(parent, trained_tau=0.001)
    child, record = restore_stage1_for_stage2(
        backbone, model, parent, _values(True, clamp=20.0), checkpoint)
    assert child.temperature.item() == pytest.approx(0.001)
    assert record["initial_logit_scale"] == pytest.approx(math.log(1000))
    assert record["raw_initial_temperature"] == pytest.approx(0.001)
    assert record["effective_initial_temperature"] == pytest.approx(0.05)
    assert record["max_logit_scale"] == 20.0
    actual = child(torch.eye(2), torch.eye(2))
    assert actual["temperature"].item() == pytest.approx(0.05)
    # Binding survives the actual Stage 2 payload builder and serialization.
    payload = stage2_checkpoint_payload(model, child,
                                       {"variant_id": "full", "temperature_init": record})
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=False)
    assert restored["metadata"]["temperature_init"] == record
    assert restored["loss_trainable_state"]["logit_scale"].item() == pytest.approx(
        math.log(1000))


def test_fixed_child_does_not_apply_the_learnable_scale_clamp():
    parent = _values(True)
    backbone, model, checkpoint = _parent_checkpoint(parent, trained_tau=0.07)
    child, record = restore_stage1_for_stage2(
        backbone, model, parent, _values(False, tau=0.001, clamp=20.0), checkpoint)
    assert child(torch.eye(2), torch.eye(2))["temperature"].item() == pytest.approx(0.001)
    assert record["effective_initial_temperature"] == pytest.approx(0.001)
    assert record["source"] == "stage2_recipe"


def test_missing_parent_temperature_metadata_is_refused():
    parent = _values()
    backbone, model, checkpoint = _parent_checkpoint(parent)
    del parent["learnable_temperature"]
    with pytest.raises(ValueError, match="Stage 1 lacks bound temperature"):
        restore_stage1_for_stage2(backbone, model, parent, _values(True), checkpoint)


def test_learnable_parent_missing_its_trained_scale_is_still_refused():
    # The helper must not weaken the Stage 1 loader to make a new child fit.
    backbone, model, checkpoint = _parent_checkpoint(_values(False))
    with pytest.raises(ValueError, match="loss_trainable_state does not cover"):
        restore_stage1_for_stage2(backbone, model, _values(True), _values(), checkpoint)


def test_fixed_parent_with_conflicting_saved_scale_is_refused():
    backbone, model, checkpoint = _parent_checkpoint(_values(True), trained_tau=0.07)
    with pytest.raises(ValueError, match="fixed temperature state disagrees"):
        restore_stage1_for_stage2(backbone, model, _values(False), _values(), checkpoint)
