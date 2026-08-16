"""Tests for n10_train_stage1's GPU-free half.

The one that matters is L1-CKPT-TRAINABLE-ONLY. torch.save(state_dict()) on the
dual tower writes ViT-bigG-14 as well -- 2.5B frozen parameters, 10.2 GB -- and
across Table 3's eleven runs that is 112 GB against 1.9 GB, on a shared volume.
The failure is silent: correct training, correct results, files sixty times
bigger, discovered after the tenth ablation.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from metafind.train.stage1 import (
    build_model,
    collate,
    load_protocols,
    trainable_state_dict,
)


class Tower(nn.Module):
    """A frozen 'backbone' and a trainable head, in the same module."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen_backbone = nn.Linear(64, 64)
        self.trainable_fusion = nn.Linear(64, 8)
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)


# --- L1-CKPT-TRAINABLE-ONLY ------------------------------------------------

def test_only_trainable_parameters_are_saved():
    state = trainable_state_dict(Tower())
    assert all("trainable_fusion" in k for k in state)
    assert not any("frozen_backbone" in k for k in state)


def test_saving_the_whole_state_dict_is_much_larger():
    """[the injection] The whole point is the ratio, so measure it."""
    m = Tower()
    ours = sum(v.numel() for v in trainable_state_dict(m).values())
    whole = sum(v.numel() for v in m.state_dict().values())
    assert whole > 5 * ours


def test_unfreezing_the_backbone_puts_it_in_the_checkpoint():
    """Keyed off requires_grad, so it tracks the actual training scope rather
    than a name list that goes stale silently on a rename."""
    m = Tower()
    assert not any("frozen_backbone" in k for k in trainable_state_dict(m))
    for p in m.frozen_backbone.parameters():
        p.requires_grad_(True)
    assert any("frozen_backbone" in k for k in trainable_state_dict(m))


def test_the_saved_tensors_are_detached_and_on_cpu():
    """A checkpoint holding graph references keeps the whole autograd graph
    alive, and one on the GPU cannot be loaded on a machine without one."""
    for v in trainable_state_dict(Tower()).values():
        assert v.device.type == "cpu"
        assert not v.requires_grad


def test_a_name_prefix_filter_would_miss_a_renamed_module():
    """Why requires_grad and not a prefix list: renaming a module keeps the
    checkpoint saving and loading while quietly omitting a trained tensor."""
    m = Tower()
    m.add_module("newly_added_head", nn.Linear(8, 4))
    state = trainable_state_dict(m)
    assert any("newly_added_head" in k for k in state), (
        "a prefix list would have skipped this module entirely"
    )


# --- protocol refusals -----------------------------------------------------

def protocols(tmp_path, **over):
    enc = {"status": "resolved", "actual_clip_train_scope": "frozen",
           "image_aggregation": "mean", "missing_modality_representation": "learned_token"}
    train = {"status": "resolved", "fusion": "masked_mlp",
             "tower_sharing": "shared_backbone_separate_fusion",
             "allow_all_masked": True, "similarity": "cosine",
             "hyperparameter_config_hash": "abc123"}
    hp = {"sha256": "abc123", "values": {
        "optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 0.1,
        "scheduler": "cosine", "batch_size": 64, "epochs": 50, "p_mask": 0.30,
        "init_temperature": 0.07, "learnable_temperature": True,
        "max_logit_scale": 100.0, "seed": 1}}
    enc.update(over.get("enc", {}))
    train.update(over.get("train", {}))
    hp.update(over.get("hp", {}))
    (tmp_path / "stage1_encoding_protocol.json").write_text(json.dumps(enc))
    (tmp_path / "stage1_protocol.json").write_text(json.dumps(train))
    (tmp_path / "stage1_hyperparameters.json").write_text(json.dumps(hp))


def use(monkeypatch, tmp_path):
    import metafind.train.stage1 as s

    monkeypatch.setattr(s.paths, "OUTPUTS", tmp_path)


def test_resolved_protocols_load(monkeypatch, tmp_path):
    protocols(tmp_path)
    use(monkeypatch, tmp_path)
    enc, train, hp = load_protocols()
    assert enc["actual_clip_train_scope"] == "frozen"
    assert train["fusion"] == "masked_mlp"


def test_a_hyperparameter_hash_mismatch_stops_the_run(monkeypatch, tmp_path):
    """G3 dereferences this hash. A protocol pointing at a different artifact
    than the one on disk means the reported hyperparameters are not the ones
    that trained the model."""
    protocols(tmp_path, train={"hyperparameter_config_hash": "notthesame"})
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as exc:
        load_protocols()
    assert "different hyperparameter artifact" in str(exc.value)


@pytest.mark.parametrize("which", ["enc", "train"])
def test_an_unresolved_protocol_stops_the_run(monkeypatch, tmp_path, which):
    protocols(tmp_path, **{which: {"status": "unresolved"}})
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        load_protocols()


def test_a_missing_hyperparameter_stops_the_run(monkeypatch, tmp_path):
    """[U-22] The artifact must NAME every value; the run refuses on a partial
    one, because a default supplied here would never reach the report."""
    protocols(tmp_path)
    hp = json.loads((tmp_path / "stage1_hyperparameters.json").read_text())
    del hp["values"]["learning_rate"]
    (tmp_path / "stage1_hyperparameters.json").write_text(json.dumps(hp))
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as exc:
        load_protocols()
    assert "learning_rate" in str(exc.value)


def test_a_missing_protocol_names_its_writer(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        load_protocols()
    assert "n05b" in str(exc.value)


# --- batching --------------------------------------------------------------

def test_collate_stacks_and_keeps_uids_aligned():
    import numpy as np

    batch = [{"uid": f"u{i}", "text": np.zeros(4, np.float32),
              "image": np.ones(4, np.float32),
              "pc": np.full((10, 6), i, np.float32)} for i in range(3)]
    out = collate(batch)
    assert out["uid"] == ["u0", "u1", "u2"]
    assert out["text"].shape == (3, 4)
    assert out["pc"].shape == (3, 10, 6)
    # row i must still be asset i -- the loss pairs query i with gallery i
    assert torch.equal(out["pc"][2], torch.full((10, 6), 2.0))


# --- L1-LOSS-STAGE1-UNIDIRECTIONAL -----------------------------------------

def built(tmp_path, **over):
    protocols(tmp_path, **over)
    return build_model(*(json.loads((tmp_path / f).read_text()) for f in (
        "stage1_encoding_protocol.json", "stage1_protocol.json",
        "stage1_hyperparameters.json")))


def test_stage1_loss_is_query_to_gallery_only(tmp_path):
    """[L1-LOSS-STAGE1-UNIDIRECTIONAL] Eq. 5 has one direction; Eq. 7a/7b's
    symmetric form is Stage 2's, and the paper is explicit about the difference.

    This test exists because setting bidirectional=True in the trainer passed
    the entire suite -- the rule was written in validation_plan.yaml and
    enforced by nothing.
    """
    _, loss = built(tmp_path)
    assert loss.cfg.bidirectional is False


def test_the_loss_actually_computes_one_direction(tmp_path):
    """Not just the flag: swapping the two arguments must change the value.

    A symmetric objective is invariant to the swap, so this distinguishes the
    configuration from the behaviour -- a flag read by nothing would pass the
    test above and fail this one.
    """
    _, loss = built(tmp_path)
    torch.manual_seed(0)
    q = torch.randn(8, 16)
    g = torch.randn(8, 16)
    forward = loss(q, g)["loss"]
    swapped = loss(g, q)["loss"]
    assert not torch.isclose(forward, swapped), (
        "the loss is symmetric under a swap, which is Eq. 7's behaviour, not Eq. 5's"
    )


def test_the_trainer_uses_the_recorded_fusion_and_sharing(tmp_path):
    """n10 must not decide these; n09 wrote them and the report cites them."""
    model, _ = built(tmp_path)
    assert model.cfg.tower_sharing == "shared_backbone_separate_fusion"
    assert model.cfg.query_fusion.kind == "masked_mlp"


@pytest.mark.parametrize("rule,expect_zero_pad", [
    ("learned_token", False),
    ("zero_pad", True),
])
def test_the_missing_modality_rule_comes_from_the_encoding_protocol(
        tmp_path, rule, expect_zero_pad):
    """[U-11] 2.6 rules out zero-padding and names no replacement. n05b chose
    learned_token; the trainer must carry that choice rather than let a
    FusionConfig default decide -- which is how it was being decided before
    n05b existed. `zero_pad` is Table 3's "Padding missing modalities with 0"."""
    model, _ = built(tmp_path, enc={"missing_modality_representation": rule})
    assert model.cfg.query_fusion.zero_pad is expect_zero_pad
