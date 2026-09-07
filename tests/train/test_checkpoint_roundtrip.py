"""Stage 1's checkpoint must round-trip EVERY trainable tensor.

The bug these pin: `save_checkpoint(model, ...)` saved only the dual tower while
the optimizer also moved `backbone.trainable_parameters()` (PointBERT and
`pc_projection`) and `loss_fn.parameters()` (`logit_scale`). Nothing failed.
The save succeeded, the reload succeeded, every shape was right, and the
fine-tuned point encoder was discarded at the end of every epoch -- so the
gallery index and Stage 2 both rebuilt PointBERT from the ORIGINAL ULIP-2
weights and Stage 1's point-tower training changed nothing downstream.

Loading the real ULIP-2 backbone here would need a 9.5 GB checkpoint, so these
use stand-ins with the same SHAPE of problem: a module the optimizer touches
that is not the dual tower.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from metafind.train.stage1 import (  # noqa: E402
    CKPT_SECTIONS,
    assert_checkpoint_covers_optimizer,
    load_stage1_checkpoint,
    trainable_state_dict,
)


class FakeBackbone:
    """Stands in for ULIPBackbone: a `.model` plus `trainable_parameters()`."""

    def __init__(self, trainable: bool = True) -> None:
        self.model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class FakeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.659))


def tower() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 3))


def sections(backbone, model, loss_fn) -> dict:
    return {
        "backbone_trainable_state": trainable_state_dict(backbone.model),
        "tower_trainable_state": trainable_state_dict(model),
        "loss_trainable_state": trainable_state_dict(loss_fn),
    }


# --- the guard itself ------------------------------------------------------

def test_the_guard_accepts_a_complete_checkpoint():
    b, m, l = FakeBackbone(), tower(), FakeLoss()
    assert_checkpoint_covers_optimizer(b, m, l, sections(b, m, l))


def test_the_guard_rejects_a_backbone_that_is_not_saved():
    """The original bug, stated as a test.

    The optimizer holds the backbone's parameters; the checkpoint holds only
    the tower. Every tensor still has a home in memory, so nothing else in the
    pipeline can notice.
    """
    b, m, l = FakeBackbone(), tower(), FakeLoss()

    class Detached:
        """A backbone whose parameters are in the optimizer but in no section."""
        model = nn.Linear(4, 4)

        def trainable_parameters(self):
            return list(nn.Linear(7, 7).parameters())

    with pytest.raises(RuntimeError, match="belong to no module"):
        assert_checkpoint_covers_optimizer(Detached(), m, l, sections(b, m, l))


def test_the_guard_rejects_an_empty_optimizer():
    b, m, l = FakeBackbone(trainable=False), tower(), FakeLoss()
    for p in list(m.parameters()) + list(l.parameters()):
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        assert_checkpoint_covers_optimizer(b, m, l, sections(b, m, l))


# --- save -> rebuild -> restore -> bit-identical ----------------------------

def test_every_trainable_tensor_survives_a_round_trip(tmp_path):
    b, m, l = FakeBackbone(), tower(), FakeLoss()
    for p in list(b.model.parameters()) + list(m.parameters()):
        with torch.no_grad():
            p.add_(torch.randn_like(p))          # move them off initialisation
    l.logit_scale.data.fill_(3.1415)

    path = tmp_path / "stage1.pt"
    torch.save({**sections(b, m, l), "trainer_version": 2, "epoch": 0,
                "train_scope": "point_encoder_and_fuser"}, path)

    b2, m2, l2 = FakeBackbone(), tower(), FakeLoss()
    load_stage1_checkpoint(b2, m2, l2, path)

    for (n1, p1), (n2, p2) in zip(sorted(b.model.named_parameters()),
                                  sorted(b2.model.named_parameters())):
        assert n1 == n2 and torch.equal(p1, p2), f"backbone {n1} differs"
    for (n1, p1), (n2, p2) in zip(sorted(m.named_parameters()),
                                  sorted(m2.named_parameters())):
        assert n1 == n2 and torch.equal(p1, p2), f"tower {n1} differs"
    assert torch.equal(l.logit_scale, l2.logit_scale)


def test_a_tower_only_checkpoint_is_refused_not_silently_accepted(tmp_path):
    """v1 files are unrecoverable, so they must fail loudly rather than load."""
    b, m, l = FakeBackbone(), tower(), FakeLoss()
    path = tmp_path / "old.pt"
    torch.save({"trainable_state": trainable_state_dict(m),
                "trainer_version": 1}, path)
    with pytest.raises(ValueError, match="missing"):
        load_stage1_checkpoint(b, m, l, path)


def test_a_section_that_omits_a_trainable_tensor_is_refused(tmp_path):
    b, m, l = FakeBackbone(), tower(), FakeLoss()
    sec = sections(b, m, l)
    dropped = sorted(sec["backbone_trainable_state"])[0]
    del sec["backbone_trainable_state"][dropped]
    path = tmp_path / "partial.pt"
    torch.save({**sec, "trainer_version": 2}, path)

    with pytest.raises(ValueError, match="does not cover"):
        load_stage1_checkpoint(b, m, l, path)


def test_new_prefixes_admit_stage2_modules_but_nothing_else(tmp_path):
    """Stage 2 adds the ESSGNN and Eq. 6's lambda; a Stage 1 file cannot hold
    them. The exemption is per-name and declared, not a blanket strict=False."""
    b, m, l = FakeBackbone(), tower(), FakeLoss()
    path = tmp_path / "s1.pt"
    torch.save({**sections(b, m, l), "trainer_version": 2}, path)

    b2, l2 = FakeBackbone(), FakeLoss()
    m2 = nn.Module()
    m2.add_module("0", nn.Linear(2, 3))           # matches the saved tower
    m2.register_parameter("layout_weight", nn.Parameter(torch.tensor(1.0)))
    load_stage1_checkpoint(b2, m2, l2, path, new_prefixes=("layout_weight",))

    m3 = nn.Module()
    m3.add_module("0", nn.Linear(2, 3))
    m3.register_parameter("something_else", nn.Parameter(torch.tensor(1.0)))
    with pytest.raises(ValueError, match="does not cover"):
        load_stage1_checkpoint(FakeBackbone(), m3, FakeLoss(), path,
                               new_prefixes=("layout_weight",))


def test_the_section_names_are_the_ones_the_loader_requires():
    assert set(CKPT_SECTIONS) == {"backbone_trainable_state",
                                  "tower_trainable_state",
                                  "loss_trainable_state"}
