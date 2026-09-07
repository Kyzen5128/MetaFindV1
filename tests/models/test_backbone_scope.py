"""Freezing has two halves and only one of them is about gradients.

`requires_grad=False` stops the optimizer. It does NOT stop dropout. ULIP-2's
PointBERT config sets `drop_path_rate: 0.1` over 18 blocks, so a point encoder
left in `train()` applies stochastic depth on every forward pass -- and the
gallery index built from it is non-deterministic while having exactly the right
shape, the right count, and no NaNs.

That is why `is_frozen()` checks both, and why these tests assert both.

The bug these pin was introduced BY the P0-1 fix, not found alongside it: Stage 1's
checkpoint can only restore its point-encoder section into a backbone whose point
encoder is trainable, so the gallery index and Stage 2 had to be built with
`train_scope="point_encoder_and_fuser"` -- and then nothing switched them back.

Loading real ULIP-2 weights needs a 9.5 GB checkpoint, so these use a stand-in
with the same structure: a `.model` holding a `point_encoder` with dropout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class FakeULIP(nn.Module):
    """The shape ULIPBackbone drives: a point_encoder plus a pc_projection."""

    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(6, 8), nn.Dropout(0.5), nn.Linear(8, 8))
        self.pc_projection = nn.Parameter(torch.randn(8, 4))
        self.other = nn.Linear(4, 4)          # stands in for the CLIP halves


class Backbone:
    """ULIPBackbone's scope logic, verbatim in behaviour, without the 9.5 GB."""

    def __init__(self, scope: str) -> None:
        self.model = FakeULIP()
        self.scope = scope
        self._apply()

    def _point_parameters(self):
        return list(self.model.point_encoder.parameters()) + [self.model.pc_projection]

    def _apply(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        if self.scope == "fuser_only":
            return
        for p in self._point_parameters():
            p.requires_grad_(True)
        self.model.point_encoder.train()

    def set_train_scope(self, scope: str) -> None:
        self.scope = scope
        self._apply()

    def is_frozen(self) -> bool:
        return (not any(p.requires_grad for p in self.model.parameters())
                and not any(m.training for m in self.model.modules()))


# --- the two halves are genuinely different ---------------------------------

def test_a_trainable_scope_leaves_the_point_encoder_in_train_mode():
    """The premise. If this ever stops being true the rest is moot."""
    b = Backbone("point_encoder_and_fuser")
    assert b.model.point_encoder.training is True
    assert any(p.requires_grad for p in b.model.parameters())
    assert not b.is_frozen()


def test_train_mode_makes_the_forward_pass_non_deterministic():
    """[NEGATIVE INJECTION] Why eval() matters independently of requires_grad.

    With dropout live, two identical inputs give two different embeddings. An
    index built this way has the right shape, the right count and no NaNs.
    """
    b = Backbone("point_encoder_and_fuser")
    x = torch.randn(4, 6)
    torch.manual_seed(0)
    with torch.no_grad():                     # no_grad does NOT disable dropout
        a = b.model.point_encoder(x)
        c = b.model.point_encoder(x)
    assert not torch.equal(a, c), (
        "dropout did not fire -- this test cannot detect the bug it exists for")


def test_freezing_makes_the_forward_pass_deterministic_again():
    b = Backbone("point_encoder_and_fuser")
    b.set_train_scope("fuser_only")
    x = torch.randn(4, 6)
    with torch.no_grad():
        a = b.model.point_encoder(x)
        c = b.model.point_encoder(x)
    assert torch.equal(a, c)


def test_requires_grad_alone_would_not_have_caught_it():
    """The check that was NOT enough.

    Turning off `requires_grad` by hand -- which is what the old Stage 2 guard
    was trying to do -- leaves the module in train() and the dropout live.
    """
    b = Backbone("point_encoder_and_fuser")
    for p in b.model.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in b.model.parameters())
    assert b.model.point_encoder.training is True, "still in train mode"
    assert not b.is_frozen(), "is_frozen() must not be satisfied by grads alone"


# --- set_train_scope round trip ---------------------------------------------

def test_set_train_scope_freezes_everything():
    b = Backbone("point_encoder_and_fuser")
    b.set_train_scope("fuser_only")
    assert b.is_frozen()
    assert not any(p.requires_grad for p in b.model.parameters())
    assert not any(m.training for m in b.model.modules())


def test_set_train_scope_can_go_back_the_other_way():
    """The restore path needs it trainable, so this must not be one-way."""
    b = Backbone("fuser_only")
    assert b.is_frozen()
    b.set_train_scope("point_encoder_and_fuser")
    assert not b.is_frozen()
    assert b.model.pc_projection.requires_grad


def test_the_clip_halves_stay_frozen_under_the_trainable_scope():
    """[U-34] `point_encoder_and_fuser` must not reach the CLIP towers."""
    b = Backbone("point_encoder_and_fuser")
    assert not b.model.other.weight.requires_grad
    assert b.model.other.training is False


# --- the real class exposes the same contract -------------------------------

def test_the_real_backbone_exposes_set_train_scope_and_is_frozen():
    """A rename would leave the two call sites silently unfrozen."""
    from metafind.models.ulip_backbone import ULIPBackbone

    assert callable(getattr(ULIPBackbone, "set_train_scope", None))
    assert callable(getattr(ULIPBackbone, "is_frozen", None))


def test_ulip_backbone_has_no_parameters_method():
    """Pins the exact shape of the no-op guard.

    Stage 2 held `for p in backbone.parameters() if hasattr(backbone, "parameters")`.
    The attribute does not exist, so the loop ran zero times and read as a
    freeze. If ULIPBackbone ever grows a `.parameters()`, that guard becomes
    live code again and this test should be revisited rather than deleted.
    """
    from metafind.models.ulip_backbone import ULIPBackbone

    assert not hasattr(ULIPBackbone, "parameters")
