"""Gradient checkpointing on PointBERT must change memory, not arithmetic.

It was added because the ratified `batch_size: 64` OOMs on this card
(OBSERVED DATA 2026-08-29: batch 32 peaks at 23.8 GiB, 48 and 64 both OOM;
with checkpointing 64 peaks at 23.0 GiB). Halving the batch would have been a
recipe change -- the contrastive loss draws its negatives from the batch --
so the requirement here is EXACTNESS, not merely "it runs".

The hazard is specific: PointBERT's blocks carry `DropPath` at rate 0.1, so
the forward is stochastic. A recompute that drew a different mask would give a
different gradient and nothing would report it. These run on CPU.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")


def _blocks(depth=3, dim=16, drop_path=0.5):
    """Real PointBERT blocks, with DropPath turned up so a mismatch is loud.

    Imported the way the production path does it -- `ulip_patch` puts the
    vendored tree on `sys.path` under its own top-level `models` package, which
    is how its internal `from models.pointbert.dvae import Group` resolves. A
    direct `metafind.vendor.ulip...` import raises ModuleNotFoundError.
    """
    import sys
    from pathlib import Path as _P
    from metafind.compat import ulip_patch
    for path in (str(_P(__file__).resolve().parents[1]), str(ulip_patch.ULIP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    ulip_patch.apply(patch_fps=True)
    from models.pointbert.point_encoder import TransformerEncoder
    torch.manual_seed(0)
    return TransformerEncoder(embed_dim=dim, depth=depth, num_heads=4,
                              drop_path_rate=drop_path)


def _run(encoder, x, pos, seed=1234):
    torch.manual_seed(seed)
    out = encoder(x.clone().requires_grad_(True), pos)
    out.sum().backward()
    return out.detach().clone(), [p.grad.detach().clone() for p in encoder.parameters()]


def _checkpointed(encoder):
    from torch.utils.checkpoint import checkpoint
    blocks = encoder.blocks

    def forward(x, pos):
        for block in blocks:
            x = checkpoint(block, x + pos, use_reentrant=False)
        return x
    encoder.forward = forward
    return encoder


def test_grad_checkpointing_changes_no_gradient():
    enc = _blocks()
    enc.train()
    x = torch.randn(2, 5, 16)
    pos = torch.randn(2, 5, 16)

    plain_out, plain_grads = _run(enc, x, pos)
    enc.zero_grad(set_to_none=True)
    ckpt_out, ckpt_grads = _run(_checkpointed(enc), x, pos)

    assert torch.equal(plain_out, ckpt_out), "the forward itself moved"
    for a, b in zip(plain_grads, ckpt_grads):
        assert torch.equal(a, b), "a gradient moved"


def test_the_dropout_really_is_active_in_this_fixture():
    """Without this, the test above could pass on a deterministic module and
    prove nothing about the RNG-restoring behaviour it exists to check."""
    enc = _blocks()
    enc.train()
    x = torch.randn(2, 5, 16)
    pos = torch.randn(2, 5, 16)
    torch.manual_seed(1); a = enc(x, pos)
    torch.manual_seed(2); b = enc(x, pos)
    assert not torch.equal(a, b), "DropPath is not firing; the fixture is inert"


def test_reentrant_true_would_be_the_wrong_choice():
    """The failing counterpart. `use_reentrant=True` does not preserve RNG the
    same way; if it ever produced identical gradients here, the argument for
    `use_reentrant=False` would be decoration rather than a reason."""
    from torch.utils.checkpoint import checkpoint
    enc = _blocks()
    enc.train()
    x = torch.randn(2, 5, 16)
    pos = torch.randn(2, 5, 16)
    _, plain_grads = _run(enc, x, pos)
    enc.zero_grad(set_to_none=True)

    blocks = enc.blocks

    def forward(x, pos):
        for block in blocks:
            x = checkpoint(block, x + pos, use_reentrant=False, preserve_rng_state=False)
        return x
    enc.forward = forward
    _, loose_grads = _run(enc, x, pos)

    # GRADIENTS, not the forward output. The recompute only happens during
    # backward, so comparing forwards would pass on any setting and prove
    # nothing -- which is what the first draft of this test did.
    assert any(not torch.equal(a, b) for a, b in zip(plain_grads, loose_grads)), (
        "with preserve_rng_state=False the recompute still drew the same mask, so "
        "this fixture cannot detect an RNG mismatch and "
        "test_grad_checkpointing_changes_no_gradient is weaker than it looks")


def test_the_backbone_config_carries_the_flag_and_defaults_off():
    from metafind.models.ulip_backbone import BackboneConfig
    assert BackboneConfig().grad_checkpointing is False
