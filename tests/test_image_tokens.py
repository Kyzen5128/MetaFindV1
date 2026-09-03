"""FusionConfig.image_tokens: the image modality as K view tokens.

[AUDIT 2026-09-03 C4] Figure 1 draws K vectors per encoder entering one Fusion
Layer; the paper never says the views are averaged first. These tests pin what
the K-token path does and, above all, that image_tokens = 1 leaves the
pre-existing 3-slot construction untouched.
"""
from __future__ import annotations

import numpy as np
import torch

from metafind.models.fusion import FusionConfig, ModalityFusion


def _cfg(**kw):
    base = dict(dim=16, kind="transformer", hidden=32, n_heads=2, n_layers=1)
    base.update(kw)
    return FusionConfig(**base)


def test_k_equals_one_is_the_old_path_bit_for_bit():
    torch.manual_seed(0)
    a = ModalityFusion(_cfg())
    torch.manual_seed(0)
    b = ModalityFusion(_cfg(image_tokens=1))
    e = {m: torch.randn(4, 16) for m in ("text", "image", "pc")}
    present = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(a(e, present), b(e, present))
    assert not hasattr(b, "view_pos")


def test_token_sequence_shape_and_absent_views_carry_the_image_mask_token():
    K = 5
    f = ModalityFusion(_cfg(image_tokens=K))
    b = 3
    e = {"text": torch.randn(b, 16), "image": torch.randn(b, K, 16),
         "pc": torch.randn(b, 16),
         "image_present": torch.tensor([[1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]], dtype=torch.bool)}
    present = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    x, act = f._stack_tokens(e, present)
    assert x.shape == (b, 1 + K + 1, 16) and act.shape == (b, 1 + K + 1)
    # row 1: only view 0 real; views 1..4 are the image mask token
    for v in range(1, K):
        assert torch.equal(x[1, 1 + v], f.mask_tokens[1])
    assert torch.equal(x[1, 1], e["image"][1, 0])
    # row 2: image modality absent -> every view token is the mask token
    for v in range(K):
        assert torch.equal(x[2, 1 + v], f.mask_tokens[1])
    assert act[1].tolist() == [True, True, False, False, False, False, True]
    assert act[2].tolist() == [True, False, False, False, False, False, True]
    with torch.no_grad():
        out = f(e, present)
    assert out.shape == (b, 16) and torch.isfinite(out).all()


def test_gallery_with_every_view_present_equals_no_mask_given():
    K = 4
    f = ModalityFusion(_cfg(image_tokens=K))
    e = {"text": torch.randn(2, 16), "image": torch.randn(2, K, 16), "pc": torch.randn(2, 16)}
    present = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        a = f(e, present)
        b = f({**e, "image_present": torch.ones(2, K, dtype=torch.bool)}, present)
    assert torch.equal(a, b)


def test_non_transformer_refuses_tokens():
    try:
        ModalityFusion(_cfg(kind="mean", image_tokens=3))
    except ValueError as err:
        assert "transformer" in str(err)
    else:
        raise AssertionError("mean fusion accepted a token sequence")


def test_collate_and_split_embeds_carry_the_view_masks():
    from metafind.train.stage1 import collate, split_embeds

    K = 3
    items = []
    for i in range(2):
        items.append({"uid": f"u{i}", "text": np.zeros(16, np.float32),
                      "image": np.zeros((K, 16), np.float32),
                      "pc": np.zeros((10000, 6), np.float32),
                      "image_present": np.ones(K, bool),
                      "q_image": np.zeros((K, 16), np.float32),
                      "q_image_present": np.array([True, False, False])})
    batch = collate(items)
    assert batch["image"].shape == (2, K, 16)
    assert batch["image_present"].shape == (2, K) and batch["q_image_present"].shape == (2, K)

    class _BB:
        def encode_pc(self, x):
            return torch.zeros(x.size(0), 16)

    q, g = split_embeds(batch, _BB(), "cpu")
    assert g["image_present"].all()
    assert q["image_present"].tolist() == [[True, False, False]] * 2
    assert q is not g
