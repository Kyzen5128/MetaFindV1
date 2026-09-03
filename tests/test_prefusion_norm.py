"""FusionConfig.prefusion_norm: present modalities enter Fusion at unit norm.

[AUDIT 2026-09-03 C8] Measured on the 10-epoch pilot: the vectors entering
Fusion had norms text 37.1 / image 40.2 / pc 139.1, and permuting the gallery's
pc alone took every Table 1 condition to single digits. The paper says nothing
about normalisation before Fusion; this flag is an IMPLEMENTATION CHOICE that
must do exactly one thing, named here so a later edit cannot widen it.
"""
from __future__ import annotations

import torch

from metafind.models.fusion import FusionConfig, ModalityFusion


def _batch(d=16, b=5, scale=(37.0, 40.0, 139.0)):
    g = torch.Generator().manual_seed(0)
    e = {m: torch.randn(b, d, generator=g) * s
         for m, s in zip(("text", "image", "pc"), scale)}
    present = torch.tensor([[True, True, True],
                            [True, False, True],
                            [False, True, True],
                            [True, True, False],
                            [False, False, True]])
    return e, present


def test_present_modalities_enter_at_unit_norm_and_mask_tokens_are_untouched():
    cfg = FusionConfig(dim=16, kind="transformer", prefusion_norm=True,
                       hidden=32, n_heads=2, n_layers=1)
    f = ModalityFusion(cfg)
    e, present = _batch()
    x = f._stack(e, present)                     # (B, 3, D)
    norms = x.norm(dim=-1)
    assert torch.allclose(norms[present], torch.ones_like(norms[present]), atol=1e-5)
    absent = ~present
    for i, m in enumerate(("text", "image", "pc")):
        rows = absent[:, i]
        if rows.any():
            assert torch.allclose(x[rows, i], f.mask_tokens[i].expand(int(rows.sum()), -1))


def test_default_leaves_the_inputs_alone():
    cfg = FusionConfig(dim=16, kind="transformer", hidden=32, n_heads=2, n_layers=1)
    f = ModalityFusion(cfg)
    e, present = _batch()
    x = f._stack(e, present)
    for i, m in enumerate(("text", "image", "pc")):
        rows = present[:, i]
        assert torch.equal(x[rows, i], e[m][rows])


def test_the_flag_changes_the_output_when_scales_differ():
    """If it did nothing, the norm disparity it exists to remove would be
    invisible to every downstream number."""
    e, present = _batch()
    torch.manual_seed(1)
    off = ModalityFusion(FusionConfig(dim=16, hidden=32, n_heads=2, n_layers=1))
    torch.manual_seed(1)
    on = ModalityFusion(FusionConfig(dim=16, hidden=32, n_heads=2, n_layers=1,
                                     prefusion_norm=True))
    with torch.no_grad():
        a, b = off(e, present), on(e, present)
    assert not torch.allclose(a, b)
