"""L1 tests for the dual-tower model (sec. 2.2/2.4/2.6, Eq. 6).

Covers L1-LAMBDA, the gallery freezing schedule, scene dropout, and the Eq. 6
residual. Each check asserts on content and carries a negative injection.
"""

from __future__ import annotations

import pytest
import torch

from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower, QueryTower
from metafind.models.essgnn import ESSGNNConfig
from metafind.models.fusion import MODALITIES, FusionConfig

D = 32


def cfg(**kw) -> DualTowerConfig:
    base = dict(
        dim=D,
        query_fusion=FusionConfig(dim=D, hidden=64, n_heads=4, n_layers=1),
        gallery_fusion=FusionConfig(dim=D, hidden=64, n_heads=4, n_layers=1),
        essgnn=ESSGNNConfig(node_feat_dim=16, edge_feat_dim=8, hidden_dim=16, out_dim=D, n_layers=2, use_io_projections=True),
    )
    return DualTowerConfig(**{**base, **kw})


def embeds(b: int = 4, d: int = D, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(b, d, generator=g) for m in MODALITIES}


def scene(n: int = 6, cfgg: ESSGNNConfig | None = None, seed: int = 0):
    cfgg = cfgg or ESSGNNConfig(node_feat_dim=16, edge_feat_dim=8, hidden_dim=16, out_dim=D, n_layers=2, use_io_projections=True)
    g = torch.Generator().manual_seed(seed)
    ei = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j]).T
    return (
        torch.randn(n, cfgg.node_feat_dim, generator=g),
        torch.randn(n, 3, generator=g) * 2,
        ei,
        torch.randn(ei.size(1), cfgg.edge_feat_dim, generator=g),
    )


# --------------------------------------------------------------- Eq. 6


def test_eq6_is_a_residual_on_the_fused_query():
    """e_query = Fusion(...) + lambda * e_layout, exactly."""
    torch.manual_seed(0)
    model = MetaFindDualTower(cfg()).eval()
    e = embeds()
    layout = torch.randn(4, D)

    with torch.no_grad():
        without = model.query(e, layout=None)
        with_layout = model.query(e, layout=layout)
        expected = without + model.query.lam * layout
    assert torch.allclose(with_layout, expected, atol=1e-6)


def test_lambda_is_learnable():
    """L1-LAMBDA: sec. 2.6 calls lambda a learnable scalar controlling layout."""
    model = MetaFindDualTower(cfg())
    assert isinstance(model.query.log_lambda, torch.nn.Parameter)
    assert model.query.lam.numel() == 1, "lambda must be a scalar, not a vector"

    e = embeds()
    model.query(e, layout=torch.randn(4, D)).sum().backward()
    assert model.query.log_lambda.grad is not None
    assert model.query.log_lambda.grad.abs().item() > 0


def test_lambda_negative_injection():
    """A constant lambda must be detectable, or the previous test proves nothing."""
    model = MetaFindDualTower(cfg())
    with torch.no_grad():
        model.query.log_lambda.zero_()
    e = embeds()
    with torch.no_grad():
        a = model.query(e, layout=torch.randn(4, D))
        b = model.query(e, layout=None)
    assert torch.allclose(a, b, atol=1e-6), "lambda=0 should nullify the layout term"


def test_layout_free_query_ignores_the_layout_branch():
    """Table 1 evaluates layout-free queries; passing None must skip Eq. 6 entirely."""
    torch.manual_seed(0)
    model = MetaFindDualTower(cfg()).eval()
    e = embeds()
    with torch.no_grad():
        a = model.query(e, layout=None)
        model.query.log_lambda.fill_(999.0)
        b = model.query(e, layout=None)
    assert torch.allclose(a, b), "lambda leaked into a layout-free query"


def test_layout_without_branch_raises():
    model = MetaFindDualTower(cfg(use_layout=False))
    with pytest.raises(ValueError, match="use_layout=False"):
        model.query(embeds(), layout=torch.randn(4, D))


def test_essgnn_width_mismatch_is_rejected_at_construction():
    """Eq. 6 adds the two vectors, so a width mismatch must fail loudly and early."""
    with pytest.raises(ValueError, match="Eq. 6"):
        QueryTower(cfg(essgnn=ESSGNNConfig(node_feat_dim=16, edge_feat_dim=8, out_dim=D + 1, use_io_projections=True)))


# --------------------------------------------------------------- scene dropout


def test_scene_dropout_rate_is_30_percent_per_batch():
    """U-32: sec. 2.6 drops the layout "in 30% of batches", so the unit is a batch.

    Measuring one large batch cannot see this rate at all -- under batch
    granularity a single batch is entirely dropped or entirely kept, so the
    within-batch mean is 0.0 or 1.0. The rate only exists across batches.
    """
    model = MetaFindDualTower(cfg())
    g = torch.Generator().manual_seed(0)
    dropped = [
        bool(model.sample_scene_dropout(8, generator=g)[0].item()) for _ in range(20_000)
    ]
    rate = sum(dropped) / len(dropped)
    assert abs(rate - 0.30) < 0.01, f"per-batch scene dropout rate {rate:.4f}"


def test_scene_dropout_batch_granularity_is_uniform_within_a_batch():
    """The whole point of batch granularity: every row shares the condition."""
    model = MetaFindDualTower(cfg())
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        mask = model.sample_scene_dropout(16, generator=g)
        assert mask.unique().numel() == 1, "a batch-level draw must not vary within the batch"


def test_scene_dropout_sample_granularity_varies_within_a_batch():
    """The variant stays available and must behave differently (U-32)."""
    model = MetaFindDualTower(cfg(scene_dropout_granularity="sample"))
    mask = model.sample_scene_dropout(200_000, generator=torch.Generator().manual_seed(0))
    rate = mask.float().mean().item()
    assert abs(rate - 0.30) < 0.01, f"per-sample scene dropout rate {rate:.4f}"
    assert mask.unique().numel() == 2, "independent draws should produce both values"


def test_scene_dropout_suppresses_only_the_marked_rows():
    torch.manual_seed(0)
    model = MetaFindDualTower(cfg()).eval()
    e = embeds()
    layout = torch.randn(4, D)
    drop = torch.tensor([True, False, True, False])

    with torch.no_grad():
        out = model.query(e, layout=layout, drop_layout=drop)
        no_layout = model.query(e, layout=None)
        full = model.query(e, layout=layout)

    assert torch.allclose(out[drop], no_layout[drop], atol=1e-6), "dropped rows kept their layout"
    assert torch.allclose(out[~drop], full[~drop], atol=1e-6), "kept rows lost their layout"


def test_scene_dropout_shape_contract():
    model = MetaFindDualTower(cfg())
    with pytest.raises(ValueError, match="drop_layout"):
        model.query(embeds(), layout=torch.randn(4, D), drop_layout=torch.zeros(3, dtype=torch.bool))


# --------------------------------------------------------------- gallery tower


def test_gallery_requires_every_modality():
    """An incomplete gallery entry is a data error; mask-filling it would corrupt the index."""
    model = MetaFindDualTower(cfg())
    e = embeds()
    with pytest.raises(ValueError, match="modality-complete"):
        model.gallery({**e, "image": None})


def test_freeze_gallery_stops_gradients_and_disables_train_mode():
    """Sec. 2.6: Stage 2 trains the query fuser and ESSGNN only."""
    model = MetaFindDualTower(cfg())
    assert not model.gallery_is_frozen()

    model.freeze_gallery()
    assert model.gallery_is_frozen()
    assert not model.gallery.training, "a frozen tower must not stay in train mode"

    q, g = model(embeds(), embeds(seed=1))
    (q.sum() + g.sum()).backward()
    assert all(p.grad is None for p in model.gallery.parameters()), "gallery received gradient"
    assert any(p.grad is not None for p in model.query.parameters()), "query got no gradient"


def test_freeze_is_reversible():
    model = MetaFindDualTower(cfg())
    model.freeze_gallery()
    model.freeze_gallery(False)
    assert not model.gallery_is_frozen()
    assert model.gallery.training


def test_trainable_parameters_shrinks_after_freezing():
    model = MetaFindDualTower(cfg())
    before = len(model.trainable_parameters())
    model.freeze_gallery()
    after = len(model.trainable_parameters())
    assert after < before, f"freezing changed nothing: {before} -> {after}"


# --------------------------------------------------------------- end to end


def test_forward_returns_aligned_towers():
    model = MetaFindDualTower(cfg())
    q, g = model(embeds(), embeds(seed=1))
    assert q.shape == g.shape == (4, D)
    assert torch.isfinite(q).all() and torch.isfinite(g).all()


def test_layout_path_end_to_end_with_batched_scenes():
    """Two scenes, one layout vector each, fed through Eq. 6."""
    torch.manual_seed(0)
    model = MetaFindDualTower(cfg())
    nf, pos, ei, ea = scene(6)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    layout = model.query.encode_layout(nf, pos, ei, ea, batch=batch)
    assert layout.shape == (2, D)

    q = model.query(embeds(b=2), layout=layout)
    assert q.shape == (2, D)
    q.sum().backward()
    assert model.query.log_lambda.grad is not None
    assert any(p.grad is not None for p in model.query.layout_encoder.parameters())


def test_single_scene_layout_gets_a_batch_dimension():
    model = MetaFindDualTower(cfg())
    nf, pos, ei, ea = scene(6)
    assert model.query.encode_layout(nf, pos, ei, ea).shape == (1, D)


def test_layout_is_translation_invariant_end_to_end():
    """The whole point of ESSGNN, checked at the level the query tower sees."""
    torch.manual_seed(0)
    model = MetaFindDualTower(cfg()).eval()
    nf, pos, ei, ea = scene(6)
    with torch.no_grad():
        a = model.query.encode_layout(nf, pos, ei, ea)
        b = model.query.encode_layout(nf, pos + 1000.0, ei, ea)
    err = (a - b).abs().max().item()
    assert err < 1e-3, f"layout moved by {err:.3e} under pure translation"
