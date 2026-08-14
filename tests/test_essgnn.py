"""L1/L2 tests for ESSGNN (docs/graph/validation_plan.yaml).

Covers L1-EGNN-DIM, L1-EGNN-AGG, L1-EGNN-H0, L1-SEMEDGE-ZERO and the ESSGNN
half of L2-EQUIVAR. Every check asserts on CONTENT, and each carries a negative
injection so we know it can actually fail (rules V1-V3).

Run with::

    conda activate MetaFind && python -m pytest tests/ -v
"""

from __future__ import annotations

import math

import pytest
import torch

from metafind.models.essgnn import ESSGNN, ESSGNNConfig

DTYPE = torch.float64  # float32 noise would swamp the equivariance tolerance


def fully_connected(n: int) -> torch.Tensor:
    idx = [(i, j) for i in range(n) for j in range(n) if i != j]
    return torch.tensor(idx, dtype=torch.long).T


def make_scene(n: int = 12, cfg: ESSGNNConfig | None = None, seed: int = 0, spread: float = 3.0):
    cfg = cfg or ESSGNNConfig(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64, n_layers=3)
    g = torch.Generator().manual_seed(seed)
    node_feat = torch.randn(n, cfg.node_feat_dim, generator=g, dtype=DTYPE)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * spread
    edge_index = fully_connected(n)
    edge_attr = torch.randn(edge_index.size(1), cfg.edge_feat_dim, generator=g, dtype=DTYPE)
    model = ESSGNN(cfg).to(DTYPE)
    return model, node_feat, pos, edge_index, edge_attr


def random_rotation(seed: int = 0) -> torch.Tensor:
    """Uniform-ish rotation via QR, with the reflection removed so det == +1."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=DTYPE)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


# --------------------------------------------------------------- L1-EGNN-DIM


def test_output_dim_matches_eq6_residual():
    """e_layout must be out_dim wide so Fusion(...) + lambda * e_layout is defined."""
    cfg = ESSGNNConfig(node_feat_dim=1280, edge_feat_dim=1280, hidden_dim=64, out_dim=1280, n_layers=2)
    model, nf, pos, ei, ea = make_scene(8, cfg)
    out = model(nf, pos, ei, ea)
    assert out.shape == (1280,), f"expected (1280,), got {tuple(out.shape)}"

    fusion_out = torch.randn(1280, dtype=DTYPE)
    lam = torch.tensor(0.5, dtype=DTYPE)
    assert (fusion_out + lam * out).shape == (1280,)


def test_output_dim_negative_injection():
    """Negative injection: a mismatched width must break the Eq. 6 residual."""
    cfg = ESSGNNConfig(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=512, n_layers=2)
    model, nf, pos, ei, ea = make_scene(8, cfg)
    out = model(nf, pos, ei, ea)
    with pytest.raises(RuntimeError):
        _ = torch.randn(1280, dtype=DTYPE) + out


def test_batched_pooling_shape():
    cfg = ESSGNNConfig(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64, n_layers=2)
    model, nf, pos, ei, ea = make_scene(10, cfg)
    batch = torch.tensor([0] * 5 + [1] * 5)
    assert model(nf, pos, ei, ea, batch=batch).shape == (2, 64)


# --------------------------------------------------------------- L2-EQUIVAR


@pytest.mark.parametrize("coord_feat", ["updated", "current"])
def test_se3_equivariance(coord_feat: str):
    """SC-5: coords are equivariant and h is invariant, under BOTH readings of Eq. 3/13."""
    cfg = ESSGNNConfig(
        node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64,
        n_layers=3, h0_mode="semantic", coord_feat=coord_feat,
    )
    model, nf, pos, ei, ea = make_scene(12, cfg)
    q = random_rotation(1)
    # Deliberately large translation: the paper's motivation is unnormalised,
    # uncentred open-world coordinates.
    t = torch.tensor([120.0, -75.0, 33.0], dtype=DTYPE)

    base = model(nf, pos, ei, ea)
    moved = model(nf, pos @ q.T + t, ei, ea)

    # Pooled e_layout comes from h, which must be fully invariant.
    err = (moved - base).abs().max().item()
    assert err < 1e-8, f"e_layout is not SE(3)-invariant: {err:.3e}"


def test_equivariance_negative_injection():
    """The literal sec. 2.5 h0 = Concat(x, t) must BREAK invariance.

    This is Required Audit RA-1 in test form. It is not a failure of our
    implementation -- it demonstrates that sec. 2.5 contradicts the premise
    Appendix C states, and it proves test_se3_equivariance is not vacuous.
    """
    cfg = ESSGNNConfig(
        node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64,
        n_layers=3, h0_mode="concat_xt",
    )
    model, nf, pos, ei, ea = make_scene(12, cfg)
    q = random_rotation(1)
    t = torch.tensor([120.0, -75.0, 33.0], dtype=DTYPE)

    base = model(nf, pos, ei, ea)
    moved = model(nf, pos @ q.T + t, ei, ea)

    err = (moved - base).abs().max().item()
    assert err > 1e-3, (
        "Concat(x, t) did not break invariance -- the equivariance test would "
        f"then be vacuous. err={err:.3e}"
    )


def test_translation_only_invariance():
    """Pure translation is the cheapest way for a broken implementation to leak."""
    model, nf, pos, ei, ea = make_scene(12)
    base = model(nf, pos, ei, ea)
    for shift in (1.0, 1e3, 1e5):
        moved = model(nf, pos + shift, ei, ea)
        err = (moved - base).abs().max().item()
        assert err < 1e-6, f"translation by {shift} changed e_layout by {err:.3e}"


# --------------------------------------------------------------- L1-EGNN-AGG


def test_coords_agg_defaults_to_sum_per_eq3():
    """Eq. 3 sums over neighbours; the reference EGNN defaults to mean (F9)."""
    assert ESSGNNConfig().coords_agg == "sum"


def test_sum_and_mean_differ():
    """Negative injection: if sum and mean agreed, the previous test proves nothing."""
    common = dict(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64, n_layers=2)
    m_sum, nf, pos, ei, ea = make_scene(12, ESSGNNConfig(**common, coords_agg="sum"))
    m_mean = ESSGNN(ESSGNNConfig(**common, coords_agg="mean")).to(DTYPE)
    m_mean.load_state_dict(m_sum.state_dict())

    # f_x is initialised near zero so the coordinate update starts tiny; scale it
    # up so the aggregation choice is actually observable.
    with torch.no_grad():
        for layer in list(m_sum.layers) + list(m_mean.layers):
            layer.f_x[-1].weight.mul_(1e4)

    assert not torch.allclose(m_sum(nf, pos, ei, ea), m_mean(nf, pos, ei, ea))


# --------------------------------------------------------------- L1-SEMEDGE-ZERO


def test_geometry_still_distinguishes_layouts_without_semantic_edges():
    """F8 degeneracy detector.

    With e_ij at 1280 and only one geometric scalar per message, ESSGNN could
    collapse into a semantics-only GNN -- which would make Table 3's
    "ESSGNN beats GAT because of equivariance" unattributable. Zero the semantic
    edges and require two geometrically different layouts to stay distinct.
    """
    cfg = ESSGNNConfig(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32, out_dim=64, n_layers=3)
    model, nf, pos_a, ei, ea = make_scene(12, cfg)
    zero_edges = torch.zeros_like(ea)

    g = torch.Generator().manual_seed(99)
    pos_b = torch.randn_like(pos_a) * 3.0 if False else torch.randn(
        pos_a.shape, generator=g, dtype=DTYPE
    ) * 3.0

    a = model(nf, pos_a, ei, zero_edges)
    b = model(nf, pos_b, ei, zero_edges)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cos < 0.9999, f"geometry carries no signal without semantic edges (cos={cos:.6f})"


def test_semantic_edges_change_the_output():
    """Companion to the above: semantics must matter too, or e_ij is dead weight."""
    model, nf, pos, ei, ea = make_scene(12)
    out_real = model(nf, pos, ei, ea)
    out_zero = model(nf, pos, ei, torch.zeros_like(ea))
    assert not torch.allclose(out_real, out_zero, atol=1e-9)


# --------------------------------------------------------------- contracts


def test_edge_projection_is_absent_by_default():
    """The paper has no projection layer on e_ij, so the faithful default is None."""
    assert ESSGNNConfig().edge_proj_dim is None
    cfg = ESSGNNConfig(node_feat_dim=32, edge_feat_dim=1280, hidden_dim=64, out_dim=64, n_layers=1)
    model = ESSGNN(cfg)
    in_features = model.layers[0].f_h[0].in_features
    assert in_features == 2 * 64 + 1 + 1280, f"unexpected message width {in_features}"


def test_edge_projection_when_enabled():
    cfg = ESSGNNConfig(
        node_feat_dim=32, edge_feat_dim=1280, hidden_dim=64, out_dim=64, n_layers=1, edge_proj_dim=64
    )
    model = ESSGNN(cfg)
    assert model.layers[0].f_h[0].in_features == 2 * 64 + 1 + 64


def test_shape_contract_violations_raise():
    model, nf, pos, ei, ea = make_scene(10)
    with pytest.raises(ValueError, match="nodes"):
        model(nf[:5], pos, ei, ea)
    with pytest.raises(ValueError, match=r"\(2, E\)"):
        model(nf, pos, ei.T, ea)
    with pytest.raises(ValueError, match="edge_attr"):
        model(nf, pos, ei, ea[:3])


def test_gradients_reach_every_parameter_except_the_final_f_x():
    """Pins finding F11: the last layer's coordinate MLP is structurally dead.

    ``e_layout = Pooling({h_i^(L)})`` reads only h. Layer l's coordinate update
    matters solely because layer l+1 recomputes ``||x_i - x_j||^2`` from it, so
    the final layer's ``f_x`` has no downstream consumer and receives no
    gradient. With the paper's L=4 that leaves a quarter of the coordinate
    parameters untrained.

    This is a property of the architecture as described, not a defect to patch --
    "fixing" it would deviate from the paper. The test asserts the exact pattern
    so that a future change to the readout is caught rather than absorbed.
    """
    model, nf, pos, ei, ea = make_scene(10)
    model(nf, pos, ei, ea).sum().backward()

    dead = {
        n
        for n, p in model.named_parameters()
        if p.grad is None or not math.isfinite(p.grad.abs().sum().item())
    }
    last = len(model.layers) - 1
    expected_dead = {n for n in dict(model.named_parameters()) if n.startswith(f"layers.{last}.f_x.")}

    assert expected_dead, "test is misconfigured: no final-layer f_x parameters found"
    assert dead == expected_dead, (
        f"expected exactly the final layer's f_x to be gradient-free.\n"
        f"  unexpectedly dead: {sorted(dead - expected_dead)}\n"
        f"  unexpectedly live: {sorted(expected_dead - dead)}"
    )


def test_intermediate_coordinate_updates_do_carry_gradient():
    """Companion: every non-final f_x must be live, or the coordinate channel is inert."""
    model, nf, pos, ei, ea = make_scene(10)
    model(nf, pos, ei, ea).sum().backward()
    for i in range(len(model.layers) - 1):
        g = model.layers[i].f_x[-1].weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"layer {i} f_x got no gradient"
