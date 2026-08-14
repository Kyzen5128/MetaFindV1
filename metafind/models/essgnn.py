"""ESSGNN -- Equivariant Spatial-Semantic Graph Neural Network (MetaFind sec. 2.5).

Encodes a room layout as a graph whose nodes carry a 3D position and a
text-derived feature, and whose edges carry an LLM-generated semantic relation
embedding, producing an SE(3)-equivariant layout vector for Eq. 6.

Mapping to the paper, and where the paper contradicts itself
------------------------------------------------------------

The method section (2.5) and the equivariance proof (Appendix C) do not agree in
three places. Each is exposed as a config flag so both readings can be run and
the difference measured, rather than silently picking one:

============  ==========================  ==========================  =============
discrepancy   sec. 2.5 says               Appendix C requires         default here
============  ==========================  ==========================  =============
F1  h^(0)     ``Concat(x_i, t_i)``        h^(0) invariant to SE(3)    ``semantic``
F-B f_x input ``h^(l+1)`` (Eq. 3)         ``m_ij`` from h^(l) (13)    ``updated``
F10 f_x range ``-> R^3``                  scalar (else Q cannot       scalar, always
                                          factor out of Eq. 13)
============  ==========================  ==========================  =============

**F1** is measured by Required Audit RA-1: ``h0_mode="concat_xt"`` reproduces the
literal sec. 2.5 wording and is *expected to fail* the equivariance test, because
concatenating raw coordinates into h makes h rotate with the scene, breaking the
premise Appendix C states outright.

**F-B** does not affect equivariance -- h^(l+1) is still invariant when h^(l) is --
so it is a genuine architectural choice. sec. 2.5 is the normative method
description, so ``"updated"`` is the default; ``"current"`` reproduces Appendix C
and matches stock EGNN.

**F10 has no flag: it is simply an error in the paper.** Eq. 3 types f_x as
``R^(2d+1+e) -> R^3``, but Eq. 13's proof factors the rotation out as
``sum (Q x_i - Q x_j) phi_x = Q sum (x_i - x_j) phi_x``, which only holds when
phi_x is scalar-valued -- elementwise scaling of a 3-vector does not commute with
rotation. The reference EGNN agrees (``coord_mlp`` ends in ``Linear(hidden, 1)``).
Implementing the literal R^3 would break the paper's central claim, so the scalar
form is used unconditionally and the discrepancy is reported instead.

Two further under-specifications are surfaced as UNKNOWN rather than guessed:

* the frozen text encoder for e_ij is only given as "e.g., CLIP or BERT", so its
  width (1280 / 768 / 512) is a config input, not a constant;
* ``Pooling`` in ``e_layout = Pooling({h_i^(L)})`` is unnamed, so ``pooling`` is
  configurable and defaults to mean.

Per Eq. 3 the neighbour aggregation is a **sum**; the reference implementation
defaults to ``mean``, so this is set explicitly (finding F9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from metafind.third_party.egnn_clean import unsorted_segment_mean, unsorted_segment_sum

__all__ = ["ESSGNNConfig", "ESSGCL", "ESSGNN"]

H0Mode = Literal["semantic", "concat_xt"]
CoordFeat = Literal["updated", "current"]
Agg = Literal["sum", "mean"]
Pool = Literal["mean", "sum", "max"]


@dataclass
class ESSGNNConfig:
    """Configuration for ESSGNN.

    Attributes:
        node_feat_dim: width of the text-derived node feature ``t_i``.
        edge_feat_dim: width of the semantic edge embedding ``e_ij``. The paper
            only says "e.g., CLIP or BERT", so this is an input (UNKNOWN U-06).
        hidden_dim: width of the message MLPs.
        out_dim: width of ``e_layout``. Must match the fusion output so Eq. 6's
            residual ``Fusion(...) + lambda * e_layout`` is well formed; ULIP-2
            embeds at 1280.
        n_layers: number of EGCL layers ``L``.
        h0_mode: ``"semantic"`` uses ``h^(0) = t_i`` (Appendix C premise);
            ``"concat_xt"`` uses the literal sec. 2.5 ``Concat(x_i, t_i)`` and is
            expected to fail equivariance. See RA-1.
        coord_feat: which features feed ``f_x`` -- ``"updated"`` follows Eq. 3,
            ``"current"`` follows Eq. 13.
        coords_agg: neighbour aggregation for the coordinate update. Eq. 3 sums.
        edge_proj_dim: optional projection of ``e_ij`` before the message MLP.
            The paper has no such layer, so ``None`` is the faithful setting.
            Provided because with a 1280-wide ``e_ij`` the single geometric
            scalar ``d_ij^2`` is one input among 2*hidden+1+1280 (finding F8);
            if the geometric signal is swamped, that is a property of the paper's
            design and must be reported, not silently corrected.
        pooling: readout over ``{h_i^(L)}``. Unnamed in the paper.
        normalize_coord_diff: divide ``(x_i - x_j)`` by its norm. Off in the
            paper and in the reference implementation.
    """

    node_feat_dim: int = 1280
    edge_feat_dim: int = 1280
    hidden_dim: int = 128
    out_dim: int = 1280
    n_layers: int = 4
    h0_mode: H0Mode = "semantic"
    coord_feat: CoordFeat = "updated"
    coords_agg: Agg = "sum"
    edge_proj_dim: int | None = None
    pooling: Pool = "mean"
    normalize_coord_diff: bool = False


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
    )


class ESSGCL(nn.Module):
    """One Equivariant Spatial-Semantic Graph Convolutional Layer.

    Implements paper Eq. 2 and Eq. 3 with the scalar-valued ``f_x`` that
    Appendix C's proof requires (see module docstring, F10).
    """

    def __init__(self, cfg: ESSGNNConfig, edge_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        # f_h : R^(2d + 1 + e) -> R^d   (Eq. 2)
        self.f_h = _mlp(2 * h + 1 + edge_dim, h, h)
        # f_x : R^(2d + 1 + e) -> R     (Eq. 3, corrected to scalar per Eq. 13)
        self.f_x = _mlp(2 * h + 1 + edge_dim, h, 1)
        # Small init keeps the initial coordinate update near identity; large
        # random displacements at step 0 make the equivariance error numerically
        # meaningless before any training has happened.
        nn.init.xavier_uniform_(self.f_x[-1].weight, gain=0.001)
        nn.init.zeros_(self.f_x[-1].bias)

    def forward(
        self, h: Tensor, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h: ``(N, hidden_dim)`` node features.
            x: ``(N, 3)`` node coordinates.
            edge_index: ``(2, E)`` long tensor of ``(row, col)`` = ``(i, j)``.
            edge_attr: ``(E, edge_dim)`` semantic edge embeddings.

        Returns:
            Updated ``(h, x)``.
        """
        row, col = edge_index[0], edge_index[1]
        coord_diff = x[row] - x[col]
        # ||x_i - x_j||^2 -- invariant under SE(3), which is what makes every
        # message invariant and hence the whole layer equivariant.
        radial = (coord_diff**2).sum(dim=-1, keepdim=True)
        if self.cfg.normalize_coord_diff:
            coord_diff = coord_diff / (radial.sqrt() + 1e-8)

        # ---- Eq. 2: feature update
        m_h = self.f_h(torch.cat([h[row], h[col], radial, edge_attr], dim=-1))
        h_next = h + unsorted_segment_sum(m_h, row, num_segments=h.size(0))

        # ---- Eq. 3: coordinate update
        h_for_x = h_next if self.cfg.coord_feat == "updated" else h
        w = self.f_x(torch.cat([h_for_x[row], h_for_x[col], radial, edge_attr], dim=-1))
        trans = coord_diff * w  # scalar * vector -> stays equivariant
        agg = (
            unsorted_segment_sum(trans, row, num_segments=x.size(0))
            if self.cfg.coords_agg == "sum"
            else unsorted_segment_mean(trans, row, num_segments=x.size(0))
        )
        return h_next, x + agg


class ESSGNN(nn.Module):
    """Full layout encoder: L x ESSGCL, then pooling to ``e_layout``."""

    def __init__(self, cfg: ESSGNNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ESSGNNConfig()

        in_dim = cfg.node_feat_dim + (3 if cfg.h0_mode == "concat_xt" else 0)
        self.embed_in = nn.Linear(in_dim, cfg.hidden_dim)
        self.embed_out = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        if cfg.edge_proj_dim is None:
            self.edge_proj: nn.Module = nn.Identity()
            edge_dim = cfg.edge_feat_dim
        else:
            self.edge_proj = nn.Linear(cfg.edge_feat_dim, cfg.edge_proj_dim)
            edge_dim = cfg.edge_proj_dim

        self.layers = nn.ModuleList(ESSGCL(cfg, edge_dim) for _ in range(cfg.n_layers))

    def forward(
        self,
        node_feat: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Encode one or more scene graphs into layout vectors.

        Args:
            node_feat: ``(N, node_feat_dim)`` text-derived features ``t_i``.
            pos: ``(N, 3)`` object positions, in the scene's own (unnormalised)
                frame -- normalising here would defeat the point of equivariance.
            edge_index: ``(2, E)``.
            edge_attr: ``(E, edge_feat_dim)``.
            batch: optional ``(N,)`` graph id per node for batched graphs. If
                omitted, all nodes are treated as one graph.

        Returns:
            ``(out_dim,)`` when ``batch`` is None, else ``(n_graphs, out_dim)``.
        """
        if node_feat.size(0) != pos.size(0):
            raise ValueError(f"node_feat has {node_feat.size(0)} nodes, pos has {pos.size(0)}")
        if edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be (2, E), got {tuple(edge_index.shape)}")
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                f"edge_attr has {edge_attr.size(0)} rows but edge_index has "
                f"{edge_index.size(1)} edges"
            )

        h0 = torch.cat([pos, node_feat], dim=-1) if self.cfg.h0_mode == "concat_xt" else node_feat
        h = self.embed_in(h0)
        x = pos
        e = self.edge_proj(edge_attr)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, e)

        h = self.embed_out(h)
        return self._pool(h, batch)

    def _pool(self, h: Tensor, batch: Tensor | None) -> Tensor:
        if batch is None:
            if self.cfg.pooling == "mean":
                return h.mean(dim=0)
            if self.cfg.pooling == "sum":
                return h.sum(dim=0)
            return h.max(dim=0).values

        n_graphs = int(batch.max().item()) + 1
        if self.cfg.pooling == "sum":
            return unsorted_segment_sum(h, batch, num_segments=n_graphs)
        if self.cfg.pooling == "mean":
            return unsorted_segment_mean(h, batch, num_segments=n_graphs)
        out = h.new_full((n_graphs, h.size(1)), float("-inf"))
        return out.index_reduce(0, batch, h, reduce="amax", include_self=True)
