"""ESSGNN -- Equivariant Spatial-Semantic Graph Neural Network (MetaFind sec. 2.5).

Encodes a room layout as a graph whose nodes carry a 3D position and a
text-derived feature, and whose edges carry an LLM-generated semantic relation
embedding, producing an SE(3)-INVARIANT layout embedding through equivariant message
passing. The distinction matters and this docstring used to blur it: the
coordinate channel is equivariant, h is invariant, and e_layout = Pooling({h})
is therefore invariant too -- which is why SC-4/5/6 are three separate
assertions rather than one.

Mapping to the paper, and where the paper contradicts itself
------------------------------------------------------------

The method section (2.5) and the equivariance proof (Appendix C) do not agree in
four places. Each is configurable WHERE A FAITHFUL ALTERNATIVE REMAINS
MATHEMATICALLY VALID, so both readings can be run and the difference measured
rather than one being picked silently. F10 is the exception and is audit-only,
because a vector-valued phi_x breaks the equivariance the paper claims:

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

**The fourth is U-17: d_ij or d_ij^2, and it now has a flag.** Sec. 2.5 defines
``d_ij^l = ||x_i^l - x_j^l||_2``; Appendix C Eq. 10-12 uses ``||x_i - x_j||^2``,
as does the original EGNN. This one gets no AUDIT, because both are SE(3)-invariant -- neither breaks the
proof, so there is no "expected to fail" assertion to write. It does get a
config flag (`distance`), because it changes the number fed to f_h and f_x and
therefore the trained model. `squared` is the primary, following Appendix C and
the reference EGNN; the resolved essgnn_arch_protocol records what a run used.

Further under-specifications are surfaced as UNKNOWN rather than guessed:

* the frozen text encoder for e_ij is only given as "e.g., CLIP or BERT", so its
  width (1280 / 768 / 512) is a config input, not a constant;
* ``Pooling`` in ``e_layout = Pooling({h_i^(L)})`` is unnamed, so ``pooling`` is
  configurable and defaults to mean;
* whether ESSGNN keeps EGNN's ``embedding_in`` / ``embedding_out`` projections
  is not stated (U-33): 2.5 writes t_i straight into h^0 and reads e_layout
  straight out of Pooling({h^(L)}), while the reference EGNN projects on both
  sides. ``use_io_projections`` selects between them;
* ``L`` is written only as "After $L$ layers" -- the paper gives no value, so
  ``n_layers`` is a hyperparameter of ours (U-22), not a reproduction target.

Per Eq. 3 the neighbour aggregation is a **sum**; the reference implementation
defaults to ``mean``, so this is set explicitly (finding F9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from metafind.vendor.egnn_clean import unsorted_segment_mean, unsorted_segment_sum

__all__ = ["ESSGNNConfig", "ESSGCL", "ESSGNN"]

H0Mode = Literal["semantic", "concat_xt"]
CoordFeat = Literal["updated", "current"]
Agg = Literal["sum", "mean"]
Distance = Literal["squared", "euclidean"]
LayerSharing = Literal["independent", "shared"]
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
        n_layers: number of EGCL layers ``L``. The paper says only
            "After $L$ layers" and never gives a value (U-22).
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

    # NO PAPER DEFAULTS. The paper contains no dimension anywhere: it writes
    # t_i in R^d and f_h : R^(2d+1+e) -> R^d, and "After L layers" without an L.
    # Widths that a checkpoint or a protocol determines are REQUIRED arguments,
    # so a caller cannot inherit a number that looks like a paper value:
    #   node_feat_dim  the t_i encoder's width (U-20, still open)
    #   edge_feat_dim  the semantic edge encoder's width (U-06, still open)
    #   out_dim        must equal the fusion output so Eq. 6's residual is well
    #                  formed; read from the loaded ULIP-2 checkpoint
    # An earlier version defaulted all three to 1280, which is the ULIP-2
    # checkpoint's width -- a measured fact about a backbone, not a value the
    # paper states. L1-EGNN-DIMS-NOT-HARDCODED forbids exactly that.
    node_feat_dim: int
    edge_feat_dim: int
    out_dim: int
    # U-33, REQUIRED. 2.5 goes t_i -> h^(0) -> L layers -> Pooling with no
    # projection on either side; the reference EGNN wraps its layers in
    # embedding_in / embedding_out and this implementation inherited that.
    # That is an ARCHITECTURE difference, so it gets no default: a `= True`
    # here would let upstream's convention win the UNKNOWN by inheritance,
    # which is the exact failure this project keeps rediscovering. The decision
    # lives in essgnn_arch_protocol and is gated by G6.
    #   True   reference-EGNN compatible; allows hidden_dim != out_dim
    #   False  literal 2.5; forces node_feat_dim == hidden_dim == out_dim,
    #          so the hidden width becomes the embedding width
    use_io_projections: bool
    # Reproduction hyperparameters chosen by us (U-22). The paper gives no
    # value for either; these are recorded in stage1_protocol and reported.
    hidden_dim: int = 128
    n_layers: int = 4
    h0_mode: H0Mode = "semantic"
    coord_feat: CoordFeat = "updated"
    # U-17. Sec. 2.5 defines d_ij = ||x_i - x_j||_2; Appendix C Eq. 10-12 uses
    # the square, as does the reference EGNN. Both are SE(3)-invariant, so
    # neither breaks the proof, but the number reaching f_h and f_x differs.
    distance: Distance = "squared"
    # U-31. The paper writes theta_h / theta_x with no layer index, so whether
    # the L layers share one parameter set is unstated. This also decides
    # whether F11 holds: with independent layers the final coordinate head has
    # no loss path, with sharing it still receives gradient from layers 1..L-1.
    layer_sharing: LayerSharing = "independent"
    coords_agg: Agg = "sum"
    edge_proj_dim: int | None = None
    pooling: Pool = "mean"
    normalize_coord_diff: bool = False

    @classmethod
    def from_protocol(
        cls,
        protocol: dict,
        *,
        node_feat_dim: int,
        edge_feat_dim: int,
        out_dim: int,
    ) -> ESSGNNConfig:
        """Build the config from a resolved essgnn_arch_protocol.

        The single supported entry point for a trainer. Hand-writing an
        ESSGNNConfig is what lets a run drift from the protocol G6 approved --
        L1-ESSGNN-ARCH-PROTOCOL-APPLIED can detect that afterwards, but not
        building the opportunity is better than catching it.

        Widths stay arguments because they come from the loaded checkpoint and
        from U-06 / U-20, not from this protocol.
        """
        if protocol.get("status") != "resolved":
            raise ValueError(
                "essgnn_arch_protocol is not resolved; G6 exists to stop Stage 2 "
                "starting here (U-33, U-17, U-26, U-31, U-22)"
            )
        required = {"architecture_family", "use_io_projections", "distance",
                    "coord_feat", "layer_sharing", "pooling", "hidden_dim",
                    "n_layers"}
        if missing := required - protocol.keys():
            raise ValueError(f"essgnn_arch_protocol is missing {sorted(missing)}")
        # [C1 / U-26] This class only builds 2.5's two-MLP layer. That is a
        # legitimate implementation as long as it is DECLARED -- what is not
        # legitimate is building it while the audit records U-26 as unresolved,
        # which is what happened while the protocol had no key for it.
        family = protocol["architecture_family"]
        if family != "sec25_two_mlp":
            raise ValueError(
                f"architecture_family is {family!r}. Only 'sec25_two_mlp' is "
                "implemented; 'appendix_shared_msg' (phi_e -> m_ij -> phi_x, "
                "phi_h) is a DIFFERENT model and has no code. A null here means "
                "C1/U-26 has not been decided -- see "
                "docs/audit/C_PAPER_CONTRADICTIONS.md#c1.")
        return cls(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            out_dim=out_dim,
            use_io_projections=protocol["use_io_projections"],
            distance=protocol["distance"],
            coord_feat=protocol["coord_feat"],
            layer_sharing=protocol["layer_sharing"],
            pooling=protocol["pooling"],
            hidden_dim=protocol["hidden_dim"],
            n_layers=protocol["n_layers"],
            **PRIMARY_INTERPRETATION,
        )



# Values the main line pins to one reading. NOT 'unambiguously stated by the
# paper' -- h0_mode=semantic in particular CONTRADICTS 2.5's literal
# Concat(x_i, t_i), and is chosen because Appendix C's premise, Eq. 2's arity
# and the Introduction's "separating spatial and semantic channels" all point
# the other way (RA-1). Unlike the fields in
# essgnn_arch_protocol these are NOT open questions a person has to answer --
# they are our primary interpretation, and a run that departs from them is a
# variant that must say so. L1-ESSGNN-PAPER-LOCKED-CONFIG asserts them.
PRIMARY_INTERPRETATION = {
    "h0_mode": "semantic",          # Appendix C's premise; Concat(x,t) is RA-1
    "coords_agg": "sum",            # Eq. 3 sums; the reference EGNN defaults to mean
    "edge_proj_dim": None,          # the paper has no such layer
    "normalize_coord_diff": False,  # nor any normalisation of (x_i - x_j)
}


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """One 2-layer MLP shape, used for BOTH f_h and f_x. [U-35]

    Paper 2.5 says only "approximated using MLPs" -- no depth, no activation,
    nothing about whether the output carries one. This shape was never a
    decision; it is what this helper happened to be.

    EGNN's Appendix C specifies THREE different shapes:

        phi_e   Linear -> Swish -> Linear -> Swish   (activation on the output)
        phi_x   Linear -> Swish -> Linear            (none)
        phi_h   Linear -> Swish -> Linear, + h_i     (residual)

    SiLU and Swish are the same function, so `f_x` below matches EGNN's phi_x
    exactly and `f_h` matches none of them -- it is missing both the trailing
    activation and the residual. That difference is now U-35 and is recorded in
    essgnn_arch_protocol.mlp_structure, which G6 requires. EGNN's appendix
    supplies the VARIANT LIST, not the answer: a dependency paper cannot fill
    in what MetaFind leaves unsaid.

    Note MetaFind writes the residual OUTSIDE f_h (Eq. 2 and Appendix C Eq. 14
    are both `h_i + sum(...)`), so f_h having no internal residual is correct
    for this paper even though EGNN's phi_h has one.
    """
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
        #
        # U-17: the paper contradicts itself on whether this is
        # the distance or its square. Sec. 2.5 defines d_ij = ||x_i - x_j||_2;
        # Appendix C Eq. 10-12 uses ||x_i - x_j||^2, as does the original EGNN.
        # Both are SE(3)-invariant, so neither breaks the proof -- unlike RA-1
        # and RA-2, there is no "expected to fail" assertion to write. But the
        # number fed to f_h and f_x differs, so the trained model differs. We
        # follow Appendix C and EGNN. Recorded as a choice, not a deduction.
        sq = (coord_diff**2).sum(dim=-1, keepdim=True)
        # U-17. Both forms are SE(3)-invariant; only the magnitude fed to the
        # MLPs differs. The protocol chooses, so a declared value is one the
        # model actually runs.
        radial = sq if self.cfg.distance == "squared" else torch.sqrt(sq + 1e-12)
        if self.cfg.normalize_coord_diff:
            coord_diff = coord_diff / (sq.sqrt() + 1e-8)

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

    def __init__(self, cfg: ESSGNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.node_feat_dim + (3 if cfg.h0_mode == "concat_xt" else 0)
        if cfg.use_io_projections:
            self.embed_in: nn.Module = nn.Linear(in_dim, cfg.hidden_dim)
            self.embed_out: nn.Module = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        else:
            # Literal 2.5: h^0 IS t_i and e_layout IS Pooling({h^(L)}), so the
            # widths have to line up on their own.
            if not (in_dim == cfg.hidden_dim == cfg.out_dim):
                raise ValueError(
                    "use_io_projections=False reproduces sec. 2.5 literally, so "
                    f"h0 width ({in_dim}), hidden_dim ({cfg.hidden_dim}) and "
                    f"out_dim ({cfg.out_dim}) must all be equal"
                )
            self.embed_in = nn.Identity()
            self.embed_out = nn.Identity()

        if cfg.edge_proj_dim is None:
            self.edge_proj: nn.Module = nn.Identity()
            edge_dim = cfg.edge_feat_dim
        else:
            self.edge_proj = nn.Linear(cfg.edge_feat_dim, cfg.edge_proj_dim)
            edge_dim = cfg.edge_proj_dim

        # [U-30] The missing-semantic-edge token, and it is genuinely LEARNED.
        # essgnn_edge_protocol records the decision as `learned_missing_token`;
        # it used to be a seeded numpy vector built in the dataset, which is not
        # a parameter, gets no gradient, enters no optimizer and lands in no
        # checkpoint -- so the protocol described something the code did not do.
        # Zeros remain forbidden either way: zero is a valid point in the edge
        # space and would read as a real relation rather than as absence
        # (L1-SEMEDGE-NO-ZEROFILL).
        self.missing_edge_token = nn.Parameter(torch.zeros(cfg.edge_feat_dim))
        nn.init.normal_(self.missing_edge_token, std=0.02)

        if cfg.layer_sharing == "shared":
            # One layer, applied L times. Not a saving -- a different model.
            shared = ESSGCL(cfg, edge_dim)
            self.layers = nn.ModuleList([shared] * cfg.n_layers)
        else:
            self.layers = nn.ModuleList(
                ESSGCL(cfg, edge_dim) for _ in range(cfg.n_layers)
            )

    def forward(
        self,
        node_feat: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
        edge_missing: Tensor | None = None,
    ) -> Tensor:
        """Encode one or more scene graphs into layout vectors.

        Args:
            node_feat: ``(N, node_feat_dim)`` text-derived features ``t_i``.
            pos: ``(N, 3)`` object positions, in the scene's own (unnormalised)
                frame. Not because centring would break equivariance -- it would
                not; ``x'_i = x_i - mean(x)`` leaves an EGNN equivariant. Because
                sec. 2.5 states the setting as "large and often unnormalized
                coordinate systems, with no guarantee that scenes are aligned or
                centered", and pre-centring removes the very global translation
                the paper claims to handle.
            edge_index: ``(2, E)``.
            edge_attr: ``(E, edge_feat_dim)``.
            batch: optional ``(N,)`` graph id per node for batched graphs. If
                omitted, all nodes are treated as one graph.
            edge_missing: optional ``(E,)`` bool. True where the scene graph has
                an edge but no LLM relation was cached for it. Those rows of
                ``edge_attr`` are replaced by the learned missing-edge token.
                Passing the token in through ``edge_attr`` instead would work
                numerically and would not train it.

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

        if edge_missing is not None:
            if edge_missing.shape != (edge_index.size(1),):
                raise ValueError(
                    f"edge_missing must be (E,) = ({edge_index.size(1)},), got "
                    f"{tuple(edge_missing.shape)}")
            edge_attr = torch.where(
                edge_missing.unsqueeze(-1),
                self.missing_edge_token.to(edge_attr.dtype).expand_as(edge_attr),
                edge_attr)

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
