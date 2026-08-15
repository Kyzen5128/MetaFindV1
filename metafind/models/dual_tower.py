"""MetaFind dual-tower retrieval model (paper sec. 2.2, 2.4, 2.6).

Two towers over a shared ULIP-2 embedding space:

* **Gallery tower** -- modality-complete: every asset has text, image and point
  cloud. Frozen after Stage 1 (sec. 2.6: "the gallery encoder is frozen").
* **Query tower** -- accepts any modality subset, plus optional layout context,
  and implements Eq. 6::

      e_query = Fusion(e_text, e_image, e_pc) + lambda * e_layout

  with ``lambda`` a learnable scalar so layout reasoning is a residual on top of
  the existing embedding space rather than a replacement for it.

What is cached and what is not
------------------------------

[CORRECTED] This said both towers consume pre-computed embeddings and that
"the point encoder cannot be fine-tuned". That describes Table 3's
`Train fuser only` row (8.7 against the full model's 11.4), not the main line.
Paper 2.6 says "Both query and gallery encoders are trained".

    text, image   frozen ViT-bigG-14 (D-1, forced by 24 GB) -> cached once
    point cloud   trainable PointBERT -> encoded every step, never cached

Only the `fuser_only` ablation may consume a cached point-cloud embedding: an
embedding cache is by construction the output of a network that is not being
updated, so caching all three IS the ablation, whatever the configuration says.
RA-3 records the remaining gap, which is ViT-bigG-14 (2.5B parameters) staying
frozen on a single card -- not the point encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from metafind.models.essgnn import ESSGNN, ESSGNNConfig
from metafind.models.fusion import MODALITIES, FusionConfig, ModalityFusion

__all__ = ["DualTowerConfig", "GalleryTower", "QueryTower", "MetaFindDualTower"]


@dataclass
class DualTowerConfig:
    """Configuration for the full dual-tower model.

    Attributes:
        dim: shared embedding width. ULIP-2 embeds at 1280 (finding F2).
        query_fusion: fusion used by the query tower.
        gallery_fusion: fusion used by the gallery tower. Kept separate because
            sec. 2.4 describes the gallery as modality-complete while the query
            side is the flexible one; tying them would prevent freezing one
            without the other.
        use_layout: enable the ESSGNN branch and Eq. 6's residual term.
        essgnn: layout encoder configuration. ``out_dim`` must equal ``dim``.
        init_lambda: initial value of the learnable layout weight. The paper
            gives no value (U-22); 1.0 keeps the residual at full strength at
            initialisation and lets training scale it down.
        scene_dropout: probability of dropping the layout term for a sample
            during training (sec. 2.6, "stochastic scene dropout (30%)").
    """

    # REQUIRED, no defaults. `dim` is the loaded checkpoint's embedding width
    # (1280 for ULIP-2 -- a measured fact about a backbone, not a value the
    # paper states), and ESSGNN's widths additionally depend on the still-open
    # U-06 / U-20 encoder choices. Defaulting either would reintroduce a
    # paper-looking constant one level above the module that just stopped
    # doing it. Build both after the checkpoint and stage1_protocol resolve.
    dim: int
    essgnn: ESSGNNConfig
    # Derived from `dim` when not given, so callers need not restate the width.
    query_fusion: FusionConfig | None = None
    gallery_fusion: FusionConfig | None = None
    use_layout: bool = True
    init_lambda: float = 1.0
    scene_dropout: float = 0.30
    # U-32. "batch" is the literal reading of 2.6 ("omitted in 30% of
    # batches"); "sample" is kept selectable as the variant.
    scene_dropout_granularity: Literal["batch", "sample"] = "batch"

    def __post_init__(self) -> None:
        if self.query_fusion is None:
            self.query_fusion = FusionConfig(dim=self.dim)
        if self.gallery_fusion is None:
            self.gallery_fusion = FusionConfig(dim=self.dim)
        if self.essgnn.out_dim != self.dim:
            raise ValueError(
                f"essgnn.out_dim ({self.essgnn.out_dim}) must equal dim ({self.dim}); "
                "Eq. 6 adds lambda * e_layout to the fused query"
            )


class GalleryTower(nn.Module):
    """Modality-complete encoder for catalogue assets."""

    def __init__(self, cfg: DualTowerConfig) -> None:
        super().__init__()
        if cfg.gallery_fusion.dim != cfg.dim:
            raise ValueError(
                f"gallery_fusion.dim {cfg.gallery_fusion.dim} != model dim {cfg.dim}"
            )
        self.fusion = ModalityFusion(cfg.gallery_fusion)

    def forward(self, embeds: dict[str, Tensor | None]) -> Tensor:
        """Encode assets that have every modality available.

        Args:
            embeds: ``{"text"|"image"|"pc": (B, D)}``. All three are required --
                an incomplete gallery entry is a data error, not a supported
                case, and silently mask-filling it would corrupt the index every
                downstream number is computed against.

        Returns:
            ``(B, D)`` gallery embeddings.
        """
        missing = [m for m in MODALITIES if embeds.get(m) is None]
        if missing:
            raise ValueError(
                f"gallery tower is modality-complete but {missing} are absent; "
                "quarantine the asset instead of encoding it"
            )
        return self.fusion(embeds)


class QueryTower(nn.Module):
    """Flexible encoder: any modality subset, plus optional layout context."""

    def __init__(self, cfg: DualTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.query_fusion.dim != cfg.dim:
            raise ValueError(f"query_fusion.dim {cfg.query_fusion.dim} != model dim {cfg.dim}")
        self.fusion = ModalityFusion(cfg.query_fusion)

        if cfg.use_layout:
            if cfg.essgnn.out_dim != cfg.dim:
                raise ValueError(
                    f"essgnn.out_dim {cfg.essgnn.out_dim} != model dim {cfg.dim}; "
                    "Eq. 6 adds e_layout to the fused query, so the widths must match"
                )
            self.layout_encoder: nn.Module | None = ESSGNN(cfg.essgnn)
            self.log_lambda = nn.Parameter(torch.tensor(float(cfg.init_lambda)))
        else:
            self.layout_encoder = None
            self.register_parameter("log_lambda", None)

    @property
    def lam(self) -> Tensor:
        if self.log_lambda is None:
            raise AttributeError("this query tower has no layout branch")
        return self.log_lambda

    def forward(
        self,
        embeds: dict[str, Tensor | None],
        present: Tensor | None = None,
        layout: Tensor | None = None,
        drop_layout: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            embeds: ``{"text"|"image"|"pc": (B, D) or None}``.
            present: ``(B, 3)`` bool presence mask; see ``ModalityFusion``.
            layout: ``(B, D)`` layout vectors from ``encode_layout``. ``None``
                means no scene context, which is the layout-free query case the
                paper's Table 1 evaluates.
            drop_layout: ``(B,)`` bool, True where the layout term is suppressed
                for this sample (sec. 2.6's stochastic scene dropout). Pass
                explicitly so the decision stays reproducible from a seed rather
                than being drawn inside the forward pass.

        Returns:
            ``(B, D)`` query embeddings.
        """
        fused = self.fusion(embeds, present)
        if layout is None:
            return fused
        if self.layout_encoder is None:
            raise ValueError("layout provided but this tower was built with use_layout=False")
        if layout.shape != fused.shape:
            raise ValueError(f"layout {tuple(layout.shape)} != fused {tuple(fused.shape)}")

        contribution = self.lam * layout
        if drop_layout is not None:
            if drop_layout.shape != (fused.size(0),):
                raise ValueError(
                    f"drop_layout must be ({fused.size(0)},), got {tuple(drop_layout.shape)}"
                )
            contribution = torch.where(drop_layout.unsqueeze(-1), torch.zeros_like(contribution), contribution)
        return fused + contribution

    def encode_layout(
        self,
        node_feat: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Run ESSGNN over a scene graph to obtain ``e_layout``."""
        if self.layout_encoder is None:
            raise ValueError("this tower was built with use_layout=False")
        out = self.layout_encoder(node_feat, pos, edge_index, edge_attr, batch=batch)
        return out.unsqueeze(0) if out.dim() == 1 else out


class MetaFindDualTower(nn.Module):
    """Both towers plus the freezing schedule the two training stages need."""

    def __init__(self, cfg: DualTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query = QueryTower(cfg)
        self.gallery = GalleryTower(cfg)

    def freeze_gallery(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the gallery tower.

        Stage 2 trains "only the query-side fuser and the ESSGNN module; the
        gallery encoder is frozen" (sec. 2.6). Freezing also puts the tower in
        eval mode so any dropout inside it stops perturbing the embeddings that
        the frozen gallery index was built from.
        """
        for p in self.gallery.parameters():
            p.requires_grad_(not frozen)
        self.gallery.train(not frozen)

    def gallery_is_frozen(self) -> bool:
        params = list(self.gallery.parameters())
        return bool(params) and not any(p.requires_grad for p in params)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def sample_scene_dropout(
        self, batch_size: int, generator: torch.Generator | None = None, device="cpu"
    ) -> Tensor:
        """Draw sec. 2.6's 30% scene dropout mask. True means "drop the layout".

        Granularity follows ``cfg.scene_dropout_granularity`` (U-32):

        * ``"batch"`` -- ONE draw for the whole batch, which is what 2.6 says:
          "the layout vector e_layout is omitted in 30% of batches". Every row
          then shares the layout condition, so the in-batch negatives of a
          contrastive step are all in the same regime.
        * ``"sample"`` -- an independent draw per row.

        The two are not interchangeable for an in-batch contrastive loss.
        Note 2.6's OTHER 30%, the Stage 1 modality masking, is explicitly
        "independently" per modality; that one is genuinely per-sample. An
        earlier version applied the per-sample reading to both.
        """
        p = self.cfg.scene_dropout
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"scene_dropout must be in [0, 1], got {p}")
        if self.cfg.scene_dropout_granularity == "batch":
            drop = torch.rand((), device=device, generator=generator) < p
            return drop.expand(batch_size).clone()
        return torch.rand(batch_size, device=device, generator=generator) < p

    def forward(
        self,
        query_embeds: dict[str, Tensor | None],
        gallery_embeds: dict[str, Tensor | None],
        present: Tensor | None = None,
        layout: Tensor | None = None,
        drop_layout: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode a batch through both towers.

        Returns:
            ``(query, gallery)``, each ``(B, D)``, aligned so row ``i`` of the
            query is the positive for row ``i`` of the gallery.
        """
        q = self.query(query_embeds, present=present, layout=layout, drop_layout=drop_layout)
        g = self.gallery(gallery_embeds)
        if q.shape != g.shape:
            raise ValueError(f"tower outputs disagree: query {tuple(q.shape)} vs gallery {tuple(g.shape)}")
        return q, g
