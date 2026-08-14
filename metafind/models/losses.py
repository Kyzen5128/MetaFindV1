"""Dual-tower contrastive objectives for MetaFind (paper Eq. 5, 7a, 7b, 8).

ULIP's ``ULIPWithImageLoss`` cannot be reused: it is a *single-tower* tri-modal
objective (pc<->text and pc<->image, four cross-entropy terms) whereas MetaFind
contrasts a fused *query* against a *gallery* embedding. See finding F3.

Stage 1 and Stage 2 use different objectives, and the difference is real:

* **Stage 1, Eq. 5** is one-directional -- query to gallery only::

      L_pre = -log [ exp(sim(f_q(Q), f_g(A))/tau)
                     / sum_{A' in B} exp(sim(f_q(Q), f_g(A'))/tau) ]

* **Stage 2, Eq. 7a/7b/8** is symmetric, averaging both directions. Table 3
  labels the full model "(Full, bidirectional)", confirming this is the headline
  setting rather than an ablation.

Under-specified in the paper
----------------------------

U-09  ``tau``. Eq. 5 calls it "a temperature hyperparameter", which reads as
      fixed, but no value is given anywhere. ULIP-2 -- whose backbone this is
      built on -- uses a *learnable* logit scale initialised to ``log(1/0.07)``,
      the CLIP convention. Both are supported; the ULIP-2 convention is the
      default because the towers inherit its embedding space.

U-10  ``sim(.,.)``, described only as "the similarity function". Cosine
      similarity is assumed, matching ULIP, OpenShape and CLIP. Recorded as an
      assumption rather than treated as given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["ContrastiveConfig", "MetaFindContrastiveLoss"]


@dataclass
class ContrastiveConfig:
    """Configuration for the dual-tower contrastive loss.

    Attributes:
        bidirectional: False reproduces Eq. 5 (Stage 1, query->gallery only);
            True reproduces Eq. 7a/7b/8 (Stage 2, symmetric).
        learnable_temperature: use a learnable logit scale (ULIP-2/CLIP
            convention) rather than a fixed tau. See U-09.
        init_temperature: tau at initialisation. 0.07 is the CLIP value ULIP-2
            adopts.
        max_logit_scale: clamp on the learnable scale. Without it the scale can
            run away early in training and saturate the softmax.
    """

    bidirectional: bool = False
    learnable_temperature: bool = True
    init_temperature: float = 0.07
    max_logit_scale: float = 100.0


class MetaFindContrastiveLoss(nn.Module):
    """InfoNCE between fused query embeddings and gallery embeddings.

    Positives are the diagonal: query ``i`` matches gallery item ``i``. Every
    other item in the batch is a negative, which is what makes batch size matter
    so much for retrieval quality (finding F5, and the reason for decision D2).
    """

    def __init__(self, cfg: ContrastiveConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ContrastiveConfig()
        if not 0.0 < cfg.init_temperature:
            raise ValueError(f"init_temperature must be positive, got {cfg.init_temperature}")

        scale = math.log(1.0 / cfg.init_temperature)
        if cfg.learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(scale))

    @property
    def temperature(self) -> Tensor:
        return 1.0 / self.logit_scale.exp()

    def forward(
        self,
        query: Tensor,
        gallery: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            query: ``(B, D)`` fused query embeddings.
            gallery: ``(B, D)`` gallery embeddings, aligned so row ``i`` is the
                positive for query ``i``.
            labels: optional ``(B,)`` positive indices. Defaults to
                ``arange(B)``; supply explicitly when the batch has been
                gathered across processes.

        Returns:
            ``loss`` plus per-direction losses, accuracies and the temperature.
        """
        if query.shape != gallery.shape:
            raise ValueError(f"query {tuple(query.shape)} != gallery {tuple(gallery.shape)}")
        if query.dim() != 2:
            raise ValueError(f"expected (B, D), got {tuple(query.shape)}")

        q = F.normalize(query, dim=-1)
        g = F.normalize(gallery, dim=-1)

        scale = self.logit_scale.exp().clamp(max=self.cfg.max_logit_scale)
        logits_q2g = scale * q @ g.t()

        if labels is None:
            labels = torch.arange(q.size(0), device=q.device)

        loss_q2g = F.cross_entropy(logits_q2g, labels)
        out = {
            "loss_q2g": loss_q2g,
            "temperature": scale.detach().reciprocal(),
            "acc_q2g": (logits_q2g.argmax(dim=-1) == labels).float().mean().detach(),
        }

        if not self.cfg.bidirectional:
            # Eq. 5 -- Stage 1 is query->gallery only.
            out["loss"] = loss_q2g
            return out

        # Eq. 7b then Eq. 8. Transposing is valid only because the gallery batch
        # is the same set of items; with a decoupled gallery this would need its
        # own logits.
        loss_g2q = F.cross_entropy(logits_q2g.t(), labels)
        out["loss_g2q"] = loss_g2q
        out["acc_g2q"] = (logits_q2g.t().argmax(dim=-1) == labels).float().mean().detach()
        out["loss"] = 0.5 * (loss_q2g + loss_g2q)
        return out
