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

tau   [PAPER FACT] MetaFind DOES give a value, and this module said for months
      that it did not. `3experiments.tex:15`, last sentence of the Baselines
      paragraph:

          "The temperature is 0.5 for all experiments."

      Eq. 5 introduces it as "a temperature hyperparameter" without a number,
      which is where the earlier reading stopped; the value is stated in the
      experimental setup instead. `docs/audit/C_PAPER_CONTRADICTIONS.md` S4 also
      lists tau among the paper's silences and says "the only stated numbers are
      the two 30% rates". Both are wrong.

      What follows for this class:

      * A run reproducing the paper's tables uses `learnable_temperature=False`
        and `init_temperature=0.5`. Anything else is a DEVIATION and has to say
        so, because tau scales every logit in Eq. 5, 7a, 7b and 8.
      * `learnable_temperature=True` with 0.07 is CLIP's and ULIP-2's
        convention, not MetaFind's. ULIP-2's Eq. 1/2 do define a learnable tau,
        which is why it was adopted -- but MetaFind states a fixed 0.5 for its
        own experiments, and a dependency's design does not override the paper
        being reproduced.

      Neither is a default here: both arrive through the hyperparameter artifact
      that ``Stage1RuntimeConfig`` requires and hashes. PAPER_TAU below is the
      value a faithful run must carry, so the number lives in code rather than
      in a comment nobody greps.

U-24  ``sim(.,.)``, described only as "the similarity function". Cosine
      similarity is assumed, matching ULIP, OpenShape and CLIP. Recorded as an
      assumption rather than treated as given, and a protocol naming anything
      else is refused rather than silently computed as cosine.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["PAPER_TAU", "ContrastiveConfig", "MetaFindContrastiveLoss"]

# [PAPER FACT] 3experiments.tex:15 -- "The temperature is 0.5 for all
# experiments." A run that does not use this is not reproducing the paper's
# tables, whatever else it gets right.
PAPER_TAU = 0.5


@dataclass
class ContrastiveConfig:
    """Configuration for the dual-tower contrastive loss.

    Attributes:
        bidirectional: False reproduces Eq. 5 (Stage 1, query->gallery only);
            True reproduces Eq. 7a/7b/8 (Stage 2, symmetric).
        learnable_temperature: use a learnable logit scale rather than a fixed
            tau. ULIP-2's paper states this for its own objective (Eq. 1/2).
            See U-22. **Defaults to False**: MetaFind fixes tau, and a
            dependency's design does not override the paper being reproduced.
        init_temperature: tau at initialisation. **Defaults to PAPER_TAU
            (0.5)**, which `3experiments.tex:15` states for all experiments.
            0.07 is CLIP's and ULIP-2's value and is still selectable, but it is
            a DEVIATION and the constructor says so.
        max_logit_scale: clamp on the learnable scale. Without it the scale can
            run away early in training and saturate the softmax. Inert when
            `learnable_temperature=False`, because a fixed scale cannot move.
    """

    bidirectional: bool = False
    # [CORRECTED 2026-08-23] These two defaulted to ULIP-2's convention
    # (learnable, 0.07) while the paper being reproduced states a fixed 0.5.
    # Every run that did not pass explicit values was therefore a DEVIATION by
    # default, and the warning below fired on the faithful path instead of the
    # unfaithful one. The defaults now reproduce the paper.
    learnable_temperature: bool = False
    init_temperature: float = PAPER_TAU
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

        # [PAPER FACT] 3experiments.tex:15 fixes tau at 0.5. Departing from it is
        # allowed -- the fixed-0.5 objective may simply train worse, and that is
        # itself a reportable result -- but it must be VISIBLE at construction,
        # not discovered when someone compares tables months later. Warn, never
        # raise: an ablation that deliberately sweeps tau is a legitimate run.
        if cfg.learnable_temperature or cfg.init_temperature != PAPER_TAU:
            warnings.warn(
                f"tau deviates from the paper: MetaFind fixes it at {PAPER_TAU} "
                f"for all experiments (3experiments.tex:15), this run uses "
                f"init_temperature={cfg.init_temperature} "
                f"learnable={cfg.learnable_temperature}. Record it as a DEVIATION "
                "if the results are compared with the paper's tables.",
                stacklevel=2,
            )

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
