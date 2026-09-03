"""Modality-aware fusion for the MetaFind query tower (sec. 2.2, 2.4, 2.6).

The query encoder must accept any subset of {text, image, point cloud} and still
produce a single embedding in the gallery's space. Section 2.4 lists five
strategies -- "mean pooling, MLP, masked MLP, gated fusion, or Transformer-based
fusion" -- all of which are implemented here because Table 3 ablates two of them.

Handling absent modalities
--------------------------

Section 2.6 is explicit that absence is *not* zero-padding::

    each modality in the query has a 30% probability of being independently
    masked. Rather than zero-padding, we apply masked embeddings

so each modality owns a learned mask token that stands in when it is absent.
``zero_pad=True`` reproduces the Table 3 ablation "Padding missing modalities
with 0", whose whole point is that it is worse.

What the paper does not pin down
--------------------------------

U-13  **RESOLVED 2026-08-23 -- this entry said "the paper never says which" and
      the paper does say.** `3experiments.tex:143`, last paragraph of the
      Ablation Studies section:

          "while simple mean pooling offers computational efficiency, MLP and
          **the final selected Transformer** outperform others under partial
          modality conditions by dynamically reweighting available inputs"

      So the default is ``transformer``, a PAPER FACT, and the Table 3 numbers
      are consistent with it: Mean 9.4 and MLPs 9.9 are both below the full
      model's 11.4, and the full model is the Transformer.

      The other four kinds stay implemented -- Mean and MLPs are Table 3 rows
      that have to be reproducible, and masked_mlp / gated remain selectable for
      ablation -- but none of them is the default any more.

U-23  What happens when all three modalities are masked. "Independently" at 30%
      each means this occurs for 2.7% of queries, leaving a query built purely
      from mask tokens -- no information, yet still contrastively matched
      against its gallery item. The literal reading is implemented
      (``allow_empty=True``, the default) with a flag to force at least one
      modality, so the effect is measurable instead of assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FusionConfig", "ModalityFusion", "MODALITIES", "sample_modality_mask"]

MODALITIES: tuple[str, ...] = ("text", "image", "pc")

FusionKind = Literal["mean", "mlp", "masked_mlp", "gated", "transformer"]


@dataclass
class FusionConfig:
    """Configuration for the query-side fusion module.

    Attributes:
        kind: fusion strategy. `transformer` is the paper's own selection
            (`3experiments.tex:143`); see U-13 above.
        dim: modality embedding width. ULIP-2 embeds at 1280.
        hidden: hidden width for the MLP/gated variants.
        n_heads: attention heads for the transformer variant.
        n_layers: transformer encoder depth.
        zero_pad: substitute zeros for absent modalities instead of learned mask
            tokens. This is the Table 3 "Padding missing modalities with 0"
            ablation, not a sensible default.
        include_absent_slots: whether the absent modality's slot still takes
            part in aggregation, carrying its mask token. See U-11 -- this is
            what makes sec. 2.6's contrast meaningful, because a slot that is
            dropped from the aggregation entirely has nothing to zero-pad.
        dropout: dropout inside the MLP/transformer variants.
    """

    # REQUIRED and first: the width comes from the loaded checkpoint, and 1280
    # is a measured fact about ULIP-2 rather than a value the paper states --
    # the paper contains no dimension anywhere.
    dim: int
    # [PAPER FACT 3experiments.tex:143] "the final selected Transformer".
    # Was `masked_mlp` on a false UNKNOWN; see U-13 in the module docstring.
    kind: FusionKind = "transformer"
    hidden: int = 2048
    n_heads: int = 8
    n_layers: int = 2
    zero_pad: bool = False
    include_absent_slots: bool = True
    dropout: float = 0.0
    # [AUDIT 2026-09-03 C8] The paper says nothing about normalising modality
    # vectors before Fusion, and neither did this module: raw vectors went in.
    # Measured on the 10-epoch pilot, the vectors entering Fusion had norms
    # text 37.1 / image 40.2 / pc 139.1 -- PointBERT trains and grew 3.5x past
    # the frozen CLIP towers -- and permuting the gallery's pc alone took every
    # Table 1 condition to single digits. With this on, each PRESENT modality
    # is L2-normalised before it enters the Transformer; the learned mask
    # tokens are left free. IMPLEMENTATION CHOICE, recorded in
    # stage1_protocol.json, off by default.
    prefusion_norm: bool = False
    # [AUDIT 2026-09-03 C4] Figure 1 draws K vectors per encoder (P1..PK,
    # I1..IK, T1..TK) entering ONE Fusion Layer; nothing in the paper averages
    # the views first. With image_tokens = K > 1 the image modality enters as
    # K view tokens -- `embeds["image"]` is (B, K, D) and the optional
    # `embeds["image_present"]` (B, K) says which views the query actually
    # holds; an absent view carries the image mask token. The text and point
    # cloud stay one token each. 1 keeps the pooled vector and the pre-existing
    # (B, 3, D) construction byte for byte. Transformer only.
    image_tokens: int = 1


def sample_modality_mask(
    batch_size: int,
    p_mask: float = 0.30,
    allow_empty: bool = True,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the stochastic modality mask of sec. 2.6.

    Args:
        batch_size: number of queries.
        p_mask: per-modality masking probability. The paper uses 0.30; Table 3
            ablates 0.10 and 0.50.
        allow_empty: if True (the literal reading), a query may lose all three
            modalities, which happens with probability ``p_mask ** 3`` -- 2.7% at
            the paper's rate. See U-23.
        device: device for the returned tensor.
        generator: optional RNG for reproducibility.

    Returns:
        ``(batch_size, 3)`` bool tensor; True means the modality is PRESENT.
    """
    if not 0.0 <= p_mask <= 1.0:
        raise ValueError(f"p_mask must be in [0, 1], got {p_mask}")
    # [P0-5] Drawn on the GENERATOR's device, then moved. Passing a CPU
    # generator to a CUDA `torch.rand` raises "Expected a cuda device type for
    # generator but found cpu", and n10 builds exactly that pairing -- a CPU
    # generator (which the DataLoader also needs) with device="cuda". Every CPU
    # test passed because the mismatch cannot occur there.
    #
    # Drawing on the generator's own device rather than fixing the pairing has a
    # second benefit worth more than the bug: one seed now yields the SAME mask
    # on CPU and on GPU, so a CPU test exercises the sequence the GPU run will
    # see. Sampling on CUDA instead would have made the two diverge silently.
    gen_device = generator.device if generator is not None else device
    present = (
        torch.rand(batch_size, len(MODALITIES), device=gen_device, generator=generator)
        >= p_mask
    )
    if not allow_empty:
        empty = ~present.any(dim=-1)
        if empty.any():
            keep = torch.randint(
                len(MODALITIES), (int(empty.sum()),), device=gen_device, generator=generator
            )
            present[empty, keep] = True
    return present.to(device)


class ModalityFusion(nn.Module):
    """Fuse an arbitrary subset of modality embeddings into one query vector."""

    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, m = cfg.dim, len(MODALITIES)

        # One learned stand-in per modality (sec. 2.6: "masked embeddings").
        # Unused when zero_pad is on, but always constructed so a checkpoint can
        # be loaded under either setting.
        self.mask_tokens = nn.Parameter(torch.zeros(m, d))
        nn.init.normal_(self.mask_tokens, std=0.02)

        if cfg.kind == "mean":
            self.head: nn.Module = nn.Identity()
        elif cfg.kind in ("mlp", "masked_mlp"):
            in_dim = m * d if cfg.kind == "masked_mlp" else d
            self.head = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, d),
            )
        elif cfg.kind == "gated":
            self.head = nn.Sequential(nn.Linear(d, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))
        elif cfg.kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.head = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.modality_pos = nn.Parameter(torch.zeros(m, d))
            nn.init.normal_(self.modality_pos, std=0.02)
            if cfg.image_tokens > 1:
                # which VIEW a token is, on top of which MODALITY it is
                self.view_pos = nn.Parameter(torch.zeros(cfg.image_tokens, d))
                nn.init.normal_(self.view_pos, std=0.02)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown fusion kind {cfg.kind!r}")
        if cfg.image_tokens < 1:
            raise ValueError(f"image_tokens must be >= 1, got {cfg.image_tokens}")
        if cfg.image_tokens > 1 and cfg.kind != "transformer":
            raise ValueError("image_tokens > 1 needs a token sequence; only the "
                             f"transformer fusion takes one, not {cfg.kind!r}")

    def _stack_tokens(self, embeds: dict[str, Tensor | None], present: Tensor):
        """image_tokens > 1: ``(B, 1+K+1, D)`` plus the per-token active mask.

        Token order is text, the K views, pc. A view is active when the image
        modality is present for that row AND `image_present` marks the view;
        inactive views carry the image mask token, exactly as an absent
        modality does in the 3-slot path.
        """
        K = self.cfg.image_tokens
        b = present.size(0)
        cols, act = [], []
        for i, name in enumerate(MODALITIES):
            e = embeds.get(name)
            keep = present[:, i : i + 1]                       # (B, 1)
            if name == "image":
                vp = embeds.get("image_present")
                if e is None:
                    if bool(keep.any()):
                        raise ValueError("image marked present but no image "
                                         "embedding was passed")
                    e = self.mask_tokens[1].expand(b, K, -1)
                    vkeep = torch.zeros(b, K, dtype=torch.bool, device=present.device)
                else:
                    if e.dim() != 3 or e.size(1) != K:
                        raise ValueError(f"image must be (B, {K}, D) under "
                                         f"image_tokens={K}, got {tuple(e.shape)}")
                    vkeep = keep.expand(-1, K)
                    if vp is not None:
                        if vp.shape != (b, K):
                            raise ValueError(f"image_present must be (B, {K}), "
                                             f"got {tuple(vp.shape)}")
                        vkeep = vkeep & vp.to(torch.bool)
                    if self.cfg.prefusion_norm:
                        e = F.normalize(e, dim=-1)
                    fill = (torch.zeros_like(e) if self.cfg.zero_pad
                            else self.mask_tokens[1].expand_as(e))
                    e = torch.where(vkeep.unsqueeze(-1), e, fill)
                cols.append(e); act.append(vkeep)
                continue
            if e is None:
                if bool(keep.any()):
                    raise ValueError(f"{name} is marked present for some rows "
                                     f"but no {name} embedding was passed")
                fill = (torch.zeros_like(self.mask_tokens[i]) if self.cfg.zero_pad
                        else self.mask_tokens[i])
                cols.append(fill.expand(b, -1).unsqueeze(1)); act.append(keep)
                continue
            if self.cfg.prefusion_norm:
                e = F.normalize(e, dim=-1)
            fill = torch.zeros_like(e) if self.cfg.zero_pad else self.mask_tokens[i].expand_as(e)
            cols.append(torch.where(keep, e, fill).unsqueeze(1)); act.append(keep)
        return torch.cat(cols, dim=1), torch.cat(act, dim=1)

    def _stack(self, embeds: dict[str, Tensor | None], present: Tensor) -> Tensor:
        """Build ``(B, 3, D)`` with absent slots filled by mask tokens or zeros."""
        cols = []
        for i, name in enumerate(MODALITIES):
            e = embeds.get(name)
            keep = present[:, i : i + 1]
            if e is None:
                if bool(keep.any()):
                    raise ValueError(
                        f"{name} is marked present for some rows but no "
                        f"{name} embedding was passed; it would have been "
                        "silently mask-filled and counted as present")
                fill = torch.zeros_like(self.mask_tokens[i]) if self.cfg.zero_pad else self.mask_tokens[i]
                cols.append(fill.expand(present.size(0), -1))
                continue
            if self.cfg.prefusion_norm:
                e = F.normalize(e, dim=-1)
            fill = torch.zeros_like(e) if self.cfg.zero_pad else self.mask_tokens[i].expand_as(e)
            cols.append(torch.where(keep, e, fill))
        return torch.stack(cols, dim=1)

    def forward(self, embeds: dict[str, Tensor | None], present: Tensor | None = None) -> Tensor:
        """
        Args:
            embeds: ``{"text"|"image"|"pc": (B, D) or None}``. A ``None`` entry
                means the modality is absent for the whole batch.
            present: ``(B, 3)`` bool, True where the modality is available. When
                omitted, every modality with a non-None entry is treated as
                present.

        Returns:
            ``(B, D)`` fused query embedding.
        """
        sizes = {embeds[m].size(0) for m in MODALITIES if embeds.get(m) is not None}
        if not sizes:
            raise ValueError("at least one modality embedding must be provided")
        if len(sizes) != 1:
            raise ValueError(f"modality embeddings disagree on batch size: {sizes}")
        b = sizes.pop()

        for name in MODALITIES:          # `image_present` is a mask, not a vector
            e = embeds.get(name)
            if e is not None and e.size(-1) != self.cfg.dim:
                raise ValueError(f"{name} embedding has width {e.size(-1)}, expected {self.cfg.dim}")

        if present is None:
            flags = [embeds.get(n) is not None for n in MODALITIES]
            ref = next(e for e in embeds.values() if e is not None)
            present = torch.tensor(flags, device=ref.device).expand(b, -1).clone()
        if present.shape != (b, len(MODALITIES)):
            raise ValueError(f"present must be {(b, len(MODALITIES))}, got {tuple(present.shape)}")

        if self.cfg.image_tokens > 1:
            return self._forward_tokens(embeds, present)

        x = self._stack(embeds, present)  # (B, 3, D)
        kind = self.cfg.kind

        # Which slots take part in the aggregation (U-11). Under the paper's
        # reading every slot participates, absent ones carrying their mask
        # token -- otherwise sec. 2.6's "rather than zero-padding" contrast has
        # no referent, since a dropped slot has nothing to pad.
        active = (
            torch.ones_like(present) if self.cfg.include_absent_slots else present
        )
        w = active.to(x.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)

        if kind == "mean":
            return (x * w).sum(dim=1) / denom

        if kind == "masked_mlp":
            # Honour include_absent_slots here too: with it off, an absent
            # slot's token must not reach the MLP. It used to be ignored for
            # this kind only, so an "absent slots excluded" masked_mlp run was
            # the included variant under the other label.
            return self.head((x * w).flatten(1))

        if kind == "mlp":
            return self.head((x * w).sum(dim=1) / denom)

        if kind == "gated":
            logits = self.head(x).squeeze(-1)  # (B, 3)
            logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            # Every slot inactive would softmax over all -inf and give NaN; fall
            # back to uniform so the mask tokens are simply averaged.
            dead = ~active.any(dim=-1, keepdim=True)
            logits = torch.where(dead, torch.zeros_like(logits), logits)
            return (x * logits.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)

        # transformer. A row with no active slot would make every key padded,
        # which yields NaN from the attention softmax, so such rows attend
        # normally over their mask tokens.
        pad = ~active
        pad = torch.where(pad.all(dim=-1, keepdim=True), torch.zeros_like(pad), pad)
        h = self.head(x + self.modality_pos, src_key_padding_mask=pad)
        return (h * w).sum(dim=1) / denom

    def _forward_tokens(self, embeds: dict[str, Tensor | None], present: Tensor) -> Tensor:
        """Transformer over text + K view tokens + pc. Same readout rule as the
        3-slot path: mean over the tokens that take part (all of them under
        include_absent_slots, the active ones otherwise)."""
        K = self.cfg.image_tokens
        x, act = self._stack_tokens(embeds, present)          # (B, T, D), (B, T)
        pos = torch.cat([self.modality_pos[0:1],
                         self.modality_pos[1:2].expand(K, -1) + self.view_pos,
                         self.modality_pos[2:3]], dim=0)      # (T, D)
        take = torch.ones_like(act) if self.cfg.include_absent_slots else act
        w = take.to(x.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)
        pad = ~take
        pad = torch.where(pad.all(dim=-1, keepdim=True), torch.zeros_like(pad), pad)
        h = self.head(x + pos, src_key_padding_mask=pad)
        return (h * w).sum(dim=1) / denom
