"""L1 tests for fusion (sec. 2.4/2.6) and the dual-tower losses (Eq. 5/7/8).

Covers L1-MASK-NOTZERO, L1-DROPOUT-30, L1-LOSS-SYMM and the fusion contracts.
Every check asserts on content and carries a negative injection (rules V1-V3).
"""

from __future__ import annotations

import pytest
import torch

from metafind.models.fusion import (
    MODALITIES,
    FusionConfig,
    ModalityFusion,
    sample_modality_mask,
)
from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss

KINDS = ["mean", "mlp", "masked_mlp", "gated", "transformer"]
D = 64
COMMON = dict(dim=D, hidden=128, n_heads=4, n_layers=1)


def embeds(b: int = 4, d: int = D, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(b, d, generator=g) for m in MODALITIES}


def drop(idx: int, b: int = 4) -> torch.Tensor:
    present = torch.ones(b, 3, dtype=torch.bool)
    present[:, idx] = False
    return present


# --------------------------------------------------------------- masking


def test_mask_rate_is_30_percent_and_independent():
    """L1-DROPOUT-30: sec. 2.6 masks each modality independently at 30%."""
    present = sample_modality_mask(200_000, p_mask=0.30, generator=torch.Generator().manual_seed(0))
    rate = 1.0 - present.float().mean(dim=0)
    assert torch.allclose(rate, torch.full((3,), 0.30), atol=0.01), f"per-modality rates {rate}"

    # Independence: the joint all-masked rate must be 0.3^3, not something else.
    empty = (~present).all(dim=-1).float().mean().item()
    assert abs(empty - 0.027) < 0.005, f"all-masked rate {empty:.4f}, expected ~0.027"


def test_mask_rate_negative_injection():
    """If the sampler ignored p_mask, the previous test would be vacuous."""
    present = sample_modality_mask(50_000, p_mask=0.50, generator=torch.Generator().manual_seed(0))
    rate = 1.0 - present.float().mean(dim=0)
    assert not torch.allclose(rate, torch.full((3,), 0.30), atol=0.01)


def test_allow_empty_flag_controls_the_all_masked_case():
    """U-23: the literal reading permits a query with no modality at all."""
    g = torch.Generator().manual_seed(1)
    loose = sample_modality_mask(20_000, 0.5, allow_empty=True, generator=g)
    assert (~loose).all(dim=-1).any(), "allow_empty=True should produce empty queries"

    g = torch.Generator().manual_seed(1)
    strict = sample_modality_mask(20_000, 0.5, allow_empty=False, generator=g)
    assert strict.any(dim=-1).all(), "allow_empty=False must keep >= 1 modality"


# --------------------------------------------------------------- L1-MASK-NOTZERO


@pytest.mark.parametrize("kind", KINDS)
def test_masked_modality_uses_a_learned_token_not_zero(kind: str):
    """L1-MASK-NOTZERO: sec. 2.6 uses masked embeddings, explicitly not zero-padding.

    Asserted the only non-circular way: two modules with identical weights
    differing solely in ``zero_pad``, same input, same presence mask, outputs
    must differ.
    """
    torch.manual_seed(0)
    masked = ModalityFusion(FusionConfig(kind=kind, **COMMON, zero_pad=False)).eval()
    zeroed = ModalityFusion(FusionConfig(kind=kind, **COMMON, zero_pad=True)).eval()
    zeroed.load_state_dict(masked.state_dict())
    with torch.no_grad():
        masked.mask_tokens.normal_(std=1.0)
        zeroed.mask_tokens.copy_(masked.mask_tokens)

    e, present = embeds(), drop(1)  # drop image
    with torch.no_grad():
        a, b = masked(e, present), zeroed(e, present)
    assert not torch.allclose(a, b, atol=1e-6), (
        "masking behaved identically to zero-padding, so the mask token is unused"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_mask_token_is_what_carries_the_difference(kind: str):
    """Companion: under zero_pad the mask token must be inert, otherwise live."""
    torch.manual_seed(0)
    e, present = embeds(), drop(1)

    zeroed = ModalityFusion(FusionConfig(kind=kind, **COMMON, zero_pad=True)).eval()
    with torch.no_grad():
        before = zeroed(e, present)
        zeroed.mask_tokens.normal_(std=5.0)
        after = zeroed(e, present)
    assert torch.allclose(before, after, atol=1e-6), "zero_pad still consulted the mask token"

    masked = ModalityFusion(FusionConfig(kind=kind, **COMMON, zero_pad=False)).eval()
    with torch.no_grad():
        before = masked(e, present)
        masked.mask_tokens.normal_(std=5.0)
        after = masked(e, present)
    assert not torch.allclose(before, after, atol=1e-6), "mask token had no effect"


def test_absent_slots_participate_by_default():
    """U-11: sec. 2.6's contrast only has a referent if the slot is aggregated."""
    assert FusionConfig(dim=D).include_absent_slots is True

    torch.manual_seed(0)
    incl = ModalityFusion(FusionConfig(kind="mean", dim=D, include_absent_slots=True)).eval()
    excl = ModalityFusion(FusionConfig(kind="mean", dim=D, include_absent_slots=False)).eval()
    excl.load_state_dict(incl.state_dict())
    with torch.no_grad():
        incl.mask_tokens.normal_(std=1.0)
        excl.mask_tokens.copy_(incl.mask_tokens)

    e = embeds(b=1)
    present = torch.tensor([[True, False, False]])
    with torch.no_grad():
        # Excluding absent slots reduces to "just the text embedding".
        assert torch.allclose(excl(e, present), e["text"], atol=1e-6)
        # Including them mixes in mask tokens, so it must differ.
        assert not torch.allclose(incl(e, present), e["text"], atol=1e-6)


# --------------------------------------------------------------- fusion contracts


@pytest.mark.parametrize("kind", KINDS)
def test_output_shape_and_finiteness(kind: str):
    fuse = ModalityFusion(FusionConfig(kind=kind, **COMMON))
    out = fuse(embeds(), torch.ones(4, 3, dtype=torch.bool))
    assert out.shape == (4, D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("include_absent", [True, False])
def test_all_modalities_masked_stays_finite(kind: str, include_absent: bool):
    """U-23: 2.7% of queries lose everything; a NaN here would poison the batch."""
    fuse = ModalityFusion(FusionConfig(kind=kind, **COMMON, include_absent_slots=include_absent))
    out = fuse(embeds(), torch.zeros(4, 3, dtype=torch.bool))
    assert torch.isfinite(out).all(), f"{kind} went non-finite with nothing present"


@pytest.mark.parametrize("kind", KINDS)
def test_absent_modality_does_not_leak_when_slots_are_excluded(kind: str):
    """Marking a modality absent must beat whatever garbage is in its tensor."""
    torch.manual_seed(0)
    fuse = ModalityFusion(
        FusionConfig(kind=kind, **COMMON, include_absent_slots=False)
    ).eval()
    e, present = embeds(), drop(2)  # drop pc
    with torch.no_grad():
        with_junk = fuse({**e, "pc": torch.randn_like(e["pc"]) * 100}, present)
        without = fuse({**e, "pc": None}, present)
    assert torch.allclose(with_junk, without, atol=1e-5), (
        "a modality marked absent still leaked into the output"
    )


def test_mean_fusion_over_present_only():
    """Under the exclude-absent reading, mean of one present modality is that modality."""
    fuse = ModalityFusion(FusionConfig(kind="mean", dim=D, include_absent_slots=False))
    e = embeds(b=1)
    present = torch.tensor([[True, False, False]])
    assert torch.allclose(fuse(e, present), e["text"], atol=1e-6)


def test_shape_contract_violations_raise():
    fuse = ModalityFusion(FusionConfig(kind="mean", dim=D))
    with pytest.raises(ValueError, match="at least one"):
        fuse({m: None for m in MODALITIES})
    with pytest.raises(ValueError, match="width"):
        fuse({"text": torch.randn(4, D + 1), "image": None, "pc": None})
    with pytest.raises(ValueError, match="batch size"):
        fuse({"text": torch.randn(4, D), "image": torch.randn(5, D), "pc": None})


# --------------------------------------------------------------- losses


def test_eq5_is_unidirectional_and_eq8_is_symmetric():
    """L1-LOSS-SYMM: Stage 1 (Eq. 5) is q->g only; Stage 2 (Eq. 8) averages both."""
    torch.manual_seed(0)
    q, g = torch.randn(8, D), torch.randn(8, D)

    stage1 = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=False))
    out1 = stage1(q, g)
    assert "loss_g2q" not in out1, "Eq. 5 must not include the gallery->query term"
    assert torch.allclose(out1["loss"], out1["loss_q2g"])

    stage2 = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))
    out2 = stage2(q, g)
    assert torch.allclose(out2["loss"], 0.5 * (out2["loss_q2g"] + out2["loss_g2q"]))


def test_bidirectional_loss_is_symmetric_under_swap():
    """Eq. 8 must be invariant to swapping the towers; Eq. 5 must not be."""
    torch.manual_seed(0)
    q, g = torch.randn(8, D), torch.randn(8, D)

    bi = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))
    assert torch.allclose(bi(q, g)["loss"], bi(g, q)["loss"], atol=1e-6)

    uni = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=False))
    assert not torch.allclose(uni(q, g)["loss"], uni(g, q)["loss"], atol=1e-4), (
        "Eq. 5 came out symmetric, so the directionality test proves nothing"
    )


def test_perfect_alignment_drives_loss_down():
    """A loss that ignored its inputs would pass every structural test above."""
    torch.manual_seed(0)
    g = torch.randn(16, D)
    loss = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))
    aligned = loss(g.clone(), g)["loss"]
    random = loss(torch.randn(16, D), g)["loss"]
    assert aligned < random, f"aligned {aligned:.4f} not better than random {random:.4f}"
    assert aligned < 0.1, f"identical embeddings should be nearly free: {aligned:.4f}"


def test_accuracy_is_perfect_when_query_equals_gallery():
    g = torch.randn(16, D)
    out = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))(g.clone(), g)
    assert out["acc_q2g"].item() == 1.0
    assert out["acc_g2q"].item() == 1.0


def test_cosine_similarity_ignores_magnitude():
    """U-24: sim is assumed cosine, so rescaling a tower must not change the loss."""
    torch.manual_seed(0)
    q, g = torch.randn(8, D), torch.randn(8, D)
    loss = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))
    assert torch.allclose(loss(q, g)["loss"], loss(q * 17.0, g)["loss"], atol=1e-5)


def test_temperature_is_learnable_by_default_and_fixed_when_asked():
    learn = MetaFindContrastiveLoss(ContrastiveConfig(learnable_temperature=True))
    assert isinstance(learn.logit_scale, torch.nn.Parameter)
    learn(torch.randn(4, D), torch.randn(4, D))["loss"].backward()
    assert learn.logit_scale.grad is not None

    fixed = MetaFindContrastiveLoss(ContrastiveConfig(learnable_temperature=False))
    assert not isinstance(fixed.logit_scale, torch.nn.Parameter)
    assert abs(fixed.temperature.item() - 0.07) < 1e-6


def test_loss_shape_contract():
    loss = MetaFindContrastiveLoss()
    with pytest.raises(ValueError, match="!="):
        loss(torch.randn(4, D), torch.randn(5, D))
    with pytest.raises(ValueError, match=r"\(B, D\)"):
        loss(torch.randn(4, 3, D), torch.randn(4, 3, D))


def test_gradients_reach_fusion_and_loss_end_to_end():
    """The two modules must actually compose; a detached seam trains nothing."""
    torch.manual_seed(0)
    fuse = ModalityFusion(FusionConfig(kind="masked_mlp", dim=D, hidden=128))
    loss = MetaFindContrastiveLoss(ContrastiveConfig(bidirectional=True))

    present = sample_modality_mask(8, 0.30, generator=torch.Generator().manual_seed(0))
    q = fuse(embeds(b=8), present)
    loss(q, torch.randn(8, D))["loss"].backward()

    dead = [n for n, p in fuse.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


# --- tau is a PAPER FACT, not a silence ------------------------------------

def test_paper_tau_matches_the_papers_stated_value():
    """[PAPER FACT] 3experiments.tex:15 "The temperature is 0.5 for all
    experiments."

    This constant existed only after the module spent months asserting the
    opposite -- both `losses.py` and `C_PAPER_CONTRADICTIONS.md` S4 listed tau
    among the paper's silences. Pinning the number here means the next reader
    who "corrects" it back to CLIP's 0.07 has to delete a test that cites the
    line, rather than editing a comment.
    """
    from metafind.models.losses import PAPER_TAU
    assert PAPER_TAU == 0.5


def test_the_paper_setting_is_silent_and_any_other_warns():
    """A deviation from a stated paper value has to be visible when it is made.

    Not raised: sweeping tau is a legitimate ablation. But it must not be
    discoverable only by diffing tables months later.
    """
    import warnings
    from metafind.models.losses import (
        PAPER_TAU, ContrastiveConfig, MetaFindContrastiveLoss)

    with warnings.catch_warnings(record=True) as clean:
        warnings.simplefilter("always")
        MetaFindContrastiveLoss(ContrastiveConfig(
            learnable_temperature=False, init_temperature=PAPER_TAU))
    assert not clean, "the paper's own setting must not warn"

    for cfg in (ContrastiveConfig(),                                    # CLIP's default
                ContrastiveConfig(learnable_temperature=False,
                                  init_temperature=0.07)):              # fixed, wrong value
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MetaFindContrastiveLoss(cfg)
        assert w and "0.5" in str(w[0].message), cfg
