"""The parts that only break on a GPU.

Everything else in this suite runs on CPU, which is why the modality-mask RNG
bug survived: `torch.rand(..., device="cuda", generator=<cpu generator>)` raises,
and n10 constructs exactly that pairing -- but the pairing is impossible to
form on a CPU-only run, so 377 passing tests said nothing about it.

These skip when no GPU is present. A skip is reported, not silently counted as
a pass; `report_skips` in the audit output is what makes that visible.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="no CUDA device on this machine")

from metafind.models.fusion import sample_modality_mask  # noqa: E402


@cuda
def test_a_cpu_generator_drives_a_cuda_mask():
    """[P0-5] n10's exact pairing: CPU generator, CUDA target.

    The DataLoader needs a CPU generator, so the trainer has one; passing it
    straight through to a CUDA `torch.rand` is what failed.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    present = sample_modality_mask(8, p_mask=0.3, device="cuda", generator=gen)
    assert present.device.type == "cuda"
    assert present.shape == (8, 3)
    assert present.dtype == torch.bool


@cuda
def test_the_same_seed_gives_the_same_mask_on_cpu_and_cuda():
    """Worth more than the bug it came from: a CPU test now exercises the
    sequence the GPU run will see. Sampling on the CUDA device instead would
    have produced a different stream from the same seed, silently."""
    a = sample_modality_mask(16, p_mask=0.3, device="cpu",
                             generator=torch.Generator(device="cpu").manual_seed(7))
    b = sample_modality_mask(16, p_mask=0.3, device="cuda",
                             generator=torch.Generator(device="cpu").manual_seed(7))
    assert torch.equal(a, b.cpu())


@cuda
def test_no_generator_still_works_on_cuda():
    present = sample_modality_mask(4, device="cuda")
    assert present.device.type == "cuda"


@cuda
def test_allow_empty_false_path_also_runs_on_cuda():
    """The `randint` branch is a second device-typed RNG call and was reached
    only when a query lost all three modalities -- 2.7% of rows at p=0.30, so a
    short smoke run can miss it entirely. Forced here."""
    gen = torch.Generator(device="cpu").manual_seed(1)
    present = sample_modality_mask(32, p_mask=1.0, allow_empty=False,
                                   device="cuda", generator=gen)
    assert present.device.type == "cuda"
    assert present.any(dim=-1).all(), "allow_empty=False left a query with nothing"


@cuda
def test_essgnn_equivariance_holds_on_cuda():
    """SC-5 on the device that will actually run it. float32 accumulation order
    differs on GPU, so the CPU tolerance is not evidence for this one."""
    from metafind.models.essgnn import ESSGNN, ESSGNNConfig

    cfg = ESSGNNConfig(node_feat_dim=32, edge_feat_dim=16, hidden_dim=32,
                       out_dim=64, n_layers=3, use_io_projections=True)
    model = ESSGNN(cfg).cuda().double()
    g = torch.Generator(device="cpu").manual_seed(0)
    nf = torch.randn(12, 32, generator=g).cuda().double()
    pos = torch.randn(12, 3, generator=g).cuda().double()
    ei = torch.stack([torch.arange(12).repeat_interleave(2) % 12,
                      torch.arange(24) % 12]).cuda()
    ea = torch.randn(24, 16, generator=g).cuda().double()

    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g).double())
    q = (q * torch.sign(torch.det(q))).cuda()
    t = torch.tensor([120.0, -75.0, 33.0]).cuda().double()

    base = model(nf, pos, ei, ea)
    moved = model(nf, pos @ q.T + t, ei, ea)
    err = (moved - base).abs().max().item()
    assert err < 1e-8, f"e_layout is not SE(3)-invariant on CUDA: {err:.3e}"
