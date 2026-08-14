"""Compatibility shims that make the upstream ULIP clone importable on modern PyTorch.

Import this module *before* importing anything from the ULIP repo::

    from metafind.compat import ulip_patch
    ulip_patch.apply()

Rationale (see docs/graph/00_FINDINGS.md F4 and F9):

* ``data/dataset_3d.py`` imports ``torch._six``, removed in PyTorch 2.0.
* ``models/pointbert/dvae.py`` imports ``knn_cuda`` at module level and instantiates
  ``KNN(k=4)``, but the only call site is commented out -- the live code path uses
  the pure-torch ``knn_point`` defined in the same file.
* ``models/pointbert/misc.py`` imports ``pointnet2_ops`` at module level. Unlike
  ``knn_cuda`` this one *is* used: ``misc.fps`` calls ``furthest_point_sample``.
  We replace ``misc.fps`` with a pure-torch implementation of the same greedy
  algorithm, so no CUDA extension has to be compiled.

The upstream clones at /home/kyzen/ULIP and /home/kyzen/egnn are never modified,
so they stay re-clonable and their git status stays clean.
"""

from __future__ import annotations

import sys
import types

import torch

__all__ = ["apply", "farthest_point_sample_idx", "fps"]

_APPLIED = False


def farthest_point_sample_idx(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Greedy farthest point sampling, returning indices.

    Mirrors ``pointnet2_ops.pointnet2_utils.furthest_point_sample``: seed at index
    0, track the running minimum squared distance to the chosen set, and take the
    argmax each step. Seeding at 0 (rather than randomly) is what makes the result
    deterministic and therefore reproducible.

    Args:
        xyz: ``(B, N, 3)`` coordinates.
        npoint: number of centres to select.

    Returns:
        ``(B, npoint)`` long tensor of indices into ``N``.
    """
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"expected (B, N, 3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    if npoint > n:
        raise ValueError(f"npoint={npoint} exceeds available points N={n}")

    device = xyz.device
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), float("inf"), dtype=xyz.dtype, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch_idx = torch.arange(b, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)

    return centroids


def fps(data: torch.Tensor, number: int) -> torch.Tensor:
    """Drop-in replacement for ``models.pointbert.misc.fps``.

    Args:
        data: ``(B, N, 3)`` coordinates.
        number: number of centres.

    Returns:
        ``(B, number, 3)`` sampled coordinates.
    """
    idx = farthest_point_sample_idx(data, number)
    return torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, data.shape[-1]))


def _install_torch_six() -> None:
    """Provide the ``torch._six`` names that dataset_3d.py still imports."""
    if "torch._six" in sys.modules:
        return
    mod = types.ModuleType("torch._six")
    mod.string_classes = str  # type: ignore[attr-defined]
    mod.int_classes = int  # type: ignore[attr-defined]
    mod.container_abcs = __import__("collections.abc", fromlist=["abc"])  # type: ignore[attr-defined]
    sys.modules["torch._six"] = mod


def _install_pointnet2_stub() -> None:
    """Stub ``pointnet2_ops`` so importing misc.py succeeds.

    The stub deliberately *raises* if anything actually calls into it. We replace
    the one real consumer (``misc.fps``) with a pure-torch version, so a call
    reaching this stub means an untested code path is live -- which should fail
    loudly rather than silently return something plausible.
    """
    if "pointnet2_ops" in sys.modules:
        return

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pointnet2_ops is intentionally not installed. A call reached the stub, "
            "which means a code path other than misc.fps needs it. Either route it "
            "through metafind.compat.ulip_patch or install the extension explicitly."
        )

    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    utils.furthest_point_sample = _unavailable  # type: ignore[attr-defined]
    utils.gather_operation = _unavailable  # type: ignore[attr-defined]

    pkg = types.ModuleType("pointnet2_ops")
    pkg.pointnet2_utils = utils  # type: ignore[attr-defined]

    sys.modules["pointnet2_ops"] = pkg
    sys.modules["pointnet2_ops.pointnet2_utils"] = utils


def _install_knn_cuda_stub() -> None:
    """Stub ``knn_cuda``. dvae.py instantiates KNN(k=4) at import time but never calls it."""
    if "knn_cuda" in sys.modules:
        return

    class _KNN:
        def __init__(self, k: int, transpose_mode: bool = False) -> None:
            self.k = k
            self.transpose_mode = transpose_mode

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError(
                "knn_cuda is intentionally not installed. The live path uses the "
                "pure-torch knn_point() in dvae.py; reaching this means the "
                "commented-out KNN path was re-enabled."
            )

    mod = types.ModuleType("knn_cuda")
    mod.KNN = _KNN  # type: ignore[attr-defined]
    sys.modules["knn_cuda"] = mod


def _patch_misc_fps() -> None:
    """Swap the pointnet2-backed ``misc.fps`` for the pure-torch one."""
    from models.pointbert import misc  # noqa: PLC0415  (import must follow the stubs)

    misc.fps = fps


def apply(patch_fps: bool = True) -> None:
    """Install every shim. Idempotent.

    Args:
        patch_fps: also import ULIP's ``misc`` and replace ``fps``. Set False if the
            ULIP repo is not on ``sys.path`` yet.
    """
    global _APPLIED
    _install_torch_six()
    _install_pointnet2_stub()
    _install_knn_cuda_stub()
    if patch_fps:
        _patch_misc_fps()
    _APPLIED = True


def is_applied() -> bool:
    return _APPLIED
