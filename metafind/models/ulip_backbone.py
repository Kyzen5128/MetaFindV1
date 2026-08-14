"""Frozen ULIP-2 backbone wrapper (decisions D1 and D2).

D1: the released ULIP-2 checkpoint is used rather than pretraining it, because
the official script assumes 8 GPUs and this machine has one 24 GB card (F5).

D2: the backbone never trains, so every asset is encoded **once** into three
1280-d vectors and everything downstream reads the cache. That turns training
into an MLP-scale problem and makes the Table 3 sweep affordable.

Why loading needs assertions
----------------------------

``load_state_dict(strict=False)`` is required here, because the checkpoint holds
only ``point_encoder`` (226 tensors), ``pc_projection`` and ``logit_scale`` --
the open_clip half is supplied by ``open_clip.create_model_and_transforms``
during construction, so 974 keys are legitimately "missing".

That leniency is dangerous: a renamed prefix would leave the point encoder at its
random initialisation and produce embeddings that look entirely plausible --
right shape, right norm -- while encoding nothing. Retrieval would simply be bad,
and the cause would be invisible. So the loader asserts that every missing key
belongs to open_clip, that the point-cloud weights are present, and that they
actually changed the module's parameters.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = ["BackboneConfig", "ULIPBackbone", "pc_norm"]

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "data/sources/ulip2/ULIP-2/pretrained_models"
    / "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
)
ULIP_REPO = Path(__file__).resolve().parents[1] / "vendor" / "ulip"

EMBED_DIM = 1280  # open_clip ViT-bigG-14 (finding F2)
PC_FEAT_DIM = 768  # PointBERT output, projected to EMBED_DIM
N_POINTS = 10000


@dataclass
class BackboneConfig:
    checkpoint: Path = DEFAULT_CKPT
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


def pc_norm(xyz: np.ndarray) -> np.ndarray:
    """Centre and unit-scale a point cloud, exactly as ULIP's dataset does.

    Mirrors ``Objaverse_Lvis_Colored.pc_norm``: subtract the centroid, divide by
    the largest radius. This must match ULIP's preprocessing exactly -- the
    checkpoint was trained on clouds normalised this way, and feeding raw
    coordinates would silently move every embedding off-distribution.

    Note this is the *asset* cloud, unrelated to scene coordinates, which stay
    unnormalised so ESSGNN's equivariance means something.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {xyz.shape}")
    centred = xyz - xyz.mean(axis=0)
    scale = np.max(np.sqrt((centred**2).sum(axis=1)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("degenerate point cloud: all points coincide")
    return centred / scale


class ULIPBackbone:
    """Loads ULIP-2 once, exposes the three modality encoders, never trains."""

    def __init__(self, cfg: BackboneConfig | None = None) -> None:
        self.cfg = cfg = cfg or BackboneConfig()
        if not cfg.checkpoint.exists():
            raise FileNotFoundError(
                f"ULIP-2 checkpoint not found at {cfg.checkpoint}. "
                "Fetch it from HF SFXX/ulip (ULIP-2/pretrained_models/)."
            )

        from metafind.compat import ulip_patch

        for p in (str(Path(__file__).resolve().parents[2]), str(ulip_patch.ULIP_ROOT)):
            if p not in sys.path:
                sys.path.insert(0, p)

        ulip_patch.apply(patch_fps=True)
        from models.ULIP_models import ULIP2_PointBERT_Colored

        with ulip_patch.ulip_cwd():
            model = ULIP2_PointBERT_Colored(argparse.Namespace(npoints=N_POINTS))

        self._load_and_verify(model, cfg.checkpoint)

        self.model = model.to(cfg.device, dtype=cfg.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = model.tokenizer
        # open_clip's preprocess pipeline lives on the constructed model, but the
        # factory does not return it, so rebuild the transform from its config.
        import open_clip

        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained=None
        )

    @staticmethod
    def _load_and_verify(model, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        required = ("pc_projection", "logit_scale")
        for key in required:
            if key not in sd:
                raise ValueError(f"checkpoint is missing {key!r}; got {sorted(sd)[:5]}")
        if sd["pc_projection"].shape != (PC_FEAT_DIM, EMBED_DIM):
            raise ValueError(
                f"pc_projection is {tuple(sd['pc_projection'].shape)}, "
                f"expected {(PC_FEAT_DIM, EMBED_DIM)} -- wrong ULIP variant?"
            )
        n_pc = sum(1 for k in sd if k.startswith("point_encoder"))
        if n_pc == 0:
            raise ValueError("checkpoint carries no point_encoder weights")

        before = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k.startswith("point_encoder") and v.dtype.is_floating_point
        }

        missing, unexpected = model.load_state_dict(sd, strict=False)

        # Everything absent must belong to open_clip, which is loaded separately.
        stray = [k for k in missing if not k.startswith("open_clip_model.")]
        if stray:
            raise ValueError(
                f"{len(stray)} non-open_clip keys were not loaded, so part of the "
                f"backbone is still randomly initialised: {stray[:5]}"
            )
        if unexpected:
            raise ValueError(f"checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}")

        # The decisive check: weights must have actually moved. Key-name drift
        # would otherwise leave a random point encoder producing embeddings of
        # the right shape and norm that encode nothing.
        after = model.state_dict()
        changed = sum(1 for k, v in before.items() if not torch.equal(v, after[k]))
        if changed == 0:
            raise ValueError(
                "loading changed no point_encoder parameter; the checkpoint did "
                "not take effect despite reporting success"
            )

    # ------------------------------------------------------------------ encoders

    @torch.no_grad()
    def encode_pc(self, clouds: np.ndarray | Tensor) -> Tensor:
        """Encode point clouds.

        Args:
            clouds: ``(B, N_POINTS, 6)`` of pre-normalised xyz plus rgb.

        Returns:
            ``(B, 1280)`` float32 on CPU.
        """
        x = torch.as_tensor(clouds) if not isinstance(clouds, Tensor) else clouds
        if x.dim() != 3 or x.shape[1:] != (N_POINTS, 6):
            raise ValueError(f"expected (B, {N_POINTS}, 6), got {tuple(x.shape)}")
        x = x.to(self.cfg.device, dtype=self.cfg.dtype)
        out = self.model.encode_pc(x)
        return self._check(out, "pc")

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> Tensor:
        """Encode captions. ``(B,)`` strings -> ``(B, 1280)``."""
        if not texts:
            raise ValueError("no texts given")
        tokens = self.tokenizer(texts).to(self.cfg.device)
        out = self.model.encode_text(tokens)
        return self._check(out, "text")

    @torch.no_grad()
    def encode_image(self, images: Tensor) -> Tensor:
        """Encode preprocessed images. ``(B, 3, H, W)`` -> ``(B, 1280)``."""
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
        out = self.model.encode_image(images.to(self.cfg.device, dtype=self.cfg.dtype))
        return self._check(out, "image")

    @staticmethod
    def _check(out: Tensor, what: str) -> Tensor:
        if out.size(-1) != EMBED_DIM:
            raise ValueError(f"{what} embedding is {out.size(-1)}-d, expected {EMBED_DIM}")
        if not torch.isfinite(out).all():
            raise ValueError(f"{what} embedding contains non-finite values")
        return out.float().cpu()
