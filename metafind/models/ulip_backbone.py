"""ULIP-2 backbone wrapper with a SPLIT freeze.

Two decisions, named rather than numbered: the formal deviation registry uses
D-1..D-6, and this module previously wrote "D1"/"D2" for unrelated local
decisions -- a one-hyphen difference that keyword search does not respect.

checkpoint_initialization: the released ULIP-2 checkpoint is the starting point
rather than pretraining it, because the official script assumes 8 GPUs and this
machine has one (F5). [CORRECTED 2026-08-22] The card is an RTX 5090 with
32,607 MiB, not the 24 GB this said. The reasoning is unaffected -- one card is
not eight -- but a wrong hardware figure in a feasibility premise gets reused as
a budget by whoever reads it next.

split_freeze_policy: the CLIP half is frozen; the POINT ENCODER IS TRAINABLE in
Stage 1. Freezing ViT-bigG-14 is what ULIP-2 specifies (its 3.3); whether that
also matches MetaFind's intent is U-34, and D-1 is conditional on it.

    paper 2.6  "Both query and gallery encoders are trained"
    paper 3.4  "Fine-tuning the entire encoder outperformed training the
                fuser only" -- Table 3 puts that row at 8.7 against 11.4

[CORRECTED] This module previously froze everything and said so: "the backbone
never trains, so every asset is encoded once into three 1280-d vectors and
everything downstream reads the cache". That IS the Table 3 `Train fuser only`
row, installed as the primary path. The specification was corrected rounds ago;
this file was not, so the documents described one experiment while the code
could only run the other.

What follows from the split:

    text, image   frozen ViT-bigG-14 (U-34's primary reading, which ULIP-2's
                  own 3.3 specifies) -> cacheable
    point cloud   trainable PointBERT + pc_projection       -> NOT cacheable,
                  because a cached embedding is by definition the output of a
                  network that is not being updated

`train_scope` selects which of the three levels in 02_BUILD_STEPS is active.
Only `fuser_only` may consume cached point-cloud embeddings.

Note on upstream, corrected: ULIP-2's paper 3.3 says it freezes OpenCLIP
("freeze it during the pre-training") and trains only the 3D encoder, so the
main line here agrees with ULIP-2 rather than deviating from it. Its public
code is a separate matter -- ULIP2_PointBERT_Colored calls eval() without
setting requires_grad=False, while the ULIP-1 factories beside it freeze
explicitly, so the omission reads as an oversight against its own paper.
Whether MetaFind wants CLIP trainable is U-34 and is open; D-1's status as a
deviation depends on it.

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
import contextlib
import sys
from typing import Literal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from metafind import paths

__all__ = ["BackboneConfig", "ULIPBackbone", "pc_norm"]

# From paths, NOT recomputed here. This used to build its own string ending in
# `data/sources/ulip2/...` while n02_download writes to `paths.ULIP2_CKPT`, which
# is `data/models/ulip2/...`. The 402 MB file had been on disk since 2026-08-15
# and every consumer reported it missing -- two spellings of one location, and
# nothing compared them because each side was internally consistent.
DEFAULT_CKPT = paths.ULIP2_CKPT
ULIP_REPO = Path(__file__).resolve().parents[1] / "vendor" / "ulip"

EMBED_DIM = 1280  # open_clip ViT-bigG-14 (finding F2)
PC_FEAT_DIM = 768  # PointBERT output, projected to EMBED_DIM
N_POINTS = 10000


TrainScope = Literal["fuser_only", "point_encoder_and_fuser", "full"]


@dataclass
class BackboneConfig:
    checkpoint: Path = DEFAULT_CKPT
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    # The main line. `fuser_only` is the Table 3 ablation; `full` also unfreezes
    # ViT-bigG-14 (D-1, audited by RA-3).
    #
    # [CORRECTED 2026-08-22] This said "does not fit on 24 GB". The card is
    # 32,607 MiB. **Whether `full` fits has NOT been re-measured against the
    # real figure** -- ViT-bigG-14 has ~2.5B parameters and an AdamW state is
    # roughly 12 bytes each before activations, so it is most likely still
    # infeasible, but that is an INFERENCE and the old sentence stated it as
    # measured. Do not quote either number as a measurement until one is taken.
    train_scope: TrainScope = "point_encoder_and_fuser"


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


DEPTH_SHELL_GREY = 0.5


def prepare_depth_shell(xyz: np.ndarray) -> np.ndarray:
    """Turn a ProcTHOR depth-shell cloud into ULIP input. [P0-4]

    n07b stores the fused shell in the SCENE's world frame, which is right --
    that frame is what the AI2-THOR bounding-box check compares against, and
    overwriting it would destroy the provenance that caught the orthographic
    depth bug. But the world frame is not what the checkpoint expects: capture
    lifts the asset to y = 40 m, so a stored cloud sits ~40 m off the origin at
    a scale in metres, while every Objaverse cloud n03 wrote is already centred
    and unit-scaled.

    Feeding both to the same PointBERT would put the two galleries in different
    input distributions -- a distribution shift OUR preprocessing created, not
    one the paper's domain gap forces. (The unavoidable gap is complete mesh
    surface versus visible depth shell; that one is real and gets declared.)

    So the normalisation happens here, at encode time, on the way in.

    Args:
        xyz: ``(N, 3)`` world-frame points.

    Returns:
        ``(1, N, 6)`` -- normalised xyz plus a flat grey, because the depth
        shell carries geometry only and a fabricated colour would be a claim.
    """
    normed = pc_norm(np.asarray(xyz, dtype=np.float64)).astype(np.float32)
    rgb = np.full_like(normed, DEPTH_SHELL_GREY)
    return np.concatenate([normed, rgb], axis=1)[None]


class ULIPBackbone:
    """Loads ULIP-2 once and exposes its three modality encoders.

    The CLIP halves are frozen and evaluated under no_grad. The point encoder
    follows `cfg.train_scope` and, on the main line, trains.
    """

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

        self.model = model.to(cfg.device, dtype=cfg.dtype)
        self._apply_train_scope()

        self.tokenizer = model.tokenizer
        # open_clip's preprocess pipeline lives on the constructed model, but the
        # factory does not return it, so rebuild the transform from its config.
        import open_clip

        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained=None
        )

    # ------------------------------------------------------------------ freezing

    def _point_parameters(self) -> list:
        """PointBERT plus the projection that maps it into the shared space."""
        params = list(self.model.point_encoder.parameters())
        if isinstance(getattr(self.model, "pc_projection", None), torch.nn.Parameter):
            params.append(self.model.pc_projection)
        return params

    def _apply_train_scope(self) -> None:
        """Freeze by scope, and put only the frozen halves in eval mode.

        eval() matters independently of requires_grad: it changes dropout and
        normalisation behaviour, so a module left in eval mode trains
        differently even with gradients enabled.
        """
        scope = self.cfg.train_scope
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        if scope == "fuser_only":
            return

        for p in self._point_parameters():
            p.requires_grad_(True)
        self.model.point_encoder.train()

        if scope == "full":
            # RA-3 records that this does not fit; the flag exists so the audit
            # can attempt it rather than assert infeasibility without trying.
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.train()

    def set_train_scope(self, scope: TrainScope) -> None:
        """Change the scope AFTER construction, re-applying grads and modes.

        Needed because loading a Stage 1 checkpoint and running inference want
        DIFFERENT scopes on the same object. The checkpoint's point-encoder
        section can only be restored into a backbone whose point encoder is
        trainable -- `load_stage1_checkpoint` checks coverage against exactly
        the `requires_grad` set -- but once restored, the gallery index and
        Stage 2 must run it frozen and in eval.

        Constructing with the inference scope and restoring afterwards is not an
        option, and constructing with the training scope and forgetting to
        switch back is the bug this method exists to make impossible to write:
        `drop_path_rate` is **0.1** in ULIP-2's PointBERT config, so a point
        encoder left in train() applies stochastic depth. The gallery index
        would come out non-deterministic, and nothing downstream checks it.
        """
        self.cfg.train_scope = scope
        self._apply_train_scope()

    def is_frozen(self) -> bool:
        """True when nothing is trainable AND nothing is in train() mode.

        Both halves matter and only one of them is about gradients.
        """
        return (not any(p.requires_grad for p in self.model.parameters())
                and not any(m.training for m in self.model.modules()))

    def trainable_parameters(self) -> list:
        """Exactly the parameters an optimizer should receive."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def named_trainable_parameters(self) -> list:
        """The same parameters, with their names.

        `trainable_parameters` cannot serve the weight-decay grouping rule that
        `upstream/ULIP/main.py:129-135` uses, because that rule reads the NAME:
        `bias` / `ln` / `bn` decide the group. Same filter, same order, names
        kept.
        """
        return [(n, p) for n, p in self.model.named_parameters()
                if p.requires_grad]

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

    def encode_pc(self, clouds: np.ndarray | Tensor) -> Tensor:
        """Encode point clouds. NOT wrapped in no_grad.

        This is the one encoder Stage 1 trains, so gradient must flow back into
        PointBERT. Callers that only need embeddings (the gallery index, the
        fuser_only ablation) wrap their own `torch.no_grad()`.

        Args:
            clouds: ``(B, N_POINTS, 6)`` of pre-normalised xyz plus rgb.

        Returns:
            ``(B, 1280)``.
        """
        x = torch.as_tensor(clouds) if not isinstance(clouds, Tensor) else clouds
        if x.dim() != 3 or x.shape[1:] != (N_POINTS, 6):
            raise ValueError(f"expected (B, {N_POINTS}, 6), got {tuple(x.shape)}")
        x = x.to(self.cfg.device, dtype=self.cfg.dtype)
        out = self.model.encode_pc(x)
        return self._check(out, "pc")

    def _clip_grad_context(self):
        """no_grad for the CLIP halves unless train_scope is `full`.

        [CORRECTED] These two were permanently decorated with @torch.no_grad().
        Under train_scope="full" that made RA-3 measure the wrong thing: the
        parameters had requires_grad=True, but no autograd graph was ever built
        through them, so ViT-bigG-14 could not receive gradient and the peak
        memory recorded was NOT that of full-encoder fine-tuning. An audit whose
        purpose is to attempt the infeasible thing and record what happens has
        to actually attempt it.
        """
        return (
            contextlib.nullcontext()
            if self.cfg.train_scope == "full"
            else torch.no_grad()
        )

    def encode_text(self, texts: list[str]) -> Tensor:
        """Encode captions. ``(B,)`` strings -> ``(B, D)``."""
        if not texts:
            raise ValueError("no texts given")
        tokens = self.tokenizer(texts).to(self.cfg.device)
        with self._clip_grad_context():
            out = self.model.encode_text(tokens)
        return self._check(out, "text")

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode preprocessed images. ``(B, 3, H, W)`` -> ``(B, D)``."""
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
        with self._clip_grad_context():
            out = self.model.encode_image(images.to(self.cfg.device, dtype=self.cfg.dtype))
        return self._check(out, "image")

    @staticmethod
    def _check(out: Tensor, what: str) -> Tensor:
        if out.size(-1) != EMBED_DIM:
            raise ValueError(f"{what} embedding is {out.size(-1)}-d, expected {EMBED_DIM}")
        if not torch.isfinite(out).all():
            raise ValueError(f"{what} embedding contains non-finite values")
        # Stay on the device. encode_pc is in Stage 1's autograd path, and a
        # .cpu() here would both break the graph's placement and force a
        # GPU->CPU->GPU round trip every step. Callers that want CPU tensors
        # (the gallery index writer, cached text/image embeddings) call .cpu()
        # themselves, where it is a deliberate transfer rather than a hidden one.
        return out.float()
