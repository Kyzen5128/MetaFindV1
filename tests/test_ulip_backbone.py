"""L1 tests for the frozen ULIP-2 backbone loader.

The checkpoint must be loaded with ``strict=False`` (974 open_clip keys come from
elsewhere), which makes silent failure easy: a renamed prefix leaves the point
encoder randomly initialised and still yields embeddings of the right shape and
norm. These tests exercise the guards against exactly that, using a stand-in
module so they run in milliseconds instead of loading ViT-bigG-14.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from metafind.models.ulip_backbone import EMBED_DIM, PC_FEAT_DIM, ULIPBackbone, pc_norm


class FakeULIP(nn.Module):
    """Same state-dict shape as ULIP2_WITH_OPENCLIP, small enough to be free."""

    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, PC_FEAT_DIM))
        self.pc_projection = nn.Parameter(torch.zeros(PC_FEAT_DIM, EMBED_DIM))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.open_clip_model = nn.Linear(4, 4)


def good_checkpoint(seed: int = 1) -> dict:
    torch.manual_seed(seed)
    ref = FakeULIP()
    with torch.no_grad():
        for p in ref.point_encoder.parameters():
            p.normal_(std=1.0)
        ref.pc_projection.normal_(std=1.0)
    sd = {k: v for k, v in ref.state_dict().items() if not k.startswith("open_clip_model")}
    return {"state_dict": sd}


def write(tmp_path, ckpt) -> str:
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return str(p)


# --------------------------------------------------------------- happy path


def test_valid_checkpoint_loads(tmp_path):
    model = FakeULIP()
    ULIPBackbone._load_and_verify(model, write(tmp_path, good_checkpoint()))
    assert model.pc_projection.abs().sum() > 0, "pc_projection stayed at its zero init"


def test_module_prefix_is_stripped(tmp_path):
    """DDP checkpoints carry a `module.` prefix that must not defeat the match."""
    ckpt = good_checkpoint()
    ckpt["state_dict"] = {f"module.{k}": v for k, v in ckpt["state_dict"].items()}
    ULIPBackbone._load_and_verify(FakeULIP(), write(tmp_path, ckpt))


# --------------------------------------------------------------- negative injections


def test_renamed_point_encoder_is_caught(tmp_path):
    """THE case this guard exists for: strict=False would accept this silently."""
    ckpt = good_checkpoint()
    ckpt["state_dict"] = {
        k.replace("point_encoder", "pointencoder"): v for k, v in ckpt["state_dict"].items()
    }
    with pytest.raises(ValueError, match="unexpected keys|no point_encoder"):
        ULIPBackbone._load_and_verify(FakeULIP(), write(tmp_path, ckpt))


def test_missing_point_encoder_is_caught(tmp_path):
    ckpt = good_checkpoint()
    ckpt["state_dict"] = {
        k: v for k, v in ckpt["state_dict"].items() if not k.startswith("point_encoder")
    }
    with pytest.raises(ValueError, match="no point_encoder|not loaded"):
        ULIPBackbone._load_and_verify(FakeULIP(), write(tmp_path, ckpt))


def test_missing_pc_projection_is_caught(tmp_path):
    ckpt = good_checkpoint()
    del ckpt["state_dict"]["pc_projection"]
    with pytest.raises(ValueError, match="pc_projection"):
        ULIPBackbone._load_and_verify(FakeULIP(), write(tmp_path, ckpt))


def test_wrong_projection_shape_is_caught(tmp_path):
    """A 512-wide projection means ULIP-1, whose embeddings are incompatible (F2)."""
    ckpt = good_checkpoint()
    ckpt["state_dict"]["pc_projection"] = torch.zeros(PC_FEAT_DIM, 512)
    with pytest.raises(ValueError, match="pc_projection is"):
        ULIPBackbone._load_and_verify(FakeULIP(), write(tmp_path, ckpt))


def test_checkpoint_identical_to_init_is_caught(tmp_path):
    """If the weights never move, the load did nothing however clean it looked."""
    model = FakeULIP()
    sd = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("open_clip_model")}
    with pytest.raises(ValueError, match="changed no point_encoder"):
        ULIPBackbone._load_and_verify(model, write(tmp_path, {"state_dict": sd}))


def test_open_clip_keys_are_allowed_to_be_absent(tmp_path):
    """974 open_clip keys are legitimately supplied elsewhere; that must not trip."""
    model = FakeULIP()
    ULIPBackbone._load_and_verify(model, write(tmp_path, good_checkpoint()))
    # No exception is the assertion; the guard must distinguish these from stray keys.


# --------------------------------------------------------------- pc_norm


def test_pc_norm_matches_ulip_preprocessing():
    """Centre at the origin, largest radius exactly 1 -- what the checkpoint expects."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(1000, 3)) * 5 + 100
    out = pc_norm(xyz)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)
    assert abs(np.sqrt((out**2).sum(axis=1)).max() - 1.0) < 1e-6


def test_pc_norm_is_translation_and_scale_invariant():
    """Two clouds differing by a rigid shift and a scale must normalise identically."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3))
    assert np.allclose(pc_norm(xyz), pc_norm(xyz * 7.0 + 33.0), atol=1e-6)


def test_pc_norm_rejects_degenerate_input():
    with pytest.raises(ValueError, match="degenerate"):
        pc_norm(np.ones((10, 3)))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        pc_norm(np.ones((10, 6)))


def test_pc_norm_negative_injection():
    """Skipping normalisation must be detectable, or the check proves nothing."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3)) * 5 + 100
    assert not np.allclose(xyz.mean(axis=0), 0, atol=1e-6)
