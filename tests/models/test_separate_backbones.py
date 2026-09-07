"""[U-16 reading B, P13] Two point paths: the clone, the seam, the checkpoint.

Stand-ins throughout -- the 9.5 GB checkpoint is not loaded here. What is
pinned: the clone shares no tensor with its parent and starts identical; the
seam routes the QUERY cloud through the clone and the gallery cloud through the
parent; a checkpoint written with two paths refuses to load into one, and one
written with one refuses a second.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from metafind.models import ulip_backbone as ub  # noqa: E402
from metafind.train import stage1  # noqa: E402


class _TinyPointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, x):                       # (B, N, 6) -> (B, 4)
        return self.blocks(x).mean(dim=1)


class _TinyULIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = _TinyPointEncoder()
        self.pc_projection = nn.Parameter(torch.randn(4, 1280) * 0.1)
        self.other = nn.Linear(2, 2)            # stands in for the CLIP towers

    def encode_pc(self, pc):
        return self.point_encoder(pc) @ self.pc_projection


def _stub_backbone(device="cpu"):
    """A `ULIPBackbone` without its __init__: the fields the clone reads."""
    bb = ub.ULIPBackbone.__new__(ub.ULIPBackbone)
    bb.cfg = ub.BackboneConfig(device=device, train_scope="point_encoder_and_fuser",
                               grad_checkpointing=False)
    bb.model = _TinyULIP()
    bb._apply_train_scope()
    return bb


def _clouds(b=3):
    return torch.randn(b, ub.N_POINTS, 6)


def test_the_clone_starts_identical_and_shares_no_tensor():
    torch.manual_seed(0)
    bb = _stub_backbone()
    q = bb.clone_point_path()
    x = _clouds()
    assert torch.allclose(bb.encode_pc(x), q.encode_pc(x))
    parent_ids = {id(p) for p in bb.model.parameters()}
    assert not any(id(p) in parent_ids for p in q.model.parameters())
    # same names as the parent's point path, so the checkpoint section reads alike
    assert {n for n, _ in q.named_trainable_parameters()} == {
        n for n, _ in bb.named_trainable_parameters()}


def test_training_the_clone_leaves_the_parent_untouched():
    bb = _stub_backbone()
    q = bb.clone_point_path()
    x = _clouds()
    before = bb.encode_pc(x).detach().clone()
    q.encode_pc(x).sum().backward()
    with torch.no_grad():
        for p in q.trainable_parameters():
            p -= 0.1 * p.grad
    assert torch.allclose(bb.encode_pc(x), before)
    assert not torch.allclose(q.encode_pc(x), before)


def test_the_seam_routes_query_pc_through_the_clone_only():
    bb = _stub_backbone()
    q = bb.clone_point_path()
    with torch.no_grad():
        q.model.pc_projection += 1.0            # make the two paths differ
    batch = {"pc": _clouds(), "text": torch.randn(3, 1280), "image": torch.randn(3, 1280)}
    query, gallery = stage1.split_embeds(batch, bb, "cpu", query_backbone=q)
    assert query is not gallery
    assert torch.allclose(gallery["pc"], bb.encode_pc(batch["pc"]))
    assert torch.allclose(query["pc"], q.encode_pc(batch["pc"]))
    assert query["text"] is gallery["text"] and query["image"] is gallery["image"]
    # without a clone the pre-existing contract holds: one dict, by identity
    query2, gallery2 = stage1.split_embeds(batch, bb, "cpu")
    assert query2 is gallery2


def test_the_checkpoint_sections_cover_the_clone():
    bb = _stub_backbone()
    q = bb.clone_point_path()
    model, loss_fn = nn.Linear(2, 2), nn.Linear(1, 1)
    sections = {"backbone_trainable_state": stage1.trainable_state_dict(bb.model),
                "tower_trainable_state": stage1.trainable_state_dict(model),
                "loss_trainable_state": stage1.trainable_state_dict(loss_fn)}
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        stage1.assert_checkpoint_covers_optimizer(bb, model, loss_fn, sections, query_backbone=q)
    sections["query_backbone_trainable_state"] = stage1.trainable_state_dict(q.model)
    stage1.assert_checkpoint_covers_optimizer(bb, model, loss_fn, sections, query_backbone=q)


def test_loading_refuses_a_mismatched_number_of_point_paths(tmp_path):
    bb = _stub_backbone()
    q = bb.clone_point_path()
    model, loss_fn = nn.Linear(2, 2), nn.Linear(1, 1)
    two = {"backbone_trainable_state": stage1.trainable_state_dict(bb.model),
           "tower_trainable_state": stage1.trainable_state_dict(model),
           "loss_trainable_state": stage1.trainable_state_dict(loss_fn),
           "query_backbone_trainable_state": stage1.trainable_state_dict(q.model)}
    p2 = tmp_path / "two.pt"; torch.save(two, p2)
    with pytest.raises(ValueError, match="query_backbone_trainable_state"):
        stage1.load_stage1_checkpoint(bb, model, loss_fn, p2)          # two paths, one given
    stage1.load_stage1_checkpoint(bb, model, loss_fn, p2, query_backbone=q)
    one = {k: v for k, v in two.items() if k != "query_backbone_trainable_state"}
    p1 = tmp_path / "one.pt"; torch.save(one, p1)
    with pytest.raises(ValueError, match="query_backbone_trainable_state"):
        stage1.load_stage1_checkpoint(bb, model, loss_fn, p1, query_backbone=q)
    stage1.load_stage1_checkpoint(bb, model, loss_fn, p1)
