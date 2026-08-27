"""Tests for n10_train_stage1's GPU-free half.

The one that matters is L1-CKPT-TRAINABLE-ONLY. torch.save(state_dict()) on the
dual tower writes ViT-bigG-14 as well -- 2.5B frozen parameters, 10.2 GB -- and
across Table 3's eleven runs that is 112 GB against 1.9 GB, on a shared volume.
The failure is silent: correct training, correct results, files sixty times
bigger, discovered after the tenth ablation.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from metafind.train.stage1 import (
    build_model,
    collate,
    load_protocols,
    trainable_state_dict,
)


class Tower(nn.Module):
    """A frozen 'backbone' and a trainable head, in the same module."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen_backbone = nn.Linear(64, 64)
        self.trainable_fusion = nn.Linear(64, 8)
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)


# --- L1-CKPT-TRAINABLE-ONLY ------------------------------------------------

def test_only_trainable_parameters_are_saved():
    state = trainable_state_dict(Tower())
    assert all("trainable_fusion" in k for k in state)
    assert not any("frozen_backbone" in k for k in state)


def test_saving_the_whole_state_dict_is_much_larger():
    """[the injection] The whole point is the ratio, so measure it."""
    m = Tower()
    ours = sum(v.numel() for v in trainable_state_dict(m).values())
    whole = sum(v.numel() for v in m.state_dict().values())
    assert whole > 5 * ours


def test_unfreezing_the_backbone_puts_it_in_the_checkpoint():
    """Keyed off requires_grad, so it tracks the actual training scope rather
    than a name list that goes stale silently on a rename."""
    m = Tower()
    assert not any("frozen_backbone" in k for k in trainable_state_dict(m))
    for p in m.frozen_backbone.parameters():
        p.requires_grad_(True)
    assert any("frozen_backbone" in k for k in trainable_state_dict(m))


def test_the_saved_tensors_are_detached_and_on_cpu():
    """A checkpoint holding graph references keeps the whole autograd graph
    alive, and one on the GPU cannot be loaded on a machine without one."""
    for v in trainable_state_dict(Tower()).values():
        assert v.device.type == "cpu"
        assert not v.requires_grad


def test_a_name_prefix_filter_would_miss_a_renamed_module():
    """Why requires_grad and not a prefix list: renaming a module keeps the
    checkpoint saving and loading while quietly omitting a trained tensor."""
    m = Tower()
    m.add_module("newly_added_head", nn.Linear(8, 4))
    state = trainable_state_dict(m)
    assert any("newly_added_head" in k for k in state), (
        "a prefix list would have skipped this module entirely"
    )


# --- protocol refusals -----------------------------------------------------

def protocols(tmp_path, **over):
    enc = {"status": "resolved", "actual_clip_train_scope": "frozen",
           "image_aggregation": "mean", "missing_modality_representation": "learned_token"}
    train = {"status": "resolved", "fusion": "masked_mlp",
             "tower_sharing": "shared_backbone_separate_fusion",
             "allow_all_masked": True, "similarity": "cosine",
             "hyperparameter_config_hash": "abc123"}
    hp = {"sha256": "abc123", "values": {
        "optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 0.1,
        "scheduler": "cosine", "batch_size": 64, "epochs": 50, "max_epochs": 250,
        "p_mask": 0.30,
        "init_temperature": 0.07, "learnable_temperature": True,
        "max_logit_scale": 100.0,
        "betas": [0.9, 0.98], "eps": 1e-8, "warmup_epochs": 1,
        "lr_start": 1e-6, "lr_end": 1e-5,
        "seed": 1}}
    enc.update(over.get("enc", {}))
    train.update(over.get("train", {}))
    hp.update(over.get("hp", {}))
    (tmp_path / "stage1_encoding_protocol.json").write_text(json.dumps(enc))
    (tmp_path / "stage1_protocol.json").write_text(json.dumps(train))
    (tmp_path / "stage1_hyperparameters.json").write_text(json.dumps(hp))


def use(monkeypatch, tmp_path):
    import metafind.train.stage1 as s

    monkeypatch.setattr(s.paths, "OUTPUTS", tmp_path)


def test_resolved_protocols_load(monkeypatch, tmp_path):
    protocols(tmp_path)
    use(monkeypatch, tmp_path)
    enc, train, hp = load_protocols()
    assert enc["actual_clip_train_scope"] == "frozen"
    assert train["fusion"] == "masked_mlp"


def test_a_hyperparameter_hash_mismatch_stops_the_run(monkeypatch, tmp_path):
    """G3 dereferences this hash. A protocol pointing at a different artifact
    than the one on disk means the reported hyperparameters are not the ones
    that trained the model."""
    protocols(tmp_path, train={"hyperparameter_config_hash": "notthesame"})
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as exc:
        load_protocols()
    assert "different hyperparameter artifact" in str(exc.value)


@pytest.mark.parametrize("which", ["enc", "train"])
def test_an_unresolved_protocol_stops_the_run(monkeypatch, tmp_path, which):
    protocols(tmp_path, **{which: {"status": "unresolved"}})
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        load_protocols()


def test_a_missing_hyperparameter_stops_the_run(monkeypatch, tmp_path):
    """[U-22] The artifact must NAME every value; the run refuses on a partial
    one, because a default supplied here would never reach the report."""
    protocols(tmp_path)
    hp = json.loads((tmp_path / "stage1_hyperparameters.json").read_text())
    del hp["values"]["learning_rate"]
    (tmp_path / "stage1_hyperparameters.json").write_text(json.dumps(hp))
    use(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as exc:
        load_protocols()
    assert "learning_rate" in str(exc.value)


def test_a_missing_protocol_names_its_writer(monkeypatch, tmp_path):
    use(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        load_protocols()
    assert "n05b" in str(exc.value)


# --- batching --------------------------------------------------------------

def test_collate_stacks_and_keeps_uids_aligned():
    import numpy as np

    batch = [{"uid": f"u{i}", "text": np.zeros(4, np.float32),
              "image": np.ones(4, np.float32),
              "pc": np.full((10, 6), i, np.float32)} for i in range(3)]
    out = collate(batch)
    assert out["uid"] == ["u0", "u1", "u2"]
    assert out["text"].shape == (3, 4)
    assert out["pc"].shape == (3, 10, 6)
    # row i must still be asset i -- the loss pairs query i with gallery i
    assert torch.equal(out["pc"][2], torch.full((10, 6), 2.0))


# --- L1-LOSS-STAGE1-UNIDIRECTIONAL -----------------------------------------

def built(tmp_path, **over):
    protocols(tmp_path, **over)
    return build_model(*(json.loads((tmp_path / f).read_text()) for f in (
        "stage1_encoding_protocol.json", "stage1_protocol.json",
        "stage1_hyperparameters.json")))


def test_stage1_loss_is_query_to_gallery_only(tmp_path):
    """[L1-LOSS-STAGE1-UNIDIRECTIONAL] Eq. 5 has one direction; Eq. 7a/7b's
    symmetric form is Stage 2's, and the paper is explicit about the difference.

    This test exists because setting bidirectional=True in the trainer passed
    the entire suite -- the rule was written in validation_plan.yaml and
    enforced by nothing.
    """
    _, loss = built(tmp_path)
    assert loss.cfg.bidirectional is False


def test_the_loss_actually_computes_one_direction(tmp_path):
    """Not just the flag: swapping the two arguments must change the value.

    A symmetric objective is invariant to the swap, so this distinguishes the
    configuration from the behaviour -- a flag read by nothing would pass the
    test above and fail this one.
    """
    _, loss = built(tmp_path)
    torch.manual_seed(0)
    q = torch.randn(8, 16)
    g = torch.randn(8, 16)
    forward = loss(q, g)["loss"]
    swapped = loss(g, q)["loss"]
    assert not torch.isclose(forward, swapped), (
        "the loss is symmetric under a swap, which is Eq. 7's behaviour, not Eq. 5's"
    )


def test_the_trainer_uses_the_recorded_fusion_and_sharing(tmp_path):
    """n10 must not decide these; n09 wrote them and the report cites them."""
    model, _ = built(tmp_path)
    assert model.cfg.tower_sharing == "shared_backbone_separate_fusion"
    assert model.cfg.query_fusion.kind == "masked_mlp"


@pytest.mark.parametrize("rule,expect_zero_pad", [
    ("learned_token", False),
    ("zero_pad", True),
])
def test_the_missing_modality_rule_comes_from_the_encoding_protocol(
        tmp_path, rule, expect_zero_pad):
    """[U-11] 2.6 rules out zero-padding and names no replacement. n05b chose
    learned_token; the trainer must carry that choice rather than let a
    FusionConfig default decide -- which is how it was being decided before
    n05b existed. `zero_pad` is Table 3's "Padding missing modalities with 0"."""
    model, _ = built(tmp_path, enc={"missing_modality_representation": rule})
    assert model.cfg.query_fusion.zero_pad is expect_zero_pad


# --- the optimizer recipe: the two pieces stage1.py used to get wrong ---------

import random  # noqa: E402
import numpy as np  # noqa: E402
from metafind.train.stage1 import cosine_schedule, weight_decay_groups  # noqa: E402


def _named(**shapes):
    """Named parameters with the given shapes; every one requires grad."""
    return [(n, torch.nn.Parameter(torch.zeros(s))) for n, s in shapes.items()]


def test_biases_layernorm_and_1d_parameters_are_not_decayed():
    """[UPSTREAM upstream/ULIP/main.py:129-135] the rule reads the NAME."""
    named = _named(**{
        "block.weight": (4, 4),        # 2-D, plain name -> decayed
        "block.bias": (4,),            # name says bias
        "ln_final.weight": (4,),       # name says ln
        "bn1.weight": (4,),            # name says bn
        "logit_scale": (),             # 0-D, ndim < 2
        "proj.weight": (8, 4),         # 2-D, plain name -> decayed
    })
    wd, non_wd = weight_decay_groups(named, 0.1)
    assert wd["weight_decay"] == 0.1
    assert non_wd["weight_decay"] == 0.0
    assert len(wd["params"]) == 2
    assert len(non_wd["params"]) == 4


def test_a_two_dimensional_ln_parameter_still_escapes_decay():
    """The predicate is OR, not AND -- the name alone is enough.

    Without this the test above passes on a rule that only looked at ndim, which
    is the plausible wrong implementation.
    """
    named = _named(**{"ln_proj.weight": (4, 4)})
    wd, non_wd = weight_decay_groups(named, 0.1)
    assert len(wd["params"]) == 0
    assert len(non_wd["params"]) == 1


def test_a_frozen_parameter_never_reaches_the_optimizer():
    named = _named(**{"a.weight": (4, 4), "b.weight": (4, 4)})
    named[1][1].requires_grad_(False)
    wd, non_wd = weight_decay_groups(named, 0.1)
    assert len(wd["params"]) == 1
    assert len(non_wd["params"]) == 0


def test_the_schedule_warms_up_from_lr_start_and_lands_on_lr_end():
    """[UPSTREAM upstream/ULIP/utils/utils.py:215-226] the shape CosineAnnealingLR
    could not express: it has no warmup, and its floor is 0, not lr_end."""
    s = cosine_schedule(base=5e-4, final=1e-5, epochs=5, niter_per_ep=100,
                        warmup_epochs=1, start_warmup=1e-6)
    assert len(s) == 500
    assert s[0] == pytest.approx(1e-6)
    assert s.max() == pytest.approx(5e-4)
    assert s[-1] == pytest.approx(1e-5, rel=1e-2)
    assert s[99] == pytest.approx(5e-4)          # warmup ends at the base rate


def test_the_schedule_is_monotonic_up_then_down():
    s = cosine_schedule(base=5e-4, final=1e-5, epochs=4, niter_per_ep=50,
                        warmup_epochs=1, start_warmup=1e-6)
    warm, cos = s[:50], s[50:]
    assert (np.diff(warm) > 0).all()
    assert (np.diff(cos) < 0).all()


def test_warmup_equal_to_epochs_is_pure_warmup_and_needs_no_guard():
    """chain_to_stage1.sh runs `--epochs 1` while warmup_epochs is 1.

    Measured: upstream handles this case and returns exactly this array. The
    guard below is for a different input; this test exists so nobody "fixes"
    this case again on the strength of a comment that once claimed it broke.
    """
    s = cosine_schedule(base=5e-4, final=1e-5, epochs=1, niter_per_ep=3,
                        warmup_epochs=1, start_warmup=1e-6)
    assert len(s) == 3
    assert s[0] == pytest.approx(1e-6)
    assert s[-1] == pytest.approx(5e-4)


def test_warmup_longer_than_the_run_is_truncated_instead_of_raising():
    """This is what actually breaks upstream: its concatenation comes out the
    wrong length and its own assert fires. Warmup is clipped to the run."""
    s = cosine_schedule(base=5e-4, final=1e-5, epochs=1, niter_per_ep=5,
                        warmup_epochs=3, start_warmup=1e-6)
    assert len(s) == 5
    assert s[0] == pytest.approx(1e-6)
    assert s[-1] == pytest.approx(5e-4)
    assert (np.diff(s) > 0).all()


def test_an_empty_loader_yields_an_empty_schedule():
    assert len(cosine_schedule(base=5e-4, final=1e-5, epochs=5, niter_per_ep=0,
                               warmup_epochs=1, start_warmup=1e-6)) == 0


def test_no_warmup_starts_at_the_base_rate():
    s = cosine_schedule(base=5e-4, final=1e-5, epochs=2, niter_per_ep=4,
                        warmup_epochs=0, start_warmup=1e-6)
    assert len(s) == 8
    assert s[0] == pytest.approx(5e-4)


# --- [U-14] the aggregation field must decide something ----------------------

def _cache(tmp_path, n_views=12, dim=8):
    """n06's cache and n03's cloud, in SEPARATE directories as on disk.

    They collide otherwise: both are `<uid>.npz`, and pointing both roots at one
    tmp_path makes the dataset load the embedding file as the point cloud.
    """
    emb, pcs = tmp_path / "emb", tmp_path / "pc"
    emb.mkdir(); pcs.mkdir()
    views = np.stack([np.full(dim, float(i)) for i in range(n_views)])
    np.savez(emb / "u.npz", text=np.zeros(dim),
             image=views.mean(axis=0), views=views)
    np.savez(pcs / "u.npz", xyz=np.zeros((4, 3)), rgb=np.zeros((4, 3)))
    return views, emb, pcs


def _dataset(monkeypatch, tmp_path, aggregation, **kw):
    import metafind.train.stage1 as m

    views, emb, pcs = _cache(tmp_path, **kw)
    monkeypatch.setattr(m.paths, "EMBEDDINGS", emb)
    monkeypatch.setattr(m.paths, "POINTCLOUDS", pcs)
    return m.Stage1Dataset(["u"], aggregation), views


def test_mean_returns_the_pooled_vector_unchanged(monkeypatch, tmp_path):
    """The current protocol. This must not move."""
    ds, views = _dataset(monkeypatch, tmp_path, "mean")
    assert np.allclose(ds[0]["image"], views.mean(axis=0))


def test_random_single_view_returns_a_view_not_the_mean(monkeypatch, tmp_path):
    """[U-14] Setting this used to change NOTHING: the field was stored and
    never read, so the protocol could record per-view sampling while training
    ran on the 12-view mean."""
    ds, views = _dataset(monkeypatch, tmp_path, "random_single_view")
    got = ds[0]["image"]
    assert any(np.allclose(got, v) for v in views), "not one of the stored views"
    assert not np.allclose(got, views.mean(axis=0)), "still the pooled vector"


def test_random_single_view_draws_a_fresh_view_per_access(monkeypatch, tmp_path):
    """Per STEP, not per asset -- upstream samples again every time the asset is
    seen (ulip2 main.tex:612)."""
    ds, _ = _dataset(monkeypatch, tmp_path, "random_single_view")
    random.seed(0)
    seen = {float(ds[0]["image"][0]) for _ in range(60)}
    assert len(seen) > 1, "the same view every time is not sampling"


def test_an_unknown_aggregation_is_refused_at_construction():
    """Refusing beats returning the pooled vector under an unrecognised label --
    the failure this whole finding was about."""
    import metafind.train.stage1 as m

    with pytest.raises(ValueError, match="unknown image_aggregation"):
        m.Stage1Dataset(["u"], "median_of_the_first_three")
