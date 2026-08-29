"""Tests for n10_train_stage1's GPU-free half.

The one that matters is L1-CKPT-TRAINABLE-ONLY. torch.save(state_dict()) on the
dual tower writes ViT-bigG-14 as well -- 2.5B frozen parameters, 10.2 GB -- and
across Table 3's eleven runs that is 112 GB against 1.9 GB, on a shared volume.
The failure is silent: correct training, correct results, files sixty times
bigger, discovered after the tenth ablation.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

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


# --------------------------------------------------------------------------
# [2026-08-28] The bug this exists for: `sched.get_last_lr()[0]` and `params`
# sat in the metrics block at stage1.py:593-596 and NEITHER NAME EXISTED. They
# were left behind when the torch scheduler was replaced by the precomputed
# `lr_schedule` array. The block runs at `step % 20 == 0`, so a real run raised
# NameError twenty steps into the first epoch: Stage 1 could not train at all.
#
# The thirteen tests above did not catch it because they test `cosine_schedule`
# and `weight_decay_groups` as pure functions and never enter the loop. Adding a
# fourteenth test of the same shape would not have caught it either.
#
# So this tests the RULE, not the case: no function in the training module may
# read a name that is not a parameter, not assigned in that function, not a
# module-level definition, and not a builtin. It is a static check because the
# loop it protects needs a 9.5 GB backbone and a GPU to enter.
def _module_level_names(tree):
    """Every name the module binds at import time, at any statement nesting.

    [ESSGNN ENGINEER 2026-08-28] Two blind spots, both found by running this
    guard over `metafind/data/*.py`, where they produced 25 false positives in
    `annotate.py` alone:

      1. Tuple unpacking. `MIN_DIM_CM, MAX_DIM_CM = 0.1, 10_000.0` has ONE
         target and it is an `ast.Tuple`, not two `ast.Name`. Reading
         `node.targets` and filtering to `ast.Name` collected neither name.
      2. Bindings inside a module-level block. `with open(...) as _fh:` then
         `LVIS_SYNSETS = json.load(_fh)[...]` binds at import time, but walking
         only `tree.body` never reaches it -- the old version special-cased
         `ast.If` to pick up imports and nothing else.

    Neither was a defect in the scanned file. Both were the guard failing to see
    legal Python, which is the worse failure: it would have reported working
    code as broken, and the natural response to 25 red lines is to weaken the
    check.

    So: walk everything EXCEPT the bodies of functions, lambdas and classes --
    those introduce their own scope and are checked separately. Anything else,
    however deeply nested in `with` / `try` / `if` / `for`, executes at import
    time and binds at module level.
    """
    SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    names, stack = set(), list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, SCOPED):
            names.add(getattr(node, "name", ""))     # the def binds its own name
            continue                                  # its body is a new scope
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                               ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                               ast.NamedExpr)):
            names |= _targets_of(node)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    names.discard("")
    return names


def _targets_of(node):
    """Names bound by one binding statement, unpacking tuples and lists."""
    def walk_target(t):
        if isinstance(t, ast.Name):
            return {t.id}
        if isinstance(t, ast.Starred):
            return walk_target(t.value)
        if isinstance(t, (ast.Tuple, ast.List)):
            return set().union(*(walk_target(e) for e in t.elts)) if t.elts else set()
        return set()          # attribute or subscript targets bind no new name

    out = set()
    for attr in ("targets", "target"):
        value = getattr(node, attr, None)
        if value is None:
            continue
        for t in (value if isinstance(value, list) else [value]):
            out |= walk_target(t)
    for item in getattr(node, "items", []):          # with ... as X
        if item.optional_vars is not None:
            out |= walk_target(item.optional_vars)
    return out


def _bound_in(fn):
    """Every name that function body could have bound by the time it is read.

    Deliberately generous -- a name bound anywhere in the function counts as
    bound everywhere in it. The test is for names that exist NOWHERE, which is
    the failure that actually happened; use-before-assignment is a different
    bug and a static walk cannot decide it without flow analysis.
    """
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            bound |= set(node.names)
    return bound


# [MASTER 2026-08-28] The first version of this test hardcoded
# `metafind/train/stage1.py`. The claim it was used to support -- "eight files
# scanned, only stage1 and stage2 hit" -- came from a throwaway run that the
# committed test could not reproduce: a conclusion whose producer does not
# produce it, which is the defect this session spent the day catching in others.
# Worse than the bookkeeping: every other module had NO standing guard, which is
# why `stage2.py`'s NameError survived until someone happened to look.
#
# `metafind/data/*.py` is deliberately NOT included. Two of its files were
# checked by hand and were clean; the rest were never checked, and listing them
# here would turn "not yet examined" into "guaranteed" by the act of writing it.
# Adding them is a separate piece of work with its own result to report.
def _scanned_modules():
    """[MASTER 2026-08-28] `data` added after the guard's own blind spots were
    fixed. It was held out while `annotate.py` produced 25 false positives, and
    those came from the guard failing to see module-level tuple unpacking and
    bindings inside `with`/`try` -- not from anything wrong in the file. Both are
    fixed, `data/` reads 0, and the calibration below is what earned the
    inclusion: a same-shaped bug seeded into a clean file in this directory goes
    red and names it."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    return sorted(p for d in ("train", "models", "data")
                  for p in (repo / "metafind" / d).glob("*.py")
                  if p.name != "__init__.py")


@pytest.mark.parametrize("module", _scanned_modules(), ids=lambda p: p.name)
def test_no_function_reads_a_name_that_does_not_exist(module):
    import builtins

    tree = ast.parse(module.read_text())
    # The module dunders exist in every module without being written there.
    # `__file__` in particular is read by real code (ulip_backbone.py:185) and is
    # not a missing name.
    DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
               "__loader__", "__builtins__", "__debug__"}
    known = _module_level_names(tree) | set(dir(builtins)) | DUNDERS

    def own_scope(fn):
        """Nodes belonging to THIS function, not to a function nested in it.

        Without this, `load_protocols`'s walk descends into its nested `read`
        and reports `read`'s own parameter as unbound. Each nested function is
        checked separately, with its enclosing scopes added.
        """
        NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        out = []
        stack = [n for n in fn.body if not isinstance(n, NESTED)]
        while stack:
            node = stack.pop()
            out.append(node)
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, NESTED):
                    stack.append(child)
        return out

    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    offenders = []
    for fn in functions:
        bound = _bound_in(fn) | known
        # a nested function may read its enclosing function's names
        for outer in functions:
            if fn is not outer and fn in ast.walk(outer):
                bound |= _bound_in(outer)
        for node in own_scope(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound:
                    offenders.append(f"{fn.name}() line {node.lineno}: {node.id}")
    assert not offenders, (
        f"{module.name}: these names are read and are bound nowhere this check "
        "can see:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\n\n"
        # [Codex 2026-08-28, MASTER ruled (b): state the limits, do not tighten]
        # The old wording was "a run reaches them and raises NameError". That
        # claims more than this check can establish, in both directions, and
        # saying so is the point -- the list below is where the next person
        # should look, not a disclaimer.
        "WHAT THIS CHECK DOES NOT ESTABLISH. It is deliberately generous: a\n"
        "name bound anywhere in a function counts as bound everywhere in it,\n"
        "because the failure it was built for is a name that exists NOWHERE.\n"
        "Verified false NEGATIVES -- these pass here and still raise at run\n"
        "time:\n"
        "    global ghost; return ghost        (no module-level `ghost`)\n"
        "    [x for x in items]; return x      (Python 3 comprehension scope)\n"
        "    except E as exc: ...; return exc  (Python deletes `exc` after)\n"
        "Verified false POSITIVE -- flagged although no run reaches it:\n"
        "    if False: return ghost\n"
        "Tightening it would make it half a type checker, and half a type\n"
        "checker is not maintained. It caught two real NameErrors today\n"
        "(stage1.py:593 and stage2.py:265) BECAUSE it is simple.")


# --------------------------------------------------------------------------
# [USER 2026-08-28] Kyzen: 「一 要」「二 對啊挑最好的」. The metric and the
# tie-break are METAFIND_NOTEBOOK.md:435-440 (D-3, ratified 2026-08-27):
# Mean R@1 across the seven Table 1 conditions, ties broken by Mean R@5.
def test_a_strictly_better_mean_r1_wins():
    from metafind.train.stage1 import better_checkpoint
    assert better_checkpoint((0.31, 0.60), (0.30, 0.99))


def test_mean_r5_breaks_a_tie_on_mean_r1():
    from metafind.train.stage1 import better_checkpoint
    assert better_checkpoint((0.30, 0.61), (0.30, 0.60))
    assert not better_checkpoint((0.30, 0.59), (0.30, 0.60))


def test_an_exact_tie_keeps_the_earlier_epoch():
    """Not a style point. `>=` here would move `best` forward on every epoch
    that improved on nothing, and the reported best epoch would stop being the
    epoch where the model was best."""
    from metafind.train.stage1 import better_checkpoint
    assert not better_checkpoint((0.30, 0.60), (0.30, 0.60))


def test_the_first_epoch_is_always_the_incumbent():
    from metafind.train.stage1 import better_checkpoint
    assert better_checkpoint((0.0, 0.0), None)


def test_no_dev_val_returns_empty_not_zero():
    """An empty gallery scoring 0.0 is how a selection metric starts choosing
    checkpoints at random. The caller has to be able to tell the two apart."""
    from metafind.train.stage1 import evaluate_dev_val
    assert evaluate_dev_val(None, None, [], "mean", "cpu", 8) == {}


def test_the_dev_phase_trains_on_dev_train_not_train():
    """D-3's whole point: dev_val is a SUBSET of train (`splits.split_dev`), so
    a dev-phase run that trained on `train` would score the model on assets it
    had just fitted. Read from the AST because entering main() needs a 9.5 GB
    backbone and a GPU.

    Two earlier versions of this test were themselves broken and are worth the
    warning: `"train" not in branch` can never pass ("train" is inside
    "dev_train"), and a regex over the source matched the sentence in the
    comment BELOW that explains the bug. A check that reads comments is checking
    prose. This one reads the parsed branch, so only real accesses count.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "metafind" / "train" / "stage1.py").read_text())

    branch = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                # ast.unparse normalises string quoting to single quotes;
                # comparing against a double-quoted spelling silently never
                # matched, and the test then failed on "the branch is gone".
                and ast.unparse(node.test).replace('"', "'")
                    == "args.phase == 'dev'"):
            branch = node.body
    assert branch is not None, "the phase branch is gone; D-3 is not implemented"

    reads_train, reads_dev_train = [], []
    for stmt in branch:
        for node in ast.walk(stmt):
            key = None
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == "splits"
                    and isinstance(node.slice, ast.Constant)):
                key = node.slice.value
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get"
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "splits"
                  and node.args and isinstance(node.args[0], ast.Constant)):
                key = node.args[0].value
            if key == "train":
                reads_train.append(node.lineno)
            elif key == "dev_train":
                reads_dev_train.append(node.lineno)

    assert reads_dev_train, "the dev phase must draw its training pool from dev_train"
    assert not reads_train, (
        f"the dev branch reaches splits['train'] at line(s) {reads_train}; "
        "dev_val is inside that pool, so scoring it would score assets the run "
        "just fitted")


def test_the_final_phase_cannot_score_dev_val():
    """[ULIP2 REVIEWER 2026-08-28] dev_val is a subset of the final phase's
    training pool, so a dev_val score inside a final run is meaningless while
    looking exactly like a real one. The guard must not rest on the list
    happening to be empty."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "metafind" / "train" / "stage1.py").read_text())

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "evaluate_dev_val"]
    assert calls, "evaluate_dev_val is never called; D-3's dev phase does nothing"

    def guards(node):
        out = []
        for parent in ast.walk(tree):
            if isinstance(parent, ast.If) and node in ast.walk(parent):
                out.append(ast.unparse(parent.test).replace('"', "'"))
        return out

    for call in calls:
        tests = guards(call)
        assert any("args.phase == 'dev'" in t for t in tests), (
            f"the evaluate_dev_val call at line {call.lineno} is not guarded by "
            f"the phase; its guards are {tests}")


# --------------------------------------------------------------------------
# [MASTER 2026-08-28, REJECT] `evaluate_dev_val` called `model.eval()` and left
# the BACKBONE in train mode. Stage 1's scope is `point_encoder_and_fuser` and
# `ulip_backbone.py:235` puts the point encoder into train(), whose stack holds
# 2 BatchNorm1d (momentum 0.1, track_running_stats=True) and 17 DropPath with
# drop_prob > 0. MASTER measured one `encode_pc` inside `torch.no_grad()`
# changing `running_mean` on both.
#
# no_grad stops gradients. It does not stop BatchNorm updating running stats.
# So every dev-val pass wrote dev-val's statistics into the model and training
# continued from there -- D-3 exists to let dev-val DECIDE without being FITTED,
# and that was fitting it, silently, with green tests and a plausible metric.
#
# Tested on a toy module rather than the 9.5 GB backbone: the defect is in the
# eval/restore mechanism, and a test that needs a GPU to run is a test that does
# not run.
def _toy_with_batchnorm():
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU())


def test_one_evaluation_does_not_move_batchnorm_running_stats():
    from metafind.train.stage1 import modules_in_eval

    toy = _toy_with_batchnorm()
    toy.train()
    x = torch.randn(8, 4)
    before = toy[1].running_mean.clone()

    with modules_in_eval(toy), torch.no_grad():
        toy(x)
    assert torch.equal(toy[1].running_mean, before), (
        "an evaluation moved the running statistics; dev-val is being fitted")

    # The refutation half, in the same test so it cannot rot separately: without
    # the guard the SAME forward pass does move them. If this ever stops being
    # true the test above has become vacuous and would pass on a broken guard.
    toy.train()
    with torch.no_grad():
        toy(x)
    assert not torch.equal(toy[1].running_mean, before), (
        "the unguarded pass no longer moves running stats, so the assertion "
        "above proves nothing -- re-derive what this test is checking")


class _SplitState(nn.Module):
    """What `ULIPBackbone.freeze()` actually leaves behind: the root in eval and
    one child in train (ulip_backbone.py:228 then :235). A flat toy cannot
    express this, and a flat toy is why the first version of the test below
    passed on a broken guard."""

    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.other = nn.Linear(4, 4)


def test_modules_in_eval_restores_the_whole_subtree_not_just_the_root():
    """[MASTER 2026-08-28] `nn.Module.train(mode)` RECURSES. Recording only the
    root's flag and restoring it with `.train(was)` drives every child to the
    root's value -- so a point encoder that arrived in train mode leaves in eval
    and stays there for the rest of the run: frozen BatchNorm statistics, all
    DropPath off, every forward different from the epoch before, no signal.

    The rule is "restore the subtree". The earlier test checked "restore the
    flag", which is one case of it, and it went green on the broken version."""
    from metafind.train.stage1 import modules_in_eval

    m = _SplitState()
    m.eval()
    m.point_encoder.train()
    assert not m.training and m.point_encoder.training       # the split state

    with modules_in_eval(m):
        assert not m.training and not m.point_encoder.training
        assert not m.point_encoder[1].training                # and the leaf

    assert not m.training, "the root was flipped to train"
    assert m.point_encoder.training, (
        "the child arrived in train mode and left in eval -- restoring the root "
        "with .train() recursed over it")
    assert m.point_encoder[1].training, "the leaf was not restored either"


def test_modules_in_eval_restores_what_it_found_rather_than_calling_train():
    """A module handed over already in eval must come back in eval."""
    from metafind.train.stage1 import modules_in_eval

    training, evaluating = _toy_with_batchnorm(), _toy_with_batchnorm()
    training.train()
    evaluating.eval()

    with modules_in_eval(training, evaluating):
        assert not training.training and not evaluating.training

    assert training.training, "a module that arrived in train mode was left in eval"
    assert not evaluating.training, "a module that arrived in eval was flipped to train"


def test_modules_in_eval_restores_even_when_the_body_raises():
    from metafind.train.stage1 import modules_in_eval

    toy = _toy_with_batchnorm()
    toy.train()
    with pytest.raises(RuntimeError):
        with modules_in_eval(toy):
            raise RuntimeError("boom")
    assert toy.training, "an exception left the model in eval mode"


def test_the_evaluator_puts_the_backbone_in_eval_not_only_the_tower():
    """The seeded-bug half of MASTER's finding, as a static check: the guard has
    to reach the backbone. `model.eval()` alone is what was there."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "metafind" / "train" / "stage1.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_dev_val")
    withs = [ast.unparse(item.context_expr)
             for node in ast.walk(fn) if isinstance(node, ast.With)
             for item in node.items]
    guard = [w for w in withs if w.startswith("modules_in_eval(")]
    assert guard, "evaluate_dev_val does not use modules_in_eval at all"
    assert "backbone" in guard[0], (
        f"the eval guard is {guard[0]!r} and does not reach the backbone; the "
        "point encoder holds the BatchNorm and DropPath that move")


# --------------------------------------------------------------------------
# [Codex 2026-08-28] Three findings that are about what happens AROUND the
# selection rather than the rule itself.
class _Args:
    def __init__(self, limit=None, phase="dev"):
        self.limit, self.phase = limit, phase


def test_a_smoke_run_cannot_overwrite_the_canonical_best_checkpoint():
    """`chain_to_stage1.sh` runs Stage 1 with --limit 200. That smoke went
    through the same selection and the same filename, so a ten-minute smoke
    started after a multi-hour development run would replace the selected
    checkpoint with one chosen over a 200-asset gallery -- same name, same
    shape, no signal."""
    # `best_paths` was replaced by `resolve_run_paths` on 2026-08-30 (CODEX
    # BLOCKER: the destination had to become a frozen object, and the fixed
    # filenames had to fail closed). The INVARIANT is unchanged and is what this
    # test protects, so it follows the function rather than being deleted.
    from metafind.train.stage1 import BEST_CKPT_PATH, resolve_run_paths

    full = resolve_run_paths(None, overwrite=True)
    smoke = resolve_run_paths(None, limit=200, overwrite=True)
    for rp in (full, smoke):
        # Release the live-run flock and remove what claiming left behind: this
        # is the real checkpoint directory, and a stray reservation there would
        # read as the record of a run that never happened.
        rp.release()
        rp.reservation.unlink(missing_ok=True)
        rp.lock.unlink(missing_ok=True)
    assert full.best_checkpoint == BEST_CKPT_PATH
    assert full.best_checkpoint != smoke.best_checkpoint
    assert full.best_record != smoke.best_record
    assert "200" in smoke.best_checkpoint.name, (
        "the smoke path should say what it was limited to")


def test_the_dev_phase_refuses_overlapping_pools():
    """splits.py enforces disjointness when it WRITES. A stale or hand-repaired
    splits.json satisfies every other check here while overlapping, and the AST
    test cannot see it -- it forbids reading splits["train"], not an overlap
    arriving inside dev_train."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "metafind" / "train" / "stage1.py").read_text())
    branch = next(n.body for n in ast.walk(tree)
                  if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and ast.unparse(n.test).replace('"', "'") == "args.phase == 'dev'")
    source = "\n".join(ast.unparse(stmt) for stmt in branch)
    assert "set(train_uids) & set(dev_val_uids)" in source, (
        "the dev branch does not check that the two pools are disjoint")
    raises = [n for stmt in branch for n in ast.walk(stmt)
              if isinstance(n, ast.Raise)]
    assert raises, "an overlap must stop the run, not warn"


def test_the_best_checkpoint_record_carries_the_pools_it_was_chosen_over():
    """A best is only meaningful against the pools it beat. Without the phase,
    the limit and the two sizes, `stage1_best.pt` from a smoke and from a full
    run are indistinguishable on disk -- the same missing-denominator problem
    U-09 names, one directory down."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "metafind" / "train" / "stage1.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "save_best")
    body = ast.unparse(fn)
    for field in ("phase", "limit", "n_train", "n_dev_val", "pools_sha256"):
        assert f"'{field}'" in body, f"the best-checkpoint record omits {field}"
    assert ".replace(" in body, "the best checkpoint is not written atomically"


# --------------------------------------------------------------------------
# [Codex 2026-08-28, ratified by MASTER] The checkpoint stored `named_parameters`
# and nothing else, so BatchNorm's running statistics -- which Stage 1 MOVES,
# because ulip_backbone.py:235 puts the point encoder in train() and PointBERT's
# stack holds BatchNorm1d -- were thrown away at every save. Reloading gave the
# trained weights on the ORIGINAL upstream statistics, and eval mode reads
# exactly those, so the restored encoder was neither the one that trained nor
# the one that was scored.
def test_a_trained_modules_running_stats_survive_save_and_reload():
    from metafind.train.stage1 import trainable_state_dict

    trained = nn.Sequential(nn.BatchNorm1d(4))
    trained.train()
    torch.manual_seed(0)
    trained(torch.randn(8, 4) * 3 + 7)          # moves running_mean off its init
    moved = trained[0].running_mean.clone()
    assert not torch.allclose(moved, torch.zeros(4)), "the fixture did not train"

    restored = nn.Sequential(nn.BatchNorm1d(4))
    restored.load_state_dict(trainable_state_dict(trained), strict=False)
    assert torch.equal(restored[0].running_mean, moved), (
        "the reloaded module has different BatchNorm statistics than the trained "
        "one -- eval mode reads these, so it is a different encoder")

    # The refutation, in the same test: a parameters-only dict does NOT carry
    # them. If this ever stops being true the assertion above proves nothing.
    params_only = {n: p.detach().cpu()
                   for n, p in trained.named_parameters() if p.requires_grad}
    naive = nn.Sequential(nn.BatchNorm1d(4))
    naive.load_state_dict(params_only, strict=False)
    assert not torch.equal(naive[0].running_mean, moved), (
        "a parameters-only checkpoint now carries running stats, so this test "
        "no longer demonstrates anything -- re-derive what it is checking")


def test_a_frozen_modules_buffers_are_still_not_saved():
    """The rule is unchanged: what upstream can rebuild byte-for-byte does not
    go in the checkpoint. Only a TRAINED module's statistics are unrebuildable."""
    from metafind.train.stage1 import trainable_state_dict

    frozen = nn.Sequential(nn.BatchNorm1d(4))
    for p in frozen.parameters():
        p.requires_grad_(False)
    assert trainable_state_dict(frozen) == {}, (
        "a frozen module contributed to the checkpoint; L1-CKPT-TRAINABLE-ONLY "
        "exists to keep the rebuildable half out")
