"""The two Stage 2 guards the ULIP2 Reviewer found untested.

Both were written on 2026-09-03 under Kyzen's delegation of the four open
items, and neither had a single test -- which for a guard is the failure it
exists to prevent, one level up.
"""
import json

import numpy as np
import pytest
from torch import nn


def _text_map(pairs, cache_meta):
    m = {a: t for a, t in pairs}
    m["_meta"] = cache_meta
    return m


def test_the_lambda_ratio_is_refused_under_a_pooling_it_does_not_hold_for():
    """[ULIP2 REVIEWER MINOR 1] `lam = ratio x ||fused||` is the intended tenth
    only if `||e_layout|| == 1`.

    That is true today, and only because pooling is `normalised_sum`
    (`essgnn.py`, `s / (s.norm() + 1e-12)`). `essgnn.Pool` legally admits mean,
    sum and max, and under `sum` -- approved hours before normalised_sum -- the
    layout term measured 27x the fused query at init, so the same formula would
    land at ~2.7x rather than a tenth. The arithmetic was correct by a protocol
    value nothing read.
    """
    from metafind.train.stage2 import derive_init_lambda

    for pooling in ("sum", "mean", "max", None):
        with pytest.raises(SystemExit, match="unit-norm layout"):
            derive_init_lambda(None, [], {}, None, 0.1, "cpu",
                               arch_protocol={"pooling": pooling})


def test_the_lambda_derivation_measures_the_fused_query_and_records_its_basis(
        monkeypatch):
    """The ratio times a MEASURED median, with the measurement written down.

    `drop_layout=True` is what makes the measured quantity `||Fusion||` alone:
    the tower returns the fused vector unchanged when there is no layout, so
    nothing of ESSGNN or lambda is in the number lambda is a tenth of.
    """
    import torch

    import metafind.train.stage2 as m

    norms = [2.0, 4.0, 6.0, 100.0]     # median 5.0, one deliberate outlier
    seen = []

    # [ULIP2 REVIEWER MINOR 1] The stub used to return `iter(())` from
    # `modules()`, so `modules_in_eval` built an empty work list and the `with`
    # block did nothing -- deleting it from `derive_init_lambda` left all four
    # tests green. A real child in train mode makes the assertion real: the
    # helper exists because BatchNorm running stats moved under `no_grad`, which
    # is exactly what a 64-sample norm pass would do here.
    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

    class Model:
        def __init__(self):
            self.query = Q()

    def fake_encode_query(model, graph, target_index, asset_id, drop_layout,
                          device, data):
        seen.append(drop_layout)
        assert not model.query.bn.training, (
            "the norm pass must run with the query tower in EVAL -- otherwise "
            "BatchNorm running stats move while measuring, which is the "
            "measurement modules_in_eval was written after")
        return torch.tensor([norms[len(seen) - 1], 0.0, 0.0])

    monkeypatch.setattr(m, "encode_query", fake_encode_query)
    samples = [(f"h{i}", i, f"a{i}") for i in range(len(norms))]
    model = Model()
    assert model.query.bn.training, "the fixture must start in train mode"
    rec = m.derive_init_lambda(model, samples, {f"h{i}": {} for i in range(4)},
                               None, 0.1, "cpu",
                               arch_protocol={"pooling": "normalised_sum"})

    assert model.query.bn.training, (
        "train mode must be RESTORED after the measurement, not left in eval")
    assert all(seen), "the norm must be measured with the layout term OFF"
    assert rec["fused_query_norm_median"] == 5.0, "median, not mean"
    assert rec["init_lambda"] == pytest.approx(0.5)
    # The record has to carry HOW, or a finished run's lambda is unrecoverable
    # from its own provenance -- which is what the literal 9.0 left behind.
    for k in ("init_lambda_ratio", "fused_query_norm_n", "fused_query_norm_min",
              "fused_query_norm_max", "pooling", "layout_norm_assumed", "basis"):
        assert k in rec, k
    assert rec["fused_query_norm_max"] == 100.0, (
        "the outlier must be visible in the record even though the median "
        "ignores it")


def test_text_and_edges_that_disagree_stop_stage_two(monkeypatch, tmp_path):
    """[MASTER DECISION 2026-09-03] The repair is deferred; this is what makes
    deferring safe.

    `_edge_key` hashes the DESCRIPTIONS, so a node text regenerated without the
    edge stage changes 146 assetIds' keys, the lookup misses, and the learned
    missing token substitutes with nothing raised. Measured on the real corpus:
    0.00% today over 215,964 sampled edges, 11.04% against a text regenerated
    alone.

    Both directions here, because a guard verified only in the passing
    direction is the one that never fires.
    """
    import metafind.train.stage2 as m

    graphs = tmp_path / "scene_graphs"
    graphs.mkdir()
    for h in range(5):
        (graphs / f"h{h}.json").write_text(json.dumps({
            "nodes": [{"index": 0, "asset_id": "A", "category": "A"},
                      {"index": 1, "asset_id": "B", "category": "B"}],
            "sem_edge_ids": [[0, 1]]}))
    monkeypatch.setattr(m.paths, "SCENE_GRAPHS", graphs)

    meta = {"prompt_version": 1, "llm_model": "m", "text_encoder_version": "v"}

    class Data:
        pass

    d = Data()
    d.text_map = _text_map([("A", "a cd"), ("B", "a tv stand")], meta)
    d.sem_cache = {m._edge_key("A", "B", d.text_map): np.zeros(4)}
    # Every edge resolves: the guard must NOT fire.
    m.Stage2Data._assert_text_and_edges_agree(d)

    # The same cache under the PRE-FIX text, which is exactly the drift the
    # guard exists for: every key misses.
    d.text_map = _text_map([("A", "a c d"), ("B", "a t v stand")], meta)
    with pytest.raises(SystemExit, match="different generations"):
        m.Stage2Data._assert_text_and_edges_agree(d)


def test_the_guard_refuses_rather_than_skipping_itself_on_an_empty_corpus(
        monkeypatch, tmp_path):
    """[ULIP2 REVIEWER MINOR 3] `if not files: return` skipped the guard exactly
    when it had nothing to check, which is not a guard."""
    import metafind.train.stage2 as m

    empty = tmp_path / "none"
    empty.mkdir()
    monkeypatch.setattr(m.paths, "SCENE_GRAPHS", empty)

    class Data:
        text_map = {"_meta": {}}
        sem_cache = {}

    with pytest.raises(SystemExit, match="no scene graphs"):
        m.Stage2Data._assert_text_and_edges_agree(Data())


def test_stage2_actually_calls_the_text_edge_guard():
    """[ULIP2 REVIEWER MINOR 2] Every other test here invokes the guard unbound
    on a duck-typed object, so deleting its call site left them all green with
    the guard uninstalled.

    That is this project's first failure family -- a check written, tested, and
    not reached by the live path -- in the file that was just written to close
    it. Same technique as the initialiser test in test_train_stage1.py.
    """
    import inspect

    import metafind.train.stage2 as m

    src = inspect.getsource(m.Stage2Data.__init__)
    assert "_assert_text_and_edges_agree(" in src, (
        "Stage2Data.__init__ no longer calls the text/edge guard, so a corpus "
        "whose node text and semantic edges disagree would train silently.")
