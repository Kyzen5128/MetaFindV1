"""Tests for n15_eval_retrieval's scorer.

The scorer is the only part of Table 1 that can be wrong quietly: a wrong
protocol produces a number, a wrong rank produces a number, and both look like
results. Every test here fixes a number by hand rather than by running the code
under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from metafind.eval.retrieval import (
    QUERY_CONDITIONS,
    condition_mask,
    normalize_for_scoring,
    rank_of_target,
    recall_at_k,
)


# --- the seven conditions ----------------------------------------------------

def test_table_one_has_exactly_the_paper_s_seven_conditions():
    """[PAPER 3experiments.tex:24] text, image, pc, and the four combinations."""
    assert set(QUERY_CONDITIONS) == {
        "text", "image", "pc", "text+image", "text+pc", "image+pc", "full"}


def test_every_condition_gives_at_least_one_modality():
    """An all-absent query is sec. 2.6's 2.7% training edge case (U-23), not a
    Table 1 row."""
    assert all(any(flags) for flags in QUERY_CONDITIONS.values())


def test_a_condition_name_matches_the_modalities_it_switches_on():
    """The names are the table's labels; a mismatch would mislabel a whole
    column and nothing downstream would notice."""
    for name, flags in QUERY_CONDITIONS.items():
        wanted = set(name.split("+")) if name != "full" else {"text", "image", "pc"}
        got = {m for m, on in zip(("text", "image", "pc"), flags) if on}
        assert got == wanted, name


def test_the_mask_is_deterministic_not_the_training_sampler():
    """Reusing `sample_modality_mask` here would hand each query a random subset
    and report it under a fixed label."""
    a = condition_mask("text+pc", 16)
    b = condition_mask("text+pc", 16)
    assert a.shape == (16, 3)
    assert bool((a == b).all())
    assert [bool(v) for v in a[0]] == [True, False, True]


def test_an_unknown_condition_is_refused():
    with pytest.raises(ValueError, match="unknown query condition"):
        condition_mask("text+layout", 4)


def test_scoring_normalisation_is_shared_float64():
    x = np.array([[3.0, 4.0], [1.0, -1.0]], dtype=np.float32)
    got = normalize_for_scoring(x)
    assert got.dtype == np.float64
    assert np.array_equal(got[0], np.array([0.6, 0.8], dtype=np.float64))
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0)


@pytest.mark.parametrize("bad", [
    np.array([1.0, 2.0]),
    np.array([[0.0, 0.0]]),
    np.array([[np.nan, 1.0]]),
])
def test_scoring_normalisation_refuses_undefined_inputs(bad):
    with pytest.raises(ValueError):
        normalize_for_scoring(bad)


# --- ranking -----------------------------------------------------------------

def test_the_target_column_is_used_not_the_diagonal():
    """Under B_full_gallery the query's asset sits wherever the index put it.

    Assuming the diagonal would score query i against asset i and measure
    nothing -- and it would still return a plausible number.
    """
    sim = np.array([[0.1, 0.9, 0.2],
                    [0.8, 0.1, 0.3]])
    assert list(rank_of_target(sim, np.array([1, 0]))) == [1, 1]
    assert list(rank_of_target(sim, np.array([0, 1]))) == [3, 3]


def test_ties_count_against_the_model():
    """A model returning identical scores everywhere must not score R@1 = 100%."""
    sim = np.ones((4, 10))
    ranks = rank_of_target(sim, np.arange(4))
    assert list(ranks) == [10, 10, 10, 10]
    assert recall_at_k(sim, np.arange(4))["R@1"] == 0.0


def test_a_perfect_model_scores_one():
    sim = np.eye(5) * 2 - 1
    m = recall_at_k(sim, np.arange(5))
    assert m["R@1"] == 1.0
    assert m["R@5"] == 1.0


def test_recall_at_five_counts_ranks_two_through_five():
    """Hand-fixed: the target is the 3rd best for every query."""
    sim = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), (4, 1))
    m = recall_at_k(sim, np.full(4, 2))
    assert m["R@1"] == 0.0
    assert m["R@5"] == 1.0


def test_every_cell_carries_its_denominators():
    """[registry:891] the postcondition -- an R@1 without n_gallery cannot be
    compared to anything, which is the whole of U-09."""
    m = recall_at_k(np.zeros((3, 40)), np.arange(3))
    assert m["n_query"] == 3
    assert m["n_gallery"] == 40


def test_a_target_outside_the_gallery_is_refused():
    """Silently clipping would score against the wrong asset."""
    with pytest.raises(ValueError, match="outside the gallery"):
        rank_of_target(np.zeros((2, 3)), np.array([0, 7]))


def test_a_target_count_that_does_not_match_the_queries_is_refused():
    with pytest.raises(ValueError, match="targets for"):
        rank_of_target(np.zeros((3, 5)), np.array([0, 1]))


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(np.zeros((2, 4)), np.arange(2), ks=(0,))


# ============================================================================
# n15's gallery source: the promoted index, or an explicitly-marked re-encode
# ============================================================================
#
# [DL-048] n15 re-encoded the whole gallery on every run and had never once
# opened `gallery_index.json`. The artifact n11 stages, G4 verifies and n12
# promotes had NO CONSUMER on the evaluation path, so Table 1's A and B would
# have been scored against bytes no gate ever saw.
#
# ⚠ SCOPE OF EVERY TEST BELOW: SYNTHETIC. The fixtures are hand-written
# vectors. Whether the promoted index and a live re-encode produce the same
# vectors for the REAL corpus is UNVERIFIED and is not tested here -- it cannot
# be, because no promoted index exists yet. `test_an_index_backed_gallery_
# scores_identically_to_the_same_vectors_re_encoded` proves that n15's index
# path and its encode path agree GIVEN THE SAME VECTORS. It proves nothing
# about whether n11's encoder and n15's encoder produce the same vectors: n11
# encodes one asset at a time, n15 encodes in batches of 64, and a different
# batch shape can select a different kernel. That measurement needs a real
# index and is not claimed by anything in this file.

import hashlib
import json
from pathlib import Path

from metafind.eval import run_retrieval as n15
from metafind.train import gallery_index

CKPT_SHA = "a" * 64
OTHER_CKPT_SHA = "b" * 64


def encoder_pair(seed: int = 0):
    """An object `gallery_encoder_sha256` can really hash.

    It reads `backbone.model` and `model.gallery` and only their
    `named_parameters`, so one object can stand in for both. The gallery tower
    RAISES if it is ever called: on a reported protocol nothing may invoke it.
    """
    import torch

    class Boom(torch.nn.Linear):
        def forward(self, *a, **k):
            raise AssertionError(
                "the gallery tower was CALLED. A reported protocol takes its "
                "gallery from the promoted index and encodes nothing.")

    torch.manual_seed(seed)
    pair = type("Pair", (), {})()
    pair.model = torch.nn.Linear(2, 2)
    pair.gallery = Boom(2, 2)
    return pair


def write_promoted(tmp_path, monkeypatch, uids, vectors, encoder_sha,
                   key_sha=CKPT_SHA, record_sha=None, gate=True):
    """A promoted registry and its .npz, written by the REAL producer.

    `gallery_index.build_index` is what n11 calls, so the fixture cannot drift
    from the format the loader expects -- which is the whole failure this file
    is about.

    `gate=False` omits `gate_record_uri` / `gate_record_sha256`, which
    `promote()` writes on every entry it publishes. An entry without them was
    not written by promotion.
    """
    record = gallery_index.build_index(
        np.asarray(vectors, dtype=np.float32), list(uids), tmp_path / "gi.npz")
    record["stage1_checkpoint_sha256"] = record_sha or key_sha
    record["gallery_encoder_sha256"] = encoder_sha
    if gate:
        record["gate_record_uri"] = str(tmp_path / "G4_gallery_freeze.yaml")
        record["gate_record_sha256"] = "d" * 64
    registry = tmp_path / "gallery_index.json"
    registry.write_text(json.dumps({key_sha: record}))
    monkeypatch.setattr(gallery_index, "PROMOTED_PATH", registry)
    return record


def fake_encode_pools(vectors_by_uid, queries, seen=None, packs=None):
    """An `encode_pools` that never touches a file, a model or a GPU.

    `seen` collects the `gallery_uids` it was asked for, so a test can assert
    that a reported protocol asked for NONE. `packs` collects the `query_pack`,
    so a test can assert the query construction reached the encoder.

    The signature MIRRORS the real `encode_pools`, positionally. It is a double,
    and a double whose parameters have drifted from the function it stands in
    for stops testing that function -- it starts testing an older one, silently.
    `query_pack` was added to the real signature on 2026-08-31 and this raised a
    TypeError rather than passing, which is the behaviour worth keeping.
    """
    def encode(backbone, model, query_uids, gallery_uids, aggregation,
               device, batch_size, query_pack=None, norms_out=None):
        if seen is not None:
            seen.append(gallery_uids)
        if packs is not None:
            packs.append(query_pack)
        gallery = None if gallery_uids is None else normalize_for_scoring(
            np.stack([vectors_by_uid[u] for u in gallery_uids]))
        return ({c: normalize_for_scoring(queries) for c in QUERY_CONDITIONS},
                gallery)
    return encode


# --- the ordering fixture ----------------------------------------------------
#
# The index is written in `sorted(train + test)`; `resolve_split(splits,
# "full")` returns `list(train) + list(test)`. These are DIFFERENT orders, and
# the fixture is built so that they are: train is listed t1 before t0.
ORDER_SPLITS = {"train": ["t1", "t0"], "test": ["x1", "x0"], "dev_val": ["t1"]}
ORDER_UIDS = ["t0", "t1", "x0", "x1"]                 # the index's own order
ONE_HOT = {u: np.eye(4, dtype=np.float32)[i] for i, u in enumerate(ORDER_UIDS)}

REPORTED_FULL = {"query_split": "test", "gallery_split": "full",
                 "reported": True}
DEV = {"query_split": "dev_val", "gallery_split": "dev_val", "reported": False}


def test_the_registry_reader_n15_calls_is_the_one_gallery_index_owns():
    """n15 must not parse `gallery_index.json` itself.

    Two readers of one registry is how the two halves drift apart: the loader
    verifies the bytes and hands back the ones it verified, and a second reader
    in this module would be a second open of the same path.

    ⚠ The skip below is conditioned on the SYMBOL's absence, so it self-clears
    the moment the loader lands. A test that still skips after that is a test
    that does not exist.
    """
    if not hasattr(gallery_index, "load_promoted_index_for_checkpoint"):
        pytest.skip("gallery_index.load_promoted_index_for_checkpoint has not "
                    "landed; n15's reported path cannot run at all")
    import inspect
    params = list(inspect.signature(
        gallery_index.load_promoted_index_for_checkpoint).parameters)
    assert params[0] == "checkpoint_sha", params
    # And n15 does not open the registry behind its back.
    src = inspect.getsource(n15)
    assert "gallery_index.json" not in src.replace(
        "`gallery_index.json`", ""), (
        "run_retrieval names the registry file outside a docstring reference; "
        "it must reach it only through load_promoted_index_for_checkpoint")


def test_an_index_backed_gallery_scores_identically_to_the_same_vectors_re_encoded(
        tmp_path, monkeypatch):
    """[B4 round 1] The ONE parity claim that is provable today.

    ⚠ It is a claim about n15, not about the corpus: given the SAME gallery
    vectors, the promoted-index path and the direct-encode path produce the
    same ranks, R@1 and R@5. It says nothing about whether n11's per-asset
    encode and n15's batch-64 encode agree on the real corpus. That remains
    UNVERIFIED until a real index exists.

    The fixture is hand-fixed at R@1 = 0.5, and that matters: two paths that
    both score 0.0 or both score 1.0 agree for free. Two of the four queries
    are their own asset's vector (rank 1) and two are some other asset's
    (rank > 1), so the equality below is between numbers that had somewhere
    else to be.
    """
    rng = np.random.default_rng(19)
    uids = [f"u{i}" for i in range(9)]
    vectors = rng.normal(size=(9, 6)).astype(np.float32)
    by_uid = dict(zip(uids, vectors))
    splits = {"train": uids[:5], "test": uids[5:], "dev_val": uids[5:]}
    # queries are u5..u8; the first two ARE their own asset, the last two are
    # u0's and u1's vectors, so their own asset cannot be the argmax.
    queries = np.stack([vectors[5], vectors[6], vectors[0], vectors[1]]
                       ).astype(np.float64)

    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, uids, vectors, enc)

    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(by_uid, queries))
    indexed, _ = n15.run_protocol(
        "reported", {"query_split": "test", "gallery_split": "full",
                     "reported": True},
        splits, pair, pair, "mean", "cpu", 4, "none", 0, 3, False,
        {"sha256": CKPT_SHA})
    direct, _ = n15.run_protocol(
        "dev", {"query_split": "test", "gallery_split": "full",
                "reported": False},
        splits, pair, pair, "mean", "cpu", 4, "none", 0, 3, False,
        {"sha256": CKPT_SHA})

    assert indexed["gallery_source"] == "promoted_index"
    assert direct["gallery_source"] == "direct_dev_encode"
    for cond in QUERY_CONDITIONS:
        a, b = indexed["conditions"][cond], direct["conditions"][cond]
        assert (a["R@1"], a["R@5"], a["hits@1"], a["hits@5"]) == \
               (b["R@1"], b["R@5"], b["hits@1"], b["hits@5"]), cond
    # The metric must not be degenerate, or the equality above proves nothing.
    assert indexed["conditions"]["full"]["R@1"] == 0.5, "fixture drifted"
    assert direct["conditions"]["full"]["R@1"] == 0.5


def test_the_index_is_reordered_to_the_protocols_own_uid_order(
        tmp_path, monkeypatch):
    """The index's row order is NOT the protocol's, and using it would score
    every query against the wrong asset while still writing a full Table 1.

    `gallery_index.main` writes `sorted(train + test)` -- here
    ["t0","t1","x0","x1"]. `resolve_split(splits, "full")` returns
    `list(train) + list(test)` -- here ["t1","t0","x1","x0"]. A naive "just use
    the index order" implementation fails this test: `targets` is built from
    the protocol's order, so query x1's target column would hold x0's vector.
    """
    assert n15.resolve_split(ORDER_SPLITS, "full") == ["t1", "t0", "x1", "x0"]
    assert ORDER_UIDS == sorted(ORDER_SPLITS["train"] + ORDER_SPLITS["test"])
    assert n15.resolve_split(ORDER_SPLITS, "full") != ORDER_UIDS, (
        "the fixture no longer distinguishes the two orders, so a naive "
        "implementation would pass it")

    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc)

    # 1. the matrix itself comes back in the PROTOCOL's order
    _, gallery = n15.gallery_from_promoted_index(
        "B", ["t1", "t0", "x1", "x0"], {"sha256": CKPT_SHA}, pair, pair)
    assert np.array_equal(gallery, np.stack(
        [ONE_HOT[u] for u in ["t1", "t0", "x1", "x0"]]).astype(np.float64))

    # 2. and the ranks that fall out of it are right. Each query's own asset is
    #    its exact argmax, so every rank is 1 -- and is NOT 1 under index order.
    queries = np.stack([ONE_HOT["x1"], ONE_HOT["x0"]]).astype(np.float64)
    monkeypatch.setattr(n15, "encode_pools",
                        fake_encode_pools(ONE_HOT, queries))
    core, rows = n15.run_protocol(
        "B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean", "cpu", 4,
        "none", 0, 3, False, {"sha256": CKPT_SHA})
    assert core["conditions"]["full"]["R@1"] == 1.0
    assert {r["target_rank"] for r in rows} == {1}


def test_a_reported_protocol_never_calls_the_gallery_encoder(
        tmp_path, monkeypatch):
    """[B3] The guarantee, asserted directly rather than inferred from branches.

    Two independent assertions, because the exception branches are not the
    guarantee: `encode_pools` is asked for NO gallery at all, and the gallery
    tower itself raises if anything invokes it (see `encoder_pair`).
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc)
    seen = []
    queries = np.stack([ONE_HOT["x1"], ONE_HOT["x0"]]).astype(np.float64)
    monkeypatch.setattr(n15, "encode_pools",
                        fake_encode_pools(ONE_HOT, queries, seen))

    n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                     "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})
    assert seen == [None], (
        f"encode_pools was asked to encode {seen}; a reported protocol must "
        "ask it for no gallery at all")


def test_a_reported_protocol_refuses_when_no_index_was_promoted(
        tmp_path, monkeypatch):
    """FAIL CLOSED. A silent re-encode here would put a number in table1.json
    that nothing verified, and nothing in the artifact would say so."""
    pair = encoder_pair()
    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))

    # no registry at all
    monkeypatch.setattr(gallery_index, "PROMOTED_PATH",
                        tmp_path / "nothing_here.json")
    with pytest.raises(FileNotFoundError, match="not found"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})

    # a registry that holds an index, but not for this checkpoint
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc,
                   key_sha=OTHER_CKPT_SHA)
    with pytest.raises(KeyError):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})


def test_an_index_whose_bytes_changed_since_promotion_is_refused(
        tmp_path, monkeypatch):
    """The bytes are re-verified on EVERY read, inside the read.

    A verify in one module and an open in another are two separate opens of one
    path: verification that does not hand back the bytes it verified has not
    verified the bytes that get scored.
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    record = write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                            [ONE_HOT[u] for u in ORDER_UIDS], enc)
    np.savez_compressed(Path(record["uri"]), ids=np.array(ORDER_UIDS),
                        embeddings=np.zeros((4, 4), dtype=np.float32))
    assert hashlib.sha256(
        Path(record["uri"]).read_bytes()).hexdigest() != record["sha256"]

    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))
    with pytest.raises(ValueError, match="hashes to"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})


def test_a_gallery_uid_absent_from_the_index_is_refused(tmp_path, monkeypatch):
    """The index must cover this protocol's gallery.

    The relation is SUPERSET, not equality -- protocol A takes 9,138 test uids
    out of a 45,692-row index -- so the check is "every uid needed is present",
    and it fires on the one that is not.
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    short = [u for u in ORDER_UIDS if u != "x0"]
    write_promoted(tmp_path, monkeypatch, short,
                   [ONE_HOT[u] for u in short], enc)
    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))
    with pytest.raises(ValueError, match="absent from the promoted index"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})

    # and the superset case is NOT refused: a subset of a larger index is what
    # protocol A is.
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc)
    _, gallery = n15.gallery_from_promoted_index(
        "A", ["x1", "x0"], {"sha256": CKPT_SHA}, pair, pair)
    assert gallery.shape == (2, 4)


def test_an_index_built_by_another_encoder_is_refused(tmp_path, monkeypatch):
    """The vectors in the index are not the ones this model would produce.

    Scoring against them would report a model that was never measured, and the
    shapes and counts would all be right.
    """
    pair = encoder_pair()
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS],
                   encoder_sha="c" * 64)
    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))
    with pytest.raises(ValueError, match="gallery encoder loaded here"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})


def test_a_record_that_disagrees_with_its_own_registry_key_is_refused(
        tmp_path, monkeypatch):
    """A key is not a field.

    `gallery_index.main` writes `stage1_checkpoint_sha256` INTO the record on
    exactly that ground, and `verified_index` checks the field is present
    without checking it agrees with the key it was filed under.
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc,
                   record_sha=OTHER_CKPT_SHA)
    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))
    with pytest.raises(ValueError, match="disagree about which"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})


def test_the_untrained_control_re_encodes_and_never_consults_the_registry(
        tmp_path, monkeypatch):
    """`--ckpt-record none` reads no checkpoint record, so it has no Stage 1
    sha256, so there is no promoted index it could be bound to.

    This is one of the two exceptions and it stays. The result says so in
    `gallery_source`, so the exception cannot be mistaken for the rule.
    """
    def refuse(*a, **k):
        raise AssertionError("the untrained control consulted the registry")

    monkeypatch.setattr(gallery_index, "load_promoted_index_for_checkpoint",
                        refuse)
    seen = []
    queries = np.stack([ONE_HOT["x1"], ONE_HOT["x0"]]).astype(np.float64)
    monkeypatch.setattr(n15, "encode_pools",
                        fake_encode_pools(ONE_HOT, queries, seen))

    core, _ = n15.run_protocol(
        "B", REPORTED_FULL, ORDER_SPLITS, None, None, "mean", "cpu", 4,
        "none", 0, 3, True, None)
    assert core["gallery_source"] == "untrained_direct_encode"
    assert seen == [["t1", "t0", "x1", "x0"]]
    assert core["gallery_index_uri"] is None
    assert core["stage1_checkpoint_sha256"] is None
    assert "ENCODED BY THIS RUN" in n15.protocol_caveat(
        REPORTED_FULL, ORDER_SPLITS, core["gallery_source"])


def test_a_development_protocol_keeps_the_direct_encode_and_says_so_in_a_field(
        tmp_path, monkeypatch):
    """C and D are not index-backed, and nothing may read them as if they were.

    The field is what makes that answerable without reading prose, and the
    caveat sentence is derived from the same value so the two cannot disagree.
    """
    def refuse(*a, **k):
        raise AssertionError("a development protocol consulted the registry")

    monkeypatch.setattr(gallery_index, "load_promoted_index_for_checkpoint",
                        refuse)
    seen = []
    queries = np.stack([ONE_HOT["t1"]]).astype(np.float64)
    monkeypatch.setattr(n15, "encode_pools",
                        fake_encode_pools(ONE_HOT, queries, seen))

    core, _ = n15.run_protocol(
        "C", DEV, ORDER_SPLITS, None, None, "mean", "cpu", 4, "none", 0, 3,
        False, {"sha256": CKPT_SHA})
    assert core["gallery_source"] == "direct_dev_encode"
    assert seen == [["t1"]]
    assert core["gallery_index_uri"] is None
    assert core["gallery_index_sha256"] is None
    assert core["gallery_encoder_sha256"] is None
    # the checkpoint is still known, and is still recorded
    assert core["stage1_checkpoint_sha256"] == CKPT_SHA

    caveat = n15.protocol_caveat(DEV, ORDER_SPLITS, core["gallery_source"])
    assert "never reported" in caveat
    assert "not index-backed" in caveat


def test_every_result_carries_the_gallery_provenance_fields(
        tmp_path, monkeypatch):
    """The field set is fixed, and `gallery_source` is never null.

    A result missing one of these cannot answer "was this number index-backed,
    and what cleared it?" without someone opening `gallery_index.json` by hand
    -- which is how a C number gets reported as a Table 1 number.

    ⚠ No count in this test's name. It said "five" and the reviewer's MINOR-2
    made it six, so the name would have been a number that rots.
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    record = write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                            [ONE_HOT[u] for u in ORDER_UIDS], enc)
    queries = np.stack([ONE_HOT["x1"], ONE_HOT["x0"]]).astype(np.float64)
    monkeypatch.setattr(n15, "encode_pools",
                        fake_encode_pools(ONE_HOT, queries))
    core, _ = n15.run_protocol(
        "B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean", "cpu", 4,
        "none", 0, 3, False, {"sha256": CKPT_SHA})

    assert core["gallery_source"] in n15.GALLERY_SOURCES
    assert core["gallery_index_uri"] == record["uri"]
    assert core["gallery_index_sha256"] == record["sha256"]
    assert core["gallery_encoder_sha256"] == enc
    assert core["stage1_checkpoint_sha256"] == CKPT_SHA
    # [REVIEWER MINOR-2] the verdict that cleared this gallery, named in the
    # result rather than reachable only by joining back through the registry
    assert core["gate_record_uri"] == record["gate_record_uri"]
    assert core["gate_record_sha256"] == record["gate_record_sha256"]
    assert all(k in core for k in (
        "gallery_source", "gallery_index_uri", "gallery_index_sha256",
        "gallery_encoder_sha256", "stage1_checkpoint_sha256",
        "gate_record_uri", "gate_record_sha256"))


def test_a_promoted_entry_that_names_no_gate_record_is_refused(
        tmp_path, monkeypatch):
    """[REVIEWER MINOR-2] n15 accepted any entry whose bytes hashed correctly.

    `promote()` cannot reach its write without a terminal G4 record saying
    PASS -- but that fired at PROMOTION time, and this read happens later
    against a JSON file nothing makes immutable. A hand-added key with a
    matching `sha256` satisfies every other check in
    `gallery_from_promoted_index`, which is exactly what this fixture is.

    ⚠ WHAT THIS PINS: that the entry NAMES a gate record. NOT that the gate
    passed, and not that the named record exists or says so. If this test is
    ever strengthened to open the gate record, the guard's own comment about
    presence-versus-verification has to be rewritten with it.
    """
    pair = encoder_pair()
    enc = gallery_index.gallery_encoder_sha256(pair, pair)
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc, gate=False)
    monkeypatch.setattr(n15, "encode_pools", fake_encode_pools(ONE_HOT, None))
    with pytest.raises(ValueError, match="names no gate record"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})

    # An empty string is not a name either: `.get(...)` alone would pass it.
    write_promoted(tmp_path, monkeypatch, ORDER_UIDS,
                   [ONE_HOT[u] for u in ORDER_UIDS], enc, gate=False)
    reg = tmp_path / "gallery_index.json"
    entry = json.loads(reg.read_text())
    entry[CKPT_SHA]["gate_record_sha256"] = ""
    reg.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="names no gate record"):
        n15.run_protocol("B", REPORTED_FULL, ORDER_SPLITS, pair, pair, "mean",
                         "cpu", 4, "none", 0, 3, False, {"sha256": CKPT_SHA})


def test_the_gallery_source_comes_from_reported_not_from_the_protocol_name():
    """Same rule as `protocol_caveat`: a protocol the artifact adds carrying
    `reported: true` must not silently get the development path."""
    unseen = {"query_split": "test", "gallery_split": "full", "reported": True}
    assert n15.gallery_source_for(unseen, False) == "promoted_index"
    assert n15.gallery_source_for(unseen, True) == "untrained_direct_encode"
    assert n15.gallery_source_for(DEV, False) == "direct_dev_encode"
    # absence is not `true`, exactly as `protocol_caveat` treats it
    assert n15.gallery_source_for(
        {"query_split": "test", "gallery_split": "test"}, False) \
        == "direct_dev_encode"
    assert set(n15.GALLERY_SOURCES) == {
        "promoted_index", "direct_dev_encode", "untrained_direct_encode"}


def test_the_three_pool_refusals_still_fire_on_the_index_backed_path(
        tmp_path, monkeypatch):
    """An empty pool, a duplicate gallery uid, and a query whose own asset is
    not in the gallery are refused BEFORE the gallery source is chosen.

    Each of those raises was written against the re-encode path. If the index
    branch had been put in front of them, a reported protocol would have
    skipped all three -- and they are the checks that keep "nothing was
    measured" out of a cell that reads 0.0000.
    """
    def refuse(*a, **k):
        raise AssertionError("the registry was consulted before the pool "
                             "refusals ran")

    monkeypatch.setattr(gallery_index, "load_promoted_index_for_checkpoint",
                        refuse)
    ck = {"sha256": CKPT_SHA}
    empty = {"train": [], "test": [], "dev_val": ["a"], "reported": True}
    with pytest.raises(ValueError, match="empty pool has no"):
        n15.run_protocol("A", {"query_split": "test", "gallery_split": "test",
                               "reported": True},
                         empty, None, None, "mean", "cpu", 4, "none", 0, 8,
                         False, ck)
    dup = {"train": [], "test": ["a", "b", "a"], "dev_val": ["a"]}
    with pytest.raises(ValueError, match="duplicate uid"):
        n15.run_protocol("A", {"query_split": "test", "gallery_split": "test",
                               "reported": True},
                         dup, None, None, "mean", "cpu", 4, "none", 0, 8,
                         False, ck)
    orphan = {"train": ["g"], "test": ["q"], "dev_val": ["g"]}
    with pytest.raises(ValueError, match="absent from the .* gallery"):
        n15.run_protocol("A", {"query_split": "test",
                               "gallery_split": "train", "reported": True},
                         orphan, None, None, "mean", "cpu", 4, "none", 0, 8,
                         False, ck)


def test_the_query_pack_reaches_the_query_pass_and_never_the_gallery_pass(
        monkeypatch, tmp_path):
    """THE SEAM, asserted on the real `encode_pools`, not on a double.

    A gallery built from the query's own second observation is the leak this
    whole change removes, wearing the change's own clothes -- and it would score
    beautifully. `encode_pools` decides the pack from the same expression that
    decides the conditions, so the two cannot come apart; this pins that.

    Also pins the ordering the pre-2026-08-31 code made impossible to get wrong
    by accident and the new code makes possible: the gallery pass must construct
    its dataset with `query_pack=None` even when the caller passed one.
    """
    import numpy as np
    import torch

    import metafind.eval.run_retrieval as rr
    import metafind.train.stage1 as st

    built = []

    class FakeDataset:
        def __init__(self, uids, aggregation, preload=False, query_pack=None):
            built.append((tuple(uids), query_pack))
            self.uids = list(uids)

        def __len__(self):
            return len(self.uids)

        def __getitem__(self, i):
            return {"uid": self.uids[i], "text": np.zeros(4, np.float32),
                    "image": np.zeros(4, np.float32),
                    "pc": np.zeros((2, 6), np.float32)}

    class FakeBackbone:
        model = None

        def encode_pc(self, x):
            return torch.zeros(x.shape[0], 4)

    class FakeModel(torch.nn.Module):
        # nn.Module because `modules_in_eval` walks `.modules()` to put the
        # towers in eval -- a plain object silently would not have been put in
        # eval either, so the base class is part of what is under test.
        # ones, not zeros: `normalize_for_scoring` refuses a zero-norm
        # embedding, and that refusal is a real guard worth not defeating.
        def query(self, embeds, present=None):
            return torch.ones(embeds["text"].shape[0], 4)

        def gallery(self, embeds):
            return torch.ones(embeds["text"].shape[0], 4)

    monkeypatch.setattr(st, "Stage1Dataset", FakeDataset)
    sentinel = object()
    rr.encode_pools(FakeBackbone(), FakeModel(), ["q0", "q1"], ["g0", "g1"],
                    "mean", "cpu", 2, sentinel)

    by_uids = dict(built)
    assert by_uids[("g0", "g1")] is None, (
        "the gallery pass was given the query pack: its embeddings would come "
        "from the query's own observation of each asset")
    assert by_uids[("q0", "q1")] is sentinel, (
        "the query pack never reached the query pass -- it would be configured "
        "and silently unread, which is how the field became decorative before")
