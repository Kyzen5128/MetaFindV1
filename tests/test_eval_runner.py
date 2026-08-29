"""n15 -- the evaluation runner that did not exist until 2026-08-30.

`eval_protocols.json` carried `reported: true` on `A_test_gallery` and
`B_full_gallery` for months with **no program reading it**. Every number this
project holds is protocol C, hardcoded in `stage1.evaluate_dev_val`. These tests
pin the things that would let that happen again, or let the new runner report a
number that is not what it says it is.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from metafind.eval import run_retrieval as n15
from metafind.eval.retrieval import rank_of_target


def unit(rng, n, d=6):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------- the scorer

@pytest.mark.parametrize("block", [1, 2, 3, 7, 10_000])
def test_the_blocked_scorer_equals_the_reference_scorer(block):
    """A blocked scorer that is off by one, only sometimes, survives review.

    It nearly did. `own` comes from a row-wise product and `sim` from a GEMM,
    and the two disagree in the last bit -- so comparing the target against
    itself registered as STRICTLY HIGHER rather than tied and every rank came
    out one too large. Measured, not hypothetical: this test caught it.
    """
    rng = np.random.default_rng(7)
    q, g = unit(rng, 11), unit(rng, 29)
    t = rng.integers(0, 29, size=11)
    r = n15.score_streaming(q, g, t, block=block)
    assert np.array_equal(r["rank"], rank_of_target(q @ g.T, t))


@pytest.mark.parametrize("block", [2, 5, 10_000])
def test_the_top_two_survive_being_split_across_blocks(block):
    """The obvious merge (`if b1 > top1: top2 = top1`) is wrong.

    It drops the block's own runner-up whenever that runner-up also beats the
    incumbent leader -- which happens exactly when both of the true top two live
    in the same block.
    """
    rng = np.random.default_rng(3)
    q, g = unit(rng, 6), unit(rng, 23)
    t = rng.integers(0, 23, size=6)
    r = n15.score_streaming(q, g, t, block=block)
    srt = np.sort(q @ g.T, axis=1)
    assert np.allclose(r["top1_score"], srt[:, -1])
    assert np.allclose(r["top2_score"], srt[:, -2])


def test_a_collapsed_model_ranks_last_not_first():
    """Ties count AGAINST the model.

    ⚠ This fixture is `np.ones`-shaped, and on its own it is a COULD-NOT-FAIL
    test: with such values the row-wise product and the GEMM agree to the last
    bit, so it passed for hours over the blocker below. Kept for readability;
    the test that does the work is the next one.
    """
    g = np.tile([[1.0, 0, 0, 0, 0, 0]], (9, 1))
    r = n15.score_streaming(g.copy(), g, np.arange(9), block=2)
    assert (r["rank"] == 9).all()
    assert (r["tie_count"] == 8).all()


@pytest.mark.parametrize("block", [1, 7, 8, 10_000])
def test_a_collapsed_model_on_data_that_splits_the_two_arithmetic_paths(block):
    """[ULIP2 REVIEWER 2026-08-30, BLOCKER] The defect n15 was built to catch,
    inside n15.

    `own` came from a row-wise product and `sim` from a GEMM. On a collapsed
    gallery the two differ by one ULP, `own` is then strictly greater than every
    column, and the run reports **rank 1 and tie_count 0** where the truth is
    rank 30 and tie_count 29. A totally collapsed model scored as perfect
    retrieval.

    MEASURED before the fix: 20 of 200 random collapsed galleries disagreed with
    the reference scorer. Data-dependent, so 90% of trials looked fine.

    It also disabled the negative control: with `higher = 0` and `tied = 0` the
    answer does not depend on which column the target is, so `shuffle_targets`
    scored identically to the real run -- the one control that was supposed to
    contradict a saturated `full`.

    This fixture uses random values, not `np.ones`, precisely because
    `np.ones` makes the two paths agree.
    """
    rng = np.random.default_rng(0)
    for _ in range(25):
        g = np.tile(rng.normal(size=(1, 5)), (30, 1))
        q = g.copy()
        t = rng.integers(0, 30, size=30)
        r = n15.score_streaming(q, g, t, block=block)
        assert np.array_equal(r["rank"], rank_of_target(q @ g.T, t))
        assert (r["rank"] == 30).all(), "a collapsed gallery must rank last"
        assert (r["tie_count"] == 29).all()


def test_the_shuffled_control_still_responds_when_the_model_has_collapsed():
    """The control has to work in the case it exists for.

    Under the blocker above, a collapsed model scored the same shuffled as
    unshuffled -- so the control agreed with a wrong result instead of
    contradicting it.
    """
    rng = np.random.default_rng(4)
    g = np.tile(rng.normal(size=(1, 8)), (50, 1))
    q = g.copy()
    true_t = np.arange(50)
    shuffled, _ = n15.apply_control("shuffle_targets", true_t, 50, seed=1)
    real = (n15.score_streaming(q, g, true_t, block=16)["rank"] == 1).mean()
    ctrl = (n15.score_streaming(q, g, shuffled, block=16)["rank"] == 1).mean()
    assert real == 0.0, "a collapsed model retrieves nothing"
    assert ctrl == 0.0


def test_the_target_does_not_inflate_its_own_margin():
    """`hardest_non_target_score` is the best score that is NOT the answer.

    If the target leaked into it, `signed_target_margin` would be <= 0 for every
    query that ranked first -- the exact opposite of what it means.
    """
    rng = np.random.default_rng(11)
    q, g = unit(rng, 8), unit(rng, 30)
    t = rng.integers(0, 30, size=8)
    r = n15.score_streaming(q, g, t, block=4)
    sim = q @ g.T
    off = sim.copy()
    off[np.arange(8), t] = -np.inf
    assert np.allclose(r["hardest_non_target_score"], off.max(axis=1))
    assert np.allclose(r["signed_target_margin"],
                       sim[np.arange(8), t] - off.max(axis=1))


def test_the_spread_statistics_exclude_the_target(): 
    rng = np.random.default_rng(5)
    q, g = unit(rng, 5), unit(rng, 17)
    t = rng.integers(0, 17, size=5)
    r = n15.score_streaming(q, g, t, block=3)
    sim = q @ g.T
    ref = np.array([np.delete(sim[i], t[i]).std() for i in range(5)])
    assert np.allclose(r["off_target_std"], ref)


def test_embedding_health_sees_a_collapsed_space():
    """`full` R@1 = 1.0000 has an understood mechanism, which is an INFERENCE.

    Effective rank is one of the measurements that can contradict it: a gallery
    whose assets all embed to one direction has effective rank near 1 whatever
    its R@1 says.
    """
    rng = np.random.default_rng(1)
    healthy = n15.embedding_health(unit(rng, 200, 16))
    collapsed = n15.embedding_health(
        unit(rng, 200, 16) * 1e-6 + np.array([[1.0] + [0] * 15]))
    assert healthy["effective_rank"] > 8
    assert collapsed["effective_rank"] < 2, (
        "a gallery of nearly identical vectors must not look full-rank")
    # ⚠ The first version of this test asserted the CENTRED rank and passed at
    # 15.4/16 on this very fixture. Centring subtracts the mean direction, which
    # under collapse is the only direction there is, so it measured the leftover
    # jitter. Both are reported now; the uncentred one is the collapse detector.
    assert collapsed["effective_rank_centred"] > 8
    assert collapsed["per_dim_std_max"] < 1e-5


# ------------------------------------------------------------- the sealed split

SEALED = {"query_split": "test", "gallery_split": "test"}
FULL = {"query_split": "test", "gallery_split": "full"}
DEV = {"query_split": "dev_val", "gallery_split": "dev_val"}


QUERY_FULL = {"query_split": "full", "gallery_split": "dev_val"}


@pytest.mark.parametrize("proto", [SEALED, FULL, QUERY_FULL])
def test_a_sealed_protocol_is_refused_without_an_explicit_unseal(proto):
    """[CODEX 2026-08-30] A sweep that reads the test split has spent it.

    No later number from this corpus is a held-out number again, and nothing in
    any artifact would show it. The guard is in code because a procedural rule
    survives exactly until someone is in a hurry.
    """
    # QUERY_FULL is [ULIP2 REVIEWER 2026-08-30]: `full` was checked on the
    # gallery side only, so `query_split: "full"` walked past the guard -- and
    # `full` is train + test by definition. Reachable exactly because protocols
    # are read rather than named.
    with pytest.raises(SystemExit, match="sealed test split"):
        n15.check_seal("A_test_gallery", proto, unsealed=False)


def test_the_development_protocol_needs_no_unseal():
    """The guard must not make ordinary development impossible.

    A guard that blocks everything is uninstalled within a day.
    """
    assert n15.check_seal("C_dev_selection", DEV, unsealed=False) is False


def test_breaking_the_seal_is_recorded_not_merely_permitted():
    assert n15.check_seal("A_test_gallery", SEALED, unsealed=True) is True


# --------------------------------------------------- protocols are read, not named

def test_a_renamed_protocol_still_runs(tmp_path):
    """The first version of this evaluation put the protocol IN the code.

    That is why `eval_protocols.json` ended up with no consumer at all. Nothing
    here may key off `A_test_gallery` as a string.
    """
    p = tmp_path / "eval_protocols.json"
    p.write_text(json.dumps({"whatever_we_call_it_next": {
        "query_split": "test", "gallery_split": "test", "gallery_size": 3}}))
    got = n15.load_protocols(p)
    assert "whatever_we_call_it_next" in got
    with pytest.raises(SystemExit, match="sealed"):
        n15.check_seal("whatever_we_call_it_next",
                       got["whatever_we_call_it_next"], unsealed=False)


def test_full_is_train_plus_test_not_a_hardcoded_count():
    """`splits.py:17` still says "9,211 versus 46,052" in prose.

    The real numbers are 9,138 and 45,692. Deriving `full` from the split lists
    is what keeps this module from inheriting that staleness.
    """
    splits = {"train": ["a", "b"], "test": ["c"], "dev_val": ["a"]}
    assert n15.resolve_split(splits, "full") == ["a", "b", "c"]
    assert n15.resolve_split(splits, "test") == ["c"]
    with pytest.raises(ValueError, match="splits.json does not have"):
        n15.resolve_split(splits, "no_such_split")


# ---------------------------------------------------------------- the controls

def test_shuffling_the_targets_collapses_the_metric_to_chance():
    """The measurement that turns the 1.0000 mechanism into a finding.

    Every checkpoint reports `full` R@1 = 1.0000 against the paper's 0.517. A
    mechanism is understood, but a mechanism being real does not make it the
    only cause. If a metric does not collapse when each query is scored against
    somebody else's asset, it was never measuring retrieval.
    """
    rng = np.random.default_rng(2)
    q = unit(rng, 300, 8)
    g = q.copy()                       # perfect retrieval, R@1 = 1.0
    true_t = np.arange(300)
    assert (n15.score_streaming(q, g, true_t, block=64)["rank"] == 1).mean() == 1.0

    shuffled, used = n15.apply_control("shuffle_targets", true_t, 300, seed=1)
    assert used == "shuffle_targets"
    r1 = (n15.score_streaming(q, g, shuffled, block=64)["rank"] == 1).mean()
    assert r1 < 0.05, f"shuffled control scored {r1}, which is not chance"


def test_the_control_that_is_not_implemented_says_so():
    """`exclude_target` was specified and is deliberately absent.

    It would recompute `hardest_non_target_score`, which every normal run
    already reports per query, and call the result an "R@1" whose target is not
    in the gallery -- a probe that cannot return a positive. Refusing loudly is
    the difference between a decision and an omission.
    """
    with pytest.raises(ValueError, match="unknown control"):
        n15.apply_control("exclude_target", np.arange(3), 3, seed=0)
    assert "exclude_target" in n15.apply_control.__doc__
    assert "NOT implemented" in n15.apply_control.__doc__


def test_a_one_entry_gallery_writes_null_not_infinity():
    """A sidecar is a machine-read file, and `Infinity` is not valid JSON.

    With one gallery entry there is no non-target, so
    `hardest_non_target_score` is -inf and the margin is +inf. `json.dumps`
    writes them happily and a strict parser rejects the whole file.
    """
    g = np.array([[1.0, 0.0]])
    r = n15.score_streaming(g.copy(), g, np.array([0]), block=4)
    assert not np.isfinite(r["signed_target_margin"][0])
    assert n15.jsonable(r["signed_target_margin"][0]) is None
    json.dumps({"m": n15.jsonable(r["signed_target_margin"][0])})
    assert n15.quantiles(r["signed_target_margin"]) == {}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ng", [999, 4569])
def test_a_collapsed_model_cannot_score_at_production_shape(dtype, ng):
    """The question that actually matters, asked in the shape we actually run.

    ⚠ The numbers first recorded for this -- 20/200 and 42/200 "misses" -- were
    measured at d = 4..40 with UNNORMALISED embeddings, a regime this project
    never uses, and then extrapolated. Production is d = 1280, L2-normalised,
    and galleries of 4,569 upward. Re-measured there:

        float32   0 trials with R@1 > 0, out of 1,120, at every gallery size
        float64   0 at every real gallery size (fails only at ng <= 13)

    The right question is "can a collapsed model score?", not "is the rank
    exactly n_gallery?" -- a rank of 4,568 instead of 4,569 is a diagnostic
    being off by one, not a model being reported as good, and conflating the two
    is what produced the retracted numbers.
    """
    rng = np.random.default_rng(11)
    for _ in range(6):
        v = rng.normal(size=(1, 1280))
        v = (v / np.linalg.norm(v)).astype(dtype)
        g = np.tile(v, (ng, 1))
        q = g[:16].copy()
        t = rng.integers(0, ng, size=16)
        # float32 goes through the reference scorer: `score_streaming` now
        # REFUSES float32 (see test_score_streaming_refuses_float32), and the
        # question here is about the metric at production shape, not about which
        # entry point is used to ask it.
        ranks = (n15.score_streaming(q, g, t, block=4096)["rank"]
                 if dtype is np.float64 else rank_of_target(q @ g.T, t))
        assert (ranks <= 1).mean() == 0.0, (
            "a totally collapsed model scored a hit at production shape")


def test_float64_makes_the_collapse_diagnostics_block_independent():
    """Why `encode_pools` scores in float64.

    R@1 is safe in either dtype. `tie_count` is not: at production shape it
    moved with the caller's block size in 7-9 of 12 float32 trials, and in 0 of
    12 float64 trials. `tie_count` is the diagnostic added to detect collapse,
    so a performance knob must not move it.
    """
    rng = np.random.default_rng(13)
    v = rng.normal(size=(1, 1280))
    v = (v / np.linalg.norm(v)).astype(np.float64)
    g = np.tile(v, (4569, 1))
    q = g[:8].copy()
    t = rng.integers(0, 4569, size=8)
    seen = {tuple(n15.score_streaming(q, g, t, block=b)["tie_count"].tolist())
            for b in (512, 1000, 4096, 4569)}
    assert len(seen) == 1, "tie_count moved with the block size in float64"


def test_the_reference_scorers_tie_test_does_not_catch_exact_collapse():
    """The docstring claim that was false on the production path.

    `rank_of_target` promised it "catches EXACT collapse". Measured on 200
    fully collapsed galleries it missed 20 of them, because a single BLAS `gemm`
    does not return the same last bit for every output column -- so byte-
    identical gallery rows do not compare equal.

    This test asserts the LIMITATION, not the fix. It is here so the corrected
    docstring cannot drift back, and so that if a tolerance is ever adopted this
    test fails loudly and has to be rewritten as the statement of a decision.

    ⚠ SCOPE: this is the TOY regime (small d, unnormalised). At production
    shape the exact rank is right and R@1 is 0 -- see the two tests above. Kept
    because it is the measurement the corrected docstring cites, and it must
    stay attached to the conditions it was taken under.
    """
    rng = np.random.default_rng(0)
    missed = 0
    for _ in range(200):
        d = int(rng.integers(4, 40))
        ng = int(rng.integers(5, 40))
        g = np.tile(rng.normal(size=(1, d)), (ng, 1))
        t = rng.integers(0, ng, size=ng)
        if not (rank_of_target(g.copy() @ g.T, t) == ng).all():
            missed += 1
    assert missed > 0, (
        "bit-equality now catches every collapsed gallery. If a tolerance was "
        "adopted, rewrite this test to state that decision; if not, the "
        "measurement that produced the corrected docstring no longer holds.")


def test_a_single_gemm_is_not_bit_reproducible_across_identical_rows():
    """The false premise underneath three rounds of fixes.

    "Same arithmetic path implies bit-equality" was written into a comment by
    the engineer and into a warning by the reviewer, and neither measured it.
    It is false: OpenBLAS uses different accumulation orders for different
    output columns.
    """
    rng = np.random.default_rng(0)
    inconsistent = 0
    for _ in range(400):
        d = int(rng.integers(3, 12))
        ng = int(rng.integers(5, 40))
        g = np.tile(rng.normal(size=(1, d)), (ng, 1))
        for width in (3, 7, 16, ng):
            if len(set((g[:1] @ g[:width].T)[0].tolist())) > 1:
                inconsistent += 1
                break
    assert inconsistent > 0, (
        "identical gallery rows now hash to one value in a single gemm. The "
        "BLAS or numpy changed; every argument that rests on 'same arithmetic "
        "path implies bit-equality' must be re-measured before it is trusted.")



def test_score_streaming_refuses_float32():
    """[ULIP2 REVIEWER 2026-08-30, MAJOR] Block-independence must be ENFORCED,
    not merely true today.

    It held only because `encode_pools` happens to return float64. Nothing in
    `score_streaming` required it, so a test, a future caller, or a "save
    memory" refactor would bring the defect back silently -- and `tie_count` is
    the diagnostic added specifically to detect collapse.

    MEASURED before the guard: with float32 a collapsed 4,569 gallery reported
    tie_count 4568 at some block sizes and 4567 at others. THIS test is what
    closes R2's MAJOR; "nobody calls it that way" is not a closure.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(1, 1280))
    v = (v / np.linalg.norm(v)).astype(np.float32)
    g = np.tile(v, (64, 1))
    with pytest.raises(ValueError, match="needs float64"):
        n15.score_streaming(g[:4].copy(), g, np.arange(4), block=16)


def test_score_streaming_accepts_what_normalize_for_scoring_produces():
    """The guard must not block the production path it exists to protect."""
    from metafind.eval.retrieval import normalize_for_scoring
    rng = np.random.default_rng(1)
    g = normalize_for_scoring(rng.normal(size=(40, 32)))
    q = normalize_for_scoring(rng.normal(size=(5, 32)))
    assert n15.score_streaming(q, g, np.arange(5), block=8)["rank"].shape == (5,)
