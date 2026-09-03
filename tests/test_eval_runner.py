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
    """The control does collapse to chance. That is ALL this test shows.

    ⚠ [ULIP2 REVIEWER 2026-08-30, BLOCKER] This test used to be described as
    "the measurement that turns the 1.0000 mechanism into a finding". It is not,
    and this very fixture is why: `g = q.copy()` IS the "both towers see
    identical input" defect, in its total form, and the control passes anyway.
    The non-discrimination is asserted directly in
    `test_the_shuffled_control_is_a_wiring_check_not_a_discriminator` below.

    What survives: if the metric does NOT collapse when each query is scored
    against somebody else's asset, it was never measuring retrieval. One
    direction only.
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


def test_the_shuffled_control_is_a_wiring_check_not_a_discriminator():
    """[ULIP2 REVIEWER 2026-08-30, BLOCKER] The green light that is not a result.

    `apply_control` claimed `shuffle_targets` tests the "both towers see
    identical input" hypothesis. Both halves are asserted here so the claim
    cannot come back as prose.

    HALF 1 -- no discriminative power. The fixture is the defect at full
    strength: the query stack and the gallery stack are the same array, so every
    query's own row is its exact argmax. Real R@1 = 1.0, shuffled R@1 ~ 1/n.
    The control passes hardest precisely when the defect is total, because
    permuting the target column moves the answer to a random column whatever
    produced the similarity.

    HALF 2 -- what it CAN detect. The rank must actually depend on which column
    is named as the target. When it did not -- `own` from a row-wise product,
    the comparisons from a GEMM, `higher` and `tied` both 0 -- the shuffled run
    scored identically to the real one, and that is the bug this control caught.
    """
    rng = np.random.default_rng(21)
    q = unit(rng, 200, 12)
    g = q.copy()                        # the defect, literally: identical input
    true_t = np.arange(200)
    shuffled, _ = n15.apply_control("shuffle_targets", true_t, 200, seed=3)

    real = (n15.score_streaming(q, g, true_t, block=32)["rank"] == 1).mean()
    ctrl = (n15.score_streaming(q, g, shuffled, block=32)["rank"] == 1).mean()
    assert real == 1.0
    assert ctrl < 0.05, (
        "if this ever fails the control has become discriminative and the "
        "docstring in apply_control has to be rewritten as a measurement")

    # HALF 2: the rank is a function of the target column, per query, not a
    # constant the shuffle cannot move.
    ranks_real = n15.score_streaming(q, g, true_t, block=32)["rank"]
    ranks_ctrl = n15.score_streaming(q, g, shuffled, block=32)["rank"]
    assert not np.array_equal(ranks_real, ranks_ctrl), (
        "the shuffled and unshuffled ranks are identical -- the rank arithmetic "
        "does not depend on `targets`, which is the one defect class this "
        "control exists to catch")

    # And the docstring must keep saying so. Whitespace-normalised, so a
    # rewrap cannot silently disarm the assertion.
    doc = " ".join(n15.apply_control.__doc__.split())
    assert "WIRING CHECK, not a discriminator" in doc
    assert "CANNOT detect (no discriminative power, all three)" in doc


def test_off_target_entropy_excludes_the_target_like_its_name_says():
    """[ULIP2 REVIEWER 2026-08-30, MINOR] The mask that covered one of two sums.

    `off_sum` / `off_sq` read the masked block; the entropy accumulators read
    the UNMASKED `sim`. So `off_target_std` excluded the answer and
    `off_target_entropy` -- same prefix, same comment, adjacent lines -- did not.

    The reference is the entropy of the softmax over the off-target columns
    only, computed the slow explicit way.
    """
    rng = np.random.default_rng(5)
    q, g = unit(rng, 5, 9), unit(rng, 17, 9)
    t = rng.integers(0, 17, size=5)
    r = n15.score_streaming(q, g, t, block=3)
    sim = q @ g.T

    def entropy(x):
        m = x.max()
        z = np.exp(x - m).sum()
        return np.log(z) + m - (x * np.exp(x - m)).sum() / z

    excl = np.array([entropy(np.delete(sim[i], t[i])) for i in range(5)])
    incl = np.array([entropy(sim[i]) for i in range(5)])
    assert np.allclose(r["off_target_entropy"], excl, atol=1e-12)
    # The two references must actually differ, or the test could not fail.
    assert not np.allclose(excl, incl, atol=1e-6)


def test_a_one_entry_gallery_does_not_take_the_log_of_zero():
    """Masking the target out of Z means a 1-row gallery has an EMPTY Z.

    `off_target_std` already clamps the same degenerate case with
    `max(n_off, 1)`. Without the matching clamp the entropy is -inf plus a
    RuntimeWarning, and `float(-inf)` is `-Infinity` in the sidecar, which is
    not valid JSON.
    """
    g = np.array([[1.0, 0.0]])
    with np.errstate(all="raise"):
        r = n15.score_streaming(g.copy(), g, np.array([0]), block=4)
    assert np.isfinite(r["off_target_entropy"][0])


# ------------------------------------------------ the caveat comes from a field

def test_the_caveat_is_read_from_reported_not_from_the_protocol_name():
    """[ULIP2 REVIEWER 2026-08-30, MAJOR] The name lookup this module forbids.

    `main` chose the caveat with
    `{"A_test_gallery": ..., "B_full_gallery": ...}.get(name, <development>)`,
    twelve lines from the docstring that says protocols are read and never
    named. A protocol the artifact adds -- the expected case, not an exotic one
    -- got "selects checkpoints, never reported" printed beside a number that is
    reported.

    `reported` had no consumer on the evaluation path at all before this. The
    only reader in the repo was a `print` in `splits.py`, which writes the file.
    """
    splits = {"train": ["t1", "t2", "t3"], "test": ["x", "y"], "dev_val": ["t1"]}
    unseen = {"query_split": "test", "gallery_split": "full", "reported": True}
    cav = n15.protocol_caveat(unseen, splits)
    assert "never reported" not in cav
    assert "[U-09]" in cav
    # the distractor count is COUNTED, not a literal carried forward
    assert "3 training assets" in cav

    dev = {"query_split": "dev_val", "gallery_split": "dev_val", "reported": False}
    assert "never reported" in n15.protocol_caveat(dev, splits)

    # A reported protocol whose gallery holds no training assets says so by
    # omission rather than by printing "0 training assets".
    a_like = {"query_split": "test", "gallery_split": "test", "reported": True}
    assert "training assets" not in n15.protocol_caveat(a_like, splits)
    assert "this project's assumption" in n15.protocol_caveat(a_like, splits)


def test_a_protocol_with_no_reported_field_is_treated_as_not_reported():
    """Absence is not `true`. `.get("reported", False)`, deliberately.

    An artifact written by an older `splits.py` has no `reported` key, and
    defaulting a missing key to "reported" would attach a paper caveat to a
    number nobody claims is a paper number.
    """
    splits = {"train": ["a"], "test": ["b"], "dev_val": ["a"]}
    assert "never reported" in n15.protocol_caveat(
        {"query_split": "test", "gallery_split": "test"}, splits)


# ------------------------------------------- a pool that cannot be scored

def test_a_duplicated_gallery_uid_is_refused_not_counted():
    """[ULIP2 REVIEWER 2026-08-30, MINOR] It was reported and then scored anyway.

    `col = {u: i for i, u in enumerate(gallery_uids)}` keeps the LAST index of a
    repeated uid. The earlier copy is then a gallery row bit-identical to the
    target it duplicates, so it ties with it -- and ties count against the model
    -- so that query's rank is 2 or worse for a reason that has nothing to do
    with the model. R@1 was depressed by a data defect whose only trace was an
    integer field.
    """
    splits = {"train": [], "test": ["a", "b", "a"], "dev_val": ["a"]}
    proto = {"query_split": "test", "gallery_split": "test"}
    with pytest.raises(ValueError, match="duplicate uid"):
        n15.run_protocol("dup", proto, splits, None, None, "mean", "cpu", 4,
                         "none", 0, 8)


def test_an_empty_pool_is_refused_rather_than_written_out_as_zero():
    """[ULIP2 REVIEWER 2026-08-30, MINOR] "Nothing measured" was written as 0.0.

    `R@1: float((ranks <= 1).mean()) if ranks.size else 0.0` produced a complete,
    normal-looking Table 1 row for a protocol with no queries.
    `stage1.evaluate_dev_val` returns `{}` on an empty pool for exactly this
    reason -- to keep "no measurement" distinguishable from "a measurement of
    zero". n15 wrote out the indistinguishable version.
    """
    splits = {"train": [], "test": [], "dev_val": ["a"]}
    with pytest.raises(ValueError, match="empty pool has no"):
        n15.run_protocol("empty_q", {"query_split": "test",
                                     "gallery_split": "dev_val"},
                         splits, None, None, "mean", "cpu", 4, "none", 0, 8)
    with pytest.raises(ValueError, match="empty pool has no"):
        n15.run_protocol("empty_g", {"query_split": "dev_val",
                                     "gallery_split": "test"},
                         splits, None, None, "mean", "cpu", 4, "none", 0, 8)


# ------------------------------------------------- the untrained control exists

def test_the_untrained_control_is_a_code_path_not_a_docstring_suggestion(monkeypatch):
    """[ULIP2 REVIEWER 2026-08-30, MAJOR] "needs no code" was not runnable.

    `apply_control` said the initialisation control needed no code: point
    `--ckpt-record` at an untrained checkpoint. But `main` calls
    `load_checkpoint_record`, which verifies the record's sha256 against the
    weights, then calls `load_stage1_checkpoint` unconditionally -- and no tool
    in this repo produces an untrained checkpoint to point at. The control was
    unreachable.

    Both guards fire before any import of torch, the backbone or the trainer, so
    this test costs nothing and touches no GPU.
    """
    import sys
    base = ["run_retrieval", "--protocol", "C_dev_selection"]

    monkeypatch.setattr(sys, "argv", base + ["--ckpt-record", "none"])
    with pytest.raises(SystemExit, match="needs --init-seed"):
        n15.main()

    monkeypatch.setattr(sys, "argv", base + ["--init-seed", "0"])
    with pytest.raises(SystemExit, match="only has meaning"):
        n15.main()


def test_untrained_does_not_mean_unpretrained():
    """The sentence this control turns on, pinned so it cannot be trimmed.

    Under `--ckpt-record none` only the two fusion towers are random. The point
    encoder is still ULIP-2's pretrained PointBERT plus `pc_projection`
    (`ulip_backbone.py` loads them from the ULIP-2 checkpoint regardless), and
    text/image still go through pretrained OpenCLIP ViT-bigG-14. Reporting this
    run as "an untrained model" one step wider than that is the error the run
    exists to avoid.
    """
    import inspect
    src = inspect.getsource(n15.main)
    assert "NOT a zero-pretraining baseline" in src
    assert "ULIP-2's pretrained PointBERT" in src


def test_the_untrained_run_loads_no_stage1_weights_and_says_so(tmp_path, monkeypatch):
    """The whole of MAJOR-1, end to end through `main`, with no GPU and no encode.

    `apply_control` used to say the initialisation control "needs no code". It
    needed code: `main` called `load_checkpoint_record` -- which verifies a
    sha256 against real weights -- and then `load_stage1_checkpoint`
    unconditionally, and nothing in this repo produces an untrained checkpoint
    to point either of them at.

    Asserted here, in order: neither loader is reached; the towers are drawn from
    `--init-seed`; the provenance says `ckpt_record: none`, `untrained: true`
    and the seed; the caveat says "untrained" WITHOUT saying "unpretrained"; and
    the output directory name carries the word.

    Everything expensive is stubbed. The point of the test is the CONTROL FLOW
    of `main`, which is where the defect was.
    """
    import sys
    import torch
    from metafind import paths, runlog
    from metafind.train import gallery_index, stage1
    from metafind.models import ulip_backbone

    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path)
    monkeypatch.setattr(paths, "LOGS", tmp_path / "logs")
    (tmp_path / "eval_protocols.json").write_text(json.dumps(
        {"C_dev_selection": {"query_split": "dev_val",
                             "gallery_split": "dev_val", "reported": False}}))
    (tmp_path / "splits.json").write_text(json.dumps(
        {"object": {"train": ["t"], "test": ["x"], "dev_val": ["d"]}}))

    called = []
    # Returns a plausible record, so a regression is caught by the `called`
    # assertion below rather than by an incidental crash.
    monkeypatch.setattr(gallery_index, "load_checkpoint_record",
                        lambda *a, **k: (called.append("record"),
                                         {"uri": "/nonexistent.pt"})[1])
    monkeypatch.setattr(stage1, "load_stage1_checkpoint",
                        lambda *a, **k: called.append("weights"))
    monkeypatch.setattr(stage1, "load_protocols",
                        lambda *a, **k: ({"image_aggregation": "mean"}, {}, {}))
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", lambda cfg: object())

    drawn = []

    def fake_build_model(encoding, training, hyperparameters):
        # Whatever the seed produced, captured through the SAME global RNG the
        # real `build_model` draws its Linear weights from.
        drawn.append(torch.rand(3).tolist())

        class M:
            def to(self, device):
                return self
        return M(), object()

    monkeypatch.setattr(stage1, "build_model", fake_build_model)

    captured = {}

    def fake_run_protocol(name, protocol, splits, *a, **k):
        captured["splits"] = splits
        captured["tail"] = a
        return ({"protocol": name, "n_query": 1, "n_gallery": 1,
                 "conditions": {"full": {"R@1": 0.0, "R@5": 0.0,
                                         "diagnostics": {"signed_target_margin": {}}}},
                 # The gallery-provenance block. Returned here so the
                 # assertion below is about the table1.json PROJECTION -- the
                 # only thing standing between run_protocol's result and the
                 # file -- rather than about run_protocol, which
                 # test_eval_retrieval.py already covers.
                 "gallery_source": "promoted_index",
                 "gallery_index_uri": "/tmp/gi.npz",
                 "gallery_index_sha256": "1" * 64,
                 "gallery_encoder_sha256": "2" * 64,
                 "stage1_checkpoint_sha256": "3" * 64,
                 "gate_record_uri": "/tmp/G4_gallery_freeze.yaml",
                 "gate_record_sha256": "4" * 64,
                 "embedding_health": {}}, [])

    monkeypatch.setattr(n15, "run_protocol", fake_run_protocol)

    monkeypatch.setattr(sys, "argv", ["run_retrieval", "--ckpt-record", "none",
                                      "--init-seed", "1234", "--device", "cpu"])
    assert n15.main() == 0

    assert called == [], (
        f"the untrained path reached {called} -- it must read no checkpoint "
        "record and load no Stage 1 weights")

    out = next((tmp_path / "eval").iterdir())
    assert "untrained" in out.name and "seed1234" in out.name, out.name

    prov = json.loads((out / "table1.json").read_text())["provenance"]
    assert prov["ckpt_record"] == "none"
    assert prov["untrained"] is True
    assert prov["init_seed"] == 1234
    assert all(v is None for v in prov["checkpoint"].values()), prov["checkpoint"]

    cav = prov["untrained_caveat"]
    assert "UNTRAINED" in cav
    assert "NOT a zero-pretraining baseline" in cav
    assert "ULIP-2" in cav and "OpenCLIP" in cav

    # The seed reaches the RNG that draws the towers, and a different seed draws
    # differently -- otherwise --init-seed would be provenance for nothing.
    torch.manual_seed(1234)
    assert drawn[0] == torch.rand(3).tolist()
    torch.manual_seed(4321)
    assert drawn[0] != torch.rand(3).tolist()

    # And the caveat for a `reported: false` protocol still came from the field.
    proto = json.loads((out / "table1.json").read_text())["protocols"]
    assert "never reported" in proto["C_dev_selection"]["caveat"]

    # `main` hands run_protocol the two arguments that decide the gallery
    # SOURCE, in that order. Asserted here because this is the only test that
    # drives `main`, and a swapped pair would send a reported protocol down the
    # untrained branch -- where it would encode its own gallery and say so in a
    # field nobody was looking at yet.
    #
    # Sliced from the END MINUS the two arguments added after this test:
    # the query pack (2026-08-31) and the degraded-render exclusion set
    # (2026-09-03). It broke loudly both times, which is the right failure --
    # this assertion is about a POSITION, so each new positional gets the slice
    # re-anchored ON PURPOSE rather than the assertion widened to `[-1:]`, which
    # would stop noticing.
    assert captured["tail"][-4:] == (True, {}, None, None), captured["tail"][-4:]

    # [REVIEWER MINOR-2] The gallery-provenance block reaches table1.json ON
    # DISK. `table1.json` drops `embedding_health` and the per-condition
    # `diagnostics`; every other key rides through, and a reader of the
    # reported artifact can name the index AND the verdict that cleared it
    # without opening `gallery_index.json` by hand.
    cell = proto["C_dev_selection"]
    assert cell["gallery_source"] == "promoted_index"
    assert cell["gallery_index_uri"] == "/tmp/gi.npz"
    assert cell["gallery_index_sha256"] == "1" * 64
    assert cell["gallery_encoder_sha256"] == "2" * 64
    assert cell["stage1_checkpoint_sha256"] == "3" * 64
    assert cell["gate_record_uri"] == "/tmp/G4_gallery_freeze.yaml"
    assert cell["gate_record_sha256"] == "4" * 64
    # and the projection really does drop what it is supposed to drop, or the
    # assertions above would pass for a projection that drops nothing
    assert "embedding_health" not in cell
    assert "diagnostics" not in cell["conditions"]["full"]


def test_the_degraded_render_exclusion_does_not_read_as_a_stale_gallery():
    """[BLOCKER, ULIP2 REVIEWER 2026-09-03] The flag could not execute at all.

    `run_protocol` filtered both pools, then compared the FILTERED gallery
    length against `protocol["gallery_size"]` and raised. Every protocol on
    disk declares a size -- A 9,138 / B 45,692 / C 4,569 / D 36,554 -- and
    `main` refuses any key not in that file, so all four raised, with a message
    blaming a stale artifact. Measured before the fix: A short by 47, B by 253,
    C by 20, D by 206.

    1,027 tests were green because none of them passed a non-empty
    `exclude_uids`. This is that test.

    The staleness guard must SURVIVE, so both halves are asserted: a genuinely
    stale declaration still raises, and a deliberate exclusion does not.
    """
    splits = {"train": [], "test": ["a", "b", "c"], "dev_val": ["a"]}
    proto = {"query_split": "test", "gallery_split": "test", "gallery_size": 3}

    # Stale by one, no exclusion: must still raise. Without this half, a check
    # deleted outright would also pass the half below.
    with pytest.raises(ValueError, match="declares gallery_size"):
        n15.run_protocol("stale", {**proto, "gallery_size": 4}, splits,
                         None, None, "mean", "cpu", 4, "none", 0, 8)

    # The same declared size, one asset excluded: must NOT raise THIS error.
    # It goes on to fail on the None backbone, which is a different failure and
    # is what "the size check let it through" looks like from here.
    try:
        n15.run_protocol("filtered", proto, splits, None, None, "mean", "cpu",
                         4, "none", 0, 8, False, None, None, {"b"})
    except ValueError as e:
        assert "declares gallery_size" not in str(e), (
            f"the exclusion was read as a stale artifact: {e}")
    except Exception:
        pass  # reached the model with a None backbone: the size check passed
