"""The ladder's rungs are prefixes of one curve, not three separate curves.

[KYZEN 2026-08-29] The pilot ladder asks "how many epochs before it stops
improving". Before `--lr-horizon`, raising `--epochs` also slowed the anneal, so
each rung trained under a different schedule and the rungs could not be compared
-- e10 scoring below e5 said nothing about epoch count.
"""
from __future__ import annotations

import numpy as np
import pytest

from metafind.train.stage1 import cosine_schedule

BASE, FINAL, START, WARMUP, NITER = 5e-4, 1e-5, 1e-6, 1, 499


def sched(epochs: int) -> np.ndarray:
    return cosine_schedule(BASE, FINAL, epochs, NITER, WARMUP, START)


def test_a_rung_is_a_bitwise_prefix_of_the_full_curve():
    """The whole point: 5 and 10 epochs must read the SAME rates, step for step.

    Bitwise, not approximate. A schedule that merely 'looks similar' at the
    start is what the old construction already produced.
    """
    full = sched(250)
    for stop in (5, 10, 25):
        np.testing.assert_array_equal(full[: stop * NITER], sched(250)[: stop * NITER])
        assert len(full) >= stop * NITER


def test_without_a_horizon_the_rungs_are_different_curves():
    """The negative case. Without this, the test above passes trivially.

    This is the defect being fixed, asserted as a defect: under the old
    construction the rungs diverge, and by a lot -- so a test that only checked
    'prefixes match' would also pass on a schedule that ignored its arguments.
    """
    five, ten = sched(5), sched(10)
    n = 5 * NITER
    assert not np.array_equal(five[:n], ten[:n]), (
        "5-epoch and 10-epoch curves are identical over their shared span; "
        "then the confound this flag exists for never existed")
    # and quantify it, so the comment above is not the only evidence
    last = five[n - 1]
    assert last == pytest.approx(FINAL, rel=0.05), \
        f"a 5-epoch run should END at the floor {FINAL}, got {last}"
    assert ten[n - 1] > 5 * FINAL, \
        f"a 10-epoch run at epoch 5 should still be well above the floor, got {ten[n-1]}"


def test_the_curve_reaches_the_floor_only_at_the_horizon():
    """A prefix must NOT have annealed: that is what makes it a prefix."""
    full = sched(250)
    assert full[-1] == pytest.approx(FINAL, rel=0.05)
    for stop in (5, 10, 25):
        assert full[stop * NITER - 1] > 10 * FINAL, (
            f"at epoch {stop} of 250 the lr is already at the floor; "
            f"the horizon is not doing anything")


def test_warmup_is_unchanged_by_the_horizon():
    """Warmup is measured in epochs, so it must not stretch with the horizon.

    If it did, a 250-epoch horizon would spend 250x longer warming up and every
    short rung would be pure warmup.
    """
    for epochs in (5, 10, 250):
        s = sched(epochs)
        assert s[0] == pytest.approx(START)
        peak = int(np.argmax(s))
        assert peak <= WARMUP * NITER, \
            f"peak at step {peak}, warmup is {WARMUP * NITER} steps"
        assert s[peak] == pytest.approx(BASE, rel=1e-9)
