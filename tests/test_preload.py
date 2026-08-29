"""Preloading returns the same data as reading from disk, byte for byte.

[KYZEN 2026-08-29] `--preload` exists to test a crash hypothesis, not to speed
anything up. That makes "it must not change a single number" the whole contract:
if the preloaded path returned even slightly different arrays, a crash-free run
under `--preload` would be uninterpretable -- we would not know whether the
machine survived because the I/O stopped or because the training changed.
"""
from __future__ import annotations

import numpy as np
import pytest

from metafind import paths
from metafind.train.stage1 import Stage1Dataset


def _uids(n: int = 6) -> list[str]:
    files = sorted(paths.EMBEDDINGS.glob("*.npz"))[:200]
    out = []
    for f in files:
        if (paths.POINTCLOUDS / f.name).exists():
            out.append(f.stem)
        if len(out) == n:
            break
    return out


@pytest.fixture(scope="module")
def uids():
    u = _uids()
    if len(u) < 2:
        pytest.skip("no encoded assets on this machine")
    return u


def test_preloaded_items_are_identical_to_disk_items(uids):
    """The contract, asserted exactly. Not allclose -- equal."""
    disk = Stage1Dataset(uids, "mean", preload=False)
    ram = Stage1Dataset(uids, "mean", preload=True)
    assert ram.cache is not None and disk.cache is None
    for i in range(len(uids)):
        a, b = disk[i], ram[i]
        assert a["uid"] == b["uid"]
        for k in ("text", "image", "pc"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs at {i}")
            assert a[k].dtype == b[k].dtype
            assert a[k].shape == b[k].shape


def test_the_comparison_can_actually_fail(uids):
    """Without this, the test above would pass on a dataset that returned zeros.

    A corrupted byte in the cache must be visible to the assertion used above.
    """
    ram = Stage1Dataset(uids, "mean", preload=True)
    ram.cache[uids[0]]["pc"][0, 0] += np.float32(1e-3)
    disk = Stage1Dataset(uids, "mean", preload=False)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(disk[0]["pc"], ram[0]["pc"])


def test_items_are_copies_so_a_write_cannot_poison_the_cache(uids):
    """The failure this guards is silent and would only show in later epochs.

    The cache is read once and then read again every epoch. If `__getitem__`
    handed out views, anything writing in-place -- an augmentation, a
    normalisation, a careless `+=` -- would permanently alter the asset for the
    rest of training, and nothing would raise.
    """
    ram = Stage1Dataset(uids, "mean", preload=True)
    before = ram.cache[uids[0]]["pc"].copy()
    item = ram[0]
    item["pc"] += np.float32(1.0)
    np.testing.assert_array_equal(ram.cache[uids[0]]["pc"], before)
    # and the next read is unaffected
    np.testing.assert_array_equal(ram[0]["pc"], before)


def test_mean_aggregation_does_not_keep_the_per_view_matrix(uids):
    """`views` is 12x the pooled vector and `mean` never reads it.

    Keeping it would roughly double the resident size for nothing, and the size
    is the one cost this flag actually has.
    """
    ram = Stage1Dataset(uids, "mean", preload=True)
    for e in ram.cache.values():
        assert "views" not in e
