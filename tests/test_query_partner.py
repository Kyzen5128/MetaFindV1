"""`Stage1Dataset(partner="same_category")`: the query's text and image come from
another asset of the same LVIS category; deterministic; never the asset itself;
the query pc stays the asset's own. Runs against the real cache when present."""
import json

import numpy as np
import pytest

from metafind import paths


@pytest.fixture(scope="module")
def uids():
    p = paths.OUTPUTS / "splits.json"
    if not p.exists():
        pytest.skip("no splits.json on this machine")
    sp = json.loads(p.read_text())["object"]
    u = sorted(sp["dev_val"])                     # the whole pool: a 400-asset slice is mostly singletons
    if not (paths.EMBEDDINGS / f"{u[0]}.npz").exists():
        pytest.skip("no embedding cache")
    return u


def test_partner_is_same_category_never_self_and_deterministic(uids):
    from metafind.train.stage1 import Stage1Dataset
    a = Stage1Dataset(uids, "mean", partner="same_category")
    b = Stage1Dataset(uids, "mean", partner="same_category")
    assert a._partner_of == b._partner_of
    cat = lambda u: json.loads((paths.ANNOTATIONS / f"{u}.json").read_text())["lvis_category"]
    same, total = 0, 0
    for u, p in a._partner_of.items():
        assert p != u
        assert p in set(uids)
        total += 1
        same += cat(u) == cat(p)
    assert same / total > 0.8                    # singletons fall back to any category


def test_query_side_reads_partner_text_and_image_but_own_pc(uids):
    from metafind.train.stage1 import Stage1Dataset
    from metafind.data.pointclouds import uid_seed
    ds = Stage1Dataset(uids, "mean", partner="same_category")
    item = ds[0]
    u, p = uids[0], ds._partner_of[uids[0]]
    cached = np.load(paths.EMBEDDINGS / f"{p}.npz")
    assert np.allclose(item["q_text"], cached["text"].astype(np.float32))
    assert np.allclose(item["q_image"], cached["views"][uid_seed(p) % 12].astype(np.float32))
    assert "q_pc" not in item                     # pc falls back to the asset's own
    assert not np.allclose(item["q_text"], item["text"])


def test_unknown_partner_policy_refused(uids):
    from metafind.train.stage1 import Stage1Dataset
    with pytest.raises(ValueError):
        Stage1Dataset(uids[:8], "mean", partner="neighbour")
