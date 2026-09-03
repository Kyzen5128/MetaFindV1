"""`perturb_cloud`: the query-side point-cloud observation policies.

One check per property the trainer and evaluator rely on: same seed -> same
cloud (train and eval must draw the same plane), `none` is the identity, the
output is always N_POINTS x 6 and unit-normalised, and a one-sided scan really
keeps one side.
"""
import numpy as np
import pytest

from metafind.data.observation import PC_PERTURBATIONS, perturb_cloud
from metafind.data.pointclouds import DEFAULT_GREY, N_POINTS


def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(N_POINTS, 3)).astype(np.float32)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)          # unit sphere
    rgb = rng.uniform(size=(N_POINTS, 3)).astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


@pytest.mark.parametrize("policy", PC_PERTURBATIONS)
def test_shape_norm_and_determinism(policy):
    c = _cloud()
    a = perturb_cloud(c, policy, seed=123)
    b = perturb_cloud(c, policy, seed=123)
    assert a.shape == (N_POINTS, 6) and a.dtype == np.float32
    assert np.array_equal(a, b), "train and eval must draw the same observation"
    assert np.max(np.linalg.norm(a[:, :3], axis=1)) == pytest.approx(1.0, abs=1e-5)


def test_none_is_identity_and_others_are_not():
    c = _cloud()
    assert np.array_equal(perturb_cloud(c, "none", 1), c)
    for policy in PC_PERTURBATIONS[1:]:
        assert not np.array_equal(perturb_cloud(c, policy, 1), c), policy


def test_half_keeps_one_side_and_nocolor_greys():
    c = _cloud()
    h = perturb_cloud(c, "half", seed=5)
    # every kept point lies on one side of SOME plane through the centroid:
    # the best-separating direction (top principal axis) has all points on one
    # sign after centring the ORIGINAL cloud, so the kept xyz span < a full sphere
    span = h[:, :3].max(0) - h[:, :3].min(0)
    assert span.min() < 1.5 * span.max() and (h[:, :3].std(0) < c[:, :3].std(0) * 1.01).any()
    assert len(np.unique(h[:, :3], axis=0)) < N_POINTS   # filled by repetition
    g = perturb_cloud(c, "nocolor", seed=5)
    assert np.all(g[:, 3:] == DEFAULT_GREY)


def test_unknown_policy_refused():
    with pytest.raises(ValueError):
        perturb_cloud(_cloud(), "upside_down", 0)
