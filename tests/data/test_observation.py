"""The §十一 observation protocol: what pairs with what, and what each side sees."""
import numpy as np
import pytest

from metafind.data.observation import (
    CACHE_SERVED,
    IMAGE_POLICIES,
    Observation,
    ObservationProtocol,
    resolve,
    view_indices,
)


def _block(**kw):
    base = {"positive_policy": {"type": "same_uid"},
            "query_observation": {"text": {"policy": "canonical"},
                                  "image": {"policy": "same_mean"},
                                  "pc": {"policy": "canonical_pc"}},
            "gallery_observation": {"text": {"policy": "canonical"},
                                    "image": {"policy": "same_mean"},
                                    "pc": {"policy": "canonical_pc"}}}
    base.update(kw)
    return base


def test_the_only_positive_policy_is_same_uid():
    """Eq. 5's denominator sums over gallery ASSETS (2methdology.tex:77), so the
    positive is the asset. Anything else is not a MetaFind reproduction."""
    assert resolve(_block()).positive_policy == "same_uid"
    with pytest.raises(ValueError, match="same_uid"):
        resolve(_block(positive_policy={"type": "same_observation"}))


def test_same_record_is_now_a_question_not_a_setting():
    """The whole point of the split. `same_record` asserted "the same" without
    saying the same WHAT; two policies that agree is a property you can read."""
    assert resolve(_block()).is_same_observation()
    p = resolve(_block(query_observation={"image": {"policy": "single_view"}}))
    assert not p.is_same_observation()
    assert p.positive_policy == "same_uid", (
        "the query drawing one view is STILL a same-uid positive -- that is "
        "exactly the case 問題 4 says the paper permits")


@pytest.mark.parametrize("policy,reason", [
    ("alternate_caption", "text"), ("resampled_pc", "pc")])
def test_a_policy_the_cache_cannot_serve_is_refused_not_degraded(policy, reason):
    """A policy that quietly falls back to its neighbour is worse than one that
    raises: the checkpoint would record a policy the towers never saw."""
    with pytest.raises(ValueError) as e:
        resolve(_block(query_observation={reason: {"policy": policy}}))
    msg = str(e.value)
    assert policy in msg
    assert "query pack" in msg, "the refusal must say what would supply it"


def test_every_image_policy_is_servable_from_the_stored_views():
    """§七 needed no work because n06 has been writing `views (12,1280)` beside
    the mean all along. If that ever stops being true this fails."""
    assert set(CACHE_SERVED["image"]) == set(IMAGE_POLICIES)
    for p in IMAGE_POLICIES:
        resolve(_block(query_observation={"image": {"policy": p}}))


def test_held_out_and_disjoint_are_complementary_by_construction():
    """Two halves of one split, so their intersection is empty because of how
    they are defined, not because a test happens to check it."""
    for uid in ("a", "b", "000074a334c541878360457c672b6c2e"):
        held = view_indices("held_out_view", uid, 12)
        rest = view_indices("disjoint_views", uid, 12)
        assert len(held) == 1 and len(rest) == 11
        assert not set(held) & set(rest)
        assert sorted(held + rest) == list(range(12))
    assert view_indices("same_mean", "a", 12) == list(range(12))


def test_the_view_rule_is_re_derivable_from_the_uid_alone():
    """No stored map to drift out of step with; `uid_seed` is the whole rule."""
    from metafind.data.pointclouds import uid_seed

    for uid in ("a", "b", "c"):
        assert view_indices("single_view", uid, 12) == [uid_seed(uid) % 12]


def test_the_protocol_round_trips_through_its_own_block():
    p = ObservationProtocol("same_uid", Observation(image="single_view"),
                            Observation())
    assert resolve(p.as_protocol()) == p
