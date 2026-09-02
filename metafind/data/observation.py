"""Which observation of an asset each tower sees.

[SPEC `workflow/REPRODUCTION_PROTOCOL_20260903.md` §四 問題 4, §十一]

The word this module exists to destroy is `same_record`. It carried two claims
at once:

    positive_policy   query A pairs with gallery A          PAPER FACT
    observation       query A and gallery A see the SAME    UNRESOLVED
                      text, the same image, the same cloud

The first is `2methdology.tex:77`, Eq. 5: the denominator sums over gallery
assets `A'`, so the positive is the asset, not the observation. The second the
paper never states, and 問題 4 is explicit that a query drawing `view_03` while
the gallery pools all twelve is STILL a same-uid positive. One token that means
both cannot express that, so a run could not say which of the two it had chosen.

Separating them costs one thing and buys another. It costs a vocabulary that is
larger than what this corpus can currently serve. It buys the ability to state,
in a protocol artifact rather than in a code path, that a run used the pooled
image on both sides -- which is the fact a reader needs to interpret a recall
number, and which `same_record` could only imply.

What this corpus can actually serve
-----------------------------------
Every embedding file already holds `views (12, 1280)` BESIDE the pooled `image`
vector, so every image policy below is reachable from the cache with no
re-encoding. That is why §七's "12 per-view features" requirement needed no
work: n06 has been writing them all along.

Text and point cloud are the opposite. A second caption's TEXT exists in every
annotation (`description_candidates`), but its VECTOR does not -- only the
canonical string was ever encoded. A second point-cloud sample does not exist
at all. Both alternate policies are therefore declared here and REFUSED at
resolution time with the reason, rather than silently falling back to the
canonical observation:

    a policy that quietly degrades to its neighbour is worse than one that
    raises, because the run records the policy it asked for.

The query pack supplied both, and it is archived, not deleted; `QueryPack`
remains the path for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- the vocabulary, per §十一 ---------------------------------------------
#
# Every value here is an OBSERVATION policy. None of them is a PAPER FACT: the
# paper states neither which observation a query draws nor how views become one
# image modality (問題 4, 問題 5, both UNRESOLVED). Choosing among them is an
# IMPLEMENTATION CHOICE and must be recorded as one.
TextPolicy = Literal["canonical", "alternate_caption"]
ImagePolicy = Literal["same_mean", "single_view", "held_out_view", "disjoint_views"]
PcPolicy = Literal["canonical_pc", "resampled_pc"]

TEXT_POLICIES = ("canonical", "alternate_caption")
IMAGE_POLICIES = ("same_mean", "single_view", "held_out_view", "disjoint_views")
PC_POLICIES = ("canonical_pc", "resampled_pc")

# Reachable from the n06 cache alone, with no re-encoding and no query pack.
CACHE_SERVED = {
    "text": ("canonical",),
    "image": IMAGE_POLICIES,          # all four: `views (12,1280)` is stored
    "pc": ("canonical_pc",),
}

# Why each unreachable policy is unreachable. Named, so a refusal says what to
# do rather than only that it will not.
NEEDS_PACK = {
    "alternate_caption":
        "the alternative caption's TEXT is in every annotation "
        "(description_candidates) but its VECTOR was never encoded -- only the "
        "canonical string went through the frozen text tower. Build a query "
        "pack (tools/make_query_pack.py) or encode the candidates.",
    "resampled_pc":
        "a second independent point-cloud sample of the same mesh does not "
        "exist on this corpus; n03 wrote one sample per asset. Build a query "
        "pack, which resamples, or run n03 again under a second seed.",
}

# The only positive policy the paper supports.
POSITIVE_POLICIES = ("same_uid",)


@dataclass(frozen=True)
class Observation:
    """One tower's observation of an asset, per modality."""

    text: str = "canonical"
    image: str = "same_mean"
    pc: str = "canonical_pc"

    def as_protocol(self) -> dict:
        return {"text": {"policy": self.text},
                "image": {"policy": self.image},
                "pc": {"policy": self.pc}}

    def needs_views(self) -> bool:
        """Whether this observation reads the per-view matrix rather than the mean."""
        return self.image != "same_mean"

    def needs_pack(self) -> tuple[str, ...]:
        return tuple(p for p in (self.text, self.image, self.pc) if p in NEEDS_PACK)


@dataclass(frozen=True)
class ObservationProtocol:
    """The whole §十一 block: what pairs with what, and what each side sees."""

    positive_policy: str
    query: Observation
    gallery: Observation

    def as_protocol(self) -> dict:
        return {"positive_policy": {"type": self.positive_policy},
                "query_observation": self.query.as_protocol(),
                "gallery_observation": self.gallery.as_protocol()}

    def is_same_observation(self) -> bool:
        """True when both towers see the same thing -- the old `same_record`.

        Kept as a QUESTION the protocol can answer, not as a mode it can be set
        to. `same_record` was a setting; this is a property of two policies that
        happen to agree, which is what makes the distinction recordable.
        """
        return self.query == self.gallery


def _one(kind: str, value, allowed: tuple[str, ...], side: str) -> str:
    if isinstance(value, dict):
        value = value.get("policy")
    if value not in allowed:
        raise ValueError(
            f"{side}_observation.{kind}: unknown policy {value!r}. "
            f"Known: {list(allowed)}")
    return value


def resolve(block: dict, *, cache_only: bool = True) -> ObservationProtocol:
    """Turn the protocol's observation block into something the dataset consumes.

    `cache_only` refuses the policies this corpus cannot serve from the n06
    cache. It defaults to True because the alternative -- accepting the policy
    and quietly serving the canonical observation instead -- is the failure this
    module exists to prevent: the checkpoint would record a policy the towers
    never saw. Pass False only when a query pack supplies the missing arm.
    """
    pos = (block.get("positive_policy") or {})
    pos = pos.get("type") if isinstance(pos, dict) else pos
    if pos not in POSITIVE_POLICIES:
        raise ValueError(
            f"positive_policy {pos!r} is not one of {list(POSITIVE_POLICIES)}. "
            "same_uid is the only one the paper supports: Eq. 5's denominator "
            "sums over gallery ASSETS (2methdology.tex:77).")

    out = {}
    for side in ("query", "gallery"):
        b = block.get(f"{side}_observation") or {}
        out[side] = Observation(
            text=_one("text", b.get("text", "canonical"), TEXT_POLICIES, side),
            image=_one("image", b.get("image", "same_mean"), IMAGE_POLICIES, side),
            pc=_one("pc", b.get("pc", "canonical_pc"), PC_POLICIES, side))

    proto = ObservationProtocol(pos, out["query"], out["gallery"])

    if cache_only:
        for side, obs in (("query", proto.query), ("gallery", proto.gallery)):
            for kind, policy in (("text", obs.text), ("image", obs.image),
                                 ("pc", obs.pc)):
                if policy not in CACHE_SERVED[kind]:
                    raise ValueError(
                        f"{side}_observation.{kind} = {policy!r} cannot be "
                        f"served from the embedding cache. {NEEDS_PACK[policy]}")
    # The gallery is modality-complete by PAPER FACT (2methdology.tex:75) and a
    # gallery drawing one view is not modality-complete in any useful sense, but
    # the paper does not forbid it either, so this is a warning-shaped fact
    # recorded in the protocol rather than a refusal.
    return proto


def view_indices(policy: str, uid: str, n_views: int) -> list[int]:
    """Which stored views this policy reads, for one asset.

    A rule over `uid_seed`, never a stored map: `uid_seed` is already the
    project's per-asset seed (`pointclouds.py`), so the image draw cannot drift
    out of step with a file that would have to be maintained beside it.

    `held_out_view` and `disjoint_views` are the two halves of one split, so
    they are defined together and their intersection is empty by construction
    rather than by a test that could be deleted.
    """
    from metafind.data.pointclouds import uid_seed

    if policy == "same_mean":
        return list(range(n_views))
    k = uid_seed(uid) % n_views
    if policy in ("single_view", "held_out_view"):
        return [k]
    if policy == "disjoint_views":
        return [i for i in range(n_views) if i != k]
    raise ValueError(f"no view rule for {policy!r}")
