"""Build every Stage 1 runtime object from the resolved protocols.

The single supported way for a Stage 1 trainer to obtain its configuration.

Why this exists
---------------

ESSGNN already learned this lesson: a protocol that a gate approves is only
worth something if the code cannot quietly use different values. Stage 1 has
the same exposure and more of it, because three separate config objects each
carry defaults that look reasonable in isolation::

    FusionConfig.kind      = "masked_mlp"    but U-13 is open
    BackboneConfig         .train_scope      but U-34 is open
    ContrastiveConfig      temperature etc.  but U-22 and U-24 are open

A trainer that writes ``FusionConfig(dim=d)`` gets masked_mlp whatever the
protocol resolved to, and nothing downstream notices. Constructing them here,
from the protocols, removes the opportunity rather than detecting the mistake
afterwards.

Two protocols, because they are decided at different times
----------------------------------------------------------

``stage1_encoding_protocol`` is resolved at n05b, BEFORE n06 encodes anything,
because its three fields decide what n06 should produce (U-15's serialization,
U-14's view aggregation, U-34's freeze scope). ``stage1_protocol`` is resolved
at n09 and used at training time (U-13, U-16, U-22, U-23, U-24).

Reading a protocol field is not consuming it
---------------------------------------------

[CORRECTED] An earlier version required all five training fields and then used
two. ``allow_all_masked`` never reached ``sample_modality_mask``; ``similarity``
was accepted as any string while the loss computed cosine unconditionally;
``hyperparameter_config_hash`` was checked for presence and dereferenced
nothing, so every value of U-22 came from library defaults while the channel
recorded a hash that pointed at them by accident at best. A gate that admits a
resolved protocol has not established that the protocol was followed. Every
field now either reaches an object or refuses the run.

No ESSGNN here
--------------

[CORRECTED] This module used to require an ``ESSGNNConfig``. Paper 2.6 makes
Stage 1 object-level cross-modal alignment with no spatial context, and the
graph does not resolve ``essgnn_arch_protocol`` until n09b, one layer after
n10 starts -- so the only way to satisfy the old signature was to invent an
architecture, which is what the tests did. Stage 1 builds ``use_layout=False``
towers; Stage 2 attaches the layout branch after G6.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

from metafind.models.dual_tower import TOWER_SHARING, DualTowerConfig, TowerSharing
from metafind.models.fusion import FusionConfig, sample_modality_mask
from metafind.models.losses import ContrastiveConfig
from metafind.models.ulip_backbone import BackboneConfig

__all__ = [
    "Stage1RuntimeConfig",
    "UnsupportedProtocol",
    "canonical_hyperparameter_hash",
    "REQUIRED_HYPERPARAMETERS",
    "SUPPORTED_SIMILARITY",
    "PRECOMPUTABLE_AGGREGATIONS",
    "PER_VIEW_AGGREGATIONS",
]

ENCODING_FIELDS = ("text_serialization", "image_aggregation", "clip_train_scope")
TRAINING_FIELDS = (
    "fusion",
    "tower_sharing",
    "allow_all_masked",
    "similarity",
    "hyperparameter_config_hash",
)

# U-24. `sim(.,.)` is never defined; cosine is our reading. Anything else must
# stop the run rather than be accepted and then ignored -- MetaFindContrastiveLoss
# normalises both sides unconditionally, so a protocol saying `dot_product`
# would have produced cosine numbers under a dot-product label.
SUPPORTED_SIMILARITY = ("cosine",)

# U-14. `n06` now caches all eleven per-view embeddings, so the split is no
# longer "can this be cached" but "how many vectors does the cache hold".
PRECOMPUTABLE_AGGREGATIONS = ("fixed_view", "mean", "max")
PER_VIEW_AGGREGATIONS = ("random_single_view",)

# U-22. The paper gives none of these. The artifact must therefore NAME each
# one -- this list is what "the hyperparameters are recorded" means in the
# report, and the run refuses to start if any is missing.
REQUIRED_HYPERPARAMETERS = (
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "batch_size",
    "epochs",
    "p_mask",
    "init_temperature",
    "learnable_temperature",
    "max_logit_scale",
    "seed",
)


class UnsupportedProtocol(ValueError):
    """A protocol field names something this code does not implement.

    Distinct from a malformed protocol: the record is well-formed and a human
    resolved it, but running it would produce numbers that do not match the
    label. Refusing is the only honest option.
    """


def canonical_hyperparameter_hash(hyperparameters: dict[str, Any]) -> str:
    """sha256 of the hyperparameter artifact, key-order independent."""
    blob = json.dumps(hyperparameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _require(protocol: dict, fields: tuple[str, ...], name: str) -> None:
    if protocol.get("status") != "resolved":
        raise ValueError(f"{name} is not resolved; its gate exists to stop this")
    if missing := set(fields) - protocol.keys():
        raise ValueError(f"{name} is missing {sorted(missing)}")


@dataclass
class Stage1RuntimeConfig:
    """Everything Stage 1 needs, derived from the two protocols."""

    backbone: BackboneConfig
    # U-16. `fully_separate` is the only reading with two backbones; the other
    # two share one, which is why this is None rather than a copy.
    gallery_backbone: BackboneConfig | None
    tower: DualTowerConfig
    loss: ContrastiveConfig
    text_serialization: str
    image_aggregation: str
    clip_train_scope: Literal["frozen", "trainable"]
    tower_sharing: TowerSharing
    similarity: str
    allow_all_masked: bool
    hyperparameters: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def may_use_cached_text_image(self) -> bool:
        """Whether n06's cache is admissible for this run.

        False under `trainable`: a cache is by construction the output of a
        network that is not being updated, so reading one would silently make
        the run the frozen variant. Under `frozen` it is always True -- n06
        stores all eleven per-view embeddings, so even an aggregation chosen
        per training step is answerable from the cache. ``cache_layout`` says
        how many vectors that cache holds.
        """
        return self.clip_train_scope == "frozen"

    @property
    def cache_layout(self) -> Literal["aggregated", "per_view", "none"]:
        """What n06 must store for this protocol.

        [CORRECTED] This used to be folded into `may_use_cached_text_image`,
        which returned False for `random_single_view` on the grounds that "one
        stored vector cannot answer it". True of the old schema; the channel
        now carries `image: uri | list[uri] per view`, so eleven frozen CLIP
        embeddings computed once serve a per-step random view exactly.
        """
        if self.clip_train_scope != "frozen":
            return "none"
        return "per_view" if self.image_aggregation in PER_VIEW_AGGREGATIONS else "aggregated"

    def sample_present_mask(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sec. 2.6's modality masking, with U-23 taken from the protocol.

        The only route by which `allow_all_masked` reaches the sampler. A
        trainer calling `sample_modality_mask` directly gets `allow_empty=True`
        whatever the protocol says, which is how the field came to be required
        by G3 and consumed by nothing.
        """
        return sample_modality_mask(
            batch_size,
            p_mask=self.hyperparameters["p_mask"],
            allow_empty=self.allow_all_masked,
            device=device,
            generator=generator,
        )

    @classmethod
    def from_protocols(
        cls,
        encoding_protocol: dict,
        training_protocol: dict,
        *,
        dim: int,
        checkpoint,
        hyperparameters: dict[str, Any],
        device: str = "cuda",
    ) -> Stage1RuntimeConfig:
        _require(encoding_protocol, ENCODING_FIELDS, "stage1_encoding_protocol")
        _require(training_protocol, TRAINING_FIELDS, "stage1_protocol")

        clip_scope = encoding_protocol["clip_train_scope"]
        if clip_scope not in ("frozen", "trainable"):
            raise ValueError(f"clip_train_scope must be frozen or trainable, got {clip_scope!r}")

        aggregation = encoding_protocol["image_aggregation"]
        if aggregation not in PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS:
            raise UnsupportedProtocol(
                f"image_aggregation={aggregation!r} is not implemented; "
                f"known: {PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS}"
            )

        similarity = training_protocol["similarity"]
        if similarity not in SUPPORTED_SIMILARITY:
            raise UnsupportedProtocol(
                f"similarity={similarity!r}: MetaFindContrastiveLoss normalises both "
                f"sides, so it can only compute {SUPPORTED_SIMILARITY}. Implement it "
                "in the loss before resolving the protocol this way -- accepting the "
                "field and computing cosine anyway mislabels the numbers."
            )

        sharing = training_protocol["tower_sharing"]
        if sharing not in TOWER_SHARING:
            raise UnsupportedProtocol(
                f"tower_sharing={sharing!r} is not one of U-16's readings {TOWER_SHARING}"
            )

        # U-22. The hash must DEREFERENCE. A presence check leaves every
        # hyperparameter on a library default while the channel records a hash,
        # which reads in the report as though the values were chosen.
        if missing := set(REQUIRED_HYPERPARAMETERS) - hyperparameters.keys():
            raise ValueError(
                f"hyperparameter artifact is missing {sorted(missing)}; U-22 says the "
                "paper gives none of these, so each must be named explicitly"
            )
        actual = canonical_hyperparameter_hash(hyperparameters)
        if actual != training_protocol["hyperparameter_config_hash"]:
            raise ValueError(
                "hyperparameter_config_hash does not match the artifact: protocol says "
                f"{training_protocol['hyperparameter_config_hash']}, artifact hashes to "
                f"{actual}. The run would train on values the gate never saw."
            )

        # U-34 maps onto the backbone's freeze scope. `trainable` is the reading
        # RA-3 measures; it is not known to fit on 24 GB.
        train_scope = "full" if clip_scope == "trainable" else "point_encoder_and_fuser"
        backbone = BackboneConfig(checkpoint=checkpoint, device=device, train_scope=train_scope)

        fusion = FusionConfig(dim=dim, kind=training_protocol["fusion"])
        # Under `fully_shared` the towers end up as ONE module, so the second
        # config is never instantiated; the tying happens in MetaFindDualTower,
        # because passing the same config to both builds two parameter sets.
        gallery_fusion = (
            fusion
            if sharing == "fully_shared"
            else FusionConfig(dim=dim, kind=training_protocol["fusion"])
        )

        return cls(
            backbone=backbone,
            gallery_backbone=(
                BackboneConfig(checkpoint=checkpoint, device=device, train_scope=train_scope)
                if sharing == "fully_separate"
                else None
            ),
            tower=DualTowerConfig(
                dim=dim,
                tower_sharing=sharing,
                # Stage 1 has no layout branch (2.6), and essgnn_arch_protocol
                # is not resolved until n09b.
                essgnn=None,
                use_layout=False,
                query_fusion=fusion,
                gallery_fusion=gallery_fusion,
            ),
            loss=ContrastiveConfig(
                bidirectional=False,  # Eq. 5 is one-directional; Eq. 7a/7b is Stage 2
                learnable_temperature=bool(hyperparameters["learnable_temperature"]),
                init_temperature=float(hyperparameters["init_temperature"]),
                max_logit_scale=float(hyperparameters["max_logit_scale"]),
            ),
            text_serialization=encoding_protocol["text_serialization"],
            image_aggregation=aggregation,
            clip_train_scope=clip_scope,
            tower_sharing=sharing,
            similarity=similarity,
            allow_all_masked=bool(training_protocol["allow_all_masked"]),
            hyperparameters=dict(hyperparameters),
        )
