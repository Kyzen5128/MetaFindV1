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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metafind.models.dual_tower import DualTowerConfig
from metafind.models.essgnn import ESSGNNConfig
from metafind.models.fusion import FusionConfig
from metafind.models.losses import ContrastiveConfig
from metafind.models.ulip_backbone import BackboneConfig

__all__ = ["Stage1RuntimeConfig"]

ENCODING_FIELDS = ("text_serialization", "image_aggregation", "clip_train_scope")
TRAINING_FIELDS = (
    "fusion",
    "tower_sharing",
    "allow_all_masked",
    "similarity",
    "hyperparameter_config_hash",
)

# U-14. Aggregations that collapse the 11 views once, at encoding time, and can
# therefore be served from a cache of one vector per asset. Anything else needs
# the per-view embeddings kept, which is why n06 reads the protocol.
PRECOMPUTABLE_AGGREGATIONS = {"fixed_view", "mean", "max"}


def _require(protocol: dict, fields: tuple[str, ...], name: str) -> None:
    if protocol.get("status") != "resolved":
        raise ValueError(f"{name} is not resolved; its gate exists to stop this")
    if missing := set(fields) - protocol.keys():
        raise ValueError(f"{name} is missing {sorted(missing)}")


@dataclass
class Stage1RuntimeConfig:
    """Everything Stage 1 needs, derived from the two protocols."""

    backbone: BackboneConfig
    tower: DualTowerConfig
    loss: ContrastiveConfig
    text_serialization: str
    image_aggregation: str
    clip_train_scope: Literal["frozen", "trainable"]

    @property
    def may_use_cached_text_image(self) -> bool:
        """Whether n06's cache is admissible for this run.

        False under `trainable`: a cache is by construction the output of a
        network that is not being updated, so reading one would silently make
        the run the frozen variant. Also false when the aggregation is chosen
        per training step, since one stored vector cannot answer it.
        """
        return (
            self.clip_train_scope == "frozen"
            and self.image_aggregation in PRECOMPUTABLE_AGGREGATIONS
        )

    @classmethod
    def from_protocols(
        cls,
        encoding_protocol: dict,
        training_protocol: dict,
        *,
        dim: int,
        essgnn: ESSGNNConfig,
        checkpoint,
        device: str = "cuda",
    ) -> Stage1RuntimeConfig:
        _require(encoding_protocol, ENCODING_FIELDS, "stage1_encoding_protocol")
        _require(training_protocol, TRAINING_FIELDS, "stage1_protocol")

        clip_scope = encoding_protocol["clip_train_scope"]
        if clip_scope not in ("frozen", "trainable"):
            raise ValueError(f"clip_train_scope must be frozen or trainable, got {clip_scope!r}")

        # U-34 maps onto the backbone's freeze scope. `trainable` is the reading
        # RA-3 measures; it is not known to fit on 24 GB.
        train_scope = "full" if clip_scope == "trainable" else "point_encoder_and_fuser"

        fusion = FusionConfig(dim=dim, kind=training_protocol["fusion"])
        # U-16. Separate objects unless the protocol ties them, so freezing one
        # tower without the other stays possible (paper 2.6 freezes the gallery
        # in Stage 2).
        gallery_fusion = (
            fusion
            if training_protocol["tower_sharing"] == "shared"
            else FusionConfig(dim=dim, kind=training_protocol["fusion"])
        )

        return cls(
            backbone=BackboneConfig(
                checkpoint=checkpoint, device=device, train_scope=train_scope
            ),
            tower=DualTowerConfig(
                dim=dim,
                essgnn=essgnn,
                query_fusion=fusion,
                gallery_fusion=gallery_fusion,
            ),
            loss=ContrastiveConfig(),
            text_serialization=encoding_protocol["text_serialization"],
            image_aggregation=encoding_protocol["image_aggregation"],
            clip_train_scope=clip_scope,
        )
