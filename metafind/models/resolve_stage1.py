"""Record the Stage 1 encoding decisions before anything encodes anything.

# IMPLEMENTS-NODE: n05b_resolve_stage1_encoding

Writes ``stage1_encoding_protocol``, ``stage1_hyperparameters``,
``variant_registry`` and ``run_progress``.

Why this node exists at all
---------------------------

[ORDERING] These decisions used to sit in ``stage1_protocol``, which n09 writes
AFTER n06 has already encoded 46,052 assets. Each of them determines what the
encoding should BE, so resolving them afterwards produced a cache built before
its own specification existed. n06 now refuses to start without this artifact.

What is decided here, and by whom
----------------------------------

Three of the four have a defensible default and are recorded with the reasoning
below. The fourth does not, and this module will not invent it:

  U-15 text_serialization      pinned template, golden-string test
  U-14 image_aggregation       mean of the 11 per-view embeddings
  U-11 missing_modality        learned token
  U-34 paper_clip_train_scope  NO DEFAULT -- must be passed explicitly

U-34 is refused a default because the two fields it splits into mean different
things. ``actual_clip_train_scope`` is what this run does, and ``frozen`` is
well supported: ULIP-2 §3.3 states "We adopt the largest version of encoders
from OpenCLIP (ViT-G/14) ... and freeze it during pre-training", and ViT-bigG-14
in the optimizer does not fit 24 GB regardless. ``paper_clip_train_scope`` is
our READING of what MetaFind requires, and the whole content of deviation D-1
is the GAP between the two. Choosing it silently would either declare a
deviation that is not one or, worse, suppress one that is -- and the evidence
genuinely points both ways:

  toward frozen     MetaFind builds on ULIP-2, whose paper freezes OpenCLIP and
                    whose Eq. 3 objective min_{E_P} trains the 3D encoder only.
                    MetaFind never says it unfreezes CLIP. 2.6's "Both query and
                    gallery encoders are trained" is about the TOWERS, which do
                    train under a frozen CLIP -- the fuser and point encoder are
                    in the optimizer either way.

  toward trainable  3.4's ablation summary: "Fine-tuning the entire encoder
                    outperformed training the fuser only." Something beyond the
                    fuser was fine-tuned; whether "the entire encoder" reaches
                    ViT-bigG-14 or stops at the point encoder is exactly what is
                    unstated.

So it is an argument with no default, and ``decided_by`` records who supplied
it. RA-3 separately measures whether the trainable reading is executable here;
that audit reports and never blocks.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from datetime import datetime, timezone

from metafind import paths, runlog
from metafind.models.stage1_config import (
    KNOWN_MISSING_MODALITY,
    PAPER_P_MASK,
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
    REQUIRED_HYPERPARAMETERS,
    canonical_hyperparameter_hash,
)

NODE = "n05b_resolve_stage1_encoding"

PROTOCOL_PATH = paths.OUTPUTS / "stage1_encoding_protocol.json"
HYPERPARAMETERS_PATH = paths.OUTPUTS / "stage1_hyperparameters.json"
VARIANT_REGISTRY_PATH = paths.OUTPUTS / "variant_registry.json"

# --- U-15 -----------------------------------------------------------------

# [U-15, B-3, D0-008 §11.2] The serialization FAMILY. On its own it is not a
# cache identity and must never be used as one: the retired value
# "metafind_v1_natural" labelled two different transformations -- the metre-based
# v1 template and the centimetre one -- so a sidecar carrying it does not say
# which serializer produced its embedding. The identity n06 stamps and validates
# is text_serialization_id() below, which is content-addressed.
TEXT_SERIALIZATION_FAMILY = "metafind_v2_cm"

# [U-15] Paper 2.3 names the fields and gives no format. RATIFIED by D0-008
# (2026-08-21) as an IMPLEMENTATION CHOICE, and held by a golden-string test
# (L1-TEXT-SERIALIZATION), because reordering two fields silently changes every
# text embedding and therefore four of Table 1's columns.
#
# Natural prose rather than "Category: chair. Material: wood." because the frozen
# CLIP text tower was trained on captions, not on labelled records -- and the
# tower is frozen, so it cannot adapt to a format it never saw. The 77-token
# limit is the other constraint: a labelled multi-line record spends its budget
# on field names.
#
# [R-1, D0-008 §11.3] Field order is description -> category -> materials ->
# dimensions -> placement, and it is OURS. The earlier comment here claimed the
# order follows paper 2.3's own sentence; it does not. 2methdology.tex:28 reads
# "object category, size dimensions, materials, and placement constraints" --
# category -> DIMENSIONS -> materials -- which is not what this template emits.
# The paper does not constrain serialization order at all, so the order stays as
# it is and the claim is withdrawn rather than the code changed. Any future claim
# that 2.3 mandates an order must cite new evidence.
#
# The dimensions arrive already rendered by _dim(), so this string carries no
# format spec for them: "one decimal, trailing .0 stripped" is not expressible as
# one. {category} likewise receives the capitalised form (S-2). This constant
# therefore does NOT fully describe the emitted string, which is exactly why the
# cache identity hashes the emitted string instead of this template.
TEXT_TEMPLATE = (
    "{description} {category} made of {materials}, "
    "roughly {width} by {length} by {height} centimetres, "
    "{placement}."
)

# [U-15] RATIFIED by D0-008 §11.3 row 8 as an IMPLEMENTATION CHOICE whose
# retrieval impact is UNKNOWN. Paper Figure 2 gives the annotation SCHEMA but
# nothing about how it becomes the string CLIP sees, so this template is ours.
# Three fields the figure prints are deliberately NOT serialised:
#
#   synset   an identifier, not language. "robot.n.01" is noise to a text tower
#            trained on captions, and it costs tokens inside a 77-token budget.
#   volume   omitted. The justification this comment used to give -- "redundant,
#            it is width * length * height, already in the string" -- is
#            WITHDRAWN (D0-008 §12.4): a frozen text tower is not guaranteed to
#            multiply three numerals, and the rendered numbers are rounded, so
#            the equivalence does not hold even arithmetically.
#   mass     kept on disk, left out of the text. The size numbers at least carry
#            a claimed correlation with the mesh bounding box (r = 0.52-0.62),
#            but that figure is UNVERIFIED in this repository and must not be
#            reported as MEASURED (D0-008 §12.4). Mass has no visual support of
#            any kind and no ground truth in Objaverse.
#
# All three stay in the annotation record; only the encoder input omits them.
PLACEMENT_PHRASES = {
    # Rendered from the four independent booleans, so the sentence says exactly
    # what the annotation says. v1 flattened an 8-label list into "typically
    # placed handheld", which is where "gaming chair ... typically placed
    # handheld" reached CLIP.
    #
    # [R-3, D0-008 §12.3] A ("onWall", "onCeiling") entry used to sit in this
    # dict and was UNREACHABLE: placement_phrase() builds its key in the fixed
    # order (onCeiling, onWall, onFloor, onObject) and retries tuple(sorted(...)),
    # and both spellings are ("onCeiling", "onWall"). Master ruled DELETE rather
    # than fix. The fallback join already emits grammatical prose for those
    # records -- "typically mounted on a ceiling or on a wall" -- so deleting
    # changes 0 serialized strings, while "fixing" it would change strings that
    # the ratification measurement in D0-008 §11.4 never covered.
    ("onCeiling",): "typically mounted on a ceiling",
    ("onWall",): "typically mounted on a wall",
    ("onFloor",): "typically placed on the floor",
    ("onObject",): "typically placed on top of other objects",
    ("onFloor", "onObject"): "typically placed on the floor or on other objects",
}
NO_PLACEMENT_PHRASE = "with no typical placement"

# CLIP's text tower truncates at 77 tokens SILENTLY, and the description leads
# the sentence -- so an overlong one does not lose itself, it loses the TAIL,
# which is where placement_constraints lives. That is the single most
# retrieval-relevant field, and it would vanish without an error.
#
# MEASURED over the first 1,325 annotations: 52 tokens median, 61 at p95, 72
# worst case; descriptions run 94 chars median, 200 worst; the fixed tail costs
# about 32 tokens with the observed materials lists (1-4 entries, 57 chars
# worst). Five tokens of headroom on a corpus where 44,000 assets have not been
# annotated yet is not headroom.
#
# [R-2, D0-008 §11.3] Correction. This comment used to claim that "EVERY
# variable-length part is bounded, not just the description". That was false as
# documentation. Three parts are bounded by a CAP: description, category, and
# the materials list. The placement clause is bounded by CONSTRUCTION, not by a
# cap -- it is built from four booleans over a closed vocabulary, so its longest
# possible form is the four single-flag phrases joined. MAX_PLACEMENT was
# defined here and never read by any code path, so it has been DELETED rather
# than left standing as an unenforced bound that reads like an enforced one.
#
# Dropping the fourth material is the right thing to lose. The annotation prompt
# asks for materials "most prominent first", so a prefix is principled -- and
# the alternative is CLIP dropping the placement constraint instead, silently.
#
# These are CHARACTER caps and CLIP counts TOKENS, so they bound the realistic
# cases and not the adversarial ones: an artificial 40-character run of a single
# letter still reaches 81 tokens. The old note here recorded "MEASURED on 1,406
# real annotations after the caps: 70 tokens worst case, none over 77". That
# sample figure is SUPERSEDED by the full-corpus measurement in D0-008 §11.4,
# which found one record at 89 true BPE tokens -- so the caps do NOT guarantee
# the budget and the residual gap is real. n06 owns the encoder, so it must
# COUNT tokens and flag an overflow rather than let CLIP truncate in silence.
# That is L1-TEXT-TOKEN-BUDGET, and it is the same principle as
# L1-SEMEDGE-NO-ZEROFILL: a degraded input must be visible, not quiet.
MAX_DESCRIPTION_CHARS = 160
MAX_CATEGORY_CHARS = 40
MAX_MATERIALS = 3

# --- U-14 -----------------------------------------------------------------

# [U-14] Paper 2.3 says each asset is "rendered from 11 orthogonal viewpoints"
# and never states the aggregation. Mean of the 11 per-view embeddings:
#
#   * it uses all eleven, which is what the paper says exist;
#   * 2.1 defines the query as containing "images q_img", PLURAL, so a single
#     view would contradict the paper's own wording on the query side;
#   * it is deterministic, which the gallery requires -- a gallery embedding
#     that changes between runs makes R@k unreproducible.
#
# ULIP-2's released loader picks a RANDOM view per training step
# (dataset_3d.py: random.choice(self.picked_rotation_degrees)). That is evidence
# about ULIP-2, not about MetaFind, and it serves a different objective: view
# augmentation while aligning a 3D encoder. A retrieval gallery cannot be built
# from it. Recorded so the difference is visible rather than assumed away.
IMAGE_AGGREGATION = "mean"

# --- U-11 -----------------------------------------------------------------

# [U-11] Paper 2.6: "Rather than zero-padding, we apply masked embeddings to
# ensure flexibility." That rules out zeros and names no replacement. A learned
# token per modality is the only one of the three candidates this codebase
# implements; the other two are named in KNOWN_MISSING_MODALITY so a protocol
# can record them and be REFUSED rather than silently reinterpreted.
MISSING_MODALITY = "learned_token"

# --- U-22 -----------------------------------------------------------------

# [U-22] The paper gives NONE of these. Every value is ours, and the artifact
# names each one so "the hyperparameters are recorded" means something in the
# report. p_mask is the exception: 2.6 states 30%, so it is a paper constant
# that Table 3 ablates, not a free choice.
DEFAULT_HYPERPARAMETERS = {
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 0.1,
    "scheduler": "cosine",
    "batch_size": 64,
    "epochs": 50,
    "p_mask": PAPER_P_MASK,
    # [C-001] tau. The two halves of this pair do NOT have the same authority and
    # must never be reported as though they did.
    #
    #   init_temperature = 0.5
    #       PAPER FACT. 3experiments.tex:15, verbatim: "The temperature is 0.5
    #       for all experiments." It is the only temperature value MetaFind
    #       states, and nothing in the source contradicts it. `losses.PAPER_TAU`
    #       carries the same number for the training side.
    #
    #   learnable_temperature = False
    #       USER-RATIFIED IMPLEMENTATION CHOICE, resting on a strongly-supported
    #       INFERENCE -- NOT a paper statement, and it must never be reported as
    #       one. MetaFind nowhere states that tau is non-learnable.
    #
    #       What the inference rests on is the authors' OWN vocabulary. They do
    #       use "learnable", and use it precisely: 2methdology.tex:54 types f_h
    #       and f_x as "two learnable functions", and :87 calls lambda "a
    #       learnable scalar". They never apply it to the contrastive
    #       temperature. Both places tau is introduced -- :79 and :99 -- name it
    #       "a temperature hyperparameter". So the paper distinguishes learnable
    #       quantities from hyperparameters in its own words, and puts tau on the
    #       hyperparameter side TWICE. Add 3experiments.tex:15's "The temperature
    #       is 0.5 for all experiments" -- a value that is optimised does not stay
    #       fixed across all experiments -- and the reading is strong.
    #
    #       Strong is not stated. This stays an INFERENCE, ratified by the user
    #       2026-08-21 as the reproduction's implementation choice.
    #
    # The previous values -- 0.07, learnable -- were CLIP's convention, carried in
    # from the wrong source. Under `losses.py:114` they now raise the deviation
    # warning, which is the correct treatment for a value the paper contradicts.
    "init_temperature": 0.5,
    "learnable_temperature": False,
    # Retained: the clamp only binds a LEARNABLE scale, so with the choice above
    # it is inert. It stays because REQUIRED_HYPERPARAMETERS demands the field and
    # because a later run that re-enables learning must not silently lose the clamp.
    "max_logit_scale": 100.0,
    "seed": 20260816,
}

# --- Table 3 ---------------------------------------------------------------

# `fusion: None` means "whatever the main line resolves" -- [U-13] the paper
# lists five fusion strategies and never says which is Full. Table 3 ablates
# Mean and MLPs separately, so Full is none of those two, and n09 resolves it.
VARIANTS = [
    {"variant_id": "full", "table3_row": "MetaFind (Full, bidirectional) w/ iterative & ESSGNN",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    # Same weights, different composition. Retraining would answer a question
    # Table 3 is not asking.
    {"variant_id": "no_iterative", "table3_row": "w/o iterative retrieval",
     "requires_training": False, "reuses_ckpt": "full", "train_scope": None,
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "parallel"},
    {"variant_id": "no_layout", "table3_row": "w/o Layout Context",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": None,
     "composition_mode": "iterative"},
    {"variant_id": "layout_gat", "table3_row": "w/ Layout Context (GAT)",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": "gat",
     "composition_mode": "iterative"},
    {"variant_id": "fusion_mean", "table3_row": "Fusion = Mean",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": "mean", "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    {"variant_id": "fusion_mlp", "table3_row": "Fusion = MLPs",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": "mlp", "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    {"variant_id": "dropout_10", "table3_row": "Modality Dropout = 10%",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": 0.10, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    {"variant_id": "dropout_50", "table3_row": "Modality Dropout = 50%",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": 0.50, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    {"variant_id": "fuser_only", "table3_row": "Train fuser only",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "fuser_only",
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "iterative"},
    # The only variant that changes missing_modality_representation, which is
    # why that field appears on a variant at all.
    {"variant_id": "zero_pad", "table3_row": "Padding missing modalities with 0",
     "requires_training": True, "reuses_ckpt": None, "train_scope": "point_encoder_and_fuser",
     "fusion": None, "dropout": PAPER_P_MASK, "layout_encoder": "essgnn",
     "composition_mode": "iterative", "missing_modality_representation": "zero_pad"},
]


def _cap(text: str, limit: int) -> str:
    """Trim to `limit` at a word boundary, keeping the trailing period if any.

    Mid-word truncation would hand the encoder a fragment ("a wooden dining
    tab"), which tokenises into something unrelated to the word it came from.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "."


def _dim(value: float) -> str:
    """[U-15, D0-008 E-1 + S-1] One decimal, a trailing ``.0`` stripped.

    The previous ``:.0f`` rendered 161 records' stored dimension as ``0``: the
    string asserted something the annotation did not say. E-1 replaced it, and
    S-1 settled that the rule applies at EVERY magnitude rather than only below
    1 cm -- the corpus's single ``2.5`` would otherwise render ``2`` and drop a
    fifth of that dimension.

    Python's format rounds half to even, so a hypothetical ``0.25`` renders
    ``0.2``. No such value exists in this corpus: the complete non-integer
    vocabulary over all 45,952 v3 records is {0.5: 155, 0.2: 7, 2.5: 1}
    (OBSERVED DATA). D0-008 ratified this formatter, not a rounding policy.
    """
    return f"{float(value):.1f}".removesuffix(".0")


def _capitalise(text: str) -> str:
    """[U-15, D0-008 S-2] Upper-case the first character and nothing else.

    ``str.capitalize()`` would lower-case the remainder ("LED lamp" -> "Led
    lamp"). This is not an a/an heuristic and not a vocabulary lookup -- E-2
    forbids both. A leading character with no upper-case form is left as it is.
    """
    return text[:1].upper() + text[1:]


def serialize_annotation(annotation: dict, template: str | None = None) -> str:
    """[U-15, L1-TEXT-SERIALIZATION] One annotation into the string CLIP sees.

    Byte-identical for a given annotation, which is what the golden-string test
    pins. Lists are joined with ", " and the last placement value is joined with
    " or " so the sentence reads as prose rather than as a serialised list --
    the encoder is frozen and was trained on captions.
    """
    # n05's validate_annotation already refuses both of these, so this cannot
    # fire on a sidecar the pipeline wrote. It is here because the alternative
    # failure is SILENT: an empty list renders as "made of ," or "typically
    # placed .", which encodes fine, ranks badly, and looks like nothing is
    # wrong. A guard that depends on a check in another module is a guard that
    # disappears the first time someone calls this function from somewhere else.
    if not annotation["materials"]:
        raise ValueError("`materials` is empty; the serialized string would "
                         "be malformed rather than merely short")

    materials = ", ".join(annotation["materials"][:MAX_MATERIALS])
    description = _cap(annotation["description"].strip(), MAX_DESCRIPTION_CHARS)
    if description and not description.endswith("."):
        description += "."
    # Resolved HERE, not in the signature. A default argument is bound once at
    # definition time, so `template=TEXT_TEMPLATE` would keep serializing with
    # the template this module had at import even after TEXT_TEMPLATE changed --
    # and text_serialization_id(), which exists to detect exactly that drift,
    # would have gone on reporting the old identity.
    return (template or TEXT_TEMPLATE).format(
        description=description,
        category=_capitalise(
            _cap(annotation["category"], MAX_CATEGORY_CHARS).rstrip(".")),
        materials=materials,
        width=_dim(annotation["width"]),
        length=_dim(annotation["length"]),
        height=_dim(annotation["height"]),
        placement=placement_phrase(annotation),
    )


def placement_phrase(annotation: dict) -> str:
    """The four booleans as one clause.

    An all-false annotation is a real answer, not an error: an abstract shape
    belongs nowhere in particular. It gets NO_PLACEMENT_PHRASE rather than being
    rejected, which is the opposite of v1, where the schema demanded a positive
    answer and `unconstrained` absorbed 30.7% of the corpus.

    Combinations outside PLACEMENT_PHRASES fall back to joining the individual
    phrases, so a new combination reads as prose instead of raising.
    """
    on = tuple(f for f in ("onCeiling", "onWall", "onFloor", "onObject")
               if annotation.get(f))
    if not on:
        return NO_PLACEMENT_PHRASE
    if (phrase := PLACEMENT_PHRASES.get(on)) is not None:
        return phrase
    if (phrase := PLACEMENT_PHRASES.get(tuple(sorted(on)))) is not None:
        return phrase
    parts = [PLACEMENT_PHRASES[(f,)] for f in on]
    return parts[0] + " or ".join([""] + [p.split(" ", 2)[-1] for p in parts[1:]])


# [B-3, D0-008 §11.2] The probes the cache identity is computed from. They are
# fixed synthetic annotations, not corpus records, so the identity does not move
# when the data does.
#
# A SUITE rather than one probe, after adversarial review: a single probe left
# most of the module invisible to the hash. Changing PLACEMENT_PHRASES[("onFloor",)]
# moved every floor-standing record's string while text_serialization_id() sat
# still, so the protocol would have kept certifying a serializer that no longer
# existed. The suite covers, by construction:
#
#   * NO_PLACEMENT_PHRASE (all four flags false);
#   * EVERY key in PLACEMENT_PHRASES -- so adding, editing or deleting one moves
#     the identity, including R-3's deleted ("onWall","onCeiling") entry;
#   * two UNMAPPED combinations, which take the fallback join;
#   * a description past MAX_DESCRIPTION_CHARS and a category past
#     MAX_CATEGORY_CHARS, so both caps are exercised;
#   * four materials against MAX_MATERIALS;
#   * an integer, a >1 fractional, a <1 fractional and a zero dimension, so the
#     S-1 formatter is exercised at every magnitude it distinguishes;
#   * a lower-case and an upper-case-run category, so S-2 is exercised.
_PROBE_BASE = {
    "category": "led wall lamp",
    "description": ("a matte white fixture with a perforated shade and a short "
                    "articulated arm, mounted flush against the surface it "
                    "hangs from, finished in a fine powder coat that scatters "
                    "the light evenly across the wall behind it"),
    "width": 30.0, "length": 2.5, "height": 0.5,
    "materials": ["metal", "plastic", "glass", "rubber"],
    "onCeiling": False, "onWall": False, "onFloor": False, "onObject": False,
}


def serialization_probes() -> list[dict]:
    """The fixed inputs whose emitted strings define the cache identity."""
    flag_sets = [
        (),                                                # NO_PLACEMENT_PHRASE
        *sorted(PLACEMENT_PHRASES),                        # every mapped case
        ("onCeiling", "onWall"),                           # fallback join (R-3)
        ("onCeiling", "onWall", "onFloor", "onObject"),    # longest fallback
    ]
    probes = [dict(_PROBE_BASE, **{f: True for f in flags}) for flags in flag_sets]
    probes.append(dict(_PROBE_BASE, onFloor=True,
                       category="a very long category name " * 3))
    probes.append(dict(_PROBE_BASE, onObject=True, category="LED",
                       width=1000.0, length=1.0, height=0.0))
    return probes


def serialization_contract() -> dict:
    """Every constant the emitted string depends on, in one canonical mapping.

    The probe suite alone is a SAMPLE, and a sample cannot cover a continuous
    parameter: adversarial review round 2 showed that moving
    MAX_DESCRIPTION_CHARS from 160 to 161 changed a real corpus record
    (020a2199c72a4f8eaea8f1212271a1b0 gained a word) while every probe still
    truncated at the same word boundary, so the identity did not move and
    load_protocol() went on accepting the old protocol.

    Hashing the constants closes that: any edit to a declared constant moves the
    identity whether or not a probe happens to notice. The probes remain, because
    the constants cannot cover the CODE -- _dim(), _capitalise(), _cap() and
    placement_phrase() are logic, not values.
    """
    return {
        "family": TEXT_SERIALIZATION_FAMILY,
        "template": TEXT_TEMPLATE,
        "max_description_chars": MAX_DESCRIPTION_CHARS,
        "max_category_chars": MAX_CATEGORY_CHARS,
        "max_materials": MAX_MATERIALS,
        "placement_phrases": {"+".join(k): v
                              for k, v in sorted(PLACEMENT_PHRASES.items())},
        "no_placement_phrase": NO_PLACEMENT_PHRASE,
    }


def serialization_id_for(serializer) -> str:
    """[B-3] The cache identity: family name + hash of what `serializer` EMITS.

    A hand-maintained version string is not a cache identity. The retired
    ``"metafind_v1_natural"`` proved it -- it labelled both the metre-based v1
    template and the centimetre one, so 5,276 sidecars carry a name that does
    not identify the transformation that produced them.

    Hashing emitted strings instead makes the identity content-addressed: it
    moves when the template, the dimension formatter, the capitalisation, either
    character cap, the materials cap, or any placement phrase moves, whether or
    not anyone remembers to bump a number.

    It takes the serializer as an argument so a CALLER can bind the identity to
    the exact callable it will execute, rather than to whatever this module
    happens to hold. n06 does that: it passes its own imported alias, so the
    protocol certifies the function that will actually run.
    """
    payload = json.dumps(
        {"contract": serialization_contract(),
         "emitted": [serializer(probe) for probe in serialization_probes()]},
        sort_keys=True, ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{TEXT_SERIALIZATION_FAMILY}@{digest[:16]}"


def text_serialization_id() -> str:
    """The identity of THIS module's serializer. `load_protocol()` in n06 refuses
    to run unless the protocol's recorded identity equals the caller's (B-2)."""
    return serialization_id_for(serialize_annotation)


# The reading travels with the decision, because the report has to state it in
# exactly these terms and NOT as "MetaFind explicitly states that OpenCLIP is
# frozen" -- MetaFind never writes that sentence. What it writes is that it
# builds on ULIP-2; ULIP-2 3.3 is what states the freeze.
CLIP_SCOPE_BASIS = {
    "frozen": (
        "MetaFind does not state a per-module freeze scope for OpenCLIP "
        "anywhere. This reproduction follows ULIP-2's design because MetaFind "
        "explicitly builds on ULIP-2, ULIP-2 3.3 states 'We adopt the largest "
        "version of encoders from OpenCLIP (ViT-G/14) ... and freeze it during "
        "pre-training', and MetaFind never states that it changes that policy. "
        "The three sentences that read otherwise do not carry it: 2.6's 'Both "
        "query and gallery encoders are trained' describes the TOWERS, which "
        "train under a frozen CLIP because the point encoder, the projection "
        "and the fuser are all in the optimizer; 3.4's 'Fine-tuning the entire "
        "encoder outperformed training the fuser only' separates the full model "
        "from the fuser-only ablation, a separation the trainable point encoder "
        "already produces; and 2.4's 'gallery encoder is frozen after "
        "pretraining' versus 2.6's 'is trained' is the Stage 1 / Stage 2 "
        "boundary, not a Stage 1 contradiction. Reopen if MetaFind's code or "
        "its authors show an optimizer update reaching OpenCLIP's parameters."
    ),
    "trainable": (
        "Read from MetaFind 3.4, 'Fine-tuning the entire encoder outperformed "
        "training the fuser only', taking 'the entire encoder' to reach "
        "OpenCLIP. MetaFind never states this per module; recorded as our "
        "reading. Under this reading a frozen run IS deviation D-1."
    ),
}


def build_protocol(paper_clip_train_scope: str, actual_clip_train_scope: str,
                   decided_by: str, confidence: str = "moderate") -> dict:
    if paper_clip_train_scope not in ("frozen", "trainable"):
        raise ValueError(f"paper_clip_train_scope must be frozen|trainable, "
                         f"got {paper_clip_train_scope!r}")
    if actual_clip_train_scope not in ("frozen", "trainable"):
        raise ValueError(f"actual_clip_train_scope must be frozen|trainable, "
                         f"got {actual_clip_train_scope!r}")
    if IMAGE_AGGREGATION not in PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS:
        raise ValueError(f"unknown image_aggregation {IMAGE_AGGREGATION!r}")
    if MISSING_MODALITY not in KNOWN_MISSING_MODALITY:
        raise ValueError(f"unknown missing_modality {MISSING_MODALITY!r}")
    return {
        "status": "resolved",
        # [B-2/B-3] The identity is content-addressed and n06 re-derives it from
        # its own imported serializer before encoding anything; the template and
        # the probe string travel with it so the artifact DESCRIBES the encoder
        # instead of merely naming it.
        "text_serialization": text_serialization_id(),
        "text_serialization_family": TEXT_SERIALIZATION_FAMILY,
        "text_serialization_contract": serialization_contract(),
        "text_template": TEXT_TEMPLATE,
        "text_serialization_probes": [serialize_annotation(p)
                                      for p in serialization_probes()],
        "image_aggregation": IMAGE_AGGREGATION,
        "paper_clip_train_scope": paper_clip_train_scope,
        "paper_clip_train_scope_basis": CLIP_SCOPE_BASIS[paper_clip_train_scope],
        "paper_clip_train_scope_confidence": confidence,
        "actual_clip_train_scope": actual_clip_train_scope,
        "missing_modality_representation": MISSING_MODALITY,
        "decided_by": decided_by,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def build_hyperparameters(decided_by: str, overrides: dict | None = None) -> dict:
    values = dict(DEFAULT_HYPERPARAMETERS)
    values.update(overrides or {})
    missing = [f for f in REQUIRED_HYPERPARAMETERS if f not in values]
    if missing:
        raise ValueError(f"hyperparameters missing {', '.join(missing)}")
    return {
        "uri": str(HYPERPARAMETERS_PATH),
        "sha256": canonical_hyperparameter_hash(values),
        "values": values,
        "decided_by": decided_by,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def _write(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record the Stage 1 encoding decisions. "
                    "--paper-clip-train-scope has no default on purpose.")
    ap.add_argument("--paper-clip-train-scope", required=True,
                    choices=("frozen", "trainable"),
                    help="U-34: OUR READING of what MetaFind requires. Not what "
                         "this run does -- see --actual-clip-train-scope. The "
                         "gap between the two is deviation D-1.")
    ap.add_argument("--actual-clip-train-scope", default="frozen",
                    choices=("frozen", "trainable"),
                    help="U-34: what this run does. Defaults to frozen, which "
                         "ULIP-2 3.3 supports and 24 GB requires.")
    ap.add_argument("--confidence", default="moderate",
                    choices=("low", "moderate", "high"),
                    help="how firm the U-34 reading is; the report quotes it")
    ap.add_argument("--decided-by", default=None,
                    help="who made the call; defaults to the invoking user")
    args = ap.parse_args()

    decided_by = args.decided_by or getpass.getuser()

    with runlog.run_progress(NODE):
        protocol = build_protocol(args.paper_clip_train_scope,
                                  args.actual_clip_train_scope, decided_by,
                                  args.confidence)
        hyperparameters = build_hyperparameters(decided_by)
        _write(PROTOCOL_PATH, protocol)
        _write(HYPERPARAMETERS_PATH, hyperparameters)
        _write(VARIANT_REGISTRY_PATH, VARIANTS)

    deviating = (args.paper_clip_train_scope == "trainable"
                 and args.actual_clip_train_scope == "frozen")
    print(f"stage1_encoding_protocol resolved by {decided_by}")
    print(f"  text_serialization  {text_serialization_id()}")
    print(f"  image_aggregation   {IMAGE_AGGREGATION}")
    print(f"  missing_modality    {MISSING_MODALITY}")
    print(f"  paper CLIP scope    {args.paper_clip_train_scope}")
    print(f"  actual CLIP scope   {args.actual_clip_train_scope}")
    print(f"  D-1 active          {deviating}"
          + ("  <- must be declared as a deviation in the report" if deviating else ""))
    print(f"  hyperparameters     sha256 {hyperparameters['sha256'][:16]}")
    print(f"  variants            {len(VARIANTS)} (Table 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
