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

TEXT_SERIALIZATION = "metafind_v1_natural"

# [U-15] Paper 2.3 names the fields and gives no format. Pinned here and held by
# a golden-string test (L1-TEXT-SERIALIZATION), because reordering two fields
# silently changes every text embedding and therefore four of Table 1's columns.
#
# Natural prose rather than "Category: chair. Material: wood." because the frozen
# CLIP text tower was trained on captions, not on labelled records -- and the
# tower is frozen, so it cannot adapt to a format it never saw. The 77-token
# limit is the other constraint: a labelled multi-line record spends its budget
# on field names.
#
# Field order follows paper 2.3's own sentence: "object category, size
# dimensions, materials, and placement constraints", with the free-text
# description first because it is the part a caption-trained encoder reads best.
TEXT_TEMPLATE = (
    "{description} A {category} made of {materials}, "
    "roughly {length:.2f} by {width:.2f} by {height:.2f} metres, "
    "typically placed {placement}."
)

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
# So EVERY variable-length part is bounded, not just the description: capping
# the description alone still let a pathological annotation reach 82 tokens
# (long category + four materials + three placement values). The annotation
# keeps its full text on disk; only the string handed to the frozen encoder is
# bounded, and L1-TEXT-SERIALIZATION pins the result.
#
# Dropping the fourth material is the right thing to lose. The annotation prompt
# asks for materials "most prominent first", so a prefix is principled -- and
# the alternative is CLIP dropping the placement constraint instead, silently.
#
# These are CHARACTER caps and CLIP counts TOKENS, so they bound the realistic
# cases and not the adversarial ones: an artificial 40-character run of a single
# letter still reaches 81 tokens. MEASURED on 1,406 real annotations after the
# caps: 70 tokens worst case, none over 77, and real categories run 6 characters
# median / 29 worst. The residual gap is n06's to close -- it owns the encoder,
# so it must COUNT tokens and flag an overflow rather than let CLIP truncate in
# silence. That is L1-TEXT-TOKEN-BUDGET, and it is the same principle as
# L1-SEMEDGE-NO-ZEROFILL: a degraded input must be visible, not quiet.
MAX_DESCRIPTION_CHARS = 160
MAX_CATEGORY_CHARS = 40
MAX_MATERIALS = 3
MAX_PLACEMENT = 2

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
    # CLIP's own convention: a learnable temperature initialised at 0.07 and
    # clamped so the logit scale cannot run away.
    "init_temperature": 0.07,
    "learnable_temperature": True,
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


def serialize_annotation(annotation: dict, template: str = TEXT_TEMPLATE) -> str:
    """[U-15, L1-TEXT-SERIALIZATION] One annotation into the string CLIP sees.

    Byte-identical for a given annotation, which is what the golden-string test
    pins. Lists are joined with ", " and the last placement value is joined with
    " or " so the sentence reads as prose rather than as a serialised list --
    the encoder is frozen and was trained on captions.
    """
    dims = annotation["dimensions"]
    materials = ", ".join(annotation["materials"][:MAX_MATERIALS])
    placement = annotation["placement_constraints"][:MAX_PLACEMENT]
    placement_text = (placement[0].replace("_", " ") if len(placement) == 1
                      else " or ".join(p.replace("_", " ") for p in placement))
    description = _cap(annotation["description"].strip(), MAX_DESCRIPTION_CHARS)
    if description and not description.endswith("."):
        description += "."
    return template.format(
        description=description,
        category=_cap(annotation["category"], MAX_CATEGORY_CHARS).rstrip("."),
        materials=materials,
        length=dims["length_m"], width=dims["width_m"], height=dims["height_m"],
        placement=placement_text,
    )


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
        "text_serialization": TEXT_SERIALIZATION,
        "text_template": TEXT_TEMPLATE,
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
    print(f"  text_serialization  {TEXT_SERIALIZATION}")
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
