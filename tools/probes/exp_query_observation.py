#!/usr/bin/env python3
"""Does the query's observation change Table 1's shape? Measure it.

[KYZEN 2026-09-03] He asked for experiments on the unresolved settings, not for
another report saying they are unresolved. 問題 4 and 問題 6 are exactly that:
the paper never states what the query observes, and `metafind/data/observation.py`
was built to express the alternatives and then never run.

This runs it. One checkpoint, one gallery, one set of queries; the ONLY thing
that varies is which image the query draws:

    same_mean       the 12-view mean, i.e. the query sees the gallery's own
                    image vector -- what every number so far was measured under
    single_view     one view, chosen by uid_seed
    held_out_view   the same single view, named for what it is on the query side
    disjoint_views  the mean of the other eleven, so query and gallery share no
                    view at all

`disjoint_views` is the interesting one: it is the closest thing this corpus can
express to "the query is a different observation of the same asset", without
re-rendering anything and without a query pack.

NO BACKBONE IS LOADED. Text and image are frozen, so their vectors come from
n06's cache -- exactly what the towers saw. That is what lets this run on CPU
beside a training job, and it is also why the point-cloud conditions are absent:
PointBERT is trainable, its output is not cached, and encoding 4,569 clouds
through it is not free. The four conditions here are the ones the image
observation actually moves.

[SENSITIVITY, NOT A PROTOCOL COMPARISON] One checkpoint evaluated four ways is
inference sensitivity. §十六 is explicit that it cannot stand in for training
under a different observation -- that needs a retrain. What it CAN say is
whether the saturation is a property of the observation or of the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metafind import paths
from metafind.data.observation import view_indices

CONDITIONS = {"text": ("text",), "image": ("image",),
              "text+image": ("text", "image")}
POLICIES = ("same_mean", "single_view", "held_out_view", "disjoint_views")


def load_tower(ckpt_path: Path, device: str):
    """The two fusion heads from a Stage 1 checkpoint. No backbone."""
    from metafind.train.stage1 import build_model, load_protocols

    encoding, training, hyperparameters = load_protocols()
    model, _loss = build_model(encoding, training, hyperparameters)
    model = model.to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["tower_trainable_state"],
                                                strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    gap = trainable - set(ck["tower_trainable_state"])
    if gap:
        raise SystemExit(f"the checkpoint does not cover {sorted(gap)[:4]}")
    if unexpected:
        raise SystemExit(f"unexpected keys in the tower state: {unexpected[:4]}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# RETRACTED 2026-09-03. Everything this module measured is withdrawn.
#
# `gallery_vectors` read only the cached text and image vectors, and `recall`
# then called the gallery tower with `"pc": torch.zeros_like(g_text)`. A zero
# tensor of the right shape is not an absent modality -- the fusion Transformer
# received it as a PRESENT third slot -- so the gallery was
#
#     Fusion_G(text, image, 0)      instead of      Fusion_G(text, image, e_pc)
#
# which contradicts sec. 2.4, verbatim: "The gallery encoder is
# modality-complete and frozen after pretraining." The repo's own derangement
# experiment had already measured this gallery to be PC-DOMINANT (deranging the
# gallery pc drops text R@1 from 66.88 to 1.80), so zeroing pc removed the
# strongest signal in it.
#
# `recall` also diverged from the production scorer twice more:
#   - ties counted FOR the model (`(sims > own).sum() + 1`), where
#     `metafind.eval.retrieval.rank_of_target` counts `higher + tied + 1`;
#   - float32 Torch GEMM, where `normalize_for_scoring` mandates float64.
#
# Found independently the same day from the ULIP2 side and by Kyzen reading the
# repo. The replacement, `tools/probes/exp_text_length.py`, scores with
# `metafind.eval.retrieval` and builds its gallery through `split_embeds`; its
# canonical arm reproduces `run_retrieval.py` to the printed digit on protocol C
# (78.4/95.0/92.1/98.8/99.9/98.7/100.0) and protocol D
# (58.0/84.6/78.8/96.5/99.6/94.1/100.0). That parity is the entry condition for
# every sensitivity experiment from now on.
#
# `load_tower` is UNAFFECTED and is still imported by the replacement.
# ---------------------------------------------------------------------------
_RETRACTED = (
    "tools/probes/exp_query_observation.py is retracted: its gallery passed a "
    "ZERO point cloud to a modality-complete gallery tower, and its recall "
    "counted ties in the model's favour in float32. Use "
    "tools/probes/exp_text_length.py, which scores with metafind.eval.retrieval "
    "and matches run_retrieval.py exactly on the canonical arm."
)


def gallery_vectors(*_a, **_k):
    """RETRACTED. Read only text and image, so every caller scored against a
    gallery missing its point cloud. Body removed: it must not be copied."""
    raise SystemExit(_RETRACTED)


def vectors(uids: list[str], policy: str, device: str):
    """Query and gallery vectors for one image policy. Cache only."""
    text, q_img, g_img = [], [], []
    for uid in uids:
        z = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        text.append(z["text"].astype(np.float32))
        g_img.append(z["image"].astype(np.float32))   # the gallery is unchanged
        views = z["views"].astype(np.float32)
        if policy == "same_mean":
            # NOT views.mean(): the stored `image` is the fp16 mean n06 wrote,
            # and recomputing it in fp32 differs in the last bits from every
            # number measured so far. Same bytes or it is a different arm.
            q_img.append(z["image"].astype(np.float32))
        else:
            idx = view_indices(policy, uid, views.shape[0])
            q_img.append(views[idx].mean(axis=0))
    t = torch.from_numpy(np.stack(text)).to(device)
    return t, torch.from_numpy(np.stack(q_img)).to(device), \
        torch.from_numpy(np.stack(g_img)).to(device)


def recall(*_a, **_k):
    """RETRACTED. Passed `"pc": torch.zeros_like(g_text)` to a modality-complete
    gallery tower, counted ties FOR the model, and scored in float32. Body
    removed: it must not be copied."""
    raise SystemExit(_RETRACTED)


def main(*_a, **_k):
    """RETRACTED. See the module note."""
    raise SystemExit(_RETRACTED)


if __name__ == "__main__":
    raise SystemExit(main())
