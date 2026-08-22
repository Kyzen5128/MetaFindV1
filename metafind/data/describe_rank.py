"""Rank description candidates by CLIP image-text similarity. ULIP-2's method.

# SUPPORTS-NODE: n05_annotate

[UPSTREAM FACT -- `ulip2_source/main.tex:677`] ULIP-2 generates several
descriptions per asset **independently**, ranks them with **CLIP-ViT-Large**
image-text similarity, and keeps the top one. Its own ablation
(`main.tex:1328`) is what settled top-1:

    top-k captions used     1      3      4      10
    accuracy             69.7   66.7   66.4   66.3

Two places where this deliberately differs from upstream, both recorded rather
than smoothed over:

**Five candidates, not ten.** `E-10`. ULIP-2 generates ten with BLIP-2-opt6.7B;
ten from a 27B model over 45,955 assets is not affordable here. **A cost
choice, not an upstream figure** -- and the ablation above compares how many to
USE, never how many to GENERATE, so "is five enough" is genuinely unmeasured.
Every candidate's score is stored so that question can be answered from the
data instead of assumed.

**Scored against all eleven views, meaned.** ULIP-2 captions one image at a
time and scores each caption against its own image, because BLIP-2 takes one
image. Ours describes the whole asset from eleven views at once -- measured
2026-08-23 on five assets, per-view captioning called one shovel a shovel, a
hand trowel, a chisel and an axe across its eleven views, while the whole-asset
pass named it correctly with more detail. So there is no single "its own image"
to score against, and the mean over the views is what asks the question the
description is actually making: does this hold from every angle. Taking the max
instead would let a sentence win by being true of one view.

The ranking model must NOT be the model `n06` encodes with (`U-M`). ULIP-2
ranked with `CLIP-ViT-Large` and trained with something else; ranking and
encoding with one model scores a description on how well it suits the encoder
rather than on how well it suits the image.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

# [UPSTREAM FACT -- `main.tex:677` names CLIP-ViT-Large by name.] The OpenAI
# release is the one that paper cites (`radford2021learning`).
RANKER_MODEL = "openai/clip-vit-large-patch14"

# `E-10`. Five is the cost choice; the scores of all five are recorded.
N_CANDIDATES = 5

RANKER_VERSION = 1


@functools.lru_cache(maxsize=1)
def _ranker(device: str = "cuda"):
    """Load once per process. Held for the life of the run, like the annotator."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(RANKER_MODEL, dtype=torch.float32).to(device)
    model.eval()
    return model, CLIPProcessor.from_pretrained(RANKER_MODEL), device


def score_candidates(view_paths: list[str], candidates: list[str],
                     device: str = "cuda") -> list[float]:
    """Mean CLIP cosine similarity of each candidate against every view.

    Returned in the order given, so a caller can keep candidate `i` aligned with
    score `i` without a second pass.
    """
    import torch
    from PIL import Image

    if not candidates:
        return []
    model, processor, dev = _ranker(device)
    images = [Image.open(p).convert("RGB") for p in view_paths]

    with torch.no_grad():
        # `truncation=True` because CLIP's text encoder stops at 77 tokens and a
        # two-sentence description can exceed it. Silently dropping the tail is
        # the documented behaviour and it applies equally to every candidate, so
        # it cannot favour one -- but it does mean a long description is scored
        # on its opening, which is worth knowing when reading the numbers.
        text_inputs = processor(text=candidates, return_tensors="pt",
                                padding=True, truncation=True).to(dev)
        image_inputs = processor(images=images, return_tensors="pt").to(dev)
        # transformers 5 returns a ModelOutput here, not a bare tensor. Taking
        # `.pooler_output` off whatever comes back keeps this working on both.
        t = model.get_text_features(**text_inputs)
        v = model.get_image_features(**image_inputs)
        t = getattr(t, "pooler_output", t)
        v = getattr(v, "pooler_output", v)
        t = t / t.norm(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
        sim = (t @ v.T).mean(dim=1)          # (candidates, views) -> per candidate
    return [float(x) for x in sim.cpu().numpy()]


def rank(view_paths: list[str], candidates: list[str],
         device: str = "cuda") -> tuple[str, list[dict]]:
    """``(winner, [{"text", "clip_score", "rank"}, ...])``.

    Every candidate is returned with its score, not just the winner: `E-10`
    promises the spread is recorded so "would ten have been better than five"
    is answerable later. Keeping only the winner would throw that away on every
    asset and it cannot be recomputed without re-running the model.
    """
    scores = score_candidates(view_paths, candidates, device)
    order = np.argsort(-np.asarray(scores))
    ranked = [{"text": candidates[i], "clip_score": scores[i], "rank": r}
              for r, i in enumerate(order)]
    return candidates[int(order[0])], ranked


def demo() -> None:
    """Does the ranker prefer the description that is actually true of the image?

    Expected truth is a synthetic image whose content is not in question -- a
    solid red square is red, and no model output decides that. A ranker that
    cannot separate "a solid red square" from "a photograph of a green forest"
    is not ranking anything, and would silently pick candidate 0 forever.
    """
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "red.png"
        Image.new("RGB", (224, 224), (220, 20, 20)).save(p)
        winner, ranked = rank([str(p)], [
            "a photograph of a dense green forest",
            "a solid red square",
            "a bowl of white rice",
        ])
        assert winner == "a solid red square", [r["text"] for r in ranked]
        assert ranked[0]["clip_score"] > ranked[-1]["clip_score"]
        print(f"demo ok: {[(r['text'][:24], round(r['clip_score'], 4)) for r in ranked]}")


if __name__ == "__main__":
    demo()
