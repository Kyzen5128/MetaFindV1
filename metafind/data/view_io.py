"""Read a rendered view. The ONE place that decides what transparency becomes.

# SUPPORTS-NODE: n05_annotate, n06_encode_text_image

n04 writes RGBA with a genuinely transparent background, because that is what
OpenShape's renderer produces (`render_single_glb.py`: `film_transparent = True`,
`set_output_format(enable_transparency=True)`). Transparency is a *storage*
property -- every consumer that feeds pixels to a model has to flatten it onto
some colour first, and that colour is a research-relevant choice: a VLM
describes a dark object on black differently from the same object on white.

[USER DECISION, 2026-08-23] **Black.**

Why this lives in one function
------------------------------

Three call sites consume views -- the annotator (`annotate_run`), the CLIP
ranker (`describe_rank`) and the encoder (`encode_text_image`). Each of them
previously did its own ``Image.open(p).convert("RGB")``.

That is not a neutral default. **PIL's RGBA->RGB conversion drops the alpha
channel and keeps whatever RGB sits underneath**, which for a transparent
Blender pixel is `(0, 0, 0)` -- so it happens to *look* like a black flatten
while being an accident rather than a decision, and it is wrong wherever a
pixel is partially transparent: a 50%-alpha white edge pixel should come out
mid-grey against black and instead comes out white.

Anti-aliased silhouettes are exactly where partial alpha lives, so this is not
a corner case: it is the outline of every object in the corpus.

One function, one composite, one recorded constant. If the background decision
ever changes, it changes here and nowhere else, and no two nodes can disagree
about what the model saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["BACKGROUND_RGB", "VIEW_IO_VERSION", "image_identity",
           "load_view_rgb", "load_views_rgb"]

# [USER DECISION `U-BG`, 2026-08-23] The colour a transparent render pixel
# becomes. Recorded as a constant rather than spelled `(0, 0, 0)` at three call
# sites, so a grep for the decision finds one answer.
BACKGROUND_RGB: tuple[int, int, int] = (0, 0, 0)

# Bumped whenever the composite rule changes, so a cached embedding can be told
# apart from one taken under a different background.
VIEW_IO_VERSION = 1


def image_identity(render_rec: dict) -> str:
    """What a downstream artifact's IMAGES were actually taken from.

    [ADDED 2026-08-24] n05 and n06 both decided "already done" without looking
    at the renders at all. The OptiX swap (`RENDERER_VERSION` 5 -> 6) re-rendered
    the corpus with different pixels, and every existing annotation and embedding
    would have stayed "complete" while describing images that had been deleted --
    no error, no warning, and the same labels on both halves.

    The digest binds the renderer generation, the composite rule in THIS module,
    and n04's twelve per-view content hashes, so it also catches a re-render that
    kept the version number. It lives here because this module is already the one
    place that decides what a consumer sees.

    An n04 record with no `view_sha256` cannot say what it rendered, and returns
    "" -- which every caller must treat as a reason to redo the work, never as a
    reason to accept it.
    """
    import hashlib

    digests = render_rec.get("view_sha256")
    if not digests:
        return ""
    h = hashlib.sha256()
    h.update(f"r{render_rec.get('renderer_version')}|v{VIEW_IO_VERSION}|".encode())
    for d in digests:
        h.update(f"{d}|".encode())
    return h.hexdigest()[:16]


def load_view_rgb(path: str | Path):
    """One rendered view as an opaque ``PIL.Image`` in RGB.

    Alpha is composited over ``BACKGROUND_RGB`` with the standard
    source-over rule, ``out = fg * a + bg * (1 - a)``, rather than discarded.

    A view that is already RGB (an older render, or a format without alpha) is
    returned unchanged -- there is nothing to composite and inventing an alpha
    would be a claim about pixels we do not have.
    """
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGBA":
        return im.convert("RGB")
    bg = Image.new("RGBA", im.size, (*BACKGROUND_RGB, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def load_views_rgb(paths):
    """``load_view_rgb`` over a sequence, in order."""
    return [load_view_rgb(p) for p in paths]


def demo() -> None:
    """The composite must be arithmetic, not alpha-dropping.

    Expected truth is fixed by the source-over formula, not by any model
    output: a 50%-alpha white pixel over black is exactly 128 (PIL rounds
    ``255 * 128/255`` to 128), and a fully transparent pixel is exactly the
    background. ``convert("RGB")`` returns 255 for the first of those, which is
    the bug this module exists to remove -- so the demo asserts the difference
    rather than only asserting the correct value.
    """
    import tempfile

    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "half.png"
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        a[..., :3] = 255           # white foreground
        a[0, 0, 3] = 255           # opaque
        a[0, 1, 3] = 128           # half transparent
        a[0, 2, 3] = 0             # fully transparent
        Image.fromarray(a, mode="RGBA").save(p)

        out = np.asarray(load_view_rgb(p))
        assert tuple(out[0, 0]) == (255, 255, 255), out[0, 0]
        assert tuple(out[0, 1]) == (128, 128, 128), out[0, 1]
        assert tuple(out[0, 2]) == BACKGROUND_RGB, out[0, 2]

        naive = np.asarray(Image.open(p).convert("RGB"))
        assert tuple(naive[0, 1]) == (255, 255, 255), (
            "convert('RGB') is expected to drop alpha and keep the white; if it "
            "no longer does, this module's rationale needs re-checking")

        # An RGB view must survive untouched.
        q = Path(d) / "opaque.png"
        Image.fromarray(np.full((4, 4, 3), 77, dtype=np.uint8)).save(q)
        assert np.asarray(load_view_rgb(q)).tolist() == np.full((4, 4, 3), 77).tolist()

    print("view_io demo ok: 255 / 128 / background, and convert('RGB') gives 255 -- OK")


if __name__ == "__main__":
    demo()
