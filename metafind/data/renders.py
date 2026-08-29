"""Render 12 views per Objaverse asset, with Blender.

# IMPLEMENTS-NODE: n04_render_views

The 46,052 in the incident notes below is the manifest n04 was HANDED, not what
it produced: ``logs/renders_index.jsonl`` holds 46,024, and the corpus that
survives annotation and reaches a gallery is 45,692 (derived in
``data/splits.py``, which carries the arithmetic). Those figures describe runs,
so they stay as measured; none of the three is a stale copy of another.

[RENDERER_VERSION 5, 2026-08-23] **This node no longer renders anything itself.**
It orchestrates; `metafind.data.render_blender` invokes OpenShape's own
`render_single_glb.py`, vendored byte-identical at
`metafind/vendor/openshape/`. Read that module's docstring for why Blender, and
what the two-line patch is.

What this file still owns: which assets are due, the per-asset sidecar, the
derived index, quarantine, resume, and the checks that a produced asset is
usable. What it no longer owns: cameras, lighting, projection, framing.

Why the change
--------------

**No rendering code exists anywhere in the ULIP lineage.** ULIP's repository has
none. ULIP-2's paper gives one sentence (`main.tex:677`). MetaFind's gives one
clause (`2methdology.tex:28`, "rendered from 11 orthogonal viewpoints"). The
only executable renderer in the chain is OpenShape's, which ULIP-2's appendix
defers to for 3D input preprocessing and whose released `.npy` schema the ULIP-2
Objaverse triplets reproduce key for key.

Three consequences, all recorded rather than smoothed over:

* **12 views, not 11.** MetaFind states eleven; OpenShape's layout is three
  polar rings of four and eleven does not divide into three rings. USER decision
  2026-08-23, a registered DEVIATION from a PAPER FACT.
* **The rings see the underside.** phi = 120 deg looks UP at the asset. The
  retired single 20 deg ring never did, and MetaFind's own Figure 2 shows views
  from below.
* **Transparent RGBA on disk.** `film_transparent = True` is upstream's. What
  transparency BECOMES is a consumer's decision and lives in
  `metafind.data.view_io`, once, so the annotator and the encoder cannot
  disagree about what the model saw.

Retired, but not deleted
------------------------

Everything from `N_VIEWS` down to `normalised_scene` describes the pyrender path
and is dead as far as `process_one` is concerned. It stays because
`tools/verify_renders_against_ulip.py` and the S-1/S-4 measurement scripts still
import `azimuth_orbit_directions` and `normalised_scene`, and because deleting
the fitted constants would delete the record of how they were arrived at. **No
value below is read by a v5 render.** The sidecar records Blender's parameters,
read back off the vendored script, not these.

Scale is destroyed on purpose, and recorded on the way past
--------------------------------------------------------

Blender normalises each asset (longest bbox side to 0.8, bbox centre to the
origin), which is what makes L1-RENDER-SCALE-INVARIANT hold: a millimetre- and a
metre-scaled copy of one mesh must render identically or the image tower learns
the modelling units of whoever uploaded the asset. The cost is that absolute
size is gone, so the annotator's "size dimensions" can only be a category prior.
`raw_bbox_extents` is written alongside so that estimate stays auditable -- it
is the axis-aligned bounding box in the file's own units, not a verified
physical size.

*Completion means the sidecar, not the pixels.* Blender stages into a temp
directory and its output is moved in only once all twelve views exist, then the
record is written last. A crash costs one re-render rather than leaving an asset
that is skipped forever with no metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
# The submodule, explicitly: `cf.process` is lazily bound in 3.11 and
# raises AttributeError until something touches `cf.ProcessPoolExecutor`.
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog

# Must be set before pyrender imports OpenGL. EGL renders on the GPU with no
# display; OSMesa is not installed here and pyglet needs a window.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

NODE = "n04_render_views"
# [RETIRED with the pyrender path] The live view count is
# `render_blender.N_VIEWS` (12) and is imported as `LIVE_N_VIEWS` below. This 11
# is MetaFind's stated number and stays as the reference the DEVIATION is
# measured against; `azimuth_orbit_directions` and `fibonacci_directions` still
# default to it because the verification tools reproduce the retired layout.
N_VIEWS = 11
RESOLUTION = 224  # U-04
PROJECTION = "orthographic"  # U-03a -- implementation choice, not an inference
CAMERA_LAYOUT = "ulip2_azimuth_orbit_11"  # U-03 -- upstream-informed choice

# [CORRECTED 2026-08-22] The up axis. Objaverse GLBs are Y-up, and renderer
# version 2 orbited +Z instead, so the camera swept OVER and UNDER every asset
# rather than around it: a 7.2x-tall lamppost rendered 7x WIDER than tall.
#
# Y-up established from the corpus, not assumed. Using LVIS categories as an
# external label, 1,195 unambiguously tall assets average normalised extents
# [x .542, y .962, z .488] and are y-longest 1,060/1,195 times; 481 flat assets
# average [x .944, y .304, z .824] and are y-longest 36/481.
#
# Confirmed independently against ULIP-2's released renders: their vase
# 9f1335d8... holds image height/width = 1.94 across ALL TWELVE views, which is
# what a rotationally symmetric object orbited about its own vertical axis does.
# Ours read 0.59 and swung between 0.47 and 1.15.
UP_AXIS = np.array([0.0, 1.0, 0.0])

# [USER DECISION `U-W`, 2026-08-22] WHITE. Not because upstream is white -- it
# is not; ULIP-2's released renders measure corner luminance 0, pure black.
#
# The engineer changed this to black to match upstream and the Reviewer measured
# the result on the criterion this milestone had already chosen for itself,
# `S-5`, retrieval against ULIP's own `image_feat` over 286 shared assets:
#
#     white background + xmag 1.10   R@1 97.2%   matched 0.9160   gap 0.3734
#     black background + xmag 1.20   R@1 95.8%   matched 0.8782   gap 0.3404
#
# Black COSTS 1.4 points. The USER's reasoning is recorded verbatim in HANDOFF:
# a criterion cannot be used when it wins and set aside when it loses.
#
# So this is an IMPLEMENTATION CHOICE that deliberately DIVERGES from upstream,
# on upstream's own metric. It is a DEVIATION and needs a registry id, which is
# the Integrator's to assign -- routed to Master, not decided here.
BACKGROUND_RGBA = [255, 255, 255, 255]

# [FITTED to ULIP-2's released renders] Orthographic half-width. Measured over
# 8 assets present in both corpora, longest silhouette side divided by 224:
#
#     ULIP           mean 0.574   (range 0.405 - 0.701)
#     xmag 1.65      mean 0.418   -- ours, every asset smaller than ULIP's
#     1.65 * (0.418 / 0.574) = 1.20
#
# [USER DECISION `U-X`, 2026-08-22] REVERTED to 1.10, the value before this
# session. Measured on `S-5` over 286 assets, 1.20 buys nothing that 1.10 does
# not: white + 1.10 scores R@1 97.2% against black + 1.20's 95.8%. Matching
# upstream's MEAN framing on 8 assets is not a reason to move a number when the
# milestone's own criterion does not improve.
#
# The per-asset ratio is NOT constant (1.16 to 1.83), so upstream's framing rule
# is not a pure rescale of ours and this matches the mean, not the rule. What it
# does fix is that version 2 threw away more than half the frame: the object
# occupied 0.418 of it, so most pixels the image tower sees were background.
#
# A unit-sphere-normalised asset cannot be clipped while xmag >= 1, and 1.10
# keeps a 10% margin -- but only once `normalised_scene` bakes node transforms
# into vertices. It did not until renderer v4, and until then the claim in this
# comment was false: pyrender re-derived the node poses it could not represent,
# and 10 of 1,500 sampled assets had foreground on the frame edge. See the
# baking note in `normalised_scene`.
#
# IMPLEMENTATION CHOICE with upstream provenance (U-O). MetaFind says nothing
# about framing, and this number is fitted, not stated. It must never be
# reported as a paper value.
ORTHO_HALF_WIDTH = 1.10

# [FITTED to ULIP-2's released renders] Version 2 used ambient 0.4 with a
# directional intensity of 3.0, which blew out light-coloured assets: a white
# snowman came back with 59.9% of its foreground pixels at 255 and its hat,
# eyes and buttons gone, against ULIP's 0.3% for the same asset.
#
# Measured over the same 8 assets (mean over 12 views each):
#
#                        clipped px   median luminance
#     ULIP                    0.2%          119
#     ambient .4 int 3.0      7.5%          118   version 2
#     ambient .2 int 1.5      0.0%           87   too dark
#     ambient .5 int 1.5      0.7%          111   chosen
#
# Tuning on the single worst asset picked a setting that was 33 luminance
# points too dark across the other seven. The value below is chosen on all
# eight, which is why it is not the one the single-asset sweep preferred.
AMBIENT_LIGHT = 0.5
DIRECTIONAL_INTENSITY = 1.5

# NOT SOLVED, and not derivable. MetaFind states no elevation; ULIP-2's paper
# gives only "12 images, spaced equally by 360/12 degrees", which fixes the
# spacing and says nothing about the height of the orbit.
#
# [CORRECTED 2026-08-22] This comment previously read "SOLVED from ULIP's
# released renders instead of chosen: see tools/solve_ulip_elevation.py". That
# tool was never written, and the engineer's own HANDOFF entry said the
# elevation was NOT solved. An attempt was made: silhouette IoU against ULIP's
# renders peaks at 15 degrees on one rotationally symmetric asset but is flat
# from 10 to 20, and R@1 over 60 assets read 100.0 / 98.3 / 98.3 at 5 / 15 / 25
# -- a one-asset spread that separates nothing.
#
# `U-03` remains UNKNOWN. 20.0 is an IMPLEMENTATION CHOICE, versioned so the
# images stay reproducible, and it must never be written up as a solved value.
ORBIT_ELEVATION_DEG = 20.0
# 1 = fibonacci sphere lattice.
# 2 = azimuth orbit about +Z -- the WRONG up axis, so every asset tumbled
#     end-over-end instead of turning. White background, xmag 1.1,
#     ambient 0.4 / intensity 3.0, uncorrected mesh frame.
# 3 = azimuth orbit about the mesh up axis (UP_AXIS), 180 degree frame
#     correction applied at load, ambient 0.5 / intensity 1.5.
#
# [CORRECTED 2026-08-22] This note previously said v3 meant "black background,
# ULIP-matched framing". `U-W` and `U-X` returned both of those to v2's values,
# so two of the three clauses had gone false while the note still read as a
# description. What actually separates v2 from v3 is the orbit axis, the frame
# correction and the exposure -- nothing else.
#
# v4 bakes node transforms into vertices before pyrender sees them. Any asset
# with a non-uniform-scale or sheared node transform was placed differently
# from what `normalised_scene` computed -- differently from what `n03` sampled,
# and sometimes outside the frame. The version bump is what makes
# `is_complete` reject a v3 sidecar; without it a re-run is a silent no-op.
# 5: [2026-08-23] pyrender -> Blender/BlenderProc CYCLES via OpenShape's own
# renderer, 11 -> 12 views (three polar rings of four, USER decision), 224 ->
# 512 px, orthographic -> perspective 35 mm, opaque white -> transparent RGBA.
# The constants below from N_VIEWS down describe the RETIRED pyrender path and
# are kept only because `azimuth_orbit_directions` and `normalised_scene` are
# still imported by the verification tools; nothing in `process_one` reads them
# any more.
# [RENDERER_VERSION 6, 2026-08-24] The denoiser is now named explicitly
# (OptiX/GPU, `render_blender.DENOISER`) instead of inheriting BlenderProc's
# CPU default. It changes the pixels, so every v5 sidecar is stale.
# NOT bumped to 7 for the 2026-08-24 guard rewrite, deliberately. The guards
# decide which renders are ACCEPTED; they do not touch a single pixel, and the
# 45,782 assets already on disk passed the stricter rules, so they pass the
# looser ones unchanged. Re-rendering them would cost 6.6 hours to produce
# byte-identical images.
#
# What DOES differ across that boundary is metadata: records written before the
# rewrite carry `blank_views` under the old std()-of-black-composite definition
# and have no `view_coverage` / `distinct_views` / `dark_views`. Nothing reads
# those fields -- only `tools/measure_render_criteria.py` writes its own copy --
# so the split is legible rather than load-bearing. It is recorded here because
# a reader comparing two sidecars deserves to know why they differ.
RENDERER_VERSION = 6

# [ADDED 2026-08-24] How many assets may fail BACK TO BACK before the run is
# treated as broken rather than unlucky. IMPLEMENTATION CHOICE: 8 workers deep,
# so one bad pool generation cannot trip it, and at the measured 116 assets/min
# it costs about 30 seconds to establish. Scattered bad meshes never reach it,
# because any success resets the counter.
# Alpha coverage below which a view is judged to have drawn NOTHING.
#
# [SET TO ZERO 2026-08-24 — the fitted constant did not survive measurement]
#
# This was 0.001, and its justification was rewritten three times. Each version
# claimed a structure in the data; each was measured and did not hold:
#
#   v1  "an order of magnitude above the empty case"
#       -- the empty case is 0. Zero has no order of magnitude. Unevaluable.
#   v2  "specks at 0.0001-0.007" with the cut at 0.001
#       -- the cut sat INSIDE the band it named, so it admitted specks by
#          construction with the comment saying so.
#   v3  "the two populations do not overlap; 0.001 sits in the gap"
#       -- non-overlap is guaranteed by construction once you cut anywhere.
#          The Reviewer then proposed a real defence, a spacing discontinuity
#          at 181->349 px (1.93x against ~1.1x typical), measured over the 123
#          quarantined-blank assets.
#
# I measured v3's defence over the FULL population of 270 rather than the 123,
# and it does not hold. Best view per asset, in pixels of 512x512:
#
#     exactly 0            28 assets
#     0 < cov < 0.001      41 assets      1 px .. 181 px
#     cov >= 0.001        201 assets
#
#     spacing around the cut (0.001 = 262.1 px):
#       36 ->  102   2.83x   <-- the LARGEST gap in the region, nowhere near the cut
#      181 ->  198   1.09x
#      198 ->  223   1.13x
#      223 ->  252   1.13x
#      252 ->  349   1.38x   <-- where 0.001 actually falls
#
# There is no discontinuity at 0.001. The distribution runs continuously from
# one pixel upward, and the only genuinely discrete feature in it is EXACTLY
# ZERO. So the honest cut is the definitional one:
#
#   "was anything drawn" is alpha > 0. It needs no constant and cannot be
#   tuned until a number behaves, which is what S-3 prohibits and what three
#   rewrites of this comment were circling.
#
# The cost is stated, not hidden: this admits 41 assets between 1 and 181
# pixels. They are NOT thereby usable -- a 1-pixel object is not describable by
# a VLM -- but usability is corpus membership, that is the USER's decision, and
# `view_coverage` is recorded per view so the population is separable at any
# time without re-rendering. A guard that answers "did the renderer draw
# anything" must not also quietly answer "should this be in Table 1".
MIN_COVERAGE = 0.0

SYSTEMIC_RUN = 64

# How many worker pools may die BACK TO BACK before the run is broken rather
# than unlucky. A pool that cannot survive one batch will not survive the next
# 92 either. 3 costs at most three batches; any asset success resets it.
POOL_DEATHS_SYSTEMIC = 3

def _fingerprint_sources():
    """Everything whose bytes decide a rendered pixel, in one place.

    `meshload` owns `FRAME_CORRECTION`. `render_blender` owns the engine, the
    view count, the denoiser and the patch applied to the vendored script --
    the 2026-08-24 OptiX change touched ONLY that file, and the fingerprint
    moved solely because `renders.py` happened to be edited in the same commit.
    That was luck, not coverage.

    `render_single_glb.py` is passed as a file rather than a module because it
    runs inside Blender's own Python; importing it here would execute it.
    """
    from metafind.data import meshload, render_blender

    return ((sys.modules[__name__], meshload, render_blender),
            (render_blender.VENDOR_SCRIPT,))


def implementation_fingerprint() -> dict[str, str]:
    """This node's own source files. See `runlog.implementation_fingerprint`."""
    mods, files = _fingerprint_sources()
    return runlog.implementation_fingerprint(*mods, extra_files=files)


def verify_fingerprint(expected: dict[str, str] | None) -> None:
    """Modules once per worker; the VENDORED SCRIPT on every asset.

    [FIXED 2026-08-24, Codex CHANGES REQUIRED] `runlog.verify_fingerprint`
    returns forever once `_FINGERPRINT_VERIFIED` is set, which is correct for
    imported modules -- a worker cannot re-import them. It is WRONG for
    `render_single_glb.py`, because `_patched_script` reads that file off disk
    for EVERY asset. A vendor edit after a worker's first task was rendered
    under the same `RENDERER_VERSION`, undetected, while a later worker's first
    task caught it -- a mixed corpus with a gate that reported clean.
    """
    import hashlib

    mods, files = _fingerprint_sources()
    if expected:
        for path in files:
            name = path.name
            want = expected.get(name)
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if want is not None and want != got:
                raise RuntimeError(
                    f"implementation changed while the run was in progress: {name}. "
                    "This worker would write artifacts the rest of the corpus does not "
                    "share, and no sidecar field would show it. Restart the run."
                )
    runlog.verify_fingerprint(expected, *mods, extra_files=files)



def azimuth_orbit_directions(n: int = N_VIEWS,
                             elevation_deg: float = ORBIT_ELEVATION_DEG) -> np.ndarray:
    """``n`` directions on one horizontal orbit -- the PRIMARY layout.

    ULIP-2's Objaverse pipeline renders "12 images per shape, spaced equally by
    30 degrees", which is a single-axis orbit, and MetaFind states it builds on
    ULIP-2. Using 11 equally spaced azimuths is the smallest change from a
    documented upstream method that satisfies MetaFind's stated count.

    This is provenance, not proof. The paper says only "11 orthogonal
    viewpoints"; nothing in it names an orbit, an axis or an elevation.

    [CORRECTED 2026-08-22] The elevation now rides on ``UP_AXIS``. Version 2
    put it on the third component while the meshes are Y-up, so the "orbit"
    tumbled every asset end-over-end. The bug was internal: the function's own
    name, its variable names and its docstring all describe an azimuth orbit
    about the up axis, and the arithmetic did something else.
    """
    az = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
    el = np.deg2rad(elevation_deg)
    # An orthonormal basis with `UP_AXIS` last, so azimuth sweeps the plane the
    # object stands on and elevation lifts out of it. Written against UP_AXIS
    # rather than hardcoding index 1: the whole defect was an index written by
    # hand where a named axis was meant.
    up = np.asarray(UP_AXIS, dtype=np.float64)
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e0 = np.cross(up, seed)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(up, e0)
    return (np.cos(el) * np.cos(az))[:, None] * e0 \
        + (np.cos(el) * np.sin(az))[:, None] * e1 \
        + (np.sin(el) * np.ones(n))[:, None] * up


def fibonacci_directions(n: int = N_VIEWS) -> np.ndarray:
    """``n`` unit vectors spread evenly over the sphere -- the VARIANT.

    Was the primary until the provenance argument above moved it. Kept
    executable because the two layouts give different multi-view coverage, and
    which one MetaFind used is genuinely unknown.
    """
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def look_at(eye: np.ndarray, target=(0.0, 0.0, 0.0), up=UP_AXIS) -> np.ndarray:
    """Camera-to-world pose looking from ``eye`` at ``target`` (OpenGL -Z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(fwd @ up)) > 0.999:  # looking along `up`: pick another reference
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -fwd, eye
    if not np.isfinite(pose).all():
        # pyrender normalises again internally (node.py:234 "U / norms"), and a
        # NaN pose reaching the GL layer can take the whole process down rather
        # than raising. Fail here, where it is one asset.
        raise ValueError(f"degenerate camera pose for eye={eye.tolist()}")
    return pose


def normalised_scene(path: Path):
    """Load the GLB with its scene transforms applied and fit it to a unit sphere.

    Returns ``(trimesh_scene, raw_bbox_extents)``. The extents are measured
    BEFORE normalisation, because afterwards there is nothing left to measure:
    every asset is the same size by construction, which is the point.
    """
    import trimesh

    from metafind.data import meshload

    # Same loader as n03, so the render and the cloud cannot end up in
    # different frames -- see `meshload`'s docstring. The 180 degree yaw
    # correction is applied here, once, for both nodes.
    scene = meshload.load_scene(path)

    # Drop everything pyrender cannot turn into a mesh, BEFORE it tries.
    # Objaverse GLBs carry Path3D curves and PointCloud geometry that
    # from_trimesh_scene rejects outright, taking the whole asset with them --
    # measured: 4 of 200 died on a Path3D alone. Dropping a curve loses a
    # decoration; keeping it loses the object.
    for name in [n for n, g in scene.geometry.items()
                 if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0]:
        scene.delete_geometry(name)
    for geom in scene.geometry.values():
        _flatten_texture(geom)

    verts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry.get(name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
            continue
        verts.append(trimesh.transform_points(geom.vertices, transform))
    if not verts:
        raise ValueError("no triangulated geometry in this GLB")

    allv = np.concatenate(verts, axis=0)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    extents = hi - lo
    centroid = allv.mean(axis=0)
    radius = float(np.linalg.norm(allv - centroid, axis=1).max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"degenerate mesh: radius {radius}")

    fit = np.eye(4)
    fit[:3, :3] /= radius
    fit[:3, 3] = -centroid / radius
    scene.apply_transform(fit)

    # Bake every node transform into its own vertices, leaving an identity
    # scene graph.
    #
    # pyrender stores a node's pose as translation + rotation quaternion +
    # scale, so a node transform carrying non-uniform scale or shear has no
    # representation there and `Scene.from_trimesh_scene` silently substitutes
    # a different matrix. Measured on 2f0ef6ad926b474189b6ef489d11954c, whose
    # node `Cube.025_0` has column norms 0.014 / 0.009 / 0.457 -- a 50x
    # non-uniform scale: the translation column came through exactly and the
    # 3x3 did not, moving that geometry from y >= -0.939 to y >= -1.148. That
    # is outside `ORTHO_HALF_WIDTH`, so it was clipped in 7 of 11 views, and it
    # is no longer the geometry `n03` sampled -- the drift `meshload` exists to
    # prevent, reappearing one layer further down.
    #
    # Baking leaves pyrender nothing to decompose. It is done after `fit` so
    # the normalisation is inside what gets baked, and per node rather than per
    # geometry because one geometry may be instanced under several transforms.
    baked = trimesh.Scene()
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry.get(name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
            continue
        placed = geom.copy()
        placed.apply_transform(transform)
        baked.add_geometry(placed, node_name=node, geom_name=node)
    if not baked.geometry:
        raise ValueError("no triangulated geometry survived baking")
    return baked, extents


def _flatten_texture(geom) -> None:
    """Make a material pyrender can accept, or drop it to plain colour.

    pyrender refuses a 2-channel (grey+alpha) texture and refuses a raw ndarray
    where it expects a PIL image -- 10 of 200 assets died on those two alone.
    Neither is a broken asset: converting the image is enough, and where that
    fails the object still renders with its base colour, which is far better
    than losing it.
    """
    import trimesh
    from PIL import Image

    vis = getattr(geom, "visual", None)
    mat = getattr(vis, "material", None)
    if mat is None:
        return
    # EVERY texture slot, not just baseColor. Fixing baseColorTexture alone
    # left 3 of 200 assets dying on emissive and normal maps -- pyrender
    # validates each slot it uses, so one unconverted map loses the asset just
    # as completely as the main one.
    for slot in ("baseColorTexture", "emissiveTexture", "normalTexture",
                 "metallicRoughnessTexture", "occlusionTexture"):
        tex = getattr(mat, slot, None)
        if tex is None:
            continue
        try:
            if isinstance(tex, np.ndarray):
                tex = Image.fromarray(tex)
            if tex.mode not in ("RGB", "RGBA"):
                tex = tex.convert("RGBA")
            setattr(mat, slot, tex)
        except Exception:  # noqa: BLE001 -- a bad map must not cost the object
            try:
                setattr(mat, slot, None)
            except Exception:  # noqa: BLE001
                geom.visual = trimesh.visual.ColorVisuals(mesh=geom)
                return


LAYOUTS = {
    "ulip2_azimuth_orbit_11": azimuth_orbit_directions,
    "fibonacci": fibonacci_directions,
}

_RENDERER: dict[int, object] = {}


def _renderer(resolution: int):
    """One OffscreenRenderer per process, created once and reused.

    Building and tearing down an EGL context per asset is what broke the
    threaded version -- 67 of 200 assets failed on eglDestroyContext -- and
    doing it 46,000 times running stalled the process version too: after ~8,600
    assets the pool had no workers left, the parent spun at 37% CPU with 20
    threads, and progress simply stopped. No error in the log, no OOM in
    dmesg, the process still alive. A hung process that keeps running is the
    failure mode a "did it exit?" check cannot see.

    Deliberately per-process module state: a renderer holds a GL context and
    cannot be pickled across a process boundary.
    """
    import pyrender

    if resolution not in _RENDERER:
        _RENDERER[resolution] = pyrender.OffscreenRenderer(resolution, resolution)
    return _RENDERER[resolution]


def render_views(path: Path, n_views: int = N_VIEWS, resolution: int = RESOLUTION,
                 projection: str = PROJECTION, layout: str = CAMERA_LAYOUT,
                 elevation_deg: float = ORBIT_ELEVATION_DEG,
                 ortho_half_width: float = ORTHO_HALF_WIDTH):
    """``(images, raw_bbox_extents)`` -- ``n_views`` HxWx3 uint8 arrays.

    ``elevation_deg`` and ``ortho_half_width`` are parameters rather than
    constants read from module scope so that the sweep which SOLVES them against
    ULIP's released renders exercises this exact function, not a copy of it.
    Their defaults are the production values.
    """
    import pyrender

    scene_tm, extents = normalised_scene(path)
    scene = pyrender.Scene.from_trimesh_scene(
        scene_tm, bg_color=BACKGROUND_RGBA, ambient_light=[AMBIENT_LIGHT] * 3)
    if projection == "orthographic":
        # Apparent size stays independent of camera distance -- the property
        # L1-RENDER-PROJECTION-CONSISTENT asserts. The half-width itself is
        # fitted to ULIP's framing; see ORTHO_HALF_WIDTH.
        camera = pyrender.OrthographicCamera(
            xmag=ortho_half_width, ymag=ortho_half_width, znear=0.01, zfar=100.0)
    elif projection == "perspective":
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, znear=0.01, zfar=100.0)
    else:
        raise ValueError(f"unknown projection {projection!r}")

    images = []
    renderer = _renderer(resolution)
    # The sphere lattice has no elevation to set; the orbit does. Passed by
    # keyword so a layout that does not take it fails loudly here rather than
    # silently ignoring the sweep's argument and returning the default.
    directions = (azimuth_orbit_directions(n_views, elevation_deg)
                  if layout == "ulip2_azimuth_orbit_11" else LAYOUTS[layout](n_views))
    for d in directions:
        pose = look_at(d * 3.0)
        cam_node = scene.add(camera, pose=pose)
        # Light rides with the camera, so every view is lit the same way. A
        # fixed world light makes some views black, which is a valid image
        # and an invalid observation.
        light_node = scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                      intensity=DIRECTIONAL_INTENSITY), pose=pose
        )
        colour, _ = renderer.render(scene)
        images.append(np.ascontiguousarray(colour[..., :3]))
        scene.remove_node(cam_node)
        scene.remove_node(light_node)
    return images, extents


def sidecar_path(out_dir: Path, uid: str) -> Path:
    return out_dir / f"{uid}.json"


def retire_stale_sidecar(out_dir: Path, uid: str) -> bool:
    """Move a failed asset's OLD sidecar aside so it stops reading as complete.

    Resume is built on "an asset is complete only once its sidecar lands", which
    is sound when the alternative is no sidecar at all. It is NOT sound after a
    `RENDERER_VERSION` bump: the asset is selected for re-render because its
    sidecar is stale, and if that re-render then fails, **the stale sidecar is
    still sitting there**. `rebuild_index` globs `*.json` and counts it, so the
    corpus reports the asset as complete while the record and the PNGs beside it
    were produced by the renderer the bump exists to replace.

    Measured on the v4 re-render: 23 assets failed and kept `renderer_version 3`
    records, and `n_indexed` counted all 23 as complete. 18 of them were lost to
    one `BrokenProcessPool` event, so they were not even bad assets.

    Renamed, not deleted -- the record is evidence about a failure, and `.stale`
    falls outside the `*.json` glob that defines the corpus.

    Numbered, because `sc.replace(".json.stale")` OVERWRITES: a second
    retirement destroyed the first one, on the exact re-render path this
    function exists for. Keeping evidence and then silently deleting the older
    evidence is the same as not keeping it.
    """
    sc = sidecar_path(out_dir, uid)
    if not sc.exists():
        return False
    target = sc.with_suffix(".json.stale")
    n = 1
    while target.exists():
        n += 1
        target = sc.with_suffix(f".json.stale.{n}")
    sc.replace(target)
    return True


def is_complete(out_dir: Path, uid: str) -> bool:
    """Completion is the sidecar plus matching digests, never the image files.

    Same contract as n03, and for the same reason: a crash after the last PNG
    and before the record leaves an asset that a restart skips forever with no
    metadata, and `renders` is what n05 and n06 consume.
    """
    sc = sidecar_path(out_dir, uid)
    if not sc.exists():
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # [ADDED 2026-08-22] Completion is relative to the CURRENT renderer. The
    # v2 -> v3 correction moved the orbit axis, the framing and the exposure
    # while every stale sidecar stayed internally consistent, so all 45,955 of
    # them passed this check and a bare re-run would have skipped the corpus and
    # reported success. n04 at least stamps its version; n03 did not even do
    # that. Found by the Reviewer, 2026-08-22, before the run.
    #
    # `!=` not `<`: a sidecar from a newer renderer is not this renderer's
    # output either, and accepting it would let a downgraded run inherit
    # artifacts it cannot reproduce.
    if rec.get("renderer_version") != RENDERER_VERSION:
        return False
    if len(rec.get("view_paths", [])) != LIVE_N_VIEWS:
        return False
    sizes = rec.get("view_bytes")
    if sizes is None or len(sizes) != LIVE_N_VIEWS:
        # Sidecars written before view_bytes existed. Upgrade in place with a
        # stat per file rather than requiring a separate migration step: the
        # check that needs the field is the one that can cheapest supply it,
        # and a migration script is one more thing to remember to run.
        try:
            sizes = [Path(v).stat().st_size for v in rec["view_paths"]]
        except (OSError, KeyError):
            return False
        rec["view_bytes"] = sizes
        tmp = sc.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec))
        tmp.replace(sc)
    # SIZE, not sha256. Re-hashing every completed asset on startup means
    # reading 11 PNGs each: at 8,927 assets that is ~2.5 GB of cold reads
    # before the first new render, and with the point-cloud node saturating
    # the same disk it never finished -- the run looked hung, and was.
    # A truncated or half-written PNG has the wrong size, which is what this
    # check is for; the sha256 stays in the record for provenance and
    # L2-RESUME, where reading the bytes is the point.
    # [ADDED 2026-08-24, Codex] `view_sha256` is what `view_io.image_identity`
    # turns into the identity n05 and n06 bind their caches to. A sidecar without
    # it yields "" -- which n05 stamps, n06 then matches ""=="" and encodes, and
    # n06's own completion check later refuses, so the asset never settles on
    # resume. Accepted item A is "trust the stored hashes"; having NO hashes is a
    # different thing and is not complete.
    if len(rec.get("view_sha256") or []) != LIVE_N_VIEWS:
        return False
    for path_str, want in zip(rec["view_paths"], sizes):
        try:
            if Path(path_str).stat().st_size != want:
                return False
        except OSError:
            return False
    return True


from metafind.data.render_blender import N_VIEWS as LIVE_N_VIEWS  # noqa: E402
from metafind.data.view_io import load_view_rgb  # noqa: E402


def process_one(uid: str, glb: Path, out_dir: Path,
                fingerprint: dict[str, str] | None = None) -> dict:
    # Before any rendering: this worker may have re-imported a module that
    # changed since the run started. See `verify_fingerprint`.
    verify_fingerprint(fingerprint)

    import numpy as np
    from PIL import Image

    from metafind.data import render_blender

    # [RENDERER_VERSION 5, 2026-08-23] Blender writes the PNGs itself, so this
    # no longer receives images and saves them -- it receives paths. The
    # subprocess stages into a temp directory and moves the files in only once
    # all twelve exist, so a killed Blender leaves `asset_dir` untouched and the
    # sidecar-is-the-completion-marker rule still holds.
    extents = render_blender.raw_extents(glb)
    asset_dir = out_dir / uid
    paths_out = render_blender.render_asset(glb, asset_dir)
    if len(paths_out) != LIVE_N_VIEWS:
        raise ValueError(f"rendered {len(paths_out)} views, expected {LIVE_N_VIEWS}")

    view_paths, view_sha, view_bytes, coverage, dark = [], [], [], [], 0
    for p in paths_out:
        # "Did anything get DRAWN here" is an ALPHA question, and only alpha
        # answers it.
        #
        # [CORRECTED 2026-08-24] This measured `std()` of the BLACK-COMPOSITED
        # image and called a flat result "blank". A pitch-black object on the
        # recorded black background composites to a uniform black frame, so an
        # asset that rendered perfectly -- alpha covering 30-47% of the frame --
        # was reported as "the asset never entered frame". Measured on this
        # corpus: of 14 quarantined "blank" assets, 4 had alpha coverage of
        # 31-47% and a maximum RGB value of 1/255. They entered frame. They are
        # black.
        #
        # Alpha is independent of the background decision, which is what makes
        # it the right test: `U-BG` can change and this check does not move.
        rgba = np.asarray(Image.open(p).convert("RGBA"))
        cov = float((rgba[..., 3] > 0).mean())
        coverage.append(round(cov, 8))
        # Recorded, never fatal: an object that IS black is a fact n05 and any
        # audit should be able to see, not a render failure.
        #
        # [2026-08-24] Composited from the array already in hand rather than
        # re-decoding the PNG through `load_view_rgb`. Same source-over rule,
        # same result, half the decodes -- 552k of them on a full run, on the
        # hot path of a 6.6-hour stage.
        if cov > 0:
            a = rgba[..., 3:4].astype(np.float32) / 255.0
            flat = rgba[..., :3].astype(np.float32) * a  # over black == U-BG
            if float(flat.std()) < 1.0:
                dark += 1
        view_paths.append(str(p))
        blob = p.read_bytes()
        view_sha.append(hashlib.sha256(blob).hexdigest())
        view_bytes.append(len(blob))

    blank = sum(1 for c in coverage if c <= MIN_COVERAGE)
    if blank == LIVE_N_VIEWS:
        raise ValueError(
            f"every view drew nothing -- max alpha coverage {max(coverage):.8f}")
    # [CORRECTED 2026-08-24] This required all 12 renders to be byte-DISTINCT and
    # said "the camera is not moving between renders". For a flat object that
    # sentence is simply false: the four edge-on views of a carpet are genuinely
    # and identically empty, the camera moved for every one of them, and the
    # remaining eight views are perfect. Measured: 148 assets were discarded this
    # way, 91 of them at exactly 9/12 -- 12 minus the four collapsed edge-on
    # views, plus one. Carpets 32, manholes 12, chessboards 10, doormats 8.
    #
    # Pixel identity CANNOT separate "the camera did not move" from "the object
    # looks the same from there" -- a featureless sphere would fail it too. Only
    # the pathological case survives here: EVERY view identical, which no real
    # object and no working camera produces.
    #
    # [CORRECTED 2026-08-24, MASTER] An earlier version of this comment claimed
    # the camera is "checked where the camera actually lives (`view_directions`,
    # asserted distinct in the test suite and recorded per asset)". IT IS NOT.
    # `VIEW_DIRECTIONS` is a hardcoded constant in `render_blender`; the record
    # restates it, and the test asserts the constant -- which cannot vary per run
    # and therefore cannot fail on a bad render. That field is where the camera
    # was SUPPOSED to be, never where it was.
    #
    # State it plainly instead: AFTER THIS CHANGE, NOTHING AT RUNTIME CHECKS THE
    # CAMERA except the all-identical case above. That is accepted as the trade
    # -- the old guard could not tell symmetry from fault either, and it was
    # discarding 201 good assets to catch a fault it never caught -- but it is a
    # gap, not a covered base, and it belongs in the run report as one.
    if len(set(view_sha)) == 1:
        raise ValueError(
            f"all {LIVE_N_VIEWS} views are byte-identical; the camera did not move")

    record = {
        "uid": uid,
        "view_paths": view_paths,
        "view_sha256": view_sha,
        "view_bytes": view_bytes,
        "raw_bbox_extents": [float(v) for v in extents],
        # [RENDERER_VERSION 5] These describe BLENDER now. The previous values
        # (orthographic, ulip2_azimuth_orbit_11, 20 deg, 224 px) described
        # pyrender and would be a lie in a v5 record.
        "projection": "perspective",
        "camera_layout": "openshape_three_rings_of_four",
        "orbit_elevation_deg": None,          # three rings, not one elevation
        "view_directions": [
            {"ring": name, "polar_deg": polar, "azimuths_deg": list(az)}
            for (name, az), polar in zip(render_blender.VIEW_DIRECTIONS, (60, 90, 120))],
        "resolution": render_blender.RESOLUTION,
        "background": "transparent_rgba",
        "renderer_version": RENDERER_VERSION,
        # Read off the artifact, not restated from constants here: the vendored
        # script's own hash and blenderproc's own version. A constant in this
        # file cannot notice that the thing it describes changed.
        "renderer": render_blender.renderer_versions(),
        # How each choice was arrived at, so a reader never has to guess which
        # of these the paper actually specifies.
        "camera_layout_source": "openshape_render_single_glb.py",
        "projection_source": "openshape_render_single_glb.py",
        "n_views_source": "USER decision 2026-08-23; DEVIATION from MetaFind's stated 11",
        "background_source": "openshape film_transparent=True",
        "blank_views": blank,
        # [ADDED 2026-08-24] The evidence the two guards above now judge, kept so
        # a reader can see WHY an asset passed rather than only that it did.
        "view_coverage": coverage,
        "distinct_views": len(set(view_sha)),
        "dark_views": dark,
    }
    sc_tmp = sidecar_path(out_dir, uid).with_suffix(".json.part")
    with sc_tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sc_tmp.replace(sidecar_path(out_dir, uid))
    return record


# WHAT THIS DOES AND DOES NOT DO. [CORRECTED 2026-08-24, Codex CHANGES REQUIRED]
# The registry declares RESOURCE and DETERMINISTIC_INPUT as different POLICIES.
# **This node does not implement them.** `todo` is rebuilt from `is_complete`
# alone on every pass, so EVERY quarantined asset -- both classes -- is retried
# on the next pass, up to the chain's `MAX_PASSES`. Nothing reads
# `failure_class`; it is written to the quarantine log and consumed by humans
# and by G3's quarantine_rate.
#
# The earlier version of this comment asserted that the classification "decides
# whether an asset is ever tried again". That was FALSE when it was written, in
# this file, today. It is left described rather than deleted because a comment
# that overstates what code does is the same defect as a comment that overstates
# what a paper says, and this node has now produced one of each.
#
# The classification still matters -- it is what tells a human whether 900 rows
# are broken meshes or a dying GPU -- which is why it is worth getting right:
#
# `isinstance(exc, (MemoryError, OSError))` could never see any of it. Rendering
# is a SUBPROCESS: every non-zero Blender exit reaches this process as the plain
# `RuntimeError` that `render_asset` raises, so the RESOURCE branch was dead and
# every failure was filed as broken geometry. Measured over the 901 real
# quarantine rows of the v5 corpus (the 36,542 fingerprint rejections excluded):
#
#     "System is out of GPU memory"    70   exit 1        -> RESOURCE
#     BrokenProcessPool                21                 -> RESOURCE
#     blank views                     537                 -> DETERMINISTIC_INPUT
#     duplicate views                 228                 -> DETERMINISTIC_INPUT
#     LinAlgError / IndexError         24                 -> DETERMINISTIC_INPUT
#     exit 245, cause not in the log   21                 -> UNKNOWN
#
# The captured message is the only evidence a dead subprocess leaves, so it is
# what gets read. `exit 245` is NOT guessed: its captured tail ends inside glTF
# import with no error, and calling it either class would be inventing a retry
# policy. UNKNOWN says so, and is reported rather than silently retried.
_RESOURCE_MARKERS = (
    "out of gpu memory",
    "out of memory",
    "failed to retain cuda context",
    "cannot allocate memory",
    # BrokenProcessPool: the WORKER died. Whatever killed it, it was not this
    # asset's geometry.
    "terminated abruptly",
    # SIGKILL. The OOM killer or an operator -- unproven which, and it does not
    # matter here: a killed process is not evidence of a defective mesh, and the
    # 7 rows this produced on 2026-08-24 were an operator stopping the run.
    "exit -9",
    # [ADDED 2026-08-24, Codex] A 900 s timeout is the machine running out of
    # time, not the mesh being malformed. `render_blender` now raises with this
    # wording so the class is decidable from the message.
    "timed out",
)


# The failure messages this node RECOGNISES as a property of the asset. A class
# has to be EARNED by one of these, never inherited by default. See `Tally.failure`.
_ASSET_FAULT_MARKERS = (
    "every view drew nothing",
    "distinct views",
    "byte-identical",
)


def _is_asset_fault(exc: BaseException) -> bool:
    """True only for failures POSITIVELY identified as a property of the asset."""
    msg = str(exc).lower()
    return any(m in msg for m in _ASSET_FAULT_MARKERS)


def failure_class(exc: BaseException) -> str:
    """RESOURCE (retry) / DETERMINISTIC_INPUT (quarantine) / UNKNOWN (report)."""
    if isinstance(exc, (MemoryError, OSError)):
        return "RESOURCE"
    msg = str(exc).lower()
    if any(m in msg for m in _RESOURCE_MARKERS):
        return "RESOURCE"
    if "exit 245" in msg:
        return "UNKNOWN"
    return "DETERMINISTIC_INPUT"


class SystemicFailure(RuntimeError):
    """Raised to leave `runlog.run_progress` by the exception path.

    [ADDED 2026-08-24, ULIP2 Reviewer BLOCKER 2] The previous attempt wrote a
    quarantine row and called that the fix, under a comment correctly stating
    that `run_progress` records SUCCESS on any normal exit. It does: `rc` is 0
    unless `except BaseException` runs, and leaving the `with` by `break` is a
    normal exit. So a systemic stop left a durable `status: SUCCESS, rc: 0` row
    -- and by that same comment, run_progress is the file a resume reads.

    The comment named the mechanism and the code under it addressed a different
    one while reading as the fix. That is this batch's own named defect, inside
    the batch that named it.
    """


class Tally:
    """The per-asset accounting `main` does, in one testable place.

    [ADDED 2026-08-24, Codex BLOCKER 1] This logic lived inline inside a
    `try/except/else` and an edit inverted it: `done += 1` ran only on the
    exception path, so an all-success run reported "0 rendered" and returned 2,
    and the cost ledger counted failures as work. Nothing caught it -- the only
    test of the circuit break asserted `SYSTEMIC_RUN >= 8`, which is a constant,
    not a behaviour. A seam did not exist; that was the reason given for not
    testing it, so the seam is the fix.
    """

    def __init__(self, systemic_run: int = SYSTEMIC_RUN) -> None:
        self.done = 0
        self.failed = 0
        self.consecutive = 0
        self.pool_deaths = 0
        self.systemic: str | None = None
        self._limit = systemic_run

    def success(self) -> None:
        self.consecutive = 0
        self.pool_deaths = 0
        self.done += 1

    def pool_death(self) -> None:
        """A whole batch lost because the pool would not stay up.

        [ADDED 2026-08-24, ULIP2 Reviewer MAJOR 3] `failure()` correctly excludes
        `BrokenProcessPool` so ONE dead pool is not 64 bad assets. But the
        batch-level handler never told the tally anything, so the fault this
        breaker was built for -- "OptiX unavailable", which kills workers at
        spawn and breaks EVERY pool -- left `consecutive` at 0 and `systemic` at
        None for all 93 batches. The breaker was blind to its own headline case.
        """
        self.pool_deaths += 1
        if self.pool_deaths >= POOL_DEATHS_SYSTEMIC and self.systemic is None:
            self.systemic = (f"{self.pool_deaths} consecutive worker pools died before "
                             "finishing their batch; this is the machine, not the assets")

    def failure(self, exc: BaseException) -> None:
        self.failed += 1
        # A dead pool marks EVERY pending future with the same
        # `BrokenProcessPool`. Counting those as separate consecutive asset
        # failures made one recoverable worker death look systemic -- the
        # opposite of what the batch handler exists to express.
        #
        # [FIXED 2026-08-24] A `DETERMINISTIC_INPUT` failure is a property of
        # the ASSET -- a flat carpet whose edge-on views are identically empty,
        # a mesh that draws nothing. It is not evidence that the machine is
        # broken, and counting it as such made the breaker fire on the one pass
        # where it is guaranteed to be wrong: the LAST pass of every run, whose
        # work-list is by construction nothing but assets that already failed.
        #
        # Measured on this run: pass 3 retried 270 known-bad assets, 64 failed
        # back to back exactly as expected, the breaker fired, and the chain
        # halted with n04 at 45,782/46,052 -- 99.4% complete and past its own
        # 95% gate. Nothing was wrong except this counter.
        #
        # This is also the first thing that READS `failure_class`. Codex found
        # that field was written to a log and consumed by nothing; the class now
        # decides whether a failure can indict the run.
        #
        # [CORRECTED 2026-08-24, ULIP2 Reviewer MAJOR-2] This tested
        # `failure_class(exc) == "DETERMINISTIC_INPUT"`, which is that
        # function's FALL-THROUGH. So the breaker's coverage collapsed to the
        # seven strings in `_RESOURCE_MARKERS`, and every UNRECOGNISED failure
        # -- a CUDA error with unseen wording, a new Blender abort, a driver
        # fault phrased differently -- became invisible to it, silently and
        # forever. Same defect as the dead `isinstance(exc, (MemoryError,
        # OSError))` branch recorded above: a safety mechanism resting on a
        # list being complete.
        #
        # A breaker fails CLOSED on the unknown, so the exclusion is POSITIVE:
        # only a failure this node can NAME as the asset's own property is
        # excused. Everything else counts.
        if isinstance(exc, BrokenProcessPool):
            return
        if _is_asset_fault(exc):
            return
        self.consecutive += 1
        if self.consecutive >= self._limit and self.systemic is None:
            self.systemic = (f"{self.consecutive} consecutive assets failed; "
                             f"last: {type(exc).__name__}: {str(exc)[:200]}")


def rebuild_index(index_path: Path, out_dir: Path) -> int:
    """Derive renders_index.jsonl from the per-asset sidecars.

    Derived, never appended: an appended index grows duplicates on resume and
    disagrees with the filesystem the moment a run is interrupted.
    """
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as f:
        for sc in sorted(out_dir.glob("*.json")):
            try:
                rec = json.loads(sc.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            # [ADDED 2026-08-24] `is_complete` refuses a sidecar from a different
            # renderer, but this function used to republish every parseable one --
            # so a stale record the runner had correctly decided to re-render was
            # still handed to n05 and n06 through the index, which is the file
            # they actually read. The gate has to hold on BOTH sides of it.
            #
            # [STRENGTHENED 2026-08-24, Codex CHANGES REQUIRED] Checking only the
            # version was still WEAKER than `is_complete`: a current-version
            # sidecar left by an interruption, a `--limit` run or corruption can
            # name the wrong number of views or files whose sizes no longer
            # match, and it still reached n05/n06. Call the real predicate
            # instead of restating part of it -- two completion rules is how they
            # drift apart.
            # [FIXED 2026-08-24, Codex] The uid was read from the RECORD and the
            # record was published from the FILE. `A.json` claiming `"uid": "B"`
            # got validated against `B.json` and then published unvalidated, and
            # a record with no `uid` was validated by filename and published
            # without one, which is a KeyError in every consumer. The filename is
            # the identity `is_complete` and `sidecar_path` both use, so the
            # record must agree with it or it is not this asset's record.
            if rec.get("uid") != sc.stem:
                continue
            if not is_complete(out_dir, sc.stem):
                continue
            f.write(json.dumps(rec) + "\n")
            n += 1
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    # [RENDERER_VERSION 5] Default 4 -> 8, measured on this card. 16 assets,
    # 192 images per configuration, all 192 produced at P<=9:
    #
    #     P=1  8.12 s/asset      P=6  3.42 s/asset      P=9  3.28 s/asset
    #     P=2  4.54 s/asset      P=8  3.36 s/asset      P=10 FAILS  48/192
    #     P=4  3.56 s/asset                             P=12 FAILS  96/192
    #
    # The ceiling is GPU memory, not CPU: each Blender process holds its own
    # CUDA context (~3 GB) and P=10 dies with "Failed to retain CUDA context
    # (Out of memory)". The 2.4x plateau is one GPU serialising eight ray
    # tracers; more processes only overlap the CPU-side load.
    #
    # NOTE the help text below carries no literal "%" -- argparse formats help
    # strings with %-substitution, and the previous text ("34% eglDestroy...")
    # made `--help` itself raise TypeError. A CLI whose help crashes is a CLI
    # nobody reads.
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent Blender processes. Above 9 the GPU runs out "
                         "of CUDA contexts and assets fail silently-ish.")
    ap.add_argument("--limit", type=int,
                    help="process at most N assets, in manifest order (smoke runs)")
    ap.add_argument("--uids-file",
                    help="newline-separated uids; renders exactly these. Use this "
                         "rather than --limit when a later node must run on the "
                         "SAME assets -- --limit takes the first N in manifest "
                         "order, which is not the same set another node's --limit "
                         "or --uids-file selects.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
    glb_by_uid = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    out_dir = paths.RENDERS
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.uids_file:
        want = [ln.strip() for ln in Path(args.uids_file).read_text().splitlines()
                if ln.strip()]
        missing = [u for u in want if u not in glb_by_uid]
        if missing:
            raise SystemExit(f"{len(missing)} uid(s) have no GLB, e.g. {missing[:3]}")
        uids = want
    todo = [(u, glb_by_uid[u]) for u in uids
            if u in glb_by_uid and (args.force or not is_complete(out_dir, u))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(uids):,} in manifest, {len(todo):,} to render", flush=True)

    quarantine, done, started = [], 0, time.time()
    # ProcessPool, not ThreadPool. With 4 threads, 67 of 200 assets failed on
    # eglDestroyContext -- an EGL context belongs to the thread that made it,
    # and the errors surfaced as per-asset exceptions, so the run reported a
    # 41% quarantine rate as though a third of Objaverse were malformed.
    ctx = mp.get_context("spawn")
    # ONE POOL PER BATCH, and a dead pool is survivable. A child process died
    # abruptly at asset 400 and BrokenProcessPool took the whole 46,052-asset
    # run down with it -- a GL driver can kill a process outright, and when it
    # does, the correct response is to lose that batch, not the corpus.
    # Rebuilding a pool costs about a second per 500 assets.
    BATCH = 500
    fingerprint = implementation_fingerprint()
    print("implementation: " + "  ".join(f"{k} {v[:12]}" for k, v in fingerprint.items()),
          flush=True)
    quarantine, done, started = [], 0, time.time()
    # [ADDED 2026-08-24, Codex CHANGES REQUIRED] A systemic failure -- a driver
    # that stops answering, a vendor edit, OptiX unavailable -- fails EVERY
    # remaining asset one at a time, and every one of them was swallowed by the
    # per-asset handler below. `main` then returned 0 because one early asset had
    # succeeded, so an unattended 6.5-hour run could quarantine 45,000 assets and
    # exit reporting success.
    #
    # A CONSECUTIVE counter over NON-asset failures, not a rate: bad meshes are
    # normal corpus behaviour and never indict the machine, while a systemic
    # fault -- a dead driver, OptiX gone, a vendor edit -- fails everything in a
    # row for reasons no asset owns. 64 is an IMPLEMENTATION CHOICE: 8 workers
    # deep, so one bad pool generation cannot reach it.
    tally = Tally()
    try:
      with runlog.run_progress(NODE) as progress:
        for start in range(0, len(todo), BATCH):
            if tally.systemic:
                break
            batch = todo[start:start + BATCH]
            try:
                with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                            max_tasks_per_child=200) as pool:
                    futures = {pool.submit(process_one, u, g, out_dir, fingerprint): u
                               for u, g in batch}
                    for fut in cf.as_completed(futures):
                        uid = futures[fut]
                        try:
                            fut.result()
                        except Exception as exc:  # noqa: BLE001 -- one bad asset must not stop the run
                            retire_stale_sidecar(out_dir, uid)
                            runlog.quarantine(NODE, [{
                                "uid": uid,
                                "failure_class": failure_class(exc),
                                "exception_type": type(exc).__name__,
                                "exception_msg": str(exc)[:400],
                                "traceback": traceback.format_exc()[-1500:],
                            }])
                            quarantine.append(uid)
                            tally.failure(exc)
                            # [FIXED 2026-08-24, Codex BLOCKER 3] Setting the flag
                            # did not stop anything: all 500 futures were already
                            # submitted and the executor drained them. Under the
                            # case this exists for -- every job hitting the 900 s
                            # timeout -- that is ~15.6 more hours after the run is
                            # already known to be broken. Cancel what has not
                            # started and stop collecting.
                            if tally.systemic:
                                for f2 in futures:
                                    f2.cancel()
                                break
                            continue
                        tally.success()
                        done = tally.done
                        if done % 200 == 0:
                            rate = done / max(time.time() - started, 1e-9) * 60
                            print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                                  f"剩餘約 {(len(todo)-done)/max(rate,1e-9):.0f} 分, "
                                  f"quarantine {len(quarantine)}", flush=True)
            except BrokenProcessPool as exc:
                # The batch is lost, not the run. Whatever this batch finished
                # is kept and the rest is retried on resume, because an asset is
                # complete only once its sidecar lands -- but see
                # `retire_stale_sidecar`: that reasoning holds for a MISSING
                # sidecar and not for a STALE one, and the difference cost 23
                # assets on 2026-08-22.
                print(f"  批次 {start//BATCH} 的 worker 崩潰，跳過該批繼續：{exc}",
                      flush=True)
                runlog.quarantine(NODE, [{
                    "uid": f"__batch_{start}", "failure_class": "RESOURCE",
                    "exception_type": "BrokenProcessPool",
                    "exception_msg": str(exc)[:400], "traceback": "",
                }])
                tally.pool_death()
                if tally.systemic:
                    break

        # Inside the context on purpose: this is what makes run_progress record
        # FAILED instead of SUCCESS.
        # [R-32] The rc has to be recorded while this block is still OPEN.
        # `run_progress` cannot see a `return` that happens after it closes, and
        # the `return 0 if done or not todo else 2` below did exactly that: a
        # run that rendered nothing returned 2, halted the chain, and left a
        # durable row saying SUCCESS / rc 0. The systemic path never had this
        # problem because it leaves by raising -- which is why raising is the
        # shape that cannot be forgotten, and this assignment is the shape that
        # can. It is used here because n04's non-systemic failure is a normal
        # outcome, not an exception.
        progress.rc = 0 if tally.done or not todo else 2
        if tally.systemic:
            raise SystemicFailure(tally.systemic)
    except SystemicFailure:
        pass
    systemic = tally.systemic
    done = tally.done
    if systemic:
        print(f"\nSTOPPED -- this is not a bad asset, it is a broken run: {systemic}",
              flush=True)
        # A row in the quarantine channel as well as the run_progress FAILED row
        # above: G3 reads quarantine to compute the rate, and a run that stopped
        # broken should say WHY there, not only THAT it failed.
        #
        # [CORRECTED 2026-08-24, ULIP2 Reviewer MINOR] This comment used to state
        # that run_progress records SUCCESS here and that a systemic stop leaves
        # a `SUCCESS / rc 0` row. Both halves are now false -- the context is
        # left by `raise SystemicFailure`, and the durable row is `FAILED / rc 1`,
        # measured. Left as a correction rather than deleted: a comment claiming
        # a defect the code no longer has is the same notch pointing the other
        # way, and it is how the next reader re-fixes something already fixed.
        runlog.quarantine(NODE, [{
            "uid": "__run", "failure_class": "RESOURCE",
            "exception_type": "SystemicFailure",
            "exception_msg": systemic[:400], "traceback": "",
        }])
    n_indexed = rebuild_index(paths.LOGS / "renders_index.jsonl", out_dir)
    runlog.cost_ledger(
        cpu_seconds=round(time.time() - started, 1),
        assets_rendered=done,
        views_written=done * LIVE_N_VIEWS,
    )
    print(f"\n{done:,} rendered this run, {tally.failed:,} failed, "
          f"{n_indexed:,} complete on disk, {len(quarantine):,} quarantined -> {out_dir}")
    # A run that tripped the systemic break did NOT succeed, however many assets
    # it finished first. Reporting 0 there is what let the chain advance to the
    # next stage on a corpus that was never rendered.
    if systemic:
        return 3
    # Single source: the same value the durable row already carries. Recomputing
    # it here is how the two would drift.
    return progress.rc


if __name__ == "__main__":
    raise SystemExit(main())
