"""Render 11 views per Objaverse asset.

# IMPLEMENTS-NODE: n04_render_views

Paper 2.3 gives one number and nothing else: assets are "rendered from 11
orthogonal viewpoints". Everything below that is a recorded choice.

  U-03a  projection.   UNRESOLVED. Eleven mutually orthogonal directions do
         not exist in R^3, so "orthogonal" cannot be literal -- but that does
         NOT make it evidence for orthographic projection. "Viewpoint"
         describes a camera pose; "orthographic" describes a projection model,
         and an author meaning the latter would more naturally have written
         "orthographic views". Orthographic is our CURRENT IMPLEMENTATION
         CHOICE, recorded as such. It is not the most likely reading of
         "orthogonal", because there is no evidence for one.
  U-03   camera placement. UNRESOLVED. The primary is an equal-azimuth orbit
         of 11 views, Δazimuth = 360/11, because MetaFind states it builds on
         ULIP-2 and ULIP-2's Objaverse pipeline is "12 images per shape,
         spaced equally by 30 degrees" -- a single-axis orbit. Swapping 12 for
         the 11 MetaFind requires is the smallest step from a documented
         upstream method. Labelled UPSTREAM-INFORMED CHOICE, not paper truth.
         Fibonacci-11 was the previous primary and stays as a variant.
         The orbit's ELEVATION is not derivable from anything and is a
         versioned implementation choice; no top view is added, because
         nothing supports one.
  U-04   resolution. 224px, matching what the image tower consumes.

Scale is destroyed on purpose, and recorded on the way past
--------------------------------------------------------

Each asset is normalised to a unit sphere before rendering, which is what makes
L1-RENDER-SCALE-INVARIANT hold: a millimetre-scaled and a metre-scaled copy of
the same mesh must produce identical images, or the image tower would be
learning the modelling units of whoever uploaded the asset. The cost is that
absolute size is gone from the render, so the annotator's "size dimensions"
(paper 2.3) can only ever be a category prior. ``raw_bbox_extents`` is written
alongside so that estimate is auditable -- see F13, and note it is the
axis-aligned bounding box in the file's own units, not a verified physical size.

Two things inherited from n03 rather than rediscovered
------------------------------------------------------

*Scene-graph transforms are applied.* 65.8% of sampled Objaverse GLBs place
geometry with a non-identity node transform. Ignoring them renders the object
collapsed on itself, and the image still looks like a plausible render of
something.

*Completion means the sidecar, not the pixels.* Files are written first and the
per-asset record last, so a crash costs one re-render rather than leaving an
asset that is skipped forever with no metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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
RENDERER_VERSION = 4

# Set once per worker process. A worker imports its modules at spawn and cannot
# change them afterwards, so one check covers every task it will ever run.
_FINGERPRINT_VERIFIED = False


def implementation_fingerprint() -> dict[str, str]:
    """sha256 of every source file that decides what a render is.

    `max_tasks_per_child=200` means workers respawn and **re-import** during a
    run, so editing one of these files while a job is running silently changes
    what that job produces. Measured 2026-08-22: roughly 1,700 assets were
    rendered with a geometry fix and stamped with the `renderer_version` that
    predated it, and **no field in any artifact revealed it** -- the corpus had
    to be discarded because no subset could be identified by inspection.

    The version field caught that by accident, not by design: had the bump
    landed WITH the fix, the corpus would have been uniformly stamped and every
    gate would have passed.

    `meshload` is included because it owns `FRAME_CORRECTION` -- a change there
    moves every asset while `renders.py` does not differ by a byte.
    """
    from metafind.data import meshload

    out: dict[str, str] = {}
    for mod in (sys.modules[__name__], meshload):
        path = Path(mod.__file__)
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def verify_fingerprint(expected: dict[str, str] | None) -> None:
    """Abort this worker if its source differs from what the run started on.

    The point is not to forbid editing a module. It is that a run's behaviour
    must not be able to change without the artifacts saying so.
    """
    global _FINGERPRINT_VERIFIED
    if not expected or _FINGERPRINT_VERIFIED:
        return
    actual = implementation_fingerprint()
    if actual != expected:
        drift = sorted(k for k in expected if actual.get(k) != expected[k])
        raise RuntimeError(
            f"implementation changed while the run was in progress: {', '.join(drift)}. "
            "This worker would write artifacts the rest of the corpus does not share, "
            "and no sidecar field would show it. Restart the run."
        )
    _FINGERPRINT_VERIFIED = True



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
    """
    sc = sidecar_path(out_dir, uid)
    if not sc.exists():
        return False
    sc.replace(sc.with_suffix(".json.stale"))
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
    if len(rec.get("view_paths", [])) != N_VIEWS:
        return False
    sizes = rec.get("view_bytes")
    if sizes is None or len(sizes) != N_VIEWS:
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
    for path_str, want in zip(rec["view_paths"], sizes):
        try:
            if Path(path_str).stat().st_size != want:
                return False
        except OSError:
            return False
    return True


def process_one(uid: str, glb: Path, out_dir: Path,
                fingerprint: dict[str, str] | None = None) -> dict:
    # Before any rendering: this worker may have re-imported a module that
    # changed since the run started. See `verify_fingerprint`.
    verify_fingerprint(fingerprint)

    from PIL import Image

    images, extents = render_views(glb)
    if len(images) != N_VIEWS:
        raise ValueError(f"rendered {len(images)} views, expected {N_VIEWS}")

    asset_dir = out_dir / uid
    asset_dir.mkdir(parents=True, exist_ok=True)
    view_paths, view_sha, view_bytes, blank = [], [], [], 0
    for i, img in enumerate(images):
        # A camera pointed at nothing returns a uniform frame. It is a valid
        # PNG of the right shape, so only the content says anything is wrong.
        if float(img.std()) < 1.0:
            blank += 1
        p = asset_dir / f"view_{i:02d}.png"
        tmp = p.with_suffix(".png.part")
        Image.fromarray(img).save(tmp, format="PNG", optimize=False)
        tmp.replace(p)
        view_paths.append(str(p))
        blob = p.read_bytes()
        view_sha.append(hashlib.sha256(blob).hexdigest())
        view_bytes.append(len(blob))

    if blank == N_VIEWS:
        raise ValueError("every view is blank -- the asset never entered frame")
    if len(set(view_sha)) != N_VIEWS:
        raise ValueError(
            f"only {len(set(view_sha))} distinct views of {N_VIEWS}; the camera "
            "is not moving between renders"
        )

    record = {
        "uid": uid,
        "view_paths": view_paths,
        "view_sha256": view_sha,
        "view_bytes": view_bytes,
        "raw_bbox_extents": [float(v) for v in extents],
        "projection": PROJECTION,
        "camera_layout": CAMERA_LAYOUT,
        "orbit_elevation_deg": ORBIT_ELEVATION_DEG,
        "resolution": RESOLUTION,
        "renderer_version": RENDERER_VERSION,
        # How each choice was arrived at, so a reader never has to guess which
        # of these the paper actually specifies. Only n_views does.
        "camera_layout_source": "upstream_informed_choice",
        "projection_source": "implementation_choice",
        "n_views_source": "paper",
        "blank_views": blank,
    }
    sc_tmp = sidecar_path(out_dir, uid).with_suffix(".json.part")
    with sc_tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sc_tmp.replace(sidecar_path(out_dir, uid))
    return record


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
                f.write(json.dumps(json.loads(sc.read_text())) + "\n")
                n += 1
            except (OSError, json.JSONDecodeError):
                continue
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Separate PROCESSES: EGL contexts are not shareable across "
                         "threads, and 4 threads produced a 34% eglDestroyContext "
                         "failure rate that looked like bad assets.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
    glb_by_uid = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    out_dir = paths.RENDERS
    out_dir.mkdir(parents=True, exist_ok=True)

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
    with runlog.run_progress(NODE):
        for start in range(0, len(todo), BATCH):
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
                                "failure_class": ("RESOURCE" if isinstance(exc, (MemoryError, OSError))
                                                  else "DETERMINISTIC_INPUT"),
                                "exception_type": type(exc).__name__,
                                "exception_msg": str(exc)[:400],
                                "traceback": traceback.format_exc()[-1500:],
                            }])
                            quarantine.append(uid)
                            continue
                        done += 1
                        if done % 200 == 0:
                            rate = done / max(time.time() - started, 1e-9) * 60
                            print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                                  f"剩餘約 {(len(todo)-done)/max(rate,1e-9):.0f} 分, "
                                  f"quarantine {len(quarantine)}", flush=True)
            except cf.process.BrokenProcessPool as exc:
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

    n_indexed = rebuild_index(paths.LOGS / "renders_index.jsonl", out_dir)
    runlog.cost_ledger(
        cpu_seconds=round(time.time() - started, 1),
        assets_rendered=done,
        views_written=done * N_VIEWS,
    )
    print(f"\n{done:,} rendered this run, {n_indexed:,} complete on disk, "
          f"{len(quarantine):,} quarantined -> {out_dir}")
    return 0 if done or not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
