"""Sample one 10,000-point xyz+rgb cloud per Objaverse asset.

# IMPLEMENTS-NODE: n03_sample_pointclouds

The output has to be readable by a frozen ULIP-2 checkpoint that was trained on
someone else's clouds, so the format is not ours to choose. Three properties are
copied from ULIP's own loader rather than decided here:

**pc_norm applies to xyz ONLY.** ``dataset_3d.py`` line 496 normalises the
coordinates -- centroid to the origin, largest radius to 1 -- and *then*
concatenates rgb. Normalising all six columns together would rescale colour by
the object's physical extent, which is a different tensor with the same shape.

**rgb lives in [0, 1], not [0, 255] -- STRONGLY INDICATED, not proven.**
Neither paper says. The evidence is that ULIP substitutes
``np.ones_like(point_set) * 0.4`` (``dataset_3d.py`` 292 and 297) where a
dataset has no colour, and on a 0-255 scale that stand-in would be about 102.

But note exactly what that shows and what it does not. Those two lines are in
the *ModelNet* path. ``Objaverse_Lvis_Colored.__getitem__`` reads ``rgb`` out
of the released .npy and concatenates it with no division and no clamp, so it
inherits whatever scale the released file already uses. The 0.4 fallback is
evidence about ULIP's colour CONVENTION, not a measurement of the Objaverse
release. Settling it needs one official .npy read directly -- that measurement
is tracked with U-02 and the result belongs here.

Getting it wrong is silent and costs a factor of 255 on the colour channel of
every asset, so ``rgb_scale: "unit"`` goes in every sidecar: a later reader can
check what was assumed instead of re-deriving it.

**10,000 points, xyzrgb.** ULIP-2's Appendix A.1 measures 10k xyzrgb at
50.6/79.1 on Objaverse-LVIS against 8k xyz at 48.9, and 50.6 is the abstract's
headline. The checkpoint we load is named for it.

What this deliberately does not do
----------------------------------

No farthest-point sampling. ULIP's ``Objaverse_Lvis_Colored`` reads clouds that
already hold exactly 10,000 points and draws from them with a shuffled
permutation; FPS appears in the ShapeNet path, not this one. Area-weighted
surface sampling is the closest honest reconstruction of "10,000 points from
this mesh", and U-02 records that our clouds are not ULIP's released clouds --
measured by L2-PC-ULIP-REF as a diagnostic, not a gate, because the paper never
claims MetaFind reuses them and Stage 1 trains the point encoder anyway.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import sys
import os
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog
from metafind.data import meshload

NODE = "n03_sample_pointclouds"
N_POINTS = 10_000
# 3: apply the scene-graph transform before sampling. Measured on 400 random
# assets, 65.8% carry a non-identity transform, and on 40 re-sampled assets 16
# changed extent by more than 1% -- ratios from 0.008x to 77x. Versions 1-2
# produced geometrically wrong clouds for a large fraction of the corpus while
# passing every G2 check.
# 4: [CORRECTED 2026-08-22] the mesh is loaded through `meshload`, which applies
# the 180 degree yaw that puts our clouds in ULIP-2's frame, and recovers the
# glTF COLOR_0 vertex colours trimesh discards when a material is present.
#
# The bump is not cosmetic. `is_complete()` reads it, so a version-3 cloud is
# NOT complete under version 4 and a bare re-run regenerates it. Leaving the
# number at 3 meant every one of the 46,052 uncorrected clouds classified as
# finished: the re-run would have skipped the entire corpus and reported
# success. Found by the Reviewer, 2026-08-22, before the run.
# 5: [P3, 2026-08-22] COLOR_0 becomes a MULTIPLIER on the base colour rather
# than a competing source. Changes the rgb channel for roughly 1,130 assets
# whose base is `flat` or `texture`; the ~146 whose base is glTF-default white
# are unchanged by construction, since x * 1 = x.
# 6: [R-12, 2026-08-22] the COLOR_0 multiplier is withdrawn from the `texture`
# class. Measured over all 37 texture assets carrying COLOR_0 that ULIP also
# publishes: modulating darkens 37 of 37 by 0.21 mean, and cosine against
# ULIP's own clouds moves 0.9005 -> 0.8980. `flat` and `gltf_default` are
# unchanged from version 5.
# 7: [2026-08-23] colour is interpolated AT the sampled point instead of being
# the mean of the triangle's three baked vertex colours, and `baseColorFactor`
# is multiplied into the texture. OpenShape §3.2 -- the source ULIP-2's appendix
# defers its 3D input preprocessing to -- says "we sample 10,000 points from the
# mesh surface and interpolate the point colors according to the mesh textures".
# Version 6 gave every point on a face one colour, which is a per-face constant.
# Changes the rgb channel of the `texture` class, 23,675 of 46,052 assets;
# `flat`, `gltf_default` and `fallback_grey` are uniform per part and are
# therefore arithmetically unchanged.
SAMPLER_VERSION = 8
RGB_SCALE = "unit"  # [0, 1]; see the module docstring
DEFAULT_GREY = 0.4  # ULIP's stand-in for a dataset with no colour channel at all
# The base colour of a PBR material carrying neither a texture nor an explicit
# `baseColorFactor`.
#
# [INFERENCE, corrected 2026-08-22] The claim "glTF 2.0 specifies
# baseColorFactor default = [1,1,1,1]" is common knowledge and it is what this
# value assumes, but **the specification text has not been read here** -- the
# glTF schema is not on disk in this project. Flagged rather than restated as a
# fact, because 8,853 `gltf_default` assets rest on it. Resolving it needs
# `material.pbrMetallicRoughness.schema.json` from the Khronos glTF repository.
#
# [OBSERVED DATA] What IS measured, and it points the same way: over the 50
# `gltf_default` assets that overlap ULIP's released clouds and carry no
# COLOR_0, ULIP's own clouds are 35.3% pure-white points against our 35.1%,
# with the all-white asset count identical at 19/50. Under the alternative --
# trimesh's 0.4 grey, numerically identical to DEFAULT_GREY, which is exactly
# why it once looked like a legitimate fallback -- our figure would be 0%.
# Upstream agrees with white on the population where the two can be compared.
GLTF_DEFAULT_BASE_COLOR = 1.0


def uid_seed(uid: str) -> int:
    """A stable per-asset seed, so the same mesh always yields the same cloud.

    Derived from the uid rather than a counter: a counter makes the cloud depend
    on how many assets were processed before it, which turns a resumed run into
    a different dataset.
    """
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big") % (2**32)


def pc_norm(xyz: np.ndarray) -> np.ndarray:
    """ULIP's normalisation, verbatim (dataset_3d.py:496-502)."""
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt((xyz**2).sum(axis=1)))
    return xyz / m


def load_parts(path: Path):
    """Every triangulated geometry in the GLB, each with explicit colours.

    Returns ``(parts, worst_colour_source, per_part_sources)``.

    [CORRECTED TWICE] First version loaded the scene, called ``to_geometry()``
    and read colours off the merged mesh: concatenating happens before
    conversion, so every per-geometry material was gone and all 40 test assets
    came out uniformly grey. Second version converted per geometry and then
    concatenated, which kept the colours but produced a mesh whose faces index
    past the end of the merged colour array -- 14 of 40 died on an IndexError.
    Nothing is merged now. Each geometry is sampled on its own, where its face
    indices and its colour array are guaranteed to agree, and the parts are
    combined only as POINTS.
    """
    import trimesh

    from metafind.data import meshload

    # The frame correction and the COLOR_0 recovery both live in `meshload` so
    # that n03 and n04 cannot drift apart -- see that module's docstring. A
    # cloud and a render in different frames raise nothing anywhere.
    scene = meshload.load_scene(path)
    color0 = meshload.color0_by_geometry(path)
    parts, sources, modulations = [], [], []
    # Iterate the scene GRAPH, not scene.geometry. Two reasons, both measured:
    #
    #   * 65.8% of 400 sampled Objaverse GLBs place their geometry with a
    #     non-identity node transform, translations up to 1.2e4. Sampling the
    #     raw geometry drops it and assembles the object collapsed on itself.
    #     On 40 re-sampled assets, 16 changed extent by over 1%, up to 77x.
    #   * One geometry can be INSTANCED at several nodes -- four identical
    #     chairs around a table are one mesh and four transforms, and looping
    #     over scene.geometry would see one chair. NOT observed in this corpus:
    #     0 of 300 sampled assets instance anything. Iterating the graph gets
    #     it for free, so it is handled, but it is not a measured problem here
    #     and must not be reported as one.
    #
    # Either way the cloud keeps 10,000 points, normalises to a unit sphere and
    # passes every G2 check. It is simply the wrong object.
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        geom.apply_transform(transform)
        # Keyed by geometry NAME, and the name is produced by the same trimesh
        # loader in both passes. Instanced geometry shares one COLOR_0 array,
        # which is correct: the colours belong to the mesh, not to the node.
        source, modulated = _colourise(geom, color0.get(geom_name))
        sources.append(source)
        modulations.append(modulated)
        parts.append(geom)

    if not parts:
        raise ValueError("no triangulated geometry in this GLB")
    order = ("fallback_grey", "gltf_default", "flat", "vertex", "face", "texture")
    # The WORST source across parts, so an asset that is nine-tenths textured
    # and one-tenth grey is not labelled "texture". coloured_point_fraction
    # carries the degree, which the label alone cannot.
    return parts, min(sources, key=order.index), sources, any(modulations)


def _colourise(geom, color0: np.ndarray | None = None) -> tuple[str, bool]:
    """Force an explicit per-vertex colour array onto one geometry.

    Returns ``(base_colour_source, color0_modulated)``.

    glTF 2.0 base-colour semantics, adopted 2026-08-22 as decision **P3**::

        base colour  =  baseColorFactor  x  baseColorTexture
        final        =  base colour      x  COLOR_0

    ``COLOR_0`` is a **linear multiplier on the base colour, not a replacement
    for it.** That distinction is the whole of `R-2`: the earlier code treated
    it as a competing source ranked "above flat", so whichever of the two the
    material path happened to reach first won and the other was discarded.

    Why P3 and not the alternatives
    -------------------------------

    Three readings were on the table -- keep the factor and ignore ``COLOR_0``
    (P1, what the code did for ~926 assets), let ``COLOR_0`` replace it (P2,
    what the docstring claimed), or multiply them (P3, the specification).

    The Reviewer measured all three the way `FIND-7` and `S-5` measure things:
    each candidate cloud through the **frozen ULIP-2 point encoder**, scored by
    cosine against **ULIP's own released cloud for the same object**. 130 assets
    where the three rules actually differ, drawn from 3,706 overlapping uids::

        P1  discard COLOR_0     mean 0.8800   median 0.8845   best on  27
        P2  COLOR_0 replaces    mean 0.9043   median 0.9204   best on  54
        P3  COLOR_0 multiplies  mean 0.9004   median 0.9195   best on  49

    **P1 is decisively worst -- 27 against 103.** `R-2` was never a
    documentation argument; the old behaviour fed the frozen encoder measurably
    worse input.

    **P2 and P3 are not separable at this n.** They differ by 0.004 mean and 5
    of 130 head-to-head, and a fair coin over 130 has sd ~5.7, so that sits
    inside one sigma. No paired significance test was run. **P2 is not the
    winner.** Where upstream can discriminate it decides (P1 is out); where it
    cannot, the glTF 2.0 specification breaks the tie, and it says multiply.

    Source naming
    -------------

    ``colour_source`` keeps naming the BASE colour -- ``texture`` / ``flat`` /
    ``gltf_default`` -- because under P3 ``COLOR_0`` modulates a source rather
    than being one. Calling a modulated texture ``vertex`` would both lose the
    base and silently shift the corpus-wide distribution. Modulation is carried
    separately as ``color0_modulated`` so it stays visible and countable.

    Which classes are modulated -- narrowed by measurement, not by principle::

        gltf_default   COLOR_0 x [1,1,1,1] = COLOR_0     modulated
        flat           COLOR_0 x baseColorFactor         modulated
        texture        COLOR_0 NOT applied               DEVIATION, see below

    Four base sources, in descending fidelity:

      texture        a baseColorTexture, sampled per vertex through the UVs
      flat           no texture, but the PBR material carries a baseColorFactor
      gltf_default   a material with neither: see GLTF_DEFAULT_BASE_COLOR
      vertex / face  colours stored on the geometry itself
      fallback_grey  nothing readable; ULIP's own 0.4 stand-in

    [CORRECTED] `flat` was missing, and its absence cost 45% of the test batch
    its colour. trimesh's ``to_color()`` on a texture-less PBR material returns
    a vertex_colors array of length FOUR -- one RGBA, not one per vertex -- so
    the length check rejected it and fell through to grey. But
    ``baseColorFactor = [204 200 176 255]`` is a real material colour; a mesh
    that is uniformly beige is not a mesh whose colour is unknown.

    [NO BLANKET except] An earlier version wrapped this in `except Exception:
    return grey`, which is why the first run produced 40 of 40 uniformly grey
    assets: ``to_color()`` raises IndexError on some of these materials and the
    handler turned a fixable bug into a plausible-looking default. Failures are
    narrow now, and the source is RECORDED per asset, so lost colour shows up
    in the summary rather than in Table 1.
    """
    import trimesh

    vis = geom.visual
    n = len(geom.vertices)

    def _rgba(colours) -> np.ndarray:
        """``(n, 4)`` uint8, expanding trimesh's collapsed single-RGBA form."""
        a = np.asarray(colours)
        if a.ndim == 1:
            a = np.tile(a[:4], (n, 1))
        if a.shape[1] == 3:
            a = np.concatenate([a, np.full((len(a), 1), 255)], axis=1)
        return a.astype(np.uint8, copy=False)

    def _commit(base, source: str, *, modulate: bool = True) -> tuple[str, bool]:
        """Install ``base``, multiplying `COLOR_0` into it unless told not to.

        Multiplication happens in [0, 1] and rounds once at the end, so a
        modulated colour cannot drift by repeated integer truncation.

        ``modulate=False`` is the `texture` carve-out -- see the class table in
        the docstring. It is a measured exception, not a shortcut.
        """
        rgba = _rgba(base)
        modulated = (modulate and color0 is not None
                     and len(np.asarray(color0)) == n)
        if modulated:
            rgba = np.clip(
                np.rint(rgba.astype(np.float64) / 255.0
                        * _rgba(color0).astype(np.float64) / 255.0 * 255.0),
                0, 255).astype(np.uint8)
        geom.visual = trimesh.visual.ColorVisuals(mesh=geom, vertex_colors=rgba)
        return source, modulated

    if isinstance(vis, trimesh.visual.TextureVisuals):
        # [SAMPLER_VERSION 8] A texture that can be read PER POINT is left
        # alone, so `sample_mesh` can interpolate it at the sampled position
        # instead of averaging the triangle's three baked vertex colours.
        #
        # This is what the named upstream source actually specifies. OpenShape
        # §3.2 -- and ULIP-2's appendix defers its 3D input preprocessing to
        # OpenShape -- says: "we sample 10,000 points from the mesh surface and
        # **interpolate the point colors according to the mesh textures**."
        # Baking to vertices first and then averaging gives every point on a
        # triangle the SAME colour, which is a per-face constant, not an
        # interpolation. On a coarse mesh with a detailed texture that is the
        # whole texture gone.
        #
        # Scope: the `texture` class, 23,675 of 46,052 assets.
        #
        # `_commit` is deliberately NOT called here -- it would overwrite
        # `geom.visual` with ColorVisuals and destroy the UVs the interpolation
        # needs. COLOR_0 stays excluded from this class exactly as `D-12` has
        # it; that decision was measured under the old per-face rule and is
        # re-measured by `V1.0` before the full run.
        if _texture_is_samplable(geom):
            return "texture", False
        try:
            converted = vis.to_color()
            vc = getattr(converted, "vertex_colors", None)
            if vc is not None and len(vc) == n:
                # A real per-vertex texture sample.
                #
                # [SCOPE NARROWED 2026-08-22, `R-12`] glTF 2.0 says COLOR_0
                # multiplies the base colour and the base colour includes the
                # texture, so the specification puts this asset in scope. **We
                # deliberately leave it out**, and that is a DEVIATION.
                #
                # Measured over ALL 37 texture assets that carry COLOR_0 and
                # also exist in ULIP's released clouds -- the whole population,
                # not a sample:
                #
                #     brightness, modulated minus not:  mean -0.2076,
                #         median -0.1821, DARKER on 37 of 37
                #     cosine against ULIP's own cloud through the frozen
                #         encoder: 0.8980 modulated vs 0.9005 not
                #
                # The darkening is certain: 37/37 at 0.21 on a 0-1 scale.
                # Whether it is *worse* is not -- 0.0025 and 16/37 sits inside
                # the noise for 37 coin flips (18.5 +/- 3), and no paired test
                # was run. What decides it is `R-11`: ULIP-2 is the reference
                # architecture, so agreeing with it is the default and only a
                # deliberate divergence is registered. Asked twice, the
                # specification-correct rule never won -- R-10 over all three
                # classes (n=130) 0.9043 unmodulated vs 0.9004, and R-12 over
                # texture alone (n=37) 0.9005 vs 0.8980.
                #
                # R-10's "tie" across the mixed population is consistent with
                # two opposite effects cancelling; splitting by class showed it.
                return _commit(vc, "texture", modulate=False)
            # `len(vc) == 4` was the test here and it is ambiguous: a single
            # RGBA has length 4, and so does a four-vertex mesh's per-vertex
            # array. `ndim == 1` says what was meant. The check above
            # (`len(vc) == n`) already catches the four-vertex case first.
            if vc is not None and np.asarray(vc).ndim == 1:
                return _commit(vc, "flat")
        except (IndexError, ValueError, TypeError, AttributeError):
            pass
        mat = getattr(vis, "material", None)
        factor = getattr(mat, "baseColorFactor", None)
        if factor is not None:
            return _commit(np.asarray(factor).ravel(), "flat")
        if mat is not None:
            return _commit([int(GLTF_DEFAULT_BASE_COLOR * 255)] * 3 + [255],
                           "gltf_default")
        # A TextureVisuals with no material at all still has to reach COLOR_0.
        if color0 is not None and len(np.asarray(color0)) == n:
            return _commit([255, 255, 255, 255], "gltf_default")
    else:
        vc = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc) == n:
            # trimesh surfaced COLOR_0 itself, so it is already the base here
            # and must NOT be multiplied in again -- that is the "squared"
            # failure the Reviewer's safety check looks for.
            return "vertex", False
        fc = getattr(vis, "face_colors", None)
        if fc is not None and len(fc) == len(geom.faces):
            return "face", False

    return _commit([int(DEFAULT_GREY * 255)] * 3 + [255], "fallback_grey")


def _material_image(vis):
    """The base-colour texture image, whichever material class holds it.

    [SAMPLER_VERSION 8, ROOT CAUSE OF A SILENT NO-OP] ``PBRMaterial`` -- what
    trimesh 5.0 builds for every glTF PBR material, which is what these assets
    use -- keeps the texture on ``baseColorTexture`` and has **no ``.image``
    attribute at all**. ``SimpleMaterial`` keeps it on ``.image``. Version 7
    checked only ``.image``, so `_texture_is_samplable` answered False for every
    PBR asset and the per-point branch never executed.

    Measured before the fix, over 200 assets drawn from the `texture` class:
    ``_texture_is_samplable`` returned True on **0 of 1,248 parts (0.00%)**.
    Reading the glTF JSON of 60 of them directly, bypassing trimesh: 60 of 60
    carry ``images`` and 54 of 60 carry a ``baseColorTexture``. The textures
    were there; the attribute name was wrong.
    """
    mat = getattr(vis, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        img = getattr(mat, "image", None)
    return img


def _texture_is_samplable(geom) -> bool:
    """Can this geometry's texture be read at an arbitrary surface point?

    Needs three things, all checked here so that `_colourise` and `_sample_part`
    can never disagree about which branch an asset takes: UV coordinates, one
    per vertex so the barycentric interpolation indexes them by the same face
    array, and an actual image to sample.

    A material carrying UVs but no image -- a baseColorFactor-only PBR material
    still typed as TextureVisuals -- stays on the ``to_color()`` path rather
    than being quarantined.
    """
    vis = getattr(geom, "visual", None)
    uv = getattr(vis, "uv", None)
    if uv is None:
        return False
    uv = np.asarray(uv)
    if len(uv) == 0 or len(uv) != len(geom.vertices):
        return False
    return _material_image(vis) is not None


def _base_colour_factor(vis) -> np.ndarray | None:
    """``baseColorFactor`` as ``(3,)`` in [0, 1], or None if the material has none.

    [IMPLEMENTATION CHOICE -- ATTRIBUTION PENDING, corrected 2026-08-23] This
    docstring previously asserted ``[USER DECISION, 2026-08-23]``. Master
    searched the history (``git log -S"_base_colour_factor"`` returns 0
    commits: the helper exists only in the uncommitted tree) and the Engineer
    searched this session's transcript; **neither can produce the USER's words
    for it**. The label is therefore withdrawn until he confirms it. He may
    well have decided it in an earlier session -- "pending" says unverified,
    not denied.

    The argument stands on its own either way, and is why the behaviour is
    kept while the attribution is not: glTF 2.0 defines the base colour as
    ``baseColorFactor * baseColorTexture``. ``trimesh``'s ``sample_color`` reads
    ``material.image`` only -- the raw texture, unmultiplied -- so the factor is
    applied by us or it is silently dropped.

    Most PBR materials carrying a texture leave the factor at [1,1,1,1], where
    this is a no-op; the ones that tint a greyscale texture are exactly the ones
    it matters for.
    """
    factor = getattr(getattr(vis, "material", None), "baseColorFactor", None)
    if factor is None:
        return None
    f = np.asarray(factor, dtype=np.float64).ravel()[:3]
    if f.size != 3:
        return None
    # glTF stores it in [0, 1]; trimesh sometimes surfaces it as uint8 [0, 255].
    if f.max() > 1.0:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def _vertex_rgb(geom) -> np.ndarray:
    """``(n_vertices, 3)`` uint8 colours, whatever shape trimesh handed back.

    trimesh COLLAPSES a uniform vertex_colors array to a single RGBA, so a mesh
    whose vertices all share one colour returns shape (4,) rather than (n, 4).
    Indexing that by a face array raises IndexError, and the asset is
    quarantined as though its geometry were broken.

    Missed entirely by a 60-asset smoke run -- it needs a small uniform-coloured
    part, and it cost 8.1% of an 800-asset sample, every one of them with the
    same message. Four times G3's quarantine ceiling, from one shape assumption.
    """
    vc = np.asarray(geom.visual.vertex_colors)
    n = len(geom.vertices)
    if vc.ndim == 1:  # one RGBA for the whole geometry
        return np.tile(vc[:3], (n, 1))
    if vc.shape[0] != n:  # per-face, or otherwise not per-vertex
        return np.tile(vc.reshape(-1, vc.shape[-1])[0, :3], (n, 1))
    return vc[:, :3]


def _allocate(areas: np.ndarray, n_points: int) -> np.ndarray:
    """Split n_points across parts in proportion to surface area.

    Largest-remainder, so the parts sum to exactly n_points: a cloud of 9,998
    points would pass no shape check and fail the one that matters.
    """
    if areas.sum() <= 0:
        counts = np.zeros(len(areas), dtype=int)
        counts[0] = n_points
        return counts
    exact = areas / areas.sum() * n_points
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[: n_points - counts.sum()]:
        counts[i] += 1
    return counts


def _sample_part(part, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(points (k,3), colours (k,3) uint8-scale float)`` for one geometry.

    [SAMPLER_VERSION 8] Both branches interpolate the colour AT the sampled
    point, which is what OpenShape §3.2 specifies. Version 6 averaged the
    triangle's three vertex colours, giving every point on a face one colour.

      * **textured part** -- the sample's UV is interpolated from the same
        barycentric weights that placed the point, then read out of the texture
        bilinearly. ``baseColorFactor`` is multiplied in afterwards, because the
        lookup returns the raw texel and glTF defines the base colour as
        factor x texture.
      * **everything else** -- the colour is per-vertex, so the barycentric
        weights are recovered and applied to the face's three vertex colours.
        For a uniformly coloured part (``flat``, ``gltf_default``,
        ``fallback_grey``, and any per-face colour) this is arithmetically
        identical to the old mean, so those classes are unchanged by
        construction.

    [SAMPLER_VERSION 8] Version 7 delegated the textured branch to
    ``sample_surface(sample_color=True)``. That call reaches
    ``material.image``, which **``PBRMaterial`` does not have** -- it raises
    ``AttributeError`` on every glTF PBR asset in this corpus. The branch was
    never taken (the predicate excluded it first), so nothing crashed and
    nothing was interpolated either. The UV lookup is done here now, against
    ``_material_image``, so the branch both fires and works.
    """
    import trimesh

    vis = part.visual
    if isinstance(vis, trimesh.visual.TextureVisuals) and _texture_is_samplable(part):
        pts, face_idx = trimesh.sample.sample_surface(part, k, seed=seed)
        pts = np.asarray(pts)
        bary = _barycentric(part, face_idx, pts)                   # (k, 3)
        corners_uv = np.asarray(vis.uv, dtype=np.float64)[part.faces[face_idx]]
        point_uv = (bary[:, :, None] * corners_uv).sum(axis=1)     # (k, 2)
        cols = trimesh.visual.color.uv_to_interpolated_color(
            point_uv, _material_image(vis))[:, :3].astype(np.float64)
        factor = _base_colour_factor(vis)
        if factor is not None:
            cols = cols * factor[None, :]
        return pts, cols

    pts, face_idx = trimesh.sample.sample_surface(part, k, seed=seed)
    pts = np.asarray(pts)
    corners = _vertex_rgb(part)[part.faces[face_idx]].astype(np.float64)  # (k,3,3)
    bary = _barycentric(part, face_idx, pts)                              # (k, 3)
    return pts, (bary[:, :, None] * corners).sum(axis=1)


def _barycentric(part, face_idx: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``(k, 3)`` weights of each point inside its own face.

    ``points_to_barycentric`` can emit a NaN on a degenerate (zero-area)
    triangle. Those faces carry no area weight so they are rarely sampled, but
    "rarely" is not "never" over 46,052 assets, and one NaN propagates into the
    stored cloud -- as a colour on one branch and as a UV on the other. Fall
    back to the face centroid for exactly those rows.
    """
    import trimesh

    bary = trimesh.triangles.points_to_barycentric(part.triangles[face_idx], pts)
    bad = ~np.isfinite(bary).all(axis=1)
    if bad.any():
        bary = bary.copy()
        bary[bad] = 1.0 / 3.0
    return bary


def sample_mesh(path: Path, seed: int, n_points: int = N_POINTS):
    """Area-weighted surface sample with per-point colour.

    Returns ``(xyz, rgb, extents, colour_source)``. ``extents`` is the
    UNtransformed bounding box in metres: once the cloud is unit-normalised the
    real scale is gone, so F13 records it here or nowhere.
    """
    import trimesh

    parts, colour_source, sources_per_part, color0_modulated = load_parts(path)
    areas = np.array([p.area for p in parts], dtype=np.float64)
    counts = _allocate(areas, n_points)

    xyz_chunks, rgb_chunks, coloured = [], [], 0
    for i, (part, k) in enumerate(zip(parts, counts)):
        if k == 0:
            continue
        # Per-part seed, so a part's points do not depend on how many points
        # the parts before it happened to receive.
        pts, cols = _sample_part(part, int(k), int(seed) + i)
        if sources_per_part[i] != "fallback_grey":
            coloured += int(k)
        xyz_chunks.append(np.asarray(pts, dtype=np.float32))
        rgb_chunks.append(np.clip(cols.astype(np.float32) / 255.0, 0.0, 1.0))

    xyz = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    if len(xyz) != n_points:
        raise ValueError(f"sampled {len(xyz)} points, expected {n_points}")

    lo = np.min([p.vertices.min(axis=0) for p in parts], axis=0)
    hi = np.max([p.vertices.max(axis=0) for p in parts], axis=0)
    return (xyz, rgb, np.asarray(hi - lo, dtype=np.float64),
            colour_source, coloured / max(n_points, 1), color0_modulated)


def sidecar_path(out: Path) -> Path:
    return out.with_suffix(".json")


def is_complete(out: Path) -> bool:
    """Whether this asset is finished, in the only sense that survives a crash.

    A present .npz proves nothing. The old check was `out.exists()`, and the
    write order was: rename the npz, return, and let the main thread append to
    a shared JSONL. Kill -9 between those two and the asset is permanently
    skipped on restart with its canonical metadata missing -- measured, not
    hypothetical: interrupting the first full run left 18 clouds with no
    record. G2 routes on centroid_offset, max_radius and per_axis_variance,
    which live only in that record.

    The per-asset sidecar is therefore the completion marker and is written
    LAST, after the npz is already in place. Crashing between them costs one
    re-sample, and re-sampling is deterministic (uid_seed), so the recovered
    cloud is byte-identical to the lost one.
    """
    sc = sidecar_path(out)
    if not (out.exists() and sc.exists()):
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # [ADDED 2026-08-22] Completion is relative to the CURRENT sampler. Without
    # this the check answered "is there an intact cloud here", which is not the
    # question a re-run asks -- it asks "is there an intact cloud produced by
    # the code I am running now". The 2026-08-22 frame and COLOR_0 corrections
    # changed the geometry and the colours while the digest of each OLD file
    # stayed self-consistent, so every stale cloud passed. A bare re-run would
    # have skipped all 46,052 and exited reporting success.
    #
    # Deliberately `!=` rather than `<`: a sidecar from a NEWER sampler is also
    # not this sampler's output, and silently accepting it would let a
    # downgraded run inherit artifacts it cannot reproduce.
    if rec.get("sampler_version") != SAMPLER_VERSION:
        return False
    # Also catches a truncated npz: a half-written file has a different digest.
    return rec.get("sha256") == hashlib.sha256(out.read_bytes()).hexdigest()


def process_one(uid: str, glb: Path, out: Path) -> dict:
    seed = uid_seed(uid)
    (xyz, rgb, extents, colour_source, coloured_fraction,
     color0_modulated) = sample_mesh(glb, seed)
    normed = pc_norm(xyz.astype(np.float64)).astype(np.float32)

    if not np.isfinite(normed).all() or not np.isfinite(rgb).all():
        raise ValueError("non-finite values after normalisation")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write through a file HANDLE. np.savez_compressed(path) appends ".npz" to
    # any name that lacks it, so passing `<uid>.part` silently produced
    # `<uid>.part.npz` and the rename below then failed on a file that was
    # never created -- 40 of 40 assets quarantined with a FileNotFoundError
    # that pointed at the destination rather than the cause.
    tmp = out.with_name(out.name + ".part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, xyz=normed, rgb=rgb)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out)

    record = {
        "uid": uid,
        "path": str(out),
        # L2-RESUME asserts preprocessing artifacts are byte-identical after a
        # kill -9 and restart. Without a digest that assertion has nothing to
        # compare, and the channel type promised one.
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_points": int(len(normed)),
        "seed": seed,
        "sampler_version": SAMPLER_VERSION,
        # [ADDED 2026-08-22] `meshload`'s docstring promised this travels into
        # every sidecar so a later reader can tell a corrected cloud from an
        # uncorrected one. It did not: the constant had zero references outside
        # its own module. A comment describing behaviour that does not happen is
        # worse than no comment, because it is believed.
        "frame_correction": meshload.FRAME_CORRECTION_ID,
        # [P3, 2026-08-22] `colour_source` names the BASE colour; COLOR_0 is a
        # multiplier on it, not a source. Recorded separately so the corpus-wide
        # source distribution is not silently reshaped by modulation, and so the
        # ~1,130 modulated assets stay countable.
        "color0_modulated": bool(color0_modulated),
        "rgb_scale": RGB_SCALE,
        "colour_source": colour_source,
        "coloured_point_fraction": round(coloured_fraction, 4),
        # [RENAMED from extents_m / volume_m3] This is the axis-aligned
        # BOUNDING BOX in the GLB's own units, not the mesh volume and not a
        # verified physical size: Objaverse authors set their own scale, and
        # nothing here checks that a metre is a metre. The old names asserted
        # both. It still matters -- once xyz is unit-normed nothing downstream
        # can recover any scale at all, and paper 2.3 wants "size dimensions"
        # in the annotations -- but it is a weak ground truth and must not be
        # used to score the annotator as if it were a strong one.
        "raw_bbox_extents": [float(v) for v in extents],
        "raw_bbox_volume": float(np.prod(extents)),
        # float64 for the STATISTICS, even though the cloud is stored float32.
        # Averaging 10,000 float32 values accumulates about 1e-5 of error, which
        # is larger than G2's 1e-5 centroid tolerance -- so 8 of 46,052 assets
        # were recorded as failing a check their data passes. Verified on one:
        # recomputed in float64 the centroid offset is 5.2e-09, not 1.15e-05.
        # The gate's threshold is not the problem and must not be widened; the
        # measurement was reporting the summation method rather than the cloud.
        "centroid_offset": float(np.abs(normed.astype(np.float64).mean(axis=0)).max()),
        "max_radius": float(np.sqrt((normed.astype(np.float64) ** 2).sum(axis=1)).max()),
        "per_axis_variance": [float(v) for v in normed.astype(np.float64).var(axis=0)],
    }

    # The completion marker, written last and atomically. Order matters: npz
    # first, sidecar second, so a sidecar can never describe a file that is
    # not there.
    sc_tmp = sidecar_path(out).with_suffix(".json.part")
    with sc_tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sc_tmp.replace(sidecar_path(out))
    return record


def rebuild_index(index_path: Path) -> int:
    """Regenerate pointclouds_index.jsonl from the per-asset sidecars.

    Derived, so it is always exactly what is on disk. The previous version was
    appended to as work completed, which meant a resumed run duplicated rows
    and an interrupted one left the index short -- and the index is what a
    reader treats as the record of the corpus.
    """
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as f:
        for sc in sorted(paths.POINTCLOUDS.glob("*.json")):
            try:
                f.write(json.dumps(json.loads(sc.read_text())) + "\n")
                n += 1
            except (OSError, json.JSONDecodeError):
                continue
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="process at most N assets (smoke runs)")
    # [2026-08-23] `--limit N` selects the first N by MANIFEST order, and
    # `--limit N` in a different node selects the first N of ITS remaining work
    # -- which is a different set. A three-node smoke run driven by --limit
    # therefore samples one set of assets, renders another, and then fails to
    # annotate because the renders it needs are for the third. Observed exactly
    # that today. `--uids-file` is how a cross-node smoke run is pinned; every
    # runner in this pipeline takes the same flag with the same meaning.
    ap.add_argument("--uids-file",
                    help="newline-separated uids; processes exactly these. Use "
                         "this, not --limit, whenever another node must run on "
                         "the same assets.")
    ap.add_argument("--force", action="store_true", help="re-sample even if the file exists")
    args = ap.parse_args()

    uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
    # n02 writes shard directories (glbs/000-021/<uid>.glb), the same layout
    # Objaverse publishes. Resolve by uid once rather than guessing the shard
    # from the manifest's .npy path, which is ULIP's naming and need not agree.
    glb_by_uid = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}

    if args.uids_file:
        want = [ln.strip() for ln in Path(args.uids_file).read_text().splitlines()
                if ln.strip()]
        missing = [u for u in want if u not in glb_by_uid]
        if missing:
            raise SystemExit(f"{len(missing)} uid(s) have no GLB, e.g. {missing[:3]}")
        uids = want

    todo = []
    for uid in uids:
        glb = glb_by_uid.get(uid)
        out = paths.POINTCLOUDS / f"{uid}.npz"
        if glb is None:
            continue  # n02 has not fetched it yet; not this node's failure
        if is_complete(out) and not args.force:
            continue
        todo.append((uid, glb, out))
    # `--limit` truncates the WORK, not the manifest: while n02 is still
    # downloading, the first N manifest entries are mostly not on disk yet, and
    # limiting there gives an empty smoke run that looks like success.
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(uids):,} in manifest, {len(todo):,} to sample", flush=True)
    paths.POINTCLOUDS.mkdir(parents=True, exist_ok=True)
    index_path = paths.LOGS / "pointclouds_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # DETERMINISTIC_INPUT -> quarantine, max_attempts 1: a non-manifold or empty
    # mesh fails identically forever, so retrying only spends time.
    quarantine, done, started = [], 0, time.time()

    # Recorded, not enforced -- and the difference is the executor below.
    #
    # `n04` aborts a worker whose source no longer matches the run, because its
    # ProcessPool respawns workers that RE-IMPORT: editing `renders.py` mid-run
    # changed what a running job produced, silently, and cost a 46,052-asset
    # corpus on 2026-08-22. `n03` uses a ThreadPool. Threads share one
    # interpreter and the module is imported once, so an edit during a run
    # cannot reach the run -- there is no drift here to catch.
    #
    # The fingerprint is still worth writing down: `runlog.code_revision()`
    # records the git SHA, which says nothing about UNCOMMITTED edits, and an
    # uncommitted edit is exactly what produced the `n04` incident.
    #
    # **Anyone converting this to a ProcessPool inherits the failure mode and
    # must add `runlog.verify_fingerprint` the way `n04` does.**
    fingerprint = runlog.implementation_fingerprint(sys.modules[__name__], meshload)
    print("implementation: " + "  ".join(f"{k} {v[:12]}" for k, v in fingerprint.items()),
          flush=True)

    with runlog.run_progress(NODE), \
            cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, u, g, o): u for u, g, o in todo}
        for fut in cf.as_completed(futures):
            uid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad mesh must not stop the run
                # The registry declares two classes with DIFFERENT policies:
                # DETERMINISTIC_INPUT quarantines after one attempt because a
                # non-manifold mesh fails identically forever, while RESOURCE
                # degrades and retries. Collapsing both into "quarantine" threw
                # away recoverable assets on a transient memory spike and
                # reported them as broken geometry.
                cls = ("RESOURCE" if isinstance(exc, (MemoryError, OSError))
                       else "DETERMINISTIC_INPUT")
                # Written NOW, not collected and flushed at the end. A 70-minute
                # run gave no view of what it was discarding until it finished,
                # and a crash lost every record because they lived only in a
                # list -- the same defect as buffering run_progress.
                runlog.quarantine(NODE, [{
                    "uid": uid, "failure_class": cls,
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                quarantine.append(cls)
                continue
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                left = (len(todo) - done) / max(rate, 1e-9)
                print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                      f"剩餘約 {left:.0f} 分, quarantine {len(quarantine)}", flush=True)

    # The aggregate index is DERIVED from the per-asset sidecars, never
    # appended to. Appending made it grow duplicates on every resumed run and
    # made it disagree with the filesystem the moment one was interrupted.
    n_indexed = rebuild_index(index_path)
    runlog.cost_ledger(
        cpu_seconds=round(time.time() - started, 1),
        assets_sampled=done,
        bytes_written=sum(
            (paths.POINTCLOUDS / f"{u}.npz").stat().st_size
            for u, _, o in todo if o.exists()
        ),
    )

    by_class = collections.Counter(quarantine)
    print(f"\n{done:,} sampled this run, {n_indexed:,} complete on disk, "
          f"{len(quarantine):,} quarantined {dict(by_class)} -> {paths.POINTCLOUDS}")
    # proceed_with_admitted: a partial corpus is a legitimate outcome here, and
    # G2_pc_sanity decides whether what survived is usable. Only an empty result
    # is a failure of this node.
    return 0 if done or not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
