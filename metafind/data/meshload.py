"""Load an Objaverse GLB once, in the corrected frame, with COLOR_0 recovered.

# SUPPORTS-NODE: n03_sample_pointclouds
# SUPPORTS-NODE: n04_render_views

Both corrections live here rather than in ``pointclouds.py`` and ``renders.py``
separately, because the two nodes read the same meshes. A frame fix applied to
one of them alone would leave the point cloud and the render in DIFFERENT
frames, and nothing downstream raises when that happens -- the cloud and the
image simply stop describing the same object from the same direction.

Correction 1 -- the frame
-------------------------

Our assets sit 180 degrees yawed about Y relative to ULIP-2's released clouds.
Measured over the 286 uids that overlap ULIP's ``000-009`` shard
(``workflow/blocks/ULIP2/evidence/n03_n04_upstream_verification.md`` FIND-6):
median Chamfer distance 0.0903 as-is against 0.0230 after a 180 degree yaw,
better on 269 of 286, and the count of assets beyond 0.1 falls from 137 to 7.

``(x, y, z) -> (-x, y, -z)``, determinant +1: a rotation, not a reflection. The
axis-aligned bounding box is INVARIANT under it -- x and z are negated, not
swapped -- so ``raw_bbox_extents`` is unchanged and n05's proportions anchor
does not move.

FIND-7 measured that this yaw does not move the ULIP-2 embedding (98.0% vs
97.5% R@1). It is corrected anyway because scene composition places assets with
real geometry, and because a corpus that disagrees with its own upstream
reference for no reason is a corpus nobody can check.

Correction 2 -- COLOR_0
-----------------------

glTF stores optional per-vertex colours in the ``COLOR_0`` attribute. **trimesh
5.0.0 discards it whenever the primitive also carries a material**: the visual
comes back as ``TextureVisuals`` and ``vertex_attributes`` is empty, under both
``process=True`` and ``process=False``. ``skip_materials=True`` recovers it as
``ColorVisuals``, verified against the raw accessor -- glTF
``[0.396, 0.392, 0.125]`` arrives as ``[101, 100, 32]``.

Measured 2026-08-22 against ULIP's official clouds, on the 62 overlapping assets
our sidecars label ``gltf_default``. Splitting them by whether the GLB carries
``COLOR_0``:

    with COLOR_0     n=12   ULIP median  0.0% white   ours 100.0% white
    without COLOR_0  n=50   ULIP median 35.3% white   ours  35.1% white

Where the attribute is absent we already match upstream, identically at 19/50 on
the all-white count. Where it is present we were wrong on every asset.
**COLOR_0 accounts for the entire discrepancy; there is no residual.** Corpus
scale, 200 assets sampled per class: 17.0% of ``gltf_default`` (about 1,505
assets), 7.5% of ``flat``, 2.0% of ``texture``.

Why the second load rather than always using ``skip_materials``: skipping
materials throws away real textures, which is the best colour source we have for
23,675 assets. The GLB's JSON chunk is read first -- cheap, no mesh decode -- and
the second load happens only for the assets that actually carry the attribute.
"""

# IMPLEMENTS-NODE: none -- this is a shared loader, not a pipeline node

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

__all__ = [
    "FRAME_CORRECTION",
    "FRAME_CORRECTION_ID",
    "has_color0",
    "color0_by_geometry",
    "load_scene",
]

# [OBSERVED DATA, FIND-6] 180 degrees about Y. det = +1.
FRAME_CORRECTION = np.diag([-1.0, 1.0, -1.0, 1.0])
# Travels into every sidecar so a later reader can tell a corrected cloud from
# an uncorrected one without re-measuring it.
FRAME_CORRECTION_ID = "yaw180_about_y@ulip2_frame"

_GLB_MAGIC = b"glTF"


def _gltf_json(path: Path) -> dict | None:
    """The JSON chunk of a binary glTF, without decoding any mesh.

    Returns ``None`` for a file that is not a GLB, rather than raising: the
    caller's job is to sample the mesh, and a non-GLB is trimesh's problem to
    report with its own message.
    """
    with open(path, "rb") as fh:
        header = fh.read(20)
        if len(header) < 20 or header[:4] != _GLB_MAGIC:
            return None
        json_len = struct.unpack("<I", header[12:16])[0]
        return json.loads(fh.read(json_len))


def has_color0(path: Path) -> bool:
    """Does any primitive in this GLB declare ``COLOR_0``?

    Reads the JSON chunk only. This is the gate that keeps the expensive second
    load off the ~83% of assets that would gain nothing from it.
    """
    js = _gltf_json(path)
    if js is None:
        return False
    return any(
        "COLOR_0" in primitive.get("attributes", {})
        for mesh in js.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    )


def color0_by_geometry(path: Path) -> dict[str, np.ndarray]:
    """``geometry name -> (n_vertices, 4) uint8``, for primitives carrying COLOR_0.

    Empty when the GLB declares no ``COLOR_0``, so the caller can treat "no
    colour attribute" and "colour attribute unreadable" the same way: neither
    contributes, and neither is an error.

    The colours come from trimesh's own ``skip_materials`` path rather than from
    a hand-rolled accessor reader, so normalisation, component type and
    sRGB handling stay trimesh's problem and cannot drift from the main load.
    Geometry names are produced by the same loader in both passes, which is what
    makes the mapping safe.
    """
    if not has_color0(path):
        return {}

    import trimesh

    try:
        scene = trimesh.load(path, force="scene", process=False, skip_materials=True)
    except Exception:  # noqa: BLE001 -- a second-chance read must never fail the asset
        return {}

    out: dict[str, np.ndarray] = {}
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        colours = getattr(getattr(geom, "visual", None), "vertex_colors", None)
        if colours is None:
            continue
        colours = np.asarray(colours)
        # trimesh COLLAPSES a uniform per-vertex array to a single RGBA, the
        # same shape quirk `pointclouds._vertex_rgb` already handles. Expand it
        # here so the caller receives one shape and only one.
        if colours.ndim == 1 and colours.size == 4:
            colours = np.tile(colours, (len(geom.vertices), 1))
        if colours.ndim != 2 or colours.shape[0] != len(geom.vertices):
            continue
        out[name] = colours.astype(np.uint8, copy=False)
    return out


def load_scene(path: Path, *, correct_frame: bool = True):
    """The GLB as a trimesh Scene, in the ULIP-2 frame.

    ``process=False`` is preserved from the original call sites: processing
    merges vertices, which silently changes the vertex count a COLOR_0 array is
    indexed by.

    ``correct_frame=False`` exists for the differential tests that need to see
    the uncorrected geometry. Nothing in the pipeline may pass it.
    """
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    if correct_frame:
        # Applied to the SCENE, so every node transform is composed with it
        # once. Applying it per geometry would miss the graph transforms that
        # 65.8% of these assets rely on.
        scene.apply_transform(FRAME_CORRECTION)
    return scene


def demo() -> None:
    """Self-check. `python -m metafind.data.meshload`

    Both assertions carry an expected value from OUTSIDE this module: the frame
    correction's algebra is checked against the property FIND-6 measured, and the
    COLOR_0 recovery is checked against the raw glTF accessor rather than
    against what this module returns.
    """
    assert np.isclose(np.linalg.det(FRAME_CORRECTION[:3, :3]), 1.0), (
        "the frame correction must be a rotation (det +1), never a reflection"
    )
    point = np.array([2.0, 3.0, 5.0, 1.0])
    assert np.allclose(FRAME_CORRECTION @ point, [-2.0, 3.0, -5.0, 1.0])

    # The axis-aligned bounding box must be invariant, or n05's proportions
    # anchor moves and every dimension in the corpus shifts with it.
    box = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0], [0.5, 0.0, 2.0]])
    turned = box @ FRAME_CORRECTION[:3, :3].T
    extents = lambda p: p.max(axis=0) - p.min(axis=0)  # noqa: E731
    assert np.allclose(extents(box), extents(turned)), (
        "a 180 degree yaw must preserve the axis-aligned extents"
    )

    from metafind import paths

    # `b3f86ea0...` is the worked example in the module docstring: its only
    # colour is COLOR_0, and the raw accessor holds [0.396, 0.392, 0.125].
    uid = "b3f86ea0972c4358a764225be1ef069f"
    found = list(paths.OBJAVERSE_GLB.rglob(f"{uid}.glb"))
    if not found:
        print("demo: corpus not present, algebra checks only -- OK")
        return
    glb = found[0]
    assert has_color0(glb), f"{uid} declares COLOR_0 in its glTF JSON"
    colours = color0_by_geometry(glb)
    assert colours, "COLOR_0 was declared and must therefore be recoverable"
    first = next(iter(colours.values()))
    expected = np.array([101, 100, 32])  # 0.396/0.392/0.125 * 255, from the accessor
    assert np.allclose(first[0, :3], expected, atol=1), (
        f"recovered {first[0, :3]}, accessor says {expected}"
    )
    assert not (first[:, :3] > 250).all(), "this asset must not come back white"
    print(f"demo: frame correction OK; COLOR_0 recovered as {first[0, :3]} -- OK")


if __name__ == "__main__":
    demo()
