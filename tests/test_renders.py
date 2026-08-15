"""L1 tests for n04_render_views.

The three checks validation_plan names for this node, plus the failure modes the
smoke runs actually produced. A render that is the wrong shape fails loudly; a
render that is blank, or identical across all 11 views, or silently scaled by
whatever units the asset was modelled in, does not.
"""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("pyrender")

from metafind.data.renders import (  # noqa: E402
    CAMERA_LAYOUT,
    N_VIEWS,
    PROJECTION,
    RESOLUTION,
    fibonacci_directions,
    look_at,
    normalised_scene,
    process_one,
    render_views,
)


def _asset(tmp_path, scale=1.0, name="a.glb"):
    """An asymmetric object, so different viewpoints genuinely differ."""
    box = trimesh.creation.box(extents=(1.0, 0.4, 0.2))
    cone = trimesh.creation.cone(radius=0.3, height=0.8)
    cone.apply_translation([0.4, 0, 0.3])
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="box")
    scene.add_geometry(cone, node_name="cone")
    scene.apply_transform(np.diag([scale, scale, scale, 1.0]))
    p = tmp_path / name
    p.write_bytes(scene.export(file_type="glb"))
    return p


# --------------------------------------------------------------- camera layout


def test_fibonacci_directions_are_unit_and_spread():
    d = fibonacci_directions(N_VIEWS)
    assert d.shape == (N_VIEWS, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # No two directions may coincide, or two of the eleven views are the same
    # picture and the asset is described by ten.
    pair_cos = d @ d.T
    np.fill_diagonal(pair_cos, -1.0)
    assert pair_cos.max() < 0.95, "two viewpoints are nearly identical"


def test_directions_are_deterministic():
    assert np.array_equal(fibonacci_directions(11), fibonacci_directions(11))


def test_look_at_survives_the_pole():
    """A camera directly above the object must not produce a degenerate basis.

    The up vector and the view direction are parallel there, so the cross
    product is zero and the pose comes out full of NaN -- which renders as a
    blank frame rather than an error.
    """
    pose = look_at(np.array([0.0, 0.0, 3.0]))
    assert np.isfinite(pose).all()
    assert np.allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)


# ------------------------------------------------------------ L1-RENDER-COUNT


def test_eleven_distinct_non_blank_views(tmp_path):
    """[L1-RENDER-COUNT] exactly 11 views, all non-empty, all distinct."""
    images, _ = render_views(_asset(tmp_path))
    assert len(images) == N_VIEWS
    for img in images:
        assert img.shape == (RESOLUTION, RESOLUTION, 3)
        assert img.std() > 1.0, "blank frame: a mis-aimed camera returns plain white"
    digests = {img.tobytes() for img in images}
    assert len(digests) == N_VIEWS, f"only {len(digests)} distinct views"


# ------------------------------------------ L1-RENDER-PROJECTION-CONSISTENT


def test_orthographic_size_does_not_change_with_distance(tmp_path):
    """[L1-RENDER-PROJECTION-CONSISTENT] Under orthographic, an object's
    apparent size must not depend on how far away the camera is.

    Goes through render_views with the real camera-construction code. An
    earlier version of this test built its own OrthographicCamera, so swapping
    the production camera for a perspective one left it green -- the test
    asserted a property of a camera it had constructed itself.
    """
    asset = _asset(tmp_path)
    ortho, _ = render_views(asset, n_views=3, projection="orthographic")
    persp, _ = render_views(asset, n_views=3, projection="perspective")

    def silhouette(img):
        return int((img.min(axis=-1) < 250).sum())

    # Same scene, same camera positions, different projection: the perspective
    # camera at distance 3 with a 45-degree fov frames far less of the unit
    # sphere than an orthographic camera spanning it exactly.
    o = np.array([silhouette(i) for i in ortho])
    p = np.array([silhouette(i) for i in persp])
    assert (o > 0).all() and (p > 0).all(), "one of the projections rendered nothing"
    assert np.abs(o - p).max() > 20, (
        f"orthographic {o.tolist()} and perspective {p.tolist()} produced the same "
        "silhouettes, so the projection setting is not reaching the camera"
    )


# ------------------------------------------------- L1-RENDER-SCALE-INVARIANT


def test_millimetre_and_metre_copies_render_identically(tmp_path):
    """[L1-RENDER-SCALE-INVARIANT] The image tower must not learn modelling units.

    Objaverse authors choose their own scale, so without the unit-sphere fit
    the same object uploaded in millimetres and in metres becomes two different
    training examples.
    """
    small, _ = render_views(_asset(tmp_path, scale=0.001, name="mm.glb"))
    large, _ = render_views(_asset(tmp_path, scale=1000.0, name="km.glb"))
    # Without this, "identical" is also satisfied by two blank frames -- which
    # is exactly what removing the unit-sphere fit produces, so the test passed
    # under an injection that broke the thing it was checking.
    for img in (*small, *large):
        assert img.std() > 1.0, "blank frame: the object is not in shot at either scale"
    for a, b in zip(small, large):
        # Not byte-equality: the fit divides by a float that differs in its last
        # bits between the two scales.
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 1.0


# ------------------------------------------------------- L1-RENDER-EXTENTS


def test_pre_normalisation_extents_are_recorded(tmp_path):
    """[L1-RENDER-EXTENTS / F13] Normalisation destroys scale, so record it first.

    Without this the annotator's "size dimensions" (paper 2.3) is a category
    prior with nothing to audit it against.
    """
    _, extents = render_views(_asset(tmp_path, scale=2.0))
    assert np.isfinite(extents).all() and (extents > 0).all()
    # The box is 1.0 x 0.4 x 0.2 at scale 2, plus a cone that extends it.
    assert extents.max() > 1.9


def test_record_carries_the_channel_contract(tmp_path):
    """The renders channel type names these; producing images without them
    defers G2/G3's checks to something that cannot perform them."""
    rec = process_one("u", _asset(tmp_path), tmp_path / "out")
    for field in ("view_paths", "view_sha256", "raw_bbox_extents", "projection",
                  "camera_layout", "resolution", "renderer_version", "blank_views"):
        assert field in rec, field
    assert len(rec["view_paths"]) == N_VIEWS
    assert len(set(rec["view_sha256"])) == N_VIEWS
    assert rec["projection"] == PROJECTION and rec["camera_layout"] == CAMERA_LAYOUT
    assert rec["resolution"] == RESOLUTION


# ------------------------------------------------------------- robustness


def test_non_mesh_geometry_does_not_lose_the_asset(tmp_path):
    """A Path3D curve alongside the mesh must cost the curve, not the object.

    Measured: 4 of 200 Objaverse assets died outright because pyrender rejects
    the whole scene when one geometry is not a Trimesh.
    """
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="box")
    scene.add_geometry(
        trimesh.load_path(np.array([[[0, 0, 0], [1, 1, 1]]], dtype=float)), node_name="curve"
    )
    p = tmp_path / "mixed.glb"
    p.write_bytes(scene.export(file_type="glb"))

    images, extents = render_views(p)
    assert len(images) == N_VIEWS
    assert np.isfinite(extents).all()
