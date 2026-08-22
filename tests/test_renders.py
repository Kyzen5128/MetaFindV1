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


def test_primary_layout_is_the_ulip2_style_orbit():
    """[U-03] The primary must be the layout with upstream provenance.

    MetaFind says only "11 orthogonal viewpoints". It states it builds on
    ULIP-2, whose Objaverse pipeline is "12 images per shape, spaced equally by
    30 degrees" -- a single-axis orbit. Fibonacci was the earlier primary and
    is now the variant; this pins which one a default render uses, because the
    two are measurably different (inter-view similarity 0.507 vs 0.442 over 100
    assets, 53 of them differing by more than 0.05).

    [CORRECTED 2026-08-22] This test used to assert that every direction shares
    a **z** component, and its own comment said so. That is the renderer-v2 bug
    written into the test: the meshes are Y-up, so an orbit about +Z tumbles the
    asset end-over-end, and this assertion held while it did. The intent -- one
    orbit, one elevation -- was always right; the hardcoded index was not.

    Written against ``UP_AXIS`` now, so the assertion follows the axis rather
    than restating a coordinate someone typed once.
    """
    from metafind.data.renders import (LAYOUTS, ORBIT_ELEVATION_DEG, UP_AXIS,
                                       azimuth_orbit_directions)

    assert CAMERA_LAYOUT == "ulip2_azimuth_orbit_11"
    assert LAYOUTS[CAMERA_LAYOUT] is azimuth_orbit_directions

    d = azimuth_orbit_directions(N_VIEWS)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    up = np.asarray(UP_AXIS, dtype=float)
    up = up / np.linalg.norm(up)
    # One orbit at one elevation: every direction makes the same angle with UP.
    lift = d @ up
    assert np.allclose(lift, lift[0]), "the orbit is not at a single elevation"
    assert abs(np.degrees(np.arcsin(lift[0])) - ORBIT_ELEVATION_DEG) < 1e-6
    # Equal azimuth steps of 360/11, measured in the plane PERPENDICULAR to up.
    e0 = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e0 = np.cross(up, e0)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(up, e0)
    az = np.degrees(np.arctan2(d @ e1, d @ e0)) % 360.0
    steps = np.diff(np.sort(az))
    assert np.allclose(steps, 360.0 / N_VIEWS, atol=1e-6)


def test_a_tall_asset_renders_tall(tmp_path):
    """The check that would have caught the v2 up-axis defect, and did not exist.

    EXPECTED-TRUTH SOURCE: the mesh's own geometry. A box three times longer on
    the up axis than on either horizontal axis must appear taller than it is
    wide from every viewpoint on a horizontal orbit -- that is what "orbit about
    the up axis" MEANS, and it needs no renderer, no paper and no upstream
    artifact to be true.

    Under renderer v2 the same box rendered WIDER than tall in most views: a
    real 7.2x-tall lamppost measured 0.14 image height/width, which is 1/7.2.
    """
    import trimesh as _tm
    from metafind.data.renders import UP_AXIS, render_views

    up = np.asarray(UP_AXIS, dtype=float)
    extents = np.where(up > 0.5, 3.0, 1.0)
    scene = _tm.Scene()
    scene.add_geometry(_tm.creation.box(extents=extents), node_name="tall")
    path = tmp_path / "tall.glb"
    path.write_bytes(scene.export(file_type="glb"))

    images, _ = render_views(path, n_views=N_VIEWS)
    ratios = []
    for img in images:
        # Foreground is whatever differs from the background CORNER, so this
        # does not care whether the background is black or white. The previous
        # black/white assumption is exactly what broke the projection test.
        bg = img[0, 0].astype(int)
        mask = np.abs(img.astype(int) - bg).max(axis=-1) > 12
        ys, xs = np.nonzero(mask)
        assert len(ys) > 50, "the tall box rendered as nothing"
        ratios.append((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))

    ratios = np.array(ratios)
    assert (ratios > 1.5).all(), (
        f"a 3:1 upright box rendered at height/width {np.round(ratios, 2).tolist()}; "
        "anything at or below 1 means the camera is orbiting the wrong axis"
    )


def test_both_layouts_stay_executable():
    """[U-03] A variant nobody can run is not a variant."""
    from metafind.data.renders import LAYOUTS

    for name, fn in LAYOUTS.items():
        d = fn(N_VIEWS)
        assert d.shape == (N_VIEWS, 3), name
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0), name


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
        # [CORRECTED 2026-08-22] Was `img.min(axis=-1) < 250`, i.e. "foreground
        # is anything darker than white". Renderer v3 matched ULIP's BLACK
        # background, and that predicate then counted every pixel in the frame:
        # both projections returned 50,176 = 224 x 224 and the test failed
        # claiming the projection setting was not reaching the camera.
        #
        # Foreground is now whatever differs from the background CORNER, which
        # holds under either background and cannot be broken by the next change
        # to it.
        bg = img[0, 0].astype(int)
        return int((np.abs(img.astype(int) - bg).max(axis=-1) > 12).sum())

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


def test_a_stale_sidecar_is_not_complete(tmp_path):
    """[ADDED 2026-08-22] A render from an older renderer must not count as done.

    EXPECTED-TRUTH SOURCE: the question a resumable run actually asks. It is not
    "is there an intact render here" -- it is "is there an intact render that
    THIS code produced". Those differ the moment the renderer changes, and the
    difference is silent: every stale sidecar stays internally consistent.

    Found by the Reviewer before the 3.3-hour regeneration. Without the version
    gate, all 45,955 v2 sidecars classified as complete, so a bare re-run would
    have skipped the entire corpus and exited reporting success.
    """
    import json

    from metafind.data.renders import RENDERER_VERSION, is_complete, process_one

    asset = _asset(tmp_path)
    out = tmp_path / "out"
    process_one("deadbeef", asset, out)
    assert is_complete(out, "deadbeef"), "a freshly written render must be complete"

    sc = out / "deadbeef.json"
    rec = json.loads(sc.read_text())
    assert rec["renderer_version"] == RENDERER_VERSION
    rec["renderer_version"] = RENDERER_VERSION - 1
    sc.write_text(json.dumps(rec))
    assert not is_complete(out, "deadbeef"), (
        "a sidecar from an older renderer was accepted as complete; a re-run "
        "would skip it and report success"
    )

    # A NEWER sidecar is not this renderer's output either. Accepting it would
    # let a downgraded run inherit artifacts it cannot reproduce.
    rec["renderer_version"] = RENDERER_VERSION + 1
    sc.write_text(json.dumps(rec))
    assert not is_complete(out, "deadbeef")


def test_a_sheared_node_transform_renders_where_trimesh_puts_it(tmp_path):
    """The geometry pyrender draws must be the geometry trimesh placed.

    `pyrender.Scene.from_trimesh_scene` stores each node's pose as
    translate . rotate(quaternion) . scale(vector), which represents
    `M = R . diag(s)` exactly -- non-uniform scale included -- but has no form
    for a sheared `M = R1 . diag(s) . R2`. Given one it silently substitutes a
    different matrix instead of raising, so the asset renders somewhere else.

    Expected truth is `trimesh.transform_points` with the scene graph's own
    transform: the library's placement, not the renderer's, and the same
    operation `n03` uses to sample point clouds. That is what makes this a test
    of agreement between the two nodes rather than of the renderer against
    itself.

    Measured before `normalised_scene` baked transforms into vertices, on this
    synthetic scene: bounds differed by 0.294 on a unit-ish object. On the real
    corpus, asset 2f0ef6ad926b474189b6ef489d11954c has a node with column norms
    0.014 / 0.009 / 0.457, whose translation came through exactly while its 3x3
    did not -- placing that geometry at y = -1.148 where the normalisation had
    put it at -0.939, outside `ORTHO_HALF_WIDTH` 1.10 and clipped in 7 of 11
    views.
    """
    pyrender = pytest.importorskip("pyrender")
    from metafind.data.renders import ORTHO_HALF_WIDTH, normalised_scene

    r1 = trimesh.transformations.rotation_matrix(0.7, [1.0, 2.0, 0.5])
    r2 = trimesh.transformations.rotation_matrix(1.1, [0.0, 1.0, 1.0])
    sheared = r1 @ np.diag([0.02, 0.9, 0.02, 1.0]) @ r2

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
                       node_name="sheared", transform=sheared)
    scene.add_geometry(trimesh.creation.box(extents=(0.3, 0.3, 0.3)), node_name="plain")
    path = tmp_path / "sheared.glb"
    path.write_bytes(scene.export(file_type="glb"))

    normalised, _ = normalised_scene(path)
    expected = np.concatenate([
        trimesh.transform_points(normalised.geometry[name].vertices, matrix)
        for node in normalised.graph.nodes_geometry
        for matrix, name in [normalised.graph[node]]
    ])

    rendered = pyrender.Scene.from_trimesh_scene(normalised)
    actual = np.concatenate([
        (rendered.get_pose(node)[:3, :3] @ prim.positions.T).T + rendered.get_pose(node)[:3, 3]
        for node in rendered.mesh_nodes for prim in node.mesh.primitives
    ])

    assert np.allclose(np.sort(expected.min(axis=0)), np.sort(actual.min(axis=0)), atol=1e-4)
    assert np.allclose(np.sort(expected.max(axis=0)), np.sort(actual.max(axis=0)), atol=1e-4)
    # Records the property the bounds assertions above protect: normalisation
    # bounds the asset by 1.0 and the frame is wider, but only if what pyrender
    # holds IS the normalised geometry. Not the detector -- injected against the
    # pre-fix code this synthetic scene still measures 0.6137, well inside the
    # frame, while the bounds assertions fail. Clipping needs a mis-placement
    # that happens to point outward, which is why the corpus rate (10/1,500 at
    # the frame edge) is far below the mis-placement rate (7/600 assets carry a
    # transform pyrender cannot represent).
    assert np.linalg.norm(actual, axis=1).max() <= 1.0 + 1e-6 < ORTHO_HALF_WIDTH


def test_a_worker_refuses_a_module_that_changed_mid_run():
    """A run's behaviour must not change without the artifacts saying so.

    Expected truth is the file's own bytes, read at call time -- not a constant
    recorded here, which would only test that two literals match.

    The mismatching call IS the negative injection: a fingerprint that disagrees
    with the file on disk is exactly what a mid-run edit produces, and on
    2026-08-22 it produced ~1,700 assets rendered with a geometry fix and stamped
    with the version that predated it.
    """
    from metafind.data import renders

    fingerprint = renders.implementation_fingerprint()
    assert set(fingerprint) == {"renders.py", "meshload.py"}, fingerprint

    renders._FINGERPRINT_VERIFIED = False
    renders.verify_fingerprint(fingerprint)  # matches -> returns

    renders._FINGERPRINT_VERIFIED = False
    with pytest.raises(RuntimeError, match="changed while the run was in progress"):
        renders.verify_fingerprint({**fingerprint, "renders.py": "0" * 64})

    renders._FINGERPRINT_VERIFIED = False
    renders.verify_fingerprint(None)  # no fingerprint -> no opinion


def test_a_failed_asset_stops_reading_as_complete(tmp_path):
    """A stale sidecar left by a failed re-render must not count as the corpus.

    Resume rests on "an asset is complete only once its sidecar lands", which is
    sound when the alternative is no sidecar. After a `RENDERER_VERSION` bump it
    is not: the asset is selected *because* its sidecar is stale, and if the
    re-render then fails the stale one is still there. Measured on the v4
    re-render before this existed -- 23 assets kept `renderer_version 3` records
    and `rebuild_index` counted every one of them as complete.
    """
    import json
    from pathlib import Path

    from metafind.data.renders import (RENDERER_VERSION, is_complete, rebuild_index,
                                       retire_stale_sidecar, sidecar_path)

    out = tmp_path / "out"
    (out / "u").mkdir(parents=True)
    views = []
    for i in range(11):
        p = out / "u" / f"view_{i:02d}.png"
        p.write_bytes(f"not a real png {i}".encode())
        views.append(str(p))
    sidecar_path(out, "u").write_text(json.dumps({
        "uid": "u", "view_paths": views,
        "view_bytes": [len(Path(v).read_bytes()) for v in views],
        "renderer_version": RENDERER_VERSION - 1,
    }))

    assert not is_complete(out, "u"), "a stale version must not read as complete"
    assert rebuild_index(tmp_path / "idx.jsonl", out) == 1, (
        "before retiring, the index counts the stale record -- this is the defect"
    )

    assert retire_stale_sidecar(out, "u") is True
    assert not sidecar_path(out, "u").exists()
    assert (out / "u.json.stale").exists(), "renamed, not deleted -- it is evidence"
    assert rebuild_index(tmp_path / "idx.jsonl", out) == 0
    assert not is_complete(out, "u")

    assert retire_stale_sidecar(out, "never-existed") is False
