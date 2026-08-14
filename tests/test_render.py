"""L1 tests for the 11-view renderer (sec. 2.3).

Runs on synthetic trimesh primitives so it needs no Objaverse download. What
matters is that the geometry of the view set and the framing are right; whether
a specific GLB loads is a data question, handled by quarantine at run time.
"""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("pyrender")

from metafind.data.render import (  # noqa: E402
    RenderConfig,
    camera_poses,
    look_at,
    normalize_mesh,
    render_asset,
    view_directions,
)


def box(extent=(1.0, 2.0, 3.0), colour=(200, 80, 80, 255)):
    m = trimesh.creation.box(extent=extent)
    m.visual.vertex_colors = colour
    return m


# --------------------------------------------------------------- view geometry


def test_eleven_views_by_default():
    """The paper's count, which is the one thing sec. 2.3 does pin down."""
    assert RenderConfig().n_views == 11
    assert view_directions(11).shape == (11, 3)


@pytest.mark.parametrize("layout", ["fibonacci", "axis_aligned"])
def test_directions_are_unit_vectors(layout: str):
    d = view_directions(11, layout)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("layout", ["fibonacci", "axis_aligned"])
def test_directions_are_distinct(layout: str):
    """Duplicated viewpoints would silently waste a slot of the 11."""
    d = view_directions(11, layout)
    gram = d @ d.T
    np.fill_diagonal(gram, -1.0)
    assert gram.max() < 0.999, "two viewpoints coincide"


def test_fibonacci_covers_the_sphere_better_than_axis_aligned():
    """Why fibonacci is the default: 11 is not a number the cube layout suits.

    Coverage is scored by the worst-case angle from any sphere direction to its
    nearest camera -- the direction a viewer would see least well.
    """
    probe = view_directions(512, "fibonacci")

    def worst_gap(dirs):
        return float(np.arccos(np.clip((probe @ dirs.T).max(axis=1), -1, 1)).max())

    fib = worst_gap(view_directions(11, "fibonacci"))
    axis = worst_gap(view_directions(11, "axis_aligned"))
    assert fib < axis, f"fibonacci {np.degrees(fib):.1f} deg vs axis {np.degrees(axis):.1f} deg"


def test_axis_aligned_starts_with_the_six_axes():
    d = view_directions(6, "axis_aligned")
    expected = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    assert np.allclose(d, expected)


def test_axis_aligned_runs_out_beyond_14():
    with pytest.raises(ValueError, match="at most"):
        view_directions(15, "axis_aligned")


def test_no_camera_lands_on_a_pole():
    """A camera exactly on a pole makes the up-vector degenerate."""
    d = view_directions(11, "fibonacci")
    assert np.abs(d[:, 2]).max() < 0.999


# --------------------------------------------------------------- poses


def test_poses_are_rigid_transforms():
    poses = camera_poses(RenderConfig())
    assert poses.shape == (11, 4, 4)
    for p in poses:
        r = p[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9), "rotation is not orthonormal"
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9), "pose includes a reflection"
        assert np.allclose(p[3], [0, 0, 0, 1])


def test_every_camera_looks_at_the_origin():
    """Cameras look down -Z, so the origin must lie along that ray."""
    for p in camera_poses(RenderConfig()):
        eye, forward = p[:3, 3], p[:3, 2]
        to_origin = -eye / np.linalg.norm(eye)
        assert np.allclose(forward, -to_origin, atol=1e-9)


def test_look_at_rejects_a_degenerate_camera():
    with pytest.raises(ValueError, match="coincide"):
        look_at(np.zeros(3), np.zeros(3))


def test_look_at_survives_the_pole():
    """The default up-vector is parallel to forward there; it must swap, not NaN."""
    p = look_at(np.array([0.0, 0.0, 3.0]), np.zeros(3))
    assert np.isfinite(p).all()
    assert np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-9)


# --------------------------------------------------------------- normalisation


def test_normalisation_centres_and_unit_scales():
    """Otherwise the tower learns modelling units instead of shape."""
    m = box()
    m.vertices += 1000.0
    m.vertices *= 50.0
    out = normalize_mesh(m)
    assert np.allclose(out.vertices.mean(axis=0), 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(out.vertices, axis=1).max(), 1.0, atol=1e-6)


def test_normalisation_is_scale_invariant():
    """A millimetre model and a metre model must render identically."""
    small, large = box(), box()
    large.vertices *= 1000.0
    assert np.allclose(normalize_mesh(small).vertices, normalize_mesh(large).vertices, atol=1e-6)


def test_normalisation_rejects_degenerate_meshes():
    m = box()
    m.vertices[:] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        normalize_mesh(m)


def test_normalisation_rejects_non_finite_meshes():
    m = box()
    m.vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        normalize_mesh(m)


# --------------------------------------------------------------- rendering


def test_render_shape_and_dtype():
    out = render_asset(box(), RenderConfig(resolution=64))
    assert out.shape == (11, 64, 64, 3)
    assert out.dtype == np.uint8


def test_every_view_contains_the_object():
    """The real failure mode: a mis-framed camera returns a blank white image."""
    out = render_asset(box(), RenderConfig(resolution=64))
    for i, img in enumerate(out):
        non_bg = int((img.reshape(-1, 3).max(axis=1) < 250).sum())
        assert non_bg > 50, f"view {i} is essentially empty ({non_bg} object pixels)"


def test_views_differ_from_one_another():
    """Eleven identical images would satisfy every shape assertion above."""
    out = render_asset(box(extent=(1.0, 2.0, 4.0)), RenderConfig(resolution=64))
    flat = out.reshape(11, -1).astype(np.int16)
    for i in range(11):
        for j in range(i + 1, 11):
            assert np.abs(flat[i] - flat[j]).mean() > 0.5, f"views {i} and {j} are identical"


def test_object_stays_inside_the_frame():
    """Margin exists so silhouettes are not clipped; verify nothing touches the edge."""
    out = render_asset(box(extent=(2.0, 2.0, 2.0)), RenderConfig(resolution=64))
    for i, img in enumerate(out):
        mask = img.reshape(64, 64, 3).max(axis=2) < 250
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        assert not border.any(), f"view {i} is clipped at the frame edge"


def test_rendering_is_deterministic():
    """Cached embeddings are only reproducible if their inputs are."""
    cfg = RenderConfig(resolution=64)
    assert np.array_equal(render_asset(box(), cfg), render_asset(box(), cfg))


def test_scale_invariance_end_to_end():
    """Same object at two modelling scales must produce identical pixels."""
    cfg = RenderConfig(resolution=64)
    big = box()
    big.vertices *= 137.0
    assert np.array_equal(render_asset(box(), cfg), render_asset(big, cfg))


def test_orthographic_projection_ignores_camera_distance():
    """Perspective would make apparent size depend on distance; orthographic must not.

    This is the assertion behind reading "orthogonal" as orthographic.
    """
    import pyrender

    cfg = RenderConfig(resolution=64)
    mesh = normalize_mesh(box())
    scene = pyrender.Scene(bg_color=(1, 1, 1, 1), ambient_light=(0.5,) * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.OrthographicCamera(xmag=1.15, ymag=1.15)
    node = scene.add(cam, pose=look_at(np.array([0.0, -3.0, 0.0]), np.zeros(3)))
    r = pyrender.OffscreenRenderer(64, 64)
    try:
        near, _ = r.render(scene)
        scene.set_pose(node, look_at(np.array([0.0, -30.0, 0.0]), np.zeros(3)))
        far, _ = r.render(scene)
    finally:
        r.delete()

    area_near = int((near.max(axis=2) < 250).sum())
    area_far = int((far.max(axis=2) < 250).sum())
    assert area_near > 0 and abs(area_near - area_far) <= 2, (
        f"apparent size changed with distance ({area_near} vs {area_far}); not orthographic"
    )


# --------------------------------------------------------------- poisoned cache


def test_zero_byte_index_is_not_treated_as_cached(tmp_path):
    """objaverse's own check is `if not os.path.exists(...)`, so a truncated
    download counts as a valid cache and the next run gets a gzip error instead
    of a retry. Existence is not correctness.
    """
    from metafind.data.render_assets import _object_paths_ok

    empty = tmp_path / "object-paths.json.gz"
    empty.write_bytes(b"")
    assert _object_paths_ok(empty) is False
    assert empty.exists(), "the file exists, which is exactly why existence is the wrong test"


def test_truncated_and_malformed_indexes_are_rejected(tmp_path):
    import gzip
    import json as _json

    from metafind.data.render_assets import _object_paths_ok

    short = tmp_path / "short.gz"
    short.write_bytes(gzip.compress(_json.dumps({"a": "b"}).encode()))
    assert _object_paths_ok(short) is False, "a valid but tiny index must be rejected"

    garbage = tmp_path / "garbage.gz"
    garbage.write_bytes(b"\x00" * 2_000_000)
    assert _object_paths_ok(garbage) is False, "non-gzip content must be rejected"

    missing = tmp_path / "nope.gz"
    assert _object_paths_ok(missing) is False


def test_valid_index_is_accepted(tmp_path):
    """Negative injections above prove nothing unless the positive case passes."""
    import gzip
    import json as _json

    from metafind.data.render_assets import _object_paths_ok

    # Real uids are 32 random hex chars, which barely compress. Sequential
    # "uid0, uid1, ..." keys gzip to well under the 1 MB floor and would make
    # this fixture unrepresentative of the file being guarded.
    import secrets

    entries = {}
    for _ in range(200_000):
        uid = secrets.token_hex(16)
        entries[uid] = f"glbs/{uid[:3]}-{uid[3:6]}/{uid}.glb"
    good = tmp_path / "good.gz"
    good.write_bytes(gzip.compress(_json.dumps(entries).encode()))

    assert good.stat().st_size > 1_000_000, f"fixture is only {good.stat().st_size} bytes"
    assert _object_paths_ok(good) is True
