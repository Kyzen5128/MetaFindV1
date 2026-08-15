"""L1 tests for n03_sample_pointclouds.

Every test here corresponds to a defect the smoke run actually produced. The
sampler looked correct three times and was wrong three times, each in a way
that raised nothing:

  * all 40 assets uniformly grey        colours read after concatenation
  * 14 of 40 quarantined                merged colour array shorter than faces
  * 27 of 60 grey                       flat PBR materials discarded
  * 40 of 40 quarantined                np.savez_compressed renamed the file

A point cloud that is the wrong shape fails loudly. A point cloud whose colour
channel is a constant does not -- it trains, it embeds, and it lowers every
Table 1 number with no error anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from metafind.data.pointclouds import (  # noqa: E402
    DEFAULT_GREY,
    N_POINTS,
    _allocate,
    _colourise,
    pc_norm,
    process_one,
    sample_mesh,
    uid_seed,
)


def _box(colour=None, texture=False):
    m = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    if texture:
        m.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(m.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=colour),
        )
    elif colour is not None:
        m.visual.vertex_colors = np.tile(np.asarray(colour, dtype=np.uint8), (len(m.vertices), 1))
    return m


# ------------------------------------------------------------------ pc_norm


def test_pc_norm_matches_ulips_definition():
    """[L1-PC-NORM] Centroid at the origin, largest radius exactly 1.

    Copied from ULIP dataset_3d.py:496-502 rather than reinvented: the frozen
    checkpoint was trained on clouds normalised that way, and a different
    normalisation is a different input distribution with an identical shape.
    """
    rng = np.random.default_rng(0)
    pc = rng.normal(size=(500, 3)) * 7.0 + 40.0
    out = pc_norm(pc)
    assert np.abs(out.mean(axis=0)).max() < 1e-12
    assert abs(np.sqrt((out**2).sum(axis=1)).max() - 1.0) < 1e-12


def test_pc_norm_touches_xyz_only():
    """rgb must not be normalised with the coordinates.

    ULIP normalises xyz and THEN concatenates rgb. Normalising all six columns
    together rescales colour by the object's physical extent -- same shape,
    different tensor, no error.
    """
    xyz = np.array([[0.0, 0, 0], [3, 0, 0], [0, 4, 0]])
    rgb = np.full((3, 3), 0.4)
    six = np.concatenate([pc_norm(xyz), rgb], axis=1)
    assert np.allclose(six[:, 3:], 0.4), "colour was altered by coordinate normalisation"


# ------------------------------------------------------------------ colour


def test_rgb_is_unit_scale_not_0_255():
    """[U-02] ULIP's own grey stand-in is 0.4, which fixes the scale.

    dataset_3d.py 292 and 297 substitute np.ones_like(pc) * 0.4 where a dataset
    has no colour. On a 0-255 scale that would be about 102. Getting this wrong
    is silent and moves every point-cloud embedding.
    """
    assert 0.0 < DEFAULT_GREY < 1.0
    xyz, rgb, _, _, _ = sample_mesh_from(_box(colour=(200, 100, 50, 255)))
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert np.allclose(rgb[0], np.array([200, 100, 50]) / 255.0, atol=1e-3)


def test_flat_pbr_material_is_a_colour_not_a_failure():
    """A texture-less PBR material still carries baseColorFactor.

    trimesh's to_color() returns a FOUR-row vertex_colors array for these -- one
    RGBA, not one per vertex -- so a naive length check rejects it. That check
    cost 45% of the first test batch its colour, replacing real beige with
    ULIP's grey.
    """
    m = _box(colour=(204, 200, 176, 255), texture=True)
    assert _colourise(m) == "flat"
    assert np.allclose(m.visual.vertex_colors[:, :3], [204, 200, 176])


def test_uncolourable_geometry_reports_itself():
    """The fallback must be visible, not just applied.

    A blanket `except Exception: return grey` is what made the first run produce
    40 uniformly grey assets and look successful. The source is recorded so a
    run that lost its colour channel is legible in the summary.
    """
    m = trimesh.creation.box()
    m.visual = trimesh.visual.TextureVisuals(
        uv=None, material=trimesh.visual.material.PBRMaterial()
    )
    assert _colourise(m) == "fallback_grey"
    assert np.allclose(m.visual.vertex_colors[:, :3], int(DEFAULT_GREY * 255))


# ------------------------------------------------------------ point budget


@pytest.mark.parametrize("areas", [[1.0], [1.0, 1.0], [0.001, 5.0, 0.7], [3, 3, 3, 3, 3, 3, 3]])
def test_allocation_sums_to_exactly_n_points(areas):
    """A 9,998-point cloud passes no shape check and fails the one that counts."""
    counts = _allocate(np.array(areas, dtype=float), N_POINTS)
    assert counts.sum() == N_POINTS
    assert (counts >= 0).all()


def test_allocation_is_area_weighted():
    counts = _allocate(np.array([9.0, 1.0]), 1000)
    assert counts[0] > counts[1] * 5


def test_allocation_survives_degenerate_areas():
    counts = _allocate(np.array([0.0, 0.0]), 100)
    assert counts.sum() == 100


# -------------------------------------------------------------- determinism


def test_seed_depends_on_uid_only():
    """A counter-based seed makes a resumed run a different dataset."""
    assert uid_seed("abc") == uid_seed("abc")
    assert uid_seed("abc") != uid_seed("abd")


def test_same_mesh_yields_the_same_cloud(tmp_path):
    m = _box(colour=(10, 200, 30, 255))
    p = tmp_path / "a.glb"
    m.export(p)
    a = sample_mesh(p, seed=uid_seed("u"), n_points=512)
    b = sample_mesh(p, seed=uid_seed("u"), n_points=512)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ------------------------------------------------------------------- output


def test_written_file_is_the_file_that_was_promised(tmp_path):
    """np.savez_compressed appends '.npz' to any name that lacks it.

    Passing `<uid>.part` therefore wrote `<uid>.part.npz`, and the atomic
    rename then failed on a file that never existed -- 40 of 40 assets
    quarantined with a FileNotFoundError naming the destination, not the cause.
    """
    m = _box(colour=(10, 200, 30, 255))
    glb = tmp_path / "a.glb"
    m.export(glb)
    out = tmp_path / "u.npz"
    rec = process_one("u", glb, out)

    assert out.exists(), "the destination was never created"
    assert not list(tmp_path.glob("*.part*")), "a temporary file survived"
    d = np.load(out)
    assert d["xyz"].shape == (N_POINTS, 3) and d["rgb"].shape == (N_POINTS, 3)
    assert rec["n_points"] == N_POINTS and rec["rgb_scale"] == "unit"


def test_sidecar_records_what_g2_will_check(tmp_path):
    """G2 reads these fields; producing the cloud without them defers the check."""
    m = _box(colour=(10, 200, 30, 255))
    glb = tmp_path / "a.glb"
    m.export(glb)
    rec = process_one("u", glb, tmp_path / "u.npz")
    for field in ("centroid_offset", "max_radius", "per_axis_variance",
                  "extents_m", "volume_m3", "colour_source", "coloured_point_fraction"):
        assert field in rec, field
    assert rec["centroid_offset"] < 1e-5
    assert abs(rec["max_radius"] - 1.0) < 1e-5
    # F13: the only place the pre-normalisation scale survives.
    assert np.allclose(sorted(rec["extents_m"]), [1.0, 2.0, 3.0], atol=1e-6)


def sample_mesh_from(mesh):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.glb"
        mesh.export(p)
        return sample_mesh(p, seed=0, n_points=256)
