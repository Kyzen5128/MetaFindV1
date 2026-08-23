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
Image = pytest.importorskip("PIL.Image")

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


def test_rgb_is_written_at_unit_scale():
    """[U-02] What we WRITE is [0, 1]; whether that matches ULIP is measured elsewhere.

    ULIP substitutes np.ones_like(pc) * 0.4 for uncoloured datasets, which on a
    0-255 scale would be ~102 -- strong evidence about its colour convention,
    but those lines are the ModelNet path. Objaverse_Lvis_Colored concatenates
    the released rgb with no division, so it inherits whatever the .npy holds.
    This test pins OUR output; the comparison against an official cloud is
    U-02's job and is not asserted here.
    """
    assert 0.0 < DEFAULT_GREY < 1.0
    xyz, rgb, _, _, _, _ = sample_mesh_from(_box(colour=(200, 100, 50, 255)))
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
    assert _colourise(m)[0] == "flat"
    assert np.allclose(m.visual.vertex_colors[:, :3], [204, 200, 176])


def test_material_without_texture_or_factor_is_gltf_white():
    """glTF 2.0 defines baseColorFactor as [1,1,1,1] when absent.

    trimesh reports 102/255 = 0.4 as ITS default for that case, numerically
    identical to DEFAULT_GREY, so treating the material as unreadable looked
    like a legitimate fallback. Measured against ULIP's released cloud for
    1dc0fe17c77e: theirs is 1.000, ours was 0.400. After this fix all six
    comparable assets match to within 0.007, mean absolute difference 0.0021.
    """
    m = trimesh.creation.box()
    m.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(m.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(),
    )
    assert _colourise(m)[0] == "gltf_default"
    assert np.allclose(m.visual.vertex_colors[:, :3], 255)


def test_colourise_always_yields_a_usable_array_and_a_named_source():
    """The invariant, rather than a case that cannot occur.

    `fallback_grey` is the terminal guard and is now UNREACHABLE through any
    observed path: trimesh substitutes a SimpleMaterial where a glTF has none,
    so the gltf_default branch fires first, and ColorVisuals regenerates
    vertex_colors to match the mesh so the length checks cannot fail. It stayed
    in because _colourise must return something, but it was not reached on any
    of the 60 smoke assets and must not be described as a live fallback.

    What matters downstream is what this test asserts: every geometry comes out
    with one colour per vertex and a source that is named.
    """
    cases = [
        _box(colour=(10, 20, 30, 255)),                      # vertex colours
        _box(colour=(204, 200, 176, 255), texture=True),     # flat PBR factor
        trimesh.creation.box(),                              # bare default
    ]
    for m in cases:
        src, modulated = _colourise(m)
        assert src in ("texture", "flat", "gltf_default", "vertex", "face", "fallback_grey")
        # No COLOR_0 was passed, so nothing may claim to have been modulated.
        assert modulated is False
        vc = m.visual.vertex_colors
        assert len(vc) == len(m.vertices), f"{src}: {len(vc)} colours for {len(m.vertices)} vertices"
        assert vc[:, :3].max() <= 255 and vc[:, :3].min() >= 0


# ------------------------------------------------------------ point budget


@pytest.mark.parametrize("areas", [[1.0], [1.0, 1.0], [0.001, 5.0, 0.7], [3, 3, 3, 3, 3, 3, 3]])
def test_allocation_sums_to_exactly_n_points(areas):
    """A 9,998-point cloud passes no shape check and fails the one that counts."""
    counts = _allocate(np.array(areas, dtype=float), N_POINTS)
    assert counts.sum() == N_POINTS
    assert (counts >= 0).all()


def test_vertex_colours_of_any_rank_are_normalised():
    """The fix for an 8.1% quarantine rate, tested where it actually lives.

    Real assets produced geometries whose ``visual.vertex_colors`` came back
    1-D -- shape (4,) for a four-vertex part -- and indexing that by a face
    array raises "too many indices for array", so the asset was quarantined as
    though its geometry were broken. 65 of 800, every one the same message; the
    60-asset smoke run had shown zero.

    The exact upstream condition inside trimesh was NOT reproduced
    synthetically: every construction tried here returns (n, 4). So this tests
    the defence rather than the trigger, which is the honest thing a test can
    assert -- and the defence is what took the sampled rate from 8.1% to 0.0%
    over 1,500 assets.
    """
    from metafind.data.pointclouds import _vertex_rgb

    class _Fake:
        def __init__(self, verts, colours):
            self.vertices = np.zeros((verts, 3))
            self.visual = type("V", (), {"vertex_colors": colours})()

    # 1-D: one RGBA for the whole geometry, which is what was observed.
    rgb = _vertex_rgb(_Fake(4, np.array([40, 60, 80, 255], dtype=np.uint8)))
    assert rgb.shape == (4, 3) and (rgb == [40, 60, 80]).all()

    # 2-D but the wrong length: not per-vertex, so it cannot be indexed by faces.
    rgb = _vertex_rgb(_Fake(6, np.tile([10, 20, 30, 255], (2, 1)).astype(np.uint8)))
    assert rgb.shape == (6, 3)

    # The ordinary case must pass through untouched.
    per_vertex = np.tile([1, 2, 3, 255], (5, 1)).astype(np.uint8)
    assert (_vertex_rgb(_Fake(5, per_vertex)) == [1, 2, 3]).all()


def test_vertex_rgb_normalises_whatever_shape_it_is_given():
    """The defence behind the fix: never index a colour array of unknown rank."""
    from metafind.data.pointclouds import _vertex_rgb

    m = trimesh.creation.box()
    m.visual = trimesh.visual.ColorVisuals(mesh=m)
    m.visual.vertex_colors = np.tile([40, 60, 80, 255], (len(m.vertices), 1))
    assert _vertex_rgb(m).shape == (len(m.vertices), 3)


def test_allocation_is_area_weighted():
    counts = _allocate(np.array([9.0, 1.0]), 1000)
    assert counts[0] > counts[1] * 5


def test_allocation_survives_degenerate_areas():
    counts = _allocate(np.array([0.0, 0.0]), 100)
    assert counts.sum() == 100


# -------------------------------------------------------------- determinism


def test_scene_graph_transform_is_applied(tmp_path):
    """[MEASURED] 65.8% of sampled Objaverse GLBs carry a non-identity transform.

    Sampling raw geometry drops it, so multi-part objects assemble collapsed on
    top of each other. The cloud still has 10,000 points, still normalises to a
    unit sphere, and still passes every G2 check -- it is just the wrong shape.
    On 40 re-sampled assets, 16 changed extent by more than 1%, up to 77x.
    """
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    scene = trimesh.Scene()
    scene.add_geometry(a, node_name="a")
    scene.add_geometry(b, node_name="b", transform=trimesh.transformations.translation_matrix([10, 0, 0]))
    p = tmp_path / "two.glb"
    p.write_bytes(scene.export(file_type="glb"))

    _, _, extents, _, _, _ = sample_mesh(p, seed=0, n_points=256)
    assert max(extents) > 10.0, (
        f"extent {max(extents):.2f} -- the 10-unit offset was dropped, so the "
        "two boxes were sampled on top of each other"
    )


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
                  "raw_bbox_extents", "raw_bbox_volume", "colour_source",
                  "coloured_point_fraction", "sha256"):
        assert field in rec, field
    assert rec["centroid_offset"] < 1e-5
    assert abs(rec["max_radius"] - 1.0) < 1e-5
    # F13: the only place the pre-normalisation scale survives.
    assert np.allclose(sorted(rec["raw_bbox_extents"]), [1.0, 2.0, 3.0], atol=1e-6)


def sample_mesh_from(mesh):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.glb"
        mesh.export(p)
        return sample_mesh(p, seed=0, n_points=256)


def test_a_stale_sidecar_is_not_complete(tmp_path):
    """[ADDED 2026-08-22] A cloud from an older sampler must not count as done.

    EXPECTED-TRUTH SOURCE: the question a resumable run asks -- "did THIS code
    produce this", not "is this file intact". n03 was worse than n04 here: it
    did not even bump its version, so the 2026-08-22 frame and COLOR_0
    corrections left all 46,052 stale clouds passing both the digest check and
    the version check. Found by the Reviewer before the regeneration.
    """
    import json

    from metafind.data.pointclouds import SAMPLER_VERSION, is_complete, process_one

    scene = trimesh.Scene()
    scene.add_geometry(_box(colour=[200, 40, 40, 255]), node_name="box")
    asset = tmp_path / "a.glb"
    asset.write_bytes(scene.export(file_type="glb"))
    npz = tmp_path / "out" / "deadbeef.npz"
    process_one("deadbeef", asset, npz)
    assert is_complete(npz), "a freshly written cloud must be complete"

    sc = npz.with_suffix(".json")
    rec = json.loads(sc.read_text())
    assert rec["sampler_version"] == SAMPLER_VERSION
    assert rec["frame_correction"], "the frame correction id must reach the sidecar"
    rec["sampler_version"] = SAMPLER_VERSION - 1
    sc.write_text(json.dumps(rec))
    assert not is_complete(npz), (
        "a sidecar from an older sampler was accepted as complete; a re-run "
        "would skip all 46,052 clouds and report success"
    )


def test_color0_multiplies_the_base_colour_and_leaves_others_untouched():
    """[P3, 2026-08-22] glTF base-colour semantics: COLOR_0 is a MULTIPLIER.

    EXPECTED-TRUTH SOURCE: arithmetic, plus the upstream measurement that chose
    P3. A half-brightness COLOR_0 over a known flat factor must yield half that
    factor -- not the factor (P1), and not the COLOR_0 (P2). The rule itself was
    decided by scoring all three against ULIP's own clouds through the frozen
    ULIP-2 point encoder over 130 assets where they differ: P1 best on 27,
    P2 on 54, P3 on 49. P1 is decisively worst; P2 and P3 sit inside one sigma,
    and the glTF specification breaks that tie.

    The control half matters as much as the modulated half: an asset with no
    COLOR_0 must come back BYTE-IDENTICAL, which is what stops P3 from quietly
    reshading the ~44,800 assets it has no business touching. Verified on real
    assets at full 10,000 points -- max absolute RGB difference 0.0000 across
    nine controls.
    """
    from metafind.data.pointclouds import _colourise

    factor = (200, 100, 50, 255)

    m = _box(colour=factor, texture=True)
    src, modulated = _colourise(m, color0=None)
    assert (src, modulated) == ("flat", False)
    base = np.asarray(m.visual.vertex_colors)[:, :3].copy()

    m2 = _box(colour=factor, texture=True)
    half = np.tile(np.array([128, 128, 128, 255], dtype=np.uint8),
                   (len(m2.vertices), 1))
    src2, modulated2 = _colourise(m2, color0=half)
    assert (src2, modulated2) == ("flat", True), (
        "the base colour must still be named `flat`; COLOR_0 modulates a source, "
        "it does not become one"
    )
    got = np.asarray(m2.visual.vertex_colors)[:, :3]

    expected = np.rint(base.astype(float) * (128.0 / 255.0)).astype(int)
    assert np.abs(got.astype(int) - expected).max() <= 1, (
        f"expected {expected[0].tolist()} from {base[0].tolist()} x 128/255, "
        f"got {got[0].tolist()}"
    )
    # P1 would have returned the factor unchanged; P2 would have returned grey.
    assert not np.array_equal(got, base), "COLOR_0 was ignored (P1)"
    assert not np.array_equal(got, half[:, :3]), "COLOR_0 replaced the base (P2)"


def test_a_white_base_makes_color0_pass_through_unchanged():
    """[P3] `gltf_default` x COLOR_0 == COLOR_0, because the base is white.

    EXPECTED-TRUTH SOURCE: x * 1 = x. This is why the ~146 assets whose base is
    glTF-default white are unaffected by the P1 -> P3 change, and it is the
    property that makes them a usable control group for the whole switch.
    """
    from metafind.data.pointclouds import _colourise

    m = trimesh.creation.box()
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial())
    colours = np.tile(np.array([10, 200, 90, 255], dtype=np.uint8),
                      (len(m.vertices), 1))
    src, modulated = _colourise(m, color0=colours)
    assert (src, modulated) == ("gltf_default", True)
    got = np.asarray(m.visual.vertex_colors)[:, :3]
    assert np.abs(got.astype(int) - colours[:, :3].astype(int)).max() <= 1


def test_texture_bases_are_not_modulated_by_color0():
    """[R-12, 2026-08-22] The `texture` carve-out, which is a DEVIATION.

    EXPECTED-TRUTH SOURCE: the measurement that narrowed the scope, not a
    principle. glTF 2.0 puts textured assets IN scope -- COLOR_0 multiplies the
    base colour and the base colour includes the texture -- so this test pins a
    deliberate departure from the specification and must fail loudly if anyone
    "fixes" it back.

    Measured over ALL 37 texture assets carrying COLOR_0 that ULIP also
    publishes, the whole population: modulating darkens 37 of 37 by 0.2076 mean
    on a 0-1 scale, and cosine against ULIP's own clouds through the frozen
    encoder moves 0.9005 -> 0.8980. `R-11` decides it -- ULIP-2 is the reference
    architecture, so agreement is the default and only deliberate divergence is
    registered.

    `flat` and `gltf_default` stay modulated; this test also pins that, because
    a narrowing applied one branch too widely would be silent.
    """
    from metafind.data.pointclouds import _colourise, _sample_part

    half = lambda m: np.tile(np.array([128, 128, 128, 255], dtype=np.uint8),  # noqa: E731
                             (len(m.vertices), 1))

    # A real per-vertex texture sample: COLOR_0 must be IGNORED.
    m = _box()
    m.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(m.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (2, 2), (180, 90, 60))))
    src, modulated = _colourise(m, color0=half(m))
    if src == "texture":
        assert modulated is False, (
            "COLOR_0 was applied to a texture base; R-12 withdrew it from that "
            "class after measuring 37/37 assets darker"
        )
        # [SAMPLER_VERSION 8] A samplable texture keeps its UVs -- `_colourise`
        # deliberately does NOT convert it to ColorVisuals, so there are no
        # `vertex_colors` to read here any more. Compare what the asset actually
        # ships instead: the sampled point colours. That is the stronger check,
        # because it is the array that reaches the encoder.
        m2 = _box()
        m2.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(m2.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (180, 90, 60))))
        _colourise(m2, color0=None)
        _, with_c0 = _sample_part(m, 64, seed=0)
        _, without = _sample_part(m2, 64, seed=0)
        assert np.array_equal(with_c0, without), (
            "a textured asset must be byte-identical with and without COLOR_0"
        )

    # The narrowing must not reach `flat`, which stays modulated.
    m3 = _box(colour=(200, 100, 50, 255), texture=True)
    src3, modulated3 = _colourise(m3, color0=half(m3))
    assert (src3, modulated3) == ("flat", True), (
        "narrowing the texture class must not switch off `flat` modulation"
    )
