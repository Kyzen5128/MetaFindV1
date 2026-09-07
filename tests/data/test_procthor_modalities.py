"""Tests for n07b_procthor_asset_modalities' GPU-free half.

The one that matters is the unprojection. It was wrong once in a way every other
check accepted: AI2-THOR returns a constant ~0.0142 depth under an orthographic
camera, so the first version produced 11 distinct views, no blank frames, 10,000
points, and a cloud 11 m across for an object 0.16 m wide.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from metafind.data.pointclouds import N_POINTS
from metafind.data.procthor_modalities import (
    DEPTH_FOV_DEG,
    LIFT_Y,
    ORBIT_RADIUS,
    fuse_depth_shell,
    orbit_camera_poses,
    strip_house,
    unproject_perspective,
)
from metafind.data.renders import N_VIEWS, ORBIT_ELEVATION_DEG


def house() -> dict:
    return {
        "rooms": [{"id": "room|0", "roomType": "Kitchen",
                   "ceilings": [{"id": "ceil|0"}]}],
        "objects": [{"assetId": "Fridge_19", "id": "Fridge|0|0",
                     "position": {"x": 1.0, "y": 0.0, "z": 2.0}},
                    {"assetId": "Chair_7", "id": "Chair|0|1",
                     "position": {"x": 3.0, "y": 0.0, "z": 2.0}}],
        "doors": [{"id": "door|0"}],
        "windows": [{"id": "win|0"}],
        "walls": [{"id": "wall|0"}, {"id": "wall|1"}],
        "proceduralParameters": {"lights": []},
    }


# --- isolation ------------------------------------------------------------

def test_stripped_house_holds_exactly_the_requested_asset():
    solo = strip_house(house(), "Chair_7")
    assert [o["assetId"] for o in solo["objects"]] == ["Chair_7"]


@pytest.mark.parametrize("field", ["doors", "windows", "walls"])
def test_the_shell_is_removed(field):
    """Leaving any of these in puts scenery behind the asset (F24)."""
    assert strip_house(house(), "Chair_7")[field] == []


def test_ceilings_are_removed_from_every_room():
    for room in strip_house(house(), "Chair_7")["rooms"]:
        assert room["ceilings"] == []


def test_the_asset_is_lifted_clear_of_the_floor():
    """The floor plane is what sliced the object in half in the probe."""
    assert strip_house(house(), "Chair_7")["objects"][0]["position"]["y"] == LIFT_Y
    assert LIFT_Y > 20, "must exceed any room's extent"


def test_stripping_does_not_mutate_the_template():
    """One Controller is reused across 1,467 assets via reset(), so a mutating
    strip would accumulate damage across the run."""
    h = house()
    before = json.dumps(h, sort_keys=True)
    strip_house(h, "Chair_7")
    assert json.dumps(h, sort_keys=True) == before


# --- camera layout --------------------------------------------------------

def test_the_orbit_uses_the_live_renderers_constants_not_copies():
    """[U-08b, BUILDER_VERSION 2] "n04-compatible" has to be enforced by import, not
    by matching numbers -- and from the LIVE renderer (`render_blender`, v7), not the
    retired pyrender module the v1 code imported (procthor_modalities.py header,
    corrected 2026-09-03)."""
    import math as _m

    import metafind.data.procthor_modalities as m
    import metafind.data.render_blender as rb

    assert m.N_VIEWS is rb.N_VIEWS and rb.N_VIEWS == 11
    assert m.ORBIT_ELEVATION_DEG is rb.ORBIT_ELEVATION_DEG
    assert m.RESOLUTION is rb.RESOLUTION
    assert m.PROJECTION == "perspective"
    assert abs(m.RGB_FOV_DEG - _m.degrees(2 * _m.atan(16 / 35))) < 1e-9   # 35 mm lens, 32 mm sensor
    assert abs(m.RGB_DISTANCE_FACTOR - 1.2 / 0.8) < 1e-12                # OpenShape framing
    assert len(orbit_camera_poses({"x": 0, "y": 0, "z": 0})) == N_VIEWS


def test_every_camera_sits_at_the_orbit_radius_from_the_centre():
    centre = {"x": 4.0, "y": 40.0, "z": 3.0}
    for pos, _ in orbit_camera_poses(centre):
        d = math.dist([pos["x"], pos["y"], pos["z"]],
                      [centre["x"], centre["y"], centre["z"]])
        assert abs(d - ORBIT_RADIUS) < 1e-9


def test_cameras_are_equally_spaced_in_azimuth():
    yaws = sorted(rot["y"] for _, rot in orbit_camera_poses({"x": 0, "y": 0, "z": 0}))
    gaps = np.diff(yaws)
    assert np.allclose(gaps, gaps[0], atol=1e-6)


def test_all_cameras_share_one_elevation():
    centre = {"x": 0.0, "y": 0.0, "z": 0.0}
    ys = {round(pos["y"], 9) for pos, _ in orbit_camera_poses(centre)}
    assert len(ys) == 1
    assert math.isclose(ys.pop(), ORBIT_RADIUS * math.sin(math.radians(ORBIT_ELEVATION_DEG)))


# --- the unprojection -----------------------------------------------------

def flat_depth(d: float, n: int = 32) -> np.ndarray:
    return np.full((n, n), d, dtype=np.float32)


def test_the_centre_pixel_lands_on_the_view_axis_at_the_reported_depth():
    """A camera `d` away, aimed at the origin, must place the centre pixel there.

    This is the assertion the orthographic-depth bug violated by 11 m.
    """
    centre = {"x": 4.0, "y": 40.0, "z": 3.0}
    pos, rot = orbit_camera_poses(centre)[0]
    d = ORBIT_RADIUS
    pts = unproject_perspective(flat_depth(d), pos, rot).reshape(32, 32, 3)
    got = pts[16, 16]
    want = np.array([centre["x"], centre["y"], centre["z"]])
    assert np.allclose(got, want, atol=0.25), f"{got} vs {want}"


def test_every_camera_agrees_on_where_the_centre_is():
    """Eleven poses, one object: if the frame conversion is wrong the clouds
    from different azimuths land in different places and the fused bounding box
    explodes -- which is exactly how the bug presented."""
    centre = {"x": 4.0, "y": 40.0, "z": 3.0}
    hits = []
    for pos, rot in orbit_camera_poses(centre):
        pts = unproject_perspective(flat_depth(ORBIT_RADIUS), pos, rot).reshape(32, 32, 3)
        hits.append(pts[16, 16])
    spread = np.ptp(np.stack(hits), axis=0)
    assert spread.max() < 0.5, f"the eleven views disagree by {spread}"


def test_the_in_plane_offset_scales_with_depth():
    """[the bug] An orthographic formula omits the depth factor, so a corner
    pixel keeps a fixed offset however far away the surface is. Perspective rays
    fan out; doubling the depth doubles the footprint."""
    pos, rot = {"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 0.0}
    near = unproject_perspective(flat_depth(1.0), pos, rot)
    far = unproject_perspective(flat_depth(2.0), pos, rot)
    assert np.ptp(far[:, 0]) > 1.9 * np.ptp(near[:, 0])


def test_the_footprint_matches_the_declared_field_of_view():
    pos, rot = {"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 0.0}
    d = 3.0
    pts = unproject_perspective(flat_depth(d), pos, rot)
    half_height = np.ptp(pts[:, 1]) / 2
    expected = math.tan(math.radians(DEPTH_FOV_DEG) / 2) * d
    # pixel centres, so the sampled extent is one pixel short of the full frustum
    assert half_height == pytest.approx(expected, rel=0.05)


# --- fusion ---------------------------------------------------------------

def test_the_cloud_has_exactly_n_points():
    poses = orbit_camera_poses({"x": 0.0, "y": 0.0, "z": 0.0})
    depths = [flat_depth(2.0) for _ in poses]
    cloud = fuse_depth_shell(depths, poses, far_cut=10.0)
    assert cloud.shape == (N_POINTS, 3)
    assert cloud.dtype == np.float32


def test_background_pixels_are_dropped_by_the_far_cut():
    """AI2-THOR reports the far plane for a ray that hits nothing. Keeping those
    would put a 20 m sphere of sky around every asset."""
    poses = orbit_camera_poses({"x": 0.0, "y": 0.0, "z": 0.0})
    d = flat_depth(2.0)
    d[:16, :] = 19.99  # half the frame is sky
    cloud = fuse_depth_shell([d] * len(poses), poses, far_cut=10.0)
    assert np.abs(cloud).max() < 6.0, "sky survived the cut"


def test_a_fully_empty_capture_is_an_error_not_an_empty_cloud():
    """An empty cloud would flow downstream as a valid-looking record."""
    poses = orbit_camera_poses({"x": 0.0, "y": 0.0, "z": 0.0})
    with pytest.raises(ValueError):
        fuse_depth_shell([flat_depth(19.99)] * len(poses), poses, far_cut=10.0)


def test_fusion_is_seeded():
    poses = orbit_camera_poses({"x": 0.0, "y": 0.0, "z": 0.0})
    depths = [flat_depth(2.0) for _ in poses]
    a = fuse_depth_shell(depths, poses, far_cut=10.0,
                         rng=np.random.default_rng(7))
    b = fuse_depth_shell(depths, poses, far_cut=10.0,
                         rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


# --- orphan sweep ----------------------------------------------------------

PS_OUTPUT = """    PID    PPID CMD
   1000       1 /home/k/.ai2thor/releases/thor-CloudRendering-abc/thor-CloudRendering-abc -screen-width 224
   1001    9999 /home/k/.ai2thor/releases/thor-CloudRendering-abc/thor-CloudRendering-abc -screen-width 224
   1002       1 python -m metafind.data.procthor_modalities
   1003       1 grep thor-CloudRendering
"""


def sweep_with(monkeypatch, ps_text):
    """Run the sweep against a fabricated `ps`, recording what it would kill.

    Killing real processes to test a killer is not a test, it is a hazard: a
    PPID that happens to be 1 on the machine running pytest would take out
    something unrelated.
    """
    import subprocess
    import metafind.data.procthor_modalities as m
    from metafind.data.procthor_modalities import ThorRenderer

    killed = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": ps_text})())
    monkeypatch.setattr(m.os, "kill", lambda pid, sig: killed.append(pid))
    ThorRenderer.sweep_orphans()
    return killed


def test_only_orphaned_thor_processes_are_killed(monkeypatch):
    """PPID == 1 is the whole criterion: this node never runs two Controllers,
    so a ProcTHOR Unity process adopted by init belongs to nobody."""
    assert sweep_with(monkeypatch, PS_OUTPUT) == [1000]


def test_a_live_child_is_left_alone(monkeypatch):
    """1001's parent is a running python; killing it would break that run."""
    assert 1001 not in sweep_with(monkeypatch, PS_OUTPUT)


def test_the_python_process_itself_is_not_killed(monkeypatch):
    """Its command line contains the module name, not the Unity binary's."""
    assert 1002 not in sweep_with(monkeypatch, PS_OUTPUT)


def test_a_matching_grep_line_is_ignored(monkeypatch):
    assert 1003 not in sweep_with(monkeypatch, PS_OUTPUT)


def test_nothing_to_sweep_kills_nothing(monkeypatch):
    assert sweep_with(monkeypatch, "    PID    PPID CMD\n   1002       1 python foo\n") == []
