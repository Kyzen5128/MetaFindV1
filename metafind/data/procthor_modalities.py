"""Isolated renders, depth-shell point clouds and text for every ProcTHOR asset.

# IMPLEMENTS-NODE: n07b_procthor_asset_modalities

Writes ``procthor_asset_modalities`` (one sidecar per asset), and
``quarantine`` / ``run_progress`` / ``cost_ledger`` via runlog.

Why this node exists
--------------------

[U-08b RESOLVED] Stage 2's target is a ProcTHOR asset and its positive is that
same asset, so the query encoder needs the asset's text, image and point cloud.
ProcTHOR's JSONL carries none of them -- but ProcTHOR is not its JSONL. The
houses exist to be loaded into AI2-THOR, which renders.

Without this node the resolution was a sentence: n13 would have had no way to
obtain e_query, and "Fridge_19 -> Fridge_19" would have named a positive nobody
could encode.

The recipe is not obvious (F24)
-------------------------------

Three attempts. A house must be stripped of doors, windows, walls AND ceilings,
and the asset lifted clear of the floor, or the floor plane fills the frame and
slices the object in half. The first version passed every numeric check I wrote
-- 11 distinct views, no blank view, background fraction 19-45% -- while the
images contained a grey post that appears in no metadata listing and a fridge
cut in two. Numbers said pass, pictures said no.

What is comparable to the gallery, and what is not
--------------------------------------------------

Images are NOT comparable to the Objaverse gallery today. The camera layout,
elevation, count, projection and resolution are imported from ``renders``
rather than copied, which was meant to make them so.

[CORRECTED 2026-09-03] This claimed that made "n04-compatible" true BY
CONSTRUCTION and that a change to n04 could not silently desynchronise the two
sides. Both halves are false, and the second is the dangerous one.

`renders.py` says of exactly those constants that they "describe the RETIRED
pyrender path ... nothing in `process_one` reads them any more". The live
Objaverse path goes through `render_blender` via `LIVE_N_VIEWS`. So the import
binds this node to a DEAD code path: the guarantee is real at the level of
Python names and says nothing about the corpus, which is the level a reader
takes it at. Measured divergence between what the two sides actually hold:

    ProcTHOR (this node)      Objaverse (the corpus)
    11 views                  12 views
    orthographic              perspective
    224 px                    512 px
    single orbit, elev 20     three polar rings of four
    AI2-THOR skybox           transparent RGBA

The import is LEFT AS IT IS and no value is changed -- moving either side is a
protocol decision, and the reproduction spec of 2026-09-03 calls the camera
protocol UNRESOLVED. What is corrected is only the claim. The two renders are
not interchangeable today, and nothing in code enforces that they ever were.

Point clouds are NOT. n03 samples a complete mesh surface and reaches occluded
faces; this back-projects depth from the same 11 views and reaches the visible
hull only. The gallery encoder is frozen, so PointBERT cannot adapt to that
shift -- which is why ``query_pointcloud`` is optional in stage2_protocol and
why every record carries provenance saying which kind it holds.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog
from metafind.data.renders import (
    N_VIEWS,
    ORBIT_ELEVATION_DEG,
    PROJECTION,
    RESOLUTION,
)
from metafind.data.pointclouds import N_POINTS

NODE = "n07b_procthor_asset_modalities"
BUILDER_VERSION = 1

# The asset is lifted this far above the room floor so nothing else can enter
# frame. 40 m is arbitrary and only has to exceed any room's extent; ProcTHOR
# rooms are a few metres tall.
LIFT_Y = 40.0

# Half-height of the orthographic frustum, as a multiple of the asset's largest
# bounding-box dimension. 0.62 clipped the top of tall assets in the probe.
ORTHO_MARGIN = 0.80
ORBIT_RADIUS = 6.0  # orthographic, so this only has to clear the geometry

# MEASURED: AI2-THOR's third-party depth frame is only correct under PERSPECTIVE
# projection. Under orthographic it returns a constant ~0.0142 across every
# foreground pixel of an object 6 m away -- not a distance, and not even varying
# across a surface seen at an angle. The same pose in perspective returns
# 5.944-6.034, which is exactly right.
#
# So each asset is captured twice at the same 11 poses: orthographic for RGB,
# because that is what makes the images comparable to n04's, and perspective for
# depth, because that is the only way the geometry is real. The projection used
# to CAPTURE depth does not change what the cloud is -- it is a visible-hull
# shell either way, and the record says so.
#
# The depth pass uses its own framing: a 45-degree field of view at 2.5x the
# asset's largest dimension. Matching the orthographic orbit would have meant a
# ~2-degree lens at 6 m, which is numerically poor for unprojection and buys
# nothing, since only the RGB has to match n04.
DEPTH_FOV_DEG = 45.0
DEPTH_DISTANCE_FACTOR = 2.5

# The cloud's extent against the box AI2-THOR reports, as a RATIO rather than an
# absolute error -- because the reference itself is not exact. MEASURED over 192
# assets: 0.94 to 1.53, median 1.005. The high end is entirely beds, whose
# duvets and pillows hang off the frame; the renders show the cloth clearly and
# the reported box does not contain it, so AI2-THOR's box tracks the collider,
# not the drape. Those clouds are RIGHT and the reference is short.
#
# An absolute threshold would have quarantined 31 of the first 163 assets for
# being correct. The bound below still catches what this check exists for: the
# orthographic-depth bug produced a ratio of 69.
BBOX_RATIO_MIN = 0.5
BBOX_RATIO_MAX = 3.0

# The ratio alone has the MIRROR of the absolute bound's flaw. Wall_Decor_
# Painting_6 is 7 mm thick; the shell measured 25 mm, a ratio of 3.4, from an
# absolute discrepancy of EIGHTEEN MILLIMETRES on an object 0.8 m across. Its
# other two axes agreed to within 4 mm. Dividing by a near-zero dimension makes
# depth quantisation look like a catastrophe.
#
# So a failure needs BOTH: a ratio outside the band AND a discrepancy that is
# large relative to the object itself. The orthographic-depth bug cleared both
# by a wide margin -- 11.06 m of error on a 0.16 m object -- while the painting
# and the beds clear neither.
BBOX_ABS_ERROR_FRACTION = 0.25


def strip_house(house: dict, asset_id: str) -> dict:
    """A house holding exactly one asset, lifted clear of everything else.

    Every removal here was necessary in the probe. Leaving walls in puts a wall
    behind the asset; leaving the floor in puts an orange plane through it.
    """
    solo = json.loads(json.dumps(house))
    solo["objects"] = [{
        "assetId": asset_id,
        "id": f"{asset_id}|0|0",
        "position": {"x": 4.0, "y": LIFT_Y, "z": 3.0},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "kinematic": True,
    }]
    solo["doors"] = []
    solo["windows"] = []
    for room in solo.get("rooms", []):
        room["ceilings"] = []
    solo["walls"] = []
    return solo


def orbit_camera_poses(centre: dict, radius: float = ORBIT_RADIUS,
                       n: int = N_VIEWS,
                       elevation_deg: float = ORBIT_ELEVATION_DEG) -> list[tuple[dict, dict]]:
    """n04's azimuth orbit, expressed in AI2-THOR's left-handed y-up frame.

    The layout is not re-derived: N_VIEWS and ORBIT_ELEVATION_DEG are imported
    from ``renders``, so the two sides cannot drift apart by someone editing one
    number. What is expressed here is only the frame change -- trimesh's z-up to
    Unity's y-up.
    """
    poses = []
    el = math.radians(elevation_deg)
    for k in range(n):
        az = 2.0 * math.pi * k / n
        pos = {
            "x": centre["x"] + radius * math.cos(el) * math.sin(az),
            "y": centre["y"] + radius * math.sin(el),
            "z": centre["z"] + radius * math.cos(el) * math.cos(az),
        }
        rot = {"x": elevation_deg, "y": (math.degrees(az) + 180.0) % 360.0, "z": 0.0}
        poses.append((pos, rot))
    return poses


def unproject_perspective(depth: np.ndarray, pos: dict, rot: dict,
                          fov_deg: float = DEPTH_FOV_DEG) -> np.ndarray:
    """Depth frame -> world points, for a PERSPECTIVE camera.

    `depth` is planar z-distance along the view axis, so the in-plane offset of
    a pixel grows with depth -- that factor is what an orthographic formula
    would omit, returning a cloud that looks plausible and is the wrong shape.

    AI2-THOR is left-handed and y-up, with rotation applied as yaw about y then
    pitch about x. None of that is asserted from this docstring: the caller
    compares the resulting cloud's bounding box against the box AI2-THOR itself
    reports, which is what caught the orthographic-depth bug in the first place.
    """
    h, w = depth.shape
    aspect = w / h
    # Pixel centres in [-1, 1], y flipped: row 0 is the top of the image.
    u = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    v = 1.0 - (np.arange(h) + 0.5) / h * 2.0
    uu, vv = np.meshgrid(u, v)

    half = math.tan(math.radians(fov_deg) / 2.0)
    cam = np.stack([uu * half * aspect * depth,
                    vv * half * depth,
                    depth], axis=-1).reshape(-1, 3)

    pitch = math.radians(rot["x"])
    yaw = math.radians(rot["y"])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    world = cam @ (r_yaw @ r_pitch).T
    return world + np.array([pos["x"], pos["y"], pos["z"]])


def fuse_depth_shell(depths: list[np.ndarray], poses: list[tuple[dict, dict]],
                     far_cut: float, n_points: int = N_POINTS,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """The 11 depth frames into one point cloud of exactly ``n_points``.

    Background pixels are dropped by depth: AI2-THOR reports the far plane for
    a ray that hits nothing, which is far beyond a lifted asset.
    """
    rng = rng or np.random.default_rng(0)
    chunks = []
    for depth, (pos, rot) in zip(depths, poses):
        finite = np.isfinite(depth)
        # Anything past the orbit radius plus the object's own extent is sky.
        hit = finite & (depth < far_cut)
        if not hit.any():
            continue
        pts = unproject_perspective(depth, pos, rot)
        chunks.append(pts[hit.reshape(-1)])
    if not chunks:
        raise ValueError("every view was empty; the asset never entered frame")
    cloud = np.concatenate(chunks, axis=0)
    idx = rng.choice(len(cloud), size=n_points, replace=len(cloud) < n_points)
    return cloud[idx].astype(np.float32)


def sidecar_path(asset_id: str) -> Path:
    return paths.PROCTHOR_MODALITIES / f"{asset_id}.json"


def is_complete(asset_id: str) -> bool:
    sc = sidecar_path(asset_id)
    if not sc.exists():
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if rec.get("builder_version") != BUILDER_VERSION:
        return False
    if len(rec.get("view_paths", [])) != N_VIEWS:
        return False
    uri = rec.get("pointcloud_uri")
    if uri is None:
        # A recorded absence, not an unfinished asset. Requiring the file here
        # would re-render every transparent asset on every resume, forever.
        return bool(rec.get("pointcloud_missing_reason"))
    return Path(uri).exists()


def asset_ids_in_use() -> list[str]:
    """[U-08c] The asset set is DERIVED, never a published figure.

    Taken from the scene graphs rather than the simulator's asset database:
    these are the assets that actually appear in the corpus Stage 2 trains on.
    Which of the two sets to use is itself recorded -- the database holds more
    (1,934 against 1,467) and those extras have no scene occurrence, so they
    could only ever be negatives.
    """
    text_map = json.loads((paths.OUTPUTS / "procthor_object_text.json").read_text())
    return sorted(text_map)


class ThorRenderer:
    """One Controller for the whole run; `reset` swaps the house per asset.

    MEASURED: killing the Python process leaves the Unity binary ORPHANED. Its
    parent becomes init, it keeps 2.1 GB of RAM and 1,545 MiB of GPU, and it
    lives until someone notices -- 46 minutes, in the run that found this,
    taking that memory from the job it was stopped to make room for.
    `Controller.stop()` at the end of main() only covers the path where main()
    reaches its end. Registered here instead, so a SIGTERM or an exception takes
    the child with it.
    """

    @staticmethod
    def sweep_orphans() -> int:
        """Kill Unity processes left behind by a previous run.

        The signal handlers below cover a clean SIGTERM. They do NOT cover
        SIGKILL, an OOM kill, or a crashed interpreter -- and the orphan that
        prompted this survived exactly that way, holding 2.1 GB of RAM and
        1,545 MiB of GPU for 46 minutes while the job it had been stopped for
        ran short of both.

        A sweep at startup is the half that does not need the previous run to
        cooperate. PPID == 1 identifies an orphan unambiguously here: this node
        never runs two Controllers at once, so a ProcTHOR Unity process adopted
        by init belongs to nobody.
        """
        import subprocess

        out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"],
                             capture_output=True, text=True).stdout
        killed = 0
        for line in out.splitlines():
            if "thor-CloudRendering" not in line or " grep " in line:
                continue
            parts = line.split()
            if len(parts) < 2 or parts[1] != "1":
                continue
            try:
                os.kill(int(parts[0]), signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass
        if killed:
            print(f"swept {killed} orphaned AI2-THOR process(es) from a "
                  "previous run", flush=True)
        return killed

    def __init__(self) -> None:
        import ai2thor
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering

        self.sweep_orphans()
        self.version = ai2thor.__version__
        self._CloudRendering = CloudRendering
        first = json.loads((paths.PROCTHOR / "train.jsonl").open().readline())
        self.template = first
        self.controller = Controller(
            scene=strip_house(first, first["objects"][0]["assetId"]),
            platform=CloudRendering, width=RESOLUTION, height=RESOLUTION,
            renderDepthImage=True, quality="Medium",
        )
        self.build_hash = self._build_hash()
        self._stopped = False
        atexit.register(self.stop)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(sig)

            def handler(signum, frame, _prev=previous):
                self.stop()
                if callable(_prev):
                    _prev(signum, frame)
                else:
                    raise SystemExit(128 + signum)

            signal.signal(sig, handler)

    def _build_hash(self) -> str:
        for p in (Path.home() / ".ai2thor" / "releases").glob("thor-CloudRendering-*"):
            if p.is_dir():
                return p.name.rsplit("-", 1)[-1]
        return "unknown"

    def render(self, asset_id: str) -> dict:
        """Two passes at the same 11 azimuths: orthographic RGB, perspective depth."""
        self.controller.reset(scene=strip_house(self.template, asset_id))
        named = [o for o in self.controller.last_event.metadata["objects"]
                 if o.get("assetId") == asset_id]
        if not named:
            raise ValueError(f"{asset_id} did not load into the stripped house")
        bb = named[0]["axisAlignedBoundingBox"]
        centre, size = bb["center"], bb["size"]
        largest = max(size.values())
        if largest <= 0:
            raise ValueError(f"{asset_id} reports a degenerate bounding box: {size}")
        ortho = largest * ORTHO_MARGIN

        frames = []
        for pos, rot in orbit_camera_poses(centre):
            ev = self.controller.step(
                action="AddThirdPartyCamera", position=pos, rotation=rot,
                orthographic=True, orthographicSize=ortho, skyboxColor="white",
            )
            if not ev.metadata["lastActionSuccess"]:
                raise ValueError(f"RGB camera failed: {ev.metadata.get('errorMessage')}")
            frames.append(ev.third_party_camera_frames[-1][..., :3])

        depth_radius = largest * DEPTH_DISTANCE_FACTOR
        depth_poses = orbit_camera_poses(centre, radius=depth_radius)
        depths = []
        for pos, rot in depth_poses:
            ev = self.controller.step(
                action="AddThirdPartyCamera", position=pos, rotation=rot,
                orthographic=False, fieldOfView=DEPTH_FOV_DEG, skyboxColor="white",
            )
            if not ev.metadata["lastActionSuccess"]:
                raise ValueError(f"depth camera failed: {ev.metadata.get('errorMessage')}")
            depths.append(ev.third_party_depth_frames[-1])

        return {"frames": frames, "depths": depths, "depth_poses": depth_poses,
                "ortho": ortho, "bb": bb,
                "far_cut": depth_radius + largest * 2.0}

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self.controller.stop()
        except Exception:  # noqa: BLE001 -- shutting down; a failure here must
            pass          # not mask the reason we are shutting down

    def __enter__(self) -> "ThorRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _write_asset(asset_id: str, cap: dict, text: str,
                 renderer: "ThorRenderer") -> dict:
    from PIL import Image

    out = paths.PROCTHOR_MODALITIES / asset_id
    out.mkdir(parents=True, exist_ok=True)
    view_paths, view_bytes = [], []
    for i, f in enumerate(cap["frames"]):
        vp = out / f"view_{i:02d}.png"
        Image.fromarray(f.astype(np.uint8)).save(vp)
        view_paths.append(str(vp))
        view_bytes.append(vp.stat().st_size)

    # MEASURED: some ProcTHOR assets never appear in AI2-THOR's depth buffer at
    # all -- Bottle_1, CD_1 and eleven Bowls returned a uniform far-plane depth
    # at every distance tried, while RGB rendered them normally. They are the
    # glass and glossy ones, and Unity's depth prepass carries opaque geometry
    # only. It is a property of the ASSET, not of the camera: the same probe at
    # 0.5, 1, 3 and 6 m gave depth for an alarm clock and nothing for a bowl.
    #
    # So this is a missing MODALITY, not a failed asset. The record is written
    # with text and eleven views and an explicit null, because 2.4 lets the query
    # side take any subset and stage2_protocol already has query_pointcloud
    # optional. Quarantining would have thrown away two good modalities; a
    # zero-filled cloud would have been indistinguishable from a real one.
    pc_path = out / "pointcloud.npz"
    cloud, missing = None, None
    try:
        cloud = fuse_depth_shell(cap["depths"], cap["depth_poses"], cap["far_cut"])
        tmp = out / "pointcloud.part.npz"
        np.savez_compressed(tmp, xyz=cloud)
        tmp.replace(pc_path)
    except ValueError as exc:
        missing = f"{exc}. AI2-THOR returned no depth for this asset at any "\
                  "distance; its material is not in the depth prepass."
        pc_path.unlink(missing_ok=True)

    # The cloud must occupy the box AI2-THOR itself reports. This is the only
    # thing standing between a correct unprojection and a plausible-looking wrong
    # one, and it is not hypothetical: the first version captured depth under an
    # orthographic camera, which AI2-THOR returns as a constant ~0.0142 whatever
    # the real distance. Every count and shape check passed; this comparison read
    # 11.063 m against an object 0.16 m wide.
    bb = cap["bb"]
    reported = np.array([bb["size"]["x"], bb["size"]["y"], bb["size"]["z"]])
    if cloud is None:
        measured = np.array([float("nan")] * 3)
        bbox_err, ratio = float("nan"), float("nan")
    else:
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        measured = hi - lo
        bbox_err = float(np.abs(measured - reported).max())
        ratio = float((measured / np.maximum(reported, 1e-6)).max())
    if cloud is not None:
        ratio_bad = not BBOX_RATIO_MIN <= ratio <= BBOX_RATIO_MAX
        scale_bad = bbox_err > BBOX_ABS_ERROR_FRACTION * float(reported.max())
        if ratio_bad and scale_bad:
            raise ValueError(
                f"cloud extent is {ratio:.2f}x the reported bounding box "
                f"(allowed {BBOX_RATIO_MIN}-{BBOX_RATIO_MAX}) AND the error is "
                f"{bbox_err:.3f} m against a largest dimension of "
                f"{reported.max():.3f} m; reported {reported.round(3).tolist()}, "
                f"measured {measured.round(3).tolist()}. The unprojection is "
                "producing geometry unrelated to the asset."
            )

    return {
        "asset_id": asset_id,
        "builder_version": BUILDER_VERSION,
        "text": text,
        "view_paths": view_paths,
        "view_bytes": view_bytes,
        "pointcloud_uri": None if cloud is None else str(pc_path),
        "pointcloud_missing_reason": missing,
        "n_points": 0 if cloud is None else int(len(cloud)),
        "pointcloud_kind": None if cloud is None else "multiview_depth_shell",
        "bbox_reported": reported.tolist(),
        "bbox_measured": measured.tolist(),
        "bbox_max_abs_error_m": bbox_err,
        "bbox_size_ratio": ratio,
        "image_protocol": {"n_views": N_VIEWS, "resolution": RESOLUTION,
                           "projection": PROJECTION,
                           "elevation_deg": ORBIT_ELEVATION_DEG,
                           "orthographic_size": cap["ortho"]},
        "depth_protocol": {"projection": "perspective",
                           "fov_deg": DEPTH_FOV_DEG,
                           "distance_factor": DEPTH_DISTANCE_FACTOR,
                           "why": "AI2-THOR returns a constant, wrong depth under "
                                  "orthographic projection; measured 0.0142 for a "
                                  "camera 6 m away"},
        "provenance": {"ai2thor_version": renderer.version,
                       "build_hash": renderer.build_hash,
                       "procthor_revision": os.environ.get(
                           "METAFIND_PROCTHOR_REV",
                           "439193522244720b86d8c81cde2e51e3a4d150cf")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    text_map_path = paths.OUTPUTS / "procthor_object_text.json"
    if not text_map_path.exists():
        print(f"{text_map_path} not found -- run n07_scene_graphs first", flush=True)
        return 2
    text_map = json.loads(text_map_path.read_text())
    paths.PROCTHOR_MODALITIES.mkdir(parents=True, exist_ok=True)

    todo = [a for a in asset_ids_in_use() if args.force or not is_complete(a)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(text_map):,} assets in use, {len(todo):,} to render", flush=True)
    if not todo:
        return 0

    renderer = ThorRenderer()
    done, quarantined, no_cloud, started = 0, 0, 0, time.time()
    worst_bbox = 0.0
    with runlog.run_progress(NODE):
        for asset_id in todo:
            try:
                cap = renderer.render(asset_id)
                rec = _write_asset(asset_id, cap, text_map[asset_id]["text"], renderer)
            except Exception as exc:  # noqa: BLE001 -- one asset must not stop the run
                runlog.quarantine(NODE, [{
                    "asset_id": asset_id,
                    "failure_class": "DETERMINISTIC_INPUT",
                    "stage": "render",
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                quarantined += 1
                continue

            sc = sidecar_path(asset_id)
            tmp = sc.with_suffix(".json.part")
            with tmp.open("w") as fh:
                json.dump(rec, fh)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(sc)
            if rec["pointcloud_uri"] is None:
                no_cloud += 1
            else:
                worst_bbox = max(worst_bbox, rec["bbox_size_ratio"])
            done += 1
            if done % 50 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                print(f"  [{done:5d}/{len(todo)}] {rate:.1f}/min, "
                      f"quarantine {quarantined}, no-cloud {no_cloud}, "
                      f"worst bbox ratio {worst_bbox:.2f}x",
                      flush=True)

    renderer.stop()
    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       assets_rendered=done, views_written=done * N_VIEWS)
    print(f"\n{done:,} rendered, {quarantined:,} quarantined, "
          f"{no_cloud:,} without a point cloud (text+image only), "
          f"worst bbox ratio {worst_bbox:.2f}x -> {paths.PROCTHOR_MODALITIES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
