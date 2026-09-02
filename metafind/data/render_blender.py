"""Drive OpenShape's Blender renderer. The only place that knows about Blender.

# SUPPORTS-NODE: n04_render_views

Why Blender and not pyrender
----------------------------

**No rendering code exists anywhere in the ULIP lineage.** The ULIP repository
has none; ULIP-2's paper gives one sentence (`main.tex:677`, "we use Blender to
render 12 images, spaced equally by 360/12 degrees"); MetaFind's paper gives
one clause (`2methdology.tex:28`, "rendered from 11 orthogonal viewpoints").
The only executable renderer in the chain is **OpenShape's**, which ULIP-2's
appendix defers to for 3D input preprocessing (`appendix.tex:10`) and whose
released `.npy` schema the ULIP-2 Objaverse triplets reproduce key for key.

`metafind/vendor/openshape/render_single_glb.py` is that file, byte-identical
to upstream `abe5aa42`. This module only invokes it.

Measured, 2026-08-23, this machine (RTX 5090, OptiX):

    pyrender  0.07-0.68 s/asset   11 views  224px  orthographic  opaque white
    Blender   3.234  s/asset      12 views  512px  perspective   transparent
                                  at 8 concurrent processes

The 100x is bought with something specific and checkable. pyrender does not
render transparent materials at all: on a bubble-tea asset it produced a straw
and some floating tapioca with **no cup**, and the cup did not reappear when the
background was switched to black or when the image was brightened 8x -- 91.15%
of that frame was exactly zero, meaning nothing was drawn there. Blender renders
the cup. Roughly half the Objaverse-LVIS corpus carries some transparency.

What differs from upstream, and why
-----------------------------------

**Two commented-out lines.** `enable_normals_output()` and
`enable_depth_output()`, plus the loops that save them. Nothing downstream
consumes normals or depth, and they are two extra render passes and two thirds
of the files (1.65M -> 552k). Measured: 8.00 s -> 7.53 s single-process, and the
colour images stay within run-to-run noise -- CYCLES is not bit-reproducible, so
two runs of the *unmodified* script already differ by up to 1/255, and the
patched-vs-unpatched difference is 2/255, the same order.

**Twelve views, not eleven.** [USER DECISION 2026-08-23.] MetaFind's paper says
eleven (`2methdology.tex:28`, PAPER FACT) and this is a **registered DEVIATION**
from it: OpenShape's twelve are three polar rings of four and eleven does not
divide into three rings.

**The denoiser is named explicitly: OptiX, on the GPU.** [USER DECISION
2026-08-24.] Upstream sets `use_denoising = True` and stops there, so the
denoiser actually in force was BlenderProc's default `INTEL` -- Intel
OpenImageDenoise, in the compositor, on the CPU. Measured on this machine,
same asset, same load, one process each: **INTEL 27.8/26.8/26.8 s vs OPTIX
5.9/5.9/6.2 s**. The images are not identical -- mean |diff| 0.1/255, 2-3% of
pixels differ by more than 2, max 50/255 -- so this is a registered DEVIATION
from the corpus rendered before 2026-08-24, which is why `RENDERER_VERSION`
goes to 6 and the whole corpus is re-rendered rather than mixed.

**Transparent RGBA is what gets stored.** Flattening is a consumer's decision
and lives in `metafind.data.view_io`, once, for every consumer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import trimesh

__all__ = [
    "BLENDERPROC", "BLENDER_INSTALL", "CAMERA_DIST", "ENGINE", "N_VIEWS",
    "RESOLUTION", "VIEW_DIRECTIONS", "render_asset", "renderer_versions",
]

_HERE = Path(__file__).resolve().parent
VENDOR_SCRIPT = _HERE.parent / "vendor" / "openshape" / "render_single_glb.py"

# The blenderproc CLI and the Blender tree it manages. Overridable so a
# different machine does not need this file edited.
BLENDERPROC = Path(os.environ.get(
    "METAFIND_BLENDERPROC", "/home/kyzen/metafind_out/bproc_env/bin/blenderproc"))
BLENDER_INSTALL = Path(os.environ.get(
    "METAFIND_BLENDER_INSTALL", "/home/kyzen/metafind_out/blender"))

# All four are upstream's own values. `camera_dist 1.2` is the number in the
# invocation comment at the top of their script, not argparse's 1.5 default.
ENGINE = "CYCLES"
N_VIEWS = 12
CAMERA_DIST = 1.2
RESOLUTION = 512

# NOT an upstream value. OpenShape's script asks for denoising
# (`scene.cycles.use_denoising = True`) but never names a denoiser, so the
# effective choice was BlenderProc's default, `DefaultConfig.denoiser = "INTEL"`
# -- Intel OpenImageDenoise, which runs in the compositor on the CPU.
# [USER DECISION 2026-08-24] switch it to OptiX, which denoises on the GPU.
DENOISER = "OPTIX"

# Upstream's `views` list, restated here so the layout is greppable without
# opening vendored code. Three polar rings of four, staggered in azimuth --
# NOT the single 360/12 orbit ULIP-2's sentence describes. phi is measured from
# +Z, so 120 deg looks UP at the object from below.
VIEW_DIRECTIONS = (
    ("phi_60_above", (30, 120, 210, 300)),
    ("phi_90_level", (60, 150, 240, 330)),
    ("phi_120_below", (0, 90, 180, 270)),
)

# Rendering is a subprocess, so a source-hash guard over this module says
# nothing about what actually drew the pixels. These three do.
def renderer_versions() -> dict[str, str]:
    """Provenance for the sidecar: what actually produced the pixels."""
    import hashlib

    out = {
        "engine": ENGINE,
        "n_views": str(N_VIEWS),
        "resolution": str(RESOLUTION),
        "camera_dist": str(CAMERA_DIST),
        "vendor_script_sha256":
            hashlib.sha256(VENDOR_SCRIPT.read_bytes()).hexdigest()[:16],
        "aux_passes": "disabled",
        "denoiser": DENOISER,
    }
    commit = VENDOR_SCRIPT.with_name("COMMIT")
    if commit.exists():
        out["openshape_commit"] = commit.read_text().strip()[:12]
    try:
        out["blenderproc"] = subprocess.run(
            [str(BLENDERPROC), "--version"], capture_output=True, text=True,
            timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out["blenderproc"] = "unknown"
    return out


def _patched_script(dst: Path) -> Path:
    """Upstream's script with only the two unused passes commented out.

    Written from the vendored bytes at call time rather than kept as a second
    checked-in copy, so the patch can never silently drift away from the file it
    claims to be a two-line edit of. Each replacement is asserted, so a change
    upstream turns into a loud failure instead of a silently unpatched run.
    """
    src = VENDOR_SCRIPT.read_text()
    edits = [
        ("    bproc.renderer.enable_normals_output()\n"
         "    bproc.renderer.enable_depth_output(activate_antialiasing=False)\n",
         "    # [METAFIND] normals/depth are unused downstream; the passes cost\n"
         "    # render time and two thirds of the files. Colour output unchanged\n"
         "    # to within CYCLES' own run-to-run noise (measured 2/255 vs 1/255).\n"),
        ('    for index, image in enumerate(data["normals"]):\n'
         '        render_path = os.path.join(args.output_dir, f"{index:03d}_normal.png")\n'
         '        save_array_as_image(image, "normals", render_path)\n',
         "    # [METAFIND] normals pass disabled\n"),
        ('    for index, image in enumerate(data["depth"]):\n'
         '        render_path = os.path.join(args.output_dir, f"{index:03d}_depth.png")\n'
         '        #save_array_as_image(image, "depth", render_path)\n'
         '        #print(image.min(), image[image < 100].max())\n'
         '        #vis_data("depth", image, None, "", save_to_file=render_path, depth_max = image[image < 100].max())\n'
         "        plt.imsave(render_path, image, cmap='gray', vmax=image[image < 100].max())\n",
         "    # [METAFIND] depth pass disabled\n"),
        ("    RendererUtility.set_cpu_threads(10)\n",
         "    RendererUtility.set_cpu_threads(10)\n"
         "    # [METAFIND] upstream leaves the denoiser unnamed, so BlenderProc's\n"
         "    # CPU default (INTEL/OIDN) was in force. Name it explicitly.\n"
         f"    RendererUtility.set_denoiser({DENOISER!r})\n"),
    ]
    for old, new in edits:
        if old not in src:
            raise RuntimeError(
                "the vendored OpenShape renderer no longer contains a block this "
                f"patch expects; refusing to render with an unverified script:\n{old[:90]}")
        src = src.replace(old, new, 1)
    dst.write_text(src)
    return dst


def raw_extents(glb: Path):
    """Axis-aligned bounding box of the asset in its own units, before Blender.

    Blender normalises the scene (longest bbox side -> 0.8, bbox centre ->
    origin) and the real scale is gone after that. `F13` wants it recorded, and
    it is cheap: loading the mesh without rendering it.

    Uses `meshload` so n03 and n04 cannot disagree about the frame -- the same
    180 degree yaw is applied on both sides. The yaw is a rotation, so the
    axis-aligned extents it produces are a permutation-with-signs of the
    unrotated ones and the numbers stay comparable either way.
    """
    import numpy as np

    from metafind.data import meshload

    scene = meshload.load_scene(glb)
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry.get(name)
        # Same filter as the point sampler (pointclouds.py): triangle meshes
        # with faces. Counting any geometry with vertices (Path3D, PointCloud)
        # gave a different bounding box from the one the annotation was
        # prompted with for 97 assets, by up to 3x.
        if (geom is None or not isinstance(geom, trimesh.Trimesh)
                or len(geom.faces) == 0):
            continue
        v = np.asarray(geom.vertices, dtype=np.float64)
        v = v @ transform[:3, :3].T + transform[:3, 3]
        lo = np.minimum(lo, v.min(axis=0))
        hi = np.maximum(hi, v.max(axis=0))
    if not np.isfinite(lo).all():
        raise ValueError("no geometry with vertices in this GLB")
    return hi - lo


def render_asset(glb: Path, asset_dir: Path, *, timeout: int = 900) -> list[Path]:
    """Render one GLB into ``asset_dir`` as ``view_00.png`` .. ``view_11.png``.

    Blender writes into a private temporary directory and the results are moved
    into place only once all `N_VIEWS` exist. A crashed or killed Blender
    therefore leaves `asset_dir` untouched rather than half-populated, which is
    what lets the caller treat "the sidecar exists" as "the asset is finished".

    Raises on a non-zero exit or a short view count, with Blender's own stderr
    tail attached -- a silent retry on a systematically broken asset would spend
    the run rediscovering the same failure.
    """
    if not BLENDERPROC.exists():
        raise FileNotFoundError(f"blenderproc not found at {BLENDERPROC}")

    asset_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mf_render_") as tmp:
        tmp = Path(tmp)
        script = _patched_script(tmp / "render_metafind.py")
        out = tmp / "views"
        out.mkdir()
        cmd = [
            str(BLENDERPROC), "run",
            "--blender-install-path", str(BLENDER_INSTALL),
            str(script),
            "--object_path", str(glb),
            "--output_dir", str(out),
            "--engine", ENGINE,
            "--num_images", str(N_VIEWS),
            "--camera_dist", str(CAMERA_DIST),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired as exc:
            # [ADDED 2026-08-24, Codex] `TimeoutExpired` propagated before the
            # failure tail below was built, so the caller's message-based
            # classifier saw only "Command ... timed out" and filed a resource
            # failure as a broken mesh. Re-raise with the word the classifier
            # can read, and with whatever the process managed to say.
            out = (exc.stdout or b"")[-400:]
            err = (exc.stderr or b"")[-400:]
            if isinstance(out, bytes):
                out, err = out.decode(errors="replace"), err.decode(errors="replace")
            raise RuntimeError(
                f"blender timed out after {timeout}s for {glb.name} -- "
                f"out of time, not out of geometry.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
            ) from exc
        produced = sorted(out.glob("[0-9][0-9][0-9].png"))
        if proc.returncode != 0 or len(produced) != N_VIEWS:
            # Upstream's `__main__` swallows exceptions and prints "Failed to
            # render", so a zero exit code does NOT mean success -- the view
            # count is the real check and is tested first-class here.
            # [FIXED 2026-08-24] Was `proc.stderr or proc.stdout`. Upstream's
            # `__main__` CATCHES its exception and prints it to STDOUT
            # (`render_single_glb.py`), while Blender writes warnings to stderr
            # on almost every asset -- so `or` discarded the actual cause
            # whenever any warning existed, and `failure_class` then classified
            # a resource failure from evidence that never mentioned it.
            tail = ("STDERR:\n" + (proc.stderr or "")[-600:]
                    + "\nSTDOUT:\n" + (proc.stdout or "")[-600:])
            raise RuntimeError(
                f"blender produced {len(produced)}/{N_VIEWS} views for {glb.name} "
                f"(exit {proc.returncode}): {tail}")

        paths = []
        for i, src in enumerate(produced):
            dst = asset_dir / f"view_{i:02d}.png"
            shutil.move(str(src), dst)
            paths.append(dst)
    return paths


def demo() -> None:
    """Render one real asset and check the properties the pipeline depends on.

    Expected truth here is structural, not aesthetic: twelve files, 512x512,
    an alpha channel that is actually used, and three view groups whose
    silhouette areas differ -- the last is what distinguishes OpenShape's three
    polar rings from a single azimuth orbit, and a renderer that quietly fell
    back to one ring would pass every other check.
    """
    import glob

    import numpy as np
    from PIL import Image

    glbs = sorted(glob.glob(
        "/home/kyzen/MetaFindV1/data/datasets/objaverse-lvis/glbs/*/*/*.glb"))
    if not glbs:
        print("demo skipped: no GLBs on this machine")
        return
    with tempfile.TemporaryDirectory() as d:
        paths = render_asset(Path(glbs[0]), Path(d) / "asset")
        assert len(paths) == N_VIEWS, len(paths)
        areas = []
        for p in paths:
            im = Image.open(p)
            assert im.size == (RESOLUTION, RESOLUTION), im.size
            assert im.mode == "RGBA", im.mode
            a = np.asarray(im)[..., 3]
            assert a.min() == 0, "no transparent pixel: film_transparent lost"
            assert a.max() == 255, "no opaque pixel: nothing was rendered"
            areas.append(float((a > 0).mean()))
        rings = [np.mean(areas[0:4]), np.mean(areas[4:8]), np.mean(areas[8:12])]
        assert len(set(round(r, 4) for r in rings)) > 1, (
            f"all three rings have the same silhouette area {rings}; the camera "
            "layout collapsed to a single elevation")
        print(f"render_blender demo ok: 12x{RESOLUTION} RGBA, "
              f"ring coverage {[round(r, 4) for r in rings]} -- OK")


if __name__ == "__main__":
    demo()
