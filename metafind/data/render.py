"""Render each asset from 11 viewpoints (paper sec. 2.3).

    "Each asset is rendered from 11 orthogonal viewpoints and annotated using
    GPT-4o."

What "orthogonal" has to mean
-----------------------------

It cannot mean eleven mutually orthogonal directions: three-dimensional space
admits at most three orthogonal axes, so six directions. It therefore means
**orthographic projection**, which is also the sensible choice for asset
thumbnails -- perspective would make the apparent size depend on camera
distance, so identical objects framed differently would encode differently.

Where the cameras go is not stated (U-17)
-----------------------------------------

The paper gives the count and the projection but no placement, and 11 matches no
standard configuration: a cube has 6 faces, an icosahedron 12 vertices. ULIP's
own convention is unrelated -- ``dataset_3d.py`` uses ``range(0, 360, 12)``,
i.e. 30 azimuths -- so 11 is MetaFind's own choice and is simply unspecified.

``fibonacci`` is the default because a Fibonacci lattice gives the most uniform
coverage of the view sphere for *any* N, including awkward ones like 11, and is
deterministic. ``axis_aligned`` offers the alternative reading, six axis
directions plus five corners. The choice is recorded in every sidecar so results
stay attributable to it.

Framing
-------

Each mesh is centred and scaled to a unit bounding sphere before rendering, so
the camera setup is identical for every asset. Without it, an object modelled in
millimetres and one modelled in metres would produce wildly different images and
the image tower would be learning modelling conventions rather than shape.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

# pyrender picks its GL backend at import time, so this must be set first.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

__all__ = ["RenderConfig", "view_directions", "camera_poses", "render_asset", "look_at"]

Layout = Literal["fibonacci", "axis_aligned"]


@dataclass
class RenderConfig:
    """Rendering settings.

    Attributes:
        n_views: viewpoints per asset. The paper says 11.
        layout: camera placement. See U-17 -- unspecified by the paper.
        resolution: output edge length. ULIP-2's released renders are 224, and
            the image tower consumes 224, so anything larger is discarded later.
        margin: fraction of padding around the unit sphere, so silhouettes are
            not clipped at the frame edge.
        bg: background colour. White matches ULIP's renders.
        ambient: ambient light level, enough that faces pointing away from the
            key light are not pure black.
        seed: reserved for future jittered layouts; the defaults are fully
            deterministic.
    """

    n_views: int = 11
    layout: Layout = "fibonacci"
    resolution: int = 224
    margin: float = 0.15
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient: float = 0.5
    seed: int = 0


def view_directions(n: int, layout: Layout = "fibonacci") -> np.ndarray:
    """Unit vectors pointing from the object toward each camera.

    Args:
        n: number of viewpoints.
        layout: ``"fibonacci"`` for near-uniform sphere coverage, or
            ``"axis_aligned"`` for the six axis directions plus cube corners.

    Returns:
        ``(n, 3)`` float64 unit vectors.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if layout == "axis_aligned":
        base = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]
        corners = [
            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1),
            (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
        ]
        dirs = (base + corners)[:n]
        if len(dirs) < n:
            raise ValueError(f"axis_aligned supports at most {len(base) + len(corners)} views")
        v = np.array(dirs, dtype=np.float64)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # Fibonacci lattice: evenly spaced in z, golden-angle rotation in azimuth.
    # Endpoints are offset by half a step so no camera lands exactly on a pole,
    # where the up-vector becomes degenerate.
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Camera-to-world matrix for a camera at ``eye`` looking at ``target``.

    Uses OpenGL convention: the camera looks down its own -Z.
    """
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    forward = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("eye and target coincide")
    forward /= norm

    # Near a pole the chosen up-vector becomes parallel to forward and the cross
    # product collapses; swap to a different axis rather than emit a NaN pose.
    if abs(float(np.dot(forward, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def camera_poses(cfg: RenderConfig) -> np.ndarray:
    """``(n_views, 4, 4)`` camera-to-world matrices on a unit-radius sphere."""
    dirs = view_directions(cfg.n_views, cfg.layout)
    origin = np.zeros(3)
    # Orthographic projection ignores distance, so any radius outside the unit
    # sphere works; 3 keeps the whole object in front of the near plane.
    return np.stack([look_at(d * 3.0, origin) for d in dirs])


def normalize_mesh(mesh) -> "object":
    """Centre a mesh at the origin and scale it into the unit sphere."""
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("scene contains no geometry")
        mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)

    if mesh.vertices.shape[0] == 0:
        raise ValueError("mesh has no vertices")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh has non-finite vertices")

    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
    if not np.isfinite(radius) or radius == 0:
        raise ValueError("degenerate mesh: all vertices coincide")
    mesh.vertices /= radius
    return mesh


def render_asset(mesh_or_path, cfg: RenderConfig | None = None) -> np.ndarray:
    """Render one asset from every viewpoint.

    Args:
        mesh_or_path: a trimesh object, or a path to a mesh file (GLB, OBJ, ...).
        cfg: rendering settings.

    Returns:
        ``(n_views, resolution, resolution, 3)`` uint8 RGB.
    """
    import pyrender
    import trimesh

    cfg = cfg or RenderConfig()
    mesh = trimesh.load(mesh_or_path, force="mesh") if isinstance(mesh_or_path, (str, os.PathLike)) else mesh_or_path
    mesh = normalize_mesh(mesh)

    scene = pyrender.Scene(bg_color=(*cfg.bg, 1.0), ambient_light=(cfg.ambient,) * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    mag = 1.0 + cfg.margin
    camera = pyrender.OrthographicCamera(xmag=mag, ymag=mag)
    light = pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=3.0)
    cam_node = scene.add(camera, pose=np.eye(4))
    # The light rides with the camera, so every view is lit from its own side.
    # A fixed light would leave some viewpoints looking at an unlit surface.
    light_node = scene.add(light, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(cfg.resolution, cfg.resolution)
    try:
        out = np.empty((cfg.n_views, cfg.resolution, cfg.resolution, 3), dtype=np.uint8)
        for i, pose in enumerate(camera_poses(cfg)):
            scene.set_pose(cam_node, pose)
            scene.set_pose(light_node, pose)
            color, _ = renderer.render(scene)
            out[i] = color[..., :3]
    finally:
        renderer.delete()
    return out
