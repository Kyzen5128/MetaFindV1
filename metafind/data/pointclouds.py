"""Sample one 10,000-point xyz+rgb cloud per Objaverse asset.

# IMPLEMENTS-NODE: n03_sample_pointclouds

The output has to be readable by a frozen ULIP-2 checkpoint that was trained on
someone else's clouds, so the format is not ours to choose. Three properties are
copied from ULIP's own loader rather than decided here:

**pc_norm applies to xyz ONLY.** ``dataset_3d.py`` line 496 normalises the
coordinates -- centroid to the origin, largest radius to 1 -- and *then*
concatenates rgb. Normalising all six columns together would rescale colour by
the object's physical extent, which is a different tensor with the same shape.

**rgb lives in [0, 1], not [0, 255].** The paper says nothing, and neither does
ULIP-2's. The decisive evidence is in ULIP's code: where a dataset has no
colour it substitutes ``np.ones_like(point_set) * 0.4`` (``dataset_3d.py`` 292
and 297) -- a mid grey. On a 0-255 scale that stand-in would be about 102. A
2.5-order-of-magnitude error here would not raise anything; it would just move
every point-cloud embedding, and the only symptom would be bad retrieval
numbers we would then go looking for elsewhere. Recorded as
``rgb_scale: "unit"`` in every sidecar so a later reader can check rather than
re-derive.

**10,000 points, xyzrgb.** ULIP-2's Appendix A.1 measures 10k xyzrgb at
50.6/79.1 on Objaverse-LVIS against 8k xyz at 48.9, and 50.6 is the abstract's
headline. The checkpoint we load is named for it.

What this deliberately does not do
----------------------------------

No farthest-point sampling. ULIP's ``Objaverse_Lvis_Colored`` reads clouds that
already hold exactly 10,000 points and draws from them with a shuffled
permutation; FPS appears in the ShapeNet path, not this one. Area-weighted
surface sampling is the closest honest reconstruction of "10,000 points from
this mesh", and U-02 records that our clouds are not ULIP's released clouds --
measured by L2-PC-ULIP-REF as a diagnostic, not a gate, because the paper never
claims MetaFind reuses them and Stage 1 trains the point encoder anyway.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths

N_POINTS = 10_000
SAMPLER_VERSION = 2  # bump whenever sampling changes; part of the cache key
RGB_SCALE = "unit"  # [0, 1]; see the module docstring
DEFAULT_GREY = 0.4  # ULIP's own stand-in for an uncoloured mesh


def uid_seed(uid: str) -> int:
    """A stable per-asset seed, so the same mesh always yields the same cloud.

    Derived from the uid rather than a counter: a counter makes the cloud depend
    on how many assets were processed before it, which turns a resumed run into
    a different dataset.
    """
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big") % (2**32)


def pc_norm(xyz: np.ndarray) -> np.ndarray:
    """ULIP's normalisation, verbatim (dataset_3d.py:496-502)."""
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt((xyz**2).sum(axis=1)))
    return xyz / m


def load_parts(path: Path):
    """Every triangulated geometry in the GLB, each with explicit colours.

    Returns ``(parts, worst_colour_source, per_part_sources)``.

    [CORRECTED TWICE] First version loaded the scene, called ``to_geometry()``
    and read colours off the merged mesh: concatenating happens before
    conversion, so every per-geometry material was gone and all 40 test assets
    came out uniformly grey. Second version converted per geometry and then
    concatenated, which kept the colours but produced a mesh whose faces index
    past the end of the merged colour array -- 14 of 40 died on an IndexError.
    Nothing is merged now. Each geometry is sampled on its own, where its face
    indices and its colour array are guaranteed to agree, and the parts are
    combined only as POINTS.
    """
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    parts, sources = [], []
    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        sources.append(_colourise(geom))
        parts.append(geom)
    if not parts:
        raise ValueError("no triangulated geometry in this GLB")
    order = ("fallback_grey", "flat", "vertex", "face", "texture")
    # The WORST source across parts, so an asset that is nine-tenths textured
    # and one-tenth grey is not labelled "texture". coloured_point_fraction
    # carries the degree, which the label alone cannot.
    return parts, min(sources, key=order.index), sources


def _colourise(geom) -> str:
    """Force an explicit per-vertex colour array onto one geometry.

    Four sources, in descending fidelity:

      texture        a baseColorTexture, sampled per vertex through the UVs
      flat           no texture, but the PBR material carries a baseColorFactor
      vertex / face  colours stored on the geometry itself
      fallback_grey  nothing readable; ULIP's own 0.4 stand-in

    [CORRECTED] `flat` was missing, and its absence cost 45% of the test batch
    its colour. trimesh's ``to_color()`` on a texture-less PBR material returns
    a vertex_colors array of length FOUR -- one RGBA, not one per vertex -- so
    the length check rejected it and fell through to grey. But
    ``baseColorFactor = [204 200 176 255]`` is a real material colour; a mesh
    that is uniformly beige is not a mesh whose colour is unknown.

    [NO BLANKET except] An earlier version wrapped this in `except Exception:
    return grey`, which is why the first run produced 40 of 40 uniformly grey
    assets: ``to_color()`` raises IndexError on some of these materials and the
    handler turned a fixable bug into a plausible-looking default. Failures are
    narrow now, and the source is RECORDED per asset, so lost colour shows up
    in the summary rather than in Table 1.
    """
    import trimesh

    vis = geom.visual
    n = len(geom.vertices)

    def _uniform(rgba, source: str) -> str:
        geom.visual = trimesh.visual.ColorVisuals(
            mesh=geom,
            vertex_colors=np.tile(np.asarray(rgba, dtype=np.uint8)[:4], (n, 1)),
        )
        return source

    if isinstance(vis, trimesh.visual.TextureVisuals):
        try:
            converted = vis.to_color()
            vc = getattr(converted, "vertex_colors", None)
            if vc is not None and len(vc) == n:
                geom.visual = converted
                return "texture"
            if vc is not None and len(vc) == 4:
                return _uniform(vc, "flat")
        except (IndexError, ValueError, TypeError, AttributeError):
            pass
        factor = getattr(getattr(vis, "material", None), "baseColorFactor", None)
        if factor is not None:
            return _uniform(np.asarray(factor).ravel(), "flat")
    else:
        vc = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc) == n:
            return "vertex"
        fc = getattr(vis, "face_colors", None)
        if fc is not None and len(fc) == len(geom.faces):
            return "face"

    return _uniform([int(DEFAULT_GREY * 255)] * 3 + [255], "fallback_grey")


def _allocate(areas: np.ndarray, n_points: int) -> np.ndarray:
    """Split n_points across parts in proportion to surface area.

    Largest-remainder, so the parts sum to exactly n_points: a cloud of 9,998
    points would pass no shape check and fail the one that matters.
    """
    if areas.sum() <= 0:
        counts = np.zeros(len(areas), dtype=int)
        counts[0] = n_points
        return counts
    exact = areas / areas.sum() * n_points
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[: n_points - counts.sum()]:
        counts[i] += 1
    return counts


def sample_mesh(path: Path, seed: int, n_points: int = N_POINTS):
    """Area-weighted surface sample with per-point colour.

    Returns ``(xyz, rgb, extents, colour_source)``. ``extents`` is the
    UNtransformed bounding box in metres: once the cloud is unit-normalised the
    real scale is gone, so F13 records it here or nowhere.
    """
    import trimesh

    parts, colour_source, sources_per_part = load_parts(path)
    areas = np.array([p.area for p in parts], dtype=np.float64)
    counts = _allocate(areas, n_points)

    xyz_chunks, rgb_chunks, coloured = [], [], 0
    for i, (part, k) in enumerate(zip(parts, counts)):
        if k == 0:
            continue
        # Per-part seed, so a part's points do not depend on how many points
        # the parts before it happened to receive.
        pts, face_idx = trimesh.sample.sample_surface(part, int(k), seed=int(seed) + i)
        tri = part.faces[face_idx]
        cols = part.visual.vertex_colors[tri][:, :, :3].mean(axis=1)
        if sources_per_part[i] != "fallback_grey":
            coloured += int(k)
        xyz_chunks.append(np.asarray(pts, dtype=np.float32))
        rgb_chunks.append(np.clip(cols.astype(np.float32) / 255.0, 0.0, 1.0))

    xyz = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    if len(xyz) != n_points:
        raise ValueError(f"sampled {len(xyz)} points, expected {n_points}")

    lo = np.min([p.vertices.min(axis=0) for p in parts], axis=0)
    hi = np.max([p.vertices.max(axis=0) for p in parts], axis=0)
    return (xyz, rgb, np.asarray(hi - lo, dtype=np.float64),
            colour_source, coloured / max(n_points, 1))


def process_one(uid: str, glb: Path, out: Path) -> dict:
    seed = uid_seed(uid)
    xyz, rgb, extents, colour_source, coloured_fraction = sample_mesh(glb, seed)
    normed = pc_norm(xyz.astype(np.float64)).astype(np.float32)

    if not np.isfinite(normed).all() or not np.isfinite(rgb).all():
        raise ValueError("non-finite values after normalisation")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write through a file HANDLE. np.savez_compressed(path) appends ".npz" to
    # any name that lacks it, so passing `<uid>.part` silently produced
    # `<uid>.part.npz` and the rename below then failed on a file that was
    # never created -- 40 of 40 assets quarantined with a FileNotFoundError
    # that pointed at the destination rather than the cause.
    tmp = out.with_name(out.name + ".part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, xyz=normed, rgb=rgb)
    tmp.replace(out)

    return {
        "uid": uid,
        "path": str(out),
        # L2-RESUME asserts preprocessing artifacts are byte-identical after a
        # kill -9 and restart. Without a digest that assertion has nothing to
        # compare, and the channel type promised one.
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_points": int(len(normed)),
        "seed": seed,
        "sampler_version": SAMPLER_VERSION,
        "rgb_scale": RGB_SCALE,
        "colour_source": colour_source,
        "coloured_point_fraction": round(coloured_fraction, 4),
        # F13: the only place the real size survives. Once xyz is unit-normed,
        # nothing downstream can recover it, and paper 2.3 wants "size
        # dimensions" in the annotations.
        "extents_m": [float(v) for v in extents],
        "volume_m3": float(np.prod(extents)),
        "centroid_offset": float(np.abs(normed.mean(axis=0)).max()),
        "max_radius": float(np.sqrt((normed**2).sum(axis=1)).max()),
        "per_axis_variance": [float(v) for v in normed.var(axis=0)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="process at most N assets (smoke runs)")
    ap.add_argument("--force", action="store_true", help="re-sample even if the file exists")
    args = ap.parse_args()

    uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
    # n02 writes shard directories (glbs/000-021/<uid>.glb), the same layout
    # Objaverse publishes. Resolve by uid once rather than guessing the shard
    # from the manifest's .npy path, which is ULIP's naming and need not agree.
    glb_by_uid = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}

    todo = []
    for uid in uids:
        glb = glb_by_uid.get(uid)
        out = paths.POINTCLOUDS / f"{uid}.npz"
        if glb is None:
            continue  # n02 has not fetched it yet; not this node's failure
        if out.exists() and not args.force:
            continue
        todo.append((uid, glb, out))
    # `--limit` truncates the WORK, not the manifest: while n02 is still
    # downloading, the first N manifest entries are mostly not on disk yet, and
    # limiting there gives an empty smoke run that looks like success.
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(uids):,} in manifest, {len(todo):,} to sample", flush=True)
    paths.POINTCLOUDS.mkdir(parents=True, exist_ok=True)
    sidecar_path = paths.LOGS / "pointclouds_index.jsonl"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    # DETERMINISTIC_INPUT -> quarantine, max_attempts 1: a non-manifold or empty
    # mesh fails identically forever, so retrying only spends time.
    quarantine, done, started = [], 0, time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool, \
            sidecar_path.open("a") as sidecar:
        futures = {pool.submit(process_one, u, g, o): u for u, g, o in todo}
        for fut in cf.as_completed(futures):
            uid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad mesh must not stop the run
                quarantine.append(
                    {"uid": uid, "node": "n03_sample_pointclouds",
                     "exception_type": type(exc).__name__,
                     "exception_msg": str(exc)[:400],
                     "traceback": traceback.format_exc()[-1500:]}
                )
                continue
            sidecar.write(json.dumps(rec) + "\n")
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                left = (len(todo) - done) / max(rate, 1e-9)
                print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                      f"剩餘約 {left:.0f} 分, quarantine {len(quarantine)}", flush=True)

    if quarantine:
        qp = paths.LOGS / "quarantine_n03.jsonl"
        with qp.open("a") as f:
            for q in quarantine:
                f.write(json.dumps(q) + "\n")

    print(f"\n{done:,} sampled, {len(quarantine):,} quarantined -> {paths.POINTCLOUDS}")
    # proceed_with_admitted: a partial corpus is a legitimate outcome here, and
    # G2_pc_sanity decides whether what survived is usable. Only an empty result
    # is a failure of this node.
    return 0 if done or not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
