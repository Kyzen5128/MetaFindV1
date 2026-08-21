"""Sample one 10,000-point xyz+rgb cloud per Objaverse asset.

# IMPLEMENTS-NODE: n03_sample_pointclouds

The output has to be readable by a frozen ULIP-2 checkpoint that was trained on
someone else's clouds, so the format is not ours to choose. Three properties are
copied from ULIP's own loader rather than decided here:

**pc_norm applies to xyz ONLY.** ``dataset_3d.py`` line 496 normalises the
coordinates -- centroid to the origin, largest radius to 1 -- and *then*
concatenates rgb. Normalising all six columns together would rescale colour by
the object's physical extent, which is a different tensor with the same shape.

**rgb lives in [0, 1], not [0, 255] -- STRONGLY INDICATED, not proven.**
Neither paper says. The evidence is that ULIP substitutes
``np.ones_like(point_set) * 0.4`` (``dataset_3d.py`` 292 and 297) where a
dataset has no colour, and on a 0-255 scale that stand-in would be about 102.

But note exactly what that shows and what it does not. Those two lines are in
the *ModelNet* path. ``Objaverse_Lvis_Colored.__getitem__`` reads ``rgb`` out
of the released .npy and concatenates it with no division and no clamp, so it
inherits whatever scale the released file already uses. The 0.4 fallback is
evidence about ULIP's colour CONVENTION, not a measurement of the Objaverse
release. Settling it needs one official .npy read directly -- that measurement
is tracked with U-02 and the result belongs here.

Getting it wrong is silent and costs a factor of 255 on the colour channel of
every asset, so ``rgb_scale: "unit"`` goes in every sidecar: a later reader can
check what was assumed instead of re-deriving it.

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
import collections
import concurrent.futures as cf
import hashlib
import json
import sys
import os
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog

NODE = "n03_sample_pointclouds"
N_POINTS = 10_000
# 3: apply the scene-graph transform before sampling. Measured on 400 random
# assets, 65.8% carry a non-identity transform, and on 40 re-sampled assets 16
# changed extent by more than 1% -- ratios from 0.008x to 77x. Versions 1-2
# produced geometrically wrong clouds for a large fraction of the corpus while
# passing every G2 check.
SAMPLER_VERSION = 3
RGB_SCALE = "unit"  # [0, 1]; see the module docstring
DEFAULT_GREY = 0.4  # ULIP's stand-in for a dataset with no colour channel at all
# glTF 2.0 specifies pbrMetallicRoughness.baseColorFactor default = [1,1,1,1].
# A PBR material with neither a texture nor an explicit factor is therefore
# WHITE, not unknown. trimesh reports 102/255 = 0.4 grey as ITS default in that
# case, which is numerically identical to DEFAULT_GREY and so looked like a
# legitimate fallback. Measured against ULIP's released cloud for
# 1dc0fe17c77e...: theirs is 1.000, ours was 0.400.
GLTF_DEFAULT_BASE_COLOR = 1.0


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

    from metafind.data import meshload

    # The frame correction and the COLOR_0 recovery both live in `meshload` so
    # that n03 and n04 cannot drift apart -- see that module's docstring. A
    # cloud and a render in different frames raise nothing anywhere.
    scene = meshload.load_scene(path)
    color0 = meshload.color0_by_geometry(path)
    parts, sources = [], []
    # Iterate the scene GRAPH, not scene.geometry. Two reasons, both measured:
    #
    #   * 65.8% of 400 sampled Objaverse GLBs place their geometry with a
    #     non-identity node transform, translations up to 1.2e4. Sampling the
    #     raw geometry drops it and assembles the object collapsed on itself.
    #     On 40 re-sampled assets, 16 changed extent by over 1%, up to 77x.
    #   * One geometry can be INSTANCED at several nodes -- four identical
    #     chairs around a table are one mesh and four transforms, and looping
    #     over scene.geometry would see one chair. NOT observed in this corpus:
    #     0 of 300 sampled assets instance anything. Iterating the graph gets
    #     it for free, so it is handled, but it is not a measured problem here
    #     and must not be reported as one.
    #
    # Either way the cloud keeps 10,000 points, normalises to a unit sphere and
    # passes every G2 check. It is simply the wrong object.
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        geom.apply_transform(transform)
        # Keyed by geometry NAME, and the name is produced by the same trimesh
        # loader in both passes. Instanced geometry shares one COLOR_0 array,
        # which is correct: the colours belong to the mesh, not to the node.
        sources.append(_colourise(geom, color0.get(geom_name)))
        parts.append(geom)

    if not parts:
        raise ValueError("no triangulated geometry in this GLB")
    order = ("fallback_grey", "gltf_default", "flat", "vertex", "face", "texture")
    # The WORST source across parts, so an asset that is nine-tenths textured
    # and one-tenth grey is not labelled "texture". coloured_point_fraction
    # carries the degree, which the label alone cannot.
    return parts, min(sources, key=order.index), sources


def _colourise(geom, color0: np.ndarray | None = None) -> str:
    """Force an explicit per-vertex colour array onto one geometry.

    ``color0`` is this geometry's glTF ``COLOR_0`` attribute when it declares
    one, recovered by ``meshload.color0_by_geometry`` because **trimesh 5.0.0
    drops it whenever the primitive also carries a material**. It is consulted
    only where the material path would otherwise produce ``flat`` or
    ``gltf_default`` -- a real ``baseColorTexture`` still wins, which is what
    the ordering below already says. Measured: `COLOR_0` accounts for the whole
    of our colour disagreement with ULIP's released clouds, and for nothing else
    (`HANDOFF.md`, `F-N03-1`).

    Four sources, in descending fidelity:

      texture        a baseColorTexture, sampled per vertex through the UVs
      flat           no texture, but the PBR material carries a baseColorFactor
      gltf_default   a material with neither: glTF 2.0 defines baseColorFactor
                     as [1,1,1,1], so the object is white
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
            # `len(vc) == 4` was the test here and it is ambiguous: a single
            # RGBA has length 4, and so does a four-vertex mesh's per-vertex
            # array. `ndim == 1` says what was meant. The check above
            # (`len(vc) == n`) already catches the four-vertex case first, so
            # this is tightening a latent ambiguity rather than the fix for the
            # 8.1% failure -- that fix is _vertex_rgb.
            if vc is not None and np.asarray(vc).ndim == 1:
                return _uniform(vc, "flat")
        except (IndexError, ValueError, TypeError, AttributeError):
            pass
        # [F-N03-1] Below `texture`, above `flat` -- exactly where the source
        # ordering above already places `vertex`. Reached only because trimesh
        # withheld the attribute from the main load; the branch further down
        # that reads `vis.vertex_colors` is the same decision for the assets
        # trimesh does hand it over for.
        if color0 is not None and len(color0) == n:
            geom.visual = trimesh.visual.ColorVisuals(
                mesh=geom, vertex_colors=np.asarray(color0, dtype=np.uint8)
            )
            return "vertex"
        mat = getattr(vis, "material", None)
        factor = getattr(mat, "baseColorFactor", None)
        if factor is not None:
            return _uniform(np.asarray(factor).ravel(), "flat")
        if mat is not None:
            # Material present, no texture, no explicit factor -> glTF's default
            # white. NOT trimesh's main_color, which is its own grey stand-in and
            # happens to equal DEFAULT_GREY, so trusting it silently replaced a
            # white object with a grey one.
            return _uniform([int(GLTF_DEFAULT_BASE_COLOR * 255)] * 3 + [255],
                            "gltf_default")
    else:
        vc = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc) == n:
            return "vertex"
        fc = getattr(vis, "face_colors", None)
        if fc is not None and len(fc) == len(geom.faces):
            return "face"

    return _uniform([int(DEFAULT_GREY * 255)] * 3 + [255], "fallback_grey")


def _vertex_rgb(geom) -> np.ndarray:
    """``(n_vertices, 3)`` uint8 colours, whatever shape trimesh handed back.

    trimesh COLLAPSES a uniform vertex_colors array to a single RGBA, so a mesh
    whose vertices all share one colour returns shape (4,) rather than (n, 4).
    Indexing that by a face array raises IndexError, and the asset is
    quarantined as though its geometry were broken.

    Missed entirely by a 60-asset smoke run -- it needs a small uniform-coloured
    part, and it cost 8.1% of an 800-asset sample, every one of them with the
    same message. Four times G3's quarantine ceiling, from one shape assumption.
    """
    vc = np.asarray(geom.visual.vertex_colors)
    n = len(geom.vertices)
    if vc.ndim == 1:  # one RGBA for the whole geometry
        return np.tile(vc[:3], (n, 1))
    if vc.shape[0] != n:  # per-face, or otherwise not per-vertex
        return np.tile(vc.reshape(-1, vc.shape[-1])[0, :3], (n, 1))
    return vc[:, :3]


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
        cols = _vertex_rgb(part)[tri].mean(axis=1)
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


def sidecar_path(out: Path) -> Path:
    return out.with_suffix(".json")


def is_complete(out: Path) -> bool:
    """Whether this asset is finished, in the only sense that survives a crash.

    A present .npz proves nothing. The old check was `out.exists()`, and the
    write order was: rename the npz, return, and let the main thread append to
    a shared JSONL. Kill -9 between those two and the asset is permanently
    skipped on restart with its canonical metadata missing -- measured, not
    hypothetical: interrupting the first full run left 18 clouds with no
    record. G2 routes on centroid_offset, max_radius and per_axis_variance,
    which live only in that record.

    The per-asset sidecar is therefore the completion marker and is written
    LAST, after the npz is already in place. Crashing between them costs one
    re-sample, and re-sampling is deterministic (uid_seed), so the recovered
    cloud is byte-identical to the lost one.
    """
    sc = sidecar_path(out)
    if not (out.exists() and sc.exists()):
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # Also catches a truncated npz: a half-written file has a different digest.
    return rec.get("sha256") == hashlib.sha256(out.read_bytes()).hexdigest()


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
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out)

    record = {
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
        # [RENAMED from extents_m / volume_m3] This is the axis-aligned
        # BOUNDING BOX in the GLB's own units, not the mesh volume and not a
        # verified physical size: Objaverse authors set their own scale, and
        # nothing here checks that a metre is a metre. The old names asserted
        # both. It still matters -- once xyz is unit-normed nothing downstream
        # can recover any scale at all, and paper 2.3 wants "size dimensions"
        # in the annotations -- but it is a weak ground truth and must not be
        # used to score the annotator as if it were a strong one.
        "raw_bbox_extents": [float(v) for v in extents],
        "raw_bbox_volume": float(np.prod(extents)),
        # float64 for the STATISTICS, even though the cloud is stored float32.
        # Averaging 10,000 float32 values accumulates about 1e-5 of error, which
        # is larger than G2's 1e-5 centroid tolerance -- so 8 of 46,052 assets
        # were recorded as failing a check their data passes. Verified on one:
        # recomputed in float64 the centroid offset is 5.2e-09, not 1.15e-05.
        # The gate's threshold is not the problem and must not be widened; the
        # measurement was reporting the summation method rather than the cloud.
        "centroid_offset": float(np.abs(normed.astype(np.float64).mean(axis=0)).max()),
        "max_radius": float(np.sqrt((normed.astype(np.float64) ** 2).sum(axis=1)).max()),
        "per_axis_variance": [float(v) for v in normed.astype(np.float64).var(axis=0)],
    }

    # The completion marker, written last and atomically. Order matters: npz
    # first, sidecar second, so a sidecar can never describe a file that is
    # not there.
    sc_tmp = sidecar_path(out).with_suffix(".json.part")
    with sc_tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sc_tmp.replace(sidecar_path(out))
    return record


def rebuild_index(index_path: Path) -> int:
    """Regenerate pointclouds_index.jsonl from the per-asset sidecars.

    Derived, so it is always exactly what is on disk. The previous version was
    appended to as work completed, which meant a resumed run duplicated rows
    and an interrupted one left the index short -- and the index is what a
    reader treats as the record of the corpus.
    """
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as f:
        for sc in sorted(paths.POINTCLOUDS.glob("*.json")):
            try:
                f.write(json.dumps(json.loads(sc.read_text())) + "\n")
                n += 1
            except (OSError, json.JSONDecodeError):
                continue
    tmp.replace(index_path)
    return n


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
        if is_complete(out) and not args.force:
            continue
        todo.append((uid, glb, out))
    # `--limit` truncates the WORK, not the manifest: while n02 is still
    # downloading, the first N manifest entries are mostly not on disk yet, and
    # limiting there gives an empty smoke run that looks like success.
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(uids):,} in manifest, {len(todo):,} to sample", flush=True)
    paths.POINTCLOUDS.mkdir(parents=True, exist_ok=True)
    index_path = paths.LOGS / "pointclouds_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # DETERMINISTIC_INPUT -> quarantine, max_attempts 1: a non-manifold or empty
    # mesh fails identically forever, so retrying only spends time.
    quarantine, done, started = [], 0, time.time()
    with runlog.run_progress(NODE), \
            cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, u, g, o): u for u, g, o in todo}
        for fut in cf.as_completed(futures):
            uid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad mesh must not stop the run
                # The registry declares two classes with DIFFERENT policies:
                # DETERMINISTIC_INPUT quarantines after one attempt because a
                # non-manifold mesh fails identically forever, while RESOURCE
                # degrades and retries. Collapsing both into "quarantine" threw
                # away recoverable assets on a transient memory spike and
                # reported them as broken geometry.
                cls = ("RESOURCE" if isinstance(exc, (MemoryError, OSError))
                       else "DETERMINISTIC_INPUT")
                # Written NOW, not collected and flushed at the end. A 70-minute
                # run gave no view of what it was discarding until it finished,
                # and a crash lost every record because they lived only in a
                # list -- the same defect as buffering run_progress.
                runlog.quarantine(NODE, [{
                    "uid": uid, "failure_class": cls,
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                quarantine.append(cls)
                continue
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                left = (len(todo) - done) / max(rate, 1e-9)
                print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                      f"剩餘約 {left:.0f} 分, quarantine {len(quarantine)}", flush=True)

    # The aggregate index is DERIVED from the per-asset sidecars, never
    # appended to. Appending made it grow duplicates on every resumed run and
    # made it disagree with the filesystem the moment one was interrupted.
    n_indexed = rebuild_index(index_path)
    runlog.cost_ledger(
        cpu_seconds=round(time.time() - started, 1),
        assets_sampled=done,
        bytes_written=sum(
            (paths.POINTCLOUDS / f"{u}.npz").stat().st_size
            for u, _, o in todo if o.exists()
        ),
    )

    by_class = collections.Counter(quarantine)
    print(f"\n{done:,} sampled this run, {n_indexed:,} complete on disk, "
          f"{len(quarantine):,} quarantined {dict(by_class)} -> {paths.POINTCLOUDS}")
    # proceed_with_admitted: a partial corpus is a legitimate outcome here, and
    # G2_pc_sanity decides whether what survived is usable. Only an empty result
    # is a failure of this node.
    return 0 if done or not todo else 2


if __name__ == "__main__":
    raise SystemExit(main())
