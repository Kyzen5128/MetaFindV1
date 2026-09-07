"""Prepare auditable inputs for a custom same/different observation evaluation.

This is an IMPLEMENTATION CHOICE, not the paper's unspecified observation
protocol. Preparation samples meshes on CPU; the evaluator must additionally
check effective text tokens with the actual loaded backbone's tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from metafind import paths
from metafind.data import pointclouds
from metafind.data.observation import view_indices
from metafind.models.resolve_stage1 import serialize_annotation

SCHEMA = "metafind.custom_table1.v1"
N_POINTS = 10000


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _source(path: Path, *, content: bool = False) -> dict:
    path = path.absolute()
    rec = {"path": str(path), "sha256": _sha(path)}
    if content:
        rec["content"] = _json(path)
    return rec


def _uids(values, name: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty UID list")
    if any(not isinstance(u, str) or not u.strip() or u in (".", "..")
           or "/" in u or "\\" in u for u in values):
        raise ValueError(f"{name} contains an invalid UID")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate UIDs")
    return list(values)


def _pools(query_uids, gallery_uids):
    q, g = _uids(query_uids, "query_uids"), _uids(gallery_uids, "gallery_uids")
    if not set(q) <= set(g):
        raise ValueError("query_uids must be a subset of gallery_uids")
    return q, g


def _finite(a, label: str):
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected finite numeric values")


def _embedding(path: Path, n_views: int) -> int:
    with np.load(path, allow_pickle=False) as z:
        if not {"text", "image", "views"} <= set(z.files):
            raise ValueError(f"{path}: missing embedding modality")
        t, i, v = z["text"], z["image"], z["views"]
        if t.ndim != 1 or not t.size or i.shape != t.shape or v.shape != (n_views, t.size):
            raise ValueError(f"{path}: inconsistent embedding shape or view count")
        for label, a in (("text", t), ("image", i), ("views", v)):
            _finite(a, label)
            if np.any(np.linalg.norm(a.astype(np.float64), axis=-1) == 0):
                raise ValueError(f"{path}: zero {label} embedding")
        return t.size


def _cloud(path: Path, *, query: bool = False) -> np.ndarray:
    if query:
        cloud = np.load(path, allow_pickle=False)
        if cloud.shape != (N_POINTS, 6) or cloud.dtype != np.float32:
            raise ValueError(f"{path}: query cloud must be float32 ({N_POINTS}, 6)")
    else:
        with np.load(path, allow_pickle=False) as z:
            xyz, rgb = z["xyz"], z["rgb"]
            if xyz.shape != (N_POINTS, 3) or rgb.shape != xyz.shape:
                raise ValueError(f"{path}: canonical xyz/rgb must each have shape ({N_POINTS}, 3)")
            cloud = np.concatenate([xyz, rgb], axis=1)
    _finite(cloud, str(path))
    if np.any(cloud[:, 3:] < 0) or np.any(cloud[:, 3:] > 1):
        raise ValueError(f"{path}: RGB must be in [0, 1]")
    if np.max(np.linalg.norm(cloud[:, :3], axis=1)) == 0:
        raise ValueError(f"{path}: degenerate cloud")
    return cloud


def _encoding(content: dict) -> int:
    n = (content.get("view_aggregation") or {}).get("n_views")
    if type(n) is not int or n < 2:
        raise ValueError("encoding protocol requires an integer n_views >= 2")
    if (content.get("status") != "resolved"
            or content.get("actual_clip_train_scope") != "frozen"
            or content.get("image_aggregation") != "mean"):
        raise ValueError("encoding protocol must be resolved, frozen CLIP, mean aggregation")
    if not isinstance(content.get("text_template"), str) or not content["text_template"]:
        raise ValueError("encoding protocol lacks text_template")
    if not content.get("text_serialization"):
        raise ValueError("encoding protocol lacks text_serialization")
    return n


def _record_content(uid: str, rec: dict, enc: dict, n_views: int, *, is_query: bool):
    for key, suffix in (("embedding", ".npz"), ("cloud", ".npz"),
                        ("annotation", ".json"), ("embedding_sidecar", ".json"),
                        ("mesh", ".glb")):
        if Path(rec[key]["path"]).name != uid + suffix:
            raise ValueError(f"{uid}: {key} belongs to a different UID")
    sidecar = _json(Path(rec["embedding_sidecar"]["path"]))
    ann = _json(Path(rec["annotation"]["path"]))
    if sidecar.get("uid") != uid or ann.get("uid", uid) != uid:
        raise ValueError(f"{uid}: source UID mismatch")
    if sidecar != rec["embedding_sidecar"]["content"]:
        raise ValueError(f"{uid}: embedding sidecar content mismatch")
    if (sidecar.get("n_views") != n_views or sidecar.get("aggregation") != "mean"
            or sidecar.get("text_serialization") != enc["text_serialization"]):
        raise ValueError(f"{uid}: embedding sidecar disagrees with encoding protocol")
    canonical = serialize_annotation(ann, template=enc["text_template"])
    if not canonical.strip() or sidecar.get("text") != canonical or rec.get("canonical_text") != canonical:
        raise ValueError(f"{uid}: canonical text differs from annotation/serializer/sidecar")
    if rec.get("query_view") != view_indices("held_out_view", uid, n_views)[0]:
        raise ValueError(f"{uid}: query_view differs from UID rule")
    if rec.get("gallery_views") != view_indices("disjoint_views", uid, n_views):
        raise ValueError(f"{uid}: gallery_views must exclude exactly its query view")
    dim = _embedding(Path(rec["embedding"]["path"]), n_views)
    cloud = _cloud(Path(rec["cloud"]["path"]))
    if "cloud_sidecar" in rec:
        sc = _json(Path(rec["cloud_sidecar"]["path"]))
        if sc != rec["cloud_sidecar"]["content"] or sc.get("uid") != uid:
            raise ValueError(f"{uid}: cloud sidecar UID/content mismatch")
        if sc.get("sha256") != rec["cloud"]["sha256"]:
            raise ValueError(f"{uid}: cloud sidecar digest mismatch")
    if is_query:
        text = rec.get("query_text")
        if not isinstance(text, str) or not text.strip() or text.strip() == canonical.strip():
            raise ValueError(f"{uid}: query_text must be nonempty and different from canonical text")
        if "query_cloud" in rec:
            qpath = Path(rec["query_cloud"]["path"])
            if qpath.name != uid + ".npy":
                raise ValueError(f"{uid}: query_cloud belongs to a different UID")
            qcloud = _cloud(qpath, query=True)
            if qcloud.shape == cloud.shape and np.array_equal(qcloud, cloud):
                raise ValueError(f"{uid}: resampled cloud equals canonical cloud")
    elif "query_text" in rec or "query_cloud" in rec:
        raise ValueError(f"{uid}: gallery-only asset has query observations")
    return dim


def prepare_protocol(query_uids, gallery_uids, out_dir: Path, *,
                     query_texts: dict[str, str], provenance: dict,
                     seed_offset=1000003) -> Path:
    """Preflight all inputs, then write a new, isolated CPU preparation directory.

    A failed preparation retains partial output for inspection, with no complete
    manifest; existing destinations are never resumed or overwritten.
    """
    q, g = _pools(query_uids, gallery_uids)
    if not isinstance(query_texts, dict) or set(query_texts) != set(q):
        raise ValueError("query_texts must cover exactly query_uids")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    if type(seed_offset) is not int or seed_offset <= 0:
        raise ValueError("seed_offset must be a positive integer")
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {out_dir}")
    for root in (paths.EMBEDDINGS, paths.POINTCLOUDS, paths.ANNOTATIONS, paths.OBJAVERSE_GLB):
        if out_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"output must not be inside canonical inputs: {out_dir}")
    encoding = _source(paths.OUTPUTS / "stage1_encoding_protocol.json", content=True)
    enc = encoding["content"]
    n_views = _encoding(enc)
    meshes = {}
    wanted = set(g)
    for p in paths.OBJAVERSE_GLB.rglob("*.glb"):
        if p.stem in wanted:
            if p.stem in meshes:
                raise ValueError(f"duplicate mesh UID: {p.stem}")
            meshes[p.stem] = p
    if wanted - set(meshes):
        raise FileNotFoundError(f"missing mesh UIDs: {sorted(wanted - set(meshes))[:5]}")
    records, dims = {}, set()
    print(f"preflight canonical sources: 0/{len(g)}", flush=True)
    for index, uid in enumerate(g, 1):
        sidecar = _source(paths.EMBEDDINGS / f"{uid}.json", content=True)
        rec = {"embedding": _source(paths.EMBEDDINGS / f"{uid}.npz"),
               "cloud": _source(paths.POINTCLOUDS / f"{uid}.npz"),
               "annotation": _source(paths.ANNOTATIONS / f"{uid}.json"),
               "mesh": _source(meshes[uid]), "embedding_sidecar": sidecar,
               "canonical_text": sidecar["content"].get("text"),
               "query_view": view_indices("held_out_view", uid, n_views)[0],
               "gallery_views": view_indices("disjoint_views", uid, n_views),
               "canonical_cloud_provenance": "UNKNOWN unless recorded in cloud_sidecar; source mesh identity may be absent"}
        sc = paths.POINTCLOUDS / f"{uid}.json"
        if sc.exists():
            rec["cloud_sidecar"] = _source(sc, content=True)
        if uid in query_texts:
            rec["query_text"] = query_texts[uid]
        dims.add(_record_content(uid, rec, enc, n_views, is_query=uid in query_texts))
        records[uid] = rec
        if index % 100 == 0 or index == len(g):
            print(f"preflight canonical sources: {index}/{len(g)}", flush=True)
    if len(dims) != 1:
        raise ValueError("mixed embedding dimensions across gallery")
    manifest = {"schema": SCHEMA, "status": "complete", "classification": "IMPLEMENTATION CHOICE",
                "query_uids": q, "gallery_uids": g,
                "data_root": str(paths.DATA.resolve()), "n_views": n_views,
                "encoding_protocol": encoding, "provenance": provenance,
                "choices": {"positive": "same_uid", "same_observations": "canonical text, cached mean image, canonical cloud on both sides",
                            "different_observations": "external query text, one UID-fixed query view, remaining views for gallery, resampled query cloud",
                            "query_view_rule": "uid_seed(uid) % n_views",
                            "gallery_view_rule": "all indices except each gallery UID's own query_view",
                            "subset_image_aggregation": "float32 arithmetic mean of cached view vectors; no per-view normalization",
                            "query_pc": {"seed_offset": seed_offset, "n_points": N_POINTS,
                                         "sampler_version": pointclouds.SAMPLER_VERSION,
                                         "sampler_source": _source(Path(pointclouds.__file__)),
                                         "meshload_source": _source(Path(pointclouds.meshload.__file__)),
                                         "normalization": "pointclouds.pc_norm(float64 xyz) -> float32; RGB unchanged"},
                            "effective_text_tokens": "runtime_required: compare canonical/query tokens using the loaded backbone tokenizer; reject equality",
                            "caveat": "A new surface sample does not prove that only the seed differs from historical canonical sampling."},
                "records": records}
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "query_pc").mkdir()
    print(f"sample query clouds: 0/{len(q)}", flush=True)
    for index, uid in enumerate(q, 1):
        rec = records[uid]
        xyz, rgb, *_ = pointclouds.sample_mesh(Path(rec["mesh"]["path"]),
                                             pointclouds.uid_seed(uid) + seed_offset, N_POINTS)
        cloud = np.concatenate([pointclouds.pc_norm(np.asarray(xyz, dtype=np.float64)).astype(np.float32),
                                np.asarray(rgb, dtype=np.float32)], axis=1)
        dest = out_dir / "query_pc" / f"{uid}.npy"
        with dest.open("xb") as fh:
            np.save(fh, cloud, allow_pickle=False)
        rec["query_cloud"] = _source(dest)
        if index % 100 == 0 or index == len(q):
            print(f"sample query clouds: {index}/{len(q)}", flush=True)
    # Recheck source digests after sampling: mutable annotations/caches cannot
    # quietly turn a preflight snapshot into a different artifact.
    print("verifying source hashes and prepared observations before publication", flush=True)
    _validate(manifest, verify=True)
    dest = out_dir / "protocol.json"
    part = dest.with_suffix(".json.part")
    with part.open("x") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    # link+unlink is an atomic, no-replace publication on this same filesystem.
    # Path.rename/replace would overwrite a destination created during sampling.
    os.link(part, dest)
    part.unlink()
    return dest


def _validate(manifest: dict, *, verify: bool):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported custom protocol schema")
    if manifest.get("status") != "complete":
        raise ValueError("custom protocol is not complete")
    pc_choice = (manifest.get("choices") or {}).get("query_pc") or {}
    if (pc_choice.get("n_points") != N_POINTS
            or type(pc_choice.get("seed_offset")) is not int or pc_choice["seed_offset"] <= 0
            or type(pc_choice.get("sampler_version")) is not int
            or pc_choice["sampler_version"] <= 0
            or not pc_choice.get("sampler_source") or not pc_choice.get("meshload_source")):
        raise ValueError("custom protocol lacks complete query sampler identity")
    q, g = _pools(manifest.get("query_uids"), manifest.get("gallery_uids"))
    if set(manifest.get("records", {})) != set(g):
        raise ValueError("protocol records must cover exactly gallery_uids")
    n = _encoding(manifest["encoding_protocol"]["content"])
    if manifest.get("n_views") != n:
        raise ValueError("manifest n_views disagrees with encoding protocol")
    if not verify:
        return
    if pc_choice["sampler_version"] != pointclouds.SAMPLER_VERSION:
        raise ValueError("query sampler version differs from the source implementation being verified")
    query_set = set(q)
    sources = [manifest["encoding_protocol"], pc_choice["sampler_source"], pc_choice["meshload_source"]]
    input_files = manifest.get("provenance", {}).get("input_files", {})
    if not isinstance(input_files, dict):
        raise ValueError("provenance.input_files must be a mapping of file identities")
    sources.extend(input_files.values())
    for uid, rec in manifest["records"].items():
        for key in ("embedding", "cloud", "annotation", "embedding_sidecar", "mesh", "cloud_sidecar", "query_cloud"):
            if key in rec:
                sources.append(rec[key])
        if uid in query_set and "query_cloud" not in rec:
            raise ValueError(f"{uid}: missing query_cloud")
    for source in sources:
        path = Path(source["path"])
        if _sha(path) != source["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        if "content" in source and _json(path) != source["content"]:
            raise ValueError(f"recorded JSON content mismatch: {path}")
    # CLI file provenance binds both bytes and their interpretation as ordered
    # pools / explicit text observations, not just a file alongside the run.
    expected_inputs = {"query_uids": q, "gallery_uids": g,
                       "query_texts": {u: manifest["records"][u].get("query_text") for u in q}}
    for key, expected in expected_inputs.items():
        if key in input_files and json.loads(Path(input_files[key]["path"]).read_text()) != expected:
            raise ValueError(f"protocol differs from provenance input file: {key}")
    dims = {_record_content(u, manifest["records"][u], manifest["encoding_protocol"]["content"],
                            n, is_query=u in query_set) for u in g}
    if len(dims) != 1:
        raise ValueError("mixed embedding dimensions across gallery")


def load_protocol(path, verify=True) -> dict:
    """Load and, by default, verify every bound source and prepared cloud.

    Effective token inequality is a separate mandatory runner check: no model
    or tokenizer is loaded here and distinct strings alone do not prove it.
    """
    manifest = _json(Path(path))
    _validate(manifest, verify=verify)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--query-uids", required=True, type=Path)
    ap.add_argument("--gallery-uids", required=True, type=Path)
    ap.add_argument("--query-texts", required=True, type=Path)
    ap.add_argument("--provenance", type=Path, help="JSON object recording split/data identity and text source")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed-offset", type=int, default=1000003)
    args = ap.parse_args(argv)
    provenance = _json(args.provenance) if args.provenance else {}
    # Bind CLI input files as evidence, not merely their mutable path names.
    provenance["input_files"] = {key: _source(getattr(args, key)) for key in
                                 ("query_uids", "gallery_uids", "query_texts")}
    if args.provenance:
        provenance["input_files"]["provenance"] = _source(args.provenance)
    result = prepare_protocol(json.loads(args.query_uids.read_text()),
                              json.loads(args.gallery_uids.read_text()), args.out_dir,
                              query_texts=_json(args.query_texts), provenance=provenance,
                              seed_offset=args.seed_offset)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
