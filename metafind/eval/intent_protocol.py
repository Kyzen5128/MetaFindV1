"""Freeze explicit furniture-intent queries and condition-specific judgments.

This is a local evaluation protocol, not the author's unpublished Table 1
protocol. No query, judgment, point sample, embedding, or model is generated.
Raw query files remain unchanged; the runner normalizes query XYZ with pc_norm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

from metafind import paths
from metafind.eval import custom_protocol as canonical
from metafind.eval.retrieval import QUERY_CONDITIONS

SCHEMA = "metafind.intent_retrieval.v1"
SPEC_SCHEMA = "metafind.intent_retrieval.spec.v1"
CHOICES = {
    "positive": "explicit binary qrels for each query condition; no target UID inference",
    "judgment_coverage": "every gallery UID judged for every requested condition",
    "gallery_modalities": ["text", "image", "pc"],
    "gallery_image_aggregation": "float32 arithmetic mean of all cached view vectors",
    "query_image_aggregation": "arithmetic mean of images encoded by the restored backbone",
    "query_pc_normalization": "pc_norm(float64 xyz) -> float32; RGB unchanged",
    "query_pc_shape": [10000, 6],
    "unjudged_policy": "reject; never infer negative relevance",
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes) -> dict:
    def nonfinite(value):
        raise ValueError(f"non-finite JSON value: {value}")
    result = json.loads(raw, object_pairs_hook=_object, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    # Also reject overflow such as 1e999, which parse_constant does not handle.
    json.dumps(result, allow_nan=False)
    return result


def _keys(obj, required, optional=(), *, label):
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be an object")
    if not set(required) <= set(obj) or set(obj) - set(required) - set(optional):
        raise ValueError(f"{label}: missing or unsupported fields")


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _ids(values, label):
    result = canonical._uids(values, label)
    if any(value != value.strip() for value in result):
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return result


def _source_shape(source):
    _keys(source, ("path", "sha256"), ("content",), label="source")
    _nonempty(source["path"], "source path")
    if not Path(source["path"]).is_absolute():
        raise ValueError("frozen source paths must be absolute")
    digest = source["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("source sha256 must be a lowercase SHA256 digest")


def verified_source_bytes(source: dict) -> bytes:
    """Return the same bytes whose digest was checked, for runner decoding."""
    _source_shape(source)
    raw = Path(source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"SHA256 mismatch: {source['path']}")
    return raw


def _capture(path: Path, *, content=False):
    path = path.resolve()
    raw = path.read_bytes()
    source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if content:
        source["content"] = _json_bytes(raw)
    return source


def _source_json(source, *, verify):
    _source_shape(source)
    if "content" not in source or not isinstance(source["content"], dict):
        raise ValueError("JSON source must preserve its literal content")
    if verify:
        observed = _json_bytes(verified_source_bytes(source))
        if observed != source["content"]:
            raise ValueError(f"recorded JSON content mismatch: {source['path']}")
    return source["content"]


def _metadata(obj, fields, label):
    if not isinstance(obj, dict) or not set(fields) <= set(obj):
        raise ValueError(f"{label}: missing provenance fields")
    for field in fields:
        _nonempty(obj[field], f"{label}.{field}")


def _query(query, gallery, *, frozen):
    _keys(query, ("query_id", "conditions", "provenance", "qrels", "judgments"),
          ("text", "images", "pointcloud"), label="query")
    _ids([query["query_id"]], "query_id")
    _metadata(query["provenance"], ("source", "creator", "method"), "query provenance")
    if "text" in query:
        _nonempty(query["text"], "query text")
    if "images" in query:
        if not isinstance(query["images"], list) or not query["images"]:
            raise ValueError("images must be a nonempty list")
        for im in query["images"]:
            _source_shape(im) if frozen else _nonempty(im, "image path")
    if "pointcloud" in query:
        pc = query["pointcloud"]
        _source_shape(pc) if frozen else _nonempty(pc, "pointcloud path")
        if Path(pc["path"] if frozen else pc).suffix.lower() != ".npy":
            raise ValueError("query pointcloud must be a raw .npy file")
    available = ("text" in query, "images" in query, "pointcloud" in query)
    conditions = query["conditions"]
    if (not isinstance(conditions, list) or not conditions
            or any(not isinstance(c, str) or c not in QUERY_CONDITIONS for c in conditions)
            or len(set(conditions)) != len(conditions)):
        raise ValueError("conditions must be unique supported condition names")
    for condition in conditions:
        if any(wanted and not have for wanted, have in zip(QUERY_CONDITIONS[condition], available)):
            raise ValueError(f"{query['query_id']}: missing raw modality for {condition}")
    for key in ("qrels", "judgments"):
        if not isinstance(query[key], dict) or set(query[key]) != set(conditions):
            raise ValueError(f"{key} must cover exactly the requested conditions")
    for condition in conditions:
        labels = query["qrels"][condition]
        if not isinstance(labels, dict) or set(labels) != set(gallery):
            raise ValueError(f"{condition}: qrels must judge every gallery UID exactly")
        if any(type(v) not in (bool, int) or v not in (0, 1) for v in labels.values()):
            raise ValueError("qrels labels must be booleans or integer 0/1")
        if not any(labels.values()):
            raise ValueError(f"{condition}: qrels must include at least one positive")
        _metadata(query["judgments"][condition], ("assessor", "method", "criteria"),
                  f"{condition} judgment provenance")


def _spec(spec):
    _keys(spec, ("schema", "gallery_uids", "queries", "provenance"), label="spec")
    if spec["schema"] != SPEC_SCHEMA:
        raise ValueError("unsupported intent spec schema")
    gallery = _ids(spec["gallery_uids"], "gallery_uids")
    if not isinstance(spec["provenance"], dict):
        raise ValueError("spec provenance must be an object")
    if not isinstance(spec["queries"], list) or not spec["queries"]:
        raise ValueError("queries must be a nonempty list")
    for query in spec["queries"]:
        _query(query, gallery, frozen=False)
    _ids([q["query_id"] for q in spec["queries"]], "query_ids")
    return gallery


def _resolved_query(query, parent, *, capture):
    result = copy.deepcopy(query)
    for key in ("images", "pointcloud"):
        if key not in query:
            continue
        originals = query[key] if key == "images" else [query[key]]
        resolved = []
        for name in originals:
            path = Path(name)
            path = (path if path.is_absolute() else parent / path).resolve()
            resolved.append(_capture(path) if capture else {"path": str(path)})
        result[key] = resolved if key == "images" else resolved[0]
    return result


def _query_metadata(query):
    result = copy.deepcopy(query)
    if "images" in result:
        result["images"] = [{"path": s["path"]} for s in result["images"]]
    if "pointcloud" in result:
        result["pointcloud"] = {"path": result["pointcloud"]["path"]}
    return result


def _gallery(uid, record, encoding, n_views, *, verify):
    _keys(record, ("embedding", "cloud", "annotation", "embedding_sidecar", "canonical_text"),
          ("cloud_sidecar",), label=f"gallery {uid}")
    raw = {}
    for key, suffix in (("embedding", ".npz"), ("cloud", ".npz"),
                        ("annotation", ".json"), ("embedding_sidecar", ".json"),
                        ("cloud_sidecar", ".json")):
        if key not in record:
            continue
        _source_shape(record[key])
        if Path(record[key]["path"]).name != uid + suffix:
            raise ValueError(f"{uid}: {key} belongs to another gallery UID")
        if verify:
            raw[key] = verified_source_bytes(record[key])
    _nonempty(record["canonical_text"], f"{uid} canonical text")
    if not verify:
        return None
    annotation = _json_bytes(raw["annotation"])
    sidecar = _json_bytes(raw["embedding_sidecar"])
    if annotation.get("uid", uid) != uid or sidecar.get("uid") != uid:
        raise ValueError(f"{uid}: gallery source UID mismatch")
    if sidecar != record["embedding_sidecar"].get("content"):
        raise ValueError(f"{uid}: embedding sidecar content mismatch")
    if (sidecar.get("n_views") != n_views or sidecar.get("aggregation") != "mean"
            or sidecar.get("text_serialization") != encoding["text_serialization"]):
        raise ValueError(f"{uid}: embedding sidecar disagrees with encoding protocol")
    text = canonical.serialize_annotation(annotation, template=encoding["text_template"])
    if not text.strip() or text != sidecar.get("text") or text != record["canonical_text"]:
        raise ValueError(f"{uid}: canonical text differs from annotation/serializer/sidecar")
    dim = canonical._embedding(io.BytesIO(raw["embedding"]), n_views)
    canonical._cloud(io.BytesIO(raw["cloud"]))
    if "cloud_sidecar" in raw:
        sc = _json_bytes(raw["cloud_sidecar"])
        if (sc != record["cloud_sidecar"].get("content") or sc.get("uid") != uid
                or sc.get("sha256") != record["cloud"]["sha256"]):
            raise ValueError(f"{uid}: cloud sidecar UID/content/digest mismatch")
    return dim


def _validate(manifest, *, verify):
    _keys(manifest, ("schema", "status", "classification", "gallery_uids", "query_ids", "queries",
                     "records", "encoding_protocol", "data_root", "n_views", "provenance",
                     "spec_source", "choices"), label="manifest")
    if (manifest["schema"] != SCHEMA or manifest["status"] != "complete"
            or manifest["classification"] != "IMPLEMENTATION CHOICE" or manifest["choices"] != CHOICES):
        raise ValueError("unsupported or incomplete intent protocol")
    _nonempty(manifest["data_root"], "data_root")
    if not Path(manifest["data_root"]).is_absolute():
        raise ValueError("data_root must be absolute")
    spec = _source_json(manifest["spec_source"], verify=verify)
    gallery = _spec(spec)
    if manifest["gallery_uids"] != gallery or manifest["provenance"] != spec["provenance"]:
        raise ValueError("manifest gallery/provenance differs from source spec")
    if manifest["query_ids"] != [q["query_id"] for q in spec["queries"]]:
        raise ValueError("manifest query IDs differ from source spec")
    if not isinstance(manifest["queries"], list) or len(manifest["queries"]) != len(spec["queries"]):
        raise ValueError("manifest query coverage differs from source spec")
    parent = Path(manifest["spec_source"]["path"]).parent
    for query, original in zip(manifest["queries"], spec["queries"]):
        _query(query, gallery, frozen=True)
        if _query_metadata(query) != _resolved_query(original, parent, capture=False):
            raise ValueError("query metadata/qrels differ from source spec")
        if verify:
            for source in query.get("images", []):
                raw = verified_source_bytes(source)
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()
                # verify() checks the container; load() also decodes the pixels.
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
            if "pointcloud" in query:
                raw = verified_source_bytes(query["pointcloud"])
                if not raw.startswith(b"\x93NUMPY"):
                    raise ValueError("query pointcloud must be an NPY array, not an embedding/archive")
                cloud = canonical._cloud(io.BytesIO(raw), query=True)
                if not (cloud[:, :3] != cloud[0, :3]).any():
                    raise ValueError("query pointcloud has no spatial extent for pc_norm")
    encoding = _source_json(manifest["encoding_protocol"], verify=verify)
    n_views = canonical._encoding(encoding)
    if type(manifest["n_views"]) is not int or manifest["n_views"] != n_views:
        raise ValueError("n_views disagrees with encoding protocol")
    if not isinstance(manifest["records"], dict) or set(manifest["records"]) != set(gallery):
        raise ValueError("gallery records must cover exactly gallery_uids")
    dims = {_gallery(uid, manifest["records"][uid], encoding, n_views, verify=verify) for uid in gallery}
    if verify and len(dims) != 1:
        raise ValueError("mixed gallery embedding dimensions")


def _sources(manifest):
    yield manifest["spec_source"]
    yield manifest["encoding_protocol"]
    for rec in manifest["records"].values():
        for key in ("embedding", "cloud", "annotation", "embedding_sidecar", "cloud_sidecar"):
            if key in rec:
                yield rec[key]
    for query in manifest["queries"]:
        yield from query.get("images", [])
        if "pointcloud" in query:
            yield query["pointcloud"]


def prepare_protocol(spec_path: Path, out_dir: Path) -> Path:
    """Validate raw inputs on CPU and publish a new manifest without overwrites."""
    spec_source = _capture(Path(spec_path), content=True)
    spec = spec_source["content"]
    gallery = _spec(spec)
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {out_dir}")
    for root in (paths.EMBEDDINGS, paths.POINTCLOUDS, paths.ANNOTATIONS, paths.OBJAVERSE_GLB):
        if out_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError("output must not be inside canonical inputs")
    encoding = _capture(paths.OUTPUTS / "stage1_encoding_protocol.json", content=True)
    n_views = canonical._encoding(encoding["content"])
    records = {}
    for uid in gallery:
        sidecar = _capture(paths.EMBEDDINGS / f"{uid}.json", content=True)
        records[uid] = {"embedding": _capture(paths.EMBEDDINGS / f"{uid}.npz"),
                        "cloud": _capture(paths.POINTCLOUDS / f"{uid}.npz"),
                        "annotation": _capture(paths.ANNOTATIONS / f"{uid}.json"),
                        "embedding_sidecar": sidecar, "canonical_text": sidecar["content"].get("text")}
        side = paths.POINTCLOUDS / f"{uid}.json"
        if side.exists():
            records[uid]["cloud_sidecar"] = _capture(side, content=True)
    queries = [_resolved_query(q, Path(spec_source["path"]).parent, capture=True) for q in spec["queries"]]
    manifest = {"schema": SCHEMA, "status": "complete", "classification": "IMPLEMENTATION CHOICE",
                "gallery_uids": gallery, "query_ids": [q["query_id"] for q in queries], "queries": queries,
                "records": records, "encoding_protocol": encoding, "data_root": str(paths.DATA.resolve()),
                "n_views": n_views, "provenance": spec["provenance"], "spec_source": spec_source,
                "choices": copy.deepcopy(CHOICES)}
    _validate(manifest, verify=True)
    # Detect observed drift between initial reads and completed preflight.
    for source in _sources(manifest):
        verified_source_bytes(source)
    out_dir.mkdir(parents=True, exist_ok=False)
    dest = out_dir / "protocol.json"
    part = out_dir / "protocol.json.part"
    with part.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.link(part, dest)  # atomic publication that never replaces another file
    part.unlink()
    return dest


def load_protocol(path, verify=True) -> dict:
    """Validate condition coverage and, by default, all referenced file bytes."""
    manifest = _json_bytes(Path(path).read_bytes())
    _validate(manifest, verify=verify)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(prepare_protocol(args.spec, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
