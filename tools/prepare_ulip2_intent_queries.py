#!/usr/bin/env python3
"""Extract specified ULIP-2 raw captions into a draft requiring relevance review.

Only shard groups joined through objaverse_meta.json are read. No embeddings,
images, models, judgments, or same-UID positives are generated. Optional XYZRGB
is copied to float32 NPY without sampling or normalization; the downstream intent
protocol must validate its required point count and collect all relevance labels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import tarfile
from pathlib import Path

import numpy as np


DEFAULT_METADATA = Path("/home/kyzen/upstream/openshape-objaverse-embeddings/objaverse_meta.json")
CAPTION_FIELDS = ("blip_caption", "msft_caption", "text")
MAX_MEMBER_BYTES = 64 * 1024 * 1024
_UID = re.compile(r"[0-9a-f]{32}\Z")


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Allow only the constructors needed by legacy NumPy object-array files."""

    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return {"ndarray": np.ndarray, "dtype": np.dtype}[name]
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            core = np._core if hasattr(np, "_core") else np.core
            return {"_reconstruct": core.multiarray._reconstruct, "scalar": core.multiarray.scalar}[name]
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def decode_numpy_record(raw: bytes) -> dict:
    """Decode one scalar object NPY; never use unrestricted allow_pickle=True."""
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported legacy NPY version: {version}")
    if shape != () or dtype != np.dtype(object):
        raise ValueError("expected a scalar object NPY containing one dictionary")
    value = _NumpyOnlyUnpickler(stream).load()
    if stream.read():
        raise ValueError("unexpected data after the NPY pickle")
    if type(value) is not np.ndarray or value.shape != () or value.dtype != np.dtype(object):
        raise ValueError("NPY header and decoded array disagree")
    record = value.item()
    if type(record) is not dict or any(not isinstance(key, str) for key in record):
        raise ValueError("NPY record must be a dictionary with string keys")
    return record


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stat(value) -> dict:
    return {"device": value.st_dev, "inode": value.st_ino, "size_bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _source(path: Path) -> dict:
    requested = path.absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"source is not a file: {requested}")
    return {"requested_path": str(requested), "path": str(resolved), "stat": _stat(resolved.stat())}


def _check_source(source: dict) -> None:
    if (Path(source["requested_path"]).resolve(strict=True) != Path(source["path"])
            or _stat(Path(source["path"]).stat()) != source["stat"]):
        raise ValueError(f"source changed during extraction: {source['requested_path']}")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    source = _source(path)
    raw = Path(source["path"]).read_bytes()
    _check_source(source)
    source["sha256"] = _digest(raw)
    value = json.loads(raw, object_pairs_hook=_json_object)
    return value, source


def _selected_metadata(metadata: dict, uids: list[str]) -> dict:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("entries"), list):
        raise ValueError("metadata must be an object containing entries: list")
    wanted, selected = set(uids), {}
    for entry in metadata["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("u"), str):
            raise ValueError("metadata entries must have a string u field")
        uid = entry["u"]
        if uid not in wanted:
            continue
        if uid in selected:
            raise ValueError(f"duplicate metadata UID: {uid}")
        glb = entry.get("glb")
        match = (re.fullmatch(r"glbs/([0-9]{3}-[0-9]{3})/" + re.escape(uid) + r"\.glb", glb)
                 if isinstance(glb, str) else None)
        if not match:
            raise ValueError(f"metadata UID/glb join mismatch: {uid}")
        selected[uid] = {"u": uid, "glb": glb, "shard": match.group(1)}
    return selected


def _caption(record: dict, field: str) -> tuple[str, dict]:
    if field not in record:
        raise ValueError(f"missing caption field: {field}")
    value = record[field]
    selection = {"field": field, "selection": "raw_string"}
    if field == "text" and isinstance(value, list):
        if len(value) != 1:
            raise ValueError("text list must contain exactly one caption; no implicit joining or first-item selection")
        value = value[0]
        selection["selection"] = "single_list_element"
        selection["index"] = 0
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"caption field {field} is empty or is not a string")
    selection["utf8_sha256"] = _digest(value.encode("utf-8"))
    return value, selection


def _pointcloud(record: dict) -> np.ndarray:
    arrays = []
    for field in ("xyz", "rgb"):
        raw = np.asarray(record.get(field))
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or raw.dtype.kind not in "iuf":
            raise ValueError(f"{field} must be a nonempty real array of shape (N, 3)")
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{field} is non-finite or cannot be represented in float32")
        arrays.append(value)
    xyz, rgb = arrays
    if len(xyz) != len(rgb):
        raise ValueError("xyz and rgb point counts differ")
    if not np.any(xyz.max(axis=0) != xyz.min(axis=0)):
        raise ValueError("xyz has no spatial extent")
    if np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("raw RGB must already be in [0, 1]; no implicit rescaling")
    return np.concatenate((xyz, rgb), axis=1)


def _publish_bytes(path: Path, raw: bytes) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(part, path)
    part.unlink()


def prepare_queries(*, shards: str | Path, asset_uids: str | Path, metadata: str | Path,
                    caption_field: str, out_dir: str | Path, include_pointcloud: bool = False) -> dict:
    """Publish draft_queries.json in a new directory; failures never gain qrels."""
    if caption_field not in CAPTION_FIELDS:
        raise ValueError(f"caption_field must be one of {CAPTION_FIELDS}")
    out = Path(out_dir).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"out_dir must be fresh: {out}")
    shard_root = Path(shards).resolve(strict=True)
    if not shard_root.is_dir():
        raise ValueError("shards must be a directory containing 000-xxx.tar.gz files")
    uids, uid_source = _read_json(Path(asset_uids))
    if (not isinstance(uids, list) or not uids
            or any(not isinstance(uid, str) or not _UID.fullmatch(uid) for uid in uids)
            or len(set(uids)) != len(uids)):
        raise ValueError("asset-uids must be a nonempty list of unique lowercase 32-hex Objaverse UIDs")
    meta, metadata_source = _read_json(Path(metadata))
    selected = _selected_metadata(meta, uids)
    del meta
    by_shard = {}
    failures, found, archives = {}, {}, {}
    for uid in uids:
        if uid not in selected:
            failures[uid] = {"source_uid": uid, "reason": "missing_metadata"}
        else:
            by_shard.setdefault(selected[uid]["shard"], set()).add(uid)
    out.mkdir(parents=True, exist_ok=False)
    if include_pointcloud:
        (out / "pointclouds").mkdir()
    for group, wanted in sorted(by_shard.items()):
        archive_path = shard_root / f"{group}.tar.gz"
        if not archive_path.exists():
            for uid in wanted:
                failures[uid] = {"source_uid": uid, "reason": "missing_archive", "path": str(archive_path)}
            continue
        archive_source = _source(archive_path)
        archives[group] = archive_source
        seen = set()
        with Path(archive_source["path"]).open("rb") as archive_file:
            if _stat(os.fstat(archive_file.fileno())) != archive_source["stat"]:
                raise ValueError(f"archive changed before opening: {archive_path}")
            with tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
                for member in archive:
                    uid = Path(member.name).stem
                    if uid not in wanted:
                        continue
                    if uid in seen:
                        raise ValueError(f"duplicate selected UID in archive: {uid}")
                    seen.add(uid)
                    if member.name != f"{group}/{uid}.npy" or not member.isfile():
                        raise ValueError(f"tar member/metadata UID join mismatch: {member.name}")
                    if not 0 < member.size <= MAX_MEMBER_BYTES:
                        failures[uid] = {"source_uid": uid, "reason": "invalid_member_size", "member": member.name}
                        continue
                    raw = archive.extractfile(member).read()
                    member_source = {"name": member.name, "size_bytes": len(raw), "sha256": _digest(raw),
                                     "tar_offset_data": member.offset_data}
                    provenance = {"source": "ULIP-2 Objaverse-LVIS shard", "creator": f"upstream {caption_field}",
                                  "method": "raw field extraction; no generation, embeddings, or relevance inference",
                                  "metadata_entry": selected[uid], "archive": archive_source,
                                  "member": member_source, "upstream_generator_checkpoint": "UNKNOWN"}
                    try:
                        record = decode_numpy_record(raw)
                        for field, expected in (("dataset", "Objaverse"), ("group", group), ("id", uid)):
                            if record.get(field) != expected:
                                raise ValueError(f"record {field} is missing or disagrees with metadata/member identity")
                        for field in ("u", "uid"):
                            if field in record and record[field] != uid:
                                raise ValueError(f"record {field} disagrees with metadata/member UID")
                        text, selection = _caption(record, caption_field)
                        pc = _pointcloud(record) if include_pointcloud else None
                    except (ValueError, TypeError, pickle.UnpicklingError, EOFError) as exc:
                        failures[uid] = {"source_uid": uid, "reason": "invalid_record", "detail": str(exc),
                                         "provenance": provenance}
                        continue
                    provenance["caption"] = selection
                    provenance["record_identity"] = {field: record[field] for field in ("dataset", "group", "id")}
                    draft = {"id": f"ulip2-{uid}", "query_id": f"ulip2-{uid}", "source_uid": uid,
                             "status": "needs_relevance_review", "text": text, "conditions": ["text"],
                             "provenance": provenance}
                    if pc is not None:
                        buffer = io.BytesIO()
                        np.save(buffer, pc, allow_pickle=False)
                        pc_bytes = buffer.getvalue()
                        pc_path = out / "pointclouds" / f"{uid}.npy"
                        _publish_bytes(pc_path, pc_bytes)
                        draft["pointcloud"] = str(pc_path)
                        draft["conditions"] = ["text", "pc", "text+pc"]
                        provenance["pointcloud"] = {"path": str(pc_path), "sha256": _digest(pc_bytes),
                                                   "shape": list(pc.shape), "dtype": "float32",
                                                   "transform": "concatenate xyz/rgb and cast float32; no sampling or normalization"}
                    found[uid] = draft
            if _stat(os.fstat(archive_file.fileno())) != archive_source["stat"]:
                raise ValueError(f"archive changed during reading: {archive_path}")
        _check_source(archive_source)
        for uid in wanted - seen:
            failures[uid] = {"source_uid": uid, "reason": "missing_member", "archive": archive_source}
    for source in (uid_source, metadata_source, *archives.values()):
        _check_source(source)
    result = {"schema": "metafind.ulip2_intent_query_draft.v1", "status": "needs_relevance_review",
              "caption_field": caption_field, "include_pointcloud": include_pointcloud,
              "requested_uids": uids, "records": [found[uid] for uid in uids if uid in found],
              "failures": [failures[uid] for uid in uids if uid in failures],
              "sources": {"asset_uids": uid_source, "metadata": metadata_source,
                          "shards_directory": str(shard_root), "archives": archives,
                          "adapter": {"path": str(Path(__file__).resolve()),
                                      "sha256": _digest(Path(__file__).read_bytes())}},
              "limitations": ["No qrels or judgments: the source UID is provenance, not an automatic positive.",
                              "Human review must define the intent and judge every gallery asset for each condition.",
                              "image_feat and thumbnail_feat are not raw images; image conditions are unavailable.",
                              "Caption generator checkpoint identity is unknown; captions are preserved as supplied.",
                              "Archives are bound by path/stat and consumed member SHA256, not a whole-archive digest."]}
    _publish_bytes(out / "draft_queries.json", (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--asset-uids", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--caption-field", choices=CAPTION_FIELDS, required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--include-pointcloud", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_queries(**vars(args))
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"{len(result['records'])} draft queries; {len(result['failures'])} failures; relevance review required", flush=True)
    return 2 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
