"""Encode every admitted asset with the FROZEN gallery tower.

# IMPLEMENTS-NODE: n11_gallery_index_staging
# IMPLEMENTS-NODE: n12_promote_index
# IMPLEMENTS-NODE: n11b_stage2_gallery_index

Writes ``gallery_index_staging`` (n11), ``gallery_index`` (n12),
``stage2_gallery_index`` (n11b), and ``run_progress`` / ``cost_ledger``.

Three nodes, one file, because they are the same operation over different
corpora and the thing that must not drift between them is the ENCODER. Splitting
them would mean three copies of "load the checkpoint, freeze the tower, hash the
weights", and the hash is the whole point.

Why staging and promotion are separate
---------------------------------------

n11 writes a staging index, G4 verifies it, n12 promotes exactly that artifact.
The alternative -- write the live index directly -- means a failed verification
leaves a partially-written index that every downstream evaluation would read as
authoritative. Promotion copies nothing: it records the digest G4 saw, and a
second differing write for the same checkpoint is an error rather than an
update.

Why the encoder hash travels with the index
--------------------------------------------

[2.6] The gallery encoder is frozen during Stage 2. That is only checkable if
the weights that produced an index are pinned to it. An index built by a drifted
encoder trains and evaluates the model against embeddings it will never produce
at inference, and nothing in the loss or the metrics would reveal it -- the
numbers would simply be wrong in a self-consistent way.

Stage 1's Objaverse index and Stage 2's ProcTHOR index NEVER merge
-------------------------------------------------------------------

[U-08a] Stage 2 draws its positives from a ProcTHOR gallery; Table 1 retrieves
from Objaverse. Merging them would change Table 1's denominator, which is
already an open question (U-09) and must not acquire a second one.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from metafind import paths, runlog

paths.setup_env()

STAGING_PATH = paths.OUTPUTS / "gallery_index_staging.json"
PROMOTED_PATH = paths.OUTPUTS / "gallery_index.json"
STAGE2_PATH = paths.OUTPUTS / "stage2_gallery_index.json"


def load_checkpoint_record(record_path: str | Path | None = None) -> dict:
    """Which Stage 1 checkpoint this index is built from.

    [CODEX MAJOR 2026-08-30] Was a fixed `paths.CHECKPOINTS / "stage1_ckpt.json"`.
    Stage 1 gained `--out-dir` so a sweep's arms stop overwriting each other, and
    with the path fixed here, a run-specific checkpoint could reach downstream
    only by being copied back over the canonical name -- which destroys the
    provenance the out-dir was added to create.

    The default is unchanged, so every existing command keeps working; naming
    the record is how a sweep's selected arm is promoted without a copy.
    """
    path = Path(record_path) if record_path else paths.CHECKPOINTS / "stage1_ckpt.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run n10_train_stage1 first")
    record = json.loads(path.read_text())
    # [CODEX MAJOR 2026-08-30] The record is a CLAIM about a file, and until now
    # nothing checked it. Codex demonstrated the consequence: a record naming
    # checkpoint A's provenance can be pointed at checkpoint B's bytes and every
    # downstream artifact inherits the wrong identity, silently. `--out-dir`
    # makes several checkpoints exist at once, so this stops being theoretical.
    weights = Path(record["uri"])
    if not weights.exists():
        raise FileNotFoundError(
            f"{path} names {weights}, which does not exist. The record and its "
            "weights have been separated.")
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(
            f"{path} records sha256 {record['sha256'][:16]}... but {weights} "
            f"hashes to {actual[:16]}.... Refusing: an index built from these "
            "bytes would carry the other checkpoint's provenance.")
    return record


GALLERY_ENCODER_HASH_VERSION = 2


def stage2_layout_settings(record: dict) -> tuple[bool, float | None]:
    """Resolve the saved layout branch, including the approved no-layout arm."""
    import math

    initial = record.get("lambda_init")
    use_layout = record.get("use_layout")
    if "use_layout" not in record:
        # Earlier Stage 2 writers already bound the variant and lambda to the
        # weights; the no-layout arm deliberately saved lambda_init=None.
        use_layout = not (record.get("variant_id") == "no_layout" and initial is None)
    if not isinstance(use_layout, bool):
        raise ValueError("Stage 2 use_layout must be boolean")
    variant = record.get("variant_id")
    if (variant == "full" and not use_layout) or (variant == "no_layout" and use_layout):
        raise ValueError(f"Stage 2 variant {variant!r} disagrees with use_layout")
    if not use_layout:
        if initial is not None:
            raise ValueError("Stage 2 without layout must have lambda_init=None")
        return False, None
    if not isinstance(initial, dict) or "init_lambda" not in initial:
        raise ValueError("Stage 2 with layout requires lambda_init.init_lambda")
    try:
        value = float(initial["init_lambda"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage 2 init_lambda must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("Stage 2 init_lambda must be finite")
    return True, value


def load_stage2_checkpoint_record(record_path, stage1_ckpt: dict,
                                  variant: str = "full", state_path=None, *,
                                  allow_legacy: bool = False) -> dict:
    """Verify ancestry and the actual Stage 2 bytes before constructing a head."""
    record_path = Path(record_path)
    rec = json.loads(record_path.read_text())
    if "uri" not in rec:
        if variant not in rec:
            raise ValueError(f"{record_path} holds no variant {variant!r}")
        rec = rec[variant]
    parent = rec.get("stage1_checkpoint_sha256")
    if parent != stage1_ckpt.get("sha256") or not parent:
        raise ValueError(f"{record_path} was fine-tuned from Stage 1 checkpoint "
                         f"{parent}, not {stage1_ckpt.get('sha256')}")
    for key in ("uri", "sha256", "lambda_init"):
        if key not in rec:
            raise ValueError(f"{record_path} lacks {key!r}; not a Stage 2 record")
    if state_path is not None and Path(state_path).resolve() != Path(rec["uri"]).resolve():
        raise ValueError(f"{record_path} names {rec['uri']}, not --stage2-state {state_path}")
    import torch
    raw = verified_checkpoint_bytes(rec)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        if not allow_legacy or rec.get("input_identity"):
            raise ValueError("Stage 2 checkpoint lacks embedded metadata; explicit legacy replay required")
        for key in ("gallery_source_status", "allow_legacy_gallery_index",
                    "temperature_init", "use_layout", "semantic_source_status",
                    "allow_legacy_semantic_inputs"):
            if key in rec:
                raise ValueError(f"Stage 2 record has unbound {key}")
    else:
        for key, value in metadata.items():
            if key not in rec or rec[key] != value:
                raise ValueError(f"Stage 2 checkpoint and record disagree: {key}")
        for key in ("input_identity", "arch_protocol", "stage2_protocol", "graph_unit",
                    "layout_input_dims", "stage1_checkpoint_sha256", "stage1_model_inputs", "lambda_init",
                    "gallery_source_status", "allow_legacy_gallery_index",
                    "temperature_init", "use_layout", "semantic_source_status",
                    "allow_legacy_semantic_inputs"):
            if key in rec and key not in metadata:
                raise ValueError(f"Stage 2 record has unbound {key}")
    if "stage1_model_inputs" in rec:
        from metafind.train.stage1 import (effective_stage1_model_inputs, fusion_config_for,
                                          load_stage1_model_config)
        bound_parent = load_stage1_model_config(stage1_ckpt["uri"], stage1_ckpt)
        enc, train, _ = effective_stage1_model_inputs(bound_parent, {}, {}, {})
        s2_enc, s2_train, _ = effective_stage1_model_inputs(
            {"model_inputs": rec["stage1_model_inputs"]}, {}, {}, {})
        for gallery in (False, True):
            if asdict(fusion_config_for(enc, train, gallery=gallery)) != asdict(
                    fusion_config_for(s2_enc, s2_train, gallery=gallery)):
                raise ValueError("Stage 2 forward configuration differs from its Stage 1 parent")
        if train["tower_sharing"] != s2_train["tower_sharing"]:
            raise ValueError("Stage 2 tower sharing differs from its Stage 1 parent")
    stage2_layout_settings(rec)
    return rec


def verified_checkpoint_bytes(record: dict) -> bytes:
    """Return the bytes hashed, so an overlay never reopens unverified state."""
    if not record.get("uri") or not record.get("sha256"):
        raise ValueError("checkpoint record requires uri and sha256")
    raw = Path(record["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"{record['uri']} does not match the sha256 in its record")
    return raw


def gallery_forward_config(backbone, model, declared_modalities=None) -> dict:
    """Forward choices absent from state_dict, excluding the query tower.

    Stage 2 legitimately changes query fusion without changing the gallery.
    Conversely, prefusion_norm and zero_pad can change a gallery vector with
    every parameter byte unchanged, so those belong to its identity.
    """
    fusion = getattr(model.gallery, "fusion", None)
    cfg = getattr(fusion, "cfg", None)
    backbone_cfg = getattr(backbone, "cfg", None)
    return {
        "gallery_class": type(model.gallery).__qualname__,
        "fusion": asdict(cfg) if is_dataclass(cfg) else None,
        "backbone_class": type(backbone.model).__qualname__,
        "backbone_dtype": str(getattr(backbone_cfg, "dtype", "unspecified")),
        "declared_modalities": list(("text", "image", "pc")
                                    if declared_modalities is None else declared_modalities),
    }


def gallery_encoder_sha256(backbone, model, include_buffers: bool = False,
                           *, hash_version: int = GALLERY_ENCODER_HASH_VERSION,
                           declared_modalities=None) -> str:
    """A digest over EVERYTHING that produces a gallery embedding.

    ``include_buffers`` adds the registered buffers -- PointBERT's BatchNorm
    running statistics, which decide the embedding in eval mode and which
    training moves. A parameters-only digest could not tell two encoders with
    the same weights and different statistics apart. New indexes are written
    with buffers included and say so in their record
    (``gallery_encoder_hash_includes_buffers``); a reader must compute the
    live digest the same way the record was, which is why this is a flag and
    not a silent change of the function.

    The gallery path is::

        text  -> OpenCLIP  --+
        image -> OpenCLIP  --+--> gallery fusion -> e_gallery
        pc    -> PointBERT -> pc_projection --+

    so hashing the fusion alone is not an encoder identity. An earlier version
    did exactly that, and the name `gallery_encoder_sha256` made it read as more
    than it was: two runs with DIFFERENT fine-tuned PointBERTs and the same
    fusion produced the same digest, and G4's "gallery encoder matches Stage 1"
    would have passed while the embeddings differed.

    Sorted by name because Python's parameter iteration order is stable but not
    guaranteed across refactors, and a hash that changes when a module is
    reordered would report drift that did not happen -- worse than no hash,
    because it teaches everyone to ignore it.
    """
    if hash_version not in (1, GALLERY_ENCODER_HASH_VERSION):
        raise ValueError(f"unsupported gallery encoder hash version {hash_version}")
    h = hashlib.sha256()
    if hash_version == GALLERY_ENCODER_HASH_VERSION:
        h.update(b"metafind-gallery-encoder-v2\0")
        h.update(json.dumps(gallery_forward_config(backbone, model, declared_modalities),
                            sort_keys=True, separators=(",", ":")).encode())
    for tag, module in (("backbone", backbone.model), ("gallery", model.gallery)):
        for name, p in sorted(module.named_parameters()):
            h.update(f"{tag}.{name}".encode())
            h.update(p.detach().cpu().numpy().tobytes())
        if include_buffers:
            for name, b in sorted(module.named_buffers()):
                h.update(f"{tag}.buffer.{name}".encode())
                h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def verify_gallery_encoder(record: dict, backbone, model, *,
                           parent_checkpoint: dict | None = None,
                           allow_legacy: bool = False, declared_modalities=None) -> str:
    """Verify v2 identity, or explicitly validate legacy parent configuration.

    A v1 record never proves which parameter-free forward flags its producer
    used. Compatibility is therefore an explicit, recorded caller choice.
    """
    version = record.get("gallery_encoder_hash_version", 1)
    if version == 1:
        if not allow_legacy:
            raise ValueError("legacy gallery encoder hash v1 lacks forward configuration; "
                             "rebuild the index or explicitly allow legacy compatibility")
        if parent_checkpoint is None:
            raise ValueError("legacy gallery compatibility requires its parent checkpoint")
        from metafind.train.stage1 import (load_stage1_model_config, load_protocols,
                                          effective_stage1_model_inputs, fusion_config_for)
        parent = load_stage1_model_config(parent_checkpoint["uri"], parent_checkpoint)
        enc, train, _ = effective_stage1_model_inputs(parent, *load_protocols())
        expected = asdict(fusion_config_for(enc, train, gallery=True))
        actual = gallery_forward_config(backbone, model, declared_modalities)["fusion"]
        if actual != expected:
            raise ValueError("legacy gallery runtime forward configuration differs from parent checkpoint")
    live = gallery_encoder_sha256(
        backbone, model, include_buffers=bool(record.get("gallery_encoder_hash_includes_buffers")),
        hash_version=version, declared_modalities=declared_modalities)
    if live != record["gallery_encoder_sha256"]:
        raise ValueError(f"gallery encoder loaded here hashes to {live[:16]} but index was built by "
                         f"{record['gallery_encoder_sha256'][:16]}; gallery identity mismatch")
    return live


def _write(path: Path, obj, dump=json.dump) -> None:
    """Atomic, fsynced write. ``dump(obj, fh)`` -- json by default.

    ``dump`` exists so G4's YAML gate record goes through THIS writer rather
    than a second one: the temp-and-rename plus fsync is the property that
    matters, and it should not have to be re-implemented per serialisation.
    ``yaml.safe_dump`` has the same ``(obj, stream)`` signature as ``json.dump``.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def build_index(embeddings: np.ndarray, ids: list[str], out: Path,
                extra: dict[str, np.ndarray] | None = None, *,
                source_identity: dict | None = None) -> dict:
    """Write the vectors and return the record that describes them.

    ``extra`` holds further per-asset arrays stored beside ``embeddings`` in
    the same file, row-aligned with ``ids``. The Stage 2 index uses it to keep
    the three raw modality vectors (text, image, point cloud) that the frozen
    ULIP-2 backbone produced for each asset, so that Stage 2 training can look
    them up instead of re-running the frozen encoders on every step.
    """
    tmp = out.with_suffix(".part.npz")
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in (extra or {}).items()}
    for k, v in arrays.items():
        if v.shape[0] != len(ids):
            raise ValueError(f"extra array {k!r} has {v.shape[0]} rows for {len(ids)} ids")
    if source_identity is not None:
        arrays["stage2_source_identity_sha256"] = np.array(_source_identity_digest(source_identity))
    np.savez_compressed(tmp, ids=np.array(ids),
                        embeddings=embeddings.astype(np.float32), **arrays)
    tmp.replace(out)
    record = {
        "uri": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
    }
    if source_identity is not None:
        record["source_identity"] = source_identity
    return record


INDEX_RECORD_FIELDS = ("uri", "sha256", "dim", "count",
                       "stage1_checkpoint_sha256", "gallery_encoder_sha256")


class IndexUnreadable(ValueError):
    """The .npz cannot be opened, or does not carry ``ids`` and ``embeddings``.

    A subclass so the documented contract ("raises ValueError") stays true, and
    a distinct type so G4 can tell MISSING EVIDENCE from a FAILED CHECK without
    matching on an exception message. Those are rc 3 and rc 2 and they are not
    interchangeable: a corrupt archive means nobody knows whether the index was
    good, while a record that misdescribes its index means somebody knows it was
    not.
    """


def verified_index(record: dict, source: str, *, include_arrays: bool = False):
    """Hash the bytes, then read THOSE bytes. Both halves, one call.

    Splitting them -- verify here, open there -- is two separate opens of one
    path, and the file can change in between. A verification that does not hand
    back the bytes it verified has verified something other than what gets used.

    ``source`` names whoever is making the claim (the promoted registry, the
    staging record) so the error says which document is wrong. Returns
    ``(ids, embeddings)`` with ``embeddings`` exactly as stored: float32,
    unnormalised, in the index's own row order.
    """
    if missing := [k for k in INDEX_RECORD_FIELDS if k not in record]:
        raise ValueError(f"{source} record is missing {missing}; it cannot "
                         "identify the index it describes")
    uri = Path(record["uri"])
    if not uri.exists():
        raise FileNotFoundError(
            f"{source} names {uri}, which does not exist. The record and its "
            "vectors have been separated.")
    # ONE read. Hashing `uri.read_bytes()` and then handing `uri` to np.load is
    # two opens of one path, and the docstring above promises the bytes hashed
    # are the bytes returned -- a promise the caller cannot keep and the callee
    # was not enforcing. The window is small and the failure is silent, which is
    # the combination this repository keeps being bitten by.
    raw = uri.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(
            f"{uri} hashes to {actual[:16]}... but {source} records "
            f"{record['sha256'][:16]}.... These are not the verified vectors.")

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
            ids = [str(x) for x in npz["ids"]]
            embeddings = npz["embeddings"]
            arrays = {k: npz[k] for k in npz.files} if include_arrays else None
    except Exception as exc:  # noqa: BLE001 -- any read failure is unreadable
        raise IndexUnreadable(f"{uri} cannot be read as an index: {exc}") from exc
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"{len(ids)} ids for {embeddings.shape[0]} vectors "
                         f"in {uri}")
    if (embeddings.shape[0], embeddings.shape[1]) != (record["count"], record["dim"]):
        raise ValueError(f"{uri} is {embeddings.shape} but {source} records "
                         f"count={record['count']} dim={record['dim']}")
    if include_arrays:
        return ids, embeddings, arrays
    return ids, embeddings


def _source_identity_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _source_file_identity(path: Path) -> dict:
    path = path.absolute()
    with path.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    return {"uri": str(path), "sha256": digest}


def verified_source_bytes(artifact: dict) -> bytes:
    """Read exactly the source bytes whose digest the producer recorded."""
    raw = Path(artifact["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
        raise ValueError(f"Stage 2 gallery source changed: {artifact['uri']}")
    return raw


def _missing_stage2_modalities(record: dict, declared) -> list[str]:
    return [m for m in declared
            if (m == "pc" and record.get("pointcloud_uri") is None)
            or (m == "image" and not record.get("view_paths"))
            or (m == "text" and not record.get("text"))]


def capture_stage2_gallery_sources(declared_modalities, *, limit: int | None = None) -> dict:
    """Snapshot selected JSONs and only the image/cloud files actually encoded.

    [DL-103] Text comes from the canonical fitted metadata map, not the copy
    made by n07b before captions and the text-template change. Validate its
    construction against the metadata annotations without modifying either
    source. Store the exact verified sentence that n11b will encode.
    View order is significant.
    Excluded records remain bound because they decide catalogue membership;
    their unavailable modalities are never read. No scene inputs are needed.
    """
    declared = tuple(declared_modalities)
    if not declared or len(set(declared)) != len(declared) or set(declared) - {"text", "image", "pc"}:
        raise ValueError(f"invalid Stage 2 gallery modalities: {declared}")
    if limit is not None and limit <= 0:
        raise ValueError("Stage 2 gallery limit must be positive")
    root = paths.PROCTHOR_MODALITIES.absolute()
    selected = sorted(root.glob("*.json"))
    if limit is not None:
        selected = selected[:limit]
    text_source, text_map, annotations = None, {}, {}
    if "text" in declared:
        from metafind.models.resolve_stage1 import serialize_fitted

        text_source = {
            "artifact": _source_file_identity(paths.OUTPUTS / "procthor_object_text.json"),
            "annotation_artifact": _source_file_identity(
                paths.OUTPUTS / "procthor_asset_annotations.json"),
            "field": "text", "template": "v3_fit",
            "serializer": "metafind.models.resolve_stage1.serialize_fitted",
        }
        text_map = json.loads(verified_source_bytes(text_source["artifact"]))
        annotations = json.loads(verified_source_bytes(text_source["annotation_artifact"]))
        if not isinstance(text_map, dict) or not isinstance(annotations, dict):
            raise ValueError("Stage 2 canonical text and metadata annotations must be asset maps")
    records, inputs, seen = [], {}, set()
    # A shared view file need only be hashed once during a snapshot.
    file_cache = {}
    def file_identity(uri):
        key = str(Path(uri).absolute())
        if key not in file_cache:
            file_cache[key] = _source_file_identity(Path(key))
        return file_cache[key]

    for path in selected:
        identity = _source_file_identity(path)
        rec = json.loads(verified_source_bytes(identity))
        records.append(identity)
        asset_id = str(rec["asset_id"])
        if asset_id in seen:
            raise ValueError(f"duplicate Stage 2 gallery asset_id: {asset_id}")
        seen.add(asset_id)
        if "text" in declared:
            node_text = text_map.get(asset_id)
            annotation = annotations.get(asset_id)
            if not isinstance(node_text, dict) or not isinstance(annotation, dict):
                raise ValueError(f"Stage 2 canonical text/annotation missing asset {asset_id!r}")
            metadata_source = annotation.get("source")
            if not isinstance(metadata_source, str) \
                    or not metadata_source.startswith("procthor_metadata_v") \
                    or not isinstance(node_text.get("source"), str) \
                    or not node_text["source"].startswith(metadata_source + "@"):
                raise ValueError(f"Stage 2 canonical text for {asset_id!r} lacks current "
                                 "metadata provenance; run procthor_metadata_text first")
            sentence = node_text.get("text")
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"Stage 2 canonical text is empty or invalid for {asset_id!r}")
            if sentence != serialize_fitted(annotation):
                raise ValueError(f"Stage 2 canonical text for {asset_id!r} is not the v3_fit "
                                 "serialization of its metadata; re-run procthor_metadata_text")
            rec["text"] = sentence
        if _missing_stage2_modalities(rec, declared):
            continue
        source = {"record_uri": identity["uri"]}
        if "text" in declared:
            source["text"] = rec["text"]
        if "image" in declared:
            source["images"] = [file_identity(v) for v in rec["view_paths"]]
        if "pc" in declared:
            source["pointcloud"] = file_identity(rec["pointcloud_uri"])
        inputs[asset_id] = source
    return {"version": 2, "declared_modalities": list(declared),
            "modality_records_root": str(root), "selection_limit": limit,
            "modality_records": records, "encoded_inputs": inputs,
            "text_source": text_source}


def verify_stage2_gallery_sources(record: dict, arrays: dict | None = None, *,
                                  allow_legacy: bool = False) -> str:
    """Check current sources against producer evidence, never backfill old indexes.

    The index embeds the identity digest so a fresh sidecar cannot promote old
    vectors to a claim about newly captured inputs. Legacy acceptance is explicit
    and reports incomplete provenance; any partial or mismatched evidence fails.
    """
    identity = record.get("source_identity")
    embedded = None if arrays is None else arrays.get("stage2_source_identity_sha256")
    if identity is None:
        if embedded is not None or not allow_legacy:
            raise ValueError("Stage 2 gallery has no bound source_identity; rebuild the index "
                             "or explicitly allow legacy gallery compatibility")
        return "legacy_unbound"
    if isinstance(identity, dict) and identity.get("version") == 1:
        raise ValueError("Stage 2 gallery source_identity version 1 binds renderer-sidecar "
                         "text, not canonical v3_fit text; rebuild n11b. Legacy permission "
                         "cannot reinterpret this index as version 2.")
    if not isinstance(identity, dict) or identity.get("version") != 2:
        raise ValueError("unsupported Stage 2 gallery source_identity")
    if arrays is not None:
        if embedded is None or np.asarray(embedded).shape != () \
                or str(np.asarray(embedded).item()) != _source_identity_digest(identity):
            raise ValueError("Stage 2 gallery source_identity is not bound to its index bytes")
        # JSON object key order has no meaning. Row order remains bound by the
        # index bytes, whereas this map identifies each row's asset membership.
        if {str(asset) for asset in arrays["ids"].tolist()} != set(identity["encoded_inputs"]):
            raise ValueError("Stage 2 gallery source identity disagrees with index asset IDs")
    declared = record.get("modality_completeness", {}).get("declared_modalities")
    if declared != identity["declared_modalities"]:
        raise ValueError("Stage 2 gallery source modality declaration disagrees with its record")
    if Path(identity["modality_records_root"]).resolve() != paths.PROCTHOR_MODALITIES.resolve():
        raise ValueError("Stage 2 gallery current modality root differs from its recorded source")
    try:
        current = capture_stage2_gallery_sources(declared, limit=identity["selection_limit"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Stage 2 gallery source bytes or membership changed; "
                         f"rebuild the index: {exc}") from exc
    if current != identity:
        raise ValueError("Stage 2 gallery source bytes or membership changed; rebuild the index")
    return "verified"


def verified_stage2_index(record: dict, parent_sha: str, declared_modalities, *,
                          allow_legacy_sources: bool = False):
    """Read one declared ProcTHOR gallery belonging to this Stage 1 parent."""
    if record.get("stage1_checkpoint_sha256") != parent_sha:
        raise ValueError("Stage 2 gallery belongs to a different Stage 1 checkpoint")
    declared = tuple(declared_modalities)
    recorded = tuple(record.get("modality_completeness", {}).get("declared_modalities", ()))
    if not declared or declared != recorded:
        raise ValueError(f"Stage 2 gallery modality declaration {recorded} != {declared}")
    ids, embeddings, arrays = verified_index(record, "Stage 2 gallery", include_arrays=True)
    if len(set(ids)) != len(ids):
        raise ValueError("Stage 2 gallery has duplicate asset IDs")
    for name in declared:
        if name not in arrays or arrays[name].shape != embeddings.shape:
            raise ValueError(f"Stage 2 gallery lacks aligned raw {name!r} vectors")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Stage 2 gallery has nonfinite {name!r} vectors")
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 2 gallery has nonfinite embeddings")
    verify_stage2_gallery_sources(record, arrays, allow_legacy=allow_legacy_sources)
    return ids, embeddings, arrays


def load_promoted_index_for_checkpoint(
    checkpoint_sha: str,
    promoted_path: Path | None = None,
) -> tuple[dict, list[str], np.ndarray]:
    """The promoted gallery index for one Stage 1 checkpoint. CONTRACT.

    Every consumer of ``gallery_index`` goes through here. n15 does not parse
    the registry itself, because the registry is a map of CLAIMS about files --
    ``{stage1_sha: {uri, sha256, dim, count, ...}}`` -- and a claim nobody
    re-checks is how an index built by one encoder gets read under another
    one's provenance.

    Re-verified on EVERY call, not once at import: the file named by the record
    lives on a shared volume, and "it was correct when this process started" is
    not the question a reader is asking.

    Args:
        checkpoint_sha: the Stage 1 checkpoint sha256 the index must belong to.
            This is the registry key AND the identity being asserted.
        promoted_path: the registry. Defaults to ``gallery_index.json``.

    Returns:
        ``(record, ids, embeddings)``. ``record`` is the registry entry, whose
        provenance fields the caller carries into its own output.
        ``embeddings`` is ``(N, D)`` **exactly as stored** -- float32,
        unnormalised, in the index's own row order -- and ``ids[i]`` names row
        ``i``. Row selection, uid ordering and the float64 normalisation are
        n15's semantics, not the registry's, and none of them happen here.

    Raises:
        FileNotFoundError: no registry, or the record names a file that is gone.
        KeyError: the registry holds no index for this checkpoint.
        ValueError: the record cannot identify its index, the bytes do not hash
            to the recorded digest, the array disagrees with the record's own
            ``count``/``dim``, or an asset id repeats. Never returns None, an
            empty result, or a partially-verified array.
    """
    path = Path(promoted_path) if promoted_path else PROMOTED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- n12 has not promoted an index. A missing "
            "registry is not an empty one; refusing to score against nothing.")
    promoted = json.loads(path.read_text())
    if checkpoint_sha not in promoted:
        raise KeyError(
            f"{path} holds no index for checkpoint {checkpoint_sha[:16]}...; "
            f"it has {sorted(k[:16] for k in promoted)}")
    record = promoted[checkpoint_sha]
    ids, embeddings = verified_index(record, str(path))
    if len(set(ids)) != len(ids):
        # A consumer building a uid -> column map from a list with repeats
        # silently loses every earlier duplicate, and the loss shows up as a
        # slightly lower R@k that nothing explains. G4 rejects duplicates too;
        # this is here because the loader is what n15 trusts, and a trust
        # boundary that relies on an upstream gate having run is not one.
        raise ValueError(f"{record['uri']} repeats "
                         f"{len(ids) - len(set(ids))} asset id(s)")
    return record, ids, embeddings


GATE_RECORD_PATH = paths.LOGS / "gates" / "G4_gallery_freeze.yaml"


def promote(gate_record_path: Path | None = None) -> int:
    """[n12] Late commit: publish the artifact G4 actually verified.

    The digest is compared rather than trusted. A staging index rebuilt between
    verification and promotion would otherwise be published under G4's verdict
    without G4 having seen it -- which is the failure the two-step exists to
    prevent, so promotion cannot be the step that reintroduces it.

    [2026-08-30] This took ``gate_passed: bool`` from a ``--gate-passed`` CLI
    flag, so an operator typing a word stood in for the evidence the gate exists
    to produce -- and nothing in the promoted registry recorded which verdict,
    if any, had been asserted. The flag is GONE rather than ignored: an argument
    that no longer decides anything still reads as "the gate is asserted here",
    and the next person to add a caller would pass it.

    Promotion now reads G4's own record and re-verifies all three identities it
    contains -- staging record bytes, index bytes, checkpoint sha -- against
    what is on disk NOW. Verifying one and inferring the others is exactly the
    hole this closes: the staging-record digest proves the record is the one G4
    read, and the index digest proves the vectors are the ones G4 scored, and
    neither implies the other.

    [L1-GATE-NORECORD] A missing gate record is NOT PASSED, never pass-by-default.

    Returns 0 on success, 3 when the gate evidence is absent or does not say
    PASS, 2 when an artifact disagrees with what the gate recorded.
    """
    if not STAGING_PATH.exists():
        print(f"{STAGING_PATH} not found -- run n11 first", flush=True)
        return 2
    staging = json.loads(STAGING_PATH.read_text())

    gate_path = Path(gate_record_path) if gate_record_path else GATE_RECORD_PATH
    if not gate_path.exists():
        print(f"{gate_path} not found -- G4_gallery_freeze has not run; "
              "a missing gate record is not a pass", flush=True)
        return 3
    try:
        gate = yaml.safe_load(gate_path.read_text())
    except yaml.YAMLError as exc:
        print(f"{gate_path} is not readable YAML: {exc}", flush=True)
        return 3
    if not isinstance(gate, dict) or gate.get("gate_id") != "G4_gallery_freeze":
        print(f"{gate_path} is not a G4_gallery_freeze record "
              f"(gate_id={gate.get('gate_id') if isinstance(gate, dict) else None})",
              flush=True)
        return 3
    if gate.get("verdict") != "PASS":
        print(f"{gate_path} records verdict {gate.get('verdict')!r} "
              f"(rc {gate.get('rc')}); refusing to promote", flush=True)
        return 3
    if gate.get("is_terminal") is not True:
        print(f"{gate_path} is not a terminal record; refusing to promote",
              flush=True)
        return 3

    # The record G4 read must be the record on disk now, byte for byte. This
    # subsumes every field inside it -- and the two digests below are still
    # compared, because "the file did not change" and "these bytes are the
    # index G4 scored" are different claims and only one of them is about the
    # .npz.
    staging_now = hashlib.sha256(STAGING_PATH.read_bytes()).hexdigest()
    if gate.get("staging_record_sha256") != staging_now:
        print(f"{STAGING_PATH} changed since G4 verified it "
              f"({str(gate.get('staging_record_sha256'))[:12]} -> "
              f"{staging_now[:12]}); refusing", flush=True)
        return 2

    for stage1_sha, record in staging.items():
        if gate.get("stage1_checkpoint_sha256") != record["stage1_checkpoint_sha256"]:
            print(f"G4 verified an index built from checkpoint "
                  f"{str(gate.get('stage1_checkpoint_sha256'))[:12]}... but the "
                  f"staging record names "
                  f"{record['stage1_checkpoint_sha256'][:12]}...; refusing",
                  flush=True)
            return 2
        # Hashed ONCE, compared against BOTH authorities. They are different
        # questions -- "is this what G4 scored?" and "is this what the staging
        # record describes?" -- and only the second one existed before.
        on_disk = hashlib.sha256(Path(record["uri"]).read_bytes()).hexdigest()
        if gate.get("index_sha256") != on_disk:
            print(f"{record['uri']} is not the artifact G4 verified "
                  f"({str(gate.get('index_sha256'))[:12]} -> {on_disk[:12]}); "
                  "refusing", flush=True)
            return 2
        if on_disk != record["sha256"]:
            print(f"{record['uri']} changed since staging "
                  f"({record['sha256'][:12]} -> {on_disk[:12]}); refusing", flush=True)
            return 2

    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()

    promoted = json.loads(PROMOTED_PATH.read_text()) if PROMOTED_PATH.exists() else {}
    for stage1_sha, record in staging.items():
        existing = promoted.get(stage1_sha)
        if existing and existing["sha256"] != record["sha256"]:
            # write_once: an index for a given checkpoint is a fact, and a
            # second differing one means two different artifacts claim the same
            # provenance.
            print(f"gallery_index[{stage1_sha[:12]}] already published with a "
                  f"different digest; refusing to overwrite", flush=True)
            return 2
        promoted[stage1_sha] = {
            **record,
            # Which verdict published this, and the bytes of that verdict. A
            # promoted index whose record cannot name the gate record that
            # cleared it is back to being an assertion.
            "gate_record_uri": str(gate_path),
            "gate_record_sha256": gate_sha,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    _write(PROMOTED_PATH, promoted)
    print(f"promoted {len(staging)} index/indices -> {PROMOTED_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stage1", "promote", "stage2"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage1-ckpt-record", default=None,
                    help="Stage 1 checkpoint record to build from. Defaults to "
                         "the canonical stage1_ckpt.json; name a run's own "
                         "record to promote a sweep arm without copying it "
                         "over the canonical file.")
    args = ap.parse_args()

    if args.mode == "promote":
        # No flag. `--gate-passed` used to let the operator assert the verdict;
        # promotion reads G4's record instead, and an unrecognised argument is
        # the loudest possible way to tell an old command that it no longer
        # means what it says.
        return promote()

    import torch
    from metafind.train.stage1 import (
        build_model, load_protocols, load_stage1_checkpoint,
        load_stage1_model_config, effective_stage1_model_inputs, stage1_backbone_kwargs)
    from metafind.models.ulip_backbone import (
        BackboneConfig, ULIPBackbone, prepare_depth_shell)

    ckpt_record = load_checkpoint_record(args.stage1_ckpt_record)
    ckpt_record = load_stage1_model_config(ckpt_record["uri"], ckpt_record)
    encoding, training, hyperparameters = effective_stage1_model_inputs(
        ckpt_record, *load_protocols())
    # Match checkpoint coverage before restoring; freeze only after loading.
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope=training["train_scope"],
                                           **stage1_backbone_kwargs(ckpt_record)))
    query_backbone = (backbone.clone_point_path()
                      if training["tower_sharing"] == "fully_separate" else None)
    model, loss_fn = build_model(encoding, training, hyperparameters)
    if training.get("freeze_gallery"):
        model.freeze_gallery(True)            # match the checkpoint's requires_grad set before loading
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt_record["uri"]),
                           query_backbone=query_backbone)
    del query_backbone  # an index never evaluates the query point path
    # Restore first, THEN freeze. The checkpoint's point-encoder section can only
    # land in a backbone whose point encoder is trainable, but the index must be
    # built with it frozen and in eval: ULIP-2's PointBERT config sets
    # drop_path_rate=0.1, so a point encoder left in train() applies stochastic
    # depth and the index comes out non-deterministic. Nothing downstream would
    # catch that -- the vectors would have the right shape and the wrong values.
    backbone.set_train_scope("fuser_only")
    assert backbone.is_frozen(), "the backbone is still trainable or in train mode"
    model.to(args.device).eval()
    model.freeze_gallery(True)
    encoder_sha = gallery_encoder_sha256(backbone, model, include_buffers=True)

    node = "n11_gallery_index_staging" if args.mode == "stage1" else "n11b_stage2_gallery_index"
    started = time.time()

    with runlog.run_progress(node):
        if args.mode == "stage1":
            from metafind.data.splits import corpus_uids
            ids = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
            ids = sorted(corpus_uids(ids))            # [D-3b] train + val + test
            if args.limit:
                ids = ids[: args.limit]
            vectors = []
            for i, uid in enumerate(ids):
                cached = np.load(paths.EMBEDDINGS / f"{uid}.npz")
                cloud = np.load(paths.POINTCLOUDS / f"{uid}.npz")
                pc = np.concatenate([cloud["xyz"], cloud["rgb"]], axis=1)[None]
                with torch.no_grad():
                    pc_vec = backbone.encode_pc(torch.from_numpy(pc.astype(np.float32)))
                    embeds = {
                        "text": torch.from_numpy(cached["text"].astype(np.float32))[None].to(args.device),
                        "image": torch.from_numpy(cached["image"].astype(np.float32))[None].to(args.device),
                        "pc": pc_vec,
                    }
                    # [2.6] the gallery encoder is modality-complete: no mask.
                    vectors.append(model.gallery(embeds)[0].cpu().numpy())
                if (i + 1) % 2000 == 0:
                    print(f"  [{i + 1:6d}/{len(ids)}]", flush=True)
            # Named by the checkpoint that produced it, NOT a fixed live path.
            # `gallery_index.npz` was overwritten by every rebuild while the
            # promoted registry kept a record per stage1_sha -- so an older
            # record's `sha256` stopped matching the bytes at its own `uri`, and
            # the write-once guarantee that promotion exists to provide was not
            # actually held by anything on disk.
            record = build_index(
                np.stack(vectors), ids,
                paths.OUTPUTS / f"gallery_index_{ckpt_record['sha256'][:16]}.npz")
            record["gallery_encoder_sha256"] = encoder_sha
            record["gallery_encoder_hash_version"] = GALLERY_ENCODER_HASH_VERSION
            record["gallery_forward_config"] = gallery_forward_config(backbone, model)
            record["gallery_encoder_hash_includes_buffers"] = True
            # [CODEX MAJOR 2026-08-30] Stated in the record, not only in the
            # dict key: Stage 2 reads the record, and a key is not a field.
            record["stage1_checkpoint_sha256"] = ckpt_record["sha256"]
            record["stage1_ckpt_record"] = str(
                args.stage1_ckpt_record
                or paths.CHECKPOINTS / "stage1_ckpt.json")
            _write(STAGING_PATH, {ckpt_record["sha256"]: record})
            print(f"\nstaged {record['count']:,} x {record['dim']} "
                  f"-> {STAGING_PATH}")
        else:
            # [2.6] "The gallery encoder is trained to be modality-complete." Until
            # DL-104 that meant: the 28 ProcTHOR assets without depth (transparent
            # materials, "every view was empty") were excluded rather than admitted
            # with a gap, because a presence mask on the gallery side had no
            # trained token behind it. The declaration below turns "complete" into
            # "has every declared modality", and the absent slot is excluded from
            # the fusion, not filled.
            # [DL-104, Kyzen 2026-09-07] The protocol now DECLARES which modalities
            # a ProcTHOR asset carries (`asset_modalities`, text + image). Only
            # those are encoded, the gallery fusion runs with the other slot
            # excluded (GalleryTower.forward(declared=...)), and an asset is
            # excluded only when a DECLARED modality is missing. A protocol
            # without the field is refused (ESSGNN REVIEWER MAJOR 1).
            s2_protocol = json.loads((paths.OUTPUTS / "stage2_protocol.json").read_text())
            if "asset_modalities" not in s2_protocol:
                raise ValueError(f"{paths.OUTPUTS / 'stage2_protocol.json'} carries no "
                                 "`asset_modalities`; it predates DL-104. Re-run n09b "
                                 "(resolve_stage2) before building the Stage 2 index.")
            declared = tuple(s2_protocol["asset_modalities"])
            encoder_sha = gallery_encoder_sha256(backbone, model, include_buffers=True,
                                                 declared_modalities=declared)
            source_identity = capture_stage2_gallery_sources(declared, limit=args.limit)
            mods = source_identity["modality_records"]
            ids, vectors, excluded = [], [], []
            # The raw modality vectors are kept beside the fused gallery vector.
            # Stage 2 freezes the whole ULIP-2 backbone, so for every asset these
            # numbers never change again; recomputing them per training step (11
            # ViT-bigG image forwards per sample) was what made one Stage 2 epoch
            # cost days. Same backbone, same inputs, same eval mode as the fused
            # vector below, so a lookup returns exactly what the per-step encode
            # used to return.
            raw = {m: [] for m in declared}
            for source_record in mods:
                rec = json.loads(verified_source_bytes(source_record))
                source = source_identity["encoded_inputs"].get(str(rec["asset_id"]))
                # The snapshot already validated the canonical text and decided
                # eligibility. n07b's historical text is not a current input.
                gaps = _missing_stage2_modalities(rec, [m for m in declared if m != "text"])
                if gaps:
                    excluded.append({"asset_id": rec["asset_id"], "missing": gaps,
                                     "reason": rec.get("pointcloud_missing_reason")
                                     if "pc" in gaps else None})
                    continue
                if source is None:
                    raise ValueError(f"Stage 2 source snapshot lacks asset {rec['asset_id']!r}")
                with torch.no_grad():
                    embeds = {}
                    if "text" in declared:
                        embeds["text"] = backbone.encode_text([source["text"]])
                    if "image" in declared:
                        view_vecs = backbone.encode_image(torch.stack([
                            backbone.preprocess(Image.open(io.BytesIO(
                                verified_source_bytes(v))).convert("RGB"))
                            for v in source["images"]]))
                        embeds["image"] = view_vecs.mean(dim=0, keepdim=True)
                    if "pc" in declared:
                        # [P0-4] pc_norm happens INSIDE prepare_depth_shell: n07b
                        # stores world-frame points (asset lifted to y=40 m), n03
                        # stores unit-normalised ones, and the checkpoint was
                        # trained on the latter. The grey channel is there because
                        # the shell has no colour, not because grey is a measurement.
                        with np.load(io.BytesIO(verified_source_bytes(source["pointcloud"]))) as cloud_file:
                            cloud = cloud_file["xyz"].astype(np.float32)
                        embeds["pc"] = backbone.encode_pc(torch.from_numpy(prepare_depth_shell(cloud)))
                    vectors.append(model.gallery(
                        embeds, declared=None if len(declared) == 3 else declared
                    )[0].cpu().numpy())
                    for m in declared:
                        raw[m].append(embeds[m][0].float().cpu().numpy())
                ids.append(rec["asset_id"])
                if len(ids) % 200 == 0:
                    print(f"  [{len(ids):5d}/{len(mods)}]", flush=True)

            # Refuse concurrent source replacement before publishing new vectors.
            verify_stage2_gallery_sources({"source_identity": source_identity,
                "modality_completeness": {"declared_modalities": list(declared)}})
            record = build_index(
                np.stack(vectors), ids,
                paths.OUTPUTS / f"stage2_gallery_{ckpt_record['sha256'][:16]}.npz",
                extra={k: np.stack(v) for k, v in raw.items()},
                source_identity=source_identity)
            record.update({
                "asset_ids": ids,
                # Names the extra arrays so a reader can tell an index that
                # carries the raw modality vectors from one built before they
                # were stored, instead of finding out with a KeyError.
                "raw_modality_arrays": sorted(raw),
                "embedding_dim": record["dim"],
                "n_assets": record["count"],
                "gallery_encoder_sha256": encoder_sha,
                "gallery_encoder_hash_version": GALLERY_ENCODER_HASH_VERSION,
                "gallery_forward_config": gallery_forward_config(backbone, model, declared),
                "gallery_encoder_hash_includes_buffers": True,
                "modality_completeness": {
                    "declared_modalities": list(declared),
                    "complete": len(ids),
                    "excluded_missing_declared": excluded,
                },
                # [CODEX MAJOR 2026-08-30] Stage 2's [G6] comment claimed the
                # index and the checkpoint were compared. Nothing compared them,
                # because the index never said which checkpoint made it.
                "stage1_checkpoint_sha256": ckpt_record["sha256"],
            })
            _write(STAGE2_PATH, record)
            print(f"\nstage2 index: {record['n_assets']:,} assets, "
                  f"{len(excluded)} excluded for missing a declared modality {declared} "
                  f"-> {STAGE2_PATH}")

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       assets_encoded=record["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
