"""Freeze explicit scene placement inputs and invoke an isolated CPU Blender.

This transports U-18 planned slots; it does not infer room geometry, cameras,
asset front directions, or the unpublished MetaFind evaluation scene list.
Only self-contained static raw GLBs are supported by this first bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

SCHEMA = "metafind.scene_placement.v1"
FRAME = "raw_gltf_y_up"
SCRIPT = Path(__file__).with_name("blender_place.py")
CHOICES = {"placement": "idesign_planned_bbox_per_axis_fit", "anchor": "bbox_center",
           "raw_mesh_yaw_offset_degrees": 180, "scene_unit_scale_length": 1.0,
           "imported_cameras_lights": "disabled", "true_asset_front": "UNKNOWN"}


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _publish(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(part, path)
    part.unlink()


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"invalid {label}: {value}")
    return value


def _vector(value, length, label, *, positive=False, color=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} requires {length} values")
    for x in value:
        _number(x, label, positive=positive)
        if color and not 0 <= x <= 1:
            raise ValueError(f"{label} must be in [0,1]")


def _matrix(value, label):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} requires a 4x4 matrix")
    for row in value:
        _vector(row, 4, label)
    if value[3] != [0, 0, 0, 1]:
        raise ValueError(f"{label} must be affine")
    # Cameras/lights are rigid poses, not arbitrary shears or scaled frames.
    for i in range(3):
        for j in range(3):
            dot = sum(value[k][i] * value[k][j] for k in range(3))
            if not math.isclose(dot, float(i == j), abs_tol=1e-6):
                raise ValueError(f"{label} rotation must be orthonormal")
    a, b, c = [row[:3] for row in value[:3]]
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if not math.isclose(det, 1., abs_tol=1e-6):
        raise ValueError(f"{label} rotation must preserve orientation")


def _identity(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")


def _provenance(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("explicit nonempty provenance is required")


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{label} requires exactly these fields: {names}")


def _slot(slot):
    _identity(slot["new_object_id"], "slot id")
    for a in ("x", "y", "z"):
        _number(slot["position"][a], "slot position")
    _number(slot["rotation"]["z_angle"], "slot yaw")
    for a in ("length", "width", "height"):
        _number(slot["size_in_meters"][a], "slot size", positive=True)
    if not isinstance(slot["placement"]["objects_in_room"], list):
        raise ValueError("slot needs explicit placement relations")


def _room(room, room_id):
    _fields(room, "room_id dimensions provenance surfaces", "room config")
    if room["room_id"] != room_id:
        raise ValueError("room identity differs from composition")
    _vector(room["dimensions"], 3, "room dimensions", positive=True)
    _provenance(room["provenance"])
    if not isinstance(room["surfaces"], list):
        raise ValueError("room surfaces must be explicit; [] is permitted")
    ids = []
    for surface in room["surfaces"]:
        _fields(surface, "id vertices rgba roughness", "room surface")
        _identity(surface["id"], "surface id")
        ids.append(surface["id"])
        if not isinstance(surface["vertices"], list) or len(surface["vertices"]) < 3:
            raise ValueError("surface needs at least three ordered vertices")
        for vertex in surface["vertices"]:
            _vector(vertex, 3, "surface vertex")
        _vector(surface["rgba"], 4, "surface rgba", color=True)
        _number(surface["roughness"], "surface roughness")
        if not 0 <= surface["roughness"] <= 1:
            raise ValueError("roughness must be in [0,1]")
    if len(set(ids)) != len(ids):
        raise ValueError("surface IDs must be unique")


def _render(config):
    if config is None:
        return
    _fields(config, "provenance engine device threads samples seed resolution transparent world color_management cameras lights", "render config")
    _fields(config["world"], "color strength", "world config")
    _fields(config["color_management"], "view_transform look exposure gamma", "color management")
    _provenance(config["provenance"])
    if config["engine"] != "CYCLES" or config["device"] != "CPU":
        raise ValueError("this bridge explicitly supports CYCLES CPU only")
    for key in ("threads", "samples"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    _vector(config["resolution"], 2, "resolution", positive=True)
    if any(not isinstance(x, int) for x in config["resolution"]):
        raise ValueError("resolution must be integer")
    if not isinstance(config["transparent"], bool):
        raise ValueError("transparent must be explicit boolean")
    _vector(config["world"]["color"], 3, "world color", color=True)
    _number(config["world"]["strength"], "world strength", nonnegative=True)
    cm = config["color_management"]
    for k in ("view_transform", "look"):
        _identity(cm[k], k)
    _number(cm["exposure"], "exposure")
    _number(cm["gamma"], "gamma", positive=True)
    if not isinstance(config["cameras"], list) or not config["cameras"]:
        raise ValueError("at least one explicit camera is required")
    if not isinstance(config["lights"], list):
        raise ValueError("lights must be explicit; [] is permitted")
    for kind in ("cameras", "lights"):
        ids = []
        for obj in config[kind]:
            _fields(obj, "id matrix_world lens_mm sensor_width_mm clip_start clip_end" if kind == "cameras"
                    else "id type shape matrix_world energy color size", kind)
            _identity(obj["id"], kind + " id")
            ids.append(obj["id"])
            _matrix(obj["matrix_world"], kind)
            if kind == "cameras":
                for k in ("lens_mm", "sensor_width_mm", "clip_start", "clip_end"):
                    _number(obj[k], k, positive=True)
                if obj["clip_start"] >= obj["clip_end"]:
                    raise ValueError("camera clip range is empty")
            else:
                if obj["type"] != "AREA" or obj["shape"] not in ("DISK", "SQUARE"):
                    raise ValueError("only explicit DISK/SQUARE AREA lights are supported")
                _number(obj["energy"], "light energy", nonnegative=True)
                _number(obj["size"], "light size", positive=True)
                _vector(obj["color"], 3, "light color", color=True)
        if len(set(ids)) != len(ids):
            raise ValueError(f"{kind} IDs must be unique")


def _raw_glb(raw):
    if len(raw) < 20:
        raise ValueError("mesh is not a GLB")
    magic, version, size, chunk_len, chunk_type = struct.unpack("<4sIIII", raw[:20])
    if magic != b"glTF" or version != 2 or size != len(raw) or chunk_type != 0x4E4F534A:
        raise ValueError("mesh must be a complete glTF 2 binary")
    data = json.loads(raw[20:20+chunk_len])
    if data.get("animations") or data.get("skins"):
        raise ValueError("animated/skinned GLBs need an explicit pose policy; static GLBs only")
    for entry in data.get("buffers", []) + data.get("images", []):
        if "uri" in entry and not entry["uri"].startswith("data:"):
            raise ValueError("GLB has external dependencies not bound by its hash")


def _artifact(record, base):
    p = Path(record["path"])
    p = p if p.is_absolute() else base / p
    raw = p.read_bytes()
    if _hash(raw) != record["sha256"]:
        raise ValueError(f"artifact hash mismatch: {p}")
    return raw


def _snapshot(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": _hash(raw), "raw_json": raw.decode("utf-8"), "content": json.loads(raw)}


def prepare_placement(composition_path: Path, asset_manifest_path: Path,
                      room_config_path: Path, out_dir: Path, *, render_config_path: Path | None = None) -> Path:
    """Copy verified static GLBs/annotations into a new immutable input bundle."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    sources = {"composition": _snapshot(composition_path), "assets": _snapshot(asset_manifest_path),
               "room": _snapshot(room_config_path)}
    if render_config_path is not None:
        sources["render"] = _snapshot(render_config_path)
    composition = sources["composition"]["content"]
    assets = sources["assets"]["content"]
    if composition.get("schema") != "metafind.scene_composition.v1" or composition.get("status") != "complete":
        raise ValueError("requires a completed versioned composition result")
    _provenance(assets["provenance"])
    room = sources["room"]["content"]
    render = sources.get("render", {}).get("content")
    _room(room, composition["room_id"])
    _render(render)
    nodes = composition["final_graph"]["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("final graph needs placed nodes")
    prepared, ids, buffers = [], [], {}
    for node in nodes:
        _slot(node["slot"])
        ids.append(node["slot"]["new_object_id"])
        uid = node["asset_id"]
        _identity(uid, "asset id")
        if uid not in assets["assets"]:
            raise ValueError(f"asset manifest lacks {uid}")
        record = assets["assets"][uid]
        if record["mesh"].get("frame") != FRAME:
            raise ValueError("requires declared raw_gltf_y_up mesh; never apply raw yaw to a corrected GLB")
        if uid not in buffers:
            base = Path(asset_manifest_path).resolve().parent
            mesh = _artifact(record["mesh"], base)
            _raw_glb(mesh)
            annotation = _artifact(record["annotation"], base)
            ann = json.loads(annotation)
            if not isinstance(ann, dict) or ann.get("uid", uid) != uid:
                raise ValueError(f"annotation UID mismatch for {uid}")
            buffers[uid] = (mesh, annotation)
        prepared.append({"asset_id": uid, "slot": node["slot"]})
    if len(set(ids)) != len(ids):
        raise ValueError("slot IDs must be unique")
    out_dir.mkdir(parents=True, exist_ok=False)
    asset_dir = out_dir / "assets"
    asset_dir.mkdir()
    frozen_assets = {}
    for i, (uid, (mesh, annotation)) in enumerate(buffers.items()):
        frozen_assets[uid] = {}
        for label, raw, suffix in (("mesh", mesh, "glb"), ("annotation", annotation, "json")):
            p = asset_dir / f"{i:05d}.{suffix}"
            with p.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            frozen_assets[uid][label] = {"path": str(p.relative_to(out_dir)), "sha256": _hash(raw)}
        frozen_assets[uid]["mesh"]["frame"] = FRAME
    result = {"schema": SCHEMA, "status": "frozen", "sources": sources,
              "room": room, "render": render, "instances": prepared, "assets": frozen_assets,
              "choices": CHOICES,
              "implementation": {"placement_sha256": _hash(Path(__file__).read_bytes()),
                                 "blender_place_sha256": _hash(SCRIPT.read_bytes())}}
    dest = out_dir / "placement.json"
    _publish(dest, result)
    return dest


def load_placement(path: Path, *, expected_sha256=None) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and _hash(raw) != expected_sha256:
        raise ValueError("placement manifest changed before Blender consumed it")
    data = json.loads(raw)
    if data.get("schema") != SCHEMA or data.get("status") != "frozen":
        raise ValueError("requires a frozen placement manifest")
    if data["choices"] != CHOICES:
        raise ValueError("placement choices differ from the implemented protocol")
    if data["implementation"] != {"placement_sha256": _hash(Path(__file__).read_bytes()),
                                   "blender_place_sha256": _hash(SCRIPT.read_bytes())}:
        raise ValueError("placement implementation changed; prepare a new bundle")
    for source in data["sources"].values():
        if _hash(source["raw_json"].encode("utf-8")) != source["sha256"] or json.loads(source["raw_json"]) != source["content"]:
            raise ValueError("source JSON content differs from its recorded bytes")
    composition = data["sources"]["composition"]["content"]
    if data["instances"] != composition["final_graph"]["nodes"]:
        raise ValueError("instances differ from the source composition")
    if data["room"] != data["sources"]["room"]["content"] or data["render"] != data["sources"].get("render", {}).get("content"):
        raise ValueError("configuration differs from frozen source")
    _room(data["room"], composition["room_id"])
    _render(data["render"])
    ids = []
    for node in data["instances"]:
        _slot(node["slot"])
        ids.append(node["slot"]["new_object_id"])
        uid = node["asset_id"]
        record = data["assets"][uid]
        if record["mesh"]["frame"] != FRAME:
            raise ValueError("frozen mesh frame is not raw_gltf_y_up")
        mesh = _artifact(record["mesh"], path.parent)
        _raw_glb(mesh)
        ann = json.loads(_artifact(record["annotation"], path.parent))
        if ann.get("uid", uid) != uid:
            raise ValueError("frozen annotation UID mismatch")
        original = data["sources"]["assets"]["content"]["assets"][uid]
        if any(record[k]["sha256"] != original[k]["sha256"] for k in ("mesh", "annotation")):
            raise ValueError("frozen assets differ from source hashes")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("placed slot IDs must be nonempty and unique")
    return data


def default_blender() -> Path:
    # Reuse the installation identity from the existing renderer; do not invoke
    # its object normalization, camera generation or GPU policy for a room.
    from metafind.data.render_blender import BLENDER_INSTALL
    matches = sorted(BLENDER_INSTALL.glob("blender-*/blender"))
    if len(matches) != 1:
        raise FileNotFoundError("pass --blender: expected exactly one installed Blender")
    return matches[0]


def run_placement(manifest_path: Path, out_dir: Path, *, blender: Path | None = None, timeout=900) -> Path:
    manifest_path, out_dir = Path(manifest_path).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    manifest_raw = manifest_path.read_bytes()
    load_placement(manifest_path, expected_sha256=_hash(manifest_raw))
    blender = Path(blender) if blender is not None else default_blender()
    if not blender.is_file() or not os.access(blender, os.X_OK):
        raise FileNotFoundError(f"Blender executable unavailable: {blender}")
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [str(blender.resolve()), "--background", "--factory-startup", "--threads", "1",
               "--python-exit-code", "3", "--python", str(SCRIPT), "--",
               "--manifest", str(manifest_path), "--manifest-sha256", _hash(manifest_raw), "--out-dir", str(out_dir)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    with (out_dir / "blender.log").open("x") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout, cwd=out_dir)
    result_path = out_dir / "result.json"
    if proc.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Blender placement failed (exit {proc.returncode}); see {out_dir / 'blender.log'}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or result["manifest_sha256"] != _hash(manifest_raw):
        raise RuntimeError("Blender returned an incomplete or unrelated result")
    for record in [result["blend"], *result["renders"]]:
        _artifact(record, out_dir)
    return result_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--composition", required=True, type=Path)
    prepare.add_argument("--assets", required=True, type=Path)
    prepare.add_argument("--room", required=True, type=Path)
    prepare.add_argument("--render", type=Path)
    prepare.add_argument("--out-dir", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--blender", type=Path)
    args = parser.parse_args(argv)
    result = (prepare_placement(args.composition, args.assets, args.room, args.out_dir, render_config_path=args.render)
              if args.command == "prepare" else run_placement(args.manifest, args.out_dir, blender=args.blender))
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
