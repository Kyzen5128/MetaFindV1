"""Adapt a completed I-Design plan and explicit observations to a scene request.

No planner, encoder, template selection, asset lookup, or rendering runs here.
Only fresh-room generation is supported: every object slot is a future query.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from metafind.scene.placement import _identity, _number, _publish, _vector

REQUEST_SCHEMA = "metafind.scene_request.v1"
PRIORS = ("south_wall", "north_wall", "east_wall", "west_wall", "middle of the room", "ceiling")
BASE_FIELDS = {"schema", "provenance", "stage1_record", "stage2_record", "variant",
               "gallery_registry", "gallery_ids", "asset_texts", "mode", "use_layout"}
BASE_PATHS = ("stage1_record", "stage2_record", "gallery_registry", "asset_texts", "semantic_cache")
OBJECT_PREPOSITIONS = {"on", "under", "above", "left of", "right of", "in front", "behind"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _snapshot(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=invalid_constant)
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "content": value}


def _path(value, base):
    _identity(value, "input path")
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required input is absent: {path}")
    return str(path)


def _prior_expectations(dims):
    # Exact geometry contract of pinned I-Design utils.get_room_priors(), not
    # room rendering geometry. Zero-thickness priors are not retrieval assets.
    x, y, z = dims
    return {
        "south_wall": ("wall", [x/2, 0, z/2], [x, 0, z], 0),
        "north_wall": ("wall", [x/2, y, z/2], [x, 0, z], 180),
        "east_wall": ("wall", [x, y/2, z/2], [y, 0, z], 270),
        "west_wall": ("wall", [0, y/2, z/2], [y, 0, z], 90),
        "middle of the room": ("floor", [x/2, y/2, 0], [x, y, 0], 0),
        "ceiling": ("ceiling", [x/2, y/2, z], [x, y, 0], 0),
    }


def _scene(scene, dims):
    if not isinstance(scene, list) or not scene:
        raise ValueError("I-Design scene must be a nonempty flat list")
    ids = []
    for item in scene:
        if not isinstance(item, dict):
            raise ValueError("scene entries must be objects")
        _identity(item.get("new_object_id"), "new_object_id")
        ids.append(item["new_object_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("scene slot/prior IDs must be unique")
    priors = {item["new_object_id"]: item for item in scene if item["new_object_id"] in PRIORS}
    if set(priors) != set(PRIORS):
        raise ValueError("raw I-Design output must contain exactly the six room priors")
    for name, (kind, xyz, sizes, yaw) in _prior_expectations(dims).items():
        item = priors[name]
        actual = [item["position"][a] for a in ("x", "y", "z")]
        actual += [item["size_in_meters"][a] for a in ("length", "width", "height")]
        actual += [item["rotation"]["z_angle"]]
        for value in actual:
            _number(value, "room prior geometry")
        if item.get("itemType") != kind or any(not math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-9)
                                               for a, b in zip(actual, xyz+sizes+[yaw])):
            raise ValueError(f"room prior disagrees with sidecar dimensions: {name}")
    slots = [item for item in scene if item["new_object_id"] not in PRIORS]
    if not slots:
        raise ValueError("scene has no object queries after separating room priors")
    slot_ids = {s["new_object_id"] for s in slots}
    for slot in slots:
        for a in ("x", "y", "z"):
            _number(slot["position"][a], "object position")
        _number(slot["rotation"]["z_angle"], "object yaw")
        for a in ("length", "width", "height"):
            _number(slot["size_in_meters"][a], "object size", positive=True)
        relations = slot["placement"]
        for group, key, allowed_ids, prepositions in (
            ("objects_in_room", "object_id", slot_ids, OBJECT_PREPOSITIONS),
            ("room_layout_elements", "layout_element_id", set(PRIORS), {"on", "in the corner"}),
        ):
            if not isinstance(relations[group], list):
                raise ValueError(f"{group} must be an explicit relation list")
            for relation in relations[group]:
                if not isinstance(relation, dict) or relation.get(key) not in allowed_ids \
                        or relation[key] == slot["new_object_id"] or relation.get("preposition") not in prepositions:
                    raise ValueError(f"invalid {group} reference for {slot['new_object_id']}")
    return slots, priors


def build_request(scene_path: Path, sidecar_path: Path, query_modalities_path: Path,
                  base_request_path: Path, out_path: Path) -> Path:
    """Write a new scene request; inputs are validated before any publication."""
    out_path = Path(out_path)
    if out_path.exists() or out_path.with_name(out_path.name + ".part").exists():
        raise FileExistsError(f"refusing to overwrite {out_path} or its partial output")
    sources = {}
    values = {}
    for name, path in (("scene", scene_path), ("sidecar", sidecar_path),
                       ("query_modalities", query_modalities_path), ("base_request", base_request_path)):
        values[name], sources[name] = _snapshot(path)
    sidecar = values["sidecar"]
    if not isinstance(sidecar, dict):
        raise ValueError("planner sidecar must be an object")
    _identity(sidecar.get("scene_id"), "sidecar scene_id")
    _vector(sidecar["room_dimensions"], 3, "room dimensions", positive=True)
    if "status" in sidecar and sidecar["status"] not in ("complete", "success"):
        raise ValueError("sidecar does not describe a completed scene")
    if "scene_graph_sha256" in sidecar and sidecar["scene_graph_sha256"] != sources["scene"]["sha256"]:
        raise ValueError("sidecar scene_graph_sha256 mismatch")
    slots, priors = _scene(values["scene"], sidecar["room_dimensions"])
    mapping = values["query_modalities"]
    ids = [s["new_object_id"] for s in slots]
    if not isinstance(mapping, dict) or set(mapping) != set(ids):
        raise ValueError("query modality mapping must cover every object slot exactly; no priors/extras")
    queries = []
    for slot in slots:
        obs = mapping[slot["new_object_id"]]
        if not isinstance(obs, dict) or not obs or set(obs) - {"text", "images", "pointcloud"}:
            raise ValueError("each slot requires explicit text/images/pointcloud modalities only")
        query = {"slot": copy.deepcopy(slot)}
        if "text" in obs:
            _identity(obs["text"], "query text")
            query["text"] = obs["text"]
        if "images" in obs:
            if not isinstance(obs["images"], list) or not obs["images"]:
                raise ValueError("images must be a nonempty ordered path list")
            query["images"] = [_path(p, Path(query_modalities_path).resolve().parent) for p in obs["images"]]
        if "pointcloud" in obs:
            query["pointcloud"] = _path(obs["pointcloud"], Path(query_modalities_path).resolve().parent)
        queries.append(query)
    base = values["base_request"]
    if not isinstance(base, dict) or not BASE_FIELDS <= set(base) or set(base)-BASE_FIELDS-{"semantic_cache"} \
            or base["schema"] != REQUEST_SCHEMA:
        raise ValueError("base request needs the explicit prepare fields; initial_graph/queries are forbidden")
    if not isinstance(base["provenance"], dict) or not base["provenance"] or "idesign_adapter" in base["provenance"]:
        raise ValueError("base provenance must be nonempty and cannot already claim idesign_adapter")
    if base["mode"] not in ("iterative", "parallel") or not isinstance(base["use_layout"], bool):
        raise ValueError("base requires explicit mode and use_layout")
    _identity(base["variant"], "variant")
    gallery = base["gallery_ids"]
    if gallery != "all":
        if not isinstance(gallery, list) or not gallery:
            raise ValueError("gallery_ids must be 'all' or a nonempty unique list")
        for uid in gallery:
            _identity(uid, "gallery UID")
        if len(set(gallery)) != len(gallery):
            raise ValueError("gallery IDs must be unique")
    result = copy.deepcopy(base)
    for field in BASE_PATHS:
        if field in result:
            result[field] = _path(result[field], Path(base_request_path).resolve().parent)
    result["initial_graph"] = {"room_id": sidecar["scene_id"], "nodes": []}
    result["queries"] = queries
    result["provenance"]["idesign_adapter"] = {
        "sources": sources, "query_order": ids, "query_source": "explicit_caller_modalities",
        "initial_graph_rule": "fresh_room_no_preexisting_assets",
        "room_dimensions": sidecar["room_dimensions"], "room_priors": priors,
        "object_count": len(slots), "room_prior_count": len(priors),
        "render_configuration": "not_provided_by_this_adapter",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Freeze what was actually read, not a later revision under the same path.
    for source in sources.values():
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("source changed while building the scene request")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _publish(out_path, result)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--query-modalities", required=True, type=Path)
    parser.add_argument("--base-request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(build_request(args.scene, args.sidecar, args.query_modalities, args.base_request, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
