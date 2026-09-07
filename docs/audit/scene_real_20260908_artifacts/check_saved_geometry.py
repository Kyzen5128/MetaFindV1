"""Independent NumPy/trimesh raw-GLB -> saved-world-vertex oracle and slot checks."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_bytes(record, base):
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    raw = path.read_bytes()
    assert digest(raw) == record["sha256"], path
    return raw


def imported_points(raw):
    # This independent source oracle supports the static, non-morph GLBs in
    # this diagnostic. It refuses a wider case instead of silently ignoring pose.
    length = struct.unpack_from("<I", raw, 12)[0]
    data = json.loads(raw[20:20 + length])
    assert not data.get("animations") and not data.get("skins")
    assert not any(p.get("targets") for m in data.get("meshes", []) for p in m.get("primitives", []))
    scene = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene", process=False)
    points = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        vertices = np.asarray(scene.geometry[geometry].vertices, dtype=np.float64)
        points.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    gltf_points = np.concatenate(points)
    # glTF Y-up -> Blender Z-up, derived directly from coordinate axes.
    return gltf_points[:, [0, 2, 1]] * np.array([1., -1., 1.])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--placement-result", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    result = json.loads(args.placement_result.read_bytes())
    extracted = json.loads((args.inspection / "extracted.json").read_bytes())
    assert result["manifest_sha256"] == digest(raw)
    assert extracted["blend_sha256"] == result["blend"]["sha256"]
    assert digest(source_bytes(result["blend"], args.placement_result.parent)) == extracted["blend_sha256"]
    assert extracted["frame"] == 0 and extracted["unit_scale"] == 1 and extracted["unit_system"] == "METRIC"
    for source in manifest["sources"].values():
        assert digest(source["raw_json"].encode()) == source["sha256"]
        assert json.loads(source["raw_json"]) == source["content"]
    assert manifest["instances"] == manifest["sources"]["composition"]["content"]["final_graph"]["nodes"]
    expected = {node["slot"]["new_object_id"]: node for node in manifest["instances"]}
    observed = {node["slot_id"]: node for node in extracted["instances"]}
    produced = {node["slot_id"]: node for node in result["instances"]}
    assert len(expected) == len(manifest["instances"]) == len(observed) == len(produced)
    assert expected.keys() == observed.keys() == produced.keys()
    assert set(extracted["surface_objects"].values()) == {s["id"] for s in manifest["room"]["surfaces"]}
    vertices_raw = (args.inspection / "world_vertices.npz").read_bytes()
    assert digest(vertices_raw) == extracted["vertices_sha256"]
    checks = []
    known_meshes = set(extracted["surface_objects"])
    with np.load(io.BytesIO(vertices_raw), allow_pickle=False) as arrays:
        for name, node in expected.items():
            evidence = observed[name]
            assert node["asset_id"] == evidence["asset_id"] == produced[name]["asset_id"]
            assert node["slot"] == produced[name]["slot"]
            names = {obj["name"] for obj in evidence["mesh_objects"]}
            assert not names.intersection(known_meshes)
            known_meshes.update(names)
            assert names == set(produced[name]["mesh_objects"])
            assert evidence["root"] == produced[name]["root"]
            actual = arrays[evidence["array"]]
            slot = node["slot"]
            origin = np.array([slot["position"][k] for k in ("x", "y", "z")])
            target = np.array([slot["size_in_meters"][k] for k in ("length", "width", "height")])
            angle = math.radians(slot["rotation"]["z_angle"] + 180.)
            c, s = math.cos(angle), math.sin(angle)
            yaw = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            local = (actual - origin) @ yaw
            center = (local.min(0) + local.max(0)) / 2
            size = np.ptp(local, axis=0)
            tolerance = 1e-5 * max(1., float(np.max(target)), float(np.max(np.abs(origin))))
            np.testing.assert_allclose(center, 0, atol=tolerance, rtol=0)
            np.testing.assert_allclose(size, target, atol=tolerance, rtol=0)
            asset = manifest["assets"][node["asset_id"]]
            assert asset["mesh"]["frame"] == "raw_gltf_y_up"
            source = imported_points(source_bytes(asset["mesh"], args.manifest.parent))
            source_center = (source.min(0) + source.max(0)) / 2
            scale = target / np.ptp(source, axis=0)
            oracle = ((source - source_center) * scale) @ yaw.T + origin
            # Importers may reorder/duplicate vertices for normals, so compare
            # the two complete point sets in both directions, not their counts.
            actual_to_source = float(cKDTree(oracle).query(actual, workers=1)[0].max())
            source_to_actual = float(cKDTree(actual).query(oracle, workers=1)[0].max())
            assert max(actual_to_source, source_to_actual) <= tolerance, (name, actual_to_source, source_to_actual)
            checks.append({"slot_id": name, "asset_id": node["asset_id"],
                           "actual_vertex_count": len(actual), "source_vertex_count": len(source),
                           "slot_frame_center": center.tolist(), "slot_frame_size": size.tolist(),
                           "expected_size": target.tolist(), "yaw_degrees": slot["rotation"]["z_angle"] + 180.,
                           "max_actual_to_source_distance": actual_to_source,
                           "max_source_to_actual_distance": source_to_actual, "absolute_tolerance": tolerance})
    assert known_meshes == set(extracted["scene_mesh_objects"])
    for render in result["renders"]:
        source_bytes(render, args.placement_result.parent)
    report = {"status": "passed", "classification": "Independent actual saved vertex geometry check, not visual quality or paper-scene validation",
              "checker_sha256": digest(Path(__file__).read_bytes()),
              "manifest_sha256": digest(raw), "blend_sha256": extracted["blend_sha256"],
              "oracle": "trimesh glTF hierarchy -> C(x,y,z)=(x,-z,y) -> actual vertex bbox center/peraxis fit -> Rz(slot_yaw+180) -> slot center; bidirectional nearest-vertex comparison",
              "no_production_bounds_or_transform_imports": True, "instances": checks}
    with (args.inspection / "verified_geometry.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
