"""Read saved evaluated geometry; no production placement/bounds imports or writes to the blend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.out_dir.mkdir(exist_ok=False)
    blend = Path(bpy.data.filepath).resolve()
    assert blend.is_file()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    roots = sorted((obj for obj in bpy.context.scene.objects if "slot_id" in obj),
                   key=lambda obj: obj["slot_id"])
    assert roots and len({obj["slot_id"] for obj in roots}) == len(roots)
    all_mesh_names = set()
    arrays, instances = {}, []
    for index, root in enumerate(roots):
        meshes = sorted((obj for obj in root.children_recursive if obj.type == "MESH"), key=lambda obj: obj.name)
        names = {obj.name for obj in meshes}
        assert names and not names.intersection(all_mesh_names)
        all_mesh_names.update(names)
        chunks, objects = [], []
        for obj in meshes:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                points = np.asarray([tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices],
                                    dtype=np.float64).reshape(-1, 3)
                assert np.isfinite(points).all()
                chunks.append(points)
                objects.append({"name": obj.name, "vertex_count": len(points),
                                "matrix_world": [list(row) for row in evaluated.matrix_world]})
            finally:
                evaluated.to_mesh_clear()
        points = np.concatenate(chunks)
        assert len(points)
        key = f"vertices_{index}"
        arrays[key] = points
        instances.append({"slot_id": root["slot_id"], "asset_id": root["asset_id"],
                          "root": root.name, "root_matrix_world": [list(row) for row in root.matrix_world],
                          "array": key, "mesh_objects": objects, "world_vertex_count": len(points)})
    np.savez_compressed(args.out_dir / "world_vertices.npz", **arrays)
    scene = bpy.context.scene
    record = {"classification": "Independent saved evaluated-vertex extraction; no render or scene save",
              "blend": str(blend), "blend_sha256": hashlib.sha256(blend.read_bytes()).hexdigest(),
              "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "vertices_sha256": hashlib.sha256((args.out_dir / "world_vertices.npz").read_bytes()).hexdigest(),
              "blender_version": bpy.app.version_string, "frame": scene.frame_current,
              "unit_system": scene.unit_settings.system, "unit_scale": scene.unit_settings.scale_length,
              "scene_mesh_objects": sorted(obj.name for obj in scene.objects if obj.type == "MESH"),
              "surface_objects": {obj.name: obj["surface_id"] for obj in scene.objects if "surface_id" in obj},
              "instances": instances}
    with (args.out_dir / "extracted.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"inspected_instances": len(instances), "vertices": sum(len(p) for p in arrays.values())}))


if __name__ == "__main__":
    main()
