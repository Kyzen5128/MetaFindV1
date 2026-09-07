"""Blender-only worker for a verified placement bundle; no model dependencies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

spec = importlib.util.spec_from_file_location("placement_contract", Path(__file__).with_name("placement.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def _bounds(objects, frame=None):
    frame = Matrix.Identity(4) if frame is None else frame
    lo, hi = Vector((math.inf,)*3), Vector((-math.inf,)*3)
    count = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            transform = frame @ evaluated.matrix_world
            # A rotated local AABB contains corners that need not belong to
            # the mesh. Scaling that approximation shrinks and offsets actual
            # geometry, while another AABB-based check falsely agrees with it.
            # Evaluated vertices also preserve a static GLB's morph weights.
            for vertex in mesh.vertices:
                point = transform @ vertex.co
                if not all(math.isfinite(x) for x in point):
                    raise ValueError("asset geometry contains nonfinite vertices")
                for axis in range(3):
                    lo[axis] = min(lo[axis], point[axis])
                    hi[axis] = max(hi[axis], point[axis])
                count += 1
        finally:
            evaluated.to_mesh_clear()
    if not count:
        raise ValueError("asset contains no mesh geometry")
    if not all(math.isfinite(x) for x in (*lo, *hi)) or any(x <= 0 for x in hi-lo):
        raise ValueError("asset bounding box must have three positive finite extents")
    return lo, hi


def _place(node, mesh_path, index):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    imported = set(bpy.data.objects) - before
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError("asset imported without meshes")
    for obj in imported:
        if obj.type in ("CAMERA", "LIGHT"):
            obj.hide_render = True
    bpy.context.view_layer.update()
    lo, hi = _bounds(meshes)
    center, extents = (lo+hi)/2, hi-lo
    slot = node["slot"]
    target = Vector([slot["size_in_meters"][k] for k in ("length", "width", "height")])
    scale = Vector([target[i]/extents[i] for i in range(3)])
    position = Vector([slot["position"][k] for k in ("x", "y", "z")])
    yaw = math.radians(slot["rotation"]["z_angle"] + 180.)
    transform = Matrix.Translation(position) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Diagonal((*scale, 1.)) @ Matrix.Translation(-center)
    root = bpy.data.objects.new(f"instance_{index:05d}", None)
    bpy.context.scene.collection.objects.link(root)
    root["slot_id"], root["asset_id"] = slot["new_object_id"], node["asset_id"]
    # Parent only the roots of THIS import, retaining every original world
    # matrix and child hierarchy. Shared source assets become separate imports.
    for obj in imported:
        if obj.parent not in imported:
            old = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_world = old
    root.matrix_world = transform
    bpy.context.view_layer.update()
    actual_lo, actual_hi = _bounds(meshes)
    # Validate in the planned slot's frame. Rotating an asymmetric multipart
    # union can change its WORLD AABB center; that does not move the slot anchor.
    pose = Matrix.Translation(position) @ Matrix.Rotation(yaw, 4, "Z")
    local_lo, local_hi = _bounds(meshes, pose.inverted())
    tolerance = 1e-4 * max(1., *target, *(abs(x) for x in position))
    if ((local_lo+local_hi)/2).length > tolerance or ((local_hi-local_lo)-target).length > tolerance:
        raise ValueError(f"actual placed bounds differ from planned slot: {slot['new_object_id']}")
    return {"asset_id": node["asset_id"], "slot_id": slot["new_object_id"], "slot": slot,
            "root": root.name, "imported_objects": sorted(obj.name for obj in imported),
            "mesh_objects": sorted(obj.name for obj in meshes), "raw_blender_bbox": [list(lo), list(hi)],
            "scale_factors": list(scale), "matrix_world": [list(row) for row in transform],
            "placed_local_bbox": [list(local_lo), list(local_hi)],
            "actual_bbox": [list(actual_lo), list(actual_hi)], "yaw_degrees": slot["rotation"]["z_angle"]+180.}


def _surfaces(room):
    for i, item in enumerate(room["surfaces"]):
        mesh = bpy.data.meshes.new(f"room_surface_{i}")
        mesh.from_pydata(item["vertices"], [], [tuple(range(len(item["vertices"])))])
        mesh.update()
        obj = bpy.data.objects.new(f"room_surface_{i}", mesh)
        obj["surface_id"] = item["id"]
        bpy.context.scene.collection.objects.link(obj)
        material = bpy.data.materials.new(f"room_material_{i}")
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = item["rgba"]
        shader.inputs["Roughness"].default_value = item["roughness"]
        shader.inputs["Alpha"].default_value = item["rgba"][3]
        mesh.materials.append(material)


def _configure_render(config):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.seed = config["seed"]
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = config["threads"]
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = config["transparent"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.display_settings.display_device = "sRGB"
    for key, value in config["color_management"].items():
        setattr(scene.view_settings, key, value)
    world = bpy.data.worlds.new("explicit_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*config["world"]["color"], 1.)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = config["world"]["strength"]
    scene.world = world
    for i, item in enumerate(config["lights"]):
        data = bpy.data.lights.new(f"explicit_light_{i}", "AREA")
        data.energy, data.color, data.shape, data.size = item["energy"], item["color"], item["shape"], item["size"]
        obj = bpy.data.objects.new(f"explicit_light_{i}", data)
        scene.collection.objects.link(obj)
        obj.matrix_world = Matrix(item["matrix_world"])
    cameras = []
    for i, item in enumerate(config["cameras"]):
        data = bpy.data.cameras.new(f"explicit_camera_{i}")
        data.type, data.lens, data.sensor_width = "PERSP", item["lens_mm"], item["sensor_width_mm"]
        data.clip_start, data.clip_end = item["clip_start"], item["clip_end"]
        obj = bpy.data.objects.new(f"explicit_camera_{i}", data)
        scene.collection.objects.link(obj)
        obj.matrix_world = Matrix(item["matrix_world"])
        cameras.append((item["id"], obj))
    return cameras


def _record(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    manifest = contract.load_placement(args.manifest, expected_sha256=args.manifest_sha256)
    out = args.out_dir
    if (out/"scene.blend").exists() or (out/"result.json").exists():
        raise FileExistsError("placement output already exists")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.
    scene.frame_set(0)
    records = []
    for i, node in enumerate(manifest["instances"]):
        record = manifest["assets"][node["asset_id"]]["mesh"]
        raw = contract._artifact(record, args.manifest.parent)
        # Import the same verified bytes; a mutable path is not reopened by bpy.
        frozen = out / f"import_{i:05d}.glb"
        with frozen.open("xb") as stream:
            stream.write(raw)
        records.append(_place(node, frozen, i))
        frozen.unlink()
        print(f"placed {i+1}/{len(manifest['instances'])}: {node['slot']['new_object_id']}", flush=True)
    _surfaces(manifest["room"])
    renders = []
    if manifest["render"] is not None:
        cameras = _configure_render(manifest["render"])
        for i, (camera_id, camera) in enumerate(cameras):
            scene.camera = camera
            dest = out / f"view_{i:03d}.png"
            if dest.exists():
                raise FileExistsError(dest)
            scene.render.filepath = str(dest)
            bpy.ops.render.render(write_still=True)
            renders.append({"camera_id": camera_id, **_record(dest)})
    # Pack textures before deleting any worker-owned files; self-contained GLBs
    # already embed their image bytes but the saved .blend should be portable.
    bpy.ops.file.pack_all()
    blend = out / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    result = {"schema": contract.SCHEMA, "status": "complete", "manifest_sha256": args.manifest_sha256,
              "classification": "IMPLEMENTATION CHOICE", "instances": records, "room": manifest["room"],
              "render_config": manifest["render"], "renders": renders, "blend": _record(blend),
              "blender_version": bpy.app.version_string, "blender_build_hash": bpy.app.build_hash.decode(),
              "device": "CPU", "implementation": manifest["implementation"]}
    contract._publish(out / "result.json", result)


if __name__ == "__main__":
    main()
