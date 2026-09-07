"""Frozen handoff and real tiny CPU Blender placement/render integration."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image
import pytest
import trimesh

from metafind.scene import placement as module


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def artifact(path, **extra):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}


def slot(name, x, yaw):
    return {"new_object_id": name, "position": {"x": x, "y": 3., "z": 1.5},
            "rotation": {"z_angle": yaw}, "size_in_meters": {"length": 2., "width": 1., "height": 3.},
            "placement": {"objects_in_room": [], "room_layout_elements": []}, "style": "retained planned style"}


@pytest.fixture
def inputs(tmp_path):
    scene = trimesh.Scene()
    # Two meshes under different translated graph nodes: local-origin imports
    # or cross-asset joins cannot satisfy both independent planned slots.
    first = trimesh.creation.box(extents=[1., 2., 1.])
    second = trimesh.creation.box(extents=[1., 1., 1.])
    first.visual.vertex_colors = [220, 40, 30, 255]
    second.visual.vertex_colors = [30, 150, 220, 255]
    scene.add_geometry(first, node_name="left", geom_name="left_mesh", transform=trimesh.transformations.translation_matrix([2., 3., 0.]))
    scene.add_geometry(second, node_name="right", geom_name="right_mesh", transform=trimesh.transformations.translation_matrix([4., 3., 0.]))
    glb = tmp_path / "raw.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    annotation = write_json(tmp_path / "annotation.json", {"uid": "asset-A", "description": "actual retrieved asset", "height": 900, "dimension_unit": "cm"})
    composition = {"schema": "metafind.scene_composition.v1", "status": "complete", "room_id": "room-A",
                   "final_graph": {"nodes": [{"asset_id": "asset-A", "slot": slot("chair-with-hyphen", 2., 90.)},
                                               {"asset_id": "asset-A", "slot": slot("another chair", 7., 45.)}]}}
    assets = {"provenance": {"source": "synthetic static fixture"},
              "assets": {"asset-A": {"mesh": artifact(glb, frame="raw_gltf_y_up"), "annotation": artifact(annotation)}}}
    room = {"room_id": "room-A", "dimensions": [10., 8., 4.], "provenance": {"source": "test-only dimensions"},
            "surfaces": [{"id": "floor", "vertices": [[0., 0., 0.], [10., 0., 0.], [10., 8., 0.], [0., 8., 0.]],
                          "rgba": [.6, .6, .6, 1.], "roughness": .7}]}
    # Explicit top-down rigid camera; these settings are fixtures, not paper defaults.
    camera_pose = [[1., 0., 0., 5.], [0., 1., 0., 4.], [0., 0., 1., 12.], [0., 0., 0., 1.]]
    render = {"provenance": {"source": "test-only CPU render"}, "engine": "CYCLES", "device": "CPU", "threads": 1,
              "samples": 2, "seed": 17, "resolution": [64, 64], "transparent": False,
              "world": {"color": [.5, .5, .5], "strength": .2},
              "color_management": {"view_transform": "Standard", "look": "None", "exposure": 0., "gamma": 1.},
              "cameras": [{"id": "test-top", "matrix_world": camera_pose, "lens_mm": 32., "sensor_width_mm": 36., "clip_start": .01, "clip_end": 100.}],
              "lights": [{"id": "test-area", "type": "AREA", "shape": "DISK", "matrix_world": camera_pose,
                          "energy": 600., "color": [1., 1., 1.], "size": 5.}]}
    return {"composition": write_json(tmp_path / "composition.json", composition),
            "assets": write_json(tmp_path / "assets.json", assets),
            "room": write_json(tmp_path / "room.json", room),
            "render": write_json(tmp_path / "render.json", render), "glb": glb}


def prepare(inputs, out, *, render=False):
    return module.prepare_placement(inputs["composition"], inputs["assets"], inputs["room"], out,
                                    render_config_path=inputs["render"] if render else None)


def test_freeze_preserves_slots_assets_and_annotation_without_using_annotation_size(inputs, tmp_path):
    path = prepare(inputs, tmp_path / "bundle")
    data = module.load_placement(path)
    assert data["instances"] == json.loads(inputs["composition"].read_text())["final_graph"]["nodes"]
    assert len(data["assets"]) == 1  # two independent instances, one immutable input asset
    assert data["render"] is None
    assert data["instances"][0]["slot"]["size_in_meters"]["height"] == 3.
    assert not path.with_name("placement.json.part").exists()
    # Frozen copies remain usable when original source files later change.
    inputs["glb"].write_bytes(b"changed original")
    module.load_placement(path)
    with pytest.raises(FileExistsError):
        prepare(inputs, path.parent)


@pytest.mark.parametrize("fault", ["missing_uid", "wrong_annotation_uid", "corrected_frame", "bad_mesh_hash", "wrong_room", "zero_size", "duplicate_slot", "missing_camera", "gpu"])
def test_preflight_rejects_before_creating_bundle(inputs, tmp_path, fault):
    field = "assets" if fault in ("missing_uid", "wrong_annotation_uid", "corrected_frame", "bad_mesh_hash") else "room" if fault == "wrong_room" else "render" if fault in ("missing_camera", "gpu") else "composition"
    data = json.loads(inputs[field].read_text())
    if fault == "missing_uid":
        data["assets"] = {}
    elif fault == "wrong_annotation_uid":
        annotation = write_json(tmp_path / "wrong.json", {"uid": "other"})
        data["assets"]["asset-A"]["annotation"] = artifact(annotation)
    elif fault == "corrected_frame":
        data["assets"]["asset-A"]["mesh"]["frame"] = "yaw180_about_y@ulip2_frame"
    elif fault == "bad_mesh_hash":
        data["assets"]["asset-A"]["mesh"]["sha256"] = "0" * 64
    elif fault == "wrong_room":
        data["room_id"] = "room-B"
    elif fault == "zero_size":
        data["final_graph"]["nodes"][0]["slot"]["size_in_meters"]["width"] = 0
    elif fault == "duplicate_slot":
        data["final_graph"]["nodes"][1]["slot"] = data["final_graph"]["nodes"][0]["slot"]
    elif fault == "missing_camera":
        data["cameras"] = []
    elif fault == "gpu":
        data["device"] = "GPU"
    write_json(inputs[field], data)
    with pytest.raises((ValueError, KeyError)):
        prepare(inputs, tmp_path / "bundle", render=True)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("fault", ["mesh", "slot", "render", "source", "source_text", "choice"])
def test_frozen_bundle_tamper_rejected(inputs, tmp_path, fault):
    path = prepare(inputs, tmp_path / "bundle", render=True)
    data = json.loads(path.read_text())
    if fault == "mesh":
        (path.parent / data["assets"]["asset-A"]["mesh"]["path"]).write_bytes(b"tampered")
    elif fault == "slot":
        data["instances"][0]["slot"]["position"]["x"] = 99.
    elif fault == "render":
        data["render"]["samples"] = 64
    elif fault == "source":
        data["sources"]["assets"]["content"]["provenance"]["source"] = "false attribution"
    elif fault == "source_text":
        data["sources"]["assets"]["raw_json"] += " "
    else:
        data["choices"]["raw_mesh_yaw_offset_degrees"] = 0
    write_json(path, data)
    with pytest.raises(ValueError):
        module.load_placement(path)


@pytest.mark.parametrize("section", ["root", "camera", "world", "color_management"])
def test_render_rejects_unconsumed_fields(inputs, tmp_path, section):
    config = json.loads(inputs["render"].read_text())
    target = config if section == "root" else config["cameras"][0] if section == "camera" else config[section]
    target["unconsumed_setting"] = 1
    write_json(inputs["render"], config)
    with pytest.raises(ValueError, match="exactly these fields"):
        prepare(inputs, tmp_path / "bundle", render=True)
    assert not (tmp_path / "bundle").exists()


def test_missing_blender_does_not_create_output(inputs, tmp_path):
    path = prepare(inputs, tmp_path / "bundle")
    with pytest.raises(FileNotFoundError):
        module.run_placement(path, tmp_path / "run", blender=tmp_path / "absent-blender")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field,value", [("animations", [{}]), ("skins", [{}]),
                                         ("images", [{"uri": "unbound-texture.png"}]),
                                         ("buffers", [{"uri": "unbound-buffer.bin"}])])
def test_nonstatic_or_external_glb_refused_without_import(inputs, tmp_path, field, value):
    raw = inputs["glb"].read_bytes()
    length = struct.unpack("<I", raw[12:16])[0]
    gltf = json.loads(raw[20:20+length])
    gltf[field] = value
    new_json = json.dumps(gltf).encode()
    new_json += b" " * (-len(new_json) % 4)
    tail = raw[20+length:]
    altered = struct.pack("<4sIIII", b"glTF", 2, 20+len(new_json)+len(tail), len(new_json), 0x4E4F534A)+new_json+tail
    inputs["glb"].write_bytes(altered)
    assets = json.loads(inputs["assets"].read_text())
    assets["assets"]["asset-A"]["mesh"] = artifact(inputs["glb"], frame="raw_gltf_y_up")
    write_json(inputs["assets"], assets)
    with pytest.raises(ValueError, match="pose policy|external dependencies"):
        prepare(inputs, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("render", [False, True])
def test_real_cpu_blender_multimesh_pose_and_render(inputs, tmp_path, render):
    try:
        blender = module.default_blender()
    except FileNotFoundError:
        pytest.skip("installed CPU Blender unavailable; no dependency installation")
    if render:
        assert module.main(["prepare", "--composition", str(inputs["composition"]), "--assets", str(inputs["assets"]),
                            "--room", str(inputs["room"]), "--render", str(inputs["render"]),
                            "--out-dir", str(tmp_path / "bundle")]) == 0
        path = tmp_path / "bundle" / "placement.json"
    else:
        path = prepare(inputs, tmp_path / "bundle")
    original_hash = hashlib.sha256(inputs["glb"].read_bytes()).hexdigest()
    if render:
        assert module.main(["run", "--manifest", str(path), "--out-dir", str(tmp_path / "run"),
                            "--blender", str(blender)]) == 0
        result_path = tmp_path / "run" / "result.json"
    else:
        result_path = module.run_placement(path, tmp_path / "run", blender=blender, timeout=120)
    result = json.loads(result_path.read_text())
    assert result["device"] == "CPU" and result["status"] == "complete"
    a, b = result["instances"]
    assert len(a["mesh_objects"]) == len(b["mesh_objects"]) == 2
    assert set(a["imported_objects"]).isdisjoint(b["imported_objects"])
    assert a["yaw_degrees"] == 270. and b["yaw_degrees"] == 225.
    for item in result["instances"]:
        np.testing.assert_allclose(item["placed_local_bbox"], [[-1., -.5, -1.5], [1., .5, 1.5]], atol=1e-5)
    # 90-degree yaw swaps horizontal extents; translation uses slot center.
    np.testing.assert_allclose(a["actual_bbox"], [[1.5, 2., 0.], [2.5, 4., 3.]], atol=1e-5)
    assert hashlib.sha256(inputs["glb"].read_bytes()).hexdigest() == original_hash
    assert result["blend"]["sha256"] == hashlib.sha256((result_path.parent / "scene.blend").read_bytes()).hexdigest()
    assert len(result["renders"]) == int(render)
    if render:
        pixels = np.array(Image.open(result_path.parent / result["renders"][0]["path"]))
        assert pixels.shape == (64, 64, 4)
        assert pixels[..., :3].std() > 10  # actual visible geometry, not an empty uniform image
    with pytest.raises(FileExistsError):
        module.run_placement(path, result_path.parent, blender=blender)
    # Independently reopen the saved .blend and inspect the scene, without rendering.
    inspector = tmp_path / "inspect_blend.py"
    inspector.write_text("import bpy, json\nfrom pathlib import Path\ns=bpy.context.scene\n"
                         "roots=[o for o in s.objects if 'slot_id' in o]\n"
                         "assert len(roots)==2\nassert len([o for o in s.objects if o.type=='MESH'])==5\n"
                         "assert s.unit_settings.scale_length==1.\n"
                         + ("assert s.cycles.device=='CPU' and s.render.threads==1\nassert s.render.resolution_x==64\n" if render else ""))
    proc = subprocess.run([str(blender), "--background", "--threads", "1", str(result_path.parent / "scene.blend"),
                           "--python-exit-code", "3", "--python", str(inspector)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_rotated_nonbox_mesh_uses_actual_vertices_for_fit_and_center(inputs, tmp_path):
    """Local AABB corners overestimate this tetrahedron after its node rotates.

    Previously the result reported a successful 2 m fit, but independent saved
    geometry was only 1.333333 m wide and its slot-frame center was -0.333333 m.
    The oracle below also independently checks raw Y-up -> Blender Z-up and
    the declared yaw + 180 degrees; it never calls the production bounds helper.
    """
    try:
        blender = module.default_blender()
    except FileNotFoundError:
        pytest.skip("installed CPU Blender unavailable; no dependency installation")
    raw_vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tetra = trimesh.Trimesh(vertices=raw_vertices,
                           faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], process=False)
    scene = trimesh.Scene()
    node_transform = trimesh.transformations.rotation_matrix(np.pi/4, [0., 1., 0.])
    scene.add_geometry(tetra, node_name="rotated_tetra", transform=node_transform)
    inputs["glb"].write_bytes(scene.export(file_type="glb"))
    assets = json.loads(inputs["assets"].read_text())
    assets["assets"]["asset-A"]["mesh"] = artifact(inputs["glb"], frame="raw_gltf_y_up")
    write_json(inputs["assets"], assets)
    composition = json.loads(inputs["composition"].read_text())
    composition["final_graph"]["nodes"] = composition["final_graph"]["nodes"][:1]
    planned = composition["final_graph"]["nodes"][0]["slot"]
    planned["rotation"]["z_angle"] = 0.
    write_json(inputs["composition"], composition)
    path = prepare(inputs, tmp_path / "bundle")
    result_path = module.run_placement(path, tmp_path / "run", blender=blender, timeout=120)

    # Numerically derive the world-space vertices from the glTF source, using
    # the documented C(x,y,z)=(x,-z,y), actual vertex extrema and planned pose.
    basis = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    imported = raw_vertices @ node_transform[:3, :3].T @ basis.T
    center = (imported.min(0) + imported.max(0)) / 2
    target = np.array([2., 1., 3.])
    scale = target / np.ptp(imported, axis=0)
    yaw = np.diag([-1., -1., 1.])
    expected = ((imported - center) * scale) @ yaw.T + np.array([2., 3., 1.5])
    oracle_path = tmp_path / "vertex_oracle.json"
    inspector = tmp_path / "inspect_tetra.py"
    inspector.write_text(
        "import bpy,json\nfrom pathlib import Path\n"
        "root=next(o for o in bpy.context.scene.objects if 'slot_id' in o)\n"
        "objects=[o for o in root.children_recursive if o.type=='MESH']\n"
        "points=[list(o.matrix_world@v.co) for o in objects for v in o.data.vertices]\n"
        f"Path({str(oracle_path)!r}).write_text(json.dumps(points))\n")
    proc = subprocess.run([str(blender), "--background", "--threads", "1",
                           str(result_path.parent / "scene.blend"), "--python-exit-code", "3",
                           "--python", str(inspector)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    actual = np.array(json.loads(oracle_path.read_text()))
    # Importers may reorder/duplicate vertices to represent face normals.
    distance = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=-1)
    assert distance.min(axis=1).max() < 1e-5
    assert distance.min(axis=0).max() < 1e-5
    local = (actual - np.array([2., 3., 1.5])) @ yaw
    np.testing.assert_allclose(np.ptp(local, axis=0), target, atol=1e-5)
    np.testing.assert_allclose((local.min(0) + local.max(0))/2, 0, atol=1e-5)
