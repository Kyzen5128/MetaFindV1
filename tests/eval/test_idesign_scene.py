"""Raw I-Design JSON -> explicit scene request, without planner/model calls."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from metafind.scene import idesign
from metafind.scene import prepare as encoder


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def object_slot(name, x):
    return {"new_object_id": name, "position": {"x": x, "y": 2., "z": .5},
            "rotation": {"z_angle": 0.}, "size_in_meters": {"length": 1., "width": 1., "height": 1.},
            "style": "original planned style", "material": "wood", "is_on_the_floor": True,
            "placement": {"objects_in_room": [], "room_layout_elements": [{"layout_element_id": "south_wall", "preposition": "on"}]}}


@pytest.fixture
def inputs(tmp_path):
    # Independent fixture from upstream get_room_priors([4,5,3]); no calls to
    # the adapter's expected-geometry implementation when constructing input.
    prior_specs = [
        ("south_wall", "wall", [2,0,1.5], [4,0,3], 0),
        ("north_wall", "wall", [2,5,1.5], [4,0,3], 180),
        ("east_wall", "wall", [4,2.5,1.5], [5,0,3], 270),
        ("west_wall", "wall", [0,2.5,1.5], [5,0,3], 90),
        ("middle of the room", "floor", [2,2.5,0], [4,5,0], 0),
        ("ceiling", "ceiling", [2,2.5,3], [4,5,0], 0),
    ]
    priors = [{"new_object_id": name, "itemType": kind, "position": dict(zip(("x","y","z"),xyz)),
               "size_in_meters": dict(zip(("length","width","height"),size)), "rotation": {"z_angle": yaw}}
              for name, kind, xyz, size, yaw in prior_specs]
    first, second = object_slot("chair_2", 1.), object_slot("table_1", 3.)
    first["placement"]["objects_in_room"] = [{"object_id": "table_1", "preposition": "on"}]
    scene = [first, priors[0], second, *priors[1:]]
    scene_path = write(tmp_path / "planner/scene_graph.json", scene)
    sidecar = {"scene_id": "scene_0007", "room_dimensions": [4.,5.,3.], "prompt": "explicit fixture prompt",
               "n_objects_requested": 2, "planner_model": "fixture-no-LLM", "idesign_revision": "fixture",
               "scene_graph_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest()}
    sidecar_path = write(tmp_path / "planner/sidecar.json", sidecar)
    observations = tmp_path / "observations"
    observations.mkdir()
    Image.new("RGB", (3,3), (10,20,30)).save(observations / "image.png")
    np.savez(observations / "pc.npz", xyz=np.zeros((10000,3),dtype=np.float32), rgb=np.ones((10000,3),dtype=np.float32))
    # Reverse mapping insertion order must not reorder the original slots.
    mapping = {"table_1": {"text": "explicit table text", "images": ["image.png"], "pointcloud": "pc.npz"},
               "chair_2": {"text": "explicit chair query; not its future retrieved annotation"}}
    mapping_path = write(observations / "queries.json", mapping)
    for name in ("s1", "s2", "gallery", "texts", "cache"):
        write(tmp_path / f"records/{name}.json", {})
    base = {"schema": encoder.REQUEST_SCHEMA, "provenance": {"source": "explicit caller base"},
            "stage1_record": "../records/s1.json", "stage2_record": "../records/s2.json", "variant": "full",
            "gallery_registry": "../records/gallery.json", "gallery_ids": ["asset-A"],
            "asset_texts": "../records/texts.json", "semantic_cache": "../records/cache.json",
            "mode": "iterative", "use_layout": True}
    return {"scene": scene_path, "sidecar": sidecar_path, "mapping": mapping_path,
            "base": write(tmp_path / "config/base.json", base)}


def build(inputs, out):
    return idesign.build_request(inputs["scene"], inputs["sidecar"], inputs["mapping"], inputs["base"], out)


def test_cli_request_preserves_order_slots_empty_g0_and_own_parent_paths(inputs, tmp_path, monkeypatch):
    out = tmp_path / "new/result/request.json"
    assert idesign.main(["--scene", str(inputs["scene"]), "--sidecar", str(inputs["sidecar"]),
                         "--query-modalities", str(inputs["mapping"]), "--base-request", str(inputs["base"]),
                         "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert idesign.REQUEST_SCHEMA == encoder.REQUEST_SCHEMA == data["schema"]
    assert data["initial_graph"] == {"room_id": "scene_0007", "nodes": []}
    assert [q["slot"]["new_object_id"] for q in data["queries"]] == ["chair_2", "table_1"]
    scene = json.loads(inputs["scene"].read_text())
    assert [q["slot"] for q in data["queries"]] == [scene[0], scene[2]]
    assert data["queries"][0]["text"].startswith("explicit chair query")
    assert data["queries"][1]["images"] == [str(tmp_path / "observations/image.png")]
    assert data["queries"][1]["pointcloud"] == str(tmp_path / "observations/pc.npz")
    assert data["stage1_record"] == str(tmp_path / "records/s1.json")
    proof = data["provenance"]["idesign_adapter"]
    assert proof["object_count"] == 2 and proof["room_prior_count"] == 6
    assert proof["room_dimensions"] == [4.,5.,3.]
    assert proof["sources"]["query_modalities"]["sha256"] == hashlib.sha256(inputs["mapping"].read_bytes()).hexdigest()
    assert not out.with_name("request.json.part").exists()
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        build(inputs, out)
    assert out.read_bytes() == before
    # Exercise the real downstream request schema and source loading. Stop at
    # the first actual model-load boundary; no backbone is loaded by this test.
    def stop_at_model(path, device):
        assert path == tmp_path / "records/s1.json"
        raise RuntimeError("schema accepted; model loading intentionally stopped")
    monkeypatch.setattr(encoder, "load_stage1", stop_at_model)
    with pytest.raises(RuntimeError, match="schema accepted"):
        encoder.prepare(out, tmp_path / "encoded")
    # The real raw query encoder also accepts the resolved modalities. A tiny
    # encoder supplies known vectors; image/PC files and transformations are real.
    backbone = SimpleNamespace(encode_text=lambda texts: torch.ones((len(texts),2)),
                               preprocess=lambda image: torch.zeros((3,2,2)),
                               encode_image=lambda images: torch.tensor([[2.,3.]]),
                               encode_pc=lambda pc: torch.tensor([[4.,5.]]))
    loaded = SimpleNamespace(backbone=backbone, query_backbone=None,
                             model=SimpleNamespace(cfg=SimpleNamespace(dim=2)), encoding={"image_aggregation": "mean"})
    encoded = encoder.encode_query(data["queries"][1], loaded, out.parent, {})
    assert set(encoded["embeds"]) == {"text", "image", "pc"}
    torch.testing.assert_close(encoded["embeds"]["pc"], torch.tensor([4.,5.]))


@pytest.mark.parametrize("fault", ["missing_prior", "duplicate_id", "missing_position", "zero_size", "bad_object_ref",
                                   "bad_room_ref", "wrong_room_dimensions", "wrong_scene_hash", "failed_sidecar",
                                   "missing_query", "extra_query", "empty_modalities", "empty_text", "missing_image",
                                   "unknown_modality", "base_initial_graph", "base_queries", "base_missing_path", "base_mode"])
def test_invalid_scene_or_mapping_never_publishes(inputs, tmp_path, fault):
    field = ("scene" if fault in ("missing_prior", "duplicate_id", "missing_position", "zero_size", "bad_object_ref", "bad_room_ref")
             else "sidecar" if fault in ("wrong_room_dimensions", "wrong_scene_hash", "failed_sidecar")
             else "base" if fault.startswith("base_") else "mapping")
    data = json.loads(inputs[field].read_text())
    if fault == "missing_prior": data.pop()
    elif fault == "duplicate_id": data[2]["new_object_id"] = data[0]["new_object_id"]
    elif fault == "missing_position": del data[0]["position"]
    elif fault == "zero_size": data[0]["size_in_meters"]["height"] = 0
    elif fault == "bad_object_ref": data[0]["placement"]["objects_in_room"][0]["object_id"] = "ceiling"
    elif fault == "bad_room_ref": data[0]["placement"]["room_layout_elements"][0]["layout_element_id"] = "table_1"
    elif fault == "wrong_room_dimensions": data["room_dimensions"] = [5.,5.,3.]
    elif fault == "wrong_scene_hash": data["scene_graph_sha256"] = "0"*64
    elif fault == "failed_sidecar": data["status"] = "failed"
    elif fault == "missing_query": del data["chair_2"]
    elif fault == "extra_query": data["ceiling"] = {"text": "not an asset query"}
    elif fault == "empty_modalities": data["chair_2"] = {}
    elif fault == "empty_text": data["chair_2"]["text"] = " "
    elif fault == "missing_image": data["table_1"]["images"] = ["absent.png"]
    elif fault == "unknown_modality": data["chair_2"]["embeds"] = [1.,2.]
    elif fault == "base_initial_graph": data["initial_graph"] = {"nodes": []}
    elif fault == "base_queries": data["queries"] = []
    elif fault == "base_missing_path": data["stage2_record"] = "absent.json"
    elif fault == "base_mode": data["mode"] = "invented"
    write(inputs[field], data)
    if field == "scene":
        sidecar = json.loads(inputs["sidecar"].read_text())
        sidecar["scene_graph_sha256"] = hashlib.sha256(inputs["scene"].read_bytes()).hexdigest()
        write(inputs["sidecar"], sidecar)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        build(inputs, tmp_path / "output/request.json")
    assert not (tmp_path / "output").exists()


def test_duplicate_json_mapping_key_is_rejected(inputs, tmp_path):
    inputs["mapping"].write_text('{"chair_2":{"text":"first"},"chair_2":{"text":"silently replaces first"},"table_1":{"text":"table"}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        build(inputs, tmp_path / "request.json")
    assert not (tmp_path / "request.json").exists()
