"""External rating handoff tests; no model calls or actual Blender execution.

The existing fixture supplies tiny real static GLBs. prepare_placement and
load_placement are real; only the external Blender result/images and judge
responses are authored fixtures. These tests prove artifact linkage and score
accounting, not geometry or the validity of a model's aesthetic judgement.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from metafind.eval import scene_scores as module
from metafind.scene import placement
from .test_scene_placement import inputs as placement_inputs


def _json(path, value):
    path.write_text(json.dumps(value))
    return path


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _complete(inputs, root, scene_id):
    root.mkdir()
    comp = json.loads(inputs["composition"].read_text())
    room = json.loads(inputs["room"].read_text())
    comp["room_id"] = room["room_id"] = scene_id
    composition = _json(root / "composition.json", comp)
    room_path = _json(root / "room.json", room)
    frozen = placement.prepare_placement(composition, inputs["assets"], room_path,
                                         root / "bundle", render_config_path=inputs["render"])
    manifest = placement.load_placement(frozen)
    output = root / "worker_fixture"
    output.mkdir()
    Image.new("RGBA", (64, 64), (55, 110, 220, 255)).save(output / "view_000.png")
    (output / "scene.blend").write_bytes(b"synthetic blend bytes; this test never runs Blender")
    instances = [{"asset_id": n["asset_id"], "slot_id": n["slot"]["new_object_id"], "slot": n["slot"]}
                 for n in manifest["instances"]]
    result = _json(output / "result.json", {
        "schema": placement.SCHEMA, "status": "complete", "manifest_sha256": _ref(frozen)["sha256"],
        "instances": instances, "room": room, "render_config": manifest["render"], "device": "CPU",
        "implementation": manifest["implementation"], "blend": _ref(output / "scene.blend"),
        "renders": [{"camera_id": "test-top", **_ref(output / "view_000.png")}],
    })
    outcome = {"method_id": "method-A", "scene_id": scene_id, "status": "complete",
               "composition": _ref(composition), "placement": _ref(frozen), "result": _ref(result),
               "judged_camera_ids": ["test-top"]}
    identity = {key + "_sha256": outcome[key]["sha256"] for key in ("composition", "placement", "result")}
    identity.update(renders={"test-top": _ref(output / "view_000.png")["sha256"]}, judged_camera_ids=["test-top"])
    return outcome, identity


@pytest.fixture
def bundle(placement_inputs, tmp_path):
    protocol = {"schema": module.PROTOCOL_SCHEMA, "status": "frozen",
                "dimensions": list(module.DIMENSIONS), "score_range": [1, 5],
                "model": {"id": "gemma-4-12B-it", "revision": "test-only-not-real-weights"},
                "prompt": "Explicit synthetic test rubric; not a proposed research prompt.",
                "generation": {"test_fixture_only": True}, "view_policy": "Explicit test-top fixture view",
                "provenance": {"source": "test-only protocol; no real judge"}}
    protocol_path = _json(tmp_path / "protocol.json", protocol)
    outcomes, identities = [], {}
    for scene in ("scene-1", "scene-2", "scene-3"):
        outcome, identity = _complete(placement_inputs, tmp_path / scene, scene)
        outcomes.append(outcome)
        identities[scene] = identity
    outcomes.append({"method_id": "method-A", "scene_id": "scene-4", "status": "incomplete",
                     "stage": "composition", "reason": "fixture semantic relation missing"})
    manifest = {"schema": module.MANIFEST_SCHEMA, "status": "frozen", "scene_ids": list(identities)+["scene-4"],
                "method_ids": ["method-A"], "outcomes": outcomes, "provenance": {"source": "test-only four scenes"}}
    manifest_path = _json(tmp_path / "manifest.json", manifest)
    records = []
    for scene, score in (("scene-1", 2), ("scene-2", 4)):
        scores = dict.fromkeys(module.DIMENSIONS, score)
        response = _json(tmp_path / (scene + "-response.json"), {"scores": scores, "fixture": True})
        records.append({"method_id": "method-A", "scene_id": scene, "status": "scored",
                        "inputs": identities[scene], "scores": scores, "raw_response": _ref(response)})
    submission = {"schema": module.SUBMISSION_SCHEMA, "protocol_sha256": _ref(protocol_path)["sha256"],
                  "manifest_sha256": _ref(manifest_path)["sha256"], "records": records}
    return {"protocol": protocol_path, "manifest": manifest_path,
            "records": _json(tmp_path / "records.json", submission), "identities": identities,
            "out": tmp_path / "imported.json"}


def _import(bundle):
    return module.import_scores(bundle["manifest"], bundle["protocol"], bundle["records"], bundle["out"])


def _edit(bundle, name, mutate, *, rebind=False):
    value = json.loads(bundle[name].read_text())
    mutate(value)
    _json(bundle[name], value)
    if rebind:
        submission = json.loads(bundle["records"].read_text())
        submission[name + "_sha256"] = _ref(bundle[name])["sha256"]
        _json(bundle["records"], submission)


def test_real_placement_handoff_and_independent_denominator_arithmetic(bundle):
    path = _import(bundle)
    validated = module.load_scores(path)
    assert validated["sources"]["manifest"]["sha256"] == _ref(bundle["manifest"])["sha256"]
    summary = module.aggregate_scores(path)
    assert summary["scores_artifact"] == _ref(path)
    assert summary["submission_sha256"] == _ref(bundle["records"])["sha256"]
    result = summary["methods"]["method-A"]
    assert (result["n_total"], result["n_complete"], result["n_incomplete"], result["n_scored"]) == (4, 3, 1, 2)
    assert result["completion_rate"] == .75
    assert result["status"] == "partial"
    assert result["n_missing_scores"] == 1 and result["n_judge_failed"] == 0
    assert result["mean_over_scored"] == dict.fromkeys(module.DIMENSIONS, 3.)  # (2 + 4) / 2, never / 4
    assert result["mean_over_complete"] == dict.fromkeys(module.DIMENSIONS, module.INSUFFICIENT)
    assert result["human"] == dict.fromkeys(module.DIMENSIONS, module.INSUFFICIENT)
    assert [r["scene_id"] for r in result["failures"]] == ["scene-3", "scene-4"]
    assert result["failures"][1]["reason"] == "fixture semantic relation missing"


def test_failed_judge_and_missing_scene_are_separate_from_completion(bundle):
    _edit(bundle, "records", lambda v: v["records"].append({"method_id": "method-A", "scene_id": "scene-3",
          "status": "failed", "inputs": bundle["identities"]["scene-3"], "reason": "fixture unparsable judge reply"}))
    result = module.aggregate_scores(_import(bundle))["methods"]["method-A"]
    assert result["n_complete"] == 3 and result["n_incomplete"] == 1
    assert result["n_judge_failed"] == 1 and result["n_missing_scores"] == 0
    assert result["mean_over_complete"]["scene_coherence"] == module.INSUFFICIENT
    assert result["failures"][0]["reason"] == "fixture unparsable judge reply"


def test_all_completed_scenes_scored_allows_only_conditional_complete_mean(bundle):
    def fill(value):
        r = copy.deepcopy(value["records"][0])
        r.update(scene_id="scene-3", inputs=bundle["identities"]["scene-3"], scores=dict.fromkeys(module.DIMENSIONS, 3))
        value["records"].append(r)
    _edit(bundle, "records", fill)
    result = module.aggregate_scores(_import(bundle))["methods"]["method-A"]
    assert result["mean_over_complete"] == dict.fromkeys(module.DIMENSIONS, 3.)
    assert result["n_total"] == 4 and result["status"] == "partial"  # fourth scene did not complete


def test_zero_scores_never_becomes_a_zero_mean(bundle):
    _edit(bundle, "records", lambda v: v.update(records=[]))
    result = module.aggregate_scores(_import(bundle))["methods"]["method-A"]
    assert result["n_scored"] == 0 and result["n_missing_scores"] == 3
    assert result["mean_over_scored"] == result["mean_over_complete"] == dict.fromkeys(module.DIMENSIONS, module.INSUFFICIENT)


def test_each_method_retains_the_same_scene_denominator_without_borrowing_scores(bundle):
    def add_method(v):
        v["method_ids"].append("method-B")
        others = copy.deepcopy(v["outcomes"])
        for r in others:
            r["method_id"] = "method-B"
        v["outcomes"].extend(others)
    _edit(bundle, "manifest", add_method, rebind=True)
    result = module.aggregate_scores(_import(bundle))["methods"]
    assert result["method-A"]["n_total"] == result["method-B"]["n_total"] == 4
    assert result["method-A"]["n_scored"] == 2
    assert result["method-B"]["n_scored"] == 0 and result["method-B"]["n_missing_scores"] == 3
    assert result["method-B"]["mean_over_scored"] == dict.fromkeys(module.DIMENSIONS, module.INSUFFICIENT)


@pytest.mark.parametrize("score", [True, "3", None, 0, 5.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_score_rejected_before_publication(bundle, score):
    _edit(bundle, "records", lambda v: v["records"][0]["scores"].update(scene_coherence=score))
    with pytest.raises(ValueError):
        _import(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("fault", ["missing_dimension", "extra_dimension", "unknown_scene", "unknown_method",
                                    "duplicate", "wrong_input", "wrong_camera_order", "incomplete_scene", "blank_failure", "malformed_id"])
def test_external_score_identity_and_completeness_are_strict(bundle, fault):
    def mutate(value):
        rec = value["records"][0]
        if fault == "missing_dimension": rec["scores"].pop("scene_coherence")
        elif fault == "extra_dimension": rec["scores"]["layout_and_furniture"] = 3
        elif fault == "unknown_scene": rec["scene_id"] = "other"
        elif fault == "unknown_method": rec["method_id"] = "other"
        elif fault == "duplicate": value["records"].append(copy.deepcopy(rec))
        elif fault == "wrong_input": rec["inputs"]["composition_sha256"] = "0" * 64
        elif fault == "wrong_camera_order": rec["inputs"]["judged_camera_ids"] = []
        elif fault == "incomplete_scene": rec["scene_id"] = "scene-4"
        elif fault == "malformed_id": rec["method_id"] = ["method-A"]
        else:
            rec.update(status="failed", reason=" ")
            del rec["scores"], rec["raw_response"]
    _edit(bundle, "records", mutate)
    with pytest.raises(ValueError): _import(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("fault", ["missing_outcome", "duplicate_outcome", "duplicate_scene", "duplicate_method", "unknown_pair", "blank_reason"])
def test_declared_denominator_cannot_silently_shrink(bundle, fault):
    def mutate(v):
        if fault == "missing_outcome": v["outcomes"].pop()
        elif fault == "duplicate_outcome": v["outcomes"].append(copy.deepcopy(v["outcomes"][0]))
        elif fault == "duplicate_scene": v["scene_ids"].append(v["scene_ids"][0])
        elif fault == "duplicate_method": v["method_ids"].append(v["method_ids"][0])
        elif fault == "unknown_pair": v["outcomes"][0]["method_id"] = "other"
        else: v["outcomes"][-1]["reason"] = ""
    _edit(bundle, "manifest", mutate, rebind=True)
    with pytest.raises(ValueError): _import(bundle)


@pytest.mark.parametrize("fault", ["missing_prompt", "missing_generation", "missing_view_policy", "not_frozen", "wrong_scale", "boolean_scale", "wrong_dimensions", "wrong_model"])
def test_no_invented_judge_protocol_defaults(bundle, fault):
    def mutate(v):
        if fault.startswith("missing_"): v.pop(fault.removeprefix("missing_"))
        elif fault == "not_frozen": v["status"] = "draft"
        elif fault == "wrong_scale": v["score_range"] = [1, 10]
        elif fault == "boolean_scale": v["score_range"] = [True, 5]
        elif fault == "wrong_dimensions": v["dimensions"] = ["layout_and_furniture"]
        else: v["model"]["id"] = "gpt-4o"
    _edit(bundle, "protocol", mutate, rebind=True)
    with pytest.raises(ValueError): _import(bundle)


@pytest.mark.parametrize("dependency", ["composition", "placement", "result", "image", "blend", "raw_response", "frozen_mesh", "protocol", "manifest", "records"])
def test_mutable_source_drift_rejected_on_reload(bundle, dependency):
    path = _import(bundle)
    manifest = json.loads(bundle["manifest"].read_text())
    outcome = manifest["outcomes"][0]
    if dependency in ("composition", "placement", "result"):
        target = Path(outcome[dependency]["path"])
    elif dependency in ("image", "blend"):
        result = json.loads(Path(outcome["result"]["path"]).read_text())
        target = Path((result["renders"][0] if dependency == "image" else result["blend"])["path"])
    elif dependency == "raw_response":
        target = Path(json.loads(bundle["records"].read_text())["records"][0][dependency]["path"])
    elif dependency == "frozen_mesh":
        placement_path = Path(outcome["placement"]["path"])
        frozen = json.loads(placement_path.read_text())
        target = placement_path.parent / frozen["assets"]["asset-A"]["mesh"]["path"]
    else:
        target = bundle[dependency]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError): module.load_scores(path)


@pytest.mark.parametrize("fault", ["asset", "camera", "placement_hash", "room", "render_config", "incomplete", "no_views", "selected_camera", "png_size"])
def test_blender_result_and_views_must_match_the_actual_placement_contract(bundle, fault):
    manifest = json.loads(bundle["manifest"].read_text())
    outcome = manifest["outcomes"][0]
    path = Path(outcome["result"]["path"])
    result = json.loads(path.read_text())
    if fault == "asset": result["instances"][0]["asset_id"] = "wrong-asset"
    elif fault == "camera": result["renders"][0]["camera_id"] = "not-rendered"
    elif fault == "placement_hash": result["manifest_sha256"] = "0" * 64
    elif fault == "room": result["room"]["dimensions"] = [99., 99., 99.]
    elif fault == "render_config": result["render_config"]["samples"] = 100
    elif fault == "incomplete": result["status"] = "failed"
    elif fault == "no_views": result["renders"] = []
    elif fault == "selected_camera": outcome["judged_camera_ids"] = ["not-rendered"]
    elif fault == "png_size":
        image_path = Path(result["renders"][0]["path"])
        Image.new("RGBA", (32, 32)).save(image_path)
        result["renders"][0].update(_ref(image_path))
    _json(path, result)
    outcome["result"] = _ref(path)
    _json(bundle["manifest"], manifest)
    _edit(bundle, "records", lambda v: v.update(manifest_sha256=_ref(bundle["manifest"])["sha256"]))
    with pytest.raises(ValueError):
        _import(bundle)


def test_empty_raw_response_cannot_support_a_scored_record(bundle):
    value = json.loads(bundle["records"].read_text())
    response = Path(value["records"][0]["raw_response"]["path"])
    response.write_bytes(b"  ")
    value["records"][0]["raw_response"] = _ref(response)
    _json(bundle["records"], value)
    with pytest.raises(ValueError, match="nonempty raw response"):
        _import(bundle)


def test_duplicate_json_keys_rejected(bundle):
    bundle["records"].write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _import(bundle)


def test_load_cannot_trust_edited_imported_scores(bundle):
    path = _import(bundle)
    value = json.loads(path.read_text())
    value["cases"][0]["scores"]["scene_coherence"] = 5
    _json(path, value)
    with pytest.raises(ValueError, match="differs from its verified inputs"):
        module.load_scores(path)


def test_final_read_detects_drift_during_validation(bundle, monkeypatch):
    original = module._recheck
    def drift(checked):
        bundle["records"].write_bytes(bundle["records"].read_bytes() + b" ")
        return original(checked)
    monkeypatch.setattr(module, "_recheck", drift)
    with pytest.raises(ValueError, match="changed during"):
        _import(bundle)
    assert not bundle["out"].exists()


def test_actual_module_cli_import_aggregate_and_no_overwrite(bundle, tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "metafind.eval.scene_scores"]
    imported = subprocess.run(command + ["import", "--manifest", str(bundle["manifest"]), "--protocol", str(bundle["protocol"]),
                                          "--records", str(bundle["records"]), "--out", str(bundle["out"])],
                              env=env, capture_output=True, text=True)
    assert imported.returncode == 0, imported.stderr
    summary_path = tmp_path / "summary.json"
    aggregated = subprocess.run(command + ["aggregate", "--scores", str(bundle["out"]), "--out", str(summary_path)],
                                env=env, capture_output=True, text=True)
    assert aggregated.returncode == 0, aggregated.stderr
    assert json.loads(summary_path.read_text())["methods"]["method-A"]["n_total"] == 4
    before = bundle["out"].read_bytes()
    with pytest.raises(FileExistsError): _import(bundle)
    assert bundle["out"].read_bytes() == before
